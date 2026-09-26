"""学習ワーカーの CLI 入口。

Rust 側 CLI（`docs/spec/03-poc/core-cli-vertical-slice/core/src/subprocess.rs`
`run_logged` 相当。本実装では TASK-34.x のジョブ管理から呼ばれる）が子プロセスとして
`python -m fandhe_edge_trainer train --request <path>` を起動する契約。

- 標準出力: 成功・失敗いずれも JSON 1 つだけ（進捗・ログは標準エラーへ出す）
- 終了コード: 成功 0（ExitCode.OK）、`WorkerError` はその `exit_code`、
  想定外の例外は `runtime_error` として 70。**引数解析の失敗を含め、
  argparse 既定の exit 2・非 JSON 出力を一切許さない**（呼び出し元の Rust 側が
  終了コードを 7 種の契約〔REQ-21〕としてしか解釈しないため。以下 `_NoExitArgumentParser`
  ・`exit_on_error=False`・`add_help=False` の 3 点で argparse の既定の
  `sys.exit`/usage 出力経路を塞ぐ）
- ネットワークへは一切接続しない（REQ-38）。乱数は seed を明示して設定する
  （kind 実装〔`kinds/c3.py`〕側で行う）
- `KeyboardInterrupt`・`SystemExit`（本ファイルの外から意図的に投げられた場合）は
  意図的に捕捉しない。OS シグナル・明示的終了は本ワーカーの 7 種の終了コード契約の
  対象外とし、Python の既定動作（130 等）のまま呼び出し元へ伝える
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import artifact as artifact_mod
from . import contract
from .errors import WorkerError
from .exitcode import ExitCode
from .kinds import resolve_kind


class _NoExitArgumentParser(argparse.ArgumentParser):
    """`error()` で `sys.exit(2)` せず `WorkerError` を送出する argparse。

    既定の `ArgumentParser.error()` は usage を stderr へ出し `sys.exit(2)` する
    （非 JSON 出力・非 7 種終了コード）。`add_subparsers()` は既定で
    `parser_class=type(self)` を使うため、本クラスから作る限りサブパーサーも
    同じ挙動になる。
    """

    def error(self, message: str) -> None:
        raise WorkerError("invalid_request", f"argument error: {message}", ExitCode.INVALID_INPUT)


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def run_train(request_path: Path) -> ExitCode:
    """train サブコマンドの本体。戻り値は終了コード。

    `request` が保持する fd（`root`・`train_path`・`out_dir`。
    `contract.py`・`guard.py` 参照）は、成功・失敗いずれの経路でも
    `finally` で必ず閉じる（`TrainRequest.close_resources()`）。
    """
    request = contract.load_request(request_path)
    try:
        examples = contract.load_train_examples(request.train_path, request.label_order)
        kind_impl = resolve_kind(request.kind, request.kind_version)

        reservation = contract.prepare_out_dir(request.out_dir)
        try:
            trained = kind_impl.train(examples, request)
            # ONNX 本体・artifact.json は、予約済み out_dir と同じ親ディレクトリ
            # 配下の作業用一時ディレクトリ（`reservation.tmp_fd`）へ、経路文字列を
            # 使わず dir_fd 相対で新規作成する（TOCTOU 対策。O_EXCL で上書きしない）。
            onnx_fd = os.open(
                artifact_mod.ONNX_FILE_NAME,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=reservation.tmp_fd,
            )
            with os.fdopen(onnx_fd, "wb") as onnx_file:
                kind_impl.export_onnx(trained, onnx_file)
            art = artifact_mod.build_artifact(
                kind=request.kind,
                kind_version=request.kind_version,
                config=trained.config,
                label_order=trained.label_order,
                # "choice"（選択肢からの判定）は Rust 側 Artifact.output_type
                # （artifact.rs）・selector_common.py の output.type 契約と同じ値。
                # 生成型の種類（REQ-19b）を追加する際はここが "choice" 以外の値を
                # 取りうるようになる（その値の集合・意味は共通コア〔REQ-15〕側で
                # 定義されるべきで、本ワーカーは種類ごとに固定値を渡すだけに留める）。
                output_type="choice",
                max_bytes=trained.max_bytes,
                candidate_label=request.kind,
            )
            artifact_mod.write_artifact(reservation.tmp_fd, art)
        except BaseException:
            # 一時ディレクトリ（自分の作業物）・予約済み out_dir（「まだ自分の
            # 予約かつ空」の場合のみ）の両方を解放する（contract.cleanup_reservation
            # のドキュメント参照。学習中に何者かが out_dir へ書き込んでいたら残す）。
            contract.cleanup_reservation(reservation)
            raise
        contract.finalize_out_dir(reservation)
    finally:
        request.close_resources()

    _emit({"status": "ok", "artifact_dir": str(request.out_dir.display), "artifact": art})
    return ExitCode.OK


def _build_parser() -> argparse.ArgumentParser:
    # add_help=False: 既定の -h/--help は argparse 内部で直接 sys.exit(0) するため、
    # error() の override だけでは塞げない（help 自体を提供しない）。
    # exit_on_error=False: 一部のエラー（型変換・choices 不正）は error() を経由せず
    # ArgumentError を送出する経路になるため、main() 側で追加捕捉する。
    parser = _NoExitArgumentParser(prog="fandhe_edge_trainer", add_help=False, exit_on_error=False)
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train", add_help=False, exit_on_error=False)
    p_train.add_argument("--request", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "train":
            exit_code = run_train(Path(args.request))
        else:  # pragma: no cover - argparse の choices で到達しない
            raise WorkerError(
                "invalid_request", f"unknown command: {args.command}", ExitCode.INVALID_INPUT
            )
    except WorkerError as e:
        _emit({"status": "error", "code": e.code, "message": e.message})
        return int(e.exit_code)
    except argparse.ArgumentError as e:
        # exit_on_error=False 時に error() を経由せず送出される経路（型変換・choices 不正）。
        _emit({"status": "error", "code": "invalid_request", "message": f"argument error: {e}"})
        return int(ExitCode.INVALID_INPUT)
    except Exception as e:
        # データ本文を含みうる例外メッセージ（str(e)）を stdout へ出さない（security.md）。
        # 例外の型名だけを stderr へ出す（traceback は出さない。ログが子プロセスの
        # stderr ログファイルへ捕獲される契約〔subprocess.rs〕上、データ本文の
        # 混入経路を最小化するため）。
        print(f"{type(e).__name__}", file=sys.stderr)
        _emit(
            {
                "status": "error",
                "code": "runtime_error",
                "message": f"unexpected error: {type(e).__name__}",
            }
        )
        return int(ExitCode.RUNTIME_ERROR)
    return int(exit_code)
