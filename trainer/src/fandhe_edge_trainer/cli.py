"""学習ワーカーの CLI 入口。

Rust 側 CLI（`docs/spec/03-poc/core-cli-vertical-slice/core/src/subprocess.rs`
`run_logged` 相当。本実装では TASK-34.x のジョブ管理から呼ばれる）が子プロセスとして
`python -m fandhe_edge_trainer train --request <path>` を起動する契約。

**`train`（公開サブコマンド）はスーパーバイザー、実体は内部サブコマンド `_worker`**
（P0-2）: `onnx.checker.check_model`・`SerializeToString`・単一の MLX 演算のような
「1 回の呼び出しが長時間かかりうる同期呼び出し」の最中は、ワーカー内の協調的な
資源チェック（`budget.py::ResourceBudget.check`）が働かない（呼び出しが返るまで
Python コードへ制御が戻らないため）。そのため、壁時計・RSS の強制打ち切りは
プロセス境界の外側（`supervisor.py`。mlx・onnx・numpy を import しない）から行う。
`_worker` は公開契約ではない（`--help` 自体を提供しないため文書化もしない）。

**`_worker` はリクエスト JSON を標準入力から受け取る**（P1: ファイルパスを
`--request <path>` で渡すと、スーパーバイザーが検証してから `_worker` が
再読込するまでの間にファイルが書き換えられうる。スーパーバイザーは検証済みの
バイト列を固定して子プロセスの標準入力へ渡す。`supervisor.py` モジュール
docstring 参照）。`_worker` は `--out-fd <n>` だけを引数に取る。

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
import stat
import sys
from pathlib import Path

from . import artifact as artifact_mod
from . import budget as budget_mod
from . import contract
from . import supervisor as supervisor_mod
from .errors import WorkerError
from .exitcode import ExitCode
from .kinds import resolve_kind
from .limits import MAX_REQUEST_BYTES


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


def _apply_rlimit_cpu_backstop(time_limit_seconds: int) -> None:
    """CPU 時間の kernel レベルの上限（`RLIMIT_CPU`）を設定する（P0-2）。

    壁時計・RSS の強制打ち切りは `supervisor.py` がプロセス境界の外側から
    行うが、CPU 律速の長い同期呼び出し（`onnx.checker.check_model` 等）は
    ワーカー自身にも保険を掛ける。ソフト上限を `time_limit_seconds`、ハード
    上限をその 5 秒後に設定する: ソフト上限到達時に `SIGXCPU`（既定の動作は
    プロセスの終了）が送られ、キャッチしなければ即座に終了する。ハード上限は
    「シグナルを無視・キャッチしても最終的に強制終了させる」ための保険。

    macOS では `RLIMIT_AS`・RSS 系の上限は事実上強制されない（`setrlimit(2)`
    の制約）ため、RSS の実効的な検査は `supervisor.py` の `ps` ポーリングに
    依存する。本関数は CPU 時間だけを対象にする。
    """
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (time_limit_seconds, time_limit_seconds + 5))
    except (ValueError, OSError, ImportError):
        # 環境によっては変更できない場合がある（権限・既存の上限等）。
        # 壁時計監視（supervisor.py）が最終防波堤になるため、ここでの失敗で
        # ワーカーの起動自体は妨げない（フェイルクローズにはしない）。
        pass


def _read_request_from_stdin() -> bytes:
    """標準入力からリクエストの生バイト列を読む（P1。`_worker` 専用）。

    スーパーバイザー（`supervisor.py::_spawn_worker_and_finalize`）は検証済みの
    バイト列を無名一時ファイルへ書いて `stdin=` として渡す（通常ファイルなので
    `read()` は EOF で確実に返る）。サイズ上限（`limits.MAX_REQUEST_BYTES`）は
    読み取り量そのもので縛る（`+1` バイトまで読み、超過を検出できるようにする。
    `contract.parse_request_bytes` 側でも同じ上限を再検査する）。

    **標準入力が通常ファイルであることを検査する**（セキュリティ監査指摘・
    REQ-39）: `_worker` は本来スーパーバイザーの子プロセスとしてのみ起動される
    契約だが、`_worker` は「公開契約ではない」内部サブコマンドであるため、
    仮に何者かがパイプ・端末（TTY）を標準入力に接続して直接起動した場合、
    書き手が現れない・EOF が来ない標準入力に対する `read()` は無期限に
    ブロックしうる（無制限待ちを作らない。REQ-39）。`os.fstat` で
    `S_ISREG` を確認し、通常ファイルでなければ読み取りを試みずに
    `invalid_request`（exit 64）で拒否する（fail-closed）。
    """
    stdin = sys.stdin.buffer if sys.stdin is not None else None
    if stdin is None:
        raise WorkerError(
            "invalid_request", "stdin is not available for _worker", ExitCode.INVALID_INPUT
        )
    try:
        st = os.fstat(stdin.fileno())
    except OSError as e:
        raise WorkerError(
            "invalid_request",
            f"stdin not stat-able: {type(e).__name__}",
            ExitCode.INVALID_INPUT,
        ) from e
    if not stat.S_ISREG(st.st_mode):
        raise WorkerError("invalid_request", "stdin must be a regular file", ExitCode.INVALID_INPUT)
    try:
        return stdin.read(MAX_REQUEST_BYTES + 1)
    except OSError as e:
        raise WorkerError(
            "invalid_request",
            f"failed to read request from stdin: {type(e).__name__}",
            ExitCode.INVALID_INPUT,
        ) from e


def run_worker_train(out_fd: int) -> ExitCode:
    """`_worker` サブコマンドの本体（実際の学習・書き出し）。戻り値は終了コード。

    **`out_dir` の名前・予約・確定・後始末には一切関与しない**（P0-1・P0-2 の
    見直し。`contract.py` モジュール docstring・`OutDirReservation` のクラス
    docstring 参照）。それらはすべてスーパーバイザー（`supervisor.py`）が担い、
    本関数には `--out-fd` で「学習・書き出しの成果物を書き込んでよい、既に
    予約済みの一時ディレクトリ」の fd 番号だけが渡される。`artifact.json`・
    `model.onnx` はその fd へ `dir_fd` 相対（`O_CREAT|O_EXCL|O_NOFOLLOW`）で
    新規作成するだけで、`out_dir` の名前を組み立てる経路は本関数のどこにも
    無い。

    **リクエストはファイルパスではなく標準入力から受け取る**（P1: スーパーバイザーが
    検証してから本関数が独自にファイルを再読込すると、検証後にファイルが
    書き換えられた場合、両者が異なる内容を見てしまう。スーパーバイザーが
    検証済みのバイト列を固定して渡す。`_read_request_from_stdin` 参照）。

    `request` が保持する fd（`root`・`train_path`・`out_dir`。
    `contract.py`・`guard.py` 参照）は、成功・失敗いずれの経路でも
    `finally` で必ず閉じる（`TrainRequest.close_resources()`。`out_dir` 側の
    fd はここで確認のためだけに開いたものであり、スーパーバイザー側が別途
    保持している実体とは独立した fd なので、ここで閉じてよい）。

    `resource_budget`（`budget.py::ResourceBudget`）はリクエストの検証直後に
    1 つだけ生成し、学習データの読み込み（`contract.load_train_examples`）・
    学習（`kind_impl.train`）・ONNX 書き出し（`kind_impl.export_onnx`）の
    すべてで同じインスタンスを使い回す（P0-1: 64 MiB・20 万件までの学習データを
    読み込む処理自体も、学習ジョブ全体の資源上限の対象にする）。
    """
    raw = _read_request_from_stdin()
    parsed = contract.parse_request_bytes(raw)
    request = contract.validate_request(parsed)
    try:
        _apply_rlimit_cpu_backstop(request.time_limit_seconds)
        resource_budget = budget_mod.ResourceBudget(
            wall_seconds=float(request.time_limit_seconds),
            rss_bytes=request.rss_limit_bytes,
            device=request.device,
        )
        examples = contract.load_train_examples(
            request.train_path, request.label_order, resource_budget=resource_budget
        )
        kind_impl = resolve_kind(request.kind, request.kind_version)

        trained = kind_impl.train(examples, request, resource_budget)
        # ONNX 本体・artifact.json は、スーパーバイザーが渡した一時ディレクトリの
        # fd（`out_fd`）へ、経路文字列を使わず dir_fd 相対で新規作成する
        # （TOCTOU 対策。O_EXCL で上書きしない）。
        onnx_fd = os.open(
            artifact_mod.ONNX_FILE_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=out_fd,
        )
        with os.fdopen(onnx_fd, "wb") as onnx_file:
            # HashingWriter: model.onnx へ書き込んだ厳密なバイト列の SHA-256 を、
            # export_onnx 側を変更せずに計算する（P0: artifact.py モジュール
            # docstring・AGENTS.md ガード層「完全性と版」参照）。
            hashing_writer = artifact_mod.HashingWriter(onnx_file)
            kind_impl.export_onnx(trained, hashing_writer)
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
            onnx_sha256=hashing_writer.hexdigest(),
        )
        artifact_mod.write_artifact(out_fd, art)
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
    # `_worker`: 公開契約ではない内部サブコマンド（P0-2 のモジュール docstring参照）。
    # `train`（supervisor.py）が子プロセスとして起動する実体。`--out-fd` は
    # スーパーバイザーが `pass_fds` で引き継いだ、出力用一時ディレクトリの fd 番号
    # （`contract.py::OutDirReservation` 参照。`_worker` はこの fd 番号以外の
    # 経路で `out_dir` を扱わない）。リクエストの内容は `--request <path>` では
    # 受け取らず標準入力から読む（P1: `_read_request_from_stdin` 参照）。
    p_worker = sub.add_parser("_worker", add_help=False, exit_on_error=False)
    p_worker.add_argument("--out-fd", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "train":
            exit_code = supervisor_mod.run_supervised_train(Path(args.request))
        elif args.command == "_worker":
            exit_code = run_worker_train(args.out_fd)
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
