"""PoC-26 学習・採点 CLI の入口（argparse・サブコマンドの配線・終了コード写像）。

REQ-41・TASK-41.1-5・#390。ローカルの Qwen2.5-0.5B-Instruct を LoRA で追加学習し、26 ラベルを
「選択肢の採点」で選んで評価器が読める `pred.jsonl` を書く PoC 用の使い捨てスクリプト群の入口で、
本番の学習ワーカー・推論ランタイム・CLI には入らない（REQ-32）。通信しない（REQ-38）。評価ロジック
（正解率・有意性）は持たない（評価器は TASK-24.1 の 1 つだけ。REQ-24〜27）。事前登録
`docs/design/poc26-preregistration.md` と追補 `docs/design/poc26-preregistration-addendum-1.md`
に従う。`python -m tools.poc26.lora_poc` が呼ぶ（`lora_poc.py` は薄い入口）。

サブコマンド: `train`（`train.py`）・`predict`（`predict.py`）・`probe`・`compare-probe`
（`probe.py`）・`export-onnx`・`verify-onnx`（`export_onnx.py`。ONNX 書き出し可否。#392）。
共通部品は `common.py`、入出力は `io_records.py`、資産は `assets.py`、採点は
`score.py`、ファイルを安全に読む処理は `safe_io.py`。

契約:

- 終了コード: 上限超過（専用の `LimitExceededError`）は 20、入力不正・引数エラーは 64、実行時
  エラーは 70、`compare-probe` の不一致は 10（REQ-21）。エラーは stderr に JSON 1 行
- `train`・`predict`・`probe` は #387 で記録した sha256（`--model-sha256`・`--config-sha256`・
  `--tokenizer-sha256`・`--tokenizer-config-sha256`。64 桁の小文字 16 進）を必須で受け、不一致は 64
- seed は必須で `random`・`numpy`（グローバルと Generator）・`mlx` へ設定する（REQ-26）。
  決定的な再現が要る確認は `--device cpu`（evaluation-contract）
- stderr・エラーメッセージにデータ本文・id を出さない（security.md）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
from tools.poc26 import qwen2_model
from tools.poc26.common import (
    MAX_SEQ_LENGTH,
    MAX_WALL_SECONDS_CAP,
    MIN_SEQ_LENGTH,
    SHA_RE,
    invalid,
)
from tools.poc26.export_onnx import cmd_export_onnx, cmd_verify_onnx
from tools.poc26.predict import cmd_predict
from tools.poc26.probe import cmd_compare_probe, cmd_probe
from tools.poc26.safe_io import LimitExceededError
from tools.poc26.train import cmd_train

from fandhe_edge_trainer import limits
from fandhe_edge_trainer.errors import WorkerError, truncate_for_message
from fandhe_edge_trainer.exitcode import ExitCode

# 引数の上限（巨大な反復・バッチ・系列長を学習前に止める）。メモリの事前見積もり
# （batch_size x max_seq_length x vocab・採点の K x 系列長）は `train.py`・`score.py` が持つ。
MAX_ITERS = 100_000
MAX_BATCH_SIZE = 64
MAX_LORA_LAYERS = 128  # config の num_hidden_layers の上限と同じ
# 壁時計の既定（秒）。学習は事前登録 4 節の 1 時間、採点は別枠で同じ 1 時間（上限は 24 時間）。
DEFAULT_MAX_WALL_SECONDS = limits.MAX_TRAIN_WALL_SECONDS
DEFAULT_MAX_SCORE_SECONDS = limits.MAX_TRAIN_WALL_SECONDS


def _sha_arg(value: str) -> str:
    """64 桁の小文字 16 進だけを受理する（期待 sha256 の引数）。"""
    if not SHA_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("must be 64 lowercase hex characters")
    return value


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        """引数エラーは引数名だけを出し、値（パス・本文を含みうる）は出さない（64）。"""
        m = re.match(r"(argument [^:]+|the following arguments are required: .*)", message)
        raise invalid(f"invalid arguments: {m.group(1) if m else 'see usage'}")


def _range(name: str, v: float, lo: float, hi: float) -> None:
    if not (lo <= v <= hi):
        raise invalid(f"{name} out of range: {lo}..{hi}")


def _build_parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="lora_poc", description=__doc__.splitlines()[0], allow_abbrev=False)
    sub = ap.add_subparsers(dest="command", required=True)

    def pins(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model-dir", type=Path, required=True)
        # #387 でベースモデルを取得した時に記録した sha256（64 桁の小文字 16 進。必須）
        for flag in ("model", "config", "tokenizer", "tokenizer-config"):
            p.add_argument(f"--{flag}-sha256", type=_sha_arg, required=True)

    def common(p: argparse.ArgumentParser) -> None:
        pins(p)
        p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
        p.add_argument("--dtype", choices=["bf16", "float32"], default="bf16")
        p.add_argument("--evidence", choices=["real_machine", "test_harness"], required=True)
        # 採点（validation・predict・probe の forward）の壁時計上限。到達は 20
        p.add_argument("--max-score-seconds", type=int, default=DEFAULT_MAX_SCORE_SECONDS)

    def scoring(p: argparse.ArgumentParser) -> None:
        p.add_argument("--definition", type=Path, required=True)
        p.add_argument("--out-dir", type=Path, required=True)
        p.add_argument("--max-seq-length", type=int, default=512)
        p.add_argument("--system-prompt-file", type=Path)

    t = sub.add_parser("train", allow_abbrev=False)
    common(t)
    scoring(t)
    t.add_argument("--train", type=Path, required=True)
    t.add_argument("--validation", type=Path, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--iters", type=int, required=True)
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--num-layers", type=int, default=8)
    t.add_argument("--rank", type=int, default=8)
    t.add_argument("--scale", type=float, default=20.0)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--max-wall-seconds", type=int, default=DEFAULT_MAX_WALL_SECONDS)
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", allow_abbrev=False)
    common(p)
    scoring(p)
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--adapter-sha256", type=_sha_arg, required=True)  # 学習時に記録した値
    p.add_argument("--input", type=Path, required=True)
    p.set_defaults(func=cmd_predict)

    # ONNX 書き出し可否の確認（#392）。MLX は常に CPU・float32 で動かす
    for name, func in (("export-onnx", cmd_export_onnx), ("verify-onnx", cmd_verify_onnx)):
        e = sub.add_parser(name, allow_abbrev=False)
        pins(e)
        e.add_argument("--adapter-dir", type=Path, required=True)
        e.add_argument("--adapter-sha256", type=_sha_arg, required=True)
        e.add_argument("--evidence", choices=["real_machine", "test_harness"], required=True)
        e.set_defaults(func=func)
        if name == "export-onnx":
            e.add_argument("--out-dir", type=Path, required=True)
        else:
            e.add_argument("--onnx-dir", type=Path, required=True)
            # ReferenceEvaluator の 1 件あたりの壁時計上限（到達は 20）
            e.add_argument("--max-score-seconds", type=int, default=DEFAULT_MAX_SCORE_SECONDS)

    pr = sub.add_parser("probe", allow_abbrev=False)
    common(pr)
    pr.add_argument("--out", type=Path, required=True)
    pr.set_defaults(func=cmd_probe)

    c = sub.add_parser("compare-probe", allow_abbrev=False)
    c.add_argument("ours", type=Path)
    c.add_argument("ref", type=Path)
    c.add_argument("--atol", type=float, default=1e-3)
    c.set_defaults(func=cmd_compare_probe)
    return ap


def _validate(a: argparse.Namespace) -> None:
    if hasattr(a, "max_seq_length"):
        _range("max-seq-length", a.max_seq_length, MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
    if a.command == "train":
        _range("seed", a.seed, limits.MIN_SEED, limits.MAX_SEED)
        _range("iters", a.iters, 1, MAX_ITERS)
        _range("lr", a.lr, 1e-12, 1.0)
        _range("batch-size", a.batch_size, 1, MAX_BATCH_SIZE)
        _range("rank", a.rank, 1, qwen2_model.MAX_LORA_RANK)
        _range("num-layers", a.num_layers, 1, MAX_LORA_LAYERS)
        _range("scale", a.scale, 1e-6, 1e4)
        _range("dropout", a.dropout, 0.0, 0.999)
        _range("max-wall-seconds", a.max_wall_seconds, 1, MAX_WALL_SECONDS_CAP)
    if hasattr(a, "max_score_seconds"):
        _range("max-score-seconds", a.max_score_seconds, 1, MAX_WALL_SECONDS_CAP)
    if a.command == "compare-probe":
        _range("atol", a.atol, 0.0, 1e6)


CACHE_LIMIT_BYTES = 512 * 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    """エントリポイント。`WorkerError` の終了コードで返し、エラーは stderr に JSON 1 行で出す。"""
    prev = mx.default_device()
    prev_cache_limit: int | None = None
    try:
        # MLX の buffer cache は CPU でほぼ再利用されず 1 ステップ約 3 GiB ずつ増え、
        # RSS 上限（8 GiB。REQ-39）を数ステップで超える（2026-10-07 実機で計測）。
        # cache を上限つきにして RSS が実際の使用量を表すようにする。計算内容は変わらない。
        prev_cache_limit = mx.set_cache_limit(CACHE_LIMIT_BYTES)
        a = _build_parser().parse_args(argv)
        _validate(a)
        if hasattr(a, "device"):
            mx.set_default_device(mx.cpu if a.device == "cpu" else mx.gpu)
        return a.func(a)
    except WorkerError as exc:
        print(json.dumps({"code": exc.code, "message": exc.message}), file=sys.stderr)
        return int(exc.exit_code)
    except LimitExceededError as exc:  # forward 中の上限超過（モデル側の検証）
        msg = json.dumps({"code": "limit_exceeded", "message": truncate_for_message(str(exc))})
        print(msg, file=sys.stderr)
        return int(ExitCode.LIMIT_EXCEEDED)
    except Exception as exc:  # 予期しない失敗。型名だけを出し、値（本文を含みうる）は出さない
        print(
            json.dumps({"code": "runtime_error", "message": f"unexpected {type(exc).__name__}"}),
            file=sys.stderr,
        )
        return int(ExitCode.RUNTIME_ERROR)
    finally:
        mx.set_default_device(prev)
        if prev_cache_limit is not None:
            mx.set_cache_limit(prev_cache_limit)
