"""PoC-26 学習・採点 CLI の共通部品（REQ-41・TASK-41.1-5・#390）。

役割: 終了コードへの写像（`WorkerError`・64 / 20 / 70）、上限つきの入力読み込み、資源予算
（`Budget`）、seed、stderr への進捗を提供する。`cli.py` から配線される各サブコマンド
（`train.py`・`predict.py`・`probe.py`）が使う。ファイルを安全に読む実体は `safe_io.py`。
データ本文・id はログ・メッセージに出さない（security.md）。
"""

from __future__ import annotations

import random
import re
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from tools.poc26.safe_io import LimitExceededError, loads_json, read_limited, sha256_hex

from fandhe_edge_trainer import limits
from fandhe_edge_trainer.errors import WorkerError, truncate_for_message
from fandhe_edge_trainer.exitcode import ExitCode

# コマンド全体の壁時計の天井（`--max-wall-seconds` の上限でもある。24 時間）。
MAX_WALL_SECONDS_CAP = 86_400
MIN_SEQ_LENGTH, MAX_SEQ_LENGTH = 8, 4096
SHA_RE = re.compile(r"[0-9a-f]{64}")


def err(code: str, message: str, exit_code: ExitCode = ExitCode.INVALID_INPUT) -> WorkerError:
    return WorkerError(code, message, exit_code)


def invalid(message: str) -> WorkerError:
    return err("invalid_input", message)


def too_large(message: str) -> WorkerError:
    return err("limit_exceeded", message, ExitCode.LIMIT_EXCEEDED)


@contextmanager
def as_input_error():
    """tokenizer・モデル読み込み・LoRA 適用の例外（入力値なし）を終了コードへ写す。

    上限超過は専用の `LimitExceededError`（-> 20）、それ以外の `ValueError` は 64。文言では
    振り分けない。
    """
    try:
        yield
    except LimitExceededError as exc:
        raise too_large(truncate_for_message(str(exc))) from None
    except ValueError as exc:
        raise invalid(truncate_for_message(str(exc))) from None


def sha256(data: bytes) -> str:
    """バイト列の sha256（`safe_io.sha256_hex`。CLI の各モジュールが共通で使う）。"""
    return sha256_hex(data)


def read_input(path: str | Path, limit: int, what: str) -> bytes:
    """symlink を辿らず通常ファイルだけを fd 経由で読み、上限超過は停止する（REQ-39）。

    実体は `safe_io.read_limited`。上限超過は終了コード 20、開けない・通常ファイルでない場合は 64。
    脅威モデルの前提（パス途中の symlink は辿る）は `safe_io` の docstring を参照（P3）。
    """
    try:
        return read_limited(path, limit, what)
    except LimitExceededError as exc:
        raise too_large(str(exc)) from None
    except ValueError as exc:
        raise invalid(str(exc)) from None


def loads(raw: bytes, what: str) -> Any:
    """NaN・Infinity を拒否して JSON を読む（不正は終了コード 64）。"""
    try:
        return loads_json(raw, what)
    except ValueError as exc:
        raise invalid(str(exc)) from None


def max_rss_bytes() -> int:
    """プロセスの最大 RSS と MLX の確保量のピークの大きい方を byte で返す。

    `ru_maxrss` は macOS では byte、Linux では KiB で返るため揃える。

    `ru_maxrss` は CPU 側の常駐量で、Metal（unified memory）の GPU バッファを
    反映しきれないことがあるため、`mx.get_peak_memory()`（MLX の確保量のピーク。mlx 0.32.2 に存在）
    と併用して大きい方で判定する。いずれも**ピーク**（単調増加）で、確認はステップ・レコードの境界
    だけ（1 回の forward / backward の途中では止められない）。
    """
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = v if sys.platform == "darwin" else v * 1024
    return max(rss, int(mx.get_peak_memory()))


class Budget:
    """1 コマンド（または 1 フェーズ）の資源上限（REQ-39）。

    - `check()`: RSS / MLX ピーク超過（8 GiB）、コマンド全体の壁時計の天井（24 時間）、`wall_limit`
      （指定時。採点フェーズの `--max-score-seconds`）の超過は停止する（`LimitExceededError`
      相当の `WorkerError` -> 20）。
    - `wall_exceeded(limit)`: 学習ループだけが使う「予算到達」の判定（事前登録 4 節: 到達は記録して
      その時点の最良で評価する。停止しない）。時間は生成時点からの経過。

    モデル読み込みは 1 回の処理（途中で止められない）なので、フェーズの壁時計上限の対象外とする
    （24 時間の天井と RSS だけを前後で確認する）。
    """

    def __init__(self, wall_limit: float | None = None) -> None:
        self.start = time.monotonic()
        self.wall_limit = wall_limit

    def check(self) -> None:
        if self.elapsed() > MAX_WALL_SECONDS_CAP:
            raise too_large("wall-clock ceiling exceeded")
        if self.wall_limit is not None and self.elapsed() > self.wall_limit:
            raise too_large("scoring wall-clock budget exceeded")
        if max_rss_bytes() > limits.MAX_TRAIN_RSS_BYTES:
            raise too_large("RSS budget exceeded")

    def wall_exceeded(self, limit: float) -> bool:
        return self.elapsed() > limit

    def elapsed(self) -> float:
        return time.monotonic() - self.start


def log(msg: str) -> None:
    """stderr への進捗。数値・固定語だけを出す（データ本文・id は出さない）。"""
    print(msg, file=sys.stderr, flush=True)


def seed_all(seed: int) -> None:
    """Python の `random`・NumPy のグローバル乱数・MLX を seed する（REQ-26）。

    バッチ順は別に `np.random.default_rng(seed)` の Generator を使う（グローバル状態に依存しない）。
    """
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def to_dtype(name: str) -> mx.Dtype:
    return mx.bfloat16 if name == "bf16" else mx.float32
