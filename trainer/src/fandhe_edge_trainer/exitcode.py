"""終了コード（REQ-21・REQ-33）の暫定ミラー。

単一真実源は Rust 側（`docs/spec/03-poc/core-cli-vertical-slice/core/src/exitcode.rs`
相当。本実装では TASK-15.x で確定する共通コア crate）。学習ワーカーは Rust の CLI から
子プロセスとして起動され、この 7 種のいずれかで終了する契約に従う。値がここと
Rust 側とで食い違わないよう、変更時は両方を同時に見直す。
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """7 種の終了コード（REQ-21）。値は sysexits(3) 由来の 64・70 を含め固定。"""

    OK = 0
    JUDGED_FAIL = 10
    OUT_OF_SCOPE = 11
    PENDING = 12
    LIMIT_EXCEEDED = 20
    INVALID_INPUT = 64
    RUNTIME_ERROR = 70
