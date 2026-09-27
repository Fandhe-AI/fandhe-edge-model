"""終了コード（REQ-21・REQ-33）のミラー。

単一真実源は Rust 側の `crates/core/src/exitcode.rs`（`fandhe-edge-core`。
TASK-21.1-1）。学習ワーカーは Rust の CLI から子プロセスとして起動され、
この 7 種のいずれかで終了する契約に従う。値がここと Rust 側とで食い違わ
ないことは、両者が共有する fixture `fixtures/exitcode/exit_codes.json` を
介して `crates/core/tests/exitcode_fixture.rs`（Rust）と
`trainer/tests/test_exitcode.py`（pytest）が機械照合する（TASK-21.1・
#179）。変更時は Rust 側・fixture・本ファイルの 3 箇所を同時に見直す。
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
