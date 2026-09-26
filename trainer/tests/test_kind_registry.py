"""モデルの種類の選択口のテスト（TASK-19.1・TASK-19.3）。"""

from __future__ import annotations

import pytest

from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode
from fandhe_edge_trainer.kinds import resolve_kind
from fandhe_edge_trainer.kinds.c3 import C3Kind


def test_resolve_kind_returns_c3_implementation() -> None:
    kind = resolve_kind("c3", 1)
    assert isinstance(kind, C3Kind)


def test_resolve_kind_rejects_unsupported_kind() -> None:
    with pytest.raises(WorkerError) as exc_info:
        resolve_kind("no-such-kind", 1)
    assert exc_info.value.code == "unsupported_kind"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_resolve_kind_rejects_unsupported_kind_version() -> None:
    with pytest.raises(WorkerError) as exc_info:
        resolve_kind("c3", 999)
    assert exc_info.value.code == "unsupported_kind_version"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
