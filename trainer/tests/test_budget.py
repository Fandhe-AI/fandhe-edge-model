"""ジョブ全体の資源上限（REQ-39: 学習時間・メモリ）のテスト（P0-B）。"""

from __future__ import annotations

import pytest

from fandhe_edge_trainer import budget
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode


def test_check_sample_steps_rejects_before_training() -> None:
    with pytest.raises(WorkerError) as exc_info:
        budget.check_sample_steps(n_examples=1_000_000, epochs=100)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_check_sample_steps_accepts_within_limit() -> None:
    budget.check_sample_steps(n_examples=10, epochs=5)  # 例外を送出しないことを確認する


def test_resource_budget_rejects_on_wall_clock_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """`time.monotonic` を進めることで、極小の `wall_seconds` でも締切り超過を検出できること。"""
    fake_time = [1000.0]
    monkeypatch.setattr(budget.time, "monotonic", lambda: fake_time[0])
    rb = budget.ResourceBudget(wall_seconds=10.0, rss_bytes=64 * 1024 * 1024 * 1024, device="cpu")
    fake_time[0] += 20.0  # 締切りを過ぎる
    with pytest.raises(WorkerError) as exc_info:
        rb.check()
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_resource_budget_rejects_on_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    """極小の `rss_bytes` を指定すると、実プロセスの RSS が必ず上限を超えること。"""
    rb = budget.ResourceBudget(wall_seconds=3600.0, rss_bytes=1, device="cpu")
    with pytest.raises(WorkerError) as exc_info:
        rb.check()
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_resource_budget_accepts_within_limits() -> None:
    rb = budget.ResourceBudget(wall_seconds=3600.0, rss_bytes=64 * 1024 * 1024 * 1024, device="cpu")
    rb.check()  # 例外を送出しないことを確認する
