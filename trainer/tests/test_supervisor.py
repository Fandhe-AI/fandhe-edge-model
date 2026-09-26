"""`supervisor.py`（P0-2: 壁時計・RSS のプロセス外強制打ち切り。P1-1: 標準出力の
並行読み出し）のテスト。

`monitor_child`・`_drain_stdout` を、実際の学習ワーカーではなくダミーの子プロセス
（`sys.executable -c "..."`）に対して直接呼ぶことで、監視ロジックだけを
高速・決定的に検証する（証拠種別: テストハーネス）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time as time_mod
from pathlib import Path

import pytest

from fandhe_edge_trainer import supervisor

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


def _spawn(code: str, *, stdout: int = subprocess.DEVNULL) -> subprocess.Popen:
    return subprocess.Popen(  # noqa: S603 - テスト専用。引数は固定・shell 不使用
        [sys.executable, "-c", code],
        stdout=stdout,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)


def test_monitor_child_kills_on_time_limit() -> None:
    proc = _spawn("import time; time.sleep(60)")
    try:
        t0 = time_mod.monotonic()
        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=0.2,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            poll_interval=0.05,
            grace_seconds=0.0,
        )
        elapsed = time_mod.monotonic() - t0
        assert reason == "time"
        assert elapsed < 5.0  # 60 秒スリープを待たずに打ち切られていること
        assert proc.poll() is not None  # 子プロセスが実際に終了している
    finally:
        _reap(proc)


def test_monitor_child_kills_on_rss_limit() -> None:
    # 約 200MiB を確保してからスリープする（RSS が確実に上限を超えるようにする）。
    proc = _spawn("b = bytearray(200 * 1024 * 1024); import time; time.sleep(60)")
    try:
        t0 = time_mod.monotonic()
        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=60.0,
            rss_limit_bytes=50 * 1024 * 1024,
            poll_interval=0.05,
            grace_seconds=0.0,
        )
        elapsed = time_mod.monotonic() - t0
        assert reason == "rss"
        assert elapsed < 10.0
        assert proc.poll() is not None
    finally:
        _reap(proc)


def test_monitor_child_returns_none_on_normal_completion() -> None:
    proc = _spawn("pass")
    reason = supervisor.monitor_child(
        proc,
        time_limit_seconds=30.0,
        rss_limit_bytes=64 * 1024 * 1024 * 1024,
        poll_interval=0.05,
        grace_seconds=0.0,
    )
    assert reason is None
    assert proc.wait(timeout=5) == 0


def test_monitor_child_fails_closed_when_ps_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ps` が使えない（監視できない）場合、野放しにせず子プロセスごと終了させる。"""
    monkeypatch.setattr(supervisor, "_PS_BIN", "/nonexistent/ps")
    proc = _spawn("import time; time.sleep(60)")
    try:
        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=60.0,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            poll_interval=0.05,
            grace_seconds=0.0,
        )
        assert reason == "monitor_failed"
        assert proc.poll() is not None
    finally:
        _reap(proc)


def test_parse_worker_stdout_accepts_single_json_object() -> None:
    payload = supervisor._parse_worker_stdout(b'{"status": "ok"}\n')
    assert payload == {"status": "ok"}


def test_parse_worker_stdout_rejects_multiple_lines() -> None:
    assert supervisor._parse_worker_stdout(b'{"a": 1}\n{"b": 2}\n') is None


def test_parse_worker_stdout_rejects_non_json() -> None:
    assert supervisor._parse_worker_stdout(b"not json at all\n") is None


# --------------------------------------------------------------------------
# P1-1: 標準出力の並行読み出し（パイプ詰まりによる誤判定の防止）
# --------------------------------------------------------------------------


def test_drain_stdout_prevents_pipe_block_during_slow_monitoring() -> None:
    """子プロセスが大量の標準出力（一般的な OS のパイプ容量 64KiB を明確に
    超える量）を書き込んでも、別スレッドで並行して読み進めていればブロックせず
    正常に完了できること（監視が誤って `limit_exceeded` と判定しないこと）。
    """
    size = 200_000  # 64KiB のパイプ容量を明確に超える
    proc = _spawn(
        f"import sys; sys.stdout.write('x' * {size}); sys.stdout.flush()",
        stdout=subprocess.PIPE,
    )
    stdout_result: dict = {}
    reader = threading.Thread(target=supervisor._drain_stdout, args=(proc.stdout, stdout_result))
    reader.start()
    try:
        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=10.0,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            poll_interval=0.05,
            grace_seconds=0.0,
        )
    finally:
        reader.join(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()
    assert reason is None  # パイプが詰まってブロックしたなら "time" になっていたはず
    assert stdout_result["oversized"] is False
    assert stdout_result["data"] == b"x" * size


def test_drain_stdout_caps_and_flags_oversized() -> None:
    size = supervisor._MAX_WORKER_STDOUT_BYTES + 1000
    proc = _spawn(
        f"import sys; sys.stdout.write('x' * {size}); sys.stdout.flush()",
        stdout=subprocess.PIPE,
    )
    result: dict = {}
    try:
        supervisor._drain_stdout(proc.stdout, result)
        proc.wait(timeout=10)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        _reap(proc)
    assert result["oversized"] is True
    assert len(result["data"]) == supervisor._MAX_WORKER_STDOUT_BYTES


def test_supervisor_module_does_not_import_mlx() -> None:
    """`supervisor.py` は監視・強制終了だけで完結する設計であり、mlx・onnx・numpy
    を import しない（P0-2）。サブプロセスで確認することで、テスト実行プロセス
    側の import 状態（他のテストが先に mlx を読み込んでいる等）に影響されない。
    """
    code = (
        f"import sys; sys.path.insert(0, {_SRC_DIR!r}); "
        "import fandhe_edge_trainer.supervisor as s; "
        "heavy = [m for m in ('mlx', 'onnx', 'numpy') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    result = subprocess.run(  # noqa: S603 - テスト専用。引数は固定・shell 不使用
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"heavy deps imported: {result.stdout.strip()}"


def test_run_supervised_train_rejects_invalid_request_without_spawning_worker(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({"schema_version": 1, "kind": "c3"}), encoding="utf-8")
    exit_code = supervisor.run_supervised_train(request_path)
    assert int(exit_code) == 64
