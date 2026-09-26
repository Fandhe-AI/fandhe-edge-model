"""`supervisor.py`（P0-2: 壁時計・RSS のプロセス外強制打ち切り）のテスト。

`monitor_child` を、実際の学習ワーカーではなくダミーの子プロセス（`sys.executable
-c "..."`）に対して直接呼ぶことで、監視ロジックだけを高速・決定的に検証する
（証拠種別: テストハーネス）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fandhe_edge_trainer import supervisor

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


def _spawn(code: str) -> subprocess.Popen:
    return subprocess.Popen(  # noqa: S603 - テスト専用。引数は固定・shell 不使用
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def test_monitor_child_kills_on_time_limit() -> None:
    proc = _spawn("import time; time.sleep(60)")
    try:
        import time as time_mod

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
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_monitor_child_kills_on_rss_limit() -> None:
    # 約 200MiB を確保してからスリープする（RSS が確実に上限を超えるようにする）。
    proc = _spawn("b = bytearray(200 * 1024 * 1024); import time; time.sleep(60)")
    try:
        import time as time_mod

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
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


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
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_parse_worker_stdout_accepts_single_json_object() -> None:
    payload = supervisor._parse_worker_stdout(b'{"status": "ok"}\n')
    assert payload == {"status": "ok"}


def test_parse_worker_stdout_rejects_multiple_lines() -> None:
    assert supervisor._parse_worker_stdout(b'{"a": 1}\n{"b": 2}\n') is None


def test_parse_worker_stdout_rejects_non_json() -> None:
    assert supervisor._parse_worker_stdout(b"not json at all\n") is None


def test_parse_worker_stdout_rejects_oversized() -> None:
    huge = b'{"a": "' + b"x" * (supervisor._MAX_WORKER_STDOUT_BYTES + 16) + b'"}'
    assert supervisor._parse_worker_stdout(huge) is None


def test_cleanup_orphaned_output_removes_only_reservation_artifacts(tmp_path: Path) -> None:
    """予約済み一時ディレクトリ（`.out.tmp-*`）とその中身は消すが、無関係な
    ファイル・出力先ディレクトリの外にあるものには一切触れないこと。
    """
    from fandhe_edge_trainer import contract, guard

    root_handle = guard.resolve_root(str(tmp_path))
    entry = guard.confine(root_handle, "out", "out_dir")
    reservation = contract.prepare_out_dir(entry)
    fd = os.open(
        "partial.onnx", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=reservation.tmp_fd
    )
    with os.fdopen(fd, "wb") as f:
        f.write(b"partial data")
    reservation.close_tmp_fd()
    entry.close()
    root_handle.close()

    # 無関係なファイル（out_dir の外）は触れられないことを確認する対照群。
    (tmp_path / "unrelated.txt").write_text("keep me", encoding="utf-8")

    supervisor._cleanup_orphaned_output(str(tmp_path), "out")

    assert not (tmp_path / "out").exists()  # 空だった予約済み out_dir は消える
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]
    assert (tmp_path / "unrelated.txt").read_text(encoding="utf-8") == "keep me"


def test_cleanup_orphaned_output_leaves_unrelated_out_dir_untouched(tmp_path: Path) -> None:
    """out_dir が（別の何かによって）予約と無関係な内容を持つ場合、削除しない。"""
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "already_here.txt").write_text("do not delete", encoding="utf-8")

    supervisor._cleanup_orphaned_output(str(tmp_path), "out")

    assert (tmp_path / "out" / "already_here.txt").read_text(encoding="utf-8") == "do not delete"


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
