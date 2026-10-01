"""`procinfo_darwin`（macOS の libproc による子プロセス状態取得）のテスト。

REQ-38・REQ-39・#327: setuid の `/bin/ps` が `sandbox-exec` 下で起動できず学習が
`runtime_error` になった問題の回帰を、実 `sandbox-exec` 下で固定する
（証拠種別: 実機。macOS 専用 API のため darwin 以外は対象外）。
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from fandhe_edge_trainer import procinfo_darwin

# libproc は macOS 専用 API のため、実機を使うテストは darwin 以外では対象外にする。
# 非 darwin の fail-closed を確かめる 1 件だけは、この印を付けず Linux でも実行する。
darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="libproc は macOS 専用 API")

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_PROFILE = "(version 1)(allow default)(deny network*)"
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


def _spawn_sleeper() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_zombie(pid: int) -> None:
    """回収せずに子が終了するのを待つ（`child_status` が回収しないことの確認に使う）。"""
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        status = procinfo_darwin.child_status(pid)
        if status is not None and status[1]:
            return
        time.sleep(0.02)
    raise AssertionError("child did not become a zombie")


def test_req39_non_darwin_platform_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-39・#327: darwin 以外では libproc を呼ばず `None`（fail-closed）。Linux でも実行する。"""
    monkeypatch.setattr(procinfo_darwin.sys, "platform", "linux")
    assert procinfo_darwin.child_status(os.getpid()) is None


def _ps_rss_bytes(pid: int) -> int:
    out = subprocess.run(  # noqa: S603 - テスト専用。引数は固定・shell 不使用
        ["/bin/ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=True
    ).stdout
    return int(out.strip()) * 1024


@darwin_only
def test_req39_rss_of_live_child_matches_ps() -> None:
    """REQ-39・#327: 生存中の子の RSS が、前後の `ps -o rss=`（KiB）×1024 の範囲に収まる。

    RSS は読む間にも変動しうるため、ps → `child_status` → ps の順に読み、前後の ps 値の
    min〜max を許容範囲にする（厳密一致は flaky）。"""
    proc = _spawn_sleeper()
    try:
        time.sleep(0.3)  # 起動直後の RSS 変動を避ける
        before = _ps_rss_bytes(proc.pid)
        got = procinfo_darwin.child_status(proc.pid)
        after = _ps_rss_bytes(proc.pid)
        assert got is not None
        rss, is_zombie = got
        assert is_zombie is False
        assert rss > 0
        assert min(before, after) <= rss <= max(before, after)
    finally:
        proc.kill()
        proc.wait(timeout=5)


@darwin_only
def test_req39_unreaped_zombie_is_detected_without_reaping() -> None:
    """REQ-39・#327: 終了済み未回収の子は `(0, True)`。検知後も `waitpid` で回収できる。"""
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    _wait_zombie(proc.pid)
    assert procinfo_darwin.child_status(proc.pid) == (0, True)
    # 回収されていない（child_status は reap しない）ので waitpid が成功する。
    reaped_pid, wait_status = os.waitpid(proc.pid, 0)
    assert reaped_pid == proc.pid
    assert os.waitstatus_to_exitcode(wait_status) == 0
    proc.returncode = 0  # Popen の二重回収を避ける


@darwin_only
def test_req39_reaped_and_nonexistent_pid_are_unknown() -> None:
    """REQ-39・#327: 回収後・存在しない pid・不正な pid は `None`（fail-closed）。"""
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    proc.wait(timeout=10)
    assert procinfo_darwin.child_status(proc.pid) is None
    assert procinfo_darwin.child_status(2**30) is None
    assert procinfo_darwin.child_status(0) is None
    assert procinfo_darwin.child_status(0x7FFFFFFF + 1) is None
    assert procinfo_darwin.child_status(-1) is None
    assert procinfo_darwin.child_status(True) is None  # type: ignore[arg-type]
    assert procinfo_darwin.child_status("1") is None  # type: ignore[arg-type]


@darwin_only
def test_req39_library_load_failure_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-39・#327: libproc を読み込めなければ `None`（例外を外へ出さない）。"""
    monkeypatch.setattr(procinfo_darwin, "_LIBPROC_PATH", "/nonexistent/libproc.dylib")
    monkeypatch.setattr(procinfo_darwin, "_loaded", False)
    monkeypatch.setattr(procinfo_darwin, "_proc_pidinfo", None)
    proc = _spawn_sleeper()
    try:
        assert procinfo_darwin.child_status(proc.pid) is None
    finally:
        proc.kill()
        proc.wait(timeout=5)


@darwin_only
def test_req38_req39_child_status_works_under_real_sandbox_exec() -> None:
    """REQ-38・REQ-39・#327 回帰: 実 `sandbox-exec` 下で `/bin/ps` が使えなくても、
    `child_status` は子に対して `(rss>0, False)` を返す。sandbox-exec が無ければ
    skip せず失敗させる（CI の python-ci は macos-14）。"""
    assert os.path.exists(_SANDBOX_EXEC), "sandbox-exec is required on macOS"
    script = textwrap.dedent(
        """
        import subprocess, sys, time
        sys.path.insert(0, sys.argv[1])
        from fandhe_edge_trainer import procinfo_darwin
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.3)
            ps_ok = True
            try:
                subprocess.run(["/bin/ps", "-p", str(child.pid)], capture_output=True, check=True)
            except (OSError, subprocess.CalledProcessError):
                ps_ok = False
            status = procinfo_darwin.child_status(child.pid)
            ok = status is not None and status[0] > 0 and status[1] is False
            print("ps_ok=%s status_ok=%s" % (ps_ok, ok))
        finally:
            child.kill()
            child.wait()
        """
    )
    result = subprocess.run(  # noqa: S603 - テスト専用。引数は固定・shell 不使用
        [_SANDBOX_EXEC, "-p", _PROFILE, sys.executable, "-c", script, _SRC_DIR],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    # `ps_ok` は情報としてだけ出力する（`ps` が使えるかは環境依存のためアサートしない）。
    assert "status_ok=True" in result.stdout.split()


@darwin_only
def test_req39_exiting_child_is_never_unknown_before_zombie_327() -> None:
    """REQ-39・#327 回帰: 短命な子が終了処理中（SRUN のまま INEXIT・TASKINFO 失敗）でも、
    ゾンビになるまで `child_status` が `None`（監視不能）を返さない。最後はゾンビとして
    検知でき、`waitpid` で回収できる。`(0, False)`（INEXIT 経路）の観測回数は実機では
    毎回あるが、タイミング依存のため assert せず、修正前実装での失敗確認で代える。"""
    for _ in range(40):
        proc = subprocess.Popen(
            [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            deadline = time.monotonic() + 10.0
            while True:
                status = procinfo_darwin.child_status(proc.pid)
                assert status is not None, "child_status returned None while child was exiting"
                if status[1]:
                    break
                assert time.monotonic() < deadline, "child did not become a zombie"
            reaped_pid, wait_status = os.waitpid(proc.pid, 0)
            assert reaped_pid == proc.pid
            assert os.waitstatus_to_exitcode(wait_status) == 0
            proc.returncode = 0
        finally:
            if proc.returncode is None:
                proc.kill()
                proc.wait(timeout=5)


_FAKE_PID = 12345


def _install_fake_libproc(
    monkeypatch: pytest.MonkeyPatch,
    states: list[tuple[int, int]],
    taskinfo_rss: int | None,
) -> None:
    """`_load` を偽の `proc_pidinfo` に差し替える（キャッシュには触れないので汚れない）。

    `states` は SHORTBSDINFO の応答（status, flags）を呼び出し順に返し、最後の値を使い回す。
    `taskinfo_rss` が `None` なら TASKINFO は失敗（戻り値 0）、値があれば offset 8 に書く。"""
    calls = {"state": 0}

    def fake(pid: int, flavor: int, arg: int, buf: object, size: int) -> int:
        assert pid == _FAKE_PID
        if flavor == 13:
            assert (arg, size) == (1, 64)
            status, flags = states[min(calls["state"], len(states) - 1)]
            calls["state"] += 1
            raw = bytearray(64)
            raw[0:4] = pid.to_bytes(4, "little")
            raw[12:16] = status.to_bytes(4, "little")
            raw[32:36] = flags.to_bytes(4, "little")
            ctypes.memmove(buf, bytes(raw), 64)
            return 64
        assert (flavor, size) == (4, 96)
        if taskinfo_rss is None:
            return 0
        raw = bytearray(96)
        raw[8:16] = taskinfo_rss.to_bytes(8, "little")
        ctypes.memmove(buf, bytes(raw), 96)
        return 96

    monkeypatch.setattr(procinfo_darwin.sys, "platform", "darwin")
    monkeypatch.setattr(procinfo_darwin, "_load", lambda: fake)


def test_req39_exiting_state_with_taskinfo_failure_is_zero_not_zombie_327(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-39・#327: TASKINFO 失敗・再読で SRUN(2)＋INEXIT(0x4) は `(0, False)`。"""
    _install_fake_libproc(monkeypatch, [(2, 0x14), (2, 0x4014)], None)
    assert procinfo_darwin.child_status(_FAKE_PID) == (0, False)


def test_req39_taskinfo_failure_without_inexit_is_unknown_327(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-39・#327: TASKINFO 失敗・再読で SRUN かつ INEXIT なし（0x4010）は `None`。"""
    _install_fake_libproc(monkeypatch, [(2, 0x10), (2, 0x4010)], None)
    assert procinfo_darwin.child_status(_FAKE_PID) is None


def test_req39_taskinfo_failure_then_zombie_is_zombie_327(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-39・#327: TASKINFO 失敗・再読で SZOMB(5) は `(0, True)`。"""
    _install_fake_libproc(monkeypatch, [(2, 0x10), (5, 0x4010)], None)
    assert procinfo_darwin.child_status(_FAKE_PID) == (0, True)


def test_req39_taskinfo_success_returns_rss_even_with_inexit_327(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-39・#327: TASKINFO 成功時は INEXIT が立っていても RSS をそのまま返す。"""
    _install_fake_libproc(monkeypatch, [(2, 0x4014)], 123456789)
    assert procinfo_darwin.child_status(_FAKE_PID) == (123456789, False)
