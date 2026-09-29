"""`supervisor.py`（P0-2: 壁時計・RSS のプロセス外強制打ち切り。P1-1: 標準出力の
並行読み出し）のテスト。

`monitor_child`・`_drain_stdout` を、実際の学習ワーカーではなくダミーの子プロセス
（`sys.executable -c "..."`）に対して直接呼ぶことで、監視ロジックだけを
高速・決定的に検証する（証拠種別: テストハーネス）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time as time_mod
from pathlib import Path

import pytest

from conftest import LABEL_ORDER, TINY_CONFIG
from fandhe_edge_trainer import contract, guard, supervisor
from fandhe_edge_trainer.exitcode import ExitCode

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


def test_monitor_child_fails_closed_when_status_race_with_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """issue #178 PR #233 レビュー再々指摘 P0: `_current_child_status` が
    `None` を返す場合（`wait` のタイムアウト直後に子が正常終了し `ps` が
    PID を見つけられない競合を含む）は、正常終了かどうかを安全に区別する
    手段が無いため、常に `monitor_failed` として fail-closed に扱う
    （以前は `proc.poll()` で確認して通常終了として扱っていたが、これは
    `killpg` を経由しない回収経路になっていたため廃止した）。

    `_current_child_status` が「子がゾンビになるまで（＝回収せずに）待って
    から None を返す」ように差し替え、競合を決定的に再現する。**回収
    （`proc.wait()`）はしない**: `_terminate_and_reap` が呼ぶ `killpg` は
    「対象がまだ回収されていない（実行中またはゾンビ）」ことを前提に
    安全とされているため、このモック自身が先に回収してしまうと
    `_terminate_and_reap` が既に無効な pid へ `killpg` する形になり、
    検証したい不変条件と矛盾する（本物の `_current_child_status` を使って
    ゾンビになるのを待つだけで、回収は一切行わない）。
    """
    proc = _spawn("import time; time.sleep(0.3)")
    real_status = supervisor._current_child_status

    def _none_after_zombie(pid: int) -> tuple[int, bool] | None:
        deadline = time_mod.monotonic() + 5.0
        while time_mod.monotonic() < deadline:
            status = real_status(pid)
            if status is not None and status[1]:  # is_zombie
                break
            time_mod.sleep(0.01)
        return None

    monkeypatch.setattr(supervisor, "_current_child_status", _none_after_zombie)
    try:
        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=30.0,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            poll_interval=0.05,
            grace_seconds=0.0,
        )
        assert reason == "monitor_failed"
    finally:
        _reap(proc)


def test_monitor_child_treats_late_zombie_detection_as_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """issue #178 PR #233 レビュー再々々指摘 P1: `ps` でゾンビ（終了済み）と
    分かっても、それだけで「締め切り内に終了した」ことにはならない。`ps`
    のポーリング間隔・`_current_child_status` 自体の所要時間により、実際
    には壁時計の締め切りを過ぎてから初めてゾンビだと気づく場合がある。
    この場合、`_classify_self_exit`（正常終了・`RLIMIT_CPU` 自己終了の
    判定）を呼ばずに `"time"` として扱わなければならない（呼び出し元
    `_monitor_worker_and_finalize` はこれを他の強制終了理由と同様に
    予約解放のみ〔確定しない〕の経路へ流す。REQ-39。成果物が確定される
    経路は `killed_reason is None` のときだけであり、`"time"` はその経路
    に入らない）。

    `time.monotonic` を差し替え、「deadline 計算時は締め切り内」→
    「ゾンビ検知直後のチェック時は締め切りを大きく超えている」という
    競合を決定的に再現する（`_current_child_status` 自体は実プロセスの
    生死を問わず常にゾンビを報告するよう差し替え、`ps` のタイミングに
    左右されないようにする）。`subprocess` 内部のタイムアウト計算は
    モジュール読み込み時に `from time import monotonic as _time` で
    束縛された別参照を使うため、本差し替えの影響を受けない
    （`_terminate_and_reap` 内の `proc.wait(timeout=10)` は正常に動く）。
    """
    proc = _spawn("import time; time.sleep(60)")
    monkeypatch.setattr(supervisor, "_current_child_status", lambda pid: (1024 * 1024, True))

    calls = {"n": 0}

    def _fake_monotonic() -> float:
        calls["n"] += 1
        # 1 回目: `deadline = time.monotonic() + time_limit_seconds` の計算
        # （締め切りは 1.0 秒後になる）。2 回目以降: ゾンビ検知直後の
        # 締め切りチェックで、締め切りを大きく超えていることにする。
        return 0.0 if calls["n"] == 1 else 1000.0

    monkeypatch.setattr(supervisor.time, "monotonic", _fake_monotonic)

    try:
        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=1.0,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            poll_interval=0.05,
            grace_seconds=0.0,
        )
        assert reason == "time"
    finally:
        _reap(proc)


def test_run_supervised_train_does_not_finalize_artifact_on_late_zombie_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue #178 PR #233 レビュー再々々指摘 P1 への end-to-end 回帰:
    実際の `_worker`（`c3`・`TINY_CONFIG`）を締め切り内に正常終了させつつ、
    `monitor_child` がそのゾンビ状態に気づくタイミングだけを締め切り超過後
    にずらし、成果物（`out_dir`）が確定されず `limit_exceeded`（exit 20）に
    なることを確認する（`test_monitor_child_treats_late_zombie_detection_as_
    timeout` の単体テストに対し、`_monitor_worker_and_finalize` の予約解放
    経路まで含めた確認）。

    `_current_child_status` を実装をそのまま呼びつつ、初めてゾンビを観測
    した瞬間にフラグを立てるラッパーへ差し替える。`time.monotonic` は
    フラグが立つまでは実時間をそのまま返し、フラグが立った後は実時間へ
    大きなオフセットを足して返す。これにより、学習自体は通常どおり
    （既定の大きな `time_limit_seconds`〔`MAX_TRAIN_WALL_SECONDS`〕の下で）
    正常に完了しつつ、`monitor_child` 側だけが「ゾンビ検知の直後には
    締め切りを大きく超えていた」状況を観測する。
    """
    train_path = tmp_path / "train.jsonl"
    _write_train_data(train_path)
    out_dir = tmp_path / "out"
    request = {
        "schema_version": 1,
        "kind": "c3",
        "kind_version": 1,
        "config": TINY_CONFIG,
        "label_order": LABEL_ORDER,
        "max_bytes": 64,
        "seed": 0,
        "device": "cpu",
        "root": str(tmp_path),
        "train_path": "train.jsonl",
        "out_dir": "out",
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    real_status = supervisor._current_child_status
    late = {"is_late": False}

    def _status_and_flag_first_zombie(pid: int) -> tuple[int, bool] | None:
        status = real_status(pid)
        if status is not None and status[1]:  # is_zombie
            late["is_late"] = True
        return status

    monkeypatch.setattr(supervisor, "_current_child_status", _status_and_flag_first_zombie)

    real_monotonic = time_mod.monotonic

    def _fake_monotonic() -> float:
        return real_monotonic() + (10_000.0 if late["is_late"] else 0.0)

    monkeypatch.setattr(supervisor.time, "monotonic", _fake_monotonic)

    exit_code = supervisor.run_supervised_train(request_path)
    assert int(exit_code) == int(ExitCode.LIMIT_EXCEEDED)

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == "error"
    assert payload["code"] == "limit_exceeded"
    assert "time" in payload["message"]

    # 成果物は確定されておらず、予約（out_dir・作業用一時ディレクトリ）も
    # 残置されていない。
    assert not out_dir.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]


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


def test_monitor_child_maps_rlimit_cpu_self_kill_to_limit_exceeded() -> None:
    """P1: ワーカー自身が `RLIMIT_CPU`（`cli.py::_apply_rlimit_cpu_backstop` と
    同様の、自プロセスへのソフト上限設定）に達して自己終了した場合、
    `runtime_error` ではなく `limit_exceeded` として扱われること（`"cpu"` が
    返り、呼び出し元〔`_spawn_worker_and_finalize`〕はこれを他の強制終了理由と
    同じく `ExitCode.LIMIT_EXCEEDED` へ写す）。

    ダミーの子プロセスが自分で `resource.setrlimit(RLIMIT_CPU, (1, 3))` を
    設定してから busy-loop することで、監視側からの強制終了ではなく、
    カーネルによる `SIGXCPU` での自己終了を再現する。soft(1) < hard(3) に
    しているのは本番（`cli.py::_apply_rlimit_cpu_backstop`）と同じ形に
    揃えるため。Linux は soft == hard だとソフト上限到達時に `SIGXCPU` を
    送らず即 `SIGKILL` になり、`reason == "cpu"` を再現できない
    （`SIGXCPU` はハード上限との間の猶予でのみ配送される）。
    """
    code = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (1, 3))\n"
        "x = 0\n"
        "while True:\n"
        "    x += 1\n"
    )
    proc = _spawn(code)
    try:
        t0 = time_mod.monotonic()
        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=60.0,  # 壁時計では打ち切られない大きさにする
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            poll_interval=0.05,
            grace_seconds=0.0,
        )
        elapsed = time_mod.monotonic() - t0
        assert reason == "cpu"
        assert elapsed < 10.0  # ソフト上限（1 秒）＋ポーリング遅延程度で終わっていること
        assert proc.poll() is not None
    finally:
        _reap(proc)


# --------------------------------------------------------------------------
# `_terminate_worker`: `_worker` のプロセスグループ全体を終了させること
# （REQ-39。issue #178 PR #233 レビュー）
# --------------------------------------------------------------------------


def test_terminate_worker_kills_grandchild_too(tmp_path: Path) -> None:
    """REQ-39・issue #178 PR #233 レビュー: 内部タイムアウト時に `_worker` 役
    （`monitor_child` に渡す `proc`）だけでなく、その子（孫プロセス）も
    一括して終了すること（`_worker` を `start_new_session=True` で別
    グループへ切り離し、`os.killpg` でグループごと終了させる）。
    """
    heartbeat = tmp_path / "grandchild-heartbeat.txt"
    grandchild_script = tmp_path / "grandchild.py"
    grandchild_script.write_text(
        "import time\n"
        f"heartbeat = {str(heartbeat)!r}\n"
        "while True:\n"
        "    with open(heartbeat, 'a') as f:\n"
        "        f.write('.')\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    # `_worker` 役（`proc`）は、自分の子（孫プロセス）を「同じグループに
    # 残したまま」（`start_new_session` を指定しない＝デフォルトで継承）
    # 起動する。孫は heartbeat ファイルへ書き続ける。
    code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(grandchild_script)!r}])\n"
        "time.sleep(60)\n"
    )
    proc = _spawn(code)
    try:
        # `monitor_child` 自身の壁時計（`time_limit_seconds=0.2`）が起動する
        # 前に、孫プロセスが実際に起動済み（heartbeat が書かれ始めている）
        # ことを確認しておく。冷えた CI ランナーでは `python -c` の起動に
        # 100〜300ms かかることがあり、確認せずに `monitor_child` を呼ぶと
        # 「孫がまだ起動していないうちにタイムアウトが発火し、heartbeat が
        # 一度も作られない」という無関係な理由でテストが flaky になる。
        startup_deadline = time_mod.monotonic() + 5.0
        while time_mod.monotonic() < startup_deadline and not heartbeat.exists():
            time_mod.sleep(0.01)
        assert heartbeat.exists(), "grandchild must have started before invoking monitor_child"

        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=0.2,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            poll_interval=0.05,
            grace_seconds=0.0,
        )
        assert reason == "time"

        size_after_kill = heartbeat.stat().st_size
        time_mod.sleep(0.3)
        assert heartbeat.stat().st_size == size_after_kill, (
            "grandchild must not still be writing after monitor_child returns"
        )
    finally:
        _reap(proc)


def _process_alive(pid: int) -> bool:
    """`os.kill(pid, 0)` でシグナルを送らずに対象の生存を確認する。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 権限の理由で確認できない場合は「存在する」とみなす（fail-closed）。
        return True
    return True


def test_monitor_child_kills_grandchild_after_worker_exits_normally(tmp_path: Path) -> None:
    """REQ-39・issue #178 PR #233 レビュー再指摘 P0「worker が先に終了すると、
    子孫プロセスが残る」への回帰テスト: `_worker` 役が孫プロセスを起動した
    直後に自分自身はすぐ正常終了しても（孫を明示的に `wait()` しない）、
    `monitor_child` が戻った後には孫プロセスも消えていること。

    孫は supervisor（本テストプロセス）の子ではなく `_worker` 役の子である
    ため、本テストプロセスからは回収されずゾンビ判定の問題は起きない
    （`os.kill(pid, 0)` でそのまま生存確認できる）。ゾンビの残骸が一瞬
    見えることがあるため、少し待ってからもう一度確認する。
    """
    heartbeat = tmp_path / "grandchild-heartbeat.txt"
    grandchild_pid_path = tmp_path / "grandchild.pid"
    grandchild_script = tmp_path / "grandchild.py"
    grandchild_script.write_text(
        "import time\n"
        f"heartbeat = {str(heartbeat)!r}\n"
        "while True:\n"
        "    with open(heartbeat, 'a') as f:\n"
        "        f.write('.')\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    # `_worker` 役（`proc`）は、孫プロセスを起動した直後（`wait()` せず）に
    # 自分自身はすぐ正常終了する。孫は同じプロセスグループに残ったままに
    # なる（`start_new_session` を指定しないため）。
    code = (
        "import subprocess, sys\n"
        f"p = subprocess.Popen([sys.executable, {str(grandchild_script)!r}])\n"
        f"open({str(grandchild_pid_path)!r}, 'w').write(str(p.pid))\n"
    )
    proc = _spawn(code)
    try:
        startup_deadline = time_mod.monotonic() + 5.0
        while time_mod.monotonic() < startup_deadline and not (
            heartbeat.exists() and grandchild_pid_path.exists()
        ):
            time_mod.sleep(0.01)
        assert heartbeat.exists(), "grandchild must have started"
        grandchild_pid = int(grandchild_pid_path.read_text().strip())

        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=30.0,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            poll_interval=0.05,
            grace_seconds=0.0,
        )
        assert reason is None
        assert proc.returncode == 0

        deadline = time_mod.monotonic() + 5.0
        alive = _process_alive(grandchild_pid)
        while alive and time_mod.monotonic() < deadline:
            time_mod.sleep(0.05)
            alive = _process_alive(grandchild_pid)
        assert not alive, "grandchild must be terminated after monitor_child returns"
    finally:
        _reap(proc)


def test_monitor_child_kills_grandchild_when_status_always_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """issue #178 PR #233 レビュー再々指摘 P0 への回帰テスト: `_current_child_status`
    が常に `None`（`ps` の恒常的な失敗を模す）を返す場合でも、`monitor_child` は
    `proc.poll()` で正常終了を装って回収するのではなく、必ず `killpg` してから
    `monitor_failed` を返し、`_worker` 役が残した孫プロセスも終了させること。
    """
    heartbeat = tmp_path / "grandchild-heartbeat.txt"
    grandchild_pid_path = tmp_path / "grandchild.pid"
    grandchild_script = tmp_path / "grandchild.py"
    grandchild_script.write_text(
        "import time\n"
        f"heartbeat = {str(heartbeat)!r}\n"
        "while True:\n"
        "    with open(heartbeat, 'a') as f:\n"
        "        f.write('.')\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    # `_worker` 役（`proc`）は、孫プロセスを起動した直後に自分自身はすぐ
    # 正常終了する（孫は同じプロセスグループに残ったままになる）。
    code = (
        "import subprocess, sys\n"
        f"p = subprocess.Popen([sys.executable, {str(grandchild_script)!r}])\n"
        f"open({str(grandchild_pid_path)!r}, 'w').write(str(p.pid))\n"
    )
    proc = _spawn(code)
    monkeypatch.setattr(supervisor, "_current_child_status", lambda pid: None)
    try:
        startup_deadline = time_mod.monotonic() + 5.0
        while time_mod.monotonic() < startup_deadline and not (
            heartbeat.exists() and grandchild_pid_path.exists()
        ):
            time_mod.sleep(0.01)
        assert heartbeat.exists(), "grandchild must have started"
        grandchild_pid = int(grandchild_pid_path.read_text().strip())

        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=30.0,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            poll_interval=0.05,
            grace_seconds=0.0,
        )
        # `ps` が常に失敗する以上、正常終了として扱ってはならない
        # （`proc.poll()` を代替の確認手段に使っていた旧実装は、これを
        # 「正常終了」として誤って返していた）。
        assert reason == "monitor_failed"

        deadline = time_mod.monotonic() + 5.0
        alive = _process_alive(grandchild_pid)
        while alive and time_mod.monotonic() < deadline:
            time_mod.sleep(0.05)
            alive = _process_alive(grandchild_pid)
        assert not alive, "grandchild must be terminated even when status is unknown"
    finally:
        _reap(proc)


# --------------------------------------------------------------------------
# lifeline（issue #178 PR #233 レビュー: Rust 側でのプロセスグループ管理から
# worker 自身が親の死を検知する方式への全面移行）
# --------------------------------------------------------------------------


def test_lifeline_write_end_not_passed_to_worker() -> None:
    """REQ-39・issue #178 PR #233 レビュー: lifeline パイプの書き込み端は
    `_worker` へ渡さない（`_spawn_worker_and_finalize` の `pass_fds` には
    読み取り端だけを含める）こと。`_worker` 役の子プロセスが書き込み端の
    fd 番号へ書き込もうとすると `OSError`（対象の fd が存在しない）に
    なることで確認する。書き込み端が漏れていると、`_worker` 自身がその
    複製を保持し続けるため、supervisor が終了してもカーネルが書き込み端を
    閉じられず、lifeline の EOF が届かなくなる。
    """
    read_fd, write_fd = os.pipe()
    try:
        code = (
            "import os\n"
            "try:\n"
            f"    os.write({write_fd}, b'x')\n"
            "except OSError:\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(1)\n"
        )
        result = subprocess.run(  # noqa: S603 - テスト専用。引数は固定・shell 不使用
            [sys.executable, "-c", code],
            pass_fds=(read_fd,),
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, "worker role must not have access to the lifeline write end"
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_lifeline_worker_and_grandchild_die_when_supervisor_is_killed(tmp_path: Path) -> None:
    """REQ-39・issue #178 PR #233 レビュー: supervisor 役を `SIGKILL` すると、
    lifeline により worker 役・孫プロセスが有界時間内に消滅すること。

    supervisor がどのような形で終了しても（正常終了・内部タイムアウトに
    よる `killpg`・外側からの `SIGKILL` を含む）、カーネルが lifeline の
    書き込み端を自動的に閉じるため、worker 側の EOF 検知は必ず働く、という
    設計の核心を検証する（`cli.py::_start_lifeline_thread` を実プロセスへ
    組み込んで確認する。証拠種別: テストハーネス）。
    """
    worker_pid_path = tmp_path / "worker.pid"
    grandchild_pid_path = tmp_path / "grandchild.pid"

    worker_script = tmp_path / "worker_role.py"
    worker_script.write_text(
        "import os, subprocess, sys, time\n"
        f"sys.path.insert(0, {_SRC_DIR!r})\n"
        "from fandhe_edge_trainer import cli\n"
        "lifeline_fd = int(sys.argv[1])\n"
        "cli._start_lifeline_thread(lifeline_fd)\n"
        f"open({str(worker_pid_path)!r}, 'w').write(str(os.getpid()))\n"
        "grandchild = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time\\nwhile True: time.sleep(0.05)']\n"
        ")\n"
        f"open({str(grandchild_pid_path)!r}, 'w').write(str(grandchild.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    supervisor_script = tmp_path / "supervisor_role.py"
    supervisor_script.write_text(
        "import os, subprocess, sys, time\n"
        "r, w = os.pipe()\n"
        f"subprocess.Popen(\n"
        f"    [sys.executable, {str(worker_script)!r}, str(r)],\n"
        "    pass_fds=(r,),\n"
        "    start_new_session=True,\n"
        ")\n"
        "os.close(r)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    supervisor_proc = subprocess.Popen(  # noqa: S603 - テスト専用。引数は固定・shell 不使用
        [sys.executable, str(supervisor_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        startup_deadline = time_mod.monotonic() + 5.0
        while time_mod.monotonic() < startup_deadline and not (
            worker_pid_path.exists() and grandchild_pid_path.exists()
        ):
            time_mod.sleep(0.01)
        assert worker_pid_path.exists(), "worker role must have started"
        assert grandchild_pid_path.exists(), "grandchild must have started"

        worker_pid = int(worker_pid_path.read_text().strip())
        grandchild_pid = int(grandchild_pid_path.read_text().strip())

        supervisor_proc.kill()  # SIGKILL
        supervisor_proc.wait(timeout=5)

        for role, pid in (("worker", worker_pid), ("grandchild", grandchild_pid)):
            death_deadline = time_mod.monotonic() + 5.0
            while time_mod.monotonic() < death_deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time_mod.sleep(0.02)
            else:
                pytest.fail(
                    f"{role} (pid {pid}) did not die within the bound after supervisor SIGKILL"
                )
    finally:
        if supervisor_proc.poll() is None:
            supervisor_proc.kill()
            supervisor_proc.wait(timeout=5)


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


def test_worker_argv_uses_isolated_mode_and_launch_script() -> None:
    """`supervisor.worker_argv`（Issue #12）が組み立てる argv が、`sys.executable`・
    `-I`（隔離モード）・実在する `trainer/launch.py` の絶対パス・
    `["_worker", "--out-fd", "<n>", "--lifeline-fd", "<n>"]` から成ることを
    具体値で確認する（`--lifeline-fd`: issue #178 PR #233 レビュー）。
    """
    argv = supervisor.worker_argv(7, 8)
    assert argv[0] == sys.executable
    assert argv[1] == "-I"
    launch_script = Path(argv[2])
    assert launch_script.is_absolute()
    assert launch_script.name == "launch.py"
    assert launch_script.is_file()
    assert argv[3:] == ["_worker", "--out-fd", "7", "--lifeline-fd", "8"]


def test_run_supervised_train_rejects_invalid_request_without_spawning_worker(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({"schema_version": 1, "kind": "c3"}), encoding="utf-8")
    exit_code = supervisor.run_supervised_train(request_path)
    assert int(exit_code) == 64


# --------------------------------------------------------------------------
# P1: 検証済みリクエストを固定した内容として子プロセスへ渡す（Codex レビュー
# 指摘）。予約後にリクエストファイルが書き換えられても、子プロセスは
# 書き換え前の（検証済みの）内容で動くこと。
# --------------------------------------------------------------------------


def _write_train_data(path: Path) -> None:
    rows = []
    for i in range(12):
        rows.append({"input": f"alpha alpha beta gamma {i}", "label": "cat_a"})
        rows.append({"input": f"delta delta epsilon zeta {i}", "label": "cat_b"})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_run_supervised_train_ignores_request_file_rewrite_after_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-39・REQ-21: `contract.prepare_out_dir`（予約）の直後にリクエスト
    ファイルの中身を別の `out_dir`（`out2`）を指すよう書き換えても、子プロセス
    （`_worker`）は予約前に検証・確定した元の内容（`out`）で動作し、成果物が
    元の `out_dir` に確定すること。`_worker` はファイルパスを再読込しない
    （検証済みのバイト列を標準入力から渡される。`supervisor.py`・`cli.py` の
    モジュール docstring 参照）。
    """
    # Issue #12: `_worker` の起動（`supervisor.worker_argv`）は `-I` 隔離モードで
    # `trainer/launch.py` を経由するため `PYTHONPATH` は不要かつ無視される。
    # 明らかに無効な値を設定しても子プロセスの起動・成功に影響しないことを示す
    # （`PYTHONPATH` が実は必要という退行があれば、この不正な値のせいで
    #  `fandhe_edge_trainer` の import に失敗し本テストが検出する）。
    monkeypatch.setenv("PYTHONPATH", "/nonexistent/should-be-ignored-by-dash-i")

    train_path = tmp_path / "train.jsonl"
    _write_train_data(train_path)
    out_dir = tmp_path / "out"
    other_out_dir = tmp_path / "out2"
    request = {
        "schema_version": 1,
        "kind": "c3",
        "kind_version": 1,
        "config": TINY_CONFIG,
        "label_order": LABEL_ORDER,
        "max_bytes": 64,
        "seed": 0,
        "device": "cpu",
        "root": str(tmp_path),
        "train_path": "train.jsonl",
        "out_dir": "out",
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    real_prepare_out_dir = contract.prepare_out_dir

    def _rewrite_request_then_prepare(entry):
        reservation = real_prepare_out_dir(entry)
        # 予約成功の直後、検証済みリクエストの確定前に、ファイル側の内容だけを
        # 別の out_dir（out2）・存在しない train_path を指すよう書き換える
        # （別プロセスによる改変を模す）。旧実装（`_worker` がファイルパスを
        # 再読込していた実装）であれば、この書き換え後の内容（存在しない
        # train_path）を読んでしまい invalid_path で exit 64 になっていた。
        # 本実装（検証済みのバイト列を固定して渡す）であれば、書き換え前の
        # 内容のまま学習が成功し exit 0 になる。この違いが本テストの識別力。
        rewritten = dict(request, out_dir="out2", train_path="nonexistent-train.jsonl")
        request_path.write_text(json.dumps(rewritten), encoding="utf-8")
        return reservation

    monkeypatch.setattr(contract, "prepare_out_dir", _rewrite_request_then_prepare)

    exit_code = supervisor.run_supervised_train(request_path)
    assert int(exit_code) == 0

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    # 成功 JSON の artifact_dir は書き換え前（検証済み）の out_dir と一致する。
    assert Path(payload["artifact_dir"]).resolve() == out_dir.resolve()
    assert (out_dir / "model.onnx").exists()
    assert (out_dir / "artifact.json").exists()
    # 書き換え後に指定された out2 へは一切書き込まれていない。
    assert not other_out_dir.exists()


# --------------------------------------------------------------------------
# セキュリティ監査指摘: `tempfile.TemporaryFile()`・write・seek が例外を送出
# した場合に、予約済みの out_dir・作業用一時ディレクトリを残置しないこと。
# --------------------------------------------------------------------------


def test_spawn_worker_and_finalize_cleans_up_reservation_when_tempfile_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: `tempfile.TemporaryFile()`（作成・write・seek を含む）が
    （ENOSPC 等で）例外を送出した場合でも、既に確保済みの予約（`out_dir`・
    作業用一時ディレクトリ）が残置されず解放されること。実際の資源枯渇を
    待たず、`tempfile.TemporaryFile` を差し替えて決定的に再現する
    （証拠種別: テストハーネス）。
    """
    root_handle = guard.resolve_root(str(tmp_path))
    entry = guard.confine(root_handle, "out", "out_dir")
    root_handle.close()
    reservation = contract.prepare_out_dir(entry)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated ENOSPC while creating the request temp file")

    monkeypatch.setattr(supervisor.tempfile, "TemporaryFile", _boom)

    try:
        with pytest.raises(OSError, match="simulated ENOSPC"):
            supervisor._spawn_worker_and_finalize(
                b'{"schema_version": 1}',
                reservation,
                time_limit_seconds=30.0,
                rss_limit_bytes=64 * 1024 * 1024 * 1024,
            )

        # 予約済み out_dir・作業用一時ディレクトリのいずれも残っていない
        # （cleanup_reservation が例外経路でも呼ばれたことの確認）。
        assert not (tmp_path / "out").exists()
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]
    finally:
        entry.close()


def test_spawn_worker_and_finalize_cleans_up_reservation_when_pipe_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """issue #178 PR #233 レビュー再々指摘 P1: lifeline パイプ作成
    （`os.pipe()`）が（fd 数の上限超過等で）`OSError` を送出した場合でも、
    既に確保済みの予約（`out_dir`・作業用一時ディレクトリ）が残置されず
    解放されること。`os.pipe` を差し替えて決定的に再現する
    （証拠種別: テストハーネス）。
    """
    root_handle = guard.resolve_root(str(tmp_path))
    entry = guard.confine(root_handle, "out", "out_dir")
    root_handle.close()
    reservation = contract.prepare_out_dir(entry)

    def _boom() -> tuple[int, int]:
        raise OSError("simulated EMFILE while creating the lifeline pipe")

    monkeypatch.setattr(supervisor.os, "pipe", _boom)

    try:
        exit_code = supervisor._spawn_worker_and_finalize(
            b'{"schema_version": 1}',
            reservation,
            time_limit_seconds=30.0,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
        )
        assert exit_code == ExitCode.RUNTIME_ERROR

        # 予約済み out_dir・作業用一時ディレクトリのいずれも残っていない
        # （cleanup_reservation が呼ばれたことの確認）。
        assert not (tmp_path / "out").exists()
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]

        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["status"] == "error"
        assert payload["code"] == "runtime_error"
    finally:
        entry.close()


# --------------------------------------------------------------------------
# 協調キャンセル（REQ-34・TASK-34.1-2・issue #145）。偽の `_worker`（MLX 不要・
# CPU のみ。証拠種別: テストハーネス〔模擬プロセス〕）で、キャンセル後に公開場所
# （`out_dir`）へ途中成果物が残らないこと、確定後のキャンセルは成功報告のまま
# であること（公開 ⇔ 成功報告）を具体値で確認する。
# --------------------------------------------------------------------------

#: 偽の `_worker`。`--out-fd` の dir へ書きかけの `model.onnx` を置き、`hang` なら
#: pid と ready を書いて眠り続ける。`ok` なら整合した成果物を書いて exit 0 する。
_FAKE_COOP_WORKER = """
import hashlib, json, os, sys, time
argv = sys.argv
out_fd = int(argv[argv.index("--out-fd") + 1])
mode = argv[argv.index("--mode") + 1]
ctl = argv[argv.index("--ctl") + 1]
def put(name, data):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=out_fd)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
if mode == "hang":
    put("model.onnx", b"partial")
    with open(os.path.join(ctl, "pid"), "w") as f:
        f.write(str(os.getpid()))
    with open(os.path.join(ctl, "ready"), "w") as f:
        f.write("1")
    time.sleep(120)
else:
    onnx = b"onnx-bytes"
    put("model.onnx", onnx)
    put("artifact.json", json.dumps({"onnx_sha256": hashlib.sha256(onnx).hexdigest()}).encode())
    print(json.dumps({"status": "ok", "artifact_dir": "x", "artifact": {}}))
"""


def _coop_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> tuple[Path, Path, Path]:
    """偽 `_worker` を差し込み、(request_path, out_dir, ctl_dir) を返す。"""
    train_path = tmp_path / "train.jsonl"
    _write_train_data(train_path)
    request = {
        "schema_version": 1,
        "kind": "c3",
        "kind_version": 1,
        "config": TINY_CONFIG,
        "label_order": LABEL_ORDER,
        "max_bytes": 64,
        "seed": 0,
        "device": "cpu",
        "root": str(tmp_path),
        "train_path": "train.jsonl",
        "out_dir": "out",
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    ctl_dir = tmp_path / "ctl"
    ctl_dir.mkdir()
    script = ctl_dir / "fake_worker.py"
    script.write_text(_FAKE_COOP_WORKER, encoding="utf-8")

    def _argv(out_fd: int, lifeline_fd: int) -> list[str]:
        return [
            sys.executable,
            str(script),
            "--out-fd",
            str(out_fd),
            "--lifeline-fd",
            str(lifeline_fd),
            "--mode",
            mode,
            "--ctl",
            str(ctl_dir),
        ]

    monkeypatch.setattr(supervisor, "worker_argv", _argv)
    return request_path, tmp_path / "out", ctl_dir


def _tmp_leftovers(tmp_path: Path) -> list[str]:
    return [p.name for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]


def test_req34_cancel_mid_run_leaves_nothing_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-34・TASK-34.1-2: 学習中にキャンセルすると、公開場所（`out_dir`）も
    作業用一時ディレクトリも残らない（PoC-19 の `out_dir_exists == False`・
    `out_dir_listing == []` に相当）。worker は止まり、出力 JSON は
    `runtime_error`（cancelled）。"""
    request_path, out_dir, ctl_dir = _coop_setup(tmp_path, monkeypatch, "hang")
    event = threading.Event()
    result: dict[str, ExitCode] = {}
    runner = threading.Thread(
        target=lambda: result.update(
            code=supervisor.run_supervised_train(request_path, cancel_event=event)
        )
    )
    runner.start()
    deadline = time_mod.monotonic() + 30
    while not (ctl_dir / "ready").exists() and time_mod.monotonic() < deadline:
        time_mod.sleep(0.02)
    assert (ctl_dir / "ready").exists()
    # 予約済み `out_dir` と、書きかけの成果物を含む一時ディレクトリがある状態。
    assert out_dir.is_dir()
    assert len(_tmp_leftovers(tmp_path)) == 1
    event.set()
    runner.join(timeout=30)
    assert not runner.is_alive()

    assert result["code"] == ExitCode.RUNTIME_ERROR
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload == {
        "status": "error",
        "code": "runtime_error",
        "message": "training cancelled by caller",
    }
    assert not out_dir.exists()
    assert _tmp_leftovers(tmp_path) == []
    pid = int((ctl_dir / "pid").read_text())
    stop_deadline = time_mod.monotonic() + 5
    while _process_alive(pid) and time_mod.monotonic() < stop_deadline:
        time_mod.sleep(0.05)
    assert not _process_alive(pid)


def test_req34_cancel_after_worker_output_before_verify_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-34: worker が成果物を書き終えて終了した後でも、検証・確定の前に届いた
    キャンセルは確定へ進まず、`out_dir` は公開されない。"""
    request_path, out_dir, _ = _coop_setup(tmp_path, monkeypatch, "ok")
    event = threading.Event()
    real_parse = supervisor._parse_worker_stdout

    def _parse_then_cancel(raw: bytes):
        payload = real_parse(raw)
        event.set()
        return payload

    monkeypatch.setattr(supervisor, "_parse_worker_stdout", _parse_then_cancel)
    code = supervisor.run_supervised_train(request_path, cancel_event=event)
    assert code == ExitCode.RUNTIME_ERROR
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["message"] == "training cancelled by caller"
    assert not out_dir.exists()
    assert _tmp_leftovers(tmp_path) == []


def test_req34_cancel_with_incomplete_cleanup_does_not_report_cooperative_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-34: 予約の解放を確認できない（rmdir 失敗）場合は所定の協調キャンセル
    応答を出さず、呼び出し側が残置ありとして扱える別メッセージを返す。"""
    request_path, out_dir, _ = _coop_setup(tmp_path, monkeypatch, "ok")
    event = threading.Event()
    real_parse = supervisor._parse_worker_stdout

    def _parse_then_cancel(raw: bytes):
        payload = real_parse(raw)
        event.set()
        return payload

    def _rmdir_fails(*args: object, **kwargs: object) -> None:
        raise PermissionError("simulated rmdir failure")

    monkeypatch.setattr(supervisor, "_parse_worker_stdout", _parse_then_cancel)
    monkeypatch.setattr(contract.os, "rmdir", _rmdir_fails)
    code = supervisor.run_supervised_train(request_path, cancel_event=event)
    assert code == ExitCode.RUNTIME_ERROR
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["message"] == "training cancelled but cleanup incomplete"
    assert out_dir.exists()


def test_req34_cancel_right_before_finalize_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-34: 整合検証（`verify_output`）の直後・確定の直前に届いたキャンセルも
    確定させない（最後のキャンセル確認）。"""
    request_path, out_dir, _ = _coop_setup(tmp_path, monkeypatch, "ok")
    event = threading.Event()
    real_verify = supervisor.artifact_mod.verify_output

    def _verify_then_cancel(dir_fd: int) -> None:
        real_verify(dir_fd)
        event.set()

    monkeypatch.setattr(supervisor.artifact_mod, "verify_output", _verify_then_cancel)
    code = supervisor.run_supervised_train(request_path, cancel_event=event)
    assert code == ExitCode.RUNTIME_ERROR
    capsys.readouterr()
    assert not out_dir.exists()
    assert _tmp_leftovers(tmp_path) == []


def test_req34_cancel_after_finalize_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-34: 確定（rename）が済んだ後に届いたキャンセルは無視し、成功を
    報告する（公開されている ⇔ 成功報告）。"""
    request_path, out_dir, _ = _coop_setup(tmp_path, monkeypatch, "ok")
    event = threading.Event()
    real_finalize = contract.finalize_out_dir

    def _finalize_then_cancel(reservation: contract.OutDirReservation) -> None:
        real_finalize(reservation)
        event.set()

    monkeypatch.setattr(contract, "finalize_out_dir", _finalize_then_cancel)
    code = supervisor.run_supervised_train(request_path, cancel_event=event)
    assert code == ExitCode.OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert sorted(os.listdir(out_dir)) == ["artifact.json", "model.onnx"]
    assert _tmp_leftovers(tmp_path) == []


def test_req34_monitor_child_returns_cancelled_and_reaps() -> None:
    """REQ-34: `cancel_event` が立つと `monitor_child` は子を止めて回収し、
    `"cancelled"` を返す。"""
    proc = _spawn("import time; time.sleep(60)")
    event = threading.Event()
    event.set()
    try:
        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=60,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            cancel_event=event,
        )
        assert reason == "cancelled"
        assert proc.returncode is not None
    finally:
        _reap(proc)


def test_req34_monitor_child_prefers_cancelled_over_normal_exit_on_zombie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-34: 既に終了した（ゾンビ）worker を検知した時点でキャンセル済みなら、
    通常終了（`None`）ではなく `"cancelled"` を返し、成果物を確定させない。"""
    proc = _spawn("pass")
    event = threading.Event()
    event.set()
    monkeypatch.setattr(supervisor, "_current_child_status", lambda pid: (1024, True))
    try:
        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=60,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            cancel_event=event,
        )
        assert reason == "cancelled"
    finally:
        _reap(proc)


def test_req34_monitor_child_prefers_cancelled_when_ps_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-34: キャンセル済みで `ps` が失敗しても `monitor_failed` にせず
    `"cancelled"` を返す（stdout drain を待たない高速経路へ流す）。"""
    proc = _spawn("hang")
    event = threading.Event()
    event.set()
    monkeypatch.setattr(supervisor, "_current_child_status", lambda pid: None)
    try:
        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=60,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            cancel_event=event,
        )
        assert reason == "cancelled"
    finally:
        _reap(proc)


def test_req34_cancel_before_reservation_does_not_reserve_or_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-34: 予約前にキャンセル済みなら out_dir を予約せず worker も起動せず、
    所定のキャンセル応答を返す。"""
    request_path, out_dir, ctl_dir = _coop_setup(tmp_path, monkeypatch, "hang")
    event = threading.Event()
    event.set()
    code = supervisor.run_supervised_train(request_path, cancel_event=event)
    assert code == ExitCode.RUNTIME_ERROR
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["message"] == "training cancelled by caller"
    assert not out_dir.exists()
    assert _tmp_leftovers(tmp_path) == []
    assert not (ctl_dir / "ready").exists()


def test_req34_cancel_after_reservation_before_spawn_releases_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-34: 予約直後（worker 起動前）にキャンセルされたら、保持中の fd で予約を
    解放して所定のキャンセル応答を返し、worker は起動しない。"""
    request_path, out_dir, ctl_dir = _coop_setup(tmp_path, monkeypatch, "hang")
    event = threading.Event()
    real_prepare = contract.prepare_out_dir

    def _prepare_then_cancel(entry):
        reservation = real_prepare(entry)
        event.set()
        return reservation

    monkeypatch.setattr(contract, "prepare_out_dir", _prepare_then_cancel)
    code = supervisor.run_supervised_train(request_path, cancel_event=event)
    assert code == ExitCode.RUNTIME_ERROR
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["message"] == "training cancelled by caller"
    assert not out_dir.exists()
    assert _tmp_leftovers(tmp_path) == []
    assert not (ctl_dir / "ready").exists()
