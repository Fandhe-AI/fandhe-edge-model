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

from conftest import LABEL_ORDER, TINY_CONFIG
from fandhe_edge_trainer import contract, guard, supervisor

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


def test_monitor_child_treats_exit_between_wait_and_ps_as_normal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`wait` のタイムアウト直後に子が正常終了し `ps` が PID を見つけられない
    競合でも、監視失敗（monitor_failed）ではなく通常の終了として扱う。

    `_current_child_rss_bytes` が「子の終了を待ってから None を返す」ように
    差し替え、競合を決定的に再現する。
    """
    proc = _spawn("import time; time.sleep(0.3)")

    def _ps_after_exit(pid: int) -> None:
        proc.wait(timeout=5)
        return None

    monkeypatch.setattr(supervisor, "_current_child_rss_bytes", _ps_after_exit)
    try:
        reason = supervisor.monitor_child(
            proc,
            time_limit_seconds=30.0,
            rss_limit_bytes=64 * 1024 * 1024 * 1024,
            poll_interval=0.05,
            grace_seconds=0.0,
        )
        assert reason is None
        assert proc.returncode == 0
    finally:
        _reap(proc)


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

    ダミーの子プロセスが自分で `resource.setrlimit(RLIMIT_CPU, (1, 1))` を
    設定してから busy-loop することで、監視側からの強制終了ではなく、
    カーネルによる `SIGXCPU` での自己終了を再現する。
    """
    code = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (1, 1))\n"
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
    monkeypatch.setenv("PYTHONPATH", _SRC_DIR)  # 子プロセスが src/ を解決できるようにする

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
