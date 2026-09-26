"""学習ワーカーのスーパーバイザー（`train`。公開サブコマンドの実体。P0-2）。

`onnx.checker.check_model`・`SerializeToString`・単一の MLX 演算のような「1 回の
呼び出しが長時間かかりうる同期呼び出し」の最中は、ワーカー内の協調的な資源チェック
（`budget.py::ResourceBudget.check`）は次に Python コードへ制御が戻るまで実行され
ない。そのため、壁時計・RSS の強制打ち切りはプロセス境界の外側から行う必要がある。

**本モジュールは意図的に mlx・onnx・numpy を import しない**（学習の実行に必要な
重い依存が無くても、別プロセス〔`_worker`〕の起動・監視・強制終了・異常終了時の
出力後始末だけなら本モジュールだけで完結する設計）。`tests/test_supervisor.py::
test_supervisor_module_does_not_import_mlx` で検証する。

流れ:
1. リクエスト JSON を `contract.load_request`/`contract.validate_request` で
   完全に検証する（`_worker` と同じ検証・同じエラーメッセージ。ここで失敗すれば
   子プロセスは起動しない）。検証だけが目的なので、得た fd はすぐ閉じる。
2. `python -m fandhe_edge_trainer _worker --request <path>` を子プロセスとして
   起動する（新しいプロセスグループ。`os.killpg` で子とその子孫をまとめて
   強制終了できるようにする）。
3. 0.1 秒間隔でポーリングする: 壁時計（`time.monotonic`）が
   `time_limit_seconds` ＋ 猶予（`_TIME_LIMIT_GRACE_SECONDS`）を超えたか、
   `ps`（絶対パス `/bin/ps`）で読んだ子プロセスの RSS が `rss_limit_bytes` を
   超えたかを見る。`ps` の実行自体に失敗したら「監視ができない」ことを
   fail-closed に扱い、子プロセスを強制終了して `runtime_error` とする
   （安全側に倒す。上限を検査できないまま野放しにしない）。
4. 超過を検出したら子プロセスのプロセスグループを `SIGKILL` し、`out_dir` の
   予約（`.{name}.tmp-*` の一時ディレクトリ・空の予約済みディレクトリ）を
   `guard.confine` で経路を再確認したうえで片付ける
   （`contract.cleanup_orphaned_reservation`）。
5. 子プロセスが自分で終了した場合: 標準出力を上限（`_MAX_WORKER_STDOUT_BYTES`）
   付きで読み、「ちょうど 1 つの妥当な JSON オブジェクトである」ことを確認して
   から、そのまま再出力する。子プロセスの終了コードが 7 種のいずれかであれば
   それをそのまま使い、そうでなければ（シグナルによる終了を含め）
   `runtime_error`（exit 70）とする。

Rust 側ジョブ管理（REQ-34）が最終的にはこの「外側のスーパーバイザー」の役割を
担う計画であり、本モジュールは Rust 側が無い・本ワーカーが単独プロセスとして
起動される場合の多層防御（defense in depth）として存在する。
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import contract, guard
from .errors import WorkerError
from .exitcode import ExitCode

#: 監視ループのポーリング間隔（秒）。
_POLL_INTERVAL_SECONDS = 0.1

#: 壁時計の上限（`time_limit_seconds`）に上乗せする猶予（秒）。ワーカー内の
#: 協調的なチェック（バッチの境目等）が自ら気づいて終了する猶予を与える。
#: それでも終わらない場合（単一の長い同期呼び出しの最中等）はここで強制終了する。
_TIME_LIMIT_GRACE_SECONDS = 5.0

#: 子プロセスの標準出力の上限（bytes）。
_MAX_WORKER_STDOUT_BYTES = 1 * 1024 * 1024

#: `ps` の絶対パス（`shell=True` を使わず、`PATH` に依存しない）。
_PS_BIN = "/bin/ps"

#: 7 種の終了コード（REQ-21）。子プロセスの終了コードがこの集合に無ければ
#: `runtime_error` として扱う。
_VALID_EXIT_CODES = {int(code) for code in ExitCode}


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _current_child_rss_bytes(pid: int) -> int | None:
    """`ps -o rss= -p <pid>`（bytes 単位に変換済み）。取得できなければ None。

    `ps` の RSS 出力は KiB 単位（BSD/macOS・Linux とも `-o rss=` は KiB）。
    """
    try:
        result = subprocess.run(  # noqa: S603 - 引数は固定リスト。shell 不使用。絶対パスの /bin/ps のみを呼ぶ
            [_PS_BIN, "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    try:
        kib = int(text)
    except ValueError:
        return None
    return kib * 1024


def _kill_process_group(proc: subprocess.Popen) -> None:
    """子プロセスのプロセスグループ全体を `SIGKILL` する（`start_new_session=True`
    で起動しているため、子とその子孫だけを含む独立したグループになっている）。
    """
    with contextlib.suppress(OSError):
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()  # プロセスグループの取得自体に失敗した場合の保険


def monitor_child(
    proc: subprocess.Popen,
    *,
    time_limit_seconds: float,
    rss_limit_bytes: int,
    poll_interval: float = _POLL_INTERVAL_SECONDS,
    grace_seconds: float = _TIME_LIMIT_GRACE_SECONDS,
) -> str | None:
    """`proc` を監視する。正常終了したら `None` を返す。

    壁時計・RSS のいずれかが上限を超えた場合、または RSS の監視自体が失敗した
    場合（`ps` の失敗）はプロセスグループを強制終了し、理由
    （`"time"`/`"rss"`/`"monitor_failed"`）を返す（呼び出し元が `proc.wait()`
    済みであることを前提にせず、本関数が確実に終了させてから返る）。

    テスト（`tests/test_supervisor.py`）は本関数を直接、ダミーの子プロセス
    （`sys.executable -c "..."`）に対して呼ぶことで、実際の学習ワーカーを
    起動せずに監視ロジックを検証する。
    """
    deadline = time.monotonic() + time_limit_seconds + grace_seconds
    while True:
        try:
            proc.wait(timeout=poll_interval)
            return None
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() > deadline:
            _kill_process_group(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
            return "time"
        rss = _current_child_rss_bytes(proc.pid)
        if rss is None:
            # 監視できないこと自体を fail-closed に扱う（上限を検査できない
            # まま子プロセスを走らせ続けない）。
            _kill_process_group(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
            return "monitor_failed"
        if rss > rss_limit_bytes:
            _kill_process_group(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
            return "rss"


def _cleanup_orphaned_output(root_raw: Any, out_dir_raw: Any) -> None:
    """強制終了させたワーカーが残したかもしれない出力予約を片付ける。

    `guard.confine` で経路を改めて確認したうえで（TOCTOU 対策。多層防御）、
    `contract.cleanup_orphaned_reservation` に委ねる。`root`・`out_dir` 自体が
    不正（既にリクエスト検証を通っているはずだが、念のため）なら何もしない
    （最悪でも「掃除できなかった」だけで、経路の閉じ込めが破れることはない）。
    """
    try:
        root_handle = guard.resolve_root(root_raw)
    except WorkerError:
        return
    try:
        entry = guard.confine(root_handle, out_dir_raw, "out_dir")
    except WorkerError:
        root_handle.close()
        return
    try:
        contract.cleanup_orphaned_reservation(entry)
    finally:
        entry.close()
        root_handle.close()


def _parse_worker_stdout(raw: bytes) -> dict[str, Any] | None:
    """子プロセスの標準出力が「ちょうど 1 つの妥当な JSON オブジェクト」で
    あることを確認する。サイズ超過・複数行・JSON でない・オブジェクトでない
    場合は `None` を返す（呼び出し側が `runtime_error` として扱う）。
    """
    if len(raw) > _MAX_WORKER_STDOUT_BYTES:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def run_supervised_train(request_path: Path) -> ExitCode:
    """`train` サブコマンドの本体。`_worker` を子プロセスとして起動・監視する。"""
    try:
        raw = contract.read_request_dict(request_path)
        request = contract.validate_request(raw)
    except WorkerError as e:
        _emit({"status": "error", "code": e.code, "message": e.message})
        return e.exit_code

    time_limit_seconds = request.time_limit_seconds
    rss_limit_bytes = request.rss_limit_bytes
    root_raw = raw.get("root")
    out_dir_raw = raw.get("out_dir")
    # 検証だけが目的で、実際の読み書きは子プロセス（_worker）が独立に行うため、
    # ここで得た fd はすぐ閉じる（スーパーバイザーは fd を長時間保持しない）。
    request.close_resources()

    argv = [sys.executable, "-m", "fandhe_edge_trainer", "_worker", "--request", str(request_path)]
    try:
        proc = subprocess.Popen(  # noqa: S603 - 引数は固定リスト。shell 不使用。sys.executable は絶対パス
            argv,
            stdout=subprocess.PIPE,
            stderr=None,  # 継承（親の stderr へ直接流す。パイプを溜めて詰まらせない）
            start_new_session=True,
        )
    except OSError as e:
        _emit(
            {
                "status": "error",
                "code": "runtime_error",
                "message": f"failed to start worker: {type(e).__name__}",
            }
        )
        return ExitCode.RUNTIME_ERROR

    killed_reason = monitor_child(
        proc, time_limit_seconds=float(time_limit_seconds), rss_limit_bytes=rss_limit_bytes
    )

    if killed_reason is not None:
        if proc.stdout is not None:
            with contextlib.suppress(OSError):
                proc.stdout.close()
        _cleanup_orphaned_output(root_raw, out_dir_raw)
        if killed_reason == "monitor_failed":
            _emit(
                {
                    "status": "error",
                    "code": "runtime_error",
                    "message": "resource monitoring failed (ps unavailable); worker terminated",
                }
            )
            return ExitCode.RUNTIME_ERROR
        _emit(
            {
                "status": "error",
                "code": "limit_exceeded",
                "message": f"worker exceeded the {killed_reason} limit and was terminated",
            }
        )
        return ExitCode.LIMIT_EXCEEDED

    stdout_bytes = proc.stdout.read(_MAX_WORKER_STDOUT_BYTES + 1) if proc.stdout else b""
    if proc.stdout is not None:
        proc.stdout.close()
    returncode = proc.returncode

    if returncode is not None and returncode < 0:
        # シグナルによる終了（例: OOM killer・外部からの kill）。
        _cleanup_orphaned_output(root_raw, out_dir_raw)
        _emit(
            {
                "status": "error",
                "code": "runtime_error",
                "message": f"worker terminated by signal {-returncode}",
            }
        )
        return ExitCode.RUNTIME_ERROR

    payload = _parse_worker_stdout(stdout_bytes)
    if payload is None:
        _emit(
            {
                "status": "error",
                "code": "runtime_error",
                "message": "worker stdout was not exactly one valid JSON object",
            }
        )
        return ExitCode.RUNTIME_ERROR

    if returncode not in _VALID_EXIT_CODES:
        _emit(
            {
                "status": "error",
                "code": "runtime_error",
                "message": f"worker exited with an unexpected code {returncode}",
            }
        )
        return ExitCode.RUNTIME_ERROR

    _emit(payload)
    return ExitCode(returncode)
