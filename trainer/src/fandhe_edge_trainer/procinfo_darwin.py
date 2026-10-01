"""macOS 用の子プロセス状態取得（libproc の `proc_pidinfo` を ctypes で呼ぶ）。

`supervisor.py` の `_current_child_status`（darwin 分岐）から呼ばれ、`_worker` の
RSS とゾンビ状態を **外部プロセスを起動せずに** 取得する。macOS の `/bin/ps` は
setuid root のため `sandbox-exec` 下では exec が EPERM になり、監視が不能になって
`runtime_error` で終わっていた（#327。REQ-38 の sandbox 下の完走・REQ-39 の資源
上限監視）。libproc の呼び出しは setuid に依存せず、sandbox の内外で同じ結果を返す
ことを実機（Apple M4 Max・macOS 27.0・Python 3.12）で確認している（証拠種別: 実機）。

契約（fail-closed）:

- 戻り値は `(rss_bytes, is_zombie)`、または状態を確実に読めないときの `None`。
  `None` を受けた呼び出し側（`monitor_child`）は子を終了して監視失敗として扱う。
- ライブラリのロード失敗・シンボル欠落・ctypes の例外・想定外のバイト数・pid の
  不一致は、すべて `None`。例外は外へ出さない。
- 本モジュールは子を回収（reap）しない。ゾンビの検知は回収せずに行う
  （`supervisor.py` の「killpg は回収より先」の不変条件を守るため）。
- darwin 以外で import しても失敗しない（ライブラリは初回呼び出しまで読み込まない）。
  darwin 以外で呼ぶと `None` を返す。

定数・オフセットの出典（xnu。固定オフセットだけを読み、構造体全体は定義しない）:

- `PROC_PIDTASKINFO = 4`・`struct proc_taskinfo`（96 bytes）: `bsd/sys/proc_info.h`。
  `pti_resident_size` は offset 8 の u64（bytes）。psutil も RSS に同じ flavor を使う。
- `PROC_PIDT_SHORTBSDINFO = 13`・`struct proc_bsdshortinfo`（64 bytes）:
  `bsd/sys/proc_info.h`。`pbsi_pid` は offset 0 の u32、`pbsi_status` は offset 12 の
  u32。`arg` を非 0 にすると未回収のゾンビも見える（`bsd/kern/proc_info.c` の
  `proc_find_zombref`）。`arg=0` ではゾンビが ESRCH になるため必ず 1 を渡す。
- `SZOMB = 5`: `bsd/sys/proc.h`。
"""

from __future__ import annotations

import ctypes
import sys
import threading
from collections.abc import Callable

#: libproc の絶対パス（`find_library` で探索しない。探索経路の差し替えを避ける）。
_LIBPROC_PATH = "/usr/lib/libproc.dylib"

_PROC_PIDTASKINFO = 4
_PROC_PIDT_SHORTBSDINFO = 13
_TASKINFO_SIZE = 96
_SHORTBSDINFO_SIZE = 64
_RESIDENT_SIZE_OFFSET = 8
_BSD_PID_OFFSET = 0
_BSD_STATUS_OFFSET = 12
_SZOMB = 5
#: `proc_pidinfo` の pid 引数は C の int のため、上限を超える値は呼ばずに `None`。
_MAX_PID = 0x7FFFFFFF

_lock = threading.Lock()
_loaded = False
_proc_pidinfo: Callable[..., int] | None = None


def _load() -> Callable[..., int] | None:
    """`proc_pidinfo` を 1 回だけ読み込んでキャッシュする。失敗は `None`（これもキャッシュ）。"""
    global _loaded, _proc_pidinfo
    with _lock:
        if _loaded:
            return _proc_pidinfo
        _loaded = True
        try:
            lib = ctypes.CDLL(_LIBPROC_PATH, use_errno=True)
            func = lib.proc_pidinfo
            func.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            func.restype = ctypes.c_int
            _proc_pidinfo = func
        except (OSError, AttributeError, TypeError, ValueError):
            _proc_pidinfo = None
        return _proc_pidinfo


def _read_state(func: Callable[..., int], pid: int) -> int | None:
    """SHORTBSDINFO（`arg=1`）で `pbsi_status` を読む。読めない・pid 不一致は `None`。"""
    buf = ctypes.create_string_buffer(_SHORTBSDINFO_SIZE)
    written = func(pid, _PROC_PIDT_SHORTBSDINFO, 1, buf, _SHORTBSDINFO_SIZE)
    if written != _SHORTBSDINFO_SIZE:
        return None
    raw = buf.raw
    got_pid = int.from_bytes(raw[_BSD_PID_OFFSET : _BSD_PID_OFFSET + 4], "little")
    if got_pid != pid:
        return None
    return int.from_bytes(raw[_BSD_STATUS_OFFSET : _BSD_STATUS_OFFSET + 4], "little")


def child_status(pid: int) -> tuple[int, bool] | None:
    """`pid` の `(rss_bytes, is_zombie)` を返す。確実に読めなければ `None`。

    1. SHORTBSDINFO で状態を読む（失敗・pid 不一致は `None`）。
    2. `SZOMB` なら `(0, True)`（ゾンビの RSS は意味を持たない）。
    3. それ以外は TASKINFO で RSS を読む。失敗したら状態を 1 回だけ読み直し、
       終了と取得の競合でゾンビへ遷移していれば `(0, True)`、そうでなければ `None`。
    """
    if sys.platform != "darwin" or type(pid) is not int or pid <= 0 or pid > _MAX_PID:
        return None
    try:
        func = _load()
        if func is None:
            return None
        state = _read_state(func, pid)
        if state is None:
            return None
        if state == _SZOMB:
            return 0, True
        buf = ctypes.create_string_buffer(_TASKINFO_SIZE)
        written = func(pid, _PROC_PIDTASKINFO, 0, buf, _TASKINFO_SIZE)
        if written != _TASKINFO_SIZE:
            return (0, True) if _read_state(func, pid) == _SZOMB else None
        rss = int.from_bytes(buf.raw[_RESIDENT_SIZE_OFFSET : _RESIDENT_SIZE_OFFSET + 8], "little")
        return rss, False
    except (OSError, ctypes.ArgumentError, TypeError, ValueError, OverflowError):
        return None
