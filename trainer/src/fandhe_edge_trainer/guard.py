"""経路の閉じ込め（REQ-39 ガード層・PoC-20）。fd ベースの走査で TOCTOU を排除する。

**多層防御**: 経路の閉じ込めの一次防御は Rust 側 CLI（呼び出し元）が担う設計だが、
本学習ワーカーは単独プロセスとしても起動されうるため、ワーカー自身でも `root`
（呼び出し元が指定する許可された作業ルート）配下への閉じ込めを検証する。

**設計変更の経緯（fd ベースへの移行）**: 当初は `os.path.realpath` で解決した
絶対パスを文字列として扱い、`os.path.commonpath` で「root 配下か」を判定していた。
しかしこの方式には TOCTOU（Time-Of-Check to Time-Of-Use）が残る: 検証（realpath
解決）から実際の使用（`os.mkdir`・ファイルの読み書き）までの間に、経路の途中の
ディレクトリが別プロセスによってシンボリックリンクへ差し替えられると、検証した
経路と実際に使われる経路が別物になりうる。

これを避けるため、本モジュールは**検証と使用を同じディレクトリ実体（fd）へ
束縛する**: `root` を 1 度だけ `os.open(..., O_DIRECTORY)` で開き、そのファイル
記述子（fd）を経由してのみ、相対パスの各構成要素を `openat` 相当
（`os.open(part, O_DIRECTORY | O_NOFOLLOW, dir_fd=...)`）で 1 段ずつ辿る。
`O_NOFOLLOW` により、途中の構成要素の**いずれか 1 つでもシンボリックリンクなら
即座に拒否**する（`symlink_not_allowed`）。これは以前の実装（root 配下を指す
シンボリックリンクは許可していた）より厳格だが、シンボリックリンクの差し替え
自体を TOCTOU の攻撃面として完全に塞ぐには、経路の途中に一切シンボリックリンクを
許さないことが最も単純で確実な方針である。

最終的に得られる `parent_fd`（最終コンポーネントの親ディレクトリの fd）は、
呼び出し側がその後の `os.open`/`os.mkdir`/`os.stat`/`os.rename` をすべて
`dir_fd=parent_fd` で行うことで、検証時点の実体に対してのみ操作が行われる
ことを保証する（`parent_fd` を握っている限り、経路文字列を再解決する必要が
一切ない）。

fd を経由した操作（`dir_fd` 引数）が使えない環境では、経路ベース呼び出しへ
黙ってフォールバックせず fail-closed で拒否する（`_assert_dir_fd_support`）。
"""

from __future__ import annotations

import errno
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .errors import WorkerError
from .exitcode import ExitCode

#: `dir_fd` 引数を要する syscall のうち、本モジュールが依存するもの。
_REQUIRED_DIR_FD_FUNCS = (os.open, os.mkdir, os.stat, os.rmdir, os.unlink, os.rename)


def _assert_dir_fd_support() -> None:
    """`dir_fd`・`follow_symlinks` を要する呼び出しがこの環境で使えることを確認する。

    使えない関数があれば、経路ベース呼び出しへ黙ってフォールバックせず
    `WorkerError`（runtime_error・exit 70）で拒否する（fail-closed。
    経路の閉じ込めの正しさより「動くこと」を優先しない）。
    """
    missing = [f.__name__ for f in _REQUIRED_DIR_FD_FUNCS if f not in os.supports_dir_fd]
    if os.stat not in os.supports_follow_symlinks:
        missing.append("stat(follow_symlinks)")
    if missing:
        raise WorkerError(
            "runtime_error",
            f"this platform lacks dir_fd/follow_symlinks support for: {missing}"
            " (required for TOCTOU-free path confinement)",
            ExitCode.RUNTIME_ERROR,
        )


def _invalid_path(message: str) -> WorkerError:
    return WorkerError("invalid_path", message, ExitCode.INVALID_INPUT)


def _symlink_not_allowed(message: str) -> WorkerError:
    return WorkerError("symlink_not_allowed", message, ExitCode.INVALID_INPUT)


class RootHandle:
    """`root`（許可された作業ルート）を一度だけ開いたディレクトリ fd で保持する。

    1 リクエストの処理中（`train_path`・`out_dir` の解決から実際の読み書きまで）
    ずっと開いたまま保持し、`close()`（`with` 文でも可）で明示的に閉じる。
    fd を経由して保持することで、`root` 自体がリクエスト処理中に削除・置換
    されても、この fd が指す実体（inode）は変わらない（POSIX の性質）。
    """

    __slots__ = ("_closed", "fd", "root_real")

    def __init__(self, root_real: Path, fd: int) -> None:
        self.root_real = root_real
        self.fd = fd
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                os.close(self.fd)
            except OSError:
                pass

    def __enter__(self) -> RootHandle:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def resolve_root(root_raw: object) -> RootHandle:
    """`root` を検証し、開いたディレクトリ fd（`RootHandle`）として返す。

    存在しない・ディレクトリでない場合は拒否する。呼び出し側は使い終わったら
    必ず `close()` する（`TrainRequest.close_resources()` 参照）。
    """
    _assert_dir_fd_support()
    if not isinstance(root_raw, str) or not root_raw:
        raise _invalid_path("root must be a non-empty string")
    if "\x00" in root_raw:
        raise _invalid_path("root must not contain NUL")
    if not os.path.isabs(root_raw):
        raise _invalid_path("root must be an absolute path")
    real = Path(os.path.realpath(root_raw))
    try:
        fd = os.open(real, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as e:
        raise _invalid_path(f"root not openable as a directory: {type(e).__name__}") from e
    return RootHandle(real, fd)


def _check_syntax(rel_raw: object, field_name: str) -> PurePosixPath:
    if not isinstance(rel_raw, str) or not rel_raw:
        raise _invalid_path(f"{field_name} must be a non-empty string")
    if "\x00" in rel_raw:
        raise _invalid_path(f"{field_name} must not contain NUL")
    if os.path.isabs(rel_raw):
        raise _invalid_path(f"{field_name} must be a relative path")
    rel = PurePosixPath(rel_raw)
    if not rel.parts:
        raise _invalid_path(f"{field_name} must not be empty")
    if ".." in rel.parts:
        raise _invalid_path(f"{field_name} must not contain '..' components")
    return rel


def _walk_parent(root_fd: int, parent_parts: tuple[str, ...], field_name: str) -> int:
    """`root_fd` から `parent_parts` を 1 段ずつ `O_NOFOLLOW` 付きで辿り、
    最後のディレクトリの fd を返す（`parent_parts` が空なら `root_fd` の複製）。

    途中の構成要素がシンボリックリンクなら `ELOOP` を検出して
    `symlink_not_allowed` を送出する。存在しない・ディレクトリでない場合は
    `invalid_path`。いずれの場合も、それまでに開いた中間 fd はすべて閉じる。

    **`O_DIRECTORY` を `open` の呼び出し自体には付けない**: macOS では
    `O_NOFOLLOW | O_DIRECTORY` を同時に指定してシンボリックリンクを開こうとすると
    `ELOOP` ではなく `ENOTDIR` が返る（実機で確認済み。Linux では `ELOOP` になる
    という記述と食い違う）。`O_NOFOLLOW` だけを付ければシンボリックリンクの検出
    （`ELOOP`）は両 OS で共通して信頼できるため、まず `O_NOFOLLOW` だけで開き、
    その後 `fstat` で `S_ISDIR` を別途確認する 2 段構成にする。
    """
    cur = os.dup(root_fd)
    for part in parent_parts:
        try:
            nxt = os.open(part, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=cur)
        except OSError as e:
            os.close(cur)
            if e.errno == errno.ELOOP:
                raise _symlink_not_allowed(f"{field_name} traverses a symlink component") from e
            raise _invalid_path(
                f"{field_name} parent does not exist or is not a directory: {type(e).__name__}"
            ) from e
        os.close(cur)
        try:
            st = os.fstat(nxt)
        except OSError as e:
            os.close(nxt)
            raise _invalid_path(f"{field_name} parent not stat-able: {type(e).__name__}") from e
        if not stat.S_ISDIR(st.st_mode):
            os.close(nxt)
            raise _invalid_path(f"{field_name} parent is not a directory")
        cur = nxt
    return cur


@dataclass
class ConfinedEntry:
    """root 配下へ dir_fd で閉じ込めた「親ディレクトリ fd + 最終コンポーネント名」。

    最終コンポーネント自体（ファイル・ディレクトリいずれも）はこの時点では
    開いていない。呼び出し側が `parent_fd` を使って `os.open`/`os.mkdir`/
    `os.stat`/`os.rename` を dir_fd 相対で行う。

    `display` はログ・エラーメッセージ用の参考パスに過ぎず、実体の存在・
    同一性を保証しない（実体の保証は `parent_fd` が担う）。
    """

    parent_fd: int
    name: str
    display: Path
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        """`parent_fd` を閉じる（複数回呼んでも安全。冪等）。"""
        if not self._closed:
            self._closed = True
            try:
                os.close(self.parent_fd)
            except OSError:
                pass


def confine(root_handle: RootHandle, rel_raw: object, field_name: str) -> ConfinedEntry:
    """`root_handle` 配下の相対パス `rel_raw` を、シンボリックリンクを一切
    追跡せずに dir_fd で閉じ込める（TOCTOU 対策。本モジュールの docstring 参照）。
    """
    rel = _check_syntax(rel_raw, field_name)
    parent_fd = _walk_parent(root_handle.fd, rel.parts[:-1], field_name)
    name = rel.parts[-1]
    display = root_handle.root_real.joinpath(*rel.parts)
    return ConfinedEntry(parent_fd=parent_fd, name=name, display=display)
