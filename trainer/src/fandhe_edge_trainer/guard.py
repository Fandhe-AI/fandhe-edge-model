"""経路の閉じ込め（REQ-39 ガード層・PoC-20）。

**多層防御**: 経路の閉じ込めの一次防御は Rust 側 CLI（呼び出し元）が担う設計だが、
本学習ワーカーは単独プロセスとしても起動されうる（テスト・手動デバッグ・将来の
呼び出し経路の変更を含む）ため、ワーカー自身でも `root`（呼び出し元が指定する
許可された作業ルート）配下への閉じ込めを検証する。Rust 側ガード層の存在を
前提に本ワーカー側の検証を省略しない（「Rust 側が検証済みのはず」という
呼び出し元契約への一方的な依存をやめる）。

`safe_join(root_real, rel_raw, field_name)` は次を行う:
1. `rel_raw` の構文検査（文字列であること・絶対パスでない・空でない・NUL を
   含まない・`..` 構成要素を含まない）。違反は `invalid_path`（exit 64）。
2. `rel_raw` の最終コンポーネントを除いた親ディレクトリを `os.path.realpath` で
   解決する（シンボリックリンクをすべて辿った実体のパスにする）。親が存在
   しない・ディレクトリでない場合は `invalid_path`。
3. 解決した親が `root_real`（あらかじめ `resolve_root` で解決済み）の配下に
   あることを `os.path.commonpath` で確認する。途中のどの構成要素がシンボリック
   リンクであっても、実体が root の外を指していれば拒否できる。違反は
   `path_outside_root`（exit 64）。
4. 最終コンポーネント自体は実体を解決せず（存在するとは限らないため。
   `out_dir` は呼び出し時点で未作成が前提）、解決済みの親へ文字列として
   結合するだけにする。

**残存する TOCTOU**: `safe_join` が返す絶対パスを実際に開く/作成するまでの間に、
別プロセスが最終コンポーネントをシンボリックリンクへ差し替える競合は理論上
残る（`root_real` を解決してから実際の I/O までの間隙）。学習データの読み込みは
最終コンポーネントを `O_NOFOLLOW` で開くことでこの残存リスクを軽減する
（`contract.py::_open_regular_file` の `nofollow` 引数）。この残存 TOCTOU の
完全な排除・経路検証の一次防御は Rust 側ガード層（REQ-39）の責務であり、
本モジュールの検証は多層防御の一枚に過ぎない。
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from .errors import WorkerError
from .exitcode import ExitCode


def _invalid_path(message: str) -> WorkerError:
    return WorkerError("invalid_path", message, ExitCode.INVALID_INPUT)


def _outside_root(message: str) -> WorkerError:
    return WorkerError("path_outside_root", message, ExitCode.INVALID_INPUT)


def resolve_root(root_raw: object) -> Path:
    """`root`（呼び出し元が指定する許可された作業ルート）を検証し、
    `realpath` で解決したうえで返す。存在しない・ディレクトリでない場合は拒否する。
    """
    if not isinstance(root_raw, str) or not root_raw:
        raise _invalid_path("root must be a non-empty string")
    if "\x00" in root_raw:
        raise _invalid_path("root must not contain NUL")
    if not os.path.isabs(root_raw):
        raise _invalid_path("root must be an absolute path")
    real = Path(os.path.realpath(root_raw))
    if not real.is_dir():
        raise _invalid_path("root does not exist or is not a directory")
    return real


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


def safe_join(root_real: Path, rel_raw: object, field_name: str) -> Path:
    """`root_real`（解決済み）配下の相対パス `rel_raw` を検証・解決する。

    戻り値は「解決済みの親ディレクトリ」+「未解決の最終コンポーネント名」
    （最終コンポーネント自体の存在は前提としない。存在確認・open 時の
    `O_NOFOLLOW` は呼び出し側の責務）。
    """
    rel = _check_syntax(rel_raw, field_name)
    parent_rel = rel.parent
    name = rel.name
    parent_raw = str(root_real) if str(parent_rel) == "." else str(root_real / parent_rel)
    parent_real = Path(os.path.realpath(parent_raw))
    if not parent_real.is_dir():
        raise _invalid_path(f"{field_name} parent does not exist or is not a directory")
    common = os.path.commonpath([str(root_real), str(parent_real)])
    if common != str(root_real):
        raise _outside_root(f"{field_name} resolves outside root")
    return parent_real / name
