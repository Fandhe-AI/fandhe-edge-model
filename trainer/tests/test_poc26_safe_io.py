"""PoC-26 の安全なファイル読み込み（`safe_io.py`）の検査（REQ-41・TASK-41.1-5・#390。REQ-39）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.poc26.safe_io import (
    LimitExceededError,
    loads_json,
    open_regular,
    read_limited,
    sha256_hex,
)


def test_read_limited_reads_regular_file(tmp_path: Path) -> None:
    """通常ファイルは上限以内ならそのまま読める。"""
    p = tmp_path / "a.bin"
    p.write_bytes(b"abc")
    assert read_limited(p, 3, "a") == b"abc"


def test_read_limited_rejects_over_limit_with_dedicated_type(tmp_path: Path) -> None:
    """REQ-39: 上限超過は LimitExceededError（ValueError の部分型）で、文言に依存しない。"""
    p = tmp_path / "a.bin"
    p.write_bytes(b"abcd")
    with pytest.raises(LimitExceededError, match=r"a too large: 4 bytes > 3"):
        read_limited(p, 3, "a")
    assert issubclass(LimitExceededError, ValueError)


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_read_limited_rejects_non_regular(tmp_path: Path, kind: str) -> None:
    """REQ-39: 存在しない・ディレクトリ・symlink（最後の要素）は ValueError（上限超過ではない）。"""
    target = tmp_path / "t.bin"
    target.write_bytes(b"x")
    path = {"missing": tmp_path / "none", "directory": tmp_path, "symlink": tmp_path / "l"}[kind]
    if kind == "symlink":
        os.symlink(target, path)
    with pytest.raises(ValueError, match=r"cannot open a|not a regular file") as info:
        read_limited(path, 10, "a")
    assert not isinstance(info.value, LimitExceededError)


def test_open_regular_rejects_nul_in_path(tmp_path: Path) -> None:
    """パスに NUL を含むと ValueError（OSError ではなく同じ型にそろえる）。"""
    with pytest.raises(ValueError, match="cannot open a"):
        open_regular(str(tmp_path / "a\0b"), 10, "a")


@pytest.mark.parametrize(
    "raw", [b"NaN", b"[Infinity]", b"\xff", b"{", b"[" * 100000, b'{"a": 1, "a": 2}']
)
def test_loads_json_rejects_invalid(raw: bytes) -> None:
    """NaN・Infinity・不正な UTF-8・不正な JSON・深い入れ子・重複キーは ValueError("invalid x")。"""
    with pytest.raises(ValueError, match=r"^invalid x$"):
        loads_json(raw, "x")


def test_loads_json_and_sha256() -> None:
    """正常系と sha256（空列の既知値）。"""
    assert loads_json(b'{"a": [1, 2]}', "x") == {"a": [1, 2]}
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
