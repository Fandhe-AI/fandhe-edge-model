"""tokenizer 入力表現のゴールデンベクタ形式の読み込みテスト（REQ-41・TASK-41.3-3・#404）。

`fixtures/preprocess/tokenizer_vectors.json` の形式のみを検査する（証拠種別: テストハーネス）。
Rust との全件一致は `crates/runtime/tests/tokenizer_golden.rs`（TASK-41.4-5・#413）。Rust 側
`crates/runtime/tests/tokenizer_vectors_format.rs` と同じ項目・同じ順で形式を検査し、合成 fixture
（`tokenizer_*_synthetic.json`）が生成器の出力と一致すること（古くないこと）も確かめる。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import gen_tokenizer_vectors as gen

# 読み込み前にサイズ上限を確認する（coding-python.md）。
_MAX_FIXTURE_BYTES = 1024 * 1024
_MAX_VECTORS = 1000
# parents[0]=trainer/tests, parents[1]=trainer, parents[2]=リポジトリ直下。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _REPO_ROOT / "fixtures" / "preprocess" / "tokenizer_vectors.json"
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _non_empty_str(obj: Any, key: str) -> str:
    value = obj.get(key) if isinstance(obj, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} missing or empty")
    return value


def check_format(raw: bytes) -> int:
    """形式を検査して件数を返す。違反は ValueError。"""
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as e:
        raise ValueError("non-ascii bytes") from e
    root = json.loads(text)
    meta = root.get("_meta")
    if not isinstance(meta, dict):
        raise ValueError("_meta missing")
    if _non_empty_str(meta, "input_representation") != "tokenizer":
        raise ValueError("input_representation must be tokenizer")
    if not _SHA_RE.fullmatch(_non_empty_str(meta, "tokenizer_json_sha256")):
        raise ValueError("tokenizer_json_sha256 must be lowercase hex64")
    _non_empty_str(meta, "reference")
    if not _DATE_RE.fullmatch(_non_empty_str(meta, "generated_on")):
        raise ValueError("generated_on must be YYYY-MM-DD")
    vectors = root.get("vectors")
    if not isinstance(vectors, list) or not 0 < len(vectors) <= _MAX_VECTORS:
        raise ValueError("vector count out of range")
    names: set[str] = set()
    for v in vectors:
        name = _non_empty_str(v, "name")
        if name in names:
            raise ValueError(f"duplicate name {name}")
        names.add(name)
        if not isinstance(v.get("input"), str):
            raise ValueError(f"{name}: input must be string")
        ids = v.get("ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{name}: ids missing or empty")
        for i in ids:
            if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i <= 0xFFFFFFFF:
                raise ValueError(f"{name}: id out of u32 range")
    return len(vectors)


def _load() -> bytes:
    assert _FIXTURE_PATH.is_file(), f"fixture not found: {_FIXTURE_PATH}"
    assert _FIXTURE_PATH.stat().st_size <= _MAX_FIXTURE_BYTES
    return _FIXTURE_PATH.read_bytes()


def test_req41_tokenizer_vectors_format_ok() -> None:
    raw = _load()
    assert check_format(raw) == 23
    first = json.loads(raw)["vectors"][0]
    assert first["name"] == "poc26_01"
    assert len(first["ids"]) == 6
    assert first["ids"][0] == 133165


_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"


@pytest.mark.parametrize(
    ("src", "dst", "message"),
    [
        (_SHA, _SHA.upper(), "tokenizer_json_sha256 must be lowercase hex64"),
        ("poc26_02", "poc26_01", "duplicate name poc26_01"),
        ("133165", "-1", "poc26_01: id out of u32 range"),
        ('"tokenizer"', '"byte"', "input_representation must be tokenizer"),
    ],
)
def test_req41_tokenizer_vectors_format_rejects_violations(
    src: str, dst: str, message: str
) -> None:
    text = _load().decode("ascii")
    assert src in text
    with pytest.raises(ValueError, match=re.escape(message)):
        check_format(text.replace(src, dst, 1).encode("ascii"))


def test_req41_synthetic_vectors_format_ok() -> None:
    raw = gen.VECTORS_PATH.read_bytes()
    assert check_format(raw) == 189


def test_req41_synthetic_fixtures_are_up_to_date() -> None:
    """生成物が古くない（再生成すると差分が出ない）。"""
    assert gen.TOKENIZER_PATH.read_text("utf-8") == gen.render_tokenizer()
    assert gen.VECTORS_PATH.read_text("utf-8") == gen.render_vectors()
