"""バイトエンコードのテスト（本リポ内 SSOT との照合契約）。

`fixtures/preprocess/byte_encoding_vectors.json`（`normalize_input`+`encode_bytes` の
共有ゴールデンベクタ。本リポ内 SSOT）を読み込み、Python 側実装がこれと一致することを
確認する。REQ-28 の推論一致契約は学習・推論の両方が同じトークン化をすることを
前提とするため、この一致は不可欠。

将来仕様（本テストのスコープ外。TASK-15.2 以降）: Rust workspace 作成後、推論
ランタイム crate（`preprocess.rs` 由来）が同じ `fixtures/preprocess/
byte_encoding_vectors.json` を読み込んで一致検証するテストを追加する契約とする
（現状は `crates/` 未作成のため Rust 側のテストは書けない。Chore #10 Deliverable B）。
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from fandhe_edge_trainer.encoding import encode_bytes, normalize_input

# 読み込み前にファイルサイズの上限を確認する（外部入力読み込みの作法。coding-python.md）。
# ローカル固定 fixture だが、想定外に巨大化した場合に無制限アロケーションへ繋げない。
_MAX_FIXTURE_BYTES = 1024 * 1024

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "preprocess" / "byte_encoding_vectors.json"
)


def _load_fixture() -> dict[str, Any]:
    size = _FIXTURE_PATH.stat().st_size
    assert size <= _MAX_FIXTURE_BYTES, (
        f"byte_encoding_vectors.json が上限（{_MAX_FIXTURE_BYTES} バイト）を超えている: {size}"
    )
    raw = _FIXTURE_PATH.read_bytes()
    # ベクタファイルは Rust 側実装との将来照合も見据え ASCII のみで書く（非 ASCII は \uXXXX）。
    raw.decode("ascii")
    return json.loads(raw)


_FIXTURE = _load_fixture()
_VECTORS: list[dict[str, Any]] = _FIXTURE["vectors"]


def test_fixture_is_ascii_only() -> None:
    # 上の _load_fixture() 内 decode("ascii") が既に検証しているが、
    # 「テストとして」明示的に確認する（fixture 破損時にここで検知する）。
    _FIXTURE_PATH.read_bytes().decode("ascii")


def test_fixture_unicode_version_matches_runtime() -> None:
    # fixture は特定の Unicode 版で手動導出した値。Python バージョン更新等で
    # unicodedata の版が変わると NFKC の結果がずれうるため、実行環境の版と
    # 記録済みの版が一致することを確認する（不一致ならベクタの再導出が必要）。
    assert _FIXTURE["_meta"]["unicode_version"] == unicodedata.unidata_version


@pytest.mark.parametrize("vector", _VECTORS, ids=[v["name"] for v in _VECTORS])
def test_normalize_input_matches_vector(vector: dict[str, Any]) -> None:
    assert normalize_input(vector["input"]) == vector["normalized"], vector["description"]


@pytest.mark.parametrize("vector", _VECTORS, ids=[v["name"] for v in _VECTORS])
def test_encode_bytes_matches_vector(vector: dict[str, Any]) -> None:
    assert encode_bytes(vector["input"], vector["max_bytes"]) == vector["ids"], vector[
        "description"
    ]
