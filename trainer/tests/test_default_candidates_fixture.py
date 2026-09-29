"""既定候補の共有 fixture と選択口の一致テスト（REQ-19・TASK-19.2。issue #77）。

Rust 側（`crates/train/src/kind_resolution.rs`）は `kind` 省略時に
`fixtures/train_contract/default_candidates.json` の各 `(kind, kind_version)` を
探索候補として学習ワーカーへ依頼する。本ファイルは、その各組が選択口
（`kinds/__init__.py::_registry`）に登録済みで、`kind_defaults.json` にも既定値が
あることを機械照合する（未登録の種類を既定にすると学習ワーカーが
`unsupported_kind` で失敗するため）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fandhe_edge_trainer import kinds

# 読み込み前にファイルサイズの上限を確認する（coding-python.md）。
_MAX_FIXTURE_BYTES = 1 * 1024 * 1024
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "train_contract"


def _load(name: str) -> dict[str, Any]:
    path = _FIXTURE_DIR / name
    assert path.stat().st_size <= _MAX_FIXTURE_BYTES, f"{name} が上限を超えている"
    raw = path.read_bytes()
    raw.decode("ascii")
    data = json.loads(raw)
    assert isinstance(data, dict)
    return data


def _default_candidates() -> list[dict[str, Any]]:
    candidates = _load("default_candidates.json")["default_candidates"]
    assert isinstance(candidates, list)
    return candidates


def test_req19_default_candidates_are_registered_in_selector() -> None:
    """REQ-19: 既定候補の各 (kind, kind_version) が選択口に登録されている。"""
    registry = kinds._registry()
    for candidate in _default_candidates():
        assert candidate["kind_version"] in registry[candidate["kind"]]


def test_req19_default_candidates_have_kind_defaults() -> None:
    """REQ-19: 既定候補の各 kind が kind_defaults.json に既定値を持つ。"""
    defaults = _load("kind_defaults.json")
    for candidate in _default_candidates():
        assert candidate["kind"] in defaults


def test_req19_default_candidates_concrete_values() -> None:
    """REQ-19・TASK-19.2: 既定候補は c1・c3（この順。同率時の優先順）。"""
    assert _default_candidates() == [
        {"kind": "c1", "kind_version": 1},
        {"kind": "c3", "kind_version": 1},
    ]
