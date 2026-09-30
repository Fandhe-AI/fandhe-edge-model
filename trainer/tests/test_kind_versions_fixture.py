"""`kind` ごとの許可 `kind_version` と共有 fixture の一致テスト（REQ-39・TASK-39.6-1）。

Rust 側ガード層（`crates/guard/src/kind_version.rs::KindVersionAllowlist::supported`）は
`fixtures/train_contract/kind_versions.json` と `crates/train/tests/
guard_kind_version_allowlist_sync.rs` で照合される。本ファイルは、その fixture が学習
ワーカー側の正本（`kinds/__init__.py::_registry`）と食い違っていないことを機械照合する。
"""

from __future__ import annotations

import json
from pathlib import Path

from fandhe_edge_trainer import kinds

# 読み込み前にファイルサイズの上限を確認する（coding-python.md）。
_MAX_FIXTURE_BYTES = 1 * 1024 * 1024
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _REPO_ROOT / "fixtures" / "train_contract" / "kind_versions.json"
_RESERVED_META_KEY = "_meta"


def test_req39_kind_versions_fixture_matches_registry() -> None:
    """REQ-39・TASK-39.6-1: fixture の kind ごとの版の集合が `_registry` と完全一致する。"""
    assert _FIXTURE_PATH.is_file(), f"共有 fixture が見つからない: {_FIXTURE_PATH}"
    size = _FIXTURE_PATH.stat().st_size
    assert size <= _MAX_FIXTURE_BYTES, f"kind_versions.json が上限を超えている: {size}"
    data = json.loads(_FIXTURE_PATH.read_bytes())
    data.pop(_RESERVED_META_KEY, None)
    fixture = {kind: sorted(versions) for kind, versions in data.items()}
    registry = {kind: sorted(versions) for kind, versions in kinds._registry().items()}
    assert fixture == registry
