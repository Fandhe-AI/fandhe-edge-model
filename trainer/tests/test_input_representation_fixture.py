"""入力表現の宣言を Rust 側と同じに解釈することの共有 fixture 照合
（REQ-41・TASK-41.3-2・#403。証拠種別: テストハーネス）。

対の Rust テストは `crates/core/tests/input_representation_fixture.rs`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fandhe_edge_trainer import artifact

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "input_representation"
    / "declaration_cases.json"
)
_MAX_FIXTURE_BYTES = 1024 * 1024


def _load() -> dict[str, Any]:
    assert _FIXTURE.stat().st_size <= _MAX_FIXTURE_BYTES
    doc: dict[str, Any] = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return doc


def test_req41_declaration_cases_match_fixture() -> None:
    doc = _load()
    assert artifact.DEFAULT_INPUT_REPRESENTATION == doc["default"]
    assert list(artifact.INPUT_REPRESENTATIONS) == doc["accepted_values"]
    assert len(doc["cases"]) >= 10
    for c in doc["cases"]:
        art = dict(doc["base_artifact"])
        if c["present"]:
            art["input_representation"] = c["value"]
        got = artifact.parse_input_representation(art)
        assert (got if got is not None else "reject") == c["expect"], c["name"]


def test_req41_build_artifact_is_interpreted_as_byte() -> None:
    art = artifact.build_artifact(
        kind="c3",
        kind_version=1,
        config={},
        label_order=["a", "b"],
        output_type="choice",
        max_bytes=64,
        candidate_label="c3",
        onnx_sha256="ab" * 32,
    )
    assert artifact.parse_input_representation(art) == "byte"
