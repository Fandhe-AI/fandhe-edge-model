"""`fixtures/onnx_parity/` の自己整合性を確認する（REQ-32・TASK-32.1-2・#113。テストハーネス）。

コミット済みの `c1.onnx`・`c3.onnx` と `cases.json`（MLX 内推論の予測ラベル）が互いに矛盾して
いないことだけを確認する。Rust 推論ランタイムでの全件一致は `crates/runtime/tests/onnx_parity.rs`
の責務。**再学習してバイト一致を確かめるテストは置かない**（python-ci は macOS arm64、開発機は
Linux x86_64 で、MLX CPU の浮動小数の結果が一致する保証がないため。`PROVENANCE.md` 参照）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from onnx.reference import ReferenceEvaluator

from conftest import ATOL_MLX_ONNX
from fandhe_edge_trainer.encoding import encode_bytes

_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "onnx_parity"


def _doc() -> dict:
    return json.loads((_DIR / "cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["c1", "c3"])
def test_fixture_sha256_matches_cases(kind: str) -> None:
    """REQ-32: cases.json 記載の sha256 がコミット済み ONNX と一致する。"""
    entry = _doc()["kinds"][kind]
    digest = hashlib.sha256((_DIR / entry["onnx"]).read_bytes()).hexdigest()
    assert digest == entry["onnx_sha256"]


@pytest.mark.parametrize("kind", ["c1", "c3"])
def test_reference_evaluator_matches_recorded_mlx_labels(kind: str) -> None:
    """REQ-32: ReferenceEvaluator の 1 件ずつの推論が、記録済み MLX ラベルと全件一致する。"""
    entry = _doc()["kinds"][kind]
    session = ReferenceEvaluator((_DIR / entry["onnx"]).as_posix())
    cases = entry["cases"]
    assert len(cases) >= 100
    assert len({c["mlx_label_index"] for c in cases}) >= 2
    for case in cases:
        ids = np.array([encode_bytes(case["input"], entry["max_bytes"])], dtype=np.int64)
        (probs,) = session.run(None, {"ids": ids})
        assert int(np.argmax(probs[0])) == case["mlx_label_index"], case["name"]
        diff = float(np.max(np.abs(probs[0] - np.array(case["mlx_probs"], dtype=np.float32))))
        assert diff <= ATOL_MLX_ONNX, case["name"]
