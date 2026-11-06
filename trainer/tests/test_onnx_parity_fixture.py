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

_ROOT = Path(__file__).resolve().parents[2] / "fixtures"
#: 既定（3 選択肢・max_bytes 48）と、qwen2_lora の fixture にそろえた 8 選択肢版（#699）。
_DIRS = ["onnx_parity", "onnx_parity_8labels"]
_KINDS = ["c1", "c3", "autoregressive"]
_8LABELS = ["red", "tan", "sky", "map", "fox", "owl", "gem", "ink"]


def _doc(d: str) -> dict:
    return json.loads((_ROOT / d / "cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("d", _DIRS)
@pytest.mark.parametrize("kind", _KINDS)
def test_fixture_sha256_matches_cases(d: str, kind: str) -> None:
    """REQ-32: cases.json 記載の sha256 がコミット済み ONNX と一致する。"""
    entry = _doc(d)["kinds"][kind]
    digest = hashlib.sha256((_ROOT / d / entry["onnx"]).read_bytes()).hexdigest()
    assert digest == entry["onnx_sha256"]


@pytest.mark.parametrize("kind", _KINDS)
def test_8labels_fixture_shape(kind: str) -> None:
    """REQ-19b・#699: 8 選択肢版は qwen2_lora と同じ選択肢・max_bytes 64、出力幅 8。"""
    entry = _doc("onnx_parity_8labels")["kinds"][kind]
    assert entry["label_order"] == _8LABELS
    assert entry["max_bytes"] == 64
    session = ReferenceEvaluator((_ROOT / "onnx_parity_8labels" / entry["onnx"]).as_posix())
    (probs,) = session.run(None, {"ids": np.array([encode_bytes("x", 64)], dtype=np.int64)})
    assert probs.shape == (1, 8)


@pytest.mark.parametrize("d", _DIRS)
@pytest.mark.parametrize("kind", _KINDS)
def test_reference_evaluator_matches_recorded_mlx_labels(d: str, kind: str) -> None:
    """REQ-32: ReferenceEvaluator の 1 件ずつの推論が、記録済み MLX ラベルと全件一致する。"""
    entry = _doc(d)["kinds"][kind]
    session = ReferenceEvaluator((_ROOT / d / entry["onnx"]).as_posix())
    cases = entry["cases"]
    assert len(cases) >= 100
    assert len({c["mlx_label_index"] for c in cases}) >= 2
    for case in cases:
        ids = np.array([encode_bytes(case["input"], entry["max_bytes"])], dtype=np.int64)
        (probs,) = session.run(None, {"ids": ids})
        assert int(np.argmax(probs[0])) == case["mlx_label_index"], case["name"]
        diff = float(np.max(np.abs(probs[0] - np.array(case["mlx_probs"], dtype=np.float32))))
        assert diff <= ATOL_MLX_ONNX, case["name"]
