"""`fixtures/qwen2_parity/` の自己整合性を確認する（REQ-41・REQ-32・#434。テストハーネス）。

コミット済みのパッケージ（`artifact.json`・`model.onnx`・shard・`tokenizer.json`）と `cases.json`
（MLX 内推論の予測ラベル・ReferenceEvaluator の確率）が互いに矛盾していないことだけを確認する。
Rust 推論ランタイムでの全件一致は `crates/runtime/tests/qwen2_parity.rs` の責務。
**再学習してバイト一致を確かめるテストは置かない**（python-ci は macOS arm64、開発機は
Linux x86_64 で、MLX CPU の浮動小数の結果が一致する保証がないため。`PROVENANCE.md` 参照）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx

from conftest import ATOL_MLX_ONNX
from fandhe_edge_trainer.qwen2_tokenizer import Qwen2Tokenizer
from test_qwen2_onnx_export import evaluator

_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "qwen2_parity"


def _sha(name: str) -> str:
    return hashlib.sha256((_DIR / name).read_bytes()).hexdigest()


def _art() -> dict:
    return json.loads((_DIR / "artifact.json").read_text(encoding="utf-8"))


def test_package_hashes_match_artifact() -> None:
    """REQ-39・REQ-41: artifact.json 記載の sha256・サイズが各ファイルと一致する。"""
    art = _art()
    assert art["onnx_sha256"] == _sha(art["onnx_file"])
    assert art["tokenizer_sha256"] == _sha(art["tokenizer_file"])
    assert art["external_data"], "fixture must exercise external data"
    for s in art["external_data"]:
        assert s["sha256"] == _sha(s["location"])
        assert s["size_bytes"] == (_DIR / s["location"]).stat().st_size


def test_reference_evaluator_matches_recorded_cases() -> None:
    """REQ-41: ReferenceEvaluator の 1 件ずつの推論が、記録済みのラベル・確率と全件一致する。"""
    art = _art()
    cases = json.loads((_DIR / "cases.json").read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 100
    assert len({c["mlx_label_index"] for c in cases}) >= 2
    tok = Qwen2Tokenizer.from_bytes((_DIR / art["tokenizer_file"]).read_bytes())
    ev = evaluator(onnx.load(str(_DIR / art["onnx_file"]), load_external_data=True))
    system = art["scoring"]["system_prompt"]
    for c in cases:
        ids = tok.build_chat_ids(system, c["input"], add_generation_prompt=True)
        (probs,) = ev.run(None, {"ids": np.array([ids], dtype=np.int64)})
        assert int(np.argmax(probs[0])) == c["mlx_label_index"], c["name"]
        diff = float(np.max(np.abs(probs[0] - np.array(c["ref_probs"], dtype=np.float32))))
        assert diff <= ATOL_MLX_ONNX, c["name"]
