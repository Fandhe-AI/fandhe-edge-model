"""MLX（`_score_choices_mlx`。対応づけ (b) の MLX 実装）と ONNX 書き出しモデル
（`_export_ar_onnx`）の数値一致を確認する（証拠種別: テストハーネス。
`kinds/c3.py`・`tests/test_c3_mlx_onnx_parity.py` と同じ考え方）。

**この検証は「MLX 学習結果」対「書き出した ONNX グラフ」の数値一致に限られる。
Rust 側の推論ランタイム（`ort` crate）を使った実機での一致確認は推論ランタイム層
（TASK-28・REQ-28。runtime-builder 担当）の責務であり、本テストはそれを代替しない。**
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from onnx.reference import ReferenceEvaluator

from conftest import (
    ATOL_MLX_ONNX,
    LABEL_ORDER,
    TINY_AR_CONFIG,
    export_onnx_to_path,
    make_examples,
    make_request,
    train_kind,
)
from fandhe_edge_trainer.encoding import encode_bytes
from fandhe_edge_trainer.kinds.autoregressive import AutoregressiveKind, _score_choices_mlx


def _softmax(scores: np.ndarray) -> np.ndarray:
    ex = np.exp(scores - scores.max(axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)


def test_ar_mlx_forward_matches_onnx_reference(tmp_path: Path) -> None:
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(kind, make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)

    texts = ["alpha alpha beta gamma", "delta delta epsilon zeta", "short", ""]
    id_lists = [encode_bytes(t, req.max_bytes) for t in texts]
    choice_id_list = [trained.choice_ids_by_label[label] for label in LABEL_ORDER]

    mlx_scores = _score_choices_mlx(trained.model, id_lists, choice_id_list)
    assert np.all(np.isfinite(mlx_scores))
    mlx_probs = _softmax(mlx_scores)

    max_len = max(len(x) for x in id_lists)
    padded = np.zeros((len(id_lists), max_len), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        padded[i, : len(ids)] = ids

    session = ReferenceEvaluator(str(onnx_path))
    (onnx_probs,) = session.run(None, {"ids": padded})

    assert mlx_probs.argmax(axis=1).tolist() == onnx_probs.argmax(axis=1).tolist()
    diff = float(np.max(np.abs(mlx_probs - onnx_probs)))
    assert diff <= ATOL_MLX_ONNX, f"max abs diff {diff} exceeds {ATOL_MLX_ONNX}"
