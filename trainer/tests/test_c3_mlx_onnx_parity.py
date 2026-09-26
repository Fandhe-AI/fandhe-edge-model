"""MLX の学習時フォワードと ONNX 書き出しモデルの数値一致を確認する（証拠種別: テストハーネス）。

書き出し（`kinds/c3.py::_export_c3_onnx`）がグラフを手組みしているため、MLX 側の
`ByteCNN.__call__` と同じ計算になっているかを直接突き合わせる。

**この検証は「MLX 学習結果」対「書き出した ONNX グラフ」の数値一致に限られる。
Rust 側の推論ランタイム（`ort` crate）を使った実機での一致確認は推論ランタイム層
（TASK-28・REQ-28。runtime-builder 担当）の責務であり、本テストはそれを代替しない。**
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from onnx.reference import ReferenceEvaluator

from conftest import ATOL_MLX_ONNX, export_onnx_to_path, make_examples, make_request
from fandhe_edge_trainer.encoding import encode_bytes
from fandhe_edge_trainer.kinds.c3 import C3Kind, _batchify


def test_mlx_forward_matches_onnx_reference(tmp_path: Path) -> None:
    kind = C3Kind()
    req = make_request(tmp_path)
    trained = kind.train(make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)

    texts = ["alpha alpha beta gamma", "delta delta epsilon zeta", "short"]
    id_lists = [encode_bytes(t, req.max_bytes) for t in texts]
    max_len = max(len(x) for x in id_lists)
    padded = np.zeros((len(id_lists), max_len), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        padded[i, : len(ids)] = ids

    x, mask = _batchify(id_lists)
    trained.model.eval()
    mlx_logits = trained.model(x, mask)
    mlx_probs = np.array(mx.softmax(mlx_logits, axis=-1), dtype=np.float32)

    session = ReferenceEvaluator(str(onnx_path))
    (onnx_probs,) = session.run(None, {"ids": padded})

    assert mlx_probs.argmax(axis=1).tolist() == onnx_probs.argmax(axis=1).tolist()
    diff = float(np.max(np.abs(mlx_probs - onnx_probs)))
    assert diff <= ATOL_MLX_ONNX, f"max abs diff {diff} exceeds {ATOL_MLX_ONNX}"
