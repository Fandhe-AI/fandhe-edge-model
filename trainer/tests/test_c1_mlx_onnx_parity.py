"""C1 の numpy 特徴量 + MLX フォワードと ONNX 書き出しモデルの数値一致を確認する
（証拠種別: テストハーネス）。

C3 の `tests/test_c3_mlx_onnx_parity.py` と同じ観点。書き出し
（`kinds/c1.py::_export_c1_onnx`）が `TfIdfVectorizer` を含むグラフを手組みして
いるため、学習側の numpy 特徴量計算（`_build_sparse_features`）+ MLX 順伝播
（`TfidfLogReg.__call__`）と同じ計算になっているかを直接突き合わせる。

**この検証は「学習側の数値計算」対「書き出した ONNX グラフ」の数値一致に限られる。
Rust 側の推論ランタイム（`ort` crate）を使った実機での一致確認は推論ランタイム層
（TASK-28・REQ-28。runtime-builder 担当）の責務であり、本テストはそれを代替しない。**
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from onnx.reference import ReferenceEvaluator

from conftest import (
    ATOL_MLX_ONNX,
    TINY_C1_CONFIG,
    export_onnx_to_path,
    make_examples,
    make_request,
    train_kind,
)
from fandhe_edge_trainer.kinds.c1 import C1Kind, _build_sparse_features, _encode_examples

#: `tests/conftest.py::TINY_C1_CONFIG` は epochs=40 だが、本テストは MLX と ONNX の
#: 数値一致のみを見るため epochs=10 の軽量な派生設定で十分（収束の良し悪しは
#: `test_c1_train.py::test_c1_train_accuracy_on_synthetic_data` 側の責務）。
_QUICK_C1_CONFIG = {**TINY_C1_CONFIG, "epochs": 10}


def _forward_and_reference_probs(tmp_path: Path, config: dict) -> tuple[np.ndarray, np.ndarray]:
    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config=config)
    examples = make_examples()
    trained = train_kind(kind, examples, req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)

    ngram_min = int(config["ngram_min"])
    ngram_max = int(config["ngram_max"])
    ids_arr, doc_lens = _encode_examples(examples, req.max_bytes, trained.resource_budget)
    n_features = trained.vocab.ns.shape[0]
    max_nnz = max(1, min(n_features, req.max_bytes * (ngram_max - ngram_min + 1)))
    idx_arr, val_arr = _build_sparse_features(
        ids_arr, doc_lens, trained.vocab, ngram_min, ngram_max, max_nnz, trained.resource_budget
    )
    trained.model.eval()
    logits = trained.model(mx.array(idx_arr), mx.array(val_arr))
    mlx_probs = np.array(mx.softmax(logits, axis=-1), dtype=np.float32)

    session = ReferenceEvaluator(str(onnx_path))
    (onnx_probs,) = session.run(None, {"ids": ids_arr.astype(np.int64)})
    return mlx_probs, onnx_probs


def test_mlx_forward_matches_onnx_reference(tmp_path: Path) -> None:
    mlx_probs, onnx_probs = _forward_and_reference_probs(tmp_path, _QUICK_C1_CONFIG)
    assert mlx_probs.argmax(axis=1).tolist() == onnx_probs.argmax(axis=1).tolist()
    diff = float(np.max(np.abs(mlx_probs - onnx_probs)))
    assert diff <= ATOL_MLX_ONNX, f"max abs diff {diff} exceeds {ATOL_MLX_ONNX}"


def test_mlx_forward_matches_onnx_reference_with_ngram_min_over_one(tmp_path: Path) -> None:
    """ngram_min > 1 の設定でも、`ngram_counts` の先頭に置く 0 件グループの
    レイアウトを含めて数値一致すること（モジュール docstring のレイアウト説明の
    実装が正しいことの回帰確認）。
    """
    config = {**_QUICK_C1_CONFIG, "ngram_min": 2, "ngram_max": 3}
    mlx_probs, onnx_probs = _forward_and_reference_probs(tmp_path, config)
    assert mlx_probs.argmax(axis=1).tolist() == onnx_probs.argmax(axis=1).tolist()
    diff = float(np.max(np.abs(mlx_probs - onnx_probs)))
    assert diff <= ATOL_MLX_ONNX, f"max abs diff {diff} exceeds {ATOL_MLX_ONNX}"
