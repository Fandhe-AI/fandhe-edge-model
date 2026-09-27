"""REQ-28: C1（TF-IDF + ロジスティック回帰）の 1 件ずつの推論とバッチ推論が
全件一致することの機械照合（証拠種別: テストハーネス）。

C3 の `tests/test_c3_batch_parity.py` と同じ観点。C1 は `TfIdfVectorizer` が
行（文書）ごとに独立して n-gram 照合を行い、かつ語彙が詰め物（id=0）を含む
n-gram を一切持たない（`kinds/c1.py` モジュール docstring の不変条件）ため、
設計上バッチ・単独の一致は成り立つはずだが、それを ONNX グラフ自体の挙動として
実際に確認する（onnx.reference の純 Python 参照実装で検証）。

**この検証は ONNX グラフ自体の契約確認に限られる。Rust 側の推論ランタイム
（`ort` crate）を使った実機・実装での一致確認は推論ランタイム層
（TASK-28・REQ-28。runtime-builder 担当）の責務であり、本テストはそれを代替しない。**
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from onnx.reference import ReferenceEvaluator

from conftest import (
    ATOL_BATCH_PARITY,
    TINY_C1_CONFIG,
    export_onnx_to_path,
    make_examples,
    make_request,
    train_kind,
)
from fandhe_edge_trainer.encoding import encode_bytes
from fandhe_edge_trainer.kinds.c1 import C1Kind


def test_c1_single_vs_batch_inference_match(tmp_path: Path) -> None:
    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config=TINY_C1_CONFIG)
    trained = train_kind(kind, make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    session = ReferenceEvaluator(str(onnx_path))

    texts = [
        "alpha alpha beta gamma extra long padded example one",
        "d",  # 最長系列より大幅に短い（詰め物が実際に入る）
        "delta delta epsilon zeta",
        "",  # 空文字列は encode_bytes により [0]（詰め物のみ）になる
    ]
    id_lists = [encode_bytes(t, req.max_bytes) for t in texts]
    max_len = max(len(x) for x in id_lists)
    assert max_len - len(id_lists[1]) >= 7

    batch = np.zeros((len(id_lists), max_len), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        batch[i, : len(ids)] = ids
    (probs_batch,) = session.run(None, {"ids": batch})

    for i, ids in enumerate(id_lists):
        single_input = np.array([ids], dtype=np.int64)
        (probs_single,) = session.run(None, {"ids": single_input})
        diff = float(np.max(np.abs(probs_single[0] - probs_batch[i])))
        assert diff <= ATOL_BATCH_PARITY, (
            f"row {i}: max abs diff {diff} exceeds {ATOL_BATCH_PARITY}"
        )


def test_c1_single_vs_batch_inference_match_with_ngram_min_over_one(tmp_path: Path) -> None:
    """ngram_min > 1（詰め物なしサイズのグループが `ngram_counts` に混じる）でも
    バッチ・単独一致が崩れないこと。
    """
    kind = C1Kind()
    req = make_request(
        tmp_path, kind="c1", config={**TINY_C1_CONFIG, "ngram_min": 2, "ngram_max": 3}
    )
    trained = train_kind(kind, make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    session = ReferenceEvaluator(str(onnx_path))

    texts = ["alpha alpha beta gamma", "d", "delta delta epsilon zeta"]
    id_lists = [encode_bytes(t, req.max_bytes) for t in texts]
    max_len = max(len(x) for x in id_lists)
    batch = np.zeros((len(id_lists), max_len), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        batch[i, : len(ids)] = ids
    (probs_batch,) = session.run(None, {"ids": batch})

    for i, ids in enumerate(id_lists):
        single_input = np.array([ids], dtype=np.int64)
        (probs_single,) = session.run(None, {"ids": single_input})
        diff = float(np.max(np.abs(probs_single[0] - probs_batch[i])))
        assert diff <= ATOL_BATCH_PARITY, (
            f"row {i}: max abs diff {diff} exceeds {ATOL_BATCH_PARITY}"
        )
