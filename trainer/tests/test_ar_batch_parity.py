"""REQ-28: 1 件ずつの推論とバッチ推論が全件一致することの機械照合
（証拠種別: テストハーネス。`kinds/c3.py`・`tests/test_c3_batch_parity.py` と
同じ考え方）。

`kinds/autoregressive.py` モジュール docstring 3 番の詰め物耐性（key マスクの
対角成分の強制許可・`CumSum` による位置計算）が実際に単独推論・バッチ推論の
一致を保っていることを、onnx.reference（純 Python の参照実装）で検証する。
空入力（`encode_bytes("") == [0]`）の行を含める（同 docstring (a) 参照。
対角成分の強制許可が無いと NaN が生じ、この一致が崩れる）。

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
    TINY_AR_CONFIG,
    export_onnx_to_path,
    make_examples,
    make_request,
    train_kind,
)
from fandhe_edge_trainer.encoding import encode_bytes
from fandhe_edge_trainer.kinds.autoregressive import AutoregressiveKind


def test_ar_single_vs_batch_inference_match(tmp_path: Path) -> None:
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(kind, make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    session = ReferenceEvaluator(str(onnx_path))

    texts = [
        "alpha alpha beta gamma extra long padded example one",
        "d",  # 最長系列より大幅に短い（入力領域の詰め物が実際に効く）
        "delta delta epsilon zeta",
        "",  # encode_bytes("") == [0]。入力領域が全て詰め物になる行
    ]
    id_lists = [encode_bytes(t, req.max_bytes) for t in texts]
    max_len = max(len(x) for x in id_lists)
    assert max_len - len(id_lists[1]) >= 4  # 詰め物が実際に効く長さの差を保証する

    batch = np.zeros((len(id_lists), max_len), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        batch[i, : len(ids)] = ids
    (probs_batch,) = session.run(None, {"ids": batch})
    assert np.all(np.isfinite(probs_batch))

    for i, ids in enumerate(id_lists):
        single_input = np.array([ids], dtype=np.int64)
        (probs_single,) = session.run(None, {"ids": single_input})
        assert np.all(np.isfinite(probs_single))
        assert probs_single[0].argmax() == probs_batch[i].argmax()
        diff = float(np.max(np.abs(probs_single[0] - probs_batch[i])))
        assert diff <= ATOL_BATCH_PARITY, (
            f"row {i}: max abs diff {diff} exceeds {ATOL_BATCH_PARITY}"
        )
