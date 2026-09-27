"""REQ-28: 1 件ずつの推論とバッチ推論が全件一致することの機械照合（証拠種別: テストハーネス）。

PoC-16（`docs/spec/03-poc/core-cli-vertical-slice/`）の診断
（`scripts/diagnose_batch_dependence.py`）で、MLX 側のバッチ推論（PoC-10 の
`ByteCNN`・`predict_probs`）が単独推論と実際に食い違うことが確認されている
（`kinds/c3.py` の docstring 2 番）。本テストは、その教訓を踏まえて ONNX 書き出し側
（`_export_c3_onnx`）が単独推論とバッチ推論で一致することを onnx.reference（純
Python の参照実装）で検証する。バッチには最長系列より 7（最大カーネル幅）以上
短い系列を含め、詰め物マスクが実際に効くようにする。

**この検証は ONNX グラフ自体の契約確認に限られる。Rust 側の推論ランタイム
（`ort` crate）を使った実機・実装での一致確認は推論ランタイム層
（TASK-28・REQ-28。runtime-builder 担当）の責務であり、本テストはそれを代替しない。**
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from onnx.reference import ReferenceEvaluator

from conftest import ATOL_BATCH_PARITY, export_onnx_to_path, make_examples, make_request, train_c3
from fandhe_edge_trainer.encoding import encode_bytes
from fandhe_edge_trainer.kinds.c3 import C3Kind


def test_single_vs_batch_inference_match(tmp_path: Path) -> None:
    kind = C3Kind()
    req = make_request(tmp_path)
    trained = train_c3(kind, make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    session = ReferenceEvaluator(str(onnx_path))

    texts = [
        "alpha alpha beta gamma extra long padded example one",
        "d",  # 最長系列より 7 バイト以上短い（マスクが実際に効く）
        "delta delta epsilon zeta",
    ]
    id_lists = [encode_bytes(t, req.max_bytes) for t in texts]
    max_len = max(len(x) for x in id_lists)
    assert max_len - len(id_lists[1]) >= 7  # カーネル幅の最大値以上の長さの差を保証する

    batch = np.zeros((len(id_lists), max_len), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        batch[i, : len(ids)] = ids
    (probs_batch,) = session.run(None, {"ids": batch})

    for i, ids in enumerate(id_lists):
        single_input = np.array([ids], dtype=np.int64)
        (probs_single,) = session.run(None, {"ids": single_input})
        assert probs_single[0].argmax() == probs_batch[i].argmax()
        diff = float(np.max(np.abs(probs_single[0] - probs_batch[i])))
        assert diff <= ATOL_BATCH_PARITY, (
            f"row {i}: max abs diff {diff} exceeds {ATOL_BATCH_PARITY}"
        )
