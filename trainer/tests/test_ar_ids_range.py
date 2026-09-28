"""`ids` の値域検査は ONNX グラフの内側では行わず、推論ランタイムのガード層
（REQ-39。パス未確定）の責務とすることの根拠を、実測で裏付けるテスト。

`kinds/autoregressive.py::_export_ar_onnx` 内の `Gather(embed_table, ...)`
直前のコメントが前提とする 2 点を確認する。

1. `VOCAB_SIZE`（=259）以上・`-VOCAB_SIZE` 未満の値は `Gather` が範囲外
   参照としてエラーで失敗する（メモリ破壊にはならない）。
2. `[-VOCAB_SIZE, -1]` の負インデックスは、ONNX の `Gather` 仕様どおり
   末尾から「黙って」wrap する（拒否されない）。

証拠種別: テストハーネス（onnx.reference。純 Python の参照実装での挙動確認。
ONNX Runtime〔`ort`〕の挙動そのものは実機未検証で、本テストが示すのは
ONNX 演算子仕様・onnx.reference の実装に基づく事実に限る）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from conftest import TINY_AR_CONFIG, export_onnx_to_path, make_examples, make_request, train_kind
from fandhe_edge_trainer.kinds.autoregressive import VOCAB_SIZE, AutoregressiveKind


@pytest.fixture
def ar_session(tmp_path: Path) -> ReferenceEvaluator:
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG, max_bytes=8)
    trained = train_kind(kind, make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    return ReferenceEvaluator(str(onnx_path))


def test_ar_ids_out_of_range_positive_raises(ar_session: ReferenceEvaluator) -> None:
    """`ids` に `VOCAB_SIZE` 以上の値を渡すと `Gather` が例外で失敗する
    （REQ-39。値域外が「黙って」別の値として扱われないことの確認）。
    """
    ids = np.array([[VOCAB_SIZE, 1, 1, 1]], dtype=np.int64)
    with pytest.raises(IndexError):
        ar_session.run(None, {"ids": ids})


def test_ar_ids_out_of_range_negative_raises(ar_session: ReferenceEvaluator) -> None:
    """`-VOCAB_SIZE` 未満の値も同様に `Gather` が例外で失敗する（REQ-39）。"""
    ids = np.array([[-VOCAB_SIZE - 1, 1, 1, 1]], dtype=np.int64)
    with pytest.raises(IndexError):
        ar_session.run(None, {"ids": ids})


def test_ar_full_model_negative_one_does_not_raise(ar_session: ReferenceEvaluator) -> None:
    """`-1`（範囲内の負インデックス）は書き出したモデル全体でも例外にならず
    有限の確率を返す（REQ-39。範囲外〔`test_ar_ids_out_of_range_negative_
    raises`〕とは異なり、「黙って」処理が進んでしまうことの確認）。
    """
    ids_neg = np.array([[-1, 1, 1, 1]], dtype=np.int64)
    (probs_neg,) = ar_session.run(None, {"ids": ids_neg})
    assert np.all(np.isfinite(probs_neg))


def _minimal_embedding_gather_model() -> onnx.ModelProto:
    """`embed_table`（`[VOCAB_SIZE, 4]`。行 `i` の全要素を値 `i` にする）への
    `Gather` だけを行う最小グラフ。`Gather` 単体の値域挙動（範囲外は例外・
    負インデックスは末尾からの wrap）を、autoregressive のフルモデルに
    混入する他の演算（マスク・位置 id 等）の影響を受けずに確認するための
    補助モデル（`test_ar_ids_negative_one_silently_wraps_to_last_vocab_row`
    が使う）。
    """
    table = numpy_helper.from_array(
        np.tile(np.arange(VOCAB_SIZE, dtype=np.float32)[:, None], (1, 4)), name="embed_table"
    )
    ids = helper.make_tensor_value_info("ids", TensorProto.INT64, ["N"])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", 4])
    node = helper.make_node("Gather", ["embed_table", "ids"], ["out"], axis=0)
    graph = helper.make_graph([node], "embed_gather", [ids], [out], initializer=[table])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def test_ar_ids_negative_one_silently_wraps_to_last_vocab_row() -> None:
    """`-1` は範囲内の負インデックスとして ONNX `Gather` 仕様どおり
    「黙って」語彙表の最終行（`VOCAB_SIZE - 1` 番目。EOS 行）へ wrap し、
    例外にならない（REQ-39。値域検査がグラフの外〔推論ランタイムの
    ガード層〕で必要な根拠を、`Gather` 単体の最小グラフで確認する）。
    """
    session = ReferenceEvaluator(_minimal_embedding_gather_model())
    (row_neg,) = session.run(None, {"ids": np.array([-1], dtype=np.int64)})
    (row_last,) = session.run(None, {"ids": np.array([VOCAB_SIZE - 1], dtype=np.int64)})
    np.testing.assert_array_equal(row_neg, row_last)
