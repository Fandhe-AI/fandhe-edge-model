"""`ids` の値域・`N`（バッチ件数）の上限を ONNX グラフの内側で fail-closed に
検査することを実測で確認するテスト（REQ-39・REQ-28。オーナー判断
2026-09-28・PR #222 レビュー指摘への対応）。

以前は `ids` の値域・`N` の上限検査を ONNX グラフの外（推論ランタイムの
ガード層。REQ-39。パス未確定）の責務としていたが、ガード層が未実装のため
実際には検査されない空隙があった。本 PR で `kinds/autoregressive.py::
_export_ar_onnx` が組み込む `Where`（`ids` 値域）・`Gather`（`N` 上限）に
よる fail-closed な検査を、この 2 点について実測で確認する。

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
from fandhe_edge_trainer.kinds.autoregressive import (
    EOS,
    MAX_INPUT_ID,
    PAD,
    SEP,
    VOCAB_SIZE,
    AutoregressiveKind,
    AutoregressiveTrainedModel,
    _ar_export_max_batch_n,
)
from fandhe_edge_trainer.limits import (
    MAX_AR_EXPORT_ATTENTION_ELEMENTS,
    MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS,
    MAX_AR_INFER_BATCH_N,
)


@pytest.fixture
def ar_trained(tmp_path: Path) -> tuple[AutoregressiveTrainedModel, Path]:
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG, max_bytes=8)
    trained = train_kind(kind, make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    return trained, onnx_path


@pytest.fixture
def ar_session(ar_trained: tuple[AutoregressiveTrainedModel, Path]) -> ReferenceEvaluator:
    _, onnx_path = ar_trained
    return ReferenceEvaluator(str(onnx_path))


def _n_guard_table_len(onnx_path: Path) -> int:
    """書き出したモデルへ埋め込まれた `n_guard_table` 初期化子の要素数
    （`= n_max + 1`）を読み取る（`_export_ar_onnx` が組み込むテーブル。
    `_ar_export_max_batch_n` の計算をテスト側で再実装せず、書き出し
    結果そのものを検証する）。
    """
    model = onnx.load(str(onnx_path))
    for init in model.graph.initializer:
        if init.name == "n_guard_table":
            dims = list(init.dims)
            assert len(dims) == 1, f"n_guard_table must be 1-D, got dims={dims}"
            return dims[0]
    raise AssertionError("n_guard_table initializer not found in exported model")


@pytest.mark.parametrize(
    "value",
    [-1, -VOCAB_SIZE, -VOCAB_SIZE - 1, SEP, EOS, VOCAB_SIZE, VOCAB_SIZE + 1000],
)
def test_ar_ids_out_of_range_or_special_raises(ar_session: ReferenceEvaluator, value: int) -> None:
    """負の値・特殊 ID（SEP・EOS）・`VOCAB_SIZE` 以上のいずれも、書き出した
    モデル全体で `Gather` が範囲外参照として例外で失敗する（REQ-39。`ids`
    値域ガードの `Where` が全ての禁止値を「確実に範囲外になる値」
    （`vocab_depth`=VOCAB_SIZE）へ書き換えていることの確認。これにより
    負インデックスが `Gather` の wrap 経路へ渡ることも無くなる）。
    """
    ids = np.array([[value, 1, 1, 1]], dtype=np.int64)
    with pytest.raises(IndexError):
        ar_session.run(None, {"ids": ids})


def test_ar_ids_mixed_valid_and_invalid_row_raises(ar_session: ReferenceEvaluator) -> None:
    """バッチ内の 1 行だけが範囲外の値を含む場合でも、バッチ全体の推論が
    失敗する（REQ-39・REQ-28。範囲外の行だけを無視して処理を続けない）。
    """
    ids = np.array([[1, 1, 1, 1], [SEP, 1, 1, 1]], dtype=np.int64)
    with pytest.raises(IndexError):
        ar_session.run(None, {"ids": ids})


def test_ar_ids_pad_and_max_input_id_boundaries_do_not_raise(
    ar_session: ReferenceEvaluator,
) -> None:
    """`PAD`（0）・`MAX_INPUT_ID`（256）は許可範囲の境界値として有限の
    確率を返す（REQ-39。値域ガードが正当な境界値を誤って拒否しないことの
    確認）。
    """
    ids = np.array([[PAD, MAX_INPUT_ID, 1, 1]], dtype=np.int64)
    (probs,) = ar_session.run(None, {"ids": ids})
    assert np.all(np.isfinite(probs))


def test_ar_ids_invalid_value_beyond_max_bytes_does_not_raise(
    ar_trained: tuple[AutoregressiveTrainedModel, Path],
) -> None:
    """`max_bytes` を超えた位置にある範囲外の値は、グラフ内の `Slice` で
    切り詰められて以降の計算に一切使われないため、値域ガードに掛からず
    例外にならない（REQ-39・REQ-28。値域ガードを T 軸の切り詰め〔Slice〕の
    **後**に置く設計判断を裏付ける。切り詰め前に検査しても、切り詰めで
    捨てられる部分は計算結果に影響しないため、追加の安全性は無い）。
    """
    trained, onnx_path = ar_trained
    session = ReferenceEvaluator(str(onnx_path))
    max_bytes = trained.max_bytes
    row = [1] * max_bytes + [SEP] * 4  # 先頭 max_bytes は有効値、以降は範囲外。
    ids = np.array([row], dtype=np.int64)
    (probs,) = session.run(None, {"ids": ids})
    assert np.all(np.isfinite(probs))


def test_ar_n_within_limit_does_not_raise(
    ar_trained: tuple[AutoregressiveTrainedModel, Path],
) -> None:
    """`N`（バッチ件数）が上限以内（本テストでは N=1）であれば `N` 上限
    ガードに掛からず有限の確率を返す（REQ-39）。
    """
    trained, onnx_path = ar_trained
    session = ReferenceEvaluator(str(onnx_path))
    ids = np.ones((1, trained.max_bytes), dtype=np.int64)
    (probs,) = session.run(None, {"ids": ids})
    assert np.all(np.isfinite(probs))


def test_ar_n_exceeds_limit_raises(ar_trained: tuple[AutoregressiveTrainedModel, Path]) -> None:
    """`N` が書き出し時に固定した `n_max`（`n_guard_table` の要素数 - 1）を
    超えると、`N` 上限ガードの `Gather` が範囲外参照として例外で失敗する
    （REQ-39。`ids` の値そのものは全行有効値にし、`N` 超過だけが原因で
    失敗することを確認する）。
    """
    trained, onnx_path = ar_trained
    session = ReferenceEvaluator(str(onnx_path))
    n_max = _n_guard_table_len(onnx_path) - 1
    ids = np.ones((n_max + 1, trained.max_bytes), dtype=np.int64)
    with pytest.raises(IndexError):
        session.run(None, {"ids": ids})


def test_ar_export_max_batch_n_ceiling_dominates_for_small_per_n_estimates() -> None:
    """per-N の見積もりが十分小さい（=モデルが小さい）場合は、モデル構成に
    依らない固定シーリング `MAX_AR_INFER_BATCH_N` が `n_max` を決める
    （REQ-39。`_ar_export_max_batch_n` の計算式を具体値で検証する）。
    """
    n_max = _ar_export_max_batch_n(per_n_attention_elements=1, per_n_choice_logprob_elements=1)
    assert n_max == MAX_AR_INFER_BATCH_N


def test_ar_export_max_batch_n_attention_term_dominates() -> None:
    """attention 側の見積もりがシーリングより厳しい場合は、そちらが
    `n_max` を決める（REQ-39。整数除算が正確に割り切れる値で検証する）。
    """
    per_n_attn = MAX_AR_EXPORT_ATTENTION_ELEMENTS // 10  # n_from_attention == 10
    n_max = _ar_export_max_batch_n(
        per_n_attention_elements=per_n_attn, per_n_choice_logprob_elements=1
    )
    assert n_max == 10


def test_ar_export_max_batch_n_choice_term_dominates() -> None:
    """選択肢対数確率側の見積もりがシーリングより厳しい場合は、そちらが
    `n_max` を決める（REQ-39。整数除算が正確に割り切れる値で検証する）。
    """
    per_n_choice = MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS // 5  # n_from_choice == 5
    n_max = _ar_export_max_batch_n(
        per_n_attention_elements=1, per_n_choice_logprob_elements=per_n_choice
    )
    assert n_max == 5


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
    例外にならない（REQ-39。`Gather` 単体では検査できない wrap 経路が
    存在することの根拠。この経路があるからこそ、`_export_ar_onnx` は
    `ids` を `Gather` へ渡す前に `Where` で範囲外の値を「確実に範囲外に
    なる正の値」へ書き換える必要がある。`test_ar_ids_out_of_range_or_
    special_raises` がその `Where` ガードの効果を確認する）。
    """
    session = ReferenceEvaluator(_minimal_embedding_gather_model())
    (row_neg,) = session.run(None, {"ids": np.array([-1], dtype=np.int64)})
    (row_last,) = session.run(None, {"ids": np.array([VOCAB_SIZE - 1], dtype=np.int64)})
    np.testing.assert_array_equal(row_neg, row_last)
