"""ONNX グラフ内で `ids` の T 軸を `max_bytes` へ Slice で切り詰める挙動の検証
（REQ-39・REQ-28。PR #222 セキュリティレビュー P0 指摘）。

`kinds/autoregressive.py` モジュール docstring 6 番の不変条件（T が
`max_bytes` を超えて渡されても、グラフ内で実際に計算へ使われる長さは
常に `max_bytes` 以下になる）を、`_encode_input_examples`/`encode_bytes`
による事前の切り詰めを経由しない「未検証な生の `ids` 配列」を直接グラフへ
渡すことで確認する（CLI・推論ランタイムの前処理を経由しない攻撃者が、
切り詰め前の長い・PAD を含む配列を直接 `ids` として渡すケースを模擬する）。

証拠種別: テストハーネス（onnx.reference。ONNX グラフ自体の契約確認に
限られ、Rust 側の推論ランタイム〔`ort` crate〕を使った実機での確認は
推論ランタイム層（TASK-28・REQ-28。runtime-builder 担当）の責務であり、
本テストはそれを代替しない）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

from conftest import (
    ATOL_BATCH_PARITY,
    TINY_AR_CONFIG,
    export_onnx_to_path,
    make_examples,
    make_request,
    train_kind,
)
from fandhe_edge_trainer.kinds.autoregressive import AutoregressiveKind


def test_ar_dense_input_longer_than_max_bytes_matches_truncated_input(tmp_path: Path) -> None:
    """T > max_bytes の密な入力（PAD を含まない）を渡しても、グラフ内の
    `Slice` が先頭 max_bytes バイトへ切り詰めた入力と同じ出力を返す
    （REQ-28・REQ-39）。修正前は `pos_table`〔行数 = 学習時の
    `max_len = max_bytes + 1 + M` で固定〕への `Gather` が範囲外参照になり
    失敗していたケース（`limits.py::MAX_AR_EXPORT_ATTENTION_ELEMENTS`
    docstring 参照）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG, max_bytes=8)
    trained = train_kind(kind, make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    session = ReferenceEvaluator(str(onnx_path))

    # 1..256 の密な（PAD を含まない）トークン列。max_bytes=8 の 2 倍以上の
    # 長さを持ち、`encode_bytes` の切り詰めを経由せず直接構築する
    # （前処理を経由しない未検証な `ids` 配列を模擬する）。
    long_dense = np.array([[(i % 250) + 1 for i in range(20)]], dtype=np.int64)
    truncated = long_dense[:, : req.max_bytes]

    (probs_long,) = session.run(None, {"ids": long_dense})
    (probs_short,) = session.run(None, {"ids": truncated})

    assert np.all(np.isfinite(probs_long))
    diff = float(np.max(np.abs(probs_long - probs_short)))
    assert diff <= ATOL_BATCH_PARITY, f"max abs diff {diff} exceeds {ATOL_BATCH_PARITY}"


def test_ar_padded_input_longer_than_max_bytes_matches_truncated_input(tmp_path: Path) -> None:
    """PAD（値 0）を含む T > max_bytes の入力（バッチ内の他の行が長いために
    詰め物される想定）でも、先頭 max_bytes 列へ切り詰めた入力と同じ出力を
    返す（REQ-28・REQ-39・Codex 再レビュー指摘の PAD ケース）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG, max_bytes=8)
    trained = train_kind(kind, make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    session = ReferenceEvaluator(str(onnx_path))

    # 実長 5（< max_bytes=8）のトークン列に、max_bytes を超える位置まで
    # PAD（0）を継ぎ足した行（バッチ内の他の行が長いために右詰め PAD
    # される状況を模擬する）。
    real_tokens = [10, 20, 30, 40, 50]
    padded_wide = np.zeros((1, 20), dtype=np.int64)
    padded_wide[0, : len(real_tokens)] = real_tokens
    truncated = padded_wide[:, : req.max_bytes]  # 実長 5 + PAD 3 列（max_bytes=8 まで）

    (probs_wide,) = session.run(None, {"ids": padded_wide})
    (probs_short,) = session.run(None, {"ids": truncated})

    assert np.all(np.isfinite(probs_wide))
    diff = float(np.max(np.abs(probs_wide - probs_short)))
    assert diff <= ATOL_BATCH_PARITY, f"max abs diff {diff} exceeds {ATOL_BATCH_PARITY}"


def test_ar_export_graph_attention_shape_bounded_by_max_bytes(tmp_path: Path) -> None:
    """T > max_bytes を渡しても、グラフ内部の attention マスク（`mask4d`。
    形状 `[N*K, 1, L, L]`、`L = T_trunc + 1 + M`）の最終 2 軸が
    `max_bytes + 1 + M` を超えないことを確認する（REQ-39。`_check_ar_export_
    resources`・`limits.py::MAX_AR_EXPORT_ATTENTION_ELEMENTS` docstring
    参照）。`mask4d` を追加出力としてグラフへ挿入し、内部テンソルの実形状を
    観測する。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG, max_bytes=8)
    trained = train_kind(kind, make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)

    model_proto = onnx.load(str(onnx_path))
    mask_output = onnx.helper.make_tensor_value_info("mask4d", onnx.TensorProto.FLOAT, None)
    model_proto.graph.output.append(mask_output)
    session = ReferenceEvaluator(model_proto)

    long_dense = np.array([[(i % 250) + 1 for i in range(30)]], dtype=np.int64)
    _, mask4d = session.run(None, {"ids": long_dense})

    m = trained.max_label_len + 1
    expected_l = req.max_bytes + 1 + m
    assert mask4d.shape[-1] == expected_l
    assert mask4d.shape[-2] == expected_l
