"""対応づけ (b) の選択肢 ID 対応づけ・予測レコード組み立て（REQ-19b・REQ-21・
REQ-27・REQ-28・REQ-39・TASK-19b.1-2・#80）のテスト。CPU・合成データ限定
（証拠種別: テストハーネス）。

`choice_posteriors`・`resolve_choice_id`・`map_scores_to_choice`・
`build_prediction_record`・`prediction_record_to_json_line` はユニットテスト
（MLX 不要・具体値）、`predict_records` は結合テスト（`TINY_AR_CONFIG`・CPU・
合成データ）で検証する。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from onnx.reference import ReferenceEvaluator

from conftest import (
    ATOL_BATCH_PARITY,
    LABEL_ORDER,
    TINY_AR_CONFIG,
    export_onnx_to_path,
    make_examples,
    make_request,
    train_kind,
)
from fandhe_edge_trainer import budget as budget_mod
from fandhe_edge_trainer.encoding import encode_bytes
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode
from fandhe_edge_trainer.kinds.autoregressive import (
    MAX_PREDICTION_ID_BYTES,
    AutoregressiveKind,
    Mapped,
    Unmapped,
    _encode_choices,
    build_prediction_record,
    map_scores_to_choice,
    predict_records,
    prediction_record_to_json_line,
    resolve_choice_id,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCORE_TOLERANCE_FIXTURE = _REPO_ROOT / "fixtures" / "score_tolerance" / "score_sum_tolerance.json"


def _load_score_sum_tolerance() -> float:
    data = json.loads(_SCORE_TOLERANCE_FIXTURE.read_text(encoding="utf-8"))
    return float(data["score_sum_tolerance"])


_MULTIBYTE_LABEL_ORDER = ["cat_a", "cat_b", "犬"]


# --- ユニットテスト: resolve_choice_id --------------------------------------


def test_req19b_resolve_choice_id_exact_match() -> None:
    """受入基準 1: 生成出力（選択肢のトークン列）が選択肢 ID と完全一致する
    具体例で、正しい選択肢 ID に対応づけられること。マルチバイトの選択肢
    （"犬"）も含める。
    """
    ids_by_label = _encode_choices(_MULTIBYTE_LABEL_ORDER)
    for label in _MULTIBYTE_LABEL_ORDER:
        assert resolve_choice_id(ids_by_label[label], _MULTIBYTE_LABEL_ORDER) == label


def test_req19b_resolve_choice_id_rejects_non_exact() -> None:
    """前方一致・余分な接尾辞・特殊トークン混入・範囲外の値・NFKC 同一視の
    いずれも `None`（一致無し）になること（完全一致以外は採用しない）。
    """
    ids_by_label = _encode_choices(["cat_a", "cat_b"])
    cat_a_ids = ids_by_label["cat_a"]

    # 前方一致（"cat" のトークン列）は "cat_a" と一致しない。
    prefix_ids = [b + 1 for b in b"cat"]
    assert resolve_choice_id(prefix_ids, ["cat_a", "cat_b"]) is None

    # 余分な接尾辞。
    suffix_ids = [b + 1 for b in b"cat_a_"]
    assert resolve_choice_id(suffix_ids, ["cat_a", "cat_b"]) is None

    # EOS(258) を含む列。
    assert resolve_choice_id([*cat_a_ids, 258], ["cat_a", "cat_b"]) is None
    # SEP(257) を含む列。
    assert resolve_choice_id([*cat_a_ids, 257], ["cat_a", "cat_b"]) is None
    # PAD(0) を含む列。
    assert resolve_choice_id([*cat_a_ids, 0], ["cat_a", "cat_b"]) is None
    # 範囲外の値（257 超過・0 未満）。
    assert resolve_choice_id([257], ["cat_a", "cat_b"]) is None
    assert resolve_choice_id([258], ["cat_a", "cat_b"]) is None
    assert resolve_choice_id([-1], ["cat_a", "cat_b"]) is None

    # NFKC で同一視されうる全角表記は完全一致しない（正規化しない方針）。
    fullwidth_ids = [b + 1 for b in "ｃａｔ＿ａ".encode()]
    assert resolve_choice_id(fullwidth_ids, ["cat_a", "cat_b"]) is None


# --- ユニットテスト: map_scores_to_choice -----------------------------------


def test_req19b_map_scores_to_choice_concrete_values() -> None:
    """`loglik=[-1.0, -2.0, -3.0]` のとき `predicted_label=="cat_a"` になり、
    `probs` が手計算の softmax 値と 1e-12 以内で一致すること。
    """
    label_order = ["cat_a", "cat_b", "cat_c"]
    choice_ids_by_label = _encode_choices(label_order)
    loglik = np.array([-1.0, -2.0, -3.0])

    mapping = map_scores_to_choice(loglik, label_order, choice_ids_by_label)

    assert isinstance(mapping, Mapped)
    assert mapping.choice_id == "cat_a"
    assert mapping.index == 0

    denom = math.exp(0.0) + math.exp(-1.0) + math.exp(-2.0)
    expected = (math.exp(0.0) / denom, math.exp(-1.0) / denom, math.exp(-2.0) / denom)
    for actual, exp_value in zip(mapping.probs, expected, strict=True):
        assert abs(actual - exp_value) <= 1e-12


def test_req19b_map_scores_tie_breaks_by_declaration_order() -> None:
    """同点 `[-1.0, -1.0, -5.0]` のとき先頭の `"cat_a"` になること
    （タイブレークは宣言順。`crates/core/src/judgment.rs` と同じ規則）。
    """
    label_order = ["cat_a", "cat_b", "cat_c"]
    choice_ids_by_label = _encode_choices(label_order)
    loglik = np.array([-1.0, -1.0, -5.0])

    mapping = map_scores_to_choice(loglik, label_order, choice_ids_by_label)

    assert isinstance(mapping, Mapped)
    assert mapping.choice_id == "cat_a"
    assert mapping.index == 0


@pytest.mark.parametrize(
    "loglik",
    [
        [float("nan"), -1.0],
        [float("inf"), -1.0],
        [float("-inf"), -1.0],
    ],
)
def test_req19b_map_scores_non_finite_is_unmapped(loglik: list[float]) -> None:
    """`NaN`・`+inf`・`-inf` を含む行は `Unmapped("invalid_score")` になり、
    `status:"error"` のレコードになること（`ok` を偽装しない。fail-closed）。
    """
    label_order = ["cat_a", "cat_b"]
    choice_ids_by_label = _encode_choices(label_order)

    mapping = map_scores_to_choice(np.array(loglik), label_order, choice_ids_by_label)

    assert isinstance(mapping, Unmapped)
    assert mapping.reason == "invalid_score"
    record = build_prediction_record("row-0", mapping, label_order)
    assert record == {"id": "row-0", "status": "error", "predicted_label": None}


def test_req19b_map_scores_length_mismatch_is_runtime_error() -> None:
    """行の長さと `label_order` の長さが一致しないとき `WorkerError`
    （runtime_error・exit 70）になること（呼び出し側のバグを黙って
    切り詰めない）。
    """
    label_order = ["cat_a", "cat_b", "cat_c"]
    choice_ids_by_label = _encode_choices(label_order)

    with pytest.raises(WorkerError) as exc_info:
        map_scores_to_choice(np.array([-1.0, -2.0]), label_order, choice_ids_by_label)
    assert exc_info.value.code == "runtime_error"
    assert exc_info.value.exit_code == ExitCode.RUNTIME_ERROR


# --- ユニットテスト: build_prediction_record --------------------------------


def test_req21_prediction_record_schema() -> None:
    """ok のレコードのキーがちょうど `{id, status, predicted_label, scores}`
    で、この順序であること。`scores` のキー順が `label_order` と同じである
    こと。error のレコードが `{id, status:"error", predicted_label:null}`
    であること。
    """
    label_order = ["cat_a", "cat_b", "cat_c"]
    mapping = Mapped(choice_id="cat_b", index=1, probs=(0.2, 0.5, 0.3))

    record = build_prediction_record("row-1", mapping, label_order)
    assert list(record.keys()) == ["id", "status", "predicted_label", "scores"]
    assert record["status"] == "ok"
    assert record["predicted_label"] == "cat_b"
    assert list(record["scores"].keys()) == label_order
    assert record["scores"] == {"cat_a": 0.2, "cat_b": 0.5, "cat_c": 0.3}

    error_record = build_prediction_record("row-2", Unmapped("no_exact_match"), label_order)
    assert error_record == {"id": "row-2", "status": "error", "predicted_label": None}
    assert "scores" not in error_record
    assert "reason" not in error_record
    assert "reason_code" not in error_record


def test_req21_scores_sum_within_shared_tolerance() -> None:
    """`fixtures/score_tolerance/score_sum_tolerance.json` の許容差内で
    `scores` の合計が 1 であり、各値が `[0, 1]` にあること。
    """
    tol = _load_score_sum_tolerance()
    label_order = ["cat_a", "cat_b", "cat_c"]
    choice_ids_by_label = _encode_choices(label_order)
    loglik = np.array([-0.3, -1.7, -4.2])

    mapping = map_scores_to_choice(loglik, label_order, choice_ids_by_label)
    assert isinstance(mapping, Mapped)
    record = build_prediction_record("row-3", mapping, label_order)

    values = list(record["scores"].values())
    assert all(0.0 <= v <= 1.0 for v in values)
    assert abs(sum(values) - 1.0) <= tol


def test_req21_json_line_rejects_nan() -> None:
    """NaN を含む dict を JSON 化しようとすると `WorkerError` になること
    （`allow_nan=False`）。正常なレコードは `json.loads` で元と同じ値に
    戻ること。
    """
    with pytest.raises(WorkerError) as exc_info:
        prediction_record_to_json_line(
            {"id": "row-4", "status": "ok", "scores": {"a": float("nan")}}
        )
    assert exc_info.value.code == "runtime_error"
    assert exc_info.value.exit_code == ExitCode.RUNTIME_ERROR

    record = {"id": "row-5", "status": "ok", "predicted_label": "cat_a", "scores": {"cat_a": 1.0}}
    line = prediction_record_to_json_line(record)
    assert json.loads(line) == record


def test_req39_prediction_id_validation() -> None:
    """空の `id`、`MAX_PREDICTION_ID_BYTES` を超える `id`、str 以外の `id` が
    `invalid_request`（exit 64）になること。エラーメッセージに入力本文
    （選択肢名等）が含まれないこと。
    """
    label_order = ["cat_a", "cat_b"]
    mapping = Mapped(choice_id="cat_a", index=0, probs=(0.9, 0.1))

    with pytest.raises(WorkerError) as exc_info:
        build_prediction_record("", mapping, label_order)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT

    too_long_id = "x" * (MAX_PREDICTION_ID_BYTES + 1)
    with pytest.raises(WorkerError) as exc_info:
        build_prediction_record(too_long_id, mapping, label_order)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
    assert too_long_id not in exc_info.value.message

    with pytest.raises(WorkerError) as exc_info:
        build_prediction_record(123, mapping, label_order)  # type: ignore[arg-type]
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


# --- 結合テスト: predict_records（TINY_AR_CONFIG・CPU・合成データ） ---------


def test_req19b_predict_records_learns_synthetic_task(tmp_path: Path) -> None:
    """`predict_records` の `predicted_label` が合成データ全件で正解ラベルと
    一致し、全件が `status=="ok"` であること（評価器が採点できる形である
    ことの確認。`test_ar_train_learns_the_synthetic_task` と同じ前提）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)

    rows = [(str(i), ex.input) for i, ex in enumerate(examples)]
    records = list(predict_records(trained, rows))

    assert len(records) == len(rows)
    assert all(r["status"] == "ok" for r in records)
    predicted = [r["predicted_label"] for r in records]
    gold = [ex.label for ex in examples]
    accuracy = sum(p == g for p, g in zip(predicted, gold, strict=True)) / len(gold)
    assert accuracy == 1.0


def test_req28_predict_records_chunking_invariant(tmp_path: Path) -> None:
    """`chunk_size=1` と `chunk_size=len(rows)` で `predicted_label` が全件
    一致し、`scores` が `ATOL_BATCH_PARITY` 以内で一致すること（REQ-28。
    1 件ずつの推論とバッチ推論の結果が一致する不変条件）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)

    rows = [(str(i), ex.input) for i, ex in enumerate(examples)]
    single = list(predict_records(trained, rows, chunk_size=1))
    batched = list(predict_records(trained, rows, chunk_size=len(rows)))

    assert [r["predicted_label"] for r in single] == [r["predicted_label"] for r in batched]
    for r_single, r_batched in zip(single, batched, strict=True):
        assert r_single["status"] == r_batched["status"] == "ok"
        for label in trained.label_order:
            diff = abs(r_single["scores"][label] - r_batched["scores"][label])
            assert diff <= ATOL_BATCH_PARITY


def test_req19b_predict_records_matches_onnx_argmax(tmp_path: Path) -> None:
    """同じ入力について、書き出した ONNX を `ReferenceEvaluator` で実行した
    `probs` の argmax が、レコードの `predicted_label` と全件一致すること
    （Python 側の確認経路と配布成果物〔ONNX〕の対応を結びつける）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)

    rows = [(str(i), ex.input) for i, ex in enumerate(examples)]
    records = list(predict_records(trained, rows))

    id_lists = [encode_bytes(text, req.max_bytes) for _row_id, text in rows]
    max_len = max(len(ids) for ids in id_lists)
    padded = np.zeros((len(id_lists), max_len), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        padded[i, : len(ids)] = ids

    session = ReferenceEvaluator(str(onnx_path))
    (onnx_probs,) = session.run(None, {"ids": padded})
    onnx_predicted = [LABEL_ORDER[i] for i in onnx_probs.argmax(axis=1)]

    assert [r["predicted_label"] for r in records] == onnx_predicted


def test_req39_predict_records_checks_budget_between_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resource_budget.check` が各チャンクの計算「前」と「後」の両方で
    呼ばれること（REQ-39。計算後だけの検査では確保済みの資源を検査する
    頃には手遅れという Codex レビュー指摘。PR #234）。"""
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)

    budget = budget_mod.ResourceBudget(wall_seconds=60.0, rss_bytes=1 << 30, device="cpu")
    calls = {"n": 0}
    real_check = budget.check

    def _counting_check() -> None:
        calls["n"] += 1
        real_check()

    monkeypatch.setattr(budget, "check", _counting_check)

    rows = [(str(i), ex.input) for i, ex in enumerate(examples)]
    chunk_size = 3
    list(predict_records(trained, rows, chunk_size=chunk_size, resource_budget=budget))

    expected_chunks = math.ceil(len(rows) / chunk_size)
    # チャンクごとに計算「前」と「後」の 2 回呼ぶ設計（PR #234 レビュー対応）。
    assert calls["n"] == expected_chunks * 2


def test_req39_predict_records_rejects_rows_over_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`rows` の件数が `MAX_AR_PREDICT_ROWS` を超える場合、一括リスト化・
    予測レコードの蓄積を始める前に `limit_exceeded`・exit 20 で拒否する
    （REQ-39。`chunk_size` はチャンクあたりの計算量しか制限しないため、
    件数自体の上限は別に必要という Codex レビュー指摘への回帰テスト。
    PR #234）。
    """
    import fandhe_edge_trainer.kinds.autoregressive as ar_module

    monkeypatch.setattr(ar_module, "MAX_AR_PREDICT_ROWS", 2)

    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)

    rows = [(str(i), ex.input) for i, ex in enumerate(examples)]
    assert len(rows) > 2

    with pytest.raises(WorkerError) as exc_info:
        predict_records(trained, rows)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_req39_predict_records_rejects_oversized_raw_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1 行の生テキストが `MAX_TRAIN_LINE_BYTES` を超える場合、`encode_bytes`
    （NFKC 正規化・UTF-8 化）を呼ぶ前に `limit_exceeded`・exit 20 で拒否する
    （REQ-39。正規化前にサイズを検査すべきという Codex レビュー指摘への
    回帰テスト。PR #234）。
    """
    import fandhe_edge_trainer.kinds.autoregressive as ar_module

    monkeypatch.setattr(ar_module, "MAX_TRAIN_LINE_BYTES", 4)

    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(kind, make_examples(), req)

    rows = [("row-0", "this text is longer than 4 bytes")]

    with pytest.raises(WorkerError) as exc_info:
        list(predict_records(trained, rows))
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_req39_predict_records_rejects_oversized_attention_elements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_score_choices_mlx` が `full`・attention テンソルを確保する前に、
    見積もり要素数（`N x K x heads x layers x length^2`）を
    `MAX_AR_EXPORT_ATTENTION_ELEMENTS` で fail-closed に拒否すること
    （REQ-39。`chunk_size` だけでは確保量を抑えきれないという Codex
    レビュー指摘への回帰テスト。PR #234）。
    """
    import fandhe_edge_trainer.kinds.autoregressive as ar_module

    monkeypatch.setattr(ar_module, "MAX_AR_EXPORT_ATTENTION_ELEMENTS", 1)

    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)

    rows = [(str(i), ex.input) for i, ex in enumerate(examples)]

    with pytest.raises(WorkerError) as exc_info:
        list(predict_records(trained, rows))
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_req19b_predict_records_empty_input(tmp_path: Path) -> None:
    """空文字列の入力（`encode_bytes("") == [0]`）でも有限の `scores` を持つ
    ok のレコードになること（REQ-28。詰め物のみの行でも NaN が伝播しない
    こと。`test_ar_score_choices_handles_empty_input` と同じ前提）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(kind, make_examples(), req)

    records = list(predict_records(trained, [("empty-row", "")]))

    assert len(records) == 1
    assert records[0]["status"] == "ok"
    assert all(math.isfinite(v) for v in records[0]["scores"].values())


# --- 回帰確認: 既存の学習・書き出しロジックへ影響しないこと -----------------


def test_ar_choice_mapping_module_still_exports_score_choices_mlx() -> None:
    """`_score_choices_mlx` を `predict_records` が再利用する契約が壊れて
    いないこと（モジュール docstring・#80 の前提）。
    """
    from fandhe_edge_trainer.kinds.autoregressive import _score_choices_mlx

    assert callable(_score_choices_mlx)
