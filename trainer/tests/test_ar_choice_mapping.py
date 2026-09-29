"""対応づけ (b) の選択肢 ID 対応づけ・予測レコード組み立て（REQ-19b・REQ-21・
REQ-27・REQ-28・REQ-39・TASK-19b.1-2・#80）のテスト。CPU・合成データ限定
（証拠種別: テストハーネス）。

`choice_posteriors`・`resolve_choice_id`・`map_scores_to_choice`・
`build_prediction_record`・`prediction_record_to_json_line` はユニットテスト
（MLX 不要・具体値）、`predict_records` は結合テスト（`TINY_AR_CONFIG`・CPU・
合成データ）で検証する。
"""

from __future__ import annotations

import dataclasses
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
    VOCAB_SIZE,
    AutoregressiveKind,
    ChoiceMappingTally,
    Mapped,
    Unmapped,
    UnmappedReason,
    _default_predict_resource_budget,
    _encode_choices,
    _max_safe_predict_chunk_size,
    build_prediction_record,
    map_scores_to_choice,
    predict_records,
    prediction_record_to_json_line,
    resolve_choice_id,
    resolve_generated_output,
)
from fandhe_edge_trainer.limits import (
    MAX_AR_EXPORT_ATTENTION_ELEMENTS,
    MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS,
    MAX_TRAIN_RSS_BYTES,
    MAX_TRAIN_WALL_SECONDS,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCORE_TOLERANCE_FIXTURE = _REPO_ROOT / "fixtures" / "score_tolerance" / "score_sum_tolerance.json"


def _load_score_sum_tolerance() -> float:
    data = json.loads(_SCORE_TOLERANCE_FIXTURE.read_text(encoding="utf-8"))
    return float(data["score_sum_tolerance"])


_MULTIBYTE_LABEL_ORDER = ["cat_a", "cat_b", "犬"]


class _AssertNoEncode(str):
    """`encode` が呼ばれたら即座に失敗する `str` サブクラス（テスト専用）。

    文字数（コードポイント数）による事前検査が `text.encode("utf-8")` より
    先に機能し、上限超過の入力では `encode` が一度も呼ばれないことを、
    実際に `encode` を失敗させて証明する（Codex レビュー指摘 P0・PR #234）。
    """

    def encode(self, *args: object, **kwargs: object) -> bytes:
        raise AssertionError("encode() must not be called before the char-count pre-check")


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


def test_req19b_map_scores_argmax_uses_softmax_probs_not_raw_loglik() -> None:
    """`loglik=[-1e-17, 0.0, -100.0]` は生の対数尤度では 2 番目
    （`"cat_b"`）が最大だが、softmax 後は丸めにより `probs[0] == probs[1]`
    の同点になる（1e-17 は float64 の丸め誤差未満のため）。出力する
    `scores`（softmax 後）に対して宣言順タイブレークで argmax を取ると
    先頭の `"cat_a"` になるべきで、`predicted_label` は
    `max(mapping.probs)` を実現する最初の添字と一致しなければならない
    （Codex レビュー #234 指摘の回帰テスト。生の対数尤度で argmax を
    決めていた旧実装では `"cat_b"`（index 1）になり、
    `crates/core/src/judgment.rs::JudgmentResult::new` の
    `predicted_choice_id` と最大スコア一致の契約に反していた）。
    """
    label_order = ["cat_a", "cat_b", "cat_c"]
    choice_ids_by_label = _encode_choices(label_order)
    loglik = np.array([-1e-17, 0.0, -100.0])

    mapping = map_scores_to_choice(loglik, label_order, choice_ids_by_label)

    assert isinstance(mapping, Mapped)
    assert mapping.probs[0] == mapping.probs[1]
    assert mapping.index == 0
    assert mapping.choice_id == "cat_a"
    max_prob = max(mapping.probs)
    assert mapping.probs[mapping.index] == max_prob


@pytest.mark.parametrize(
    "loglik",
    [
        [float("nan"), -1.0],
        [float("inf"), -1.0],
        [float("-inf"), -1.0],
    ],
)
def test_req19b_map_scores_non_finite_is_unmapped(loglik: list[float]) -> None:
    """`NaN`・`+inf`・`-inf` を含む行は `Unmapped(UnmappedReason.INVALID_SCORE)` になり、
    `status:"error"` のレコードになること（`ok` を偽装しない。fail-closed）。
    """
    label_order = ["cat_a", "cat_b"]
    choice_ids_by_label = _encode_choices(label_order)

    mapping = map_scores_to_choice(np.array(loglik), label_order, choice_ids_by_label)

    assert isinstance(mapping, Unmapped)
    assert mapping.reason is UnmappedReason.INVALID_SCORE
    assert mapping.reason is not UnmappedReason.NO_CHOICE_MATCH
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

    for reason in (UnmappedReason.NO_CHOICE_MATCH, UnmappedReason.INVALID_SCORE):
        error_record = build_prediction_record("row-2", Unmapped(reason), label_order)
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


def test_req39_build_prediction_record_rejects_long_id_before_full_utf8_encode() -> None:
    """`build_prediction_record` が `id` の文字数（コードポイント数）で
    先に足切りしてから `record_id.encode("utf-8")` を呼ぶこと（REQ-39。
    `_guarded_encode_bytes` と同種の予防で、検査より先に `encode` を呼ぶと
    巨大な `id` でメモリを無制限に消費しうる）。
    """
    label_order = ["cat_a", "cat_b"]
    mapping = Mapped(choice_id="cat_a", index=0, probs=(0.9, 0.1))
    record_id = _AssertNoEncode("x" * (MAX_PREDICTION_ID_BYTES + 1))

    with pytest.raises(WorkerError) as exc_info:
        build_prediction_record(record_id, mapping, label_order)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_req39_build_prediction_record_rejects_lone_surrogate_id() -> None:
    """`id` に孤立サロゲート（有効な UTF-8 で表現できない `str`）が含まれる
    場合、`UnicodeEncodeError` を送出せず `invalid_request`・exit 64 で
    拒否されること（advisor レビュー指摘・PR #234 フォローアップ。
    `_guarded_encode_bytes` の同種の穴〔孤立サロゲート〕が `record_id.
    encode("utf-8")` にも残っていた）。エラーメッセージに `id` 自体を
    含めない（security.md）。
    """
    label_order = ["cat_a", "cat_b"]
    mapping = Mapped(choice_id="cat_a", index=0, probs=(0.9, 0.1))
    lone_surrogate_id = "row-\ud800"

    with pytest.raises(WorkerError) as exc_info:
        build_prediction_record(lone_surrogate_id, mapping, label_order)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
    assert lone_surrogate_id not in exc_info.value.message


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


def test_req39_predict_records_rejects_oversized_raw_input_before_full_utf8_encode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1 行の生テキストが `MAX_TRAIN_LINE_BYTES` を超える場合、文字数
    （コードポイント数）による足切りが `text.encode("utf-8")` より先に働き、
    `encode` が一度も呼ばれずに拒否されること（REQ-39。UTF-8 のバイト数は
    文字数以上であることを利用した事前検査。`text.encode("utf-8")` を検査
    前に呼ぶと 1 件の巨大な入力でメモリを無制限に消費しうるという Codex
    レビュー指摘 P0・PR #234 への回帰テスト）。
    """
    import fandhe_edge_trainer.kinds.autoregressive as ar_module

    monkeypatch.setattr(ar_module, "MAX_TRAIN_LINE_BYTES", 4)

    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(kind, make_examples(), req)

    text = _AssertNoEncode("this text is longer than 4 chars")
    rows = [("row-0", text)]

    with pytest.raises(WorkerError) as exc_info:
        list(predict_records(trained, rows))
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_req39_predict_records_accepts_tuple_rows_without_full_list_copy(
    tmp_path: Path,
) -> None:
    """`rows` に `list` 以外のスライス可能な `Sequence`（`tuple`）を渡しても
    `list` を渡した場合と同じ結果になること（REQ-39。`predict_records` が
    `list(rows)` で全件コピーしてから渡す設計をやめ、`rows` をそのまま
    `_predict_records_stream` へ渡すようにした変更〔Codex レビュー指摘 P0・
    PR #234〕への回帰テスト。上限内の件数でもコピー自体が検査前に RSS を
    消費してしまうため、コピーをせずに動作することを確認する）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)

    rows_list = [(str(i), ex.input) for i, ex in enumerate(examples)]
    rows_tuple = tuple(rows_list)

    from_list = list(predict_records(trained, rows_list))
    from_tuple = list(predict_records(trained, rows_tuple))

    assert from_list == from_tuple


def test_req39_predict_records_checks_budget_before_first_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """計算後の `resource_budget.check()` が、チャンク内のレコードを 1 件でも
    `yield` する前に呼ばれること（REQ-39。Codex レビュー指摘 P1・PR #234
    への回帰テスト。計算後の検査を全件 `yield` した後まで遅らせると、
    呼び出し元が途中で反復を止めた場合に検査自体が行われず、継続する場合も
    上限超過後の結果が先に呼び出し元へ渡ってしまう）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)

    budget = budget_mod.ResourceBudget(wall_seconds=60.0, rss_bytes=1 << 30, device="cpu")
    calls = {"n": 0}

    def _raising_check() -> None:
        calls["n"] += 1
        # 1 回目はチャンク計算「前」の検査（通過させる）、2 回目は
        # `_score_choices_mlx` 直後・最初の yield 前の検査（ここで拒否する）。
        if calls["n"] == 2:
            raise WorkerError("limit_exceeded", "budget exceeded (test)", ExitCode.LIMIT_EXCEEDED)

    monkeypatch.setattr(budget, "check", _raising_check)

    rows = [(str(i), ex.input) for i, ex in enumerate(examples)]
    gen = predict_records(trained, rows, chunk_size=len(rows), resource_budget=budget)

    with pytest.raises(WorkerError) as exc_info:
        next(gen)
    assert exc_info.value.code == "limit_exceeded"
    assert calls["n"] == 2


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


def test_req39_predict_records_rejects_oversized_logprob_elements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_score_choices_mlx` が `logits`・`log_softmax`（`[N*K, length,
    VOCAB_SIZE]`）を確保する前に、見積もり要素数（`N x K x length x
    vocab_size`）を `MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS` で fail-closed
    に拒否すること（REQ-39。attention の見積もりだけでは vocab 次元の
    確保量を見逃すという Cursor Bugbot 指摘への回帰テスト。PR #234）。

    `MAX_AR_EXPORT_ATTENTION_ELEMENTS` は十分大きいまま
    （attention 側の検査を通過させる）にし、
    `MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS` のみを絞ることで、
    vocab 次元の検査が単独で機能することを確認する。
    """
    import fandhe_edge_trainer.kinds.autoregressive as ar_module

    monkeypatch.setattr(ar_module, "MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS", 1)

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


# --- ユニットテスト: _guarded_encode_bytes（孤立サロゲート） -----------------


def test_req39_predict_records_rejects_lone_surrogate_input(tmp_path: Path) -> None:
    """孤立サロゲート（有効な UTF-8 で表現できない `str`）を含む入力が
    `UnicodeEncodeError` を送出せず `invalid_request`・exit 64 で拒否される
    こと（Codex レビュー指摘 P1・PR #234。`json.loads('"\\ud800"')` 等で
    生成されうる、サロゲートペアでない単体のサロゲートを想定する）。エラー
    メッセージに入力本文を含めない（security.md）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(kind, make_examples(), req)

    lone_surrogate = "abc\ud800def"
    rows = [("row-0", lone_surrogate)]

    with pytest.raises(WorkerError) as exc_info:
        list(predict_records(trained, rows))
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
    assert lone_surrogate not in exc_info.value.message


# --- ユニットテスト: predict_records の既定資源予算の解決順 -----------------


def test_req39_predict_records_reuses_trained_resource_budget_by_default(
    tmp_path: Path,
) -> None:
    """`resource_budget` を省略した場合、`_default_predict_resource_budget`
    ではなく `trained.resource_budget`（学習時に使ったのと同じインスタンス）
    が使われること（advisor レビュー指摘・PR #234 フォローアップ。`device`
    がそのジョブの実際のリクエスト値を反映したインスタンスを優先する）。
    `trained.resource_budget` を期限切れにすると、`resource_budget` 未指定
    でも `limit_exceeded` で打ち切られることで確認する。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)
    # 生成直後に必ず期限切れになる予算（壁時計 -1 秒。0.0 だと
    # `time.monotonic()` の分解能次第でまだ期限内と判定されうるため、
    # 確実に過去になる負値を使う）へ差し替える。
    expired = budget_mod.ResourceBudget(
        wall_seconds=-1.0, rss_bytes=MAX_TRAIN_RSS_BYTES, device="cpu"
    )
    trained_with_expired_budget = dataclasses.replace(trained, resource_budget=expired)

    rows = [(str(i), ex.input) for i, ex in enumerate(examples)]
    with pytest.raises(WorkerError) as exc_info:
        list(predict_records(trained_with_expired_budget, rows, chunk_size=1))

    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_req39_predict_records_falls_back_to_default_budget_without_trained_budget(
    tmp_path: Path,
) -> None:
    """`resource_budget` 省略かつ `trained.resource_budget` も `None`
    （`AutoregressiveTrainedModel` を学習の外で直接構築した場合）のときだけ
    `_default_predict_resource_budget` が使われること（advisor レビュー
    指摘・PR #234 フォローアップ）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)
    trained_without_budget = dataclasses.replace(trained, resource_budget=None)

    def _expired_default_budget() -> budget_mod.ResourceBudget:
        return budget_mod.ResourceBudget(
            wall_seconds=-1.0, rss_bytes=MAX_TRAIN_RSS_BYTES, device="cpu"
        )

    import fandhe_edge_trainer.kinds.autoregressive as ar_module

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ar_module, "_default_predict_resource_budget", _expired_default_budget)
        rows = [(str(i), ex.input) for i, ex in enumerate(examples)]
        with pytest.raises(WorkerError) as exc_info:
            list(predict_records(trained_without_budget, rows, chunk_size=1))

    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_req39_default_predict_resource_budget_uses_train_defaults() -> None:
    """既定予算が学習側の既定値（`MAX_TRAIN_WALL_SECONDS`・
    `MAX_TRAIN_RSS_BYTES`）・`device="cpu"` をそのまま使うこと（学習・予測で
    別々の既定値を持たない設計の確認）。
    """
    budget = _default_predict_resource_budget()
    assert budget.wall_seconds == float(MAX_TRAIN_WALL_SECONDS)
    assert budget.rss_bytes == MAX_TRAIN_RSS_BYTES
    assert budget.device == "cpu"


# --- ユニットテスト: _max_safe_predict_chunk_size ----------------------------


def test_req39_max_safe_predict_chunk_size_bounds_default_ar_config() -> None:
    """既定の学習設定（`layers=2, heads=4, max_bytes=512`）・3 ラベル・
    短い選択肢（`max_label_len=8`）で、既定の `batch_size=32` がそのままでは
    `_score_choices_mlx` の見積もり上限を超えるが、`_max_safe_predict_
    chunk_size` が返す上限まで `chunk_size` を落とせば超えないこと
    （Cursor Bugbot 指摘・PR #234 への回帰テスト。具体値で確認する）。
    """
    heads, layers, k, m, max_bytes = 4, 2, 3, 9, 512
    length = max_bytes + 1 + m

    unsafe_n = 32
    unsafe_attn_elements = unsafe_n * k * heads * layers * length * length
    assert unsafe_attn_elements > MAX_AR_EXPORT_ATTENTION_ELEMENTS

    safe_n = _max_safe_predict_chunk_size(heads=heads, layers=layers, k=k, m=m, max_bytes=max_bytes)
    assert 0 < safe_n < unsafe_n

    safe_attn_elements = safe_n * k * heads * layers * length * length
    safe_logprob_elements = safe_n * k * length * VOCAB_SIZE
    assert safe_attn_elements <= MAX_AR_EXPORT_ATTENTION_ELEMENTS
    assert safe_logprob_elements <= MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS


def test_req39_max_safe_predict_chunk_size_floors_at_one() -> None:
    """どれだけ厳しい上限でも 0 ではなく 1 を返すこと（`chunk_size=0` は
    `range(0, n_rows, 0)` で `ValueError` になり、既存の
    `chunk_size = max(1, ...)` クランプと矛盾するため）。
    """
    safe_n = _max_safe_predict_chunk_size(heads=64, layers=32, k=4096, m=1024, max_bytes=1024)
    assert safe_n == 1


def test_req39_predict_records_auto_shrinks_chunk_size_to_stay_within_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """既定の `chunk_size`（`trained.config["batch_size"]`）が
    `_score_choices_mlx` の見積もり上限を超える設定でも、`predict_records`
    が自動的にチャンクサイズを縮めて全件を正常に処理できること（Cursor
    Bugbot 指摘・PR #234。上限自体は変更しない）。
    """
    import fandhe_edge_trainer.kinds.autoregressive as ar_module

    # TINY_AR_CONFIG（heads=2, layers=1）・2 ラベル（"cat_a"/"cat_b"。
    # max_label_len=5 → m=6）・小さい max_bytes での length は
    # 8 + 1 + 6 = 15。上限を極小値へ差し替え、既定の batch_size（8）が
    # そのままでは超過するが、自動的に縮めたチャンクサイズなら収まる
    # ようにする。
    monkeypatch.setattr(ar_module, "MAX_AR_EXPORT_ATTENTION_ELEMENTS", 2_000)

    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG, max_bytes=8)
    examples = make_examples()
    trained = train_kind(kind, examples, req)

    rows = [(str(i), ex.input) for i, ex in enumerate(examples)]
    records = list(predict_records(trained, rows))

    assert len(records) == len(rows)
    assert all(r["status"] == "ok" for r in records)


# --- 回帰確認: 既存の学習・書き出しロジックへ影響しないこと -----------------


def test_ar_choice_mapping_module_still_exports_score_choices_mlx() -> None:
    """`_score_choices_mlx` を `predict_records` が再利用する契約が壊れて
    いないこと（モジュール docstring・#80 の前提）。
    """
    from fandhe_edge_trainer.kinds.autoregressive import _score_choices_mlx

    assert callable(_score_choices_mlx)


# --- #81（TASK-19b.2）: 判定不能の型と防御的な契約検査（自由生成した出力の判定は未実装）---


def _tokens(text: str) -> list[int]:
    return [b + 1 for b in text.encode("utf-8")]


def test_req19b_resolve_generated_output_no_match_is_undecidable() -> None:
    """どの選択肢 ID にも一致しない生成出力は判定不能になり、例外を出さない。"""
    label_order = ["cat_a", "cat_b"]
    for text in ("cat_c", "cat", "cat_a ", "", "CAT_A"):
        assert resolve_generated_output(_tokens(text), label_order) == Unmapped(
            UnmappedReason.NO_CHOICE_MATCH
        )


def test_req19b_resolve_generated_output_invalid_tokens_is_undecidable() -> None:
    """PAD・SEP・EOS・範囲外・不正な UTF-8 も判定不能になる。"""
    label_order = ["cat_a", "cat_b"]
    invalid = [
        [0],
        [257],
        [258],
        [-1],
        [300],
        [*_tokens("cat_a"), 0],
        [0xFF + 1],
    ]
    for tokens in invalid:
        assert resolve_generated_output(tokens, label_order) == Unmapped(
            UnmappedReason.NO_CHOICE_MATCH
        )


def test_req19b_resolve_generated_output_exact_match() -> None:
    """完全一致する生成出力は選択肢 ID（str）へ解決される。"""
    assert resolve_generated_output(_tokens("cat_b"), ["cat_a", "cat_b"]) == "cat_b"


def test_req19b_map_scores_no_choice_match_when_tokens_mismatch() -> None:
    """契約崩れ（選択肢のトークン列がどの ID とも一致しない）は判定不能になる。"""
    label_order = ["cat_a", "cat_b"]
    broken = {"cat_a": _tokens("zzz"), "cat_b": _tokens("cat_b")}
    mapping = map_scores_to_choice(np.array([-0.1, -3.0]), label_order, broken)
    assert mapping == Unmapped(UnmappedReason.NO_CHOICE_MATCH)


def test_req19b_unmapped_rejects_non_enum_reason() -> None:
    """`UnmappedReason` 以外の reason は `TypeError`。"""
    with pytest.raises(TypeError):
        Unmapped("no_exact_match")  # type: ignore[arg-type]


def test_req19b_choice_mapping_tally_counts() -> None:
    """理由別に件数を数え、ログ行は件数だけを含む。"""
    tally = ChoiceMappingTally()
    mapped = Mapped(choice_id="cat_a", index=0, probs=(0.6, 0.4))
    tally.record(mapped)
    tally.record(mapped)
    tally.record(Unmapped(UnmappedReason.NO_CHOICE_MATCH))
    tally.record(Unmapped(UnmappedReason.INVALID_SCORE))
    assert (tally.mapped, tally.no_choice_match, tally.invalid_score) == (2, 1, 1)
    assert tally.to_log_line() == (
        "autoregressive choice mapping: mapped=2 no_choice_match=1 invalid_score=1"
    )


def test_req19b_predict_records_reports_no_choice_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """判定不能は `status:"error"` になり、tally の no_choice_match に数えられる。
    tally なしの呼び出しは挙動が変わらない。
    """
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(AutoregressiveKind(), make_examples(), req)
    rows = [(f"r{i}", f"text {i}") for i in range(3)]
    baseline = list(predict_records(trained, rows, resource_budget=trained.resource_budget))
    assert all(r["status"] == "ok" for r in baseline)

    import fandhe_edge_trainer.kinds.autoregressive as ar

    monkeypatch.setattr(
        ar, "resolve_generated_output", lambda *_a: Unmapped(UnmappedReason.NO_CHOICE_MATCH)
    )
    tally = ChoiceMappingTally()
    records = list(
        predict_records(trained, rows, resource_budget=trained.resource_budget, tally=tally)
    )
    assert records == [
        {"id": f"r{i}", "status": "error", "predicted_label": None} for i in range(3)
    ]
    assert (tally.mapped, tally.no_choice_match, tally.invalid_score) == (0, 3, 0)
