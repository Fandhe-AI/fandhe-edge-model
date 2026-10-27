"""qwen2_lora の選択肢スコアリングによる予測（REQ-41・REQ-19b・REQ-39・TASK-41.5-4・#427）。

極小の合成モデルで確かめる。証拠種別: テストハーネス（CPU）。
決定性・範囲外エラーは test_qwen2_lora_train（#428）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import fandhe_edge_trainer.kinds.autoregressive as ar
import fandhe_edge_trainer.kinds.qwen2_lora as q
from fandhe_edge_trainer import predict
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode
from fandhe_edge_trainer.prediction import PREDICTION_FIELDS
from test_qwen2_lora_train import _train

ROWS = [("a", "short text one"), ("b", "another input"), ("c", "third")]


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory):
    return _train(tmp_path_factory.mktemp("q"), iters=1)[1]


def test_predictions_are_choice_ids_or_error(trained, capsys) -> None:
    """REQ-41・REQ-19b: 予測は label_order の要素か status:"error" だけ。"""
    recs = predict.predict_validation("qwen2_lora", trained, ROWS, trained.resource_budget)
    assert [r["id"] for r in recs] == ["a", "b", "c"]
    for r in recs:
        assert list(r) == list(PREDICTION_FIELDS)
        assert r["status"] == "ok"
        assert r["predicted_label"] in trained.label_order
    assert capsys.readouterr().err == (
        "qwen2_lora choice mapping: mapped=3 no_choice_match=0 invalid_score=0\n"
    )


def test_scores_are_posteriors_and_argmax(trained) -> None:
    """REQ-19b: scores は label_order 順で合計 1、predicted_label は最大。"""
    for r in q.predict_records(trained, ROWS):
        assert list(r["scores"]) == trained.label_order
        assert math.isclose(sum(r["scores"].values()), 1.0, abs_tol=1e-6)
        assert r["predicted_label"] == max(r["scores"], key=r["scores"].get)


def test_non_finite_score_is_error_without_leaking(trained, monkeypatch, capsys) -> None:
    """REQ-19b: 非有限スコアは status:"error"。stderr は件数のみ。"""
    n = len(trained.label_order)
    monkeypatch.setattr(q, "score_labels", lambda *_a, **_k: np.full(n, np.nan))
    recs = predict.predict_validation("qwen2_lora", trained, ROWS, trained.resource_budget)
    assert recs[0] == {"id": "a", "status": "error", "predicted_label": None}
    assert capsys.readouterr().err == (
        "qwen2_lora choice mapping: mapped=0 no_choice_match=0 invalid_score=3\n"
    )


def test_unresolvable_choice_is_error(trained, monkeypatch, capsys) -> None:
    """REQ-19b: 選択肢 ID に解決できなければ no_choice_match に計上して error。"""
    monkeypatch.setattr(
        ar, "resolve_generated_output", lambda *_a: ar.Unmapped(ar.UnmappedReason.NO_CHOICE_MATCH)
    )
    recs = predict.predict_validation("qwen2_lora", trained, ROWS, trained.resource_budget)
    assert {r["status"] for r in recs} == {"error"}
    assert "no_choice_match=3" in capsys.readouterr().err


def test_too_long_input_stops_before_scoring(trained, monkeypatch) -> None:
    """REQ-39: max_seq_length 超過は切り詰めず limit_exceeded。採点前に止まる。"""

    def boom(*_a, **_k):
        raise AssertionError("scored")

    monkeypatch.setattr(q, "score_labels", boom)
    with pytest.raises(WorkerError) as e:
        q.predict_records(trained, [("x", "word " * 400)])
    assert e.value.exit_code == ExitCode.LIMIT_EXCEEDED
    assert "word" not in str(e.value)


def test_chunking_does_not_change_scores(trained) -> None:
    """REQ-28 の前段: チャンク分割は丸めの範囲でしか結果を変えない。"""
    tok = trained.tokenizer
    end = tok.special_ids["im_end"]
    ids = [[*tok.encode(x), end] for x in trained.label_order]
    system = q._system_prompt(trained.label_order, trained.config)
    prompt = tok.build_chat_ids(system, "hi", add_generation_prompt=True)
    pad = tok.special_ids["endoftext"]
    a = q.score_labels(trained.model, prompt, ids, pad, chunk=1)
    b = q.score_labels(trained.model, prompt, ids, pad, chunk=8)
    assert np.allclose(a, b, atol=1e-4)
    assert int(a.argmax()) == int(b.argmax())


def test_single_and_batch_predictions_match(trained) -> None:
    """REQ-28: 長さの異なる複数入力で、1 件ずつの結果と一括の結果が全件一致する。"""
    rows = [*ROWS, ("d", "word " * 6), ("e", "x")]
    batch = q.predict_records(trained, rows)
    single = [r for row in rows for r in q.predict_records(trained, [row])]
    keys = ("id", "status", "predicted_label")
    assert [{k: r[k] for k in keys} for r in single] == [{k: r[k] for k in keys} for r in batch]
