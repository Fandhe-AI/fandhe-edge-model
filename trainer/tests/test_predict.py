"""学習直後の validation 予測（REQ-18・REQ-19・REQ-27。issue #84 PR #238・選択肢 2）。

`predict.py` の振り分け・`kinds/c1.py::predict_labels`・`kinds/c3.py::predict_labels`
の予測が、(1) 書き出した ONNX（`onnx.reference.ReferenceEvaluator`）と同じラベルを
返すこと、(2) チャンクの分け方（バッチ）に依存しないこと、(3) 正解ラベルを受け取らず
`{id, status, predicted_label}` だけを返すこと（証拠種別: テストハーネス・CPU）。
CLI 経由の確認は `tests/test_cli.py`。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from onnx.reference import ReferenceEvaluator

from conftest import (
    LABEL_ORDER,
    TINY_AR_CONFIG,
    TINY_C1_CONFIG,
    TINY_CONFIG,
    export_onnx_to_path,
    make_examples,
    make_request,
    train_kind,
)
from fandhe_edge_trainer import predict
from fandhe_edge_trainer.encoding import encode_bytes
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode
from fandhe_edge_trainer.kinds.autoregressive import AutoregressiveKind, predict_records
from fandhe_edge_trainer.kinds.c1 import C1Kind
from fandhe_edge_trainer.kinds.c3 import C3Kind
from fandhe_edge_trainer.prediction import PREDICTION_FIELDS

#: 学習データの分布に近い入力（cat_a／cat_b の境界が明確なもの）と短い入力。
_TEXTS = [
    "alpha alpha beta gamma 3",
    "delta delta epsilon zeta 5",
    "alpha beta gamma",
    "epsilon zeta delta",
    "short",
    "alpha alpha beta gamma 11",
    "delta delta epsilon zeta 0",
]

_KINDS = [
    pytest.param("c1", C1Kind, TINY_C1_CONFIG, id="c1"),
    pytest.param("c3", C3Kind, TINY_CONFIG, id="c3"),
]


def _rows(texts: list[str]) -> list[tuple[str, str]]:
    return [(f"rec-{i}", text) for i, text in enumerate(texts)]


def _onnx_labels(onnx_path: Path, texts: list[str], max_bytes: int) -> list[str]:
    id_lists = [encode_bytes(t, max_bytes) for t in texts]
    padded = np.zeros((len(id_lists), max(len(x) for x in id_lists)), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        padded[i, : len(ids)] = ids
    (probs,) = ReferenceEvaluator(str(onnx_path)).run(None, {"ids": padded})
    return [LABEL_ORDER[int(i)] for i in probs.argmax(axis=1)]


@pytest.mark.parametrize(("kind_name", "kind_cls", "config"), _KINDS)
def test_req19_training_time_prediction_matches_exported_onnx(
    tmp_path: Path, kind_name: str, kind_cls: type, config: dict
) -> None:
    """REQ-19・REQ-28: 学習時の予測（MLX 順伝播の argmax）と、書き出した ONNX を
    `ReferenceEvaluator` で評価した argmax が、全入力で同じラベルになること。
    """
    kind = kind_cls()
    req = make_request(tmp_path, kind=kind_name, config=config)
    trained = train_kind(kind, make_examples(), req)
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)

    records = predict.predict_validation(kind_name, trained, _rows(_TEXTS), trained.resource_budget)
    expected = _onnx_labels(onnx_path, _TEXTS, req.max_bytes)
    assert [r["predicted_label"] for r in records] == expected
    # 合成データは自明に分離可能なため、分布内の入力は正しいクラスへ入る。
    assert [r["predicted_label"] for r in records[:2]] == ["cat_a", "cat_b"]
    assert [r["id"] for r in records] == [f"rec-{i}" for i in range(len(_TEXTS))]
    assert all(list(r) == list(PREDICTION_FIELDS) and r["status"] == "ok" for r in records)


@pytest.mark.parametrize(("kind_name", "kind_cls", "config"), _KINDS)
def test_req28_prediction_does_not_depend_on_chunking(
    tmp_path: Path, kind_name: str, kind_cls: type, config: dict
) -> None:
    """REQ-28: 1 件ずつの予測と、複数チャンクに分かれる一括予測が全件一致する
    （`batch_size`=8 の設定で 24 件 = 3 チャンク）。
    """
    req = make_request(tmp_path, kind=kind_name, config=config)
    trained = train_kind(kind_cls(), make_examples(), req)
    texts = [ex.input for ex in make_examples()]
    rows = _rows(texts)
    together = predict.predict_validation(kind_name, trained, rows, trained.resource_budget)
    one_by_one = [
        predict.predict_validation(kind_name, trained, [row], trained.resource_budget)[0]
        for row in rows
    ]
    assert together == one_by_one
    assert len(together) == 24


def test_req27_autoregressive_records_drop_scores_and_match_predict_records(
    tmp_path: Path,
) -> None:
    """REQ-27: autoregressive は既存の `predict_records` の結果から `scores` を
    除いた `{id, status, predicted_label}` を返し、ラベル・状態は変わらない。
    """
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(AutoregressiveKind(), make_examples(), req)
    rows = _rows(_TEXTS[:4])
    records = predict.predict_validation("autoregressive", trained, rows, trained.resource_budget)
    reference = list(predict_records(trained, rows, resource_budget=trained.resource_budget))
    assert len(records) == len(reference) == 4
    for record, ref in zip(records, reference, strict=True):
        assert list(record) == list(PREDICTION_FIELDS)
        assert record == {key: ref[key] for key in PREDICTION_FIELDS}
        assert "scores" not in record


def test_req18_unsupported_kind_is_rejected_as_invalid_request() -> None:
    """REQ-18: 予測経路を持たない kind は `invalid_request`／exit 64 で拒否する。"""
    with pytest.raises(WorkerError) as exc_info:
        predict.require_validation_prediction_support("bogus")
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
    for supported in ("c1", "c3", "autoregressive"):
        predict.require_validation_prediction_support(supported)


def test_req19b_predict_validation_logs_undecidable_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-19b・TASK-19b.2: 判定不能は `status:"error"` で返り、stderr には
    件数の 1 行だけが出る（id・入力本文は出さない）。
    """
    import fandhe_edge_trainer.kinds.autoregressive as ar

    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(AutoregressiveKind(), make_examples(), req)
    monkeypatch.setattr(
        ar,
        "resolve_generated_output",
        lambda *_a: ar.Unmapped(ar.UnmappedReason.NO_CHOICE_MATCH),
    )
    rows = [("marker-id-1", "secret-body-1"), ("marker-id-2", "secret-body-2")]
    capsys.readouterr()
    records = predict.predict_validation("autoregressive", trained, rows, trained.resource_budget)
    assert records == [
        {"id": "marker-id-1", "status": "error", "predicted_label": None},
        {"id": "marker-id-2", "status": "error", "predicted_label": None},
    ]
    err = capsys.readouterr().err
    assert err == "autoregressive choice mapping: mapped=0 no_choice_match=2 invalid_score=0\n"
