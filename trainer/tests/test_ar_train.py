"""autoregressive（REQ-19b・TASK-19b.1-1・#79）の学習・ONNX 書き出しの
end-to-end テスト。CPU・小規模データ限定（証拠種別: テストハーネス）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
import pytest

from conftest import (
    LABEL_ORDER,
    TINY_AR_CONFIG,
    export_onnx_to_path,
    make_examples,
    make_request,
    train_kind,
)
from fandhe_edge_trainer import artifact as artifact_mod
from fandhe_edge_trainer.encoding import encode_bytes
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode
from fandhe_edge_trainer.kinds.autoregressive import (
    AutoregressiveKind,
    _ar_param_count,
    _encode_choices,
    _reject_duplicate_choice_encoding,
    _score_choices_mlx,
)


def _train_and_export(tmp_path: Path, seed: int = 0) -> Path:
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG, seed=seed)
    examples = make_examples()
    trained = train_kind(kind, examples, req)
    onnx_path = tmp_path / f"model_{seed}.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    return onnx_path


def test_ar_train_produces_valid_onnx_model(tmp_path: Path) -> None:
    """REQ-19b・TASK-19b.1-1: 学習・ONNX 書き出しが onnx.checker を通り、
    入出力契約が C1・C3 と同じ（`ids: INT64 ["N","T"]` → `probs: FLOAT
    ["N", K]`）であること。
    """
    onnx_path = _train_and_export(tmp_path)
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    graph = model.graph
    assert [i.name for i in graph.input] == ["ids"]
    assert [o.name for o in graph.output] == ["probs"]

    ids_info = graph.input[0].type.tensor_type
    assert ids_info.elem_type == onnx.TensorProto.INT64
    ids_dims = [d.dim_param or d.dim_value for d in ids_info.shape.dim]
    assert ids_dims == ["N", "T"]

    probs_info = graph.output[0].type.tensor_type
    assert probs_info.elem_type == onnx.TensorProto.FLOAT
    probs_dims = [d.dim_param or d.dim_value for d in probs_info.shape.dim]
    assert probs_dims == ["N", 2]  # 2 == len(LABEL_ORDER)


def test_ar_artifact_fields(tmp_path: Path) -> None:
    """artifact.json が Rust 側 Artifact 構造体のフィールドと一致すること。"""
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(kind, make_examples(), req)
    art = artifact_mod.build_artifact(
        kind=req.kind,
        kind_version=req.kind_version,
        config=trained.config,
        label_order=trained.label_order,
        output_type="choice",
        max_bytes=trained.max_bytes,
        candidate_label="autoregressive",
        onnx_sha256="0" * 64,
    )
    assert art["kind"] == "autoregressive"
    assert art["kind_version"] == 1
    assert art["label_order"] == LABEL_ORDER
    assert art["output_type"] == "choice"
    assert art["onnx_file"] == "model.onnx"
    assert art["candidate_label"] == "autoregressive"
    assert art["config"]["dims"] == TINY_AR_CONFIG["dims"]


def test_ar_training_is_deterministic_on_cpu(tmp_path: Path) -> None:
    """evaluation-contract: 同一 seed・CPU の学習結果は sha256 一致（証拠種別: テストハーネス）。"""
    onnx_a = _train_and_export(tmp_path / "run_a", seed=0)
    onnx_b = _train_and_export(tmp_path / "run_b", seed=0)

    def sha256(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    assert sha256(onnx_a) == sha256(onnx_b)


def test_ar_train_learns_the_synthetic_task(tmp_path: Path) -> None:
    """golden: 極小モデルでも合成データ（自明に分離可能な 2 クラス）を学習でき、
    `_score_choices_mlx`（対応づけ (b) の MLX 実装）の argmax が全件正解する
    こと（モジュール docstring・実装計画 4 章「学習→_score_choices_mlx の
    正しさをテストで確認」に対応する golden テスト）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)

    choice_id_list = [trained.choice_ids_by_label[label] for label in LABEL_ORDER]
    id_lists = [encode_bytes(ex.input, req.max_bytes) for ex in examples]
    scores = _score_choices_mlx(trained.model, id_lists, choice_id_list)
    assert np.all(np.isfinite(scores))
    predicted = [LABEL_ORDER[i] for i in scores.argmax(axis=1)]
    gold = [ex.label for ex in examples]
    accuracy = sum(p == g for p, g in zip(predicted, gold, strict=True)) / len(gold)
    assert accuracy == 1.0


def test_ar_score_choices_handles_empty_input(tmp_path: Path) -> None:
    """REQ-28: `encode_bytes("") == [0]`（入力領域が全て詰め物になる行）でも
    `_score_choices_mlx` が有限値を返すこと（クラス docstring 3 番 (a) の
    対角成分の強制許可が無いと、この行の hidden state が NaN になり伝播する）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(kind, make_examples(), req)
    choice_id_list = [trained.choice_ids_by_label[label] for label in LABEL_ORDER]
    scores = _score_choices_mlx(trained.model, [encode_bytes("", req.max_bytes)], choice_id_list)
    assert np.all(np.isfinite(scores))


def test_ar_train_rejects_wrong_type_config(tmp_path: Path) -> None:
    """TASK-19.1: config の上書きが不正な型なら invalid_config（exit 64）で拒否する。"""
    kind = AutoregressiveKind()
    req = make_request(
        tmp_path, kind="autoregressive", config={**TINY_AR_CONFIG, "epochs": "not-an-int"}
    )
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


@pytest.mark.parametrize(
    "field", ["layers", "dims", "heads", "epochs", "batch_size", "warmup_steps"]
)
@pytest.mark.parametrize("value", [True, False])
def test_ar_train_rejects_bool_for_integer_config_fields(
    tmp_path: Path, field: str, value: bool
) -> None:
    """P1: `bool` は `int` のサブクラスのため、明示的に除外しないと素通りしうる
    （`kinds/c3.py` の同名テストと同じ回帰防止。REQ-39）。
    """
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config={**TINY_AR_CONFIG, field: value})
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_ar_train_rejects_dims_not_divisible_by_heads(tmp_path: Path) -> None:
    kind = AutoregressiveKind()
    req = make_request(
        tmp_path, kind="autoregressive", config={**TINY_AR_CONFIG, "dims": 9, "heads": 2}
    )
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"


def test_ar_train_rejects_config_upper_bound_violation(tmp_path: Path) -> None:
    from fandhe_edge_trainer.limits import MAX_AR_DIMS

    kind = AutoregressiveKind()
    req = make_request(
        tmp_path, kind="autoregressive", config={**TINY_AR_CONFIG, "dims": MAX_AR_DIMS + 1}
    )
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"


def test_ar_train_rejects_nan_lr(tmp_path: Path) -> None:
    kind = AutoregressiveKind()
    req = make_request(
        tmp_path, kind="autoregressive", config={**TINY_AR_CONFIG, "lr": float("nan")}
    )
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"


def test_ar_train_rejects_infinite_weight_decay(tmp_path: Path) -> None:
    kind = AutoregressiveKind()
    req = make_request(
        tmp_path, kind="autoregressive", config={**TINY_AR_CONFIG, "weight_decay": float("inf")}
    )
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"


def test_ar_train_rejects_unknown_config_field(tmp_path: Path) -> None:
    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config={**TINY_AR_CONFIG, "bogus_field": 1})
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"


def test_ar_train_rejects_sample_steps_over_limit(tmp_path: Path) -> None:
    """P0-B: examples 件数 × epochs が上限を超える場合、1 バッチも回さず拒否する。"""
    from fandhe_edge_trainer.contract import TrainExample
    from fandhe_edge_trainer.limits import MAX_AR_EPOCHS, MAX_TRAIN_SAMPLE_STEPS

    kind = AutoregressiveKind()
    epochs = MAX_AR_EPOCHS
    n_examples = MAX_TRAIN_SAMPLE_STEPS // epochs + 10
    examples = [TrainExample(input="alpha beta", label="cat_a") for _ in range(n_examples)]
    req = make_request(tmp_path, kind="autoregressive", config={**TINY_AR_CONFIG, "epochs": epochs})
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, examples, req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_ar_train_rejects_total_tokens_over_limit(tmp_path: Path) -> None:
    """P0-1: examples 件数 × max_bytes の見積もりが上限を超える場合、
    エンコードを一切行わず拒否する。
    """
    from fandhe_edge_trainer.contract import TrainExample
    from fandhe_edge_trainer.limits import MAX_MAX_BYTES, MAX_TRAIN_TOTAL_TOKENS

    kind = AutoregressiveKind()
    max_bytes = MAX_MAX_BYTES
    n_examples = MAX_TRAIN_TOTAL_TOKENS // max_bytes + 10
    examples = [TrainExample(input="alpha beta", label="cat_a") for _ in range(n_examples)]
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG, max_bytes=max_bytes)
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, examples, req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_ar_train_rejects_model_too_large_before_any_training_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P0-2: パラメータ数から見積もったモデルサイズが上限（40 MiB）を超える場合、
    `ByteDecoder` を構築する前（＝ 1 バッチも学習する前）に拒否すること。
    """
    from fandhe_edge_trainer.kinds import autoregressive as ar_mod

    def _must_not_be_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ByteDecoder must not be instantiated when the size check fails")

    monkeypatch.setattr(ar_mod, "ByteDecoder", _must_not_be_called)

    kind = AutoregressiveKind()
    req = make_request(
        tmp_path,
        kind="autoregressive",
        config={**TINY_AR_CONFIG, "layers": 8, "dims": 1024, "heads": 16},
    )
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_ar_train_rejects_attention_elements_over_limit(tmp_path: Path) -> None:
    """`MAX_AR_ATTENTION_ELEMENTS`: `batch_size x heads x max_len^2` の見積もりが
    上限を超える場合、1 バッチも回さず拒否すること。
    """
    from fandhe_edge_trainer.limits import MAX_AR_BATCH_SIZE

    kind = AutoregressiveKind()
    req = make_request(
        tmp_path,
        kind="autoregressive",
        config={**TINY_AR_CONFIG, "batch_size": MAX_AR_BATCH_SIZE, "heads": 2, "dims": 8},
        max_bytes=4096,
    )
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_ar_export_rejects_choice_logprob_elements_over_limit(tmp_path: Path) -> None:
    """`MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS`: 選択肢の展開幅（`k_classes x m x
    VOCAB_SIZE`）が上限を超える場合、グラフ構築前（N=1 でも過大な書き出しに
    なる時点）で拒否すること（セキュリティ監査 P0 指摘。PR #222）。

    実際の学習は極小データで行い、書き出し直前に `label_order`・
    `choice_ids_by_label` だけを合成の大量ラベルへ差し替えて上限超過を再現する
    （K x M x VOCAB_SIZE は学習後にモデルへ触れる前に判定できるため、この
    差し替えでも `_export_ar_onnx` の検査対象は変わらない）。
    """
    import dataclasses

    from fandhe_edge_trainer.kinds.autoregressive import _encode_choices
    from fandhe_edge_trainer.limits import MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS

    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(kind, make_examples(), req)

    # k_classes x m x VOCAB_SIZE (259) > MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS を
    # 満たす合成ラベル集合（1000 ラベル x 800 バイトの選択肢文字列）。
    huge_label_order = [f"label_{i}" + ("x" * 799) for i in range(1000)]
    huge_choice_ids_by_label = _encode_choices(huge_label_order)
    max_label_len = max(len(c) for c in huge_choice_ids_by_label.values())
    assert len(huge_label_order) * (max_label_len + 1) * 259 > MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS

    huge_trained = dataclasses.replace(
        trained,
        label_order=huge_label_order,
        choice_ids_by_label=huge_choice_ids_by_label,
        max_label_len=max_label_len,
    )

    with pytest.raises(WorkerError) as exc_info:
        export_onnx_to_path(kind, huge_trained, tmp_path / "should_not_be_written.onnx")
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED
    assert (
        not (tmp_path / "should_not_be_written.onnx").exists()
        or (tmp_path / "should_not_be_written.onnx").stat().st_size == 0
    )


def test_ar_export_attention_elements_formula() -> None:
    """`_ar_export_attention_elements`: `N x K x heads x layers x (T+1+M)^2`
    を具体値で計算すること（REQ-39・PR #222 レビュー指摘）。`N` を含む
    計算式であることを N=1・N=2 の 2 通りで確認する（推論ランタイム側が
    実測の `N`・`T` をそのまま渡せる形にするため。`limits.py::
    MAX_AR_EXPORT_ATTENTION_ELEMENTS` docstring 参照）。

    `k_classes=3, t_bound=10, m=4, layers=2, heads=4` のとき
    `L = 10+1+4 = 15`・`N=1: 3 x 4 x 2 x 15^2 = 5400`・
    `N=2: 2 x 5400 = 10800`。
    """
    from fandhe_edge_trainer.kinds.autoregressive import _ar_export_attention_elements

    assert (
        _ar_export_attention_elements(n=1, k_classes=3, t_bound=10, m=4, layers=2, heads=4) == 5400
    )
    assert (
        _ar_export_attention_elements(n=2, k_classes=3, t_bound=10, m=4, layers=2, heads=4) == 10800
    )


def test_ar_export_rejects_attention_elements_over_limit(tmp_path: Path) -> None:
    """`MAX_AR_EXPORT_ATTENTION_ELEMENTS`: `T`（動的軸）を、密な入力（詰め物
    を含まない入力）における構造上の上限（学習時の `max_bytes`）まで
    広げたときの推論 1 件（N=1）あたりの decoder attention 要素数
    （`K x heads x layers x (T+1+M)^2`）が上限を超える場合、グラフ構築前に
    拒否すること（REQ-39・PR #222 レビュー指摘。ONNX 推論時の動的入力 N・T
    に資源上限が無いという P0 指摘への対応。詰め物を多く含む長い入力・
    N>1 の上限は本検査では検査できない。`limits.py::
    MAX_AR_EXPORT_ATTENTION_ELEMENTS` docstring 参照）。

    実際の学習は極小データ（`TINY_AR_CONFIG`）で行い、書き出し直前に
    `max_bytes` だけを大きな合成値へ差し替えて T の構造上の上限を広げる
    （`_export_ar_onnx` は `trained.max_bytes` を `t_bound` としてこの検査に
    使うため、この差し替えでも `_export_ar_onnx` の検査対象は変わらない）。
    合成した `max_bytes=20000` は、K x M x VOCAB_SIZE で決まる
    `MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS` の検査（T に依存しない）は
    超えないが、T を含む attention 要素数の検査は超える値として選んでいる。
    """
    import dataclasses

    from fandhe_edge_trainer.kinds.autoregressive import _ar_export_attention_elements
    from fandhe_edge_trainer.limits import (
        MAX_AR_EXPORT_ATTENTION_ELEMENTS,
        MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS,
    )

    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    trained = train_kind(kind, make_examples(), req)

    huge_max_bytes = 20000
    m = trained.max_label_len + 1
    k_classes = len(trained.label_order)
    heads = int(trained.config["heads"])
    layers = int(trained.config["layers"])

    assert k_classes * m * 259 <= MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS
    elements = _ar_export_attention_elements(1, k_classes, huge_max_bytes, m, layers, heads)
    assert elements == 1_601_120_196
    assert elements > MAX_AR_EXPORT_ATTENTION_ELEMENTS

    huge_trained = dataclasses.replace(trained, max_bytes=huge_max_bytes)

    with pytest.raises(WorkerError) as exc_info:
        export_onnx_to_path(kind, huge_trained, tmp_path / "should_not_be_written2.onnx")
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED
    assert (
        not (tmp_path / "should_not_be_written2.onnx").exists()
        or (tmp_path / "should_not_be_written2.onnx").stat().st_size == 0
    )


def test_ar_encode_choices_assigns_distinct_sequences() -> None:
    """`_encode_choices` が相異なるラベルへ相異なる系列を割り当てること。"""
    ids_by_label = _encode_choices(["a", "b"])
    assert ids_by_label["a"] != ids_by_label["b"]
    assert ids_by_label["a"] == [ord("a") + 1]


def test_reject_duplicate_choice_encoding_rejects_collision() -> None:
    """`_reject_duplicate_choice_encoding`（UTF-8 の単射性のため `_encode_choices`
    経由では実際には起こり得ない衝突経路を、独立した純粋関数として直接検証する）。
    """
    with pytest.raises(WorkerError) as exc_info:
        _reject_duplicate_choice_encoding(["a", "b"], [[1, 2], [1, 2]])
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
    assert "label_order[1]" in exc_info.value.message


def test_reject_duplicate_choice_encoding_accepts_distinct_sequences() -> None:
    _reject_duplicate_choice_encoding(["a", "b"], [[1, 2], [3, 4]])  # 例外を送出しない


def test_ar_param_count_matches_actual_model() -> None:
    """`_ar_param_count` の解析的な見積もりが、実際に構築したモデルの
    パラメータ総数と一致すること（P0-2 の上限判定の前提が正しいことの確認）。
    """
    from mlx.utils import tree_flatten

    from fandhe_edge_trainer.kinds.autoregressive import ByteDecoder

    layers, dims, heads, max_len = 2, 16, 4, 37
    model = ByteDecoder(layers, dims, heads, dropout=0.0, max_len=max_len)
    actual = sum(v.size for _, v in tree_flatten(model.parameters()))
    assert _ar_param_count(layers, dims, heads, max_len) == actual


def test_ar_train_rejects_tiny_time_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P0-B: `time_limit_seconds` を極小にし、`time.monotonic` を進めて締切り超過を検出する。"""
    from fandhe_edge_trainer import budget as budget_mod

    real_monotonic = budget_mod.time.monotonic
    calls = {"n": 0}

    def _fake_monotonic() -> float:
        calls["n"] += 1
        if calls["n"] == 1:
            return real_monotonic()
        return real_monotonic() + 10_000.0

    monkeypatch.setattr(budget_mod.time, "monotonic", _fake_monotonic)

    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG, time_limit_seconds=1)
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_ar_train_diverges_is_reported_as_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """損失が有限でなくなった場合、`training_diverged`（PENDING=12）として
    報告すること（`kinds/c3.py::C3Kind.train` と同じ対応づけ）。
    """
    import mlx.core as mx

    from fandhe_edge_trainer.kinds import autoregressive as ar_mod

    monkeypatch.setattr(ar_mod, "_is_finite_loss", lambda _v: False)

    kind = AutoregressiveKind()
    req = make_request(tmp_path, kind="autoregressive", config=TINY_AR_CONFIG)
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "training_diverged"
    assert exc_info.value.exit_code == ExitCode.PENDING
    mx.set_default_device(mx.cpu)  # 後続テストへ非決定的な device 状態を残さない
