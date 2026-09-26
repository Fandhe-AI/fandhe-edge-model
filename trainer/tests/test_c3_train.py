"""C3（TASK-19.2 の既定候補）の学習・ONNX 書き出しの end-to-end テスト。CPU・小規模データ限定。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import onnx
import pytest

from conftest import TINY_CONFIG, export_onnx_to_path, make_examples, make_request
from fandhe_edge_trainer import artifact as artifact_mod
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode
from fandhe_edge_trainer.kinds.c3 import C3Kind


def _train_and_export(tmp_path: Path, seed: int = 0) -> Path:
    kind = C3Kind()
    req = make_request(tmp_path, seed=seed)
    examples = make_examples()
    trained = kind.train(examples, req)
    onnx_path = tmp_path / f"model_{seed}.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    return onnx_path


def test_c3_train_produces_valid_onnx_model(tmp_path: Path) -> None:
    """REQ-19b・TASK-19.2: 学習・ONNX 書き出しが onnx.checker を通ること。"""
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


def test_c3_artifact_fields(tmp_path: Path) -> None:
    """artifact.json が Rust 側 Artifact 構造体のフィールドと一致すること。"""
    kind = C3Kind()
    req = make_request(tmp_path)
    trained = kind.train(make_examples(), req)
    art = artifact_mod.build_artifact(
        kind=req.kind,
        kind_version=req.kind_version,
        config=trained.config,
        label_order=trained.label_order,
        output_type="choice",
        max_bytes=trained.max_bytes,
        candidate_label="c3",
    )
    assert art["kind"] == "c3"
    assert art["kind_version"] == 1
    assert art["selector_version"] == artifact_mod.SELECTOR_VERSION
    assert art["label_order"] == ["cat_a", "cat_b"]
    assert art["output_type"] == "choice"
    assert art["max_bytes"] == 64
    assert art["onnx_file"] == "model.onnx"
    assert art["candidate_label"] == "c3"
    assert art["config"]["emb"] == TINY_CONFIG["emb"]
    assert isinstance(art["created_utc"], str)
    assert art["created_utc"].endswith("Z")


def test_c3_training_is_deterministic_on_cpu(tmp_path: Path) -> None:
    """evaluation-contract: 同一 seed・CPU の学習結果は sha256 一致（証拠種別: テストハーネス）。"""
    onnx_a = _train_and_export(tmp_path / "run_a", seed=0)
    onnx_b = _train_and_export(tmp_path / "run_b", seed=0)

    def sha256(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    assert sha256(onnx_a) == sha256(onnx_b)


def test_c3_train_rejects_wrong_type_config(tmp_path: Path) -> None:
    """TASK-19.1: config の上書きが不正な型なら invalid_config（exit 64）で拒否する。"""
    kind = C3Kind()
    req = make_request(tmp_path, config={**TINY_CONFIG, "epochs": "not-an-int"})
    with pytest.raises(WorkerError) as exc_info:
        kind.train(make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_c3_train_rejects_even_width(tmp_path: Path) -> None:
    """偶数のカーネル幅は padding が出力長を +1 させ、REQ-28 の一致契約を壊すため拒否する。"""
    kind = C3Kind()
    req = make_request(tmp_path, config={**TINY_CONFIG, "widths": [3, 4]})
    with pytest.raises(WorkerError) as exc_info:
        kind.train(make_examples(), req)
    assert exc_info.value.code == "invalid_config"


def test_c3_train_rejects_config_upper_bound_violation(tmp_path: Path) -> None:
    """項目 2: emb・filters・epochs・batch_size は上限を超えると invalid_config で拒否する。"""
    from fandhe_edge_trainer.limits import MAX_C3_EMB

    kind = C3Kind()
    req = make_request(tmp_path, config={**TINY_CONFIG, "emb": MAX_C3_EMB + 1})
    with pytest.raises(WorkerError) as exc_info:
        kind.train(make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_c3_train_rejects_nan_lr(tmp_path: Path) -> None:
    """項目 2: lr が NaN の場合、範囲比較（`nan <= x` は常に False）をすり抜けず拒否する。"""
    kind = C3Kind()
    req = make_request(tmp_path, config={**TINY_CONFIG, "lr": float("nan")})
    with pytest.raises(WorkerError) as exc_info:
        kind.train(make_examples(), req)
    assert exc_info.value.code == "invalid_config"


def test_c3_train_rejects_infinite_lr(tmp_path: Path) -> None:
    kind = C3Kind()
    req = make_request(tmp_path, config={**TINY_CONFIG, "lr": float("inf")})
    with pytest.raises(WorkerError) as exc_info:
        kind.train(make_examples(), req)
    assert exc_info.value.code == "invalid_config"


def test_c3_train_rejects_too_many_widths(tmp_path: Path) -> None:
    from fandhe_edge_trainer.limits import MAX_C3_WIDTHS

    kind = C3Kind()
    widths = [2 * i + 1 for i in range(MAX_C3_WIDTHS + 1)]
    req = make_request(tmp_path, config={**TINY_CONFIG, "widths": widths})
    with pytest.raises(WorkerError) as exc_info:
        kind.train(make_examples(), req)
    assert exc_info.value.code == "invalid_config"


def test_c3_export_rejects_nonzero_pad_embedding_row(tmp_path: Path) -> None:
    """クラス docstring 3 番の不変条件（詰め物行は厳密ゼロ）を fail-closed で検証する。"""
    kind = C3Kind()
    req = make_request(tmp_path)
    trained = kind.train(make_examples(), req)

    w = np.array(trained.model.embed.weight, dtype=np.float32)
    w[0, 0] = 1.0  # 不変条件を意図的に壊す
    trained.model.embed.weight = mx.array(w)

    with pytest.raises(WorkerError) as exc_info:
        export_onnx_to_path(kind, trained, tmp_path / "broken.onnx")
    assert exc_info.value.code == "runtime_error"
    assert exc_info.value.exit_code == ExitCode.RUNTIME_ERROR


def test_c3_train_rejects_sample_steps_over_limit(tmp_path: Path) -> None:
    """P0-B: examples 件数 × epochs が上限を超える場合、1 バッチも回さず拒否する。

    `epochs` は `MAX_C3_EPOCHS`（config 検証の上限）に固定し、examples 件数の方を
    増やして総ステップ数を上限超過させる（バイトエンコードは検査より後に行われる
    ため、examples はダミーの入力文字列で十分。学習は 1 バッチも回らない）。
    """
    from fandhe_edge_trainer.contract import TrainExample
    from fandhe_edge_trainer.limits import MAX_C3_EPOCHS, MAX_TRAIN_SAMPLE_STEPS

    kind = C3Kind()
    epochs = MAX_C3_EPOCHS
    n_examples = MAX_TRAIN_SAMPLE_STEPS // epochs + 10
    examples = [TrainExample(input="alpha beta", label="cat_a") for _ in range(n_examples)]
    req = make_request(tmp_path, config={**TINY_CONFIG, "epochs": epochs})
    with pytest.raises(WorkerError) as exc_info:
        kind.train(examples, req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_c3_train_rejects_tiny_time_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P0-B: `time_limit_seconds` を極小にし、`time.monotonic` を進めて締切り超過を検出する。"""
    from fandhe_edge_trainer import budget as budget_mod

    real_monotonic = budget_mod.time.monotonic
    calls = {"n": 0}

    def _fake_monotonic() -> float:
        calls["n"] += 1
        # __post_init__ の締切り計算では実時刻を使い、2 回目以降の呼び出し
        # （学習ループ内の check()）で大きく先の時刻を返して締切りを超えさせる。
        if calls["n"] == 1:
            return real_monotonic()
        return real_monotonic() + 10_000.0

    monkeypatch.setattr(budget_mod.time, "monotonic", _fake_monotonic)

    kind = C3Kind()
    req = make_request(tmp_path, config=TINY_CONFIG, time_limit_seconds=1)
    with pytest.raises(WorkerError) as exc_info:
        kind.train(make_examples(), req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_c3_train_rejects_tiny_rss_limit(tmp_path: Path) -> None:
    """P0-B: `rss_limit_bytes` を極小（1 byte）にすると、実プロセスの RSS が必ず超過する。"""
    kind = C3Kind()
    req = make_request(tmp_path, config=TINY_CONFIG, rss_limit_bytes=1)
    with pytest.raises(WorkerError) as exc_info:
        kind.train(make_examples(), req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_c3_train_rejects_total_tokens_over_limit(tmp_path: Path) -> None:
    """P0-1: examples 件数 × max_bytes の見積もりが上限を超える場合、
    エンコード（Python のリストのリストとしての無制限確保）を一切行わず拒否する。
    """
    from fandhe_edge_trainer.contract import TrainExample
    from fandhe_edge_trainer.limits import MAX_MAX_BYTES, MAX_TRAIN_TOTAL_TOKENS

    kind = C3Kind()
    max_bytes = MAX_MAX_BYTES  # 4096
    n_examples = MAX_TRAIN_TOTAL_TOKENS // max_bytes + 10
    examples = [TrainExample(input="alpha beta", label="cat_a") for _ in range(n_examples)]
    req = make_request(tmp_path, config=TINY_CONFIG, max_bytes=max_bytes)
    with pytest.raises(WorkerError) as exc_info:
        kind.train(examples, req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_c3_train_encoding_is_resource_bounded_and_order_preserving(tmp_path: Path) -> None:
    """P0-1: 資源上限を検査したエンコード経路でも、決定性（同一 seed で同一モデル）が
    保たれること（`_encode_examples` が examples の並び順を変えないことの間接確認）。
    """
    kind = C3Kind()
    req_a = make_request(tmp_path / "a", seed=0)
    req_b = make_request(tmp_path / "b", seed=0)
    trained_a = kind.train(make_examples(), req_a)
    trained_b = kind.train(make_examples(), req_b)
    onnx_a = tmp_path / "a.onnx"
    onnx_b = tmp_path / "b.onnx"
    export_onnx_to_path(kind, trained_a, onnx_a)
    export_onnx_to_path(kind, trained_b, onnx_b)
    assert (
        hashlib.sha256(onnx_a.read_bytes()).hexdigest()
        == hashlib.sha256(onnx_b.read_bytes()).hexdigest()
    )


def test_c3_train_rejects_model_too_large_before_any_training_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P0-2: パラメータ数から見積もったモデルサイズが上限（40 MiB）を超える場合、
    `ByteCNN` を構築する前（＝ 1 バッチも学習する前）に拒否すること。
    """
    from fandhe_edge_trainer.kinds import c3 as c3_mod
    from fandhe_edge_trainer.limits import MAX_C3_EMB, MAX_C3_FILTERS

    def _must_not_be_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ByteCNN must not be instantiated when the model-size check fails")

    monkeypatch.setattr(c3_mod, "ByteCNN", _must_not_be_called)

    kind = C3Kind()
    # emb・filters とも許容上限（1024）まで上げると、既定の widths=[3,5,7] で
    # パラメータ数が 40 MiB（float32 換算）を大きく超える（実測 ≈ 61 MiB 相当）。
    req = make_request(
        tmp_path, config={**TINY_CONFIG, "emb": MAX_C3_EMB, "filters": MAX_C3_FILTERS}
    )
    with pytest.raises(WorkerError) as exc_info:
        kind.train(make_examples(), req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_c3_param_count_matches_actual_model() -> None:
    """`_c3_param_count` の解析的な見積もりが、実際に構築したモデルの
    パラメータ総数と一致すること（P0-2 の上限判定の前提が正しいことの確認）。
    """
    from mlx.utils import tree_flatten

    from fandhe_edge_trainer.kinds.c3 import ByteCNN, _c3_param_count

    n_classes, emb, filters, widths = 3, 8, 16, (3, 5, 7)
    model = ByteCNN(n_classes, emb, filters, widths, dropout=0.0)
    actual = sum(v.size for _, v in tree_flatten(model.parameters()))
    assert _c3_param_count(n_classes, emb, filters, widths) == actual
