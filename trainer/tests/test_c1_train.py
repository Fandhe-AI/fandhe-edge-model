"""C1（TASK-19.2 のもう一つの既定候補。バイト n-gram TF-IDF + ロジスティック回帰）の
学習・ONNX 書き出しの end-to-end テスト。CPU・小規模データ限定。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper

from conftest import (
    ATOL_BATCH_PARITY,
    TINY_C1_CONFIG,
    export_onnx_to_path,
    make_examples,
    make_request,
    train_kind,
)
from fandhe_edge_trainer import artifact as artifact_mod
from fandhe_edge_trainer.encoding import encode_bytes
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode
from fandhe_edge_trainer.kinds.c1 import C1Kind


def _train_and_export(
    tmp_path: Path, seed: int = 0, config: dict | None = None
) -> tuple[Path, int]:
    """学習・書き出しを行い、`(onnx_path, max_bytes)` を返す。

    呼び出し側が `encode_bytes` へ渡す `max_bytes` は、ここで実際に使った
    `TrainRequest.max_bytes` と必ず一致させる（ハードコードした定数を別途
    埋め込むと、`conftest.make_request` の既定値が変わった際にテストが
    暗黙に矛盾する）。
    """
    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", seed=seed, config=config or TINY_C1_CONFIG)
    examples = make_examples()
    trained = train_kind(kind, examples, req)
    onnx_path = tmp_path / f"model_{seed}.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    return onnx_path, req.max_bytes


def test_c1_train_produces_valid_onnx_model(tmp_path: Path) -> None:
    """REQ-19b・TASK-19.2: 学習・ONNX 書き出しが onnx.checker を通ること。"""
    onnx_path, _max_bytes = _train_and_export(tmp_path)
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


def test_c1_artifact_fields(tmp_path: Path) -> None:
    """artifact.json が Rust 側 Artifact 構造体のフィールドと一致し、語彙が config へ漏れない。"""
    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config=TINY_C1_CONFIG)
    trained = train_kind(kind, make_examples(), req)
    art = artifact_mod.build_artifact(
        kind=req.kind,
        kind_version=req.kind_version,
        config=trained.config,
        label_order=trained.label_order,
        output_type="choice",
        max_bytes=trained.max_bytes,
        candidate_label="c1",
        onnx_sha256="0" * 64,
    )
    assert art["kind"] == "c1"
    assert art["label_order"] == ["cat_a", "cat_b"]
    assert art["output_type"] == "choice"
    assert art["onnx_file"] == "model.onnx"
    assert art["candidate_label"] == "c1"
    # config にはハイパーパラメータのみを含み、語彙（データ由来の値）は含めない。
    assert set(art["config"]) == set(TINY_C1_CONFIG)
    assert art["config"]["ngram_max"] == TINY_C1_CONFIG["ngram_max"]


def test_c1_training_is_deterministic_on_cpu(tmp_path: Path) -> None:
    """evaluation-contract: 同一 seed・CPU の学習結果は sha256 一致（証拠種別: テストハーネス）。"""
    onnx_a, _ = _train_and_export(tmp_path / "run_a", seed=0)
    onnx_b, _ = _train_and_export(tmp_path / "run_b", seed=0)

    def sha256(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    assert sha256(onnx_a) == sha256(onnx_b)


def test_c1_train_accuracy_on_synthetic_data(tmp_path: Path) -> None:
    """合成データ（`conftest.make_examples`）で学習後の訓練正解率が十分高いこと。"""
    from onnx.reference import ReferenceEvaluator

    onnx_path, max_bytes = _train_and_export(tmp_path)
    session = ReferenceEvaluator(str(onnx_path))
    examples = make_examples()
    label_order = ["cat_a", "cat_b"]
    correct = 0
    for ex in examples:
        ids = encode_bytes(ex.input, max_bytes)
        (probs,) = session.run(None, {"ids": np.array([ids], dtype=np.int64)})
        pred = label_order[int(probs[0].argmax())]
        correct += int(pred == ex.label)
    accuracy = correct / len(examples)
    assert accuracy >= 0.9, f"train accuracy too low: {accuracy}"


def test_c1_train_rejects_wrong_type_config(tmp_path: Path) -> None:
    """TASK-19.1: config の上書きが不正な型なら invalid_config（exit 64）で拒否する。"""
    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config={**TINY_C1_CONFIG, "epochs": "not-an-int"})
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_c1_train_rejects_unknown_config_field(tmp_path: Path) -> None:
    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config={**TINY_C1_CONFIG, "unknown_field": 1})
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_c1_train_rejects_ngram_min_greater_than_max(tmp_path: Path) -> None:
    kind = C1Kind()
    req = make_request(
        tmp_path, kind="c1", config={**TINY_C1_CONFIG, "ngram_min": 4, "ngram_max": 2}
    )
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_c1_train_rejects_ngram_over_upper_bound(tmp_path: Path) -> None:
    from fandhe_edge_trainer.limits import MAX_C1_NGRAM

    kind = C1Kind()
    req = make_request(
        tmp_path,
        kind="c1",
        config={**TINY_C1_CONFIG, "ngram_max": MAX_C1_NGRAM + 1, "ngram_min": 1},
    )
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


@pytest.mark.parametrize(
    "field", ["ngram_min", "ngram_max", "min_df", "max_features", "epochs", "batch_size"]
)
@pytest.mark.parametrize("value", [True, False])
def test_c1_train_rejects_bool_for_integer_config_fields(
    tmp_path: Path, field: str, value: bool
) -> None:
    """P1: 真偽値は int のサブクラスであるため、config の整数フィールドの検証は
    明示的に bool を除外しないと素通りしうる（C3 の同名テストと同じ回帰確認）。
    """
    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config={**TINY_C1_CONFIG, field: value})
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_c1_train_rejects_nan_c(tmp_path: Path) -> None:
    """項目 2: C が NaN の場合、範囲比較（`nan <= x` は常に False）をすり抜けず拒否する。"""
    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config={**TINY_C1_CONFIG, "C": float("nan")})
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_c1_train_rejects_infinite_lr(tmp_path: Path) -> None:
    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config={**TINY_C1_CONFIG, "lr": float("inf")})
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_c1_train_rejects_zero_c(tmp_path: Path) -> None:
    """C は (0, MAX] の範囲。0 は下限の排他境界のため拒否する。"""
    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config={**TINY_C1_CONFIG, "C": 0.0})
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_c1_train_excludes_ngrams_containing_pad_token(tmp_path: Path) -> None:
    """語彙構築が詰め物（id=0）を含む n-gram を絶対に含めないこと（ngram_min=1）。

    空文字列は `contract.load_train_examples` を経由すれば拒否されるが、本関数
    （`C1Kind.train`）自体はその前提に頼らず、`contract` の検証を経ない経路
    （テストが `TrainExample` を直接構築して渡す経路）でも不変条件を保つ必要が
    ある。`encode_bytes("")` は詰め物専用の `[0]` を返す（`encoding.py`）ため、
    空文字列を多数含む学習データで語彙を構築し、`vocab.keys` を base-257 から
    復号して 0 を含まないことを直接確認する（P0 回帰: `_encode_examples` が
    この `[0]` 行の実長を誤って 1 と数えると、語彙が id=0 を含む n-gram を
    持ちうる）。
    """
    from fandhe_edge_trainer.contract import TrainExample
    from fandhe_edge_trainer.kinds.c1 import _decode_ngram_keys

    kind = C1Kind()
    examples = [TrainExample(input="", label="cat_a") for _ in range(20)] + [
        TrainExample(input="ab cd ef gh ij", label="cat_b") for _ in range(20)
    ]
    req = make_request(
        tmp_path, kind="c1", config={**TINY_C1_CONFIG, "ngram_min": 1, "ngram_max": 2}
    )
    trained = train_kind(kind, examples, req)
    for n in sorted(set(trained.vocab.ns.tolist())):
        mask = trained.vocab.ns == n
        keys_n = trained.vocab.keys[mask]
        if keys_n.size == 0:
            continue
        digits = _decode_ngram_keys(keys_n, int(n))
        assert not np.any(digits == 0)

    # 書き出しも成功し（不変条件違反として拒否されない）、空文字列（詰め物のみ
    # の行）に対する推論が単独・バッチのいずれでも一致すること（REQ-28）。
    onnx_path = tmp_path / "model.onnx"
    export_onnx_to_path(kind, trained, onnx_path)
    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator(str(onnx_path))
    empty_ids = encode_bytes("", req.max_bytes)
    other_ids = encode_bytes("ab cd ef gh ij", req.max_bytes)
    max_len = max(len(empty_ids), len(other_ids))
    batch = np.zeros((2, max_len), dtype=np.int64)
    batch[0, : len(empty_ids)] = empty_ids
    batch[1, : len(other_ids)] = other_ids
    (probs_batch,) = session.run(None, {"ids": batch})
    (probs_single,) = session.run(None, {"ids": np.array([empty_ids], dtype=np.int64)})
    assert np.max(np.abs(probs_single[0] - probs_batch[0])) <= ATOL_BATCH_PARITY


def test_c1_export_rejects_pad_token_in_vocabulary(tmp_path: Path) -> None:
    """クラス docstring の不変条件（語彙に詰め物 id=0 を含めない）を、書き出し側でも
    fail-closed で検証する。語彙を直接壊して不変条件違反を注入する。
    """
    import dataclasses

    from fandhe_edge_trainer.kinds.c1 import _export_c1_onnx

    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config=TINY_C1_CONFIG)
    trained = train_kind(kind, make_examples(), req)

    broken_keys = trained.vocab.keys.copy()
    broken_keys[0] = 0  # 復号すると全桁 0 になり、不変条件違反を表す
    broken_ns = trained.vocab.ns.copy()
    broken_ns[0] = 1
    broken_vocab = dataclasses.replace(trained.vocab, ns=broken_ns, keys=broken_keys)
    broken_trained = dataclasses.replace(trained, vocab=broken_vocab)

    with open(tmp_path / "broken.onnx", "wb") as f:
        with pytest.raises(WorkerError) as exc_info:
            _export_c1_onnx(broken_trained, f)
    assert exc_info.value.code == "runtime_error"
    assert exc_info.value.exit_code == ExitCode.RUNTIME_ERROR


def test_c1_train_rejects_empty_vocabulary(tmp_path: Path) -> None:
    """min_df が高すぎて語彙が空になる場合、invalid_data（exit 64）で拒否する。"""
    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config={**TINY_C1_CONFIG, "min_df": 1000})
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "invalid_data"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_c1_train_rejects_sample_steps_over_limit(tmp_path: Path) -> None:
    """P0-B: examples 件数 x epochs が上限を超える場合、1 バッチも回さず拒否する
    （C3 の同名テストと同じ理由・同じ構成）。
    """
    from fandhe_edge_trainer.contract import TrainExample
    from fandhe_edge_trainer.limits import MAX_C1_EPOCHS, MAX_TRAIN_SAMPLE_STEPS

    kind = C1Kind()
    epochs = MAX_C1_EPOCHS
    n_examples = MAX_TRAIN_SAMPLE_STEPS // epochs + 10
    examples = [TrainExample(input="alpha beta", label="cat_a") for _ in range(n_examples)]
    req = make_request(tmp_path, kind="c1", config={**TINY_C1_CONFIG, "epochs": epochs})
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, examples, req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_c1_train_rejects_total_tokens_over_limit(tmp_path: Path) -> None:
    """P0-1: examples 件数 x max_bytes の見積もりが上限を超える場合、
    エンコードを一切行わず拒否する（C3 の同名テストと同じ理由）。
    """
    from fandhe_edge_trainer.contract import TrainExample
    from fandhe_edge_trainer.limits import MAX_MAX_BYTES, MAX_TRAIN_TOTAL_TOKENS

    kind = C1Kind()
    max_bytes = MAX_MAX_BYTES  # 4096
    n_examples = MAX_TRAIN_TOTAL_TOKENS // max_bytes + 10
    examples = [TrainExample(input="alpha beta", label="cat_a") for _ in range(n_examples)]
    req = make_request(tmp_path, kind="c1", config=TINY_C1_CONFIG, max_bytes=max_bytes)
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, examples, req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_c1_train_rejects_model_too_large_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P0-2: 語彙構築直後、`TfidfLogReg` を構築する前にモデルサイズの上限検査で拒否すること。"""
    from fandhe_edge_trainer.kinds import c1 as c1_mod

    def _must_not_be_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("TfidfLogReg must not be instantiated when the size check fails")

    monkeypatch.setattr(c1_mod, "TfidfLogReg", _must_not_be_called)
    monkeypatch.setattr(c1_mod.budget_mod, "MAX_MODEL_BYTES", 1)

    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config=TINY_C1_CONFIG)
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_c1_max_features_bounds_vocabulary_size(tmp_path: Path) -> None:
    """max_features の上限が効き、語彙の列数がちょうどその値に切り詰められること。

    合成データ（`conftest.make_examples`）は ngram_min=ngram_max=1 で 3 より
    多い相異なるバイト値（`_TEXT_A`・`_TEXT_B` の文字種）を含むため、
    `max_features=3` は実際に効いて厳密に 3 列へ切り詰められる（`<=` ではなく
    `==` で検証し、上限が「たまたま下回っている」だけでないことを確認する）。
    """
    kind = C1Kind()
    req = make_request(
        tmp_path,
        kind="c1",
        config={**TINY_C1_CONFIG, "ngram_min": 1, "ngram_max": 1, "max_features": 3},
    )
    trained = train_kind(kind, make_examples(), req)
    assert trained.vocab.ns.shape[0] == 3


def test_c1_ngram_min_greater_than_one_achieves_high_training_accuracy(tmp_path: Path) -> None:
    """ngram_min > 1（例: 2..3）でも書き出した ONNX モデルの訓練正解率が十分高いこと。

    ONNX グラフの `ngram_counts` レイアウト自体が ngram_min > 1 で正しいことは
    `tests/test_c1_batch_parity.py::test_c1_single_vs_batch_inference_match_with_ngram_min_over_one`・
    `tests/test_c1_mlx_onnx_parity.py::test_mlx_forward_matches_onnx_reference_with_ngram_min_over_one`
    が数値一致で検証する（本テストの名前が示唆していた「ONNX 参照実装との
    一致」はそちら側の責務であり、本テストは ngram_min > 1 の設定でも実用上
    妥当な精度が出ることの確認に限定する）。
    """
    from onnx.reference import ReferenceEvaluator

    onnx_path, max_bytes = _train_and_export(
        tmp_path, config={**TINY_C1_CONFIG, "ngram_min": 2, "ngram_max": 3}
    )
    session = ReferenceEvaluator(str(onnx_path))
    examples = make_examples()
    label_order = ["cat_a", "cat_b"]
    correct = 0
    for ex in examples:
        ids = encode_bytes(ex.input, max_bytes)
        (probs,) = session.run(None, {"ids": np.array([ids], dtype=np.int64)})
        pred = label_order[int(probs[0].argmax())]
        correct += int(pred == ex.label)
    accuracy = correct / len(examples)
    assert accuracy >= 0.9, f"train accuracy too low: {accuracy}"


def test_c1_batch_element_cap_rejects_before_allocating_minibatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """セキュリティ監査 P0: ミニバッチのフォワードが確保する要素数
    （`batch_size x max_nnz x n_classes`）が上限を超える場合、1 バッチも
    確保せず `limit_exceeded`（exit 20）で拒否すること。

    他の既存チェック（sample_steps・total_tokens・SPARSE_ELEMENTS・
    model_bytes）をすべて通過する構成でも、この積だけが跳ね上がりうる
    ことを示すため、`MAX_C1_BATCH_ELEMENTS` を小さく monkeypatch して
    小規模データのままで再現する。
    """
    from fandhe_edge_trainer.kinds import c1 as c1_mod

    monkeypatch.setattr(c1_mod, "MAX_C1_BATCH_ELEMENTS", 1)

    def _must_not_be_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_build_sparse_features must not run when the batch cap fails")

    monkeypatch.setattr(c1_mod, "_build_sparse_features", _must_not_be_called)

    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config=TINY_C1_CONFIG)
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_c1_train_rejects_label_not_in_label_order(tmp_path: Path) -> None:
    """セキュリティ監査 P2-3: `label_id[ex.label]` の素の `KeyError`（exit 70 に
    化ける）ではなく、`invalid_data`（exit 64）として拒否すること。

    `contract.load_train_examples` を経由すれば通常ここでラベル不整合は
    弾かれているが、`C1Kind.train` はその前提に頼らない（テストは
    `TrainExample` を直接構築して `contract` の検証を経ない経路を使う）。
    """
    from fandhe_edge_trainer.contract import TrainExample

    kind = C1Kind()
    examples = [*make_examples(), TrainExample(input="unseen label text", label="cat_z")]
    req = make_request(tmp_path, kind="c1", config=TINY_C1_CONFIG)
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, examples, req)
    assert exc_info.value.code == "invalid_data"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
    assert "cat_z" not in exc_info.value.message  # データ本文（ラベル値）を漏らさない


def test_c1_ngram_key_space_does_not_overflow_int64() -> None:
    """セキュリティ監査 P2-2: `(n, base257 key)` の結合キーが int64 に収まる
    という不変条件を、実際の定数値から再計算して確認する（モジュール読み込み
    時の `if` チェック〔`kinds/c1.py`〕が検出すべき条件と同じ式を、テスト側でも
    独立に検証する回帰確認）。
    """
    from fandhe_edge_trainer.kinds.c1 import _COMBINE_BASE
    from fandhe_edge_trainer.limits import MAX_C1_NGRAM

    assert _COMBINE_BASE == 257**MAX_C1_NGRAM
    assert (MAX_C1_NGRAM + 1) * _COMBINE_BASE < 2**63


def test_c1_vocab_candidates_limit_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """セキュリティ監査 P1-2: `MAX_C1_VOCAB_CANDIDATES` を超えた時点で
    `limit_exceeded`（exit 20）を送出すること（monkeypatch で小さくして再現）。
    """
    from fandhe_edge_trainer.kinds import c1 as c1_mod

    monkeypatch.setattr(c1_mod, "MAX_C1_VOCAB_CANDIDATES", 2)

    kind = C1Kind()
    # 合成データ（`make_examples`）は ngram_min=ngram_max=1 で 2 個より多い
    # 相異なるバイト値を含むため、候補数の上限（2）をすぐに超える。
    req = make_request(
        tmp_path, kind="c1", config={**TINY_C1_CONFIG, "ngram_min": 1, "ngram_max": 1}
    )
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_c1_sparse_elements_limit_exceeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """セキュリティ監査 P1-2: `MAX_C1_SPARSE_ELEMENTS` を超えた時点で
    `limit_exceeded`（exit 20）を送出し、疎特徴量の確保（`_build_sparse_features`）
    を一切行わないこと。
    """
    from fandhe_edge_trainer.kinds import c1 as c1_mod

    monkeypatch.setattr(c1_mod, "MAX_C1_SPARSE_ELEMENTS", 1)

    def _must_not_be_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_build_sparse_features must not run when the sparse cap fails")

    monkeypatch.setattr(c1_mod, "_build_sparse_features", _must_not_be_called)

    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config=TINY_C1_CONFIG)
    with pytest.raises(WorkerError) as exc_info:
        train_kind(kind, make_examples(), req)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_c1_vocab_chunking_matches_unchunked_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """セキュリティ監査 P2-1: `_build_vocabulary` のチャンクサイズを動的に
    決める変更が、チャンク境界に関わらず同じ語彙を生成すること（1 チャンク
    あたりの目標窓要素数を極端に小さくし、多数の小チャンクに分割されても
    結果が変わらないことを確認する）。
    """
    from fandhe_edge_trainer.kinds import c1 as c1_mod

    kind = C1Kind()
    req = make_request(tmp_path, kind="c1", config=TINY_C1_CONFIG)
    baseline = train_kind(kind, make_examples(), req)

    monkeypatch.setattr(c1_mod, "MAX_C1_VOCAB_CHUNK_WINDOW_ELEMENTS", 1)
    req2 = make_request(tmp_path / "chunked", kind="c1", config=TINY_C1_CONFIG)
    chunked = train_kind(kind, make_examples(), req2)

    assert baseline.vocab.ns.tolist() == chunked.vocab.ns.tolist()
    assert baseline.vocab.keys.tolist() == chunked.vocab.keys.tolist()
    np.testing.assert_allclose(baseline.vocab.idf, chunked.vocab.idf, atol=1e-9)


def test_c1_golden_vocabulary_and_sparse_features_match_hand_computed_formula(
    tmp_path: Path,
) -> None:
    """レビュー P1-1: 3 件の小さな文書・ngram 1..2 で、語彙の idf と 1 文書分の
    疎特徴量を式（`kinds/c1.py` モジュール docstring の sublinear TF・idf・
    L2 正規化の式）から独立に計算した手計算値と具体値で突き合わせる。

    文書:
      - doc0="aa" -> バイト [97,97] -> トークン [98,98]
      - doc1="ab" -> バイト [97,98] -> トークン [98,99]
      - doc2="bb" -> バイト [98,98] -> トークン [99,99]

    unigram（n=1）の文書頻度: token 98 は doc0・doc1 に出現（df=2）、
    token 99 は doc1・doc2 に出現（df=2）。
    bigram（n=2）はいずれも 1 文書にしか出現しない（df=1）ため、語彙の列順
    （n 昇順・キー昇順）は [(1,98), (1,99), (2,(98,98)), (2,(98,99)), (2,(99,99))]。
    """
    import math

    from fandhe_edge_trainer import budget as budget_mod
    from fandhe_edge_trainer.contract import TrainExample
    from fandhe_edge_trainer.kinds.c1 import (
        _build_sparse_features,
        _build_vocabulary,
        _encode_examples,
        _vocab_tfidf_layout,
    )
    from fandhe_edge_trainer.limits import MAX_TRAIN_RSS_BYTES, MAX_TRAIN_WALL_SECONDS

    examples = [
        TrainExample(input="aa", label="cat_a"),
        TrainExample(input="ab", label="cat_b"),
        TrainExample(input="bb", label="cat_a"),
    ]
    budget = budget_mod.ResourceBudget(
        wall_seconds=float(MAX_TRAIN_WALL_SECONDS), rss_bytes=MAX_TRAIN_RSS_BYTES, device="cpu"
    )
    ids_arr, doc_lens = _encode_examples(examples, max_bytes=64, resource_budget=budget)
    assert doc_lens.tolist() == [2, 2, 2]

    ngram_min, ngram_max = 1, 2
    vocab = _build_vocabulary(
        ids_arr, doc_lens, ngram_min, ngram_max, min_df=1, max_features=1000, resource_budget=budget
    )

    key_98_98 = 98 * 257 + 98
    key_98_99 = 98 * 257 + 99
    key_99_99 = 99 * 257 + 99
    assert vocab.ns.tolist() == [1, 1, 2, 2, 2]
    assert vocab.keys.tolist() == [98, 99, key_98_98, key_98_99, key_99_99]

    n_docs = 3
    dfs = [2, 2, 1, 1, 1]
    expected_idf = np.array(
        [math.log((1.0 + n_docs) / (1.0 + df)) + 1.0 for df in dfs], dtype=np.float32
    )
    np.testing.assert_allclose(vocab.idf, expected_idf, atol=1e-6)

    max_nnz = 5
    idx_arr, val_arr = _build_sparse_features(
        ids_arr, doc_lens, vocab, ngram_min, ngram_max, max_nnz, budget
    )

    # doc0 = "aa"（トークン [98,98]）: unigram 98 が tf=2（重複する 2 つの窓）、
    # bigram (98,98) が tf=1。他の列は一致しない（tf=0）。
    w_uni98 = (1.0 + math.log(2)) * expected_idf[0]
    w_bi9898 = (1.0 + math.log(1)) * expected_idf[2]
    norm0 = math.sqrt(w_uni98**2 + w_bi9898**2)
    expected_row0 = np.zeros(max_nnz, dtype=np.float32)
    expected_row0[0] = w_uni98 / norm0  # 列 0 = (n=1, key=98)
    expected_row0[1] = w_bi9898 / norm0  # 列 2 = (n=2, key=98*257+98) が idx_arr[0][1] に入る
    expected_idx0 = np.array([0, 2, 0, 0, 0], dtype=np.int32)

    assert idx_arr[0].tolist() == expected_idx0.tolist()
    np.testing.assert_allclose(val_arr[0], expected_row0, atol=1e-6)

    # ONNX 側の TfIdfVectorizer（"tf" の生カウント）も同じ手計算値と一致すること。
    ngram_counts, pool_int64s = _vocab_tfidf_layout(vocab, ngram_min, ngram_max)
    tf_node = helper.make_node(
        "TfIdfVectorizer",
        ["ids"],
        ["tf"],
        mode="TF",
        min_gram_length=ngram_min,
        max_gram_length=ngram_max,
        max_skip_count=0,
        ngram_counts=[int(v) for v in ngram_counts],
        ngram_indexes=list(range(vocab.ns.shape[0])),
        pool_int64s=[int(v) for v in pool_int64s],
    )
    tf_graph = helper.make_graph(
        [tf_node],
        "tf_only",
        [helper.make_tensor_value_info("ids", onnx.TensorProto.INT64, ["N", "T"])],
        [helper.make_tensor_value_info("tf", onnx.TensorProto.FLOAT, ["N", vocab.ns.shape[0]])],
    )
    tf_model = helper.make_model(tf_graph, opset_imports=[helper.make_opsetid("", 13)])
    tf_model.ir_version = 8
    onnx.checker.check_model(tf_model)

    from onnx.reference import ReferenceEvaluator

    session = ReferenceEvaluator(tf_model)
    doc0_ids = np.array([[98, 98]], dtype=np.int64)
    (tf_out,) = session.run(None, {"ids": doc0_ids})
    np.testing.assert_allclose(tf_out[0], np.array([2.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))
