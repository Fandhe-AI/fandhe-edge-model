"""qwen2_lora の LoRA 追加学習（REQ-41・REQ-19・REQ-39・TASK-41.5-3・#426）。

極小の合成モデル（`test_qwen2_base_model.make_dir`）で完走を確かめる。実重み・GPU は使わない。
証拠種別: テストハーネス（CPU）。全件一致の決定性・範囲外エラーの網羅は #428。
"""

from __future__ import annotations

import random
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tools.poc26 import synthetic

from conftest import make_budget, make_examples, make_request
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode
from fandhe_edge_trainer.kinds import resolve_kind
from fandhe_edge_trainer.kinds.qwen2_lora import DEFAULT_CONFIG, _seed_all
from test_qwen2_base_model import expected, make_dir


def _cfg(tmp_path: Path, **over) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    # chat template と system prompt で 64 token を超えるため、位置数を広げた極小モデルにする
    d = make_dir(tmp_path, config={**synthetic.tiny_config(), "max_position_embeddings": 256})
    return {
        "base_model_dir": d.name,
        "base_model_sha256": expected(d),
        "dtype": "float32",
        "iters": 3,
        "batch_size": 2,
        "max_seq_length": 200,
        "lora_rank": 2,
        "lora_num_layers": 2,
        "lr": 0.01,
        **over,
    }


def _train(tmp_path: Path, seed: int = 0, n: int = 4, **over):
    req = make_request(
        tmp_path, kind="qwen2_lora", config=_cfg(tmp_path, **over), seed=seed, keep_root=True
    )
    return req, resolve_kind("qwen2_lora", 1).train(make_examples(n), req, make_budget(req))


def _lora(trained) -> dict[str, mx.array]:
    return {k: v for k, v in tree_flatten(trained.model.parameters()) if "lora_" in k}


def test_train_completes_and_only_lora_is_trainable(tmp_path: Path) -> None:
    """REQ-41・TASK-41.5-3: 合成モデルで学習が完走し、学習対象は LoRA 行列だけになる。"""
    req, t = _train(tmp_path)
    assert t.iters_done == 3
    assert np.isfinite(t.final_loss)
    assert t.config == {**DEFAULT_CONFIG, **req.config}
    assert t.label_order == req.label_order
    names = {k for k, _ in tree_flatten(t.model.trainable_parameters())}
    assert names
    assert all(k.endswith(("lora_a", "lora_b")) for k in names)
    assert any(float(mx.abs(v).max()) > 0 for k, v in _lora(t).items() if k.endswith("lora_b"))


def test_lora_settings_come_from_request(tmp_path: Path) -> None:
    """REQ-41: rank と対象層数が学習リクエストの config から効く。"""
    _, t = _train(tmp_path, lora_rank=3, lora_num_layers=1)
    a = {k: v for k, v in _lora(t).items() if k.endswith("lora_a")}
    assert len(a) == 7  # 1 ブロック x 7 Linear
    assert all(v.shape[1] == 3 for v in a.values())
    assert all(k.startswith("model.layers.1.") for k in a)  # 末尾ブロックだけ


def test_seed_all_sets_every_generator() -> None:
    """REQ-19・REQ-41: Python・NumPy・MLX のそれぞれを seed すると同じ列が出る。"""

    def draw() -> tuple[float, float, float]:
        return random.random(), float(np.random.rand()), float(mx.random.uniform().item())  # noqa: S311

    _seed_all(5)
    first = draw()
    _seed_all(5)
    assert draw() == first
    _seed_all(6)
    assert draw() != first


def test_lora_init_depends_on_seed(tmp_path: Path) -> None:
    """REQ-41: lora_a の初期値（lr を極小にして観測）は seed で決まる。"""
    _, t1 = _train(tmp_path / "a", seed=7, iters=1, lr=1e-12)
    _, t2 = _train(tmp_path / "b", seed=7, iters=1, lr=1e-12)
    _, t3 = _train(tmp_path / "c", seed=8, iters=1, lr=1e-12)
    key = "model.layers.1.self_attn.q_proj.lora_a"
    a1, a2, a3 = (_lora(t)[key] for t in (t1, t2, t3))
    assert mx.allclose(a1, a2, atol=1e-6).item()
    assert not mx.allclose(a1, a3, atol=1e-6).item()


@pytest.mark.parametrize(
    "over",
    [
        {"unknown_key": 1},
        {"lora_rank": 0},
        {"lora_num_layers": 99},
        {"dtype": "float16"},
        {"lr": float("nan")},
        {"base_model_dir": "../x"},
    ],
)
def test_invalid_input_is_rejected(tmp_path: Path, over: dict) -> None:
    """REQ-39・REQ-19: 不正な config・経路（`..`）は 64 で止まる。"""
    with pytest.raises(WorkerError) as e:
        _train(tmp_path, **over)
    assert e.value.exit_code == ExitCode.INVALID_INPUT


def test_missing_required_config_is_rejected(tmp_path: Path) -> None:
    """REQ-41: base_model_dir が無ければ invalid_config。"""
    req = make_request(tmp_path, kind="qwen2_lora", config={"iters": 1}, keep_root=True)
    with pytest.raises(WorkerError) as e:
        resolve_kind("qwen2_lora", 1).train(make_examples(2), req, make_budget(req))
    assert (e.value.code, e.value.exit_code) == ("invalid_config", ExitCode.INVALID_INPUT)


def test_too_few_examples_and_too_long_sequence(tmp_path: Path) -> None:
    """REQ-39: 件数不足は invalid_data(64)、長すぎる列は切り詰めず limit_exceeded(20)。"""
    with pytest.raises(WorkerError) as e:
        _train(tmp_path / "a", n=1, batch_size=4)  # 2 件 < 4
    assert (e.value.code, e.value.exit_code) == ("invalid_data", ExitCode.INVALID_INPUT)
    with pytest.raises(WorkerError) as e:
        _train(tmp_path / "b", max_seq_length=100)
    assert e.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_make_batches_keeps_remainder_with_longest() -> None:
    """端数バッチ（最長側）を捨てない（REQ-41・TASK-41.5-3）。"""
    from fandhe_edge_trainer.kinds.qwen2_lora import make_batches

    assert make_batches([5, 1, 3, 2, 4, 6, 7], 4) == [[1, 3, 2, 4], [0, 5, 6]]


def test_activation_bytes_estimate() -> None:
    """保持活性の見積もり式（REQ-39）。"""
    from types import SimpleNamespace

    from fandhe_edge_trainer.kinds.qwen2_lora import _activation_bytes

    c = SimpleNamespace(
        hidden_size=100, intermediate_size=200, num_attention_heads=4, vocab_size=50
    )
    cfg = {"batch_size": 2, "max_seq_length": 10, "lora_num_layers": 3}
    # 層ごとの保持活性＋損失計算の logits（3 本分）
    assert _activation_bytes(c, cfg) == 4 * 20 * (1000 + 600 + 40) * 3 + 12 * 20 * 50


def test_huge_int_config_is_invalid_config() -> None:
    """REQ-39: 10**400 は OverflowError でなく invalid_config(64)。"""
    from fandhe_edge_trainer.kinds.qwen2_lora import _num_in

    with pytest.raises(WorkerError) as e:
        _num_in({"lr": 10**400}, "lr", 0.0, 1.0, lo_open=True)
    assert e.value.exit_code == ExitCode.INVALID_INPUT


def test_total_tokens_checked_before_encoding() -> None:
    """REQ-39: 件数 x max_seq_length が上限超なら列を保持する前に limit_exceeded(20)。"""
    from types import SimpleNamespace

    from fandhe_edge_trainer.kinds.qwen2_lora import _build_sequences

    class Boom:
        def encode(self, *_a: object) -> list[int]:
            raise AssertionError("must not encode")

    with pytest.raises(WorkerError) as e:
        _build_sequences(
            [None] * 60_000,  # type: ignore[list-item]
            SimpleNamespace(label_order=["a"]),  # type: ignore[arg-type]
            Boom(),
            {"max_seq_length": 1000},
            10,
            None,  # type: ignore[arg-type]
        )
    assert e.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_export_onnx_is_not_implemented() -> None:
    """REQ-41・#431: export_onnx は未実装で runtime_error(70)。"""
    with pytest.raises(WorkerError) as e:
        resolve_kind("qwen2_lora", 1).export_onnx(None, None)  # type: ignore[arg-type]
    assert e.value.exit_code == ExitCode.RUNTIME_ERROR
