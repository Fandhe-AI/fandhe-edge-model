"""PoC-26 Qwen2 モデル・LoRA の検査（REQ-41・TASK-41.1-5・#390。テストハーネス・CPU）。

極小の合成モデル（`tools/poc26/synthetic.py`）で挙動を固定する。実重みは使わない。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.poc26 import qwen2_model, synthetic
from tools.poc26.qwen2_model import (
    LoRALinear,
    apply_lora,
    load_qwen2,
    trainable_parameter_count,
)

IDS = mx.array([[1, 5, 9, 20, 7], [3, 3, 8, 2, 4]])


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(mx.cpu)  # 後続テストへ device 状態を残さない


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return synthetic.write_model_dir(tmp_path)


def test_load_and_forward_shape(model_dir: Path) -> None:
    """REQ-41: 読み込めて logits は [B, L, vocab]。"""
    m = load_qwen2(model_dir, dtype=mx.float32)
    out = m(IDS)
    assert out.shape == (2, 5, synthetic.TINY_VOCAB)
    assert bool(mx.all(mx.isfinite(out)))


def test_bf16_forward_on_cpu(model_dir: Path) -> None:
    """REQ-41: bf16 読み込みが CPU で forward できる（PoC の CPU 決定性確認の前提）。"""
    m = load_qwen2(model_dir, dtype=mx.bfloat16)
    out = m(IDS)
    assert out.dtype == mx.bfloat16
    assert out.shape == (2, 5, synthetic.TINY_VOCAB)


def _mk(p: Path) -> Path:
    p.mkdir()
    return p


def test_tie_uses_embedding_and_untied_uses_lm_head(tmp_path: Path) -> None:
    """REQ-41: tie では lm_head が無く logits は埋め込み行列との積。非 tie は lm_head を使う。"""
    m = load_qwen2(synthetic.write_model_dir(_mk(tmp_path / "t")), dtype=mx.float32)
    assert not hasattr(m, "lm_head")
    h = m.model.embed_tokens(IDS)
    for layer in m.model.layers:
        h = layer(h, "causal")
    expected = m.model.norm(h) @ m.model.embed_tokens.weight.T
    assert mx.allclose(m(IDS), expected, atol=1e-5).item()
    d = synthetic.write_model_dir(_mk(tmp_path / "u"), tie=False)
    u = load_qwen2(d, dtype=mx.float32)
    assert u.lm_head.weight.shape == (synthetic.TINY_VOCAB, 16)
    assert u(IDS).shape == (2, 5, synthetic.TINY_VOCAB)


def test_lora_initially_equals_base(model_dir: Path) -> None:
    """REQ-41: LoRA 適用直後（b = 0）は base と出力一致（atol 1e-6）。"""
    m = load_qwen2(model_dir, dtype=mx.float32)
    before = m(IDS)
    apply_lora(m, num_layers=2, rank=8, scale=20.0, dropout=0.0)
    assert isinstance(m.model.layers[1].self_attn.q_proj, LoRALinear)
    assert mx.allclose(before, m(IDS), atol=1e-6).item()


def test_lora_a_range_dtype_and_b_zero(model_dir: Path) -> None:
    """REQ-41: a は U(±1/sqrt(in))・b は 0・float32（bf16 モデルでも）。"""
    m = load_qwen2(model_dir, dtype=mx.bfloat16)
    apply_lora(m, num_layers=1, rank=8, scale=20.0, dropout=0.0)
    lin = m.model.layers[1].mlp.down_proj  # in = 32
    assert lin.lora_a.dtype == mx.float32
    assert lin.lora_b.dtype == mx.float32
    assert lin.lora_a.shape == (32, 8)
    assert lin.lora_b.shape == (8, 16)
    assert mx.max(mx.abs(lin.lora_a)).item() <= 32**-0.5
    assert mx.max(mx.abs(lin.lora_b)).item() == 0.0
    assert not isinstance(m.model.layers[0].self_attn.q_proj, LoRALinear)


@pytest.mark.parametrize(("num_layers", "expected"), [(1, 2048), (2, 4096)])
def test_trainable_parameter_count(model_dir: Path, num_layers: int, expected: int) -> None:
    """REQ-41: 層数×7 Linear×rank×(in+out)。q/o 16→16・k/v 16→8・gate/up 16→32・down 32→16"""
    m = load_qwen2(model_dir, dtype=mx.float32)
    apply_lora(m, num_layers=num_layers, rank=8, scale=20.0, dropout=0.0)
    per_layer = 8 * ((16 + 16) + (16 + 8) * 2 + (16 + 16) + (16 + 32) * 2 + (32 + 16))
    assert per_layer * num_layers == expected
    assert trainable_parameter_count(m) == expected


def test_lora_nonzero_b_changes_output_and_grads_only_lora(model_dir: Path) -> None:
    """REQ-41: b を動かすと出力が変わり、勾配の対象は LoRA のみ。"""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    m = load_qwen2(model_dir, dtype=mx.float32)
    apply_lora(m, num_layers=1, rank=4, scale=20.0, dropout=0.0)
    names = [k for k, _ in tree_flatten(m.trainable_parameters())]
    assert names
    assert all(("lora_a" in k or "lora_b" in k) for k in names)
    before = m(IDS)
    lin = m.model.layers[1].self_attn.q_proj
    lin.lora_b = mx.ones_like(lin.lora_b) * 0.1
    assert not mx.allclose(before, m(IDS), atol=1e-4).item()
    loss_grad = nn.value_and_grad(m, lambda mm: mm(IDS).sum())
    _, grads = loss_grad(m)
    assert {k for k, _ in tree_flatten(grads)} == set(names)


@pytest.mark.parametrize("pad_id", [0, 7, 200])
def test_pad_mask_matches_unpadded(model_dir: Path, pad_id: int) -> None:
    """REQ-41: 右 pad（id は任意）+ マスクで、実 token 位置の logits が pad なしと一致する。"""
    m = load_qwen2(model_dir, dtype=mx.float32)
    short = IDS[:1, :3]
    padded = mx.concatenate([short, mx.array([[pad_id, pad_id]])], axis=1)
    am = mx.array([[1, 1, 1, 0, 0]])
    got = m(padded, am)
    assert mx.allclose(m(short), got[:, :3], atol=1e-5).item()
    assert mx.allclose(m(IDS), m(IDS, mx.ones((2, 5), dtype=mx.int32)), atol=1e-5).item()
    assert bool(mx.all(mx.isfinite(got)))


def _np_reference(wdir: Path, ids: list[int]) -> np.ndarray:
    """独立参照（numpy float64）。RMSNorm・bias・RoPE half-split・GQA repeat・causal・SwiGLU。"""
    cfg = json.loads((wdir / "config.json").read_text())
    w = {
        k: np.array(v, dtype=np.float64)
        for k, v in mx.load(str(wdir / "model.safetensors")).items()
    }
    h_n, n_h, n_kv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    d = h_n // n_h
    n = len(ids)

    def rms(x, wt):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + cfg["rms_norm_eps"]) * wt

    def rope(x):  # x: [n, heads, d]。(x[i], x[i + d/2]) を対にする
        ang = np.arange(n)[:, None] * cfg["rope_theta"] ** (-np.arange(0, d, 2) / d)
        cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)

    x = w["model.embed_tokens.weight"][ids]
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        y = rms(x, w[p + "input_layernorm.weight"])
        proj = lambda nm, y=y, p=p: y @ w[p + f"self_attn.{nm}_proj.weight"].T  # noqa: E731
        q = rope((proj("q") + w[p + "self_attn.q_proj.bias"]).reshape(n, n_h, d))
        k = rope((proj("k") + w[p + "self_attn.k_proj.bias"]).reshape(n, n_kv, d))
        v = (proj("v") + w[p + "self_attn.v_proj.bias"]).reshape(n, n_kv, d)
        k, v = np.repeat(k, n_h // n_kv, axis=1), np.repeat(v, n_h // n_kv, axis=1)
        sc = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(d)
        sc = np.where(np.tril(np.ones((n, n), bool)), sc, -np.inf)
        pr = np.exp(sc - sc.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        att = np.einsum("hqk,khd->qhd", pr, v).reshape(n, -1)
        x = x + att @ w[p + "self_attn.o_proj.weight"].T
        y = rms(x, w[p + "post_attention_layernorm.weight"])
        g = y @ w[p + "mlp.gate_proj.weight"].T
        silu = g / (1 + np.exp(-g))
        x = x + (silu * (y @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
    return rms(x, w["model.norm.weight"]) @ w["model.embed_tokens.weight"].T


@pytest.mark.parametrize("rope_theta", [10000.0, 100.0])
def test_logits_match_numpy_reference(tmp_path: Path, rope_theta: float) -> None:
    """REQ-41: numpy 独立実装の logits と一致（L=10・小さい rope_theta で RoPE の対を検出）"""
    d = synthetic.write_model_dir(tmp_path, rope_theta=rope_theta)
    ids = [1, 5, 9, 20, 7, 3, 8, 2, 4, 6]
    got = np.array(load_qwen2(d, dtype=mx.float32)(mx.array([ids])))[0]
    assert np.abs(got - _np_reference(d, ids)).max() < 1e-4


def _resave(path: Path, weights: dict) -> None:
    """遅延読み込み中の同一ファイルを上書きしないよう、評価してから置き換える。"""
    mx.eval(weights)
    path.unlink()
    mx.save_safetensors(str(path), weights)


def _rewrite_config(d: Path, **changes: object) -> None:
    cfg = json.loads((d / "config.json").read_text())
    cfg.update(changes)
    (d / "config.json").write_text(json.dumps(cfg))


@pytest.mark.parametrize(
    "changes",
    [
        {"model_type": "llama"},
        {"rope_scaling": {"type": "linear", "factor": 2.0}},
        {"use_sliding_window": True},
        {"num_key_value_heads": 3},
        {"hidden_size": 0},
    ],
)
def test_unsupported_config_rejected(model_dir: Path, changes: dict) -> None:
    """REQ-41: 対応範囲外の config は ValueError。"""
    _rewrite_config(model_dir, **changes)
    with pytest.raises(ValueError, match=r"unsupported|invalid"):
        load_qwen2(model_dir, dtype=mx.float32)


def test_untied_without_lm_head_rejected(model_dir: Path) -> None:
    """REQ-41: tie=false で lm_head.weight が無いと停止する。"""
    _rewrite_config(model_dir, tie_word_embeddings=False)
    with pytest.raises(ValueError, match="lm_head"):
        load_qwen2(model_dir, dtype=mx.float32)


def test_missing_and_extra_keys_rejected(model_dir: Path) -> None:
    """REQ-41: 重みキーの欠落・余剰は停止する。"""
    path = model_dir / "model.safetensors"
    w = mx.load(str(path))
    dropped = {k: v for k, v in w.items() if k != "model.norm.weight"}
    _resave(path, dropped)
    with pytest.raises(ValueError, match=r"missing=1, unexpected=0"):
        load_qwen2(model_dir, dtype=mx.float32)
    _resave(path, {**w, "model.extra.weight": mx.zeros((1,))})
    with pytest.raises(ValueError, match=r"missing=0, unexpected=1"):
        load_qwen2(model_dir, dtype=mx.float32)


def test_shape_mismatch_rejected(model_dir: Path) -> None:
    """REQ-41: 形状違いの重みは load_weights(strict) が拒否する。"""
    path = model_dir / "model.safetensors"
    w = mx.load(str(path))
    w["model.norm.weight"] = mx.zeros((3,))
    _resave(path, w)
    with pytest.raises(ValueError, match="norm"):
        load_qwen2(model_dir, dtype=mx.float32)


def test_oversize_files_rejected(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-39・REQ-41: config.json / model.safetensors は上限超過を読む前に拒否する。"""
    monkeypatch.setattr(qwen2_model, "MAX_MODEL_BYTES", 20000)
    with pytest.raises(ValueError, match=r"model\.safetensors too large"):
        load_qwen2(model_dir, dtype=mx.float32)
    monkeypatch.undo()
    _rewrite_config(model_dir, pad="x" * qwen2_model.MAX_CONFIG_BYTES)
    with pytest.raises(ValueError, match=r"config\.json too large"):
        load_qwen2(model_dir, dtype=mx.float32)


def test_apply_lora_argument_validation(model_dir: Path) -> None:
    """REQ-41: 範囲外の num_layers / rank / dropout は ValueError。"""
    m = load_qwen2(model_dir, dtype=mx.float32)
    for kw in ({"num_layers": 0}, {"num_layers": 3}, {"rank": 0}, {"dropout": 1.0}):
        args = {"num_layers": 1, "rank": 8, "scale": 20.0, "dropout": 0.0, **kw}
        with pytest.raises(ValueError, match=r"out of range|invalid LoRA"):
            apply_lora(m, **args)


@pytest.mark.parametrize(
    "changes",
    [
        {"num_hidden_layers": 129},
        {"hidden_size": 16400},
        {"intermediate_size": 65537},
        {"num_attention_heads": 257},
        {"vocab_size": (1 << 20) + 1},
        {"hidden_size": 18, "num_attention_heads": 2, "num_key_value_heads": 1},  # head_dim 9
        {"rope_theta": float("nan")},
    ],
)
def test_config_limits_rejected(model_dir: Path, changes: dict) -> None:
    """REQ-39・REQ-41: config の上限超過・奇数 head_dim・NaN は構築前に拒否する。"""
    _rewrite_config(model_dir, **changes)
    with pytest.raises(ValueError, match=r"too large|invalid config"):
        load_qwen2(model_dir, dtype=mx.float32)


def test_estimated_size_rejected_before_build(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: 見積もり（パラメータ数 × dtype バイト）が上限超過なら構築前に拒否する。"""
    monkeypatch.setattr(qwen2_model, "MAX_MODEL_MEMORY_BYTES", 100)
    with pytest.raises(ValueError, match="larger than the supported size"):
        load_qwen2(model_dir, dtype=mx.float32)


def test_real_qwen_size_fits_limits() -> None:
    """REQ-41: Qwen2.5-0.5B の形状は bf16 で 1 GiB 未満、float32 でもメモリ上限内（概算 494M）。"""
    cfg = qwen2_model.Qwen2Config(896, 24, 14, 2, 4864, 151936, 1e-6, 1e6, True)
    assert 490_000_000 < cfg.estimated_params() < 500_000_000
    assert cfg.estimated_params() * 2 <= qwen2_model.MAX_MODEL_BYTES
    assert cfg.estimated_params() * 4 <= qwen2_model.MAX_MODEL_MEMORY_BYTES


def test_invalid_config_bytes_are_value_error(model_dir: Path) -> None:
    """REQ-39: 非 UTF-8・深い入れ子・NaN 定数は固定メッセージの ValueError。"""
    cfg = model_dir / "config.json"
    for data in (b"\xff\xfe", b"[" * 30000, b'{"rope_theta": NaN}', b"[]"):
        cfg.write_bytes(data)
        with pytest.raises(ValueError, match=r"^invalid config\.json$"):
            load_qwen2(model_dir, dtype=mx.float32)


def test_symlink_and_non_regular_rejected(model_dir: Path, tmp_path: Path) -> None:
    """REQ-39: symlink・通常ファイル以外は開かない（O_NOFOLLOW・S_ISREG）。"""
    real = model_dir / "model.safetensors"
    link_dir = tmp_path / "linked"
    link_dir.mkdir()
    (link_dir / "config.json").write_bytes((model_dir / "config.json").read_bytes())
    (link_dir / "model.safetensors").symlink_to(real)
    with pytest.raises(ValueError, match=r"cannot open model\.safetensors"):
        load_qwen2(link_dir, dtype=mx.float32)
    real.unlink()
    real.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        load_qwen2(model_dir, dtype=mx.float32)


def test_model_swapped_after_check_reads_verified_content(
    model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: 検証後にパスが別ファイルへ差し替わっても、検証済み fd の中身が読まれる。"""
    expected = np.array(load_qwen2(model_dir, dtype=mx.float32)(IDS))
    other = synthetic.write_model_dir(_mk(tmp_path / "other"), seed=1)
    real_open = qwen2_model._open_regular

    def swapping_open(path: Path, limit: int, what: str):
        result = real_open(path, limit, what)
        if what == "model.safetensors":
            os.replace(other / "model.safetensors", path)  # 新しい inode に差し替える
        return result

    monkeypatch.setattr(qwen2_model, "_open_regular", swapping_open)
    got = np.array(load_qwen2(model_dir, dtype=mx.float32)(IDS))
    assert np.abs(got - expected).max() < 1e-6


def test_lora_bf16_model_forward_and_grad(model_dir: Path) -> None:
    """REQ-41: bf16 モデルに LoRA を適用しても forward・勾配が通り、出力は bf16。"""
    import mlx.nn as nn

    m = load_qwen2(model_dir, dtype=mx.bfloat16)
    apply_lora(m, num_layers=2, rank=8, scale=20.0, dropout=0.0)
    out = m(IDS)
    assert out.dtype == mx.bfloat16
    loss, grads = nn.value_and_grad(m, lambda mm: mm(IDS).astype(mx.float32).sum())(m)
    assert bool(mx.isfinite(loss))
    from mlx.utils import tree_flatten

    flat = tree_flatten(grads)
    assert len(flat) == 2 * 7 * 2
    assert all(g.dtype == mx.float32 for _, g in flat)


def test_tie_with_lm_head_conflict_message(model_dir: Path) -> None:
    """REQ-41: tie=true なのに lm_head.weight がある食い違いを明示して拒否する。"""
    path = model_dir / "model.safetensors"
    w = mx.load(str(path))
    _resave(path, {**w, "lm_head.weight": mx.zeros((synthetic.TINY_VOCAB, 16))})
    with pytest.raises(ValueError, match="tie_word_embeddings=true conflicts with lm_head"):
        load_qwen2(model_dir, dtype=mx.float32)


@pytest.mark.parametrize(
    "kw",
    [
        {"rank": True},
        {"rank": 8.0},
        {"rank": 257},
        {"num_layers": True},
        {"scale": float("nan")},
        {"scale": float("inf")},
        {"dropout": float("nan")},
        {"dropout": -0.1},
        {"scale": "20"},
    ],
)
def test_apply_lora_strict_types(model_dir: Path, kw: dict) -> None:
    """REQ-41: bool・float の rank、上限超過、非有限の scale / dropout を拒否する。"""
    m = load_qwen2(model_dir, dtype=mx.float32)
    args = {"num_layers": 1, "rank": 8, "scale": 20.0, "dropout": 0.0, **kw}
    with pytest.raises(ValueError, match=r"invalid LoRA|out of range"):
        apply_lora(m, **args)
