"""PoC-26 predict の `--quantize 4bit`・`--kv-reuse`（REQ-41・TASK-41.11-3・#583）。

証拠種別: テストハーネス（CPU）。

極小の合成 Qwen2（量子化の group 64 に合わせ hidden=64）で確かめる。実重みは使わない。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tools.poc26 import synthetic
from tools.poc26.kv_reuse import score_labels_kv
from tools.poc26.lora_poc import main
from tools.poc26.qwen2_model import (
    LoRALinear,
    Qwen2Config,
    Qwen2Model,
    apply_lora,
    fuse_lora,
    lora_delta,
    quantize_model,
)
from tools.poc26.score import score_labels

from test_poc26_lora_poc import _predict_argv, make_data, make_dir, train_args


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)
    return None


def _model(tmp_path: Path, *, tie: bool = True) -> Qwen2Model:
    d = synthetic.write_model_dir(tmp_path, tie=tie, hidden=64)
    m = Qwen2Model(Qwen2Config.from_file(d / "config.json"))
    m.load_weights(list(mx.load(str(d / "model.safetensors")).items()))
    apply_lora(m, num_layers=2, rank=4, scale=2.0, dropout=0.0, seed=0)
    # B を非零にして LoRA が出力に効くようにする
    key = mx.random.key(1)
    for blk in m.model.layers:
        for lin in (blk.self_attn.q_proj, blk.mlp.down_proj):
            key, sub = mx.random.split(key)
            lin.lora_b = mx.random.normal(lin.lora_b.shape, key=sub) * 0.1
    mx.eval(m.parameters())
    m.eval()
    return m


@pytest.mark.parametrize("tie", [True, False])
def test_kv_reuse_matches_full_forward(tmp_path: Path, tie: bool) -> None:
    """REQ-41: KV 再利用のスコアは KV なしと 1e-4 以内（pad・長さばらばら・チャンク跨ぎ）。"""
    m = _model(tmp_path, tie=tie)
    rng = np.random.default_rng(0)
    prompt = [int(x) for x in rng.integers(0, 250, 11)]
    lens = [3, 5, 2, 6, 4, 1, 7, 3, 4, 2]  # 10 件 > チャンク 8
    labels = [[int(x) for x in rng.integers(0, 250, n)] for n in lens]
    ref = score_labels(m, prompt, labels, 262, chunk=8)
    got = score_labels_kv(m, prompt, labels, 262, 8)
    assert got.shape == (10,)
    np.testing.assert_allclose(got, ref, atol=1e-4)
    np.testing.assert_allclose(score_labels_kv(m, prompt, labels, 262, 3), ref, atol=1e-4)


def test_fuse_lora_equals_lora_forward(tmp_path: Path) -> None:
    """REQ-41: 統合後の forward は LoRA ありと一致し、LoRALinear は残らない。"""
    m = _model(tmp_path)
    ids = mx.array([[1, 5, 9, 20, 7, 3]])
    before = m(ids)
    fuse_lora(m)
    assert not any(isinstance(x, LoRALinear) for _, x in m.named_modules())
    np.testing.assert_allclose(np.array(m(ids)), np.array(before), atol=1e-4)


def test_quantize_scope_and_fuse_before_quantize(tmp_path: Path) -> None:
    """REQ-41: Linear・Embedding が量子化層になり、統合済み重みを量子化したものと一致する。"""
    m = _model(tmp_path)
    expect = {}
    for i, blk in enumerate(m.model.layers):
        for owner, name in (("self_attn", "q_proj"), ("mlp", "down_proj")):
            lin = getattr(getattr(blk, owner), name)
            if isinstance(lin, LoRALinear):
                expect[f"model.layers.{i}.{owner}.{name}"] = lin.linear.weight + lora_delta(lin)
    fuse_lora(m)
    quantize_model(m)
    mods = dict(m.named_modules())
    assert isinstance(mods["model.embed_tokens"], nn.QuantizedEmbedding)
    assert all(isinstance(x, nn.QuantizedLinear) for k, x in mods.items() if k.endswith("_proj"))
    flat = dict(tree_flatten(m.parameters()))
    for k, w in expect.items():
        q, s, b = mx.quantize(w, group_size=64, bits=4)
        assert mx.array_equal(flat[k + ".weight"], q).item()
        assert mx.array_equal(flat[k + ".scales"], s).item()
        assert mx.array_equal(flat[k + ".biases"], b).item()


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("qkv")
    model, data = make_dir(tmp, hidden=64), make_data(tmp)
    out = tmp / "adapter"
    assert main(train_args(model, data, out)) == 0
    return {"tmp": tmp, "model": model, "data": data, "outs": [out]}


def _run(env: dict, name: str, *extra: str) -> tuple[Path, dict, list[dict]]:
    out = env["tmp"] / name
    assert main(_predict_argv(env, env["outs"][0], out, *extra)) == 0
    run = json.loads((out / "run.json").read_text())
    raw = [json.loads(x) for x in (out / "raw_scores.jsonl").read_text().splitlines()]
    return out, run, raw


def test_default_has_no_new_outputs(env: dict) -> None:
    """REQ-41: 既定は従来の出力のまま（新規キー・ファイルなし）。ピークメモリは従来どおり記録。"""
    out, run, _ = _run(env, "d")
    assert "quantization" not in run
    assert "kv_reuse" not in run
    assert not (out / "quantized_weights.safetensors").exists()
    assert "kv_reuse" not in run["scoring"]
    assert run["mlx_peak_memory_bytes"] > 0
    assert run["args"]["quantize"] == "none"
    assert run["args"]["kv_reuse"] is False


def test_cli_kv_reuse_matches_default(env: dict) -> None:
    """REQ-41: --kv-reuse の raw スコアは既定と 1e-4 以内で、run.json に長さの平均が載る。"""
    _, _, base = _run(env, "b")
    _, run, kv = _run(env, "kv", "--kv-reuse")
    assert run["kv_reuse"] is True
    assert run["scoring"]["prefix_tokens_mean"] > 0
    assert run["scoring"]["continuation_tokens_mean"] > 1
    for a, b in zip(base, kv, strict=True):
        assert a["id"] == b["id"]
        for k in a["loglik"]:
            assert abs(a["loglik"][k] - b["loglik"][k]) < 1e-4


@pytest.mark.parametrize("kv", [False, True])
def test_cli_quantize_records_file_hash(env: dict, kv: bool) -> None:
    """REQ-41: --quantize 4bit は保存ファイルの bytes・sha256 を記録する（KV 併用可）。"""
    out, run, raw = _run(env, f"q{kv}", "--quantize", "4bit", *(["--kv-reuse"] if kv else []))
    f = out / "quantized_weights.safetensors"
    q = run["quantization"]
    assert (q["bits"], q["group_size"], q["scope"], q["order"]) == (
        4,
        64,
        "linear+embedding",
        "fuse_lora_then_quantize",
    )
    assert q["weights_bytes"] == f.stat().st_size
    assert q["weights_sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    names = set(mx.load(str(f)))
    assert "model.embed_tokens.scales" in names
    assert not any("lora" in n for n in names)
    assert all(np.isfinite(v) for r in raw for v in r["loglik"].values())
