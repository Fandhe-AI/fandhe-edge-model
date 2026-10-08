"""PoC-26 候補 P の ONNX 書き出し（`export_onnx.py`）の検査（REQ-41・TASK-41.1-7・#392）。

極小の合成 Qwen2（GQA あり）と合成 tokenizer で、LoRA 統合・ONNX の手組み・集計・CLI の
`export-onnx` / `verify-onnx` を CPU で決定的に確認する（テストハーネス。実重みは使わない）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.poc26 import qwen2_model, synthetic
from tools.poc26.export_onnx import _Graph, build_onnx, fused_weights, summarize
from tools.poc26.lora_poc import main

from test_poc26_lora_poc import _sha, make_data, make_dir, pin_flags, train_args

INITIALIZERS_TINY = 39  # 合成モデル（2 層・tie）の実測値
ATOL = 1e-4  # float32 の演算順序の差。実測は 1e-6 台


def _lora_model(*, tie: bool = True) -> qwen2_model.Qwen2Model:
    """末尾 1 層だけ LoRA を付け、lora_b を非 0 にした合成モデル（統合の差が logits に出る）。"""
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    cfg = qwen2_model.Qwen2Config.from_bytes(json.dumps(synthetic.tiny_config(tie=tie)).encode())
    model = qwen2_model.Qwen2Model(cfg)
    qwen2_model.apply_lora(model, num_layers=1, rank=2, scale=20.0, dropout=0.0, seed=1)
    last = model.model.layers[-1]
    for lin in (last.self_attn.q_proj, last.self_attn.v_proj, last.mlp.down_proj):
        lin.lora_b = mx.random.normal(lin.lora_b.shape) * 0.1
    mx.eval(model.parameters())
    return model


def _export(model: qwen2_model.Qwen2Model, out: Path) -> tuple[Path, Path]:
    out.mkdir()
    mp, dp = out / "model.onnx", out / "model.onnx.data"
    with dp.open("wb") as f:
        proto = build_onnx(model.config, fused_weights(model), _Graph(f, dp.name))
    onnx.save_model(proto, str(mp))
    return mp, dp


@pytest.mark.parametrize("tie", [True, False])
def test_onnx_matches_mlx_with_lora_and_gqa(tmp_path: Path, tie: bool) -> None:
    """REQ-41: 統合 ONNX の logits が LoRA 付き MLX と許容差内で一致し argmax も一致する。"""
    model = _lora_model(tie=tie)
    mp, _ = _export(model, tmp_path / "o")
    onnx.checker.check_model(str(mp))
    ref = ReferenceEvaluator(onnx.load(str(mp)))
    for n in (1, 7, 20):  # 動的 T
        ids = np.random.default_rng(n).integers(0, 300, (1, n))
        want = np.array(model(mx.array(ids)))
        got = ref.run(None, {"input_ids": ids.astype(np.int64)})[0]
        assert got.shape == want.shape == (1, n, 300)
        assert np.abs(got - want).max() < ATOL
        assert (got.argmax(-1) == want.argmax(-1)).all()


def test_lora_fuse_changes_weights_by_scale_times_ab() -> None:
    """REQ-41: 統合重みは `W + scale * (A @ B).T`（LoRA なしの重みとは異なる）。"""
    model = _lora_model()
    lin = model.model.layers[-1].self_attn.q_proj
    want = np.array(lin.linear.weight) + 20.0 * (np.array(lin.lora_a) @ np.array(lin.lora_b)).T
    fused = fused_weights(model)
    np.testing.assert_allclose(fused["model.layers.1.self_attn.q_proj.weight"], want, atol=1e-6)
    assert np.abs(want - np.array(lin.linear.weight)).max() > 1e-3
    first = np.array(model.model.layers[0].self_attn.q_proj.weight)  # LoRA なしの層はそのまま
    np.testing.assert_array_equal(fused["model.layers.0.self_attn.q_proj.weight"], first)


def test_summary_lists_ops_and_runtime_gaps(tmp_path: Path) -> None:
    """REQ-41: 集計は opset 13・演算数・ランタイム未対応 op・上限超過を返す。"""
    mp, dp = _export(_lora_model(), tmp_path / "o")
    s = summarize(mp, dp)
    assert (s["opset"], s["ir_version"]) == (13, 8)
    assert s["op_counts"]["Softmax"] == 2
    assert s["op_counts"]["MatMul"] > 0
    assert sum(s["op_counts"].values()) == s["node_count"]
    # ランタイムは固定テンプレートのみ。MatMul・Where・Cos などは未対応として列挙される
    assert {"MatMul", "Where", "Cos", "Sin", "Reshape", "Split"} <= set(
        s["runtime"]["unsupported_ops"]
    )
    assert "Softmax" not in s["runtime"]["unsupported_ops"]
    lim = s["runtime"]["limits"]
    assert s["initializer_count"] == INITIALIZERS_TINY
    assert lim["initializer_count"] == {"limit": 64, "actual": INITIALIZERS_TINY, "exceeded": False}
    assert lim["node_count"]["actual"] == s["node_count"] > 128  # 合成 2 層でもノード上限は超える
    assert lim["node_count"]["exceeded"] is True
    assert lim["max_dims"] == {"limit": 4, "actual": 2, "exceeded": False}
    assert s["runtime"]["external_data_used"] is True
    assert s["runtime"]["external_data_accepted"] is False
    assert s["external_data_bytes"] > 0
    assert s["dtype"] == "float32"


@pytest.fixture(scope="module")
def adapter_env(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("export")
    model, data = make_dir(tmp), make_data(tmp)
    out = tmp / "train"
    assert main(train_args(model, data, out)) == 0
    return {"tmp": tmp, "model": model, "adapter": out, "data": data}


def test_cli_export_then_verify(adapter_env: dict, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-41: 学習済み adapter から `export-onnx` で書き出し、`verify-onnx` で一致を確認できる。"""
    a = adapter_env
    flags = [
        *["--model-dir", str(a["model"]), "--adapter-dir", str(a["adapter"])],
        *["--adapter-sha256", _sha(a["adapter"] / "adapters.safetensors")],
        *pin_flags(a["model"]),
        *["--evidence", "test_harness"],
    ]
    onnx_dir = a["tmp"] / "onnx"
    assert main(["export-onnx", *flags, "--out-dir", str(onnx_dir)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["checker"], summary["opset"]) == ("passed", 13)
    assert (onnx_dir / "export_summary.json").is_file()
    assert main(["verify-onnx", *flags, "--onnx-dir", str(onnx_dir)]) == 0
    res = json.loads(capsys.readouterr().out)["prompts"]
    assert len(res) == 2
    for r in res:
        assert r["max_abs_error"] < ATOL
        assert r["argmax_match_positions"] == r["positions"]
    assert main(["export-onnx", *flags, "--out-dir", str(onnx_dir)]) == 64  # 上書きしない


def test_cli_rejects_wrong_adapter_sha(adapter_env: dict) -> None:
    """REQ-41: adapter の sha256 が違えば 64 で止まり、出力先を作らない。"""
    a = adapter_env
    flags = [
        *["--model-dir", str(a["model"]), "--adapter-dir", str(a["adapter"])],
        *["--adapter-sha256", "0" * 64, *pin_flags(a["model"])],
        *["--evidence", "test_harness", "--out-dir", str(a["tmp"] / "never")],
    ]
    assert main(["export-onnx", *flags]) == 64
    assert not (a["tmp"] / "never").exists()


def _flags(a: dict, adapter: Path) -> list[str]:
    return [
        *["--model-dir", str(a["model"]), "--adapter-dir", str(adapter)],
        *["--adapter-sha256", _sha(adapter / "adapters.safetensors")],
        *pin_flags(a["model"]),
        *["--evidence", "test_harness"],
    ]


def test_verify_matches_with_dropout_adapter(
    adapter_env: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-41: dropout 0.5 の adapter でも MLX 側が eval モードで決定的になり、照合が一致する。"""
    a = adapter_env
    drop = a["tmp"] / "train_drop"
    assert main(train_args(a["model"], a["data"], drop, extra=["--dropout", "0.5"])) == 0
    flags = _flags(a, drop)
    onnx_dir = a["tmp"] / "onnx_drop"
    assert main(["export-onnx", *flags, "--out-dir", str(onnx_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["adapter_config"]["dropout"] == 0.5
    assert main(["verify-onnx", *flags, "--onnx-dir", str(onnx_dir)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["adapter_dropout"] == 0.5
    for r in out["prompts"]:
        assert r["max_abs_error"] < ATOL
        assert r["argmax_match_positions"] == r["positions"]


def test_verify_rejects_tampered_onnx(
    adapter_env: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-41・REQ-39: 書き出し時の sha256 と違うファイルは verify が読む前に 64 で拒否する。"""
    a = adapter_env
    flags = _flags(a, a["adapter"])
    onnx_dir = a["tmp"] / "onnx_tamper"
    assert main(["export-onnx", *flags, "--out-dir", str(onnx_dir)]) == 0
    capsys.readouterr()
    data = onnx_dir / "model.onnx.data"
    raw = bytearray(data.read_bytes())
    raw[0] ^= 1
    data.write_bytes(bytes(raw))
    assert main(["verify-onnx", *flags, "--onnx-dir", str(onnx_dir)]) == 64
