"""PoC-26 のシャード構成のベースモデル識別値（REQ-39 の完全性。REQ-41・#552）のテスト。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.poc26 import assets as assets_mod
from tools.poc26.lora_poc import main

from test_poc26_lora_poc import make_data, make_dir, pin_values, train_args
from test_poc26_shards import INDEX, _h, _shard, _shard_flags, _with_index_pin


def _predict_sharded(model: Path, data: dict, adapter: Path, out: Path, shards: dict) -> int:
    pins = pin_values(model)
    pins["model"] = _h(model / INDEX)
    argv = [
        "predict",
        *["--model-dir", str(model), "--definition", str(data["definition"])],
        *["--adapter-dir", str(adapter), "--input", str(data["validation"])],
        *["--out-dir", str(out), "--max-seq-length", "1024", "--device", "cpu"],
        *["--dtype", "float32", "--evidence", "test_harness"],
        *["--adapter-sha256", _h(adapter / "adapters.safetensors")],
        *[x for k, v in pins.items() for x in (f"--{k}-sha256", v)],
        *_shard_flags(shards),
    ]
    return main(argv)


def test_single_identity_is_plain_model_sha256(tmp_path: Path) -> None:
    """REQ-39・#552: 単一ファイルの識別値は従来の model.safetensors の sha256 のまま。"""
    model = make_dir(tmp_path)
    v = pin_values(model)
    pins = assets_mod.Pins(v["model"], v["config"], v["tokenizer"], v["tokenizer-config"])
    assert assets_mod.base_identity(pins) == _h(model / "model.safetensors")


def test_shard_identity_binds_contents_and_rejects_other_shards(tmp_path: Path) -> None:
    """REQ-39・#552: 同じ index でも中身の違うシャードの組は adapter 照合で拒否される。"""
    model, data = make_dir(tmp_path), make_data(tmp_path)
    shards = _shard(model)
    out = tmp_path / "o1"
    argv = _with_index_pin(train_args(model, data, out), model) + _shard_flags(shards)
    assert main(argv) == 0
    ident = assets_mod.shard_identity(_h(model / INDEX), sorted(shards.items()))
    run = json.loads((out / "run.json").read_text())
    assert run["base_model_identity_sha256"] == ident
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["base_model_sha256"] == ident
    assert ident != _h(model / INDEX)
    assert _predict_sharded(model, data, out, tmp_path / "p1", shards) == 0
    # 1 シャードだけ中身と sha を差し替える（index は同じ。sha も合わせるので load 自体は通る）
    w = mx.load(str(model / "b.safetensors"))
    mx.eval(w)
    mx.save_safetensors(str(model / "b.safetensors"), {k: v + 0.5 for k, v in w.items()})
    other = {**shards, "b.safetensors": _h(model / "b.safetensors")}
    assert _predict_sharded(model, data, out, tmp_path / "p2", other) == 64
