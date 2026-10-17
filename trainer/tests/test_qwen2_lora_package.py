"""qwen2_lora の自己完結パッケージ（REQ-41・REQ-39・REQ-40・TASK-41.6-3・#432）。

CLI（`launch.py train`）で極小の合成ベースモデルを学習し、成果物ディレクトリを別の場所へ複写して
（ベースモデル無しで）推論に必要なものが揃うことを確かめる。証拠種別: テストハーネス
（ReferenceEvaluator・CPU）。実重み・Rust ランタイムとの全件一致は #434。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from tools.poc26 import synthetic

from fandhe_edge_trainer.qwen2_tokenizer import Qwen2Tokenizer
from test_qwen2_base_model import expected, make_dir
from test_qwen2_onnx_export import evaluator

_LAUNCH = str(Path(__file__).resolve().parent.parent / "launch.py")
_VALIDATION = [
    {"id": "v1", "input": "alpha alpha beta gamma 7"},
    {"id": "v2", "input": "delta delta epsilon zeta 7"},
    {"id": "v3", "input": "alpha beta"},
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _train_via_cli(root: Path) -> tuple[dict, Path, Path]:
    """`root` に合成ベースモデルと学習データを置き、CLI で学習する。(結果, out, base) を返す。"""
    root.mkdir()
    base = make_dir(root, config={**synthetic.tiny_config(), "max_position_embeddings": 256})
    rows = []
    for i in range(12):
        rows.append({"input": f"alpha alpha beta gamma {i}", "label": "cat_a"})
        rows.append({"input": f"delta delta epsilon zeta {i}", "label": "cat_b"})
    (root / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    request = {
        "schema_version": 1,
        "kind": "qwen2_lora",
        "kind_version": 1,
        "config": {
            "base_model_dir": base.name,
            "base_model_sha256": expected(base),
            "base_model_name": "tiny-synthetic",
            "base_model_license": "MIT",
            "base_model_origin": "https://example.com/tiny",
            "dtype": "float32",
            "iters": 30,
            "batch_size": 4,
            "max_seq_length": 200,
            "lora_rank": 4,
            "lora_num_layers": 2,
            "lr": 0.05,
        },
        "label_order": ["cat_a", "cat_b"],
        "max_bytes": 64,
        "seed": 0,
        "device": "cpu",
        "root": str(root),
        "train_path": "train.jsonl",
        "out_dir": "out",
        "validation_inputs": _VALIDATION,
    }
    (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
    # sys.executable を引数リストで呼ぶだけ（shell=True なし）
    r = subprocess.run(  # noqa: S603
        [sys.executable, "-I", _LAUNCH, "train", "--request", str(root / "request.json")],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert r.returncode == 0, (r.stdout[:500], r.stderr[:2000])
    return json.loads(r.stdout.strip()), root / "out", base


def test_self_contained_package(tmp_path: Path) -> None:
    """REQ-41・REQ-39・REQ-40・#432: 構成・sha256・来歴が揃い、複写先（ベース無し）で推論できる。"""
    payload, out, base = _train_via_cli(tmp_path / "w")
    art = json.loads((out / "artifact.json").read_text(encoding="utf-8"))
    assert payload["artifact"] == art
    assert art["input_representation"] == "tokenizer"

    # 構成は過不足なし（artifact.json・model.onnx・tokenizer.json・記録された shard のみ）
    shards = art["external_data"]
    assert shards, "tiny model should have external shards"
    assert {p.name for p in out.iterdir()} == {
        "artifact.json",
        "model.onnx",
        "tokenizer.json",
        *(s["location"] for s in shards),
    }
    assert art["onnx_sha256"] == _sha(out / "model.onnx")
    assert art["tokenizer_file"] == "tokenizer.json"
    assert art["tokenizer_sha256"] == _sha(out / "tokenizer.json")
    assert (out / "tokenizer.json").read_bytes() == (base / "tokenizer.json").read_bytes()
    for s in shards:
        f = out / s["location"]
        assert (s["size_bytes"], s["sha256"]) == (f.stat().st_size, _sha(f))

    # 来歴（ベースモデルの全ファイル・名前・license・origin）
    bm = art["base_model"]
    assert {f["path"]: f["sha256"] for f in bm["files"]} == expected(base)
    assert {f["path"]: f["size_bytes"] for f in bm["files"]} == {
        p.name: p.stat().st_size for p in base.iterdir() if p.is_file()
    }
    assert (bm["model_name"], bm["license"], bm["origin"]) == (
        "tiny-synthetic",
        "MIT",
        "https://example.com/tiny",
    )
    assert [c["id"] for c in art["scoring"]["choices"]] == ["cat_a", "cat_b"]

    # ベースモデルの無い複写先で推論する
    pkg = tmp_path / "pkg"
    shutil.copytree(out, pkg)
    tok = Qwen2Tokenizer.from_bytes((pkg / "tokenizer.json").read_bytes())
    ev = evaluator(onnx.load(str(pkg / "model.onnx"), load_external_data=True))
    system = art["scoring"]["system_prompt"]
    predicted = {p["id"]: p["predicted_label"] for p in payload["validation_predictions"]}
    checked = 0
    for row in _VALIDATION:
        ids = tok.build_chat_ids(system, row["input"], add_generation_prompt=True)
        (probs,) = ev.run(None, {"ids": np.array([ids], dtype=np.int64)})
        p = probs[0]
        if abs(p[0] - p[1]) < 0.02:  # bf16 丸めで順位が入れ替わりうる僅差は除く
            continue
        checked += 1
        assert art["scoring"]["choices"][int(np.argmax(p))]["id"] == predicted[row["id"]]
    assert checked >= 2
