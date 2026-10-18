"""qwen2_lora の評価契約テスト用 fixture（`fixtures/qwen2_eval_contract/`）の生成器（REQ-41・#439）

役割: `gen_qwen2_parity_fixture.py` と同じ合成ベースモデル・学習データ・検証入力で、
qwen2_lora（train seed 0・1・2）と c1・c3（seed 0）を学習ワーカーの CLI（`launch.py train`、
CPU）で学習し、各 `validation_predictions` の予測ラベルだけを `predictions.json` に記録する。
Rust 側 `crates/train/tests/qwen2_lora_evaluation_contract.rs` が McNemar・Holm・Wilson を
評価器の関数で確かめる入力になる（証拠種別: テストハーネス。実重みではない）。

- 合成データのみ。通信なし（REQ-38）。パッケージは捨て、予測ラベルだけを残す
- fail-closed: qwen2_lora seed 0 の予測が `fixtures/qwen2_parity/cases.json` の MLX ラベルと
  不一致なら失敗する（許容しない）
- 手順: `uv --no-config run --locked --offline --directory trainer python
  tools/gen_qwen2_eval_contract_fixture.py --out ../fixtures/qwen2_eval_contract --overwrite`
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import onnx

_TRAINER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TRAINER))
sys.path.insert(0, str(_TRAINER / "src"))

from tools import gen_qwen2_parity_fixture as parity  # noqa: E402

LABELS = parity.LABELS
_PARITY_CASES = _TRAINER.parent / "fixtures" / "qwen2_parity" / "cases.json"
#: tests/conftest.py の極小設定と同じ（C1・C3）。
_C1 = {
    "ngram_min": 1,
    "ngram_max": 3,
    "min_df": 1,
    "max_features": 1000,
    "C": 1.0,
    "epochs": 40,
    "batch_size": 8,
    "lr": 0.5,
}
_C3 = {"epochs": 5, "batch_size": 8, "emb": 8, "filters": 8, "widths": [3, 5, 7], "dropout": 0.0}
_RUNS = [("c1", 0), ("c3", 0), ("qwen2_lora", 0), ("qwen2_lora", 1), ("qwen2_lora", 2)]


def _train(root: Path, kind: str, seed: int, config: dict, texts: list[str]) -> list[str | None]:
    """1 run を学習し、validation 予測ラベル（判定不能は None）を入力順で返す。"""
    request = {
        "schema_version": 1,
        "kind": kind,
        "kind_version": 1,
        "config": config,
        "label_order": LABELS,
        "max_bytes": 64,
        "seed": seed,
        "device": "cpu",
        "root": str(root),
        "train_path": "train.jsonl",
        "out_dir": f"out-{kind}-{seed}",
        "validation_inputs": [{"id": f"c{i:03d}", "input": t} for i, t in enumerate(texts)],
    }
    req_path = root / f"request-{kind}-{seed}.json"
    req_path.write_text(json.dumps(request), encoding="utf-8")
    r = subprocess.run(  # noqa: S603  引数リスト・shell なし
        [sys.executable, "-I", parity._LAUNCH, "train", "--request", str(req_path)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if r.returncode != 0:
        raise SystemExit(f"train failed: {kind} seed={seed} {r.returncode} {r.stderr[:2000]}")
    preds = json.loads(r.stdout.strip())["validation_predictions"]
    if [p["id"] for p in preds] != [f"c{i:03d}" for i in range(len(texts))]:
        raise SystemExit(f"prediction ids mismatch: {kind} seed={seed}")
    out = [p["predicted_label"] for p in preds]
    if any(x is not None and x not in LABELS for x in out):
        raise SystemExit(f"prediction outside label_order: {kind} seed={seed}")
    return out


def generate(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"{out_dir} is not empty (use --overwrite)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "w"
        root.mkdir()
        base = parity._base_model(root)
        rng = random.Random(parity.SEED)  # noqa: S311  合成データ用の決定的乱数
        rows = []
        for i in range(20):
            for label in LABELS:
                words = [rng.choice(parity._WORDS[label]) for _ in range(3)]
                rows.append({"input": " ".join(words) + f" {i}", "label": label})
        (root / "train.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        texts = parity._inputs(random.Random(parity.SEED + 1))  # noqa: S311
        qwen_config = {
            "base_model_dir": base.name,
            "base_model_sha256": {p.name: parity._sha(p) for p in sorted(base.iterdir())},
            "base_model_name": "tiny-synthetic",
            "base_model_license": "MIT",
            "base_model_origin": "https://example.com/tiny",
            "dtype": "float32",
            "iters": 20,
            "batch_size": 8,
            "max_seq_length": 700,
            "lora_rank": 4,
            "lora_num_layers": 2,
            "lr": 0.01,
        }
        configs = {"c1": _C1, "c3": _C3, "qwen2_lora": qwen_config}
        predictions = {f"{k}:seed{s}": _train(root, k, s, configs[k], texts) for k, s in _RUNS}

    if len(texts) != 120:
        raise SystemExit(f"unexpected case count: {len(texts)}")
    parity_cases = json.loads(_PARITY_CASES.read_text(encoding="utf-8"))["cases"]
    got = predictions["qwen2_lora:seed0"]
    for c in parity_cases:
        if got[int(c["name"][1:])] != LABELS[c["mlx_label_index"]]:
            raise SystemExit(f"qwen2_lora seed0 differs from qwen2_parity at {c['name']}")
    golds = [LABELS[i % len(LABELS)] for i in range(len(texts))]
    meta = {
        "runs": [f"{k}:seed{s}" for k, s in _RUNS],
        "label_order": LABELS,
        "gold_rule": "gold of case i is label_order[i % 8]",
        "data_seed": parity.SEED,
        "input_seed": parity.SEED + 1,
        "base_seed": parity.BASE_SEED,
        "config": qwen_config | {"base_model_sha256": "<omitted>"},
        "env": {
            "platform": platform.platform(),
            "mlx": getattr(mx, "__version__", "unknown"),
            "onnx": onnx.__version__,
            "numpy": np.__version__,
        },
    }
    doc = {
        "_meta": meta,
        "train_labels": [r["label"] for r in rows],
        "cases": [{"name": f"c{i:03d}", "gold": g} for i, g in enumerate(golds)],
        "predictions": predictions,
    }
    (out_dir / "predictions.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print({k: sum(g == p for g, p in zip(golds, v, strict=True)) for k, v in predictions.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    generate(Path(a.out), a.overwrite)


if __name__ == "__main__":
    main()
