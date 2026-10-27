"""qwen2_lora の全件一致 fixture（`fixtures/qwen2_parity/`）の生成器（REQ-41・REQ-32・#434）。

役割: 極小の合成ベースモデル（`tools/poc26/synthetic.py` の config・tokenizer と
seed 固定の乱数重み）を学習ワーカーの CLI（`launch.py train`）で CPU・固定 seed で学習し、
成果物ディレクトリ（`artifact.json`・`model.onnx`・`weights-*.bin`・`tokenizer.json`）を
そのまま `--out` へ複写する。
あわせて各ケースの学習側の予測（`validation_predictions` = MLX 内推論）と、書き出した ONNX を
`onnx.reference.ReferenceEvaluator` で推論した確率を `cases.json` に記録する。Rust 推論
ランタイム（`crates/runtime/tests/qwen2_parity.rs`）が、Python なしの自作ランタイムの予測ラベル
と全件一致を確かめる（証拠種別: テストハーネス。実重みではない）。

- 合成データのみ（外部データ・LLM 生成物・実データは使わない）。通信なし（REQ-38）
- 統合重みは bf16 に丸められるため、f32 の MLX と順位が入れ替わりうる僅差
  （上位 2 つの確率差が `NEAR_TIE_MARGIN` 未満）のケースは記録から除外し、件数を記録する
  （僅差でない MLX との不一致は除外せず失敗させる）
- 手順: `uv --no-config run --locked --directory trainer python tools/gen_qwen2_parity_fixture.py
  --out ../fixtures/qwen2_parity --overwrite`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import onnx
from mlx.utils import tree_flatten
from onnx import reference

_TRAINER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TRAINER))
sys.path.insert(0, str(_TRAINER / "src"))

from tools.poc26 import synthetic  # noqa: E402

from fandhe_edge_trainer.qwen2_model import Qwen2Config, Qwen2Model  # noqa: E402
from fandhe_edge_trainer.qwen2_tokenizer import Qwen2Tokenizer  # noqa: E402

LABELS = ["red", "tan", "sky", "map", "fox", "owl", "gem", "ink"]
SEED = 20261017
#: 上位 2 つの確率差がこの値未満のケースは僅差として除外する（bf16 丸めでの順位入れ替わり対策）。
NEAR_TIE_MARGIN = 1e-3
N_CASES = 120
WEIGHT_STD = 1.0
#: ベースモデルの重みの seed。予測ラベルが入力で変わる（2 種類以上が十分な件数で現れる）値を
#: 探索して固定した（seed 0 は全件が同じラベルになった。入力の違いは最終位置の状態に
#: ほとんど効かないため、学習ではなくベース重みの seed で入力依存を得ている）。
BASE_SEED = 11
_WORDS = {lb: [lb + str(j) for j in range(4)] for lb in LABELS}
_LAUNCH = str(_TRAINER / "launch.py")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _base_model(root: Path) -> Path:
    """`root/base` に config・tokenizer・seed 固定の f32 重みを置く。"""
    return write_base_model(root / "base")


def write_base_model(d: Path, **config_overrides: Any) -> Path:
    """`d`（新規に作る）に config・tokenizer・seed 固定の f32 重みを置く。

    `gen_qwen2_tiny_base_model.py` も使う。`config_overrides` は config の上書き
    （既定は parity と同じ）。
    """
    d.mkdir()
    config = {**synthetic.tiny_config(), "max_position_embeddings": 1024, **config_overrides}
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    synthetic.write_tokenizer_json(d)
    model = Qwen2Model(Qwen2Config.from_bytes(json.dumps(config).encode()))
    rng = np.random.default_rng(BASE_SEED)
    w = {
        k: mx.array(rng.normal(0.0, WEIGHT_STD, v.shape).astype(np.float32))
        for k, v in tree_flatten(model.parameters())
    }
    mx.save_safetensors(str(d / "model.safetensors"), w)
    return d


def _inputs(rng: random.Random) -> list[str]:
    every = [w for ws in _WORDS.values() for w in ws]
    out = []
    for i in range(N_CASES):
        label = LABELS[i % len(LABELS)]
        words = [rng.choice(_WORDS[label]) for _ in range(rng.randint(2, 8))]
        words += [rng.choice(every) for _ in range(rng.randint(0, 5))]
        rng.shuffle(words)
        text = " ".join(words)
        if i % 7 == 3:
            text = text.upper()
        elif i % 7 == 5:
            text = "  " + text.replace(" ", "   ") + "\t"
        out.append(text)
    return out


class GatherElements(reference.op_run.OpRun):
    """標準の GatherElements は軸の長さが 64 を超えると失敗するため置き換える。"""

    op_domain = ""

    def _run(self, data: Any, indices: Any, axis: int = 0) -> Any:
        return (np.take_along_axis(data, indices, axis=axis),)


def _ref_probs(out: Path, ids: list[int]) -> np.ndarray:
    ev = reference.ReferenceEvaluator(
        onnx.load(str(out / "model.onnx"), load_external_data=True), new_ops=[GatherElements]
    )
    (probs,) = ev.run(None, {"ids": np.array([ids], dtype=np.int64)})
    return probs[0]


def generate(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"{out_dir} is not empty (use --overwrite)")
        shutil.rmtree(out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "w"
        root.mkdir()
        base = _base_model(root)
        rng = random.Random(SEED)  # noqa: S311  合成データ用の決定的乱数
        rows = []
        for i in range(20):
            for label in LABELS:
                words = [rng.choice(_WORDS[label]) for _ in range(3)]
                rows.append({"input": " ".join(words) + f" {i}", "label": label})
        (root / "train.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        texts = _inputs(random.Random(SEED + 1))  # noqa: S311
        config = {
            "base_model_dir": base.name,
            "base_model_sha256": {p.name: _sha(p) for p in sorted(base.iterdir())},
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
        request = {
            "schema_version": 1,
            "kind": "qwen2_lora",
            "kind_version": 1,
            "config": config,
            "label_order": LABELS,
            "max_bytes": 64,
            "seed": 0,
            "device": "cpu",
            "root": str(root),
            "train_path": "train.jsonl",
            "out_dir": "out",
            "validation_inputs": [{"id": f"c{i:03d}", "input": t} for i, t in enumerate(texts)],
        }
        (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
        r = subprocess.run(  # noqa: S603  引数リスト・shell なし
            [sys.executable, "-I", _LAUNCH, "train", "--request", str(root / "request.json")],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if r.returncode != 0:
            raise SystemExit(f"train failed: {r.returncode} {r.stderr[:2000]}")
        payload = json.loads(r.stdout.strip())
        art = json.loads((root / "out" / "artifact.json").read_text(encoding="utf-8"))
        shutil.copytree(root / "out", out_dir)

    tok = Qwen2Tokenizer.from_bytes((out_dir / "tokenizer.json").read_bytes())
    system = art["scoring"]["system_prompt"]
    predicted = {p["id"]: p["predicted_label"] for p in payload["validation_predictions"]}
    cases, excluded = [], 0
    for i, text in enumerate(texts):
        cid = f"c{i:03d}"
        ids = tok.build_chat_ids(system, text, add_generation_prompt=True)
        p = _ref_probs(out_dir, ids)
        order = np.argsort(-p, kind="stable")
        mlx_index = LABELS.index(predicted[cid])
        if float(p[order[0]] - p[order[1]]) < NEAR_TIE_MARGIN:
            excluded += 1
            continue
        if int(order[0]) != mlx_index:
            # 僅差でない不一致は書き出し不具合の疑いがあるため、除外せず生成を失敗させる
            raise SystemExit(f"non-near-tie mismatch: {cid} ref={int(order[0])} mlx={mlx_index}")
        cases.append(
            {
                "name": cid,
                "input": text,
                "mlx_label_index": mlx_index,
                "ref_probs": [float(x) for x in p],
                "margin": float(p[order[0]] - p[order[1]]),
            }
        )
    labels_seen = sorted({c["mlx_label_index"] for c in cases})
    if len(cases) < 100 or len(labels_seen) < 2:
        counts = {lb: list(predicted.values()).count(lb) for lb in LABELS}
        raise SystemExit(
            f"too few usable cases: {len(cases)} (excluded {excluded}, mlx labels {counts})"
        )
    meta = {
        "kind": "qwen2_lora",
        "data_seed": SEED,
        "input_seed": SEED + 1,
        "train_seed": 0,
        "base_seed": BASE_SEED,
        "weight_std": WEIGHT_STD,
        "label_order": LABELS,
        "near_tie_margin": NEAR_TIE_MARGIN,
        "excluded_near_tie_or_mismatch": excluded,
        "config": config | {"base_model_sha256": "<omitted>"},
        "env": {
            "platform": platform.platform(),
            "mlx": getattr(mx, "__version__", "unknown"),
            "onnx": onnx.__version__,
            "numpy": np.__version__,
        },
    }
    (out_dir / "cases.json").write_text(
        json.dumps({"_meta": meta, "cases": cases}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"cases={len(cases)} excluded={excluded}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    generate(Path(a.out), a.overwrite)


if __name__ == "__main__":
    main()
