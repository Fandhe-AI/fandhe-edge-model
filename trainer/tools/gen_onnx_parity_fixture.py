"""ONNX 一致確認用 fixture（`fixtures/onnx_parity/`）の生成器（REQ-32・TASK-32.1-2・#113）。

役割: 極小設定で C1・C3 を CPU・固定 seed で学習し、書き出した ONNX と「MLX 内推論の
予測ラベル・確率」を `cases.json` に記録する。Rust 推論ランタイム（`crates/runtime`）の
結合テスト（`tests/onnx_parity.rs`）が、学習依存の無い ONNX 推論の予測ラベルとこの記録の
全件一致を確かめる（証拠種別: テストハーネス。PoC-14 実測の踏襲）。

- 出力先は `--out` で受け、`c1.onnx`・`c3.onnx`・`cases.json` の 3 つだけを書く。既存ファイルの
  上書きは `--overwrite` を要求する
- 合成データのみ（外部データ・LLM 生成物・実データは使わない）。通信なし（REQ-38）
- 僅差（上位 2 つの確率差が `NEAR_TIE_MARGIN` 未満）のケースは、f32 の演算順序差で argmax が
  割れうるため記録から除外し、除外件数を記録する
- 手順: `uv run --locked --directory trainer python tools/gen_onnx_parity_fixture.py
  --out ../fixtures/onnx_parity`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fandhe_edge_trainer import budget as budget_mod
from fandhe_edge_trainer import guard
from fandhe_edge_trainer.contract import TrainExample, TrainRequest
from fandhe_edge_trainer.encoding import encode_bytes
from fandhe_edge_trainer.kinds import c1 as c1_mod
from fandhe_edge_trainer.kinds import c3 as c3_mod

LABEL_ORDER = ["alpha", "beta", "gamma"]
MAX_BYTES = 48
SEED = 20260929
#: 上位 2 つの確率差がこの値未満のケースは僅差として除外する。
NEAR_TIE_MARGIN = 1e-3
#: MLX と ReferenceEvaluator の確率差の上限（`tests/conftest.py::ATOL_MLX_ONNX` と同値）。
ATOL_MLX_ONNX = 1e-5

C1_CONFIG = {
    "ngram_min": 1,
    "ngram_max": 3,
    "min_df": 1,
    "max_features": 1000,
    "C": 1.0,
    "epochs": 40,
    "batch_size": 8,
    "lr": 0.5,
}
C3_CONFIG = {
    "epochs": 5,
    "batch_size": 8,
    "emb": 8,
    "filters": 8,
    "widths": [3, 5, 7],
    "dropout": 0.0,
}

_WORDS = {
    "alpha": ["alpha", "apple", "amber", "arrow", "atlas"],
    "beta": ["beta", "berry", "bison", "brick", "bloom"],
    "gamma": ["gamma", "grape", "ghost", "glide", "grain"],
}
_VECTORS_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "preprocess" / "byte_encoding_vectors.json"
)


def _train_examples(rng: random.Random) -> list[TrainExample]:
    """3 ラベルの合成学習データ（各ラベル固有の語＋共通の雑音語）。"""
    noise = ["one", "two", "red", "blue", "ok"]
    out: list[TrainExample] = []
    for label in LABEL_ORDER:
        for i in range(30):
            words = [rng.choice(_WORDS[label]) for _ in range(3)] + [rng.choice(noise)]
            rng.shuffle(words)
            out.append(TrainExample(input=" ".join(words) + f" {i}", label=label))
    return out


def _case_inputs(rng: random.Random) -> list[tuple[str, str]]:
    """評価ケースの入力（共有ゴールデンベクタの入力＋合成入力）。名前は識別用。"""
    cases: list[tuple[str, str]] = []
    vectors = json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]
    for v in vectors:
        cases.append((f"vector_{v['name']}", v["input"]))
    every = [w for ws in _WORDS.values() for w in ws]
    for i in range(120):
        n = rng.randint(1, 12)
        text = " ".join(rng.choice(every) for _ in range(n))
        style = i % 6
        if style == 1:
            text = text.upper()
        elif style == 2:
            text = "  " + text.replace(" ", "   ") + "\t"
        elif style == 3:
            text = text + " あいう é́"
        elif style == 4:
            text = "".join(chr(0xFF21 + (ord(c) - 65)) if "A" <= c <= "Z" else c for c in text)
        elif style == 5:
            text = text * 3  # max_bytes 超過で切り詰め
        cases.append((f"synthetic_{i:03d}", text))
    return cases


def _make_request(root: Path, kind: str, config: dict[str, Any]) -> TrainRequest:
    root.mkdir(parents=True, exist_ok=True)
    handle = guard.resolve_root(str(root))
    train_path = guard.confine(handle, "train.jsonl", "train_path")
    out_dir = guard.confine(handle, "out", "out_dir")
    handle.close()
    return TrainRequest(
        kind=kind,
        kind_version=1,
        config=dict(config),
        label_order=list(LABEL_ORDER),
        max_bytes=MAX_BYTES,
        seed=SEED,
        device="cpu",
        root=handle,
        train_path=train_path,
        out_dir=out_dir,
        time_limit_seconds=600,
        rss_limit_bytes=8 * 1024 * 1024 * 1024,
    )


def _softmax_rows(logits: mx.array) -> np.ndarray:
    return np.array(mx.softmax(logits, axis=-1), dtype=np.float32)


def _mlx_probs_c1(trained: Any, texts: list[str]) -> np.ndarray:
    cfg = trained.config
    nmin, nmax = int(cfg["ngram_min"]), int(cfg["ngram_max"])
    budget = trained.resource_budget
    ids_arr, doc_lens = c1_mod._encode_texts(texts, trained.max_bytes, budget)
    n_features = trained.vocab.ns.shape[0]
    max_nnz = max(1, min(n_features, trained.max_bytes * (nmax - nmin + 1)))
    idx_arr, val_arr = c1_mod._build_sparse_features(
        ids_arr, doc_lens, trained.vocab, nmin, nmax, max_nnz, budget
    )
    return _softmax_rows(trained.model(mx.array(idx_arr), mx.array(val_arr)))


def _mlx_probs_c3(trained: Any, texts: list[str]) -> np.ndarray:
    trained.model.eval()
    ids_arr = c3_mod._encode_texts(texts, trained.max_bytes, trained.resource_budget)
    mask = mx.array((ids_arr > 0).astype(np.float32))
    return _softmax_rows(trained.model(mx.array(ids_arr), mask))


def _build_kind(
    name: str, kind: Any, mlx_probs: Any, config: dict[str, Any], out_dir: Path, work: Path
) -> dict[str, Any]:
    rng = random.Random(SEED)  # noqa: S311  合成データ用の決定的乱数（暗号用途ではない）
    examples = _train_examples(rng)
    request = _make_request(work / name, kind=name, config=config)
    onnx_path = out_dir / f"{name}.onnx"
    try:
        budget = budget_mod.ResourceBudget(
            wall_seconds=float(request.time_limit_seconds),
            rss_bytes=request.rss_limit_bytes,
            device=request.device,
        )
        trained = kind.train(examples, request, budget)
        with onnx_path.open("wb") as f:
            kind.export_onnx(trained, f)
    finally:
        request.close_resources()
    onnx_bytes = onnx_path.read_bytes()

    named = _case_inputs(random.Random(SEED + 1))  # noqa: S311
    probs = np.concatenate(
        [mlx_probs(trained, [t for _n, t in named[i : i + 16]]) for i in range(0, len(named), 16)]
    )
    session = ReferenceEvaluator(onnx_path.as_posix())
    cases: list[dict[str, Any]] = []
    excluded = 0
    max_diff = 0.0
    for (cname, text), p in zip(named, probs, strict=True):
        ids = np.array([encode_bytes(text, MAX_BYTES)], dtype=np.int64)
        (ref,) = session.run(None, {"ids": ids})
        diff = float(np.max(np.abs(ref[0] - p)))
        if diff > ATOL_MLX_ONNX:
            raise SystemExit(f"{name}: MLX/ONNX reference prob diff {diff} at {cname}")
        max_diff = max(max_diff, diff)
        order = np.argsort(-p, kind="stable")
        margin = float(p[order[0]] - p[order[1]])
        if margin < NEAR_TIE_MARGIN:
            excluded += 1
            continue
        if int(np.argmax(ref[0])) != int(order[0]):
            raise SystemExit(f"{name}: reference argmax differs at {cname}")
        cases.append(
            {
                "name": cname,
                "input": text,
                "mlx_label_index": int(order[0]),
                "mlx_probs": [float(x) for x in p],
                "top2_margin": margin,
            }
        )
    seen = {c["mlx_label_index"] for c in cases}
    if len(seen) < 2:
        raise SystemExit(f"{name}: predictions are not diverse: {sorted(seen)}")
    return {
        "onnx": f"{name}.onnx",
        "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "max_bytes": MAX_BYTES,
        "label_order": LABEL_ORDER,
        "config": config,
        "seed": SEED,
        "near_tie_margin": NEAR_TIE_MARGIN,
        "excluded_near_tie": excluded,
        "max_ref_prob_diff": max_diff,
        "cases": cases,
    }


def main() -> None:
    """引数を解釈し、C1・C3 の fixture を生成する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="出力先ディレクトリ（存在すること）")
    parser.add_argument("--overwrite", action="store_true", help="既存ファイルの上書きを許可する")
    args = parser.parse_args()
    out_dir = Path(args.out).resolve()
    if not out_dir.is_dir():
        raise SystemExit(f"output directory does not exist: {out_dir}")
    targets = [out_dir / n for n in ("c1.onnx", "c3.onnx", "cases.json")]
    if not args.overwrite and any(t.exists() for t in targets):
        raise SystemExit("output files already exist; pass --overwrite to replace them")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        kinds = {
            "c1": _build_kind("c1", c1_mod.C1Kind(), _mlx_probs_c1, C1_CONFIG, out_dir, work),
            "c3": _build_kind("c3", c3_mod.C3Kind(), _mlx_probs_c3, C3_CONFIG, out_dir, work),
        }
    doc = {
        "_meta": {
            "description": (
                "MLX-side predictions for C1/C3 ONNX parity (REQ-32, TASK-32.1-2). "
                "Generated by trainer/tools/gen_onnx_parity_fixture.py on CPU with fixed seeds; "
                "see PROVENANCE.md. Do not relax values to make a test pass."
            ),
            "evidence": "test harness",
            "generated_on": f"{platform.system()} {platform.machine()}",
            "versions": {
                "mlx": getattr(mx, "__version__", "unknown"),
                "onnx": onnx.__version__,
                "numpy": np.__version__,
            },
        },
        "kinds": kinds,
    }
    (out_dir / "cases.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for k, v in kinds.items():
        sys.stdout.write(f"{k}: cases={len(v['cases'])} excluded={v['excluded_near_tie']}\n")


if __name__ == "__main__":
    main()
