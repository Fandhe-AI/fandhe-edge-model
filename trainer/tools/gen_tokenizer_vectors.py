"""合成 tokenizer とゴールデンベクタの生成器（REQ-41・TASK-41.4-5・#413。テストハーネス）。

役割: 合成 `tokenizer.json`（`tools/poc26/synthetic.py`）と、学習側の製品実装
`fandhe_edge_trainer.qwen2_tokenizer.Qwen2Tokenizer.encode` が出す期待 id 列を
`fixtures/preprocess/` へ書き出す。Rust 推論ランタイムの結合テスト
（`crates/runtime/tests/tokenizer_golden.rs`・`env_isolation.rs`）が全件一致を確かめる。
実物 `tokenizer.json` は同梱しないため、CI の一致確認は合成で行う。通信しない（REQ-38）。

手順: `uv --no-config run --locked --directory trainer python tools/gen_tokenizer_vectors.py`
`render_tokenizer()`・`render_vectors()` は pytest が生成物の古さ検出に使う。
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tools.poc26 import synthetic

from fandhe_edge_trainer.qwen2_tokenizer import Qwen2Tokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "fixtures" / "preprocess"
TOKENIZER_PATH = OUT_DIR / "tokenizer_synthetic.json"
VECTORS_PATH = OUT_DIR / "tokenizer_vectors_synthetic.json"
REAL_VECTORS_PATH = OUT_DIR / "tokenizer_vectors.json"

# 合成 merges・NFC・特殊トークン文字列に当たる手書き入力。
_HAND = [
    "hello", "hell", "help", " hello", " t", "ell", "hello hello", "é", "é",
    "<|im_end|>", "<|im_start|>user", "a\r\nb", "x\ty", "　", "'s", "123",
]  # fmt: skip
_ALPHABET = [*"hello t \t\r\n　'!019１", "あ", "漢", "é", "́", "\U0001f600"]


def render_tokenizer() -> str:
    """合成 `tokenizer.json` の全文（ASCII・決定的）。"""
    with tempfile.TemporaryDirectory() as tmp:
        doc = json.loads(synthetic.write_tokenizer_json(Path(tmp)).read_text("utf-8"))
    return json.dumps(doc, ensure_ascii=True, indent=2) + "\n"


def render_vectors() -> str:
    """合成 vectors JSON の全文（ASCII・決定的）。"""
    raw = render_tokenizer().encode("ascii")
    tok = Qwen2Tokenizer.from_bytes(raw)
    real = json.loads(REAL_VECTORS_PATH.read_text("utf-8"))["vectors"]
    rng = random.Random(413)  # noqa: S311  合成データ用の決定的乱数
    rand = ["".join(rng.choice(_ALPHABET) for _ in range(rng.randint(1, 30))) for _ in range(150)]
    inputs = [*(v["input"] for v in real), *_HAND, *rand]
    vectors = [
        {"name": f"syn_{i:03d}", "input": s, "ids": tok.encode(s)} for i, s in enumerate(inputs, 1)
    ]
    doc = {
        "_meta": {
            "description": (
                "tokenizer 合成 fixture のゴールデンベクタ（REQ-41・TASK-41.4-5・#413）。"
                "不一致時はコードを直し、ベクタを緩めない。"
            ),
            "input_representation": "tokenizer",
            "tokenizer_json_sha256": hashlib.sha256(raw).hexdigest(),
            "reference": "fandhe_edge_trainer.qwen2_tokenizer",
            "generated_on": "2026-10-17",
        },
        "vectors": vectors,
    }
    return json.dumps(doc, ensure_ascii=True, indent=2) + "\n"


if __name__ == "__main__":
    TOKENIZER_PATH.write_text(render_tokenizer(), encoding="utf-8", newline="\n")
    VECTORS_PATH.write_text(render_vectors(), encoding="utf-8", newline="\n")
