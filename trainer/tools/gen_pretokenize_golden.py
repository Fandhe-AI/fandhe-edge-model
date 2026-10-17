"""Qwen2 事前分割の判定表と golden の生成器（REQ-41・TASK-41.4-3・#411。テストハーネス）。

役割: Rust 推論ランタイム（`crates/runtime/src/tokenizer`）の Split 状態機械が使う
`\\p{L}`・`\\p{N}` の範囲表と White_Space を Python 3.12（UCD 15.0）の `unicodedata` から
`unicode_tables.rs` として書き出し、Python 参照実装（`tools/poc26/qwen2_tokenizer.py`）の
`pre_tokenize` の結果を `fixtures/poc26/pretokenize_golden.json` に記録する。Rust 側の結合テスト
`crates/runtime/tests/tokenizer_pretokenize.rs` が全件一致を確かめる。通信しない（REQ-38）。

手順: `uv --no-config run --locked --directory trainer python tools/gen_pretokenize_golden.py`
`render_tables()`・`render_golden()` は pytest が生成物の古さ検出に使う。
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.poc26 import qwen2_tokenizer, synthetic
from tools.poc26.qwen2_tokenizer import Qwen2Tokenizer

ROOT = Path(__file__).resolve().parents[2]
TABLES_PATH = ROOT / "crates/runtime/src/tokenizer/unicode_tables.rs"
GOLDEN_PATH = ROOT / "fixtures/poc26/pretokenize_golden.json"
TOKENIZER_GOLDEN_PATH = ROOT / "fixtures/poc26/tokenizer_golden.json"
GENERATOR = "trainer/tools/gen_pretokenize_golden.py"


def _ranges(prefix: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for cp in range(0x110000):
        if unicodedata.category(chr(cp))[0] == prefix:
            if out and out[-1][1] == cp - 1:
                out[-1] = (out[-1][0], cp)
            else:
                out.append((cp, cp))
    return out


def _fmt(name: str, ranges: list[tuple[int, int]], doc: str) -> str:
    body = "".join(f"    (0x{lo:X}, 0x{hi:X}),\n" for lo, hi in ranges)
    return f"/// {doc}\n#[rustfmt::skip]\npub(super) const {name}: &[(u32, u32)] = &[\n{body}];\n"


def render_tables() -> str:
    """`unicode_tables.rs` の全文を返す（決定的）。"""
    major, minor, patch = (int(x) for x in unicodedata.unidata_version.split("."))
    ws = sorted(qwen2_tokenizer._WHITE_SPACE_CODEPOINTS)
    ws_body = "".join(f"    0x{c:X},\n" for c in ws)
    return (
        f"//! 生成物。手で編集しない（生成: `{GENERATOR}`。REQ-41・TASK-41.4-3・#411）。\n"
        "//!\n"
        "//! Split 状態機械（`pretokenize.rs`）の文字分類表。出典は Python 3.12 の `unicodedata`\n"
        f"//! （UCD {major}.{minor}.{patch}）と参照実装の White_Space。Unicode 版は承認済みの\n"
        "//! `unicode-normalization`（15.0）と揃えてあり、一致はテストで固定している。\n"
        "//! 出典の帰属とライセンス全文は `UNICODE-LICENSE.txt`（同ディレクトリ）と\n"
        "//! `fixtures/poc26/PROVENANCE.md`。\n\n"
        + _fmt("LETTER_RANGES", _ranges("L"), "`\\p{L}`（昇順・重なりなし・両端含む）。")
        + "\n"
        + _fmt("NUMBER_RANGES", _ranges("N"), "`\\p{N}`（昇順・重なりなし・両端含む）。")
        + "\n/// Unicode White_Space（昇順）。\n#[rustfmt::skip]\n"
        f"pub(super) const WHITE_SPACE: &[u32] = &[\n{ws_body}];\n"
    )


_HAND = [
    "I'm",
    "DON'T",
    "a, b",
    "hi!!\n",
    "x  y",
    "a\n\nb",
    "a \nb",
    "2024",
    "年1月",
    "x$5",
    "ab-cd",
    "a\x1cb",
    "x　　y",
    "x  y",
    "x\x85\x85y",
    "x  ",
    "a\r\nb",
    "'S",
    "I'LL",
    "x²",
    "",
    " ",
    "  ",
    " x",
    "x ",
    "  \n  x",
    "\t x",
    "!!\r\n\r\n",
    "x \n y",
    " x",
    "a​b",
    "a\x1fb",
    "café",
    "café",
    "Ⅻ",
    "٣٤",
    "'s'S'ſ'K'Ll'RE'x",
    "’s",
    "\U0001f600漢한 a",
    "a  \n\n  b",
    "'",
    "''s",
    "x'ver",
    "x've",
    "x'vE",
    "ſs'ſ",
]
_ALPHABET = [*"abXYzあ漢é1٣²Ⅻ!$,.-_ \t\r\n　 ​́'", "ſ", "\U0001f600"]


def _texts() -> list[str]:
    probes = [c["text"] for c in json.loads(TOKENIZER_GOLDEN_PATH.read_text("utf-8"))["cases"]]
    rng = random.Random(411)  # noqa: S311  合成データ用の決定的乱数（暗号用途ではない）
    rand = ["".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 40))) for _ in range(300)]
    return [*_HAND, *probes, *rand]


def _tokenizer(nfc: bool) -> Qwen2Tokenizer:
    with tempfile.TemporaryDirectory() as tmp:
        path = synthetic.write_tokenizer_json(Path(tmp))
        doc = json.loads(path.read_text("utf-8"))
    if not nfc:
        doc["normalizer"] = None
    return Qwen2Tokenizer(doc)


def build_cases() -> list[dict]:
    """golden のケース列（nfc=true 全件＋nfc=false は 1 件おき）。"""
    on, off = _tokenizer(True), _tokenizer(False)
    texts = _texts()
    cases = [{"text": t, "nfc": True, "pieces": on.pre_tokenize(t)} for t in texts]
    cases += [{"text": t, "nfc": False, "pieces": off.pre_tokenize(t)} for t in texts[::2]]
    return cases


def build_limit_cases() -> list[dict]:
    """NFC 前後の上限境界。U+0344 は NFC で 2 文字（4 バイト）に伸びる（REQ-39）。"""
    tok = _tokenizer(True)
    out = []
    for count in (65_536, 65_537):
        entry: dict = {"char": "̈́", "count": count, "nfc": True}
        try:
            entry["pieces_len"] = len(tok.pre_tokenize("̈́" * count))
        except qwen2_tokenizer.LimitExceededError:
            entry["error"] = "input text is too long"
        out.append(entry)
    return out


def render_golden() -> str:
    """golden JSON の全文を返す（決定的）。"""
    doc = {
        "source": {
            "generator": GENERATOR,
            "python_unicodedata_version": unicodedata.unidata_version,
        },
        "cases": build_cases(),
        "limit_cases": build_limit_cases(),
    }
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n"


if __name__ == "__main__":
    TABLES_PATH.write_text(render_tables(), encoding="utf-8")
    GOLDEN_PATH.write_text(render_golden(), encoding="utf-8")
