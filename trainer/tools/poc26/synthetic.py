"""PoC-26 のテスト用合成データ（実重み・実データは使わない。REQ-41・TASK-41.1-5・#390）。

tokenizer 部分: 実物と同じ normalizer / pre_tokenizer / decoder 設定を持つ極小 `tokenizer.json`。
語彙は 256 バイト文字（id = バイト値）＋ 5 件の merges ＋ 特殊 3 件。モデル部分は PR-B で追加する。
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.poc26.qwen2_tokenizer import SPLIT_REGEX, bytes_to_unicode

# merges（順位順）と得られる token id。256 以降は merges の出現順に採番する。
MERGES = [("Ġ", "t"), ("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), ("e", "l")]
ID_ENDOFTEXT, ID_IM_START, ID_IM_END = 262, 263, 264


def write_tokenizer_json(directory: Path, *, merges_as_lists: bool = False) -> Path:
    """極小 `tokenizer.json` を `directory` へ書いてパスを返す。"""
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    for left, right in MERGES:
        vocab[left + right] = len(vocab)
    doc = {
        "added_tokens": [
            {"id": ID_ENDOFTEXT, "content": "<|endoftext|>", "special": True},
            {"id": ID_IM_START, "content": "<|im_start|>", "special": True},
            {"id": ID_IM_END, "content": "<|im_end|>", "special": True},
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": SPLIT_REGEX},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        },
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [list(m) if merges_as_lists else " ".join(m) for m in MERGES],
        },
    }
    path = directory / "tokenizer.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path
