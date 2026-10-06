"""PoC-26 のテスト用合成データ（実重み・実データは使わない。REQ-41・TASK-41.1-5・#390）。

tokenizer 部分: 実物と同じ normalizer / pre_tokenizer / decoder 設定を持つ極小 `tokenizer.json`。
語彙は 256 バイト文字（id = バイト値）＋ 5 件の merges ＋ 特殊 3 件。
モデル部分: 極小 Qwen2 形状（hidden 16・2 層・heads 2・kv 1・intermediate 32）の config と重み。
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


#: 極小モデルの語彙数（合成 tokenizer の最大 id 264 を含む）。
TINY_VOCAB = 300


def tiny_config(*, tie: bool = True, rope_theta: float = 10000.0) -> dict:
    """極小 Qwen2 の config.json 内容。"""
    return {
        "model_type": "qwen2",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 32,
        "vocab_size": TINY_VOCAB,
        "rms_norm_eps": 1e-6,
        "rope_theta": rope_theta,
        "tie_word_embeddings": tie,
        "use_sliding_window": False,
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "max_position_embeddings": 64,
    }


def write_model_dir(
    directory: Path, *, seed: int = 0, tie: bool = True, rope_theta: float = 10000.0
) -> Path:
    """極小 Qwen2 の `config.json` と `model.safetensors`（float32・seed 固定）を書く。"""
    # tokenizer のテストが mlx / qwen2_model なしで動くよう、モデル部分だけ遅延 import する。
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from tools.poc26.qwen2_model import Qwen2Config, Qwen2Model

    (directory / "config.json").write_text(
        json.dumps(tiny_config(tie=tie, rope_theta=rope_theta)), encoding="utf-8"
    )
    model = Qwen2Model(Qwen2Config.from_file(directory / "config.json"))
    rng = np.random.default_rng(seed)
    weights = {
        k: mx.array(rng.normal(0.0, 0.2, v.shape).astype(np.float32))
        for k, v in tree_flatten(model.parameters())
    }
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    return directory
