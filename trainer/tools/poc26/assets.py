"""PoC-26 学習・採点 CLI のベースモデル資産（REQ-41・TASK-41.1-5・#390）。

役割: system プロンプト（ラベルを宣言順に列挙。全文は追補 1 に記録）、`tokenizer.json`・
`tokenizer_config.json`・`config.json` の読み込みと sha256 照合（#387 で記録した期待値 `Pins`）、
ベースモデルの読み込み（PR-B の `load_qwen2` に委ね、`model.safetensors`・`config.json` の
sha256 照合・形状検証・メモリ見積もりを迂回しない）を担う。`train.py`・`predict.py`・`probe.py`
から呼ばれる。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from tools.poc26 import qwen2_model
from tools.poc26.common import (
    as_input_error,
    invalid,
    loads,
    read_input,
    sha256,
)
from tools.poc26.qwen2_model import (
    Qwen2Config,
    Qwen2Model,
    load_qwen2,
)
from tools.poc26.qwen2_tokenizer import MAX_TOKENIZER_JSON_BYTES, Qwen2Tokenizer

MAX_TOKENIZER_CONFIG_BYTES = 1 << 20


MAX_SYSTEM_PROMPT_BYTES = 64 * 1024


_HEAD = (
    "You are a tool selector for Playwright MCP. The user describes a browser task in Japanese. "
    "Reply with exactly one name from the list below and nothing else. "
    "Reply none when no tool applies.\nNames:"
)


def build_system_prompt(label_order: list[str]) -> str:
    """固定の system プロンプト（ラベルを宣言順に列挙する）。全文は追補 1 に記録する。"""
    return "\n".join([_HEAD, *(f"- {x}" for x in label_order)])


@dataclass
class Assets:
    """モデルディレクトリから読んだ tokenizer と各ファイルの sha256（`run.json` 用）。"""

    tok: Qwen2Tokenizer
    config: Qwen2Config
    hashes: dict[str, str]


@dataclass(frozen=True)
class Pins:
    """#387 でベースモデルを取得した時に記録した、各ファイルの期待 sha256（信頼できる記録）。

    読んだバイト列のハッシュが一致しなければ解釈・読み込みの前に拒否する（終了コード 64）。
    `tokenizer.json`・`tokenizer_config.json` も固定する理由: token id（学習・採点の入力）と
    chat template の前提が、記録したものと同一であることを保証するため（#387 の記録と一貫）。
    """

    model: str
    config: str
    tokenizer: str
    tokenizer_config: str


def check_pin(raw: bytes, expected: str, what: str) -> None:
    if sha256(raw) != expected:
        raise invalid(f"{what} sha256 does not match the expected value")


def load_assets(model_dir: Path, pins: Pins) -> Assets:
    """`tokenizer.json`・`tokenizer_config.json`・`config.json` を上限つきで 1 度ずつ読む。

    各バイト列の sha256 を `pins` と照合してから解釈する（不一致は 64）。

    `tokenizer_config.json` は `eos_token`=`<|im_end|>`・`pad_token`=`<|endoftext|>` と、
    `chat_template` が tools なし経路の骨格（`<|im_start|>`・`<|im_end|>`・`assistant`・
    `add_generation_prompt`）を含むことを確認する。これは**粗い整合確認**で、テンプレート全文の
    一致は保証しない。全文は sha256 を記録し、既知の値との照合を人間の実機確認とする
    （追補 1。`build_chat_ids` が固定で組み立てる形が実物と一致する根拠は sha256 照合）。
    """
    tok_raw = read_input(model_dir / "tokenizer.json", MAX_TOKENIZER_JSON_BYTES, "tokenizer.json")
    check_pin(tok_raw, pins.tokenizer, "tokenizer.json")
    with as_input_error():
        tok = Qwen2Tokenizer(loads(tok_raw, "tokenizer.json"))
    cfg_raw = read_input(
        model_dir / "tokenizer_config.json", MAX_TOKENIZER_CONFIG_BYTES, "tokenizer_config.json"
    )
    check_pin(cfg_raw, pins.tokenizer_config, "tokenizer_config.json")
    cfg = loads(cfg_raw, "tokenizer_config.json")
    template = cfg.get("chat_template") if isinstance(cfg, dict) else None
    if not isinstance(template, str):
        raise invalid("tokenizer_config.json chat_template must be a string")
    if cfg.get("eos_token") != "<|im_end|>" or cfg.get("pad_token") != "<|endoftext|>":
        raise invalid("tokenizer_config.json eos_token / pad_token do not match the expected")
    needed = ("<|im_start|>", "<|im_end|>", "assistant", "add_generation_prompt")
    if not all(s in template for s in needed):
        raise invalid("chat_template does not look like the expected Qwen2.5 template")
    # config.json は読んだバイト列の sha256 を照合してからパースする。`load_qwen2` も同じ期待値で
    # 読み直して照合するため、2 回の読み込みの内容は同一であることが保証される。
    model_cfg = read_input(model_dir / "config.json", qwen2_model.MAX_CONFIG_BYTES, "config.json")
    check_pin(model_cfg, pins.config, "config.json")
    with as_input_error():
        config = Qwen2Config.from_bytes(model_cfg)
    return Assets(
        tok,
        config,
        {
            "tokenizer_json_sha256": sha256(tok_raw),
            "tokenizer_config_sha256": sha256(cfg_raw),
            "chat_template_sha256": sha256(template.encode("utf-8")),
            "config_json_sha256": sha256(model_cfg),
        },
    )


def load_system_prompt(path: Path | None, label_order: list[str]) -> str:
    if path is None:
        return build_system_prompt(label_order)
    raw = read_input(path, MAX_SYSTEM_PROMPT_BYTES, "system prompt file")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise invalid("system prompt file is not valid utf-8") from None
    if not text.strip():
        raise invalid("system prompt file is empty")
    return text


def load_base_model(model_dir: Path, dtype: mx.Dtype, pins: Pins) -> tuple[Qwen2Model, str, int]:
    """PR-B の `load_qwen2` で読む。`model.safetensors`・`config.json` の sha256 照合も同関数。

    返す sha256 は照合済みの期待値（`pins.model`）。不一致・形状不正・上限超過は `load_qwen2` が
    構築前に拒否する（`ValueError` -> 64、`LimitExceededError` -> 20）。サイズは照合した
    バイト列の長さ（記録用）。
    """
    with as_input_error():
        model = load_qwen2(
            model_dir,
            dtype=dtype,
            expected_sha256=pins.model,
            expected_config_sha256=pins.config,
        )
    size = model.source_bytes  # 照合したバイト列の長さ（fstat の値ではない）
    return model, pins.model, size


def pins_from_args(a: argparse.Namespace) -> Pins:
    return Pins(a.model_sha256, a.config_sha256, a.tokenizer_sha256, a.tokenizer_config_sha256)
