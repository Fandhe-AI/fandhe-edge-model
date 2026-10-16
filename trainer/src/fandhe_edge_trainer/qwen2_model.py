"""Qwen2 系ベースモデル（config・safetensors・tokenizer）の安全な読み込みと forward（MLX）。

REQ-41・REQ-39・TASK-41.5-2・#425。追加学習の種類 `qwen2_lora`（`kinds/qwen2_lora.py`）が、
利用者がローカルに置いたベースモデルを読むための入口 `load_base_model` と、選択肢採点用の
logits `[B, L, vocab]` を返す `Qwen2Model` を提供する。LoRA の付与と学習は #426、選択肢の採点は
#427 が本モジュールの上に載せる。PoC-26 の `tools/poc26/qwen2_model.py` から読み込みと forward
だけを移植したもの（PoC は証跡として残す）。推論ランタイム・配布パッケージには入らない（REQ-32）。
通信しない・重みを自動取得しない（REQ-38）。mlx はこのモジュールの import 時に読み込むため、
選択口（`kinds/__init__.py`）からは遅延 import のまま呼ぶこと。

読み込みの安全方針（REQ-39）:

- 重みは `.safetensors` の名前だけを受け付け、`mx.load(format="safetensors")` でのみ解釈する
  （pickle 系の `.bin`・`.pt` は名前の段階で拒否。偽装された中身は解析失敗で `invalid_data`）。
- 経路は `guard.confine` で閉じ込め、ディレクトリ fd 起点の `O_NOFOLLOW` でメンバーを開く
  （検証後の差し替え対策。ディレクトリは列挙しない）。
- 各ファイルは 1 度だけ読み、そのバイト列の sha256 を呼び出し側の期待値と照合してから解釈する
  （照合と使用の間で差し替えられない）。期待値のない読み込み経路は持たない。
- 読む前に fstat でサイズ（1 ファイル・合計・件数）とメモリ見積もりを `ResourceBudget` と比べる。
- 対応範囲外の設定（`_SUPPORTED`）は `invalid_config`、壊れたファイル・形状 / キーの不整合は
  `invalid_data`、sha256 不一致は `integrity_check_failed` で止める。
  メッセージにパス・内容を載せない。

Rust 側の学習リクエストへ「ベースモデルのディレクトリと期待 sha256」を載せる配線は後続
（本モジュールは引数で受け取るだけ）。
"""

from __future__ import annotations

import errno
import io
import math
import os
import re
import stat
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .budget import ResourceBudget
from .errors import WorkerError
from .exitcode import ExitCode
from .guard import RootHandle, confine
from .limits import (
    MAX_BASE_MODEL_CONFIG_BYTES,
    MAX_BASE_MODEL_FILE_BYTES,
    MAX_BASE_MODEL_TOTAL_BYTES,
    MAX_BASE_MODEL_WEIGHT_FILES,
    MAX_TOKENIZER_JSON_BYTES,
)
from .qwen2_tokenizer import Qwen2Tokenizer, limit_error, loads_json, sha256_hex

_SUPPORTED = "supported: model_type=qwen2, no rope_scaling, use_sliding_window=false"

#: forward が同時に保持するメモリの見積もり上限（暴走止め。読み込みの上限は `ResourceBudget`）。
#: 学習時の予算との連動は #426。
MAX_FORWARD_MEMORY_BYTES = 8 << 30

#: forward 入力の上限（token 数 B×L・logits 要素数 B×L×vocab・マスク要素数 B×L×L）。
MAX_FORWARD_TOKENS = 1 << 20
MAX_FORWARD_ELEMENTS = 1 << 31

#: config の `max_position_embeddings` の上限（これを超える設定は拒否）。
MAX_POSITIONS_LIMIT = 1 << 20

#: config の各整数の上限（Qwen2.5 の 0.5B〜数 B 級を許し、巨大割り当てを構築前に止める）。
_CONFIG_LIMITS = {
    "hidden_size": 16384,
    "num_hidden_layers": 128,
    "num_attention_heads": 256,
    "num_key_value_heads": 256,
    "intermediate_size": 65536,
    "vocab_size": 1 << 20,
}

_FLOAT_DTYPES = (mx.float16, mx.bfloat16, mx.float32)

CONFIG_NAME = "config.json"
TOKENIZER_NAME = "tokenizer.json"
_WEIGHT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.safetensors")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _invalid_config(message: str) -> WorkerError:
    return WorkerError("invalid_config", message, ExitCode.INVALID_INPUT)


def _invalid_data(message: str) -> WorkerError:
    return WorkerError("invalid_data", message, ExitCode.INVALID_INPUT)


def _invalid_request(message: str) -> WorkerError:
    return WorkerError("invalid_request", message, ExitCode.INVALID_INPUT)


@dataclass(frozen=True)
class Qwen2Config:
    """Qwen2 の形状設定（`config.json` から）。"""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int

    @property
    def head_dim(self) -> int:
        """1 head あたりの次元（hidden_size / num_attention_heads）。"""
        return self.hidden_size // self.num_attention_heads

    def estimated_params(self) -> int:
        """構築前に見積もるパラメータ総数（重み・bias・norm・tie でなければ lm_head）。"""
        h, kv = self.hidden_size, self.num_key_value_heads * self.head_dim
        layer = 2 * h * h + 2 * h * kv + h + 2 * kv + 3 * h * self.intermediate_size + 2 * h
        total = self.vocab_size * h + self.num_hidden_layers * layer + h
        return total + (0 if self.tie_word_embeddings else self.vocab_size * h)

    def expected_shapes(self) -> dict[str, tuple[int, ...]]:
        """モデルを構築せずに、期待する重みキー名と形状を返す（HF の命名・Linear は (out, in)）。"""
        h, d = self.hidden_size, self.head_dim
        q, kv, inter = (
            self.num_attention_heads * d,
            self.num_key_value_heads * d,
            self.intermediate_size,
        )
        shapes: dict[str, tuple[int, ...]] = {
            "model.embed_tokens.weight": (self.vocab_size, h),
            "model.norm.weight": (h,),
        }
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (self.vocab_size, h)
        for i in range(self.num_hidden_layers):
            p = f"model.layers.{i}."
            shapes |= {
                p + "self_attn.q_proj.weight": (q, h),
                p + "self_attn.k_proj.weight": (kv, h),
                p + "self_attn.v_proj.weight": (kv, h),
                p + "self_attn.o_proj.weight": (h, q),
                p + "self_attn.q_proj.bias": (q,),
                p + "self_attn.k_proj.bias": (kv,),
                p + "self_attn.v_proj.bias": (kv,),
                p + "mlp.gate_proj.weight": (inter, h),
                p + "mlp.up_proj.weight": (inter, h),
                p + "mlp.down_proj.weight": (h, inter),
                p + "input_layernorm.weight": (h,),
                p + "post_attention_layernorm.weight": (h,),
            }
        return shapes

    def expected_keys(self) -> set[str]:
        """モデルを構築せずに、期待する重みキー名の集合を返す（HF の命名）。"""
        keys = {"model.embed_tokens.weight", "model.norm.weight"}
        if not self.tie_word_embeddings:
            keys.add("lm_head.weight")
        for i in range(self.num_hidden_layers):
            p = f"model.layers.{i}."
            keys |= {p + f"self_attn.{n}_proj.weight" for n in "qkvo"}
            keys |= {p + f"self_attn.{n}_proj.bias" for n in "qkv"}
            keys |= {p + f"mlp.{n}_proj.weight" for n in ("gate", "up", "down")}
            keys |= {p + "input_layernorm.weight", p + "post_attention_layernorm.weight"}
        return keys

    @classmethod
    def from_bytes(cls, raw: bytes) -> Qwen2Config:
        """`config.json` のバイト列を検証して読む（検証済みバイト列の再利用口）。"""
        doc = loads_json(raw, "config.json")
        if not isinstance(doc, dict):
            raise _invalid_config("invalid config.json")
        if doc.get("model_type") != "qwen2":
            raise _invalid_config(f"unsupported model_type; {_SUPPORTED}")
        if doc.get("rope_scaling") is not None:
            raise _invalid_config(f"unsupported rope_scaling; {_SUPPORTED}")
        if doc.get("use_sliding_window", False):
            raise _invalid_config(f"unsupported use_sliding_window; {_SUPPORTED}")
        # forward の計算を固定している設定は、許可値だけを受理する（無視して誤った計算をしない）。
        if doc.get("hidden_act") != "silu":
            raise _invalid_config(f"unsupported hidden_act; {_SUPPORTED}, hidden_act=silu")
        for key, allowed in (
            ("attention_dropout", 0),
            ("partial_rotary_factor", 1),
            ("mlp_bias", False),
            ("attention_bias", True),  # Qwen2 は q/k/v に bias を持つ（実装が固定）
        ):
            if key in doc:
                v = doc[key]
                same = v == allowed and isinstance(v, bool) == isinstance(allowed, bool)
                if not same:
                    raise _invalid_config(f"unsupported {key}; {_SUPPORTED}")
        ints = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
        ]
        for k in ints:
            v = doc.get(k)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise _invalid_config(f"invalid config field: {k}")
            if v > _CONFIG_LIMITS.get(k, MAX_POSITIONS_LIMIT):
                raise limit_error(f"config field too large: {k}")
        if (
            doc["hidden_size"] % doc["num_attention_heads"]
            or doc["num_attention_heads"] % doc["num_key_value_heads"]
        ):
            raise _invalid_config("invalid config: heads do not divide hidden_size / kv heads")
        if (
            "head_dim" in doc
            and doc["head_dim"] != doc["hidden_size"] // doc["num_attention_heads"]
        ):
            raise _invalid_config(f"unsupported head_dim; {_SUPPORTED}")
        if (doc["hidden_size"] // doc["num_attention_heads"]) % 2:
            raise _invalid_config("invalid config: head_dim must be even (RoPE)")
        eps, theta = doc.get("rms_norm_eps"), doc.get("rope_theta")
        floats = {}
        for name, v in (("rms_norm_eps", eps), ("rope_theta", theta)):
            try:
                ok = isinstance(v, (int, float)) and not isinstance(v, bool)
                floats[name] = float(v) if ok else math.nan
            except OverflowError:  # float に収まらない巨大整数
                floats[name] = math.nan
            if not math.isfinite(floats[name]) or floats[name] <= 0:
                raise _invalid_config(f"invalid config field: {name}") from None
        tie = doc.get("tie_word_embeddings", True)
        if not isinstance(tie, bool):
            raise _invalid_config("invalid config field: tie_word_embeddings")
        return cls(
            **{k: doc[k] for k in ints},
            rms_norm_eps=floats["rms_norm_eps"],
            rope_theta=floats["rope_theta"],
            tie_word_embeddings=tie,
        )


class Attention(nn.Module):
    """GQA・q/k/v bias・RoPE（非 traditional）の因果 attention。"""

    def __init__(self, c: Qwen2Config) -> None:
        super().__init__()
        d = c.head_dim
        self.n_heads, self.n_kv, self.head_dim = c.num_attention_heads, c.num_key_value_heads, d
        self.q_proj = nn.Linear(c.hidden_size, self.n_heads * d, bias=True)
        self.k_proj = nn.Linear(c.hidden_size, self.n_kv * d, bias=True)
        self.v_proj = nn.Linear(c.hidden_size, self.n_kv * d, bias=True)
        self.o_proj = nn.Linear(self.n_heads * d, c.hidden_size, bias=False)
        self.rope = nn.RoPE(d, traditional=False, base=c.rope_theta)

    def __call__(self, x: mx.array, mask: mx.array | str) -> mx.array:
        b, n, _ = x.shape
        q = self.q_proj(x).reshape(b, n, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, n, self.n_kv, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, n, self.n_kv, -1).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(
            self.rope(q), self.rope(k), v, scale=self.head_dim**-0.5, mask=mask
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, n, -1))


class MLP(nn.Module):
    """SwiGLU（silu(gate) * up -> down。bias なし）。"""

    def __init__(self, c: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """pre-norm（RMSNorm）の Transformer ブロック。"""

    def __init__(self, c: Qwen2Config) -> None:
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | str) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Inner(nn.Module):
    """HF のキー名（`model.*`）に揃えるための内側の本体。"""

    def __init__(self, c: Qwen2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = [Block(c) for _ in range(c.num_hidden_layers)]
        self.norm = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)


class Qwen2Model(nn.Module):
    """Qwen2 本体。tie の場合は `embed_tokens.as_linear` で logits を出す。"""

    def __init__(self, c: Qwen2Config) -> None:
        super().__init__()
        self.config = c
        self.model = Qwen2Inner(c)
        if not c.tie_word_embeddings:
            self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def forward_bytes(self, b: int, n: int, with_mask: bool, with_logits: bool = True) -> int:
        """forward が同時に保持するバイト数の保守的な見積もり（確保前の拒否用。REQ-39）。

        合計 = 現在のパラメータ（実 dtype。LoRA を含む）
             + logits（B×L×vocab。float32 化を想定し 4 バイト以上）
             + マスク（B×L×L × 1。指定時のみ）
             + 注意スコア（B×heads×L×L。SDPA の実装によらず 1 層分を 4 バイト以上で）
             + 活性（B×L×(8×hidden + 3×intermediate)。残差・norm・q/k/v・MLP 中間を数個分）。
        `with_logits=False` は logits を除いた量（`hidden_states` 用。logits は `project` が自分の
        分を見積もる）。実測より大きめに出す設計で、厳密な上限ではなく暴走防止の目安。
        """
        c = self.config
        size = self.model.embed_tokens.weight.dtype.size
        wide = max(4, size)
        params = sum(v.nbytes for _, v in tree_flatten(self.parameters()))
        logits = b * n * c.vocab_size * wide if with_logits else 0
        mask = b * n * n if with_mask else 0
        scores = b * c.num_attention_heads * n * n * wide
        acts = b * n * (8 * c.hidden_size + 3 * c.intermediate_size) * size
        return params + logits + mask + scores + acts

    def _check_input(
        self, ids: mx.array, attention_mask: mx.array | None, *, with_logits: bool = True
    ) -> None:
        """確保の前に入力とメモリ見積もりを検証する（REQ-39）。

        形の不正は invalid_request、超過は limit_exceeded。入力の型・範囲・マスク・B・L の
        検証は常に行う。
        メモリ見積もりは `with_logits` で分かれる: True（`__call__`）は全位置 logits を含む全体、
        False（`hidden_states`）は logits を除いた同時保持量（パラメータ＋マスク＋注意スコア
        1 層分＋活性）。logits は `project` が自分の作る分を判定する。
        """
        if ids.ndim != 2 or 0 in ids.shape:
            raise _invalid_request("ids must be a non-empty [B, L] array")
        b, n = ids.shape
        c = self.config
        if n > c.max_position_embeddings:
            raise limit_error(f"sequence too long: {n} > {c.max_position_embeddings}")
        if not mx.issubdtype(ids.dtype, mx.integer):
            raise _invalid_request("ids must have an integer dtype")
        if b * n > MAX_FORWARD_TOKENS:
            raise limit_error(f"too many tokens: {b * n} > {MAX_FORWARD_TOKENS}")
        if with_logits and b * n * c.vocab_size > MAX_FORWARD_ELEMENTS:
            raise limit_error("logits would be too large for one forward")
        if attention_mask is not None:
            if attention_mask.shape != ids.shape:
                raise _invalid_request("attention_mask shape must equal ids shape")
            if b * n * n > MAX_FORWARD_ELEMENTS:
                raise limit_error("attention mask would be too large for one forward")
        if (
            self.forward_bytes(b, n, attention_mask is not None, with_logits)
            > MAX_FORWARD_MEMORY_BYTES
        ):
            raise limit_error("forward would exceed the memory limit")
        # 値の検証（embed 参照・マスク構築の前）。
        if mx.min(ids).item() < 0 or mx.max(ids).item() >= c.vocab_size:
            raise _invalid_request("token id out of range [0, vocab_size)")
        if attention_mask is not None:
            am = attention_mask
            if am.dtype != mx.bool_ and not mx.issubdtype(am.dtype, mx.integer):
                raise _invalid_request("attention_mask must be bool or integer")
            # 値域は縮小変換の前に元の dtype で検査する（uint64 の巨大値が 1 に化けない）。
            if mx.min(am).item() < 0 or mx.max(am).item() > 1:
                raise _invalid_request("attention_mask values must be 0 or 1")
            am = am.astype(mx.int32)
            if mx.min(am[:, 0]).item() != 1:
                raise _invalid_request(
                    "attention_mask rows must start with a real token (right pad only)"
                )
            if n > 1 and mx.max(am[:, 1:] - am[:, :-1]).item() > 0:
                raise _invalid_request("attention_mask must be 1s followed by 0s (right pad only)")

    def __call__(self, ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """`ids` `[B, L]` から logits `[B, L, vocab]` を返す（`hidden_states`＋`project`）。"""
        self._check_input(ids, attention_mask, with_logits=True)  # 全位置 logits を含む全体で判定
        return self.project(
            self._hidden(ids, attention_mask),
            source_shape=(ids.shape[0], ids.shape[1]),
            with_mask=attention_mask is not None,
        )

    def hidden_states(self, ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """`ids` `[B, L]` から最終 norm 後の隠れ状態 `[B, L, hidden]` を返す。

        採点・損失は必要な位置の隠れ状態だけを `project` へ渡し、全位置の `[B, L, vocab]` を
        float32 に広げない（REQ-39 のメモリ上限。#390 PR-C）。入力の型・範囲・マスク・B・L の検証は
        forward と同じで、メモリ見積もりは logits を除いた量（logits は `project` が判定する）。

        `attention_mask` は右 pad 用の `[B, L]`（真 = 実 token）。因果マスクと key 側の pad 除外を
        合成する。pad の query 行が全遮蔽で NaN にならないよう対角は常に許可する。
        **左 pad は非対応**（位置 id は 0 始まりで、pad 分だけ RoPE 位置がずれる）。

        入力は `max_position_embeddings`（config 必須）以下の長さ、B×L・B×L×vocab・B×L×L に
        上限があり、超過は limit_exceeded。

        dropout の有効 / 無効（train / eval）は呼び出し側の責務で、本 forward は切り替えない
        （PR-C の学習ループが `model.train()` / `model.eval()` を呼ぶ）。
        """
        self._check_input(ids, attention_mask, with_logits=False)
        return self._hidden(ids, attention_mask)

    def _hidden(self, ids: mx.array, attention_mask: mx.array | None) -> mx.array:
        """検証済みの入力から隠れ状態を計算する（検査は呼び出し側）。"""
        mask: mx.array | str = "causal"
        if attention_mask is not None:
            n = ids.shape[1]
            causal = mx.tril(mx.ones((n, n), dtype=mx.bool_))
            keys = (attention_mask != 0)[:, None, None, :]
            mask = (causal[None, None] & keys) | mx.eye(n, dtype=mx.bool_)[None, None]
        inner = self.model
        h = inner.embed_tokens(ids)
        for layer in inner.layers:
            h = layer(h, mask)
        return inner.norm(h)

    def project(self, h: mx.array, *, source_shape: tuple[int, int], with_mask: bool) -> mx.array:
        """隠れ状態 `[..., hidden]` を logits `[..., vocab]` へ射影する（tie なら埋め込み共有）。

        確保の前に形と logits の要素数（`MAX_FORWARD_ELEMENTS`）・同時保持量を検証する。同時保持量は
        呼び出し元が保持している隠れ状態・活性を含めて合算する: `source_shape=(B, L)`（`h` の元に
        なった `hidden_states` の入力形状）と `with_mask`（そのマスクの有無）から
        `forward_bytes(B, L, with_mask, with_logits=False)`（hidden_states が保持する量）を求め、
        この呼び出しが作る logits のバイト数と足して `MAX_FORWARD_MEMORY_BYTES` と比べる。
        """
        c = self.config
        if len(source_shape) != 2 or not all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in source_shape
        ):
            raise _invalid_request("source_shape must be (B, L) of positive ints")
        if h.ndim < 1 or h.shape[-1] != c.hidden_size or 0 in h.shape:
            raise _invalid_request("hidden states must be a non-empty [..., hidden_size] array")
        if not mx.issubdtype(h.dtype, mx.floating):
            raise _invalid_request("hidden states must have a floating dtype")
        rows = h.size // c.hidden_size
        if rows * c.vocab_size > MAX_FORWARD_ELEMENTS:
            raise limit_error("logits would be too large for one projection")
        held = self.forward_bytes(*source_shape, with_mask, with_logits=False)
        if held + rows * c.vocab_size * max(4, h.dtype.size) > MAX_FORWARD_MEMORY_BYTES:
            raise limit_error("projection would exceed the memory limit")
        if c.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)


def load_peak_bytes(config: Qwen2Config, sizes: list[int], dtype: mx.Dtype) -> int:
    """読み込みの同時保持量の最大（バイト）。各段階の max で上限判定に使う（REQ-39）。

    ファイル（シャード）i を順に処理し、`_read_shard` はバイト列を手放してから dtype 変換する。
    P_i = 先に変換済みのファイルの変換後サイズの和、conv_i = ファイル i の変換後サイズ
    （ファイルの dtype は読む前は不明なので 2 バイト幅と仮定して `S_i × dtype.size / 2`。
    float32 のファイルなら実際はこれ以下）、T = 最大テンソル 1 個分の余裕
    （`max(vocab, intermediate) × hidden × 4`）として:
      読み込み: P_i + S_i（バイト列）+ S_i（`mx.load` した重み。ファイルの dtype）
      変換:     P_i + S_i（読み込んだ重み）+ conv_i
      最後:     C（= params × dtype）+ T
    モデルの既定パラメータ（float32）は MLX の遅延評価で実体化されないまま `load_weights` で
    置き換わるため数えない（#552）。
    """
    t = max(config.vocab_size, config.intermediate_size) * config.hidden_size * 4
    peak = config.estimated_params() * dtype.size + t
    done = 0
    for s in sizes:
        conv = s * dtype.size // 2
        peak = max(peak, done + 2 * s, done + s + conv)
        done += conv
    return peak


def _check_keys(config: Qwen2Config, present: set[str], where: str) -> None:
    """重みキーの過不足を config 由来の期待値と照合する（tie と lm_head の矛盾も拒否）。"""
    expected = config.expected_keys()
    missing, extra = expected - present, present - expected
    if config.tie_word_embeddings and "lm_head.weight" in extra:
        raise _invalid_data(
            f"tie_word_embeddings=true conflicts with lm_head.weight present in {where}"
        )
    if missing or extra:
        hint = ""
        if "lm_head.weight" in missing:
            hint = "; tie_word_embeddings=false requires lm_head.weight"
        raise _invalid_data(
            f"weight keys mismatch (missing={len(missing)}, unexpected={len(extra)}); "
            f"{_SUPPORTED}{hint}"
        )


def _check_shard_tensors(shard: dict[str, mx.array], shapes: dict[str, tuple[int, ...]]) -> None:
    """変換（astype）の前に、1 ファイル分の全テンソルの形状と dtype（浮動小数のみ）を照合する。

    期待値は config 由来。未知のキーはキー検査で拒否する。関数に分けるのは、
    ループ変数が元テンソルを次のシャードまで掴み続けないようにするため
    （`load_peak_bytes` は前シャードの元テンソルを数えない）。
    """
    for k, w in shard.items():
        if k in shapes and tuple(w.shape) != shapes[k]:
            raise _invalid_data("weight shape mismatch")
        if k in shapes and w.dtype not in _FLOAT_DTYPES:
            raise _invalid_data("weight dtype must be float16, bfloat16 or float32")


def _check_expected(expected_sha256: object) -> list[str]:
    """期待 sha256 の写像を検証し、名前順の重みファイル名を返す（ファイルは開かない）。

    キーは `config.json`・`tokenizer.json` と、`*.safetensors` の単純なファイル名だけを許す
    （`pytorch_model.bin`・`../x`・区切り文字を含む名前は開く前に拒否）。
    """
    if not isinstance(expected_sha256, dict):
        raise _invalid_request("expected_sha256 must be a mapping of file name to sha256")
    weights: list[str] = []
    for name, sha in expected_sha256.items():
        if not isinstance(name, str) or not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
            raise _invalid_request("expected sha256 must be 64 lowercase hex characters")
        if name in (CONFIG_NAME, TOKENIZER_NAME):
            continue
        if not _WEIGHT_NAME_RE.fullmatch(name):
            raise _invalid_request("base model file name is not allowed")
        weights.append(name)
    for required in (CONFIG_NAME, TOKENIZER_NAME):
        if required not in expected_sha256:
            raise _invalid_request(f"expected sha256 for {required} is required")
    if not weights:
        raise _invalid_request("at least one .safetensors file is required")
    if len(weights) > MAX_BASE_MODEL_WEIGHT_FILES:
        raise limit_error(f"too many weight files: > {MAX_BASE_MODEL_WEIGHT_FILES}")
    return sorted(weights)


def _open_dir(root: RootHandle, rel_dir: str) -> int:
    """`root` 配下のベースモデルのディレクトリを閉じ込めて開き、ディレクトリ fd を返す。"""
    entry = confine(root, rel_dir, "base_model_dir")
    try:
        try:
            fd = os.open(
                entry.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=entry.parent_fd,
            )
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise WorkerError(
                    "symlink_not_allowed",
                    "base_model_dir is a symlink",
                    ExitCode.INVALID_INPUT,
                ) from e
            raise WorkerError(
                "invalid_path", "base_model_dir is not openable", ExitCode.INVALID_INPUT
            ) from e
    finally:
        entry.close()
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
        os.close(fd)
        raise WorkerError(
            "invalid_path", "base_model_dir is not a directory", ExitCode.INVALID_INPUT
        )
    return fd


def _open_member(dir_fd: int, name: str, limit: int, what: str) -> tuple[int, int]:
    """ディレクトリ fd 起点で通常ファイルを開き、(fd, サイズ) を返す。symlink・FIFO は拒否。"""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise WorkerError(
                "symlink_not_allowed", f"{what} is a symlink", ExitCode.INVALID_INPUT
            ) from e
        raise WorkerError(
            "invalid_path", f"{what} is missing or not openable", ExitCode.INVALID_INPUT
        ) from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise WorkerError(
                "invalid_path", f"{what} is not a regular file", ExitCode.INVALID_INPUT
            )
        if st.st_size > limit:
            raise limit_error(f"{what} too large: {st.st_size} bytes > {limit}")
    except BaseException:
        os.close(fd)
        raise
    return fd, st.st_size


def _read_member(
    dir_fd: int, name: str, limit: int, what: str, expected_sha: str, expected_size: int | None
) -> bytes:
    """メンバーを 1 度だけ読み、sha256 を照合した同じバイト列を返す（REQ-39）。

    `expected_size` があれば、サイズ確認後に変化していないことも確かめる（読み込み中の増加も
    `size + 1` まで読んで検出する）。不一致は `integrity_check_failed`（exit 70）。
    """
    fd, size = _open_member(dir_fd, name, limit, what)
    with os.fdopen(fd, "rb") as f:
        if expected_size is not None and size != expected_size:
            raise limit_error(f"{what} changed size after the size check")
        data = f.read(size + 1)
    if len(data) != size:
        raise limit_error(f"{what} changed size while reading")
    if sha256_hex(data) != expected_sha:
        raise WorkerError(
            "integrity_check_failed", f"{what} sha256 mismatch", ExitCode.RUNTIME_ERROR
        )
    return data


def _load_shard(data: bytes, what: str) -> dict[str, mx.array]:
    """照合済みのバイト列を safetensors として解釈する（`mx.load` のみ。pickle は読まない）。"""
    try:
        weights = mx.load(io.BytesIO(data), format="safetensors")
        mx.eval(weights)
    except (RuntimeError, ValueError, OSError):
        raise _invalid_data(f"invalid {what}") from None
    if not isinstance(weights, dict):
        raise _invalid_data(f"invalid {what}")
    return weights


@dataclass(frozen=True)
class BaseModel:
    """読み込み済みのベースモデル一式（`load_base_model` の戻り値）。

    `file_sha256` は照合に使った（= 解釈したバイト列の）sha256。後続の LoRA 学習・成果物の
    来歴（REQ-40）が参照する。
    """

    model: Qwen2Model
    tokenizer: Qwen2Tokenizer
    config: Qwen2Config
    file_sha256: dict[str, str]
    weight_bytes: int


def load_base_model(
    root: RootHandle,
    rel_dir: str,
    *,
    expected_sha256: dict[str, str],
    dtype: mx.Dtype,
    resource_budget: ResourceBudget,
) -> BaseModel:
    """`root` 配下のベースモデルのディレクトリを検証して読み、`dtype` のモデルを返す。

    `expected_sha256` は {ファイル名: sha256}。`config.json`・`tokenizer.json` と 1 件以上の
    `*.safetensors`（複数ならシャード。index は使わず、テンソルの和とキー検査で過不足を確かめる）
    を指定する。順序: 期待値の検証 -> config（照合後に解釈。対応範囲外はここで止まる）-> tokenizer
    （照合後に解釈。vocab_size と整合）-> 重みの種類・サイズ確認（読まない）-> メモリ見積もりと
    `resource_budget.rss_bytes` の比較 -> 1 ファイルずつ
    「読む・照合・解釈・形状 / dtype 検査・変換」（各ファイル後に `resource_budget.check()`）
    -> キー検査 -> モデル構築。
    """
    weight_names = _check_expected(expected_sha256)
    if dtype not in _FLOAT_DTYPES:
        raise _invalid_request("dtype must be float16, bfloat16 or float32")
    dir_fd = _open_dir(root, rel_dir)
    try:
        config = Qwen2Config.from_bytes(
            _read_member(
                dir_fd,
                CONFIG_NAME,
                MAX_BASE_MODEL_CONFIG_BYTES,
                CONFIG_NAME,
                expected_sha256[CONFIG_NAME],
                None,
            )
        )
        tokenizer = Qwen2Tokenizer.from_bytes(
            _read_member(
                dir_fd,
                TOKENIZER_NAME,
                MAX_TOKENIZER_JSON_BYTES,
                TOKENIZER_NAME,
                expected_sha256[TOKENIZER_NAME],
                None,
            )
        )
        if tokenizer.max_id >= config.vocab_size:
            raise _invalid_config("tokenizer has ids outside config vocab_size")
        sizes: list[int] = []
        for name in weight_names:  # 種類・サイズを読む前に確認する
            fd, size = _open_member(dir_fd, name, MAX_BASE_MODEL_FILE_BYTES, "weight file")
            os.close(fd)
            sizes.append(size)
        if sum(sizes) > MAX_BASE_MODEL_TOTAL_BYTES:
            raise limit_error(f"weight files too large: {sum(sizes)} bytes")
        if (
            config.estimated_params() * 2 > MAX_BASE_MODEL_TOTAL_BYTES
            or load_peak_bytes(config, sizes, dtype) > resource_budget.rss_bytes
        ):
            raise limit_error("model would exceed the memory budget")
        shapes = config.expected_shapes()
        weights: dict[str, mx.array] = {}
        unknown: set[str] = set()
        for name, size in zip(weight_names, sizes, strict=True):
            data = _read_member(
                dir_fd, name, MAX_BASE_MODEL_FILE_BYTES, "weight file", expected_sha256[name], size
            )
            shard = _load_shard(data, "weight file")
            del data
            _check_shard_tensors(shard, shapes)
            if shard.keys() & (weights.keys() | unknown):
                raise _invalid_data("duplicate weight key across files")
            for k in list(shard):  # 1 キーずつ変換して元を即解放する
                if k not in shapes:
                    del shard[k]  # 後段のキー検査で拒否するため名前だけ残す
                    unknown.add(k)
                    continue
                converted = shard.pop(k).astype(dtype)
                mx.eval(converted)
                weights[k] = converted
                del converted
            del shard
            resource_budget.check()
        _check_keys(config, set(weights) | unknown, "weight files")
        model = Qwen2Model(config)
        model.load_weights(list(weights.items()), strict=True)
        del weights
        mx.eval(model.parameters())
    finally:
        os.close(dir_fd)
    return BaseModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        file_sha256=dict(expected_sha256),
        weight_bytes=sum(sizes),
    )
