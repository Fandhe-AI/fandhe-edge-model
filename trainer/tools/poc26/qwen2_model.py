"""PoC-26 用の Qwen2 モデルと LoRA（MLX 自作。REQ-41・TASK-41.1-5・#390）。

役割: ローカルの Qwen2.5 系 `config.json`・`model.safetensors` を読み、選択肢採点用の
logits `[B, L, vocab]` を返す forward と、mlx-lm と同形の LoRA を提供する。学習ループ・採点・CLI は
別モジュール（PR-C）。推論ランタイム・配布パッケージには入らない（REQ-32）。通信しない（REQ-38）。

対応範囲は Qwen2 の密結合・tie 可・RoPE スケーリングなし・sliding window なしに限る。
範囲外の設定・重みキーの過不足は `ValueError` で停止する（外部入力は読み込み前に stat で
サイズを検査。REQ-39）。
"""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

#: `config.json` の上限（64 KiB）と `model.safetensors` の上限（1 GiB。実物は約 988MB）。
MAX_CONFIG_BYTES = 64 * 1024
MAX_MODEL_BYTES = 1 << 30

_SUPPORTED = "supported: model_type=qwen2, no rope_scaling, use_sliding_window=false"


#: メモリ上の重みの上限（4 GiB。0.5B 級の float32 昇格 約 2 GB を許す）。
MAX_MODEL_MEMORY_BYTES = 4 << 30

#: config の各整数の上限（Qwen2.5 の 0.5B〜数 B 級を許し、巨大割り当てを構築前に止める）。
_CONFIG_LIMITS = {
    "hidden_size": 16384,
    "num_hidden_layers": 128,
    "num_attention_heads": 256,
    "num_key_value_heads": 256,
    "intermediate_size": 65536,
    "vocab_size": 1 << 20,
}

MAX_LORA_RANK = 256


def _open_regular(path: Path, limit: int, what: str) -> tuple[int, os.stat_result]:
    """symlink を辿らず通常ファイルだけを開き、サイズ上限を確認して (fd, stat) を返す。"""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise ValueError(f"cannot open {what}") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"{what} is not a regular file")
        if st.st_size > limit:
            raise ValueError(f"{what} too large: {st.st_size} bytes > {limit}")
    except BaseException:
        os.close(fd)
        raise
    return fd, st


def _read_limited(path: Path, limit: int, what: str) -> bytes:
    """通常ファイルを fd 経由で読む（open 後の fstat で上限確認。読み超過も拒否）。"""
    fd, _ = _open_regular(path, limit, what)
    with os.fdopen(fd, "rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{what} too large: > {limit} bytes")
    return data


def _identity(st: os.stat_result) -> tuple[int, int, int, int]:
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


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

    @classmethod
    def from_file(cls, path: Path) -> Qwen2Config:
        """`config.json` を検証して読む。対応範囲外は ValueError。"""
        raw = _read_limited(path, MAX_CONFIG_BYTES, "config.json")

        def _no_const(_: str) -> None:
            raise ValueError

        try:
            doc = json.loads(raw.decode("utf-8"), parse_constant=_no_const)
        except (ValueError, RecursionError):  # JSONDecodeError・UnicodeDecodeError を含む
            raise ValueError("invalid config.json") from None
        if not isinstance(doc, dict):
            raise ValueError("invalid config.json")
        if doc.get("model_type") != "qwen2":
            raise ValueError(f"unsupported model_type; {_SUPPORTED}")
        if doc.get("rope_scaling") is not None:
            raise ValueError(f"unsupported rope_scaling; {_SUPPORTED}")
        if doc.get("use_sliding_window", False):
            raise ValueError(f"unsupported use_sliding_window; {_SUPPORTED}")
        ints = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
        ]
        for k in ints:
            v = doc.get(k)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"invalid config field: {k}")
            if v > _CONFIG_LIMITS[k]:
                raise ValueError(f"config field too large: {k}")
        if (
            doc["hidden_size"] % doc["num_attention_heads"]
            or doc["num_attention_heads"] % doc["num_key_value_heads"]
        ):
            raise ValueError("invalid config: heads do not divide hidden_size / kv heads")
        if (doc["hidden_size"] // doc["num_attention_heads"]) % 2:
            raise ValueError("invalid config: head_dim must be even (RoPE)")
        eps, theta = doc.get("rms_norm_eps"), doc.get("rope_theta")
        for name, v in (("rms_norm_eps", eps), ("rope_theta", theta)):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError(f"invalid config field: {name}")
            if v <= 0:
                raise ValueError(f"invalid config field: {name}")
        tie = doc.get("tie_word_embeddings", True)
        if not isinstance(tie, bool):
            raise ValueError("invalid config field: tie_word_embeddings")
        return cls(
            **{k: doc[k] for k in ints},
            rms_norm_eps=float(eps),
            rope_theta=float(theta),
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

    def __call__(self, ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """`ids` `[B, L]` から logits `[B, L, vocab]` を返す。

        `attention_mask` は右 pad 用の `[B, L]`（真 = 実 token）。因果マスクと key 側の pad 除外を
        合成する。pad の query 行が全遮蔽で NaN にならないよう対角は常に許可する。
        **左 pad は非対応**（位置 id は 0 始まりで、pad 分だけ RoPE 位置がずれる）。

        dropout の有効 / 無効（train / eval）は呼び出し側の責務で、本 forward は切り替えない
        （PR-C の学習ループが `model.train()` / `model.eval()` を呼ぶ）。
        """
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
        h = inner.norm(h)
        if self.config.tie_word_embeddings:
            return inner.embed_tokens.as_linear(h)
        return self.lm_head(h)


class LoRALinear(nn.Module):
    """mlx-lm と同形の LoRA。a ~ U(±1/sqrt(in))・b = 0・float32。"""

    def __init__(self, base: nn.Module, rank: int, scale: float, dropout: float) -> None:
        super().__init__()
        out_dim, in_dim = base.weight.shape
        self.linear = base
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
        bound = 1.0 / math.sqrt(in_dim)
        self.lora_a = mx.random.uniform(-bound, bound, (in_dim, rank), dtype=mx.float32)
        self.lora_b = mx.zeros((rank, out_dim), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


_LORA_TARGETS = (
    ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ("mlp", ("gate_proj", "up_proj", "down_proj")),
)


def apply_lora(
    model: Qwen2Model, *, num_layers: int, rank: int, scale: float, dropout: float
) -> None:
    """base を freeze し、末尾 `num_layers` ブロックの 7 Linear を LoRA に置き換える。"""
    layers = model.model.layers
    for v in (num_layers, rank):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("invalid LoRA rank / num_layers: must be int")
    if not 1 <= num_layers <= len(layers):
        raise ValueError(f"num_layers out of range: 1..{len(layers)}")
    if not 1 <= rank <= MAX_LORA_RANK:
        raise ValueError(f"invalid LoRA rank: 1..{MAX_LORA_RANK}")
    for v in (scale, dropout):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError("invalid LoRA scale / dropout: must be finite number")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("invalid LoRA dropout: 0 <= dropout < 1")
    model.freeze()
    for block in layers[-num_layers:]:
        for owner, names in _LORA_TARGETS:
            parent = getattr(block, owner)
            for name in names:
                setattr(parent, name, LoRALinear(getattr(parent, name), rank, scale, dropout))


def trainable_parameter_count(model: nn.Module) -> int:
    """学習可能パラメータの総数。"""
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


def load_qwen2(model_dir: Path, *, dtype: mx.Dtype) -> Qwen2Model:
    """`config.json`・`model.safetensors` を検証して読み、`dtype` へ揃えたモデルを返す。

    `model.safetensors` は 1 GiB 上限を読み込み前に確認し、キーの過不足は `strict=True` の
    前に検査する。
    """
    model_dir = Path(model_dir)
    config = Qwen2Config.from_file(model_dir / "config.json")
    params = config.estimated_params()
    # 構築（nn.Linear 等の割り当て）より前に、ファイル上とメモリ上の大きさを見積もって拒否する。
    if params * 2 > MAX_MODEL_BYTES or params * dtype.size > MAX_MODEL_MEMORY_BYTES:
        raise ValueError("config implies a model larger than the supported size limit")
    model = Qwen2Model(config)
    path = model_dir / "model.safetensors"
    fd, before = _open_regular(path, MAX_MODEL_BYTES, "model.safetensors")
    try:
        # mx.load は fd を受け取れないため、fd を開いたまま読み、前後の同一性を比較して
        # 検査後の差し替え（TOCTOU）を検出する。
        try:
            weights = mx.load(str(path))
            mx.eval(weights)
        except (RuntimeError, ValueError, OSError):
            raise ValueError("invalid model.safetensors") from None
        after, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
    finally:
        os.close(fd)
    if not (_identity(before) == _identity(after) == _identity(current)):
        raise ValueError("model.safetensors changed while loading")
    if not isinstance(weights, dict):
        raise ValueError("invalid model.safetensors")
    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing, extra = expected - weights.keys(), weights.keys() - expected
    if config.tie_word_embeddings and "lm_head.weight" in extra:
        raise ValueError(
            "tie_word_embeddings=true conflicts with lm_head.weight present in model.safetensors"
        )
    if missing or extra:
        hint = ""
        if "lm_head.weight" in missing:
            hint = "; tie_word_embeddings=false requires lm_head.weight"
        raise ValueError(
            f"weight keys mismatch (missing={len(missing)}, unexpected={len(extra)}); "
            f"{_SUPPORTED}{hint}"
        )
    model.load_weights([(k, v.astype(dtype)) for k, v in weights.items()], strict=True)
    mx.eval(model.parameters())
    return model
