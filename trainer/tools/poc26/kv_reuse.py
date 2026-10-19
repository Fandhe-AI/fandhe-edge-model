"""PoC-26 の採点で共通プレフィックスの K/V を使い回す（REQ-41・TASK-41.11-3・#583）。

役割: 1 件の prompt（全選択肢に共通な先頭）を batch 1 で 1 回だけ forward して各層の K/V を持ち、
選択肢の続き（選択肢 token＋`<|im_end|>`）だけを RoPE 位置オフセット＝プレフィックス長で forward
する。スコア定義は `score.score_labels` と同じ（続き token の対数確率の合計）で、KV なしと数値的に
一致する。続きの先頭 token はプレフィックス末尾位置の隠れ状態から予測する。
`score.score_records` が `--kv-reuse` 指定時に呼ぶ。推論ランタイムには入らない（REQ-32）。

共通部分の取り方: 各選択肢の続きは先頭 token が互いに異なる保証がないが、プレフィックスは prompt
（`assistant\\n` まで）に固定する。選択肢側の共通 token は続きに含めて採点するため結果は変わらない。
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tools.poc26.qwen2_model import Attention, Qwen2Model


def _attend(
    attn: Attention, x: mx.array, mask: mx.array | str, past: tuple[mx.array, mx.array] | None
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    """`Attention.__call__` と同じ計算。past があれば RoPE 位置をずらして K/V を連結する。"""
    b, n, _ = x.shape
    off = 0 if past is None else past[0].shape[2]
    q = attn.q_proj(x).reshape(b, n, attn.n_heads, -1).transpose(0, 2, 1, 3)
    k = attn.k_proj(x).reshape(b, n, attn.n_kv, -1).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(b, n, attn.n_kv, -1).transpose(0, 2, 1, 3)
    q, k = attn.rope(q, offset=off), attn.rope(k, offset=off)
    kv = (k, v)  # 連結前（この呼び出しの分だけ）を返す
    if past is not None:
        k = mx.concatenate([mx.broadcast_to(past[0], (b, *past[0].shape[1:])), k], axis=2)
        v = mx.concatenate([mx.broadcast_to(past[1], (b, *past[1].shape[1:])), v], axis=2)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.head_dim**-0.5, mask=mask)
    return attn.o_proj(out.transpose(0, 2, 1, 3).reshape(b, n, -1)), kv


def prefix_states(
    model: Qwen2Model, prompt_ids: list[int]
) -> tuple[list[tuple[mx.array, mx.array]], mx.array]:
    """プレフィックスを batch 1 で forward し、各層の (K, V) と末尾の隠れ状態 `[1, H]` を返す。"""
    ids = mx.array(np.array([prompt_ids], dtype=np.int32))
    model._check_input(ids, None, with_logits=False)
    inner = model.model
    h = inner.embed_tokens(ids)
    cache = []
    for layer in inner.layers:
        a, kv = _attend(layer.self_attn, layer.input_layernorm(h), "causal", None)
        h = h + a
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        cache.append(kv)
    return cache, inner.norm(h)[:, -1, :]


def score_chunk_kv(
    model: Qwen2Model,
    cache: list[tuple[mx.array, mx.array]],
    h_last: mx.array,
    prefix_len: int,
    label_ids: list[list[int]],
    pad_id: int,
) -> np.ndarray:
    """ラベルの部分集合を、プレフィックスの K/V を使って 1 回の forward で採点する。

    続きは右 pad。続きの query はプレフィックス全体＋自分までの因果部分を見る（pad は後ろにあり
    実 token から見えない。pad 行は対角が許可されるので NaN にならない）。
    """
    k = len(label_ids)
    w = max(len(c) for c in label_ids)
    seqs = np.full((k, w), pad_id, dtype=np.int32)
    for i, c in enumerate(label_ids):
        seqs[i, : len(c)] = c
    ids = mx.array(seqs)
    model._check_input(ids, None, with_logits=False)
    causal = mx.tril(mx.ones((w, w), dtype=mx.bool_))
    mask = mx.concatenate([mx.ones((w, prefix_len), dtype=mx.bool_), causal], axis=1)
    inner = model.model
    h = inner.embed_tokens(ids)
    for layer, past in zip(inner.layers, cache, strict=True):
        a, _ = _attend(layer.self_attn, layer.input_layernorm(h), mask, past)
        h = h + a
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    flat = mx.concatenate([inner.norm(h).reshape(k * w, -1), h_last.astype(h.dtype)], axis=0)
    pos: list[int] = []
    tgt: list[int] = []
    owner: list[int] = []
    for i, c in enumerate(label_ids):
        for j, t in enumerate(c):
            pos.append(k * w if j == 0 else i * w + j - 1)  # j=0 はプレフィックス末尾から予測
            tgt.append(t)
            owner.append(i)
    hv = flat[mx.array(np.array(pos, dtype=np.int32))]
    logits = model.project(hv, source_shape=(k, prefix_len + w), with_mask=False).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, mx.array(np.array(tgt, dtype=np.int32)), reduction="none")
    mx.eval(ce)
    return -np.bincount(owner, weights=np.array(ce, dtype=np.float64), minlength=k)


def score_labels_kv(
    model: Qwen2Model,
    prompt_ids: list[int],
    label_ids: list[list[int]],
    pad_id: int,
    chunk: int,
) -> np.ndarray:
    """`score.score_labels` と同じ値を、プレフィックス 1 回の forward で求める。"""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    cache, h_last = prefix_states(model, prompt_ids)
    mx.eval([x for kv in cache for x in kv], h_last)
    parts = [
        score_chunk_kv(model, cache, h_last, len(prompt_ids), label_ids[i : i + chunk], pad_id)
        for i in range(0, len(label_ids), chunk)
    ]
    return np.concatenate(parts)
