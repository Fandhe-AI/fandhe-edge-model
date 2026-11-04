"""7B 相当の学習上限の見積もりを実重みなしで固定する（REQ-39・REQ-41・#689）。

形状だけの遅延配列で `apply_lora` の見積もりを通す（評価・確保はしない）。
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from fandhe_edge_trainer import limits
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.kinds.qwen2_lora import _activation_bytes
from fandhe_edge_trainer.qwen2_model import (
    Qwen2Config,
    Qwen2Model,
    apply_lora,
    load_peak_bytes,
)

CFG_7B = Qwen2Config(
    hidden_size=3584,
    num_hidden_layers=28,
    num_attention_heads=28,
    num_key_value_heads=4,
    intermediate_size=18944,
    vocab_size=152064,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    tie_word_embeddings=False,
    max_position_embeddings=32768,
)
#: 実機の ~/poc26/qwen2.5-7b-instruct の shard サイズ（バイト。合計 15,231,271,888）。
SHARDS_7B = [3945441440, 3864726352, 3864726424, 3556377672]
LORA = {"num_layers": 8, "rank": 8, "scale": 20.0, "dropout": 0.0, "seed": 0}


def _estimate(batch: int) -> int:
    """`apply_lora` が比べる見積もり（現在の重み＋LoRA 4 組＋活性）を再計算する。"""
    cfg = {"batch_size": batch, "max_seq_length": 512, "lora_num_layers": 8}
    return CFG_7B.estimated_params() * 2 + _lora_added() + _activation_bytes(CFG_7B, cfg)


def _lora_added() -> int:
    c = CFG_7B
    kv = c.num_key_value_heads * c.head_dim
    h, i = c.hidden_size, c.intermediate_size
    dims = 2 * (h + h) + 2 * (h + kv) + 3 * (h + i)  # q,o / k,v / gate,up,down の in+out
    return 16 * 8 * dims * 8


def _lazy_7b() -> Qwen2Model:
    m = Qwen2Model(CFG_7B)
    m.set_dtype(mx.bfloat16)  # 遅延のまま。評価しない
    return m


def test_7b_train_limit_value() -> None:
    """REQ-39・#689: qwen2_lora の学習 RSS 上限は 24 GiB、他 kind は 8 GiB のまま。"""
    assert limits.MAX_QWEN2_LORA_TRAIN_RSS_BYTES == 25769803776
    assert limits.max_train_rss_bytes_for_kind("qwen2_lora") == 25769803776
    assert limits.max_train_rss_bytes_for_kind("c3") == 8 << 30


def test_7b_apply_lora_estimate_values() -> None:
    """REQ-39・#689: 7B・LoRA 8 層・seq 512 の見積もりは batch 4 が 26,073,508,864、batch 2 が
    20,698,508,288 バイト。"""
    assert _estimate(4) == 26073508864
    assert _estimate(2) == 20698508288


def test_7b_apply_lora_batch4_rejected_batch2_accepted() -> None:
    """REQ-39・#689: 上限 24 GiB で batch 4 は拒否され、batch 2 は通る。"""
    limit = limits.MAX_QWEN2_LORA_TRAIN_RSS_BYTES
    for batch, ok in ((4, False), (2, True)):
        cfg = {"batch_size": batch, "max_seq_length": 512, "lora_num_layers": 8}
        m = _lazy_7b()
        kw = {**LORA, "rss_bytes": limit, "activation_bytes": _activation_bytes(CFG_7B, cfg)}
        if ok:
            apply_lora(m, **kw)
        else:
            with pytest.raises(WorkerError) as exc:
                apply_lora(m, **kw)
            assert exc.value.code == "limit_exceeded"


def test_7b_load_peak_between_12_and_24_gib() -> None:
    """REQ-39・#689: 7B の読み込みピークは 12 GiB 超（旧上限では拒否）かつ 24 GiB 未満。"""
    peak = load_peak_bytes(CFG_7B, SHARDS_7B, mx.bfloat16)
    assert 12 << 30 < peak < 24 << 30
