"""PoC-26 学習・採点 CLI のプロンプト組み立てと選択肢採点（REQ-41・TASK-41.1-5・#390）。

役割: chat template（tools なし経路）の id 列の組み立て（`Prompting`）、各ラベルの token 列＋
`<|im_end|>` の条件付き対数尤度の合計（長さ正規化なし。`SCORE_CHUNK` 件ずつ forward）、
`pred.jsonl`・`raw_scores.jsonl` の行の生成を担う。対応づけは既存の対応づけ (b)
（`map_scores_to_choice`・`build_prediction_record`）に委ね、非有限スコアは `status:"error"`
（`scores` なし）になる。`train.py`（validation 採点）・`predict.py` から呼ばれる。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tools.poc26.assets import (
    Assets,
)
from tools.poc26.common import (
    Budget,
    as_input_error,
    invalid,
    too_large,
)
from tools.poc26.io_records import Record
from tools.poc26.qwen2_model import (
    Qwen2Model,
)
from tools.poc26.qwen2_tokenizer import Qwen2Tokenizer

from fandhe_edge_trainer.kinds.autoregressive import (
    build_prediction_record,
    map_scores_to_choice,
    prediction_record_to_json_line,
)

MAX_SCORE_TOKENS = 1 << 20


SCORE_CHUNK = 8  # 採点 1 回の forward に入れるラベル数（K が大きくてもメモリが線形に伸びない）


@dataclass
class Prompting:
    """プロンプトの組み立てに必要な固定情報。"""

    tok: Qwen2Tokenizer
    system: str
    label_order: list[str]
    label_ids: list[list[int]]
    pad_id: int
    vocab_size: int
    max_seq_length: int

    def encode(self, text: str) -> list[int]:
        """本文を encode する。語彙外 id・不正な文字は入力エラーにする。"""
        with as_input_error():
            return self._checked(self.tok.encode(text))

    def prompt_ids(self, text: str) -> list[int]:
        """採点用の prompt（末尾は `<|im_start|>assistant\\n`）。"""
        with as_input_error():
            return self._checked(
                self.tok.build_chat_ids(self.system, text, add_generation_prompt=True)
            )

    def train_ids(self, text: str, label: str) -> list[int]:
        """学習用の列（prompt + `{label}<|im_end|>\\n`）。"""
        with as_input_error():
            return self._checked(
                self.tok.build_chat_ids(self.system, text, label, add_generation_prompt=False)
            )

    def _checked(self, ids: list[int]) -> list[int]:
        # 語彙外 id は mlx の埋め込み参照が未定義動作になるため、forward の前に拒否する。
        if ids and max(ids) >= self.vocab_size:
            raise invalid("token id outside the model vocabulary")
        return ids

    def check_length(self, n: int) -> None:
        """切り詰めずに停止する（`truncated_count` は常に 0）。"""
        if n > self.max_seq_length:
            raise too_large("sequence exceeds max_seq_length")


def build_prompting(
    assets: Assets, system: str, label_order: list[str], vocab_size: int, max_seq_length: int
) -> Prompting:
    """`build_prompting_unchecked` の `ValueError`（入力値を含まない）を入力エラーへ写す。"""
    if max_seq_length > assets.config.max_position_embeddings:
        raise invalid("max-seq-length exceeds the model's max_position_embeddings")
    with as_input_error():
        return build_prompting_unchecked(assets, system, label_order, vocab_size, max_seq_length)


def build_prompting_unchecked(
    assets: Assets,
    system: str,
    label_order: list[str],
    vocab_size: int,
    max_seq_length: int,
) -> Prompting:
    """各ラベルの token 列＋`<|im_end|>` を作る。文脈依存の分割になるラベルは拒否する。"""
    tok = assets.tok
    end = tok.special_ids["im_end"]
    header = tok.encode("assistant\n")
    label_ids = []
    for label in label_order:
        body = tok.encode(label)
        # 学習の列は `assistant\n{label}` を 1 区間で encode する。採点でラベルを単独で encode した
        # 列と一致しないと、学習と採点で別の token 列を見ることになる。
        if not body or tok.encode("assistant\n" + label) != header + body:
            raise invalid("a label tokenizes differently in context")
        label_ids.append([*body, end])
    p = Prompting(
        tok,
        system,
        label_order,
        label_ids,
        tok.special_ids["endoftext"],
        vocab_size,
        max_seq_length,
    )
    for ids in label_ids:
        p._checked(ids)
    p._checked([p.pad_id, *tok.build_chat_ids("", "", add_generation_prompt=True)])
    return p


def score_chunk(
    model: Qwen2Model, prompt_ids: list[int], label_ids: list[list[int]], pad_id: int
) -> np.ndarray:
    """ラベルの部分集合を 1 回の forward で採点する（`score_labels` の 1 チャンク）。"""
    p, k = len(prompt_ids), len(label_ids)
    width = p + max(len(c) for c in label_ids)
    seqs = np.full((k, width), pad_id, dtype=np.int32)
    pos: list[int] = []
    tgt: list[int] = []
    owner: list[int] = []
    for i, c in enumerate(label_ids):
        seqs[i, :p] = prompt_ids
        seqs[i, p : p + len(c)] = c
        for j, t in enumerate(c):
            pos.append(i * width + p - 1 + j)  # この位置の出力が c[j] を予測する
            tgt.append(t)
            owner.append(i)
    h = model.hidden_states(mx.array(seqs))
    hv = h.reshape(-1, h.shape[-1])[mx.array(np.array(pos, dtype=np.int32))]
    logits = model.project(hv, source_shape=(k, width), with_mask=False).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, mx.array(np.array(tgt, dtype=np.int32)), reduction="none")
    mx.eval(ce)
    return -np.bincount(owner, weights=np.array(ce, dtype=np.float64), minlength=k)


def score_labels(
    model: Qwen2Model,
    prompt_ids: list[int],
    label_ids: list[list[int]],
    pad_id: int,
    chunk: int = SCORE_CHUNK,
) -> np.ndarray:
    """K 個のラベルの対数尤度の合計（長さ正規化なし）を求める。

    各列は `prompt + label + <|im_end|>` を右 pad し、因果マスクだけで採点する（pad は後ろにあり
    実 token から見えない）。隠れ状態は採点に要る位置だけ取り出して float32 の logits にする。
    K が大きいときのメモリのため、`chunk` 件ずつに分けて forward する（各ラベルの値は他のラベルに
    依存しないので、分割しても結果は変わらない。浮動小数の丸めの範囲）。
    """
    if chunk < 1:
        raise ValueError("chunk must be positive")
    parts = [
        score_chunk(model, prompt_ids, label_ids[i : i + chunk], pad_id)
        for i in range(0, len(label_ids), chunk)
    ]
    return np.concatenate(parts)


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(0.95 * len(ordered)) - 1, 0)]


def prepare_prompts(ctx: Prompting, records: list[Record], budget: Budget) -> list[list[int]]:
    """全レコードの prompt を作り、長さ・採点のメモリ上限を学習・採点の前に検査する。

    `max_seq_length` 超過は切り詰めず停止（20）。採点 1 件の総 token 数（K x 幅）が
    `MAX_SCORE_TOKENS` を超えても停止する（20）。
    """
    longest = max(len(c) for c in ctx.label_ids)
    prompts = []
    for rec in records:
        budget.check()
        ids = ctx.prompt_ids(rec.input)
        ctx.check_length(len(ids) + longest)
        if len(ctx.label_ids) * (len(ids) + longest) > MAX_SCORE_TOKENS:
            raise too_large("scoring batch (choices x sequence length) exceeds the limit")
        prompts.append(ids)
    return prompts


def score_records(
    model: Qwen2Model,
    ctx: Prompting,
    records: list[Record],
    prompts: list[list[int]],
    budget: Budget,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """各レコードを採点し、`pred.jsonl`・`raw_scores.jsonl` の行と採点時間の統計を返す。

    `prompts` は `prepare_prompts` で検査済みのもの。対応づけと予測レコードの組み立ては既存の
    対応づけ (b)（`map_scores_to_choice`・`build_prediction_record`）に委ねる。非有限の対数尤度は
    `status:"error"`（`scores` なし）になり、`raw_scores.jsonl` では null で記録する。
    """
    choice_ids = {x: [b + 1 for b in x.encode("utf-8")] for x in ctx.label_order}
    model.eval()
    pred, raw, secs = [], [], []
    for rec, ids in zip(records, prompts, strict=True):
        budget.check()
        t0 = time.perf_counter()
        loglik = score_labels(model, ids, ctx.label_ids, ctx.pad_id)
        secs.append(time.perf_counter() - t0)
        budget.check()  # forward の後にも確認する（最後の 1 件で超過しても 20）
        mapping = map_scores_to_choice(loglik, ctx.label_order, choice_ids)
        pred.append(
            prediction_record_to_json_line(
                build_prediction_record(rec.id, mapping, ctx.label_order)
            )
        )
        finite = [float(v) if math.isfinite(v) else None for v in loglik]
        raw.append(
            json.dumps(
                {"id": rec.id, "loglik": dict(zip(ctx.label_order, finite, strict=True))},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    stats = {
        "count": len(secs),
        "total_seconds": sum(secs),
        "mean_seconds": sum(secs) / len(secs),
        "p95_seconds": p95(secs),
        "max_seconds": max(secs),
        "choices_per_prompt": len(ctx.label_ids),  # 1 件あたりの選択肢数 K（26）
        "forward_chunk": SCORE_CHUNK,  # K を何件ずつ forward するか（合算は分割に依らない）
    }
    return pred, raw, stats
