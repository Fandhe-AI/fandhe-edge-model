"""kind="autoregressive"（バイト単位の小型自己回帰 decoder。REQ-19b・TASK-19b.1-1・#79）。

PoC-24（`docs/spec/03-poc/model-kind-selector/scripts/kinds/autoregressive_kind.py`。
`ByteDecoder`・`DecoderLayer`・`score_choices`・対応づけ (b) 相当）の移植。対応づけ
(b) は、入力を条件に各選択肢の表記＋終端（EOS）の条件付き対数尤度（長さ正規化
なし）を求め、選択肢間で softmax を取って事後確率とするもの（PoC-24
`preregistration.md` 3 節で事前登録済み）。以下の点は PoC-24 から意図的に変更している。

1. **train データのみで学習する**（`kinds/c3.py` の docstring 1 番と同じ理由）。
   PoC-24 は validation 正解率で早期終了・最良エポック復元・grid 探索（S/M）を
   行っていたが、本ワーカーはこれを一切持ち込まない。代わりに固定エポック数
   （既定 `epochs=40`。PoC-24 の S サイズと学習設定〔`layers=2, dims=128,
   heads=4, dropout=0.1, lr=5e-4, warmup_steps=100, weight_decay=1e-4,
   batch_size=32`〕を既定値として採用）で最後まで学習し、最終エポックの重みを
   そのまま使う。`--config-from` も持ち込まない。
2. **選択肢の対応づけ（対応づけ (b)）を ONNX グラフの内側に組み込む**
   （`_export_ar_onnx`）。PoC-24 は Python 側（`score_choices`）で MLX モデルを
   都度呼び出していたが、本実装は C1・C3 と同じ入出力契約
   （`ids: INT64 ["N","T"]` → `probs: FLOAT ["N", K]`）を守るため、K 個の
   選択肢すべてを 1 回の ONNX 実行でスコアリングするグラフを書き出す
   （§2.3。推論ランタイム側に種類固有の前処理・デコードを持ち込まない）。
3. **詰め物がトークン列の途中に混ざっても単独推論とバッチ推論が一致する**
   よう、key マスクと位置 id の計算を作り直す（REQ-28）。PoC-24 は
   `_pad_batch`（バッチ内の最大長に右詰めパディング）した後、パディング済みの
   `x` をそのまま decoder へ渡し、位置 id は `mx.arange(T)` を無条件に使って
   いた。バッチの入力領域を `[SEP | choice | EOS | PAD]` の前に連結する設計では、
   バッチ内の他系列より短い入力は SEP の手前に詰め物（id=0）が入り、
   `mx.arange(T)` によるナイーブな位置付けだと同じ入力でもバッチ内の位置が
   ずれる（`kinds/c3.py` の docstring 2 番と同種の問題。PoC-16 の診断
   `scripts/diagnose_batch_dependence.py` が C3 で実際に検出した不一致と
   同根）。本実装は次の 2 点を MLX 側フォワード（`ByteDecoder.__call__`）と
   ONNX 側グラフ（`_export_ar_onnx`）の両方に実装し、同じ 1 箇所の定数
   （`_MASK_NEG_VALUE`）を参照する:
   (a) key 側の詰め物マスク（`full_ids > 0`）を causal マスクへ加算する。
       ただし対角成分（自分自身の key）は詰め物であっても常に許可する
       （マスク行列の docstring 参照。詰め物のみで構成される query 行が
       全て `-inf` になり softmax が NaN になるのを防ぐ。空入力
       `encode_bytes("") == [0]` の行がこれに当たる）。
   (b) 位置 id を `arange` ではなく `CumSum(full_ids > 0) - 1`（0 未満は 0 へ
       clamp）で求める。実トークンは詰め物の位置に関係なく連続した位置を
       受け取る。
4. **選択肢ブロックを固定幅（`M = max_label_len + 1`）にし、decoder を
   1 回だけ実行する**（§2.3）。K 個の選択肢を `[N, K, T+1+M]` へ展開してから
   `[N*K, T+1+M]` へ reshape し、1 回のバッチ推論として decoder を通す。
   選択肢領域の位置 `T..T+M-1` のロジットから、次のトークン
   `choice_tokens[k][0..M)` の対数確率を取り出し、静的マスク
   （`choice_tokens != PAD`）で EOS 以降の詰め物を除いてから合計する。
5. **選択肢（`label_order` の各要素）は NFKC 正規化しない**。学習・推論の
   入力（`input`）は `encoding.encode_bytes`（NFKC 正規化を含む。C1・C3 と
   共通）でエンコードするが、選択肢は選択肢 ID そのものであり、正規化で
   複数の表記が同一の選択肢へ縮退してよい対象ではないため、生の UTF-8
   バイト+1 でエンコードする（`_encode_choices`）。

opset 13 の制約（LayerNormalization は opset 17・Gelu は opset 20 から）により、
LayerNorm・GELU（厳密形。`math.erf` 相当）・multi-head attention はいずれも
基本演算（`ReduceMean`・`Sub`・`Mul`・`Sqrt`・`Div`・`Erf`・`MatMul`・`Softmax` 等）
で手組みする。MLX 側の既定値（`nn.LayerNorm` の eps=1e-5・population variance、
`nn.gelu` の厳密形 `x*(1+erf(x/sqrt(2)))/2`、`nn.Linear` の重み形状
`(output_dims, input_dims)`）は `trainer/.venv` の `mlx/nn/layers/
{normalization,activations,linear}.py` を実装時に確認済み（コード内の該当箇所に
根拠を記す）。

#79 の範囲はモデル・学習・ONNX 書き出し・選択口への登録までで、Python 側で
1 件ずつの予測レコードを組み立てて評価器へ渡す処理（#80・TASK-19b.2）は
含まない。`_score_choices_mlx`（対応づけ (b) の MLX 実装。一致試験・訓練後の
簡易正解率確認に使う）は #80 が再利用できるよう公開関数として残す。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import IO, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .. import budget as budget_mod
from ..contract import TrainExample, TrainRequest
from ..encoding import encode_bytes
from ..errors import WorkerError, truncate_list_for_message
from ..exitcode import ExitCode
from ..limits import (
    MAX_AR_ATTENTION_ELEMENTS,
    MAX_AR_BATCH_SIZE,
    MAX_AR_DIMS,
    MAX_AR_EPOCHS,
    MAX_AR_EXPORT_ATTENTION_ELEMENTS,
    MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS,
    MAX_AR_HEADS,
    MAX_AR_LAYERS,
    MAX_AR_LR,
    MAX_AR_WARMUP_STEPS,
    MAX_AR_WEIGHT_DECAY,
)

KIND = "autoregressive"
KIND_VERSION = 1

#: 特殊トークン（PoC-24 と同じ値）。`encoding.encode_bytes` が使う「バイト値+1」
#: の語彙（1..256）に、選択肢との区切り（SEP）・終端（EOS）を追加する。
#: PAD=0 は `encoding.encode_bytes` の詰め物規約と共通（空入力は `[0]` になる。
#: クラス docstring 3 番参照）。
PAD, SEP, EOS = 0, 257, 258
VOCAB_SIZE = 259

#: 詰め物位置をマスクする際に加える負の大きな値（`kinds/c3.py::_MASK_NEG_VALUE`
#: と同じ考え方）。MLX 側フォワード（`_build_additive_mask`）と ONNX 側グラフ
#: （`_export_ar_onnx`）の両方がこの 1 箇所を参照する。
_MASK_NEG_VALUE = -1e9

#: 既定ハイパーパラメータ（クラス docstring 1 番。PoC-24 の S サイズ・学習設定を踏襲）。
DEFAULT_CONFIG: dict[str, Any] = {
    "layers": 2,
    "dims": 128,
    "heads": 4,
    "dropout": 0.1,
    "lr": 5e-4,
    "warmup_steps": 100,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 40,
}

_CONFIG_FIELDS = set(DEFAULT_CONFIG)


def _validate_config(cfg: dict[str, Any]) -> None:
    """`request.config` によるハイパーパラメータ上書きを検証する（`kinds/c3.py::
    _validate_config` と同じ考え方。TASK-19.1 の入口）。
    """
    unknown = set(cfg) - _CONFIG_FIELDS
    if unknown:
        raise WorkerError(
            "invalid_config",
            f"unknown config fields: {truncate_list_for_message(sorted(unknown))}",
            ExitCode.INVALID_INPUT,
        )

    def _bounded_int(name: str, lower: int, upper: int) -> None:
        v = cfg[name]
        if not isinstance(v, int) or isinstance(v, bool) or not (lower <= v <= upper):
            raise WorkerError(
                "invalid_config",
                f"config.{name} must be an integer in [{lower}, {upper}]",
                ExitCode.INVALID_INPUT,
            )

    _bounded_int("layers", 1, MAX_AR_LAYERS)
    _bounded_int("dims", 1, MAX_AR_DIMS)
    _bounded_int("heads", 1, MAX_AR_HEADS)
    _bounded_int("epochs", 1, MAX_AR_EPOCHS)
    _bounded_int("batch_size", 1, MAX_AR_BATCH_SIZE)
    _bounded_int("warmup_steps", 1, MAX_AR_WARMUP_STEPS)

    if cfg["dims"] % cfg["heads"] != 0:
        raise WorkerError(
            "invalid_config",
            "config.dims must be divisible by config.heads",
            ExitCode.INVALID_INPUT,
        )

    def _finite_number(
        name: str, lo: float, hi: float, *, lo_inclusive: bool, hi_inclusive: bool
    ) -> None:
        v = cfg[name]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise WorkerError(
                "invalid_config", f"config.{name} must be a finite number", ExitCode.INVALID_INPUT
            )
        lo_ok = (lo <= v) if lo_inclusive else (lo < v)
        hi_ok = (v <= hi) if hi_inclusive else (v < hi)
        if not (lo_ok and hi_ok):
            lo_b, hi_b = ("[" if lo_inclusive else "("), ("]" if hi_inclusive else ")")
            raise WorkerError(
                "invalid_config",
                f"config.{name} must be in {lo_b}{lo}, {hi}{hi_b}",
                ExitCode.INVALID_INPUT,
            )

    _finite_number("lr", 0.0, MAX_AR_LR, lo_inclusive=False, hi_inclusive=True)
    _finite_number("weight_decay", 0.0, MAX_AR_WEIGHT_DECAY, lo_inclusive=True, hi_inclusive=True)
    _finite_number("dropout", 0.0, 1.0, lo_inclusive=True, hi_inclusive=False)


def _is_finite_loss(value: float) -> bool:
    """学習ループの発散検出（`math.isfinite` を切り出した薄いラッパー）。

    `_validate_config` も同じ `math.isfinite` を使うため、テストで発散だけを
    模擬したい場合にモジュールレベルの `math.isfinite` をそのまま monkeypatch
    すると config 検証まで巻き込んでしまう。学習ループのこの 1 箇所だけを
    差し替えられるよう独立した関数にしてある（`tests/test_ar_train.py::
    test_ar_train_diverges_is_reported_as_pending` 参照）。
    """
    return math.isfinite(value)


def _reject_duplicate_choice_encoding(label_order: list[str], ids_list: list[list[int]]) -> None:
    """`label_order[i]` の各エンコード結果に重複があれば拒否する（防御的検査。
    `_encode_choices` docstring 参照）。UTF-8 エンコードは相異なる文字列に対して
    単射なので実際には起こり得ないが、対応づけ (b) の正しさが選択肢ごとの
    系列の一意性に依存するため、独立した純粋関数として切り出し fail-closed に
    検証する（`label` の値そのものはエラーメッセージへ含めない。security.md。
    添字だけを報告する）。
    """
    seen: set[tuple[int, ...]] = set()
    for i, ids in enumerate(ids_list):
        key = tuple(ids)
        if key in seen:
            raise WorkerError(
                "invalid_config",
                f"label_order[{i}] encodes to the same byte sequence as another label",
                ExitCode.INVALID_INPUT,
            )
        seen.add(key)


def _encode_choices(label_order: list[str]) -> dict[str, list[int]]:
    """選択肢（label）を生の UTF-8 バイト+1 へエンコードする（クラス docstring 5 番）。

    `contract.py::_validate_label_order` が非空・256 バイト以下・重複無しを
    検証済みだが、エンコード後の系列が偶然一致した場合に備え、ここでも
    `_reject_duplicate_choice_encoding` で防御的に衝突を検出する。
    """
    ids_list = [[b + 1 for b in label.encode("utf-8")] for label in label_order]
    _reject_duplicate_choice_encoding(label_order, ids_list)
    return dict(zip(label_order, ids_list, strict=True))


def _ar_param_count(layers: int, dims: int, heads: int, max_len: int) -> int:
    """`ByteDecoder(layers, dims, heads, dropout, max_len)` のパラメータ総数を、
    モデルを実際に構築せずに解析的に求める（`kinds/c3.py::_c3_param_count` と
    同じ考え方。サイズ超過のモデルを構築する前に拒否するため）。

    内訳: `embed`（`VOCAB_SIZE × dims`）・`pos`（`max_len × dims`）・
    各 `DecoderLayer`（`ln1`・`ln2` 各 `2*dims`、`wq`/`wk`/`wv`/`wo` 各
    `dims*dims+dims`、`ff1`（`dims*(4*dims)+4*dims`）、`ff2`
    （`(4*dims)*dims+dims`）。1 層あたり `12*dims^2 + 13*dims`）・最終 `ln`
    （`2*dims`）・`out`（`dims*VOCAB_SIZE+VOCAB_SIZE`）。`heads`・`dropout` は
    パラメータを持たない。
    """
    embed_params = VOCAB_SIZE * dims
    pos_params = max_len * dims
    layer_params = 12 * dims * dims + 13 * dims
    final_ln_params = 2 * dims
    out_params = dims * VOCAB_SIZE + VOCAB_SIZE
    return embed_params + pos_params + layers * layer_params + final_ln_params + out_params


def _build_additive_mask(full_ids: mx.array) -> mx.array:
    """causal self-attention 用の加算マスク（`[B, 1, L, L]`）を作る。

    クラス docstring 3 番 (a) の不変条件（key 側の詰め物マスクと causal マスクを
    足し合わせるが、対角成分は詰め物であっても常に許可する）を実装する。
    対角成分を強制的に許可しない場合、詰め物だけで構成される query 行
    （例: `encode_bytes("") == [0]` の行の位置 0）が全て `-inf` になり softmax が
    NaN になる（その NaN は key としてこの位置を参照する後続の全ての行へ
    伝播し、REQ-28 の一致契約を壊す）。対角成分は自分自身の hidden state にしか
    影響せず、詰め物位置の出力自体は決して gather されない（`_score_choices_mlx`・
    `_export_ar_onnx` のいずれも選択肢領域の位置だけを読む）ため、常に許可しても
    対応づけ (b) の結果は変わらない。

    ONNX 側グラフ（`_export_ar_onnx`）もまったく同じ式（causal 比較 →
    詰め物ペナルティの加算 → 対角成分を `(1 - diag)` で強制的にゼロへ戻す）を
    実装する（opset 13 に `Trilu`/`EyeLike` を使わず `Range`/`Greater`/`Equal` で
    組む必要があるため、比較演算だけで完結するこの式にしてある）。
    """
    length = full_ids.shape[1]
    idx = mx.arange(length)
    row = idx[:, None]
    col = idx[None, :]
    causal_f = (col > row).astype(mx.float32) * _MASK_NEG_VALUE  # [L, L]
    valid_key = (full_ids > 0).astype(mx.float32)  # [B, L]
    pad_pen = (1.0 - valid_key) * _MASK_NEG_VALUE  # [B, L]
    base = causal_f[None, :, :] + pad_pen[:, None, :]  # [B, L, L]
    diag_f = (row == col).astype(mx.float32)  # [L, L]
    final = base * (1.0 - diag_f[None, :, :])  # [B, L, L]
    return final[:, None, :, :]  # [B, 1, L, L]（heads 次元へブロードキャスト）


def _positions(full_ids: mx.array) -> mx.array:
    """位置 id を `CumSum(full_ids > 0) - 1`（0 未満は 0 へ clamp）で求める
    （クラス docstring 3 番 (b)）。実トークンは詰め物の位置に関係なく連続した
    位置を受け取る。
    """
    valid = (full_ids > 0).astype(mx.int32)
    pos = mx.cumsum(valid, axis=1) - 1
    return mx.maximum(pos, 0)


class DecoderLayer(nn.Module):
    """pre-LN の decoder 層（causal self-attention のみ。cross-attention なし）。

    `nn.MultiHeadAttention` を使わず、q/k/v/o の `nn.Linear` と明示的な
    attention 計算で実装する（モジュール docstring 参照。ONNX への写像を
    1 対 1 にするため）。bias 有り・スケール `1/sqrt(head_dim)` は ONNX 側
    （`_export_ar_onnx`）と揃える。
    """

    def __init__(self, dims: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dims // heads
        self.ln1 = nn.LayerNorm(dims)
        self.wq = nn.Linear(dims, dims)
        self.wk = nn.Linear(dims, dims)
        self.wv = nn.Linear(dims, dims)
        self.wo = nn.Linear(dims, dims)
        self.ln2 = nn.LayerNorm(dims)
        self.ff1 = nn.Linear(dims, dims * 4)
        self.ff2 = nn.Linear(dims * 4, dims)
        self.drop = nn.Dropout(dropout)

    def _attn(self, h: mx.array, mask: mx.array) -> mx.array:
        b, length, dims = h.shape
        q = self.wq(h).reshape(b, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(h).reshape(b, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(h).reshape(b, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale + mask
        attn = mx.softmax(scores, axis=-1)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, length, dims)
        return self.wo(ctx)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        h = self.ln1(x)
        x = x + self.drop(self._attn(h, mask))
        h = self.ln2(x)
        return x + self.drop(self.ff2(nn.gelu(self.ff1(h))))


class ByteDecoder(nn.Module):
    """バイト単位の小型自己回帰 decoder（PoC-24 `ByteDecoder` の移植。マスク付き
    causal self-attention。クラス docstring 3 番の詰め物耐性を持つ）。
    """

    def __init__(self, layers: int, dims: int, heads: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, dims)
        self.pos = nn.Embedding(max_len, dims)
        self.layers = [DecoderLayer(dims, heads, dropout) for _ in range(layers)]
        self.ln = nn.LayerNorm(dims)
        self.out = nn.Linear(dims, VOCAB_SIZE)

    def __call__(self, full_ids: mx.array) -> mx.array:
        mask = _build_additive_mask(full_ids)
        pos_idx = _positions(full_ids)
        h = self.embed(full_ids) + self.pos(pos_idx)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.ln(h)
        return self.out(h)


def _build_choice_block(choice_id_list: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """選択肢の一覧を固定幅 `[K, M]` の `choice_tokens`（各行 = バイト+1 → EOS →
    PAD）と、その有効性マスク `[K, M]`（`choice_tokens != PAD`）へ変換する
    （§2.3。`M = max_label_len + 1`）。
    """
    k = len(choice_id_list)
    m = max((len(c) for c in choice_id_list), default=0) + 1
    tokens = np.zeros((k, m), dtype=np.int64)
    for i, c in enumerate(choice_id_list):
        tokens[i, : len(c)] = c
        tokens[i, len(c)] = EOS
    valid = (tokens != PAD).astype(np.float64)
    return tokens, valid


def _score_choices_mlx(
    model: ByteDecoder, ids_batch: list[list[int]], choice_id_list: list[list[int]]
) -> np.ndarray:
    """対応づけ (b) の MLX 実装（モジュール docstring 2 番）。

    `ids_batch`（N 件。各要素は `encode_bytes` が返す、詰め物を含まない
    可変長のトークン列）と `choice_id_list`（K 件の選択肢。`_encode_choices`
    が返す、EOS・PAD を含まない可変長のトークン列）から、`[N, K]` の
    条件付き対数尤度合計（choice+EOS の対数尤度の合計。長さ正規化なし）を返す。

    `#80`（TASK-19b.2）はこの関数を再利用して 1 件ずつの予測レコードを
    組み立てる想定（モジュール docstring）。本関数自体は正規化前の対数尤度を
    返すだけで、predicted_label・scores への変換は呼び出し側の責務とする
    （`test_ar_train.py` の golden テスト・簡易正解率確認が呼び出し元）。
    """
    model.eval()
    n = len(ids_batch)
    choice_tokens, valid_mask = _build_choice_block(choice_id_list)
    k, m = choice_tokens.shape
    t = max(1, max(len(x) for x in ids_batch))
    length = t + 1 + m

    full = np.zeros((n * k, length), dtype=np.int32)
    for row_n, ids in enumerate(ids_batch):
        base = row_n * k
        full[base : base + k, : len(ids)] = ids
        full[base : base + k, t] = SEP
        full[base : base + k, t + 1 : t + 1 + m] = choice_tokens

    logits = model(mx.array(full))
    logp = nn.log_softmax(logits, axis=-1)
    choice_logp = logp[:, t : t + m, :]  # [N*K, M, V]
    target = np.tile(choice_tokens, (n, 1))  # 行順は n*K+k（full の構築と同じ順）
    target_mx = mx.array(target)[..., None]
    gathered = mx.take_along_axis(choice_logp, target_mx, axis=-1)[..., 0]  # [N*K, M]
    valid_tile = mx.array(np.tile(valid_mask, (n, 1)).astype(np.float32))
    summed = (gathered * valid_tile).sum(axis=1)  # [N*K]
    mx.eval(summed)
    return np.array(summed, dtype=np.float64).reshape(n, k)


def _encode_input_examples(
    examples: list[TrainExample], max_bytes: int, resource_budget: budget_mod.ResourceBudget
) -> tuple[np.ndarray, np.ndarray]:
    """学習データ全体を、あらかじめ確保した numpy int32 配列へ行ごとにエンコードする
    （`kinds/c3.py::_encode_examples` と同じ考え方。P0-1）。

    戻り値は `(ids, lengths)`。`ids` は詰め物列を実際の最大長まで切り詰めた
    2 次元配列（各行の実長は `lengths[i]`。`lengths[i]` 列より後ろは 0 詰め）。
    """
    n = len(examples)
    arr = np.zeros((n, max_bytes), dtype=np.int32)
    lengths = np.zeros((n,), dtype=np.int32)
    max_len = 1
    for i, ex in enumerate(examples):
        row = encode_bytes(ex.input, max_bytes)
        length = len(row)
        arr[i, :length] = row
        lengths[i] = length
        if length > max_len:
            max_len = length
        if (i + 1) % 1024 == 0:
            resource_budget.check()
    resource_budget.check()
    return arr[:, :max_len], lengths


def _build_train_batch(
    ids_arr: np.ndarray,
    lengths: np.ndarray,
    idx: np.ndarray,
    labels: list[str],
    choice_ids_by_label: dict[str, list[int]],
    m: int,
) -> tuple[mx.array, mx.array]:
    """学習バッチ 1 つ分の `full_ids`（`[B, T_batch+1+M]`）と損失マスク
    （`[B, T_batch+M]`。`full_ids[:, 1:]` を教師とする shift 済みの座標系）を
    組み立てる。

    入力領域の幅 `T_batch` はこのバッチ内の実長の最大値（バッチ間で可変）、
    選択肢領域の幅 `M` は学習全体で固定（`AutoregressiveKind.train` が
    `label_order` 全体から求めた `max_label_len + 1`。推論時の固定選択肢
    ブロック幅と同じ値を使うことで、学習・推論で「位置 T_batch..T_batch+M-1 が
    選択肢領域」という座標系を揃える）。
    """
    t_batch = int(lengths[idx].max())
    length = t_batch + 1 + m
    full = np.zeros((len(idx), length), dtype=np.int32)
    mask = np.zeros((len(idx), length - 1), dtype=np.float32)
    for row_i, ex_i in enumerate(idx):
        real_len = int(lengths[ex_i])
        full[row_i, :real_len] = ids_arr[ex_i, :real_len]
        full[row_i, t_batch] = SEP
        choice = choice_ids_by_label[labels[ex_i]]
        full[row_i, t_batch + 1 : t_batch + 1 + len(choice)] = choice
        full[row_i, t_batch + 1 + len(choice)] = EOS
        mask[row_i, t_batch : t_batch + len(choice) + 1] = 1.0
    return mx.array(full), mx.array(mask)


@dataclass(frozen=True)
class AutoregressiveTrainedModel:
    """autoregressive の学習結果（ONNX 書き出しに要る情報一式）。

    `config`・`label_order`・`max_bytes` は `kinds/__init__.py::TrainedModel`
    プロトコルが宣言する共通フィールド。`max_label_len`・`choice_ids_by_label`
    は ONNX 書き出し（選択肢ブロックの構築）に必要な、この種類固有の情報。
    """

    model: ByteDecoder
    config: dict[str, Any] = field(default_factory=dict)
    label_order: list[str] = field(default_factory=list)
    max_bytes: int = 512
    max_label_len: int = 0
    choice_ids_by_label: dict[str, list[int]] = field(default_factory=dict)
    seed: int = 0
    epochs_run: int = 0
    resource_budget: budget_mod.ResourceBudget | None = None


class AutoregressiveKind:
    """選択口（`kinds/__init__.py`）が呼ぶ autoregressive の学習・書き出し実装。"""

    def train(
        self,
        examples: list[TrainExample],
        request: TrainRequest,
        resource_budget: budget_mod.ResourceBudget,
    ) -> AutoregressiveTrainedModel:
        cfg = {**DEFAULT_CONFIG, **request.config}
        _validate_config(cfg)

        label_order = list(request.label_order)
        choice_ids_by_label = _encode_choices(label_order)
        max_label_len = max(len(c) for c in choice_ids_by_label.values())
        m = max_label_len + 1  # 選択肢領域の幅（choice バイト列 + EOS）

        layers = int(cfg["layers"])
        dims = int(cfg["dims"])
        heads = int(cfg["heads"])
        epochs = int(cfg["epochs"])
        batch_size = int(cfg["batch_size"])
        max_len = request.max_bytes + 1 + m

        # 学習開始前の見積もりベースの資源上限検査（モデルを構築する前・
        # 1 バッチも回す前に拒否できる。`kinds/c3.py::C3Kind.train` と同じ順序）。
        budget_mod.check_model_bytes(_ar_param_count(layers, dims, heads, max_len))
        budget_mod.check_sample_steps(len(examples), epochs)
        budget_mod.check_total_tokens(len(examples), request.max_bytes)
        attn_elements = batch_size * heads * max_len * max_len
        if attn_elements > MAX_AR_ATTENTION_ELEMENTS:
            raise WorkerError(
                "limit_exceeded",
                f"estimated attention elements {attn_elements} exceeds limit"
                f" {MAX_AR_ATTENTION_ELEMENTS} (batch_size x heads x max_len^2)",
                ExitCode.LIMIT_EXCEEDED,
            )

        mx.set_default_device(mx.cpu if request.device == "cpu" else mx.gpu)
        mx.random.seed(request.seed)
        rng = np.random.default_rng(request.seed)

        model = ByteDecoder(layers, dims, heads, cfg["dropout"], max_len)
        mx.eval(model.parameters())

        warmup_steps = int(cfg["warmup_steps"])
        lr = float(cfg["lr"])
        sched = optim.join_schedules(
            [optim.linear_schedule(0.0, lr, warmup_steps), optim.linear_schedule(lr, lr, 1)],
            [warmup_steps],
        )
        opt = optim.AdamW(learning_rate=sched, weight_decay=cfg["weight_decay"])

        ids_arr, lengths = _encode_input_examples(examples, request.max_bytes, resource_budget)
        labels = [ex.label for ex in examples]

        def loss_fn(mdl: ByteDecoder, full_ids: mx.array, mask: mx.array) -> mx.array:
            logits = mdl(full_ids)
            shifted_logits = logits[:, :-1, :]
            targets = full_ids[:, 1:]
            ce = nn.losses.cross_entropy(shifted_logits, targets, reduction="none")
            denom = mx.maximum(mask.sum(), 1.0)
            return (ce * mask).sum() / denom

        step = nn.value_and_grad(model, loss_fn)
        model.train()
        for _epoch in range(epochs):
            order = rng.permutation(len(examples))
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                full_ids, mask = _build_train_batch(
                    ids_arr, lengths, idx, labels, choice_ids_by_label, m
                )
                loss, grads = step(model, full_ids, mask)
                opt.update(model, grads)
                mx.eval(model.parameters(), opt.state, loss)
                if not _is_finite_loss(float(loss.item())):
                    # `kinds/c3.py::C3Kind.train` と同じ対応づけ（PENDING=12。
                    # 「唯一の候補が発散した」状態を、複数候補からの再選定
                    # 〔TASK-18.x〕を見越した意味づけのまま踏襲する）。
                    raise WorkerError(
                        "training_diverged",
                        "training loss became non-finite",
                        ExitCode.PENDING,
                    )
                resource_budget.check()
        model.eval()

        return AutoregressiveTrainedModel(
            model=model,
            config=cfg,
            label_order=label_order,
            max_bytes=request.max_bytes,
            max_label_len=max_label_len,
            choice_ids_by_label=choice_ids_by_label,
            seed=request.seed,
            epochs_run=epochs,
            resource_budget=resource_budget,
        )

    def export_onnx(self, trained: AutoregressiveTrainedModel, out: IO[bytes]) -> None:
        if trained.resource_budget is not None:
            trained.resource_budget.check()
        _export_ar_onnx(trained, out)


def _f32(arr: np.ndarray, name: str) -> TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name=name)


def _i64(arr: np.ndarray, name: str) -> TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.int64), name=name)


def _onnx_layernorm(
    nodes: list, initializers: list, x: str, weight: np.ndarray, bias: np.ndarray, prefix: str
) -> str:
    """`mx.fast.layer_norm` 相当（`y = (x-E[x])/sqrt(Var[x]+eps) * weight + bias`。
    population variance・`eps=1e-5`）を基本演算で組む（opset 13 に
    `LayerNormalization` が無いため。`trainer/.venv` の `mlx/nn/layers/
    normalization.py::LayerNorm.__call__` で eps・分散の式を確認済み。
    モジュール docstring 参照）。`ReduceMean` は opset 13 でも `axes` が
    attribute のまま（`ReduceSum` とは異なり opset 18 まで attribute）。
    """
    eps_name = f"{prefix}_eps"
    initializers.append(_f32(np.array(1e-5), eps_name))
    w_name, b_name = f"{prefix}_w", f"{prefix}_b"
    initializers.append(_f32(weight, w_name))
    initializers.append(_f32(bias, b_name))

    mean = f"{prefix}_mean"
    nodes.append(helper.make_node("ReduceMean", [x], [mean], axes=[-1], keepdims=1))
    centered = f"{prefix}_centered"
    nodes.append(helper.make_node("Sub", [x, mean], [centered]))
    sq = f"{prefix}_sq"
    nodes.append(helper.make_node("Mul", [centered, centered], [sq]))
    var = f"{prefix}_var"
    nodes.append(helper.make_node("ReduceMean", [sq], [var], axes=[-1], keepdims=1))
    var_eps = f"{prefix}_var_eps"
    nodes.append(helper.make_node("Add", [var, eps_name], [var_eps]))
    std = f"{prefix}_std"
    nodes.append(helper.make_node("Sqrt", [var_eps], [std]))
    normed = f"{prefix}_normed"
    nodes.append(helper.make_node("Div", [centered, std], [normed]))
    scaled = f"{prefix}_scaled"
    nodes.append(helper.make_node("Mul", [normed, w_name], [scaled]))
    out_name = f"{prefix}_out"
    nodes.append(helper.make_node("Add", [scaled, b_name], [out_name]))
    return out_name


def _onnx_linear(
    nodes: list, initializers: list, x: str, weight: np.ndarray, bias: np.ndarray, prefix: str
) -> str:
    """`nn.Linear`（重み形状 `(output_dims, input_dims)`。`trainer/.venv` の
    `mlx/nn/layers/linear.py::Linear` で確認済み）を `MatMul` + `Add` で組む
    （入力が 3 次元 `[B,L,D_in]` のため `Gemm` は使わない。モジュール
    docstring 参照）。
    """
    wt_name = f"{prefix}_wT"
    initializers.append(_f32(weight.T, wt_name))
    b_name = f"{prefix}_b"
    initializers.append(_f32(bias, b_name))
    mm = f"{prefix}_mm"
    nodes.append(helper.make_node("MatMul", [x, wt_name], [mm]))
    out_name = f"{prefix}_out"
    nodes.append(helper.make_node("Add", [mm, b_name], [out_name]))
    return out_name


def _onnx_gelu(nodes: list, initializers: list, x: str, prefix: str) -> str:
    """`nn.gelu` の厳密形 `x*(1+erf(x/sqrt(2)))/2`（`trainer/.venv` の
    `mlx/nn/layers/activations.py::gelu` で確認済み。近似形〔tanh・sigmoid〕
    ではない）を `Erf` で組む（opset 20 の `Gelu` を使わない）。
    """
    inv_sqrt2_name = f"{prefix}_inv_sqrt2"
    initializers.append(_f32(np.array(1.0 / math.sqrt(2.0)), inv_sqrt2_name))
    one_name = f"{prefix}_one"
    initializers.append(_f32(np.array(1.0), one_name))
    half_name = f"{prefix}_half"
    initializers.append(_f32(np.array(0.5), half_name))

    scaled = f"{prefix}_scaled"
    nodes.append(helper.make_node("Mul", [x, inv_sqrt2_name], [scaled]))
    erf = f"{prefix}_erf"
    nodes.append(helper.make_node("Erf", [scaled], [erf]))
    one_plus_erf = f"{prefix}_one_plus_erf"
    nodes.append(helper.make_node("Add", [erf, one_name], [one_plus_erf]))
    x_times = f"{prefix}_x_times"
    nodes.append(helper.make_node("Mul", [x, one_plus_erf], [x_times]))
    out_name = f"{prefix}_out"
    nodes.append(helper.make_node("Mul", [x_times, half_name], [out_name]))
    return out_name


def _onnx_decoder_layer(
    nodes: list,
    initializers: list,
    x: str,
    mask4d: str,
    heads: int,
    head_dim: int,
    params: dict,
    prefix: str,
) -> str:
    """1 つの `DecoderLayer`（pre-LN・causal self-attention・GELU FF）を組む。

    MLX 側 `DecoderLayer.__call__`/`_attn` と同じ式（LayerNorm → q/k/v/o
    Linear → reshape してヘッド分割 → scaled dot-product attention（マスク
    加算）→ 出力 Linear → 残差加算 → LayerNorm → FF1 → GELU → FF2 →
    残差加算）を、学習は行わない（dropout は恒等写像として省略。ONNX グラフは
    推論専用）。
    """
    ln1_w = np.array(params["ln1"]["weight"], dtype=np.float32)
    ln1_b = np.array(params["ln1"]["bias"], dtype=np.float32)
    h1 = _onnx_layernorm(nodes, initializers, x, ln1_w, ln1_b, f"{prefix}_ln1")

    def _proj(name: str, key: str) -> str:
        w = np.array(params[key]["weight"], dtype=np.float32)
        b = np.array(params[key]["bias"], dtype=np.float32)
        return _onnx_linear(nodes, initializers, h1, w, b, f"{prefix}_{name}")

    q = _proj("q", "wq")
    k = _proj("k", "wk")
    v = _proj("v", "wv")

    # [N*K, L, dims] -> [N*K, L, heads, head_dim] -> [N*K, heads, L, head_dim]
    shape_name = f"{prefix}_head_shape"
    initializers.append(_i64(np.array([0, 0, heads, head_dim]), shape_name))
    perm = [0, 2, 1, 3]

    def _split_heads(name: str, src: str) -> str:
        reshaped = f"{prefix}_{name}_split"
        nodes.append(helper.make_node("Reshape", [src, shape_name], [reshaped]))
        transposed = f"{prefix}_{name}_heads"
        nodes.append(helper.make_node("Transpose", [reshaped], [transposed], perm=perm))
        return transposed

    q_h = _split_heads("q", q)
    k_h = _split_heads("k", k)
    v_h = _split_heads("v", v)

    k_t = f"{prefix}_k_t"
    nodes.append(helper.make_node("Transpose", [k_h], [k_t], perm=[0, 1, 3, 2]))
    scores_raw = f"{prefix}_scores_raw"
    nodes.append(helper.make_node("MatMul", [q_h, k_t], [scores_raw]))
    scale_name = f"{prefix}_scale"
    initializers.append(_f32(np.array(1.0 / math.sqrt(head_dim)), scale_name))
    scores_scaled = f"{prefix}_scores_scaled"
    nodes.append(helper.make_node("Mul", [scores_raw, scale_name], [scores_scaled]))
    scores_masked = f"{prefix}_scores_masked"
    nodes.append(helper.make_node("Add", [scores_scaled, mask4d], [scores_masked]))
    attn = f"{prefix}_attn"
    nodes.append(helper.make_node("Softmax", [scores_masked], [attn], axis=-1))
    ctx_heads = f"{prefix}_ctx_heads"
    nodes.append(helper.make_node("MatMul", [attn, v_h], [ctx_heads]))
    ctx_t = f"{prefix}_ctx_t"
    nodes.append(helper.make_node("Transpose", [ctx_heads], [ctx_t], perm=[0, 2, 1, 3]))
    merge_shape = f"{prefix}_merge_shape"
    initializers.append(_i64(np.array([0, 0, heads * head_dim]), merge_shape))
    ctx = f"{prefix}_ctx"
    nodes.append(helper.make_node("Reshape", [ctx_t, merge_shape], [ctx]))

    wo_w = np.array(params["wo"]["weight"], dtype=np.float32)
    wo_b = np.array(params["wo"]["bias"], dtype=np.float32)
    attn_out = _onnx_linear(nodes, initializers, ctx, wo_w, wo_b, f"{prefix}_wo")

    resid1 = f"{prefix}_resid1"
    nodes.append(helper.make_node("Add", [x, attn_out], [resid1]))

    ln2_w = np.array(params["ln2"]["weight"], dtype=np.float32)
    ln2_b = np.array(params["ln2"]["bias"], dtype=np.float32)
    h2 = _onnx_layernorm(nodes, initializers, resid1, ln2_w, ln2_b, f"{prefix}_ln2")

    ff1_w = np.array(params["ff1"]["weight"], dtype=np.float32)
    ff1_b = np.array(params["ff1"]["bias"], dtype=np.float32)
    ff1_out = _onnx_linear(nodes, initializers, h2, ff1_w, ff1_b, f"{prefix}_ff1")
    gelu_out = _onnx_gelu(nodes, initializers, ff1_out, f"{prefix}_gelu")
    ff2_w = np.array(params["ff2"]["weight"], dtype=np.float32)
    ff2_b = np.array(params["ff2"]["bias"], dtype=np.float32)
    ff2_out = _onnx_linear(nodes, initializers, gelu_out, ff2_w, ff2_b, f"{prefix}_ff2")

    resid2 = f"{prefix}_resid2"
    nodes.append(helper.make_node("Add", [resid1, ff2_out], [resid2]))
    return resid2


def _ar_export_attention_elements(
    k_classes: int, t_bound: int, m: int, layers: int, heads: int
) -> int:
    """書き出す ONNX グラフが `N=1`・`T=t_bound`（書き出し時点で構造上
    許容される最大の入力長。`limits.py::MAX_AR_EXPORT_ATTENTION_ELEMENTS`
    docstring 参照）で推論されたときの、decoder attention の要素数の
    見積もり（`K × heads × layers × L^2`。`L = T + 1 + M`）を返す
    （REQ-39・PR #222 レビュー指摘）。

    decoder は選択肢展開後の `[N*K, L, L]` 形状で attention を計算するため
    `K` を乗じ、各層で同形状のテンソルを確保しうるため `layers` を乗じる
    （層間でメモリが解放される実装でも、上限側は保守的に見積もる）。
    引数はすべて 0 以上の整数であること（負数・非整数は呼び出し側の
    バグであり、ここでは検査しない。すべて学習時に確定する固定値のみを
    渡す契約のため）。
    """
    length = t_bound + 1 + m
    return k_classes * heads * layers * length * length


def _check_ar_export_resources(
    k_classes: int, t_bound: int, m: int, layers: int, heads: int
) -> None:
    """`_ar_export_attention_elements` の見積もりが
    `MAX_AR_EXPORT_ATTENTION_ELEMENTS` を超えるとき `limit_exceeded` で
    fail-closed に拒否する（REQ-39・PR #222 レビュー指摘）。グラフ構築前に
    呼ぶことで、過大な attention テンソルを実際に確保する前に停止する。
    """
    elements = _ar_export_attention_elements(k_classes, t_bound, m, layers, heads)
    if elements > MAX_AR_EXPORT_ATTENTION_ELEMENTS:
        raise WorkerError(
            "limit_exceeded",
            f"estimated per-example attention elements {elements} (at N=1, T={t_bound})"
            f" exceeds limit {MAX_AR_EXPORT_ATTENTION_ELEMENTS}"
            " (k_classes x heads x layers x (t_bound+1+m)^2)",
            ExitCode.LIMIT_EXCEEDED,
        )


def _export_ar_onnx(trained: AutoregressiveTrainedModel, out: IO[bytes]) -> None:
    """§2.1〜2.3 のグラフを手組みし、`out` へ ONNX protobuf を書き出す
    （`kinds/c3.py::_export_c3_onnx` と同じ「経路は一切扱わない」契約。
    `kinds/__init__.py::Kind.export_onnx` docstring 参照）。

    入出力契約は C1・C3 と同じ（`ids: INT64 ["N","T"]` → `probs: FLOAT
    ["N", K]`）。opset 13・`ir_version=8`。opset 13 の `Unsqueeze`/`Squeeze`/
    `ReduceSum` は `axes` が第 2 入力（属性ではない）である点に注意（opset 18 で
    属性へ統一される前の過渡期の仕様。`ReduceMean`/`ReduceMax` は opset 13 でも
    引き続き `axes` 属性のまま。`onnx.checker.check_model` で機械的に検証する）。
    書き出しの各段（テンソル抽出・展開用テンソルの構築・グラフ構築・
    `check_model` 後・保存後）で資源上限（REQ-39）を検査する。
    """
    budget_check = trained.resource_budget.check if trained.resource_budget is not None else None

    cfg = trained.config
    dims = int(cfg["dims"])
    heads = int(cfg["heads"])
    head_dim = dims // heads
    layers = int(cfg["layers"])
    label_order = trained.label_order
    k_classes = len(label_order)
    choice_id_list = [trained.choice_ids_by_label[label] for label in label_order]
    choice_tokens, choice_valid = _build_choice_block(choice_id_list)
    m = choice_tokens.shape[1]

    # 推論 1 件（N=1）あたりの選択肢対数確率抽出テンソル（`choice_logp`・
    # `choice_onehot`・`masked_vocab`。いずれも [K,M,VOCAB_SIZE]）の要素数を
    # グラフ構築前に検査する（`limits.py::MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS`
    # docstring 参照。セキュリティ監査 P0 指摘。PR #222）。K（選択肢数）・
    # M（選択肢の最大バイト長+1）は学習時に固定される値のため、ここで
    # 拒否すれば N=1 でも過大な書き出しを fail-closed にできる。バッチ件数
    # N 分の上限は推論ランタイム・ガード層側の責務であり、本チェックでは
    # 検査できない（後述の out-of-scope 記録参照）。
    choice_logprob_elements = k_classes * m * VOCAB_SIZE
    if choice_logprob_elements > MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS:
        raise WorkerError(
            "limit_exceeded",
            f"estimated per-example choice log-prob elements {choice_logprob_elements}"
            f" exceeds limit {MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS} (k_classes x m x"
            " vocab_size)",
            ExitCode.LIMIT_EXCEEDED,
        )

    # 推論時に渡されうる T（動的軸）を、位置埋め込み表 `pos_table` の行数
    # （学習時に固定された `max_len = max_bytes + 1 + m`）から構造上の
    # 上限へ落とし込み、N=1・T=t_bound（書き出し可能な最大構成）での
    # decoder attention 要素数を見積もって検査する
    # （`limits.py::MAX_AR_EXPORT_ATTENTION_ELEMENTS` docstring・
    # `_ar_export_attention_elements` docstring 参照。REQ-39・PR #222
    # レビュー指摘）。バッチ件数 N（>1）分の上限は本検査では検査できない
    # （同 docstring 参照。推論ランタイム・ガード層側の将来対応）。
    t_bound = trained.max_bytes
    _check_ar_export_resources(k_classes, t_bound, m, layers, heads)

    params = trained.model.parameters()
    embed = np.array(params["embed"]["weight"], dtype=np.float32)  # [VOCAB, dims]
    pos_table = np.array(params["pos"]["weight"], dtype=np.float32)  # [max_len, dims]
    ln_w = np.array(params["ln"]["weight"], dtype=np.float32)
    ln_b = np.array(params["ln"]["bias"], dtype=np.float32)
    out_w = np.array(params["out"]["weight"], dtype=np.float32)  # [VOCAB, dims]
    out_b = np.array(params["out"]["bias"], dtype=np.float32)
    if budget_check is not None:
        budget_check()  # 段 1: 重みの抽出後

    ids = helper.make_tensor_value_info("ids", TensorProto.INT64, ["N", "T"])
    probs_out = helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["N", k_classes])

    initializers = [
        _i64(choice_tokens, "choice_tokens"),  # [K, M]
        _f32(choice_valid, "choice_valid"),  # [K, M]
        numpy_helper.from_array(np.array([[[SEP]]], dtype=np.int64), name="sep_block"),  # [1,1,1]
        numpy_helper.from_array(np.array(0, dtype=np.int64), name="zero_scalar"),  # 0-d
        numpy_helper.from_array(np.array(1, dtype=np.int64), name="one_scalar"),  # 0-d
        _i64(np.array([0]), "idx0"),
        _i64(np.array([1]), "idx1"),
        _i64(np.array([2]), "idx2"),
        _i64(np.array([-1]), "negidx"),
        _i64(np.array([1]), "one_vec"),
        _i64(np.array([k_classes]), "k_vec"),
        _i64(np.array([m]), "m_vec"),
        _f32(np.array(1.0), "one_f32"),
        _f32(np.array(_MASK_NEG_VALUE), "neg_big_f32"),
        numpy_helper.from_array(np.array(VOCAB_SIZE, dtype=np.int64), name="vocab_depth"),  # 0-d
        _f32(np.array([0.0, 1.0]), "onehot_values"),  # [2]（off_value, on_value）
        _f32(embed, "embed_table"),
        _f32(pos_table, "pos_table"),
    ]
    nodes: list = []

    # --- N・T の取得（動的軸） ---
    nodes.append(helper.make_node("Shape", ["ids"], ["shape_ids"]))
    nodes.append(helper.make_node("Gather", ["shape_ids", "idx0"], ["n_vec"], axis=0))
    nodes.append(helper.make_node("Gather", ["shape_ids", "idx1"], ["t_vec"], axis=0))

    # --- 派生する形状ベクトル（いずれも 1 要素以上の 1 次元 int64 配列） ---
    nodes.append(helper.make_node("Add", ["t_vec", "m_vec"], ["t_plus_m_vec"]))  # T+M
    nodes.append(helper.make_node("Add", ["t_vec", "one_vec"], ["t_plus_1_vec"]))  # T+1
    nodes.append(helper.make_node("Add", ["t_plus_1_vec", "m_vec"], ["l_vec"]))  # L=T+1+M
    nodes.append(helper.make_node("Squeeze", ["l_vec", "idx0"], ["l_scalar"]))
    nodes.append(helper.make_node("Concat", ["n_vec", "k_vec", "t_vec"], ["shape_nkt"], axis=0))
    nodes.append(helper.make_node("Concat", ["n_vec", "k_vec", "m_vec"], ["shape_nkm"], axis=0))
    nodes.append(helper.make_node("Concat", ["n_vec", "k_vec", "one_vec"], ["shape_nk1"], axis=0))
    nodes.append(helper.make_node("Concat", ["negidx", "l_vec"], ["shape_flat_l"], axis=0))
    nodes.append(helper.make_node("Concat", ["negidx", "m_vec"], ["shape_flat_m"], axis=0))
    nodes.append(helper.make_node("Concat", ["n_vec", "k_vec"], ["shape_nk"], axis=0))

    # --- K 個の選択肢へ展開してから [N*K, L] へ reshape する（§2.3） ---
    nodes.append(helper.make_node("Unsqueeze", ["ids", "idx1"], ["ids_exp3"]))  # [N,1,T]
    nodes.append(helper.make_node("Expand", ["ids_exp3", "shape_nkt"], ["ids_exp"]))  # [N,K,T]

    nodes.append(helper.make_node("Unsqueeze", ["choice_tokens", "idx0"], ["choice_exp3"]))
    nodes.append(helper.make_node("Expand", ["choice_exp3", "shape_nkm"], ["choice_exp"]))
    nodes.append(helper.make_node("Unsqueeze", ["choice_valid", "idx0"], ["choice_valid_exp3"]))
    nodes.append(
        helper.make_node("Expand", ["choice_valid_exp3", "shape_nkm"], ["choice_valid_exp"])
    )
    nodes.append(helper.make_node("Expand", ["sep_block", "shape_nk1"], ["sep_exp"]))  # [N,K,1]

    nodes.append(
        helper.make_node("Concat", ["ids_exp", "sep_exp", "choice_exp"], ["full_ids_3d"], axis=2)
    )
    nodes.append(helper.make_node("Reshape", ["full_ids_3d", "shape_flat_l"], ["full_ids_2d"]))
    nodes.append(helper.make_node("Reshape", ["choice_exp", "shape_flat_m"], ["choice_flat"]))
    nodes.append(
        helper.make_node("Reshape", ["choice_valid_exp", "shape_flat_m"], ["choice_valid_flat"])
    )
    if budget_check is not None:
        budget_check()  # 段 2: 展開・reshape 用テンソルの構築後

    # --- key の詰め物マスク・causal マスク（クラス docstring 3 番 (a)） ---
    nodes.append(helper.make_node("Greater", ["full_ids_2d", "zero_scalar"], ["valid_key_bool"]))
    nodes.append(
        helper.make_node("Cast", ["valid_key_bool"], ["valid_key_f"], to=TensorProto.FLOAT)
    )
    nodes.append(helper.make_node("Sub", ["one_f32", "valid_key_f"], ["inv_valid_key"]))
    nodes.append(helper.make_node("Mul", ["inv_valid_key", "neg_big_f32"], ["pad_pen"]))  # [N*K,L]

    nodes.append(
        helper.make_node("Range", ["zero_scalar", "l_scalar", "one_scalar"], ["range_l"])
    )  # [L] int64
    nodes.append(helper.make_node("Unsqueeze", ["range_l", "idx1"], ["row_idx"]))  # [L,1]
    nodes.append(helper.make_node("Unsqueeze", ["range_l", "idx0"], ["col_idx"]))  # [1,L]
    nodes.append(helper.make_node("Greater", ["col_idx", "row_idx"], ["causal_bool"]))  # [L,L]
    nodes.append(helper.make_node("Cast", ["causal_bool"], ["causal_bool_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", ["causal_bool_f", "neg_big_f32"], ["causal_f"]))
    nodes.append(helper.make_node("Equal", ["row_idx", "col_idx"], ["diag_bool"]))  # [L,L]
    nodes.append(helper.make_node("Cast", ["diag_bool"], ["diag_f"], to=TensorProto.FLOAT))

    nodes.append(helper.make_node("Unsqueeze", ["causal_f", "idx0"], ["causal_f_3d"]))  # [1,L,L]
    nodes.append(helper.make_node("Unsqueeze", ["pad_pen", "idx1"], ["pad_pen_3d"]))  # [N*K,1,L]
    nodes.append(helper.make_node("Add", ["causal_f_3d", "pad_pen_3d"], ["mask_base"]))  # [N*K,L,L]
    nodes.append(helper.make_node("Unsqueeze", ["diag_f", "idx0"], ["diag_f_3d"]))  # [1,L,L]
    nodes.append(helper.make_node("Sub", ["one_f32", "diag_f_3d"], ["diag_keep"]))
    nodes.append(helper.make_node("Mul", ["mask_base", "diag_keep"], ["mask_final"]))  # [N*K,L,L]
    nodes.append(helper.make_node("Unsqueeze", ["mask_final", "idx1"], ["mask4d"]))  # [N*K,1,L,L]

    # --- 位置 id（クラス docstring 3 番 (b)） ---
    nodes.append(helper.make_node("CumSum", ["valid_key_f", "one_scalar"], ["cumsum_valid"]))
    nodes.append(helper.make_node("Sub", ["cumsum_valid", "one_f32"], ["pos_f"]))
    nodes.append(helper.make_node("Relu", ["pos_f"], ["pos_f_clamped"]))
    nodes.append(helper.make_node("Cast", ["pos_f_clamped"], ["pos_i64"], to=TensorProto.INT64))

    # --- 埋め込み ---
    nodes.append(helper.make_node("Gather", ["embed_table", "full_ids_2d"], ["token_emb"], axis=0))
    nodes.append(helper.make_node("Gather", ["pos_table", "pos_i64"], ["pos_emb"], axis=0))
    nodes.append(helper.make_node("Add", ["token_emb", "pos_emb"], ["h0"]))
    if budget_check is not None:
        budget_check()  # 段 3: マスク・位置・埋め込みの構築後

    # --- decoder 層 ---
    h = "h0"
    for layer_i in range(layers):
        layer_params = params["layers"][layer_i]
        h = _onnx_decoder_layer(
            nodes, initializers, h, "mask4d", heads, head_dim, layer_params, f"layer{layer_i}"
        )
        if budget_check is not None:
            budget_check()  # 段 4-n: 各 decoder 層のテンソル抽出後

    ln_out = _onnx_layernorm(nodes, initializers, h, ln_w, ln_b, "final_ln")
    logits = _onnx_linear(nodes, initializers, ln_out, out_w, out_b, "out_proj")
    nodes.append(helper.make_node("LogSoftmax", [logits], ["logp"], axis=-1))  # [N*K,L,VOCAB]

    # --- 選択肢領域の対数尤度を取り出して合計する（§2.3） ---
    nodes.append(
        helper.make_node(
            "Slice", ["logp", "t_vec", "t_plus_m_vec", "idx1", "one_vec"], ["choice_logp"]
        )
    )  # [N*K, M, VOCAB]
    # `GatherElements`（onnx.reference の実装は `numpy.choose` に依存し、
    # gather 対象の軸の要素数が 64 を超えると `ValueError` になる。VOCAB_SIZE=259
    # がこれを超えるため使えない）の代わりに one-hot + 積 + 総和で同じ「対象
    # トークンの対数確率だけを取り出す」計算を行う（`onnx.reference` の
    # `ReferenceEvaluator` を使うテスト〔`tests/test_ar_*.py`〕を実機なしで
    # 通すための、実装系の制約を避けた書き方。数学的には `GatherElements` と
    # 同値）。
    nodes.append(
        helper.make_node(
            "OneHot", ["choice_flat", "vocab_depth", "onehot_values"], ["choice_onehot"], axis=-1
        )
    )  # [N*K, M, VOCAB]
    nodes.append(helper.make_node("Mul", ["choice_logp", "choice_onehot"], ["masked_vocab"]))
    nodes.append(
        helper.make_node("ReduceSum", ["masked_vocab", "idx2"], ["gathered"], keepdims=0)
    )  # [N*K, M]
    nodes.append(helper.make_node("Mul", ["gathered", "choice_valid_flat"], ["masked_logp"]))
    nodes.append(
        helper.make_node("ReduceSum", ["masked_logp", "idx1"], ["summed"], keepdims=0)
    )  # [N*K]（opset13: ReduceSum の axes は第 2 入力）
    nodes.append(helper.make_node("Reshape", ["summed", "shape_nk"], ["probs_pre"]))  # [N,K]
    nodes.append(helper.make_node("Softmax", ["probs_pre"], ["probs"], axis=1))
    if budget_check is not None:
        budget_check()  # 段 5: グラフ構築後（check_model の前）

    graph = helper.make_graph(
        nodes, "autoregressive_decoder", [ids], [probs_out], initializer=initializers
    )
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model_proto.ir_version = 8
    onnx.checker.check_model(model_proto)
    if budget_check is not None:
        budget_check()  # 段 6: check_model 後（save の前）
    out.write(model_proto.SerializeToString())
    if budget_check is not None:
        budget_check()  # 段 7: save 後
