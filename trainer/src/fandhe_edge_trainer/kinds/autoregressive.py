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
6. **`ids` の T 軸（動的軸）を ONNX グラフの内側で `max_bytes`（学習時の
   最大長）へ Slice し、以降の全ての計算をこの切り詰め済みの列に対して
   行う**（`_export_ar_onnx` 冒頭の `Slice(ids, 0, max_bytes, axis=1)`。
   PR #222 セキュリティレビュー P0 指摘・REQ-39）。切り詰めは先頭
   `max_bytes` バイトを残す（`encoding.encode_bytes` の
   「先頭 max_bytes バイトで切り詰める」規則・`fixtures/preprocess/
   byte_encoding_vectors.json` の `truncate_long_ascii` ケースと一致させる。
   ONNX `Slice` は `ends` が実際の次元数を超える場合は次元数へ丸められる
   ため〔onnx.reference で実測確認済み〕、`T ≤ max_bytes` の入力には
   影響しない）。これにより、PAD を含む・含まないに関わらずグラフ内の
   `T`（`Slice` 後）は常に `max_bytes` 以下となり、位置埋め込み表
   `pos_table` への範囲外参照も、decoder attention の `L = T+1+M` の
   二乗拡大も、学習時に固定した上限を超えない
   （`_check_ar_export_resources`・`limits.py::
   MAX_AR_EXPORT_ATTENTION_ELEMENTS` 参照）。`N`（バッチ件数）の上限と
   `ids` の値域（`[0, 256]` の範囲外は SEP/EOS ID との衝突や ONNX
   `Gather` の範囲外参照になりうる）は、いずれも ONNX グラフの内側で
   fail-closed に検査する（オーナー判断 2026-09-28・PR #222 レビュー
   指摘。以前は推論ランタイム・ガード層〔REQ-39。パス未確定〕の責務と
   していたが、ガード層が未実装のため空隙が残っていた）。`ids` は
   範囲外の値を `Where` で「必ず範囲外になる値」へ書き換えてから
   埋め込みの `Gather` に渡し、`N` は書き出し時に決まる `n_max`（モデル
   構成ごとに算出。`limits.py::MAX_AR_INFER_BATCH_N` docstring 参照）を
   超えると固定長テーブルの `Gather` が範囲外参照で失敗する
   （`_export_ar_onnx` 内の該当コメント参照）。ただし、この失敗は
   `runtime_error`（exit 70）にしかならず機械可読な `invalid_input` には
   ならないため、推論ランタイム・ガード層（REQ-39）でも同じ条件を
   事前に `invalid_input` として拒否すべき点は変わらない。

opset 13 の制約（LayerNormalization は opset 17・Gelu は opset 20 から）により、
LayerNorm・GELU（厳密形。`math.erf` 相当）・multi-head attention はいずれも
基本演算（`ReduceMean`・`Sub`・`Mul`・`Sqrt`・`Div`・`Erf`・`MatMul`・`Softmax` 等）
で手組みする。MLX 側の既定値（`nn.LayerNorm` の eps=1e-5・population variance、
`nn.gelu` の厳密形 `x*(1+erf(x/sqrt(2)))/2`、`nn.Linear` の重み形状
`(output_dims, input_dims)`）は `trainer/.venv` の `mlx/nn/layers/
{normalization,activations,linear}.py` を実装時に確認済み（コード内の該当箇所に
根拠を記す）。

#79 の範囲はモデル・学習・ONNX 書き出し・選択口への登録までで、Python 側で
1 件ずつの予測レコードを組み立てて評価器へ渡す処理は #80（TASK-19b.1-2）が
担う。判定不能を別の status として区別する扱い（#81・TASK-19b.2）は含まない。
`_score_choices_mlx`（対応づけ (b) の MLX 実装。一致試験・訓練後の簡易正解率
確認に使う）は #80 がそのまま再利用する。

## #80（TASK-19b.1-2）が追加する対応づけ・予測レコード組み立て

`_score_choices_mlx` が返す `[N, K]` の生の対数尤度を、1 件ずつの予測レコード
（評価器〔`crates/eval`〕・データ契約〔`crates/data::eval_input`〕が読める
`{id, status, predicted_label, scores}`。`crates/core/src/judgment.rs` の
正常系スキーマと同じ）へ変換する（`choice_posteriors` → `resolve_choice_id` →
`map_scores_to_choice` → `build_prediction_record`。`predict_records` が
この一連をチャンク処理でまとめる）。Issue #80 は当初「完全一致・前方一致等の
規則」を挙げていたが、PoC-24 が事前登録した対応づけは (b)（本ファイルが実装
する softmax ベースの方式）のみで、前方一致は PoC-24 のどの記録にも無いため
採用しない（`docs/spec/03-poc/model-kind-selector/preregistration.md` 3 節）。
「完全一致」は `resolve_choice_id`（選択された選択肢のトークン列を UTF-8
バイト列として `label_order` の各要素のバイト列と完全一致させる）が (b) の
最終段として担う。PoC-24 の `reason_code`（対応づけ不能の理由を JSON に含め
る設計）は意図的に落とす（`crates/core::JudgmentStatus` が現状 `Ok` のみの
ため、Rust 側スキーマに無いフィールドを Python 側で増やさない。
coding-python.md）。対応づけ不能（`Unmapped`）は現状 `status:"error"` として
評価の分母に含まれる。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
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
    MAX_AR_INFER_BATCH_N,
    MAX_AR_LAYERS,
    MAX_AR_LR,
    MAX_AR_PREDICT_ROWS,
    MAX_AR_WARMUP_STEPS,
    MAX_AR_WEIGHT_DECAY,
    MAX_TRAIN_LINE_BYTES,
)

KIND = "autoregressive"
KIND_VERSION = 1

#: 特殊トークン（PoC-24 と同じ値）。`encoding.encode_bytes` が使う「バイト値+1」
#: の語彙（1..256）に、選択肢との区切り（SEP）・終端（EOS）を追加する。
#: PAD=0 は `encoding.encode_bytes` の詰め物規約と共通（空入力は `[0]` になる。
#: クラス docstring 3 番参照）。
PAD, SEP, EOS = 0, 257, 258
VOCAB_SIZE = 259

#: 推論入力 `ids`（外部入力。ONNX グラフの `ids` テンソル）として許可する
#: 値の上限（`encoding.encode_bytes` が返す「バイト値+1」語彙 1..256 と
#: PAD=0 を合わせた `[0, MAX_INPUT_ID]`）。SEP（257）・EOS（258）はグラフが
#: 選択肢展開の際に内部で挿入する特殊トークンであり、外部入力として
#: 渡されてよい値ではない（`_export_ar_onnx` の `ids` 値域ガード参照。
#: REQ-39・PR #222 レビュー指摘）。
MAX_INPUT_ID = SEP - 1

#: 詰め物位置をマスクする際に加える負の大きな値（`kinds/c3.py::_MASK_NEG_VALUE`
#: と同じ考え方）。MLX 側フォワード（`_build_additive_mask`）と ONNX 側グラフ
#: （`_export_ar_onnx`）の両方がこの 1 箇所を参照する。
_MASK_NEG_VALUE = -1e9

#: `build_prediction_record` が受け付ける予測レコードの `id`（入力の識別子。
#: 入力本文そのものは入れない。security.md）のバイト長上限（#80・REQ-39）。
#: `crates/core/src/judgment.rs::MAX_INPUT_ID_BYTES` と同じ値を使う（Rust 側
#: の `infer` 工程〔TASK-33.1。現状未配線〕が受理する `id` の上限と揃え、
#: 学習ワーカー側で先に拒否できるようにする）。
MAX_PREDICTION_ID_BYTES = 1024

#: `DecoderLayer._attn` が 1 層あたりに保持する `[..., L, L]` 形状のテンソル数
#: （`scores`＝`softmax` 適用前のスケール済みスコア、`attn`＝`softmax` の出力
#: 確率）。MLX の自動微分は逆伝播で `softmax` の勾配計算に出力（`attn`）を、
#: `*scale + mask` の勾配計算に入力（`scores`）を、それぞれ必要とするため、
#: 学習時はこの 2 テンソルが同時にメモリ上に残り得る（Codex P0 指摘。
#: `AutoregressiveKind.train` の資源上限検査 `attn_elements` 参照。REQ-39）。
#: `ctx`（`attn @ v`）は `[..., L, head_dim]` で `head_dim ≤ L` の典型的な
#: 設定では `L^2` より小さいため、保守的な見積もりとして数えない。
_AR_TRAIN_ATTN_RETAINED_TENSORS = 2

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

    `predict_records`（#80・TASK-19b.1-2）がこの関数を再利用して 1 件ずつの
    予測レコードを組み立てる（モジュール docstring）。本関数自体は正規化前の
    対数尤度を返すだけで、predicted_label・scores への変換は呼び出し側
    （`map_scores_to_choice`・`build_prediction_record`）の責務とする
    （`test_ar_train.py` の golden テスト・簡易正解率確認も直接の呼び出し元）。

    `full`（`[N*K, length]`）・decoder の attention（`[N*K, heads, length,
    length]`）を確保する前に、確保見込み要素数を `MAX_AR_EXPORT_ATTENTION_
    ELEMENTS`（`N × K × heads × layers × L^2`。書き出し時 `_check_ar_export_
    resources` が同じ式で検査する上限を、実行時の同じ形状に対しても流用
    する）で fail-closed に拒否する。`chunk_size`（`MAX_AR_BATCH_SIZE` 以下）
    はチャンクあたりの計算量を抑えるが、選択肢数 `K`・系列長 `length` は
    モデル構成・データに依存するため、`chunk_size` だけでは確保量を
    抑えきれない（Codex レビュー指摘 P0・PR #234。REQ-39「資源の上限」）。

    `logits`・`log_softmax`（いずれも `[N*K, length, VOCAB_SIZE]`）は
    attention とは別に vocab 次元 `VOCAB_SIZE` 分の要素数を確保するため、
    attention の見積もりだけでは vocab サイズが大きい構成を見逃す
    （`_export_ar_onnx` の `choice_logprob_elements` 検査は書き出し時の
    `[K, M, VOCAB_SIZE]` のみを見ており、本関数の `[N*K, length,
    VOCAB_SIZE]`〔`length` は選択肢領域 `M` より広い〕には及ばない。
    Cursor Bugbot 指摘・PR #234）。`full` を確保する前に、この形状の
    見積もり要素数を `MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS` で
    fail-closed に拒否する。
    """
    model.eval()
    n = len(ids_batch)
    choice_tokens, valid_mask = _build_choice_block(choice_id_list)
    k, m = choice_tokens.shape
    t = max(1, max(len(x) for x in ids_batch))
    length = t + 1 + m

    layers_n = len(model.layers)
    heads = model.layers[0].heads if model.layers else 0
    attn_elements = n * k * heads * layers_n * length * length
    if attn_elements > MAX_AR_EXPORT_ATTENTION_ELEMENTS:
        raise WorkerError(
            "limit_exceeded",
            f"estimated attention elements {attn_elements} exceeds limit"
            f" {MAX_AR_EXPORT_ATTENTION_ELEMENTS} (N x K x heads x layers x length^2)",
            ExitCode.LIMIT_EXCEEDED,
        )

    logprob_elements = n * k * length * VOCAB_SIZE
    if logprob_elements > MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS:
        raise WorkerError(
            "limit_exceeded",
            f"estimated logits/log-softmax elements {logprob_elements} exceeds limit"
            f" {MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS} (N x K x length x vocab_size)",
            ExitCode.LIMIT_EXCEEDED,
        )

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


@dataclass(frozen=True)
class Mapped:
    """対応づけ (b) が選択肢 ID の解決まで成功したことを表す（#80）。

    壊れた値（`choice_id` が `None` になりうる等）を表現できない型にする
    ため、`Unmapped` と分けた 2 型で `ChoiceMapping` を構成する
    （coding-rust.md「判定結果・状態は enum で表し壊れた値を表現できない
    型にする」と同じ考え方を Python 側でも踏襲する）。

    `probs` は `label_order` の宣言順に並べた事後確率
    （`choice_posteriors` の出力をそのまま保持する）。`index` は argmax の
    添字（タイブレークは宣言順。`crates/core/src/judgment.rs` の
    `predicted_choice_id` タイブレーク規則と揃える）。
    """

    choice_id: str
    index: int
    probs: tuple[float, ...]


@dataclass(frozen=True)
class Unmapped:
    """対応づけ (b) が選択肢 ID を解決できなかったことを表す（#80）。

    `reason` はテストと将来の #81（TASK-19b.2。判定不能を別の status として
    区別する扱い）のために保持するだけの内部値で、JSON の予測レコードには
    出さない（`build_prediction_record` 参照。理由は
    `"invalid_score"`〔`choice_posteriors` が非有限と判定〕・
    `"no_exact_match"`〔`resolve_choice_id` が argmax の選択肢と一致する ID
    を解決できない防御的経路。対応づけ (b) では理論上起こらない〕の 2 つ）。
    """

    reason: str


#: 対応づけ (b) の結果を表す型（`map_scores_to_choice` の戻り値）。
ChoiceMapping = Mapped | Unmapped


def choice_posteriors(loglik_row: np.ndarray) -> np.ndarray | None:
    """対応づけ (b) の事後確率（選択肢間の softmax）を求める（#80。PoC-24
    `_predict_one` 相当。`docs/spec/03-poc/model-kind-selector/
    preregistration.md` 3 節の事前登録どおり、長さ正規化を行わない
    `_score_choices_mlx` の出力へそのまま softmax を適用する）。

    `loglik_row` は `_score_choices_mlx` が返す `[N, K]` の 1 行。float64 で
    `x - max` を引いてから `exp` を取り合計で割る、数値的に安定化した
    softmax を計算する。入力が 1 次元・長さ 1 以上・全要素が有限であること
    を事前に検証し、満たさなければ `None` を返す（呼び出し側で
    `Unmapped("invalid_score")` にする。fail-closed）。出力も有限であること
    を再確認し、満たさなければ同様に `None` を返す（`ok` を装わない）。
    """
    if loglik_row.ndim != 1 or loglik_row.shape[0] < 1:
        return None
    row = np.asarray(loglik_row, dtype=np.float64)
    if not np.all(np.isfinite(row)):
        return None
    shifted = row - row.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    if not np.all(np.isfinite(probs)):
        return None
    return probs


def resolve_choice_id(choice_tokens: Sequence[int], label_order: Sequence[str]) -> str | None:
    """選択肢のトークン列（バイト+1。`_encode_choices` と同じ表現）を、
    `label_order` の中からバイト単位で完全一致する選択肢 ID へ解決する
    （#80。受入基準の「完全一致する具体例」を担う純粋関数。前方一致・
    NFKC 等の正規化・大小文字の同一視はしない。モジュール docstring 5 番の
    「選択肢は NFKC 正規化しない」方針と整合させる）。

    各トークンが `_encode_choices` が使う「バイト値+1」の語彙 `1..256` の
    範囲内であることをまず確認する。PAD（0）・SEP（257）・EOS（258）・
    範囲外の値が 1 つでも混ざっていれば、デコードを試みず `None` を返す
    （例外は送出しない。「復号できない列」を単に「一致無し」として扱う）。
    範囲内であれば `token - 1` へ戻したバイト列を組み立て、`label_order` の
    各要素を UTF-8 エンコードしたバイト列と直接比較する（str へのデコード
    を経由しないため、`choice_tokens` が有効な UTF-8 でなくても例外を出さず
    「一致無し」を返せる）。
    """
    for token in choice_tokens:
        if not (1 <= token <= 256):
            return None
    decoded_bytes = bytes(token - 1 for token in choice_tokens)
    for label in label_order:
        if decoded_bytes == label.encode("utf-8"):
            return label
    return None


def map_scores_to_choice(
    loglik_row: np.ndarray,
    label_order: Sequence[str],
    choice_ids_by_label: dict[str, list[int]],
) -> ChoiceMapping:
    """対応づけ (b) の最終段（#80）。`_score_choices_mlx` が返す 1 行の対数
    尤度合計を、事後確率つきの選択肢 ID（`Mapped`）または対応づけ不能
    （`Unmapped`）へ変換する。

    手順: (1) `choice_posteriors` で事後確率を求める。非有限・不正な形なら
    `Unmapped("invalid_score")` を返す。(2) `argmax`（タイブレークは
    `label_order` の宣言順で先頭。`np.argmax` は同点のとき最初の添字を返す
    ため追加の分岐は要らない）で最大の選択肢を選ぶ。(3) その選択肢の
    トークン列を `resolve_choice_id` で実際に ID へ解決し、選んだ添字の
    `label_order[idx]` と一致することを確認する。対応づけ (b) では
    `choice_ids_by_label[label_order[idx]]` は `label_order[idx]` 自身の
    トークン列なので理論上必ず一致するが、`_encode_choices` の呼び出し
    契約が崩れた場合に `ok` を偽装しないよう、fail-closed に
    `Unmapped("no_exact_match")` へ倒す経路を残す。

    `len(loglik_row)` と `len(label_order)` の不一致は、呼び出し側が学習・
    書き出し時と異なる選択肢集合を渡した実装バグであり、データの問題では
    ないため `WorkerError`（runtime_error・exit 70）で即座に停止する
    （黙って切り詰めない。coding-rust.md「外部入力の経路では添字アクセス
    を使わず明示的に処理する」と同じ fail-closed の考え方）。
    """
    if len(loglik_row) != len(label_order):
        raise WorkerError(
            "runtime_error",
            f"loglik row length {len(loglik_row)} does not match label_order length"
            f" {len(label_order)}",
            ExitCode.RUNTIME_ERROR,
        )

    probs = choice_posteriors(np.asarray(loglik_row))
    if probs is None:
        return Unmapped("invalid_score")

    idx = int(np.argmax(np.asarray(loglik_row)))
    label = label_order[idx]
    resolved = resolve_choice_id(choice_ids_by_label[label], label_order)
    if resolved != label:
        return Unmapped("no_exact_match")
    return Mapped(choice_id=resolved, index=idx, probs=tuple(float(p) for p in probs))


def build_prediction_record(
    record_id: str, mapping: ChoiceMapping, label_order: Sequence[str]
) -> dict[str, Any]:
    """1 件の予測レコード（評価器が採点できる形。`crates/core/src/
    judgment.rs` の正常系スキーマ `{id, status, predicted_label, scores}` と
    揃える。#80）を組み立てる。

    `Mapped` のとき `status:"ok"`・`scores` は `label_order` の宣言順で
    `{label: 事後確率}` を持つ dict にする（Python 3.7+ の dict は挿入順を
    保つため、後から並べ替えない。`judgment.rs` の「`scores` のキー順は
    定義ファイルの `options` の宣言順で固定する」契約と揃える）。
    `Unmapped` のとき `status:"error"`・`predicted_label: None` とし、
    `scores` は付けない（いずれも `crates/data/src/eval_input.rs` が受理
    する値）。`reason_code` 等、Rust 側スキーマに無いフィールドは追加しない
    （PoC-24 にあった `reason_code` は意図的に落とす。
    coding-python.md「Python 側で独自のフィールドを増やさない」）。

    `Unmapped` は現状 `status:"error"` として評価の分母に含まれる。#81・
    TASK-19b.2 で「判定不能」を別の status として区別する場合は、Rust 側
    （`crates/data::eval_input` の status 許可集合・
    `crates/core::JudgmentStatus`）へ先に値を追加しなければ
    `unknown_status` として拒否される。

    `id` には入力本文を入れない契約（security.md）は呼び出し側
    （`predict_records`）が守る前提で、ここでは型・非空・バイト長のみを
    検証する（[`MAX_PREDICTION_ID_BYTES`]）。
    """
    if not isinstance(record_id, str) or not record_id:
        raise WorkerError(
            "invalid_request", "prediction id must be a non-empty string", ExitCode.INVALID_INPUT
        )
    # UTF-8 のバイト数は文字数（コードポイント数）以上であることを利用し、
    # まず文字数で足切りする（`.encode("utf-8")` は文字数の取得と違い入力
    # 全体を確保するため、検査より先に呼ぶと巨大な `id` でメモリを無制限に
    # 消費しうる。`_guarded_encode_bytes` の P0 レビュー指摘〔PR #234〕と
    # 同種の問題のため同じ順序で予防する。REQ-39）。
    id_char_len = len(record_id)
    if id_char_len > MAX_PREDICTION_ID_BYTES:
        raise WorkerError(
            "invalid_request",
            f"prediction id exceeds {MAX_PREDICTION_ID_BYTES} utf-8 bytes"
            f" ({id_char_len} chars, utf-8 encoding would be at least that many bytes)",
            ExitCode.INVALID_INPUT,
        )
    id_len = len(record_id.encode("utf-8"))
    if id_len > MAX_PREDICTION_ID_BYTES:
        raise WorkerError(
            "invalid_request",
            f"prediction id exceeds {MAX_PREDICTION_ID_BYTES} utf-8 bytes ({id_len} bytes)",
            ExitCode.INVALID_INPUT,
        )

    if isinstance(mapping, Mapped):
        scores = {label: mapping.probs[i] for i, label in enumerate(label_order)}
        return {
            "id": record_id,
            "status": "ok",
            "predicted_label": mapping.choice_id,
            "scores": scores,
        }
    return {"id": record_id, "status": "error", "predicted_label": None}


def _guarded_encode_bytes(text: str, max_bytes: int) -> list[int]:
    """`predict_records` から呼ぶ前に、正規化前の生入力のバイト数を検査
    してから `encoding.encode_bytes` を呼ぶ（REQ-39）。

    `encode_bytes` は NFKC 正規化・UTF-8 化を行った「後」に `max_bytes` へ
    切り詰めるため、この検査を経ずに呼ぶと 1 件の巨大な入力で正規化コスト
    （壁時計・メモリ）が無制限になる（Codex レビュー指摘 P0・PR #234）。
    上限には学習データの行読み込み（`contract.py`）と同じ `MAX_TRAIN_
    LINE_BYTES` を流用する（同じ「1 件の入力テキスト」という種類の上限を
    2 箇所で別々の値として持たないため）。

    `text.encode("utf-8")` そのものが入力全体のバイト列を新たに確保する
    ため、検査の「前」にこれを呼ぶと上限判定より先に巨大な入力でメモリを
    消費しうる（Codex レビュー指摘 P0・PR #234。#80 で `_guarded_encode_
    bytes` を新設した際に混入した回帰）。UTF-8 のバイト数は文字数
    （コードポイント数）以上であることを利用し、まず `len(text)`
    （O(1)。`encode` を伴わない）で足切りしてから `encode("utf-8")` を
    呼ぶことで、確保量を高々 `4 * MAX_TRAIN_LINE_BYTES`
    （UTF-8 の 1 コードポイントあたり最大 4 バイト）に抑える。
    """
    char_len = len(text)
    if char_len > MAX_TRAIN_LINE_BYTES:
        raise WorkerError(
            "limit_exceeded",
            f"prediction input {char_len} chars exceeds limit {MAX_TRAIN_LINE_BYTES} bytes"
            " (utf-8 encoding would be at least that many bytes)",
            ExitCode.LIMIT_EXCEEDED,
        )
    raw_len = len(text.encode("utf-8"))
    if raw_len > MAX_TRAIN_LINE_BYTES:
        raise WorkerError(
            "limit_exceeded",
            f"prediction input {raw_len} bytes exceeds limit {MAX_TRAIN_LINE_BYTES} bytes",
            ExitCode.LIMIT_EXCEEDED,
        )
    return encode_bytes(text, max_bytes)


def predict_records(
    trained: AutoregressiveTrainedModel,
    rows: Sequence[tuple[str, str]],
    *,
    chunk_size: int | None = None,
    resource_budget: budget_mod.ResourceBudget | None = None,
) -> Iterator[dict[str, Any]]:
    """学習ワーカー内で対応づけ (b) を確認するための推論経路（#80。PoC-24
    `predict` 相当）。`rows` は `(id, input)` の列。

    正解ラベル（gold）は受け取らない（REQ-27。推論関数へは `input` だけを
    渡すという評価の独立性を、学習ワーカー内の確認経路でも維持する）。

    戻り値はジェネレータ（呼び出し元が 1 件ずつ消費する）。以前は全件を
    `records: list[dict]` へ蓄積してから返していたが、`chunk_size` は
    1 チャンクあたりの計算量しか制限せず、`rows` 自体の件数上限
    （`MAX_AR_PREDICT_ROWS`）は数百万件のオーダーになりうるため、全件を
    リストへ蓄積する設計では上限内でも RSS が際限なく増える
    （Codex レビュー指摘 P0・PR #234。security.md「ガード層: 資源の上限」）。
    呼び出し元（将来の CLI `infer --input-file`。TASK-33.1）は
    `prediction_record_to_json_line` で 1 件ずつ書き出す想定のため、
    チャンクごとの逐次出力（yield）に変更し、全件保持を避ける。
    **呼び出し元は必ずイテレートすること**（この関数はジェネレータ関数の
    ため、呼ぶだけでは本体〔件数上限検査を除く〕は実行されない）。

    件数上限検査（`MAX_AR_PREDICT_ROWS`）だけは呼び出し直後・同期的に行う
    （fail-closed。呼び出し元がイテレートし忘れても、明らかに上限超過の
    呼び出しは即座に拒否する）。件数上限検査より後は `rows`（`Sequence`）
    をそのまま `_predict_records_stream` へ渡し、`list(rows)` による全件
    コピーはしない（上限内の件数〔`MAX_AR_PREDICT_ROWS` は数百万件の
    オーダー〕でも、コピーそのものが検査前に RSS を消費してしまうという
    Codex レビュー指摘 P0・PR #234。呼び出し元は list・tuple 等スライス
    可能な `Sequence` を渡す契約とする）。

    `chunk_size`（既定は `trained.config["batch_size"]`、上限は
    `MAX_AR_BATCH_SIZE`）ごとに `_score_choices_mlx` を呼ぶ。`resource_budget.
    check()` は各チャンクにつき、(1) そのチャンクの `id_lists`（正規化後の
    バイト列）を確保する「前」、(2) `_score_choices_mlx` が `[N*K, L]`・
    attention `[N*K, heads, L, L]` を確保した直後・最初の `yield` を呼ぶ
    「前」の 2 回呼ぶ（REQ-39）。計算後の検査を全件 `yield` した後まで
    遅らせると、呼び出し元が途中で反復を止めた場合に検査自体が行われず、
    継続する場合も上限超過後の結果が先に呼び出し元へ渡ってしまう
    （Codex レビュー指摘 P1・PR #234）。`_score_choices_mlx` 自体も
    確保前に `MAX_AR_EXPORT_ATTENTION_ELEMENTS` で見積もりベースの拒否を
    行うため、`resource_budget` は実測 RSS による最終防御として併用する。

    各行の生テキストは `encode_bytes` を呼ぶ前（NFKC 正規化・UTF-8 化の前）
    に `MAX_TRAIN_LINE_BYTES` でバイト数を検査する（`encode_bytes` は
    `max_bytes` への切り詰めを正規化の「後」に行うため、検査なしでは
    1 件の巨大な入力が正規化コストを無制限に消費しうる。学習データ側は
    `contract.py` の行読み込みで同じ上限を既に保証しているため、予測経路
    でも同じ上限値を流用する。Codex レビュー指摘 P0・PR #234）。

    `id` の重複は検査しない（重複の検査はデータ契約層
    `crates/data::eval_input` の責務であり、ここで評価ロジックを再実装
    しない。coding-python.md「評価ロジックを Python に再実装しない」）。
    """
    if chunk_size is None:
        chunk_size = int(trained.config.get("batch_size", DEFAULT_CONFIG["batch_size"]))
    chunk_size = max(1, min(int(chunk_size), MAX_AR_BATCH_SIZE))

    choice_id_list = [trained.choice_ids_by_label[label] for label in trained.label_order]

    # `rows` は型上は `Sequence`（呼び出し元が既に全件を保持している前提）
    # だが、ここで無制限にリスト化・レコード蓄積をしないよう、件数を
    # 蓄積前に検査する（`chunk_size` は 1 チャンクの計算量しか制限しない。
    # MAX_AR_PREDICT_ROWS docstring 参照）。この検査はジェネレータの外
    # （呼び出しの時点で同期的）に置き、呼び出し元が結果をイテレートし
    # 忘れても上限超過を確実に拒否する。
    n_rows = len(rows)
    if n_rows > MAX_AR_PREDICT_ROWS:
        raise WorkerError(
            "limit_exceeded",
            f"prediction rows {n_rows} exceeds limit {MAX_AR_PREDICT_ROWS}",
            ExitCode.LIMIT_EXCEEDED,
        )

    # `list(rows)` で全件コピーしない（`rows` は既に上で件数検査済みだが、
    # 上限内〔数百万件のオーダー〕でもコピー自体が検査前に RSS を消費する。
    # Codex レビュー指摘 P0・PR #234）。`rows` をそのまま渡し、チャンクへの
    # スライスは `_predict_records_stream` 側でチャンクぶんだけ行う。
    return _predict_records_stream(trained, rows, chunk_size, choice_id_list, resource_budget)


def _predict_records_stream(
    trained: AutoregressiveTrainedModel,
    rows: Sequence[tuple[str, str]],
    chunk_size: int,
    choice_id_list: list[list[int]],
    resource_budget: budget_mod.ResourceBudget | None,
) -> Iterator[dict[str, Any]]:
    """`predict_records` のチャンクごとの逐次出力本体（件数上限検査済みの
    `rows` を受け取る）。全件を `list` へ蓄積せず 1 件ずつ `yield` する。

    `resource_budget.check()` はチャンクごとに 2 回呼ぶ: (1) そのチャンクの
    `id_lists` を確保する「前」（チャンクのスライス・エンコードより前）、
    (2) `_score_choices_mlx` の呼び出し「直後・最初の `yield` より前」
    （Codex レビュー指摘 P1・PR #234。計算後の検査を全レコード `yield` した
    後まで遅らせると、呼び出し元が途中で反復を止めた場合に検査されず、
    継続する場合も上限超過後の結果が先に渡ってしまう）。
    """
    n_rows = len(rows)
    for start in range(0, n_rows, chunk_size):
        if resource_budget is not None:
            resource_budget.check()
        chunk = rows[start : start + chunk_size]
        id_lists = [_guarded_encode_bytes(text, trained.max_bytes) for _row_id, text in chunk]
        scores = _score_choices_mlx(trained.model, id_lists, choice_id_list)
        if resource_budget is not None:
            resource_budget.check()
        for (row_id, _text), row in zip(chunk, scores, strict=True):
            mapping = map_scores_to_choice(row, trained.label_order, trained.choice_ids_by_label)
            yield build_prediction_record(row_id, mapping, trained.label_order)


def prediction_record_to_json_line(record: dict[str, Any]) -> str:
    """予測レコードを JSON 1 行へ直列化する（#80。CLI の `infer --input-file`
    契約〔evaluation-contract.md「入出力契約」〕と同じ「1 行 1 JSON」の形）。

    鍵の順序は `id → status → predicted_label → scores`（`judgment.rs` の
    直列化順に揃える。`build_prediction_record` がこの順で dict を作り、
    `json.dumps` はその挿入順をそのまま書き出す）。`allow_nan=False` を
    指定し、NaN・Infinity が紛れ込んだレコードを `WorkerError`
    （runtime_error・exit 70）へ倒す（既定の `json.dumps` は `NaN` という
    構文上不正な JSON リテラルをそのまま出してしまうため。「ソフトウェア
    とデータの完全性」security.md）。
    """
    try:
        return json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except ValueError as exc:
        raise WorkerError(
            "runtime_error",
            f"prediction record failed to serialize to valid JSON: {exc}",
            ExitCode.RUNTIME_ERROR,
        ) from exc


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
        # `layers` を乗じる理由: `ByteDecoder.__call__` は `layers` 個の
        # `DecoderLayer` を直列に適用するが、MLX の自動微分は逆伝播のために
        # 各層のフォワード時の中間テンソルを保持する。全層分が学習ループの
        # 1 ステップ内で同時にメモリ上に残り得るため、1 層あたりの見積もりに
        # `layers` を掛けないと過小評価になる（Codex P0 指摘。修正前は
        # `layers` を含まず、`layers=32`・`heads=dims=64`・`batch_size=1`・
        # `max_bytes≈1700` のような設定が判定を素通りしていた。REQ-39）。
        # `_AR_TRAIN_ATTN_RETAINED_TENSORS`（=2）を乗じる理由は同定数の
        # docstring 参照（`scores`・`attn` の 2 テンソルが逆伝播用に残る）。
        attn_elements = (
            batch_size * heads * max_len * max_len * layers * _AR_TRAIN_ATTN_RETAINED_TENSORS
        )
        if attn_elements > MAX_AR_ATTENTION_ELEMENTS:
            raise WorkerError(
                "limit_exceeded",
                f"estimated attention elements {attn_elements} exceeds limit"
                f" {MAX_AR_ATTENTION_ELEMENTS} (batch_size x heads x max_len^2 x layers x"
                f" {_AR_TRAIN_ATTN_RETAINED_TENSORS})",
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
    n: int, k_classes: int, t_bound: int, m: int, layers: int, heads: int
) -> int:
    """decoder attention が確保しうる要素数の見積もり（`N × K × heads ×
    layers × L^2`。`L = T + 1 + M`）を返す（REQ-39・PR #222 レビュー指摘）。

    推論ランタイム側で `N`・`T` の実測値を使ってそのまま検査できるよう、
    `N`・`T` を固定値に決め打ちしない計算式として実装している（本ワーカー
    〔Python〕は実際の推論時の `N`・`T` を観測できないため、書き出し時には
    `_check_ar_export_resources` が `N=1` を渡す。呼び出し側の docstring
    参照）。

    decoder は選択肢展開後の `[N*K, L, L]` 形状で attention を計算するため
    `N`・`K` を乗じ、各層で同形状のテンソルを確保しうるため `layers` を
    乗じる（層間でメモリが解放される実装でも、上限側は保守的に見積もる）。
    引数はすべて 0 以上の整数であること（負数・非整数は呼び出し側の
    バグであり、ここでは検査しない。すべて学習時に確定する固定値、
    または呼び出し側が検証済みの `N`・`T` のみを渡す契約のため）。

    **学習側の見積もり（`AutoregressiveKind.train` の `attn_elements`）との
    違い**: 学習側は `_AR_TRAIN_ATTN_RETAINED_TENSORS`（=2）を追加で乗じる
    （逆伝播のため `scores`・`softmax` 出力の 2 テンソルを同時に保持しうる
    ため。Codex P0 指摘・REQ-39）。推論（本関数が担う ONNX 書き出し）には
    逆伝播が無く、ONNX ランタイムは各層の attention 出力を次の層へ渡した
    後に解放できる（1 層のフォワード計算に必要な一時テンソルだけを保持
    すればよい）ため、学習側と同じ「保持テンソル数」の乗数は不要と判断
    した。本関数はそれでも `layers` を乗じる保守的な見積もりを維持して
    いる（上記コメント参照。実際に層ごとに解放される実装でも上限側は
    安全側に倒す）ため、学習側の追加乗数を持ち込まなくても fail-closed な
    安全側の見積もりを保てる。
    """
    length = t_bound + 1 + m
    return n * k_classes * heads * layers * length * length


def _check_ar_export_resources(
    k_classes: int, t_bound: int, m: int, layers: int, heads: int
) -> None:
    """`_ar_export_attention_elements` の `N=1`・`T=t_bound` での見積もりが
    `MAX_AR_EXPORT_ATTENTION_ELEMENTS` を超えるとき `limit_exceeded` で
    fail-closed に拒否する（REQ-39・PR #222 レビュー指摘）。グラフ構築前に
    呼ぶことで、過大な attention テンソルを実際に確保する前に停止する。

    `t_bound`（`= trained.max_bytes`）は、`_export_ar_onnx` が組み込む
    `Slice(ids, 0, max_bytes, axis=1)` により、推論時に渡される `T` が
    どのような値（PAD を多く含む場合を含む）であっても、グラフ内部で
    実際に計算に使われる長さの**厳密な上限**になる（モジュール docstring
    6 番参照。以前はここが「密な入力かつ PAD なし」に限った近似だったが、
    Slice の導入で PAD の有無に関わらず保証されるようになった）。
    `N`（バッチ件数）についてのみ、本ワーカーは実測値を観測できないため
    `N=1` の最小ケースを検査する。`N > 1` 分の上限は、この関数が返す
    N=1 相当の見積もり（`elements`）を使って `_ar_export_max_batch_n` が
    算出する `n_max` を ONNX グラフ内の固定長テーブル `Gather` へ埋め込み、
    グラフの内側で fail-closed に検査する（オーナー判断 2026-09-28・
    PR #222 レビュー指摘。`_export_ar_onnx` 参照）。
    """
    elements = _ar_export_attention_elements(1, k_classes, t_bound, m, layers, heads)
    if elements > MAX_AR_EXPORT_ATTENTION_ELEMENTS:
        raise WorkerError(
            "limit_exceeded",
            f"estimated per-example attention elements {elements} (at N=1, T={t_bound})"
            f" exceeds limit {MAX_AR_EXPORT_ATTENTION_ELEMENTS}"
            " (n x k_classes x heads x layers x (t_bound+1+m)^2, n=1)",
            ExitCode.LIMIT_EXCEEDED,
        )


def _ar_export_max_batch_n(
    per_n_attention_elements: int, per_n_choice_logprob_elements: int
) -> int:
    """このモデル構成での推論バッチ件数 `N` の上限（`n_max`）を返す
    （REQ-39・オーナー判断 2026-09-28・PR #222 レビュー指摘）。

    `_export_ar_onnx` が ONNX グラフ内へ埋め込む固定長テーブル
    `Gather` の長さ（`n_max + 1`）を決めるために使う。次の 3 値の
    最小値を取る:

    1. `MAX_AR_INFER_BATCH_N`（モデル構成に依らない固定シーリング。
       `limits.py` docstring 参照）
    2. `MAX_AR_EXPORT_ATTENTION_ELEMENTS // per_n_attention_elements`
       （attention テンソルが `N` に比例して膨らむことから逆算した上限）
    3. `MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS // per_n_choice_logprob_elements`
       （選択肢対数確率テンソルが `N` に比例して膨らむことから逆算した上限）

    呼び出し側（`_export_ar_onnx`）は `_check_ar_export_resources`・
    `choice_logprob_elements` の検査で `per_n_attention_elements ≤
    MAX_AR_EXPORT_ATTENTION_ELEMENTS`・`per_n_choice_logprob_elements ≤
    MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS` を既に確認済みのため、2・3 の
    商はいずれも 1 以上になる（`max(..., 1)` は 0 除算・0 除算相当の
    n_max=0 を避ける防御であり、通常経路では発動しない）。
    """
    n_from_attention = MAX_AR_EXPORT_ATTENTION_ELEMENTS // max(per_n_attention_elements, 1)
    n_from_choice = MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS // max(per_n_choice_logprob_elements, 1)
    return max(1, min(MAX_AR_INFER_BATCH_N, n_from_attention, n_from_choice))


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
    # N 分の上限は、この N=1 相当の値を `_ar_export_max_batch_n` へ渡して
    # 算出する `n_max` を、後述の ONNX グラフ内固定長テーブル `Gather` で
    # 検査する（オーナー判断 2026-09-28・PR #222 レビュー指摘）。
    choice_logprob_elements = k_classes * m * VOCAB_SIZE
    if choice_logprob_elements > MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS:
        raise WorkerError(
            "limit_exceeded",
            f"estimated per-example choice log-prob elements {choice_logprob_elements}"
            f" exceeds limit {MAX_AR_EXPORT_CHOICE_LOGPROB_ELEMENTS} (k_classes x m x"
            " vocab_size)",
            ExitCode.LIMIT_EXCEEDED,
        )

    # 推論時に渡されうる T（動的軸）は、グラフ内で組み込む
    # `Slice(ids, 0, max_bytes, axis=1)`（後述）により学習時の `max_bytes`
    # を厳密な上限としてクランプされる（モジュール docstring 6 番・
    # `_check_ar_export_resources` docstring 参照。PAD の有無に関わらず
    # 成立する）。よって N=1・T=t_bound（`= trained.max_bytes`。Slice 後に
    # グラフが取りうる最大の T）での decoder attention 要素数を見積もって
    # 検査すれば、書き出し可能な最大構成を確実に拒否できる
    # （`limits.py::MAX_AR_EXPORT_ATTENTION_ELEMENTS` docstring・
    # `_ar_export_attention_elements` docstring 参照。REQ-39・PR #222
    # レビュー指摘）。
    t_bound = trained.max_bytes
    _check_ar_export_resources(k_classes, t_bound, m, layers, heads)

    # このモデル構成での推論バッチ件数 N の上限（`n_max`）を算出し、後述の
    # ONNX グラフ内固定長テーブル `Gather`（`n_guard_table`）へ埋め込む
    # 長さとして使う（`_ar_export_max_batch_n` docstring 参照。REQ-39・
    # オーナー判断 2026-09-28・PR #222 レビュー指摘）。
    per_n_attention_elements = _ar_export_attention_elements(
        1, k_classes, t_bound, m, layers, heads
    )
    n_max = _ar_export_max_batch_n(per_n_attention_elements, choice_logprob_elements)

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
        # `ids` 値域ガード用（`MAX_INPUT_ID` docstring・下記コメント参照。
        # REQ-39・オーナー判断 2026-09-28・PR #222 レビュー指摘）。
        numpy_helper.from_array(
            np.array(MAX_INPUT_ID + 1, dtype=np.int64), name="max_input_id_plus_1"
        ),  # 0-d
        # `N` 上限ガード用の固定長テーブル（`n_max + 1` 要素の 0 埋め。
        # `_ar_export_max_batch_n` が算出した `n_max` をこのモデルの
        # 書き出し時に固定する。REQ-39・オーナー判断 2026-09-28・
        # PR #222 レビュー指摘）。
        _i64(np.zeros(n_max + 1, dtype=np.int64), "n_guard_table"),
        _i64(np.array([0]), "idx0"),
        _i64(np.array([1]), "idx1"),
        _i64(np.array([2]), "idx2"),
        _i64(np.array([-1]), "negidx"),
        _i64(np.array([1]), "one_vec"),
        _i64(np.array([k_classes]), "k_vec"),
        _i64(np.array([m]), "m_vec"),
        _i64(np.array([trained.max_bytes]), "max_bytes_vec"),
        _f32(np.array(1.0), "one_f32"),
        _f32(np.array(_MASK_NEG_VALUE), "neg_big_f32"),
        numpy_helper.from_array(np.array(VOCAB_SIZE, dtype=np.int64), name="vocab_depth"),  # 0-d
        _f32(np.array([0.0, 1.0]), "onehot_values"),  # [2]（off_value, on_value）
        _f32(embed, "embed_table"),
        _f32(pos_table, "pos_table"),
    ]
    nodes: list = []

    # --- T 軸（動的軸）を max_bytes へ Slice で切り詰める（モジュール
    # docstring 6 番・REQ-39・PR #222 セキュリティレビュー P0 指摘）。
    # PAD を含む・含まない入力に関わらず、以降のグラフ全体が参照する
    # `ids_trunc` の T は常に `max_bytes` 以下になる。opset 13 の `Slice` は
    # data・starts・ends・axes を入力で受ける（属性ではない）。`ends` が
    # 実際の次元数を超える場合は次元数へ丸められる（ONNX 仕様。
    # onnx.reference で実測確認済み）ため、`T ≤ max_bytes` の入力には
    # 影響しない。切り詰めは先頭 `max_bytes` バイトを残す
    # （`starts=[0]`）方向で、`encoding.encode_bytes` の切り詰め規則
    # （`fixtures/preprocess/byte_encoding_vectors.json::
    # truncate_long_ascii`）と一致させる。
    nodes.append(helper.make_node("Slice", ["ids", "idx0", "max_bytes_vec", "idx1"], ["ids_trunc"]))

    # --- `ids` の値域ガード（REQ-39・オーナー判断 2026-09-28・PR #222
    # レビュー指摘。Slice の直後・選択肢展開より前に適用する）。外部入力
    # `ids` として許可するのは `encoding.encode_bytes` の語彙 `[0,
    # MAX_INPUT_ID]`（PAD=0・バイト値+1=1..256）のみで、SEP（257）・
    # EOS（258）はグラフが選択肢展開の際に内部で挿入する特殊トークンの
    # ため、外部入力としては許可しない。範囲外（負数・SEP・EOS・
    # `VOCAB_SIZE` 以上いずれも）の値は `Where` で「必ず範囲外になる値」
    # （`vocab_depth`=VOCAB_SIZE=259。埋め込み表 `embed_table` の行数と
    # 同じ値のため、`Gather(embed_table, ...)` は必ず範囲外参照で失敗
    # する）へ書き換えてから `ids_exp3`（選択肢展開の起点）へ渡す。
    #
    # 切り詰め（Slice）の**後**に検査する理由: `max_bytes` を超えた
    # 部分は Slice で捨てられ、以降のどの演算にも一切使われない
    # （モジュール docstring 6 番参照）ため、切り詰め前に検査しても
    # 意味のある追加の安全性は無く、切り詰め後の短い列だけを検査すれば
    # グラフに実際に効く全ての値を確実に検査できる（計算量も T ではなく
    # 切り詰め後の長さで済む）。
    #
    # 負のインデックスを `Where` の前に弾く理由: ONNX の `Gather` は
    # `[-VOCAB_SIZE, -1]` の負インデックスを末尾から「黙って」wrap する
    # （例: `-1` は EOS 行）ため、`Where` を経由せずに負値が直接
    # `Gather` へ渡ると、別の正当な値として何のエラーも無く処理されて
    # しまう（`tests/test_ar_ids_range.py::
    # test_ar_ids_negative_one_silently_wraps_to_last_vocab_row` で実測
    # 確認済み）。`Where` によるガードはこの wrap 経路を経由させず、
    # 範囲外の値をすべて「確実に範囲外になる」正の値（VOCAB_SIZE）へ
    # 統一することで、負値・特殊 ID・VOCAB_SIZE 以上のいずれも同じ
    # fail-closed な `Gather` 失敗に帰着させる。
    #
    # このガードは「範囲外 Gather はエラーになる」という前提（下の
    # embed_table への `Gather` 直前のコメント参照）に依存する。その前提の
    # 根拠・適用範囲・多層防御としての位置づけは、同コメントへ集約する
    # （Cursor Medium 指摘）。
    nodes.append(helper.make_node("GreaterOrEqual", ["ids_trunc", "zero_scalar"], ["id_ge_zero"]))
    nodes.append(helper.make_node("Less", ["ids_trunc", "max_input_id_plus_1"], ["id_lt_sep"]))
    nodes.append(helper.make_node("And", ["id_ge_zero", "id_lt_sep"], ["id_in_range"]))
    nodes.append(
        helper.make_node("Where", ["id_in_range", "ids_trunc", "vocab_depth"], ["ids_guarded"])
    )

    # --- N・T の取得（動的軸。T は Slice 後の値） ---
    nodes.append(helper.make_node("Shape", ["ids_trunc"], ["shape_ids"]))
    nodes.append(helper.make_node("Gather", ["shape_ids", "idx0"], ["n_vec"], axis=0))
    nodes.append(helper.make_node("Gather", ["shape_ids", "idx1"], ["t_vec"], axis=0))

    # --- `N`（バッチ件数）上限ガード（REQ-39・オーナー判断 2026-09-28・
    # PR #222 レビュー指摘）。`n_guard_table`（長さ `n_max + 1`。
    # `_ar_export_max_batch_n` がこのモデル構成から算出した値を書き出し
    # 時に固定する）を実際の `N`（`n_vec`）で `Gather` する。`N` は
    # `Shape` から得られる非負の次元値であり、`n_max` を超えると
    # `Gather` が範囲外参照で失敗する（負のインデックス wrap の懸念は
    # ここでは生じない。`N` が負になることは無いため）。ガード結果
    # （常に 0）を `n_vec` へ加算して `n_vec_checked` を作り、以降の
    # 形状計算をすべてこちら経由にすることで、グラフ最適化による
    # デッドコード除去でガードが消えない（`n_vec_checked` は最終出力
    # `probs` の形状計算へ直結する依存経路に乗る）。「範囲外 Gather は
    # エラーになる」という前提の根拠・適用範囲は、下の embed_table への
    # `Gather` 直前のコメントへ集約する（Cursor Medium 指摘）。
    nodes.append(helper.make_node("Gather", ["n_guard_table", "n_vec"], ["n_guard_zero"], axis=0))
    nodes.append(helper.make_node("Add", ["n_vec", "n_guard_zero"], ["n_vec_checked"]))

    # --- 派生する形状ベクトル（いずれも 1 要素以上の 1 次元 int64 配列） ---
    nodes.append(helper.make_node("Add", ["t_vec", "m_vec"], ["t_plus_m_vec"]))  # T+M
    nodes.append(helper.make_node("Add", ["t_vec", "one_vec"], ["t_plus_1_vec"]))  # T+1
    nodes.append(helper.make_node("Add", ["t_plus_1_vec", "m_vec"], ["l_vec"]))  # L=T+1+M
    nodes.append(helper.make_node("Squeeze", ["l_vec", "idx0"], ["l_scalar"]))
    nodes.append(
        helper.make_node("Concat", ["n_vec_checked", "k_vec", "t_vec"], ["shape_nkt"], axis=0)
    )
    nodes.append(
        helper.make_node("Concat", ["n_vec_checked", "k_vec", "m_vec"], ["shape_nkm"], axis=0)
    )
    nodes.append(
        helper.make_node("Concat", ["n_vec_checked", "k_vec", "one_vec"], ["shape_nk1"], axis=0)
    )
    nodes.append(helper.make_node("Concat", ["negidx", "l_vec"], ["shape_flat_l"], axis=0))
    nodes.append(helper.make_node("Concat", ["negidx", "m_vec"], ["shape_flat_m"], axis=0))
    nodes.append(helper.make_node("Concat", ["n_vec_checked", "k_vec"], ["shape_nk"], axis=0))

    # --- K 個の選択肢へ展開してから [N*K, L] へ reshape する（§2.3） ---
    nodes.append(helper.make_node("Unsqueeze", ["ids_guarded", "idx1"], ["ids_exp3"]))  # [N,1,T]
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
    # `full_ids_2d` に含まれる外部入力由来の値は、上流の `ids` 値域ガード
    # （`ids_guarded` を参照するコメント参照）で既に `[0, MAX_INPUT_ID]`
    # へ矯正済み（範囲外は `vocab_depth`=VOCAB_SIZE へ書き換え済みのため、
    # この `Gather` が範囲外参照で fail-closed に失敗する）。`full_ids_2d`
    # にはグラフが内部で挿入する SEP（257）・選択肢トークン（`choice_flat`
    # 由来。学習時に固定された語彙内の正当な値）も混在するが、これらは
    # 外部入力ではなく検査対象外でよい（REQ-39・オーナー判断
    # 2026-09-28・PR #222 レビュー指摘。以前はここで値域検査を一切行わず
    # 推論ランタイム・ガード層側の将来対応としていたが、グラフ内で
    # fail-closed にする方針へ変更した）。
    #
    # 上記の `ids` 値域ガード・`N` 上限ガードは、いずれも「範囲外
    # インデックスの `Gather` はエラーになる」ことを安全側の前提にしている
    # （Cursor Medium 指摘。この前提の根拠・適用範囲・多層防御としての
    # 位置づけを次の 3 点として明記する）。
    #
    # (a) 根拠: ONNX の `Gather` 演算子仕様（opset 13。opset 1 以来不変）は
    #     "It is an error if any of the index values are out of bounds"
    #     と明記しており、範囲外インデックスの挙動は実装依存の未規定
    #     ではなく仕様違反（エラーにすべき条件）として定義されている。
    #     onnx.reference は実際に `IndexError` を送出する（本ファイルの
    #     `tests/test_ar_ids_range.py` で実測確認済み）。
    # (b) 適用範囲: 配布先の推論ランタイムは CPU 上の ONNX Runtime（`ort`
    #     crate）を前提とし、CPU Execution Provider は上記仕様どおり
    #     エラーを返す（本ワーカーは学習側であり `ort` を持たないため
    #     onnx.reference までが実機未検証の範囲。証拠種別: テスト
    #     ハーネス）。仕様に厳密に従わない Execution Provider（一部の
    #     GPU EP 等、範囲外インデックスを clamp・wrap する可能性がある
    #     実装）への依存は本ガードの前提外とする。
    # (c) 位置づけ: 本ガードはグラフ内での fail-closed な多層防御の
    #     一層であり、これに加えて推論ランタイム・ガード層（REQ-39。
    #     パス未確定）が推論の入口で `ids` の値域（`[0, MAX_INPUT_ID]`
    #     外）と `N`（`n_max` 超過）を `invalid_input` として事前検査
    #     する必要がある（本ガードの `Gather` 失敗は `runtime_error`・
    #     exit 70 にしかならず、機械可読な入力エラーにはならないため）。
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
