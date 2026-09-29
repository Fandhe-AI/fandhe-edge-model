"""kind="c1"（バイト n-gram TF-IDF + 多クラスロジスティック回帰。REQ-19・TASK-19.2）。

C3（`kinds/c3.py`）と並ぶ既定候補のもう一つ。PoC は文字 n-gram + scikit-learn を
使っていたが、本実装は以下の 2 点を意図的に変更している。

1. **入力表現はバイトのみ**（README「入力方針（要点）」・`encoding.py`）。
   文字 n-gram ではなく、C3 と同じトークン化（`encode_bytes`。バイト値 +1、
   0 は詰め物）から得たバイト値の n-gram を特徴量にする。PoC との差異は
   spec 側の Issue（fandhe-edge-model-spec#1）で記録済み。
2. **依存は mlx・numpy・onnx のみ**（scikit-learn・scipy は未承認。
   dependency-policy.md）。TF-IDF の語彙構築・重み計算・L2 正規化・
   ロジスティック回帰の学習をすべて numpy・MLX で自前実装し、ONNX 側も
   `TfIdfVectorizer` 演算子を手組みのグラフへ組み込む。

**詰め物トークン（id=0）を含む n-gram は語彙に絶対に入れない**（C3 の
docstring 2〜3 番と同種の理由）。ONNX の `TfIdfVectorizer` は入力行の
全長（バッチ内で最も長い系列に合わせて 0 で埋められた列も含む）に対して
n-gram 照合を行うため、語彙が id=0 を含む n-gram を 1 つでも持つと、
その n-gram がバッチ内の他系列由来の詰め物と単独系列末尾の詰め物とで
異なる形で照合されうる（REQ-28 の一致契約が崩れる）。本実装は語彙構築
段階で入力行の実長（詰め物より前の部分）だけから n-gram を抽出することで
そもそも id=0 を含む n-gram を生成しない。書き出し直前にも pool に 0 が
無いことを検証し、違反があれば fail-closed で拒否する。

**TF-IDF の重み付けは ONNX 標準の `mode="TFIDF"` を使わない**。標準モードは
`tf * idf`（線形）のみで、sklearn 互換の sublinear TF（`1 + ln(tf)`）を
表現できないため、`mode="TF"`（生カウント）の出力に対して sublinear
変換・idf 乗算・L2 正規化を後続ノードとして手組みする（本モジュールの
docstring 末尾・`_export_c1_onnx` 参照）。学習側の numpy 特徴量計算
（`_build_sparse_features`）もまったく同じ式を使う（数値一致がテスト
`tests/test_c1_mlx_onnx_parity.py` の前提）。

**`TfIdfVectorizer` の `ngram_counts`/`pool_int64s` レイアウト**: ONNX の
リファレンス実装（`onnx/reference/ops/op_tfidf_vectorizer.py`）を確認した
結果、`ngram_counts[i]` は「長さ `i+1` の n-gram が `pool_int64s` 内で
始まる要素位置」を表し、**サイズ 1 から `ngram_max` までを隙間なく数える**
（`min_gram_length` はどのサイズを実際に照合対象にするかを決めるだけで、
`ngram_counts` の添字とサイズの対応 `i -> i+1` そのものはずらさない）。
そのため `ngram_min > 1` の場合、`ngram_counts` の要素数は
`ngram_max`（`ngram_max - ngram_min + 1` ではない）とし、`ngram_min` 未満の
サイズには要素数 0（開始位置を進めない）を割り当てる。この挙動は
`onnx.reference.ReferenceEvaluator` で実際に検証済み（`tests/test_c1_batch_parity.py`・
`tests/test_c1_mlx_onnx_parity.py` が n-gram レンジ 1..1 以外のケースでも
一致することを確認する）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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
    MAX_C1_BATCH_ELEMENTS,
    MAX_C1_BATCH_SIZE,
    MAX_C1_C,
    MAX_C1_EPOCHS,
    MAX_C1_LR,
    MAX_C1_MAX_FEATURES,
    MAX_C1_MIN_DF,
    MAX_C1_NGRAM,
    MAX_C1_SPARSE_ELEMENTS,
    MAX_C1_VOCAB_CANDIDATES,
    MAX_C1_VOCAB_CHUNK_WINDOW_ELEMENTS,
    MIN_C1_C,
    MIN_C1_MIN_DF,
)
from ..prediction import ok_prediction_record

KIND = "c1"
KIND_VERSION = 1

#: L2 正規化でゼロ除算を避けるための下限（`_export_c1_onnx` の `denom` ノード・
#: `_build_sparse_features` の学習側 L2 正規化の両方で同じ値を使う。
#: ゼロベクトルはゼロのまま保つという評価契約の要請
#: 〔evaluation-contract.md〕を、割り算経路でも壊さない程度に小さい値で
#: あれば具体的な値そのものに意味は無い）。
_L2_EPS = 1e-12

#: 既定ハイパーパラメータ。`ngram_max` はバイト表現で日本語 1 文字 ≈ 3 バイトに
#: 相当する範囲を狙う（PoC の文字 n-gram (1,3)/(2,4) に近い実効的な文脈幅）。
#: `epochs`・`batch_size`・`lr` は本ワーカーの合成データ（`tests/conftest.py`）で
#: 収束することを確認した値。
DEFAULT_CONFIG: dict[str, Any] = {
    "ngram_min": 1,
    "ngram_max": 4,
    "min_df": 2,
    "max_features": 200_000,
    "C": 1.0,
    "epochs": 30,
    "batch_size": 64,
    "lr": 0.5,
}

_CONFIG_FIELDS = set(DEFAULT_CONFIG)


def _validate_config(cfg: dict[str, Any]) -> None:
    """`request.config` によるハイパーパラメータ上書きを検証する（C3 の
    `_validate_config` と同じ理由・同じパターン。TASK-19.1 の入口）。

    語彙（n-gram の値そのもの）はデータ由来であり、ここでは検証しない
    （検証対象はあくまでハイパーパラメータ）。
    """
    unknown = set(cfg) - _CONFIG_FIELDS
    if unknown:
        raise WorkerError(
            "invalid_config",
            f"unknown config fields: {truncate_list_for_message(sorted(unknown))}",
            ExitCode.INVALID_INPUT,
        )

    def _bounded_int(name: str, lo: int, upper: int) -> None:
        v = cfg[name]
        if not isinstance(v, int) or isinstance(v, bool) or not (lo <= v <= upper):
            raise WorkerError(
                "invalid_config",
                f"config.{name} must be an integer in [{lo}, {upper}]",
                ExitCode.INVALID_INPUT,
            )

    _bounded_int("ngram_min", 1, MAX_C1_NGRAM)
    _bounded_int("ngram_max", 1, MAX_C1_NGRAM)
    if cfg["ngram_min"] > cfg["ngram_max"]:
        raise WorkerError(
            "invalid_config", "config.ngram_min must be <= config.ngram_max", ExitCode.INVALID_INPUT
        )
    _bounded_int("min_df", MIN_C1_MIN_DF, MAX_C1_MIN_DF)
    _bounded_int("max_features", 1, MAX_C1_MAX_FEATURES)
    _bounded_int("epochs", 1, MAX_C1_EPOCHS)
    _bounded_int("batch_size", 1, MAX_C1_BATCH_SIZE)

    def _finite_number(
        name: str, lo: float, hi: float, *, lo_inclusive: bool, hi_inclusive: bool
    ) -> None:
        v = cfg[name]
        # NaN は範囲比較（`<=`・`<`）が常に False になるため、math.isfinite で
        # 明示的に弾く（C3 の同名ヘルパーと同じ理由）。
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

    _finite_number("C", MIN_C1_C, MAX_C1_C, lo_inclusive=True, hi_inclusive=True)
    _finite_number("lr", 0.0, MAX_C1_LR, lo_inclusive=False, hi_inclusive=True)


def _base257_powers(n: int) -> np.ndarray:
    """base-257 キー化の桁の重み `[257^(n-1), ..., 257^1, 257^0]`（int64）。

    トークン値は 1..256（`encoding.encode_bytes` の規約）なので、257 進数の
    各桁として扱える（0 は詰め物専用で、この基数変換には現れない前提。
    呼び出し側が詰め物を含む窓を渡さないことで保証する）。
    """
    return (257 ** np.arange(n - 1, -1, -1)).astype(np.int64)


def _ngram_keys(row: np.ndarray, doc_len: int, n: int) -> np.ndarray:
    """1 文書分のトークン列（先頭 `doc_len` 個だけが実データ。残りは詰め物）から、
    長さ `n` の n-gram を base-257 の int64 キー列へ変換する（重複を含む）。

    `doc_len` より後ろ（詰め物 id=0 の領域）は一切参照しないため、id=0 を
    含む n-gram は生成されない（クラス docstring の不変条件）。呼び出し側
    （`_encode_examples`）が `doc_len` を「実データの長さ（0 個の場合を含む）」
    として正しく計算していることが前提だが、`row[:doc_len]` 自体に想定外の
    0 が混入していた場合（呼び出し側のバグ・将来の変更）に備え、0 を含む窓は
    ここでも二重に除外する（多層防御。REQ-28 の一致契約に関わる不変条件のため）。
    """
    if doc_len < n:
        return np.empty(0, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(row[:doc_len], n)
    valid = ~np.any(windows == 0, axis=-1)
    if not np.all(valid):
        windows = windows[valid]
    return windows.astype(np.int64) @ _base257_powers(n)


def _decode_ngram_keys(keys: np.ndarray, n: int) -> np.ndarray:
    """base-257 キー列を、長さ `n` のトークン列（形状 `(len(keys), n)`）へ復号する。

    `_ngram_keys` の逆変換（ONNX の `pool_int64s` へ実際のトークン列を
    書き出すために使う。`TfIdfVectorizer` は元のトークン列そのものを
    `pool_int64s` として要求し、base-257 キーは学習側の内部表現に過ぎない）。
    """
    digits = np.empty((len(keys), n), dtype=np.int64)
    remainder = keys.astype(np.int64).copy()
    for i in range(n - 1, -1, -1):
        power = 257**i
        digits[:, n - 1 - i] = remainder // power
        remainder = remainder % power
    return digits


@dataclass(frozen=True)
class _Vocabulary:
    """学習データから構築した語彙（列順は `n` 昇順・キー昇順で固定）。

    データ由来の情報（`ns`・`keys`）は artifact の `config` へは一切出さない
    （クラス docstring・`C1TrainedModel` 参照）。
    """

    ns: np.ndarray  # (F,) int32。各語彙列の n
    keys: np.ndarray  # (F,) int64。各語彙列の base-257 キー（n 内で昇順）
    idf: np.ndarray  # (F,) float32。smooth idf


#: `(n, base257 key)` の対を 1 本の int64 へ束ねる基数（`_build_vocabulary` が
#: 文書頻度を数える際の「結合キー」に使う）。`MAX_C1_NGRAM` 桁分の base-257
#: キー空間は `257 ** MAX_C1_NGRAM ≈ 1.2e17` に収まり、`n`（最大 `MAX_C1_NGRAM`
#: = 7）を掛けても `7 * 1.2e17 ≈ 8.4e17` で int64 の範囲（最大 ≈ 9.2e18）に
#: 十分収まる。`n` に依存せず固定値にすることで、設定値（`ngram_max`）が
#: 変わっても結合キーの意味がぶれない。
_COMBINE_BASE = 257**MAX_C1_NGRAM

# セキュリティ監査 P2-2: 結合キー（`n * _COMBINE_BASE + key`）が int64 の範囲へ
# 収まることをモジュール読み込み時に検査する（`assert` は `-O` 実行で無効化
# されうるため使わない。明示的な `if` + `raise` にする）。最大の結合キーは
# `n=MAX_C1_NGRAM`・`key=_COMBINE_BASE-1` のときの
# `MAX_C1_NGRAM * _COMBINE_BASE + (_COMBINE_BASE - 1) < (MAX_C1_NGRAM + 1) *
# _COMBINE_BASE` で抑えられる。`MAX_C1_NGRAM`（limits.py）を変更する際は、
# この不変条件が壊れていないかをここで即座に検出する。
if (MAX_C1_NGRAM + 1) * _COMBINE_BASE >= 2**63:
    raise RuntimeError(
        "internal invariant violated: MAX_C1_NGRAM is too large for int64 combined "
        "vocabulary keys (n * 257**MAX_C1_NGRAM + key would overflow)"
    )


def _build_vocabulary(
    ids_arr: np.ndarray,
    doc_lens: np.ndarray,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
    max_features: int,
    resource_budget: budget_mod.ResourceBudget,
) -> _Vocabulary:
    """学習データ全体から語彙（列順固定）を構築する（1 パス目: 文書頻度の計数）。

    numpy で完結させる（Python dict のキーごとのループは使わない）: 1 文書分の
    n-gram 抽出・文書内の重複除去（`np.unique`。文書頻度は「その n-gram を
    含む文書数」であり文書内の出現回数ではないため、文書ごとに一意化してから
    数える）は文書ごとの `numpy` 呼び出しで行うが、文書頻度の集計そのものは
    チャンク単位で `np.concatenate` → `np.unique(..., return_counts=True)` →
    既存の集計と `np.unique(..., return_inverse=True)` で再マージする形で
    行い、相異なる `(n, key)` の対 1 つごとに Python の辞書操作を挟まない。

    メモリの最悪値の見積もり（セキュリティ監査 P2-1: チャンクサイズを固定値
    にせず、1 チャンクの一時配列サイズが確保前に見積もった上限を超えないよう
    動的に決める）: 1 文書あたりの候補 n-gram 数（窓の総数）は `ids_arr` の
    実際の列幅 `max_len`（`request.max_bytes` 以下）を用いて
    `max_len * (ngram_max - ngram_min + 1)` で頭打ちにできる（各サイズごとに
    高々 `max_len` 個の窓）。1 チャンクあたりの窓要素数の目標上限
    `MAX_C1_VOCAB_CHUNK_WINDOW_ELEMENTS` をこの 1 文書あたり上限で割った件数を
    `chunk_size`（1 チャンクに含める文書数）とすることで、1 チャンク分の
    一時配列（`doc_key_arrays`・`chunk_all`・`chunk_keys`/`chunk_counts`）の
    合計要素数を `MAX_C1_VOCAB_CHUNK_WINDOW_ELEMENTS` 程度（int64 換算で
    既定値約 32 MiB）に抑える。集計後に保持する「これまでに見た相異なる
    `(n, key)`」配列（`running_keys`/`running_counts`）は
    `MAX_C1_VOCAB_CANDIDATES` 件を超えたら打ち切る（REQ-39。1 件あたり
    int64 2 本 = 16 bytes なので上限でも高々 1 GiB 未満）。

    文書を `chunk_size` 件ごとのチャンクに分けて処理し、チャンク間で
    `resource_budget.check()`（壁時計・RSS）を呼ぶ。
    """
    n_docs = ids_arr.shape[0]
    max_len = int(ids_arr.shape[1]) if ids_arr.ndim == 2 and ids_arr.shape[1] > 0 else 1
    ngram_span = max(1, ngram_max - ngram_min + 1)
    per_doc_window_upper_bound = max(1, max_len * ngram_span)
    chunk_size = max(1, MAX_C1_VOCAB_CHUNK_WINDOW_ELEMENTS // per_doc_window_upper_bound)
    running_keys = np.empty(0, dtype=np.int64)
    running_counts = np.empty(0, dtype=np.int64)
    for chunk_start in range(0, n_docs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_docs)
        doc_key_arrays: list[np.ndarray] = []
        for i in range(chunk_start, chunk_end):
            doc_len = int(doc_lens[i])
            row = ids_arr[i]
            per_n_unique: list[np.ndarray] = []
            for n in range(ngram_min, ngram_max + 1):
                keys = _ngram_keys(row, doc_len, n)
                if keys.size == 0:
                    continue
                combined = n * _COMBINE_BASE + keys
                per_n_unique.append(np.unique(combined))
            if per_n_unique:
                doc_key_arrays.append(np.concatenate(per_n_unique))
        if doc_key_arrays:
            chunk_all = np.concatenate(doc_key_arrays)
            chunk_keys, chunk_counts = np.unique(chunk_all, return_counts=True)
            merged_keys = np.concatenate([running_keys, chunk_keys])
            merged_counts = np.concatenate([running_counts, chunk_counts])
            running_keys, inverse = np.unique(merged_keys, return_inverse=True)
            running_counts = np.zeros(running_keys.shape[0], dtype=np.int64)
            np.add.at(running_counts, inverse, merged_counts)
            if running_keys.size > MAX_C1_VOCAB_CANDIDATES:
                raise WorkerError(
                    "limit_exceeded",
                    f"vocabulary candidate count exceeds {MAX_C1_VOCAB_CANDIDATES}",
                    ExitCode.LIMIT_EXCEEDED,
                )
        resource_budget.check()

    keep_mask = running_counts >= min_df
    kept_keys = running_keys[keep_mask]
    kept_dfs = running_counts[keep_mask]
    if kept_keys.size == 0:
        raise WorkerError(
            "invalid_data",
            "vocabulary is empty after min_df filtering",
            ExitCode.INVALID_INPUT,
        )
    kept_ns = (kept_keys // _COMBINE_BASE).astype(np.int32)
    kept_raw_keys = (kept_keys % _COMBINE_BASE).astype(np.int64)

    # 上限超過時は (df 降順, n 昇順, key 昇順) の決定的順で上位 max_features 件を残す。
    # np.lexsort はキーの末尾要素を最優先（第一キー）として扱う。
    top_order = np.lexsort((kept_raw_keys, kept_ns, -kept_dfs))
    if top_order.size > max_features:
        top_order = top_order[:max_features]
    sel_ns = kept_ns[top_order]
    sel_keys = kept_raw_keys[top_order]
    sel_dfs = kept_dfs[top_order]

    # 最終列順は (n 昇順, key 昇順) に固定する（ONNX 側の pool レイアウトと一致させる）。
    final_order = np.lexsort((sel_keys, sel_ns))
    ns = sel_ns[final_order]
    keys = sel_keys[final_order]
    dfs = sel_dfs[final_order]
    # sklearn 互換の smooth idf: ln((1+N)/(1+df)) + 1（float32）。
    idf = (np.log((1.0 + n_docs) / (1.0 + dfs.astype(np.float64))) + 1.0).astype(np.float32)
    return _Vocabulary(ns=ns, keys=keys, idf=idf)


def _build_sparse_features(
    ids_arr: np.ndarray,
    doc_lens: np.ndarray,
    vocab: _Vocabulary,
    ngram_min: int,
    ngram_max: int,
    max_nnz: int,
    resource_budget: budget_mod.ResourceBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """語彙確定後（2 パス目）、各文書の疎な TF-IDF 特徴量を固定幅へ詰め物して返す。

    戻り値は `(idx_arr, val_arr)`（形状いずれも `(n_docs, max_nnz)`）。
    `val_arr` は sublinear TF（`1 + ln(tf)`。tf=0 は 0）× idf → L2 正規化済みの
    float32（ONNX 側 `_export_c1_onnx` のグラフと同じ式）。詰め物位置は
    `idx=0, val=0.0`（値 0 が寄与を消すため、列 0 が実在の語彙列であっても
    干渉しない。`kinds/c1.py::TfidfLogReg.__call__` 参照）。

    列インデックスの検索は `n` ごとに語彙をソート済み配列として保持し
    `np.searchsorted` で行う（Python dict のループを避ける。C3 の
    `_encode_examples` と同様、あらかじめ確保した固定幅の配列へ行ごとに
    書き込む設計〔P0-1〕を踏襲する）。
    """
    group_keys: dict[int, np.ndarray] = {}
    group_col_start: dict[int, int] = {}
    col = 0
    for n in range(ngram_min, ngram_max + 1):
        mask = vocab.ns == n
        group_keys[n] = vocab.keys[mask]
        group_col_start[n] = col
        col += int(np.sum(mask))

    n_docs = ids_arr.shape[0]
    idx_arr = np.zeros((n_docs, max_nnz), dtype=np.int32)
    val_arr = np.zeros((n_docs, max_nnz), dtype=np.float32)

    for i in range(n_docs):
        doc_len = int(doc_lens[i])
        row = ids_arr[i]
        matched_cols: list[np.ndarray] = []
        for n in range(ngram_min, ngram_max + 1):
            keys_n = group_keys[n]
            if keys_n.size == 0:
                continue
            keys = _ngram_keys(row, doc_len, n)
            if keys.size == 0:
                continue
            pos = np.searchsorted(keys_n, keys)
            pos_clipped = np.clip(pos, 0, keys_n.size - 1)
            hit = keys_n[pos_clipped] == keys
            if not np.any(hit):
                continue
            matched_cols.append(group_col_start[n] + pos_clipped[hit])

        if matched_cols:
            all_cols = np.concatenate(matched_cols)
            uniq_cols, tf_counts = np.unique(all_cols, return_counts=True)
            tf = tf_counts.astype(np.float32)
            sublinear = 1.0 + np.log(tf)  # tf は np.unique の件数なので常に >=1
            weighted = sublinear * vocab.idf[uniq_cols]
            norm = float(np.sqrt(np.sum(weighted * weighted)))
            denom = max(norm, _L2_EPS)
            normalized = weighted / denom
            nnz = uniq_cols.size
            idx_arr[i, :nnz] = uniq_cols
            val_arr[i, :nnz] = normalized

        if (i + 1) % 1024 == 0:
            resource_budget.check()
    resource_budget.check()  # 端数分の確認
    return idx_arr, val_arr


class TfidfLogReg(nn.Module):
    """疎な TF-IDF 特徴量に対する多クラスロジスティック回帰（softmax）。

    `weight`（F×K）・`bias`（K,）はゼロ初期化する。損失（交差エントロピー +
    L2 正則化）が凸なため対称性の破れが不要で、ゼロ初期化のままでも学習が
    進む（決定性の観点でも、乱数初期化を挟まない方が単純）。
    """

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.weight = mx.zeros((n_features, n_classes), dtype=mx.float32)
        self.bias = mx.zeros((n_classes,), dtype=mx.float32)

    def __call__(self, idx: mx.array, val: mx.array) -> mx.array:
        """`idx`・`val`: 形状 `(B, nnz)`。詰め物位置（`val=0.0`）は
        どの `idx` を引いても寄与がゼロになる（`_build_sparse_features` 参照）。
        """
        gathered = mx.take(self.weight, idx, axis=0)  # (B, nnz, K)
        weighted = gathered * val[..., None]
        return mx.sum(weighted, axis=1) + self.bias


def _encode_examples(
    examples: list[TrainExample], max_bytes: int, resource_budget: budget_mod.ResourceBudget
) -> tuple[np.ndarray, np.ndarray]:
    """学習データ全体を固定幅 int32 配列へエンコードし、各行の実長も返す
    （`_encode_texts` へ入力本文だけを渡す。ラベルは使わない）。
    """
    return _encode_texts([ex.input for ex in examples], max_bytes, resource_budget)


def _encode_texts(
    texts: Sequence[str], max_bytes: int, resource_budget: budget_mod.ResourceBudget
) -> tuple[np.ndarray, np.ndarray]:
    """入力本文の列を固定幅 int32 配列へエンコードし、各行の実長も返す
    （学習データ・学習直後の validation 予測〔`predict_labels`〕で共有する）。

    C3 の `_encode_examples`（P0-1: あらかじめ確保した配列へ行ごとに書き込む
    設計）と同じ理由・同じパターン。C1 は詰め物より前の実長（`doc_lens`）を
    語彙構築・特徴量抽出（`_build_vocabulary`・`_build_sparse_features`）で
    直接使うため、C3 の `_encode_examples` を再利用せずここで独立に持つ
    （戻り値の形が異なる。C3 側の実装は改変しない）。

    **`doc_lens` は「実データ（詰め物ではない）トークンの個数」を表す**。
    `encode_bytes` は正規化後に空となる入力（本来 `contract.load_train_examples`
    で拒否されるが、本関数はそれより下位の層でありその前提に頼らない）に対し
    詰め物専用の `[0]`（長さ 1・値 0）を返す（`encoding.py` の docstring 参照）。
    これは「実データが 1 個ある」ことを意味しないため、`row == [0]` の場合は
    実長を 0 として扱う。ここを `len(row)` のまま扱うと、後続の n-gram 抽出
    （`_ngram_keys`）が `row[:doc_len]` に詰め物専用の 0 を含めてしまい、
    語彙が id=0 を含む n-gram を持ちうる（クラス docstring の不変条件が
    崩れる。REQ-28 の一致契約に関わる）。
    """
    n = len(texts)
    arr = np.zeros((n, max_bytes), dtype=np.int32)
    lens = np.zeros((n,), dtype=np.int32)
    max_len = 1
    for i, text in enumerate(texts):
        row = encode_bytes(text, max_bytes)
        length = len(row)
        arr[i, :length] = row
        lens[i] = 0 if (length == 1 and row[0] == 0) else length
        if length > max_len:
            max_len = length
        if (i + 1) % 1024 == 0:
            resource_budget.check()
    resource_budget.check()
    return arr[:, :max_len], lens


def _c1_model_bytes_estimate(pool_len: int, n_features: int, n_classes: int) -> int:
    """ONNX 書き出しモデルのバイト数を、語彙・パラメータの形状から見積もる
    （`budget_mod.check_model_bytes` は float32 換算〔4 bytes/要素〕の
    `param_count` を受け取る契約のため、int64 領域〔`pool_int64s`・
    `ngram_indexes`〕も 4 bytes 単位に換算した「見積もり要素数」を返す。
    切り上げにより実際のバイト数を下回らないようにする〔安全側〕）。

    内訳: `pool_int64s`（`pool_len` 個 × 8 bytes）・`ngram_indexes`
    （`n_features` 個 × 8 bytes）・`idf`（`n_features` 個 × 4 bytes）・
    `weight`（`n_features * n_classes` 個 × 4 bytes）・`bias`
    （`n_classes` 個 × 4 bytes）。
    """
    total_bytes = (
        pool_len * 8 + n_features * 8 + n_features * 4 + n_features * n_classes * 4 + n_classes * 4
    )
    return -(-total_bytes // 4)  # ceil(total_bytes / 4)


@dataclass(frozen=True)
class C1TrainedModel:
    """C1 の学習結果（ONNX 書き出しに要る情報一式）。

    `config`・`label_order`・`max_bytes` は `kinds/__init__.py::TrainedModel`
    プロトコルが宣言する共通フィールド。`vocab` はデータ由来の情報のため
    `config` には含めない（クラス docstring の不変条件。artifact.json の
    `config` フィールドへ n-gram の実値が漏れないようにする）。
    """

    model: TfidfLogReg
    vocab: _Vocabulary
    config: dict[str, Any] = field(default_factory=dict)
    label_order: list[str] = field(default_factory=list)
    max_bytes: int = 512
    seed: int = 0
    epochs_run: int = 0
    resource_budget: budget_mod.ResourceBudget | None = None


class C1Kind:
    """選択口（`kinds/__init__.py`）が呼ぶ C1 の学習・書き出し実装。"""

    def train(
        self,
        examples: list[TrainExample],
        request: TrainRequest,
        resource_budget: budget_mod.ResourceBudget,
    ) -> C1TrainedModel:
        """呼び出し元（`cli.py`）が生成した `resource_budget` をそのまま使う
        （C3 と同じ契約。学習データの読み込み・語彙構築・学習・書き出しを
        1 つの予算として扱う）。
        """
        cfg = {**DEFAULT_CONFIG, **request.config}
        _validate_config(cfg)
        ngram_min = int(cfg["ngram_min"])
        ngram_max = int(cfg["ngram_max"])
        min_df = int(cfg["min_df"])
        max_features = int(cfg["max_features"])
        c_value = float(cfg["C"])
        epochs = int(cfg["epochs"])
        batch_size = int(cfg["batch_size"])

        label_order = request.label_order
        label_id = {label: i for i, label in enumerate(label_order)}
        n_classes = len(label_order)
        n_examples = len(examples)

        # P0-B: 学習開始前に総ステップ数（examples 件数 x epochs）の見積もりで
        # 上限を検査する（REQ-39。C3 と同じ理由・同じ呼び出し順）。
        budget_mod.check_sample_steps(n_examples, epochs)
        # P0-1: エンコード前に総トークン数（examples 件数 x max_bytes）の見積もりで
        # 上限を検査する。
        budget_mod.check_total_tokens(n_examples, request.max_bytes)

        mx.set_default_device(mx.cpu if request.device == "cpu" else mx.gpu)
        mx.random.seed(request.seed)
        rng = np.random.default_rng(request.seed)

        ids_arr, doc_lens = _encode_examples(examples, request.max_bytes, resource_budget)
        vocab = _build_vocabulary(
            ids_arr, doc_lens, ngram_min, ngram_max, min_df, max_features, resource_budget
        )
        n_features = vocab.ns.shape[0]

        # 語彙構築直後（モデルサイズが確定した時点）でモデルサイズの上限を検査する
        # （P0-2 の考え方の準用。`ByteCNN` を構築する前に検査する C3 と同じ順序）。
        pool_len = int(np.sum(vocab.ns))  # 各語彙エントリが pool へ n 要素ずつ書き込む
        budget_mod.check_model_bytes(_c1_model_bytes_estimate(pool_len, n_features, n_classes))

        # 疎特徴量を固定幅へ詰め物する前に、確保する配列の総要素数を検査する
        # （P0-1 の考え方の準用。文書 1 件あたりの上限 nnz は「各 n の窓の数の和」
        # で頭打ちにできる: doc_len 以下 x n の種類数）。
        max_nnz = min(n_features, request.max_bytes * (ngram_max - ngram_min + 1))
        max_nnz = max(max_nnz, 1)
        sparse_elements = n_examples * max_nnz
        if sparse_elements > MAX_C1_SPARSE_ELEMENTS:
            raise WorkerError(
                "limit_exceeded",
                f"estimated sparse feature elements {sparse_elements} exceeds limit "
                f"{MAX_C1_SPARSE_ELEMENTS} (examples x max_nnz)",
                ExitCode.LIMIT_EXCEEDED,
            )
        # セキュリティ監査 P0: ミニバッチのフォワード（`TfidfLogReg.__call__` の
        # `mx.take` による gather `(B, nnz, K)` とそれに続く要素積）が確保する
        # 要素数を、実際に 1 バッチも確保する前に見積もって検査する。
        # `MAX_C1_SPARSE_ELEMENTS`（学習データ全体の疎表現）を満たしていても、
        # `batch_size × max_nnz × n_classes` の積は別の変数の掛け算で決まる
        # ため独立に跳ね上がりうる（`limits.py::MAX_C1_BATCH_ELEMENTS` の
        # コメント参照）。
        batch_elements = batch_size * max_nnz * n_classes
        if batch_elements > MAX_C1_BATCH_ELEMENTS:
            raise WorkerError(
                "limit_exceeded",
                f"estimated mini-batch activation elements {batch_elements} exceeds limit "
                f"{MAX_C1_BATCH_ELEMENTS} (batch_size x max_nnz x n_classes)",
                ExitCode.LIMIT_EXCEEDED,
            )
        idx_arr, val_arr = _build_sparse_features(
            ids_arr, doc_lens, vocab, ngram_min, ngram_max, max_nnz, resource_budget
        )
        labels_list: list[int] = []
        for ex in examples:
            label_idx = label_id.get(ex.label)
            if label_idx is None:
                # P2-3: `label_id[ex.label]` の素の KeyError は exit 70
                # （runtime_error）に化けてしまう。呼び出し元（`contract.
                # load_train_examples`）は通常ここで既に弾いているはずだが、
                # 本関数はその前提に頼らず、データ由来の不整合として
                # invalid_data（exit 64）で明示的に拒否する（ラベル本文は
                # メッセージへ含めない。security.md）。
                raise WorkerError(
                    "invalid_data",
                    "train example label not in label_order",
                    ExitCode.INVALID_INPUT,
                )
            labels_list.append(label_idx)
        labels_arr = np.array(labels_list, dtype=np.int32)

        model = TfidfLogReg(n_features, n_classes)
        mx.eval(model.parameters())
        opt = optim.Adam(learning_rate=cfg["lr"])

        # sklearn 風の正則化強度 C: 平均交差エントロピー損失に対する罰則係数を
        # `1/(C * N) * 0.5 * ||W||^2` とする（切片は罰しない）。C が大きいほど
        # 正則化が弱くなる sklearn の慣習を踏襲する（クラス docstring・
        # モジュール docstring 参照）。
        l2_coef = 0.5 / (c_value * n_examples)
        # C の下限（MIN_C1_C）で有限になるはずだが、設定誤りを発散（training_diverged）
        # と取り違えないよう、学習開始前にも確かめる。
        if not math.isfinite(l2_coef):
            raise WorkerError(
                "invalid_config",
                "config.C yields a non-finite regularization coefficient",
                ExitCode.INVALID_INPUT,
            )

        def loss_fn(mdl: TfidfLogReg, idx: mx.array, val: mx.array, y: mx.array) -> mx.array:
            logits = mdl(idx, val)
            ce = nn.losses.cross_entropy(logits, y, reduction="mean")
            reg = l2_coef * mx.sum(mdl.weight * mdl.weight)
            return ce + reg

        step = nn.value_and_grad(model, loss_fn)
        model.train()
        for _epoch in range(epochs):
            order = rng.permutation(n_examples)
            for start in range(0, len(order), batch_size):
                batch_idx = order[start : start + batch_size]
                batch_val = val_arr[batch_idx]
                # セキュリティ監査 P0(b): fail-closed の上限（`MAX_C1_BATCH_ELEMENTS`）
                # は最悪ケース（`max_nnz` 全体）で検査済みだが、実際のミニバッチ
                # 内で使われている列数（詰め物でない実データの nnz）は多くの
                # 場合それより小さい。詰め物は常に `val=0.0`（`_build_sparse_features`
                # 参照）なので、そのバッチ内の最大実 nnz まで列を切り詰めても
                # 結果は変わらない（切り詰めた列はすべて寄与ゼロだったため）。
                # 無駄な確保・計算を減らす（最悪ケースの fail-closed 判定自体は
                # 緩めない）。
                batch_max_nnz = max(1, int(np.count_nonzero(batch_val, axis=1).max()))
                idx = mx.array(idx_arr[batch_idx, :batch_max_nnz])
                val = mx.array(batch_val[:, :batch_max_nnz])
                y = mx.array(labels_arr[batch_idx])
                loss, grads = step(model, idx, val, y)
                opt.update(model, grads)
                mx.eval(model.parameters(), opt.state, loss)
                if not math.isfinite(float(loss.item())):
                    # C3 の同名チェックと同じ理由・同じ終了コード（PENDING）を使う。
                    raise WorkerError(
                        "training_diverged",
                        "training loss became non-finite",
                        ExitCode.PENDING,
                    )
                resource_budget.check()
        model.eval()

        return C1TrainedModel(
            model=model,
            vocab=vocab,
            config=cfg,
            label_order=list(label_order),
            max_bytes=request.max_bytes,
            seed=request.seed,
            epochs_run=epochs,
            resource_budget=resource_budget,
        )

    def export_onnx(self, trained: C1TrainedModel, out: IO[bytes]) -> None:
        if trained.resource_budget is not None:
            trained.resource_budget.check()
        _export_c1_onnx(trained, out)


def predict_labels(
    trained: C1TrainedModel,
    rows: Sequence[tuple[str, str]],
    resource_budget: budget_mod.ResourceBudget | None = None,
) -> list[dict[str, Any]]:
    """学習直後の validation 予測（REQ-18・REQ-27。`predict.py` から呼ばれる）。

    `rows` は `(id, input)` の列で、**正解ラベルは受け取らない**。学習時と同じ
    特徴量計算（`_encode_texts`・`_build_sparse_features`）と順伝播
    （`TfidfLogReg.__call__`）を通し、argmax のラベルを返す（同点は
    `label_order` の先頭側。numpy の `argmax` は最初の最大値を返す）。戻り値は
    入力と同じ順序・件数の `{id, status:"ok", predicted_label}`。

    チャンクは学習と同じ `batch_size` を、`MAX_C1_BATCH_ELEMENTS`（ミニバッチの
    活性化要素数の上限。学習開始前に検査済みの `batch_size × max_nnz ×
    n_classes`）を超えない範囲へ縮めたもの。チャンクごとに `resource_budget` を
    検査する（REQ-39。省略時は学習で使ったインスタンス）。
    """
    budget = resource_budget if resource_budget is not None else trained.resource_budget
    if budget is None:
        raise WorkerError(
            "runtime_error",
            "prediction requires a resource budget",
            ExitCode.RUNTIME_ERROR,
        )
    cfg = trained.config
    ngram_min = int(cfg["ngram_min"])
    ngram_max = int(cfg["ngram_max"])
    n_features = trained.vocab.ns.shape[0]
    n_classes = len(trained.label_order)
    max_nnz = max(1, min(n_features, trained.max_bytes * (ngram_max - ngram_min + 1)))
    chunk_size = max(1, min(int(cfg["batch_size"]), MAX_C1_BATCH_ELEMENTS // (max_nnz * n_classes)))
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), chunk_size):
        budget.check()
        chunk = rows[start : start + chunk_size]
        ids_arr, doc_lens = _encode_texts([text for _rid, text in chunk], trained.max_bytes, budget)
        idx_arr, val_arr = _build_sparse_features(
            ids_arr, doc_lens, trained.vocab, ngram_min, ngram_max, max_nnz, budget
        )
        batch_max_nnz = max(1, int(np.count_nonzero(val_arr, axis=1).max()))
        logits = trained.model(
            mx.array(idx_arr[:, :batch_max_nnz]), mx.array(val_arr[:, :batch_max_nnz])
        )
        chosen = np.argmax(np.array(logits, dtype=np.float32), axis=1)
        budget.check()
        for (rid, _text), label_index in zip(chunk, chosen, strict=True):
            records.append(ok_prediction_record(rid, trained.label_order[int(label_index)]))
    return records


def _f32(arr: np.ndarray, name: str) -> TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name=name)


def _i64(arr: np.ndarray, name: str) -> TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.int64), name=name)


def _vocab_tfidf_layout(
    vocab: _Vocabulary, ngram_min: int, ngram_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """語彙から ONNX `TfIdfVectorizer` の `ngram_counts`（長さ `ngram_max`）・
    `pool_int64s`（データ由来の生トークン列）を構築する（モジュール
    docstring のレイアウト説明を参照）。

    書き出し（`_export_c1_onnx`）と、そのレイアウトが正しいことを独立に
    確認する golden テスト（`tests/test_c1_train.py`）の両方から使う
    （書き出しと同じロジックを 2 重実装しない）。
    """
    ngram_counts = np.zeros(ngram_max, dtype=np.int64)
    pool_chunks: list[np.ndarray] = []
    cumulative = 0
    for n in range(1, ngram_max + 1):
        ngram_counts[n - 1] = cumulative
        if n < ngram_min:
            continue
        mask = vocab.ns == n
        keys_n = vocab.keys[mask]
        if keys_n.size == 0:
            continue
        digits = _decode_ngram_keys(keys_n, n)  # (count, n)。列順は語彙と同じ (n, key) 順
        if np.any(digits == 0):
            # クラス docstring の不変条件（詰め物 id=0 を含む n-gram を語彙に
            # 入れない）が崩れている。バッチ推論と単独推論の一致契約
            # （REQ-28）を保てないため書き出しを拒否する（fail-closed）。
            raise WorkerError(
                "runtime_error",
                "internal invariant violated: vocabulary contains a pad token (id=0)",
                ExitCode.RUNTIME_ERROR,
            )
        pool_chunks.append(digits.reshape(-1))
        cumulative += int(digits.size)
    pool_int64s = (
        np.concatenate(pool_chunks) if pool_chunks else np.empty(0, dtype=np.int64)
    ).astype(np.int64)
    return ngram_counts, pool_int64s


def _export_c1_onnx(trained: C1TrainedModel, out: IO[bytes]) -> None:
    """C1 の学習結果を ONNX グラフ（`TfIdfVectorizer` + sublinear/idf/L2 正規化 +
    `Gemm` + `Softmax`）へ書き出す。モジュール docstring・`kinds/__init__.py::
    Kind.export_onnx` の契約（経路は扱わずバイト列を書き込むだけ）に従う。
    """
    budget_check = trained.resource_budget.check if trained.resource_budget is not None else None

    cfg = trained.config
    ngram_min = int(cfg["ngram_min"])
    ngram_max = int(cfg["ngram_max"])
    vocab = trained.vocab
    n_features = vocab.ns.shape[0]
    n_classes = len(trained.label_order)

    params = trained.model.parameters()
    weight = np.array(params["weight"], dtype=np.float32)  # (F, K)
    bias = np.array(params["bias"], dtype=np.float32)  # (K,)
    if budget_check is not None:
        budget_check()  # 段 1: パラメータ抽出後

    ngram_counts, pool_int64s = _vocab_tfidf_layout(vocab, ngram_min, ngram_max)
    if budget_check is not None:
        budget_check()  # 段 2: 語彙のレイアウト構築後

    ids = helper.make_tensor_value_info("ids", TensorProto.INT64, ["N", "T"])
    probs_out = helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["N", n_classes])

    initializers = [
        _f32(vocab.idf, "idf"),
        _f32(np.array(1.0), "one_f32"),
        _f32(np.array(0.0), "zero_f32"),
        _f32(np.array(_L2_EPS), "eps_f32"),
        _i64(np.array([1]), "reduce_axes_1"),
        _f32(weight, "weight"),
        _f32(bias, "bias"),
    ]
    nodes = [
        helper.make_node(
            "TfIdfVectorizer",
            ["ids"],
            ["tf"],
            mode="TF",
            min_gram_length=ngram_min,
            max_gram_length=ngram_max,
            max_skip_count=0,
            ngram_counts=[int(v) for v in ngram_counts],
            ngram_indexes=list(range(n_features)),
            pool_int64s=[int(v) for v in pool_int64s],
        ),
        # sublinear TF: tf=0 -> 0、tf>0 -> 1 + ln(tf)。
        # Log(0) を作らないよう Max(tf, 1.0) してから Log を取り、
        # tf=0 の寄与は Greater(tf, 0) の 0/1 判定で別途足し戻す。
        helper.make_node("Max", ["tf", "one_f32"], ["tf_clip"]),
        helper.make_node("Log", ["tf_clip"], ["log_tf"]),
        helper.make_node("Greater", ["tf", "zero_f32"], ["presence_bool"]),
        helper.make_node("Cast", ["presence_bool"], ["presence_f"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["log_tf", "presence_f"], ["sublinear"]),
        helper.make_node("Mul", ["sublinear", "idf"], ["weighted"]),
        helper.make_node("Mul", ["weighted", "weighted"], ["sq"]),
        # opset13 の ReduceSum は axes を属性ではなく入力で渡す。
        helper.make_node("ReduceSum", ["sq", "reduce_axes_1"], ["sumsq"], keepdims=1),
        helper.make_node("Sqrt", ["sumsq"], ["norm"]),
        helper.make_node("Max", ["norm", "eps_f32"], ["denom"]),
        helper.make_node("Div", ["weighted", "denom"], ["xnorm"]),
        helper.make_node("Gemm", ["xnorm", "weight", "bias"], ["logits"], alpha=1.0, beta=1.0),
        helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
    ]
    if budget_check is not None:
        budget_check()  # 段 3: グラフ構築後（check_model の前）

    graph = helper.make_graph(
        nodes, "c1_tfidf_logreg", [ids], [probs_out], initializer=initializers
    )
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model_proto.ir_version = 8
    onnx.checker.check_model(model_proto)
    if budget_check is not None:
        budget_check()  # 段 4: check_model 後（save の前）
    # 経路は一切扱わない（C3 の `_export_c3_onnx` と同じ理由。
    # `onnx.save` ではなく直接シリアライズすることで、決定性テスト
    # （byte-identical）が `SerializeToString()` の出力のみへ依存する）。
    out.write(model_proto.SerializeToString())
    if budget_check is not None:
        budget_check()  # 段 5: save 後
