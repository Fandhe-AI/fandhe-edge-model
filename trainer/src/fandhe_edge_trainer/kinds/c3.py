"""kind="c3"（バイト入力の小型 CNN。ByteCNN）。

PoC-10（`docs/spec/03-poc/scratch-classifier/scripts/train_mlx.py`。
`ByteCNN`・`train_config` 相当）の移植。以下の点は元 PoC から意図的に変更している。

1. **train データのみで学習する**（本ワーカーの担当外である評価・選定・作り直し判定は
   TASK-18.x/20.x・評価器 TASK-24.1 の責務）。PoC-10 は validation 正解率で早期終了・
   最良エポック復元・学習率グリッド探索を行っていたが、これは全て validation 依存の
   ため持ち込まない。代わりに固定エポック数（既定 `epochs=40`。PoC-10 の学習率
   グリッドの代表値 `lr=1e-3` を既定値として採用）で最後まで学習し、最終エポックの
   重みをそのまま使う。
2. **ONNX 書き出しがバッチ推論に対応する**（REQ-28: 1 件ずつの推論とバッチ推論が
   全件一致する契約）。PoC-16（`docs/spec/03-poc/core-cli-vertical-slice/`）の診断
   （`scripts/diagnose_batch_dependence.py`。README「成功基準 3」節）で、PoC-10 の
   `ByteCNN.__call__` を使う MLX 側のバッチ推論（`predict_probs`。詰め物位置の
   埋め込みが畳み込みへ混ざる）が、バッチ 1 の推論と実際に食い違うことが確認されて
   いる（2/650 件、最大スコア差 0.2135）。PoC-14
   （`docs/spec/03-poc/pc-inference-performance/scripts/export.py::export_c3_onnx`）は
   バッチ次元を 1 に固定しており、この問題を当時作り込んではいなかった。本実装は
   ONNX 側の入力 `ids` の形状を `["N", "T"]`（バッチ・時間の両方を動的軸）にする
   ため、同種の不一致を新たに作り込まないよう、詰め物トークン（id=0）が最大値
   プーリングへ影響しないよう時間軸方向に `_MASK_NEG_VALUE` を加算してからプーリング
   する。この処理は MLX 側の学習時フォワード（`ByteCNN.__call__`）と ONNX 側の
   両方に実装し、同一入力を単独で渡した場合とパディングして他の系列とバッチ化して
   渡した場合とで確率が一致するようにする（学習ワーカー側のテスト
   `tests/test_c3_batch_parity.py` で検証。**この検証は ONNX 書き出しモデル自体の
   契約を満たすことの確認であり、最終的な推論ランタイム〔Rust 側 `ort` crate〕での
   一致確認は TASK-28（推論ランタイム層）・評価器〔REQ-28〕の責務**。ONNX 書き出しの
   前提条件としてここでも検証する）。
3. **embedding の詰め物行（id=0 に対応する行）を常にゼロへ固定する**。詰め物トークンの
   埋め込みが非ゼロだと、畳み込みが「暗黙のゼロパディング境界」と「バッチ内の
   他系列由来の詰め物」を区別できず、単独推論とバッチ推論の結果が食い違いうる
   （2 で述べた一致契約が崩れる）。学習の各更新後にこの行を明示的にゼロへ戻し、
   書き出し前に厳密ゼロであることを検証する（fail-closed）。
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
    MAX_C3_BATCH_SIZE,
    MAX_C3_EMB,
    MAX_C3_EPOCHS,
    MAX_C3_FILTERS,
    MAX_C3_LR,
    MAX_C3_WEIGHT_DECAY,
    MAX_C3_WIDTH_VALUE,
    MAX_C3_WIDTHS,
    MIN_C3_WIDTHS,
)
from ..prediction import ok_prediction_record

KIND = "c3"
KIND_VERSION = 1
_N_TOKENS = 257  # バイト値 0..255 + 1（詰め物 id=0 用に 1 つ空ける）

#: 詰め物位置をマスクする際に加える負の大きな値。ソフトマックス前の logits へ加算する
#: ため、確率がほぼ 0 になる程度に十分大きい負値であれば具体的な値に意味はない
#: （PoC-10 `ByteCNN.__call__` の `-1e9` を踏襲）。MLX 側フォワード（`ByteCNN.__call__`）と
#: ONNX 側グラフ（`_export_c3_onnx`）の両方がこの 1 箇所を参照することで、値の食い違い
#: によって REQ-28 の一致契約が崩れることを防ぐ。
_MASK_NEG_VALUE = -1e9

#: 既定ハイパーパラメータ。PoC-10 の学習率グリッド {1e-3, 3e-3} の代表値・
#: アーキテクチャ既定値（emb/filters/widths/dropout）をそのまま踏襲する。
#: `epochs`・`batch_size` は PoC-10 の値を踏襲するが、PoC-10 が行っていた
#: validation 依存の早期終了は行わないため固定エポック数として扱う。
DEFAULT_CONFIG: dict[str, Any] = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 40,
    "batch_size": 64,
    "emb": 64,
    "filters": 128,
    "widths": [3, 5, 7],
    "dropout": 0.3,
}


class ByteCNN(nn.Module):
    """バイト入力の小型 CNN（PoC-10 `ByteCNN` の移植。マスク付き最大値プーリング）。"""

    def __init__(
        self, n_classes: int, emb: int, filters: int, widths: tuple[int, ...], dropout: float
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(_N_TOKENS, emb)
        self.convs = [nn.Conv1d(emb, filters, k, padding=k // 2) for k in widths]
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(widths), n_classes)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        h = self.embed(x)  # (B, T, E)
        neg = (1.0 - mask)[..., None] * _MASK_NEG_VALUE
        pooled = [mx.max(nn.relu(c(h)) + neg, axis=1) for c in self.convs]
        return self.out(self.drop(mx.concatenate(pooled, axis=-1)))


_CONFIG_FIELDS = set(DEFAULT_CONFIG)


def _validate_config(cfg: dict[str, Any]) -> None:
    """`request.config` によるハイパーパラメータ上書きを検証する（TASK-19.1 の入口）。

    未検証のまま `int(cfg["epochs"])` 等へ渡すと、型不正が `WorkerError` ではなく
    素の例外（exit 70）に化けて REQ-21 の終了コード対応を崩す。ここで
    `invalid_config`（exit 64）として弾く。上限値は REQ-39 の資源上限の考え方を
    モデル構成の妥当性検証へ準用したもの（`limits.py` の暫定値。学習時間・
    メモリ確保量の頭打ちが目的で、これを超えたからといって必ず資源が尽きるわけ
    ではないが、既定値から大きく外れた構成を無条件には許可しない）。
    カーネル幅の偶数値も、ONNX 側の `pads=[k//2, k//2]` が出力長を +1 させ
    REQ-28 のマスク処理が壊れるため拒否する。
    """
    unknown = set(cfg) - _CONFIG_FIELDS
    if unknown:
        # フィールド名はリクエスト JSON の全体サイズ上限まで利用者が自由に長く
        # できるため、切り詰めてから埋め込む（P1-1）。
        raise WorkerError(
            "invalid_config",
            f"unknown config fields: {truncate_list_for_message(sorted(unknown))}",
            ExitCode.INVALID_INPUT,
        )

    def _bounded_int(name: str, upper: int) -> None:
        v = cfg[name]
        if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= upper):
            raise WorkerError(
                "invalid_config",
                f"config.{name} must be an integer in [1, {upper}]",
                ExitCode.INVALID_INPUT,
            )

    _bounded_int("epochs", MAX_C3_EPOCHS)
    _bounded_int("batch_size", MAX_C3_BATCH_SIZE)
    _bounded_int("emb", MAX_C3_EMB)
    _bounded_int("filters", MAX_C3_FILTERS)

    def _finite_number(
        name: str, lo: float, hi: float, *, lo_inclusive: bool, hi_inclusive: bool
    ) -> None:
        v = cfg[name]
        # math.isfinite は NaN・+-Infinity をすべて拒否する。範囲比較（`<=`・`<`）だけに
        # 頼ると NaN との比較は常に False になるため、上下限のどちらの外側にも
        # 判定されず素通りしてしまう（range チェックだけでは NaN を弾けない）。
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

    _finite_number("lr", 0.0, MAX_C3_LR, lo_inclusive=False, hi_inclusive=True)
    _finite_number("weight_decay", 0.0, MAX_C3_WEIGHT_DECAY, lo_inclusive=True, hi_inclusive=True)
    _finite_number("dropout", 0.0, 1.0, lo_inclusive=True, hi_inclusive=False)

    widths = cfg["widths"]
    if not isinstance(widths, list) or not (MIN_C3_WIDTHS <= len(widths) <= MAX_C3_WIDTHS):
        raise WorkerError(
            "invalid_config",
            f"config.widths must be a list of length [{MIN_C3_WIDTHS}, {MAX_C3_WIDTHS}]",
            ExitCode.INVALID_INPUT,
        )
    for w in widths:
        if (
            not isinstance(w, int)
            or isinstance(w, bool)
            or not (1 <= w <= MAX_C3_WIDTH_VALUE)
            or w % 2 == 0
        ):
            raise WorkerError(
                "invalid_config",
                f"config.widths must be odd integers in [1, {MAX_C3_WIDTH_VALUE}]"
                " (even width breaks padding parity)",
                ExitCode.INVALID_INPUT,
            )


def _zero_pad_row(model: ByteCNN) -> None:
    """embedding の id=0（詰め物）行を厳密にゼロへ揃える（クラスの docstring 3 番）。"""
    w = np.array(model.embed.weight, dtype=np.float32)
    w[0, :] = 0.0
    model.embed.weight = mx.array(w)


def _batchify(id_lists: list[list[int]]) -> tuple[mx.array, mx.array]:
    """可変長のトークン列を 0 パディングして (ids, mask) のバッチへ変換する。

    `C3Kind.train` の学習ループ自体はこの関数を使わない（`_encode_examples` が
    あらかじめ確保した配列から直接スライスする。P0-1）。任意の可変長トークン列を
    まとめてバッチ化したいテスト（`tests/test_c3_mlx_onnx_parity.py` 等）向けに
    残してある小さなユーティリティ。
    """
    length = max(1, max(len(s) for s in id_lists))
    arr = np.zeros((len(id_lists), length), dtype=np.int32)
    for i, s in enumerate(id_lists):
        arr[i, : len(s)] = s
    mask = (arr > 0).astype(np.float32)
    return mx.array(arr), mx.array(mask)


def _encode_examples(
    examples: list[TrainExample], max_bytes: int, resource_budget: budget_mod.ResourceBudget
) -> np.ndarray:
    """学習データ全体を `_encode_texts` でエンコードする（入力本文だけを渡す。
    ラベルは使わない）。
    """
    return _encode_texts([ex.input for ex in examples], max_bytes, resource_budget)


def _encode_texts(
    texts: Sequence[str], max_bytes: int, resource_budget: budget_mod.ResourceBudget
) -> np.ndarray:
    """入力本文の列を、あらかじめ確保した numpy int32 配列へ行ごとにエンコードする
    （学習データ・学習直後の validation 予測〔`predict_labels`〕で共有する）。

    P0-1: Python のリストのリスト（`[[encode_bytes(...)], ...]`）として全件を
    保持すると、呼び出し前の `budget_mod.check_total_tokens` による見積もり
    検査を経ずに、examples 件数 × max_bytes に比例したメモリを確保してしまう。
    ここでは検査済みの上限に収まるサイズの配列を 1 回だけ確保し（`np.zeros`）、
    行ごとに `encode_bytes` の結果を書き込む。1024 行ごとに
    `resource_budget.check()` を呼び、エンコード自体が長時間・大量メモリに
    ならないかも監視する。

    戻り値は詰め物列を実際の最大長まで切り詰めた 2 次元配列（形状
    `(len(examples), 実際の最大長)`）。詰め物 id=0 の行（`encode_bytes("")`）を
    含め、既存の `_batchify` と同じ「id 0 = 詰め物」の規約に従う。
    """
    n = len(texts)
    arr = np.zeros((n, max_bytes), dtype=np.int32)
    max_len = 1
    for i, text in enumerate(texts):
        row = encode_bytes(text, max_bytes)
        length = len(row)
        arr[i, :length] = row
        if length > max_len:
            max_len = length
        if (i + 1) % 1024 == 0:
            resource_budget.check()
    resource_budget.check()  # 端数分（1024 の倍数に満たない残り）の確認
    return arr[:, :max_len]


def _c3_param_count(n_classes: int, emb: int, filters: int, widths: tuple[int, ...]) -> int:
    """`ByteCNN(n_classes, emb, filters, widths, dropout)` のパラメータ総数を、
    モデルを実際に構築せずに解析的に求める（P0-2: サイズ超過のモデルを
    構築する前に拒否するため）。

    内訳: `embed`（`_N_TOKENS × emb`）・各 `convs[i]`（重み `filters × k × emb`
    + バイアス `filters`。`k` はカーネル幅）・`out`（重み
    `n_classes × (filters × len(widths))` + バイアス `n_classes`）。
    `dropout` はパラメータを持たないため関与しない。
    """
    embed_params = _N_TOKENS * emb
    conv_params = sum(filters * k * emb + filters for k in widths)
    out_params = n_classes * (filters * len(widths)) + n_classes
    return embed_params + conv_params + out_params


@dataclass(frozen=True)
class C3TrainedModel:
    """C3 の学習結果（ONNX 書き出しに要る情報一式）。

    `config`・`label_order`・`max_bytes` は `kinds/__init__.py::TrainedModel`
    プロトコルが宣言する共通フィールド（`cli.py::run_train` が種類非依存で読む）。
    `resource_budget` は学習ループで使ったものと同じインスタンスを持ち回り、
    ONNX 書き出し（`export_onnx`）でも 1 回だけ資源上限を検査する（P0-B。
    書き出し自体は学習ほど長時間・大量のメモリを使わないが、念のため検査する）。
    """

    model: ByteCNN
    config: dict[str, Any] = field(default_factory=dict)
    label_order: list[str] = field(default_factory=list)
    max_bytes: int = 512
    seed: int = 0
    epochs_run: int = 0
    resource_budget: budget_mod.ResourceBudget | None = None


class C3Kind:
    """選択口（`kinds/__init__.py`）が呼ぶ C3 の学習・書き出し実装。"""

    def train(
        self,
        examples: list[TrainExample],
        request: TrainRequest,
        resource_budget: budget_mod.ResourceBudget,
    ) -> C3TrainedModel:
        """呼び出し元（`cli.py`）が生成した `resource_budget` をそのまま使う
        （P0-1: 学習データの読み込み・学習・書き出しを 1 つの予算として扱う。
        ここで新しい `ResourceBudget` を作らない）。
        """
        cfg = {**DEFAULT_CONFIG, **request.config}
        _validate_config(cfg)
        widths = tuple(int(w) for w in cfg["widths"])
        label_order = request.label_order
        label_id = {label: i for i, label in enumerate(label_order)}
        n_classes = len(label_order)

        epochs = int(cfg["epochs"])
        emb = int(cfg["emb"])
        filters = int(cfg["filters"])
        # P0-2: モデルを実際に構築する前に、パラメータ数から見積もったサイズが
        # 上限（REQ-30 の配布パッケージ容量目安 40MB を準用）を超えないか検査する。
        budget_mod.check_model_bytes(_c3_param_count(n_classes, emb, filters, widths))
        # P0-B: 学習開始前に総ステップ数（examples 件数 × epochs）の見積もりで
        # 上限を検査する（REQ-39。1 ステップも回さずに reject できる）。
        budget_mod.check_sample_steps(len(examples), epochs)
        # P0-1: エンコード前に総トークン数（examples 件数 × max_bytes）の見積もりで
        # 上限を検査する（Python のリストのリストとして無検査のまま確保しない）。
        budget_mod.check_total_tokens(len(examples), request.max_bytes)

        mx.set_default_device(mx.cpu if request.device == "cpu" else mx.gpu)
        mx.random.seed(request.seed)
        rng = np.random.default_rng(request.seed)

        model = ByteCNN(n_classes, emb, filters, widths, cfg["dropout"])
        mx.eval(model.parameters())
        _zero_pad_row(model)

        opt = optim.AdamW(learning_rate=cfg["lr"], weight_decay=cfg["weight_decay"])

        # P0-1: あらかじめ確保した numpy 配列へエンコードする（Python のリストの
        # リストにしない）。行の並びは examples の順序のまま（決定性テスト・
        # rng.permutation の対象インデックスの意味を変えない）。
        ids_arr = _encode_examples(examples, request.max_bytes, resource_budget)
        labels_arr = np.array([label_id[ex.label] for ex in examples], dtype=np.int32)

        def loss_fn(mdl: ByteCNN, x: mx.array, m: mx.array, y: mx.array) -> mx.array:
            return nn.losses.cross_entropy(mdl(x, m), y, reduction="mean")

        step = nn.value_and_grad(model, loss_fn)
        batch_size = int(cfg["batch_size"])
        model.train()
        for _epoch in range(epochs):
            order = rng.permutation(len(examples))
            for start in range(0, len(order), batch_size):
                batch_idx = order[start : start + batch_size]
                x_np = ids_arr[batch_idx]
                m_np = (x_np > 0).astype(np.float32)
                x = mx.array(x_np)
                m = mx.array(m_np)
                y = mx.array(labels_arr[batch_idx])
                loss, grads = step(model, x, m, y)
                opt.update(model, grads)
                _zero_pad_row(model)
                mx.eval(model.parameters(), opt.state, loss)
                if not math.isfinite(float(loss.item())):
                    # PoC-16 `exitcode.rs` の PENDING（12）=「使える候補が 1 つもない
                    # （全候補が発散）」という割り当てに倣い、この学習ワーカー内で
                    # 唯一の候補が発散した状態を PENDING として扱う（REQ-21: 保留は
                    # 「合否を判定するための情報が現時点で欠ける」状態を表す。複数候補
                    # からの再選定〔TASK-18.x〕は本ワーカーの範囲外だが、終了コードの
                    # 意味はそれを見越して PoC のマッピングに合わせておく）。
                    raise WorkerError(
                        "training_diverged",
                        "training loss became non-finite",
                        ExitCode.PENDING,
                    )
                # P0-B: 壁時計・RSS（device="gpu" なら MLX active memory も）を
                # バッチごとに検査する（REQ-39。budget.py::ResourceBudget 参照）。
                resource_budget.check()
        model.eval()
        return C3TrainedModel(
            model=model,
            config=cfg,
            label_order=list(label_order),
            max_bytes=request.max_bytes,
            seed=request.seed,
            epochs_run=epochs,
            resource_budget=resource_budget,
        )

    def export_onnx(self, trained: C3TrainedModel, out: IO[bytes]) -> None:
        if trained.resource_budget is not None:
            trained.resource_budget.check()
        _export_c3_onnx(trained, out)


def predict_labels(
    trained: C3TrainedModel,
    rows: Sequence[tuple[str, str]],
    resource_budget: budget_mod.ResourceBudget | None = None,
) -> list[dict[str, Any]]:
    """学習直後の validation 予測（REQ-18・REQ-27。`predict.py` から呼ばれる）。

    `rows` は `(id, input)` の列で、**正解ラベルは受け取らない**。学習時と同じ
    エンコード（`_encode_texts`）・マスク（`ids > 0`）・順伝播
    （`ByteCNN.__call__`）を通し、argmax のラベルを返す（同点は `label_order` の
    先頭側。numpy の `argmax` は最初の最大値を返す）。戻り値は入力と同じ順序・
    件数の `{id, status:"ok", predicted_label}`。

    チャンクは学習と同じ `batch_size`（学習開始前に資源検査済みの大きさ）。
    チャンクごとに `resource_budget` を検査する（REQ-39。省略時は学習で使った
    インスタンス）。詰め物位置はマスクされるため、チャンクの分け方で結果は変わらない
    （REQ-28。`tests/test_c3_batch_parity.py`）。
    """
    budget = resource_budget if resource_budget is not None else trained.resource_budget
    if budget is None:
        raise WorkerError(
            "runtime_error",
            "prediction requires a resource budget",
            ExitCode.RUNTIME_ERROR,
        )
    chunk_size = max(1, min(int(trained.config["batch_size"]), MAX_C3_BATCH_SIZE))
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), chunk_size):
        budget.check()
        chunk = rows[start : start + chunk_size]
        ids_arr = _encode_texts([text for _rid, text in chunk], trained.max_bytes, budget)
        logits = trained.model(mx.array(ids_arr), mx.array((ids_arr > 0).astype(np.float32)))
        chosen = np.argmax(np.array(logits, dtype=np.float32), axis=1)
        budget.check()
        for (rid, _text), label_index in zip(chunk, chosen, strict=True):
            records.append(ok_prediction_record(rid, trained.label_order[int(label_index)]))
    return records


def _f32(arr: np.ndarray, name: str) -> TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name=name)


def _i64(arr: np.ndarray, name: str) -> TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.int64), name=name)


def _export_c3_onnx(trained: C3TrainedModel, out: IO[bytes]) -> None:
    # P0-2: 書き出しの各段（テンソル抽出・グラフ構築・検証・保存）の間で
    # 資源上限（REQ-39）を検査する。`resource_budget` は学習ループと同じ
    # インスタンス（`C3Kind.export_onnx` が渡す）で、無い場合は検査をスキップする
    # （テスト等で `resource_budget=None` の `C3TrainedModel` を直接構築した場合）。
    budget_check = trained.resource_budget.check if trained.resource_budget is not None else None

    widths = tuple(int(w) for w in trained.config["widths"])
    n_classes = len(trained.label_order)
    params = trained.model.parameters()

    embed = np.array(params["embed"]["weight"], dtype=np.float32)  # (257, E)
    if not np.array_equal(embed[0, :], np.zeros_like(embed[0, :])):
        # クラス docstring 3 番の不変条件（詰め物行は厳密ゼロ）が崩れている。
        # バッチ推論と単独推論の一致契約（REQ-28）が保てないため書き出しを拒否する。
        raise WorkerError(
            "runtime_error",
            "internal invariant violated: pad embedding row is not exactly zero",
            ExitCode.RUNTIME_ERROR,
        )
    if budget_check is not None:
        budget_check()  # 段 1: embedding の抽出後

    ids = helper.make_tensor_value_info("ids", TensorProto.INT64, ["N", "T"])
    probs_out = helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["N", n_classes])

    initializers = [
        _f32(embed, "embed"),
        _i64(np.array(0), "zero_i64"),
        _f32(np.array(1.0), "one_f32"),
        _f32(np.array(_MASK_NEG_VALUE), "neg_big_f32"),
        _i64(np.array([1]), "unsqueeze_axes_1"),
    ]
    nodes = [
        helper.make_node("Gather", ["embed", "ids"], ["emb_bth"], axis=0),
        helper.make_node("Transpose", ["emb_bth"], ["emb_bct"], perm=[0, 2, 1]),
        # マスク = ids > 0（詰め物 id=0 だけを検出する。ids は常に 0 以上）
        helper.make_node("Greater", ["ids", "zero_i64"], ["mask_bool"]),
        helper.make_node("Cast", ["mask_bool"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["one_f32", "mask_f"], ["inv_mask"]),
        helper.make_node("Mul", ["inv_mask", "neg_big_f32"], ["neg"]),
        helper.make_node("Unsqueeze", ["neg", "unsqueeze_axes_1"], ["neg_unsq"]),
    ]
    pooled_names = []
    for i, k in enumerate(widths):
        cw = np.array(params["convs"][i]["weight"], dtype=np.float32)  # (filters, k, emb)
        cb = np.array(params["convs"][i]["bias"], dtype=np.float32)  # (filters,)
        cw_onnx = np.transpose(cw, (0, 2, 1))  # (filters, emb, k)
        initializers.append(_f32(cw_onnx, f"conv{i}_w"))
        initializers.append(_f32(cb, f"conv{i}_b"))
        pad = k // 2
        nodes.append(
            helper.make_node(
                "Conv",
                ["emb_bct", f"conv{i}_w", f"conv{i}_b"],
                [f"conv{i}_out"],
                kernel_shape=[k],
                pads=[pad, pad],
                strides=[1],
            )
        )
        nodes.append(helper.make_node("Relu", [f"conv{i}_out"], [f"relu{i}_out"]))
        nodes.append(helper.make_node("Add", [f"relu{i}_out", "neg_unsq"], [f"masked{i}_out"]))
        nodes.append(
            helper.make_node("ReduceMax", [f"masked{i}_out"], [f"pool{i}"], axes=[2], keepdims=0)
        )
        pooled_names.append(f"pool{i}")
        if budget_check is not None:
            budget_check()  # 段 2: 各 conv ブランチのテンソル抽出後
    nodes.append(helper.make_node("Concat", pooled_names, ["pooled"], axis=1))

    out_w = np.array(params["out"]["weight"], dtype=np.float32)  # (n_classes, filters*len(widths))
    out_b = np.array(params["out"]["bias"], dtype=np.float32)  # (n_classes,)
    out_wT = out_w.T
    initializers.append(_f32(out_wT, "out_wT"))
    initializers.append(_f32(out_b, "out_b"))
    nodes.append(
        helper.make_node("Gemm", ["pooled", "out_wT", "out_b"], ["logits"], alpha=1.0, beta=1.0)
    )
    nodes.append(helper.make_node("Softmax", ["logits"], ["probs"], axis=1))
    if budget_check is not None:
        budget_check()  # 段 3: グラフ構築後（check_model の前）

    graph = helper.make_graph(nodes, "c3_cnn", [ids], [probs_out], initializer=initializers)
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model_proto.ir_version = 8
    onnx.checker.check_model(model_proto)
    if budget_check is not None:
        budget_check()  # 段 4: check_model 後（save の前）
    # 経路は一切扱わない（呼び出し元が開いたファイルオブジェクトへ書き込むだけ。
    # kinds/__init__.py::Kind.export_onnx の docstring 参照）。`onnx.save` ではなく
    # 直接シリアライズする: `onnx.save` はファイルオブジェクトも受け付けるが、
    # 明示的に protobuf バイト列を書き込む方が経路非依存の契約として単純で、
    # `onnx.save` がファイルオブジェクトに対して行う内部実装（seek 等）へ
    # 依存しない。決定性テスト（byte-identical）は `SerializeToString()` の
    # 出力がそのまま `onnx.save` の protobuf 形式と同じであることに依存する。
    out.write(model_proto.SerializeToString())
    if budget_check is not None:
        budget_check()  # 段 5: save 後
