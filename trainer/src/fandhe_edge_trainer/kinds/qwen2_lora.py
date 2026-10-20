"""追加学習の種類 `qwen2_lora`（ベースモデル Qwen2 系＋LoRA）の学習（REQ-41・TASK-41.5-3・#426）。

学習ワーカーの選択口（`kinds/__init__.py::_registry`）から呼ばれる。利用者がローカルに置いた
ベースモデル（`qwen2_model.load_base_model`・#425）に LoRA を付け、選択肢のラベル文字列を
chat template 形式で生成する損失で MLX 学習する。出力は選択肢の採点だけで、自由生成はしない
（REQ-19b の対応づけ (b) と同じ考え方）。既定候補には入れず、利用者が明示したときだけ使う。

ベースモデルの所在は種類固有の `config` で受ける（トップレベルの入出力契約は変えない）:

- `base_model_dir`: `request.root` からの相対パス（必須。経路は `guard.confine` が閉じ込める）
- `base_model_sha256`: {ファイル名: sha256}（必須。照合は `load_base_model`）
- `base_model_name`: 来歴の名前（必須）。`base_model_license`・`base_model_origin`: 任意

seed は Python・NumPy・MLX のそれぞれで明示的に設定する（バッチ順は seed 付き permutation、
LoRA の初期化は `mx.random.key(seed)`）。時間予算に達したら打ち切らず `limit_exceeded`（20）
で止める。予測は選択肢の条件付き対数尤度の採点（`predict_records`。#427。長さ正規化なし）で、
argmax を対応づけ (b)（`autoregressive.map_scores_to_choice`）で選択肢 ID に解決し、解決できない
ものは `status:"error"`（判定不能）にする。ONNX の採点グラフと shard の書き出しは
`qwen2_onnx.export_scoring_onnx`（#431）。`export_onnx`（Kind 契約）は `open_member` で shard と
`tokenizer.json` を成果物ディレクトリへ作り、`artifact.json` へ足す追加欄（tokenizer・shard の
sha256・選択肢表・ベースモデルの来歴）を返す（自己完結パッケージ。TASK-41.6-3・#432）。
来歴の `base_model_name`（必須）・`base_model_license`・`base_model_origin`（任意。無ければ
`null`。捏造しない）は種類固有の `config` で受ける。
mlx は関数内で遅延 import する（選択口の import を軽く保つ）。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import IO, Any

import numpy as np

from .. import budget, limits
from ..contract import TrainExample, TrainRequest
from ..errors import WorkerError
from ..exitcode import ExitCode
from .autoregressive import ChoiceMappingTally, build_prediction_record, map_scores_to_choice

# PoC-26 追補 1・追補 2 の固定値と P15 の代表構成（iters 300・lr 1e-4）。変える場合は
# fixtures/train_contract/kind_defaults.json も同時に更新する。
DEFAULT_CONFIG: dict[str, Any] = {
    "lora_rank": 8,
    "lora_scale": 20.0,
    "lora_dropout": 0.0,
    "lora_num_layers": 8,
    "lr": 0.0001,
    "iters": 300,
    "batch_size": 4,
    "max_seq_length": 512,
    "dtype": "bfloat16",
}

#: 既定値を持たない必須項目（利用者ごとに異なるため `DEFAULT_CONFIG` に入れない）。
_REQUIRED_FIELDS = ("base_model_dir", "base_model_sha256", "base_model_name")
#: 既定値を持たない任意項目（無ければ来歴は `null`）。
_OPTIONAL_FIELDS = ("base_model_license", "base_model_origin")
_CONFIG_FIELDS = (
    frozenset(DEFAULT_CONFIG) | frozenset(_REQUIRED_FIELDS) | frozenset(_OPTIONAL_FIELDS)
)
#: 来歴文字列の上限（Rust data crate の `base_model` 検証と同じ規則）。
_NAME_MAX, _LICENSE_MAX, _ORIGIN_MAX = 256, 256, 1024
_DTYPES = ("bfloat16", "float32")

#: 採点 1 回の forward に入れる選択肢の数（K が大きくてもメモリが線形に伸びない。PoC-26 と同値）
SCORE_CHUNK = 8


def _invalid_config(message: str) -> WorkerError:
    return WorkerError("invalid_config", message, ExitCode.INVALID_INPUT)


def _int_in(cfg: dict[str, Any], key: str, lo: int, hi: int) -> int:
    v = cfg[key]
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise _invalid_config(f"config.{key} must be an integer in {lo}..{hi}")
    return v


def _num_in(cfg: dict[str, Any], key: str, lo: float, hi: float, *, lo_open: bool) -> float:
    v = cfg[key]
    ok = isinstance(v, (int, float)) and not isinstance(v, bool)
    if ok:
        try:
            ok = math.isfinite(v)  # 10**400 等の巨大整数は float 変換で OverflowError になる
        except OverflowError:
            ok = False
    if not ok or v > hi or (v <= lo if lo_open else v < lo):
        lb = "(" if lo_open else "["
        raise _invalid_config(f"config.{key} must be a finite number in {lb}{lo}, {hi}]")
    return float(v)


def _check_provenance_text(cfg: dict[str, Any]) -> None:
    """来歴文字列の検査（ローカル絶対パス・資格情報の漏えいを防ぐ。REQ-40・REQ-39）。"""

    def text(key: str, limit: int) -> str | None:
        v = cfg.get(key)
        if v is None and key != "base_model_name":
            return None
        if not isinstance(v, str) or not v or len(v.encode("utf-8")) > limit:
            raise _invalid_config(f"config.{key} must be a non-empty string up to {limit} bytes")
        if v != v.strip() or any(ord(c) < 0x20 or ord(c) == 0x7F for c in v):
            raise _invalid_config(f"config.{key} must not have control characters or edge spaces")
        return v

    text("base_model_name", _NAME_MAX)
    text("base_model_license", _LICENSE_MAX)
    origin = text("base_model_origin", _ORIGIN_MAX)
    if origin is not None and (
        not origin.startswith("https://") or any(c in origin for c in "@?# \\")
    ):
        raise _invalid_config("config.base_model_origin must be an https:// URL without @ ? #")


def _validate_config(cfg: dict[str, Any]) -> None:
    """実効 config（既定値＋リクエスト）を検証する。違反は `invalid_config`（64）。"""
    unknown = set(cfg) - _CONFIG_FIELDS
    if unknown:
        raise _invalid_config(
            f"unknown config fields: {len(unknown)} field(s) (supported: {sorted(_CONFIG_FIELDS)})"
        )
    for k in _REQUIRED_FIELDS:
        if k not in cfg:
            raise _invalid_config(f"config.{k} is required")
    d = cfg["base_model_dir"]
    if not isinstance(d, str) or not d:
        raise _invalid_config("config.base_model_dir must be a non-empty string")
    if not isinstance(cfg["base_model_sha256"], dict):
        raise _invalid_config("config.base_model_sha256 must be an object")
    _check_provenance_text(cfg)
    # 範囲検査は `apply_lora` と重複する。`apply_lora` は単体でも公開されるため残し、こちらは
    # ベース読み込み（重い処理）の前に弾くための早期検査として置く。
    _int_in(cfg, "lora_rank", 1, limits.MAX_LORA_RANK)
    _int_in(cfg, "lora_num_layers", 1, 1 << 10)  # ベースの層数との比較は `apply_lora`
    _int_in(cfg, "iters", 1, limits.MAX_LORA_ITERS)
    _int_in(cfg, "batch_size", 1, limits.MAX_LORA_BATCH_SIZE)
    _int_in(cfg, "max_seq_length", limits.MIN_LORA_SEQ_LENGTH, limits.MAX_LORA_SEQ_LENGTH)
    _num_in(cfg, "lora_scale", 0.0, limits.MAX_LORA_SCALE, lo_open=True)
    _num_in(cfg, "lora_dropout", 0.0, 0.999999, lo_open=False)
    _num_in(cfg, "lr", 0.0, limits.MAX_LORA_LR, lo_open=True)
    if cfg["dtype"] not in _DTYPES:
        raise _invalid_config(f"config.dtype must be one of {list(_DTYPES)}")


def _activation_bytes(c: Any, cfg: dict[str, Any]) -> int:
    """LoRA 層の逆伝播用に保持する活性の上限寄りの見積もり（float32 換算。REQ-39）。

    1 token・1 層あたり hidden の約 10 本と intermediate の 3 本、attention 重み
    （head 数 x 系列長）を保持するとみなす。保持が必要なのは LoRA を付ける層のみ。
    """
    tokens = cfg["batch_size"] * cfg["max_seq_length"]
    per_token = 10 * c.hidden_size + 3 * c.intermediate_size
    per_token += c.num_attention_heads * cfg["max_seq_length"]
    layers_bytes = 4 * tokens * per_token * cfg["lora_num_layers"]
    # 損失計算の float32 logits（順伝播・softmax・勾配の 3 本分の上限寄り見積もり）
    return layers_bytes + 3 * 4 * tokens * c.vocab_size


def _seed_all(seed: int) -> None:
    """Python・NumPy・MLX の乱数を明示的に seed する（決定性の要件。REQ-19・REQ-41）。"""
    import mlx.core as mx

    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    mx.random.seed(seed)


def _system_prompt(label_order: list[str]) -> str:
    """選択肢を列挙する汎用の固定文（学習と #427 の採点で同じ文を使う）。"""
    return "Answer with exactly one of: " + ", ".join(label_order) + "."


def make_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    """長さ順に並べて `batch_size` 件ずつに区切る。端数バッチ（最長側）も捨てずに残す。"""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def collate(seqs: list[list[int]], idxs: list[int], pad_id: int) -> tuple[Any, Any, Any]:
    """バッチを右 pad し、損失に使う位置（pad 以外の次 token 予測）の平坦な添字と正解を返す。"""
    import mlx.core as mx

    n = max(len(seqs[i]) for i in idxs)
    arr = np.full((len(idxs), n), pad_id, dtype=np.int32)
    pos: list[int] = []
    tgt: list[int] = []
    for r, i in enumerate(idxs):
        s = seqs[i]
        arr[r, : len(s)] = s
        for j in range(len(s) - 1):
            pos.append(r * (n - 1) + j)
            tgt.append(s[j + 1])
    return (
        mx.array(arr[:, :-1]),
        mx.array(np.array(pos, dtype=np.int32)),
        mx.array(np.array(tgt, dtype=np.int32)),
    )


def batch_loss(model: Any, inputs: Any, pos: Any, tgt: Any) -> Any:
    """pad 以外の全 token の cross_entropy 平均（float32。prompt もマスクしない）。

    損失位置の隠れ状態だけを float32 の logits にし、全 [B,L,vocab] は作らない。
    """
    import mlx.core as mx
    import mlx.nn as nn

    h = model.hidden_states(inputs)
    hv = h.reshape(-1, h.shape[-1])[pos]
    logits = model.project(
        hv, source_shape=(inputs.shape[0], inputs.shape[1]), with_mask=False
    ).astype(mx.float32)
    return nn.losses.cross_entropy(logits, tgt, reduction="mean")


@dataclass
class Qwen2LoraTrainedModel:
    """`qwen2_lora` の学習結果。`config`・`label_order`・`max_bytes` は `TrainedModel` の必須項目。

    `base_file_sha256` は照合に使ったベースモデルのファイルの sha256（来歴。REQ-40）。
    `max_bytes` は入力表現が token のため使わず、持ち回すだけ（長さは `max_seq_length`）。
    """

    config: dict[str, Any]
    label_order: list[str]
    max_bytes: int
    model: Any
    tokenizer: Any
    base_file_sha256: dict[str, str]
    final_loss: float
    iters_done: int
    seed: int
    resource_budget: budget.ResourceBudget
    #: 配布パッケージへ複写する `tokenizer.json` の照合済みバイト列と来歴用のサイズ（#432）
    tokenizer_bytes: bytes = b""
    base_file_sizes: dict[str, int] = field(default_factory=dict)


def _build_sequences(
    examples: list[TrainExample],
    request: TrainRequest,
    tokenizer: Any,
    cfg: dict[str, Any],
    vocab_size: int,
    resource_budget: budget.ResourceBudget,
) -> tuple[list[list[int]], int]:
    """全件の学習列（chat template＋ラベル＋`<|im_end|>`）を組み立て、長さを確認する。"""
    # 保持前に総トークン数の上限を件数 x max_seq_length で見積もる（REQ-39）
    budget.check_total_tokens(len(examples), cfg["max_seq_length"])
    system = _system_prompt(request.label_order)
    header = tokenizer.encode("assistant\n")
    for label in request.label_order:
        body = tokenizer.encode(label)
        # 学習列は `assistant\n{label}` を 1 区間で encode する。ラベル単独の列と一致しないと
        # 学習と採点（#427）で別の token 列を見る。
        if not body or tokenizer.encode("assistant\n" + label) != header + body:
            raise _invalid_config("a label tokenizes differently in context")
    seqs: list[list[int]] = []
    for ex in examples:
        resource_budget.check()
        ids = tokenizer.build_chat_ids(system, ex.input, ex.label, add_generation_prompt=False)
        if len(ids) > cfg["max_seq_length"]:
            # 切り詰めると学習対象のラベル部分が欠けるため、止める。
            raise WorkerError(
                "limit_exceeded",
                "a training sequence exceeds max_seq_length",
                ExitCode.LIMIT_EXCEEDED,
            )
        if ids and max(ids) >= vocab_size:
            raise WorkerError(
                "invalid_data", "token id outside the model vocabulary", ExitCode.INVALID_INPUT
            )
        seqs.append(ids)
    return seqs, tokenizer.special_ids["endoftext"]


def _train_loop(
    model: Any,
    seqs: list[list[int]],
    *,
    pad_id: int,
    cfg: dict[str, Any],
    seed: int,
    resource_budget: budget.ResourceBudget,
) -> tuple[float, int]:
    """Adam・定数 lr で `iters` ステップ学習する。予算超過は `ResourceBudget.check()` が止める。"""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    batches = make_batches([len(s) for s in seqs], cfg["batch_size"])
    rng = np.random.default_rng(seed)
    opt = optim.Adam(learning_rate=cfg["lr"])
    step_fn = nn.value_and_grad(model, batch_loss)
    model.train()
    order: list[int] = []
    loss_value = float("nan")
    for _ in range(cfg["iters"]):
        resource_budget.check()
        if not order:
            order = [int(i) for i in rng.permutation(len(batches))]
        inputs, pos, tgt = collate(seqs, batches[order.pop(0)], pad_id)
        loss, grads = step_fn(model, inputs, pos, tgt)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise WorkerError("runtime_error", "loss is not finite", ExitCode.RUNTIME_ERROR)
    resource_budget.check()
    model.eval()
    return loss_value, cfg["iters"]


def score_chunk(model: Any, prompt_ids: list[int], label_ids: list[list[int]], pad_id: int) -> Any:
    """選択肢の部分集合を 1 回の forward で採点する（`score_labels` の 1 チャンク）。

    各列は `prompt + label + <|im_end|>` を右 pad する（pad は後ろで因果マスクにより実 token から
    見えない）。採点位置の隠れ状態だけを float32 の logits にし、符号反転した交差エントロピーを
    選択肢ごとに合計して対数尤度（float64）で返す。
    """
    import mlx.core as mx
    import mlx.nn as nn

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
    model: Any,
    prompt_ids: list[int],
    label_ids: list[list[int]],
    pad_id: int,
    chunk: int = SCORE_CHUNK,
) -> Any:
    """全選択肢の対数尤度の合計（長さ正規化なし。PoC-26 追補 1 の固定値）。

    `chunk` 件ずつ forward する。各選択肢の値は他に依存しないので、分割しても結果は
    浮動小数の丸めの範囲でしか変わらない。
    """
    if chunk < 1:
        raise ValueError("chunk must be positive")
    return np.concatenate(
        [
            score_chunk(model, prompt_ids, label_ids[i : i + chunk], pad_id)
            for i in range(0, len(label_ids), chunk)
        ]
    )


def choice_token_ids(tokenizer: Any, label_order: list[str], vocab_size: int) -> list[list[int]]:
    """選択肢ごとの採点対象 token 列（ラベル token 列＋`<|im_end|>`）を返す。

    `predict_records`（MLX の採点）と `qwen2_onnx.export_scoring_onnx`（グラフ内の定数）が
    同じ列を使うための共有点（REQ-41。ずれると書き出しモデルと学習側の予測が食い違う）。
    語彙外の id を含むなら `invalid_data`（64）。
    """
    end = tokenizer.special_ids["im_end"]
    label_ids = [[*tokenizer.encode(x), end] for x in label_order]
    if any(t >= vocab_size for ids in label_ids for t in ids):
        raise WorkerError(
            "invalid_data", "token id outside the model vocabulary", ExitCode.INVALID_INPUT
        )
    return label_ids


def predict_records(
    trained: Qwen2LoraTrainedModel,
    rows: list[tuple[str, str]],
    *,
    resource_budget: budget.ResourceBudget | None = None,
    tally: ChoiceMappingTally | None = None,
) -> list[dict[str, Any]]:
    """`rows`（`(id, input)` の列）を選択肢スコアリングで予測する（REQ-41・REQ-19b・TASK-41.5-4）。

    学習ワーカーの `predict.predict_validation`（学習直後の validation 予測）から呼ばれる。
    **正解ラベルは受け取らない**（REQ-27）。自由生成はせず、各選択肢の条件付き対数尤度の argmax を
    対応づけ (b) で選択肢 ID に解決する。非有限スコア・解決不能は `status:"error"`（判定不能）。
    長さ超過は切り詰めず採点前に `limit_exceeded`（20）で止める。エラーに入力本文は含めない。
    """
    rb = resource_budget or trained.resource_budget
    cfg, tok, order = trained.config, trained.tokenizer, trained.label_order
    # 保持前に総量を見積もる。件数は contract の validation 件数上限で既に制限される。
    # ponytail: 全件をリストで返す。大量件数を流す用途ではジェネレータ化する。
    budget.check_total_tokens(len(rows), cfg["max_seq_length"])
    vocab = trained.model.config.vocab_size
    label_ids = choice_token_ids(tok, order, vocab)
    longest = max(len(c) for c in label_ids)
    system = _system_prompt(order)
    prompts: list[list[int]] = []
    for _, text in rows:
        rb.check()
        ids = tok.build_chat_ids(system, text, add_generation_prompt=True)
        if ids and max(ids) >= vocab:
            raise WorkerError(
                "invalid_data", "token id outside the model vocabulary", ExitCode.INVALID_INPUT
            )
        if len(ids) + longest > cfg["max_seq_length"]:
            raise WorkerError(
                "limit_exceeded",
                "a prediction sequence exceeds max_seq_length",
                ExitCode.LIMIT_EXCEEDED,
            )
        prompts.append(ids)
    pad_id = tok.special_ids["endoftext"]
    # 対応づけ (b) の表現（UTF-8 バイト値 + 1）に合わせる。
    choice_ids = {x: [b + 1 for b in x.encode("utf-8")] for x in order}
    trained.model.eval()
    out: list[dict[str, Any]] = []
    for (rid, _), ids in zip(rows, prompts, strict=True):
        rb.check()
        loglik = score_labels(trained.model, ids, label_ids, pad_id)
        rb.check()  # forward の後にも確認する（最後の 1 件で超過しても 20）
        mapping = map_scores_to_choice(loglik, order, choice_ids)
        if tally is not None:
            tally.record(mapping)
        out.append(build_prediction_record(rid, mapping, order))
    return out


class Qwen2LoraKind:
    """`Kind` プロトコルの実装。学習 #426・予測 #427・グラフ #431・パッケージ書き出し #432。"""

    def train(
        self,
        examples: list[TrainExample],
        request: TrainRequest,
        resource_budget: budget.ResourceBudget,
    ) -> Qwen2LoraTrainedModel:
        """ベースモデルを読み、LoRA を付けて学習する（REQ-41・TASK-41.5-3）。"""
        cfg = {**DEFAULT_CONFIG, **request.config}
        _validate_config(cfg)

        import mlx.core as mx

        from ..qwen2_model import apply_lora, load_base_model

        mx.set_default_device(mx.gpu if request.device == "gpu" else mx.cpu)
        _seed_all(request.seed)
        if len(examples) < cfg["batch_size"]:
            raise WorkerError(
                "invalid_data", "fewer training examples than batch_size", ExitCode.INVALID_INPUT
            )
        dtype = mx.bfloat16 if cfg["dtype"] == "bfloat16" else mx.float32
        base = load_base_model(
            request.root,
            cfg["base_model_dir"],
            expected_sha256=cfg["base_model_sha256"],
            dtype=dtype,
            resource_budget=resource_budget,
        )
        c = base.config
        if cfg["max_seq_length"] > c.max_position_embeddings:
            raise _invalid_config(
                f"max_seq_length exceeds the model's max_position_embeddings "
                f"({c.max_position_embeddings})"
            )
        logits_elements = cfg["batch_size"] * cfg["max_seq_length"] * c.vocab_size
        if logits_elements > limits.MAX_LOSS_LOGITS_ELEMENTS:
            raise WorkerError(
                "limit_exceeded",
                "batch_size x max_seq_length x vocab exceeds the memory limit",
                ExitCode.LIMIT_EXCEEDED,
            )
        seqs, pad_id = _build_sequences(
            examples, request, base.tokenizer, cfg, c.vocab_size, resource_budget
        )
        apply_lora(
            base.model,
            num_layers=cfg["lora_num_layers"],
            rank=cfg["lora_rank"],
            scale=float(cfg["lora_scale"]),
            dropout=float(cfg["lora_dropout"]),
            seed=request.seed,
            rss_bytes=resource_budget.rss_bytes,
            activation_bytes=_activation_bytes(c, cfg),
        )
        loss, done = _train_loop(
            base.model,
            seqs,
            pad_id=pad_id,
            cfg=cfg,
            seed=request.seed,
            resource_budget=resource_budget,
        )
        return Qwen2LoraTrainedModel(
            config=cfg,
            label_order=list(request.label_order),
            max_bytes=request.max_bytes,
            model=base.model,
            tokenizer=base.tokenizer,
            base_file_sha256=base.file_sha256,
            final_loss=loss,
            iters_done=done,
            seed=request.seed,
            resource_budget=resource_budget,
            tokenizer_bytes=base.tokenizer_bytes,
            base_file_sizes=dict(base.file_sizes),
        )

    def export_onnx(
        self,
        trained: Qwen2LoraTrainedModel,
        out: IO[bytes],
        open_member: Callable[[str], IO[bytes]] | None = None,
    ) -> dict[str, Any]:
        """自己完結パッケージを書く（TASK-41.6-3・#432）。

        `out` へ model.onnx、`open_member` で外部データ shard と `tokenizer.json`（照合済みの
        バイト列をそのまま複写）を作り、`artifact.json` の追加欄（tokenizer・shard の sha256、
        選択肢表、ベースモデルの来歴）を返す。`open_member` が無ければ契約違反で `runtime_error`。
        """
        if open_member is None:
            raise WorkerError("runtime_error", "open_member is required", ExitCode.RUNTIME_ERROR)
        from ..artifact import INPUT_REPRESENTATION_KEY, TOKENIZER_FILE_NAME, HashingWriter
        from ..qwen2_onnx import export_scoring_onnx

        system_ids = trained.tokenizer.system_turn_ids(_system_prompt(trained.label_order))
        shards = export_scoring_onnx(trained, out, open_member, system_ids=system_ids)
        with open_member(TOKENIZER_FILE_NAME) as f:
            tok_writer = HashingWriter(f)
            tok_writer.write(trained.tokenizer_bytes)
        cfg, order = trained.config, trained.label_order
        token_ids = choice_token_ids(trained.tokenizer, order, trained.model.config.vocab_size)
        return {
            INPUT_REPRESENTATION_KEY: "tokenizer",
            "tokenizer_file": TOKENIZER_FILE_NAME,
            "tokenizer_sha256": tok_writer.hexdigest(),
            "external_data": [
                {"location": sh.name, "size_bytes": sh.size, "sha256": sh.sha256} for sh in shards
            ],
            "scoring": {
                "system_prompt": _system_prompt(order),
                "choices": [
                    {"id": label, "token_ids": ids}
                    for label, ids in zip(order, token_ids, strict=True)
                ],
            },
            "base_model": {
                "files": [
                    {
                        "path": name,
                        "sha256": trained.base_file_sha256[name],
                        "size_bytes": trained.base_file_sizes[name],
                    }
                    for name in sorted(trained.base_file_sha256)
                ],
                "license": cfg.get("base_model_license"),
                "model_name": cfg["base_model_name"],
                "origin": cfg.get("base_model_origin"),
            },
        }
