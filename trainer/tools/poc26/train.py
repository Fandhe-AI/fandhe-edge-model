"""PoC-26 学習ループと `train` サブコマンド（REQ-41・TASK-41.1-5・#390）。

役割: 長さ順ソート → batch → seed 付き permutation のバッチ作成（端数は捨てる）、pad 以外の全 token
の cross_entropy 平均の損失（float32）、Adam・定数 lr の学習ループ、壁時計予算（`--max-wall-seconds`
は学習ループだけに適用し、到達したら打ち切って記録）を担い、`cmd_train` が学習 -> validation 採点 ->
出力一式を書く。入力の検証をすべて終えてから出力ディレクトリを作る。`cli.py` から配線される。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from tools.poc26.assets import (
    load_assets,
    load_base_model,
    load_system_prompt,
    pins_from_args,
)
from tools.poc26.common import (
    MAX_WALL_SECONDS_CAP,
    Budget,
    as_input_error,
    err,
    invalid,
    log,
    max_rss_bytes,
    seed_all,
    sha256,
    to_dtype,
    too_large,
)
from tools.poc26.io_records import (
    LORA_KEYS,
    OutputDir,
    adapters_to_bytes,
    check_out_dir,
    common_run,
    json_text,
    load_label_order,
    load_records,
    write_scores,
)
from tools.poc26.qwen2_model import (
    Qwen2Model,
    apply_lora,
    trainable_parameter_count,
)
from tools.poc26.score import build_prompting, prepare_prompts, score_records

from fandhe_edge_trainer.exitcode import ExitCode

# メモリの事前見積もりの上限（REQ-39）: batch_size x max_seq_length x vocab（損失位置の float32
# logits の要素数の上限）。backward は logits・softmax の勾配・bf16 の写しなどで forward の約 3 倍を
# 保持しうるため、5e8 要素（float32 で 2 GB、x3 = 6 GB < RSS 上限 8 GiB）に抑える。既定の実物
# 4 x 512 x 151936 = 3.1e8 は通る。係数 3 は保守的な見積もりで実測ではない（推定）。実際の超過は
# `common.max_rss_bytes`（RSS と MLX ピークの大きい方）をステップごとに確認して 20 で止める。
MAX_LOSS_LOGITS_ELEMENTS = 500_000_000


def make_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    """長さ順に並べて `batch_size` 件ずつに区切る。端数は捨てる（mlx-lm の `iterate_batches`）。"""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i : i + batch_size] for i in range(0, len(order) - batch_size + 1, batch_size)]


def collate(
    seqs: list[list[int]], idxs: list[int], pad_id: int
) -> tuple[mx.array, mx.array, mx.array]:
    """バッチを右 pad し、損失に使う位置（pad 以外の次 token 予測）の平坦な添字と正解を返す。"""
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


def batch_loss(model: Qwen2Model, inputs: mx.array, pos: mx.array, tgt: mx.array) -> mx.array:
    """pad 以外の全 token の cross_entropy 平均（float32。prompt もマスクしない。P4 と同じ）。

    # ponytail: 損失位置の隠れ状態だけを float32 の logits にする。全 [B,L,vocab] は作らない。
    """
    h = model.hidden_states(inputs)
    logits = model.project(h.reshape(-1, h.shape[-1])[pos]).astype(mx.float32)
    return nn.losses.cross_entropy(logits, tgt, reduction="mean")


@dataclass(frozen=True)
class TrainResult:
    """学習ループの結果。`budget_reached` は壁時計の予算に達して打ち切ったことを表す。"""

    final_loss: float
    iters_done: int
    budget_reached: bool


def train_loop(
    model: Qwen2Model,
    seqs: list[list[int]],
    *,
    pad_id: int,
    iters: int,
    lr: float,
    batch_size: int,
    seed: int,
    budget: Budget,
    max_wall_seconds: float = MAX_WALL_SECONDS_CAP,
) -> TrainResult:
    """Adam・定数 lr で最大 `iters` ステップ学習する。

    壁時計が `max_wall_seconds` を超えたら（1 ステップ以上行った後に）打ち切って返す。事前登録
    4 節「予算到達は記録してその時点の最良で評価する」に従い、停止ではなく `budget_reached` で
    伝える（呼び出し側が adapter を保存して採点へ進む）。RSS 超過は `budget.check()` が停止する。
    """
    batches = make_batches([len(s) for s in seqs], batch_size)
    if not batches:
        raise invalid("not enough training records for one batch")
    rng = np.random.default_rng(seed)
    opt = optim.Adam(learning_rate=lr)
    step_fn = nn.value_and_grad(model, batch_loss)
    model.train()
    order: list[int] = []
    loss_value = float("nan")
    done = 0
    reached = False
    for step in range(1, iters + 1):
        budget.check()
        if step > 1 and budget.wall_exceeded(max_wall_seconds):
            reached = True
            break
        if not order:
            order = [int(i) for i in rng.permutation(len(batches))]
        inputs, pos, tgt = collate(seqs, batches[order.pop(0)], pad_id)
        loss, grads = step_fn(model, inputs, pos, tgt)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise err("runtime_error", "loss is not finite", ExitCode.RUNTIME_ERROR)
        done = step
        if step % 10 == 0 or step == iters:
            log(f"step {step}/{iters} loss {loss_value:.6f} elapsed {budget.elapsed():.1f}s")
    if reached:
        log(f"budget reached: stopped after {done}/{iters} steps")
    model.eval()
    return TrainResult(loss_value, done, reached)


def check_train_memory(a: argparse.Namespace, vocab_size: int) -> None:
    """学習のメモリ見積もり（損失 logits の要素数。backward の保持は上の係数で見込む）。"""
    if a.batch_size * a.max_seq_length * vocab_size > MAX_LOSS_LOGITS_ELEMENTS:
        raise too_large("batch_size x max_seq_length x vocab exceeds the memory limit")


def cmd_train(a: argparse.Namespace) -> int:
    """`train`: 学習 -> validation 採点 -> 出力一式（REQ-41・TASK-41.1-5）。

    入力の読み込み・検証（定義・tokenizer・データ・全列の長さ・モデル・LoRA の適用）をすべて終えて
    から出力用の一時ディレクトリを作り、全出力が揃ってから `out-dir` へ rename する（失敗時は
    一時ディレクトリを消すので、半端な出力は残らず同じ `out-dir` で再実行できる）。
    採点（validation）には学習とは別の壁時計上限（`--max-score-seconds`。超過は 20）が掛かる。
    """
    budget = Budget()
    check_out_dir(a.out_dir)  # 存在すれば重い処理の前に拒否する
    seed_all(a.seed)
    label_order, def_sha = load_label_order(a.definition)
    assets = load_assets(a.model_dir, pins_from_args(a))
    system = load_system_prompt(a.system_prompt_file, label_order)
    check_train_memory(a, assets.config.vocab_size)
    ctx = build_prompting(assets, system, label_order, assets.config.vocab_size, a.max_seq_length)
    train, train_sha = load_records(a.train, labels=set(label_order), what="train data")
    val, val_sha = load_records(a.validation, labels=None, what="validation data")
    if len(train) < a.batch_size:
        raise invalid("not enough training records for one batch")
    seqs = []
    for rec in train:
        budget.check()
        ids = ctx.train_ids(rec.input, rec.label or "")
        ctx.check_length(len(ids))
        seqs.append(ids)
    val_prompts = prepare_prompts(ctx, val, budget)  # 学習前に全件検査する
    model, sha, size = load_base_model(a.model_dir, to_dtype(a.dtype), pins_from_args(a))
    with as_input_error():
        apply_lora(
            model,
            num_layers=a.num_layers,
            rank=a.rank,
            scale=a.scale,
            dropout=a.dropout,
            seed=a.seed,  # LoRA 初期化は seed から導いた鍵（グローバル乱数に依存しない）
        )
    n_params = trainable_parameter_count(model)
    log(f"train: records={len(train)} validation={len(val)} trainable_parameters={n_params}")
    result = train_loop(
        model, seqs, pad_id=ctx.pad_id, iters=a.iters, lr=a.lr, batch_size=a.batch_size,
        seed=a.seed, budget=budget, max_wall_seconds=a.max_wall_seconds,
    )  # fmt: skip
    train_seconds = budget.elapsed()
    adapter_cfg = {
        "rank": a.rank,
        "scale": a.scale,
        "dropout": a.dropout,
        "num_layers": a.num_layers,
        "lora_init_seed": a.seed,  # lora_a の初期化鍵は mx.random.key(seed) から導く
        "keys": LORA_KEYS,
        "base_model_sha256": sha,
        "config_sha256": a.config_sha256,
        "tokenizer_sha256": a.tokenizer_sha256,
        "tokenizer_config_sha256": a.tokenizer_config_sha256,
        "dtype": a.dtype,
        "system_prompt_sha256": sha256(system.encode("utf-8")),
        "label_order": label_order,
        "max_seq_length": a.max_seq_length,
    }
    # 設定は adapters.safetensors の metadata に埋め込む（`--adapter-sha256` がファイル全体の
    # ハッシュとして重みと設定の両方を守る）。adapter_config.json は人間向けの写し。
    adapter_bytes = adapters_to_bytes(dict(tree_flatten(model.trainable_parameters())), adapter_cfg)
    adapters_sha = sha256(adapter_bytes)
    pred, raw, stats = score_records(
        model, ctx, val, val_prompts, Budget(wall_limit=a.max_score_seconds)
    )
    run = common_run(a, assets.hashes, sha, size)
    run.update(
        {
            "definition_sha256": def_sha,
            "train_data_sha256": train_sha,
            "validation_data_sha256": val_sha,
            "system_prompt_sha256": adapter_cfg["system_prompt_sha256"],
            "adapters_sha256": adapters_sha,
            "label_count": len(label_order),
            "train_records": len(train),
            "validation_records": len(val),
            "seed": a.seed,
            "iters": a.iters,
            "iters_done": result.iters_done,
            "budget_reached": result.budget_reached,
            "max_wall_seconds": a.max_wall_seconds,
            "max_score_seconds": a.max_score_seconds,
            "lr": a.lr,
            "trainable_parameters": n_params,
            "final_loss": result.final_loss,
            "train_seconds": train_seconds,
            "validation_scoring": stats,
            "elapsed_seconds": budget.elapsed(),
            "max_rss_bytes": max_rss_bytes(),  # RSS と MLX ピークの大きい方（common.max_rss_bytes）
        }
    )
    with OutputDir(a.out_dir) as out:
        out.write("adapters.safetensors", adapter_bytes)
        out.write("adapter_config.json", json_text(adapter_cfg))
        write_scores(out, pred, raw)
        out.write("run.json", json_text(run))
        out.commit()
    return 0
