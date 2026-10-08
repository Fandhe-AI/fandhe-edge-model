"""PoC-26 `predict` サブコマンド（REQ-41・TASK-41.1-5・#390）。

役割: 学習済み adapter で任意の `{id,input}` JSONL を採点する。adapter は `adapter_config.json` の
sha256・ベースモデル / config / tokenizer の sha256・ラベル順・system プロンプト・dtype・
max_seq_length を学習時の記録と照合し、重みは構造・形状・有限性を検査してから使う。1 件あたりの
採点時間を `run.json` に記録する（#393 の材料）。`cli.py` から配線される。
"""

from __future__ import annotations

import argparse

import mlx.core as mx
from mlx.utils import tree_flatten
from tools.poc26.assets import (
    load_assets,
    load_base_model,
    load_system_prompt,
    pins_from_args,
)
from tools.poc26.common import (
    Budget,
    as_input_error,
    invalid,
    max_rss_bytes,
    read_input,
    sha256,
    to_dtype,
)
from tools.poc26.io_records import (
    MAX_ADAPTER_BYTES,
    OutputDir,
    check_out_dir,
    common_run,
    json_text,
    load_adapter,
    load_label_order,
    load_records,
    read_adapter_config,
    write_scores,
)
from tools.poc26.qwen2_model import Qwen2Model, apply_lora
from tools.poc26.score import build_prompting, prepare_prompts, score_records

ADAPTER_DTYPES = (mx.float32, mx.bfloat16, mx.float16)


def attach_adapter(model: Qwen2Model, cfg: dict, weights: dict[str, mx.array]) -> None:
    """`model` へ LoRA を適用し、構造・dtype・有限性を検査した adapter 重みを float32 で載せる。

    `cmd_predict` と ONNX 書き出し（`export_onnx.py`。REQ-41・TASK-41.1-7・#392）が共有する。
    不一致は終了コード 64。
    """
    with as_input_error():
        apply_lora(
            model,
            num_layers=cfg["num_layers"],
            rank=cfg["rank"],
            scale=cfg["scale"],
            dropout=cfg["dropout"],
            seed=cfg["lora_init_seed"],
        )
    expected = dict(tree_flatten(model.trainable_parameters()))
    if weights.keys() != expected.keys() or any(
        weights[k].shape != v.shape for k, v in expected.items()
    ):
        raise invalid("adapter weights do not match the model structure")
    if any(v.dtype not in ADAPTER_DTYPES for v in weights.values()):
        raise invalid("adapter weight dtype must be float32, bfloat16 or float16")
    if not all(bool(mx.all(mx.isfinite(v)).item()) for v in weights.values()):
        raise invalid("adapter weights are not finite")
    model.load_weights([(k, v.astype(mx.float32)) for k, v in weights.items()], strict=False)
    mx.eval(model.parameters())


def cmd_predict(a: argparse.Namespace) -> int:
    """`predict`: 学習済み adapter で任意の `{id,input}` JSONL を採点する。

    adapter は `adapter_config.json` の sha256・ベースモデル・ラベル順・system プロンプト・
    dtype・max_seq_length と照合し、重みは構造・形状・dtype・有限性を検査してから使う。adapter の
    バイト列は必須引数 `--adapter-sha256` と `adapter_config.json` の記録値の両方と照合する。入力の
    検証をすべて終えてから出力用の一時ディレクトリを作り、全出力が揃ってから rename する（失敗時は
    残さない）。採点には壁時計上限（`--max-score-seconds`。超過は 20。モデル読み込みは対象外）。
    """
    budget = Budget()
    check_out_dir(a.out_dir)
    # adapter は検証済み fd から読んだバイト列の sha256 を `--adapter-sha256` と照合する。設定
    # （rank・scale・sha256 群など）は同じバイト列の metadata が正本で、adapter_config.json は
    # 人間向けの写し（一致しなければ 64）。ファイル全体のハッシュなので設定も完全性で守られる。
    adapter_bytes = read_input(
        a.adapter_dir / "adapters.safetensors", MAX_ADAPTER_BYTES, "adapters"
    )
    if sha256(adapter_bytes) != a.adapter_sha256:
        raise invalid("adapters.safetensors does not match the expected sha256")
    weights, cfg = load_adapter(adapter_bytes)  # sha256 を照合したのと同じバイト列
    del adapter_bytes
    if read_adapter_config(a.adapter_dir / "adapter_config.json") != cfg:
        raise invalid("adapter_config.json does not match the adapters.safetensors metadata")
    if cfg["dtype"] != a.dtype or cfg["max_seq_length"] != a.max_seq_length:
        raise invalid("dtype / max-seq-length differ from the values the adapter was trained with")
    label_order, def_sha = load_label_order(a.definition)
    assets = load_assets(a.model_dir, pins_from_args(a))
    system = load_system_prompt(a.system_prompt_file, label_order)
    if cfg["label_order"] != label_order:
        raise invalid("adapter was trained with a different label order")
    if cfg["system_prompt_sha256"] != sha256(system.encode("utf-8")):
        raise invalid("adapter was trained with a different system prompt")
    ctx = build_prompting(assets, system, label_order, assets.config.vocab_size, a.max_seq_length)
    records, in_sha = load_records(a.input, labels=None, what="input data")
    if not 0 <= a.warmup < len(records):  # モデル読み込み前に弾く（統計が空になる指定）
        raise invalid("warmup must be at least 0 and less than the number of records")
    prompts = prepare_prompts(ctx, records, budget)
    model, sha, size = load_base_model(a.model_dir, to_dtype(a.dtype), pins_from_args(a))
    if cfg["base_model_sha256"] != sha:
        raise invalid("adapter was trained on a different base model")
    if (cfg["config_sha256"], cfg["tokenizer_sha256"], cfg["tokenizer_config_sha256"]) != (
        a.config_sha256,
        a.tokenizer_sha256,
        a.tokenizer_config_sha256,
    ):
        raise invalid("adapter was trained with a different config / tokenizer")
    attach_adapter(model, cfg, weights)
    pred, raw, stats = score_records(
        model, ctx, records, prompts, Budget(wall_limit=a.max_score_seconds), warmup=a.warmup
    )
    run = common_run(a, assets.hashes, sha, size)
    run.update(
        {
            "definition_sha256": def_sha,
            "input_data_sha256": in_sha,
            "adapter_config": cfg,
            "adapters_sha256": a.adapter_sha256,
            "max_score_seconds": a.max_score_seconds,
            "records": len(records),
            "scoring": stats,
            "elapsed_seconds": budget.elapsed(),
            "max_rss_bytes": max_rss_bytes(),
            "mlx_peak_memory_bytes": int(mx.get_peak_memory()),  # #393 の併記用
        }
    )
    with OutputDir(a.out_dir) as out:
        write_scores(out, pred, raw)
        out.write("run.json", json_text(run))
        out.commit()
    return 0
