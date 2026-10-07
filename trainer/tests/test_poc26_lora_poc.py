"""PoC-26 学習・採点 CLI（`lora_poc.py`）の検査（REQ-41・TASK-41.1-5・#390。テストハーネス・CPU）。

極小の合成モデル・合成 tokenizer・合成データ（前方一致するラベル対を含む）で `main([...])` を
完走させる。実重み・実データは使わない（実機の確認手順は追補 1）。
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.poc26 import assets as assets_mod
from tools.poc26 import common, io_records, qwen2_model, synthetic
from tools.poc26 import probe as probe_mod
from tools.poc26 import score as score_mod
from tools.poc26 import train as train_mod
from tools.poc26.assets import build_system_prompt, load_assets
from tools.poc26.lora_poc import main
from tools.poc26.score import build_prompting, score_labels
from tools.poc26.train import batch_loss, collate, make_batches

from fandhe_edge_trainer.errors import WorkerError

LABELS = [
    "browser_navigate",
    "browser_navigate_back",
    "browser_network_request",
    "browser_network_requests",
    "none",
]
SECRET = "SECRETBODY"  # noqa: S105 (テスト本文の目印で秘密情報ではない)
ROOT = Path(__file__).resolve().parents[2]


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def make_dir(tmp: Path, *, config_patch: dict | None = None) -> Path:
    """合成モデル・tokenizer・tokenizer_config を `tmp/model` に書く。"""
    d = tmp / "model"
    d.mkdir()
    synthetic.write_model_dir(d)
    # 合成モデルの max_position_embeddings（64）は CLI の prompt 長より短いため広げる
    cfg_path = d / "config.json"
    cfg_path.write_text(
        json.dumps({**json.loads(cfg_path.read_text()), "max_position_embeddings": 2048})
    )
    synthetic.write_tokenizer_json(d)
    cfg = {
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
        "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
    }
    cfg.update(config_patch or {})
    (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def make_data(tmp: Path) -> dict[str, Path]:
    definition = {"options": [{"id": x} for x in LABELS]}
    (tmp / "definition.json").write_text(json.dumps(definition), encoding="utf-8")
    train = [
        {
            "id": f"t{i}",
            "input": f"{SECRET} {i} " + "x" * (i % 3),
            "output": {"intent": LABELS[i % len(LABELS)]},
            "group_id": f"g{i}",
        }
        for i in range(10)
    ]
    val = [{"id": f"secret-id-{i}", "input": f"{SECRET} v{i}"} for i in range(4)]
    return {
        "definition": tmp / "definition.json",
        "train": _jsonl(tmp / "train.jsonl", train),
        "validation": _jsonl(tmp / "val.jsonl", val),
    }


def _sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin_values(model: Path) -> dict[str, str]:
    """#387 で記録する想定の期待 sha256（合成モデルの実ファイルから計算する）。"""
    return {
        "model": _sha_of(model / "model.safetensors"),
        "config": _sha_of(model / "config.json"),
        "tokenizer": _sha_of(model / "tokenizer.json"),
        "tokenizer-config": _sha_of(model / "tokenizer_config.json"),
    }


def pin_flags(model: Path) -> list[str]:
    return [x for k, v in pin_values(model).items() for x in (f"--{k}-sha256", v)]


def pins(model: Path) -> assets_mod.Pins:
    v = pin_values(model)
    return assets_mod.Pins(v["model"], v["config"], v["tokenizer"], v["tokenizer-config"])


def train_args(
    model: Path, data: dict[str, Path], out: Path, seed: int = 0, extra: list[str] | None = None
) -> list[str]:
    return [
        "train",
        *["--model-dir", str(model), "--definition", str(data["definition"])],
        *["--train", str(data["train"]), "--validation", str(data["validation"])],
        *["--out-dir", str(out), "--seed", str(seed), "--iters", "3", "--lr", "1e-3"],
        *["--batch-size", "2", "--num-layers", "2", "--rank", "2", "--max-seq-length", "1024"],
        *["--device", "cpu", "--dtype", "float32", "--evidence", "test_harness"],
        *pin_flags(model),
        *(extra or []),
    ]


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("lora")
    model, data = make_dir(tmp), make_data(tmp)
    outs = [tmp / "out_a", tmp / "out_b", tmp / "out_c"]
    assert main(train_args(model, data, outs[0], seed=0)) == 0
    assert main(train_args(model, data, outs[1], seed=0)) == 0
    assert main(train_args(model, data, outs[2], seed=1)) == 0
    return {"tmp": tmp, "model": model, "data": data, "outs": outs}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pred_jsonl_shape(env: dict) -> None:
    """REQ-41・REQ-33: pred.jsonl は {id,status,predicted_label,scores} のみ・scores 合計 1。"""
    lines = (env["outs"][0] / "pred.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert list(rec) == ["id", "status", "predicted_label", "scores"]
        assert rec["id"] == f"secret-id-{i}"
        assert rec["status"] == "ok"
        assert list(rec["scores"]) == LABELS
        assert abs(sum(rec["scores"].values()) - 1.0) < 1e-6
        best = max(rec["scores"].values())
        assert rec["predicted_label"] == next(k for k in LABELS if rec["scores"][k] == best)


def test_raw_scores_consistent_with_scores(env: dict) -> None:
    """REQ-41: raw_scores の対数尤度の softmax が scores に一致する（長さ正規化なし）。"""
    pred = [json.loads(x) for x in (env["outs"][0] / "pred.jsonl").read_text().splitlines()]
    raw = [json.loads(x) for x in (env["outs"][0] / "raw_scores.jsonl").read_text().splitlines()]
    for p, r in zip(pred, raw, strict=True):
        assert set(r) == {"id", "loglik"}
        assert r["id"] == p["id"]
        v = np.array([r["loglik"][k] for k in LABELS], dtype=np.float64)
        e = np.exp(v - v.max())
        got = np.array([p["scores"][k] for k in LABELS])
        assert np.allclose(e / e.sum(), got, atol=1e-9)


def test_run_json_required_fields(env: dict) -> None:
    """REQ-41・REQ-26: run.json の必須項目と証拠の種別。"""
    run = json.loads((env["outs"][0] / "run.json").read_text(encoding="utf-8"))
    for key in (
        "evidence",
        "seed",
        "mlx_version",
        "device",
        "dtype",
        "model_safetensors_bytes",
        "model_safetensors_sha256",
        "tokenizer_json_sha256",
        "tokenizer_config_sha256",
        "chat_template_sha256",
        "trainable_parameters",
        "iters",
        "final_loss",
        "elapsed_seconds",
        "max_rss_bytes",
        "truncated_count",
        "validation_scoring",
        "system_prompt_sha256",
    ):
        assert key in run, key
    assert run["evidence"] == "test_harness"
    assert run["truncated_count"] == 0
    assert run["seed"] == 0
    assert run["iters"] == 3
    assert math.isfinite(run["final_loss"])
    # 2 層 x 7 Linear: rank 2 x (in + out) の総和（hidden 16・kv 8・intermediate 32）
    dims = [(16, 16), (16, 8), (16, 8), (16, 16), (16, 32), (16, 32), (32, 16)]
    assert run["trainable_parameters"] == 2 * sum(2 * (i + o) for i, o in dims)
    assert run["validation_scoring"]["count"] == 4
    assert run["validation_scoring"]["p95_seconds"] > 0
    assert (env["outs"][0] / "adapter_config.json").is_file()


def test_same_seed_is_deterministic_on_cpu(env: dict) -> None:
    """REQ-26・evaluation-contract: seed 0 を 2 回 -> pred.jsonl・adapter の sha256 が一致。"""
    a, b, c = env["outs"]
    for name in ("pred.jsonl", "adapters.safetensors", "raw_scores.jsonl"):
        assert _sha(a / name) == _sha(b / name), name
    assert _sha(a / "adapters.safetensors") != _sha(c / "adapters.safetensors")


def test_predict_reproduces_validation(env: dict, tmp_path: Path) -> None:
    """REQ-41: 保存した adapter を predict で読み直すと学習直後の採点と同じ結果になる。"""
    out = tmp_path / "pred_out"
    argv = [
        "predict",
        *["--model-dir", str(env["model"]), "--definition", str(env["data"]["definition"])],
        *["--adapter-dir", str(env["outs"][0]), "--input", str(env["data"]["validation"])],
        *["--out-dir", str(out), "--max-seq-length", "1024", "--device", "cpu"],
        *["--dtype", "float32", "--evidence", "test_harness"],
        *pin_flags(env["model"]),
        *["--adapter-sha256", _sha(env["outs"][0] / "adapters.safetensors")],
    ]
    assert main(argv) == 0
    want = [json.loads(x) for x in (env["outs"][0] / "pred.jsonl").read_text().splitlines()]
    got = [json.loads(x) for x in (out / "pred.jsonl").read_text().splitlines()]
    for w, g in zip(want, got, strict=True):
        assert w["predicted_label"] == g["predicted_label"]
        assert all(abs(w["scores"][k] - g["scores"][k]) < 1e-6 for k in LABELS)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["scoring"]["choices_per_prompt"] == len(LABELS)
    # 別のラベル順・別のベースで学習した adapter は拒否する
    bad = tmp_path / "bad_adapter"
    bad.mkdir()
    cfg = json.loads((env["outs"][0] / "adapter_config.json").read_text())
    cfg["base_model_sha256"] = "0" * 64
    (bad / "adapter_config.json").write_text(json.dumps(cfg))
    (bad / "adapters.safetensors").write_bytes(
        (env["outs"][0] / "adapters.safetensors").read_bytes()
    )
    argv[argv.index("--adapter-dir") + 1] = str(bad)
    argv[argv.index("--out-dir") + 1] = str(tmp_path / "pred_bad")
    assert main(argv) == 64


@pytest.fixture
def ctx_model(env: dict):
    mx.set_default_device(mx.cpu)
    p = pins(env["model"])
    assets = load_assets(env["model"], p)
    model = qwen2_model.load_qwen2(
        env["model"], dtype=mx.float32, expected_sha256=p.model, expected_config_sha256=p.config
    )
    ctx = build_prompting(assets, build_system_prompt(LABELS), LABELS, 300, 1024)
    return ctx, model


def test_labels_end_with_im_end_and_prefix_pair(ctx_model) -> None:
    """REQ-41: 前方一致のラベル対でも `<|im_end|>` を含むので接頭辞にならない。"""
    ctx, _ = ctx_model
    end = synthetic.ID_IM_END
    assert all(ids[-1] == end for ids in ctx.label_ids)
    short, long = ctx.label_ids[0], ctx.label_ids[1]
    assert long[: len(short) - 1] == short[:-1]  # 本文は前方一致する
    assert long[: len(short)] != short  # EOS を含めると接頭辞ではない


def test_train_sequence_equals_prompt_plus_label(ctx_model) -> None:
    """REQ-41: 学習列は採点の prompt + ラベル + `<|im_end|>` + 改行と一致する。"""
    ctx, _ = ctx_model
    nl = ctx.encode("\n")
    for k, label in enumerate(LABELS):
        assert ctx.train_ids("abc", label) == ctx.prompt_ids("abc") + ctx.label_ids[k] + nl


def test_score_labels_matches_full_forward(ctx_model) -> None:
    """REQ-41: 位置選択つきの採点が、全 logits の log_softmax から手計算した値と一致する。"""
    ctx, model = ctx_model
    prompt = ctx.prompt_ids("some input")
    got = score_labels(model, prompt, ctx.label_ids, ctx.pad_id)
    for k, c in enumerate(ctx.label_ids):
        seq = [*prompt, *c]
        logits = np.array(model(mx.array([seq])).astype(mx.float32))[0].astype(np.float64)
        logp = logits - np.log(
            np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)
        )
        logp -= logits.max(-1, keepdims=True)
        want = sum(logp[len(prompt) - 1 + j, t] for j, t in enumerate(c))
        assert abs(got[k] - want) < 1e-3, k


def test_batch_loss_matches_full_forward(ctx_model) -> None:
    """REQ-41: 損失は pad 以外の全 token の cross_entropy 平均（全 logits からの手計算と一致）。"""
    ctx, model = ctx_model
    seqs = [ctx.train_ids("a", "none"), ctx.train_ids("longer input", "browser_navigate")]
    inputs, pos, tgt = collate(seqs, [0, 1], ctx.pad_id)
    got = float(batch_loss(model, inputs, pos, tgt).item())
    losses = []
    for s in seqs:
        logits = np.array(model(mx.array([s[:-1]])).astype(mx.float32))[0].astype(np.float64)
        m = logits.max(-1, keepdims=True)
        logp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
        losses += [-logp[j, s[j + 1]] for j in range(len(s) - 1)]
    assert abs(got - float(np.mean(losses))) < 1e-4


def test_make_batches_sorted_and_drops_remainder() -> None:
    """REQ-41: 長さ順に batch_size 件ずつ区切り、端数は捨てる（mlx-lm と同じ）。"""
    assert make_batches([5, 1, 3, 2, 4], 2) == [[1, 3], [2, 4]]
    assert make_batches([5], 2) == []


def test_max_seq_length_stops_without_truncation(
    env: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-39: max_seq_length 超過は切り詰めず停止（20）。本文・id は stderr に出ない。"""
    out = tmp_path / "o"
    argv = train_args(env["model"], env["data"], out, extra=["--max-seq-length", "100"])
    assert main(argv) == 20
    err = capsys.readouterr().err
    assert json.loads(err.strip().splitlines()[-1])["code"] == "limit_exceeded"
    assert SECRET not in err
    assert "secret-id" not in err
    assert not (out / "pred.jsonl").exists()


def test_stderr_has_no_body_or_id(
    env: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """security.md: 正常終了時も stderr にデータ本文・id を出さない。"""
    assert main(train_args(env["model"], env["data"], tmp_path / "o")) == 0
    err = capsys.readouterr().err
    assert "step 3/3 loss" in err
    assert SECRET not in err
    assert "secret-id" not in err


def test_refuses_existing_output(env: dict, capsys: pytest.CaptureFixture[str]) -> None:
    """既存の出力ファイルを上書きしない（64）。"""
    out = env["outs"][0]
    assert main(train_args(env["model"], env["data"], out)) == 64


@pytest.mark.parametrize(
    "patch",
    [
        {"chat_template": None},
        {"chat_template": "no markers"},
        {"eos_token": "<|endoftext|>"},
        {"pad_token": "x"},
    ],
)
def test_chat_template_and_special_tokens_checked(tmp_path: Path, patch: dict) -> None:
    """tokenizer_config.json の chat_template・eos/pad が想定と違えば停止（64）。"""
    model = make_dir(tmp_path, config_patch=patch)
    data = make_data(tmp_path)
    assert main(train_args(model, data, tmp_path / "o")) == 64


def test_chat_template_sha256_is_recorded(env: dict) -> None:
    """chat_template の sha256 を run.json に記録する（既知値との照合は実機確認。追補 1）。"""
    run = json.loads((env["outs"][0] / "run.json").read_text(encoding="utf-8"))
    cfg = json.loads((env["model"] / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert run["chat_template_sha256"] == hashlib.sha256(cfg["chat_template"].encode()).hexdigest()


def _bad_data_cases(tmp: Path) -> dict[str, bytes]:
    good = json.dumps({"id": "a", "input": "x", "output": {"intent": "none"}}).encode()
    return {
        "label_outside": json.dumps(
            {"id": "a", "input": "x", "output": {"intent": "nope"}}
        ).encode(),
        "duplicate_id": good + b"\n" + good,
        "nan_literal": b'{"id":"a","input":"x","output":{"intent":"none"},"n":NaN}',
        "not_utf8": b'{"id":"a","input":"\xff","output":{"intent":"none"}}',
        "no_records": b"\n",
        "missing_input": b'{"id":"a","output":{"intent":"none"}}',
    }


@pytest.mark.parametrize(
    "case",
    ["label_outside", "duplicate_id", "nan_literal", "not_utf8", "no_records", "missing_input"],
)
def test_invalid_training_data_is_rejected(env: dict, tmp_path: Path, case: str) -> None:
    """REQ-39: 不正な学習データは 64 で停止する。"""
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(_bad_data_cases(tmp_path)[case])
    data = {**env["data"], "train": bad}
    assert main(train_args(env["model"], data, tmp_path / "o")) == 64


def test_oversized_line_and_symlink(env: dict, tmp_path: Path) -> None:
    """REQ-39: 行サイズ超過は 20、symlink の入力は 64。"""
    big = tmp_path / "big.jsonl"
    row = {"id": "a", "input": "x" * (70 * 1024), "output": {"intent": "none"}}
    big.write_text(json.dumps(row) + "\n", encoding="utf-8")
    data = {**env["data"], "train": big}
    assert main(train_args(env["model"], data, tmp_path / "o1")) == 20
    link = tmp_path / "link.jsonl"
    os.symlink(env["data"]["train"], link)
    data = {**env["data"], "train": link}
    assert main(train_args(env["model"], data, tmp_path / "o2")) == 64


def test_argument_errors_map_to_64(env: dict, tmp_path: Path) -> None:
    """REQ-21: 引数エラー（argparse の 2 ではなく）・範囲外は 64。"""
    base = train_args(env["model"], env["data"], tmp_path / "o")
    assert main(["train"]) == 64
    assert main([*base[: -0 or None], "--bogus"]) == 64
    for flag, value in (("--seed", "-1"), ("--iters", "0"), ("--lr", "nan"), ("--batch-size", "0")):
        argv = [*base]
        argv[argv.index(flag) + 1] = value
        assert main(argv) == 64, flag


def test_duplicate_definition_ids_rejected(env: dict, tmp_path: Path) -> None:
    """REQ-39: 定義の options の id 重複は 64。"""
    d = tmp_path / "definition.json"
    d.write_text(json.dumps({"options": [{"id": "a"}, {"id": "a"}]}))
    data = {**env["data"], "definition": d}
    assert main(train_args(env["model"], data, tmp_path / "o")) == 64


def test_system_prompt_lists_labels_in_order() -> None:
    """REQ-41: system プロンプトはラベルを宣言順に列挙する。"""
    text = build_system_prompt(["b", "a"])
    assert text.endswith("Names:\n- b\n- a")


def test_addendum_records_the_exact_system_prompt() -> None:
    """REQ-41: 追補 1 に載せた system プロンプト全文が実装の出力と一致する（記録の乖離を防ぐ）。"""
    prereg = (ROOT / "docs/design/poc26-preregistration.md").read_text(encoding="utf-8")
    addendum = (ROOT / "docs/design/poc26-preregistration-addendum-1.md").read_text(
        encoding="utf-8"
    )
    labels = _prereg_labels(prereg)
    assert len(labels) == 26
    assert labels[-1] == "none"
    assert build_system_prompt(labels) in addendum


def _prereg_labels(text: str) -> list[str]:
    import re

    def names(row: str) -> list[str]:
        return re.findall(r"`([a-z_]+)`", row)

    rows = [
        x
        for x in text.splitlines()
        if x.startswith("| Core automation（24）")
        or x.startswith("| Tab management（1）")
        or x.startswith("| 対象外（1）")
    ]
    return [n for r in rows for n in names(r.split("|")[2])]


def test_probe_and_compare(env: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-41: probe の出力を compare-probe が照合する（一致 0・logits 差 / id 差で 10）。"""
    out = tmp_path / "probe.json"
    argv = ["probe", "--model-dir", str(env["model"]), "--out", str(out), "--device", "cpu"]
    argv += pin_flags(env["model"])
    assert main([*argv, "--dtype", "float32", "--evidence", "test_harness"]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["cases"]) == 20
    assert all(c["roundtrip_ok"] for c in doc["cases"])
    assert doc["evidence"] == "test_harness"
    assert len(doc["greedy"]) == 3
    capsys.readouterr()
    assert main(["compare-probe", str(out), str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "match"

    def variant(name: str, mutate) -> Path:
        d = json.loads(out.read_text(encoding="utf-8"))
        mutate(d)
        p = tmp_path / name
        p.write_text(json.dumps(d), encoding="utf-8")
        return p

    def bump(d: dict) -> None:
        arr = np.frombuffer(base64.b64decode(d["cases"][0]["last_logits_f32_b64"]), "<f4").copy()
        arr[0] += 1.0
        d["cases"][0]["last_logits_f32_b64"] = base64.b64encode(arr.tobytes()).decode()

    near = variant("near.json", lambda d: None)
    assert main(["compare-probe", str(out), str(near)]) == 0
    assert main(["compare-probe", str(out), str(variant("bump.json", bump))]) == 10
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["logits_mismatches"] == 1
    ids = variant("ids.json", lambda d: d["cases"][0]["ids"].append(1))
    assert main(["compare-probe", str(out), str(ids)]) == 10
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["id_mismatches"] == 1
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    assert main(["compare-probe", str(out), str(empty)]) == 64


def test_bf16_train_smoke_on_cpu(env: dict, tmp_path: Path) -> None:
    """REQ-41: 既定の bf16（LoRA は float32）でも CPU で完走し、有限の確率を返す。"""
    argv = train_args(env["model"], env["data"], tmp_path / "o")
    argv[argv.index("--dtype") + 1] = "bf16"
    assert main(argv) == 0
    for line in (tmp_path / "o" / "pred.jsonl").read_text().splitlines():
        assert json.loads(line)["status"] == "ok"


# ---- 以下: レビュー・監査指摘（#390 PR-C）への追加テスト --------------------------------------


def _predict_argv(
    env: dict, adapter_dir: Path, out: Path, *extra: str, adapter_sha: str | None = None
) -> list[str]:
    sha = adapter_sha or _sha(adapter_dir / "adapters.safetensors")
    return [
        "predict",
        *["--model-dir", str(env["model"]), "--definition", str(env["data"]["definition"])],
        *["--adapter-dir", str(adapter_dir), "--input", str(env["data"]["validation"])],
        *["--out-dir", str(out), "--max-seq-length", "1024", "--device", "cpu"],
        *["--dtype", "float32", "--evidence", "test_harness", *pin_flags(env["model"]), *extra],
        *["--adapter-sha256", sha],
    ]


def _copy_adapter(env: dict, dst: Path) -> Path:
    dst.mkdir()
    for name in ("adapter_config.json", "adapters.safetensors"):
        (dst / name).write_bytes((env["outs"][0] / name).read_bytes())
    return dst


def _rebuild_adapter(env: dict, dst: Path, weights_fn=None, cfg_fn=None) -> str:
    """adapter を metadata つきで作り直し（重み・設定を加工可）、adapter_config.json も更新して
    新しいファイル全体の sha256 を返す。"""
    d = _copy_adapter(env, dst)
    w, meta = mx.load(str(d / "adapters.safetensors"), return_metadata=True)
    cfg = json.loads(meta[io_records.ADAPTER_METADATA_KEY])
    if weights_fn:
        w = weights_fn(w)
    if cfg_fn:
        cfg_fn(cfg)
    (d / "adapters.safetensors").unlink()
    (d / "adapters.safetensors").write_bytes(io_records.adapters_to_bytes(w, cfg))
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    return _sha(d / "adapters.safetensors")


def test_adapter_sha256_recorded_and_checked(env: dict, tmp_path: Path) -> None:
    """REQ-39: adapters の sha256 を記録・照合する。dtype・max_seq_length 不一致は 64。"""
    cfg = json.loads((env["outs"][0] / "adapter_config.json").read_text())
    run = json.loads((env["outs"][0] / "run.json").read_text())
    want = _sha(env["outs"][0] / "adapters.safetensors")
    assert (
        "adapters_sha256" not in cfg
    )  # ファイル全体のハッシュは --adapter-sha256（設定は metadata）
    assert run["adapters_sha256"] == want
    tampered = _copy_adapter(env, tmp_path / "t")
    with (tampered / "adapters.safetensors").open("ab") as f:
        f.write(b"\0")
    assert main(_predict_argv(env, tampered, tmp_path / "o1", adapter_sha=want)) == 64
    ok = _copy_adapter(env, tmp_path / "ok")
    argv = _predict_argv(env, ok, tmp_path / "o2")
    argv[argv.index("--dtype") + 1] = "bf16"
    assert main(argv) == 64
    argv = _predict_argv(env, ok, tmp_path / "o3")
    argv[argv.index("--max-seq-length") + 1] = "512"
    assert main(argv) == 64
    assert not (tmp_path / "o1").exists()


def test_predict_rejects_non_finite_weights(
    env: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-39: 非有限の adapter 重み（sha256・metadata は一致）は拒否する。"""

    def nan_weights(w: dict) -> dict:
        key = next(iter(w))
        return {**w, key: mx.full(w[key].shape, float("nan"))}

    sha = _rebuild_adapter(env, tmp_path / "nan", weights_fn=nan_weights)
    assert main(_predict_argv(env, tmp_path / "nan", tmp_path / "o", adapter_sha=sha)) == 64
    assert "not finite" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_adapter_config_extra_keys_are_dropped(env: dict, tmp_path: Path) -> None:
    """余分なキーは検証対象外として捨て、run.json にも書かない。"""
    d = _copy_adapter(env, tmp_path / "x")
    cfg = json.loads((d / "adapter_config.json").read_text())
    cfg["zzz_unknown"] = "x"
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    assert main(_predict_argv(env, d, tmp_path / "o")) == 0
    run = json.loads((tmp_path / "o" / "run.json").read_text())
    assert "zzz_unknown" not in run["adapter_config"]
    assert set(run["adapter_config"]) == set(io_records.ADAPTER_CONFIG_KEYS)


def test_run_json_args_use_basenames_only(env: dict) -> None:
    """run.json の args のパスはファイル名だけ（ディレクトリ名を残さない）。"""
    run = json.loads((env["outs"][0] / "run.json").read_text())
    for k in ("model_dir", "definition", "train", "validation", "out_dir"):
        assert "/" not in run["args"][k], k
    assert "func" not in run["args"]


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--rank", "0"),
        ("--rank", "100000"),
        ("--num-layers", "0"),
        ("--num-layers", "999"),
        ("--scale", "nan"),
        ("--scale", "0"),
        ("--dropout", "1"),
        ("--dropout", "-0.1"),
        ("--max-wall-seconds", "0"),
        ("--max-wall-seconds", "86401"),
        ("--num-layers", "5"),  # モデルは 2 層: apply_lora が拒否する
    ],
)
def test_lora_arguments_are_range_checked(env: dict, tmp_path: Path, flag: str, value: str) -> None:
    """REQ-39: LoRA の rank・層数・scale・dropout・予算は範囲検査し、出力ディレクトリを作らない。"""
    out = tmp_path / "o"
    argv = train_args(env["model"], env["data"], out)
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]
    assert main(argv) == 64
    assert not out.exists()


def test_max_wall_seconds_flag_accepted(env: dict, tmp_path: Path) -> None:
    """`--max-wall-seconds` は train で指定でき、run.json に記録される。"""
    argv = train_args(
        env["model"], env["data"], tmp_path / "o", extra=["--max-wall-seconds", "7200"]
    )
    assert main(argv) == 0
    run = json.loads((tmp_path / "o" / "run.json").read_text())
    assert run["max_wall_seconds"] == 7200
    assert run["budget_reached"] is False
    assert run["iters_done"] == run["iters"] == 3


def test_budget_reached_stops_training_and_continues(
    env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """事前登録 4 節: 壁時計の予算に達したら打ち切り、adapter を保存し記録して採点へ進む。"""
    calls = {"n": 0}

    def exceeded(self: common.Budget, limit: float) -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    monkeypatch.setattr(common.Budget, "wall_exceeded", exceeded)
    out = tmp_path / "o"
    argv = train_args(env["model"], env["data"], out)
    argv[argv.index("--iters") + 1] = "6"
    assert main(argv) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["budget_reached"] is True
    assert run["iters"] == 6
    assert run["iters_done"] == 2
    assert (out / "adapters.safetensors").is_file()
    assert len((out / "pred.jsonl").read_text().splitlines()) == 4


def test_invalid_input_leaves_no_output_dir(env: dict, tmp_path: Path) -> None:
    """入力の検証を終えてから出力ディレクトリを作る（失敗で空のディレクトリを残さない）。"""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id":"a","input":"x","output":{"intent":"nope"}}\n')
    out = tmp_path / "o1"
    assert main(train_args(env["model"], {**env["data"], "train": bad}, out)) == 64
    assert not out.exists()
    # validation の長さ超過も学習前に検出する（train は収まる）
    long_val = _jsonl(tmp_path / "v.jsonl", [{"id": "v", "input": "y" * 3000}])
    out2 = tmp_path / "o2"
    argv = train_args(env["model"], {**env["data"], "validation": long_val}, out2)
    assert main(argv) == 20
    assert not out2.exists()


def test_memory_estimates_are_enforced(
    env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: 学習の logits 要素数・採点の K x 系列長の上限超過は 20。"""
    monkeypatch.setattr(train_mod, "MAX_LOSS_LOGITS_ELEMENTS", 1000)
    out = tmp_path / "o1"
    assert main(train_args(env["model"], env["data"], out)) == 20
    assert not out.exists()
    monkeypatch.undo()
    monkeypatch.setattr(score_mod, "MAX_SCORE_TOKENS", 100)
    out2 = tmp_path / "o2"
    assert main(train_args(env["model"], env["data"], out2)) == 20
    assert not out2.exists()


def test_score_labels_chunking_is_equivalent(ctx_model) -> None:
    """REQ-41: ラベルをチャンクに分けて forward しても、分割なしと同じ値になる。"""
    ctx, model = ctx_model
    prompt = ctx.prompt_ids("chunk input")
    whole = score_labels(model, prompt, ctx.label_ids, ctx.pad_id, chunk=100)
    for chunk in (1, 2, 3):
        got = score_labels(model, prompt, ctx.label_ids, ctx.pad_id, chunk=chunk)
        assert np.allclose(got, whole, atol=1e-4), chunk


def test_non_finite_scores_become_error_without_scores(
    ctx_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-21: 非有限スコアは status:"error"・scores なし・predicted_label null。raw は null。"""
    ctx, model = ctx_model
    bad = np.array([0.0, float("nan"), -1.0, -2.0, -3.0])
    monkeypatch.setattr(score_mod, "score_labels", lambda *a, **k: bad)
    recs = [io_records.Record("r1", "abc", None)]
    prompts = score_mod.prepare_prompts(ctx, recs, common.Budget())
    pred, raw, _ = score_mod.score_records(model, ctx, recs, prompts, common.Budget())
    rec = json.loads(pred[0])
    assert rec == {"id": "r1", "status": "error", "predicted_label": None}
    assert json.loads(raw[0])["loglik"][LABELS[1]] is None


def _seqs(ctx) -> list[list[int]]:
    return [ctx.train_ids(f"in {i} " + "x" * i, LABELS[i % 5]) for i in range(12)]


def _capture_order(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    seen: list[tuple[int, ...]] = []
    real = train_mod.collate

    def spy(seqs, idxs, pad_id):
        seen.append(tuple(idxs))
        return real(seqs, idxs, pad_id)

    monkeypatch.setattr(train_mod, "collate", spy)
    return seen


def _fresh_lora_model(env: dict):
    p = pins(env["model"])
    model = qwen2_model.load_qwen2(
        env["model"], dtype=mx.float32, expected_sha256=p.model, expected_config_sha256=p.config
    )
    qwen2_model.apply_lora(model, num_layers=2, rank=2, scale=20.0, dropout=0.0, seed=0)
    return model


def test_batch_order_depends_on_seed_and_reshuffles(
    env: dict, ctx_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-26: バッチ順は seed で決まる。iters がバッチ数を超えると再シャッフルする。"""
    ctx, _ = ctx_model
    seqs = _seqs(ctx)
    n_batches = len(train_mod.make_batches([len(s) for s in seqs], 2))
    assert n_batches == 6
    runs = {}
    for seed in (0, 0, 1):
        seen = _capture_order(monkeypatch)
        mx.set_default_device(mx.cpu)
        train_mod.train_loop(
            _fresh_lora_model(env), seqs, pad_id=ctx.pad_id, iters=2 * n_batches + 1, lr=1e-3,
            batch_size=2, seed=seed, budget=common.Budget(),
        )  # fmt: skip
        runs.setdefault(seed, []).append(list(seen))
    assert runs[0][0] == runs[0][1]
    assert runs[0][0] != runs[1][0]
    order = runs[0][0]
    assert len(order) == 2 * n_batches + 1
    first, second = order[:n_batches], order[n_batches : 2 * n_batches]
    assert sorted(first) == sorted(second)  # 各エポックは全バッチを 1 回ずつ
    assert len(set(first)) == n_batches
    assert first != second  # 新しい permutation（この seed では順序が変わる）


def test_training_updates_only_lora_parameters(env: dict, ctx_model) -> None:
    """REQ-41: 学習後、base の重みは不変で LoRA だけが更新される（lora_b は 0 から動く）。"""
    from mlx.utils import tree_flatten

    ctx, _ = ctx_model
    model = _fresh_lora_model(env)
    before = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    train_mod.train_loop(
        model, _seqs(ctx), pad_id=ctx.pad_id, iters=3, lr=1e-2, batch_size=2, seed=0,
        budget=common.Budget(),
    )  # fmt: skip
    after = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    assert before.keys() == after.keys()
    base_keys = [k for k in before if "lora_" not in k]
    assert base_keys
    for k in base_keys:
        assert np.array_equal(before[k], after[k]), k
    lora_b = [k for k in before if k.endswith("lora_b")]
    assert lora_b
    assert all(np.all(before[k] == 0) for k in lora_b)
    assert any(np.any(after[k] != 0) for k in lora_b)


def test_limit_exceeded_is_a_dedicated_type() -> None:
    """上限超過（20）は専用の例外型で判別し、文言には依存しない。"""
    from tools.poc26.safe_io import LimitExceededError

    from fandhe_edge_trainer.exitcode import ExitCode

    with pytest.raises(WorkerError) as exc:
        with common.as_input_error():
            raise LimitExceededError("anything")
    assert exc.value.exit_code == ExitCode.LIMIT_EXCEEDED
    with pytest.raises(WorkerError) as exc2:
        with common.as_input_error():
            raise ValueError("this file is too large")  # 文言が似ていても 64
    assert exc2.value.exit_code == ExitCode.INVALID_INPUT


def test_output_files_are_created_exclusively(tmp_path: Path) -> None:
    """既存ファイルは上書きしない（排他作成）。同じ名前を 2 回書くと 64。"""
    target = tmp_path / "out"
    with io_records.OutputDir(target) as out:
        out.write("a.txt", "x")
        with pytest.raises(WorkerError):
            out.write("a.txt", "y")
        out.commit()
    assert (target / "a.txt").read_text() == "x"


def test_seed_all_seeds_three_systems() -> None:
    """REQ-26: random・numpy グローバル・mlx を seed する。"""
    import random

    common.seed_all(5)
    a = (random.random(), float(np.random.rand()), float(mx.random.uniform().item()))  # noqa: S311
    common.seed_all(5)
    b = (random.random(), float(np.random.rand()), float(mx.random.uniform().item()))  # noqa: S311
    assert a == b


@pytest.mark.parametrize("which", ["model", "config", "tokenizer", "tokenizer-config"])
def test_sha256_pins_are_enforced(env: dict, tmp_path: Path, which: str) -> None:
    """REQ-39: 期待 sha256（#387 の記録）と一致しなければ 64。出力ディレクトリは作らない。"""
    out = tmp_path / "o"
    argv = train_args(env["model"], env["data"], out)
    argv[argv.index(f"--{which}-sha256") + 1] = "0" * 64
    assert main(argv) == 64
    assert not out.exists()


def test_sha256_pins_are_required_and_validated(env: dict, tmp_path: Path) -> None:
    """期待 sha256 は必須引数で、64 桁の小文字 16 進だけを受理する（64）。"""
    base = train_args(env["model"], env["data"], tmp_path / "o")
    k = base.index("--model-sha256")
    assert main(base[:k] + base[k + 2 :]) == 64  # 欠落
    for bad in ("abc", "A" * 64, "g" * 64):
        argv = [*base]
        argv[k + 1] = bad
        assert main(argv) == 64


def test_pins_are_recorded_and_checked_by_predict(env: dict, tmp_path: Path) -> None:
    """adapter_config・run.json に pin を記録し、predict は記録値とも照合する。"""
    v = pin_values(env["model"])
    cfg = json.loads((env["outs"][0] / "adapter_config.json").read_text())
    run = json.loads((env["outs"][0] / "run.json").read_text())
    assert cfg["base_model_sha256"] == v["model"]
    assert cfg["config_sha256"] == v["config"]
    assert cfg["tokenizer_sha256"] == v["tokenizer"]
    assert cfg["tokenizer_config_sha256"] == v["tokenizer-config"]
    assert run["model_safetensors_sha256"] == v["model"]
    assert run["config_json_sha256"] == v["config"]
    assert run["tokenizer_json_sha256"] == v["tokenizer"]
    assert run["sha256_pinned"] is True
    sha = _rebuild_adapter(
        env, tmp_path / "a", cfg_fn=lambda c: c.update(tokenizer_sha256="1" * 64)
    )  # 学習時と別の tokenizer だったことにする（metadata ごと整合させる）
    argv = _predict_argv(env, tmp_path / "a", tmp_path / "o", adapter_sha=sha)
    assert main(argv) == 64
    assert not (tmp_path / "o").exists()


def test_load_base_model_goes_through_load_qwen2(env: dict) -> None:
    """REQ-39: 重みの読み込みは PR-B の load_qwen2（sha256 照合・形状検証）だけを通る。"""
    model, sha, size = assets_mod.load_base_model(env["model"], mx.float32, pins(env["model"]))
    path = env["model"] / "model.safetensors"
    assert sha == _sha(path)
    assert size == path.stat().st_size
    assert model.config.vocab_size == 300


def test_max_seq_length_above_model_positions_is_rejected(env: dict, tmp_path: Path) -> None:
    """REQ-39: max-seq-length が config の max_position_embeddings を超えたら 64（出力なし）。"""
    out = tmp_path / "o"
    argv = train_args(env["model"], env["data"], out)
    argv[argv.index("--max-seq-length") + 1] = "4096"
    assert main(argv) == 64
    assert not out.exists()


def test_project_and_hidden_states_share_forward_validation(
    ctx_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: hidden_states と project も forward と同じ検証を受ける（超過は LimitExceeded）。"""
    from tools.poc26.safe_io import LimitExceededError

    _, model = ctx_model
    with pytest.raises(ValueError, match="hidden"):
        model.project(mx.zeros((2, 5)), source_shape=(1, 2), with_mask=False)  # 幅が違う
    monkeypatch.setattr(qwen2_model, "MAX_FORWARD_ELEMENTS", 10)
    with pytest.raises(LimitExceededError):
        model.project(mx.zeros((4, model.config.hidden_size)), source_shape=(1, 4), with_mask=False)
    with pytest.raises(LimitExceededError):
        model(mx.zeros((1, 8), dtype=mx.int32))  # 全位置 logits は従来どおり拒否される
    monkeypatch.setattr(qwen2_model, "MAX_FORWARD_TOKENS", 4)
    with pytest.raises(LimitExceededError):
        model.hidden_states(
            mx.zeros((1, 8), dtype=mx.int32)
        )  # トークン数の上限は hidden にも掛かる


def _leftovers(parent: Path, name: str) -> list[str]:
    return sorted(p.name for p in parent.iterdir() if p.name.startswith((name, f".{name}.tmp-")))


def test_failure_during_scoring_leaves_nothing_and_rerun_works(
    env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: 採点中の失敗で out-dir も一時ディレクトリも残らず、同じ out-dir で再実行できる。"""
    out = tmp_path / "o"

    def boom(*_a, **_k):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(score_mod, "score_labels", boom)
    assert main(train_args(env["model"], env["data"], out)) == 70
    assert _leftovers(tmp_path, "o") == []
    monkeypatch.undo()
    assert main(train_args(env["model"], env["data"], out)) == 0
    assert _leftovers(tmp_path, "o") == ["o"]


def test_outputs_have_private_permissions(env: dict) -> None:
    """REQ-39: 出力ファイルは 0600、出力ディレクトリは 0700。"""
    out = env["outs"][0]
    assert (out.stat().st_mode & 0o777) == 0o700
    for p in out.iterdir():
        assert (p.stat().st_mode & 0o777) == 0o600, p.name


def test_out_dir_parent_must_exist(env: dict, tmp_path: Path) -> None:
    """out-dir の親ディレクトリが無ければ 64（mkdir -p しない）。"""
    assert main(train_args(env["model"], env["data"], tmp_path / "missing" / "o")) == 64
    assert not (tmp_path / "missing").exists()


def test_scoring_wall_budget_stops_with_limit_exceeded(
    env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: 採点の壁時計上限（--max-score-seconds）到達は 20。出力は残らない。"""
    import time

    real = score_mod.score_labels

    def slow(*a, **k):
        time.sleep(1.1)
        return real(*a, **k)

    monkeypatch.setattr(score_mod, "score_labels", slow)
    out = tmp_path / "o"
    argv = train_args(env["model"], env["data"], out, extra=["--max-score-seconds", "1"])
    assert main(argv) == 20
    assert _leftovers(tmp_path, "o") == []


@pytest.mark.parametrize("value", ["0", "86401"])
def test_max_score_seconds_is_range_checked(env: dict, tmp_path: Path, value: str) -> None:
    """--max-score-seconds は 1..86400 だけ（64）。"""
    argv = train_args(
        env["model"], env["data"], tmp_path / "o", extra=["--max-score-seconds", value]
    )
    assert main(argv) == 64


def test_argparse_errors_hide_values_and_abbreviations_are_rejected(
    env: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """引数エラーは値を出さず引数名だけ（64）。`--se` のような省略形は受理しない。"""
    argv = train_args(env["model"], env["data"], tmp_path / "o")
    argv[argv.index("--seed") + 1] = "SECRETVALUE"
    assert main(argv) == 64
    err = capsys.readouterr().err
    assert "SECRETVALUE" not in err
    assert "--seed" in err
    argv = train_args(env["model"], env["data"], tmp_path / "o2")
    argv[argv.index("--seed")] = "--se"
    assert main(argv) == 64


def test_adapter_sha256_argument_is_required_and_checked(env: dict, tmp_path: Path) -> None:
    """predict の --adapter-sha256 は必須で、読んだバイト列と照合する（不一致は 64）。"""
    argv = _predict_argv(env, env["outs"][0], tmp_path / "o1")
    k = argv.index("--adapter-sha256")
    assert main(argv[:k] + argv[k + 2 :]) == 64
    assert main(_predict_argv(env, env["outs"][0], tmp_path / "o2", adapter_sha="2" * 64)) == 64
    assert _leftovers(tmp_path, "o") == []


def test_predict_rejects_unsupported_adapter_dtype(
    env: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-39: adapter の重み dtype は float32 / bfloat16 / float16 だけ。"""
    sha = _rebuild_adapter(
        env, tmp_path / "i32", weights_fn=lambda w: {k: v.astype(mx.int32) for k, v in w.items()}
    )
    assert main(_predict_argv(env, tmp_path / "i32", tmp_path / "o", adapter_sha=sha)) == 64
    assert "dtype" in capsys.readouterr().err


def test_adapter_config_is_protected_by_the_adapter_hash(
    env: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """P1: 設定は adapters.safetensors の metadata に入り、--adapter-sha256 が守る。

    adapter_config.json（人間向けの写し）の scale だけを書き換えると metadata と不一致で 64。
    """
    d = _copy_adapter(env, tmp_path / "t")
    cfg = json.loads((d / "adapter_config.json").read_text())
    _, meta = mx.load(str(d / "adapters.safetensors"), return_metadata=True)
    assert json.loads(meta[io_records.ADAPTER_METADATA_KEY]) == cfg
    cfg["scale"] = 1000.0
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    assert main(_predict_argv(env, d, tmp_path / "o")) == 64
    assert "does not match" in capsys.readouterr().err
    # metadata の設定を書き換えたファイルは、元の --adapter-sha256 とは別のハッシュになり拒否される
    sha = _rebuild_adapter(env, tmp_path / "m", cfg_fn=lambda c: c.update(scale=1000.0))
    orig = _sha(env["outs"][0] / "adapters.safetensors")
    assert sha != orig
    assert main(_predict_argv(env, tmp_path / "m", tmp_path / "o2", adapter_sha=orig)) == 64


def test_run_json_records_verified_size_and_memory_peak(env: dict) -> None:
    """run.json のモデルサイズは照合したバイト列の長さ。メモリは RSS と MLX ピークの大きい方。"""
    run = json.loads((env["outs"][0] / "run.json").read_text())
    assert run["model_safetensors_bytes"] == (env["model"] / "model.safetensors").stat().st_size
    assert run["max_rss_bytes"] >= int(mx.get_peak_memory())
    assert run["max_score_seconds"] == 3600
    assert run["validation_scoring"]["forward_chunk"] == score_mod.SCORE_CHUNK
    assert "score_chunk" not in run["validation_scoring"]


def test_probe_output_is_exclusive_private_and_needs_parent(env: dict, tmp_path: Path) -> None:
    """probe の出力は親ディレクトリ必須・排他作成・0600。"""
    base = ["probe", "--model-dir", str(env["model"]), "--device", "cpu", "--dtype", "float32"]
    base += ["--evidence", "test_harness", *pin_flags(env["model"])]
    assert main([*base, "--out", str(tmp_path / "nope" / "p.json")]) == 64
    out = tmp_path / "p.json"
    assert main([*base, "--out", str(out)]) == 0
    assert (out.stat().st_mode & 0o777) == 0o600
    assert main([*base, "--out", str(out)]) == 64


def test_compare_probes_rejects_bool_vocab_size() -> None:
    """vocab_size は bool を受理しない（ループの外で 1 度だけ検査）。"""
    case = {"text": "a", "ids": [1], "last_logits_f32_b64": ""}
    doc = {"vocab_size": True, "cases": [case]}
    with pytest.raises(WorkerError):
        probe_mod.compare_probes(doc, doc, 1e-3)


def test_safe_decode_reraises_limit_exceeded() -> None:
    """probe.safe_decode は LimitExceededError を握りつぶさず再送出する（undecodable は別）。"""
    from tools.poc26.safe_io import LimitExceededError

    class Tok:
        def __init__(self, exc: Exception) -> None:
            self.exc = exc

        def decode(self, _ids: list[int]) -> str:
            raise self.exc

    with pytest.raises(LimitExceededError):
        probe_mod.safe_decode(Tok(LimitExceededError("x")), [1])
    assert probe_mod.safe_decode(Tok(ValueError("unknown token id")), [1]) == "<undecodable>"


def test_budget_is_checked_after_the_last_forward(
    ctx_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: 最後の forward の後に壁時計上限を超えていれば 20（forward の前だけでは見逃す）。"""
    ctx, model = ctx_model
    recs = [io_records.Record("r1", "abc", None)]
    prompts = score_mod.prepare_prompts(ctx, recs, common.Budget())
    budget = common.Budget(wall_limit=1000.0)
    calls = {"n": 0}
    real = score_mod.score_labels

    def slow_last(*a, **k):
        calls["n"] += 1
        budget.start -= 5000.0  # この forward の間に上限を超えた状況を作る
        return real(*a, **k)

    monkeypatch.setattr(score_mod, "score_labels", slow_last)
    # forward の前の check は通る（start を動かすのは forward の中）
    with pytest.raises(WorkerError) as exc:
        score_mod.score_records(model, ctx, recs, prompts, budget)
    assert calls["n"] == 1
    assert exc.value.exit_code == 20


def test_commit_refuses_existing_destination_and_never_overwrites(tmp_path: Path) -> None:
    """P1: 宛先を mkdir で排他作成してから rename する。既存・直前に作られた宛先は上書きしない。"""
    target = tmp_path / "out"

    def attempt() -> None:
        with io_records.OutputDir(target) as out:
            out.write("a.txt", "x")
            target.mkdir()  # 確定直前に他者が作った
            (target / "theirs.txt").write_text("keep")
            out.commit()

    with pytest.raises(WorkerError):
        attempt()
    assert (target / "theirs.txt").read_text() == "keep"
    assert not (target / "a.txt").exists()
    assert _leftovers(tmp_path, "out") == ["out"]  # 一時ディレクトリは消える


def test_commit_fails_if_someone_fills_the_destination_after_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: mkdir と rename の間に宛先へファイルを置かれたら rename が失敗し、上書きは起きない。"""
    target = tmp_path / "out"
    real_rename = os.rename

    def racing_rename(src, dst):
        (Path(dst) / "intruder.txt").write_text("theirs")  # mkdir 直後に割り込む
        return real_rename(src, dst)

    monkeypatch.setattr(io_records.os, "rename", racing_rename)

    def attempt() -> None:
        with io_records.OutputDir(target) as out:
            out.write("a.txt", "x")
            out.commit()

    with pytest.raises(WorkerError):
        attempt()
    assert (target / "intruder.txt").read_text() == "theirs"  # 他者のファイルはそのまま
    assert not (target / "a.txt").exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")] == []


@pytest.mark.parametrize("bad", ["!!!", "QUJD=", "é", "A"])
def test_decode_logits_maps_base64_errors_to_input_error(bad: str) -> None:
    """P1: 不正な base64（padding・非 ASCII・非 alphabet）は 64（未処理の例外にしない）。"""
    with pytest.raises(WorkerError) as exc:
        probe_mod.decode_logits(bad, 4)
    assert exc.value.exit_code == 64


def test_hidden_states_is_not_rejected_by_logits_sized_estimate(
    ctx_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: hidden_states は logits を除いた量で判定する。

    全位置 logits を含む見積もりでは拒否される大きさでも、採点（score_labels の project は必要位置の
    logits だけ）は拒否されない。`__call__`（全位置 logits）は従来どおり拒否される。
    """
    from tools.poc26.safe_io import LimitExceededError

    ctx, model = ctx_model
    prompt = ctx.prompt_ids("limit probe")
    longest = max(len(c) for c in ctx.label_ids)
    n, k = len(prompt) + longest, len(ctx.label_ids)
    without = model.forward_bytes(k, n, False, with_logits=False)
    with_logits = model.forward_bytes(k, n, False, with_logits=True)
    assert with_logits > without
    # hidden は通り、全体（logits 込み）は通らない境界に上限を置く
    monkeypatch.setattr(qwen2_model, "MAX_MODEL_MEMORY_BYTES", (without + with_logits) // 2)
    ids = mx.zeros((k, n), dtype=mx.int32)
    assert model.hidden_states(ids).shape == (k, n, model.config.hidden_size)
    with pytest.raises(LimitExceededError):
        model(ids)
    scores = score_labels(model, prompt, ctx.label_ids, ctx.pad_id, chunk=k)
    assert scores.shape == (k,)


def test_train_wall_budget_starts_at_the_training_loop(
    ctx_model, env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: 学習の壁時計予算は学習ループ開始時から数える（事前処理の時間で減らない）。"""
    ctx, _ = ctx_model
    now = {"t": 0.0}
    monkeypatch.setattr(common.time, "monotonic", lambda: now["t"])
    budget = common.Budget()
    now["t"] = 50_000.0  # 事前処理（読み込み・トークナイズ）に 50000 秒かかった状況
    model = _fresh_lora_model(env)
    result = train_mod.train_loop(
        model, _seqs(ctx), pad_id=ctx.pad_id, iters=3, lr=1e-3, batch_size=2, seed=0,
        budget=budget, max_wall_seconds=100,
    )  # fmt: skip
    assert result.iters_done == 3
    assert result.budget_reached is False
    # ループ中に学習予算を超えれば従来どおり打ち切る
    budget2 = common.Budget()
    real_collate = train_mod.collate

    def advancing(*a, **k):
        now["t"] += 60.0
        return real_collate(*a, **k)

    monkeypatch.setattr(train_mod, "collate", advancing)
    result2 = train_mod.train_loop(
        _fresh_lora_model(env), _seqs(ctx), pad_id=ctx.pad_id, iters=6, lr=1e-3,
        batch_size=2, seed=0, budget=budget2, max_wall_seconds=100,
    )  # fmt: skip
    assert result2.budget_reached is True
    assert result2.iters_done == 2


def test_compare_probes_requires_matching_vocab_sizes() -> None:
    """P1: 両側の vocab_size が存在し一致すること。不一致・欠落は入力の不整合として 64。"""
    case = {"text": "a", "ids": [1], "last_logits_f32_b64": base64.b64encode(b"\0" * 8).decode()}
    ok = {"vocab_size": 2, "cases": [case]}
    assert probe_mod.compare_probes(ok, ok, 1e-3)["status"] == "match"
    for other in ({"vocab_size": 3, "cases": [case]}, {"cases": [case]}):
        with pytest.raises(WorkerError) as exc:
            probe_mod.compare_probes(ok, other, 1e-3)
        assert exc.value.exit_code == 64
        with pytest.raises(WorkerError):
            probe_mod.compare_probes(other, ok, 1e-3)


def test_record_id_limit_is_counted_in_utf8_bytes(tmp_path: Path) -> None:
    """P1: id の上限は UTF-8 バイト数（文字数は上限内でもバイト数で超えれば拒否）。"""
    limit = io_records.MAX_PREDICTION_ID_BYTES
    ok = "あ" * (limit // 3)  # 3 バイト x 341 = 1023 バイト
    too_long = "あ" * (limit // 3 + 1)  # 文字数 342 <= 1024 だが 1026 バイト
    assert len(too_long) <= limit
    assert io_records.check_id(ok) == ok
    with pytest.raises(WorkerError) as exc:
        io_records.check_id(too_long)
    assert exc.value.exit_code == 64
    path = _jsonl(tmp_path / "v.jsonl", [{"id": too_long, "input": "x"}])
    with pytest.raises(WorkerError):
        io_records.load_records(path, labels=None, what="input data")


def test_budget_reached_is_recorded_when_exceeded_during_the_last_step(
    ctx_model, env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2: 最終ステップ中に予算を超えたら budget_reached を記録する（iters_done は全件）。"""
    ctx, _ = ctx_model
    now = {"t": 0.0}
    monkeypatch.setattr(common.time, "monotonic", lambda: now["t"])
    budget = common.Budget()
    real_collate = train_mod.collate

    def advancing(*a, **k):
        now["t"] += 60.0  # 1 ステップごとに 60 秒（3 ステップ目の途中で 150 秒を超える）
        return real_collate(*a, **k)

    monkeypatch.setattr(train_mod, "collate", advancing)
    result = train_mod.train_loop(
        _fresh_lora_model(env), _seqs(ctx), pad_id=ctx.pad_id, iters=3, lr=1e-3,
        batch_size=2, seed=0, budget=budget, max_wall_seconds=150,
    )  # fmt: skip
    assert result.iters_done == 3
    assert result.budget_reached is True


def test_project_adds_the_callers_held_hidden_and_activation_bytes(
    ctx_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: project は呼び出し元が保持する隠れ状態・活性を合算して判定する。

    hidden_states 単独・logits 単独では上限内でも、合算で超える境界では project が拒否する。
    実際の採点形状（K x 幅の hidden と必要位置の logits）では通る。
    """
    from tools.poc26.safe_io import LimitExceededError

    ctx, model = ctx_model
    c = model.config
    k, n = 8, 600  # 大きめの hidden（source_shape）
    held = model.forward_bytes(k, n, False, with_logits=False)
    rows = 4000  # 必要位置の logits
    logits_bytes = rows * c.vocab_size * 4
    # 個別には通り、合算で超える上限（held < limit, logits_bytes + params < limit, 合計 > limit）
    limit = held + logits_bytes - 1
    params = model.forward_bytes(1, 1, False, with_logits=False)
    assert limit > held
    assert limit > params + logits_bytes
    monkeypatch.setattr(qwen2_model, "MAX_MODEL_MEMORY_BYTES", limit)
    h = mx.zeros((rows, c.hidden_size))
    with pytest.raises(LimitExceededError):
        model.project(h, source_shape=(k, n), with_mask=False)
    # 小さい source_shape（保持量が小さい）なら同じ logits でも通る
    out = model.project(h, source_shape=(1, 8), with_mask=False)
    assert out.shape == (rows, c.vocab_size)
    # 実際の採点形状（chunk = K 件）は通る
    monkeypatch.undo()
    prompt = ctx.prompt_ids("shape check")
    assert score_labels(model, prompt, ctx.label_ids, ctx.pad_id).shape == (len(ctx.label_ids),)


def test_main_limits_mlx_cache_and_restores_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-39: 実行中は MLX の buffer cache を上限つきにし、終了後に元へ戻す。

    CPU では cache が溜まって RSS 上限を数ステップで超えるため。
    """
    from tools.poc26 import cli

    seen: list[int] = []

    def fake_cmd(_a: object) -> int:
        seen.append(mx.set_cache_limit(cli.CACHE_LIMIT_BYTES))
        return 0

    before = 3 * 1024 * 1024 * 1024
    original = mx.set_cache_limit(before)
    try:
        p = cli._build_parser()
        monkeypatch.setattr(cli, "_build_parser", lambda: p)
        for action in p._subparsers._group_actions[0].choices.values():
            if action.prog.endswith("compare-probe"):
                action.set_defaults(func=fake_cmd)
        assert main(["compare-probe", "a.json", "b.json"]) == 0
        assert seen == [512 * 1024 * 1024]
        assert mx.set_cache_limit(before) == before
    finally:
        mx.set_cache_limit(original)
