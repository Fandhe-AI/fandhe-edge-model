"""Qwen2 ベースモデル・トークナイザーの読み込み検査（REQ-41・REQ-39・REQ-19・TASK-41.5-2・#425）。

極小の合成モデル（`tools/poc26/synthetic.py` の config・tokenizer と、本テストで作る seed 固定の
float32 重み）で挙動を固定する。実重みは使わない。証拠種別: テストハーネス（CPU）。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.poc26 import qwen2_tokenizer as poc_tokenizer
from tools.poc26 import synthetic

from fandhe_edge_trainer import limits, qwen2_model
from fandhe_edge_trainer.budget import ResourceBudget
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode
from fandhe_edge_trainer.guard import resolve_root
from fandhe_edge_trainer.qwen2_model import (
    BaseModel,
    Qwen2Config,
    Qwen2Model,
    load_base_model,
)
from fandhe_edge_trainer.qwen2_tokenizer import Qwen2Tokenizer

IDS = mx.array([[1, 5, 9, 20, 7], [3, 3, 8, 2, 4]])


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weights(config: dict, seed: int = 0) -> dict[str, mx.array]:
    model = Qwen2Model(Qwen2Config.from_bytes(json.dumps(config).encode()))
    rng = np.random.default_rng(seed)
    return {
        k: mx.array(rng.normal(0.0, 0.2, v.shape).astype(np.float32))
        for k, v in tree_flatten(model.parameters())
    }


def make_dir(
    base: Path, *, config: dict | None = None, split: bool = False, name: str = "m"
) -> Path:
    """`base/name` に config・tokenizer・重み（単一または 2 分割）を置く。"""
    d = base / name
    d.mkdir()
    config = config or synthetic.tiny_config()
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    synthetic.write_tokenizer_json(d)
    w = _weights({**synthetic.tiny_config(), "vocab_size": config["vocab_size"]})
    if split:
        keys = sorted(w)
        mx.save_safetensors(str(d / "model-1.safetensors"), {k: w[k] for k in keys[::2]})
        mx.save_safetensors(str(d / "model-2.safetensors"), {k: w[k] for k in keys[1::2]})
    else:
        mx.save_safetensors(str(d / "model.safetensors"), w)
    return d


def expected(d: Path) -> dict[str, str]:
    return {p.name: _sha(p) for p in sorted(d.iterdir()) if p.is_file()}


def budget(rss: int = 8 << 30) -> ResourceBudget:
    return ResourceBudget(wall_seconds=600.0, rss_bytes=rss, device="cpu")


def load(base: Path, rel: str = "m", *, exp: dict[str, str] | None = None, **kw) -> BaseModel:
    with resolve_root(str(base)) as root:
        return load_base_model(
            root,
            rel,
            expected_sha256=exp if exp is not None else expected(base / rel),
            dtype=kw.pop("dtype", mx.float32),
            resource_budget=kw.pop("resource_budget", budget()),
        )


def fails(exc: pytest.ExceptionInfo[WorkerError], code: str, exit_code: ExitCode) -> None:
    assert exc.value.code == code
    assert exc.value.exit_code == exit_code


# --- 正常系 ---------------------------------------------------------------


def test_load_single_file(tmp_path: Path) -> None:
    """REQ-41: 単一ファイルの合成モデルを読め、logits の形・有限性・sha256 の記録が正しい。"""
    d = make_dir(tmp_path)
    loaded = load(tmp_path)
    logits = loaded.model(IDS)
    assert logits.shape == (2, 5, synthetic.TINY_VOCAB)
    assert bool(mx.all(mx.isfinite(logits)).item())
    assert loaded.file_sha256 == expected(d)
    assert loaded.file_sha256["model.safetensors"] == _sha(d / "model.safetensors")
    assert loaded.weight_bytes == (d / "model.safetensors").stat().st_size
    assert loaded.config.hidden_size == 16


def test_load_split_files_matches_single(tmp_path: Path) -> None:
    """REQ-41: 重みを 2 ファイルに分けても読め、単一ファイルと forward の出力が一致する。"""
    make_dir(tmp_path, name="a")
    make_dir(tmp_path, name="b", split=True)
    one, two = load(tmp_path, "a"), load(tmp_path, "b")
    assert mx.array_equal(one.model(IDS), two.model(IDS)).item()


def test_dtype_conversion(tmp_path: Path) -> None:
    """REQ-41: dtype を指定すると重みがその dtype に変換される。"""
    make_dir(tmp_path)
    loaded = load(tmp_path, dtype=mx.float16)
    assert loaded.model.model.embed_tokens.weight.dtype == mx.float16


def test_tokenizer_encode_values(tmp_path: Path) -> None:
    """REQ-41: 読み込んだ tokenizer が merges どおりの id 列を返す。"""
    make_dir(tmp_path)
    tok = load(tmp_path).tokenizer
    assert tok.encode("hello") == [260]
    assert tok.encode(" hello") == [32, 260]
    assert tok.decode([260, 32, 260]) == "hello hello"
    assert tok.special_ids == {"im_start": 263, "im_end": 264, "endoftext": 262}


def test_tokenizer_matches_poc_implementation(tmp_path: Path) -> None:
    """REQ-41: 移植した tokenizer の encode が PoC 実装と全件一致する（golden 契約を崩さない）。"""
    d = make_dir(tmp_path)
    mine = Qwen2Tokenizer.from_bytes((d / "tokenizer.json").read_bytes())
    poc = poc_tokenizer.Qwen2Tokenizer.from_file(d / "tokenizer.json")
    texts = ["hello help", "I'm DON'T x$5", "年1月  x　　y", "a\r\nb  \n\nc", "ell ll hell"]
    golden = Path(__file__).resolve().parents[2] / "fixtures" / "poc26" / "tokenizer_golden.json"
    if golden.is_file():  # 生成済みなら入力文だけを使う（ids は使わない）
        texts += [c["text"] for c in json.loads(golden.read_text(encoding="utf-8"))["cases"]]
    for text in texts:
        assert mine.encode(text) == poc.encode(text), text


# --- 対応範囲外 -----------------------------------------------------------


@pytest.mark.parametrize(
    "patch",
    [
        {"model_type": "llama"},
        {"rope_scaling": {"type": "linear", "factor": 2.0}},
        {"use_sliding_window": True},
        {"hidden_act": "gelu"},
    ],
)
def test_unsupported_architecture_rejected(tmp_path: Path, patch: dict) -> None:
    """REQ-19: 対応範囲外のアーキテクチャは invalid_config（64）で、対応範囲を示して止まる。"""
    make_dir(tmp_path, config={**synthetic.tiny_config(), **patch})
    with pytest.raises(WorkerError) as exc:
        load(tmp_path)
    fails(exc, "invalid_config", ExitCode.INVALID_INPUT)
    assert "supported:" in exc.value.message


# --- pickle 系・名前 ------------------------------------------------------


def test_pickle_disguised_as_safetensors_rejected(tmp_path: Path) -> None:
    """REQ-39: pickle の先頭バイトを .safetensors の名前で置いても解釈されず invalid_data。"""
    d = make_dir(tmp_path)
    (d / "model.safetensors").write_bytes(b"\x80\x04\x95" + b"\x00" * 32 + b".")
    with pytest.raises(WorkerError) as exc:
        load(tmp_path)
    fails(exc, "invalid_data", ExitCode.INVALID_INPUT)


@pytest.mark.parametrize(
    "name", ["pytorch_model.bin", "model.pt", "../x.safetensors", "a/b.safetensors"]
)
def test_non_safetensors_names_rejected_before_open(tmp_path: Path, name: str) -> None:
    """REQ-39: .bin・.pt・区切り文字を含む名前は開く前に invalid_request。"""
    d = make_dir(tmp_path)
    exp = expected(d)
    exp[name] = "0" * 64
    with pytest.raises(WorkerError) as exc:
        load(tmp_path, exp=exp)
    fails(exc, "invalid_request", ExitCode.INVALID_INPUT)


@pytest.mark.parametrize("drop", ["config.json", "tokenizer.json", "model.safetensors"])
def test_missing_expected_entry_rejected(tmp_path: Path, drop: str) -> None:
    """REQ-39: 必須ファイルの期待値が欠けていれば invalid_request（照合なしで読まない）。"""
    d = make_dir(tmp_path)
    exp = expected(d)
    del exp[drop]
    with pytest.raises(WorkerError) as exc:
        load(tmp_path, exp=exp)
    fails(exc, "invalid_request", ExitCode.INVALID_INPUT)


def test_malformed_expected_sha_rejected(tmp_path: Path) -> None:
    """REQ-39: 期待 sha256 が 64 桁の小文字 16 進でなければ invalid_request。"""
    d = make_dir(tmp_path)
    exp = expected(d) | {"config.json": "ABC"}
    with pytest.raises(WorkerError) as exc:
        load(tmp_path, exp=exp)
    fails(exc, "invalid_request", ExitCode.INVALID_INPUT)


# --- 完全性 ---------------------------------------------------------------


@pytest.mark.parametrize("target", ["model.safetensors", "config.json", "tokenizer.json"])
def test_sha256_mismatch_rejected(tmp_path: Path, target: str) -> None:
    """REQ-39: 期待値と異なるファイルは integrity_check_failed（70）。"""
    d = make_dir(tmp_path)
    exp = expected(d)
    path = d / target
    path.write_bytes(path.read_bytes() + b" ")  # 期待値は書き換え前のまま
    with pytest.raises(WorkerError) as exc:
        load(tmp_path, exp=exp)
    fails(exc, "integrity_check_failed", ExitCode.RUNTIME_ERROR)


# --- 経路 -----------------------------------------------------------------


def test_symlinked_weight_rejected(tmp_path: Path) -> None:
    """REQ-39: 重みが symlink なら symlink_not_allowed。"""
    d = make_dir(tmp_path)
    exp = expected(d)
    real = d / "real.bin"
    (d / "model.safetensors").rename(real)
    os.symlink(real, d / "model.safetensors")
    with pytest.raises(WorkerError) as exc:
        load(tmp_path, exp=exp)
    fails(exc, "symlink_not_allowed", ExitCode.INVALID_INPUT)


def test_symlinked_dir_rejected(tmp_path: Path) -> None:
    """REQ-39: ベースモデルのディレクトリ自体が symlink なら symlink_not_allowed。"""
    d = make_dir(tmp_path)
    os.symlink(d, tmp_path / "link")
    with pytest.raises(WorkerError) as exc:
        load(tmp_path, "link", exp=expected(d))
    fails(exc, "symlink_not_allowed", ExitCode.INVALID_INPUT)


def test_parent_traversal_rejected(tmp_path: Path) -> None:
    """REQ-39: rel_dir の `..` は invalid_path。"""
    d = make_dir(tmp_path)
    with pytest.raises(WorkerError) as exc:
        load(tmp_path, "m/../m", exp=expected(d))
    fails(exc, "invalid_path", ExitCode.INVALID_INPUT)


def test_missing_weight_file_rejected(tmp_path: Path) -> None:
    """REQ-39: 期待値にあるが存在しない重みは invalid_path。"""
    d = make_dir(tmp_path)
    exp = expected(d) | {"missing.safetensors": "0" * 64}
    with pytest.raises(WorkerError) as exc:
        load(tmp_path, exp=exp)
    fails(exc, "invalid_path", ExitCode.INVALID_INPUT)


# --- 上限 -----------------------------------------------------------------


def test_file_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-39: 1 ファイルのサイズ上限超過は読む前に limit_exceeded（20）。"""
    make_dir(tmp_path)
    monkeypatch.setattr(qwen2_model, "MAX_BASE_MODEL_FILE_BYTES", 100)
    with pytest.raises(WorkerError) as exc:
        load(tmp_path)
    fails(exc, "limit_exceeded", ExitCode.LIMIT_EXCEEDED)


def test_total_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-39: 合計サイズの上限超過は limit_exceeded。"""
    make_dir(tmp_path, split=True)
    monkeypatch.setattr(qwen2_model, "MAX_BASE_MODEL_TOTAL_BYTES", 100)
    with pytest.raises(WorkerError) as exc:
        load(tmp_path)
    fails(exc, "limit_exceeded", ExitCode.LIMIT_EXCEEDED)


def test_weight_file_count_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-39: 重みファイルの件数上限超過は limit_exceeded。"""
    make_dir(tmp_path, split=True)
    monkeypatch.setattr(qwen2_model, "MAX_BASE_MODEL_WEIGHT_FILES", 1)
    with pytest.raises(WorkerError) as exc:
        load(tmp_path)
    fails(exc, "limit_exceeded", ExitCode.LIMIT_EXCEEDED)


def test_memory_budget_limit(tmp_path: Path) -> None:
    """REQ-39: メモリ見積もりが予算を超えるなら、モデルを作る前に limit_exceeded。"""
    make_dir(tmp_path)
    with pytest.raises(WorkerError) as exc:
        load(tmp_path, resource_budget=budget(rss=1000))
    fails(exc, "limit_exceeded", ExitCode.LIMIT_EXCEEDED)


def test_limit_values_match_rust_guard() -> None:
    """REQ-39: 準用した上限が Rust ガード層・runtime の値（#400・#580）と同じ。"""
    assert limits.MAX_BASE_MODEL_FILE_BYTES == 1 << 30
    assert limits.MAX_BASE_MODEL_TOTAL_BYTES == 20 << 30
    assert limits.MAX_BASE_MODEL_WEIGHT_FILES == 256
    assert limits.MAX_TOKENIZER_JSON_BYTES == 16 * 1024 * 1024


# --- 整合 -----------------------------------------------------------------


def test_missing_layer_rejected(tmp_path: Path) -> None:
    """REQ-41: 層が 1 つ欠けた重みはキー不一致で invalid_data。"""
    d = make_dir(tmp_path)
    w = mx.load(str(d / "model.safetensors"))
    mx.eval(w)
    w = {k: v for k, v in w.items() if not k.startswith("model.layers.1.")}
    mx.save_safetensors(str(d / "model.safetensors"), w)
    with pytest.raises(WorkerError) as exc:
        load(tmp_path)
    fails(exc, "invalid_data", ExitCode.INVALID_INPUT)


def test_shape_mismatch_rejected(tmp_path: Path) -> None:
    """REQ-41: 形状が config と合わない重みは invalid_data。"""
    d = make_dir(tmp_path)
    w = mx.load(str(d / "model.safetensors"))
    mx.eval(w)
    w["model.norm.weight"] = mx.zeros((17,))
    mx.save_safetensors(str(d / "model.safetensors"), w)
    with pytest.raises(WorkerError) as exc:
        load(tmp_path)
    fails(exc, "invalid_data", ExitCode.INVALID_INPUT)


def test_tokenizer_ids_beyond_vocab_size_rejected(tmp_path: Path) -> None:
    """REQ-41: tokenizer の最大 id が config.vocab_size 以上なら invalid_config。"""
    make_dir(tmp_path, config={**synthetic.tiny_config(), "vocab_size": 264})
    with pytest.raises(WorkerError) as exc:
        load(tmp_path)
    fails(exc, "invalid_config", ExitCode.INVALID_INPUT)


def test_malformed_tokenizer_rejected(tmp_path: Path) -> None:
    """REQ-41: 壊れた tokenizer.json は invalid_data。"""
    d = make_dir(tmp_path)
    (d / "tokenizer.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(WorkerError) as exc:
        load(tmp_path, exp=expected(d))
    fails(exc, "invalid_data", ExitCode.INVALID_INPUT)
