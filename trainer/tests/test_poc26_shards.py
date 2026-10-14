"""PoC-26 ツールの 1.5B・7B 対応（シャード読み込み・上限の上書き・最長 batch 先頭）のテスト。

REQ-41・REQ-39・PoC-26・#552・追補 2 の 4・5 節。合成の極小モデルだけを使い、実重みは読まない。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.poc26 import common, qwen2_model, synthetic
from tools.poc26 import train as train_mod
from tools.poc26.lora_poc import main
from tools.poc26.qwen2_model import apply_lora, load_qwen2

from fandhe_edge_trainer import limits
from test_poc26_lora_poc import _sha, make_data, make_dir, pin_values, train_args

IDS = mx.array([[1, 5, 9, 20, 7], [3, 3, 8, 2, 4]])
INDEX = "model.safetensors.index.json"


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def _h(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_index(model_dir: Path, weight_map: dict) -> str:
    (model_dir / INDEX).write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return _h(model_dir / INDEX)


def _shard(model_dir: Path, names: tuple[str, ...] = ("a.safetensors", "b.safetensors")) -> dict:
    """単一 `model.safetensors` を 2 シャード＋ index に分割する。{シャード名: sha256} を返す。"""
    weights = mx.load(str(model_dir / "model.safetensors"))
    keys = sorted(weights)
    parts = {names[0]: keys[: len(keys) // 2], names[1]: keys[len(keys) // 2 :]}
    for n, ks in parts.items():
        mx.save_safetensors(str(model_dir / n), {k: weights[k] for k in ks})
    write_index(model_dir, {k: n for n, ks in parts.items() for k in ks})
    return {n: _h(model_dir / n) for n in parts}


def _load_sharded(d: Path, shards: dict, dtype=mx.float32, index_sha: str | None = None):
    return load_qwen2(
        d,
        dtype=dtype,
        expected_sha256=index_sha or _h(d / INDEX),
        expected_config_sha256=_h(d / "config.json"),
        shard_sha256=shards,
    )


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path, dict]:
    (tmp_path / "s").mkdir()
    single = synthetic.write_model_dir(tmp_path / "s")
    sharded = tmp_path / "sh"
    shutil.copytree(single, sharded)
    return single, sharded, _shard(sharded)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_sharded_load_equals_single_file(pair, dtype) -> None:
    """REQ-41・#552: シャード読み込みは単一ファイルと同じ重み。"""
    single, sharded, shards = pair
    ref = load_qwen2(
        single,
        dtype=dtype,
        expected_sha256=_h(single / "model.safetensors"),
        expected_config_sha256=_h(single / "config.json"),
    )
    m = _load_sharded(sharded, shards, dtype)
    a, b = np.array(m(IDS).astype(mx.float32)), np.array(ref(IDS).astype(mx.float32))
    assert np.array_equal(a, b)
    assert m.source_bytes == sum((sharded / n).stat().st_size for n in shards)
    assert [x["file"] for x in m.shard_info] == sorted(shards)
    assert ref.shard_info is None


def test_shard_sha256_mismatch_missing_and_extra_rejected(pair) -> None:
    """REQ-39・#552: シャードの sha256 不一致・指定の欠け・余りは拒否する。"""
    _, d, shards = pair
    with pytest.raises(ValueError, match="model shard sha256 mismatch"):
        _load_sharded(d, {**shards, "a.safetensors": "0" * 64})
    with pytest.raises(ValueError, match="do not match the shards in the index"):
        _load_sharded(d, {"a.safetensors": shards["a.safetensors"]})
    with pytest.raises(ValueError, match="do not match the shards in the index"):
        _load_sharded(d, {**shards, "c.safetensors": "1" * 64})
    with pytest.raises(ValueError, match=r"index\.json sha256 mismatch"):
        _load_sharded(d, shards, index_sha="0" * 64)
    (d / "a.safetensors").write_bytes((d / "a.safetensors").read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _load_sharded(d, shards)


@pytest.mark.parametrize("bad", ["../x.safetensors", "/etc/x.safetensors", "sub/x.safetensors"])
def test_index_path_escape_rejected(pair, bad: str) -> None:
    """REQ-39・#552: index のシャード名の `../`・絶対パス・区切り文字は拒否する。"""
    _, d, shards = pair
    wm = json.loads((d / INDEX).read_text())["weight_map"]
    wm[sorted(wm)[0]] = bad
    sha = write_index(d, wm)
    with pytest.raises(ValueError, match="invalid shard name"):
        _load_sharded(d, {**shards, bad: "2" * 64}, index_sha=sha)


def test_symlinked_shard_rejected(pair, tmp_path: Path) -> None:
    """REQ-39・#552: シャードが symlink なら拒否する（O_NOFOLLOW）。"""
    _, d, shards = pair
    real = tmp_path / "real.safetensors"
    shutil.move(d / "a.safetensors", real)
    (d / "a.safetensors").symlink_to(real)
    with pytest.raises(ValueError, match="cannot open model shard"):
        _load_sharded(d, shards)


def test_total_model_bytes_limit_applies_to_sum_of_shards(pair, monkeypatch) -> None:
    """REQ-39・#552: モデル合計バイトの上限は全シャードの合計に掛かる。"""
    _, d, shards = pair
    sizes = [(d / n).stat().st_size for n in shards]
    monkeypatch.setattr(qwen2_model, "MAX_MODEL_BYTES", max(sizes) + 1)
    with pytest.raises(ValueError, match="model files too large"):
        _load_sharded(d, shards)
    monkeypatch.setattr(qwen2_model, "MAX_MODEL_BYTES", sum(sizes))
    assert _load_sharded(d, shards).config.hidden_size == 16


def test_tie_false_lm_head_shape_checked(tmp_path: Path) -> None:
    """REQ-39・#552: tie=false の lm_head.weight は (vocab, hidden) 以外なら拒否する。"""
    d = synthetic.write_model_dir(tmp_path, tie=False)
    w = mx.load(str(d / "model.safetensors"))
    mx.eval(w)  # 上書き前に実体化する（遅延読み込みのため）
    assert w["lm_head.weight"].shape == (synthetic.TINY_VOCAB, 16)

    def load():
        return load_qwen2(
            d,
            dtype=mx.float32,
            expected_sha256=_h(d / "model.safetensors"),
            expected_config_sha256=_h(d / "config.json"),
        )

    assert load().config.tie_word_embeddings is False
    mx.save_safetensors(str(d / "model.safetensors"), {**w, "lm_head.weight": mx.zeros((16, 300))})
    with pytest.raises(ValueError, match="weight shape mismatch"):
        load()


def _shard_flags(shards: dict) -> list[str]:
    return [x for n, s in shards.items() for x in ("--shard-sha256", f"{n}={s}")]


def _with_index_pin(argv: list[str], model: Path) -> list[str]:
    """`--model-sha256` を index の sha256 に差し替える。"""
    argv = list(argv)
    argv[argv.index("--model-sha256") + 1] = _h(model / INDEX)
    return argv


def test_cli_sharded_train_matches_single_and_records(tmp_path: Path) -> None:
    """REQ-41・#552: シャードの train は単一ファイルと同じ adapter・予測で、run.json に記録する。"""
    model, data = make_dir(tmp_path), make_data(tmp_path)
    single_out, shard_out = tmp_path / "o1", tmp_path / "o2"
    assert main(train_args(model, data, single_out)) == 0
    shards = _shard(model)  # 同じ dir に a/b と index を追加（model.safetensors は残す）
    argv = _with_index_pin(train_args(model, data, shard_out), model) + _shard_flags(shards)
    assert main(argv) == 0
    for name in ("pred.jsonl", "raw_scores.jsonl"):
        assert _sha(single_out / name) == _sha(shard_out / name), name
    # adapter ファイルは metadata の base_model_sha256（単一 vs index）が違うので、重みで比べる
    w1 = mx.load(str(single_out / "adapters.safetensors"))
    w2 = mx.load(str(shard_out / "adapters.safetensors"))
    assert w1.keys() == w2.keys()
    assert all(bool(mx.array_equal(w1[k], w2[k]).item()) for k in w1)
    run = json.loads((shard_out / "run.json").read_text())
    assert run["model_safetensors_sha256"] == _h(model / INDEX)
    assert [s["file"] for s in run["model_safetensors_shards"]] == sorted(shards)
    total = sum(s["bytes"] for s in run["model_safetensors_shards"])
    assert run["model_safetensors_bytes"] == total
    single = json.loads((single_out / "run.json").read_text())
    assert "model_safetensors_shards" not in single  # 単一ファイル時の既存キーは変えない
    assert single["model_safetensors_sha256"] == pin_values(model)["model"]


def test_cli_shard_mismatch_duplicate_and_malformed_are_64(tmp_path: Path) -> None:
    """REQ-39・#552: CLI でもシャードの不一致・重複指定・形式不正・欠けは 64。"""
    model, data = make_dir(tmp_path), make_data(tmp_path)
    shards = _shard(model)
    base = _with_index_pin(train_args(model, data, tmp_path / "o"), model)
    ok = _shard_flags(shards)  # [flag, a=sha, flag, b=sha]
    bad_sha = ["--shard-sha256", f"a.safetensors={'0' * 64}", ok[2], ok[3]]
    assert main([*base, *bad_sha]) == 64
    assert main([*base, *ok, ok[0], ok[1]]) == 64  # 重複
    assert main([*base, "--shard-sha256", "no-equals"]) == 64
    assert main([*base, ok[0], ok[1]]) == 64  # b が無い


def test_limit_overrides_take_effect_and_are_recorded(tmp_path: Path, monkeypatch) -> None:
    """REQ-39・#552: 既定の上限では拒否され、PoC 専用の引数で上書きすると通り、適用値を記録する。"""
    model, data = make_dir(tmp_path), make_data(tmp_path)
    size = (model / "model.safetensors").stat().st_size
    # 既定値を実物より小さくした状況（実際の既定 1 GiB の代役。既定値そのものは不変）
    monkeypatch.setattr(qwen2_model, "MAX_MODEL_BYTES", size - 1)
    assert main(train_args(model, data, tmp_path / "o1")) == 20
    assert (
        main(train_args(model, data, tmp_path / "o2", extra=["--max-model-bytes", str(size)])) == 0
    )
    assert qwen2_model.MAX_MODEL_BYTES == size - 1  # 上書きは終了時に復元される
    run = json.loads((tmp_path / "o2" / "run.json").read_text())
    assert run["applied_limits"]["max_model_bytes"] == size
    assert run["applied_limits"]["max_memory_bytes"] == 8 << 30
    assert run["applied_limits"]["max_train_rss_bytes"] == 8 << 30
    assert run["mx_device_info"] == {
        k: v for k, v in mx.device_info().items() if isinstance(v, (int, float, str))
    }
    assert run["mlx_peak_memory_bytes"] >= 0
    # メモリ見積もり・学習メモリ上限の上書き
    monkeypatch.undo()
    extra = ["--max-memory-bytes", "100"]
    assert main(train_args(model, data, tmp_path / "o3", extra=extra)) == 20
    assert common.train_rss_limit == 8 << 30
    extra = ["--max-train-rss-bytes", "1"]
    assert main(train_args(model, data, tmp_path / "o4", extra=extra)) == 20
    assert common.train_rss_limit == 8 << 30


@pytest.mark.parametrize(
    ("flag", "ceiling"),
    [
        ("--max-model-bytes", 32 << 30),
        ("--max-memory-bytes", 30 << 30),
        ("--max-train-rss-bytes", 30 << 30),
    ],
)
def test_limit_override_ceiling(tmp_path: Path, flag: str, ceiling: int) -> None:
    """REQ-39・#552: 上書き値の天井を超える指定・0 は 64。"""
    model, data = make_dir(tmp_path), make_data(tmp_path)
    for i, v in enumerate((ceiling + 1, 0)):
        extra = [flag, str(v)]
        assert main(train_args(model, data, tmp_path / f"o{i}", extra=extra)) == 64


def test_defaults_unchanged() -> None:
    """REQ-39・#552: 既定の上限は従来の値のまま。"""
    assert qwen2_model.MAX_MODEL_BYTES == 1 << 30
    assert qwen2_model.MAX_MODEL_MEMORY_BYTES == 8 << 30
    assert limits.MAX_TRAIN_RSS_BYTES == 8 << 30
    assert common.train_rss_limit == 8 << 30


def test_longest_batch_first_helper() -> None:
    """REQ-41・#552: 最長 batch（最後）を先頭へ。他の相対順は保つ。"""
    assert train_mod.longest_batch_first([2, 0, 3, 1], 4) == [3, 2, 0, 1]
    assert train_mod.longest_batch_first([3, 1, 0, 2], 4) == [3, 1, 0, 2]


def _first_batches(tmp_path: Path, *, longest: bool, steps: int):
    tmp_path.mkdir()
    d = synthetic.write_model_dir(tmp_path)
    m = load_qwen2(
        d,
        dtype=mx.float32,
        expected_sha256=_h(d / "model.safetensors"),
        expected_config_sha256=_h(d / "config.json"),
    )
    apply_lora(m, num_layers=1, rank=2, scale=1.0, dropout=0.0, seed=0)
    seqs = [[1 + (j % 5) for j in range(3 + 2 * i)] for i in range(8)]  # 長さ 3,5,...,17
    seen: list[list[int]] = []
    real = train_mod.collate

    def spy(s, idxs, pad_id):
        seen.append(list(idxs))
        return real(s, idxs, pad_id)

    train_mod.collate = spy
    try:
        train_mod.train_loop(
            m, seqs, pad_id=0, iters=steps, lr=1e-3, batch_size=2, seed=3,
            budget=common.Budget(), longest_first=longest,
        )  # fmt: skip
    finally:
        train_mod.collate = real
    return seen, seqs


def test_train_loop_longest_batch_first(tmp_path: Path) -> None:
    """REQ-41・#552: 有効時は最初の 1 step が最長 batch。既定は従来のバッチ順と完全に同じ。"""
    off, seqs = _first_batches(tmp_path / "a", longest=False, steps=4)
    on, _ = _first_batches(tmp_path / "b", longest=True, steps=4)
    assert max(len(seqs[i]) for i in on[0]) == max(len(s) for s in seqs)
    assert sorted(map(sorted, on)) == sorted(map(sorted, off))  # 同じ 4 batch を 1 周
    # 既定（無効）の順序は従来の rng 順列のまま
    perm = [int(i) for i in np.random.default_rng(3).permutation(4)]
    batches = train_mod.make_batches([len(s) for s in seqs], 2)
    assert off == [batches[i] for i in perm]


def test_set_limits_rejects_bool_and_out_of_range(monkeypatch) -> None:
    """REQ-39・#552: 上書き値の bool・範囲外は拒否する。"""
    monkeypatch.setattr(qwen2_model, "MAX_MODEL_BYTES", qwen2_model.MAX_MODEL_BYTES)
    monkeypatch.setattr(qwen2_model, "MAX_MODEL_MEMORY_BYTES", qwen2_model.MAX_MODEL_MEMORY_BYTES)
    for kw in ({"model_bytes": True}, {"memory_bytes": True}, {"memory_bytes": 0}):
        with pytest.raises(ValueError, match="out of range"):
            qwen2_model.set_limits(**kw)
    with pytest.raises(ValueError, match="out of range"):
        qwen2_model.set_limits(memory_bytes=(30 << 30) + 1)


def test_memory_override_limits_forward_estimate(tmp_path: Path, monkeypatch) -> None:
    """REQ-39・#552: memory_bytes の上書きで forward の見積もりが上限超過になる。"""
    d = synthetic.write_model_dir(tmp_path)
    m = load_qwen2(
        d,
        dtype=mx.float32,
        expected_sha256=_h(d / "model.safetensors"),
        expected_config_sha256=_h(d / "config.json"),
    )
    monkeypatch.setattr(qwen2_model, "MAX_MODEL_MEMORY_BYTES", qwen2_model.MAX_MODEL_MEMORY_BYTES)
    need = m.forward_bytes(2, 5, False)
    qwen2_model.set_limits(memory_bytes=need - 1)
    with pytest.raises(qwen2_model.LimitExceededError):
        m(IDS)
    qwen2_model.set_limits(memory_bytes=need)
    assert m(IDS).shape == (2, 5, synthetic.TINY_VOCAB)


def test_rss_override_larger_than_default_passes(tmp_path: Path, monkeypatch) -> None:
    """REQ-39・#552: 学習メモリ上限は既定より小さければ止まり、大きくすると通る。"""
    monkeypatch.setattr(common, "train_rss_limit", 1)
    with pytest.raises(Exception, match="RSS budget"):
        common.Budget().check()
    monkeypatch.setattr(common, "train_rss_limit", common.max_rss_bytes() + (1 << 30))
    common.Budget().check()


def test_content_rewritten_after_check_uses_verified_bytes(pair, monkeypatch) -> None:
    """REQ-39・#552: 照合後に同サイズで中身を書き換え mtime を戻しても、照合したバイト列を使う。"""
    import os

    _, d, shards = pair
    ref = _load_sharded(d, shards)
    real_sha = hashlib.sha256
    path = d / "a.safetensors"
    orig = path.read_bytes()
    st = path.stat()
    done = []

    class Spy:
        def __init__(self, data):
            self.h = real_sha(data)
            if len(data) == len(orig) and not done:
                done.append(1)  # 照合の直後と同じ位置（sha256 計算後）に相当する書き換え
                path.write_bytes(bytes(len(orig)))
                os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))

        def hexdigest(self):
            return self.h.hexdigest()

    monkeypatch.setattr(qwen2_model.hashlib, "sha256", Spy)
    m = _load_sharded(d, shards)
    assert done
    a, b = np.array(m(IDS)), np.array(ref(IDS))
    assert np.array_equal(a, b)  # 書き換え後の（ゼロ）内容ではなく、照合した内容で読めている


def test_file_grown_after_size_check_rejected(pair, monkeypatch) -> None:
    """REQ-39・#552: サイズ検査の後にファイルが大きくなっていたら、読まずに拒否する。"""
    _, d, shards = pair
    real = qwen2_model._open_regular
    calls = []

    def growing(path, limit, what):
        calls.append(path.name)
        if len(calls) == 3:  # 事前 stat は 2 回（a・b）、3 回目が a の読み込み
            with path.open("ab") as f:
                f.write(b"\0" * 64)
            return real(path, limit + 64, what)  # 上限側は通し、サイズ不一致の検査で止める
        return real(path, limit, what)

    monkeypatch.setattr(qwen2_model, "_open_regular", growing)
    with pytest.raises(qwen2_model.LimitExceededError, match="changed size"):
        _load_sharded(d, shards)
