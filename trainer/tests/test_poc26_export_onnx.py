"""PoC-26 候補 P の ONNX 書き出し（`export_onnx.py`）の検査（REQ-41・TASK-41.1-7・#392）。

極小の合成 Qwen2（GQA あり）と合成 tokenizer で、LoRA 統合・ONNX の手組み・集計・CLI の
`export-onnx` / `verify-onnx` を CPU で決定的に確認する（テストハーネス。実重みは使わない）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.poc26 import qwen2_model, synthetic
from tools.poc26.export_onnx import _Graph, build_onnx, fused_weights, summarize
from tools.poc26.lora_poc import main

from test_poc26_lora_poc import _sha, make_data, make_dir, pin_flags, train_args

INITIALIZERS_TINY = 39  # 合成モデル（2 層・tie）の実測値
ATOL = 1e-4  # float32 の演算順序の差。実測は 1e-6 台


def _lora_model(*, tie: bool = True) -> qwen2_model.Qwen2Model:
    """末尾 1 層だけ LoRA を付け、lora_b を非 0 にした合成モデル（統合の差が logits に出る）。"""
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    cfg = qwen2_model.Qwen2Config.from_bytes(json.dumps(synthetic.tiny_config(tie=tie)).encode())
    model = qwen2_model.Qwen2Model(cfg)
    qwen2_model.apply_lora(model, num_layers=1, rank=2, scale=20.0, dropout=0.0, seed=1)
    last = model.model.layers[-1]
    for lin in (last.self_attn.q_proj, last.self_attn.v_proj, last.mlp.down_proj):
        lin.lora_b = mx.random.normal(lin.lora_b.shape) * 0.1
    mx.eval(model.parameters())
    return model


def _export(model: qwen2_model.Qwen2Model, out: Path) -> tuple[Path, Path]:
    out.mkdir()
    mp, dp = out / "model.onnx", out / "model.onnx.data"
    with dp.open("wb") as f:
        proto = build_onnx(model.config, fused_weights(model), _Graph(f, dp.name))
    onnx.save_model(proto, str(mp))
    return mp, dp


@pytest.mark.parametrize("tie", [True, False])
def test_onnx_matches_mlx_with_lora_and_gqa(tmp_path: Path, tie: bool) -> None:
    """REQ-41: 統合 ONNX の logits が LoRA 付き MLX と許容差内で一致し argmax も一致する。"""
    model = _lora_model(tie=tie)
    mp, _ = _export(model, tmp_path / "o")
    onnx.checker.check_model(str(mp))
    ref = ReferenceEvaluator(onnx.load(str(mp)))
    for n in (1, 7, 20):  # 動的 T
        ids = np.random.default_rng(n).integers(0, 300, (1, n))
        want = np.array(model(mx.array(ids)))
        got = ref.run(None, {"input_ids": ids.astype(np.int64)})[0]
        assert got.shape == want.shape == (1, n, 300)
        assert np.abs(got - want).max() < ATOL
        assert (got.argmax(-1) == want.argmax(-1)).all()


def test_lora_fuse_changes_weights_by_scale_times_ab() -> None:
    """REQ-41: 統合重みは `W + scale * (A @ B).T`（LoRA なしの重みとは異なる）。"""
    model = _lora_model()
    lin = model.model.layers[-1].self_attn.q_proj
    want = np.array(lin.linear.weight) + 20.0 * (np.array(lin.lora_a) @ np.array(lin.lora_b)).T
    fused = fused_weights(model)
    np.testing.assert_allclose(fused["model.layers.1.self_attn.q_proj.weight"], want, atol=1e-6)
    assert np.abs(want - np.array(lin.linear.weight)).max() > 1e-3
    first = np.array(model.model.layers[0].self_attn.q_proj.weight)  # LoRA なしの層はそのまま
    np.testing.assert_array_equal(fused["model.layers.0.self_attn.q_proj.weight"], first)


def test_summary_lists_ops_and_runtime_gaps(tmp_path: Path) -> None:
    """REQ-41: 集計は opset 13・演算数・ランタイム未対応 op・上限超過を返す。"""
    model = _lora_model()
    mp, dp = _export(model, tmp_path / "o")
    s = summarize(mp, dp)
    assert (s["opset"], s["ir_version"]) == (13, 8)
    assert s["op_counts"]["Softmax"] == 2
    assert s["op_counts"]["MatMul"] > 0
    assert sum(s["op_counts"].values()) == s["node_count"]
    # ランタイムは固定テンプレートのみ。MatMul・Where・Cos などは未対応として列挙される
    assert {"MatMul", "Where", "Cos", "Sin", "Reshape", "Split"} <= set(
        s["runtime"]["unsupported_ops"]
    )
    assert "Softmax" not in s["runtime"]["unsupported_ops"]
    lim = s["runtime"]["limits"]
    assert s["initializer_count"] == INITIALIZERS_TINY
    assert lim["initializer_count"] == {"limit": 64, "actual": INITIALIZERS_TINY, "exceeded": False}
    assert lim["node_count"]["actual"] == s["node_count"] > 128  # 合成 2 層でもノード上限は超える
    assert lim["node_count"]["exceeded"] is True
    assert lim["max_dims"] == {"limit": 4, "actual": 2, "exceeded": False}
    assert s["runtime"]["external_data_used"] is True
    assert s["runtime"]["external_data_accepted"] is False
    assert s["external_data_bytes"] > 0
    assert s["dtype"] == "float32"


@pytest.fixture(scope="module")
def adapter_env(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("export")
    model, data = make_dir(tmp), make_data(tmp)
    out = tmp / "train"
    assert main(train_args(model, data, out)) == 0
    return {"tmp": tmp, "model": model, "adapter": out, "data": data}


def test_cli_export_then_verify(adapter_env: dict, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-41: 学習済み adapter から `export-onnx` で書き出し、`verify-onnx` で一致を確認できる。"""
    a = adapter_env
    flags = [
        *["--model-dir", str(a["model"]), "--adapter-dir", str(a["adapter"])],
        *["--adapter-sha256", _sha(a["adapter"] / "adapters.safetensors")],
        *pin_flags(a["model"]),
        *["--evidence", "test_harness"],
    ]
    onnx_dir = a["tmp"] / "onnx"
    assert main(["export-onnx", *flags, "--out-dir", str(onnx_dir)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["checker"], summary["opset"]) == ("passed", 13)
    assert (onnx_dir / "export_summary.json").is_file()
    assert main(["verify-onnx", *flags, "--onnx-dir", str(onnx_dir)]) == 0
    res = json.loads(capsys.readouterr().out)["prompts"]
    assert len(res) == 2
    for r in res:
        assert r["max_abs_error"] < ATOL
        assert r["argmax_match_positions"] == r["positions"]
    assert main(["export-onnx", *flags, "--out-dir", str(onnx_dir)]) == 64  # 上書きしない


def test_cli_rejects_wrong_adapter_sha(adapter_env: dict) -> None:
    """REQ-41: adapter の sha256 が違えば 64 で止まり、出力先を作らない。"""
    a = adapter_env
    flags = [
        *["--model-dir", str(a["model"]), "--adapter-dir", str(a["adapter"])],
        *["--adapter-sha256", "0" * 64, *pin_flags(a["model"])],
        *["--evidence", "test_harness", "--out-dir", str(a["tmp"] / "never")],
    ]
    assert main(["export-onnx", *flags]) == 64
    assert not (a["tmp"] / "never").exists()


def _flags(a: dict, adapter: Path) -> list[str]:
    return [
        *["--model-dir", str(a["model"]), "--adapter-dir", str(adapter)],
        *["--adapter-sha256", _sha(adapter / "adapters.safetensors")],
        *pin_flags(a["model"]),
        *["--evidence", "test_harness"],
    ]


def test_verify_matches_with_dropout_adapter(
    adapter_env: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-41: dropout 0.5 の adapter でも MLX 側が eval モードで決定的になり、照合が一致する。"""
    a = adapter_env
    drop = a["tmp"] / "train_drop"
    assert main(train_args(a["model"], a["data"], drop, extra=["--dropout", "0.5"])) == 0
    flags = _flags(a, drop)
    onnx_dir = a["tmp"] / "onnx_drop"
    assert main(["export-onnx", *flags, "--out-dir", str(onnx_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["adapter_config"]["dropout"] == 0.5
    assert main(["verify-onnx", *flags, "--onnx-dir", str(onnx_dir)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["adapter_dropout"] == 0.5
    for r in out["prompts"]:
        assert r["max_abs_error"] < ATOL
        assert r["argmax_match_positions"] == r["positions"]


def test_verify_rejects_tampered_onnx(
    adapter_env: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-41・REQ-39: 書き出し時の sha256 と違うファイルは verify が読む前に 64 で拒否する。"""
    a = adapter_env
    flags = _flags(a, a["adapter"])
    onnx_dir = a["tmp"] / "onnx_tamper"
    assert main(["export-onnx", *flags, "--out-dir", str(onnx_dir)]) == 0
    capsys.readouterr()
    data = onnx_dir / "model.onnx.data"
    raw = bytearray(data.read_bytes())
    raw[0] ^= 1
    data.write_bytes(bytes(raw))
    assert main(["verify-onnx", *flags, "--onnx-dir", str(onnx_dir)]) == 64


# --- 資源予算・照合済みバイトの読み込み・summary 検証（REQ-41・REQ-39） -----------------------


def _expected(model: qwen2_model.Qwen2Model) -> onnx.ModelProto:
    from tools.poc26.export_onnx import expected_graph

    return expected_graph(model, "model.onnx.data")


def _summary_for(mp: Path, dp: Path) -> dict:
    import hashlib

    return {
        "model_onnx_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
        "model_onnx_data_sha256": hashlib.sha256(dp.read_bytes()).hexdigest(),
    }


def test_memory_estimate_over_budget_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-39: 同時保持量の見積もりが RSS 上限を超えるなら確保前に 20 で止める。"""
    from tools.poc26.export_onnx import check_memory

    from fandhe_edge_trainer import limits
    from fandhe_edge_trainer.errors import WorkerError
    from fandhe_edge_trainer.exitcode import ExitCode

    monkeypatch.setattr(limits, "MAX_TRAIN_RSS_BYTES", 1000)
    check_memory(1000, "x")  # 境界は通る
    with pytest.raises(WorkerError) as ei:
        check_memory(1001, "x")
    assert ei.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_cli_export_and_verify_reject_when_estimate_exceeds_budget(
    adapter_env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: 予算を極小にすると export は重みコピー前に、verify は照合の前に 20 で拒否する。"""
    from fandhe_edge_trainer import limits

    a = adapter_env
    flags = _flags(a, a["adapter"])
    ok = a["tmp"] / "onnx_budget"
    assert main(["export-onnx", *flags, "--out-dir", str(ok)]) == 0
    monkeypatch.setattr(limits, "MAX_TRAIN_RSS_BYTES", 1)
    assert main(["export-onnx", *flags, "--out-dir", str(a["tmp"] / "onnx_over")]) == 20
    assert not (a["tmp"] / "onnx_over").exists()
    assert main(["verify-onnx", *flags, "--onnx-dir", str(ok)]) == 20


def test_graph_checks_budget_per_weight(tmp_path: Path) -> None:
    """REQ-39: 重み 1 本ごとに Budget.check が呼ばれ、超過の例外はそのまま伝わる。"""

    class Over:
        def check(self) -> None:
            raise RuntimeError("over")

    with (tmp_path / "d").open("wb") as f, pytest.raises(RuntimeError, match="over"):
        _Graph(f, "d", Over()).weight("w", np.zeros((64, 64), np.float32))  # type: ignore[arg-type]


def test_verified_load_ignores_swap_after_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-41・REQ-39: 開いた後にパスのファイルを差し替えても、使うのは照合済みバイトだけ。"""
    from tools.poc26 import export_onnx as ex
    from tools.poc26.common import Budget

    model = _lora_model()
    mp, dp = _export(model, tmp_path / "o")
    summary = _summary_for(mp, dp)
    want = ReferenceEvaluator(onnx.load(str(mp)))  # 差し替え前の正しい結果
    ids = np.arange(5, dtype=np.int64)[None]
    expected = want.run(None, {"input_ids": ids})[0]

    def swap(path: Path) -> None:
        tmp = path.with_name(path.name + ".evil")
        tmp.write_bytes(b"\xff" * path.stat().st_size)
        tmp.replace(path)  # 別 inode で置換（既に開いた fd は元の inode を指す）

    real_open, real_read = ex.open_regular, ex.read_input

    def open_then_swap(path, limit, what):
        r = real_open(path, limit, what)
        swap(Path(path))
        return r

    def read_then_swap(path, limit, what):
        r = real_read(path, limit, what)
        swap(Path(path))
        return r

    monkeypatch.setattr(ex, "open_regular", open_then_swap)
    monkeypatch.setattr(ex, "read_input", read_then_swap)
    monkeypatch.setattr(onnx, "load", lambda *_a, **_k: pytest.fail("path re-read"))
    proto = ex.load_verified_onnx(mp.parent, summary, Budget(), _expected(model))
    assert not any(t.data_location == onnx.TensorProto.EXTERNAL for t in proto.graph.initializer)
    got = ReferenceEvaluator(proto).run(None, {"input_ids": ids})[0]
    np.testing.assert_array_equal(got, expected)
    assert dp.read_bytes()[:1] == b"\xff"  # 差し替えは実際に起きている


@pytest.mark.parametrize("target", ["model.onnx", "model.onnx.data"])
def test_verified_load_rejects_mismatch(tmp_path: Path, target: str) -> None:
    """REQ-41・REQ-39: 照合値と違う内容は invalid_input(64) で、解析に進まない。"""
    from tools.poc26.common import Budget
    from tools.poc26.export_onnx import load_verified_onnx

    from fandhe_edge_trainer.errors import WorkerError

    model = _lora_model()
    mp, dp = _export(model, tmp_path / "o")
    summary = _summary_for(mp, dp)
    p = mp.parent / target
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(WorkerError) as ei:
        load_verified_onnx(mp.parent, summary, Budget(), _expected(model))
    assert ei.value.exit_code == 64


@pytest.mark.parametrize(
    "summary",
    [
        [],
        None,
        "x",
        {},
        {"model_onnx_sha256": "0" * 64},  # data 欠落
        {"model_onnx_sha256": "0" * 64, "model_onnx_data_sha256": None},
        {"model_onnx_sha256": "0" * 64, "model_onnx_data_sha256": 5},
        {"model_onnx_sha256": "0" * 63, "model_onnx_data_sha256": "0" * 64},
        {"model_onnx_sha256": "0" * 64, "model_onnx_data_sha256": "G" * 64},
        {"model_onnx_sha256": "A" * 64, "model_onnx_data_sha256": "0" * 64},
    ],
)
def test_summary_shape_is_validated(summary: object) -> None:
    """REQ-39: export_summary.json のルート型・ハッシュ欄の型と形式が不正なら 64。"""
    from tools.poc26.export_onnx import _summary_hashes

    from fandhe_edge_trainer.errors import WorkerError

    with pytest.raises(WorkerError) as ei:
        _summary_hashes(summary)
    assert ei.value.exit_code == 64


def _tamper_tensor(proto: onnx.ModelProto, **kw: object) -> onnx.ModelProto:
    """最初の external tensor の宣言（length / offset / dims）を書き換える。"""
    t = next(t for t in proto.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL)
    for e in t.external_data:
        if e.key in kw:
            e.value = str(kw[e.key])
    if "dims" in kw:
        del t.dims[:]
        t.dims.extend(kw["dims"])  # type: ignore[arg-type]
    return proto


@pytest.mark.parametrize(
    "kw",
    [
        {"length": 1 << 60},  # 巨大 length
        {"dims": [1 << 40, 1 << 40]},  # 巨大 dims（np.int64 ならあふれる積）
        {"dims": [-4, -4]},  # 負の次元（積は正）
        {"length": -4},
        {"offset": -64},
        {"offset": 1 << 40},  # offset+length がファイルを超える
        {"length": 4},  # 次元積×4 と不一致
    ],
)
def test_declared_external_ranges_rejected_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kw: dict
) -> None:
    """REQ-41・REQ-39: 宣言値が不正なら bytearray を確保する前に 64 で拒否する。"""
    from tools.poc26 import export_onnx as ex

    from fandhe_edge_trainer.errors import WorkerError

    mp, dp = _export(_lora_model(), tmp_path / "o")
    proto = _tamper_tensor(onnx.load(str(mp), load_external_data=False), **kw)

    class NoAlloc(bytearray):
        def __init__(self, *a: object) -> None:
            pytest.fail("allocated before validation")

    monkeypatch.setattr(ex, "bytearray", NoAlloc, raising=False)
    with pytest.raises(WorkerError) as ei:
        ex._external_tensors(proto, "model.onnx.data", dp.stat().st_size)
    assert ei.value.exit_code == 64


def test_overlapping_external_ranges_rejected(tmp_path: Path) -> None:
    """REQ-41・REQ-39: 範囲が重なる宣言は 64。"""
    from tools.poc26 import export_onnx as ex

    from fandhe_edge_trainer.errors import WorkerError

    mp, dp = _export(_lora_model(), tmp_path / "o")
    proto = onnx.load(str(mp), load_external_data=False)
    ext = [t for t in proto.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    first = {e.key: e.value for e in ext[0].external_data}["offset"]
    for e in ext[1].external_data:
        if e.key == "offset":
            e.value = first  # 先頭と同じ位置へ重ねる
    with pytest.raises(WorkerError) as ei:
        ex._external_tensors(proto, "model.onnx.data", dp.stat().st_size)
    assert ei.value.exit_code == 64


def test_declared_length_counts_toward_memory_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: 見積もりは宣言長の合計を含み、上限超過は確保前に 20。"""
    from tools.poc26 import export_onnx as ex
    from tools.poc26.common import Budget

    from fandhe_edge_trainer import limits
    from fandhe_edge_trainer.errors import WorkerError

    model = _lora_model()
    mp, dp = _export(model, tmp_path / "o")
    monkeypatch.setattr(limits, "MAX_TRAIN_RSS_BYTES", 1 << 30)  # 1 GiB 固定分だけで超過
    with pytest.raises(WorkerError) as ei:
        ex.load_verified_onnx(mp.parent, _summary_for(mp, dp), Budget(), _expected(model))
    assert ei.value.exit_code == 20


# --- 構造照合・出力検証（任意グラフの実行を許さない。REQ-41・REQ-39） --------------------------


def _rewrite(tmp_path: Path, mutate) -> tuple[Path, dict, qwen2_model.Qwen2Model]:
    """model.onnx を改変して書き戻し、改変後のバイト列に対する summary を返す。"""
    model = _lora_model()
    mp, dp = _export(model, tmp_path / "o")
    proto = onnx.load(str(mp), load_external_data=False)
    mutate(proto)
    mp.write_bytes(proto.SerializeToString())
    return mp, _summary_for(mp, dp), model


def _add_loop(p: onnx.ModelProto) -> None:
    p.graph.node.append(onnx.helper.make_node("Loop", ["", "", ""], ["lp"], name="evil"))


def _add_constant_of_shape(p: onnx.ModelProto) -> None:
    p.graph.node.append(onnx.helper.make_node("ConstantOfShape", ["c_1"], ["big"], name="evil"))


def _change_attribute(p: onnx.ModelProto) -> None:
    node = next(n for n in p.graph.node if n.op_type == "Softmax")
    node.attribute[0].i = 1


def _change_dims(p: onnx.ModelProto) -> None:
    t = next(t for t in p.graph.initializer if t.name == "model.norm.weight")
    t.dims[0] += 1


@pytest.mark.parametrize(
    "mutate", [_add_loop, _add_constant_of_shape, _change_attribute, _change_dims]
)
def test_structure_mismatch_rejected_before_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate
) -> None:
    """REQ-41・REQ-39: Loop / ConstantOfShape 追加・attribute / dims 改変は構築前に 64。"""
    from tools.poc26 import export_onnx as ex
    from tools.poc26.common import Budget

    from fandhe_edge_trainer.errors import WorkerError

    mp, summary, model = _rewrite(tmp_path, mutate)
    monkeypatch.setattr(
        ReferenceEvaluator, "__init__", lambda *_a, **_k: pytest.fail("evaluator built")
    )
    with pytest.raises(WorkerError) as ei:
        ex.load_verified_onnx(mp.parent, summary, Budget(), _expected(model))
    assert ei.value.exit_code == 64


def test_unmodified_graph_passes_structure_check(tmp_path: Path) -> None:
    """REQ-41: 無改変なら構造照合を通り、重みは inline になる（期待グラフは値なしで組んだもの）。"""
    from tools.poc26.common import Budget
    from tools.poc26.export_onnx import load_verified_onnx

    model = _lora_model()
    mp, dp = _export(model, tmp_path / "o")
    proto = load_verified_onnx(mp.parent, _summary_for(mp, dp), Budget(), _expected(model))
    assert len(proto.graph.node) > 0


@pytest.mark.parametrize(
    ("outs", "tokens"),
    [
        ([np.zeros((1, 1, 300), np.float32)], 5),  # 形状 [1,1,V]（broadcast で通ってしまう形）
        ([np.full((1, 5, 300), np.nan, np.float32)], 5),  # NaN
        ([np.full((1, 5, 300), np.inf, np.float32)], 5),
        ([np.zeros((1, 5, 300), np.float64)], 5),  # dtype
        ([], 5),  # 出力 0 個
        ([np.zeros((1, 5, 300), np.float32)] * 2, 5),  # 出力 2 個
    ],
)
def test_output_validation_rejects(outs: list, tokens: int) -> None:
    """REQ-41・REQ-39: 出力の個数・dtype・形状 [1,T,V]・有限性が違えば 64。"""
    from tools.poc26.export_onnx import _checked_logits

    from fandhe_edge_trainer.errors import WorkerError

    with pytest.raises(WorkerError) as ei:
        _checked_logits(outs, tokens, 300)
    assert ei.value.exit_code == 64
    ok = np.zeros((1, 5, 300), np.float32)
    assert _checked_logits([ok], 5, 300) is ok


def test_intermediate_bound_concrete_value() -> None:
    """REQ-39: 中間テンソルの上界は config と T で決まる（具体値）。"""
    from tools.poc26.export_onnx import intermediate_bound

    cfg = _lora_model().config  # hidden 32?: 値は config から式どおりに計算して固定する
    wide = max(cfg.hidden_size, cfg.intermediate_size)
    want = 4 * (
        cfg.num_hidden_layers * (40 * 7 * wide + 4 * cfg.num_attention_heads * 49)
        + 2 * 7 * cfg.vocab_size
    )
    assert intermediate_bound(cfg, 7) == want > 0
