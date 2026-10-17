"""qwen2_lora の LoRA 統合 ONNX 書き出し（REQ-41・TASK-41.6-2・#431）。

極小の合成モデル（CPU）で確かめる。証拠種別: テストハーネス（ReferenceEvaluator・MLX CPU）。
演算の範囲は `docs/design/qwen2-scoring-onnx-ops.md`（#415）の表を読んで照合する（文書が正）。
"""

from __future__ import annotations

import hashlib
import io
import itertools
import re
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import onnx
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from onnx import TensorProto, reference
from onnx.reference.op_run import OpRun

import fandhe_edge_trainer.kinds.qwen2_lora as q
import fandhe_edge_trainer.qwen2_onnx as qo
from fandhe_edge_trainer import limits
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode
from fandhe_edge_trainer.qwen2_model import Qwen2Config, Qwen2Model, apply_lora
from test_qwen2_lora_train import _train

DOC = Path(__file__).resolve().parents[2] / "docs/design/qwen2-scoring-onnx-ops.md"
SHARD_MAX, THRESHOLD = 10000, 256
PROMPTS = ["short text one", "another input", "third"]


def allowed_ops() -> set[str]:
    """設計文書 §3 の表の第 1 列（"Add / Sub / Mul" は分割）。"""
    text = DOC.read_text(encoding="utf-8").split("## 3. 演算一覧")[1].split("\n## 4.")[0]
    ops: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"\| ([A-Za-z /]+) \|", line)
        if m and m.group(1).strip() != "演算":
            ops |= {o.strip() for o in m.group(1).split("/")}
    return ops


class Keep(io.BytesIO):
    def close(self) -> None:  # 内容を読むため閉じない
        pass


def export(trained, **kw):
    files: dict[str, Keep] = {}

    def open_shard(name: str):
        files[name] = Keep()
        return files[name]

    out = io.BytesIO()
    infos = qo.export_scoring_onnx(trained, out, open_shard, **kw)
    return out.getvalue(), {n: f.getvalue() for n, f in files.items()}, infos


class GatherElements(OpRun):
    """ReferenceEvaluator 標準の GatherElements は `np.choose`（選択肢 64 件まで）を使い、軸の長さが
    64 を超える（語彙 300 以上の）採点グラフで失敗する。同じ意味の `take_along_axis` で置き換える
    （テスト専用。実行器の仕様は設計文書 §3 の GatherElements の行）。"""

    op_domain = ""

    def _run(self, data, indices, axis=0):
        return (np.take_along_axis(data, indices, axis=axis),)


def evaluator(proto: onnx.ModelProto) -> reference.ReferenceEvaluator:
    return reference.ReferenceEvaluator(proto, new_ops=[GatherElements])


def load(model_bytes: bytes, shards: dict[str, bytes], tmp: Path) -> onnx.ModelProto:
    for n, b in shards.items():
        (tmp / n).write_bytes(b)
    (tmp / "model.onnx").write_bytes(model_bytes)
    return onnx.load(str(tmp / "model.onnx"), load_external_data=True)


def _stand_in(base, cfg: Qwen2Config, *, random_b: bool = True):
    """学習済みの代わりに、乱数重み＋LoRA（b も乱数）のモデルを持つ最小の `trained`。"""
    rng = np.random.default_rng(1)
    model = Qwen2Model(cfg)
    flat = tree_flatten(model.parameters())
    model.update(
        tree_unflatten(
            [(k, mx.array(rng.normal(0, 0.2, v.shape).astype(np.float32))) for k, v in flat]
        )
    )
    apply_lora(model, num_layers=2, rank=2, scale=4.0, dropout=0.0, seed=0, rss_bytes=1 << 33)
    if random_b:
        for blk in model.model.layers[-2:]:
            for lin in (blk.self_attn.q_proj, blk.mlp.down_proj):
                lin.lora_b = mx.array(rng.normal(0, 0.3, lin.lora_b.shape).astype(np.float32))
    model.eval()
    return SimpleNamespace(
        model=model,
        tokenizer=base.tokenizer,
        label_order=base.label_order,
        resource_budget=base.resource_budget,
    )


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return _train(tmp_path_factory.mktemp("q"), iters=2)[1]


@pytest.fixture(scope="module", params=["tie", "untied"])
def exported(request, base, tmp_path_factory):
    c = base.model.config
    trained = (
        base
        if request.param == "tie"
        else _stand_in(base, Qwen2Config(**{**c.__dict__, "tie_word_embeddings": False}))
    )
    mb, shards, infos = export(trained, shard_max_bytes=SHARD_MAX, external_threshold=THRESHOLD)
    return trained, mb, shards, infos, load(mb, shards, tmp_path_factory.mktemp("x"))


def _r(a: np.ndarray) -> np.ndarray:
    """bf16 へ丸めて f32 へ戻す。"""
    u = np.frombuffer(qo.bf16_bytes(a), dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32).reshape(a.shape)


def reference_model(trained) -> Qwen2Model:
    """統合・bf16 丸め済みの重みを持つ f32 の MLX モデル（数値照合の基準）。"""
    c = trained.model.config
    m, inner = trained.model, trained.model.model
    p: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": _r(qo._f32(inner.embed_tokens.weight)),
        "model.norm.weight": _r(qo._f32(inner.norm.weight)),
    }
    if not c.tie_word_embeddings:
        p["lm_head.weight"] = _r(qo._f32(m.lm_head.weight))
    for i, blk in enumerate(inner.layers):
        k = f"model.layers.{i}."
        p[k + "input_layernorm.weight"] = _r(qo._f32(blk.input_layernorm.weight))
        p[k + "post_attention_layernorm.weight"] = _r(qo._f32(blk.post_attention_layernorm.weight))
        for pre, owner, names in (
            ("self_attn.", blk.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp.", blk.mlp, ("gate_proj", "up_proj", "down_proj")),
        ):
            for n in names:
                w, b = qo._fused(getattr(owner, n))
                p[k + pre + n + ".weight"] = _r(w)
                if b is not None:
                    p[k + pre + n + ".bias"] = _r(b)
    ref = Qwen2Model(c)
    ref.update(tree_unflatten([(k, mx.array(v)) for k, v in p.items()]))
    ref.eval()
    return ref


def test_ops_within_design_list(exported) -> None:
    """REQ-41・TASK-41.6-2: 演算・属性・dtype・rank が設計文書 §3・§4 の範囲に収まる。"""
    proto = exported[4]
    ops = allowed_ops()
    assert len(ops) == 28
    assert [(o.domain, o.version) for o in proto.opset_import] == [("", 13)]
    assert proto.ir_version == 8
    g = proto.graph
    assert {n.op_type for n in g.node} <= ops
    for n in g.node:
        assert n.domain == ""
        assert len(n.attribute) <= 16
        assert len(n.input) <= 16
        assert len(n.name.encode()) <= 128
        assert all(a.type != onnx.AttributeProto.GRAPH for a in n.attribute)
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        if n.op_type == "Gemm":
            assert a == {"transB": 1}
        elif n.op_type == "Cast":
            assert a == {"to": TensorProto.FLOAT}
        elif n.op_type in ("Softmax", "LogSoftmax"):
            assert a == {"axis": -1}
        elif n.op_type == "ReduceMean":
            assert a == {"axes": [-1], "keepdims": 1}
        elif n.op_type == "ReduceSum":
            assert a == {"keepdims": 0}
        elif n.op_type == "Split":
            assert a == {"axis": -1}
    for t in g.initializer:
        assert t.data_type in (TensorProto.FLOAT, TensorProto.INT64, TensorProto.BFLOAT16)
        assert len(t.dims) <= 2
    for vi in onnx.shape_inference.infer_shapes(proto).graph.value_info:
        assert len(vi.type.tensor_type.shape.dim) <= 5


def test_io_contract(exported) -> None:
    """REQ-33: 入出力は ids INT64 [N,T] → probs FLOAT [N,K]。"""
    trained, _, _, _, proto = exported
    (i,), (o,) = proto.graph.input, proto.graph.output
    assert (i.name, i.type.tensor_type.elem_type) == ("ids", TensorProto.INT64)
    assert (o.name, o.type.tensor_type.elem_type) == ("probs", TensorProto.FLOAT)
    assert o.type.tensor_type.shape.dim[1].dim_value == len(trained.label_order)


def test_numeric_match_with_fused_bf16_reference(exported) -> None:
    """REQ-41・REQ-28: ONNX の probs が、統合・bf16 丸め済み重みの MLX 採点と一致する。"""
    trained, _, _, _, proto = exported
    ref = reference_model(trained)
    tok, order = trained.tokenizer, trained.label_order
    label_ids = q.choice_token_ids(tok, order, trained.model.config.vocab_size)
    pad = tok.special_ids["endoftext"]
    sess = evaluator(proto)
    for text in PROMPTS:
        ids = tok.build_chat_ids(q._system_prompt(order), text, add_generation_prompt=True)
        ll = q.score_labels(ref, ids, label_ids, pad)
        want = np.exp(ll - ll.max())
        want /= want.sum()
        (got,) = sess.run(None, {"ids": np.array([ids], dtype=np.int64)})
        assert got.shape == (1, len(order))
        np.testing.assert_allclose(got[0], want, atol=TOL)
        assert int(got[0].argmax()) == int(want.argmax())


#: f32 の演算順（Gemm・Softmax の縮約順）の違いに由来する差の上限。実測の最大差は 7.2e-7
#: （合成モデル 2 構成・3 入力、2026-10-17）で、その約 14 倍。緩めて通すことはしない。
TOL = 1e-5


def test_lora_is_merged(exported) -> None:
    """REQ-41: 統合後の重みはベース重みと異なる（LoRA が効いている）。"""
    lin = exported[0].model.model.layers[-1].self_attn.q_proj
    assert not np.array_equal(qo._fused(lin)[0], qo._f32(lin.linear.weight))


def test_shard_rules(exported) -> None:
    """REQ-39・#580・#581: shard の上限・key・location・範囲・sha256。"""
    _, mb, shards, infos, _ = exported
    assert len(shards) >= 2
    assert [i.name for i in infos] == [qo.shard_name(n) for n in range(len(infos))]
    assert sorted(shards) == [i.name for i in infos]
    for i in infos:
        assert i.size == len(shards[i.name]) <= SHARD_MAX
        assert i.sha256 == hashlib.sha256(shards[i.name]).hexdigest()
        assert re.fullmatch(r"[A-Za-z0-9._-]{1,128}", i.name)
        assert not i.name.startswith(".")
    raw = onnx.load_from_string(mb)  # 外部データ未解決の本体
    spans: dict[str, list[tuple[int, int]]] = {}
    for t in raw.graph.initializer:
        if t.data_location != TensorProto.EXTERNAL:
            continue
        kv = {e.key: e.value for e in t.external_data}
        assert set(kv) == {"location", "offset", "length"}
        off, ln = int(kv["offset"]), int(kv["length"])
        assert off + ln <= len(shards[kv["location"]])
        spans.setdefault(kv["location"], []).append((off, off + ln))
    assert spans
    for s in spans.values():
        s.sort()
        assert all(a[1] <= b[0] for a, b in itertools.pairwise(s))
    assert not any(t.data_type == TensorProto.BOOL for t in raw.graph.initializer)


def test_tensor_over_shard_limit_is_limit_exceeded(base) -> None:
    """REQ-39: 1 テンソルが shard 上限を超えるなら limit_exceeded(20)。"""
    with pytest.raises(WorkerError) as e:
        export(base, shard_max_bytes=500, external_threshold=THRESHOLD)
    assert e.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_estimated_size_over_limit_is_limit_exceeded(base, monkeypatch) -> None:
    """REQ-39・#582: 種類別のモデル上限を超える推定は、shard を開く前に limit_exceeded。"""
    monkeypatch.setattr(limits, "MAX_MODEL_BYTES_BY_KIND", {"qwen2_lora": 1000})
    opened: list[str] = []
    with pytest.raises(WorkerError) as e:
        qo.export_scoring_onnx(base, io.BytesIO(), lambda n: opened.append(n) or Keep())
    assert e.value.exit_code == ExitCode.LIMIT_EXCEEDED
    assert opened == []


def test_too_many_shards_is_limit_exceeded(base, monkeypatch) -> None:
    """REQ-39・#580: shard の件数上限を超えるなら limit_exceeded。"""
    monkeypatch.setattr(qo, "MAX_SHARDS", 1)
    with pytest.raises(WorkerError) as e:
        export(base, shard_max_bytes=SHARD_MAX, external_threshold=THRESHOLD)
    assert e.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_non_finite_weight_is_runtime_error() -> None:
    """非有限の重みは書かずに runtime_error(70)。"""
    with pytest.raises(WorkerError) as e:
        qo.bf16_bytes(np.array([1.0, np.nan], dtype=np.float32))
    assert e.value.exit_code == ExitCode.RUNTIME_ERROR


def test_bf16_rounding_is_rne() -> None:
    """bf16 は最近接偶数丸め。1+2^-8 は偶数側（1.0）、1+3*2^-8 は 1+2^-6 側へ。"""
    a = np.array([1 + 2**-8, 1 + 3 * 2**-8, -2.5], dtype=np.float32)
    assert _r(a).tolist() == [1.0, 1 + 2**-6, -2.5]


def test_node_and_initializer_counts_at_28_layers(base) -> None:
    """REQ-41・#431: 層数 28（1.5B と同じ層構造・次元は極小）の件数が #581 の暫定上限内。

    証拠種別: テストハーネス。件数は層数と tie の有無だけで決まる。値は設計文書 §4 に記録する。
    """
    c = base.model.config
    t = _stand_in(base, Qwen2Config(**{**c.__dict__, "num_hidden_layers": 28}), random_b=False)
    mb, _, _ = export(t, shard_max_bytes=SHARD_MAX, external_threshold=THRESHOLD)
    proto = onnx.load_from_string(mb)
    print("nodes", len(proto.graph.node), "initializers", len(proto.graph.initializer))
    assert len(proto.graph.node) <= 4096
    assert len(proto.graph.initializer) <= 1024
