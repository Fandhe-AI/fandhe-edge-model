"""`qwen2_lora` の LoRA 統合済み重みを ONNX の採点グラフとして書き出す（REQ-41・#431）。

学習ワーカーの `Qwen2LoraKind` 側（#432 が CLI の `_worker` へ接続する）から呼ばれる。出力は
`docs/design/qwen2-scoring-onnx-ops.md`（#415。演算一覧が唯一の正）の範囲に収めた opset 13・IR 8 の
グラフで、入出力契約は `ids: INT64 [N,T]` → `probs: FLOAT [N,K]`（N は常に 1）。選択肢の
採点（ラベル token 列＋`<|im_end|>` の対数尤度の合計 → 選択肢間 softmax。`kinds/qwen2_lora.py` の
`score_chunk` と同じ）はグラフの中で行い、共通 prefix は 1 回だけ forward して各層の K/V を
選択肢ブロックから参照する（KV の使い回し）。

重みは 1 テンソルずつ `W' = W + scale * (A @ B).T`（f32）で統合して bf16（RNE）にし、閾値以上は
外部データの shard（`weights-NNNNN.bin`）へ逐次書く。全量を同時には持たない。shard 名・
sha256 の記録（artifact.json）は `Qwen2LoraKind.export_onnx`（#432）、範囲外構成の除外理由の
記録は #433、Rust ランタイムとの全件一致は #434 の担当で、本モジュールは model.onnx と
shard を呼び出し側のストリームへ書くだけ。
経路は扱わない（書き込み先は `out` と `open_shard` が返すストリームのみ）。通信しない（REQ-38）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Any

import numpy as np
import onnx
from onnx import TensorProto, helper

from . import limits
from .artifact import HashingWriter
from .errors import WorkerError
from .exitcode import ExitCode
from .kinds.qwen2_lora import Qwen2LoraTrainedModel, choice_token_ids

OPSET = 13
IR_VERSION = 8
#: マスクの遮蔽値。f32 の Softmax で exp が 0 になる十分小さな有限値（-inf は使わない）。
NEG = -1e30
#: shard 1 件の上限（#580・#581。Rust ガード層と同値）と件数の上限。
SHARD_MAX_BYTES = 1 << 30
MAX_SHARDS = 256
#: これ未満の initializer は inline の raw_data にする。
EXTERNAL_THRESHOLD = 1024
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class ShardInfo:
    """書き出した shard 1 件。artifact.json への記録は `Qwen2LoraKind.export_onnx`（#432）。"""

    name: str
    size: int
    sha256: str


def _limit(message: str) -> WorkerError:
    return WorkerError("limit_exceeded", message, ExitCode.LIMIT_EXCEEDED)


def shard_name(index: int) -> str:
    """shard のファイル名。Rust 側 `location_ok` の字句規則（英数字と `._-`）を満たす。"""
    return f"weights-{index:05d}.bin"


def bf16_bytes(a: np.ndarray) -> bytes:
    """f32 配列を bf16（RNE。リトルエンディアンの生バイト）へ丸める。非有限値は `runtime_error`。"""
    a = np.ascontiguousarray(a, dtype="<f4")
    if not np.isfinite(a).all():
        raise WorkerError("runtime_error", "non-finite weight", ExitCode.RUNTIME_ERROR)
    u = a.view(np.uint32)
    r = (u + ((u >> 16) & 1) + np.uint32(0x7FFF)) >> 16
    r = r.astype("<u2")
    # 丸めで指数部が全ビット 1 になる場合（f32 最大有限値の +inf 化等）も拒否する
    if ((r & 0x7F80) == 0x7F80).any():
        raise WorkerError(
            "runtime_error", "non-finite weight after bf16 rounding", ExitCode.RUNTIME_ERROR
        )
    return r.tobytes()


class _Shards:
    """外部データの shard 群。1 テンソルは 1 shard に収め、超えるなら次の shard へ切り替える。"""

    def __init__(self, open_shard: Callable[[str], IO[bytes]], max_bytes: int) -> None:
        self._open, self._max = open_shard, max_bytes
        self._cur: tuple[str, IO[bytes], HashingWriter, int] | None = None
        self.done: list[ShardInfo] = []

    def _close(self) -> None:
        if self._cur is not None:
            name, f, h, size = self._cur
            f.close()
            self.done.append(ShardInfo(name, size, h.hexdigest()))
            self._cur = None

    def add(self, data: bytes) -> tuple[str, int]:
        """`data` を書き、(shard 名, offset) を返す。"""
        if len(data) > self._max:
            raise _limit("a tensor exceeds the shard size limit")
        if self._cur is not None and self._cur[3] + len(data) > self._max:
            self._close()
        if self._cur is None:
            if len(self.done) >= MAX_SHARDS:
                raise _limit("too many weight shards")
            name = shard_name(len(self.done))
            f = self._open(name)
            self._cur = (name, f, HashingWriter(f), 0)
        name, f, h, size = self._cur
        h.write(data)
        self._cur = (name, f, h, size + len(data))
        return name, size

    def finish(self) -> list[ShardInfo]:
        self._close()
        return self.done


class _Graph:
    """ノード・initializer を溜めるビルダー。重みは bf16 にして shard か inline へ書く。"""

    def __init__(self, shards: _Shards, threshold: int, rb: Any) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._shards, self._threshold, self._rb = shards, threshold, rb
        self._n = 0
        self._consts: dict[tuple, str] = {}

    def fresh(self, base: str) -> str:
        self._n += 1
        return f"{base}_{self._n}"

    def const(self, value: Any, dtype: Any = np.float32) -> str:
        """小さな定数（FLOAT・INT64 の inline initializer）。同じ値は 1 つを共有する。"""
        arr = np.asarray(value, dtype=dtype)
        key = (arr.dtype.str, arr.shape, arr.tobytes())
        if key not in self._consts:
            name = self.fresh("c")
            self.inits.append(onnx.numpy_helper.from_array(arr, name))
            self._consts[key] = name
        return self._consts[key]

    def i64(self, *vals: int) -> str:
        return self.const(list(vals), np.int64)

    def weight(self, name: str, arr: np.ndarray) -> str:
        """bf16 の重み。`Cast(to=FLOAT)` の出力名を返す（消費側は常に f32 として使う）。"""
        self._rb.check()
        raw = bf16_bytes(arr)
        t = TensorProto()
        t.name, t.data_type = name, TensorProto.BFLOAT16
        t.dims.extend(arr.shape)
        if len(raw) < self._threshold:
            t.raw_data = raw
        else:
            loc, off = self._shards.add(raw)
            t.data_location = TensorProto.EXTERNAL
            # `set_external_data` は raw_data を要求するため、位置情報を直接持たせる
            for k, v in (("location", loc), ("offset", off), ("length", len(raw))):
                e = t.external_data.add()
                e.key, e.value = k, str(v)
        self.inits.append(t)
        return self.op("Cast", [name], to=TensorProto.FLOAT)

    def op(self, op_type: str, inputs: list[str], out: str | None = None, **attrs: Any) -> str:
        out = out or self.fresh(op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [out], name=self.fresh("n"), **attrs))
        return out

    def split2(self, x: str, half: int) -> tuple[str, str]:
        a, b = self.fresh("sp"), self.fresh("sp")
        self.nodes.append(
            helper.make_node(
                "Split", [x, self.i64(half, half)], [a, b], name=self.fresh("n"), axis=-1
            )
        )
        return a, b


@dataclass(frozen=True)
class _Dims:
    nh: int
    nkv: int
    hd: int
    eps: float

    @property
    def grp(self) -> int:
        return self.nh // self.nkv


def _rms(g: _Graph, d: _Dims, x: str, w: str) -> str:
    m = g.op("ReduceMean", [g.op("Mul", [x, x])], axes=[-1], keepdims=1)
    r = g.op("Reciprocal", [g.op("Sqrt", [g.op("Add", [m, g.const(d.eps)])])])
    return g.op("Mul", [g.op("Mul", [x, r]), w])


def _lin(g: _Graph, x: str, w: str, b: str | None = None) -> str:
    return g.op("Gemm", [x, w, b] if b else [x, w], transB=1)


def _rope_table(g: _Graph, pos: str, hd: int, theta: float) -> tuple[str, str]:
    """INT64 の位置 `[S]` から RoPE の cos・sin `[S, hd/2]`（HF の式 1/theta**(2i/hd)）。"""
    inv = (theta ** (-np.arange(hd // 2, dtype=np.float64) / (hd // 2))).astype(np.float32)
    posf = g.op("Unsqueeze", [g.op("Cast", [pos], to=TensorProto.FLOAT), g.i64(1)])
    ang = g.op("Mul", [posf, g.const(inv[None, :])])
    return g.op("Cos", [ang]), g.op("Sin", [ang])


def _rope(g: _Graph, x: str, cs: tuple[str, str], hd: int) -> str:
    cos, sin = cs
    x1, x2 = g.split2(x, hd // 2)
    lo = g.op("Sub", [g.op("Mul", [x1, cos]), g.op("Mul", [x2, sin])])
    hi = g.op("Add", [g.op("Mul", [x1, sin]), g.op("Mul", [x2, cos])])
    return g.op("Concat", [lo, hi], axis=-1)


def _layer(
    g: _Graph,
    d: _Dims,
    w: dict[str, str],
    x: str,
    b: int,
    cs: tuple[str, str],
    mask: str,
    kv_prefix: tuple[str, str] | None,
) -> tuple[str, str, str]:
    """1 層。prefix（b=1・kv_prefix なし）と選択肢ブロック（b=K・kv_prefix あり）で共有する。

    隠れ状態は `[b*S, H]`。`(出力, K, V)` を返し、K・V は RoPE 後の `[b, nkv, 1, S, hd]`。
    """
    nh, nkv, hd, grp = d.nh, d.nkv, d.hd, d.grp
    a = _rms(g, d, x, w["ln1"])
    q = g.op("Reshape", [_lin(g, a, w["q"], w["qb"]), g.i64(b, -1, nkv, grp, hd)])
    q = _rope(g, g.op("Transpose", [q], perm=[0, 2, 3, 1, 4]), cs, hd)  # [b,nkv,grp,S,hd]
    k = g.op("Reshape", [_lin(g, a, w["k"], w["kb"]), g.i64(b, -1, nkv, hd)])
    k = _rope(g, g.op("Transpose", [k], perm=[0, 2, 1, 3]), cs, hd)
    k = g.op("Unsqueeze", [k, g.i64(2)])  # [b,nkv,1,S,hd]
    v = g.op("Reshape", [_lin(g, a, w["v"], w["vb"]), g.i64(b, -1, nkv, hd)])
    v = g.op("Unsqueeze", [g.op("Transpose", [v], perm=[0, 2, 1, 3]), g.i64(2)])
    kf, vf = k, v
    if kv_prefix is not None:
        shape = g.i64(b, 1, 1, 1, 1)
        kf = g.op("Concat", [g.op("Expand", [kv_prefix[0], shape]), k], axis=3)
        vf = g.op("Concat", [g.op("Expand", [kv_prefix[1], shape]), v], axis=3)
    s = g.op("MatMul", [q, g.op("Transpose", [kf], perm=[0, 1, 2, 4, 3])])
    s = g.op("Where", [mask, g.const(NEG), g.op("Mul", [s, g.const(hd**-0.5)])])
    ctx = g.op("MatMul", [g.op("Softmax", [s], axis=-1), vf])  # [b,nkv,grp,S,hd]
    ctx = g.op("Reshape", [g.op("Transpose", [ctx], perm=[0, 3, 1, 2, 4]), g.i64(-1, nh * hd)])
    h = g.op("Add", [x, _lin(g, ctx, w["o"])])
    m = _rms(g, d, h, w["ln2"])
    gate = _lin(g, m, w["gate"])
    act = g.op("Mul", [gate, g.op("Sigmoid", [gate])])
    h = g.op("Add", [h, _lin(g, g.op("Mul", [act, _lin(g, m, w["up"])]), w["down"])])
    return h, k, v


def _f32(a: Any) -> np.ndarray:
    import mlx.core as mx

    return np.array(a.astype(mx.float32))


def _fused(lin: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Linear か LoRALinear の (重み, bias)。LoRA は `W + scale * (A @ B).T`（f32）で統合する。"""
    from .qwen2_model import LoRALinear

    base = lin.linear if isinstance(lin, LoRALinear) else lin
    w = _f32(base.weight)
    if isinstance(lin, LoRALinear):
        w = w + (lin.scale * (_f32(lin.lora_a) @ _f32(lin.lora_b)).T).astype(np.float32)
    return w, (_f32(base.bias) if "bias" in base else None)


def export_scoring_onnx(
    trained: Qwen2LoraTrainedModel,
    out: IO[bytes],
    open_shard: Callable[[str], IO[bytes]],
    *,
    shard_max_bytes: int = SHARD_MAX_BYTES,
    external_threshold: int = EXTERNAL_THRESHOLD,
) -> list[ShardInfo]:
    """採点グラフ（model.onnx）を `out` へ、外部データを `open_shard(name)` のストリームへ書く。

    `open_shard` が返すストリームは本関数が閉じる。返す `ShardInfo`（名前・大きさ・sha256）の
    記録は呼び出し側（`Qwen2LoraKind.export_onnx`。#432）の責務。上限超過（推定サイズ・
    1 テンソルが shard 上限超・shard 件数）は書く前または書き始めに `limit_exceeded`（20）、
    非有限の重みは `runtime_error`（70）。
    """
    model, tok, order = trained.model, trained.tokenizer, trained.label_order
    c = model.config
    cap = limits.max_model_bytes_for_kind("qwen2_lora")
    if c.estimated_params() * 2 > cap:
        raise _limit("estimated model size exceeds the limit")
    cand = np.array(
        _pad(choice_token_ids(tok, order, c.vocab_size), tok.special_ids["endoftext"]),
        dtype=np.int64,
    )  # [K, M]
    k_n, m = cand.shape
    if m < 2:  # ラベル 1 token 以上＋im_end なので起きない。起きたらグラフの前提が崩れる
        raise WorkerError("runtime_error", "choice block is too short", ExitCode.RUNTIME_ERROR)
    weights = np.zeros((k_n, m), dtype=np.float32)
    for i, ids in enumerate(choice_token_ids(tok, order, c.vocab_size)):
        weights[i, : len(ids)] = 1.0  # pad 位置は 0

    shards = _Shards(open_shard, shard_max_bytes)
    try:
        g = _Graph(shards, external_threshold, trained.resource_budget)
        proto = _build(g, model, c, cand, weights, k_n, m)
        shards_done = shards.finish()
    except BaseException:
        shards._close()  # 途中で止めても開いた shard は閉じる
        raise
    if sum(s.size for s in shards_done) > cap:
        raise _limit("written model size exceeds the limit")
    _check(proto)
    out.write(proto.SerializeToString())
    return shards_done


def _check(proto: onnx.ModelProto) -> None:
    """`onnx.checker` で演算・属性・型を検査する。

    checker は外部データの実ファイルを要求するため、外部の重みは「グラフ入力」に置き換えた
    写しで検査する（重みの値は検査対象外。値は書き出し時に有限性を確認済み）。
    """
    copy = onnx.ModelProto()
    copy.CopyFrom(proto)
    keep = [t for t in copy.graph.initializer if t.data_location != TensorProto.EXTERNAL]
    for t in copy.graph.initializer:
        if t.data_location == TensorProto.EXTERNAL:
            copy.graph.input.append(
                helper.make_tensor_value_info(t.name, t.data_type, list(t.dims))
            )
    del copy.graph.initializer[:]
    copy.graph.initializer.extend(keep)
    onnx.checker.check_model(copy, full_check=True)


def _pad(rows: list[list[int]], pad_id: int) -> list[list[int]]:
    m = max(len(r) for r in rows)
    return [r + [pad_id] * (m - len(r)) for r in rows]


def _build(
    g: _Graph, model: Any, c: Any, cand: np.ndarray, weights: np.ndarray, k_n: int, m: int
) -> onnx.ModelProto:
    d = _Dims(c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rms_norm_eps)
    inner = model.model
    embed = g.weight("embed", _f32(inner.embed_tokens.weight))
    ids = "ids"
    flat = g.op("Reshape", [ids, g.i64(-1)])
    t_len = g.op("Gather", [g.op("Shape", [ids]), g.const(1, np.int64)], axis=0)  # スカラー T
    zero, one = g.const(0, np.int64), g.const(1, np.int64)
    pos = g.op("Range", [zero, t_len, one])
    bpos = g.op("Add", [t_len, g.const(np.arange(m), np.int64)])  # [M]。ブロックの位置
    total = g.op("Range", [zero, g.op("Add", [t_len, g.const(m, np.int64)]), one])
    # 因果マスク（真 = 遮蔽）。prefix は [T,T]、ブロックは [M,T+M]（prefix は全部見える）
    pmask = g.op(
        "Greater", [g.op("Unsqueeze", [pos, g.i64(0)]), g.op("Unsqueeze", [pos, g.i64(1)])]
    )
    bmask = g.op(
        "Greater", [g.op("Unsqueeze", [total, g.i64(0)]), g.op("Unsqueeze", [bpos, g.i64(1)])]
    )
    pcs, bcs = _rope_table(g, pos, d.hd, c.rope_theta), _rope_table(g, bpos, d.hd, c.rope_theta)

    layer_w: list[dict[str, str]] = []
    kvs: list[tuple[str, str]] = []
    x = g.op("Gather", [embed, flat], axis=0)  # [T,H]
    for i, blk in enumerate(inner.layers):
        p = f"l{i}."
        w: dict[str, str] = {
            "ln1": g.weight(p + "ln1", _f32(blk.input_layernorm.weight)),
            "ln2": g.weight(p + "ln2", _f32(blk.post_attention_layernorm.weight)),
        }
        for key, lin in (
            ("q", blk.self_attn.q_proj),
            ("k", blk.self_attn.k_proj),
            ("v", blk.self_attn.v_proj),
            ("o", blk.self_attn.o_proj),
            ("gate", blk.mlp.gate_proj),
            ("up", blk.mlp.up_proj),
            ("down", blk.mlp.down_proj),
        ):
            wt, bias = _fused(lin)
            w[key] = g.weight(p + key, wt)
            if bias is not None:
                w[key + "b"] = g.weight(p + key + "b", bias)
        layer_w.append(w)
        x, k, v = _layer(g, d, w, x, 1, pcs, pmask, None)
        kvs.append((k, v))
    last = g.op("Slice", [x, g.i64(-1), g.i64(_INT64_MAX), g.i64(0)])  # prefix の最終行 [1,H]

    y = g.op("Gather", [embed, g.const(cand, np.int64)], axis=0)  # [K,M,H]
    y = g.op("Reshape", [y, g.i64(-1, c.hidden_size)])
    for w, kv in zip(layer_w, kvs, strict=True):
        y, _, _ = _layer(g, d, w, y, k_n, bcs, bmask, kv)

    # 行 j の予測対象は cand[j]（先頭は prefix の最終行、以降はブロックの先頭 M-1 行）
    y = g.op("Reshape", [y, g.i64(k_n, m, c.hidden_size)])
    y = g.op("Slice", [y, g.i64(0), g.i64(m - 1), g.i64(1)])
    head = g.op("Expand", [g.op("Reshape", [last, g.i64(1, 1, c.hidden_size)]), g.i64(k_n, 1, 1)])
    h = g.op("Reshape", [g.op("Concat", [head, y], axis=1), g.i64(-1, c.hidden_size)])
    h = _rms(g, d, h, g.weight("norm", _f32(inner.norm.weight)))
    if c.tie_word_embeddings:
        lm = embed
    else:
        lm = g.weight("lm_head", _f32(model.lm_head.weight))
    logp = g.op(
        "LogSoftmax",
        [g.op("Reshape", [g.op("Gemm", [h, lm], transB=1), g.i64(k_n, m, -1)])],
        axis=-1,
    )
    picked = g.op(
        "GatherElements",
        [logp, g.op("Unsqueeze", [g.const(cand, np.int64), g.i64(2)])],
        axis=2,
    )  # [K,M,1]。最後の軸（ONNX 1.23 の ReferenceEvaluator は axis=-1 を受け付けないため正の値）
    picked = g.op("Reshape", [picked, g.i64(k_n, m)])
    ll = g.op("ReduceSum", [g.op("Mul", [picked, g.const(weights)]), g.i64(1)], keepdims=0)
    g.op("Softmax", [g.op("Unsqueeze", [ll, g.i64(0)])], out="probs", axis=-1)

    graph = helper.make_graph(
        g.nodes,
        "qwen2_lora_scoring",
        [helper.make_tensor_value_info(ids, TensorProto.INT64, ["N", "T"])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["N", k_n])],
        initializer=g.inits,
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    proto.ir_version = IR_VERSION
    return proto
