"""`qwen2_lora` の LoRA 統合済み重みを ONNX の採点グラフとして書き出す（REQ-41・#431）。

学習ワーカーの `Qwen2LoraKind` 側（#432 が CLI の `_worker` へ接続する）から呼ばれる。出力は
`docs/design/qwen2-scoring-onnx-ops.md`（#415。演算一覧が唯一の正）の範囲に収めた opset 13・IR 8 の
グラフで、入出力契約は `ids: INT64 [N,T]` → `probs: FLOAT [N,K]`（N は常に 1）。選択肢の
採点（ラベル token 列＋`<|im_end|>` の対数尤度の合計 → 選択肢間 softmax。`kinds/qwen2_lora.py` の
`score_chunk` と同じ）はグラフの中で行い、共通 prefix は 1 回だけ forward して各層の K/V を
選択肢ブロックから参照する（KV の使い回し）。選択肢の token 列は木（接頭辞 = ノード）にし、
共通の頭と pad の行を計算しない: ノード列を b=1 で全層へ通し（各ノードは prefix の全部と
自分の祖先だけを見て、RoPE の位置は T＋深さ）、lm_head は root と子を持つノードの行だけ、
各選択肢の対数尤度は経路上のノードの対数確率の和にする（REQ-41・#684）。
system 区間（`ids` の先頭 P 個。入力に依らず不変）の K/V は書き出し時に numpy（f32）で求めて
f32 の initializer として同梱し、グラフは `ids[:, P:]` だけを forward する（REQ-41・#634）。
グラフ入力は従来どおり全体の `ids` で、`ids[:P]` は読み捨てる（system 区間の一致は呼び出し側が
`Qwen2Tokenizer.system_turn_ids` で保証する）。

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
#: shard 1 件の上限（2 GiB）と件数の上限。Rust ガード層 `split_files::MAX_SHARD_FILE_BYTES` と同値。
#: 7B の埋め込み / lm_head 1 テンソル（約 1.09 GB）が 1 shard に収まる大きさ。
#: オーナー判断 2026-10-25・#654（証拠種別: 推定）。
SHARD_MAX_BYTES = 2 << 30
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


def bf16_round(a: np.ndarray) -> np.ndarray:
    """bf16（RNE）へ丸めて f32 へ戻す。グラフ内の `Cast(bf16→f32)` と同値。"""
    u = np.frombuffer(bf16_bytes(a), dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32).reshape(a.shape)


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
        self._store(name, TensorProto.BFLOAT16, arr.shape, bf16_bytes(arr))
        return self.op("Cast", [name], to=TensorProto.FLOAT)

    def kv(self, name: str, arr: np.ndarray) -> str:
        """system 区間の K/V（f32 の rank 2 initializer）。名前をそのまま返す（#634）。"""
        self._rb.check()
        self._store(name, TensorProto.FLOAT, arr.shape, np.ascontiguousarray(arr, "<f4").tobytes())
        return name

    def _store(self, name: str, dtype: int, shape: tuple[int, ...], raw: bytes) -> None:
        t = TensorProto()
        t.name, t.data_type = name, dtype
        t.dims.extend(shape)
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
    cs: tuple[str, str],
    mask: str,
    kv_prefix: tuple[str, str] | None,
    *,
    additive_mask: bool = False,
) -> tuple[str, str, str]:
    """1 層。入力テキスト（kv_prefix は system の同梱 K/V）と選択肢の木のノード列（
    kv_prefix は system＋入力テキストの K/V。#684）で共有する。

    `additive_mask` が偽なら mask は BOOL（真 = 遮蔽）で `Where`、真なら FLOAT（遮蔽 = NEG）で
    スコアに加える（BOOL は Concat できないため、実行時に長さが決まる列を含む mask に使う）。

    隠れ状態は `[S, H]`。`(出力, K, V)` を返し、K・V は kv_prefix を連結した後の
    RoPE 後 `[1, nkv, 1, 全長, hd]`（次の段の kv_prefix にそのまま使える）。
    """
    nh, nkv, hd, grp = d.nh, d.nkv, d.hd, d.grp
    a = _rms(g, d, x, w["ln1"])
    q = g.op("Reshape", [_lin(g, a, w["q"], w["qb"]), g.i64(1, -1, nkv, grp, hd)])
    q = _rope(g, g.op("Transpose", [q], perm=[0, 2, 3, 1, 4]), cs, hd)  # [1,nkv,grp,S,hd]
    k = g.op("Reshape", [_lin(g, a, w["k"], w["kb"]), g.i64(1, -1, nkv, hd)])
    k = _rope(g, g.op("Transpose", [k], perm=[0, 2, 1, 3]), cs, hd)
    k = g.op("Unsqueeze", [k, g.i64(2)])  # [1,nkv,1,S,hd]
    v = g.op("Reshape", [_lin(g, a, w["v"], w["vb"]), g.i64(1, -1, nkv, hd)])
    v = g.op("Unsqueeze", [g.op("Transpose", [v], perm=[0, 2, 1, 3]), g.i64(2)])
    kf, vf = k, v
    if kv_prefix is not None:
        pk, pv = kv_prefix
        kf = g.op("Concat", [pk, k], axis=3)
        vf = g.op("Concat", [pv, v], axis=3)
    s = g.op("MatMul", [q, g.op("Transpose", [kf], perm=[0, 1, 2, 4, 3])])
    s = g.op("Mul", [s, g.const(hd**-0.5)])
    s = g.op("Add", [s, mask]) if additive_mask else g.op("Where", [mask, g.const(NEG), s])
    ctx = g.op("MatMul", [g.op("Softmax", [s], axis=-1), vf])  # [1,nkv,grp,S,hd]
    ctx = g.op("Reshape", [g.op("Transpose", [ctx], perm=[0, 3, 1, 2, 4]), g.i64(-1, nh * hd)])
    h = g.op("Add", [x, _lin(g, ctx, w["o"])])
    m = _rms(g, d, h, w["ln2"])
    gate = _lin(g, m, w["gate"])
    act = g.op("Mul", [gate, g.op("Sigmoid", [gate])])
    h = g.op("Add", [h, _lin(g, g.op("Mul", [act, _lin(g, m, w["up"])]), w["down"])])
    return h, kf, vf


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


def _np_rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    r = np.float32(1) / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(eps))
    return (x * r * w).astype(np.float32)


def _np_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    x1, x2 = np.split(x, 2, axis=-1)
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def _np_prefix_layer(
    d: _Dims, rw: dict[str, np.ndarray], x: np.ndarray, rope: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """system 区間 1 層分の f32 forward（`_layer` の kv_prefix なしと同じ式）。

    `rw` は bf16 丸め済み（グラフの `Cast(bf16→f32)` 後と同値）の重み。`(出力, K, V)` を返し、
    K・V は RoPE 後の `[nkv, P, hd]`。縮約順は onnx / Rust と異なりうる。K/V の直接比較は
    せず、MLX 参照との採点全件一致（許容差 1e-5・テスト）で間接的に確認する（REQ-28・#634）。
    """
    n = x.shape[0]
    cos, sin = rope

    def lin(a: np.ndarray, key: str) -> np.ndarray:
        y = a @ rw[key].T
        return y + rw[key + "b"] if key + "b" in rw else y

    a = _np_rms(x, rw["ln1"], d.eps)
    q = lin(a, "q").reshape(n, d.nkv, d.grp, d.hd).transpose(1, 2, 0, 3)
    k = lin(a, "k").reshape(n, d.nkv, d.hd).transpose(1, 0, 2)
    v = lin(a, "v").reshape(n, d.nkv, d.hd).transpose(1, 0, 2)
    q, k = _np_rope(q, cos, sin), _np_rope(k, cos, sin)
    s = np.einsum("ngih,njh->ngij", q, k) * np.float32(d.hd**-0.5)
    s = np.where(np.triu(np.ones((n, n), dtype=bool), 1), np.float32(NEG), s)
    s = np.exp(s - s.max(-1, keepdims=True))
    p = s / s.sum(-1, keepdims=True)
    ctx = np.einsum("ngij,njh->ngih", p, v).transpose(2, 0, 1, 3).reshape(n, d.nh * d.hd)
    h = x + lin(ctx, "o")
    m = _np_rms(h, rw["ln2"], d.eps)
    gate = lin(m, "gate")
    act = gate / (np.float32(1) + np.exp(-gate))
    h = h + lin(act * lin(m, "up"), "down")
    return h.astype(np.float32), k.astype(np.float32), v.astype(np.float32)


def export_scoring_onnx(
    trained: Qwen2LoraTrainedModel,
    out: IO[bytes],
    open_shard: Callable[[str], IO[bytes]],
    *,
    system_ids: list[int],
    shard_max_bytes: int = SHARD_MAX_BYTES,
    external_threshold: int = EXTERNAL_THRESHOLD,
) -> list[ShardInfo]:
    """採点グラフ（model.onnx）を `out` へ、外部データを `open_shard(name)` のストリームへ書く。

    `open_shard` が返すストリームは本関数が閉じる。返す `ShardInfo`（名前・大きさ・sha256）の
    記録は呼び出し側（`Qwen2LoraKind.export_onnx`。#432）の責務。上限超過（推定サイズ・
    1 テンソルが shard 上限超・shard 件数）は書く前または書き始めに `limit_exceeded`（20）、
    非有限の重みは `runtime_error`（70）。

    `system_ids` は `Qwen2Tokenizer.system_turn_ids` の system 区間の id 列（P 個）。その K/V を
    グラフへ同梱する（#634）。グラフの前提として実行時の `ids` は P+1 個以上（入力側が 1 行以上）
    必要で、通常経路の `build_chat_ids` は P+3 以上を作る。T<=P の検査はグラフ内に無い。
    """
    if not system_ids:
        raise WorkerError("runtime_error", "system_ids is empty", ExitCode.RUNTIME_ERROR)
    model, tok, order = trained.model, trained.tokenizer, trained.label_order
    c = model.config
    cap = limits.max_model_bytes_for_kind("qwen2_lora")
    if c.estimated_params() * 2 > cap:
        raise _limit("estimated model size exceeds the limit")
    seqs = choice_token_ids(tok, order, c.vocab_size)
    # ラベル 1 token 以上＋im_end なので起きない。起きたら前提が崩れる
    if any(len(x) < 2 for x in seqs):
        raise WorkerError("runtime_error", "choice sequence is too short", ExitCode.RUNTIME_ERROR)

    shards = _Shards(open_shard, shard_max_bytes)
    try:
        g = _Graph(shards, external_threshold, trained.resource_budget)
        proto = _build(g, model, c, _choice_trie(seqs), len(seqs), system_ids)
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


@dataclass(frozen=True)
class _Trie:
    """選択肢の token 列の木（REQ-41・#684）。ノード = 空でない異なる接頭辞。

    ノードは (深さ, 接頭辞) の昇順で並べる（入力順に依らず決定的。親が子より先）。
    `parents` は子を持つ接頭辞（空接頭辞 = root を先頭）で、lm_head の行になる。
    """

    token: list[int]  # ノードの最後の token [N]
    depth: list[int]  # 0 始まりの深さ [N]
    anc: np.ndarray  # [N,N] f32。ノード c が i の祖先または自分なら 0、それ以外は NEG
    rows: list[int]  # 子を持つノード（root 以外）の添字。lm_head の行 1.. に対応
    slots: np.ndarray  # [P,C] int64。親の行ごとの子 token（余りは 0）
    path: np.ndarray  # [K,P*C] f32。選択肢の経路上のノードの slot を 1 にする


def _choice_trie(seqs: list[list[int]]) -> _Trie:
    prefixes = sorted(
        {tuple(s[:i]) for s in seqs for i in range(1, len(s) + 1)}, key=lambda p: (len(p), p)
    )
    index = {p: i for i, p in enumerate(prefixes)}
    parents = sorted({p[:-1] for p in prefixes}, key=lambda p: (len(p), p))  # () が先頭
    children: dict[tuple[int, ...], list[tuple[int, ...]]] = {q: [] for q in parents}
    for p in prefixes:
        children[p[:-1]].append(p)
    width = max(len(v) for v in children.values())
    slot_of: dict[tuple[int, ...], int] = {}
    slots = np.zeros((len(parents), width), dtype=np.int64)
    for r, q in enumerate(parents):
        for j, p in enumerate(children[q]):
            slot_of[p] = r * width + j
            slots[r, j] = p[-1]
    anc = np.full((len(prefixes), len(prefixes)), NEG, dtype=np.float32)
    for i, p in enumerate(prefixes):
        for n in range(1, len(p) + 1):
            anc[i, index[p[:n]]] = 0.0
    path = np.zeros((len(seqs), len(parents) * width), dtype=np.float32)
    for k, sq in enumerate(seqs):
        for n in range(1, len(sq) + 1):
            path[k, slot_of[tuple(sq[:n])]] = 1.0
    return _Trie(
        token=[p[-1] for p in prefixes],
        depth=[len(p) - 1 for p in prefixes],
        anc=anc,
        rows=[index[q] for q in parents[1:]],
        slots=slots,
        path=path,
    )


def _build(
    g: _Graph,
    model: Any,
    c: Any,
    trie: _Trie,
    k_n: int,
    system_ids: list[int],
) -> onnx.ModelProto:
    d = _Dims(c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rms_norm_eps)
    inner = model.model
    emb = _f32(inner.embed_tokens.weight)
    n_sys = len(system_ids)
    if not all(0 <= t < emb.shape[0] for t in system_ids):
        raise WorkerError("runtime_error", "system id out of vocab", ExitCode.RUNTIME_ERROR)
    # system 区間の隠れ状態（f32）。層ごとに更新し、K/V を同梱する（#634）
    sx = bf16_round(emb[np.array(system_ids, dtype=np.int64)])
    embed = g.weight("embed", emb)
    del emb  # 全埋め込みの f32 を層ループの間保持しない
    spos = np.arange(n_sys, dtype=np.float32)[:, None]
    inv = (c.rope_theta ** (-np.arange(d.hd // 2, dtype=np.float64) / (d.hd // 2))).astype(
        np.float32
    )
    srope = (np.cos(spos * inv[None, :]), np.sin(spos * inv[None, :]))

    ids = "ids"
    sys_n = g.i64(n_sys)
    flat = g.op("Reshape", [g.op("Slice", [ids, sys_n, g.i64(_INT64_MAX), g.i64(1)]), g.i64(-1)])
    # スカラー T（ids 全体の長さ）
    t_len = g.op("Gather", [g.op("Shape", [ids]), g.const(1, np.int64)], axis=0)
    zero, one = g.const(0, np.int64), g.const(1, np.int64)
    pos = g.op("Range", [g.const(n_sys, np.int64), t_len, one])  # 入力テキストの位置 P..T-1
    keys = g.op("Range", [zero, t_len, one])  # 入力テキスト側のキー位置 0..T-1
    # 入力テキストの因果マスク（真 = 遮蔽）
    pmask = g.op(
        "Greater", [g.op("Unsqueeze", [keys, g.i64(0)]), g.op("Unsqueeze", [pos, g.i64(1)])]
    )
    # 選択肢の木のノード列: 位置は T + 深さ。マスクは [N,T+N] の加算形式で、prefix 側 T 列は
    # 全部見え（0）、ノード側 N 列は自分と祖先だけ見える（REQ-41・#684）
    n_nodes = len(trie.token)
    bpos = g.op("Add", [t_len, g.const(trie.depth, np.int64)])  # [N]
    zrow = g.op("Mul", [g.op("Cast", [keys], to=TensorProto.FLOAT), g.const(0.0)])  # [T]
    zeros = g.op(
        "Add", [g.op("Unsqueeze", [zrow, g.i64(0)]), g.const(np.zeros((n_nodes, 1), np.float32))]
    )
    bmask = g.op("Concat", [zeros, g.const(trie.anc)], axis=1)
    pcs, bcs = _rope_table(g, pos, d.hd, c.rope_theta), _rope_table(g, bpos, d.hd, c.rope_theta)

    layer_w: list[dict[str, str]] = []
    kvs: list[tuple[str, str]] = []
    x = g.op("Gather", [embed, flat], axis=0)  # [T-P,H]
    kv_shape = g.i64(1, d.nkv, 1, n_sys, d.hd)
    for i, blk in enumerate(inner.layers):
        p = f"l{i}."
        rw: dict[str, np.ndarray] = {}  # この層の bf16 丸め済み重み（system の forward 用）
        w: dict[str, str] = {}
        for key, arr in (
            ("ln1", _f32(blk.input_layernorm.weight)),
            ("ln2", _f32(blk.post_attention_layernorm.weight)),
        ):
            w[key] = g.weight(p + key, arr)
            rw[key] = bf16_round(arr)
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
            rw[key] = bf16_round(wt)
            if bias is not None:
                w[key + "b"] = g.weight(p + key + "b", bias)
                rw[key + "b"] = bf16_round(bias)
        layer_w.append(w)
        sx, sk, sv = _np_prefix_layer(d, rw, sx, srope)
        pk = g.op("Reshape", [g.kv(p + "sysk", sk.reshape(-1, d.hd)), kv_shape])
        pv = g.op("Reshape", [g.kv(p + "sysv", sv.reshape(-1, d.hd)), kv_shape])
        x, k, v = _layer(g, d, w, x, pcs, pmask, (pk, pv))
        kvs.append((k, v))
    last = g.op("Slice", [x, g.i64(-1), g.i64(_INT64_MAX), g.i64(0)])  # 入力テキストの最終行 [1,H]

    y = g.op("Gather", [embed, g.const(trie.token, np.int64)], axis=0)  # [N,H]
    for w, kv in zip(layer_w, kvs, strict=True):
        y, _, _ = _layer(g, d, w, y, bcs, bmask, kv, additive_mask=True)

    # lm_head は root（入力テキストの最終行）と子を持つノードの行だけ。行 r の次 token の
    # 対数確率を slots[r] で拾い、各選択肢の対数尤度を経路の和（path との積）にする
    rows = g.op("Gather", [y, g.const(trie.rows, np.int64)], axis=0)
    h = g.op("Concat", [last, rows], axis=0)
    h = _rms(g, d, h, g.weight("norm", _f32(inner.norm.weight)))
    if c.tie_word_embeddings:
        lm = embed
    else:
        lm = g.weight("lm_head", _f32(model.lm_head.weight))
    logp = g.op("LogSoftmax", [g.op("Gemm", [h, lm], transB=1)], axis=-1)  # [P,V]
    # 最後の軸（ONNX 1.23 の ReferenceEvaluator は axis=-1 を受け付けないため正の値）
    picked = g.op("GatherElements", [logp, g.const(trie.slots, np.int64)], axis=1)  # [P,C]
    ll = g.op("MatMul", [g.const(trie.path), g.op("Reshape", [picked, g.i64(-1, 1)])])  # [K,1]
    g.op("Softmax", [g.op("Reshape", [ll, g.i64(1, -1)])], out="probs", axis=-1)

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
