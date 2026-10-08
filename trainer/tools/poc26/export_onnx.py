"""PoC-26 候補 P（Qwen2.5-0.5B + LoRA）の ONNX 書き出し可否の確認（REQ-41・TASK-41.1-7・#392）。

役割: 学習済み adapter を base 重みへ統合（float32）し、opset 13・float32 の ONNX を手組みで
書き出して `onnx.checker` で検査し、演算の集計と推論ランタイム（`crates/runtime/src/onnx/`）の
対応・上限との突き合わせを JSON で返す（`export-onnx`）。書き出した ONNX を
`onnx.reference.ReferenceEvaluator` で動かし、LoRA 付き MLX モデルの logits と比べる
（`verify-onnx`）。`cli.py` から配線される。
成功基準 3（PoC-26 事前登録）の記録用で、推論ランタイム・配布パッケージには入らない（REQ-32）。
通信しない（REQ-38）。データ本文は読まない（検証プロンプトは固定の英文）。MLX は CPU で動かす。

重みは 2 GB を超えるため external data（`<name>.data` 1 ファイル）に置く。protobuf の 2 GB 上限と
重みの二重保持を避けるため、データファイルは自前で逐次書き、`TensorProto` には位置と長さだけを
持たせる。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import onnx
from mlx.utils import tree_flatten
from onnx import TensorProto, helper, numpy_helper
from tools.poc26.assets import (
    Pins,
    load_assets,
    load_base_model,
    pins_from_args,
)
from tools.poc26.common import SHA_RE, Budget, invalid, loads, read_input, sha256, too_large
from tools.poc26.io_records import MAX_ADAPTER_BYTES, check_out_dir, json_text, load_adapter
from tools.poc26.predict import attach_adapter
from tools.poc26.qwen2_model import LoRALinear, Qwen2Config, Qwen2Model
from tools.poc26.safe_io import LimitExceededError, open_regular

from fandhe_edge_trainer import limits

OPSET = 13
IR_VERSION = (
    8  # 推論ランタイムが固定する値（crates/runtime/src/onnx/mod.rs の EXPECTED_IR_VERSION）
)
EXTERNAL_THRESHOLD = 1024  # これ以上のバイト数の initializer は external data に置く
NEG_INF = (
    -1e30
)  # 因果マスクで遮蔽した位置のスコア（softmax で 0。対角は常に許可されるため NaN にならない）

#: 推論ランタイムが受理する op_type（出典: crates/runtime/src/onnx/c1.rs:53-88 の C1 テンプレートと
#: c3.rs:111-201 の C3 テンプレートの和集合）。ランタイムは固定テンプレート照合のみで、
#: 任意グラフを実行する汎用実装ではない。
RUNTIME_SUPPORTED_OPS = frozenset(
    {
        "Add", "Cast", "Concat", "Conv", "Div", "Gather", "Gemm", "Greater", "Log", "Max", "Mul",
        "ReduceMax", "ReduceSum", "Relu", "Softmax", "Sqrt", "Sub", "TfIdfVectorizer",
        "Transpose", "Unsqueeze",
    }
)  # fmt: skip

#: 推論ランタイムの上限（出典: crates/runtime/src/onnx/mod.rs:66,79 と proto.rs:23,25,35）。
RUNTIME_LIMITS = {
    "model_file_bytes": 44 * 1024 * 1024,
    "node_count": 128,
    "initializer_count": 64,
    "max_dims": 4,
}
RUNTIME_DTYPES = ("FLOAT", "INT64")  # 受理する dtype（proto.rs の DT_FLOAT・DT_INT64）
RUNTIME_OPSET = 13

_PINS_FROM_CFG = (
    "base_model_sha256",
    "config_sha256",
    "tokenizer_sha256",
    "tokenizer_config_sha256",
)

#: verify 用の固定プロンプト（データ本文ではない合成の英文。数十トークン以内）。
VERIFY_SYSTEM = "You are a tool selector."
VERIFY_USERS = ("Open the page and take a screenshot.", "Click the login button, then wait.")


# --- LoRA の統合 -------------------------------------------------------------------------


def fused_weights(model: Qwen2Model) -> dict[str, np.ndarray]:
    """LoRA を base へ統合した重みを HF のキー名・float32・Linear は (out, in) で返す。

    mlx の Linear 重みは (out, in)、lora_a は (in, r)、lora_b は (r, out) なので
    `W' = W + scale * (A @ B).T`（`LoRALinear.__call__` の `y + scale * (x @ A @ B)` と等価）。
    """
    out: dict[str, np.ndarray] = {}
    c = model.config
    inner = model.model
    out["model.embed_tokens.weight"] = np.array(inner.embed_tokens.weight, dtype=np.float32)
    out["model.norm.weight"] = np.array(inner.norm.weight, dtype=np.float32)
    if not c.tie_word_embeddings:
        out["lm_head.weight"] = np.array(model.lm_head.weight, dtype=np.float32)
    for i, block in enumerate(inner.layers):
        p = f"model.layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            out[p + norm + ".weight"] = np.array(getattr(block, norm).weight, dtype=np.float32)
        for owner, names in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        ):
            for name in names:
                lin = getattr(getattr(block, owner), name)
                w = lin.linear.weight if isinstance(lin, LoRALinear) else lin.weight
                w = np.array(w, dtype=np.float32)
                if isinstance(lin, LoRALinear):
                    delta = lin.scale * (np.array(lin.lora_a) @ np.array(lin.lora_b)).T
                    w = w + delta.astype(np.float32)
                key = f"{p}{owner}.{name}."
                out[key + "weight"] = w
                base = lin.linear if isinstance(lin, LoRALinear) else lin
                if "bias" in base:
                    out[key + "bias"] = np.array(base.bias, dtype=np.float32)
    return out


# --- ONNX の手組み -----------------------------------------------------------------------


class _Graph:
    """ノードと initializer を溜めるビルダー。大きな重みは external data へ逐次書く。"""

    def __init__(self, data_file: Any, data_name: str, budget: Budget | None = None) -> None:
        self._budget = budget  # 重み 1 本ごとに RSS・壁時計を確認する（REQ-39）
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._f, self._name, self._offset, self._n = data_file, data_name, 0, 0
        self._consts: dict[tuple, str] = {}

    def _fresh(self, base: str) -> str:
        self._n += 1
        return f"{base}_{self._n}"

    def const(self, value: Any, dtype: Any = np.float32) -> str:
        """小さな定数（inline initializer）。"""
        arr = np.asarray(value, dtype=dtype)
        key = (arr.dtype.str, arr.shape, arr.tobytes())
        if key not in self._consts:  # 同じ定数は 1 つの initializer を共有する
            name = self._fresh("c")
            self.inits.append(numpy_helper.from_array(arr, name))
            self._consts[key] = name
        return self._consts[key]

    def weight(self, name: str, arr: np.ndarray) -> str:
        """重み。`EXTERNAL_THRESHOLD` 以上なら external data へ書き、位置と長さだけを持つ。"""
        if self._budget is not None:
            self._budget.check()
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.nbytes < EXTERNAL_THRESHOLD:
            self.inits.append(numpy_helper.from_array(arr, name))
            return name
        pad = -self._offset % 64  # 64 byte 境界へ揃える
        self._f.write(b"\0" * pad)
        self._offset += pad
        t = TensorProto()
        t.name, t.data_type = name, TensorProto.FLOAT
        t.dims.extend(arr.shape)
        t.data_location = TensorProto.EXTERNAL
        # `set_external_data` は raw_data を要求するため、位置情報を直接持たせる
        for k, v in (("location", self._name), ("offset", self._offset), ("length", arr.nbytes)):
            e = t.external_data.add()
            e.key, e.value = k, str(v)
        self._f.write(memoryview(arr).cast("B"))
        self._offset += arr.nbytes
        self.inits.append(t)
        return name

    def op(self, op_type: str, inputs: list[str], out: str | None = None, **attrs: Any) -> str:
        out = out or self._fresh(op_type.lower())
        self.nodes.append(
            helper.make_node(op_type, inputs, [out], name=self._fresh("n_" + op_type), **attrs)
        )
        return out

    def split2(self, x: str, half: int) -> tuple[str, str]:
        a, b = self._fresh("sp"), self._fresh("sp")
        self.nodes.append(
            helper.make_node(
                "Split",
                [x, self.const([half, half], np.int64)],
                [a, b],
                name=self._fresh("n_Split"),
                axis=-1,
            )
        )
        return a, b


def _i64(g: _Graph, *vals: int) -> str:
    return g.const(list(vals), np.int64)


def build_onnx(c: Qwen2Config, w: dict[str, np.ndarray], g: _Graph) -> onnx.ModelProto:
    """Qwen2 の forward（RMSNorm・RoPE 非 traditional・GQA・SwiGLU・tie）を opset 13 の演算で組む。

    入力 `input_ids: INT64[1, T]`（動的 T）、出力 `logits: FLOAT[1, T, V]`。因果マスクは
    Range・Greater・Where で作る（Trilu は opset 14 のため使わない）。
    """
    nh, nkv, hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
    grp, half = nh // nkv, c.head_dim // 2
    ids = "input_ids"

    t_len = g.op("Gather", [g.op("Shape", [ids]), g.const(1, np.int64)])  # 0 次元の T
    pos = g.op("Range", [g.const(0, np.int64), t_len, g.const(1, np.int64)])
    inv_freq = (c.rope_theta ** (-np.arange(half, dtype=np.float64) / half)).astype(np.float32)
    posf = g.op("Unsqueeze", [g.op("Cast", [pos], to=TensorProto.FLOAT), _i64(g, 1)])
    ang = g.op("Mul", [posf, g.const(inv_freq[None, :])])  # [T, half]
    cos, sin = g.op("Cos", [ang]), g.op("Sin", [ang])
    future = g.op(
        "Greater", [g.op("Unsqueeze", [pos, _i64(g, 0)]), g.op("Unsqueeze", [pos, _i64(g, 1)])]
    )  # [T, T]。列 > 行 = 未来の位置
    neg, scale = g.const(NEG_INF), g.const(hd**-0.5)

    def rms(x: str, key: str) -> str:
        m = g.op("ReduceMean", [g.op("Mul", [x, x])], axes=[-1], keepdims=1)
        r = g.op("Reciprocal", [g.op("Sqrt", [g.op("Add", [m, g.const(c.rms_norm_eps)])])])
        return g.op("Mul", [g.op("Mul", [x, r]), g.weight(key, w[key])])

    def linear(x: str, key: str) -> str:
        y = g.op("MatMul", [x, g.weight(key + "T", w[key + ".weight"].T)])
        return (
            g.op("Add", [y, g.weight(key + ".bias", w[key + ".bias"])]) if key + ".bias" in w else y
        )

    def rope(x: str) -> str:
        x1, x2 = g.split2(x, half)
        lo = g.op("Sub", [g.op("Mul", [x1, cos]), g.op("Mul", [x2, sin])])
        hi = g.op("Add", [g.op("Mul", [x1, sin]), g.op("Mul", [x2, cos])])
        return g.op("Concat", [lo, hi], axis=-1)

    def heads(x: str, *shape_head: int) -> str:
        """[1,T,n*hd] -> [1,n,(grp),T,hd]（Reshape の 0 は入力の同位置の次元を写す）。"""
        r = g.op("Reshape", [x, _i64(g, 0, 0, *shape_head)])
        perm = [0, 2, 3, 1, 4] if len(shape_head) == 3 else [0, 2, 1, 3]
        return g.op("Transpose", [r], perm=perm)

    embed = g.weight("model.embed_tokens.weight", w["model.embed_tokens.weight"])
    h = g.op("Gather", [embed, ids])
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        a = rms(h, p + "input_layernorm.weight")
        q = heads(linear(a, p + "self_attn.q_proj"), nkv, grp, hd)  # [1,nkv,grp,T,hd]
        k = g.op("Unsqueeze", [rope(heads(linear(a, p + "self_attn.k_proj"), nkv, hd)), _i64(g, 2)])
        v = g.op("Unsqueeze", [heads(linear(a, p + "self_attn.v_proj"), nkv, hd), _i64(g, 2)])
        s = g.op("MatMul", [rope(q), g.op("Transpose", [k], perm=[0, 1, 2, 4, 3])])
        s = g.op("Where", [future, neg, g.op("Mul", [s, scale])])
        ctx = g.op("MatMul", [g.op("Softmax", [s], axis=-1), v])  # [1,nkv,grp,T,hd]
        ctx = g.op("Reshape", [g.op("Transpose", [ctx], perm=[0, 3, 1, 2, 4]), _i64(g, 0, 0, -1)])
        h = g.op("Add", [h, linear(ctx, p + "self_attn.o_proj")])
        m = rms(h, p + "post_attention_layernorm.weight")
        gate = linear(m, p + "mlp.gate_proj")
        act = g.op("Mul", [gate, g.op("Sigmoid", [gate])])
        h = g.op(
            "Add",
            [h, linear(g.op("Mul", [act, linear(m, p + "mlp.up_proj")]), p + "mlp.down_proj")],
        )
    h = rms(h, "model.norm.weight")
    if c.tie_word_embeddings:
        head = g.op("Transpose", [embed], perm=[1, 0])
    else:
        head = g.weight("lm_head.weightT", w["lm_head.weight"].T)
    g.op("MatMul", [h, head], out="logits")

    graph = helper.make_graph(
        g.nodes,
        "qwen2_lora_fused",
        [helper.make_tensor_value_info(ids, TensorProto.INT64, [1, "T"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "T", c.vocab_size])],
        initializer=g.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model


# --- 集計 --------------------------------------------------------------------------------


def summarize(model_path: Path, data_path: Path) -> dict[str, Any]:
    """演算・initializer・サイズの集計と、推論ランタイムの対応・上限との突き合わせ。"""
    m = onnx.load(str(model_path), load_external_data=False)
    ops = Counter(n.op_type for n in m.graph.node)
    dtypes: Counter[str] = Counter(
        TensorProto.DataType.Name(t.data_type) for t in m.graph.initializer
    )
    itemsize = {"FLOAT": 4, "INT64": 8}
    init_bytes = sum(
        int(np.prod(t.dims, dtype=np.int64)) * itemsize[TensorProto.DataType.Name(t.data_type)]
        for t in m.graph.initializer
    )
    max_rank = max((len(t.dims) for t in m.graph.initializer), default=0)
    file_bytes = model_path.stat().st_size + data_path.stat().st_size
    unsupported = sorted(set(ops) - RUNTIME_SUPPORTED_OPS)
    actual = {
        "model_file_bytes": file_bytes,
        "node_count": len(m.graph.node),
        "initializer_count": len(m.graph.initializer),
        "max_dims": max_rank,
    }
    limits = {
        k: {"limit": v, "actual": actual[k], "exceeded": actual[k] > v}
        for k, v in RUNTIME_LIMITS.items()
    }
    opset = next(o.version for o in m.opset_import if o.domain == "")
    return {
        "opset": opset,
        "ir_version": m.ir_version,
        "dtype": "float32",
        "input": "input_ids: INT64[1,T]",
        "output": "logits: FLOAT[1,T,V]",
        "node_count": len(m.graph.node),
        "op_counts": dict(sorted(ops.items())),
        "initializer_count": len(m.graph.initializer),
        "initializer_dtypes": dict(dtypes),
        "initializer_total_bytes": init_bytes,
        "model_file_bytes": model_path.stat().st_size,
        "external_data_bytes": data_path.stat().st_size,
        "total_file_bytes": file_bytes,
        "runtime": {
            "supported_ops": sorted(RUNTIME_SUPPORTED_OPS),
            "unsupported_ops": unsupported,
            "opset_ok": opset == RUNTIME_OPSET,
            "dtypes_ok": all(d in RUNTIME_DTYPES for d in dtypes),
            "external_data_used": True,
            "external_data_accepted": False,  # 外部データは拒否（proto.rs）
            "limits": limits,
        },
    }


# --- サブコマンド ------------------------------------------------------------------------


def check_memory(estimate: int, what: str) -> None:
    """同時保持量の見積もりが RSS 上限（`limits.MAX_TRAIN_RSS_BYTES`）超なら 20 で止める。

    確保の前に呼ぶ（`Budget.check` はピーク実測で確保後にしか検知できないため。REQ-39）。
    """
    if estimate > limits.MAX_TRAIN_RSS_BYTES:
        raise too_large(f"{what} needs about {estimate} bytes, over the RSS budget")


def _load_fused_source(a: argparse.Namespace) -> tuple[Qwen2Model, dict[str, Any], Pins]:
    """base（float32）と adapter を読み、LoRA 付きのモデル（未統合）と adapter 設定を返す。"""
    mx.set_default_device(mx.cpu)
    pins = pins_from_args(a)
    adapter_bytes = read_input(
        a.adapter_dir / "adapters.safetensors", MAX_ADAPTER_BYTES, "adapters"
    )
    if sha256(adapter_bytes) != a.adapter_sha256:
        raise invalid("adapters.safetensors does not match the expected sha256")
    weights, cfg = load_adapter(adapter_bytes)
    del adapter_bytes
    if tuple(cfg[k] for k in _PINS_FROM_CFG) != (
        pins.model,
        pins.config,
        pins.tokenizer,
        pins.tokenizer_config,
    ):
        raise invalid("adapter was trained with a different base model / config / tokenizer")
    model, _sha, _size = load_base_model(a.model_dir, mx.float32, pins)
    attach_adapter(model, cfg, weights)
    model.eval()  # dropout > 0 の adapter でも logits を決定的にする（score.py・probe.py と同じ）
    return model, cfg, pins


#: 照合時に読む 2 ファイルのサイズ上限（実物: モデル本体 約 150 KB・外部データ 約 1.98 GB）。
MAX_ONNX_BYTES = 16 << 20
MAX_ONNX_DATA_BYTES = 4 << 30


def _sha256_file(path: Path, limit: int, what: str) -> str:
    """通常ファイルを開いてサイズ上限を確認し、全体を 1 MiB ずつ読んで sha256 を返す（REQ-39）。"""
    try:
        fd, _ = open_regular(path, limit, what)
    except LimitExceededError as exc:
        raise too_large(str(exc)) from None
    except ValueError as exc:
        raise invalid(str(exc)) from None
    h = hashlib.sha256()
    with os.fdopen(fd, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def cmd_export_onnx(a: argparse.Namespace) -> int:
    """`export-onnx`: LoRA 統合 -> ONNX 手組み -> `onnx.checker` -> 集計 JSON を出力先へ書く。

    出力先は 0700 で新規作成し、途中で失敗したら作ったディレクトリごと消す。`OutputDir`（一時
    ディレクトリ経由の rename）は 2 GB 超の外部データを二重に置くため使わない。
    """
    t0 = time.monotonic()
    budget = Budget()
    check_out_dir(a.out_dir)
    model, cfg, _ = _load_fused_source(a)
    budget.check()
    # MLX モデル保持中に全重みの numpy コピーを作る。最大の 1 本は転置で更にコピーされる
    sizes = [int(v.nbytes) for _, v in tree_flatten(model.parameters())]
    check_memory(2 * sum(sizes) + max(sizes, default=0), "export-onnx weights")
    weights = fused_weights(model)
    budget.check()
    a.out_dir.mkdir(mode=0o700)
    try:
        model_path, data_path = a.out_dir / "model.onnx", a.out_dir / "model.onnx.data"
        # O_EXCL 相当の新規作成（上書きしない）
        fd = os.open(data_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            proto = build_onnx(model.config, weights, _Graph(f, data_path.name, budget))
        onnx.save_model(proto, str(model_path))
        del weights, proto
        budget.check()
        onnx.checker.check_model(str(model_path))  # パス渡し（2 GB 超でも検査できる）
        budget.check()
        summary = {
            "evidence": a.evidence,
            "command": a.command,
            "onnx_version": onnx.__version__,
            "mlx_version": mx.__version__,
            "adapter_config": {
                k: cfg[k] for k in ("rank", "scale", "dropout", "num_layers", "lora_init_seed")
            },
            "checker": "passed",
            **summarize(model_path, data_path),
            # verify-onnx が読む前に照合する
            "model_onnx_sha256": _sha256_file(model_path, MAX_ONNX_BYTES, "model.onnx"),
            "model_onnx_data_sha256": _sha256_file(
                data_path, MAX_ONNX_DATA_BYTES, "model.onnx.data"
            ),
            "elapsed_seconds": round(time.monotonic() - t0, 1),
        }
        text = json_text(summary)
        (a.out_dir / "export_summary.json").write_text(text, encoding="utf-8")
    except BaseException:
        shutil.rmtree(a.out_dir, ignore_errors=True)
        raise
    print(text, end="")
    return 0


_SUMMARY_HASH_KEYS = ("model_onnx_sha256", "model_onnx_data_sha256")


def _summary_hashes(summary: Any) -> dict[str, str]:
    """`export_summary.json` のルート型と 2 つのハッシュ欄（64 桁小文字 hex）を検証（REQ-39）。"""
    if not isinstance(summary, dict):
        raise invalid("export_summary.json must be a JSON object")
    out: dict[str, str] = {}
    for key in _SUMMARY_HASH_KEYS:
        v = summary.get(key)
        if not isinstance(v, str) or not SHA_RE.fullmatch(v):
            raise invalid(f"export_summary.json: {key} must be a 64-char lowercase hex string")
        out[key] = v
    return out


def _external_tensors(proto: onnx.ModelProto, data_name: str) -> list[tuple[Any, int, int]]:
    """external data を持つ initializer を (tensor, offset, length) で offset 順に返す。"""
    found = []
    for t in proto.graph.initializer:
        if t.data_location != TensorProto.EXTERNAL:
            continue
        kv = {e.key: e.value for e in t.external_data}
        try:
            offset, length = int(kv["offset"]), int(kv["length"])
        except (KeyError, ValueError):
            raise invalid("initializer has malformed external data") from None
        expected = int(np.prod(t.dims, dtype=np.int64)) * 4
        if (
            kv.get("location") != data_name
            or t.data_type != TensorProto.FLOAT
            or offset < 0
            or length != expected
        ):
            raise invalid("initializer has unexpected external data")
        found.append((t, offset, length))
    found.sort(key=lambda x: x[1])
    return found


def load_verified_onnx(onnx_dir: Path, summary: Any, budget: Budget) -> onnx.ModelProto:
    """sha256 を照合した**同じバイト列**だけから、重みを inline に持つ ModelProto を作る。

    REQ-41・REQ-39: ファイルは fd で開いて読み（通常ファイル・symlink・サイズ検証は
    `open_regular` / `read_input`）、パスを開き直さない。照合後に差し替えられても、
    解析に使うのは照合したバイト列だけ。model.onnx.data は 1 回の逐次読みで sha256 を計算しながら
    各 tensor の raw_data へ詰める（全体を別に持たず、同時保持は proto 分＋1 本分）。
    """
    hashes = _summary_hashes(summary)
    raw = read_input(onnx_dir / "model.onnx", MAX_ONNX_BYTES, "model.onnx")
    if sha256(raw) != hashes["model_onnx_sha256"]:
        raise invalid("model.onnx does not match the sha256 recorded at export")
    try:
        proto = onnx.load_model_from_string(raw)
    except Exception:  # protobuf の DecodeError 等。入力値は載せない
        raise invalid("model.onnx is not a valid ONNX model") from None
    del raw
    tensors = _external_tensors(proto, "model.onnx.data")
    try:
        fd, st = open_regular(onnx_dir / "model.onnx.data", MAX_ONNX_DATA_BYTES, "model.onnx.data")
    except LimitExceededError as exc:
        raise too_large(str(exc)) from None
    except ValueError as exc:
        raise invalid(str(exc)) from None
    with os.fdopen(fd, "rb") as f:
        # 重み本体（raw_data）＋ evaluator 内の numpy 化の分を見込む
        check_memory(2 * st.st_size + (1 << 30), "verify-onnx weights")
        h, pos = hashlib.sha256(), 0

        def take(n: int, sink: bytearray | None) -> None:
            nonlocal pos
            got = 0
            while got < n:
                chunk = f.read(min(1 << 20, n - got))
                if not chunk:
                    raise invalid("model.onnx.data is truncated")
                h.update(chunk)
                if sink is not None:
                    sink[got : got + len(chunk)] = chunk
                got += len(chunk)
            pos += n

        for t, offset, length in tensors:
            if offset < pos:
                raise invalid("initializer external data overlaps")
            take(offset - pos, None)
            buf = bytearray(length)
            take(length, buf)
            t.raw_data = bytes(buf)
            del buf, t.external_data[:]
            t.data_location = TensorProto.DEFAULT
            budget.check()
        while chunk := f.read(1 << 20):  # 残り（末尾のパディング等）も照合対象
            h.update(chunk)
            pos += len(chunk)
            if pos > MAX_ONNX_DATA_BYTES:
                raise too_large("model.onnx.data too large")
    if h.hexdigest() != hashes["model_onnx_data_sha256"]:
        raise invalid("model.onnx.data does not match the sha256 recorded at export")
    return proto


def cmd_verify_onnx(a: argparse.Namespace) -> int:
    """`verify-onnx`: 書き出した ONNX（ReferenceEvaluator）と LoRA 付き MLX の logits を比べる。

    **合否は出さない**（事前登録に許容差が無いため。一致の判定は人が行い、数値だけを出す）。
    `export_summary.json` の検証と、2 ファイルの fd 読みでの sha256 照合・解析は
    `load_verified_onnx`（不一致は 64。パスの開き直しをしない）。資源は `Budget`（RSS・24 時間の
    天井）で各境界を確認し、ReferenceEvaluator の実行は `--max-score-seconds` の壁時計上限を
    1 件ごとの前後で確認する（実行中の割り込みはできない）。ピークメモリ（float32 の概算）:
    モデル読み込み時 約 4 GB、MLX の logits を取って解放した後の ONNX 段階は 約 2 GB（raw_data）＋
    evaluator 内の numpy 約 2 GB。両段階を同時には持たない。
    """
    from onnx.reference import ReferenceEvaluator

    t0 = time.monotonic()
    budget = Budget()
    summary = loads(
        read_input(a.onnx_dir / "export_summary.json", MAX_ONNX_BYTES, "export_summary.json"),
        "export_summary.json",
    )
    _summary_hashes(summary)  # 重い処理の前に形式だけ先に確認する
    model, cfg, pins = _load_fused_source(a)
    budget.check()
    assets = load_assets(a.model_dir, pins)
    ids_list = [
        assets.tok.build_chat_ids(VERIFY_SYSTEM, user, add_generation_prompt=True)
        for user in VERIFY_USERS
    ]
    wants = [np.array(model(mx.array([ids])), dtype=np.float32) for ids in ids_list]
    del model  # MLX の重みを解放してから ONNX を読む（ピークを重ねない）
    mx.clear_cache()
    budget.check()
    proto = load_verified_onnx(a.onnx_dir, summary, budget)
    ref = ReferenceEvaluator(proto)
    del proto
    budget.check()
    run_budget = Budget(wall_limit=a.max_score_seconds)
    results = []
    for ids, want in zip(ids_list, wants, strict=True):
        run_budget.check()
        got = ref.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0]
        run_budget.check()
        budget.check()
        diff = np.abs(want - got)
        am_w, am_g = want.argmax(-1), got.argmax(-1)
        results.append(
            {
                "tokens": len(ids),
                "max_abs_error": float(diff.max()),
                "max_abs_logit": float(np.abs(want).max()),
                "argmax_match_positions": int((am_w == am_g).sum()),
                "positions": len(ids),
                "last_position_argmax_match": bool(am_w[0, -1] == am_g[0, -1]),
            }
        )
    out = {
        "evidence": a.evidence,
        "reference": "onnx.reference.ReferenceEvaluator (numpy, float32)",
        "mlx": "LoRA-attached model, float32, CPU, eval mode",
        "adapter_dropout": cfg["dropout"],
        "note": "numbers only; the match decision is made by a human",
        "prompts": results,
        "elapsed_seconds": round(time.monotonic() - t0, 1),
    }
    print(json_text(out), end="")
    return 0
