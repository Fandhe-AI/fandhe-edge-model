//! `qwen2_lora` の採点グラフの実行器（REQ-41・REQ-32・REQ-28・REQ-39・TASK-41.6-5・#434）。
//!
//! # 役割と呼び出し文脈
//!
//! 学習ワーカー（`qwen2_onnx.py::export_scoring_onnx`）が書き出した LoRA 統合済みの採点グラフ
//! （`ids: INT64 [1,T]` → `probs: FLOAT [1,K]`）を、#416〜#419 の演算カーネル
//! （[`super::ops`]）だけで実行する。[`super::OnnxBackend`] の `Inner::Scoring` から呼ばれる。
//! 前処理（chat template の id 列）は [`crate::preprocess::TokenizerPreprocessor`]。
//!
//! # 汎用 ONNX ランタイムではない
//!
//! 書き出し順（トポロジカル順）にノードを走査して演算名で既存カーネルへ振り分ける、**許可リスト付きの
//! 有界インタプリタ**。許可は `docs/design/qwen2-scoring-onnx-ops.md` §3 の 28 種 × domain（既定のみ）×
//! 属性（書き出し器が書く固定値のみ）で、外れたものは読み込み時に
//! [`OnnxLoadError::UnsupportedGraph`] で拒否する（fail-closed）。dtype と rank は実行時に
//! カーネルが検査し、不一致は [`BackendError::Failed`] になる。
//!
//! # 資源
//!
//! - bf16 の重みは `Cast(to=FLOAT)` で f32 に展開せず、`u16` のまま保持した遅延 f32 として
//!   Gemm の B と Gather の data にだけ渡す（カーネル内で参照行だけ展開。設計書 §2・§4）
//! - 中間値は最後の消費で解放する（1.5B 級の RSS を有界にする）
//! - ノード境界ごとに期限を確認する。要素数・rank の上限はカーネルが `checked_mul` で照合する
//! - 最後の Softmax（グラフ出力 `probs`）だけ f64（C1・C3 と同じ）

use super::ops::arith::{
    add, add_i64, bf16_to_f32, cast_bf16_to_f32, cast_i64_to_f32, gemm_trans_b, matmul, mul,
    reciprocal, sqrt, sub,
};
use super::ops::mask_rope::{cos, greater_i64, range_i64, sin, where_f32};
use super::ops::norm_act::{
    log_softmax_last_axis, reduce_mean_last_axis, reduce_sum, sigmoid, softmax_last_axis,
    softmax_last_axis_f64,
};
use super::ops::shape::{
    concat, expand, gather_axis0, gather_elements_last_axis, reshape, shape, slice, split,
    transpose, unsqueeze,
};
use super::ops::{OpError, Tensor};
use super::proto::{DT_FLOAT, Dim, GraphProto, NodeProto, TensorData};
use super::{OnnxLoadError, check_graph_io, check_n_classes};
use crate::pipeline::{BackendError, MAX_INFER_TOKEN_IDS};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 1 件の計算時間の上限（REQ-39。C1・C3 と同じく共通コアの `limits::INFER_TIME_LIMIT` が出所）。
const MAX_SCORING_DURATION: Duration = fandhe_edge_core::limits::INFER_TIME_LIMIT;
/// グラフ出力の名前（書き出し器が固定）。
const OUTPUT: &str = "probs";

/// 実行中の値。bf16 の重みは [`Value::RawBf16`]（Cast だけが受ける）→ [`Value::LazyF32`]
/// （Gemm の B・Gather の data が直接読み、その他は f32 に展開して使う）。
#[derive(Clone)]
enum Value {
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    I64(Tensor<i64>),
    Bool(Tensor<bool>),
    RawBf16(Arc<Tensor<u16>>),
    LazyF32(Arc<Tensor<u16>>),
}

/// 検証済みの 1 ノード。属性は読み込み時に固定値と照合済みで、ここには実行に要る分だけ残す。
enum Op {
    Add,
    Sub,
    Mul,
    Sqrt,
    Reciprocal,
    Sigmoid,
    Sin,
    Cos,
    Cast,
    MatMul,
    Gemm,
    Shape,
    Gather,
    GatherElements { axis: i64 },
    Reshape,
    Unsqueeze,
    Expand,
    Transpose { perm: Vec<i64> },
    Concat { axis: i64 },
    Slice,
    Split { axis: i64 },
    Range,
    Greater,
    Where,
    ReduceMean,
    ReduceSum,
    Softmax { f64_out: bool },
    LogSoftmax,
}

struct Step {
    op: Op,
    ins: Vec<usize>,
    outs: Vec<usize>,
}

/// 読み込み済みの採点グラフ。`&self` の純粋計算で、呼び出し間の状態を持たない（REQ-28）。
pub(super) struct ScoringModel {
    steps: Vec<Step>,
    /// 名前 → スロット番号の数（initializer・入力・中間値・出力）。
    n_slots: usize,
    base: Vec<Option<Arc<Value>>>,
    uses: Vec<usize>,
    ids_slot: usize,
    out_slot: usize,
    n_classes: usize,
}

fn bad<T>() -> Result<T, OnnxLoadError> {
    Err(OnnxLoadError::UnsupportedGraph)
}

/// 属性の名前集合が `allowed` の部分集合であること。
fn attrs_within(n: &NodeProto, allowed: &[&str]) -> Result<(), OnnxLoadError> {
    let dup = n
        .attrs
        .iter()
        .enumerate()
        .any(|(i, a)| n.attrs.iter().skip(i + 1).any(|b| b.name == a.name));
    if dup || !n.attrs.iter().all(|a| allowed.contains(&a.name.as_str())) {
        return bad();
    }
    Ok(())
}

/// INT 属性が省略または期待値であること。
fn int_default(n: &NodeProto, name: &str, want: i64) -> Result<(), OnnxLoadError> {
    match n.attr(name) {
        None => Ok(()),
        Some(a) if a.int() == Some(want) => Ok(()),
        Some(_) => bad(),
    }
}

/// FLOAT 属性が省略または 1.0 であること。
fn float_one(n: &NodeProto, name: &str) -> Result<(), OnnxLoadError> {
    match n.attr(name) {
        None => Ok(()),
        Some(a) if a.float() == Some(1.0) => Ok(()),
        Some(_) => bad(),
    }
}

fn req_int(n: &NodeProto, name: &str) -> Result<i64, OnnxLoadError> {
    n.attr(name)
        .and_then(|a| a.int())
        .ok_or(OnnxLoadError::UnsupportedGraph)
}

fn arity(
    n: &NodeProto,
    ins: std::ops::RangeInclusive<usize>,
    outs: usize,
) -> Result<(), OnnxLoadError> {
    if ins.contains(&n.inputs.len()) && n.outputs.len() == outs {
        Ok(())
    } else {
        bad()
    }
}

/// ノードを許可リストと照合して [`Op`] にする。
fn compile_node(n: &NodeProto) -> Result<Op, OnnxLoadError> {
    if !n.domain.is_empty() {
        return bad();
    }
    let (op, ins, outs, allowed): (Op, _, usize, &[&str]) = match n.op_type.as_str() {
        "Add" => (Op::Add, 2..=2, 1, &[]),
        "Sub" => (Op::Sub, 2..=2, 1, &[]),
        "Mul" => (Op::Mul, 2..=2, 1, &[]),
        "Sqrt" => (Op::Sqrt, 1..=1, 1, &[]),
        "Reciprocal" => (Op::Reciprocal, 1..=1, 1, &[]),
        "Sigmoid" => (Op::Sigmoid, 1..=1, 1, &[]),
        "Sin" => (Op::Sin, 1..=1, 1, &[]),
        "Cos" => (Op::Cos, 1..=1, 1, &[]),
        "MatMul" => (Op::MatMul, 2..=2, 1, &[]),
        "Shape" => (Op::Shape, 1..=1, 1, &[]),
        "Reshape" => (Op::Reshape, 2..=2, 1, &[]),
        "Unsqueeze" => (Op::Unsqueeze, 2..=2, 1, &[]),
        "Expand" => (Op::Expand, 2..=2, 1, &[]),
        "Slice" => (Op::Slice, 3..=5, 1, &[]),
        "Range" => (Op::Range, 3..=3, 1, &[]),
        "Greater" => (Op::Greater, 2..=2, 1, &[]),
        "Where" => (Op::Where, 3..=3, 1, &[]),
        "ReduceSum" => (Op::ReduceSum, 2..=2, 1, &["keepdims"]),
        "Cast" => (Op::Cast, 1..=1, 1, &["to"]),
        "Gemm" => (Op::Gemm, 2..=3, 1, &["transA", "transB", "alpha", "beta"]),
        "Gather" => (Op::Gather, 2..=2, 1, &["axis"]),
        "GatherElements" => (
            Op::GatherElements {
                axis: req_int(n, "axis")?,
            },
            2..=2,
            1,
            &["axis"],
        ),
        "Transpose" => {
            let perm = n
                .attr("perm")
                .and_then(|a| a.ints())
                .ok_or(OnnxLoadError::UnsupportedGraph)?;
            (
                Op::Transpose {
                    perm: perm.to_vec(),
                },
                1..=1,
                1,
                &["perm"],
            )
        }
        "Concat" => (
            Op::Concat {
                axis: req_int(n, "axis")?,
            },
            1..=16,
            1,
            &["axis"],
        ),
        "Split" => (
            Op::Split {
                axis: req_int(n, "axis")?,
            },
            2..=2,
            2,
            &["axis"],
        ),
        "ReduceMean" => (Op::ReduceMean, 1..=1, 1, &["axes", "keepdims"]),
        "Softmax" => (
            Op::Softmax {
                f64_out: n.outputs.first().is_some_and(|o| o == OUTPUT),
            },
            1..=1,
            1,
            &["axis"],
        ),
        "LogSoftmax" => (Op::LogSoftmax, 1..=1, 1, &["axis"]),
        _ => return bad(),
    };
    arity(n, ins, outs)?;
    attrs_within(n, allowed)?;
    match &op {
        Op::Cast => {
            if req_int(n, "to")? != DT_FLOAT {
                return bad();
            }
        }
        Op::Gemm => {
            if req_int(n, "transB")? != 1 {
                return bad();
            }
            int_default(n, "transA", 0)?;
            float_one(n, "alpha")?;
            float_one(n, "beta")?;
        }
        Op::Gather => int_default(n, "axis", 0)?,
        Op::ReduceMean => {
            if n.attr("axes").and_then(|a| a.ints()) != Some(&[-1][..]) {
                return bad();
            }
            if req_int(n, "keepdims")? != 1 {
                return bad();
            }
        }
        Op::ReduceSum if req_int(n, "keepdims")? != 0 => return bad(),
        Op::Softmax { .. } | Op::LogSoftmax if req_int(n, "axis")? != -1 => return bad(),
        _ => {}
    }
    Ok(op)
}

fn init_value(t: &super::proto::TensorProto) -> Result<Value, OnnxLoadError> {
    let dims: Vec<usize> = t
        .dims
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| OnnxLoadError::UnsupportedTensor))
        .collect::<Result<_, _>>()?;
    let te = |_: OpError| OnnxLoadError::UnsupportedTensor;
    Ok(match &t.data {
        TensorData::F32(v) if v.iter().all(|x| x.is_finite()) => {
            Value::F32(Tensor::new(dims, v.clone()).map_err(te)?)
        }
        TensorData::I64(v) => Value::I64(Tensor::new(dims, v.clone()).map_err(te)?),
        TensorData::Bf16(v) => Value::RawBf16(Arc::new(Tensor::new(dims, v.clone()).map_err(te)?)),
        _ => return Err(OnnxLoadError::UnsupportedTensor),
    })
}

impl ScoringModel {
    /// 外部データ解決済みのグラフを照合して実行器にする（`resolve_external` の後に呼ぶ）。
    pub(super) fn from_graph(graph: &GraphProto) -> Result<Self, OnnxLoadError> {
        let k = match graph.outputs.as_slice() {
            [o] if o.name == OUTPUT => match o.dims.as_slice() {
                [Dim::Param(_), Dim::Value(k)] => {
                    usize::try_from(*k).map_err(|_| OnnxLoadError::UnsupportedGraph)?
                }
                _ => return bad(),
            },
            _ => return bad(),
        };
        check_n_classes(k)?;
        check_graph_io(graph, k)?;

        let mut slots: HashMap<&str, usize> = HashMap::new();
        let mut base: Vec<Option<Arc<Value>>> = Vec::new();
        // initializer（名前の重複は拒否）
        for t in &graph.initializers {
            if slots.insert(t.name.as_str(), base.len()).is_some() {
                return bad();
            }
            base.push(Some(Arc::new(init_value(t)?)));
        }
        if slots.insert("ids", base.len()).is_some() {
            return bad();
        }
        let ids_slot = base.len();
        base.push(None);

        let mut steps = Vec::with_capacity(graph.nodes.len());
        for n in &graph.nodes {
            let op = compile_node(n)?;
            // 入力は定義済み（トポロジカル順）であること
            let ins = n
                .inputs
                .iter()
                .map(|i| {
                    slots
                        .get(i.as_str())
                        .copied()
                        .ok_or(OnnxLoadError::UnsupportedGraph)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut outs = Vec::new();
            for o in &n.outputs {
                if slots.insert(o.as_str(), base.len()).is_some() {
                    return bad();
                }
                outs.push(base.len());
                base.push(None);
            }
            steps.push(Step { op, ins, outs });
        }
        let out_slot = slots
            .get(OUTPUT)
            .copied()
            .ok_or(OnnxLoadError::UnsupportedGraph)?;
        // probs は Softmax が作る（f64 出力の指定を持つこと）
        let produced_by_softmax = steps.iter().any(|s| {
            matches!(s.op, Op::Softmax { f64_out: true }) && s.outs.first() == Some(&out_slot)
        });
        if !produced_by_softmax {
            return bad();
        }
        let mut uses = vec![0usize; base.len()];
        for s in &steps {
            for &i in &s.ins {
                if let Some(u) = uses.get_mut(i) {
                    *u += 1;
                }
            }
        }
        Ok(Self {
            steps,
            n_slots: base.len(),
            base,
            uses,
            ids_slot,
            out_slot,
            n_classes: k,
        })
    }

    pub(super) fn n_classes(&self) -> usize {
        self.n_classes
    }

    pub(super) fn scores(&self, ids: &[i64]) -> Result<Vec<f64>, BackendError> {
        self.run(ids, MAX_SCORING_DURATION)
    }

    pub(super) fn scores_limited(
        &self,
        ids: &[i64],
        limit: Duration,
    ) -> Result<Vec<f64>, BackendError> {
        self.run(ids, limit.min(MAX_SCORING_DURATION))
    }

    fn run(&self, ids: &[i64], limit: Duration) -> Result<Vec<f64>, BackendError> {
        if ids.is_empty() || ids.len() > MAX_INFER_TOKEN_IDS {
            return Err(BackendError::InvalidSequenceLength);
        }
        let started = Instant::now();
        // Instant が表せない巨大な期限は上限扱い（INFER_TIME_LIMIT より先は意味がない）
        let until = started + limit;
        let mut f = Frame {
            vals: self.base.clone(),
            left: self.uses.clone(),
            first_dyn: self.ids_slot,
            live: 0,
            held: 0,
        };
        debug_assert_eq!(f.vals.len(), self.n_slots);
        let t = Tensor::new(vec![1, ids.len()], ids.to_vec()).map_err(op_err)?;
        *f.vals.get_mut(self.ids_slot).ok_or(BackendError::Failed)? = Some(Arc::new(Value::I64(t)));
        for step in &self.steps {
            if started.elapsed() > limit {
                return Err(BackendError::TimeLimitExceeded);
            }
            let outs = exec(step, &mut f, until)?;
            if outs.len() != step.outs.len() {
                return Err(BackendError::Failed);
            }
            // 出力の保持量検査は入力がまだ生きている状態で行い、済んでから入力を手放す
            for (&slot, v) in step.outs.iter().zip(outs) {
                f.store(slot, v, slot == self.out_slot)?;
            }
            f.release();
        }
        // 最終演算が期限を跨いだ場合も超過として返す
        if started.elapsed() > limit {
            return Err(BackendError::TimeLimitExceeded);
        }
        let out = f.vals.get(self.out_slot).and_then(|v| v.as_deref());
        match out {
            Some(Value::F64(t)) if t.shape() == [1, self.n_classes] => Ok(t.data().to_vec()),
            _ => Err(BackendError::Failed),
        }
    }
}

fn op_err(e: OpError) -> BackendError {
    match e {
        OpError::IndexOutOfRange => BackendError::InvalidTokenId,
        OpError::DeadlineExceeded => BackendError::TimeLimitExceeded,
        _ => BackendError::Failed,
    }
}

fn op<T>(r: Result<T, OpError>) -> Result<T, BackendError> {
    r.map_err(op_err)
}

/// 1 回の実行の値表。最後の消費で値を手放す。
struct Frame {
    vals: Vec<Option<Arc<Value>>>,
    left: Vec<usize>,
    /// これ以降のスロットは実行中に作る値（initializer は共有で数えない）。
    first_dyn: usize,
    /// 実行中に保持している値の合計バイト数。
    live: usize,
    /// 実行中の演算が最後に消費した入力のバイト数。演算が終わるまで `live` に残す（入力の
    /// Arc は演算が終わるまで解放されない）。
    held: usize,
}

/// 同時に保持する中間値の合計バイト数の上限（REQ-39。guard の RSS 上限 2 GiB と同じ桁）。
const MAX_LIVE_BYTES: usize = 1 << 31;

fn value_bytes(v: &Value) -> usize {
    match v {
        Value::F32(t) => t.data().len() * 4,
        Value::F64(t) => t.data().len() * 8,
        Value::I64(t) => t.data().len() * 8,
        Value::Bool(t) => t.data().len(),
        // bf16 の重みは initializer の共有バッファで、実確保の中間値ではないので数えない
        Value::RawBf16(_) | Value::LazyF32(_) => 0,
    }
}

impl Frame {
    /// 値を表に置く。後の消費者がなく出力でもない値は保存せず、保持量が上限を超えるなら拒否する。
    fn store(&mut self, slot: usize, v: Value, is_output: bool) -> Result<(), BackendError> {
        let uses = self.left.get(slot).copied().ok_or(BackendError::Failed)?;
        if uses == 0 && !is_output {
            return Ok(());
        }
        let live = self.live.saturating_add(value_bytes(&v));
        if live > MAX_LIVE_BYTES {
            return Err(BackendError::Failed);
        }
        self.live = live;
        *self.vals.get_mut(slot).ok_or(BackendError::Failed)? = Some(Arc::new(v));
        Ok(())
    }

    fn take(&mut self, slot: usize) -> Result<Arc<Value>, BackendError> {
        let left = self.left.get_mut(slot).ok_or(BackendError::Failed)?;
        *left = left.saturating_sub(1);
        let last = *left == 0;
        let cell = self.vals.get_mut(slot).ok_or(BackendError::Failed)?;
        let v = if last { cell.take() } else { cell.clone() };
        if last
            && slot >= self.first_dyn
            && let Some(x) = &v
        {
            self.held = self.held.saturating_add(value_bytes(x));
        }
        v.ok_or(BackendError::Failed)
    }

    /// 演算が終わった入力を `live` から手放す。
    fn release(&mut self) {
        self.live = self.live.saturating_sub(self.held);
        self.held = 0;
    }
}

fn own(a: Arc<Value>) -> Value {
    Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())
}

fn f32v(v: &Value) -> Result<Cow<'_, Tensor<f32>>, BackendError> {
    match v {
        Value::F32(t) => Ok(Cow::Borrowed(t)),
        Value::LazyF32(t) => Ok(Cow::Owned(op(cast_bf16_to_f32(t))?)),
        _ => Err(BackendError::Failed),
    }
}

fn i64v(v: &Value) -> Result<&Tensor<i64>, BackendError> {
    match v {
        Value::I64(t) => Ok(t),
        _ => Err(BackendError::Failed),
    }
}

/// 要素型に依らない形状系の演算を、F32・I64・BOOL の同じ型で返す。
macro_rules! same_type {
    ($v:expr, $t:ident => $e:expr) => {
        match $v {
            Value::F32($t) => Value::F32(op($e)?),
            Value::I64($t) => Value::I64(op($e)?),
            Value::Bool($t) => Value::Bool(op($e)?),
            _ => return Err(BackendError::Failed),
        }
    };
}

fn shape_of(v: &Value) -> &[usize] {
    match v {
        Value::F32(t) => t.shape(),
        Value::F64(t) => t.shape(),
        Value::I64(t) => t.shape(),
        Value::Bool(t) => t.shape(),
        Value::RawBf16(t) | Value::LazyF32(t) => t.shape(),
    }
}

fn numel(v: &Value) -> usize {
    shape_of(v).iter().fold(1usize, |a, &d| a.saturating_mul(d))
}

/// 値 1 要素の実バイト数（確保前検査の dtype 別見積もり用）。
fn elem_bytes(v: &Value) -> usize {
    match v {
        Value::F32(_) | Value::LazyF32(_) | Value::RawBf16(_) => 4,
        Value::F64(_) | Value::I64(_) => 8,
        Value::Bool(_) => 1,
    }
}

/// 出力バイト数の上界（出力 dtype の実サイズ × 出力形状から求めた要素数）。確保前の保持量検査に
/// 使う（REQ-39）。縮む演算は出力形状から、MatMul は broadcast 後の batch で数える。
/// 上界を作れない入力は `None`。
fn out_bytes_bound(op: &Op, a: &[Arc<Value>]) -> Option<usize> {
    let v = |i: usize| a.get(i).map(|x| x.as_ref());
    let bcast = |dims: &[&[usize]]| {
        let rank = dims.iter().map(|d| d.len()).max().unwrap_or(0);
        (0..rank).fold(1usize, |acc, k| {
            let m = dims
                .iter()
                .filter_map(|d| (d.len() + k).checked_sub(rank).and_then(|j| d.get(j)))
                .copied()
                .max()
                .unwrap_or(1);
            acc.saturating_mul(m)
        })
    };
    let prod = |d: &[usize]| d.iter().fold(1usize, |a, &d| a.saturating_mul(d));
    let (n, size) = match op {
        Op::Gemm => {
            let (x, w) = (shape_of(v(0)?), shape_of(v(1)?));
            let rows = prod(x.get(..x.len().saturating_sub(1))?);
            (rows.saturating_mul(*w.first()?), 4)
        }
        Op::MatMul => {
            let (x, w) = (shape_of(v(0)?), shape_of(v(1)?));
            let (xb, wb) = (
                x.get(..x.len().saturating_sub(2))?,
                w.get(..w.len().saturating_sub(2))?,
            );
            let m = if x.len() >= 2 {
                x.get(x.len() - 2).copied()?
            } else {
                1
            };
            let n = if w.len() >= 2 { *w.last()? } else { 1 };
            (bcast(&[xb, wb]).saturating_mul(m).saturating_mul(n), 4)
        }
        Op::Gather => {
            let (d, i) = (v(0)?, v(1)?);
            let row = numel(d) / shape_of(d).first().copied().filter(|&n| n > 0).unwrap_or(1);
            (numel(i).saturating_mul(row), elem_bytes(d))
        }
        Op::Expand => {
            let dims: Vec<usize> = i64v(v(1)?)
                .ok()?
                .data()
                .iter()
                .map(|&d| usize::try_from(d).ok())
                .collect::<Option<_>>()?;
            (bcast(&[shape_of(v(0)?), &dims]), elem_bytes(v(0)?))
        }
        Op::Range => {
            let g = |i: usize| i64v(v(i)?).ok()?.data().first().copied();
            let (s, l, d) = (i128::from(g(0)?), i128::from(g(1)?), i128::from(g(2)?));
            if d == 0 {
                return None;
            }
            let diff = l - s;
            let n = if (diff > 0) == (d > 0) && diff != 0 {
                (diff.abs() + d.abs() - 1) / d.abs()
            } else {
                0
            };
            (usize::try_from(n).unwrap_or(usize::MAX), 8)
        }
        Op::Shape => (shape_of(v(0)?).len(), 8),
        Op::GatherElements { .. } => (numel(v(1)?), 4),
        Op::Cast => (numel(v(0)?), 4),
        Op::Softmax { f64_out: true } => (numel(v(0)?), 8),
        // 出力が入力 0 以下の大きさで同じ dtype の演算
        Op::ReduceMean
        | Op::ReduceSum
        | Op::Softmax { .. }
        | Op::LogSoftmax
        | Op::Slice
        | Op::Transpose { .. }
        | Op::Split { .. } => (numel(v(0)?), elem_bytes(v(0)?)),
        Op::Concat { .. } => (a.iter().fold(0usize, |s, x| s.saturating_add(numel(x))), 4),
        Op::Greater => {
            let shapes: Vec<&[usize]> = a.iter().map(|x| shape_of(x)).collect();
            (bcast(&shapes), 1)
        }
        _ => {
            // 要素ごとの演算（Add・Sub・Mul・Where 等）は broadcast 後の形状
            let shapes: Vec<&[usize]> = a.iter().map(|x| shape_of(x)).collect();
            let size = a.iter().map(|x| elem_bytes(x)).max().unwrap_or(4);
            (bcast(&shapes), size)
        }
    };
    Some(n.saturating_mul(size))
}

fn exec(step: &Step, f: &mut Frame, until: Instant) -> Result<Vec<Value>, BackendError> {
    let mut a: Vec<Arc<Value>> = Vec::with_capacity(step.ins.len());
    for &i in &step.ins {
        a.push(f.take(i)?);
    }
    // 所有権を要る形状演算は、入力を唯一の持ち主にしてからデータをコピーせず渡す
    if matches!(step.op, Op::Reshape | Op::Unsqueeze) {
        let mut it = a.into_iter();
        let (x, s) = (it.next(), it.next());
        // データは移動するだけで二重にならないので、入力は先に手放す
        f.release();
        let x = x.ok_or(BackendError::Failed)?;
        // 他の消費者と共有中だと own() が全体を複製する。確保前に保持量へ合算して検査する（REQ-39）
        if Arc::strong_count(&x) > 1 && f.live.saturating_add(value_bytes(&x)) > MAX_LIVE_BYTES {
            return Err(BackendError::Failed);
        }
        let (x, s) = (own(x), s.ok_or(BackendError::Failed)?);
        let s = i64v(&s)?;
        return Ok(vec![if matches!(step.op, Op::Reshape) {
            same_type!(x, t => reshape(t, s))
        } else {
            same_type!(x, t => unsqueeze(t, s))
        }]);
    }
    // f32 へ展開する遅延重みの一時領域を、確保前に保持量の上限へ予約して検査する（REQ-39）。
    // Gemm の重み・Gather の表・Shape・Cast は展開せずに読むので数えない。
    let expanded = a
        .iter()
        .enumerate()
        .filter(|(i, _)| match step.op {
            Op::Gemm => *i != 1,
            Op::Gather => *i != 0,
            Op::Shape | Op::Cast => false,
            _ => true,
        })
        .fold(0usize, |acc, (_, v)| match v.as_ref() {
            Value::LazyF32(t) => acc.saturating_add(t.data().len().saturating_mul(4)),
            _ => acc,
        });
    let out = out_bytes_bound(&step.op, &a).ok_or(BackendError::Failed)?;
    if f.live.saturating_add(expanded).saturating_add(out) > MAX_LIVE_BYTES {
        return Err(BackendError::Failed);
    }
    let arg = |i: usize| -> Result<&Value, BackendError> {
        a.get(i).map(|v| v.as_ref()).ok_or(BackendError::Failed)
    };
    let one = |v: Value| Ok(vec![v]);
    match &step.op {
        Op::Add => match (arg(0)?, arg(1)?) {
            (Value::I64(x), Value::I64(y)) => one(Value::I64(op(add_i64(x, y))?)),
            (x, y) => one(Value::F32(op(add(f32v(x)?.as_ref(), f32v(y)?.as_ref()))?)),
        },
        Op::Sub => one(Value::F32(op(sub(
            f32v(arg(0)?)?.as_ref(),
            f32v(arg(1)?)?.as_ref(),
        ))?)),
        Op::Mul => one(Value::F32(op(mul(
            f32v(arg(0)?)?.as_ref(),
            f32v(arg(1)?)?.as_ref(),
        ))?)),
        Op::Sqrt => one(Value::F32(op(sqrt(f32v(arg(0)?)?.as_ref()))?)),
        Op::Reciprocal => one(Value::F32(op(reciprocal(f32v(arg(0)?)?.as_ref()))?)),
        Op::Sigmoid => one(Value::F32(op(sigmoid(f32v(arg(0)?)?.as_ref()))?)),
        Op::Sin => one(Value::F32(op(sin(f32v(arg(0)?)?.as_ref()))?)),
        Op::Cos => one(Value::F32(op(cos(f32v(arg(0)?)?.as_ref()))?)),
        Op::Cast => match arg(0)? {
            Value::RawBf16(t) => one(Value::LazyF32(t.clone())),
            Value::I64(t) => one(Value::F32(op(cast_i64_to_f32(t))?)),
            _ => Err(BackendError::Failed),
        },
        Op::MatMul => one(Value::F32(op(matmul(
            f32v(arg(0)?)?.as_ref(),
            f32v(arg(1)?)?.as_ref(),
            Some(until),
        ))?)),
        Op::Gemm => {
            let x = f32v(arg(0)?)?;
            let c = match a.get(2) {
                Some(c) => Some(f32v(c)?),
                None => None,
            };
            let out = match arg(1)? {
                Value::LazyF32(w) => gemm_trans_b(&x, w, c.as_deref(), bf16_to_f32, Some(until)),
                Value::F32(w) => gemm_trans_b(&x, w, c.as_deref(), |v| v, Some(until)),
                _ => return Err(BackendError::Failed),
            };
            one(Value::F32(op(out)?))
        }
        Op::Shape => {
            let s = match arg(0)? {
                Value::F32(t) => shape(t),
                Value::I64(t) => shape(t),
                Value::Bool(t) => shape(t),
                Value::LazyF32(t) => shape(t.as_ref()),
                _ => return Err(BackendError::Failed),
            };
            one(Value::I64(op(s)?))
        }
        Op::Gather => {
            let idx = i64v(arg(1)?)?;
            match arg(0)? {
                Value::I64(d) => one(Value::I64(op(gather_axis0(d, idx, |v| v))?)),
                Value::F32(d) => one(Value::F32(op(gather_axis0(d, idx, |v| v))?)),
                Value::LazyF32(d) => {
                    one(Value::F32(op(gather_axis0(d.as_ref(), idx, bf16_to_f32))?))
                }
                _ => Err(BackendError::Failed),
            }
        }
        Op::GatherElements { axis } => {
            let d = f32v(arg(0)?)?;
            let last = i64::try_from(d.shape().len()).map_err(|_| BackendError::Failed)? - 1;
            if *axis != -1 && *axis != last {
                return Err(BackendError::Failed);
            }
            one(Value::F32(op(gather_elements_last_axis(
                &d,
                i64v(arg(1)?)?,
            ))?))
        }
        Op::Reshape | Op::Unsqueeze => Err(BackendError::Failed),
        Op::Expand => {
            let s = i64v(arg(1)?)?;
            one(same_type!(arg(0)?, t => expand(t, s)))
        }
        Op::Transpose { perm } => one(same_type!(arg(0)?, t => transpose(t, perm))),
        Op::Concat { axis } => {
            let mut ts: Vec<Cow<'_, Tensor<f32>>> = Vec::with_capacity(a.len());
            for v in &a {
                ts.push(f32v(v)?);
            }
            let refs: Vec<&Tensor<f32>> = ts.iter().map(|c| c.as_ref()).collect();
            one(Value::F32(op(concat(&refs, *axis))?))
        }
        Op::Slice => {
            let starts = i64v(arg(1)?)?;
            let ends = i64v(arg(2)?)?;
            let axes = a.get(3).map(|v| i64v(v)).transpose()?;
            let steps = a.get(4).map(|v| i64v(v)).transpose()?;
            one(same_type!(arg(0)?, t => slice(t, starts, ends, axes, steps)))
        }
        Op::Split { axis } => {
            let s = i64v(arg(1)?)?;
            let parts = op(split(f32v(arg(0)?)?.as_ref(), *axis, s))?;
            Ok(parts.into_iter().map(Value::F32).collect())
        }
        Op::Range => one(Value::I64(op(range_i64(
            i64v(arg(0)?)?,
            i64v(arg(1)?)?,
            i64v(arg(2)?)?,
        ))?)),
        Op::Greater => one(Value::Bool(op(greater_i64(
            i64v(arg(0)?)?,
            i64v(arg(1)?)?,
        ))?)),
        Op::Where => match arg(0)? {
            Value::Bool(c) => one(Value::F32(op(where_f32(
                c,
                f32v(arg(1)?)?.as_ref(),
                f32v(arg(2)?)?.as_ref(),
            ))?)),
            _ => Err(BackendError::Failed),
        },
        Op::ReduceMean => one(Value::F32(op(reduce_mean_last_axis(
            f32v(arg(0)?)?.as_ref(),
        ))?)),
        Op::ReduceSum => one(Value::F32(op(reduce_sum(
            f32v(arg(0)?)?.as_ref(),
            i64v(arg(1)?)?,
        ))?)),
        Op::Softmax { f64_out: true } => one(Value::F64(op(softmax_last_axis_f64(
            f32v(arg(0)?)?.as_ref(),
        ))?)),
        Op::Softmax { f64_out: false } => {
            one(Value::F32(op(softmax_last_axis(f32v(arg(0)?)?.as_ref()))?))
        }
        Op::LogSoftmax => one(Value::F32(op(log_softmax_last_axis(
            f32v(arg(0)?)?.as_ref(),
        ))?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Frame {
        Frame {
            vals: vec![None; 3],
            left: vec![0, 1, 1],
            first_dyn: 0,
            live: 0,
            held: 0,
        }
    }

    fn f32s(n: usize) -> Value {
        Value::F32(Tensor::new(vec![n], vec![0.0; n]).unwrap())
    }

    #[test]
    fn req39_unused_intermediate_is_not_kept() {
        let mut f = frame();
        f.store(0, f32s(4), false).unwrap();
        assert!(f.vals[0].is_none());
        assert_eq!(f.live, 0);
        f.store(0, f32s(4), true).unwrap();
        assert!(f.vals[0].is_some());
    }

    #[test]
    fn req39_live_bytes_are_capped_and_released() {
        let mut f = frame();
        f.store(1, f32s(8), false).unwrap();
        assert_eq!(f.live, 32);
        f.live = MAX_LIVE_BYTES;
        assert_eq!(f.store(2, f32s(1), false), Err(BackendError::Failed));
        f.live = 32;
        f.take(1).unwrap();
        assert_eq!(f.live, 32);
        f.release();
        assert_eq!(f.live, 0);
        // 共有 initializer の bf16 は数えない
        let w = Value::LazyF32(Arc::new(Tensor::new(vec![2], vec![0u16; 2]).unwrap()));
        assert_eq!(value_bytes(&w), 0);
    }

    fn i64s(shape: Vec<usize>, v: Vec<i64>) -> Arc<Value> {
        Arc::new(Value::I64(Tensor::new(shape, v).unwrap()))
    }

    #[test]
    fn req39_bound_range_min_does_not_panic() {
        let a = [
            i64s(vec![], vec![0]),
            i64s(vec![], vec![i64::MIN]),
            i64s(vec![], vec![-1]),
        ];
        assert!(out_bytes_bound(&Op::Range, &a).unwrap() > MAX_LIVE_BYTES);
    }

    #[test]
    fn req39_bound_uses_i64_size_and_broadcast_batch() {
        let e = [i64s(vec![], vec![1]), i64s(vec![1], vec![1 << 28])];
        assert_eq!(out_bytes_bound(&Op::Expand, &e), Some((1usize << 28) * 8));
        let x = Arc::new(f32s_shape(vec![4, 2, 3, 5]));
        let w = Arc::new(f32s_shape(vec![4, 2, 5, 7]));
        assert_eq!(
            out_bytes_bound(&Op::MatMul, &[x, w]),
            Some(4 * 2 * 3 * 7 * 4)
        );
    }

    fn f32s_shape(shape: Vec<usize>) -> Value {
        let n = shape.iter().product();
        Value::F32(Tensor::new(shape, vec![0.0; n]).unwrap())
    }
}
