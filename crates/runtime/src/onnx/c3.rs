//! C3（バイト CNN）の ONNX テンプレート照合と順伝播（REQ-32・REQ-28・REQ-39・TASK-32.1-2・#113）。
//!
//! # 役割
//!
//! [`super::OnnxBackend`] の C3 用の実行器。学習ワーカーの `kinds/c3.py::_export_c3_onnx` が書き出す
//! グラフ（`Gather` → `Transpose` → マスク → 幅ごとの `Conv`・`Relu`・マスク加算・`ReduceMax` →
//! `Concat` → `Gemm` → `Softmax`）とノード列・入出力名・属性・initializer を完全一致で照合し、
//! 一致したものだけを実行する（許可制）。
//!
//! # 計算とパディング（REQ-28）
//!
//! 1 系列（バッチ次元なし）について、埋め込み → 相互相関（左右に `k/2` のゼロ詰め。`k` は奇数で
//! 出力長が系列長と一致）＋バイアス → ReLU → 詰め物位置（id=0）への大きな負値の加算 → 時間方向の
//! 最大値 → 連結 → `Gemm` を f32 で計算し、最後だけ f64 で softmax する（[`super`] の「数値」）。
//! 詰め物の埋め込み行は厳密に 0（読み込み時に検査）で、詰め物位置は最大値から除かれるため、
//! 系列長をそろえるためのパディングが結果へ混入しない。バッチ推論は 1 系列ずつ本関数を呼ぶだけ
//! （[`crate::pipeline`]）。
//!
//! # 資源
//!
//! 系列長は [`super::MAX_MAX_BYTES`] 以下に制限する（前処理の `max_bytes` と同じ上限）。
//! 幅の個数・`k`・埋め込み次元・フィルタ数は学習ワーカーの上限
//! （`limits.py::MAX_C3_*`）と同値の定数で制限し、重みの総量はモデルファイルの大きさで有界。

use std::time::{Duration, Instant};

use super::proto::{GraphProto, NodeProto};
use super::{
    Inits, MAX_MAX_BYTES, N_TOKENS, OnnxLoadError, attr_f32_is, attr_int_is, attr_ints_are,
    check_graph_io, check_n_classes, check_node, scalar_is, shape_is, softmax_f64,
};
use crate::pipeline::BackendError;

/// 幅（畳み込み枝）の個数の上限（`limits.py::MAX_C3_WIDTHS`）。
const MAX_WIDTHS: usize = 8;
/// カーネル幅の上限（`limits.py::MAX_C3_WIDTH_VALUE`）。
const MAX_KERNEL: usize = 31;
/// 埋め込み次元の上限（`limits.py::MAX_C3_EMB`）。
const MAX_EMB: usize = 1024;
/// 1 枝のフィルタ数の上限（`limits.py::MAX_C3_FILTERS`）。
const MAX_FILTERS: usize = 1024;
/// 詰め物位置へ加える負値。書き出し器（`kinds/c3.py::_MASK_NEG_VALUE`）の固定値と一致するものだけを
/// 受理する。任意の負数（例: -0.001）を許すと、詰め物位置が実トークン位置より大きくなり
/// プーリングへ混入する（REQ-28）。
const MASK_NEG_VALUE: f32 = -1e9;
/// 1 系列の計算時間の上限（REQ-39。`latency::DEFAULT_LATENCY_PER_INFER_TIMEOUT_NS` と同値）。
/// 受理したモデルは最大 4096 トークン × 1024 次元 × 31 カーネル × フィルタ数の走査になりうるため、
/// モデルファイルの大きさの上限だけでは時間を抑えられない。
const MAX_INFER_DURATION: Duration = Duration::from_secs(10);
/// 幅に依存しない前段のノード数と、後段（`Concat`・`Gemm`・`Softmax`）のノード数。
const PREFIX_NODES: usize = 7;
const SUFFIX_NODES: usize = 3;
/// 1 枝あたりのノード数（`Conv`・`Relu`・`Add`・`ReduceMax`）。
const BRANCH_NODES: usize = 4;

/// 畳み込み 1 枝。
struct Branch {
    kernel: usize,
    filters: usize,
    /// 形状 `[filters, emb, kernel]` の行優先（ONNX の Conv の重みの並び）。
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// 照合済みの C3 モデル。
pub(super) struct C3Model {
    emb: usize,
    /// 形状 `[257, emb]` の行優先。0 行目（詰め物）は厳密に 0。
    embed: Vec<f32>,
    branches: Vec<Branch>,
    neg_big: f32,
    /// 形状 `[sum(filters), n_classes]` の行優先。
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    n_classes: usize,
}

fn bad() -> OnnxLoadError {
    OnnxLoadError::UnsupportedGraph
}

fn to_usize(v: i64) -> Result<usize, OnnxLoadError> {
    usize::try_from(v).map_err(|_| bad())
}

impl C3Model {
    pub(super) fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// グラフをテンプレートと照合し、パラメータを取り出す。
    pub(super) fn from_graph(graph: &GraphProto) -> Result<Self, OnnxLoadError> {
        let nodes = &graph.nodes;
        let body = nodes
            .len()
            .checked_sub(PREFIX_NODES + SUFFIX_NODES)
            .ok_or_else(bad)?;
        if body % BRANCH_NODES != 0 {
            return Err(bad());
        }
        let n_widths = body / BRANCH_NODES;
        if !(1..=MAX_WIDTHS).contains(&n_widths) {
            return Err(bad());
        }
        let node = |i: usize| -> Result<&NodeProto, OnnxLoadError> { nodes.get(i).ok_or_else(bad) };

        // 前段 7 ノード
        check_node(
            node(0)?,
            "Gather",
            &["embed", "ids"],
            &["emb_bth"],
            &["axis"],
        )?;
        attr_int_is(node(0)?, "axis", 0)?;
        check_node(node(1)?, "Transpose", &["emb_bth"], &["emb_bct"], &["perm"])?;
        attr_ints_are(node(1)?, "perm", &[0, 2, 1])?;
        check_node(
            node(2)?,
            "Greater",
            &["ids", "zero_i64"],
            &["mask_bool"],
            &[],
        )?;
        check_node(node(3)?, "Cast", &["mask_bool"], &["mask_f"], &["to"])?;
        attr_int_is(node(3)?, "to", 1)?;
        check_node(node(4)?, "Sub", &["one_f32", "mask_f"], &["inv_mask"], &[])?;
        check_node(node(5)?, "Mul", &["inv_mask", "neg_big_f32"], &["neg"], &[])?;
        check_node(
            node(6)?,
            "Unsqueeze",
            &["neg", "unsqueeze_axes_1"],
            &["neg_unsq"],
            &[],
        )?;

        // 幅ごとの 4 ノード
        let mut kernels = Vec::with_capacity(n_widths);
        let pool_names: Vec<String> = (0..n_widths).map(|i| format!("pool{i}")).collect();
        for i in 0..n_widths {
            let base = PREFIX_NODES + BRANCH_NODES * i;
            let (w, b) = (format!("conv{i}_w"), format!("conv{i}_b"));
            let (conv, relu, masked, pool) = (
                format!("conv{i}_out"),
                format!("relu{i}_out"),
                format!("masked{i}_out"),
                format!("pool{i}"),
            );
            let conv_node = node(base)?;
            check_node(
                conv_node,
                "Conv",
                &["emb_bct", &w, &b],
                &[&conv],
                &["kernel_shape", "pads", "strides"],
            )?;
            let kernel = match conv_node.attr("kernel_shape").and_then(|a| a.ints()) {
                Some([k]) => to_usize(*k)?,
                _ => return Err(bad()),
            };
            // 出力長が系列長と一致する条件（奇数）。偶数は書き出し器も拒否している
            if kernel == 0 || kernel > MAX_KERNEL || kernel % 2 == 0 {
                return Err(bad());
            }
            let pad = i64::try_from(kernel / 2).map_err(|_| bad())?;
            attr_ints_are(conv_node, "pads", &[pad, pad])?;
            attr_ints_are(conv_node, "strides", &[1])?;
            check_node(node(base + 1)?, "Relu", &[&conv], &[&relu], &[])?;
            check_node(
                node(base + 2)?,
                "Add",
                &[&relu, "neg_unsq"],
                &[&masked],
                &[],
            )?;
            let rm = node(base + 3)?;
            check_node(rm, "ReduceMax", &[&masked], &[&pool], &["axes", "keepdims"])?;
            attr_ints_are(rm, "axes", &[2])?;
            attr_int_is(rm, "keepdims", 0)?;
            kernels.push(kernel);
        }

        // 後段 3 ノード
        let tail = PREFIX_NODES + BRANCH_NODES * n_widths;
        let pool_refs: Vec<&str> = pool_names.iter().map(String::as_str).collect();
        let concat = node(tail)?;
        check_node(concat, "Concat", &pool_refs, &["pooled"], &["axis"])?;
        attr_int_is(concat, "axis", 1)?;
        let gemm = node(tail + 1)?;
        check_node(
            gemm,
            "Gemm",
            &["pooled", "out_wT", "out_b"],
            &["logits"],
            &["alpha", "beta"],
        )?;
        attr_f32_is(gemm, "alpha", 1.0)?;
        attr_f32_is(gemm, "beta", 1.0)?;
        let softmax = node(tail + 2)?;
        check_node(softmax, "Softmax", &["logits"], &["probs"], &["axis"])?;
        attr_int_is(softmax, "axis", 1)?;

        // initializer
        let inits = Inits::new(graph)?;
        let mut names: Vec<String> = [
            "embed",
            "zero_i64",
            "one_f32",
            "neg_big_f32",
            "unsqueeze_axes_1",
            "out_wT",
            "out_b",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        for i in 0..n_widths {
            names.push(format!("conv{i}_w"));
            names.push(format!("conv{i}_b"));
        }
        inits.require_exactly(&names)?;

        let (edims, _) = inits.f32("embed")?;
        let emb = match edims {
            [n, e] if to_usize(*n)? == N_TOKENS => to_usize(*e)?,
            _ => return Err(bad()),
        };
        if emb == 0 || emb > MAX_EMB {
            return Err(bad());
        }
        let embed = inits.f32_shaped("embed", &[N_TOKENS, emb])?;
        if !embed
            .get(..emb)
            .is_some_and(|row| row.iter().all(|x| x.to_bits() & 0x7fff_ffff == 0))
        {
            // 詰め物行が厳密に 0 でないと、パディングが畳み込みへ混入しうる（REQ-28）
            return Err(bad());
        }
        match inits.i64("zero_i64")? {
            (d, [0]) => shape_is(d, &[1])?,
            _ => return Err(bad()),
        }
        scalar_is(inits.f32_scalar("one_f32")?, 1.0)?;
        let neg_big = inits.f32_scalar("neg_big_f32")?;
        if neg_big.to_bits() != MASK_NEG_VALUE.to_bits() {
            return Err(bad());
        }
        match inits.i64("unsqueeze_axes_1")? {
            (d, [1]) => shape_is(d, &[1])?,
            _ => return Err(bad()),
        }

        let mut branches = Vec::with_capacity(n_widths);
        let mut total_filters = 0usize;
        for (i, &kernel) in kernels.iter().enumerate() {
            let (wdims, _) = inits.f32(&format!("conv{i}_w"))?;
            let filters = match wdims {
                [f, _, _] => to_usize(*f)?,
                _ => return Err(bad()),
            };
            if filters == 0 || filters > MAX_FILTERS {
                return Err(bad());
            }
            let weight = inits.f32_shaped(&format!("conv{i}_w"), &[filters, emb, kernel])?;
            let bias = inits.f32_shaped(&format!("conv{i}_b"), &[filters])?;
            total_filters = total_filters.checked_add(filters).ok_or_else(bad)?;
            branches.push(Branch {
                kernel,
                filters,
                weight: weight.to_vec(),
                bias: bias.to_vec(),
            });
        }
        let (odims, _) = inits.f32("out_wT")?;
        let n_classes = match odims {
            [rows, k] if to_usize(*rows)? == total_filters => to_usize(*k)?,
            _ => return Err(bad()),
        };
        check_n_classes(n_classes)?;
        let out_w = inits.f32_shaped("out_wT", &[total_filters, n_classes])?;
        let out_b = inits.f32_shaped("out_b", &[n_classes])?;
        check_graph_io(graph, n_classes)?;

        Ok(Self {
            emb,
            embed: embed.to_vec(),
            branches,
            neg_big,
            out_w: out_w.to_vec(),
            out_b: out_b.to_vec(),
            n_classes,
        })
    }

    /// 1 系列のスコア（確率。選択肢の宣言順）。`ids` は空でなく、各値が `0..257`（呼び出し側で検査済み）。
    pub(super) fn scores(&self, ids: &[i64]) -> Result<Vec<f64>, BackendError> {
        self.scores_within(ids, MAX_INFER_DURATION)
    }

    /// 計算時間の上限付きのスコア計算。各位置（フィルタ 1 本 × 1 位置の走査）・出力層の各反復・softmax の前で
    /// 経過時間を検査し、超過したら [`BackendError::TimeLimitExceeded`] で打ち切る（REQ-39）。
    fn scores_within(&self, ids: &[i64], limit: Duration) -> Result<Vec<f64>, BackendError> {
        let started = Instant::now();
        let t_len = ids.len();
        if t_len == 0 || t_len > MAX_MAX_BYTES {
            return Err(BackendError::InvalidSequenceLength);
        }
        let e = self.emb;
        // 埋め込み [T, E]（行優先）と、実トークン位置か否かの印 [T]
        let mut x = Vec::with_capacity(t_len.saturating_mul(e));
        let mut is_real = Vec::with_capacity(t_len);
        for &id in ids {
            let id = usize::try_from(id).map_err(|_| BackendError::InvalidTokenId)?;
            let start = id.checked_mul(e).ok_or(BackendError::InvalidTokenId)?;
            let row = self
                .embed
                .get(start..start + e)
                .ok_or(BackendError::InvalidTokenId)?;
            x.extend_from_slice(row);
            is_real.push(id > 0);
        }

        let mut pooled: Vec<f32> = Vec::new();
        for br in &self.branches {
            let pad = br.kernel / 2;
            for f in 0..br.filters {
                let bias = *br.bias.get(f).ok_or(BackendError::Failed)?;
                let mut best = f32::NEG_INFINITY;
                for (t, &real) in is_real.iter().enumerate() {
                    if started.elapsed() >= limit {
                        return Err(BackendError::TimeLimitExceeded);
                    }
                    let mut acc = 0.0f32;
                    for ch in 0..e {
                        let base = f
                            .checked_mul(e)
                            .and_then(|v| v.checked_add(ch))
                            .and_then(|v| v.checked_mul(br.kernel))
                            .ok_or(BackendError::Failed)?;
                        let wrow = br
                            .weight
                            .get(base..base + br.kernel)
                            .ok_or(BackendError::Failed)?;
                        for (j, &wv) in wrow.iter().enumerate() {
                            // 左右 pad のゼロ詰め: 範囲外の位置は 0 を掛けるだけなので飛ばす
                            let pos = match (t + j).checked_sub(pad) {
                                Some(p) if p < t_len => p,
                                _ => continue,
                            };
                            let xv = *x.get(pos * e + ch).ok_or(BackendError::Failed)?;
                            acc += wv * xv;
                        }
                    }
                    // 詰め物位置は最大値の候補から明示的に除外する。加算によるマスクでは
                    // 有限でも巨大な重み・バイアスで詰め物位置が選ばれうる（REQ-28）
                    if real {
                        let relu = (acc + bias).max(0.0);
                        best = best.max(relu);
                    }
                }
                // 実トークンが 1 つも無い系列のみ、書き出し器のマスク値へ倒す（候補なしの -inf を出さない）
                pooled.push(if best == f32::NEG_INFINITY {
                    self.neg_big
                } else {
                    best
                });
            }
        }

        let k = self.n_classes;
        let mut logits = vec![0.0f32; k];
        for (i, &p) in pooled.iter().enumerate() {
            // 出力層の反復でも期限を検査する（許可された最大構成では pooled × out_w も大きい）
            if started.elapsed() >= limit {
                return Err(BackendError::TimeLimitExceeded);
            }
            let start = i.checked_mul(k).ok_or(BackendError::Failed)?;
            let row = self
                .out_w
                .get(start..start + k)
                .ok_or(BackendError::Failed)?;
            for (acc, w) in logits.iter_mut().zip(row) {
                *acc += p * w;
            }
        }
        for (acc, b) in logits.iter_mut().zip(&self.out_b) {
            *acc += b;
        }
        // softmax の前にも検査し、期限超過の結果を返さない
        if started.elapsed() >= limit {
            return Err(BackendError::TimeLimitExceeded);
        }
        softmax_f64(&logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::{ModelKind, OnnxBackend};

    /// REQ-39: 時間上限 0 では最初の位置で打ち切られる（超過時に打ち切れる経路）。
    #[test]
    fn req39_c3_scores_abort_when_time_limit_exceeded() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/onnx_parity/c3.onnx");
        let bytes = std::fs::read(path).expect("fixture read");
        let backend = OnnxBackend::from_bytes(&bytes, ModelKind::C3).expect("load");
        let crate::onnx::Inner::C3(m) = &backend.inner else {
            panic!("c3 expected");
        };
        assert_eq!(
            m.scores_within(&[1, 2, 3], Duration::ZERO).err(),
            Some(BackendError::TimeLimitExceeded)
        );
        assert!(m.scores_within(&[1, 2, 3], MAX_INFER_DURATION).is_ok());
    }
}
