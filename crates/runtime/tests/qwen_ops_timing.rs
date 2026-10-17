//! Qwen2.5-1.5B 形状での演算ごとの処理時間の計測（REQ-41・REQ-32・TASK-41.7-7・#421）。
//!
//! 役割: #416〜#419 のカーネル（自作・std のみ・単一スレッド）が 1.5B の 1 層の形状でどれだけかかるかの
//! 目安を出す。結果は `docs/design/qwen2-scoring-onnx-ops.md` 7 節に記録する。
//! 実行は重い形状での計測で、正しさの検査ではないため `#[ignore]` で既定の `make test` から分離する
//! （既定集合で動くテストを移したものではない）。`--release` で実行すること。
//!
//! 証拠種別: テストハーネス（合成の bf16 重み・1 層・T=128・K=4・M=8）。1 推論あたりの値は推定。
//! 実機の p95・RSS は TASK-41.11（#440・#442）が担う。時間のしきい値は置かない。
//! テスト専用のため `unwrap`・添字を使ってよい。

use fandhe_edge_runtime::onnx::ops::arith::*;
use fandhe_edge_runtime::onnx::ops::mask_rope::*;
use fandhe_edge_runtime::onnx::ops::norm_act::*;
use fandhe_edge_runtime::onnx::ops::shape::*;
use fandhe_edge_runtime::onnx::ops::{OpError, Tensor};
use std::hint::black_box;
use std::time::Instant;

const H: usize = 1536;
const NH: usize = 12;
const NKV: usize = 2;
const HD: usize = 128;
const INTER: usize = 8960;
const V: usize = 151_936;
const LAYERS: u64 = 28;
const T: usize = 128;
const K: usize = 4;
const M: usize = 8;
const RUNS: usize = 3;

/// 結果が有限か（f32 以外は常に真）。
trait Fin {
    fn fin(&self) -> bool;
}
impl Fin for Tensor<f32> {
    fn fin(&self) -> bool {
        self.data().iter().all(|v| v.is_finite())
    }
}
impl Fin for Tensor<i64> {
    fn fin(&self) -> bool {
        true
    }
}
impl Fin for Tensor<bool> {
    fn fin(&self) -> bool {
        true
    }
}
impl Fin for Tensor<f64> {
    fn fin(&self) -> bool {
        self.data().iter().all(|v| v.is_finite())
    }
}

struct Row {
    op: &'static str,
    shape: String,
    median_ns: u128,
    /// 1 層あたりの回数（層をまたぐ処理は 0）。
    per_layer: u64,
    once: bool,
}

fn bench<R: Fin>(
    rows: &mut Vec<Row>,
    op: &'static str,
    shape: &str,
    per_layer: u64,
    once: bool,
    mut f: impl FnMut() -> Result<R, OpError>,
) {
    let mut ns: Vec<u128> = (0..RUNS)
        .map(|_| {
            let t0 = Instant::now();
            let r = black_box(f()).expect("op ok");
            let d = t0.elapsed().as_nanos();
            assert!(r.fin(), "non-finite output: {op}");
            d
        })
        .collect();
    ns.sort_unstable();
    let med = ns[RUNS / 2];
    println!(
        "{{\"op\":\"{op}\",\"shape\":\"{shape}\",\"median_ns\":{med},\"evidence\":\"test_harness\",\"profile\":\"{}\"}}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    rows.push(Row {
        op,
        shape: shape.into(),
        median_ns: med,
        per_layer,
        once,
    });
}

fn f32t(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let d = (0..n).map(|i| ((i % 97) as f32 - 48.0) / 64.0).collect();
    Tensor::new(shape.to_vec(), d).unwrap()
}
fn bf(shape: &[usize]) -> Tensor<u16> {
    let n: usize = shape.iter().product();
    // 0x3c00 付近の小さな正の bf16（有限）。
    let d = (0..n).map(|i| 0x3c00 + (i & 0x7f) as u16).collect();
    Tensor::new(shape.to_vec(), d).unwrap()
}
fn dims(d: &[usize]) -> Tensor<i64> {
    Tensor::new(vec![d.len()], d.iter().map(|&x| x as i64).collect()).unwrap()
}
fn iv(v: &[i64]) -> Tensor<i64> {
    Tensor::new(vec![v.len()], v.to_vec()).unwrap()
}

#[test]
#[ignore = "measurement at Qwen2.5-1.5B shapes; run with --release (TASK-41.7-7)"]
fn req41_qwen_ops_timing_at_1_5b_shapes() {
    let mut r: Vec<Row> = Vec::new();
    let g = |x: &Tensor<f32>, w: &Tensor<u16>, b: Option<&Tensor<f32>>| {
        gemm_trans_b(x, w, b, bf16_to_f32, None)
    };

    // 線形層（prefix は T 行、選択肢ブロックは K*M 行）。
    let xp = f32t(&[T, H]);
    let xb = f32t(&[K * M, H]);
    let (wq, wkv, wi, wd) = (
        bf(&[H, H]),
        bf(&[NKV * HD, H]),
        bf(&[INTER, H]),
        bf(&[H, INTER]),
    );
    let bq = f32t(&[H]);
    let bkv = f32t(&[NKV * HD]);
    let xi = f32t(&[T, INTER]);
    for (x, tag) in [(&xp, T), (&xb, K * M)] {
        bench(
            &mut r,
            "Gemm q/o",
            &format!("[{tag},{H}]x[{H},{H}]"),
            2,
            false,
            || g(x, &wq, Some(&bq)),
        );
        bench(
            &mut r,
            "Gemm k/v",
            &format!("[{tag},{H}]x[{},{H}]", NKV * HD),
            2,
            false,
            || g(x, &wkv, Some(&bkv)),
        );
        bench(
            &mut r,
            "Gemm gate/up",
            &format!("[{tag},{H}]x[{INTER},{H}]"),
            2,
            false,
            || g(x, &wi, None),
        );
    }
    bench(
        &mut r,
        "Gemm down",
        &format!("[{T},{INTER}]x[{H},{INTER}]"),
        1,
        false,
        || g(&xi, &wd, None),
    );
    let xib = f32t(&[K * M, INTER]);
    bench(
        &mut r,
        "Gemm down",
        &format!("[{},{INTER}]x[{H},{INTER}]", K * M),
        1,
        false,
        || g(&xib, &wd, None),
    );

    // 注意（prefix: q [1,2,6,T,hd] × kT [1,2,1,hd,T]、ブロック: [K,2,6,M,hd] × [K,2,1,hd,T+M]）。
    let grp = NH / NKV;
    let (q1, k1) = (f32t(&[1, NKV, grp, T, HD]), f32t(&[1, NKV, 1, HD, T]));
    bench(
        &mut r,
        "MatMul qk",
        &format!("[1,{NKV},{grp},{T},{HD}]x[1,{NKV},1,{HD},{T}]"),
        1,
        false,
        || matmul(&q1, &k1, None),
    );
    let s1 = f32t(&[1, NKV, grp, T, T]);
    let v1 = f32t(&[1, NKV, 1, T, HD]);
    bench(
        &mut r,
        "MatMul av",
        &format!("[1,{NKV},{grp},{T},{T}]x[1,{NKV},1,{T},{HD}]"),
        1,
        false,
        || matmul(&s1, &v1, None),
    );
    let s = T + M;
    let (q2, k2) = (f32t(&[K, NKV, grp, M, HD]), f32t(&[K, NKV, 1, HD, s]));
    bench(
        &mut r,
        "MatMul qk",
        &format!("[{K},{NKV},{grp},{M},{HD}]x[{K},{NKV},1,{HD},{s}]"),
        1,
        false,
        || matmul(&q2, &k2, None),
    );
    let (s2, v2) = (f32t(&[K, NKV, grp, M, s]), f32t(&[K, NKV, 1, s, HD]));
    bench(
        &mut r,
        "MatMul av",
        &format!("[{K},{NKV},{grp},{M},{s}]x[{K},{NKV},1,{s},{HD}]"),
        1,
        false,
        || matmul(&s2, &v2, None),
    );
    bench(
        &mut r,
        "Softmax",
        &format!("[1,{NKV},{grp},{T},{T}]"),
        1,
        false,
        || softmax_last_axis(&s1),
    );

    // マスク（Where は 3 入力のブロードキャスト）。
    let fut = greater_i64(
        &Tensor::new(vec![1, T], (0..T as i64).collect()).unwrap(),
        &Tensor::new(vec![T, 1], (0..T as i64).collect()).unwrap(),
    )
    .unwrap();
    let neg = f32t(&[]);
    bench(
        &mut r,
        "Where",
        &format!("[{T},{T}]x[]x[1,{NKV},{grp},{T},{T}]"),
        1,
        false,
        || where_f32(&fut, &neg, &s1),
    );
    bench(
        &mut r,
        "Greater",
        &format!("[1,{T}]>[{T},1]"),
        1,
        false,
        || {
            greater_i64(
                &Tensor::new(vec![1, T], (0..T as i64).collect()).unwrap(),
                &Tensor::new(vec![T, 1], (0..T as i64).collect()).unwrap(),
            )
        },
    );

    // RMSNorm の構成演算（prefix 形状。1 層に 2 回）。
    let w = f32t(&[H]);
    let eps = f32t(&[]);
    let var = Tensor::new(vec![T, 1], vec![0.25f32; T]).unwrap();
    bench(&mut r, "Mul x*x", &format!("[{T},{H}]"), 2, false, || {
        mul(&xp, &xp)
    });
    bench(
        &mut r,
        "ReduceMean",
        &format!("[{T},{H}]"),
        2,
        false,
        || reduce_mean_last_axis(&xp),
    );
    bench(&mut r, "Add eps", &format!("[{T},1]"), 2, false, || {
        add(&var, &eps)
    });
    bench(&mut r, "Sqrt", &format!("[{T},1]"), 2, false, || sqrt(&var));
    bench(&mut r, "Reciprocal", &format!("[{T},1]"), 2, false, || {
        reciprocal(&var)
    });
    bench(
        &mut r,
        "Mul x*r",
        &format!("[{T},{H}]x[{T},1]"),
        2,
        false,
        || mul(&xp, &var),
    );
    bench(
        &mut r,
        "Mul *w",
        &format!("[{T},{H}]x[{H}]"),
        2,
        false,
        || mul(&xp, &w),
    );

    // SwiGLU・残差。
    bench(
        &mut r,
        "Sigmoid",
        &format!("[{T},{INTER}]"),
        1,
        false,
        || sigmoid(&xi),
    );
    bench(
        &mut r,
        "Mul g*sig",
        &format!("[{T},{INTER}]"),
        2,
        false,
        || mul(&xi, &xi),
    );
    bench(
        &mut r,
        "Add residual",
        &format!("[{T},{H}]"),
        2,
        false,
        || add(&xp, &xp),
    );

    // RoPE（q・k に 1 回ずつ。表は 1 回で全層共通だが、ここでは層あたり 1 回として数える）。
    let qh = f32t(&[1, NH, T, HD]);
    let cs = f32t(&[T, HD / 2]);
    bench(
        &mut r,
        "Split",
        &format!("[1,{NH},{T},{HD}]"),
        2,
        false,
        || split(&qh, -1, &dims(&[HD / 2, HD / 2])).map(|mut v| v.remove(0)),
    );
    let half = f32t(&[1, NH, T, HD / 2]);
    bench(
        &mut r,
        "Mul rope",
        &format!("[1,{NH},{T},{HD}/2]x[{T},{HD}/2]"),
        8,
        false,
        || mul(&half, &cs),
    );
    bench(
        &mut r,
        "Sub rope",
        &format!("[1,{NH},{T},{HD}/2]"),
        2,
        false,
        || sub(&half, &half),
    );
    bench(
        &mut r,
        "Concat rope",
        &format!("[1,{NH},{T},{HD}/2]x2"),
        2,
        false,
        || concat(&[&half, &half], -1),
    );
    bench(
        &mut r,
        "Sin",
        &format!("[{T},{}]", HD / 2),
        1,
        false,
        || sin(&cs),
    );
    bench(
        &mut r,
        "Cos",
        &format!("[{T},{}]", HD / 2),
        1,
        false,
        || cos(&cs),
    );

    // 形状操作。
    bench(
        &mut r,
        "Transpose",
        &format!("[1,{T},{NH},{HD}]"),
        3,
        false,
        || transpose(&f32t(&[1, T, NH, HD]), &[0, 2, 1, 3]),
    );
    let pk = f32t(&[1, NKV, T, HD]);
    bench(
        &mut r,
        "Expand kv",
        &format!("[1,{NKV},{T},{HD}]->[{K},{NKV},{T},{HD}]"),
        2,
        false,
        || expand(&pk, &dims(&[K, NKV, T, HD])),
    );
    let ek = f32t(&[K, NKV, T, HD]);
    let bk = f32t(&[K, NKV, M, HD]);
    bench(
        &mut r,
        "Concat kv",
        &format!("[{K},{NKV},{T}+{M},{HD}]"),
        2,
        false,
        || concat(&[&ek, &bk], 2),
    );

    // 層をまたがない処理。
    let emb = bf(&[V, H]);
    let ids = Tensor::new(
        vec![T],
        (0..T as i64).map(|i| (i * 1181) % V as i64).collect(),
    )
    .unwrap();
    bench(
        &mut r,
        "Gather embed",
        &format!("[{V},{H}] ids[{T}]"),
        0,
        true,
        || gather_axis0(&emb, &ids, bf16_to_f32),
    );
    let hx = f32t(&[K * M, H]);
    bench(
        &mut r,
        "Gemm lm_head",
        &format!("[{},{H}]x[{V},{H}]", K * M),
        0,
        true,
        || g(&hx, &emb, None),
    );
    let lg = f32t(&[K * M, V]);
    bench(
        &mut r,
        "LogSoftmax",
        &format!("[{},{V}]", K * M),
        0,
        true,
        || log_softmax_last_axis(&lg),
    );
    let lp = f32t(&[K, M, V]);
    let ci = Tensor::new(
        vec![K, M, 1],
        (0..(K * M) as i64).map(|i| i * 997 % V as i64).collect(),
    )
    .unwrap();
    bench(
        &mut r,
        "GatherElements",
        &format!("[{K},{M},{V}] idx[{K},{M},1]"),
        0,
        true,
        || gather_elements_last_axis(&lp, &ci),
    );
    let pk3 = f32t(&[K, M, 1]);
    bench(
        &mut r,
        "ReduceSum",
        &format!("[{K},{M},1]"),
        0,
        true,
        || reduce_sum(&pk3, &iv(&[1, 2])),
    );

    // 1 推論あたりの推定 = 1 層の合計 x 28 + 層をまたがない処理（推定。実機の p95 ではない）。
    let layer_ns: u128 = r
        .iter()
        .filter(|x| !x.once)
        .map(|x| x.median_ns * u128::from(x.per_layer))
        .sum();
    let once_ns: u128 = r.iter().filter(|x| x.once).map(|x| x.median_ns).sum();
    let total = layer_ns * u128::from(LAYERS as u32) + once_ns;
    println!(
        "{{\"summary\":\"per_inference_estimate\",\"layer_ns\":{layer_ns},\"once_ns\":{once_ns},\"layers\":{LAYERS},\"total_ns\":{total},\"evidence\":\"estimate\"}}"
    );
    let mut top: Vec<&Row> = r.iter().collect();
    top.sort_by_key(|x| std::cmp::Reverse(x.median_ns * u128::from(x.per_layer.max(1))));
    for x in top.iter().take(5) {
        println!(
            "top: {} {} {}ns x{}",
            x.op, x.shape, x.median_ns, x.per_layer
        );
    }
}
