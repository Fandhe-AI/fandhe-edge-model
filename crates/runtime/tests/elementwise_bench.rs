//! qwen2 の大きい要素演算（Sigmoid・Mul・LogSoftmax・Transpose）の所要時間ベンチ（REQ-31・REQ-41・#648）。
//!
//! 役割: 並列化（#648）の前後比較用の参考値を出す。統合テストに置く理由は gemm_bench と同じ
//! （本番と同じ lib ビルドで測る。#643）。
//!
//! 実行: `cargo test --release -p fandhe-edge-runtime --test elementwise_bench -- --ignored --nocapture`
//! 値は実行環境に依存する。証拠種別はテストハーネス（実機の判定は人が行う）。テスト専用のため `unwrap` を使ってよい。

use fandhe_edge_runtime::onnx::ops::Tensor;
use fandhe_edge_runtime::onnx::ops::arith::mul;
use fandhe_edge_runtime::onnx::ops::norm_act::{log_softmax_last_axis, sigmoid};
use fandhe_edge_runtime::onnx::ops::shape::transpose;
use std::time::Instant;

fn t(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(
        shape.to_vec(),
        (0..n).map(|i| (i as f32 * 0.001).sin() * 3.0).collect(),
    )
    .unwrap()
}

/// 1 回ウォームアップしてから 7 回測った中央値（ms）。
fn median_ms(mut f: impl FnMut()) -> f64 {
    f();
    let mut v: Vec<f64> = (0..7)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[3]
}

#[test]
#[ignore = "実機の静かな状態で手動実行する性能測定。REQ-31・REQ-41・#648"]
fn elementwise_bench_qwen2_shapes() {
    let a = t(&[156, 8960]);
    let b = t(&[156, 8960]);
    let ms = median_ms(|| drop(std::hint::black_box(sigmoid(&a).unwrap())));
    println!("sigmoid [156,8960]: {ms:.2} ms");
    let ms = median_ms(|| drop(std::hint::black_box(mul(&a, &b).unwrap())));
    println!("mul [156,8960]x[156,8960]: {ms:.2} ms");
    let l = t(&[26, 6, 37984]);
    let ms = median_ms(|| drop(std::hint::black_box(log_softmax_last_axis(&l).unwrap())));
    println!("log_softmax [26,6,37984]: {ms:.2} ms");
    let k = t(&[26, 2, 1, 195, 128]);
    let ms = median_ms(|| {
        drop(std::hint::black_box(
            transpose(&k, &[0, 1, 2, 4, 3]).unwrap(),
        ))
    });
    println!("transpose [26,2,1,195,128]: {ms:.2} ms");
}
