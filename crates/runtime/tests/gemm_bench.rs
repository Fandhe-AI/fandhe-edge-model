//! qwen2 の MLP（gate/up）相当の形での Gemm の性能測定ベンチ（REQ-31・REQ-41・#633・#643）。
//!
//! 役割: 1 スレッドと既定スレッド数の GFLOPS を出し、両者の結果の一致も照合する（REQ-28）。
//! 統合テストに置く理由: lib の単体テスト（cfg(test)）のビルドでは `gemm_item` のコード生成が
//! 本番の lib ビルドと異なり（SLP ベクトル化が崩れ）、遅く出ることがあるため（#636・#643）。
//! 統合テストは本番と同じ lib ビルドを呼ぶ。
//!
//! 実行: `cargo test --release -p fandhe-edge-runtime --test gemm_bench -- --ignored --nocapture`
//! 値は実行環境に依存する。静かな実機で手動実行し、証拠種別は「実機」として記録する。
//! テスト専用のため `unwrap`・添字を使ってよい。

use fandhe_edge_runtime::onnx::ops::Tensor;
use fandhe_edge_runtime::onnx::ops::arith::{bf16_to_f32, gemm_trans_b_with};
use std::time::Instant;

#[test]
#[ignore = "実機の静かな状態で手動実行する性能測定。REQ-31・REQ-41・#643"]
fn gemm_bench_qwen2_mlp_shape() {
    let (m, n, k) = (320usize, 8960usize, 1536usize);
    // 外部 crate を使わない seed 固定の LCG で決定的に値を作る
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    let a = Tensor::new(
        vec![m, k],
        (0..m * k)
            .map(|_| (next() % 2001) as f32 / 1000.0 - 1.0)
            .collect(),
    )
    .unwrap();
    // bf16 は f32 の上位 16 bit。指数を抑えるため [-1, 1] の f32 から作る
    let wdata: Vec<u16> = (0..n * k)
        .map(|_| ((((next() % 2001) as f32 / 1000.0 - 1.0).to_bits()) >> 16) as u16)
        .collect();
    let w = Tensor::new(vec![n, k], wdata).unwrap();
    let flops = 2.0 * (m * n * k) as f64;
    let avail = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);

    let mut outs = Vec::new();
    for (label, th) in [("single", Some(1)), ("default", None)] {
        let run = || gemm_trans_b_with(th, &a, &w, None, bf16_to_f32, None).unwrap();
        outs.push(run()); // ウォームアップ
        let mut secs: Vec<f64> = (0..5)
            .map(|_| {
                let t0 = Instant::now();
                std::hint::black_box(run());
                t0.elapsed().as_secs_f64()
            })
            .collect();
        secs.sort_by(f64::total_cmp);
        let med = secs[2];
        println!(
            "gemm_bench {label}: median={med:.4}s gflops={:.1} (available_parallelism={avail})",
            flops / med / 1e9
        );
    }
    assert_eq!(outs[0].data(), outs[1].data());
}
