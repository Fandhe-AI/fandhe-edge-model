//! 推論のみの待ち時間の反復計測ハーネス（REQ-31・TASK-31.1-1・#127）。
//!
//! # 役割
//!
//! 構築済みの [`InferencePipeline`] に対し、`infer_one`（バッチ 1 件と同じ共通経路）を
//! warmup 回だけ空回しした後に `iters` 回計測し、1 回ごとの所要時間（ns）を反復順に返す。
//! 出典は PoC-14 の `ref_bench.py` と PoC runtime の `bench`（warmup 20・iters 1000）。
//!
//! # 計測区間
//!
//! 区間は推論 1 回の呼び出しのみ。CLI 起動・モデルロード・入力ファイル読み込みは区間外で、
//! パイプライン構築（ロード）は呼び出し側が本関数の前に済ませる（REQ-31 の対象範囲）。
//!
//! # 範囲外（実装済みを装わない）
//!
//! - p95 算出・レポート・参考値の明記: #128（TASK-31.1-2）。上限照合と `limit_exceeded`:
//!   TASK-31.2・#129。上限ちょうどの境界判定: #130。JSON 出力・CLI 接続: TASK-33.x
//! - 実前処理（#112）・ONNX 推論（#113）が未実装のため、実モデルでの計測は #113 完了後
//! - 実機（静かな Mac）での実計測と実測値の記録は人間の作業。本モジュールのテストの
//!   証拠種別はテストハーネス（偽の時計・模擬バックエンド）のみ

use crate::pipeline::{
    InferError, InferencePipeline, MAX_INFER_BATCH_LEN, Prediction, Preprocessor, ScoringBackend,
};
use std::hint::black_box;
use std::time::Instant;

/// warmup 回数の既定値（PoC-14 の bench に合わせる）。
pub const DEFAULT_LATENCY_WARMUP: usize = 20;
/// 計測回数の既定値（PoC-14 の bench に合わせる）。
pub const DEFAULT_LATENCY_ITERS: usize = 1000;
/// warmup 回数の上限（REQ-39 の暫定資源上限。無限に近い待ちを作らない）。
pub const MAX_LATENCY_WARMUP: usize = 1_000_000;
/// 計測回数の上限（REQ-39 の暫定資源上限。`Vec<u64>` で約 8 MiB）。
pub const MAX_LATENCY_ITERS: usize = 1_000_000;

/// 単調時計の継ぎ目。テストでは偽の時計を注入して計測値を決定的にする。
pub trait Clock {
    /// 単調に増加する現在時刻（ns。起点は実装依存で差分のみ意味を持つ）。
    fn now_ns(&self) -> u64;
}

/// `std::time::Instant` に基づく単調時計。
#[derive(Debug, Clone, Copy)]
pub struct MonotonicClock {
    origin: Instant,
}

impl MonotonicClock {
    /// 現在時刻を起点にして作る。
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    fn now_ns(&self) -> u64 {
        // u64 の ns は約 584 年分で実用上溢れない。溢れた場合は飽和させる。
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}

/// 計測ハーネスのエラー（入力本文を保持しない）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LatencyError {
    /// 計測用の入力が 0 件。
    NoInputs,
    /// 計測用の入力が [`MAX_INFER_BATCH_LEN`] を超える（REQ-39）。
    TooManyInputs {
        /// 入力件数。
        len: usize,
        /// 上限。
        limit: usize,
    },
    /// 計測回数が 0。
    ZeroIterations,
    /// 計測回数が [`MAX_LATENCY_ITERS`] を超える（REQ-39）。
    TooManyIterations {
        /// 指定値。
        iters: usize,
        /// 上限。
        limit: usize,
    },
    /// warmup 回数が [`MAX_LATENCY_WARMUP`] を超える（REQ-39）。
    TooManyWarmup {
        /// 指定値。
        warmup: usize,
        /// 上限。
        limit: usize,
    },
    /// 推論が失敗した（fail-closed。失敗経路の時間を計測値に混ぜない）。
    Inference {
        /// 失敗した段階。
        phase: LatencyPhase,
        /// 段階内の 0 始まりの反復番号。
        iteration: usize,
        /// 推論エラーの機械可読コード。
        code: &'static str,
    },
    /// 時計が逆行した。
    NonMonotonicClock {
        /// 計測の 0 始まりの反復番号。
        iteration: usize,
    },
}

/// 推論失敗が起きた段階。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyPhase {
    /// warmup（計測値に含まれない）。
    Warmup,
    /// 計測。
    Measure,
}

impl LatencyError {
    /// 機械可読なエラーコード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::NoInputs => "no_inputs",
            Self::TooManyInputs { .. } => "too_many_inputs",
            Self::ZeroIterations => "zero_iterations",
            Self::TooManyIterations { .. } => "too_many_iterations",
            Self::TooManyWarmup { .. } => "too_many_warmup",
            Self::Inference { .. } => "inference_failed",
            Self::NonMonotonicClock { .. } => "non_monotonic_clock",
        }
    }
}

/// 計測設定。構築時に上限を検証し、壊れた値を表現させない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyConfig {
    warmup: usize,
    iters: usize,
}

impl LatencyConfig {
    /// warmup 回数と計測回数から作る。`iters` は 1 以上、各上限以下。
    pub fn new(warmup: usize, iters: usize) -> Result<Self, LatencyError> {
        if iters == 0 {
            return Err(LatencyError::ZeroIterations);
        }
        if iters > MAX_LATENCY_ITERS {
            return Err(LatencyError::TooManyIterations {
                iters,
                limit: MAX_LATENCY_ITERS,
            });
        }
        if warmup > MAX_LATENCY_WARMUP {
            return Err(LatencyError::TooManyWarmup {
                warmup,
                limit: MAX_LATENCY_WARMUP,
            });
        }
        Ok(Self { warmup, iters })
    }

    /// warmup 回数。
    pub fn warmup(&self) -> usize {
        self.warmup
    }

    /// 計測回数。
    pub fn iters(&self) -> usize {
        self.iters
    }
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            warmup: DEFAULT_LATENCY_WARMUP,
            iters: DEFAULT_LATENCY_ITERS,
        }
    }
}

/// 計測結果。個々の計測値（ns）を反復順に保持する。p95・合否は持たない（#128・#129）。
#[derive(Clone, PartialEq, Eq)]
pub struct LatencySamples {
    warmup: usize,
    iters: usize,
    samples_ns: Vec<u64>,
}

impl LatencySamples {
    /// 実行した warmup 回数（計測値には含まれない）。
    pub fn warmup(&self) -> usize {
        self.warmup
    }

    /// 計測回数（`samples_ns().len()` と一致）。
    pub fn iters(&self) -> usize {
        self.iters
    }

    /// 1 回ごとの所要時間（ns）。反復順。
    pub fn samples_ns(&self) -> &[u64] {
        &self.samples_ns
    }
}

impl std::fmt::Debug for LatencySamples {
    /// 件数のみ表示する。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LatencySamples(warmup={}, iters={})",
            self.warmup, self.iters
        )
    }
}

/// 推論のみの待ち時間を `config.iters()` 回計測する。
///
/// 計測対象は `pipeline.infer_one`（バッチ 1 件と同じ共通経路。REQ-28）の 1 回の呼び出しのみ。
/// `inputs` は循環して使う。推論関数へ渡すのは入力文字列だけ（REQ-27）。
/// 呼び出し元（想定）: CLI の bench 工程（TASK-33.x）・#128 のレポート生成。
pub fn measure_latency<P: Preprocessor, B: ScoringBackend, C: Clock>(
    pipeline: &InferencePipeline<P, B>,
    inputs: &[&str],
    config: &LatencyConfig,
    clock: &C,
) -> Result<LatencySamples, LatencyError> {
    measure_with(|input| pipeline.infer_one(input), inputs, config, clock)
}

/// `measure_latency` の本体。推論関数を注入できる形にして経路と切り離す。
fn measure_with<F, C>(
    mut predict: F,
    inputs: &[&str],
    config: &LatencyConfig,
    clock: &C,
) -> Result<LatencySamples, LatencyError>
where
    F: FnMut(&str) -> Result<Prediction, InferError>,
    C: Clock,
{
    if inputs.is_empty() {
        return Err(LatencyError::NoInputs);
    }
    if inputs.len() > MAX_INFER_BATCH_LEN {
        return Err(LatencyError::TooManyInputs {
            len: inputs.len(),
            limit: MAX_INFER_BATCH_LEN,
        });
    }

    for (iteration, input) in inputs.iter().cycle().take(config.warmup).enumerate() {
        match predict(black_box(input)) {
            Ok(p) => {
                black_box(p);
            }
            Err(e) => {
                return Err(LatencyError::Inference {
                    phase: LatencyPhase::Warmup,
                    iteration,
                    code: e.code(),
                });
            }
        }
    }

    let mut samples_ns = Vec::with_capacity(config.iters);
    for (iteration, input) in inputs.iter().cycle().take(config.iters).enumerate() {
        let t0 = clock.now_ns();
        let result = black_box(predict(black_box(input)));
        let t1 = clock.now_ns();
        if let Err(e) = result {
            return Err(LatencyError::Inference {
                phase: LatencyPhase::Measure,
                iteration,
                code: e.code(),
            });
        }
        let elapsed = t1
            .checked_sub(t0)
            .ok_or(LatencyError::NonMonotonicClock { iteration })?;
        samples_ns.push(elapsed);
    }

    Ok(LatencySamples {
        warmup: config.warmup,
        iters: config.iters,
        samples_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn req39_latency_config_rejects_zero_iterations() {
        assert_eq!(LatencyConfig::new(0, 0), Err(LatencyError::ZeroIterations));
    }

    #[test]
    fn req39_latency_config_iters_boundary() {
        assert!(LatencyConfig::new(0, MAX_LATENCY_ITERS).is_ok());
        assert_eq!(
            LatencyConfig::new(0, MAX_LATENCY_ITERS + 1),
            Err(LatencyError::TooManyIterations {
                iters: MAX_LATENCY_ITERS + 1,
                limit: MAX_LATENCY_ITERS
            })
        );
    }

    #[test]
    fn req39_latency_config_warmup_boundary() {
        assert!(LatencyConfig::new(MAX_LATENCY_WARMUP, 1).is_ok());
        assert_eq!(
            LatencyConfig::new(MAX_LATENCY_WARMUP + 1, 1),
            Err(LatencyError::TooManyWarmup {
                warmup: MAX_LATENCY_WARMUP + 1,
                limit: MAX_LATENCY_WARMUP
            })
        );
    }

    #[test]
    fn default_config_matches_poc14_values() {
        let c = LatencyConfig::default();
        assert_eq!((c.warmup(), c.iters()), (20, 1000));
    }

    #[test]
    fn latency_error_codes() {
        let cases: Vec<(LatencyError, &str)> = vec![
            (LatencyError::NoInputs, "no_inputs"),
            (
                LatencyError::TooManyInputs { len: 2, limit: 1 },
                "too_many_inputs",
            ),
            (LatencyError::ZeroIterations, "zero_iterations"),
            (
                LatencyError::TooManyIterations { iters: 2, limit: 1 },
                "too_many_iterations",
            ),
            (
                LatencyError::TooManyWarmup {
                    warmup: 2,
                    limit: 1,
                },
                "too_many_warmup",
            ),
            (
                LatencyError::Inference {
                    phase: LatencyPhase::Measure,
                    iteration: 0,
                    code: "backend_failed",
                },
                "inference_failed",
            ),
            (
                LatencyError::NonMonotonicClock { iteration: 0 },
                "non_monotonic_clock",
            ),
        ];
        for (e, code) in cases {
            assert_eq!(e.code(), code);
        }
    }

    #[test]
    fn monotonic_clock_is_non_decreasing() {
        let c = MonotonicClock::new();
        let a = c.now_ns();
        let b = c.now_ns();
        assert!(b >= a);
    }
}
