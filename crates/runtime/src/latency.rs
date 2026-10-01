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
//! # 資源上限（REQ-39）
//!
//! 総入力バイト数は推論前に検査する。計測全体の期限は各反復の開始前、推論 1 回の上限は
//! 呼び出しの復帰後に検査し、超過は計測失敗として返す。同期呼び出し中の推論を強制中断は
//! できない（協調的な検出。完全に停止する推論の中断は呼び出し側の子プロセス制御の責務）。
//!
//! # 範囲外（実装済みを装わない）
//!
//! - p95 算出・レポート・参考値の明記は [`crate::latency_report`]（#128・TASK-31.1-2）で実装済み。
//!   上限照合と `limit_exceeded`: [`crate::latency_limit`]（TASK-31.2・#129）で実装済み。上限ちょうどの境界判定: #130。
//!   CLI 接続: `package` 工程が定義の `limits.max_infer_p95_us` があるときだけ計測・照合する（#338）。
//!   JSON 出力: #340
//! - 実前処理（#112）・ONNX 推論（#113）が未実装のため、実モデルでの計測は #113 完了後
//! - 実機（静かな Mac）での実計測と実測値の記録は人間の作業。本モジュールのテストの
//!   証拠種別はテストハーネス（偽の時計・模擬バックエンド）のみ

use crate::pipeline::{
    InferError, InferencePipeline, MAX_INFER_BATCH_LEN, MAX_INFER_BATCH_TOTAL_BYTES, Prediction,
    Preprocessor, ScoringBackend,
};
use fandhe_edge_core::limits;
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

/// 計測全体（warmup と計測の合計）の時間上限の既定値（ns。10 分。REQ-39 の暫定値）。
pub const DEFAULT_LATENCY_TOTAL_TIMEOUT_NS: u64 = 600_000_000_000;
/// 推論 1 回あたりの時間上限の既定値（ns。10 秒。REQ-39 の暫定値）。値の出所は共通コアの `limits`。
pub const DEFAULT_LATENCY_PER_INFER_TIMEOUT_NS: u64 = limits::INFER_TIME_LIMIT_NS;
/// 各時間上限として指定できる最大値（ns。1 時間。REQ-39 の暫定値）。値の出所は共通コアの `limits`。
pub const MAX_LATENCY_TIMEOUT_NS: u64 = limits::MAX_TIME_LIMIT_NS;

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
    /// 計測用の入力の総バイト数が [`MAX_INFER_BATCH_TOTAL_BYTES`] を超える（推論前に検査。REQ-39）。
    TotalInputTooLarge {
        /// 総入力バイト数（飽和加算）。
        total: usize,
        /// 上限。
        limit: usize,
    },
    /// 時間上限が 0、または [`MAX_LATENCY_TIMEOUT_NS`] を超える（REQ-39）。
    InvalidTimeout {
        /// 指定値（ns）。
        timeout_ns: u64,
    },
    /// 計測全体の時間上限を超えた（各反復の境界で検査する。REQ-39）。
    DeadlineExceeded {
        /// 超過した段階。
        phase: LatencyPhase,
        /// 段階内の 0 始まりの反復番号（次に実行するはずだった反復）。
        iteration: usize,
        /// 上限（ns）。
        limit_ns: u64,
    },
    /// 推論 1 回が時間上限を超えた（呼び出しが戻った後に検出する。REQ-39）。
    InferenceTimeout {
        /// 超過した段階。
        phase: LatencyPhase,
        /// 段階内の 0 始まりの反復番号。
        iteration: usize,
        /// 上限（ns）。
        limit_ns: u64,
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
        /// 逆行を検出した段階。
        phase: LatencyPhase,
        /// 段階内の 0 始まりの反復番号。
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
            Self::TotalInputTooLarge { .. } => "total_input_too_large",
            Self::InvalidTimeout { .. } => "invalid_timeout",
            Self::DeadlineExceeded { .. } => "deadline_exceeded",
            Self::InferenceTimeout { .. } => "inference_timeout",
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
    total_timeout_ns: u64,
    per_infer_timeout_ns: u64,
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
        Ok(Self {
            warmup,
            iters,
            total_timeout_ns: DEFAULT_LATENCY_TOTAL_TIMEOUT_NS,
            per_infer_timeout_ns: DEFAULT_LATENCY_PER_INFER_TIMEOUT_NS,
        })
    }

    /// 計測全体・推論 1 回の時間上限（ns）を差し替える。各値は 1 以上 [`MAX_LATENCY_TIMEOUT_NS`] 以下。
    ///
    /// 上限の検出は協調的で、同期呼び出し中の推論を中断はできない。反復の境界（全体）と
    /// 呼び出しの復帰後（1 回）に検査し、超過したら計測を失敗として返す（REQ-39）。
    pub fn with_timeouts(
        mut self,
        total_timeout_ns: u64,
        per_infer_timeout_ns: u64,
    ) -> Result<Self, LatencyError> {
        for t in [total_timeout_ns, per_infer_timeout_ns] {
            if t == 0 || t > MAX_LATENCY_TIMEOUT_NS {
                return Err(LatencyError::InvalidTimeout { timeout_ns: t });
            }
        }
        self.total_timeout_ns = total_timeout_ns;
        self.per_infer_timeout_ns = per_infer_timeout_ns;
        Ok(self)
    }

    /// 計測全体の時間上限（ns）。
    pub fn total_timeout_ns(&self) -> u64 {
        self.total_timeout_ns
    }

    /// 推論 1 回の時間上限（ns）。
    pub fn per_infer_timeout_ns(&self) -> u64 {
        self.per_infer_timeout_ns
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
            total_timeout_ns: DEFAULT_LATENCY_TOTAL_TIMEOUT_NS,
            per_infer_timeout_ns: DEFAULT_LATENCY_PER_INFER_TIMEOUT_NS,
        }
    }
}

/// 計測結果。個々の計測値（ns）を反復順に保持する。p95 は [`crate::latency_report`]、合否は持たない（上限照合は [`crate::latency_limit`]）。
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

/// 他モジュールの単体テスト用に計測値から直接 [`LatencySamples`] を作る（warmup 0）。
#[cfg(test)]
pub(crate) fn test_samples(v: &[u64]) -> LatencySamples {
    LatencySamples {
        warmup: 0,
        iters: v.len(),
        samples_ns: v.to_vec(),
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
/// 呼び出し元: CLI の `package` 工程（`limits.max_infer_p95_us` があるとき。#338）・#128 のレポート生成。
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

    let total = inputs
        .iter()
        .fold(0usize, |acc, x| acc.saturating_add(x.len()));
    if total > MAX_INFER_BATCH_TOTAL_BYTES {
        return Err(LatencyError::TotalInputTooLarge {
            total,
            limit: MAX_INFER_BATCH_TOTAL_BYTES,
        });
    }

    let start = clock.now_ns();
    // 直前に観測した時刻。反復間・検査間をまたいで単調性を検査するために保持する（REQ-39）。
    let last = std::cell::Cell::new(start);
    // 時計を読み、直前の観測より戻っていれば NonMonotonicClock で失敗させる。
    let read_clock = |phase: LatencyPhase, iteration: usize| {
        let now = clock.now_ns();
        if now < last.get() {
            return Err(LatencyError::NonMonotonicClock { phase, iteration });
        }
        last.set(now);
        Ok(now)
    };
    // 各反復の開始前に計測全体の期限を検査する。開始時刻との差も checked_sub で検査する。
    let check_deadline = |phase: LatencyPhase, iteration: usize| {
        let now = read_clock(phase, iteration)?;
        let since_start = now
            .checked_sub(start)
            .ok_or(LatencyError::NonMonotonicClock { phase, iteration })?;
        if since_start > config.total_timeout_ns {
            Err(LatencyError::DeadlineExceeded {
                phase,
                iteration,
                limit_ns: config.total_timeout_ns,
            })
        } else {
            Ok(())
        }
    };

    for (iteration, input) in inputs.iter().cycle().take(config.warmup).enumerate() {
        check_deadline(LatencyPhase::Warmup, iteration)?;
        let t0 = read_clock(LatencyPhase::Warmup, iteration)?;
        let result = predict(black_box(input));
        let t1 = read_clock(LatencyPhase::Warmup, iteration)?;
        match result {
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
        // warmup でも時計の逆行を 0 ns 扱いにせず、計測段階と同様に失敗させる（REQ-39）。
        let warmup_elapsed = t1.checked_sub(t0).ok_or(LatencyError::NonMonotonicClock {
            phase: LatencyPhase::Warmup,
            iteration,
        })?;
        if warmup_elapsed > config.per_infer_timeout_ns {
            return Err(LatencyError::InferenceTimeout {
                phase: LatencyPhase::Warmup,
                iteration,
                limit_ns: config.per_infer_timeout_ns,
            });
        }
    }

    let mut samples_ns = Vec::with_capacity(config.iters);
    for (iteration, input) in inputs.iter().cycle().take(config.iters).enumerate() {
        check_deadline(LatencyPhase::Measure, iteration)?;
        let t0 = read_clock(LatencyPhase::Measure, iteration)?;
        let result = black_box(predict(black_box(input)));
        let t1 = read_clock(LatencyPhase::Measure, iteration)?;
        if let Err(e) = result {
            return Err(LatencyError::Inference {
                phase: LatencyPhase::Measure,
                iteration,
                code: e.code(),
            });
        }
        let elapsed = t1.checked_sub(t0).ok_or(LatencyError::NonMonotonicClock {
            phase: LatencyPhase::Measure,
            iteration,
        })?;
        if elapsed > config.per_infer_timeout_ns {
            return Err(LatencyError::InferenceTimeout {
                phase: LatencyPhase::Measure,
                iteration,
                limit_ns: config.per_infer_timeout_ns,
            });
        }
        samples_ns.push(elapsed);
    }

    // 最後の推論が全体期限を超えて終わった場合も成功として返さない（fail-closed。REQ-39）。
    check_deadline(LatencyPhase::Measure, config.iters)?;

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
                LatencyError::TotalInputTooLarge { total: 2, limit: 1 },
                "total_input_too_large",
            ),
            (
                LatencyError::InvalidTimeout { timeout_ns: 0 },
                "invalid_timeout",
            ),
            (
                LatencyError::DeadlineExceeded {
                    phase: LatencyPhase::Measure,
                    iteration: 0,
                    limit_ns: 1,
                },
                "deadline_exceeded",
            ),
            (
                LatencyError::InferenceTimeout {
                    phase: LatencyPhase::Measure,
                    iteration: 0,
                    limit_ns: 1,
                },
                "inference_timeout",
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
                LatencyError::NonMonotonicClock {
                    phase: LatencyPhase::Measure,
                    iteration: 0,
                },
                "non_monotonic_clock",
            ),
        ];
        for (e, code) in cases {
            assert_eq!(e.code(), code);
        }
    }

    use std::cell::Cell;

    /// 呼ばれるたびに `step` ns 進む偽の時計。
    struct StepClock {
        now: Cell<u64>,
        step: u64,
    }

    impl Clock for StepClock {
        fn now_ns(&self) -> u64 {
            let v = self.now.get();
            self.now.set(v + self.step);
            v
        }
    }

    struct StubPre;
    impl Preprocessor for StubPre {
        fn preprocess(
            &self,
            _input: &str,
        ) -> Result<crate::pipeline::TokenIds, crate::pipeline::PreprocessError> {
            Ok(crate::pipeline::TokenIds::new(vec![1]))
        }
    }

    struct StubBackend;
    impl ScoringBackend for StubBackend {
        /// テスト用スタブ: 計算は即時で打ち切り対象の反復を持たないため、`scores` と同じ結果を返す。
        fn scores_limited(
            &self,
            ids: &crate::pipeline::TokenIds,
            _limit: std::time::Duration,
        ) -> Result<Vec<f64>, crate::pipeline::BackendError> {
            self.scores(ids)
        }
        fn scores(
            &self,
            _ids: &crate::pipeline::TokenIds,
        ) -> Result<Vec<f64>, crate::pipeline::BackendError> {
            Ok(vec![0.75, 0.25])
        }
    }

    fn ok_prediction() -> Result<Prediction, InferError> {
        InferencePipeline::new(StubPre, StubBackend).infer_one("x")
    }

    #[test]
    fn req39_total_input_bytes_checked_before_inference() {
        let big = "a".repeat(1024 * 1024);
        let inputs: Vec<&str> = (0..65).map(|_| big.as_str()).collect();
        let mut called = false;
        let clock = StepClock {
            now: Cell::new(0),
            step: 1,
        };
        let r = measure_with(
            |_| {
                called = true;
                ok_prediction()
            },
            &inputs,
            &LatencyConfig::default(),
            &clock,
        );
        assert_eq!(
            r.unwrap_err(),
            LatencyError::TotalInputTooLarge {
                total: 65 * 1024 * 1024,
                limit: MAX_INFER_BATCH_TOTAL_BYTES
            }
        );
        assert!(!called);
    }

    #[test]
    fn req39_timeouts_validated() {
        let c = LatencyConfig::default();
        assert_eq!(
            c.with_timeouts(0, 1),
            Err(LatencyError::InvalidTimeout { timeout_ns: 0 })
        );
        assert_eq!(
            c.with_timeouts(1, MAX_LATENCY_TIMEOUT_NS + 1),
            Err(LatencyError::InvalidTimeout {
                timeout_ns: MAX_LATENCY_TIMEOUT_NS + 1
            })
        );
        assert!(c.with_timeouts(MAX_LATENCY_TIMEOUT_NS, 1).is_ok());
    }

    #[test]
    fn req39_total_deadline_aborts_measurement() {
        // now_ns 1 回ごとに 10 ns 進む。各反復は 2 回呼ぶので 1 反復 20 ns。全体上限 50 ns。
        let clock = StepClock {
            now: Cell::new(0),
            step: 10,
        };
        let cfg = LatencyConfig::new(0, 1000)
            .unwrap()
            .with_timeouts(50, 1_000)
            .unwrap();
        let err = measure_with(|_| ok_prediction(), &["x"], &cfg, &clock).unwrap_err();
        assert!(matches!(
            err,
            LatencyError::DeadlineExceeded {
                phase: LatencyPhase::Measure,
                limit_ns: 50,
                ..
            }
        ));
    }

    #[test]
    fn req39_per_inference_timeout_in_measure_and_warmup() {
        // 反復ごとに 10 ns かかる（t0 と t1 の差）。1 回上限 5 ns を超える。
        let cfg_measure = LatencyConfig::new(0, 3)
            .unwrap()
            .with_timeouts(1_000_000, 5)
            .unwrap();
        let clock = StepClock {
            now: Cell::new(0),
            step: 10,
        };
        assert_eq!(
            measure_with(|_| ok_prediction(), &["x"], &cfg_measure, &clock).unwrap_err(),
            LatencyError::InferenceTimeout {
                phase: LatencyPhase::Measure,
                iteration: 0,
                limit_ns: 5
            }
        );
        let cfg_warm = LatencyConfig::new(3, 1)
            .unwrap()
            .with_timeouts(1_000_000, 5)
            .unwrap();
        let clock = StepClock {
            now: Cell::new(0),
            step: 10,
        };
        assert_eq!(
            measure_with(|_| ok_prediction(), &["x"], &cfg_warm, &clock).unwrap_err(),
            LatencyError::InferenceTimeout {
                phase: LatencyPhase::Warmup,
                iteration: 0,
                limit_ns: 5
            }
        );
    }

    #[test]
    fn req39_total_deadline_checked_after_last_iteration() {
        // 1 反復 = t0,t1 の 2 回 + 開始前検査 1 回 + start 1 回。iters=1 で最後の検査のみが超過する。
        let clock = StepClock {
            now: Cell::new(0),
            step: 10,
        };
        // start=0, check(10), t0=20, t1=30, 最終 check=40 → 40 > 35。開始前検査 10 は超えない。
        let cfg = LatencyConfig::new(0, 1)
            .unwrap()
            .with_timeouts(35, 1_000)
            .unwrap();
        let err = measure_with(|_| ok_prediction(), &["x"], &cfg, &clock).unwrap_err();
        assert_eq!(
            err,
            LatencyError::DeadlineExceeded {
                phase: LatencyPhase::Measure,
                iteration: 1,
                limit_ns: 35
            }
        );
    }

    #[test]
    fn req39_warmup_clock_regression_is_error() {
        struct BackClock(Cell<u64>);
        impl Clock for BackClock {
            fn now_ns(&self) -> u64 {
                let v = self.0.get();
                self.0.set(v.saturating_sub(1));
                v
            }
        }
        let cfg = LatencyConfig::new(1, 1).unwrap();
        let err = measure_with(
            |_| ok_prediction(),
            &["x"],
            &cfg,
            &BackClock(Cell::new(1_000)),
        )
        .unwrap_err();
        assert_eq!(
            err,
            LatencyError::NonMonotonicClock {
                phase: LatencyPhase::Warmup,
                iteration: 0
            }
        );
    }

    #[test]
    fn req39_clock_regression_between_iterations_is_error() {
        struct SeqClock(Cell<usize>);
        impl Clock for SeqClock {
            fn now_ns(&self) -> u64 {
                let seq = [100u64, 110, 120, 130, 50, 60, 70];
                let i = self.0.get();
                self.0.set(i + 1);
                seq.get(i).copied().unwrap_or(1_000)
            }
        }
        // start=100, 検査 110, t0=120, t1=130, 次反復の検査 50（反復間で逆行）。
        let cfg = LatencyConfig::new(0, 2).unwrap();
        let err =
            measure_with(|_| ok_prediction(), &["x"], &cfg, &SeqClock(Cell::new(0))).unwrap_err();
        assert_eq!(
            err,
            LatencyError::NonMonotonicClock {
                phase: LatencyPhase::Measure,
                iteration: 1
            }
        );
    }

    #[test]
    fn req39_warmup_deadline_exceeded() {
        let clock = StepClock {
            now: Cell::new(0),
            step: 100,
        };
        let cfg = LatencyConfig::new(1000, 1)
            .unwrap()
            .with_timeouts(50, 1_000_000)
            .unwrap();
        // 最初の check_deadline で start から 100 ns 経過し 50 ns 上限を超える。
        let err = measure_with(|_| ok_prediction(), &["x"], &cfg, &clock).unwrap_err();
        assert_eq!(
            err,
            LatencyError::DeadlineExceeded {
                phase: LatencyPhase::Warmup,
                iteration: 0,
                limit_ns: 50
            }
        );
    }

    #[test]
    fn monotonic_clock_is_non_decreasing() {
        let c = MonotonicClock::new();
        let a = c.now_ns();
        let b = c.now_ns();
        assert!(b >= a);
    }
}
