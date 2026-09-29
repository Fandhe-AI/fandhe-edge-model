//! 待ち時間上限超過 → `limit_exceeded` の結合テスト（REQ-31・REQ-21・TASK-31.2・#129）。
//! 証拠種別: テストハーネス（偽の時計・模擬バックエンド）。実機計測ではない。
//! 本番データでの `limit_exceeded` 再実演は未実施。

use fandhe_edge_runtime::latency::*;
use fandhe_edge_runtime::latency_limit::*;
use fandhe_edge_runtime::latency_report::*;
use fandhe_edge_runtime::package_outcome::*;
use fandhe_edge_runtime::pipeline::*;
use std::cell::Cell;
use std::rc::Rc;

#[derive(Clone)]
struct FakeClock(Rc<Cell<u64>>);
impl Clock for FakeClock {
    fn now_ns(&self) -> u64 {
        self.0.get()
    }
}

struct Pre;
impl Preprocessor for Pre {
    fn preprocess(&self, _input: &str) -> Result<TokenIds, PreprocessError> {
        Ok(TokenIds::new(vec![1]))
    }
}

/// 呼び出し番号 k ごとに時計を `base + step * k` 進める。
struct TickBackend {
    clock: Rc<Cell<u64>>,
    calls: Cell<u64>,
    base: u64,
    step: u64,
}
impl ScoringBackend for TickBackend {
    fn scores(&self, _ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        let k = self.calls.get();
        self.calls.set(k + 1);
        self.clock.set(self.clock.get() + self.base + self.step * k);
        Ok(vec![0.75, 0.25])
    }

    fn scores_limited(
        &self,
        ids: &TokenIds,
        _limit: std::time::Duration,
    ) -> Result<Vec<f64>, BackendError> {
        // このテストは時間上限を使わないため、計算は scores と同一にする。
        self.scores(ids)
    }
}

fn report(base: u64, step: u64) -> LatencyReport {
    let cell = Rc::new(Cell::new(0));
    let pipeline = InferencePipeline::new(
        Pre,
        TickBackend {
            clock: cell.clone(),
            calls: Cell::new(0),
            base,
            step,
        },
    );
    let cfg = LatencyConfig::new(0, 100).unwrap();
    let s = measure_latency(&pipeline, &["a"], &cfg, &FakeClock(cell)).unwrap();
    summarize_latency(&s).unwrap()
}

fn limit(ns: u64) -> Option<LatencyLimit> {
    Some(LatencyLimit::from_ns(ns).unwrap())
}

fn code(check: &LatencyLimitCheck, q: PackageQualityJudgment) -> u8 {
    resolve_package_outcome(check.breach().as_slice(), q)
        .exit_code
        .code()
}

#[test]
fn req31_above_reference_but_within_user_limit_is_ok() {
    // p95 300ms は参考値 250ms を上回るが、利用者上限 500ms 以内なので合格。
    let r = report(300_000_000, 0);
    let c = check_latency_limit(&r, limit(500_000_000));
    assert_eq!(
        c,
        LatencyLimitCheck::Within {
            p95_ns: 300_000_000,
            limit_ns: 500_000_000
        }
    );
    assert_eq!(code(&c, PackageQualityJudgment::Pass), 0);
}

#[test]
fn req31_below_reference_but_over_user_limit_is_limit_exceeded() {
    // p95 100ms は参考値未満だが、利用者上限 50ms を超えるので 20。
    let r = report(100_000_000, 0);
    let c = check_latency_limit(&r, limit(50_000_000));
    assert_eq!(
        c,
        LatencyLimitCheck::Exceeded(LimitBreach::Latency {
            measured_p95_ns: 100_000_000,
            limit_ns: 50_000_000
        })
    );
    assert_eq!(code(&c, PackageQualityJudgment::Pass), 20);
}

#[test]
fn req31_ceiled_p95_is_used_against_limit() {
    // 厳密 p95 は 194.05 で切り上げて 195。上限 194 は超過、195 は合格。
    let r = report(100, 1);
    assert_eq!(
        check_latency_limit(&r, limit(194)).breach(),
        Some(LimitBreach::Latency {
            measured_p95_ns: 195,
            limit_ns: 194
        })
    );
    assert_eq!(
        check_latency_limit(&r, limit(195)),
        LatencyLimitCheck::Within {
            p95_ns: 195,
            limit_ns: 195
        }
    );
}

#[test]
fn req21_limit_exceeded_takes_priority_over_quality() {
    let r = report(100, 0);
    let c = check_latency_limit(&r, limit(50));
    for q in [
        PackageQualityJudgment::Pass,
        PackageQualityJudgment::Fail,
        PackageQualityJudgment::Undeterminable,
    ] {
        let o = resolve_package_outcome(c.breach().as_slice(), q);
        assert_eq!(o.exit_code.code(), 20);
        assert_eq!(o.verdict, PackageVerdict::LimitExceeded);
    }
}

#[test]
fn req31_not_configured_defers_to_quality_judgment() {
    let r = report(100, 0);
    let c = check_latency_limit(&r, None);
    assert_eq!(c, LatencyLimitCheck::NotConfigured { p95_ns: 100 });
    assert_eq!(code(&c, PackageQualityJudgment::Fail), 10);
}

#[test]
fn req31_invalid_limit_is_invalid_input() {
    let e = LatencyLimit::from_ns(0).unwrap_err();
    assert_eq!(e.exit_code().code(), 64);
}
