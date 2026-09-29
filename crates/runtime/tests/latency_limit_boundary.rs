//! 待ち時間上限ちょうどの境界判定の結合テスト（REQ-31・REQ-21・TASK-31.3・#130）。
//!
//! 計測経路（`InferencePipeline` → `measure_latency` → `summarize_latency` → `p95_ns()`）から
//! `LimitBreach::latency_if_exceeded` → `resolve_package_outcome` までを公開 API だけで通し、
//! p95 が利用者の上限と等しいときは合格、超えたときだけ `limit_exceeded`（20）になることを固定する。
//! 境界規則は `LimitBreach::latency_if_exceeded` の 1 箇所に集約されており、本テストは再実装せず
//! 結果を照合する。
//!
//! 証拠種別: テストハーネス（偽の時計・模擬バックエンド）。実機計測ではなく PoC 実測もない。
//! 上限ちょうどの扱いは PoC-14 で未確認。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_runtime::latency::*;
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
}

fn report(base: u64, step: u64, iters: usize) -> LatencyReport {
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
    let cfg = LatencyConfig::new(0, iters).unwrap();
    let s = measure_latency(&pipeline, &["a"], &cfg, &FakeClock(cell)).unwrap();
    summarize_latency(&s).unwrap()
}

fn breaches(r: &LatencyReport, limit_ns: u64) -> Vec<LimitBreach> {
    LimitBreach::latency_if_exceeded(r.p95_ns(), limit_ns)
        .into_iter()
        .collect()
}

const QUALITIES: [PackageQualityJudgment; 4] = [
    PackageQualityJudgment::Pass,
    PackageQualityJudgment::Fail,
    PackageQualityJudgment::Undeterminable,
    PackageQualityJudgment::NotDefined,
];

#[test]
fn req31_p95_exactly_equal_to_user_limit_is_not_limit_exceeded() {
    let r = report(195, 0, 100);
    assert_eq!(r.p95_ns(), 195);
    assert_eq!(r.p95().frac_hundredths(), 0);
    let b = breaches(&r, 195);
    assert!(b.is_empty());
    // 上限ちょうどは品質判定へ委ねる（0・10・12・0）。
    let codes: Vec<u8> = QUALITIES
        .iter()
        .map(|q| resolve_package_outcome(&b, *q).exit_code.code())
        .collect();
    assert_eq!(codes, vec![0, 10, 12, 0]);
}

#[test]
fn req31_p95_one_ns_over_user_limit_is_limit_exceeded() {
    let r = report(195, 0, 100);
    let b = breaches(&r, 194);
    assert_eq!(
        b,
        vec![LimitBreach::Latency {
            measured_p95_ns: 195,
            limit_ns: 194
        }]
    );
    for q in QUALITIES {
        let o = resolve_package_outcome(&b, q);
        assert_eq!(o.exit_code, ExitCode::LimitExceeded);
        assert_eq!(o.exit_code.code(), 20);
        assert_eq!(o.verdict, PackageVerdict::LimitExceeded);
    }
}

#[test]
fn req31_p95_one_ns_below_user_limit_is_pass() {
    let r = report(195, 0, 100);
    let b = breaches(&r, 196);
    assert!(b.is_empty());
    let o = resolve_package_outcome(&b, PackageQualityJudgment::Pass);
    assert_eq!(o.exit_code.code(), 0);
    assert_eq!(o.verdict, PackageVerdict::Pass);
}

#[test]
fn req31_fractional_p95_just_above_limit_is_fail_closed() {
    // 計測値 [1000, 1001]。厳密 p95 は 1000.95 で、切り上げて 1001。
    let r = report(1_000, 1, 2);
    assert_eq!(r.p95().floor_ns(), 1_000);
    assert_eq!(r.p95().frac_hundredths(), 95);
    assert_eq!(r.p95_ns(), 1_001);
    let b = breaches(&r, 1_000);
    assert_eq!(
        b,
        vec![LimitBreach::Latency {
            measured_p95_ns: 1_001,
            limit_ns: 1_000
        }]
    );
    assert_eq!(
        resolve_package_outcome(&b, PackageQualityJudgment::Pass)
            .exit_code
            .code(),
        20
    );
    // 上限 1001 なら p95（1001）と等しいので合格。
    assert!(breaches(&r, 1_001).is_empty());
}

#[test]
fn req31_equal_boundary_uses_user_limit_not_reference() {
    // 参考値 250ms を超える p95 でも、利用者の上限と等しければ合格。
    let over_ref = report(300_000_000, 0, 100);
    assert_eq!(
        over_ref.reference_comparison(),
        ReferenceComparison::AtOrAbove
    );
    let b = breaches(&over_ref, 300_000_000);
    assert!(b.is_empty());
    assert_eq!(
        resolve_package_outcome(&b, PackageQualityJudgment::Pass)
            .exit_code
            .code(),
        0
    );
    // 参考値ちょうどの p95 も、合否は上限との比較だけで決まる。
    let at_ref = report(250_000_000, 0, 100);
    assert_eq!(at_ref.p95_ns(), 250_000_000);
    assert_eq!(
        at_ref.reference_comparison(),
        ReferenceComparison::AtOrAbove
    );
    assert!(breaches(&at_ref, 250_000_000).is_empty());
    assert_eq!(breaches(&at_ref, 249_999_999).len(), 1);
}
