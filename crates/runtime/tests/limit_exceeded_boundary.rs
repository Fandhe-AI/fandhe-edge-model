//! 容量・待ち時間の上限超過が合否判定に依らず `limit_exceeded`（20）になる合流点テスト
//! （REQ-21 境界値・REQ-30・REQ-31・TASK-21.3・#131。親 issue のクローズアウト確認）。
//!
//! 容量（`CapacityBreakdown` → `check_capacity_limit`）と待ち時間（偽の時計での計測 →
//! `check_latency_limit`）の実照合関数の結果を並べて `resolve_package_outcome` へ渡し、
//! 4 判定（Pass・Fail・判定不能・基準未設定）すべてで終了コードを具体値で固定する。
//! 境界規則は `LimitBreach` の 1 箇所にあり、本テストは再実装しない。
//!
//! 証拠種別: テストハーネス（合成サイズ・偽の時計と模擬バックエンド）。
//! 本番データでの `limit_exceeded` の再実演は未実施（PoC-16 で本番データ確認済みは
//! `ok`・`judged_fail` のみ）。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_runtime::capacity::{CapacityBreakdown, PackageComponent};
use fandhe_edge_runtime::capacity_limit::{CapacityLimit, check_capacity_limit};
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

/// 1 回の推論ごとに時計を一定時間進める模擬バックエンド。
struct TickBackend {
    clock: Rc<Cell<u64>>,
    step: u64,
}
impl ScoringBackend for TickBackend {
    fn scores(&self, _ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        self.clock.set(self.clock.get() + self.step);
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

/// 各回ちょうど `step` ns の計測から報告を作る（p95 = `step`）。
fn report(step: u64, iters: usize) -> LatencyReport {
    let cell = Rc::new(Cell::new(0));
    let pipeline = InferencePipeline::new(
        Pre,
        TickBackend {
            clock: cell.clone(),
            step,
        },
    );
    let cfg = LatencyConfig::new(0, iters).unwrap();
    let s = measure_latency(&pipeline, &["a"], &cfg, &FakeClock(cell)).unwrap();
    summarize_latency(&s).unwrap()
}

/// 合計 1549 バイト（1000 + 300 + 249）の容量内訳。
fn breakdown() -> CapacityBreakdown {
    CapacityBreakdown::from_sizes([
        (PackageComponent::Weights, 1000),
        (PackageComponent::LabelTable, 300),
        (PackageComponent::Metadata, 249),
    ])
    .unwrap()
}

const QUALITIES: [PackageQualityJudgment; 4] = [
    PackageQualityJudgment::Pass,
    PackageQualityJudgment::Fail,
    PackageQualityJudgment::Undeterminable,
    PackageQualityJudgment::NotDefined,
];

/// 容量 → 待ち時間の順に実照合関数から超過を集める。
fn breaches(cap: Option<u64>, lat: Option<u64>) -> Vec<LimitBreach> {
    let c = check_capacity_limit(
        &breakdown(),
        cap.map(|b| CapacityLimit::from_bytes(b).unwrap()),
    );
    let l = check_latency_limit(
        &report(100, 10),
        lat.map(|n| LatencyLimit::from_ns(n).unwrap()),
    );
    c.breach().into_iter().chain(l.breach()).collect()
}

fn codes(b: &[LimitBreach]) -> Vec<u8> {
    QUALITIES
        .iter()
        .map(|q| resolve_package_outcome(b, *q).exit_code.code())
        .collect()
}

const CAP: LimitBreach = LimitBreach::Capacity {
    measured_bytes: 1549,
    limit_bytes: 1548,
};
const LAT: LimitBreach = LimitBreach::Latency {
    measured_p95_ns: 100,
    limit_ns: 99,
};

fn assert_all_limit_exceeded(b: &[LimitBreach]) {
    for q in QUALITIES {
        let o = resolve_package_outcome(b, q);
        assert_eq!(o.exit_code, ExitCode::LimitExceeded);
        assert_eq!(o.exit_code.code(), 20);
        assert_eq!(o.verdict, PackageVerdict::LimitExceeded);
    }
    assert_eq!(codes(b), vec![20, 20, 20, 20]);
}

#[test]
fn req21_capacity_only_breach_is_20_for_every_judgment() {
    let b = breaches(Some(1548), Some(100));
    assert_eq!(b, vec![CAP]);
    assert_all_limit_exceeded(&b);
}

#[test]
fn req21_latency_only_breach_is_20_for_every_judgment() {
    let b = breaches(Some(1549), Some(99));
    assert_eq!(b, vec![LAT]);
    assert_all_limit_exceeded(&b);
}

#[test]
fn req21_both_breach_is_20_and_keeps_order() {
    let b = breaches(Some(1548), Some(99));
    assert_eq!(b, vec![CAP, LAT]);
    assert_all_limit_exceeded(&b);
    let o = resolve_package_outcome(&b, PackageQualityJudgment::Pass);
    assert_eq!(o.breaches, vec![CAP, LAT]);
}

#[test]
fn req21_both_equal_to_limit_keep_quality_codes() {
    let b = breaches(Some(1549), Some(100));
    assert!(b.is_empty());
    assert_eq!(codes(&b), vec![0, 10, 12, 0]);
}

#[test]
fn req21_no_limits_configured_keep_quality_codes() {
    let b = breaches(None, None);
    assert!(b.is_empty());
    assert_eq!(codes(&b), vec![0, 10, 12, 0]);
}
