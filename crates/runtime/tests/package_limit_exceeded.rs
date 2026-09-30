//! package 工程の終了コード決定の公開 API 結合テスト（REQ-21・REQ-31・TASK-21.3-1・#132・TASK-21.3-2・#133）。
//!
//! 容量の境界規則は `LimitBreach::capacity_if_exceeded`（TASK-30.2・#124）経由で確認する。
//! `measure_package` からの end-to-end 確認は `tests/capacity_limit_exceeded.rs`。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_runtime::package_outcome::{
    LimitBreach, PackageQualityJudgment, resolve_package_outcome,
};

#[test]
fn req21_limit_exceeded_is_20_regardless_of_judgment() {
    let breach = [LimitBreach::Capacity {
        measured_bytes: 41,
        limit_bytes: 40,
    }];
    let fail = resolve_package_outcome(&breach, PackageQualityJudgment::Fail);
    let pass = resolve_package_outcome(&breach, PackageQualityJudgment::Pass);
    assert_eq!(fail.exit_code, ExitCode::LimitExceeded);
    assert_eq!(pass.exit_code, ExitCode::LimitExceeded);
}

#[test]
fn req21_within_limit_maps_fail_to_10_and_pass_to_0() {
    let fail = resolve_package_outcome(&[], PackageQualityJudgment::Fail);
    let pass = resolve_package_outcome(&[], PackageQualityJudgment::Pass);
    assert_eq!(fail.exit_code.code(), 10);
    assert_eq!(pass.exit_code.code(), 0);
}

#[test]
fn req31_latency_over_limit_is_20_even_when_judged_pass() {
    let breach: Vec<LimitBreach> = LimitBreach::latency_if_exceeded(250_000_001, 250_000_000)
        .into_iter()
        .collect();
    let o = resolve_package_outcome(&breach, PackageQualityJudgment::Pass);
    assert_eq!(o.exit_code, ExitCode::LimitExceeded);
    assert_eq!(o.exit_code.code(), 20);
}

#[test]
fn req31_latency_equal_to_limit_is_not_limit_exceeded() {
    let breach: Vec<LimitBreach> = LimitBreach::latency_if_exceeded(250_000_000, 250_000_000)
        .into_iter()
        .collect();
    assert!(breach.is_empty());
    assert_eq!(
        resolve_package_outcome(&breach, PackageQualityJudgment::Fail)
            .exit_code
            .code(),
        10
    );
    assert_eq!(
        resolve_package_outcome(&breach, PackageQualityJudgment::Pass)
            .exit_code
            .code(),
        0
    );
}

/// REQ-30・REQ-21: 上限ちょうどは超過でなく品質判定の終了コードを維持し、+1 は 20。
#[test]
fn req30_capacity_boundary_via_capacity_if_exceeded() {
    let at = LimitBreach::capacity_if_exceeded(1549, 1549)
        .into_iter()
        .collect::<Vec<_>>();
    let over = LimitBreach::capacity_if_exceeded(1550, 1549)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(
        resolve_package_outcome(&at, PackageQualityJudgment::Fail)
            .exit_code
            .code(),
        10
    );
    assert_eq!(
        resolve_package_outcome(&at, PackageQualityJudgment::NotDefined)
            .exit_code
            .code(),
        0
    );
    assert_eq!(
        resolve_package_outcome(&over, PackageQualityJudgment::Pass)
            .exit_code
            .code(),
        20
    );
}
