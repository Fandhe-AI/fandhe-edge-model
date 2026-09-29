//! package 工程の終了コード決定の公開 API 結合テスト（REQ-21・TASK-21.3-1・#132）。
//!
//! 上限との照合（TASK-30.2・#124）は未マージのため、超過は入力として与える。
//! `measure_package` からの end-to-end 確認は #124 のマージ後に行う。

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
