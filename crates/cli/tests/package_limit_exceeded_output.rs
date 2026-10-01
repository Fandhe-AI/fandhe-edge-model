//! 上限超過の終了コード決定から JSON 出力までの結合テスト
//! （REQ-21・REQ-33・REQ-30・REQ-31・TASK-21.3・#131）。
//!
//! 超過は実照合関数（`check_capacity_limit`・`LimitBreach::latency_if_exceeded`）から得て、
//! `resolve_package_outcome` → `emit_package_outcome` を通し、stdout が 1 JSON 1 行で
//! パス・計測値を含まず、終了コードが合否判定に依らず 20 になることを固定する。
//!
//! 証拠種別: テストハーネス（合成サイズ）。本番データでの `limit_exceeded` の再実演は未実施
//! （PoC-16 で本番データ確認済みは `ok`・`judged_fail` のみ）。

use fandhe_edge_cli::stage_output::emit_package_outcome;
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_runtime::capacity::{CapacityBreakdown, PackageComponent};
use fandhe_edge_runtime::capacity_limit::{CapacityLimit, check_capacity_limit};
use fandhe_edge_runtime::package_outcome::{
    LimitBreach, PackageQualityJudgment, resolve_package_outcome,
};

const QUALITIES: [PackageQualityJudgment; 4] = [
    PackageQualityJudgment::Pass,
    PackageQualityJudgment::Fail,
    PackageQualityJudgment::Undeterminable,
    PackageQualityJudgment::NotDefined,
];

fn cap_breach(limit: u64) -> Vec<LimitBreach> {
    let b = CapacityBreakdown::from_sizes([(PackageComponent::Weights, 1549)]).unwrap();
    check_capacity_limit(&b, Some(CapacityLimit::from_bytes(limit).unwrap()))
        .breach()
        .into_iter()
        .collect()
}

fn lat_breach(limit: u64) -> Vec<LimitBreach> {
    LimitBreach::latency_if_exceeded(100, limit)
        .into_iter()
        .collect()
}

fn emit(b: &[LimitBreach], q: PackageQualityJudgment) -> (ExitCode, String) {
    let mut out = Vec::new();
    let code = emit_package_outcome(&mut out, &resolve_package_outcome(b, q)).unwrap();
    (code, String::from_utf8(out).unwrap())
}

const LIMIT_JSON: &str = "{\"code\":\"limit_exceeded\",\"message\":\"resource limit exceeded\"}\n";
// exit 10・12 は合否基準が定義されているときの判定項目つきの形（#328）。
const FAIL_JSON: &str = "{\"code\":\"judged_fail\",\"message\":\"judged as fail\",\"step\":\"package\",\"judgment\":\"fail\",\"acceptance_defined\":true}\n";
const PENDING_JSON: &str = "{\"code\":\"pending\",\"message\":\"result is pending\",\"step\":\"package\",\"judgment\":\"undeterminable\",\"acceptance_defined\":true}\n";
const PASS_JSON: &str =
    "{\"step\":\"package\",\"status\":\"ok\",\"judgment\":\"pass\",\"acceptance_defined\":true}\n";
const NOT_DEFINED_JSON: &str =
    "{\"step\":\"package\",\"status\":\"ok\",\"judgment\":null,\"acceptance_defined\":false}\n";

#[test]
fn req21_breach_outputs_limit_exceeded_json_for_every_judgment() {
    let mut both = cap_breach(1548);
    both.extend(lat_breach(99));
    assert_eq!(both.len(), 2);
    for b in [cap_breach(1548), lat_breach(99), both] {
        assert!(!b.is_empty());
        for q in QUALITIES {
            let (code, out) = emit(&b, q);
            assert_eq!(code, ExitCode::LimitExceeded);
            assert_eq!(code.code(), 20);
            assert_eq!(out, LIMIT_JSON);
        }
    }
}

#[test]
fn req21_equal_to_limit_keeps_quality_outputs() {
    let mut b = cap_breach(1549);
    b.extend(lat_breach(100));
    assert!(b.is_empty());
    let (code, out) = emit(&b, PackageQualityJudgment::Fail);
    assert_eq!((code.code(), out.as_str()), (10, FAIL_JSON));
    let (code, out) = emit(&b, PackageQualityJudgment::Undeterminable);
    assert_eq!((code.code(), out.as_str()), (12, PENDING_JSON));
    let (code, out) = emit(&b, PackageQualityJudgment::Pass);
    assert_eq!((code.code(), out.as_str()), (0, PASS_JSON));
    let (code, out) = emit(&b, PackageQualityJudgment::NotDefined);
    assert_eq!((code.code(), out.as_str()), (0, NOT_DEFINED_JSON));
}
