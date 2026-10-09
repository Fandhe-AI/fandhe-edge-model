//! 上限超過の終了コード決定から JSON 出力までの結合テスト
//! （REQ-21・REQ-33・REQ-30・REQ-31・TASK-21.3・#131・#340）。
//!
//! 超過は実照合関数（`check_capacity_limit`・`LimitBreach::latency_if_exceeded`）から得て、
//! `resolve_package_outcome` → `emit_package_outcome` を通し、stdout が 1 JSON 1 行で
//! パスを含まず、終了コードが合否判定に依らず 20 になることを固定する。計測値（`capacity`・
//! `infer_p95`）の `exceeded` で、どちらの上限を超えたかを区別できる（#340）。
//!
//! 証拠種別: テストハーネス（合成サイズ）。本番データでの `limit_exceeded` の再実演は未実施
//! （PoC-16 で本番データ確認済みは `ok`・`judged_fail` のみ）。

use fandhe_edge_cli::stage_output::emit_package_outcome;
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::stage_report::{
    InferP95, PackageCapacity, PackageCapacityComponents, PackageComponentSize, PackageMetrics,
};
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

/// p95 100 ns に対する上限 `limit_ns` の超過。
fn lat_breach(limit_ns: u64) -> Vec<LimitBreach> {
    LimitBreach::latency_if_exceeded(100, limit_ns)
        .into_iter()
        .collect()
}

/// 合成の計測値（重み 1549 バイトのみ）。`exceeded` は照合結果（超過の有無）から渡す。
fn metrics(limit_bytes: u64, cap_exceeded: bool, p95: Option<(u64, bool)>) -> PackageMetrics {
    let c = PackageComponentSize::new;
    PackageMetrics {
        capacity: PackageCapacity::new(
            1549,
            Some(limit_bytes),
            cap_exceeded,
            (40_000_000, false),
            PackageCapacityComponents::new(c(1549, 1), c(0, 0), c(0, 0), c(0, 0), c(0, 0)),
        ),
        infer_p95: p95.map(|(limit_us, e)| InferP95::new(1, limit_us, e)),
    }
}

fn emit(b: &[LimitBreach], q: PackageQualityJudgment, m: &PackageMetrics) -> (ExitCode, String) {
    let mut out = Vec::new();
    let code = emit_package_outcome(&mut out, &resolve_package_outcome(b, q), m).unwrap();
    (code, String::from_utf8(out).unwrap())
}

const COMPONENTS: &str = "\"components\":{\"weights\":{\"bytes\":1549,\"file_count\":1},\"vocab_or_feature_transform\":{\"bytes\":0,\"file_count\":0},\"label_table\":{\"bytes\":0,\"file_count\":0},\"calibration\":{\"bytes\":0,\"file_count\":0},\"metadata\":{\"bytes\":0,\"file_count\":0}}";

fn cap_json(limit: u64, exceeded: bool) -> String {
    format!(
        "\"capacity\":{{\"total_bytes\":1549,\"limit_bytes\":{limit},\"exceeded\":{exceeded},\"guideline_bytes\":40000000,\"over_guideline\":false,{COMPONENTS}}}"
    )
}

fn limit_json(cap: &str, p95: &str) -> String {
    format!(
        "{{\"code\":\"limit_exceeded\",\"message\":\"resource limit exceeded\",\"step\":\"package\",{cap},\"infer_p95\":{p95}}}\n"
    )
}

const P95_NULL: &str = "null";
const P95_OVER: &str = "{\"p95_us\":1,\"limit_us\":0,\"exceeded\":true}";
const P95_WITHIN: &str = "{\"p95_us\":1,\"limit_us\":1,\"exceeded\":false}";

/// 超過の種類（容量だけ・p95 だけ・両方）が各 `exceeded` の組で区別できる。
#[test]
fn req21_issue340_breach_outputs_distinguish_which_limit_was_exceeded() {
    // 容量だけ。p95 の上限は未設定（null）。
    let (code, out) = emit(
        &cap_breach(1548),
        PackageQualityJudgment::Pass,
        &metrics(1548, true, None),
    );
    assert_eq!(
        (code.code(), out),
        (20, limit_json(&cap_json(1548, true), P95_NULL))
    );

    // p95 だけ（容量は上限内）。
    let (code, out) = emit(
        &lat_breach(99),
        PackageQualityJudgment::Pass,
        &metrics(1549, false, Some((0, true))),
    );
    assert_eq!(
        (code.code(), out),
        (20, limit_json(&cap_json(1549, false), P95_OVER))
    );

    // 両方。
    let mut both = cap_breach(1548);
    both.extend(lat_breach(99));
    assert_eq!(both.len(), 2);
    let (code, out) = emit(
        &both,
        PackageQualityJudgment::Pass,
        &metrics(1548, true, Some((0, true))),
    );
    assert_eq!(
        (code.code(), out),
        (20, limit_json(&cap_json(1548, true), P95_OVER))
    );
}

/// 超過の出力は合否判定に依らず exit 20（REQ-21）。
#[test]
fn req21_breach_outputs_limit_exceeded_json_for_every_judgment() {
    for q in QUALITIES {
        let (code, out) = emit(&cap_breach(1548), q, &metrics(1548, true, None));
        assert_eq!(code, ExitCode::LimitExceeded);
        assert_eq!(out, limit_json(&cap_json(1548, true), P95_NULL));
    }
}

/// 上限ちょうどは超過でなく、exit 0・10・12 の JSON に `exceeded:false` が載る。
#[test]
fn req21_equal_to_limit_keeps_quality_outputs() {
    let mut b = cap_breach(1549);
    b.extend(lat_breach(100));
    assert!(b.is_empty());
    let m = metrics(1549, false, Some((1, false)));
    let tail = format!("{},\"infer_p95\":{P95_WITHIN}}}\n", cap_json(1549, false));

    let (code, out) = emit(&b, PackageQualityJudgment::Fail, &m);
    assert_eq!(
        (code.code(), out),
        (
            10,
            format!(
                "{{\"code\":\"judged_fail\",\"message\":\"judged as fail\",\"step\":\"package\",\"judgment\":\"fail\",\"acceptance_defined\":true,{tail}"
            )
        )
    );
    let (code, out) = emit(&b, PackageQualityJudgment::Undeterminable, &m);
    assert_eq!(
        (code.code(), out),
        (
            12,
            format!(
                "{{\"code\":\"pending\",\"message\":\"result is pending\",\"step\":\"package\",\"judgment\":\"undeterminable\",\"acceptance_defined\":true,{tail}"
            )
        )
    );
    let (code, out) = emit(&b, PackageQualityJudgment::Pass, &m);
    assert_eq!(
        (code.code(), out),
        (
            0,
            format!(
                "{{\"step\":\"package\",\"status\":\"ok\",\"judgment\":\"pass\",\"acceptance_defined\":true,{tail}"
            )
        )
    );
    let (code, out) = emit(&b, PackageQualityJudgment::NotDefined, &m);
    assert_eq!(
        (code.code(), out),
        (
            0,
            format!(
                "{{\"step\":\"package\",\"status\":\"ok\",\"judgment\":null,\"acceptance_defined\":false,{tail}"
            )
        )
    );
}
