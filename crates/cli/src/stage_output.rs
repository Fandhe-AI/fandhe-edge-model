//! `package` 工程の結果を stdout の JSON へ写す CLI 層の変換（REQ-21・REQ-33・
//! TASK-33.2-2・#139）。
//!
//! # 呼び出し文脈
//!
//! `package` 工程（TASK-33.1-2・#136 で `main.rs` の `Invocation::Run` から結線する）が
//! `fandhe_edge_runtime::package_outcome::resolve_package_outcome` の結果を
//! [`emit_package_outcome`] へ渡す。上限超過（`limit_exceeded`）が合否判定に優先する
//! 規則は runtime 側で決まっており、ここでは再実装しない。
//!
//! # 契約
//!
//! すべての出力の末尾に計測値 `capacity`・`infer_p95`（#340・REQ-30・REQ-31）が付く（exit 64・70 を除く）。
//! exit 0・10・12 ではさらに末尾に、版管理台帳へ記録した版 `version`（`{"id","previous"}`。#491・REQ-39）が付く。
//! exit 20 は `package/` も台帳も作らないため `version` を載せない。
//!
//! - exit 0（`Pass`・`NotDefined`）: core の `PackageReport` を JSON 1 行で stdout へ
//! - exit 10・12（`Fail`・`Undeterminable`）: 合否基準が定義されているときにだけ生じる結果として、
//!   `{"code","message"}` に判定項目（`step`・`judgment`・`acceptance_defined`）を足した core の
//!   `PackageJudgedReport` を JSON 1 行で stdout へ（REQ-21・REQ-33・#328。`message` は
//!   `error_report::default_message` の固定語彙）
//! - exit 20（`LimitExceeded`）: core の `PackageLimitExceededReport`
//!   （`{"code","message","step","capacity","infer_p95"}`。超過の種類は各 `exceeded`。#340）
//! - 不整合の `runtime_error`: 確定済みの `{"code","message"}` へ流す。`breaches` と計測値の
//!   `exceeded` が食い違う場合もここへ倒す（fail-closed）
//!
//! # evaluate の skipped（REQ-17・REQ-33・TASK-33.3・#140）
//!
//! [`evaluate_start`] が評価データの状態を data 層の `evaluate_gate` へ通し、評価データ未定義なら
//! [`EvaluateStart::Skipped`]（[`emit_evaluate_skipped`] で `status:"skipped"`・exit 0）、
//! 凍結記録と一致すれば [`EvaluateStart::Proceed`]、不一致・矛盾は `ErrorReport`（exit ≠ 0。
//! skipped へ落とさない）へ振り分ける。skip 判定は CLI で再実装しない。
//! `Proceed` の後段（評価器の呼び出し・結果 JSON・評価完了の記録）は `stages::evaluate`
//! （#136・#314）が担う。同工程は PoC-16 と同様に、候補が学習済みか等の確認より前に skip 判定を行う。
//!
//! 証拠種別: テストハーネス（バイナリでの完走は `tests/pipeline_e2e.rs`。#136）。

use crate::error_report::{ToErrorReport, default_message, emit_error_report};
use crate::output::{
    write_evaluate_report, write_package_judged_report, write_package_limit_exceeded_report,
    write_package_report,
};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::{
    EvaluateReport, PackageJudgedReport, PackageLimitExceededReport, PackageMetrics, PackageReport,
    PackageVersion,
};
use fandhe_edge_data::eval_freeze::{EvalDataState, EvaluateGate, FreezeRecord, evaluate_gate};
use fandhe_edge_runtime::package_outcome::{LimitBreach, PackageOutcome, PackageVerdict};
use std::io::{self, Write};

/// `package` 工程の stdout へ出す JSON の 4 分岐（#328・#340）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackageStageOutput {
    /// exit 0（`Pass`・`NotDefined`）。
    Report(PackageReport),
    /// exit 10・12（`Fail`・`Undeterminable`。判定項目つき）。
    Judged(PackageJudgedReport),
    /// exit 20（容量・p95 の上限超過。計測値つき）。
    LimitExceeded(PackageLimitExceededReport),
    /// 不整合の `runtime_error`（`{"code","message"}`）。
    Error(ErrorReport),
}

impl PackageStageOutput {
    /// この出力に対応する終了コード。
    #[must_use]
    pub fn exit_code(&self) -> ExitCode {
        match self {
            PackageStageOutput::Report(_) => ExitCode::Ok,
            PackageStageOutput::Judged(report) => report.exit_code(),
            PackageStageOutput::LimitExceeded(report) => report.exit_code(),
            PackageStageOutput::Error(report) => report.code,
        }
    }
}

/// [`PackageOutcome`] と計測値を [`PackageStageOutput`] の 4 分岐へ写す。
///
/// `Report`（exit 0）は `verdict` が Pass / NotDefined・`exit_code == ExitCode::Ok`・`breaches` が
/// 空のときに限る。3 フィールドが不整合、または `breaches` と計測値の `exceeded` が食い違えば
/// `runtime_error` の `Error` を返す（超過を exit 0 へ変えない。REQ-21・#340）。`verdict` は
/// ワイルドカード無しで網羅し、区分が増えたらコンパイルエラーで気付けるようにする。
#[must_use]
pub fn package_outcome_report(
    outcome: &PackageOutcome,
    metrics: &PackageMetrics,
    version: Option<&PackageVersion>,
) -> PackageStageOutput {
    let runtime_error = || {
        PackageStageOutput::Error(ErrorReport::new(
            ExitCode::RuntimeError,
            default_message(ExitCode::RuntimeError),
        ))
    };
    // exit_code と verdict は別フィールドで不整合を構築できるため、verdict から期待される
    // 終了コードと一致しない場合は fail-closed で runtime_error へ倒す（上限超過などを
    // exit 0 へ変えない。REQ-21）。
    let expected = match outcome.verdict {
        PackageVerdict::Pass | PackageVerdict::NotDefined => ExitCode::Ok,
        PackageVerdict::Fail => ExitCode::JudgedFail,
        PackageVerdict::Undeterminable => ExitCode::Pending,
        PackageVerdict::LimitExceeded => ExitCode::LimitExceeded,
    };
    // 上限超過（breaches 非空）は合否より優先する（REQ-30・REQ-31・REQ-39）。breaches が
    // あるのに verdict / exit_code が LimitExceeded でなければ不整合として拒否する。
    let breach_inconsistent = !outcome.breaches.is_empty()
        && (outcome.verdict != PackageVerdict::LimitExceeded
            || outcome.exit_code != ExitCode::LimitExceeded);
    if breach_inconsistent || outcome.exit_code != expected {
        return runtime_error();
    }
    // 計測値の `exceeded` は breaches と一致しなければならない（LimitBreach は non_exhaustive のため
    // `matches!` で数える）。
    let capacity_breach = outcome
        .breaches
        .iter()
        .any(|b| matches!(b, LimitBreach::Capacity { .. }));
    let latency_breach = outcome
        .breaches
        .iter()
        .any(|b| matches!(b, LimitBreach::Latency { .. }));
    if capacity_breach != metrics.capacity.exceeded()
        || latency_breach != metrics.infer_p95.is_some_and(|p| p.exceeded())
    {
        return runtime_error();
    }
    let metrics = *metrics;
    // 公開した（exit 0・10・12）のに版が無い組は記録の欠落として fail-closed（#491）。
    let published = |make: &dyn Fn(PackageVersion) -> PackageStageOutput| {
        version.cloned().map_or_else(runtime_error, make)
    };
    match outcome.verdict {
        PackageVerdict::Pass => {
            published(&|v| PackageStageOutput::Report(PackageReport::pass(metrics, v)))
        }
        PackageVerdict::NotDefined => published(&|v| {
            PackageStageOutput::Report(PackageReport::acceptance_not_defined(metrics, v))
        }),
        PackageVerdict::Fail => published(&|v| {
            PackageStageOutput::Judged(PackageJudgedReport::fail(
                default_message(ExitCode::JudgedFail).to_string(),
                metrics,
                v,
            ))
        }),
        PackageVerdict::Undeterminable => published(&|v| {
            PackageStageOutput::Judged(PackageJudgedReport::undeterminable(
                default_message(ExitCode::Pending).to_string(),
                metrics,
                v,
            ))
        }),
        PackageVerdict::LimitExceeded => PackageLimitExceededReport::new(
            default_message(ExitCode::LimitExceeded).to_string(),
            metrics,
        )
        .map_or_else(runtime_error, PackageStageOutput::LimitExceeded),
    }
}

/// [`package_outcome_report`] の結果に応じて stdout へ JSON を 1 回だけ書き、終了コードを返す。
///
/// # Errors
/// 書き込み・flush・直列化の失敗を `io::Error` で返す（部分書き込み後は呼び出し側が
/// 後続の出力を打ち切ること。`output` の契約）。
pub fn emit_package_outcome<W: Write>(
    out: &mut W,
    outcome: &PackageOutcome,
    metrics: &PackageMetrics,
    version: Option<&PackageVersion>,
) -> io::Result<ExitCode> {
    match package_outcome_report(outcome, metrics, version) {
        PackageStageOutput::Report(report) => write_package_report(out, &report),
        PackageStageOutput::Judged(report) => write_package_judged_report(out, &report),
        PackageStageOutput::LimitExceeded(report) => {
            write_package_limit_exceeded_report(out, &report)
        }
        PackageStageOutput::Error(report) => emit_error_report(out, &report),
    }
}

/// `evaluate` 工程の開始判定の結果（REQ-17・TASK-33.3）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluateStart {
    /// 評価データ未定義。[`emit_evaluate_skipped`] で skipped（exit 0）を出す。
    Skipped(EvaluateReport),
    /// 凍結記録と実データが一致した。評価器へ進む（後段は `stages::evaluate`）。
    Proceed(FreezeRecord),
}

/// 評価データの状態と実データから `evaluate` を進めるか skipped にするかを決める。
///
/// 必ず data 層の `evaluate_gate` を経由する。データが渡されているのに skipped へ落とす経路、
/// 凍結記録との不一致を通す経路は作らない（fail-closed。REQ-17・REQ-27）。
///
/// # Errors
/// `FreezeError` を `ErrorReport`（`invalid_input`・`limit_exceeded` 等。exit ≠ 0）へ写して返す。
pub fn evaluate_start(
    state: &EvalDataState,
    actual_bytes: &[u8],
) -> Result<EvaluateStart, ErrorReport> {
    match evaluate_gate(state, actual_bytes) {
        Ok(EvaluateGate::Skip) => Ok(EvaluateStart::Skipped(EvaluateReport::skipped())),
        Ok(EvaluateGate::Proceed(record)) => Ok(EvaluateStart::Proceed(record)),
        Err(error) => Err(error.to_error_report()),
    }
}

/// skipped の `evaluate` 結果を stdout へ JSON 1 行で書き、[`ExitCode::Ok`] を返す。
///
/// # Errors
/// 書き込み・flush・直列化の失敗を `io::Error` で返す（`output` の契約）。
pub fn emit_evaluate_skipped<W: Write>(
    out: &mut W,
    report: &EvaluateReport,
) -> io::Result<ExitCode> {
    write_evaluate_report(out, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SKIPPED_LINE: &str = "{\"step\":\"evaluate\",\"status\":\"skipped\",\"reason\":\"evaluation_data_not_defined\"}\n";

    /// REQ-17・REQ-33: 評価データなし（空）は skipped・exit 0。
    #[test]
    fn req17_req33_not_provided_empty_bytes_is_skipped() {
        let start = evaluate_start(&EvalDataState::NotProvided, b"").expect("start");
        let EvaluateStart::Skipped(report) = start else {
            panic!("expected Skipped");
        };
        assert_eq!(report, EvaluateReport::skipped());
        let mut buf = Vec::new();
        let code = emit_evaluate_skipped(&mut buf, &report).expect("emit");
        assert_eq!(code, ExitCode::Ok);
        assert_eq!(String::from_utf8(buf).expect("utf8"), SKIPPED_LINE);
    }

    /// REQ-17: NotProvided なのに実データがあれば invalid_input で停止する。
    #[test]
    fn req17_not_provided_with_data_fails_closed() {
        let err = evaluate_start(&EvalDataState::NotProvided, b"x").expect_err("must fail");
        assert_eq!(err.code, ExitCode::InvalidInput);
    }

    /// REQ-17・REQ-27: 凍結記録とのハッシュ不一致は invalid_input で停止する。
    #[test]
    fn req17_frozen_hash_mismatch_fails_closed() {
        let record = fandhe_edge_data::eval_freeze::freeze_eval_data(b"a").expect("freeze");
        let err = evaluate_start(&EvalDataState::Frozen(record), b"b").expect_err("must fail");
        assert_eq!(err.code, ExitCode::InvalidInput);
    }

    /// REQ-17: 一致時は Proceed（skipped を出さない）。
    #[test]
    fn req17_frozen_matching_bytes_proceeds() {
        let record = fandhe_edge_data::eval_freeze::freeze_eval_data(b"a").expect("freeze");
        let start = evaluate_start(&EvalDataState::Frozen(record.clone()), b"a").expect("start");
        assert_eq!(start, EvaluateStart::Proceed(record));
    }

    /// REQ-21: data 層の EvalStatus::Skipped の終了コードと出力関数の戻り値が一致する。
    #[test]
    fn req21_skipped_exit_code_matches_eval_status() {
        let mut buf = Vec::new();
        let code = emit_evaluate_skipped(&mut buf, &EvaluateReport::skipped()).expect("emit");
        assert_eq!(
            code,
            fandhe_edge_data::eval_freeze::EvalStatus::Skipped.exit_code()
        );
        assert_eq!(code, ExitCode::Ok);
    }
    use fandhe_edge_core::stage_report::{
        InferP95, PackageCapacity, PackageCapacityComponents, PackageComponentSize,
    };
    use fandhe_edge_runtime::package_outcome::{PackageQualityJudgment, resolve_package_outcome};

    const CAP_OK: &str = "\"capacity\":{\"total_bytes\":125,\"limit_bytes\":40000000,\"exceeded\":false,\"guideline_bytes\":40000000,\"over_guideline\":false,\"components\":{\"weights\":{\"bytes\":100,\"file_count\":1},\"vocab_or_feature_transform\":{\"bytes\":0,\"file_count\":0},\"label_table\":{\"bytes\":20,\"file_count\":1},\"calibration\":{\"bytes\":0,\"file_count\":0},\"metadata\":{\"bytes\":5,\"file_count\":1}}}";
    const CAP_OVER: &str = "\"capacity\":{\"total_bytes\":125,\"limit_bytes\":100,\"exceeded\":true,\"guideline_bytes\":40000000,\"over_guideline\":false,\"components\":{\"weights\":{\"bytes\":100,\"file_count\":1},\"vocab_or_feature_transform\":{\"bytes\":0,\"file_count\":0},\"label_table\":{\"bytes\":20,\"file_count\":1},\"calibration\":{\"bytes\":0,\"file_count\":0},\"metadata\":{\"bytes\":5,\"file_count\":1}}}";
    const RUNTIME_ERROR: &str = "{\"code\":\"runtime_error\",\"message\":\"runtime error\"}\n";

    fn metrics(capacity_exceeded: bool, p95: Option<(u64, u64, bool)>) -> PackageMetrics {
        let c = PackageComponentSize::new;
        let limit = if capacity_exceeded { 100 } else { 40_000_000 };
        PackageMetrics {
            capacity: PackageCapacity::new(
                125,
                Some(limit),
                capacity_exceeded,
                (40_000_000, false),
                PackageCapacityComponents::new(c(100, 1), c(0, 0), c(20, 1), c(0, 0), c(5, 1)),
            ),
            infer_p95: p95.map(|(p, l, e)| InferP95::new(p, l, e)),
        }
    }

    /// 記録した版の期待値（#491）。
    const VER: &str = ",\"version\":{\"id\":\"v1\",\"previous\":null}";

    fn v1() -> PackageVersion {
        PackageVersion::new("v1".to_string(), None)
    }

    /// REQ-39・REQ-21・#491: 公開した結果（exit 0・10・12）に版が無ければ runtime_error（記録の欠落を成功に
    /// しない）。exit 20 は版の有無によらず `version` を載せない。
    #[test]
    fn req39_issue491_version_is_required_for_published_outcomes_only() {
        for q in [
            PackageQualityJudgment::Pass,
            PackageQualityJudgment::Fail,
            PackageQualityJudgment::Undeterminable,
            PackageQualityJudgment::NotDefined,
        ] {
            let o = resolve_package_outcome(&[], q);
            let mut buf = Vec::new();
            let code =
                emit_package_outcome(&mut buf, &o, &metrics(false, None), None).expect("emit");
            assert_eq!(code, ExitCode::RuntimeError);
            assert_eq!(String::from_utf8(buf).expect("utf8"), RUNTIME_ERROR);
        }
        let o = resolve_package_outcome(&[cap_breach()], PackageQualityJudgment::Pass);
        for version in [None, Some(&v1())] {
            let mut buf = Vec::new();
            let code =
                emit_package_outcome(&mut buf, &o, &metrics(true, None), version).expect("emit");
            assert_eq!(code, ExitCode::LimitExceeded);
            assert!(!String::from_utf8(buf).expect("utf8").contains("version"));
        }
    }

    fn cap_breach() -> LimitBreach {
        LimitBreach::Capacity {
            measured_bytes: 125,
            limit_bytes: 100,
        }
    }

    fn lat_breach() -> LimitBreach {
        LimitBreach::Latency {
            measured_p95_ns: 7_000,
            limit_ns: 6_000,
        }
    }

    fn emit(
        breaches: &[LimitBreach],
        q: PackageQualityJudgment,
        m: &PackageMetrics,
    ) -> (ExitCode, String) {
        let mut buf = Vec::new();
        let code = emit_package_outcome(
            &mut buf,
            &resolve_package_outcome(breaches, q),
            m,
            Some(&v1()),
        )
        .expect("emit");
        (code, String::from_utf8(buf).expect("utf8"))
    }

    /// REQ-33・#340: Pass は exit 0 で judgment と計測値を含む JSON。
    #[test]
    fn req33_pass_emits_report() {
        let (code, out) = emit(&[], PackageQualityJudgment::Pass, &metrics(false, None));
        assert_eq!(code, ExitCode::Ok);
        assert_eq!(
            out,
            format!(
                "{{\"step\":\"package\",\"status\":\"ok\",\"judgment\":\"pass\",\"acceptance_defined\":true,{CAP_OK},\"infer_p95\":null{VER}}}\n"
            )
        );
    }

    /// REQ-33・REQ-31・#340: 基準未設定は judgment が null で、p95 は上限があれば値が載る。
    #[test]
    fn req33_not_defined_emits_report() {
        let m = metrics(false, Some((5000, 6000, false)));
        let (code, out) = emit(&[], PackageQualityJudgment::NotDefined, &m);
        assert_eq!(code, ExitCode::Ok);
        assert_eq!(
            out,
            format!(
                "{{\"step\":\"package\",\"status\":\"ok\",\"judgment\":null,\"acceptance_defined\":false,{CAP_OK},\"infer_p95\":{{\"p95_us\":5000,\"limit_us\":6000,\"exceeded\":false}}{VER}}}\n"
            )
        );
    }

    /// REQ-21・REQ-33・#328・#340: 不合格・判定不能は判定項目つき、上限超過は計測値つきの exit 20。
    #[test]
    fn req21_non_ok_verdicts_use_judged_or_limit_exceeded_report() {
        let (code, out) = emit(&[], PackageQualityJudgment::Fail, &metrics(false, None));
        assert_eq!(code, ExitCode::JudgedFail);
        assert_eq!(
            out,
            format!(
                "{{\"code\":\"judged_fail\",\"message\":\"judged as fail\",\"step\":\"package\",\"judgment\":\"fail\",\"acceptance_defined\":true,{CAP_OK},\"infer_p95\":null{VER}}}\n"
            )
        );

        let (code, out) = emit(
            &[],
            PackageQualityJudgment::Undeterminable,
            &metrics(false, None),
        );
        assert_eq!(code, ExitCode::Pending);
        assert_eq!(
            out,
            format!(
                "{{\"code\":\"pending\",\"message\":\"result is pending\",\"step\":\"package\",\"judgment\":\"undeterminable\",\"acceptance_defined\":true,{CAP_OK},\"infer_p95\":null{VER}}}\n"
            )
        );

        // 容量だけの超過（p95 の上限なし）。
        let (code, out) = emit(
            &[cap_breach()],
            PackageQualityJudgment::Pass,
            &metrics(true, None),
        );
        assert_eq!(code, ExitCode::LimitExceeded);
        assert_eq!(
            out,
            format!(
                "{{\"code\":\"limit_exceeded\",\"message\":\"resource limit exceeded\",\"step\":\"package\",{CAP_OVER},\"infer_p95\":null}}\n"
            )
        );

        // p95 だけの超過は capacity.exceeded=false・infer_p95.exceeded=true で区別できる。
        let (code, out) = emit(
            &[lat_breach()],
            PackageQualityJudgment::Pass,
            &metrics(false, Some((7, 6, true))),
        );
        assert_eq!(code, ExitCode::LimitExceeded);
        assert_eq!(
            out,
            format!(
                "{{\"code\":\"limit_exceeded\",\"message\":\"resource limit exceeded\",\"step\":\"package\",{CAP_OK},\"infer_p95\":{{\"p95_us\":7,\"limit_us\":6,\"exceeded\":true}}}}\n"
            )
        );
    }

    /// 不変条件: `Report(_)` は exit_code が Ok のときに限る。
    #[test]
    fn req21_report_ok_only_when_exit_code_ok() {
        for q in [
            PackageQualityJudgment::Pass,
            PackageQualityJudgment::Fail,
            PackageQualityJudgment::Undeterminable,
            PackageQualityJudgment::NotDefined,
        ] {
            let o = resolve_package_outcome(&[], q);
            let output = package_outcome_report(&o, &metrics(false, None), Some(&v1()));
            assert_eq!(
                matches!(output, PackageStageOutput::Report(_)),
                o.exit_code == ExitCode::Ok
            );
            assert_eq!(output.exit_code(), o.exit_code);
        }
    }

    /// REQ-21: exit_code と verdict の不整合（上限超過を Pass で exit 0 にする等）は
    /// runtime_error へ倒れ、正常系 JSON を出さない。
    #[test]
    fn req21_inconsistent_outcome_fails_closed() {
        let m = metrics(false, None);
        let mut o = resolve_package_outcome(&[], PackageQualityJudgment::Pass);
        o.exit_code = ExitCode::LimitExceeded;
        let mut buf = Vec::new();
        let code = emit_package_outcome(&mut buf, &o, &m, Some(&v1())).expect("emit");
        assert_eq!(code, ExitCode::RuntimeError);
        assert_eq!(String::from_utf8(buf).expect("utf8"), RUNTIME_ERROR);

        let mut o = resolve_package_outcome(&[], PackageQualityJudgment::Fail);
        o.exit_code = ExitCode::Ok;
        assert!(matches!(
            package_outcome_report(&o, &m, Some(&v1())),
            PackageStageOutput::Error(_)
        ));
    }

    /// REQ-21・REQ-30・REQ-31・REQ-39: breaches が非空なのに Pass / Ok の組を渡しても
    /// 成功 JSON を出さず runtime_error へ倒れる。
    #[test]
    fn req21_breaches_with_pass_ok_fails_closed() {
        let m = metrics(true, None);
        let mut o = resolve_package_outcome(&[], PackageQualityJudgment::Pass);
        o.breaches = vec![cap_breach()];
        let mut buf = Vec::new();
        let code = emit_package_outcome(&mut buf, &o, &m, Some(&v1())).expect("emit");
        assert_eq!(code, ExitCode::RuntimeError);
        assert_eq!(String::from_utf8(buf).expect("utf8"), RUNTIME_ERROR);

        // verdict だけ LimitExceeded で exit_code が Ok の組も拒否する。
        o.verdict = PackageVerdict::LimitExceeded;
        assert!(matches!(
            package_outcome_report(&o, &m, Some(&v1())),
            PackageStageOutput::Error(_)
        ));

        // breaches 非空で verdict Fail・exit_code JudgedFail の組も runtime_error にする。
        let mut o = resolve_package_outcome(&[], PackageQualityJudgment::Fail);
        o.breaches = vec![lat_breach()];
        let mut buf = Vec::new();
        let code = emit_package_outcome(
            &mut buf,
            &o,
            &metrics(false, Some((7, 6, true))),
            Some(&v1()),
        )
        .expect("emit");
        assert_eq!(code, ExitCode::RuntimeError);
    }

    /// REQ-30・TASK-41.9・#406: 上限未設定で目安超過は警告（`over_guideline:true`）だけで、
    /// `limit_bytes:null`・`exceeded:false` のまま exit 0（判定不能なら 12）になる。
    #[test]
    fn req30_unset_limit_over_guideline_is_warning_only() {
        let c = PackageComponentSize::new;
        let m = PackageMetrics {
            capacity: PackageCapacity::new(
                41_000_000,
                None,
                false,
                (40_000_000, true),
                PackageCapacityComponents::new(
                    c(41_000_000, 1),
                    c(0, 0),
                    c(0, 0),
                    c(0, 0),
                    c(0, 0),
                ),
            ),
            infer_p95: None,
        };
        let cap = "\"capacity\":{\"total_bytes\":41000000,\"limit_bytes\":null,\"exceeded\":false,\"guideline_bytes\":40000000,\"over_guideline\":true,\"components\":{\"weights\":{\"bytes\":41000000,\"file_count\":1},\"vocab_or_feature_transform\":{\"bytes\":0,\"file_count\":0},\"label_table\":{\"bytes\":0,\"file_count\":0},\"calibration\":{\"bytes\":0,\"file_count\":0},\"metadata\":{\"bytes\":0,\"file_count\":0}}}";
        let (code, out) = emit(&[], PackageQualityJudgment::NotDefined, &m);
        assert_eq!(code, ExitCode::Ok);
        assert_eq!(
            out,
            format!(
                "{{\"step\":\"package\",\"status\":\"ok\",\"judgment\":null,\"acceptance_defined\":false,{cap},\"infer_p95\":null{VER}}}\n"
            )
        );
        let (code, _) = emit(&[], PackageQualityJudgment::Undeterminable, &m);
        assert_eq!(code, ExitCode::Pending);
    }

    /// REQ-21・REQ-30・REQ-31・#340: breaches と計測値の `exceeded` が食い違えば runtime_error
    /// （超過があるのに `exceeded:false`・超過が無いのに `exceeded:true`・p95 の値が無いのに
    /// 遅延の超過がある）。
    #[test]
    fn req21_issue340_metrics_exceeded_mismatch_fails_closed() {
        let cases: [(Vec<LimitBreach>, PackageMetrics); 5] = [
            (vec![cap_breach()], metrics(false, None)),
            (vec![], metrics(true, None)),
            (vec![lat_breach()], metrics(false, Some((7, 6, false)))),
            (vec![], metrics(false, Some((7, 6, true)))),
            (vec![lat_breach()], metrics(false, None)),
        ];
        for (breaches, m) in cases {
            let q = PackageQualityJudgment::Pass;
            let (code, out) = emit(&breaches, q, &m);
            assert_eq!(code, ExitCode::RuntimeError, "{breaches:?}");
            assert_eq!(out, RUNTIME_ERROR);
        }
    }
}
