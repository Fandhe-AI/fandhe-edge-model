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
//! - exit 0（`Pass`・`NotDefined`）: core の `PackageReport` を JSON 1 行で stdout へ
//! - exit 10・12（`Fail`・`Undeterminable`）: 合否基準が定義されているときにだけ生じる結果として、
//!   `{"code","message"}` に判定項目（`step`・`judgment`・`acceptance_defined`）を足した core の
//!   `PackageJudgedReport` を JSON 1 行で stdout へ（REQ-21・REQ-33・#328。`message` は
//!   `error_report::default_message` の固定語彙）
//! - exit 20（`LimitExceeded`）・不整合の `runtime_error`: 新しい形は作らず、確定済みの
//!   `{"code","message"}` へ流す
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
use crate::output::{write_evaluate_report, write_package_judged_report, write_package_report};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::{EvaluateReport, PackageJudgedReport, PackageReport};
use fandhe_edge_data::eval_freeze::{EvalDataState, EvaluateGate, FreezeRecord, evaluate_gate};
use fandhe_edge_runtime::package_outcome::{PackageOutcome, PackageVerdict};
use std::io::{self, Write};

/// `package` 工程の stdout へ出す JSON の 3 分岐（#328）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackageStageOutput {
    /// exit 0（`Pass`・`NotDefined`）。
    Report(PackageReport),
    /// exit 10・12（`Fail`・`Undeterminable`。判定項目つき）。
    Judged(PackageJudgedReport),
    /// exit 20 および不整合の `runtime_error`（`{"code","message"}`）。
    Error(ErrorReport),
}

impl PackageStageOutput {
    /// この出力に対応する終了コード。
    #[must_use]
    pub fn exit_code(&self) -> ExitCode {
        match self {
            PackageStageOutput::Report(_) => ExitCode::Ok,
            PackageStageOutput::Judged(report) => report.exit_code(),
            PackageStageOutput::Error(report) => report.code,
        }
    }
}

/// [`PackageOutcome`] を [`PackageStageOutput`] の 3 分岐へ写す。
///
/// `Report`（exit 0）は `verdict` が Pass / NotDefined・`exit_code == ExitCode::Ok`・`breaches` が
/// 空のときに限る。3 フィールドが不整合なら `runtime_error` の `Error` を返す。`verdict` は
/// ワイルドカード無しで網羅し、区分が増えたらコンパイルエラーで気付けるようにする。
#[must_use]
pub fn package_outcome_report(outcome: &PackageOutcome) -> PackageStageOutput {
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
        return PackageStageOutput::Error(ErrorReport::new(
            ExitCode::RuntimeError,
            default_message(ExitCode::RuntimeError),
        ));
    }
    match outcome.verdict {
        PackageVerdict::Pass => PackageStageOutput::Report(PackageReport::pass()),
        PackageVerdict::NotDefined => {
            PackageStageOutput::Report(PackageReport::acceptance_not_defined())
        }
        PackageVerdict::Fail => PackageStageOutput::Judged(PackageJudgedReport::fail(
            default_message(ExitCode::JudgedFail).to_string(),
        )),
        PackageVerdict::Undeterminable => PackageStageOutput::Judged(
            PackageJudgedReport::undeterminable(default_message(ExitCode::Pending).to_string()),
        ),
        PackageVerdict::LimitExceeded => PackageStageOutput::Error(ErrorReport::new(
            outcome.exit_code,
            default_message(outcome.exit_code),
        )),
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
) -> io::Result<ExitCode> {
    match package_outcome_report(outcome) {
        PackageStageOutput::Report(report) => write_package_report(out, &report),
        PackageStageOutput::Judged(report) => write_package_judged_report(out, &report),
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
    use fandhe_edge_runtime::package_outcome::{
        LimitBreach, PackageQualityJudgment, resolve_package_outcome,
    };

    fn emit(breaches: &[LimitBreach], q: PackageQualityJudgment) -> (ExitCode, String) {
        let mut buf = Vec::new();
        let code =
            emit_package_outcome(&mut buf, &resolve_package_outcome(breaches, q)).expect("emit");
        (code, String::from_utf8(buf).expect("utf8"))
    }

    /// REQ-33: Pass は exit 0 で judgment を含む JSON。
    #[test]
    fn req33_pass_emits_report() {
        let (code, out) = emit(&[], PackageQualityJudgment::Pass);
        assert_eq!(code, ExitCode::Ok);
        assert_eq!(
            out,
            "{\"step\":\"package\",\"status\":\"ok\",\"judgment\":\"pass\",\"acceptance_defined\":true}\n"
        );
    }

    /// REQ-33: 基準未設定は exit 0 で judgment が null。
    #[test]
    fn req33_not_defined_emits_report() {
        let (code, out) = emit(&[], PackageQualityJudgment::NotDefined);
        assert_eq!(code, ExitCode::Ok);
        assert_eq!(
            out,
            "{\"step\":\"package\",\"status\":\"ok\",\"judgment\":null,\"acceptance_defined\":false}\n"
        );
    }

    /// REQ-21・REQ-33・#328: 不合格・判定不能は判定項目つき、上限超過は確定済みの ErrorReport。
    #[test]
    fn req21_non_ok_verdicts_use_judged_or_error_report() {
        let (code, out) = emit(&[], PackageQualityJudgment::Fail);
        assert_eq!(code, ExitCode::JudgedFail);
        assert_eq!(
            out,
            "{\"code\":\"judged_fail\",\"message\":\"judged as fail\",\"step\":\"package\",\"judgment\":\"fail\",\"acceptance_defined\":true}\n"
        );

        let (code, out) = emit(&[], PackageQualityJudgment::Undeterminable);
        assert_eq!(code, ExitCode::Pending);
        assert_eq!(
            out,
            "{\"code\":\"pending\",\"message\":\"result is pending\",\"step\":\"package\",\"judgment\":\"undeterminable\",\"acceptance_defined\":true}\n"
        );

        let breach = LimitBreach::Capacity {
            measured_bytes: 2,
            limit_bytes: 1,
        };
        let (code, out) = emit(&[breach], PackageQualityJudgment::Pass);
        assert_eq!(code, ExitCode::LimitExceeded);
        assert_eq!(
            out,
            "{\"code\":\"limit_exceeded\",\"message\":\"resource limit exceeded\"}\n"
        );
    }

    /// 不変条件: `Ok(_)` は exit_code が Ok のときに限る。
    #[test]
    fn req21_report_ok_only_when_exit_code_ok() {
        for q in [
            PackageQualityJudgment::Pass,
            PackageQualityJudgment::Fail,
            PackageQualityJudgment::Undeterminable,
            PackageQualityJudgment::NotDefined,
        ] {
            let o = resolve_package_outcome(&[], q);
            let output = package_outcome_report(&o);
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
        let mut o = resolve_package_outcome(&[], PackageQualityJudgment::Pass);
        o.exit_code = ExitCode::LimitExceeded;
        let mut buf = Vec::new();
        let code = emit_package_outcome(&mut buf, &o).expect("emit");
        assert_eq!(code, ExitCode::RuntimeError);
        assert_eq!(
            String::from_utf8(buf).expect("utf8"),
            "{\"code\":\"runtime_error\",\"message\":\"runtime error\"}\n"
        );

        let mut o = resolve_package_outcome(&[], PackageQualityJudgment::Fail);
        o.exit_code = ExitCode::Ok;
        assert!(matches!(
            package_outcome_report(&o),
            PackageStageOutput::Error(_)
        ));
    }

    /// REQ-21・REQ-30・REQ-31・REQ-39: breaches が非空なのに Pass / Ok の組を渡しても
    /// 成功 JSON を出さず runtime_error へ倒れる。
    #[test]
    fn req21_breaches_with_pass_ok_fails_closed() {
        let breach = LimitBreach::Capacity {
            measured_bytes: 2,
            limit_bytes: 1,
        };
        let mut o = resolve_package_outcome(&[], PackageQualityJudgment::Pass);
        o.breaches = vec![breach];
        let mut buf = Vec::new();
        let code = emit_package_outcome(&mut buf, &o).expect("emit");
        assert_eq!(code, ExitCode::RuntimeError);
        assert_eq!(
            String::from_utf8(buf).expect("utf8"),
            "{\"code\":\"runtime_error\",\"message\":\"runtime error\"}\n"
        );

        // verdict だけ LimitExceeded で exit_code が Ok の組も拒否する。
        o.verdict = PackageVerdict::LimitExceeded;
        assert!(matches!(
            package_outcome_report(&o),
            PackageStageOutput::Error(_)
        ));

        // breaches 非空で verdict Fail・exit_code JudgedFail の組も runtime_error にする。
        let mut o = resolve_package_outcome(&[], PackageQualityJudgment::Fail);
        o.breaches = vec![LimitBreach::Latency {
            measured_p95_ns: 2,
            limit_ns: 1,
        }];
        let mut buf = Vec::new();
        let code = emit_package_outcome(&mut buf, &o).expect("emit");
        assert_eq!(code, ExitCode::RuntimeError);
    }
}
