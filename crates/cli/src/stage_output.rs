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
//! - exit ≠ 0（`Fail`・`Undeterminable`・`LimitExceeded`）: 新しい形は作らず、確定済みの
//!   `{"code","message"}`（`error_report::default_message` の固定語彙）へ流す
//!
//! 証拠種別: テストハーネス（バイナリでの完走は #136）。

use crate::error_report::{default_message, emit_error_report};
use crate::output::write_package_report;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::PackageReport;
use fandhe_edge_runtime::package_outcome::{PackageOutcome, PackageVerdict};
use std::io::{self, Write};

/// [`PackageOutcome`] を正常系の [`PackageReport`]、または異常系の [`ErrorReport`] へ写す。
///
/// `Ok` を返すのは `verdict` が Pass / NotDefined かつ `exit_code == ExitCode::Ok` のときに限る。
/// 両者が不整合なら `runtime_error` の `Err` を返す。`verdict` は
/// ワイルドカード無しで網羅し、区分が増えたらコンパイルエラーで気付けるようにする。
///
/// # Errors
/// exit ≠ 0 の区分は `Err(ErrorReport)`（エラー処理ではなく出力形の振り分け）。
pub fn package_outcome_report(outcome: &PackageOutcome) -> Result<PackageReport, ErrorReport> {
    // exit_code と verdict は別フィールドで不整合を構築できるため、verdict から期待される
    // 終了コードと一致しない場合は fail-closed で runtime_error へ倒す（上限超過などを
    // exit 0 へ変えない。REQ-21）。
    let expected = match outcome.verdict {
        PackageVerdict::Pass | PackageVerdict::NotDefined => ExitCode::Ok,
        PackageVerdict::Fail => ExitCode::JudgedFail,
        PackageVerdict::Undeterminable => ExitCode::Pending,
        PackageVerdict::LimitExceeded => ExitCode::LimitExceeded,
    };
    if outcome.exit_code != expected {
        return Err(ErrorReport::new(
            ExitCode::RuntimeError,
            default_message(ExitCode::RuntimeError),
        ));
    }
    match outcome.verdict {
        PackageVerdict::Pass => Ok(PackageReport::pass()),
        PackageVerdict::NotDefined => Ok(PackageReport::acceptance_not_defined()),
        PackageVerdict::Fail | PackageVerdict::Undeterminable | PackageVerdict::LimitExceeded => {
            Err(ErrorReport::new(
                outcome.exit_code,
                default_message(outcome.exit_code),
            ))
        }
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
        Ok(report) => write_package_report(out, &report),
        Err(report) => emit_error_report(out, &report),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// REQ-21: 不合格・判定不能・上限超過は確定済みの ErrorReport へ流れる。
    #[test]
    fn req21_non_ok_verdicts_use_error_report() {
        let (code, out) = emit(&[], PackageQualityJudgment::Fail);
        assert_eq!(code, ExitCode::JudgedFail);
        assert_eq!(
            out,
            "{\"code\":\"judged_fail\",\"message\":\"judged as fail\"}\n"
        );

        let (code, out) = emit(&[], PackageQualityJudgment::Undeterminable);
        assert_eq!(code, ExitCode::Pending);
        assert_eq!(
            out,
            "{\"code\":\"pending\",\"message\":\"result is pending\"}\n"
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
            assert_eq!(
                package_outcome_report(&o).is_ok(),
                o.exit_code == ExitCode::Ok
            );
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
        assert!(package_outcome_report(&o).is_err());
    }
}
