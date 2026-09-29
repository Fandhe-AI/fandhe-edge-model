//! 各層のエラーを終了コード 7 種と `{"code","message"}` の JSON 1 行へ変換
//! する CLI 層の変換モジュール（REQ-21・REQ-33・TASK-33.2-1・#138）。
//!
//! # 呼び出し文脈
//!
//! CLI の各工程（`register → … → infer`）が下位層（共通コア・データ契約・
//! 学習ワーカー）から受け取った `Result` の `Err` を、[`emit_error`] で
//! stdout へ JSON 1 行として書き、戻り値の [`ExitCode`] で終了する。
//! `main.rs` への配線は TASK-33.1-2（#136）・TASK-33.2-2（#139）で行う
//! （本モジュールは配線しない）。
//!
//! # 層間契約
//!
//! 下位 crate は CLI に依存しない。変換（[`ToErrorReport`]）は CLI 層に置き、
//! `ErrorReport` のスキーマ・`ExitCode` の値・`output::write_error_report`
//! の挙動は変更しない（REQ-21・REQ-33。変更はユーザー承認事項）。
//!
//! # message 方針（`.claude/rules/security.md`）
//!
//! `message` は英語の固定語彙とし、データ本文・パス・利用者指定の値・
//! 学習ワーカー由来の文字列を含めない。`Display` を使うのは文字列フィール
//! ドを持たない型（または文書化済みで持たない型）に限り、`SearchError` の
//! `Display`・`WorkerFailure::message` は使わない。
//!
//! # `Ok` の扱い
//!
//! [`emit_error_report`] は `report.code` を書き換えずに返す（`Ok` も素通し。
//! help 出力が `code:"ok"` を使うため）。層のエラーから来る変換
//! （[`ToErrorReport`] の実装）は `Ok` を返してはならない。

use crate::output::{
    definition_error_report, infer_input_error_report, judgment_error_report, write_error_report,
};
use fandhe_edge_core::definition::DefinitionError;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::infer_input::InferInputError;
use fandhe_edge_core::judgment::JudgmentError;
use fandhe_edge_data::eval_freeze::FreezeError;
use fandhe_edge_train::error::{TrainProcessError, TrainRequestError, TrainResultError};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::search::SearchError;
use std::io::{self, Write};

/// 終了コードごとの固定の英語 message（データを含まない）。
///
/// 7 種すべてを網羅する `match`（ワイルドカードなし）。終了コードが増えた
/// 場合にコンパイルエラーで気付けるようにする（REQ-21）。
#[must_use]
pub const fn default_message(code: ExitCode) -> &'static str {
    match code {
        ExitCode::Ok => "ok",
        ExitCode::JudgedFail => "judged as fail",
        ExitCode::OutOfScope => "input is out of scope",
        ExitCode::Pending => "result is pending",
        ExitCode::LimitExceeded => "resource limit exceeded",
        ExitCode::InvalidInput => "invalid input",
        ExitCode::RuntimeError => "runtime error",
    }
}

/// 層のエラー型を [`ErrorReport`] へ写す CLI 層の trait。
///
/// 実装の不変条件: 返す `code` は `ExitCode::Ok` にならず、`message` は
/// データ本文・パス・利用者指定値・ワーカー由来の文字列を含まない。
pub trait ToErrorReport {
    /// 異常系の [`ErrorReport`] を作る。
    fn to_error_report(&self) -> ErrorReport;
}

/// `report` を JSON 1 行で `out` へ書き、終了コードを返す。
///
/// `report.code` は書き換えず返す（`Ok` を含む）。
///
/// # Errors
/// 書き込み・flush の失敗は `io::Error` として呼び出し側へ伝える。
/// `ExitCode::RuntimeError` へ丸めない（正常に出力できた `runtime_error`
/// の報告と区別するため）。部分書き込み後は出力が壊れているため、呼び出し
/// 側は後続の出力を打ち切ること（`write_error_report` の契約）。
pub fn emit_error_report<W: Write>(out: &mut W, report: &ErrorReport) -> io::Result<ExitCode> {
    write_error_report(out, report)
}

/// 層のエラーを変換して [`emit_error_report`] で書く薄いヘルパー。
///
/// # Errors
/// [`emit_error_report`] と同じ（書き込み失敗を `io::Error` で伝える）。
pub fn emit_error<W: Write, E: ToErrorReport + ?Sized>(
    out: &mut W,
    err: &E,
) -> io::Result<ExitCode> {
    emit_error_report(out, &err.to_error_report())
}

/// 学習ワーカーの結果が失敗（`TrainOutcome::Error`）のときの
/// [`ErrorReport`]。`TrainOutcome::Ok` は異常系ではないため `None`。
///
/// `code` は `FailureCode::exit_code()`（`training_diverged` は 12）。
/// message は固定語彙のみで、ワーカー由来の `message` は使わない（学習
/// データを含みうるため）。
#[must_use]
pub fn train_outcome_error_report(outcome: &TrainOutcome) -> Option<ErrorReport> {
    match outcome {
        TrainOutcome::Ok(_) => None,
        TrainOutcome::Error(failure) => {
            let failure_code = failure.failure_code();
            Some(ErrorReport::new(
                failure_code.exit_code(),
                format!("train worker failed: {}", failure_code.as_str()),
            ))
        }
    }
}

impl ToErrorReport for DefinitionError {
    fn to_error_report(&self) -> ErrorReport {
        definition_error_report(self)
    }
}

impl ToErrorReport for InferInputError {
    fn to_error_report(&self) -> ErrorReport {
        infer_input_error_report(self)
    }
}

impl ToErrorReport for JudgmentError {
    fn to_error_report(&self) -> ErrorReport {
        judgment_error_report(self)
    }
}

/// `Display` は sha256・バイト長のみで、評価データ本文・パスを含まない。
impl ToErrorReport for FreezeError {
    fn to_error_report(&self) -> ErrorReport {
        ErrorReport::new(self.exit_code(), self.to_string())
    }
}

/// `Display` は数値・`&'static str`・`Category` のみ（文字列フィールドなし）。
impl ToErrorReport for TrainRequestError {
    fn to_error_report(&self) -> ErrorReport {
        ErrorReport::new(self.exit_code(), self.to_string())
    }
}

/// `Display` は数値・`&'static str`・`Category` のみ（文字列フィールドなし）。
impl ToErrorReport for TrainResultError {
    fn to_error_report(&self) -> ErrorReport {
        ErrorReport::new(self.exit_code(), self.to_string())
    }
}

/// `Display` はフィールド名・数値・`io::ErrorKind` のみ（内側のエラーも
/// 同様に文字列フィールドを持たない）。
impl ToErrorReport for TrainProcessError {
    fn to_error_report(&self) -> ErrorReport {
        ErrorReport::new(self.exit_code(), self.to_string())
    }
}

/// `SearchError` の `Display` は汎用の `E`・`Internal { detail }` を含みう
/// るため使わず、終了コードごとの固定文にする。
impl<E> ToErrorReport for SearchError<E> {
    fn to_error_report(&self) -> ErrorReport {
        let code = self.exit_code();
        ErrorReport::new(code, format!("candidate search failed: {}", code.name()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-21: 7 種の固定 message の具体値。
    #[test]
    fn req21_default_message_for_all_seven_codes() {
        let expected = [
            (ExitCode::Ok, "ok"),
            (ExitCode::JudgedFail, "judged as fail"),
            (ExitCode::OutOfScope, "input is out of scope"),
            (ExitCode::Pending, "result is pending"),
            (ExitCode::LimitExceeded, "resource limit exceeded"),
            (ExitCode::InvalidInput, "invalid input"),
            (ExitCode::RuntimeError, "runtime error"),
        ];
        assert_eq!(expected.len(), ExitCode::ALL.len());
        for (code, message) in expected {
            assert_eq!(default_message(code), message);
        }
    }

    /// REQ-21・REQ-33: 書き込み失敗は `Err` で伝播し、`RuntimeError` へ丸めない。
    #[test]
    fn req21_emit_propagates_io_error_when_write_fails() {
        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("simulated"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let report = ErrorReport::new(ExitCode::InvalidInput, "invalid input");
        let error = emit_error_report(&mut FailingWriter, &report).unwrap_err();
        assert_eq!(error.to_string(), "simulated");
    }

    /// REQ-21: 成功時は `report.code` をそのまま返す（`Ok` も素通し）。
    #[test]
    fn req21_emit_passes_code_through_including_ok() {
        for code in ExitCode::ALL {
            let mut buffer: Vec<u8> = Vec::new();
            let report = ErrorReport::new(code, default_message(code));
            assert_eq!(emit_error_report(&mut buffer, &report).unwrap(), code);
            assert_eq!(buffer.iter().filter(|b| **b == b'\n').count(), 1);
        }
    }
}
