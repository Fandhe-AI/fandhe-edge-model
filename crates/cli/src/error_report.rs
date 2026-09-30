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
use fandhe_edge_core::artifact_meta::ArtifactMetaError;
use fandhe_edge_core::definition::DefinitionError;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::FsError;
use fandhe_edge_core::infer_input::InferInputError;
use fandhe_edge_core::judgment::JudgmentError;
use fandhe_edge_data::eval_freeze::FreezeError;
use fandhe_edge_guard::format::FormatRejection;
use fandhe_edge_guard::path::PathRejection;
use fandhe_edge_guard::resource::{GuardRunError, ResourceKind, ResourceLimitExceeded};
use fandhe_edge_runtime::pipeline::{BackendError, BatchError, InferError};
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

/// 経路の閉じ込めの拒否（REQ-39・TASK-39.4-2・#159）。`Display` は candidate の
/// パスを含みうるため使わず、`reason_code` の固定語彙だけを出す。
impl ToErrorReport for PathRejection {
    fn to_error_report(&self) -> ErrorReport {
        ErrorReport::new(
            self.exit_code(),
            format!("path rejected: {}", self.reason_code()),
        )
    }
}

/// 形式・拡張子・サイズの拒否。理由コードだけの固定語彙で、パス・拡張子・内容を含めない（REQ-39）。
impl ToErrorReport for FormatRejection {
    fn to_error_report(&self) -> ErrorReport {
        let code = self.exit_code();
        if code == ExitCode::LimitExceeded {
            return ErrorReport::new(code, "model file exceeds size limit");
        }
        ErrorReport::new(code, format!("format rejected: {}", self.reason_code()))
    }
}

/// 資源上限の超過の記録（REQ-39・TASK-39.5-1・#170）。固定語彙のみで、入力・子の出力・パスを含めない。
impl ToErrorReport for ResourceLimitExceeded {
    fn to_error_report(&self) -> ErrorReport {
        let code = self.exit_code();
        match self.kind() {
            ResourceKind::Time => ErrorReport::new(code, "inference time limit exceeded"),
            _ => ErrorReport::new(code, default_message(code)),
        }
    }
}

/// 時間上限 runner 自体の失敗。理由コードだけの固定語彙で、パス・本文を含めない（REQ-39）。
impl ToErrorReport for GuardRunError {
    fn to_error_report(&self) -> ErrorReport {
        ErrorReport::new(
            self.exit_code(),
            format!("guarded run failed: {}", self.code()),
        )
    }
}

/// `artifact.json` の解釈エラー。入力値を含めない固定文へ写す（REQ-39）。
impl ToErrorReport for ArtifactMetaError {
    fn to_error_report(&self) -> ErrorReport {
        ErrorReport::new(ExitCode::InvalidInput, "artifact metadata is invalid")
    }
}

/// 開いたファイルの上限付き読み込みの失敗。サイズ超過は `limit_exceeded`（REQ-39）、
/// それ以外は実行時エラー。`Display` はパスを含むため使わない。
impl ToErrorReport for FsError {
    fn to_error_report(&self) -> ErrorReport {
        match self {
            FsError::TooLarge { .. } => ErrorReport::new(
                ExitCode::LimitExceeded,
                "artifact metadata exceeds size limit",
            ),
            _ => ErrorReport::new(ExitCode::RuntimeError, "cannot read artifact metadata"),
        }
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

/// 推論 1 件の失敗。上限超過は `limit_exceeded`、それ以外は `runtime_error`。message は固定語彙
/// （長さ・入力本文を含めない）。`InferError` は他 crate の `non_exhaustive` で網羅 `match` を
/// 書けないため、将来の variant は `runtime_error` へ倒す（fail-closed。`Ok` は返さない）。
impl ToErrorReport for InferError {
    fn to_error_report(&self) -> ErrorReport {
        let code = match self {
            InferError::InputTooLarge { .. }
            | InferError::TooManyTokens { .. }
            | InferError::TooManyScores { .. } => ExitCode::LimitExceeded,
            // 1 件の計算時間上限（REQ-39）は `DeadlineExceeded` と同じく `limit_exceeded`（REQ-21）。
            InferError::Backend(BackendError::TimeLimitExceeded) => ExitCode::LimitExceeded,
            InferError::Preprocess(_) | InferError::Backend(_) | InferError::InvalidScores => {
                ExitCode::RuntimeError
            }
            _ => ExitCode::RuntimeError,
        };
        ErrorReport::new(code, default_message(code))
    }
}

/// バッチ全体の失敗。上限超過は `limit_exceeded`。`non_exhaustive` のため将来の variant は
/// `runtime_error` へ倒す（fail-closed）。
impl ToErrorReport for BatchError {
    fn to_error_report(&self) -> ErrorReport {
        let code = match self {
            BatchError::TooManyInputs { .. }
            | BatchError::TotalInputTooLarge { .. }
            | BatchError::ResultTooLarge { .. }
            | BatchError::DeadlineExceeded => ExitCode::LimitExceeded,
            _ => ExitCode::RuntimeError,
        };
        ErrorReport::new(code, default_message(code))
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

    /// REQ-39・TASK-39.5-1: 時間超過は `limit_exceeded`(20)・固定文に写り、runner の失敗は 64/70 に写る。
    #[test]
    fn req39_resource_limit_records_map_to_fixed_reports() {
        use fandhe_edge_guard::resource::INFER_TIME_LIMIT;
        use std::time::Duration;
        let rec = ResourceLimitExceeded::time(INFER_TIME_LIMIT, Duration::from_secs(10), true);
        let report = rec.to_error_report();
        assert_eq!(report.code, ExitCode::LimitExceeded);
        assert_eq!(report.message, "inference time limit exceeded");
        assert_eq!(
            GuardRunError::InvalidProgram.to_error_report(),
            ErrorReport::new(
                ExitCode::InvalidInput,
                "guarded run failed: invalid_program"
            )
        );
        assert_eq!(
            GuardRunError::ReapTimeout.to_error_report(),
            ErrorReport::new(ExitCode::RuntimeError, "guarded run failed: reap_timeout")
        );
    }

    /// REQ-21・REQ-39: 推論・バッチの失敗は固定語彙の code/message に写り `Ok` にならない。
    #[test]
    fn req21_infer_and_batch_errors_map_to_fixed_reports() {
        use fandhe_edge_runtime::pipeline::PreprocessError;
        let limit = ErrorReport::new(ExitCode::LimitExceeded, "resource limit exceeded");
        let runtime = ErrorReport::new(ExitCode::RuntimeError, "runtime error");
        let infer_cases = [
            (
                InferError::InputTooLarge { len: 2, limit: 1 },
                limit.clone(),
            ),
            (
                InferError::TooManyTokens { len: 2, limit: 1 },
                limit.clone(),
            ),
            (
                InferError::TooManyScores { len: 2, limit: 1 },
                limit.clone(),
            ),
            (
                InferError::Backend(BackendError::TimeLimitExceeded),
                limit.clone(),
            ),
            (InferError::Backend(BackendError::Failed), runtime.clone()),
            (InferError::InvalidScores, runtime.clone()),
        ];
        for (error, expected) in infer_cases {
            assert_eq!(error.to_error_report(), expected);
        }
        let _ = PreprocessError::Failed;
        let batch_cases = [
            BatchError::TooManyInputs { len: 2, limit: 1 },
            BatchError::TotalInputTooLarge { total: 2, limit: 1 },
            BatchError::ResultTooLarge { limit: 1 },
            BatchError::DeadlineExceeded,
        ];
        for error in batch_cases {
            assert_eq!(error.to_error_report(), limit);
        }
    }

    /// REQ-39・REQ-21: 経路の拒否は invalid_input と固定 message で、候補パスを含まない。
    #[test]
    fn req39_path_rejection_maps_to_invalid_input_without_path() {
        use fandhe_edge_guard::path::{EscapeKind, PathRejection};
        let r = PathRejection::Escapes {
            candidate: std::path::PathBuf::from("../secret/outside"),
            kind: EscapeKind::ParentTraversal,
        }
        .to_error_report();
        assert_eq!(r.code, ExitCode::InvalidInput);
        assert_eq!(r.message, "path rejected: path_escapes_root");
        let r = PathRejection::NotDirectory {
            candidate: std::path::PathBuf::from("x"),
        }
        .to_error_report();
        assert_eq!(r.message, "path rejected: not_directory");
        assert_eq!(
            PathRejection::UnsupportedPlatform.to_error_report().code,
            ExitCode::RuntimeError
        );
    }

    /// REQ-39: artifact.json の不正・サイズ超過・I/O 失敗の写像。
    #[test]
    fn req39_artifact_meta_and_fs_errors_map_to_fixed_reports() {
        let r = ArtifactMetaError::Malformed.to_error_report();
        assert_eq!(r.code, ExitCode::InvalidInput);
        assert_eq!(r.message, "artifact metadata is invalid");
        let r = FsError::TooLarge {
            path: std::path::PathBuf::from("/secret/dir"),
            size: 2,
            limit: 1,
        }
        .to_error_report();
        assert_eq!(r.code, ExitCode::LimitExceeded);
        assert_eq!(r.message, "artifact metadata exceeds size limit");
        let r = FsError::NotRegularFile {
            path: std::path::PathBuf::from("/secret/dir"),
        }
        .to_error_report();
        assert_eq!(r.code, ExitCode::RuntimeError);
        assert!(!r.message.contains("secret"));
    }
}
