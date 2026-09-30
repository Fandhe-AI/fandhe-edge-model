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
use fandhe_edge_eval::eval_data_invariance::EvalDataInvarianceError;
use fandhe_edge_eval::final_test_once::{AcquireError, ApplyOnceError};
use fandhe_edge_eval::invariance::EvaluationInvarianceError;
use fandhe_edge_guard::format::FormatRejection;
use fandhe_edge_guard::kind::KindRejection;
use fandhe_edge_guard::kind_version::KindVersionRejection;
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
            ResourceKind::Memory => ErrorReport::new(code, "inference memory limit exceeded"),
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

/// `kind` の許可リスト・構文検査の拒否（REQ-39・TASK-39.2-4・#156）。`reason_code` の固定語彙だけを
/// 使い、`Display`（許可一覧・入力由来の値を含みうる）は使わない。
impl ToErrorReport for KindRejection {
    fn to_error_report(&self) -> ErrorReport {
        ErrorReport::new(
            self.exit_code(),
            format!("kind rejected: {}", self.reason_code()),
        )
    }
}

/// `kind_version` の許可リスト拒否（REQ-39・TASK-39.6-1・#174）。`reason_code` の固定語彙だけを使い、
/// 入力値（版番号）を含めない。
impl ToErrorReport for KindVersionRejection {
    fn to_error_report(&self) -> ErrorReport {
        ErrorReport::new(
            self.exit_code(),
            format!("kind_version rejected: {}", self.reason_code()),
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

/// `evaluate` の推論クロージャ（[`fandhe_edge_eval::final_test_once::apply_once`] の `predict`）が
/// 返す失敗。本文・パスを運ばない固定の区分のみ（REQ-27・REQ-39。#314）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalPredictFailure {
    /// 1 件の推論が時間上限（`INFER_TIME_LIMIT`）を超えた。
    TimeLimit,
    /// 推論に使うモデルが、評価前に検証した実体と一致しない。
    ModelChanged,
    /// バックエンドの組み立てなど、その他の失敗。
    Failed,
}

/// [`fandhe_edge_eval::final_test_once::apply_once`] の失敗の全体（評価データ側・モデル側・
/// ロック取得・推論の入れ子）。
pub type ApplyOnceFailure =
    EvalDataInvarianceError<EvaluationInvarianceError<ApplyOnceError<EvalPredictFailure>>>;

/// 最終 test の台帳操作（事前登録・ロック取得）の失敗を [`ErrorReport`] にする。
///
/// message は固定語彙で、`AcquireError` が持つパス・代表構成 ID の実値は出さない（security.md）。
/// `AcquireError` は `non_exhaustive` のため、将来の variant は `runtime_error` へ倒す。
#[must_use]
pub fn acquire_error_report(error: &AcquireError) -> ErrorReport {
    match error {
        AcquireError::AlreadyApplied { .. } => ErrorReport::new(
            ExitCode::InvalidInput,
            "candidate has already been evaluated on the frozen data",
        ),
        AcquireError::NotRegistered
        | AcquireError::UnregisteredConfig
        | AcquireError::WeightsNotRegistered
        | AcquireError::ComponentNotRegistered { .. } => ErrorReport::new(
            ExitCode::InvalidInput,
            "candidate is not registered for evaluation",
        ),
        AcquireError::RegistryTampered { .. }
        | AcquireError::RegistryInvalid { .. }
        | AcquireError::LedgerDirInvalid { .. }
        | AcquireError::AlreadyRegistered { .. } => {
            ErrorReport::new(ExitCode::InvalidInput, "final test ledger is invalid")
        }
        AcquireError::InvalidConfigId { .. } => {
            ErrorReport::new(ExitCode::InvalidInput, "evaluation config id is invalid")
        }
        AcquireError::SelectionChanged => ErrorReport::new(
            ExitCode::InvalidInput,
            "selection differs from the one fixed at the first evaluation",
        ),
        AcquireError::SelectionNotPinned => {
            ErrorReport::new(ExitCode::InvalidInput, "evaluation has not been completed")
        }
        AcquireError::WeightsDigest {
            source: FsError::TooLarge { .. },
        } => ErrorReport::new(
            ExitCode::LimitExceeded,
            default_message(ExitCode::LimitExceeded),
        ),
        _ => ErrorReport::new(ExitCode::RuntimeError, "cannot evaluate candidate"),
    }
}

/// `apply_once` の失敗を [`ErrorReport`] にする（REQ-27・REQ-39・#314）。
///
/// 凍結記録との不一致・評価中の評価データ / モデルの変化・適用済み・未登録・台帳の不整合は
/// `invalid_input`、サイズ・時間の上限超過は `limit_exceeded`、その他の I/O 失敗は
/// `runtime_error`。message は固定語彙で、評価データの本文・パス・ハッシュの実値を含めない。
/// 各エラー型は `non_exhaustive` のため、将来の variant は `runtime_error` へ倒す（fail-closed）。
#[must_use]
pub fn apply_once_error_report(error: &ApplyOnceFailure) -> ErrorReport {
    let invalid = |m: &str| ErrorReport::new(ExitCode::InvalidInput, m);
    let limit = || {
        ErrorReport::new(
            ExitCode::LimitExceeded,
            default_message(ExitCode::LimitExceeded),
        )
    };
    let failed = || ErrorReport::new(ExitCode::RuntimeError, "cannot evaluate candidate");
    match error {
        EvalDataInvarianceError::FrozenRecordMismatch { .. } => {
            invalid("evaluation data does not match the freeze record")
        }
        EvalDataInvarianceError::ChangedDuringEvaluation { .. } => {
            invalid("evaluation data changed during evaluation")
        }
        EvalDataInvarianceError::TooLarge { .. } => limit(),
        EvalDataInvarianceError::NotRegularFile { .. } => invalid("evaluation data is invalid"),
        EvalDataInvarianceError::Evaluation(inner) => match inner {
            EvaluationInvarianceError::Changed(_) => invalid("model changed during evaluation"),
            EvaluationInvarianceError::TooLarge { .. } => limit(),
            EvaluationInvarianceError::NotRegularFile { .. } => invalid("model file is invalid"),
            EvaluationInvarianceError::Evaluation(apply) => match apply {
                ApplyOnceError::Acquire(acquire) => acquire_error_report(acquire),
                ApplyOnceError::Decode => invalid("evaluation data is invalid"),
                ApplyOnceError::Prediction(EvalPredictFailure::TimeLimit) => limit(),
                ApplyOnceError::Prediction(EvalPredictFailure::ModelChanged) => {
                    invalid("model changed during evaluation")
                }
                ApplyOnceError::Prediction(EvalPredictFailure::Failed) => failed(),
                _ => failed(),
            },
            _ => failed(),
        },
        _ => failed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_edge_core::hash::Sha256Digest;
    use fandhe_edge_eval::final_test_once::AppliedBy;
    use fandhe_edge_guard::kind::KindAllowlist;

    fn nest(inner: ApplyOnceError<EvalPredictFailure>) -> ApplyOnceFailure {
        EvalDataInvarianceError::Evaluation(EvaluationInvarianceError::Evaluation(inner))
    }

    /// REQ-27・REQ-39・REQ-21: `apply_once` の各失敗の終了コードと固定 message が完全一致し、
    /// パス・ハッシュの実値を含まない。
    #[test]
    fn req27_apply_once_failures_map_to_fixed_reports() {
        let digest = Sha256Digest::of_bytes(b"x");
        let path = std::path::PathBuf::from("/secret/path");
        let cases: Vec<(ApplyOnceFailure, ExitCode, &str)> = vec![
            (
                EvalDataInvarianceError::FrozenRecordMismatch {
                    expected_sha256: digest,
                    actual_sha256: digest,
                    expected_byte_len: 1,
                    actual_byte_len: 2,
                },
                ExitCode::InvalidInput,
                "evaluation data does not match the freeze record",
            ),
            (
                EvalDataInvarianceError::ChangedDuringEvaluation {
                    before: digest,
                    after: digest,
                },
                ExitCode::InvalidInput,
                "evaluation data changed during evaluation",
            ),
            (
                nest(ApplyOnceError::Acquire(AcquireError::AlreadyApplied {
                    by: AppliedBy::RepresentativeConfig,
                    lock_path: path.clone(),
                })),
                ExitCode::InvalidInput,
                "candidate has already been evaluated on the frozen data",
            ),
            (
                nest(ApplyOnceError::Acquire(AcquireError::UnregisteredConfig)),
                ExitCode::InvalidInput,
                "candidate is not registered for evaluation",
            ),
            (
                nest(ApplyOnceError::Acquire(AcquireError::WeightsNotRegistered)),
                ExitCode::InvalidInput,
                "candidate is not registered for evaluation",
            ),
            (
                nest(ApplyOnceError::Acquire(AcquireError::LedgerDirInvalid {
                    path: path.clone(),
                })),
                ExitCode::InvalidInput,
                "final test ledger is invalid",
            ),
            (
                nest(ApplyOnceError::Acquire(AcquireError::RegistryTampered {
                    reason: "x",
                })),
                ExitCode::InvalidInput,
                "final test ledger is invalid",
            ),
            (
                nest(ApplyOnceError::Acquire(AcquireError::Io {
                    path,
                    source: std::io::Error::other("boom"),
                })),
                ExitCode::RuntimeError,
                "cannot evaluate candidate",
            ),
            (
                nest(ApplyOnceError::Decode),
                ExitCode::InvalidInput,
                "evaluation data is invalid",
            ),
            (
                nest(ApplyOnceError::Prediction(EvalPredictFailure::TimeLimit)),
                ExitCode::LimitExceeded,
                "resource limit exceeded",
            ),
            (
                nest(ApplyOnceError::Prediction(EvalPredictFailure::ModelChanged)),
                ExitCode::InvalidInput,
                "model changed during evaluation",
            ),
            (
                nest(ApplyOnceError::Prediction(EvalPredictFailure::Failed)),
                ExitCode::RuntimeError,
                "cannot evaluate candidate",
            ),
        ];
        for (error, code, message) in cases {
            let report = apply_once_error_report(&error);
            assert_eq!((report.code, report.message.as_str()), (code, message));
        }
    }

    /// REQ-39・REQ-21: `kind` の拒否は invalid_input・固定語彙（入力値を含めない）。
    #[test]
    fn req39_kind_rejection_maps_to_invalid_input_fixed_message() {
        let allow = KindAllowlist::supported();
        let cases = [
            ("pt", "kind rejected: unsupported_kind"),
            ("", "kind rejected: malformed_kind"),
            ("c3; rm -rf ~", "kind rejected: malformed_kind"),
            (&"a".repeat(65), "kind rejected: malformed_kind"),
        ];
        for (kind, message) in cases {
            let rejection = allow.check(kind).expect_err("must be rejected");
            let report = rejection.to_error_report();
            assert_eq!(report.code, ExitCode::InvalidInput);
            assert_eq!(report.message, message);
        }
    }

    /// REQ-39・REQ-21・TASK-39.6-1: `kind_version` の拒否は invalid_input・固定語彙（版番号を含めない）。
    #[test]
    fn req39_kind_version_rejection_maps_to_invalid_input_fixed_message() {
        let kind = KindAllowlist::supported().check("c3").expect("allowed");
        let rejection = fandhe_edge_guard::kind_version::KindVersionAllowlist::supported()
            .check(kind, 99)
            .expect_err("must be rejected");
        let report = rejection.to_error_report();
        assert_eq!(report.code, ExitCode::InvalidInput);
        assert_eq!(
            report.message,
            "kind_version rejected: unsupported_kind_version"
        );
    }

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
        let mem = ResourceLimitExceeded::memory(
            INFER_TIME_LIMIT,
            2_147_483_648,
            2_190_000_000,
            Duration::from_secs(1),
            true,
        );
        assert_eq!(
            mem.to_error_report(),
            ErrorReport::new(ExitCode::LimitExceeded, "inference memory limit exceeded")
        );
        assert_eq!(
            GuardRunError::MemoryProbe.to_error_report(),
            ErrorReport::new(
                ExitCode::RuntimeError,
                "guarded run failed: memory_probe_failed"
            )
        );
        assert_eq!(
            GuardRunError::MemoryLimitUnsupported.to_error_report(),
            ErrorReport::new(
                ExitCode::RuntimeError,
                "guarded run failed: memory_limit_unsupported"
            )
        );
        assert_eq!(
            GuardRunError::InvalidProgram.to_error_report(),
            ErrorReport::new(
                ExitCode::InvalidInput,
                "guarded run failed: invalid_program"
            )
        );
        assert_eq!(
            GuardRunError::KillFailed.to_error_report(),
            ErrorReport::new(ExitCode::RuntimeError, "guarded run failed: kill_failed")
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
