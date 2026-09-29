//! 各層のエラー → 終了コード 7 種と `{"code","message"}` JSON 1 行への変換の
//! 契約テスト（REQ-21・REQ-33・TASK-33.2-1・#138。証拠種別: テストハーネス）。
//!
//! 資格情報に見える値はすべてダミー（`.claude/rules/security.md`）。

use fandhe_edge_cli::error_report::{
    ToErrorReport, default_message, emit_error, emit_error_report, train_outcome_error_report,
};
use fandhe_edge_core::definition::DefinitionError;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::infer_input::InferInputError;
use fandhe_edge_core::judgment::JudgmentError;
use fandhe_edge_data::eval_freeze::{EvalDataState, FreezeError, evaluate_gate, freeze_eval_data};
use fandhe_edge_train::error::{TrainProcessError, TrainRequestError, TrainResultError};
use fandhe_edge_train::request::TrainRequest;
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::search::SearchError;

const MARKER: &str = "sk-test-dummy-secret-marker";

/// 出力が 1 行の JSON で、期待どおりの `code`・戻り値になっていること。
fn assert_emitted(report: &ErrorReport, name: &str, numeric: u8) -> String {
    let mut buffer: Vec<u8> = Vec::new();
    let code = emit_error_report(&mut buffer, report);
    assert_eq!(code, report.code);
    assert_eq!(code.code(), numeric);
    assert_eq!(
        std::process::ExitCode::from(code),
        std::process::ExitCode::from(numeric)
    );
    let text = String::from_utf8(buffer).unwrap();
    assert_eq!(text.matches('\n').count(), 1);
    assert!(text.ends_with('\n'));
    assert!(
        text.starts_with(&format!("{{\"code\":\"{name}\",\"message\":\"")),
        "unexpected output: {text}"
    );
    text
}

/// REQ-21: 7 種それぞれの `code` 名・数値・固定 message の厳密一致。
#[test]
fn req21_seven_exit_codes_map_to_exact_json() {
    let table: [(ExitCode, &str, u8, &str); 7] = [
        (ExitCode::Ok, "ok", 0, "ok"),
        (ExitCode::JudgedFail, "judged_fail", 10, "judged as fail"),
        (
            ExitCode::OutOfScope,
            "out_of_scope",
            11,
            "input is out of scope",
        ),
        (ExitCode::Pending, "pending", 12, "result is pending"),
        (
            ExitCode::LimitExceeded,
            "limit_exceeded",
            20,
            "resource limit exceeded",
        ),
        (ExitCode::InvalidInput, "invalid_input", 64, "invalid input"),
        (ExitCode::RuntimeError, "runtime_error", 70, "runtime error"),
    ];
    assert_eq!(table.len(), ExitCode::ALL.len());
    for (code, name, numeric, message) in table {
        assert_eq!(default_message(code), message);
        let report = ErrorReport::new(code, default_message(code));
        let text = assert_emitted(&report, name, numeric);
        assert_eq!(
            text,
            format!("{{\"code\":\"{name}\",\"message\":\"{message}\"}}\n")
        );
    }
}

fn check<E: ToErrorReport>(err: &E, name: &str, numeric: u8) -> ErrorReport {
    let report = err.to_error_report();
    assert_ne!(
        report.code,
        ExitCode::Ok,
        "layer errors must never map to ok"
    );
    assert_emitted(&report, name, numeric);
    let mut buffer: Vec<u8> = Vec::new();
    assert_eq!(emit_error(&mut buffer, err), report.code);
    report
}

fn minimal_request() -> TrainRequest {
    TrainRequest::from_json_slice(
        include_bytes!("../../../fixtures/train_contract/request_minimal.json").as_slice(),
    )
    .unwrap()
}

fn worker_error_outcome(code: &str, message: &str) -> TrainOutcome {
    let line = format!("{{\"status\":\"error\",\"code\":\"{code}\",\"message\":\"{message}\"}}\n");
    TrainOutcome::from_worker_stdout(line.as_bytes(), &minimal_request()).unwrap()
}

/// REQ-21: 各層のエラーから実際に届く値の具体値。
#[test]
fn req21_layer_errors_map_to_expected_codes() {
    check(&DefinitionError::EmptyOptions, "invalid_input", 64);
    check(
        &DefinitionError::TooLarge {
            path: None,
            size: 2,
            limit: 1,
        },
        "limit_exceeded",
        20,
    );
    check(
        &InferInputError::TooLarge { len: 2, limit: 1 },
        "limit_exceeded",
        20,
    );
    check(
        &JudgmentError::ScoreSumNotOne { sum: 0.5 },
        "runtime_error",
        70,
    );

    let record = freeze_eval_data(b"frozen").unwrap();
    let mismatch = evaluate_gate(&EvalDataState::Frozen(record), b"changed").unwrap_err();
    assert!(matches!(mismatch, FreezeError::HashMismatch { .. }));
    check(&mismatch, "invalid_input", 64);
    check(&FreezeError::LengthOverflow, "limit_exceeded", 20);

    check(
        &TrainRequestError::TooLarge { size: 2, limit: 1 },
        "limit_exceeded",
        20,
    );
    check(&TrainResultError::NotUtf8, "runtime_error", 70);
    check(
        &TrainProcessError::WallTimeout {
            limit_ms: 1,
            child_reaped: true,
        },
        "limit_exceeded",
        20,
    );
    check(&TrainProcessError::InvalidJobDir, "invalid_input", 64);
    check(&SearchError::<String>::EmptyCandidates, "invalid_input", 64);
    let report = check(
        &SearchError::<String>::Internal {
            detail: MARKER.to_string(),
        },
        "runtime_error",
        70,
    );
    assert_eq!(report.message, "candidate search failed: runtime_error");
}

/// REQ-21: 学習ワーカーの失敗は `training_diverged` が 12（pending）へ写る。
#[test]
fn req21_train_outcome_error_maps_via_failure_code() {
    let diverged = worker_error_outcome("training_diverged", "boom");
    let report = train_outcome_error_report(&diverged).unwrap();
    assert_eq!(report.code, ExitCode::Pending);
    assert_eq!(report.message, "train worker failed: training_diverged");
    assert_emitted(&report, "pending", 12);

    let limit = worker_error_outcome("limit_exceeded", "boom");
    let report = train_outcome_error_report(&limit).unwrap();
    assert_eq!(report.code, ExitCode::LimitExceeded);
    assert_emitted(&report, "limit_exceeded", 20);
}

/// security.md: 文字列を持つ variant・ワーカー由来 message のマーカーが
/// エラー JSON に現れない。
#[test]
fn req21_error_json_does_not_leak_data_or_credentials() {
    let path = std::path::PathBuf::from(format!("/home/alice/{MARKER}/def.json"));
    let definition_errors = [
        DefinitionError::Read {
            path,
            source: std::io::Error::other("boom"),
        },
        DefinitionError::UnsupportedSchema {
            schema: MARKER.to_string(),
        },
        DefinitionError::DuplicateOptionId {
            id: MARKER.to_string(),
        },
    ];
    for err in &definition_errors {
        assert!(err.to_string().contains(MARKER), "premise: Display leaks");
        let line = err.to_error_report().to_json_line().unwrap();
        assert!(!line.contains(MARKER), "leaked: {line}");
        let mut buffer: Vec<u8> = Vec::new();
        let _ = emit_error(&mut buffer, err);
        assert!(!String::from_utf8(buffer).unwrap().contains(MARKER));
    }

    let search = SearchError::<String>::Internal {
        detail: MARKER.to_string(),
    };
    assert!(
        search.to_string().contains(MARKER),
        "premise: Display leaks"
    );
    let line = search.to_error_report().to_json_line().unwrap();
    assert!(!line.contains(MARKER));

    let outcome = worker_error_outcome("runtime_error", MARKER);
    let TrainOutcome::Error(failure) = &outcome else {
        panic!("expected worker error outcome");
    };
    assert!(failure.message().contains(MARKER), "premise: message leaks");
    let line = train_outcome_error_report(&outcome)
        .unwrap()
        .to_json_line()
        .unwrap();
    assert!(!line.contains(MARKER));
}
