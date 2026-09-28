//! 結合テスト: 型不正の推論入力から `ErrorReport` の JSON 1 行出力まで
//! （TASK-21.2）。
//!
//! `fandhe-edge-cli` は `serde_json` に直接依存しないため（`Cargo.toml`
//! 参照。依存の追加はユーザー承認事項。
//! `.claude/rules/dependency-policy.md`）、出力の読み戻しは文字列一致で
//! 検証する（実装計画 3.3 の方針）。

use fandhe_edge_cli::output::{infer_input_error_report, write_error_report};
use fandhe_edge_core::definition::{InputRepresentation, IoSchema};
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::infer_input::InferInput;

fn io_bytes() -> IoSchema {
    IoSchema {
        input: InputRepresentation::Bytes,
    }
}

/// TASK-21.2: 型不正の推論入力行（`input` が配列）を
/// `InferInput::parse` → `infer_input_error_report` → `write_error_report`
/// に通すと、JSON 1 行の `ErrorReport` が書かれ、戻り値の `ExitCode` が 64
/// になること。入力本文（配列の中身）が出力に含まれないこと。
#[test]
fn req21_invalid_input_writes_single_error_json_line_with_exit_64() {
    let secret_marker = "should-not-leak-into-error-output";
    let text = format!(r#"{{"id":"row-1","input":["{secret_marker}"]}}"#);

    let err = InferInput::parse(&text, &io_bytes()).expect_err("type mismatch must be rejected");
    let report = infer_input_error_report(&err);

    let mut buffer: Vec<u8> = Vec::new();
    let exit_code = write_error_report(&mut buffer, &report).unwrap();

    assert_eq!(exit_code, ExitCode::InvalidInput);
    assert_eq!(exit_code.code(), 64);
    assert_eq!(
        std::process::ExitCode::from(exit_code),
        std::process::ExitCode::from(64u8)
    );

    let output = String::from_utf8(buffer).unwrap();
    assert_eq!(
        output.matches('\n').count(),
        1,
        "must write exactly one line"
    );
    assert!(output.starts_with(r#"{"code":"invalid_input","message":"#));
    assert!(output.trim_end().ends_with('}'));
    assert!(
        !output.contains(secret_marker),
        "input body must not leak into error output"
    );
}

/// TASK-21.2: 未知フィールド（未対応の入力種別キー）でも同様に 64 で
/// JSON 1 行が書かれること。
#[test]
fn req21_unknown_field_writes_single_error_json_line_with_exit_64() {
    let text = r#"{"id":"row-1","input":"x","image":"unsupported"}"#;
    let err = InferInput::parse(text, &io_bytes()).expect_err("unknown field must be rejected");
    let report = infer_input_error_report(&err);

    let mut buffer: Vec<u8> = Vec::new();
    let exit_code = write_error_report(&mut buffer, &report).unwrap();

    assert_eq!(exit_code.code(), 64);
    let output = String::from_utf8(buffer).unwrap();
    assert_eq!(output.matches('\n').count(), 1);
    assert!(!output.contains("image"));
    assert!(!output.contains("unsupported"));
}

/// TASK-23.2 整合: 空入力は `InferInput::parse` 自体が成功するため、本
/// 経路（エラー出力）には到達しないことを確認する（回帰防止の明示テス
/// ト）。
#[test]
fn req21_empty_input_does_not_reach_error_output_path() {
    let text = r#"{"id":"row-1","input":""}"#;
    let result = InferInput::parse(text, &io_bytes());
    assert!(result.is_ok(), "empty input must not be treated as invalid");
}
