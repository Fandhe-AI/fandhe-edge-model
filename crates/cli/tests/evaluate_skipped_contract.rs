//! 評価データ未定義の `evaluate` が skipped・exit 0 で終わることの結合テスト
//! （REQ-17・REQ-33・TASK-33.3・#140）。
//!
//! 証拠種別: テストハーネス（バイナリでの完走は `pipeline_e2e.rs`。#136）。期待値の `status:"skipped"`・exit 0 は
//! PoC-16 縦断 2（評価データなしの evaluate）の実測に基づく（`docs/spec` は読まない）。
//! `reason` は英語の固定語彙へ置き換えている（日本語出力規約）。

use fandhe_edge_cli::args::Subcommand;
use fandhe_edge_cli::log::StderrLog;
use fandhe_edge_cli::output::write_error_report;
use fandhe_edge_cli::stage_output::{EvaluateStart, emit_evaluate_skipped, evaluate_start};
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_data::eval_freeze::EvalDataState;

const SKIPPED_LINE: &str =
    "{\"step\":\"evaluate\",\"status\":\"skipped\",\"reason\":\"evaluation_data_not_defined\"}\n";

fn run_skipped(stdout: &mut Vec<u8>) -> ExitCode {
    match evaluate_start(&EvalDataState::NotProvided, b"").expect("start") {
        EvaluateStart::Skipped(report) => emit_evaluate_skipped(stdout, &report).expect("emit"),
        EvaluateStart::Proceed(_) => panic!("must skip"),
    }
}

/// AC1: 評価データなしは skipped の JSON 1 行と exit 0。
#[test]
fn req33_evaluate_without_eval_data_writes_skipped_and_exit_zero() {
    let mut stdout = Vec::new();
    let code = run_skipped(&mut stdout);
    assert_eq!(code, ExitCode::Ok);
    assert_eq!(
        std::process::ExitCode::from(code),
        std::process::ExitCode::from(0u8)
    );
    assert_eq!(String::from_utf8(stdout).expect("utf8"), SKIPPED_LINE);
}

/// ログは stderr 側だけに出て stdout に混ざらない。
#[test]
fn req33_evaluate_skipped_logs_stay_on_stderr() {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    {
        let mut log = StderrLog::new(&mut stderr);
        log.info(Subcommand::Evaluate, "start");
        run_skipped(&mut stdout);
        log.info(Subcommand::Evaluate, "done");
    }
    let stdout = String::from_utf8(stdout).expect("utf8");
    assert_eq!(stdout, SKIPPED_LINE);
    assert_eq!(stdout.matches('\n').count(), 1);
    assert!(!stdout.contains("fandhe-edge:"));
    let stderr = String::from_utf8(stderr).expect("utf8");
    assert_eq!(stderr.lines().count(), 2);
    for line in stderr.lines() {
        assert!(line.starts_with("fandhe-edge: evaluate: "), "{line}");
    }
}

/// fail-closed: データがあるのに未提供扱いなら skipped にせず invalid_input（exit 64）。
#[test]
fn req17_evaluate_not_provided_with_data_is_invalid_input() {
    let report = evaluate_start(&EvalDataState::NotProvided, b"x").expect_err("must fail");
    let mut stdout = Vec::new();
    let code = write_error_report(&mut stdout, &report).expect("emit");
    assert_eq!(code, ExitCode::InvalidInput);
    assert_eq!(
        std::process::ExitCode::from(code),
        std::process::ExitCode::from(64u8)
    );
    let out = String::from_utf8(stdout).expect("utf8");
    assert!(out.starts_with("{\"code\":\"invalid_input\""), "{out}");
    assert!(!out.contains("skipped"));
}
