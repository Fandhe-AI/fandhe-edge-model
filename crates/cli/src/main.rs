//! `fandhe-edge` バイナリの入口（薄い配線のみ。REQ-33・TASK-33.1-1）。
//!
//! `args_os()`（非 UTF-8 でも panic しない）を [`fandhe_edge_cli::args::parse`]
//! に渡し、結果で分岐する。ロジックはすべて lib 側に置く。
//!
//! - help: 「1 呼び出し 1 JSON」契約（REQ-33）を守るため、確定済みの
//!   `ErrorReport` の形（`{"code":"ok","message":"<help テキスト>"}`）を
//!   stdout に JSON 1 行で出し exit 0。help 用の新スキーマ・フィールドは
//!   作らない（stderr は使わない）。
//! - 引数エラー: TASK-21.2 で確定済みの `ErrorReport` を stdout に JSON 1 行、
//!   exit 64（フィールドは増やさない）。
//! - 解析に成功したコマンド: 下位層への接続は TASK-33.1-2（#136）の範囲の
//!   ため、完走を装わず `runtime_error`（exit 70）で未実装を返す。

use fandhe_edge_cli::args::{self, Invocation};
use fandhe_edge_cli::output::write_error_report;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};

fn main() -> std::process::ExitCode {
    let report = match args::parse(std::env::args_os().skip(1)) {
        Ok(Invocation::Help(topic)) => ErrorReport::new(ExitCode::Ok, args::render_help(topic)),
        Ok(Invocation::Run(_)) => {
            // TASK-33.1-2（#136）で各工程を下位層へ接続して置き換える。
            eprintln!("fandhe-edge: stage execution is not implemented yet (TASK-33.1-2)");
            ErrorReport::new(
                ExitCode::RuntimeError,
                "stage not implemented yet (TASK-33.1-2)",
            )
        }
        Err(e) => args::args_error_report(&e),
    };
    // 書き込み失敗時は追記・リトライせず runtime_error とする（output.rs の方針）。
    match write_error_report(&mut std::io::stdout().lock(), &report) {
        Ok(code) => code.into(),
        Err(_) => ExitCode::RuntimeError.into(),
    }
}
