//! `fandhe-edge` バイナリの入口（薄い配線のみ。REQ-33・TASK-33.1-1・TASK-33.1-2）。
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
//! - 解析に成功したコマンド: カレントディレクトリ（経路の閉じ込めの基準）とともに
//!   [`fandhe_edge_cli::stages::run`] へ渡す。7 工程の下位層への接続と、未接続の区間
//!   （`evaluate` の評価データありの本体等）は `stages` の doc を参照。
//!   `infer` の経路・形式のガード（REQ-39・#159）も `stages::infer` の中で通る。

use fandhe_edge_cli::args::{self, Invocation};
use fandhe_edge_cli::output::write_error_report;
use fandhe_edge_cli::stages;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};

fn main() -> std::process::ExitCode {
    let report = match args::parse(std::env::args_os().skip(1)) {
        Ok(Invocation::Help(topic)) => ErrorReport::new(ExitCode::Ok, args::render_help(topic)),
        Ok(Invocation::Run(command)) => {
            let Ok(cwd) = std::env::current_dir() else {
                return emit(&ErrorReport::new(
                    ExitCode::RuntimeError,
                    "cannot resolve working directory",
                ));
            };
            // 書き込み失敗時は追記・リトライせず runtime_error とする（output.rs の方針）。
            return match stages::run(&mut std::io::stdout().lock(), &command, &cwd) {
                Ok(code) => code.into(),
                Err(_) => ExitCode::RuntimeError.into(),
            };
        }
        Err(e) => args::args_error_report(&e),
    };
    emit(&report)
}

/// `report` を JSON 1 行で stdout へ書き、終了コードを返す。
fn emit(report: &ErrorReport) -> std::process::ExitCode {
    match write_error_report(&mut std::io::stdout().lock(), report) {
        Ok(code) => code.into(),
        Err(_) => ExitCode::RuntimeError.into(),
    }
}
