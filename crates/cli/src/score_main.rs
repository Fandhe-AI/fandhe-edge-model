//! `fandhe-edge-score` バイナリの入口（PoC-26 用。7 工程の契約外。薄い配線のみ。REQ-41・TASK-41.1・#445）。
//!
//! 引数解析・採点・出力・時間上限は lib 側（[`fandhe_edge_cli::score_predictions`]）。成功時は JSON 1 行を
//! stdout へ書き exit 0、失敗時は既存 CLI と同じ `ErrorReport` の JSON 1 行と終了コード 7 種。

use fandhe_edge_cli::output::write_error_report;
use fandhe_edge_cli::score_predictions::{emit, parse_args};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};

fn main() -> std::process::ExitCode {
    let mut out = std::io::stdout().lock();
    let written = match std::env::current_dir() {
        Ok(cwd) => match parse_args(std::env::args_os().skip(1)) {
            Ok(args) => emit(&mut out, &args, &cwd),
            Err(report) => write_error_report(&mut out, &report),
        },
        Err(_) => write_error_report(
            &mut out,
            &ErrorReport::new(ExitCode::RuntimeError, "cannot resolve working directory"),
        ),
    };
    // 書き込み失敗は runtime_error（output.rs と同じ方針。追記・リトライしない）。
    written.unwrap_or(ExitCode::RuntimeError).into()
}
