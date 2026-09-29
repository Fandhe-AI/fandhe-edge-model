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
//! - `infer`: 解析成功後、`--package` と `artifact.json` の `onnx_file` を経路ガード
//!   （`infer_guard`。REQ-39・#159）へ通し、拒否は `invalid_input`（64）等の JSON 1 行で
//!   終える。ガードを通過した後の推論本体は #136 で未接続のため、下のスタブ（70）へ進む。
//! - 解析に成功したコマンド: 下位層への接続は TASK-33.1-2（#136）の範囲の
//!   ため、完走を装わず `runtime_error`（exit 70）で未実装を返す。

use fandhe_edge_cli::args::{self, Command, Invocation};
use fandhe_edge_cli::infer_guard::guard_infer_paths;
use fandhe_edge_cli::log::StderrLog;
use fandhe_edge_cli::output::write_error_report;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};

fn main() -> std::process::ExitCode {
    let report = match args::parse(std::env::args_os().skip(1)) {
        Ok(Invocation::Help(topic)) => ErrorReport::new(ExitCode::Ok, args::render_help(topic)),
        Ok(Invocation::Run(Command::Infer(infer_args))) => {
            // 経路ガードを通過するまで何も読まない（REQ-39）。workspace はカレントディレクトリ。
            match std::env::current_dir() {
                Err(_) => {
                    ErrorReport::new(ExitCode::RuntimeError, "cannot resolve working directory")
                }
                Ok(cwd) => match guard_infer_paths(&cwd, &infer_args) {
                    Err(report) => report,
                    // TASK-33.1-2（#136）で、開いた `File` から推論へ接続して置き換える。
                    Ok(_guarded) => ErrorReport::new(
                        ExitCode::RuntimeError,
                        "stage not implemented yet (TASK-33.1-2)",
                    ),
                },
            }
        }
        Ok(Invocation::Run(_)) => {
            // TASK-33.1-2（#136）で各工程を下位層へ接続して置き換える。
            // ログは stderr 専用の経路で出す（stdout の JSON と混ぜない。TASK-33.2-2）。
            // #136 はこの分岐から各工程を呼び、package なら
            // `stage_output::emit_package_outcome` へ結果を渡す。
            StderrLog::new(std::io::stderr().lock())
                .info_top("stage execution is not implemented yet (TASK-33.1-2)");
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
