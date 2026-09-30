//! 7 工程（`register → inspect → train → evaluate → select → package → infer`）から下位層
//! （core・data・train・eval・runtime・guard）への接続（REQ-33・TASK-33.1-2・#136）。
//!
//! # 呼び出し文脈
//!
//! `main.rs` が引数解析（[`crate::args::parse`]）に成功した後、[`run`] を 1 回呼ぶ。各工程は
//! 業務ロジックを持たず、下位層の関数を呼んで結果を JSON 1 つ（`infer --input-file` のみ
//! 1 行 1 JSON。REQ-33）へ写す薄い配線に留める。工程間の受け渡しは `--project-dir` 配下の
//! ファイルで行う（規約は [`crate::project`]）。
//!
//! # 完走の範囲（実装済みを装わない）
//!
//! - `register`・`inspect`・`train`・`select`・`package`・`infer` は下位層へ接続済み
//! - `evaluate` は評価データ未定義なら `skipped`（exit 0）。**評価データありの評価本体
//!   （凍結データでの推論・指標算出・評価完了の結果型）は評価器の結線 TASK で未実装**で、
//!   `runtime_error`（70）を返す（評価済みを装わない。[`evaluate`] 参照）
//! - `package` は容量（REQ-30）のみ計測し、p95（REQ-31）と合否基準は未接続で
//!   `judgment:null`・`acceptance_defined:false`（`pass` を出さない）
//! - `infer --out` は未実装（`runtime_error`）
//!
//! # 未検証の項目（「検証済み」ではない）
//!
//! 外部台帳による sha256 完全性（#168・TASK-39.3-2）、版管理台帳による前版への復帰（#174・
//! TASK-39.6-1。`infer` の `kind_version` 許可リスト自体は接続済み）、読み込み上限の正式値（#172・TASK-39.5-3）は未検証・暫定値のまま。

use std::io::{self, Write};
use std::path::Path;

use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::{
    EvaluateReport, InspectStageReport, RegisterReport, SelectReport, TrainReport,
};
use fandhe_edge_runtime::package_outcome::PackageOutcome;

use crate::args::Command;
use crate::error_report::emit_error_report;
use crate::output::write_stage_line;
use crate::stage_output::{emit_evaluate_skipped, emit_package_outcome};

pub mod evaluate;
pub mod infer;
pub mod inspect;
pub mod package;
pub mod register;
pub mod select;
pub mod train;

/// `infer` 以外の工程が成功したときの結果（stdout の JSON 1 つへ写す）。
enum Done {
    Register(RegisterReport),
    Inspect(InspectStageReport),
    Train(TrainReport),
    Evaluate(EvaluateReport),
    Select(SelectReport),
    Package(PackageOutcome),
}

/// `command` を実行し、結果（または `ErrorReport`）を `out` へ書いて終了コードを返す。
///
/// `cwd` は経路の閉じ込めの基準ディレクトリ（`--definition`・`--project-dir`・`--package`・
/// `--input-file` は cwd 配下に限る。REQ-39）。
///
/// # Errors
/// `out` への書き込み失敗（呼び出し側は exit 70 に写し、追加の出力をしない）。
pub fn run<W: Write>(out: &mut W, command: &Command, cwd: &Path) -> io::Result<ExitCode> {
    let result: Result<Done, ErrorReport> = match command {
        Command::Register(args) => register::run(args, cwd).map(Done::Register),
        Command::Inspect(args) => inspect::run(args, cwd).map(Done::Inspect),
        Command::Train(args) => train::run(args, cwd).map(Done::Train),
        Command::Evaluate(args) => evaluate::run(args, cwd).map(Done::Evaluate),
        Command::Select(args) => select::run(args, cwd).map(Done::Select),
        Command::Package(args) => package::run(args, cwd).map(Done::Package),
        Command::Infer(args) => return infer::run(out, args, cwd),
    };
    match result {
        Ok(Done::Register(r)) => write_stage_line(out, r.to_json_line()),
        Ok(Done::Inspect(r)) => write_stage_line(out, r.to_json_line()),
        Ok(Done::Train(r)) => write_stage_line(out, r.to_json_line()),
        Ok(Done::Select(r)) => write_stage_line(out, r.to_json_line()),
        Ok(Done::Evaluate(r)) => emit_evaluate_skipped(out, &r),
        Ok(Done::Package(outcome)) => emit_package_outcome(out, &outcome),
        Err(report) => emit_error_report(out, &report),
    }
}
