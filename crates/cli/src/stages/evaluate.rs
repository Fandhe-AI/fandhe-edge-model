//! `evaluate` 工程（REQ-17・REQ-21・REQ-27・REQ-33・TASK-33.1-2・#136）。
//!
//! # 実装済みの範囲と未実装の範囲
//!
//! 最初に評価データの状態を判定する（[`crate::stage_output::evaluate_start`]。候補が学習済みか
//! の確認より前。#140 の申し送り）。
//!
//! - 評価データ未定義: `status:"skipped"`・exit 0（評価済みを装わない。REQ-17）
//! - 評価データあり: 凍結記録とのハッシュ一致を確認する（不一致は `invalid_input`。fail-closed）。
//!   一致した後の**評価本体（凍結データでの推論・指標算出・評価完了の結果 JSON 型）は
//!   未実装**で、`runtime_error`（70）を返す。評価器（`fandhe-edge-eval`）の CLI 結線は
//!   別 TASK の範囲（REQ-24〜27）。評価が完了したと誤認させないため exit 0 にしない。

use std::path::Path;

use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::EvaluateReport;
use fandhe_edge_data::eval_freeze::EvalDataState;

use crate::args::EvaluateArgs;
use crate::project::Project;
use crate::stage_output::{EvaluateStart, evaluate_start};

use super::inspect::load_evaluation_bytes;

/// `evaluate` を実行する。
///
/// # Errors
/// 凍結記録との不一致は `invalid_input`（64）。評価データありの評価本体は未実装のため
/// `runtime_error`（70）。
pub fn run(args: &EvaluateArgs, cwd: &Path) -> Result<EvaluateReport, ErrorReport> {
    let project = Project::open(cwd, &args.project_dir)?;
    // 評価データの有無・凍結記録とのハッシュ一致を先に判定する（内部で `evaluate_start` を通す）。
    if load_evaluation_bytes(&project)?.is_none() {
        return match evaluate_start(&EvalDataState::NotProvided, b"")? {
            EvaluateStart::Skipped(report) => Ok(report),
            EvaluateStart::Proceed(_) => Err(ErrorReport::new(
                ExitCode::RuntimeError,
                "unexpected evaluation state",
            )),
        };
    }
    // TODO(評価器の結線 TASK・REQ-24〜27): 凍結した評価データを推論関数（input のみ）へ渡し、
    // 指標を算出して評価完了の結果を返す。それまでは評価済みを装わない。
    Err(ErrorReport::new(
        ExitCode::RuntimeError,
        "evaluation on frozen data is not implemented yet",
    ))
}
