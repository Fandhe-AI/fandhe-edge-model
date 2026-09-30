//! `select` 工程: 学習済みの候補から validation 正解率が最も高い候補を選ぶ
//! （REQ-18・REQ-27・REQ-33・TASK-33.1-2・#136）。
//!
//! # 手順
//!
//! 各候補の `request.json`・`result.json` を再検証つきで読み戻し（[`super::train::load_trained`]）、
//! 学習ジョブが返した validation 予測を、`train.jsonl` の正解ラベルと突き合わせて正解率を出す
//! （評価器を呼ぶだけで再実装しない。[`fandhe_edge_train::stage_files::validation_accuracy`]）。
//! 最高正解率の候補を [`select_best`]（同率は宣言順で先。乱数なし）で選び、`selection_record.json`
//! へ記録する。学習済みの候補が 1 件も無ければ `pending`（12）。
//!
//! 採点の前に `split.json` をデータから再現して照合し（[`super::train::verified_split`]）、各候補の
//! 保存済み `request.json` の validation（id・input）が分割記録の validation 全体と一致することを
//! 確認する（不一致は `invalid_input`。記録の改変による部分集合での選定を防ぐ。REQ-27）。
//!
//! # 未接続
//!
//! McNemar・Holm による選定結果の有意性判定（[`fandhe_edge_train::selection_significance`]）は
//! 本工程に**未接続**で、記録にも含めない（合否には使わない。REQ-25）。最終 test は使わない
//! （REQ-27。validation のみで選ぶ）。

use std::collections::BTreeMap;
use std::path::Path;

use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::SelectReport;
use fandhe_edge_data::split::Split;
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::search::{EvaluatedCandidate, SelectionDecision, select_best};
use fandhe_edge_train::stage_files::{SelectionRecord, validation_accuracy};

use crate::args::SelectArgs;
use crate::error_report::default_message;
use crate::project::{Project, SELECTION_FILE, fail, invalid, runtime};

use super::train::{load_trained, resolve_candidates, verified_split};

/// `select` を実行する。
///
/// # Errors
/// 学習済みの候補が無ければ `pending`（12）、既存の選定記録があれば `invalid_input`（64）、
/// 予測と入力の不一致・採点失敗は `runtime_error`（70）。
pub fn run(args: &SelectArgs, cwd: &Path) -> Result<SelectReport, ErrorReport> {
    let project = Project::open(cwd, &args.project_dir)?;
    if project.exists(SELECTION_FILE)? {
        return Err(crate::project::invalid("selection record already exists"));
    }
    let definition = project.load_definition()?;
    let records = project.load_records(&definition)?;
    let gold: BTreeMap<&str, &str> = records
        .iter()
        .map(|r| (r.id.as_str(), r.label_id.as_str()))
        .collect();
    // 分割記録をデータから再現して照合し、選定に使う validation 集合を記録側から決める。
    // 保存済みの `request.json` の validation を信用すると、記録の改変で任意の部分集合の
    // 正解率により候補を選べてしまう（REQ-27）。
    let split = verified_split(&project, &records)?;
    let expected_validation: BTreeMap<&str, &str> = records
        .iter()
        .filter(|r| split.by_record.get(&r.id) == Some(&Split::Validation))
        .map(|r| (r.id.as_str(), r.input.as_str()))
        .collect();
    let labels: Vec<&str> = definition.options().iter().map(|c| c.id.as_str()).collect();
    let candidates = resolve_candidates(&project, &definition, 0)?;

    let mut evaluated = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let Some((request, outcome)) = load_trained(&project, index)? else {
            continue;
        };
        let TrainOutcome::Ok(success) = &outcome else {
            continue;
        };
        let inputs = request
            .validation_inputs()
            .ok_or_else(|| runtime("train result has no validation inputs"))?;
        let predictions = success
            .validation_predictions()
            .ok_or_else(|| runtime("train result has no validation predictions"))?;
        // request の validation は、分割記録の validation 全体と id・input が完全に一致すること。
        let matches_split = inputs.len() == expected_validation.len()
            && inputs
                .iter()
                .all(|v| expected_validation.get(v.id()) == Some(&v.input()));
        if !matches_split {
            return Err(invalid("train request does not match the split record"));
        }
        let ids: Vec<&str> = inputs.iter().map(|v| v.id()).collect();
        let gold_labels = ids
            .iter()
            .map(|id| gold.get(id).copied())
            .collect::<Option<Vec<&str>>>()
            .ok_or_else(|| runtime("validation record is missing"))?;
        let accuracy = validation_accuracy(&labels, &ids, &gold_labels, predictions)
            .map_err(|_| runtime("cannot score validation predictions"))?;
        evaluated.push((index, candidate.candidate_id.as_str(), accuracy));
    }

    let inputs: Vec<EvaluatedCandidate<'_>> = evaluated
        .iter()
        .map(|(_, id, accuracy)| EvaluatedCandidate {
            candidate_id: id,
            accuracy: *accuracy,
        })
        .collect();
    let decision = select_best(&inputs).map_err(|_| runtime("cannot select a candidate"))?;
    let SelectionDecision::Selected {
        candidate_id,
        validation_accuracy: accuracy,
        rule,
        ..
    } = decision
    else {
        return Err(fail(ExitCode::Pending, default_message(ExitCode::Pending)));
    };
    let Some((index, _, _)) = evaluated.iter().find(|(_, id, _)| *id == candidate_id) else {
        return Err(runtime("selected candidate is not evaluated"));
    };
    let record = SelectionRecord {
        candidate_index: *index,
        candidate_id: candidate_id.clone(),
        rule,
        validation_correct: accuracy.correct,
        validation_total: accuracy.total,
    };
    let json = record
        .to_json_vec()
        .map_err(|_| runtime("cannot serialize selection record"))?;
    project.write_new(SELECTION_FILE, &json)?;
    Ok(SelectReport::new(*index, candidate_id))
}
