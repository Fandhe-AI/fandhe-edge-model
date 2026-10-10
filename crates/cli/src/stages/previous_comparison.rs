//! `evaluate --previous-project-dir` の旧モデルとの比較（REQ-26・TASK-26.1・26.2・#488・#489。
//! REQ-17・REQ-24・REQ-27・REQ-39）。
//!
//! # 役割
//!
//! 旧モデルは**再推論しない**。旧プロジェクトが 1 回限りの適用で保存した
//! `candidates/<i>/evaluation_predictions.jsonl`（#445）を、旧の評価記録の `predictions_sha256`・
//! `evaluation_sha256`・`definition_sha256` で束縛して読むだけ（照合と読み込みは `fandhe-edge-score` と
//! 共有する [`check_record_binding`]・[`classify_predictions`]。再実装しない）。旧モデルの重み・ONNX は
//! 読まない。旧プロジェクトは `Project::open`（cwd 配下・保持 fd 起点）で開き、書き込まない。新旧どちらの
//! 台帳（`final_test_ledger/`）にも触れない（REQ-27 の 1 回限りと衝突しない）。
//!
//! # 手順（`evaluate` の `apply_once` の前に [`prepare_previous`]、`finish` の中で [`PreparedPrevious::compare`]）
//!
//! 1. 旧の `selection_record.json` → 選定候補 `i` → `candidates/i/evaluation_record.json`
//!    （上限 `MAX_EVALUATION_RECORD_BYTES`。無ければ評価未完了として拒否）
//! 2. 記録の束縛: `predictions_sha256` 必須、`definition_sha256` が旧の定義の正準化と一致、旧の凍結記録と
//!    評価データが一致（`load_frozen_evaluation`）し記録とも一致、予測ファイル（上限 `MAX_PROJECT_FILE_BYTES`）の
//!    sha256 が一致、行数 = `total`
//! 3. 比較対象: 旧の `evaluation_sha256` が新の凍結 sha256 と同じなら `same`、異なれば `common_subset`。
//!    どちらも id・input・正解ラベルがすべて一致するレコードを共通レコードとする（同じバイト列なら全件）
//! 4. 正誤は評価器の `significance::correctness`（`is_correct` の公開入口）で各側のラベル空間で求め、
//!    `regression::regression_report` で 2×2 と前提を得る（`n_common == 0` は件数 `null`、前提は
//!    `compare_label_sets`）。`merges` の写像は実装しない
//!
//! 違反は `invalid_input`、上限超過は `limit_exceeded`（いずれも適用権を取る前）。比較は記録・報告のみで、
//! 終了コード・合否・`package` の照合に使わない。p 値・有意性判定は出さない。推論関数へ渡す情報は増やさない
//! （新側の正誤は適用済みの `Outcome` から求める。REQ-27）。

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::evaluation_record::{
    ComparisonEvaluationData, ComparisonPremiseKind, EvaluationRecord, MAX_EVALUATION_RECORD_BYTES,
    PreviousComparisonRecord, PreviousModelRecord, RegressionCountsRecord,
};
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::stage_report::{
    EvaluateComparison, EvaluateInterval, EvaluateRegressionCounts,
};
use fandhe_edge_data::eval_freeze::FreezeRecord;
use fandhe_edge_data::inspect::ValidRecord;
use fandhe_edge_eval::metrics::{EvalRecord, Outcome};
use fandhe_edge_eval::regression::{
    ComparisonPremise, IdentifiedCorrectness, RegressionCounts, compare_label_sets,
    regression_report,
};
use fandhe_edge_eval::significance::correctness;
use fandhe_edge_eval::wilson::WilsonInterval;
use fandhe_edge_train::search::MAX_CANDIDATE_ID_BYTES;
use fandhe_edge_train::stage_files::SelectionRecord;

use crate::error_report::ToErrorReport;
use crate::project::{
    EVALUATION_PREDICTIONS_FILE, EVALUATION_RECORD_FILE, MAX_PROJECT_FILE_BYTES, Project,
    SELECTION_FILE, invalid, runtime,
};
use crate::score_predictions::{check_record_binding, classify_predictions, gold_jsonl};

use super::evaluate::evaluation_records;
use super::inspect::load_frozen_evaluation;
use super::train::candidate_rel;

/// 旧・新の共通レコード 1 件（新の評価データの行位置・id・旧の正誤）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct CommonRow {
    current_index: usize,
    id: String,
    previous_correct: bool,
}

/// 適用前に確定した旧側の材料（[`prepare_previous`] が作り、`finish` で [`Self::compare`] する）。
#[derive(Debug)]
pub(super) struct PreparedPrevious {
    previous: PreviousModelRecord,
    previous_labels: Vec<String>,
    evaluation_data: ComparisonEvaluationData,
    n_previous_only: u64,
    n_current_only: u64,
    common: Vec<CommonRow>,
}

/// 旧プロジェクトを読み、記録との束縛を照合して、比較対象（共通レコードと旧の正誤）を確定する。
///
/// `current_records` は新の凍結評価データを `inspect_bytes` で分解したもの（評価器へ渡すのと同じ行順）。
/// `apply_once` の前に呼ぶ（失敗しても適用権を使わない。REQ-27）。
///
/// # Errors
/// 旧プロジェクトが cwd 外・未選定・未評価・記録と不一致・予測の形式不正は `invalid_input`、
/// 読み込み上限の超過は `limit_exceeded`。
pub(super) fn prepare_previous(
    cwd: &Path,
    previous_dir: &Path,
    current_definition: &Definition,
    current_freeze: &FreezeRecord,
    current_records: &[ValidRecord],
) -> Result<PreparedPrevious, ErrorReport> {
    let old = Project::open(cwd, previous_dir)?;
    let old_definition = old.load_definition()?;
    let Some(selection) = old.read_optional(SELECTION_FILE, 64 * 1024)? else {
        return Err(invalid(
            "previous candidate selection has not been recorded",
        ));
    };
    let selection = SelectionRecord::from_json_slice(&selection)
        .map_err(|_| invalid("previous selection record is invalid"))?;
    let index = selection.candidate_index;
    let Some(record_bytes) = old.read_optional(
        candidate_rel(index).join(EVALUATION_RECORD_FILE),
        MAX_EVALUATION_RECORD_BYTES,
    )?
    else {
        return Err(invalid("previous candidate has not been evaluated"));
    };
    let record = EvaluationRecord::from_json_slice(&record_bytes)
        .map_err(|_| invalid("previous evaluation record is malformed"))?;
    check_previous_record_shape(&record, index)?;
    if record.predictions_sha256.is_none() {
        return Err(invalid("previous evaluation record has no predictions"));
    }
    let old_definition_sha256 = old_definition
        .canonical_hash()
        .map_err(|_| runtime("cannot hash definition"))?
        .to_hex();
    if record.definition_sha256 != old_definition_sha256 {
        return Err(invalid(
            "previous evaluation record does not match its definition",
        ));
    }
    // 旧の凍結記録と評価データの一致（不一致は `load_frozen_evaluation` が `invalid_input` で止める）。
    let Some((old_freeze, old_eval_bytes)) = load_frozen_evaluation(&old)? else {
        return Err(invalid("previous project has no evaluation data"));
    };
    let Some(prediction_bytes) = old.read_optional(
        candidate_rel(index).join(EVALUATION_PREDICTIONS_FILE),
        MAX_PROJECT_FILE_BYTES,
    )?
    else {
        return Err(invalid("previous evaluation predictions are missing"));
    };
    check_record_binding(
        &record,
        &Sha256Digest::of_bytes(&prediction_bytes).to_hex(),
        &old_freeze.sha256().to_hex(),
        old_freeze.byte_len(),
        &old_definition_sha256,
    )?;
    let old_records =
        evaluation_records(&old_eval_bytes, &old_definition).map_err(|e| e.to_error_report())?;
    let old_labels: Vec<String> = old_definition
        .options()
        .iter()
        .map(|c| c.id.clone())
        .collect();
    let label_set: BTreeSet<String> = old_labels.iter().cloned().collect();
    let classified = classify_predictions(
        &prediction_bytes,
        &gold_jsonl(&old_records),
        &label_set,
        "previous",
    )?;
    if u64::try_from(classified.outcomes.len()).ok() != Some(record.total) {
        return Err(invalid(
            "previous predictions do not match the evaluation record",
        ));
    }
    let label_refs: Vec<&str> = old_labels.iter().map(String::as_str).collect();
    let eval_records: Vec<EvalRecord<'_>> = classified
        .golds
        .iter()
        .zip(&classified.outcomes)
        .map(|(gold, outcome)| EvalRecord { gold, outcome })
        .collect();
    let previous_correct = correctness(&label_refs, &eval_records)
        .map_err(|_| invalid("previous predictions are invalid"))?;
    let correct_by_id: HashMap<&str, bool> = classified
        .ids
        .iter()
        .map(String::as_str)
        .zip(previous_correct)
        .collect();

    let common = common_rows(&old_records, current_records, &correct_by_id)?;
    let evaluation_data = if record.evaluation_sha256 == current_freeze.sha256().to_hex() {
        ComparisonEvaluationData::Same
    } else {
        ComparisonEvaluationData::CommonSubset
    };
    let n_common = common.len();
    let prepared = PreparedPrevious {
        previous: PreviousModelRecord {
            candidate_id: record.candidate_id,
            onnx_sha256: record.onnx_sha256,
            definition_sha256: record.definition_sha256,
            evaluation_sha256: record.evaluation_sha256,
        },
        previous_labels: old_labels,
        evaluation_data,
        n_previous_only: to_u64(old_records.len().saturating_sub(n_common))?,
        n_current_only: to_u64(current_records.len().saturating_sub(n_common))?,
        common,
    };
    // 比較の入力（ラベル・id・件数）が評価器に受理されることを適用前に確かめる。適用後に変わるのは
    // 新側の正誤の真偽値だけで、それは評価器の検査対象にならない（適用後に比較で失敗しない）。
    let current_labels: Vec<&str> = current_definition
        .options()
        .iter()
        .map(|c| c.id.as_str())
        .collect();
    let dry_run = vec![false; current_records.len()];
    prepared
        .compare(&current_labels, &dry_run)
        .map_err(|_| invalid("previous comparison is not possible"))?;
    Ok(prepared)
}

/// 旧の評価記録のうち、新しい評価記録へ写す値の形を確かめる（記録の上限内に収めるため。REQ-39）。
fn check_previous_record_shape(record: &EvaluationRecord, index: usize) -> Result<(), ErrorReport> {
    let id = &record.candidate_id;
    let id_ok =
        !id.is_empty() && id.len() <= MAX_CANDIDATE_ID_BYTES && !id.chars().any(char::is_control);
    if record.candidate_index != index
        || !id_ok
        || record.onnx_sha256.parse::<Sha256Digest>().is_err()
    {
        return Err(invalid("previous evaluation record is malformed"));
    }
    Ok(())
}

fn to_u64(n: usize) -> Result<u64, ErrorReport> {
    u64::try_from(n).map_err(|_| runtime("count is out of range"))
}

/// 新の評価データの行順に、id・input・正解ラベルがすべて旧と一致するレコードを集める（REQ-26・#488）。
fn common_rows(
    old_records: &[ValidRecord],
    current_records: &[ValidRecord],
    correct_by_id: &HashMap<&str, bool>,
) -> Result<Vec<CommonRow>, ErrorReport> {
    let old_by_id: HashMap<&str, &ValidRecord> =
        old_records.iter().map(|r| (r.id.as_str(), r)).collect();
    let mut common = Vec::new();
    for (current_index, record) in current_records.iter().enumerate() {
        let Some(old) = old_by_id.get(record.id.as_str()) else {
            continue;
        };
        if old.input != record.input || old.label_id != record.label_id {
            continue;
        }
        let previous_correct = *correct_by_id
            .get(record.id.as_str())
            .ok_or_else(|| runtime("previous predictions are incomplete"))?;
        common.push(CommonRow {
            current_index,
            id: record.id.clone(),
            previous_correct,
        });
    }
    Ok(common)
}

impl PreparedPrevious {
    /// 新側の正誤（新の評価データの行順。評価器の `correctness` で求めたもの）と突き合わせ、stdout と
    /// 評価記録の比較欄を作る（`finish` の中で呼ぶ）。
    ///
    /// # Errors
    /// 評価器の比較が失敗したら `runtime_error`（適用前の試し実行で入力は検査済み）。
    pub(super) fn compare(
        &self,
        current_labels: &[&str],
        current_correct: &[bool],
    ) -> Result<(EvaluateComparison, PreviousComparisonRecord), ErrorReport> {
        let fail = || runtime("previous comparison failed");
        let previous_labels: Vec<&str> = self.previous_labels.iter().map(String::as_str).collect();
        let (premise, counts) = if self.common.is_empty() {
            let premise =
                compare_label_sets(&previous_labels, current_labels).map_err(|_| fail())?;
            (premise, None)
        } else {
            let previous: Vec<IdentifiedCorrectness<'_>> = self
                .common
                .iter()
                .map(|c| IdentifiedCorrectness {
                    id: &c.id,
                    correct: c.previous_correct,
                })
                .collect();
            let current = self
                .common
                .iter()
                .map(|c| {
                    current_correct
                        .get(c.current_index)
                        .map(|correct| IdentifiedCorrectness {
                            id: &c.id,
                            correct: *correct,
                        })
                        .ok_or_else(fail)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let report = regression_report(&previous_labels, current_labels, &previous, &current)
                .map_err(|_| fail())?;
            (report.premise().clone(), Some(*report.counts()))
        };
        let (premise_kind, removed_labels, added_labels) = match premise {
            ComparisonPremise::SameLabelSet => {
                (ComparisonPremiseKind::SameLabelSet, Vec::new(), Vec::new())
            }
            ComparisonPremise::LabelSetDiffers { removed, added } => {
                (ComparisonPremiseKind::LabelSetDiffers, removed, added)
            }
            _ => return Err(fail()),
        };
        let n_common = to_u64(self.common.len())?;
        let report_counts = counts.as_ref().map(stdout_counts).transpose()?;
        let comparison = EvaluateComparison {
            previous: self.previous.clone(),
            premise: premise_kind,
            removed_labels,
            added_labels,
            evaluation_data: self.evaluation_data,
            n_common,
            n_previous_only: self.n_previous_only,
            n_current_only: self.n_current_only,
            counts: report_counts,
        };
        let record = PreviousComparisonRecord {
            previous: self.previous.clone(),
            premise: premise_kind,
            evaluation_data: self.evaluation_data,
            n_common,
            counts: counts.as_ref().map(|c| RegressionCountsRecord {
                n: c.n(),
                both_correct: c.both_correct(),
                correct_to_incorrect: c.correct_to_incorrect(),
                incorrect_to_correct: c.incorrect_to_correct(),
                both_wrong: c.both_wrong(),
            }),
        };
        Ok((comparison, record))
    }
}

/// 評価器の件数を stdout の形へ写す（遷移率の Wilson 95% 区間を付ける）。
fn stdout_counts(c: &RegressionCounts) -> Result<EvaluateRegressionCounts, ErrorReport> {
    let interval = |w: Option<WilsonInterval>| {
        w.map(|w| EvaluateInterval {
            lo: w.lo(),
            hi: w.hi(),
        })
        .ok_or_else(|| runtime("previous comparison failed"))
    };
    Ok(EvaluateRegressionCounts {
        n: c.n(),
        both_correct: c.both_correct(),
        correct_to_incorrect: c.correct_to_incorrect(),
        incorrect_to_correct: c.incorrect_to_correct(),
        both_wrong: c.both_wrong(),
        correct_to_incorrect_ci95: interval(c.correct_to_incorrect_ci95())?,
        incorrect_to_correct_ci95: interval(c.incorrect_to_correct_ci95())?,
    })
}

/// 適用済みの予測（新の評価データの行順）から、評価器の正誤規則で新側の正誤を求める。
///
/// # Errors
/// 評価器が拒否したら `runtime_error`。
pub(super) fn current_correctness(
    labels: &[&str],
    golds: &[String],
    outcomes: &[Outcome],
) -> Result<Vec<bool>, ErrorReport> {
    let records: Vec<EvalRecord<'_>> = golds
        .iter()
        .zip(outcomes)
        .map(|(gold, outcome)| EvalRecord { gold, outcome })
        .collect();
    correctness(labels, &records).map_err(|_| runtime("previous comparison failed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared(previous_labels: &[&str], previous_correct: &[bool]) -> PreparedPrevious {
        PreparedPrevious {
            previous: PreviousModelRecord {
                candidate_id: "c1".to_string(),
                onnx_sha256: "1".repeat(64),
                definition_sha256: "2".repeat(64),
                evaluation_sha256: "3".repeat(64),
            },
            previous_labels: previous_labels.iter().map(|l| l.to_string()).collect(),
            evaluation_data: ComparisonEvaluationData::Same,
            n_previous_only: 0,
            n_current_only: 0,
            common: previous_correct
                .iter()
                .enumerate()
                .map(|(i, c)| CommonRow {
                    current_index: i,
                    id: format!("e{i}"),
                    previous_correct: *c,
                })
                .collect(),
        }
    }

    fn close(actual: &EvaluateInterval, lo: f64, hi: f64) -> bool {
        (actual.lo - lo).abs() < 1e-9 && (actual.hi - hi).abs() < 1e-9
    }

    /// REQ-26・TASK-26.1・#488: 同一の評価データ・同一のラベル集合で、2×2 の件数が手計算と一致し、遷移率の
    /// Wilson 95% 区間（z = 1.96）が手計算と許容差 1e-9 で一致する。記録には区間を持たない件数だけが入る。
    #[test]
    fn req26_issue488_counts_and_wilson_match_hand_calculation() {
        // 旧・新の正誤: 両方正解 5・正解→不正解 2・不正解→正解 1・両方不正解 2（n = 10）。
        let previous = [
            true, true, true, true, true, true, true, false, false, false,
        ];
        let current = [
            true, true, true, true, true, false, false, true, false, false,
        ];
        let p = prepared(&["a", "b"], &previous);
        let (report, record) = p.compare(&["b", "a"], &current).expect("compare");
        assert_eq!(report.premise, ComparisonPremiseKind::SameLabelSet);
        assert!(report.removed_labels.is_empty() && report.added_labels.is_empty());
        assert_eq!(report.n_common, 10);
        let c = report.counts.expect("counts");
        assert_eq!(
            (
                c.n,
                c.both_correct,
                c.correct_to_incorrect,
                c.incorrect_to_correct,
                c.both_wrong
            ),
            (10, 5, 2, 1, 2)
        );
        // 2/10: p = 0.2、z² = 3.8416、分母 1.38416、中心 0.39208/1.38416、幅 (1.96/1.38416)·√(0.016+0.009604)。
        assert!(
            close(
                &c.correct_to_incorrect_ci95,
                0.056_680_947_980_693_314,
                0.509_843_153_279_276_5
            ),
            "{c:?}"
        );
        // 1/10: p = 0.1、中心 0.29208/1.38416、幅 (1.96/1.38416)·√(0.009+0.009604)。
        assert!(
            close(
                &c.incorrect_to_correct_ci95,
                0.017_875_749_515_721_13,
                0.404_156_385_497_572_1
            ),
            "{c:?}"
        );
        assert_eq!(
            record.counts,
            Some(RegressionCountsRecord {
                n: 10,
                both_correct: 5,
                correct_to_incorrect: 2,
                incorrect_to_correct: 1,
                both_wrong: 2,
            })
        );
        assert_eq!(record.n_common, 10);
        assert_eq!(record.evaluation_data, ComparisonEvaluationData::Same);
    }

    /// REQ-26・TASK-26.2・#489: ラベル集合が異なれば `label_set_differs` で、削除・追加のラベルを各側の
    /// 宣言順で返す（エラーにしない）。共通レコード 0 件では `counts` は `null`（0 で埋めない）。
    #[test]
    fn req26_issue489_label_set_differs_and_empty_common_has_null_counts() {
        let p = prepared(&["a", "b", "c"], &[true, false, true]);
        let (report, record) = p
            .compare(&["d", "a", "b"], &[false, false, true])
            .expect("compare");
        assert_eq!(report.premise, ComparisonPremiseKind::LabelSetDiffers);
        assert_eq!(report.removed_labels, vec!["c".to_string()]);
        assert_eq!(report.added_labels, vec!["d".to_string()]);
        let c = report.counts.expect("counts");
        assert_eq!(
            (
                c.both_correct,
                c.correct_to_incorrect,
                c.incorrect_to_correct,
                c.both_wrong
            ),
            (1, 1, 0, 1)
        );
        assert!(close(
            &c.incorrect_to_correct_ci95,
            0.0,
            0.561_506_080_449_017_7
        ));
        assert!(close(
            &c.correct_to_incorrect_ci95,
            0.061_490_315_276_160_556,
            0.792_345_044_873_512
        ));
        assert_eq!(record.premise, ComparisonPremiseKind::LabelSetDiffers);

        let empty = prepared(&["a", "b", "c"], &[]);
        let (report, record) = empty.compare(&["a", "b"], &[true]).expect("empty");
        assert_eq!(report.n_common, 0);
        assert_eq!(report.counts, None);
        assert_eq!(record.counts, None);
        assert_eq!(report.premise, ComparisonPremiseKind::LabelSetDiffers);
        assert_eq!(report.removed_labels, vec!["c".to_string()]);
    }

    fn record(id: &str, input: &str, label: &str) -> ValidRecord {
        ValidRecord {
            line: 1,
            id: id.to_string(),
            input: input.to_string(),
            label_id: label.to_string(),
            output_key: String::new(),
            output_original: String::new(),
            tags: None,
            group_id: None,
        }
    }

    /// REQ-26・REQ-17・#488: 評価データが異なるときの共通レコードは、id・input・正解ラベルがすべて一致する
    /// 行だけ（新の行順）。id が同じでも input・ラベルが違う行は含めない。
    #[test]
    fn req26_issue488_common_rows_require_same_id_input_and_label() {
        let old = [
            record("1", "x", "a"),
            record("2", "y", "b"),
            record("3", "z", "a"),
            record("4", "w", "b"),
        ];
        let current = [
            record("4", "w", "b"),
            record("2", "y-changed", "b"),
            record("3", "z", "b"),
            record("1", "x", "a"),
            record("5", "v", "a"),
        ];
        let correct: HashMap<&str, bool> =
            [("1", true), ("2", false), ("3", true), ("4", false)].into();
        let rows = common_rows(&old, &current, &correct).expect("rows");
        assert_eq!(
            rows,
            vec![
                CommonRow {
                    current_index: 0,
                    id: "4".to_string(),
                    previous_correct: false,
                },
                CommonRow {
                    current_index: 3,
                    id: "1".to_string(),
                    previous_correct: true,
                },
            ]
        );
    }
}
