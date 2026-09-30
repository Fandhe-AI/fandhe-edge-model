//! CLI の `train`・`select`・`package` 工程が作業ディレクトリへ置くファイルの読み書き補助
//! （REQ-18・REQ-33・TASK-33.1-2・#136）。
//!
//! # 位置づけ
//!
//! `fandhe-edge-cli` は `serde_json` に依存しない（`.claude/rules/dependency-policy.md`）ため、
//! 学習ワーカー層が持つ JSON 境界（学習用データの JSONL・学習結果の保存・選定記録）の
//! 直列化と、候補ごとの validation 正解率の算出をここへ置く。評価ロジックは評価器
//! （`fandhe-edge-eval`）を呼ぶだけで再実装しない（評価器は TASK-24.1 の 1 つだけ）。
//!
//! # 入出力
//!
//! - [`trainer_jsonl`]: trainer が受け付ける `{"input","label"}`（2 キー限定）の JSONL を作る
//! - [`outcome_json_vec`]: 学習結果の保存用 JSON（保存後も
//!   [`crate::result::TrainOutcome::from_worker_stdout`] で再検証つきで読み戻す）
//! - [`validation_accuracy`]: 学習ジョブが返した validation 予測と正解ラベルから正解率を出す
//! - [`SelectionRecord`]: `select` の記録（`package` が選定候補を読み戻す）。有意性判定
//!   （[`crate::selection_significance`]）は本記録に含めていない（未接続）
//!
//! エラーはデータ本文・ラベル・パスを含まない固定の列挙値で返す（`security.md`）。

use fandhe_edge_eval::metrics::{self, EvalRecord, Outcome, Ratio};
use serde::{Deserialize, Serialize};

use crate::result::{TrainOutcome, ValidationPrediction, ValidationPredictionStatus};

/// 保存・採点の失敗（内容を含まない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageFileError {
    /// JSON の直列化に失敗した。
    Serialize,
    /// 保存済みの記録を読めない・妥当でない。
    Malformed,
    /// 予測の件数・`id` 順序が入力と一致しない、または採点に失敗した。
    Scoring,
}

impl std::fmt::Display for StageFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageFileError::Serialize => write!(f, "failed to serialize stage file"),
            StageFileError::Malformed => write!(f, "stage file is malformed"),
            StageFileError::Scoring => write!(f, "failed to score validation predictions"),
        }
    }
}

impl std::error::Error for StageFileError {}

/// `(input, label)` の列から trainer 形式（1 行 `{"input","label"}`）の JSONL を作る。
///
/// # Errors
/// 直列化に失敗した場合（実務上は起こらない）。
pub fn trainer_jsonl<'a, I>(rows: I) -> Result<Vec<u8>, StageFileError>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let mut out = Vec::new();
    for (input, label) in rows {
        let line = serde_json::to_vec(&serde_json::json!({"input": input, "label": label}))
            .map_err(|_| StageFileError::Serialize)?;
        out.extend_from_slice(&line);
        out.push(b'\n');
    }
    Ok(out)
}

/// 学習結果を保存用の JSON 1 行（末尾改行つき）へ直列化する（ワーカーの標準出力と同じ形）。
///
/// # Errors
/// 直列化に失敗した場合（実務上は起こらない）。
pub fn outcome_json_vec(outcome: &TrainOutcome) -> Result<Vec<u8>, StageFileError> {
    let mut bytes = serde_json::to_vec(outcome).map_err(|_| StageFileError::Serialize)?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// 学習ジョブが返した validation 予測（入力順）を、正解ラベル `gold`（同じ順序）と突き合わせて
/// 全体正解率を返す。件数・`id` の順序が入力 `ids` と一致しなければ [`StageFileError::Scoring`]。
///
/// abstain・error・ラベル集合外は不正解として数える（評価器の規則。REQ-24）。
///
/// # Errors
/// 件数・`id` 順序の不一致、または評価器がエラーを返した場合。
pub fn validation_accuracy(
    labels: &[&str],
    ids: &[&str],
    gold: &[&str],
    predictions: &[ValidationPrediction],
) -> Result<Ratio, StageFileError> {
    if predictions.len() != ids.len() || gold.len() != ids.len() {
        return Err(StageFileError::Scoring);
    }
    let mut outcomes = Vec::with_capacity(predictions.len());
    for (prediction, expected_id) in predictions.iter().zip(ids.iter()) {
        if prediction.id() != *expected_id {
            return Err(StageFileError::Scoring);
        }
        outcomes.push(match (prediction.status(), prediction.predicted_label()) {
            (ValidationPredictionStatus::Ok, Some(label)) => Outcome::Label(label.to_string()),
            (ValidationPredictionStatus::Abstain, _) => Outcome::Abstain,
            (ValidationPredictionStatus::Error | ValidationPredictionStatus::Ok, _) => {
                Outcome::Error
            }
        });
    }
    let records: Vec<EvalRecord<'_>> = gold
        .iter()
        .zip(outcomes.iter())
        .map(|(&gold, outcome)| EvalRecord { gold, outcome })
        .collect();
    let m =
        metrics::evaluate_single_select(labels, &records).map_err(|_| StageFileError::Scoring)?;
    Ok(m.accuracy.overall)
}

/// `select` の記録（`selection_record.json`）。`package` が選定候補を読み戻す。
///
/// 有意性判定（McNemar・Holm）は含まない（未接続。合否には使わない）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionRecord {
    /// 選定した候補の添字（`--candidate` と同じ）。
    pub candidate_index: usize,
    /// 選定した候補 ID。
    pub candidate_id: String,
    /// 選定規則名。
    pub rule: String,
    /// 選定候補の validation 正解数。
    pub validation_correct: u64,
    /// validation の件数。
    pub validation_total: u64,
}

impl SelectionRecord {
    /// JSON 1 行（末尾改行つき）へ直列化する。
    ///
    /// # Errors
    /// 直列化に失敗した場合。
    pub fn to_json_vec(&self) -> Result<Vec<u8>, StageFileError> {
        let mut bytes = serde_json::to_vec(self).map_err(|_| StageFileError::Serialize)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// 保存済みの記録を読む（未知キー・型違いは拒否）。
    ///
    /// # Errors
    /// JSON として不正、または形が合わない場合。
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, StageFileError> {
        serde_json::from_slice(bytes).map_err(|_| StageFileError::Malformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-18: trainer 形式の JSONL は `{"input","label"}` の 2 キーだけを 1 行ずつ持つ。
    #[test]
    fn req18_trainer_jsonl_has_only_input_and_label() {
        let bytes = trainer_jsonl([("a \"q\"", "x"), ("b", "y")]).expect("jsonl");
        assert_eq!(
            String::from_utf8(bytes).expect("utf8"),
            "{\"input\":\"a \\\"q\\\"\",\"label\":\"x\"}\n{\"input\":\"b\",\"label\":\"y\"}\n"
        );
    }

    /// REQ-18: 選定記録は往復でき、未知キーは拒否する。
    #[test]
    fn req18_selection_record_round_trips_and_rejects_unknown_keys() {
        let r = SelectionRecord {
            candidate_index: 1,
            candidate_id: "c3".to_string(),
            rule: "validation_accuracy_desc_then_candidate_order".to_string(),
            validation_correct: 3,
            validation_total: 4,
        };
        let bytes = r.to_json_vec().expect("json");
        assert_eq!(SelectionRecord::from_json_slice(&bytes), Ok(r));
        assert_eq!(
            SelectionRecord::from_json_slice(br#"{"candidate_index":0,"extra":1}"#),
            Err(StageFileError::Malformed)
        );
    }
}
