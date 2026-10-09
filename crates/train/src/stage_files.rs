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
//! - [`search_record_json_vec`]: `train --all` の探索記録 `search_record.json`（#483）
//! - [`outcome_json_vec`]: 学習結果の保存用 JSON（保存後も
//!   [`crate::result::TrainOutcome::from_worker_stdout`] で再検証つきで読み戻す）
//! - [`validation_accuracy`]: 学習ジョブが返した validation 予測と正解ラベルから正解率を出す
//! - [`SelectionRecord`]: `select` の記録（`package` が選定候補を読み戻す）。有意性判定
//!   （[`crate::selection_significance`] の結果）は定義に `baseline_comparison` があるときだけ持つ（#481）
//!
//! エラーはデータ本文・ラベル・パスを含まない固定の列挙値で返す（`security.md`）。

use fandhe_edge_core::evaluation_record::SelectionSignificanceRecord;
use fandhe_edge_eval::metrics::{self, EvalRecord, Outcome, Ratio};
use serde::{Deserialize, Serialize};

use crate::result::{TrainOutcome, ValidationPrediction, ValidationPredictionStatus};
use crate::search::SearchRecord;

/// 保存・採点の失敗（内容を含まない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageFileError {
    /// JSON の直列化に失敗した。
    Serialize,
    /// 保存済みの記録を読めない・妥当でない。
    Malformed,
    /// 予測の件数・`id` 順序が入力と一致しない、または採点に失敗した。
    Scoring,
    /// 生成する学習用データが上限（`core::limits::TRAINER_JSONL_MAX_BYTES`）を超える（REQ-39）。
    LimitExceeded,
}

impl std::fmt::Display for StageFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageFileError::Serialize => write!(f, "failed to serialize stage file"),
            StageFileError::Malformed => write!(f, "stage file is malformed"),
            StageFileError::Scoring => write!(f, "failed to score validation predictions"),
            StageFileError::LimitExceeded => write!(f, "stage file exceeds size limit"),
        }
    }
}

impl std::error::Error for StageFileError {}

/// `(input, label)` の列から trainer 形式（1 行 `{"input","label"}`）の JSONL を作る。
///
/// 生成量は `core::limits::TRAINER_JSONL_MAX_BYTES`（暫定。REQ-39）で打ち切る。学習前に呼ばれ、
/// 超過なら学習ワーカーを起動しない（CLI の `train` 工程が `limit_exceeded` に写す）。
///
/// # Errors
/// 直列化に失敗した場合（実務上は起こらない）、または生成量が上限を超えた場合（`LimitExceeded`）。
pub fn trainer_jsonl<'a, I>(rows: I) -> Result<Vec<u8>, StageFileError>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let limit = usize::try_from(fandhe_edge_core::limits::TRAINER_JSONL_MAX_BYTES)
        .map_err(|_| StageFileError::LimitExceeded)?;
    trainer_jsonl_with_limit(rows, limit)
}

/// [`trainer_jsonl`] の上限を指定できる版（上限ちょうどは成功、1 バイトでも超えれば
/// `LimitExceeded`。変換後の累積バイト数〔改行を含む〕を checked 演算で数える）。
pub(crate) fn trainer_jsonl_with_limit<'a, I>(
    rows: I,
    limit: usize,
) -> Result<Vec<u8>, StageFileError>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let mut out = Vec::new();
    for (input, label) in rows {
        let line = serde_json::to_vec(&serde_json::json!({"input": input, "label": label}))
            .map_err(|_| StageFileError::Serialize)?;
        let total = out
            .len()
            .checked_add(line.len())
            .and_then(|n| n.checked_add(1))
            .ok_or(StageFileError::LimitExceeded)?;
        if total > limit {
            return Err(StageFileError::LimitExceeded);
        }
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

/// `train --all` が書く探索記録 `search_record.json` の上限（バイト。読み書き共通。REQ-18・REQ-39・
/// TASK-18.2・#483）。候補数の上限（256）でも数十 KiB に収まる大きさで、超えるものは書かない。
pub const MAX_SEARCH_RECORD_BYTES: usize = 1024 * 1024;

/// 探索記録（[`SearchRecord`]）を保存用の JSON 1 行（末尾改行つき）へ直列化する。validation の予測本文は
/// [`SearchRecord`] の直列化が出さない（REQ-27・security.md）。
///
/// # Errors
/// 直列化に失敗した場合（`Serialize`）、または [`MAX_SEARCH_RECORD_BYTES`] を超える場合（`LimitExceeded`）。
pub fn search_record_json_vec(record: &SearchRecord) -> Result<Vec<u8>, StageFileError> {
    let mut bytes = serde_json::to_vec(record).map_err(|_| StageFileError::Serialize)?;
    bytes.push(b'\n');
    if bytes.len() > MAX_SEARCH_RECORD_BYTES {
        return Err(StageFileError::LimitExceeded);
    }
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
    if gold.len() != ids.len() {
        return Err(StageFileError::Scoring);
    }
    let outcomes = validation_outcomes(ids, predictions)?;
    let records: Vec<EvalRecord<'_>> = gold
        .iter()
        .zip(outcomes.iter())
        .map(|(&gold, outcome)| EvalRecord { gold, outcome })
        .collect();
    let m =
        metrics::evaluate_single_select(labels, &records).map_err(|_| StageFileError::Scoring)?;
    Ok(m.accuracy.overall)
}

/// 学習ジョブが返した validation 予測（入力順）を評価器の [`Outcome`] 列へ写す（[`validation_accuracy`] と
/// `select` の有意性判定〔#481〕が同じ写像を使う）。件数・`id` の順序が `ids` と一致しなければ
/// [`StageFileError::Scoring`]。
///
/// `ok` かつラベルありは [`Outcome::Label`]、`abstain` は [`Outcome::Abstain`]、それ以外は [`Outcome::Error`]。
///
/// # Errors
/// 件数・`id` 順序の不一致。
pub fn validation_outcomes(
    ids: &[&str],
    predictions: &[ValidationPrediction],
) -> Result<Vec<Outcome>, StageFileError> {
    if predictions.len() != ids.len() {
        return Err(StageFileError::Scoring);
    }
    predictions
        .iter()
        .zip(ids.iter())
        .map(|(prediction, expected_id)| {
            if prediction.id() != *expected_id {
                return Err(StageFileError::Scoring);
            }
            Ok(match (prediction.status(), prediction.predicted_label()) {
                (ValidationPredictionStatus::Ok, Some(label)) => Outcome::Label(label.to_string()),
                (ValidationPredictionStatus::Abstain, _) => Outcome::Abstain,
                (ValidationPredictionStatus::Error | ValidationPredictionStatus::Ok, _) => {
                    Outcome::Error
                }
            })
        })
        .collect()
}

/// `select` の記録（`selection_record.json`）。`package` が選定候補を読み戻す。
///
/// 有意性判定（McNemar・Holm）は定義に `baseline_comparison` があるときだけ `significance` に残す
/// （記録のみ。選定・合否には使わない。REQ-25・#481）。
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
    /// 語彙ファイル超過構成として選定対象から除外した候補（REQ-30・TASK-30.3・#125）。
    /// 除外が無いときはキーごと省略する（従来の記録と互換）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_candidates: Vec<ExcludedCandidate>,
    /// 選定候補の下限基準に対する有意性判定（validation のみ・Holm 補正後。REQ-18・REQ-25・
    /// TASK-18.3・#481）。定義に `baseline_comparison` が無いときはキーごと省略する（従来の記録と互換）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub significance: Option<SelectionSignificanceRecord>,
}

/// 語彙ファイルを持つ構成が容量の目安（40MB）を超えたため選定対象から外した候補の記録
/// （REQ-30・TASK-30.3・#125）。数値と固定コードのみを持ち、パス・入力本文を含めない。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcludedCandidate {
    /// 除外した候補の添字。
    pub candidate_index: usize,
    /// 除外した候補 ID。
    pub candidate_id: String,
    /// 機械可読な除外理由（`vocab_package_over_guideline`）。
    pub reason: String,
    /// パッケージ合計バイト数。
    pub total_bytes: u64,
    /// 目安（バイト）。
    pub guideline_bytes: u64,
}

/// 全候補が容量の目安超過で除外され、選定記録を作れなかった `select` の除外結果
/// （`selection_exclusions.json`。REQ-30・TASK-30.3・#125）。
///
/// 要素は [`SelectionRecord::excluded_candidates`] と同じ形。プロジェクト内部の記録で、CLI の
/// stdout JSON と `selection_record.json` の形は変えない（`package`・`evaluate`・`infer` は読まない）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionExclusions {
    /// 除外した候補。
    pub excluded_candidates: Vec<ExcludedCandidate>,
}

impl SelectionExclusions {
    /// JSON 1 行（末尾改行つき）へ直列化する。
    ///
    /// # Errors
    /// 直列化に失敗した場合。
    pub fn to_json_vec(&self) -> Result<Vec<u8>, StageFileError> {
        let mut bytes = serde_json::to_vec(self).map_err(|_| StageFileError::Serialize)?;
        bytes.push(b'\n');
        Ok(bytes)
    }
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

    /// REQ-39: 生成量が上限ちょうど（改行込み 26 バイト）なら成功し、1 バイト小さい上限では
    /// `LimitExceeded`。複数行では累積で数える。
    #[test]
    fn req39_trainer_jsonl_limit_is_exact_and_cumulative() {
        let row = [("b", "y")];
        let bytes = trainer_jsonl_with_limit(row, 26).expect("exactly at the limit");
        assert_eq!(bytes.len(), 26);
        assert_eq!(
            trainer_jsonl_with_limit(row, 25),
            Err(StageFileError::LimitExceeded)
        );
        assert_eq!(
            trainer_jsonl_with_limit([("b", "y"), ("b", "y")], 52).map(|b| b.len()),
            Ok(52)
        );
        assert_eq!(
            trainer_jsonl_with_limit([("b", "y"), ("b", "y")], 51),
            Err(StageFileError::LimitExceeded)
        );
    }

    /// REQ-39: 既定の上限は入力読み込み上限（64 MiB）の 4 倍（268435456 バイト）。
    #[test]
    fn req39_trainer_jsonl_default_limit_is_four_times_input_limit() {
        assert_eq!(
            fandhe_edge_core::limits::TRAINER_JSONL_MAX_BYTES,
            268_435_456
        );
        assert_eq!(trainer_jsonl([("b", "y")]).map(|b| b.len()), Ok(26));
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
            excluded_candidates: Vec::new(),
            significance: None,
        };
        let bytes = r.to_json_vec().expect("json");
        // 除外が無い記録は従来どおり `excluded_candidates` キーを出さない。
        assert!(!String::from_utf8_lossy(&bytes).contains("excluded_candidates"));
        assert_eq!(SelectionRecord::from_json_slice(&bytes), Ok(r));
        assert_eq!(
            SelectionRecord::from_json_slice(br#"{"candidate_index":0,"extra":1}"#),
            Err(StageFileError::Malformed)
        );
    }

    /// REQ-30・TASK-30.3: 語彙超過の除外記録は往復でき、理由コードと数値を保つ。
    #[test]
    fn req30_selection_record_with_exclusion_round_trips() {
        let r = SelectionRecord {
            candidate_index: 0,
            candidate_id: "c1".to_string(),
            rule: "r".to_string(),
            validation_correct: 1,
            validation_total: 2,
            excluded_candidates: vec![ExcludedCandidate {
                candidate_index: 1,
                candidate_id: "qwen".to_string(),
                reason: "vocab_package_over_guideline".to_string(),
                total_bytes: 46_365_993,
                guideline_bytes: 40_000_000,
            }],
            significance: None,
        };
        let bytes = r.to_json_vec().expect("json");
        assert!(String::from_utf8_lossy(&bytes).contains("\"total_bytes\":46365993"));
        assert_eq!(SelectionRecord::from_json_slice(&bytes), Ok(r));
    }

    /// REQ-25・#481: 有意性判定つきの記録は往復でき、欄の無い従来の記録も読める（`None`）。
    #[test]
    fn req25_issue481_selection_record_significance_is_optional() {
        use fandhe_edge_core::evaluation_record::BaselineComparisonVerdict;
        let old = br#"{"candidate_index":0,"candidate_id":"c1","rule":"r","validation_correct":1,"validation_total":2}"#;
        let r = SelectionRecord::from_json_slice(old).expect("old record");
        assert_eq!(r.significance, None);
        let r = SelectionRecord {
            significance: Some(SelectionSignificanceRecord {
                majority_label: "a".to_string(),
                baseline_correct: 1,
                b: 1,
                c: 0,
                required_n: 7,
                family_size: 2,
                verdict: BaselineComparisonVerdict::Undeterminable,
            }),
            ..r
        };
        let bytes = r.to_json_vec().expect("json");
        assert_eq!(
            String::from_utf8_lossy(&bytes),
            "{\"candidate_index\":0,\"candidate_id\":\"c1\",\"rule\":\"r\",\"validation_correct\":1,\"validation_total\":2,\"significance\":{\"majority_label\":\"a\",\"baseline_correct\":1,\"b\":1,\"c\":0,\"required_n\":7,\"family_size\":2,\"verdict\":\"undeterminable\"}}\n"
        );
        assert_eq!(SelectionRecord::from_json_slice(&bytes), Ok(r));
    }
}
