//! 診断レポートの基礎統計（行数・ユニーク数・ラベル別件数）。
//!
//! CLI の `evaluate` 工程（配線は issue #140 で未実装）が、学習データ・評価データの
//! それぞれについて 1 回ずつ [`basic_stats`] を呼び、精度の変化要因を読み解く材料を
//! 得る想定（REQ-29 正常系・TASK-29.1-1・issue #107）。定義は PoC-11 の
//! `build_series.py::stat_of`（rows / unique_inputs / unique_outputs / label_counts）
//! に対応する。期待値の出典ではなく統計の定義の出典であり、テストは手組みの合成
//! データ（証拠種別: テストハーネス）で検証する。
//!
//! - **診断専用**: 結果は合否判定に使わない（TASK-29.1・REQ-29「精度の目安は検討中」）。
//! - **正規化は呼び出し側の責務**: `input` は呼び出し側が正規化済み（データ契約層の
//!   `NfkcWhitespaceNormalizer` 等）の文字列を渡す。本モジュールはバイト一致で
//!   ユニーク数を数え、数え方の規則 ID は `input_key_rule` として結果へ残す。
//!   層の境界のため data 層へは依存しない。
//! - **本文を保持しない**: 結果・エラーは件数とラベル ID のみを持ち、入力本文を
//!   複製・転記しない（`.claude/rules/security.md`）。
//! - 引数は共有参照のみで書き換えない（REQ-27 の評価前後ハッシュ不変と両立）。
//!
//! 未実装: 混同しやすいラベルの組・レポート統合（TASK-29.1-2・#108）、診断限界の
//! 明記（TASK-29.2・#109）、データ量水準別報告（TASK-29.3・#110）、group 数、
//! JSON 直列化（CLI 層）。

use std::collections::BTreeSet;
use std::fmt;

use crate::baseline::validate_label_order;
use crate::metrics::EvalError;
use crate::significance::MAX_EVAL_RECORDS;

/// 集計対象の 1 行（入力とラベル ID。いずれも借用）。
#[derive(Debug, Clone, Copy)]
pub struct StatsRow<'a> {
    /// 正規化済みの入力（呼び出し側の責務）。
    pub input: &'a str,
    /// ラベル ID。
    pub label: &'a str,
}

/// ラベルごとの件数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelCount {
    /// ラベル ID。
    pub label: String,
    /// 出現件数（0 もありうる）。
    pub count: u64,
}

/// 1 データセットの基礎統計（診断専用。合否判定には使わない）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicStats {
    /// 行数。
    pub n_rows: u64,
    /// ユニークな入力数（`input_key_rule` の規則で数える）。
    pub unique_inputs: u64,
    /// 1 件以上出現したラベルの数（PoC-11 の `unique_outputs`）。
    pub unique_labels: u64,
    /// 宣言順のラベル別件数（0 件のラベルも含む）。
    pub label_counts: Vec<LabelCount>,
    /// ラベル別件数の最小値。
    pub min_label_count: u64,
    /// 最小件数に並ぶラベル（宣言順）。
    pub min_labels: Vec<String>,
    /// ユニーク数の数え方の規則 ID（呼び出し側が渡した値。未正規化なら `None`）。
    pub input_key_rule: Option<&'static str>,
}

/// [`basic_stats`] のエラー。メッセージは英語で、入力本文・ラベル値を含めない。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiagnosticsError {
    /// ラベル集合の検証エラー。
    Labels(EvalError),
    /// 行が 0 件（集計済みを装わない）。
    EmptyRows,
    /// 行数が [`MAX_EVAL_RECORDS`] を超える（REQ-39。走査前に拒否）。
    TooManyRows {
        /// 渡された行数。
        n_rows: usize,
        /// 上限。
        limit: usize,
    },
    /// 行のラベルがラベル集合に無い。
    UnknownLabel {
        /// 行位置。
        index: usize,
    },
    /// 桁あふれ・添字不整合（理論上到達しない）。
    Internal {
        /// 詳細（英語）。
        detail: String,
    },
}

impl fmt::Display for DiagnosticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticsError::Labels(err) => write!(f, "{err}"),
            DiagnosticsError::EmptyRows => write!(f, "rows must not be empty"),
            DiagnosticsError::TooManyRows { n_rows, limit } => {
                write!(f, "too many rows: {n_rows} (limit: {limit})")
            }
            DiagnosticsError::UnknownLabel { index } => {
                write!(f, "unknown label at row index {index}")
            }
            DiagnosticsError::Internal { detail } => {
                write!(f, "internal diagnostics error: {detail}")
            }
        }
    }
}

impl std::error::Error for DiagnosticsError {}

fn internal(detail: &str) -> DiagnosticsError {
    DiagnosticsError::Internal {
        detail: detail.to_string(),
    }
}

/// 基礎統計を集計する（REQ-29 正常系・TASK-29.1-1）。
///
/// `labels` は宣言順のラベル ID。学習データ・評価データそれぞれで 1 回ずつ呼ぶ。
/// 決定的（`BTreeSet` のみ・浮動小数なし）で panic しない。
pub fn basic_stats(
    labels: &[&str],
    rows: &[StatsRow<'_>],
    input_key_rule: Option<&'static str>,
) -> Result<BasicStats, DiagnosticsError> {
    if rows.len() > MAX_EVAL_RECORDS {
        return Err(DiagnosticsError::TooManyRows {
            n_rows: rows.len(),
            limit: MAX_EVAL_RECORDS,
        });
    }
    let index = validate_label_order(labels).map_err(DiagnosticsError::Labels)?;
    if rows.is_empty() {
        return Err(DiagnosticsError::EmptyRows);
    }

    let mut counts: Vec<u64> = vec![0; labels.len()];
    let mut inputs: BTreeSet<&str> = BTreeSet::new();
    for (i, row) in rows.iter().enumerate() {
        let &pos = index
            .get(row.label)
            .ok_or(DiagnosticsError::UnknownLabel { index: i })?;
        let slot = counts
            .get_mut(pos)
            .ok_or_else(|| internal("label position out of bounds of counts table"))?;
        *slot = slot
            .checked_add(1)
            .ok_or_else(|| internal("count overflow while tallying labels"))?;
        inputs.insert(row.input);
    }

    let min_label_count = *counts
        .iter()
        .min()
        .ok_or_else(|| internal("empty counts table"))?;
    let unique_labels = counts.iter().filter(|&&c| c > 0).count();
    let mut label_counts = Vec::with_capacity(labels.len());
    let mut min_labels = Vec::new();
    for (&label, &count) in labels.iter().zip(counts.iter()) {
        label_counts.push(LabelCount {
            label: label.to_string(),
            count,
        });
        if count == min_label_count {
            min_labels.push(label.to_string());
        }
    }

    Ok(BasicStats {
        n_rows: u64::try_from(rows.len()).map_err(|_| internal("row count conversion"))?,
        unique_inputs: u64::try_from(inputs.len())
            .map_err(|_| internal("unique input count conversion"))?,
        unique_labels: u64::try_from(unique_labels)
            .map_err(|_| internal("unique label count conversion"))?,
        label_counts,
        min_label_count,
        min_labels,
        input_key_rule,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-29・TASK-29.1-1: 1 行・1 ラベルの境界。
    #[test]
    fn single_row_single_label() {
        let rows = [StatsRow {
            input: "x",
            label: "A",
        }];
        let s = basic_stats(&["A"], &rows, None).unwrap();
        assert_eq!(s.n_rows, 1);
        assert_eq!(s.unique_inputs, 1);
        assert_eq!(s.unique_labels, 1);
        assert_eq!(s.min_label_count, 1);
        assert_eq!(s.min_labels, vec!["A".to_string()]);
    }

    /// REQ-29: エラーメッセージに入力本文・ラベル値を含めない。
    #[test]
    fn unknown_label_message_hides_values() {
        let rows = [StatsRow {
            input: "secret-body",
            label: "secret-label",
        }];
        let err = basic_stats(&["A"], &rows, None).unwrap_err();
        assert_eq!(err, DiagnosticsError::UnknownLabel { index: 0 });
        let msg = err.to_string();
        assert_eq!(msg, "unknown label at row index 0");
        assert!(!msg.contains("secret"));
    }
}
