//! 下限基準（majority）の予測生成。
//!
//! REQ-25 の下限基準比較（`b > c` かつ `p < 0.05` の判定）は本モジュールの
//! 出力（学習データから求めた多数決ラベル）と、[`crate::significance`] の
//! 統計判定を組み合わせて実現する（TASK-25.1-2・issue #65）。CLI の
//! `evaluate` 工程（REQ-33）が、データ契約層で検証済みの学習ラベルを渡して
//! 本モジュールを呼ぶ想定。
//!
//! # 評価契約との関係（REQ-17・REQ-27）
//!
//! [`fit_majority`] は **学習データのラベルだけ** から最頻ラベルを決める。
//! 評価データの gold・統計を下限基準へ渡してはならない（評価データの統計を
//! 推論側・下限基準側へ漏らさないという評価の独立性の不変条件）。呼び出し側
//! はこの関数へ評価データを渡さないこと。
//!
//! # 対象外（本 issue の範囲外）
//!
//! - 文字 n-gram 規則等の `simple_rule` 下限基準は未実装（依存
//!   `unicode-normalization` の承認と、入力表現（byte のみ方針）との整合の
//!   判断が要るため。`.claude/rules/dependency-policy.md`）

use crate::mcnemar::McNemarError;
use crate::metrics::{EvalError, MAX_LABELS};
use std::collections::BTreeMap;
use std::fmt;

/// 本 crate の下限基準比較まわり（[`fit_majority`]・
/// [`crate::significance::compare_with_baseline`]）が共通で返すエラー。
///
/// [`crate::metrics::EvalError`]（ラベル集合・混同行列の文脈）・
/// [`crate::mcnemar::McNemarError`]（統計計算コアの文脈）を包み、下限基準の
/// 予測生成（[`fit_majority`]）固有の失敗（学習ラベルが空・未知）と、
/// 有意性判定固有の失敗（評価レコードが空・gold が未知）を 1 つの型に
/// まとめる。呼び出し側（CLI の `evaluate` 工程）が終了コード（REQ-21）へ
/// 写す際に、`match` 1 箇所で全経路を扱えるようにする狙い（写像そのものは
/// 本モジュールの範囲外）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BaselineError {
    /// ラベル集合の検証エラー（空・空 ID・重複・上限超過）。
    Eval(EvalError),
    /// McNemar 検定コアのエラー（不一致ペア数の上限超過・桁あふれ等）。
    McNemar(McNemarError),
    /// 学習ラベルが 0 件（多数決を決められない）。
    EmptyTrainLabels,
    /// 学習ラベルにラベル集合へ存在しない ID が含まれる。
    UnknownTrainLabel {
        /// `train_labels` 内での位置（0 始まり）。
        index: usize,
    },
    /// 評価レコードが 0 件（0 除算を避け、評価済みを装わない）。
    EmptyRecords,
    /// 正解ラベルがラベル集合に存在しない（データ契約層で除外・警告される
    /// 前提だが、本層は fail-closed でエラーを返す）。
    UnknownGoldLabel {
        /// `records` 内での位置（0 始まり）。
        index: usize,
    },
    /// 理論上到達しないはずの内部不整合（fail-closed のガード）。
    Internal {
        /// 診断用の詳細（データ本文は含めない）。
        detail: String,
    },
}

impl From<EvalError> for BaselineError {
    fn from(err: EvalError) -> Self {
        BaselineError::Eval(err)
    }
}

impl From<McNemarError> for BaselineError {
    fn from(err: McNemarError) -> Self {
        BaselineError::McNemar(err)
    }
}

impl fmt::Display for BaselineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BaselineError::Eval(err) => write!(f, "{err}"),
            BaselineError::McNemar(err) => write!(f, "{err}"),
            BaselineError::EmptyTrainLabels => write!(f, "train labels must not be empty"),
            BaselineError::UnknownTrainLabel { index } => {
                write!(f, "unknown train label at index {index}")
            }
            BaselineError::EmptyRecords => write!(f, "records must not be empty"),
            BaselineError::UnknownGoldLabel { index } => {
                write!(f, "unknown gold label at record index {index}")
            }
            BaselineError::Internal { detail } => {
                write!(f, "internal baseline comparison error: {detail}")
            }
        }
    }
}

impl std::error::Error for BaselineError {}

/// ラベル集合を検証し、宣言順の索引（ID → 位置）を返す。
///
/// [`crate::metrics`] のラベル検証（`EmptyLabels`・`EmptyLabelId`・
/// `DuplicateLabel`・`TooManyLabels`）と同じ規則を、本モジュール内で小さく
/// 再実装したもの（`metrics.rs` は並行 PR〔TASK-24.3・issue #62〕が変更中
/// のため、共通化はここでは行わない。後続の refactor 候補）。
fn validate_label_order<'a>(
    label_order: &[&'a str],
) -> Result<BTreeMap<&'a str, usize>, EvalError> {
    if label_order.is_empty() {
        return Err(EvalError::EmptyLabels);
    }
    if label_order.len() > MAX_LABELS {
        return Err(EvalError::TooManyLabels {
            n_labels: label_order.len(),
            limit: MAX_LABELS,
        });
    }

    let mut index: BTreeMap<&'a str, usize> = BTreeMap::new();
    for (pos, &label) in label_order.iter().enumerate() {
        if label.is_empty() {
            return Err(EvalError::EmptyLabelId);
        }
        if index.insert(label, pos).is_some() {
            return Err(EvalError::DuplicateLabel {
                label: label.to_string(),
            });
        }
    }
    Ok(index)
}

/// 学習データのラベルだけから最頻ラベル（下限基準）を求める。
///
/// 同数の場合は `label_order` の宣言順で先に現れるラベルを採用する
/// （PoC-10 `baselines.py::fit_majority` と同じ規則）。
///
/// # 引数
///
/// - `label_order`: ラベル集合（ID を宣言順で並べたもの）。重複・空文字列・
///   `MAX_LABELS` 超過は拒否する
/// - `train_labels`: 学習データの正解ラベル列。**評価データの gold を渡し
///   てはならない**（モジュール docs 参照）
///
/// # エラー
///
/// - `label_order` が空・空 ID・重複・上限超過 → [`BaselineError::Eval`]
/// - `train_labels` が空 → [`BaselineError::EmptyTrainLabels`]
/// - `train_labels` に `label_order` に無いラベルがある →
///   [`BaselineError::UnknownTrainLabel`]
///
/// # 資源上限（REQ-39）
///
/// 件数表は `label_order.len()`（`MAX_LABELS` 以下）個の `u64` のみを確保し、
/// `train_labels.len()` に比例した確保はしない。加算はすべて `checked_add`
/// で行う。
pub fn fit_majority<'a>(
    label_order: &[&'a str],
    train_labels: &[&str],
) -> Result<&'a str, BaselineError> {
    let index = validate_label_order(label_order)?;

    if train_labels.is_empty() {
        return Err(BaselineError::EmptyTrainLabels);
    }

    let mut counts: Vec<u64> = vec![0; label_order.len()];
    for (i, &label) in train_labels.iter().enumerate() {
        let &pos = index
            .get(label)
            .ok_or(BaselineError::UnknownTrainLabel { index: i })?;
        let slot = counts
            .get_mut(pos)
            .ok_or(BaselineError::Eval(EvalError::Internal {
                detail: "label position out of bounds of counts table".to_string(),
            }))?;
        *slot = slot
            .checked_add(1)
            .ok_or(BaselineError::Eval(EvalError::Internal {
                detail: "count overflow while tallying train labels".to_string(),
            }))?;
    }

    // 宣言順で最初に最大値を取るラベルを選ぶ（同数時は宣言順で先着優先）。
    let mut best_pos = 0usize;
    let mut best_count = 0u64;
    for (pos, &count) in counts.iter().enumerate() {
        if count > best_count {
            best_count = count;
            best_pos = pos;
        }
    }

    label_order
        .get(best_pos)
        .copied()
        .ok_or(BaselineError::Eval(EvalError::Internal {
            detail: "best label position out of bounds of label_order".to_string(),
        }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 同数時は宣言順で先のラベルを選ぶ（PoC selftest 相当）。
    #[test]
    fn majority_ties_break_by_declaration_order() {
        let labels = ["A", "B", "C"];
        let train = ["B", "A", "B", "A"];
        assert_eq!(fit_majority(&labels, &train).unwrap(), "A");
    }

    /// 別の同数ケース（先頭が最頻でない場合）。
    #[test]
    fn majority_ties_break_by_declaration_order_second_case() {
        let labels = ["A", "B", "C"];
        let train = ["C", "C", "B"];
        assert_eq!(fit_majority(&labels, &train).unwrap(), "C");
    }

    /// 明確な最頻ラベルを選ぶ。
    #[test]
    fn majority_picks_most_frequent_label() {
        let labels = ["A", "B", "C"];
        let train = ["A", "B", "B", "B", "C"];
        assert_eq!(fit_majority(&labels, &train).unwrap(), "B");
    }

    /// 学習ラベルが空ならエラー。
    #[test]
    fn empty_train_labels_is_error() {
        let labels = ["A", "B"];
        let err = fit_majority(&labels, &[]).unwrap_err();
        assert_eq!(err, BaselineError::EmptyTrainLabels);
    }

    /// 学習ラベルに未知の ID があればエラー（index 付き）。
    #[test]
    fn unknown_train_label_is_error_with_index() {
        let labels = ["A", "B"];
        let train = ["A", "Z", "B"];
        let err = fit_majority(&labels, &train).unwrap_err();
        assert_eq!(err, BaselineError::UnknownTrainLabel { index: 1 });
    }

    /// ラベル集合が空ならエラー。
    #[test]
    fn empty_label_order_is_error() {
        let labels: [&str; 0] = [];
        let err = fit_majority(&labels, &["A"]).unwrap_err();
        assert_eq!(err, BaselineError::Eval(EvalError::EmptyLabels));
    }

    /// ラベル集合に重複があればエラー。
    #[test]
    fn duplicate_label_is_error() {
        let labels = ["A", "A"];
        let err = fit_majority(&labels, &["A"]).unwrap_err();
        assert_eq!(
            err,
            BaselineError::Eval(EvalError::DuplicateLabel {
                label: "A".to_string()
            })
        );
    }

    /// ラベル集合に空 ID があればエラー。
    #[test]
    fn empty_label_id_is_error() {
        let labels = ["A", ""];
        let err = fit_majority(&labels, &["A"]).unwrap_err();
        assert_eq!(err, BaselineError::Eval(EvalError::EmptyLabelId));
    }

    /// ラベル数が上限を超えればエラー（REQ-39）。
    #[test]
    fn too_many_labels_is_rejected() {
        let owned_labels: Vec<String> = (0..=MAX_LABELS).map(|i| format!("L{i}")).collect();
        let labels: Vec<&str> = owned_labels.iter().map(String::as_str).collect();
        let err = fit_majority(&labels, &["L0"]).unwrap_err();
        assert_eq!(
            err,
            BaselineError::Eval(EvalError::TooManyLabels {
                n_labels: MAX_LABELS + 1,
                limit: MAX_LABELS,
            })
        );
    }
}
