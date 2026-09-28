//! 旧モデルとの比較による回帰件数の算出（正常系）。
//!
//! 作り直し後のモデルを同じ評価データで旧モデルと比較し、「正解→不正解
//! （回帰）」「不正解→正解（改善）」の件数を報告する（REQ-26 正常系・
//! TASK-26.1-1・issue #100）。将来 CLI の `evaluate` 工程（REQ-33）・
//! 作り直し判定（[`crate::significance`] と対になる `RebuildDecision` 接続・
//! TASK-20.1）から呼ばれる想定で、本 issue では回帰件数の算出のみを担う。
//!
//! 移植元は PoC-19（`03-poc/model-lifecycle/scripts/exp_rebuild.py` の
//! `paired_bootstrap_ci` 内の `b`/`c` 集計）。ただし PoC のペアブートストラップ
//! CI は移植しない。信頼区間の付与は Wilson 95% CI を採用する
//! （`.claude/rules/evaluation-contract.md`「有意性・指標」・
//! TASK-26.1-2・issue #101）。
//!
//! # 信頼区間（TASK-26.1-2・issue #101）
//!
//! [`RegressionCounts::correct_to_incorrect_ci95`]・
//! [`RegressionCounts::incorrect_to_correct_ci95`] は、それぞれの遷移件数を
//! 比較対象の総件数 `n` で割った率（`correct_to_incorrect / n`・
//! `incorrect_to_correct / n`）に対する Wilson 95% 信頼区間を返す。
//! 評価契約（`.claude/rules/evaluation-contract.md`「有意性・指標」）が
//! 採用する Wilson を [`crate::wilson`]（TASK-24.1-2・issue #60）から
//! そのまま再利用し、評価器を層内で再実装しない。
//!
//! PoC-19 の `paired_bootstrap_ci` が出していたのは、正解率の差
//! （`acc_new - acc_old`）に対する**ペアブートストラップ**信頼区間であり、
//! 本モジュールが返す「遷移率（正解→不正解・不正解→正解）の Wilson 区間」
//! とは対象が異なる。ペアブートストラップは移植せず、評価契約の定める
//! Wilson を採用する。
//!
//! z は常に [`crate::wilson::WILSON_Z_95`]（1.96）を使う。任意の z を
//! 受け取る API は公開しない（評価契約は 95% と定めている）。
//!
//! # 方向の対応（取り違え防止）
//!
//! [`crate::mcnemar::paired_counts`] を
//! `paired_counts(candidate_correct = current(新), baseline_correct = previous(旧))`
//! と呼ぶ。このとき:
//!
//! - `b_candidate_only`（新のみ正解） = **不正解→正解（改善）** =
//!   [`RegressionCounts::incorrect_to_correct`]
//! - `c_baseline_only`（旧のみ正解） = **正解→不正解（回帰）** =
//!   [`RegressionCounts::correct_to_incorrect`]
//!
//! PoC-19 の `b_correct_to_incorrect`（旧→新で「正解→不正解」を `b` と
//! 呼んでいた）とは文字（b/c）の対応が逆になる点に注意する。
//!
//! # 対象外（本 issue の範囲外）
//!
//! - 旧・新でラベル集合が異なる場合（PoC-19 P3 の統合パターン）の前提明記
//!   （TASK-26.2）。[`compare_with_previous`] は旧・新が同一のラベル集合を
//!   持つ場合の API で、ラベル集合が異なる比較は [`regression_counts`]
//!   （各モデル自身のラベル空間で正誤を求めた bool 列を渡す経路）を使う
//! - 再現性（3 seed の CI 重なり。TASK-26.3）
//! - TASK-20.1 の作り直し判定（`RebuildDecision`）を受けて作り直し後にのみ
//!   比較する接続（親 #99 / 後続）
//! - CLI `evaluate` 工程への配線・JSON 化・終了コードへの写像
//!   （#140・TASK-33.x）
//!
//! # 評価契約との関係（REQ-27）
//!
//! 入力は `&` 参照でのみ受け取り、書き換えない。推論関数には触れず、
//! 既に得られている [`Outcome`]（評価済みの予測結果）を受け取るだけ。
//! ファイル I/O は行わない。

use crate::baseline;
use crate::mcnemar::{self, McNemarError};
use crate::metrics::Outcome;
use crate::significance::MAX_EVAL_RECORDS;
use crate::wilson::{self, WilsonInterval};
use std::fmt;

/// [`regression_counts`]・[`compare_with_previous`] が返しうるエラー。
///
/// [`BaselineError`]（下限基準比較専用の意味を持つ）とは意図的に分け、
/// 回帰件数算出に固有の失敗だけを表す薄い型にする。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegressionError {
    /// ラベル集合の検証エラー（空・空 ID・重複・上限超過）。
    Labels(crate::metrics::EvalError),
    /// 比較対象の行が 0 件（評価済みを装わない）。
    EmptyRecords,
    /// 件数が [`MAX_EVAL_RECORDS`] を超える（REQ-39 資源の上限。
    /// [`compare_with_previous`] では 2 本の `Vec<bool>` を確保する前に、
    /// [`regression_counts`] では（追加のアロケーションを行わないが）
    /// 処理前に、同じ上限規則で拒否する）。
    TooManyRecords {
        /// 渡された件数。
        n_records: usize,
        /// 上限（[`MAX_EVAL_RECORDS`]）。
        limit: usize,
    },
    /// `previous`・`current` の正誤列の長さが一致しない。
    LengthMismatch {
        /// 旧モデル側の長さ。
        previous: usize,
        /// 新モデル側の長さ。
        current: usize,
    },
    /// 正解ラベルがラベル集合に存在しない（fail-closed。黙って除外しない）。
    UnknownGoldLabel {
        /// `records` 内での位置（0 始まり）。
        index: usize,
    },
    /// 件数の合計が `u64` の範囲を超える（[`mcnemar::paired_counts`] から）。
    CountOverflow,
    /// 理論上到達しないはずの内部不整合（fail-closed のガード。事前に長さを
    /// 検証しているため通常到達しない）。
    Internal {
        /// 診断用の詳細（データ本文は含めない）。
        detail: String,
    },
}

impl From<crate::metrics::EvalError> for RegressionError {
    fn from(err: crate::metrics::EvalError) -> Self {
        RegressionError::Labels(err)
    }
}

impl From<McNemarError> for RegressionError {
    fn from(err: McNemarError) -> Self {
        match err {
            McNemarError::LengthMismatch {
                candidate,
                baseline,
            } => RegressionError::LengthMismatch {
                previous: baseline,
                current: candidate,
            },
            McNemarError::CountOverflow => RegressionError::CountOverflow,
            other => RegressionError::Internal {
                detail: format!("unexpected mcnemar error after length validation: {other}"),
            },
        }
    }
}

impl fmt::Display for RegressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegressionError::Labels(err) => write!(f, "{err}"),
            RegressionError::EmptyRecords => write!(f, "records must not be empty"),
            RegressionError::TooManyRecords { n_records, limit } => {
                write!(f, "too many records: {n_records} (limit: {limit})")
            }
            RegressionError::LengthMismatch { previous, current } => {
                write!(
                    f,
                    "previous/current length mismatch: {previous} vs {current}"
                )
            }
            RegressionError::UnknownGoldLabel { index } => {
                write!(f, "unknown gold label at record index {index}")
            }
            RegressionError::CountOverflow => write!(f, "count overflow while computing counts"),
            RegressionError::Internal { detail } => {
                write!(f, "internal regression computation error: {detail}")
            }
        }
    }
}

impl std::error::Error for RegressionError {}

/// 旧モデル・新モデルを 1 行に対応させた入力。
///
/// index のずれをスライスの並行走査ではなく型で防ぐ（[`crate::significance::PairedRecord`]
/// と同じ設計）。
#[derive(Debug, Clone, Copy)]
pub struct RegressionRecord<'a> {
    /// 正解ラベル ID。
    pub gold: &'a str,
    /// 旧モデルの推論結果。
    pub previous: &'a Outcome,
    /// 新モデルの推論結果。
    pub current: &'a Outcome,
}

/// 旧モデル・新モデルの比較結果（2×2 の分割表）。
///
/// フィールドは非公開にし、アクセサ経由で公開する。全フィールド `u64` の
/// 内部計算結果から構築するため、外部から壊れた値（合計が `n` と一致しない
/// 等）を組み立てられない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegressionCounts {
    n: u64,
    both_correct: u64,
    correct_to_incorrect: u64,
    incorrect_to_correct: u64,
    both_wrong: u64,
}

impl RegressionCounts {
    /// 比較対象の総件数。
    pub fn n(&self) -> u64 {
        self.n
    }

    /// 旧・新ともに正解した件数。
    pub fn both_correct(&self) -> u64 {
        self.both_correct
    }

    /// 正解→不正解（回帰）の件数（旧のみ正解 = McNemar の `c`）。
    pub fn correct_to_incorrect(&self) -> u64 {
        self.correct_to_incorrect
    }

    /// 不正解→正解（改善）の件数（新のみ正解 = McNemar の `b`）。
    pub fn incorrect_to_correct(&self) -> u64 {
        self.incorrect_to_correct
    }

    /// 旧・新ともに不正解だった件数。
    pub fn both_wrong(&self) -> u64 {
        self.both_wrong
    }

    /// 旧モデルの正解件数（`both_correct + correct_to_incorrect`）。
    ///
    /// [`mcnemar::paired_counts`] が `n` の非オーバーフローを検証済みの
    /// 内訳から求めるため、ここでの再計算も溢れない
    /// （`BaselineComparison::baseline_correct` と同じ根拠）。
    pub fn previous_correct(&self) -> u64 {
        self.both_correct + self.correct_to_incorrect
    }

    /// 新モデルの正解件数（`both_correct + incorrect_to_correct`）。
    pub fn current_correct(&self) -> u64 {
        self.both_correct + self.incorrect_to_correct
    }

    /// 正解→不正解（回帰）の率 `correct_to_incorrect / n` の Wilson 95%
    /// 信頼区間（REQ-26・TASK-26.1-2・issue #101）。
    ///
    /// `n`（[`RegressionCounts::n`]）は [`EmptyRecords`](RegressionError::EmptyRecords)
    /// の検査で必ず 0 より大きく、`correct_to_incorrect` は
    /// [`mcnemar::paired_counts`] の集計から `n` を超えないことが保証される
    /// ため実際には常に `Some` を返すが、将来この不変条件が変わっても
    /// panic させないよう [`crate::wilson::wilson_ci95`] と同じく `Option`
    /// のまま返す。
    pub fn correct_to_incorrect_ci95(&self) -> Option<WilsonInterval> {
        wilson::wilson_ci95(self.correct_to_incorrect, self.n)
    }

    /// 不正解→正解（改善）の率 `incorrect_to_correct / n` の Wilson 95%
    /// 信頼区間（REQ-26・TASK-26.1-2・issue #101）。
    ///
    /// `Option` を返す理由は
    /// [`correct_to_incorrect_ci95`](Self::correct_to_incorrect_ci95) と同じ。
    pub fn incorrect_to_correct_ci95(&self) -> Option<WilsonInterval> {
        wilson::wilson_ci95(self.incorrect_to_correct, self.n)
    }
}

/// 旧モデル・新モデルそれぞれの行ごとの正誤（bool 列。正解なら `true`）から
/// [`RegressionCounts`] を求める。
///
/// 各モデル自身のラベル空間で求めた正誤ビットを渡す経路のため、旧・新で
/// ラベル集合が異なる場合（PoC-19 P3 の統合パターン）もここへ渡せる
/// （ラベル集合相違の前提明記自体は TASK-26.2 の対象で、本関数はその前提を
/// 検査しない）。
///
/// 本関数自体は受け取ったスライスをそのまま検査するだけで、追加の
/// `Vec` を確保しない（2 本の `Vec<bool>` を確保するのは呼び出し元
/// [`compare_with_previous`] 側）。ただし件数の上限規則は
/// [`compare_with_previous`] と揃え、検査前に同じ [`MAX_EVAL_RECORDS`]
/// で拒否する。
///
/// # エラー
///
/// - `previous_correct`・`current_correct` がともに空 →
///   [`RegressionError::EmptyRecords`]（評価済みを装わない。0 件は
///   [`mcnemar::paired_counts`] のように「全件 0」を黙って返さない）
/// - `previous_correct`・`current_correct` のいずれかの長さが
///   [`MAX_EVAL_RECORDS`] を超える → [`RegressionError::TooManyRecords`]
///   （REQ-39）
/// - 長さが一致しない → [`RegressionError::LengthMismatch`]
pub fn regression_counts(
    previous_correct: &[bool],
    current_correct: &[bool],
) -> Result<RegressionCounts, RegressionError> {
    if previous_correct.is_empty() && current_correct.is_empty() {
        return Err(RegressionError::EmptyRecords);
    }
    if previous_correct.len() > MAX_EVAL_RECORDS {
        return Err(RegressionError::TooManyRecords {
            n_records: previous_correct.len(),
            limit: MAX_EVAL_RECORDS,
        });
    }
    if current_correct.len() > MAX_EVAL_RECORDS {
        return Err(RegressionError::TooManyRecords {
            n_records: current_correct.len(),
            limit: MAX_EVAL_RECORDS,
        });
    }
    if previous_correct.len() != current_correct.len() {
        return Err(RegressionError::LengthMismatch {
            previous: previous_correct.len(),
            current: current_correct.len(),
        });
    }

    let counts = mcnemar::paired_counts(current_correct, previous_correct)?;

    Ok(RegressionCounts {
        n: counts.n,
        both_correct: counts.both_correct,
        correct_to_incorrect: counts.c_baseline_only,
        incorrect_to_correct: counts.b_candidate_only,
        both_wrong: counts.both_wrong,
    })
}

/// ラベル集合・旧モデル・新モデルの対応データから [`RegressionCounts`] を
/// 求める（旧・新が同一のラベル集合を持つ場合の record 単位 API）。
///
/// 手順: [`baseline::validate_label_order`] でラベル検証 →（空・上限）
/// チェック → 各行で gold を検証しつつ [`crate::significance::is_correct`]
/// （評価器の正誤規則。TASK-24.1-2 と同じ規則）で旧・新の正誤を求める →
/// [`regression_counts`]。
///
/// `labels` の検証規則（空・空 ID・重複・上限超過）は
/// [`crate::significance::correctness`]・[`crate::baseline::fit_majority`] と
/// 揃える（Review 前例。TASK-25.1-2・issue #65）。
///
/// # エラー
///
/// - `labels` が空・空 ID・重複・上限超過 → [`RegressionError::Labels`]
/// - `records` が空 → [`RegressionError::EmptyRecords`]
/// - `records.len()` が [`MAX_EVAL_RECORDS`] を超える → 2 本の
///   `Vec<bool>` を確保する前に [`RegressionError::TooManyRecords`]
///   （REQ-39）
/// - gold がラベル集合に無い行がある → [`RegressionError::UnknownGoldLabel`]
///   （fail-closed。黙って除外しない）
pub fn compare_with_previous(
    labels: &[&str],
    records: &[RegressionRecord<'_>],
) -> Result<RegressionCounts, RegressionError> {
    let index = baseline::validate_label_order(labels).map_err(RegressionError::Labels)?;

    if records.is_empty() {
        return Err(RegressionError::EmptyRecords);
    }
    if records.len() > MAX_EVAL_RECORDS {
        return Err(RegressionError::TooManyRecords {
            n_records: records.len(),
            limit: MAX_EVAL_RECORDS,
        });
    }

    let mut previous_correct = Vec::with_capacity(records.len());
    let mut current_correct = Vec::with_capacity(records.len());
    for (i, record) in records.iter().enumerate() {
        if !index.contains_key(record.gold) {
            return Err(RegressionError::UnknownGoldLabel { index: i });
        }
        previous_correct.push(crate::significance::is_correct(
            record.gold,
            record.previous,
        ));
        current_correct.push(crate::significance::is_correct(record.gold, record.current));
    }

    regression_counts(&previous_correct, &current_correct)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-26・TASK-26.1-1: Outcome の各種別（`Abstain`・`Error`・`Invalid`・
    /// 未知ラベル）はすべて不正解として数えられる。
    #[test]
    fn non_label_outcomes_count_as_incorrect() {
        let labels = ["A"];
        let gold_label = Outcome::Label("A".to_string());
        let abstain = Outcome::Abstain;
        let error = Outcome::Error;
        let invalid = Outcome::Invalid;
        let unknown = Outcome::Label("Z".to_string());

        // 旧が正解・新が Abstain → 回帰 1。
        let records = [RegressionRecord {
            gold: "A",
            previous: &gold_label,
            current: &abstain,
        }];
        let counts = compare_with_previous(&labels, &records).unwrap();
        assert_eq!(counts.correct_to_incorrect(), 1);
        assert_eq!(counts.incorrect_to_correct(), 0);

        // 旧が Error・新が正解 → 改善 1。
        let records = [RegressionRecord {
            gold: "A",
            previous: &error,
            current: &gold_label,
        }];
        let counts = compare_with_previous(&labels, &records).unwrap();
        assert_eq!(counts.correct_to_incorrect(), 0);
        assert_eq!(counts.incorrect_to_correct(), 1);

        // 旧が Invalid・新が未知ラベル → どちらも不正解のまま（両方不正解）。
        let records = [RegressionRecord {
            gold: "A",
            previous: &invalid,
            current: &unknown,
        }];
        let counts = compare_with_previous(&labels, &records).unwrap();
        assert_eq!(counts.both_wrong(), 1);
        assert_eq!(counts.correct_to_incorrect(), 0);
        assert_eq!(counts.incorrect_to_correct(), 0);
    }

    /// 空入力は `EmptyRecords`（`compare_with_previous`・`regression_counts`
    /// の両方）。
    #[test]
    fn empty_records_is_error() {
        let labels = ["A"];
        let err = compare_with_previous(&labels, &[]).unwrap_err();
        assert_eq!(err, RegressionError::EmptyRecords);

        let err = regression_counts(&[], &[]).unwrap_err();
        assert_eq!(err, RegressionError::EmptyRecords);
    }

    /// `regression_counts`: 長さ不一致は `LengthMismatch`。
    #[test]
    fn regression_counts_length_mismatch() {
        let err = regression_counts(&[true], &[true, false]).unwrap_err();
        assert_eq!(
            err,
            RegressionError::LengthMismatch {
                previous: 1,
                current: 2,
            }
        );
    }

    /// 未知 gold は index 付きでエラーになる（2 行目）。
    #[test]
    fn unknown_gold_label_is_error_with_index() {
        let labels = ["A", "B"];
        let ok = Outcome::Label("A".to_string());
        let records = [
            RegressionRecord {
                gold: "A",
                previous: &ok,
                current: &ok,
            },
            RegressionRecord {
                gold: "Z",
                previous: &ok,
                current: &ok,
            },
        ];
        let err = compare_with_previous(&labels, &records).unwrap_err();
        assert_eq!(err, RegressionError::UnknownGoldLabel { index: 1 });
    }

    /// ラベル集合が空なら `Labels(EmptyLabels)`。
    #[test]
    fn empty_label_order_is_error() {
        let labels: [&str; 0] = [];
        let ok = Outcome::Label("A".to_string());
        let records = [RegressionRecord {
            gold: "A",
            previous: &ok,
            current: &ok,
        }];
        let err = compare_with_previous(&labels, &records).unwrap_err();
        assert_eq!(
            err,
            RegressionError::Labels(crate::metrics::EvalError::EmptyLabels)
        );
    }

    /// ラベル集合に重複があれば `Labels(DuplicateLabel)`。
    #[test]
    fn duplicate_label_order_is_error() {
        let labels = ["A", "A"];
        let ok = Outcome::Label("A".to_string());
        let records = [RegressionRecord {
            gold: "A",
            previous: &ok,
            current: &ok,
        }];
        let err = compare_with_previous(&labels, &records).unwrap_err();
        assert_eq!(
            err,
            RegressionError::Labels(crate::metrics::EvalError::DuplicateLabel {
                label: "A".to_string()
            })
        );
    }

    /// REQ-39: `compare_with_previous` は `records.len()` が
    /// [`MAX_EVAL_RECORDS`] を超える場合、2 本の `Vec<bool>` を確保する前に
    /// `TooManyRecords` で拒否する。
    #[test]
    fn compare_with_previous_too_many_records_is_error() {
        let labels = ["A"];
        let ok = Outcome::Label("A".to_string());
        let records: Vec<RegressionRecord<'_>> = (0..=MAX_EVAL_RECORDS)
            .map(|_| RegressionRecord {
                gold: "A",
                previous: &ok,
                current: &ok,
            })
            .collect();
        let err = compare_with_previous(&labels, &records).unwrap_err();
        assert_eq!(
            err,
            RegressionError::TooManyRecords {
                n_records: MAX_EVAL_RECORDS + 1,
                limit: MAX_EVAL_RECORDS,
            }
        );
    }

    /// REQ-39: `regression_counts` も同じ規則で件数を確保前に検証する。
    #[test]
    fn regression_counts_too_many_records_is_error() {
        let previous: Vec<bool> = vec![true; MAX_EVAL_RECORDS + 1];
        let current: Vec<bool> = vec![true; MAX_EVAL_RECORDS + 1];
        let err = regression_counts(&previous, &current).unwrap_err();
        assert_eq!(
            err,
            RegressionError::TooManyRecords {
                n_records: MAX_EVAL_RECORDS + 1,
                limit: MAX_EVAL_RECORDS,
            }
        );
    }

    /// 全件一致（回帰 0・改善 0）の境界値。
    #[test]
    fn all_agree_gives_zero_transitions() {
        let labels = ["A", "B"];
        let correct_a = Outcome::Label("A".to_string());
        let wrong_a = Outcome::Label("B".to_string());
        let records = [
            RegressionRecord {
                gold: "A",
                previous: &correct_a,
                current: &correct_a,
            },
            RegressionRecord {
                gold: "A",
                previous: &wrong_a,
                current: &wrong_a,
            },
        ];
        let counts = compare_with_previous(&labels, &records).unwrap();
        assert_eq!(counts.correct_to_incorrect(), 0);
        assert_eq!(counts.incorrect_to_correct(), 0);
        assert_eq!(counts.both_correct(), 1);
        assert_eq!(counts.both_wrong(), 1);
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// REQ-26・TASK-26.1-2: 遷移 0 件（全件一致）では回帰・改善どちらの率も
    /// `k=0` の Wilson 区間になり、`lo` は `0.0` に完全一致する。
    #[test]
    fn zero_transitions_ci95_lo_is_zero() {
        let labels = ["A", "B"];
        let correct = Outcome::Label("A".to_string());
        let wrong = Outcome::Label("B".to_string());
        let records = [
            RegressionRecord {
                gold: "A",
                previous: &correct,
                current: &correct,
            },
            RegressionRecord {
                gold: "A",
                previous: &wrong,
                current: &wrong,
            },
        ];
        let counts = compare_with_previous(&labels, &records).unwrap();

        let regression_ci = counts.correct_to_incorrect_ci95().unwrap();
        assert_eq!(regression_ci.lo(), 0.0);
        assert!(approx_eq(regression_ci.hi(), 0.6576280471103807));

        let improvement_ci = counts.incorrect_to_correct_ci95().unwrap();
        assert_eq!(improvement_ci.lo(), 0.0);
        assert!(approx_eq(improvement_ci.hi(), 0.6576280471103807));
    }

    /// REQ-26・TASK-26.1-2: 全件が回帰（正解→不正解）の境界値では
    /// `hi` が `1.0` に完全一致する。
    #[test]
    fn all_transitions_ci95_hi_is_one() {
        let labels = ["A", "B"];
        let correct = Outcome::Label("A".to_string());
        let wrong = Outcome::Label("B".to_string());
        let records: Vec<RegressionRecord<'_>> = (0..7)
            .map(|_| RegressionRecord {
                gold: "A",
                previous: &correct,
                current: &wrong,
            })
            .collect();
        let counts = compare_with_previous(&labels, &records).unwrap();

        let regression_ci = counts.correct_to_incorrect_ci95().unwrap();
        assert!(approx_eq(regression_ci.lo(), 0.6456611570247934));
        assert_eq!(regression_ci.hi(), 1.0);

        let improvement_ci = counts.incorrect_to_correct_ci95().unwrap();
        assert_eq!(improvement_ci.lo(), 0.0);
        assert!(approx_eq(improvement_ci.hi(), 0.35433884297520657));
    }

    /// REQ-26・TASK-26.1-2: 両メソッドとも `z = WILSON_Z_95`（1.96）を使う。
    #[test]
    fn ci95_uses_wilson_z_95() {
        let labels = ["A", "B"];
        let correct = Outcome::Label("A".to_string());
        let wrong = Outcome::Label("B".to_string());
        let records = [
            RegressionRecord {
                gold: "A",
                previous: &correct,
                current: &wrong,
            },
            RegressionRecord {
                gold: "A",
                previous: &wrong,
                current: &correct,
            },
        ];
        let counts = compare_with_previous(&labels, &records).unwrap();

        assert_eq!(
            counts.correct_to_incorrect_ci95().unwrap().z(),
            wilson::WILSON_Z_95
        );
        assert_eq!(
            counts.incorrect_to_correct_ci95().unwrap().z(),
            wilson::WILSON_Z_95
        );
    }

    /// REQ-26・TASK-26.1-2: `correct_to_incorrect_ci95`・
    /// `incorrect_to_correct_ci95` は `wilson::wilson_ci95(count, n)` の
    /// 呼び出しと一致する（評価器を層内で二重に実装していないことの確認）。
    #[test]
    fn ci95_matches_wilson_module_directly() {
        let labels = ["A", "B", "C"];
        let correct = Outcome::Label("A".to_string());
        let wrong = Outcome::Label("B".to_string());
        let records = [
            RegressionRecord {
                gold: "A",
                previous: &correct,
                current: &correct,
            },
            RegressionRecord {
                gold: "A",
                previous: &correct,
                current: &wrong,
            },
            RegressionRecord {
                gold: "A",
                previous: &correct,
                current: &wrong,
            },
            RegressionRecord {
                gold: "A",
                previous: &wrong,
                current: &correct,
            },
            RegressionRecord {
                gold: "A",
                previous: &wrong,
                current: &wrong,
            },
        ];
        let counts = compare_with_previous(&labels, &records).unwrap();

        assert_eq!(
            counts.correct_to_incorrect_ci95(),
            wilson::wilson_ci95(counts.correct_to_incorrect(), counts.n())
        );
        assert_eq!(
            counts.incorrect_to_correct_ci95(),
            wilson::wilson_ci95(counts.incorrect_to_correct(), counts.n())
        );
    }

    /// `Display` は英語で、データ本文を含まない（index・件数のみ）。
    #[test]
    fn display_is_english_and_excludes_record_content() {
        let err = RegressionError::UnknownGoldLabel { index: 3 };
        assert_eq!(err.to_string(), "unknown gold label at record index 3");

        let err = RegressionError::TooManyRecords {
            n_records: 5,
            limit: 3,
        };
        assert_eq!(err.to_string(), "too many records: 5 (limit: 3)");
    }
}
