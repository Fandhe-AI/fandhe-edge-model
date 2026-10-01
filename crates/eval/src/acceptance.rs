//! 正解率の合否基準判定（REQ-24・REQ-33・#328）。
//!
//! CLI の `package` 工程が、評価記録の `correct`・`total` と定義ファイルの
//! `acceptance.min_accuracy_bp`（1 万分率）を [`judge_min_accuracy`] へ渡し、返った
//! [`AcceptanceVerdict`] を `PackageQualityJudgment`（runtime）へ写す。評価ロジックは
//! 評価器へ集約する方針（TASK-24.1）に従い、判定は本 crate に置く。`fandhe-edge-core` の
//! `definition` には依存せず、プリミティブで受け取る（lib.rs「層の境界」）。
//!
//! # 判定規則
//!
//! 正解率の Wilson 95% 信頼区間（[`crate::wilson::wilson_ci95`]。式の出典は同モジュール）を
//! `[lo, hi]`、基準を `threshold = min_accuracy_bp / 10000` とする。
//!
//! - `lo >= threshold`（許容差 [`ACCEPTANCE_TOLERANCE`]）: [`AcceptanceVerdict::Pass`]
//! - `hi < threshold`（同許容差）: [`AcceptanceVerdict::Fail`]
//! - それ以外（区間が基準をまたぐ）: [`AcceptanceVerdict::Undeterminable`]
//! - `total == 0`: [`AcceptanceVerdict::Undeterminable`]（件数不足を合格扱いにしない。REQ-24）
//!
//! 点推定値ではなく区間で決めるのは、少件数の偶然で合格・不合格と言い切らないため。
//! 証拠の種別: テストハーネス（手計算値との照合。実機測定ではない）。

use crate::wilson::wilson_ci95;

/// 浮動小数の比較に使う許容差（評価契約の 1e-9）。下限が基準とちょうど等しい場合を
/// 丸め誤差で取りこぼさないために明示する。
pub const ACCEPTANCE_TOLERANCE: f64 = 1e-9;

/// 基準の上限（1 万分率で 100%）。`fandhe_edge_core::definition::MAX_MIN_ACCURACY_BP` と同値。
const MAX_BASIS_POINTS: u32 = 10_000;

/// 合否基準に対する 3 値判定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptanceVerdict {
    /// 区間の下限が基準以上（基準を満たすと言える）。
    Pass,
    /// 区間の上限が基準未満（基準を満たさないと言える）。
    Fail,
    /// 判定できない（区間が基準をまたぐ・件数 0）。合格扱いにしない。
    Undeterminable,
}

/// [`judge_min_accuracy`] の入力不正（検証済みの入力では起きない。fail-closed の経路）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptanceInputError {
    /// `min_accuracy_bp` が 10000 を超えている。
    BasisPointsOutOfRange,
    /// `correct` が `total` を超えている。
    CorrectExceedsTotal,
}

impl std::fmt::Display for AcceptanceInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcceptanceInputError::BasisPointsOutOfRange => {
                write!(f, "min_accuracy_bp must be at most {MAX_BASIS_POINTS}")
            }
            AcceptanceInputError::CorrectExceedsTotal => {
                write!(f, "correct must not exceed total")
            }
        }
    }
}

impl std::error::Error for AcceptanceInputError {}

/// 正解数 `correct`／総数 `total` が正解率の下限 `min_accuracy_bp`（1 万分率）を
/// 満たすかを Wilson 95% 区間で 3 値判定する（REQ-24・REQ-33・#328）。
///
/// # Errors
/// `min_accuracy_bp > 10000` または `correct > total` のとき。
pub fn judge_min_accuracy(
    correct: u64,
    total: u64,
    min_accuracy_bp: u32,
) -> Result<AcceptanceVerdict, AcceptanceInputError> {
    if min_accuracy_bp > MAX_BASIS_POINTS {
        return Err(AcceptanceInputError::BasisPointsOutOfRange);
    }
    if correct > total {
        return Err(AcceptanceInputError::CorrectExceedsTotal);
    }
    if total == 0 {
        return Ok(AcceptanceVerdict::Undeterminable);
    }
    let Some(interval) = wilson_ci95(correct, total) else {
        return Ok(AcceptanceVerdict::Undeterminable);
    };
    let threshold = f64::from(min_accuracy_bp) / f64::from(MAX_BASIS_POINTS);
    Ok(classify(interval.lo(), interval.hi(), threshold))
}

/// 区間 `[lo, hi]` と基準 `threshold` から 3 値を決める（許容差つき。単体テストの seam）。
fn classify(lo: f64, hi: f64, threshold: f64) -> AcceptanceVerdict {
    if lo >= threshold - ACCEPTANCE_TOLERANCE {
        AcceptanceVerdict::Pass
    } else if hi < threshold - ACCEPTANCE_TOLERANCE {
        AcceptanceVerdict::Fail
    } else {
        AcceptanceVerdict::Undeterminable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // n=12・z=1.96 の Wilson 区間（手計算）: c=12 は lo = 12/(12+z^2) ≈ 0.757493、
    // c=0 は hi = z^2/(12+z^2) ≈ 0.242495。

    /// REQ-24・#328: 全問正解でも下限が基準未満なら判定不能、以下なら合格。
    #[test]
    fn req24_issue328_all_correct_pass_and_undeterminable_boundary() {
        assert_eq!(
            judge_min_accuracy(12, 12, 7500),
            Ok(AcceptanceVerdict::Pass)
        );
        assert_eq!(
            judge_min_accuracy(12, 12, 7600),
            Ok(AcceptanceVerdict::Undeterminable)
        );
    }

    /// REQ-24・#328: 全問不正解で上限が基準未満なら不合格、以上なら判定不能。
    #[test]
    fn req24_issue328_all_wrong_fail_and_undeterminable_boundary() {
        assert_eq!(judge_min_accuracy(0, 12, 2500), Ok(AcceptanceVerdict::Fail));
        assert_eq!(
            judge_min_accuracy(0, 12, 2400),
            Ok(AcceptanceVerdict::Undeterminable)
        );
    }

    /// REQ-24・#328: 区間が基準をまたぐ中間の正解率は判定不能。
    #[test]
    fn req24_issue328_straddling_interval_is_undeterminable() {
        assert_eq!(
            judge_min_accuracy(6, 12, 5000),
            Ok(AcceptanceVerdict::Undeterminable)
        );
    }

    /// REQ-24・#328: 基準 0 は件数があれば常に合格、件数 0 は基準 0 でも判定不能。
    #[test]
    fn req24_issue328_zero_threshold_and_zero_total() {
        assert_eq!(judge_min_accuracy(0, 12, 0), Ok(AcceptanceVerdict::Pass));
        assert_eq!(
            judge_min_accuracy(0, 0, 0),
            Ok(AcceptanceVerdict::Undeterminable)
        );
    }

    /// REQ-24・#328: 入力不正は `Err`（合否を返さない）。
    #[test]
    fn req24_issue328_invalid_inputs_are_errors() {
        assert_eq!(
            judge_min_accuracy(1, 2, 10_001),
            Err(AcceptanceInputError::BasisPointsOutOfRange)
        );
        assert_eq!(
            judge_min_accuracy(3, 2, 5000),
            Err(AcceptanceInputError::CorrectExceedsTotal)
        );
        assert_eq!(
            AcceptanceInputError::CorrectExceedsTotal.to_string(),
            "correct must not exceed total"
        );
    }

    /// REQ-24・#328: 許容差 1e-9 の内外（下限が基準とちょうど等しければ合格）。
    #[test]
    fn req24_issue328_classify_tolerance_boundaries() {
        let t = 0.75;
        assert_eq!(classify(t, 1.0, t), AcceptanceVerdict::Pass);
        assert_eq!(classify(t - 0.5e-9, 1.0, t), AcceptanceVerdict::Pass);
        assert_eq!(
            classify(t - 2e-9, 1.0, t),
            AcceptanceVerdict::Undeterminable
        );
        assert_eq!(classify(0.0, t - 2e-9, t), AcceptanceVerdict::Fail);
        assert_eq!(
            classify(0.0, t - 0.5e-9, t),
            AcceptanceVerdict::Undeterminable
        );
    }
}
