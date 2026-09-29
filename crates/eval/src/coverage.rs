//! 評価データ適用時の coverage の記録と表示（REQ-22 境界値・TASK-22.3・issue #98）。
//!
//! 校正済みの T・τ を評価データへ適用したとき、保留にならず実際に答えた行の
//! 割合（coverage）を [`CoverageReport`] として記録する。記録は
//! [`crate::abstention::compare_abstention_with_out_of_scope`] の同一走査で行い
//! （[`crate::abstention::AbstentionComparison::coverage`]）、件数を別経路で
//! 再計算しない。評価ロジックは TASK-24.1 の 1 つだけに集約する方針で、本モジュール
//! は集計済みの件数から比率を組み立てて表示するだけである。
//!
//! # 定義
//!
//! coverage = (全件 − 保留件数) / 全件。argmax が「対象外」ラベルの行
//! （TASK-22.2）は保留ではなく答えた側に数える。これは `adopted_error` の分母
//! （全件 − 保留）と同じ数え方で、内訳は [`CoverageReport::out_of_scope`] と
//! [`CoverageReport::in_scope_answered`] で読み分けられる。対象外ラベルを指定した
//! 場合、coverage は「確信度 ≥ τ の割合」（PoC-12 の定義）とは一致しない
//! （確信度が τ 未満でも argmax が対象外なら答えた側）。指定しない場合のみ一致する。
//!
//! # 80% は参考値であり合否条件ではない
//!
//! PoC-12 の実測では validation の coverage 80.2% に対し評価データでは
//! 65.5〜78.8%、PoC-25 でも 80% を下回った（証拠種別: 実機。本モジュールでは再現
//! しない）。Conditional Go 条件により 80% 到達は合否条件にしない。そのため
//! 参考目標との比較は真偽値ではなく情報用の [`ReferenceTargetComparison`] で表し、
//! coverage の値は `compare_abstention*` の戻り値・エラーに影響しない。
//!
//! # 対象外
//!
//! JSON への直列化と CLI `evaluate` 工程への配線は CLI 側（issue #140・
//! TASK-33.x）の責務。REQ-29 の診断レポート（abstain_rate 等）は別タスク。

use std::fmt;

use crate::calibration::{CalibrationError, TARGET_COVERAGE_DENOMINATOR};
use crate::metrics::Ratio;

/// 参考目標（80%）との比較結果。情報用であり合否ではない（モジュール doc）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceTargetComparison {
    /// coverage が参考目標以上（境界を含む）。
    AtOrAbove,
    /// coverage が参考目標未満。
    Below,
}

/// 評価データへの適用結果から記録した coverage（REQ-22 境界値・TASK-22.3）。
///
/// フィールドは非公開で、構築は [`record_coverage`] に集約する（指標と矛盾する
/// レポートを外部で作らせない）。合否を表すフィールドは持たない。
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageReport {
    total: u64,
    abstained: u64,
    answered: u64,
    out_of_scope: u64,
    coverage: Ratio,
    validation_coverage: Ratio,
    reference_target: Ratio,
    comparison: ReferenceTargetComparison,
}

impl CoverageReport {
    /// 評価データの全件数。
    pub fn total(&self) -> u64 {
        self.total
    }

    /// 保留にならず答えた件数（対象外を含む）。
    pub fn answered(&self) -> u64 {
        self.answered
    }

    /// 保留件数。
    pub fn abstained(&self) -> u64 {
        self.abstained
    }

    /// argmax が対象外ラベルだった件数（答えた側に含まれる）。
    pub fn out_of_scope(&self) -> u64 {
        self.out_of_scope
    }

    /// 対象外を除いて答えた件数（`answered - out_of_scope`。構築時に
    /// `out_of_scope <= answered` を検証済み）。
    pub fn in_scope_answered(&self) -> u64 {
        self.answered.saturating_sub(self.out_of_scope)
    }

    /// 評価データでの coverage（分子 = 答えた件数、分母 = 全件）。
    pub fn coverage(&self) -> Ratio {
        self.coverage
    }

    /// 校正時の validation coverage（対比用。PoC-12 の報告形）。
    pub fn validation_coverage(&self) -> Ratio {
        self.validation_coverage
    }

    /// 参考目標（80% = 4/5。`calibrate` の目標と同じ定数から導出）。
    pub fn reference_target(&self) -> Ratio {
        self.reference_target
    }

    /// 参考目標との比較（情報用。合否には使わない）。
    pub fn reference_target_comparison(&self) -> ReferenceTargetComparison {
        self.comparison
    }
}

fn percent(ratio: &Ratio) -> String {
    format!("{:.1}%", ratio.value() * 100.0)
}

impl fmt::Display for CoverageReport {
    /// 件数と割合のみを英語で出力する（データ本文を含めない）。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cmp = match self.comparison {
            ReferenceTargetComparison::AtOrAbove => "at or above",
            ReferenceTargetComparison::Below => "below",
        };
        write!(
            f,
            "coverage {}/{} ({}) [in-scope {}, out-of-scope {}, abstained {}]; \
             validation coverage {}/{} ({}); reference target {} ({}; informational only, \
             not a pass/fail criterion)",
            self.answered,
            self.total,
            percent(&self.coverage),
            self.in_scope_answered(),
            self.out_of_scope,
            self.abstained,
            self.validation_coverage.numerator(),
            self.validation_coverage.denominator(),
            percent(&self.validation_coverage),
            percent(&self.reference_target),
            cmp,
        )
    }
}

fn internal(detail: &str) -> CalibrationError {
    CalibrationError::Internal {
        detail: detail.to_string(),
    }
}

/// 集計済みの件数から [`CoverageReport`] を作る（`crate::abstention` から呼ぶ）。
///
/// 不整合（`abstained > total`・`out_of_scope > answered`・全件 0）は
/// [`CalibrationError::Internal`] で fail-closed にする。
pub(crate) fn record_coverage(
    total: u64,
    abstained: u64,
    out_of_scope: u64,
    validation_coverage: Ratio,
) -> Result<CoverageReport, CalibrationError> {
    let answered = total
        .checked_sub(abstained)
        .ok_or_else(|| internal("abstained count exceeds total"))?;
    if out_of_scope > answered {
        return Err(internal("out-of-scope count exceeds answered count"));
    }
    let coverage =
        Ratio::new(answered, total).ok_or_else(|| internal("coverage has zero denominator"))?;
    let denominator = u64::try_from(TARGET_COVERAGE_DENOMINATOR)
        .map_err(|_| internal("target denominator conversion failed"))?;
    let target_numerator = denominator
        .checked_sub(1)
        .ok_or_else(|| internal("target denominator underflow"))?;
    let reference_target = Ratio::new(target_numerator, denominator)
        .ok_or_else(|| internal("reference target construction failed"))?;
    // 整数の交差乗算で比較する（f64 の >= を使わない）。u128 は u64 同士の積で
    // オーバーフローしない。
    let lhs = u128::from(answered) * u128::from(reference_target.denominator());
    let rhs = u128::from(total) * u128::from(reference_target.numerator());
    let comparison = if lhs >= rhs {
        ReferenceTargetComparison::AtOrAbove
    } else {
        ReferenceTargetComparison::Below
    };
    Ok(CoverageReport {
        total,
        abstained,
        answered,
        out_of_scope,
        coverage,
        validation_coverage,
        reference_target,
        comparison,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn val() -> Ratio {
        Ratio::new(4, 5).unwrap()
    }

    /// REQ-22 境界値: 不整合な件数は Internal になる。
    #[test]
    fn req22_record_coverage_rejects_inconsistent_counts() {
        assert!(matches!(
            record_coverage(3, 4, 0, val()),
            Err(CalibrationError::Internal { .. })
        ));
        assert!(matches!(
            record_coverage(4, 3, 2, val()),
            Err(CalibrationError::Internal { .. })
        ));
        assert!(matches!(
            record_coverage(0, 0, 0, val()),
            Err(CalibrationError::Internal { .. })
        ));
    }

    /// REQ-22 境界値: 参考目標は 4/5 で、境界（4/5 ちょうど）は AtOrAbove。
    #[test]
    fn req22_reference_target_boundary() {
        let at = record_coverage(5, 1, 0, val()).unwrap();
        assert_eq!(at.reference_target().numerator(), 4);
        assert_eq!(at.reference_target().denominator(), 5);
        assert_eq!(
            at.reference_target_comparison(),
            ReferenceTargetComparison::AtOrAbove
        );
        let below = record_coverage(6, 2, 0, val()).unwrap();
        assert_eq!(
            below.reference_target_comparison(),
            ReferenceTargetComparison::Below
        );
        let above = record_coverage(5, 0, 0, val()).unwrap();
        assert_eq!(
            above.reference_target_comparison(),
            ReferenceTargetComparison::AtOrAbove
        );
    }
}
