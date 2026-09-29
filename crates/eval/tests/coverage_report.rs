//! 評価データ適用時の coverage 記録・表示の結合テスト（REQ-22 境界値・
//! TASK-22.3・issue #98）。
//!
//! 証拠の種別: テストハーネス（手計算の件数との照合。PoC-12・PoC-25 の実機値は
//! 再現しない）。`docs/spec` は参照しない。浮動小数の許容差は 1e-9。
//! 80% は参考値で合否条件ではないため、80% 未満でも `Ok` になることを確認する。

use fandhe_edge_eval::abstention::{
    OutOfScopeLabel, compare_abstention, compare_abstention_with_out_of_scope,
};
use fandhe_edge_eval::calibration::{Calibration, CalibrationRecord, calibrate};
use fandhe_edge_eval::coverage::ReferenceTargetComparison;

const FLOAT_EPSILON: f64 = 1e-9;
const LABELS: [&str; 3] = ["l0", "l1", "out_of_scope"];

const HIGH_OOS: [f64; 3] = [0.0, 0.0, 8.0];
const LOW_OOS: [f64; 3] = [0.0, 0.05, 0.1];
const HIGH_L0: [f64; 3] = [8.0, 0.0, 0.0];
const LOW_L0: [f64; 3] = [0.3, 0.1, 0.0];
const MID_L0: [f64; 3] = [1.0, 0.0, 0.0];

type Rows = Vec<(String, Vec<f64>)>;

fn rows_of(gold: &str, logits: &[f64], n: usize) -> Rows {
    (0..n)
        .map(|_| (gold.to_string(), logits.to_vec()))
        .collect()
}

fn as_records(rows: &Rows) -> Vec<CalibrationRecord<'_>> {
    rows.iter()
        .map(|(gold, logits)| CalibrationRecord {
            gold: gold.as_str(),
            logits: logits.as_slice(),
        })
        .collect()
}

/// 校正用 validation（32 行）。確信度の低い 6 行だけが τ 未満で、
/// validation coverage は 26/32。
fn calibrated() -> Calibration {
    let mut rows = rows_of("l1", &LOW_L0, 3);
    rows.extend(rows_of("l1", &LOW_OOS, 3));
    rows.extend(rows_of("l0", &MID_L0, 6));
    rows.extend(rows_of("l0", &HIGH_L0, 16));
    rows.extend(rows_of("out_of_scope", &HIGH_OOS, 4));
    calibrate(&LABELS, &as_records(&rows)).expect("valid validation")
}

/// 評価データ 5 件: 高確信 l0 ×2、低確信 l0（gold l1）、低確信 対象外、
/// 高確信 対象外。
fn eval_rows() -> Rows {
    let mut rows = rows_of("l0", &HIGH_L0, 2);
    rows.extend(rows_of("l1", &LOW_L0, 1));
    rows.extend(rows_of("out_of_scope", &LOW_OOS, 1));
    rows.extend(rows_of("out_of_scope", &HIGH_OOS, 1));
    rows
}

/// REQ-22 境界値: 評価データで 5 件中 3 件だけ答えた場合の coverage が記録され、
/// 参考目標（80%）未満でも `Ok`（合否に影響しない）。
#[test]
fn req22_coverage_recorded_below_reference_target_is_informational() {
    let calibration = calibrated();
    let rows = eval_rows();
    let comparison = compare_abstention(&LABELS, &calibration, &as_records(&rows))
        .expect("coverage below 80% must not fail");
    let coverage = comparison.coverage();
    assert_eq!(coverage.coverage().numerator(), 3);
    assert_eq!(coverage.coverage().denominator(), 5);
    assert!((coverage.coverage().value() - 0.6).abs() < FLOAT_EPSILON);
    assert_eq!(coverage.total(), 5);
    assert_eq!(coverage.answered(), 3);
    assert_eq!(coverage.abstained(), 2);
    assert_eq!(coverage.out_of_scope(), 0);
    assert_eq!(coverage.in_scope_answered(), 3);
    assert_eq!(
        coverage.reference_target_comparison(),
        ReferenceTargetComparison::Below
    );
    assert_eq!(coverage.validation_coverage().numerator(), 26);
    assert_eq!(coverage.validation_coverage().denominator(), 32);
}

/// REQ-22 境界値: coverage が 80% 未満でも他の指標（誤り率・件数）は既知解の
/// とおりで、coverage に依存しない。
#[test]
fn req22_coverage_below_target_does_not_change_other_results() {
    let calibration = calibrated();
    let rows = eval_rows();
    let comparison = compare_abstention(&LABELS, &calibration, &as_records(&rows)).unwrap();
    // 保留なし: 5 件中 4 件正解 → 誤り率 1/5。
    assert_eq!(comparison.unconditional_error().numerator(), 1);
    assert_eq!(comparison.unconditional_error().denominator(), 5);
    // 保留込み: 採用 3 件すべて正解 → 誤り率 0/3。
    let adopted = comparison.adopted_error().expect("3 adopted rows");
    assert_eq!(adopted.numerator(), 0);
    assert_eq!(adopted.denominator(), 3);
    assert_eq!(comparison.out_of_scope(), 0);
    assert_eq!(comparison.with_abstention().outcome_counts.abstain, 2);
    assert_eq!(comparison.without_abstention().outcome_counts.abstain, 0);
}

/// REQ-22 境界値: 4/5 ちょうどは `AtOrAbove`、それ未満は `Below`。
#[test]
fn req22_coverage_exactly_reference_target() {
    let calibration = calibrated();
    let mut rows = rows_of("l0", &HIGH_L0, 4);
    rows.extend(rows_of("l1", &LOW_L0, 1));
    let at = compare_abstention(&LABELS, &calibration, &as_records(&rows)).unwrap();
    assert_eq!(at.coverage().coverage().numerator(), 4);
    assert_eq!(at.coverage().coverage().denominator(), 5);
    assert_eq!(
        at.coverage().reference_target_comparison(),
        ReferenceTargetComparison::AtOrAbove
    );

    let mut rows = rows_of("l0", &HIGH_L0, 4);
    rows.extend(rows_of("l1", &LOW_L0, 2));
    let below = compare_abstention(&LABELS, &calibration, &as_records(&rows)).unwrap();
    assert_eq!(below.coverage().coverage().numerator(), 4);
    assert_eq!(below.coverage().coverage().denominator(), 6);
    assert_eq!(
        below.coverage().reference_target_comparison(),
        ReferenceTargetComparison::Below
    );
}

/// REQ-22 境界値: 全件保留なら coverage は 0/n（`Some`）で、採用誤り率は `None`。
#[test]
fn req22_coverage_all_abstained_is_zero() {
    let calibration = calibrated();
    let rows = rows_of("l0", &LOW_L0, 3);
    let comparison = compare_abstention(&LABELS, &calibration, &as_records(&rows)).unwrap();
    assert_eq!(comparison.coverage().coverage().numerator(), 0);
    assert_eq!(comparison.coverage().coverage().denominator(), 3);
    assert_eq!(comparison.coverage().abstained(), 3);
    assert!(comparison.adopted_error().is_none());
}

/// REQ-22 境界値・TASK-22.2: 対象外は答えた側に数える。対象外ラベル未指定の
/// 同データとは coverage が異なる（PoC-12 の「確信度 ≥ τ の割合」との乖離）。
#[test]
fn req22_coverage_counts_out_of_scope_as_answered() {
    let calibration = calibrated();
    let rows = eval_rows();
    let oos = OutOfScopeLabel::new(&calibration, "out_of_scope").unwrap();
    let with =
        compare_abstention_with_out_of_scope(&LABELS, &calibration, Some(&oos), &as_records(&rows))
            .unwrap();
    let c = with.coverage();
    assert_eq!(c.answered(), 4);
    assert_eq!(c.abstained(), 1);
    assert_eq!(c.out_of_scope(), 2);
    assert_eq!(c.in_scope_answered(), 2);
    assert_eq!(c.coverage().numerator(), 4);
    assert_eq!(c.coverage().denominator(), 5);
    assert_eq!(
        c.reference_target_comparison(),
        ReferenceTargetComparison::AtOrAbove
    );

    let without = compare_abstention(&LABELS, &calibration, &as_records(&rows)).unwrap();
    assert_eq!(without.coverage().answered(), 3);
}

/// REQ-22 境界値: coverage の分子は保留込み側の採用件数（`adopted_decision` の
/// 分母）と一致する。
#[test]
fn req22_coverage_numerator_equals_adopted_denominator() {
    let calibration = calibrated();
    let rows = eval_rows();
    let comparison = compare_abstention(&LABELS, &calibration, &as_records(&rows)).unwrap();
    let adopted = comparison
        .with_abstention()
        .accuracy
        .adopted_decision
        .expect("some adopted rows");
    assert_eq!(
        comparison.coverage().coverage().numerator(),
        adopted.denominator()
    );
}

/// REQ-22 境界値: 表示文字列（英語・固定精度・参考値である旨を含む）。
#[test]
fn req22_coverage_display_text() {
    let calibration = calibrated();
    let rows = eval_rows();
    let oos = OutOfScopeLabel::new(&calibration, "out_of_scope").unwrap();
    let comparison =
        compare_abstention_with_out_of_scope(&LABELS, &calibration, Some(&oos), &as_records(&rows))
            .unwrap();
    assert_eq!(
        comparison.coverage().to_string(),
        "coverage 4/5 (80.0%) [in-scope 2, out-of-scope 2, abstained 1]; \
         validation coverage 26/32 (81.2%); reference target 80.0% (at or above; \
         informational only, not a pass/fail criterion)"
    );
}
