//! 「対象外」ラベルによる扱いの結合テスト（REQ-22 異常系・TASK-22.2・issue #97）。
//!
//! 証拠の種別: テストハーネス（閉形式の式・手計算の件数との照合）。
//! `docs/spec` は参照しない（`.claude/rules/spec-reference.md`「運用」）。
//! 浮動小数の許容差は 1e-9（`.claude/rules/evaluation-contract.md`「決定性」）。
//! 期待する確信度は `chosen_temperature()` から `1 / Σ exp((z_j - z_max)/T)` で
//! 独立に計算し、テスト対象の出力をそのまま貼り付けない。

use fandhe_edge_eval::abstention::{
    AbstentionDecision, OutOfScopeLabel, compare_abstention, compare_abstention_with_out_of_scope,
    decide_abstention, decide_abstention_with_out_of_scope,
};
use fandhe_edge_eval::calibration::{Calibration, CalibrationError, CalibrationRecord, calibrate};
use fandhe_edge_eval::metrics::Outcome;

const FLOAT_EPSILON: f64 = 1e-9;

const LABELS: [&str; 3] = ["l0", "l1", "out_of_scope"];

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < FLOAT_EPSILON
}

/// 閉形式の top1 確率 `1 / Σ_j exp((z_j - z_max)/T)`。
fn expected_top1(logits: &[f64], temperature: f64) -> f64 {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits.iter().map(|z| ((z - max) / temperature).exp()).sum();
    1.0 / sum
}

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

/// 校正用 validation（32 行）。τ は top1 確率の 20% 分位点（昇順で
/// `floor(0.2*(n-1)) = 6` 番目）なので、確信度の低い 6 行（LOW_L0 ×3・
/// LOW_OOS ×3）だけが τ を下回り、7 番目の値（MID_L0）が τ になる。
fn calibrated() -> Calibration {
    let mut rows = rows_of("l1", &LOW_L0, 3);
    rows.extend(rows_of("l1", &LOW_OOS, 3));
    rows.extend(rows_of("l0", &MID_L0, 6));
    rows.extend(rows_of("l0", &HIGH_L0, 16));
    rows.extend(rows_of("out_of_scope", &HIGH_OOS, 4));
    calibrate(&LABELS, &as_records(&rows)).expect("valid validation")
}

/// 前提: 高確信度の行は τ 以上、低確信度の行は τ 未満（テスト自身の妥当性検査）。
fn assert_preconditions(calibration: &Calibration) {
    let t = calibration.chosen_temperature();
    let tau = calibration.threshold();
    assert!(
        expected_top1(&HIGH_OOS, t) >= tau,
        "HIGH_OOS must be >= tau"
    );
    assert!(expected_top1(&HIGH_L0, t) >= tau, "HIGH_L0 must be >= tau");
    assert!(expected_top1(&LOW_OOS, t) < tau, "LOW_OOS must be < tau");
    assert!(expected_top1(&LOW_L0, t) < tau, "LOW_L0 must be < tau");
}

/// REQ-22 異常系（受入の中核）: 同一の校正・同一のロジット（argmax = 対象外、
/// 確信度 < τ）で、対象外指定ありなら「対象外」、指定なしなら「保留」になる。
#[test]
fn req22_out_of_scope_label_overrides_threshold_abstain() {
    let calibration = calibrated();
    assert_preconditions(&calibration);
    let oos = OutOfScopeLabel::new(&calibration, "out_of_scope").expect("label exists");
    assert_eq!(oos.index(), 2);
    assert_eq!(oos.id(), "out_of_scope");

    let with =
        decide_abstention_with_out_of_scope(&calibration, Some(&oos), &LOW_OOS).expect("valid");
    match &with {
        AbstentionDecision::OutOfScope {
            label_index,
            label,
            confidence,
        } => {
            assert_eq!(*label_index, 2);
            assert_eq!(label, "out_of_scope");
            let expected = expected_top1(&LOW_OOS, calibration.chosen_temperature());
            assert!(
                approx_eq(*confidence, expected),
                "{confidence} vs {expected}"
            );
        }
        other => panic!("expected OutOfScope, got {other:?}"),
    }
    assert_eq!(
        with.to_outcome(),
        Outcome::Label("out_of_scope".to_string())
    );

    let without = decide_abstention(&calibration, &LOW_OOS).expect("valid");
    assert!(matches!(without, AbstentionDecision::Abstain { .. }));
    assert_eq!(without.to_outcome(), Outcome::Abstain);
    // `None` 指定は従来関数と同じ。
    assert_eq!(
        decide_abstention_with_out_of_scope(&calibration, None, &LOW_OOS).expect("valid"),
        without
    );
}

/// 確信度が τ 以上でも argmax が対象外なら `Adopt` ではなく `OutOfScope`。
#[test]
fn req22_out_of_scope_label_with_high_confidence() {
    let calibration = calibrated();
    assert_preconditions(&calibration);
    let oos = OutOfScopeLabel::new(&calibration, "out_of_scope").expect("label exists");
    let d =
        decide_abstention_with_out_of_scope(&calibration, Some(&oos), &HIGH_OOS).expect("valid");
    assert!(matches!(
        d,
        AbstentionDecision::OutOfScope { label_index: 2, .. }
    ));
    let plain = decide_abstention(&calibration, &HIGH_OOS).expect("valid");
    assert!(matches!(
        plain,
        AbstentionDecision::Adopt { label_index: 2, .. }
    ));
}

/// argmax が通常ラベルなら、対象外指定ありでもしきい値判定が維持される。
#[test]
fn req22_threshold_still_applies_to_regular_labels() {
    let calibration = calibrated();
    assert_preconditions(&calibration);
    let oos = OutOfScopeLabel::new(&calibration, "out_of_scope").expect("label exists");
    let low =
        decide_abstention_with_out_of_scope(&calibration, Some(&oos), &LOW_L0).expect("valid");
    assert!(matches!(low, AbstentionDecision::Abstain { .. }));
    let high =
        decide_abstention_with_out_of_scope(&calibration, Some(&oos), &HIGH_L0).expect("valid");
    match high {
        AbstentionDecision::Adopt {
            label_index, label, ..
        } => {
            assert_eq!(label_index, 0);
            assert_eq!(label, "l0");
        }
        other => panic!("expected Adopt, got {other:?}"),
    }
}

/// 同値（タイ）は宣言順の先頭が勝つ。対象外が末尾なら通常ラベルが、先頭なら
/// 対象外が argmax になる。
#[test]
fn req22_tie_follows_declaration_order() {
    let calibration = calibrated();
    let oos = OutOfScopeLabel::new(&calibration, "out_of_scope").expect("label exists");
    let tie = [0.0, 0.0, 0.0];
    let d = decide_abstention_with_out_of_scope(&calibration, Some(&oos), &tie).expect("valid");
    assert!(!matches!(d, AbstentionDecision::OutOfScope { .. }));

    // 対象外ラベルが宣言順の先頭のラベル集合。
    let labels = ["out_of_scope", "a", "b"];
    let mut rows = rows_of("a", &[3.0, 0.0, 0.0], 6);
    rows.extend(rows_of("out_of_scope", &[3.0, 0.0, 0.0], 2));
    rows.extend(rows_of("b", &[0.0, 0.0, 3.0], 4));
    let cal_first = calibrate(&labels, &as_records(&rows)).expect("valid");
    let oos_first = OutOfScopeLabel::new(&cal_first, "out_of_scope").expect("label exists");
    assert_eq!(oos_first.index(), 0);
    let d = decide_abstention_with_out_of_scope(&cal_first, Some(&oos_first), &tie).expect("valid");
    assert!(matches!(
        d,
        AbstentionDecision::OutOfScope { label_index: 0, .. }
    ));
}

/// ラベル集合に無い ID の指定は fail-closed。
#[test]
fn req22_unknown_out_of_scope_label_is_rejected() {
    let calibration = calibrated();
    let err = OutOfScopeLabel::new(&calibration, "missing").expect_err("unknown id");
    assert_eq!(
        err,
        CalibrationError::UnknownOutOfScopeLabel {
            id: "missing".to_string()
        }
    );
}

/// 別の校正結果（同じ位置に別 ID）で作った指定は、判定・比較の両方で拒否する。
#[test]
fn req22_out_of_scope_label_from_other_calibration_is_rejected() {
    let calibration = calibrated();
    let other_labels = ["l0", "l1", "other"];
    let rows = {
        let mut r = rows_of("l0", &HIGH_L0, 6);
        r.extend(rows_of("l1", &LOW_L0, 4));
        r.extend(rows_of("other", &HIGH_OOS, 4));
        r
    };
    let other = calibrate(&other_labels, &as_records(&rows)).expect("valid");
    let foreign = OutOfScopeLabel::new(&other, "other").expect("label exists");

    let expected = CalibrationError::OutOfScopeLabelMismatch {
        index: 2,
        expected: "other".to_string(),
        found: Some("out_of_scope".to_string()),
    };
    assert_eq!(
        decide_abstention_with_out_of_scope(&calibration, Some(&foreign), &HIGH_OOS)
            .expect_err("mismatch"),
        expected
    );
    let eval_rows = rows_of("l0", &HIGH_L0, 2);
    assert_eq!(
        compare_abstention_with_out_of_scope(
            &LABELS,
            &calibration,
            Some(&foreign),
            &as_records(&eval_rows)
        )
        .expect_err("mismatch"),
        expected
    );
}

/// `None` 指定の比較結果は従来関数と一致し、対象外件数は 0。
#[test]
fn req22_compare_without_out_of_scope_matches_existing() {
    let calibration = calibrated();
    let mut rows = rows_of("l0", &HIGH_L0, 3);
    rows.extend(rows_of("l1", &LOW_L0, 2));
    let records = as_records(&rows);
    let base = compare_abstention(&LABELS, &calibration, &records).expect("valid");
    let none =
        compare_abstention_with_out_of_scope(&LABELS, &calibration, None, &records).expect("valid");
    assert_eq!(base, none);
    assert_eq!(none.out_of_scope(), 0);
}

/// 既知解: 対象外 argmax の行・通常ラベルで保留になる行・採用行・gold=対象外の行を
/// 混ぜた評価データで、件数・誤り率を手計算値と照合する。
///
/// 評価データ（12 行）:
/// - HIGH_L0 gold l0 ×3（採用・正解）
/// - LOW_L0 gold l1 ×2（通常ラベルで保留。保留なし側は l0 予測で誤り）
/// - LOW_OOS gold out_of_scope ×3（対象外。保留なし側は対象外予測で正解）
/// - HIGH_OOS gold l1 ×2（対象外。gold は通常ラベルなので誤り）
/// - LOW_OOS gold l0 ×2（対象外。gold は通常ラベルなので誤り）
#[test]
fn req22_known_answer_counts_with_out_of_scope() {
    let calibration = calibrated();
    assert_preconditions(&calibration);
    let oos = OutOfScopeLabel::new(&calibration, "out_of_scope").expect("label exists");

    let mut rows = rows_of("l0", &HIGH_L0, 3);
    rows.extend(rows_of("l1", &LOW_L0, 2));
    rows.extend(rows_of("out_of_scope", &LOW_OOS, 3));
    rows.extend(rows_of("l1", &HIGH_OOS, 2));
    rows.extend(rows_of("l0", &LOW_OOS, 2));
    let records = as_records(&rows);
    assert_eq!(records.len(), 12);

    let with = compare_abstention_with_out_of_scope(&LABELS, &calibration, Some(&oos), &records)
        .expect("valid");
    // 対象外 argmax = 3 + 2 + 2 = 7 件。
    assert_eq!(with.out_of_scope(), 7);
    // 保留は通常ラベルの LOW_L0 の 2 件のみ。
    assert_eq!(with.with_abstention().outcome_counts.abstain, 2);
    // 保留なし（全行 argmax 採用）の正解は 3（HIGH_L0）+ 3（gold=対象外）= 6 / 12。
    let uncond = with.unconditional_error();
    assert_eq!((uncond.numerator(), uncond.denominator()), (6, 12));
    // 保留込みの採用は 10 件。正解は 3 + 3 = 6、誤りは 4 / 10。
    let adopted = with.adopted_error().expect("adopted rows exist");
    assert_eq!((adopted.numerator(), adopted.denominator()), (4, 10));

    // 指定なしとの差: LOW_OOS の 5 件が保留になり、採用は 5 件へ減る。
    let without =
        compare_abstention_with_out_of_scope(&LABELS, &calibration, None, &records).expect("valid");
    assert_eq!(without.out_of_scope(), 0);
    assert_eq!(without.with_abstention().outcome_counts.abstain, 7);
    let adopted_without = without.adopted_error().expect("adopted rows exist");
    // 採用 5 件（HIGH_L0 ×3 正解、HIGH_OOS ×2 gold l1 で誤り）→ 誤り 2 / 5。
    assert_eq!(
        (adopted_without.numerator(), adopted_without.denominator()),
        (2, 5)
    );
    // 保留なし側の誤り率は対象外指定の有無で変わらない。
    assert_eq!(without.unconditional_error(), uncond);
}
