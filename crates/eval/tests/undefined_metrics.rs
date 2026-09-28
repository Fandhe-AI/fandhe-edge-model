//! 分母 0 の指標の異常系結合テスト（REQ-24 異常系・TASK-24.2・issue #61）。
//!
//! 出典: PoC-9 `03-poc/evaluation-contract/fixtures/anomaly/09-unseen-class`・
//! `fixtures/anomaly/11-all-abstain`。`docs/spec` は読み込まず、期待値は
//! テストファイルへ直接書く（`.claude/rules/spec-reference.md` のビルド
//! 独立方針）。
//!
//! 証拠の種別: テストハーネス（PoC-9 実測・手計算値との照合）。
//! 浮動小数の比較は許容差 1e-9 で行う（評価契約:
//! `.claude/rules/evaluation-contract.md`）。

use fandhe_edge_eval::metrics::{EvalRecord, Outcome, evaluate_single_select};

const EPSILON: f64 = 1e-9;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

/// PoC-9 `09-unseen-class` を移植する。
///
/// labels: A, B, C。gold は A, A, B, B。予測は A, B, B, A（全件 ok）。
/// C はラベル集合に宣言されているが、gold にも予測にも一度も現れない
/// （support=0・predicted_count=0）ため、precision・recall・f1 のいずれも
/// `None`（未定義）になり、Macro-F1 の平均から除外される。
///
/// A: tp=1（1件目）, support=2, predicted_count=2（1,4件目）→ P=R=F1=1/2。
/// B: tp=1（3件目）, support=2, predicted_count=2（2,3件目）→ P=R=F1=1/2。
#[test]
fn unseen_class_is_null_and_excluded_from_macro_f1() {
    let out_a = Outcome::Label("A".to_string());
    let out_b = Outcome::Label("B".to_string());
    let records = vec![
        EvalRecord {
            gold: "A",
            outcome: &out_a,
        },
        EvalRecord {
            gold: "A",
            outcome: &out_b,
        },
        EvalRecord {
            gold: "B",
            outcome: &out_b,
        },
        EvalRecord {
            gold: "B",
            outcome: &out_a,
        },
    ];
    let labels = ["A", "B", "C"];
    let metrics = evaluate_single_select(&labels, &records).expect("valid input");

    let a = &metrics.per_label[0];
    assert_eq!(a.label, "A");
    assert!(approx_eq(a.precision.expect("defined"), 0.5));
    assert!(approx_eq(a.recall.expect("defined"), 0.5));
    assert!(approx_eq(a.f1.expect("defined"), 0.5));

    let b = &metrics.per_label[1];
    assert_eq!(b.label, "B");
    assert!(approx_eq(b.precision.expect("defined"), 0.5));
    assert!(approx_eq(b.recall.expect("defined"), 0.5));
    assert!(approx_eq(b.f1.expect("defined"), 0.5));

    // C: support=0・predicted_count=0 → precision・recall・f1 すべて None。
    let c = &metrics.per_label[2];
    assert_eq!(c.label, "C");
    assert_eq!(c.support, 0);
    assert_eq!(c.predicted_count, 0);
    assert_eq!(c.precision, None, "C の precision は未定義（null）");
    assert_eq!(c.recall, None, "C の recall は未定義（null）");
    assert_eq!(c.f1, None, "C の f1 は未定義（null）");

    // Macro-F1 は A・B だけの平均（C は除外）。
    assert!(approx_eq(metrics.macro_f1.value().expect("defined"), 0.5));
    assert_eq!(
        metrics.macro_f1.excluded_labels(),
        ["C".to_string()],
        "受入基準 2: 平均から外したラベル（C）が列挙される"
    );
}

/// PoC-9 `11-all-abstain` を移植する。
///
/// labels: A, B。gold は A, B, A。全件 `Outcome::Abstain`（推論が保留）。
/// 予測が 1 件も無いため、A・B ともに precision が `None`（未定義）になる
/// （受入基準 1: 予測 0 件のラベルの適合率が null として返る、の直接確認）。
/// recall は support > 0 のため定義でき、f1 も `Some(0.0)` になるため
/// `excluded_labels` は空になる。
#[test]
fn all_abstain_precision_is_null_but_not_excluded() {
    let records = vec![
        EvalRecord {
            gold: "A",
            outcome: &Outcome::Abstain,
        },
        EvalRecord {
            gold: "B",
            outcome: &Outcome::Abstain,
        },
        EvalRecord {
            gold: "A",
            outcome: &Outcome::Abstain,
        },
    ];
    let labels = ["A", "B"];
    let metrics = evaluate_single_select(&labels, &records).expect("valid input");

    assert_eq!(metrics.outcome_counts.abstain, 3);
    assert_eq!(metrics.outcome_counts.ok, 0);

    let a = &metrics.per_label[0];
    assert_eq!(a.label, "A");
    assert_eq!(a.support, 2);
    assert_eq!(a.predicted_count, 0);
    assert_eq!(a.precision, None, "A: 予測 0 件のため precision は null");
    assert!(approx_eq(a.recall.expect("defined"), 0.0));
    assert!(approx_eq(a.f1.expect("defined"), 0.0));

    let b = &metrics.per_label[1];
    assert_eq!(b.label, "B");
    assert_eq!(b.support, 1);
    assert_eq!(b.predicted_count, 0);
    assert_eq!(b.precision, None, "B: 予測 0 件のため precision は null");
    assert!(approx_eq(b.recall.expect("defined"), 0.0));
    assert!(approx_eq(b.f1.expect("defined"), 0.0));

    // f1 はどちらも Some(0.0) のため、excluded_labels は空。
    assert_eq!(
        metrics.macro_f1.excluded_labels(),
        &[] as &[String],
        "precision が null でも f1 が定義できるラベルは除外されない"
    );
    assert!(approx_eq(metrics.macro_f1.value().expect("defined"), 0.0));

    // accuracy: 全件 abstain のため overall は 0/3、adopted_decision は
    // 分母 0（n_total - abstain = 0）で null（評価契約: 分母 0 は null）。
    assert_eq!(metrics.accuracy.overall.numerator(), 0);
    assert_eq!(metrics.accuracy.overall.denominator(), 3);
    assert_eq!(
        metrics.accuracy.adopted_decision, None,
        "全件 abstain のとき adopted_decision の分母は 0 のため null"
    );
}
