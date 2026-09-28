//! 既知解データセットによる結合テスト（REQ-24 正常系・TASK-24.1-1・issue #59）。
//!
//! 出典: PoC-9 `03-poc/evaluation-contract/fixtures/known/single-select`
//! （24 件。gold は "AAAAAABBBBBBCCCCCCDDDDDD"）と manifest.md「11. 凍結前の
//! 修正」（2026-09-23。F1 の定義を `2TP/(2TP+FP+FN)` に修正した経緯）。
//! `docs/spec` は読み込まず、期待値はテストファイルへ直接書く
//! （`.claude/rules/spec-reference.md` のビルド独立方針）。
//!
//! 証拠の種別: テストハーネス（既知解テスト。手計算値との照合）。
//! sklearn 照合（8/8 ケース）は TASK-24.1-2（issue #60）が担当する。
//!
//! 浮動小数の比較は許容差 1e-9 で行う（評価契約: `.claude/rules/evaluation-contract.md`）。

use fandhe_edge_eval::metrics::{ConfusionColumn, EvalRecord, Outcome, evaluate_single_select};

const EPSILON: f64 = 1e-9;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

/// PoC-9 known/single-select の 24 件を移植する。
///
/// gold: 1-6 = A, 7-12 = B, 13-18 = C, 19-24 = D（ss-01 〜 ss-24 に対応）。
/// - ss-05（index 4, gold A）: abstain
/// - ss-06（index 5, gold A）: error
/// - ss-12（index 11, gold B）: 未知ラベル "E"（invalid 扱い）
/// - ss-18（index 17, gold C）: 空文字列 ""（invalid 扱い）
///
/// 残りは、Issue #59 の実装計画に記載された既知解の混同行列
/// （行: A,B,C,D／列: A,B,C,D,invalid,abstain,error）に一致するように構成した。
///
/// - A行: 3,1,0,0,0,1,1
/// - B行: 1,4,0,0,1,0,0
/// - C行: 0,0,5,0,1,0,0
/// - D行: 2,2,2,0,0,0,0
fn known_answer_outcomes() -> Vec<Outcome> {
    vec![
        // gold = A (index 0-5)
        Outcome::Label("A".to_string()),
        Outcome::Label("A".to_string()),
        Outcome::Label("A".to_string()),
        Outcome::Label("B".to_string()),
        Outcome::Abstain,
        Outcome::Error,
        // gold = B (index 6-11)
        Outcome::Label("A".to_string()),
        Outcome::Label("B".to_string()),
        Outcome::Label("B".to_string()),
        Outcome::Label("B".to_string()),
        Outcome::Label("B".to_string()),
        Outcome::Label("E".to_string()),
        // gold = C (index 12-17)
        Outcome::Label("C".to_string()),
        Outcome::Label("C".to_string()),
        Outcome::Label("C".to_string()),
        Outcome::Label("C".to_string()),
        Outcome::Label("C".to_string()),
        Outcome::Label("".to_string()),
        // gold = D (index 18-23)
        Outcome::Label("A".to_string()),
        Outcome::Label("A".to_string()),
        Outcome::Label("B".to_string()),
        Outcome::Label("B".to_string()),
        Outcome::Label("C".to_string()),
        Outcome::Label("C".to_string()),
    ]
}

fn known_answer_golds() -> Vec<&'static str> {
    let mut golds = Vec::with_capacity(24);
    for label in ["A", "B", "C", "D"] {
        for _ in 0..6 {
            golds.push(label);
        }
    }
    golds
}

/// REQ-24 正常系・TASK-24.1-1: 既知解データセットで正解率・per_label・
/// Macro-F1・混同行列が手計算値と一致する。
#[test]
fn known_answer_single_select_matches_hand_calculated_values() {
    let outcomes = known_answer_outcomes();
    let golds = known_answer_golds();
    let records: Vec<EvalRecord> = golds
        .iter()
        .zip(outcomes.iter())
        .map(|(gold, outcome)| EvalRecord { gold, outcome })
        .collect();

    let labels = ["A", "B", "C", "D"];
    let metrics = evaluate_single_select(&labels, &records).expect("valid known-answer input");

    assert_eq!(metrics.n_total, 24);
    assert_eq!(metrics.outcome_counts.ok, 20);
    assert_eq!(metrics.outcome_counts.invalid, 2);
    assert_eq!(metrics.outcome_counts.abstain, 1);
    assert_eq!(metrics.outcome_counts.error, 1);

    // 混同行列（行: A,B,C,D／列: A,B,C,D,invalid,abstain,error）。
    let expected_rows: [[u64; 7]; 4] = [
        [3, 1, 0, 0, 0, 1, 1],
        [1, 4, 0, 0, 1, 0, 0],
        [0, 0, 5, 0, 1, 0, 0],
        [2, 2, 2, 0, 0, 0, 0],
    ];
    let columns = [
        ConfusionColumn::Label(0),
        ConfusionColumn::Label(1),
        ConfusionColumn::Label(2),
        ConfusionColumn::Label(3),
        ConfusionColumn::Invalid,
        ConfusionColumn::Abstain,
        ConfusionColumn::Error,
    ];
    for (row_index, expected_row) in expected_rows.iter().enumerate() {
        for (col_index, &column) in columns.iter().enumerate() {
            let actual = metrics
                .confusion
                .get(row_index, column)
                .unwrap_or_else(|| panic!("row {row_index} col {col_index} out of range"));
            assert_eq!(
                actual, expected_row[col_index],
                "confusion[{row_index}][{col_index}] mismatch"
            );
        }
    }

    // support は各ラベル 6 件、predicted_count は A=6, B=7, C=7, D=0。
    assert_eq!(metrics.per_label[0].label, "A");
    assert_eq!(metrics.per_label[0].support, 6);
    assert_eq!(metrics.per_label[0].predicted_count, 6);
    assert_eq!(metrics.per_label[1].support, 6);
    assert_eq!(metrics.per_label[1].predicted_count, 7);
    assert_eq!(metrics.per_label[2].support, 6);
    assert_eq!(metrics.per_label[2].predicted_count, 7);
    assert_eq!(metrics.per_label[3].support, 6);
    assert_eq!(metrics.per_label[3].predicted_count, 0);

    // per_label: A: P=R=F1=3/6, B: P=4/7,R=4/6,F1=8/13, C: P=5/7,R=5/6,F1=10/13,
    // D: precision=None, recall=0.0, F1=Some(0.0)。
    let a = &metrics.per_label[0];
    assert!(approx_eq(a.precision.expect("defined"), 3.0 / 6.0));
    assert!(approx_eq(a.recall.expect("defined"), 3.0 / 6.0));
    assert!(approx_eq(a.f1.expect("defined"), 3.0 / 6.0));

    let b = &metrics.per_label[1];
    assert!(approx_eq(b.precision.expect("defined"), 4.0 / 7.0));
    assert!(approx_eq(b.recall.expect("defined"), 4.0 / 6.0));
    assert!(approx_eq(b.f1.expect("defined"), 8.0 / 13.0));

    let c = &metrics.per_label[2];
    assert!(approx_eq(c.precision.expect("defined"), 5.0 / 7.0));
    assert!(approx_eq(c.recall.expect("defined"), 5.0 / 6.0));
    assert!(approx_eq(c.f1.expect("defined"), 10.0 / 13.0));

    let d = &metrics.per_label[3];
    assert_eq!(
        d.precision, None,
        "D の precision は predicted_count=0 のため None"
    );
    assert!(approx_eq(d.recall.expect("defined"), 0.0));
    assert!(
        approx_eq(d.f1.expect("D の F1 は Some(0.0)。None にはならない"), 0.0),
        "F1 を調和平均で定義すると D が None になり、Macro-F1 の平均から誤って除外される"
    );

    // Macro-F1 = 49/104（修正前の誤り 49/78 にはならないこと）。
    let macro_f1 = metrics
        .macro_f1
        .value()
        .expect("all four labels have Some(f1)");
    assert!(approx_eq(macro_f1, 49.0 / 104.0));
    assert!(
        !approx_eq(macro_f1, 49.0 / 78.0),
        "F1 を調和平均で定義したときの誤った値（49/78）を再現していないこと"
    );
    // D は precision が None だが f1 が Some(0.0) のため除外されない
    // （REQ-24 異常系・TASK-24.2: 除外条件は f1 == None のみ）。
    assert_eq!(
        metrics.macro_f1.excluded_labels(),
        &[] as &[String],
        "全ラベルの f1 が定義できるため excluded_labels は空"
    );

    // accuracy.overall = 12/24, adopted_decision = 12/23（分母は abstain を除いた 23）。
    assert_eq!(metrics.accuracy.overall.numerator(), 12);
    assert_eq!(metrics.accuracy.overall.denominator(), 24);
    assert!(approx_eq(metrics.accuracy.overall.value(), 12.0 / 24.0));
    let adopted = metrics.accuracy.adopted_decision.expect("not all abstain");
    assert_eq!(adopted.numerator(), 12);
    assert_eq!(adopted.denominator(), 23);
    assert!(approx_eq(adopted.value(), 12.0 / 23.0));
}

/// REQ-24 正常系・TASK-24.1-1: majority 下限基準（全件 "A" 予測）で
/// 正解率・Macro-F1 が手計算値と一致する（修正前の誤り 0.4 を再現しない）。
#[test]
fn majority_baseline_matches_hand_calculated_values() {
    let golds = known_answer_golds();
    let outcome_a = Outcome::Label("A".to_string());
    let records: Vec<EvalRecord> = golds
        .iter()
        .map(|gold| EvalRecord {
            gold,
            outcome: &outcome_a,
        })
        .collect();

    let labels = ["A", "B", "C", "D"];
    let metrics = evaluate_single_select(&labels, &records).expect("valid majority baseline");

    assert_eq!(metrics.n_total, 24);
    assert_eq!(metrics.outcome_counts.ok, 24);
    assert_eq!(metrics.outcome_counts.invalid, 0);
    assert_eq!(metrics.outcome_counts.abstain, 0);
    assert_eq!(metrics.outcome_counts.error, 0);

    // accuracy.overall = 6/24 = 0.25。
    assert_eq!(metrics.accuracy.overall.numerator(), 6);
    assert_eq!(metrics.accuracy.overall.denominator(), 24);
    assert!(approx_eq(metrics.accuracy.overall.value(), 6.0 / 24.0));
    // abstain が 0 件なので adopted_decision は overall と同じ値。
    let adopted = metrics
        .accuracy
        .adopted_decision
        .expect("no abstain in majority baseline");
    assert_eq!(adopted.numerator(), 6);
    assert_eq!(adopted.denominator(), 24);

    // A: P=6/24, R=1.0, F1=2/5。B/C/D: precision=None, recall=0.0, F1=Some(0.0)。
    let a = &metrics.per_label[0];
    assert!(approx_eq(a.precision.expect("defined"), 6.0 / 24.0));
    assert!(approx_eq(a.recall.expect("defined"), 1.0));
    assert!(approx_eq(a.f1.expect("defined"), 2.0 / 5.0));

    for label_metrics in &metrics.per_label[1..] {
        assert_eq!(
            label_metrics.precision, None,
            "{} の precision は predicted_count=0 のため None",
            label_metrics.label
        );
        assert!(approx_eq(label_metrics.recall.expect("defined"), 0.0));
        assert!(approx_eq(
            label_metrics.f1.expect("Some(0.0) であって None ではない"),
            0.0
        ));
    }

    // Macro-F1 = 0.1（修正前の誤り 0.4 を再現しないこと）。
    let macro_f1 = metrics
        .macro_f1
        .value()
        .expect("all four labels have Some(f1)");
    assert!(approx_eq(macro_f1, 0.1));
    assert!(
        !approx_eq(macro_f1, 0.4),
        "F1 を調和平均で定義したときの誤った値（0.4）を再現していないこと"
    );
    // B・C・D は precision が None だが f1 が Some(0.0) のため除外されない。
    assert_eq!(
        metrics.macro_f1.excluded_labels(),
        &[] as &[String],
        "全ラベルの f1 が定義できるため excluded_labels は空"
    );
}
