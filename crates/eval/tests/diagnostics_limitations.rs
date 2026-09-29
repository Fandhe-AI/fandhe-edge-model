//! ラベル数変動時の診断限界の注記の結合テスト（REQ-29 異常系・TASK-29.2・issue #109。
//! 証拠種別: テストハーネス。手組みの合成データの具体値で検証する。PoC-11 の
//! 「限界と所見」5 項に対応する定義の確認であり、期待値の出典ではない）。

use fandhe_edge_eval::diagnostics::{
    DiagnosticLimitation, DiagnosticReport, InputKey, StatsRow, basic_stats, diagnostic_report,
    note_label_count_change,
};
use fandhe_edge_eval::metrics::{EvalRecord, Outcome, evaluate_single_select};

const NOTE: &str = "label count differs between previous and current label sets; accuracy is not directly comparable because the output definition changed";

/// (input, gold, predicted) の並びから実レポートを作る（train と eval は同じ行を使う）。
fn report_of(labels: &[&str], rows: &[(&str, &str, &str)]) -> DiagnosticReport {
    let stats_rows: Vec<StatsRow<'_>> = rows
        .iter()
        .map(|(i, g, _)| StatsRow { input: i, label: g })
        .collect();
    let train = basic_stats(labels, &stats_rows, InputKey::ByteExact).unwrap();
    let eval = basic_stats(labels, &stats_rows, InputKey::ByteExact).unwrap();
    let outcomes: Vec<Outcome> = rows
        .iter()
        .map(|(_, _, p)| Outcome::Label((*p).to_string()))
        .collect();
    let records: Vec<EvalRecord<'_>> = rows
        .iter()
        .zip(outcomes.iter())
        .map(|((_, g, _), o)| EvalRecord {
            gold: g,
            outcome: o,
        })
        .collect();
    let m = evaluate_single_select(labels, &records).unwrap();
    diagnostic_report(train, eval, &m, 5).unwrap()
}

fn labels_of(report: &DiagnosticReport) -> Vec<String> {
    report
        .eval()
        .label_counts
        .iter()
        .map(|l| l.label.clone())
        .collect()
}

fn old_report() -> DiagnosticReport {
    report_of(
        &["A", "B", "C"],
        &[
            ("a1", "A", "A"),
            ("b1", "B", "A"),
            ("c1", "C", "C"),
            ("c2", "C", "B"),
        ],
    )
}

fn new_report() -> DiagnosticReport {
    // A と B を AB へ統合した 2 ラベルの定義を模す。
    report_of(
        &["AB", "C"],
        &[
            ("a1", "AB", "AB"),
            ("b1", "AB", "AB"),
            ("c1", "C", "C"),
            ("c2", "C", "AB"),
        ],
    )
}

/// 受入: ラベル数が異なる 2 つのデータセットの診断レポートに限界の注記が含まれる。
#[test]
fn label_count_decrease_is_noted_with_exact_values() {
    let old = old_report();
    let old_labels = labels_of(&old);
    let old_refs: Vec<&str> = old_labels.iter().map(String::as_str).collect();
    let noted = note_label_count_change(new_report(), &old_refs).unwrap();
    assert_eq!(
        noted.limitations(),
        &[DiagnosticLimitation::LabelCountChanged {
            previous: 3,
            current: 2
        }]
    );
    assert_eq!(noted.limitations()[0].as_str(), "label_count_changed");
    assert_eq!(noted.limitations()[0].note(), NOTE);
}

/// 逆向き（旧 2 → 新 3）でも注記される。
#[test]
fn label_count_increase_is_noted_with_exact_values() {
    let noted = note_label_count_change(old_report(), &["AB", "C"]).unwrap();
    assert_eq!(
        noted.limitations(),
        &[DiagnosticLimitation::LabelCountChanged {
            previous: 2,
            current: 3
        }]
    );
}

/// 宣言集合が同じで、評価データに未出現のラベルがあるだけなら注記しない
/// （観測ラベル数ではなく宣言ラベル数で判定する）。
#[test]
fn unobserved_label_does_not_trigger_note() {
    let new = report_of(&["A", "B", "C"], &[("a1", "A", "A"), ("b1", "B", "B")]);
    assert_eq!(new.eval().unique_labels, 2);
    let noted = note_label_count_change(new, &["A", "B", "C"]).unwrap();
    assert!(noted.limitations().is_empty());
}

/// ラベル数が同じでラベル ID が異なる場合は本タスクでは注記しない。ラベル集合の相違の
/// 扱いは後続の検討事項（REQ-26 の `regression` モジュールの `ComparisonPremise` を参照）。
#[test]
fn same_count_different_ids_is_not_noted_in_this_task() {
    let noted = note_label_count_change(old_report(), &["A", "B", "D"]).unwrap();
    assert!(noted.limitations().is_empty());
}

/// 注記の付与は既存フィールドを変えない。
#[test]
fn note_does_not_alter_existing_fields() {
    let before = new_report();
    let noted = note_label_count_change(before.clone(), &["A", "B", "C"]).unwrap();
    assert_eq!(noted.train(), before.train());
    assert_eq!(noted.eval(), before.eval());
    assert_eq!(noted.confusable_pairs(), before.confusable_pairs());
}
