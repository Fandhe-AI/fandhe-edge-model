//! 基礎統計の結合テスト（REQ-29 正常系・TASK-29.1-1・issue #107。
//! 証拠種別: テストハーネス。手組みの合成データの具体値で検証する）。

use fandhe_edge_eval::diagnostics::{
    DiagnosticsError, InputKey, LabelCount, StatsRow, basic_stats,
};
use fandhe_edge_eval::metrics::EvalError;
use fandhe_edge_eval::significance::MAX_EVAL_RECORDS;

fn row<'a>(input: &'a str, label: &'a str) -> StatsRow<'a> {
    StatsRow { input, label }
}

fn synthetic() -> Vec<StatsRow<'static>> {
    vec![
        row("i1", "A"),
        row("i1", "A"),
        row("i2", "A"),
        row("i3", "A"),
        row("i4", "B"),
        row("i4", "B"),
        row("i5", "B"),
        row("i6", "C"),
        row("i6", "C"),
        row("i7", "C"),
    ]
}

fn lc(label: &str, count: u64) -> LabelCount {
    LabelCount {
        label: label.to_string(),
        count,
    }
}

#[test]
fn known_synthetic_dataset() {
    let s = basic_stats(&["A", "B", "C", "D"], &synthetic(), InputKey::ByteExact).unwrap();
    assert_eq!(s.n_rows, 10);
    assert_eq!(s.unique_inputs, 7);
    assert_eq!(s.unique_labels, 3);
    assert_eq!(
        s.label_counts,
        vec![lc("A", 4), lc("B", 3), lc("C", 3), lc("D", 0)]
    );
    assert_eq!(s.min_label_count, 0);
    assert_eq!(s.min_labels, vec!["D".to_string()]);
    assert_eq!(s.input_key_rule, "byte_exact");
}

#[test]
fn tied_minimum_lists_all_in_declaration_order() {
    let s = basic_stats(&["A", "B", "C"], &synthetic(), InputKey::ByteExact).unwrap();
    assert_eq!(s.min_label_count, 3);
    assert_eq!(s.min_labels, vec!["B".to_string(), "C".to_string()]);
    assert_eq!(s.input_key_rule, "byte_exact");
}

#[test]
fn unique_inputs_is_byte_equality_unless_normalizer_given() {
    let rows = [row("ＡＢ", "A"), row("AB", "A")];
    assert_eq!(
        basic_stats(&["A"], &rows, InputKey::ByteExact)
            .unwrap()
            .unique_inputs,
        2
    );
    let key = InputKey::Normalized {
        rule: "test_fullwidth_ab",
        normalize: |s| s.replace('Ａ', "A").replace('Ｂ', "B"),
    };
    let s = basic_stats(&["A"], &rows, key).unwrap();
    assert_eq!(s.unique_inputs, 1);
    assert_eq!(s.input_key_rule, "test_fullwidth_ab");
    let rows = [row("AB", "A"), row("AB", "A")];
    assert_eq!(
        basic_stats(&["A"], &rows, InputKey::ByteExact)
            .unwrap()
            .unique_inputs,
        1
    );
}

#[test]
fn errors() {
    let rows = synthetic();
    assert_eq!(
        basic_stats(&["A", "B"], &rows, InputKey::ByteExact).unwrap_err(),
        DiagnosticsError::UnknownLabel { index: 7 }
    );
    assert_eq!(
        basic_stats(&["A"], &[], InputKey::ByteExact).unwrap_err(),
        DiagnosticsError::EmptyRows
    );
    assert_eq!(
        basic_stats(&[], &rows, InputKey::ByteExact).unwrap_err(),
        DiagnosticsError::Labels(EvalError::EmptyLabels)
    );
    assert!(matches!(
        basic_stats(&["A", "A"], &rows, InputKey::ByteExact).unwrap_err(),
        DiagnosticsError::Labels(_)
    ));
    let long = "x".repeat(100_000);
    assert!(matches!(
        basic_stats(&[long.as_str()], &rows, InputKey::ByteExact).unwrap_err(),
        DiagnosticsError::Labels(EvalError::LabelTooLong { .. })
    ));
    let many = vec![row("i", "A"); MAX_EVAL_RECORDS + 1];
    assert_eq!(
        basic_stats(&["A"], &many, InputKey::ByteExact).unwrap_err(),
        DiagnosticsError::TooManyRows {
            n_rows: MAX_EVAL_RECORDS + 1,
            limit: MAX_EVAL_RECORDS
        }
    );
}

#[test]
fn deterministic_under_row_reordering() {
    let mut rows = synthetic();
    let a = basic_stats(&["A", "B", "C", "D"], &rows, InputKey::ByteExact).unwrap();
    rows.reverse();
    let b = basic_stats(&["A", "B", "C", "D"], &rows, InputKey::ByteExact).unwrap();
    assert_eq!(a, b);
}
