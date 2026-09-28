//! REQ-16 受け入れテスト: PoC-9 clean フィクスチャに対する検査レポート
//! （TASK-16.1-2・issue #39）が PoC-9 `evaluator/inspect.py` の
//! `inspect_split` 実測値（`fixtures/data_inspect/PROVENANCE.md` 記載）と
//! 一致すること。
//!
//! 証拠種別: テストハーネス（`docs/spec` 抜きで完結するよう本リポへ移植
//! 済みのフィクスチャを使う。`.claude/rules/spec-reference.md`）。

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use fandhe_edge_data::inspect::inspect_records;

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("data_inspect")
        .join("clean")
        .join(name)
}

fn read_fixture(name: &str) -> String {
    fs::read_to_string(fixture_path(name))
        .unwrap_or_else(|e| panic!("fixture {name} の読み込みに失敗: {e}"))
}

fn nine_intent_labels() -> BTreeSet<String> {
    [
        "create_task",
        "complete_task",
        "delete_task",
        "list_tasks",
        "set_reminder",
        "create_note",
        "search_notes",
        "schedule_event",
        "none",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

/// PoC-9 `train.jsonl` 実測値: rows=10・unique_inputs=10・unique_outputs=10・
/// intent_counts={"create_task":10}・min_intent_count=10。
#[test]
fn req16_clean_train_report_matches_poc9() {
    let content = read_fixture("train.jsonl");
    let labels = nine_intent_labels();

    let outcome = inspect_records(&content, &labels).expect("有効なラベル集合");

    assert!(outcome.anomalies.is_empty(), "clean データは誤検出 0 件");
    let report = outcome.report;
    assert_eq!(report.valid_rows, 10);
    assert_eq!(report.unique_inputs, 10);
    assert_eq!(report.unique_outputs, 10);

    let mut expected_counts = BTreeMap::new();
    expected_counts.insert("create_task".to_string(), 10);
    assert_eq!(report.label_counts, expected_counts);
    assert_eq!(report.min_label_count, Some(10));

    let expected_missing: Vec<String> = [
        "complete_task",
        "create_note",
        "delete_task",
        "list_tasks",
        "none",
        "schedule_event",
        "search_notes",
        "set_reminder",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    assert_eq!(report.labels_without_records, expected_missing);
}

/// PoC-9 `test.jsonl` 実測値: rows=10・unique_inputs=10・**unique_outputs=7**
/// （intent 種類数 4 とは一致しない。`output` 全体を数える根拠。
/// `fixtures/data_inspect/PROVENANCE.md` 参照）・
/// intent_counts={"create_task":2,"complete_task":3,"delete_task":3,"list_tasks":2}・
/// min_intent_count=2。
#[test]
fn req16_clean_test_report_matches_poc9() {
    let content = read_fixture("test.jsonl");
    let labels = nine_intent_labels();

    let outcome = inspect_records(&content, &labels).expect("有効なラベル集合");

    assert!(outcome.anomalies.is_empty(), "clean データは誤検出 0 件");
    let report = outcome.report;
    assert_eq!(report.valid_rows, 10);
    assert_eq!(report.unique_inputs, 10);
    assert_eq!(
        report.unique_outputs, 7,
        "output 全体の異なり数（intent 種類数の 4 ではない）"
    );

    let mut expected_counts = BTreeMap::new();
    expected_counts.insert("complete_task".to_string(), 3);
    expected_counts.insert("create_task".to_string(), 2);
    expected_counts.insert("delete_task".to_string(), 3);
    expected_counts.insert("list_tasks".to_string(), 2);
    assert_eq!(report.label_counts, expected_counts);
    assert_eq!(report.min_label_count, Some(2));

    let expected_missing: Vec<String> = [
        "create_note",
        "none",
        "schedule_event",
        "search_notes",
        "set_reminder",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    assert_eq!(report.labels_without_records, expected_missing);
}
