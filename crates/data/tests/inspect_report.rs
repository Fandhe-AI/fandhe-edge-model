//! `fandhe-edge-data::inspect` の検査レポート（件数・ラベル別集計）の結合テスト
//! （REQ-16・TASK-16.1-2・issue #39）。
//!
//! [`crate::report::InspectReport`] を [`InspectOutcome::report`] 経由で確認する。
//! 証拠の種別: テストハーネス（インラインの具体値フィクスチャ）。実機・実データでの
//! 測定は行わない（.claude/rules/evaluation-contract.md）。

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use fandhe_edge_data::inspect::inspect_records;

fn valid_label_ids(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|s| s.to_string()).collect()
}

/// REQ-16 受け入れ条件: clean データを検査すると誤検出 0 件で、行数・ユニーク
/// 入力/出力数・ラベル別件数と最少件数を含む具体的なレポートを返すこと。
#[test]
fn req16_clean_data_report_has_concrete_counts() {
    let content = "\
{\"id\":\"r1\",\"input\":\"in1\",\"output\":{\"intent\":\"tier-s__low\"}}
{\"id\":\"r2\",\"input\":\"in2\",\"output\":{\"intent\":\"tier-s__low\"}}
{\"id\":\"r3\",\"input\":\"in3\",\"output\":{\"intent\":\"tier-s__low\"}}
{\"id\":\"r4\",\"input\":\"in4\",\"output\":{\"intent\":\"tier-s__high\"}}
{\"id\":\"r5\",\"input\":\"in5\",\"output\":{\"intent\":\"tier-s__high\"}}";
    let labels = valid_label_ids(&["tier-s__low", "tier-s__high"]);

    let outcome = inspect_records(content, &labels).expect("有効なラベル集合");

    assert!(outcome.anomalies.is_empty(), "REQ-16 正常系: 誤検出は 0 件");
    let report = outcome.report;
    assert_eq!(report.total_rows, 5);
    assert_eq!(report.valid_rows, 5);
    assert_eq!(report.anomalous_rows, 0);
    assert_eq!(report.unique_inputs, 5);
    assert_eq!(report.unique_outputs, 2);
    let mut expected_counts = BTreeMap::new();
    expected_counts.insert("tier-s__high".to_string(), 2);
    expected_counts.insert("tier-s__low".to_string(), 3);
    assert_eq!(report.label_counts, expected_counts);
    assert_eq!(report.min_label_count, Some(2));
    assert!(report.labels_without_records.is_empty());
}

/// 空行は `total_rows` に数えず、異常行は「異常を出した行の数」として
/// `anomalous_rows` に数えること（異常の件数 `anomalies.len()` とは別）。
#[test]
fn req16_blank_and_anomalous_lines_are_counted_in_total_rows_only() {
    let content = "\n\
{\"input\":\"x\"}\n\
   \n\
{\"id\":\"r1\",\"input\":\"in1\",\"output\":{\"intent\":\"ok\"}}\n\
{\"id\":\"r2\",\"input\":\"in2\",\"output\":{\"intent\":\"ok\"}}\n\
{\"id\":\"r3\",\"input\":\"in3\",\"output\":{\"intent\":\"ok\"}}";
    let labels = valid_label_ids(&["ok"]);

    let outcome = inspect_records(content, &labels).expect("有効なラベル集合");

    // 1 行目・3 行目は空行（数えない）。2 行目は id・output 欠落で 2 件の異常。
    assert_eq!(outcome.anomalies.len(), 2);
    assert_eq!(outcome.valid_records.len(), 3);
    let report = outcome.report;
    assert_eq!(report.total_rows, 4, "空行 2 行を除く行数");
    assert_eq!(report.anomalous_rows, 1, "異常を出した行は 1 行のみ");
    assert_eq!(report.valid_rows, 3);
    assert_eq!(
        report.label_counts.get("ok").copied(),
        Some(3),
        "異常行は集計対象から除外される"
    );
}

/// `MalformedJson`・`DuplicateKey` はいずれも検査ループの早期 `continue` 経路
/// （`anomalous_rows` 導出元である `total_rows - valid_records.len()` が
/// 数える対象）だが、通常の `MissingField` 等とは異なる分岐であるため
/// それぞれ単独で `anomalous_rows` に反映されることを確認する。
#[test]
fn req16_malformed_json_and_duplicate_key_rows_count_as_anomalous_rows() {
    let malformed_json = "{not json}\n\
{\"id\":\"r1\",\"input\":\"in1\",\"output\":{\"intent\":\"ok\"}}";
    let labels = valid_label_ids(&["ok"]);

    let outcome = inspect_records(malformed_json, &labels).expect("有効なラベル集合");
    assert_eq!(outcome.report.total_rows, 2);
    assert_eq!(outcome.report.valid_rows, 1);
    assert_eq!(
        outcome.report.anomalous_rows, 1,
        "MalformedJson の 1 行のみ異常"
    );

    let duplicate_key = "{\"id\":\"r1\",\"id\":\"r1\",\"input\":\"in1\",\"output\":{\"intent\":\"ok\"}}\n\
{\"id\":\"r2\",\"input\":\"in2\",\"output\":{\"intent\":\"ok\"}}";

    let outcome = inspect_records(duplicate_key, &labels).expect("有効なラベル集合");
    assert_eq!(outcome.report.total_rows, 2);
    assert_eq!(outcome.report.valid_rows, 1);
    assert_eq!(
        outcome.report.anomalous_rows, 1,
        "DuplicateKey の 1 行のみ異常"
    );
}

/// 異なる `id` で同一 `input` を持つ 2 行は異常として扱わず（重複入力検出は
/// TASK-16.2 の範囲）、`unique_inputs` のみが減ること。
#[test]
fn req16_exact_duplicate_input_reduces_unique_inputs_without_anomaly() {
    let content = "\
{\"id\":\"r1\",\"input\":\"same\",\"output\":{\"intent\":\"ok\"}}
{\"id\":\"r2\",\"input\":\"same\",\"output\":{\"intent\":\"ok\"}}
{\"id\":\"r3\",\"input\":\"other\",\"output\":{\"intent\":\"ok\"}}";
    let labels = valid_label_ids(&["ok"]);

    let outcome = inspect_records(content, &labels).expect("有効なラベル集合");

    assert!(outcome.anomalies.is_empty());
    let report = outcome.report;
    assert_eq!(report.valid_rows, 3);
    assert_eq!(report.unique_inputs, report.valid_rows - 1);
}

/// 定義済みラベルのうち観測が 0 件のものは `label_counts` に現れず
/// `labels_without_records` に列挙されること（ラベル欠落の可視化）。
/// `min_label_count` は観測されたラベルのみが対象（PoC-9 `min_intent_count`
/// と同じ意味）で、0 件ラベルの混入で `0` にはならないこと。
#[test]
fn req16_defined_label_without_records_has_zero_count() {
    let content = "{\"id\":\"r1\",\"input\":\"in1\",\"output\":{\"intent\":\"a\"}}";
    let labels = valid_label_ids(&["a", "b", "c"]);

    let outcome = inspect_records(content, &labels).expect("有効なラベル集合");

    let report = outcome.report;
    assert_eq!(report.label_counts.get("a").copied(), Some(1));
    assert_eq!(report.label_counts.get("b"), None);
    assert_eq!(report.label_counts.get("c"), None);
    assert_eq!(report.min_label_count, Some(1));
    assert_eq!(
        report.labels_without_records,
        vec!["b".to_string(), "c".to_string()]
    );
}

/// 空白の有無で異なる文字列は正規化せず別入力として数えること
/// （NFKC 正規化等は未実装。将来の正規化差し替え時にこのテストで気づける）。
#[test]
fn req16_whitespace_variants_are_distinct_inputs() {
    let content = "\
{\"id\":\"r1\",\"input\":\"a b\",\"output\":{\"intent\":\"ok\"}}
{\"id\":\"r2\",\"input\":\"a  b\",\"output\":{\"intent\":\"ok\"}}";
    let labels = valid_label_ids(&["ok"]);

    let outcome = inspect_records(content, &labels).expect("有効なラベル集合");

    assert!(outcome.anomalies.is_empty());
    assert_eq!(outcome.report.unique_inputs, 2);
}
