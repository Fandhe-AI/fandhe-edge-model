//! メタデータ混入検出の異常系テスト（REQ-16・TASK-16.2-2・PoC-9 追補 A-10）。
//!
//! `fixtures/data_contract/injection/{clean,metadata-mixed}/`（手書きの合成
//! データ。PR #188 レビュー指摘 P0・reviewThread PRRT_kwDOUq-SxM6mg8C0 を
//! 受け、PoC-9 からのバイト単位コピーを廃止し全面差し替えた。出典は
//! `fixtures/data_contract/injection/PROVENANCE.md` に記録）を使い、
//! clean データで誤検出 0 件・metadata-mixed データで既知の 3 件を全件検出
//! することを確認する。

mod support;

use std::collections::BTreeSet;
use std::path::Path;

use fandhe_edge_data::consistency::{MetadataMixReason, find_contradictions, find_metadata_mixed};
use fandhe_edge_data::normalize::PyWhitespaceNormalizer;
use support::{ContradictionTestRecord, MetadataTestRecord, load_jsonl_pool};

fn load_pool(dir_name: &str) -> Vec<support::RawRecord> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest_dir
        .join("../../fixtures/data_contract/injection")
        .join(dir_name);
    load_jsonl_pool(&[&dir.join("train.jsonl"), &dir.join("test.jsonl")])
}

/// REQ-16: clean データではメタデータ混入 0 件・矛盾 0 件
/// （`answer.json` の `false_positive_expectation: 0` に対応）。
#[test]
fn req16_metadata_injection_clean_has_no_false_positives() {
    let raw = load_pool("clean");
    assert_eq!(raw.len(), 10);

    let metadata_records: Vec<MetadataTestRecord> = raw
        .iter()
        .cloned()
        .map(MetadataTestRecord::from_raw)
        .collect();
    let metadata_report = find_metadata_mixed(&metadata_records).expect("id は重複・空のはず無し");
    assert_eq!(
        metadata_report.count, 0,
        "clean データで誤検出: {:?}",
        metadata_report.hits
    );

    let contradiction_records: Vec<ContradictionTestRecord> = raw
        .into_iter()
        .map(ContradictionTestRecord::from_raw)
        .collect();
    let contradiction_report = find_contradictions(&contradiction_records, &PyWhitespaceNormalizer)
        .expect("id は重複・空のはず無し");
    assert_eq!(
        contradiction_report.distinct_inputs, 0,
        "clean データで矛盾を誤検出した"
    );
}

/// REQ-16: metadata-mixed データで既知の 3 件（`syn-tr-1`・
/// `syn-tr-2`・`syn-te-1`）を全件検出する（`answer.json` の
/// `count: 3` に対応）。理由は PoC-9 A-10 の `detection_rule` が示す種類ごとに
/// 分類され、実測した集合を具体値で確認する。
#[test]
fn req16_metadata_injection_metadata_mixed_detects_known_three() {
    let raw = load_pool("metadata-mixed");
    assert_eq!(raw.len(), 10);

    let records: Vec<MetadataTestRecord> =
        raw.into_iter().map(MetadataTestRecord::from_raw).collect();
    let report = find_metadata_mixed(&records).expect("id は重複・空のはず無し");

    assert_eq!(
        report.count, 3,
        "検出件数が answer.json の count と異なる: {:?}",
        report.hits
    );

    let expected_ids: BTreeSet<String> = ["syn-tr-1", "syn-tr-2", "syn-te-1"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let actual_ids: BTreeSet<String> = report.hits.keys().cloned().collect();
    assert_eq!(actual_ids, expected_ids);

    // `answer.json` の `kind` に対応する理由（output_intent_in_input ->
    // GoldLabelInInput・id_in_input -> IdInInput・
    // output_arguments_in_input -> GoldSerializationInInput）。
    assert_eq!(
        report.hits.get("syn-tr-1"),
        Some(&BTreeSet::from([MetadataMixReason::GoldLabelInInput])),
    );
    assert_eq!(
        report.hits.get("syn-tr-2"),
        Some(&BTreeSet::from([MetadataMixReason::IdInInput])),
    );
    assert_eq!(
        report.hits.get("syn-te-1"),
        Some(&BTreeSet::from([
            MetadataMixReason::GoldSerializationInInput
        ])),
    );
}
