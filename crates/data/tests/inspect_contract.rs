//! `fandhe-edge-data::inspect` の結合テスト（REQ-16・TASK-16.1-1・親 issue #37）。
//!
//! 複数の異常種別と正常レコードが混在する 1 つの JSONL 本文を検査し、
//! 検出件数・内部コード文字列・妥当なレコード件数を具体値で確認する。
//! 証拠の種別: テストハーネス（インラインの具体値フィクスチャ）。実機・実データでの
//! 測定は行わない（.claude/rules/evaluation-contract.md）。

use std::collections::BTreeSet;

use fandhe_edge_data::inspect::{AnomalyCode, inspect_records};

fn valid_label_ids() -> BTreeSet<String> {
    ["tier-s__low", "tier-s__high"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

/// 親 issue #37 の受け入れ条件（clean データで誤検出 0 件）を単独ケースとしても確認する。
#[test]
fn clean_data_produces_zero_anomalies() {
    let content = "\
{\"id\":\"r1\",\"input\":\"hello\",\"output\":{\"intent\":\"tier-s__low\"}}
{\"id\":\"r2\",\"input\":\"world\",\"output\":{\"intent\":\"tier-s__high\"},\"tags\":[\"a\"],\"group_id\":\"g1\"}
{\"id\":\"r3\",\"input\":\"foo\",\"output\":{\"intent\":\"tier-s__low\"}}";

    let outcome = inspect_records(content, &valid_label_ids()).expect("有効なラベル集合");

    assert_eq!(outcome.anomalies.len(), 0, "REQ-16 正常系: 誤検出は 0 件");
    assert_eq!(outcome.valid_records.len(), 3);
}

/// 複数種の異常が混在する JSONL を検査し、件数・コード文字列・妥当レコード件数を確認する。
#[test]
fn mixed_anomalies_are_all_detected() {
    let content = "\
{\"id\":\"ok1\",\"input\":\"hello\",\"output\":{\"intent\":\"tier-s__low\"}}
{not json}
[1,2,3]
{\"input\":\"missing id\",\"output\":{\"intent\":\"tier-s__low\"}}
{\"id\":42,\"input\":\"bad id type\",\"output\":{\"intent\":\"tier-s__low\"}}
{\"id\":\"ok2\",\"output\":{\"intent\":\"tier-s__low\"}}
{\"id\":\"ok3\",\"input\":\"bad output type\",\"output\":\"nope\"}
{\"id\":\"ok4\",\"input\":\"missing intent\",\"output\":{}}
{\"id\":\"ok5\",\"input\":\"unknown label\",\"output\":{\"intent\":\"does-not-exist\"}}
{\"id\":\"ok6\",\"input\":\"bad tags\",\"output\":{\"intent\":\"tier-s__low\"},\"tags\":\"x\"}
{\"id\":\"ok7\",\"input\":\"bad tag element\",\"output\":{\"intent\":\"tier-s__low\"},\"tags\":[\"a\",1]}
{\"id\":\"ok8\",\"input\":\"bad group_id\",\"output\":{\"intent\":\"tier-s__low\"},\"group_id\":1}
{\"id\":\"dup\",\"input\":\"first\",\"output\":{\"intent\":\"tier-s__low\"}}
{\"id\":\"dup\",\"input\":\"second\",\"output\":{\"intent\":\"tier-s__low\"}}
{\"id\":\"ok9\",\"input\":\"clean\",\"output\":{\"intent\":\"tier-s__high\"}}
";

    let outcome = inspect_records(content, &valid_label_ids()).expect("有効なラベル集合");

    let codes: Vec<&'static str> = outcome.anomalies.iter().map(|a| a.code.code()).collect();
    assert_eq!(
        codes,
        vec![
            "malformed_json",   // 2 行目
            "malformed_record", // 3 行目
            "missing_field",    // 4 行目: id
            "type_mismatch",    // 5 行目: id が number
            "missing_field",    // 6 行目: input
            "type_mismatch",    // 7 行目: output が object でない
            "missing_field",    // 8 行目: output.intent
            "unknown_label",    // 9 行目: intent が未知
            "type_mismatch",    // 10 行目: tags が array でない
            "type_mismatch",    // 11 行目: tags[] の要素が string でない
            "type_mismatch",    // 12 行目: group_id が string でない
            "duplicate_id",     // 14 行目: id "dup" の重複
        ]
    );
    assert_eq!(outcome.anomalies.len(), 12);

    // 妥当なレコード: ok1・ok3（output 型不正で intent 検査自体スキップされるが id/input は妥当のため
    // valid_records には残らない点に注意）は record_has_error が立つため除外される。
    // 妥当と判定されるのは ok1・dup（両方）・ok9 の 4 件。
    let valid_ids: Vec<&str> = outcome
        .valid_records
        .iter()
        .map(|r| r.id.as_str())
        .collect();
    assert_eq!(valid_ids, vec!["ok1", "dup", "dup", "ok9"]);
    assert_eq!(outcome.valid_records.len(), 4);
}

/// 有効なラベル ID 集合が空の場合はエラーを返す。
#[test]
fn empty_label_set_returns_error() {
    let content = "{\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"ok\"}}";
    let empty: BTreeSet<String> = BTreeSet::new();

    assert!(inspect_records(content, &empty).is_err());
}

/// `RecordAnomaly` の `TypeMismatch` バリアントを 1 件、具体値で確認する
/// （internal 型を直接構築できることの回帰確認を兼ねる）。
#[test]
fn type_mismatch_variant_carries_expected_and_actual() {
    let content = "{\"id\":\"r1\",\"input\":1,\"output\":{\"intent\":\"tier-s__low\"}}";

    let outcome = inspect_records(content, &valid_label_ids()).unwrap();

    assert_eq!(outcome.anomalies.len(), 1);
    assert_eq!(
        outcome.anomalies[0].code,
        AnomalyCode::TypeMismatch {
            expected: "string",
            actual: "number",
        }
    );
    assert_eq!(outcome.anomalies[0].field, "input");
}
