//! `Definition::identity`・`Definition::canonical_hash`（公開 API）の結合テスト
//! （REQ-15・TASK-15.5）。ユニットテスト（`canonical.rs` 内）はゴールデン値の
//! 固定を主眼にするのに対し、ここでは公開 API 経由での境界値（AC1〜AC3）を
//! 確認する。ファイル I/O は使わない。

use fandhe_edge_core::definition::Definition;

const BASE_JSON: &str = r#"{
    "schema": "fandhe-edge-model-definition/v1",
    "name": "sample_topic",
    "version": 1,
    "judgment_type": "single_select",
    "options": [
        { "id": "yes", "display_name": "Yes", "description": "肯定" },
        { "id": "no", "display_name": "No", "description": "否定" }
    ],
    "io": { "input": "bytes" }
}"#;

/// AC1: 表示名・説明だけを変えた定義でも、選択肢 ID の集合が同じなら
/// 同一の定義として受理される（同一性が等しい）。
#[test]
fn req15_task15_5_public_api_display_name_only_change_is_same_identity() {
    let changed_json = r#"{
        "schema": "fandhe-edge-model-definition/v1",
        "name": "sample_topic",
        "version": 1,
        "judgment_type": "single_select",
        "options": [
            { "id": "yes", "display_name": "はい", "description": "肯定に変更" },
            { "id": "no", "display_name": "いいえ", "description": "否定に変更" }
        ],
        "io": { "input": "bytes" }
    }"#;

    let base = Definition::parse(BASE_JSON).expect("valid なはず");
    let changed = Definition::parse(changed_json).expect("valid なはず");

    assert_eq!(base.identity(), changed.identity());
}

/// AC2: 表示名・説明だけを変えた 2 つの定義は、同一性では同一だが、
/// 定義全体の正準化ハッシュでは不一致になる。
#[test]
fn req15_task15_5_public_api_display_name_only_change_differs_in_hash() {
    let changed_json = r#"{
        "schema": "fandhe-edge-model-definition/v1",
        "name": "sample_topic",
        "version": 1,
        "judgment_type": "single_select",
        "options": [
            { "id": "yes", "display_name": "はい", "description": "肯定" },
            { "id": "no", "display_name": "No", "description": "否定" }
        ],
        "io": { "input": "bytes" }
    }"#;

    let base = Definition::parse(BASE_JSON).expect("valid なはず");
    let changed = Definition::parse(changed_json).expect("valid なはず");

    assert_eq!(base.identity(), changed.identity());
    let base_hash = base.canonical_hash().expect("失敗しないはず");
    let changed_hash = changed.canonical_hash().expect("失敗しないはず");
    assert_ne!(base_hash, changed_hash);
    assert_ne!(base_hash.to_hex(), changed_hash.to_hex());
}

/// AC3: キーの順序や空白だけが違う 2 つの定義は、定義全体の正準化ハッシュが
/// 一致する（`Definition::parse` を経由することでキー順・空白の揺れを吸収する）。
#[test]
fn req15_task15_5_public_api_key_order_and_whitespace_do_not_change_hash() {
    let reordered_json = "{\n  \"options\": [\n    { \"display_name\": \"Yes\", \"id\": \"yes\", \"description\": \"肯定\" },\n    { \"description\": \"否定\", \"display_name\": \"No\", \"id\": \"no\" }\n  ],\n  \"version\": 1,\n  \"name\": \"sample_topic\",\n  \"io\": { \"input\": \"bytes\" },\n  \"schema\": \"fandhe-edge-model-definition/v1\",\n  \"judgment_type\": \"single_select\"\n}\n";

    let base = Definition::parse(BASE_JSON).expect("valid なはず");
    let reordered = Definition::parse(reordered_json).expect("valid なはず");

    assert_eq!(base.identity(), reordered.identity());
    assert_eq!(
        base.canonical_hash().expect("失敗しないはず"),
        reordered.canonical_hash().expect("失敗しないはず")
    );
}

/// 選択肢 ID が異なれば同一性も異なる（境界値の反対側の確認）。
#[test]
fn req15_task15_5_public_api_different_option_ids_differ_in_identity() {
    let different_ids_json = r#"{
        "schema": "fandhe-edge-model-definition/v1",
        "name": "sample_topic",
        "version": 1,
        "judgment_type": "single_select",
        "options": [
            { "id": "yes", "display_name": "Yes", "description": "肯定" },
            { "id": "unsure", "display_name": "Unsure", "description": "不明" }
        ],
        "io": { "input": "bytes" }
    }"#;

    let base = Definition::parse(BASE_JSON).expect("valid なはず");
    let different = Definition::parse(different_ids_json).expect("valid なはず");

    assert_ne!(base.identity(), different.identity());
}
