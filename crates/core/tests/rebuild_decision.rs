//! `rebuild::decide_rebuild`（公開 API）の結合テスト（REQ-20・TASK-20.1-1・
//! issue #90）。AC1: 同一の定義ファイルを 2 つ渡すと「変更なし」を表す値が
//! 返ることを、公開 API 経由の具体値で確認する。ファイル I/O は使わない。

use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::rebuild::{RebuildDecision, decide_rebuild};

const DEFINITION_A_JSON: &str = r#"{
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

/// `canonical.rs`（TASK-15.5）で独立に確認済みのゴールデン値
/// （証拠の種別: テストハーネス）。
const DEFINITION_A_HASH_HEX: &str =
    "db372ae2b27530fafc7adbc02daa542a8bc1901d7ce916c2a05552d052aef7ac";

/// AC1: 同一の定義ファイルを 2 つ渡すと `RebuildDecision::Unchanged` が返り、
/// ハッシュはゴールデン値と一致する。
#[test]
fn req20_task20_1_1_public_api_identical_definitions_are_unchanged() {
    let def_a = Definition::parse(DEFINITION_A_JSON).expect("固定 fixture は valid なはず");
    let def_b = Definition::parse(DEFINITION_A_JSON).expect("固定 fixture は valid なはず");

    let decision = decide_rebuild(&def_a, &def_b).expect("失敗しないはず");

    match decision {
        RebuildDecision::Unchanged { hash } => {
            assert_eq!(hash.to_hex(), DEFINITION_A_HASH_HEX);
        }
        other => panic!("Unchanged を期待したが {other:?} だった"),
    }
}
