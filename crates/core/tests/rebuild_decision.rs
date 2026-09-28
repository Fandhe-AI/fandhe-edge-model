//! `rebuild::decide_rebuild`（公開 API）の結合テスト
//! （REQ-20・TASK-20.1-1・issue #90／TASK-20.1-2・issue #91／
//! TASK-20.2・issue #92／TASK-20.3・issue #93）。
//!
//! - AC1（#90）: 同一の定義ファイルを 2 つ渡すと「変更なし」を表す値が
//!   返ることを、公開 API 経由の具体値で確認する（ファイル I/O は使わない）
//! - PoC-19 の 5 パターンのうち P1〜P4（#91・P4 の差分詳細は #92）:
//!   `fixtures/rebuild/poc19/`（出典・変換規則は同ディレクトリの
//!   `PROVENANCE.md` を参照）を `Definition::load` 経由で読み込み、
//!   公開 API のみで判定結果を確認する
//! - P5（判定型変更）: 本番の `Definition::load` は `judgment_type:
//!   "multi_select"` を `DefinitionError::UnsupportedValue` で拒否する。
//!   これは本番の観測挙動（学習へ進まない）で、PoC-19 P5 の「必要
//!   （案内のみ）」と整合する。`RebuildDecision::Required(JudgmentTypeChanged)`
//!   の具体値は `#[cfg(test)]` 限定の seam を使う都合上、結合テストからは
//!   見えない `crates/core/src/rebuild.rs` の unit test で確認する
//!   （`rebuild.rs` モジュール doc 2.2 節を参照）。
//! - ハッシュ完全一致時に比較処理を実行しないことの保証（#93）:
//!   `decide_rebuild_with` の seam を使った呼び出し回数の機械照合は private
//!   関数を扱うため `rebuild.rs` の unit test（`req20_task20_3_*`）で行う。
//!   本ファイルでは公開 API（`Definition::load`）経由でファイルから読み込んだ
//!   同一の定義ファイルが `Unchanged` になり、そのハッシュが両者の
//!   `canonical_hash()` と一致することを確認する。

use fandhe_edge_core::definition::{Definition, DefinitionError, FieldPath};
use fandhe_edge_core::rebuild::{RebuildDecision, RebuildReason, decide_rebuild};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

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

/// `fixtures/rebuild/poc19/<name>` を `Definition::load` で読み込む
/// （文字列連結ではなく `Path::join` で組み立てる。REQ-39 のサイズ検査
/// 〔`crate::fs::read_bounded`〕経由）。
fn load_poc19_fixture(name: &str) -> Definition {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("rebuild")
        .join("poc19")
        .join(name);
    Definition::load(&path).expect("固定 fixture は valid なはず")
}

/// P1（追加）: `v2_8rm.json` → `v1_9.json`。`tier-xl__high` が追加され
/// `Required(OptionIdsChanged)` の 1 件になる（出典・パターン対応は
/// `fixtures/rebuild/poc19/PROVENANCE.md`。REQ-20・TASK-20.1-2・issue #91）。
#[test]
fn req20_task20_1_2_public_api_poc19_p1_added_option_requires_rebuild() {
    let old = load_poc19_fixture("v2_8rm.json");
    let new = load_poc19_fixture("v1_9.json");

    let decision = decide_rebuild(&old, &new).expect("失敗しないはず");

    let expected_added: BTreeSet<String> = ["tier-xl__high".to_string()].into_iter().collect();
    match decision {
        RebuildDecision::Required(required) => {
            assert_eq!(
                required.reasons(),
                &[RebuildReason::OptionIdsChanged {
                    added: expected_added,
                    removed: BTreeSet::new(),
                }]
            );
        }
        other => panic!("Required を期待したが {other:?} だった"),
    }
}

/// P2（削除）: `v1_9.json` → `v2_8rm.json`。`tier-xl__high` が削除され
/// `Required(OptionIdsChanged)` の 1 件になる（REQ-20・TASK-20.1-2・issue #91）。
#[test]
fn req20_task20_1_2_public_api_poc19_p2_removed_option_requires_rebuild() {
    let old = load_poc19_fixture("v1_9.json");
    let new = load_poc19_fixture("v2_8rm.json");

    let decision = decide_rebuild(&old, &new).expect("失敗しないはず");

    let expected_removed: BTreeSet<String> = ["tier-xl__high".to_string()].into_iter().collect();
    match decision {
        RebuildDecision::Required(required) => {
            assert_eq!(
                required.reasons(),
                &[RebuildReason::OptionIdsChanged {
                    added: BTreeSet::new(),
                    removed: expected_removed,
                }]
            );
        }
        other => panic!("Required を期待したが {other:?} だった"),
    }
}

/// P3（統合）: `v1_9.json` → `v3_8merge.json`。`tier-l__medium` +
/// `tier-l__high` が `tier-l__midhigh` へ統合され、`added`・`removed` の
/// 双方が非空の `OptionIdsChanged` 1 件になる（統合を独立の理由にしない
/// 判断は `rebuild.rs` モジュール doc 2.1 節を参照。REQ-20・TASK-20.1-2・
/// issue #91）。
#[test]
fn req20_task20_1_2_public_api_poc19_p3_merged_options_require_rebuild() {
    let old = load_poc19_fixture("v1_9.json");
    let new = load_poc19_fixture("v3_8merge.json");

    let decision = decide_rebuild(&old, &new).expect("失敗しないはず");

    let expected_added: BTreeSet<String> = ["tier-l__midhigh".to_string()].into_iter().collect();
    let expected_removed: BTreeSet<String> =
        ["tier-l__medium".to_string(), "tier-l__high".to_string()]
            .into_iter()
            .collect();
    match decision {
        RebuildDecision::Required(required) => {
            assert_eq!(
                required.reasons(),
                &[RebuildReason::OptionIdsChanged {
                    added: expected_added,
                    removed: expected_removed,
                }]
            );
        }
        other => panic!("Required を期待したが {other:?} だった"),
    }
}

/// P4（表示名・説明のみ）: `v1_9.json` → `v4_rename.json`。選択肢 ID 集合・
/// 判定型は同一で `NotRequired` になり、`tier-xs__low` の表示名・
/// `tier-xl__high` の説明が変わったことが公開 API・アクセサのみで具体値
/// 照合できる（AC1 の主証拠。出典: `fixtures/rebuild/poc19/PROVENANCE.md`。
/// REQ-20 異常系・TASK-20.1-2・issue #91・TASK-20.2・issue #92）。
#[test]
fn req20_task20_2_public_api_poc19_p4_display_only_change_is_not_required() {
    let old = load_poc19_fixture("v1_9.json");
    let new = load_poc19_fixture("v4_rename.json");

    let decision = decide_rebuild(&old, &new).expect("失敗しないはず");

    let expected_display_name_changed: BTreeSet<String> =
        ["tier-xs__low".to_string()].into_iter().collect();
    let expected_description_changed: BTreeSet<String> =
        ["tier-xl__high".to_string()].into_iter().collect();
    match decision {
        RebuildDecision::NotRequired(not_required) => {
            assert_eq!(
                not_required.display_name_changed(),
                &expected_display_name_changed
            );
            assert_eq!(
                not_required.description_changed(),
                &expected_description_changed
            );
        }
        other => panic!("NotRequired を期待したが {other:?} だった"),
    }
}

/// P5（判定型変更）: `v5_multi.json`（`judgment_type: "multi_select"`）は
/// 本番の `Definition::load` が `DefinitionError::UnsupportedValue { field:
/// FieldPath::JudgmentType }` で拒否する。これは本番の観測挙動（学習へ
/// 進まない）で、PoC-19 P5 の「必要（案内のみ）」と整合する
/// （`Required(JudgmentTypeChanged)` の具体値は `rebuild.rs` の unit test
/// で確認する。REQ-20・TASK-20.1-2・issue #91）。
#[test]
fn req20_task20_1_2_poc19_p5_multi_select_definition_is_rejected() {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("rebuild")
        .join("poc19")
        .join("v5_multi.json");

    let err = Definition::load(&path).expect_err("multi_select は拒否されるはず");

    match err {
        DefinitionError::UnsupportedValue { field } => {
            assert_eq!(field, FieldPath::JudgmentType);
        }
        other => panic!("UnsupportedValue を期待したが {other:?} だった"),
    }
}

/// ハッシュ完全一致時の「変更なし」扱い（REQ-20 境界値）を、公開 API のみで
/// 確認する。`fixtures/rebuild/poc19/v1_9.json` を `Definition::load` で
/// 2 回読み込み、`Unchanged { hash }` が返り、その `hash` が新旧双方の
/// `canonical_hash()` と一致することを確認する（比較処理を実行しないことの
/// 呼び出し回数の機械照合は private seam を扱うため `rebuild.rs` の
/// unit test（`req20_task20_3_*`）で行う。証拠種別: テストハーネス。
/// REQ-20・TASK-20.3・issue #93）。
#[test]
fn req20_task20_3_public_api_identical_definition_files_are_unchanged() {
    let old = load_poc19_fixture("v1_9.json");
    let new = load_poc19_fixture("v1_9.json");

    let decision = decide_rebuild(&old, &new).expect("失敗しないはず");

    match decision {
        RebuildDecision::Unchanged { hash } => {
            assert_eq!(hash, old.canonical_hash().expect("失敗しないはず"));
            assert_eq!(hash, new.canonical_hash().expect("失敗しないはず"));
        }
        other => panic!("Unchanged を期待したが {other:?} だった"),
    }
}

/// キー順・空白が異なる（`serde_json::Value` 経由で再直列化した）入力でも、
/// 正準化ハッシュが一致すれば `Unchanged` になることを公開 API のみで確認する
/// （REQ-20 境界値・TASK-20.3・issue #93）。
#[test]
fn req20_task20_3_public_api_reserialized_definition_is_unchanged() {
    let old = load_poc19_fixture("v1_9.json");
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("rebuild")
        .join("poc19")
        .join("v1_9.json");
    let raw = std::fs::read_to_string(&path).expect("固定 fixture は読み込めるはず");
    let value: serde_json::Value =
        serde_json::from_str(&raw).expect("固定 fixture は valid JSON のはず");
    let reserialized = serde_json::to_string(&value).expect("Value の再直列化は失敗しないはず");
    let new = Definition::parse(&reserialized).expect("valid なはず");

    let decision = decide_rebuild(&old, &new).expect("失敗しないはず");

    match decision {
        RebuildDecision::Unchanged { hash } => {
            assert_eq!(hash, old.canonical_hash().expect("失敗しないはず"));
            assert_eq!(hash, new.canonical_hash().expect("失敗しないはず"));
        }
        other => panic!("Unchanged を期待したが {other:?} だった"),
    }
}
