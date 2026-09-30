//! ガード層の `kind_version` 許可リストと学習ワーカーの選択口の同期照合（REQ-39・TASK-39.6-1・#174）。
//!
//! 連鎖は「Python `_registry` ⇔ 共有 fixture `kind_versions.json`（Python 側テストで照合済み）
//! ⇔ `KindVersionAllowlist::supported()`」。

use fandhe_edge_guard::kind_version::KindVersionAllowlist;
use std::collections::BTreeSet;

/// REQ-39: fixture の `(kind, 版)` の集合（予約キー `_meta` を除く）と guard の既定集合が完全一致する。
#[test]
fn req39_guard_supported_kind_versions_match_fixture() {
    let text = include_str!("../../../fixtures/train_contract/kind_versions.json");
    let value: serde_json::Value = serde_json::from_str(text).unwrap();
    let obj = value.as_object().unwrap();
    let mut fixture: BTreeSet<(String, u32)> = BTreeSet::new();
    for (kind, versions) in obj.iter().filter(|(k, _)| k.as_str() != "_meta") {
        for v in versions.as_array().unwrap() {
            fixture.insert((kind.clone(), u32::try_from(v.as_u64().unwrap()).unwrap()));
        }
    }
    let guard: BTreeSet<(String, u32)> = KindVersionAllowlist::supported()
        .iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    assert_eq!(fixture, guard);
}
