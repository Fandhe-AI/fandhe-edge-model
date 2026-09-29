//! ガード層の既定 `kind` 集合と学習ワーカーの選択口の同期照合（REQ-39・TASK-39.2-3・#155）。
//!
//! 連鎖は「Python `_registry` ⇔ 共有 fixture `kind_defaults.json`（Python 側テストで照合済み）
//! ⇔ `KindAllowlist::supported()`」。

use fandhe_edge_guard::kind::KindAllowlist;
use std::collections::BTreeSet;

/// REQ-39: fixture のキー（予約キー `_meta` を除く）と guard の既定集合が完全一致する。
#[test]
fn req39_guard_supported_kinds_match_kind_defaults_fixture() {
    let text = include_str!("../../../fixtures/train_contract/kind_defaults.json");
    let value: serde_json::Value = serde_json::from_str(text).unwrap();
    let obj = value.as_object().unwrap();
    let fixture: BTreeSet<&str> = obj
        .keys()
        .map(String::as_str)
        .filter(|k| *k != "_meta")
        .collect();
    let guard_set = KindAllowlist::supported();
    let guard: BTreeSet<&str> = guard_set.iter().collect();
    assert_eq!(fixture, guard);
}
