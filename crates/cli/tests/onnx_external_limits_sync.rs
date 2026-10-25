//! runtime の外部データ上限とガード層の上限の一致照合（REQ-39・TASK-41.7-8・#581・#654）。
//!
//! runtime は guard に依存しない（層の境界）ため、両方に依存する cli のテストで照合する。
//! 共有 fixture `fixtures/train_contract/limits.json` の `max_external_shard_bytes` とも照合する。
//! どちらかだけを変えると失敗する。証拠種別: テストハーネス。

use fandhe_edge_guard::split_files::{
    MAX_PACKAGE_TOTAL_BYTES, MAX_SHARD_FILE_BYTES, MAX_SPLIT_FILES,
};
use fandhe_edge_runtime::onnx::FINETUNE_MODEL_LIMITS;

#[test]
fn req39_runtime_external_limits_match_guard() {
    let ext = FINETUNE_MODEL_LIMITS.external.expect("external is allowed");
    assert_eq!(ext.max_shard_bytes, MAX_SHARD_FILE_BYTES);
    assert_eq!(ext.max_total_bytes, MAX_PACKAGE_TOTAL_BYTES);
    assert_eq!(ext.max_shards, MAX_SPLIT_FILES);
}

/// 共有 fixture の `max_external_shard_bytes` がガード層の値と一致する。
/// serde_json は cli の依存に無いため、キーの直後の数値を素朴に取り出す。
#[test]
fn req39_guard_shard_limit_matches_shared_fixture() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/train_contract/limits.json"
    );
    let text = std::fs::read_to_string(path).expect("read limits.json");
    let key = "\"max_external_shard_bytes\"";
    let rest = &text[text.find(key).expect("key present") + key.len()..];
    let digits: String = rest
        .trim_start_matches([':', ' '])
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    assert_eq!(digits.parse::<u64>().expect("number"), MAX_SHARD_FILE_BYTES);
}
