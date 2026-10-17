//! runtime の外部データ上限とガード層の上限の一致照合（REQ-39・TASK-41.7-8・#581）。
//!
//! runtime は guard に依存しない（層の境界）ため、両方に依存する cli のテストで照合する。
//! どちらかだけを変えると失敗する。証拠種別: テストハーネス。

use fandhe_edge_guard::file_size::MAX_READ_FILE_BYTES;
use fandhe_edge_guard::split_files::{MAX_PACKAGE_TOTAL_BYTES, MAX_SPLIT_FILES};
use fandhe_edge_runtime::onnx::FINETUNE_MODEL_LIMITS;

#[test]
fn req39_runtime_external_limits_match_guard() {
    let ext = FINETUNE_MODEL_LIMITS.external.expect("external is allowed");
    assert_eq!(ext.max_shard_bytes, MAX_READ_FILE_BYTES);
    assert_eq!(ext.max_total_bytes, MAX_PACKAGE_TOTAL_BYTES);
    assert_eq!(ext.max_shards, MAX_SPLIT_FILES);
}
