//! REQ-41・REQ-39・#665: ベースモデルの重み 1 ファイルの上限が共有 fixture
//! `fixtures/lora_limits/base_model_file.json` と一致すること。学習ワーカー側の
//! `limits.py::MAX_BASE_MODEL_FILE_BYTES` は同じ fixture を読む Python のテストで照合する
//! （`lora_seq_length_fixture.rs` と同じパターン）。`docs/spec` は参照しない。

use std::path::PathBuf;

use fandhe_edge_guard::base_model::MAX_BASE_MODEL_FILE_BYTES;

/// 読み込み前のサイズ上限（REQ-39）。
const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

#[test]
fn req41_base_model_file_limit_matches_shared_fixture() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/lora_limits/base_model_file.json");
    let len = std::fs::metadata(&path).expect("stat fixture").len();
    assert!(len <= MAX_FIXTURE_BYTES, "fixture too large: {len}");
    // cli は serde_json に依存しない（依存追加なし）ため、キーの直後の整数だけを std で読む。
    let text = String::from_utf8(std::fs::read(&path).expect("read fixture")).expect("utf8");
    let rest = text
        .split("\"max_base_model_file_bytes\":")
        .nth(1)
        .expect("key");
    let digits: String = rest
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    assert_eq!(
        digits.parse::<u64>().expect("integer"),
        MAX_BASE_MODEL_FILE_BYTES
    );
}
