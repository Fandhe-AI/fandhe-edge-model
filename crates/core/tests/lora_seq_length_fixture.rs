//! REQ-41・REQ-39: `qwen2_lora` の `max_seq_length` 範囲が共有 fixture
//! `fixtures/lora_limits/lora_seq_length.json` と一致すること。学習ワーカー側は
//! `trainer/tests/test_lora_seq_length_fixture.py` が同じ fixture を読んで `limits.py` と照合する
//! （`score_sum_tolerance_fixture.rs` と同じパターン）。`docs/spec` は参照しない。

use std::fs;
use std::path::PathBuf;

use fandhe_edge_core::limits::{MAX_LORA_SEQ_LENGTH, MIN_LORA_SEQ_LENGTH};
use serde::Deserialize;

/// 読み込み前のサイズ上限（REQ-39）。
const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Deserialize)]
struct Fixture {
    min_lora_seq_length: u64,
    max_lora_seq_length: u64,
}

#[test]
fn req41_lora_seq_length_matches_shared_fixture() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("lora_limits")
        .join("lora_seq_length.json");
    let len = fs::metadata(&path).expect("stat fixture").len();
    assert!(len <= MAX_FIXTURE_BYTES, "fixture too large: {len}");
    let fx: Fixture =
        serde_json::from_slice(&fs::read(&path).expect("read fixture")).expect("parse fixture");
    assert_eq!(MIN_LORA_SEQ_LENGTH, fx.min_lora_seq_length);
    assert_eq!(MAX_LORA_SEQ_LENGTH, fx.max_lora_seq_length);
}
