//! 学習側と共有するゴールデンベクタとの全件一致（REQ-41・REQ-32・TASK-41.4-5・#413）。
//!
//! - 合成 tokenizer（`fixtures/preprocess/tokenizer_synthetic.json`）: 期待 id は学習側の製品実装
//!   `Qwen2Tokenizer.encode` が出す（生成器 `trainer/tools/gen_tokenizer_vectors.py`）。CI で実行する。
//!   証拠種別: テストハーネス
//! - 実物 Qwen2.5-0.5B-Instruct の `tokenizer.json`: HF `tokenizers` 由来の期待 id
//!   （`tokenizer_vectors.json`）との一致。実物は同梱しないため `#[ignore]` の実機前提テストで、
//!   `FANDHE_EDGE_QWEN_DIR` を要する。証拠種別: 実機
//!
//! 不一致のときは Rust 側を直し、fixture は直さない。`env -i` での同照合は `env_isolation.rs`。

#[path = "common/tokenizer_vectors.rs"]
mod tokenizer_vectors;

use std::path::PathBuf;
use tokenizer_vectors::{check_vectors, preprocess_fixture};

/// REQ-41: 合成 fixture の 189 件が全件一致する。
#[test]
fn req41_synthetic_vectors_all_match() {
    let n = check_vectors(
        &preprocess_fixture("tokenizer_synthetic.json"),
        &preprocess_fixture("tokenizer_vectors_synthetic.json"),
    );
    assert_eq!(n, 189);
}

/// REQ-41: 実物 `tokenizer.json` での 23 件の全件一致（実機前提。`FANDHE_EDGE_QWEN_DIR` 必須）。
#[test]
#[ignore = "REQ-41: real-machine; needs FANDHE_EDGE_QWEN_DIR/tokenizer.json"]
fn req41_real_tokenizer_vectors_all_match() {
    let dir = std::env::var_os("FANDHE_EDGE_QWEN_DIR")
        .expect("FANDHE_EDGE_QWEN_DIR must be set for this test");
    let n = check_vectors(
        &PathBuf::from(dir).join("tokenizer.json"),
        &preprocess_fixture("tokenizer_vectors.json"),
    );
    assert_eq!(n, 23);
}
