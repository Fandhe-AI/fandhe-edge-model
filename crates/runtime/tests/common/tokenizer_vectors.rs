//! 共有ゴールデンベクタ（`fixtures/preprocess/tokenizer_vectors*.json`）との全件一致の照合
//! （REQ-41・REQ-32・TASK-41.4-5・#413。テスト専用）。
//!
//! `tokenizer_golden.rs`（合成・実物）と `env_isolation.rs`（`env -i` の子）が共有する。
//! 不一致はコード側を直す（fixture・期待値を緩めない）。失敗メッセージには `name` だけを出し、
//! 入力本文は出さない（security.md）。

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_runtime::tokenizer::TokenizerSpec;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

/// vectors ファイルの読み込み上限（読み込み前に確認。REQ-39）。
const MAX_VECTORS_BYTES: u64 = 1024 * 1024;

/// `fixtures/preprocess/<name>` の絶対パス（`CARGO_MANIFEST_DIR` 由来なので環境変数が空でも解決できる）。
pub fn preprocess_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("preprocess")
        .join(name)
}

/// `tokenizer` を vectors の `_meta.tokenizer_json_sha256` で照合しながら読み、全 vector で
/// `encode(input) == ids` を確かめて件数を返す。
pub fn check_vectors(tokenizer: &Path, vectors: &Path) -> usize {
    let len = fs::metadata(vectors).expect("vectors metadata").len();
    assert!(len <= MAX_VECTORS_BYTES, "vectors too large: {len}");
    let root: Value =
        serde_json::from_str(&fs::read_to_string(vectors).expect("vectors read")).expect("json");
    let sha: Sha256Digest = root
        .get("_meta")
        .and_then(|m| m.get("tokenizer_json_sha256"))
        .and_then(Value::as_str)
        .expect("sha field")
        .parse()
        .expect("sha parse");
    let spec = TokenizerSpec::load_path(tokenizer, &sha).expect("load tokenizer");
    let list = root
        .get("vectors")
        .and_then(Value::as_array)
        .expect("vectors array");
    assert!(!list.is_empty(), "no vectors");
    for v in list {
        let name = v.get("name").and_then(Value::as_str).expect("name");
        let input = v.get("input").and_then(Value::as_str).expect("input");
        let expected: Vec<u32> = v
            .get("ids")
            .and_then(Value::as_array)
            .expect("ids")
            .iter()
            .map(|n| u32::try_from(n.as_u64().expect("id")).expect("u32"))
            .collect();
        let got = spec.encode(input).unwrap_or_default();
        assert_eq!(got, expected, "{name}: ids differ");
    }
    list.len()
}
