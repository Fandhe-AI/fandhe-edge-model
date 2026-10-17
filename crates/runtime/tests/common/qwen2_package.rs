//! `fixtures/qwen2_parity/`（qwen2_lora の自己完結パッケージ）の読み込み（テスト専用。REQ-41・REQ-32・
//! TASK-41.6-5・#434）。
//!
//! `qwen2_parity.rs`（全件一致）と `env_isolation.rs`（`env -i` の子）が共有する。CLI の `infer` 配線
//! （`artifact.json` の解釈）は未実装のため、ここで `artifact.json` から必要な欄だけを読み、
//! 実行時と同じ検証（サイズ上限・sha256 照合・shard の完全性）を通して推論パイプラインを組む。
//! 失敗メッセージはファイル名・件数のみで、入力本文・重みは出さない（security.md）。

use fandhe_edge_core::fs::read_bounded;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_runtime::onnx::{ExternalShard, ModelKind, OnnxBackend};
use fandhe_edge_runtime::pipeline::InferencePipeline;
use fandhe_edge_runtime::preprocess::TokenizerPreprocessor;
use fandhe_edge_runtime::tokenizer::TokenizerSpec;
use serde_json::Value;
use std::fs::{self, File};
use std::path::PathBuf;

/// JSON 類の読み込み上限（読み込み前に確認。REQ-39）。
const MAX_JSON_BYTES: u64 = 4 * 1024 * 1024;
/// `model.onnx` の読み込み上限（`FINETUNE_MODEL_LIMITS` と同じ 64 MiB）。
const MAX_ONNX_BYTES: u64 = 64 * 1024 * 1024;

pub struct Case {
    pub name: String,
    pub input: String,
    pub label: usize,
    pub probs: Vec<f64>,
}

pub type Pipeline = InferencePipeline<TokenizerPreprocessor, OnnxBackend>;

pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("qwen2_parity")
}

fn read_json(name: &str) -> Value {
    let path = fixture_dir().join(name);
    let len = fs::metadata(&path).expect("json metadata").len();
    assert!(len <= MAX_JSON_BYTES, "{name} too large: {len}");
    serde_json::from_str(&fs::read_to_string(&path).expect("json read")).expect("json parse")
}

fn sha_of(v: &Value) -> Sha256Digest {
    v.as_str().expect("sha string").parse().expect("sha parse")
}

/// `model.onnx` のバイト列と、開いた shard の組（`from_bytes_with_shards` の入力）。
pub fn read_package() -> (Vec<u8>, Vec<ExternalShard>) {
    let art = read_json("artifact.json");
    assert_eq!(art["kind"], "qwen2_lora");
    assert_eq!(art["input_representation"], "tokenizer");
    let dir = fixture_dir();
    let onnx_bytes = read_bounded(
        &dir.join(art["onnx_file"].as_str().expect("onnx_file")),
        MAX_ONNX_BYTES,
    )
    .expect("onnx read");
    assert_eq!(
        Sha256Digest::of_bytes(&onnx_bytes),
        sha_of(&art["onnx_sha256"])
    );
    let shards: Vec<ExternalShard> = art["external_data"]
        .as_array()
        .expect("external_data")
        .iter()
        .map(|s| {
            let location = s["location"].as_str().expect("location").to_string();
            ExternalShard {
                file: File::open(dir.join(&location)).expect("shard open"),
                location,
                sha256: sha_of(&s["sha256"]),
            }
        })
        .collect();
    assert!(!shards.is_empty(), "fixture must exercise external data");
    (onnx_bytes, shards)
}

/// パッケージから推論パイプラインを組む。返す `usize` は `label_order` の長さ。
pub fn load_pipeline() -> (Pipeline, usize) {
    let art = read_json("artifact.json");
    let dir = fixture_dir();
    let tok_sha = sha_of(&art["tokenizer_sha256"]);
    let tok_path = dir.join(art["tokenizer_file"].as_str().expect("tokenizer_file"));
    let spec = TokenizerSpec::load_path(&tok_path, &tok_sha).expect("tokenizer load");
    let (onnx_bytes, shards) = read_package();
    let backend = OnnxBackend::from_bytes_with_shards(&onnx_bytes, ModelKind::Qwen2Lora, shards)
        .expect("load qwen2_lora");
    let k = art["label_order"].as_array().expect("label_order").len();
    assert_eq!(backend.n_classes(), k);
    assert_eq!(
        art["scoring"]["choices"].as_array().expect("choices").len(),
        k
    );
    let pre = TokenizerPreprocessor::new(
        spec,
        art["scoring"]["system_prompt"]
            .as_str()
            .expect("system_prompt")
            .to_string(),
    )
    .expect("system prompt within limit");
    (InferencePipeline::new(pre, backend), k)
}

pub fn load_cases() -> Vec<Case> {
    read_json("cases.json")["cases"]
        .as_array()
        .expect("cases array")
        .iter()
        .map(|c| Case {
            name: c["name"].as_str().expect("name").to_string(),
            input: c["input"].as_str().expect("input").to_string(),
            label: usize::try_from(c["mlx_label_index"].as_u64().expect("label")).expect("usize"),
            probs: c["ref_probs"]
                .as_array()
                .expect("probs")
                .iter()
                .map(|p| p.as_f64().expect("prob"))
                .collect(),
        })
        .collect()
}
