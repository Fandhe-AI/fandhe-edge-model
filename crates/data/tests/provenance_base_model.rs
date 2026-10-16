//! ベースモデル来歴の公開 API 結合テスト（REQ-41・REQ-40・TASK-41.2-3・issue #399。
//! 証拠種別: テストハーネス）。

use fandhe_edge_core::artifact_meta::{ArtifactMeta, MAX_ARTIFACT_META_BYTES};
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_data::provenance::ModelName;
use fandhe_edge_data::provenance::base_model::{
    BaseModelFile, BaseModelProvenance, BaseModelProvenanceError, base_model_to_json,
    base_model_to_json_value,
};

const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

fn files() -> Vec<BaseModelFile> {
    vec![
        BaseModelFile::new("tokenizer.json", 0, Sha256Digest::of_bytes(b"")),
        BaseModelFile::new("model.safetensors", 3, Sha256Digest::of_bytes(b"abc")),
    ]
}

fn record(files: Vec<BaseModelFile>) -> BaseModelProvenance {
    BaseModelProvenance::new(
        ModelName::new("Qwen/Qwen2.5-1.5B-Instruct").unwrap(),
        "Apache-2.0",
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
        files,
    )
    .unwrap()
}

#[test]
fn req41_records_name_license_origin_and_file_hashes() {
    let expected = format!(
        r#"{{"files":[{{"path":"model.safetensors","sha256":"{ABC}","size_bytes":3}},{{"path":"tokenizer.json","sha256":"{EMPTY}","size_bytes":0}}],"license":"Apache-2.0","model_name":"Qwen/Qwen2.5-1.5B-Instruct","origin":"https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct"}}"#
    );
    assert_eq!(base_model_to_json(&record(files())), expected);
}

#[test]
fn req41_output_is_independent_of_input_order() {
    let mut rev = files();
    rev.reverse();
    assert_eq!(
        base_model_to_json(&record(files())),
        base_model_to_json(&record(rev))
    );
}

#[test]
fn req41_artifact_json_with_base_model_still_parses() {
    let meta = serde_json::json!({
        "onnx_file": "model.onnx", "kind": "c1", "kind_version": 1, "max_bytes": 1024,
        "label_order": ["a", "b"], "onnx_sha256": ABC,
        "base_model": base_model_to_json_value(&record(files())),
    });
    let parsed = ArtifactMeta::parse(meta.to_string().as_bytes()).unwrap();
    assert_eq!(parsed.onnx_file(), "model.onnx");
    assert_eq!(parsed.kind(), "c1");
    assert_eq!(parsed.onnx_sha256(), ABC);
}

#[test]
fn req41_max_size_record_fits_artifact_meta_limit() {
    let big = (0..256)
        .map(|i| {
            let p = format!("{i:03}{}", "x".repeat(509));
            BaseModelFile::new(&p, u64::MAX, Sha256Digest::of_bytes(b"x"))
        })
        .collect();
    let p = BaseModelProvenance::new(
        ModelName::new(&"n".repeat(256)).unwrap(),
        &"l".repeat(256),
        &"o".repeat(1024),
        big,
    )
    .unwrap();
    assert!(u64::try_from(base_model_to_json(&p).len()).unwrap() <= MAX_ARTIFACT_META_BYTES);
}

#[test]
fn req41_error_display_omits_input() {
    let e = BaseModelProvenance::new(
        ModelName::new("m").unwrap(),
        "MIT",
        "SECRET-ORIGIN\n",
        files(),
    )
    .unwrap_err();
    assert_eq!(e, BaseModelProvenanceError::InvalidOrigin);
    assert!(!e.to_string().contains("SECRET"));
}
