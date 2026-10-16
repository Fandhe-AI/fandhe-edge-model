//! 入力表現の宣言を学習側（Python）と同じに解釈することの共有 fixture 照合
//! （REQ-41・TASK-41.3-2・#403。証拠種別: テストハーネス）。
//!
//! 対の Python テストは `trainer/tests/test_input_representation_fixture.py`。どちらも
//! `fixtures/input_representation/declaration_cases.json` を読み、期待値を緩めない。

use fandhe_edge_core::artifact_meta::{
    ArtifactMeta, ArtifactMetaError, PackageInputRepresentation,
};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

fn load(rel: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(rel);
    let len = fs::metadata(&path).expect("metadata").len();
    assert!(len <= MAX_FIXTURE_BYTES, "{rel}: too large");
    serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("json")
}

fn name_of(r: PackageInputRepresentation) -> &'static str {
    match r {
        PackageInputRepresentation::Byte => "byte",
        PackageInputRepresentation::Tokenizer => "tokenizer",
    }
}

/// REQ-41: 全ケースで Rust の解釈が fixture の期待値と一致する。
#[test]
fn req41_declaration_cases_match_fixture() {
    let doc = load("input_representation/declaration_cases.json");
    assert_eq!(
        name_of(PackageInputRepresentation::default()),
        doc["default"].as_str().expect("default")
    );
    assert_eq!(
        doc["accepted_values"],
        serde_json::json!(["byte", "tokenizer"])
    );
    let cases = doc["cases"].as_array().expect("cases");
    assert!(cases.len() >= 10);
    for c in cases {
        let name = c["name"].as_str().expect("name");
        let mut art = doc["base_artifact"].clone();
        if c["present"].as_bool().expect("present") {
            art["input_representation"] = c["value"].clone();
        }
        let got = match ArtifactMeta::parse(art.to_string().as_bytes()) {
            Ok(m) => name_of(m.input_representation()),
            Err(ArtifactMetaError::Malformed) => "reject",
            Err(_) => "other-error",
        };
        assert_eq!(got, c["expect"].as_str().expect("expect"), "case {name}");
    }
}

/// REQ-41: 学習側が実際に書く `artifact`（学習結果 JSON の共有 fixture）は byte と解釈される。
#[test]
fn req41_trainer_written_artifact_is_byte() {
    let doc = load("train_contract/result_ok.json");
    let meta = ArtifactMeta::parse(doc["artifact"].to_string().as_bytes()).expect("parse");
    assert_eq!(
        meta.input_representation(),
        PackageInputRepresentation::Byte
    );
}
