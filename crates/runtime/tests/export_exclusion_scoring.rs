//! 追加学習の種類 `qwen2_lora` の対応範囲外の除外と理由の記録の結合テスト
//! （REQ-41 異常系・TASK-41.6-4・#433）。
//!
//! 証拠種別: テストハーネス。ONNX はコミット済み C3 の演算名を差し替えた合成物、トークナイザーは
//! 合成の最小 JSON で、実物の Qwen 重みは使わない（実行器が未接続のため範囲内でも
//! `parity_unverified`。接続は #434）。失敗メッセージに入力本文を出さない（security.md）。

#[allow(dead_code)] // 共有の編集器のうち本テストで使わない関数がある
#[path = "common/proto_builder.rs"]
mod proto_builder;

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_runtime::export_exclusion::{
    CandidateDecision, EvidenceKind, ExclusionReason, ExportConfig, ExportKind, InferenceRuntime,
    NumericFormat, ScreeningError, screen_candidate, screen_scoring_candidate,
};
use fandhe_edge_runtime::onnx::{FINETUNE_MODEL_LIMITS, ModelKind, scan_node_ops};
use fandhe_edge_runtime::tokenizer::{SPLIT_REGEX, bytes_to_unicode};
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;

fn c3_bytes() -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/onnx_parity/c3.onnx");
    fs::read(p).expect("read c3 fixture")
}

fn tokenizer_json() -> Value {
    let mut vocab = serde_json::Map::new();
    for (i, c) in bytes_to_unicode().iter().enumerate() {
        vocab.insert(c.to_string(), json!(i));
    }
    vocab.insert("ab".into(), json!(256));
    let special = |id: u32, content: &str| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false,
               "rstrip": false, "normalized": false, "special": true})
    };
    let bl = json!({"type": "ByteLevel", "add_prefix_space": false,
                    "trim_offsets": false, "use_regex": false});
    json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [special(257, "<|im_start|>"), special(258, "<|im_end|>"),
                         special(259, "<|endoftext|>")],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": SPLIT_REGEX}, "behavior": "Isolated", "invert": false},
            bl.clone()
        ]},
        "post_processor": bl.clone(),
        "decoder": bl,
        "model": {"type": "BPE", "dropout": null, "unk_token": null,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": false, "byte_fallback": false, "vocab": vocab, "merges": ["a b"]}
    })
}

struct Files {
    dir: PathBuf,
    onnx: PathBuf,
    onnx_sha: Sha256Digest,
    tok: PathBuf,
    tok_sha: Sha256Digest,
}

impl Drop for Files {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn put(tag: &str, onnx: &[u8], tok: &[u8]) -> Files {
    let dir = std::env::temp_dir().join(format!("fe433-{}-{tag}", std::process::id()));
    fs::create_dir_all(&dir).expect("mkdir");
    let (onnx_p, tok_p) = (dir.join("model.onnx"), dir.join("tokenizer.json"));
    fs::write(&onnx_p, onnx).expect("write onnx");
    fs::write(&tok_p, tok).expect("write tok");
    Files {
        dir,
        onnx: onnx_p,
        onnx_sha: Sha256Digest::of_bytes(onnx),
        tok: tok_p,
        tok_sha: Sha256Digest::of_bytes(tok),
    }
}

fn config() -> ExportConfig {
    ExportConfig {
        kind: ExportKind::Qwen2Lora,
        runtime: InferenceRuntime::Own,
        format: NumericFormat::F32,
    }
}

fn run(f: &Files) -> Result<CandidateDecision, ScreeningError> {
    run_with(config(), f)
}

fn run_with(c: ExportConfig, f: &Files) -> Result<CandidateDecision, ScreeningError> {
    screen_scoring_candidate(c, &f.onnx, &f.onnx_sha, &f.tok, &f.tok_sha)
}

fn excluded(
    d: Result<CandidateDecision, ScreeningError>,
) -> fandhe_edge_runtime::export_exclusion::ExclusionRecord {
    match d.expect("decision") {
        CandidateDecision::Excluded(r) => r,
        CandidateDecision::Eligible(_) => panic!("must be excluded"),
    }
}

fn all_nodes_to(bytes: &[u8], op: &str) -> Vec<u8> {
    let n = scan_node_ops(bytes, &FINETUNE_MODEL_LIMITS)
        .expect("scan")
        .len();
    (0..n).fold(bytes.to_vec(), |b, i| proto_builder::set_op_type(&b, i, op))
}

/// REQ-41: 範囲外の演算（C3 の Conv 等）を含む採点グラフは演算名つきで除外される。
#[test]
fn req41_unsupported_ops_excluded() {
    let tok = tokenizer_json().to_string();
    let c3 = c3_bytes();
    let f = put("ops", &c3, tok.as_bytes());
    let r = excluded(run(&f));
    assert_eq!(r.code(), "unsupported_ops");
    assert_eq!(r.evidence, EvidenceKind::TestHarness);
    let ExclusionReason::UnsupportedOps { ops, total } = &r.reason else {
        panic!("reason");
    };
    let expected: Vec<String> = EXPECTED_C3_UNSUPPORTED
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(ops, &expected);
    assert_eq!(*total, expected.len());

    let erf = proto_builder::set_op_type(&c3, 0, "Erf");
    let f2 = put("ops2", &erf, tok.as_bytes());
    let r2 = excluded(run(&f2));
    let ExclusionReason::UnsupportedOps { ops, .. } = &r2.reason else {
        panic!("reason");
    };
    assert!(ops.iter().any(|o| o == "Erf"), "{ops:?}");
}

/// REQ-41: 未対応のトークナイザー（未知キー・WordPiece）は節名つきで除外される。
#[test]
fn req41_unsupported_tokenizer_excluded() {
    let c3 = c3_bytes();
    let mut with_key = tokenizer_json();
    with_key["unknown_key"] = json!(1);
    let mut wordpiece = tokenizer_json();
    wordpiece["model"]["type"] = json!("WordPiece");
    for (tag, v, section) in [("k", with_key, "top-level"), ("w", wordpiece, "model")] {
        let f = put(tag, &c3, v.to_string().as_bytes());
        let r = excluded(run(&f));
        assert_eq!(r.code(), "unsupported_tokenizer", "{tag}");
        assert_eq!(r.detail_code(), Some(section), "{tag}");
    }
}

/// REQ-39・REQ-41: トークナイザーの完全性・破損・上限は除外にせず全体停止にする。
#[test]
fn req39_tokenizer_integrity_and_malformed_propagate() {
    let c3 = c3_bytes();
    let good = tokenizer_json().to_string();
    let mut f = put("t1", &c3, good.as_bytes());
    f.tok_sha = Sha256Digest::of_bytes(b"other");
    let e = run(&f).expect_err("integrity");
    assert!(matches!(e, ScreeningError::Tokenizer(_)));
    assert_eq!(e.code(), "integrity_mismatch");

    let f = put("t2", &c3, b"{not json");
    assert_eq!(run(&f).expect_err("malformed").code(), "malformed_json");

    let big = vec![b' '; 16 * 1024 * 1024 + 1];
    let f = put("t3", &c3, &big);
    assert_eq!(run(&f).expect_err("limit").code(), "limit_exceeded");
}

/// REQ-39・REQ-41: トークナイザーが範囲外でも ONNX の sha256 不一致は全体停止（完全性が先）。
#[test]
fn req39_scoring_integrity_checked_before_exclusion() {
    let mut wordpiece = tokenizer_json();
    wordpiece["model"]["type"] = json!("WordPiece");
    let mut f = put("i", &c3_bytes(), wordpiece.to_string().as_bytes());
    f.onnx_sha = Sha256Digest::of_bytes(b"other");
    let e = run(&f).expect_err("integrity");
    assert!(matches!(e, ScreeningError::Load(_)));
    assert_eq!(e.code(), "integrity_mismatch");
}

/// REQ-41: 範囲内の構造でも実行器が未接続のため `Eligible` にせず `parity_unverified` で除外する。
#[test]
fn req41_in_range_scoring_package_is_parity_unverified() {
    let adds = all_nodes_to(&c3_bytes(), "Add");
    let f = put("ok", &adds, tokenizer_json().to_string().as_bytes());
    let r = excluded(run(&f));
    assert_eq!(r.code(), "parity_unverified");
    assert_eq!(r.evidence, EvidenceKind::TestHarness);
}

/// REQ-41: opset 欠落などの形式不成立は読み込み拒否（runtime_rejected_model）で除外される。
#[test]
fn req41_wrong_opset_is_runtime_rejected() {
    use proto_builder::Val;
    let opset14 = proto_builder::edit_model(&c3_bytes(), |m| {
        let i = m.iter().position(|(n, _)| *n == 8).expect("opset");
        let Val::Len(b) = &m[i].1 else {
            panic!("opset")
        };
        let mut inner = proto_builder::parse(b);
        let v = inner.iter().position(|(n, _)| *n == 2).expect("version");
        inner[v].1 = Val::Varint(14);
        m[i].1 = Val::Len(proto_builder::encode(&inner));
    });
    let f = put("op", &opset14, tokenizer_json().to_string().as_bytes());
    let r = excluded(run(&f));
    assert_eq!(r.code(), "runtime_rejected_model");
    assert_eq!(r.detail_code(), Some("unsupported_opset"));
}

/// REQ-32・REQ-41: Tract × F32 等の未提供ランタイムは `runtime_not_available`。
#[test]
fn req32_runtime_not_available_for_scoring() {
    let f = put("rt", &c3_bytes(), tokenizer_json().to_string().as_bytes());
    let c = ExportConfig {
        runtime: InferenceRuntime::Tract,
        ..config()
    };
    assert_eq!(excluded(run_with(c, &f)).code(), "runtime_not_available");
}

/// REQ-41: 推論側に実行器の無い種類は従来経路（`screen_candidate`）で `unsupported_kind` の除外になり、
/// `load` は呼ばれない。
#[test]
fn req41_model_kind_less_config_rejected_in_legacy_path() {
    let r = excluded(screen_candidate(
        config(),
        |_: ModelKind| panic!("load must not be called"),
        &[],
    ));
    assert_eq!(r.code(), "runtime_rejected_model");
    assert_eq!(r.detail_code(), Some("unsupported_kind"));
}

const EXPECTED_C3_UNSUPPORTED: [&str; 3] = ["Conv", "ReduceMax", "Relu"];
