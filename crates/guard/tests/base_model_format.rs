//! ベースモデルの形式許可リスト（safetensors・tokenizer JSON）と pickle 系の拒否の結合テスト
//! （REQ-39・REQ-41・TASK-41.2-1・#397。証拠種別: テストハーネス）。
//!
//! 公開 API の `open_checked_file` を `FormatAllowlist::base_model()` で呼び、pickle の偽装と
//! 許可リスト外の形式が `invalid_input`（64）で拒否されること、正しい形式が通ることを固定する。
//! 経路の閉じ込め・ファイル名との照合は #398 の責務で、ここでは使わない。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_guard::format::{FileFormat, FormatAllowlist, FormatRejection, open_checked_file};
use std::path::PathBuf;

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fandhe-guard-bm-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn safetensors() -> Vec<u8> {
    let hdr = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut v = (hdr.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(hdr.as_bytes());
    v.extend_from_slice(&[0; 4]);
    v
}

const TOKENIZER: &[u8] = br#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}"#;
const MIN_ONNX: &[u8] = &[0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78];

/// (ファイル名, 内容, 検出される形式)。
fn rejected_cases() -> Vec<(&'static str, Vec<u8>, FileFormat)> {
    let mut pkl4 = vec![0x80, 0x04];
    pkl4.extend_from_slice(b"canary-not-printed");
    let mut broken = (1000u64).to_le_bytes().to_vec();
    broken.extend_from_slice(b"{}");
    vec![
        (
            "model.safetensors",
            vec![0x80, 0x02, b'c'],
            FileFormat::Pickle,
        ),
        ("tokenizer.json", pkl4.clone(), FileFormat::Pickle),
        (
            "pytorch_model.bin",
            vec![0x80, 0x05, 0x95],
            FileFormat::Pickle,
        ),
        ("proto0.bin", b"cos\nsystem\n".to_vec(), FileFormat::Unknown),
        ("model.pt", b"PK\x03\x04rest".to_vec(), FileFormat::Zip),
        ("w.npy", b"\x93NUMPY\x01\x00".to_vec(), FileFormat::Npy),
        ("w.gguf", b"GGUF\x03\x00".to_vec(), FileFormat::Gguf),
        ("m.onnx", MIN_ONNX.to_vec(), FileFormat::Onnx),
        ("empty", Vec::new(), FileFormat::Unknown),
        ("broken.safetensors", broken, FileFormat::Unknown),
    ]
}

/// REQ-39・TASK-41.2-1: pickle の偽装と許可リスト外の形式は invalid_input(64) で拒否される。
#[test]
fn req39_base_model_rejects_pickle_and_others() {
    let dir = temp_dir("reject");
    let al = FormatAllowlist::base_model();
    for (name, bytes, detected) in rejected_cases() {
        let p = dir.join(name);
        std::fs::write(&p, &bytes).unwrap();
        let err = open_checked_file(&p, &al, 1 << 20).unwrap_err();
        match &err {
            FormatRejection::NotAllowed { detected: d, .. } => assert_eq!(*d, detected, "{name}"),
            other => panic!("{name}: unexpected {other:?}"),
        }
        assert_eq!(err.exit_code(), ExitCode::InvalidInput, "{name}");
        assert_eq!(err.exit_code().code(), 64);
        assert_eq!(err.reason_code(), "format_not_allowed");
        // 情報漏えいなし: パスも内容も出さない。
        let msg = err.to_string();
        assert!(!msg.contains(name) && !msg.contains("canary"), "{msg}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// REQ-39・TASK-41.2-1: 陽性対照。正しい safetensors・tokenizer.json は拡張子に依らず通る。
#[test]
fn req39_base_model_accepts_valid_by_content() {
    let dir = temp_dir("accept");
    let al = FormatAllowlist::base_model();
    let st = safetensors();
    let p = dir.join("weights.dat");
    std::fs::write(&p, &st).unwrap();
    let ok = open_checked_file(&p, &al, 1 << 20).unwrap();
    assert_eq!(ok.format(), FileFormat::Safetensors);
    assert_eq!(ok.as_bytes(), st.as_slice());
    let p = dir.join("tokenizer.json");
    std::fs::write(&p, TOKENIZER).unwrap();
    let ok = open_checked_file(&p, &al, 1 << 20).unwrap();
    assert_eq!(ok.format(), FileFormat::TokenizerJson);
    // onnx_only は両者を拒否する（従来どおり）。
    let err = open_checked_file(&p, &FormatAllowlist::onnx_only(), 1 << 20).unwrap_err();
    assert_eq!(err.exit_code(), ExitCode::InvalidInput);
    std::fs::remove_dir_all(&dir).unwrap();
}
