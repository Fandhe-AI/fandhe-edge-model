//! pickle 偽装・非 ONNX ファイルの拒否（REQ-39・TASK-39.2-2・#154・PoC-20 ケース 2 (a)(b)(d)。
//! 証拠種別: テストハーネス）。
//!
//! マーカー pickle は「unpickle されたらマーカーパスに空ファイルを作る」だけの無害なバイト列で、
//! テスト内で生成する。Rust のガード層は pickle を逆シリアル化しないため、マーカー不在の確認は
//! 将来の回帰（外部プロセスでの解析導入等）を検出する番兵である。

// 経路検証コア（`open_confined`）は Linux・macOS のみ対応し、他 OS では常に `UnsupportedPlatform` を返す
// （REQ-39・TASK-39.4-1）。内容検査へ到達するテストは対応 OS に限定し、共有ヘルパーの未使用警告は許容する。
#![cfg_attr(
    not(any(target_os = "linux", target_os = "macos")),
    allow(dead_code, unused_imports)
)]

use fandhe_edge_guard::format::{FileFormat, FormatRejection};
use fandhe_edge_guard::model_file::{ModelFileRejection, open_onnx_model_file};
use std::fs;
use std::path::{Path, PathBuf};

const MIN_ONNX: [u8; 9] = [0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78];
const CANARY: &str = "CANARY_SECRET";

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fandhe-guard-disg-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// プロトコル 2/4/5 の先頭を持つ pickle 形（GLOBAL builtins.open ＋ REDUCE）。マーカー作成のみで外部コマンドなし。
fn marker_pickle(proto: u8, marker: &Path) -> Vec<u8> {
    let mut b = vec![0x80, proto];
    b.extend_from_slice(b"cbuiltins\nopen\n(S'");
    b.extend_from_slice(marker.to_string_lossy().as_bytes());
    b.extend_from_slice(b"'\nS'w'\ntR.");
    b.extend_from_slice(CANARY.as_bytes());
    b
}

/// プロトコル 0 のテキスト形式。
fn marker_pickle_text(marker: &Path) -> Vec<u8> {
    format!(
        "cbuiltins\nopen\n(S'{}'\nS'w'\ntR.{CANARY}",
        marker.display()
    )
    .into_bytes()
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    fs::write(&p, bytes).unwrap();
    p
}

fn assert_no_leak(dir: &Path, msg: &str) {
    assert!(!msg.contains(CANARY));
    assert!(!msg.contains(&*dir.to_string_lossy()));
    assert!(!msg.contains("fandhe-guard-disg"));
}

fn assert_not_allowed(dir: &Path, name: &str, bytes: &[u8], expected: FileFormat) {
    write(dir, name, bytes);
    let err = open_onnx_model_file(dir, Path::new(name), 1 << 20).unwrap_err();
    assert_eq!(err.exit_code().code(), 64);
    assert_eq!(err.reason_code(), "format_not_allowed");
    match &err {
        ModelFileRejection::Format(FormatRejection::NotAllowed { detected, allowed }) => {
            assert_eq!(*detected, expected, "{name}");
            assert_eq!(allowed, &vec![FileFormat::Onnx]);
        }
        other => panic!("unexpected: {other:?}"),
    }
    assert_no_leak(dir, &err.to_string());
}

fn file_count(dir: &Path) -> usize {
    fs::read_dir(dir).unwrap().count()
}

/// PoC-20 case_a: `.onnx` として置いた pickle（プロトコル 2/4/5・テキスト）を拒否し、マーカーは 0 件。
#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn req39_pickle_disguised_as_onnx_is_rejected() {
    let dir = temp_dir("a");
    let marker = dir.join("MARKER");
    for proto in [2u8, 4, 5] {
        assert_not_allowed(
            &dir,
            "model.onnx",
            &marker_pickle(proto, &marker),
            FileFormat::Pickle,
        );
    }
    assert_not_allowed(
        &dir,
        "model.onnx",
        &marker_pickle_text(&marker),
        FileFormat::Unknown,
    );
    assert!(!marker.exists());
    assert_eq!(file_count(&dir), 1);
    fs::remove_dir_all(&dir).unwrap();
}

/// PoC-20 case_b: `.pt`・`.npy` として置いた pickle は拡張子で拒否（読まない）。
#[test]
fn req39_pickle_as_pt_or_npy_is_rejected_by_extension() {
    let dir = temp_dir("b");
    let marker = dir.join("MARKER");
    for name in ["model.pt", "model.npy"] {
        write(&dir, name, &marker_pickle(2, &marker));
        let err = open_onnx_model_file(&dir, Path::new(name), 1 << 20).unwrap_err();
        assert!(matches!(
            err,
            ModelFileRejection::Format(FormatRejection::ExtensionNotAllowed { .. })
        ));
        assert_eq!(err.exit_code().code(), 64);
        assert_eq!(err.reason_code(), "extension_not_allowed");
        assert_no_leak(&dir, &err.to_string());
    }
    assert!(!marker.exists());
    assert_eq!(file_count(&dir), 2);
    fs::remove_dir_all(&dir).unwrap();
}

/// 実形式（zip・npy・gguf）の偽装と PoC-20 case_d（非 ONNX）・空ファイルの拒否。
#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn req39_non_onnx_content_is_rejected() {
    let dir = temp_dir("d");
    let marker = dir.join("MARKER");
    let mut zip = b"PK\x03\x04".to_vec();
    zip.extend_from_slice(b"archive/data.pkl");
    assert_not_allowed(&dir, "model.onnx", &zip, FileFormat::Zip);
    let mut npy = b"\x93NUMPY\x01\x00".to_vec();
    npy.extend_from_slice(b"{'descr': '|O'}");
    npy.extend_from_slice(&marker_pickle(2, &marker));
    assert_not_allowed(&dir, "model.onnx", &npy, FileFormat::Npy);
    let junk = b"NOT AN ONNX FILE".repeat(100);
    assert_eq!(junk.len(), 1600);
    assert_not_allowed(&dir, "model.onnx", &junk, FileFormat::Unknown);
    assert_not_allowed(&dir, "model.onnx", b"", FileFormat::Unknown);
    assert_not_allowed(
        &dir,
        "model.onnx",
        b"GGUF\x03\x00\x00\x00",
        FileFormat::Gguf,
    );
    assert!(!marker.exists());
    assert_eq!(file_count(&dir), 1);
    fs::remove_dir_all(&dir).unwrap();
}

/// 陽性対照と、拡張子検査・内容検査が独立に効くこと。
#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn req39_valid_onnx_passes_and_checks_are_independent() {
    let dir = temp_dir("ok");
    write(&dir, "model.onnx", &MIN_ONNX);
    let f = open_onnx_model_file(&dir, Path::new("model.onnx"), 1 << 20).unwrap();
    assert_eq!(f.format(), FileFormat::Onnx);
    assert_eq!(f.as_bytes(), &MIN_ONNX);
    write(&dir, "model.pt", &MIN_ONNX);
    assert!(matches!(
        open_onnx_model_file(&dir, Path::new("model.pt"), 1 << 20),
        Err(ModelFileRejection::Format(
            FormatRejection::ExtensionNotAllowed { .. }
        ))
    ));
    fs::remove_dir_all(&dir).unwrap();
}

/// 経路検証未対応 OS（Windows 等）では、拡張子が適格でも内容検査へ進まず fail-closed で拒否する
/// （`UnsupportedPlatform`・終了コード 70。REQ-39・TASK-39.4-1）。
#[test]
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn req39_unsupported_platform_fails_closed_for_pickle_disguise() {
    use fandhe_edge_guard::path::PathRejection;
    let dir = temp_dir("unsupported");
    let marker = dir.join("MARKER");
    write(&dir, "model.onnx", &marker_pickle(2, &marker));
    let err = open_onnx_model_file(&dir, Path::new("model.onnx"), 1 << 20).unwrap_err();
    assert!(matches!(
        err,
        ModelFileRejection::Path(PathRejection::UnsupportedPlatform)
    ));
    assert_eq!(err.exit_code().code(), 70);
    assert_eq!(err.reason_code(), "unsupported_platform");
    assert!(!marker.exists());
    fs::remove_dir_all(&dir).unwrap();
}
