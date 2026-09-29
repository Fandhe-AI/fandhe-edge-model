//! 公開 API 経由の結合テスト（REQ-39・TASK-39.2-1・#153）。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_guard::format::{
    AllowedFormat, FileFormat, FormatAllowlist, FormatRejection, check_file_format,
};
use std::path::PathBuf;

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fandhe-guard-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// REQ-39・TASK-39.2-1: 内容で判定する（拡張子 .onnx の pickle は拒否、ONNX 形は通過）。
#[test]
fn req39_check_file_format_by_content() {
    let dir = temp_dir("content");
    let onnx = dir.join("m.onnx");
    let fake = dir.join("fake.onnx");
    std::fs::write(&onnx, [0x08, 0x07, 0x3a, 0x00]).unwrap();
    std::fs::write(&fake, [0x80, 0x04, 0x95, 0x00]).unwrap();
    let al = FormatAllowlist::onnx_only();
    let ok: AllowedFormat = check_file_format(&onnx, &al).unwrap();
    assert_eq!(ok.format(), FileFormat::Onnx);
    match check_file_format(&fake, &al).unwrap_err() {
        FormatRejection::NotAllowed { detected, .. } => assert_eq!(detected, FileFormat::Pickle),
        other => panic!("unexpected: {other}"),
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// REQ-39・TASK-39.2-1: 存在しないパス・ディレクトリは Io として InvalidInput(64) で拒否する。
#[test]
fn req39_check_file_format_io_rejections() {
    let dir = temp_dir("io");
    let al = FormatAllowlist::onnx_only();
    for p in [dir.join("missing"), dir.clone()] {
        let err = check_file_format(&p, &al).unwrap_err();
        assert!(matches!(err, FormatRejection::Io(_)));
        assert_eq!(err.exit_code(), ExitCode::InvalidInput);
        assert_eq!(err.reason_code(), "file_unreadable");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// REQ-39・TASK-39.2-1: 先頭 64KiB に graph が無い有効な ONNX 形（大きな doc_string の後に graph）を通す。
#[test]
fn req39_onnx_graph_beyond_prefix_is_allowed() {
    let dir = temp_dir("bigdoc");
    let path = dir.join("big.onnx");
    // ir_version=7, doc_string(field 6, LEN 100000 バイト), graph(field 7, 空)。
    let mut b = vec![0x08, 0x07, 0x32, 0xa0, 0x8d, 0x06];
    b.extend(std::iter::repeat_n(b'x', 100_000));
    b.extend_from_slice(&[0x3a, 0x00]);
    std::fs::write(&path, &b).unwrap();
    let ok = check_file_format(&path, &FormatAllowlist::onnx_only()).unwrap();
    assert_eq!(ok.format(), FileFormat::Onnx);
    // graph が無いまま大きな doc_string だけなら拒否する。
    b.truncate(b.len() - 2);
    std::fs::write(&path, &b).unwrap();
    assert!(check_file_format(&path, &FormatAllowlist::onnx_only()).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// REQ-39・TASK-39.2-1: 実ファイル末尾で切れた protobuf は graph の後でも拒否する。
#[test]
fn req39_onnx_truncated_after_graph_is_rejected() {
    let dir = temp_dir("trunc");
    let path = dir.join("t.onnx");
    std::fs::write(&path, [0x08, 0x07, 0x3a, 0x00, 0x12, 0x80]).unwrap();
    match check_file_format(&path, &FormatAllowlist::onnx_only()).unwrap_err() {
        FormatRejection::NotAllowed { detected, .. } => assert_eq!(detected, FileFormat::Unknown),
        other => panic!("unexpected: {other}"),
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
