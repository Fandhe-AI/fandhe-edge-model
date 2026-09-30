//! ファイルサイズ上限（暫定 1 GiB）の読み込み前検証の結合テスト（REQ-39・TASK-39.5-3・#172・PoC-20 ケース 3）。
//!
//! スパースファイル（`set_len`。ディスクを消費しない）で 1 GiB の境界を検証する（証拠種別: テストハーネス）。
//! 上限内の 1 GiB ファイルには読み込み系の関数を呼ばない（メモリへ 1 GiB を読み込んでしまうため）。
#![cfg(any(target_os = "linux", target_os = "macos"))]

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::fs::FsError;
use fandhe_edge_guard::file_size::{
    FileSizeRejection, MAX_READ_FILE_BYTES, SizeCheckedOpenRejection, check_file_size,
    open_confined_size_checked,
};
use fandhe_edge_guard::format::{FormatAllowlist, FormatRejection, check_open_file};
use fandhe_edge_guard::path::PathRejection;
use std::fs::{File, OpenOptions};
use std::path::PathBuf;

const GIB: u64 = 1_073_741_824;

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("fe-guard-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sparse(dir: &TempDir, name: &str, len: u64) -> PathBuf {
    let p = dir.0.join(name);
    File::create(&p).unwrap().set_len(len).unwrap();
    p
}

#[test]
fn req39_file_size_exactly_1gib_is_accepted() {
    let d = TempDir::new("sz-ok");
    let f = File::open(sparse(&d, "a.bin", GIB)).unwrap();
    assert_eq!(check_file_size(&f).unwrap(), 1_073_741_824);
}

#[test]
fn req39_file_size_1gib_plus_1_is_rejected() {
    let d = TempDir::new("sz-ng");
    let f = File::open(sparse(&d, "a.bin", GIB + 1)).unwrap();
    let e = check_file_size(&f).unwrap_err();
    assert!(matches!(
        e,
        FileSizeRejection::TooLarge {
            size: 1_073_741_825,
            limit: 1_073_741_824
        }
    ));
    assert_eq!(e.exit_code(), ExitCode::LimitExceeded);
    assert_eq!(e.reason_code(), "file_size_limit_exceeded");
}

#[test]
fn req39_file_size_poc20_case3_1gib_plus_1kib() {
    let d = TempDir::new("sz-poc");
    let f = File::open(sparse(&d, "a.bin", GIB + 1024)).unwrap();
    assert!(matches!(
        check_file_size(&f),
        Err(FileSizeRejection::TooLarge {
            size: 1_073_742_848,
            ..
        })
    ));
}

/// 読み取り権限のないハンドルでも TooLarge になる = 読まずに判定している。
#[test]
fn req39_file_size_rejected_before_read() {
    let d = TempDir::new("sz-noread");
    let p = sparse(&d, "a.bin", GIB + 1);
    let f = OpenOptions::new().write(true).open(p).unwrap();
    assert!(matches!(
        check_file_size(&f),
        Err(FileSizeRejection::TooLarge { .. })
    ));
}

#[test]
fn req39_format_check_clamps_to_ceiling() {
    let d = TempDir::new("sz-clamp");
    let f = File::open(sparse(&d, "a.bin", GIB + 1)).unwrap();
    let p = d.0.join("a.bin");
    let e = check_open_file(f, &p, &FormatAllowlist::onnx_only(), u64::MAX).unwrap_err();
    assert_eq!(e.exit_code(), ExitCode::LimitExceeded);
    match e {
        FormatRejection::Io(FsError::TooLarge { size, limit, .. }) => {
            assert_eq!(size, 1_073_741_825);
            assert_eq!(limit, MAX_READ_FILE_BYTES);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn req39_open_confined_size_checked_boundaries() {
    let d = TempDir::new("sz-conf");
    let root = d.0.join("root");
    std::fs::create_dir_all(&root).unwrap();
    File::create(root.join("ok.bin"))
        .unwrap()
        .set_len(GIB)
        .unwrap();
    File::create(root.join("big.bin"))
        .unwrap()
        .set_len(GIB + 1)
        .unwrap();
    File::create(d.0.join("outside.bin")).unwrap();

    let ok = open_confined_size_checked(&root, "ok.bin".as_ref(), u64::MAX).unwrap();
    assert_eq!(ok.size(), 1_073_741_824);
    match open_confined_size_checked(&root, "big.bin".as_ref(), u64::MAX) {
        Err(SizeCheckedOpenRejection::Size(FileSizeRejection::TooLarge { size, .. })) => {
            assert_eq!(size, 1_073_741_825)
        }
        other => panic!("unexpected {other:?}"),
    }
    assert!(matches!(
        open_confined_size_checked(&root, "../outside.bin".as_ref(), u64::MAX),
        Err(SizeCheckedOpenRejection::Path(
            PathRejection::Escapes { .. }
        ))
    ));
}
