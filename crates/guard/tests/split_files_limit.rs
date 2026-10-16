//! 分割ファイル群の受け付けと合計上限の結合テスト（REQ-39・REQ-41・TASK-41.2-5・#580）。
//! スパースファイルで 1 GiB・8 GiB の境界を検証する（証拠種別: テストハーネス）。
#![cfg(any(target_os = "linux", target_os = "macos"))]

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_guard::file_size::FileSizeRejection;
use fandhe_edge_guard::path::{ConfinedPath, open_confined};
use fandhe_edge_guard::split_files::{
    MAX_PACKAGE_TOTAL_BYTES, MAX_SPLIT_FILES, SplitFilesRejection, check_split_files,
    effective_total_limit, verify_split_files,
};
use std::fs::File;
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
    fn open(&self, name: &str) -> (File, ConfinedPath) {
        open_confined(&self.0, name.as_ref()).unwrap()
    }
    fn write(&self, name: &str, bytes: &[u8]) {
        std::fs::write(self.0.join(name), bytes).unwrap();
    }
    fn sparse(&self, name: &str, len: u64) {
        File::create(self.0.join(name))
            .unwrap()
            .set_len(len)
            .unwrap();
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn three(d: &TempDir) -> Vec<(File, ConfinedPath)> {
    d.write("a", b"aaaaaaaaaa");
    d.write("b", b"bbbbbbbbbb");
    d.write("c", b"cccccccccc");
    vec![d.open("a"), d.open("b"), d.open("c")]
}

#[test]
fn req41_split_files_accepted_with_sizes_and_hashes() {
    let d = TempDir::new("sp-ok");
    let r = check_split_files(three(&d), MAX_PACKAGE_TOTAL_BYTES).unwrap();
    assert_eq!(r.total(), 30);
    assert_eq!(r.limit(), 8_589_934_592);
    let sizes: Vec<u64> = r.files().iter().map(|f| f.size()).collect();
    assert_eq!(sizes, vec![10, 10, 10]);
    assert_eq!(r.files()[1].sha256(), Sha256Digest::of_bytes(b"bbbbbbbbbb"));
}

#[test]
fn req39_total_limit_boundary() {
    let d = TempDir::new("sp-bound");
    let e = check_split_files(three(&d), 29).unwrap_err();
    assert!(matches!(
        e,
        SplitFilesRejection::TotalTooLarge {
            total: 30,
            limit: 29
        }
    ));
    assert_eq!(e.exit_code(), ExitCode::LimitExceeded);
    assert_eq!(e.reason_code(), "package_total_size_limit_exceeded");
    assert!(check_split_files(three(&d), 30).is_ok());
}

#[test]
fn req41_total_over_8gib_rejected_without_reading() {
    let d = TempDir::new("sp-8g");
    let mut m = Vec::new();
    for i in 0..9 {
        let n = format!("s{i}");
        d.sparse(&n, GIB);
        m.push(d.open(&n));
    }
    let e = check_split_files(m, u64::MAX).unwrap_err();
    assert!(matches!(
        e,
        SplitFilesRejection::TotalTooLarge {
            total: 9_663_676_416,
            limit: 8_589_934_592
        }
    ));
}

#[test]
fn req39_single_file_over_1gib_rejected() {
    let d = TempDir::new("sp-1g");
    d.write("a", b"x");
    d.sparse("big", GIB + 1);
    let e = check_split_files(vec![d.open("a"), d.open("big")], u64::MAX).unwrap_err();
    assert!(matches!(
        e,
        SplitFilesRejection::File {
            index: 1,
            source: FileSizeRejection::TooLarge {
                size: 1_073_741_825,
                ..
            }
        }
    ));
    assert_eq!(e.exit_code(), ExitCode::LimitExceeded);
    assert_eq!(e.reason_code(), "file_size_limit_exceeded");
}

#[test]
fn req41_verify_split_files_hash_checks() {
    let d = TempDir::new("sp-verify");
    let want: Vec<Sha256Digest> = ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]
        .iter()
        .map(|s| Sha256Digest::of_bytes(s.as_bytes()))
        .collect();
    assert!(verify_split_files(three(&d), &want, u64::MAX).is_ok());

    d.write("b", b"bbbbbbbbbX");
    let m = vec![d.open("a"), d.open("b"), d.open("c")];
    let e = verify_split_files(m, &want, u64::MAX).unwrap_err();
    assert!(matches!(e, SplitFilesRejection::HashMismatch { index: 1 }));
    assert_eq!(e.exit_code(), ExitCode::InvalidInput);
    assert_eq!(e.reason_code(), "split_file_hash_mismatch");

    let e = verify_split_files(three(&d), &want[..2], u64::MAX).unwrap_err();
    assert!(matches!(
        e,
        SplitFilesRejection::DigestCountMismatch {
            expected: 2,
            actual: 3
        }
    ));
    assert_eq!(e.exit_code(), ExitCode::InvalidInput);
}

#[test]
fn req39_empty_too_many_and_directory() {
    let d = TempDir::new("sp-misc");
    let e = check_split_files(vec![], u64::MAX).unwrap_err();
    assert!(matches!(e, SplitFilesRejection::Empty));
    assert_eq!(e.exit_code(), ExitCode::InvalidInput);

    d.write("a", b"x");
    let m: Vec<_> = (0..=MAX_SPLIT_FILES).map(|_| d.open("a")).collect();
    let e = check_split_files(m, u64::MAX).unwrap_err();
    assert!(matches!(
        e,
        SplitFilesRejection::TooMany {
            count: 257,
            limit: 256
        }
    ));
    assert_eq!(e.exit_code(), ExitCode::LimitExceeded);

    let dir = File::open(&d.0).unwrap();
    let (_, p) = d.open("a");
    let e = check_split_files(vec![(dir, p)], u64::MAX).unwrap_err();
    assert!(matches!(
        e,
        SplitFilesRejection::File {
            index: 0,
            source: FileSizeRejection::NotRegularFile
        }
    ));
    assert_eq!(e.exit_code(), ExitCode::InvalidInput);
}

#[test]
fn req39_effective_total_limit_clamps() {
    assert_eq!(effective_total_limit(u64::MAX), 8_589_934_592);
    assert_eq!(effective_total_limit(100), 100);
}

/// REQ-39: 読み進めたハンドルでも sha256 は先頭からの全体（末尾だけ・空のハッシュにならない）。
#[test]
fn req39_consumed_handle_hashed_from_start() {
    use std::io::Read;
    let d = TempDir::new("sp-rewind");
    d.write("a", b"aaaaaaaaaa");
    let (mut f, p) = d.open("a");
    let mut buf = [0u8; 6];
    f.read_exact(&mut buf).unwrap();
    let r = check_split_files(vec![(f, p)], u64::MAX).unwrap();
    assert_eq!(r.files()[0].sha256(), Sha256Digest::of_bytes(b"aaaaaaaaaa"));
}
