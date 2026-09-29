//! 容量計測コアの結合テスト（REQ-30・TASK-30.1-1・#122。テストハーネス）。

use fandhe_edge_runtime::capacity::{
    CapacityError, ComponentBytes, PackageComponent, PackageFile, measure_package,
};
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        // 予測可能なパスを削除しない。create_dir は既存パスで失敗するため、
        // 衝突時は名前を変えて再試行し、自分が作ったものだけを後片付けする。
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        for _ in 0..100 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!(
                "fandhe-runtime-{tag}-{}-{nanos}-{n}",
                std::process::id()
            ));
            match std::fs::create_dir(&p) {
                Ok(()) => return Self(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create temp dir: {e}"),
            }
        }
        panic!("could not create a unique temp dir");
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write(dir: &Path, name: &str, len: usize) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, vec![0u8; len]).unwrap();
    p
}

fn pf(component: PackageComponent, path: PathBuf) -> PackageFile {
    PackageFile { component, path }
}

#[test]
fn req30_measure_package_breakdown_and_total() {
    let d = TempDir::new("ok");
    let files = [
        pf(
            PackageComponent::Weights,
            write(d.path(), "model.onnx", 1000),
        ),
        pf(
            PackageComponent::LabelTable,
            write(d.path(), "labels.json", 37),
        ),
        pf(
            PackageComponent::Metadata,
            write(d.path(), "meta.json", 300),
        ),
        pf(
            PackageComponent::Metadata,
            write(d.path(), "meta2.json", 212),
        ),
    ];
    let b = measure_package(&files).unwrap();
    assert_eq!(b.component(PackageComponent::Weights).bytes, 1000);
    assert_eq!(b.component(PackageComponent::LabelTable).bytes, 37);
    assert_eq!(
        b.component(PackageComponent::Metadata),
        ComponentBytes {
            bytes: 512,
            file_count: 2
        }
    );
    assert_eq!(b.component(PackageComponent::Calibration).file_count, 0);
    assert_eq!(b.total_bytes(), 1549);
}

#[test]
fn req30_directory_and_missing_are_rejected() {
    let d = TempDir::new("dir");
    let r = measure_package(&[pf(PackageComponent::Weights, d.path().to_path_buf())]);
    assert!(matches!(r, Err(CapacityError::File(_))));
    let r = measure_package(&[pf(PackageComponent::Weights, d.path().join("none"))]);
    assert!(matches!(r, Err(CapacityError::File(_))));
}

#[test]
fn req30_duplicate_path_and_empty_are_rejected() {
    let d = TempDir::new("dup");
    let p = write(d.path(), "a", 1);
    let r = measure_package(&[
        pf(PackageComponent::Weights, p.clone()),
        pf(PackageComponent::Metadata, p),
    ]);
    assert!(matches!(r, Err(CapacityError::DuplicatePath { .. })));
    assert!(matches!(
        measure_package(&[]),
        Err(CapacityError::EmptyPackage)
    ));
}

/// 別表記（`./a`・`sub/../a`）の同一ファイルも二重計上せず DuplicatePath にする。
#[test]
fn req30_alias_spelling_of_same_file_is_duplicate() {
    let d = TempDir::new("alias");
    let p = write(d.path(), "a", 1);
    std::fs::create_dir(d.path().join("sub")).unwrap();
    let alias = d.path().join("sub").join("..").join("a");
    let r = measure_package(&[
        pf(PackageComponent::Weights, p),
        pf(PackageComponent::Metadata, alias),
    ]);
    assert!(matches!(r, Err(CapacityError::DuplicatePath { .. })));
}

/// ハードリンクの別名も開いたハンドルの同一性で検出し、二重計上しない（REQ-30・REQ-39）。
#[cfg(unix)]
#[test]
fn req30_hardlink_alias_is_duplicate() {
    let d = TempDir::new("hardlink");
    let p = write(d.path(), "a", 3);
    let alias = d.path().join("b");
    std::fs::hard_link(&p, &alias).unwrap();
    let r = measure_package(&[
        pf(PackageComponent::Weights, p),
        pf(PackageComponent::Metadata, alias),
    ]);
    assert!(matches!(r, Err(CapacityError::DuplicatePath { .. })));
}

#[cfg(unix)]
#[test]
fn req30_symlink_is_rejected() {
    let d = TempDir::new("sym");
    let target = write(d.path(), "real", 5);
    let link = d.path().join("link");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let r = measure_package(&[pf(PackageComponent::Weights, link)]);
    assert!(matches!(r, Err(CapacityError::SymlinkRejected { .. })));
}

#[cfg(target_os = "linux")]
#[test]
fn req30_fifo_does_not_block() {
    let d = TempDir::new("fifo");
    let fifo = d.path().join("fifo");
    let status = std::process::Command::new("mkfifo").arg(&fifo).status();
    if !matches!(status, Ok(s) if s.success()) {
        return;
    }
    let r = measure_package(&[pf(PackageComponent::Weights, fifo)]);
    assert!(matches!(
        r,
        Err(CapacityError::File(
            fandhe_edge_core::fs::FsError::NotRegularFile { .. }
        ))
    ));
}

/// 一時ディレクトリは既存パスを消さずに一意に作る（レビュー指摘 P0・REQ-39 の完全性）。
#[test]
fn temp_dir_is_unique_and_never_removes_existing_content() {
    let a = TempDir::new("uniq");
    let b = TempDir::new("uniq");
    assert_ne!(a.path(), b.path());
    let marker = a.path().join("keep.txt");
    std::fs::write(&marker, b"x").unwrap();
    let _c = TempDir::new("uniq");
    assert_eq!(std::fs::read(&marker).unwrap(), b"x");
}
