//! 容量計測コアの結合テスト（REQ-30・TASK-30.1-1・#122。テストハーネス）。

use fandhe_edge_runtime::capacity::{
    CapacityError, ComponentBytes, PackageComponent, PackageFile, measure_package,
};
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("fandhe-runtime-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
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
