//! 容量計測から JSON 出力までの結合テスト（REQ-30・TASK-30.1-2・#123。
//! 証拠種別: テストハーネス（生成物））。

use fandhe_edge_cli::output::{capacity_error_report, write_package_capacity};
use fandhe_edge_runtime::capacity::{PackageComponent, PackageFile, measure_package};
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
                "fandhe-cli-{tag}-{}-{nanos}-{n}",
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

fn make(dir: &Path, name: &str, len: usize, component: PackageComponent) -> PackageFile {
    let path = dir.join(name);
    std::fs::write(&path, vec![0u8; len]).unwrap();
    PackageFile { component, path }
}

#[test]
fn req30_measured_package_is_written_as_capacity_json() {
    let d = TempDir::new("ok");
    let files = [
        make(d.path(), "model.onnx", 1000, PackageComponent::Weights),
        make(d.path(), "labels.json", 37, PackageComponent::LabelTable),
        make(d.path(), "artifact.json", 512, PackageComponent::Metadata),
    ];
    let b = measure_package(&files).unwrap();
    let mut buf = Vec::new();
    let code = write_package_capacity(&mut buf, &b).unwrap();
    assert_eq!(code.code(), 0);
    let expected = concat!(
        "{\"capacity\":{\"total_bytes\":1549,\"components\":{",
        "\"weights\":{\"bytes\":1000,\"file_count\":1},",
        "\"vocab_or_feature_transform\":{\"bytes\":0,\"file_count\":0},",
        "\"label_table\":{\"bytes\":37,\"file_count\":1},",
        "\"calibration\":{\"bytes\":0,\"file_count\":0},",
        "\"metadata\":{\"bytes\":512,\"file_count\":1}}}}\n"
    );
    assert_eq!(String::from_utf8(buf).unwrap(), expected);
}

#[test]
fn req30_duplicate_file_maps_to_invalid_input_without_path() {
    let d = TempDir::new("dup");
    let f = make(d.path(), "model.onnx", 10, PackageComponent::Weights);
    let err = measure_package(&[f.clone(), f]).unwrap_err();
    let report = capacity_error_report(&err);
    assert_eq!(report.code.code(), 64);
    let dir = d.path().to_string_lossy().into_owned();
    assert!(!report.message.contains(&dir), "{}", report.message);
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
