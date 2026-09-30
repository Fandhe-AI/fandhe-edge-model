//! 実ファイルの容量計測から語彙超過構成の除外までの結合テスト（REQ-30・TASK-30.3・#125）。
//!
//! 証拠種別: テストハーネス（スパースファイルによる生成物）。PoC-13 の実機実測値
//! （Qwen2.5 語彙流用 46,365,993 バイト）の再計測ではない。

use fandhe_edge_runtime::capacity::{PackageComponent, PackageFile, measure_package};
use fandhe_edge_runtime::vocab_exclusion::{VocabDecision, screen_vocab_candidates};
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        // 自分が作ったディレクトリだけを後片付けする（capacity_generated_packages.rs と同じ方式）。
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        for _ in 0..100 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!(
                "fandhe-runtime-vocab-{}-{nanos}-{n}",
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
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sparse(dir: &Path, name: &str, len: u64) -> PathBuf {
    let p = dir.join(name);
    let f = std::fs::File::create(&p).expect("create");
    f.set_len(len).expect("set_len");
    p
}

fn pf(component: PackageComponent, path: PathBuf) -> PackageFile {
    PackageFile { component, path }
}

#[test]
fn req30_measured_qwen_like_package_is_excluded() {
    let dir = TempDir::new();
    let files = [
        pf(
            PackageComponent::Weights,
            sparse(&dir.0, "w.bin", 39_333_412),
        ),
        pf(
            PackageComponent::VocabOrFeatureTransform,
            sparse(&dir.0, "v.json", 7_031_673),
        ),
        pf(PackageComponent::LabelTable, sparse(&dir.0, "l.json", 544)),
        pf(PackageComponent::Metadata, sparse(&dir.0, "m.json", 364)),
    ];
    let b = measure_package(&files).expect("measure");
    assert_eq!(b.total_bytes(), 46_365_993);
    let s = screen_vocab_candidates(&[("qwen", b, true)]).expect("screen");
    assert!(s.eligible().is_empty());
    assert_eq!(s.excluded().len(), 1);
}

#[test]
fn req30_onnx_parity_c1_c3_have_no_vocab_and_stay_eligible() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/onnx_parity");
    let mut cands = Vec::new();
    for name in ["c1", "c3"] {
        let files = [pf(
            PackageComponent::Weights,
            base.join(format!("{name}.onnx")),
        )];
        let b = measure_package(&files).expect("measure");
        assert_eq!(
            b.component(PackageComponent::VocabOrFeatureTransform)
                .file_count,
            0
        );
        cands.push((name, b, false));
    }
    let s = screen_vocab_candidates(&cands).expect("screen");
    assert_eq!(s.eligible(), vec!["c1", "c3"]);
    assert!(
        s.records
            .iter()
            .all(|r| r.decision == VocabDecision::Eligible)
    );
}
