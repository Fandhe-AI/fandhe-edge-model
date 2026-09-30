//! 新方式 C1・C3 の生成物での容量内訳の結合テスト（REQ-30・TASK-30.1・#121）。
//!
//! 証拠種別: テストハーネス（生成物）。`fixtures/onnx_parity/` の ONNX は学習ワーカーが極小設定・
//! 合成データから生成したもので、Mac 実機で実データを学習した成果物の計測ではない
//! （実機計測は人の担当。AGENTS.md 参照）。fixture の出典は `fixtures/onnx_parity/PROVENANCE.md`。
//! 生成器 `trainer/tools/gen_onnx_parity_fixture.py` を再実行した場合は下の期待値を更新する。
//!
//! C1 は語彙・特徴量変換を ONNX グラフ内に持つ新方式のため、語彙は別ファイルとして数えない。
//! 旧方式の実測値 2.57MB は引き継がず、実ファイルのサイズから測り直した値だけを検証する。

use fandhe_edge_runtime::capacity::{
    CapacityBreakdown, PackageComponent, PackageFile, measure_package,
};
use std::path::{Path, PathBuf};

const C1_ONNX_BYTES: u64 = 8764;
const C3_ONNX_BYTES: u64 = 13952;
const DEFINITION_JSON: &[u8] = b"{\"labels\":[\"a\",\"b\"]}";
const ARTIFACT_JSON: &[u8] = b"{\"kind\":\"c1\"}";

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        // 自分が作ったディレクトリだけを後片付けする（capacity.rs と同じ方式）。
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

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/onnx_parity")
        .join(name)
}

/// `package` 工程と同じ分類（ONNX→重み・定義→選択肢表・artifact→メタデータ）で計測する。
///
/// 3 OS 共通で成立する: 入口は `measure_package`（パスから開く版）で、同一ファイルの重複判定に
/// Unix は (dev, inode)、Unix 以外は `canonicalize` した実パスを使うため、複数ファイルでも
/// `IdentityUnavailable` にならない。`IdentityUnavailable` を返すのはハンドルを受け取る
/// `measure_opened_files`（Unix 以外で 2 件以上）だけで、本テストは経由しない（REQ-30・REQ-39）。
fn measure(tag: &str, onnx_name: &str) -> (CapacityBreakdown, u64) {
    let d = TempDir::new(tag);
    let onnx = d.path().join("model.onnx");
    std::fs::copy(fixture(onnx_name), &onnx).unwrap();
    let onnx_len = std::fs::metadata(&onnx).unwrap().len();
    let def = d.path().join("definition.json");
    std::fs::write(&def, DEFINITION_JSON).unwrap();
    let art = d.path().join("artifact.json");
    std::fs::write(&art, ARTIFACT_JSON).unwrap();
    let files = [
        PackageFile {
            component: PackageComponent::Weights,
            path: onnx,
        },
        PackageFile {
            component: PackageComponent::LabelTable,
            path: def,
        },
        PackageFile {
            component: PackageComponent::Metadata,
            path: art,
        },
    ];
    (measure_package(&files).unwrap(), onnx_len)
}

fn assert_breakdown(b: &CapacityBreakdown, onnx_len: u64, expected_onnx: u64) {
    assert_eq!(
        onnx_len, expected_onnx,
        "fixture ONNX size changed; update the expected value when regenerating fixtures"
    );
    let w = b.component(PackageComponent::Weights);
    assert_eq!((w.bytes, w.file_count), (expected_onnx, 1));
    let v = b.component(PackageComponent::VocabOrFeatureTransform);
    assert_eq!((v.bytes, v.file_count), (0, 0));
    let l = b.component(PackageComponent::LabelTable);
    assert_eq!((l.bytes, l.file_count), (DEFINITION_JSON.len() as u64, 1));
    let c = b.component(PackageComponent::Calibration);
    assert_eq!((c.bytes, c.file_count), (0, 0));
    let m = b.component(PackageComponent::Metadata);
    assert_eq!((m.bytes, m.file_count), (ARTIFACT_JSON.len() as u64, 1));
    let sum: u64 = b.entries().iter().map(|(_, e)| e.bytes).sum();
    assert_eq!(b.total_bytes(), sum);
    assert_eq!(
        b.total_bytes(),
        expected_onnx + DEFINITION_JSON.len() as u64 + ARTIFACT_JSON.len() as u64
    );
}

#[test]
fn req30_c1_new_scheme_vocab_inside_onnx_breakdown() {
    let (b, len) = measure("c1", "c1.onnx");
    assert_breakdown(&b, len, C1_ONNX_BYTES);
    // 旧方式の実測値（約 2.57MB）を引き継いでいない。
    assert!(b.total_bytes() < 100_000);
}

#[test]
fn req30_c3_breakdown() {
    let (b, len) = measure("c3", "c3.onnx");
    assert_breakdown(&b, len, C3_ONNX_BYTES);
}
