//! パッケージ単位の閉じ込めの結合テスト（REQ-39・PoC-20 ケース 1・TASK-39.4-2・#159。
//! 証拠種別: テストハーネス。合成ディレクトリ構成）。
//!
//! 構成: `<tmp>/workspace/pkg/{artifact.json, model.onnx}` と `<tmp>/outside/...`。
//! `open_confined` は Linux・macOS 限定のため、ファイル全体をそれらに限る。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_guard::package::confine_package;
use fandhe_edge_guard::path::PathRejection;

struct Sandbox {
    base: PathBuf,
}

impl Sandbox {
    fn new(label: &str) -> Self {
        let base =
            std::env::temp_dir().join(format!("fandhe-guard-pkg-{}-{}", std::process::id(), label));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("workspace/pkg")).expect("mkdir pkg");
        fs::create_dir_all(base.join("outside/fake_pkg")).expect("mkdir outside");
        fs::write(base.join("workspace/pkg/artifact.json"), b"{}").expect("write");
        fs::write(base.join("workspace/pkg/model.onnx"), b"onnx").expect("write");
        fs::write(base.join("outside/model.onnx"), b"onnx").expect("write");
        fs::write(base.join("outside/fake_pkg/artifact.json"), b"{}").expect("write");
        Sandbox { base }
    }
    fn workspace(&self) -> PathBuf {
        self.base.join("workspace")
    }
    fn outside(&self) -> PathBuf {
        self.base.join("outside")
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

fn reason<T: std::fmt::Debug>(r: Result<T, PathRejection>) -> (&'static str, ExitCode) {
    let e = r.expect_err("must be rejected");
    (e.reason_code(), e.exit_code())
}

/// REQ-39: workspace 内のパッケージとメンバーは通る。
#[test]
fn req39_inner_package_and_member_are_accepted() {
    let sb = Sandbox::new("ok");
    let pkg = confine_package(&sb.workspace(), Path::new("pkg")).expect("confine");
    let (_f, real) = pkg.open_member(Path::new("model.onnx")).expect("open");
    assert!(real.as_path().ends_with("pkg/model.onnx"));
}

/// REQ-39: パッケージが通常ファイルなら not_directory（64）。
#[test]
fn req39_package_that_is_a_file_is_not_directory() {
    let sb = Sandbox::new("notdir");
    let r = confine_package(&sb.workspace(), Path::new("pkg/model.onnx"));
    assert_eq!(reason(r), ("not_directory", ExitCode::InvalidInput));
}

/// REQ-39: `../outside` と外への絶対パスと外への symlink は path_escapes_root。
#[test]
fn req39_package_outside_workspace_is_rejected() {
    let sb = Sandbox::new("pkgout");
    let r = confine_package(&sb.workspace(), Path::new("../outside/fake_pkg"));
    assert_eq!(reason(r), ("path_escapes_root", ExitCode::InvalidInput));
    let r = confine_package(&sb.workspace(), &sb.outside().join("fake_pkg"));
    assert_eq!(reason(r), ("path_escapes_root", ExitCode::InvalidInput));
    symlink(sb.outside().join("fake_pkg"), sb.workspace().join("link")).expect("symlink");
    let r = confine_package(&sb.workspace(), Path::new("link"));
    assert_eq!(reason(r), ("path_escapes_root", ExitCode::InvalidInput));
}

/// REQ-39: onnx_file がパッケージの外（workspace の外）を指す形は拒否する。
#[test]
fn req39_member_escaping_package_is_rejected() {
    let sb = Sandbox::new("member");
    let pkg = confine_package(&sb.workspace(), Path::new("pkg")).expect("confine");
    let r = pkg.open_member(Path::new("../../outside/model.onnx"));
    assert_eq!(reason(r), ("path_escapes_root", ExitCode::InvalidInput));
    let r = pkg.open_member(&sb.outside().join("model.onnx"));
    assert_eq!(reason(r), ("path_escapes_root", ExitCode::InvalidInput));
    symlink(
        sb.outside().join("model.onnx"),
        sb.workspace().join("pkg/evil.onnx"),
    )
    .expect("symlink");
    let r = pkg.open_member(Path::new("evil.onnx"));
    assert_eq!(reason(r), ("path_escapes_root", ExitCode::InvalidInput));
}

/// REQ-39: confine 後にパッケージディレクトリを外を指す symlink へ差し替えても、
/// 保持した fd 起点の open_member は差し替え先を読まず拒否する（TOCTOU）。削除済みのディレクトリ
/// fd は実パスが解決できないため fail-closed（拒否の種別は OS 依存のため問わない）。
#[test]
fn req39_package_dir_swapped_to_outside_symlink_is_rejected() {
    let sb = Sandbox::new("swap");
    fs::write(sb.outside().join("fake_pkg/model.onnx"), b"onnx").expect("write");
    let pkg = confine_package(&sb.workspace(), Path::new("pkg")).expect("confine");
    fs::remove_dir_all(sb.workspace().join("pkg")).expect("rm pkg");
    symlink(sb.outside().join("fake_pkg"), sb.workspace().join("pkg")).expect("swap");
    assert!(pkg.open_member(Path::new("model.onnx")).is_err());
}
