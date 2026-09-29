//! 経路の閉じ込めの結合テスト（REQ-39・PoC-20 ケース 1・TASK-39.4-1・#158。
//! 証拠種別: テストハーネス。合成ディレクトリ構成）。
//!
//! 構成: `<tmp>/workspace/pkg/` と `<tmp>/outside/{secret_marker.txt, model.onnx, fake_pkg/}`。

use std::fs;
use std::path::{Path, PathBuf};

use fandhe_edge_core::exitcode::ExitCode;
#[cfg(unix)]
use fandhe_edge_guard::path::open_confined;
use fandhe_edge_guard::path::{EscapeKind, PathRejection, safe_join};

/// 一時ディレクトリ（Drop で削除）。外部 crate を使わない。
struct Sandbox {
    base: PathBuf,
}

impl Sandbox {
    fn new(label: &str) -> Self {
        let base = std::env::temp_dir().join(format!(
            "fandhe-guard-path-{}-{}",
            std::process::id(),
            label
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("workspace/pkg")).expect("mkdir workspace/pkg");
        fs::create_dir_all(base.join("outside/fake_pkg")).expect("mkdir outside/fake_pkg");
        fs::write(base.join("outside/secret_marker.txt"), b"secret").expect("write marker");
        fs::write(base.join("outside/model.onnx"), b"onnx").expect("write model");
        fs::write(base.join("workspace/pkg/model.onnx"), b"onnx").expect("write inner model");
        Sandbox { base }
    }
    fn workspace(&self) -> PathBuf {
        self.base.join("workspace")
    }
    fn pkg(&self) -> PathBuf {
        self.base.join("workspace/pkg")
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

fn assert_escapes(result: Result<impl std::fmt::Debug, PathRejection>, expected: EscapeKind) {
    match result {
        Err(e) => {
            assert_eq!(e.exit_code(), ExitCode::InvalidInput);
            assert_eq!(e.reason_code(), "path_escapes_root");
            match e {
                PathRejection::Escapes { kind, .. } => assert_eq!(kind, expected),
                other => panic!("expected Escapes, got {other:?}"),
            }
        }
        Ok(v) => panic!("expected Escapes, got Ok({v:?})"),
    }
}

#[cfg(unix)]
fn symlink(target: &Path, link: &Path) {
    std::os::unix::fs::symlink(target, link).expect("symlink");
}

/// PoC-20 ケース 1 (a): 外向き相対パスで workspace 外のファイルを指す。
#[test]
fn req39_parent_traversal_to_outside_file_is_rejected() {
    let sb = Sandbox::new("a");
    assert_escapes(
        safe_join(&sb.pkg(), Path::new("../../outside/secret_marker.txt")),
        EscapeKind::ParentTraversal,
    );
}

/// PoC-20 ケース 1 (a'): 外向き相対パスで workspace 外の ONNX を指す。
#[test]
fn req39_parent_traversal_to_outside_onnx_is_rejected() {
    let sb = Sandbox::new("a2");
    assert_escapes(
        safe_join(&sb.pkg(), Path::new("../../outside/model.onnx")),
        EscapeKind::ParentTraversal,
    );
}

/// PoC-20 ケース 1 (b): `--package` に外向き相対パスで外部ディレクトリを指す。
#[test]
fn req39_parent_traversal_to_outside_package_is_rejected() {
    let sb = Sandbox::new("b");
    assert_escapes(
        safe_join(&sb.workspace(), Path::new("../outside/fake_pkg")),
        EscapeKind::ParentTraversal,
    );
}

/// PoC-20 ケース 1 (c): workspace 内の symlink ディレクトリ経由で外部ファイルを指す。
#[cfg(unix)]
#[test]
fn req39_symlink_dir_to_outside_is_rejected() {
    let sb = Sandbox::new("c");
    symlink(&sb.outside(), &sb.workspace().join("link_dir"));
    assert_escapes(
        safe_join(&sb.workspace(), Path::new("link_dir/secret_marker.txt")),
        EscapeKind::Symlink,
    );
}

/// PoC-20 ケース 1 (c'): `--package` 自体が外部ディレクトリへの symlink。
#[cfg(unix)]
#[test]
fn req39_symlink_package_to_outside_is_rejected() {
    let sb = Sandbox::new("c2");
    symlink(
        &sb.outside().join("fake_pkg"),
        &sb.workspace().join("link_pkg"),
    );
    assert_escapes(
        safe_join(&sb.workspace(), Path::new("link_pkg")),
        EscapeKind::Symlink,
    );
}

#[test]
fn req39_inside_paths_are_accepted_and_canonical() {
    let sb = Sandbox::new("ok");
    let canon_pkg = fs::canonicalize(sb.pkg()).expect("canon");
    let expected = canon_pkg.join("model.onnx");

    let got = safe_join(&sb.workspace(), Path::new("pkg/model.onnx")).expect("plain");
    assert_eq!(got.as_path(), expected.as_path());

    let got = safe_join(&sb.workspace(), Path::new("pkg/../pkg/model.onnx")).expect("dotdot");
    assert_eq!(got.into_path_buf(), expected);

    let got = safe_join(&sb.pkg(), Path::new(".")).expect("root itself");
    assert_eq!(got.as_path(), canon_pkg.as_path());

    let abs = sb.pkg().join("model.onnx");
    let got = safe_join(&sb.workspace(), &abs).expect("absolute inside");
    assert_eq!(got.as_path(), expected.as_path());
}

#[cfg(unix)]
#[test]
fn req39_symlink_inside_workspace_is_accepted() {
    let sb = Sandbox::new("inlink");
    symlink(&sb.pkg(), &sb.workspace().join("alias"));
    let got = safe_join(&sb.workspace(), Path::new("alias/model.onnx")).expect("inner symlink");
    let expected = fs::canonicalize(sb.pkg().join("model.onnx")).expect("canon");
    assert_eq!(got.as_path(), expected.as_path());
}

#[test]
fn req39_absolute_path_outside_root_is_rejected() {
    let sb = Sandbox::new("abs");
    assert_escapes(
        safe_join(&sb.workspace(), &sb.outside().join("model.onnx")),
        EscapeKind::Absolute,
    );
}

/// 兄弟の接頭辞ディレクトリ（`workspace-evil`）を文字列前方一致で誤許可しない。
#[test]
fn req39_sibling_prefix_directory_is_rejected() {
    let sb = Sandbox::new("prefix");
    let evil = sb.base.join("workspace-evil");
    fs::create_dir_all(&evil).expect("mkdir evil");
    fs::write(evil.join("m.onnx"), b"x").expect("write");
    assert_escapes(
        safe_join(&sb.workspace(), &evil.join("m.onnx")),
        EscapeKind::Absolute,
    );
}

#[test]
fn req39_unresolvable_and_invalid_roots_are_rejected() {
    let sb = Sandbox::new("bad");
    let e = safe_join(&sb.workspace(), Path::new("pkg/missing.onnx")).unwrap_err();
    assert_eq!(e.reason_code(), "path_unresolvable");
    assert_eq!(e.exit_code(), ExitCode::InvalidInput);

    let e = safe_join(&sb.base.join("no_such_root"), Path::new("x")).unwrap_err();
    assert_eq!(e.reason_code(), "root_unresolvable");
    assert_eq!(e.exit_code(), ExitCode::InvalidInput);

    let e = safe_join(&sb.pkg().join("model.onnx"), Path::new(".")).unwrap_err();
    assert_eq!(e.reason_code(), "root_not_directory");
    assert_eq!(e.exit_code(), ExitCode::InvalidInput);

    let e = safe_join(&sb.workspace(), Path::new("")).unwrap_err();
    assert_eq!(e.reason_code(), "empty_path");
    assert_eq!(e.exit_code(), ExitCode::InvalidInput);
}

/// 字句的にルート配下の絶対パスが symlink で外へ出る場合は Absolute でなく Symlink（REQ-39）。
#[cfg(unix)]
#[test]
fn req39_absolute_under_root_via_symlink_is_symlink_kind() {
    let sb = Sandbox::new("abs_link");
    symlink(&sb.outside(), &sb.workspace().join("link_dir"));
    let candidate = sb.workspace().join("link_dir").join("secret_marker.txt");
    assert_escapes(safe_join(&sb.workspace(), &candidate), EscapeKind::Symlink);
}

/// 検証と open を一体で行う経路（TOCTOU 対策。REQ-39）。ルート配下のファイルは開ける。
#[cfg(unix)]
#[test]
fn req39_open_confined_opens_inside_file() {
    let sb = Sandbox::new("open_ok");
    let (mut f, got) =
        open_confined(&sb.workspace(), Path::new("pkg/model.onnx")).expect("open inside");
    let expected = fs::canonicalize(sb.pkg().join("model.onnx")).expect("canon");
    assert_eq!(got.as_path(), expected.as_path());
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut f, &mut buf).expect("read");
    assert_eq!(buf, fs::read(sb.pkg().join("model.onnx")).expect("read"));
}

/// 外部を指す入力は open せず拒否する（REQ-39）。
#[cfg(unix)]
#[test]
fn req39_open_confined_rejects_outside() {
    let sb = Sandbox::new("open_out");
    symlink(&sb.outside(), &sb.workspace().join("link_dir"));
    assert_escapes(
        open_confined(&sb.workspace(), Path::new("link_dir/secret_marker.txt")),
        EscapeKind::Symlink,
    );
    assert_escapes(
        open_confined(&sb.workspace(), Path::new("../outside/secret_marker.txt")),
        EscapeKind::ParentTraversal,
    );
}

/// ディレクトリ・FIFO は通常ファイルではないため open せず拒否し、FIFO でも停止しない（REQ-39）。
#[cfg(unix)]
#[test]
fn req39_open_confined_rejects_non_regular_files_without_hanging() {
    let sb = Sandbox::new("open_nonreg");
    let fifo = sb.workspace().join("pipe");
    let status = std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .expect("mkfifo");
    assert!(status.success());
    for name in ["pipe", "pkg"] {
        match open_confined(&sb.workspace(), Path::new(name)) {
            Err(e @ PathRejection::NotRegularFile { .. }) => {
                assert_eq!(e.reason_code(), "not_regular_file");
                assert_eq!(e.exit_code(), ExitCode::InvalidInput);
            }
            other => panic!("expected NotRegularFile for {name}, got {other:?}"),
        }
    }
}

/// ルートが FIFO でも `open_confined` は停止せず `RootNotDirectory` を返す（REQ-39）。
#[cfg(target_os = "linux")]
#[test]
fn req39_open_confined_root_fifo_does_not_hang() {
    let sb = Sandbox::new("root_fifo");
    let fifo = sb.base.join("rootfifo");
    let status = std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .expect("mkfifo");
    assert!(status.success());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let r = open_confined(&fifo, Path::new("x")).map(|_| ());
        let _ = tx.send(r);
    });
    match rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(Err(PathRejection::RootNotDirectory)) => {}
        other => panic!("expected RootNotDirectory without hanging, got {other:?}"),
    }
}
