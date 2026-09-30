//! `infer` の経路の閉じ込めの E2E（バイナリを起動。REQ-39・PoC-20 ケース 1・REQ-21・
//! TASK-39.4-2・#159。証拠種別: テストハーネス。合成ディレクトリ構成）。
//!
//! 構成: `<tmp>/workspace/`（カレントディレクトリ）と `<tmp>/outside/`。workspace の外を指す
//! `--package`・`onnx_file` は `invalid_input`（64）の JSON 1 行で拒否され、外のパスを
//! 出力へ含めないこと。正常対照として、内側の有効なパッケージはガードを通過して
//! スタブ（`runtime_error`・70。#136 で置換予定）に到達すること。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// 最小の ONNX 形（形式検査を通る）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
const MIN_ONNX: [u8; 9] = [0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78];

const ESCAPES: &str =
    "{\"code\":\"invalid_input\",\"message\":\"path rejected: path_escapes_root\"}\n";
#[cfg(any(target_os = "linux", target_os = "macos"))]
const STUB: &str =
    "{\"code\":\"runtime_error\",\"message\":\"stage not implemented yet (TASK-33.1-2)\"}\n";

struct Sandbox {
    base: PathBuf,
}

impl Sandbox {
    fn new(label: &str) -> Self {
        let base = std::env::temp_dir().join(format!(
            "fandhe-cli-confine-{}-{}",
            std::process::id(),
            label
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("workspace")).expect("mkdir workspace");
        fs::create_dir_all(base.join("outside/fake_pkg")).expect("mkdir outside");
        fs::write(base.join("outside/secret_marker.txt"), b"secret").expect("write");
        fs::write(base.join("outside/model.onnx"), b"onnx").expect("write");
        fs::write(
            base.join("outside/fake_pkg/artifact.json"),
            br#"{"onnx_file":"model.onnx","kind":"c3","kind_version":1}"#,
        )
        .expect("write");
        fs::write(base.join("outside/fake_pkg/model.onnx"), b"onnx").expect("write");
        Sandbox { base }
    }
    fn workspace(&self) -> PathBuf {
        self.base.join("workspace")
    }
    fn outside(&self) -> PathBuf {
        self.base.join("outside")
    }
    /// workspace 内にパッケージ `name` を作る。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn make_pkg(&self, name: &str, artifact_json: &str) -> PathBuf {
        let dir = self.workspace().join(name);
        fs::create_dir_all(&dir).expect("mkdir pkg");
        fs::write(dir.join("artifact.json"), artifact_json).expect("write");
        fs::write(dir.join("model.onnx"), MIN_ONNX).expect("write");
        dir
    }
    /// `fandhe-edge infer --package <package> --text a` を workspace で起動する。
    fn infer(&self, package: &Path) -> (Option<i32>, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
            .current_dir(self.workspace())
            .args(["infer", "--package"])
            .arg(package)
            .args(["--text", "a"])
            .stdin(Stdio::null())
            .output()
            .expect("run");
        (
            out.status.code(),
            String::from_utf8(out.stdout).expect("utf8"),
        )
    }
    fn assert_rejected_escapes(&self, package: &Path) {
        let (code, stdout) = self.infer(package);
        assert_eq!(code, Some(64), "stdout={stdout}");
        assert_eq!(stdout, ESCAPES);
        assert!(!stdout.contains("outside"));
        assert!(!stdout.contains(self.base.to_string_lossy().as_ref()));
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

/// REQ-39・PoC-20 ケース 1（b）: `--package ../outside/fake_pkg` は 64。
#[test]
fn req39_package_parent_traversal_is_rejected() {
    let sb = Sandbox::new("dotdot");
    sb.assert_rejected_escapes(Path::new("../outside/fake_pkg"));
}

/// REQ-39: 外への絶対パスの `--package` は 64。
#[test]
fn req39_package_absolute_outside_is_rejected() {
    let sb = Sandbox::new("abs");
    sb.assert_rejected_escapes(&sb.outside().join("fake_pkg"));
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod unix_only {
    use super::*;
    use std::os::unix::fs::symlink;

    /// REQ-39（正常対照）: 内側の有効なパッケージはガードを通過してスタブ（70）に到達する。
    #[test]
    fn req39_valid_inner_package_passes_guard_to_stub() {
        let sb = Sandbox::new("ok");
        sb.make_pkg(
            "pkg",
            r#"{"onnx_file":"model.onnx","kind":"c3","kind_version":1}"#,
        );
        let (code, stdout) = sb.infer(Path::new("pkg"));
        assert_eq!(code, Some(70));
        assert_eq!(stdout, STUB);
    }

    /// REQ-39・PoC-20 ケース 1（a）: `onnx_file` が外の秘密ファイルへ出る相対パス。
    #[test]
    fn req39_onnx_file_parent_traversal_is_rejected() {
        let sb = Sandbox::new("a");
        sb.make_pkg(
            "pkg",
            r#"{"onnx_file":"../../outside/secret_marker.txt","kind":"c3","kind_version":1}"#,
        );
        sb.assert_rejected_escapes(Path::new("pkg"));
    }

    /// REQ-39: `onnx_file` が外の実 ONNX ファイルへ出る相対パス。
    #[test]
    fn req39_onnx_file_to_outside_model_is_rejected() {
        let sb = Sandbox::new("a2");
        sb.make_pkg(
            "pkg",
            r#"{"onnx_file":"../../outside/model.onnx","kind":"c3","kind_version":1}"#,
        );
        sb.assert_rejected_escapes(Path::new("pkg"));
    }

    /// REQ-39: `onnx_file` が外への絶対パス。
    #[test]
    fn req39_onnx_file_absolute_outside_is_rejected() {
        let sb = Sandbox::new("a3");
        let abs = sb.outside().join("model.onnx");
        let json = format!(
            r#"{{"onnx_file":{},"kind":"c3","kind_version":1}}"#,
            serde_escape(&abs)
        );
        sb.make_pkg("pkg", &json);
        sb.assert_rejected_escapes(Path::new("pkg"));
    }

    /// JSON 文字列リテラルへ最低限エスケープする（テスト用。`"` と `\` のみ）。
    fn serde_escape(p: &Path) -> String {
        let s = p
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        format!("\"{s}\"")
    }

    /// REQ-39: パッケージ内の model.onnx 自体が外を指す symlink。
    #[test]
    fn req39_model_symlink_to_outside_is_rejected() {
        let sb = Sandbox::new("msym");
        let dir = sb.make_pkg(
            "pkg",
            r#"{"onnx_file":"model.onnx","kind":"c3","kind_version":1}"#,
        );
        fs::remove_file(dir.join("model.onnx")).expect("rm");
        symlink(sb.outside().join("model.onnx"), dir.join("model.onnx")).expect("symlink");
        sb.assert_rejected_escapes(Path::new("pkg"));
    }

    /// REQ-39・PoC-20 ケース 1（c）: workspace 内の symlink が外のディレクトリを指す
    /// （その先に artifact.json が無くても、外への参照として拒否する）。
    #[test]
    fn req39_package_symlink_to_outside_dir_is_rejected() {
        let sb = Sandbox::new("c");
        symlink(sb.outside(), sb.workspace().join("link")).expect("symlink");
        sb.assert_rejected_escapes(Path::new("link"));
    }

    /// REQ-39: workspace 内の symlink が外の有効なパッケージを指す（c'）。
    #[test]
    fn req39_package_symlink_to_outside_valid_package_is_rejected() {
        let sb = Sandbox::new("c2");
        symlink(sb.outside().join("fake_pkg"), sb.workspace().join("link")).expect("symlink");
        sb.assert_rejected_escapes(Path::new("link"));
    }

    /// REQ-39・REQ-21: artifact.json が無ければ path_unresolvable（64）。
    #[test]
    fn req39_missing_artifact_json_is_invalid_input() {
        let sb = Sandbox::new("noart");
        fs::create_dir_all(sb.workspace().join("pkg")).expect("mkdir");
        let (code, stdout) = sb.infer(Path::new("pkg"));
        assert_eq!(code, Some(64));
        assert_eq!(
            stdout,
            "{\"code\":\"invalid_input\",\"message\":\"path rejected: path_unresolvable\"}\n"
        );
    }

    /// REQ-39: 不正な JSON は 64（artifact metadata is invalid）。
    #[test]
    fn req39_malformed_artifact_json_is_invalid_input() {
        let sb = Sandbox::new("bad");
        sb.make_pkg("pkg", "not json");
        let (code, stdout) = sb.infer(Path::new("pkg"));
        assert_eq!(code, Some(64));
        assert_eq!(
            stdout,
            "{\"code\":\"invalid_input\",\"message\":\"artifact metadata is invalid\"}\n"
        );
    }

    /// REQ-39: `--package` が通常ファイルなら not_directory（64）。
    #[test]
    fn req39_package_that_is_a_file_is_invalid_input() {
        let sb = Sandbox::new("file");
        fs::write(sb.workspace().join("afile"), b"x").expect("write");
        let (code, stdout) = sb.infer(Path::new("afile"));
        assert_eq!(code, Some(64));
        assert_eq!(
            stdout,
            "{\"code\":\"invalid_input\",\"message\":\"path rejected: not_directory\"}\n"
        );
    }

    /// REQ-39（形式の許可制）: pickle 偽装（拡張子は .onnx だが中身は pickle）は invalid_input（64）。
    #[test]
    fn req39_pickle_disguised_onnx_is_rejected() {
        let sb = Sandbox::new("pickle");
        let dir = sb.make_pkg(
            "pkg",
            r#"{"onnx_file":"model.onnx","kind":"c3","kind_version":1}"#,
        );
        fs::write(dir.join("model.onnx"), b"\x80\x04\x95\x00\x00\x00.").expect("write");
        let (code, stdout) = sb.infer(Path::new("pkg"));
        assert_eq!(code, Some(64));
        assert_eq!(
            stdout,
            "{\"code\":\"invalid_input\",\"message\":\"format rejected: format_not_allowed\"}\n"
        );
    }

    /// REQ-39: 1 MiB を超える artifact.json は limit_exceeded（20）。
    #[test]
    fn req39_oversized_artifact_json_is_limit_exceeded() {
        let sb = Sandbox::new("big");
        let dir = sb.make_pkg("pkg", "{}");
        fs::write(dir.join("artifact.json"), vec![b' '; 1024 * 1024 + 1]).expect("write");
        let (code, stdout) = sb.infer(Path::new("pkg"));
        assert_eq!(code, Some(20));
        assert_eq!(
            stdout,
            "{\"code\":\"limit_exceeded\",\"message\":\"artifact metadata exceeds size limit\"}\n"
        );
    }

    /// REQ-39（資源の上限）: 上限を超える ONNX ファイルは読み込み前に limit_exceeded（20）。
    /// sparse ファイル（`set_len`）で実データを書かずに 44 MiB 超を作る。
    #[test]
    fn req39_oversized_onnx_is_limit_exceeded() {
        let sb = Sandbox::new("bigonnx");
        let dir = sb.make_pkg(
            "pkg",
            r#"{"onnx_file":"model.onnx","kind":"c3","kind_version":1}"#,
        );
        let f = fs::OpenOptions::new()
            .write(true)
            .open(dir.join("model.onnx"))
            .expect("open");
        f.set_len(44 * 1024 * 1024 + 1).expect("set_len");
        let (code, stdout) = sb.infer(Path::new("pkg"));
        assert_eq!(code, Some(20));
        assert_eq!(
            stdout,
            "{\"code\":\"limit_exceeded\",\"message\":\"model file exceeds size limit\"}\n"
        );
    }
}
