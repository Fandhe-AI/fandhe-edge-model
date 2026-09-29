//! REQ-32 の境界値: Python・MLX が PATH に無い `env -i` 相当のプロセスでも、推論ランタイム層の
//! 推論が成功することを確認する（TASK-32.3・#115）。
//!
//! 方式: テスト実行ファイル自身を、環境変数を全消去（`env_clear`）した子プロセスとして再起動し、
//! 子で C1・C3 の fixture（`fixtures/onnx_parity/`）に対する単体・バッチ推論を実行して exit 0 を確かめる。
//!
//! 証拠種別: テストハーネス（CI の rust-ci では Linux・macOS runner で実行される。Mac 実機の
//! `otool -L` による動的リンク確認は `scripts/check-runtime-linkage.sh` を人間が実行する）。
//!
//! 対象外: CLI `fandhe-edge infer` の経路は工程の下位層への接続（TASK-33.1-2・#136）が未完のため
//! 含まない。#136 完了後に CLI バイナリ版を追加する（実装済みを装わない）。
//!
//! `#![cfg(unix)]` の理由: `env -i` は POSIX の概念で、検証環境は Mac。失敗メッセージには
//! ケース名・件数のみを出し、入力本文と環境変数の値は出さない（security.md）。

#![cfg(unix)]

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_runtime::onnx::{ModelKind, load_pipeline};
use serde_json::Value;
use std::ffi::OsStr;
use std::fs;
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// `cases.json` の読み込み上限（読み込み前に確認。REQ-39）。
const MAX_CASES_BYTES: u64 = 4 * 1024 * 1024;
/// 子プロセスの実時間上限（REQ-39）。
const CHILD_TIMEOUT: Duration = Duration::from_secs(60);
/// 子の stdout・stderr を保持する上限。
const MAX_CAPTURE_BYTES: u64 = 1024 * 1024;
/// 親から呼ぶ子テストの名前（変更時は親の `1 passed` 判定が失敗して気づける）。
const CHILD_TEST: &str = "req32_env_i_child_infers_c1_c3_fixture";
const KEYS_PREFIX: &str = "FANDHE_ENV_KEYS=";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("onnx_parity")
}

fn check_kind(doc: &Value, kind: ModelKind) {
    let entry = &doc["kinds"][kind.as_str()];
    let max_bytes =
        usize::try_from(entry["max_bytes"].as_u64().expect("max_bytes")).expect("usize");
    let sha: Sha256Digest = entry["onnx_sha256"]
        .as_str()
        .expect("sha")
        .parse()
        .expect("sha parse");
    let path = fixture_dir().join(entry["onnx"].as_str().expect("onnx"));
    let pipeline = load_pipeline(kind, max_bytes, &path, &sha).expect("load pipeline");
    let cases = entry["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "{}: no cases", kind.as_str());
    let mut singles = Vec::new();
    let mut inputs: Vec<&str> = Vec::new();
    for c in cases {
        let name = c["name"].as_str().expect("name");
        let input = c["input"].as_str().expect("input");
        let expected =
            usize::try_from(c["mlx_label_index"].as_u64().expect("label")).expect("usize");
        let p = pipeline.infer_one(input).expect("infer_one");
        assert_eq!(
            p.label_index(),
            expected,
            "{}: {name}: label",
            kind.as_str()
        );
        singles.push(p);
        inputs.push(input);
    }
    // REQ-28: バッチ推論が単体推論と全件一致する
    let batch = pipeline.infer_batch(&inputs).expect("infer_batch");
    assert_eq!(batch.len(), singles.len(), "{}: batch size", kind.as_str());
    for (i, (b, s)) in batch.iter().zip(&singles).enumerate() {
        assert_eq!(
            b.as_ref().expect("batch item"),
            s,
            "{}: case #{i}: batch differs from single",
            kind.as_str()
        );
    }
}

/// 子として動くテスト。環境変数が空でも `CARGO_MANIFEST_DIR`（コンパイル時定数）から fixture を
/// 解決して推論し、最後に環境変数のキー名だけを出す（環境が実際に消えていた証明。値は出さない）。
/// 単独実行しても害はない。
#[test]
fn req32_env_i_child_infers_c1_c3_fixture() {
    let path = fixture_dir().join("cases.json");
    let len = fs::metadata(&path).expect("cases metadata").len();
    assert!(len <= MAX_CASES_BYTES, "cases.json too large: {len}");
    let doc: Value =
        serde_json::from_str(&fs::read_to_string(&path).expect("cases read")).expect("json");
    check_kind(&doc, ModelKind::C1);
    check_kind(&doc, ModelKind::C3);
    let mut keys: Vec<String> = std::env::vars_os()
        .map(|(k, _)| k.to_string_lossy().into_owned())
        .collect();
    keys.sort();
    println!("{KEYS_PREFIX}{}", keys.join(","));
}

struct ChildOutcome {
    success: bool,
    stdout: String,
}

fn capture<R: Read + Send + 'static>(r: R) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = r.take(MAX_CAPTURE_BYTES).read_to_end(&mut buf);
        String::from_utf8_lossy(&buf).into_owned()
    })
}

/// 環境を空にした子として自分自身を再起動する。`path_env` が `Some` のときだけ `PATH` を設定する。
fn run_isolated_child(path_env: Option<&OsStr>) -> ChildOutcome {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", CHILD_TEST, "--test-threads=1", "--nocapture"])
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(p) = path_env {
        cmd.env("PATH", p);
    }
    let mut child = cmd.spawn().expect("spawn child");
    let h_out = capture(child.stdout.take().expect("stdout"));
    let h_err = capture(child.stderr.take().expect("stderr"));
    let start = Instant::now();
    let status = loop {
        if let Some(st) = child.try_wait().expect("try_wait") {
            break st;
        }
        if start.elapsed() > CHILD_TIMEOUT {
            child.kill().ok();
            child.wait().ok();
            panic!("isolated child timed out");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let stdout = h_out.join().expect("join");
    let _ = h_err.join();
    ChildOutcome {
        success: status.success(),
        stdout,
    }
}

fn env_keys_line(stdout: &str) -> Vec<String> {
    // `--nocapture` では harness の "test <name> ... " が同じ行の先頭に付くため、部分一致で探す
    let line = stdout
        .lines()
        .find_map(|l| l.split_once(KEYS_PREFIX).map(|(_, rest)| rest))
        .expect("keys line");
    line.split(',')
        .filter(|k| !k.is_empty())
        .map(str::to_string)
        .collect()
}

fn assert_child_ok(o: &ChildOutcome, expected_keys: &[&str]) {
    assert!(o.success, "isolated child did not exit 0");
    assert!(
        o.stdout.contains("test result: ok. 1 passed"),
        "child test did not run exactly once"
    );
    assert_eq!(env_keys_line(&o.stdout), expected_keys);
}

/// REQ-32: 厳密な `env -i`（PATH も無い）で推論が exit 0 になる。
#[test]
fn req32_inference_succeeds_under_env_i_without_path() {
    let o = run_isolated_child(None);
    assert_child_ok(&o, &[]);
}

/// REQ-32: `env -i PATH=/usr/bin:/bin`（PoC-16 と同じ形）で推論が exit 0 になる。
#[test]
fn req32_inference_succeeds_under_env_i_with_system_path() {
    let o = run_isolated_child(Some(OsStr::new("/usr/bin:/bin")));
    assert_child_ok(&o, &["PATH"]);
}

/// 一時ディレクトリの Drop 削除ガード（成否に関わらず削除する）。
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_trap(dir: &Path, name: &str, marker: &Path) {
    let file = dir.join(name);
    let body = format!("#!/bin/sh\n: > '{}'\nexit 0\n", marker.display());
    fs::write(&file, body).expect("write trap");
    fs::set_permissions(&file, fs::Permissions::from_mode(0o700)).expect("chmod trap");
}

/// REQ-32: PATH 上に python・python3・uv の罠があっても、推論は Python を起動しない
/// （「たまたま PATH に無かった」ことではなく「起動していない」ことを示す）。
#[test]
fn req32_inference_does_not_invoke_python_from_path() {
    let dir = std::env::temp_dir().join(format!("fandhe-env-isolation-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir(&dir).expect("mkdir");
    let _guard = TempDir(dir.clone());
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).expect("chmod dir");
    let marker = dir.join("trap-invoked.marker");
    for name in ["python", "python3", "uv"] {
        write_trap(&dir, name, &marker);
    }
    let o = run_isolated_child(Some(dir.as_os_str()));
    assert_child_ok(&o, &["PATH"]);
    assert!(!marker.exists(), "a Python trap on PATH was invoked");
}
