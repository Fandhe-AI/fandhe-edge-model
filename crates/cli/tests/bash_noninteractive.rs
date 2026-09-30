//! Bash（`sh`）経由の非対話実行の結合テスト（REQ-36・TASK-36.1-1・#149）。
//!
//! `scripts/cli-infer-noninteractive.sh` を子プロセスで起動し、終了コードと
//! stdout の JSON を具体値で照合する。証拠種別はテストハーネス（実機の
//! Claude Code Bash ツールではない）。`infer` の実推論（TASK-33.1-2・#136 で接続）の
//! exit 0 経路は `req36_infer_real_package_exit_zero_via_sh`（共有 fixture の ONNX を置いた
//! 合成パッケージ）で確認する。TASK-36.1（#148）で、実推論の単体（3 ラベル）・バッチ（`--input-file`）を
//! 予測ラベルの具体値で照合し、実推論経路の実行記録（`req36_run_record_real_infer_*`）も確認する。
//! 末尾の `req36_run_record_*` は実行記録（opt-in の `FANDHE_EDGE_RECORD_DIR`。
//! TASK-36.1-2・#150）の保存形式を具体値で照合する。
//! Windows では `sh` を前提にできないため unix に限定する。

#![cfg(unix)]

use fandhe_edge_cli::args::{self, Subcommand};
use fandhe_edge_cli::output::write_error_report;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// 子プロセスの上限時間（資源上限。REQ-39）。
const TIMEOUT: Duration = Duration::from_secs(30);

struct Out {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("scripts")
        .join("cli-infer-noninteractive.sh")
}

fn run_script(args: &[&str]) -> Out {
    run_script_in(None, args)
}

/// カレントディレクトリを指定して起動する（`infer` の経路ガードは cwd を workspace とする。#159）。
fn run_script_in(cwd: Option<&std::path::Path>, args: &[&str]) -> Out {
    run_script_in_env(cwd, args, &[])
}

/// `run_script` に環境変数を追加で渡す版（実行記録の検証用。TASK-36.1-2）。
fn run_script_env(args: &[&str], envs: &[(&str, &str)]) -> Out {
    run_script_in_env(None, args, envs)
}

fn run_script_in_env(cwd: Option<&std::path::Path>, args: &[&str], envs: &[(&str, &str)]) -> Out {
    let mut cmd = Command::new("sh");
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let mut child = cmd
        // 独立したプロセスグループで起動し、タイムアウト時に子孫も終了できるようにする
        .process_group(0)
        .arg(script_path())
        .args(args)
        .env("FANDHE_EDGE_BIN", env!("CARGO_BIN_EXE_fandhe-edge"))
        .envs(envs.iter().copied())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn sh");
    let mut so = child.stdout.take().expect("stdout");
    let mut se = child.stderr.take().expect("stderr");
    let pgid = child.id();
    let h_out = std::thread::spawn(move || {
        let mut s = String::new();
        so.read_to_string(&mut s).ok();
        s
    });
    let h_err = std::thread::spawn(move || {
        let mut s = String::new();
        se.read_to_string(&mut s).ok();
        s
    });
    let start = Instant::now();
    let status = loop {
        if let Some(st) = child.try_wait().expect("try_wait") {
            break st;
        }
        if start.elapsed() > TIMEOUT {
            // グループ全体を kill する（CLI がパイプを保持しても読み取りが戻る）
            Command::new("kill")
                .args(["-KILL", &format!("-{pgid}")])
                .status()
                .ok();
            child.kill().ok();
            child.wait().ok();
            h_out.join().ok();
            h_err.join().ok();
            panic!("script timed out");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    Out {
        code: status.code(),
        stdout: h_out.join().expect("join"),
        stderr: h_err.join().expect("join"),
    }
}

fn expected_stdout(report: &ErrorReport) -> String {
    let mut buf = Vec::new();
    write_error_report(&mut buf, report).expect("write");
    String::from_utf8(buf).expect("utf8")
}

/// help 経路の exit 0 を具体値で照合する（実推論の exit 0 は `req36_infer_real_package_exit_zero_via_sh`）。
#[test]
fn req36_infer_help_via_sh_exits_0_with_exact_json() {
    let o = run_script(&["--help"]);
    assert_eq!(o.code, Some(0));
    let expected = expected_stdout(&ErrorReport::new(
        ExitCode::Ok,
        args::render_help(Some(Subcommand::Infer)),
    ));
    assert_eq!(o.stdout, expected);
    assert_eq!(o.stderr, "exit_code=0\n");
}

/// バイナリ不在でも stdout 1 JSON・stderr の exit_code=70 を守ること。
#[test]
fn req36_missing_binary_returns_json_and_exit_code_70() {
    let out = Command::new("sh")
        .arg(script_path())
        .arg("--help")
        .env("FANDHE_EDGE_BIN", "/nonexistent/fandhe-edge")
        .stdin(Stdio::null())
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(70));
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge binary not found or not executable\"}\n"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr).lines().last(),
        Some("exit_code=70")
    );
}

/// FANDHE_EDGE_BIN にディレクトリを指定しても（-x は通る）通常ファイル検査で弾かれ、
/// runtime_error の JSON と exit 70 になること（REQ-21・REQ-36）。
#[test]
fn req36_directory_as_binary_returns_json_and_exit_code_70() {
    let out = Command::new("sh")
        .arg(script_path())
        .arg("--help")
        .env("FANDHE_EDGE_BIN", env!("CARGO_MANIFEST_DIR"))
        .stdin(Stdio::null())
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(70));
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge binary not found or not executable\"}\n"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stderr).lines().last(),
        Some("exit_code=70")
    );
}

/// 引数なしでも set -u で abort せず exit_code 行が出ること。
#[test]
fn req36_no_args_still_reports_exit_code() {
    let o = run_script(&[]);
    let code = o.code.expect("exit code");
    assert_ne!(code, 0);
    let expected = format!("exit_code={code}");
    assert_eq!(o.stderr.lines().last(), Some(expected.as_str()));
}

/// スクリプトが非ゼロの終了コードを握りつぶさないこと。
/// 経路ガード（#159）を通る最小のパッケージ（メタデータの必須項目が不足）を置いた一時 workspace で
/// 実行し、ガード通過後の検査が返す 64 が伝わることを確認する（TASK-33.1-2・#136）。
#[test]
fn req36_infer_nonzero_exit_is_propagated_via_sh() {
    let ws = std::env::temp_dir().join(format!("fandhe-noninteractive-{}-ws", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(ws.join("p")).expect("mkdir");
    std::fs::write(
        ws.join("p/artifact.json"),
        r#"{"onnx_file":"model.onnx","kind":"c3","kind_version":1}"#,
    )
    .expect("write");
    // 最小の ONNX 形（形式検査を通る）。
    std::fs::write(
        ws.join("p/model.onnx"),
        [0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78],
    )
    .expect("write");
    let o = run_script_in(Some(&ws), &["--package", "p", "--text", "a"]);
    let _ = std::fs::remove_dir_all(&ws);
    // Linux・macOS 以外の unix ではガードが fail-closed で 70（unsupported_platform）を返す。
    let (code, exit_line, report) = if cfg!(any(target_os = "linux", target_os = "macos")) {
        (
            64,
            "exit_code=64",
            ErrorReport::new(ExitCode::InvalidInput, "artifact metadata is invalid"),
        )
    } else {
        (
            70,
            "exit_code=70",
            ErrorReport::new(
                ExitCode::RuntimeError,
                "path rejected: unsupported_platform",
            ),
        )
    };
    assert_eq!(o.code, Some(code));
    assert_eq!(o.stdout, expected_stdout(&report));
    assert_eq!(o.stderr.lines().last(), Some(exit_line));
}

/// 共有 fixture の C1 の ONNX（選択肢 alpha/beta/gamma）を置いた合成パッケージを一時ディレクトリへ作り、
/// その workspace（cwd にする親ディレクトリ）を返す。パッケージは `<ws>/p`。
/// `max_bytes`・`label_order` は `fixtures/onnx_parity/cases.json` の `kinds.c1` と一致させている
/// （出所は `fixtures/onnx_parity/PROVENANCE.md`）。呼び出し側がテスト終了時に削除する。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn make_real_package(name: &str) -> PathBuf {
    let ws = std::env::temp_dir().join(format!(
        "fandhe-noninteractive-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(ws.join("p")).expect("mkdir");
    let onnx = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/onnx_parity/c1.onnx"),
    )
    .expect("fixture onnx");
    let sha = fandhe_edge_core::hash::Sha256Digest::of_bytes(&onnx).to_hex();
    std::fs::write(ws.join("p/model.onnx"), &onnx).expect("write onnx");
    std::fs::write(
        ws.join("p/artifact.json"),
        format!(
            r#"{{"kind":"c1","kind_version":1,"max_bytes":48,"label_order":["alpha","beta","gamma"],"onnx_file":"model.onnx","onnx_sha256":"{sha}"}}"#
        ),
    )
    .expect("write meta");
    std::fs::write(
        ws.join("p/definition.json"),
        r#"{"schema":"fandhe-edge-model-definition/v1","name":"sh_real","version":1,"judgment_type":"single_select","options":[{"id":"alpha","display_name":"a","description":"d"},{"id":"beta","display_name":"b","description":"d"},{"id":"gamma","display_name":"g","description":"d"}],"io":{"input":"bytes"}}"#,
    )
    .expect("write definition");
    ws
}

/// 余裕（top2_margin）の大きい C1 のケース（入力, 期待ラベル）。`cases.json` の `kinds.c1` の
/// `synthetic_118`（margin 0.888）・`synthetic_004`（0.869）・`synthetic_119`（0.680）を逐語で使う。
/// 不一致は期待値を緩めず原因を調査する（`cases.json` の `_meta` の方針）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
const REAL_CASES: [(&str, &str); 3] = [
    ("beta", "beta"),
    ("ghost grain", "gamma"),
    ("apple amberapple amberapple amber", "alpha"),
];

/// 判定 JSON 1 行の前置（id・status・predicted_label と scores の先頭キー）の期待文字列。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn real_prefix(id: &str, label: &str) -> String {
    format!(
        "{{\"id\":\"{id}\",\"status\":\"ok\",\"predicted_label\":\"{label}\",\"scores\":{{\"alpha\":"
    )
}

/// `scores` のキーが label_order（alpha → beta → gamma）の順に並ぶこと。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn assert_scores_in_label_order(line: &str) {
    let pos: Vec<usize> = ["\"alpha\":", "\"beta\":", "\"gamma\":"]
        .iter()
        .map(|k| line.rfind(k).unwrap_or_else(|| panic!("{k} in {line}")))
        .collect();
    assert!(pos[0] < pos[1] && pos[1] < pos[2], "{line}");
}

/// 実パッケージ（共有 fixture の C1 の ONNX・選択肢 alpha/beta/gamma）で、スクリプト経由の
/// `infer --text` が exit 0 と判定 JSON 1 行を具体値（予測ラベル）で返すこと
/// （REQ-36・REQ-33・TASK-36.1・#148・#136。証拠種別: テストハーネス）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn req36_infer_real_package_exit_zero_via_sh() {
    let ws = make_real_package("real");
    for (input, label) in REAL_CASES {
        let o = run_script_in(Some(&ws), &["--package", "p", "--text", input]);
        assert_eq!(o.code, Some(0), "stdout: {}", o.stdout);
        assert!(o.stdout.ends_with("}\n"), "stdout: {}", o.stdout);
        assert_eq!(o.stdout.lines().count(), 1);
        assert!(
            o.stdout.starts_with(&real_prefix("input", label)),
            "input={input:?} stdout: {}",
            o.stdout
        );
        assert_scores_in_label_order(&o.stdout);
        assert_eq!(o.stderr.lines().last(), Some("exit_code=0"));
    }
    let _ = std::fs::remove_dir_all(&ws);
}

/// 実パッケージのバッチ（`--input-file`。REQ-33 の唯一の例外の 1 行 1 JSON）が入力順に exit 0 で返り、
/// 単体推論（`req36_infer_real_package_exit_zero_via_sh`）と同じラベルになること（REQ-28・REQ-36・#148）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn req36_infer_real_package_batch_via_sh() {
    let ws = make_real_package("batch");
    let mut lines = String::new();
    for (i, (input, _)) in REAL_CASES.iter().enumerate() {
        lines.push_str(&format!(
            "{{\"id\":\"r{}\",\"input\":\"{input}\"}}\n",
            i + 1
        ));
    }
    std::fs::write(ws.join("in.jsonl"), lines).expect("write input");
    let o = run_script_in(Some(&ws), &["--package", "p", "--input-file", "in.jsonl"]);
    let _ = std::fs::remove_dir_all(&ws);
    assert_eq!(o.code, Some(0), "stdout: {}", o.stdout);
    assert!(o.stdout.ends_with("}\n"), "stdout: {}", o.stdout);
    let out: Vec<&str> = o.stdout.lines().collect();
    assert_eq!(out.len(), 3, "stdout: {}", o.stdout);
    for (i, ((_, label), line)) in REAL_CASES.iter().zip(&out).enumerate() {
        assert!(
            line.starts_with(&real_prefix(&format!("r{}", i + 1), label)),
            "{line}"
        );
        assert_scores_in_label_order(line);
    }
    assert_eq!(o.stderr.lines().last(), Some("exit_code=0"));
}

/// 偽の実行ファイル（sh スクリプト）を一時ディレクトリへ作り、スクリプト経由で実行する。
fn run_with_fake_bin(name: &str, body: &str) -> Out {
    run_with_fake_bin_env(name, body, &[])
}

/// `run_with_fake_bin` に環境変数（期限の上書きなど）を追加で渡す版。
fn run_with_fake_bin_env(name: &str, body: &str, envs: &[(&str, &str)]) -> Out {
    run_with_fake_bin_args(name, body, &["--help"], envs)
}

/// 引数も指定できる版（`--input-file` のバッチ判別の検証用）。
fn run_with_fake_bin_args(name: &str, body: &str, args: &[&str], envs: &[(&str, &str)]) -> Out {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!(
        "fandhe-noninteractive-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let bin = dir.join("fake-bin");
    std::fs::write(&bin, format!("#!/bin/sh\n{body}\n")).expect("write");
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    let out = Command::new("sh")
        .arg(script_path())
        .args(args)
        .env("FANDHE_EDGE_BIN", &bin)
        .envs(envs.iter().copied())
        .stdin(Stdio::null())
        .output()
        .expect("run");
    std::fs::remove_dir_all(&dir).ok();
    Out {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// シグナル終了など契約外の終了値は runtime_error(70) の JSON 1 つへ写ること（REQ-21）。
#[test]
fn req21_signal_termination_maps_to_runtime_error_70() {
    let o = run_with_fake_bin("signal", "kill -9 $$");
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge terminated abnormally\"}\n"
    );
    assert_eq!(o.stderr.lines().last(), Some("exit_code=70"));
}

/// 出力後に契約外の 127 を返した場合、既存出力（`code":"ok"` 等）は破棄され、
/// JSON の code と終了コードが runtime_error(70) で一致する 1 JSON になること（REQ-21・REQ-33）。
#[test]
fn req21_output_then_127_is_replaced_by_runtime_error_json() {
    let o = run_with_fake_bin("out127", "echo '{\"code\":\"ok\"}'\nexit 127");
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge terminated abnormally\"}\n"
    );
    assert_eq!(o.stderr.lines().last(), Some("exit_code=70"));
}

/// 許可された終了コード（70・0）でも stdout が空なら runtime_error(70) の JSON を補うこと
/// （REQ-21・REQ-33）。
#[test]
fn req33_empty_stdout_with_allowed_exit_codes_gets_runtime_error_json() {
    for (name, body) in [("empty70", "exit 70"), ("empty0", "exit 0")] {
        let o = run_with_fake_bin(name, body);
        assert_eq!(o.code, Some(70), "{name}");
        assert_eq!(
            o.stdout,
            "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge produced no output\"}\n",
            "{name}"
        );
        assert_eq!(o.stderr.lines().last(), Some("exit_code=70"), "{name}");
    }
}

/// 期限を超えた子は終了され、runtime_error(70) の JSON 1 つが返ること（REQ-39・REQ-21）。
#[test]
fn req39_timeout_kills_child_and_returns_runtime_error_70() {
    let started = Instant::now();
    let o = run_with_fake_bin_env(
        "timeout",
        "exec sleep 60",
        &[("FANDHE_EDGE_TIMEOUT_SECS", "1")],
    );
    assert!(started.elapsed() < Duration::from_secs(20));
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge timed out\"}\n"
    );
    assert_eq!(o.stderr.lines().last(), Some("exit_code=70"));
}

/// stdout が容量上限（1 MiB）を超えた子は終了され、途中までの出力は破棄されて
/// runtime_error(70) の JSON 1 つが返ること（REQ-39・REQ-21）。
#[test]
fn req39_output_over_limit_returns_runtime_error_70() {
    let o = run_with_fake_bin("bigout", "exec yes");
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge output exceeded size limit\"}\n"
    );
    assert_eq!(o.stderr.lines().last(), Some("exit_code=70"));
}

/// stderr が容量上限（64 KiB）を超えた子は終了され、runtime_error(70) の JSON 1 つが
/// 返ること。stderr へ本文は中継されない（REQ-39・REQ-21）。
#[test]
fn req39_stderr_over_limit_returns_runtime_error_70() {
    let o = run_with_fake_bin("bigerr", "exec yes 1>&2");
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge stderr exceeded size limit\"}\n"
    );
    assert_eq!(o.stderr, "exit_code=70\n");
}

/// 許可された終了コードでも、複数 JSON・途中切れの stdout は中継されず
/// runtime_error(70) の JSON 1 つへ置き換わること（REQ-33）。
#[test]
fn req33_invalid_json_stdout_is_replaced_by_runtime_error_json() {
    let cases = [
        (
            "twojson",
            "echo '{\"code\":\"ok\"}'\necho '{\"code\":\"ok\"}'\nexit 0",
        ),
        (
            "sameline",
            "echo '{\"code\":\"ok\"}{\"code\":\"ok\"}'\nexit 0",
        ),
        (
            "truncated",
            "printf '{\"code\":\"ok\",\"message\":\"a'\nexit 0",
        ),
        ("notobject", "echo 'hello'\nexit 10"),
    ];
    for (name, body) in cases {
        let o = run_with_fake_bin(name, body);
        assert_eq!(o.code, Some(70), "{name}");
        assert_eq!(
            o.stdout,
            "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge produced invalid output\"}\n",
            "{name}"
        );
        assert_eq!(o.stderr.lines().last(), Some("exit_code=70"), "{name}");
    }
}

/// 文字列内の括弧・エスケープを含む正しい JSON 1 つは、許可された終了コードのまま中継されること。
#[test]
fn req33_valid_json_with_braces_in_strings_is_relayed() {
    let o = run_with_fake_bin(
        "validjson",
        "echo '{\"code\":\"judged_fail\",\"message\":\"a}{ \\\"q\\\" ]\"}'\nexit 10",
    );
    assert_eq!(o.code, Some(10));
    assert_eq!(
        o.stdout,
        "{\"code\":\"judged_fail\",\"message\":\"a}{ \\\"q\\\" ]\"}\n"
    );
    assert_eq!(o.stderr, "exit_code=10\n");
}

/// 先頭 0 の期限指定（08 など）は 8 進数として解釈されず既定値へ戻り、異常終了しないこと。
#[test]
fn req39_leading_zero_timeout_falls_back_to_default() {
    for v in ["08", "09", "00"] {
        let o = run_with_fake_bin_env(
            "leadzero",
            "echo '{\"code\":\"ok\"}'\nexit 0",
            &[("FANDHE_EDGE_TIMEOUT_SECS", v)],
        );
        assert_eq!(o.code, Some(0), "{v}");
        assert_eq!(o.stdout, "{\"code\":\"ok\"}\n", "{v}");
        assert_eq!(o.stderr, "exit_code=0\n", "{v}");
    }
}

/// 期限超過時は子孫プロセスもプロセスグループごと終了されること（REQ-39）。
#[test]
fn req39_timeout_kills_descendants() {
    let marker = std::env::temp_dir().join(format!("fandhe-desc-{}", std::process::id()));
    let body = format!(
        "(sleep 3; echo alive >'{}') &\nexec sleep 60",
        marker.display()
    );
    let o = run_with_fake_bin_env("desc", &body, &[("FANDHE_EDGE_TIMEOUT_SECS", "1")]);
    assert_eq!(o.code, Some(70));
    std::thread::sleep(Duration::from_secs(4));
    let survived = marker.exists();
    std::fs::remove_file(&marker).ok();
    assert!(!survived, "descendant survived the timeout");
}

const INVALID_OUTPUT: &str =
    "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge produced invalid output\"}\n";
const CODE_MISMATCH: &str = "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge output code does not match exit code\"}\n";

/// 括弧が釣り合っていても JSON 構文として不正な出力は中継されないこと（REQ-33）。
#[test]
fn req33_syntactically_invalid_json_is_replaced_even_if_braces_balance() {
    let cases = [
        ("trailingcomma", r#"{"code":"ok",}"#),
        ("barekey", "{invalid}"),
        ("leadingzero", r#"{"code":"ok","n":01}"#),
        ("badescape", r#"{"code":"ok","m":"a\qb"}"#),
        ("missingcolon", r#"{"code":"ok" "m":1}"#),
        ("trailinggarbage", r#"{"code":"ok"} x"#),
        ("badliteral", r#"{"code":"ok","v":tru}"#),
        ("emptykey", "{\"\"}"),
    ];
    for (name, json) in cases {
        let o = run_with_fake_bin(name, &format!("echo '{json}'\nexit 0"));
        assert_eq!(o.code, Some(70), "{name}");
        assert_eq!(o.stdout, INVALID_OUTPUT, "{name}");
        assert_eq!(o.stderr.lines().last(), Some("exit_code=70"), "{name}");
    }
}

/// 数値・入れ子・エスケープを含む構文的に正しい JSON は中継されること（REQ-33）。
#[test]
fn req33_valid_nested_json_is_relayed() {
    let json = r#"{"code":"ok","a":[1,-2.5e+3,{"b":null,"c":true}],"u":"\u00e9\n","e":{}}"#;
    let o = run_with_fake_bin("nested", &format!("printf '%s\\n' '{json}'\nexit 0"));
    assert_eq!(o.code, Some(0));
    assert_eq!(o.stdout, format!("{json}\n"));
}

/// 終了コードと JSON の `code` の対応（7 種）が一致するときだけ中継し、
/// 不一致は runtime_error(70) へ置き換えること。対応表は core の `ExitCode::ALL`
/// （fixtures/exitcode/exit_codes.json と core のテストが照合）を正とする（REQ-21）。
#[test]
fn req21_exit_code_and_json_code_correspondence_is_enforced() {
    for exit in ExitCode::ALL {
        for other in ExitCode::ALL {
            let json = format!("{{\"code\":\"{}\",\"message\":\"m\"}}", other.name());
            let o = run_with_fake_bin("map", &format!("echo '{json}'\nexit {}", exit.code()));
            if exit == other {
                assert_eq!(o.code, Some(i32::from(exit.code())), "{exit:?}");
                assert_eq!(o.stdout, format!("{json}\n"), "{exit:?}");
            } else {
                assert_eq!(o.code, Some(70), "{exit:?}/{other:?}");
                assert_eq!(o.stdout, CODE_MISMATCH, "{exit:?}/{other:?}");
                assert_eq!(o.stderr.lines().last(), Some("exit_code=70"));
            }
        }
    }
}

/// exit 0 で `code` が runtime_error・非 0 で `code` 欠落・code が文字列でない場合は不一致。
#[test]
fn req21_mismatch_edge_cases_map_to_runtime_error_70() {
    let cases = [
        ("zero_rt", "echo '{\"code\":\"runtime_error\"}'\nexit 0"),
        ("nonzero_nocode", "echo '{\"message\":\"x\"}'\nexit 10"),
        ("code_number", "echo '{\"code\":10}'\nexit 10"),
        (
            "code_dup",
            "echo '{\"code\":\"ok\",\"code\":\"ok\"}'\nexit 0",
        ),
        ("unknown_name", "echo '{\"code\":\"weird\"}'\nexit 0"),
    ];
    for (name, body) in cases {
        let o = run_with_fake_bin(name, body);
        assert_eq!(o.code, Some(70), "{name}");
        assert_eq!(o.stdout, CODE_MISMATCH, "{name}");
    }
}

/// バッチ（`--input-file`）は 1 行 1 JSON を全行検証して中継すること（REQ-33）。
#[test]
fn req33_batch_input_file_relays_every_line() {
    let body = "echo '{\"code\":\"ok\",\"label\":\"a\"}'\necho '{\"label\":\"b\"}'\nexit 0";
    for args in [["--input-file", "x.jsonl"], ["--input-file=x.jsonl", "--x"]] {
        let o = run_with_fake_bin_args("batch", body, &args, &[]);
        assert_eq!(o.code, Some(0));
        assert_eq!(
            o.stdout,
            "{\"code\":\"ok\",\"label\":\"a\"}\n{\"label\":\"b\"}\n"
        );
        assert_eq!(o.stderr, "exit_code=0\n");
    }
}

/// バッチでも不正な行・空行・既知でない `code`・非 0 終了で最終行の不一致は置き換える。
/// バッチでない呼び出しの複数行は引き続き拒否する（REQ-33・REQ-21）。
#[test]
fn req33_batch_rejects_bad_lines_and_single_mode_rejects_multiline() {
    let args = ["--input-file", "x.jsonl"];
    let bad = [
        (
            "badline",
            "echo '{\"a\":1}'\necho '{bad}'\nexit 0",
            INVALID_OUTPUT,
        ),
        (
            "blankline",
            "echo '{\"a\":1}'\necho\necho '{\"a\":2}'\nexit 0",
            INVALID_OUTPUT,
        ),
        ("array", "echo '[1]'\nexit 0", INVALID_OUTPUT),
        (
            "unknown",
            "echo '{\"code\":\"weird\"}'\nexit 0",
            CODE_MISMATCH,
        ),
        (
            "lastmismatch",
            "echo '{\"code\":\"ok\"}'\nexit 10",
            CODE_MISMATCH,
        ),
    ];
    for (name, body, expected) in bad {
        let o = run_with_fake_bin_args(name, body, &args, &[]);
        assert_eq!(o.code, Some(70), "{name}");
        assert_eq!(o.stdout, expected, "{name}");
    }
    let ok = run_with_fake_bin_args(
        "batchfail",
        "echo '{\"code\":\"ok\"}'\necho '{\"code\":\"judged_fail\"}'\nexit 10",
        &args,
        &[],
    );
    assert_eq!(ok.code, Some(10));
    let single = run_with_fake_bin(
        "single",
        "echo '{\"code\":\"ok\"}'\necho '{\"code\":\"ok\"}'\nexit 0",
    );
    assert_eq!(single.code, Some(70));
    assert_eq!(single.stdout, INVALID_OUTPUT);
}

/// 親が終了しても残ったバックグラウンドの子孫に期限が適用され、終了・回収されること
/// （親の終了で上限が外れない。REQ-39）。
#[test]
fn req39_descendant_outliving_parent_is_bounded_by_timeout() {
    let marker = std::env::temp_dir().join(format!("fandhe-orphan-{}", std::process::id()));
    let body = format!(
        "echo '{{\"code\":\"ok\"}}'\n(sleep 3; echo alive >'{}') &\nexit 0",
        marker.display()
    );
    let started = Instant::now();
    let o = run_with_fake_bin_env("orphan", &body, &[("FANDHE_EDGE_TIMEOUT_SECS", "1")]);
    assert!(started.elapsed() < Duration::from_secs(20));
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge timed out\"}\n"
    );
    std::thread::sleep(Duration::from_secs(4));
    let survived = marker.exists();
    std::fs::remove_file(&marker).ok();
    assert!(!survived, "orphaned descendant survived the timeout");
}

/// 親が終了した後に子孫が出力し続けても容量上限が適用されること（REQ-39）。
#[test]
fn req39_descendant_output_after_parent_exit_hits_output_limit() {
    let o = run_with_fake_bin(
        "orphanflood",
        "echo '{\"code\":\"ok\"}'\n(exec yes) &\nexit 0",
    );
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge output exceeded size limit\"}\n"
    );
}

/// 他オプションの値として現れた `--input-file` はバッチ指定とみなさず、複数行は拒否する。
/// 値として消費された `--id` 等の後に本物の `--input-file` があればバッチ（REQ-33）。
#[test]
fn req33_input_file_as_option_value_is_not_batch() {
    let body = "echo '{\"code\":\"ok\"}'\necho '{\"code\":\"ok\"}'\nexit 0";
    for args in [
        vec!["--text", "--input-file"],
        vec!["--package", "--input-file", "--text", "a"],
        vec!["--text=--input-file"],
    ] {
        let o = run_with_fake_bin_args("valbatch", body, &args, &[]);
        assert_eq!(o.code, Some(70), "{args:?}");
        assert_eq!(o.stdout, INVALID_OUTPUT, "{args:?}");
    }
    let o = run_with_fake_bin_args(
        "realbatch",
        body,
        &["--package", "p", "--id", "i", "--input-file", "f"],
        &[],
    );
    assert_eq!(o.code, Some(0));
}

/// top-level のキー・`code` の値のエスケープは復号してから判定する（REQ-21）。
/// `code` 以外のキーはエスケープがあっても素通しし、`code` に復号されるキーは照合対象になる。
#[test]
fn req21_escaped_keys_and_code_values_are_decoded_before_comparison() {
    // (名前, JSON, 終了コード, 期待する終了コード, 中継されるか)
    let cases = [
        ("esc_normal_key", r#"{"code":"ok","name":"x"}"#, 0, 0, true),
        ("esc_simple_key", r#"{"code":"ok","a\nb":"x"}"#, 0, 0, true),
        ("esc_surrogate_key", r#"{"code":"ok","😀":1}"#, 0, 0, true),
        ("esc_code_key_match", r#"{"code":"ok"}"#, 0, 0, true),
        (
            "esc_code_key_upper_hex",
            r#"{"code":"runtime_error"}"#,
            70,
            70,
            true,
        ),
        (
            "esc_code_key_mismatch",
            r#"{"code":"runtime_error"}"#,
            0,
            70,
            false,
        ),
        ("esc_value_match", r#"{"code":"ok"}"#, 0, 0, true),
        (
            "esc_value_underscore",
            r#"{"code":"runtime_error"}"#,
            70,
            70,
            true,
        ),
        (
            "esc_value_mismatch",
            r#"{"code":"runtime_error"}"#,
            0,
            70,
            false,
        ),
        ("esc_dup_code", r#"{"code":"ok","code":"ok"}"#, 0, 70, false),
    ];
    for (name, json, exit, expected, relayed) in cases {
        let o = run_with_fake_bin(name, &format!("printf '%s\\n' '{json}'\nexit {exit}"));
        assert_eq!(o.code, Some(expected), "{name}");
        if relayed {
            assert_eq!(o.stdout, format!("{json}\n"), "{name}");
        } else {
            assert_eq!(o.stdout, CODE_MISMATCH, "{name}");
        }
    }
}

/// 復号できない不正なエスケープはキーでも値でも 70（不正な出力）になる（REQ-33）。
#[test]
fn req33_invalid_escapes_in_keys_and_values_are_rejected() {
    for (name, json) in [
        ("badkeyhex", r#"{"code":"ok","a\u12G4":1}"#),
        ("badkeyesc", r#"{"code":"ok","a\qb":1}"#),
        ("badvaluehex", r#"{"code":"o\u00zzk"}"#),
    ] {
        let o = run_with_fake_bin(name, &format!("printf '%s\\n' '{json}'\nexit 0"));
        assert_eq!(o.code, Some(70), "{name}");
        assert_eq!(o.stdout, INVALID_OUTPUT, "{name}");
    }
}

/// `--out`（`--out=...` 形式を含む）は CLI を起動せず invalid_input(64) で拒否し、
/// 出力を stdout 経由だけに閉じ込める。値として現れる `--out` は誤検出しない（REQ-39・REQ-33）。
#[test]
fn req39_out_option_is_rejected_without_launching_cli() {
    let dir = std::env::temp_dir().join(format!("fandhe-out-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let launched = dir.join("launched");
    let target = dir.join("secret-out-path");
    let body = format!(
        "echo x >'{}'\necho '{{\"code\":\"ok\"}}'\nexit 0",
        launched.display()
    );
    let t = target.display().to_string();
    let t_eq = format!("--out={t}");
    for args in [
        vec!["--input-file", "f", "--out", t.as_str()],
        vec!["--input-file", "f", t_eq.as_str()],
    ] {
        let o = run_with_fake_bin_args("outopt", &body, &args, &[]);
        assert_eq!(o.code, Some(64), "{args:?}");
        assert_eq!(
            o.stdout,
            "{\"code\":\"invalid_input\",\"message\":\"--out is not supported by the non-interactive wrapper\"}\n"
        );
        assert!(!o.stdout.contains(&t) && !o.stderr.contains(&t));
        assert_eq!(o.stderr.lines().last(), Some("exit_code=64"));
        assert!(!launched.exists(), "CLI must not be launched");
    }
    // 他オプションの値として現れる `--out` は拒否しない（引数走査は CLI と同じ規則）
    for args in [
        vec!["--text", "--out"],
        vec!["--text=--out"],
        vec!["--id", "--out", "--text", "a"],
    ] {
        let o = run_with_fake_bin_args("outvalue", &body, &args, &[]);
        assert_eq!(o.code, Some(0), "{args:?}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// 末尾に改行の無い出力は 1 行 1 JSON の契約に反するため置き換える（REQ-33）。
#[test]
fn req33_missing_trailing_newline_is_rejected() {
    for args in [vec!["--help"], vec!["--input-file", "f"]] {
        let o = run_with_fake_bin_args("nonl", "printf '{\"code\":\"ok\"}'\nexit 0", &args, &[]);
        assert_eq!(o.code, Some(70), "{args:?}");
        assert_eq!(o.stdout, INVALID_OUTPUT, "{args:?}");
    }
}

/// ps の取得に失敗したら子孫の終了とみなさず fail-closed（グループを終了して 70）。
/// 生きている子孫は残さない（REQ-39）。
#[test]
fn req39_ps_failure_is_fail_closed_and_kills_group() {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("fandhe-fakeps-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let ps = dir.join("ps");
    std::fs::write(&ps, "#!/bin/sh\nexit 1\n").expect("write");
    std::fs::set_permissions(&ps, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    let marker = std::env::temp_dir().join(format!("fandhe-psfail-{}", std::process::id()));
    let body = format!(
        "echo '{{\"code\":\"ok\"}}'\n(sleep 3; echo alive >'{}') &\nexit 0",
        marker.display()
    );
    let path = format!(
        "{}:{}",
        dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let o = run_with_fake_bin_args("psfail", &body, &["--help"], &[("PATH", &path)]);
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"fandhe-edge process monitoring failed\"}\n"
    );
    std::thread::sleep(Duration::from_secs(4));
    let survived = marker.exists();
    std::fs::remove_file(&marker).ok();
    std::fs::remove_dir_all(&dir).ok();
    assert!(!survived, "descendant survived ps failure");
}

/// 出力に不正な UTF-8 が含まれる場合は中継せず 70 に置き換え、正当な多バイト文字は通す
/// （REQ-33）。
#[test]
fn req33_invalid_utf8_output_is_rejected() {
    let o = run_with_fake_bin(
        "badutf8",
        "printf '{\"code\":\"ok\",\"m\":\"\\377\"}\\n'\nexit 0",
    );
    assert_eq!(o.code, Some(70));
    assert_eq!(o.stdout, INVALID_OUTPUT);
    let ok = run_with_fake_bin(
        "goodutf8",
        "printf '{\"code\":\"ok\",\"m\":\"\\303\\251\"}\\n'\nexit 0",
    );
    assert_eq!(ok.code, Some(0));
    assert_eq!(ok.stdout, "{\"code\":\"ok\",\"m\":\"\u{e9}\"}\n");
}

/// ラッパーが SIGKILL で突然死しても、監視役が子のプロセスグループを終了すること（REQ-39）。
#[test]
fn req39_wrapper_sigkill_does_not_leave_child_group() {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("fandhe-wdeath-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let marker = dir.join("alive");
    let bin = dir.join("fake-bin");
    std::fs::write(
        &bin,
        format!(
            "#!/bin/sh\n(sleep 4; echo alive >'{}') &\nexec sleep 60\n",
            marker.display()
        ),
    )
    .expect("write");
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    let mut wrapper = Command::new("sh")
        .arg(script_path())
        .arg("--help")
        .env("FANDHE_EDGE_BIN", &bin)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn");
    std::thread::sleep(Duration::from_millis(800));
    wrapper.kill().expect("kill wrapper");
    wrapper.wait().ok();
    std::thread::sleep(Duration::from_secs(6));
    let survived = marker.exists();
    std::fs::remove_dir_all(&dir).ok();
    assert!(!survived, "child group survived wrapper SIGKILL");
}

/// ラッパーが SIGTERM で止められても trap で子のグループを終了すること（REQ-39）。
#[test]
fn req39_wrapper_sigterm_does_not_leave_child_group() {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("fandhe-wterm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let marker = dir.join("alive");
    let bin = dir.join("fake-bin");
    std::fs::write(
        &bin,
        format!(
            "#!/bin/sh\n(sleep 3; echo alive >'{}') &\nexec sleep 60\n",
            marker.display()
        ),
    )
    .expect("write");
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    let wrapper = Command::new("sh")
        .arg(script_path())
        .arg("--help")
        .env("FANDHE_EDGE_BIN", &bin)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn");
    std::thread::sleep(Duration::from_millis(800));
    Command::new("kill")
        .args(["-TERM", &wrapper.id().to_string()])
        .status()
        .ok();
    std::thread::sleep(Duration::from_secs(5));
    let survived = marker.exists();
    std::fs::remove_dir_all(&dir).ok();
    let mut wrapper = wrapper;
    wrapper.wait().ok();
    assert!(!survived, "child group survived wrapper SIGTERM");
}

/// 書き込み時点で上限が効いていること（ラッパーの内部を覗かず、偽バイナリ側で証明する）。
/// 偽バイナリは 64 KiB のチャンクを合計 64 MiB まで stdout（stderr 版も）へ書き、成功のたびに
/// テストが所有するディレクトリの計数ファイルへチャンク数を記録する。`head -c` がパイプを
/// 閉じると SIGPIPE で止まるため、累計は「上限 + パイプバッファ分の余裕（1 MiB）」以下で
/// 総量には達しない。計数はラッパーの終了後に読む（監視との競合を避ける。REQ-39・REQ-21）。
#[test]
fn req39_output_caps_are_enforced_at_write_time() {
    const CHUNK: u64 = 65_536;
    const TOTAL_CHUNKS: u64 = 1024; // 64 MiB
    const SLACK: u64 = 1_048_576;
    let cases = [
        ("errcap", "1>&2", 65_536_u64, "stderr exceeded size limit"),
        ("outcap", "", 1_048_576_u64, "output exceeded size limit"),
    ];
    for (name, redirect, limit, message) in cases {
        let dir = std::env::temp_dir().join(format!("fandhe-cap-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let counter = dir.join("chunks");
        let body = format!(
            "i=0\necho 0 >'{0}'\nwhile [ $i -lt {TOTAL_CHUNKS} ]; do\n  head -c {CHUNK} /dev/zero {redirect} || exit 0\n  i=$((i+1))\n  echo $i >>'{0}'\ndone\nexit 0",
            counter.display()
        );
        let o = run_with_fake_bin(name, &body);
        // 追記方式のため、子が停止直前に書き込み途中でも最終行の完全な値だけを採用する。
        // 子は開始直後に 0 を書くので完全な値は必ず 1 行以上ある。読めない・有効行が無い場合は
        // 計測不能として失敗させる（0 に丸めて上限検証を通さない）
        let raw = std::fs::read_to_string(&counter)
            .unwrap_or_else(|e| panic!("{name}: cannot read chunk counter: {e}"));
        let chunks: u64 = raw
            .lines()
            .rev()
            .find_map(|l| l.trim().parse().ok())
            .unwrap_or_else(|| panic!("{name}: chunk counter has no complete value"));
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(o.code, Some(70), "{name}");
        assert_eq!(
            o.stdout,
            format!("{{\"code\":\"runtime_error\",\"message\":\"fandhe-edge {message}\"}}\n"),
            "{name}"
        );
        assert!(o.stdout.len() as u64 <= limit, "{name}");
        let written = chunks.saturating_mul(CHUNK);
        assert!(
            written <= limit + SLACK,
            "{name}: fake child wrote {written} bytes before stopping"
        );
        assert!(
            written < TOTAL_CHUNKS * CHUNK,
            "{name}: child was never stopped"
        );
    }
}

/// 非 ASCII の `\uXXXX`・サロゲートペア・16 進の大文字小文字は正当な JSON として扱い、
/// `code` 以外は素通し、`code` に関わる場合は復号後の値で照合する（REQ-21・REQ-33）。
#[test]
fn req33_unicode_escapes_are_valid_and_compared_after_decoding() {
    let cases = [
        ("uni_key", r#"{"code":"ok","é€":1}"#, 0, 0, true),
        ("uni_value", r#"{"code":"ok","m":"é€😀"}"#, 0, 0, true),
        ("pair_upper", r#"{"code":"ok","😀":"😀"}"#, 0, 0, true),
        ("pair_mixed", r#"{"code":"ok","😀":1}"#, 0, 0, true),
        ("code_nonascii_value", r#"{"code":"ék"}"#, 0, 70, false),
        ("code_pair_value", r#"{"code":"😀"}"#, 0, 70, false),
        (
            "code_nonascii_key",
            r#"{"céde":"runtime_error"}"#,
            0,
            0,
            true,
        ),
    ];
    for (name, json, exit, expected, relayed) in cases {
        let o = run_with_fake_bin(name, &format!("printf '%s\n' '{json}'\nexit {exit}"));
        assert_eq!(o.code, Some(expected), "{name}");
        if relayed {
            assert_eq!(o.stdout, format!("{json}\n"), "{name}");
        } else {
            assert_eq!(o.stdout, CODE_MISMATCH, "{name}");
        }
    }
}

/// 単独の high / low サロゲートは、キー・値・入れ子のどこにあっても fail-closed（70）にする
/// （REQ-33）。
#[test]
fn req33_lone_surrogates_are_rejected() {
    let cases = [
        ("lone_high_value", r#"{"code":"ok","m":"\ud83d"}"#),
        ("lone_high_then_char", r#"{"code":"ok","m":"\ud83dx"}"#),
        ("high_then_high", r#"{"code":"ok","m":"\ud83d\ud83d"}"#),
        ("high_then_nonsurrogate", r#"{"code":"ok","m":"\ud83dA"}"#),
        ("lone_low_value", r#"{"code":"ok","m":"\ude00"}"#),
        ("low_then_high", r#"{"code":"ok","m":"\ude00\ud83d"}"#),
        ("lone_high_key", r#"{"code":"ok","\ud83d":1}"#),
        ("lone_low_key", r#"{"code":"ok","\uDE00":1}"#),
        ("lone_nested", r#"{"code":"ok","a":[{"b":"\uDC00"}]}"#),
        ("lone_in_code", r#"{"code":"\ud83d"}"#),
    ];
    for (name, json) in cases {
        let o = run_with_fake_bin(name, &format!("printf '%s\n' '{json}'\nexit 0"));
        assert_eq!(o.code, Some(70), "{name}");
        assert_eq!(o.stdout, INVALID_OUTPUT, "{name}");
    }
}

/// ラッパーが 70 へ置き換える経路では、CLI の生の stderr を中継せず、診断行だけを出す。
/// 置き換えない経路（契約どおりの出力）は CLI の stderr を中継する（REQ-21・REQ-33・REQ-39）。
#[test]
fn req21_replaced_results_never_relay_raw_cli_stderr() {
    let noise = "echo 'secret-diagnostic' 1>&2\n";
    let replaced = [
        ("r_invalid", "echo '{bad}'\nexit 0", None),
        ("r_mismatch", "echo '{\"code\":\"ok\"}'\nexit 10", None),
        ("r_empty", "exit 0", None),
        ("r_abnormal", "echo '{\"code\":\"ok\"}'\nexit 127", None),
        ("r_stderr_limit", "exec yes 1>&2", None),
        ("r_output_limit", "exec yes", None),
        ("r_timeout", "exec sleep 60", Some("1")),
    ];
    for (name, body, timeout) in replaced {
        let envs: Vec<(&str, &str)> = timeout
            .map(|s| vec![("FANDHE_EDGE_TIMEOUT_SECS", s)])
            .unwrap_or_default();
        let o = run_with_fake_bin_env(name, &format!("{noise}{body}"), &envs);
        assert_eq!(o.code, Some(70), "{name}");
        assert_eq!(o.stderr, "exit_code=70\n", "{name}");
    }
    let o = run_with_fake_bin(
        "r_passthrough",
        &format!("{noise}echo '{{\"code\":\"judged_fail\"}}'\nexit 10"),
    );
    assert_eq!(o.code, Some(10));
    assert_eq!(o.stderr, "secret-diagnostic\nexit_code=10\n");
}
// ---- 実行記録（REQ-36・TASK-36.1-2・#150）----
// 証拠種別はテストハーネス（fake bin・help 経路・実パッケージの実推論経路）。cli は serde_json に依存しないため、
// JSON は解析せず期待文字列との完全一致で照合する。

const RECORD_SCHEMA: &str = "fandhe-edge.run-record/1";

/// テスト名と PID で一意な記録ディレクトリを作る。
fn record_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fandhe-record-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).expect("mkdir record dir");
    d
}

/// 記録ディレクトリ直下のファイルを (名前, 内容) で返す。
fn record_files(dir: &std::path::Path) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = std::fs::read_dir(dir)
        .expect("read_dir")
        .map(|e| {
            let e = e.expect("entry");
            (
                e.file_name().to_string_lossy().into_owned(),
                String::from_utf8_lossy(&std::fs::read(e.path()).expect("read")).into_owned(),
            )
        })
        .collect();
    v.sort();
    v
}

/// バイト列の sha256（16 進 64 桁）。スクリプトと同じ外部ツール（sha256sum / shasum）で求める。
fn sha256_hex(data: &[u8]) -> String {
    use std::io::Write;
    for (tool, extra) in [("sha256sum", None), ("shasum", Some("256"))] {
        let mut cmd = Command::new(tool);
        if let Some(n) = extra {
            cmd.args(["-a", n]);
        }
        let Ok(mut child) = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        else {
            continue;
        };
        child
            .stdin
            .take()
            .expect("stdin")
            .write_all(data)
            .expect("write");
        let out = child.wait_with_output().expect("wait");
        let text = String::from_utf8(out.stdout).expect("utf8");
        return text.split(' ').next().expect("hash").to_string();
    }
    panic!("no sha256 tool");
}

/// 記録の stdout / stderr 要約（本文を含まず、バイト数と sha256 だけ）の期待文字列。
fn summary_json(data: &[u8]) -> String {
    format!(
        "{{\"bytes\":{},\"sha256\":\"{}\"}}",
        data.len(),
        sha256_hex(data)
    )
}

/// UTC の暦日時（YYYY-MM-DDTHH:MM:SSZ）を epoch 秒へ（days_from_civil）。
fn epoch_of(ts: &str) -> i64 {
    let b = ts.as_bytes();
    assert_eq!(ts.len(), 20, "{ts}");
    assert!(b[4] == b'-' && b[7] == b'-' && b[10] == b'T' && b[13] == b':' && b[16] == b':');
    assert_eq!(b[19], b'Z');
    let n = |r: std::ops::Range<usize>| -> i64 { ts[r].parse().expect("digits") };
    let (y, m, d) = (n(0..4), n(5..7), n(8..10));
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    days * 86400 + n(11..13) * 3600 + n(14..16) * 60 + n(17..19)
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("epoch")
        .as_secs() as i64
}

/// 記録ファイルを 1 つだけ取り出す。
fn only_record(dir: &std::path::Path) -> String {
    let files = record_files(dir);
    assert_eq!(files.len(), 1, "{files:?}");
    assert!(files[0].0.starts_with("run-record."), "{}", files[0].0);
    files[0].1.clone()
}

/// started_at を切り出す（`"started_at":"` の直後 20 文字）。
fn started_at_of(rec: &str) -> String {
    let key = "\"started_at\":\"";
    let i = rec.find(key).expect("started_at") + key.len();
    rec[i..i + 20].to_string()
}

/// 受け入れ条件: 5 項目（command・started_at・exit_code・stdout・stderr）が具体値で残る。
#[test]
fn req36_run_record_contains_five_fields_with_exact_values() {
    let dir = record_dir("five");
    let before = now_secs();
    let o = run_script_env(
        &["--help"],
        &[("FANDHE_EDGE_RECORD_DIR", dir.to_str().unwrap())],
    );
    let after = now_secs();
    assert_eq!(o.code, Some(0));
    let help = expected_stdout(&ErrorReport::new(
        ExitCode::Ok,
        args::render_help(Some(Subcommand::Infer)),
    ));
    assert_eq!(o.stdout, help);
    // 記録しても契約の出力（stderr）は変わらない
    assert_eq!(o.stderr, "exit_code=0\n");
    let rec = only_record(&dir);
    let started = started_at_of(&rec);
    let t = epoch_of(&started);
    assert!(
        t >= before - 1 && t <= after + 1,
        "{started} {before} {after}"
    );
    let expected = format!(
        "{{\"schema\":\"{RECORD_SCHEMA}\",\"command\":[\"fandhe-edge\",\"infer\",\"--help\"],\"started_at\":\"{started}\",\"exit_code\":0,\"stdout\":{},\"stderr\":{}}}\n",
        summary_json(help.as_bytes()),
        summary_json(b"")
    );
    assert_eq!(rec, expected);
    std::fs::remove_dir_all(&dir).ok();
}

/// 実推論経路（実パッケージ・exit 0）でも実行記録が残り、入力本文と判定結果が平文で残らないこと
/// （REQ-36・TASK-36.1-2・#150・#148。security.md のデータ本文の転記禁止）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn req36_run_record_real_infer_exit_zero_with_exact_values() {
    let ws = make_real_package("recreal");
    let dir = record_dir("real-infer");
    let before = now_secs();
    let o = run_script_in_env(
        Some(&ws),
        &["--package", "p", "--text", "beta"],
        &[("FANDHE_EDGE_RECORD_DIR", dir.to_str().unwrap())],
    );
    let after = now_secs();
    assert_eq!(o.code, Some(0), "stdout: {}", o.stdout);
    assert!(
        o.stdout.starts_with(&real_prefix("input", "beta")),
        "{}",
        o.stdout
    );
    assert_eq!(o.stderr.lines().last(), Some("exit_code=0"));
    let cli_stderr = o.stderr.strip_suffix("exit_code=0\n").expect("diag line");
    let rec = only_record(&dir);
    let started = started_at_of(&rec);
    let t = epoch_of(&started);
    assert!(
        t >= before - 1 && t <= after + 1,
        "{started} {before} {after}"
    );
    let expected = format!(
        "{{\"schema\":\"{RECORD_SCHEMA}\",\"command\":[\"fandhe-edge\",\"infer\",\"--package\",\"p\",\"--text\",\"<redacted>\"],\"started_at\":\"{started}\",\"exit_code\":0,\"stdout\":{},\"stderr\":{}}}\n",
        summary_json(o.stdout.as_bytes()),
        summary_json(cli_stderr.as_bytes())
    );
    assert_eq!(rec, expected);
    assert!(!rec.contains("predicted_label"));
    assert!(!rec.contains("\"beta\""));
    std::fs::remove_dir_all(&dir).ok();
    let _ = std::fs::remove_dir_all(&ws);
}

/// `--text`・`--id` の値（空白区切りと `=` 形式の両方）は記録に残らない。
#[test]
fn req36_run_record_redacts_text_and_id_values() {
    let body = "echo '{\"code\":\"ok\"}'\nexit 0";
    let cases: [(&[&str], &str); 2] = [
        (
            &[
                "--package",
                "p",
                "--text",
                "SECRET-BODY-xyz",
                "--id",
                "SECRET-ID-abc",
            ],
            "[\"fandhe-edge\",\"infer\",\"--package\",\"p\",\"--text\",\"<redacted>\",\"--id\",\"<redacted>\"]",
        ),
        (
            &["--text=SECRET2", "--id=SECRET3"],
            "[\"fandhe-edge\",\"infer\",\"--text=<redacted>\",\"--id=<redacted>\"]",
        ),
    ];
    for (i, (args, expected_cmd)) in cases.iter().enumerate() {
        let dir = record_dir(&format!("redact{i}"));
        let d = dir.to_str().unwrap();
        let o = run_with_fake_bin_args("redact", body, args, &[("FANDHE_EDGE_RECORD_DIR", d)]);
        assert_eq!(o.code, Some(0));
        let rec = only_record(&dir);
        assert!(
            rec.contains(&format!("\"command\":{expected_cmd},")),
            "{rec}"
        );
        assert!(!rec.contains("SECRET"), "{rec}");
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// 未知のオプション・位置引数は（CLI が拒否する場合でも）本文を記録に残さず伏せ字にする。
#[test]
fn req36_run_record_redacts_unknown_tokens_and_positionals() {
    let dir = record_dir("unknown");
    let d = dir.to_str().unwrap();
    let body = "echo '{\"code\":\"invalid_input\"}'\nexit 64";
    let o = run_with_fake_bin_args(
        "record-unknown",
        body,
        &[
            "SECRET-BODY-pos",
            "--bogus",
            "SECRET-VAL",
            "--bogus=SECRET-EQ",
        ],
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(64));
    let rec = only_record(&dir);
    assert!(
        rec.contains("\"command\":[\"fandhe-edge\",\"infer\",\"<redacted>\",\"<redacted>\",\"<redacted>\",\"<redacted>\"],"),
        "{rec}"
    );
    assert!(!rec.contains("SECRET"), "{rec}");
    std::fs::remove_dir_all(&dir).ok();
}

/// 等号形式（`--package=PATH`・`--input-file=PATH`）も空白区切りと同じくパス値を記録し、
/// `--text=`・`--id=` の値は伏せる（REQ-36・TASK-36.1-2）。
#[test]
fn req36_run_record_equals_form_records_path_values() {
    let dir = record_dir("equals");
    let d = dir.to_str().unwrap();
    let body = "echo '{\"code\":\"invalid_input\"}'\nexit 64";
    let o = run_with_fake_bin_args(
        "record-equals",
        body,
        &[
            "--package=pkg/a",
            "--input-file=in/b.jsonl",
            "--text=SECRET-TEXT",
            "--id=SECRET-ID",
        ],
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(64));
    let rec = only_record(&dir);
    assert!(
        rec.contains("\"command\":[\"fandhe-edge\",\"infer\",\"--package=pkg/a\",\"--input-file=in/b.jsonl\",\"--text=<redacted>\",\"--id=<redacted>\"],"),
        "{rec}"
    );
    assert!(!rec.contains("SECRET"), "{rec}");
    std::fs::remove_dir_all(&dir).ok();
}

/// パス値が UTF-8 として不正なら固定値へ置き換わり、記録は妥当な JSON のままになる。
#[test]
fn req36_run_record_invalid_utf8_path_arg_is_replaced() {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::PermissionsExt;
    let dir = record_dir("badpath");
    let bindir = record_dir("badpath-bin");
    let bin = bindir.join("fake-bin");
    std::fs::write(&bin, "#!/bin/sh\necho '{\"code\":\"ok\"}'\nexit 0\n").expect("write");
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    let bad = std::ffi::OsStr::from_bytes(b"p\xff\xfe");
    let out = Command::new("sh")
        .arg(script_path())
        .arg("--package")
        .arg(bad)
        .env("FANDHE_EDGE_BIN", &bin)
        .env("FANDHE_EDGE_RECORD_DIR", &dir)
        .stdin(Stdio::null())
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(0));
    let rec = only_record(&dir);
    assert!(
        rec.contains("\"command\":[\"fandhe-edge\",\"infer\",\"--package\",\"<invalid utf-8>\"],"),
        "{rec}"
    );
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&bindir).ok();
}

/// パス値の末尾改行（と途中の改行）は記録の command で `\n` として保持される（REQ-36・TASK-36.1-2）。
#[test]
fn req36_run_record_path_arg_keeps_trailing_newline() {
    let cases: [(&str, &str); 3] = [
        ("p\n", "p\\n"),
        ("p\n\n", "p\\n\\n"),
        ("a\nb\n", "a\\nb\\n"),
    ];
    for (i, (value, escaped)) in cases.iter().enumerate() {
        let dir = record_dir(&format!("trailnl{i}"));
        let d = dir.to_str().unwrap();
        let body = "echo '{\"code\":\"ok\"}'\nexit 0";
        let o = run_with_fake_bin_args(
            "trailnl",
            body,
            &["--package", value],
            &[("FANDHE_EDGE_RECORD_DIR", d)],
        );
        assert_eq!(o.code, Some(0));
        let rec = only_record(&dir);
        assert!(
            rec.contains(&format!(
                "\"command\":[\"fandhe-edge\",\"infer\",\"--package\",\"{escaped}\"],"
            )),
            "{rec}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// 非ゼロ終了が残り、stdout・stderr は本文でなくバイト数と sha256 だけが残る。
#[test]
fn req36_run_record_captures_nonzero_exit_and_stderr() {
    let dir = record_dir("nonzero");
    let d = dir.to_str().unwrap();
    let body = "printf 'warn: \"q\"\\tx\\n' 1>&2\necho '{\"code\":\"invalid_input\",\"message\":\"m\"}'\nexit 64";
    let o = run_with_fake_bin_args(
        "nonzero",
        body,
        &["--help"],
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(64));
    let rec = only_record(&dir);
    let started = started_at_of(&rec);
    let expected = format!(
        "{{\"schema\":\"{RECORD_SCHEMA}\",\"command\":[\"fandhe-edge\",\"infer\",\"--help\"],\"started_at\":\"{started}\",\"exit_code\":64,\"stdout\":{{\"bytes\":39,\"sha256\":\"811012849462321a2c240f5762e37bb0388f93aa9ad8e00ceb4bac1e5442e14f\"}},\"stderr\":{{\"bytes\":12,\"sha256\":\"2fc168109588af3b2a1c21a9ffaeacbe058c79716df3ce55292016d13c40d66f\"}}}}\n"
    );
    assert_eq!(rec, expected);
    // 本文（引用符付きのメッセージ・stderr の文言）は記録に残らない
    assert!(!rec.contains("warn"), "{rec}");
    assert!(!rec.contains("invalid_input"), "{rec}");
    std::fs::remove_dir_all(&dir).ok();
}

/// 置き換え後（呼び出し元が受け取る値）が記録される。
#[test]
fn req36_run_record_stores_normalized_output_after_replacement() {
    let dir = record_dir("normalized");
    let d = dir.to_str().unwrap();
    let o = run_with_fake_bin_args(
        "normalized",
        "echo '{bad}'\nexit 0",
        &["--help"],
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(70));
    let rec = only_record(&dir);
    assert!(rec.contains("\"exit_code\":70,"), "{rec}");
    assert!(
        rec.contains(&format!(
            "\"stdout\":{},",
            summary_json(o.stdout.as_bytes())
        )),
        "{rec}"
    );
    assert!(
        rec.contains(&format!("\"stderr\":{}}}", summary_json(b""))),
        "{rec}"
    );
    assert!(!rec.contains("runtime_error"), "{rec}");
    std::fs::remove_dir_all(&dir).ok();
}

/// バッチの複数行 stdout は全体で 1 つの要約（バイト数と sha256）になる。
#[test]
fn req36_run_record_batch_multiline_stdout_is_one_string() {
    let dir = record_dir("batch");
    let d = dir.to_str().unwrap();
    let body = "echo '{\"a\":1}'\necho '{\"b\":2}'\nexit 0";
    let o = run_with_fake_bin_args(
        "record-batch",
        body,
        &["--input-file", "f"],
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(0));
    let rec = only_record(&dir);
    assert!(
        rec.contains("\"command\":[\"fandhe-edge\",\"infer\",\"--input-file\",\"f\"]"),
        "{rec}"
    );
    assert!(
        rec.contains(&format!(
            "\"stdout\":{},",
            summary_json(b"{\"a\":1}\n{\"b\":2}\n")
        )),
        "{rec}"
    );
    assert!(!rec.contains("\\\"a\\\""), "{rec}");
    std::fs::remove_dir_all(&dir).ok();
}

/// UTF-8 として不正・NUL を含む stderr でも、要約（バイト数・sha256）で妥当な記録になる。
#[test]
fn req36_run_record_invalid_utf8_stderr_is_replaced() {
    let dir = record_dir("badstderr");
    let d = dir.to_str().unwrap();
    let body = "printf '\\377\\000' 1>&2\necho '{\"code\":\"ok\"}'\nexit 0";
    let o = run_with_fake_bin_args(
        "badstderr",
        body,
        &["--help"],
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(0));
    let rec = only_record(&dir);
    // 不正な UTF-8・NUL を含む生バイト列もそのままハッシュされ、記録は妥当な JSON のまま
    assert!(
        rec.contains(&format!("\"stderr\":{}}}", summary_json(b"\xff\x00"))),
        "{rec}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// 記録ディレクトリが通常ファイル・不在・symlink なら CLI を起動せず 64 で拒否する。
#[test]
fn req36_run_record_dir_not_directory_is_rejected_without_launching_cli() {
    let base = record_dir("baddir");
    let marker = base.join("launched");
    let file = base.join("plain-file");
    std::fs::write(&file, "x").expect("write");
    let link = base.join("link");
    let real = base.join("real");
    std::fs::create_dir_all(&real).expect("mkdir");
    std::os::unix::fs::symlink(&real, &link).expect("symlink");
    let missing = base.join("missing");
    let body = format!(
        "echo x >'{}'\necho '{{\"code\":\"ok\"}}'\nexit 0",
        marker.display()
    );
    // 末尾の / や /. を付けた symlink も拒否する（-L の素通り対策）
    let link_slash = PathBuf::from(format!("{}/", link.display()));
    let link_dot = PathBuf::from(format!("{}/.", link.display()));
    let link_slashes = PathBuf::from(format!("{}//", link.display()));
    for p in [
        &file,
        &missing,
        &link,
        &link_slash,
        &link_dot,
        &link_slashes,
    ] {
        let o = run_with_fake_bin_args(
            "baddir",
            &body,
            &["--help"],
            &[("FANDHE_EDGE_RECORD_DIR", p.to_str().unwrap())],
        );
        assert_eq!(o.code, Some(64), "{p:?}");
        assert_eq!(
            o.stdout,
            "{\"code\":\"invalid_input\",\"message\":\"FANDHE_EDGE_RECORD_DIR must be an existing directory\"}\n"
        );
        assert!(!o.stdout.contains(p.to_str().unwrap()));
        assert_eq!(o.stderr, "exit_code=64\n");
        assert!(!marker.exists(), "CLI must not be launched");
    }
    assert!(record_files(&real).is_empty());
    std::fs::remove_dir_all(&base).ok();
}

/// CLI の stdout・stderr に含まれる利用者由来の値は記録に残らない（security.md）。
#[test]
fn req36_run_record_does_not_store_user_derived_output() {
    let dir = record_dir("userderived");
    let d = dir.to_str().unwrap();
    let body =
        "echo 'SECRET-ERR-line' 1>&2\necho '{\"code\":\"ok\",\"id\":\"SECRET-ID-abc\"}'\nexit 0";
    let o = run_with_fake_bin_args(
        "record-userderived",
        body,
        &["--help"],
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(0));
    assert!(o.stdout.contains("SECRET-ID-abc"));
    let rec = only_record(&dir);
    assert!(!rec.contains("SECRET"), "{rec}");
    std::fs::remove_dir_all(&dir).ok();
}

/// 親ディレクトリが symlink でも、正規化した物理パスの直下にだけ記録し、
/// 検査後に記録先が差し替えられたら記録を残さず 70 にする（fail-closed）。
#[test]
fn req36_run_record_resolves_parent_symlink_and_detects_swap() {
    let base = record_dir("parentlink");
    let real = base.join("real");
    let sub = real.join("sub");
    std::fs::create_dir_all(&sub).expect("mkdir");
    let link = base.join("link");
    std::os::unix::fs::symlink(&real, &link).expect("symlink");
    // 親が symlink の経路でも、実体（real/sub）へ記録される
    let via = link.join("sub");
    let o = run_with_fake_bin_args(
        "record-parentlink",
        "echo '{\"code\":\"ok\"}'\nexit 0",
        &["--help"],
        &[("FANDHE_EDGE_RECORD_DIR", via.to_str().unwrap())],
    );
    assert_eq!(o.code, Some(0));
    assert_eq!(record_files(&sub).len(), 1);
    // 実行中に記録先（正規化済みの物理パス）が別ディレクトリへの symlink に差し替えられる
    let evil = base.join("evil");
    std::fs::create_dir_all(&evil).expect("mkdir");
    let body = format!(
        "rm -rf '{s}'\nln -s '{e}' '{s}'\necho '{{\"code\":\"ok\"}}'\nexit 0",
        s = sub.display(),
        e = evil.display()
    );
    let o = run_with_fake_bin_args(
        "record-parentlink-swap",
        &body,
        &["--help"],
        &[("FANDHE_EDGE_RECORD_DIR", via.to_str().unwrap())],
    );
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"failed to save run record\"}\n"
    );
    assert!(
        record_files(&evil).is_empty(),
        "record must not land in swapped dir"
    );
    std::fs::remove_dir_all(&base).ok();
}

/// 実行中に記録先が同じパスの別ディレクトリへ入れ替えられたら、パス文字列が同じでも
/// 記録を残さず 70 にする（ディレクトリ識別子の照合。REQ-39）。
#[test]
fn req36_run_record_detects_same_path_directory_replacement() {
    let base = record_dir("replaced");
    let dir = base.join("rec");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let moved = base.join("moved");
    let body = format!(
        "mv '{d}' '{m}'\nmkdir '{d}'\necho '{{\"code\":\"ok\"}}'\nexit 0",
        d = dir.display(),
        m = moved.display()
    );
    let o = run_with_fake_bin_args(
        "record-replaced",
        &body,
        &["--help"],
        &[("FANDHE_EDGE_RECORD_DIR", dir.to_str().unwrap())],
    );
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"failed to save run record\"}\n"
    );
    assert!(record_files(&dir).is_empty());
    assert!(record_files(&moved).is_empty());
    std::fs::remove_dir_all(&base).ok();
}

/// 保存に失敗したら記録済みを装わず runtime_error(70) にする（fail-closed）。
#[test]
fn req36_run_record_write_failure_is_fail_closed_70() {
    let dir = record_dir("writefail");
    let d = dir.to_str().unwrap();
    let body = format!("rmdir '{d}'\necho '{{\"code\":\"ok\"}}'\nexit 0");
    let o = run_with_fake_bin_args(
        "writefail",
        &body,
        &["--help"],
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"failed to save run record\"}\n"
    );
    assert_eq!(o.stderr, "exit_code=70\n");
    assert!(!dir.exists());
}

/// 引数の件数が上限（64 件）を超えたら、記録の command を切り詰めて固定の印
/// `<truncated>` を 1 度だけ付ける。推論本体の終了コード・出力は変えない
/// （REQ-39・TASK-36.1-2。資源上限）。
#[test]
fn req39_run_record_command_is_truncated_at_arg_count_limit() {
    let dir = record_dir("argcount");
    let d = dir.to_str().unwrap();
    let args: Vec<String> = (0..200).map(|i| format!("SECRET-{i}")).collect();
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let o = run_with_fake_bin_args(
        "record-argcount",
        "echo '{\"code\":\"invalid_input\"}'\nexit 64",
        &refs,
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(64));
    assert_eq!(o.stdout, "{\"code\":\"invalid_input\"}\n");
    let rec = only_record(&dir);
    let redacted = "\"<redacted>\",".repeat(64);
    let want = format!("\"command\":[\"fandhe-edge\",\"infer\",{redacted}\"<truncated>\"],");
    // 記録する引数は 64 件まで。印は末尾に 1 つだけ
    assert!(rec.contains(&want), "{rec}");
    assert_eq!(rec.matches("<truncated>").count(), 1, "{rec}");
    assert!(!rec.contains("SECRET"), "{rec}");
    std::fs::remove_dir_all(&dir).ok();
}

/// 総バイト数（4096）を超えるパス値の列、および 1 値が長すぎる値は切り詰める。
#[test]
fn req39_run_record_command_is_truncated_at_byte_limit() {
    let dir = record_dir("argbytes");
    let d = dir.to_str().unwrap();
    // 400 文字のパス値 × 12 個（各 --package。記録される総量は 4096 バイトを超える）
    let long = "p".repeat(400);
    let mut args: Vec<&str> = Vec::new();
    for _ in 0..12 {
        args.push("--package");
        args.push(&long);
    }
    let o = run_with_fake_bin_args(
        "record-argbytes",
        "echo '{\"code\":\"invalid_input\"}'\nexit 64",
        &args,
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(64));
    let rec = only_record(&dir);
    assert_eq!(rec.matches("<truncated>").count(), 1, "{rec}");
    let start = rec.find("\"command\":[").expect("command");
    let end = rec[start..].find("],\"started_at\"").expect("end") + start;
    assert!(
        end - start <= 4096 + 64,
        "command part too large: {}",
        end - start
    );
    std::fs::remove_dir_all(&dir).ok();

    // 1 値が 512 文字を超えたらその値だけ `<truncated>` に置き換える
    let dir = record_dir("argbytes-one");
    let d = dir.to_str().unwrap();
    let huge = "q".repeat(5000);
    let o = run_with_fake_bin_args(
        "record-argbytes-one",
        "echo '{\"code\":\"invalid_input\"}'\nexit 64",
        &["--package", &huge, "--help"],
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(64));
    let rec = only_record(&dir);
    assert!(
        rec.contains(
            "\"command\":[\"fandhe-edge\",\"infer\",\"--package\",\"<truncated>\",\"--help\"],"
        ),
        "{rec}"
    );
    assert!(!rec.contains("qqqq"), "{rec}");
    std::fs::remove_dir_all(&dir).ok();
}

/// 記録ファイルの作成後に記録先ディレクトリが別の場所へ移動されても、作成済みの
/// ファイルを元のディレクトリから確実に消し、70 を返す（一時名・最終名のどちらも残さない。
/// パスの再解決に依存しない）。PATH 上の `stat` シムが、記録先に何か作られた後の最初の
/// 呼び出しで移動を起こす（検査後の移動の模擬。証拠種別: テストハーネス）。
#[test]
fn req39_run_record_moved_dir_leaves_no_record_after_failure() {
    let base = record_dir("moved-after");
    let dir = base.join("rec");
    let moved = base.join("moved");
    let shim = base.join("shim");
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::create_dir_all(&shim).expect("mkdir");
    {
        use std::os::unix::fs::PermissionsExt;
        let real_stat = Command::new("sh")
            .args(["-c", "command -v stat"])
            .output()
            .expect("command -v stat");
        let real_stat = String::from_utf8_lossy(&real_stat.stdout)
            .trim()
            .to_string();
        assert!(!real_stat.is_empty());
        let stat = shim.join("stat");
        std::fs::write(
            &stat,
            format!(
                "#!/bin/sh\nif [ -d '{d}' ] && [ -n \"$(ls -A '{d}')\" ]; then /bin/mv '{d}' '{m}'; fi\nexec '{r}' \"$@\"\n",
                d = dir.display(),
                m = moved.display(),
                r = real_stat
            ),
        )
        .expect("write shim");
        std::fs::set_permissions(&stat, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    }
    let path = format!(
        "{}:{}",
        shim.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let o = run_with_fake_bin_args(
        "record-moved-after",
        "echo '{\"code\":\"ok\"}'\nexit 0",
        &["--help"],
        &[
            ("FANDHE_EDGE_RECORD_DIR", dir.to_str().unwrap()),
            ("PATH", &path),
        ],
    );
    assert_eq!(o.code, Some(70));
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"failed to save run record\"}\n"
    );
    assert!(moved.is_dir(), "shim must have moved the directory");
    let left: Vec<String> = std::fs::read_dir(&moved)
        .expect("read_dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    assert!(left.is_empty(), "leftover in moved dir: {left:?}");
    assert!(!dir.exists());
    std::fs::remove_dir_all(&base).ok();
}

/// 確定先の名前に既存ファイルがあっても上書きせず、別の名前で再試行して記録する。
/// 衝突が上限回数続いたら 70 の固定文エラーにし、一時名も残さない
/// （REQ-39・TASK-36.1-2）。PATH 上の `ln` シムが、確定先へ既存ファイルを先に作って衝突を起こす。
#[test]
fn req39_run_record_never_overwrites_existing_final_name() {
    use std::os::unix::fs::PermissionsExt;
    for (label, always) in [("collide-once", false), ("collide-always", true)] {
        let base = record_dir(label);
        let dir = base.join("rec");
        let shim = base.join("shim");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::create_dir_all(&shim).expect("mkdir");
        let real_ln = Command::new("sh")
            .args(["-c", "command -v ln"])
            .output()
            .expect("command -v ln");
        let real_ln = String::from_utf8_lossy(&real_ln.stdout).trim().to_string();
        assert!(!real_ln.is_empty());
        let marker = base.join("collided");
        let ln = shim.join("ln");
        std::fs::write(
            &ln,
            format!(
                "#!/bin/sh\nfor a in \"$@\"; do last=$a; done\nif [ '{a}' = 1 ] || [ ! -e '{m}' ]; then\n  echo PRE-EXISTING > \"$last\"; : > '{m}'\nfi\nexec '{r}' \"$@\"\n",
                a = i32::from(always),
                m = marker.display(),
                r = real_ln
            ),
        )
        .expect("write shim");
        std::fs::set_permissions(&ln, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        let path = format!(
            "{}:{}",
            shim.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let o = run_with_fake_bin_args(
            label,
            "echo '{\"code\":\"ok\"}'\nexit 0",
            &["--help"],
            &[
                ("FANDHE_EDGE_RECORD_DIR", dir.to_str().unwrap()),
                ("PATH", &path),
            ],
        );
        // 1 回の衝突は再試行で成功し、上限到達は 70 の固定文エラーに統一される
        if always {
            assert_eq!(o.code, Some(70), "{label}");
            assert_eq!(o.stdout, SAVE_FAILED_JSON, "{label}");
        } else {
            assert_eq!(o.code, Some(0), "{label}");
            assert_eq!(o.stdout, "{\"code\":\"ok\"}\n", "{label}");
        }
        let files = record_files(&dir);
        let pre: Vec<_> = files.iter().filter(|f| f.1 == "PRE-EXISTING\n").collect();
        let recs: Vec<_> = files
            .iter()
            .filter(|f| f.1.starts_with("{\"schema\""))
            .collect();
        assert!(!pre.is_empty(), "{label}: {files:?}");
        assert_eq!(
            pre.len() + recs.len(),
            files.len(),
            "{label}: leftover {files:?}"
        );
        if always {
            // 10 回すべて衝突: 既存 10 件が無傷で残り、記録は作られない
            assert_eq!(pre.len(), 10, "{files:?}");
            assert!(recs.is_empty(), "{files:?}");
        } else {
            assert_eq!(pre.len(), 1, "{files:?}");
            assert_eq!(recs.len(), 1, "{files:?}");
        }
        std::fs::remove_dir_all(&base).ok();
    }
}

const SAVE_FAILED_JSON: &str =
    "{\"code\":\"runtime_error\",\"message\":\"failed to save run record\"}\n";

/// 一時ファイルを作れない場合も、成功として返さず 70 の固定文エラーにする
/// （REQ-39・TASK-36.1-2）。記録先を読み取り専用にして作成を失敗させる。
#[test]
fn req39_run_record_temp_create_failure_is_70() {
    use std::os::unix::fs::PermissionsExt;
    let dir = record_dir("tmpfail");
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500)).expect("chmod");
    if std::fs::write(dir.join("probe"), b"x").is_ok() {
        // 書き込み権限を無視できる実行者（root 等）ではこの失敗を作れない
        std::fs::remove_file(dir.join("probe")).ok();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).ok();
        std::fs::remove_dir_all(&dir).ok();
        eprintln!("note: cannot make directory read-only for this user; test not exercised");
        return;
    }
    let o = run_with_fake_bin_args(
        "tmpfail",
        "echo '{\"code\":\"ok\"}'\nexit 0",
        &["--help"],
        &[("FANDHE_EDGE_RECORD_DIR", dir.to_str().unwrap())],
    );
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).expect("chmod");
    assert_eq!(o.code, Some(70));
    assert_eq!(o.stdout, SAVE_FAILED_JSON);
    assert!(record_files(&dir).is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

/// 書き込みに失敗（ファイルサイズ上限超過）しても 70 の固定文エラーになり、一時ファイルを残さない。
/// `ulimit -f` で記録のサイズを超える上限をかけて実行する（証拠種別: テストハーネス）。
#[test]
fn req39_run_record_write_failure_is_70_without_leftover() {
    let dir = record_dir("writefail-limit");
    let bin_dir = record_dir("writefail-limit-bin");
    let bin = bin_dir.join("fake-bin");
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(&bin, "#!/bin/sh\necho '{\"code\":\"ok\"}'\nexit 0\n").expect("write");
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    }
    // 引数 64 件で記録が 1024 バイトを超える（ulimit -f 1 は 512 または 1024 バイト）
    let mut args: Vec<String> = vec!["--help".to_string()];
    args.extend((0..63).map(|i| format!("tok{i}")));
    let out = Command::new("sh")
        .arg("-c")
        .arg("ulimit -f 1; exec sh \"$0\" \"$@\"")
        .arg(script_path())
        .args(&args)
        .env("FANDHE_EDGE_BIN", &bin)
        .env("FANDHE_EDGE_RECORD_DIR", &dir)
        .stdin(Stdio::null())
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(70));
    assert_eq!(String::from_utf8_lossy(&out.stdout), SAVE_FAILED_JSON);
    assert!(record_files(&dir).is_empty());
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&bin_dir).ok();
}

/// 確定先の名前がディレクトリ・ディレクトリを指す symlink になっても、その中へ記録を作らない
/// （ln はディレクトリの中へリンクを作るため、確定後の同一実体検証で弾き別の名前で再試行する）。
/// PATH 上の `ln` シムが、ln の直前に確定先をディレクトリ・symlink へ差し替える。
#[test]
fn req39_run_record_destination_directory_is_never_linked_into() {
    use std::os::unix::fs::PermissionsExt;
    for (label, symlink) in [("dest-dir", false), ("dest-symlink", true)] {
        let base = record_dir(label);
        let dir = base.join("rec");
        let shim = base.join("shim");
        let target = base.join("target");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::create_dir_all(&shim).expect("mkdir");
        std::fs::create_dir_all(&target).expect("mkdir");
        let real_ln = Command::new("sh")
            .args(["-c", "command -v ln"])
            .output()
            .expect("command -v ln");
        let real_ln = String::from_utf8_lossy(&real_ln.stdout).trim().to_string();
        let marker = base.join("swapped");
        let make = if symlink {
            format!("/bin/ln -s '{}' \"$last\"", target.display())
        } else {
            "mkdir \"$last\"".to_string()
        };
        let ln = shim.join("ln");
        std::fs::write(
            &ln,
            format!(
                "#!/bin/sh\nfor a in \"$@\"; do last=$a; done\nif [ ! -e '{m}' ]; then\n  {make}; : > '{m}'\nfi\nexec '{r}' \"$@\"\n",
                m = marker.display(),
                r = real_ln
            ),
        )
        .expect("write shim");
        std::fs::set_permissions(&ln, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        let path = format!(
            "{}:{}",
            shim.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let o = run_with_fake_bin_args(
            label,
            "echo '{\"code\":\"ok\"}'\nexit 0",
            &["--help"],
            &[
                ("FANDHE_EDGE_RECORD_DIR", dir.to_str().unwrap()),
                ("PATH", &path),
            ],
        );
        assert_eq!(o.code, Some(0), "{label}");
        let n_target = std::fs::read_dir(&target).expect("read_dir").count();
        assert_eq!(n_target, 0, "{label}: link created through symlink");
        // 差し替えられたディレクトリ（または symlink）の中には何も作られず、別の名前の記録が 1 件できる
        let mut regular = 0;
        for e in std::fs::read_dir(&dir).expect("read_dir") {
            let e = e.expect("entry");
            let ft = e.file_type().expect("type");
            if ft.is_dir() {
                assert_eq!(
                    std::fs::read_dir(e.path()).expect("read_dir").count(),
                    0,
                    "{label}"
                );
            } else if ft.is_file() {
                regular += 1;
            }
        }
        assert_eq!(regular, 1, "{label}");
        std::fs::remove_dir_all(&base).ok();
    }
}

/// 未設定・空文字では何も作らず出力も変えない（既定の経路の回帰）。
#[test]
fn req36_no_record_dir_creates_nothing() {
    let dir = record_dir("unset");
    let o = run_with_fake_bin_args(
        "unset",
        "echo '{\"code\":\"ok\"}'\nexit 0",
        &["--help"],
        &[("FANDHE_EDGE_RECORD_DIR", "")],
    );
    assert_eq!(o.code, Some(0));
    assert_eq!(o.stdout, "{\"code\":\"ok\"}\n");
    assert_eq!(o.stderr, "exit_code=0\n");
    assert!(record_files(&dir).is_empty());
    std::fs::remove_dir_all(&dir).ok();
}
