//! Bash（`sh`）経由の非対話実行の結合テスト（REQ-36・TASK-36.1-1・#149）。
//!
//! `scripts/cli-infer-noninteractive.sh` を子プロセスで起動し、終了コードと
//! stdout の JSON を具体値で照合する。証拠種別はテストハーネス（実機の
//! Claude Code Bash ツールではない）。`infer` の実推論経路は TASK-33.1-2
//! （#136）・#112・#113 が未接続のため、現時点の exit 0 経路は help のみ。
//! 実推論の exit 0 ケースはそれらの完了後にここへ追加する。
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
    let mut child = Command::new("sh")
        // 独立したプロセスグループで起動し、タイムアウト時に子孫も終了できるようにする
        .process_group(0)
        .arg(script_path())
        .args(args)
        .env("FANDHE_EDGE_BIN", env!("CARGO_BIN_EXE_fandhe-edge"))
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

/// 現時点で `infer` が exit 0 になる唯一の経路（help）。
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
/// #136 で工程が接続されたらこの期待を置き換える。
#[test]
fn req36_infer_nonzero_exit_is_propagated_via_sh() {
    let o = run_script(&["--package", "p", "--text", "a"]);
    assert_eq!(o.code, Some(70));
    let expected = expected_stdout(&ErrorReport::new(
        ExitCode::RuntimeError,
        "stage not implemented yet (TASK-33.1-2)",
    ));
    assert_eq!(o.stdout, expected);
    assert_eq!(o.stderr.lines().last(), Some("exit_code=70"));
}

/// 偽の実行ファイル（sh スクリプト）を一時ディレクトリへ作り、スクリプト経由で実行する。
fn run_with_fake_bin(name: &str, body: &str) -> Out {
    run_with_fake_bin_env(name, body, &[])
}

/// `run_with_fake_bin` に環境変数（期限の上書きなど）を追加で渡す版。
fn run_with_fake_bin_env(name: &str, body: &str, envs: &[(&str, &str)]) -> Out {
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
        .arg("--help")
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
