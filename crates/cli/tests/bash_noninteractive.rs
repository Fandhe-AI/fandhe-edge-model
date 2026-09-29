//! Bash（`sh`）経由の非対話実行の結合テスト（REQ-36・TASK-36.1-1・#149）。
//!
//! `scripts/cli-infer-noninteractive.sh` を子プロセスで起動し、終了コードと
//! stdout の JSON を具体値で照合する。証拠種別はテストハーネス（実機の
//! Claude Code Bash ツールではない）。`infer` の実推論経路は TASK-33.1-2
//! （#136）・#112・#113 が未接続のため、現時点の exit 0 経路は help のみ。
//! 実推論の exit 0 ケースはそれらの完了後にここへ追加する。
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
    run_script_env(args, &[])
}

/// `run_script` に環境変数を追加で渡す版（実行記録の検証用。TASK-36.1-2）。
fn run_script_env(args: &[&str], envs: &[(&str, &str)]) -> Out {
    let mut child = Command::new("sh")
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
            "i=0\nwhile [ $i -lt {TOTAL_CHUNKS} ]; do\n  head -c {CHUNK} /dev/zero {redirect} || exit 0\n  i=$((i+1))\n  echo $i >'{}'\ndone\nexit 0",
            counter.display()
        );
        let o = run_with_fake_bin(name, &body);
        let chunks: u64 = std::fs::read_to_string(&counter)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(u64::MAX);
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
// 証拠種別はテストハーネス（fake bin・help 経路）。cli は serde_json に依存しないため、
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

/// 期待値側の JSON 文字列エスケープ（`"`・`\`・制御文字）。
fn json_str(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
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
        "{{\"schema\":\"{RECORD_SCHEMA}\",\"command\":[\"fandhe-edge\",\"infer\",\"--help\"],\"started_at\":\"{started}\",\"exit_code\":0,\"stdout\":\"{}\",\"stderr\":\"\",\"stderr_replaced\":false}}\n",
        json_str(&help)
    );
    assert_eq!(rec, expected);
    std::fs::remove_dir_all(&dir).ok();
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
        "unknown",
        body,
        &[
            "SECRET-BODY-pos",
            "--bogus",
            "SECRET-VAL",
            "--package=SECRET-PKG",
        ],
        &[("FANDHE_EDGE_RECORD_DIR", d)],
    );
    assert_eq!(o.code, Some(64));
    let rec = only_record(&dir);
    assert!(
        rec.contains("\"command\":[\"fandhe-edge\",\"infer\",\"<redacted>\",\"<redacted>\",\"<redacted>\",\"--package=<redacted>\"],"),
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

/// 非ゼロ終了・stderr の引用符とタブがエスケープされて残る。
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
        "{{\"schema\":\"{RECORD_SCHEMA}\",\"command\":[\"fandhe-edge\",\"infer\",\"--help\"],\"started_at\":\"{started}\",\"exit_code\":64,\"stdout\":\"{{\\\"code\\\":\\\"invalid_input\\\",\\\"message\\\":\\\"m\\\"}}\\n\",\"stderr\":\"warn: \\\"q\\\"\\u0009x\\n\",\"stderr_replaced\":false}}\n"
    );
    assert_eq!(rec, expected);
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
        rec.contains(&format!("\"stdout\":\"{}\"", json_str(&o.stdout))),
        "{rec}"
    );
    assert!(
        rec.contains("\"stderr\":\"\",\"stderr_replaced\":false}"),
        "{rec}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// バッチの複数行 stdout は 1 つの文字列に収まる。
#[test]
fn req36_run_record_batch_multiline_stdout_is_one_string() {
    let dir = record_dir("batch");
    let d = dir.to_str().unwrap();
    let body = "echo '{\"a\":1}'\necho '{\"b\":2}'\nexit 0";
    let o = run_with_fake_bin_args(
        "batch",
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
        rec.contains("\"stdout\":\"{\\\"a\\\":1}\\n{\\\"b\\\":2}\\n\","),
        "{rec}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// UTF-8 として不正・NUL を含む stderr は固定文字列に置き換わる。
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
    assert!(
        rec.contains(
            "\"stderr\":\"<stderr not representable as UTF-8 text>\",\"stderr_replaced\":true}"
        ),
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
    for p in [&file, &missing, &link] {
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
