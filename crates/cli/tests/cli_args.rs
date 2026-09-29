//! CLI バイナリの引数解析の結合テスト（REQ-33・TASK-33.1-1）。
//!
//! 工程の実行は TASK-33.1-2（#136）で接続するため、解析に成功したコマンドは
//! 現状 `runtime_error`（70）を返す。#136 でこの期待を置き換える。

use fandhe_edge_cli::args::{Subcommand, options};
use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .args(args)
        .output()
        .expect("spawn fandhe-edge")
}

fn one_json_line(o: &Output, code: &str) {
    let out = String::from_utf8_lossy(&o.stdout);
    assert_eq!(out.lines().count(), 1, "stdout: {out}");
    assert!(
        out.contains(&format!("\"code\":\"{code}\"")),
        "stdout: {out}"
    );
}

#[test]
fn req33_every_subcommand_help_lists_options_as_one_json_line() {
    for sub in Subcommand::ALL {
        let o = run(&[sub.name(), "--help"]);
        assert_eq!(o.status.code(), Some(0));
        assert!(o.stderr.is_empty());
        one_json_line(&o, "ok");
        let err = String::from_utf8_lossy(&o.stdout);
        for opt in options(sub) {
            assert!(err.contains(opt.name), "{} lacks {}", sub.name(), opt.name);
        }
    }
}

#[test]
fn req33_top_level_help_lists_seven_subcommands() {
    let o = run(&["--help"]);
    assert_eq!(o.status.code(), Some(0));
    assert!(o.stderr.is_empty());
    one_json_line(&o, "ok");
    let err = String::from_utf8_lossy(&o.stdout);
    for sub in Subcommand::ALL {
        assert!(err.contains(sub.name()));
    }
}

#[test]
fn req33_no_args_is_invalid_input() {
    let o = run(&[]);
    assert_eq!(o.status.code(), Some(64));
    one_json_line(&o, "invalid_input");
}

#[test]
fn req33_conflicting_infer_source_is_invalid_input() {
    let o = run(&[
        "infer",
        "--package",
        "p",
        "--text",
        "a",
        "--input-file",
        "b",
    ]);
    assert_eq!(o.status.code(), Some(64));
    one_json_line(&o, "invalid_input");
}

#[test]
fn req33_parsed_command_reports_not_implemented() {
    let o = run(&[
        "register",
        "--definition",
        "def.json",
        "--project-dir",
        "proj",
    ]);
    assert_eq!(o.status.code(), Some(70));
    one_json_line(&o, "runtime_error");
}

#[cfg(unix)]
#[test]
fn req33_non_utf8_argument_does_not_panic() {
    use std::os::unix::ffi::OsStrExt;
    let bad = std::ffi::OsStr::from_bytes(&[0xff]);
    let o = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .arg(bad)
        .output()
        .expect("spawn");
    assert_eq!(o.status.code(), Some(64));
    one_json_line(&o, "invalid_input");
}

/// REQ-33: 解析に成功した 7 サブコマンドはいずれも stdout へちょうど 1 行の JSON だけを出す
/// （1 行 1 JSON の例外は `infer --input-file` の実行が接続される #136 以降で、成功時のみ
/// `infer_batch` が担う。それ以外の出力形を変えていないことの回帰確認。TASK-33.4）。
#[test]
fn req33_every_parsed_subcommand_emits_exactly_one_json_line() {
    let cases: [&[&str]; 8] = [
        &["register", "--definition", "d.json", "--project-dir", "p"],
        &["inspect", "--project-dir", "p"],
        &["train", "--project-dir", "p", "--candidate", "0"],
        &["evaluate", "--project-dir", "p", "--candidate", "0"],
        &["select", "--project-dir", "p"],
        &["package", "--project-dir", "p"],
        &["infer", "--package", "p", "--text", "hello"],
        &["infer", "--package", "p", "--input-file", "in.jsonl"],
    ];
    for args in cases {
        let o = run(args);
        assert_eq!(
            o.stdout.iter().filter(|b| **b == b'\n').count(),
            1,
            "args: {args:?}"
        );
        one_json_line(&o, "runtime_error");
    }
}
