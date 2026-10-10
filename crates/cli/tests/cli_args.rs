//! CLI バイナリの引数解析の結合テスト（REQ-33・TASK-33.1-1）。
//!
//! 工程の実行は TASK-33.1-2（#136）で接続済み。ここでは存在しないパスを渡した解析成功コマンドが
//! `invalid_input`（64）の JSON 1 行で終わることだけを確認する（7 工程の完走は `pipeline_e2e.rs`）。

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
fn req33_parsed_command_with_missing_input_is_invalid_input() {
    let o = run(&[
        "register",
        "--definition",
        "def.json",
        "--project-dir",
        "proj",
    ]);
    assert_eq!(o.status.code(), Some(64));
    one_json_line(&o, "invalid_input");
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

/// REQ-33: 解析に成功した 7 サブコマンドはいずれも（入力が無ければ）stdout へちょうど 1 行の
/// JSON だけを出す（1 行 1 JSON の例外は成功時の `infer --input-file` のみ。TASK-33.4）。
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
        // どの工程も、存在しないパスは経路の閉じ込め（REQ-39）が invalid_input で拒否する。
        assert_eq!(o.status.code(), Some(64), "args: {args:?}");
        one_json_line(&o, "invalid_input");
    }
}

/// REQ-34・TASK-34.3・#486: 再開の口は作らない。`train` の `--resume` は（学習・`--status`・`--cancel` の
/// どれと併せても）未知オプションの `invalid_input`（64）で、作業ディレクトリに触れる前に止まる。
#[test]
fn req34_train_resume_option_is_unknown_invalid_input() {
    let cases: [&[&str]; 4] = [
        &[
            "train",
            "--project-dir",
            "p",
            "--candidate",
            "0",
            "--resume",
        ],
        &["train", "--project-dir", "p", "--all", "--resume"],
        &["train", "--project-dir", "p", "--status", "--resume"],
        &["train", "--project-dir", "p", "--cancel", "--resume"],
    ];
    for args in cases {
        let o = run(args);
        assert_eq!(o.status.code(), Some(64), "args: {args:?}");
        assert_eq!(
            String::from_utf8_lossy(&o.stdout),
            "{\"code\":\"invalid_input\",\"message\":\"unknown option for subcommand train\"}\n",
            "args: {args:?}"
        );
    }
}

/// REQ-34・REQ-33・REQ-39・TASK-34.1・#484: `train` の学習（`--candidate N`／`--all`）・`--status`・`--cancel` は
/// 三者排他で、`--status`／`--cancel` と学習専用のオプション（`--all`・`--budget-seconds`・`--smoke`・
/// `--train-seed`）の併用は、作業ディレクトリに触れる前に `invalid_input`（64）の固定 message で止まる。
#[test]
fn req34_train_operations_are_mutually_exclusive() {
    let cases: [(&[&str], &str); 13] = [
        (
            &["--status", "--cancel"],
            "option --cancel cannot be used with --status",
        ),
        (
            &["--cancel", "--status"],
            "option --cancel cannot be used with --status",
        ),
        (
            &["--status", "--cancel", "--candidate", "0"],
            "option --cancel cannot be used with --status",
        ),
        (
            &["--cancel", "--all"],
            "option --all cannot be used with --cancel",
        ),
        (
            &["--cancel", "--candidate", "0", "--all"],
            "option --all cannot be used with --cancel",
        ),
        (
            &["--cancel", "--budget-seconds", "10"],
            "option --budget-seconds cannot be used with --cancel",
        ),
        (
            &["--cancel", "--smoke"],
            "option --smoke cannot be used with --cancel",
        ),
        (
            &["--cancel", "--train-seed", "1"],
            "option --train-seed cannot be used with --cancel",
        ),
        (
            &["--status", "--all"],
            "option --all cannot be used with --status",
        ),
        (
            &["--status", "--budget-seconds", "10"],
            "option --budget-seconds cannot be used with --status",
        ),
        (
            &["--status", "--smoke"],
            "option --smoke cannot be used with --status",
        ),
        (
            &["--status", "--train-seed", "1"],
            "option --train-seed cannot be used with --status",
        ),
        (
            &["--candidate", "0", "--all"],
            "options --candidate and --all cannot be used together",
        ),
    ];
    for (extra, message) in cases {
        let mut args = vec!["train", "--project-dir", "p"];
        args.extend_from_slice(extra);
        let o = run(&args);
        assert_eq!(o.status.code(), Some(64), "args: {args:?}");
        assert_eq!(
            String::from_utf8_lossy(&o.stdout),
            format!("{{\"code\":\"invalid_input\",\"message\":\"{message}\"}}\n"),
            "args: {args:?}"
        );
    }
}
