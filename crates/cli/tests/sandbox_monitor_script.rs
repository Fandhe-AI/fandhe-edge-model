//! `scripts/sandbox-monitor.sh`・`scripts/sandbox_deny_report.py`（sandbox 下の拒否ログ監視と
//! 通信拒否 0 件の自動判定）の結合テスト（REQ-38・TASK-38.1-2・#163）。
//!
//! 証拠種別: テストハーネス。偽の `log`（`fixtures/sandbox_deny_log/` の合成データを出す）・
//! 偽の launcher・偽の CLI を一時ディレクトリへ書き出して使うため、**実際の通信遮断も実際の
//! `log stream` も使っていない**。macOS 実機での sandbox 下の完走確認と拒否ログの記録は人の
//! 担当で、本テストはその証拠にならない（手順は `AGENTS.md`「実機前提テスト」）。ここで検証するのは
//! 監視の制御（開始・停止・fail-closed）と集計器の判定規則・出力契約。
//! 陽性対照（検出手段が機能することの確認）は TASK-38.2 の担当で、ここでは行わない
//! （出力の `positive_control` が `not_run` であることだけを固定する）。
//! Windows では `sh` を前提にできないため unix に限定する。

#![cfg(unix)]

use std::fs;
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// 子プロセスの上限時間（資源上限。REQ-39）。
const TIMEOUT: Duration = Duration::from_secs(60);
const PROFILE: &str = "(version 1)(allow default)(deny network*)";
const PREDICATE: &str = "process == \"kernel\" AND eventMessage CONTAINS \"deny\"";
const INFER_MARKER: &str = "MARKER_TEXT_9f3a";

static SEQ: AtomicU32 = AtomicU32::new(0);

struct Out {
    code: Option<i32>,
    stdout: String,
}

/// テストごとの作業ディレクトリと偽のコマンド群。
struct Env {
    dir: PathBuf,
    launcher: PathBuf,
    cli: PathBuf,
    log: PathBuf,
    cli_log: PathBuf,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn fixture(name: &str) -> PathBuf {
    repo_root()
        .join("fixtures")
        .join("sandbox_deny_log")
        .join(name)
}

fn write_exe(path: &Path, body: &str) {
    fs::write(path, body).expect("write");
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).expect("chmod");
}

impl Env {
    fn new() -> Self {
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("fandhe-sandbox-monitor-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir");
        let launcher = dir.join("fake-sandbox-exec");
        let cli = dir.join("fake-cli");
        let log = dir.join("fake-log");
        let cli_log = dir.join("cli.log");
        write_exe(
            &launcher,
            &format!(
                "#!/bin/sh\n\
                 [ \"$1\" = \"-p\" ] || exit 99\n\
                 [ \"$2\" = \"{PROFILE}\" ] || exit 99\n\
                 shift 2\n\
                 FAKE_SANDBOXED=1 exec \"$@\"\n"
            ),
        );
        write_exe(
            &cli,
            &format!(
                "#!/bin/sh\n\
                 [ \"${{FAKE_SANDBOXED:-}}\" = 1 ] || exit 99\n\
                 echo \"$1\" >> \"{}\"\n\
                 if [ \"${{FAKE_FAIL_STAGE:-}}\" = \"$1\" ]; then exit \"${{FAKE_FAIL_RC:-70}}\"; fi\n\
                 echo '{{\"code\":\"ok\"}}'\n\
                 exit 0\n",
                cli_log.display()
            ),
        );
        // 偽の log: 引数が `stream --style ndjson --predicate <定数>` と完全一致しなければ 99。
        // 合成 fixture を出力し、早期終了モード以外は停止（TERM）まで待つ
        write_exe(
            &log,
            &format!(
                "#!/bin/sh\n\
                 [ $# -eq 5 ] && [ \"$1\" = stream ] && [ \"$2\" = --style ] && [ \"$3\" = ndjson ] \
                 && [ \"$4\" = --predicate ] && [ \"$5\" = '{PREDICATE}' ] || exit 99\n\
                 cat \"$FAKE_STREAM_FIXTURE\"\n\
                 [ \"${{FAKE_LOG_MODE:-}}\" = early ] && exit 0\n\
                 exec sleep 300\n"
            ),
        );
        Env {
            dir,
            launcher,
            cli,
            log,
            cli_log,
        }
    }

    fn project(&self) -> PathBuf {
        self.dir.join("project")
    }

    fn out(&self) -> PathBuf {
        self.dir.join("out")
    }

    fn base_args(&self) -> Vec<String> {
        vec![
            "--definition".into(),
            self.dir.join("definition.json").display().to_string(),
            "--project-dir".into(),
            self.project().display().to_string(),
            "--out-dir".into(),
            self.out().display().to_string(),
        ]
    }

    fn run(&self, fixture_name: &str, args: &[String], envs: &[(&str, &str)]) -> Out {
        self.run_with(fixture_name, args, envs, Some(&self.log))
    }

    fn run_with(
        &self,
        fixture_name: &str,
        args: &[String],
        envs: &[(&str, &str)],
        log: Option<&Path>,
    ) -> Out {
        let script = repo_root().join("scripts").join("sandbox-monitor.sh");
        let mut cmd = Command::new("sh");
        cmd.process_group(0)
            .arg(script)
            .args(args)
            .env_remove("FANDHE_EDGE_LOG_CMD")
            .env("FANDHE_EDGE_SANDBOX_EXEC", &self.launcher)
            .env("FANDHE_EDGE_BIN", &self.cli)
            .env("FANDHE_EDGE_LOG_STREAM_WARMUP_SECS", "0")
            .env("FANDHE_EDGE_LOG_STREAM_TAIL_SECS", "0")
            .env("FAKE_STREAM_FIXTURE", fixture(fixture_name))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if let Some(l) = log {
            cmd.env("FANDHE_EDGE_LOG_CMD", l);
        }
        for (k, v) in envs {
            cmd.env(k, v);
        }
        spawn_and_collect(cmd)
    }

    fn cli_calls(&self) -> Vec<String> {
        fs::read_to_string(&self.cli_log)
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn spawn_and_collect(mut cmd: Command) -> Out {
    let mut child = cmd.spawn().expect("spawn");
    let mut so = child.stdout.take().expect("stdout");
    let pgid = child.id();
    let h_out = std::thread::spawn(move || {
        let mut s = String::new();
        so.read_to_string(&mut s).ok();
        s
    });
    let start = Instant::now();
    let status = loop {
        if let Some(st) = child.try_wait().expect("try_wait") {
            break st;
        }
        if start.elapsed() > TIMEOUT {
            Command::new("kill")
                .args(["-KILL", &format!("-{pgid}")])
                .status()
                .ok();
            child.kill().ok();
            child.wait().ok();
            h_out.join().ok();
            panic!("script timed out");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    Out {
        code: status.code(),
        stdout: h_out.join().expect("join"),
    }
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// 集計 JSON の 1 行に、指定した断片がそのまま含まれることを検査する。
fn has(o: &Out, frag: &str) {
    assert!(
        o.stdout.contains(frag),
        "missing {frag} in stdout: {}",
        o.stdout
    );
}

fn single_line(o: &Out) {
    assert_eq!(
        o.stdout.trim_end().lines().count(),
        1,
        "stdout must be one JSON line: {}",
        o.stdout
    );
}

/// 拒否ログが clean で run が成功すると、通信拒否 0 件・exit 0 になる（PoC-16 の誤検出 2 種を含む）。
#[test]
fn req38_clean_stream_reports_zero_network_denials() {
    let e = Env::new();
    let o = e.run("clean.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    single_line(&o);
    has(&o, "\"network_verdict\": \"zero_network_denials\"");
    has(&o, "\"run_exit_code\": 0");
    has(&o, "\"stream_lines\": 5");
    has(&o, "\"parsed_events\": 4");
    has(&o, "\"deny_events\": 4");
    has(&o, "\"duplicate_reports\": 2");
    has(&o, "\"network_deny_events\": 0");
    has(&o, "\"tool_network_deny_events\": 0");
    has(&o, "\"unattributed_network_deny_events\": 0");
    assert!(e.out().join("network_report.json").is_file());
    assert!(e.out().join("monitor.meta.json").is_file());
    // predicate の引用符が JSON として正しくエスケープされている（`\"` の 1 重）
    let meta = std::fs::read_to_string(e.out().join("monitor.meta.json")).unwrap();
    assert!(
        meta.contains(r#""predicate":"process == \"kernel\" AND eventMessage CONTAINS \"deny\"""#),
        "{meta}"
    );
    assert!(e.out().join("run").join("run.meta.json").is_file());
    assert_eq!(e.cli_calls().len(), 7);
}

/// プロセス名が許可リストでも PID を照合できなければ帰属不明で pending(12)（名前だけで
/// 本ツール起因と断定しない。現状の sandbox-run.sh は PID を記録しない）。
#[test]
fn req38_name_only_match_is_unattributed_not_judged_fail() {
    let e = Env::new();
    let o = e.run("tool_python.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"network_verdict\": \"unattributed_network_denials\"");
    has(&o, "\"tool_network_deny_events\": 0");
    has(&o, "\"unattributed_network_deny_events\": 1");
    let report = fs::read_to_string(e.out().join("network_report.json")).expect("report");
    assert!(report.contains("\"process\": \"python3.12\""));
    assert!(report.contains("\"operation\": \"network-outbound\""));
}

/// 重複報告（`3 duplicate reports for`）は元の 1 件と合算して発生回数（4）に数える。
#[test]
fn req38_duplicate_report_is_merged_into_network_events() {
    let e = Env::new();
    let o = e.run("tool_duplicate.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"duplicate_reports\": 3");
    has(&o, "\"network_deny_events\": 4");
    has(&o, "\"unattributed_network_deny_events\": 4");
}

/// 元の行と重複要約行が両方ある場合、同一イベントとして 1+3 回（4 回。5 回ではない）に数える。
#[test]
fn req38_duplicate_summary_with_original_line_is_not_double_counted() {
    let e = Env::new();
    let o = run_report(
        &e.dir,
        &fixture("duplicate_with_original.ndjson"),
        Some(&meta(0, T1, T2)),
        T0,
        T3,
    );
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"deny_events\": 2");
    has(&o, "\"duplicate_reports\": 3");
    has(&o, "\"network_deny_events\": 4");
    has(&o, "\"unattributed_network_deny_events\": 4");
    let report = fs::read_to_string(e.dir.join("report.json")).expect("report");
    assert_eq!(report.matches("\"attribution\"").count(), 1, "{report}");
    assert!(report.contains("\"occurrences\": 4"), "{report}");
}

/// 別イベント（PID・対象が異なる）の要約行は元の行を消費しない（過小計上の防止）。
#[test]
fn req38_duplicate_summary_of_different_event_is_counted_separately() {
    let e = Env::new();
    let stream = e.dir.join("mixed.ndjson");
    let header = fs::read_to_string(fixture("clean.ndjson")).expect("fixture");
    let header = header.lines().next().expect("header").to_string();
    let body = "{\"eventMessage\":\"Sandbox: zz(9) deny(1) network-outbound 10.0.0.1:1\"}\n\
                {\"eventMessage\":\"2 duplicate reports for Sandbox: zz(9) deny(1) network-outbound 10.0.0.2:1\"}\n";
    fs::write(&stream, format!("{header}\n{body}")).expect("write");
    let o = run_report(&e.dir, &stream, Some(&meta(0, T1, T2)), T0, T3);
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"network_deny_events\": 4");
}

/// 拒否ログの生文字列（通信先・許可リスト外のプロセス名・形式外の行の本文）は、レポートにも
/// stdout にも出さない（P0。件数・固定語彙・ダイジェストだけ）。
#[test]
fn req38_report_never_contains_raw_log_strings() {
    let e = Env::new();
    let o = run_report(
        &e.dir,
        &fixture("leak_probe.ndjson"),
        Some(&meta(0, T1, T2)),
        T0,
        T3,
    );
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"network_deny_events\": 2");
    let report = fs::read_to_string(e.dir.join("report.json")).expect("report");
    for needle in [
        "secret-host",
        "example.invalid",
        "8443",
        "PrivateAppName",
        "unformatted-secret-body",
        "secretuser",
        "/Users/",
    ] {
        assert!(!report.contains(needle), "report leaked {needle}");
        assert!(!o.stdout.contains(needle), "stdout leaked {needle}");
    }
    assert!(report.contains("\"target_digest\""));
}

/// 監視スクリプト経由でも、`network_report.json` と `monitor.meta.json` に生文字列は残らない
/// （生ログ `log_stream.ndjson` は 0600 の入力そのもので、対象外）。
#[test]
fn req38_monitor_outputs_do_not_contain_raw_log_strings() {
    let e = Env::new();
    let o = e.run("leak_probe.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    for name in ["network_report.json", "monitor.meta.json"] {
        let body = fs::read_to_string(e.out().join(name)).expect("read");
        for needle in [
            "secret-host",
            "PrivateAppName",
            "unformatted-secret-body",
            "secretuser",
        ] {
            assert!(!body.contains(needle), "{name} leaked {needle}");
        }
    }
    for needle in [
        "secret-host",
        "PrivateAppName",
        "unformatted-secret-body",
        "secretuser",
    ] {
        assert!(!o.stdout.contains(needle), "stdout leaked {needle}");
    }
}

/// `--candidates` は log stream を開始する前に 1〜16 を検証し、範囲外は 64（CLI も起動しない）。
#[test]
fn req33_candidates_range_is_validated_before_monitoring() {
    for bad in ["0", "17", "abc", "-1", "01", ""] {
        let e = Env::new();
        let mut args = e.base_args();
        args.extend(s(&["--candidates", bad]));
        let started = Instant::now();
        let o = e.run("clean.ndjson", &args, &[]);
        assert_eq!(o.code, Some(64), "candidates={bad}: {}", o.stdout);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(e.cli_calls().is_empty());
        assert!(!e.out().join("log_stream.ndjson").exists());
    }
}

/// 帰属不明のプロセスの通信拒否は合格にせず pending(12)。
#[test]
fn req38_unattributed_network_denial_is_pending() {
    let e = Env::new();
    let o = e.run("unattributed.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"network_verdict\": \"unattributed_network_denials\"");
    has(&o, "\"unattributed_network_deny_events\": 1");
}

/// 形式に合わないが deny と network を含む行は fail-closed で pending(12)。
#[test]
fn req38_unrecognized_network_deny_line_is_not_ok() {
    let e = Env::new();
    let o = e.run("unrecognized.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"unrecognized_deny_events\": 1");
}

/// JSON として読めない行があれば判定不能（70）。読めない行に拒否が隠れる可能性を排除する。
#[test]
fn req38_unparseable_line_is_undeterminable() {
    let e = Env::new();
    let o = e.run("garbage.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "\"network_verdict\": \"undeterminable\"");
}

/// ヘッダ行が無い（`log stream` が機能していない）ときは sandbox-run.sh を起動せず 70（fail-closed）。
#[test]
fn req38_missing_header_fails_closed_without_running_cli() {
    let e = Env::new();
    let o = e.run("no_header.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    assert!(e.cli_calls().is_empty());
    assert!(!e.out().join("run").exists());
}

/// log が無ければ sandbox-run.sh（CLI）を起動せず 70（監視なしで実行して証拠を装わない）。
#[test]
fn req38_missing_log_command_fails_closed_without_running_cli() {
    let e = Env::new();
    let missing = e.dir.join("no-such-log");
    let o = e.run_with("clean.ndjson", &e.base_args(), &[], Some(&missing));
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    assert!(e.cli_calls().is_empty());
    assert!(!e.out().join("run").exists());
}

/// stream が実行の前に終了していたら sandbox-run.sh を起動せず 70（fail-closed）。
#[test]
fn req38_stream_dying_early_fails_closed_without_running_cli() {
    let e = Env::new();
    let o = e.run(
        "clean.ndjson",
        &e.base_args(),
        &[("FAKE_LOG_MODE", "early")],
    );
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    assert!(e.cli_calls().is_empty());
}

/// 拒否 0 件でも run が失敗していたら終了コードを伝搬する（完走を装わない）。
#[test]
fn req38_run_failure_propagates_with_zero_denials() {
    let e = Env::new();
    let o = e.run(
        "clean.ndjson",
        &e.base_args(),
        &[("FAKE_FAIL_STAGE", "register")],
    );
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "\"network_verdict\": \"zero_network_denials\"");
    has(&o, "\"run_exit_code\": 70");
    assert_eq!(e.cli_calls(), vec!["register".to_string()]);
}

/// 上書きは test_harness として記録し、陽性対照は未実施と明示する。
#[test]
fn req38_overrides_are_recorded_as_test_harness() {
    let e = Env::new();
    let o = e.run("clean.ndjson", &e.base_args(), &[]);
    has(&o, "\"evidence_hint\": \"test_harness\"");
    has(&o, "\"log_stream_override\": true");
    has(&o, "\"positive_control\": \"not_run\"");
    assert!(!o.stdout.contains("実機"));
}

/// `--infer-text` の値とプロジェクトのパスは stdout にも出力先のどのファイルにも残らない。
#[test]
fn req38_infer_text_and_paths_not_in_outputs() {
    let e = Env::new();
    let mut args = e.base_args();
    args.extend(s(&["--infer-text", INFER_MARKER]));
    let o = e.run("clean.ndjson", &args, &[]);
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    let project = e.project().display().to_string();
    assert!(!o.stdout.contains(INFER_MARKER));
    assert!(!o.stdout.contains(&project));
    for entry in walk(&e.out()) {
        let body = fs::read_to_string(&entry).unwrap_or_default();
        assert!(!body.contains(INFER_MARKER), "{}", entry.display());
        assert!(!body.contains(&project), "{}", entry.display());
    }
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    for ent in fs::read_dir(dir).expect("read_dir") {
        let p = ent.expect("entry").path();
        if p.is_dir() {
            v.extend(walk(&p));
        } else {
            v.push(p);
        }
    }
    v
}

/// 出力先・プロジェクトの検証と未知オプションはいずれも 64（監視も起動しない）。
#[test]
fn req33_out_dir_and_project_dir_validation() {
    let e = Env::new();
    fs::create_dir_all(e.out()).expect("mkdir");
    fs::write(e.out().join("x"), "x").expect("write");
    let o = e.run("clean.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(64), "{}", o.stdout);

    let e2 = Env::new();
    let nested = e2.out().join("project");
    let args = s(&[
        "--definition",
        "d.json",
        "--project-dir",
        &nested.display().to_string(),
        "--out-dir",
        &e2.out().display().to_string(),
    ]);
    let o2 = e2.run("clean.ndjson", &args, &[]);
    assert_eq!(o2.code, Some(64), "{}", o2.stdout);

    let e3 = Env::new();
    let o3 = e3.run("clean.ndjson", &s(&["--bogus"]), &[]);
    assert_eq!(o3.code, Some(64), "{}", o3.stdout);
    assert!(e.cli_calls().is_empty() && e2.cli_calls().is_empty() && e3.cli_calls().is_empty());
}

/// 生ログは所有者のみ読める（0600）。他アプリのイベントを含むため。
#[test]
fn req38_raw_stream_file_is_owner_only() {
    let e = Env::new();
    let o = e.run("clean.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    let mode = fs::metadata(e.out().join("log_stream.ndjson"))
        .expect("stat")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600);
}

// ---- 集計器の直接テスト（`python3 -I scripts/sandbox_deny_report.py`） ----

fn run_report(dir: &Path, stream: &Path, meta: Option<&str>, start: &str, stop: &str) -> Out {
    let meta_path = dir.join("run.meta.json");
    let _ = fs::remove_file(&meta_path);
    if let Some(m) = meta {
        fs::write(&meta_path, m).expect("meta");
    }
    let mut cmd = Command::new("python3");
    cmd.process_group(0)
        .args(["-I"])
        .arg(repo_root().join("scripts").join("sandbox_deny_report.py"))
        .arg("--stream")
        .arg(stream)
        .arg("--run-meta")
        .arg(&meta_path)
        .args([
            "--monitor-started-utc",
            start,
            "--monitor-stopped-utc",
            stop,
        ])
        .arg("--report-out")
        .arg(dir.join("report.json"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    spawn_and_collect(cmd)
}

fn meta(exit_code: i32, started: &str, ended: &str) -> String {
    format!(
        "{{\"exit_code\":{exit_code},\"started_utc\":\"{started}\",\"ended_utc\":\"{ended}\",\
         \"evidence_hint\":\"requires_human_review\",\"sandbox_exec_override\":false}}"
    )
}

fn meta_with_pids(exit_code: i32, started: &str, ended: &str, pids: &str) -> String {
    let base = meta(exit_code, started, ended);
    let trimmed = base.trim_end_matches('}');
    format!("{trimmed},\"process_pids\":{pids}}}")
}

const T0: &str = "2026-01-01T00:00:00Z";
const T1: &str = "2026-01-01T00:00:10Z";
const T2: &str = "2026-01-01T00:00:20Z";
const T3: &str = "2026-01-01T00:01:00Z";

/// 正規表現の境界: 括弧を含むプロセス名・単数形の重複報告・`network-bind`・`network*`。
/// `sh(55)` は許可リストの名前かつ `process_pids` に含まれるため本ツール起因（重複 1 件を
/// 合算して 2 回）、`Foo(bar)` は帰属不明。実機の証拠にはならない。
#[test]
fn req38_report_regex_boundaries() {
    let e = Env::new();
    let o = run_report(
        &e.dir,
        &fixture("edge_cases.ndjson"),
        Some(&meta_with_pids(0, T1, T2, "[55]")),
        T0,
        T3,
    );
    assert_eq!(o.code, Some(10), "{}", o.stdout);
    has(&o, "\"network_deny_events\": 3");
    has(&o, "\"tool_network_deny_events\": 2");
    has(&o, "\"unattributed_network_deny_events\": 1");
    has(&o, "\"duplicate_reports\": 1");
    has(&o, "\"evidence_hint\": \"requires_human_review\"");
    let report = fs::read_to_string(e.dir.join("report.json")).expect("report");
    // 許可リスト外のプロセス名は生文字列でなく固定語彙 `other` で記録する
    assert!(report.contains("\"process\": \"other\""));
    assert!(!report.contains("Foo(bar)"));
    assert!(report.contains("\"pid\": 321"));
}

/// 1 行の長さが上限（64 KiB）を超えたら判定不能。
#[test]
fn req39_report_line_length_limit_is_undeterminable() {
    let e = Env::new();
    let stream = e.dir.join("long.ndjson");
    let header = fs::read_to_string(fixture("clean.ndjson")).expect("fixture");
    let header = header.lines().next().expect("header").to_string();
    let long = format!("{{\"eventMessage\":\"{}\"}}", "a".repeat(70 * 1024));
    fs::write(&stream, format!("{header}\n{long}\n")).expect("write");
    let o = run_report(&e.dir, &stream, Some(&meta(0, T1, T2)), T0, T3);
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "\"network_verdict\": \"undeterminable\"");
}

/// run.meta.json の欠落・契約外の exit_code は判定不能。
#[test]
fn req38_report_run_meta_contract_is_enforced() {
    let e = Env::new();
    let clean = fixture("clean.ndjson");
    let missing = run_report(&e.dir, &clean, None, T0, T3);
    assert_eq!(missing.code, Some(70), "{}", missing.stdout);
    let bad = run_report(&e.dir, &clean, Some(&meta(3, T1, T2)), T0, T3);
    assert_eq!(bad.code, Some(70), "{}", bad.stdout);
    has(&bad, "\"network_verdict\": \"undeterminable\"");
}

/// 監視の窓が run を覆っていない（開始が run より後）場合は判定不能。
#[test]
fn req38_report_time_window_must_cover_run() {
    let e = Env::new();
    let clean = fixture("clean.ndjson");
    let late_start = run_report(&e.dir, &clean, Some(&meta(0, T1, T2)), T2, T3);
    assert_eq!(late_start.code, Some(70), "{}", late_start.stdout);
    let early_stop = run_report(&e.dir, &clean, Some(&meta(0, T1, T3)), T0, T2);
    assert_eq!(early_stop.code, Some(70), "{}", early_stop.stdout);
    let ok = run_report(&e.dir, &clean, Some(&meta(0, T1, T2)), T0, T3);
    assert_eq!(ok.code, Some(0), "{}", ok.stdout);
}

/// PID が `process_pids` に無ければ、名前が許可リストでも帰属不明（pending）。
#[test]
fn req38_report_pid_mismatch_is_unattributed() {
    let e = Env::new();
    let o = run_report(
        &e.dir,
        &fixture("tool_python.ndjson"),
        Some(&meta_with_pids(0, T1, T2, "[1, 2]")),
        T0,
        T3,
    );
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    let o2 = run_report(
        &e.dir,
        &fixture("tool_python.ndjson"),
        Some(&meta_with_pids(0, T1, T2, "[5001]")),
        T0,
        T3,
    );
    assert_eq!(o2.code, Some(10), "{}", o2.stdout);
    has(&o2, "\"tool_network_deny_events\": 1");
}

/// `process_pids` が配列でない・負数を含む場合は判定不能。
#[test]
fn req38_report_invalid_process_pids_is_undeterminable() {
    let e = Env::new();
    for bad in ["\"x\"", "[-1]", "[true]"] {
        let o = run_report(
            &e.dir,
            &fixture("clean.ndjson"),
            Some(&meta_with_pids(0, T1, T2, bad)),
            T0,
            T3,
        );
        assert_eq!(o.code, Some(70), "{bad}: {}", o.stdout);
    }
}

/// 保持するレコードは 1000 件まで。超過分も件数には含め、切り詰めを明示する（REQ-39）。
#[test]
fn req39_report_records_are_capped_but_counts_are_exact() {
    let e = Env::new();
    let stream = e.dir.join("many.ndjson");
    let header = fs::read_to_string(fixture("clean.ndjson")).expect("fixture");
    let header = header.lines().next().expect("header").to_string();
    let line = "{\"eventMessage\":\"Sandbox: zz(9) deny(1) network-outbound 10.0.0.1:1\"}\n";
    fs::write(&stream, format!("{header}\n{}", line.repeat(1500))).expect("write");
    let o = run_report(&e.dir, &stream, Some(&meta(0, T1, T2)), T0, T3);
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"network_deny_events\": 1500");
    has(&o, "\"stream_lines\": 1501");
    let report = fs::read_to_string(e.dir.join("report.json")).expect("report");
    assert!(report.contains("\"network_denials_truncated\": true"));
    assert_eq!(report.matches("\"attribution\"").count(), 1000);
}
