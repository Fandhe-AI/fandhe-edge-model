//! `scripts/sandbox-monitor.sh`・`scripts/sandbox_deny_report.py`（sandbox 下の拒否ログ監視と
//! 通信拒否 0 件の自動判定）の結合テスト（REQ-38・TASK-38.1-2・#163）。
//!
//! 証拠種別: テストハーネス。偽の `log`（`fixtures/sandbox_deny_log/` の合成データを出す）・
//! 偽の launcher・偽の CLI を一時ディレクトリへ書き出して使うため、**実際の通信遮断も実際の
//! `log stream` も使っていない**。macOS 実機での sandbox 下の完走確認と拒否ログの記録は人の
//! 担当で、本テストはその証拠にならない（手順は `AGENTS.md`「実機前提テスト」）。ここで検証するのは
//! 監視の制御（開始・停止・fail-closed）と集計器の判定規則・出力契約。
//! 陽性対照（TASK-38.2・#164。検出手段が機能することの確認）は偽の curl と、PoC-16 実測の形
//! （`Sandbox: curl(<pid>) deny(1) network-outbound /private/var/run/mDNSResponder`）を模した
//! 偽の log の行で検査する。実機の curl・実際の遮断は使っていない（証拠種別: テストハーネス）。
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
/// SIGTERM を受けた後に fake の log が書く 1 行（停止後に遅れて届く拒否行の模擬）
const LATE_LINE: &str =
    "{\\\"eventMessage\\\": \\\"Sandbox: LateApp(888) deny(1) network-outbound 192.0.2.7:443\\\"}";
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
    curl: PathBuf,
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
        let curl = dir.join("fake-curl");
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
        // 偽の curl: 受け取る引数が監視スクリプトの定数の並びと完全一致しなければ 99。
        // 自分の PID を記録し（偽の log が拒否行を出す根拠）、FAKE_CURL_RC（既定 6。PoC-16 の
        // 実測値）で終わる
        write_exe(
            &curl,
            "#!/bin/sh\n\
             [ \"${FAKE_SANDBOXED:-}\" = 1 ] || exit 99\n\
             [ \"$*\" = \"-q --noproxy * --silent --output /dev/null --max-time 3 --connect-timeout 2 https://example.com\" ] || exit 99\n\
             echo $$ > \"$FAKE_PC_PID_FILE\"\n\
             exit \"${FAKE_CURL_RC:-6}\"\n",
        );
        write_exe(
            &cli,
            &format!(
                "#!/bin/sh\n\
                 [ \"${{FAKE_SANDBOXED:-}}\" = 1 ] || exit 99\n\
                 echo \"$1\" >> \"{}\"\n\
                 if [ \"${{FAKE_FAIL_STAGE:-}}\" = \"$1\" ]; then exit \"${{FAKE_FAIL_RC:-70}}\"; fi\n\
                 if [ \"$1\" = select ]; then\n\
                 echo '{{\"step\":\"select\",\"status\":\"ok\",\"candidate\":0}}'\n\
                 exit 0\n\
                 fi\n\
                 if [ \"$1\" = evaluate ] && [ -n \"${{FAKE_EVAL_OUT:-}}\" ]; then\n\
                 printf '%s\\n' \"$FAKE_EVAL_OUT\"\n\
                 exit 0\n\
                 fi\n\
                 if [ \"$1\" = infer ]; then\n\
                 echo '{{\"id\":\"input\",\"status\":\"ok\",\"predicted_label\":\"a\"}}'\n\
                 exit 0\n\
                 fi\n\
                 printf '{{\"step\":\"%s\",\"status\":\"ok\"}}\\n' \"$1\"\n\
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
                 [ \"${{FAKE_LOG_MODE:-}}\" != flood_err ] || head -c 300000 /dev/zero >&2\n\
                 cat \"$FAKE_STREAM_FIXTURE\"\n\
                 [ \"${{FAKE_LOG_MODE:-}}\" = early ] && exit 0\n\
                 if [ \"${{FAKE_LOG_MODE:-}}\" != pc_missing ]; then\n\
                 n=0\n\
                 while [ ! -s \"$FAKE_PC_PID_FILE\" ] && [ $n -lt 200 ]; do sleep 0.05; n=$((n+1)); done\n\
                 [ -s \"$FAKE_PC_PID_FILE\" ] && printf '{{\"eventMessage\":\"Sandbox: curl(%s) deny(1) network-outbound /private/var/run/mDNSResponder\",\"timestamp\":\"%s\"}}\\n' \"$(cat \"$FAKE_PC_PID_FILE\")\" \"$(date '+%Y-%m-%d %H:%M:%S.000000%z')\"\n\
                 fi\n\
                 if [ \"${{FAKE_LOG_MODE:-}}\" = die_mid_run ]; then\n\
                 until [ -s \"$FAKE_CLI_LOG\" ]; do sleep 0.05; done\n\
                 exit 0\n\
                 fi\n\
                 if [ \"${{FAKE_LOG_MODE:-}}\" = late_line ]; then\n\
                 trap 'printf \"%s\\n\" \"{LATE_LINE}\"; exit 0' TERM\n\
                 sleep 300 &\n\
                 wait\n\
                 fi\n\
                 exec sleep 300\n"
            ),
        );
        Env {
            dir,
            launcher,
            curl,
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
            .env("FANDHE_EDGE_CURL_CMD", &self.curl)
            .env("FAKE_PC_PID_FILE", self.dir.join("pc.pid"))
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
    has(&o, "\"stream_lines\": 6");
    has(&o, "\"parsed_events\": 5");
    has(&o, "\"deny_events\": 5");
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

/// REQ-38・#469: `--extended` は sandbox-run.sh へ引き渡され、監視窓の中で追加工程（7 + 6 回の起動）が
/// 完走して通信拒否 0 件になる。証拠種別はテストハーネス（偽の log・偽の launcher）。
#[test]
fn req38_extended_is_passed_through_the_monitor() {
    let e = Env::new();
    let mut args = e.base_args();
    args.push("--extended".to_string());
    let eval = r#"{"step":"evaluate","status":"ok","calibration":{},"abstention":{}}"#;
    let o = e.run("clean.ndjson", &args, &[("FAKE_EVAL_OUT", eval)]);
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    has(&o, "\"network_verdict\": \"zero_network_denials\"");
    assert_eq!(e.cli_calls().len(), 13);
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
    has(&o, "\"network_deny_events\": 1");
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

/// 拒否行（`deny` を含む）なのに形式を読み取れない行は、`network` を含むか否かにかかわらず
/// 判定不能(70)にする（「0 件」側へ倒さない。fail-closed）。
#[test]
fn req38_unrecognized_deny_line_is_undeterminable() {
    for name in ["unrecognized.ndjson", "unrecognized_no_network.ndjson"] {
        let e = Env::new();
        let o = e.run(name, &e.base_args(), &[]);
        assert_eq!(o.code, Some(70), "{name}: {}", o.stdout);
        has(&o, "\"network_verdict\": \"undeterminable\"");
        has(&o, "log stream contains a deny line in an unknown format");
        assert!(!o.stdout.contains("unknown-layout"), "{}", o.stdout);
    }
}

/// 拒否行ではない行（`denied` 等）は無視して件数に残し、判定を 70 にしない（判別子は語境界の
/// `deny`）。空行・壊れた JSON は従来どおり 70。
#[test]
fn req38_non_deny_line_is_ignored_and_counted() {
    let e = Env::new();
    let header = fs::read_to_string(fixture("clean.ndjson")).expect("fixture");
    let header = header.lines().next().expect("header").to_string();
    let stream = e.dir.join("ignored.ndjson");
    fs::write(
        &stream,
        format!("{header}\n{{\"eventMessage\":\"something was denied here\"}}\n"),
    )
    .expect("write");
    let o = run_report(&e.dir, &stream, Some(&meta(0, T1, T2)), T0, T3);
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"ignored_non_deny_events\": 1");
    has(&o, "\"deny_events\": 0");
    let blank = e.dir.join("blank.ndjson");
    fs::write(&blank, format!("{header}\n\n")).expect("write");
    let o2 = run_report(&e.dir, &blank, Some(&meta(0, T1, T2)), T0, T3);
    assert_eq!(o2.code, Some(70), "{}", o2.stdout);
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

/// 上書きは test_harness として記録し、陽性対照は検出済みと明示する。
#[test]
fn req38_overrides_are_recorded_as_test_harness() {
    let e = Env::new();
    let o = e.run("clean.ndjson", &e.base_args(), &[]);
    has(&o, "\"evidence_hint\": \"test_harness\"");
    has(&o, "\"log_stream_override\": true");
    has(&o, "\"positive_control\": \"detected\"");
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
    // 記録の exit_code と同じ値を「実際の終了コード」として渡す（一致するふつうの実行）
    let actual = meta
        .and_then(|m| m.split("\"exit_code\":").nth(1))
        .and_then(|r| r.split(|c: char| !c.is_ascii_digit()).next())
        .unwrap_or("0")
        .to_string();
    run_report_with_actual(dir, stream, meta, start, stop, &actual)
}

fn run_report_extra(dir: &Path, stream: &Path, meta: Option<&str>, extra: &[&str]) -> Out {
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
        .args(["--monitor-started-utc", T0, "--monitor-stopped-utc", T3])
        .args(["--run-exit-code", "0"])
        .args(extra)
        .arg("--report-out")
        .arg(dir.join("report.json"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    spawn_and_collect(cmd)
}

fn run_report_with_actual(
    dir: &Path,
    stream: &Path,
    meta: Option<&str>,
    start: &str,
    stop: &str,
    actual_exit: &str,
) -> Out {
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
        .args(["--run-exit-code", actual_exit])
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
    assert_eq!(ok.code, Some(12), "{}", ok.stdout);
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

/// 記録（run.meta.json）の exit_code と実行スクリプトの実際の終了コードが食い違う場合は、
/// 拒否 0 件でも判定不能(70)にする（完走の誤認を防ぐ。fail-closed）。
#[test]
fn req38_report_run_exit_code_mismatch_is_undeterminable() {
    let e = Env::new();
    let o = run_report_with_actual(
        &e.dir,
        &fixture("clean.ndjson"),
        Some(&meta(0, T1, T2)),
        T0,
        T3,
        "70",
    );
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "\"network_verdict\": \"undeterminable\"");
    has(
        &o,
        "run record exit_code does not match the actual exit status",
    );
    let ok = run_report_with_actual(
        &e.dir,
        &fixture("clean.ndjson"),
        Some(&meta(0, T1, T2)),
        T0,
        T3,
        "0",
    );
    assert_eq!(ok.code, Some(12), "{}", ok.stdout);
}

/// 帰属の第一の根拠は PID（`process_pids`）で、プロセス名は根拠にしない。`python3.11`・
/// `python3.13` など版が違っても、PID が採取済みなら本ツール起因になる。`process` ラベルは
/// `python3` と任意の `.N` だけを名前のまま出し、`python3-evil` や未知の名前は `other` にする。
#[test]
fn req38_attribution_is_pid_first_and_label_is_normalized() {
    let e = Env::new();
    let stream = e.dir.join("attr.ndjson");
    let header = fs::read_to_string(fixture("clean.ndjson")).expect("fixture");
    let header = header.lines().next().expect("header").to_string();
    let mut body = String::new();
    for (name, pid) in [
        ("python3.11", 4201),
        ("python3.13", 4202),
        ("python3-evil", 4203),
        ("PrivateAppName", 4204),
        ("python3.13", 9999),
    ] {
        body.push_str(&format!(
            "{{\"eventMessage\":\"Sandbox: {name}({pid}) deny(1) network-outbound 10.0.0.9:1\"}}\n"
        ));
    }
    fs::write(&stream, format!("{header}\n{body}")).expect("write");
    let o = run_report(
        &e.dir,
        &stream,
        Some(&meta_with_pids(0, T1, T2, "[4201, 4202, 4203, 4204]")),
        T0,
        T3,
    );
    assert_eq!(o.code, Some(10), "{}", o.stdout);
    has(&o, "\"tool_network_deny_events\": 4");
    // PID が採取されていなければ名前が python3.13 でも帰属不明
    has(&o, "\"unattributed_network_deny_events\": 1");
    let report = fs::read_to_string(e.dir.join("report.json")).expect("report");
    assert!(report.contains("\"process\": \"python3.11\""), "{report}");
    assert!(report.contains("\"process\": \"python3.13\""), "{report}");
    assert_eq!(
        report.matches("\"process\": \"other\"").count(),
        2,
        "{report}"
    );
    assert!(!report.contains("PrivateAppName") && !report.contains("python3-evil"));
}

/// 最終終了コードの優先順を全組み合わせで固定する（`decide()` の表）。run の終了コード
/// {0, 10, 20, 64, 70} × 拒否 {なし, 本ツール起因, 帰属不明}。run が 70 なら拒否があっても 70、
/// 拒否件数はレポートに残る。それ以外は 本ツール起因 10 > 帰属不明 12 > run の終了コード。
/// 監視の異常（stream 停止）は run が 0 でも 70。
#[test]
fn req38_final_exit_code_priority_table() {
    let e = Env::new();
    let none = fixture("clean.ndjson");
    let tool = fixture("tool_python.ndjson");
    let unattr = fixture("unattributed.ndjson");
    for run in [0, 10, 20, 64, 70] {
        for (label, stream, pids, tool_n, unattr_n, base) in [
            ("none", &none, "[1]", 0, 0, if run == 0 { 12 } else { run }),
            ("tool", &tool, "[5001]", 1, 0, 10),
            ("unattributed", &unattr, "[1]", 0, 1, 12),
        ] {
            let o = run_report(
                &e.dir,
                stream,
                Some(&meta_with_pids(run, T1, T2, pids)),
                T0,
                T3,
            );
            let expected = if run == 70 { 70 } else { base };
            assert_eq!(o.code, Some(expected), "run={run} {label}: {}", o.stdout);
            has(&o, &format!("\"tool_network_deny_events\": {tool_n}"));
            has(
                &o,
                &format!("\"unattributed_network_deny_events\": {unattr_n}"),
            );
        }
    }
    // 監視の異常は run の終了コードにかかわらず 70
    let o = run_report_extra(&e.dir, &none, Some(&meta(0, T1, T2)), &["--stream-died"]);
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "\"network_verdict\": \"undeterminable\"");
}

/// `log` の stderr は保存しない（内容を使わず、容量の問題を生じさせない）。大量に書かれても
/// 出力先にファイルは作られず、判定は通常どおり（REQ-39）。
#[test]
fn req39_log_stderr_is_not_stored() {
    let e = Env::new();
    let o = e.run(
        "clean.ndjson",
        &e.base_args(),
        &[("FAKE_LOG_MODE", "flood_err")],
    );
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    for entry in walk(&e.out()) {
        let len = fs::metadata(&entry).expect("stat").len();
        assert!(len < 100_000, "{} is {len} bytes", entry.display());
        assert!(
            !entry.to_string_lossy().contains("stderr"),
            "{}",
            entry.display()
        );
    }
}

/// 停止（SIGTERM）の後に遅れて書かれた拒否行も集計に含める。停止手順が log の終了を確認する前に
/// 集計へ進むと、この 1 件を取りこぼして「0 件」と誤判定する（REQ-38）。
#[test]
fn req38_denial_written_after_stop_signal_is_counted() {
    let e = Env::new();
    let o = e.run(
        "clean.ndjson",
        &e.base_args(),
        &[("FAKE_LOG_MODE", "late_line")],
    );
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"network_deny_events\": 1");
    has(&o, "\"unattributed_network_deny_events\": 1");
}

/// ヘッダを出して起動確認を通った後、実行中に log が exit 0 で終了した場合は、監視が抜けた
/// 時間帯があるため 0 件と判定せず 70 にする（REQ-38。終了済みの子を `kill -0` で生存と誤認しない）。
#[test]
fn req38_log_exiting_mid_run_is_undeterminable() {
    let e = Env::new();
    let cli_log = e.cli_log.display().to_string();
    let o = e.run(
        "clean.ndjson",
        &e.base_args(),
        &[
            ("FAKE_LOG_MODE", "die_mid_run"),
            ("FAKE_CLI_LOG", &cli_log),
            ("FANDHE_EDGE_LOG_STREAM_TAIL_SECS", "1"),
        ],
    );
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "\"network_verdict\": \"undeterminable\"");
    has(&o, "log stream ended before the monitoring window closed");
    assert_eq!(e.cli_calls().len(), 7);
}

/// 工程グループの PID は `pgrep -g` で列挙し、6〜7 桁の PID でも帰属が tool になる（`ps` の列幅に
/// 依存しない。REQ-38）。偽の `pgrep` が 7 桁の PID を返す。実機の `pgrep` の出力ではない。
#[test]
fn req38_seven_digit_pids_from_pgrep_are_attributed_to_tool() {
    let e = Env::new();
    let bin = e.dir.join("fakebin");
    fs::create_dir_all(&bin).expect("mkdir");
    write_exe(
        &bin.join("pgrep"),
        "#!/bin/sh\n[ \"$1\" = -g ] || exit 2\necho 1234567\necho 7654321\n",
    );
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let o = e.run("tool_bigpid.ndjson", &e.base_args(), &[("PATH", &path)]);
    assert_eq!(o.code, Some(10), "{}", o.stdout);
    has(&o, "\"tool_network_deny_events\": 1");
    let meta = fs::read_to_string(e.out().join("run").join("run.meta.json")).expect("meta");
    assert!(
        meta.contains("\"process_pids\":[1234567,7654321]"),
        "{meta}"
    );
}

/// pgrep が使えない・空を返す場合は PID を記録せず、帰属不明の拒否は pending(12)（fail-closed）。
#[test]
fn req38_pgrep_failure_leaves_denials_unattributed() {
    let e = Env::new();
    let bin = e.dir.join("fakebin");
    fs::create_dir_all(&bin).expect("mkdir");
    write_exe(&bin.join("pgrep"), "#!/bin/sh\nexit 1\n");
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let o = e.run("tool_bigpid.ndjson", &e.base_args(), &[("PATH", &path)]);
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    let meta = fs::read_to_string(e.out().join("run").join("run.meta.json")).expect("meta");
    assert!(meta.contains("\"process_pids\":[]"), "{meta}");
}

/// python3 が 3.9 未満なら、sandbox-run.sh を起動する前に判定不能(70)（REQ-38）。偽の
/// `python3` が版の検査（`version_info`）だけ失敗させる。
#[test]
fn req38_old_python3_is_undeterminable_before_run() {
    let e = Env::new();
    let bin = e.dir.join("fakebin");
    fs::create_dir_all(&bin).expect("mkdir");
    write_exe(
        &bin.join("python3"),
        "#!/bin/sh\ncase \"$*\" in *version_info*) exit 1 ;; esac\nexit 0\n",
    );
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let o = e.run("clean.ndjson", &e.base_args(), &[("PATH", &path)]);
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "python3 3.9 or newer is required for the report");
    assert!(e.cli_calls().is_empty());
}

/// 集計器は Python 3.9 の文法で構文エラーにならない（`ast.parse` の feature_version=(3, 9)。
/// 文法のみの検査で、3.10 以降の標準ライブラリ API の使用は検出できない）。
#[test]
fn req38_report_script_parses_with_python39_grammar() {
    let script = repo_root().join("scripts").join("sandbox_deny_report.py");
    let out = Command::new("python3")
        .args([
            "-c",
            "import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read(), \
             feature_version=(3, 9))",
        ])
        .arg(&script)
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn write_flood(path: &Path, pairs: bool, n: usize) {
    use std::io::Write;
    let header = fs::read_to_string(fixture("clean.ndjson")).expect("fixture");
    let header = header.lines().next().expect("header").to_string();
    let mut f = std::io::BufWriter::new(fs::File::create(path).expect("create"));
    writeln!(f, "{header}").expect("write");
    for i in 0..n {
        let ev = format!("Sandbox: zz(9) deny(1) network-outbound h{i}");
        writeln!(f, "{{\"eventMessage\":\"{ev}\"}}").expect("write");
        if pairs {
            writeln!(f, "{{\"eventMessage\":\"2 duplicate reports for {ev}\"}}").expect("write");
        }
    }
}

/// 異なるイベントと重複報告の組が大量に続いても、照合済みのキーが残らないため上限（10 万キー）に
/// 達せず完走し、発生回数は 1+2 回ずつで数えられる（REQ-39）。
#[test]
fn req39_matched_duplicate_pairs_do_not_accumulate_pending_keys() {
    let e = Env::new();
    let stream = e.dir.join("pairs.ndjson");
    write_flood(&stream, true, 100_500);
    let o = run_report(&e.dir, &stream, Some(&meta(0, T1, T2)), T0, T3);
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"network_deny_events\": 301500");
}

/// 未照合の元イベントが上限（10 万キー）を超えたら判定不能(70)にする（REQ-39）。
#[test]
fn req39_pending_key_limit_is_undeterminable() {
    let e = Env::new();
    let stream = e.dir.join("unmatched.ndjson");
    write_flood(&stream, false, 100_500);
    let o = run_report(&e.dir, &stream, Some(&meta(0, T1, T2)), T0, T3);
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "log stream exceeds the pending event limit");
}
// ---- 陽性対照（REQ-38・TASK-38.2・#164。証拠種別: テストハーネス） ----

const MDNS_DENY: &str =
    "Sandbox: curl(4242) deny(1) network-outbound /private/var/run/mDNSResponder";
const LOOPBACK_DENY: &str = "Sandbox: curl(4242) deny(1) network-outbound 127.0.0.1:9";

/// ヘッダ行 + 指定した `eventMessage` の行からなる合成ストリームを書く。
fn write_stream(dir: &Path, messages: &[&str]) -> PathBuf {
    let header = fs::read_to_string(fixture("clean.ndjson")).expect("fixture");
    let header = header.lines().next().expect("header").to_string();
    let path = dir.join("pc_stream.ndjson");
    let mut body = format!("{header}\n");
    for m in messages {
        body.push_str(&format!(
            "{{\"eventMessage\":\"{m}\",\"timestamp\":\"2026-01-01 09:00:00.500000+0900\"}}\n"
        ));
    }
    fs::write(&path, body).expect("write");
    path
}

fn control_meta(pid: &str, rc: &str, started: &str, ended: &str) -> String {
    format!(
        "{{\"pid\":{pid},\"exit_code\":{rc},\"started_utc\":\"{started}\",\"ended_utc\":\"{ended}\",\
         \"curl_override\":true,\"sandbox_exec_override\":true}}"
    )
}

fn control_meta_flags(curl: &str, launcher: &str) -> String {
    format!(
        "{{\"pid\":4242,\"exit_code\":6,\"started_utc\":\"{T0}\",\"ended_utc\":\"{T0}\",\
         \"curl_override\":{curl},\"sandbox_exec_override\":{launcher}}}"
    )
}

/// 陽性対照の記録つきで集計器を直接実行する（実行の記録は T1〜T2、監視窓は T0〜T3）。
fn run_report_control(dir: &Path, stream: &Path, run_meta: &str, control: &str) -> Out {
    let control_path = dir.join("positive_control.meta.json");
    fs::write(&control_path, control).expect("control meta");
    run_report_extra(
        dir,
        stream,
        Some(run_meta),
        &[
            "--positive-control-meta",
            control_path.to_str().expect("utf8"),
        ],
    )
}

/// 陽性対照の拒否が監視で記録されなければ、本実行（sandbox-run.sh）を起動せず 70 で止める
/// （REQ-38・TASK-38.2。陽性対照を本実行のゲートにする。検出手段が機能しない状態で
/// 「0 件」を装わない）。
#[test]
fn req38_positive_control_not_detected_stops_before_run() {
    let e = Env::new();
    let o = e.run(
        "clean.ndjson",
        &e.base_args(),
        &[("FAKE_LOG_MODE", "pc_missing")],
    );
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "positive control denial was not observed");
    has(&o, "sandbox run was not started");
    assert!(e.cli_calls().is_empty());
}

/// curl の終了コードが 0（通信が成功した＝遮断が効いていない）なら、本実行を起動せず 70
/// （REQ-38・TASK-38.2。ゲート）。
#[test]
fn req38_positive_control_curl_success_stops_before_run() {
    let e = Env::new();
    let o = e.run("clean.ndjson", &e.base_args(), &[("FAKE_CURL_RC", "0")]);
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "positive control command succeeded");
    has(&o, "sandbox run was not started");
    assert!(e.cli_calls().is_empty());
}

/// PoC-16 実測の形の拒否行が陽性対照として検出され、tool・unattributed の件数に混ざらない。
#[test]
fn req38_positive_control_detected_with_poc16_deny_line() {
    let e = Env::new();
    let o = e.run("clean.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    has(&o, "\"positive_control\": \"detected\"");
    has(&o, "\"positive_control_network_deny_events\": 1");
    has(&o, "\"tool_network_deny_events\": 0");
    has(&o, "\"unattributed_network_deny_events\": 0");
    has(&o, "\"network_deny_events\": 0");
    has(&o, "\"positive_control_exit_code\": 6");
    // 陽性対照の対象文字列は生のまま残さない（P0）
    let report = fs::read_to_string(e.out().join("network_report.json")).expect("report");
    assert!(!report.contains("mDNSResponder"), "{report}");
    assert!(
        report.contains("\"attribution\": \"positive_control\""),
        "{report}"
    );
}

/// 陽性対照は sandbox-run.sh より前に、同じ launcher・プロファイルで実行される。
#[test]
fn req38_positive_control_runs_before_sandbox_run() {
    let e = Env::new();
    let o = e.run("clean.ndjson", &e.base_args(), &[]);
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    assert_eq!(e.cli_calls().len(), 7);
    // 偽の launcher はプロファイルが違えば 99 を返し、偽の curl は launcher 経由でなければ 99 を
    // 返す。PID が記録されていれば、定数のプロファイルと引数で実行された
    assert!(e.dir.join("pc.pid").is_file());
    let pc = fs::read_to_string(e.out().join("positive_control.meta.json")).expect("pc meta");
    let run = fs::read_to_string(e.out().join("run").join("run.meta.json")).expect("run meta");
    let field = |s: &str, key: &str| -> String {
        s.split(&format!("\"{key}\":\""))
            .nth(1)
            .and_then(|r| r.split('"').next())
            .unwrap_or_default()
            .to_string()
    };
    let control_ended = field(&pc, "ended_utc");
    let run_started = field(&run, "started_utc");
    assert!(
        !control_ended.is_empty() && !run_started.is_empty(),
        "{pc} {run}"
    );
    assert!(
        control_ended <= run_started,
        "{control_ended} > {run_started}"
    );
    let mode = fs::metadata(e.out().join("positive_control.meta.json"))
        .expect("stat")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600);
}

/// curl が無い・実行できないときは sandbox-run.sh を起動せず 70（fail-closed）。
#[test]
fn req38_missing_curl_does_not_start_run() {
    let e = Env::new();
    let o = e.run(
        "clean.ndjson",
        &e.base_args(),
        &[("FANDHE_EDGE_CURL_CMD", "/nonexistent/fake-curl")],
    );
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "curl not found or not executable");
    assert!(e.cli_calls().is_empty());
}

/// 集計器: 陽性対照の PID の拒否は、PoC-16 の mDNSResponder の形・ループバックの形のどちらでも
/// `positive_control` に分類される。
#[test]
fn req38_report_classifies_control_pid_denials_as_positive_control() {
    let e = Env::new();
    for deny in [MDNS_DENY, LOOPBACK_DENY] {
        let stream = write_stream(&e.dir, &[deny]);
        let o = run_report_control(
            &e.dir,
            &stream,
            &meta_with_pids(0, T1, T2, "[5001]"),
            &control_meta("4242", "6", T0, T0),
        );
        assert_eq!(o.code, Some(0), "{}", o.stdout);
        has(&o, "\"positive_control\": \"detected\"");
        has(&o, "\"positive_control_network_deny_events\": 1");
        has(&o, "\"unattributed_network_deny_events\": 0");
    }
    // 陽性対照の PID 以外の拒否は従来どおり帰属不明で pending
    let stream = write_stream(
        &e.dir,
        &[
            MDNS_DENY,
            "Sandbox: zz(9) deny(1) network-outbound 10.0.0.1:1",
        ],
    );
    let o = run_report_control(
        &e.dir,
        &stream,
        &meta_with_pids(0, T1, T2, "[5001]"),
        &control_meta("4242", "6", T0, T0),
    );
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"positive_control_network_deny_events\": 1");
    has(&o, "\"unattributed_network_deny_events\": 1");
}

/// 陽性対照の PID が `process_pids` に含まれる（PID 再利用で帰属が曖昧）なら判定不能 70。
#[test]
fn req38_report_control_pid_overlapping_run_pids_is_undeterminable() {
    let e = Env::new();
    let stream = write_stream(&e.dir, &[MDNS_DENY]);
    let o = run_report_control(
        &e.dir,
        &stream,
        &meta_with_pids(0, T1, T2, "[4242]"),
        &control_meta("4242", "6", T0, T0),
    );
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "positive control pid overlaps with the run processes");
}

/// 時刻の順序（監視開始 <= 対照開始 <= 対照終了 <= run 開始）が崩れていたら判定不能 70。
#[test]
fn req38_report_control_time_order_violation_is_undeterminable() {
    let e = Env::new();
    let stream = write_stream(&e.dir, &[MDNS_DENY]);
    for (cs, ce) in [(T1, T0), (T2, T2), (T0, T2)] {
        let o = run_report_control(
            &e.dir,
            &stream,
            &meta(0, T1, T2),
            &control_meta("4242", "6", cs, ce),
        );
        assert_eq!(o.code, Some(70), "{cs} {ce}: {}", o.stdout);
        has(
            &o,
            "positive control is not ordered inside the monitoring window",
        );
    }
}

/// 陽性対照の記録の型・値が契約外なら判定不能 70。
#[test]
fn req38_report_invalid_control_meta_is_undeterminable() {
    let e = Env::new();
    let stream = write_stream(&e.dir, &[MDNS_DENY]);
    for bad in [
        control_meta("\"4242\"", "6", T0, T0),
        control_meta("0", "6", T0, T0),
        control_meta("true", "6", T0, T0),
        control_meta("4242", "\"6\"", T0, T0),
        control_meta("4242", "6", "yesterday", T0),
        "[]".to_string(),
        "not json".to_string(),
    ] {
        let o = run_report_control(&e.dir, &stream, &meta(0, T1, T2), &bad);
        assert_eq!(o.code, Some(70), "{bad}: {}", o.stdout);
        has(&o, "\"network_verdict\": \"undeterminable\"");
        has(&o, "\"positive_control\": \"not_evaluated\"");
    }
}

/// 陽性対照の記録が無い単体の再集計では、0 件でも終了コード 0 を返さず pending(12)。
#[test]
fn req38_report_without_positive_control_never_returns_ok() {
    let e = Env::new();
    let o = run_report(
        &e.dir,
        &fixture("clean.ndjson"),
        Some(&meta(0, T1, T2)),
        T0,
        T3,
    );
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"positive_control\": \"not_run\"");
    has(
        &o,
        "no network denials were observed but the positive control was not run",
    );
}

/// curl・launcher のどちらかを差し替えた陽性対照は、本物の launcher・log でも test_harness（REQ-38）。
/// 差し替えなし（false / false）のときだけ requires_human_review。
#[test]
fn req38_control_override_flags_set_test_harness_hint() {
    let e = Env::new();
    let stream = write_stream(&e.dir, &[MDNS_DENY]);
    for (curl, launcher, hint) in [
        ("true", "false", "test_harness"),
        ("false", "true", "test_harness"),
        ("true", "true", "test_harness"),
        ("false", "false", "requires_human_review"),
    ] {
        let o = run_report_control(
            &e.dir,
            &stream,
            &meta_with_pids(0, T1, T2, "[5001]"),
            &control_meta_flags(curl, launcher),
        );
        assert_eq!(o.code, Some(0), "{curl}/{launcher}: {}", o.stdout);
        has(&o, &format!("\"evidence_hint\": \"{hint}\""));
    }
}

/// 上書きフラグが bool でない・欠落している記録は判定不能 70。
#[test]
fn req38_control_override_flags_must_be_bool() {
    let e = Env::new();
    let stream = write_stream(&e.dir, &[MDNS_DENY]);
    for bad in [
        control_meta_flags("\"true\"", "false"),
        control_meta_flags("false", "1"),
        format!("{{\"pid\":4242,\"exit_code\":6,\"started_utc\":\"{T0}\",\"ended_utc\":\"{T0}\"}}"),
    ] {
        let o = run_report_control(&e.dir, &stream, &meta(0, T1, T2), &bad);
        assert_eq!(o.code, Some(70), "{bad}: {}", o.stdout);
        has(&o, "\"positive_control\": \"not_evaluated\"");
    }
}
/// 陽性対照と同じ PID でも、イベント時刻が対照の実行区間の外なら陽性対照にしない
/// （PID 再利用で別プロセスの拒否が tool・unattributed から消えない。REQ-38・TASK-38.2・#164）。
#[test]
fn req38_report_control_pid_outside_control_window_is_not_positive_control() {
    let e = Env::new();
    let header = fs::read_to_string(fixture("clean.ndjson")).expect("fixture");
    let header = header.lines().next().expect("header").to_string();
    let line = |ts: &str| format!("{{\"eventMessage\":\"{MDNS_DENY}\",\"timestamp\":\"{ts}\"}}\n");
    // 対照の区間は T0（= 2026-01-01 09:00:00+0900）。区間内 1 件と、5 分後の同じ PID 1 件
    let body = format!(
        "{header}\n{}{}",
        line("2026-01-01 09:00:00.500000+0900"),
        line("2026-01-01 09:05:00.000000+0900"),
    );
    let stream = e.dir.join("pc_ts_stream.ndjson");
    fs::write(&stream, body).expect("write");
    let o = run_report_control(
        &e.dir,
        &stream,
        &meta_with_pids(0, T1, T2, "[5001]"),
        &control_meta("4242", "6", T0, T0),
    );
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"positive_control\": \"detected\"");
    has(&o, "\"positive_control_network_deny_events\": 1");
    has(&o, "\"unattributed_network_deny_events\": 1");
}

/// 陽性対照の記録があり、同じ PID の拒否行に timestamp が無ければ判定不能 70
/// （PID 再利用を時刻で区別できないため。REQ-38・TASK-38.2・#164）。
#[test]
fn req38_report_control_pid_line_without_timestamp_is_undeterminable() {
    let e = Env::new();
    let header = fs::read_to_string(fixture("clean.ndjson")).expect("fixture");
    let header = header.lines().next().expect("header").to_string();
    let stream = e.dir.join("pc_no_ts_stream.ndjson");
    fs::write(
        &stream,
        format!("{header}\n{{\"eventMessage\":\"{MDNS_DENY}\"}}\n"),
    )
    .expect("write");
    let o = run_report_control(
        &e.dir,
        &stream,
        &meta_with_pids(0, T1, T2, "[5001]"),
        &control_meta("4242", "6", T0, T0),
    );
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "positive control candidate deny line has no timestamp");
}

/// 照合済みの要約行は、自身の時刻が区間外でも元イベントの帰属（陽性対照）で数える。
#[test]
fn req38_report_duplicate_summary_inherits_original_attribution() {
    let e = Env::new();
    let header = fs::read_to_string(fixture("clean.ndjson")).expect("fixture");
    let header = header.lines().next().expect("header").to_string();
    let line =
        |msg: &str, ts: &str| format!("{{\"eventMessage\":\"{msg}\",\"timestamp\":\"{ts}\"}}\n");
    let dup = format!("2 duplicate reports for {MDNS_DENY}");
    let body = format!(
        "{header}\n{}{}",
        line(MDNS_DENY, "2026-01-01 09:00:00.500000+0900"),
        line(&dup, "2026-01-01 09:05:00.000000+0900"),
    );
    let stream = e.dir.join("pc_dup_stream.ndjson");
    fs::write(&stream, body).expect("write");
    let o = run_report_control(
        &e.dir,
        &stream,
        &meta_with_pids(0, T1, T2, "[5001]"),
        &control_meta("4242", "6", T0, T0),
    );
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    has(&o, "\"positive_control_network_deny_events\": 3");
    has(&o, "\"unattributed_network_deny_events\": 0");
}

/// timestamp が文字列でない・形式外のイベントは判定不能 70（fail-closed）。
#[test]
fn req38_report_invalid_event_timestamp_is_undeterminable() {
    let e = Env::new();
    let header = fs::read_to_string(fixture("clean.ndjson")).expect("fixture");
    let header = header.lines().next().expect("header").to_string();
    for ts in ["\"yesterday\"", "12345"] {
        let body = format!("{header}\n{{\"eventMessage\":\"{MDNS_DENY}\",\"timestamp\":{ts}}}\n");
        let stream = e.dir.join("pc_bad_ts.ndjson");
        fs::write(&stream, body).expect("write");
        let o = run_report_control(
            &e.dir,
            &stream,
            &meta_with_pids(0, T1, T2, "[5001]"),
            &control_meta("4242", "6", T0, T0),
        );
        assert_eq!(o.code, Some(70), "{ts}: {}", o.stdout);
        has(&o, "log stream event timestamp is invalid");
    }
}

/// curl・launcher の上書きは、stream-overflow で早期に打ち切られる経路でも
/// `evidence_hint` を test_harness にする（契約: curl override はすべて test_harness）。
#[test]
fn req38_control_override_hint_survives_early_abort() {
    let e = Env::new();
    let stream = write_stream(&e.dir, &[MDNS_DENY]);
    let control_path = e.dir.join("positive_control.meta.json");
    fs::write(&control_path, control_meta_flags("true", "false")).expect("control meta");
    let o = run_report_extra(
        &e.dir,
        &stream,
        Some(&meta_with_pids(0, T1, T2, "[5001]")),
        &[
            "--positive-control-meta",
            control_path.to_str().expect("utf8"),
            "--stream-overflow",
        ],
    );
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "\"evidence_hint\": \"test_harness\"");
}
// ---- `System Policy:` 形式の拒否行（Issue #331。合成データによるテストハーネス） ----

/// 非通信操作の `System Policy:` 拒否行（他プロセスの `file-read-data`）が監視窓に入っても、
/// 判定不能(70)にならず run の終了コード（0）を伝搬する（REQ-38・#331）。
#[test]
fn req38_system_policy_non_network_deny_is_ignored_not_undeterminable() {
    let e = Env::new();
    // 陽性対照の拒否（detected のため必要）の後ろに、fixture と同形の非通信行を置く
    let stream = write_stream(
        &e.dir,
        &[
            MDNS_DENY,
            "System Policy: SomeOtherApp(777) deny(1) file-read-data /synthetic/path/other-app.dat",
        ],
    );
    let o = run_report_control(
        &e.dir,
        &stream,
        &meta(0, T1, T2),
        &control_meta("4242", "6", T0, T0),
    );
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    has(&o, "\"deny_events\": 2");
    has(&o, "\"network_deny_events\": 0");
    has(&o, "\"unattributed_network_deny_events\": 0");
}

/// `System Policy:` の `network*` 拒否は `Sandbox:` と同じ規則で帰属・件数に反映される。
/// 元の行 1 + 重複要約 2 で発生回数 3（二重計上しない）。PID 一致で judged_fail(10)、
/// 不一致で pending(12)、陽性対照 PID なら positive_control（REQ-38・#331）。
#[test]
fn req38_system_policy_network_deny_is_attributed_like_sandbox() {
    let e = Env::new();
    let o = run_report(
        &e.dir,
        &fixture("system_policy_network.ndjson"),
        Some(&meta_with_pids(0, T1, T2, "[5001]")),
        T0,
        T3,
    );
    assert_eq!(o.code, Some(10), "{}", o.stdout);
    has(&o, "\"deny_events\": 2");
    has(&o, "\"network_deny_events\": 3");
    has(&o, "\"tool_network_deny_events\": 3");
    let o = run_report(
        &e.dir,
        &fixture("system_policy_network.ndjson"),
        Some(&meta_with_pids(0, T1, T2, "[1]")),
        T0,
        T3,
    );
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"unattributed_network_deny_events\": 3");
    let stream = write_stream(
        &e.dir,
        &["System Policy: curl(4242) deny(1) network-outbound 192.0.2.9:443"],
    );
    let o = run_report_control(
        &e.dir,
        &stream,
        &meta_with_pids(0, T1, T2, "[5001]"),
        &control_meta("4242", "6", T0, T0),
    );
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    has(&o, "\"positive_control_network_deny_events\": 1");
}

/// 接頭辞が違う行同士は同一イベントとして照合しない（`Sandbox:` の元の行 + `System Policy:` の
/// 要約行は照合されず 1 + 1 + 2 回の 4 回に数える。過大側＝
/// fail-closed 方向。REQ-38・#331）。
#[test]
fn req38_prefix_mismatch_duplicate_is_not_merged() {
    let e = Env::new();
    let stream = write_stream(
        &e.dir,
        &[
            "Sandbox: zz(9) deny(1) network-outbound 10.0.0.1:1",
            "2 duplicate reports for System Policy: zz(9) deny(1) network-outbound 10.0.0.1:1",
        ],
    );
    let o = run_report(&e.dir, &stream, Some(&meta(0, T1, T2)), T0, T3);
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    has(&o, "\"network_deny_events\": 4");
}

/// `System Policy:` で始まるが構造を欠く deny 行は判定不能(70)のまま。生文字列は出さない
/// （REQ-38・#331）。
#[test]
fn req38_system_policy_unrecognized_deny_line_is_undeterminable() {
    let e = Env::new();
    let o = run_report(
        &e.dir,
        &fixture("system_policy_unrecognized.ndjson"),
        Some(&meta(0, T1, T2)),
        T0,
        T3,
    );
    assert_eq!(o.code, Some(70), "{}", o.stdout);
    has(&o, "log stream contains a deny line in an unknown format");
    assert!(!o.stdout.contains("unknown-layout"), "{}", o.stdout);
}

/// `System Policy:` 形式でもレポート・stdout に通信先・プロセス名の生文字列を出さない（P0。#331）。
#[test]
fn req38_system_policy_report_never_contains_raw_log_strings() {
    let e = Env::new();
    let o = run_report(
        &e.dir,
        &fixture("system_policy_network.ndjson"),
        Some(&meta(0, T1, T2)),
        T0,
        T3,
    );
    assert_eq!(o.code, Some(12), "{}", o.stdout);
    let report = fs::read_to_string(e.dir.join("report.json")).expect("report");
    for needle in ["secret-host", "example.invalid", "8443", "PrivateAppName"] {
        assert!(!report.contains(needle), "report leaked {needle}");
        assert!(!o.stdout.contains(needle), "stdout leaked {needle}");
    }
}
