#![cfg(unix)]
//! 実バイナリ `fandhe-edge` と実 trainer（MLX CPU）で、`scripts/sandbox-monitor.sh` →
//! `scripts/sandbox-run.sh` の全チェーンを通して 7 工程が完走し、本ツール起因の通信拒否が
//! 0 件と判定されることの結合テスト（REQ-38・TASK-38.1・#161）。
//!
//! # 役割分担
//!
//! `sandbox_run_script.rs`・`sandbox_monitor_script.rs` は偽の CLI を使い、`pipeline_e2e.rs` は
//! 偽の学習ワーカーでスクリプトを通らない。本ファイルは「実 CLI＋実 trainer＋両スクリプト」を
//! 通す唯一のテストで、学習・推論が実際に完走する経路がスクリプトの判定と接続することを確かめる。
//!
//! # 証拠の種別
//!
//! テストハーネス。**実際の通信遮断も実際の `log stream` も使っていない**（launcher は
//! プロファイル文字列を検査して素通しする偽物、`log` は合成 fixture を出す偽物）。学習データは
//! 合成データ・CPU のみ。macOS 実機の sandbox 下での完走確認と拒否ログの記録は人の担当で、
//! 本テストはその証拠にならない（手順は `docs/design/sandbox-offline-check-procedure.md`）。
//! 陽性対照（TASK-38.2・#164）は偽の curl と偽の `log` の拒否行で通し、`detected` を確かめる。
//! 2 経路を通す: 評価なし（`--smoke`・`fixtures/sandbox_run/`。`evaluate` は `skipped`）と、
//! 評価あり（`--smoke` なし・`fixtures/sandbox_run_eval/`。凍結した評価データへ 1 回だけ適用して
//! `evaluate` が `ok`。#314・#348）。評価ありの実機確認は未実施（人の担当）。
//!
//! # 既定のテスト集合から分離する理由（`.claude/rules/ci.md`）
//!
//! 必要な環境（`make py-sync` で同期した `trainer/.venv` と MLX CPU）が `rust-ci` の runner に無い
//! 新規テストで、既定集合から移したものではない。`make test-trainer-integration` が `--ignored`
//! で実行し、`python-ci`（macos-14 arm64）とローカルの `make ci` で実際に走る。
//!
//! # テスト名と Makefile の同期
//!
//! `Makefile` の `test-trainer-integration` は `--exact` でテスト名を指定し `1 passed` を検査する。
//! テスト名を変えるときは Makefile も更新すること。

use std::fs;
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// 子プロセスの上限時間（資源上限。REQ-39）。実 trainer の学習（評価ありは smoke なしの c1・30 epochs）を含む。
const TIMEOUT: Duration = Duration::from_secs(900);
const PROFILE: &str = "(version 1)(allow default)(deny network*)";
const PREDICATE: &str = "process == \"kernel\" AND eventMessage CONTAINS \"deny\"";
const INFER_MARKER: &str = "MARKER_TEXT_9f3a";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root")
}

fn write_exe(path: &Path, body: &str) {
    fs::write(path, body).expect("write");
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).expect("chmod");
}

/// テスト固有の作業ディレクトリ（Drop で削除）。cwd として子を起動する（経路の閉じ込め。REQ-39）。
struct Work(PathBuf);

impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// 子を独立プロセスグループで起動し、上限時間内に stdout を回収する。超過時はグループごと KILL。
fn run_with_timeout(mut cmd: Command) -> (Option<i32>, String) {
    cmd.process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = cmd.spawn().expect("spawn");
    let mut so = child.stdout.take().expect("stdout");
    let pgid = child.id();
    let reader = std::thread::spawn(move || {
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
            reader.join().ok();
            panic!("sandbox chain timed out");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    (status.code(), reader.join().expect("join"))
}

/// 1 回のチェーン実行の結果。`_work` は Drop で作業ディレクトリを削除する。
struct Chain {
    code: Option<i32>,
    stdout: String,
    meta: String,
    dir: PathBuf,
    expected_deny: usize,
    _work: Work,
}

/// `sandbox-monitor.sh` を偽 launcher・偽 `log`・偽 curl の下で実 CLI・実 trainer に対して実行する。
/// `case` は作業ディレクトリ名の接尾辞（同一プロセス内で複数テストが並走しても衝突させない）。
/// `fixture` の `files` を作業ディレクトリへコピーし、`smoke` のときだけ `--smoke` を渡す。
fn run_chain(case: &str, fixture: &str, files: &[&str], smoke: bool) -> Chain {
    let root = repo_root();
    let dir =
        std::env::temp_dir().join(format!("fandhe-sandbox-real-{case}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("mkdir");
    let dir = dir.canonicalize().expect("canonicalize");
    let work = Work(dir.clone());

    // fixture は原本を変更しないようコピーして使う
    for name in files {
        fs::copy(
            root.join("fixtures").join(fixture).join(name),
            dir.join(name),
        )
        .expect("copy");
    }
    let clean = root.join("fixtures/sandbox_deny_log/clean.ndjson");
    let clean_text = fs::read_to_string(&clean).expect("clean fixture");
    // ヘッダ行を除く拒否行の件数（監視が実際にログを読んだことの確認に使う）
    let expected_deny = clean_text.lines().count() - 1;

    let launcher = dir.join("fake-sandbox-exec");
    write_exe(
        &launcher,
        &format!(
            "#!/bin/sh\n[ \"$1\" = \"-p\" ] || exit 99\n[ \"$2\" = \"{PROFILE}\" ] || exit 99\nshift 2\nexec \"$@\"\n"
        ),
    );
    // 偽 log は stop（SIGTERM）まで生存させる必要がある。寿命が TIMEOUT（900 秒）より短いと、
    // 実 trainer が長引いた場合に監視が「stop 前に log が死んだ」無効な窓として
    // runtime_error を返すため、TIMEOUT を超える 1200 秒とする。
    let log = dir.join("fake-log");
    write_exe(
        &log,
        &format!(
            "#!/bin/sh\n[ $# -eq 5 ] && [ \"$1\" = stream ] && [ \"$2\" = --style ] && [ \"$3\" = ndjson ] \
             && [ \"$4\" = --predicate ] && [ \"$5\" = '{PREDICATE}' ] || exit 99\n\
             cat \"{}\"\n\
             n=0\n\
             while [ ! -s \"$FAKE_PC_PID_FILE\" ] && [ $n -lt 200 ]; do sleep 0.05; n=$((n+1)); done\n\
             [ -s \"$FAKE_PC_PID_FILE\" ] && printf '{{\"eventMessage\":\"Sandbox: curl(%s) deny(1) network-outbound /private/var/run/mDNSResponder\",\"timestamp\":\"%s\"}}\\n' \"$(cat \"$FAKE_PC_PID_FILE\")\" \"$(date '+%Y-%m-%d %H:%M:%S.000000%z')\"\n\
             exec sleep 1200\n",
            clean.display()
        ),
    );
    // 偽 curl: 実際の通信は行わない（陽性対照に実 curl を使うと unsandboxed の偽 launcher の下で
    // 実通信になるため）。PID を記録して PoC-16 実測と同じ終了コード 6 で終わる
    let curl = dir.join("fake-curl");
    write_exe(
        &curl,
        "#!/bin/sh\necho $$ > \"$FAKE_PC_PID_FILE\"\nexit 6\n",
    );

    let mut cmd = Command::new("sh");
    cmd.current_dir(&dir)
        .arg(root.join("scripts/sandbox-monitor.sh"))
        .args(["--definition", "definition.json"])
        .args(["--project-dir", "project"])
        .args(["--out-dir", "out"])
        .args(["--candidates", "1"]);
    if smoke {
        cmd.arg("--smoke");
    }
    cmd.args(["--infer-text", INFER_MARKER])
        .env("FANDHE_EDGE_SANDBOX_EXEC", &launcher)
        .env("FANDHE_EDGE_LOG_CMD", &log)
        .env("FANDHE_EDGE_CURL_CMD", &curl)
        .env("FAKE_PC_PID_FILE", dir.join("pc.pid"))
        .env("FANDHE_EDGE_BIN", env!("CARGO_BIN_EXE_fandhe-edge"))
        .env("FANDHE_EDGE_TRAINER_DIR", root.join("trainer"))
        .env("FANDHE_EDGE_LOG_STREAM_WARMUP_SECS", "0")
        .env("FANDHE_EDGE_LOG_STREAM_TAIL_SECS", "0");
    let (code, stdout) = run_with_timeout(cmd);
    let meta = fs::read_to_string(dir.join("out/run/run.meta.json")).unwrap_or_default();
    Chain {
        code,
        stdout,
        meta,
        dir,
        expected_deny,
        _work: work,
    }
}

/// 両経路に共通の確認（完走・0 件判定・陽性対照・7 工程の順序・漏えい検査）。
/// `evaluate_status` は evaluate 工程の期待 status（評価データなしは `skipped`、ありは `ok`）。
/// evaluate 以外の工程は status が null（`skipped` を別工程に見逃さない。REQ-17）。
fn assert_chain_completed(chain: &Chain, evaluate_status: &str) {
    let Chain {
        code,
        stdout,
        meta,
        dir,
        expected_deny,
        ..
    } = chain;
    assert_eq!(*code, Some(0), "{stdout}");
    assert_eq!(stdout.trim_end().lines().count(), 1, "{stdout}");
    for frag in [
        "\"code\": \"ok\"".to_string(),
        "\"network_verdict\": \"zero_network_denials\"".to_string(),
        "\"run_exit_code\": 0".to_string(),
        "\"tool_network_deny_events\": 0".to_string(),
        "\"unattributed_network_deny_events\": 0".to_string(),
        "\"network_deny_events\": 0".to_string(),
        // 陽性対照の拒否行（1 行）を含む
        format!("\"deny_events\": {}", expected_deny + 1),
        "\"positive_control\": \"detected\"".to_string(),
        "\"positive_control_network_deny_events\": 1".to_string(),
        "\"evidence_hint\": \"test_harness\"".to_string(),
        "\"log_stream_override\": true".to_string(),
    ] {
        assert!(stdout.contains(&frag), "missing {frag} in: {stdout}");
    }

    assert!(meta.contains("\"sandbox_exec_override\":true"), "{meta}");
    assert!(!meta.contains("\"process_pids\":[]"), "{meta}");
    // 7 工程がこの順（select → evaluate。選定してから評価する。REQ-27）に並ぶ
    let mut pos = 0;
    for step in [
        "register", "inspect", "train", "select", "evaluate", "package", "infer",
    ] {
        let needle = format!("\"step\":\"{step}\"");
        let found = meta[pos..]
            .find(&needle)
            .unwrap_or_else(|| panic!("step {step} missing or out of order: {meta}"));
        pos += found + needle.len();
    }
    assert_eq!(meta.matches("\"step\":").count(), 7, "{meta}");
    // 工程ごとの entry（平坦な JSON object）を切り出し、evaluate だけが status 値を持つことを確認する
    let entries: Vec<&str> = meta
        .split("{\"step\":\"")
        .skip(1)
        .map(|chunk| chunk.split('}').next().unwrap_or(chunk))
        .collect();
    assert_eq!(entries.len(), 7, "{meta}");
    let expected_status = format!("\"status\":\"{evaluate_status}\"");
    for entry in entries {
        let name = entry.split('"').next().unwrap_or("");
        assert_eq!(
            entry.contains(&expected_status),
            name == "evaluate",
            "step {name}: {entry}"
        );
        assert_eq!(
            entry.contains("\"status\":null"),
            name != "evaluate",
            "step {name}: {entry}"
        );
    }
    assert_eq!(meta.matches(&expected_status).count(), 1, "{meta}");
    assert_eq!(meta.matches("\"status\":null").count(), 6, "{meta}");
    // 全体（1）と 7 工程（7）の exit_code がすべて 0
    assert_eq!(meta.matches("\"exit_code\":0").count(), 8, "{meta}");

    // 利用者の値・一時パスが出力物へ漏れない（生文字列非保存）
    let tmp = dir.display().to_string();
    let mut outputs = vec![stdout.clone(), meta.clone()];
    for f in ["network_report.json", "monitor.meta.json"] {
        outputs.push(fs::read_to_string(dir.join("out").join(f)).expect("output file"));
    }
    for o in &outputs {
        assert!(!o.contains(INFER_MARKER), "infer text leaked: {o}");
        assert!(!o.contains(&tmp), "temp path leaked: {o}");
    }
}

/// 実 CLI・実 trainer・両スクリプトを通して 7 工程が完走し、本ツール起因の通信拒否が 0 件になる
/// （REQ-38 正常系のうち、実バイナリでの完走とスクリプトの判定の接続。評価データなし・`--smoke`・
/// `evaluate` は `skipped`。証拠種別: テストハーネス）。
#[test]
#[ignore = "requires trainer/.venv (make py-sync) and MLX CPU; run via make test-trainer-integration"]
fn req38_real_pipeline_completes_under_monitor_with_zero_tool_denials() {
    let chain = run_chain(
        "smoke",
        "sandbox_run",
        &["definition.json", "train.jsonl"],
        true,
    );
    assert_chain_completed(&chain, "skipped");
}

/// 評価データあり（`--smoke` なし）で 7 工程が完走し、`evaluate` が凍結データへ 1 回だけ適用されて
/// `status:"ok"` になり、評価完了記録と公開済みパッケージが残る（REQ-38・REQ-27・REQ-17・#348。
/// 証拠種別: テストハーネス。実機の証拠ではない）。
#[test]
#[ignore = "requires trainer/.venv (make py-sync) and MLX CPU; run via make test-trainer-integration"]
fn req38_real_pipeline_with_evaluation_data_completes_under_monitor_with_zero_tool_denials() {
    let chain = run_chain(
        "eval",
        "sandbox_run_eval",
        &["definition.json", "train.jsonl", "evaluation.jsonl"],
        false,
    );
    assert_chain_completed(&chain, "ok");
    let records = fs::read_dir(chain.dir.join("project/candidates"))
        .expect("candidates dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().join("evaluation_record.json").is_file())
        .count();
    assert_eq!(records, 1, "exactly one evaluation_record.json expected");
    assert!(chain.dir.join("project/package").is_dir());
}
