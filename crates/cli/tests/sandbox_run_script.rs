//! `scripts/sandbox-run.sh`（sandbox 下の 7 工程実行スクリプト）の結合テスト
//! （REQ-38・TASK-38.1-1・#162）。
//!
//! 証拠種別: テストハーネス。偽の launcher（`sandbox-exec` の代役）と偽の CLI を
//! 一時ディレクトリへ書き出して使うため、**実際の通信遮断は行っていない**。
//! macOS 実機での sandbox 下の完走確認と拒否ログの記録は人の担当で、本テストは
//! その証拠にならない（実機の手順は `AGENTS.md`「実機前提テスト」）。ここで検証するのは
//! スクリプトの制御（sandbox 経由の起動・工程の順序・停止・記録・資源上限・終了コード）。
//! 実バイナリは #136（TASK-33.1-2）で 7 工程を接続済み。ここでは実バイナリで register・inspect
//! が完走し、学習ワーカーが無い環境では train で止まること（完走を装わない）を確認する
//! （7 工程の完走は `pipeline_e2e.rs`）。
//! 拒否ログの監視と 0 件判定は #163（TASK-38.1-2）の担当。
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
    launch_log: PathBuf,
    cli_log: PathBuf,
}

impl Env {
    fn new() -> Self {
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("fandhe-sandbox-run-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir");
        let launcher = dir.join("fake-sandbox-exec");
        let cli = dir.join("fake-cli");
        let launch_log = dir.join("launch.log");
        let cli_log = dir.join("cli.log");
        // 偽の launcher: `-p <profile> <実行ファイル> ...` の形を検査し、目印を付けて exec する
        write_exe(
            &launcher,
            &format!(
                "#!/bin/sh\n\
                 [ \"$1\" = \"-p\" ] || exit 99\n\
                 echo \"$2\" >> \"{}\"\n\
                 [ \"$2\" = \"{PROFILE}\" ] || exit 99\n\
                 shift 2\n\
                 FAKE_SANDBOXED=1 exec \"$@\"\n",
                launch_log.display()
            ),
        );
        // 偽の CLI: sandbox を経由しない起動は 99。挙動は環境変数で切り替える
        write_exe(
            &cli,
            &format!(
                "#!/bin/sh\n\
                 [ \"${{FAKE_SANDBOXED:-}}\" = 1 ] || exit 99\n\
                 stage=$1\n\
                 cand=-\n\
                 prev=\n\
                 for a in \"$@\"; do\n\
                 [ \"$prev\" = --candidate ] && cand=$a\n\
                 prev=$a\n\
                 done\n\
                 echo \"$stage $cand\" >> \"{}\"\n\
                 echo \"$*\" >> \"{}.args\"\n\
                 [ -z \"${{FAKE_SLEEP_SECS:-}}\" ] || sleep \"$FAKE_SLEEP_SECS\"\n\
                 if [ \"${{FAKE_HANG_STAGE:-}}\" = \"$stage\" ]; then\n\
                 case \"${{FAKE_HANG_KIND:-sleep}}\" in\n\
                 yes) exec yes ;;\n\
                 *) sleep 300 &\n\
                 echo $! > \"$FAKE_PIDFILE\"\n\
                 sleep 300 ;;\n\
                 esac\n\
                 fi\n\
                 if [ \"${{FAKE_FAIL_STAGE:-}}\" = \"$stage\" ]; then\n\
                 exit \"${{FAKE_FAIL_RC:-70}}\"\n\
                 fi\n\
                 if [ \"$stage\" = evaluate ] && [ -n \"${{FAKE_EVAL_OUT:-}}\" ]; then\n\
                 printf '%s\\n' \"$FAKE_EVAL_OUT\"\n\
                 exit 0\n\
                 fi\n\
                 if [ \"$stage\" = evaluate ] && [ -n \"${{FAKE_EVAL_SKIPPED:-}}\" ]; then\n\
                 echo '{{\"step\":\"evaluate\",\"status\":\"skipped\"}}'\n\
                 exit 0\n\
                 fi\n\
                 if [ -n \"${{FAKE_STAGE_OUT_SET:-}}\" ]; then\n\
                 printf '%s' \"$FAKE_STAGE_OUT\"\n\
                 exit 0\n\
                 fi\n\
                 if [ \"$stage\" = infer ]; then\n\
                 echo '{{\"id\":\"input\",\"status\":\"ok\",\"predicted_label\":\"a\"}}'\n\
                 exit 0\n\
                 fi\n\
                 printf '{{\"step\":\"%s\",\"status\":\"ok\"}}\\n' \"$stage\"\n\
                 exit 0\n",
                cli_log.display(),
                cli_log.display()
            ),
        );
        Env {
            dir,
            launcher,
            cli,
            launch_log,
            cli_log,
        }
    }

    fn project(&self) -> PathBuf {
        self.dir.join("project")
    }

    fn out(&self) -> PathBuf {
        self.dir.join("out")
    }

    fn definition(&self) -> PathBuf {
        self.dir.join("definition.json")
    }

    fn base_args(&self) -> Vec<String> {
        vec![
            "--definition".into(),
            self.definition().display().to_string(),
            "--project-dir".into(),
            self.project().display().to_string(),
            "--out-dir".into(),
            self.out().display().to_string(),
        ]
    }

    fn lines(path: &Path) -> Vec<String> {
        fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn cli_calls(&self) -> Vec<String> {
        Self::lines(&self.cli_log)
    }

    /// 偽の CLI が受け取った引数全体（1 呼び出し 1 行）。
    fn cli_args(&self) -> Vec<String> {
        let mut path = self.cli_log.clone().into_os_string();
        path.push(".args");
        Self::lines(Path::new(&path))
    }

    fn launch_calls(&self) -> Vec<String> {
        Self::lines(&self.launch_log)
    }

    fn run(&self, args: &[String], envs: &[(&str, &str)]) -> Out {
        self.run_with(args, envs, Some(&self.launcher), Some(&self.cli))
    }

    fn run_with(
        &self,
        args: &[String],
        envs: &[(&str, &str)],
        launcher: Option<&Path>,
        cli: Option<&Path>,
    ) -> Out {
        self.run_in(args, envs, launcher, cli, None)
    }

    /// `run_with` に加えてスクリプトの cwd を指定できる版（実バイナリは経路の閉じ込めのため
    /// 定義ファイルを含むディレクトリを cwd にする必要がある。REQ-39）。
    fn run_in(
        &self,
        args: &[String],
        envs: &[(&str, &str)],
        launcher: Option<&Path>,
        cli: Option<&Path>,
        cwd: Option<&Path>,
    ) -> Out {
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("scripts")
            .join("sandbox-run.sh");
        let mut cmd = Command::new("sh");
        cmd.process_group(0)
            .arg(script)
            .args(args)
            .env_remove("FANDHE_EDGE_SANDBOX_EXEC")
            .env_remove("FANDHE_EDGE_BIN")
            .env("FAKE_PIDFILE", self.dir.join("pid"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        if let Some(l) = launcher {
            cmd.env("FANDHE_EDGE_SANDBOX_EXEC", l);
        }
        if let Some(c) = cli {
            cmd.env("FANDHE_EDGE_BIN", c);
        }
        for (k, v) in envs {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().expect("spawn sh");
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
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn write_exe(path: &Path, body: &str) {
    fs::write(path, body).expect("write");
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).expect("chmod");
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// stdout の集計 JSON から工程名と候補の並びを取り出す（`step` の出現順）。
fn step_names(stdout: &str) -> Vec<String> {
    stdout
        .split("{\"step\":\"")
        .skip(1)
        .filter_map(|p| p.split('"').next())
        .map(str::to_string)
        .collect()
}

const ALL_STAGES: [&str; 7] = [
    "register", "inspect", "train", "evaluate", "select", "package", "infer",
];

/// 全工程が偽の launcher 経由で順に完走し exit 0・集計 JSON が ok になる。
#[test]
fn req38_all_stages_complete_under_fake_sandbox() {
    let e = Env::new();
    let o = e.run(&e.base_args(), &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert!(o.stdout.starts_with("{\"code\":\"ok\","), "{}", o.stdout);
    assert!(o.stdout.contains("\"failed_step\":null"));
    assert_eq!(step_names(&o.stdout), ALL_STAGES);
    assert_eq!(e.launch_calls().len(), 7);
    assert_eq!(
        e.cli_calls(),
        [
            "register -",
            "inspect -",
            "train 0",
            "evaluate 0",
            "select -",
            "package -",
            "infer -"
        ]
    );
}

/// `--candidates 2` で train・evaluate が候補 0・1 の順に 2 回ずつ実行される。
#[test]
fn req38_candidates_param_runs_each_candidate() {
    let e = Env::new();
    let mut args = e.base_args();
    args.extend(s(&["--candidates=2", "--smoke"]));
    let o = e.run(&args, &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(
        e.cli_calls(),
        [
            "register -",
            "inspect -",
            "train 0",
            "train 1",
            "evaluate 0",
            "evaluate 1",
            "select -",
            "package -",
            "infer -"
        ]
    );
}

/// REQ-27: `--smoke` のときだけ package に検証専用の `--allow-smoke` を渡す（短縮学習の候補を
/// package するため）。`--smoke` なしでは渡さない（配布用の経路で smoke を許さない）。
#[test]
fn req27_allow_smoke_is_passed_to_package_only_with_smoke() {
    let package_args = |e: &Env| -> String {
        e.cli_args()
            .into_iter()
            .find(|l| l.starts_with("package "))
            .expect("package call")
    };
    let e = Env::new();
    let mut args = e.base_args();
    args.push("--smoke".to_string());
    let o = e.run(&args, &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert!(
        package_args(&e).ends_with(" --allow-smoke"),
        "{}",
        package_args(&e)
    );

    let e = Env::new();
    let o = e.run(&e.base_args(), &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert!(
        !package_args(&e).contains("--allow-smoke"),
        "{}",
        package_args(&e)
    );
}

/// launcher へ渡す遮断プロファイルが PoC-14/16 と完全一致する（弱める経路がない）。
#[test]
fn req38_profile_argv_is_exact() {
    let e = Env::new();
    let o = e.run(&e.base_args(), &[]);
    assert_eq!(o.code, Some(0));
    let calls = e.launch_calls();
    assert_eq!(calls.len(), 7);
    assert!(calls.iter().all(|c| c == PROFILE), "{calls:?}");
    assert!(
        o.stdout
            .contains(&format!("\"sandbox_profile\":\"{PROFILE}\""))
    );
}

/// CLI は sandbox を経由せずに起動されない（偽の CLI は経由しないと 99 を返す）。
#[test]
fn req38_cli_is_never_started_outside_sandbox() {
    let e = Env::new();
    let o = e.run(&e.base_args(), &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(e.cli_calls().len(), e.launch_calls().len());
}

/// launcher が無い・通常ファイルでない場合は CLI を起動せず 70（fail-closed）。
#[test]
fn req38_missing_launcher_fails_closed_without_running_cli() {
    let e = Env::new();
    let missing = e.dir.join("nonexistent");
    let o = e.run_with(&e.base_args(), &[], Some(&missing), Some(&e.cli));
    assert_eq!(o.code, Some(70));
    assert!(o.stdout.starts_with("{\"code\":\"runtime_error\""));
    assert!(e.cli_calls().is_empty());
    let o = e.run_with(&e.base_args(), &[], Some(&e.dir), Some(&e.cli));
    assert_eq!(o.code, Some(70));
    assert!(e.cli_calls().is_empty());
}

/// 上書きが無く既定の /usr/bin/sandbox-exec も無い環境（Linux 等）でも 70 で停止する。
#[cfg(not(target_os = "macos"))]
#[test]
fn req38_no_default_launcher_fails_closed_without_running_cli() {
    let e = Env::new();
    let o = e.run_with(&e.base_args(), &[], None, Some(&e.cli));
    assert_eq!(o.code, Some(70));
    assert!(e.cli_calls().is_empty());
}

/// train が limit_exceeded(20) なら 20 で停止し、evaluate 以降は起動しない。
#[test]
fn req21_stops_at_failing_train_with_limit_exceeded() {
    let e = Env::new();
    let o = e.run(
        &e.base_args(),
        &[("FAKE_FAIL_STAGE", "train"), ("FAKE_FAIL_RC", "20")],
    );
    assert_eq!(o.code, Some(20), "stdout={}", o.stdout);
    assert!(o.stdout.starts_with("{\"code\":\"limit_exceeded\""));
    assert!(o.stdout.contains("\"failed_step\":\"train\""));
    assert_eq!(e.cli_calls(), ["register -", "inspect -", "train 0"]);
}

/// 契約外の終了コード（3）は runtime_error(70) へ写す。
#[test]
fn req21_unknown_exit_code_maps_to_runtime_error() {
    let e = Env::new();
    let o = e.run(
        &e.base_args(),
        &[("FAKE_FAIL_STAGE", "inspect"), ("FAKE_FAIL_RC", "3")],
    );
    assert_eq!(o.code, Some(70));
    assert!(o.stdout.contains("\"failed_step\":\"inspect\""));
}

/// evaluate の status:"skipped"（exit 0）は続行しつつ skipped と記録する（評価済みを装わない）。
#[test]
fn req17_evaluate_skipped_is_reported_distinctly() {
    let e = Env::new();
    let o = e.run(&e.base_args(), &[("FAKE_EVAL_SKIPPED", "1")]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert!(
        o.stdout.contains(
            "{\"step\":\"evaluate\",\"candidate\":0,\"exit_code\":0,\"code\":\"ok\",\"status\":\"skipped\"}"
        ),
        "{}",
        o.stdout
    );
}

/// skipped 判定は JSON の構造で行う。空白入りの正当な JSON は検出し、入れ子の status・文字列中の
/// 断片は誤検出しない。解析できない出力は判定不能として 70（REQ-17・REQ-21）。
#[test]
fn req17_evaluate_skipped_detection_is_structural() {
    let e = Env::new();
    let o = e.run(
        &e.base_args(),
        &[(
            "FAKE_EVAL_OUT",
            "{ \"step\" : \"evaluate\", \"status\" : \"skipped\" }",
        )],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert!(o.stdout.contains("\"status\":\"skipped\"}"), "{}", o.stdout);

    let e = Env::new();
    let o = e.run(
        &e.base_args(),
        &[(
            "FAKE_EVAL_OUT",
            "{\"step\":\"evaluate\",\"status\":\"ok\",\"note\":\"\\\"status\\\":\\\"skipped\\\"\",\"inner\":{\"status\":\"skipped\"}}",
        )],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert!(
        o.stdout
            .contains("\"step\":\"evaluate\",\"candidate\":0,\"exit_code\":0,\"code\":\"ok\",\"status\":\"ok\""),
        "{}",
        o.stdout
    );

    let e = Env::new();
    let o = e.run(&e.base_args(), &[("FAKE_EVAL_OUT", "not json")]);
    assert_eq!(o.code, Some(70), "stdout={}", o.stdout);
    assert!(o.stdout.contains("\"failed_step\":\"evaluate\""));
}

/// 工程の期限切れで工程グループごと終了し 70 を返す。
#[test]
fn req39_step_timeout_kills_descendants_and_returns_70() {
    let e = Env::new();
    let o = e.run(
        &e.base_args(),
        &[
            ("FANDHE_EDGE_SANDBOX_STEP_TIMEOUT_SECS", "1"),
            ("FAKE_HANG_STAGE", "inspect"),
        ],
    );
    assert_eq!(o.code, Some(70), "stdout={}", o.stdout);
    assert!(o.stdout.contains("\"failed_step\":\"inspect\""));
    let pid = fs::read_to_string(e.dir.join("pid")).expect("pidfile");
    let pid = pid.trim();
    // 子孫（sleep）が残っていないこと（KILL 後の回収を少し待つ）
    let start = Instant::now();
    loop {
        let alive = Command::new("kill")
            .args(["-0", pid])
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !alive {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(5), "descendant alive");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// stdout が上限（1 MiB）を超えたら 70。
#[test]
fn req39_step_output_over_limit_returns_70() {
    let e = Env::new();
    let o = e.run(
        &e.base_args(),
        &[("FAKE_HANG_STAGE", "register"), ("FAKE_HANG_KIND", "yes")],
    );
    assert_eq!(o.code, Some(70), "stdout={}", o.stdout);
    assert!(o.stdout.contains("\"failed_step\":\"register\""));
}

/// 既存の --project-dir・空でない --out-dir は 64 で拒否し、CLI を起動せず既存の中身を消さない。
#[test]
fn req33_existing_dirs_are_rejected_with_64() {
    let e = Env::new();
    fs::create_dir_all(e.project()).expect("mkdir");
    fs::write(e.project().join("keep"), "x").expect("write");
    let o = e.run(&e.base_args(), &[]);
    assert_eq!(o.code, Some(64));
    assert!(e.project().join("keep").exists());
    assert!(e.cli_calls().is_empty());

    let e = Env::new();
    fs::create_dir_all(e.out()).expect("mkdir");
    fs::write(e.out().join("keep"), "x").expect("write");
    let o = e.run(&e.base_args(), &[]);
    assert_eq!(o.code, Some(64));
    assert!(e.out().join("keep").exists());
    assert!(e.cli_calls().is_empty());
}

/// 書き込み不能な既存の空 --out-dir は、CLI を起動する前に runtime_error（70）の JSON で返す（REQ-21・REQ-33）。
#[test]
fn req33_unwritable_out_dir_returns_runtime_error_json() {
    let e = Env::new();
    fs::create_dir_all(e.out()).expect("mkdir");
    fs::set_permissions(e.out(), fs::Permissions::from_mode(0o555)).expect("chmod");
    // root など権限を無視する実行環境では前提が成り立たないため、その場合だけ検証を省く
    let writable = fs::write(e.out().join("probe"), "x").is_ok();
    if writable {
        fs::remove_file(e.out().join("probe")).ok();
    } else {
        let o = e.run(&e.base_args(), &[]);
        assert_eq!(o.code, Some(70), "stdout={}", o.stdout);
        assert_eq!(
            o.stdout,
            "{\"code\":\"runtime_error\",\"message\":\"cannot write run record\"}\n"
        );
        assert!(e.cli_calls().is_empty());
    }
    fs::set_permissions(e.out(), fs::Permissions::from_mode(0o755)).expect("chmod");
}

/// 必須の欠落・未知のオプション・不正な --candidates は 64。
#[test]
fn req33_invalid_args_are_rejected_with_64() {
    let e = Env::new();
    assert_eq!(e.run(&s(&["--definition", "d"]), &[]).code, Some(64));
    let mut unknown = e.base_args();
    unknown.push("--bogus".into());
    assert_eq!(e.run(&unknown, &[]).code, Some(64));
    for bad in ["0", "01", "17", "x", ""] {
        let mut a = e.base_args();
        a.extend(s(&["--candidates", bad]));
        assert_eq!(e.run(&a, &[]).code, Some(64), "candidates={bad}");
    }
    let mut dup = e.base_args();
    dup.extend(s(&["--definition", "again"]));
    assert_eq!(e.run(&dup, &[]).code, Some(64));
    assert!(e.cli_calls().is_empty());
}

/// 上書き実行は stdout と run.meta.json に override と test_harness を記録し、時刻は UTC 形式。
#[test]
fn req38_override_is_recorded_in_output_and_meta() {
    let e = Env::new();
    let o = e.run(&e.base_args(), &[]);
    assert_eq!(o.code, Some(0));
    assert!(o.stdout.contains("\"sandbox_exec_override\":true"));
    let meta = fs::read_to_string(e.out().join("run.meta.json")).expect("meta");
    assert!(meta.contains("\"sandbox_exec_override\":true"));
    assert!(meta.contains("\"evidence_hint\":\"test_harness\""));
    let field = |key: &str| -> String {
        let p = format!("\"{key}\":\"");
        let rest = meta.split(&p).nth(1).expect(key);
        rest.split('"').next().expect(key).to_string()
    };
    let (a, b) = (field("started_utc"), field("ended_utc"));
    for t in [&a, &b] {
        assert_eq!(t.len(), 20, "{t}");
        assert!(
            t.ends_with('Z') && t.as_bytes().get(10) == Some(&b'T'),
            "{t}"
        );
    }
    assert!(a <= b);
}

/// --infer-text の本文は out-dir の記録のどこにも現れない（データ本文をログへ残さない）。
#[test]
fn req38_infer_text_is_not_written_to_logs() {
    let e = Env::new();
    let mut args = e.base_args();
    args.extend(s(&["--infer-text", INFER_MARKER]));
    let o = e.run(&args, &[]);
    assert_eq!(o.code, Some(0));
    assert!(!o.stdout.contains(INFER_MARKER));
    fn walk(p: &Path, hits: &mut Vec<PathBuf>) {
        for ent in fs::read_dir(p).expect("read_dir") {
            let path = ent.expect("ent").path();
            if path.is_dir() {
                walk(&path, hits);
            } else if fs::read_to_string(&path)
                .unwrap_or_default()
                .contains(INFER_MARKER)
            {
                hits.push(path);
            }
        }
    }
    let mut hits = Vec::new();
    walk(&e.out(), &mut hits);
    assert!(hits.is_empty(), "{hits:?}");
}

/// 実バイナリ（#136 で接続済み）で register・inspect が完走し（code の無い exit 0 の工程結果 JSON を
/// スクリプトが受理する）、学習ワーカーが無い環境（存在しない `FANDHE_EDGE_TRAINER_DIR`）では
/// train で止まる。完走を装わない確認（REQ-33・REQ-38。証拠種別: テストハーネス）。
#[test]
fn req38_real_binary_completes_register_and_inspect_then_stops_at_train() {
    let e = Env::new();
    fs::write(
        e.definition(),
        r#"{"schema":"fandhe-edge-model-definition/v1","name":"sandbox_real","version":1,"judgment_type":"single_select","options":[{"id":"a","display_name":"a","description":"d"},{"id":"b","display_name":"b","description":"d"}],"io":{"input":"bytes"}}"#,
    )
    .expect("definition");
    let mut rows = String::new();
    for i in 0..20 {
        for l in ["a", "b"] {
            rows.push_str(&format!(
                "{{\"id\":\"{l}{i}\",\"input\":\"{l} text {i}\",\"output\":{{\"intent\":\"{l}\"}},\"group_id\":\"g{l}{i}\"}}\n"
            ));
        }
    }
    fs::write(e.dir.join("train.jsonl"), rows).expect("train data");
    let real = PathBuf::from(env!("CARGO_BIN_EXE_fandhe-edge"));
    let missing_trainer = e.dir.join("no-such-trainer");
    let o = e.run_in(
        &e.base_args(),
        &[(
            "FANDHE_EDGE_TRAINER_DIR",
            &missing_trainer.display().to_string(),
        )],
        Some(&e.launcher),
        Some(&real),
        Some(&e.dir),
    );
    assert_ne!(o.code, Some(0), "stdout={}", o.stdout);
    assert!(
        o.stdout.contains("\"failed_step\":\"train\""),
        "stdout={}",
        o.stdout
    );
    assert_eq!(step_names(&o.stdout), ["register", "inspect", "train"]);
    assert!(
        o.stdout
            .contains("\"step\":\"register\",\"candidate\":null,\"exit_code\":0"),
        "stdout={}",
        o.stdout
    );
    assert_eq!(e.launch_calls().len(), 3);
}

/// exit 0 の工程結果 JSON は `code` を持たなくてよいが、`step` が工程名と一致し `status` が "ok"
/// でなければ 70（REQ-33。TASK-33.1-2・#136）。
#[test]
fn req33_zero_exit_stage_json_must_match_stage_contract() {
    let e = Env::new();
    let o = e.run(
        &e.base_args(),
        &[
            ("FAKE_STAGE_OUT", "{\"step\":\"x\",\"status\":\"ok\"}\n"),
            ("FAKE_STAGE_OUT_SET", "1"),
        ],
    );
    // 工程名と一致しない step は契約外（REQ-33。register の出力が register でない）
    assert_eq!(o.code, Some(70), "stdout={}", o.stdout);
}

/// --out-dir が --project-dir と同一・配下・祖先だと 64 で拒否し、CLI を起動せず
/// 未作成の project-dir を作らない（REQ-33・REQ-38）。
#[test]
fn req33_out_dir_overlapping_project_dir_is_rejected() {
    let e = Env::new();
    let project = e.project().display().to_string();
    let cases = [
        (project.clone(), project.clone()),
        (project.clone(), format!("{project}/out")),
        (format!("{project}/nested"), e.dir.display().to_string()),
        (project.clone(), format!("{project}/sub/../../project")),
    ];
    for (proj, out) in cases {
        let o = e.run(
            &s(&[
                "--definition",
                &e.definition().display().to_string(),
                "--project-dir",
                &proj,
                "--out-dir",
                &out,
            ]),
            &[],
        );
        assert_eq!(
            o.code,
            Some(64),
            "proj={proj} out={out} stdout={}",
            o.stdout
        );
        assert!(!e.project().exists(), "proj={proj} out={out}");
        assert!(e.cli_calls().is_empty());
    }
}

/// 終了コード 0 でも stdout が空・不正 JSON・複数 JSON・code 不整合なら 70 で停止する
/// （REQ-21・REQ-33）。
#[test]
fn req33_zero_exit_with_invalid_stage_output_stops_with_70() {
    let bad = [
        "",
        "not json",
        "{\"code\":\"ok\"}\n{\"code\":\"ok\"}",
        "[]",
        "{\"code\":\"judged_fail\"}",
        "{}",
        // code の無い任意の JSON・工程名の不一致・契約外の status・infer 形の出力は契約外
        "{\"foo\":1}",
        "{\"step\":\"inspect\",\"status\":\"ok\"}",
        "{\"step\":\"register\",\"status\":\"failed\"}",
        "{\"step\":\"register\",\"status\":\"skipped\"}",
        "{\"id\":\"input\",\"status\":\"ok\",\"predicted_label\":\"a\"}",
    ];
    for out in bad {
        let e = Env::new();
        let o = e.run(
            &e.base_args(),
            &[("FAKE_STAGE_OUT", out), ("FAKE_STAGE_OUT_SET", "1")],
        );
        assert_eq!(o.code, Some(70), "out={out:?} stdout={}", o.stdout);
        assert!(
            o.stdout.contains("\"failed_step\":\"register\""),
            "{}",
            o.stdout
        );
        assert_eq!(e.cli_calls().len(), 1, "out={out:?}");
    }
}

/// run.meta.json の `process_pids` に、工程グループのプロセスの PID（数値の配列）を記録する。
/// 拒否ログの帰属判定（`sandbox_deny_report.py`）が PID で本ツール起因を照合するための記録
/// （REQ-38・TASK-38.1-2）。工程が 0.1 秒のポーリングより長く動く場合に少なくとも 1 件入る。
#[test]
fn req38_run_meta_records_process_pids_of_step_group() {
    let e = Env::new();
    let o = e.run(&e.base_args(), &[("FAKE_SLEEP_SECS", "1")]);
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    let meta = fs::read_to_string(e.out().join("run.meta.json")).expect("meta");
    let start =
        meta.find("\"process_pids\":[").expect("process_pids key") + "\"process_pids\":[".len();
    let end = start + meta[start..].find(']').expect("closing bracket");
    let pids: Vec<&str> = meta[start..end].split(',').collect();
    assert!(!pids.is_empty() && !pids[0].is_empty(), "{meta}");
    for p in pids {
        assert!(p.parse::<u32>().is_ok(), "non-numeric pid {p:?} in {meta}");
    }
}
