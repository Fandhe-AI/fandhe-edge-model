//! `crate::process::run_train`（issue #178。REQ-21・REQ-34・REQ-39）の結合
//! テスト兼 MLX 不要の偽ワーカー。
//!
//! # なぜ `harness = false` の単一バイナリに同居させるか
//!
//! libtest（既定の harness）付きのテストバイナリを、学習ワーカーの代わりに
//! 子プロセスとして再実行すると、本体（`running 1 test` 等）が標準出力へ
//! 出てしまい、学習結果契約の「JSON 1 行」を守れない。`[[bin]]` にすると
//! 製品バイナリとして出荷されてしまい、`CARGO_BIN_EXE_*` は `[[bin]]` にしか
//! 使えない。シェル・Python の偽ワーカーは Windows・venv の無い rust-ci で
//! 動かない。そのため本ファイル自身を `WorkerLauncher` の `python` として
//! 渡し、`main()` が「学習ワーカーとして起動されたか」「テストランナーとして
//! 実行されたか」を argv で分岐する（issue #178 実装計画 3.8）。
//!
//! # 偽ワーカーへの挙動の指定
//!
//! `run_train` は子プロセスの環境を `env_clear()` するため、環境変数では
//! 挙動を渡せない。代わりに `WorkerLauncher` の `launch_script`（実体は
//! `launch.py` という名前の任意のテキストファイル。`WorkerLauncher::new` は
//! ファイル名だけを検査し中身は検査しない）に挙動を表す短いモード文字列を
//! 書き込み、偽ワーカーはそれを読んで分岐する。

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode as ProcessExitCode;
use std::time::Duration;

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_train::process::{ENV_ALLOWLIST, RunLimits, WorkerLauncher, run_train};
use fandhe_edge_train::request::{Device, TrainRequest, TrainRequestParams};

/// [`TrainRequest`] の `root`（実在しない絶対パス。`root` の symlink 解決は
/// ベストエフォートで、存在しない場合は文字列のまま扱われる。
/// `crates/train/src/result.rs::canonicalize_root_best_effort` 参照）。
const FIXTURE_ROOT: &str = "/fandhe-edge-worker-process-fixture-root";

/// `c3` の `DEFAULT_CONFIG`（`fixtures/train_contract/kind_defaults.json`と
/// 一致する具体値）を埋め込んだ、成功結果 JSON の固定テンプレート。
/// `{ARTIFACT_DIR}` を実際の `artifact_dir` へ置換して使う。
const OK_JSON_TEMPLATE: &str = r#"{"status":"ok","artifact_dir":"{ARTIFACT_DIR}","artifact":{"kind":"c3","kind_version":1,"selector_version":"0.1","config":{"lr":0.001,"weight_decay":0.0001,"epochs":40,"batch_size":64,"emb":64,"filters":128,"widths":[3,5,7],"dropout":0.3},"label_order":["a","b"],"output_type":"choice","max_bytes":512,"onnx_file":"model.onnx","onnx_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","created_utc":"2026-09-28T00:00:00Z","candidate_label":"c3"}}"#;

fn ok_json(artifact_dir: &str) -> String {
    OK_JSON_TEMPLATE.replace("{ARTIFACT_DIR}", artifact_dir)
}

// ============================================================
// main: 偽ワーカーとテストランナーの分岐（issue #178 実装計画 3.8）
// ============================================================

fn main() -> ProcessExitCode {
    let args: Vec<String> = std::env::args().collect();
    // `run_train` が組み立てる固定 argv:
    // [program, "-I", launch_script, "train", "--request", request_path]
    let is_worker_invocation = args.get(1).map(String::as_str) == Some("-I")
        && args.get(3).map(String::as_str) == Some("train")
        && args.get(4).map(String::as_str) == Some("--request");
    if is_worker_invocation {
        let launch_script = args.get(2).expect("launch_script arg present");
        let request_path = args.get(5).expect("request path arg present");
        // `run_fake_worker` の戻り値型は `!`（内部で必ず
        // `std::process::exit` する）ため、ここには戻ってこない。
        run_fake_worker(launch_script, request_path);
    }
    run_test_suite()
}

/// `launch_script`（モード文字列が書かれたファイル）の内容に応じて偽ワーカー
/// として振る舞う。プロセスの終了は分岐内で `std::process::exit` する
/// （`main` の戻り値ではプロセスの実終了コードを細かく制御できないため）。
fn run_fake_worker(launch_script: &str, request_path: &str) -> ! {
    let mode = std::fs::read_to_string(launch_script).unwrap_or_default();
    let mode = mode.trim();
    match mode {
        "ok" => {
            print!("{}", ok_json(&format!("{FIXTURE_ROOT}/out")));
            std::process::exit(0);
        }
        "error_invalid_request" => {
            print!(r#"{{"status":"error","code":"invalid_request","message":"m"}}"#);
            std::process::exit(64);
        }
        "error_training_diverged" => {
            print!(r#"{{"status":"error","code":"training_diverged","message":"m"}}"#);
            std::process::exit(12);
        }
        "mismatch_ok_exit64" => {
            // 結果 JSON は成功だが、プロセスの実終了コードは 64
            // （REQ-39「完全性と版」: 一致しない場合は拒否されるはず）。
            print!("{}", ok_json(&format!("{FIXTURE_ROOT}/out")));
            std::process::exit(64);
        }
        "exit3" => {
            // 7 種の終了コードのいずれでもない値。
            std::process::exit(3);
        }
        "abort" => {
            std::process::abort();
        }
        "big_stdout" => {
            // `MAX_RESULT_BYTES`（1 MiB）を超える標準出力。
            let huge = vec![b'a'; 2 * 1024 * 1024];
            let mut stdout = std::io::stdout();
            let _ = stdout.write_all(&huge);
            std::process::exit(0);
        }
        "big_stderr" => {
            // `MAX_WORKER_STDERR_BYTES`（64 KiB）を大幅に超える標準エラー
            // 出力の後、正常な標準出力を続ける。
            let huge = vec![b'b'; 4 * 1024 * 1024];
            let mut stderr = std::io::stderr();
            let _ = stderr.write_all(&huge);
            print!("{}", ok_json(&format!("{FIXTURE_ROOT}/out")));
            std::process::exit(0);
        }
        "hang" => {
            // `heartbeat.txt`（子プロセスの cwd = job_dir）へ 50ms ごとに
            // 1 バイト追記し続け、標準出力を閉じずに待つ（外側の壁時計
            // タイムアウトで kill されるまで自発的に終了しない）。
            loop {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("heartbeat.txt")
                {
                    let _ = f.write_all(b".");
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        "record" => {
            // argv・受け取った request.json の内容・環境変数名の一覧を
            // cwd（= job_dir）の `record.json` へ書く（issue #178 実装計画
            // 6.1「record」ケース）。
            let argv: Vec<String> = std::env::args().collect();
            let request_bytes =
                std::fs::read(request_path).unwrap_or_else(|e| panic!("read request: {e}"));
            let mut env_names: Vec<String> = std::env::vars().map(|(k, _)| k).collect();
            env_names.sort();
            let record = serde_json::json!({
                "argv": argv,
                "request_base64_len": request_bytes.len(),
                "request_utf8": String::from_utf8_lossy(&request_bytes),
                "env_names": env_names,
            });
            std::fs::write(
                "record.json",
                serde_json::to_vec(&record).expect("serialize record"),
            )
            .expect("write record.json");
            print!("{}", ok_json(&format!("{FIXTURE_ROOT}/out")));
            std::process::exit(0);
        }
        other => {
            eprintln!("fake worker: unknown mode {other:?}");
            std::process::exit(70);
        }
    }
}

// ============================================================
// テストランナー
// ============================================================

/// 各テストケースの実行結果。
struct CaseResult {
    name: &'static str,
    ok: bool,
    detail: String,
}

/// ケース名と実行関数の組。`clippy::type_complexity` を避けるための型別名。
type CaseFn = fn(&Path) -> Result<(), String>;

fn run_test_suite() -> ProcessExitCode {
    let cases: Vec<(&'static str, CaseFn)> = vec![
        ("ok_outcome", case_ok_outcome),
        ("error_invalid_request", case_error_invalid_request),
        ("error_training_diverged", case_error_training_diverged),
        ("mismatch_ok_exit64", case_mismatch_ok_exit64),
        ("unknown_exit_code", case_unknown_exit_code),
        ("abort", case_abort),
        ("big_stdout", case_big_stdout),
        ("big_stderr", case_big_stderr),
        ("timeout_hang", case_timeout_hang),
        ("record_argv_and_request", case_record),
        ("invalid_job_dir", case_invalid_job_dir),
        ("existing_request_file_rejected", case_existing_request_file),
    ];

    let mut results = Vec::new();
    for (name, case_fn) in cases {
        let case_dir = case_tmp_dir(name);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| case_fn(&case_dir)));
        let result = match outcome {
            Ok(Ok(())) => CaseResult {
                name,
                ok: true,
                detail: String::new(),
            },
            Ok(Err(detail)) => CaseResult {
                name,
                ok: false,
                detail,
            },
            Err(panic) => CaseResult {
                name,
                ok: false,
                detail: format!("panicked: {}", panic_message(&panic)),
            },
        };
        results.push(result);
    }

    let mut all_ok = true;
    for result in &results {
        if result.ok {
            println!("ok - {}", result.name);
        } else {
            all_ok = false;
            println!("FAIL - {}: {}", result.name, result.detail);
        }
    }

    if all_ok {
        println!("worker_process: {} cases passed", results.len());
        ProcessExitCode::SUCCESS
    } else {
        ProcessExitCode::FAILURE
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// ケースごとの一意な一時ディレクトリを作る（`CARGO_TARGET_TMPDIR` 配下。
/// ケース名＋pid＋固定サフィックスで衝突を避ける）。
fn case_tmp_dir(case_name: &str) -> PathBuf {
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let dir = base.join(format!("worker-process-{case_name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create case tmp dir");
    dir
}

/// `case_dir` 配下に `launch.py`（挙動指定の中身）を書き、`WorkerLauncher`
/// を組み立てる。
fn make_launcher(case_dir: &Path, mode: &str) -> WorkerLauncher {
    let launch_script = case_dir.join("launch.py");
    std::fs::write(&launch_script, mode).expect("write launch.py");
    let python = std::env::current_exe().expect("resolve current_exe");
    WorkerLauncher::new(python, launch_script).expect("valid launcher")
}

fn make_request(time_limit_seconds: Option<u32>) -> TrainRequest {
    TrainRequest::new(TrainRequestParams {
        kind: "c3".to_string(),
        kind_version: 1,
        config: serde_json::Map::new(),
        label_order: vec!["a".to_string(), "b".to_string()],
        max_bytes: 512,
        seed: 0,
        device: Device::Cpu,
        root: FIXTURE_ROOT.to_string(),
        train_path: "train.jsonl".to_string(),
        out_dir: "out".to_string(),
        time_limit_seconds,
        rss_limit_bytes: None,
    })
    .expect("valid request params")
}

fn expect_eq<T: PartialEq + std::fmt::Debug>(
    actual: T,
    expected: T,
    what: &str,
) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{what}: expected {expected:?}, got {actual:?}"))
    }
}

fn expect_true(cond: bool, what: &str) -> Result<(), String> {
    if cond {
        Ok(())
    } else {
        Err(format!("{what}: condition was false"))
    }
}

/// 受け入れ条件 1: 正常終了時に結果 JSON を `TrainOutcome` として受け取れる。
fn case_ok_outcome(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "ok");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    let run = run_train(&launcher, &request, case_dir, &limits)
        .map_err(|e| format!("run_train failed: {e}"))?;
    expect_eq(run.exit_code(), ExitCode::Ok, "exit_code")?;
    match run.outcome() {
        fandhe_edge_train::result::TrainOutcome::Ok(success) => {
            expect_eq(success.artifact().kind(), "c3", "artifact.kind")?;
            expect_eq(
                success.artifact_dir(),
                format!("{FIXTURE_ROOT}/out").as_str(),
                "artifact_dir",
            )?;
        }
        fandhe_edge_train::result::TrainOutcome::Error(_) => {
            return Err("expected Ok outcome".to_string());
        }
    }
    expect_true(
        !case_dir.join("request.json").exists(),
        "request.json must be deleted after success",
    )
}

/// 受け入れ条件 3: ワーカーのエラー（`invalid_request`）が `InvalidInput`
/// （64）へ写る。
fn case_error_invalid_request(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "error_invalid_request");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    let run = run_train(&launcher, &request, case_dir, &limits)
        .map_err(|e| format!("run_train failed: {e}"))?;
    expect_eq(run.exit_code(), ExitCode::InvalidInput, "exit_code")
}

/// 受け入れ条件 3: `training_diverged` が `Pending`（12）へ写る。
fn case_error_training_diverged(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "error_training_diverged");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    let run = run_train(&launcher, &request, case_dir, &limits)
        .map_err(|e| format!("run_train failed: {e}"))?;
    expect_eq(run.exit_code(), ExitCode::Pending, "exit_code")
}

/// 受け入れ条件 3: 結果 JSON（成功）とプロセスの実終了コード（64）が
/// 食い違う場合は `ExitCodeMismatch`（→ `RuntimeError`）として拒否する。
fn case_mismatch_ok_exit64(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "mismatch_ok_exit64");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    match run_train(&launcher, &request, case_dir, &limits) {
        Err(e) => expect_eq(e.exit_code(), ExitCode::RuntimeError, "exit_code"),
        Ok(_) => Err("expected ExitCodeMismatch error".to_string()),
    }
}

/// 受け入れ条件 3: 7 種以外のプロセス終了コード（3）は `RuntimeError` へ
/// 写る。
fn case_unknown_exit_code(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "exit3");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    match run_train(&launcher, &request, case_dir, &limits) {
        Err(e) => expect_eq(e.exit_code(), ExitCode::RuntimeError, "exit_code"),
        Ok(_) => Err("expected UnknownExitCode error".to_string()),
    }
}

/// `abort()` はシグナル終了（unix）／`STATUS_STACK_BUFFER_OVERRUN`
/// 相当の非標準終了コード（windows）のいずれであっても `RuntimeError`
/// （70）へ写る。
fn case_abort(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "abort");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    match run_train(&launcher, &request, case_dir, &limits) {
        Err(e) => expect_eq(e.exit_code(), ExitCode::RuntimeError, "exit_code"),
        Ok(_) => Err("expected abort to be rejected".to_string()),
    }
}

/// 巨大出力: `MAX_RESULT_BYTES` を超える標準出力は締め切り前に `TooLarge`
/// として拒否される（パイプ詰まりで固まらない）。
fn case_big_stdout(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "big_stdout");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    match run_train(&launcher, &request, case_dir, &limits) {
        Err(e) => expect_eq(e.exit_code(), ExitCode::RuntimeError, "exit_code"),
        Ok(_) => Err("expected TooLarge stdout to be rejected".to_string()),
    }
}

/// 巨大 stderr: `MAX_WORKER_STDERR_BYTES` を超えた分は読み捨てられ、正常な
/// 標準出力は成功として受理される。
fn case_big_stderr(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "big_stderr");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    let run = run_train(&launcher, &request, case_dir, &limits)
        .map_err(|e| format!("run_train failed: {e}"))?;
    expect_eq(run.exit_code(), ExitCode::Ok, "exit_code")?;
    expect_eq(
        run.worker_stderr().len(),
        fandhe_edge_train::limits::MAX_WORKER_STDERR_BYTES,
        "worker_stderr length",
    )?;
    expect_true(run.stderr_truncated(), "stderr_truncated")
}

/// 受け入れ条件 2: タイムアウトで子プロセスを確実に終了させ、
/// `LimitExceeded`（20）で返る。返った後に heartbeat が伸びない
/// （子プロセスが残っていない）ことも確認する。
fn case_timeout_hang(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "hang");
    let request = make_request(Some(1));
    let limits = RunLimits::for_request(&request)
        .with_wall_timeout(Duration::from_millis(500))
        .expect("tighten wall timeout");
    let started = std::time::Instant::now();
    let err = match run_train(&launcher, &request, case_dir, &limits) {
        Err(e) => e,
        Ok(_) => return Err("expected WallTimeout error".to_string()),
    };
    expect_eq(err.exit_code(), ExitCode::LimitExceeded, "exit_code")?;
    expect_eq(err.reason_code(), "limit_exceeded", "reason_code")?;
    expect_true(
        started.elapsed() < Duration::from_secs(10),
        "must return well before the 10s safety margin",
    )?;
    let heartbeat = case_dir.join("heartbeat.txt");
    let len_after_return = std::fs::metadata(&heartbeat).map(|m| m.len()).unwrap_or(0);
    std::thread::sleep(Duration::from_millis(300));
    let len_after_wait = std::fs::metadata(&heartbeat).map(|m| m.len()).unwrap_or(0);
    expect_eq(
        len_after_wait,
        len_after_return,
        "heartbeat must not grow after run_train returns (no orphaned process)",
    )
}

/// `record` モード: argv・request.json の内容・環境変数名の許可リスト
/// 準拠を確認する（issue #178 実装計画 3.1〜3.3・6.1「record」ケース）。
fn case_record(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "record");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    let _run = run_train(&launcher, &request, case_dir, &limits)
        .map_err(|e| format!("run_train failed: {e}"))?;

    let record_path = case_dir.join("record.json");
    let record_bytes = std::fs::read(&record_path).map_err(|e| format!("read record.json: {e}"))?;
    let record: serde_json::Value =
        serde_json::from_slice(&record_bytes).map_err(|e| format!("parse record.json: {e}"))?;

    let argv = record
        .get("argv")
        .and_then(|v| v.as_array())
        .ok_or("record.argv missing")?;
    let argv: Vec<String> = argv
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect();
    let expected_launch_script = case_dir.join("launch.py");
    let expected_request_path = case_dir.join("request.json");
    let expected_argv = vec![
        argv.first().cloned().unwrap_or_default(), // program path (current_exe)。検証しない。
        "-I".to_string(),
        expected_launch_script.to_string_lossy().to_string(),
        "train".to_string(),
        "--request".to_string(),
        expected_request_path.to_string_lossy().to_string(),
    ];
    expect_eq(argv.clone(), expected_argv, "argv")?;

    let request_utf8 = record
        .get("request_utf8")
        .and_then(|v| v.as_str())
        .ok_or("record.request_utf8 missing")?;
    let expected_bytes = request.to_json_vec().expect("serialize request");
    expect_eq(
        request_utf8.as_bytes().to_vec(),
        expected_bytes,
        "request.json bytes seen by worker",
    )?;

    expect_true(
        !expected_request_path.exists(),
        "request.json must be deleted after run_train returns",
    )?;

    let env_names = record
        .get("env_names")
        .and_then(|v| v.as_array())
        .ok_or("record.env_names missing")?;
    for name in env_names {
        let name = name.as_str().unwrap_or_default();
        expect_true(
            ENV_ALLOWLIST.contains(&name),
            &format!("child env var {name:?} must be in ENV_ALLOWLIST"),
        )?;
    }
    Ok(())
}

/// 起動口・job_dir の不正: 存在しない `job_dir` は `InvalidJobDir` として
/// 拒否する（子プロセスを起動しない）。
fn case_invalid_job_dir(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "ok");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    let missing_dir = case_dir.join("does-not-exist");
    match run_train(&launcher, &request, &missing_dir, &limits) {
        Err(e) => expect_eq(e.exit_code(), ExitCode::InvalidInput, "exit_code"),
        Ok(_) => Err("expected InvalidJobDir error".to_string()),
    }
}

/// 既存の `request.json` は上書きしない（`RequestFileGuard` の
/// `create_new`。issue #178 実装計画 3.2）。
fn case_existing_request_file(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "ok");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    std::fs::write(case_dir.join("request.json"), b"pre-existing")
        .expect("pre-create request.json");
    let result = run_train(&launcher, &request, case_dir, &limits);
    // 既存ファイルは削除しない（呼び出し元が作った物を勝手に消さない）。
    let existing = std::fs::read(case_dir.join("request.json")).unwrap_or_default();
    expect_true(
        existing == b"pre-existing",
        "pre-existing request.json must be left untouched",
    )?;
    match result {
        Err(e) => expect_eq(e.exit_code(), ExitCode::RuntimeError, "exit_code"),
        Ok(_) => Err("expected RequestWrite(AlreadyExists) error".to_string()),
    }
}
