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
#[cfg(unix)]
use fandhe_edge_train::process::{ENV_ALLOWLIST, SUPERVISOR_GROUP_MANAGED_ENV};
use fandhe_edge_train::process::{RunLimits, WorkerLauncher, run_train};
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
        "ok_slow_exit" => {
            // issue #178 PR #233 レビュー再々々指摘 P1「stdout の EOF は
            // supervisor の終了を保証しない」の再現・検証用モード。結果
            // JSON を出力（フラッシュ）した後、実際にプロセスが終了する
            // までの間に短い遅延（インタプリタのシャットダウン処理を
            // 模す）を挟む。`EOF_EXIT_GRACE`（2 秒）より十分短くすることで、
            // `run_train` が猶予内に自発的な終了を確認し、成功として
            // 扱うことを検証する。
            //
            // 標準ライブラリだけでは自プロセスの stdout（fd 1）を
            // `exit()` に先立って明示的に閉じる安全な手段が無い
            // （`unsafe` な生 fd 操作が要る。依存・`unsafe` の追加は
            // 禁止）ため、本モードは「EOF の観測が実際のプロセス終了より
            // 先行する」という codex 指摘の競合そのものではなく、
            // `run_train` 側の猶予ポーリング（`poll_wait_bounded` が複数
            // 回ポーリングしてから終了を確認する経路）を確実に運動させる
            // ことで、猶予の導入が正常系を壊していないことを検証する。
            print!("{}", ok_json(&format!("{FIXTURE_ROOT}/out")));
            use std::io::Write as _;
            let _ = std::io::stdout().flush();
            std::thread::sleep(Duration::from_millis(300));
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
        "hang_with_orphan" => {
            // issue #178 PR #233 レビュー再々指摘の再現・検証用モード。
            // 新しいアーキテクチャでは `_worker`（ここでは「孫プロセス」）を
            // 別セッションへ切り離さない（`trainer/src/fandhe_edge_trainer/
            // supervisor.py` が `start_new_session` を使わない設計に変えた
            // ことのハーネス側の対応。`.process_group(0)` を**呼ばない**ことで、
            // `run_train` が確立したプロセスグループへそのまま留まる）。
            // 自分自身（このテストバイナリ）を「孫プロセス」として起動し、
            // 孫の pid を `orphan.pid`（cwd = job_dir）へ書いてから、自分
            // （「supervisor 役」）は標準出力を閉じずに応答不能なまま待ち
            // 続ける（外側の壁時計タイムアウトで、プロセスグループごと
            // `SIGKILL` されるまで自発的に終了しない）。
            #[cfg(unix)]
            {
                let exe = std::env::current_exe().expect("resolve current_exe for orphan");
                let launch_script = std::env::current_dir()
                    .expect("cwd")
                    .join("orphan-launch.py");
                std::fs::write(&launch_script, "orphan_hang").expect("write orphan launch.py");
                let mut command = std::process::Command::new(&exe);
                command
                    .arg("-I")
                    .arg(&launch_script)
                    .arg("train")
                    .arg("--request")
                    .arg(request_path);
                let grandchild = command.spawn().expect("spawn orphan grandchild");
                std::fs::write("orphan.pid", grandchild.id().to_string())
                    .expect("write orphan.pid");
                // 生成した `Child` を `wait()` せずに drop すると
                // `clippy::zombie_processes` に抵触する。本テストの主眼は
                // 「supervisor 役から見て孫プロセスが生きたまま応答不能に
                // なる」状況の再現であり、`wait()` 自体は本題ではないため、
                // 別スレッドへ切り出して回収する（孫プロセスは外側の
                // タイムアウトで `SIGKILL` されるまで終了しないため、この
                // `wait()` は `run_train` 側の強制終了後に完了する）。
                std::thread::spawn(move || {
                    let mut grandchild = grandchild;
                    let _ = grandchild.wait();
                });
            }
            loop {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        "exit_with_orphan" => {
            // issue #178 PR #233 レビュー再々指摘の再現・検証用モード。
            // `hang_with_orphan` と同様に孫プロセス（`_worker` 役）を
            // **同じプロセスグループに留めたまま**起動し（`.process_group(0)`
            // を呼ばない）、孫の標準出力を自分（supervisor 役）の標準出力
            // （`run_train` が読み取るパイプ）へ継承させたまま、supervisor
            // 役自身は正常終了する。孫が標準出力の書き手を握り続けるため、
            // `run_train` の標準出力読み取りは EOF に達せず、壁時計予算を
            // 使い切って `WallTimeout` になる。旧実装（`ps` によるプロセス
            // ツリー走査）で必要だった「孫プロセスの起動後に猶予を置いて
            // からでないと supervisor が終了してはならない」という制約
            // （スナップショットのタイミング依存）は、プロセスグループへの
            // 一括 `SIGKILL` 方式では不要になったため削除した
            // （`case_exit_with_orphan_kills_orphan` 参照）。
            #[cfg(unix)]
            {
                let exe = std::env::current_exe().expect("resolve current_exe for orphan");
                let launch_script = std::env::current_dir()
                    .expect("cwd")
                    .join("orphan-launch.py");
                std::fs::write(&launch_script, "orphan_hang").expect("write orphan launch.py");
                let mut command = std::process::Command::new(&exe);
                command
                    .arg("-I")
                    .arg(&launch_script)
                    .arg("train")
                    .arg("--request")
                    .arg(request_path)
                    // 標準エラー出力は継承させない（検証したいのは標準
                    // 出力側の読み取り未完了だけに絞るため）。
                    .stderr(std::process::Stdio::null());
                let grandchild = command.spawn().expect("spawn orphan grandchild");
                std::fs::write("orphan.pid", grandchild.id().to_string())
                    .expect("write orphan.pid");
                std::thread::spawn(move || {
                    let mut grandchild = grandchild;
                    let _ = grandchild.wait();
                });
            }
            print!("{}", ok_json(&format!("{FIXTURE_ROOT}/out")));
            std::process::exit(0);
        }
        "orphan_hang" => {
            // `hang_with_orphan` が起動する「孫プロセス」役。`heartbeat.txt`
            // （cwd = job_dir。親と同じ cwd を継承）へ書き続け、外側の
            // タイムアウト経路が本プロセスも回収できたかをテストから
            // 確認できるようにする。
            loop {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("orphan-heartbeat.txt")
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
    // `run_train` は unix 限定（issue #178 PR #233 レビュー再指摘 P0
    // 「Windows で正常終了後の孤児ワーカーを停止できない」への対応として
    // windows 等は `UnsupportedPlatform` を返す fail-closed 版へ差し替えた。
    // そのため、実ワーカー起動を前提とするケース群は unix 限定とし、
    // windows 等では fail-closed の確認ケースだけを実行する
    // （`crate::process` モジュール doc「windows（対象外・fail-closed）」
    // 参照）。
    #[cfg(unix)]
    let mut cases: Vec<(&'static str, CaseFn)> = vec![
        ("ok_outcome", case_ok_outcome),
        ("ok_slow_exit", case_ok_slow_exit),
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
    #[cfg(unix)]
    cases.push(("timeout_hang_kills_orphan", case_timeout_hang_kills_orphan));
    #[cfg(unix)]
    cases.push((
        "exit_with_orphan_kills_orphan",
        case_exit_with_orphan_kills_orphan,
    ));
    #[cfg(not(unix))]
    let cases: Vec<(&'static str, CaseFn)> =
        vec![("unsupported_platform", case_unsupported_platform)];

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
        // ケース終了時に一時ディレクトリを削除する（issue #178 実装計画
        // 3.8）。`run_train` が既に子プロセスを kill・回収済みのため
        // （`timeout_hang` を含む）、削除は安全。失敗しても後続ケースの
        // 判定には影響しないため無視する。
        let _ = std::fs::remove_dir_all(&case_dir);
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
#[cfg(unix)]
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

/// issue #178 PR #233 レビュー再々々指摘 P1「stdout の EOF は supervisor の
/// 終了を保証しない」への回帰テスト（REQ-39）: 結果 JSON を出力してから
/// 少し（`EOF_EXIT_GRACE` より十分短い時間）遅れて `exit(0)` する供給元を、
/// 誤って `TerminatedBySignal`／`WallTimeout` 等に分類せず、正しく `Ok` と
/// 判定できること。
#[cfg(unix)]
fn case_ok_slow_exit(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "ok_slow_exit");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    let run = run_train(&launcher, &request, case_dir, &limits)
        .map_err(|e| format!("run_train failed: {e}"))?;
    expect_eq(run.exit_code(), ExitCode::Ok, "exit_code")
}

/// 受け入れ条件 3: ワーカーのエラー（`invalid_request`）が `InvalidInput`
/// （64）へ写る。
#[cfg(unix)]
fn case_error_invalid_request(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "error_invalid_request");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    let run = run_train(&launcher, &request, case_dir, &limits)
        .map_err(|e| format!("run_train failed: {e}"))?;
    expect_eq(run.exit_code(), ExitCode::InvalidInput, "exit_code")
}

/// 受け入れ条件 3: `training_diverged` が `Pending`（12）へ写る。
#[cfg(unix)]
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
#[cfg(unix)]
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
#[cfg(unix)]
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
#[cfg(unix)]
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
#[cfg(unix)]
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
#[cfg(unix)]
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
#[cfg(unix)]
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

/// issue #178 PR #233 レビュー再々指摘の再現・検証: supervisor 役
/// （`hang_with_orphan`）が同じプロセスグループ内で起動した孫プロセス
/// （`_worker` 役。応答不能な supervisor に代わって走り続ける）も、外側の
/// 壁時計タイムアウト（プロセスグループへの一括 `SIGKILL`）で確実に終了
/// することを確認する（REQ-39）。
#[cfg(unix)]
fn case_timeout_hang_kills_orphan(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "hang_with_orphan");
    let request = make_request(Some(1));
    let limits = RunLimits::for_request(&request)
        .with_wall_timeout(Duration::from_millis(500))
        .expect("tighten wall timeout");

    // 孫プロセスが実際に起動したことを確認してから検証したいため、
    // `run_train`（500ms の壁時計タイムアウト）を別スレッドで実行しつつ、
    // メインスレッドで `orphan.pid` の出現を短いポーリングで待つ
    // （プロセスグループへの一括 `SIGKILL` 方式では、掃除の成否はもはや
    // `ps` スナップショットのタイミングに依存しないが、そもそも孫プロセスが
    // 起動する前に検証してしまうと「生存していない」ことが偽陽性になる
    // ため、本ポーリングは引き続き必要）。ポーリングの成否は掃除の正しさ
    // とは別の検証であり、タイムアウト発火前に孫プロセスの起動を確認
    // できなかった場合は「起動待ちタイムアウト」として区別できるよう
    // 別メッセージで報告する。
    let orphan_pid_path = case_dir.join("orphan.pid");
    let run_train_handle = {
        let launcher = launcher.clone();
        let request = request.clone();
        let case_dir = case_dir.to_path_buf();
        std::thread::spawn(move || run_train(&launcher, &request, &case_dir, &limits))
    };

    let poll_deadline = std::time::Instant::now() + Duration::from_millis(450);
    let mut orphan_spawned = false;
    while std::time::Instant::now() < poll_deadline {
        if orphan_pid_path.exists() {
            orphan_spawned = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }

    let err = match run_train_handle
        .join()
        .map_err(|_| "run_train thread panicked".to_string())?
    {
        Err(e) => e,
        Ok(_) => return Err("expected WallTimeout error".to_string()),
    };
    expect_eq(err.exit_code(), ExitCode::LimitExceeded, "exit_code")?;
    expect_true(
        orphan_spawned,
        "grandchild (orphan.pid) must appear before the wall timeout fires \
         (otherwise the tree-kill snapshot cannot have included it)",
    )?;

    let orphan_pid_text = std::fs::read_to_string(&orphan_pid_path)
        .map_err(|e| format!("read orphan.pid: {e} (grandchild may not have started in time)"))?;
    let orphan_pid: u32 = orphan_pid_text
        .trim()
        .parse()
        .map_err(|e| format!("parse orphan.pid {orphan_pid_text:?}: {e}"))?;

    // 孫プロセスが生きていれば heartbeat が伸び続けるはずなので、少し待って
    // `orphan-heartbeat.txt` が伸びていないことを確認する（`run_train` の
    // 戻り値だけでなく、実際に孫プロセスが止まったことを外部から観測する）。
    let heartbeat = case_dir.join("orphan-heartbeat.txt");
    let len_after_return = std::fs::metadata(&heartbeat).map(|m| m.len()).unwrap_or(0);
    std::thread::sleep(Duration::from_millis(500));
    let len_after_wait = std::fs::metadata(&heartbeat).map(|m| m.len()).unwrap_or(0);
    expect_eq(
        len_after_wait,
        len_after_return,
        "orphan heartbeat must not grow after run_train returns (grandchild must be killed too)",
    )?;

    // `kill -0 <pid>` は送信対象が存在すれば 0、存在しなければ非 0 で
    // 終了する（`/bin/kill` は本テストが検証対象とする `run_train` 側の
    // 実装が使う同じバイナリ。テスト側の検証にも同じ絶対パスの外部
    // コマンドを再利用する）。
    let status = std::process::Command::new("/bin/kill")
        .args(["-0", &orphan_pid.to_string()])
        .env_clear()
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| format!("failed to run /bin/kill -0: {e}"))?;
    expect_true(
        !status.success(),
        "orphan process must no longer exist after run_train returns",
    )
}

/// issue #178 PR #233 レビュー再々指摘の再現・検証: supervisor 役
/// （`exit_with_orphan`）が正常終了した後も、同じプロセスグループ内で
/// 起動した孫プロセス（`_worker` 役）が標準出力の書き手を握り続けている
/// ケースで、`run_train` の完了検知（標準出力の EOF 待ち）が壁時計予算を
/// 使い切って `WallTimeout` になり、かつプロセスグループへの一括
/// `SIGKILL` で孫プロセスを確実に停止できることを確認する（REQ-39）。
///
/// 以前（`ps` によるプロセスツリー走査方式）は、supervisor が既に終了して
/// いると `ppid` チェーンを辿れず、ポーリング中に記録した pid スナップ
/// ショットへの fallback が必要だった。プロセスグループへの一括シグナルに
/// 全面移行したことで、supervisor の生死やタイミングに関係なく孫プロセスへ
/// 確実に届く（モジュール doc「プロセスグループによる一括終了」参照）。
#[cfg(unix)]
fn case_exit_with_orphan_kills_orphan(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "exit_with_orphan");
    let request = make_request(Some(1));
    let limits = RunLimits::for_request(&request)
        .with_wall_timeout(Duration::from_millis(500))
        .expect("tighten wall timeout");

    let orphan_pid_path = case_dir.join("orphan.pid");
    let started = std::time::Instant::now();
    let err = match run_train(&launcher, &request, case_dir, &limits) {
        Err(e) => e,
        Ok(_) => {
            return Err("expected WallTimeout error (orphan holds stdout pipe open)".to_string());
        }
    };
    expect_true(
        orphan_pid_path.exists(),
        "grandchild (orphan.pid) must have been spawned before supervisor exited",
    )?;
    expect_eq(err.exit_code(), ExitCode::LimitExceeded, "exit_code")?;
    // supervisor 役はほぼ即座に正常終了するが、孫プロセスが標準出力を
    // 握り続けるため、`run_train` の標準出力読み取りは EOF に達せず、
    // 壁時計予算（500ms）を使い切って `WallTimeout` になる。無期限の
    // ハングにはならないことを安全マージン込みで確認する。
    expect_true(
        started.elapsed() < Duration::from_secs(10),
        "must return well before an unbounded hang would",
    )?;

    let orphan_pid_text = std::fs::read_to_string(&orphan_pid_path)
        .map_err(|e| format!("read orphan.pid: {e} (grandchild may not have started in time)"))?;
    let orphan_pid: u32 = orphan_pid_text
        .trim()
        .parse()
        .map_err(|e| format!("parse orphan.pid {orphan_pid_text:?}: {e}"))?;

    // 孫プロセスが生きていれば heartbeat が伸び続けるはずなので、少し待って
    // `orphan-heartbeat.txt` が伸びていないことを確認する。
    let heartbeat = case_dir.join("orphan-heartbeat.txt");
    let len_after_return = std::fs::metadata(&heartbeat).map(|m| m.len()).unwrap_or(0);
    std::thread::sleep(Duration::from_millis(500));
    let len_after_wait = std::fs::metadata(&heartbeat).map(|m| m.len()).unwrap_or(0);
    expect_eq(
        len_after_wait,
        len_after_return,
        "orphan heartbeat must not grow after run_train returns (grandchild must be killed too)",
    )?;

    let status = std::process::Command::new("/bin/kill")
        .args(["-0", &orphan_pid.to_string()])
        .env_clear()
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| format!("failed to run /bin/kill -0: {e}"))?;
    expect_true(
        !status.success(),
        "orphan process must no longer exist after run_train returns",
    )
}

/// `record` モード: argv・request.json の内容・環境変数名の許可リスト
/// 準拠を確認する（issue #178 実装計画 3.1〜3.3・6.1「record」ケース）。
#[cfg(unix)]
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
        // `SUPERVISOR_GROUP_MANAGED_ENV` は `ENV_ALLOWLIST`（親環境からの
        // 継承リスト）には含まれない。`run_train` が固定値
        // （`SUPERVISOR_GROUP_MANAGED_VALUE`）を明示的に設定する内部境界の
        // 環境変数であり、親プロセスの環境値を継承するものではないため
        // （issue #178 PR #233 レビュー再々々指摘 P0）。
        expect_true(
            ENV_ALLOWLIST.contains(&name) || name == SUPERVISOR_GROUP_MANAGED_ENV,
            &format!(
                "child env var {name:?} must be in ENV_ALLOWLIST or be the group-managed marker"
            ),
        )?;
    }
    let env_names_set: Vec<&str> = env_names
        .iter()
        .map(|v| v.as_str().unwrap_or_default())
        .collect();
    expect_true(
        env_names_set.contains(&SUPERVISOR_GROUP_MANAGED_ENV),
        "run_train must always set SUPERVISOR_GROUP_MANAGED_ENV on the worker process",
    )?;
    Ok(())
}

/// 起動口・job_dir の不正: 存在しない `job_dir` は `InvalidJobDir` として
/// 拒否する（子プロセスを起動しない）。
#[cfg(unix)]
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
#[cfg(unix)]
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

/// issue #178 PR #233 レビュー再指摘 P0「Windows で正常終了後の孤児ワーカー
/// を停止できない」への対応: unix 以外では `run_train` は子プロセスを一切
/// 起動せず、即座に `UnsupportedPlatform`（`RuntimeError`＝70）を返す
/// （fail-closed。`crate::process` モジュール doc「windows（対象外・
/// fail-closed）」参照）。偽ワーカー（`ok` モード）を指す起動口を渡しても
/// 起動されないことを、`request.json` が書き込まれないことで確認する。
#[cfg(not(unix))]
fn case_unsupported_platform(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "ok");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    match run_train(&launcher, &request, case_dir, &limits) {
        Err(e) => {
            expect_eq(e.exit_code(), ExitCode::RuntimeError, "exit_code")?;
            expect_true(
                !case_dir.join("request.json").exists(),
                "request.json must not be written when platform is unsupported",
            )
        }
        Ok(_) => Err("expected UnsupportedPlatform error".to_string()),
    }
}
