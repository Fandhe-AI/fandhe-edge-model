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
use fandhe_edge_train::job::{CancelOutcome, JobState, TrainJob};
#[cfg(unix)]
use fandhe_edge_train::process::ENV_ALLOWLIST;
#[cfg(unix)]
use fandhe_edge_train::process::TrainRunEnd;
#[cfg(unix)]
use fandhe_edge_train::process::WorkerCandidateRunner;
use fandhe_edge_train::process::{RunLimits, WorkerLauncher, run_train};
#[cfg(unix)]
use fandhe_edge_train::request::ValidationInput;
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
            // issue #178 PR #233 レビュー: 結果 JSON を出力（フラッシュ）
            // した後、実際にプロセスが終了するまでの間に短い遅延
            // （インタプリタのシャットダウン処理を模す）を挟んでも、
            // `run_train` が正しく成功として扱うことを検証する。
            //
            // `run_train` は標準出力の EOF を待った後、残りの壁時計予算の
            // 範囲で `try_wait()` をポーリングして実際の終了を待つ
            // （モジュール doc「完了検知・タイムアウト時の回収順序」
            // 参照）。標準ライブラリだけでは自プロセスの stdout（fd 1）を
            // `exit()` に先立って明示的に閉じる安全な手段が無い（`unsafe`
            // な生 fd 操作が要る。依存・`unsafe` の追加は禁止）ため、本
            // モードでは EOF の観測とプロセスの実終了がほぼ同時になるが、
            // 「出力後すぐには終了しない供給元」が誤ってタイムアウト扱いに
            // ならないことを確認する回帰テストとして意味を持つ。
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
        "hang_with_lifeline_orphan" => {
            // issue #178 PR #233 レビュー: Rust 側は直接の子（supervisor 役
            // ＝このプロセス）だけを把握・終了させればよく、孫プロセス
            // （`_worker` 役）の掃除は学習ワーカー側の lifeline に委ねる、
            // という新しい設計の核心を、実プロセスを使って end-to-end で
            // 検証する（`case_timeout_hang_kills_orphan` 参照）。
            //
            // 実際の `_worker`（`cli.py::_start_lifeline_thread`）は、
            // supervisor が `pass_fds` で渡した pipe の読み取り端を
            // block read し、supervisor の死（＝書き込み端の自動クローズ）
            // を EOF として検知して自己終了する。本モードはこれを Rust の
            // `Child::stdin`（`Stdio::piped()`）で模す: 孫の標準入力
            // （読み取り端）だけを孫へ渡し、書き込み端（`ChildStdin`）は
            // このプロセス（supervisor 役）が握り続ける。このプロセスが
            // 外側の壁時計タイムアウトで `Child::kill()`（`SIGKILL`）
            // されると、カーネルが書き込み端を含む全 fd を自動的に閉じる
            // ため、孫は何もしなくても EOF を観測できる。
            #[cfg(unix)]
            let _lifeline_write_end_keepalive = {
                use std::os::unix::process::CommandExt;
                let exe = std::env::current_exe().expect("resolve current_exe for orphan");
                let launch_script = std::env::current_dir()
                    .expect("cwd")
                    .join("orphan-launch.py");
                std::fs::write(&launch_script, "orphan_lifeline_hang")
                    .expect("write orphan launch.py");
                let mut command = std::process::Command::new(&exe);
                command
                    .arg("-I")
                    .arg(&launch_script)
                    .arg("train")
                    .arg("--request")
                    .arg(request_path)
                    // 孫を自分自身のプロセスグループのリーダーにする
                    // （実際の `_worker` が `start_new_session=True` で
                    // 別セッションのリーダーになるのと同じ理由: lifeline の
                    // EOF 検知時に「自分自身のグループ」へ `SIGKILL` できる
                    // ようにする）。
                    .process_group(0)
                    .stdin(std::process::Stdio::piped());
                let mut grandchild = command.spawn().expect("spawn orphan grandchild");
                std::fs::write("orphan.pid", grandchild.id().to_string())
                    .expect("write orphan.pid");
                // 標準入力（lifeline の書き込み端）を取り出し、この変数に
                // 束縛したまま関数の最後（＝無限ループで戻らない）まで
                // 保持する。孫には渡さない複製であり、このプロセスが
                // 終了する（正常終了・`SIGKILL` のいずれでも）まで開いた
                // ままになる。
                let stdin_keepalive = grandchild.stdin.take();
                // `wait()` せずに drop すると `clippy::zombie_processes` に
                // 抵触するため、別スレッドへ切り出して回収する（孫は
                // lifeline 経由の自己終了、または外側のタイムアウトで
                // `SIGKILL` されるまで終了しない）。
                std::thread::spawn(move || {
                    let mut grandchild = grandchild;
                    let _ = grandchild.wait();
                });
                stdin_keepalive
            };
            loop {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        "orphan_lifeline_hang" => {
            // `hang_with_lifeline_orphan` が起動する「孫プロセス」
            // （`_worker` 役）。実際の `cli.py::_start_lifeline_thread` と
            // 同じ仕組み（標準入力＝lifeline の読み取り端を別スレッドで
            // block read し、EOF を観測したら自分自身のプロセスグループへ
            // `SIGKILL` を送って自己終了する）を再現する。それまでの間は
            // `orphan-heartbeat.txt`（cwd = job_dir。親と同じ cwd を継承）
            // へ書き続け、外側からテストが生存を確認できるようにする。
            let pid = std::process::id();
            std::thread::spawn(move || {
                let mut stdin = std::io::stdin();
                let mut buf = [0u8; 1];
                loop {
                    match std::io::Read::read(&mut stdin, &mut buf) {
                        Ok(0) => break, // EOF: 親（supervisor 役）が死んだ。
                        Ok(_) => continue,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    }
                }
                let _ = std::process::Command::new("/bin/kill")
                    .args(["-KILL", "--", &format!("-{pid}")])
                    .env_clear()
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
            });
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
        "ok_with_orphan_stderr" => {
            // issue #178 PR #233 レビュー再々指摘 P1「stderr の回収待ちが
            // 残りの壁時計予算を無視して最大 `READER_DRAIN_TIMEOUT`（5 秒）
            // 待ってしまい、締め切りを過ぎた後に `StderrIncomplete`
            // （runtime_error=70）を誤って返す」の回帰。孫プロセスに
            // stderr（fd 2）の複製を継承させたまま（`Command` は明示的に
            // `.stderr(...)` を指定しない限り親の fd をそのまま継承する）
            // このプロセス（supervisor 役）自身は正常終了する。孫が
            // stderr を握り続ける間、`run_train` 側の stderr 読み取り
            // スレッドは EOF に達せずブロックし続ける。
            let exe = std::env::current_exe().expect("resolve current_exe for stderr holder");
            let launch_script = std::env::current_dir()
                .expect("cwd")
                .join("stderr-holder-launch.py");
            std::fs::write(&launch_script, "stderr_holder").expect("write stderr-holder launch.py");
            let mut command = std::process::Command::new(&exe);
            command
                .arg("-I")
                .arg(&launch_script)
                .arg("train")
                .arg("--request")
                .arg(request_path)
                // 標準入力・標準出力は明示的に `null` にする。指定しなければ
                // 孫は標準出力（fd 1）の複製も継承してしまい、`run_train` の
                // stdout 側の待ち（`stdout_wait`）が先にタイムアウトして
                // `WallTimeout` を返してしまう（本テストが検証したい
                // stderr 側の待ちに到達する前に、別の経路で同じ結果が
                // 出てしまい、本回帰テストが何も検証しないことになる）。
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null());
            // stderr だけは明示的に指定しない（親の fd をそのまま継承。
            // これが run_train が読み取っているパイプの書き込み端であり、
            // 孫がこれを握り続けることで stderr 側だけを EOF させない）。
            let grandchild = command.spawn().expect("spawn stderr holder grandchild");
            // `wait()` せずに drop すると `clippy::zombie_processes` に抵触する
            // ため、別スレッドへ切り出して回収する（孫は `stderr_holder`
            // モードの sleep 後に自発的に終了する）。
            std::thread::spawn(move || {
                let mut grandchild = grandchild;
                let _ = grandchild.wait();
            });
            print!("{}", ok_json(&format!("{FIXTURE_ROOT}/out")));
            std::process::exit(0);
        }
        "stderr_holder" => {
            // `ok_with_orphan_stderr` が起動する孫プロセス。親から継承した
            // stderr の複製を保持したまま、外側の壁時計タイムアウトより
            // 十分長く眠ってから終了する（標準出力には何も書かない。
            // `run_train` は直接の子＝`ok_with_orphan_stderr` 役の標準出力
            // だけを見るため孫の標準出力は無関係）。
            std::thread::sleep(Duration::from_secs(2));
            std::process::exit(0);
        }
        "ok_validation" | "ok_validation_big" | "ok_validation_over_cap" => {
            // issue #84 PR #238・選択肢 2: リクエストの `validation_inputs` の
            // `id` から `validation_predictions` を組み立てて返す（実際の
            // 学習ワーカーが学習直後に行う予測の代役）。`artifact_dir` は
            // リクエストの `root`／`out_dir` から作る（候補ごとに `out_dir` が
            // 異なるため）。`ok_validation_big` は長い id（リクエストが運ぶ）を返すため
            // 結果が既定上限（1 MiB）より大きくなる。`ok_validation_over_cap` は
            // さらに 70 MiB の空白を続けて validation 付きの上限（64 MiB）も超える。
            let request_json: serde_json::Value = serde_json::from_slice(
                &std::fs::read(request_path).unwrap_or_else(|e| panic!("read request: {e}")),
            )
            .expect("request json");
            let root = request_json["root"].as_str().expect("root");
            let out_dir = request_json["out_dir"].as_str().expect("out_dir");
            let mut value: serde_json::Value =
                serde_json::from_str(&ok_json(&format!("{root}/{out_dir}")))
                    .expect("ok json template");
            let inputs = request_json["validation_inputs"]
                .as_array()
                .expect("validation_inputs must be present");
            let predictions: Vec<serde_json::Value> = inputs
                .iter()
                .enumerate()
                .map(|(i, item)| {
                    // 各要素は `id`・`input` の 2 キーだけ（正解ラベルを持たない。REQ-27）。
                    assert_eq!(item.as_object().expect("object").len(), 2);
                    // 許可済みのラベル（`["a","b"]`）だけを返す。
                    let label = if i % 2 == 0 { "a" } else { "b" };
                    serde_json::json!({
                        "id": item["id"],
                        "status": "ok",
                        "predicted_label": label,
                    })
                })
                .collect();
            value["validation_predictions"] = serde_json::Value::Array(predictions);
            print!("{}", serde_json::to_string(&value).expect("serialize"));
            if mode == "ok_validation_over_cap" {
                let padding = vec![b' '; 70 * 1024 * 1024];
                let mut stdout = std::io::stdout();
                let _ = stdout.write_all(&padding);
            }
            std::process::exit(0);
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
        (
            "wall_timeout_when_grandchild_holds_stderr_open",
            case_wall_timeout_when_grandchild_holds_stderr_open,
        ),
        ("record_argv_and_request", case_record),
        (
            "validation_predictions_are_returned",
            case_validation_predictions_are_returned,
        ),
        (
            "validation_result_above_default_cap_is_accepted",
            case_validation_result_above_default_cap_is_accepted,
        ),
        (
            "validation_result_above_validation_cap_is_rejected",
            case_validation_result_above_validation_cap_is_rejected,
        ),
        (
            "search_scores_inside_the_training_job",
            case_search_scores_inside_the_training_job,
        ),
        (
            "search_records_wall_timeout_as_candidate_timeout",
            case_search_records_wall_timeout_as_candidate_timeout,
        ),
        ("invalid_job_dir", case_invalid_job_dir),
        ("existing_request_file_rejected", case_existing_request_file),
    ];
    #[cfg(unix)]
    cases.push(("timeout_hang_kills_orphan", case_timeout_hang_kills_orphan));
    #[cfg(unix)]
    cases.push(("cancel_hang", case_cancel_hang));
    #[cfg(unix)]
    cases.push((
        "cancel_kills_lifeline_orphan",
        case_cancel_kills_lifeline_orphan,
    ));
    #[cfg(unix)]
    cases.push(("cancel_before_start", case_cancel_before_start));
    #[cfg(unix)]
    cases.push((
        "cancel_after_exit_is_not_cancelled",
        case_cancel_after_exit_is_not_cancelled,
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

/// issue #178 PR #233 レビューへの回帰テスト（REQ-39）: 結果 JSON を出力
/// してから少し（壁時計予算に対して十分短い時間）遅れて `exit(0)` する
/// 供給元を、誤って `TerminatedBySignal`／`WallTimeout` 等に分類せず、
/// 正しく `Ok` と判定できること。
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

/// issue #178 PR #233 レビュー再々指摘 P1: 直接の子（supervisor 役）は
/// 締め切り内に正常終了し標準出力も読み切れるが、孫プロセスが標準エラー
/// 出力（fd 2）の複製を握ったまま締め切りを超えて生き続ける場合、
/// `StderrIncomplete`（runtime_error=70）ではなく `WallTimeout`
/// （limit_exceeded=20）として分類されること。孫が stderr の複製を
/// `READER_DRAIN_TIMEOUT`（5 秒）より十分長く（かつテストが速く終わる
/// 程度に短く）握り続けるようにし、壁時計の残り予算だけで打ち切られる
/// ことを確認する（`wall_timeout` は他のケースと同じ 500ms とし、冷えた
/// CI ランナーでの起動遅延が偽陰性〔旧経路のまま偶然 `WallTimeout` に
/// ならず素通りする〕を招かないようにする）。孫は stderr を 2 秒間
/// 握り続けるため、経過時間が 1.5 秒未満であることも合わせて確認し、
/// 「新しい stderr 側のガードで打ち切られた」ことと「孫の終了を待って
/// 約 2 秒後に戻った」ことを区別できるようにする。
#[cfg(unix)]
fn case_wall_timeout_when_grandchild_holds_stderr_open(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "ok_with_orphan_stderr");
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
        started.elapsed() < Duration::from_millis(1500),
        "must return around the wall timeout, not after waiting ~2s for the \
         grandchild holding stderr open to exit on its own",
    )
}

#[cfg(unix)]
fn make_request_with_validation(n: usize) -> TrainRequest {
    make_request(Some(30))
        .with_validation_inputs(
            (0..n)
                .map(|i| ValidationInput::new(format!("r{i}"), format!("input {i}")))
                .collect(),
        )
        .expect("valid validation inputs")
}

/// 選択肢 2（issue #84 PR #238・REQ-18・REQ-27）: `validation_inputs` を付けた
/// リクエストに対し、子プロセスの結果 JSON から `validation_predictions` を
/// `SuccessOutcome` として受け取れる（`id` は入力と同順・同件数）。
#[cfg(unix)]
fn case_validation_predictions_are_returned(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "ok_validation");
    let request = make_request_with_validation(4);
    let limits = RunLimits::for_request(&request);
    let run = run_train(&launcher, &request, case_dir, &limits)
        .map_err(|e| format!("run_train failed: {e}"))?;
    expect_eq(run.exit_code(), ExitCode::Ok, "exit_code")?;
    let fandhe_edge_train::result::TrainOutcome::Ok(success) = run.outcome() else {
        return Err("expected Ok outcome".to_string());
    };
    let predictions = success
        .validation_predictions()
        .ok_or_else(|| "validation_predictions must be present".to_string())?;
    let actual: Vec<(String, Option<String>)> = predictions
        .iter()
        .map(|p| (p.id().to_string(), p.predicted_label().map(str::to_string)))
        .collect();
    let expected: Vec<(String, Option<String>)> =
        [("r0", "a"), ("r1", "b"), ("r2", "a"), ("r3", "b")]
            .iter()
            .map(|(id, label)| ((*id).to_string(), Some((*label).to_string())))
            .collect();
    expect_eq(actual, expected, "validation_predictions")
}

/// 予測列を含む結果は、`validation_inputs` 付きのときだけ既定の上限
/// （`MAX_RESULT_BYTES` = 1 MiB）を超えても受理される（リクエストごとに計算した
/// 上限。`TrainRequest::max_result_bytes`）。1000 件 × 1000 バイトの id で、
/// 結果は約 1.05 MB（許可済みのラベルだけを返す正常な結果）。
#[cfg(unix)]
fn case_validation_result_above_default_cap_is_accepted(case_dir: &Path) -> Result<(), String> {
    const N: usize = 1000;
    const ID_LEN: usize = 1000;
    expect_true(
        N * (ID_LEN + 50) > fandhe_edge_train::limits::MAX_RESULT_BYTES,
        "the test result must exceed the default cap",
    )?;
    let launcher = make_launcher(case_dir, "ok_validation_big");
    let request = make_request(Some(30))
        .with_validation_inputs(
            (0..N)
                .map(|i| ValidationInput::new(format!("{i:0>ID_LEN$}"), String::new()))
                .collect(),
        )
        .expect("valid validation inputs");
    expect_true(
        request.max_result_bytes() > fandhe_edge_train::limits::MAX_RESULT_BYTES,
        "the per-request cap must exceed the default cap",
    )?;
    let limits = RunLimits::for_request(&request);
    let run = run_train(&launcher, &request, case_dir, &limits)
        .map_err(|e| format!("run_train failed: {e}"))?;
    let fandhe_edge_train::result::TrainOutcome::Ok(success) = run.outcome() else {
        return Err("expected Ok outcome".to_string());
    };
    expect_eq(
        success.validation_predictions().map(<[_]>::len),
        Some(N),
        "prediction count",
    )
}

/// リクエストごとに計算した上限（`TrainRequest::max_result_bytes`）を超える
/// 標準出力（70 MiB の空白を続ける）は `TooLarge` として拒否される（`RuntimeError`）。
#[cfg(unix)]
fn case_validation_result_above_validation_cap_is_rejected(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "ok_validation_over_cap");
    let request = make_request_with_validation(2);
    let limits = RunLimits::for_request(&request);
    match run_train(&launcher, &request, case_dir, &limits) {
        Err(e) => expect_eq(e.exit_code(), ExitCode::RuntimeError, "exit_code"),
        Ok(_) => Err("expected the oversized result to be rejected".to_string()),
    }
}

/// [`fandhe_edge_data::split::Groupable`] の最小実装（search の結合テストと同じ設計）。
#[cfg(unix)]
struct SplitItem {
    id: String,
    group_id: String,
    label: String,
    input: Vec<u8>,
}

#[cfg(unix)]
impl fandhe_edge_data::split::Groupable for SplitItem {
    fn id(&self) -> &str {
        &self.id
    }
    fn group_id(&self) -> &str {
        &self.group_id
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn input(&self) -> &[u8] {
        &self.input
    }
}

/// `record_ids`・`inputs`・`gold` を validation split として持つ凍結記録を作る
/// （比率 validation: 1.0。中身のハッシュを含む）。
#[cfg(unix)]
fn frozen_validation_split(
    record_ids: &[&str],
    inputs: &[&[u8]],
    gold: &[&str],
) -> fandhe_edge_data::split_record::SplitRecord {
    let items: Vec<SplitItem> = record_ids
        .iter()
        .zip(inputs.iter())
        .zip(gold.iter())
        .enumerate()
        .map(|(i, ((id, input), label))| SplitItem {
            id: (*id).to_string(),
            group_id: format!("g{i}"),
            label: (*label).to_string(),
            input: input.to_vec(),
        })
        .collect();
    let ratios = fandhe_edge_data::split::SplitRatios {
        train: 0.0,
        validation: 1.0,
        test: 0.0,
    };
    fandhe_edge_data::split_record::split_and_record(&items, 0, &ratios)
        .expect("valid ratios")
        .record()
        .clone()
}

/// 探索の入力（label_order は偽ワーカーの成果物と同じ `["a","b"]`）。
#[cfg(unix)]
fn search_candidates(ids: &[&str]) -> Vec<fandhe_edge_train::search::SearchCandidate> {
    ids.iter()
        .enumerate()
        .map(|(i, id)| fandhe_edge_train::search::SearchCandidate {
            candidate_id: (*id).to_string(),
            params: TrainRequestParams {
                kind: "c3".to_string(),
                kind_version: 1,
                config: serde_json::Map::new(),
                label_order: vec!["a".to_string(), "b".to_string()],
                max_bytes: 512,
                seed: i as u32,
                device: Device::Cpu,
                root: FIXTURE_ROOT.to_string(),
                train_path: "train.jsonl".to_string(),
                out_dir: format!("out-{id}"),
                time_limit_seconds: None,
                rss_limit_bytes: None,
            },
        })
        .collect()
}

/// 選択肢 2（issue #84 PR #238）の結合: `run_search` が `WorkerCandidateRunner`
/// （実際の子プロセス経路〔`run_train`〕）で候補を実行し、学習ジョブの結果に
/// 含まれる予測列から validation 正解率を算出して選定できる。
#[cfg(unix)]
fn case_search_scores_inside_the_training_job(case_dir: &Path) -> Result<(), String> {
    use fandhe_edge_train::search::{CandidateSearchResult, SearchBudget, SearchInput, run_search};
    use fandhe_edge_train::time_allotment::{PerCandidatePolicy, SystemClock};

    let launcher = make_launcher(case_dir, "ok_validation");
    let mut runner = WorkerCandidateRunner::new(&launcher, case_dir);
    let record_ids = ["r0", "r1", "r2", "r3"];
    let inputs: [&[u8]; 4] = [b"input 0", b"input 1", b"input 2", b"input 3"];
    // 偽ワーカーは a,b,a,b と予測する。gold は 3/4 が一致する並び。
    let gold = ["a", "b", "a", "a"];
    let split = frozen_validation_split(&record_ids, &inputs, &gold);
    let input = SearchInput {
        label_order: &["a", "b"],
        validation_gold: &gold,
        validation_record_ids: &record_ids,
        validation_inputs: &inputs,
        validation_split_record: &split,
        candidates: search_candidates(&["c3-a", "c3-b"]),
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };
    let record = run_search(&mut runner, &SystemClock::new(), input)
        .map_err(|e| format!("run_search failed: {e}"))?;
    expect_eq(record.candidates.len(), 2, "candidate count")?;
    for entry in &record.candidates {
        match &entry.result {
            CandidateSearchResult::Evaluated {
                validation_accuracy,
            } => {
                expect_eq(validation_accuracy.correct, 3, "correct")?;
                expect_eq(validation_accuracy.total, 4, "total")?;
            }
            other => return Err(format!("expected Evaluated, got {other:?}")),
        }
    }
    match &record.selection {
        fandhe_edge_train::search::SelectionDecision::Selected { candidate_id, .. } => {
            expect_eq(candidate_id.as_str(), "c3-a", "selected candidate")
        }
        other => Err(format!("expected Selected, got {other:?}")),
    }
}

/// 選択肢 2（issue #84 PR #238・REQ-39）の結合: 子プロセスが壁時計の締め切りで
/// 強制終了された候補は、探索全体の失敗ではなく `training_timed_out` として
/// 記録され、以降の候補は未着手になる（スレッド・`recv_timeout` は使わない）。
#[cfg(unix)]
fn case_search_records_wall_timeout_as_candidate_timeout(case_dir: &Path) -> Result<(), String> {
    use fandhe_edge_train::search::{CandidateSearchResult, SearchBudget, SearchInput, run_search};
    use fandhe_edge_train::time_allotment::{PerCandidatePolicy, SystemClock};

    let launcher = make_launcher(case_dir, "hang");
    let mut runner = WorkerCandidateRunner::new(&launcher, case_dir)
        .with_wall_timeout(Duration::from_millis(500));
    let record_ids = ["r0", "r1"];
    let inputs: [&[u8]; 2] = [b"input 0", b"input 1"];
    let gold = ["a", "b"];
    let split = frozen_validation_split(&record_ids, &inputs, &gold);
    let input = SearchInput {
        label_order: &["a", "b"],
        validation_gold: &gold,
        validation_record_ids: &record_ids,
        validation_inputs: &inputs,
        validation_split_record: &split,
        candidates: search_candidates(&["c3-a", "c3-b"]),
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::Fixed(std::num::NonZeroU32::new(1).expect("non-zero")),
    };
    let started = std::time::Instant::now();
    let record = run_search(&mut runner, &SystemClock::new(), input)
        .map_err(|e| format!("run_search must not fail on a candidate timeout: {e}"))?;
    expect_true(
        started.elapsed() < Duration::from_secs(10),
        "must return well before the 10s safety margin",
    )?;
    expect_eq(record.candidates.len(), 2, "candidate count")?;
    expect_eq(
        record.candidates[0].result.clone(),
        CandidateSearchResult::TrainingTimedOut,
        "first candidate",
    )?;
    // 候補単位の時間切れは探索全体を止めない: 予算が残っているため 2 件目も
    // 実行され（`hang` のため同じく時間切れ）、どちらも `training_timed_out`。
    expect_eq(
        record.candidates[1].result.clone(),
        CandidateSearchResult::TrainingTimedOut,
        "second candidate",
    )?;
    expect_eq(
        record.selection.clone(),
        fandhe_edge_train::search::SelectionDecision::NoEligibleCandidate,
        "selection",
    )
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

/// issue #178 PR #233 レビュー: Rust 側は直接の子（supervisor 役）だけを
/// 把握・`SIGKILL` すればよく、孫プロセス（`_worker` 役。lifeline を模した
/// もの）の掃除は Rust の関与なしに、カーネルが供給元の死で lifeline の
/// 書き込み端を自動的に閉じることを通じて、孫プロセス自身が自己終了する
/// ことを end-to-end で確認する（REQ-39）。`run_train` はこの孫プロセスの
/// 存在を一切知らない。
///
/// 実際の supervisor.py・`_worker` を Rust のテストハーネスから直接
/// 起動できないため、lifeline の仕組み自体（fd の継承・EOF 検知・自己
/// `killpg`）は Rust のフェイクワーカーで再現している
/// （`hang_with_lifeline_orphan`・`orphan_lifeline_hang` モード参照）。
/// 実際の supervisor.py・`_worker` を用いた同等の確認は
/// `trainer/tests/test_supervisor.py::
/// test_lifeline_worker_and_grandchild_die_when_supervisor_is_killed`
/// （Python 側）で行う。
#[cfg(unix)]
fn case_timeout_hang_kills_orphan(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "hang_with_lifeline_orphan");
    let request = make_request(Some(1));
    let limits = RunLimits::for_request(&request)
        .with_wall_timeout(Duration::from_millis(500))
        .expect("tighten wall timeout");

    // 孫プロセスが実際に起動したことを確認してから検証したいため、
    // `run_train`（500ms の壁時計タイムアウト）を別スレッドで実行しつつ、
    // メインスレッドで `orphan.pid` の出現を短いポーリングで待つ
    // （そもそも孫プロセスが起動する前に検証してしまうと「生存していない」
    // ことが偽陽性になるため）。ポーリングの成否は lifeline の正しさとは
    // 別の検証であり、タイムアウト発火前に孫プロセスの起動を確認できな
    // かった場合は「起動待ちタイムアウト」として区別できるよう別
    // メッセージで報告する。
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
        "grandchild (orphan.pid) must appear before the wall timeout fires",
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

/// `path` が現れるまで最大 `max` 待つ（キャンセルのタイミングを sleep
/// ではなく子の進行で決定的にする）。
#[cfg(unix)]
fn wait_for_file(path: &Path, max: Duration) -> bool {
    let deadline = std::time::Instant::now() + max;
    while std::time::Instant::now() < deadline {
        if path.exists() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    path.exists()
}

/// `TrainJob::run` を別スレッドで走らせ、`ready` が現れたらキャンセルして
/// 結果を返す共通手順。壁時計は 20 秒に設定し、キャンセル起因の停止と
/// 壁時計起因の停止を区別できるようにする。
#[cfg(unix)]
fn run_and_cancel_when(
    case_dir: &Path,
    mode: &str,
    ready: &Path,
) -> Result<(CancelOutcome, TrainRunEnd, JobState, Duration), String> {
    let launcher = make_launcher(case_dir, mode);
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request)
        .with_wall_timeout(Duration::from_secs(20))
        .map_err(|e| format!("with_wall_timeout: {e}"))?;
    let job = TrainJob::new();
    let handle = job.handle();
    let dir = case_dir.to_path_buf();
    let worker = std::thread::spawn(move || job.run(&launcher, &request, &dir, &limits));
    if !wait_for_file(ready, Duration::from_secs(10)) {
        // 進行しないまま残さないよう、待ちきれなくても止める。
        handle.cancel();
        let _ = worker.join();
        return Err(format!("{} did not appear in time", ready.display()));
    }
    let started = std::time::Instant::now();
    let outcome = handle.cancel();
    let end = worker
        .join()
        .map_err(|_| "job thread panicked".to_string())?
        .map_err(|e| format!("run returned error: {e}"))?;
    Ok((outcome, end, handle.state(), started.elapsed()))
}

/// REQ-34・TASK-34.1-1: キャンセル要求で hang 中の学習プロセスが止まり、
/// ジョブ状態が `Cancelled` になる。
#[cfg(unix)]
fn case_cancel_hang(case_dir: &Path) -> Result<(), String> {
    let heartbeat = case_dir.join("heartbeat.txt");
    let (outcome, end, state, elapsed) = run_and_cancel_when(case_dir, "hang", &heartbeat)?;
    expect_eq(outcome, CancelOutcome::Requested, "cancel outcome")?;
    let TrainRunEnd::Cancelled(run) = end else {
        return Err("expected Cancelled".to_string());
    };
    expect_eq(run.child_spawned(), true, "child_spawned")?;
    expect_eq(run.child_reaped(), true, "child_reaped")?;
    expect_eq(run.signal(), Some(9), "signal")?;
    expect_eq(state, JobState::Cancelled, "job state")?;
    expect_true(
        elapsed < Duration::from_secs(5),
        "must stop well before the 20s wall timeout",
    )?;
    let before = std::fs::metadata(&heartbeat).map(|m| m.len()).unwrap_or(0);
    std::thread::sleep(Duration::from_millis(300));
    let after = std::fs::metadata(&heartbeat).map(|m| m.len()).unwrap_or(0);
    expect_eq(after, before, "heartbeat must not grow after cancel")
}

/// REQ-34・REQ-39: キャンセル後、lifeline で孫プロセスも止まる。
#[cfg(unix)]
fn case_cancel_kills_lifeline_orphan(case_dir: &Path) -> Result<(), String> {
    let pid_path = case_dir.join("orphan.pid");
    let (_, end, state, _) = run_and_cancel_when(case_dir, "hang_with_lifeline_orphan", &pid_path)?;
    expect_true(
        matches!(end, TrainRunEnd::Cancelled(_)),
        "expected Cancelled",
    )?;
    expect_eq(state, JobState::Cancelled, "job state")?;
    let hb = case_dir.join("orphan-heartbeat.txt");
    // 孫の停止（lifeline の EOF 検知）を待ってから、伸びが止まることを見る。
    std::thread::sleep(Duration::from_millis(500));
    let before = std::fs::metadata(&hb).map(|m| m.len()).unwrap_or(0);
    std::thread::sleep(Duration::from_millis(500));
    let after = std::fs::metadata(&hb).map(|m| m.len()).unwrap_or(0);
    expect_eq(after, before, "orphan heartbeat must not grow after cancel")
}

/// REQ-34: 起動前のキャンセルは子を起動せず、`request.json` も残さない。
#[cfg(unix)]
fn case_cancel_before_start(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "hang");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    let job = TrainJob::new();
    let handle = job.handle();
    expect_eq(handle.cancel(), CancelOutcome::Requested, "cancel outcome")?;
    let end = job
        .run(&launcher, &request, case_dir, &limits)
        .map_err(|e| format!("run returned error: {e}"))?;
    let TrainRunEnd::Cancelled(run) = end else {
        return Err("expected Cancelled".to_string());
    };
    expect_eq(run.child_spawned(), false, "child_spawned")?;
    expect_eq(run.signal(), None, "signal")?;
    expect_eq(handle.state(), JobState::Cancelled, "job state")?;
    expect_true(!case_dir.join("heartbeat.txt").exists(), "no heartbeat")?;
    expect_true(!case_dir.join("request.json").exists(), "no request.json")
}

/// REQ-34: 終了後の `cancel()` は何もせず、状態は `Succeeded` のまま。
#[cfg(unix)]
fn case_cancel_after_exit_is_not_cancelled(case_dir: &Path) -> Result<(), String> {
    let launcher = make_launcher(case_dir, "ok");
    let request = make_request(Some(30));
    let limits = RunLimits::for_request(&request);
    let job = TrainJob::new();
    let handle = job.handle();
    let end = job
        .run(&launcher, &request, case_dir, &limits)
        .map_err(|e| format!("run returned error: {e}"))?;
    let TrainRunEnd::Completed(run) = end else {
        return Err("expected Completed".to_string());
    };
    expect_eq(run.exit_code(), ExitCode::Ok, "exit_code")?;
    expect_eq(
        handle.cancel(),
        CancelOutcome::AlreadyFinished,
        "cancel outcome",
    )?;
    expect_eq(handle.state(), JobState::Succeeded, "job state")
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
        expect_true(
            ENV_ALLOWLIST.contains(&name),
            &format!("child env var {name:?} must be in ENV_ALLOWLIST"),
        )?;
    }
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
