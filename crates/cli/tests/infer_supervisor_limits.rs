//! 監督層（`infer_supervisor`）の時間・RSS 上限が実プロセスで効くこと（REQ-39・REQ-21・
//! TASK-39.5-3・#577）。
//!
//! 証拠種別: テストハーネス（テストバイナリ自身の再実行を子にし、模擬の小さな上限を渡す）。
//! 実 CLI の推論が 2 GiB を超える実測ではない。子の挙動は環境変数 `FANDHE_SUP_TEST_CHILD` で選ぶ。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use fandhe_edge_cli::infer_supervisor::run_supervised;
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_guard::resource::{GuardedCommand, MemoryLimit, RunConfig, TimeLimit};
use std::time::Duration;

const MODE_ENV: &str = "FANDHE_SUP_TEST_CHILD";
const DIR_ENV: &str = "FANDHE_SUP_TEST_DIR";
const MIB: usize = 1024 * 1024;

/// 自己再実行される子の本体。環境変数が無い通常実行では何もしない。
#[test]
fn child_entry() {
    let Ok(mode) = std::env::var(MODE_ENV) else {
        return;
    };
    match mode.as_str() {
        "alloc" => {
            // 圧縮されにくい内容で 8 MiB ずつ確保して保持する（20 秒で自走終了。暴走防止）。
            let mut chunks: Vec<Vec<u8>> = Vec::new();
            let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
            for _ in 0..(256 * MIB / (8 * MIB)) {
                let mut v = vec![0u8; 8 * MIB];
                for b in v.chunks_mut(8) {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    let bytes = x.to_le_bytes();
                    let n = b.len();
                    b.copy_from_slice(&bytes[..n]);
                }
                chunks.push(std::hint::black_box(v));
                std::thread::sleep(Duration::from_millis(5));
            }
            std::thread::sleep(Duration::from_secs(20));
            std::hint::black_box(&chunks);
        }
        "sleep" => std::thread::sleep(Duration::from_secs(30)),
        // `package` の kill 後の残骸を模す: ステージング（と、`publish` ならさらに `package/`）と台帳を作って眠る。
        "stage" | "publish" => {
            let proj = std::path::PathBuf::from(std::env::var(DIR_ENV).expect("dir"));
            std::fs::create_dir(proj.join("package.staging")).expect("staging");
            std::fs::write(proj.join("version_ledger.json"), b"{}").expect("ledger");
            if mode == "publish" {
                std::fs::create_dir(proj.join("package")).expect("package");
            }
            std::thread::sleep(Duration::from_secs(30));
        }
        // 組み立て中の worker が後始末なしに落ちる（panic と同じ 101 で終わる）。
        "crash" => {
            let proj = std::path::PathBuf::from(std::env::var(DIR_ENV).expect("dir"));
            std::fs::create_dir(proj.join("package.staging")).expect("staging");
            std::process::exit(101);
        }
        "ok" => println!("{{\"id\":\"input\",\"status\":\"ok\"}}"),
        _ => {}
    }
}

fn child(mode: &str) -> GuardedCommand {
    GuardedCommand::new(std::env::current_exe().expect("exe"))
        .expect("abs")
        .arg("--exact")
        .arg("child_entry")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(MODE_ENV, mode)
}

fn supervise(mode: &str, config: &RunConfig) -> (ExitCode, String) {
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = run_supervised(&mut out, &mut err, &child(mode), config).expect("io");
    (code, String::from_utf8(out).expect("utf8"))
}

fn config(time_ms: u64) -> RunConfig {
    RunConfig::new(
        TimeLimit::new(Duration::from_millis(time_ms)).expect("time"),
        1024 * 1024,
        64 * 1024,
    )
    .expect("config")
}

/// REQ-39: RSS 上限を超えたら exit 20・`limit_exceeded`（メモリ）。
#[test]
fn req39_memory_overrun_maps_to_limit_exceeded() {
    let cfg = config(20_000).with_memory_limit(MemoryLimit::new(64 * MIB as u64).expect("mem"));
    let (code, text) = supervise("alloc", &cfg);
    assert_eq!(code, ExitCode::LimitExceeded);
    assert_eq!(
        text,
        "{\"code\":\"limit_exceeded\",\"message\":\"inference memory limit exceeded\"}\n"
    );
}

/// REQ-39: 時間上限を超えたら exit 20・`limit_exceeded`（時間）。
#[test]
fn req39_time_overrun_maps_to_limit_exceeded() {
    let (code, text) = supervise("sleep", &config(300));
    assert_eq!(code, ExitCode::LimitExceeded);
    assert_eq!(
        text,
        "{\"code\":\"limit_exceeded\",\"message\":\"inference time limit exceeded\"}\n"
    );
}

/// REQ-39・REQ-41・TASK-41.8・#435: 追加学習の種類の設定（62.5 秒・24 GiB）を実プロセスに適用しても、
/// 上限内の子は止められず出力が中継される。24 GiB の上限が `run_with_limits` に受理される。
#[test]
fn req39_qwen2_lora_limits_apply_and_allow_within_limit_child() {
    use fandhe_edge_cli::args::InferSource;
    use fandhe_edge_cli::infer_supervisor::supervisor_config;
    use fandhe_edge_guard::resource::safety_limits_for_kind;
    let source = InferSource::Text {
        text: "x".to_string(),
        id: None,
    };
    let cfg = supervisor_config(&source, safety_limits_for_kind("qwen2_lora")).expect("config");
    assert_eq!(cfg.time_limit(), Duration::from_millis(62_500));
    assert_eq!(cfg.memory_limit(), Some(24 * 1024 * 1024 * 1024));
    let (code, text) = supervise("ok", &cfg);
    assert_eq!(code, ExitCode::Ok);
    assert!(
        text.contains("{\"id\":\"input\",\"status\":\"ok\"}\n"),
        "{text}"
    );
}

/// REQ-39: 上限内の子の stdout は変えずに中継する。テストバイナリの出力（libtest の行）が混ざるため、
/// 終了コードが 0 でない場合の写像だけでなく、中継で JSON 行が失われないことを確かめる。
#[test]
fn req39_within_limits_output_is_relayed() {
    let (code, text) = supervise("ok", &config(20_000));
    assert_eq!(code, ExitCode::Ok);
    assert!(
        text.contains("{\"id\":\"input\",\"status\":\"ok\"}\n"),
        "{text}"
    );
}

/// 一時 cwd と、その下の空のプロジェクト `proj` を作る。
fn package_project(case: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let cwd = std::fs::canonicalize(std::env::temp_dir())
        .expect("temp")
        .join(format!("fandhe-sup-{case}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cwd);
    let proj = cwd.join("proj");
    std::fs::create_dir_all(&proj).expect("proj");
    (cwd, proj)
}

/// 偽の package worker（`mode`）を 300 ms で kill させ、launch 前の記録に基づく後始末つきで監督する。
fn supervise_package(
    mode: &str,
    cwd: &std::path::Path,
    proj: &std::path::Path,
) -> (ExitCode, String) {
    use fandhe_edge_cli::infer_supervisor::{PackageRemnants, Workload, run_supervised_cleaning};
    let remnants = PackageRemnants::snapshot(cwd, std::path::Path::new("proj")).expect("lock");
    assert!(remnants.is_some());
    let cmd = child(mode).env(DIR_ENV, proj.to_str().expect("utf8"));
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = run_supervised_cleaning(
        Workload::Packaging,
        &mut out,
        &mut err,
        &cmd,
        &config(1_500),
        || {
            // 後始末の直前に、worker が（kill 前に）ステージングを作っていたことを確かめる。
            if mode != "sleep" {
                assert!(
                    proj.join("package.staging").is_dir(),
                    "worker did not stage"
                );
            }
            if let Some(r) = &remnants {
                r.discard_created();
            }
        },
    )
    .expect("io");
    (code, String::from_utf8(out).expect("utf8"))
}

/// REQ-39・#662: kill された `package` の worker が作ったステージングと台帳（`package/` は無い）は、
/// launch 前に無かったので親が片付ける。出力は `limit_exceeded` の固定語彙のまま。
#[test]
fn req39_issue662_killed_package_removes_staging_and_ledger_created_by_worker() {
    let (cwd, proj) = package_project("kill");
    let (code, text) = supervise_package("stage", &cwd, &proj);
    assert_eq!(code, ExitCode::LimitExceeded);
    assert_eq!(
        text,
        "{\"code\":\"limit_exceeded\",\"message\":\"packaging time limit exceeded\"}\n"
    );
    assert!(!proj.join("package.staging").exists());
    assert!(!proj.join("version_ledger.json").exists());
    assert!(!proj.join("package").exists());
}

/// REQ-39・#662: launch 前からあったステージング・台帳は利用者の物として消さない。
#[test]
fn req39_issue662_killed_package_keeps_staging_and_ledger_that_existed_before_launch() {
    let (cwd, proj) = package_project("keep");
    std::fs::create_dir(proj.join("package.staging")).expect("staging");
    std::fs::write(proj.join("package.staging").join("note"), b"x").expect("note");
    std::fs::write(proj.join("version_ledger.json"), b"old").expect("ledger");
    let (code, _) = supervise_package("sleep", &cwd, &proj);
    assert_eq!(code, ExitCode::LimitExceeded);
    assert!(proj.join("package.staging").join("note").is_file());
    assert_eq!(
        std::fs::read(proj.join("version_ledger.json")).expect("ledger"),
        b"old"
    );
}

/// REQ-39・#662: `package/` が出来ている場合、`package/` と対の台帳は消さない（ステージングだけ片付ける）。
#[test]
fn req39_issue662_killed_package_keeps_ledger_when_package_exists() {
    let (cwd, proj) = package_project("published");
    let (code, _) = supervise_package("publish", &cwd, &proj);
    assert_eq!(code, ExitCode::LimitExceeded);
    assert!(proj.join("package").is_dir());
    assert!(proj.join("version_ledger.json").is_file());
    assert!(!proj.join("package.staging").exists());
}

/// REQ-39・#662: 7 種に写らない終了（panic・シグナル死）で残ったステージングも親が片付ける。出力は
/// `runtime_error`。
#[test]
fn req39_issue662_crashed_package_removes_staging_created_by_worker() {
    let (cwd, proj) = package_project("crash");
    let (code, _) = supervise_package("crash", &cwd, &proj);
    assert_eq!(code, ExitCode::RuntimeError);
    assert!(!proj.join("package.staging").exists());
}

/// REQ-39・#662: 同じプロジェクトの `package` の親は 1 つだけ。ロックを持つ間の 2 つ目は 64 で拒否し、
/// 解放後は取れる（後始末が他の `package` の作ったものを消さないための排他）。
#[test]
fn req39_issue662_second_package_on_same_project_is_rejected_while_locked() {
    use fandhe_edge_cli::infer_supervisor::PackageRemnants;
    let (cwd, _proj) = package_project("lock");
    let dir = std::path::Path::new("proj");
    let first = PackageRemnants::snapshot(&cwd, dir).expect("first");
    assert!(first.is_some());
    let err = PackageRemnants::snapshot(&cwd, dir)
        .err()
        .expect("second is rejected");
    assert_eq!(err.code, ExitCode::InvalidInput);
    assert_eq!(err.message, "package already running");
    drop(first);
    assert!(
        PackageRemnants::snapshot(&cwd, dir)
            .expect("after release")
            .is_some()
    );
}
