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

/// REQ-39・REQ-41・TASK-41.8・#435: 追加学習の種類の設定（62.5 秒・8 GiB）を実プロセスに適用しても、
/// 上限内の子は止められず出力が中継される。8 GiB の上限が `run_with_limits` に受理される。
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
    assert_eq!(cfg.memory_limit(), Some(8 * 1024 * 1024 * 1024));
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
