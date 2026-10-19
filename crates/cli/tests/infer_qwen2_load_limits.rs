//! `qwen2_lora` の読み込み段階（shard の読み込み・sha256 照合・bf16 展開）が、推論呼び出しだけでなく
//! worker プロセス全体の時間・RSS の安全上限の下で走ること（REQ-39・REQ-41・REQ-21・#626）。
//!
//! 監督層（`infer_supervisor`）は `infer` を worker 子プロセスで起動し、起動から終了までの全体
//! （パッケージの読み込みを含む）に時間と RSS の上限を掛ける。ここでは実 CLI を worker として、
//! 模擬の小さな上限を渡し、読み込み中の超過が `limit_exceeded`（20）で止まることを固定する。
//! 証拠種別: テストハーネス（合成の極小パッケージ・模擬の上限。実重みの読み込みではない）。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use fandhe_edge_cli::infer_supervisor::{WORKER_ENV, run_supervised};
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_guard::resource::{GuardedCommand, MemoryLimit, RunConfig, TimeLimit};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn workspace(tag: &str) -> PathBuf {
    let ws = std::env::temp_dir().join(format!("fandhe-qwen2-load-{}-{tag}", std::process::id()));
    let _ = fs::remove_dir_all(&ws);
    let pkg = ws.join("p");
    fs::create_dir_all(&pkg).expect("mkdir");
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/qwen2_parity");
    for f in [
        "artifact.json",
        "model.onnx",
        "tokenizer.json",
        "weights-00000.bin",
    ] {
        fs::copy(src.join(f), pkg.join(f)).expect("copy");
    }
    let ids = ["red", "tan", "sky", "map", "fox", "owl", "gem", "ink"];
    let options: Vec<String> = ids
        .iter()
        .map(|l| format!(r#"{{"id":"{l}","display_name":"{l}","description":"d"}}"#))
        .collect();
    fs::write(
        pkg.join("definition.json"),
        format!(
            r#"{{"schema":"fandhe-edge-model-definition/v1","name":"q","version":1,"judgment_type":"single_select","options":[{}],"io":{{"input":"bytes"}}}}"#,
            options.join(",")
        ),
    )
    .expect("definition");
    ws
}

fn supervise(ws: &Path, config: &RunConfig) -> (ExitCode, String) {
    let mut cmd = GuardedCommand::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .expect("abs")
        .current_dir(ws)
        .env(WORKER_ENV, "1");
    for a in ["infer", "--package", "p", "--text", "hello"] {
        cmd = cmd.arg(a);
    }
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = run_supervised(&mut out, &mut err, &cmd, config).expect("io");
    (code, String::from_utf8(out).expect("utf8"))
}

fn config(time: Duration) -> RunConfig {
    RunConfig::new(TimeLimit::new(time).expect("time"), 1024 * 1024, 64 * 1024).expect("config")
}

/// 上限が十分なら同じ worker が推論まで完走する（対照。上の超過が上限由来であることの確認）。
#[test]
fn req39_qwen2_lora_worker_completes_within_limits() {
    let ws = workspace("ok");
    let (code, text) = supervise(&ws, &config(Duration::from_secs(30)));
    assert_eq!(code, ExitCode::Ok, "{text}");
    assert!(
        text.starts_with("{\"id\":\"input\",\"status\":\"ok\""),
        "{text}"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: 読み込みを含む worker の時間が上限を超えたら exit 20（読み込みだけが長くても止まる）。
/// 実測でこの worker は完走まで約 0.3 秒かかり、20 ms の上限は読み込み中に超過する。
#[test]
fn req39_qwen2_lora_load_phase_time_overrun_is_limit_exceeded() {
    let ws = workspace("time");
    let (code, text) = supervise(&ws, &config(Duration::from_millis(20)));
    assert_eq!(code, ExitCode::LimitExceeded, "{text}");
    assert_eq!(
        text,
        "{\"code\":\"limit_exceeded\",\"message\":\"inference time limit exceeded\"}\n"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: 読み込みを含む worker の RSS が上限を超えたら exit 20（メモリ）。
#[test]
fn req39_qwen2_lora_load_phase_memory_overrun_is_limit_exceeded() {
    let ws = workspace("mem");
    let cfg = config(Duration::from_secs(30))
        .with_memory_limit(MemoryLimit::new(1024 * 1024).expect("mem"));
    let (code, text) = supervise(&ws, &cfg);
    assert_eq!(code, ExitCode::LimitExceeded, "{text}");
    assert_eq!(
        text,
        "{\"code\":\"limit_exceeded\",\"message\":\"inference memory limit exceeded\"}\n"
    );
    let _ = fs::remove_dir_all(&ws);
}
