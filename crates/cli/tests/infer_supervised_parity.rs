//! 監督つき `infer`（親が worker を子プロセスで起動）の出力が、監督なし（worker 直接実行）と
//! バイト単位で一致すること（REQ-28・REQ-39・TASK-39.5-3・#577）。
//!
//! 共有 fixture の C1 の ONNX でパッケージを組み、`--text`・`--input-file`（`--out` あり・なし）を
//! 両方の経路で実行して stdout・終了コード・`--out` の中身を比べる。あわせてバッチの各行が
//! 1 件ずつの実行と一致する。証拠種別: テストハーネス（実プロセス・合成パッケージ）。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const DEFINITION: &str = r#"{"schema":"fandhe-edge-model-definition/v1","name":"kv","version":1,"judgment_type":"single_select","options":[{"id":"alpha","display_name":"a","description":"d"},{"id":"beta","display_name":"b","description":"d"},{"id":"gamma","display_name":"g","description":"d"}],"io":{"input":"bytes"}}"#;
const INPUTS: [&str; 3] = ["hello world", "foo bar", "zzz"];

fn workspace(tag: &str) -> PathBuf {
    let ws = std::env::temp_dir().join(format!("fandhe-sup-parity-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(ws.join("p")).expect("mkdir");
    let onnx = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/onnx_parity/c1.onnx"),
    )
    .expect("fixture onnx");
    let sha = fandhe_edge_core::hash::Sha256Digest::of_bytes(&onnx).to_hex();
    std::fs::write(ws.join("p/model.onnx"), &onnx).expect("write onnx");
    std::fs::write(
        ws.join("p/artifact.json"),
        format!(
            r#"{{"kind":"c1","kind_version":1,"max_bytes":48,"label_order":["alpha","beta","gamma"],"onnx_file":"model.onnx","onnx_sha256":"{sha}"}}"#
        ),
    )
    .expect("write meta");
    std::fs::write(ws.join("p/definition.json"), DEFINITION).expect("write definition");
    let lines: String = INPUTS
        .iter()
        .enumerate()
        .map(|(i, t)| format!("{{\"id\":\"r{i}\",\"input\":\"{t}\"}}\n"))
        .collect();
    std::fs::write(ws.join("in.jsonl"), lines).expect("write input");
    ws
}

/// `worker` が真なら監督を外して実行する（`FANDHE_EDGE_INFER_WORKER=1`）。
fn run(ws: &Path, worker: bool, args: &[&str]) -> (Option<i32>, Vec<u8>) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"));
    cmd.arg("infer")
        .args(["--package", "p"])
        .args(args)
        .current_dir(ws)
        .stdin(Stdio::null());
    if worker {
        cmd.env("FANDHE_EDGE_INFER_WORKER", "1");
    }
    let out = cmd.output().expect("run");
    (out.status.code(), out.stdout)
}

/// REQ-39・REQ-28: `--text` は監督の有無で同一出力。
#[test]
fn req39_text_output_identical_with_and_without_supervisor() {
    let ws = workspace("text");
    for t in INPUTS {
        let a = run(&ws, false, &["--text", t]);
        let b = run(&ws, true, &["--text", t]);
        assert_eq!(a.0, Some(0));
        assert_eq!(a, b);
    }
    let _ = std::fs::remove_dir_all(&ws);
}

/// REQ-39・REQ-28: `--input-file` は監督の有無で同一出力で、各行が 1 件ずつの結果と一致する。
#[test]
fn req39_batch_output_identical_and_matches_single() {
    let ws = workspace("batch");
    let a = run(&ws, false, &["--input-file", "in.jsonl"]);
    let b = run(&ws, true, &["--input-file", "in.jsonl"]);
    assert_eq!(a.0, Some(0));
    assert_eq!(a, b);
    let text = String::from_utf8(a.1).expect("utf8");
    assert_eq!(text.lines().count(), INPUTS.len());
    for (i, (line, t)) in text.lines().zip(INPUTS).enumerate() {
        let single = run(&ws, true, &["--text", t, "--id", &format!("r{i}")]);
        assert_eq!(format!("{line}\n").into_bytes(), single.1);
    }
    let _ = std::fs::remove_dir_all(&ws);
}

/// REQ-39: `--out` あり。stdout の要約と OUT の中身が監督の有無で同一。
#[test]
fn req39_batch_out_identical_with_and_without_supervisor() {
    let ws = workspace("out");
    let a = run(
        &ws,
        false,
        &["--input-file", "in.jsonl", "--out", "a.jsonl"],
    );
    let b = run(
        &ws,
        true,
        &["--input-file", "in.jsonl", "--out", "a2.jsonl"],
    );
    assert_eq!(a.0, Some(0), "{}", String::from_utf8_lossy(&a.1));
    assert_eq!(b.0, Some(0));
    let norm = |v: &[u8], name: &str| String::from_utf8_lossy(v).replace(name, "OUT");
    assert_eq!(norm(&a.1, "a.jsonl"), norm(&b.1, "a2.jsonl"));
    let fa = std::fs::read(ws.join("a.jsonl")).expect("a");
    let fb = std::fs::read(ws.join("a2.jsonl")).expect("b");
    assert_eq!(fa, fb);
    let _ = std::fs::remove_dir_all(&ws);
}

/// REQ-39: 異常終了（存在しないパッケージ）も同一出力・同一終了コード（64）。
#[test]
fn req39_invalid_input_identical_with_and_without_supervisor() {
    let ws = workspace("bad");
    let run_bad = |worker: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"));
        cmd.args(["infer", "--package", "nope", "--text", "x"])
            .current_dir(&ws)
            .stdin(Stdio::null());
        if worker {
            cmd.env("FANDHE_EDGE_INFER_WORKER", "1");
        }
        let o = cmd.output().expect("run");
        (o.status.code(), o.stdout)
    };
    let a = run_bad(false);
    assert_eq!(a.0, Some(64));
    assert_eq!(a, run_bad(true));
    let _ = std::fs::remove_dir_all(&ws);
}
