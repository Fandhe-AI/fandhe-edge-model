//! `infer` の `kind_version` 許可リスト検証（REQ-39・TASK-33.1-2・#136）。
//!
//! 共有 fixture の C1 の ONNX（`fixtures/onnx_parity/c1.onnx`）でパッケージを組み、許可された版
//! （1）は推論に進み、未許可の版（0・2）は `invalid_input`（64）で拒否されることを確かめる
//! （証拠種別: テストハーネス）。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::path::Path;
use std::process::{Command, Stdio};

const DEFINITION: &str = r#"{"schema":"fandhe-edge-model-definition/v1","name":"kv","version":1,"judgment_type":"single_select","options":[{"id":"alpha","display_name":"a","description":"d"},{"id":"beta","display_name":"b","description":"d"},{"id":"gamma","display_name":"g","description":"d"}],"io":{"input":"bytes"}}"#;

fn run_with_version(tag: &str, kind_version: u32) -> (Option<i32>, String) {
    let ws = std::env::temp_dir().join(format!("fandhe-kind-version-{}-{tag}", std::process::id()));
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
            r#"{{"kind":"c1","kind_version":{kind_version},"max_bytes":48,"label_order":["alpha","beta","gamma"],"onnx_file":"model.onnx","onnx_sha256":"{sha}"}}"#
        ),
    )
    .expect("write meta");
    std::fs::write(ws.join("p/definition.json"), DEFINITION).expect("write definition");
    let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .args(["infer", "--package", "p", "--text", "hello world"])
        .current_dir(&ws)
        .stdin(Stdio::null())
        .output()
        .expect("run");
    let _ = std::fs::remove_dir_all(&ws);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// REQ-39: 許可された版（1）は推論に進む。
#[test]
fn req39_allowed_kind_version_infers() {
    let (code, stdout) = run_with_version("v1", 1);
    assert_eq!(code, Some(0), "stdout: {stdout}");
    assert!(
        stdout.starts_with("{\"id\":\"input\",\"status\":\"ok\""),
        "{stdout}"
    );
}

/// REQ-39: 未許可の版は推論に進まず `invalid_input`（64）で拒否する。
#[test]
fn req39_unlisted_kind_version_is_rejected() {
    for (tag, version) in [("v0", 0), ("v2", 2)] {
        let (code, stdout) = run_with_version(tag, version);
        assert_eq!(code, Some(64), "version={version} stdout: {stdout}");
        assert_eq!(
            stdout,
            "{\"code\":\"invalid_input\",\"message\":\"unsupported kind_version\"}\n"
        );
    }
}
