//! `infer` の入力表現の宣言（REQ-41・TASK-41.3-1・#402）。
//!
//! 共有 fixture の C1 の ONNX でパッケージを組み、宣言なしと `byte` は同じ結果で推論に進み、
//! `tokenizer`（推論未対応。TASK-41.4）と不正な値は `invalid_input`（64）で拒否されることを確かめる
//! （証拠種別: テストハーネス）。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::path::Path;
use std::process::{Command, Stdio};

const DEFINITION: &str = r#"{"schema":"fandhe-edge-model-definition/v1","name":"ir","version":1,"judgment_type":"single_select","options":[{"id":"alpha","display_name":"a","description":"d"},{"id":"beta","display_name":"b","description":"d"},{"id":"gamma","display_name":"g","description":"d"}],"io":{"input":"bytes"}}"#;

fn run(tag: &str, extra: &str) -> (Option<i32>, String) {
    let ws = std::env::temp_dir().join(format!("fandhe-input-repr-{}-{tag}", std::process::id()));
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
            r#"{{"kind":"c1","kind_version":1,"max_bytes":48,"label_order":["alpha","beta","gamma"],"onnx_file":"model.onnx","onnx_sha256":"{sha}"{extra}}}"#
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

/// 宣言なし（既存パッケージ）と `byte` は同じ出力で推論に進む。
#[test]
fn req41_undeclared_equals_byte() {
    let (c0, undeclared) = run("none", "");
    let (c1, byte) = run("byte", r#","input_representation":"byte""#);
    assert_eq!((c0, c1), (Some(0), Some(0)), "{undeclared} / {byte}");
    assert!(undeclared.starts_with("{\"id\":\"input\",\"status\":\"ok\""));
    assert_eq!(undeclared, byte);
}

/// `tokenizer` は推論未対応のため 64 で拒否する。
#[test]
fn req41_tokenizer_is_rejected_until_runtime_supports_it() {
    let (code, stdout) = run("tok", r#","input_representation":"tokenizer""#);
    assert_eq!(code, Some(64), "{stdout}");
    assert_eq!(
        stdout,
        "{\"code\":\"invalid_input\",\"message\":\"unsupported input representation\"}\n"
    );
}

/// 未知の値・`null` は artifact.json 不正として 64 で拒否する。
#[test]
fn req41_unknown_or_null_is_rejected() {
    for (tag, extra) in [
        ("bytes", r#","input_representation":"bytes""#),
        ("null", r#","input_representation":null"#),
    ] {
        let (code, stdout) = run(tag, extra);
        assert_eq!(code, Some(64), "{tag}: {stdout}");
        assert_eq!(
            stdout, "{\"code\":\"invalid_input\",\"message\":\"artifact metadata is invalid\"}\n",
            "{tag}"
        );
    }
}
