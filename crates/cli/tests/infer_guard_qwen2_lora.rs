//! 追加学習の種類 `qwen2_lora` の完全な形のパッケージで、ガード層（経路・形式・kind・kind_version・
//! sha256）が `infer` の経路に結合して効くこと（REQ-39・REQ-41・TASK-41.8・#435。
//! 証拠種別: テストハーネス・合成パッケージ）。
//!
//! 推論本体（全件一致）は `infer_qwen2_parity.rs`（#626）の範囲で、本ファイルの正常形はガードを通過して
//! 次の層（`tokenizer.json` の記録ハッシュ照合。合成の最小 ONNX・ダミーのハッシュのため 64）に到達する
//! ことだけを固定する。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use fandhe_edge_core::hash::Sha256Digest;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MIN_ONNX: [u8; 9] = [0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78];
const CANARY: &str = "CANARY_SECRET";
const DEFINITION: &str = r#"{"schema":"fandhe-edge-model-definition/v1","name":"q","version":1,"judgment_type":"single_select","options":[{"id":"positive","display_name":"p","description":"d"},{"id":"negative","display_name":"n","description":"d"}],"io":{"input":"bytes"}}"#;
/// `tokenizer_sha256` はダミー（0 埋め）なので、`tokenizer.json` の記録ハッシュ照合は必ず不一致になる。
const REACHED_NEXT_LAYER: &str = "{\"code\":\"invalid_input\",\"message\":\"package tokenizer does not match its recorded hash\"}\n";

fn err(code: &str, message: &str) -> String {
    format!("{{\"code\":\"{code}\",\"message\":\"{message}\"}}\n")
}

struct Pkg {
    base: PathBuf,
}

impl Drop for Pkg {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

impl Pkg {
    /// 共有 fixture `result_ok_qwen2_lora.json` の `artifact` と同じ欄の完全形で組む。
    fn new(
        label: &str,
        kind: &str,
        version: &str,
        onnx_file: &str,
        onnx: &[u8],
        sha: &str,
    ) -> Self {
        let base =
            std::env::temp_dir().join(format!("fandhe-cli-q2l-{}-{label}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("p")).expect("mkdir");
        let meta = format!(
            r#"{{"kind":{kind},"kind_version":{version},"label_order":["positive","negative"],"max_bytes":512,"onnx_file":"{onnx_file}","onnx_sha256":"{sha}","input_representation":"tokenizer","tokenizer_file":"tokenizer.json","tokenizer_sha256":"{z}","external_data":[{{"location":"weights-00000.bin","size_bytes":4,"sha256":"{z}"}}],"scoring":{{"system_prompt":"x","choices":[{{"id":"positive","token_ids":[1]}},{{"id":"negative","token_ids":[2]}}]}}}}"#,
            z = "0".repeat(64)
        );
        fs::write(base.join("p/artifact.json"), meta).expect("meta");
        fs::write(base.join("p/definition.json"), DEFINITION).expect("def");
        fs::write(base.join("p").join(onnx_file), onnx).expect("onnx");
        fs::write(base.join("p/tokenizer.json"), r#"{"model":{"type":"BPE"}}"#).expect("tok");
        fs::write(base.join("p/weights-00000.bin"), b"abcd").expect("shard");
        Pkg { base }
    }

    fn ok(label: &str) -> Self {
        let sha = Sha256Digest::of_bytes(&MIN_ONNX).to_hex();
        Self::new(label, "\"qwen2_lora\"", "1", "model.onnx", &MIN_ONNX, &sha)
    }

    fn marker(&self) -> PathBuf {
        self.base.join("marker_created")
    }

    fn run(&self) -> (Option<i32>, String, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
            .current_dir(&self.base)
            .args(["infer", "--package", "p", "--text", "a"])
            .stdin(Stdio::null())
            .output()
            .expect("run");
        (
            out.status.code(),
            String::from_utf8(out.stdout).expect("utf8"),
            String::from_utf8(out.stderr).expect("utf8"),
        )
    }

    fn assert_infer(&self, code: i32, stdout: &str) {
        let (c, so, se) = self.run();
        assert_eq!(c, Some(code), "stdout={so}");
        assert_eq!(so, stdout);
        assert_eq!(so.lines().count(), 1);
        let base = self.base.to_string_lossy();
        for t in [&so, &se] {
            assert!(!t.contains(&*base) && !t.contains(CANARY));
        }
        assert!(!self.marker().exists(), "marker must not be created");
    }
}

fn marker_pickle(marker: &Path) -> Vec<u8> {
    let mut b = vec![0x80, 4];
    b.extend_from_slice(b"cbuiltins\nopen\n(S'");
    b.extend_from_slice(marker.to_string_lossy().as_bytes());
    b.extend_from_slice(b"'\nS'w'\ntR.");
    b.extend_from_slice(CANARY.as_bytes());
    b
}

/// 正常形はガード（経路・形式・kind・版・sha256）を通過して次の層に到達する。
#[test]
fn req39_qwen2_lora_valid_package_passes_guard() {
    Pkg::ok("ok").assert_infer(64, REACHED_NEXT_LAYER);
}

/// `.onnx` として置いた pickle は形式の許可制で拒否され、マーカーを作らない。
#[test]
fn req39_qwen2_lora_pickle_disguised_as_onnx_is_rejected() {
    let p = Pkg::ok("pickle");
    let bytes = marker_pickle(&p.marker());
    fs::write(p.base.join("p/model.onnx"), bytes).expect("write");
    p.assert_infer(
        64,
        &err("invalid_input", "format rejected: format_not_allowed"),
    );
}

/// ONNX 枠の拡張子は `.onnx` のみ（pickle 系・safetensors は本種類でも拒否）。
#[test]
fn req39_qwen2_lora_non_onnx_extensions_are_rejected() {
    for name in ["model.pt", "model.bin", "model.safetensors"] {
        let sha = Sha256Digest::of_bytes(&MIN_ONNX).to_hex();
        let p = Pkg::new("ext", "\"qwen2_lora\"", "1", name, &MIN_ONNX, &sha);
        p.assert_infer(
            64,
            &err("invalid_input", "format rejected: extension_not_allowed"),
        );
    }
}

/// 版の検査はモデルのバイト列より先に効く（モデルが pickle でも版の拒否が返る）。
#[test]
fn req39_qwen2_lora_kind_version_is_checked_before_model_bytes() {
    for v in ["0", "2", "99"] {
        let p = Pkg::new("ver", "\"qwen2_lora\"", v, "model.onnx", b"x", "0");
        let bytes = marker_pickle(&p.marker());
        fs::write(p.base.join("p/model.onnx"), bytes).expect("write");
        p.assert_infer(
            64,
            &err(
                "invalid_input",
                "kind_version rejected: unsupported_kind_version",
            ),
        );
    }
}

/// 種別の綴り違い・大文字・余白は拒否される。
#[test]
fn req39_qwen2_lora_kind_spelling_variants_are_rejected() {
    for (k, reason) in [
        ("qwen2-lora", "malformed_kind"),
        ("Qwen2_LoRA", "malformed_kind"),
        ("qwen2_lora ", "malformed_kind"),
        ("qwen2_lora2", "unsupported_kind"),
    ] {
        let sha = Sha256Digest::of_bytes(&MIN_ONNX).to_hex();
        let p = Pkg::new(
            "kind",
            &format!("\"{k}\""),
            "1",
            "model.onnx",
            &MIN_ONNX,
            &sha,
        );
        p.assert_infer(
            64,
            &err("invalid_input", &format!("kind rejected: {reason}")),
        );
    }
}

/// 読み込み前のサイズ上限（44 MiB）が本種類でも効く（疎ファイル）。
#[test]
fn req39_qwen2_lora_oversized_model_is_limit_exceeded() {
    let p = Pkg::ok("big");
    let f = fs::File::create(p.base.join("p/model.onnx")).expect("create");
    f.set_len(fandhe_edge_runtime::onnx::MAX_MODEL_FILE_BYTES + 1)
        .expect("set_len");
    drop(f);
    let (c, so, _) = p.run();
    assert_eq!(c, Some(20), "{so}");
    assert!(so.starts_with("{\"code\":\"limit_exceeded\""), "{so}");
    assert_eq!(so.lines().count(), 1);
}

/// 記録された sha256 と ONNX の不一致は拒否される（完全性）。
#[test]
fn req39_qwen2_lora_sha256_mismatch_is_rejected() {
    let p = Pkg::new(
        "sha",
        "\"qwen2_lora\"",
        "1",
        "model.onnx",
        &MIN_ONNX,
        &"1".repeat(64),
    );
    p.assert_infer(
        64,
        &err(
            "invalid_input",
            "package model does not match its recorded hash",
        ),
    );
}
