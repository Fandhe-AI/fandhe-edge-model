//! `infer` のガード層統合の E2E（バイナリを起動。REQ-39・REQ-33・REQ-21・TASK-39.2-4・#156・
//! PoC-20 ケース 2。証拠種別: テストハーネス。合成パッケージ）。
//!
//! pickle 偽装・非 ONNX ファイル・非対応の `kind` が、`invalid_input`（64）の JSON 1 行で拒否され、
//! 出力へパス・`kind` の値・ファイル内容を含めないことを文字列の完全一致で確認する。
//! 正常対照として有効なパッケージはガードを通過し、スタブ（`runtime_error`・70。#136 で置換予定）へ到達する。
//! 内容検査は Linux・macOS のみ対応のため、本ファイルのテストは両 OS に限定する。

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod unix_only {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    const MIN_ONNX: [u8; 9] = [0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78];
    const CANARY: &str = "CANARY_SECRET";
    const STUB: &str =
        "{\"code\":\"runtime_error\",\"message\":\"stage not implemented yet (TASK-33.1-2)\"}\n";
    const FORMAT_NOT_ALLOWED: &str =
        "{\"code\":\"invalid_input\",\"message\":\"format rejected: format_not_allowed\"}\n";
    const EXTENSION_NOT_ALLOWED: &str =
        "{\"code\":\"invalid_input\",\"message\":\"format rejected: extension_not_allowed\"}\n";
    const UNSUPPORTED_KIND: &str =
        "{\"code\":\"invalid_input\",\"message\":\"kind rejected: unsupported_kind\"}\n";
    const MALFORMED_KIND: &str =
        "{\"code\":\"invalid_input\",\"message\":\"kind rejected: malformed_kind\"}\n";
    const UNSUPPORTED_KIND_VERSION: &str = "{\"code\":\"invalid_input\",\"message\":\"kind_version rejected: unsupported_kind_version\"}\n";
    const META_INVALID: &str =
        "{\"code\":\"invalid_input\",\"message\":\"artifact metadata is invalid\"}\n";

    struct Sandbox {
        base: PathBuf,
    }

    impl Drop for Sandbox {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    impl Sandbox {
        fn new(label: &str) -> Self {
            let base = std::env::temp_dir().join(format!(
                "fandhe-cli-guardkind-{}-{}",
                std::process::id(),
                label
            ));
            let _ = fs::remove_dir_all(&base);
            fs::create_dir_all(base.join("pkg")).expect("mkdir");
            Sandbox { base }
        }

        /// `pkg/` に `artifact.json` と `model_name` の中身 `model` を置く。
        fn setup(&self, artifact_json: &str, model_name: &str, model: &[u8]) {
            fs::write(self.base.join("pkg/artifact.json"), artifact_json).expect("write");
            fs::write(self.base.join("pkg").join(model_name), model).expect("write");
        }

        fn marker(&self) -> PathBuf {
            self.base.join("marker_created")
        }

        /// `fandhe-edge infer --package pkg --text a` を起動し、終了コードと stdout を検証する。
        fn assert_infer(&self, expected_code: i32, expected_stdout: &str) {
            let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
                .current_dir(&self.base)
                .args(["infer", "--package", "pkg", "--text", "a"])
                .stdin(Stdio::null())
                .output()
                .expect("run");
            let stdout = String::from_utf8(out.stdout).expect("utf8");
            let stderr = String::from_utf8(out.stderr).expect("utf8");
            assert_eq!(out.status.code(), Some(expected_code), "stdout={stdout}");
            assert_eq!(stdout, expected_stdout);
            assert_eq!(stdout.lines().count(), 1);
            let base = self.base.to_string_lossy();
            for text in [&stdout, &stderr] {
                assert!(!text.contains(&*base));
                assert!(!text.contains(CANARY));
            }
            assert!(!self.marker().exists(), "marker must not be created");
        }
    }

    fn meta(kind_json: &str, onnx: &str) -> String {
        meta_v(kind_json, "1", onnx)
    }

    fn meta_v(kind_json: &str, version_json: &str, onnx: &str) -> String {
        format!(r#"{{"onnx_file":"{onnx}","kind":{kind_json},"kind_version":{version_json}}}"#)
    }

    /// 「unpickle されたらマーカーに空ファイルを作る」だけの無害な pickle 形。
    fn marker_pickle(proto: u8, marker: &Path) -> Vec<u8> {
        let mut b = vec![0x80, proto];
        b.extend_from_slice(b"cbuiltins\nopen\n(S'");
        b.extend_from_slice(marker.to_string_lossy().as_bytes());
        b.extend_from_slice(b"'\nS'w'\ntR.");
        b.extend_from_slice(CANARY.as_bytes());
        b
    }

    fn marker_pickle_text(marker: &Path) -> Vec<u8> {
        format!(
            "cbuiltins\nopen\n(S'{}'\nS'w'\ntR.{CANARY}",
            marker.display()
        )
        .into_bytes()
    }

    /// REQ-39・PoC-20 ケース 2: `.onnx` として置いた pickle を拒否し、マーカーを作らない。
    #[test]
    fn req39_pickle_disguised_as_onnx_is_rejected() {
        let sb = Sandbox::new("pickle-onnx");
        let m = sb.marker();
        let payloads = [
            marker_pickle(2, &m),
            marker_pickle(4, &m),
            marker_pickle(5, &m),
            marker_pickle_text(&m),
        ];
        for p in payloads {
            sb.setup(&meta("\"c3\"", "model.onnx"), "model.onnx", &p);
            sb.assert_infer(64, FORMAT_NOT_ALLOWED);
        }
    }

    /// REQ-39: `.pt`・`.npy` として置いた pickle は拡張子で拒否する。
    #[test]
    fn req39_pickle_with_pickle_extensions_is_rejected() {
        let sb = Sandbox::new("pickle-ext");
        let p = marker_pickle(4, &sb.marker());
        for name in ["model.pt", "model.npy"] {
            sb.setup(&meta("\"c3\"", name), name, &p);
            sb.assert_infer(64, EXTENSION_NOT_ALLOWED);
        }
    }

    /// REQ-39: zip・npy・GGUF・不明バイト列は ONNX ではないため拒否する。
    #[test]
    fn req39_non_onnx_files_are_rejected() {
        let sb = Sandbox::new("non-onnx");
        let contents: [&[u8]; 4] = [
            b"PK\x03\x04rest-of-zip",
            b"\x93NUMPY\x01\x00rest",
            b"GGUFrest",
            b"not an onnx model",
        ];
        for c in contents {
            sb.setup(&meta("\"c3\"", "model.onnx"), "model.onnx", c);
            sb.assert_infer(64, FORMAT_NOT_ALLOWED);
        }
    }

    /// REQ-39: 構文は正しいが許可リストにない `kind` は `unsupported_kind`。
    #[test]
    fn req39_unsupported_kind_is_rejected() {
        let sb = Sandbox::new("kind-unsupported");
        for k in ["\"pt\"", "\"pickle\""] {
            sb.setup(&meta(k, "model.onnx"), "model.onnx", &MIN_ONNX);
            sb.assert_infer(64, UNSUPPORTED_KIND);
        }
    }

    /// REQ-39・PoC-20: インジェクション形・空・大文字・過長の `kind` は `malformed_kind`。
    #[test]
    fn req39_malformed_kind_is_rejected() {
        let sb = Sandbox::new("kind-malformed");
        let long = format!("\"{}\"", "a".repeat(65));
        for k in ["\"c3; rm -rf ~\"", "\"\"", "\"C3\"", long.as_str()] {
            sb.setup(&meta(k, "model.onnx"), "model.onnx", &MIN_ONNX);
            sb.assert_infer(64, MALFORMED_KIND);
        }
    }

    /// REQ-39: `kind` の欠落・型違いは必須違反として `artifact metadata is invalid`。
    #[test]
    fn req39_missing_or_non_string_kind_is_rejected() {
        let sb = Sandbox::new("kind-missing");
        let metas = [
            r#"{"onnx_file":"model.onnx","kind_version":1}"#.to_string(),
            meta("1", "model.onnx"),
            meta("null", "model.onnx"),
            meta("[]", "model.onnx"),
        ];
        for m in metas {
            sb.setup(&m, "model.onnx", &MIN_ONNX);
            sb.assert_infer(64, META_INVALID);
        }
    }

    /// REQ-39: `kind` の検査はモデルのバイト列を開くより先に効く（モデルが pickle でも kind の拒否が返る）。
    #[test]
    fn req39_kind_is_checked_before_model_bytes() {
        let sb = Sandbox::new("kind-order");
        let p = marker_pickle(4, &sb.marker());
        sb.setup(&meta("\"pt\"", "model.onnx"), "model.onnx", &p);
        sb.assert_infer(64, UNSUPPORTED_KIND);
    }

    /// 正常対照: 許可された `kind` と ONNX はガードを通過し、スタブ（70）へ到達する。
    #[test]
    fn req39_allowed_kinds_reach_stub() {
        let sb = Sandbox::new("kind-ok");
        for k in ["\"c1\"", "\"c3\"", "\"autoregressive\""] {
            sb.setup(&meta(k, "model.onnx"), "model.onnx", &MIN_ONNX);
            sb.assert_infer(70, STUB);
        }
    }

    /// REQ-39・PoC-20 ケース 7（テストハーネス）: 許可リスト外の `kind_version` は推論前に拒否され、
    /// 出力に版番号・パス・CANARY を含めない。
    #[test]
    fn req39_kind_version_99_is_rejected_before_inference() {
        let sb = Sandbox::new("kv-99");
        for (k, v) in [
            ("\"c3\"", "99"),
            ("\"c1\"", "99"),
            ("\"autoregressive\"", "99"),
            ("\"c3\"", "0"),
            ("\"c3\"", "2"),
            ("\"c3\"", "4294967295"),
        ] {
            sb.setup(&meta_v(k, v, "model.onnx"), "model.onnx", &MIN_ONNX);
            sb.assert_infer(64, UNSUPPORTED_KIND_VERSION);
        }
    }

    /// REQ-39: 版の検査はモデルのバイト列を開くより先に効く（モデルが pickle でも版の拒否が返る）。
    #[test]
    fn req39_kind_version_is_checked_before_model_bytes() {
        let sb = Sandbox::new("kv-order");
        let p = marker_pickle(4, &sb.marker());
        sb.setup(&meta_v("\"c3\"", "99", "model.onnx"), "model.onnx", &p);
        sb.assert_infer(64, UNSUPPORTED_KIND_VERSION);
    }

    /// REQ-39: `kind` の検査が `kind_version` より先に効く。
    #[test]
    fn req39_kind_is_checked_before_kind_version() {
        let sb = Sandbox::new("kv-kind-first");
        sb.setup(
            &meta_v("\"pt\"", "99", "model.onnx"),
            "model.onnx",
            &MIN_ONNX,
        );
        sb.assert_infer(64, UNSUPPORTED_KIND);
    }

    /// REQ-39: `kind_version` の型違い・範囲外は `artifact metadata is invalid`（欠落は後方互換で許可）。
    #[test]
    fn req39_missing_or_non_integer_kind_version_is_rejected() {
        let sb = Sandbox::new("kv-invalid");
        let mut metas = Vec::new();
        for v in ["\"1\"", "-1", "1.5", "null", "4294967296"] {
            metas.push(meta_v("\"c3\"", v, "model.onnx"));
        }
        for m in metas {
            sb.setup(&m, "model.onnx", &MIN_ONNX);
            sb.assert_infer(64, META_INVALID);
        }
    }
}
