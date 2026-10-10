//! 破損パッケージ（切り詰め・バイト反転）の拒否の E2E（バイナリを起動。REQ-39「異常系（版の検証）」・
//! TASK-39.6-2・#175・PoC-20 ケース 7。証拠種別: テストハーネス。コミット済み fixture からの合成パッケージ）。
//!
//! PoC-20 の破損レシピ（A: 先頭 `len/3` バイトだけ残す、B: オフセット 100..200 を `^= 0xFF`）で作った
//! C1・C3 のパッケージが、推論に進まず `invalid_input`（64）の JSON 1 行で拒否されることを完全一致で確認する。
//!
//! spec との差異: PoC-20 は ONNX Runtime の解析失敗（exit 70）だったが、本リポでは自作ランタイムの前段の
//! ガード層（形式検査）が先に拒否するため exit 64 になる。どちらも「非ゼロ終了＋機械可読 JSON の
//! `code` / `message`」（PoC-20 §4 の明確なエラー）を満たし、拒否する層がより手前にあるだけである。
//! ガードを通過した破損は 2 層目のランタイム（`malformed_protobuf`）が拒否することも lib レベルで確認する。
//!
//! 制約: 重み領域のバイト反転は protobuf 解析では検出できず、検出は sha256 照合が担う（本ファイルでは扱わない。
//! #168 参照）。fixture（`trainer/tools/gen_onnx_parity_fixture.py` で再生成）が変わり構造領域がずれた場合は、
//! 期待値を再導出する（許容を緩めない）。100..200 は構造領域の内側にある。
//! 形式検査・閉じ込めは Linux・macOS のみ対応のため、本ファイルのテストは両 OS に限定する。

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod unix_only {
    use fandhe_edge_cli::args::{InferArgs, InferSource};
    use fandhe_edge_cli::infer_guard::check_infer_path_and_format;
    use fandhe_edge_runtime::onnx::{ModelKind, OnnxBackend};
    use std::fs;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    const FORMAT_NOT_ALLOWED: &str =
        "{\"code\":\"invalid_input\",\"message\":\"format rejected: format_not_allowed\"}\n";
    const KINDS: [(&str, &str); 2] = [("c1", "c1.onnx"), ("c3", "c3.onnx")];

    fn fixture(name: &str) -> Vec<u8> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("fixtures")
            .join("onnx_parity")
            .join(name);
        fs::read(p).expect("fixture read")
    }

    /// PoC レシピ A: 先頭 `len/3` バイトだけ残す。
    fn truncate_third(b: &[u8]) -> Vec<u8> {
        b.iter().copied().take(b.len() / 3).collect()
    }

    /// 末尾 `n` バイトを切り落とす。
    fn truncate_tail(b: &[u8], n: usize) -> Vec<u8> {
        b.iter().copied().take(b.len().saturating_sub(n)).collect()
    }

    /// PoC レシピ B: オフセット 100..200 の各バイトを反転する。
    fn flip_100_200(b: &[u8]) -> Vec<u8> {
        let mut v = b.to_vec();
        for x in v.iter_mut().skip(100).take(100) {
            *x ^= 0xFF;
        }
        v
    }

    /// 末尾から `from_end` バイト目 1 バイトだけ反転する（ガードを通り抜ける破損）。
    fn flip_one_from_end(b: &[u8], from_end: usize) -> Vec<u8> {
        let mut v = b.to_vec();
        let idx = v.len().saturating_sub(from_end);
        if let Some(x) = v.get_mut(idx) {
            *x ^= 0xFF;
        }
        v
    }

    struct Sandbox {
        base: PathBuf,
    }

    impl Drop for Sandbox {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    impl Sandbox {
        fn new(label: &str, kind: &str, model: &[u8]) -> Self {
            let base = std::env::temp_dir().join(format!(
                "fandhe-cli-corrupt-{}-{}-{}",
                std::process::id(),
                label,
                kind
            ));
            let _ = fs::remove_dir_all(&base);
            fs::create_dir_all(base.join("pkg")).expect("mkdir");
            let meta = format!(r#"{{"onnx_file":"model.onnx","kind":"{kind}","kind_version":1}}"#);
            fs::write(base.join("pkg/artifact.json"), meta).expect("write");
            fs::write(base.join("pkg/model.onnx"), model).expect("write");
            Sandbox { base }
        }

        fn package_files(&self) -> Vec<String> {
            let mut names: Vec<String> = fs::read_dir(self.base.join("pkg"))
                .expect("readdir")
                .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        }

        fn args() -> InferArgs {
            InferArgs {
                package: "pkg".into(),
                source: InferSource::Text {
                    text: "a".into(),
                    id: None,
                },
                version: None,
            }
        }

        /// バイナリで `infer` を起動し、`invalid_input`（64）の固定 1 行で拒否されることを確認する。
        fn assert_infer_rejected(&self, ctx: &str) {
            let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
                .current_dir(&self.base)
                .args(["infer", "--package", "pkg", "--text", "a"])
                .stdin(Stdio::null())
                .output()
                .expect("run");
            let stdout = String::from_utf8(out.stdout).expect("utf8");
            let stderr = String::from_utf8(out.stderr).expect("utf8");
            assert_eq!(out.status.code(), Some(64), "{ctx}: stdout={stdout}");
            assert_eq!(stdout, FORMAT_NOT_ALLOWED, "{ctx}");
            assert_eq!(stdout.lines().count(), 1, "{ctx}");
            let base = self.base.to_string_lossy();
            for text in [&stdout, &stderr] {
                assert!(!text.contains(&*base), "{ctx}: path leaked");
            }
            assert_eq!(
                self.package_files(),
                ["artifact.json", "model.onnx"],
                "{ctx}: no side effects"
            );
        }
    }

    fn run_recipe(label: &str, recipe: impl Fn(&[u8]) -> Vec<u8>) {
        for (kind, file) in KINDS {
            let good = fixture(file);
            let bad = recipe(&good);
            assert_ne!(bad, good, "{kind}/{label}: recipe must change bytes");
            let sb = Sandbox::new(label, kind, &bad);
            sb.assert_infer_rejected(&format!("{kind}/{label}"));
        }
    }

    /// REQ-39・TASK-39.6-2・PoC-20 ケース 7（レシピ A）: 先頭 `len/3` に切り詰めたパッケージを拒否する。
    #[test]
    fn req39_truncated_package_third_is_rejected() {
        run_recipe("trunc3", truncate_third);
    }

    /// REQ-39・TASK-39.6-2・PoC-20 ケース 7（レシピ B）: オフセット 100..200 を反転したパッケージを拒否する。
    #[test]
    fn req39_byte_flipped_package_100_200_is_rejected() {
        run_recipe("flip", flip_100_200);
    }

    /// REQ-39・TASK-39.6-2: ほぼ全体を残した切り詰め（末尾 1 バイト欠け）も拒否する。
    #[test]
    fn req39_tail_truncated_package_is_rejected() {
        run_recipe("tail1", |b| truncate_tail(b, 1));
    }

    /// REQ-39: 陽性対照。無改変のパッケージはガードを通り、ランタイムに読み込める。
    #[test]
    fn req39_uncorrupted_package_passes_guard_and_loads() {
        for (kind, file) in KINDS {
            let good = fixture(file);
            let sb = Sandbox::new("good", kind, &good);
            let checked =
                check_infer_path_and_format(&sb.base, &Sandbox::args()).expect("guard must pass");
            assert_eq!(checked.onnx.as_bytes(), good.as_slice(), "{kind}");
            let k = ModelKind::parse(kind).expect("kind");
            assert!(
                OnnxBackend::from_bytes(checked.onnx.as_bytes(), k).is_ok(),
                "{kind}"
            );
        }
    }

    /// REQ-39: ガードを通り抜けた破損（`len-10` の 1 バイト反転）は 2 層目のランタイムが
    /// `malformed_protobuf` で拒否する。検査したバイト列と読み込むバイト列は同一（TOCTOU なし）。
    #[test]
    fn req39_corruption_past_guard_is_rejected_by_runtime() {
        for (kind, file) in KINDS {
            let bad = flip_one_from_end(&fixture(file), 10);
            let sb = Sandbox::new("pastguard", kind, &bad);
            let checked = check_infer_path_and_format(&sb.base, &Sandbox::args())
                .expect("guard passes this corruption");
            assert_eq!(checked.onnx.as_bytes(), bad.as_slice(), "{kind}");
            let k = ModelKind::parse(kind).expect("kind");
            let err = OnnxBackend::from_bytes(checked.onnx.as_bytes(), k)
                .expect_err("runtime must reject");
            assert_eq!(err.code(), "malformed_protobuf", "{kind}");
        }
    }
}
