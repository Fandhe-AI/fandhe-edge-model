//! モデルファイル（ONNX）を開く入口。拡張子と内容の照合で偽装を拒否する
//! （REQ-39「形式・種別の検査」・TASK-39.2-2・#154・PoC-20 ケース 2）。
//!
//! 役割は「拡張子の字句検査 → [`crate::path::open_confined`] による閉じ込め付き open →
//! [`crate::format`] の内容検査への委譲」だけで、形式判定は再実装しない。
//! CLI の `infer`・`package`・`register` 等への統合は TASK-39.2-4（#156）、CLI 引数への
//! 経路検証の組み込みは TASK-39.4-2（#159）で行う。検査順序は 拡張子 → 経路 → サイズ → 形式。
//!
//! 閉じ込め検証と内容検査の間でパスを開き直さない（検証後の差し替えでルート外を読ませない。
//! REQ-39・PoC-20）。[`open_onnx_model_file`] は `open_confined` が返したハンドルをそのまま
//! 内容検査へ渡す。
//!
//! - 拡張子が `.onnx` でなければ、ファイルを開かず読まずに拒否する（不適格な未信頼ファイルのバイトに触れない）
//! - pickle・zip・npy 等の中身は解釈・展開・実行しない。内容が ONNX でなければ `NotAllowed` で拒否する
//! - 拡張子の文字列は利用者由来のため、エラーの値にもメッセージにも含めない
//!
//! PoC の `endswith(".onnx")` より意図的に厳しい: 大文字小文字を区別し（`model.ONNX` は拒否）、
//! `Path::extension()` を使うためドットファイル `.onnx`・`a.onnx.pt`・拡張子なしも拒否する。

use crate::format::{CheckedFile, FormatAllowlist, FormatRejection, check_open_file};
use crate::path::{PathRejection, open_confined};
use fandhe_edge_core::exitcode::ExitCode;
use std::fmt;
use std::path::Path;

/// モデルファイルとして許可する拡張子（REQ-39・TASK-39.2-2）。
pub const MODEL_FILE_EXTENSION: &str = "onnx";

/// モデルファイルを開く際の拒否理由。経路の閉じ込め違反と形式・拡張子の拒否をまとめる。
#[derive(Debug)]
#[non_exhaustive]
pub enum ModelFileRejection {
    /// 経路の閉じ込め・open の拒否（[`PathRejection`]）。
    Path(PathRejection),
    /// 拡張子・形式・サイズの拒否（[`FormatRejection`]）。
    Format(FormatRejection),
}

impl ModelFileRejection {
    /// 終了コード（REQ-21）。内包する拒否理由のものをそのまま返す。
    pub fn exit_code(&self) -> ExitCode {
        match self {
            ModelFileRejection::Path(e) => e.exit_code(),
            ModelFileRejection::Format(e) => e.exit_code(),
        }
    }

    /// 機械可読な理由コード（英語の snake_case）。
    pub const fn reason_code(&self) -> &'static str {
        match self {
            ModelFileRejection::Path(e) => e.reason_code(),
            ModelFileRejection::Format(e) => e.reason_code(),
        }
    }
}

impl fmt::Display for ModelFileRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelFileRejection::Path(e) => e.fmt(f),
            ModelFileRejection::Format(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for ModelFileRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelFileRejection::Path(e) => Some(e),
            ModelFileRejection::Format(e) => Some(e),
        }
    }
}

fn check_extension(path: &Path) -> Result<(), FormatRejection> {
    if path.extension().is_none_or(|e| e != MODEL_FILE_EXTENSION) {
        return Err(FormatRejection::ExtensionNotAllowed {
            expected: MODEL_FILE_EXTENSION,
        });
    }
    Ok(())
}

/// `root` 配下に閉じ込めて開き、拡張子が `.onnx` かつ内容が ONNX 形のファイルだけを返す
/// （pickle 偽装・非 ONNX の拒否）。
///
/// 拡張子不適格は `ExtensionNotAllowed`（終了コード 64・`extension_not_allowed`）で、ファイルには触れない。
/// 続けて [`open_confined`] で開き、返されたハンドルを開き直さずに内容検査へ渡す（検証後の差し替え対策）。
/// 内容が ONNX でなければ `NotAllowed`、上限超過等は [`check_open_file`] と同じ。
/// 解決後の実体パスの拡張子も再確認する。
pub fn open_onnx_model_file(
    root: &Path,
    candidate: &Path,
    max_bytes: u64,
) -> Result<CheckedFile, ModelFileRejection> {
    check_extension(candidate).map_err(ModelFileRejection::Format)?;
    let (file, confined) = open_confined(root, candidate).map_err(ModelFileRejection::Path)?;
    check_extension(confined.as_path()).map_err(ModelFileRejection::Format)?;
    check_open_file(
        file,
        confined.as_path(),
        &FormatAllowlist::onnx_only(),
        max_bytes,
    )
    .map_err(ModelFileRejection::Format)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONNX: [u8; 9] = [0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78];

    fn ext_rejected(p: &str) -> bool {
        // 存在しないルートで呼ぶ: 拡張子で拒否されれば Path 拒否にならない（開かずに拒否）。
        matches!(
            open_onnx_model_file(Path::new("/nonexistent-root"), Path::new(p), 1024),
            Err(ModelFileRejection::Format(
                FormatRejection::ExtensionNotAllowed { .. }
            ))
        )
    }

    /// REQ-39・TASK-39.2-2: 拡張子判定の境界。
    #[test]
    fn req39_extension_boundaries() {
        for p in [
            "m.pt",
            "m.npy",
            "m.ONNX",
            "m.Onnx",
            ".onnx",
            "m",
            "m.onnx.pt",
            "m.onnx.",
        ] {
            assert!(ext_rejected(p), "{p}");
        }
        for p in ["m.onnx", "dir/a.pt.onnx"] {
            assert!(!ext_rejected(p), "{p}");
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("fe-guard-mf-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// REQ-39・TASK-39.2-2: ルート配下の ONNX は通り、pickle 偽装は拒否される。
    #[test]
    fn req39_open_confined_onnx_and_disguise() {
        let root = tmp("ok");
        std::fs::write(root.join("m.onnx"), ONNX).unwrap();
        std::fs::write(root.join("fake.onnx"), [0x80, 0x04, 0x95]).unwrap();
        let ok = open_onnx_model_file(&root, Path::new("m.onnx"), 1024).unwrap();
        assert_eq!(ok.as_bytes(), &ONNX);
        let err = open_onnx_model_file(&root, Path::new("fake.onnx"), 1024).unwrap_err();
        assert_eq!(err.reason_code(), "format_not_allowed");
        assert_eq!(err.exit_code(), ExitCode::InvalidInput);
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// REQ-39・PoC-20: ルート外への参照（`..`・symlink）は内容検査の前に拒否される。
    #[test]
    fn req39_confinement_is_enforced() {
        let base = tmp("esc");
        let root = base.join("root");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(base.join("outside.onnx"), ONNX).unwrap();
        let err = open_onnx_model_file(&root, Path::new("../outside.onnx"), 1024).unwrap_err();
        assert!(matches!(err, ModelFileRejection::Path(_)), "{err}");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(base.join("outside.onnx"), root.join("link.onnx")).unwrap();
            let err = open_onnx_model_file(&root, Path::new("link.onnx"), 1024).unwrap_err();
            assert!(matches!(err, ModelFileRejection::Path(_)), "{err}");
        }
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// REQ-39: 上限超過は `LimitExceeded`。
    #[test]
    fn req39_size_limit() {
        let root = tmp("big");
        std::fs::write(root.join("m.onnx"), ONNX).unwrap();
        let err = open_onnx_model_file(&root, Path::new("m.onnx"), 4).unwrap_err();
        assert_eq!(err.exit_code(), ExitCode::LimitExceeded);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
