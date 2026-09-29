//! モデルファイル（ONNX）を開く入口。拡張子と内容の照合で偽装を拒否する
//! （REQ-39「形式・種別の検査」・TASK-39.2-2・#154・PoC-20 ケース 2）。
//!
//! 役割は「拡張子の字句検査 → [`crate::format`] の内容検査への委譲」だけで、形式判定は再実装しない。
//! CLI の `infer`・`package`・`register` 等への統合は TASK-39.2-4（#156）、経路の閉じ込めとの
//! 連携は TASK-39.4-2（#159）で行う。検査順序は 経路 → 拡張子 → サイズ → 形式。
//!
//! - 拡張子が `.onnx` でなければ、ファイルを開かず読まずに拒否する（不適格な未信頼ファイルのバイトに触れない）
//! - pickle・zip・npy 等の中身は解釈・展開・実行しない。内容が ONNX でなければ `NotAllowed` で拒否する
//! - 拡張子の文字列は利用者由来のため、エラーの値にもメッセージにも含めない
//!
//! PoC の `endswith(".onnx")` より意図的に厳しい: 大文字小文字を区別し（`model.ONNX` は拒否）、
//! `Path::extension()` を使うためドットファイル `.onnx`・`a.onnx.pt`・拡張子なしも拒否する。

use crate::format::{CheckedFile, FormatAllowlist, FormatRejection, open_checked_file};
use std::path::Path;

/// モデルファイルとして許可する拡張子（REQ-39・TASK-39.2-2）。
pub const MODEL_FILE_EXTENSION: &str = "onnx";

/// 拡張子が `.onnx` で、内容が ONNX 形のファイルだけを開く（pickle 偽装・非 ONNX の拒否）。
///
/// 拡張子不適格は `ExtensionNotAllowed`（終了コード 64・`extension_not_allowed`）で、ファイルには触れない。
/// 内容が ONNX でなければ `NotAllowed`、上限超過等は [`open_checked_file`] と同じ。
pub fn open_onnx_model_file(path: &Path, max_bytes: u64) -> Result<CheckedFile, FormatRejection> {
    if path.extension().is_none_or(|e| e != MODEL_FILE_EXTENSION) {
        return Err(FormatRejection::ExtensionNotAllowed {
            expected: MODEL_FILE_EXTENSION,
        });
    }
    open_checked_file(path, &FormatAllowlist::onnx_only(), max_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ext_rejected(p: &str) -> bool {
        // 存在しないパスで呼ぶ: 拡張子で拒否されれば Io にならない（読まずに拒否）。
        matches!(
            open_onnx_model_file(Path::new(p), 1024),
            Err(FormatRejection::ExtensionNotAllowed { .. })
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
}
