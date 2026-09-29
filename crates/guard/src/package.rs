//! パッケージ単位の経路の閉じ込め（REQ-39・PoC-20 ケース 1・TASK-39.4-2・#159）。
//!
//! CLI の `infer --package <dir>` と、パッケージ内メタデータ `artifact.json` の `onnx_file`
//! を、二段の境界で閉じ込める。CLI・MCP・TUI が同じ検査を通れるよう、JSON の解釈を持たない
//! ガード層に置く（REQ-33・REQ-36・REQ-37）。
//!
//! # 二段の境界
//!
//! 1. パッケージディレクトリは workspace（呼び出し側が渡すルート。CLI ではカレント
//!    ディレクトリ）配下でなければならない（[`confine_package`]）
//! 2. パッケージ内のメンバー（`artifact.json`・`onnx_file`）はパッケージ配下でなければ
//!    ならず、加えて開いた fd の実パスが workspace 配下であることを再確認する
//!    （[`ConfinedPackage::open_member`]）。[`crate::path::open_confined`] はルートを
//!    パスから開き直すため、`confine_package` の後にパッケージディレクトリ（またはその
//!    途中の成分）が外を指す symlink へ差し替えられる競合を、ガード単体では防げない。
//!    workspace の再確認でこの窓を塞ぐ
//!
//! # 呼び出し元の義務
//!
//! 返した [`File`] だけを読み、パスから開き直さない（開き直すと TOCTOU が残る）。
//! 検査の順序は「経路 → サイズ → 形式」（[`crate::path`] のモジュール文書）。
//!
//! # 判定（fail-closed）
//!
//! Linux・macOS 以外では [`PathRejection::UnsupportedPlatform`]（`open_member` 側）で拒否する。

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::path::{ConfinedPath, EscapeKind, PathRejection, open_confined, safe_join};

/// workspace 配下へ閉じ込め済みのパッケージディレクトリ。[`confine_package`] だけが生成する。
#[derive(Debug, Clone)]
pub struct ConfinedPackage {
    /// 正準化した workspace（メンバーの実パスの包含確認に使う）。
    workspace: PathBuf,
    /// 正準化したパッケージディレクトリ。
    dir: ConfinedPath,
}

/// `package`（利用者由来の `--package`）を `workspace` 配下へ閉じ込め、ディレクトリであることを確認する。
///
/// # Errors
/// [`safe_join`] の拒否（`../`・ルート外の絶対パス・symlink・存在しない対象）と、
/// 対象がディレクトリでない場合の [`PathRejection::NotDirectory`]。
pub fn confine_package(workspace: &Path, package: &Path) -> Result<ConfinedPackage, PathRejection> {
    let canon_workspace = std::fs::canonicalize(workspace)
        .map_err(|source| PathRejection::RootUnresolvable { source })?;
    let dir = safe_join(&canon_workspace, package)?;
    if !dir.as_path().is_dir() {
        return Err(PathRejection::NotDirectory {
            candidate: package.to_path_buf(),
        });
    }
    Ok(ConfinedPackage {
        workspace: canon_workspace,
        dir,
    })
}

impl ConfinedPackage {
    /// 正準化したパッケージディレクトリ（表示・診断用。ファイルを開き直すために使わない）。
    pub fn dir(&self) -> &Path {
        self.dir.as_path()
    }

    /// パッケージ内のメンバーを、パッケージ配下かつ workspace 配下であることを確認して開く。
    ///
    /// # Errors
    /// [`open_confined`] の拒否に加え、開いた実体が workspace の外なら
    /// [`PathRejection::Escapes`]（[`EscapeKind::Symlink`]）。
    pub fn open_member(&self, member: &Path) -> Result<(File, ConfinedPath), PathRejection> {
        let (file, real) = open_confined(self.dir.as_path(), member)?;
        // 成分単位の包含判定（文字列の前方一致は使わない）。
        if !real.as_path().starts_with(&self.workspace) {
            return Err(PathRejection::Escapes {
                candidate: member.to_path_buf(),
                kind: EscapeKind::Symlink,
            });
        }
        Ok((file, real))
    }
}
