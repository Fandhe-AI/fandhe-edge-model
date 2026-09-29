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
//!    （[`ConfinedPackage::open_member`]）。`confine_package` は検証したパッケージの
//!    ディレクトリ fd を `O_NOFOLLOW` で開いて保持し（実パスが検証結果と一致しなければ拒否）、
//!    メンバーはその fd を起点に開く。検証後にパッケージのパスが別ディレクトリ（symlink 等）へ
//!    差し替えられても、検証済みのパッケージ以外は読めない
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
use std::path::Path;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::path::PathBuf;

#[cfg(any(target_os = "linux", target_os = "macos"))]
use crate::path::EscapeKind;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use crate::path::{ConfinedDir, open_dir_confined};
use crate::path::{ConfinedPath, PathRejection, safe_join};

/// workspace 配下へ閉じ込め済みのパッケージディレクトリ。[`confine_package`] だけが生成する。
///
/// Linux・macOS では検証時に開いたディレクトリ fd を保持し、メンバーはこの fd を起点に開く
/// （パッケージのパスを開き直さない。検証後の差し替えでも元のパッケージの外は読めない）。
#[derive(Debug)]
pub struct ConfinedPackage {
    /// 正準化した workspace（メンバーの実パスの包含確認に使う）。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    workspace: PathBuf,
    /// 正準化したパッケージディレクトリ。
    dir: ConfinedPath,
    /// 検証時に開いて保持するパッケージのディレクトリ fd。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    handle: ConfinedDir,
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
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    let handle = open_dir_confined(&dir)?;
    Ok(ConfinedPackage {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        workspace: canon_workspace,
        dir,
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        handle,
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
    /// [`crate::path::open_confined`] と同じ拒否に加え、開いた実体が workspace の外なら
    /// [`PathRejection::Escapes`]（[`EscapeKind::Symlink`]）。
    pub fn open_member(&self, member: &Path) -> Result<(File, ConfinedPath), PathRejection> {
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = member;
            Err(PathRejection::UnsupportedPlatform)
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            let (file, real) = self.handle.open_member(member)?;
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
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::*;
    use std::io::Read;
    use std::os::unix::fs::symlink;

    /// REQ-39: 検証後にパッケージのパスが workspace 内の別ディレクトリ（symlink）へ差し替えられても、
    /// 保持した fd 経由で元のパッケージのメンバーを読む（差し替え先は読まない）。
    #[test]
    fn req39_package_swap_after_confine_keeps_original() {
        let base =
            std::env::temp_dir().join(format!("fandhe-guard-pkgswap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("pkg")).expect("mkdir");
        std::fs::create_dir_all(base.join("other")).expect("mkdir");
        std::fs::write(base.join("pkg/artifact.json"), b"original").expect("write");
        std::fs::write(base.join("other/artifact.json"), b"swapped").expect("write");

        let pkg = confine_package(&base, Path::new("pkg")).expect("confine");
        std::fs::rename(base.join("pkg"), base.join("pkg_moved")).expect("rename");
        symlink(base.join("other"), base.join("pkg")).expect("symlink");

        let (mut f, _) = pkg
            .open_member(Path::new("artifact.json"))
            .expect("open by fd");
        let mut s = String::new();
        f.read_to_string(&mut s).expect("read");
        assert_eq!(s, "original");
        let _ = std::fs::remove_dir_all(&base);
    }
}
