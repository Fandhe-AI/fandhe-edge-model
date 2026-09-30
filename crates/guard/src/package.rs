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

impl ConfinedPackage {
    /// 新規ファイルを作って書き込み用に返す（既存・symlink は拒否）。親は保持 fd 起点で辿る
    /// （検証後の親の差し替えでも外へ書かない。REQ-39）。Linux・macOS 以外は拒否（fail-closed）。
    ///
    /// # Errors
    /// [`crate::path::ConfinedDir::create_new_member`] と同じ。
    pub fn create_new_member(&self, member: &Path) -> Result<File, PathRejection> {
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = member;
            Err(PathRejection::UnsupportedPlatform)
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            self.handle.create_new_member(member)
        }
    }

    /// ディレクトリを所有者のみ（0700）で新規作成する（既存なら拒否）。
    ///
    /// # Errors
    /// [`crate::path::ConfinedDir::create_dir_member`] と同じ。
    pub fn create_dir_member(&self, member: &Path) -> Result<(), PathRejection> {
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = member;
            Err(PathRejection::UnsupportedPlatform)
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            self.handle.create_dir_member(member)
        }
    }

    /// 書き込み失敗後の片付け用にファイルを削除する。
    ///
    /// # Errors
    /// [`crate::path::ConfinedDir::remove_file_member`] と同じ。
    pub fn remove_file_member(&self, member: &Path) -> Result<(), PathRejection> {
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = member;
            Err(PathRejection::UnsupportedPlatform)
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            self.handle.remove_file_member(member)
        }
    }
}

impl ConfinedPackage {
    /// `member`（本パッケージ配下のディレクトリ）を fd 起点で開き、その fd を保持する新しい
    /// [`ConfinedPackage`] を返す（作成直後のディレクトリの同一性を fd で握る。REQ-39）。
    ///
    /// # Errors
    /// [`crate::path::ConfinedDir::open_dir_member`] と同じ。
    pub fn open_subdir(&self, member: &Path) -> Result<ConfinedPackage, PathRejection> {
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = member;
            Err(PathRejection::UnsupportedPlatform)
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            let handle = self.handle.open_dir_member(member)?;
            if !handle.as_path().starts_with(&self.workspace) {
                return Err(PathRejection::Escapes {
                    candidate: member.to_path_buf(),
                    kind: EscapeKind::Symlink,
                });
            }
            Ok(ConfinedPackage {
                workspace: self.workspace.clone(),
                dir: ConfinedPath::from_verified(handle.as_path().to_path_buf()),
                handle,
            })
        }
    }

    /// 本パッケージの中身を、保持した fd 起点で再帰的に削除する（本ディレクトリは残す）。
    ///
    /// # Errors
    /// [`crate::path::ConfinedDir::clear_contents`] と同じ。
    pub fn clear_contents(&self) -> Result<(), PathRejection> {
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            Err(PathRejection::UnsupportedPlatform)
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            self.handle.clear_contents()
        }
    }

    /// `member` が `created`（本工程が作って fd を保持しているディレクトリ）と同一の実体で、
    /// かつ空のときだけ削除する。差し替えられていれば何も消さず `Ok(false)`。
    ///
    /// # Errors
    /// 経路の拒否・削除失敗。
    pub fn remove_empty_dir_member_if_same(
        &self,
        member: &Path,
        created: &ConfinedPackage,
    ) -> Result<bool, PathRejection> {
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = (member, created);
            Err(PathRejection::UnsupportedPlatform)
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            if !self.handle.is_same_dir_member(member, &created.handle)? {
                return Ok(false);
            }
            self.handle.remove_empty_dir_member(member)?;
            Ok(true)
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

    /// REQ-39: `confine_package` の後に子ディレクトリが外部へ向く symlink へ差し替えられても、
    /// `create_new_member` / `create_dir_member` は外へ書かず拒否する。
    #[test]
    fn req39_create_member_rejects_parent_swapped_to_symlink() {
        let base = std::env::temp_dir().join(format!("fandhe-guard-wswap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("pkg/data")).expect("mkdir");
        std::fs::create_dir_all(base.join("outside")).expect("mkdir");

        let pkg = confine_package(&base, Path::new("pkg")).expect("confine");
        std::fs::remove_dir(base.join("pkg/data")).expect("rmdir");
        symlink(base.join("outside"), base.join("pkg/data")).expect("symlink");

        let file = pkg.create_new_member(Path::new("data/x.json"));
        assert!(
            matches!(file, Err(PathRejection::Escapes { .. })),
            "{file:?}"
        );
        let dir = pkg.create_dir_member(Path::new("data/sub"));
        assert!(matches!(dir, Err(PathRejection::Escapes { .. })), "{dir:?}");
        assert!(!base.join("outside/x.json").exists());
        assert!(!base.join("outside/sub").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// REQ-39: `confine_package` の後にパッケージのディレクトリ自体が移動されたら、書き込み系
    /// （`create_new_member`・`create_dir_member`・`remove_file_member`）は移動後の場所へ書かず拒否する。
    #[test]
    fn req39_write_rejects_package_dir_moved_after_confine() {
        let base = std::env::temp_dir().join(format!("fandhe-guard-wmoved-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("pkg/data")).expect("mkdir");
        std::fs::write(base.join("pkg/old.txt"), b"x").expect("write");
        let pkg = confine_package(&base, Path::new("pkg")).expect("confine");
        std::fs::create_dir_all(base.join("elsewhere")).expect("mkdir");
        std::fs::rename(base.join("pkg"), base.join("elsewhere/pkg")).expect("move");

        for rel in ["new.txt", "data/new.txt"] {
            let r = pkg.create_new_member(Path::new(rel));
            assert!(matches!(r, Err(PathRejection::Escapes { .. })), "{r:?}");
        }
        let d = pkg.create_dir_member(Path::new("sub"));
        assert!(matches!(d, Err(PathRejection::Escapes { .. })), "{d:?}");
        let rm = pkg.remove_file_member(Path::new("old.txt"));
        assert!(matches!(rm, Err(PathRejection::Escapes { .. })), "{rm:?}");
        assert!(!base.join("elsewhere/pkg/new.txt").exists());
        assert!(!base.join("elsewhere/pkg/data/new.txt").exists());
        assert!(!base.join("elsewhere/pkg/sub").exists());
        assert!(base.join("elsewhere/pkg/old.txt").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// REQ-39: 既存の名前（symlink を含む）は上書きせず `AlreadyExists` で拒否し、`..` は拒否する。
    /// 書き込み後の `remove_file_member` でファイルが消える。
    #[test]
    fn req39_create_member_exclusive_and_confined_names() {
        let base = std::env::temp_dir().join(format!("fandhe-guard-wexcl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("pkg")).expect("mkdir");
        std::fs::write(base.join("target.txt"), b"keep").expect("write");
        symlink(base.join("target.txt"), base.join("pkg/link")).expect("symlink");

        let pkg = confine_package(&base, Path::new("pkg")).expect("confine");
        let e = pkg
            .create_new_member(Path::new("link"))
            .expect_err("exists");
        assert!(
            matches!(&e, PathRejection::Unresolvable { source, .. }
                if source.kind() == std::io::ErrorKind::AlreadyExists),
            "{e:?}"
        );
        assert_eq!(
            std::fs::read(base.join("target.txt")).expect("read"),
            b"keep"
        );
        assert!(matches!(
            pkg.create_new_member(Path::new("../evil")),
            Err(PathRejection::Escapes { .. })
        ));
        pkg.create_new_member(Path::new("new.txt")).expect("create");
        assert!(base.join("pkg/new.txt").exists());
        pkg.remove_file_member(Path::new("new.txt"))
            .expect("remove");
        assert!(!base.join("pkg/new.txt").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// REQ-39: 保持 fd 起点の後始末は、名前が別ディレクトリへ差し替えられても差し替え先を消さない。
    #[test]
    fn req39_remove_created_dir_leaves_swapped_directory_untouched() {
        let base = std::env::temp_dir().join(format!("fandhe-guard-rmswap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("ws")).expect("mkdir");
        let ws = confine_package(&base, Path::new("ws")).expect("confine");
        ws.create_dir_member(Path::new("made")).expect("mkdirat");
        ws.create_new_member(Path::new("made/a.txt")).expect("file");
        let created = ws.open_subdir(Path::new("made")).expect("open subdir");

        // 作成後に、同名を別の実ディレクトリ（中身あり）へ差し替える。
        std::fs::rename(base.join("ws/made"), base.join("ws/made_orig")).expect("rename");
        std::fs::create_dir(base.join("ws/made")).expect("mkdir swapped");
        std::fs::write(base.join("ws/made/victim.txt"), b"keep").expect("write");

        created.clear_contents().expect("clear via fd");
        let removed = ws
            .remove_empty_dir_member_if_same(Path::new("made"), &created)
            .expect("compare");
        assert!(!removed);
        assert_eq!(
            std::fs::read(base.join("ws/made/victim.txt")).expect("victim"),
            b"keep"
        );
        // 保持していた実体（移動後の made_orig）の中身は消えている。
        assert!(!base.join("ws/made_orig/a.txt").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// REQ-39: 差し替えが無ければ、中身（入れ子・symlink を含む）ごと作成したディレクトリを消す。
    /// symlink は追従せず、リンク先は消さない。
    #[test]
    fn req39_remove_created_dir_removes_tree_without_following_symlink() {
        let base = std::env::temp_dir().join(format!("fandhe-guard-rmtree-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("ws")).expect("mkdir");
        std::fs::create_dir_all(base.join("outside")).expect("mkdir");
        std::fs::write(base.join("outside/keep.txt"), b"keep").expect("write");
        let ws = confine_package(&base, Path::new("ws")).expect("confine");
        ws.create_dir_member(Path::new("made")).expect("mkdirat");
        ws.create_dir_member(Path::new("made/sub"))
            .expect("mkdirat");
        ws.create_new_member(Path::new("made/sub/a.txt"))
            .expect("file");
        symlink(base.join("outside"), base.join("ws/made/link")).expect("symlink");
        let created = ws.open_subdir(Path::new("made")).expect("open subdir");

        created.clear_contents().expect("clear");
        let removed = ws
            .remove_empty_dir_member_if_same(Path::new("made"), &created)
            .expect("remove");
        assert!(removed);
        assert!(!base.join("ws/made").exists());
        assert_eq!(
            std::fs::read(base.join("outside/keep.txt")).expect("keep"),
            b"keep"
        );
        let _ = std::fs::remove_dir_all(&base);
    }
}
