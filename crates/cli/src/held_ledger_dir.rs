//! 保持したディレクトリ fd 起点の最終 test 台帳（REQ-27・REQ-39・#314・#168）。
//!
//! 評価器の [`FinalTestLedger`](fandhe_edge_eval::final_test_once::FinalTestLedger) は、台帳の
//! 手順（事前登録・封印・適用ロック・完了記録）を 1 か所に持ち、ファイル操作を
//! [`LedgerDir`] 越しに行う。本モジュールはその CLI 側の実装で、`evaluate`・`package` が
//! `final_test_ledger/` を開いた fd（[`crate::project::Project::open_subdir`]）を起点に、ガード層の
//! `openat`（`O_NOFOLLOW`）系の操作だけで台帳を読み書きする。パスの正規化・再解決をしないため、
//! 検証後に台帳やメンバーのパスが symlink へ差し替えられても、プロジェクトの外を読み書きしない
//! （差し替えられた実体は fd の実パスがずれるためガード層が fail-closed で拒否する）。
//! 手順そのものはここに複製しない（[`crate::frozen_dir`] と同じ構造）。
//!
//! Linux・macOS 限定（ガード層の fd 操作の target 依存と同じ。`lib.rs` で cfg により局所化する）。
//! それ以外の OS では本モジュールを持たず、台帳を使う工程は `runtime_error` で拒否する
//! （fail-closed。Windows は M10 時点で対象外）。

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use fandhe_edge_core::fs::FsError;
use fandhe_edge_eval::ledger_dir::{EntryKind, LedgerDir};
use fandhe_edge_guard::package::ConfinedPackage;
use fandhe_edge_guard::path::PathRejection;

/// 保持 fd で開いた台帳ディレクトリ（`final_test_ledger/`）。
pub struct HeldLedgerDir {
    dir: ConfinedPackage,
}

impl HeldLedgerDir {
    /// 開いた台帳ディレクトリのハンドルから作る。
    #[must_use]
    pub fn new(dir: ConfinedPackage) -> Self {
        Self { dir }
    }

    /// 台帳直下の `rel`（`<scope>` または `<scope>/<name>`）を、エラー表示用の絶対パスにする。
    fn display(&self, rel: &str) -> PathBuf {
        self.dir.dir().join(rel)
    }
}

/// 閉じ込めの拒否を `io::Error` にする（`kind` は保つ。`NotFound`・`AlreadyExists` を台帳ロジックが
/// 判別するため。message は固定語彙でパスを含めない）。
fn to_io(rejection: &PathRejection) -> io::Error {
    match rejection {
        PathRejection::Unresolvable { source, .. } => {
            io::Error::new(source.kind(), "ledger operation failed")
        }
        _ => io::Error::new(io::ErrorKind::InvalidInput, "ledger path rejected"),
    }
}

impl LedgerDir for HeldLedgerDir {
    fn create_dir(&self, rel: &str) -> io::Result<()> {
        self.dir
            .create_dir_member(Path::new(rel))
            .map_err(|e| to_io(&e))
    }

    fn entry_kind(&self, rel: &str) -> io::Result<EntryKind> {
        let member = Path::new(rel);
        match self.dir.open_subdir(member) {
            Ok(_) => return Ok(EntryKind::Dir),
            Err(e @ PathRejection::Unresolvable { .. }) => return Err(to_io(&e)),
            // ディレクトリでない・symlink: 次にファイルとして調べる。
            Err(_) => {}
        }
        match self.dir.open_member(member) {
            Ok((file, _)) => Ok(if file.metadata()?.is_file() {
                EntryKind::File
            } else {
                EntryKind::Other
            }),
            Err(e @ PathRejection::Unresolvable { .. }) => Err(to_io(&e)),
            // symlink・通常ファイルでない実体は「その他」（辿らない）。
            Err(_) => Ok(EntryKind::Other),
        }
    }

    fn create_new_file(&self, rel: &str) -> io::Result<File> {
        let file = self
            .dir
            .create_new_member(Path::new(rel))
            .map_err(|e| to_io(&e))?;
        // ガード層は umask 任せの 0666 で作る。台帳のファイルは所有者のみ（0600）にそろえる。
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(file)
    }

    fn read_bounded(&self, rel: &str, limit: u64) -> Result<Vec<u8>, FsError> {
        let path = self.display(rel);
        match self.dir.open_member(Path::new(rel)) {
            Ok((file, _)) => fandhe_edge_core::fs::read_bounded_open_file(file, &path, limit),
            Err(PathRejection::NotRegularFile { .. }) => Err(FsError::NotRegularFile { path }),
            Err(e) => Err(FsError::Read {
                path,
                source: to_io(&e),
            }),
        }
    }

    fn is_read_only_file(&self, rel: &str) -> bool {
        self.dir
            .open_member(Path::new(rel))
            .ok()
            .and_then(|(file, _)| file.metadata().ok())
            .is_some_and(|m| m.is_file() && m.permissions().readonly())
    }

    fn make_read_only(&self, rel: &str) -> io::Result<()> {
        let (file, _) = self
            .dir
            .open_member(Path::new(rel))
            .map_err(|e| to_io(&e))?;
        let mut perms = file.metadata()?.permissions();
        perms.set_readonly(true);
        file.set_permissions(perms)
    }

    fn list_names(&self, rel: &str, limit: usize) -> io::Result<Vec<Option<String>>> {
        let sub = self
            .dir
            .open_subdir(Path::new(rel))
            .map_err(|e| to_io(&e))?;
        Ok(sub
            .list_entry_names(limit)?
            .into_iter()
            .map(|n| n.into_string().ok())
            .collect())
    }

    fn sync_dir(&self, rel: &str) -> io::Result<()> {
        if rel.is_empty() {
            return self.dir.sync_all();
        }
        self.dir
            .open_subdir(Path::new(rel))
            .map_err(|e| to_io(&e))?
            .sync_all()
    }

    fn display_path(&self, rel: &str) -> PathBuf {
        self.display(rel)
    }
}
