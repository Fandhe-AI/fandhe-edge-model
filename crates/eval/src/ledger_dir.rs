//! 最終 test の台帳ディレクトリ操作の抽象（REQ-27・REQ-39・#314・#168）。
//!
//! [`crate::final_test_once::FinalTestLedger`] のファイル操作（メンバーの検査・上限付き読み込み・
//! 排他的な作成・読み取り専用化・列挙・永続化）はすべてこの trait 越しに行う。台帳の手順
//! （事前登録・封印・適用ロック・完了記録）は `final_test_once` の 1 か所に置き、ここは
//! 「どの実体に対して操作するか」だけを差し替える。データ契約層の
//! `fandhe_edge_data::frozen_placement::PlacementDir` と同じ形の継ぎ目で、eval は guard・rustix に
//! 依存しないため、fd 起点の実装は呼び出し側（CLI。保持したディレクトリ fd 起点の `openat`）が
//! 書く。パス版は [`StdLedgerDir`]（std のパス操作。[`FinalTestLedger::open`] が使う）。
//!
//! 名前引数 `rel` は台帳直下からの相対で、`<scope>` または `<scope>/<name>` の 1 段の入れ子のみ
//! （コードが生成した hex 名だけ。呼び出し側の文字列は入らない）。`""` は台帳ディレクトリ自身
//! （[`LedgerDir::sync_dir`] のみ）。実装は台帳の外へ出してはならず、シンボリックリンクを辿らない
//! 操作にすること（検証後の差し替えで閉じ込め外を読み書きしない。REQ-39）。
//!
//! [`FinalTestLedger::open`]: crate::final_test_once::FinalTestLedger::open

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use fandhe_edge_core::fs::FsError;

/// 台帳内のエントリの種別（シンボリックリンクは辿らない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EntryKind {
    /// 実ディレクトリ。
    Dir,
    /// 通常ファイル。
    File,
    /// それ以外（シンボリックリンク・FIFO 等）。
    Other,
}

/// 台帳ディレクトリの操作（fd 起点・パス起点のどちらの実装も同じ台帳ロジックを通す）。
pub trait LedgerDir: Send + Sync {
    /// `rel` にディレクトリを所有者のみ（0700）で新規作成する（既存なら `AlreadyExists`）。
    fn create_dir(&self, rel: &str) -> io::Result<()>;
    /// `rel` の種別（辿らない）。無ければ `NotFound` の `Err`。
    fn entry_kind(&self, rel: &str) -> io::Result<EntryKind>;
    /// `rel` に新規ファイルを排他的に作り、書き込み用に返す（既存なら `AlreadyExists`。0600）。
    fn create_new_file(&self, rel: &str) -> io::Result<File>;
    /// `rel` の通常ファイルを `limit` バイトまでの上限付きで読む（通常ファイルでなければ拒否）。
    ///
    /// # Errors
    /// [`FsError`]（開けない・上限超過・通常ファイルでない）。
    fn read_bounded(&self, rel: &str, limit: u64) -> Result<Vec<u8>, FsError>;
    /// `rel` が読み取り専用の通常ファイルか（辿らない。判定できなければ偽）。
    fn is_read_only_file(&self, rel: &str) -> bool;
    /// `rel` の通常ファイルから書き込み権を外す。
    fn make_read_only(&self, rel: &str) -> io::Result<()>;
    /// `rel`（ディレクトリ）の直下のエントリ名を最大 `limit + 1` 件返す（UTF-8 でない名前は
    /// `None`。件数が `limit` を超えたかを呼び出し側が判定する。走査量の上限。REQ-39）。
    fn list_names(&self, rel: &str, limit: usize) -> io::Result<Vec<Option<String>>>;
    /// `rel`（`""` は台帳自身）のディレクトリエントリを永続化する。
    fn sync_dir(&self, rel: &str) -> io::Result<()>;
    /// エラー表示専用のパス（ファイルを開き直すために使わない）。
    fn display_path(&self, rel: &str) -> PathBuf;
}

/// std のパス操作による [`LedgerDir`]（[`crate::final_test_once::FinalTestLedger::open`] が使う）。
///
/// パスを都度解決するため、検証後の差し替えには弱い。信頼できる作業ディレクトリ（テスト・
/// 単体利用）向けで、CLI は fd 起点の実装を渡す（REQ-39）。
#[derive(Debug, Clone)]
pub struct StdLedgerDir {
    root: PathBuf,
}

impl StdLedgerDir {
    /// `root` を台帳ディレクトリとして包む（存在確認は呼び出し側）。
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// 台帳ディレクトリのパス。
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path(&self, rel: &str) -> PathBuf {
        if rel.is_empty() {
            self.root.clone()
        } else {
            self.root.join(rel)
        }
    }
}

impl LedgerDir for StdLedgerDir {
    fn create_dir(&self, rel: &str) -> io::Result<()> {
        #[cfg_attr(not(unix), allow(unused_mut))]
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt as _;
            builder.mode(0o700);
        }
        builder.create(self.path(rel))
    }

    fn entry_kind(&self, rel: &str) -> io::Result<EntryKind> {
        let meta = fs::symlink_metadata(self.path(rel))?;
        Ok(if meta.is_dir() {
            EntryKind::Dir
        } else if meta.file_type().is_file() {
            EntryKind::File
        } else {
            EntryKind::Other
        })
    }

    fn create_new_file(&self, rel: &str) -> io::Result<File> {
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            opts.mode(0o600);
        }
        opts.open(self.path(rel))
    }

    fn read_bounded(&self, rel: &str, limit: u64) -> Result<Vec<u8>, FsError> {
        fandhe_edge_core::fs::read_bounded(&self.path(rel), limit)
    }

    fn is_read_only_file(&self, rel: &str) -> bool {
        fs::symlink_metadata(self.path(rel))
            .map(|m| m.file_type().is_file() && m.permissions().readonly())
            .unwrap_or(false)
    }

    fn make_read_only(&self, rel: &str) -> io::Result<()> {
        let path = self.path(rel);
        let mut perms = fs::metadata(&path)?.permissions();
        perms.set_readonly(true);
        fs::set_permissions(&path, perms)
    }

    fn list_names(&self, rel: &str, limit: usize) -> io::Result<Vec<Option<String>>> {
        let mut names = Vec::new();
        for entry in fs::read_dir(self.path(rel))? {
            names.push(entry?.file_name().into_string().ok());
            if names.len() > limit {
                break;
            }
        }
        Ok(names)
    }

    #[cfg(unix)]
    fn sync_dir(&self, rel: &str) -> io::Result<()> {
        File::open(self.path(rel)).and_then(|d| d.sync_all())
    }

    /// 非 Unix ではディレクトリハンドルの `sync_all` が使えない（ロックファイル自体は `sync_all`
    /// 済みのため成功扱い。エントリの永続化は OS 任せ）。
    #[cfg(not(unix))]
    fn sync_dir(&self, _rel: &str) -> io::Result<()> {
        Ok(())
    }

    fn display_path(&self, rel: &str) -> PathBuf {
        self.path(rel)
    }
}
