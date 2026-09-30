//! 保持したディレクトリ fd 起点の凍結配置先（REQ-17・REQ-39・TASK-33.1-2・#136）。
//!
//! data 層の [`fandhe_edge_data::frozen_placement::place_read_only_bytes`] は、凍結配置の手順
//! （照合 → ステージング → 0400 化 → 書き込み拒否のプローブ → 置き換えない公開 → 片付け）を 1 か所に
//! 持ち、ファイル操作を [`PlacementDir`] 越しに行う。本モジュールはその CLI 側の実装で、`register` が
//! `data/` を開いた fd（[`crate::project::Project::open_subdir`]）を起点に、ガード層の `openat`
//! （`O_NOFOLLOW`）系の操作だけで配置する。パスの正規化・再解決をしないので、検証後にパスが
//! symlink へ差し替えられてもプロジェクトの外を読み書きしない。手順そのものはここに複製しない。
//!
//! Linux・macOS 限定（ガード層の fd 操作の target 依存と同じ。`lib.rs` で cfg により局所化する）。
//! それ以外の OS では本モジュールを持たず、`register` は評価データの配置を従来どおり
//! `runtime_error` で拒否する（fail-closed。Windows は M10 時点で対象外）。

use std::fs::File;
use std::io;
use std::path::Path;

use fandhe_edge_data::frozen_placement::PlacementDir;
use fandhe_edge_guard::package::ConfinedPackage;
use fandhe_edge_guard::path::PathRejection;

/// 保持 fd で開いた配置先ディレクトリ（`data/`）。
pub struct HeldPlacementDir {
    dir: ConfinedPackage,
}

impl HeldPlacementDir {
    /// 開いた `data/` のハンドルから作る。
    #[must_use]
    pub fn new(dir: ConfinedPackage) -> Self {
        Self { dir }
    }
}

/// 閉じ込めの拒否を `io::Error` にする（`kind` は保つ。`AlreadyExists`・`PermissionDenied` を
/// data 層が判別するため。message は固定語彙でパスを含めない）。
fn to_io(rejection: &PathRejection) -> io::Error {
    match rejection {
        PathRejection::Unresolvable { source, .. } => {
            io::Error::new(source.kind(), "placement operation failed")
        }
        _ => io::Error::new(io::ErrorKind::InvalidInput, "placement path rejected"),
    }
}

impl PlacementDir for HeldPlacementDir {
    fn dir_mode_and_uid(&self) -> io::Result<(u32, u32)> {
        use std::os::unix::fs::MetadataExt as _;
        let meta = self.dir.metadata()?;
        Ok((meta.mode() & 0o7777, meta.uid()))
    }

    fn create_private_dir(&self, name: &str) -> io::Result<()> {
        self.dir
            .create_dir_member(Path::new(name))
            .map_err(|e| to_io(&e))
    }

    fn entry_uid(&self, name: &str) -> io::Result<u32> {
        use std::os::unix::fs::MetadataExt as _;
        let sub = self
            .dir
            .open_subdir(Path::new(name))
            .map_err(|e| to_io(&e))?;
        Ok(sub.metadata()?.uid())
    }

    fn create_new_file(&self, rel: &str) -> io::Result<File> {
        self.dir
            .create_new_member(Path::new(rel))
            .map_err(|e| to_io(&e))
    }

    fn probe_append(&self, rel: &str) -> io::Result<()> {
        self.dir
            .open_member_append(Path::new(rel))
            .map(|_| ())
            .map_err(|e| to_io(&e))
    }

    fn publish_no_replace(&self, rel: &str, name: &str) -> io::Result<()> {
        self.dir
            .rename_member(Path::new(rel), Path::new(name))
            .map_err(|e| to_io(&e))
    }

    fn stat_entry(&self, name: &str) -> io::Result<(u64, u64, u32)> {
        use std::os::unix::fs::MetadataExt as _;
        let (file, _) = self
            .dir
            .open_member(Path::new(name))
            .map_err(|e| to_io(&e))?;
        let meta = file.metadata()?;
        Ok((meta.dev(), meta.ino(), meta.mode() & 0o7777))
    }

    fn remove_dir_tree(&self, name: &str) -> io::Result<()> {
        let member = Path::new(name);
        let sub = self.dir.open_subdir(member).map_err(|e| to_io(&e))?;
        sub.clear_contents().map_err(|e| to_io(&e))?;
        // 名前が今開いた実体と同一のときだけ消す（差し替えられていれば他所には触れない）。
        match self.dir.remove_empty_dir_member_if_same(member, &sub) {
            Ok(true) => Ok(()),
            Ok(false) => Err(io::Error::other("staging directory was replaced")),
            Err(e) => Err(to_io(&e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_edge_data::eval_freeze::freeze_eval_data;
    use fandhe_edge_data::frozen_placement::place_read_only_bytes;
    use fandhe_edge_guard::package::confine_package;

    /// REQ-39・REQ-17: 配置先を保持 fd で開いた後に、そのパスが外への symlink へ差し替えられても、
    /// 外のディレクトリには何も書かれない。保持 fd の実パスが開いた場所（`data/`）の配下でなくなる
    /// （退避された）ため、ガード層が fail-closed で拒否し、配置は失敗する（外へは書かない）。
    /// 差し替えは開いた後・配置の前に単一スレッドで行う（競合のタイミングを作れないため、
    /// 「検証後の差し替え」を直列化して再現している）。
    #[test]
    fn req39_held_dir_fails_closed_when_path_swapped_after_open() {
        let base = std::env::temp_dir().join(format!("fandhe-held-dir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("data")).expect("data");
        std::fs::create_dir_all(base.join("outside")).expect("outside");
        // `data/` は 0700 の管理ディレクトリ（`create_dir_member` と同じ契約）。
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(base.join("data"), std::fs::Permissions::from_mode(0o700))
                .expect("chmod");
            std::fs::set_permissions(base.join("outside"), std::fs::Permissions::from_mode(0o700))
                .expect("chmod");
        }
        let held = HeldPlacementDir::new(confine_package(&base, Path::new("data")).expect("open"));
        // 開いた後にパスを差し替える: 元の `data/` を退避し、同じ名前を外への symlink にする。
        std::fs::rename(base.join("data"), base.join("data_moved")).expect("move");
        std::os::unix::fs::symlink(base.join("outside"), base.join("data")).expect("symlink");

        let bytes = b"{\"id\":\"a\"}\n";
        let record = freeze_eval_data(bytes).expect("record");
        let result = place_read_only_bytes(bytes, &held, "evaluation.jsonl", &record, 1024);
        assert!(result.is_err(), "placement must fail closed: {result:?}");
        assert_eq!(
            std::fs::read_dir(base.join("data_moved"))
                .expect("read_dir")
                .count(),
            0,
            "no residue in the moved directory"
        );
        assert_eq!(
            std::fs::read_dir(base.join("outside"))
                .expect("read_dir")
                .count(),
            0,
            "nothing may be written outside"
        );
        let _ = std::fs::remove_dir_all(&base);
    }
}
