//! 所有者専用の置き場所とその中のファイルを、差し替えに耐えて扱う（issue #711。REQ-39）。
//!
//! cli の判定履歴（`history.jsonl`・`history.lock`）が使う。cli は `rustix` を持たないため、`openat` と
//! `fstat` の検査をここに置く。
//!
//! - 置き場所は [`PrivateDir::open`] が `O_NOFOLLOW | O_DIRECTORY` で開いた fd を `fstat` して確かめる
//!   （実ディレクトリ・権限ちょうど 0700・所有者が自分）。以後の操作（開く・名前を変える）はすべてこの fd 起点で、
//!   パスを引き直さない（検査の後に置き場所ごと差し替えられても、検査した実体だけを触る）
//! - ファイルは `openat` に `O_NOFOLLOW | O_NONBLOCK` を付けて開き（symlink・FIFO へ差し替えられても追わない・
//!   止まらない）、開いた fd を `fstat` して、通常ファイル・group / other のビットなし（`mode & 0o077 == 0`）・
//!   所有者が自分であることを確かめる
//! - 「自分」の uid は、置き場所の中に `O_EXCL` で新規作成してすぐ消す空ファイルの所有者から得る
//!   （新規作成した inode の所有者は実効ユーザー。`geteuid` の `unsafe` な FFI・依存の追加を避けるための選択で、
//!   `fandhe_edge_data::frozen_placement`・cli の `serve.lock` と同じ考え方）。0700 でも所有者が他人の置き場所
//!   （root で他人のディレクトリを指した場合）は、この空ファイルを作って消すだけで、対象のファイルには触らない

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::fs::File;
use std::io;
use std::os::fd::OwnedFd;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

use rustix::fs::{AtFlags, FileType, Mode, OFlags, fstat, open, openat, renameat, unlinkat};
use rustix::io::Errno;

/// uid を得るための空ファイルの名前に付ける通し番号（同じプロセスの並行呼び出しで名前を分ける）。
static PROBE_SEQ: AtomicU32 = AtomicU32::new(0);

fn denied() -> io::Error {
    io::ErrorKind::PermissionDenied.into()
}

/// 検査済みの置き場所（symlink でない 0700 の実ディレクトリで、所有者が自分）。
pub struct PrivateDir {
    fd: OwnedFd,
    /// 自分の実効 uid（ファイルの所有者と突き合わせる）。
    uid: u32,
}

impl PrivateDir {
    /// `dir` を開いて検査する。
    ///
    /// # Errors
    /// `dir` が symlink・実ディレクトリでない・権限が 0700 でない・所有者が自分でない、開けない場合
    /// （`PermissionDenied` など）。
    pub fn open(dir: &Path) -> io::Result<Self> {
        Self::open_as(dir, None)
    }

    /// `uid` が `None` なら自分の uid を調べて使う（`Some` はテストで所有者の不一致を作るための口）。
    fn open_as(dir: &Path, uid: Option<u32>) -> io::Result<Self> {
        let fd = open(
            dir,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )?;
        let stat = fstat(&fd)?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::Directory
            || stat.st_mode & 0o777 != 0o700
        {
            return Err(denied());
        }
        let uid = match uid {
            Some(uid) => uid,
            None => probe_uid(&fd)?,
        };
        if stat.st_uid != uid {
            return Err(denied());
        }
        Ok(Self { fd, uid })
    }

    /// `name` を追記用（`O_WRONLY | O_APPEND`）に、無ければ 0600 で作って開く。
    ///
    /// # Errors
    /// `name` が symlink・通常ファイルでない・group / other の権限ビットがある・所有者が自分でない、
    /// 開けない場合（`PermissionDenied` など）。FIFO で止まらない。
    pub fn open_append(&self, name: &str) -> io::Result<File> {
        // 同じ名前を複数プロセスが同時に新規作成すると、macOS では O_CREAT の open が ENOENT で
        // 失敗することがある（実測。#711）。作成の競合として数回だけ再試行する。
        let mut attempt = 0;
        let fd = loop {
            match openat(
                &self.fd,
                name,
                OFlags::WRONLY
                    | OFlags::APPEND
                    | OFlags::CREATE
                    | OFlags::NOFOLLOW
                    | OFlags::NONBLOCK
                    | OFlags::CLOEXEC,
                Mode::from_raw_mode(0o600),
            ) {
                Err(Errno::NOENT) if attempt < 5 => attempt += 1,
                other => break other?,
            }
        };
        let stat = fstat(&fd)?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
            || stat.st_mode & 0o077 != 0
            || stat.st_uid != self.uid
        {
            return Err(denied());
        }
        Ok(File::from(fd))
    }

    /// 置き場所の中で `from` を `to` へ名前変更する（`to` があれば置き換える）。
    /// 呼び出し側は、`from` を [`Self::open_append`] で検査してから呼ぶ。
    ///
    /// # Errors
    /// `renameat` が失敗した場合。
    pub fn rename(&self, from: &str, to: &str) -> io::Result<()> {
        Ok(renameat(&self.fd, from, &self.fd, to)?)
    }
}

/// 置き場所に空ファイルを `O_EXCL` で新規作成し、その所有者（＝自分の実効 uid）を返す。ファイルはすぐ消す。
fn probe_uid(dir: &OwnedFd) -> io::Result<u32> {
    let pid = std::process::id();
    // 前の実行が残した同名のファイル（pid の再利用）に当たったら、番号を変えて数回だけ試す。
    for _ in 0..5 {
        let name = format!(
            ".uid-probe-{pid}-{}",
            PROBE_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        match openat(
            dir,
            &name,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        ) {
            Ok(fd) => {
                let stat = fstat(&fd);
                let _ = unlinkat(dir, &name, AtFlags::empty());
                return Ok(stat?.st_uid);
            }
            Err(Errno::EXIST) => {}
            Err(e) => return Err(e.into()),
        }
    }
    Err(io::ErrorKind::AlreadyExists.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("fe-pf-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("mkdir");
        fs::set_permissions(&d, fs::Permissions::from_mode(0o700)).expect("chmod");
        d
    }

    fn names(d: &Path) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(d)
            .expect("ls")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    /// REQ-39・#711: 新規は 0600 の通常ファイル。既存の 0644・symlink・0755 の置き場所は拒否する。
    #[test]
    fn creates_0600_and_rejects_loose_or_linked() {
        let d = tmp("a");
        let dir = PrivateDir::open(&d).expect("dir");
        let f = dir.open_append("h").expect("create");
        assert_eq!(
            f.metadata().expect("meta").permissions().mode() & 0o777,
            0o600
        );
        // uid を得るための空ファイルは残らない。
        assert_eq!(names(&d), ["h"]);
        fs::set_permissions(d.join("h"), fs::Permissions::from_mode(0o644)).expect("chmod");
        assert_eq!(
            dir.open_append("h").expect_err("loose").kind(),
            io::ErrorKind::PermissionDenied
        );
        symlink(d.join("h"), d.join("l")).expect("symlink");
        assert!(dir.open_append("l").is_err());
        fs::set_permissions(&d, fs::Permissions::from_mode(0o755)).expect("chmod");
        assert_eq!(
            PrivateDir::open(&d).err().expect("dir").kind(),
            io::ErrorKind::PermissionDenied
        );
        let _ = fs::remove_dir_all(&d);
    }

    /// REQ-39・#711: 0700 の置き場所・0600 のファイルでも、所有者が自分でなければ拒否する
    /// （他人の uid のファイルは root でないと作れないため、「自分」の側を別の uid にして不一致を作る）。
    #[test]
    fn rejects_owner_mismatch() {
        let d = tmp("own");
        let me = fs::metadata(&d).expect("meta").uid();
        // 調べた uid は、自分が作ったディレクトリの所有者と同じ。
        assert_eq!(PrivateDir::open(&d).expect("dir").uid, me);
        let other = me.wrapping_add(1);
        assert_eq!(
            PrivateDir::open_as(&d, Some(other))
                .err()
                .expect("dir owner")
                .kind(),
            io::ErrorKind::PermissionDenied
        );
        assert_eq!(names(&d), Vec::<String>::new());

        let mut dir = PrivateDir::open(&d).expect("dir");
        drop(dir.open_append("h").expect("create"));
        dir.uid = other;
        assert_eq!(
            dir.open_append("h").expect_err("file owner").kind(),
            io::ErrorKind::PermissionDenied
        );
        let _ = fs::remove_dir_all(&d);
    }

    /// REQ-39・#711: 名前変更は検査した置き場所の fd 起点で、パスが別の場所へ差し替わっても元の実体の中で行う。
    #[test]
    fn rename_stays_in_opened_dir() {
        let (d, moved) = (tmp("ren"), tmp("ren-moved"));
        fs::remove_dir(&moved).expect("rmdir");
        let dir = PrivateDir::open(&d).expect("dir");
        drop(dir.open_append("a").expect("create"));
        // 置き場所のパスを別の実体へ差し替える。
        fs::rename(&d, &moved).expect("mv");
        fs::create_dir(&d).expect("mkdir");
        dir.rename("a", "b").expect("rename");
        assert_eq!(names(&moved), ["b"]);
        assert_eq!(names(&d), Vec::<String>::new());
        let _ = fs::remove_dir_all(&d);
        let _ = fs::remove_dir_all(&moved);
    }
}
