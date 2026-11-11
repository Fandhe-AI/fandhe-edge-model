//! 所有者専用の置き場所にある 1 ファイルを、差し替えに耐えて開く（issue #711。REQ-39）。
//!
//! cli の判定履歴（`history.jsonl`・`history.lock`）が使う。cli は `rustix` を持たないため、`openat` と
//! `fstat` の検査をここに置く。置き場所は `O_NOFOLLOW | O_DIRECTORY` で開いた fd を `fstat` し
//! （実ディレクトリ・権限ちょうど 0700）、ファイルはその fd 起点の `openat` に `O_NOFOLLOW | O_NONBLOCK`
//! を付けて開く（検査と open の間に symlink・FIFO へ差し替えられても追わない・止まらない）。
//! 開いた後に fd を `fstat` して、通常ファイルで group / other のビットが無い（`mode & 0o077 == 0`）ことを確かめる。
//! 置き場所の所有者が自分かどうかは確かめない（std で自分の uid を取れないため）。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::fs::File;
use std::io;
use std::path::Path;

use rustix::fs::{FileType, Mode, OFlags, fstat, open, openat};

/// `dir` の `name` を追記用（`O_WRONLY | O_APPEND`）に、無ければ 0600 で作って開く。
///
/// # Errors
/// `dir` が symlink・実ディレクトリでない・権限が 0700 でない、`name` が symlink・通常ファイルでない・
/// group / other の権限ビットがある、開けない場合（`PermissionDenied` など）。FIFO で止まらない。
pub fn open_private_append(dir: &Path, name: &str) -> io::Result<File> {
    let denied = || io::Error::from(io::ErrorKind::PermissionDenied);
    let dir_fd = open(
        dir,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )?;
    let dir_stat = fstat(&dir_fd)?;
    if FileType::from_raw_mode(dir_stat.st_mode) != FileType::Directory
        || dir_stat.st_mode & 0o777 != 0o700
    {
        return Err(denied());
    }
    // 同じ名前を複数プロセスが同時に新規作成すると、macOS では O_CREAT の open が ENOENT で
    // 失敗することがある（実測。#711）。作成の競合として数回だけ再試行する。
    let mut attempt = 0;
    let fd = loop {
        match openat(
            &dir_fd,
            name,
            OFlags::WRONLY
                | OFlags::APPEND
                | OFlags::CREATE
                | OFlags::NOFOLLOW
                | OFlags::NONBLOCK
                | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        ) {
            Err(rustix::io::Errno::NOENT) if attempt < 5 => attempt += 1,
            other => break other?,
        }
    };
    let stat = fstat(&fd)?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile || stat.st_mode & 0o077 != 0 {
        return Err(denied());
    }
    Ok(File::from(fd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::{PermissionsExt, symlink};

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("fe-pf-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("mkdir");
        fs::set_permissions(&d, fs::Permissions::from_mode(0o700)).expect("chmod");
        d
    }

    /// REQ-39・#711: 新規は 0600 の通常ファイル。既存の 0644・symlink・0755 の置き場所は拒否する。
    #[test]
    fn creates_0600_and_rejects_loose_or_linked() {
        let d = tmp("a");
        let f = open_private_append(&d, "h").expect("create");
        assert_eq!(
            f.metadata().expect("meta").permissions().mode() & 0o777,
            0o600
        );
        fs::set_permissions(d.join("h"), fs::Permissions::from_mode(0o644)).expect("chmod");
        assert_eq!(
            open_private_append(&d, "h").expect_err("loose").kind(),
            io::ErrorKind::PermissionDenied
        );
        symlink(d.join("h"), d.join("l")).expect("symlink");
        assert!(open_private_append(&d, "l").is_err());
        fs::set_permissions(&d, fs::Permissions::from_mode(0o755)).expect("chmod");
        assert_eq!(
            open_private_append(&d, "n").expect_err("dir").kind(),
            io::ErrorKind::PermissionDenied
        );
        let _ = fs::remove_dir_all(&d);
    }
}
