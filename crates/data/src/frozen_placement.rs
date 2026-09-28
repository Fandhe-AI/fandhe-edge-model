//! 凍結済み評価データの読み取り専用配置と、直接書き込みの拒否確認
//! （REQ-39・REQ-17・TASK-17.2-2）。
//!
//! # 呼び出し文脈
//!
//! 将来の CLI `evaluate` 工程（REQ-33・TASK-33.3・issue #140）から、
//! [`crate::eval_freeze::freeze_eval_data`] で凍結記録を作った直後に呼ばれる
//! 想定の順序は「凍結（[`crate::eval_freeze`]）→ 配置（本モジュール）→
//! 評価時の再計算（[`crate::eval_freeze::evaluate_gate`]）」。本モジュールは
//! ハッシュ照合を一切行わない（凍結記録との結び付けは呼び出し側の責務）。
//!
//! # 責務の境界（本モジュールが行わないこと）
//!
//! - **経路の閉じ込め（`../`・絶対パス・symlink によるルート外参照の拒否）は
//!   行わない**（TASK-39.x・issue #157/#158 の対象。本モジュールが受け取る
//!   `path` はガード層を通過済みであることを前提とする。crate 全体の前提
//!   条件。`crates/data/src/lib.rs`）。ただし symlink そのものへの権限変更は
//!   本モジュール自身の安全のため拒否する（後述）
//! - **読み込み前のサイズ上限検査は行わない**（issue #172 の対象）。本
//!   モジュールは評価データ本体を読み込まず、`chmod` 相当の権限変更のみ行う
//! - **ハッシュ不一致検知（凍結記録との突き合わせ）は行わない**
//!   （TASK-17.3・issue #49 の対象）
//! - **親ディレクトリ単位の読み取り専用化は行わない**。ファイルの mode だけ
//!   では、書き込み可能な親ディレクトリ内での unlink・rename による
//!   差し替えは防げない（PoC-20 もファイル単位の `chmod 444` のみを実測
//!   している）。差し替えは凍結記録の sha256 による事後検知
//!   （[`crate::eval_freeze::evaluate_gate`]・issue #49）で捕捉する
//! - **root 実行下での書き込み防止は保証しない**。root は mode `0o444` でも
//!   書き込めるため、[`place_read_only`] は書き込みを防げない配置を
//!   「配置済み」と装わず、[`PlacementError::WriteNotRejected`] で
//!   fail-closed に失敗させる（後述「root・ACL の扱い」）
//! - **CLI の出力 JSON 全体の形は決めない**。入出力契約は TASK-33.3 に委ねる
//!
//! # 手順（[`place_read_only`]）
//!
//! 1. `symlink_metadata` で symlink・非通常ファイルを拒否する（リンク先が
//!    評価ディレクトリ外かもしれないファイルの権限を、本関数の副作用で
//!    書き換えないため）
//! 2. [`fandhe_edge_core::fs::open_regular_file_for_read`] で開く
//! 3. unix では、手順 1 の `(dev, ino)` と開いたハンドルのそれを突き合わせ、
//!    検査からオープンまでの差し替え（TOCTOU）を検出する
//! 4. ハンドル経由で権限を `0o444`（読み取り専用・setuid/setgid/sticky・
//!    実行ビットなし）に変更する（unix）。非 unix ではパス経由で
//!    読み取り専用属性を立てる（Windows の `File::set_permissions` は
//!    読み取り専用ハンドルでは失敗しうるため）
//! 5. 反映を確認する
//! 6. [`verify_direct_write_rejected`] を呼び、実際に書き込みが拒否される
//!    ことを確認できた場合に限り成功とする（fail-closed）
//!
//! # root・ACL の扱い（安全側に倒した判断）
//!
//! mode ビットの設定だけでは、root（`CAP_DAC_OVERRIDE`）・書き込みを許す
//! POSIX ACL・mode を無視するファイルシステムのいずれでも書き込みを防げない
//! ことがある。[`place_read_only`] は mode を設定した「つもり」で終わらず、
//! 必ず [`verify_direct_write_rejected`] の実際の書き込み試行で拒否を
//! 確認し、拒否を確認できなければ [`PlacementError::WriteNotRejected`] を
//! 返して fail-closed にする（`.claude/rules/coding-rust.md`
//! 「未実装・簡易実装の箇所は実装済みを装わない」）。root 実行を許すか
//! どうかの方針は CLI 配線（TASK-33.3）で決めるべき事項とする。
//!
//! # セキュリティ上の注意
//!
//! [`PlacementError`] の `Display` は英語固定で、評価データ本文を含めない
//! （`.claude/rules/security.md`）。本モジュールはパスをシェル・子プロセスへ
//! 渡さない（`PathBuf` のみで扱う。テストの FIFO 作成のみ例外的に子プロセス
//! `mkfifo` を使う）。
//!
//! # 出典
//!
//! [`verify_direct_write_rejected`] の非破壊プローブ（`append` モードで
//! 開いて即座に閉じる）は PoC-20 ケース 5
//! （`docs/spec/03-poc/safety-hardening/scripts/case5_eval_integrity.py`）の
//! `open(eval_path, "a")` を移植したもの。PoC は `chmod 444` した評価
//! ファイルを追記モードで開くと `PermissionError`（errno 13）になることを
//! 実測している（証拠の種別: テストハーネス）。

use fandhe_edge_core::fs::{FsError, open_regular_file_for_read};
use std::fmt;
use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// [`place_read_only`] が成功した証跡（読み取り専用配置が成立したことを
/// 表す）。フィールドは非公開にし、[`place_read_only`] を通してしか作れない
/// （形だけの「配置済み」偽装を防ぐ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadOnlyPlacement {
    #[cfg(unix)]
    mode: u32,
    #[cfg(not(unix))]
    readonly: bool,
}

impl ReadOnlyPlacement {
    /// 配置後の unix パーミッションビット（`mode & 0o7777`）。
    #[cfg(unix)]
    #[must_use]
    pub fn mode(&self) -> u32 {
        self.mode
    }

    /// 配置後に読み取り専用属性が立っているか（非 unix）。
    #[cfg(not(unix))]
    #[must_use]
    pub fn is_readonly(&self) -> bool {
        self.readonly
    }
}

/// [`place_read_only`]・[`verify_direct_write_rejected`] が失敗する理由。
#[derive(Debug)]
#[non_exhaustive]
pub enum PlacementError {
    /// パス先が symlink だった。リンク先（評価ディレクトリ外かもしれない
    /// ファイル）への副作用を避けるため、権限変更を一切行わずに拒否する。
    Symlink { path: PathBuf },
    /// パス先が通常ファイルではない（ディレクトリ・FIFO・ソケット等）。
    NotRegularFile { path: PathBuf },
    /// 検査時（`symlink_metadata`）とオープン後のハンドルとで実体が異なる
    /// （`(dev, ino)` 不一致。unix 限定の TOCTOU 対策）。検査からオープンの
    /// 間にパスが差し替えられた可能性がある。
    Replaced { path: PathBuf },
    /// 権限を読み取り専用へ変更した後も、実際の書き込み試行
    /// （[`verify_direct_write_rejected`]）が拒否されなかった。
    ///
    /// root（`CAP_DAC_OVERRIDE`）・書き込みを許す ACL・mode を無視する
    /// ファイルシステム等、mode ビットの設定だけでは書き込みを防げない
    /// 状況を検出する（fail-closed。モジュール doc「root・ACL の扱い」）。
    WriteNotRejected { path: PathBuf },
    /// 本モジュールの前提条件（通常ファイル判定・開いての読み込み）を
    /// 満たせなかった（[`fandhe_edge_core::fs`] 由来）。
    Fs(FsError),
    /// 権限変更・メタデータ取得などの I/O エラー。
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl fmt::Display for PlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlacementError::Symlink { path } => {
                write!(
                    f,
                    "{} is a symlink, refusing to change permissions",
                    path.display()
                )
            }
            PlacementError::NotRegularFile { path } => {
                write!(f, "{} is not a regular file", path.display())
            }
            PlacementError::Replaced { path } => write!(
                f,
                "{} was replaced between the pre-check and opening it",
                path.display()
            ),
            PlacementError::WriteNotRejected { path } => write!(
                f,
                "{} is still writable after setting read-only permissions",
                path.display()
            ),
            PlacementError::Fs(source) => write!(f, "{source}"),
            PlacementError::Io { path, source } => {
                write!(
                    f,
                    "failed to change permissions on {}: {source}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for PlacementError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PlacementError::Fs(source) => Some(source),
            PlacementError::Io { source, .. } => Some(source),
            PlacementError::Symlink { .. }
            | PlacementError::NotRegularFile { .. }
            | PlacementError::Replaced { .. }
            | PlacementError::WriteNotRejected { .. } => None,
        }
    }
}

impl From<FsError> for PlacementError {
    fn from(source: FsError) -> Self {
        PlacementError::Fs(source)
    }
}

/// symlink・非通常ファイルを拒否する事前検査（[`place_read_only`]・
/// [`verify_direct_write_rejected`] の両方が使う共通の入口）。
///
/// `std::fs::symlink_metadata` はリンクを辿らないため、symlink 自身の種別を
/// 判定できる（`std::fs::metadata` はリンクを辿ってしまい、リンク先が
/// 通常ファイルであれば symlink であることを見逃す）。
fn check_not_symlink(path: &Path) -> Result<(), PlacementError> {
    let meta = std::fs::symlink_metadata(path).map_err(|source| PlacementError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if meta.file_type().is_symlink() {
        return Err(PlacementError::Symlink {
            path: path.to_path_buf(),
        });
    }
    if !meta.file_type().is_file() {
        return Err(PlacementError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

#[cfg(unix)]
fn dev_ino(meta: &std::fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt as _;
    (meta.dev(), meta.ino())
}

/// 評価データ本体を読み取り専用配置にする（REQ-39・REQ-17・TASK-17.2-2）。
///
/// モジュール doc「手順」を参照。冪等: 既に `0o444`（unix）／読み取り専用
/// （非 unix）であるファイルに対して呼んでも成功する。
pub fn place_read_only(path: &Path) -> Result<ReadOnlyPlacement, PlacementError> {
    check_not_symlink(path)?;

    let file = open_regular_file_for_read(path)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        let pre_meta = std::fs::symlink_metadata(path).map_err(|source| PlacementError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let opened_meta = file.metadata().map_err(|source| PlacementError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if dev_ino(&pre_meta) != dev_ino(&opened_meta) {
            return Err(PlacementError::Replaced {
                path: path.to_path_buf(),
            });
        }

        const READ_ONLY_MODE: u32 = 0o444;
        file.set_permissions(std::fs::Permissions::from_mode(READ_ONLY_MODE))
            .map_err(|source| PlacementError::Io {
                path: path.to_path_buf(),
                source,
            })?;

        let confirmed = file.metadata().map_err(|source| PlacementError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mode = confirmed.permissions().mode() & 0o7777;
        if mode != READ_ONLY_MODE {
            return Err(PlacementError::Io {
                path: path.to_path_buf(),
                source: std::io::Error::other(format!(
                    "permissions did not converge to {READ_ONLY_MODE:o} (got {mode:o})"
                )),
            });
        }

        verify_direct_write_rejected(path)?;
        Ok(ReadOnlyPlacement { mode })
    }

    #[cfg(not(unix))]
    {
        // SAFETY(非 unix): 読み取り専用ハンドル経由の権限変更は Windows で
        // `FILE_WRITE_ATTRIBUTES` を要求され失敗しうるため、パス経由で行う
        // （モジュール doc「手順」参照）。事前検査（`check_not_symlink`・
        // `open_regular_file_for_read`）は既に通過済みだが、パス経由のため
        // ここでの TOCTOU 対策は unix ほど強くない（M10 時点で対象外 OS）。
        drop(file);
        let mut perms = std::fs::metadata(path)
            .map_err(|source| PlacementError::Io {
                path: path.to_path_buf(),
                source,
            })?
            .permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(path, perms).map_err(|source| PlacementError::Io {
            path: path.to_path_buf(),
            source,
        })?;

        let readonly = std::fs::metadata(path)
            .map_err(|source| PlacementError::Io {
                path: path.to_path_buf(),
                source,
            })?
            .permissions()
            .readonly();
        if !readonly {
            return Err(PlacementError::Io {
                path: path.to_path_buf(),
                source: std::io::Error::other("read-only attribute did not converge"),
            });
        }

        verify_direct_write_rejected(path)?;
        Ok(ReadOnlyPlacement { readonly })
    }
}

/// 直接の書き込み試行が権限エラーで拒否されることを、非破壊的に確認する
/// （REQ-39・TASK-17.2-2。出典: PoC-20 ケース 5）。
///
/// `OpenOptions::new().append(true)` のみを使い `create`・`truncate` は
/// 付けない。開けても 1 バイトも書かずに即座に閉じるため、呼び出し前後で
/// ファイルの内容・mtime は変わらない。
pub fn verify_direct_write_rejected(path: &Path) -> Result<(), PlacementError> {
    check_not_symlink(path)?;

    match OpenOptions::new().append(true).open(path) {
        Err(err) if err.kind() == ErrorKind::PermissionDenied => Ok(()),
        Ok(_) => Err(PlacementError::WriteNotRejected {
            path: path.to_path_buf(),
        }),
        Err(source) => Err(PlacementError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の一時ファイルを、成否に関わらず削除するガード（RAII）。
    /// 削除の前に権限を書き込み可へ戻す（読み取り専用のままだと環境に
    /// よっては削除できないため、また後始末を確実にするため）。
    struct TempFileGuard(PathBuf);

    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o644));
            }
            #[cfg(not(unix))]
            {
                if let Ok(meta) = std::fs::metadata(&self.0) {
                    let mut perms = meta.permissions();
                    #[allow(
                        clippy::permissions_set_readonly_false,
                        reason = "テスト後片付けで Windows の読み取り専用属性を解除するため"
                    )]
                    perms.set_readonly(false);
                    let _ = std::fs::set_permissions(&self.0, perms);
                }
            }
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn write_unique_temp_file(label: &str, bytes: &[u8]) -> TempFileGuard {
        let pid = std::process::id();
        for attempt in 0..1000u32 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let candidate = std::env::temp_dir().join(format!(
                "fandhe-edge-data-frozen-placement-unit-{pid}-{label}-{attempt}-{nanos}"
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(mut file) => {
                    use std::io::Write as _;
                    file.write_all(bytes).expect("書き込みに失敗しないはず");
                    return TempFileGuard(candidate);
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => panic!("一時ファイルの作成に失敗しないはず: {err}"),
            }
        }
        panic!("一意な一時ファイルを作成できなかった");
    }

    /// unix で実行ユーザーが root（uid 0）かどうかを判定する（`libc` 不要。
    /// 一時ファイルの `MetadataExt::uid()` が呼び出しユーザーの実効 uid と
    /// 一致することを利用する）。
    #[cfg(unix)]
    fn running_as_root(path: &Path) -> bool {
        use std::os::unix::fs::MetadataExt as _;
        std::fs::metadata(path)
            .map(|meta| meta.uid() == 0)
            .unwrap_or(false)
    }

    /// REQ-39・TASK-17.2-2: 非 root 環境で、読み取り専用配置後に直接の
    /// 書き込み試行（append・truncate 双方）が権限エラーで拒否される。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_rejects_direct_writes_when_not_root() {
        let guard = write_unique_temp_file("accept-basic", b"eval data");
        if running_as_root(&guard.0) {
            // root 分岐は別テスト（`req39_place_read_only_fails_closed_when_root`）で扱う。
            return;
        }

        let placement = place_read_only(&guard.0).expect("非 root では配置が成功するはず");
        assert_eq!(placement.mode(), 0o444);

        match OpenOptions::new().append(true).open(&guard.0) {
            Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
            Ok(_) => panic!("append open は拒否されるはず"),
        }
        match std::fs::write(&guard.0, b"tampered") {
            Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
            Ok(()) => panic!("truncate write は拒否されるはず"),
        }

        // 内容が変わっていないこと（読み取りは読み取り専用でも成功する）。
        let content = std::fs::read(&guard.0).expect("読み取りは成功するはず");
        assert_eq!(content, b"eval data");
    }

    /// REQ-39・TASK-17.2-2: root 実行下では mode `0o444` でも書き込めるため、
    /// `place_read_only` は `WriteNotRejected` で fail-closed に失敗する
    /// （モジュール doc「root・ACL の扱い」）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_fails_closed_when_root() {
        let guard = write_unique_temp_file("root-branch", b"eval data");
        if !running_as_root(&guard.0) {
            // 非 root 分岐は `req39_place_read_only_rejects_direct_writes_when_not_root` で扱う。
            return;
        }

        match place_read_only(&guard.0) {
            Err(PlacementError::WriteNotRejected { .. }) => {}
            other => panic!("root では WriteNotRejected を期待したが {other:?} だった"),
        }
    }

    /// REQ-39: `place_read_only` は冪等（2 回呼んでも成功し mode が変わらない）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_is_idempotent() {
        let guard = write_unique_temp_file("idempotent", b"eval data");
        if running_as_root(&guard.0) {
            return;
        }

        let first = place_read_only(&guard.0).expect("1 回目は成功するはず");
        let second = place_read_only(&guard.0).expect("2 回目も成功するはず");
        assert_eq!(first, second);
        assert_eq!(second.mode(), 0o444);
    }

    /// REQ-39: 書き込み可能なファイルへ `verify_direct_write_rejected` を
    /// 呼ぶと、root かどうかに関わらず決定的に判定される。
    #[cfg(unix)]
    #[test]
    fn req39_verify_direct_write_rejected_on_writable_file() {
        let guard = write_unique_temp_file("writable-probe", b"eval data");
        let result = verify_direct_write_rejected(&guard.0);
        if running_as_root(&guard.0) {
            assert!(
                result.is_ok(),
                "root は書き込み可能ファイルへの append open に成功するはず"
            );
        } else {
            match result {
                Err(PlacementError::WriteNotRejected { .. }) => {}
                other => panic!("WriteNotRejected を期待したが {other:?} だった"),
            }
        }
        // 非破壊プローブ: append open が成功しても中身は変わらない。
        assert_eq!(
            std::fs::read(&guard.0).expect("読み取りに失敗しないはず"),
            b"eval data"
        );
    }

    /// REQ-39: ディレクトリを渡すと `NotRegularFile` になる。
    #[test]
    fn req39_place_read_only_rejects_directory() {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-unit-{}-dir-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir(&dir).expect("テスト用ディレクトリを作成できるはず");

        let result = place_read_only(&dir);
        std::fs::remove_dir(&dir).expect("テスト用ディレクトリを削除できるはず");

        match result.expect_err("ディレクトリは通常ファイルではないため拒否されるはず")
        {
            PlacementError::NotRegularFile { .. } => {}
            other => panic!("NotRegularFile を期待したが {other:?} だった"),
        }
    }

    /// REQ-39: 存在しないパスを渡すと `Io` エラーになる（`symlink_metadata`
    /// の失敗として検出される）。
    #[test]
    fn req39_place_read_only_reports_io_error_for_missing_path() {
        let missing =
            std::env::temp_dir().join("fandhe-edge-data-frozen-placement-does-not-exist.jsonl");
        match place_read_only(&missing) {
            Err(PlacementError::Io { .. }) => {}
            other => panic!("Io エラーを期待したが {other:?} だった"),
        }
    }

    /// REQ-39: symlink を渡すと `Symlink` として拒否され、リンク先の権限は
    /// 変更されない（`../` 等のルート外参照を防ぐガード層とは別に、本
    /// モジュール自身もリンク先への副作用を避ける）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_rejects_symlink_without_touching_target() {
        let target = write_unique_temp_file("symlink-target", b"eval data");
        let link_path = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-unit-{}-symlink-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::os::unix::fs::symlink(&target.0, &link_path).expect("symlink を作成できるはず");

        let result = place_read_only(&link_path);
        let _ = std::fs::remove_file(&link_path);

        match result.expect_err("symlink は拒否されるはず") {
            PlacementError::Symlink { .. } => {}
            other => panic!("Symlink を期待したが {other:?} だった"),
        }

        // リンク先の mode が元のまま（書き込み可）であることを確認する。
        use std::os::unix::fs::PermissionsExt as _;
        let target_mode = std::fs::metadata(&target.0)
            .expect("リンク先のメタデータを取得できるはず")
            .permissions()
            .mode()
            & 0o7777;
        assert_ne!(
            target_mode, 0o444,
            "symlink 拒否がリンク先の権限を変えてはならない"
        );
    }

    /// REQ-39: FIFO（名前付きパイプ）を渡しても無期限に停止せず
    /// `NotRegularFile`（`open_regular_file_for_read` 経由）になる。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn req39_place_read_only_rejects_fifo_without_blocking() {
        let path = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-unit-{}-fifo-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let status = std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .expect("mkfifo コマンドを起動できるはず");
        assert!(status.success(), "mkfifo が成功するはず");

        let result = place_read_only(&path);
        std::fs::remove_file(&path).expect("FIFO を削除できるはず");

        match result.expect_err("FIFO は通常ファイルではないため拒否されるはず")
        {
            PlacementError::NotRegularFile { .. }
            | PlacementError::Fs(FsError::NotRegularFile { .. }) => {}
            other => panic!("NotRegularFile を期待したが {other:?} だった"),
        }
    }

    /// REQ-39: `Display` が英語固定で、評価データ本文を含まない。
    #[test]
    fn req39_display_is_english_and_excludes_data_body() {
        let path = PathBuf::from("/tmp/example-eval.jsonl");
        let err = PlacementError::WriteNotRejected { path: path.clone() };
        let text = err.to_string();
        assert!(text.is_ascii());
        assert!(text.contains("writable"));
    }
}
