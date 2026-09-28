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
//! - **ハードリンクされたファイルへの権限変更は拒否する**（unix。
//!   `st_nlink != 1`）。`(dev, ino)` の一致だけではハードリンクを検出
//!   できず、許可ルート内のパスがルート外ファイルへのハードリンク
//!   だった場合に `set_permissions` がそのルート外 inode の権限を
//!   変更してしまう（issue #227 codex[bot] P0 指摘）。本モジュールは
//!   経路の閉じ込め自体は行わない前提のため、リンク数を見て「単独の
//!   実体か」を確認することでこの経路を閉じる
//! - **非 unix では読み取り専用配置そのものを拒否する**
//!   （[`PlacementError::UnsupportedPlatform`]）。Windows には
//!   `(dev, ino)` 相当の安価な同一性検査手段が無く、検査用ハンドルを
//!   閉じてパス経由で権限変更する実装は検査後の差し替え（TOCTOU）を
//!   防げない（issue #227 codex[bot] P1 指摘）。ハンドルに結び付けた
//!   権限変更（Win32 API・`unsafe` FFI が必要）は M10 時点で対象外の
//!   OS 向けの実装として見送り、「実装済みを装わない」
//!   （`.claude/rules/coding-rust.md`）ため fail-closed に拒否する
//! - **CLI の出力 JSON 全体の形は決めない**。入出力契約は TASK-33.3 に委ねる
//!
//! # 手順（[`place_read_only`]）
//!
//! 1. `symlink_metadata` で symlink・非通常ファイルを拒否する（リンク先が
//!    評価ディレクトリ外かもしれないファイルの権限を、本関数の副作用で
//!    書き換えないため）
//! 2. [`fandhe_edge_core::fs::open_regular_file_for_read`] で開く（非 unix
//!    ではここで [`PlacementError::UnsupportedPlatform`] として拒否する）
//! 3. unix では、手順 1 の `(dev, ino)` と開いたハンドルのそれを突き合わせ、
//!    検査からオープンまでの差し替え（TOCTOU）を検出する。続けて
//!    ハンドルの `st_nlink` が 1 であることを確認し、ハードリンクされた
//!    ファイル（ルート外ファイルへのハードリンクかもしれない）への
//!    権限変更を拒否する
//! 4. ハンドル経由で権限を `0o444`（読み取り専用・setuid/setgid/sticky・
//!    実行ビットなし）に変更する（unix のみ。非 unix は手順 2 で既に
//!    拒否済み）
//! 5. 反映を確認する
//! 6. [`verify_direct_write_rejected`] を呼び、実際に書き込みが拒否される
//!    ことを確認できた場合に限り成功とする（fail-closed）
//!
//! # 書き込みプローブの TOCTOU 対策（Linux）
//!
//! [`verify_direct_write_rejected`] は、対象への読み取り専用ハンドルを
//! 開いて `(dev, ino)` を検査時点と突き合わせた後、その書き込みプローブ
//! （`append` での再オープン）をパス経由ではなく
//! `/proc/self/fd/<fd>` というカーネル提供の疑似シンボリックリンク経由で
//! 行う（[`reopen_append_via_fd`]。Linux 限定）。これによりディレクトリ
//! エントリの再探索が発生しないため、識別済みハンドルを得た後にパス上の
//! ファイルが別実体へ差し替えられても、プローブは常に元の実体を対象に
//! し続ける。単純にパスを再オープンして `PermissionDenied` 後に
//! `(dev, ino)` を再検査するだけの実装では、「再検査までの間にファイルが
//! 差し替えられ、たまたま同じ `(dev, ino)` に戻っていた」場合に、実際に
//! 拒否されたのが差し替え後の別ファイルだったのかを区別できない（issue
//! #227 codex[bot] P0 指摘）。macOS の `/dev/fd`（`fdescfs`）は同種の
//! 疑似シンボリックリンクに見えるが `dup()` 相当の実装であり、要求した
//! フラグに関わらず元のディスクリプタのアクセスモードを越える再オープンを
//! 拒否するため、対象ファイルの実際の権限を検査できない（issue #227
//! cursor[bot] 指摘: Write probe broken on macOS。実際に CI の
//! macOS 実行で「書き込み可能なはずのファイルへの直接書き込みが拒否と
//! 誤判定される」形で顕在化した）。そのため macOS を含む Linux 以外の
//! unix・非 unix 環境（M10 時点で macOS 以外は検証対象外）では、この fd
//! 直参照が使えないためパス再オープン＋再検査のベストエフォート対策に
//! 留める（[`verify_identity_after_permission_denied`]）。
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
    // unix のみ。非 unix では `place_read_only` が
    // `PlacementError::UnsupportedPlatform` を返すため本型は構築されない
    // （モジュール doc「責務の境界」）。
    mode: u32,
    // 配置確認時点（`place_read_only` 内の最終検査）の `(dev, ino)`。
    // 呼び出し側が「この戻り値が指す実体は今もこれか」を後から突き合わせ
    // られるようにする（issue #227 codex[bot] P0 指摘: 戻り値だけでは
    // 呼び出し元が使うパスの読み取り専用状態を保証できないため、検証手段
    // 自体を戻り値に持たせる）。関数の返り値が確定した「その瞬間」の実体を
    // 表すのみで、返り値を受け取った後の差し替え（TOCTOU）はこのフィールド
    // だけでは検出できない。その事後検知は凍結記録の sha256
    // （[`crate::eval_freeze::evaluate_gate`]・issue #49）に委ねる。
    dev_ino: (u64, u64),
}

impl ReadOnlyPlacement {
    /// 配置後の unix パーミッションビット（`mode & 0o7777`）。
    #[must_use]
    pub fn mode(&self) -> u32 {
        self.mode
    }

    /// 配置確認時点の `(dev, ino)`。呼び出し側が後から
    /// `std::fs::symlink_metadata` 等で同じ実体かどうかを突き合わせるために
    /// 公開する（issue #227 codex[bot] P0 指摘への対応）。
    #[must_use]
    pub fn dev_ino(&self) -> (u64, u64) {
        self.dev_ino
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
    /// パス先がハードリンクされている（`st_nlink != 1`。unix 限定）。
    ///
    /// `(dev, ino)` の一致だけではハードリンクを検出できず、許可ルート内の
    /// パスがルート外ファイルへのハードリンクだった場合に権限変更がその
    /// ルート外 inode に及んでしまう（issue #227 codex[bot] P0 指摘）。
    /// 単独の実体（リンク数 1）であることを確認できない限り権限変更を
    /// 拒否する（fail-closed）。
    HardLinked { path: PathBuf, nlink: u64 },
    /// 非 unix プラットフォームでは読み取り専用配置そのものを拒否する。
    ///
    /// Windows には `(dev, ino)` 相当の安価な同一性検査手段が無く、検査用
    /// ハンドルを閉じてパス経由で権限変更すると検査後の差し替え
    /// （TOCTOU）を防げない（issue #227 codex[bot] P1 指摘）。ハンドルに
    /// 結び付けた権限変更には Win32 API 呼び出し（`unsafe` FFI）が要るが
    /// M10 時点で Windows は対象外 OS のため実装せず、「実装済みを装わ
    /// ない」（`.claude/rules/coding-rust.md`）方針で fail-closed に拒否
    /// する。将来 Windows 対応する際は、ハンドルに結び付けた同一性検査
    /// 付きの実装に置き換える（TASK-17.2-2 の将来仕様）。
    UnsupportedPlatform { path: PathBuf },
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
            PlacementError::HardLinked { path, nlink } => {
                write!(
                    f,
                    "{} has {nlink} hard links, refusing to change permissions",
                    path.display()
                )
            }
            PlacementError::UnsupportedPlatform { path } => write!(
                f,
                "{} cannot be placed read-only on this platform",
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
            | PlacementError::WriteNotRejected { .. }
            | PlacementError::HardLinked { .. }
            | PlacementError::UnsupportedPlatform { .. } => None,
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
///
/// 呼び出し側が TOCTOU 検査（`open` 後の `(dev, ino)` 突き合わせ）に使える
/// よう、取得したメタデータをそのまま返す（このメタデータを捨てて後で
/// 取り直すと、検査からオープンまでの間の差し替えを見逃す。issue #227
/// レビュー指摘）。
fn check_not_symlink(path: &Path) -> Result<std::fs::Metadata, PlacementError> {
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
    Ok(meta)
}

#[cfg(unix)]
fn dev_ino(meta: &std::fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt as _;
    (meta.dev(), meta.ino())
}

/// ハードリンク数（`st_nlink`）を返す（unix 限定。[`PlacementError::HardLinked`]
/// の判定に使う）。
#[cfg(unix)]
fn nlink_count(meta: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt as _;
    meta.nlink()
}

/// 評価データ本体を読み取り専用配置にする（REQ-39・REQ-17・TASK-17.2-2）。
///
/// モジュール doc「手順」を参照。unix では冪等（既に `0o444` であるファイル
/// に対して呼んでも成功する）。非 unix では常に
/// [`PlacementError::UnsupportedPlatform`] を返す（モジュール doc
/// 「責務の境界」）。
pub fn place_read_only(path: &Path) -> Result<ReadOnlyPlacement, PlacementError> {
    let pre_meta = check_not_symlink(path)?;

    let file = open_regular_file_for_read(path)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        // `pre_meta` は `check_not_symlink` が `open` より前に取得した
        // symlink_metadata（このスコープに入る前に別のファイルへ差し替え
        // られていないかの基準）。ここで新たに `symlink_metadata` を取り
        // 直すと、その取り直し自体が新しい TOCTOU 窓になる（issue #227
        // codex P0 指摘）ため、必ず `check_not_symlink` が返したメタデータ
        // を使う。
        let opened_meta = file.metadata().map_err(|source| PlacementError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if dev_ino(&pre_meta) != dev_ino(&opened_meta) {
            return Err(PlacementError::Replaced {
                path: path.to_path_buf(),
            });
        }

        // ハードリンクされたファイル（`st_nlink != 1`）への権限変更は
        // 拒否する。`(dev, ino)` の一致検査はパスの差し替え（TOCTOU）は
        // 検出できるが、パス自体がルート外ファイルへのハードリンク
        // だった場合は検出できない（issue #227 codex[bot] P0 指摘）。
        let nlink = nlink_count(&opened_meta);
        if nlink != 1 {
            return Err(PlacementError::HardLinked {
                path: path.to_path_buf(),
                nlink,
            });
        }
        // 権限変更前の mode を控える。直後の再検査（後述）でハードリンクの
        // 発生を検出した場合、権限変更（副作用）を元に戻すために使う。
        let original_mode = opened_meta.permissions().mode() & 0o7777;

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

        // nlink の確認（手順 3）と `set_permissions`（直前）の間に、同じ
        // 所有者が許可ルート外へこの inode へのハードリンクを追加した場合、
        // 直前の権限変更はそのルート外パスにも及んでしまう（issue #227
        // codex[bot] P0 指摘）。ここで nlink を取り直し、増えていれば
        // 権限変更前の mode へ戻したうえで拒否する（検出と原状回復。
        // 「後始末なしの post-check」だけでは、ルート外 inode を 0o444 の
        // ままにして失敗を報告するだけになり、副作用が残ってしまうため）。
        // これでも「取り直し」と「戻す」の間の窓は残るが、権限変更直後に
        // 即座に取り直すことでその窓を可能な限り狭める。
        let confirmed_nlink = nlink_count(&confirmed);
        if confirmed_nlink != 1 {
            let _ = file.set_permissions(std::fs::Permissions::from_mode(original_mode));
            return Err(PlacementError::HardLinked {
                path: path.to_path_buf(),
                nlink: confirmed_nlink,
            });
        }

        let confirmed_dev_ino = dev_ino(&confirmed);

        #[cfg(target_os = "linux")]
        {
            // 既に開いている読み取り専用ハンドル `file` の fd をそのまま
            // 使い、`/proc/self/fd/<fd>` 経由で書き込みプローブを行う。
            // ここまで一度も新たにパスを開き直していないため、`file` を
            // 開いた時点（手順 2）以降、権限変更・書き込み確認のいずれも
            // パス経由のディレクトリエントリ再探索を経由しない。これにより
            // `verify_direct_write_rejected_checked` を（新たにパスを開き
            // 直す形で）呼ぶ場合に残っていた、権限変更後・書き込み確認前の
            // TOCTOU 窓そのものを構造的に閉じる（issue #227 codex[bot] P0
            // 指摘: 検査後に対象パスを差し替えても `place_read_only` が
            // 成功を返してしまう問題への対応）。
            match reopen_append_via_fd(&file) {
                Err(err) if err.kind() == ErrorKind::PermissionDenied => {}
                Ok(_opened) => {
                    return Err(PlacementError::WriteNotRejected {
                        path: path.to_path_buf(),
                    });
                }
                Err(source) => {
                    return Err(PlacementError::Io {
                        path: path.to_path_buf(),
                        source,
                    });
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            // macOS 等（Linux 以外の unix）では `/dev/fd` 等のハンドル直参照
            // 手法が使えない、または信頼できない（macOS の `/dev/fd` は
            // dup 相当で、元のハンドルのアクセスモードを越える再オープンを
            // 要求フラグに関わらず拒否するため、対象ファイルの実際の権限を
            // 検査できない。issue #227 cursor[bot] 指摘）。そのためパスを
            // 開き直すベストエフォートの確認に留める。このハンドルの
            // `(dev, ino)` を期待値として渡し、権限変更後・書き込み確認前に
            // パスが別のファイルへ差し替えられていないかも突き合わせる
            // （issue #227 codex P1 指摘）。
            verify_direct_write_rejected_checked(path, Some(confirmed_dev_ino))?;
        }

        // 関数の返り値が確定する直前に、パス上の実体が検査してきたものと
        // 今もなお同一であることを最後にもう一度確認する（issue #227
        // codex[bot] P0 指摘: 検査の「間」または「直後」に対象パスを
        // 差し替えても成功を返してしまう問題）。これは関数内で起こり得る
        // 差し替えを閉じるものであり、この関数が `Ok` を返した「後」の
        // 差し替えまでは防げない（クライアント側の検査では原理的に防げ
        // ない）。返り値を受け取った後の差し替えは、凍結記録の sha256 に
        // よる事後検知（[`crate::eval_freeze::evaluate_gate`]・issue #49）
        // に委ねる。
        let final_meta = std::fs::symlink_metadata(path).map_err(|source| PlacementError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if final_meta.file_type().is_symlink() {
            return Err(PlacementError::Symlink {
                path: path.to_path_buf(),
            });
        }
        if dev_ino(&final_meta) != confirmed_dev_ino {
            return Err(PlacementError::Replaced {
                path: path.to_path_buf(),
            });
        }

        Ok(ReadOnlyPlacement {
            mode,
            dev_ino: confirmed_dev_ino,
        })
    }

    #[cfg(not(unix))]
    {
        // 非 unix（Windows）では読み取り専用配置そのものを拒否する
        // （[`PlacementError::UnsupportedPlatform`]。モジュール doc
        // 「手順」・「責務の境界」参照）。検査用ハンドルを閉じてパス経由で
        // 権限変更する実装は、検査（`check_not_symlink`）とオープン
        // （`open_regular_file_for_read`）の後に path が symlink や
        // 別ファイルへ差し替えられても検出できない（issue #227
        // codex[bot] P1 指摘）。Windows には unix の `(dev, ino)` に相当する
        // 安価な同一性検査手段が無く、ハンドルに結び付けた権限変更には
        // `unsafe` な Win32 API 呼び出しが要るため、M10 時点で対象外の
        // Windows 向けに未検証の実装を「実装済みを装う」形で残さず、
        // fail-closed に拒否する。
        let _ = (&pre_meta, file);
        Err(PlacementError::UnsupportedPlatform {
            path: path.to_path_buf(),
        })
    }
}

/// 直接の書き込み試行が権限エラーで拒否されることを、非破壊的に確認する
/// （REQ-39・TASK-17.2-2。出典: PoC-20 ケース 5）。
///
/// `OpenOptions::new().append(true)` のみを使い `create`・`truncate` は
/// 付けない。開けても 1 バイトも書かずに即座に閉じるため、呼び出し前後で
/// ファイルの内容・mtime は変わらない。
///
/// 単独呼び出し（`expected_dev_ino` なし）でも、事前検査
/// （`check_not_symlink`）で得た `(dev, ino)` と実際に書き込みを試みた
/// ハンドルのそれを突き合わせ、検査からオープンの間の差し替えを検出する
/// （unix 限定。issue #227 cursor[bot] 指摘: TOCTOU check uses post-open
/// stat）。
pub fn verify_direct_write_rejected(path: &Path) -> Result<(), PlacementError> {
    verify_direct_write_rejected_checked(path, None)
}

/// `append` 用の `open` が `PermissionDenied` を返した直後に、検査時点
/// （[`check_not_symlink`]）の `(dev, ino)`（`pre_dev_ino`）と、現在の
/// パスの実体が同一であることを確認する（unix 限定）。
///
/// `PermissionDenied` というだけでは、検査時と同一のファイルへの拒否
/// なのか、検査後に「別の読み取り専用ファイル」へ差し替えられ、その
/// 別ファイルへの拒否を検査対象への拒否と誤認しているのかを区別
/// できない（issue #227 codex P0 指摘。fail-closed の TOCTOU 対策）。
/// `open` 呼び出し直後に改めて `symlink_metadata` を取得し `(dev, ino)`
/// が検査時と一致することを確認できて初めて、この拒否を検査対象の
/// ファイルへの拒否とみなす。これでも「失敗した `open`」と「直後の
/// 再検査」の間の窓は残る。Linux では [`reopen_append_via_fd`]
/// がこの窓自体を構造的に閉じるため、本関数は Linux 以外の環境（macOS を
/// 含む。macOS の `/dev/fd` が書き込みプローブに使えない理由は
/// [`fd_magic_path`] のドキュメント参照。M10 時点で macOS 以外は検証
/// 対象外。`.claude/rules/coding-rust.md`「クロスプラットフォーム」）
/// 向けのベストエフォートな代替としてのみ使う
/// （[`verify_direct_write_rejected_checked`] のフォールバック分岐）。
#[cfg(all(unix, not(target_os = "linux")))]
fn verify_identity_after_permission_denied(
    path: &Path,
    pre_dev_ino: (u64, u64),
) -> Result<(), PlacementError> {
    let post_meta = check_not_symlink(path)?;
    if dev_ino(&post_meta) != pre_dev_ino {
        return Err(PlacementError::Replaced {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

/// `/proc/self/fd/<fd>`（Linux 限定）を返す。パス上のディレクトリエントリを
/// 再探索せず、既に開いているファイルディスクリプタが指す実体を直接指す
/// カーネル提供の疑似シンボリックリンクである。このパスを `open` すると、
/// 対象の inode そのものを（元のパスを一切介さずに）別のアクセスモードで
/// 開き直せる。
///
/// macOS の `/dev/fd`（`fdescfs`）は同種の見た目を持つが `dup()` 相当の
/// 実装で、要求したフラグに関わらず元のディスクリプタのアクセスモードを
/// 越える再オープンを拒否する（対象ファイルの実際の権限を検査できない。
/// issue #227 cursor[bot] 指摘: Write probe broken on macOS）。そのため
/// macOS 向けの `fd_magic_path` は用意しない。macOS は
/// [`verify_identity_after_permission_denied`] によるパス再オープン＋
/// 再検査のベストエフォート経路を使う。
#[cfg(target_os = "linux")]
fn fd_magic_path(fd: std::os::unix::io::RawFd) -> PathBuf {
    PathBuf::from(format!("/proc/self/fd/{fd}"))
}

/// 事前に開いた読み取り専用ハンドル `read_handle` が指す実体そのものを、
/// [`fd_magic_path`] 経由で `append` モードとして開き直す（Linux 限定。
/// 上記 `fd_magic_path` のドキュメント参照）。パスの再探索を一切行わない
/// ため、`read_handle` を得た後にパス上のエントリが別ファイルへ
/// 差し替えられても、この再オープンは影響を受けない（issue #227 codex P0
/// 指摘: `append` の `open` が `PermissionDenied` を返した後、再検査までの
/// 間にファイルが差し替えられると、`(dev, ino)` が一致していても実際に
/// 拒否されたのが別ファイルだった可能性がある、という TOCTOU 窓を構造的に
/// 閉じる）。
#[cfg(target_os = "linux")]
fn reopen_append_via_fd(read_handle: &std::fs::File) -> std::io::Result<std::fs::File> {
    use std::os::unix::io::AsRawFd as _;
    OpenOptions::new()
        .append(true)
        .open(fd_magic_path(read_handle.as_raw_fd()))
}

/// [`verify_direct_write_rejected`] の内部実装。`expected_dev_ino` が
/// `Some` の場合（[`place_read_only`] からの呼び出し）、事前検査で得た
/// `(dev, ino)` と本関数が改めて得た `(dev, ino)` の一致も要求する。これに
/// より、権限変更後・本確認前にパスが別のファイルへ差し替えられていた
/// 場合も差し替え前のファイルと同一であることを確認できなければ
/// `Replaced` で拒否する（issue #227 codex P1 指摘）。
///
/// Linux では、書き込みプローブそのものを [`reopen_append_via_fd`]
/// で行い、対象を識別してから書き込みを試みるまでの間、パス経由の再
/// オープンを一切行わない（issue #227 codex P0 指摘の TOCTOU 対策。上記
/// ドキュメント参照）。Linux 以外の unix・非 unix 環境（macOS を含む。
/// `/dev/fd` が使えない理由は [`fd_magic_path`] のドキュメント参照。M10
/// 時点で macOS 以外は検証対象外。`.claude/rules/coding-rust.md`
/// 「クロスプラットフォーム」）では、パスを直接開いて `PermissionDenied`
/// の直後に再検査するベストエフォートの対策に留める
/// （[`verify_identity_after_permission_denied`]）。
fn verify_direct_write_rejected_checked(
    path: &Path,
    #[cfg_attr(not(unix), allow(unused_variables))] expected_dev_ino: Option<(u64, u64)>,
) -> Result<(), PlacementError> {
    let pre_meta = check_not_symlink(path)?;
    #[cfg(unix)]
    let pre_dev_ino = dev_ino(&pre_meta);
    #[cfg(not(unix))]
    let _ = &pre_meta;

    #[cfg(unix)]
    if let Some(expected) = expected_dev_ino
        && pre_dev_ino != expected
    {
        return Err(PlacementError::Replaced {
            path: path.to_path_buf(),
        });
    }

    #[cfg(target_os = "linux")]
    {
        // 読み取り専用ハンドルを開いて対象の inode に結び付ける。この
        // ハンドルの (dev, ino) が検査時と一致することを確認できて
        // 初めて、以後の書き込みプローブ（`reopen_append_via_fd`）を
        // 「検査対象そのもの」への操作とみなせる。
        let read_handle = open_regular_file_for_read(path)?;
        let opened_meta = read_handle
            .metadata()
            .map_err(|source| PlacementError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        if dev_ino(&opened_meta) != pre_dev_ino {
            return Err(PlacementError::Replaced {
                path: path.to_path_buf(),
            });
        }

        match reopen_append_via_fd(&read_handle) {
            Err(err) if err.kind() == ErrorKind::PermissionDenied => Ok(()),
            Ok(_opened) => Err(PlacementError::WriteNotRejected {
                path: path.to_path_buf(),
            }),
            Err(source) => Err(PlacementError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        match OpenOptions::new().append(true).open(path) {
            Err(err) if err.kind() == ErrorKind::PermissionDenied => {
                #[cfg(unix)]
                verify_identity_after_permission_denied(path, pre_dev_ino)?;
                Ok(())
            }
            Ok(opened) => {
                // 開けてしまった（＝書き込み可能）場合でも、それが検査時と
                // 同一のファイルであることを確認してから `WriteNotRejected`
                // として報告する。差し替え後の別ファイルが書き込み可能
                // だっただけなら `Replaced` で区別する。
                #[cfg(unix)]
                {
                    let opened_meta = opened.metadata().map_err(|source| PlacementError::Io {
                        path: path.to_path_buf(),
                        source,
                    })?;
                    if dev_ino(&opened_meta) != pre_dev_ino {
                        return Err(PlacementError::Replaced {
                            path: path.to_path_buf(),
                        });
                    }
                }
                #[cfg(not(unix))]
                {
                    drop(opened);
                }
                Err(PlacementError::WriteNotRejected {
                    path: path.to_path_buf(),
                })
            }
            Err(source) => Err(PlacementError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の一時ファイルを、成否に関わらず削除するガード（RAII）。
    /// 削除の前に権限を書き込み可へ戻す（読み取り専用のままだと環境に
    /// よっては削除できないため、また後始末を確実にするため）。
    ///
    /// unix の permission bit（`0o444`）に依存するテストからしか使わない
    /// ため `#[cfg(unix)]`。windows では未使用となり clippy `-D warnings`
    /// で fail する（issue #227 CI 指摘）。
    #[cfg(unix)]
    struct TempFileGuard(PathBuf);

    #[cfg(unix)]
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt as _;
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o644));
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[cfg(unix)]
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

    /// REQ-39: 書き込み可能なファイル（所有者に書き込み権限がある通常の
    /// mode）へ `verify_direct_write_rejected` を呼ぶと、root かどうかに
    /// 関わらず `WriteNotRejected` になる（所有者の書き込み権限は root で
    /// なくても append open を成功させるため、root 分岐で期待値を変える
    /// 理由がない。以前のテストは `result.is_ok()`〔＝書き込み拒否〕を
    /// root の「append open に成功するはず」という逆の主張の根拠にして
    /// おり期待値が反転していた。issue #227 cursor[bot] 指摘: Root test
    /// inverts write-rejection result）。
    #[cfg(unix)]
    #[test]
    fn req39_verify_direct_write_rejected_on_writable_file() {
        let guard = write_unique_temp_file("writable-probe", b"eval data");
        match verify_direct_write_rejected(&guard.0) {
            Err(PlacementError::WriteNotRejected { .. }) => {}
            other => panic!("WriteNotRejected を期待したが {other:?} だった"),
        }
        // 非破壊プローブ: append open が成功しても中身は変わらない。
        assert_eq!(
            std::fs::read(&guard.0).expect("読み取りに失敗しないはず"),
            b"eval data"
        );
    }

    /// REQ-39・TASK-17.2-2: `append` open が `PermissionDenied` を返した
    /// 場合でも、事前検査で得た `(dev, ino)`（`pre_dev_ino`）と現在のパス
    /// の実体が一致しなければ `Replaced` として拒否する（issue #227 codex
    /// P0 指摘: 書き込み拒否確認は `(dev, ino)` 未確認で `Ok` を返す）。
    ///
    /// 実際の TOCTOU（`check_not_symlink` と `open` の間の差し替え）を
    /// 再現する代わりに、`place_read_only` で読み取り専用配置済みの
    /// ファイル A の `pre_dev_ino` を握った状態で、パス上のファイルを
    /// 別の読み取り専用ファイル B へ差し替えてから呼び出すことで、
    /// 「開けなかった対象が検査時のファイルと異なる」状況を決定的に
    /// 再現する。
    ///
    /// `verify_identity_after_permission_denied` は Linux では
    /// フォールバック分岐でも使わなくなった（[`super::reopen_append_via_fd`]
    /// が TOCTOU 窓を構造的に閉じるため）。macOS は `/dev/fd` が書き込み
    /// プローブに使えない（issue #227 cursor[bot] 指摘）ためこの
    /// フォールバックを使い続ける。本関数の定義と同じ cfg でゲートし、
    /// macOS を含む Linux 以外の unix 環境向けのテストとして残す。
    #[cfg(all(unix, not(target_os = "linux")))]
    #[test]
    fn req39_verify_identity_after_permission_denied_detects_replacement() {
        use std::os::unix::fs::MetadataExt as _;

        let guard_a = write_unique_temp_file("permdenied-replace-a", b"file a");
        if running_as_root(&guard_a.0) {
            // root では append open 自体が PermissionDenied にならない。
            return;
        }
        let placement = place_read_only(&guard_a.0).expect("A の読み取り専用配置は成功するはず");
        assert_eq!(placement.mode(), 0o444);
        let pre_meta =
            std::fs::symlink_metadata(&guard_a.0).expect("A のメタデータを取得できるはず");
        let pre_dev_ino = (pre_meta.dev(), pre_meta.ino());

        // A とは別 inode の読み取り専用ファイル B を用意し、A のパスへ
        // rename で差し替える（検査からオープンの間の差し替えと同じ
        // 効果を、決定的な手順で再現する）。
        let guard_b = write_unique_temp_file("permdenied-replace-b", b"file b");
        let placement_b = place_read_only(&guard_b.0).expect("B の読み取り専用配置は成功するはず");
        assert_eq!(placement_b.mode(), 0o444);
        let b_meta = std::fs::symlink_metadata(&guard_b.0).expect("B のメタデータを取得できるはず");
        assert_ne!(
            (b_meta.dev(), b_meta.ino()),
            pre_dev_ino,
            "A と B の (dev, ino) は異なるはず"
        );
        std::fs::rename(&guard_b.0, &guard_a.0).expect("B を A のパスへ差し替えられるはず");
        // rename 済みで guard_b のパスはもう存在しないため、Drop での
        // 後始末を A 側の guard に任せる（guard_b の Drop は空振りになる）。
        std::mem::forget(guard_b);

        match verify_identity_after_permission_denied(&guard_a.0, pre_dev_ino) {
            Err(PlacementError::Replaced { .. }) => {}
            other => panic!("Replaced を期待したが {other:?} だった"),
        }
    }

    /// REQ-39・TASK-17.2-2: `reopen_append_via_fd` は、対象への読み取り専用
    /// ハンドルを取得した後にパス上のファイルが（書き込み可能な）別実体へ
    /// 差し替えられても、その差し替え後のファイルではなく、ハンドルが
    /// 指す元の実体（読み取り専用）を対象に書き込みを試みる（issue #227
    /// codex P0 指摘: `append` の `open` が `PermissionDenied` を返した後、
    /// 再検査までにファイルが差し替えられると `(dev, ino)` の再検査だけ
    /// では別ファイルへの拒否を見逃しうる、という TOCTOU 窓そのものを
    /// 構造的に閉じることを示す）。Linux 限定（macOS では
    /// `reopen_append_via_fd` 自体を用意していない。issue #227 cursor[bot]
    /// 指摘: Write probe broken on macOS）。
    #[cfg(target_os = "linux")]
    #[test]
    fn req39_reopen_append_via_fd_ignores_path_replacement_after_handle_open() {
        let guard_a = write_unique_temp_file("fdbound-replace-a", b"file a");
        if running_as_root(&guard_a.0) {
            // root では append open 自体が PermissionDenied にならない。
            return;
        }
        let placement = place_read_only(&guard_a.0).expect("A の読み取り専用配置は成功するはず");
        assert_eq!(placement.mode(), 0o444);

        // `verify_direct_write_rejected_checked` が内部で行う手順のうち、
        // 「識別済みハンドルを取得する」ところまでを模する。
        let read_handle = open_regular_file_for_read(&guard_a.0).expect("A を開けるはず");

        // ハンドル取得後にパス上のファイルを書き込み可能な B へ差し替える
        // （検査後・書き込みプローブ前の TOCTOU 窓を決定的な手順で再現する）。
        let guard_b = write_unique_temp_file("fdbound-replace-b", b"file b");
        std::fs::rename(&guard_b.0, &guard_a.0).expect("B を A のパスへ差し替えられるはず");
        std::mem::forget(guard_b);

        // fd 直参照はパスの再探索を行わないため、差し替え後の書き込み
        // 可能な B ではなく、依然として読み取り専用の A を対象にする。
        match reopen_append_via_fd(&read_handle) {
            Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
            Ok(_) => panic!("fd 直参照は差し替え後の書き込み可能な B の影響を受けないはず"),
        }
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

    /// REQ-39: ハードリンクされたファイルを渡すと `HardLinked` として
    /// 拒否され、リンク先（同一 inode を指すもう一方のパス）の権限も
    /// 変更されない（issue #227 codex[bot] P0 指摘。`(dev, ino)` の一致
    /// 検査だけではハードリンクを検出できないため、別途 `st_nlink` を
    /// 見て拒否することを確認する）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_rejects_hard_linked_file() {
        let original = write_unique_temp_file("hardlink-original", b"eval data");
        let link_path = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-unit-{}-hardlink-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::hard_link(&original.0, &link_path).expect("ハードリンクを作成できるはず");

        let result = place_read_only(&link_path);
        let _ = std::fs::remove_file(&link_path);

        match result.expect_err("ハードリンクされたファイルは拒否されるはず") {
            PlacementError::HardLinked { nlink, .. } => assert_eq!(nlink, 2),
            other => panic!("HardLinked を期待したが {other:?} だった"),
        }

        // 元ファイル（同一 inode）の mode が元のまま（書き込み可）で
        // あることを確認する（本モジュールの副作用がルート外実体へ
        // 及んでいないことの直接証拠）。
        use std::os::unix::fs::PermissionsExt as _;
        let original_mode = std::fs::metadata(&original.0)
            .expect("元ファイルのメタデータを取得できるはず")
            .permissions()
            .mode()
            & 0o7777;
        assert_ne!(
            original_mode, 0o444,
            "ハードリンク拒否が元ファイルの権限を変えてはならない"
        );
    }

    /// REQ-39: 非 unix では `place_read_only` が常に `UnsupportedPlatform`
    /// を返す（issue #227 codex[bot] P1 指摘。ハンドルに結び付けない
    /// パス経由の権限変更は TOCTOU を防げないため fail-closed に拒否する
    /// 方針。モジュール doc「責務の境界」）。
    #[cfg(not(unix))]
    #[test]
    fn req39_place_read_only_is_unsupported_on_non_unix() {
        let path = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-unit-{}-non-unix-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, b"eval data").expect("テスト用ファイルを作成できるはず");

        let result = place_read_only(&path);
        let _ = std::fs::remove_file(&path);

        match result.expect_err("非 unix では常に拒否されるはず") {
            PlacementError::UnsupportedPlatform { .. } => {}
            other => panic!("UnsupportedPlatform を期待したが {other:?} だった"),
        }
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
