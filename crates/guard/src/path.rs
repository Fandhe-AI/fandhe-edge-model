//! 経路の閉じ込め（`safe_join` 相当。REQ-39・PoC-20 ケース 1・TASK-39.4-1・#158）。
//!
//! 利用者由来のパス（`--package`・パッケージ内メタデータの `onnx_file` 等）が、指定した
//! ルート（workspace・パッケージディレクトリ）の外を指していないかを、読み込みの手前で検査する。
//! `../` による外向き相対パス・ルート外への絶対パス・symlink による外部参照を区別して拒否する。
//!
//! # 呼び出し文脈
//!
//! - CLI の `infer --package` や、パッケージ内の `onnx_file` を開く前に、後続の TASK-39.4-2
//!   （#159）が本モジュールを呼ぶ（CLI 引数への組み込みと `invalid_input` の E2E は #159 の責務）
//! - 検査の順序は「経路 → サイズ → 形式」。ファイルを読む場合は [`open_confined`] が返す
//!   [`File`] を使い、サイズ上限（TASK-39.5）・形式の検査へ渡す。[`ConfinedPath`] を
//!   `fandhe_edge_core::fs` 等でパスから開き直してはならない（検証後に親ディレクトリを
//!   symlink へ差し替えられる TOCTOU が残る）。[`safe_join`] 単体はパスの検証のみで、
//!   開く処理の安全性は保証しない
//!
//! # 判定（fail-closed）
//!
//! 1. 字句的な事前判定: 相対パスの `..` がルートより上へ戻るなら [`EscapeKind::ParentTraversal`]
//! 2. `canonicalize` で `..` と symlink を実体まで解決し、正準化したルートとの包含を
//!    成分単位（`Path::starts_with`）で判定する。文字列の前方一致は使わない
//!    （`/ws` が `/ws-evil` を受け入れる誤りを防ぐ）
//! 3. 解決できない・存在しない対象は拒否する。まだ存在しない出力先パスの検証は本モジュールの対象外
//!
//! `link/../..` のように symlink の後ろへ `..` を置く形は、字句判定側で拒否しうる。安全側の
//! 偽陽性として許容する。
//!
//! # 残る TOCTOU（[`open_confined`]）
//!
//! `openat`・`O_NOFOLLOW` によるディレクトリハンドル相対 open は新規依存（`libc` 等）が
//! 必要で未承認のため使わない。代わりに、通常ファイル検証つき（`O_NONBLOCK`・FIFO / デバイスの
//! 拒否。`fandhe_edge_core::fs`）で開いた後、開いた fd 自身の実体を検査する。
//!
//! - Linux: ルートのディレクトリハンドルを最初に開いて保持し、ルートの実パスはそのハンドル
//!   （`/proc/self/fd/<fd>`）から得る。open もハンドル起点（`/proc/self/fd/<root fd>/<相対>`）で
//!   行い、開いた fd の実パスがルート配下であることを確認する。fd に対する検査のため、
//!   検証後の差し替えでは外部ファイルを通せない
//! - macOS: std だけで fd の実パスを取れないため、ルートは 1 度だけ正準化して固定し、open 後に
//!   固定ルートで再検証して dev / inode の一致を確認する。再解決と比較の間の差し替えは検出
//!   できない残余リスクがあり、`F_GETPATH` / `openat`（`libc` 依存の承認）は依存承認後の課題とする
//! - その他の OS（Windows・他の unix 等）: 非ブロッキング open と実体確認を保証できないため
//!   [`PathRejection::UnsupportedPlatform`] で拒否する（fail-closed。M10 時点で対象外）
//!
//! 拒否結果は [`PathRejection::exit_code`] と [`PathRejection::reason_code`] で、REQ-21 の
//! 終了コード（`invalid_input`=64 等）と機械可読な理由コードへ写せる。

use std::fmt;
use std::fs::File;
use std::io;
use std::path::{Component, Path, PathBuf};

use fandhe_edge_core::exitcode::ExitCode;

/// ルート外へ出る経路の種別。拒否理由の内訳を区別するために持つ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EscapeKind {
    /// 相対パスの `..` がルートより上へ戻る。
    ParentTraversal,
    /// ルート外を指す絶対パス。
    Absolute,
    /// 字句的にはルート配下だが、symlink の解決先がルート外。
    Symlink,
}

impl EscapeKind {
    /// 機械可読な識別子（英語の snake_case）。
    pub const fn name(&self) -> &'static str {
        match self {
            EscapeKind::ParentTraversal => "parent_traversal",
            EscapeKind::Absolute => "absolute",
            EscapeKind::Symlink => "symlink",
        }
    }
}

/// 経路検証の拒否理由。メッセージは英語で、正準化したルートの絶対パスは含めない
/// （ホームディレクトリ等が CLI の JSON へ漏れるのを避ける）。
#[derive(Debug)]
#[non_exhaustive]
pub enum PathRejection {
    /// candidate が空。
    EmptyPath,
    /// ルートが存在しない、または解決できない。
    RootUnresolvable { source: io::Error },
    /// ルートがディレクトリでない。
    RootNotDirectory,
    /// candidate がルートの外を指す。
    Escapes {
        candidate: PathBuf,
        kind: EscapeKind,
    },
    /// candidate が存在しない・解決できない。
    Unresolvable {
        candidate: PathBuf,
        source: io::Error,
    },
    /// 通常ファイルではない（FIFO・デバイス・ディレクトリ等。読み込みの無期限停止を避ける）。
    NotRegularFile { candidate: PathBuf },
    /// 開いたファイルの実体を検証できない OS のため拒否する（fail-closed）。
    UnsupportedPlatform,
}

/// 権限・資源・中断など環境起因の失敗だけを実行時エラー（70）とし、それ以外は
/// 利用者が渡したパスの不正（存在しない・ファイル配下を辿る・symlink ループ・
/// 名前の長すぎ等。`NotFound`・`NotADirectory`・`FilesystemLoop`・`InvalidFilename`・
/// `InvalidInput` ほか）として入力不正（64）に写す（REQ-21）。
fn io_exit_code(source: &io::Error) -> ExitCode {
    match source.kind() {
        io::ErrorKind::PermissionDenied
        | io::ErrorKind::OutOfMemory
        | io::ErrorKind::TimedOut
        | io::ErrorKind::Interrupted => ExitCode::RuntimeError,
        _ => ExitCode::InvalidInput,
    }
}

impl PathRejection {
    /// 終了コード（REQ-21）。閉じ込め違反・空パス・存在しない対象は `InvalidInput`、
    /// 権限・資源等の環境起因の I/O 失敗のみ `RuntimeError`（他の I/O 失敗は入力不正）。
    pub fn exit_code(&self) -> ExitCode {
        match self {
            PathRejection::EmptyPath
            | PathRejection::RootNotDirectory
            | PathRejection::NotRegularFile { .. }
            | PathRejection::Escapes { .. } => ExitCode::InvalidInput,
            PathRejection::UnsupportedPlatform => ExitCode::RuntimeError,
            PathRejection::RootUnresolvable { source }
            | PathRejection::Unresolvable { source, .. } => io_exit_code(source),
        }
    }

    /// 機械可読な理由コード（英語の snake_case）。
    pub const fn reason_code(&self) -> &'static str {
        match self {
            PathRejection::EmptyPath => "empty_path",
            PathRejection::RootUnresolvable { .. } => "root_unresolvable",
            PathRejection::RootNotDirectory => "root_not_directory",
            PathRejection::Escapes { .. } => "path_escapes_root",
            PathRejection::Unresolvable { .. } => "path_unresolvable",
            PathRejection::NotRegularFile { .. } => "not_regular_file",
            PathRejection::UnsupportedPlatform => "unsupported_platform",
        }
    }
}

impl fmt::Display for PathRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathRejection::EmptyPath => write!(f, "path is empty"),
            PathRejection::RootUnresolvable { source } => {
                write!(f, "root directory cannot be resolved: {source}")
            }
            PathRejection::RootNotDirectory => write!(f, "root is not a directory"),
            PathRejection::Escapes { candidate, kind } => write!(
                f,
                "path escapes the root ({}): {}",
                kind.name(),
                candidate.display()
            ),
            PathRejection::Unresolvable { candidate, source } => {
                write!(
                    f,
                    "path cannot be resolved: {}: {source}",
                    candidate.display()
                )
            }
            PathRejection::NotRegularFile { candidate } => {
                write!(f, "path is not a regular file: {}", candidate.display())
            }
            PathRejection::UnsupportedPlatform => {
                write!(f, "confined open is not supported on this platform")
            }
        }
    }
}

impl std::error::Error for PathRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PathRejection::RootUnresolvable { source }
            | PathRejection::Unresolvable { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// [`safe_join`] を通過した、正準化済みでルート配下にあるパス。
///
/// [`safe_join`] だけが生成できる。後続の読み込み処理が「検証済みのパス」を型で要求できる。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfinedPath(PathBuf);

impl ConfinedPath {
    /// 正準化済みのパスを借用する。
    pub fn as_path(&self) -> &Path {
        &self.0
    }

    /// 正準化済みのパスを取り出す。
    pub fn into_path_buf(self) -> PathBuf {
        self.0
    }
}

/// 絶対パス（ルート・ドライブ接頭辞を持つ）か。
fn is_absolute_like(p: &Path) -> bool {
    p.components()
        .any(|c| matches!(c, Component::RootDir | Component::Prefix(_)))
}

/// 相対パスの `..` がルートより上へ戻るか（ファイルシステムに触れない字句判定）。
fn lexically_escapes(p: &Path) -> bool {
    let mut depth: usize = 0;
    for c in p.components() {
        match c {
            Component::ParentDir => match depth.checked_sub(1) {
                Some(d) => depth = d,
                None => return true,
            },
            Component::Normal(_) => depth = depth.saturating_add(1),
            _ => {}
        }
    }
    false
}

/// `candidate` を `root` 配下へ閉じ込めて検証し、正準化済みのパスを返す。
///
/// - `root`: 閉じ込めの基準ディレクトリ（存在するディレクトリであること）
/// - `candidate`: 利用者由来のパス。相対・絶対のどちらも受け付け、存在する対象だけが通る
///
/// ルート自身（`.`）は許可する。文字列の前方一致ではなく成分単位で包含を判定する。
pub fn safe_join(root: &Path, candidate: &Path) -> Result<ConfinedPath, PathRejection> {
    let canon_root =
        std::fs::canonicalize(root).map_err(|source| PathRejection::RootUnresolvable { source })?;
    resolve_under(&canon_root, root, candidate)
}

/// 確定済みの正準化ルート `canon_root` に対して candidate を検証する内部関数。
///
/// [`open_confined`] が「最初に確定したルート」を使い回せるよう、ルートの正準化を呼び出し側へ
/// 出している（検証と open の間にルートを再解決しない）。`given_root` は拒否種別
/// （Absolute / Symlink）の判別にだけ使う。
fn resolve_under(
    canon_root: &Path,
    given_root: &Path,
    candidate: &Path,
) -> Result<ConfinedPath, PathRejection> {
    if candidate.as_os_str().is_empty() {
        return Err(PathRejection::EmptyPath);
    }
    if !canon_root.is_dir() {
        return Err(PathRejection::RootNotDirectory);
    }

    let absolute = is_absolute_like(candidate);
    if !absolute && lexically_escapes(candidate) {
        return Err(PathRejection::Escapes {
            candidate: candidate.to_path_buf(),
            kind: EscapeKind::ParentTraversal,
        });
    }

    // 絶対パスの candidate は `join` により candidate 自身に置き換わる。
    let resolved = std::fs::canonicalize(canon_root.join(candidate)).map_err(|source| {
        PathRejection::Unresolvable {
            candidate: candidate.to_path_buf(),
            source,
        }
    })?;

    if resolved.starts_with(canon_root) {
        Ok(ConfinedPath(resolved))
    } else {
        // 絶対パスでも、字句的にルート配下（与えられたルートまたは正準化後のルートの下）から
        // symlink で外へ出る場合は Symlink とする。字句的にも外なら Absolute。
        let lexically_under_root =
            candidate.starts_with(canon_root) || candidate.starts_with(given_root);
        Err(PathRejection::Escapes {
            candidate: candidate.to_path_buf(),
            kind: if absolute && !lexically_under_root {
                EscapeKind::Absolute
            } else {
                EscapeKind::Symlink
            },
        })
    }
}

/// 検証と open を一体で行い、開いたファイル自身がルート配下の実体であることを確認して返す。
///
/// [`safe_join`] 単体では検証から open までの間に親ディレクトリを symlink へ差し替えられる
/// （TOCTOU）。本関数は (1) ルートを 1 度だけ確定して保持、(2) そのルートに対し検証、
/// (3) 通常ファイル検証つき（FIFO・デバイスを `O_NONBLOCK` と種別確認で拒否）で open、
/// (4) 開いた fd の実体を検査する。検査方法と OS ごとの残余リスクはモジュールドキュメントの
/// 「残る TOCTOU」を参照。呼び出し側は返した [`File`] だけを読み、パスを再度 open しないこと。
///
/// Linux・macOS 以外では [`PathRejection::UnsupportedPlatform`] で拒否する（fail-closed。
/// 非ブロッキング open と fd の実体検査を保証できる OS に限るため。他の unix を含む）。
pub fn open_confined(root: &Path, candidate: &Path) -> Result<(File, ConfinedPath), PathRejection> {
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (root, candidate);
        Err(PathRejection::UnsupportedPlatform)
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        open_confined_impl(root, candidate)
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn open_regular(path: &Path, candidate: &Path) -> Result<File, PathRejection> {
    fandhe_edge_core::fs::open_regular_file_for_read(path).map_err(|e| match e {
        fandhe_edge_core::fs::FsError::NotRegularFile { .. } => PathRejection::NotRegularFile {
            candidate: candidate.to_path_buf(),
        },
        fandhe_edge_core::fs::FsError::Read { source, .. } => PathRejection::Unresolvable {
            candidate: candidate.to_path_buf(),
            source,
        },
        // FsError は non_exhaustive。未知の失敗は fail-closed で入力不正として扱う。
        _ => PathRejection::Unresolvable {
            candidate: candidate.to_path_buf(),
            source: io::Error::from(io::ErrorKind::InvalidInput),
        },
    })
}

/// Linux: ルートのディレクトリハンドルを先に開いて保持し、以降はハンドルを起点にする。
///
/// ルートの実パスは保持した fd（`/proc/self/fd/<fd>`）から得るため、検証後にルートの
/// パス名を symlink へ差し替えても参照先は変わらない。open も `/proc/self/fd/<root fd>/<相対>`
/// でハンドル起点にし、最後に開いた fd 自身の実パスがルート配下であることを確認する。
#[cfg(target_os = "linux")]
fn open_confined_impl(
    root: &Path,
    candidate: &Path,
) -> Result<(File, ConfinedPath), PathRejection> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;

    // ルートが FIFO 等のブロックする特殊ファイルでも open で停止しないよう `O_NONBLOCK` で開き、
    // ディレクトリか否かは開いた fd の metadata で確認する（REQ-39。値は Linux 全アーキテクチャ共通）。
    const O_NONBLOCK: i32 = 0o4000;
    let root_handle = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NONBLOCK)
        .open(root)
        .map_err(|source| PathRejection::RootUnresolvable { source })?;
    let root_meta = root_handle
        .metadata()
        .map_err(|source| PathRejection::RootUnresolvable { source })?;
    if !root_meta.is_dir() {
        return Err(PathRejection::RootNotDirectory);
    }
    let root_link = PathBuf::from(format!("/proc/self/fd/{}", root_handle.as_raw_fd()));
    let canon_root = std::fs::read_link(&root_link)
        .map_err(|source| PathRejection::RootUnresolvable { source })?;

    let first = resolve_under(&canon_root, root, candidate)?;
    let relative =
        first
            .as_path()
            .strip_prefix(&canon_root)
            .map_err(|_| PathRejection::Escapes {
                candidate: candidate.to_path_buf(),
                kind: EscapeKind::Symlink,
            })?;
    let file = open_regular(&root_link.join(relative), candidate)?;

    let fd_link = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
    let real = std::fs::read_link(&fd_link).map_err(|source| PathRejection::Unresolvable {
        candidate: candidate.to_path_buf(),
        source,
    })?;
    // ルートは保持したハンドルから再取得する（パス名の再解決を挟まない）。
    let canon_root_now = std::fs::read_link(&root_link)
        .map_err(|source| PathRejection::RootUnresolvable { source })?;
    if !real.starts_with(&canon_root_now) {
        return Err(PathRejection::Escapes {
            candidate: candidate.to_path_buf(),
            kind: EscapeKind::Symlink,
        });
    }
    Ok((file, ConfinedPath(real)))
}

/// macOS: ルートを 1 度だけ正準化して固定し、open 後に固定ルートで再検証して dev / inode を比較する。
///
/// 開いた fd と再解決先の dev / inode が一致する場合のみ通すため、通過するのは「ルート配下に
/// 実在するファイルと同一 inode」に限られる（差し替え中に外部ファイルを開いても、同一 inode
/// でない限り拒否される）。std だけでは fd の実パスを取れないため、再解決と比較の間の差し替えは
/// 検出できない残余リスクがある（`F_GETPATH` / `openat` は `libc` 依存の承認後の課題）。
#[cfg(target_os = "macos")]
fn open_confined_impl(
    root: &Path,
    candidate: &Path,
) -> Result<(File, ConfinedPath), PathRejection> {
    use std::os::unix::fs::MetadataExt;

    let canon_root =
        std::fs::canonicalize(root).map_err(|source| PathRejection::RootUnresolvable { source })?;
    let first = resolve_under(&canon_root, root, candidate)?;
    let file = open_regular(first.as_path(), candidate)?;
    let unresolvable = |source| PathRejection::Unresolvable {
        candidate: candidate.to_path_buf(),
        source,
    };
    let second = resolve_under(&canon_root, root, candidate)?;
    let opened = file.metadata().map_err(unresolvable)?;
    let resolved = std::fs::metadata(second.as_path()).map_err(unresolvable)?;
    if opened.dev() != resolved.dev() || opened.ino() != resolved.ino() {
        return Err(PathRejection::Escapes {
            candidate: candidate.to_path_buf(),
            kind: EscapeKind::Symlink,
        });
    }
    Ok((file, second))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexical_normalization_cases() {
        assert!(!lexically_escapes(Path::new("a/./b")));
        assert!(!lexically_escapes(Path::new("a/../b")));
        assert!(!lexically_escapes(Path::new("a/b/../..")));
        assert!(lexically_escapes(Path::new("..")));
        assert!(lexically_escapes(Path::new("a/../../b")));
    }

    // libc 依存を増やさないため Linux / macOS の errno 値を直接持つ。
    fn libc_enotdir() -> i32 {
        20
    }
    fn libc_eloop() -> i32 {
        if cfg!(target_os = "macos") { 62 } else { 40 }
    }
    fn libc_enametoolong() -> i32 {
        if cfg!(target_os = "macos") { 63 } else { 36 }
    }

    #[test]
    fn reason_and_exit_codes_are_concrete() {
        let esc = PathRejection::Escapes {
            candidate: PathBuf::from("../x"),
            kind: EscapeKind::ParentTraversal,
        };
        assert_eq!(esc.exit_code(), ExitCode::InvalidInput);
        assert_eq!(esc.reason_code(), "path_escapes_root");
        assert_eq!(PathRejection::EmptyPath.reason_code(), "empty_path");
        assert_eq!(
            PathRejection::RootNotDirectory.reason_code(),
            "root_not_directory"
        );
        let nrf = PathRejection::NotRegularFile {
            candidate: PathBuf::from("x"),
        };
        assert_eq!(nrf.reason_code(), "not_regular_file");
        assert_eq!(nrf.exit_code(), ExitCode::InvalidInput);
        assert_eq!(
            PathRejection::UnsupportedPlatform.reason_code(),
            "unsupported_platform"
        );
        assert_eq!(
            PathRejection::UnsupportedPlatform.exit_code(),
            ExitCode::RuntimeError
        );
        let denied = PathRejection::Unresolvable {
            candidate: PathBuf::from("x"),
            source: io::Error::from(io::ErrorKind::PermissionDenied),
        };
        assert_eq!(denied.exit_code(), ExitCode::RuntimeError);
        assert_eq!(denied.reason_code(), "path_unresolvable");
        // 利用者入力起因の I/O エラー（ENOTDIR・ELOOP・ENAMETOOLONG 相当）は 64。
        for code in [libc_enotdir(), libc_eloop(), libc_enametoolong()] {
            let r = PathRejection::Unresolvable {
                candidate: PathBuf::from("x"),
                source: io::Error::from_raw_os_error(code),
            };
            assert_eq!(r.exit_code(), ExitCode::InvalidInput, "os error {code}");
        }
        assert_eq!(EscapeKind::Symlink.name(), "symlink");
        assert_eq!(EscapeKind::Absolute.name(), "absolute");
    }

    #[test]
    fn display_omits_root_and_is_english() {
        let esc = PathRejection::Escapes {
            candidate: PathBuf::from("../x"),
            kind: EscapeKind::ParentTraversal,
        };
        assert_eq!(
            esc.to_string(),
            "path escapes the root (parent_traversal): ../x"
        );
    }
}
