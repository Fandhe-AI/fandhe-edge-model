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
//! - Linux: `/proc/self/fd/<fd>` が指す実パスがルート配下であることを確認する。fd に対する
//!   検査のため、検証後の差し替えでは外部ファイルを通せない
//! - macOS: std だけで fd の実パスを取れないため、open 後にパスを再解決し dev / inode の一致を
//!   確認する。再解決と比較の間の差し替えは検出できない残余リスクがあり、`F_GETPATH`
//!   （`libc` 依存の承認）または `openat` 方式は依存承認後の課題とする
//! - その他の OS（Windows 等）: 実体を確認する手段がないため [`PathRejection::UnsupportedPlatform`]
//!   で拒否する（fail-closed。M10 時点で対象外）
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
    if candidate.as_os_str().is_empty() {
        return Err(PathRejection::EmptyPath);
    }
    let canon_root =
        std::fs::canonicalize(root).map_err(|source| PathRejection::RootUnresolvable { source })?;
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

    if resolved.starts_with(&canon_root) {
        Ok(ConfinedPath(resolved))
    } else {
        // 絶対パスでも、字句的にルート配下（与えられたルートまたは正準化後のルートの下）から
        // symlink で外へ出る場合は Symlink とする。字句的にも外なら Absolute。
        let lexically_under_root =
            candidate.starts_with(&canon_root) || candidate.starts_with(root);
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
/// （TOCTOU）。本関数は (1) [`safe_join`] で検証、(2) 通常ファイル検証つき（FIFO・デバイスを
/// `O_NONBLOCK` と種別確認で拒否）で open、(3) 開いた fd の実体を検査する。検査方法と
/// OS ごとの残余リスクはモジュールドキュメントの「残る TOCTOU」を参照。呼び出し側は返した
/// [`File`] だけを読み、パスを再度 open しないこと。
///
/// unix 以外では [`PathRejection::UnsupportedPlatform`] で拒否する（fail-closed）。
pub fn open_confined(root: &Path, candidate: &Path) -> Result<(File, ConfinedPath), PathRejection> {
    #[cfg(not(unix))]
    {
        let _ = (root, candidate);
        Err(PathRejection::UnsupportedPlatform)
    }
    #[cfg(unix)]
    {
        open_confined_unix(root, candidate)
    }
}

#[cfg(unix)]
fn open_confined_unix(
    root: &Path,
    candidate: &Path,
) -> Result<(File, ConfinedPath), PathRejection> {
    let first = safe_join(root, candidate)?;
    let file =
        fandhe_edge_core::fs::open_regular_file_for_read(first.as_path()).map_err(|e| match e {
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
        })?;
    let escaped = || PathRejection::Escapes {
        candidate: candidate.to_path_buf(),
        kind: EscapeKind::Symlink,
    };
    let canon_root =
        std::fs::canonicalize(root).map_err(|source| PathRejection::RootUnresolvable { source })?;

    // Linux: 開いた fd 自身の実パスを検査する（パスの再解決を挟まない）。
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let fd_link = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
        let real = std::fs::read_link(&fd_link).map_err(|source| PathRejection::Unresolvable {
            candidate: candidate.to_path_buf(),
            source,
        })?;
        if !real.starts_with(&canon_root) {
            return Err(escaped());
        }
        Ok((file, ConfinedPath(real)))
    }

    // その他の unix（macOS）: open 後に再解決し、dev / inode の一致を確認する（残余リスクあり）。
    #[cfg(not(target_os = "linux"))]
    {
        use std::os::unix::fs::MetadataExt;
        let second = safe_join(root, candidate)?;
        let opened = file
            .metadata()
            .map_err(|source| PathRejection::Unresolvable {
                candidate: candidate.to_path_buf(),
                source,
            })?;
        let resolved =
            std::fs::metadata(second.as_path()).map_err(|source| PathRejection::Unresolvable {
                candidate: candidate.to_path_buf(),
                source,
            })?;
        if opened.dev() != resolved.dev() || opened.ino() != resolved.ino() {
            return Err(escaped());
        }
        Ok((file, second))
    }
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
