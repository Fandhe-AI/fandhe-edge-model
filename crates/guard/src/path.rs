//! 経路の閉じ込め（`safe_join` 相当。REQ-39・PoC-20 ケース 1・TASK-39.4-1・#158）。
//!
//! 利用者由来のパス（`--package`・パッケージ内メタデータの `onnx_file` 等）が、指定した
//! ルート（workspace・パッケージディレクトリ）の外を指していないかを、読み込みの手前で検査する。
//! `../` による外向き相対パス・ルート外への絶対パス・symlink による外部参照を区別して拒否する。
//!
//! # 呼び出し文脈
//!
//! - CLI の `infer --package` や、パッケージ内の `onnx_file` を開く前に、TASK-39.4-2
//!   （#159）が [`crate::package`] 経由で本モジュールを呼ぶ（CLI 側の組み込みは
//!   `fandhe_edge_cli::infer_guard`）
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
//! # TOCTOU への対処（[`open_confined`]）
//!
//! パスを繰り返し解決して同一性を推定する方式は使わない。ルートのディレクトリ fd を最初に
//! 開いて保持し、そこを起点に各成分を `openat`（中間は `O_DIRECTORY | O_NOFOLLOW`、最後は
//! `O_NOFOLLOW`）で開く。検証後に親ディレクトリや対象が symlink へ差し替えられても
//! `O_NOFOLLOW` により開けず、外部ファイルは通らない。`openat` は `rustix`
//! （2026-09-29 オーナー承認。`unsafe` 不要）を使う。開いた fd は `fstat` で通常ファイルか
//! 確認する（`O_NONBLOCK` で FIFO・デバイスの open が停止しない）。加えて fd の実パス
//! （Linux: `/proc/self/fd`・macOS: `F_GETPATH`）が最初に確定したルートの実パス配下であることを
//! 確認する（検証後にルート自体が外へ移動されても拒否する。実パスを得られなければ拒否）。
//!
//! - Linux・macOS 以外（Windows・他の unix 等）は [`PathRejection::UnsupportedPlatform`] で
//!   拒否する（fail-closed。M10 時点で対象外）
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
    /// ディレクトリであるべき対象（パッケージディレクトリ）がディレクトリでない
    /// （`package::confine_package`。TASK-39.4-2・#159）。
    NotDirectory { candidate: PathBuf },
}

/// 環境・資源起因の errno（Linux / macOS の値。生の値で判定し、追加の依存を持たない）。
///
/// EPERM・EACCES（権限）・EIO・ENOMEM・ENFILE・EMFILE・ENOSPC・EINTR・EAGAIN・ETIMEDOUT・EDQUOT。
/// 番号が OS で異なるもの（EAGAIN・ETIMEDOUT・EDQUOT）は `cfg!(target_os = "macos")` で切り替える。
fn is_environmental_errno(code: i32) -> bool {
    let mac = cfg!(target_os = "macos");
    let eagain = if mac { 35 } else { 11 };
    let etimedout = if mac { 60 } else { 110 };
    let edquot = if mac { 69 } else { 122 };
    matches!(code, 1 | 4 | 5 | 12 | 13 | 23 | 24 | 28)
        || [eagain, etimedout, edquot].contains(&code)
}

/// 権限・資源（fd・メモリ・ディスク枯渇）・中断など環境起因の失敗だけを実行時エラー（70）とし、
/// それ以外は利用者が渡したパスの不正（存在しない・ファイル配下を辿る・symlink ループ・
/// 名前の長すぎ等。`NotFound`・`NotADirectory`・`FilesystemLoop`・`InvalidFilename`・
/// `InvalidInput` ほか）として入力不正（64）に写す（REQ-21）。
fn io_exit_code(source: &io::Error) -> ExitCode {
    if source.raw_os_error().is_some_and(is_environmental_errno) {
        return ExitCode::RuntimeError;
    }
    match source.kind() {
        io::ErrorKind::PermissionDenied
        | io::ErrorKind::OutOfMemory
        | io::ErrorKind::TimedOut
        | io::ErrorKind::WouldBlock
        | io::ErrorKind::StorageFull
        | io::ErrorKind::QuotaExceeded
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
            | PathRejection::NotDirectory { .. }
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
            PathRejection::NotDirectory { .. } => "not_directory",
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
            PathRejection::NotDirectory { candidate } => {
                write!(f, "path is not a directory: {}", candidate.display())
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
    /// 保持 fd の実パス確認を済ませたパスから作る（同一 crate 内の fd 起点の開き直し専用）。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub(crate) fn from_verified(path: PathBuf) -> Self {
        Self(path)
    }

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

/// `.` と `..` をファイルシステムに触れず字句的に畳む（`..` が戻れない先頭ではそのまま保持する）。
fn lexical_normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                // 末尾が通常の成分のときだけ打ち消す。ルート直下の `..` はルートに留まり（POSIX）、
                // 相対パスの先頭や `..` の直後では `..` を保持する（連続する `..` を潰さない）。
                let ends_with_normal =
                    matches!(out.components().next_back(), Some(Component::Normal(_)));
                if ends_with_normal {
                    out.pop();
                } else if !out.has_root() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
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
        // `Path::starts_with` は `..` を畳まないため、字句正規化してから判定する
        // （`/root/../etc` をルート配下と誤認して Symlink にしない）。
        let normalized = lexical_normalize(candidate);
        let lexically_under_root =
            normalized.starts_with(canon_root) || normalized.starts_with(given_root);
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
/// (3) ルートの fd を起点に `openat`（`O_NOFOLLOW`）で成分ごとに開き、(4) 開いた fd を `fstat` して
/// 通常ファイルか確認する。詳細はモジュールドキュメントの「TOCTOU への対処」を参照。
/// 呼び出し側は返した [`File`] だけを読み、パスを再度 open しないこと。
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

/// 検証後・openat 前の差し替えを決定的に再現するテスト用フック（`cfg(test)` のみ）。
#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod test_hooks {
    use std::cell::RefCell;

    type Hook = Box<dyn FnOnce()>;

    thread_local! {
        static AFTER_VALIDATION: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    /// 次の `open_confined`（同一スレッド）の検証後に一度だけ実行するフックを登録する。
    pub(super) fn set_after_validation(hook: Hook) {
        AFTER_VALIDATION.with(|h| *h.borrow_mut() = Some(hook));
    }

    pub(super) fn run_after_validation() {
        if let Some(hook) = AFTER_VALIDATION.with(|h| h.borrow_mut().take()) {
            hook();
        }
    }
}

/// rustix の errno を `io::Error` へ写す（`std` feature を使わず生の errno 値で変換する）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn errno_to_io(e: rustix::io::Errno) -> io::Error {
    io::Error::from_raw_os_error(e.raw_os_error())
}

/// ディレクトリ fd を起点に成分ごとに開いて閉じ込めを強制する（Linux・macOS）。
///
/// 1. ルートを 1 度だけ `O_DIRECTORY` で開いて fd を保持する。以降のパス解決はこの fd が起点で、
///    ルートのパス名を差し替えても参照先は変わらない（`O_NONBLOCK` は FIFO 等での停止を避ける
///    ためで、`O_DIRECTORY` により FIFO は `ENOTDIR` で拒否される）
/// 2. ルートの実パスを fd から得る（Linux: `/proc/self/fd`・macOS: `F_GETPATH`）
/// 3. [`safe_join`] 相当の検証（字句判定・正準化・包含判定。拒否種別の判別を含む）で、ルートからの
///    相対の `Normal` 成分列を得る。正準化済みのため成分に symlink は含まれない
/// 4. 各成分を `openat` で開く。中間は `O_DIRECTORY | O_NOFOLLOW`、最後は `O_NOFOLLOW`。検証後に
///    成分が symlink へ差し替えられていれば `ELOOP` / `ENOTDIR` となり、外部へは出られない
///    （パスの再解決を行わない。TASK-39.4-1・#158 の再照合競合の指摘への対処）
/// 5. 開いた fd を `fstat` して通常ファイルであることを確認し、fd の実パスが最初に確定したルートの
///    実パス配下であることも確認する（ルート自体の移動対策。両 OS）
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn open_confined_impl(
    root: &Path,
    candidate: &Path,
) -> Result<(File, ConfinedPath), PathRejection> {
    use rustix::fs::{Mode, OFlags};
    use rustix::io::Errno;

    let root_fd = rustix::fs::open(
        root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|e| match e {
        Errno::NOTDIR => PathRejection::RootNotDirectory,
        other => PathRejection::RootUnresolvable {
            source: errno_to_io(other),
        },
    })?;
    open_confined_from_fd(&root_fd, root, candidate)
}

/// 保持済みのルートディレクトリ fd を起点に、[`open_confined_impl`] の手順 2〜5 を行う。
///
/// `given_root` は拒否種別の判別にだけ使う。ルートの実パスは fd から得る（パス名を再解決しない）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn open_confined_from_fd(
    root_fd: &rustix::fd::OwnedFd,
    given_root: &Path,
    candidate: &Path,
) -> Result<(File, ConfinedPath), PathRejection> {
    use rustix::fs::{Mode, OFlags, fstat, openat};
    use rustix::io::Errno;

    let escapes = || PathRejection::Escapes {
        candidate: candidate.to_path_buf(),
        kind: EscapeKind::Symlink,
    };
    let unresolvable = |e: Errno| PathRejection::Unresolvable {
        candidate: candidate.to_path_buf(),
        source: errno_to_io(e),
    };
    let root = given_root;
    let canon_root =
        fd_real_path(root_fd).map_err(|source| PathRejection::RootUnresolvable { source })?;

    let first = resolve_under(&canon_root, root, candidate)?;
    let relative = first
        .as_path()
        .strip_prefix(&canon_root)
        .map_err(|_| escapes())?;
    let mut names = Vec::new();
    for c in relative.components() {
        match c {
            Component::Normal(n) => names.push(n),
            // 正準化済みなら現れない。現れたら fail-closed。
            _ => return Err(escapes()),
        }
    }
    let Some((last, parents)) = names.split_last() else {
        // ルート自身はファイルではない。
        return Err(PathRejection::NotRegularFile {
            candidate: candidate.to_path_buf(),
        });
    };

    // テスト時だけ、検証（正準化）の後・openat の前に差し替えを挟む（公開 API には現れない）。
    #[cfg(test)]
    test_hooks::run_after_validation();

    let mut owned: Option<rustix::fd::OwnedFd> = None;
    for name in parents {
        let cur = owned.as_ref().unwrap_or(root_fd);
        let next = openat(
            cur,
            *name,
            OFlags::RDONLY
                | OFlags::DIRECTORY
                | OFlags::NOFOLLOW
                | OFlags::NONBLOCK
                | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|e| match e {
            // 検証後に symlink 等へ差し替えられた（symlink は Linux で ENOTDIR・macOS で ELOOP）。
            Errno::LOOP | Errno::NOTDIR => escapes(),
            other => unresolvable(other),
        })?;
        owned = Some(next);
    }
    let dir = owned.as_ref().unwrap_or(root_fd);
    let fd = openat(
        dir,
        *last,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|e| match e {
        Errno::LOOP => escapes(),
        other => unresolvable(other),
    })?;

    let stat = fstat(&fd).map_err(unresolvable)?;
    if rustix::fs::FileType::from_raw_mode(stat.st_mode) != rustix::fs::FileType::RegularFile {
        return Err(PathRejection::NotRegularFile {
            candidate: candidate.to_path_buf(),
        });
    }
    // 返すパスは、検証時の `first` ではなく開いた fd 自身の実パス（openat までの間に通常ファイルが
    // 別のファイルへ差し替えられても、File と ConfinedPath が別の対象を指さない）。
    let real = ensure_real_path_under(&fd, &canon_root, candidate)?;
    Ok((File::from(fd), ConfinedPath(real)))
}

/// 検証時に開いたまま保持するディレクトリ fd（Linux・macOS）。
///
/// パッケージのように「検証後に同じディレクトリ配下のメンバーを複数回開く」用途で、ディレクトリを
/// パスから開き直さない（差し替え競合を塞ぐ）。[`open_dir_confined`] だけが生成する。
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[derive(Debug)]
pub struct ConfinedDir {
    fd: rustix::fd::OwnedFd,
    real: PathBuf,
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl ConfinedDir {
    /// 開いた時点の実パス（表示・診断用）。
    pub fn as_path(&self) -> &Path {
        &self.real
    }

    /// 保持している fd を起点に `candidate`（本ディレクトリ配下の相対パス）を開く。
    ///
    /// [`open_confined`] と同じ検査・拒否を、ディレクトリのパスを再解決せずに行う。
    ///
    /// # Errors
    /// [`open_confined`] と同じ。
    pub fn open_member(&self, candidate: &Path) -> Result<(File, ConfinedPath), PathRejection> {
        open_confined_from_fd(&self.fd, &self.real, candidate)
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl ConfinedDir {
    /// `rel`（本ディレクトリ配下の相対パス）の親までを、保持した fd を起点に成分ごとに
    /// `openat`（`O_DIRECTORY | O_NOFOLLOW`）で辿り、親ディレクトリ fd（`None` は本ディレクトリ自身）
    /// と末尾の名前を返す。パスを開き直さないため、検証後に親が symlink へ差し替えられても
    /// 外へは出られない（REQ-39。書き込み系の閉じ込め）。
    fn open_parent_of(
        &self,
        rel: &Path,
    ) -> Result<(Option<rustix::fd::OwnedFd>, std::ffi::OsString), PathRejection> {
        use rustix::fs::{Mode, OFlags, openat};
        use rustix::io::Errno;

        let escapes = || PathRejection::Escapes {
            candidate: rel.to_path_buf(),
            kind: EscapeKind::Symlink,
        };
        let mut names = Vec::new();
        for c in rel.components() {
            match c {
                Component::Normal(n) => names.push(n),
                Component::CurDir => {}
                Component::ParentDir => {
                    return Err(PathRejection::Escapes {
                        candidate: rel.to_path_buf(),
                        kind: EscapeKind::ParentTraversal,
                    });
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(PathRejection::Escapes {
                        candidate: rel.to_path_buf(),
                        kind: EscapeKind::Absolute,
                    });
                }
            }
        }
        let Some((last, parents)) = names.split_last() else {
            return Err(PathRejection::EmptyPath);
        };
        let mut owned: Option<rustix::fd::OwnedFd> = None;
        for name in parents {
            let cur = owned.as_ref().unwrap_or(&self.fd);
            let next = openat(
                cur,
                *name,
                OFlags::RDONLY
                    | OFlags::DIRECTORY
                    | OFlags::NOFOLLOW
                    | OFlags::NONBLOCK
                    | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|e| match e {
                Errno::LOOP | Errno::NOTDIR => escapes(),
                other => PathRejection::Unresolvable {
                    candidate: rel.to_path_buf(),
                    source: errno_to_io(other),
                },
            })?;
            owned = Some(next);
        }
        // 書き込み側でも読み取り側（`ensure_real_path_under`）と同じく、親ディレクトリ fd の実パスが
        // 本ディレクトリの実パス配下であることを確認する。保持 fd 起点の `openat` は、検証後に本
        // ディレクトリ自体が移動された・bind mount された場合に移動後の場所へ書けてしまうため
        // （REQ-39。fail-closed。実パスを得られない場合も拒否する）。
        let parent_fd = owned.as_ref().unwrap_or(&self.fd);
        ensure_real_path_under(parent_fd, &self.real, rel)?;
        Ok((owned, (*last).to_os_string()))
    }

    /// `rel` に新規の通常ファイルを `O_CREAT | O_EXCL | O_NOFOLLOW` で作って書き込み用に返す。
    ///
    /// 親は保持した fd 起点で辿る（[`ConfinedDir::open_member`] と同じ閉じ込め）。既存（symlink を
    /// 含む）の名前は `AlreadyExists` の [`PathRejection::Unresolvable`] で拒否する。
    ///
    /// # Errors
    /// 経路の拒否・既存・作成失敗。
    pub fn create_new_member(&self, rel: &Path) -> Result<File, PathRejection> {
        use rustix::fs::{Mode, OFlags, openat};

        let (parent, name) = self.open_parent_of(rel)?;
        let dir = parent.as_ref().unwrap_or(&self.fd);
        let fd = openat(
            dir,
            name.as_os_str(),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o666),
        )
        .map_err(|e| PathRejection::Unresolvable {
            candidate: rel.to_path_buf(),
            source: errno_to_io(e),
        })?;
        Ok(File::from(fd))
    }

    /// `rel` にディレクトリを所有者のみ（0700）で新規作成する（既存なら `AlreadyExists`）。
    ///
    /// # Errors
    /// 経路の拒否・既存・作成失敗。
    pub fn create_dir_member(&self, rel: &Path) -> Result<(), PathRejection> {
        use rustix::fs::{Mode, mkdirat};

        let (parent, name) = self.open_parent_of(rel)?;
        let dir = parent.as_ref().unwrap_or(&self.fd);
        mkdirat(dir, name.as_os_str(), Mode::from_raw_mode(0o700)).map_err(|e| {
            PathRejection::Unresolvable {
                candidate: rel.to_path_buf(),
                source: errno_to_io(e),
            }
        })
    }

    /// `rel` の通常ファイル（またはリンク自身）を削除する（書き込み失敗後の片付け用。ディレクトリは消さない）。
    ///
    /// # Errors
    /// 経路の拒否・削除失敗。
    pub fn remove_file_member(&self, rel: &Path) -> Result<(), PathRejection> {
        use rustix::fs::{AtFlags, unlinkat};

        let (parent, name) = self.open_parent_of(rel)?;
        let dir = parent.as_ref().unwrap_or(&self.fd);
        unlinkat(dir, name.as_os_str(), AtFlags::empty()).map_err(|e| PathRejection::Unresolvable {
            candidate: rel.to_path_buf(),
            source: errno_to_io(e),
        })
    }
}

/// 再帰削除の最大深さ（異常に深い木・循環でスタックを使い切らない。REQ-39）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
const MAX_REMOVE_DEPTH: usize = 64;

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl ConfinedDir {
    /// `rel`（本ディレクトリ配下）のディレクトリを `O_NOFOLLOW` で開き、fd を保持する新しい
    /// [`ConfinedDir`] を返す（作成直後のディレクトリの同一性を fd で握るために使う）。
    ///
    /// # Errors
    /// 経路の拒否・ディレクトリでない・実パスが本ディレクトリ配下でない場合。
    pub fn open_dir_member(&self, rel: &Path) -> Result<ConfinedDir, PathRejection> {
        use rustix::fs::{Mode, OFlags, openat};
        use rustix::io::Errno;

        let (parent, name) = self.open_parent_of(rel)?;
        let dir = parent.as_ref().unwrap_or(&self.fd);
        let fd = openat(
            dir,
            name.as_os_str(),
            OFlags::RDONLY
                | OFlags::DIRECTORY
                | OFlags::NOFOLLOW
                | OFlags::NONBLOCK
                | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|e| match e {
            Errno::LOOP | Errno::NOTDIR => PathRejection::Escapes {
                candidate: rel.to_path_buf(),
                kind: EscapeKind::Symlink,
            },
            other => PathRejection::Unresolvable {
                candidate: rel.to_path_buf(),
                source: errno_to_io(other),
            },
        })?;
        let real = ensure_real_path_under(&fd, &self.real, rel)?;
        Ok(ConfinedDir { fd, real })
    }

    /// 本ディレクトリの中身を、保持した fd を起点に再帰的に削除する（本ディレクトリ自身は残す）。
    ///
    /// パスを再解決しないため、保持後に名前が別のディレクトリへ差し替えられても、削除されるのは
    /// 保持している実体の中身だけ（失敗時の後始末が他所を消さない。REQ-39）。symlink は追従せず
    /// リンク自身を消す。深さは [`MAX_REMOVE_DEPTH`] で打ち切る。
    ///
    /// # Errors
    /// 列挙・削除の失敗、深さ超過。
    pub fn clear_contents(&self) -> Result<(), PathRejection> {
        clear_dir_fd(&self.fd, 0).map_err(|source| PathRejection::Unresolvable {
            candidate: PathBuf::new(),
            source,
        })
    }

    /// `rel` が指すディレクトリが `other` と同一の実体（デバイス・inode が一致）か。
    ///
    /// # Errors
    /// 経路の拒否・メタデータ取得失敗。
    pub fn is_same_dir_member(
        &self,
        rel: &Path,
        other: &ConfinedDir,
    ) -> Result<bool, PathRejection> {
        let opened = self.open_dir_member(rel)?;
        let unresolvable = |e: rustix::io::Errno| PathRejection::Unresolvable {
            candidate: rel.to_path_buf(),
            source: errno_to_io(e),
        };
        let a = rustix::fs::fstat(&opened.fd).map_err(unresolvable)?;
        let b = rustix::fs::fstat(&other.fd).map_err(unresolvable)?;
        Ok(a.st_dev == b.st_dev && a.st_ino == b.st_ino)
    }

    /// `rel` の空ディレクトリを削除する（空でなければ失敗する。`unlinkat(AT_REMOVEDIR)`）。
    ///
    /// # Errors
    /// 経路の拒否・削除失敗（空でない場合を含む）。
    pub fn remove_empty_dir_member(&self, rel: &Path) -> Result<(), PathRejection> {
        use rustix::fs::{AtFlags, unlinkat};

        let (parent, name) = self.open_parent_of(rel)?;
        let dir = parent.as_ref().unwrap_or(&self.fd);
        unlinkat(dir, name.as_os_str(), AtFlags::REMOVEDIR).map_err(|e| {
            PathRejection::Unresolvable {
                candidate: rel.to_path_buf(),
                source: errno_to_io(e),
            }
        })
    }
}

/// `dir` の直下を再帰的に削除する（`clear_contents` の実体。fd 起点・`O_NOFOLLOW`）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn clear_dir_fd(dir: &rustix::fd::OwnedFd, depth: usize) -> io::Result<()> {
    use rustix::fs::{AtFlags, Dir, FileType, Mode, OFlags, openat, statat, unlinkat};
    use std::os::unix::ffi::OsStrExt;

    if depth >= MAX_REMOVE_DEPTH {
        return Err(io::Error::other("directory tree is too deep"));
    }
    // 削除中の列挙を避けるため、先に名前を集める。
    let mut names: Vec<std::ffi::OsString> = Vec::new();
    let iter = Dir::read_from(dir).map_err(errno_to_io)?;
    for entry in iter {
        let entry = entry.map_err(errno_to_io)?;
        let bytes = entry.file_name().to_bytes();
        if bytes == b"." || bytes == b".." {
            continue;
        }
        names.push(std::ffi::OsStr::from_bytes(bytes).to_os_string());
    }
    for name in names {
        let stat = statat(dir, name.as_os_str(), AtFlags::SYMLINK_NOFOLLOW).map_err(errno_to_io)?;
        if FileType::from_raw_mode(stat.st_mode) == FileType::Directory {
            let child = openat(
                dir,
                name.as_os_str(),
                OFlags::RDONLY
                    | OFlags::DIRECTORY
                    | OFlags::NOFOLLOW
                    | OFlags::NONBLOCK
                    | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(errno_to_io)?;
            clear_dir_fd(&child, depth + 1)?;
            unlinkat(dir, name.as_os_str(), AtFlags::REMOVEDIR).map_err(errno_to_io)?;
        } else {
            unlinkat(dir, name.as_os_str(), AtFlags::empty()).map_err(errno_to_io)?;
        }
    }
    Ok(())
}

/// `dir`（[`safe_join`] 済みの正準パス）を `O_NOFOLLOW` で開いて fd を保持する。
///
/// 開いた fd の実パスが `dir` と一致することを確認する。検証後・open 前に別ディレクトリへ
/// 差し替えられていれば [`PathRejection::Escapes`]（[`EscapeKind::Symlink`]）で拒否する。
///
/// # Errors
/// 開けない・実パスを得られない・実パスが `dir` と異なる場合。
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub fn open_dir_confined(dir: &ConfinedPath) -> Result<ConfinedDir, PathRejection> {
    use rustix::fs::{Mode, OFlags};
    use rustix::io::Errno;

    let fd = rustix::fs::open(
        dir.as_path(),
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|e| match e {
        Errno::NOTDIR => PathRejection::NotDirectory {
            candidate: dir.as_path().to_path_buf(),
        },
        Errno::LOOP => PathRejection::Escapes {
            candidate: dir.as_path().to_path_buf(),
            kind: EscapeKind::Symlink,
        },
        other => PathRejection::RootUnresolvable {
            source: errno_to_io(other),
        },
    })?;
    let real = fd_real_path(&fd).map_err(|source| PathRejection::RootUnresolvable { source })?;
    if real != dir.as_path() {
        return Err(PathRejection::Escapes {
            candidate: dir.as_path().to_path_buf(),
            kind: EscapeKind::Symlink,
        });
    }
    Ok(ConfinedDir { fd, real })
}

/// 開いた fd の実パスが、最初に確定したルートの実パス配下であることを確認する（fail-closed）。
///
/// 検証後にルートのディレクトリ自体がルート外へ移動された場合、fd 起点の `openat` は移動後の
/// ディレクトリ内のファイルを開けてしまう。fd の実パスを最初に確定した `canon_root` と比較して
/// 拒否する。実パスを得られない場合（Linux で `/proc` が使えない等）も拒否する。
/// Linux は `/proc/self/fd`、macOS は `F_GETPATH` を使う（REQ-39・TASK-39.4-1）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn ensure_real_path_under(
    fd: &rustix::fd::OwnedFd,
    canon_root: &Path,
    candidate: &Path,
) -> Result<PathBuf, PathRejection> {
    // unlink 済み（`st_nlink == 0`）なら、開いた後に対象が差し替え・削除されている。fd 由来のパスは
    // 実在しない名前（Linux の `(deleted)` 付き等）になるため、文字列に頼らず nlink で拒否する。
    let stat = rustix::fs::fstat(fd).map_err(|e| PathRejection::Unresolvable {
        candidate: candidate.to_path_buf(),
        source: errno_to_io(e),
    })?;
    if stat.st_nlink == 0 {
        return Err(PathRejection::Escapes {
            candidate: candidate.to_path_buf(),
            kind: EscapeKind::Symlink,
        });
    }
    let real = fd_real_path(fd).map_err(|source| PathRejection::Unresolvable {
        candidate: candidate.to_path_buf(),
        source,
    })?;
    if real.starts_with(canon_root) {
        Ok(real)
    } else {
        Err(PathRejection::Escapes {
            candidate: candidate.to_path_buf(),
            kind: EscapeKind::Symlink,
        })
    }
}

/// fd 自身の実パスを得る（Linux: `/proc/self/fd/<fd>` の `read_link`）。
#[cfg(target_os = "linux")]
fn fd_real_path(fd: &rustix::fd::OwnedFd) -> io::Result<PathBuf> {
    use std::os::fd::AsRawFd;
    std::fs::read_link(format!("/proc/self/fd/{}", fd.as_raw_fd()))
}

/// fd 自身の実パスを得る（macOS: `F_GETPATH`）。
#[cfg(target_os = "macos")]
fn fd_real_path(fd: &rustix::fd::OwnedFd) -> io::Result<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    let c = rustix::fs::getpath(fd).map_err(errno_to_io)?;
    Ok(PathBuf::from(std::ffi::OsStr::from_bytes(c.as_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 検証後にルートのディレクトリがルート外へ移動された場合、開いた fd の実パスが
    /// 最初のルート配下でなくなり拒否される（REQ-39・TASK-39.4-1。Linux・macOS）。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn req39_real_path_check_rejects_root_moved_away() {
        use rustix::fs::{Mode, OFlags, openat};
        let base = std::env::temp_dir().join(format!("fandhe-guard-moved-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("ws")).expect("mkdir ws");
        std::fs::create_dir_all(base.join("elsewhere")).expect("mkdir elsewhere");
        std::fs::write(base.join("ws/f.txt"), b"x").expect("write");
        let canon_root = std::fs::canonicalize(base.join("ws")).expect("canon");
        let root_fd = rustix::fs::open(
            &canon_root,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .expect("open root");
        let fd = openat(
            &root_fd,
            "f.txt",
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .expect("open file");
        let cand = Path::new("f.txt");
        let real = ensure_real_path_under(&fd, &canon_root, cand).expect("under root");
        assert_eq!(real, canon_root.join("f.txt"));
        std::fs::rename(base.join("ws"), base.join("elsewhere/ws")).expect("move root away");
        match ensure_real_path_under(&fd, &canon_root, cand) {
            Err(PathRejection::Escapes {
                kind: EscapeKind::Symlink,
                ..
            }) => {}
            other => panic!("expected Escapes, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// 開いた後に同名の別ファイルへ差し替えられ、fd が unlink 済みの対象になった場合は、実在しない
    /// パスを ConfinedPath として返さず拒否する（`st_nlink == 0`。文字列照合に頼らない。
    /// REQ-39・TASK-39.4-1）。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn req39_replaced_after_open_is_rejected_not_returned_as_deleted_path() {
        use rustix::fs::{Mode, OFlags, openat};
        let base = std::env::temp_dir().join(format!("fandhe-guard-fdpath-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("ws")).expect("mkdir ws");
        std::fs::write(base.join("ws/f.txt"), b"old").expect("write old");
        std::fs::write(base.join("ws/g.txt"), b"new").expect("write new");
        let canon_root = std::fs::canonicalize(base.join("ws")).expect("canon");
        let root_fd = rustix::fs::open(
            &canon_root,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .expect("open root");
        let fd = openat(
            &root_fd,
            "f.txt",
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .expect("open file");
        std::fs::rename(base.join("ws/g.txt"), base.join("ws/f.txt")).expect("swap");
        match ensure_real_path_under(&fd, &canon_root, Path::new("f.txt")) {
            Err(PathRejection::Escapes {
                kind: EscapeKind::Symlink,
                ..
            }) => {}
            other => panic!("expected Escapes for unlinked fd, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// 正準化後に残る親ディレクトリ（`stage`）を、検証と openat の間に外部への symlink へ差し替えても
    /// 外部ファイルは開かれず、Escapes で拒否される（決定的な再現。REQ-39・TASK-39.4-1・#158。
    /// 証拠種別: テストハーネス。Linux・macOS）。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn req39_parent_swapped_between_validation_and_openat_is_rejected() {
        let base = std::env::temp_dir().join(format!("fandhe-guard-hook-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let ws = base.join("ws");
        std::fs::create_dir_all(ws.join("stage")).expect("mkdir stage");
        std::fs::create_dir_all(base.join("outside")).expect("mkdir outside");
        std::fs::write(ws.join("stage/model.onnx"), b"inside").expect("write inside");
        std::fs::write(base.join("outside/model.onnx"), b"outside").expect("write outside");
        let (stage, moved, outside) = (
            ws.join("stage"),
            ws.join("stage_real"),
            base.join("outside"),
        );
        test_hooks::set_after_validation(Box::new(move || {
            std::fs::rename(&stage, &moved).expect("move stage");
            std::os::unix::fs::symlink(&outside, &stage).expect("symlink stage");
        }));
        match open_confined(&ws, Path::new("stage/model.onnx")) {
            Err(PathRejection::Escapes {
                kind: EscapeKind::Symlink,
                ..
            }) => {}
            other => panic!("expected Escapes(Symlink), got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn dotted_absolute_escape_is_classified_absolute() {
        let base = std::env::temp_dir().join(format!("fandhe-guard-dots-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("ws")).expect("mkdir ws");
        std::fs::write(base.join("secret.txt"), b"s").expect("write");
        let ws = base.join("ws");
        // 絶対パスが `..` でルートの外へ出る: 字句的にも外なので Absolute。
        let cand = ws.join("../secret.txt");
        match safe_join(&ws, &cand) {
            Err(PathRejection::Escapes { kind, .. }) => assert_eq!(kind, EscapeKind::Absolute),
            other => panic!("expected Escapes(Absolute), got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn lexical_normalize_cases() {
        assert_eq!(
            lexical_normalize(Path::new("/a/b/../c/./d")),
            PathBuf::from("/a/c/d")
        );
        assert_eq!(lexical_normalize(Path::new("/a/../..")), PathBuf::from("/"));
        assert_eq!(lexical_normalize(Path::new("../x")), PathBuf::from("../x"));
        assert_eq!(
            lexical_normalize(Path::new("../..")),
            PathBuf::from("../..")
        );
        assert_eq!(
            lexical_normalize(Path::new("../../x")),
            PathBuf::from("../../x")
        );
        assert_eq!(
            lexical_normalize(Path::new("a/../../b")),
            PathBuf::from("../b")
        );
        assert_eq!(
            lexical_normalize(Path::new("/../../x")),
            PathBuf::from("/x")
        );
        assert_eq!(lexical_normalize(Path::new("a/..")), PathBuf::new());
    }

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

    /// errno 別の終了コード写像を表で固定する（REQ-21・REQ-39）。fd 枯渇（EMFILE・ENFILE）などの
    /// 環境起因は 70、利用者が渡したパスの不正は 64。
    #[test]
    fn req21_errno_to_exit_code_table() {
        let mac = cfg!(target_os = "macos");
        let table: [(&str, i32, ExitCode); 15] = [
            ("EPERM", 1, ExitCode::RuntimeError),
            ("EINTR", 4, ExitCode::RuntimeError),
            ("EIO", 5, ExitCode::RuntimeError),
            ("ENOMEM", 12, ExitCode::RuntimeError),
            ("EACCES", 13, ExitCode::RuntimeError),
            ("ENFILE", 23, ExitCode::RuntimeError),
            ("EMFILE", 24, ExitCode::RuntimeError),
            ("ENOSPC", 28, ExitCode::RuntimeError),
            ("EAGAIN", if mac { 35 } else { 11 }, ExitCode::RuntimeError),
            (
                "ETIMEDOUT",
                if mac { 60 } else { 110 },
                ExitCode::RuntimeError,
            ),
            ("EDQUOT", if mac { 69 } else { 122 }, ExitCode::RuntimeError),
            ("ENOENT", 2, ExitCode::InvalidInput),
            ("ENOTDIR", libc_enotdir(), ExitCode::InvalidInput),
            ("ELOOP", libc_eloop(), ExitCode::InvalidInput),
            ("ENAMETOOLONG", libc_enametoolong(), ExitCode::InvalidInput),
        ];
        for (name, code, expected) in table {
            let r = PathRejection::Unresolvable {
                candidate: PathBuf::from("x"),
                source: io::Error::from_raw_os_error(code),
            };
            assert_eq!(r.exit_code(), expected, "{name} ({code})");
            let root = PathRejection::RootUnresolvable {
                source: io::Error::from_raw_os_error(code),
            };
            assert_eq!(root.exit_code(), expected, "root {name} ({code})");
        }
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
