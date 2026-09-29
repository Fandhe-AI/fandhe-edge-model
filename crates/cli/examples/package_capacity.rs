//! 容量内訳を人が実機で計測するための開発用ツール（REQ-30・TASK-30.1-2・#123）。
//!
//! 使い方: `cargo run -p fandhe-edge-cli --example package_capacity -- weights=<path> metadata=<path> ...`
//! 引数は `<component>=<path>`（component は `weights`・`vocab_or_feature_transform`・
//! `label_table`・`calibration`・`metadata`）。成功時は `capacity` の JSON 1 行を出して exit 0、
//! 失敗時は `{"code","message"}` を出して対応する終了コードで終える。
//!
//! 位置づけ: 配布物・推論経路には入らず、`package` 工程の正式な CLI 契約でもない。
//! argv は untrusted のため、パスはカレントディレクトリ（計測対象ルート）配下の相対パスに限り、
//! 絶対パス・`..`・親経由の symlink によるルート外参照を拒否する（REQ-39）。非 UTF-8 の引数は
//! InvalidInput として返す。引数は 16 件までで、開く前に件数を検証する。経路違反以外のファイル系の失敗
//! （不在・権限・ディレクトリ・FIFO・末尾 symlink）は `capacity_error_report` の公開メッセージへ写し、
//! 重複は計測コアが拒否する。構成要素の分類は配布パッケージ形式
//! （TASK-28・32）が確定するまでの暫定で、C1 は `model.onnx` を `weights`、`artifact.json` を
//! `metadata` として渡す（語彙は ONNX 内に保持されるため `vocab_or_feature_transform` は 0 件）。

use fandhe_edge_cli::output::{capacity_error_report, write_error_report, write_package_capacity};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::{FsError, open_regular_file_for_read};
use fandhe_edge_runtime::capacity::{CapacityError, PackageComponent, measure_opened_files};
use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

/// 計測対象ルート（カレントディレクトリ）配下のファイルを検証つきで開き、ハンドルを返す。
///
/// 絶対パス・`..` 成分は拒否する。親ディレクトリは正規化（symlink 解決）してルート配下を確認し、
/// 通常ファイルとして開いたのち、開いたハンドルが「正規化後もルート配下の同じパスに実在する
/// 同一ファイル」であることを再確認する。検証後・オープン前に親ディレクトリがルート外への
/// symlink に差し替えられても、開いたハンドルは再解決したパスと一致せず拒否される
/// （検証と取得の間の TOCTOU 対策。REQ-39）。以降の計測はパスではなくこのハンドルで行う。
fn open_confined(root: &Path, raw: &Path) -> Result<(PathBuf, File), ErrorReport> {
    let denied = || {
        ErrorReport::new(
            ExitCode::InvalidInput,
            "path must be a relative path inside the working directory",
        )
    };
    // 経路の閉じ込めを満たした後のファイル系の失敗は、REQ-21 の公開メッセージへ写す
    // （不在・権限・FIFO・ディレクトリ・symlink を「経路違反」に丸めない）
    let read_failure = |path: &Path, source: std::io::Error| {
        capacity_error_report(&CapacityError::File(FsError::Read {
            path: path.to_path_buf(),
            source,
        }))
    };
    if raw.is_absolute()
        || !raw
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(denied());
    }
    let joined = root.join(raw);
    let name = joined.file_name().ok_or_else(denied)?;
    let parent = joined.parent().ok_or_else(denied)?;
    let parent = parent
        .canonicalize()
        .map_err(|e| read_failure(&joined, e))?;
    if !parent.starts_with(root) {
        return Err(denied());
    }
    let path = parent.join(name);
    // 末尾が symlink の場合は開く前に拒否する（開いた後の再確認でも検出される）
    let link_meta = std::fs::symlink_metadata(&path).map_err(|e| read_failure(&path, e))?;
    if link_meta.file_type().is_symlink() {
        return Err(capacity_error_report(&CapacityError::SymlinkRejected {
            path: path.clone(),
        }));
    }
    let file = open_regular_file_for_read(&path)
        .map_err(|e| capacity_error_report(&CapacityError::File(e)))?;
    // 開いた後に同じパスを再解決し、ルート配下・同一パス・同一ファイルであることを確認する。
    // 差し替えの兆候は経路違反ではなく symlink 差し替え（計測コアと同じ扱い）として拒否する
    let replaced = || capacity_error_report(&CapacityError::SymlinkRejected { path: path.clone() });
    let resolved = path.canonicalize().map_err(|_| replaced())?;
    if resolved != path || !resolved.starts_with(root) {
        return Err(replaced());
    }
    let opened = file.metadata().map_err(|e| read_failure(&path, e))?;
    enforce_size_limit(&path, opened.len(), MAX_FILE_BYTES)?;
    let current = std::fs::metadata(&resolved).map_err(|_| replaced())?;
    if !same_file(&opened, &current) {
        return Err(replaced());
    }
    Ok((path, file))
}

/// 2 つのメタデータが同一ファイルのものか（Unix ではデバイス・inode の一致）。
/// Unix 以外は標準ライブラリだけではハンドル由来のファイル ID を取れないため、近似せず
/// 常に「同一と証明できない」として拒否する（fail-closed。M10 時点で対象外。REQ-39）。
#[cfg(unix)]
fn same_file(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    a.dev() == b.dev() && a.ino() == b.ino()
}

#[cfg(not(unix))]
fn same_file(_a: &std::fs::Metadata, _b: &std::fs::Metadata) -> bool {
    false
}

/// 計測対象 1 ファイルの大きさの上限（バイト）。配布物の容量目安（40MB。REQ-30）を大きく
/// 上回る値で、異常に大きい入力を開いたハンドルのメタデータ段階で拒否する（REQ-39）。
const MAX_FILE_BYTES: u64 = 1 << 30;

/// 開いたハンドルのサイズが `limit` を超えたら [`FsError::TooLarge`] で拒否する（REQ-39）。
fn enforce_size_limit(path: &Path, size: u64, limit: u64) -> Result<(), ErrorReport> {
    if size > limit {
        return Err(capacity_error_report(&CapacityError::File(
            FsError::TooLarge {
                path: path.to_path_buf(),
                size,
                limit,
            },
        )));
    }
    Ok(())
}

/// 受け付ける引数の上限。構成要素の種類数（5）に余裕を持たせた値で、重複指定は計測コアが拒否する。
const MAX_ARGS: usize = 16;

fn parse(
    args: &[OsString],
    root: &Path,
) -> Result<Vec<(PackageComponent, PathBuf, File)>, ErrorReport> {
    let invalid = || {
        ErrorReport::new(
            ExitCode::InvalidInput,
            "invalid argument, expected <component>=<path>",
        )
    };
    // 開いたハンドルを保持するため、開く前に件数を検証して FD・メモリの枯渇を防ぐ（REQ-39）。
    // 構成要素は重複不可なので、正当な指定は構成要素の種類数を超えない
    if args.len() > MAX_ARGS {
        return Err(ErrorReport::new(
            ExitCode::InvalidInput,
            "too many arguments",
        ));
    }
    let mut files = Vec::with_capacity(args.len());
    for arg in args {
        // 非 UTF-8 の引数は panic させず InvalidInput に写す
        let arg = arg.to_str().ok_or_else(invalid)?;
        let (name, path) = arg.split_once('=').ok_or_else(invalid)?;
        let component = PackageComponent::all()
            .into_iter()
            .find(|c| c.as_str() == name)
            .ok_or_else(invalid)?;
        let (path, file) = open_confined(root, Path::new(path))?;
        files.push((component, path, file));
    }
    Ok(files)
}

fn main() -> std::process::ExitCode {
    // 収集前に件数を打ち切る（REQ-39）。上限 +1 件だけ取れば超過を parse が検出できるため、
    // untrusted な引数の件数に比例したメモリ確保をしない
    let args: Vec<OsString> = std::env::args_os().skip(1).take(MAX_ARGS + 1).collect();
    let mut out = std::io::stdout().lock();
    let result = std::env::current_dir()
        .and_then(|d| d.canonicalize())
        .map_err(|_| ErrorReport::new(ExitCode::RuntimeError, "cannot resolve working directory"))
        .and_then(|root| parse(&args, &root))
        .and_then(|files| measure_opened_files(&files).map_err(|e| capacity_error_report(&e)));
    let code = match result {
        Ok(breakdown) => write_package_capacity(&mut out, &breakdown),
        Err(report) => write_error_report(&mut out, &report),
    };
    let _ = out.flush();
    match code {
        Ok(c) => std::process::ExitCode::from(c.code()),
        Err(_) => std::process::ExitCode::from(ExitCode::RuntimeError.code()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> PathBuf {
        // 予測可能なパスを再利用・上書きしない。一意なディレクトリを create_dir で排他作成し、
        // ファイルは create_new で作る（衝突時は名前を変えて再試行）。
        use std::io::Write as _;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        for _ in 0..100 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir()
                .join(format!("fandhe_cap_ex_{}_{nanos}_{n}", std::process::id()));
            match std::fs::create_dir(&dir) {
                Ok(()) => {
                    std::fs::create_dir(dir.join("sub")).unwrap();
                    let mut f = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(dir.join("sub").join("m.json"))
                        .unwrap();
                    f.write_all(b"{}").unwrap();
                    return dir.canonicalize().unwrap();
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create temp dir: {e}"),
            }
        }
        panic!("could not create a unique temp dir");
    }

    /// REQ-39: ルート配下の相対パスは受理され、絶対パス・`..` は InvalidInput になる。
    #[test]
    fn req39_open_confined_rejects_absolute_and_parent_paths() {
        let root = root();
        assert!(open_confined(&root, Path::new("sub/m.json")).is_ok());
        // 末尾が symlink のファイルは拒否される（ルート配下の別ファイルへの link でも）
        #[cfg(unix)]
        {
            let l = root.join("sub").join("tail_link");
            let _ = std::fs::remove_file(&l);
            std::os::unix::fs::symlink(root.join("sub").join("m.json"), &l).unwrap();
            assert!(open_confined(&root, Path::new("sub/tail_link")).is_err());
        }
        assert!(open_confined(&root, Path::new("/etc/passwd")).is_err());
        assert!(open_confined(&root, Path::new("../x")).is_err());
        assert!(open_confined(&root, Path::new("sub/../../x")).is_err());
    }

    /// REQ-39: 上限を超えるサイズは LimitExceeded（終了コード 20）で拒否し、上限ちょうどは通す。
    #[test]
    fn req39_size_limit_rejects_oversized_file() {
        let p = Path::new("x");
        assert!(enforce_size_limit(p, 10, 10).is_ok());
        let e = enforce_size_limit(p, 11, 10).unwrap_err();
        assert_eq!(e.code, ExitCode::LimitExceeded);
    }

    /// REQ-39: 親ディレクトリ経由のリンクによるルート外参照は拒否される。
    #[cfg(unix)]
    #[test]
    fn req39_open_confined_rejects_linked_parent_outside_root() {
        let root = root();
        let link = root.join("escape");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink("/etc", &link).unwrap();
        assert!(open_confined(&root, Path::new("escape/passwd")).is_err());
    }

    /// REQ-39: 引数件数の上限を超えたら、ファイルを開く前に InvalidInput で拒否する。
    #[test]
    fn req39_too_many_arguments_rejected_before_open() {
        let root = root();
        let args: Vec<OsString> = (0..=MAX_ARGS)
            .map(|_| OsString::from("metadata=sub/m.json"))
            .collect();
        let err = parse(&args, &root).unwrap_err();
        assert_eq!(err.message, "too many arguments");
    }

    /// REQ-21: 経路内の不在ファイル・ディレクトリ・末尾 symlink は経路違反ではなく
    /// 容量計測の公開メッセージになる。
    #[test]
    fn req21_file_errors_are_not_path_confinement() {
        let root = root();
        let missing = open_confined(&root, Path::new("sub/none.json")).unwrap_err();
        assert_eq!(missing.message, "package file is not readable");
        #[cfg(unix)]
        {
            let dir = open_confined(&root, Path::new("sub")).unwrap_err();
            assert_eq!(dir.message, "package file is not a regular file");
            let l = root.join("sub").join("tail_link2");
            let _ = std::fs::remove_file(&l);
            std::os::unix::fs::symlink(root.join("sub").join("m.json"), &l).unwrap();
            let e = open_confined(&root, Path::new("sub/tail_link2")).unwrap_err();
            assert_eq!(
                e.message,
                "package file is a symlink or was replaced during measurement"
            );
        }
    }

    /// REQ-21: 非 UTF-8 の引数は panic せず InvalidInput の ErrorReport になる。
    #[cfg(unix)]
    #[test]
    fn req21_non_utf8_argument_is_invalid_input() {
        use std::os::unix::ffi::OsStringExt;
        let root = root();
        let arg = OsString::from_vec(b"metadata=\xff".to_vec());
        assert!(parse(&[arg], &root).is_err());
    }
}
