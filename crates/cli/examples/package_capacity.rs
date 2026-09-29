//! 容量内訳を人が実機で計測するための開発用ツール（REQ-30・TASK-30.1-2・#123）。
//!
//! 使い方: `cargo run -p fandhe-edge-cli --example package_capacity -- weights=<path> metadata=<path> ...`
//! 引数は `<component>=<path>`（component は `weights`・`vocab_or_feature_transform`・
//! `label_table`・`calibration`・`metadata`）。成功時は `capacity` の JSON 1 行を出して exit 0、
//! 失敗時は `{"code","message"}` を出して対応する終了コードで終える。
//!
//! 対応 OS: Linux・macOS のみ。それ以外（M10 時点で対象外）は、ガード層の経路の閉じ込めが
//! `UnsupportedPlatform`（終了コード 70）で全ファイルを拒否する（fail-closed。REQ-39）。
//!
//! 位置づけ: 配布物・推論経路には入らず、`package` 工程の正式な CLI 契約でもない。
//! argv は untrusted のため、パスの検証と open はガード層の `open_confined`（ルート fd 起点の
//! `openat`＋`O_NOFOLLOW`）に一体で委ね、ルート（カレントディレクトリ）外への `..`・絶対パス・
//! symlink 参照を拒否する（REQ-39）。拒否理由は `reason_code` だけを固定文で返す。非 UTF-8 の引数は
//! InvalidInput として返す。引数は 16 件までで、開く前に件数を検証する。ファイルサイズは開いた直後と
//! 計測直前に上限（1GiB）と照合し、重複は計測コアが拒否する。構成要素の分類は配布パッケージ形式
//! （TASK-28・32）が確定するまでの暫定で、C1 は `model.onnx` を `weights`、`artifact.json` を
//! `metadata` として渡す（語彙は ONNX 内に保持されるため `vocab_or_feature_transform` は 0 件）。

use fandhe_edge_cli::output::{capacity_error_report, write_error_report, write_package_capacity};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::FsError;
use fandhe_edge_guard::path::open_confined as guard_open_confined;
use fandhe_edge_runtime::capacity::{CapacityError, PackageComponent, measure_opened_files};
use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// 計測対象ルート配下のファイルを、ガード層の経路の閉じ込めつきで開いてハンドルを返す。
///
/// 検証と open は `fandhe_edge_guard::path::open_confined` に一体で委ねる。ルートの fd を起点に
/// 各成分を `openat`（`O_NOFOLLOW`）で開くため、検証後に親ディレクトリや対象が symlink へ
/// 差し替えられてもルート外は開けない（TOCTOU 対策。REQ-39・TASK-39.4-1）。Linux・macOS 以外は
/// `UnsupportedPlatform`（終了コード 70）で拒否される（fail-closed）。以降の計測はパスではなく
/// このハンドルで行う。拒否は `reason_code` だけを英語の固定文へ写し、パスは JSON へ出さない。
fn open_confined(root: &Path, raw: &Path) -> Result<(PathBuf, File), ErrorReport> {
    let (file, confined) = guard_open_confined(root, raw).map_err(|e| {
        ErrorReport::new(e.exit_code(), format!("path rejected: {}", e.reason_code()))
    })?;
    Ok((confined.into_path_buf(), file))
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

/// 計測直前に、保持しているハンドルの現在のサイズを `limit` と再照合する（REQ-39）。
/// オープン後に追記されて上限を超えたファイルを、計測値として受理しない。
fn enforce_measured_sizes(
    files: &[(PackageComponent, PathBuf, File)],
    limit: u64,
) -> Result<(), ErrorReport> {
    for (_, path, file) in files {
        let meta = file.metadata().map_err(|e| {
            capacity_error_report(&CapacityError::File(FsError::Read {
                path: path.clone(),
                source: e,
            }))
        })?;
        enforce_size_limit(path, meta.len(), limit)?;
    }
    Ok(())
}

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
        .and_then(|files| {
            enforce_measured_sizes(&files, MAX_FILE_BYTES)?;
            measure_opened_files(&files).map_err(|e| capacity_error_report(&e))
        });
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

    /// REQ-39: Linux・macOS ではルート配下の相対パスは受理され、`..`・ルート外の絶対パス・
    /// ルート外を指す末尾 symlink は InvalidInput（64）で拒否される。ルート配下の実体へ解決される
    /// symlink はガード層の仕様どおり実体として受理される。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn req39_open_confined_accepts_inside_and_rejects_escapes() {
        let root = root();
        assert!(open_confined(&root, Path::new("sub/m.json")).is_ok());
        std::os::unix::fs::symlink(root.join("sub").join("m.json"), root.join("sub/in_link"))
            .unwrap();
        assert!(open_confined(&root, Path::new("sub/in_link")).is_ok());
        std::os::unix::fs::symlink("/etc/passwd", root.join("sub/out_link")).unwrap();
        for bad in ["sub/out_link", "/etc/passwd", "../x", "sub/../../x"] {
            let e = open_confined(&root, Path::new(bad)).unwrap_err();
            assert_eq!(e.code, ExitCode::InvalidInput, "{bad}");
            assert!(e.message.starts_with("path rejected: "), "{bad}");
        }
    }

    /// REQ-39: 上記以外の OS は UnsupportedPlatform（70・固定文）で常に拒否する（fail-closed）。
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    #[test]
    fn req39_open_confined_is_unsupported_on_other_platforms() {
        let root = root();
        let e = open_confined(&root, Path::new("sub/m.json")).unwrap_err();
        assert_eq!(e.code, ExitCode::RuntimeError);
        assert_eq!(e.message, "path rejected: unsupported_platform");
    }

    /// REQ-39: 上限を超えるサイズは LimitExceeded（終了コード 20）で拒否し、上限ちょうどは通す。
    #[test]
    fn req39_size_limit_rejects_oversized_file() {
        let p = Path::new("x");
        assert!(enforce_size_limit(p, 10, 10).is_ok());
        let e = enforce_size_limit(p, 11, 10).unwrap_err();
        assert_eq!(e.code, ExitCode::LimitExceeded);
    }

    /// REQ-39: オープン後に上限を超えて成長したファイルは、計測時の再検証で拒否される。
    #[test]
    fn req39_measure_time_size_recheck_rejects_grown_file() {
        use std::io::Write as _;
        let root = root();
        let path = root.join("sub").join("m.json");
        let file = File::open(&path).unwrap();
        let files = vec![(PackageComponent::Metadata, path.clone(), file)];
        assert!(enforce_measured_sizes(&files, 2).is_ok());
        let mut w = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        w.write_all(b"xxxx").unwrap();
        let e = enforce_measured_sizes(&files, 2).unwrap_err();
        assert_eq!(e.code, ExitCode::LimitExceeded);
    }

    /// REQ-39: 親ディレクトリ経由のリンクによるルート外参照は拒否される。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn req39_open_confined_rejects_linked_parent_outside_root() {
        let root = root();
        std::os::unix::fs::symlink("/etc", root.join("escape")).unwrap();
        let e = open_confined(&root, Path::new("escape/passwd")).unwrap_err();
        assert_eq!(e.code, ExitCode::InvalidInput);
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

    /// REQ-21: 不在ファイル・ディレクトリは InvalidInput の固定文（パスを含まない）になる。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn req21_missing_and_directory_are_invalid_input() {
        let root = root();
        let missing = open_confined(&root, Path::new("sub/none.json")).unwrap_err();
        assert_eq!(missing.code, ExitCode::InvalidInput);
        assert_eq!(missing.message, "path rejected: path_unresolvable");
        let dir = open_confined(&root, Path::new("sub")).unwrap_err();
        assert_eq!(dir.code, ExitCode::InvalidInput);
        assert_eq!(dir.message, "path rejected: not_regular_file");
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
