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
//! InvalidInput として返す。末尾の symlink・FIFO・重複は計測コア側で拒否される。構成要素の分類は配布パッケージ形式
//! （TASK-28・32）が確定するまでの暫定で、C1 は `model.onnx` を `weights`、`artifact.json` を
//! `metadata` として渡す（語彙は ONNX 内に保持されるため `vocab_or_feature_transform` は 0 件）。

use fandhe_edge_cli::output::{capacity_error_report, write_error_report, write_package_capacity};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_runtime::capacity::{PackageComponent, PackageFile, measure_package};
use std::ffi::OsString;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

/// 計測対象ルート（カレントディレクトリ）配下であることを検証し、計測に渡すパスを返す。
///
/// 絶対パス・`..` 成分は拒否し、親ディレクトリを正規化（symlink 解決）してルート配下であることを
/// 確認する（親経由の symlink によるルート外参照を拒否。REQ-39）。末尾の要素は正規化せず
/// 計測コア側の symlink 拒否に委ねる。
fn confine(root: &Path, raw: &Path) -> Result<PathBuf, ErrorReport> {
    let denied = || {
        ErrorReport::new(
            ExitCode::InvalidInput,
            "path must be a relative path inside the working directory",
        )
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
    let parent = parent.canonicalize().map_err(|_| denied())?;
    if !parent.starts_with(root) {
        return Err(denied());
    }
    Ok(parent.join(name))
}

fn parse(args: &[OsString], root: &Path) -> Result<Vec<PackageFile>, ErrorReport> {
    let invalid = || {
        ErrorReport::new(
            ExitCode::InvalidInput,
            "invalid argument, expected <component>=<path>",
        )
    };
    let mut files = Vec::new();
    for arg in args {
        // 非 UTF-8 の引数は panic させず InvalidInput に写す
        let arg = arg.to_str().ok_or_else(invalid)?;
        let (name, path) = arg.split_once('=').ok_or_else(invalid)?;
        let component = PackageComponent::all()
            .into_iter()
            .find(|c| c.as_str() == name)
            .ok_or_else(invalid)?;
        files.push(PackageFile {
            component,
            path: confine(root, Path::new(path))?,
        });
    }
    Ok(files)
}

fn main() -> std::process::ExitCode {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let mut out = std::io::stdout().lock();
    let result = std::env::current_dir()
        .and_then(|d| d.canonicalize())
        .map_err(|_| ErrorReport::new(ExitCode::RuntimeError, "cannot resolve working directory"))
        .and_then(|root| parse(&args, &root))
        .and_then(|files| measure_package(&files).map_err(|e| capacity_error_report(&e)));
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
        let dir = std::env::temp_dir().join(format!("fandhe_cap_ex_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub").join("m.json"), b"{}").unwrap();
        dir.canonicalize().unwrap()
    }

    /// REQ-39: ルート配下の相対パスは受理され、絶対パス・`..` は InvalidInput になる。
    #[test]
    fn req39_confine_rejects_absolute_and_parent_paths() {
        let root = root();
        assert!(confine(&root, Path::new("sub/m.json")).is_ok());
        assert!(confine(&root, Path::new("/etc/passwd")).is_err());
        assert!(confine(&root, Path::new("../x")).is_err());
        assert!(confine(&root, Path::new("sub/../../x")).is_err());
    }

    /// REQ-39: 親ディレクトリ経由のリンクによるルート外参照は拒否される。
    #[cfg(unix)]
    #[test]
    fn req39_confine_rejects_linked_parent_outside_root() {
        let root = root();
        let link = root.join("escape");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink("/etc", &link).unwrap();
        assert!(confine(&root, Path::new("escape/passwd")).is_err());
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
