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
use fandhe_edge_core::fs::open_regular_file_for_read;
use fandhe_edge_runtime::capacity::{PackageComponent, measure_opened_files};
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
    let path = parent.join(name);
    // 末尾が symlink の場合は開く前に拒否する（開いた後の再確認でも検出される）
    let link_meta = std::fs::symlink_metadata(&path).map_err(|_| denied())?;
    if link_meta.file_type().is_symlink() {
        return Err(denied());
    }
    let file = open_regular_file_for_read(&path).map_err(|_| denied())?;
    // 開いた後に同じパスを再解決し、ルート配下・同一パス・同一ファイルであることを確認する
    let resolved = path.canonicalize().map_err(|_| denied())?;
    if resolved != path || !resolved.starts_with(root) {
        return Err(denied());
    }
    let opened = file.metadata().map_err(|_| denied())?;
    let current = std::fs::metadata(&resolved).map_err(|_| denied())?;
    if !same_file(&opened, &current) {
        return Err(denied());
    }
    Ok((path, file))
}

/// 2 つのメタデータが同一ファイルのものか（Unix ではデバイス・inode の一致）。
/// Unix 以外は M10 時点で対象外のため、種別とサイズの一致で近似する（fail-closed 側の近似）。
#[cfg(unix)]
fn same_file(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    a.dev() == b.dev() && a.ino() == b.ino()
}

#[cfg(not(unix))]
fn same_file(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    a.file_type() == b.file_type() && a.len() == b.len() && a.modified().ok() == b.modified().ok()
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
    let mut files = Vec::new();
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
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
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
        let dir = std::env::temp_dir().join(format!("fandhe_cap_ex_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub").join("m.json"), b"{}").unwrap();
        dir.canonicalize().unwrap()
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
