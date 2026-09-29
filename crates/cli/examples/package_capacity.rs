//! 容量内訳を人が実機で計測するための開発用ツール（REQ-30・TASK-30.1-2・#123）。
//!
//! 使い方: `cargo run -p fandhe-edge-cli --example package_capacity -- weights=<path> metadata=<path> ...`
//! 引数は `<component>=<path>`（component は `weights`・`vocab_or_feature_transform`・
//! `label_table`・`calibration`・`metadata`）。成功時は `capacity` の JSON 1 行を出して exit 0、
//! 失敗時は `{"code","message"}` を出して対応する終了コードで終える。
//!
//! 位置づけ: 配布物・推論経路には入らず、`package` 工程の正式な CLI 契約でもない。
//! argv は untrusted だが、ルートへの閉じ込めはガード層（TASK-39.x）の範囲でここでは行わない
//! （symlink・FIFO・重複は計測コア側で拒否される）。構成要素の分類は配布パッケージ形式
//! （TASK-28・32）が確定するまでの暫定で、C1 は `model.onnx` を `weights`、`artifact.json` を
//! `metadata` として渡す（語彙は ONNX 内に保持されるため `vocab_or_feature_transform` は 0 件）。

use fandhe_edge_cli::output::{capacity_error_report, write_error_report, write_package_capacity};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_runtime::capacity::{PackageComponent, PackageFile, measure_package};
use std::io::Write;
use std::path::PathBuf;

fn parse(args: &[String]) -> Result<Vec<PackageFile>, ErrorReport> {
    let invalid = || {
        ErrorReport::new(
            ExitCode::InvalidInput,
            "invalid argument, expected <component>=<path>",
        )
    };
    let mut files = Vec::new();
    for arg in args {
        let (name, path) = arg.split_once('=').ok_or_else(invalid)?;
        let component = PackageComponent::all()
            .into_iter()
            .find(|c| c.as_str() == name)
            .ok_or_else(invalid)?;
        files.push(PackageFile {
            component,
            path: PathBuf::from(path),
        });
    }
    Ok(files)
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut out = std::io::stdout().lock();
    let result = parse(&args)
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
