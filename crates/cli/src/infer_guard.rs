//! `infer` の `--package` と `onnx_file` を経路の閉じ込めへ通す統合（REQ-39・PoC-20 ケース 1・
//! TASK-39.4-2・#159）。
//!
//! # 呼び出し文脈
//!
//! `main.rs` が `infer` の引数解析に成功した直後に [`guard_infer_paths`] を呼ぶ。拒否は
//! [`ErrorReport`]（`invalid_input`=64 等。message は `path rejected: <reason_code>` の固定語彙で、
//! パス・`onnx_file` の値・`artifact.json` の本文を含めない）として呼び出し側が JSON 1 行で出力する。
//!
//! # 検査の順序（fail-closed）
//!
//! 1. `--package` を workspace（カレントディレクトリ）配下へ閉じ込め、ディレクトリであることを確認
//! 2. パッケージ配下の `artifact.json` を、開いた fd から上限付きで読む
//! 3. `onnx_file` を取り出し、パッケージ配下（かつ workspace 配下）の ONNX ファイルを開き、
//!    開いた fd のサイズがランタイムの読み込み上限（`MAX_MODEL_FILE_BYTES`）以下であることを確認
//!    （超過は `limit_exceeded`=20。REQ-39 資源の上限）
//!
//! ONNX の形式検査・サイズ上限の正式値の確定・sha256 照合は本モジュールの範囲外（TASK-39.2-4・
//! TASK-39.5・TASK-39.6）。`--input-file`・`--out` の閉じ込めも範囲外。
//!
//! # 後続（#136）への申し送り
//!
//! 返す [`GuardedInferInputs::onnx`] は検証つきで開いた [`File`]。推論への接続では、この
//! `File` からバイト列を読んでランタイムへ渡すこと。パスから開き直すと検証後の差し替え
//! （TOCTOU）が残るため、パスを受け取って自前で開く読み込み API は使わない。

use std::fs::File;
use std::path::Path;

use fandhe_edge_core::artifact_meta::{ArtifactOnnxRef, MAX_ARTIFACT_META_BYTES};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::read_open_file_bounded;
use fandhe_edge_guard::package::{ConfinedPackage, confine_package};
use fandhe_edge_guard::path::ConfinedPath;
use fandhe_edge_runtime::onnx::MAX_MODEL_FILE_BYTES;

use crate::args::InferArgs;
use crate::error_report::ToErrorReport;

/// パッケージ内のメタデータのファイル名（パッケージ形式の確定は TASK-28/32）。
const ARTIFACT_META_FILE: &str = "artifact.json";

/// 経路ガードを通過した `infer` の入力。
#[derive(Debug)]
pub struct GuardedInferInputs {
    /// workspace 配下へ閉じ込め済みのパッケージ。
    pub package: ConfinedPackage,
    /// 検証つきで開いた ONNX ファイル。読むときはこの `File` だけを使う。
    pub onnx: File,
    /// `onnx` の実パス（表示・診断用。開き直さない）。
    pub onnx_path: ConfinedPath,
}

/// `args.package` と `artifact.json` の `onnx_file` を経路検証し、ONNX ファイルを開く。
///
/// `workspace` はカレントディレクトリ（CLI の規約。容量計測 example と同じ）。
///
/// # Errors
/// 経路の拒否は `invalid_input`（64）、`artifact.json` の上限超過は `limit_exceeded`（20）、
/// 不正な内容は `invalid_input`、I/O 起因・非対応 OS は `runtime_error`（70）。
pub fn guard_infer_paths(
    workspace: &Path,
    args: &InferArgs,
) -> Result<GuardedInferInputs, ErrorReport> {
    let package = confine_package(workspace, &args.package).map_err(|e| e.to_error_report())?;
    let (mut meta_file, _) = package
        .open_member(Path::new(ARTIFACT_META_FILE))
        .map_err(|e| e.to_error_report())?;
    let bytes = read_open_file_bounded(
        &mut meta_file,
        Path::new(ARTIFACT_META_FILE),
        MAX_ARTIFACT_META_BYTES,
    )
    .map_err(|e| e.to_error_report())?;
    let onnx_ref = ArtifactOnnxRef::parse(&bytes).map_err(|e| e.to_error_report())?;
    let (onnx, onnx_path) = package
        .open_member(Path::new(onnx_ref.onnx_file()))
        .map_err(|e| e.to_error_report())?;
    // 開いた fd のメタデータでサイズを確認する（パスから再取得しない。読み込み前の上限確認。REQ-39）。
    let onnx_len = onnx
        .metadata()
        .map_err(|_| ErrorReport::new(ExitCode::RuntimeError, "cannot read model metadata"))?
        .len();
    if onnx_len > MAX_MODEL_FILE_BYTES {
        return Err(ErrorReport::new(
            ExitCode::LimitExceeded,
            "model file exceeds size limit",
        ));
    }
    Ok(GuardedInferInputs {
        package,
        onnx,
        onnx_path,
    })
}
