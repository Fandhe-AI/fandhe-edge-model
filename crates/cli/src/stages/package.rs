//! `package` 工程: 選定した候補から配布パッケージを組み立て、容量を計測する
//! （REQ-30・REQ-32・REQ-33・REQ-39・TASK-33.1-2・#136）。
//!
//! # 手順
//!
//! 1. `selection_record.json`（`select` の記録）から選定候補を読み、`request.json`・`result.json`
//!    を再検証つきで読み戻す（[`super::train::load_trained`]）
//! 2. 選定候補の学習ワーカー出力（`artifact.json`・ONNX ファイル）と登録済みの `definition.json`
//!    （選択肢表）を `package/` へ新規コピーする（既存の `package/` は拒否。上書きしない）
//! 3. `artifact.json` の `onnx_sha256` とコピーした ONNX の sha256 の一致を確認する
//!    （パッケージの自己整合性。**外部台帳による完全性検証〔#168〕の代替ではない**）
//! 4. 容量を計測し（[`measure_opened_files_with_limit`]。REQ-30）、上限超過は `limit_exceeded`
//!
//! # 未接続（実装済みを装わない）
//!
//! p95 の計測（REQ-31・`LimitBreach::Latency`）と合否基準は未接続。定義ファイルに合否基準の欄が
//! 無いため [`PackageQualityJudgment::NotDefined`] とし、`judgment:null`・
//! `acceptance_defined:false`・exit 0 を返す（`pass` は出さない）。

use std::path::{Component, Path};

use fandhe_edge_core::artifact_meta::{ArtifactMeta, MAX_ARTIFACT_META_BYTES};
use fandhe_edge_core::definition::MAX_DEFINITION_FILE_BYTES;
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_runtime::capacity::{
    MAX_FILE_BYTES, PackageComponent, measure_opened_files_with_limit,
};
use fandhe_edge_runtime::onnx::MAX_MODEL_FILE_BYTES;
use fandhe_edge_runtime::package_outcome::{
    LimitBreach, PackageOutcome, PackageQualityJudgment, resolve_package_outcome,
};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::stage_files::SelectionRecord;

use crate::args::PackageArgs;
use crate::error_report::ToErrorReport;
use crate::project::{DEFINITION_FILE, PACKAGE_DIR, Project, SELECTION_FILE, invalid, runtime};

use super::train::load_trained;

/// 配布パッケージ内のメタデータのファイル名（`infer_guard` と同じ）。
const ARTIFACT_META_FILE: &str = "artifact.json";

/// 容量の上限（バイト。REQ-30 の目安 40MB。暫定の固定値）。
const CAPACITY_LIMIT_BYTES: u64 = 40_000_000;

/// `package` を実行する。
///
/// # Errors
/// 選定記録が無い・既存の `package/`・自己整合性の不一致は `invalid_input`（64）、
/// I/O 失敗は `runtime_error`（70）。容量の上限超過は [`PackageOutcome`]（`limit_exceeded`）。
pub fn run(args: &PackageArgs, cwd: &Path) -> Result<PackageOutcome, ErrorReport> {
    let project = Project::open(cwd, &args.project_dir)?;
    let selection_bytes = project.read(SELECTION_FILE, 64 * 1024)?;
    let selection = SelectionRecord::from_json_slice(&selection_bytes)
        .map_err(|_| invalid("selection record is invalid"))?;
    let (_, outcome) = load_trained(&project, selection.candidate_index)?
        .ok_or_else(|| invalid("selected candidate is not trained"))?;
    let TrainOutcome::Ok(success) = &outcome else {
        return Err(invalid("selected candidate has no artifact"));
    };
    // `artifact_dir` は再検証済みの絶対パス（候補の root 配下）。プロジェクト内の相対パスへ戻す。
    let artifact_rel = Path::new(success.artifact_dir())
        .strip_prefix(project.dir())
        .map_err(|_| runtime("artifact directory is outside the project"))?
        .to_path_buf();
    let onnx_file = success.artifact().onnx_file();
    if !is_single_component(onnx_file) {
        return Err(invalid("onnx file name is invalid"));
    }

    let meta_bytes = project.read(
        artifact_rel.join(ARTIFACT_META_FILE),
        MAX_ARTIFACT_META_BYTES,
    )?;
    let onnx_bytes = project.read(artifact_rel.join(onnx_file), MAX_MODEL_FILE_BYTES)?;
    let definition_bytes = project.read(DEFINITION_FILE, MAX_DEFINITION_FILE_BYTES)?;
    let meta = ArtifactMeta::parse(&meta_bytes).map_err(|e| e.to_error_report())?;
    if meta.onnx_file() != onnx_file
        || meta.onnx_sha256() != Sha256Digest::of_bytes(&onnx_bytes).to_hex()
    {
        return Err(invalid("artifact metadata does not match the model file"));
    }

    if project.exists(PACKAGE_DIR) {
        return Err(invalid("package directory already exists"));
    }
    project.create_dir(PACKAGE_DIR)?;
    let pkg = Path::new(PACKAGE_DIR);
    project.write_new(pkg.join(ARTIFACT_META_FILE), &meta_bytes)?;
    project.write_new(pkg.join(onnx_file), &onnx_bytes)?;
    project.write_new(pkg.join(DEFINITION_FILE), &definition_bytes)?;

    // 容量の計測は、書き出したパッケージのファイルを閉じ込めつきで開いたハンドルで行う（REQ-30）。
    let mut files = Vec::new();
    for (component, name) in [
        (PackageComponent::Weights, onnx_file),
        (PackageComponent::LabelTable, DEFINITION_FILE),
        (PackageComponent::Metadata, ARTIFACT_META_FILE),
    ] {
        let (file, path) = project.open_file(pkg.join(name))?;
        files.push((component, path, file));
    }
    let breakdown = measure_opened_files_with_limit(&files, MAX_FILE_BYTES)
        .map_err(|e| crate::output::capacity_error_report(&e))?;
    let mut breaches = Vec::new();
    if breakdown.total_bytes() > CAPACITY_LIMIT_BYTES {
        breaches.push(LimitBreach::Capacity {
            measured_bytes: breakdown.total_bytes(),
            limit_bytes: CAPACITY_LIMIT_BYTES,
        });
    }
    Ok(resolve_package_outcome(
        &breaches,
        PackageQualityJudgment::NotDefined,
    ))
}

/// 単一の通常の名前（区切り・`..`・絶対パスを含まない）か。
fn is_single_component(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    )
}
