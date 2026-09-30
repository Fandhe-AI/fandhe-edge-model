//! `package` 工程: 選定した候補から配布パッケージを組み立て、容量を計測する
//! （REQ-30・REQ-32・REQ-33・REQ-39・TASK-33.1-2・#136）。
//!
//! # 手順
//!
//! 1. `selection_record.json`（`select` の記録）から選定候補を読み、`request.json`・`result.json`
//!    を再検証つきで読み戻す（[`super::train::load_trained`]）。記録の `candidate_id`・添字の既定候補・
//!    学習リクエストの `kind` の一致も確認する
//! 2. 選定候補の学習ワーカー出力（`artifact.json`・ONNX ファイル）と登録済みの `definition.json`
//!    （選択肢表）を `package/` へ新規コピーする（既存の `package/` は拒否。上書きしない）
//! 3. `artifact.json` の `onnx_sha256` とコピーした ONNX の sha256 の一致、および `kind`・
//!    `label_order`・`max_bytes` の定義・選定候補との一致を確認する
//!    （パッケージの自己整合性。**外部台帳による完全性検証〔#168〕の代替ではない**）
//! 4. 容量を計測し（[`measure_opened_files_with_limit`]。REQ-30）、上限超過は `limit_exceeded`
//!
//! 2〜4 の途中で失敗した場合は、本工程が作った `package/` を削除する（再実行できなくなる半端な
//! パッケージを残さない）。容量の上限超過は成功扱いで `package/` を残す。
//!
//! # 未接続（実装済みを装わない）
//!
//! p95 の計測（REQ-31・`LimitBreach::Latency`）と合否基準は未接続。定義ファイルに合否基準の欄が
//! 無いため [`PackageQualityJudgment::NotDefined`] とし、`judgment:null`・
//! `acceptance_defined:false`・exit 0 を返す（`pass` は出さない）。

use std::path::{Component, Path};

use fandhe_edge_core::artifact_meta::{ArtifactMeta, MAX_ARTIFACT_META_BYTES};
use fandhe_edge_core::definition::{Definition, MAX_DEFINITION_FILE_BYTES};
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_guard::kind::KindAllowlist;
use fandhe_edge_runtime::capacity::{
    MAX_FILE_BYTES, PackageComponent, measure_opened_files_with_limit,
};
use fandhe_edge_runtime::onnx::{MAX_MAX_BYTES, MAX_MODEL_FILE_BYTES, MIN_MAX_BYTES};
use fandhe_edge_runtime::package_outcome::{
    LimitBreach, PackageOutcome, PackageQualityJudgment, resolve_package_outcome,
};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::stage_files::SelectionRecord;

use crate::args::PackageArgs;
use crate::error_report::ToErrorReport;
use crate::project::{
    DEFINITION_FILE, PACKAGE_DIR, Project, SELECTION_FILE, invalid, parse_definition, runtime,
};

use super::train::{load_trained, resolve_candidates};

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
    let definition_bytes = project.read(DEFINITION_FILE, MAX_DEFINITION_FILE_BYTES)?;
    let definition = parse_definition(&definition_bytes)?;
    // 選定記録の `candidate_index` だけで学習結果を読まず、その添字の既定候補の ID・kind が
    // 記録と一致することを確認する（別の候補を配布しない。REQ-27・REQ-39）。
    let candidates = resolve_candidates(&project, &definition, selection.candidate_index)?;
    let candidate = candidates
        .get(selection.candidate_index)
        .filter(|c| c.candidate_id == selection.candidate_id)
        .ok_or_else(|| invalid("selection record does not match the candidate"))?;
    let (request, outcome) = load_trained(&project, selection.candidate_index)?
        .ok_or_else(|| invalid("selected candidate is not trained"))?;
    if request.kind() != candidate.params.kind
        || request.kind_version() != candidate.params.kind_version
    {
        return Err(invalid("selection record does not match the candidate"));
    }
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
    let meta = ArtifactMeta::parse(&meta_bytes).map_err(|e| e.to_error_report())?;
    if meta.onnx_file() != onnx_file
        || meta.onnx_sha256() != Sha256Digest::of_bytes(&onnx_bytes).to_hex()
    {
        return Err(invalid("artifact metadata does not match the model file"));
    }
    check_meta_consistency(&meta, &definition, request.kind(), request.max_bytes())?;

    if project.exists(PACKAGE_DIR)? {
        return Err(invalid("package directory already exists"));
    }
    let package_dir = project.create_dir_tracked(PACKAGE_DIR)?;
    // 組み立て・容量計測のどこかで失敗したら、本工程が作った `package/` を片付ける。
    // 半端なパッケージが残ると再実行が「既存」で恒久的に拒否され、`infer --package` に
    // 誤った成果物として渡される恐れがあるため（best effort。容量の上限超過は成功扱いで残す）。
    let breakdown = match assemble_and_measure(
        &project,
        onnx_file,
        &meta_bytes,
        &onnx_bytes,
        &definition_bytes,
    ) {
        Ok(breakdown) => breakdown,
        Err(report) => {
            project.remove_created_dir(&package_dir);
            return Err(report);
        }
    };
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

/// `package/` へ 3 ファイルを新規に書き、閉じ込めつきで開いたハンドルで容量を計測する（REQ-30）。
///
/// 呼び出し元（[`run`]）は `package/` を作成済みで、失敗時の後始末は呼び出し元が行う。
fn assemble_and_measure(
    project: &Project,
    onnx_file: &str,
    meta_bytes: &[u8],
    onnx_bytes: &[u8],
    definition_bytes: &[u8],
) -> Result<fandhe_edge_runtime::capacity::CapacityBreakdown, ErrorReport> {
    let pkg = Path::new(PACKAGE_DIR);
    project.write_new(pkg.join(ARTIFACT_META_FILE), meta_bytes)?;
    project.write_new(pkg.join(onnx_file), onnx_bytes)?;
    project.write_new(pkg.join(DEFINITION_FILE), definition_bytes)?;

    let mut files = Vec::new();
    for (component, name) in [
        (PackageComponent::Weights, onnx_file),
        (PackageComponent::LabelTable, DEFINITION_FILE),
        (PackageComponent::Metadata, ARTIFACT_META_FILE),
    ] {
        let (file, path) = project.open_file(pkg.join(name))?;
        files.push((component, path, file));
    }
    measure_opened_files_with_limit(&files, MAX_FILE_BYTES)
        .map_err(|e| crate::output::capacity_error_report(&e))
}

/// `artifact.json` の `kind`・`label_order`・`max_bytes` を、選定候補の学習リクエストと定義に照合する
/// （`infer` が読み込み時に行う検査と同じ観点。食い違うパッケージを作らない。REQ-32・REQ-39）。
fn check_meta_consistency(
    meta: &ArtifactMeta,
    definition: &Definition,
    trained_kind: &str,
    trained_max_bytes: u32,
) -> Result<(), ErrorReport> {
    KindAllowlist::supported()
        .check(meta.kind())
        .map_err(|e| e.to_error_report())?;
    if meta.kind() != trained_kind {
        return Err(invalid(
            "artifact kind does not match the selected candidate",
        ));
    }
    let option_ids = definition.options().iter().map(|c| c.id.as_str());
    if !meta.label_order().iter().map(String::as_str).eq(option_ids) {
        return Err(invalid("package label order does not match definition"));
    }
    if meta.max_bytes() != trained_max_bytes {
        return Err(invalid(
            "artifact max_bytes does not match the selected candidate",
        ));
    }
    let in_range = usize::try_from(meta.max_bytes())
        .is_ok_and(|n| (MIN_MAX_BYTES..=MAX_MAX_BYTES).contains(&n));
    if !in_range {
        return Err(invalid("package max_bytes is out of range"));
    }
    Ok(())
}

/// 単一の通常の名前（区切り・`..`・絶対パスを含まない）か。
fn is_single_component(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    )
}
