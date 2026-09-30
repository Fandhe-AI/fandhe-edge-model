//! `package` 工程: 選定した候補から配布パッケージを組み立て、容量を計測する
//! （REQ-30・REQ-32・REQ-33・REQ-39・TASK-33.1-2・#136）。
//!
//! # 手順
//!
//! 0. 開始時（ステージングを作る前）に評価データの凍結ハッシュを確認する（不一致・凍結記録の欠落は
//!    `invalid_input` で停止。REQ-17）。評価データがあるプロジェクトは、評価の完了記録が無い限り
//!    `invalid_input`（`evaluation has not been completed`）で拒否する。完了記録は未実装のため当面は
//!    常に拒否する（#314 で evaluate を評価器へ接続し、完了記録を本工程が確認する。REQ-24〜27）
//! 1. `selection_record.json`（`select` の記録）から選定候補を読み、`request.json`・`result.json`
//!    を再検証つきで読み戻す（[`super::train::load_trained`]）。記録の `candidate_id`・添字の既定候補・
//!    学習リクエストの `kind` の一致も確認する
//! 2. 選定候補の学習ワーカー出力（`artifact.json`・ONNX ファイル）と登録済みの `definition.json`
//!    （選択肢表）をステージングへ新規コピーする（既存の `package/` は拒否。上書きしない）
//! 3. `artifact.json` の `onnx_sha256` とコピーした ONNX の sha256 の一致、および `kind`・
//!    `label_order`・`max_bytes` の定義・選定候補との一致を確認する
//!    （パッケージの自己整合性。**外部台帳による完全性検証〔#168〕の代替ではない**）
//!    あわせて、公開前に `infer` と同じ検証（ガード層の形式許可リスト・ONNX の読み込み。
//!    [`super::infer::load_backend`]）を通す
//!    `train --smoke` の結果は `--allow-smoke`（検証専用）が無ければ拒否する（REQ-27）
//! 4. 容量を計測し（[`measure_opened_files_with_limit`]。REQ-30）、上限超過は `limit_exceeded`
//!
//! 2〜4 は `package.staging/` で行い、容量が上限内のときだけ `package/` へ原子的に名前替えして
//! 公開する。途中の失敗・容量の上限超過ではステージングを片付け、`package/` を作らない
//! （推論可能な場所に半端・超過のパッケージを残さない。既存の `package/` は事前に拒否し、
//! 置き換えも削除もしない）。容量は計測して上限照合（`limit_exceeded` の判定）に使う。
//!
//! # 未接続（実装済みを装わない）
//!
//! 容量内訳の stdout 出力は未接続。現状の出力は `PackageOutcome` 由来の JSON（成功時。`judgment`・`acceptance_defined` のみ）または
//! `{"code","message"}`（超過時）で、内訳は載らない。内訳の JSON 部品は
//! `output::package_capacity_json`（#123）にあり、接続は入出力契約（REQ-33）の変更を伴うため別途扱う。
//!
//! p95 の計測（REQ-31・`LimitBreach::Latency`）と合否基準は未接続。定義ファイルに合否基準の欄が
//! 無いため [`PackageQualityJudgment::NotDefined`] とし、`judgment:null`・
//! `acceptance_defined:false`・exit 0 を返す（`pass` は出さない）。

use std::path::{Component, Path};

use fandhe_edge_core::artifact_meta::{ArtifactMeta, MAX_ARTIFACT_META_BYTES};
use fandhe_edge_core::definition::{Definition, MAX_DEFINITION_FILE_BYTES};
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_core::fs::read_bounded_open_file;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_guard::format::{FormatAllowlist, check_bytes};
use fandhe_edge_guard::kind::KindAllowlist;
use fandhe_edge_runtime::capacity::{
    MAX_FILE_BYTES, PackageComponent, measure_opened_files_with_limit,
};
use fandhe_edge_runtime::onnx::{MAX_MAX_BYTES, MAX_MODEL_FILE_BYTES, MIN_MAX_BYTES, ModelKind};
use fandhe_edge_runtime::package_outcome::{
    LimitBreach, PackageOutcome, PackageQualityJudgment, resolve_package_outcome,
};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::stage_files::SelectionRecord;

use crate::args::PackageArgs;
use crate::error_report::ToErrorReport;
use crate::project::{
    CreatedDir, DEFINITION_FILE, PACKAGE_DIR, PACKAGE_STAGING_DIR, Project, SELECTION_FILE,
    fs_report, invalid, parse_definition,
};

use super::infer::load_backend;
use super::select::compute_selection;
use super::train::{
    candidate_rel, load_trained, request_is_smoke_trained, request_matches_candidate,
    resolve_candidates, verified_split,
};

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
    // 副作用（ステージングの作成など）の前に、評価データが凍結記録どおりか確認し（REQ-17）、
    // 評価データがあるプロジェクトは評価の完了記録が無い限り公開を拒否する（fail-closed。REQ-27）。
    // TODO(#314・REQ-24〜27): evaluate 工程を評価器へ接続して評価の完了記録を残し、ここでその記録を
    // 確認する。完了記録の形式がまだ無いため、当面は評価データがあれば常に拒否する（実装済みを装わない）。
    if super::inspect::load_evaluation_bytes(&project)?.is_some() {
        return Err(invalid("evaluation has not been completed"));
    }
    let selection_bytes = project.read(SELECTION_FILE, 64 * 1024)?;
    let selection = SelectionRecord::from_json_slice(&selection_bytes)
        .map_err(|_| invalid("selection record is invalid"))?;
    let definition_bytes = project.read(DEFINITION_FILE, MAX_DEFINITION_FILE_BYTES)?;
    let definition = parse_definition(&definition_bytes)?;
    // 選定記録の `candidate_index` だけで学習結果を読まず、その添字の既定候補の ID・kind が
    // 記録と一致することを確認する（別の候補を配布しない。REQ-27・REQ-39）。
    // 記録された選定結果を信じず、`select` と同じ関数で保存済みの候補結果・分割データから
    // 選定をやり直し、記録と完全一致することを確認する（記録の改変対策。validation のみを使い、
    // 凍結 test には触れない。REQ-27）。
    if compute_selection(&project, &definition)?.as_ref() != Some(&selection) {
        return Err(invalid("selection record does not match the candidate"));
    }
    // 期待する seed・validation 入力は固定値ではなく `split.json` の記録とデータから求める
    // （`compute_selection` が分割を検証済みだが、期待値の組み立てのため同じ検証をもう一度通す）。
    let records = project.load_records(&definition)?;
    let (split, seed) = verified_split(&project, &records)?;
    let candidates = resolve_candidates(&project, &definition, selection.candidate_index, seed)?;
    let candidate = candidates
        .get(selection.candidate_index)
        .filter(|c| c.candidate_id == selection.candidate_id)
        .ok_or_else(|| invalid("selection record does not match the candidate"))?;
    let (request, outcome) = load_trained(&project, selection.candidate_index)?
        .ok_or_else(|| invalid("selected candidate is not trained"))?;
    if !request_matches_candidate(&request, &candidate.params, &records, &split) {
        return Err(invalid("selection record does not match the candidate"));
    }
    // 短縮学習（`train --smoke`）の結果は、検証専用の `--allow-smoke` を明示しない限り配布しない
    // （`select` は smoke の結果も選定できるため、ここが配布の関門。REQ-27）。
    if request_is_smoke_trained(&request, &candidate.params) && !args.allow_smoke {
        return Err(invalid("smoke-trained candidate cannot be packaged"));
    }
    let TrainOutcome::Ok(success) = &outcome else {
        return Err(invalid("selected candidate has no artifact"));
    };
    // 成果物は選定候補のディレクトリ（`candidates/<N>/`）配下から読む。候補ディレクトリは `Project` の
    // 保持 fd からの相対オープン（`O_NOFOLLOW`・`O_DIRECTORY`）で得る（パスの正規化・開き直しをしない）。
    // `artifact_dir` は文字列の前方一致でなく、閉じ込め済みの候補ディレクトリからの相対パスとして求める（リクエスト・結果の差し替えで候補の外を読まない。REQ-39）。
    let candidate_dir = project.open_subdir(candidate_rel(selection.candidate_index))?;
    let artifact_rel = Path::new(success.artifact_dir())
        .strip_prefix(candidate_dir.dir())
        .map_err(|_| invalid("artifact directory is outside the candidate directory"))?
        .to_path_buf();
    let onnx_file = success.artifact().onnx_file();
    if !is_single_component(onnx_file) {
        return Err(invalid("onnx file name is invalid"));
    }
    let read_member = |name: &str, limit: u64| -> Result<Vec<u8>, ErrorReport> {
        let (file, real) = candidate_dir
            .open_member(&artifact_rel.join(name))
            .map_err(|e| e.to_error_report())?;
        read_bounded_open_file(file, real.as_path(), limit).map_err(|e| fs_report(&e))
    };
    let meta_bytes = read_member(ARTIFACT_META_FILE, MAX_ARTIFACT_META_BYTES)?;
    let onnx_bytes = read_member(onnx_file, MAX_MODEL_FILE_BYTES)?;
    let meta = ArtifactMeta::parse(&meta_bytes).map_err(|e| e.to_error_report())?;
    if meta.onnx_file() != onnx_file
        || meta.onnx_sha256() != Sha256Digest::of_bytes(&onnx_bytes).to_hex()
    {
        return Err(invalid("artifact metadata does not match the model file"));
    }
    check_meta_consistency(
        &meta,
        &definition,
        request.kind(),
        request.kind_version(),
        request.max_bytes(),
    )?;
    // 公開前に、`infer` が同じパッケージを読むときと同じ検証を通す（ハッシュが一致しても ONNX として
    // 読めないファイルを公開しない）。形式の許可リスト（ガード層）→ `kind_version` の許可リスト・
    // ONNX の読み込み・出力サイズの一致（`infer` の `load_backend` と共通。REQ-39）。
    check_bytes(onnx_bytes.clone(), &FormatAllowlist::onnx_only())
        .map_err(|e| e.to_error_report())?;
    let kind = ModelKind::parse(meta.kind()).map_err(|_| invalid("unsupported model kind"))?;
    load_backend(
        &onnx_bytes,
        kind,
        meta.kind_version(),
        definition.options().len(),
    )?;

    if project.exists(PACKAGE_DIR)? {
        return Err(invalid("package directory already exists"));
    }
    // 組み立て・容量計測はステージングで行い、上限内のときだけ `package/` へ原子的に公開する
    // （容量超過のパッケージを `infer --package` で使える場所に残さない。既存の `package/` は
    // 事前に拒否済みで、置き換えも削除もしない。REQ-30・REQ-39）。
    let staging = project.create_dir_tracked(PACKAGE_STAGING_DIR)?;
    // 組み立て・容量計測のどこかで失敗したら、本工程が作ったステージングを片付ける（best effort）。
    // 半端なステージングが残ると再実行が「既存」で拒否されるため。
    let breakdown = match assemble_and_measure(
        &project,
        onnx_file,
        &meta_bytes,
        &onnx_bytes,
        &definition_bytes,
    ) {
        Ok(breakdown) => breakdown,
        Err(report) => {
            let _ = project.remove_created_dir(&staging);
            return Err(report);
        }
    };
    let breaches = finalize_staging(
        &project,
        &staging,
        breakdown.total_bytes(),
        CAPACITY_LIMIT_BYTES,
    )?;
    Ok(resolve_package_outcome(
        &breaches,
        PackageQualityJudgment::NotDefined,
    ))
}

/// 計測した容量が上限内ならステージングを `package/` へ原子的に公開し、超過なら公開せず片付ける
/// （REQ-30・REQ-39）。公開に失敗した場合もステージングを片付けてエラーを返す。
///
/// 上限を超えた場合は [`LimitBreach::Capacity`] を返し、`package/` は作らない。既存の `package/` は
/// 触らない（呼び出し元が事前に不在を確認済み。公開は `RENAME_NOREPLACE` 相当で置き換えない）。
fn finalize_staging(
    project: &Project,
    staging: &CreatedDir,
    measured_bytes: u64,
    limit_bytes: u64,
) -> Result<Vec<LimitBreach>, ErrorReport> {
    if measured_bytes > limit_bytes {
        let _ = project.remove_created_dir(staging);
        return Ok(vec![LimitBreach::Capacity {
            measured_bytes,
            limit_bytes,
        }]);
    }
    if let Err(report) = project.publish_dir(PACKAGE_STAGING_DIR, PACKAGE_DIR) {
        let _ = project.remove_created_dir(staging);
        return Err(report);
    }
    Ok(Vec::new())
}

/// ステージングへ 3 ファイルを新規に書き、閉じ込めつきで開いたハンドルで容量を計測する（REQ-30）。
///
/// 呼び出し元（[`run`]）はステージングを作成済みで、失敗時の後始末は呼び出し元が行う。
fn assemble_and_measure(
    project: &Project,
    onnx_file: &str,
    meta_bytes: &[u8],
    onnx_bytes: &[u8],
    definition_bytes: &[u8],
) -> Result<fandhe_edge_runtime::capacity::CapacityBreakdown, ErrorReport> {
    let pkg = Path::new(PACKAGE_STAGING_DIR);
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
    trained_kind_version: u32,
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
    if meta.kind_version() != trained_kind_version {
        return Err(invalid(
            "artifact kind_version does not match the selected candidate",
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// 一時 cwd の下に `proj/` を新規作成し、ステージングにファイルを 1 つ置いて返す。
    fn setup(case: &str) -> (std::path::PathBuf, Project, CreatedDir) {
        let cwd =
            std::env::temp_dir().join(format!("package-staging-{case}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cwd);
        std::fs::create_dir_all(&cwd).expect("cwd");
        let project = Project::create(&cwd, Path::new("proj")).expect("project");
        let staging = project
            .create_dir_tracked(PACKAGE_STAGING_DIR)
            .expect("staging");
        project
            .write_new(Path::new(PACKAGE_STAGING_DIR).join("artifact.json"), b"new")
            .expect("write");
        (cwd, project, staging)
    }

    /// REQ-30・REQ-39: 上限超過では `package/` を作らず、ステージングも残さない。
    #[test]
    fn req30_over_limit_leaves_no_package_and_no_staging() {
        let (cwd, project, staging) = setup("over");
        let breaches = finalize_staging(&project, &staging, 41, 40).expect("finalize");
        assert_eq!(
            breaches,
            vec![LimitBreach::Capacity {
                measured_bytes: 41,
                limit_bytes: 40
            }]
        );
        assert!(!project.exists(PACKAGE_DIR).expect("exists"));
        assert!(!project.exists(PACKAGE_STAGING_DIR).expect("exists"));
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-30・REQ-39: 上限ちょうど（超過でない）なら `package/` として公開され、ステージングは消える。
    #[test]
    fn req30_within_limit_publishes_package() {
        let (cwd, project, staging) = setup("within");
        let breaches = finalize_staging(&project, &staging, 40, 40).expect("finalize");
        assert_eq!(breaches, Vec::new());
        assert!(!project.exists(PACKAGE_STAGING_DIR).expect("exists"));
        let bytes = project
            .read(Path::new(PACKAGE_DIR).join("artifact.json"), 16)
            .expect("read published");
        assert_eq!(bytes, b"new");
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-30・REQ-39: 超過時に以前公開済みの `package/` があっても消さず・置き換えない。
    #[test]
    fn req30_over_limit_keeps_previously_published_package() {
        let (cwd, project, staging) = setup("keep");
        // 以前の公開物（ステージングの名前替えより前に存在した状態を模擬）。
        project.create_dir(PACKAGE_DIR).expect("old package");
        project
            .write_new(Path::new(PACKAGE_DIR).join("artifact.json"), b"old")
            .expect("old file");
        let breaches = finalize_staging(&project, &staging, 41, 40).expect("finalize");
        assert_eq!(breaches.len(), 1);
        let bytes = project
            .read(Path::new(PACKAGE_DIR).join("artifact.json"), 16)
            .expect("read old");
        assert_eq!(bytes, b"old");
        assert!(!project.exists(PACKAGE_STAGING_DIR).expect("exists"));
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-39: 公開先が既にあれば（空でも）置き換えず、ステージングを片付けて `invalid_input`。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn req39_publish_does_not_replace_existing_package() {
        let (cwd, project, staging) = setup("noreplace");
        project.create_dir(PACKAGE_DIR).expect("empty package");
        let err = finalize_staging(&project, &staging, 1, 40).expect_err("must not replace");
        assert_eq!(err.code, fandhe_edge_core::exitcode::ExitCode::InvalidInput);
        assert!(!project.exists(PACKAGE_STAGING_DIR).expect("exists"));
        let _ = std::fs::remove_dir_all(&cwd);
    }
}
