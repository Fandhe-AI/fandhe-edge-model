//! `train` 工程: 候補 1 件を学習ワーカー（子プロセス）で学習する（REQ-18・REQ-19・REQ-27・
//! REQ-33・REQ-34・REQ-39・TASK-33.1-2・#136）。
//!
//! # 手順
//!
//! 1. `split.json`（`inspect` の記録）で分割を再現・検証する（不一致は停止。REQ-17）
//! 2. `kind` を省略した既定候補（[`resolve_kind_candidates`]。暫定・オーナー確認前）の
//!    `--candidate` 番目を選ぶ（範囲外は `invalid_input`）
//! 3. `candidates/<N>/` に train 分割の `train_input.jsonl`（trainer 形式）・学習リクエスト
//!    `request.json` を置き、[`run_train`] で子プロセスを起動する（固定 argv・シェル非経由・
//!    許可リストの環境のみ・壁時計タイムアウトつき。REQ-39）
//! 4. 結果を `result.json` へ保存する（`select` が再検証つきで読み戻す）
//!
//! 開始時（副作用の前）に評価データの凍結ハッシュを確認し、不一致・凍結記録の欠落は `invalid_input` で停止する
//! （[`super::inspect::ensure_evaluation_frozen`]。REQ-17）。
//!
//! 学習ワーカーへ渡す validation は `id` と `input` のみ（正解ラベルは渡さない。REQ-27）。
//! 作成後に失敗した場合は、その呼び出しで作った `candidates/<N>/` だけを保持 fd 起点で片付ける
//! （同じ `--candidate` を再試行できる。チェックポイントからの再開は提供しない。REQ-34）。
//!
//! # 学習ワーカーの発見（暫定）
//!
//! 環境変数 [`TRAINER_DIR_ENV`]（絶対パスのみ）。未設定なら開発ツリーの `trainer/`
//! （ビルド時の `CARGO_MANIFEST_DIR` 起点）。配布形態は未確定のため暫定（オーナー確認事項）。

use std::path::{Path, PathBuf};

use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::TrainReport;
use fandhe_edge_data::inspect::ValidRecord;
use fandhe_edge_data::split::{Split, SplitResult};
use fandhe_edge_data::split_record::SplitRecord;
use fandhe_edge_train::kind_resolution::{CommonTrainParams, resolve_kind_candidates};
use fandhe_edge_train::limits::{MAX_REQUEST_BYTES, MAX_RESULT_BYTES_WITH_VALIDATION};
use fandhe_edge_train::process::{RunLimits, WorkerLauncher, run_train};
use fandhe_edge_train::request::{
    Device, TrainRequest, TrainRequestParams, ValidationInput, label_order_from_definition,
};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::search::SearchCandidate;
use fandhe_edge_train::stage_files::{StageFileError, outcome_json_vec, trainer_jsonl};

use super::inspect::split_rows;
use crate::args::TrainArgs;
use crate::error_report::{ToErrorReport, train_outcome_error_report};
use crate::project::{
    CANDIDATES_DIR, DEFAULT_MAX_BYTES, JOB_DIR, MODEL_DIR, Project, REQUEST_FILE, RESULT_FILE,
    SPLIT_FILE, TRAIN_INPUT_FILE, fail, invalid, runtime,
};

/// 学習ワーカーのディレクトリ（`launch.py` と `.venv`）を指す環境変数。絶対パスのみ受理する。
pub const TRAINER_DIR_ENV: &str = "FANDHE_EDGE_TRAINER_DIR";

/// 環境変数がなければ使う開発ツリーの `trainer/`（暫定）。
const DEFAULT_TRAINER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../trainer");

/// 学習ワーカーのランチャーを作る。
///
/// # Errors
/// 環境変数が絶対パスでない場合は `invalid_input`、ランチャーの検証失敗は下位層の写像。
pub fn worker_launcher() -> Result<WorkerLauncher, ErrorReport> {
    let dir: PathBuf = match std::env::var_os(TRAINER_DIR_ENV) {
        Some(v) => {
            let p = PathBuf::from(v);
            if !p.is_absolute() {
                return Err(invalid("trainer directory must be an absolute path"));
            }
            p
        }
        None => std::fs::canonicalize(DEFAULT_TRAINER_DIR)
            .map_err(|_| runtime("trainer directory is not available"))?,
    };
    WorkerLauncher::from_trainer_dir(&dir).map_err(|e| e.to_error_report())
}

/// 候補ディレクトリのプロジェクト内の相対パス。
#[must_use]
pub fn candidate_rel(index: usize) -> PathBuf {
    Path::new(CANDIDATES_DIR).join(index.to_string())
}

/// 既定候補の一覧を解決する（`root` は候補ディレクトリの絶対パス。文字列としてのみ使う）。
///
/// # Errors
/// 定義・パラメータの検査失敗は `invalid_input`。
pub fn resolve_candidates(
    project: &Project,
    definition: &Definition,
    index: usize,
    seed: u32,
) -> Result<Vec<SearchCandidate>, ErrorReport> {
    let label_order = label_order_from_definition(definition)
        .map_err(|e| e.to_error_report())?
        .into_vec();
    let root = project
        .path(candidate_rel(index))
        .to_str()
        .map(str::to_string)
        .ok_or_else(|| invalid("project path is not valid UTF-8"))?;
    let resolution = resolve_kind_candidates(
        None,
        CommonTrainParams {
            label_order,
            max_bytes: DEFAULT_MAX_BYTES,
            seed,
            device: Device::Cpu,
            root,
            train_path: TRAIN_INPUT_FILE.to_string(),
            out_dir: MODEL_DIR.to_string(),
            time_limit_seconds: None,
            rss_limit_bytes: None,
        },
    )
    .map_err(|e| {
        ErrorReport::new(
            e.exit_code(),
            format!("kind resolution failed: {}", e.reason_code()),
        )
    })?;
    Ok(resolution.into_candidates())
}

/// `train` を実行する。
///
/// # Errors
/// 前提（`inspect` 済み）の欠落・候補の範囲外・既存の候補ディレクトリは `invalid_input`（64）、
/// ワーカーの失敗は結果の失敗コードに応じた終了コード、I/O 失敗は `runtime_error`（70）。
pub fn run(args: &TrainArgs, cwd: &Path) -> Result<TrainReport, ErrorReport> {
    let project = Project::open(cwd, &args.project_dir)?;
    // 副作用（学習・選定・書き出し）の前に、評価データが凍結記録どおりか確認する（REQ-17）。
    super::inspect::ensure_evaluation_frozen(&project)?;
    let definition = project.load_definition()?;
    let records = project.load_records(&definition)?;
    let (split, seed) = verified_split(&project, &records)?;

    let mut candidates = resolve_candidates(&project, &definition, args.candidate, seed)?;
    if args.candidate >= candidates.len() {
        return Err(invalid("candidate index is out of range"));
    }
    let mut candidate = candidates.swap_remove(args.candidate);
    if args.smoke {
        // 動作確認用の軽量実行（エポック数のみ 1 へ）。全種類の既定設定が `epochs` を持つ。
        candidate
            .params
            .config
            .insert("epochs".to_string(), 1.into());
    }

    let train_rows = records
        .iter()
        .filter(|r| split.by_record.get(&r.id) == Some(&Split::Train));
    let train_jsonl = trainer_jsonl(train_rows.map(|r| (r.input.as_str(), r.label_id.as_str())))
        .map_err(|e| stage_file_error_report(e, "cannot build training data"))?;
    let validation: Vec<ValidationInput> = records
        .iter()
        .filter(|r| split.by_record.get(&r.id) == Some(&Split::Validation))
        .map(|r| ValidationInput::new(r.id.clone(), r.input.clone()))
        .collect();
    let request = TrainRequest::new(candidate.params)
        .and_then(|r| r.with_validation_inputs(validation))
        .map_err(|e| e.to_error_report())?;
    let request_json = request.to_json_vec().map_err(|e| e.to_error_report())?;
    let launcher = worker_launcher()?;

    let rel = candidate_rel(args.candidate);
    if !project.exists(CANDIDATES_DIR)? {
        project.create_dir(CANDIDATES_DIR)?;
    }
    // 以降の失敗では、今回作った候補ディレクトリだけを片付けてから返す（同じ `--candidate` を
    // 再試行できるようにする。名前替えの公開方式は使わない: 結果の `artifact_dir` は絶対パスで
    // 記録されるため、移動すると記録と実体がずれる。REQ-34: 再開は提供せず、やり直しは新規）。
    let created = project.create_dir_tracked(&rel)?;
    let trained = train_in_candidate_dir(
        &project,
        &rel,
        &launcher,
        &request,
        &train_jsonl,
        &request_json,
    );
    match trained {
        Ok(()) => Ok(TrainReport::new(args.candidate, candidate.candidate_id)),
        Err(mut report) => {
            if !project.remove_created_dir(&created) {
                // 元のエラーの終了コードは変えず、残骸があることだけ固定文言で付記する。
                report
                    .message
                    .push_str("; candidate directory could not be cleaned up");
            }
            Err(report)
        }
    }
}

/// 作成済みの候補ディレクトリへ学習入力を置き、学習ワーカーを実行して結果を保存する。
/// 失敗時の後始末は呼び出し元（[`run`]）が行う。
fn train_in_candidate_dir(
    project: &Project,
    rel: &Path,
    launcher: &WorkerLauncher,
    request: &TrainRequest,
    train_jsonl: &[u8],
    request_json: &[u8],
) -> Result<(), ErrorReport> {
    let job_dir = project.create_dir(rel.join(JOB_DIR))?;
    project.write_new(rel.join(TRAIN_INPUT_FILE), train_jsonl)?;
    project.write_new(rel.join(REQUEST_FILE), request_json)?;

    let run = run_train(
        launcher,
        request,
        &job_dir,
        &RunLimits::for_request(request),
    )
    .map_err(|e| e.to_error_report())?;
    if let Some(report) = train_outcome_error_report(run.outcome()) {
        return Err(report);
    }
    let TrainOutcome::Ok(_) = run.outcome() else {
        return Err(ErrorReport::new(
            ExitCode::RuntimeError,
            "unexpected train outcome",
        ));
    };
    let result_json =
        outcome_json_vec(run.outcome()).map_err(|_| runtime("cannot serialize train result"))?;
    project.write_new(rel.join(RESULT_FILE), &result_json)
}

/// `split.json`（`inspect` の記録）を読み、取り込んだデータから分割を再現して照合する
/// （不一致は `invalid_input`。REQ-17・REQ-27）。`train`・`select` が同じ検証を通す。
///
/// 戻り値の seed は記録された値（`train` が学習リクエストの seed に使う）。
///
/// # Errors
/// 記録が読めない・不正・データと一致しない場合は `invalid_input`（64）等。
pub fn verified_split(
    project: &Project,
    records: &[ValidRecord],
) -> Result<(SplitResult, u32), ErrorReport> {
    let rows = split_rows(records)?;
    let split_record = read_split_record(project)?;
    let seed =
        u32::try_from(split_record.seed()).map_err(|_| invalid("split record is invalid"))?;
    let result = split_record
        .verify_against(&rows)
        .map_err(|_| invalid("split record does not match the data"))?;
    Ok((result, seed))
}

/// `split.json` を読んで解析する（検証はしない。`package` が記録済みの seed を取り出すのにも使う）。
///
/// # Errors
/// 記録が読めない・不正な場合は `invalid_input`（64）等。
pub fn read_split_record(project: &Project) -> Result<SplitRecord, ErrorReport> {
    let split_bytes = project.read(SPLIT_FILE, crate::project::MAX_PROJECT_FILE_BYTES)?;
    let split_text =
        std::str::from_utf8(&split_bytes).map_err(|_| invalid("split record is invalid"))?;
    SplitRecord::from_json_str(split_text).map_err(|_| invalid("split record is invalid"))
}

/// 学習済みの候補 `index` の学習リクエストと結果を読み戻す（学習済みでなければ `None`）。
///
/// 結果は保存されたファイルをそのまま信用せず、学習ワーカーの標準出力と同じ再検証
/// （[`TrainOutcome::from_worker_stdout`]。`artifact_dir` の閉じ込め・`kind`・`label_order` 等の
/// リクエストとの一致）を通す（REQ-39）。
///
/// # Errors
/// 保存ファイルが読めない・再検証に失敗した場合。
pub fn load_trained(
    project: &Project,
    index: usize,
) -> Result<Option<(TrainRequest, TrainOutcome)>, ErrorReport> {
    let rel = candidate_rel(index);
    let Some(result_bytes) = project.read_optional(
        rel.join(RESULT_FILE),
        MAX_RESULT_BYTES_WITH_VALIDATION as u64,
    )?
    else {
        return Ok(None);
    };
    let request_bytes = project.read(rel.join(REQUEST_FILE), MAX_REQUEST_BYTES as u64)?;
    let request = TrainRequest::from_json_slice(&request_bytes).map_err(|e| e.to_error_report())?;
    // 保存済みの結果の再検証失敗（`artifact_dir` が期待する場所と違う等）は、ワーカーの実行時エラー
    // ではなくプロジェクト内の記録の不整合・改ざんなので `invalid_input`（固定語彙。REQ-39）。
    let outcome = TrainOutcome::from_worker_stdout(&result_bytes, &request)
        .map_err(|_| invalid("stored train result is invalid"))?;
    Ok(Some((request, outcome)))
}

/// 保存済みの学習リクエストが、既定候補 `params` から作られたものと同じ種類・構成かを返す
/// （`select`・`package` が、別の種類の学習結果を候補 N として扱わないための照合。REQ-27・REQ-39）。
///
/// 比べるのは `kind`・`kind_version`・`label_order`・`max_bytes`・`seed`・`root`・`out_dir` と、
/// `epochs` を除く `config`。`epochs` は `train --smoke` が 1 へ上書きする唯一の項目のため除く。
///
/// `params` は `train` と同じ [`resolve_candidates`]（候補番号 N から `root`・`out_dir` を組み立てる
/// 唯一の関数）で作ったものでなければならない。`root`・`out_dir` を比べるのは、保存済みの
/// リクエストと結果の差し替えで `artifact_dir` が候補ディレクトリの外を指すのを防ぐため
/// （`load_trained` は保存済みリクエストを基準に `artifact_dir` を検証するため。REQ-39）。`root` は
/// 絶対パスで記録されるので、プロジェクトのディレクトリを移動すると一致せず拒否される
/// （fail-closed として許容する）。制限値（時間・メモリ）は比べない。
#[must_use]
pub fn request_matches_candidate(request: &TrainRequest, params: &TrainRequestParams) -> bool {
    request.kind() == params.kind
        && request.kind_version() == params.kind_version
        && request.label_order().as_slice() == params.label_order.as_slice()
        && request.max_bytes() == params.max_bytes
        && request.seed() == params.seed
        && request.root() == params.root
        && request.out_dir() == params.out_dir
        && request
            .config()
            .iter()
            .filter(|(k, _)| k.as_str() != "epochs")
            .eq(params.config.iter().filter(|(k, _)| k.as_str() != "epochs"))
}

/// 学習用データの生成失敗を [`ErrorReport`] にする。上限超過は `limit_exceeded`（20。REQ-39）で、
/// 学習ワーカーの起動より前に止まる。それ以外は `runtime_error`（`message` は固定語彙）。
fn stage_file_error_report(error: StageFileError, message: &str) -> ErrorReport {
    match error {
        StageFileError::LimitExceeded => {
            fail(ExitCode::LimitExceeded, "training data exceeds size limit")
        }
        _ => runtime(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39: 学習用データの上限超過は `limit_exceeded`（20）、それ以外の失敗は `runtime_error`（70）。
    #[test]
    fn req39_stage_file_limit_maps_to_limit_exceeded() {
        let report = stage_file_error_report(StageFileError::LimitExceeded, "x");
        assert_eq!(report.code, ExitCode::LimitExceeded);
        assert_eq!(report.message, "training data exceeds size limit");
        let report =
            stage_file_error_report(StageFileError::Serialize, "cannot build training data");
        assert_eq!(report.code, ExitCode::RuntimeError);
        assert_eq!(report.message, "cannot build training data");
    }
}
