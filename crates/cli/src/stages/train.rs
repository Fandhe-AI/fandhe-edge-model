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
//! 学習ワーカーへ渡す validation は `id` と `input` のみ（正解ラベルは渡さない。REQ-27）。
//! 失敗した候補の `candidates/<N>/` は残る（チェックポイントからの再開は提供しない。REQ-34）。
//!
//! # 学習ワーカーの発見（暫定）
//!
//! 環境変数 [`TRAINER_DIR_ENV`]（絶対パスのみ）。未設定なら開発ツリーの `trainer/`
//! （ビルド時の `CARGO_MANIFEST_DIR` 起点）。配布形態は未確定のため暫定（オーナー確認事項）。

use std::path::{Path, PathBuf};

use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::TrainReport;
use fandhe_edge_data::split::Split;
use fandhe_edge_data::split_record::SplitRecord;
use fandhe_edge_train::kind_resolution::{CommonTrainParams, resolve_kind_candidates};
use fandhe_edge_train::limits::{MAX_REQUEST_BYTES, MAX_RESULT_BYTES_WITH_VALIDATION};
use fandhe_edge_train::process::{RunLimits, WorkerLauncher, run_train};
use fandhe_edge_train::request::{
    Device, TrainRequest, ValidationInput, label_order_from_definition,
};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::search::SearchCandidate;
use fandhe_edge_train::stage_files::{outcome_json_vec, trainer_jsonl};

use super::inspect::split_rows;
use crate::args::TrainArgs;
use crate::error_report::{ToErrorReport, train_outcome_error_report};
use crate::project::{
    CANDIDATES_DIR, DEFAULT_MAX_BYTES, JOB_DIR, MODEL_DIR, Project, REQUEST_FILE, RESULT_FILE,
    SPLIT_FILE, TRAIN_INPUT_FILE, TRAIN_SEED, invalid, runtime,
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
            seed: TRAIN_SEED,
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
    let definition = project.load_definition()?;
    let records = project.load_records(&definition)?;
    let rows = split_rows(&records)?;

    let split_bytes = project.read(SPLIT_FILE, crate::project::MAX_PROJECT_FILE_BYTES)?;
    let split_text =
        std::str::from_utf8(&split_bytes).map_err(|_| invalid("split record is invalid"))?;
    let split_record =
        SplitRecord::from_json_str(split_text).map_err(|_| invalid("split record is invalid"))?;
    let split = split_record
        .verify_against(&rows)
        .map_err(|_| invalid("split record does not match the data"))?;

    let mut candidates = resolve_candidates(&project, &definition, args.candidate)?;
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
        .map_err(|_| runtime("cannot build training data"))?;
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
    if !project.exists(CANDIDATES_DIR) {
        project.create_dir(CANDIDATES_DIR)?;
    }
    project.create_dir(&rel)?;
    let job_dir = project.create_dir(rel.join(JOB_DIR))?;
    project.write_new(rel.join(TRAIN_INPUT_FILE), &train_jsonl)?;
    project.write_new(rel.join(REQUEST_FILE), &request_json)?;

    let run = run_train(
        &launcher,
        &request,
        &job_dir,
        &RunLimits::for_request(&request),
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
    project.write_new(rel.join(RESULT_FILE), &result_json)?;
    Ok(TrainReport::new(args.candidate, candidate.candidate_id))
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
    let outcome = TrainOutcome::from_worker_stdout(&result_bytes, &request)
        .map_err(|e| e.to_error_report())?;
    Ok(Some((request, outcome)))
}
