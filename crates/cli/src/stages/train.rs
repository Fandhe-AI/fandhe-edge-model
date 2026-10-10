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
//!
//! # ジョブ記録とやり直し（REQ-34・TASK-34.2・TASK-34.3・#485）
//!
//! 学習ジョブは単発・`--all` とも [`TrainJob::run_recorded`] で実行し、`candidates/<N>/job/` に
//! `job.json`（状態）と `job.lock`（生存確認の flock。PID は使わない）を残す。ジョブが失敗で終わった
//! 単発の `train` は `candidates/<N>/` を片付けずに残し（[`status`] が報告できるように）、ジョブの
//! 開始前・成功後の失敗では、その呼び出しで作った `candidates/<N>/` だけを保持 fd 起点で片付ける。
//!
//! 開始時に `candidates/<N>/` が既にあれば [`discard_failed_candidate`] の規則で扱う: `result.json` あり・
//! `succeeded`・記録なしは `candidate directory already exists`、`running`・`cancelling` は `job is running`
//! （いずれも 64）、`failed`・`cancelled`（記録が `running` のまま所有者が消えたものは、状態確認が `failed`＋
//! `owner_lost` へ書き戻してから）は保持 fd 起点で丸ごと消して新規に学習する（チェックポイントからの再開は
//! 提供しない）。
//!
//! # `--status`（#485）
//!
//! [`status`] は記録のある候補のジョブ状態（[`read_job_status`] の結果をそのまま）とやり直し案内を返す。
//! 学習の副作用が無いため凍結の検査はしない。書き込みは `running` の残骸を `failed`＋`owner_lost` へ
//! 書き戻す `job.json` だけ（MCP では参照系だが、この書き戻しを伴う。REQ-36）。
//!
//! # `--all`（探索予算内の全候補。REQ-18・TASK-18.1・TASK-18.2・#482・#483）
//!
//! [`run_all`] は既定候補の全件を、学習ワーカー層の [`run_search`]（1 プロセス・1 時計で探索予算全体を
//! 追跡し、持ち時間は均等割り固定）と [`WorkerCandidateRunner`] で宣言順に学習する。探索・選定・予算到達の
//! 判定は学習ワーカー層のものをそのまま使い、本モジュールは再実装しない。候補 N の学習ジョブの直前に
//! `candidates/<N>/` を作って `train_input.jsonl`・`request.json`（配分した持ち時間入り）・`job/` を置き、
//! 探索後に `evaluated` の候補だけ `result.json` を書く。それ以外（予算到達・失敗・未着手）の候補ディレクトリは
//! 片付ける（選定対象にしない）。探索記録はプロジェクト直下の `search_record.json`（#483）。
//!
//! 終了コード: `evaluated` が 1 件以上なら 0。0 件で予算到達があれば `limit_exceeded`（20）、予算到達が
//! 無ければ宣言順で最初の失敗候補のワーカー失敗コードの写像（単発の `train` と同じ）。探索自体の失敗は
//! `runtime_error`（70。候補の準備・子プロセスの起動失敗は単発の `train` と同じ写像）。探索が完了していれば
//! 失敗時も `search_record.json` は残す。
//!
//! # 学習ワーカーの発見（暫定）
//!
//! 環境変数 [`TRAINER_DIR_ENV`]（絶対パスのみ）。未設定なら開発ツリーの `trainer/`
//! （ビルド時の `CARGO_MANIFEST_DIR` 起点）。配布形態は未確定のため暫定（オーナー確認事項）。

use std::path::{Path, PathBuf};

use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::{TrainAllCandidate, TrainAllReport, TrainReport};
use fandhe_edge_data::inspect::ValidRecord;
use fandhe_edge_data::preprocess_boundary::{EmptyInputConsistency, compare_empty_input_encodings};
use fandhe_edge_data::split::{Split, SplitResult};
use fandhe_edge_data::split_record::SplitRecord;
use fandhe_edge_train::error::TrainProcessError;
use fandhe_edge_train::job::{JobState, TrainJob};
use fandhe_edge_train::job_record::{JobRecordError, JobStatusReport, read_job_status, unix_now};
use fandhe_edge_train::kind_resolution::{CommonTrainParams, resolve_kind_candidates};
use fandhe_edge_train::limits::{MAX_REQUEST_BYTES, MAX_RESULT_BYTES_WITH_VALIDATION};
use fandhe_edge_train::process::{RunLimits, TrainRunEnd, WorkerCandidateRunner, WorkerLauncher};
use fandhe_edge_train::request::{
    Device, TrainRequest, TrainRequestParams, ValidationInput, label_order_from_definition,
};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::search::{
    CandidateSearchResult, SearchBudget, SearchCandidate, SearchError, SearchInput, SearchRecord,
    run_search,
};
use fandhe_edge_train::stage_files::{
    MAX_SEARCH_RECORD_BYTES, StageFileError, TrainStatusEntry, allotted_time_limit_seconds,
    outcome_json_vec, search_record_json_vec, trainer_jsonl,
};
use fandhe_edge_train::time_allotment::{
    CandidateRunner, CandidateTimeError, Clock, PerCandidatePolicy, SystemClock,
};

use super::inspect::split_rows;
use crate::args::TrainArgs;
use crate::error_report::{ToErrorReport, train_outcome_error_report};
use crate::project::{
    CANDIDATES_DIR, CreatedDir, DEFAULT_MAX_BYTES, JOB_DIR, MODEL_DIR, Project, REQUEST_FILE,
    RESULT_FILE, SEARCH_RECORD_FILE, SPLIT_FILE, TRAIN_INPUT_FILE, TRAIN_SEED_FILE, fail, invalid,
    runtime,
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

/// 分割の `train` に割り当てられた行だけを入力順に返す（`train` 工程の学習データと、`package` の
/// p95 計測入力が共有する。validation・test 分割と凍結した評価データは含めない。REQ-27）。
///
/// 判定は `Split::Train` との一致 1 箇所に限る（否定条件で書くと `Split::Test` が混入するため）。
pub(crate) fn train_rows<'a>(
    records: &'a [ValidRecord],
    split: &'a SplitResult,
) -> impl Iterator<Item = &'a ValidRecord> + 'a {
    records
        .iter()
        .filter(|r| split.by_record.get(&r.id) == Some(&Split::Train))
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

/// `train --candidate <index>` を実行する。
///
/// # Errors
/// 前提（`inspect` 済み）の欠落・候補の範囲外・やり直せない既存の候補ディレクトリ（モジュール doc
/// 「ジョブ記録とやり直し」）・既存の `search_record.json`（`train --all` 済み）は `invalid_input`（64）、
/// ワーカーの失敗は結果の失敗コードに応じた終了コード、I/O 失敗は `runtime_error`（70）。
pub fn run(args: &TrainArgs, index: usize, cwd: &Path) -> Result<TrainReport, ErrorReport> {
    let project = Project::open(cwd, &args.project_dir)?;
    // 副作用（学習・選定・書き出し）の前に、評価データが凍結記録どおりか確認する（REQ-17）。
    super::inspect::ensure_evaluation_frozen(&project)?;
    let definition = project.load_definition()?;
    let records = project.load_records(&definition)?;
    let (split, split_seed) = verified_split(&project, &records)?;
    // `train --all` 済みのプロジェクトへ予算外で候補を足さない（予算内と予算外で学習した候補が混ざって
    // 選定されるのを防ぐ。`--all` が既存の候補ディレクトリを拒否するのと対称。REQ-18・#482）。
    if project.exists(SEARCH_RECORD_FILE)? {
        return Err(invalid("search record already exists"));
    }
    // 学習 seed は既定で分割の seed。`--train-seed` は学習だけを上書きし、分割・凍結は変えない。
    // 上書き時は実際に使った値を `train_seed.txt` に記録し、下流（evaluate・select・package）は
    // [`effective_train_seed`] でそれを正とする（`request.json` の seed 改ざんは従来どおり照合で弾く）。
    let seed = args.train_seed.unwrap_or(split_seed);

    let mut candidates = resolve_candidates(&project, &definition, index, seed)?;
    if index >= candidates.len() {
        return Err(invalid("candidate index is out of range"));
    }
    let mut candidate = candidates.swap_remove(index);
    if args.smoke {
        apply_smoke(&mut candidate);
    }

    let train_jsonl = trainer_jsonl(
        train_rows(&records, &split).map(|r| (r.input.as_str(), r.label_id.as_str())),
    )
    .map_err(|e| stage_file_error_report(e, "cannot build training data"))?;
    let request = build_train_request(candidate.params, &records, &split)?;
    let launcher = worker_launcher()?;

    let rel = candidate_rel(index);
    if !project.exists(CANDIDATES_DIR)? {
        project.create_dir(CANDIDATES_DIR)?;
    }
    if project.exists(&rel)? {
        discard_failed_candidate(&project, index)?;
    }
    // ジョブの開始前・成功後の失敗では、今回作った候補ディレクトリだけを片付けてから返す（同じ
    // `--candidate` を再試行できるようにする。名前替えの公開方式は使わない: 結果の `artifact_dir` は
    // 絶対パスで記録されるため、移動すると記録と実体がずれる）。ジョブが失敗で終わった場合は
    // `--status` のために残す（次の `train` が [`discard_failed_candidate`] で消す。REQ-34）。
    let created = project.create_dir_tracked(&rel)?;
    let trained = train_in_candidate_dir(
        &project,
        &rel,
        &launcher,
        &request,
        &train_jsonl,
        args.train_seed,
    );
    match trained {
        Ok(()) => Ok(TrainReport::new(index, candidate.candidate_id)),
        Err(TrainFailure {
            mut report,
            job_failed,
        }) => {
            if !job_failed && !project.remove_created_dir(&created) {
                // 元のエラーの終了コードは変えず、残骸があることだけ固定文言で付記する。
                report
                    .message
                    .push_str("; candidate directory could not be cleaned up");
            }
            Err(report)
        }
    }
}

/// 開始時に既にある `candidates/<index>/` を、モジュール doc「ジョブ記録とやり直し」の規則で扱う
/// （やり直せるときだけ丸ごと消す。REQ-34・#485）。
///
/// 判定より先に候補ディレクトリの fd を保持し、削除は判定した実体と同一のときだけ行う（判定後に
/// 差し替えられた別のディレクトリを消さない。REQ-39）。
fn discard_failed_candidate(project: &Project, index: usize) -> Result<(), ErrorReport> {
    let rel = candidate_rel(index);
    let held = project.track_existing_dir(&rel)?;
    if project.exists(rel.join(RESULT_FILE))? {
        return Err(invalid("candidate directory already exists"));
    }
    match job_status(project, index)?.map(|job| job.state) {
        Some(JobState::Failed | JobState::Cancelled) => {}
        Some(JobState::Queued | JobState::Running | JobState::Cancelling) => {
            return Err(invalid("job is running"));
        }
        // `succeeded`（`result.json` の保存前に止まった等）・記録なしは、何が残っているか分からないため
        // 消さない（保守側）。
        Some(JobState::Succeeded) | None => {
            return Err(invalid("candidate directory already exists"));
        }
    }
    if project.remove_created_dir(&held) {
        Ok(())
    } else {
        Err(runtime("candidate directory could not be cleaned up"))
    }
}

/// 単発の `train` の失敗。`job_failed` はジョブ記録が失敗で終わった（候補ディレクトリを残す）か。
struct TrainFailure {
    report: ErrorReport,
    job_failed: bool,
}

impl TrainFailure {
    /// ジョブの開始前・成功後の失敗（候補ディレクトリは片付ける）。
    fn cleanup(report: ErrorReport) -> Self {
        Self {
            report,
            job_failed: false,
        }
    }

    /// ジョブが失敗で終わった（候補ディレクトリは `--status` のために残す）。
    fn job(report: ErrorReport) -> Self {
        Self {
            report,
            job_failed: true,
        }
    }
}

/// 作成済みの候補ディレクトリへ学習入力を置き、学習ジョブを記録つきで実行して結果を保存する。
/// 失敗時の後始末は呼び出し元（[`run`]）が [`TrainFailure`] に従って行う。
fn train_in_candidate_dir(
    project: &Project,
    rel: &Path,
    launcher: &WorkerLauncher,
    request: &TrainRequest,
    train_jsonl: &[u8],
    train_seed_override: Option<u32>,
) -> Result<(), TrainFailure> {
    let job_dir = prepare_candidate_dir(project, rel, request, train_jsonl, train_seed_override)
        .map_err(TrainFailure::cleanup)?;

    let outcome = run_recorded_job(launcher, request, &job_dir).map_err(|e| match e {
        RecordedJobError::Run(e) => TrainFailure::job(e.to_error_report()),
        RecordedJobError::Begin(e) | RecordedJobError::Record(e) => {
            TrainFailure::cleanup(e.to_error_report())
        }
    })?;
    if let Some(report) = train_outcome_error_report(&outcome) {
        return Err(TrainFailure::job(report));
    }
    let TrainOutcome::Ok(success) = &outcome else {
        return Err(TrainFailure::cleanup(runtime("unexpected train outcome")));
    };
    check_empty_input_preprocessing(success.empty_input_ids(), request.max_bytes())
        .map_err(TrainFailure::cleanup)?;
    let result_json = outcome_json_vec(&outcome)
        .map_err(|_| TrainFailure::cleanup(runtime("cannot serialize train result")))?;
    project
        .write_new(rel.join(RESULT_FILE), &result_json)
        .map_err(TrainFailure::cleanup)
}

/// [`run_recorded_job`] の失敗。
enum RecordedJobError {
    /// ジョブ記録を開始できなかった（子は起動していない・記録は残らない）。
    Begin(JobRecordError),
    /// ジョブが失敗した（終端記録は `failed`。書けなかった場合も次の状態確認が `owner_lost` を検出する）。
    Run(TrainProcessError),
    /// 学習は成功したが終端記録を書けなかった（記録と結果が食い違うため成功として扱わない）。
    Record(JobRecordError),
}

/// 学習ジョブを `job_dir` に記録しながら実行する（単発の `train` と `train --all` が共有する。REQ-34・
/// TASK-34.2・#485）。ワーカーが返した失敗（`TrainOutcome::Error`）は `Ok` で返す（呼び出し元が写す）。
fn run_recorded_job(
    launcher: &WorkerLauncher,
    request: &TrainRequest,
    job_dir: &Path,
) -> Result<TrainOutcome, RecordedJobError> {
    let recorded = TrainJob::new()
        .run_recorded(launcher, request, job_dir, &RunLimits::for_request(request))
        .map_err(RecordedJobError::Begin)?;
    let outcome = match recorded.run {
        Ok(TrainRunEnd::Completed(run)) => run.outcome().clone(),
        // キャンセルの口を持たないため起こらない。`run_train` と同じ写像にする（キャンセルは #484）。
        Ok(TrainRunEnd::Cancelled(_)) => {
            return Err(RecordedJobError::Run(TrainProcessError::Wait {
                kind: std::io::ErrorKind::Interrupted,
            }));
        }
        Err(e) => return Err(RecordedJobError::Run(e)),
    };
    match (recorded.record, &outcome) {
        (Err(e), TrainOutcome::Ok(_)) => Err(RecordedJobError::Record(e)),
        _ => Ok(outcome),
    }
}

/// 候補 `index` のジョブ記録を確認する（記録が無ければ `None`）。経路は保持 fd 起点の `open_subdir`
/// （`O_NOFOLLOW`）で検証してから、学習ワーカー層の [`read_job_status`] に渡す（REQ-34・REQ-39）。
fn job_status(project: &Project, index: usize) -> Result<Option<JobStatusReport>, ErrorReport> {
    let rel = candidate_rel(index).join(JOB_DIR);
    if !project.exists(&rel)? {
        return Ok(None);
    }
    let job_dir = project.open_subdir(&rel)?;
    match read_job_status(job_dir.dir(), unix_now()) {
        Ok(report) => Ok(Some(report)),
        // `job.json`・`job.lock` のどちらも無い（ジョブを開始する前に止まった）。
        Err(JobRecordError::Io {
            kind: std::io::ErrorKind::NotFound,
        }) => Ok(None),
        Err(e) => Err(e.to_error_report()),
    }
}

/// `--status` で列挙する候補ディレクトリ数の上限（REQ-39）。
const MAX_STATUS_CANDIDATES: usize = 1024;

/// `train --status [--candidate N]`（REQ-34・TASK-34.2・TASK-34.3・#485）。
///
/// `candidate` 省略時は記録のある候補だけを添字順に返す（無ければ空）。凍結の検査はしない。
///
/// # Errors
/// `--candidate N` に記録が無い・記録の形式の不正は `invalid_input`（64）、記録のサイズ超過・候補が
/// 多すぎる場合は `limit_exceeded`（20）、lock・I/O の失敗は `runtime_error`（70）。
pub fn status(
    args: &TrainArgs,
    candidate: Option<usize>,
    cwd: &Path,
) -> Result<Vec<TrainStatusEntry>, ErrorReport> {
    let project = Project::open(cwd, &args.project_dir)?;
    if let Some(index) = candidate {
        let job = job_status(&project, index)?.ok_or_else(|| invalid("job record not found"))?;
        return Ok(vec![TrainStatusEntry::new(index, job)]);
    }
    if !project.exists(CANDIDATES_DIR)? {
        return Ok(Vec::new());
    }
    let names = project
        .open_subdir(CANDIDATES_DIR)?
        .list_entry_names(MAX_STATUS_CANDIDATES)
        .map_err(|_| runtime("cannot list project directory"))?;
    if names.len() > MAX_STATUS_CANDIDATES {
        return Err(fail(
            ExitCode::LimitExceeded,
            "too many candidate directories",
        ));
    }
    // 添字の正準形（`candidate_rel` が作る名前）だけを候補とする。
    let mut indices: Vec<usize> = names
        .iter()
        .filter_map(|n| n.to_str())
        .filter_map(|s| s.parse::<usize>().ok().filter(|i| i.to_string() == s))
        .collect();
    indices.sort_unstable();
    let mut jobs = Vec::new();
    for index in indices {
        if let Some(job) = job_status(&project, index)? {
            jobs.push(TrainStatusEntry::new(index, job));
        }
    }
    Ok(jobs)
}

/// 作成済みの候補ディレクトリ `rel` へ `job/`・`train_input.jsonl`・`request.json`（・`--train-seed` の
/// 記録）を置き、`job/` の絶対パスを返す（単発の `train` と `train --all` が共有する）。
fn prepare_candidate_dir(
    project: &Project,
    rel: &Path,
    request: &TrainRequest,
    train_jsonl: &[u8],
    train_seed_override: Option<u32>,
) -> Result<PathBuf, ErrorReport> {
    let request_json = request.to_json_vec().map_err(|e| e.to_error_report())?;
    let job_dir = project.create_dir(rel.join(JOB_DIR))?;
    project.write_new(rel.join(TRAIN_INPUT_FILE), train_jsonl)?;
    project.write_new(rel.join(REQUEST_FILE), &request_json)?;
    if let Some(seed) = train_seed_override {
        project.write_new(rel.join(TRAIN_SEED_FILE), seed.to_string().as_bytes())?;
    }
    Ok(job_dir)
}

/// `train --smoke`: 動作確認用の軽量実行（エポック数のみ 1 へ）。全種類の既定設定が `epochs` を持つ。
fn apply_smoke(candidate: &mut SearchCandidate) {
    candidate
        .params
        .config
        .insert("epochs".to_string(), 1.into());
}

/// 候補 1 件の学習ジョブを、用意済みの `job/` で実行する関数の型（本番は [`run_recorded_job`] で
/// `job/` に記録しながら実行する。テストは偽の実行で差し替える）。
type JobFn<'a> = dyn FnMut(&TrainRequest, &Path) -> Result<TrainOutcome, AllRunError> + 'a;

/// `train --all` を実行する（REQ-18・REQ-27・REQ-34・REQ-39・TASK-18.1・TASK-18.2・#482・#483）。
///
/// # Errors
/// 前提の欠落・既存の候補ディレクトリ・既存の `search_record.json` は `invalid_input`（64）。
/// 評価済みの候補が無いときの終了コードはモジュール doc「`--all`」のとおり。
pub fn run_all(
    args: &TrainArgs,
    budget: SearchBudget,
    cwd: &Path,
) -> Result<TrainAllReport, ErrorReport> {
    let launcher_job = || -> Result<Box<JobFn<'static>>, ErrorReport> {
        let launcher = worker_launcher()?;
        Ok(Box::new(move |request: &TrainRequest, job_dir: &Path| {
            run_recorded_job(&launcher, request, job_dir).map_err(|e| match e {
                RecordedJobError::Run(e) => AllRunError::Process(e),
                RecordedJobError::Begin(e) | RecordedJobError::Record(e) => {
                    AllRunError::Report(e.to_error_report())
                }
            })
        }))
    };
    run_all_with(args, budget, cwd, launcher_job, &SystemClock::new())
}

/// [`run_all`] の本体。学習ジョブの実行（`make_job`。前提の確認後に 1 回だけ呼ぶ）と時計を差し替えられる。
fn run_all_with<'a, M, C>(
    args: &TrainArgs,
    budget: SearchBudget,
    cwd: &Path,
    make_job: M,
    clock: &C,
) -> Result<TrainAllReport, ErrorReport>
where
    M: FnOnce() -> Result<Box<JobFn<'a>>, ErrorReport>,
    C: Clock,
{
    let project = Project::open(cwd, &args.project_dir)?;
    // 副作用の前に、評価データが凍結記録どおりか確認する（REQ-17）。
    super::inspect::ensure_evaluation_frozen(&project)?;
    let definition = project.load_definition()?;
    let records = project.load_records(&definition)?;
    let (split, split_seed) = verified_split(&project, &records)?;
    let split_record = read_split_record(&project)?;
    let seed = args.train_seed.unwrap_or(split_seed);

    // 候補 N の `root` は `candidates/<N>`。単発の `train --candidate N` と同じ解決を候補ごとに行う。
    let n_candidates = resolve_candidates(&project, &definition, 0, seed)?.len();
    let mut candidates = Vec::with_capacity(n_candidates);
    for index in 0..n_candidates {
        let mut candidate = resolve_candidates(&project, &definition, index, seed)?
            .into_iter()
            .nth(index)
            .ok_or_else(|| runtime("candidate resolution is inconsistent"))?;
        if args.smoke {
            apply_smoke(&mut candidate);
        }
        candidates.push(candidate);
    }
    // やり直しは新規（REQ-34）。既存の探索記録・候補ディレクトリがあれば何も作らずに止める。
    if project.exists(SEARCH_RECORD_FILE)? {
        return Err(invalid("search record already exists"));
    }
    for index in 0..n_candidates {
        if project.exists(candidate_rel(index))? {
            return Err(invalid("candidate directory already exists"));
        }
    }

    let train_jsonl = trainer_jsonl(
        train_rows(&records, &split).map(|r| (r.input.as_str(), r.label_id.as_str())),
    )
    .map_err(|e| stage_file_error_report(e, "cannot build training data"))?;
    let label_order = label_order_from_definition(&definition)
        .map_err(|e| e.to_error_report())?
        .into_vec();
    let label_refs: Vec<&str> = label_order.iter().map(String::as_str).collect();
    // validation は分割記録の validation のみ（凍結した最終 test・評価データは使わない。REQ-27）。
    // 並びは `build_train_request` と同じ（`select`・`package` の照合が通るように）。
    let validation: Vec<&ValidRecord> = records
        .iter()
        .filter(|r| split.by_record.get(&r.id) == Some(&Split::Validation))
        .collect();
    let gold: Vec<&str> = validation.iter().map(|r| r.label_id.as_str()).collect();
    let ids: Vec<&str> = validation.iter().map(|r| r.id.as_str()).collect();
    let inputs: Vec<&[u8]> = validation.iter().map(|r| r.input.as_bytes()).collect();
    let roots: Vec<String> = candidates.iter().map(|c| c.params.root.clone()).collect();
    let job = make_job()?;

    let mut runner = CandidateDirRunner {
        project: &project,
        parent: None,
        train_jsonl: &train_jsonl,
        train_seed_override: args.train_seed,
        created: std::iter::repeat_with(|| None).take(roots.len()).collect(),
        outcomes: vec![None; roots.len()],
        roots,
        job,
    };
    let searched = run_search(
        &mut runner,
        clock,
        SearchInput {
            label_order: &label_refs,
            validation_gold: &gold,
            validation_record_ids: &ids,
            validation_inputs: &inputs,
            validation_split_record: &split_record,
            candidates,
            budget,
            policy: PerCandidatePolicy::EvenSplit,
        },
    );
    let record = match searched {
        Ok(record) => record,
        Err(error) => return Err(runner.clean_up(search_error_report(error))),
    };
    if let Err(report) = runner.publish(&record) {
        return Err(runner.clean_up(report));
    }
    all_outcome(&record, &runner.outcomes)
}

/// 探索結果から `train --all` の終了コードと stdout を決める（モジュール doc「`--all`」）。
fn all_outcome(
    record: &SearchRecord,
    outcomes: &[Option<TrainOutcome>],
) -> Result<TrainAllReport, ErrorReport> {
    let is_evaluated =
        |r: &CandidateSearchResult| matches!(r, CandidateSearchResult::Evaluated { .. });
    if !record.candidates.iter().any(|e| is_evaluated(&e.result)) {
        if record.budget_reached {
            return Err(fail(
                ExitCode::LimitExceeded,
                "search budget reached before any candidate was evaluated",
            ));
        }
        // 宣言順で最初の失敗候補。ワーカーの失敗は単発の `train` と同じ写像、採点の失敗は 70。
        let first_failure = record.candidates.iter().enumerate().find(|(_, e)| {
            matches!(
                e.result,
                CandidateSearchResult::TrainingNotCompleted | CandidateSearchResult::ScoringFailed
            )
        });
        return Err(first_failure
            .and_then(|(index, _)| outcomes.get(index).and_then(Option::as_ref))
            .and_then(train_outcome_error_report)
            .unwrap_or_else(|| runtime("no candidate was evaluated")));
    }
    let candidates = record
        .candidates
        .iter()
        .enumerate()
        .map(|(index, entry)| TrainAllCandidate {
            candidate: index,
            kind: entry.candidate_id.clone(),
            result: (&entry.result).into(),
            budget_reached: entry.budget_reached().map(Into::into),
        })
        .collect();
    Ok(TrainAllReport::new(
        record.budget_seconds,
        record.budget_reached,
        record.total_elapsed_ms,
        candidates,
    ))
}

/// 探索自体の失敗の写像。学習リクエストの組み立て（上限超過は 20 等）・候補の準備・学習ジョブの起動の
/// 失敗は単発の `train` と同じ写像、それ以外は `runtime_error`（70。分割は探索の前に検証済みのため、
/// 探索の入力検証で止まるのは内部の不整合）。
fn search_error_report(error: SearchError<AllRunError>) -> ErrorReport {
    match error {
        SearchError::InvalidRequest { source, .. }
        | SearchError::Candidate {
            source: CandidateTimeError::Request(source),
            ..
        } => source.to_error_report(),
        SearchError::Candidate {
            source: CandidateTimeError::Runner(AllRunError::Report(report)),
            ..
        } => report,
        SearchError::Candidate {
            source: CandidateTimeError::Runner(AllRunError::Process(process)),
            ..
        } => process.to_error_report(),
        _ => runtime("candidate search failed"),
    }
}

/// [`CandidateDirRunner`] の失敗（学習ジョブの子プロセスの失敗と、候補の準備・検査の失敗）。
enum AllRunError {
    Process(TrainProcessError),
    Report(ErrorReport),
}

impl From<TrainProcessError> for AllRunError {
    fn from(e: TrainProcessError) -> Self {
        Self::Process(e)
    }
}

/// `train --all` の実行器: 候補の学習ジョブの直前に `candidates/<N>/` を作って入力を置き、学習ジョブを
/// 実行して結果を控える（[`run_search`] の [`CandidateRunner`]）。候補は `request.root()` で特定する。
struct CandidateDirRunner<'p, 'j> {
    project: &'p Project,
    /// この呼び出しで作った `candidates/`（最初の候補の学習ジョブの直前に作る。失敗時の後始末用）。
    parent: Option<CreatedDir>,
    roots: Vec<String>,
    train_jsonl: &'p [u8],
    train_seed_override: Option<u32>,
    /// 候補ごとに作ったディレクトリ（後始末用）。
    created: Vec<Option<CreatedDir>>,
    /// 候補ごとの学習結果（`result.json` と失敗コードの写像に使う）。
    outcomes: Vec<Option<TrainOutcome>>,
    job: Box<JobFn<'j>>,
}

impl CandidateRunner for CandidateDirRunner<'_, '_> {
    type Error = AllRunError;

    fn run(&mut self, request: &TrainRequest) -> Result<TrainOutcome, AllRunError> {
        let index = self
            .roots
            .iter()
            .position(|root| root == request.root())
            .ok_or_else(|| AllRunError::Report(runtime("unexpected candidate request")))?;
        let rel = candidate_rel(index);
        if self.parent.is_none()
            && !self
                .project
                .exists(CANDIDATES_DIR)
                .map_err(AllRunError::Report)?
        {
            let parent = self
                .project
                .create_dir_tracked(CANDIDATES_DIR)
                .map_err(AllRunError::Report)?;
            self.parent = Some(parent);
        }
        let created = self
            .project
            .create_dir_tracked(&rel)
            .map_err(AllRunError::Report)?;
        if let Some(slot) = self.created.get_mut(index) {
            *slot = Some(created);
        }
        let job_dir = prepare_candidate_dir(
            self.project,
            &rel,
            request,
            self.train_jsonl,
            self.train_seed_override,
        )
        .map_err(AllRunError::Report)?;
        let outcome = (self.job)(request, &job_dir)?;
        if let TrainOutcome::Ok(success) = &outcome {
            check_empty_input_preprocessing(success.empty_input_ids(), request.max_bytes())
                .map_err(AllRunError::Report)?;
        }
        if let Some(slot) = self.outcomes.get_mut(index) {
            *slot = Some(outcome.clone());
        }
        Ok(outcome)
    }

    fn is_wall_timeout(error: &AllRunError) -> bool {
        matches!(error, AllRunError::Process(e) if <WorkerCandidateRunner<'_> as CandidateRunner>::is_wall_timeout(e))
    }
}

impl CandidateDirRunner<'_, '_> {
    /// `evaluated` の候補だけ `result.json` を書き、それ以外の候補ディレクトリを片付けてから、最後に探索記録を
    /// 書く（途中で失敗しても「`evaluated` と記録したのに `result.json` が無い」状態を残さない。失敗時は
    /// 呼び出し元が [`Self::clean_up`] で候補ディレクトリをすべて片付ける）。
    fn publish(&mut self, record: &SearchRecord) -> Result<(), ErrorReport> {
        let bytes = search_record_json_vec(record)
            .map_err(|_| runtime("cannot serialize search record"))?;
        let mut cleaned = true;
        for (index, entry) in record.candidates.iter().enumerate() {
            if matches!(entry.result, CandidateSearchResult::Evaluated { .. }) {
                let outcome = self
                    .outcomes
                    .get(index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| runtime("evaluated candidate has no train result"))?;
                let result_json = outcome_json_vec(outcome)
                    .map_err(|_| runtime("cannot serialize train result"))?;
                self.project
                    .write_new(candidate_rel(index).join(RESULT_FILE), &result_json)?;
            } else if let Some(created) = self.created.get_mut(index).and_then(Option::take) {
                cleaned &= self.project.remove_created_dir(&created);
            }
        }
        if !cleaned {
            return Err(runtime("candidate directory could not be cleaned up"));
        }
        // 最後に原子的に公開する（途中終了で部分的な記録が残り再実行を塞がない。REQ-34・REQ-39）。
        self.project.publish_new_file(SEARCH_RECORD_FILE, &bytes)
    }

    /// 作った候補ディレクトリ（と、この呼び出しで作った `candidates/`）をすべて片付け、`report` を返す
    /// （片付けられなければ固定文言を付記する）。
    fn clean_up(&mut self, mut report: ErrorReport) -> ErrorReport {
        let mut cleaned = true;
        for created in self.created.iter_mut().filter_map(Option::take) {
            cleaned &= self.project.remove_created_dir(&created);
        }
        if let Some(parent) = self.parent.take() {
            cleaned &= self.project.remove_created_dir(&parent);
        }
        if !cleaned {
            report
                .message
                .push_str("; candidate directory could not be cleaned up");
        }
        report
    }
}

/// 学習ワーカー（Python）が報告した空入力のトークン列を、推論ランタイム（Rust）の前処理
/// （`encode_bytes(normalize_input(""), max_bytes)`）と照合する（REQ-23 境界値・TASK-23.2・#476）。
///
/// CLI 内の評価経路と推論経路は同じ Rust の pipeline なので食い違わない。食い違いうるのは学習時の
/// 前処理（`trainer/`）と推論ランタイムの間だけで、その検知をここで行う。食い違いは利用者入力が原因では
/// ないため `runtime_error`（70）・固定 message で止める（`max_bytes` は学習リクエストの値で、成果物の
/// `max_bytes` と一致することは `TrainOutcome::from_worker_stdout` が保証済み）。
fn check_empty_input_preprocessing(trainer_ids: &[i64], max_bytes: u32) -> Result<(), ErrorReport> {
    let runtime_ids = fandhe_edge_runtime::preprocess::encode_bytes(
        &fandhe_edge_runtime::preprocess::normalize_input(""),
        max_bytes as usize,
    );
    match compare_empty_input_encodings(&runtime_ids, trainer_ids) {
        Ok(EmptyInputConsistency::Consistent(_)) => Ok(()),
        Ok(EmptyInputConsistency::Diverged { .. }) | Err(_) => Err(runtime(
            "empty input preprocessing diverges between trainer and runtime",
        )),
    }
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

/// 候補 `index` の学習 seed を返す。`train --train-seed` で上書きされた候補は `train_seed.txt` の値、
/// それ以外は `split_seed`（`split.json` の seed）。`evaluate`・`select`・`package` が期待する
/// 学習リクエストを組み立てるときに使う（REQ-17・REQ-41）。
///
/// # Errors
/// 記録が読めない・`u32` として不正な場合は `invalid_input`（64）。
pub fn effective_train_seed(
    project: &Project,
    index: usize,
    split_seed: u32,
) -> Result<u32, ErrorReport> {
    let Some(bytes) = project.read_optional(candidate_rel(index).join(TRAIN_SEED_FILE), 16)? else {
        return Ok(split_seed);
    };
    // 正準形（`u32::to_string` と完全一致。`+7`・`007`・末尾改行は不可）だけを受理する。
    std::str::from_utf8(&bytes)
        .ok()
        .and_then(|t| t.parse::<u32>().ok().filter(|v| v.to_string() == t))
        .ok_or_else(|| invalid("train seed record is invalid"))
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

/// `train --all` 済みのプロジェクトで、探索記録が候補 `index`（ID `candidate_id`）へ配分した持ち時間（秒）を
/// 返す（[`request_matches_candidate`] の期待値。REQ-18・REQ-27・#482）。`search_record.json` が無い（単発の
/// `train` だけの）プロジェクトは `None` で、時間制限は既定値と厳密に照合される。
///
/// 記録は保持 fd 起点・[`MAX_SEARCH_RECORD_BYTES`] の上限で読む（REQ-39）。
///
/// # Errors
/// 記録が壊れている・候補が記録に無い・ID が違う・`evaluated` でない場合は `invalid_input`（64。fail-closed）。
/// 読み込みの失敗は [`Project::read_optional`] と同じ写像（上限超過は 20、I/O 失敗は 70）。
pub fn allotted_time_limit(
    project: &Project,
    index: usize,
    candidate_id: &str,
) -> Result<Option<u32>, ErrorReport> {
    let Some(bytes) = project.read_optional(SEARCH_RECORD_FILE, MAX_SEARCH_RECORD_BYTES as u64)?
    else {
        return Ok(None);
    };
    allotted_time_limit_seconds(&bytes, index, candidate_id)
        .map(Some)
        .map_err(|_| invalid("search record is invalid"))
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

/// 候補の学習リクエストを組み立てる（`train` が学習ワーカーへ渡すものと、`select`・`package` が
/// 期待値として組み立てるものの、唯一の作り方。REQ-27・REQ-39）。
///
/// `params`（[`resolve_candidates`] の結果）に、分割記録の validation 入力（`id` と `input` のみ。
/// 正解ラベルは渡さない）を付ける。`train_path`・`device`・制限値・`root`・`out_dir` などは
/// すべて `params` から決まる。
///
/// # Errors
/// リクエストの検証失敗（`TrainRequestError` の写像）。
pub fn build_train_request(
    params: TrainRequestParams,
    records: &[ValidRecord],
    split: &SplitResult,
) -> Result<TrainRequest, ErrorReport> {
    let validation: Vec<ValidationInput> = records
        .iter()
        .filter(|r| split.by_record.get(&r.id) == Some(&Split::Validation))
        .map(|r| ValidationInput::new(r.id.clone(), r.input.clone()))
        .collect();
    TrainRequest::new(params)
        .and_then(|r| r.with_validation_inputs(validation))
        .map_err(|e| e.to_error_report())
}

/// 保存済みの学習リクエストが、期待する学習リクエストと（`epochs`・`time_limit_seconds` を除いて）完全に一致するかを返す
/// （`select`・`package` が、別の構成・別の場所の学習結果を候補 N として扱わないための照合。
/// REQ-27・REQ-39）。
///
/// 期待値は、`train` と同じ [`build_train_request`] に、`train` と同じ [`resolve_candidates`] で作った
/// 候補 N の `params` と、プロジェクトのデータ・`split.json`（`records`・`split`）を渡して丸ごと
/// 組み立て、保存済みのリクエストと構造全体で比べる。項目を列挙しないので、`TrainRequest` に項目が
/// 増えても自動で照合対象になる（`train_path`・`device`・時間と RSS の制限・validation 入力・
/// `root`・`out_dir` を含む）。`root` は絶対パスで記録されるので、プロジェクトのディレクトリを
/// 移動すると一致せず拒否される（fail-closed として許容する）。
///
/// **例外は `epochs` の 1 か所だけ**: `train --smoke` が 1 へ上書きする項目のため、期待値の `epochs` を
/// 保存済みリクエストの値に揃えてから比べる（smoke かどうかは別に [`request_is_smoke_trained`] で判定する）。
/// `time_limit_seconds` は除外しない: `allotted_time_limit` が `Some`（`train --all` の候補。
/// [`allotted_time_limit`] で探索記録から読んだ配分値）ならその値、`None`（単発の `train`）なら既定値を
/// 期待値にして厳密に比べる（保存後に時間制限だけを書き換えた request を通さない。REQ-18・REQ-27・#482）。
/// 実行時の環境でしか決まらない値（学習ワーカーの実行ファイルのパス等）はリクエストに含まれないため、
/// 除外した項目は他にない。
#[must_use]
pub fn request_matches_candidate(
    request: &TrainRequest,
    params: &TrainRequestParams,
    allotted_time_limit: Option<u32>,
    records: &[ValidRecord],
    split: &SplitResult,
) -> bool {
    let mut expected_params = params.clone();
    if allotted_time_limit.is_some() {
        expected_params.time_limit_seconds = allotted_time_limit;
    }
    match request.config().get("epochs") {
        Some(epochs) => {
            expected_params
                .config
                .insert("epochs".to_string(), epochs.clone());
        }
        None => {
            expected_params.config.remove("epochs");
        }
    }
    build_train_request(expected_params, records, split).is_ok_and(|expected| &expected == request)
}

/// 保存済みの学習リクエストが `train --smoke`（`epochs` を 1 へ上書きした短縮学習）のものか
/// （既定候補 `params` の `epochs` と異なるか）を返す。`request_matches_candidate` が `epochs` を
/// 比較から外す理由と同じ項目を見る判定で、smoke かどうかの判定はここに 1 つだけ置く。
/// `select` は smoke の結果も採点・選定できるが、`package` は既定で拒否する（REQ-27）。
#[must_use]
pub fn request_is_smoke_trained(request: &TrainRequest, params: &TrainRequestParams) -> bool {
    request.config().get("epochs") != params.config.get("epochs")
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

/// 分割つきの合成レコード（`train_rows`・`package` の p95 計測入力のテストが共有する）。
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::collections::BTreeMap;

    /// `(id, split)` の並びから、入力が `input-<id>`・ラベルが `yes` のレコードと分割結果を作る。
    pub(crate) fn records_and_split(rows: &[(&str, Split)]) -> (Vec<ValidRecord>, SplitResult) {
        let labeled: Vec<(&str, Split, &str)> = rows
            .iter()
            .map(|(id, split)| (*id, *split, "yes"))
            .collect();
        records_and_split_labeled(&labeled)
    }

    /// `(id, split, label_id)` の並びから、入力が `input-<id>` のレコードと分割結果を作る。
    pub(crate) fn records_and_split_labeled(
        rows: &[(&str, Split, &str)],
    ) -> (Vec<ValidRecord>, SplitResult) {
        let records = rows
            .iter()
            .enumerate()
            .map(|(i, (id, _, label))| ValidRecord {
                line: i + 1,
                id: (*id).to_string(),
                input: format!("input-{id}"),
                label_id: (*label).to_string(),
                output_key: String::new(),
                output_original: String::new(),
                tags: None,
                group_id: None,
            })
            .collect();
        let by_record: BTreeMap<String, Split> = rows
            .iter()
            .map(|(id, split, _)| ((*id).to_string(), *split))
            .collect();
        let split = SplitResult {
            by_record,
            by_group: BTreeMap::new(),
            per_label: Vec::new(),
            rule_id: "test",
        };
        (records, split)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::records_and_split;
    use super::*;

    /// REQ-27・#338: `train_rows` は Train の行だけを入力順に返し、Validation・Test を除く。
    #[test]
    fn req27_issue338_train_rows_returns_only_train_in_order() {
        let (records, split) = records_and_split(&[
            ("a", Split::Train),
            ("b", Split::Validation),
            ("c", Split::Test),
            ("d", Split::Train),
        ]);
        let ids: Vec<&str> = train_rows(&records, &split)
            .map(|r| r.id.as_str())
            .collect();
        assert_eq!(ids, vec!["a", "d"]);
    }

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
/// `train --all` の探索予算の配線（REQ-18・REQ-34・REQ-39・TASK-18.1・TASK-18.2・#482・#483）。
///
/// 証拠の種別: テストハーネス。学習ジョブは偽の実行（`JobFn`）、時計は偽の単調時計で、子プロセスも
/// MLX も使わない（GPU を使わない）。データは本モジュールが生成する合成データのみ。
#[cfg(all(test, unix))]
mod all_tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::time::Duration;

    use fandhe_edge_core::hash::Sha256Digest;
    use fandhe_edge_train::error::TrainRequestError;
    use fandhe_edge_train::time_allotment::TimeAllotmentError;

    use crate::args::{InspectArgs, RegisterArgs, SelectArgs, TrainOp, TrainTarget};

    const LABELS: [&str; 3] = ["alpha", "beta", "gamma"];
    /// `c1` の既定設定（`fixtures/train_contract/kind_defaults.json` と同じ具体値。結果の `config` は
    /// 既定値と完全一致する必要がある）。
    const C1_CONFIG: &str = r#"{"ngram_min":1,"ngram_max":4,"min_df":2,"max_features":200000,"C":1.0,"epochs":30,"batch_size":64,"lr":0.5}"#;

    /// 呼ばれるたびに 300 ms 進む偽の単調時計（候補の学習ジョブの外の処理時間の模擬。学習ジョブの
    /// 所要時間は `JobFn` が `now` を進めて表す）。
    struct FakeClock {
        now_ms: Rc<Cell<u64>>,
    }

    impl Clock for FakeClock {
        fn monotonic(&self) -> Duration {
            let now = self.now_ms.get() + 300;
            self.now_ms.set(now);
            Duration::from_millis(now)
        }
        fn unix_millis(&self) -> Result<u64, TimeAllotmentError> {
            Ok(1_790_000_000_000)
        }
    }

    /// `register`・`inspect` 済みのプロジェクト `proj` を持つ作業ディレクトリ（cwd）。
    fn inspected_workdir(case: &str) -> PathBuf {
        let dir = std::fs::canonicalize(std::env::temp_dir())
            .expect("temp dir")
            .join(format!("fandhe-train-all-{case}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("def")).expect("mkdir");
        let options: Vec<String> = LABELS
            .iter()
            .map(|l| format!(r#"{{"id":"{l}","display_name":"{l}","description":"dummy"}}"#))
            .collect();
        std::fs::write(
            dir.join("def/definition.json"),
            format!(
                r#"{{"schema":"fandhe-edge-model-definition/v1","name":"train_all","version":1,"judgment_type":"single_select","options":[{}],"io":{{"input":"bytes"}}}}"#,
                options.join(",")
            ),
        )
        .expect("definition");
        let mut data = String::new();
        for i in 0..30 {
            for l in LABELS {
                data.push_str(&format!(
                    "{{\"id\":\"{l}-{i}\",\"input\":\"{l} sample {i}\",\"output\":{{\"intent\":\"{l}\"}},\"group_id\":\"g-{l}-{i}\"}}\n"
                ));
            }
        }
        std::fs::write(dir.join("def/train.jsonl"), data).expect("train");
        super::super::register::run(
            &RegisterArgs {
                definition: "def/definition.json".into(),
                project_dir: "proj".into(),
            },
            &dir,
        )
        .expect("register");
        super::super::inspect::run(
            &InspectArgs {
                project_dir: "proj".into(),
                seed: 42,
            },
            &dir,
        )
        .expect("inspect");
        dir
    }

    fn all_args() -> TrainArgs {
        TrainArgs {
            project_dir: "proj".into(),
            op: TrainOp::Run(TrainTarget::All {
                budget: SearchBudget::default(),
            }),
            smoke: false,
            train_seed: None,
        }
    }

    /// 偽の `c1` 学習: 成果物（共有 fixture の ONNX と `artifact.json`）を置き、validation を常に `alpha` と
    /// 予測した成功結果を返す。
    fn fake_c1_success(request: &TrainRequest) -> TrainOutcome {
        fake_c1_outcome(request, "[0]")
    }

    /// [`fake_c1_success`] の空入力のトークン列を指定できる版（`[0]` が推論ランタイムと一致する値）。
    fn fake_c1_outcome(request: &TrainRequest, empty_input_ids: &str) -> TrainOutcome {
        assert_eq!(request.kind(), "c1", "only c1 is expected to run");
        let out_dir = format!("{}/{}", request.root(), request.out_dir());
        std::fs::create_dir(&out_dir).expect("out dir");
        let onnx = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/onnx_parity/c1.onnx"
        ))
        .expect("fixture onnx");
        std::fs::write(format!("{out_dir}/model.onnx"), &onnx).expect("onnx");
        let sha = Sha256Digest::of_bytes(&onnx).to_hex();
        let labels = r#""alpha","beta","gamma""#;
        let (version, max_bytes) = (request.kind_version(), request.max_bytes());
        std::fs::write(
            format!("{out_dir}/artifact.json"),
            format!(
                r#"{{"kind":"c1","kind_version":{version},"max_bytes":{max_bytes},"label_order":[{labels}],"onnx_file":"model.onnx","onnx_sha256":"{sha}"}}"#
            ),
        )
        .expect("artifact.json");
        let predictions: Vec<String> = request
            .validation_inputs()
            .unwrap_or_default()
            .iter()
            .map(|v| {
                format!(
                    r#"{{"id":"{}","status":"ok","predicted_label":"alpha"}}"#,
                    v.id()
                )
            })
            .collect();
        let stdout = format!(
            r#"{{"status":"ok","artifact_dir":"{out_dir}","artifact":{{"kind":"c1","kind_version":{version},"selector_version":"0.1","config":{C1_CONFIG},"label_order":[{labels}],"output_type":"choice","max_bytes":{max_bytes},"onnx_file":"model.onnx","onnx_sha256":"{sha}","created_utc":"2026-09-30T00:00:00Z","candidate_label":"c1"}},"empty_input_ids":{empty_input_ids},"validation_predictions":[{}]}}"#,
            predictions.join(",")
        );
        TrainOutcome::from_worker_stdout(stdout.as_bytes(), request).expect("fake outcome")
    }

    /// REQ-18・TASK-18.1・TASK-18.2・#482・#483: 予算 3 秒（開始時の残り 2 秒台を均等割りし、1 候補目の
    /// 持ち時間は 1 秒）で 1 候補目が持ち時間内（900 ms）に完了し、候補の外の処理と合わせて 2.4 秒を
    /// 使うと、残り予算が 1 秒未満になり 2 候補目は未着手（`not_started`・
    /// `search_budget`）。評価済みが 1 件あるので exit 0・`budget_reached:true`。`result.json` は
    /// 1 候補目だけ、2 候補目の候補ディレクトリは作られず、`search_record.json` が残り、`select` は
    /// 1 候補目を選ぶ（保存済み `request.json` の `time_limit_seconds:1` が照合を通る）。
    #[test]
    fn req18_issue482_small_budget_evaluates_first_and_leaves_second_not_started() {
        let cwd = inspected_workdir("small");
        let now_ms = Rc::new(Cell::new(0));
        let clock = FakeClock {
            now_ms: Rc::clone(&now_ms),
        };
        let job_now = Rc::clone(&now_ms);
        let make_job = move || -> Result<Box<JobFn<'static>>, ErrorReport> {
            Ok(Box::new(move |request: &TrainRequest, _job: &Path| {
                job_now.set(job_now.get() + 600);
                Ok(fake_c1_success(request))
            }))
        };
        let budget = SearchBudget::new(3).expect("budget");
        let report =
            run_all_with(&all_args(), budget, &cwd, make_job, &clock).expect("train --all");
        assert_eq!(
            report.to_json_line().expect("json"),
            "{\"step\":\"train\",\"status\":\"ok\",\"budget_seconds\":3,\"budget_reached\":true,\"total_elapsed_ms\":2700,\"candidates\":[{\"candidate\":0,\"kind\":\"c1\",\"result\":\"evaluated\",\"budget_reached\":null},{\"candidate\":1,\"kind\":\"c3\",\"result\":\"not_started\",\"budget_reached\":\"search_budget\"}]}"
        );
        let proj = cwd.join("proj");
        assert!(proj.join("candidates/0/result.json").is_file());
        assert!(!proj.join("candidates/1").exists());
        let request =
            std::fs::read_to_string(proj.join("candidates/0/request.json")).expect("request.json");
        assert!(request.contains("\"time_limit_seconds\":1,"), "{request}");
        let record = std::fs::read_to_string(proj.join(SEARCH_RECORD_FILE)).expect("record");
        assert!(
            record.starts_with("{\"budget_seconds\":3,\"per_candidate_policy\":\"even_split\",")
        );
        assert!(record.ends_with("\"budget_reached\":true}\n"), "{record}");
        assert!(!record.contains("predicted"), "{record}");

        let selected = super::super::select::run(
            &SelectArgs {
                project_dir: "proj".into(),
            },
            &cwd,
        )
        .expect("select")
        .to_json_line()
        .expect("json");
        assert!(
            selected.starts_with(
                "{\"step\":\"select\",\"status\":\"ok\",\"candidate\":0,\"kind\":\"c1\""
            ),
            "{selected}"
        );
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-18・REQ-34・#482: 全候補の学習ワーカーが失敗（予算到達なし）すると、宣言順で最初の失敗候補の
    /// 失敗コードの写像（`training_diverged` は 12）で終わる。候補ディレクトリはすべて片付けられ、
    /// `search_record.json` は残る。再実行は既存の探索記録で `invalid_input`（やり直しは新規。REQ-34）。
    #[test]
    fn req18_issue482_all_failed_maps_first_worker_failure_and_keeps_record() {
        let cwd = inspected_workdir("failed");
        let clock = FakeClock {
            now_ms: Rc::new(Cell::new(0)),
        };
        let make_job = || -> Result<Box<JobFn<'static>>, ErrorReport> {
            Ok(Box::new(|request: &TrainRequest, _job: &Path| {
                let stdout = br#"{"status":"error","code":"training_diverged","message":"x"}"#;
                Ok(TrainOutcome::from_worker_stdout(stdout, request).expect("failure"))
            }))
        };
        let error = run_all_with(&all_args(), SearchBudget::default(), &cwd, make_job, &clock)
            .expect_err("no candidate evaluated");
        assert_eq!(error.code, ExitCode::Pending);
        assert_eq!(error.message, "train worker failed: training_diverged");
        let proj = cwd.join("proj");
        assert!(!proj.join("candidates/0").exists());
        assert!(!proj.join("candidates/1").exists());
        let record = std::fs::read_to_string(proj.join(SEARCH_RECORD_FILE)).expect("record");
        assert!(
            record.contains("\"result\":\"training_not_completed\""),
            "{record}"
        );
        assert!(record.ends_with("\"budget_reached\":false}\n"), "{record}");

        let again = run_all_with(
            &all_args(),
            SearchBudget::default(),
            &cwd,
            || -> Result<Box<JobFn<'static>>, ErrorReport> { panic!("must not start jobs") },
            &clock,
        )
        .expect_err("existing record");
        assert_eq!(
            error_pair(&again),
            (ExitCode::InvalidInput, "search record already exists")
        );
        // `train --all` 済みのプロジェクトへの単発の `train --candidate` も同じく拒否する（予算内と予算外で
        // 学習した候補を混ぜない。REQ-18）。学習ワーカーの発見より前に止まり、候補ディレクトリを作らない。
        let single = TrainArgs {
            op: TrainOp::Run(TrainTarget::Candidate(0)),
            ..all_args()
        };
        let rejected = run(&single, 0, &cwd).expect_err("existing record");
        assert_eq!(
            error_pair(&rejected),
            (ExitCode::InvalidInput, "search record already exists")
        );
        assert!(!proj.join("candidates/0").exists());
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-18・REQ-39・#482: 学習ジョブが壁時計の締め切りで止められた（`WallTimeout`）候補は、探索全体の
    /// 失敗ではなく候補の時間切れ（`training_timed_out`・`candidate_time_limit`）として扱われ、候補
    /// ディレクトリは片付けられる。評価済みの候補（c1）は `result.json` とともに残り、exit 0。
    #[test]
    fn req18_issue482_wall_timeout_is_candidate_time_limit_and_keeps_evaluated() {
        let cwd = inspected_workdir("walltimeout");
        let clock = FakeClock {
            now_ms: Rc::new(Cell::new(0)),
        };
        let make_job = || -> Result<Box<JobFn<'static>>, ErrorReport> {
            Ok(Box::new(|request: &TrainRequest, _job: &Path| {
                if request.kind() == "c3" {
                    return Err(TrainProcessError::WallTimeout {
                        limit_ms: 1000,
                        child_reaped: true,
                    }
                    .into());
                }
                Ok(fake_c1_success(request))
            }))
        };
        let report = run_all_with(&all_args(), SearchBudget::default(), &cwd, make_job, &clock)
            .expect("train --all");
        let line = report.to_json_line().expect("json");
        assert!(
            line.starts_with("{\"step\":\"train\",\"status\":\"ok\",\"budget_seconds\":3600,\"budget_reached\":true,"),
            "{line}"
        );
        assert!(
            line.ends_with("\"candidates\":[{\"candidate\":0,\"kind\":\"c1\",\"result\":\"evaluated\",\"budget_reached\":null},{\"candidate\":1,\"kind\":\"c3\",\"result\":\"training_timed_out\",\"budget_reached\":\"candidate_time_limit\"}]}"),
            "{line}"
        );
        let proj = cwd.join("proj");
        assert!(proj.join("candidates/0/result.json").is_file());
        assert!(!proj.join("candidates/1").exists());
        assert!(proj.join(SEARCH_RECORD_FILE).is_file());
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-34・REQ-39・#482: 時間切れ以外の学習ジョブの失敗（起動失敗）は探索全体を止め、単発の `train` と
    /// 同じ写像（`runtime_error`・70）で返す。それまでに作った候補ディレクトリと、この呼び出しで作った
    /// `candidates/` はすべて片付けられ、探索記録は書かない（探索が完了していないため）。
    #[test]
    fn req34_issue482_process_failure_cleans_every_candidate_dir() {
        let cwd = inspected_workdir("spawnfail");
        let clock = FakeClock {
            now_ms: Rc::new(Cell::new(0)),
        };
        let make_job = || -> Result<Box<JobFn<'static>>, ErrorReport> {
            Ok(Box::new(|request: &TrainRequest, _job: &Path| {
                if request.kind() == "c3" {
                    return Err(TrainProcessError::Spawn {
                        kind: std::io::ErrorKind::NotFound,
                    }
                    .into());
                }
                Ok(fake_c1_success(request))
            }))
        };
        let error = run_all_with(&all_args(), SearchBudget::default(), &cwd, make_job, &clock)
            .expect_err("spawn failure");
        assert_eq!(
            error_pair(&error),
            (
                ExitCode::RuntimeError,
                "failed to spawn worker process: NotFound"
            )
        );
        let proj = cwd.join("proj");
        assert!(!proj.join("candidates").exists());
        assert!(!proj.join(SEARCH_RECORD_FILE).exists());
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-23・TASK-23.2・#482: 学習ワーカーの空入力の前処理が推論ランタイムと食い違うと、単発の `train` と
    /// 同じ `runtime_error`（70）・固定 message で止まり、候補ディレクトリ・`candidates/` を残さない。
    #[test]
    fn req23_issue482_empty_input_divergence_cleans_every_candidate_dir() {
        let cwd = inspected_workdir("diverge");
        let clock = FakeClock {
            now_ms: Rc::new(Cell::new(0)),
        };
        let make_job = || -> Result<Box<JobFn<'static>>, ErrorReport> {
            Ok(Box::new(|request: &TrainRequest, _job: &Path| {
                Ok(fake_c1_outcome(request, "[0,0]"))
            }))
        };
        let error = run_all_with(&all_args(), SearchBudget::default(), &cwd, make_job, &clock)
            .expect_err("divergence");
        assert_eq!(
            error_pair(&error),
            (
                ExitCode::RuntimeError,
                "empty input preprocessing diverges between trainer and runtime"
            )
        );
        let proj = cwd.join("proj");
        assert!(!proj.join("candidates").exists());
        assert!(!proj.join(SEARCH_RECORD_FILE).exists());
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-21・REQ-39・#482: 探索自体の失敗の写像。学習リクエストの組み立ての失敗（`Request`・
    /// `InvalidRequest`）は単発の `train` と同じ写像（設定の上限超過は `limit_exceeded`・20）、学習ジョブの
    /// 起動失敗は `TrainProcessError` の写像、それ以外は `runtime_error`（70）・固定 message。
    #[test]
    fn req21_issue482_search_error_maps_like_single_train() {
        let too_large = || TrainRequestError::ConfigTooLarge { limit: 1 };
        let request_error = SearchError::<AllRunError>::Candidate {
            index: 0,
            source: CandidateTimeError::Request(too_large()),
        };
        assert_eq!(
            error_pair(&search_error_report(request_error)),
            (
                ExitCode::LimitExceeded,
                "train request config exceeds 1 bytes limit"
            )
        );
        let invalid_request = SearchError::<AllRunError>::InvalidRequest {
            index: 1,
            source: too_large(),
        };
        assert_eq!(
            search_error_report(invalid_request).code,
            ExitCode::LimitExceeded
        );
        let wall = SearchError::Candidate {
            index: 0,
            source: CandidateTimeError::Runner(AllRunError::Process(
                TrainProcessError::WallTimeout {
                    limit_ms: 5,
                    child_reaped: true,
                },
            )),
        };
        assert_eq!(search_error_report(wall).code, ExitCode::LimitExceeded);
        let prepared = SearchError::Candidate {
            index: 0,
            source: CandidateTimeError::Runner(AllRunError::Report(invalid("x"))),
        };
        assert_eq!(
            error_pair(&search_error_report(prepared)),
            (ExitCode::InvalidInput, "x")
        );
        assert_eq!(
            error_pair(&search_error_report(SearchError::EmptyCandidates)),
            (ExitCode::RuntimeError, "candidate search failed")
        );
        assert_eq!(
            error_pair(&search_error_report(
                SearchError::ValidationSplitHashMismatch
            )),
            (ExitCode::RuntimeError, "candidate search failed")
        );
    }

    fn error_pair(report: &ErrorReport) -> (ExitCode, &str) {
        (report.code, report.message.as_str())
    }
}

/// `train --status` とジョブ記録の配線（REQ-34・TASK-34.2・TASK-34.3・REQ-21・#485）。
///
/// 証拠の種別: テストハーネス。記録は学習ワーカー層の `JobRecorder` で直接作り、学習ワーカーは
/// 起動しない（GPU を使わない）。
#[cfg(all(test, unix))]
mod status_tests {
    use super::*;
    use fandhe_edge_train::job_record::{JobFailure, JobRecorder};
    use fandhe_edge_train::stage_files::train_status_json_line;

    use crate::args::TrainOp;

    /// `proj/` だけを持つ作業ディレクトリ（cwd。正準化済み）。
    fn workdir(case: &str) -> PathBuf {
        let dir = std::fs::canonicalize(std::env::temp_dir())
            .expect("temp dir")
            .join(format!("fandhe-train-status-{case}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("proj")).expect("mkdir");
        dir
    }

    /// `candidates/<index>/job/` を作って絶対パスを返す。
    fn job_dir(cwd: &Path, index: usize) -> PathBuf {
        let dir = cwd.join(format!("proj/candidates/{index}/job"));
        std::fs::create_dir_all(&dir).expect("job dir");
        dir
    }

    fn status_line(cwd: &Path, candidate: Option<usize>) -> Result<String, ErrorReport> {
        let args = TrainArgs {
            project_dir: "proj".into(),
            op: TrainOp::Status(candidate),
            smoke: false,
            train_seed: None,
        };
        status(&args, candidate, cwd).map(|jobs| train_status_json_line(&jobs).expect("json"))
    }

    const NOT_INSPECTED: &str = r#"{"resumable":false,"action":"restart_from_scratch","reason_code":"resume_not_supported","out_dir_action":"not_inspected","message":"Resume is not supported. Restart the job from scratch; the state of out_dir was not inspected, so check it before reuse."}"#;

    /// REQ-34・#485 (a): 終端記録（`succeeded`・`failed`）をそのまま写し、記録のある候補だけを添字順に
    /// 並べる（記録の無い候補・添字の正準形でない名前は出さない）。`failed` にだけやり直し案内が付く。
    /// プロジェクトは定義・凍結記録を持たない（`--status` は凍結の検査をしない）。
    #[test]
    fn req34_status_reports_terminal_records_exactly() {
        let cwd = workdir("terminal");
        JobRecorder::begin(&job_dir(&cwd, 10), 100)
            .expect("begin")
            .finish(JobState::Succeeded, None, 120)
            .expect("finish");
        JobRecorder::begin(&job_dir(&cwd, 2), 100)
            .expect("begin")
            .finish(
                JobState::Failed,
                Some(JobFailure::Error {
                    code: ExitCode::Pending,
                }),
                130,
            )
            .expect("finish");
        job_dir(&cwd, 3);
        std::fs::create_dir_all(cwd.join("proj/candidates/07/job")).expect("non-canonical");
        assert_eq!(
            status_line(&cwd, None).expect("status"),
            format!(
                r#"{{"step":"train","status":"ok","jobs":[{{"candidate":2,"job":{{"state":"failed","crash_detected":false,"failure":{{"kind":"error","code":"pending"}},"record_updated":false}},"restart":{NOT_INSPECTED}}},{{"candidate":10,"job":{{"state":"succeeded","crash_detected":false,"failure":null,"record_updated":false}},"restart":null}}]}}"#
            )
        );
        assert_eq!(
            status_line(&cwd, Some(10)).expect("status"),
            r#"{"step":"train","status":"ok","jobs":[{"candidate":10,"job":{"state":"succeeded","crash_detected":false,"failure":null,"record_updated":false},"restart":null}]}"#
        );
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-34・#485 (b): 別スレッドが `JobRecorder::begin` で lock を保持している間は `running`
    /// （クラッシュと誤報しない）。保持者が `finish` すれば終端記録を返す。
    #[test]
    fn req34_status_reports_running_while_lock_is_held() {
        let cwd = workdir("running");
        let dir = job_dir(&cwd, 0);
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let owner = std::thread::spawn(move || {
            let recorder = JobRecorder::begin(&dir, 100).expect("begin");
            held_tx.send(()).expect("send");
            done_rx.recv().expect("recv");
            recorder
                .finish(JobState::Succeeded, None, 110)
                .expect("finish");
        });
        held_rx.recv().expect("held");
        assert_eq!(
            status_line(&cwd, Some(0)).expect("status"),
            r#"{"step":"train","status":"ok","jobs":[{"candidate":0,"job":{"state":"running","crash_detected":false,"failure":null,"record_updated":false},"restart":null}]}"#
        );
        done_tx.send(()).expect("done");
        owner.join().expect("owner");
        assert!(
            status_line(&cwd, Some(0))
                .expect("status")
                .contains(r#""state":"succeeded""#)
        );
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-34・#485 (c): `running` の記録が残り lock の保持者がいない（所有プロセスが落ちた）と、
    /// `failed`＋`owner_lost` を書き戻して `record_updated:true`、2 回目は同じ記録を `false` で返す。
    #[test]
    fn req34_status_detects_owner_lost_once() {
        let cwd = workdir("ownerlost");
        // `finish` せずに drop すると lock だけが解放され、記録は `running` のまま残る。
        drop(JobRecorder::begin(&job_dir(&cwd, 1), 100).expect("begin"));
        let crashed = |updated: bool| {
            let line = status_line(&cwd, None).expect("status");
            let (head, tail) = line
                .split_once(r#""detected_at_unix":"#)
                .expect("detected_at_unix");
            assert_eq!(
                head,
                r#"{"step":"train","status":"ok","jobs":[{"candidate":1,"job":{"state":"failed","crash_detected":true,"failure":{"kind":"crashed","cause":"owner_lost","signal":null,"#
            );
            let (time, rest) = tail.split_once('}').expect("time");
            assert!(time.parse::<u64>().is_ok_and(|t| t >= 100), "{time}");
            assert_eq!(
                rest,
                format!(r#","record_updated":{updated}}},"restart":{NOT_INSPECTED}}}]}}"#)
            );
        };
        crashed(true);
        crashed(false);
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-34・REQ-21・#485 (d): `--candidate N` に記録が無ければ `invalid_input`・`job record not found`
    /// （候補ディレクトリが無い・`job/` に記録が無いのどちらも）。省略時に記録が無ければ `jobs:[]`。
    #[test]
    fn req34_status_without_record_is_invalid_input() {
        let cwd = workdir("missing");
        assert_eq!(
            status_line(&cwd, None).expect("status"),
            r#"{"step":"train","status":"ok","jobs":[]}"#
        );
        let missing = status_line(&cwd, Some(0)).expect_err("no candidate dir");
        assert_eq!(
            (missing.code, missing.message.as_str()),
            (ExitCode::InvalidInput, "job record not found")
        );
        job_dir(&cwd, 0);
        let empty = status_line(&cwd, Some(0)).expect_err("no record");
        assert_eq!(
            (empty.code, empty.message.as_str()),
            (ExitCode::InvalidInput, "job record not found")
        );
        assert_eq!(
            status_line(&cwd, None).expect("status"),
            r#"{"step":"train","status":"ok","jobs":[]}"#
        );
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-34・REQ-39・#485: 壊れた記録は `invalid_input`・固定 message、`job/` が symlink なら経路の
    /// 拒否（`invalid_input`）でリンク先を読まない。
    #[test]
    fn req34_status_rejects_malformed_record_and_symlinked_job_dir() {
        let cwd = workdir("malformed");
        let dir = job_dir(&cwd, 0);
        std::fs::write(dir.join("job.json"), b"{").expect("job.json");
        let malformed = status_line(&cwd, Some(0)).expect_err("malformed");
        assert_eq!(
            (malformed.code, malformed.message.as_str()),
            (ExitCode::InvalidInput, "job record is malformed")
        );
        std::fs::create_dir_all(cwd.join("proj/candidates/1")).expect("mkdir");
        std::os::unix::fs::symlink(&dir, cwd.join("proj/candidates/1/job")).expect("symlink");
        let escaped = status_line(&cwd, Some(1)).expect_err("symlink");
        assert_eq!(escaped.code, ExitCode::InvalidInput, "{}", escaped.message);
        let _ = std::fs::remove_dir_all(&cwd);
    }
}
