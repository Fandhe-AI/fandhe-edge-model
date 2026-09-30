//! 学習ジョブ記録（`job.json`）の永続化とクラッシュ検出（REQ-34・TASK-34.2・issue #146）。
//!
//! 学習中のプロセスが異常終了しても、後から状態を確認するとクラッシュと判別できる
//! ようにするモジュール。ジョブを所有する Rust 側（[`crate::job::TrainJob::run_recorded`]）
//! が [`JobRecorder`] で記録を書き、状態確認の入口 [`read_job_status`]（将来の CLI の
//! 状態確認の中核。CLI への配線は TASK-33.x で、REQ-33 の 7 工程の変更は承認事項の
//! ため本 crate では露出しない）が記録を読んで、必要なら `failed`＋クラッシュへ
//! 遷移させる。PoC-19（`03-poc/model-lifecycle`）の「`running` のまま残った記録を
//! 状態確認で `failed` へ」の方式に相当する（証拠種別: テストハーネス）。
//!
//! # ファイル
//!
//! `job_dir` に 2 つを置く（`process.rs` の `request.json` と同じディレクトリ）。
//!
//! - `job.json`: 記録本体（[`JobRecord`]）。`tmp` への書き込み→`sync_all`→`rename` で
//!   原子的に置き換える。データ本文・パス・ワーカーの stderr は含めない。
//! - `job.lock`: 生存確認用の advisory lock（std の `File::try_lock`。unix では
//!   `flock`）。ジョブを所有するプロセスがジョブの生存期間中ずっと保持し、カーネルが
//!   プロセスの終了時（`SIGKILL`・panic を含む）に必ず解放する。`job.json` は
//!   rename で inode が変わるため lock の対象にせず、別ファイルにする。
//!
//! PID での生存確認・シグナル送信はしない。記録の pid は PID 再利用で信頼できず、
//! std に `kill(pid, 0)` は無く、`ps` の解析は同種の脆さを持ち込み、native 呼び出しは
//! `unsafe` か依存追加（いずれも承認事項）になるため。
//!
//! # クラッシュの分類（`supervisor.py::_classify_self_exit` との関係）
//!
//! | 観測点 | 事象 | 分類 |
//! | ------ | ---- | ---- |
//! | supervisor が `_worker` を監視 | シグナル終了で `_classify_self_exit` が `None`（CPU 消費がソフト上限未満の `SIGKILL`・`SIGSEGV`・OOM killer 等の外部要因） | **crash**（[`CrashCause::WorkerSignal`]） |
//! | 同上 | `"cpu"`（`SIGXCPU`、または CPU 消費がソフト上限以上の `SIGKILL`）・`time`・`rss` 等 | crash ではない（`failed`＋[`JobFailure::Error`]。`limit_exceeded` 等。REQ-39 の資源上限） |
//! | Rust が supervisor を監視 | supervisor がシグナルで終了（[`TrainProcessError::TerminatedBySignal`]） | **crash**（[`CrashCause::SupervisorSignal`]） |
//! | 状態確認 | 非終端の記録なのに `job.lock` を誰も保持していない（所有プロセス自身が落ちた） | **crash**（[`CrashCause::OwnerLost`]） |
//! | Rust の壁時計締め切り | [`TrainProcessError::WallTimeout`] | crash ではない（`limit_exceeded`） |
//!
//! `WorkerSignal` の判定は supervisor の固定文言に依存する（学習結果契約に専用
//! フィールドを足すのは契約変更で承認事項のため）。`code == runtime_error` かつ
//! `message` が [`WORKER_SIGNAL_MESSAGE_PREFIX`] の直後に正の整数だけが続く完全一致の
//! ときだけ crash とし、接頭辞は共有 fixture
//! （`fixtures/train_contract/worker_crash_message.json`）で Python 側と照合する。
//! worker 自身が返した同形の文言は supervisor が別文言へ置き換える（偽装防止）。
//!
//! # 書き手と読み手の順序（不変条件）
//!
//! - 書き手: `begin` は `job.lock` の作成・取得→`job.json`（`running`）の書き込みの順。
//!   `finish` は終端記録の書き込みを**終えてから** lock を解放する。
//! - 読み手: 非終端の記録を読んだら `job.lock` の取得を試み、取得できたら lock を
//!   保持したまま `job.json` を読み直す（読んでから lock を取るまでの間に正常終了
//!   したジョブを crash と誤報しない）。なお非終端のまま取得できたときだけ、lock
//!   保持中に限って `failed`＋`OwnerLost` を書き戻す。`job.lock` が無い場合は生存中の
//!   可能性を否定できないため crash と断定しない（[`JobRecordError::LockMissing`]。
//!   fail-closed）。
//! - `finish` を呼ばずに drop（panic 等）された場合も lock は解放されるため、状態確認が
//!   `OwnerLost` を検出する（意図した挙動）。
//! - 読み手同士は `job.check.lock`（blocking の advisory lock）で直列化する。同時に
//!   検出した読み手は先の書き戻しを待ち、取得後に記録を読み直して終端記録を返す
//!   （書き戻し中の `job.lock` 保持を所有者の生存と誤認して `running` と誤報しない）。
//!   次回以降の呼び出しは終端記録を返すだけ（冪等）。
//!
//! # 責務境界・未実装（実装済みを装わない）
//!
//! - `job_dir` の経路の閉じ込めはガード層（REQ-39・TASK-39.x）と CLI 配線（TASK-33.x）
//!   の責務。本モジュールは絶対パスの既存ディレクトリで、`job.json`・`job.lock` が
//!   symlink でないことだけを検査する（検査後の差し替えの余地は残る）。
//! - CLI への状態確認の露出（TASK-33.x）・やり直しの案内と `SIGKILL` フォールバック後の
//!   残置物の掃除（#147・TASK-34.3）は未実装。再開（チェックポイント）は提供しない
//!   （REQ-34）ため記録に `resumable` は持たない。
//! - 記録のスキーマ（`schema_version: 1`）は暫定の内部形式で、spec に明記が無い。

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use fandhe_edge_core::exitcode::ExitCode;
use serde::{Deserialize, Serialize};

use crate::error::TrainProcessError;
use crate::job::JobState;
use crate::limits::MAX_JOB_RECORD_BYTES;
use crate::process::TrainRunEnd;
use crate::result::TrainOutcome;

/// 記録本体のファイル名。
pub const JOB_RECORD_FILE: &str = "job.json";
/// 生存確認用 lock のファイル名。
pub const JOB_LOCK_FILE: &str = "job.lock";
/// 原子的置き換えの一時ファイル名。
/// 状態確認（読み手）同士を直列化する lock ファイル名。所有者の生存確認用 `job.lock` とは別。
pub const JOB_CHECK_LOCK_FILE: &str = "job.check.lock";
const JOB_RECORD_TMP_FILE: &str = "job.json.tmp";
/// 記録の `schema_version`。
pub const JOB_RECORD_SCHEMA_VERSION: u32 = 1;

/// supervisor が worker のシグナル終了を報告する固定文言の接頭辞
/// （`trainer/.../supervisor.py::_WORKER_SIGNAL_MESSAGE_PREFIX` と一致させる。
/// 共有 fixture で照合）。
pub const WORKER_SIGNAL_MESSAGE_PREFIX: &str = "worker terminated by signal ";

/// クラッシュの原因（観測点。モジュール doc の分類表）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrashCause {
    /// `_worker` が資源上限以外の理由でシグナル終了した（supervisor が報告）。
    WorkerSignal,
    /// 直接の子（supervisor）がシグナルで終了した。
    SupervisorSignal,
    /// ジョブを所有する Rust プロセス自身が終了記録なしに消えた（状態確認が検出）。
    OwnerLost,
}

/// `failed` の内訳。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum JobFailure {
    /// クラッシュ（異常終了）。
    Crashed {
        /// 観測点。
        cause: CrashCause,
        /// 終了させたシグナル番号（分かる場合のみ）。
        signal: Option<i32>,
        /// 検出時刻（UNIX 秒）。
        detected_at_unix: u64,
    },
    /// クラッシュ以外の失敗（`limit_exceeded` 等。7 種の終了コード名で表す）。
    Error {
        /// 終了コード名（`ok` は失敗として表現できない）。
        code: ExitCode,
    },
}

/// `job.json` の内容。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobRecord {
    /// スキーマ版。
    pub schema_version: u32,
    /// ジョブ状態。
    pub state: JobState,
    /// 開始時刻（UNIX 秒）。
    pub started_at_unix: u64,
    /// 終端に達した時刻（UNIX 秒。非終端は `None`）。
    pub finished_at_unix: Option<u64>,
    /// `failed` のときの内訳（`failed` のときだけ `Some`）。
    pub failure: Option<JobFailure>,
}

impl JobRecord {
    /// 読み込んだ記録の整合性検査（壊れた組を表現させない。fail-closed）。
    fn validate(&self) -> Result<(), JobRecordError> {
        if self.schema_version != JOB_RECORD_SCHEMA_VERSION {
            return Err(JobRecordError::UnsupportedSchemaVersion);
        }
        let failure_ok = match (self.state, &self.failure) {
            (JobState::Failed, Some(JobFailure::Error { code })) => *code != ExitCode::Ok,
            (JobState::Failed, Some(JobFailure::Crashed { .. })) | (_, None) => {
                self.state != JobState::Failed || self.failure.is_some()
            }
            (_, Some(_)) => false,
        };
        let finished_ok = self.finished_at_unix.is_some() == self.state.is_terminal();
        if failure_ok && finished_ok {
            Ok(())
        } else {
            Err(JobRecordError::Malformed)
        }
    }
}

/// 状態確認（[`read_job_status`]）の結果。CLI 配線時にそのまま JSON 1 つとして出せる形。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JobStatusReport {
    /// ジョブ状態。
    pub state: JobState,
    /// クラッシュ（[`JobFailure::Crashed`]）として終わっているか。
    pub crash_detected: bool,
    /// `failed` の内訳。
    pub failure: Option<JobFailure>,
    /// この呼び出しが記録を `failed` へ書き換えたか（クラッシュの初回検出）。
    pub record_updated: bool,
}

/// ジョブ記録の操作で起こりうるエラー。`Display` は固定の英語文で、パスや
/// ファイル内容を含めない（`.claude/rules/security.md`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JobRecordError {
    /// `job_dir` が絶対パスの既存ディレクトリ（symlink でない）ではない。
    InvalidJobDir,
    /// `job.json`・`job.lock` が既にある（`job_dir` の再利用を拒否する）。
    AlreadyExists,
    /// 別のプロセスが `job.lock` を保持している。
    LockUnavailable,
    /// 非終端の記録なのに `job.lock` が無い（生存中の可能性を否定できない）。
    LockMissing,
    /// `job.json` または `job.lock` が symlink・通常ファイルでない。
    NotRegularFile,
    /// `job.json` が上限（[`MAX_JOB_RECORD_BYTES`]）を超える。
    TooLarge,
    /// `job.json` が壊れている（JSON・未知フィールド・整合性）。
    Malformed,
    /// 未対応の `schema_version`。
    UnsupportedSchemaVersion,
    /// `finish` に渡した終端状態と失敗内訳の組が不正。
    InvalidFinalState,
    /// 入出力エラー。
    Io {
        /// 元のエラー種別。
        kind: std::io::ErrorKind,
    },
}

impl std::fmt::Display for JobRecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::InvalidJobDir => "job directory must be an absolute existing directory",
            Self::AlreadyExists => "job record already exists in the job directory",
            Self::LockUnavailable => "job lock is held by another process",
            Self::LockMissing => "job lock file is missing for a non-terminal job record",
            Self::NotRegularFile => "job record files must be regular files",
            Self::TooLarge => "job record exceeds the size limit",
            Self::Malformed => "job record is malformed",
            Self::UnsupportedSchemaVersion => "job record schema version is not supported",
            Self::InvalidFinalState => "final job state and failure do not match",
            Self::Io { .. } => "job record i/o failed",
        };
        f.write_str(text)
    }
}

impl std::error::Error for JobRecordError {}

fn io_err(e: &std::io::Error) -> JobRecordError {
    JobRecordError::Io { kind: e.kind() }
}

/// 現在時刻（UNIX 秒）。時計が UNIX epoch より前なら 0。
#[must_use]
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `supervisor.py` の固定文言から worker のシグナル番号を取り出す。接頭辞の直後が
/// 正の 10 進整数だけのときに限る（符号・空白・余分な文字は拒否）。
#[must_use]
pub fn parse_worker_signal_message(message: &str) -> Option<i32> {
    let rest = message.strip_prefix(WORKER_SIGNAL_MESSAGE_PREFIX)?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.parse::<i32>().ok().filter(|n| *n > 0)
}

/// 1 回の学習実行の結果を、記録する終端状態と失敗内訳へ分類する（純関数）。
///
/// 分類規則はモジュール doc の表。`now_unix` はクラッシュの検出時刻として使う。
#[must_use]
pub fn classify_run_end(
    result: &Result<TrainRunEnd, TrainProcessError>,
    now_unix: u64,
) -> (JobState, Option<JobFailure>) {
    match result {
        Ok(TrainRunEnd::Completed(run)) => match run.outcome() {
            TrainOutcome::Ok(_) => (JobState::Succeeded, None),
            TrainOutcome::Error(failure) => {
                let code = run.exit_code();
                let signal = (code == ExitCode::RuntimeError)
                    .then(|| parse_worker_signal_message(failure.message()))
                    .flatten();
                let failure = match signal {
                    Some(signal) => JobFailure::Crashed {
                        cause: CrashCause::WorkerSignal,
                        signal: Some(signal),
                        detected_at_unix: now_unix,
                    },
                    None => JobFailure::Error { code },
                };
                (JobState::Failed, Some(failure))
            }
        },
        Ok(TrainRunEnd::Cancelled(_)) => (JobState::Cancelled, None),
        Err(TrainProcessError::TerminatedBySignal) => (
            JobState::Failed,
            Some(JobFailure::Crashed {
                cause: CrashCause::SupervisorSignal,
                signal: None,
                detected_at_unix: now_unix,
            }),
        ),
        Err(e) => (
            JobState::Failed,
            Some(JobFailure::Error {
                code: e.exit_code(),
            }),
        ),
    }
}

/// `job_dir` が絶対パスの既存ディレクトリ（symlink でない）であることを確認する。
fn check_job_dir(job_dir: &Path) -> Result<(), JobRecordError> {
    if !job_dir.is_absolute() {
        return Err(JobRecordError::InvalidJobDir);
    }
    match std::fs::symlink_metadata(job_dir) {
        Ok(meta) if meta.is_dir() => Ok(()),
        _ => Err(JobRecordError::InvalidJobDir),
    }
}

/// `path` が存在するなら通常ファイルであることを確認する（symlink を拒否）。
/// 存在しなければ `Ok(false)`。
fn regular_file_exists(path: &Path) -> Result<bool, JobRecordError> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.is_file() => Ok(true),
        Ok(_) => Err(JobRecordError::NotRegularFile),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(io_err(&e)),
    }
}

fn create_new_private(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        // 記録は所有者だけが読み書きできればよい。
        options.mode(0o600);
    }
    options.open(path)
}

/// 記録を tmp への書き込み→`sync_all`→`rename` で原子的に置き換える。呼び出し元は
/// `job.lock` を保持していること（書き手を 1 人に保つ）。
fn write_record_atomic(job_dir: &Path, record: &JobRecord) -> Result<(), JobRecordError> {
    let tmp = job_dir.join(JOB_RECORD_TMP_FILE);
    // lock 保持中の書き手は 1 人なので、前回の書き込み途中で残った tmp は消してよい
    // （symlink であればリンク自体が消え、`create_new` は追跡しない）。
    match std::fs::remove_file(&tmp) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(io_err(&e)),
    }
    let bytes = serde_json::to_vec(record).map_err(|_| JobRecordError::Malformed)?;
    let write = || -> std::io::Result<()> {
        let mut file = create_new_private(&tmp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, job_dir.join(JOB_RECORD_FILE))
    };
    write().map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        io_err(&e)
    })
}

/// `job.json` を上限つきで読み、厳格に parse・検証する。
fn read_record(job_dir: &Path) -> Result<JobRecord, JobRecordError> {
    let path = job_dir.join(JOB_RECORD_FILE);
    if !regular_file_exists(&path)? {
        return Err(JobRecordError::Io {
            kind: std::io::ErrorKind::NotFound,
        });
    }
    let file = File::open(&path).map_err(|e| io_err(&e))?;
    let len = file.metadata().map_err(|e| io_err(&e))?.len();
    if len > u64::try_from(MAX_JOB_RECORD_BYTES).unwrap_or(u64::MAX) {
        return Err(JobRecordError::TooLarge);
    }
    let mut buf = Vec::new();
    // メタデータ確認後に肥大しても上限を超えて読まない。
    file.take(u64::try_from(MAX_JOB_RECORD_BYTES).unwrap_or(u64::MAX) + 1)
        .read_to_end(&mut buf)
        .map_err(|e| io_err(&e))?;
    if buf.len() > MAX_JOB_RECORD_BYTES {
        return Err(JobRecordError::TooLarge);
    }
    // 版の違いは未知フィールドより先に判別できるよう、版だけを先に見る。
    let value: serde_json::Value =
        serde_json::from_slice(&buf).map_err(|_| JobRecordError::Malformed)?;
    match value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
    {
        Some(v) if v == u64::from(JOB_RECORD_SCHEMA_VERSION) => {}
        Some(_) => return Err(JobRecordError::UnsupportedSchemaVersion),
        None => return Err(JobRecordError::Malformed),
    }
    let record: JobRecord = serde_json::from_value(value).map_err(|_| JobRecordError::Malformed)?;
    record.validate()?;
    Ok(record)
}

/// ジョブの所有者が持つ記録の書き手。生存期間中 `job.lock` を保持する。
///
/// [`crate::job::TrainJob::run_recorded`] が [`Self::begin`]→学習実行→[`Self::finish`]
/// の順に使う。`finish` を呼ばずに drop された場合の挙動はモジュール doc。
#[derive(Debug)]
pub struct JobRecorder {
    // 保持し続けることが目的のフィールド（drop でカーネルが lock を解放する）。
    _lock: File,
    job_dir: PathBuf,
    started_at_unix: u64,
}

impl JobRecorder {
    /// `job_dir` に `job.lock` を作って保持し、`running` の記録を書く。
    ///
    /// # Errors
    /// `job_dir` が不正・記録ファイルが既にある・書き込み失敗。失敗時は `job.lock` を
    /// 残さない（可能な範囲で）。
    pub fn begin(job_dir: &Path, now_unix: u64) -> Result<Self, JobRecordError> {
        check_job_dir(job_dir)?;
        let record_path = job_dir.join(JOB_RECORD_FILE);
        let lock_path = job_dir.join(JOB_LOCK_FILE);
        if regular_file_exists(&record_path)? || regular_file_exists(&lock_path)? {
            return Err(JobRecordError::AlreadyExists);
        }
        let lock = create_new_private(&lock_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                JobRecordError::AlreadyExists
            } else {
                io_err(&e)
            }
        })?;
        let rollback = |e: JobRecordError| {
            let _ = std::fs::remove_file(&lock_path);
            e
        };
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(rollback(JobRecordError::LockUnavailable)),
            Err(TryLockError::Error(e)) => return Err(rollback(io_err(&e))),
        }
        let record = JobRecord {
            schema_version: JOB_RECORD_SCHEMA_VERSION,
            state: JobState::Running,
            started_at_unix: now_unix,
            finished_at_unix: None,
            failure: None,
        };
        write_record_atomic(job_dir, &record).map_err(rollback)?;
        Ok(Self {
            _lock: lock,
            job_dir: job_dir.to_path_buf(),
            started_at_unix: now_unix,
        })
    }

    /// 終端記録を原子的に書き、**書き終えてから** lock を解放する。
    ///
    /// # Errors
    /// `state` が終端でない・`failure` が `Failed` と対応しない（[`JobRecordError::InvalidFinalState`]。
    /// この場合は何も書かず、drop 後に状態確認が `OwnerLost` を検出する）、または書き込み失敗。
    pub fn finish(
        self,
        state: JobState,
        failure: Option<JobFailure>,
        now_unix: u64,
    ) -> Result<(), JobRecordError> {
        let record = JobRecord {
            schema_version: JOB_RECORD_SCHEMA_VERSION,
            state,
            started_at_unix: self.started_at_unix,
            finished_at_unix: Some(now_unix),
            failure,
        };
        if !state.is_terminal() {
            return Err(JobRecordError::InvalidFinalState);
        }
        record
            .validate()
            .map_err(|_| JobRecordError::InvalidFinalState)?;
        write_record_atomic(&self.job_dir, &record)
        // ここで `self` が drop され lock が解放される（書き込み後）。
    }
}

fn report_of(record: &JobRecord, record_updated: bool) -> JobStatusReport {
    JobStatusReport {
        state: record.state,
        crash_detected: matches!(record.failure, Some(JobFailure::Crashed { .. })),
        failure: record.failure.clone(),
        record_updated,
    }
}

/// ジョブ状態を確認する（状態確認の中核。CLI への配線は TASK-33.x）。
///
/// 終端の記録はそのまま返す。非終端の記録は `job.lock` の保持者がいなければ
/// 所有プロセスが終了記録なしに消えたと判断し、`failed`＋[`CrashCause::OwnerLost`] を
/// 書き戻して返す（初回のみ `record_updated: true`。以降は冪等）。順序の不変条件は
/// モジュール doc。
///
/// # Errors
/// `job_dir` 不正・記録の欠落や破損・`job.lock` の欠落（[`JobRecordError::LockMissing`]）・
/// 書き戻し失敗。
pub fn read_job_status(job_dir: &Path, now_unix: u64) -> Result<JobStatusReport, JobRecordError> {
    check_job_dir(job_dir)?;
    let record = read_record(job_dir)?;
    if record.state.is_terminal() {
        return Ok(report_of(&record, false));
    }
    let lock_path = job_dir.join(JOB_LOCK_FILE);
    if !regular_file_exists(&lock_path)? {
        return Err(JobRecordError::LockMissing);
    }
    // 読み手同士を直列化する。書き戻し中の別の読み手による `job.lock` の保持を
    // 「所有者が生存中」と誤認しないため、`job.lock` の取得試行と書き戻しをこの
    // lock の保持中に行い、待たされた読み手は取得後に記録を読み直す。
    let check_path = job_dir.join(JOB_CHECK_LOCK_FILE);
    regular_file_exists(&check_path)?;
    let mut check_options = OpenOptions::new();
    check_options.write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        check_options.mode(0o600);
    }
    let check_lock = check_options.open(&check_path).map_err(|e| io_err(&e))?;
    check_lock.lock().map_err(|e| io_err(&e))?;
    let record = read_record(job_dir)?;
    if record.state.is_terminal() {
        return Ok(report_of(&record, false));
    }
    let lock = File::open(&lock_path).map_err(|e| io_err(&e))?;
    match lock.try_lock() {
        // 読み手は直列化済みなので、取れないのは所有プロセスが生存中のため。
        Err(TryLockError::WouldBlock) => Ok(report_of(&record, false)),
        Err(TryLockError::Error(e)) => Err(io_err(&e)),
        Ok(()) => {
            // lock を保持したまま読み直す。最初の読み込み後に正常終了していれば
            // その終端記録を返し、crash と誤報しない。
            let current = read_record(job_dir)?;
            if current.state.is_terminal() {
                return Ok(report_of(&current, false));
            }
            let crashed = JobRecord {
                schema_version: JOB_RECORD_SCHEMA_VERSION,
                state: JobState::Failed,
                started_at_unix: current.started_at_unix,
                finished_at_unix: Some(now_unix),
                failure: Some(JobFailure::Crashed {
                    cause: CrashCause::OwnerLost,
                    signal: None,
                    detected_at_unix: now_unix,
                }),
            };
            write_record_atomic(job_dir, &crashed)?;
            Ok(report_of(&crashed, true))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{Device, TrainRequest, TrainRequestParams};

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-job-record-{}-{}-{}",
            name,
            std::process::id(),
            unix_now()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create tmp dir");
        dir
    }

    fn worker_error(code: ExitCode, message: &str) -> Result<TrainRunEnd, TrainProcessError> {
        let code_name = match code {
            ExitCode::LimitExceeded => "limit_exceeded",
            _ => "runtime_error",
        };
        let json = format!(r#"{{"status":"error","code":"{code_name}","message":"{message}"}}"#);
        let request = TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: "/fandhe-edge-job-record-root".to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: None,
            rss_limit_bytes: None,
        })
        .expect("valid request");
        let outcome =
            TrainOutcome::from_worker_stdout(json.as_bytes(), &request).expect("valid error");
        Ok(TrainRunEnd::Completed(crate::process::TrainRun::for_test(
            outcome, code,
        )))
    }

    /// REQ-34・TASK-34.2: supervisor が報告した worker のシグナル終了は crash。
    #[test]
    fn req34_worker_signal_message_is_crash() {
        let r = worker_error(ExitCode::RuntimeError, "worker terminated by signal 9");
        assert_eq!(
            classify_run_end(&r, 100),
            (
                JobState::Failed,
                Some(JobFailure::Crashed {
                    cause: CrashCause::WorkerSignal,
                    signal: Some(9),
                    detected_at_unix: 100
                })
            )
        );
    }

    /// REQ-34・REQ-39: CPU 上限による自己終了（`_classify_self_exit` が `"cpu"`）は
    /// crash ではなく `limit_exceeded`。
    #[test]
    fn req34_cpu_limit_exit_is_not_crash() {
        let r = worker_error(
            ExitCode::LimitExceeded,
            "worker exceeded the cpu limit and was terminated",
        );
        assert_eq!(
            classify_run_end(&r, 1),
            (
                JobState::Failed,
                Some(JobFailure::Error {
                    code: ExitCode::LimitExceeded
                })
            )
        );
    }

    /// REQ-34: 文言が完全一致でない・runtime_error でない場合は crash にしない。
    #[test]
    fn req34_non_matching_messages_are_errors() {
        let r = worker_error(ExitCode::RuntimeError, "worker terminated by signal x");
        assert_eq!(
            classify_run_end(&r, 1).1,
            Some(JobFailure::Error {
                code: ExitCode::RuntimeError
            })
        );
        let r = worker_error(ExitCode::LimitExceeded, "worker terminated by signal 9");
        assert_eq!(
            classify_run_end(&r, 1).1,
            Some(JobFailure::Error {
                code: ExitCode::LimitExceeded
            })
        );
    }

    /// REQ-34: supervisor 自身のシグナル終了は crash、壁時計超過は crash ではない。
    #[test]
    fn req34_process_errors_are_classified() {
        assert_eq!(
            classify_run_end(&Err(TrainProcessError::TerminatedBySignal), 7),
            (
                JobState::Failed,
                Some(JobFailure::Crashed {
                    cause: CrashCause::SupervisorSignal,
                    signal: None,
                    detected_at_unix: 7
                })
            )
        );
        assert_eq!(
            classify_run_end(
                &Err(TrainProcessError::WallTimeout {
                    limit_ms: 1,
                    child_reaped: true
                }),
                7
            ),
            (
                JobState::Failed,
                Some(JobFailure::Error {
                    code: ExitCode::LimitExceeded
                })
            )
        );
    }

    /// REQ-34: 実行中は所有者が lock を保持しており crash と判定されない。
    #[test]
    fn req34_running_owner_is_not_crash() {
        let dir = tmp_dir("running");
        let recorder = JobRecorder::begin(&dir, 10).expect("begin");
        let report = read_job_status(&dir, 20).expect("status");
        assert_eq!(
            report,
            JobStatusReport {
                state: JobState::Running,
                crash_detected: false,
                failure: None,
                record_updated: false
            }
        );
        drop(recorder);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-34・PoC-19 相当: `finish` されずに所有者が消えると、状態確認が `failed`＋
    /// `owner_lost` を検出し、2 回目以降は冪等（`detected_at_unix` 不変）。
    #[test]
    fn req34_owner_lost_is_detected_once_and_idempotent() {
        let dir = tmp_dir("owner-lost");
        drop(JobRecorder::begin(&dir, 10).expect("begin"));
        let first = read_job_status(&dir, 20).expect("first");
        let expected_failure = JobFailure::Crashed {
            cause: CrashCause::OwnerLost,
            signal: None,
            detected_at_unix: 20,
        };
        assert_eq!(
            first,
            JobStatusReport {
                state: JobState::Failed,
                crash_detected: true,
                failure: Some(expected_failure.clone()),
                record_updated: true
            }
        );
        let second = read_job_status(&dir, 99).expect("second");
        assert_eq!(
            second,
            JobStatusReport {
                state: JobState::Failed,
                crash_detected: true,
                failure: Some(expected_failure),
                record_updated: false
            }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-34: 所有者の消滅後に複数の読み手が同時に状態確認しても、全員が
    /// `failed`＋crash を返し（`running` と誤報しない）、書き戻しは 1 回だけ。
    #[test]
    fn req34_concurrent_readers_all_see_crash() {
        for round in 0..20 {
            let dir = tmp_dir(&format!("concurrent-{round}"));
            drop(JobRecorder::begin(&dir, 10).expect("begin"));
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let d = dir.clone();
                    std::thread::spawn(move || read_job_status(&d, 20).expect("status"))
                })
                .collect();
            let reports: Vec<JobStatusReport> = handles
                .into_iter()
                .map(|h| h.join().expect("join"))
                .collect();
            assert!(
                reports
                    .iter()
                    .all(|r| r.state == JobState::Failed && r.crash_detected)
            );
            assert_eq!(reports.iter().filter(|r| r.record_updated).count(), 1);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// REQ-34: 正常終了は succeeded・crash なし。
    #[test]
    fn req34_finished_job_is_terminal_without_crash() {
        let dir = tmp_dir("finished");
        let recorder = JobRecorder::begin(&dir, 10).expect("begin");
        recorder
            .finish(JobState::Succeeded, None, 30)
            .expect("finish");
        let report = read_job_status(&dir, 40).expect("status");
        assert_eq!(
            report,
            JobStatusReport {
                state: JobState::Succeeded,
                crash_detected: false,
                failure: None,
                record_updated: false
            }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-34: `finish` は終端でない状態・failed と対応しない失敗内訳を拒否する。
    #[test]
    fn req34_finish_rejects_inconsistent_final_state() {
        let dir = tmp_dir("bad-finish");
        let recorder = JobRecorder::begin(&dir, 1).expect("begin");
        assert_eq!(
            recorder.finish(JobState::Running, None, 2),
            Err(JobRecordError::InvalidFinalState)
        );
        let recorder2 = tmp_dir("bad-finish2");
        let r = JobRecorder::begin(&recorder2, 1).expect("begin");
        assert_eq!(
            r.finish(JobState::Failed, None, 2),
            Err(JobRecordError::InvalidFinalState)
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&recorder2);
    }

    /// REQ-34: 同じ `job_dir` の再利用は拒否する。
    #[test]
    fn req34_begin_twice_is_rejected() {
        let dir = tmp_dir("twice");
        let _first = JobRecorder::begin(&dir, 1).expect("begin");
        assert_eq!(
            JobRecorder::begin(&dir, 2).map(|_| ()),
            Err(JobRecordError::AlreadyExists)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-34・REQ-39: 相対パス・存在しないディレクトリは拒否する。
    #[test]
    fn req34_invalid_job_dir_is_rejected() {
        assert_eq!(
            JobRecorder::begin(Path::new("relative/dir"), 1).map(|_| ()),
            Err(JobRecordError::InvalidJobDir)
        );
        assert_eq!(
            read_job_status(Path::new("/fandhe-edge-no-such-dir-146"), 1),
            Err(JobRecordError::InvalidJobDir)
        );
    }

    /// REQ-34: 非終端の記録で `job.lock` が無い場合は crash と断定せず拒否する。
    #[test]
    fn req34_missing_lock_is_fail_closed() {
        let dir = tmp_dir("no-lock");
        drop(JobRecorder::begin(&dir, 1).expect("begin"));
        std::fs::remove_file(dir.join(JOB_LOCK_FILE)).expect("remove lock");
        assert_eq!(read_job_status(&dir, 2), Err(JobRecordError::LockMissing));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write_raw(dir: &Path, body: &str) {
        std::fs::write(dir.join(JOB_RECORD_FILE), body).expect("write raw record");
    }

    /// REQ-34・REQ-39: 未知フィールド・未対応の版・壊れた JSON・不整合を拒否する。
    #[test]
    fn req34_malformed_records_are_rejected() {
        let dir = tmp_dir("malformed");
        let base = r#""state":"succeeded","started_at_unix":1,"finished_at_unix":2,"failure":null"#;
        write_raw(&dir, &format!(r#"{{"schema_version":1,{base},"extra":1}}"#));
        assert_eq!(read_job_status(&dir, 3), Err(JobRecordError::Malformed));
        write_raw(&dir, &format!(r#"{{"schema_version":2,{base}}}"#));
        assert_eq!(
            read_job_status(&dir, 3),
            Err(JobRecordError::UnsupportedSchemaVersion)
        );
        write_raw(&dir, "{not json");
        assert_eq!(read_job_status(&dir, 3), Err(JobRecordError::Malformed));
        // failed なのに failure が無い。
        write_raw(
            &dir,
            r#"{"schema_version":1,"state":"failed","started_at_unix":1,"finished_at_unix":2,"failure":null}"#,
        );
        assert_eq!(read_job_status(&dir, 3), Err(JobRecordError::Malformed));
        // 終端なのに finished_at が無い。
        write_raw(
            &dir,
            r#"{"schema_version":1,"state":"succeeded","started_at_unix":1,"finished_at_unix":null,"failure":null}"#,
        );
        assert_eq!(read_job_status(&dir, 3), Err(JobRecordError::Malformed));
        // failure の未知フィールド。
        write_raw(
            &dir,
            r#"{"schema_version":1,"state":"failed","started_at_unix":1,"finished_at_unix":2,"failure":{"kind":"error","code":"runtime_error","x":1}}"#,
        );
        assert_eq!(read_job_status(&dir, 3), Err(JobRecordError::Malformed));
        // ok は失敗として表現できない。
        write_raw(
            &dir,
            r#"{"schema_version":1,"state":"failed","started_at_unix":1,"finished_at_unix":2,"failure":{"kind":"error","code":"ok"}}"#,
        );
        assert_eq!(read_job_status(&dir, 3), Err(JobRecordError::Malformed));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-39: 上限超過の記録は読み込み前に拒否する。
    #[test]
    fn req34_oversized_record_is_rejected() {
        let dir = tmp_dir("oversized");
        write_raw(&dir, &" ".repeat(MAX_JOB_RECORD_BYTES + 1));
        assert_eq!(read_job_status(&dir, 1), Err(JobRecordError::TooLarge));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-39: `job.json` が symlink なら追跡せず拒否する。
    #[cfg(unix)]
    #[test]
    fn req34_symlinked_record_is_rejected() {
        let dir = tmp_dir("symlink");
        let target = dir.join("real.json");
        std::fs::write(&target, "{}").expect("write target");
        std::os::unix::fs::symlink(&target, dir.join(JOB_RECORD_FILE)).expect("symlink");
        assert_eq!(
            read_job_status(&dir, 1),
            Err(JobRecordError::NotRegularFile)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-34: `WorkerSignal` の判定は接頭辞の直後が正の整数だけのときに限る。
    #[test]
    fn req34_parse_worker_signal_message() {
        assert_eq!(
            parse_worker_signal_message("worker terminated by signal 9"),
            Some(9)
        );
        for bad in [
            "worker terminated by signal ",
            "worker terminated by signal 0",
            "worker terminated by signal -9",
            "worker terminated by signal +9",
            "worker terminated by signal 9 ",
            "worker terminated by signal 99999999999",
            "other",
        ] {
            assert_eq!(parse_worker_signal_message(bad), None, "{bad}");
        }
    }
}
