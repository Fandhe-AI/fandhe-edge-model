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
//! - `job.json`: 記録本体（[`JobRecord`]）。`tmp` への書き込み→`sync_all`→`rename`→親ディレクトリの
//!   `sync_all` で原子的に置き換える。データ本文・パス・ワーカーの stderr は含めない。
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
//! - 書き手: `begin` は lock 済みの `job.lock` の公開→`job.json`（`running`）の書き込みの順。
//!   `finish` は終端記録の書き込みを**終えてから** lock を解放する。
//! - 読み手: 非終端の記録を読んだら `job.lock` の取得を試み、取得できたら lock を
//!   保持したまま `job.json` を読み直す（読んでから lock を取るまでの間に正常終了
//!   したジョブを crash と誤報しない）。なお非終端のまま取得できたときだけ、lock
//!   保持中に限って `failed`＋`OwnerLost` を書き戻す。`job.lock` が無い場合は生存中の
//!   可能性を否定できないため crash と断定しない（[`JobRecordError::LockMissing`]。
//!   fail-closed）。
//! - `begin` の `job.lock` は lock 済みの一時ファイルの `hard_link` で公開する（公開の
//!   瞬間から所有者が lock 保持中）。公開後〜`job.json` 確定前に所有プロセスが落ちて
//!   `job.lock` だけが残った場合、状態確認は `job.json` 無しでも lock を取れれば
//!   `failed`＋`OwnerLost` の記録を新規に書いて回復する（取れなければ初期化中の
//!   `running`）。公開前に落ちた場合は `job.lock.init.*` が残りうるが状態確認には
//!   影響しない（掃除は TASK-34.3・#147）。
//! - `job.lock` の fd は `O_CLOEXEC`（std の `File` の既定）で開くため、学習ワーカー等の
//!   子プロセスへ `exec` 後は継承されない。所有者の消滅後に子プロセスが lock を保持し
//!   続けて crash を見逃すことはない（`fork`〜`exec` 間の一過性の保持だけは
//!   [`try_lock_settled`] の再試行で吸収する。テスト
//!   `req34_owner_lost_is_detected_while_spawned_child_survives` で固定）。
//! - `begin` が `job.json` の書き込み（rename 後の `sync_dir` を含む）に失敗した場合は、
//!   `job.json` と `job.lock` の両方を取り除き、`job_dir` を再利用できる状態に戻す。
//! - 所有者と同一プロセスから状態確認を呼んでも、プロセス内レジストリ（dev, ino）で
//!   所有者の生存を判定し `job.lock` の取得を試みない（fcntl 系の lock 実装では同一
//!   プロセス内の取得が成功し、close で所有者の lock が失われるため）。
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

use std::collections::HashSet;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
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
/// `begin` の lock 済み一時ファイル名の重複回避用（同一プロセス内の並行 `begin`）。
static INIT_COUNTER: AtomicU64 = AtomicU64::new(0);
/// このプロセス内で所有者（[`JobRecorder`]）が保持中の `job.lock` の識別子（dev, ino）。
///
/// 状態確認が所有者と同一プロセスで呼ばれた場合に、advisory lock の実装差（fcntl 系は
/// プロセス単位で、同一プロセス内の取得が成功し、close で所有者の lock まで失われる）に
/// 依らず「所有者は生存中」と判定するための補助（REQ-34。別プロセスからの確認は lock で判定）。
static HELD_LOCKS: Mutex<Option<HashSet<(u64, u64)>>> = Mutex::new(None);

/// `file` の (dev, ino)。unix 以外では `None`（レジストリを使わない）。
fn file_identity(file: &File) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        file.metadata().ok().map(|m| (m.dev(), m.ino()))
    }
    #[cfg(not(unix))]
    {
        let _ = file;
        None
    }
}

/// [`HELD_LOCKS`] への登録。drop で解除する。
#[derive(Debug)]
struct InProcessHold(Option<(u64, u64)>);

impl InProcessHold {
    fn register(file: &File) -> Self {
        let id = file_identity(file);
        if let Some(id) = id {
            let mut guard = HELD_LOCKS.lock().unwrap_or_else(|e| e.into_inner());
            guard.get_or_insert_with(HashSet::new).insert(id);
        }
        Self(id)
    }
}

impl Drop for InProcessHold {
    fn drop(&mut self) {
        if let Some(id) = self.0 {
            let mut guard = HELD_LOCKS.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(set) = guard.as_mut() {
                set.remove(&id);
            }
        }
    }
}

/// `file` が同一プロセス内の所有者に保持されているか。
fn held_in_process(file: &File) -> bool {
    file_identity(file).is_some_and(|id| {
        let guard = HELD_LOCKS.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().is_some_and(|set| set.contains(&id))
    })
}

/// 他者保持の `job.lock` を生存と断定する前の再試行回数と間隔（[`try_lock_settled`]）。
const LOCK_SETTLE_RETRIES: u32 = 10;
const LOCK_SETTLE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(10);
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
        std::fs::rename(&tmp, job_dir.join(JOB_RECORD_FILE))?;
        // rename の永続化のため親ディレクトリも sync する。電源断で rename が失われ、
        // 正常終了したジョブを後から `OwnerLost` と誤記録するのを防ぐ（REQ-34）。
        sync_dir(job_dir)
    };
    write().map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        io_err(&e)
    })
}

/// ディレクトリのエントリ変更（rename）を永続化する。unix 以外ではディレクトリの
/// open が使えないため何もしない。
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        File::open(dir)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        Ok(())
    }
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
    // 同一プロセス内の状態確認に所有者の生存を示す（drop で解除）。
    _hold: InProcessHold,
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
        // `job.lock` は「既に lock 済み」の状態でだけ公開する。先に `job.lock` を作ってから
        // lock すると、作成〜取得の隙間に状態確認が lock を取り、初期化中の所有者を
        // 落ちたと誤検出する。lock 済みの一時ファイルを `hard_link`（既存なら失敗する
        // 排他的な公開）で `job.lock` にする。一時ファイルは公開後に消す（公開前に
        // 落ちた場合の残置物は `job.lock` ではないため状態確認には影響しない）。
        let init_path = job_dir.join(format!(
            "{JOB_LOCK_FILE}.init.{}.{}",
            std::process::id(),
            INIT_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let lock = create_new_private(&init_path).map_err(|e| io_err(&e))?;
        let discard_init = || {
            let _ = std::fs::remove_file(&init_path);
        };
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                discard_init();
                return Err(JobRecordError::LockUnavailable);
            }
            Err(TryLockError::Error(e)) => {
                discard_init();
                return Err(io_err(&e));
            }
        }
        let hold = InProcessHold::register(&lock);
        let published = std::fs::hard_link(&init_path, &lock_path);
        discard_init();
        published.map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                JobRecordError::AlreadyExists
            } else {
                io_err(&e)
            }
        })?;
        // ここから `job.lock` は自分のもの。以降の失敗では取り除く。
        // `job.json` は rename 後の `sync_dir` の失敗でも公開済みになりうる。非終端の
        // 記録だけが残ると `LockMissing`・`AlreadyExists` で `job_dir` を回復も再利用も
        // できないため、lock 保持中に記録も取り除いてから lock を外す。
        let rollback = |e: JobRecordError| {
            let _ = std::fs::remove_file(&record_path);
            let _ = std::fs::remove_file(&lock_path);
            e
        };
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
            _hold: hold,
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
    let record_path = job_dir.join(JOB_RECORD_FILE);
    let lock_path = job_dir.join(JOB_LOCK_FILE);
    // `job.json` が無く `job.lock` だけがある場合は、`begin` が `job.lock` の公開後・
    // `job.json`（`running`）の確定前に所有プロセスが落ちた残置物か、初期化中の
    // 所有者かのどちらか。lock の保持有無で区別する（下の直列化区間で判定）。
    let record_exists = regular_file_exists(&record_path)?;
    let lock_exists = regular_file_exists(&lock_path)?;
    let first = if record_exists {
        let record = read_record(job_dir)?;
        if record.state.is_terminal() {
            return Ok(report_of(&record, false));
        }
        Some(record)
    } else {
        None
    };
    if !lock_exists {
        return match first {
            // 非終端の記録なのに lock が無い。生存中の可能性を否定できない。
            Some(_) => Err(JobRecordError::LockMissing),
            // 何も始まっていない（従来どおり記録欠落）。
            None => Err(JobRecordError::Io {
                kind: std::io::ErrorKind::NotFound,
            }),
        };
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
    let current = read_current(job_dir)?;
    if let Some(record) = current.as_ref().filter(|r| r.state.is_terminal()) {
        return Ok(report_of(record, false));
    }
    let lock = File::open(&lock_path).map_err(|e| io_err(&e))?;
    // 同一プロセスの所有者が保持中なら、lock の取得を試みず（fcntl 系の実装では取得が
    // 成功して close 時に所有者の lock を奪うため）生存中として扱う。
    let alive_in_process = held_in_process(&lock);
    let acquired = if alive_in_process {
        false
    } else {
        try_lock_settled(&lock)?
    };
    match acquired {
        // 読み手は直列化済みなので、取れないのは所有プロセスが生存中のため
        // （記録がまだ無い場合は初期化中。`running` として返す）。
        false => Ok(match current {
            Some(record) => report_of(&record, false),
            None => report_of(&initializing_record(&lock_path, now_unix), false),
        }),
        true => {
            // lock を保持したまま読み直す。最初の読み込み後に正常終了していれば
            // その終端記録を返し、crash と誤報しない。
            let current = read_current(job_dir)?;
            if let Some(record) = current.as_ref().filter(|r| r.state.is_terminal()) {
                return Ok(report_of(record, false));
            }
            let started_at_unix = match current {
                Some(record) => record.started_at_unix,
                // `job.json` の確定前に落ちた場合の開始時刻は、公開済みの `job.lock`
                // の更新時刻で近似する（無ければ検出時刻）。
                None => initializing_record(&lock_path, now_unix).started_at_unix,
            };
            let crashed = JobRecord {
                schema_version: JOB_RECORD_SCHEMA_VERSION,
                state: JobState::Failed,
                started_at_unix,
                finished_at_unix: Some(now_unix.max(started_at_unix)),
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

/// `job.json` があれば読む。無ければ `None`（`begin` の初期化途中）。
fn read_current(job_dir: &Path) -> Result<Option<JobRecord>, JobRecordError> {
    if regular_file_exists(&job_dir.join(JOB_RECORD_FILE))? {
        read_record(job_dir).map(Some)
    } else {
        Ok(None)
    }
}

/// `job.json` 確定前の初期化中（または初期化途中で落ちた）ジョブの暫定記録。
/// 開始時刻は `job.lock` の更新時刻（`now_unix` を上限）。取れなければ `now_unix`。
fn initializing_record(lock_path: &Path, now_unix: u64) -> JobRecord {
    let started_at_unix = std::fs::metadata(lock_path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(now_unix, |d| d.as_secs().min(now_unix));
    JobRecord {
        schema_version: JOB_RECORD_SCHEMA_VERSION,
        state: JobState::Running,
        started_at_unix,
        finished_at_unix: None,
        failure: None,
    }
}

/// `lock` の取得を試み、取れたら `true`、他者が保持していれば `false`。
///
/// 他者の保持が「所有者の生存」ではなく、同一マシンで並行する `fork`〜`exec` 間の
/// 子プロセスが継承した fd による一過性のもの（`O_CLOEXEC` は `exec` で閉じるため
/// 短時間だけ保持が見える）である場合を除くため、短時間だけ再試行してから
/// 生存と判断する（`running` の確認は上限 [`LOCK_SETTLE_RETRIES`] × 間隔だけ遅れる）。
fn try_lock_settled(lock: &File) -> Result<bool, JobRecordError> {
    for attempt in 0..=LOCK_SETTLE_RETRIES {
        match lock.try_lock() {
            Ok(()) => return Ok(true),
            Err(TryLockError::Error(e)) => return Err(io_err(&e)),
            Err(TryLockError::WouldBlock) => {
                if attempt < LOCK_SETTLE_RETRIES {
                    std::thread::sleep(LOCK_SETTLE_INTERVAL);
                }
            }
        }
    }
    Ok(false)
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

    /// REQ-34: 所有者と同一プロセスでの状態確認を繰り返しても crash と誤記録せず、
    /// 所有者の lock も奪われないため、その後の `finish` の終端記録が正しく読める。
    #[test]
    fn req34_in_process_status_checks_do_not_steal_owner_lock() {
        let dir = tmp_dir("inproc");
        let recorder = JobRecorder::begin(&dir, 10).expect("begin");
        for _ in 0..3 {
            let report = read_job_status(&dir, 20).expect("status");
            assert_eq!(report.state, JobState::Running);
            assert!(!report.crash_detected);
            assert!(!report.record_updated);
        }
        recorder
            .finish(JobState::Succeeded, None, 30)
            .expect("finish");
        let report = read_job_status(&dir, 40).expect("status");
        assert_eq!(report.state, JobState::Succeeded);
        assert!(!report.crash_detected);
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

    /// REQ-34: 所有者の lock は子プロセスへ継承されない（`O_CLOEXEC`）。所有者が
    /// `finish` せず消えたとき、`exec` 済みの子プロセスが生き残っていても crash を検出する。
    #[cfg(unix)]
    #[test]
    fn req34_owner_lost_is_detected_while_spawned_child_survives() {
        let dir = tmp_dir("child-survives");
        let recorder = JobRecorder::begin(&dir, 10).expect("begin");
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        drop(recorder);
        let report = read_job_status(&dir, 20);
        let _ = child.kill();
        let _ = child.wait();
        let report = report.expect("status");
        assert_eq!(report.state, JobState::Failed);
        assert!(report.crash_detected);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-34: `job.json` が無い（初期化途中）状態でも、`exec` 済みの子プロセスが
    /// 生き残っていて所有者の lock が継承されていなければ crash として回復できる。
    #[cfg(unix)]
    #[test]
    fn req34_lock_only_owner_lost_is_recovered_while_child_survives() {
        let dir = tmp_dir("lock-only-child");
        let lock_path = dir.join(JOB_LOCK_FILE);
        std::fs::write(&lock_path, b"").expect("lock file");
        let lock = File::open(&lock_path).expect("open");
        lock.try_lock().expect("hold");
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        drop(lock);
        let report = read_job_status(&dir, 20);
        let _ = child.kill();
        let _ = child.wait();
        let report = report.expect("status");
        assert_eq!(report.state, JobState::Failed);
        assert!(report.crash_detected);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-34: `begin` の `job.json` 書き込みが失敗しても `job.json`・`job.lock` を残さず、
    /// `job_dir` を再利用できる（rename 後の `sync_dir` 失敗の後始末と同じ経路）。
    #[test]
    fn req34_begin_failure_leaves_dir_reusable() {
        let dir = tmp_dir("begin-fail");
        // tmp の位置をディレクトリで塞ぎ、書き込みを失敗させる。
        std::fs::create_dir(dir.join(JOB_RECORD_TMP_FILE)).expect("block tmp");
        assert!(JobRecorder::begin(&dir, 10).is_err());
        assert!(!dir.join(JOB_LOCK_FILE).exists());
        assert!(!dir.join(JOB_RECORD_FILE).exists());
        std::fs::remove_dir(dir.join(JOB_RECORD_TMP_FILE)).expect("unblock");
        let recorder = JobRecorder::begin(&dir, 11).expect("reuse");
        drop(recorder);
        let _ = std::fs::remove_dir_all(&dir);
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

    /// REQ-34: `begin` が `job.json` の確定前に落ちて `job.lock` だけが残った場合、状態確認が
    /// `failed`＋`owner_lost` へ回復し（冪等）、記録欠落で失敗し続けない。
    #[test]
    fn req34_lock_without_record_is_recovered_as_owner_lost() {
        let dir = tmp_dir("lock-only");
        std::fs::write(dir.join(JOB_LOCK_FILE), b"").expect("leftover lock");
        let first = read_job_status(&dir, 50).expect("first");
        assert_eq!(first.state, JobState::Failed);
        assert!(first.crash_detected);
        assert!(first.record_updated);
        let second = read_job_status(&dir, 99).expect("second");
        assert_eq!(second.failure, first.failure);
        assert!(!second.record_updated);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-34: `job.json` が無くても `job.lock` を所有者が保持中なら初期化中の `running`
    /// として返し、crash と誤報・記録の書き込みをしない。
    #[test]
    fn req34_lock_held_without_record_is_initializing() {
        let dir = tmp_dir("lock-held");
        let lock = File::create(dir.join(JOB_LOCK_FILE)).expect("lock");
        lock.try_lock().expect("hold");
        let report = read_job_status(&dir, 5).expect("status");
        assert_eq!(report.state, JobState::Running);
        assert!(!report.crash_detected);
        assert!(!report.record_updated);
        assert!(!dir.join(JOB_RECORD_FILE).exists());
        drop(lock);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-34: 状態確認による回復後の `begin` は既存記録として拒否する（`job_dir` は再利用しない）。
    #[test]
    fn req34_begin_after_recovery_is_rejected_without_leftovers() {
        let dir = tmp_dir("after-recovery");
        std::fs::write(dir.join(JOB_LOCK_FILE), b"").expect("leftover lock");
        read_job_status(&dir, 1).expect("recover");
        assert_eq!(
            JobRecorder::begin(&dir, 2).map(|_| ()),
            Err(JobRecordError::AlreadyExists)
        );
        let names: Vec<String> = std::fs::read_dir(&dir)
            .expect("read_dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.iter().all(|n| !n.contains(".init.")), "{names:?}");
        let _ = std::fs::remove_dir_all(&dir);
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
