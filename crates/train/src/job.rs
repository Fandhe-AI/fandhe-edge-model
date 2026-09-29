//! 学習ジョブの状態遷移とキャンセル操作（REQ-34・TASK-34.1-1・issue #144）。
//!
//! Rust 側ジョブ管理（CLI `train` 工程への配線は TASK-33.x）が、
//! [`crate::process::run_train_cancellable`] を包んで「実行中 → キャンセル中
//! → キャンセル済み」等の状態を管理する。語彙は PoC-19 の
//! `queued → running → succeeded|failed|cancelled` に合わせる。
//!
//! # 遷移表（[`transition`]）
//!
//! | from | event | to |
//! | ---- | ----- | -- |
//! | Queued | Start | Running |
//! | Queued | CancelRequested | Cancelled |
//! | Running | CancelRequested | Cancelling |
//! | Running | Completed | Succeeded |
//! | Running | FailedRun | Failed |
//! | Cancelling | CancelConfirmed | Cancelled |
//! | Cancelling | Completed | Succeeded（キャンセルが間に合わなかった） |
//! | Cancelling | FailedRun | Failed（同上） |
//!
//! 終端（Cancelled・Succeeded・Failed）からの遷移と表に無い組は
//! [`InvalidTransition`]。
//!
//! # 未実装（実装済みを装わない）
//!
//! - `job.json` への永続化・クラッシュ検出・状態確認コマンドは #146
//!   （TASK-34.2）、やり直しの案内は #147（TASK-34.3）。
//! - キャンセル（`SIGKILL`）後に残る予約済み `out_dir`・tmp の扱いは #145
//!   （TASK-34.1-2）。本モジュールは公開場所に何も残らないことを主張しない。
//! - キャンセルの終了コード写像は TASK-33.x の承認事項。

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::error::TrainProcessError;
use crate::process::{
    CancelToken, CancelledRun, RunLimits, TrainRunEnd, WorkerLauncher, run_train_cancellable,
};
use crate::request::TrainRequest;
use crate::result::TrainOutcome;

/// 学習ジョブの状態。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    /// 起動前。
    Queued,
    /// 学習ワーカー実行中。
    Running,
    /// キャンセル要求を受け付けてから、kill・回収が終わるまでの窓。
    Cancelling,
    /// キャンセルで終わった（終端）。
    Cancelled,
    /// 正常に結果を得た（終端）。
    Succeeded,
    /// 失敗した（終端）。
    Failed,
}

impl JobState {
    /// 終端状態か。
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Cancelled | Self::Succeeded | Self::Failed)
    }
}

/// 状態遷移を起こす事象。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobEvent {
    /// 実行開始。
    Start,
    /// キャンセル要求の受付。
    CancelRequested,
    /// ワーカーが正常な結果を返した。
    Completed,
    /// ワーカーがエラー、または実行自体が失敗した。
    FailedRun,
    /// キャンセルで子を止めたことを確認した。
    CancelConfirmed,
}

/// 遷移表に無い `(状態, 事象)` の組。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidTransition {
    /// 遷移前の状態。
    pub from: JobState,
    /// 与えられた事象。
    pub event: JobEvent,
}

/// 遷移表（モジュール doc）に従う純関数。
///
/// # Errors
/// 表に無い組（終端からの遷移を含む）は [`InvalidTransition`]。
pub const fn transition(from: JobState, event: JobEvent) -> Result<JobState, InvalidTransition> {
    use JobEvent as E;
    use JobState as S;
    match (from, event) {
        (S::Queued, E::Start) => Ok(S::Running),
        (S::Queued, E::CancelRequested) => Ok(S::Cancelled),
        (S::Running, E::CancelRequested) => Ok(S::Cancelling),
        (S::Running | S::Cancelling, E::Completed) => Ok(S::Succeeded),
        (S::Running | S::Cancelling, E::FailedRun) => Ok(S::Failed),
        (S::Cancelling, E::CancelConfirmed) => Ok(S::Cancelled),
        _ => Err(InvalidTransition { from, event }),
    }
}

/// [`JobHandle::cancel`] の結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelOutcome {
    /// キャンセルを受け付けた。
    Requested,
    /// 既にキャンセル中。
    AlreadyCancelling,
    /// 既に終端で、何も送らない（終了後の kill を型で塞ぐ）。
    AlreadyFinished,
}

fn lock(state: &Mutex<JobState>) -> MutexGuard<'_, JobState> {
    // poison しても状態は `Copy` の値で壊れないため回復して使う
    // （ライブラリで panic しない）。
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// 実行中のジョブを外から操作・観測する口。`Clone + Send + Sync`。
#[derive(Debug, Clone)]
pub struct JobHandle {
    state: Arc<Mutex<JobState>>,
    token: CancelToken,
}

impl JobHandle {
    /// 現在の状態。
    #[must_use]
    pub fn state(&self) -> JobState {
        *lock(&self.state)
    }

    /// キャンセルを要求する。`Queued` は即 `Cancelled`、`Running` は
    /// `Cancelling` にしてトークンを立てる。
    pub fn cancel(&self) -> CancelOutcome {
        let mut guard = lock(&self.state);
        match *guard {
            JobState::Cancelling => CancelOutcome::AlreadyCancelling,
            s if s.is_terminal() => CancelOutcome::AlreadyFinished,
            s => {
                if let Ok(next) = transition(s, JobEvent::CancelRequested) {
                    *guard = next;
                }
                self.token.cancel();
                CancelOutcome::Requested
            }
        }
    }
}

/// 1 回の学習実行を状態遷移つきで行うジョブ。
#[derive(Debug)]
pub struct TrainJob {
    state: Arc<Mutex<JobState>>,
    token: CancelToken,
}

impl Default for TrainJob {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainJob {
    /// `Queued` のジョブを作る。
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(JobState::Queued)),
            token: CancelToken::new(),
        }
    }

    /// 別スレッドからキャンセル・観測するためのハンドル。
    #[must_use]
    pub fn handle(&self) -> JobHandle {
        JobHandle {
            state: Arc::clone(&self.state),
            token: self.token.clone(),
        }
    }

    fn apply(&self, event: JobEvent) {
        let mut guard = lock(&self.state);
        if let Ok(next) = transition(*guard, event) {
            *guard = next;
        }
    }

    /// 学習ワーカーを実行し、結果に応じて状態を遷移させる。子を待つ間は
    /// ロックを持たない。
    ///
    /// # Errors
    /// [`crate::process::run_train_cancellable`] のエラー（状態は `Failed`）。
    pub fn run(
        self,
        launcher: &WorkerLauncher,
        request: &TrainRequest,
        job_dir: &Path,
        limits: &RunLimits,
    ) -> Result<TrainRunEnd, TrainProcessError> {
        // `Queued` のままキャンセル済みなら `Start` は無効遷移になり、状態は
        // `Cancelled`（終端）のまま。この場合は入力検証（`InvalidRunLimits`）
        // より取り消しを優先し、子を起動せず `Cancelled` を返す。検証エラーを
        // 返すと、終端 `Cancelled` からの `FailedRun` は無効遷移として捨てられ、
        // 戻り値と観測状態が食い違うため。
        self.apply(JobEvent::Start);
        if self.token.is_cancelled() && *lock(&self.state) == JobState::Cancelled {
            return Ok(TrainRunEnd::Cancelled(CancelledRun::before_start()));
        }
        let result = run_train_cancellable(launcher, request, job_dir, limits, &self.token);
        let event = match &result {
            Ok(TrainRunEnd::Completed(run)) => match run.outcome() {
                TrainOutcome::Ok(_) => JobEvent::Completed,
                TrainOutcome::Error(_) => JobEvent::FailedRun,
            },
            Ok(TrainRunEnd::Cancelled(_)) => JobEvent::CancelConfirmed,
            Err(_) => JobEvent::FailedRun,
        };
        self.apply(event);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use JobEvent as E;
    use JobState as S;

    const ALL_STATES: [JobState; 6] = [
        S::Queued,
        S::Running,
        S::Cancelling,
        S::Cancelled,
        S::Succeeded,
        S::Failed,
    ];
    const ALL_EVENTS: [JobEvent; 5] = [
        E::Start,
        E::CancelRequested,
        E::Completed,
        E::FailedRun,
        E::CancelConfirmed,
    ];

    /// REQ-34: 遷移表の全セル（許可 8 件・他はすべて拒否）。
    #[test]
    fn req34_transition_table_all_cells() {
        let allowed = [
            (S::Queued, E::Start, S::Running),
            (S::Queued, E::CancelRequested, S::Cancelled),
            (S::Running, E::CancelRequested, S::Cancelling),
            (S::Running, E::Completed, S::Succeeded),
            (S::Running, E::FailedRun, S::Failed),
            (S::Cancelling, E::CancelConfirmed, S::Cancelled),
            (S::Cancelling, E::Completed, S::Succeeded),
            (S::Cancelling, E::FailedRun, S::Failed),
        ];
        for from in ALL_STATES {
            for event in ALL_EVENTS {
                let expected = allowed
                    .iter()
                    .find(|(f, e, _)| *f == from && *e == event)
                    .map(|(_, _, t)| *t);
                match expected {
                    Some(to) => assert_eq!(transition(from, event), Ok(to)),
                    None => assert_eq!(
                        transition(from, event),
                        Err(InvalidTransition { from, event })
                    ),
                }
            }
        }
    }

    /// REQ-34: 終端状態はどの事象でも遷移しない。
    #[test]
    fn req34_terminal_states_reject_all_events() {
        for s in [S::Cancelled, S::Succeeded, S::Failed] {
            assert!(s.is_terminal());
            for e in ALL_EVENTS {
                assert!(transition(s, e).is_err());
            }
        }
        assert!(!S::Cancelling.is_terminal());
    }

    /// REQ-34: `Queued` のキャンセルは即 `Cancelled`。
    #[test]
    fn req34_cancel_outcome_queued() {
        let job = TrainJob::new();
        let h = job.handle();
        assert_eq!(h.cancel(), CancelOutcome::Requested);
        assert_eq!(h.state(), S::Cancelled);
        assert_eq!(h.cancel(), CancelOutcome::AlreadyFinished);
    }

    /// REQ-34: `Running` のキャンセルは `Cancelling` とトークン、二重は
    /// `AlreadyCancelling`、終端は `AlreadyFinished`。
    #[test]
    fn req34_cancel_outcome_running_and_repeat() {
        let job = TrainJob::new();
        job.apply(E::Start);
        let h = job.handle();
        assert_eq!(h.state(), S::Running);
        assert!(!job.token.is_cancelled());
        assert_eq!(h.cancel(), CancelOutcome::Requested);
        assert_eq!(h.state(), S::Cancelling);
        assert!(job.token.is_cancelled());
        assert_eq!(h.cancel(), CancelOutcome::AlreadyCancelling);
        job.apply(E::CancelConfirmed);
        assert_eq!(h.cancel(), CancelOutcome::AlreadyFinished);
    }

    /// REQ-34: 終端後の `cancel()` はトークンを立てない。
    #[test]
    fn req34_cancel_after_success_does_not_set_token() {
        let job = TrainJob::new();
        job.apply(E::Start);
        job.apply(E::Completed);
        assert_eq!(job.handle().cancel(), CancelOutcome::AlreadyFinished);
        assert!(!job.token.is_cancelled());
    }

    /// REQ-34: トークンはクローン間で共有される。
    #[test]
    fn req34_cancel_token_shared_across_clones() {
        let a = CancelToken::new();
        let b = a.clone();
        assert!(!b.is_cancelled());
        a.cancel();
        assert!(b.is_cancelled());
    }

    /// REQ-34・レビュー指摘 P1: `Queued` のままキャンセルされたジョブは、
    /// 他の入力検証エラーがあっても `Cancelled` を返し、観測状態と一致する
    /// （戻り値と状態の食い違いを防ぐ）。
    #[test]
    fn req34_pre_start_cancel_wins_over_invalid_limits() {
        use crate::request::{Device, TrainRequestParams};
        use std::time::Duration;

        let make = |limit: u32| {
            TrainRequest::new(TrainRequestParams {
                kind: "c3".to_string(),
                kind_version: 1,
                config: serde_json::Map::new(),
                label_order: vec!["a".to_string(), "b".to_string()],
                max_bytes: 512,
                seed: 0,
                device: Device::Cpu,
                root: "/fandhe-edge-fixture-root".to_string(),
                train_path: "train.jsonl".to_string(),
                out_dir: "out".to_string(),
                time_limit_seconds: Some(limit),
                rss_limit_bytes: None,
            })
            .expect("valid request params")
        };
        let request = make(10);
        // 別リクエスト由来の緩い RunLimits（実行されていれば InvalidRunLimits）。
        let long = RunLimits::for_request(&make(3000));
        assert!(long.wall_timeout() > Duration::from_secs(10));

        let dir =
            std::env::temp_dir().join(format!("fandhe-edge-train-job-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create dir");
        let python = dir.join("python");
        let script = dir.join("launch.py");
        std::fs::write(&python, b"").expect("stub python");
        std::fs::write(&script, b"").expect("stub script");
        let launcher = WorkerLauncher::new(python, script).expect("launcher");

        let job = TrainJob::new();
        let handle = job.handle();
        assert_eq!(handle.cancel(), CancelOutcome::Requested);
        let result = job.run(&launcher, &request, &dir, &long);
        let _ = std::fs::remove_dir_all(&dir);
        assert!(matches!(result, Ok(TrainRunEnd::Cancelled(_))));
        assert_eq!(handle.state(), S::Cancelled);
    }

    /// REQ-34: `Start` の直後にキャンセルが届いた場合（状態 `Cancelling`）も、
    /// OS によらず戻り値は `Cancelled`、状態も `Cancelled` で一致する
    /// （非 unix の `run_train_cancellable` も起動前キャンセルを優先する）。
    #[test]
    fn req34_cancel_right_after_start_matches_state_on_every_os() {
        use crate::request::{Device, TrainRequestParams};
        let request = TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: "/fandhe-edge-fixture-root".to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: Some(10),
            rss_limit_bytes: None,
        })
        .expect("valid request params");
        let limits = RunLimits::for_request(&request);
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-train-job-after-start-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create dir");
        let python = dir.join("python");
        let script = dir.join("launch.py");
        std::fs::write(&python, b"").expect("stub python");
        std::fs::write(&script, b"").expect("stub script");
        let launcher = WorkerLauncher::new(python, script).expect("launcher");

        let job = TrainJob::new();
        let handle = job.handle();
        job.apply(JobEvent::Start);
        assert_eq!(handle.cancel(), CancelOutcome::Requested);
        assert_eq!(handle.state(), S::Cancelling);
        let result = job.run(&launcher, &request, &dir, &limits);
        let _ = std::fs::remove_dir_all(&dir);
        assert!(matches!(result, Ok(TrainRunEnd::Cancelled(_))));
        assert_eq!(handle.state(), S::Cancelled);
    }
}
