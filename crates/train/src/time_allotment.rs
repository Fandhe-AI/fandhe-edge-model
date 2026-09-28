//! 探索予算のうち候補 1 件へ配分する持ち時間の算出・記録（REQ-18・REQ-39。
//! TASK-18.1-1・issue #83）。
//!
//! # 呼び出し文脈
//!
//! 2026-09-27 のオーナー判断により、探索予算（既定 1 時間）は全候補の合計に
//! 対する予算とし、候補ごとの持ち時間も別途設ける。全候補の合計予算の管理・
//! 候補の比較・選定の記録は [`crate::search`]（issue #84・TASK-18.1-2）が
//! 本モジュールの [`allot`]・[`run_candidate`] を繰り返し呼ぶ形で実装済みで、
//! 本モジュールはそこから呼ばれる「残り予算・残り候補数から 1 候補分の
//! 持ち時間を決め、その候補を実行し、実行結果を記録する」部分だけを担う。
//!
//! 学習ワーカー（`trainer/`）を実際に子プロセスとして起動する処理は #178
//! （REQ-34。CLI・ジョブ管理からの子プロセス起動・タイムアウトでの kill・
//! 終了コード写像）の対象で、本モジュールには含まない。[`CandidateRunner`]
//! は #178 がその実行器を実装するための差し替え可能な接合点（trait）を
//! 用意するだけのスタブである。
//!
//! 「予算到達」を合格扱いしない判定（TASK-18.2）も本モジュールの対象外。
//! 本モジュールが返す [`CandidateTimeStatus`] はあくまで観測結果の記録で
//! あり、選定ロジックへどう反映するかは呼び出し元（TASK-18.2）が
//! 決める。
//!
//! # 持ち時間の決め方
//!
//! PoC-17 の `--per-candidate-seconds` と同じ考え方（残り予算を残り候補数で
//! 均等割りする）を [`PerCandidatePolicy::EvenSplit`] として実装する。
//! 早く終わった候補の余り時間は、次回の呼び出しで残り予算が増えている
//! ことで自然に後続候補へ回る（`allot` を毎回呼び直す設計。呼び出し元が
//! 状態を持ち回す必要はない）。[`PerCandidatePolicy::Fixed`] は固定値を
//! 指定する場合に使う。いずれも [`crate::limits::MAX_TRAIN_WALL_SECONDS`]
//! （1 学習ジョブあたりの壁時計上限。学習ワーカー側の絶対上限）を超えない。
//!
//! # 実機での確認手順（人間担当・未実施・証拠種別: 実機）
//!
//! 本 Issue の受け入れ条件はテストハーネス（`FakeClock`／`FakeRunner` 相当の
//! モック。`crates/train/tests/candidate_time_limit.rs`）で満たすが、実際の
//! 学習ワーカーに対する打ち切りの実効性は
//! Mac 実機（Apple Silicon）で別途確認する（PoC-17 相当）。Agent はこの
//! 手順を実行しない。
//!
//! 1. `device:"gpu"` で `time_limit_seconds` を小さく（例: 5）指定した学習
//!    リクエスト JSON を用意する
//! 2. `trainer/launch.py train` で学習ワーカーを起動する
//! 3. 標準出力が `{"status":"error","code":"limit_exceeded",...}`
//!    （終了コード 20）で終わること、起動から終了までの経過時間が
//!    「指定した持ち時間 + supervisor.py の猶予 5 秒」以内であることを
//!    確認する

use std::num::{NonZeroU32, NonZeroUsize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::error::TrainRequestError;
use crate::limits::MAX_TRAIN_WALL_SECONDS;
use crate::request::{TrainRequest, TrainRequestParams};
use crate::result::{FailureCode, TrainOutcome};

/// 候補 1 件あたりの持ち時間の決め方（呼び出し元が選ぶ方針）。
///
/// 将来のバリアント追加に備え `#[non_exhaustive]` にする（呼び出し元の
/// `match` は必ず `_` を用意する）。`Serialize` は探索記録
/// （[`crate::search::SearchRecord::per_candidate_policy`]。#84・
/// TASK-18.1-2）への往復検証用で、挙動には影響しない。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PerCandidatePolicy {
    /// 残り予算を残り候補数で均等割りする（既定）。PoC-17
    /// `--per-candidate-seconds` と同じ考え方。
    EvenSplit,
    /// 指定値を使う。[`MAX_TRAIN_WALL_SECONDS`] を超える指定は
    /// [`TimeAllotmentError::FixedExceedsJobLimit`] として拒否する
    /// （黙って丸めない。fail-closed）。
    Fixed(NonZeroU32),
}

/// [`allot`] の失敗（呼び出し元の入力誤り・時計の異常）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TimeAllotmentError {
    /// [`PerCandidatePolicy::Fixed`] の指定値が [`MAX_TRAIN_WALL_SECONDS`]
    /// を超える。
    FixedExceedsJobLimit { requested: u32 },
    /// 時計が使用不能（[`Clock::unix_millis`] が `UNIX_EPOCH` より前の
    /// 時刻を返した・`u64` の桁あふれ等）。外部入力ではなく実行環境由来の
    /// 異常のため、`panic` ではなく `Err` として呼び出し元に伝える
    /// （`.claude/rules/coding-rust.md`「ライブラリコードでは `Result` を
    /// 返し、panic させない」）。
    ClockUnavailable,
    /// `remaining_candidates`（`NonZeroUsize`）が `u64` へ変換できない
    /// （32bit 未満のターゲットを想定した防御的な分岐。本リポの検証環境
    /// 〔Apple Silicon・64bit〕では到達しない）。
    CandidateCountTooLarge,
}

impl std::fmt::Display for TimeAllotmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeAllotmentError::FixedExceedsJobLimit { requested } => write!(
                f,
                "fixed per-candidate time limit {requested} exceeds job limit {MAX_TRAIN_WALL_SECONDS}"
            ),
            TimeAllotmentError::ClockUnavailable => write!(f, "system clock is unavailable"),
            TimeAllotmentError::CandidateCountTooLarge => {
                write!(f, "remaining candidate count does not fit in u64")
            }
        }
    }
}

impl std::error::Error for TimeAllotmentError {}

/// 1..=[`MAX_TRAIN_WALL_SECONDS`] を型で保証する持ち時間（秒）。
///
/// コンストラクタは非公開にし、[`allot`] を通った値だけが構築できる
/// （壊れた値〔0 秒・上限超過〕を表現できない型にする。
/// `.claude/rules/coding-rust.md`「公開 API・型設計」）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllottedSeconds(u32);

impl AllottedSeconds {
    fn new(seconds: u32) -> Option<Self> {
        if (1..=MAX_TRAIN_WALL_SECONDS).contains(&seconds) {
            Some(Self(seconds))
        } else {
            None
        }
    }

    /// 決定した持ち時間（秒）。
    #[must_use]
    pub fn get(self) -> u32 {
        self.0
    }
}

/// [`allot`] の結果。算出値が 1 秒未満の場合は必ず [`Allotment::Exhausted`]
/// を返し、0 秒を [`AllottedSeconds`] として渡さない（学習ワーカー側
/// `contract.py` が `time_limit_seconds < 1` を拒否する契約と揃える）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Allotment {
    /// 候補へ配分できる持ち時間。
    Granted(AllottedSeconds),
    /// 残り予算が尽きており、これ以上候補を実行する時間がない。
    Exhausted,
}

/// 残り予算・残り候補数・方針から、候補 1 件分の持ち時間を算出する
/// （TASK-18.1-1）。
///
/// 計算はすべて整数演算（`u64`・`u32`）で行い、`f64`・`as` による切り捨ては
/// 使わない（`.claude/rules/coding-rust.md`「数値・決定性」）。桁あふれは
/// `checked_*` 演算で検出し、`remaining_budget_seconds` がどれだけ大きくても
/// （`u64::MAX` でも）panic しない。
///
/// # Errors
///
/// [`PerCandidatePolicy::Fixed`] の指定値が [`MAX_TRAIN_WALL_SECONDS`] を
/// 超える場合に [`TimeAllotmentError::FixedExceedsJobLimit`] を返す。
pub fn allot(
    remaining_budget_seconds: u64,
    remaining_candidates: NonZeroUsize,
    policy: PerCandidatePolicy,
) -> Result<Allotment, TimeAllotmentError> {
    let job_limit = u64::from(MAX_TRAIN_WALL_SECONDS);
    let seconds = match policy {
        PerCandidatePolicy::EvenSplit => {
            // `NonZeroUsize` → `u64`: `usize` から `u64` への `From` 実装は
            // 無い環境がある（32bit ターゲット等）ため `try_from` を使う。
            // 本リポの検証環境（Apple Silicon・64bit）では失敗しないが、
            // 外部入力に依存しない値でも `unwrap` は避ける方針に従う。
            let candidates = u64::try_from(remaining_candidates.get())
                .map_err(|_| TimeAllotmentError::CandidateCountTooLarge)?;
            let per_candidate = remaining_budget_seconds / candidates;
            per_candidate.min(job_limit)
        }
        PerCandidatePolicy::Fixed(fixed) => {
            let requested = fixed.get();
            let fixed_u64 = u64::from(requested);
            if fixed_u64 > job_limit {
                return Err(TimeAllotmentError::FixedExceedsJobLimit { requested });
            }
            fixed_u64.min(remaining_budget_seconds)
        }
    };
    let Ok(seconds) = u32::try_from(seconds) else {
        // `job_limit`（3600）を超えないことが上の `min` で保証されるため
        // 到達しないが、`as` による切り捨てを避けるため `try_from` で
        // 明示的に処理する（防御的な分岐）。
        return Ok(Allotment::Exhausted);
    };
    match AllottedSeconds::new(seconds) {
        Some(allotted) => Ok(Allotment::Granted(allotted)),
        None => Ok(Allotment::Exhausted),
    }
}

/// [`allot`] で決めた持ち時間を学習リクエストへ反映する（下げる方向のみ）。
///
/// `params.time_limit_seconds` が明示されている場合、その値の妥当性検査
/// （1..=[`MAX_TRAIN_WALL_SECONDS`]）は [`TrainRequest::new`] と同じ規則
/// （[`TrainRequestError::InvalidTimeLimitSeconds`]）で行い、`min` で
/// 範囲外の値を隠さない。検査を通った後、実際にリクエストへ渡す値は
/// 「利用者が指定した値」と「探索予算から配分された持ち時間」の小さい方
/// にする。
///
/// # Errors
///
/// `params.time_limit_seconds` が範囲外、または他のフィールドが
/// [`TrainRequest::new`] の検証を満たさない場合に
/// [`TrainRequestError`] を返す。
pub fn build_request_with_allotment(
    mut params: TrainRequestParams,
    allotted: AllottedSeconds,
) -> Result<TrainRequest, TrainRequestError> {
    if let Some(requested) = params.time_limit_seconds
        && !(1..=MAX_TRAIN_WALL_SECONDS).contains(&requested)
    {
        return Err(TrainRequestError::InvalidTimeLimitSeconds);
    }
    let requested = params.time_limit_seconds.unwrap_or(MAX_TRAIN_WALL_SECONDS);
    params.time_limit_seconds = Some(requested.min(allotted.get()));
    TrainRequest::new(params)
}

/// 経過時間・記録用の壁時計時刻を取得する（テストで差し替え可能にする
/// ための境界。[`run_candidate`] は経過時間の計測に
/// [`Clock::monotonic`]、記録に [`Clock::unix_millis`] を使い分ける）。
pub trait Clock {
    /// 起点からの単調増加な経過時間。打ち切り判定・`elapsed_ms` の算出に
    /// 使う（システム時刻の変更〔NTP 補正・夏時間〕の影響を受けない）。
    fn monotonic(&self) -> Duration;
    /// 記録用の壁時計時刻（UNIX エポックからのミリ秒）。人が読める時刻を
    /// 記録に残すために使い、経過時間の計測には使わない。
    fn unix_millis(&self) -> Result<u64, TimeAllotmentError>;
}

/// 実時計（[`Instant`]・[`SystemTime`]）を使う [`Clock`] の実装。
pub struct SystemClock {
    start: Instant,
}

impl SystemClock {
    /// 現在時刻を起点として計測を開始する。
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn monotonic(&self) -> Duration {
        self.start.elapsed()
    }

    fn unix_millis(&self) -> Result<u64, TimeAllotmentError> {
        let since_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| TimeAllotmentError::ClockUnavailable)?;
        u64::try_from(since_epoch.as_millis()).map_err(|_| TimeAllotmentError::ClockUnavailable)
    }
}

/// 学習ワーカーを候補 1 件分だけ実行する実行器（#178 が子プロセス起動を
/// 実装する接合点）。
///
/// 本 crate（issue #83・TASK-18.1-1）にはこの trait の実装を含めない。
/// `crates/train` の呼び出し元（CLI・ジョブ管理。#178）が子プロセス起動・
/// タイムアウトでの kill・終了コード写像を実装した上で、この trait を
/// 実装する想定（REQ-34・REQ-39）。
pub trait CandidateRunner {
    /// 実行器固有のエラー型（子プロセスの起動失敗等）。
    type Error;
    /// `request` を学習ワーカーへ渡して実行し、結果を返す。
    fn run(&mut self, request: &TrainRequest) -> Result<TrainOutcome, Self::Error>;
}

/// 候補 1 件の打ち切り分類（TASK-18.2 の「予算到達を合格扱いしない」判定は
/// 本モジュールの対象外。ここでは観測結果の記録に留める）。
///
/// `code == "limit_exceeded"` は学習ワーカー側で時間以外の資源上限
/// （RSS・ステップ数・トークン数・モデルサイズ・`RLIMIT_CPU` 等）にも
/// 使われる共通コードであり、本モジュールはワーカーから打ち切り原因を
/// 型付きで受け取っていない（実行器 [`CandidateRunner`] は #178 が実装する
/// スタブで、原因の受け渡しは未実装）。そのため `code == "limit_exceeded"`
/// の打ち切りは [`LimitExceeded`](CandidateTimeStatus::LimitExceeded) 1 種
/// にまとめ、単調時計で測った経過時間が持ち時間以上だったかを観測値として
/// 添えるに留め、原因を断定しない（P1 指摘。#178 が原因を型付きで供給する
/// ようになった時点で分類を分けられるようにする。REQ-18・REQ-39）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CandidateTimeStatus {
    /// 学習ワーカーが成功した。
    Completed,
    /// `code == "limit_exceeded"` による打ち切り。原因（持ち時間・RSS 等の
    /// 他の資源上限）はワーカー出力から特定できないため断定しない。
    LimitExceeded {
        /// 単調時計で測った経過時間が、配分した持ち時間以上だったかの
        /// 観測値（原因の断定ではない。終了処理を含む `runner.run` の
        /// 所要時間には時間以外の上限超過後の終了猶予も含まれうるため、
        /// `true` は「時間超過の可能性が高い」ことを示すに留まる）。
        elapsed_reached_time_limit: bool,
    },
    /// 持ち時間超過以外のワーカー失敗。
    Failed {
        /// ワーカーが返した失敗コード。
        code: FailureCode,
    },
}

/// 候補 1 件の実行記録（TASK-18.1-1）。
///
/// フィールドは非公開にし、読み取り専用アクセサのみを公開する
/// （`crates/train` の他の型〔`TrainRequest`・`ArtifactRecord`〕と同じ
/// 設計。別 crate が未検証の値から記録を直接組み立てられないようにする）。
/// `Serialize` はテスト・将来の記録永続化（#84）向けの往復検証用。
/// `status` は [`CandidateTimeStatus`] のタグをレコード直下へ展開する
/// （`#[serde(flatten)]`）: `serde_json::to_value(record)["status"]` が
/// `"limit_exceeded"` のようなタグ文字列そのものになる。
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct CandidateTimeRecord {
    /// 実際にリクエストへ渡した持ち時間（秒）。
    time_limit_seconds: u32,
    /// 開始時刻（UNIX ミリ秒）。
    started_at_unix_ms: u64,
    /// 締切り時刻（`started_at_unix_ms + time_limit_seconds * 1000`）。
    deadline_at_unix_ms: u64,
    /// 終了を観測した時刻（UNIX ミリ秒）。
    ended_at_unix_ms: u64,
    /// 単調時計で測った経過時間（ミリ秒）。
    elapsed_ms: u64,
    /// 打ち切り分類。
    #[serde(flatten)]
    status: CandidateTimeStatus,
}

impl CandidateTimeRecord {
    /// 実際にリクエストへ渡した持ち時間（秒）。
    #[must_use]
    pub fn time_limit_seconds(&self) -> u32 {
        self.time_limit_seconds
    }
    /// 開始時刻（UNIX ミリ秒）。
    #[must_use]
    pub fn started_at_unix_ms(&self) -> u64 {
        self.started_at_unix_ms
    }
    /// 締切り時刻（UNIX ミリ秒）。
    #[must_use]
    pub fn deadline_at_unix_ms(&self) -> u64 {
        self.deadline_at_unix_ms
    }
    /// 終了を観測した時刻（UNIX ミリ秒）。
    #[must_use]
    pub fn ended_at_unix_ms(&self) -> u64 {
        self.ended_at_unix_ms
    }
    /// 単調時計で測った経過時間（ミリ秒）。
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed_ms
    }
    /// 打ち切り分類。
    #[must_use]
    pub fn status(&self) -> CandidateTimeStatus {
        self.status
    }
}

/// [`run_candidate`] の結果（記録と、選定〔#84〕に使う実行結果の組）。
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRun {
    record: CandidateTimeRecord,
    outcome: TrainOutcome,
}

impl CandidateRun {
    /// 候補 1 件の実行記録。
    #[must_use]
    pub fn record(&self) -> &CandidateTimeRecord {
        &self.record
    }
    /// 学習ワーカーの実行結果（#84 が候補の選定に使う）。
    #[must_use]
    pub fn outcome(&self) -> &TrainOutcome {
        &self.outcome
    }
}

/// [`run_candidate`] のエラー。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CandidateTimeError<E> {
    /// リクエストの組み立て（[`build_request_with_allotment`]）が失敗した。
    Request(TrainRequestError),
    /// 時計の異常。
    Clock(TimeAllotmentError),
    /// 実行器（[`CandidateRunner`]）が失敗した。
    Runner(E),
}

impl<E: std::fmt::Display> std::fmt::Display for CandidateTimeError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CandidateTimeError::Request(e) => write!(f, "failed to build train request: {e}"),
            CandidateTimeError::Clock(e) => write!(f, "clock error: {e}"),
            CandidateTimeError::Runner(e) => write!(f, "candidate runner failed: {e}"),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for CandidateTimeError<E> {}

/// 候補 1 件を実行し、持ち時間・経過時間・打ち切り分類を記録する
/// （TASK-18.1-1）。
///
/// 手順: (1) [`build_request_with_allotment`] でリクエストを作る →
/// (2) 開始時刻（単調時計・壁時計）を取る → (3) `runner.run(&request)` を
/// 呼ぶ → (4) 終了時刻を取る → (5) 結果を分類して記録する。
///
/// 経過時間は単調時計（[`Clock::monotonic`]）で測り、記録に残す時刻は
/// 壁時計（[`Clock::unix_millis`]）で取る（役割が異なる: 単調時計は
/// システム時刻の変更の影響を受けない打ち切り判定に、壁時計は人が読める
/// 記録に使う）。
///
/// ワーカーが返す `message` 文字列は解析しない（打ち切り分類には
/// [`crate::result::WorkerFailure::failure_code`] と Rust 側で測った
/// 経過時間だけを使う。ワーカーがメッセージを偽っても分類は変わらない）。
///
/// # Errors
///
/// リクエストの組み立て・時計の取得・実行器のいずれかが失敗した場合に
/// [`CandidateTimeError`] を返す（この場合、記録は作らない）。
pub fn run_candidate<R, C>(
    runner: &mut R,
    clock: &C,
    params: TrainRequestParams,
    allotted: AllottedSeconds,
) -> Result<CandidateRun, CandidateTimeError<R::Error>>
where
    R: CandidateRunner,
    C: Clock,
{
    let request =
        build_request_with_allotment(params, allotted).map_err(CandidateTimeError::Request)?;
    let time_limit_seconds = request.time_limit_seconds();

    let started_mono = clock.monotonic();
    let started_at_unix_ms = clock.unix_millis().map_err(CandidateTimeError::Clock)?;
    let deadline_at_unix_ms = started_at_unix_ms
        .checked_add(u64::from(time_limit_seconds).saturating_mul(1000))
        .ok_or(CandidateTimeError::Clock(
            TimeAllotmentError::ClockUnavailable,
        ))?;

    let outcome = runner.run(&request).map_err(CandidateTimeError::Runner)?;

    let ended_mono = clock.monotonic();
    let ended_at_unix_ms = clock.unix_millis().map_err(CandidateTimeError::Clock)?;
    // 単調時計は定義上非減少だが、`Clock` は差し替え可能なテスト用実装
    // （`FakeClock`）を許すため、逆転しても panic しない
    // （`.claude/rules/coding-rust.md`「ライブラリコードでは `Result` を
    // 返し、panic させない」）。
    let elapsed = ended_mono
        .checked_sub(started_mono)
        .ok_or(CandidateTimeError::Clock(
            TimeAllotmentError::ClockUnavailable,
        ))?;
    let elapsed_ms = u64::try_from(elapsed.as_millis())
        .map_err(|_| CandidateTimeError::Clock(TimeAllotmentError::ClockUnavailable))?;

    let status = classify(&outcome, elapsed_ms, time_limit_seconds);

    Ok(CandidateRun {
        record: CandidateTimeRecord {
            time_limit_seconds,
            started_at_unix_ms,
            deadline_at_unix_ms,
            ended_at_unix_ms,
            elapsed_ms,
            status,
        },
        outcome,
    })
}

/// [`run_candidate`] の分類規則（[`CandidateTimeStatus`] のドキュメント
/// 参照）。`code == "limit_exceeded"` の打ち切り原因（持ち時間か、RSS 等の
/// 他の資源上限か）はワーカー出力から特定できないため断定せず、経過時間が
/// 持ち時間以上だったかの観測値だけを添える。
fn classify(
    outcome: &TrainOutcome,
    elapsed_ms: u64,
    time_limit_seconds: u32,
) -> CandidateTimeStatus {
    let TrainOutcome::Error(failure) = outcome else {
        return CandidateTimeStatus::Completed;
    };
    if failure.failure_code() != FailureCode::LimitExceeded {
        return CandidateTimeStatus::Failed {
            code: failure.failure_code(),
        };
    }
    let time_limit_ms = u64::from(time_limit_seconds).saturating_mul(1000);
    CandidateTimeStatus::LimitExceeded {
        elapsed_reached_time_limit: elapsed_ms >= time_limit_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::Device;

    fn candidates(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("test candidate count must be non-zero")
    }

    /// REQ-18・TASK-18.1-1: `EvenSplit` は残り予算 / 残り候補数の商（整数
    /// 除算）を返す。
    #[test]
    fn task18_1_1_even_split_divides_evenly() {
        let allotment =
            allot(3600, candidates(9), PerCandidatePolicy::EvenSplit).expect("valid allotment");
        assert_eq!(
            allotment,
            Allotment::Granted(AllottedSeconds::new(400).expect("valid"))
        );
    }

    /// REQ-18・TASK-18.1-1: `EvenSplit` は `MAX_TRAIN_WALL_SECONDS` で
    /// 頭打ちになる。
    #[test]
    fn task18_1_1_even_split_caps_at_job_limit() {
        let allotment =
            allot(100_000, candidates(2), PerCandidatePolicy::EvenSplit).expect("valid allotment");
        assert_eq!(
            allotment,
            Allotment::Granted(AllottedSeconds::new(MAX_TRAIN_WALL_SECONDS).expect("valid"))
        );
    }

    /// REQ-18・TASK-18.1-1: 商が 1 秒未満なら `Exhausted`。
    #[test]
    fn task18_1_1_even_split_exhausted_when_quotient_is_zero() {
        let allotment =
            allot(5, candidates(10), PerCandidatePolicy::EvenSplit).expect("valid allotment");
        assert_eq!(allotment, Allotment::Exhausted);
    }

    /// REQ-18・TASK-18.1-1: 残り予算 0 秒は必ず `Exhausted`（0 秒を
    /// `AllottedSeconds` として渡さない）。
    #[test]
    fn task18_1_1_even_split_exhausted_when_budget_is_zero() {
        let allotment =
            allot(0, candidates(1), PerCandidatePolicy::EvenSplit).expect("valid allotment");
        assert_eq!(allotment, Allotment::Exhausted);
    }

    /// REQ-18・TASK-18.1-1: 残り予算が `u64::MAX` でも桁あふれせず、
    /// `MAX_TRAIN_WALL_SECONDS` で頭打ちになる。
    #[test]
    fn task18_1_1_even_split_does_not_overflow_on_max_budget() {
        let allotment =
            allot(u64::MAX, candidates(1), PerCandidatePolicy::EvenSplit).expect("valid allotment");
        assert_eq!(
            allotment,
            Allotment::Granted(AllottedSeconds::new(MAX_TRAIN_WALL_SECONDS).expect("valid"))
        );
    }

    /// REQ-18・TASK-18.1-1: `Fixed` は指定値をそのまま使う（予算に余裕が
    /// ある場合）。
    #[test]
    fn task18_1_1_fixed_uses_requested_value_when_budget_allows() {
        let fixed = NonZeroU32::new(60).expect("non-zero");
        let allotment =
            allot(3600, candidates(1), PerCandidatePolicy::Fixed(fixed)).expect("valid allotment");
        assert_eq!(
            allotment,
            Allotment::Granted(AllottedSeconds::new(60).expect("valid"))
        );
    }

    /// REQ-18・TASK-18.1-1: `Fixed` は残り予算より大きい場合、残り予算まで
    /// 下げる。
    #[test]
    fn task18_1_1_fixed_is_capped_by_remaining_budget() {
        let fixed = NonZeroU32::new(60).expect("non-zero");
        let allotment =
            allot(30, candidates(1), PerCandidatePolicy::Fixed(fixed)).expect("valid allotment");
        assert_eq!(
            allotment,
            Allotment::Granted(AllottedSeconds::new(30).expect("valid"))
        );
    }

    /// REQ-18・TASK-18.1-1: `Fixed(MAX_TRAIN_WALL_SECONDS)` はそのまま受理する。
    #[test]
    fn task18_1_1_fixed_at_job_limit_is_accepted() {
        let fixed = NonZeroU32::new(MAX_TRAIN_WALL_SECONDS).expect("non-zero");
        let allotment = allot(
            MAX_TRAIN_WALL_SECONDS.into(),
            candidates(1),
            PerCandidatePolicy::Fixed(fixed),
        )
        .expect("valid allotment");
        assert_eq!(
            allotment,
            Allotment::Granted(AllottedSeconds::new(MAX_TRAIN_WALL_SECONDS).expect("valid"))
        );
    }

    /// REQ-18・TASK-18.1-1: `Fixed` が `MAX_TRAIN_WALL_SECONDS` を 1 秒でも
    /// 超えると拒否する（黙って丸めない。fail-closed）。
    #[test]
    fn task18_1_1_fixed_exceeding_job_limit_is_rejected() {
        let fixed = NonZeroU32::new(MAX_TRAIN_WALL_SECONDS + 1).expect("non-zero");
        let err = allot(u64::MAX, candidates(1), PerCandidatePolicy::Fixed(fixed)).unwrap_err();
        assert_eq!(
            err,
            TimeAllotmentError::FixedExceedsJobLimit {
                requested: MAX_TRAIN_WALL_SECONDS + 1
            }
        );
    }

    fn valid_params() -> TrainRequestParams {
        TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: "/tmp/fandhe-edge-time-allotment-test".to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: None,
            rss_limit_bytes: None,
        }
    }

    /// REQ-18・TASK-18.1-1: `time_limit_seconds` 省略時は配分値がそのまま
    /// リクエストへ渡る（`to_json_vec` にキーが含まれる。`MAX_TRAIN_WALL_SECONDS`
    /// と異なる値のため既定値扱いで省略されない）。
    #[test]
    fn task18_1_1_build_request_uses_allotment_when_omitted() {
        let allotted = AllottedSeconds::new(120).expect("valid");
        let request =
            build_request_with_allotment(valid_params(), allotted).expect("valid request");
        assert_eq!(request.time_limit_seconds(), 120);
        let bytes = request.to_json_vec().expect("serialize");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("valid json");
        assert_eq!(
            value.get("time_limit_seconds"),
            Some(&serde_json::json!(120))
        );
    }

    /// REQ-18・TASK-18.1-1: 利用者指定値が配分値より小さければ、指定値を
    /// そのまま使う（下げる方向のみ。上げない）。
    #[test]
    fn task18_1_1_build_request_keeps_smaller_explicit_value() {
        let mut params = valid_params();
        params.time_limit_seconds = Some(30);
        let allotted = AllottedSeconds::new(120).expect("valid");
        let request = build_request_with_allotment(params, allotted).expect("valid request");
        assert_eq!(request.time_limit_seconds(), 30);
    }

    /// REQ-18・TASK-18.1-1: 利用者指定値が配分値より大きければ、配分値まで
    /// 下げる。
    #[test]
    fn task18_1_1_build_request_lowers_larger_explicit_value() {
        let mut params = valid_params();
        params.time_limit_seconds = Some(600);
        let allotted = AllottedSeconds::new(120).expect("valid");
        let request = build_request_with_allotment(params, allotted).expect("valid request");
        assert_eq!(request.time_limit_seconds(), 120);
    }

    /// REQ-18・TASK-18.1-1: 利用者指定値が範囲外（0・上限超過）なら、
    /// `min` で隠さず `InvalidTimeLimitSeconds` を返す。
    #[test]
    fn task18_1_1_build_request_rejects_out_of_range_explicit_value() {
        let allotted = AllottedSeconds::new(120).expect("valid");
        for bad in [0u32, MAX_TRAIN_WALL_SECONDS + 1] {
            let mut params = valid_params();
            params.time_limit_seconds = Some(bad);
            let err = build_request_with_allotment(params, allotted).unwrap_err();
            assert_eq!(
                err,
                TrainRequestError::InvalidTimeLimitSeconds,
                "case: {bad}"
            );
        }
    }

    /// REQ-18・TASK-18.1-1: 配分値が `MAX_TRAIN_WALL_SECONDS`（既定値）で
    /// 利用者指定が無い場合は、`to_json_vec` でキーが省かれる既存挙動の
    /// まま。
    #[test]
    fn task18_1_1_build_request_omits_key_when_allotment_equals_default() {
        let allotted = AllottedSeconds::new(MAX_TRAIN_WALL_SECONDS).expect("valid");
        let request =
            build_request_with_allotment(valid_params(), allotted).expect("valid request");
        assert_eq!(request.time_limit_seconds(), MAX_TRAIN_WALL_SECONDS);
        let bytes = request.to_json_vec().expect("serialize");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("valid json");
        assert!(value.get("time_limit_seconds").is_none());
    }
}
