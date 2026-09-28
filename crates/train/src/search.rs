//! 探索予算全体の管理・複数候補の比較・選定の記録（REQ-18・REQ-39。
//! TASK-18.1-2・issue #84）。
//!
//! # 呼び出し文脈
//!
//! [`crate::time_allotment`]（issue #83・TASK-18.1-1）が「残り予算・残り
//! 候補数から候補 1 件分の持ち時間を決め、その候補を実行し、実行結果を
//! 記録する」部分を担うのに対し、本モジュールはそれを候補の宣言順に
//! 繰り返し呼び、探索予算全体の消費を追跡し、validation 正解率が最も
//! 高い候補を選ぶ上位ロジックを実装する。学習ワーカー（`trainer/`）を
//! 子プロセスとして実際に起動する処理（[`crate::time_allotment::CandidateRunner`]
//! の実装）は #178（REQ-34）の対象で、本モジュールには含まない。
//!
//! validation の推論も同様にスタブ（[`ValidationScorer`]）とする。Rust の
//! 推論ランタイムはまだ存在せず（REQ-28/30〜32 は未着手）、本 crate は
//! 学習ワーカー層に位置するため推論経路の型を持ち込まない
//! （`.claude/rules/coding-rust.md`「推論ランタイムは学習側に依存しない」の
//! 逆方向。学習ワーカー層が推論ランタイムに依存するのも層の境界違反）。
//! [`ValidationScorer`] の実装は推論ランタイム（REQ-28 系）またはジョブ管理
//! （#178）が担う。
//!
//! 正解率の算出は評価器 [`fandhe_edge_eval::metrics::evaluate_single_select`]
//! に委譲し、本 crate では再実装しない（`.claude/rules/coding-rust.md`
//! 「評価器は TASK-24.1 の 1 つだけに集約し、他の層で評価ロジックを再実装
//! しない」）。
//!
//! # 評価契約との関係（REQ-27）
//!
//! - [`SearchInput::validation_gold`] は **validation 分割のみ**を渡す想定。
//!   凍結した最終 test を渡してはならない（最終 test の適用は 1 回限りで、
//!   候補・しきい値の選び直しに使わない。TASK-27.3 で強制の仕組みを実装
//!   予定だが、本モジュールは呼び出し元の責務として doc で明示するに留める）
//! - [`ValidationScorer::predict_validation`] へは `candidate_id` と学習
//!   成果物（[`crate::result::SuccessOutcome`]）だけを渡し、
//!   `validation_gold`（正解ラベル）は渡さない（REQ-27「推論関数には
//!   `input` だけを渡す」の学習ワーカー層での対応。gold を渡さない制約は
//!   trait の引数リストという型のレベルで保証される）
//!
//! # PoC-17 との差異
//!
//! PoC-17 の `baseline_rule`（`02-poc-plan.md` PoC-17）は同率のとき
//! `package_bytes`（配布サイズ）が小さい方を選ぶが、[`crate::result::ArtifactRecord`]
//! は容量を持たないため本 Issue では採用しない。同率は「宣言順で先の候補」
//! （[`select_best`] の規則名 `"validation_accuracy_desc_then_candidate_order"`）
//! で解消し、同率だった候補 ID の一覧（[`SelectionDecision::Selected::tied_candidate_ids`]）
//! を記録に残す。
//!
//! # スコープ外（他 Issue が対象）
//!
//! - McNemar・Holm による有意性判定と、その選定記録への埋め込み（TASK-18.3-1・
//!   issue #87）
//! - 「予算到達」を合格扱いにしない判定・記録上のラベル付け（TASK-18.2・
//!   issue #85）
//! - [`ValidationScorer`]・[`crate::time_allotment::CandidateRunner`] の実装
//!   （推論ランタイム REQ-28 系、子プロセス起動 #178）
//! - 記録のファイルへの永続化・CLI `select` 工程の JSON 出力・終了コードへの
//!   写像（TASK-33.x）
//! - `package_bytes`・速度（p95）を使う PoC-17 の threshold／pareto 選定
//! - 自動選択と固定規則で結果が分かれた場合の最終 test 適用（TASK-18.4。
//!   見送り確定）
//!
//! # 実機での確認手順（人間担当・未実施・証拠種別: 実機）
//!
//! 本 Issue の受け入れ条件はテストハーネス（`crates/train/tests/search_budget_selection.rs`）
//! で満たすが、実際の学習ワーカーに対する探索の実効性は Mac 実機
//! （Apple Silicon）で別途確認する（PoC-17 相当）。Agent はこの手順を実行
//! しない。
//!
//! 1. 既定候補 C1・C3 の構成違いを用意し、`device:"gpu"` で探索予算（既定
//!    3600 秒）内に収まることを確認する
//! 2. 各候補の validation 正解率と、選定された候補・その正解率が記録に
//!    残ることを確認する
//! 3. 探索予算に対して候補の合計所要時間が超過する構成を用意し、未着手の
//!    候補が `not_started/budget_exhausted` として記録されることを確認する

use std::collections::BTreeSet;
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::Duration;

use fandhe_edge_eval::metrics::{self, EvalError, EvalRecord, Outcome, Ratio};
use fandhe_edge_eval::significance::MAX_EVAL_RECORDS;

use crate::error::TrainRequestError;
use crate::request::{TrainRequest, TrainRequestParams};
use crate::result::{SuccessOutcome, TrainOutcome};
use crate::time_allotment::{
    Allotment, CandidateRunner, CandidateTimeError, CandidateTimeRecord, Clock, PerCandidatePolicy,
    allot, run_candidate,
};

/// 探索予算の既定値（秒）。全候補の合計に対する予算（2026-09-27 オーナー
/// 判断）。REQ-18 の目安「1 時間」に対応する暫定値。
pub const DEFAULT_SEARCH_BUDGET_SECONDS: u64 = 3600;

/// 1 回の探索で許容する候補数の上限（REQ-39 資源上限）。PoC-17 は 9 候補
/// だったことを踏まえた暫定値。
pub const MAX_SEARCH_CANDIDATES: usize = 256;

/// 候補 ID の最大バイト数（REQ-39 資源上限）。`crates/eval` の
/// `selection_significance` モジュール（issue #87・TASK-18.3-1）と同じ値を
/// 使い、両モジュールで候補 ID の検証規則を揃える。
pub const MAX_CANDIDATE_ID_BYTES: usize = 128;

/// 候補数 × validation 件数の積の上限（REQ-39 資源上限）。保持する
/// [`Outcome`] の総数（各候補の validation 予測をすべて保持するため）を
/// 事前に抑える。
pub const MAX_SEARCH_OUTCOME_CELLS: u64 = 10_000_000;

/// 探索予算全体（秒）。0 秒は表現できない（`allot` が 0 秒を
/// [`Allotment::Exhausted`] として扱う契約と揃える）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchBudget(NonZeroU64);

impl SearchBudget {
    /// 秒数を指定して構築する。`seconds == 0` は `None`。
    #[must_use]
    pub fn new(seconds: u64) -> Option<Self> {
        NonZeroU64::new(seconds).map(Self)
    }

    /// 予算（秒）。
    #[must_use]
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl Default for SearchBudget {
    fn default() -> Self {
        // `DEFAULT_SEARCH_BUDGET_SECONDS`（3600）は非ゼロの定数であり、
        // ここでの `NonZeroU64::new` は必ず `Some` を返す。定数自体が
        // 変更されても 0 になることはない値のため `expect` で失敗を
        // 表明する（コンパイル時定数に対する防御であり、外部入力の経路
        // ではない）。
        Self(
            NonZeroU64::new(DEFAULT_SEARCH_BUDGET_SECONDS)
                .expect("DEFAULT_SEARCH_BUDGET_SECONDS must be non-zero"),
        )
    }
}

/// 探索対象の候補 1 件（候補 ID と学習リクエストの構成要素）。
#[derive(Debug, Clone)]
pub struct SearchCandidate {
    /// 探索・選定の記録で使う候補 ID（学習ワーカーの `kind` とは別の概念。
    /// 同じ `kind` でも構成違いの候補を区別するために使う）。
    pub candidate_id: String,
    /// 学習リクエストの構成要素（`label_order` は [`SearchInput::label_order`]
    /// と一致していなければならない）。
    pub params: TrainRequestParams,
}

/// [`run_search`] への入力一式。
pub struct SearchInput<'a> {
    /// ラベル集合（宣言順）。各候補の `params.label_order` と一致すること。
    pub label_order: &'a [&'a str],
    /// validation 分割の正解ラベル（**validation のみ**。凍結した最終 test
    /// を渡さない。モジュール doc「評価契約との関係」参照）。
    pub validation_gold: &'a [&'a str],
    /// 探索対象の候補（宣言順に実行する。乱数は使わない）。
    pub candidates: Vec<SearchCandidate>,
    /// 探索予算全体（秒）。
    pub budget: SearchBudget,
    /// 候補 1 件あたりの持ち時間の決め方（[`crate::time_allotment::allot`]
    /// へそのまま渡す）。
    pub policy: PerCandidatePolicy,
}

/// 学習済み候補で validation 入力を推論する接合点（trait）。
///
/// 本 crate にはこの trait の実装を含めない（推論ランタイム・ジョブ管理が
/// 実装する想定のスタブ。モジュール doc 参照）。
///
/// `candidate_id` と学習成果物だけを受け取り、`validation_gold`
/// （正解ラベル）は受け取らない（REQ-27。gold を渡さない制約は引数リスト
/// という型のレベルで保証される）。validation 入力そのものは実装側が
/// 自分で保持する。
///
/// # 時間上限（REQ-39・P0 指摘対応）
///
/// `time_limit` は [`run_search`] がこの呼び出し時点で残っている探索予算
/// （探索予算全体 − ここまでの経過時間）を渡す。本 trait は同期呼び出しの
/// ため `run_search` 側からこの呼び出し自体を打ち切ることはできない
/// （[`crate::time_allotment::CandidateRunner`] のような子プロセス経由の
/// 実行器ではなく、プロセス内の関数呼び出しであるため）。実装側
/// （推論ランタイム・ジョブ管理。#178 等）が `time_limit` を守る責務を持つ。
/// `run_search` は呼び出し前後の経過時間を計測し、`time_limit` を守れずに
/// 探索予算全体を超過したことを事後検出した場合、その候補を選定対象から
/// 除外し（[`CandidateSearchResult::ScoringExceededBudget`]）、以降の候補は
/// 未着手として記録する（fail-closed。「呼び出し中の時間制限がないため
/// 超過後も選定されてしまう」ことを防ぐ）。
pub trait ValidationScorer {
    /// 実装固有のエラー型。
    type Error;
    /// `candidate_id` の学習成果物で validation 入力を推論し、
    /// [`SearchInput::validation_gold`] と同じ順・同じ件数の [`Outcome`] 列を
    /// 返す。`time_limit` はこの呼び出し時点で残っている探索予算全体
    /// （trait doc「時間上限」参照）。
    fn predict_validation(
        &mut self,
        candidate_id: &str,
        artifact: &SuccessOutcome,
        time_limit: Duration,
    ) -> Result<Vec<Outcome>, Self::Error>;
}

/// validation 正解率（[`Ratio`] の往復検証用の直列化可能な写像）。
///
/// `Ratio` は往復検証用の `Serialize` を持たないため、選定記録
/// （[`SearchRecord`]）に載せるための専用の型を用意する。
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct ValidationAccuracy {
    /// 正解件数。
    pub correct: u64,
    /// 評価件数（validation gold の件数）。
    pub total: u64,
    /// `correct as f64 / total as f64`。
    pub value: f64,
}

impl From<Ratio> for ValidationAccuracy {
    fn from(ratio: Ratio) -> Self {
        Self {
            correct: ratio.numerator(),
            total: ratio.denominator(),
            value: ratio.value(),
        }
    }
}

/// 候補 1 件の未着手理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum NotStartedReason {
    /// 探索予算全体が尽きた（これ以降の候補も実行しない）。
    BudgetExhausted,
}

/// 候補 1 件の探索結果の分類。
///
/// `validation_outcomes`（[`Outcome`] の列）は JSON へ出さない
/// （評価データ・予測の本文を記録に残さない。security.md「データ本文を
/// ログ・エラーメッセージへ転記しない」）。評価済み候補の予測列は
/// [`CandidateSearchEntry::validation_outcomes`] のアクセサでのみ公開し、
/// issue #87（TASK-18.3-1・McNemar）が再推論せずに使えるようにする。
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "result", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CandidateSearchResult {
    /// 学習・validation 推論・正解率算出まで完了した。
    Evaluated {
        /// validation 正解率。
        validation_accuracy: ValidationAccuracy,
    },
    /// 学習ワーカーが成功しなかった（[`CandidateTimeRecord::status`] に詳細）。
    /// validation 推論は行っていない。
    TrainingNotCompleted,
    /// 学習は成功したが、[`ValidationScorer::predict_validation`] が失敗した。
    /// エラー内容は記録しない（security.md）。
    ScoringFailed,
    /// 学習・validation 推論・正解率算出まで完了したが、採点
    /// （[`ValidationScorer::predict_validation`]）の呼び出しに時間がかかり
    /// 探索予算全体を使い切った（P0 指摘対応。[`ValidationScorer`] trait doc
    /// 「時間上限」参照）。正解率は算出できているが、探索予算を超過した後の
    /// 結果を選定に使うと「合格・選定扱いにしてはならない」という REQ-39
    /// の資源上限に反するため、[`select_best`] の対象から除外する
    /// （選定対象外だが正解率自体は記録として残す）。
    ScoringExceededBudget {
        /// 参考値としての validation 正解率（選定には使わない）。
        validation_accuracy: ValidationAccuracy,
    },
    /// 探索予算全体が尽きたため実行しなかった。
    NotStarted {
        /// 未着手の理由。
        reason: NotStartedReason,
    },
}

/// 候補 1 件の探索記録（時刻・打ち切り分類・探索結果の組）。
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CandidateSearchEntry {
    /// 候補 ID。
    pub candidate_id: String,
    /// この候補の順番が回ってきた時点での、探索開始からの経過時間
    /// （ミリ秒）。実行順が回ってこなかった候補（探索予算が尽きた時点で
    /// 後続に控えていた候補）は `None`。
    pub elapsed_at_start_ms: Option<u64>,
    /// [`crate::time_allotment::run_candidate`] が返す持ち時間・打ち切り
    /// 分類の記録（候補が実行された場合のみ `Some`）。
    pub time: Option<CandidateTimeRecord>,
    /// 探索結果の分類。
    #[serde(flatten)]
    pub result: CandidateSearchResult,
    /// validation 推論結果（評価済み候補のみ）。JSON には出さない
    /// （モジュール doc「評価契約との関係」・[`CandidateSearchResult`] doc
    /// 参照）。
    #[serde(skip)]
    validation_outcomes: Option<Vec<Outcome>>,
}

impl CandidateSearchEntry {
    /// validation 推論結果（評価済み候補のみ `Some`）。issue #87
    /// （TASK-18.3-1）が McNemar 検定に使うための読み取り専用アクセサ。
    #[must_use]
    pub fn validation_outcomes(&self) -> Option<&[Outcome]> {
        self.validation_outcomes.as_deref()
    }
}

/// 選定の結果。
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SelectionDecision {
    /// 最高 validation 正解率の候補を選定した。
    Selected {
        /// 選定した候補 ID。
        candidate_id: String,
        /// 選定した候補の validation 正解率。
        validation_accuracy: ValidationAccuracy,
        /// 選定規則名（`"validation_accuracy_desc_then_candidate_order"` 固定。
        /// モジュール doc「PoC-17 との差異」参照）。
        rule: String,
        /// 最高正解率で同率だった候補 ID（宣言順。選定した候補自身も含む）。
        tied_candidate_ids: Vec<String>,
    },
    /// 評価済みの候補が 1 件もなく、選定できなかった。
    NoEligibleCandidate,
}

/// 探索全体の記録。
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SearchRecord {
    /// 探索予算全体（秒）。
    pub budget_seconds: u64,
    /// 候補 1 件あたりの持ち時間の決め方。
    pub per_candidate_policy: PerCandidatePolicy,
    /// 探索を開始した壁時計時刻（UNIX ミリ秒）。
    pub started_at_unix_ms: u64,
    /// 探索全体の経過時間（単調時計。ミリ秒）。
    pub total_elapsed_ms: u64,
    /// validation gold の件数。
    pub validation_records: u64,
    /// 候補ごとの記録（宣言順）。
    pub candidates: Vec<CandidateSearchEntry>,
    /// 選定結果。
    pub selection: SelectionDecision,
}

/// [`run_search`]・事前検証（[`validate_input`]）のエラー。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SearchError<E> {
    /// 候補が 0 件。
    EmptyCandidates,
    /// 候補数が [`MAX_SEARCH_CANDIDATES`] を超える。
    TooManyCandidates { n_candidates: usize },
    /// validation gold が 0 件。
    EmptyValidation,
    /// validation gold の件数が [`MAX_EVAL_RECORDS`] を超える。
    TooManyValidationRecords { n_records: usize },
    /// 候補数 × validation 件数の積が [`MAX_SEARCH_OUTCOME_CELLS`] を超える。
    TooManyOutcomeCells,
    /// `label_order` が空・空文字列を含む・重複を含む。
    InvalidLabelOrder,
    /// `validation_gold` の要素が `label_order` に存在しない。
    UnknownValidationGold { index: usize },
    /// 候補 ID が空・[`MAX_CANDIDATE_ID_BYTES`] 超過・制御文字を含む。
    InvalidCandidateId { index: usize },
    /// 候補 ID が他の候補と重複している。
    DuplicateCandidateId { index: usize },
    /// 候補の `params.label_order` が `label_order` と一致しない。
    LabelOrderMismatch { index: usize },
    /// 候補間で `(root, out_dir)` が重複している。
    DuplicateOutDir { index: usize },
    /// 候補のリクエスト構成要素が [`TrainRequest::new`] の検証を満たさない。
    InvalidRequest {
        index: usize,
        source: TrainRequestError,
    },
    /// 持ち時間の配分に失敗した（[`allot`] のエラー）。
    Allotment(crate::time_allotment::TimeAllotmentError),
    /// 時計の異常。
    Clock(crate::time_allotment::TimeAllotmentError),
    /// 候補の実行（[`run_candidate`]）が失敗した。探索全体を中断する。
    Candidate {
        index: usize,
        source: CandidateTimeError<E>,
    },
    /// [`ValidationScorer::predict_validation`] が返した件数が
    /// `validation_gold` と一致しない（契約違反。fail-closed で探索全体を
    /// 中断する）。
    ScorerOutputMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
    /// 評価器（[`fandhe_edge_eval::metrics::evaluate_single_select`]）が
    /// 失敗した。
    Eval(EvalError),
    /// 選定処理の内部矛盾（理論上到達しない防御的分岐）。
    Internal { detail: String },
}

impl<E: std::fmt::Display> std::fmt::Display for SearchError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::EmptyCandidates => write!(f, "candidates must not be empty"),
            SearchError::TooManyCandidates { n_candidates } => {
                write!(
                    f,
                    "too many candidates: {n_candidates} exceeds limit {MAX_SEARCH_CANDIDATES}"
                )
            }
            SearchError::EmptyValidation => write!(f, "validation_gold must not be empty"),
            SearchError::TooManyValidationRecords { n_records } => write!(
                f,
                "too many validation records: {n_records} exceeds limit {MAX_EVAL_RECORDS}"
            ),
            SearchError::TooManyOutcomeCells => write!(
                f,
                "candidates * validation_gold exceeds limit {MAX_SEARCH_OUTCOME_CELLS}"
            ),
            SearchError::InvalidLabelOrder => write!(
                f,
                "label_order must be non-empty and free of empty or duplicate ids"
            ),
            SearchError::UnknownValidationGold { index } => {
                write!(f, "unknown validation gold label at index {index}")
            }
            SearchError::InvalidCandidateId { index } => {
                write!(f, "invalid candidate id at index {index}")
            }
            SearchError::DuplicateCandidateId { index } => {
                write!(f, "duplicate candidate id at index {index}")
            }
            SearchError::LabelOrderMismatch { index } => {
                write!(f, "candidate label_order mismatch at index {index}")
            }
            SearchError::DuplicateOutDir { index } => {
                write!(f, "duplicate (root, out_dir) at index {index}")
            }
            SearchError::InvalidRequest { index, source } => {
                write!(f, "invalid train request at index {index}: {source}")
            }
            SearchError::Allotment(e) => write!(f, "time allotment error: {e}"),
            SearchError::Clock(e) => write!(f, "clock error: {e}"),
            SearchError::Candidate { index, source } => {
                write!(f, "candidate {index} failed: {source}")
            }
            SearchError::ScorerOutputMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "scorer output mismatch at index {index}: expected {expected}, got {actual}"
            ),
            SearchError::Eval(e) => write!(f, "evaluator error: {e}"),
            SearchError::Internal { detail } => write!(f, "internal search error: {detail}"),
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for SearchError<E> {}

/// 候補 ID の検証（非空・[`MAX_CANDIDATE_ID_BYTES`] 以下・制御文字なし）。
fn validate_candidate_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= MAX_CANDIDATE_ID_BYTES && !id.chars().any(|c| c.is_control())
}

/// 事前検証（予算・runner を一切消費しない。fail-closed）。
fn validate_input<E>(input: &SearchInput<'_>) -> Result<(), SearchError<E>> {
    if input.candidates.is_empty() {
        return Err(SearchError::EmptyCandidates);
    }
    if input.candidates.len() > MAX_SEARCH_CANDIDATES {
        return Err(SearchError::TooManyCandidates {
            n_candidates: input.candidates.len(),
        });
    }
    if input.validation_gold.is_empty() {
        return Err(SearchError::EmptyValidation);
    }
    if input.validation_gold.len() > MAX_EVAL_RECORDS {
        return Err(SearchError::TooManyValidationRecords {
            n_records: input.validation_gold.len(),
        });
    }
    let candidates_u64 =
        u64::try_from(input.candidates.len()).map_err(|_| SearchError::Internal {
            detail: "candidate count does not fit in u64".to_string(),
        })?;
    let gold_u64 =
        u64::try_from(input.validation_gold.len()).map_err(|_| SearchError::Internal {
            detail: "validation record count does not fit in u64".to_string(),
        })?;
    let cells = candidates_u64
        .checked_mul(gold_u64)
        .ok_or(SearchError::TooManyOutcomeCells)?;
    if cells > MAX_SEARCH_OUTCOME_CELLS {
        return Err(SearchError::TooManyOutcomeCells);
    }

    // `label_order` の妥当性: 空・空文字列・重複を拒否する。
    if input.label_order.is_empty() {
        return Err(SearchError::InvalidLabelOrder);
    }
    let mut label_set: BTreeSet<&str> = BTreeSet::new();
    for &label in input.label_order {
        if label.is_empty() || !label_set.insert(label) {
            return Err(SearchError::InvalidLabelOrder);
        }
    }

    // `validation_gold` の全要素が `label_order` に含まれること。
    for (index, &gold) in input.validation_gold.iter().enumerate() {
        if !label_set.contains(gold) {
            return Err(SearchError::UnknownValidationGold { index });
        }
    }

    // 候補 ID の検証・重複検出、`label_order` 一致、`(root, out_dir)` 重複、
    // リクエストとしての妥当性。
    let mut seen_ids: BTreeSet<&str> = BTreeSet::new();
    let mut seen_out_dirs: BTreeSet<(&str, &str)> = BTreeSet::new();
    for (index, candidate) in input.candidates.iter().enumerate() {
        if !validate_candidate_id(&candidate.candidate_id) {
            return Err(SearchError::InvalidCandidateId { index });
        }
        if !seen_ids.insert(candidate.candidate_id.as_str()) {
            return Err(SearchError::DuplicateCandidateId { index });
        }
        let params_label_order_matches = candidate.params.label_order.len()
            == input.label_order.len()
            && candidate
                .params
                .label_order
                .iter()
                .zip(input.label_order.iter())
                .all(|(a, b)| a.as_str() == *b);
        if !params_label_order_matches {
            return Err(SearchError::LabelOrderMismatch { index });
        }
        let out_dir_key = (
            candidate.params.root.as_str(),
            candidate.params.out_dir.as_str(),
        );
        if !seen_out_dirs.insert(out_dir_key) {
            return Err(SearchError::DuplicateOutDir { index });
        }
        TrainRequest::new(candidate.params.clone())
            .map_err(|source| SearchError::InvalidRequest { index, source })?;
    }

    Ok(())
}

/// 評価済み候補 1 件（[`select_best`] への入力）。
#[derive(Debug, Clone, Copy)]
pub struct EvaluatedCandidate<'a> {
    /// 候補 ID。
    pub candidate_id: &'a str,
    /// validation 正解率。
    pub accuracy: Ratio,
}

/// [`select_best`] のエラー（理論上到達しない防御的分岐）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectBestError {
    /// 候補間で正解率の分母（validation gold 件数）が食い違う。
    MismatchedDenominators,
}

/// validation 正解率が最も高い候補を選ぶ純関数（乱数を使わない。
/// `records` の走査順に依存しない決定的な処理）。
///
/// 比較は `Ratio::numerator()`（整数）で行い、`f64` の比較は使わない
/// （`.claude/rules/coding-rust.md`「数値・決定性」）。同率のときは
/// 宣言順で先の候補を選ぶ。
///
/// # Errors
///
/// 全候補の分母（validation gold 件数）が一致しない場合に
/// [`SelectBestError::MismatchedDenominators`] を返す（理論上、同一の
/// `validation_gold` から算出される限り到達しない）。
pub fn select_best(
    evaluated: &[EvaluatedCandidate<'_>],
) -> Result<SelectionDecision, SelectBestError> {
    let Some(first) = evaluated.first() else {
        return Ok(SelectionDecision::NoEligibleCandidate);
    };
    let denominator = first.accuracy.denominator();
    if evaluated
        .iter()
        .any(|c| c.accuracy.denominator() != denominator)
    {
        return Err(SelectBestError::MismatchedDenominators);
    }

    let mut best_numerator: Option<u64> = None;
    let mut best_index: Option<usize> = None;
    for (index, candidate) in evaluated.iter().enumerate() {
        let numerator = candidate.accuracy.numerator();
        let is_better = match best_numerator {
            None => true,
            Some(current_best) => numerator > current_best,
        };
        if is_better {
            best_numerator = Some(numerator);
            best_index = Some(index);
        }
    }
    let (Some(best_numerator), Some(best_index)) = (best_numerator, best_index) else {
        return Ok(SelectionDecision::NoEligibleCandidate);
    };
    let Some(best) = evaluated.get(best_index) else {
        return Err(SelectBestError::MismatchedDenominators);
    };

    let tied_candidate_ids: Vec<String> = evaluated
        .iter()
        .filter(|c| c.accuracy.numerator() == best_numerator)
        .map(|c| c.candidate_id.to_string())
        .collect();

    Ok(SelectionDecision::Selected {
        candidate_id: best.candidate_id.to_string(),
        validation_accuracy: ValidationAccuracy::from(best.accuracy),
        rule: "validation_accuracy_desc_then_candidate_order".to_string(),
        tied_candidate_ids,
    })
}

/// 宣言順に残っている候補すべてを、実行順が回ってこなかった候補として
/// `entries` へ記録する（P1 指摘対応・REQ-18「候補ごとの選定記録」）。
///
/// [`run_search`] が探索予算全体を使い切ったと判断した時点（[`Allotment::Exhausted`]
/// または [`ValidationScorer::predict_validation`] の呼び出しが予算を超過した
/// 時点）で、宣言順にまだ控えていた候補を `iter` から取り出し尽くす。
/// これらの候補には「順番が回ってきた」時点の経過時間が存在しないため
/// `elapsed_at_start_ms: None`・`time: None` とする
/// （[`CandidateSearchEntry::elapsed_at_start_ms`] doc 参照）。
fn drain_remaining_as_not_started(
    entries: &mut Vec<CandidateSearchEntry>,
    iter: &mut std::iter::Enumerate<std::vec::IntoIter<SearchCandidate>>,
) {
    for (_, candidate) in iter {
        entries.push(CandidateSearchEntry {
            candidate_id: candidate.candidate_id,
            elapsed_at_start_ms: None,
            time: None,
            result: CandidateSearchResult::NotStarted {
                reason: NotStartedReason::BudgetExhausted,
            },
            validation_outcomes: None,
        });
    }
}

/// 探索予算全体を管理し、複数候補を学習・比較し、選定結果を記録する
/// （TASK-18.1-2・issue #84）。
///
/// 手順: (1) 予算・runner を消費する前にすべての事前検証を行う
/// （[`validate_input`]） → (2) 候補を宣言順に実行し、探索予算の消費を
/// 追跡する → (3) 学習が成功した候補について
/// [`ValidationScorer::predict_validation`] を呼び、評価器で正解率を
/// 算出する → (4) 評価済みの候補から最高正解率の候補を選ぶ
/// （[`select_best`]）。
///
/// # Errors
///
/// 事前検証・候補の実行・評価器のいずれかが失敗した場合に
/// [`SearchError`] を返す。候補単位の失敗（学習が完了しなかった・scorer が
/// 失敗した）は探索全体を中断せず、その候補を該当する分類で記録して次の
/// 候補へ進む。
pub fn run_search<R, S, C>(
    runner: &mut R,
    scorer: &mut S,
    clock: &C,
    input: SearchInput<'_>,
) -> Result<SearchRecord, SearchError<R::Error>>
where
    R: CandidateRunner,
    S: ValidationScorer,
    C: Clock,
{
    validate_input(&input)?;

    let budget_seconds = input.budget.get();
    let budget_ms = budget_seconds.saturating_mul(1000);
    let started_mono = clock.monotonic();
    let started_at_unix_ms = clock.unix_millis().map_err(SearchError::Clock)?;

    let mut entries: Vec<CandidateSearchEntry> = Vec::new();
    let mut evaluated_owned: Vec<(String, Ratio)> = Vec::new();
    let n_candidates = input.candidates.len();

    // 探索開始からの単調経過時間（ミリ秒）を求める（複数箇所〔候補開始時・
    // 採点呼び出し前後〕から呼ぶため共通化する）。
    let elapsed_ms_since_start = |clock: &C| -> Result<u64, SearchError<R::Error>> {
        clock
            .monotonic()
            .checked_sub(started_mono)
            .ok_or(SearchError::Clock(
                crate::time_allotment::TimeAllotmentError::ClockUnavailable,
            ))
            .and_then(|d| {
                u64::try_from(d.as_millis()).map_err(|_| {
                    SearchError::Clock(crate::time_allotment::TimeAllotmentError::ClockUnavailable)
                })
            })
    };

    let mut candidates_iter = input.candidates.into_iter().enumerate();
    while let Some((index, candidate)) = candidates_iter.next() {
        let elapsed_ms = elapsed_ms_since_start(clock)?;
        let remaining_ms = budget_ms.saturating_sub(elapsed_ms);
        let remaining_seconds = remaining_ms / 1000;
        // 未着手候補数（本候補を含む残り件数）。`n_candidates >= index + 1`
        // であり、探索の入力検証で候補数は 1 件以上であることを確認済み。
        let remaining_count = n_candidates.saturating_sub(index);
        let Some(remaining_count) = NonZeroUsize::new(remaining_count) else {
            return Err(SearchError::Internal {
                detail: "remaining candidate count must not be zero".to_string(),
            });
        };

        let allotment = allot(remaining_seconds, remaining_count, input.policy)
            .map_err(SearchError::Allotment)?;
        let allotted = match allotment {
            Allotment::Granted(allotted) => allotted,
            Allotment::Exhausted => {
                entries.push(CandidateSearchEntry {
                    candidate_id: candidate.candidate_id,
                    elapsed_at_start_ms: Some(elapsed_ms),
                    time: None,
                    result: CandidateSearchResult::NotStarted {
                        reason: NotStartedReason::BudgetExhausted,
                    },
                    validation_outcomes: None,
                });
                // P1 指摘対応（REQ-18）: 予算が尽きた時点で宣言順に控えていた
                // 残り候補も、実行順が回ってこなかったこと（本候補を含まない）
                // を記録に残す（宣言順の全候補記録という契約。モジュール doc
                // `CandidateSearchEntry::elapsed_at_start_ms` 参照）。
                drain_remaining_as_not_started(&mut entries, &mut candidates_iter);
                break;
            }
        };

        let run = run_candidate(runner, clock, candidate.params, allotted)
            .map_err(|source| SearchError::Candidate { index, source })?;

        match run.outcome() {
            TrainOutcome::Ok(success) => {
                // P0 指摘対応（REQ-39）: 採点呼び出しの直前に残っている探索
                // 予算全体を `time_limit` として scorer へ渡す（trait doc
                // 「時間上限」参照。呼び出し自体を打ち切ることはできない）。
                let elapsed_before_scoring_ms = elapsed_ms_since_start(clock)?;
                let remaining_for_scoring_ms = budget_ms.saturating_sub(elapsed_before_scoring_ms);
                let time_limit = Duration::from_millis(remaining_for_scoring_ms);
                match scorer.predict_validation(&candidate.candidate_id, success, time_limit) {
                    Ok(outcomes) => {
                        if outcomes.len() != input.validation_gold.len() {
                            return Err(SearchError::ScorerOutputMismatch {
                                index,
                                expected: input.validation_gold.len(),
                                actual: outcomes.len(),
                            });
                        }
                        let eval_records: Vec<EvalRecord<'_>> = input
                            .validation_gold
                            .iter()
                            .zip(outcomes.iter())
                            .map(|(&gold, outcome)| EvalRecord { gold, outcome })
                            .collect();
                        let metrics =
                            metrics::evaluate_single_select(input.label_order, &eval_records)
                                .map_err(SearchError::Eval)?;
                        let accuracy = metrics.accuracy.overall;

                        // 採点呼び出しに時間がかかり、探索予算全体を使い切って
                        // いたら選定対象から除外する（P0 指摘対応。「超過後も
                        // 最後の候補なら Selected を返してしまう」ことを防ぐ。
                        // fail-closed: 正解率自体は参考値として記録するが
                        // `evaluated_owned` へは積まない）。
                        let elapsed_after_scoring_ms = elapsed_ms_since_start(clock)?;
                        if elapsed_after_scoring_ms > budget_ms {
                            entries.push(CandidateSearchEntry {
                                candidate_id: candidate.candidate_id,
                                elapsed_at_start_ms: Some(elapsed_ms),
                                time: Some(run.record().clone()),
                                result: CandidateSearchResult::ScoringExceededBudget {
                                    validation_accuracy: ValidationAccuracy::from(accuracy),
                                },
                                validation_outcomes: None,
                            });
                            drain_remaining_as_not_started(&mut entries, &mut candidates_iter);
                            break;
                        }

                        evaluated_owned.push((candidate.candidate_id.clone(), accuracy));
                        entries.push(CandidateSearchEntry {
                            candidate_id: candidate.candidate_id,
                            elapsed_at_start_ms: Some(elapsed_ms),
                            time: Some(run.record().clone()),
                            result: CandidateSearchResult::Evaluated {
                                validation_accuracy: ValidationAccuracy::from(accuracy),
                            },
                            validation_outcomes: Some(outcomes),
                        });
                    }
                    Err(_scorer_error) => {
                        entries.push(CandidateSearchEntry {
                            candidate_id: candidate.candidate_id,
                            elapsed_at_start_ms: Some(elapsed_ms),
                            time: Some(run.record().clone()),
                            result: CandidateSearchResult::ScoringFailed,
                            validation_outcomes: None,
                        });
                    }
                }
            }
            TrainOutcome::Error(_) => {
                entries.push(CandidateSearchEntry {
                    candidate_id: candidate.candidate_id,
                    elapsed_at_start_ms: Some(elapsed_ms),
                    time: Some(run.record().clone()),
                    result: CandidateSearchResult::TrainingNotCompleted,
                    validation_outcomes: None,
                });
            }
        }
    }

    let ended_mono = clock.monotonic();
    let total_elapsed_ms = ended_mono
        .checked_sub(started_mono)
        .ok_or_else(|| {
            SearchError::Clock(crate::time_allotment::TimeAllotmentError::ClockUnavailable)
        })
        .and_then(|d| {
            u64::try_from(d.as_millis()).map_err(|_| {
                SearchError::Clock(crate::time_allotment::TimeAllotmentError::ClockUnavailable)
            })
        })?;

    let evaluated: Vec<EvaluatedCandidate<'_>> = evaluated_owned
        .iter()
        .map(|(candidate_id, accuracy)| EvaluatedCandidate {
            candidate_id: candidate_id.as_str(),
            accuracy: *accuracy,
        })
        .collect();
    let selection = select_best(&evaluated).map_err(|e| SearchError::Internal {
        detail: format!("{e:?}"),
    })?;

    let validation_records =
        u64::try_from(input.validation_gold.len()).map_err(|_| SearchError::Internal {
            detail: "validation record count does not fit in u64".to_string(),
        })?;

    Ok(SearchRecord {
        budget_seconds,
        per_candidate_policy: input.policy,
        started_at_unix_ms,
        total_elapsed_ms,
        validation_records,
        candidates: entries,
        selection,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ratio(numerator: u64, denominator: u64) -> Ratio {
        // `metrics::evaluate_single_select` を経由して有効な `Ratio` を作る
        // （`Ratio::new` は非公開のため、公開 API を通す）。
        let labels = ["ok", "ng"];
        let mut records_owned: Vec<Outcome> = Vec::new();
        for i in 0..denominator {
            if i < numerator {
                records_owned.push(Outcome::Label("ok".to_string()));
            } else {
                records_owned.push(Outcome::Label("ng".to_string()));
            }
        }
        let gold: Vec<&str> = (0..denominator).map(|_| "ok").collect();
        let records: Vec<EvalRecord<'_>> = gold
            .iter()
            .zip(records_owned.iter())
            .map(|(&g, o)| EvalRecord {
                gold: g,
                outcome: o,
            })
            .collect();
        metrics::evaluate_single_select(&labels, &records)
            .expect("valid metrics")
            .accuracy
            .overall
    }

    /// REQ-18・TASK-18.1-2: 最高正解率の候補を選ぶ（同率なし）。
    #[test]
    fn task18_1_2_select_best_picks_highest_accuracy() {
        let evaluated = vec![
            EvaluatedCandidate {
                candidate_id: "c3-a",
                accuracy: ratio(7, 10),
            },
            EvaluatedCandidate {
                candidate_id: "c3-b",
                accuracy: ratio(9, 10),
            },
            EvaluatedCandidate {
                candidate_id: "c3-c",
                accuracy: ratio(8, 10),
            },
        ];
        let decision = select_best(&evaluated).expect("valid selection");
        match decision {
            SelectionDecision::Selected {
                candidate_id,
                validation_accuracy,
                rule,
                tied_candidate_ids,
            } => {
                assert_eq!(candidate_id, "c3-b");
                assert_eq!(validation_accuracy.correct, 9);
                assert_eq!(validation_accuracy.total, 10);
                assert_eq!(rule, "validation_accuracy_desc_then_candidate_order");
                assert_eq!(tied_candidate_ids, vec!["c3-b".to_string()]);
            }
            SelectionDecision::NoEligibleCandidate => panic!("expected Selected"),
        }
    }

    /// REQ-18・TASK-18.1-2: 同率のときは宣言順で先の候補を選び、
    /// `tied_candidate_ids` に両方が入る。
    #[test]
    fn task18_1_2_select_best_ties_pick_first_declared() {
        let evaluated = vec![
            EvaluatedCandidate {
                candidate_id: "c3-a",
                accuracy: ratio(8, 10),
            },
            EvaluatedCandidate {
                candidate_id: "c3-b",
                accuracy: ratio(8, 10),
            },
        ];
        let decision = select_best(&evaluated).expect("valid selection");
        match decision {
            SelectionDecision::Selected {
                candidate_id,
                tied_candidate_ids,
                ..
            } => {
                assert_eq!(candidate_id, "c3-a");
                assert_eq!(
                    tied_candidate_ids,
                    vec!["c3-a".to_string(), "c3-b".to_string()]
                );
            }
            SelectionDecision::NoEligibleCandidate => panic!("expected Selected"),
        }
    }

    /// REQ-18・TASK-18.1-2: 評価済み候補が 0 件なら `NoEligibleCandidate`。
    #[test]
    fn task18_1_2_select_best_no_eligible_candidate() {
        let evaluated: Vec<EvaluatedCandidate<'_>> = Vec::new();
        let decision = select_best(&evaluated).expect("valid selection");
        assert_eq!(decision, SelectionDecision::NoEligibleCandidate);
    }

    /// REQ-18・TASK-18.1-2: 分母が食い違えば `MismatchedDenominators`。
    #[test]
    fn task18_1_2_select_best_rejects_mismatched_denominators() {
        let evaluated = vec![
            EvaluatedCandidate {
                candidate_id: "c3-a",
                accuracy: ratio(5, 10),
            },
            EvaluatedCandidate {
                candidate_id: "c3-b",
                accuracy: ratio(5, 8),
            },
        ];
        let err = select_best(&evaluated).unwrap_err();
        assert_eq!(err, SelectBestError::MismatchedDenominators);
    }

    /// REQ-18・TASK-18.1-2: 候補 ID の検証（空・上限超過・制御文字）。
    #[test]
    fn task18_1_2_validate_candidate_id_rules() {
        assert!(validate_candidate_id("c3-a"));
        assert!(!validate_candidate_id(""));
        assert!(!validate_candidate_id(
            &"a".repeat(MAX_CANDIDATE_ID_BYTES + 1)
        ));
        assert!(!validate_candidate_id("c3\u{0}a"));
        assert!(!validate_candidate_id("c3\na"));
    }

    fn valid_params(root: &str, out_dir: &str) -> TrainRequestParams {
        TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["positive".to_string(), "negative".to_string()],
            max_bytes: 512,
            seed: 0,
            device: crate::request::Device::Cpu,
            root: root.to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: out_dir.to_string(),
            time_limit_seconds: None,
            rss_limit_bytes: None,
        }
    }

    fn base_input<'a>(
        label_order: &'a [&'a str],
        validation_gold: &'a [&'a str],
        candidates: Vec<SearchCandidate>,
    ) -> SearchInput<'a> {
        SearchInput {
            label_order,
            validation_gold,
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        }
    }

    /// REQ-18・TASK-18.1-2・REQ-39: 候補 0 件は `EmptyCandidates`。
    #[test]
    fn task18_1_2_validate_input_rejects_empty_candidates() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let input = base_input(&label_order, &gold, Vec::new());
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::EmptyCandidates);
    }

    /// REQ-18・TASK-18.1-2: validation gold が空は `EmptyValidation`。
    #[test]
    fn task18_1_2_validate_input_rejects_empty_validation() {
        let label_order = ["positive", "negative"];
        let gold: [&str; 0] = [];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::EmptyValidation);
    }

    /// REQ-18・TASK-18.1-2: `label_order` に無い gold ラベルは
    /// `UnknownValidationGold`。
    #[test]
    fn task18_1_2_validate_input_rejects_unknown_validation_gold() {
        let label_order = ["positive", "negative"];
        let gold = ["positive", "unknown"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::UnknownValidationGold { index: 1 });
    }

    /// REQ-18・TASK-18.1-2: 候補 ID の重複は `DuplicateCandidateId`。
    #[test]
    fn task18_1_2_validate_input_rejects_duplicate_candidate_id() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params("/root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params("/root", "out/b"),
            },
        ];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateCandidateId { index: 1 });
    }

    /// REQ-18・TASK-18.1-2: 候補の `label_order` 不一致は `LabelOrderMismatch`。
    #[test]
    fn task18_1_2_validate_input_rejects_label_order_mismatch() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let mut params = valid_params("/root", "out/a");
        params.label_order = vec!["negative".to_string(), "positive".to_string()];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params,
        }];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::LabelOrderMismatch { index: 0 });
    }

    /// REQ-18・TASK-18.1-2・REQ-39: `(root, out_dir)` の重複は
    /// `DuplicateOutDir`。
    #[test]
    fn task18_1_2_validate_input_rejects_duplicate_out_dir() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params("/root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/root", "out/a"),
            },
        ];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateOutDir { index: 1 });
    }

    /// REQ-18・TASK-18.1-2: リクエストとして不正な構成要素は
    /// `InvalidRequest`。
    #[test]
    fn task18_1_2_validate_input_rejects_invalid_request() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let mut params = valid_params("/root", "out/a");
        params.kind = String::new();
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params,
        }];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert!(matches!(
            err,
            SearchError::InvalidRequest {
                index: 0,
                source: TrainRequestError::EmptyKind
            }
        ));
    }

    /// REQ-18・TASK-18.1-2・REQ-39: 候補数が [`MAX_SEARCH_CANDIDATES`] を
    /// 超えると `TooManyCandidates`（確保前に拒否する）。
    #[test]
    fn task18_1_2_validate_input_rejects_too_many_candidates() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates: Vec<SearchCandidate> = (0..=MAX_SEARCH_CANDIDATES)
            .map(|i| SearchCandidate {
                candidate_id: format!("c{i}"),
                params: valid_params("/root", &format!("out/{i}")),
            })
            .collect();
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(
            err,
            SearchError::TooManyCandidates {
                n_candidates: MAX_SEARCH_CANDIDATES + 1
            }
        );
    }

    /// REQ-18・TASK-18.1-2・REQ-39: validation 件数が [`MAX_EVAL_RECORDS`]
    /// を超えると `TooManyValidationRecords`。
    #[test]
    fn task18_1_2_validate_input_rejects_too_many_validation_records() {
        let label_order = ["positive", "negative"];
        let gold: Vec<&str> = vec!["positive"; MAX_EVAL_RECORDS + 1];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(
            err,
            SearchError::TooManyValidationRecords {
                n_records: MAX_EVAL_RECORDS + 1
            }
        );
    }

    /// REQ-18・TASK-18.1-2・REQ-39: 候補数 × validation 件数の積が
    /// [`MAX_SEARCH_OUTCOME_CELLS`] を超えると `TooManyOutcomeCells`
    /// （それぞれ個別の上限〔`MAX_SEARCH_CANDIDATES`・`MAX_EVAL_RECORDS`〕は
    /// 超えない構成で確認する）。
    #[test]
    fn task18_1_2_validate_input_rejects_too_many_outcome_cells() {
        let label_order = ["positive", "negative"];
        let gold: Vec<&str> = vec!["positive"; 40_000];
        let candidates: Vec<SearchCandidate> = (0..MAX_SEARCH_CANDIDATES)
            .map(|i| SearchCandidate {
                candidate_id: format!("c{i}"),
                params: valid_params("/root", &format!("out/{i}")),
            })
            .collect();
        assert!(candidates.len() <= MAX_SEARCH_CANDIDATES);
        assert!(gold.len() <= MAX_EVAL_RECORDS);
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::TooManyOutcomeCells);
    }

    /// REQ-18・TASK-18.1-2: `label_order` の空文字列・重複は
    /// `InvalidLabelOrder`。
    #[test]
    fn task18_1_2_validate_input_rejects_invalid_label_order() {
        let gold = ["positive"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        for label_order in [vec!["positive", ""], vec!["positive", "positive"]] {
            let input = base_input(&label_order, &gold, candidates.clone());
            let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
            assert_eq!(err, SearchError::InvalidLabelOrder, "case: {label_order:?}");
        }
    }

    /// REQ-18・TASK-18.1-2・REQ-39: 候補 ID が不正（空・制御文字混入）だと
    /// `InvalidCandidateId`。
    #[test]
    fn task18_1_2_validate_input_rejects_invalid_candidate_id() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3\na".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::InvalidCandidateId { index: 0 });
    }
}
