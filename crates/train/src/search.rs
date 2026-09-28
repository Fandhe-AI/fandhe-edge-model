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
//! - [`ValidationScorer::predict_validation`] へは `candidate_id`・学習
//!   成果物（[`crate::result::SuccessOutcome`]）・[`ValidationInputRecord`]
//!   の列（record_id・byte 入力の組）を渡し、`validation_gold`
//!   （正解ラベル）は渡さない（REQ-27「推論関数には `input` だけを渡す」の
//!   学習ワーカー層での対応。gold を渡さない制約は trait の引数リストと
//!   いう型のレベルで保証される）。入力そのものも `run_search` が権威ある
//!   値として渡すのは、scorer が自身で保持する別データ（record_id は
//!   揃っているが中身が異なる入力）を使ってしまうことを防ぐため
//!   （P0 指摘対応。issue #84 PR #238 レビュー）
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
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use fandhe_edge_eval::metrics::{self, EvalError, EvalRecord, Outcome, Ratio};
use fandhe_edge_eval::significance::MAX_EVAL_RECORDS;

use crate::error::TrainRequestError;
use crate::limits::{MAX_LABEL_BYTES, MAX_LABELS, MAX_VALIDATION_INPUT_TOTAL_BYTES};
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

/// [`SearchBudget`] の最大値（秒。REQ-39 資源上限・P0 指摘対応。issue #84
/// PR #238 レビュー）。
///
/// `SearchBudget::new` は非ゼロの `u64` を無条件に受理していたため、探索
/// 全体を極端に長く（`u64::MAX` 秒 ≒ 5,800 億年）実行できる経路があった。
/// 新しい値を作らず、1 候補あたりの持ち時間の上限
/// （[`crate::limits::MAX_TRAIN_WALL_SECONDS`]）× 1 回の探索で許容する
/// 候補数の上限（[`MAX_SEARCH_CANDIDATES`]）という、既存の 2 定数の積から
/// 導く（コーディネーター指摘の提案どおり）。全候補が 1 候補あたりの上限
/// いっぱいまで直列に時間を使っても届かない規模を意図した上限であり、
/// 実際の探索はこれよりずっと早く完了する想定（既定値
/// [`DEFAULT_SEARCH_BUDGET_SECONDS`] は 3600 秒で、本上限の 1/256）。
///
/// `as` によるキャストはコンパイル時定数同士の変換であり、外部入力の経路
/// ではないため `.claude/rules/coding-rust.md`「外部入力では `as` を使わ
/// ない」の対象外（`u64::from` は const 文脈でまだ安定化されていないため
/// `as` を使う。`MAX_TRAIN_WALL_SECONDS`〔u32・3600〕・`MAX_SEARCH_CANDIDATES`
/// 〔usize・256〕はいずれも `u64` への拡大変換で桁あふれしない）。
pub const MAX_SEARCH_BUDGET_SECONDS: u64 =
    crate::limits::MAX_TRAIN_WALL_SECONDS as u64 * MAX_SEARCH_CANDIDATES as u64;

/// 探索予算全体（秒）。0 秒は表現できない（`allot` が 0 秒を
/// [`Allotment::Exhausted`] として扱う契約と揃える）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchBudget(NonZeroU64);

impl SearchBudget {
    /// 秒数を指定して構築する。`seconds == 0`、または
    /// [`MAX_SEARCH_BUDGET_SECONDS`] を超える場合は `None`（REQ-39 資源
    /// 上限・P0 指摘対応。issue #84 PR #238 レビュー）。
    #[must_use]
    pub fn new(seconds: u64) -> Option<Self> {
        if seconds > MAX_SEARCH_BUDGET_SECONDS {
            return None;
        }
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
    /// [`validation_gold`](Self::validation_gold) と同じ順・同じ件数の
    /// validation レコード識別子（凍結済み validation split のレコード ID）。
    /// [`ValidationScorer::predict_validation`] へ
    /// [`validation_inputs`](Self::validation_inputs) と組にして渡し、戻り値の
    /// [`ScoredOutcome::record_id`] 列と突き合わせることで、scorer が
    /// `run_search` の意図した順序と異なる予測（件数は同じだが順序が違う・
    /// 別の record_id の予測）を返していないかを検証する（REQ-27
    /// 「評価の独立性」・P0 指摘対応。issue #84 PR #238 レビュー。scorer が
    /// record_id は正しいが中身の異なる入力を独自に保持しているケースの
    /// 防止は [`validation_inputs`](Self::validation_inputs) が担う）。
    /// 正解ラベルは含まない。`validate_input`（`crate::search`）が `BTreeSet`
    /// へ追加する前に、1 件あたり
    /// [`fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES`]・合計
    /// [`crate::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES`] を超えていないか
    /// 検証する（P1 指摘対応・REQ-39「資源の上限」。issue #84 PR #238
    /// レビュー）。
    pub validation_record_ids: &'a [&'a str],
    /// [`validation_record_ids`](Self::validation_record_ids) と同じ順・
    /// 同じ件数の byte 入力（README「入力表現は byte のみ」）。
    /// [`run_search`] が [`ValidationScorer::predict_validation`] へ
    /// `record_id`・`input` の組として渡す（正解ラベルは渡さない。REQ-27）。
    /// scorer が凍結済み validation split とは異なる入力（自身が独自に
    /// 保持していた古い・別のデータ）で推論することを防ぐため、`run_search`
    /// が権威ある入力を明示的に渡す設計にしている（P0 指摘対応。issue #84
    /// PR #238 レビュー: record_id の一致だけでは、scorer が record_id は
    /// 揃っているが中身が異なる入力を保持していた場合を検出できない）。
    /// `validate_input`（`crate::search`）が scorer 呼び出し前に、1 件あたり
    /// [`fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES`]・合計
    /// [`crate::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES`] を超えていないか
    /// 検証する（`SearchInput` はデータ契約層を経由しない呼び出し元も直接
    /// 組み立てられる公開 API のため。REQ-39「資源の上限」・P0 指摘対応。
    /// issue #84 PR #238 レビュー）。
    pub validation_inputs: &'a [&'a [u8]],
    /// 凍結済み validation split の記録（REQ-17・REQ-27。P0 指摘対応。
    /// issue #84 PR #238 レビュー）。`validate_input` は
    /// [`validation_record_ids`](Self::validation_record_ids) を昇順に並べ
    /// 直した上で、[`crate::split_record`] と同じ正準化ハッシュ規則
    /// （[`fandhe_edge_core::canonical::canonical_sha256_hex`]。新しい規則は
    /// 作らず、`crates/data::split_record` が使うのと同じ関数を再利用する）
    /// で再計算し、この記録の `validation` split のハッシュ
    /// （[`fandhe_edge_data::split_record::SplitRecord::digest`]・
    /// `fandhe_edge_data::split::Split::Validation`）と一致するかを、採点
    /// （[`ValidationScorer::predict_validation`]）を呼び出す前に確認する。
    /// 一致しなければ [`SearchError::ValidationSplitHashMismatch`] で
    /// fail-closed に停止する。凍結した最終 test 分割の record_ids を誤って
    /// validation として渡した場合も、その記録の `validation` split の
    /// ハッシュとは一致しないため同じ経路で拒否される（`SplitRecord` は
    /// train・validation・test の 3 split をまとめて保持する 1 つの記録
    /// であり、常に `Split::Validation` の digest だけと比較することで
    /// 「どの split の記録として渡されたか」を暗黙に検証する。分割ごとに
    /// 別の "kind" フィールドを持たないため、追加の種別照合は不要）。
    pub validation_split_record: &'a fandhe_edge_data::split_record::SplitRecord,
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
/// `candidate_id`・学習成果物・[`ValidationInputRecord`] の列（record_id・
/// byte 入力の組）を受け取り、`validation_gold`（正解ラベル）は受け取らない
/// （REQ-27。gold を渡さない制約は引数リストという型のレベルで保証される）。
/// validation 入力そのものは [`run_search`] が [`SearchInput`] から権威ある
/// 値として渡す（P0 指摘対応・REQ-27「評価の独立性」。issue #84 PR #238
/// レビュー: 実装側が独自に入力を保持する設計だと、scorer が
/// `run_search` の意図した validation 集合と異なるデータ〔件数・record_id
/// は同じだが中身が違う〕を使って推論しても検出できない）。
///
/// # 時間上限（REQ-39・P0 指摘対応。issue #84 PR #238 レビュー）
///
/// `time_limit` は [`run_search`] がこの呼び出し時点で残っている探索予算
/// （探索予算全体 − ここまでの経過時間）を渡す。**この締め切りは
/// `run_search` が強制する契約であり、実装が守ることを期待するだけの
/// 助言ではない**（旧版は「実装側が守る責務を持つ」とする助言的な契約
/// だったが、締め切りを守らない実装を信用してよい理由がなく、fail-closed
/// にならなかったため強制する契約へ改めた）。
///
/// 具体的には、`run_search` は本メソッドの呼び出しを、`self`（scorer 自身の
/// 所有権）を専用スレッドへ渡して実行し、`time_limit` を
/// `std::sync::mpsc::Receiver::recv_timeout` の待ち時間として使う。
/// `time_limit` 以内に戻り値が届かなかった場合、`run_search` は戻り値を
/// 一切使わず、その候補を
/// [`CandidateSearchResult::ScoringTimedOut`]（既存の探索予算超過と同じ
/// 「以降の候補を未着手にして探索を終える」扱い）として記録する。
/// **締め切りを過ぎて実行中のスレッドは取り残す**（`join` を待たない。
/// 実装が `time_limit` を過ぎても戻らない場合、そのスレッドはプロセスが
/// 終了するまで動き続けうる。`run_search` はその後の呼び出しで scorer の
/// 所有権を取り戻せないため、以降の候補も採点できない。「時間上限を守る
/// 実装だけを受け付ける」契約であり、本 trait を実装する側は
/// `time_limit` 以内に必ず戻ることが要求される）。
///
/// この専用スレッドへ渡す都合上、本 trait は `Send + 'static` を要求する
/// （実装・[`SuccessOutcome`]・[`ValidationInputRecord`] の所有データは
/// いずれもスレッド境界を越えられる必要がある。`records` はスレッド内で
/// 所有データから組み立て直す）。
///
/// 締め切り内に戻った場合、`run_search` は次の 3 段階で経過時間を確認し、
/// 各段階で探索予算全体を使い切っていることを検出した場合はそれ以降の
/// 重い処理を行わない（fail-closed。「期限後も重い処理を続ける」ことを
/// 防ぐ）:
///
/// 1. 学習完了直後（本呼び出しの前）にすでに 0 であることを検出した場合は
///    本呼び出しを行わない（[`CandidateSearchResult::ScoringSkippedBudgetExhausted`]）
/// 2. 本呼び出しから戻った直後、正解率算出（`EvalRecord` の構築・評価器
///    `evaluate_single_select` の呼び出し）より前に検出した場合は、
///    正解率を算出せずに打ち切る
///    （同じく [`CandidateSearchResult::ScoringSkippedBudgetExhausted`]。
///    P1 指摘対応: 評価器という重い処理〔最大 `MAX_SEARCH_OUTCOME_CELLS`
///    件〕を予算超過後に呼び出さない）
/// 3. 評価器の呼び出しまで完了し正解率を算出できた後に検出した場合は、
///    その候補を選定対象から除外する
///    （[`CandidateSearchResult::ScoringExceededBudget`]。正解率は参考値
///    として記録する）
///
/// いずれの段階で打ち切っても以降の候補は未着手として記録する。
pub trait ValidationScorer: Send + 'static {
    /// 実装固有のエラー型。専用スレッドから
    /// `std::sync::mpsc::Sender::send` で送り返すため `Send + 'static` を
    /// 要求する（trait doc「時間上限」参照）。
    type Error: Send + 'static;
    /// `candidate_id` の学習成果物で `records`（[`run_search`] が
    /// [`SearchInput::validation_record_ids`]・[`SearchInput::validation_inputs`]
    /// から組み立てて渡す権威ある validation 入力。**正解ラベルは含まない**。
    /// REQ-27）を推論し、[`ScoredOutcome`] の列を返す。**戻り値の
    /// `record_id` 列は `records` の `record_id` 列と（順序を含めて）完全に
    /// 一致させなければならない**（位置で対応づける。実装は `records` の
    /// 順に予測を並べて返す）。[`run_search`] はこの一致（および件数の
    /// 一致）を検証し、いずれかが崩れている場合は scorer のエラーと同じ
    /// 扱い（[`CandidateSearchResult::ScoringFailed`]。候補単位で選定対象外
    /// にし、探索全体は中断せず次候補へ進む）にする（P0/P1 指摘対応・
    /// issue #84 PR #238 レビュー: 件数だけを照合すると、件数が同じ
    /// 別データ・順序違いの予測でも正解率を算出できてしまい、評価の独立性
    /// 〔REQ-27〕が壊れる。件数不一致を探索全体の致命的エラーにすると、
    /// それまでの候補の記録を失う非対称が生じるため、record_id 不一致と
    /// 同じ扱いに統一した。`run_search` が入力そのものも渡すのは、scorer が
    /// 自身で保持する別データ〔record_id は揃っているが中身が異なる〕を
    /// 使うことも防ぐため）。`time_limit` はこの呼び出し時点で残っている
    /// 探索予算全体（trait doc「時間上限」参照）。
    fn predict_validation(
        &mut self,
        candidate_id: &str,
        artifact: &SuccessOutcome,
        records: &[ValidationInputRecord<'_>],
        time_limit: Duration,
    ) -> Result<Vec<ScoredOutcome>, Self::Error>;
}

/// [`ValidationScorer::predict_validation`] へ渡す validation 入力 1 件
/// （record_id・byte 入力の組。正解ラベルは含まない。REQ-27・P0 指摘対応。
/// issue #84 PR #238 レビュー）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationInputRecord<'a> {
    /// validation レコード識別子（[`SearchInput::validation_record_ids`] の
    /// 要素）。
    pub record_id: &'a str,
    /// byte 入力（README「入力表現は byte のみ」。
    /// [`SearchInput::validation_inputs`] の要素）。
    pub input: &'a [u8],
}

/// [`ValidationScorer::predict_validation`] が返す予測 1 件
/// （record_id 付き。P0 指摘対応・REQ-27。issue #84 PR #238 レビュー）。
///
/// `record_id` は [`SearchInput::validation_record_ids`] の要素と対応する
/// 識別子で、[`run_search`] が戻り値の並びを検証するために使う。正解
/// ラベルは含まない。
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredOutcome {
    /// validation レコード識別子。
    pub record_id: String,
    /// 推論結果。
    pub outcome: Outcome,
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
    /// 学習は成功したが採点が無効だった。次の 3 通りをまとめて表す:
    /// (1) [`ValidationScorer::predict_validation`] が `Err` を返した場合
    /// （エラー内容は記録しない。security.md）。
    /// (2) `predict_validation` は成功したが、戻り値の件数が
    /// `validation_gold` と一致しなかった場合（P1 指摘対応。issue #84
    /// PR #238 レビュー。以前は探索全体を打ち切る `SearchError` にしており、
    /// それまでの候補の記録を失っていた非対称を解消した）。
    /// (3) `predict_validation` は成功したが、戻り値の `record_id` 列が
    /// `run_search` の渡した validation レコードと（順序を含めて）一致しな
    /// かった場合（P0 指摘対応・REQ-27「評価の独立性」。issue #84 PR #238
    /// レビュー）。
    ///
    /// (2)・(3) はいずれも件数だけ・record_id だけの部分的な一致では
    /// 別データ・順序違いの予測を見抜けないため、契約違反を採点失敗と同じ
    /// 扱いにする。事前検証（`validate_input`）が検出する入力側の件数不一致
    /// （`ValidationRecordIdCountMismatch`・`ValidationInputCountMismatch`）
    /// とは別の契約層（scorer の実行時の戻り値）であり、そちらは引き続き
    /// 致命的な `SearchError` のままにする。
    ScoringFailed,
    /// 学習・validation 推論・正解率算出まで完了したが、採点
    /// （[`ValidationScorer::predict_validation`]）または評価器
    /// （`evaluate_single_select`）の呼び出しに時間がかかり探索予算全体を
    /// 使い切った（P0/P1 指摘対応。[`ValidationScorer`] trait doc「時間上限」
    /// 参照）。正解率は算出できているが、探索予算を超過した後の結果を選定に
    /// 使うと「合格・選定扱いにしてはならない」という REQ-39 の資源上限に
    /// 反するため、[`select_best`] の対象から除外する（選定対象外だが正解率
    /// 自体は記録として残す）。評価器の呼び出し前に超過が確定していた場合は
    /// 評価器を呼ばずに正解率も算出しないため、代わりに
    /// [`ScoringSkippedBudgetExhausted`](Self::ScoringSkippedBudgetExhausted)
    /// になる（issue #84 PR #238 レビュー）。
    ScoringExceededBudget {
        /// 参考値としての validation 正解率（選定には使わない）。
        validation_accuracy: ValidationAccuracy,
    },
    /// 探索予算全体を使い切ったため、正解率の算出につながる処理を打ち切った
    /// （P0/P1 指摘対応・REQ-39。issue #84 PR #238 レビュー）。次の 2 通りを
    /// まとめて表す:
    ///
    /// 1. 採点（[`ValidationScorer::predict_validation`]）を呼び出す前の
    ///    時点で探索予算全体を使い切っていたため、採点自体を呼び出さな
    ///    かった場合
    /// 2. 採点は呼び出し・完了したが、直後に確認した時点で探索予算全体を
    ///    使い切っていたため、正解率算出（`EvalRecord` の構築・
    ///    `evaluate_single_select` の呼び出し。最大 `MAX_SEARCH_OUTCOME_CELLS`
    ///    件）を行わなかった場合（P1 指摘対応。評価器の呼び出しは資源を
    ///    要するため、予算超過が確定した時点でスキップし「期限後も重い処理を
    ///    続ける」経路を作らない）
    ///
    /// いずれも正解率を算出していない（できない）ため `0` 等の値で埋めない
    /// （evaluation-contract「分母が 0 の指標は `null`」と同じ「実測できない
    /// 値を捏造しない」方針）。[`ScoringExceededBudget`](Self::ScoringExceededBudget)
    /// は評価器の呼び出しまで完了し正解率を算出できた後に超過を検出した
    /// 場合で、本バリアントとは正解率の有無で区別する。選定対象外。
    ScoringSkippedBudgetExhausted,
    /// 候補の実測時間（[`CandidateTimeRecord::elapsed_ms`]）が、その候補へ
    /// 配分した持ち時間（[`CandidateTimeRecord::time_limit_seconds`]）を
    /// 超えていた（P0 指摘対応・REQ-39。issue #84 PR #238 レビュー。ちょうど
    /// 一致した場合は含めない。[`run_search`] 実装のコメント参照）。
    /// `run_candidate`（[`crate::time_allotment::run_candidate`]）は
    /// ワーカーが `Ok` を返せば `TrainOutcome::Ok` を返すだけで、実測時間が
    /// 割当時間を超えていても `TrainOutcome` 単体では判別できない
    /// （[`crate::time_allotment::CandidateTimeStatus::Completed`] は
    /// 持ち時間超過の有無を問わない）。持ち時間を超えた成功結果を採点・
    /// 選定へ進めると、資源の上限（ガード層）を守らずに「予算到達を合格
    /// 扱いにしない」契約に反するため、採点（[`ValidationScorer`]）を
    /// 呼び出す前に除外する。正解率は算出していないため `0` 等の値で
    /// 埋めない（[`ScoringSkippedBudgetExhausted`](Self::ScoringSkippedBudgetExhausted)
    /// と同じ方針）。選定対象外。探索全体の予算はまだ残っている可能性が
    /// あるため、この候補だけを除外して次候補へ進む（探索全体を打ち切る
    /// `NotStarted`・`ScoringSkippedBudgetExhausted` とは異なり、以降の
    /// `while` ループは継続する）。
    TrainingExceededTimeLimit,
    /// 採点（[`ValidationScorer::predict_validation`]）が締め切り
    /// （`time_limit`）内に戻らなかった（P0 指摘対応・REQ-39。issue #84
    /// PR #238 レビュー。trait doc「時間上限」参照）。`run_search` は
    /// 戻り値を待たずに諦め、呼び出しスレッドを取り残す（正解率は算出して
    /// いないため `0` 等の値で埋めない）。scorer の所有権を取り戻せないため
    /// 以降の候補も採点できず、既存の探索予算超過と同じ扱いで残り候補を
    /// 未着手として記録し探索を終える。選定対象外。
    ScoringTimedOut,
    /// 探索予算全体が尽きたため実行しなかった。
    NotStarted {
        /// 未着手の理由。
        reason: NotStartedReason,
    },
}

/// 候補 1 件の探索記録（時刻・打ち切り分類・探索結果の組）。
///
/// `Debug` は手書きする（下記）。`validation_outcomes`（[`Outcome`] の列。
/// `Outcome::Label` は scorer が返す予測ラベル文字列を保持する）を
/// `derive(Debug)` のまま `{:?}` で出力すると、`#[serde(skip)]` で JSON へは
/// 出していないにもかかわらず、デバッグ出力（ログ等）から予測ラベルの
/// 内容がそのまま漏れてしまう（security.md「秘密情報の混入防止」。P0
/// 指摘対応。issue #84 PR #238 レビュー）。
#[derive(Clone, PartialEq, serde::Serialize)]
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

/// [`CandidateSearchEntry`] の手書き `Debug` が使う補助型。
/// `validation_outcomes` の要素数だけを表示し、予測ラベルの内容
/// （`Outcome::Label` の文字列）は一切出さない。
struct RedactedOutcomeCount(usize);

impl std::fmt::Debug for RedactedOutcomeCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<redacted: {} outcome(s)>", self.0)
    }
}

impl std::fmt::Debug for CandidateSearchEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandidateSearchEntry")
            .field("candidate_id", &self.candidate_id)
            .field("elapsed_at_start_ms", &self.elapsed_at_start_ms)
            .field("time", &self.time)
            .field("result", &self.result)
            .field(
                "validation_outcomes",
                &self
                    .validation_outcomes
                    .as_ref()
                    .map(|outcomes| RedactedOutcomeCount(outcomes.len())),
            )
            .finish()
    }
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
    /// `validation_record_ids` の件数が `validation_gold` と一致しない
    /// （P0 指摘対応・REQ-27。issue #84 PR #238 レビュー）。
    ValidationRecordIdCountMismatch { expected: usize, actual: usize },
    /// `validation_record_ids` に空文字列の要素が含まれる。
    InvalidValidationRecordId { index: usize },
    /// `validation_record_ids` に重複した要素が含まれる（scorer からの
    /// 戻り値を順序で一意に対応づけられなくなるため拒否する）。
    DuplicateValidationRecordId { index: usize },
    /// `validation_record_ids` の 1 件が
    /// [`fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES`] を超える
    /// （P1 指摘対応・REQ-39「資源の上限」。issue #84 PR #238 レビュー）。
    ValidationRecordIdTooLong {
        index: usize,
        size: usize,
        limit: usize,
    },
    /// `validation_record_ids` の合計バイト数が
    /// [`crate::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES`] を超える
    /// （P1 指摘対応・REQ-39「資源の上限」。issue #84 PR #238 レビュー）。
    ValidationRecordIdTotalBytesExceeded { total: usize, limit: usize },
    /// `validation_inputs` の件数が `validation_gold` と一致しない
    /// （P0 指摘対応・REQ-27。issue #84 PR #238 レビュー）。
    ValidationInputCountMismatch { expected: usize, actual: usize },
    /// `validation_inputs` の 1 件が
    /// [`fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES`] を超える
    /// （P0 指摘対応・REQ-39「資源の上限」。issue #84 PR #238 レビュー）。
    ValidationInputTooLarge {
        index: usize,
        size: usize,
        limit: usize,
    },
    /// `validation_inputs` の合計バイト数が
    /// [`crate::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES`] を超える
    /// （P0 指摘対応・REQ-39「資源の上限」。issue #84 PR #238 レビュー）。
    ValidationInputTotalBytesExceeded { total: usize, limit: usize },
    /// `validation_record_ids` から再計算したハッシュが、
    /// [`SearchInput::validation_split_record`] の `validation` split の
    /// ハッシュと一致しない（REQ-17・REQ-27・P0 指摘対応。issue #84
    /// PR #238 レビュー。凍結した最終 test 分割を validation として渡した
    /// 場合もこの経路で拒否される。採点〔`ValidationScorer::predict_validation`〕
    /// を呼び出す前に fail-closed で停止する）。
    ValidationSplitHashMismatch,
    /// 候補 ID が空・[`MAX_CANDIDATE_ID_BYTES`] 超過・制御文字を含む。
    InvalidCandidateId { index: usize },
    /// 候補 ID が他の候補と重複している。
    DuplicateCandidateId { index: usize },
    /// 候補の `params.label_order` が `label_order` と一致しない。
    LabelOrderMismatch { index: usize },
    /// 候補間で `root` と `out_dir` を結合した出力先（symlink 解決後）が
    /// 重複、または一方が他方の祖先（親ディレクトリ）になっている
    /// （codex review PR #238 P1 指摘。`canonicalized_out_dir_key`・
    /// `out_dirs_conflict` 参照）。
    DuplicateOutDir { index: usize },
    /// 出力先の symlink 解決（[`canonicalized_out_dir_key`]）が失敗した
    /// （P1 指摘対応・REQ-39。issue #84 PR #238 レビュー。「存在しない」
    /// 以外の理由での失敗を fail-closed に倒す。権限不足・symlink ループ
    /// 等）。
    OutDirCanonicalizeFailed { index: usize },
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
            SearchError::ValidationRecordIdCountMismatch { expected, actual } => write!(
                f,
                "validation_record_ids count mismatch: expected {expected}, got {actual}"
            ),
            SearchError::InvalidValidationRecordId { index } => {
                write!(f, "invalid (empty) validation record id at index {index}")
            }
            SearchError::DuplicateValidationRecordId { index } => {
                write!(f, "duplicate validation record id at index {index}")
            }
            SearchError::ValidationRecordIdTooLong { index, size, limit } => write!(
                f,
                "validation record id at index {index} is too long: {size} bytes exceeds limit {limit}"
            ),
            SearchError::ValidationRecordIdTotalBytesExceeded { total, limit } => write!(
                f,
                "validation_record_ids total size {total} bytes exceeds limit {limit}"
            ),
            SearchError::ValidationInputCountMismatch { expected, actual } => write!(
                f,
                "validation_inputs count mismatch: expected {expected}, got {actual}"
            ),
            SearchError::ValidationInputTooLarge { index, size, limit } => write!(
                f,
                "validation input at index {index} is too large: {size} bytes exceeds limit {limit}"
            ),
            SearchError::ValidationInputTotalBytesExceeded { total, limit } => write!(
                f,
                "validation_inputs total size {total} bytes exceeds limit {limit}"
            ),
            SearchError::ValidationSplitHashMismatch => write!(
                f,
                "validation_record_ids hash does not match the frozen validation split record"
            ),
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
            SearchError::OutDirCanonicalizeFailed { index } => {
                write!(f, "failed to canonicalize out_dir at index {index}")
            }
            SearchError::InvalidRequest { index, source } => {
                write!(f, "invalid train request at index {index}: {source}")
            }
            SearchError::Allotment(e) => write!(f, "time allotment error: {e}"),
            SearchError::Clock(e) => write!(f, "clock error: {e}"),
            SearchError::Candidate { index, source } => {
                write!(f, "candidate {index} failed: {source}")
            }
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

/// `(root, out_dir)` 重複検出用に、パスを構成要素の列へ正規化する
/// （REQ-39 経路の検証。codex review PR #238 P1 指摘）。
///
/// `crates/train/src/request.rs` の `check_relative_path_syntax` と同じ
/// 「`/` 区切りの構成要素のうち空要素・`.` 単体は無視する」規則で比較用の
/// 表現を作る。これにより `out/a`・`out/./a`・`out//a`・`out/a/` は同一の
/// 出力先として重複検出される。本関数は分割・除去のみを行い、`..` 構成要素
/// はそのまま残す。`..` の字句上の解決（親ディレクトリへの遡上）は
/// [`normalized_joined_components`] が呼び出し元として行う（`root` 側に
/// `..` が含まれうるため、単に「`TrainRequest::new` が拒否する」とは言え
/// ない。同関数の doc 参照）。
fn normalized_path_components(value: &str) -> Vec<&str> {
    value
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect()
}

/// `root` と `out_dir` を結合した後の出力先を、構成要素の列へ正規化する
/// （codex review PR #238 P1 指摘: `root` と `out_dir` を別々に正規化して
/// 組として比較すると、`root="/a", out_dir="out/x"` と
/// `root="/a/out", out_dir="x"` が同じ `/a/out/x` を指していても
/// 別物として扱われてしまう）。
///
/// `out_dir` は `..` を含まない相対パスであることを [`TrainRequest::new`]
/// が強制するが、`root` は絶対パスであること（`check_root_syntax`）しか
/// 強制しておらず `..` 構成要素を含みうる（`root="/root/x/..", out_dir=
/// "out/a"` は `root="/root", out_dir="out/a"` と同じ `/root/out/a` を
/// 指す）。そのため単純な連結では不十分で、結合後の構成要素列に対して
/// スタックによる字句上の `..` 解決（一つ前の構成要素を取り除く。スタック
/// が空のまま `..` に出会った場合は無視してそれ以上遡らない）を行う。
/// これは文字列としての字句解決のみであり、途中の構成要素が symlink で
/// ある場合の実体解決（`canonicalize`）は行わない。symlink の解決は
/// 呼び出し元（[`canonicalized_out_dir_key`]）の責務とする（P1 指摘対応・
/// issue #84 PR #238 レビュー: 字句上の正規化だけでは、symlink を経由して
/// 同じ実ディレクトリを指す 2 候補を重複として検出できなかった）。
///
/// 呼び出し元（[`validate_input`]）は本関数を呼ぶ前に必ず
/// [`TrainRequest::new`] を通す。`out_dir` が空・`..` を含む等の不正な値の
/// まま本関数へ渡すと、正規化後の構成要素列が空や root 側へ食い込んだ値に
/// なり得て、無関係な候補との間に偽陽性の重複判定を招くため。
fn normalized_joined_components<'a>(root: &'a str, out_dir: &'a str) -> Vec<&'a str> {
    let mut resolved: Vec<&'a str> = Vec::new();
    for part in normalized_path_components(root)
        .into_iter()
        .chain(normalized_path_components(out_dir))
    {
        if part == ".." {
            resolved.pop();
        } else {
            resolved.push(part);
        }
    }
    resolved
}

/// 2 つの正規化済み出力先が同一か、一方が他方の祖先（親ディレクトリ）に
/// あたるかを判定する。
///
/// 完全一致だけでなく祖先・子孫関係も衝突として扱う。一方の出力先が
/// 他方の配下にある場合、学習ワーカーの成果物書き出し（ディレクトリ丸ごと
/// の書き込み）が他方の候補の成果物と混在・上書きし得るため（REQ-39
/// 「経路の閉じ込め」）。外部入力由来のパスを扱うため添字アクセス
/// （`[]`）は使わず `zip` で比較する（`.claude/rules/coding-rust.md`）。
/// `T` は構成要素の型（字句上の比較には `&str`、symlink 解決後の比較
/// には `String`〔[`canonicalized_out_dir_key`] 参照〕を使う）。
fn out_dirs_conflict<T: PartialEq>(a: &[T], b: &[T]) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

/// `root`・`out_dir` を結合した出力先を、symlink を解決した実体パスの
/// 構成要素列へ正規化する（P1 指摘対応・REQ-39。issue #84 PR #238
/// レビュー）。
///
/// [`normalized_joined_components`] による字句上の正規化（`.`・`..`・
/// 空要素の解決）だけでは、symlink を経由して同じ実ディレクトリを指す
/// 2 つの候補（例: `root="/tmp/link", out_dir="x"` と
/// `root="/tmp/real", out_dir="x"`。`/tmp/link` が `/tmp/real` への
/// symlink）を見分けられない。本関数は結合後のパスのうち、存在する
/// 最も深い祖先ディレクトリを [`std::fs::canonicalize`] で実体パスへ
/// 解決し、まだ存在しない残りの構成要素をそのまま連結することで、
/// symlink 越しの重複も検出できるようにする（`root` は絶対パスのため
/// `/`（ファイルシステムのルート）は通常必ず存在し、祖先を遡る過程は
/// 必ず終端する）。
///
/// # Errors
///
/// すべての祖先（`/` を含む）の `canonicalize` が失敗した場合、最後に
/// 観測した `io::Error` を fail-closed でそのまま返す（コーディネーター
/// 指摘どおり「canonicalize に失敗したら止める側へ倒す」）。`/` は通常の
/// 環境では必ず存在し読み取り可能なため、実務上はこの経路に到達しない
/// 想定（理論上到達しない防御的分岐）。存在しない・確認できない
/// （権限不足で `PermissionDenied` になる場合を含む）祖先は、より浅い
/// 祖先で再試行するためエラーにしない（PermissionDenied は「存在しない」
/// と POSIX では確実には区別できないため。関数 doc 参照）。
fn canonicalized_out_dir_key(root: &str, out_dir: &str) -> std::io::Result<Vec<String>> {
    let components = normalized_joined_components(root, out_dir);
    let mut last_error: Option<std::io::Error> = None;
    for existing_len in (0..=components.len()).rev() {
        let mut candidate = PathBuf::from("/");
        for part in &components[..existing_len] {
            candidate.push(part);
        }
        match std::fs::canonicalize(&candidate) {
            Ok(canonical) => {
                let mut resolved = canonical;
                for part in &components[existing_len..] {
                    resolved.push(part);
                }
                return Ok(path_components_to_strings(&resolved));
            }
            // まだ存在しない祖先、または存在の確認自体ができない祖先
            // （例: `/root` のように途中のディレクトリの検索〔execute〕
            // 権限が無い場合、実際には存在しない配下パスでも OS は
            // `NotFound` ではなく `PermissionDenied` を返すことがある。
            // POSIX の性質上「存在しない」と「権限不足で確認できない」を
            // 確実に区別する手段は無いため、いずれもより浅い祖先で
            // 再試行する。最終的に必ず試す `/`（ファイルシステムの
            // ルート）はどの OS でも通常読み取り可能なため、実務上この
            // ループは必ずどこかで成功する）。
            Err(e) => {
                last_error = Some(e);
                continue;
            }
        }
    }
    // 理論上到達しない防御的分岐: `/` の canonicalize にすら失敗した場合
    // （極端に制限された実行環境等）は、直前に観測した失敗を
    // fail-closed でそのまま返す（コーディネーター指摘どおり「canonicalize
    // に失敗したら止める側へ倒す」）。
    Err(last_error.unwrap_or_else(|| {
        std::io::Error::other("failed to canonicalize any ancestor of out_dir, including \"/\"")
    }))
}

/// [`Path`] の通常の構成要素（ルート・カレントディレクトリ・親ディレクトリ
/// 参照を除く）を所有文字列の列にする（[`canonicalized_out_dir_key`] 用）。
/// 非 UTF-8 のパス（他候補との比較にのみ使い、ファイルを開かないため
/// 情報を失っても安全側に倒れる。`to_string_lossy` で復元不能文字は
/// 置換文字に変換する）。
fn path_components_to_strings(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect()
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

    // `label_order` の妥当性: 空・件数上限・各要素のバイト長上限・空文字列・
    // 重複を拒否する。件数・バイト長の上限確認は `BTreeSet` を作る前に行う
    // （P0 指摘対応・REQ-39。issue #84 PR #238 レビュー）: `label_order` は
    // `SearchInput` の公開フィールドで、後段の `TrainRequest::new`
    // （`LabelOrder::new`）による上限検証より前に集合を組み立てていたため、
    // 上限を大きく超える入力を渡されると検証前に時間・メモリを消費して
    // しまう。上限は新設せず、`TrainRequest`（学習リクエスト）の
    // `label_order` と同じ契約の定数（[`MAX_LABELS`]・[`MAX_LABEL_BYTES`]。
    // `crate::limits`）をそのまま再利用する。
    if input.label_order.is_empty() {
        return Err(SearchError::InvalidLabelOrder);
    }
    if input.label_order.len() > MAX_LABELS {
        return Err(SearchError::InvalidLabelOrder);
    }
    if input
        .label_order
        .iter()
        .any(|label| label.len() > MAX_LABEL_BYTES)
    {
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

    // `validation_record_ids` の妥当性（P0/P1 指摘対応・REQ-27・REQ-39。
    // issue #84 PR #238 レビュー）: `validation_gold` と同じ件数・各要素の
    // バイト長上限・合計バイト数上限・空文字列なし・重複なしを要求する。
    // 件数不一致・空文字列・重複のいずれも、`predict_validation` の戻り値と
    // 順序で対応づけられなくなるため事前に拒否する（`validation_gold.len()`
    // はすでに [`MAX_EVAL_RECORDS`] 以下と確認済みのため、件数の上限は
    // 新たに設けない）。1 件あたりのバイト長は、推論入力 1 件の識別子
    // （`id`）の上限 [`fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES`]
    // （train・infer で共有する既存の ID 系上限）をそのまま使う
    // （`validation_record_ids` は `crate::request::TrainRequest` の
    // `label_order`／候補 ID とは別の識別子であり、性質が最も近いのは
    // 推論入力 1 件の識別子であるため。P1 指摘対応）。合計バイト数は、
    // 同じく公開 API `SearchInput` の集合サイズを抑える
    // [`crate::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES`] を再利用する
    // （record_id 用に新しい定数は起こさず、同じ上限を「1 つの
    // `SearchInput` フィールドが保持できる合計バイト数」の共通の目安として
    // 適用する。承認事項として報告）。`BTreeSet` へ追加する前に長さ・合計を
    // 検証する（REQ-39「集合へ追加する前の検証」）。
    if input.validation_record_ids.len() != input.validation_gold.len() {
        return Err(SearchError::ValidationRecordIdCountMismatch {
            expected: input.validation_gold.len(),
            actual: input.validation_record_ids.len(),
        });
    }
    let mut seen_record_ids: BTreeSet<&str> = BTreeSet::new();
    let mut record_ids_total_bytes: usize = 0;
    for (index, &record_id) in input.validation_record_ids.iter().enumerate() {
        if record_id.is_empty() {
            return Err(SearchError::InvalidValidationRecordId { index });
        }
        if record_id.len() > fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES {
            return Err(SearchError::ValidationRecordIdTooLong {
                index,
                size: record_id.len(),
                limit: fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES,
            });
        }
        record_ids_total_bytes = record_ids_total_bytes
            .checked_add(record_id.len())
            // 理論上到達しない防御的分岐（直前の要素ごとの上限チェックと
            // 同じ方針）。
            .ok_or_else(|| SearchError::Internal {
                detail: "validation_record_ids total bytes overflows usize".to_string(),
            })?;
        if record_ids_total_bytes > MAX_VALIDATION_INPUT_TOTAL_BYTES {
            return Err(SearchError::ValidationRecordIdTotalBytesExceeded {
                total: record_ids_total_bytes,
                limit: MAX_VALIDATION_INPUT_TOTAL_BYTES,
            });
        }
        if !seen_record_ids.insert(record_id) {
            return Err(SearchError::DuplicateValidationRecordId { index });
        }
    }

    // `validation_inputs` の件数も `validation_gold` と一致すること
    // （P0 指摘対応・REQ-27。issue #84 PR #238 レビュー）。
    if input.validation_inputs.len() != input.validation_gold.len() {
        return Err(SearchError::ValidationInputCountMismatch {
            expected: input.validation_gold.len(),
            actual: input.validation_inputs.len(),
        });
    }

    // `validation_inputs` の 1 件あたり・合計のバイト数上限（P0 指摘対応・
    // REQ-39「資源の上限」。issue #84 PR #238 レビュー）: `SearchInput` は
    // 公開 API で、データ契約層（`crates/data`）を経由しない呼び出し元が
    // 直接値を渡せるため、`ValidationScorer::predict_validation` を呼び出す
    // 前に本層でも検証する（データ契約層の検証に依存しない。fail-closed）。
    // 1 件あたりは推論入力 1 件の上限
    // （`fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES`。train・infer
    // で共有）をそのまま使う。合計は `crate::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES`
    // を使う（doc コメント参照）。
    let mut validation_inputs_total_bytes: usize = 0;
    for (index, &validation_input) in input.validation_inputs.iter().enumerate() {
        if validation_input.len() > fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES {
            return Err(SearchError::ValidationInputTooLarge {
                index,
                size: validation_input.len(),
                limit: fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES,
            });
        }
        validation_inputs_total_bytes = validation_inputs_total_bytes
            .checked_add(validation_input.len())
            // 理論上到達しない防御的分岐: 各要素は直前で
            // `MAX_INFER_INPUT_BYTES`（1 MiB）以下と確認済みで、累積も
            // 上限（`MAX_VALIDATION_INPUT_TOTAL_BYTES`）を超えた時点で
            // 早期に拒否するため、`usize` の桁あふれには到達しない
            // （`validate_input` の他の桁あふれガードと同じ方針）。
            .ok_or_else(|| SearchError::Internal {
                detail: "validation_inputs total bytes overflows usize".to_string(),
            })?;
        if validation_inputs_total_bytes > MAX_VALIDATION_INPUT_TOTAL_BYTES {
            return Err(SearchError::ValidationInputTotalBytesExceeded {
                total: validation_inputs_total_bytes,
                limit: MAX_VALIDATION_INPUT_TOTAL_BYTES,
            });
        }
    }

    // `validation_record_ids` が凍結済み validation split の記録と一致する
    // ことを、採点（scorer）を呼び出す前に確認する（P0 指摘対応・REQ-17・
    // REQ-27「評価の独立性」。issue #84 PR #238 レビュー）。`SearchInput`
    // は公開 API で、呼び出し元が凍結済み分割と無関係な・または凍結した
    // 最終 test 分割の record_ids を validation として渡す経路を塞げない
    // ため、`run_search` 自身が権威ある記録
    // （[`SearchInput::validation_split_record`]）と突き合わせる。
    //
    // ハッシュの計算規則は `fandhe_edge_data::split_record` が使うのと
    // 同じ [`fandhe_edge_core::canonical::canonical_sha256_hex`]
    // （「各 split のレコード ID を昇順に並べた正準化 JSON をハッシュする」
    // 規則。`crates/data/src/split_record.rs` モジュール doc 参照）をそのまま
    // 再利用し、新しい正準化規則は作らない。`validation_record_ids` は
    // 重複なし・空文字列なしを検証済みのため、ここでは並べ替えのみ行う。
    let mut sorted_record_ids: Vec<String> = input
        .validation_record_ids
        .iter()
        .map(|&id| id.to_string())
        .collect();
    sorted_record_ids.sort();
    let recomputed_hash = fandhe_edge_core::canonical::canonical_sha256_hex(&sorted_record_ids)
        // 理論上到達しない防御的分岐: `Vec<String>` は常に有効な JSON へ
        // 正準化できるため、失敗はしない想定（`split_record.rs` の
        // `verify_hashes` も同じ関数を同じ理由で `Result` のまま伝播する）。
        .map_err(|_| SearchError::Internal {
            detail: "failed to canonicalize validation_record_ids for hashing".to_string(),
        })?;
    let expected_digest = input
        .validation_split_record
        .digest(fandhe_edge_data::split::Split::Validation);
    if recomputed_hash != expected_digest.sha256() {
        return Err(SearchError::ValidationSplitHashMismatch);
    }

    // 候補 ID の検証・重複検出、`label_order` 一致、`(root, out_dir)` 重複、
    // リクエストとしての妥当性。
    let mut seen_ids: BTreeSet<&str> = BTreeSet::new();
    // 完全一致だけでなく祖先・子孫関係も検出するため、`BTreeSet` ではなく
    // これまでに見た正規化済み出力先の一覧を保持して総当たりで比較する
    // （`MAX_SEARCH_CANDIDATES` で件数上限があるため O(n^2) で問題ない）。
    // `String` を保持するのは、symlink 解決（`canonicalized_out_dir_key`）
    // が実体パスを新たに構築するため、`candidate.params` を借用したままでは
    // 表現できないため（P1 指摘対応・issue #84 PR #238 レビュー）。
    let mut seen_out_dirs: Vec<Vec<String>> = Vec::new();
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
        // `(root, out_dir)` の重複判定より先に `TrainRequest::new` を通す。
        // `out_dir` が空・`..` を含む等の不正な値のまま
        // `canonicalized_out_dir_key` へ渡すと、正規化後の構成要素列が
        // 空（またはロールバックで root 側へ食い込む）になり得て、
        // 無関係な候補と偽陽性の `DuplicateOutDir` を報告してしまう
        // （codex review PR #238 P1 指摘のレビューで判明）。
        TrainRequest::new(candidate.params.clone())
            .map_err(|source| SearchError::InvalidRequest { index, source })?;
        // symlink を解決した実体パスで重複を判定する（P1 指摘対応・REQ-39。
        // issue #84 PR #238 レビュー）。
        let out_dir_key = canonicalized_out_dir_key(
            candidate.params.root.as_str(),
            candidate.params.out_dir.as_str(),
        )
        .map_err(|_| SearchError::OutDirCanonicalizeFailed { index })?;
        if seen_out_dirs
            .iter()
            .any(|seen| out_dirs_conflict(seen, &out_dir_key))
        {
            return Err(SearchError::DuplicateOutDir { index });
        }
        seen_out_dirs.push(out_dir_key);
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

/// [`ValidationInputRecord`] の所有版（`'static`・`Send`）。
///
/// [`call_predict_validation_with_deadline`] が scorer 呼び出し専用スレッド
/// へ渡すために使う（P0 指摘対応・REQ-39。issue #84 PR #238 レビュー）。
/// スレッド境界を越えるには所有データが必要で、[`SearchInput`] から借用した
/// `&str`／`&[u8]` のまま渡すことはできない。呼び出しスレッド内で
/// [`ValidationInputRecord`]（借用版）を本データから組み立て直して
/// [`ValidationScorer::predict_validation`] へ渡す。
#[derive(Debug, Clone)]
struct OwnedValidationInputRecord {
    record_id: String,
    input: Vec<u8>,
}

/// [`ValidationScorer::predict_validation`] を締め切り付きで呼び出した結果
/// （P0 指摘対応・REQ-39。issue #84 PR #238 レビュー）。
enum TimedPredictOutcome<E> {
    /// 締め切り内に戻った。
    Completed(Result<Vec<ScoredOutcome>, E>),
    /// 締め切りを過ぎても戻らなかった（呼び出しスレッドは取り残す）。
    TimedOut,
}

/// `scorer`（所有権）を専用スレッドへ渡して
/// [`ValidationScorer::predict_validation`] を呼び出し、`time_limit` を
/// 締め切りとして強制する（P0 指摘対応・REQ-39。issue #84 PR #238 レビュー。
/// trait doc「時間上限」参照）。
///
/// # 所有権の設計（判断理由）
///
/// `std::thread::spawn`（非スコープ）へ渡すクロージャは `'static` でなければ
/// ならない。`run_search` は締め切りを過ぎたスレッドを `join` せずに取り残す
/// 設計（trait doc参照）のため、`std::thread::scope` のようなスコープ付き
/// スレッド（関数を抜ける前に必ず `join` される）は使えない（スコープを
/// 抜けようとすると実行中のスレッドの完了を待ってしまい、締め切りを
/// 強制する意味がなくなる）。そのため呼び出し側が借用している
/// `&SuccessOutcome`・`&[ValidationInputRecord<'_>]` をそのまま渡すことは
/// できず、`scorer: S`（所有権ごと）・`artifact: SuccessOutcome`
/// （`.clone()` 済み）・`records: Arc<Vec<OwnedValidationInputRecord>>`
/// （全候補で共有するため複製せず `Arc` で参照カウントする）を渡す。
/// 締め切り内に戻れば `scorer` の所有権をチャネル経由で呼び出し元へ返し、
/// 次候補の呼び出しに使い回す。締め切りを過ぎた場合は `scorer` を含む
/// スレッドを丸ごと諦め、`run_search` はそれ以降 scorer を持たない
/// （呼び出し元の [`Option<S>`] が `None` のままになる。「時間上限を守る
/// 実装だけを受け付ける」契約〔trait doc〕の帰結として、以降の候補は
/// 採点できない）。
fn call_predict_validation_with_deadline<S: ValidationScorer>(
    mut scorer: S,
    candidate_id: String,
    artifact: SuccessOutcome,
    records: Arc<Vec<OwnedValidationInputRecord>>,
    time_limit: Duration,
) -> (Option<S>, TimedPredictOutcome<S::Error>) {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let borrowed_records: Vec<ValidationInputRecord<'_>> = records
            .iter()
            .map(|record| ValidationInputRecord {
                record_id: record.record_id.as_str(),
                input: record.input.as_slice(),
            })
            .collect();
        let result =
            scorer.predict_validation(&candidate_id, &artifact, &borrowed_records, time_limit);
        // 受信側が締め切りを過ぎて `rx` を破棄済みなら送信は失敗するが、
        // その場合はこの結果を使う相手がいないだけなので無視してよい
        // （取り残したスレッドをここで正常終了させる。trait doc参照）。
        let _ = tx.send((scorer, result));
    });
    match rx.recv_timeout(time_limit) {
        Ok((returned_scorer, result)) => (
            Some(returned_scorer),
            TimedPredictOutcome::Completed(result),
        ),
        Err(_timeout_or_disconnected) => (None, TimedPredictOutcome::TimedOut),
    }
}

/// 宣言順に残っている候補すべてを、実行順が回ってこなかった候補として
/// `entries` へ記録する（P1 指摘対応・REQ-18「候補ごとの選定記録」）。
///
/// [`run_search`] が探索予算全体を使い切ったと判断した時点（[`Allotment::Exhausted`]、
/// [`ValidationScorer::predict_validation`] の呼び出しが予算を超過した時点、
/// または採点の戻り値が契約違反〔件数不一致・record_id 不一致〕だった後の
/// 予算確認で使い切っていた時点。issue #84 PR #238 レビュー）で、宣言順に
/// まだ控えていた候補を `iter` から取り出し尽くす。
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
/// 失敗した・scorer の戻り値の件数または record_id 列が一致しなかった
/// 〔REQ-27。issue #84 PR #238 レビュー〕）は探索全体を中断せず、その候補を
/// 該当する分類で記録して次の候補へ進む。
pub fn run_search<R, S, C>(
    runner: &mut R,
    scorer: S,
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

    // 全候補で共有する validation 入力（record_id・byte 入力の組）を
    // 所有データとして 1 回だけ組み立てる（`validate_input` が件数一致を
    // 確認済みのため `zip` で安全に構築できる。P0 指摘対応・REQ-27・
    // REQ-39。issue #84 PR #238 レビュー: scorer へ権威ある入力を明示的に
    // 渡し、scorer 側が独自に保持する別データを使わせない）。所有データに
    // するのは、採点の締め切りを強制するために scorer 呼び出しを専用
    // スレッドへ渡す必要があり（[`call_predict_validation_with_deadline`]
    // 参照）、借用データのまま `'static` を要求するスレッド境界を越えられ
    // ないため（[`OwnedValidationInputRecord`] doc 参照）。`Arc` で包み、
    // 候補ごとに複製せず参照カウントだけ増やす。
    let validation_scorer_records: Arc<Vec<OwnedValidationInputRecord>> = Arc::new(
        input
            .validation_record_ids
            .iter()
            .zip(input.validation_inputs.iter())
            .map(|(&record_id, &input_bytes)| OwnedValidationInputRecord {
                record_id: record_id.to_string(),
                input: input_bytes.to_vec(),
            })
            .collect(),
    );
    // 締め切りを過ぎて scorer を取り残した後は `None` になり、以降の候補は
    // 採点できない（`call_predict_validation_with_deadline` doc・trait doc
    // 「時間上限」参照）。
    let mut scorer_holder: Option<S> = Some(scorer);

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

                // P0 指摘対応（REQ-39・issue #84 PR #238 レビュー）: 学習
                // だけで探索予算全体を使い切っていた場合、採点
                // （`predict_validation`）を呼び出さずに打ち切る。trait doc
                // 「時間上限」の通り呼び出し自体を打ち切れないため、
                // 予算が残っていないと分かっている呼び出しをそもそも行わない
                // ことが唯一の資源上限の守り方になる（呼び出し後の事後検出
                // だけに頼ると、無駄な呼び出し自体は防げない）。
                if elapsed_before_scoring_ms >= budget_ms {
                    entries.push(CandidateSearchEntry {
                        candidate_id: candidate.candidate_id,
                        elapsed_at_start_ms: Some(elapsed_ms),
                        time: Some(run.record().clone()),
                        result: CandidateSearchResult::ScoringSkippedBudgetExhausted,
                        validation_outcomes: None,
                    });
                    drain_remaining_as_not_started(&mut entries, &mut candidates_iter);
                    break;
                }

                // P0 指摘対応（REQ-39・issue #84 PR #238 レビュー）: 探索
                // 予算全体はまだ残っていても、この候補自身の実測時間
                // （`elapsed_ms`）が、この候補へ配分した持ち時間
                // （`time_limit_seconds`）を超えていないかを、採点
                // （`predict_validation`）を呼び出す前に確認する。
                // `run_candidate` はワーカーが `Ok` を返せば実測時間を問わず
                // `TrainOutcome::Ok` を返す（`CandidateTimeStatus::Completed`
                // も持ち時間超過の有無を区別しない）ため、ここで確認せずに
                // 採点・選定へ進めると、持ち時間を超えた成功結果が選定され
                // 得る（指摘本文のシナリオ: 予算 3600 秒を 2 候補へ均等配分
                // し、最初の候補が割当 1800 秒を超えて 2000 秒で成功した
                // 場合）。ちょうど持ち時間に達した時点（`==`）は超過扱いに
                // しない（`>` の厳密不等号）: 単調時計はミリ秒単位で丸まり、
                // 割当時間ぴったりで完了する候補は珍しくない（本モジュール
                // doc「実機での確認手順」の
                // 「持ち時間 + supervisor.py の猶予 5 秒」のとおり、
                // わずかな超過は学習ワーカー側の終了処理に想定内で含まれる）。
                // 探索予算全体の判定（直前の `elapsed_before_scoring_ms >=
                // budget_ms`。ちょうど到達した時点も合格にしない）とは
                // 意図的に異なる規則: あちらは探索予算という共有資源の枯渇を
                // 検出するもので、こちらは候補 1 件へ配分した持ち時間からの
                // 逸脱を検出するものであり、境界の扱いを揃える必要はない。
                let candidate_time_limit_ms =
                    u64::from(run.record().time_limit_seconds()).saturating_mul(1000);
                if run.record().elapsed_ms() > candidate_time_limit_ms {
                    entries.push(CandidateSearchEntry {
                        candidate_id: candidate.candidate_id,
                        elapsed_at_start_ms: Some(elapsed_ms),
                        time: Some(run.record().clone()),
                        result: CandidateSearchResult::TrainingExceededTimeLimit,
                        validation_outcomes: None,
                    });
                    continue;
                }

                let time_limit = Duration::from_millis(remaining_for_scoring_ms);
                // P0 指摘対応（REQ-39。issue #84 PR #238 レビュー）: scorer
                // の所有権を取り出し、締め切り付きの専用スレッドで呼び出す
                // （`call_predict_validation_with_deadline` doc・trait doc
                // 「時間上限」参照）。`scorer_holder` が `None` になるのは
                // 直前の候補で締め切りを過ぎて scorer を取り残した場合のみ
                // だが、その場合は必ずその場で残り候補を未着手にして
                // `break` しているため、次の周回に到達した時点では常に
                // `Some` のはずである（理論上到達しない防御的分岐）。
                let current_scorer = scorer_holder.take().ok_or_else(|| SearchError::Internal {
                    detail: "scorer was unavailable after a previous timeout".to_string(),
                })?;
                let (returned_scorer, timed_outcome) = call_predict_validation_with_deadline(
                    current_scorer,
                    candidate.candidate_id.clone(),
                    success.clone(),
                    Arc::clone(&validation_scorer_records),
                    time_limit,
                );
                scorer_holder = returned_scorer;
                let predict_result = match timed_outcome {
                    TimedPredictOutcome::Completed(result) => result,
                    TimedPredictOutcome::TimedOut => {
                        // P0 指摘対応（REQ-39。issue #84 PR #238 レビュー）:
                        // 締め切りを過ぎた戻り値は使わず、既存の探索予算
                        // 超過と同じ「残り候補を未着手にして探索を終える」
                        // 扱いにする（`scorer_holder` はすでに `None` で、
                        // 以降の候補も採点できないため）。
                        entries.push(CandidateSearchEntry {
                            candidate_id: candidate.candidate_id,
                            elapsed_at_start_ms: Some(elapsed_ms),
                            time: Some(run.record().clone()),
                            result: CandidateSearchResult::ScoringTimedOut,
                            validation_outcomes: None,
                        });
                        drain_remaining_as_not_started(&mut entries, &mut candidates_iter);
                        break;
                    }
                };
                match predict_result {
                    Ok(scored_outcomes) => {
                        // P0/P1 指摘対応（REQ-27・評価の独立性。issue #84
                        // PR #238 レビュー）: 戻り値の契約違反（件数不一致・
                        // record_id の不一致）はいずれも候補単位の
                        // `ScoringFailed`（scorer のエラーと同じ扱い）として
                        // 記録し、探索全体は中断せず次候補へ進む。以前は件数
                        // 不一致だけ `SearchError::ScorerOutputMismatch` で
                        // 探索全体を打ち切っており、それまでの候補の記録が
                        // 失われる非対称があった（前回報告の指摘。事前検証
                        // `validate_input` 側の件数不一致〔入力そのものの
                        // 契約違反〕は引き続き致命的エラーのままにする。
                        // ここで扱うのは scorer の実行時の戻り値という別の
                        // 契約層）。件数が一致しない場合は record_id の
                        // `zip` が短い方に切り詰められて `false` になり得る
                        // ため、件数チェックを先に行う。
                        let outcomes_valid = scored_outcomes.len() == input.validation_gold.len()
                            && scored_outcomes
                                .iter()
                                .zip(input.validation_record_ids.iter())
                                .all(|(scored, &expected_id)| scored.record_id == expected_id);
                        if !outcomes_valid {
                            entries.push(CandidateSearchEntry {
                                candidate_id: candidate.candidate_id,
                                elapsed_at_start_ms: Some(elapsed_ms),
                                time: Some(run.record().clone()),
                                result: CandidateSearchResult::ScoringFailed,
                                validation_outcomes: None,
                            });

                            // 下の `Err` 分岐と同じく、呼び出し後の経過時間を
                            // 確認してから次候補へ進む（採点中に予算を使い
                            // 切っていれば残り候補を未着手にして打ち切る）。
                            let elapsed_after_scoring_ms = elapsed_ms_since_start(clock)?;
                            if elapsed_after_scoring_ms >= budget_ms {
                                drain_remaining_as_not_started(&mut entries, &mut candidates_iter);
                                break;
                            }
                            continue;
                        }
                        let outcomes: Vec<Outcome> = scored_outcomes
                            .into_iter()
                            .map(|scored| scored.outcome)
                            .collect();

                        // P1 指摘対応（REQ-39。issue #84 PR #238 レビュー）:
                        // `predict_validation` から戻った直後、`EvalRecord`
                        // の構築（最大 `MAX_SEARCH_OUTCOME_CELLS` 件）・
                        // `evaluate_single_select` の呼び出しより前に探索予算
                        // 全体を確認する。ここで確認せずに評価器まで進めると、
                        // 採点だけで予算を使い切っていても重い評価処理
                        // （最大 1,000 万セル）を最後まで走らせてしまい、
                        // 「期限後も重い処理を続ける」経路が残る（REQ-39
                        // 「資源の上限」）。ここでは正解率をまだ算出していない
                        // ため、呼び出し前に打ち切る既存経路
                        // （[`CandidateSearchResult::ScoringSkippedBudgetExhausted`]）
                        // と同じ「正解率を算出できないまま打ち切る」扱いにする
                        // （評価器を呼ばない点は同じで、採点自体は呼び出し済み
                        // という違いはあるが、公開結果型に新しいバリアントを
                        // 増やさずに済む。JSON 契約の変更が要る場合は
                        // 実装せず承認事項として報告する方針〔delegation-impl〕）。
                        let elapsed_after_predict_ms = elapsed_ms_since_start(clock)?;
                        if elapsed_after_predict_ms >= budget_ms {
                            entries.push(CandidateSearchEntry {
                                candidate_id: candidate.candidate_id,
                                elapsed_at_start_ms: Some(elapsed_ms),
                                time: Some(run.record().clone()),
                                result: CandidateSearchResult::ScoringSkippedBudgetExhausted,
                                validation_outcomes: None,
                            });
                            drain_remaining_as_not_started(&mut entries, &mut candidates_iter);
                            break;
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

                        // 評価器（`evaluate_single_select`）の呼び出しに時間が
                        // かかり、探索予算全体を使い切って
                        // いたら選定対象から除外する（P0 指摘対応。「超過後も
                        // 最後の候補なら Selected を返してしまう」ことを防ぐ。
                        // fail-closed: 正解率自体は参考値として記録するが
                        // `evaluated_owned` へは積まない）。ちょうど予算に
                        // 達した時点（`==`）も「予算到達を合格扱いにしない」
                        // （evaluation-contract）に含めるため `>=` で判定する
                        // （issue #84 PR #238 レビュー・P0 指摘対応）。
                        let elapsed_after_scoring_ms = elapsed_ms_since_start(clock)?;
                        if elapsed_after_scoring_ms >= budget_ms {
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

                        // P1 指摘対応（REQ-18・REQ-39。issue #84 PR #238
                        // レビュー）: 採点が失敗した場合も、成功時に
                        // `predict_validation` から戻った直後へ移した判定
                        // （成功時は `ScoringSkippedBudgetExhausted` を参照）
                        // と同じく呼び出し後の経過時間を確認する。確認せずに
                        // 次候補へ進むと、採点中に探索予算を使い切っていても
                        // 次候補が `run_candidate` に渡ってしまい、予算超過後の
                        // 学習を防げない（採点の成否で「呼び出し後に予算を
                        // 使い切ったか」の扱いを変えない）。
                        let elapsed_after_scoring_ms = elapsed_ms_since_start(clock)?;
                        if elapsed_after_scoring_ms >= budget_ms {
                            drain_remaining_as_not_started(&mut entries, &mut candidates_iter);
                            break;
                        }
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

    /// `validation_gold` と同じ件数の record_id 列を生成する（`"r0"`・`"r1"`
    /// ...）。テスト専用ヘルパーのため `'static` へ leak して返す（P0・
    /// REQ-27 指摘対応。issue #84 PR #238 レビュー）。
    fn make_record_ids(n: usize) -> &'static [&'static str] {
        let ids: Vec<&'static str> = (0..n)
            .map(|i| -> &'static str { Box::leak(format!("r{i}").into_boxed_str()) })
            .collect();
        Box::leak(ids.into_boxed_slice())
    }

    /// `validation_gold` と同じ件数の byte 入力を生成する（本モジュールの
    /// 事前検証・件数検証は入力の中身を見ないため、空スライスの繰り返しで
    /// 十分。P0・REQ-27 指摘対応。issue #84 PR #238 レビュー）。
    fn make_validation_inputs(n: usize) -> &'static [&'static [u8]] {
        let inputs: Vec<&'static [u8]> = (0..n).map(|_| -> &'static [u8] { &[] }).collect();
        Box::leak(inputs.into_boxed_slice())
    }

    /// [`fandhe_edge_data::split::Groupable`] の最小実装（テスト専用）。
    struct TestGroupable {
        id: String,
        group_id: String,
    }

    impl fandhe_edge_data::split::Groupable for TestGroupable {
        fn id(&self) -> &str {
            &self.id
        }
        fn group_id(&self) -> &str {
            &self.group_id
        }
        fn label(&self) -> &str {
            "positive"
        }
    }

    /// [`make_record_ids`] と同じ `["r0", "r1", ...]` を validation split の
    /// record_ids として持つ [`fandhe_edge_data::split_record::SplitRecord`]
    /// を組み立てる（P0 指摘対応・REQ-17・REQ-27。issue #84 PR #238
    /// レビュー）。`validation: 1.0`（train・test は `0.0`）を指定すると、
    /// 比率が厳密に `0.0` の split には一切割り付けない契約
    /// （`fandhe_edge_data::split::alloc_counts` doc）により、全レコードが
    /// validation split に入る。テスト専用ヘルパーのため `'static` へ leak
    /// して返す。
    fn make_split_record(n: usize) -> &'static fandhe_edge_data::split_record::SplitRecord {
        let records: Vec<TestGroupable> = (0..n)
            .map(|i| TestGroupable {
                id: format!("r{i}"),
                group_id: format!("g{i}"),
            })
            .collect();
        let ratios = fandhe_edge_data::split::SplitRatios {
            train: 0.0,
            validation: 1.0,
            test: 0.0,
        };
        let recorded = fandhe_edge_data::split_record::split_and_record(&records, 0, &ratios)
            .expect("valid ratios");
        Box::leak(Box::new(recorded.record().clone()))
    }

    fn base_input<'a>(
        label_order: &'a [&'a str],
        validation_gold: &'a [&'a str],
        candidates: Vec<SearchCandidate>,
    ) -> SearchInput<'a> {
        SearchInput {
            label_order,
            validation_gold,
            validation_record_ids: make_record_ids(validation_gold.len()),
            validation_inputs: make_validation_inputs(validation_gold.len()),
            validation_split_record: make_split_record(validation_gold.len()),
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

    /// REQ-18・TASK-18.1-2・REQ-27（P0 指摘対応。issue #84 PR #238 レビュー）:
    /// `validation_record_ids` の件数が `validation_gold` と一致しないと
    /// `ValidationRecordIdCountMismatch`。
    #[test]
    fn task18_1_2_validate_input_rejects_validation_record_id_count_mismatch() {
        let label_order = ["positive", "negative"];
        let gold = ["positive", "negative"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let input = SearchInput {
            label_order: &label_order,
            validation_gold: &gold,
            validation_record_ids: &["r0"],
            validation_inputs: make_validation_inputs(2),
            validation_split_record: make_split_record(2),
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(
            err,
            SearchError::ValidationRecordIdCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    /// REQ-18・TASK-18.1-2・REQ-27（P0 指摘対応。issue #84 PR #238 レビュー）:
    /// `validation_record_ids` に空文字列が含まれると
    /// `InvalidValidationRecordId`。
    #[test]
    fn task18_1_2_validate_input_rejects_empty_validation_record_id() {
        let label_order = ["positive", "negative"];
        let gold = ["positive", "negative"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let input = SearchInput {
            label_order: &label_order,
            validation_gold: &gold,
            validation_record_ids: &["r0", ""],
            validation_inputs: make_validation_inputs(2),
            validation_split_record: make_split_record(2),
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::InvalidValidationRecordId { index: 1 });
    }

    /// REQ-18・TASK-18.1-2・REQ-27（P0 指摘対応。issue #84 PR #238 レビュー）:
    /// `validation_record_ids` に重複があると `DuplicateValidationRecordId`。
    #[test]
    fn task18_1_2_validate_input_rejects_duplicate_validation_record_id() {
        let label_order = ["positive", "negative"];
        let gold = ["positive", "negative"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let input = SearchInput {
            label_order: &label_order,
            validation_gold: &gold,
            validation_record_ids: &["r0", "r0"],
            validation_inputs: make_validation_inputs(2),
            validation_split_record: make_split_record(2),
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateValidationRecordId { index: 1 });
    }

    /// REQ-18・TASK-18.1-2・REQ-39（P1 指摘対応。issue #84 PR #238 レビュー）:
    /// `validation_record_ids` の 1 件が
    /// `fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES` を超えると
    /// `ValidationRecordIdTooLong`（`BTreeSet` へ追加する前に拒否する）。
    #[test]
    fn task18_1_2_validate_input_rejects_validation_record_id_too_long() {
        let label_order = ["positive", "negative"];
        let gold = ["positive", "negative"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let over_limit_id = "a".repeat(fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES + 1);
        let record_ids = ["r0", over_limit_id.as_str()];
        let input = SearchInput {
            label_order: &label_order,
            validation_gold: &gold,
            validation_record_ids: &record_ids,
            validation_inputs: make_validation_inputs(2),
            validation_split_record: make_split_record(2),
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(
            err,
            SearchError::ValidationRecordIdTooLong {
                index: 1,
                size: fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES + 1,
                limit: fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES,
            }
        );
    }

    /// REQ-18・TASK-18.1-2・REQ-39（P1 指摘対応。issue #84 PR #238 レビュー）:
    /// `validation_record_ids` の合計バイト数が
    /// `MAX_VALIDATION_INPUT_TOTAL_BYTES` を超えると
    /// `ValidationRecordIdTotalBytesExceeded`。個々の要素は
    /// `MAX_INPUT_ID_BYTES` ちょうどに収まっているため、1 件あたりの上限
    /// チェックだけでは検出できず合計チェックが必要なことを示す。
    #[test]
    fn task18_1_2_validate_input_rejects_validation_record_ids_total_bytes_exceeded() {
        let label_order = ["positive", "negative"];
        let per_id_bytes = fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES;
        let n_records = MAX_VALIDATION_INPUT_TOTAL_BYTES / per_id_bytes + 1;
        let gold: Vec<&str> = (0..n_records).map(|_| "positive").collect();
        // 各要素は重複検出（`DuplicateValidationRecordId`）に先に引っかから
        // ないよう、長さ `per_id_bytes` ちょうどのまま一意な値にする
        // （先頭を index の 10 進表現で埋め、残りを `0` で埋める）。
        let record_ids: Vec<String> = (0..n_records)
            .map(|i| format!("{i:0>width$}", width = per_id_bytes))
            .collect();
        let record_id_refs: Vec<&str> = record_ids.iter().map(String::as_str).collect();
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let input = SearchInput {
            label_order: &label_order,
            validation_gold: &gold,
            validation_record_ids: &record_id_refs,
            validation_inputs: make_validation_inputs(n_records),
            validation_split_record: make_split_record(n_records),
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(
            err,
            SearchError::ValidationRecordIdTotalBytesExceeded {
                total: per_id_bytes * n_records,
                limit: MAX_VALIDATION_INPUT_TOTAL_BYTES,
            }
        );
    }

    /// REQ-18・TASK-18.1-2・REQ-27（P0 指摘対応。issue #84 PR #238 レビュー）:
    /// `validation_inputs` の件数が `validation_gold` と一致しないと
    /// `ValidationInputCountMismatch`。
    #[test]
    fn task18_1_2_validate_input_rejects_validation_input_count_mismatch() {
        let label_order = ["positive", "negative"];
        let gold = ["positive", "negative"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let input = SearchInput {
            label_order: &label_order,
            validation_gold: &gold,
            validation_record_ids: make_record_ids(2),
            validation_inputs: make_validation_inputs(1),
            validation_split_record: make_split_record(2),
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(
            err,
            SearchError::ValidationInputCountMismatch {
                expected: 2,
                actual: 1
            }
        );
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

    /// REQ-18・TASK-18.1-2・REQ-39: `out/a` と `out/./a`（`.` 構成要素）・
    /// `out//a`（連続スラッシュ）・`out/a/`（末尾スラッシュ）は正規化後に
    /// 同一の出力先を指すため、正規化前の文字列が異なっていても
    /// `DuplicateOutDir` として検出する（codex review PR #238 P1 指摘の
    /// 回帰テスト）。
    #[test]
    fn task18_1_2_validate_input_rejects_duplicate_out_dir_after_normalization() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params("/root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/root", "out/./a"),
            },
            SearchCandidate {
                candidate_id: "c3-c".to_string(),
                params: valid_params("/root", "out//a"),
            },
            SearchCandidate {
                candidate_id: "c3-d".to_string(),
                params: valid_params("/root", "out/a/"),
            },
        ];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateOutDir { index: 1 });
    }

    /// REQ-18・TASK-18.1-2・REQ-39: `root` と `out_dir` の境界をずらしても
    /// 結合後の出力先が一致すれば `DuplicateOutDir`（
    /// `root="/root", out_dir="out/a"` と `root="/root/out", out_dir="a"` は
    /// いずれも `/root/out/a` を指す。codex review PR #238 P1 指摘の
    /// 回帰テスト）。
    #[test]
    fn task18_1_2_validate_input_rejects_duplicate_out_dir_across_root_boundary() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params("/root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/root/out", "a"),
            },
        ];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateOutDir { index: 1 });
    }

    /// REQ-18・TASK-18.1-2・REQ-39: 一方の出力先が他方の祖先（親ディレクト
    /// リ）にあたる場合も、成果物の混在・上書きが起こり得るため
    /// `DuplicateOutDir` として拒否する（codex review PR #238 P1 指摘）。
    #[test]
    fn task18_1_2_validate_input_rejects_nested_out_dir() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params("/root", "out"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/root", "out/sub"),
            },
        ];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateOutDir { index: 1 });
    }

    /// REQ-18・TASK-18.1-2・REQ-39: `root` に `..` 構成要素が含まれていても
    /// 結合後の出力先を字句上で解決してから重複判定する
    /// （`root="/root/x/.."` は `root="/root"` と同じ。`check_root_syntax`
    /// は `..` を拒否しないため、`out_dir` 側だけでなく `root` 側の `..` も
    /// 考慮する必要がある。codex review PR #238 P1 指摘の対応中に判明した回帰テスト）。
    #[test]
    fn task18_1_2_validate_input_rejects_duplicate_out_dir_with_dotdot_in_root() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params("/root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/root/x/..", "out/a"),
            },
        ];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateOutDir { index: 1 });
    }

    /// REQ-18・TASK-18.1-2・REQ-39: `root` の `..` がルートより上へ遡ろうと
    /// してもスタックが空のまま無視され（クランプ）、`root="/.."` は
    /// `root="/"` と同じ出力先として扱われる（codex review PR #238 P1 指摘の対応中に判明した回帰テスト）。
    #[test]
    fn task18_1_2_validate_input_rejects_duplicate_out_dir_with_dotdot_clamped_at_root() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params("/", "a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/..", "a"),
            },
        ];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateOutDir { index: 1 });
    }

    /// REQ-18・TASK-18.1-2: 兄弟ディレクトリ（互いの祖先・子孫にならない
    /// 別出力先）は重複として扱わない（`out_dirs_conflict` が偽陽性を出さ
    /// ないことの確認）。
    #[test]
    fn task18_1_2_validate_input_accepts_sibling_out_dirs() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params("/root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/root", "out/b"),
            },
        ];
        let input = base_input(&label_order, &gold, candidates);
        assert!(validate_input::<std::convert::Infallible>(&input).is_ok());
    }

    /// REQ-18・TASK-18.1-2・REQ-39（P1 指摘対応。issue #84 PR #238
    /// レビュー）: symlink を経由して同じ実ディレクトリを指す 2 候補は
    /// `DuplicateOutDir` になる（`canonicalized_out_dir_key` が symlink を
    /// 解決することの確認。`cfg(unix)`: symlink 作成に
    /// `std::os::unix::fs::symlink` を使うため）。
    #[cfg(unix)]
    #[test]
    fn task18_1_2_validate_input_rejects_duplicate_out_dir_via_symlink() {
        let base = std::env::temp_dir().join(format!(
            "fandhe-edge-train-test-symlink-{}-{}",
            std::process::id(),
            "task18_1_2_validate_input_rejects_duplicate_out_dir_via_symlink"
        ));
        let real_dir = base.join("real");
        let link_path = base.join("link");
        std::fs::create_dir_all(&real_dir).expect("create real dir");
        std::os::unix::fs::symlink(&real_dir, &link_path).expect("create symlink");

        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params(real_dir.to_str().expect("utf-8 path"), "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params(link_path.to_str().expect("utf-8 path"), "out/a"),
            },
        ];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();

        // 後片付け（テスト失敗時も best-effort で削除する）。
        let _ = std::fs::remove_dir_all(&base);

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

    /// REQ-18・TASK-18.1-2・REQ-39（P0 指摘対応。issue #84 PR #238 レビュー）:
    /// `label_order` の件数が [`MAX_LABELS`] を 1 件超えると、`BTreeSet` を
    /// 組み立てる前に `InvalidLabelOrder` として拒否する（巨大な
    /// `label_order` を渡されても、後段の `TrainRequest::new` による上限
    /// 検証を待たずに検証前の時間・メモリ消費を避ける）。
    #[test]
    fn task18_1_2_validate_input_rejects_label_order_count_over_limit() {
        let gold = ["positive"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let label_order: Vec<String> = (0..=MAX_LABELS).map(|i| format!("l{i}")).collect();
        let label_order_refs: Vec<&str> = label_order.iter().map(String::as_str).collect();
        let input = base_input(&label_order_refs, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::InvalidLabelOrder);
    }

    /// REQ-18・TASK-18.1-2・REQ-39（P0 指摘対応。issue #84 PR #238 レビュー）:
    /// `label_order` の 1 要素が [`MAX_LABEL_BYTES`] を超えると
    /// `InvalidLabelOrder`。
    #[test]
    fn task18_1_2_validate_input_rejects_label_order_element_over_byte_limit() {
        let gold = ["positive"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params("/root", "out/a"),
        }];
        let long_label = "a".repeat(MAX_LABEL_BYTES + 1);
        let label_order = vec![long_label.as_str(), "negative"];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::InvalidLabelOrder);
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
