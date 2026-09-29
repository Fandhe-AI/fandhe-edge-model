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
//! 子プロセスとして起動する実行器は [`crate::time_allotment::CandidateRunner`]
//! で、実装は [`crate::process::WorkerCandidateRunner`]（`run_train` を使う）。
//!
//! # validation の採点は学習ジョブの中で行う（選択肢 2）
//!
//! 候補の validation 予測は、その候補の学習ジョブ（子プロセス）の**内側**で
//! 学習直後に行う（issue #84 PR #238 レビュー・オーナー承認の選択肢 2）。
//! 学習済みモデルはそのプロセスのメモリ上にしか無く、成果物（ONNX）から
//! 読み戻す経路は学習ワーカーに存在しないため、別プロセス・別スレッドでの
//! 採点は成立しない。各候補の学習リクエストへ validation 入力
//! （[`crate::request::ValidationInput`]。`{id, input}` のみ）を付け、結果の
//! [`crate::result::SuccessOutcome::validation_predictions`] を
//! [`SearchInput::validation_gold`] と突き合わせて正解率を算出する。
//!
//! - 予測時間は学習と同じ持ち時間・壁時計の締め切りに含まれる。締め切り超過は
//!   実行器が強制終了する（[`crate::time_allotment::CandidateRunner::is_wall_timeout`]）
//!   ため、本モジュールはスレッドも `recv_timeout` も持たない。1 候補の時間切れは
//!   その候補だけの記録（[`CandidateSearchResult::TrainingTimedOut`]）で、探索
//!   予算が残っていれば次の候補へ進む（予算切れのときだけ残りを未着手にする）
//! - 予測経路を持たない `kind` は学習ワーカーが学習前に拒否する
//!   （`invalid_request`。候補は [`CandidateSearchResult::TrainingNotCompleted`]）
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
//!   予定だが、本モジュールは呼び出し元の責務として doc で明示するに留める）。
//!   `validation_record_ids` は凍結済み validation split の記録と照合する
//!   （[`SearchError::ValidationSplitHashMismatch`]。学習を始める前に検査する）
//! - 学習ワーカーへ渡すのは validation の `id` と `input` だけで、
//!   `validation_gold`（正解ラベル）は渡さない（REQ-27「推論関数には `input`
//!   だけを渡す」。[`crate::request::ValidationInput`] は gold を持てない型）。
//!   予測列の `id` 列・件数が [`SearchInput::validation_record_ids`] と
//!   （順序を含めて）一致しない場合は、その候補を
//!   [`CandidateSearchResult::ScoringFailed`] として選定対象外にする
//! - 予測ラベル・入力本文は `Debug` にも記録の JSON にも出さない
//!   （security.md「データ本文を転記しない」）
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
//! - 推論ランタイム（REQ-28 系）を使った採点（本モジュールの採点は学習ワーカー
//!   内の学習直後の予測。ONNX 書き出しモデルとの一致は各 kind のテストで確認する）
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

use fandhe_edge_eval::metrics::{self, EvalError, EvalRecord, Outcome, Ratio};
use fandhe_edge_eval::significance::MAX_EVAL_RECORDS;

use crate::error::TrainRequestError;
use crate::limits::{MAX_LABEL_BYTES, MAX_LABELS, MAX_VALIDATION_INPUT_TOTAL_BYTES};
use crate::request::{TrainRequest, TrainRequestParams, ValidationInput, check_config_size};
use crate::result::{TrainOutcome, ValidationPrediction, ValidationPredictionStatus};
use crate::time_allotment::{
    Allotment, CandidateRunner, CandidateTimeError, CandidateTimeRecord, CandidateTimeStatus,
    Clock, PerCandidatePolicy, allot, run_candidate_with_validation_inputs,
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
    /// 各候補の学習リクエストへ [`validation_inputs`](Self::validation_inputs)
    /// と組（[`ValidationInput`]）にして渡し、結果の予測列の `id` 列と
    /// （順序を含めて）突き合わせることで、学習ワーカーが意図した順序と異なる
    /// 予測（件数は同じだが順序が違う・別の record_id の予測）を返していない
    /// かを検証する（REQ-27「評価の独立性」・P0 指摘対応。issue #84 PR #238
    /// レビュー）。正解ラベルは含まない。`validate_input`（`crate::search`）が
    /// `BTreeSet` へ追加する前に、1 件あたり
    /// [`fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES`]・合計
    /// [`crate::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES`] を超えていないか
    /// 検証する（P1 指摘対応・REQ-39「資源の上限」。issue #84 PR #238
    /// レビュー）。
    pub validation_record_ids: &'a [&'a str],
    /// [`validation_record_ids`](Self::validation_record_ids) と同じ順・
    /// 同じ件数の入力（README「入力表現は byte のみ」）。各候補の学習リクエストの
    /// `validation_inputs`（[`ValidationInput`]。`id` と `input` のみ。正解
    /// ラベルは渡さない。REQ-27）として学習ワーカーへ渡され、学習直後に
    /// 予測される。**UTF-8 でなければならない**（リクエスト JSON の文字列に
    /// 載せるため。UTF-8 でない入力は、子プロセスを起動する前に
    /// [`SearchError::ValidationInputNotUtf8`]〔`invalid_input`〕で拒否する）。
    /// `validate_input`（`crate::search`）が学習を始める前に、1 件あたり
    /// [`fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES`]・合計
    /// [`crate::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES`] を超えていないか、
    /// および各候補のリクエストに収まる大きさか（実効上限は
    /// [`crate::limits::MAX_REQUEST_BYTES`]。同定数の doc 参照）を検証する
    /// （`SearchInput` はデータ契約層を経由しない呼び出し元も直接組み立てられる
    /// 公開 API のため。REQ-39「資源の上限」・P0 指摘対応。issue #84 PR #238
    /// レビュー）。
    pub validation_inputs: &'a [&'a [u8]],
    /// 凍結済み validation split の記録（REQ-17・REQ-27。P0 指摘対応。
    /// issue #84 PR #238 レビュー）。`validate_input` は
    /// [`validation_record_ids`](Self::validation_record_ids) を昇順に並べ
    /// 直した上で、[`crate::split_record`] と同じ正準化ハッシュ規則
    /// （[`fandhe_edge_core::canonical::canonical_sha256_hex`]。新しい規則は
    /// 作らず、`crates/data::split_record` が使うのと同じ関数を再利用する）
    /// で再計算し、この記録の `validation` split のハッシュ
    /// （[`fandhe_edge_data::split_record::SplitRecord::digest`]・
    /// `fandhe_edge_data::split::Split::Validation`）と一致するかを、**学習を
    /// 始める前**に確認する（採点は学習ジョブの中で行うため、この検査が
    /// ジョブ全体の前提になる）。一致しなければ
    /// [`SearchError::ValidationSplitHashMismatch`] で fail-closed に停止する。
    /// 凍結した最終 test 分割の record_ids を誤って validation として渡した
    /// 場合も、その記録の `validation` split のハッシュとは一致しないため
    /// 同じ経路で拒否される（`SplitRecord` は train・validation・test の
    /// 3 split をまとめて保持する 1 つの記録であり、常に `Split::Validation`
    /// の digest だけと比較することで「どの split の記録として渡されたか」を
    /// 暗黙に検証する。分割ごとに別の "kind" フィールドを持たないため、追加の
    /// 種別照合は不要）。
    pub validation_split_record: &'a fandhe_edge_data::split_record::SplitRecord,
    /// 探索対象の候補（宣言順に実行する。乱数は使わない）。
    pub candidates: Vec<SearchCandidate>,
    /// 探索予算全体（秒）。
    pub budget: SearchBudget,
    /// 候補 1 件あたりの持ち時間の決め方（[`crate::time_allotment::allot`]
    /// へそのまま渡す）。
    pub policy: PerCandidatePolicy,
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
    /// 学習ジョブは成功したが、採点（学習直後の validation 予測列）が無効
    /// だった。次の 2 通りをまとめて表す:
    /// (1) 成功結果に予測列が無い（リクエストに `validation_inputs` を付けて
    /// いるため通常到達しないが、結果を信頼せず fail-closed に倒す）。
    /// (2) 予測列の件数、または `id` 列が `run_search` の渡した validation
    /// レコードと（順序を含めて）一致しなかった（P0/P1 指摘対応・REQ-27
    /// 「評価の独立性」。issue #84 PR #238 レビュー: 件数だけを照合すると、
    /// 件数が同じ別データ・順序違いの予測でも正解率を算出できてしまう。
    /// 探索全体を打ち切る `SearchError` にするとそれまでの候補の記録を失う
    /// 非対称が生じるため、候補単位の失敗に統一した）。
    ///
    /// 事前検証（`validate_input`）が検出する入力側の件数不一致
    /// （`ValidationRecordIdCountMismatch`・`ValidationInputCountMismatch`）
    /// とは別の契約層（ワーカーの実行時の戻り値）であり、そちらは引き続き
    /// 致命的な `SearchError` のままにする。
    ScoringFailed,
    /// 学習・予測・正解率算出まで完了したが、評価器
    /// （`evaluate_single_select`）の呼び出しに時間がかかり探索予算全体を
    /// 使い切った（P0/P1 指摘対応。issue #84 PR #238 レビュー）。正解率は
    /// 算出できているが、探索予算を超過した後の結果を選定に使うと「合格・選定
    /// 扱いにしてはならない」という REQ-39 の資源上限に反するため、
    /// [`select_best`] の対象から除外する（選定対象外だが正解率自体は記録として
    /// 残す）。評価器の呼び出し前に超過が確定していた場合は評価器を呼ばずに
    /// 正解率も算出しないため、代わりに
    /// [`ScoringSkippedBudgetExhausted`](Self::ScoringSkippedBudgetExhausted)
    /// になる。
    ScoringExceededBudget {
        /// 参考値としての validation 正解率（選定には使わない）。
        validation_accuracy: ValidationAccuracy,
    },
    /// 学習ジョブ（学習＋学習直後の validation 予測）が終わった時点で探索予算
    /// 全体を使い切っていたため、正解率算出（`EvalRecord` の構築・
    /// `evaluate_single_select` の呼び出し。最大 `MAX_SEARCH_OUTCOME_CELLS`
    /// 件）を行わなかった（P1 指摘対応・REQ-39。評価器の呼び出しは資源を
    /// 要するため、予算超過が確定した時点でスキップし「期限後も重い処理を
    /// 続ける」経路を作らない）。
    ///
    /// 正解率を算出していない（できない）ため `0` 等の値で埋めない
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
    /// 扱いにしない」契約に反するため、正解率の算出（評価器の呼び出し）の前に
    /// 除外する。正解率は算出していないため `0` 等の値で
    /// 埋めない（[`ScoringSkippedBudgetExhausted`](Self::ScoringSkippedBudgetExhausted)
    /// と同じ方針）。選定対象外。探索全体の予算はまだ残っている可能性が
    /// あるため、この候補だけを除外して次候補へ進む（探索全体を打ち切る
    /// `NotStarted`・`ScoringSkippedBudgetExhausted` とは異なり、以降の
    /// `while` ループは継続する）。
    TrainingExceededTimeLimit,
    /// 学習ジョブ（学習＋学習直後の validation 予測。予測時間も同じ持ち時間の
    /// 中）が、その候補の持ち時間を使い切って打ち切られた（P0 指摘対応・
    /// REQ-39。issue #84 PR #238・選択肢 2）。次の 2 通りをまとめて表す:
    ///
    /// 1. 実行器が壁時計の締め切りで子プロセスを強制終了した
    ///    （[`CandidateRunner::is_wall_timeout`]。`time` は `None`）
    /// 2. 学習ワーカー自身が `limit_exceeded` を報告し、Rust 側で測った経過時間が
    ///    持ち時間に達していた
    ///    （[`CandidateTimeStatus::LimitExceeded`] の `elapsed_reached_time_limit`
    ///    が `true`。学習後・予測の最初の資源検査で持ち時間超過が検出された
    ///    場合を含む。RSS 等の他の資源上限は、経過時間が持ち時間に達して
    ///    いなければこちらではなく [`TrainingNotCompleted`](Self::TrainingNotCompleted)）
    ///
    /// 正解率は算出していない。選定対象外。**候補単位の時間切れであり、探索全体の
    /// 予算切れではない**: 探索予算が残っていれば次の候補へ進み、予算を使い
    /// 切ったときだけ、残りの候補が [`NotStarted`](Self::NotStarted)
    /// （`BudgetExhausted`）になる（P1 指摘対応。issue #84 PR #238 レビュー）。
    TrainingTimedOut,
    /// 探索予算全体が尽きたため実行しなかった。
    NotStarted {
        /// 未着手の理由。
        reason: NotStartedReason,
    },
}

/// 候補 1 件の探索記録（時刻・打ち切り分類・探索結果の組）。
///
/// `Debug` は手書きする（下記）。`validation_outcomes`（[`Outcome`] の列。
/// `Outcome::Label` は学習ジョブが返す予測ラベル文字列を保持する）を
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
    /// `validation_record_ids` に重複した要素が含まれる（学習ジョブの
    /// 予測列を順序・`id` で一意に対応づけられなくなるため拒否する）。
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
    /// 場合もこの経路で拒否される。学習を始める前に fail-closed で停止する）。
    ValidationSplitHashMismatch,
    /// `validation_record_ids`・`validation_inputs`・`validation_gold`（正解
    /// ラベル）から再計算した**中身のハッシュ**が、
    /// [`SearchInput::validation_split_record`] の `validation` split の中身の
    /// ハッシュ（[`fandhe_edge_data::split_record::SplitDigest::content_sha256`]）と
    /// 一致しない、または記録が中身のハッシュを持たない（REQ-17・REQ-27・P0
    /// 指摘対応。issue #84 PR #238 レビュー）。ID を変えずに `input` だけ、または
    /// 正解ラベルだけを差し替えた入力もここで拒否する。学習を始める前に
    /// fail-closed で停止する。
    ValidationContentHashMismatch,
    /// `validation_inputs[index]` が UTF-8 として読めない（REQ-27・REQ-39。
    /// 学習リクエスト JSON の文字列に載せられないため、子プロセスを起動する
    /// 前に `invalid_input` として拒否する。issue #84 PR #238・選択肢 2）。
    ValidationInputNotUtf8 { index: usize },
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
            SearchError::ValidationInputNotUtf8 { index } => {
                write!(f, "validation input at index {index} is not valid utf-8")
            }
            SearchError::ValidationContentHashMismatch => write!(
                f,
                "validation contents (ids, inputs, gold labels) do not match the frozen validation split record"
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

impl<E> SearchError<E> {
    /// REQ-21 の終了コードへの対応づけ（issue #84 PR #238・選択肢 2。
    /// UTF-8 でない validation 入力を子プロセス起動前に `invalid_input`〔64〕で
    /// 拒否する契約を、型で実装する）。
    ///
    /// - 資源上限（件数・バイト数）の超過は `LimitExceeded`（20）
    /// - それ以外の入力検証の違反（空・重複・不一致・UTF-8 でない・凍結済み
    ///   validation split とのハッシュ不一致等）は `InvalidInput`（64）
    /// - 学習リクエストの検証エラー・実行の失敗は、内側のエラーの対応づけに従う
    ///   （実行器のエラー型 `E` は終了コードを持たないため `RuntimeError`）
    /// - 時計・持ち時間配分・評価器・内部矛盾は `RuntimeError`（70）
    #[must_use]
    pub fn exit_code(&self) -> fandhe_edge_core::exitcode::ExitCode {
        use fandhe_edge_core::exitcode::ExitCode;
        match self {
            SearchError::TooManyCandidates { .. }
            | SearchError::TooManyValidationRecords { .. }
            | SearchError::TooManyOutcomeCells
            | SearchError::ValidationRecordIdTooLong { .. }
            | SearchError::ValidationRecordIdTotalBytesExceeded { .. }
            | SearchError::ValidationInputTooLarge { .. }
            | SearchError::ValidationInputTotalBytesExceeded { .. } => ExitCode::LimitExceeded,
            SearchError::EmptyCandidates
            | SearchError::EmptyValidation
            | SearchError::InvalidLabelOrder
            | SearchError::UnknownValidationGold { .. }
            | SearchError::ValidationRecordIdCountMismatch { .. }
            | SearchError::InvalidValidationRecordId { .. }
            | SearchError::DuplicateValidationRecordId { .. }
            | SearchError::ValidationInputCountMismatch { .. }
            | SearchError::ValidationSplitHashMismatch
            | SearchError::ValidationContentHashMismatch
            | SearchError::ValidationInputNotUtf8 { .. }
            | SearchError::InvalidCandidateId { .. }
            | SearchError::DuplicateCandidateId { .. }
            | SearchError::LabelOrderMismatch { .. }
            | SearchError::DuplicateOutDir { .. } => ExitCode::InvalidInput,
            SearchError::InvalidRequest { source, .. } => source.exit_code(),
            SearchError::Candidate { source, .. } => match source {
                CandidateTimeError::Request(e) => e.exit_code(),
                _ => ExitCode::RuntimeError,
            },
            SearchError::OutDirCanonicalizeFailed { .. }
            | SearchError::Allotment(_)
            | SearchError::Clock(_)
            | SearchError::Eval(_)
            | SearchError::Internal { .. } => ExitCode::RuntimeError,
        }
    }

    /// [`exit_code`](Self::exit_code) に対応する機械可読コード
    /// （`limit_exceeded`・`invalid_input`・`runtime_error`）。
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        use fandhe_edge_core::exitcode::ExitCode;
        match self.exit_code() {
            ExitCode::LimitExceeded => "limit_exceeded",
            ExitCode::InvalidInput => "invalid_input",
            _ => "runtime_error",
        }
    }
}

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
/// はそのまま残す（`..` の解決は、symlink を実体へ解決した後で
/// [`canonicalized_out_dir_key`] が行う。[`joined_components`] 参照）。
fn normalized_path_components(value: &str) -> Vec<&str> {
    value
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect()
}

/// `root` と `out_dir` を結合した出力先の構成要素列を返す。空要素・`.` だけを
/// 取り除き、**`..` は畳まずそのまま残す**（P1 指摘対応。issue #84 PR #238
/// レビュー〔Cursor〕: `..` を文字列の上で先に畳むと、途中の構成要素が symlink
/// のとき実体と合わなくなる。例: `link2 -> real/sub` のとき、`/t/link2/..` の
/// 実体は `/t/real` だが、字句上で畳むと `/t` になる）。`..` の解決は
/// [`canonicalized_out_dir_key`] が、存在する接頭辞を `canonicalize` した後で
/// （実体に対して正しい順序で）行う。
///
/// `root`・`out_dir` とも `..` 構成要素は [`TrainRequest::new`] が拒否済み
/// （`root` は #256）。本関数と [`canonicalized_out_dir_key`] の `..` 処理は
/// 多層防御として残す。呼び出し元（[`validate_input`]）は本関数を呼ぶ前に
/// 必ず [`TrainRequest::new`] を通す。
fn joined_components<'a>(root: &'a str, out_dir: &'a str) -> Vec<&'a str> {
    normalized_path_components(root)
        .into_iter()
        .chain(normalized_path_components(out_dir))
        .collect()
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
/// [`joined_components`] による字句上の正規化（`.`・空要素の除去）だけでは、
/// symlink を経由して同じ実ディレクトリを指す
/// 2 つの候補（例: `root="/tmp/link", out_dir="x"` と
/// `root="/tmp/real", out_dir="x"`。`/tmp/link` が `/tmp/real` への
/// symlink）を見分けられない。本関数は結合後のパスのうち、存在する
/// 最も深い祖先ディレクトリを [`std::fs::canonicalize`] で実体パスへ
/// 解決し、まだ存在しない残りの構成要素をそのまま連結することで、
/// symlink 越しの重複も検出できるようにする。祖先を遡る過程は「存在しない」
/// （`NotFound`）ことが確定している間だけ続け、`root` は絶対パスのため
/// 最終的に必ず `/`（ファイルシステムのルート。通常必ず存在する）で終端
/// する。`..` は畳まずに接頭辞ごと `canonicalize` へ渡し（実体に対して正しい
/// 順序で解決される）、存在しない残りの要素に `..` があれば拒否する
/// （[`joined_components`]・関数内コメント参照）。`NotFound` 以外の理由（権限不足等）での失敗は、より浅い祖先へ
/// 読み替えずに即座に拒否する（P1 指摘対応・issue #84 PR #238 レビュー。
/// 関数 doc「# Errors」参照）。
///
/// # Errors
///
/// `canonicalize` が `NotFound`（対象が存在しないことが確定している）以外の
/// 理由で失敗した場合、その `io::Error` を fail-closed でそのまま返す
/// （P1 指摘対応・issue #84 PR #238 レビュー: 「存在しない」ことが確かな
/// 場合だけより浅い祖先を試し、`PermissionDenied` 等それ以外のエラーは
/// 権限不足で symlink かどうか判定できない可能性があるため、黙ってより
/// 浅い祖先へ読み替えず拒否する）。`NotFound` が続いた場合、最終的に
/// `/`（ファイルシステムのルート）を試す。`/` は通常の環境では必ず存在し
/// 読み取り可能なため、`/` 自体の `canonicalize` が失敗する経路は理論上
/// 到達しない防御的分岐とする。
fn canonicalized_out_dir_key(root: &str, out_dir: &str) -> std::io::Result<Vec<String>> {
    let components = joined_components(root, out_dir);
    for existing_len in (0..=components.len()).rev() {
        let mut candidate = PathBuf::from("/");
        for part in components.iter().take(existing_len) {
            candidate.push(part);
        }
        match std::fs::canonicalize(&candidate) {
            Ok(canonical) => {
                // 存在する接頭辞（`..` を含んでも `canonicalize` が実体に対して
                // 正しい順序で解決する）の後ろは、まだ存在しない要素。そこに
                // `..` があると、文字列の上で畳んでも実体と合う保証が無い
                // （存在しない要素を経由した `..` の行き先は、後から作られる
                // ものによって変わる）ため、拒否する（P1 指摘対応）。
                let rest: Vec<&str> = components.iter().skip(existing_len).copied().collect();
                if rest.contains(&"..") {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "parent directory reference after the deepest existing ancestor",
                    ));
                }
                let mut resolved = canonical;
                for part in rest {
                    resolved.push(part);
                }
                return Ok(path_components_to_strings(&resolved));
            }
            // 「存在しない」ことが確定している場合だけ、より浅い祖先で
            // 再試行する（P1 指摘対応・issue #84 PR #238 レビュー）。
            // `PermissionDenied` 等それ以外の理由による失敗は、対象が
            // 実際には存在するのに中身を確認できない可能性がある
            // （symlink かどうかも含めて判定できない）ため、より浅い祖先へ
            // 黙って読み替えず fail-closed で拒否する（下の
            // `Err(e) => return Err(e)`）。`existing_len == 0` は
            // ファイルシステムのルート `/` で、通常はどの環境でも読み取り
            // 可能なため、実務上この分岐（`NotFound` での再試行）は
            // `existing_len > 0` の間だけで完結する想定。
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && existing_len > 0 => continue,
            Err(e) => return Err(e),
        }
    }
    // 理論上到達しない防御的分岐: 上のループは `existing_len == 0`
    // （`/`）まで必ず 1 度は試行し、その時点で `Ok` か `Err` のいずれかを
    // 返すため、ループを抜けて本行へ到達することはない。
    std::fs::canonicalize("/").map(|p| path_components_to_strings(&p))
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
fn validate_input<E>(input: &SearchInput<'_>) -> Result<Vec<ValidationInput>, SearchError<E>> {
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
    // 直接値を渡せるため、学習を始める
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

    // `validation_inputs` を UTF-8 の `String` へ変換する（REQ-27・REQ-39。
    // 選択肢 2: 学習ワーカーへはリクエスト JSON の文字列として渡すため、UTF-8
    // でない入力は子プロセスを起動する前に拒否する）。`validation_record_ids`
    // と組にした [`ValidationInput`]（`id` と `input` だけ。正解ラベルを持てない
    // 型）を作り、全候補のリクエストへ共有する。件数・各要素の長さ・合計は
    // 直前までに検証済み。
    let mut validation_request_inputs: Vec<ValidationInput> =
        Vec::with_capacity(input.validation_inputs.len());
    for (index, (&record_id, &input_bytes)) in input
        .validation_record_ids
        .iter()
        .zip(input.validation_inputs.iter())
        .enumerate()
    {
        let text = std::str::from_utf8(input_bytes)
            .map_err(|_| SearchError::ValidationInputNotUtf8 { index })?;
        validation_request_inputs.push(ValidationInput::new(
            record_id.to_string(),
            text.to_string(),
        ));
    }

    // `validation_record_ids` が凍結済み validation split の記録と一致する
    // ことを、学習を始める前に確認する（P0 指摘対応・REQ-17・
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

    // 中身（`id`・`input`・正解ラベル）のハッシュも、凍結記録の validation split と
    // 照合する（P0 指摘対応・REQ-17・REQ-27。issue #84 PR #238 レビュー）。ID
    // ハッシュだけでは、ID を変えずに `input` や正解ラベルだけを差し替えた
    // 入力を検出できない。計算は分割を作る側（`split_and_record`）と同じ
    // `fandhe_edge_data::split_record::content_sha256_hex` の 1 関数で、ここでも
    // 新しい規則は作らない。中身のハッシュを持たない古い記録（`None`）は、
    // 中身を保証できないため拒否する（fail-closed）。
    let content_records: Vec<fandhe_edge_data::split_record::ContentRecord<'_>> = input
        .validation_record_ids
        .iter()
        .zip(input.validation_inputs.iter())
        .zip(input.validation_gold.iter())
        .map(
            |((&id, &input_bytes), &gold)| fandhe_edge_data::split_record::ContentRecord {
                id,
                input: input_bytes,
                label: gold,
            },
        )
        .collect();
    let recomputed_content_hash =
        fandhe_edge_data::split_record::content_sha256_hex(&content_records)
            // 理論上到達しない防御的分岐（ID ハッシュの計算と同じ理由）。
            .map_err(|_| SearchError::Internal {
                detail: "failed to canonicalize validation contents for hashing".to_string(),
            })?;
    if expected_digest.content_sha256() != Some(recomputed_content_hash.as_str()) {
        return Err(SearchError::ValidationContentHashMismatch);
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
        // 学習ジョブ内採点用の validation 入力を付けた形でも検証する
        // （選択肢 2）。件数・長さ・重複は上で検証済みだが、リクエスト全体の
        // 大きさ（実効上限は `MAX_REQUEST_BYTES`）は候補の `config` 等との
        // 合計で決まるため、実際に JSON へ直列化して確認する（学習を始めてから
        // 大きさ超過に気づくことを避ける。fail-closed）。
        // 巨大な `config` は複製の前に拒否する（REQ-39・issue #255）。下の
        // `to_json_vec` は `config` と validation 入力の合計の検査として残す。
        check_config_size(&candidate.params.config)
            .map_err(|source| SearchError::InvalidRequest { index, source })?;
        let candidate_request = TrainRequest::new(candidate.params.clone())
            .and_then(|request| request.with_validation_inputs(validation_request_inputs.clone()))
            .map_err(|source| SearchError::InvalidRequest { index, source })?;
        candidate_request
            .to_json_vec()
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

    Ok(validation_request_inputs)
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
/// [`run_search`] が探索予算全体を使い切ったと判断した時点（[`Allotment::Exhausted`]、
/// 学習ジョブ〔学習＋予測〕が終わった時点、または評価器の呼び出し後に予算を
/// 使い切っていた場合。候補 1 件の時間切れ
/// 〔[`CandidateSearchResult::TrainingTimedOut`]〕は予算切れではないため含めない。
/// issue #84 PR #238 レビュー）で、宣言順にまだ控えていた候補を `iter` から取り出し尽くす。
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

/// 学習ジョブの成功結果の予測列（[`ValidationPrediction`]）を、
/// [`SearchInput::validation_record_ids`] と突き合わせて [`Outcome`] の列へ
/// 変換する（REQ-27）。件数、または `id` 列が（順序を含めて）一致しない場合は
/// `None`（呼び出し元が [`CandidateSearchResult::ScoringFailed`] にする）。
/// `Ok` は予測ラベル、`Abstain`・`Error` はそれぞれ対応する [`Outcome`]。
fn outcomes_from_predictions(
    predictions: &[ValidationPrediction],
    record_ids: &[&str],
) -> Option<Vec<Outcome>> {
    if predictions.len() != record_ids.len() {
        return None;
    }
    let mut outcomes = Vec::with_capacity(predictions.len());
    for (prediction, &expected_id) in predictions.iter().zip(record_ids.iter()) {
        if prediction.id() != expected_id {
            return None;
        }
        let outcome = match (prediction.status(), prediction.predicted_label()) {
            (ValidationPredictionStatus::Ok, Some(label)) => Outcome::Label(label.to_string()),
            (ValidationPredictionStatus::Abstain, _) => Outcome::Abstain,
            // `Ok` なのにラベルが無い組み合わせは、結果の解析
            // （`crate::result`）が拒否済みで通常到達しない。予測を信用せず
            // `Error` として数える。
            (ValidationPredictionStatus::Error | ValidationPredictionStatus::Ok, _) => {
                Outcome::Error
            }
        };
        outcomes.push(outcome);
    }
    Some(outcomes)
}

/// 探索予算全体を管理し、複数候補を学習・比較し、選定結果を記録する
/// （TASK-18.1-2・issue #84）。
///
/// 手順: (1) 予算・runner を消費する前にすべての事前検証を行う
/// （[`validate_input`]。凍結済み validation split とのハッシュ照合を含む） →
/// (2) 候補を宣言順に実行し、探索予算の消費を追跡する。各候補の学習リクエストへ
/// validation 入力を付け、学習ジョブが学習直後に予測して返す
/// （モジュール doc「validation の採点は学習ジョブの中で行う」） →
/// (3) 予測列を validation gold と突き合わせ、評価器で正解率を算出する →
/// (4) 評価済みの候補から最高正解率の候補を選ぶ（[`select_best`]）。
///
/// # Errors
///
/// 事前検証・候補の実行・評価器のいずれかが失敗した場合に
/// [`SearchError`] を返す。候補単位の失敗（学習が完了しなかった・持ち時間を
/// 使い切った・予測列の件数または `id` 列が一致しなかった
/// 〔REQ-27。issue #84 PR #238 レビュー〕）は探索全体を中断せず、その候補を
/// 該当する分類で記録する（持ち時間切れも候補単位の記録で、探索予算が残っていれば次の候補へ進む）。
pub fn run_search<R, C>(
    runner: &mut R,
    clock: &C,
    input: SearchInput<'_>,
) -> Result<SearchRecord, SearchError<R::Error>>
where
    R: CandidateRunner,
    C: Clock,
{
    let validation_request_inputs = validate_input(&input)?;

    let budget_seconds = input.budget.get();
    let budget_ms = budget_seconds.saturating_mul(1000);
    let started_mono = clock.monotonic();
    let started_at_unix_ms = clock.unix_millis().map_err(SearchError::Clock)?;

    let mut entries: Vec<CandidateSearchEntry> = Vec::new();
    let mut evaluated_owned: Vec<(String, Ratio)> = Vec::new();
    let n_candidates = input.candidates.len();

    // 探索開始からの単調経過時間（ミリ秒）を求める（複数箇所から呼ぶため
    // 共通化する）。
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

        // 学習ジョブ（学習＋学習直後の validation 予測）を実行する。予測時間も
        // この候補の持ち時間・壁時計の締め切りに含まれる。
        let run = match run_candidate_with_validation_inputs(
            runner,
            clock,
            candidate.params,
            allotted,
            Some(validation_request_inputs.clone()),
        ) {
            Ok(run) => run,
            Err(CandidateTimeError::Runner(error)) if R::is_wall_timeout(&error) => {
                // 実行器が壁時計の締め切りで子プロセスを強制終了した。探索全体の
                // 失敗ではなく候補単位の時間切れとして記録する（REQ-39。実行器の
                // 記録〔`CandidateTimeRecord`〕は作られない）。候補ごとの時間切れと
                // 探索全体の予算切れは分けて扱い（P1 指摘対応。issue #84 PR #238
                // レビュー）、探索予算が残っていれば次の候補へ進む。予算を使い切って
                // いれば、次の周回の `allot` が `Exhausted` を返し、残りの候補を
                // `BudgetExhausted` にする。採点用スレッドはもう無いため、
                // 1 候補の時間切れで探索を止める必要はない。
                entries.push(CandidateSearchEntry {
                    candidate_id: candidate.candidate_id,
                    elapsed_at_start_ms: Some(elapsed_ms),
                    time: None,
                    result: CandidateSearchResult::TrainingTimedOut,
                    validation_outcomes: None,
                });
                continue;
            }
            Err(source) => return Err(SearchError::Candidate { index, source }),
        };

        match run.outcome() {
            TrainOutcome::Ok(success) => {
                // P1 指摘対応（REQ-39・issue #84 PR #238 レビュー）: 学習ジョブが
                // 終わった時点で探索予算全体を使い切っていた場合、評価器
                // （`EvalRecord` の構築・`evaluate_single_select`。最大
                // `MAX_SEARCH_OUTCOME_CELLS` 件）を呼び出さずに打ち切る（「期限後も
                // 重い処理を続ける」経路を作らない）。ちょうど予算に達した時点
                // （`==`）も「予算到達を合格扱いにしない」ため `>=`。
                let elapsed_after_job_ms = elapsed_ms_since_start(clock)?;
                if elapsed_after_job_ms >= budget_ms {
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

                // P0 指摘対応（REQ-39・issue #84 PR #238 レビュー）: 探索予算全体
                // はまだ残っていても、この候補自身の実測時間（`elapsed_ms`）が
                // 配分した持ち時間（`time_limit_seconds`）を超えていないかを、
                // 正解率の算出前に確認する。`run_candidate` はワーカーが `Ok` を
                // 返せば実測時間を問わず `TrainOutcome::Ok` を返す
                // （`CandidateTimeStatus::Completed` も持ち時間超過の有無を区別
                // しない）ため、ここで確認せずに進めると、持ち時間を超えた成功
                // 結果が選定され得る。学習ジョブは予測時間を含むため、予測が
                // 持ち時間を食った場合もここで除外される。ちょうど持ち時間に
                // 達した時点（`==`）は超過扱いにしない（`>` の厳密不等号）:
                // 単調時計はミリ秒単位で丸まり、割当時間ぴったりで完了する候補は
                // 珍しくなく、わずかな超過は学習ワーカーの終了処理に想定内で
                // 含まれる。探索予算全体の判定（上の `>=`）とは意図的に異なる
                // 規則: あちらは共有資源の枯渇、こちらは候補 1 件へ配分した
                // 持ち時間からの逸脱を検出する。
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

                // 予測列を validation gold・record_id と突き合わせる（REQ-27）。
                // 不一致・欠落は候補単位の `ScoringFailed`（探索全体は中断しない）。
                let outcomes = success.validation_predictions().and_then(|predictions| {
                    outcomes_from_predictions(predictions, input.validation_record_ids)
                });
                let Some(outcomes) = outcomes else {
                    entries.push(CandidateSearchEntry {
                        candidate_id: candidate.candidate_id,
                        elapsed_at_start_ms: Some(elapsed_ms),
                        time: Some(run.record().clone()),
                        result: CandidateSearchResult::ScoringFailed,
                        validation_outcomes: None,
                    });
                    continue;
                };

                let eval_records: Vec<EvalRecord<'_>> = input
                    .validation_gold
                    .iter()
                    .zip(outcomes.iter())
                    .map(|(&gold, outcome)| EvalRecord { gold, outcome })
                    .collect();
                let metrics = metrics::evaluate_single_select(input.label_order, &eval_records)
                    .map_err(SearchError::Eval)?;
                let accuracy = metrics.accuracy.overall;

                // 評価器（`evaluate_single_select`）の呼び出しに時間がかかり、
                // 探索予算全体を使い切っていたら選定対象から除外する
                // （P0 指摘対応。「超過後も最後の候補なら Selected を返して
                // しまう」ことを防ぐ。fail-closed: 正解率自体は参考値として
                // 記録するが `evaluated_owned` へは積まない）。ちょうど予算に
                // 達した時点（`==`）も合格にしないため `>=` で判定する
                // （issue #84 PR #238 レビュー・P0 指摘対応）。
                let elapsed_after_eval_ms = elapsed_ms_since_start(clock)?;
                if elapsed_after_eval_ms >= budget_ms {
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
            TrainOutcome::Error(_) => {
                // 学習ワーカー自身が `limit_exceeded` を報告し、Rust 側で測った
                // 経過時間が持ち時間に達していた場合は、実行器が強制終了した
                // 場合（上の `is_wall_timeout`）と同じ候補単位の時間切れとして
                // 扱い、探索予算が残っていれば次の候補へ進む（学習後・予測の最初の資源検査で持ち時間超過が検出された
                // 場合など。REQ-39。issue #84 PR #238・選択肢 2）。RSS 等の他の
                // 資源上限による `limit_exceeded` は、経過時間が持ち時間に達して
                // いなければここに入らず、通常の学習失敗として次候補へ進む。
                let timed_out = matches!(
                    run.record().status(),
                    CandidateTimeStatus::LimitExceeded {
                        elapsed_reached_time_limit: true
                    }
                );
                entries.push(CandidateSearchEntry {
                    candidate_id: candidate.candidate_id,
                    elapsed_at_start_ms: Some(elapsed_ms),
                    time: Some(run.record().clone()),
                    result: if timed_out {
                        CandidateSearchResult::TrainingTimedOut
                    } else {
                        CandidateSearchResult::TrainingNotCompleted
                    },
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
        fn input(&self) -> &[u8] {
            &[]
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
                params: valid_params("/nonexistent-fandhe-root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params("/nonexistent-fandhe-root", "out/b"),
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
        let mut params = valid_params("/nonexistent-fandhe-root", "out/a");
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
                params: valid_params("/nonexistent-fandhe-root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
                params: valid_params("/nonexistent-fandhe-root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/nonexistent-fandhe-root", "out/./a"),
            },
            SearchCandidate {
                candidate_id: "c3-c".to_string(),
                params: valid_params("/nonexistent-fandhe-root", "out//a"),
            },
            SearchCandidate {
                candidate_id: "c3-d".to_string(),
                params: valid_params("/nonexistent-fandhe-root", "out/a/"),
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
                params: valid_params("/nonexistent-fandhe-root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/nonexistent-fandhe-root/out", "a"),
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
                params: valid_params("/nonexistent-fandhe-root", "out"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/nonexistent-fandhe-root", "out/sub"),
            },
        ];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateOutDir { index: 1 });
    }

    /// REQ-39・#256: `..` 構成要素を含む `root` は [`TrainRequest::new`] が先に
    /// 拒否するため、重複判定へ進まず `InvalidRequest`（`root` の `invalid_path`）
    /// になる。
    #[test]
    fn req39_validate_input_rejects_dotdot_in_root() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: valid_params("/nonexistent-fandhe-root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/nonexistent-fandhe-root/x/..", "out/a"),
            },
        ];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(
            err,
            SearchError::InvalidRequest {
                index: 1,
                source: TrainRequestError::InvalidPath { field: "root" },
            }
        );
    }

    /// REQ-18・TASK-18.1-2・REQ-39（P1 指摘対応。issue #84 PR #238 レビュー
    /// 〔Cursor〕）: 存在しない構成要素の後ろに `..` があるパスは
    /// `canonicalized_out_dir_key` が `InvalidInput` で拒否する。`TrainRequest` が
    /// `..` を拒否する（#256）ため、多層防御として helper を直接検証する。
    #[test]
    fn task18_1_2_canonicalized_key_rejects_dotdot_after_nonexistent_component() {
        let err = canonicalized_out_dir_key("/nonexistent-fandhe-root/x/..", "out/a").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    /// REQ-18・TASK-18.1-2・REQ-39（P1 指摘対応。issue #84 PR #238 レビュー
    /// 〔Cursor〕）: 存在する構成要素の `..` は `canonicalize` が実体に対して
    /// 解決するため、`root="/t/real/sub/.."` のキーは `root="/t/real"` と一致する。
    /// `TrainRequest` が `..` を拒否する（#256）ため、多層防御として
    /// `canonicalized_out_dir_key` を直接検証する。
    #[cfg(unix)]
    #[test]
    fn task18_1_2_canonicalized_key_resolves_existing_dotdot() {
        let base = std::env::temp_dir().join(format!(
            "fandhe-edge-train-test-dotdot-{}-existing",
            std::process::id()
        ));
        let sub = base.join("real").join("sub");
        std::fs::create_dir_all(&sub).expect("create dirs");
        let real = base.join("real");
        let real_str = real.to_str().expect("utf-8 path");
        let via_dotdot = format!("{real_str}/sub/..");
        let plain = canonicalized_out_dir_key(real_str, "out/a");
        let dotdot = canonicalized_out_dir_key(&via_dotdot, "out/a");
        let _ = std::fs::remove_dir_all(&base);
        assert_eq!(plain.expect("plain"), dotdot.expect("dotdot"));
    }

    /// REQ-18・TASK-18.1-2・REQ-39（P1 指摘対応。issue #84 PR #238 レビュー
    /// 〔Cursor〕の回帰テスト。`cfg(unix)`）: `link2 -> real/sub` のとき
    /// `root="/t/link2/.."` の実体は `/t/real`（symlink を先に解決してから `..`
    /// を適用する）。`..` を文字列の上で先に畳むと `/t` になる。`TrainRequest` が
    /// `..` を拒否する（#256）ため、多層防御として helper を直接検証する。
    #[cfg(unix)]
    #[test]
    fn task18_1_2_canonicalized_key_resolves_symlink_then_dotdot() {
        let base = std::env::temp_dir().join(format!(
            "fandhe-edge-train-test-symlink-dotdot-{}",
            std::process::id()
        ));
        let real = base.join("real");
        let sub = real.join("sub");
        std::fs::create_dir_all(&sub).expect("create dirs");
        let link2 = base.join("link2");
        std::os::unix::fs::symlink(&sub, &link2).expect("create symlink");
        let real_str = real.to_str().expect("utf-8 path");
        let via = format!("{}/..", link2.to_str().expect("utf-8 path"));
        let plain = canonicalized_out_dir_key(real_str, "out/a");
        let dotdot = canonicalized_out_dir_key(&via, "out/a");
        let _ = std::fs::remove_dir_all(&base);
        assert_eq!(plain.expect("plain"), dotdot.expect("dotdot"));
    }

    /// REQ-18・TASK-18.1-2・REQ-39: ルートより上へ遡る `..` はクランプされ、
    /// `root="/.."` のキーは `root="/"` と一致する。`TrainRequest` が `..` を
    /// 拒否する（#256）ため、多層防御として helper を直接検証する。
    #[test]
    fn task18_1_2_canonicalized_key_clamps_dotdot_at_root() {
        let plain = canonicalized_out_dir_key("/", "a").expect("plain");
        let dotdot = canonicalized_out_dir_key("/..", "a").expect("dotdot");
        assert_eq!(plain, dotdot);
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
                params: valid_params("/nonexistent-fandhe-root", "out/a"),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: valid_params("/nonexistent-fandhe-root", "out/b"),
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

    /// REQ-18・TASK-18.1-2・REQ-39（P1 指摘対応。issue #84 PR #238
    /// レビュー）: `canonicalize` が権限不足（`PermissionDenied`）で失敗する
    /// 祖先は、より浅い祖先へ黙って読み替えず `OutDirCanonicalizeFailed` で
    /// 拒否する（`NotFound` だけを再試行対象にする）。
    ///
    /// root で実行された場合、DAC（パーミッションビット）が無視されるため
    /// 本来期待する `PermissionDenied` が発生しない。その場合はテストを
    /// skip せず、実測した挙動（`NotFound` として解決され検証を通過する）
    /// に期待値を切り替える（コーディネーター指摘: 「root で実行された
    /// 場合は skip ではなく、条件に合わせて期待値を変える」）。
    #[cfg(unix)]
    #[test]
    fn task18_1_2_validate_input_rejects_out_dir_when_canonicalize_denies_permission() {
        let base = std::env::temp_dir().join(format!(
            "fandhe-edge-train-test-permission-{}-{}",
            std::process::id(),
            "task18_1_2_validate_input_rejects_out_dir_when_canonicalize_denies_permission"
        ));
        let restricted_dir = base.join("restricted");
        std::fs::create_dir_all(&restricted_dir).expect("create restricted dir");
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&restricted_dir)
                .expect("stat restricted dir")
                .permissions();
            perms.set_mode(0o000);
            std::fs::set_permissions(&restricted_dir, perms).expect("chmod 000");
        }

        // 実際に権限チェックが効くかどうかを、同じ形の probe 操作で実測する
        // （root 実行時は DAC を無視するため `PermissionDenied` にならない）。
        let probe_path = restricted_dir.join("sub").join("a");
        let permission_enforced = matches!(
            std::fs::canonicalize(&probe_path),
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied
        );

        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: valid_params(restricted_dir.to_str().expect("utf-8 path"), "sub/a"),
        }];
        let input = base_input(&label_order, &gold, candidates);
        let result = validate_input::<std::convert::Infallible>(&input);

        // 後片付け: chmod を戻してから削除する（0o000 のままだと
        // `remove_dir_all` 自体が失敗しうるため）。best-effort。
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = std::fs::metadata(&restricted_dir) {
                let mut perms = metadata.permissions();
                perms.set_mode(0o700);
                let _ = std::fs::set_permissions(&restricted_dir, perms);
            }
        }
        let _ = std::fs::remove_dir_all(&base);

        if permission_enforced {
            assert_eq!(
                result.unwrap_err(),
                SearchError::OutDirCanonicalizeFailed { index: 0 }
            );
        } else {
            // root 実行など、権限チェックが効かない環境では検証を通過する
            // （`NotFound` としてより浅い祖先〔`restricted_dir` 自身〕へ
            // 解決できるため）。
            assert!(
                result.is_ok(),
                "expected validate_input to succeed when permission is not enforced (e.g. running as root), got {result:?}"
            );
        }
    }

    /// REQ-18・TASK-18.1-2: リクエストとして不正な構成要素は
    /// `InvalidRequest`。
    #[test]
    fn task18_1_2_validate_input_rejects_invalid_request() {
        let label_order = ["positive", "negative"];
        let gold = ["positive"];
        let mut params = valid_params("/nonexistent-fandhe-root", "out/a");
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
                params: valid_params("/nonexistent-fandhe-root", &format!("out/{i}")),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
                params: valid_params("/nonexistent-fandhe-root", &format!("out/{i}")),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
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
            params: valid_params("/nonexistent-fandhe-root", "out/a"),
        }];
        let input = base_input(&label_order, &gold, candidates);
        let err = validate_input::<std::convert::Infallible>(&input).unwrap_err();
        assert_eq!(err, SearchError::InvalidCandidateId { index: 0 });
    }
}
