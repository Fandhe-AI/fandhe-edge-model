//! 選定結果への McNemar・Holm 有意性判定の付与（TASK-18.3-1・issue #87）。
//!
//! REQ-18（探索予算付き自動選択）は「下限基準（最頻ラベル）に対する
//! McNemar 有意性判定付きで代表構成を選ぶ」ことを求める。本モジュールは
//! [`fandhe_edge_eval::baseline`]（学習ラベルから多数決の下限基準を導出）・
//! [`fandhe_edge_eval::significance`]（McNemar 検定＋α=0.05 判定）・
//! [`fandhe_edge_eval::holm`]（複数候補の Holm 補正）を接続し、評価済みの
//! 選定候補列から「選ばれた候補が下限基準を有意に上回るか」を求める。
//! 評価ロジック（McNemar・Holm・多数決）そのものは再実装しない
//! （`.claude/rules/coding-rust.md`「crate 構成と層の境界」: 評価器は
//! TASK-24.1 系の 1 つに集約する）。
//!
//! # 呼び出し文脈
//!
//! 選定（TASK-18.1-2・#84。本 issue の時点で選定記録型は未実装）が、
//! 評価済みの各候補の validation 予測を集めた時点で本モジュールを呼び、
//! 戻り値 [`SelectionSignificance`] を選定記録へ値として埋め込む想定。
//! CLI の JSON 化（TASK-33.x）・終了コードへの写像は本モジュールの範囲外。
//!
//! # 評価契約との関係（REQ-27・REQ-25）
//!
//! - 比較に使うのは選定に使った **validation 分割のみ**。凍結した最終 test
//!   は使わない（最終 test への適用は 1 回限りという不変条件。TASK-18.4 は
//!   見送り）
//! - 下限基準（多数決）は **学習データのラベルからのみ** 導出する
//!   （`train_labels` と `validation_gold` を別フィールドで受け取り、
//!   validation の gold から下限基準を導出させない。評価の独立性の不変条件）
//! - α・「有意に上回る」の判定規則（`b > c` かつ `p < 0.05`）・Holm 補正の
//!   手順は [`fandhe_edge_eval`] に一任し、本モジュールでは再定義・緩和しない
//! - Holm の族サイズ（[`fandhe_edge_eval::holm::FamilySize`]）は呼び出し側が
//!   事前登録した値（脱落候補を含む）をそのまま渡す。実行時の候補数から
//!   自動算出しない（`fandhe_edge_eval::holm` の「事前登録の規則」参照）
//! - 必要件数（[`fandhe_edge_eval::significance::RequiredSampleSize`]）は
//!   既定値を持たず、呼び出し側が必須引数で渡す（PR #230・TASK-25.2 の
//!   Connor 式による算出がマージされ次第、呼び出し側が算出値を渡すだけで
//!   本モジュールの変更は不要）
//!
//! # 対象外（本 issue の範囲外）
//!
//! - 選定記録型・探索予算の積算・最高正解率の選定そのもの（TASK-18.1-2・#84）
//! - 件数不足時の「判定不能」を選定が合格扱いにしないことの確認テスト
//!   （TASK-18.3-2・#88。`verdict()` が [`fandhe_edge_eval::significance::BaselineVerdict`]
//!   をそのまま返すため #88 は API 変更なしでテストを書ける）
//! - 「予算到達」候補を合格扱いにしない判定（TASK-18.2・#85）
//! - 「3 seed すべてで有意」の集約
//! - 脱落候補を個別に「有意でない」と記録・報告する規則（本モジュールは
//!   族サイズを事前登録値で固定する保守側の補正のみ行う）
//! - JSON 直列化・終了コード写像（TASK-33.x・#140）

use std::collections::BTreeSet;
use std::fmt;

use fandhe_edge_eval::baseline::{self, BaselineError};
use fandhe_edge_eval::holm::{self, FamilySize, HolmComparison, HolmError};
use fandhe_edge_eval::mcnemar::PairedCounts;
use fandhe_edge_eval::metrics::Outcome;
use fandhe_edge_eval::significance::{
    self, BaselineComparison, BaselineVerdict, PairedRecord, RequiredSampleSize,
};

/// 候補 ID のバイト長上限（REQ-39 資源の上限）。
///
/// PoC-17 の候補 ID（例 `c1-a-i8`）を十分収める暫定値。データ契約層の
/// 確定値が無い段階の暫定値で、[`fandhe_edge_eval::significance::MAX_EVAL_RECORDS`]・
/// [`fandhe_edge_eval::holm::MAX_FAMILY_SIZE`] と同様、実測に基づく調整は
/// 後続 TASK で行う。
pub const MAX_CANDIDATE_ID_BYTES: usize = 128;

/// 候補数（`family_size`）× 評価件数（`validation_gold.len()`）の積の上限
/// （REQ-39 資源の上限。reviewer 指摘 PR #236）。
///
/// [`holm::MAX_FAMILY_SIZE`]（10,000）・[`significance::MAX_EVAL_RECORDS`]
/// （1,000,000）はそれぞれ単体の上限であり、両方が上限内でも積は最大
/// 100 億（候補数×評価件数分の `PairedRecord` 生成＋McNemar 比較）に達し、
/// 処理時間の上限（security.md「ガード層: 資源の上限」）を満たせない。
/// [`assess_selection_significance`] は `PairedRecord` の `Vec` を確保する
/// メインループより前に、この積を検証して超過時は計算を開始せず拒否する。
/// [`fandhe_edge_eval::mcnemar::MAX_DISCORDANT_PAIRS`]（1000 万）と同じ
/// オーダーの暫定値とし、実測に基づく調整は後続 TASK で行う。
pub const MAX_CANDIDATE_RECORD_PRODUCT: u64 = 10_000_000;

/// `train_labels`（学習データの正解ラベル列）の件数上限
/// （REQ-39 資源の上限。Codex 指摘 PR #236・thread PRRT_kwDOUq-SxM6mrgbM）。
///
/// [`assess_selection_significance`] の `validation_gold`（評価データ）は
/// [`significance::MAX_EVAL_RECORDS`] で確保前に上限検証しているが、
/// `train_labels` は検証なしで [`baseline::fit_majority`] へ渡され全件走査
/// されていた。`train_labels` に比例したアロケーションは発生しないものの、
/// 走査自体の処理時間に上限が無いままだったため、[`significance::MAX_EVAL_RECORDS`]
/// と同じ数量オーダーの暫定値を計算開始前の上限として設ける。実測に基づく
/// 調整は後続 TASK で行う。
pub const MAX_TRAIN_LABELS: usize = significance::MAX_EVAL_RECORDS;

/// 1 候補分の validation 予測。
///
/// `outcomes` は [`SelectionSignificanceInput::validation_gold`] と
/// 同じ順・同じ件数でなければならない（[`assess_selection_significance`]
/// が検証する）。
#[derive(Debug, Clone, Copy)]
pub struct CandidateValidation<'a> {
    /// 候補 ID（PoC-17 の命名規則。例 `c1-a-i8`）。
    pub candidate_id: &'a str,
    /// validation 分割に対する候補モデルの推論結果（gold と同じ順）。
    pub outcomes: &'a [Outcome],
}

/// [`assess_selection_significance`] への入力。
///
/// すべて参照のみで受け取り、書き換えない（REQ-27 評価の独立性）。
pub struct SelectionSignificanceInput<'a> {
    /// ラベル集合（`Definition::options()` の `id` を宣言順で並べたもの）。
    pub label_order: &'a [&'a str],
    /// 学習データの正解ラベル列。下限基準（多数決）の導出専用で、
    /// **validation の gold を渡してはならない**（評価の独立性）。
    pub train_labels: &'a [&'a str],
    /// validation 分割の正解ラベル列。
    pub validation_gold: &'a [&'a str],
    /// 評価済みの全候補（選定候補を含む）。
    pub candidates: &'a [CandidateValidation<'a>],
    /// 選定された候補の ID（`candidates` 内に存在しなければならない）。
    pub selected_candidate_id: &'a str,
    /// 下限基準比較に必要な最小評価件数（事前登録値。既定値なし）。
    pub required: RequiredSampleSize,
    /// Holm 補正の族サイズ（事前登録した候補総数。脱落候補を含む）。
    pub family_size: FamilySize,
}

/// [`assess_selection_significance`] が返しうるエラー。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SelectionSignificanceError {
    /// `candidates` が 0 件。
    EmptyCandidates,
    /// 候補 ID が不正（空・[`MAX_CANDIDATE_ID_BYTES`] 超過・制御文字を含む）。
    InvalidCandidateId {
        /// `candidates` 内での位置（0 始まり）。
        index: usize,
    },
    /// 候補 ID が重複している。
    DuplicateCandidateId {
        /// `candidates` 内での位置（0 始まり）。重複が検出された 2 件目の位置。
        index: usize,
    },
    /// `selected_candidate_id` が `candidates` 内に見つからない。
    SelectedCandidateNotFound,
    /// 候補の `outcomes` 件数が `validation_gold` と一致しない。
    OutcomeCountMismatch {
        /// `candidates` 内での位置（0 始まり）。
        index: usize,
        /// 期待した件数（`validation_gold.len()`）。
        expected: usize,
        /// 実際の件数。
        actual: usize,
    },
    /// `validation_gold` の件数が上限を超える（REQ-39。確保前検証）。
    TooManyRecords {
        /// 渡された件数。
        n_records: usize,
        /// 上限（[`fandhe_edge_eval::significance::MAX_EVAL_RECORDS`]）。
        limit: usize,
    },
    /// 候補数 × 評価件数の積が上限（[`MAX_CANDIDATE_RECORD_PRODUCT`]）を
    /// 超える（REQ-39。`PairedRecord` の `Vec` を確保するメインループの
    /// 前に拒否する）。
    TooManyComparisons {
        /// 候補数（`family_size` ではなく実際に渡された `candidates.len()`）。
        n_candidates: usize,
        /// 評価件数（`validation_gold.len()`）。
        n_records: usize,
        /// 上限（[`MAX_CANDIDATE_RECORD_PRODUCT`]）。
        limit: u64,
    },
    /// `train_labels` の件数が上限（[`MAX_TRAIN_LABELS`]）を超える
    /// （REQ-39。[`baseline::fit_majority`] の全件走査より前に拒否する）。
    TooManyTrainLabels {
        /// 渡された件数。
        n_labels: usize,
        /// 上限（[`MAX_TRAIN_LABELS`]）。
        limit: usize,
    },
    /// 下限基準（多数決）の導出に失敗した。
    Baseline(BaselineError),
    /// Holm 補正に失敗した。
    Holm(HolmError),
}

impl From<BaselineError> for SelectionSignificanceError {
    fn from(err: BaselineError) -> Self {
        SelectionSignificanceError::Baseline(err)
    }
}

impl From<HolmError> for SelectionSignificanceError {
    fn from(err: HolmError) -> Self {
        SelectionSignificanceError::Holm(err)
    }
}

impl fmt::Display for SelectionSignificanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectionSignificanceError::EmptyCandidates => {
                write!(f, "candidates must not be empty")
            }
            SelectionSignificanceError::InvalidCandidateId { index } => {
                write!(f, "invalid candidate id at index {index}")
            }
            SelectionSignificanceError::DuplicateCandidateId { index } => {
                write!(f, "duplicate candidate id at index {index}")
            }
            SelectionSignificanceError::SelectedCandidateNotFound => {
                write!(f, "selected candidate id was not found among candidates")
            }
            SelectionSignificanceError::OutcomeCountMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "candidate at index {index} has {actual} outcomes, expected {expected}"
            ),
            SelectionSignificanceError::TooManyRecords { n_records, limit } => {
                write!(f, "too many records: {n_records} (limit: {limit})")
            }
            SelectionSignificanceError::TooManyComparisons {
                n_candidates,
                n_records,
                limit,
            } => write!(
                f,
                "too many comparisons: {n_candidates} candidates x {n_records} records (limit: {limit})"
            ),
            SelectionSignificanceError::TooManyTrainLabels { n_labels, limit } => {
                write!(f, "too many train labels: {n_labels} (limit: {limit})")
            }
            SelectionSignificanceError::Baseline(err) => write!(f, "{err}"),
            SelectionSignificanceError::Holm(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for SelectionSignificanceError {}

/// 選定候補の McNemar・Holm 有意性判定結果。
///
/// フィールドは非公開にし、アクセサ経由で公開する（[`BaselineComparison`]・
/// [`HolmComparison`] と同じ設計。`PValue`・`McNemarExact` を含む型を保持
/// するため `PartialEq` は derive しない）。
#[derive(Debug, Clone, Copy)]
pub struct SelectionSignificance {
    candidate_id_len: usize,
    majority_label_len: usize,
    family_size: FamilySize,
    holm: HolmComparison,
}

impl SelectionSignificance {
    /// 選定候補が下限基準を有意に上回るかの最終判定（Holm 補正後）。
    ///
    /// 評価件数が必要件数未満だった場合は判定不能
    /// （[`BaselineVerdict::Undeterminable`]）のまま維持される。
    #[must_use]
    pub fn verdict(&self) -> BaselineVerdict {
        self.holm.verdict()
    }

    /// Holm 補正前（生）の有意性判定。
    #[must_use]
    pub fn raw_verdict(&self) -> BaselineVerdict {
        self.holm.comparison().verdict()
    }

    /// Holm 補正後の p 値。
    #[must_use]
    pub fn adjusted_p(&self) -> fandhe_edge_eval::mcnemar::PValue {
        self.holm.adjusted_p()
    }

    /// Holm 補正前（生）の両側 p 値。
    #[must_use]
    pub fn raw_p(&self) -> fandhe_edge_eval::mcnemar::PValue {
        self.holm.comparison().test().p_two_sided()
    }

    /// 対応のある正誤の集計（n・both_correct・b・c・both_wrong）。
    #[must_use]
    pub fn counts(&self) -> PairedCounts {
        self.holm.comparison().counts()
    }

    /// 補正前の McNemar 比較結果一式。
    #[must_use]
    pub fn comparison(&self) -> BaselineComparison {
        self.holm.comparison()
    }

    /// Holm 補正の結果一式。
    #[must_use]
    pub fn holm(&self) -> HolmComparison {
        self.holm
    }

    /// この判定で使った Holm の族サイズ（事前登録値）。
    #[must_use]
    pub fn family_size(&self) -> FamilySize {
        self.family_size
    }

    /// 候補 ID のバイト長（データ本文の漏えい防止のため文字列自体は
    /// 保持しない。呼び出し側は入力の `selected_candidate_id` をそのまま
    /// 使えるため、アクセサとしては長さのみを公開する）。
    #[must_use]
    pub fn candidate_id_len(&self) -> usize {
        self.candidate_id_len
    }

    /// 下限基準（多数決）ラベルのバイト長。
    #[must_use]
    pub fn majority_label_len(&self) -> usize {
        self.majority_label_len
    }
}

/// 候補 ID が非空・[`MAX_CANDIDATE_ID_BYTES`] 以下・制御文字を含まないか検証する。
fn validate_candidate_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= MAX_CANDIDATE_ID_BYTES && !id.chars().any(|c| c.is_control())
}

/// 評価済みの選定候補列から、選定候補の McNemar・Holm 有意性判定を求める。
///
/// # 手順
///
/// 1. `candidates` が空 → [`SelectionSignificanceError::EmptyCandidates`]
/// 2. 候補ごとの ID を検証（非空・上限以下・制御文字なし）→
///    [`SelectionSignificanceError::InvalidCandidateId`]。重複 →
///    [`SelectionSignificanceError::DuplicateCandidateId`]
/// 3. `selected_candidate_id` を検証したうえで候補列から検索 → 見つからなければ
///    [`SelectionSignificanceError::SelectedCandidateNotFound`]
/// 4. `validation_gold.len()` が上限
///    （[`fandhe_edge_eval::significance::MAX_EVAL_RECORDS`]）以下か確認
///    （[`PairedRecord`] の `Vec` 確保前。REQ-39）。
/// 5. 候補数（`n_candidates`）× `validation_gold.len()` の積が上限
///    （[`MAX_CANDIDATE_RECORD_PRODUCT`]）以下か確認（メインループ〔6〕の
///    前。REQ-39）→ [`SelectionSignificanceError::TooManyComparisons`]。
///    各候補の `outcomes.len() == validation_gold.len()` を確認 →
///    [`SelectionSignificanceError::OutcomeCountMismatch`]
/// 6. `train_labels.len()` が上限（[`MAX_TRAIN_LABELS`]）以下か確認
///    （[`baseline::fit_majority`] の全件走査より前。REQ-39）→
///    [`SelectionSignificanceError::TooManyTrainLabels`]
/// 7. [`baseline::fit_majority`] で `train_labels` から多数決の下限基準を求める
/// 8. 候補ごとに [`significance::compare_with_baseline`] を呼び、
///    [`BaselineComparison`] を候補順に集める
/// 9. [`holm::compare_candidates_with_holm`] で全候補を 1 つの族として
///    Holm 補正し、選定候補の [`HolmComparison`] を取り出す
///
/// # 資源上限（REQ-39）
///
/// `validation_gold.len()` の上限検証を [`PairedRecord`] の作業用 `Vec` を
/// 確保する前に行う。作業用 `Vec` は 1 本を候補ごとに `clear()` して再利用し、
/// 候補数×件数の確保はしない。候補数（`family_size`）・件数
/// （`validation_gold.len()`）それぞれの上限は
/// [`holm::compare_candidates_with_holm`]（[`holm::MAX_FAMILY_SIZE`]）・
/// [`significance::MAX_EVAL_RECORDS`] に委ねるが、両方が上限内でも積が
/// 候補数×件数に比例するメインループの処理時間を膨大にしうるため、
/// 本関数が [`MAX_CANDIDATE_RECORD_PRODUCT`] で積を計算開始前に検証する
/// （reviewer 指摘 PR #236）。`train_labels`（[`baseline::fit_majority`] が
/// 全件走査する）も同様に [`MAX_TRAIN_LABELS`] で計算開始前に件数上限を
/// 検証する（Codex 指摘 PR #236）。
///
/// # 評価契約（REQ-27）
///
/// 入力はすべて参照のみで受け取り、書き換えない。`train_labels` と
/// `validation_gold` を混同しない（下限基準は `train_labels` のみから導出）。
pub fn assess_selection_significance(
    input: &SelectionSignificanceInput<'_>,
) -> Result<SelectionSignificance, SelectionSignificanceError> {
    if input.candidates.is_empty() {
        return Err(SelectionSignificanceError::EmptyCandidates);
    }

    // `family_size` を候補数に対して検証する（REQ-39。候補 ID 検証・
    // McNemar 計算など候補数に比例する処理より前に置く）。
    // `holm::compare_candidates_with_holm` 内でも同じ検証が行われるため
    // 挙動は変わらない（`FamilySizeTooLarge`/`FamilySizeTooSmall` の
    // 早期化のみ）。`coding-rust.md`「サイズ・件数を上限検証してから
    // アロケーションに使う」・`security.md`「資源の上限」に従う。
    let n_candidates = input.candidates.len();
    let m = input.family_size.get();
    if m > holm::MAX_FAMILY_SIZE {
        return Err(SelectionSignificanceError::Holm(
            HolmError::FamilySizeTooLarge {
                family_size: m,
                limit: holm::MAX_FAMILY_SIZE,
            },
        ));
    }
    if m < n_candidates {
        return Err(SelectionSignificanceError::Holm(
            HolmError::FamilySizeTooSmall {
                family_size: m,
                n_tests: n_candidates,
            },
        ));
    }

    // 候補 ID の検証・重複検出（決定的な `BTreeSet` を使う）。
    let mut seen_ids: BTreeSet<&str> = BTreeSet::new();
    for (i, candidate) in input.candidates.iter().enumerate() {
        if !validate_candidate_id(candidate.candidate_id) {
            return Err(SelectionSignificanceError::InvalidCandidateId { index: i });
        }
        if !seen_ids.insert(candidate.candidate_id) {
            return Err(SelectionSignificanceError::DuplicateCandidateId { index: i });
        }
    }

    if !validate_candidate_id(input.selected_candidate_id) {
        return Err(SelectionSignificanceError::SelectedCandidateNotFound);
    }
    let selected_index = input
        .candidates
        .iter()
        .position(|c| c.candidate_id == input.selected_candidate_id)
        .ok_or(SelectionSignificanceError::SelectedCandidateNotFound)?;

    // `PairedRecord` の `Vec` 確保前に件数上限を検証する（REQ-39）。
    let n = input.validation_gold.len();
    if n > significance::MAX_EVAL_RECORDS {
        return Err(SelectionSignificanceError::TooManyRecords {
            n_records: n,
            limit: significance::MAX_EVAL_RECORDS,
        });
    }

    // 候補数 × 評価件数の積を検証する（REQ-39。reviewer 指摘 PR #236）。
    // `n_candidates`（≤ MAX_FAMILY_SIZE）・`n`（≤ MAX_EVAL_RECORDS）が
    // それぞれ単体の上限内でも、積はメインループ（`PairedRecord` 生成＋
    // 候補ごとの McNemar 比較）の処理時間に比例するため、後続のメインループ
    // より前に、単体の上限より小さい積の上限で計算開始前に拒否する。
    // `u64` へ変換してから乗算し（`usize` のオーバーフローを避ける）、
    // 実際の積が `u64::MAX` を超えることはない
    // （n_candidates ≤ 10_000・n ≤ 1_000_000 のため）。
    let comparison_product = (n_candidates as u64).saturating_mul(n as u64);
    if comparison_product > MAX_CANDIDATE_RECORD_PRODUCT {
        return Err(SelectionSignificanceError::TooManyComparisons {
            n_candidates,
            n_records: n,
            limit: MAX_CANDIDATE_RECORD_PRODUCT,
        });
    }

    for (i, candidate) in input.candidates.iter().enumerate() {
        if candidate.outcomes.len() != n {
            return Err(SelectionSignificanceError::OutcomeCountMismatch {
                index: i,
                expected: n,
                actual: candidate.outcomes.len(),
            });
        }
    }

    // `train_labels` の件数上限を [`baseline::fit_majority`] の全件走査より
    // 前に検証する（REQ-39。Codex 指摘 PR #236・thread PRRT_kwDOUq-SxM6mrgbM）。
    let n_train_labels = input.train_labels.len();
    if n_train_labels > MAX_TRAIN_LABELS {
        return Err(SelectionSignificanceError::TooManyTrainLabels {
            n_labels: n_train_labels,
            limit: MAX_TRAIN_LABELS,
        });
    }

    // 下限基準（多数決）は学習ラベルからのみ導出する（評価の独立性）。
    let majority_label = baseline::fit_majority(input.label_order, input.train_labels)?;
    let majority_label_len = majority_label.len();
    let baseline_outcome = Outcome::Label(majority_label.to_string());
    // 全行で同じ下限基準予測を参照する（行ごとの `Outcome` 確保をしない）。
    let baseline_outcomes: Vec<&Outcome> = vec![&baseline_outcome; n];

    // 作業用 `Vec` を 1 本だけ確保し、候補ごとに `clear()` して再利用する。
    let mut records: Vec<PairedRecord<'_>> = Vec::with_capacity(n);
    let mut comparisons: Vec<BaselineComparison> = Vec::with_capacity(input.candidates.len());
    for candidate in input.candidates {
        records.clear();
        for i in 0..n {
            let gold = input.validation_gold.get(i).ok_or_else(|| {
                SelectionSignificanceError::Baseline(BaselineError::Internal {
                    detail: "validation gold index out of bounds".to_string(),
                })
            })?;
            let candidate_outcome = candidate.outcomes.get(i).ok_or_else(|| {
                SelectionSignificanceError::Baseline(BaselineError::Internal {
                    detail: "candidate outcome index out of bounds".to_string(),
                })
            })?;
            let baseline_outcome_ref = baseline_outcomes.get(i).ok_or_else(|| {
                SelectionSignificanceError::Baseline(BaselineError::Internal {
                    detail: "baseline outcome index out of bounds".to_string(),
                })
            })?;
            records.push(PairedRecord {
                gold,
                candidate: candidate_outcome,
                baseline: baseline_outcome_ref,
            });
        }
        let comparison =
            significance::compare_with_baseline(input.label_order, &records, input.required)?;
        comparisons.push(comparison);
    }

    let holm_results = holm::compare_candidates_with_holm(&comparisons, input.family_size)?;
    let selected_holm = holm_results
        .get(selected_index)
        .copied()
        .ok_or(SelectionSignificanceError::SelectedCandidateNotFound)?;

    Ok(SelectionSignificance {
        candidate_id_len: input.selected_candidate_id.len(),
        majority_label_len,
        family_size: input.family_size,
        holm: selected_holm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(n: u64) -> RequiredSampleSize {
        RequiredSampleSize::new(n).expect("test helper requires a non-zero value")
    }

    fn family(n: usize) -> FamilySize {
        FamilySize::new(n).expect("test helper requires a non-zero family size")
    }

    /// 決定的に (both_correct, b, c, both_wrong) から validation 行を組み立てる。
    /// gold=A・候補正解: both_correct 行 / gold=B・候補正解: b 行 /
    /// gold=A・候補不正解: c 行（誤りの種類を混ぜる）/
    /// gold=B・候補不正解: both_wrong 行。
    fn build_outcomes(
        both_correct: usize,
        b: usize,
        c: usize,
        both_wrong: usize,
    ) -> (Vec<&'static str>, Vec<Outcome>) {
        let mut gold = Vec::new();
        let mut outcomes = Vec::new();
        for _ in 0..both_correct {
            gold.push("A");
            outcomes.push(Outcome::Label("A".to_string()));
        }
        for _ in 0..b {
            gold.push("B");
            outcomes.push(Outcome::Label("B".to_string()));
        }
        let wrong_kinds = [
            Outcome::Label("B".to_string()),
            Outcome::Invalid,
            Outcome::Abstain,
            Outcome::Error,
        ];
        for i in 0..c {
            gold.push("A");
            outcomes.push(wrong_kinds[i % wrong_kinds.len()].clone());
        }
        for _ in 0..both_wrong {
            gold.push("B");
            outcomes.push(Outcome::Label("C".to_string()));
        }
        (gold, outcomes)
    }

    fn train_labels_majority_a() -> Vec<&'static str> {
        // A×5, B×3, C×2 → majority = "A"。
        vec!["A", "A", "A", "A", "A", "B", "B", "B", "C", "C"]
    }

    /// 単一候補・有意（p ≈ 0.049041748046875 < 0.05）。
    /// TASK-18.3-1・REQ-18・REQ-25。証拠種別: テストハーネス。
    #[test]
    fn single_candidate_significantly_better() {
        let (gold, outcomes) = build_outcomes(100, 13, 4, 104);
        let gold_refs: Vec<&str> = gold.clone();
        let train_labels = train_labels_majority_a();
        let candidates = [CandidateValidation {
            candidate_id: "c1-a-i8",
            outcomes: &outcomes,
        }];
        let labels = ["A", "B", "C"];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold_refs,
            candidates: &candidates,
            selected_candidate_id: "c1-a-i8",
            required: req(221),
            family_size: family(1),
        };

        let result = assess_selection_significance(&input).unwrap();
        assert_eq!(result.counts().n, 221);
        assert_eq!(result.counts().b_candidate_only, 13);
        assert_eq!(result.counts().c_baseline_only, 4);
        assert!((result.raw_p().value() - 0.049_041_748_046_875).abs() < 1e-9);
        assert!((result.adjusted_p().value() - 0.049_041_748_046_875).abs() < 1e-9);
        assert_eq!(result.verdict(), BaselineVerdict::SignificantlyBetter);
        assert_eq!(result.raw_verdict(), BaselineVerdict::SignificantlyBetter);
        assert_eq!(result.candidate_id_len(), "c1-a-i8".len());
        assert_eq!(result.majority_label_len(), "A".len());

        // eval の `compare_with_baseline` を直接呼んだ p と一致することを
        // 確認する（評価ロジックを再実装していないことの照合）。
        let baseline_outcome = Outcome::Label("A".to_string());
        let records: Vec<PairedRecord<'_>> = gold_refs
            .iter()
            .zip(outcomes.iter())
            .map(|(g, o)| PairedRecord {
                gold: g,
                candidate: o,
                baseline: &baseline_outcome,
            })
            .collect();
        let direct = significance::compare_with_baseline(&labels, &records, req(221)).unwrap();
        assert!((direct.test().p_two_sided().value() - result.raw_p().value()).abs() < 1e-9);
    }

    /// 単一候補・非有意（b=c=10 → p=1.0）。
    #[test]
    fn single_candidate_not_significantly_better() {
        let (gold, outcomes) = build_outcomes(0, 10, 10, 0);
        let train_labels = train_labels_majority_a();
        let candidates = [CandidateValidation {
            candidate_id: "c1-a-i8",
            outcomes: &outcomes,
        }];
        let labels = ["A", "B", "C"];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "c1-a-i8",
            required: req(20),
            family_size: family(1),
        };

        let result = assess_selection_significance(&input).unwrap();
        assert!((result.raw_p().value() - 1.0).abs() < 1e-9);
        assert_eq!(result.verdict(), BaselineVerdict::NotSignificantlyBetter);
    }

    /// 全候補で共有する validation_gold（gold1）の上で、`b`・`c` 件数だけを
    /// 指定して別候補の予測列を組み立てる。gold1 は A×104・B×117
    /// （[`build_outcomes`] の `(100, 13, 4, 104)` が生成する並び）。
    /// 先頭から `b` 件の B-gold 行を候補正解（B）、先頭から `c` 件の
    /// A-gold 行を候補不正解（下限基準 "A" とは異なる "B"）にし、
    /// 残りは下限基準と同じ側（both_correct / both_wrong）に倒すことで、
    /// 指定した `b`・`c` 以外は McNemar の不一致ペアに寄与しないようにする。
    fn build_second_candidate_outcomes(gold: &[&str], b: usize, c: usize) -> Vec<Outcome> {
        let mut b_left = b;
        let mut c_left = c;
        gold.iter()
            .map(|&g| {
                if g == "B" && b_left > 0 {
                    b_left -= 1;
                    Outcome::Label("B".to_string())
                } else if g == "A" && c_left > 0 {
                    c_left -= 1;
                    Outcome::Label("B".to_string())
                } else if g == "B" {
                    // both_wrong 側（下限基準 "A" も候補も不正解）。
                    Outcome::Label("C".to_string())
                } else {
                    // both_correct 側（下限基準 "A" も候補も正解）。
                    Outcome::Label("A".to_string())
                }
            })
            .collect()
    }

    /// Holm 補正で判定が変わる（選定候補・他候補 2 件、m=2）。
    /// TASK-18.3-1・REQ-18・REQ-25。証拠種別: テストハーネス。
    #[test]
    fn holm_correction_changes_verdict() {
        let (gold1, outcomes1) = build_outcomes(100, 13, 4, 104);
        assert_eq!(gold1.len(), 221);
        let outcomes2 = build_second_candidate_outcomes(&gold1, 10, 10);
        let train_labels = train_labels_majority_a();
        let labels = ["A", "B", "C"];

        let candidates = [
            CandidateValidation {
                candidate_id: "c1-a-i8",
                outcomes: &outcomes1,
            },
            CandidateValidation {
                candidate_id: "c3-b64",
                outcomes: &outcomes2,
            },
        ];

        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold1,
            candidates: &candidates,
            selected_candidate_id: "c1-a-i8",
            required: req(221),
            family_size: family(2),
        };

        let result = assess_selection_significance(&input).unwrap();
        assert!((result.raw_p().value() - 0.049_041_748_046_875).abs() < 1e-9);
        assert_eq!(result.raw_verdict(), BaselineVerdict::SignificantlyBetter);
        assert!((result.adjusted_p().value() - 0.098_083_496_093_75).abs() < 1e-9);
        assert_eq!(result.verdict(), BaselineVerdict::NotSignificantlyBetter);
    }

    /// 候補が空ならエラー。
    #[test]
    fn empty_candidates_is_error() {
        let labels = ["A", "B"];
        let train_labels = ["A", "B"];
        let gold: Vec<&str> = Vec::new();
        let candidates: [CandidateValidation<'_>; 0] = [];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "c1",
            required: req(1),
            family_size: family(1),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(err, SelectionSignificanceError::EmptyCandidates);
    }

    /// 候補 ID が空文字列ならエラー。
    #[test]
    fn empty_candidate_id_is_error() {
        let labels = ["A", "B"];
        let train_labels = ["A", "B"];
        let outcomes = [Outcome::Label("A".to_string())];
        let gold = ["A"];
        let candidates = [CandidateValidation {
            candidate_id: "",
            outcomes: &outcomes,
        }];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "",
            required: req(1),
            family_size: family(1),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(
            err,
            SelectionSignificanceError::InvalidCandidateId { index: 0 }
        );
    }

    /// 候補 ID が上限（128 バイト）を超えるとエラー。
    #[test]
    fn too_long_candidate_id_is_error() {
        let labels = ["A", "B"];
        let train_labels = ["A", "B"];
        let outcomes = [Outcome::Label("A".to_string())];
        let gold = ["A"];
        let long_id = "x".repeat(MAX_CANDIDATE_ID_BYTES + 1);
        let candidates = [CandidateValidation {
            candidate_id: &long_id,
            outcomes: &outcomes,
        }];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: &long_id,
            required: req(1),
            family_size: family(1),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(
            err,
            SelectionSignificanceError::InvalidCandidateId { index: 0 }
        );
    }

    /// 候補 ID に制御文字（改行）を含むとエラー。
    #[test]
    fn candidate_id_with_control_char_is_error() {
        let labels = ["A", "B"];
        let train_labels = ["A", "B"];
        let outcomes = [Outcome::Label("A".to_string())];
        let gold = ["A"];
        let candidates = [CandidateValidation {
            candidate_id: "c1\na",
            outcomes: &outcomes,
        }];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "c1\na",
            required: req(1),
            family_size: family(1),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(
            err,
            SelectionSignificanceError::InvalidCandidateId { index: 0 }
        );
    }

    /// 候補 ID の重複はエラー。
    #[test]
    fn duplicate_candidate_id_is_error() {
        let labels = ["A", "B"];
        let train_labels = ["A", "B"];
        let outcomes = [Outcome::Label("A".to_string())];
        let gold = ["A"];
        let candidates = [
            CandidateValidation {
                candidate_id: "c1",
                outcomes: &outcomes,
            },
            CandidateValidation {
                candidate_id: "c1",
                outcomes: &outcomes,
            },
        ];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "c1",
            required: req(1),
            family_size: family(2),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(
            err,
            SelectionSignificanceError::DuplicateCandidateId { index: 1 }
        );
    }

    /// 選定候補が候補列に存在しなければエラー。
    #[test]
    fn selected_candidate_not_found_is_error() {
        let labels = ["A", "B"];
        let train_labels = ["A", "B"];
        let outcomes = [Outcome::Label("A".to_string())];
        let gold = ["A"];
        let candidates = [CandidateValidation {
            candidate_id: "c1",
            outcomes: &outcomes,
        }];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "c2",
            required: req(1),
            family_size: family(1),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(err, SelectionSignificanceError::SelectedCandidateNotFound);
    }

    /// 候補の outcomes 件数が validation_gold と一致しなければエラー。
    #[test]
    fn outcome_count_mismatch_is_error() {
        let labels = ["A", "B"];
        let train_labels = ["A", "B"];
        let outcomes = [
            Outcome::Label("A".to_string()),
            Outcome::Label("B".to_string()),
        ];
        let gold = ["A"];
        let candidates = [CandidateValidation {
            candidate_id: "c1",
            outcomes: &outcomes,
        }];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "c1",
            required: req(1),
            family_size: family(1),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(
            err,
            SelectionSignificanceError::OutcomeCountMismatch {
                index: 0,
                expected: 1,
                actual: 2,
            }
        );
    }

    /// `validation_gold` の件数が上限（[`significance::MAX_EVAL_RECORDS`]）を
    /// 超えると、`PairedRecord` の `Vec` を確保する前に `TooManyRecords` で
    /// 拒否する（REQ-39 資源の上限。`crates/eval` 側の同種の上限テスト
    /// `correctness_too_many_records_is_error` 等と対にする）。
    #[test]
    fn too_many_records_is_error() {
        let labels = ["A"];
        let train_labels = ["A"];
        let outcomes = [Outcome::Label("A".to_string())];
        let gold: Vec<&str> = vec!["A"; significance::MAX_EVAL_RECORDS + 1];
        let candidates = [CandidateValidation {
            candidate_id: "c1",
            outcomes: &outcomes,
        }];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "c1",
            required: req(1),
            family_size: family(1),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(
            err,
            SelectionSignificanceError::TooManyRecords {
                n_records: significance::MAX_EVAL_RECORDS + 1,
                limit: significance::MAX_EVAL_RECORDS,
            }
        );
    }

    /// 候補数（`family_size` 上限一杯）と評価件数がそれぞれ
    /// [`holm::MAX_FAMILY_SIZE`]・[`significance::MAX_EVAL_RECORDS`] の
    /// 個別上限内でも、積が [`MAX_CANDIDATE_RECORD_PRODUCT`] を超えると
    /// `PairedRecord` を組み立てるメインループの前に `TooManyComparisons`
    /// で拒否する（REQ-39 資源の上限。reviewer 指摘 PR #236）。
    #[test]
    fn too_many_comparisons_is_error() {
        let labels = ["A"];
        let train_labels = ["A"];
        let outcomes = [Outcome::Label("A".to_string())];
        let n_candidates = holm::MAX_FAMILY_SIZE;
        // 個々の上限（MAX_FAMILY_SIZE・MAX_EVAL_RECORDS）は満たすが、
        // 積が MAX_CANDIDATE_RECORD_PRODUCT をわずかに超える件数にする。
        let n_records = (MAX_CANDIDATE_RECORD_PRODUCT / n_candidates as u64) as usize + 1;
        assert!(n_records <= significance::MAX_EVAL_RECORDS);
        let candidate_ids: Vec<String> = (0..n_candidates).map(|i| format!("c{i}")).collect();
        let candidates: Vec<CandidateValidation<'_>> = candidate_ids
            .iter()
            .map(|id| CandidateValidation {
                candidate_id: id.as_str(),
                outcomes: &outcomes,
            })
            .collect();
        let gold: Vec<&str> = vec!["A"; n_records];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: candidate_ids[0].as_str(),
            required: req(1),
            family_size: family(n_candidates),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(
            err,
            SelectionSignificanceError::TooManyComparisons {
                n_candidates,
                n_records,
                limit: MAX_CANDIDATE_RECORD_PRODUCT,
            }
        );
    }

    /// `train_labels` の件数が上限（[`MAX_TRAIN_LABELS`]）を超えると、
    /// [`baseline::fit_majority`] の全件走査より前に `TooManyTrainLabels`
    /// で拒否する（REQ-39 資源の上限。Codex 指摘 PR #236・
    /// thread PRRT_kwDOUq-SxM6mrgbM。`too_many_records_is_error` と対にする）。
    #[test]
    fn too_many_train_labels_is_error() {
        let labels = ["A"];
        let train_labels: Vec<&str> = vec!["A"; MAX_TRAIN_LABELS + 1];
        let outcomes = [Outcome::Label("A".to_string())];
        let gold = ["A"];
        let candidates = [CandidateValidation {
            candidate_id: "c1",
            outcomes: &outcomes,
        }];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "c1",
            required: req(1),
            family_size: family(1),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(
            err,
            SelectionSignificanceError::TooManyTrainLabels {
                n_labels: MAX_TRAIN_LABELS + 1,
                limit: MAX_TRAIN_LABELS,
            }
        );
    }

    /// 族サイズが候補数未満なら Holm 側のエラーとして伝播する。
    #[test]
    fn family_size_too_small_propagates_holm_error() {
        let labels = ["A", "B"];
        let train_labels = ["A", "B"];
        let outcomes = [Outcome::Label("A".to_string())];
        let gold = ["A"];
        let candidates = [
            CandidateValidation {
                candidate_id: "c1",
                outcomes: &outcomes,
            },
            CandidateValidation {
                candidate_id: "c2",
                outcomes: &outcomes,
            },
        ];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "c1",
            required: req(1),
            family_size: family(1),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(
            err,
            SelectionSignificanceError::Holm(HolmError::FamilySizeTooSmall {
                family_size: 1,
                n_tests: 2,
            })
        );
    }

    /// 学習ラベルに未知ラベルがあれば `Baseline` エラーとして伝播する。
    #[test]
    fn unknown_train_label_propagates_baseline_error() {
        let labels = ["A", "B"];
        let train_labels = ["A", "Z"];
        let outcomes = [Outcome::Label("A".to_string())];
        let gold = ["A"];
        let candidates = [CandidateValidation {
            candidate_id: "c1",
            outcomes: &outcomes,
        }];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "c1",
            required: req(1),
            family_size: family(1),
        };
        let err = assess_selection_significance(&input).unwrap_err();
        assert_eq!(
            err,
            SelectionSignificanceError::Baseline(BaselineError::UnknownTrainLabel { index: 1 })
        );
    }

    /// `Debug` 表示にデータ本文（gold・予測文字列）が現れないこと
    /// （候補 ID・多数決ラベルは長さのみを公開する設計のため、そもそも
    /// アクセサ経由では文字列自体を取得できない）。
    #[test]
    fn error_display_does_not_leak_record_content() {
        let err = SelectionSignificanceError::OutcomeCountMismatch {
            index: 0,
            expected: 1,
            actual: 2,
        };
        let msg = format!("{err}");
        assert!(!msg.contains("gold"));
        assert_eq!(msg, "candidate at index 0 has 2 outcomes, expected 1");
    }
}
