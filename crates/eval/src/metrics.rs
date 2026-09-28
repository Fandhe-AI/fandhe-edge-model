//! 正解率・ラベル別指標（適合率・再現率・F1）・Macro-F1・混同行列の算出。
//!
//! CLI の `evaluate` 工程（REQ-33）から、データ契約層で検証済みの
//! gold・推論結果を受け取って呼ばれる想定（本モジュール自体はファイル I/O・
//! JSON 逆シリアル化を行わず、型付きのメモリ上のスライスだけを受け取る）。
//! REQ-24（評価器の正しさ）の正常系・TASK-24.1-1・issue #59 に対応する。
//!
//! # 評価契約との関係
//!
//! - 分母が 0 の指標は `Option<f64>` の `None` として返し、0 や 1 で埋めない
//!   （`.claude/rules/evaluation-contract.md`「有意性・指標」）。平均から
//!   除いたラベルの列挙は [`MacroF1::excluded_labels`] が担う（REQ-24 異常系・
//!   TASK-24.2・issue #61）。`None` を JSON の `null` へ写す直列化は本 crate の
//!   責務ではなく、CLI の `evaluate` 工程（TASK-33.1）が担う（本 crate は
//!   JSON 逆シリアル化・直列化を行わない方針。モジュール冒頭参照）
//! - 入力（[`EvalRecord`]）は `&` 参照でのみ受け取り、書き換えない
//!   （REQ-27: 評価の前後で評価データのハッシュが一致すること）
//! - F1 は `2*TP / (2*TP + FP + FN)` で定義する（適合率・再現率の調和平均
//!   ではない）。調和平均で書くと、適合率が未定義（`None`）のラベルの F1 まで
//!   `None` になり、Macro-F1 の算出から誤って除外されてしまう
//!   （PoC-9 manifest.md「11. 凍結前の修正」2026-09-23。この誤りにより
//!   既知解データセットの Macro-F1 が 49/78・0.4 という誤った値になっていた
//!   経緯がある。本実装ではこの誤りを再現しない）
//!
//! Wilson 95% 信頼区間は本モジュールではなく [`crate::wilson`] に実装し、
//! [`Ratio::ci95`] から呼び出す（TASK-24.1-2・issue #60）。
//!
//! # 資源上限
//!
//! 計算量は `records` の長さとラベル数に比例し、確保するメモリは
//! ラベル数の 2 乗程度（混同行列）に限られる。呼び出し側の定義ファイル検査
//! （`fandhe-edge-core::MAX_DEFINITION_FILE_BYTES`）はファイルサイズの上限のみで
//! ラベル（選択肢）件数の上限を持たないため、本モジュール自身が
//! [`ConfusionMatrix::new`] の確保前に [`MAX_LABELS`] でラベル数を検証し、
//! 超過時は確保せず [`EvalError::TooManyLabels`] を返す（REQ-39 ガード層
//! 「資源の上限」。`records` の件数上限は呼び出し側の責務のまま）。

use std::collections::BTreeMap;
use std::fmt;

/// 予測 1 件の結果（PoC-9 の `type_valid_single` に相当する分類）。
///
/// `Label` の中身がラベル集合に存在しない場合（未知のラベル・空文字列）は、
/// [`evaluate_single_select`] が混同行列の `Invalid` 列へ数える
/// （PoC-9 既知解の ss-12 `"E"`・ss-18 `""` と同じ扱い）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// 推論が ok で、ラベル文字列を返した（ラベル集合に無い場合は Invalid 扱い）。
    Label(String),
    /// 推論は ok だが、ラベル以外の形（型が不正）を返した。
    Invalid,
    /// 推論が保留（判定不能）を返した。
    Abstain,
    /// 推論がエラーで終了した。
    Error,
}

/// 評価 1 件（正解ラベルと予測結果の組）。所有権は取らない（REQ-27）。
#[derive(Debug, Clone, Copy)]
pub struct EvalRecord<'a> {
    /// 正解ラベル ID。ラベル集合に存在しない場合 [`EvalError::UnknownGoldLabel`]。
    pub gold: &'a str,
    /// 推論結果。
    pub outcome: &'a Outcome,
}

/// [`evaluate_single_select`] が返しうるエラー。
///
/// 外部入力（ラベル集合・評価レコード）の異常を fail-closed で表現し、
/// panic させない（`.claude/rules/coding-rust.md`「エラーハンドリング・外部入力」）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvalError {
    /// ラベル集合が空。
    EmptyLabels,
    /// ラベル集合に空文字列の ID が含まれる。
    EmptyLabelId,
    /// ラベル集合に重複した ID が含まれる。
    DuplicateLabel {
        /// 重複したラベル ID。
        label: String,
    },
    /// 評価レコードが 0 件（0 除算を避け、評価済みを装わない）。
    EmptyRecords,
    /// 正解ラベルがラベル集合に存在しない（データ契約層で除外・警告される前提だが、
    /// 本層は fail-closed でエラーを返す）。
    UnknownGoldLabel {
        /// `records` 内での位置（0 始まり）。
        index: usize,
    },
    /// 集計中の桁あふれ、またはラベル添字・行列添字の不整合。
    ///
    /// `labels`・`records` の事前検証を通った時点で理論上到達しないが、
    /// fail-closed のため checked 演算・`Vec` の範囲外添字アクセスの
    /// ガードとして用意する。`UnknownGoldLabel`（正解ラベルがラベル集合に
    /// 存在しないという、入力そのものの異常）とは意味が異なるため分離した。
    Internal {
        /// 発生箇所の説明（人が読める短い文字列。機械照合はしない）。
        detail: String,
    },
    /// ラベル数が上限（[`MAX_LABELS`]）を超える。
    ///
    /// 混同行列は `n_labels * (n_labels + 3)` 個の `u64` を確保する
    /// （[`ConfusionMatrix::new`]）ため、呼び出し側の定義ファイル検査
    /// （`fandhe-edge-core::MAX_DEFINITION_FILE_BYTES`。ファイルサイズの
    /// 上限のみでラベル（選択肢）件数の上限は無い）を通った入力でも
    /// 巨大なラベル数を渡せば無制限のアロケーションになりうる。
    /// ガード層の資源上限（REQ-39・`.claude/rules/security.md`）に従い、
    /// 確保前にここで拒否する。
    TooManyLabels {
        /// 渡されたラベル数。
        n_labels: usize,
        /// 上限（[`MAX_LABELS`]）。
        limit: usize,
    },
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::EmptyLabels => write!(f, "label set must not be empty"),
            EvalError::EmptyLabelId => write!(f, "label set must not contain an empty id"),
            EvalError::DuplicateLabel { label } => {
                write!(f, "duplicate label id: {label}")
            }
            EvalError::EmptyRecords => write!(f, "records must not be empty"),
            EvalError::UnknownGoldLabel { index } => {
                write!(f, "unknown gold label at record index {index}")
            }
            EvalError::Internal { detail } => {
                write!(f, "internal aggregation error: {detail}")
            }
            EvalError::TooManyLabels { n_labels, limit } => {
                write!(f, "too many labels: {n_labels} exceeds limit {limit}")
            }
        }
    }
}

impl std::error::Error for EvalError {}

/// 分子・分母を保持する比率。フィールドは非公開にし、分母 0 の壊れた値
/// （例: `denominator: 0` かつ `value: f64::NAN`）や `numerator > denominator`
/// の壊れた値を外部から構築できないようにする（`.claude/rules/coding-rust.md`
/// 「判定結果...は壊れた値を表現できない型にする」）。生成は [`Ratio::new`] に
/// 集約する。不変条件: `0 < denominator` かつ `numerator <= denominator`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ratio {
    numerator: u64,
    denominator: u64,
    value: f64,
}

impl Ratio {
    /// 分母が 0、または `numerator > denominator`（壊れた比率）のとき `None`
    /// を返す（評価契約: 分母 0 は null。本モジュール内の呼び出しは常に
    /// `numerator <= denominator` を満たすため挙動は変わらないが、型で
    /// 不変条件を守るための検証を追加する）。
    fn new(numerator: u64, denominator: u64) -> Option<Ratio> {
        if denominator == 0 || numerator > denominator {
            return None;
        }
        Some(Ratio {
            numerator,
            denominator,
            value: numerator as f64 / denominator as f64,
        })
    }

    /// 分子。
    pub fn numerator(&self) -> u64 {
        self.numerator
    }

    /// 分母（0 にはならない。0 になりうる場合は [`Ratio::new`] が `None` を返す）。
    pub fn denominator(&self) -> u64 {
        self.denominator
    }

    /// `numerator as f64 / denominator as f64`。
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Wilson 95% 信頼区間（REQ-24・REQ-26・TASK-24.1-2・issue #60）。
    ///
    /// `Ratio` の不変条件（`0 < denominator`・`numerator <= denominator`）の
    /// 下では常に `Some` を返すが、`unwrap`/`expect` で panic させないため
    /// `Option` のまま返す（`.claude/rules/coding-rust.md`「エラーハンドリング」）。
    /// 算出は [`crate::wilson`] モジュールに集約する。
    pub fn ci95(&self) -> Option<crate::wilson::WilsonInterval> {
        crate::wilson::ratio_ci95(self)
    }
}

/// 予測 1 件ずつの正解率算出に使う outcome の分類件数。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OutcomeCounts {
    /// `Outcome::Label(s)` かつ `s` がラベル集合に存在する件数。
    pub ok: u64,
    /// `Outcome::Invalid`、または `Outcome::Label(s)` かつ `s` がラベル集合に無い件数。
    pub invalid: u64,
    /// `Outcome::Abstain` の件数。
    pub abstain: u64,
    /// `Outcome::Error` の件数。
    pub error: u64,
}

/// 全体正解率・採用判断正解率をまとめた型。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Accuracy {
    /// 分母を全件（abstain・error・invalid を不正解として数える）とする正解率。
    pub overall: Ratio,
    /// 分母を `n_total - abstain` とする正解率（error は分母に含める）。
    /// 全件 abstain のときは `None`。
    pub adopted_decision: Option<Ratio>,
}

/// 1 ラベル分の指標。
///
/// `precision`・`recall`・`f1` はいずれも分母が 0 のとき `None`（「未定義」。
/// 評価契約 `.claude/rules/evaluation-contract.md`「有意性・指標」に従い、
/// 0 や 1 で埋めない）。JSON では `null` として表す（直列化は CLI 側の責務。
/// モジュール冒頭「評価契約との関係」参照）。
#[derive(Debug, Clone, PartialEq)]
pub struct LabelMetrics {
    /// ラベル ID。
    pub label: String,
    /// 正解件数（このラベルが gold である件数。混同行列の行和）。
    pub support: u64,
    /// このラベルが予測された件数（混同行列の列和）。
    pub predicted_count: u64,
    /// True Positive。
    pub tp: u64,
    /// False Positive。
    pub fp: u64,
    /// False Negative（Rust の予約語 `fn` を避けた命名）。
    pub fn_: u64,
    /// `tp / predicted_count`。`predicted_count == 0`（このラベルが 1 件も
    /// 予測されなかった）のとき `None`（未定義。受入基準 1: 予測 0 件の
    /// ラベルの適合率が null として返る）。
    pub precision: Option<f64>,
    /// `tp / support`。`support == 0`（このラベルが 1 件も正解でない）のとき
    /// `None`（未定義）。
    pub recall: Option<f64>,
    /// `2*tp / (2*tp + fp + fn_)`。`tp+fp+fn_ == 0`（このラベルが予測にも
    /// 正解にも一度も現れない）のとき `None`（未定義）。[`MacroF1`] は
    /// この `None` を「除外」の判定条件として使う（`precision`・`recall`
    /// が `None` でも `f1` が `Some` なら除外しない）。
    pub f1: Option<f64>,
}

/// 混同行列の列（宣言順のラベル、または非ラベル outcome）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfusionColumn {
    /// 宣言順のラベル添字（[`SingleSelectMetrics`] の `per_label` と対応）。
    Label(usize),
    /// 型が不正、または未知のラベルへの予測。
    Invalid,
    /// 保留。
    Abstain,
    /// 推論エラー。
    Error,
}

/// 混同行列。行は正解ラベル（宣言順）、列は宣言順のラベル + `Invalid` + `Abstain` + `Error`。
///
/// 添字アクセス（`[]`）を公開せず、[`ConfusionMatrix::get`] 経由でのみ参照させる
/// （外部入力由来の添字で panic させないため）。
#[derive(Debug, Clone, PartialEq)]
pub struct ConfusionMatrix {
    /// `rows[gold_index][column_index]`。列の並びは `Label(0..n)`, `Invalid`, `Abstain`, `Error`。
    rows: Vec<Vec<u64>>,
    n_labels: usize,
}

/// 混同行列（ラベル数 `n_labels` の 2 乗程度）の確保前に許容するラベル数の上限。
///
/// `ConfusionMatrix::new` は `n_labels * (n_labels + 3)` 個の `u64`（8 byte）を
/// 確保する。定義ファイルの検査（`fandhe-edge-core::MAX_DEFINITION_FILE_BYTES`）は
/// ファイルサイズの上限のみでラベル（選択肢）件数の上限を持たないため、
/// ここで確保前に拒否する（REQ-39 ガード層「資源の上限」）。
/// `MAX_LABELS`（4096）の下での最大確保量は
/// `4096 * 4099 * 8 byte` ≈ 134 MiB で、実用上の選択肢数（数十〜数百）に対して
/// 十分な余裕を持ちつつ、単一の判定でプロセスを終了させうる規模の
/// アロケーションを防ぐ。
pub const MAX_LABELS: usize = 4096;

impl ConfusionMatrix {
    /// 混同行列を確保する。`n_labels` が [`MAX_LABELS`] を超える場合、
    /// または行列サイズの計算が桁あふれする場合は確保せずに
    /// [`EvalError::TooManyLabels`] / [`EvalError::Internal`] を返す
    /// （REQ-39: 確保前にサイズを検証する）。
    fn new(n_labels: usize) -> Result<ConfusionMatrix, EvalError> {
        if n_labels > MAX_LABELS {
            return Err(EvalError::TooManyLabels {
                n_labels,
                limit: MAX_LABELS,
            });
        }
        // 列数はラベル数 + 3（Invalid・Abstain・Error）。
        let n_columns = n_labels.checked_add(3).ok_or_else(|| EvalError::Internal {
            detail: "confusion matrix column count overflow".to_string(),
        })?;
        // 総セル数を確保前に checked 演算で確認する（`MAX_LABELS` の検証済み
        // 範囲では桁あふれしないが、上限値そのものの整合性を守るため残す）。
        n_labels
            .checked_mul(n_columns)
            .ok_or_else(|| EvalError::Internal {
                detail: "confusion matrix cell count overflow".to_string(),
            })?;
        Ok(ConfusionMatrix {
            rows: vec![vec![0u64; n_columns]; n_labels],
            n_labels,
        })
    }

    fn column_index(&self, column: ConfusionColumn) -> Option<usize> {
        match column {
            ConfusionColumn::Label(i) => (i < self.n_labels).then_some(i),
            ConfusionColumn::Invalid => Some(self.n_labels),
            ConfusionColumn::Abstain => Some(self.n_labels.saturating_add(1)),
            ConfusionColumn::Error => Some(self.n_labels.saturating_add(2)),
        }
    }

    /// `gold_index` 行・`column` 列の件数を返す。範囲外は `None`。
    pub fn get(&self, gold_index: usize, column: ConfusionColumn) -> Option<u64> {
        let col = self.column_index(column)?;
        self.rows.get(gold_index)?.get(col).copied()
    }

    /// `record_index` はエラーメッセージにのみ使う（`records` 内での位置）。
    fn increment(
        &mut self,
        gold_index: usize,
        column: ConfusionColumn,
        record_index: usize,
    ) -> Result<(), EvalError> {
        let col = self
            .column_index(column)
            .ok_or_else(|| EvalError::Internal {
                detail: format!(
                    "confusion matrix column out of range at record index {record_index}"
                ),
            })?;
        let cell = self
            .rows
            .get_mut(gold_index)
            .and_then(|row| row.get_mut(col))
            .ok_or_else(|| EvalError::Internal {
                detail: format!(
                    "confusion matrix cell out of range at record index {record_index}"
                ),
            })?;
        *cell = cell.checked_add(1).ok_or_else(|| EvalError::Internal {
            detail: format!("confusion matrix count overflow at record index {record_index}"),
        })?;
        Ok(())
    }

    /// 行数（ラベル数）。
    pub fn n_labels(&self) -> usize {
        self.n_labels
    }
}

/// Macro-F1 と、平均から除いたラベルの列挙（REQ-24 異常系・TASK-24.2・issue #61）。
///
/// 除外の条件は「F1 が定義できない（[`LabelMetrics::f1`] が `None`。
/// `tp+fp+fn_ == 0`）」ラベルだけで、`precision`・`recall` 単独の `None`
/// （予測 0 件・正解 0 件）では除外しない。PoC-9 manifest「11. 凍結前の
/// 修正」（2026-09-23）で、この境界を誤って広げた（precision が None の
/// ラベルまで除外した）ことで既知解データセットの Macro-F1 が
/// 49/78（誤り）になった経緯があり、本実装ではこの誤りを再現しない。
///
/// フィールドは非公開にし、構築は [`evaluate_single_select`] 内に集約する
/// （壊れた値、例えば `value.is_some()` なのに全ラベルが `excluded_labels`
/// に含まれる状態を外部から作らせない）。
///
/// JSON への写し方（直列化は CLI の `evaluate` 工程・TASK-33.1 が担う）:
/// `{"macro_f1": {"value": <number|null>, "excluded_labels": [...]}}`。
/// `value` が `None` のときは `null`。`excluded_labels` は宣言順。
#[derive(Debug, Clone, PartialEq)]
pub struct MacroF1 {
    value: Option<f64>,
    excluded_labels: Vec<String>,
}

impl MacroF1 {
    /// F1 が定義できたラベルだけの算術平均。1 つも定義できなければ `None`
    /// （評価契約: 分母 0 の指標は null。[`EvalRecord`] が 1 件以上あり、
    /// gold が必ずラベル集合に含まれるため、support ≥ 1 のラベルが
    /// 少なくとも 1 つ存在し、実際には到達しない分岐だが、型の誠実さの
    /// ため `Option` のまま残す）。
    pub fn value(&self) -> Option<f64> {
        self.value
    }

    /// 平均から除いたラベル（宣言順）。F1 が定義できたラベルだけを
    /// 使う場合は空になる。
    pub fn excluded_labels(&self) -> &[String] {
        &self.excluded_labels
    }
}

/// 単一選択（single-select）の評価指標一式。
#[derive(Debug, Clone, PartialEq)]
pub struct SingleSelectMetrics {
    /// 評価件数（`records.len()`）。
    pub n_total: u64,
    /// outcome の分類件数。
    pub outcome_counts: OutcomeCounts,
    /// 正解率。
    pub accuracy: Accuracy,
    /// ラベル別指標（宣言順）。
    pub per_label: Vec<LabelMetrics>,
    /// Macro-F1 と除外ラベルの列挙（REQ-24 異常系・TASK-24.2・issue #61）。
    pub macro_f1: MacroF1,
    /// 混同行列。
    pub confusion: ConfusionMatrix,
    /// 型と意味の正しさの分離集計（REQ-24 境界値・TASK-24.3・issue #62）。
    /// 詳細な定義は [`crate::quadrant`] モジュールのドキュメントコメント参照。
    pub type_meaning_quadrant: crate::quadrant::TypeMeaningQuadrant,
}

/// 正解率・ラベル別指標・Macro-F1・混同行列を算出する（REQ-24 正常系・TASK-24.1-1）。
///
/// - `labels`: 宣言順のラベル ID（`fandhe-edge-core::Definition::options()` の
///   `id` を宣言順に渡す想定。空・空 ID・重複は [`EvalError`] を返す）
/// - `records`: 評価 1 件ずつの gold・推論結果（0 件は [`EvalError::EmptyRecords`]）
///
/// 混同行列・`per_label` の行・列の並びは `labels` の宣言順に従う
/// （PoC-9 のアルファベット順ソートとは異なる。`Definition::options` が
/// 宣言順を保つ方針〔PoC-9 追補 A-10〕に合わせるため）。
///
/// 乱数は使わず、結果は `records` の走査順に依存しない（決定的）。
pub fn evaluate_single_select(
    labels: &[&str],
    records: &[EvalRecord],
) -> Result<SingleSelectMetrics, EvalError> {
    if labels.is_empty() {
        return Err(EvalError::EmptyLabels);
    }
    if records.is_empty() {
        return Err(EvalError::EmptyRecords);
    }
    // REQ-39（ガード層・資源の上限）: `labels` 件数の検証は索引（`BTreeMap`）
    // 構築より前に行う。索引構築自体が `labels` 全件の挿入・文字列比較を伴う
    // ため、`ConfusionMatrix::new` の確保前検証（後段）だけでは上限超過の
    // 入力でも索引構築が先に走ってしまう（codex/review 指摘）。
    if labels.len() > MAX_LABELS {
        return Err(EvalError::TooManyLabels {
            n_labels: labels.len(),
            limit: MAX_LABELS,
        });
    }

    // ラベル ID → 宣言順の添字。`BTreeMap` を使い、`HashMap` によるハッシュ順の
    // 非決定性を避ける（.claude/rules/coding-rust.md「数値・決定性」）。
    let mut label_index: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, &label) in labels.iter().enumerate() {
        if label.is_empty() {
            return Err(EvalError::EmptyLabelId);
        }
        if label_index.insert(label, i).is_some() {
            return Err(EvalError::DuplicateLabel {
                label: label.to_string(),
            });
        }
    }

    let n_labels = labels.len();
    let mut confusion = ConfusionMatrix::new(n_labels)?;
    let mut support = vec![0u64; n_labels];
    let mut predicted_count = vec![0u64; n_labels];
    let mut outcome_counts = OutcomeCounts::default();
    let mut correct: u64 = 0;
    let mut quadrant = crate::quadrant::TypeMeaningQuadrant::default();

    // checked 演算のオーバーフロー時に埋める `EvalError`。record 位置由来だが
    // 「正解ラベル未知」ではなく内部の集計不整合なので `Internal` を使う
    // （`UnknownGoldLabel` は record 中の `gold` がラベル集合に無い場合専用）。
    let overflow_at = |index: usize, what: &str| EvalError::Internal {
        detail: format!("{what} overflow at record index {index}"),
    };

    for (index, record) in records.iter().enumerate() {
        let gold_index = *label_index
            .get(record.gold)
            .ok_or(EvalError::UnknownGoldLabel { index })?;
        let support_slot = support
            .get_mut(gold_index)
            .ok_or_else(|| EvalError::Internal {
                detail: format!("support index out of range at record index {index}"),
            })?;
        *support_slot = support_slot
            .checked_add(1)
            .ok_or_else(|| overflow_at(index, "support"))?;

        let column = match record.outcome {
            Outcome::Label(predicted) => match label_index.get(predicted.as_str()) {
                Some(&predicted_index) => {
                    outcome_counts.ok = outcome_counts
                        .ok
                        .checked_add(1)
                        .ok_or_else(|| overflow_at(index, "outcome_counts.ok"))?;
                    if predicted_index == gold_index {
                        correct = correct
                            .checked_add(1)
                            .ok_or_else(|| overflow_at(index, "correct"))?;
                    }
                    let slot = predicted_count.get_mut(predicted_index).ok_or_else(|| {
                        EvalError::Internal {
                            detail: format!(
                                "predicted_count index out of range at record index {index}"
                            ),
                        }
                    })?;
                    *slot = slot
                        .checked_add(1)
                        .ok_or_else(|| overflow_at(index, "predicted_count"))?;
                    ConfusionColumn::Label(predicted_index)
                }
                None => {
                    // ラベル集合に無い予測（未知ラベル・空文字列）は invalid 扱い
                    // （PoC-9 既知解 ss-12 "E"・ss-18 "" と同じ扱い）。
                    outcome_counts.invalid = outcome_counts
                        .invalid
                        .checked_add(1)
                        .ok_or_else(|| overflow_at(index, "outcome_counts.invalid"))?;
                    ConfusionColumn::Invalid
                }
            },
            Outcome::Invalid => {
                outcome_counts.invalid = outcome_counts
                    .invalid
                    .checked_add(1)
                    .ok_or_else(|| overflow_at(index, "outcome_counts.invalid"))?;
                ConfusionColumn::Invalid
            }
            Outcome::Abstain => {
                outcome_counts.abstain = outcome_counts
                    .abstain
                    .checked_add(1)
                    .ok_or_else(|| overflow_at(index, "outcome_counts.abstain"))?;
                ConfusionColumn::Abstain
            }
            Outcome::Error => {
                outcome_counts.error = outcome_counts
                    .error
                    .checked_add(1)
                    .ok_or_else(|| overflow_at(index, "outcome_counts.error"))?;
                ConfusionColumn::Error
            }
        };

        confusion.increment(gold_index, column, index)?;

        // 型と意味の正しさの分離集計（REQ-24 境界値・TASK-24.3）。
        // `column` は直前の match で決定済みの分類（ラベル一致・型不正・
        // abstain・error）を再利用し、判定基準を重複させない。
        let cell = match column {
            ConfusionColumn::Label(predicted_index) if predicted_index == gold_index => {
                crate::quadrant::TypeMeaningCell::TypeOkMeaningOk
            }
            ConfusionColumn::Label(_) => crate::quadrant::TypeMeaningCell::TypeOkMeaningNg,
            ConfusionColumn::Invalid => crate::quadrant::TypeMeaningCell::TypeNg,
            ConfusionColumn::Abstain => crate::quadrant::TypeMeaningCell::Abstain,
            ConfusionColumn::Error => crate::quadrant::TypeMeaningCell::Error,
        };
        quadrant.increment(cell, index)?;
    }

    // fail-closed: 5 セルの合計が評価件数と一致しない場合は内部不整合として
    // 拒否する（評価済みを装わない。`.claude/rules/coding-rust.md`）。
    let n_total: u64 = records.len() as u64;
    if quadrant.total() != Some(n_total) {
        return Err(EvalError::Internal {
            detail: "type_meaning_quadrant total mismatch".to_string(),
        });
    }

    // ラベル添字 `i` に由来する内部不整合（`support`・`predicted_count` は
    // `n_labels` 件で確保済みのため理論上到達しないが、外部入力の経路では
    // `[]` を使わず fail-closed で扱う。`records` 内の位置とは無関係なので
    // `EvalError::UnknownGoldLabel` ではなく `Internal` を使う）。
    let label_internal = |label_index: usize, what: &str| EvalError::Internal {
        detail: format!("{what} out of range at label index {label_index}"),
    };

    let mut per_label = Vec::with_capacity(n_labels);
    let mut f1_sum = 0.0f64;
    let mut f1_count: u64 = 0;
    // F1 が定義できなかった（`tp+fp+fn_ == 0`）ラベルの宣言順の列挙。
    // 件数は検証済みの `n_labels`（MAX_LABELS 以下）を超えない。
    let mut excluded_labels: Vec<String> = Vec::new();
    for (i, &label) in labels.iter().enumerate() {
        let tp = confusion
            .get(i, ConfusionColumn::Label(i))
            .ok_or_else(|| label_internal(i, "confusion"))?;
        let support_i = *support.get(i).ok_or_else(|| label_internal(i, "support"))?;
        let predicted_i = *predicted_count
            .get(i)
            .ok_or_else(|| label_internal(i, "predicted_count"))?;
        let fp = predicted_i
            .checked_sub(tp)
            .ok_or_else(|| label_internal(i, "false_positive"))?;
        let fn_ = support_i
            .checked_sub(tp)
            .ok_or_else(|| label_internal(i, "false_negative"))?;

        let precision = if predicted_i == 0 {
            None
        } else {
            Some(tp as f64 / predicted_i as f64)
        };
        let recall = if support_i == 0 {
            None
        } else {
            Some(tp as f64 / support_i as f64)
        };
        // F1 = 2TP / (2TP + FP + FN)。適合率・再現率の調和平均ではない
        // （モジュール冒頭のドキュメントコメント参照）。
        let f1_denominator = tp
            .checked_mul(2)
            .and_then(|v| v.checked_add(fp))
            .and_then(|v| v.checked_add(fn_))
            .ok_or_else(|| label_internal(i, "f1_denominator"))?;
        let f1 = if f1_denominator == 0 {
            None
        } else {
            let numerator = (tp as f64) * 2.0;
            Some(numerator / f1_denominator as f64)
        };
        if let Some(f1_value) = f1 {
            f1_sum += f1_value;
            f1_count = f1_count
                .checked_add(1)
                .ok_or_else(|| label_internal(i, "f1_count"))?;
        } else {
            // 除外条件は f1 == None（tp+fp+fn_ == 0）だけ。precision・recall
            // 単独の None では除外しない（MacroF1 のドキュメント参照）。
            excluded_labels.push(label.to_string());
        }

        per_label.push(LabelMetrics {
            label: label.to_string(),
            support: support_i,
            predicted_count: predicted_i,
            tp,
            fp,
            fn_,
            precision,
            recall,
            f1,
        });
    }

    let macro_f1_value = if f1_count == 0 {
        None
    } else {
        Some(f1_sum / f1_count as f64)
    };
    let macro_f1 = MacroF1 {
        value: macro_f1_value,
        excluded_labels,
    };

    let overall = Ratio::new(correct, n_total).ok_or(EvalError::EmptyRecords)?;
    let adopted_denominator = n_total
        .checked_sub(outcome_counts.abstain)
        .ok_or(EvalError::EmptyRecords)?;
    let adopted_decision = Ratio::new(correct, adopted_denominator);

    Ok(SingleSelectMetrics {
        n_total,
        outcome_counts,
        accuracy: Accuracy {
            overall,
            adopted_decision,
        },
        per_label,
        macro_f1,
        confusion,
        type_meaning_quadrant: quadrant,
    })
}

/// 浮動小数を許容差 1e-9 で比較する（テスト専用。評価契約の許容差に合わせる）。
#[cfg(test)]
fn approx_eq(a: f64, b: f64) -> bool {
    const FLOAT_EPSILON: f64 = 1e-9;
    (a - b).abs() < FLOAT_EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-24・TASK-24.1-1: 空のラベル集合はエラーになる。
    #[test]
    fn empty_labels_is_error() {
        let records: Vec<EvalRecord> = vec![];
        let outcome = Outcome::Label("A".to_string());
        let record = EvalRecord {
            gold: "A",
            outcome: &outcome,
        };
        let records_with_one = vec![record];
        let result = evaluate_single_select(&[], &records_with_one);
        assert_eq!(result, Err(EvalError::EmptyLabels));
        // records が空でも labels が空ならまず EmptyLabels を返す。
        let result_both_empty = evaluate_single_select(&[], &records);
        assert_eq!(result_both_empty, Err(EvalError::EmptyLabels));
    }

    /// REQ-24・TASK-24.1-1: ラベル ID の重複はエラーになる。
    #[test]
    fn duplicate_label_is_error() {
        let outcome = Outcome::Label("A".to_string());
        let records = vec![EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        let result = evaluate_single_select(&["A", "B", "A"], &records);
        assert_eq!(
            result,
            Err(EvalError::DuplicateLabel {
                label: "A".to_string()
            })
        );
    }

    /// REQ-24・TASK-24.1-1: 空文字列のラベル ID はエラーになる。
    #[test]
    fn empty_label_id_is_error() {
        let outcome = Outcome::Label("A".to_string());
        let records = vec![EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        let result = evaluate_single_select(&["A", ""], &records);
        assert_eq!(result, Err(EvalError::EmptyLabelId));
    }

    /// REQ-24・TASK-24.1-1: 評価件数 0 はエラーになる（0 除算を避け、評価済みを装わない）。
    #[test]
    fn empty_records_is_error() {
        let records: Vec<EvalRecord> = vec![];
        let result = evaluate_single_select(&["A", "B"], &records);
        assert_eq!(result, Err(EvalError::EmptyRecords));
    }

    /// REQ-39・TASK-24.1-1（codex/review 指摘対応）: ラベル数が [`MAX_LABELS`]
    /// を超える場合、混同行列を確保する前に `TooManyLabels` で拒否する。
    #[test]
    fn too_many_labels_is_rejected_before_allocation() {
        let owned_labels: Vec<String> = (0..=MAX_LABELS).map(|i| format!("L{i}")).collect();
        let labels: Vec<&str> = owned_labels.iter().map(String::as_str).collect();
        let outcome = Outcome::Label("L0".to_string());
        let records = vec![EvalRecord {
            gold: "L0",
            outcome: &outcome,
        }];
        let result = evaluate_single_select(&labels, &records);
        assert_eq!(
            result,
            Err(EvalError::TooManyLabels {
                n_labels: MAX_LABELS + 1,
                limit: MAX_LABELS,
            })
        );
    }

    /// REQ-39・TASK-24.1-1: ちょうど [`MAX_LABELS`] 件のラベルは許容される
    /// （境界値。超過のみを拒否し、上限そのものは正常に処理できることを確認する）。
    #[test]
    fn max_labels_boundary_is_accepted() {
        let owned_labels: Vec<String> = (0..MAX_LABELS).map(|i| format!("L{i}")).collect();
        let labels: Vec<&str> = owned_labels.iter().map(String::as_str).collect();
        let outcome = Outcome::Label("L0".to_string());
        let records = vec![EvalRecord {
            gold: "L0",
            outcome: &outcome,
        }];
        let result = evaluate_single_select(&labels, &records);
        assert!(result.is_ok());
    }

    /// REQ-24・TASK-24.1-1: 未知の正解ラベルはインデックス付きでエラーになる。
    #[test]
    fn unknown_gold_label_reports_index() {
        let outcome_a = Outcome::Label("A".to_string());
        let outcome_z = Outcome::Label("A".to_string());
        let records = vec![
            EvalRecord {
                gold: "A",
                outcome: &outcome_a,
            },
            EvalRecord {
                gold: "Z",
                outcome: &outcome_z,
            },
        ];
        let result = evaluate_single_select(&["A", "B"], &records);
        assert_eq!(result, Err(EvalError::UnknownGoldLabel { index: 1 }));
    }

    /// REQ-24・TASK-24.1-1: 全問正解（2 ラベル）で、正解率 1.0・Macro-F1 1.0・
    /// 混同行列が対角だけになること。
    #[test]
    fn all_correct_two_labels() {
        let out_a = Outcome::Label("A".to_string());
        let out_b = Outcome::Label("B".to_string());
        let records = vec![
            EvalRecord {
                gold: "A",
                outcome: &out_a,
            },
            EvalRecord {
                gold: "B",
                outcome: &out_b,
            },
        ];
        let metrics = evaluate_single_select(&["A", "B"], &records).expect("valid input");
        assert_eq!(metrics.n_total, 2);
        assert!(approx_eq(metrics.accuracy.overall.value(), 1.0));
        assert_eq!(metrics.accuracy.overall.numerator(), 2);
        assert_eq!(metrics.accuracy.overall.denominator(), 2);
        assert!(approx_eq(
            metrics
                .accuracy
                .adopted_decision
                .expect("no abstain")
                .value(),
            1.0
        ));
        assert!(approx_eq(metrics.macro_f1.value().expect("defined"), 1.0));
        assert!(metrics.macro_f1.excluded_labels().is_empty());
        assert_eq!(metrics.confusion.get(0, ConfusionColumn::Label(0)), Some(1));
        assert_eq!(metrics.confusion.get(0, ConfusionColumn::Label(1)), Some(0));
        assert_eq!(metrics.confusion.get(1, ConfusionColumn::Label(0)), Some(0));
        assert_eq!(metrics.confusion.get(1, ConfusionColumn::Label(1)), Some(1));
        // REQ-24 境界値・TASK-24.3: 全問正解なので type_ok_meaning_ok=2、
        // 他のセルは 0。
        assert_eq!(metrics.type_meaning_quadrant.type_ok_meaning_ok(), 2);
        assert_eq!(metrics.type_meaning_quadrant.type_ok_meaning_ng(), 0);
        assert_eq!(metrics.type_meaning_quadrant.type_ng_count(), 0);
        assert_eq!(metrics.type_meaning_quadrant.abstain(), 0);
        assert_eq!(metrics.type_meaning_quadrant.error(), 0);
        assert_eq!(metrics.type_meaning_quadrant.total(), Some(2));
    }

    /// REQ-24・TASK-24.1-1: 宣言順（["B","A"]）に混同行列・per_label の並びが従う。
    #[test]
    fn declaration_order_controls_row_column_order() {
        let out_a = Outcome::Label("A".to_string());
        let out_b = Outcome::Label("B".to_string());
        let records = vec![
            EvalRecord {
                gold: "A",
                outcome: &out_a,
            },
            EvalRecord {
                gold: "B",
                outcome: &out_b,
            },
        ];
        let metrics = evaluate_single_select(&["B", "A"], &records).expect("valid input");
        // 宣言順 B, A なので per_label[0] は B、per_label[1] は A。
        assert_eq!(metrics.per_label[0].label, "B");
        assert_eq!(metrics.per_label[1].label, "A");
        // gold="B" は宣言順で行 0、gold="A" は行 1。
        assert_eq!(metrics.confusion.get(0, ConfusionColumn::Label(0)), Some(1)); // B行・B列
        assert_eq!(metrics.confusion.get(1, ConfusionColumn::Label(1)), Some(1)); // A行・A列
    }

    /// REQ-24・TASK-24.1-1: 未知ラベルへの予測は invalid 列へ数える。
    #[test]
    fn unknown_predicted_label_counts_as_invalid() {
        let out_unknown = Outcome::Label("Z".to_string());
        let records = vec![EvalRecord {
            gold: "A",
            outcome: &out_unknown,
        }];
        let metrics = evaluate_single_select(&["A", "B"], &records).expect("valid input");
        assert_eq!(metrics.outcome_counts.invalid, 1);
        assert_eq!(metrics.confusion.get(0, ConfusionColumn::Invalid), Some(1));
        assert!(approx_eq(metrics.accuracy.overall.value(), 0.0));
        // 評価契約（.claude/rules/evaluation-contract.md「有意性・指標」）:
        // 分母 0 の指標は null（`None`）とし、平均から除外する。
        // A: support=1・tp=0・fp=0・fn_=1 → f1 = 2*0/(2*0+0+1) = 0.0。
        assert_eq!(metrics.per_label[0].label, "A");
        assert_eq!(metrics.per_label[0].support, 1);
        assert_eq!(metrics.per_label[0].f1, Some(0.0));
        // B: support=0（predicted_count も 0）→ precision・recall・f1 いずれも None。
        assert_eq!(metrics.per_label[1].label, "B");
        assert_eq!(metrics.per_label[1].support, 0);
        assert_eq!(metrics.per_label[1].precision, None);
        assert_eq!(metrics.per_label[1].recall, None);
        assert_eq!(metrics.per_label[1].f1, None);
        // macro_f1 は f1 が定義できた A だけの平均（B は除外）。
        assert_eq!(metrics.macro_f1.value(), Some(0.0));
        assert_eq!(metrics.macro_f1.excluded_labels(), ["B".to_string()]);
        // REQ-24 境界値・TASK-24.3: 未知ラベルへの予測は type_ng_count=1。
        // 意味の 2 セルには数えない（型不正の行は意味を判定できないため）。
        assert_eq!(metrics.type_meaning_quadrant.type_ng_count(), 1);
        assert_eq!(metrics.type_meaning_quadrant.type_ok_meaning_ok(), 0);
        assert_eq!(metrics.type_meaning_quadrant.type_ok_meaning_ng(), 0);
        assert_eq!(metrics.type_meaning_quadrant.total(), Some(1));
    }

    /// REQ-24 異常系・TASK-24.2: `excluded_labels` は宣言順に列挙され、
    /// アルファベット順にはソートされない。
    #[test]
    fn excluded_labels_follow_declaration_order() {
        // 宣言順は C, A, B。C・B は gold にも予測にも一度も現れず f1 が
        // None になる（tp+fp+fn_ == 0）ため除外され、A だけで評価する。
        let out_a = Outcome::Label("A".to_string());
        let records = vec![EvalRecord {
            gold: "A",
            outcome: &out_a,
        }];
        let metrics = evaluate_single_select(&["C", "A", "B"], &records).expect("valid input");
        assert_eq!(
            metrics.macro_f1.excluded_labels(),
            ["C".to_string(), "B".to_string()],
            "宣言順（C, A, B のうち A を除く）で列挙されること"
        );
        assert_eq!(metrics.macro_f1.value(), Some(1.0));
    }

    /// REQ-24 異常系・TASK-24.2: 正解が 1 件も無く予測だけあるラベルは、
    /// recall が None・precision が Some(0.0) になるが、f1 は Some(0.0) の
    /// ため `excluded_labels` には含まれない（除外条件は f1 == None のみ）。
    #[test]
    fn label_predicted_but_absent_from_gold_is_not_excluded() {
        // gold は常に "A"。"B" は正解に一度も現れないが 1 回予測される。
        let out_a = Outcome::Label("A".to_string());
        let out_b = Outcome::Label("B".to_string());
        let records = vec![
            EvalRecord {
                gold: "A",
                outcome: &out_a,
            },
            EvalRecord {
                gold: "A",
                outcome: &out_b,
            },
        ];
        let metrics = evaluate_single_select(&["A", "B"], &records).expect("valid input");
        let b = &metrics.per_label[1];
        assert_eq!(b.label, "B");
        assert_eq!(b.recall, None, "B は support=0 のため recall は None");
        assert_eq!(
            b.precision,
            Some(0.0),
            "B は predicted_count=1・tp=0 のため precision は Some(0.0)"
        );
        assert_eq!(
            b.f1,
            Some(0.0),
            "B は tp+fp+fn_=1 のため f1 は Some(0.0)（None ではない）"
        );
        assert!(
            !metrics
                .macro_f1
                .excluded_labels()
                .contains(&"B".to_string()),
            "f1 が定義できるラベルは precision・recall が None でも除外されない"
        );
    }
}
