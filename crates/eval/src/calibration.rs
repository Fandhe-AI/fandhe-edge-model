//! 温度スケーリングによる確信度校正（温度 T）と保留しきい値（τ）の算出。
//!
//! REQ-22（保留・確率の校正）のうち、`validation` 分割から T と τ を
//! **決める計算**だけを担当する（TASK-22.1-1・issue #95・親 #94）。保留状態
//! （`Outcome::Abstain`）への接続と保留込み／保留なしの誤り率の比較は
//! [`crate::abstention`]（TASK-22.1-2・issue #96）、「対象外」ラベルによる
//! 扱い（TASK-22.2・issue #97）も同モジュールが担当し、coverage の記録と表示
//! （TASK-22.3）は [`crate::coverage`] が担当する。いずれも本モジュールの対象外。
//!
//! 移植元は PoC-12 の `03-poc/abstention-calibration/scripts/calibrate.py`
//! （事前登録の 3〜4 節）。本実装は次の点で PoC-12 から変更している。
//!
//! 1. **T の探索を bounded Brent ではなく導関数の二分法にした**。
//!    scipy が本リポに無く（依存の追加なし）、`xatol` 由来の探索誤差では
//!    1e-9 の具体値照合ができないため。NLL は `β = 1/T` について凸関数
//!    （log-sum-exp が凸、残りは線形）であり、導関数 `h(β)` は単調非減少。
//!    閉形式の解と 1e-9 以内で一致することをテストで確認する
//!    （PoC のログにある `T*` との一致は検証しない。テスト内で明記）。
//! 2. **ロジットは `d_k = z_k − max_k z_k`（最大値を引いた差分）で扱う**。
//!    `β·z_k` を直接使うと、有限だが絶対値の大きいロジットで
//!    `exp(β·z_k)` がオーバーフローしうる。差分を取れば `d_k <= 0` が
//!    常に成り立ち、`exp(β·d_k)` は `(0, 1]` に収まる。
//! 3. **`f64::NEG_INFINITY` を正当な入力として受け付ける**（学習に出てこな
//!    かったクラスのロジット。PoC-12 `c1_logits_to_full` と同じ想定）。
//!    NaN・`+∞` は拒否する（fail-closed。REQ-23 不正なスコアの扱い）。
//!    `d_k` の計算で `−∞` の項は確率 0 として自然に扱われるが、
//!    `Σ p_k·d_k` では `0 × (−∞) = NaN` になるため明示的に飛ばす。
//! 4. **gold のロジットが `−∞` の行は導関数に寄与させない**。この行は
//!    どの T でも `p_gold = 0` になるため、勾配の平均には含めない
//!    （分母から除外）。報告用の NLL にはこの行 1 件につき PoC-12 と同じ
//!    定数 `−ln(`[`NLL_CLIP_EPSILON`]`)` を加える。ECE・τ の計算では通常の
//!    行として扱う（argmax・top1 確率は通常どおり計算できるため）。
//! 5. **gold のロジットが有限の行では `clip(p, 1e-12)` を適用しない**。
//!    log-sum-exp で `−log p_gold` を正確に計算するため `log(0)` は
//!    発生しない（PoC-12 は `clip` していたため、`p_gold < 1e-12` の
//!    極端な行でだけ値が変わりうる）。
//!
//! # 評価契約との関係（REQ-17・REQ-27）
//!
//! - 呼び出し側は **`validation` 分割のみ**を渡すこと。評価データ・最終 test
//!   を渡して T・τ を選び直してはならない（型では強制できない制約であり、
//!   呼び出し側の責務として明記する）。
//! - 入力（[`CalibrationRecord`]）は `&` 参照でのみ受け取り、書き換えない。
//! - 乱数・`HashMap` は使わない（`Vec`・宣言順の走査のみ。決定的）。
//!
//! # 資源上限（REQ-39）
//!
//! [`MAX_CALIBRATION_CELLS`] は `n_records * n_labels` の **計算量**の上限
//! （二分法の各反復で全行を走査するため）であり、メモリ確保量の上限では
//! ない。実際に確保するのは行ごとの `d` ベクトル（`n_labels` 件）と
//! top1 確率の `Vec<f64>`（`n_records` 件）で、行ごとの確率行列は保持し
//! ない。件数はいずれも確保・計算の前に検証する（`unwrap`・`expect`・
//! `[]` 添字アクセスは使わない）。
//!
//! 確信度が τ 未満の入力を `Outcome::Abstain` として保留込み／保留なしの
//! 誤り率を比べる処理は [`crate::abstention`]（TASK-22.1-2・issue #96）が
//! 本モジュールの [`Calibration`]・[`top1_probability`]・[`preprocess_logits`]
//! を再利用して実装する。
//!
//! # 対象外
//!
//! - 「対象外」ラベルによる処理は [`crate::abstention`] に実装済み
//!   （TASK-22.2・issue #97）。本モジュールの T・τ の選び方は対象外ラベルの
//!   有無で変えない
//! - T・τ の永続化・配布パッケージへの格納（REQ-30）・CLI への配線（issue #140）

use std::fmt;

use crate::metrics::{self, EvalError, Ratio};
use crate::significance;

/// 温度スケーリングの探索範囲の下限（PoC-12 事前登録）。
pub const TEMPERATURE_MIN: f64 = 0.05;
/// 温度スケーリングの探索範囲の上限（PoC-12 事前登録）。
pub const TEMPERATURE_MAX: f64 = 20.0;
/// ECE（Expected Calibration Error）の固定ビン数（等幅・最後のビンだけ 1.0 を含む）。
pub const ECE_BINS: usize = 15;
/// gold のロジットが `−∞` の行の報告用 NLL に加える定数（`−ln(1e-12)`）。
/// PoC-12 の `clip(p, 1e-12)` と同じ下限確率に対応する。
pub const NLL_CLIP_EPSILON: f64 = 1e-12;
/// τ の目標 coverage（80%。C-7 のユーザー回答で確定。定数のため引数には
/// しない）。[`select_threshold`] のドキュメントコメント参照: 実際の τ は
/// `numpy.quantile(q=0.20, method="lower")` に一致させる規則で決まり、
/// 常に coverage ≥ 80% を満たすが、`n` が 5 の倍数のとき、その条件を満たす
/// 最大の τ より 1 段低い値になりうる（PoC-12 の規則をそのまま踏襲する）。
pub(crate) const TARGET_COVERAGE_DENOMINATOR: usize = 5;
/// 二分法の反復回数の上限（決定的な停止条件。無限ループを作らない。REQ-39）。
const MAX_BISECTION_ITERATIONS: u32 = 200;

/// `n_records * n_labels`（二分法の各反復で走査するセル数）の上限。
///
/// メモリ確保量の上限ではなく、二分法（最大 [`MAX_BISECTION_ITERATIONS`]
/// 回）× ECE・NLL・τ の各パスで生じる計算量の上限。呼び出し側の定義ファイル
/// 検査・データ契約層の検証を通った入力でも、巨大な行数×ラベル数の組は
/// 単一の校正計算で長時間占有しうるため、計算前に拒否する
/// （REQ-39 ガード層「資源の上限」）。
pub const MAX_CALIBRATION_CELLS: usize = 50_000_000;

/// 校正 1 件分の入力（validation の 1 行）。所有権は取らない（REQ-27）。
#[derive(Debug, Clone, Copy)]
pub struct CalibrationRecord<'a> {
    /// 正解ラベル ID。
    pub gold: &'a str,
    /// ロジット（softmax 前のスコア）。`labels` の宣言順に並べる。
    /// `f64::NEG_INFINITY` は「学習に出てこなかったクラス」として正当な
    /// 入力（モジュール冒頭 3 節）。NaN・`+∞` は拒否する。
    pub logits: &'a [f64],
}

/// [`calibrate`]・[`select_threshold`] が返しうるエラー。
///
/// 外部入力（ロジット・ラベル集合）の異常を fail-closed で表現し、panic
/// させない（`.claude/rules/coding-rust.md`「エラーハンドリング・外部入力」）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CalibrationError {
    /// ラベル集合の検証エラー（[`crate::metrics::build_label_index`] と共有）。
    InvalidLabels(EvalError),
    /// 評価レコードが 0 件。
    EmptyRecords,
    /// レコード件数が上限（[`significance::MAX_EVAL_RECORDS`]）を超える。
    TooManyRecords {
        /// 渡されたレコード件数。
        n: usize,
        /// 上限。
        limit: usize,
    },
    /// `n_records * n_labels` が [`MAX_CALIBRATION_CELLS`] を超える。
    TooManyCells {
        /// レコード件数。
        n_records: usize,
        /// ラベル数。
        n_labels: usize,
        /// 上限。
        limit: usize,
    },
    /// 正解ラベルがラベル集合に存在しない。
    UnknownGoldLabel {
        /// `records` 内での位置（0 始まり）。
        index: usize,
    },
    /// `logits` の長さがラベル数と一致しない。
    LogitLengthMismatch {
        /// `records` 内での位置（0 始まり）。
        index: usize,
        /// 期待する長さ（ラベル数）。
        expected: usize,
        /// 実際の長さ。
        actual: usize,
    },
    /// ロジットに NaN または `+∞` が含まれる（`−∞` は正当な入力。3 節）。
    NonFiniteLogit {
        /// `records` 内での位置（0 始まり）。
        index: usize,
        /// `labels` 内での位置（0 始まり）。
        label_index: usize,
    },
    /// 1 行に有限のロジットが 1 つも無い（全て `−∞`）。
    NoFiniteLogit {
        /// `records` 内での位置（0 始まり）。
        index: usize,
    },
    /// [`select_threshold`] に渡された top1 確率に NaN・非有限値が含まれる
    /// （[`calibrate`] 内部の呼び出しでは発生しないが、公開関数として
    /// 独立に呼ばれる場合〔`crate::abstention`・TASK-22.3〕の外部入力検証）。
    NonFiniteTop1 {
        /// `top1` スライス内での位置（0 始まり）。
        index: usize,
    },
    /// [`select_threshold`] に渡された top1 確率が確率として不正な範囲
    /// （`0.0..=1.0` の外）。有限だが `-1.0` や `2.0` のような値を確率・
    /// 保留しきい値として扱わない（REQ-23 不正なスコアの扱い）。
    Top1OutOfRange {
        /// `top1` スライス内での位置（0 始まり）。
        index: usize,
    },
    /// top1 確率のスライスが空（0 件からは τ を決定できない）。
    EmptyTop1,
    /// [`select_threshold`] に渡された `top1` の件数が上限
    /// （[`significance::MAX_EVAL_RECORDS`]）を超える。[`calibrate`] を
    /// 経由せず外部から直接呼べる公開関数のため、確保・ソート前に
    /// 件数を検証する（REQ-39 ガード層「資源の上限」）。
    TooManyTop1 {
        /// 渡された件数。
        n: usize,
        /// 上限。
        limit: usize,
    },
    /// 校正計算の中間・最終結果が非有限になった。個々のロジット・
    /// ロジット差分（`d_k`）は有限でも、`β·d_gold` 等の掛け算・累積で
    /// オーバーフローし `±∞` になりうるため、返却前に検証する
    /// （評価契約の fail-closed 原則。REQ-27・`.claude/rules/evaluation-contract.md`）。
    NonFiniteResult {
        /// 発生箇所の説明（人が読める短い文字列。機械照合はしない）。
        detail: String,
    },
    /// 集計中の内部不整合（理論上到達しないが fail-closed のため用意する）。
    Internal {
        /// 発生箇所の説明（人が読める短い文字列。機械照合はしない）。
        detail: String,
    },
    /// [`crate::abstention::compare_abstention`] に渡されたラベル集合の件数が、
    /// `calibration`（[`calibrate`] を呼んだときのラベル集合）と一致しない
    /// （[`Calibration::n_labels`]。REQ-17・REQ-27: 校正した対象と異なるラベル
    /// 集合で評価データを走査し、τ を暗黙に別の意味へ読み替えることを防ぐ）。
    LabelCountMismatch {
        /// 校正時のラベル数（[`Calibration::n_labels`]）。
        calibrated: usize,
        /// 渡されたラベル数。
        given: usize,
    },
    /// [`crate::abstention::compare_abstention`] に渡されたラベル集合が、件数は
    /// 一致するものの宣言順の ID が一致しない（[`Calibration::labels`]。
    /// REQ-17・REQ-27: 件数だけを見ると、同数の別ラベル集合や宣言順を
    /// 並べ替えた集合でも検査を通過してしまい、校正時とは異なる添字を
    /// 予測ラベルとして解釈しうる〔評価の独立性の抵触〕。最初に食い違う
    /// 位置を返し、原因箇所を特定できるようにする）。
    LabelMismatch {
        /// 最初に食い違う宣言順添字。
        index: usize,
        /// 校正時のその位置のラベル ID。
        calibrated: String,
        /// 渡されたその位置のラベル ID。
        given: String,
    },
    /// 対象外ラベルとして指定された ID が、`calibration` のラベル集合に無い
    /// （REQ-22 異常系・TASK-22.2・issue #97。設定誤りを隠さず、黙って
    /// しきい値のみの判定へ戻さない fail-closed）。
    UnknownOutOfScopeLabel {
        /// 指定された対象外ラベル ID。
        id: String,
    },
    /// 対象外ラベルの指定（[`crate::abstention::OutOfScopeLabel`]）が、判定に
    /// 使う `calibration` のラベル集合と食い違う（別の校正結果との取り違え。
    /// REQ-17・REQ-27・TASK-22.2）。
    OutOfScopeLabelMismatch {
        /// 指定時に解決した宣言順添字。
        index: usize,
        /// 指定された対象外ラベル ID。
        expected: String,
        /// 判定に使う `calibration` のその位置のラベル ID（範囲外なら `None`）。
        found: Option<String>,
    },
    /// [`crate::abstention::compare_abstention`] が内部で呼ぶ
    /// [`crate::metrics::evaluate_single_select`] が返したエラー（評価ロジックを
    /// 本モジュール・`abstention` モジュールで再実装せず、評価器の唯一の実装
    /// 〔TASK-24.1〕へ委譲するために包む）。
    Evaluation(EvalError),
    /// [`crate::abstention::decide_abstention_with_parameters`] に渡された温度・しきい値・対象外ラベルの
    /// 添字が範囲外（`infer` が配布パッケージから読んだ値の防御。REQ-22・REQ-39・#497）。
    InvalidParameter {
        /// 範囲外だった引数名。
        name: &'static str,
    },
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalibrationError::InvalidLabels(err) => write!(f, "invalid label set: {err}"),
            CalibrationError::EmptyRecords => write!(f, "calibration records must not be empty"),
            CalibrationError::TooManyRecords { n, limit } => {
                write!(f, "too many calibration records: {n} exceeds limit {limit}")
            }
            CalibrationError::TooManyCells {
                n_records,
                n_labels,
                limit,
            } => write!(
                f,
                "too many calibration cells: {n_records} records * {n_labels} labels exceeds limit {limit}"
            ),
            CalibrationError::UnknownGoldLabel { index } => {
                write!(f, "unknown gold label at record index {index}")
            }
            CalibrationError::LogitLengthMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "logit length mismatch at record index {index}: expected {expected}, got {actual}"
            ),
            CalibrationError::NonFiniteLogit { index, label_index } => write!(
                f,
                "non-finite logit (NaN or +inf) at record index {index}, label index {label_index}"
            ),
            CalibrationError::NoFiniteLogit { index } => {
                write!(f, "no finite logit at record index {index}")
            }
            CalibrationError::NonFiniteTop1 { index } => {
                write!(f, "non-finite top1 probability at index {index}")
            }
            CalibrationError::Top1OutOfRange { index } => {
                write!(
                    f,
                    "top1 probability out of range [0.0, 1.0] at index {index}"
                )
            }
            CalibrationError::EmptyTop1 => write!(f, "top1 slice must not be empty"),
            CalibrationError::TooManyTop1 { n, limit } => {
                write!(f, "too many top1 values: {n} exceeds limit {limit}")
            }
            CalibrationError::NonFiniteResult { detail } => {
                write!(f, "non-finite calibration result: {detail}")
            }
            CalibrationError::Internal { detail } => {
                write!(f, "internal calibration error: {detail}")
            }
            CalibrationError::LabelCountMismatch { calibrated, given } => {
                write!(
                    f,
                    "label count mismatch: calibrated with {calibrated}, got {given}"
                )
            }
            CalibrationError::LabelMismatch {
                index,
                calibrated,
                given,
            } => write!(
                f,
                "label mismatch at index {index}: calibrated with \"{calibrated}\", got \"{given}\""
            ),
            CalibrationError::UnknownOutOfScopeLabel { id } => write!(
                f,
                "out-of-scope label \"{id}\" is not in the calibrated label set"
            ),
            CalibrationError::OutOfScopeLabelMismatch {
                index,
                expected,
                found,
            } => match found {
                Some(found) => write!(
                    f,
                    "out-of-scope label mismatch at index {index}: expected \"{expected}\", calibration has \"{found}\""
                ),
                None => write!(
                    f,
                    "out-of-scope label mismatch at index {index}: expected \"{expected}\", calibration has no such index"
                ),
            },
            CalibrationError::Evaluation(err) => {
                write!(f, "evaluation failed: {err}")
            }
            CalibrationError::InvalidParameter { name } => {
                write!(f, "invalid calibration parameter: {name}")
            }
        }
    }
}

impl std::error::Error for CalibrationError {}

/// 温度 T・しきい値 τ の校正結果。
///
/// フィールドは非公開にし、構築は [`calibrate`] 内に集約する（壊れた値
/// ―― 例えば `adopted() == true` なのに `chosen_temperature() != temperature_star()`
/// ―― を外部から作らせない）。
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    temperature_star: f64,
    chosen_temperature: f64,
    adopted: bool,
    threshold: f64,
    validation_coverage: Ratio,
    nll_t1: f64,
    nll_t_star: f64,
    ece_t1: f64,
    ece_t_star: f64,
    n_validation: u64,
    /// 校正時のラベル ID（宣言順、[`calibrate`] に渡した `labels` と同じ
    /// 並び）。件数だけでなく ID・宣言順そのものを保持し、
    /// [`crate::abstention::compare_abstention`] が評価データのラベル集合と
    /// 同一性で照合できるようにする（REQ-17・REQ-27。codex/review 指摘
    /// 対応: 件数一致だけでは同数の別ラベル集合・並べ替えを見逃す）。
    labels: Vec<String>,
}

impl Calibration {
    /// 二分法（または端点判定）で求めた温度 T*（採否によらない）。
    pub fn temperature_star(&self) -> f64 {
        self.temperature_star
    }

    /// 実際に採用する温度（`adopted()` が真なら `temperature_star()`、
    /// 偽なら `1.0`）。
    pub fn chosen_temperature(&self) -> f64 {
        self.chosen_temperature
    }

    /// T* を採用したか（`ECE(T*) < ECE(1)` の厳密な `<` で判定。同値なら
    /// 採用しない）。
    pub fn adopted(&self) -> bool {
        self.adopted
    }

    /// 保留しきい値 τ（`chosen_temperature()` での top1 確率の分位点）。
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// validation での coverage（`chosen_temperature()` の下で top1 ≥ τ の割合）。
    pub fn validation_coverage(&self) -> Ratio {
        self.validation_coverage
    }

    /// T=1（校正なし）での平均 NLL。
    pub fn nll_t1(&self) -> f64 {
        self.nll_t1
    }

    /// T* での平均 NLL（採否によらず T* そのものでの値）。
    pub fn nll_t_star(&self) -> f64 {
        self.nll_t_star
    }

    /// T=1（校正なし）での ECE。
    pub fn ece_t1(&self) -> f64 {
        self.ece_t1
    }

    /// T* での ECE（採否によらず T* そのものでの値）。
    pub fn ece_t_star(&self) -> f64 {
        self.ece_t_star
    }

    /// 校正に使った validation の件数。
    pub fn n_validation(&self) -> u64 {
        self.n_validation
    }

    /// 校正に使ったラベル数（[`calibrate`] に渡した `labels.len()`）。
    ///
    /// [`crate::abstention::compare_abstention`] が、評価データに渡された
    /// ラベル集合の件数がこの校正結果と一致するかを確認するために使う
    /// （REQ-17・REQ-27: 校正した対象と異なるラベル集合を暗黙に混ぜない）。
    pub fn n_labels(&self) -> usize {
        self.labels.len()
    }

    /// 校正時のラベル ID（宣言順、[`calibrate`] に渡した `labels` と同じ並び）。
    ///
    /// [`crate::abstention::compare_abstention`] が、評価データに渡された
    /// ラベル集合を件数だけでなく ID・宣言順の同一性で照合するために使う
    /// （REQ-17・REQ-27）。
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// `1.0 / chosen_temperature()`（β を求める式を 1 箇所に集約する。
    /// [`beta_of_temperature`] を使う）。[`crate::abstention`] が確信度
    /// （校正後の top1 確率）を計算する際に [`calibrate`] 内部と同じ β を
    /// 使うために `pub(crate)` で公開する。
    pub(crate) fn chosen_beta(&self) -> f64 {
        beta_of_temperature(self.chosen_temperature)
    }
}

/// `β = 1/T`（温度からベータへの変換式を 1 箇所に集約する）。
pub(crate) fn beta_of_temperature(temperature: f64) -> f64 {
    1.0 / temperature
}

/// 1 行分の前処理済みロジット。`d[k] = logits[k] - max`（有限値のみ）で、
/// `−∞` だった要素は `f64::NEG_INFINITY` のまま保持する。
struct RowData {
    /// 最大値を引いた差分（宣言順、`labels` と同じ長さ）。
    d: Vec<f64>,
    /// 正解ラベルの宣言順添字。
    gold_index: usize,
    /// `d[gold_index]` が有限なら `Some`（導関数・NLL の通常項に使う）。
    d_gold_if_finite: Option<f64>,
    /// argmax の宣言順添字（同値は先頭を採用。numpy と同じ）。
    argmax_index: usize,
}

/// 1 行分のロジットの検証・前処理（3〜5 節の入力規則）のうち、**gold に
/// 依存しない部分**（長さ検証・NaN/`+∞` 拒否・`d = z − max` の差分・argmax
/// の決定・桁あふれ検出）を担う。[`build_row`]（[`calibrate`] 内部・gold 付き）
/// と [`crate::abstention`] の判定（gold 無し。REQ-27: 推論関数へは `input`
/// 由来の情報だけを渡す）が同じロジットの解釈を共有するための共通経路。
///
/// 戻り値は `(d, argmax_index)`。`d` は宣言順・`logits` と同じ長さで、
/// 有限要素は `logits[k] - max_finite`、`−∞` だった要素は
/// `f64::NEG_INFINITY` のまま保持する。
pub(crate) fn preprocess_logits(
    index: usize,
    logits: &[f64],
    n_labels: usize,
) -> Result<(Vec<f64>, usize), CalibrationError> {
    if logits.len() != n_labels {
        return Err(CalibrationError::LogitLengthMismatch {
            index,
            expected: n_labels,
            actual: logits.len(),
        });
    }

    let mut max_finite = f64::NEG_INFINITY;
    let mut argmax_index: Option<usize> = None;
    for (label_pos, &z) in logits.iter().enumerate() {
        if z.is_nan() || (z.is_infinite() && z.is_sign_positive()) {
            return Err(CalibrationError::NonFiniteLogit {
                index,
                label_index: label_pos,
            });
        }
        if z.is_finite() && z > max_finite {
            max_finite = z;
            argmax_index = Some(label_pos);
        }
    }
    let argmax_index = argmax_index.ok_or(CalibrationError::NoFiniteLogit { index })?;

    let mut d: Vec<f64> = Vec::with_capacity(logits.len());
    for (label_pos, &z) in logits.iter().enumerate() {
        if z.is_finite() {
            let diff = z - max_finite;
            // 有限のロジット同士の減算が `±∞` に桁あふれした場合（例:
            // `z = -f64::MAX`・`max_finite = f64::MAX`）、この後段は
            // `diff.is_finite()` を「gold のロジットが正当な `−∞` 入力
            // だった」ケースと区別できず、silently 除外（確率 0 扱い）
            // してしまう。REQ-27 の fail-closed 原則に反するため、有限
            // 入力から非有限差分が生じた時点でここで拒否する。
            if !diff.is_finite() {
                return Err(CalibrationError::NonFiniteResult {
                    detail: format!(
                        "finite logit difference overflowed to non-finite at record index {index}, label index {label_pos} (logit={z}, max_finite={max_finite})"
                    ),
                });
            }
            d.push(diff);
        } else {
            d.push(z);
        }
    }

    Ok((d, argmax_index))
}

/// 1 行の検証・前処理（gold を引いて [`RowData`] を組み立てる）。
fn build_row(
    index: usize,
    record: &CalibrationRecord,
    label_index: &std::collections::BTreeMap<&str, usize>,
    n_labels: usize,
) -> Result<RowData, CalibrationError> {
    let gold_index = *label_index
        .get(record.gold)
        .ok_or(CalibrationError::UnknownGoldLabel { index })?;
    let (d, argmax_index) = preprocess_logits(index, record.logits, n_labels)?;
    let d_gold_if_finite = d.get(gold_index).copied().filter(|v| v.is_finite());

    Ok(RowData {
        d,
        gold_index,
        d_gold_if_finite,
        argmax_index,
    })
}

/// `Σ_k exp(β·d_k)`（有限要素のみ）と `Σ_k exp(β·d_k)·d_k` を 1 回の走査で
/// まとめて計算する。`d_k <= 0` のため `exp(β·d_k) ∈ (0, 1]` でオーバー
/// フローしない（2 節）。
pub(crate) fn sum_exp_and_weighted(beta: f64, d: &[f64]) -> (f64, f64) {
    let mut sum_exp = 0.0f64;
    let mut weighted = 0.0f64;
    for &dk in d {
        if dk.is_finite() {
            let e = (beta * dk).exp();
            sum_exp += e;
            weighted += e * dk;
        }
    }
    (sum_exp, weighted)
}

/// 行ごとの top1 確率 `= 1 / Σ_k exp(β·d_k)`（`d` の最大値は 0 になるよう
/// 正規化済みのため、`exp(β·0) = 1` が top1 の分子になる）。
pub(crate) fn top1_probability(beta: f64, d: &[f64]) -> f64 {
    let (sum_exp, _) = sum_exp_and_weighted(beta, d);
    1.0 / sum_exp
}

/// NLL の導関数 `h(β) = mean_i( Σ_k p_k(β)·d_k − d_gold )`。
/// gold のロジットが `−∞` の行（[`RowData::d_gold_if_finite`] が `None`）は
/// 分母・分子の両方から除外する（4 節）。該当行が 1 件も無ければ `0.0`
/// （目的関数が定義できない＝平ら扱い。呼び出し側で T*=1.0 に落とす）。
fn h_of_beta(beta: f64, rows: &[RowData]) -> f64 {
    let mut sum = 0.0f64;
    let mut count: u64 = 0;
    for row in rows {
        if let Some(d_gold) = row.d_gold_if_finite {
            let (sum_exp, weighted) = sum_exp_and_weighted(beta, &row.d);
            sum += weighted / sum_exp - d_gold;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// 報告用の平均 NLL。gold が有限の行は `ln(Σexp(β·d)) − β·d_gold`、gold が
/// `−∞` の行は定数 `−ln(`[`NLL_CLIP_EPSILON`]`)` を加える（4 節）。
fn mean_nll(beta: f64, rows: &[RowData]) -> f64 {
    let clip_constant = -(NLL_CLIP_EPSILON.ln());
    let mut sum = 0.0f64;
    for row in rows {
        let contrib = match row.d_gold_if_finite {
            Some(d_gold) => {
                let (sum_exp, _) = sum_exp_and_weighted(beta, &row.d);
                sum_exp.ln() - beta * d_gold
            }
            None => clip_constant,
        };
        sum += contrib;
    }
    sum / rows.len() as f64
}

/// ECE（Expected Calibration Error）。[`ECE_BINS`] 個の等幅ビン（境界
/// `i / ECE_BINS`、最後のビンだけ上端を含む）に top1 確率で振り分け、
/// `|平均信頼度 − 正解率|` を件数で重み付けして平均する。
fn ece_of_beta(beta: f64, rows: &[RowData]) -> f64 {
    let mut bin_count = [0u64; ECE_BINS];
    let mut bin_conf_sum = [0.0f64; ECE_BINS];
    let mut bin_correct = [0u64; ECE_BINS];

    for row in rows {
        let conf = top1_probability(beta, &row.d);
        let bin = bin_of_confidence(conf);
        if let (Some(count), Some(conf_sum), Some(correct)) = (
            bin_count.get_mut(bin),
            bin_conf_sum.get_mut(bin),
            bin_correct.get_mut(bin),
        ) {
            *count += 1;
            *conf_sum += conf;
            if row.argmax_index == row.gold_index {
                *correct += 1;
            }
        }
    }

    let n_total: u64 = rows.len() as u64;
    let mut weighted_gap_sum = 0.0f64;
    for ((&count, &conf_sum), &correct) in bin_count
        .iter()
        .zip(bin_conf_sum.iter())
        .zip(bin_correct.iter())
    {
        if count == 0 {
            continue;
        }
        let avg_conf = conf_sum / count as f64;
        let acc = correct as f64 / count as f64;
        weighted_gap_sum += (avg_conf - acc).abs() * count as f64;
    }
    weighted_gap_sum / n_total as f64
}

/// 信頼度 `conf` が属するビンの添字（`lo <= conf < hi`、最後のビンだけ
/// `lo <= conf <= hi`）。境界は `i as f64 / ECE_BINS as f64`。
fn bin_of_confidence(conf: f64) -> usize {
    for bin in 0..ECE_BINS {
        let lo = bin as f64 / ECE_BINS as f64;
        let hi = (bin + 1) as f64 / ECE_BINS as f64;
        let in_bin = if bin + 1 == ECE_BINS {
            conf >= lo && conf <= hi
        } else {
            conf >= lo && conf < hi
        };
        if in_bin {
            return bin;
        }
    }
    // conf が [0, 1] の範囲外（呼び出し元は top1 確率のみを渡すため理論上
    // 到達しないが、fail-closed のため最後のビンへ寄せる）。
    ECE_BINS - 1
}

/// `top1` の分位点で保留しきい値 τ を決める（`numpy.quantile(q=0.20,
/// method="lower")` と同じ、すなわち `floor(0.2 * (n-1))` 番目〔0 始まり・
/// 昇順〕の値）。目標 coverage 80%（`= 1 - 1/5`）は定数（C-7）で、引数には
/// しない。この規則は常に coverage ≥ 80% を満たすが、`n` が 5 の倍数の
/// ときは、その条件を満たす最大の τ より 1 段低い値になりうる
/// （PoC-12 の規則をそのまま踏襲するための仕様。例: n=5 では
/// `floor(0.2*4)=0` 番目を返すが、1 番目〔昇順〕でも coverage=4/5=80% を
/// 満たす）。
///
/// [`calibrate`] の内部だけでなく、保留状態への接続（[`crate::abstention`]・
/// TASK-22.1-2・issue #96）・coverage の記録と表示（TASK-22.3）からも再利用
/// できるよう独立した公開関数にする。
pub fn select_threshold(top1: &[f64]) -> Result<f64, CalibrationError> {
    if top1.is_empty() {
        return Err(CalibrationError::EmptyTop1);
    }
    // `calibrate` を経由せず外部から直接呼べる公開関数のため、複製・
    // ソート（確保・計算量が O(n log n)）の前に件数を検証する
    // （REQ-39 ガード層「資源の上限」）。
    if top1.len() > significance::MAX_EVAL_RECORDS {
        return Err(CalibrationError::TooManyTop1 {
            n: top1.len(),
            limit: significance::MAX_EVAL_RECORDS,
        });
    }
    for (index, &v) in top1.iter().enumerate() {
        if !v.is_finite() {
            return Err(CalibrationError::NonFiniteTop1 { index });
        }
        if !(0.0..=1.0).contains(&v) {
            return Err(CalibrationError::Top1OutOfRange { index });
        }
    }

    let mut sorted: Vec<f64> = top1.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    // `numpy.quantile(q=0.20, method="lower")` は `floor(0.2*(n-1))` 番目。
    // `(n-1) / 5`（整数演算）は `floor(0.2*(n-1))` と同じ値になる
    // （`0.2 = 1/5`）。
    let idx = (n - 1) / TARGET_COVERAGE_DENOMINATOR;
    sorted
        .get(idx)
        .copied()
        .ok_or_else(|| CalibrationError::Internal {
            detail: format!("threshold index {idx} out of range for {n} values"),
        })
}

/// 温度 T・しきい値 τ を validation データから計算する（REQ-22 正常系・
/// TASK-22.1-1）。
///
/// - `labels`: 宣言順のラベル ID（[`crate::metrics::evaluate_single_select`]
///   と同じ規約。空・空 ID・重複・上限超過は [`CalibrationError::InvalidLabels`]）
/// - `records`: validation 1 件ずつの gold・ロジット。**validation 分割
///   以外を渡してはならない**（評価データ・最終 test の凍結を破る。
///   REQ-17・REQ-27。型では強制できない呼び出し側の責務）
///
/// 乱数は使わず、同一入力に対して常にビット単位で同じ結果を返す
/// （決定的。`.claude/rules/evaluation-contract.md`「決定性」）。
pub fn calibrate(
    labels: &[&str],
    records: &[CalibrationRecord],
) -> Result<Calibration, CalibrationError> {
    let label_index =
        metrics::build_label_index(labels).map_err(CalibrationError::InvalidLabels)?;
    let n_labels = labels.len();

    if records.is_empty() {
        return Err(CalibrationError::EmptyRecords);
    }
    if records.len() > significance::MAX_EVAL_RECORDS {
        return Err(CalibrationError::TooManyRecords {
            n: records.len(),
            limit: significance::MAX_EVAL_RECORDS,
        });
    }
    let n_cells = n_labels
        .checked_mul(records.len())
        .ok_or(CalibrationError::TooManyCells {
            n_records: records.len(),
            n_labels,
            limit: MAX_CALIBRATION_CELLS,
        })?;
    if n_cells > MAX_CALIBRATION_CELLS {
        return Err(CalibrationError::TooManyCells {
            n_records: records.len(),
            n_labels,
            limit: MAX_CALIBRATION_CELLS,
        });
    }

    let mut rows: Vec<RowData> = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        rows.push(build_row(index, record, &label_index, n_labels)?);
    }

    // β = 1/T の探索範囲。β_lo は T=TEMPERATURE_MAX、β_hi は T=TEMPERATURE_MIN
    // に対応する（1 節）。
    let beta_lo_bound = 1.0 / TEMPERATURE_MAX;
    let beta_hi_bound = 1.0 / TEMPERATURE_MIN;
    let h_lo = h_of_beta(beta_lo_bound, &rows);
    let h_hi = h_of_beta(beta_hi_bound, &rows);

    let temperature_star = if h_lo >= 0.0 && h_hi <= 0.0 {
        // h が両端で符号を持たない（凸関数の導関数が単調非減少のため、
        // これは h ≡ 0、すなわち目的関数が平らであることを意味する）。
        1.0
    } else if h_lo >= 0.0 {
        1.0 / beta_lo_bound
    } else if h_hi <= 0.0 {
        1.0 / beta_hi_bound
    } else {
        let mut lo = beta_lo_bound;
        let mut hi = beta_hi_bound;
        let mut iterations: u32 = 0;
        while iterations < MAX_BISECTION_ITERATIONS {
            let mid = lo + (hi - lo) / 2.0;
            if mid <= lo || mid >= hi {
                // 浮動小数点の精度限界に達した（これ以上は縮まらない）。
                break;
            }
            let h_mid = h_of_beta(mid, &rows);
            if h_mid > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
            iterations += 1;
        }
        1.0 / (lo + (hi - lo) / 2.0)
    };

    let beta_star = beta_of_temperature(temperature_star);
    let nll_t1 = mean_nll(1.0, &rows);
    let nll_t_star = mean_nll(beta_star, &rows);
    let ece_t1 = ece_of_beta(1.0, &rows);
    let ece_t_star = ece_of_beta(beta_star, &rows);

    // 各ロジット・差分（`d_k`）は有限でも、`β·d_gold` の掛け算や
    // `Σ exp(β·d_k)` の累積でオーバーフローし、NLL が `±∞` になりうる
    // （例: `d_gold` が極端な有限の負値で `β` を掛けると桁あふれする）。
    // 評価契約の fail-closed 原則（REQ-27）に従い、返却前に検証する。
    if !nll_t1.is_finite() {
        return Err(CalibrationError::NonFiniteResult {
            detail: "mean_nll at T=1.0 is not finite".to_string(),
        });
    }
    if !nll_t_star.is_finite() {
        return Err(CalibrationError::NonFiniteResult {
            detail: "mean_nll at T=T* is not finite".to_string(),
        });
    }

    // 採否: 厳密な `<`（同値なら採用しない）。
    let adopted = ece_t_star < ece_t1;
    let chosen_temperature = if adopted { temperature_star } else { 1.0 };
    let chosen_beta = beta_of_temperature(chosen_temperature);

    let top1_values: Vec<f64> = rows
        .iter()
        .map(|row| top1_probability(chosen_beta, &row.d))
        .collect();
    let threshold = select_threshold(&top1_values)?;

    let n = top1_values.len() as u64;
    let covered = top1_values.iter().filter(|&&v| v >= threshold).count() as u64;
    let validation_coverage = Ratio::new(covered, n).ok_or_else(|| CalibrationError::Internal {
        detail: "validation coverage ratio has zero denominator".to_string(),
    })?;

    Ok(Calibration {
        temperature_star,
        chosen_temperature,
        adopted,
        threshold,
        validation_coverage,
        nll_t1,
        nll_t_star,
        ece_t1,
        ece_t_star,
        n_validation: records.len() as u64,
        labels: labels.iter().map(|&label| label.to_string()).collect(),
    })
}

/// テスト専用の許容差付き比較（評価契約の許容差 1e-9 に合わせる）。
#[cfg(test)]
fn approx_eq(a: f64, b: f64) -> bool {
    const FLOAT_EPSILON: f64 = 1e-9;
    (a - b).abs() < FLOAT_EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TASK-22.1-1: softmax（top1 確率の分子分母）の和が 1 になる
    /// （2 ラベル・有限ロジットのみ）。
    #[test]
    fn req22_sum_exp_normalizes_to_full_probability_mass() {
        let d = vec![0.0, -2.0];
        let (sum_exp, _) = sum_exp_and_weighted(1.0, &d);
        let p0 = 1.0 / sum_exp;
        let p1 = (1.0f64 * -2.0).exp() / sum_exp;
        assert!(approx_eq(p0 + p1, 1.0));
    }

    /// TASK-22.1-1: `−∞` の要素は確率 0 として扱われ、和に寄与しない。
    #[test]
    fn req22_negative_infinity_logit_contributes_zero_probability() {
        let d = vec![0.0, f64::NEG_INFINITY, -1.0];
        let (sum_exp, weighted) = sum_exp_and_weighted(1.0, &d);
        // -∞ の項は exp(-∞) = 0 として和から自然に消える。
        let expected_sum_exp = 1.0 + (-1.0f64).exp();
        assert!(approx_eq(sum_exp, expected_sum_exp));
        let expected_weighted = -(-1.0f64).exp();
        assert!(approx_eq(weighted, expected_weighted));
    }

    /// TASK-22.1-1: ECE のビン境界（`conf=0.0` は先頭ビン、`conf=1.0` は
    /// 最後のビン）。
    #[test]
    fn req22_ece_bin_boundaries() {
        assert_eq!(bin_of_confidence(0.0), 0);
        assert_eq!(bin_of_confidence(1.0), ECE_BINS - 1);
        assert_eq!(bin_of_confidence(14.0 / 15.0), ECE_BINS - 1);
        assert_eq!(bin_of_confidence(13.0 / 15.0), 13);
    }

    /// TASK-22.1-1: `h(β)` は β について単調非減少（凸関数の導関数）。
    #[test]
    fn req22_h_is_monotonic_nondecreasing_in_beta() {
        let rows = vec![RowData {
            d: vec![0.0, -2.0],
            gold_index: 0,
            d_gold_if_finite: Some(0.0),
            argmax_index: 0,
        }];
        let h_small = h_of_beta(0.1, &rows);
        let h_mid = h_of_beta(1.0, &rows);
        let h_large = h_of_beta(10.0, &rows);
        assert!(h_small <= h_mid);
        assert!(h_mid <= h_large);
    }

    /// TASK-22.1-1: gold 行が 1 件も無ければ `h_of_beta` は 0.0（平ら扱い）。
    #[test]
    fn req22_h_is_zero_when_no_finite_gold_rows() {
        let rows = vec![RowData {
            d: vec![0.0, f64::NEG_INFINITY],
            gold_index: 1,
            d_gold_if_finite: None,
            argmax_index: 0,
        }];
        assert!(approx_eq(h_of_beta(1.0, &rows), 0.0));
    }

    /// REQ-39（ガード層「資源の上限」）: `select_threshold` は `calibrate`
    /// を経由せず外部から直接渡された `top1` の件数を、確保・ソート前に
    /// 上限（[`significance::MAX_EVAL_RECORDS`]）で検証し拒否する。
    #[test]
    fn req39_select_threshold_rejects_top1_exceeding_max_records() {
        // 実際に上限件数の Vec を確保せず、上限超過を境界値のみで確認する
        // （`MAX_EVAL_RECORDS` は 100 万件でテスト自体が重くなるため）。
        let n = significance::MAX_EVAL_RECORDS + 1;
        let top1 = vec![0.5_f64; n];
        let err = select_threshold(&top1).expect_err("upper limit must be rejected");
        assert_eq!(
            err,
            CalibrationError::TooManyTop1 {
                n,
                limit: significance::MAX_EVAL_RECORDS,
            }
        );
    }

    /// REQ-23（不正なスコアの扱い）: `select_threshold` は `0.0..=1.0` の
    /// 範囲外の値を確率として受け入れない（有限だが不正な `-1.0`・`2.0`）。
    #[test]
    fn req23_select_threshold_rejects_out_of_range_probability() {
        let err =
            select_threshold(&[0.5, -1.0, 0.9]).expect_err("negative probability must be rejected");
        assert_eq!(err, CalibrationError::Top1OutOfRange { index: 1 });

        let err = select_threshold(&[0.5, 2.0, 0.9]).expect_err("probability > 1 must be rejected");
        assert_eq!(err, CalibrationError::Top1OutOfRange { index: 1 });
    }

    /// REQ-23: `0.0`・`1.0` は範囲の境界として許可される。
    #[test]
    fn req23_select_threshold_accepts_boundary_probabilities() {
        assert!(select_threshold(&[0.0, 1.0, 0.5]).is_ok());
    }

    /// REQ-27（評価契約の fail-closed 原則）: `calibrate` は個々のロジット
    /// が有限でも、正規化後の差分 `d_gold` が極端な有限値になり `mean_nll`
    /// の計算過程がオーバーフローする入力を `NonFiniteResult` として拒否
    /// し、無限大の NLL を含む `Calibration` を返さない。gold ではない
    /// ラベルのロジットが `f64::MAX / 2`（argmax）、gold のロジットが
    /// `-f64::MAX / 2` のとき、正規化後の `d_gold = -f64::MAX / 2 -
    /// f64::MAX / 2 = -f64::MAX` は有限（オーバーフローしない厳密な減算）
    /// だが、`mean_nll` の `sum_exp.ln() − β·d_gold`（`β=1.0`）が 1 行あたり
    /// `f64::MAX` になり、2 行分を合計する `Σ` でオーバーフローして
    /// `+∞` になる（個々の値は有限でも合算で非有限になる例）。
    #[test]
    fn req27_calibrate_rejects_overflow_to_nonfinite_nll() {
        let labels = ["a", "b"];
        let logits: [f64; 2] = [-f64::MAX / 2.0, f64::MAX / 2.0];
        let records = vec![
            CalibrationRecord {
                gold: "a",
                logits: &logits,
            },
            CalibrationRecord {
                gold: "a",
                logits: &logits,
            },
        ];
        let err =
            calibrate(&labels, &records).expect_err("overflow to non-finite NLL must be rejected");
        assert!(matches!(err, CalibrationError::NonFiniteResult { .. }));
    }

    /// REQ-27（評価契約の fail-closed 原則）: gold のロジットと argmax の
    /// ロジットがともに有限でも、正規化の減算 `z - max_finite` 自体が
    /// `−∞` へ桁あふれする入力（`z = -f64::MAX`・`max_finite = f64::MAX`）
    /// では、`d_gold_if_finite` が「gold が正当な `−∞` 入力だった」場合と
    /// 区別できなくなり、silently 除外（gold の確率が 0 であるかのように
    /// `mean_nll` の clip 定数を適用）してしまう回帰を防ぐ。`calibrate` は
    /// この桁あふれを `NonFiniteResult` として拒否しなければならない
    /// （codex/review 指摘: PR #241 threadId PRRT_kwDOUq-SxM6ms6lh）。
    #[test]
    fn req27_calibrate_rejects_finite_logit_difference_overflow() {
        let labels = ["a", "b"];
        let logits: [f64; 2] = [-f64::MAX, f64::MAX];
        let records = vec![CalibrationRecord {
            gold: "a",
            logits: &logits,
        }];
        let err = calibrate(&labels, &records)
            .expect_err("finite logit difference overflow must be rejected, not silently excluded");
        assert!(matches!(err, CalibrationError::NonFiniteResult { .. }));
    }
}
