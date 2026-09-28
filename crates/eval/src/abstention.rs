//! 確信度（校正後の top1 確率）としきい値 τ の比較による保留判定と、
//! 保留込み／保留なしの誤り率の比較（REQ-22 正常系・TASK-22.1-2・issue #96）。
//!
//! [`crate::calibration`]（TASK-22.1-1・issue #95）が `validation` から求めた
//! 温度 T・しきい値 τ（[`crate::calibration::Calibration`]）を使い、評価データ
//! の各行が「採用」か「保留」かを決め（[`decide_abstention`]）、両者を
//! [`crate::metrics::evaluate_single_select`] に渡して得た指標から
//! 保留込み／保留なしの誤り率を比較する（[`compare_abstention`]）。
//!
//! CLI の `evaluate` 工程（REQ-33・issue #140）から、校正済みの
//! [`crate::calibration::Calibration`] と評価データ（validation に限らない）を
//! 渡して呼ばれる想定（本モジュール自体は推論経路には入らない。`lib.rs`
//! 「層の境界・不変条件」参照）。
//!
//! # 評価契約との関係（REQ-17・REQ-27）
//!
//! - [`decide_abstention`] は **正解ラベル（gold）を受け取らない**。推論関数
//!   （将来の推論ランタイム）に相当する経路であり、`input`（ロジット）以外の
//!   情報（正解ラベル・分割情報・評価データの統計）を渡さないという評価の
//!   独立性の原則を、この関数のシグネチャ自体で示す。
//! - [`compare_abstention`] は `calibration`（`&Calibration`）から温度 T・
//!   しきい値 τ を**読むだけ**で、渡された評価データから選び直すことは
//!   ない（`select_threshold` を呼ばない）。評価データで τ を引き直すと、
//!   凍結した `validation` 分割だけから校正するという契約（REQ-17）が崩れる。
//!
//! # 資源上限（REQ-39）
//!
//! [`compare_abstention`] は評価件数・ラベル数を確保前に検証する
//! （[`crate::significance::MAX_EVAL_RECORDS`]・
//! [`crate::calibration::MAX_CALIBRATION_CELLS`]。[`crate::metrics::MAX_LABELS`]
//! は [`crate::metrics::evaluate_single_select`] 内部で検証される）。
//! `Outcome::Label(String)` は行ごとに確保せず、ラベルごとに 1 つ
//! （`n_labels` 件）と `Outcome::Abstain` 1 つを事前に作り、各行の
//! `EvalRecord` はそれらへの参照として構築する（行数分の確保は
//! `EvalRecord` の `Vec` 2 本のみ）。
//!
//! # 対象外
//!
//! - 「対象外」ラベルによる処理（TASK-22.2）
//! - coverage の記録・表示（TASK-22.3）
//! - REQ-22 正常系の 95% ブートストラップ信頼区間（PoC-12 は `n_boot=2000`・
//!   `seed=12` の対応のあるブートストラップを使うが、本 issue の受入は
//!   具体値での比較のみ。実装には乱数と依存の判断が要る）
//! - 推論ランタイム側での保留判定（REQ-28・REQ-32）。評価器は推論経路に
//!   入らない（`lib.rs`「層の境界・不変条件」）。将来ランタイムで同じ規則が
//!   要る場合は、判定規則を下位層へ移して共有し、本モジュールの重複実装を
//!   避ける方針とする
//! - T・τ の永続化・配布パッケージへの格納（REQ-30）・CLI `evaluate` 工程への
//!   配線（issue #140）

use crate::calibration::{self, Calibration, CalibrationError};
use crate::metrics::{self, EvalRecord, Outcome, Ratio};
use crate::significance;

/// 1 行分の保留判定結果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AbstentionDecision {
    /// 採用（確信度が τ 以上）。`label_index` は argmax の宣言順添字。
    Adopt {
        /// argmax の宣言順添字（同値は宣言順の先頭。`calibrate` と同じ規則）。
        label_index: usize,
        /// 校正後の top1 確率（確信度）。
        confidence: f64,
    },
    /// 保留（確信度が τ 未満）。
    Abstain {
        /// 校正後の top1 確率（確信度）。
        confidence: f64,
    },
}

impl AbstentionDecision {
    /// 確信度（採用・保留のいずれでも取得できる）。
    pub fn confidence(&self) -> f64 {
        match self {
            AbstentionDecision::Adopt { confidence, .. } => *confidence,
            AbstentionDecision::Abstain { confidence } => *confidence,
        }
    }

    /// [`crate::metrics::Outcome`] へ変換する。
    ///
    /// `label_index` の意味は「`calibration` に渡したのと同じ宣言順での
    /// argmax 添字」であり（[`decide_abstention`] の契約）、ラベル ID の
    /// 解決は `calibration.labels()`（校正時に保持した宣言順の ID。
    /// [`Calibration::labels`]）からのみ行う。呼び出し元が任意のラベル集合を
    /// 渡せる形にすると、校正時と異なる集合・並びを渡されたときに誤った
    /// 添字を正常なラベルとして解釈しうる（`compare_abstention` が
    /// `CalibrationError::LabelMismatch` で防いでいるのと同じ問題。
    /// codex/review 指摘・REQ-17・REQ-27）。`self` がどの `Calibration` から
    /// 得た決定かを型で保証できないため、`decide_abstention` に渡したのと
    /// **同じ** `calibration` を渡す責務は呼び出し元にあるが、少なくとも
    /// ラベル ID は外部から任意の値を渡せないようにする。
    ///
    /// [`compare_abstention`] は行ごとに `Outcome::Label(String)` を確保しない
    /// 経路（モジュール冒頭「資源上限」参照）を使うため、この変換は独立した
    /// 呼び出し元（CLI 配線・issue #140 等）向けの利便関数として用意する。
    /// ラベル添字は `get()` で引き、範囲外は `[]` を使わず
    /// [`CalibrationError::Internal`] を返す（`decide_abstention` が
    /// `calibration.n_labels()` の範囲内でしか `label_index` を作らないため
    /// 通常到達しないが、fail-closed のため検査する）。
    pub fn to_outcome(&self, calibration: &Calibration) -> Result<Outcome, CalibrationError> {
        match self {
            AbstentionDecision::Adopt { label_index, .. } => {
                let label = calibration.labels().get(*label_index).ok_or_else(|| {
                    CalibrationError::Internal {
                        detail: format!(
                            "label index {label_index} out of range for to_outcome ({} labels)",
                            calibration.n_labels()
                        ),
                    }
                })?;
                Ok(Outcome::Label(label.clone()))
            }
            AbstentionDecision::Abstain { .. } => Ok(Outcome::Abstain),
        }
    }
}

/// `records` 内での位置付きで保留判定する内部経路。[`decide_abstention`]
/// （公開・1 行版）と [`compare_abstention`]（複数行）の両方がこれを使い、
/// エラーメッセージの `index` を実際の行位置に揃える。
fn decide_abstention_at(
    index: usize,
    calibration: &Calibration,
    logits: &[f64],
) -> Result<AbstentionDecision, CalibrationError> {
    let (d, argmax_index) = calibration::preprocess_logits(index, logits, calibration.n_labels())?;
    let confidence = calibration::top1_probability(calibration.chosen_beta(), &d);
    if confidence >= calibration.threshold() {
        Ok(AbstentionDecision::Adopt {
            label_index: argmax_index,
            confidence,
        })
    } else {
        Ok(AbstentionDecision::Abstain { confidence })
    }
}

/// 確信度（校正後の top1 確率）としきい値 τ を比べ、採用か保留かを決める
/// （REQ-22 正常系・TASK-22.1-2）。
///
/// - `calibration`: [`crate::calibration::calibrate`] が返した T・τ（**評価
///   データからここで選び直すことはない**。REQ-17・REQ-27）
/// - `logits`: 判定対象 1 件分のロジット（`calibration` を計算したときと
///   同じ宣言順・同じ長さ）。**gold（正解ラベル）は受け取らない**（REQ-27:
///   推論関数には `input` だけを渡す）
///
/// `logits.len() != calibration.n_labels()` は
/// [`CalibrationError::LogitLengthMismatch`]（この 1 行版では `index` は
/// 常に `0`）。NaN・`+∞` は [`CalibrationError::NonFiniteLogit`]、全要素が
/// `−∞` なら [`CalibrationError::NoFiniteLogit`]（[`crate::calibration`]
/// モジュール冒頭 3 節と同じ入力規則）。
///
/// 判定は `confidence >= calibration.threshold()` で採用、`<` で保留
/// （PoC-12 `p[top] >= tau` と `calibrate` 内部の coverage 計算
/// `v >= threshold` に一致する境界。厳密な `>=`）。
pub fn decide_abstention(
    calibration: &Calibration,
    logits: &[f64],
) -> Result<AbstentionDecision, CalibrationError> {
    decide_abstention_at(0, calibration, logits)
}

/// 保留込み／保留なしの評価指標と、そこから導く誤り率の比較。
///
/// フィールドは非公開にし、構築は [`compare_abstention`] 内に集約する
/// （壊れた値、例えば `adopted_error()` が `with_abstention()` と矛盾する
/// 状態を外部から作らせない）。
#[derive(Debug, Clone, PartialEq)]
pub struct AbstentionComparison {
    without_abstention: metrics::SingleSelectMetrics,
    with_abstention: metrics::SingleSelectMetrics,
    unconditional_error: Ratio,
    adopted_error: Option<Ratio>,
}

impl AbstentionComparison {
    /// 保留なし（全行で argmax を採用）の評価指標。
    pub fn without_abstention(&self) -> &metrics::SingleSelectMetrics {
        &self.without_abstention
    }

    /// 保留込み（確信度が τ 未満の行を `Outcome::Abstain` とする）の評価指標。
    pub fn with_abstention(&self) -> &metrics::SingleSelectMetrics {
        &self.with_abstention
    }

    /// 保留なしの誤り率（分母 = 全件）。
    pub fn unconditional_error(&self) -> Ratio {
        self.unconditional_error
    }

    /// 保留込みの誤り率（分母 = 採用件数 = 全件 − 保留件数）。全件保留なら
    /// `None`（評価契約: 分母 0 の指標は `null`。0 や 1 で埋めない）。
    pub fn adopted_error(&self) -> Option<Ratio> {
        self.adopted_error
    }
}

/// 評価データ（validation に限らない）に対し、`calibration` の T・τ を使って
/// 保留込み／保留なしの誤り率を比較する（REQ-22 正常系・TASK-22.1-2）。
///
/// - `labels`: 宣言順のラベル ID（[`crate::metrics::evaluate_single_select`]・
///   [`crate::calibration::calibrate`] と同じ規約）。`calibration` を計算した
///   ときの `labels.len()` と一致しなければ
///   [`CalibrationError::LabelCountMismatch`]、件数は一致しても宣言順の ID が
///   一致しなければ [`CalibrationError::LabelMismatch`]（同数の別ラベル集合・
///   並べ替えで検査を通過し、校正時とは異なる添字を予測ラベルとして解釈する
///   ことを防ぐ。校正した対象と異なるラベル集合で評価データを走査しない。
///   REQ-17・REQ-27）
/// - `calibration`: [`crate::calibration::calibrate`] が返した T・τ。ここから
///   **読むだけ**で、`records` から τ を選び直すことはない
/// - `records`: 評価 1 件ずつの gold・ロジット（validation に限らず、最終
///   test でもよい。ただし凍結した最終 test への適用は呼び出し側の責務で
///   1 回限りにすること。REQ-27・`.claude/rules/evaluation-contract.md`）
///
/// 各行の判定はまずロジットだけで行い（[`decide_abstention_at`]。gold を
/// 渡さない）、その後に gold と組にして評価器（[`crate::metrics`]）へ渡す
/// （評価ロジックの再実装をしない。TASK-24.1 の 1 つだけに評価ロジックを
/// 集約する方針。`lib.rs`）。
pub fn compare_abstention(
    labels: &[&str],
    calibration: &Calibration,
    records: &[calibration::CalibrationRecord],
) -> Result<AbstentionComparison, CalibrationError> {
    let label_index =
        metrics::build_label_index(labels).map_err(CalibrationError::InvalidLabels)?;
    let n_labels = labels.len();
    if n_labels != calibration.n_labels() {
        return Err(CalibrationError::LabelCountMismatch {
            calibrated: calibration.n_labels(),
            given: n_labels,
        });
    }
    // 件数一致だけでは、同数の別ラベル集合や宣言順を並べ替えた集合でも
    // 検査を通過してしまい、校正時とは異なる添字を予測ラベルとして解釈し
    // うる（codex/review 指摘・REQ-17・REQ-27）。宣言順で ID そのものの
    // 同一性を確認する。
    for (index, (calibrated_label, &given_label)) in
        calibration.labels().iter().zip(labels.iter()).enumerate()
    {
        if calibrated_label != given_label {
            return Err(CalibrationError::LabelMismatch {
                index,
                calibrated: calibrated_label.clone(),
                given: given_label.to_string(),
            });
        }
    }
    // ラベルは `label_index`（`build_label_index` の戻り値）で検証済みだが、
    // gold の照合自体は `evaluate_single_select` に委譲する（評価ロジックの
    // 再実装をしない方針）。ここでは検証結果を使わないため drop する。
    drop(label_index);

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
            limit: calibration::MAX_CALIBRATION_CELLS,
        })?;
    if n_cells > calibration::MAX_CALIBRATION_CELLS {
        return Err(CalibrationError::TooManyCells {
            n_records: records.len(),
            n_labels,
            limit: calibration::MAX_CALIBRATION_CELLS,
        });
    }

    // 行ごとに `Outcome::Label(String)` を確保しない。ラベルごとに 1 つ
    // （宣言順）と `Outcome::Abstain` 1 つを事前に作り、各行の `EvalRecord`
    // はそれらへの参照にする（モジュール冒頭「資源上限」）。
    let label_outcomes: Vec<Outcome> = labels
        .iter()
        .map(|&label| Outcome::Label(label.to_string()))
        .collect();
    let abstain_outcome = Outcome::Abstain;

    let mut without_records: Vec<EvalRecord> = Vec::with_capacity(records.len());
    let mut with_records: Vec<EvalRecord> = Vec::with_capacity(records.len());

    for (index, record) in records.iter().enumerate() {
        let decision = decide_abstention_at(index, calibration, record.logits)?;
        let label_index = match decision {
            AbstentionDecision::Adopt { label_index, .. } => label_index,
            AbstentionDecision::Abstain { .. } => {
                // 保留の行でも「保留なし」側は argmax を採用するため、
                // argmax の添字は判定結果に含まれない。`decide_abstention_at`
                // 自体は argmax を常に計算しているが `Abstain` はそれを
                // 破棄する設計のため、ここでもう一度ロジットを前処理する。
                let (_, argmax_index) =
                    calibration::preprocess_logits(index, record.logits, calibration.n_labels())?;
                argmax_index
            }
        };
        let label_outcome =
            label_outcomes
                .get(label_index)
                .ok_or_else(|| CalibrationError::Internal {
                    detail: format!(
                        "argmax label index {label_index} out of range at record index {index}"
                    ),
                })?;

        without_records.push(EvalRecord {
            gold: record.gold,
            outcome: label_outcome,
        });
        let with_outcome = match decision {
            AbstentionDecision::Adopt { .. } => label_outcome,
            AbstentionDecision::Abstain { .. } => &abstain_outcome,
        };
        with_records.push(EvalRecord {
            gold: record.gold,
            outcome: with_outcome,
        });
    }

    let without_abstention = metrics::evaluate_single_select(labels, &without_records)
        .map_err(CalibrationError::Evaluation)?;
    let with_abstention = metrics::evaluate_single_select(labels, &with_records)
        .map_err(CalibrationError::Evaluation)?;

    let unconditional_error =
        ratio_complement(without_abstention.accuracy.overall).ok_or_else(|| {
            CalibrationError::Internal {
                detail: "unconditional error ratio computation failed".to_string(),
            }
        })?;
    let adopted_error = match with_abstention.accuracy.adopted_decision {
        Some(adopted_accuracy) => {
            Some(
                ratio_complement(adopted_accuracy).ok_or_else(|| CalibrationError::Internal {
                    detail: "adopted error ratio computation failed".to_string(),
                })?,
            )
        }
        None => None,
    };

    Ok(AbstentionComparison {
        without_abstention,
        with_abstention,
        unconditional_error,
        adopted_error,
    })
}

/// `1 - ratio`（正解率から誤り率へ）を分子分母の整数演算で計算する
/// （浮動小数の減算による誤差を避け、`Ratio` の不変条件を保ったまま
/// 誤り率を組み立てる）。`ratio` は [`crate::metrics::evaluate_single_select`]
/// が返す正解率で、常に `numerator <= denominator` を満たすため
/// `checked_sub` は失敗しないはずだが、fail-closed のため `Option` のまま
/// 扱う。
fn ratio_complement(ratio: Ratio) -> Option<Ratio> {
    let errors = ratio.denominator().checked_sub(ratio.numerator())?;
    Ratio::new(errors, ratio.denominator())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{CalibrationRecord, calibrate};

    const LABELS: [&str; 3] = ["l0", "l1", "l2"];

    /// テスト専用の許容差付き比較（評価契約の許容差 1e-9 に合わせる）。
    fn approx_eq(a: f64, b: f64) -> bool {
        const FLOAT_EPSILON: f64 = 1e-9;
        (a - b).abs() < FLOAT_EPSILON
    }

    /// ひな形 `(a, n)`: argmax の位置 `p` に `a`、それ以外に `0.0` を置いた
    /// ロジットを持つ行を `n` 件生成する。同じひな形の行は全く同じロジット
    /// ベクトルにする（3 ラベルでは `(e+e)+1` と `(1+e)+e` が 1 ulp ずれうる
    /// ため、argmax の位置がぶれないようにする。計画のコメント参照）。
    /// `correct` 件は gold=`p`（正解）、残りは gold=`(p+1) % 3`（不正解）。
    fn template_rows(
        a: f64,
        p: usize,
        correct: usize,
        incorrect: usize,
    ) -> Vec<(String, [f64; 3])> {
        let mut logits = [0.0f64; 3];
        logits[p] = a;
        let mut rows = Vec::with_capacity(correct + incorrect);
        for _ in 0..correct {
            rows.push((LABELS[p].to_string(), logits));
        }
        for _ in 0..incorrect {
            rows.push((LABELS[(p + 1) % 3].to_string(), logits));
        }
        rows
    }

    fn as_records(rows: &[(String, [f64; 3])]) -> Vec<CalibrationRecord<'_>> {
        rows.iter()
            .map(|(gold, logits)| CalibrationRecord {
                gold: gold.as_str(),
                logits,
            })
            .collect()
    }

    /// 4 節の C1（3 seed とも同一）の validation・eval ひな形。
    fn c1_validation() -> Vec<(String, [f64; 3])> {
        let mut rows = Vec::new();
        rows.extend(template_rows(0.2, 0, 3, 8));
        rows.extend(template_rows(0.6, 1, 6, 7));
        rows.extend(template_rows(1.2, 2, 10, 6));
        rows.extend(template_rows(2.0, 0, 14, 6));
        rows
    }

    fn c1_eval() -> Vec<(String, [f64; 3])> {
        let mut rows = Vec::new();
        rows.extend(template_rows(0.2, 0, 5, 15));
        rows.extend(template_rows(0.6, 1, 8, 12));
        rows.extend(template_rows(1.2, 2, 15, 10));
        rows.extend(template_rows(2.0, 0, 25, 10));
        rows
    }

    /// 4 節の C3 seed0 の validation・eval ひな形。
    fn c3_seed0_validation() -> Vec<(String, [f64; 3])> {
        let mut rows = Vec::new();
        rows.extend(template_rows(0.3, 0, 2, 7));
        rows.extend(template_rows(0.8, 1, 5, 6));
        rows.extend(template_rows(1.5, 2, 10, 5));
        rows.extend(template_rows(2.5, 0, 12, 3));
        rows
    }

    fn c3_seed0_eval() -> Vec<(String, [f64; 3])> {
        let mut rows = Vec::new();
        rows.extend(template_rows(0.3, 0, 8, 22));
        rows.extend(template_rows(0.8, 1, 10, 15));
        rows.extend(template_rows(1.5, 2, 22, 13));
        rows.extend(template_rows(2.5, 0, 24, 6));
        rows
    }

    /// 4 節の C3 seed1 の validation・eval ひな形。
    fn c3_seed1_validation() -> Vec<(String, [f64; 3])> {
        let mut rows = Vec::new();
        rows.extend(template_rows(0.25, 0, 1, 6));
        rows.extend(template_rows(0.7, 1, 4, 5));
        rows.extend(template_rows(1.4, 2, 8, 4));
        rows.extend(template_rows(2.2, 0, 9, 3));
        rows
    }

    fn c3_seed1_eval() -> Vec<(String, [f64; 3])> {
        let mut rows = Vec::new();
        rows.extend(template_rows(0.25, 0, 5, 20));
        rows.extend(template_rows(0.7, 1, 8, 12));
        rows.extend(template_rows(1.4, 2, 16, 9));
        rows.extend(template_rows(2.2, 0, 15, 5));
        rows
    }

    /// 4 節の C3 seed2 の validation・eval ひな形。
    fn c3_seed2_validation() -> Vec<(String, [f64; 3])> {
        let mut rows = Vec::new();
        rows.extend(template_rows(0.4, 0, 3, 7));
        rows.extend(template_rows(0.9, 1, 7, 8));
        rows.extend(template_rows(1.6, 2, 10, 5));
        rows.extend(template_rows(2.4, 0, 12, 3));
        rows
    }

    fn c3_seed2_eval() -> Vec<(String, [f64; 3])> {
        let mut rows = Vec::new();
        rows.extend(template_rows(0.4, 0, 5, 13));
        rows.extend(template_rows(0.9, 1, 10, 12));
        rows.extend(template_rows(1.6, 2, 19, 11));
        rows.extend(template_rows(2.4, 0, 24, 6));
        rows
    }

    /// 6 構成 1 つ分の期待値（4 節の表と対応）。
    struct ExpectedConfig {
        name: &'static str,
        validation: fn() -> Vec<(String, [f64; 3])>,
        eval: fn() -> Vec<(String, [f64; 3])>,
        n_val: usize,
        validation_covered: u64,
        unconditional_errors: u64,
        unconditional_total: u64,
        adopted_errors: u64,
        adopted_total: u64,
        abstain_count: u64,
    }

    const CONFIGS: [ExpectedConfig; 6] = [
        ExpectedConfig {
            name: "C1 seed0",
            validation: c1_validation,
            eval: c1_eval,
            n_val: 60,
            validation_covered: 49,
            unconditional_errors: 47,
            unconditional_total: 100,
            adopted_errors: 32,
            adopted_total: 80,
            abstain_count: 20,
        },
        ExpectedConfig {
            name: "C1 seed1",
            validation: c1_validation,
            eval: c1_eval,
            n_val: 60,
            validation_covered: 49,
            unconditional_errors: 47,
            unconditional_total: 100,
            adopted_errors: 32,
            adopted_total: 80,
            abstain_count: 20,
        },
        ExpectedConfig {
            name: "C1 seed2",
            validation: c1_validation,
            eval: c1_eval,
            n_val: 60,
            validation_covered: 49,
            unconditional_errors: 47,
            unconditional_total: 100,
            adopted_errors: 32,
            adopted_total: 80,
            abstain_count: 20,
        },
        ExpectedConfig {
            name: "C3 seed0",
            validation: c3_seed0_validation,
            eval: c3_seed0_eval,
            n_val: 50,
            validation_covered: 41,
            unconditional_errors: 56,
            unconditional_total: 120,
            adopted_errors: 34,
            adopted_total: 90,
            abstain_count: 30,
        },
        ExpectedConfig {
            name: "C3 seed1",
            validation: c3_seed1_validation,
            eval: c3_seed1_eval,
            n_val: 40,
            validation_covered: 33,
            unconditional_errors: 46,
            unconditional_total: 90,
            adopted_errors: 26,
            adopted_total: 65,
            abstain_count: 25,
        },
        ExpectedConfig {
            name: "C3 seed2",
            validation: c3_seed2_validation,
            eval: c3_seed2_eval,
            n_val: 55,
            validation_covered: 45,
            unconditional_errors: 42,
            unconditional_total: 100,
            adopted_errors: 29,
            adopted_total: 82,
            abstain_count: 18,
        },
    ];

    /// REQ-22 正常系（PoC-12 の 6 構成相当。TASK-22.1-2）: `calibrate`
    /// （validation）→ `compare_abstention`（eval）を実行し、保留込みの
    /// 誤り率が保留なしの誤り率を下回ることを分数の完全一致・比較で確認する。
    /// 6 構成は 4 データセット（C1 は 3 seed とも同一データ。PoC-12 で C1 が
    /// seed 間で完全に同一だった経緯に対応する。4 節参照）。
    #[test]
    fn req22_six_configs_abstention_lowers_error_rate() {
        for config in &CONFIGS {
            let validation_rows = (config.validation)();
            let validation_records = as_records(&validation_rows);
            let calibration = calibrate(&LABELS, &validation_records)
                .unwrap_or_else(|e| panic!("{}: calibrate failed: {e}", config.name));

            let eval_rows = (config.eval)();
            let eval_records = as_records(&eval_rows);
            let comparison = compare_abstention(&LABELS, &calibration, &eval_records)
                .unwrap_or_else(|e| panic!("{}: compare_abstention failed: {e}", config.name));

            let uncond = comparison.unconditional_error();
            assert_eq!(
                uncond.numerator(),
                config.unconditional_errors,
                "{}: unconditional error numerator",
                config.name
            );
            assert_eq!(
                uncond.denominator(),
                config.unconditional_total,
                "{}: unconditional error denominator",
                config.name
            );

            let adopted = comparison
                .adopted_error()
                .unwrap_or_else(|| panic!("{}: adopted_error must be Some", config.name));
            assert_eq!(
                adopted.numerator(),
                config.adopted_errors,
                "{}: adopted error numerator",
                config.name
            );
            assert_eq!(
                adopted.denominator(),
                config.adopted_total,
                "{}: adopted error denominator",
                config.name
            );

            // 分数の比較（保留込み < 保留なし）。分母が異なるため通分して比較する。
            assert!(
                adopted.numerator() * uncond.denominator()
                    < uncond.numerator() * adopted.denominator(),
                "{}: adopted error must be lower than unconditional error",
                config.name
            );
            // `value()` の 1e-9 照合は補助。
            assert!(
                adopted.value() < uncond.value() - 1e-9,
                "{}: adopted().value() must be lower than uncond().value()",
                config.name
            );
        }
    }

    /// REQ-22: `with_abstention()` の `outcome_counts` の abstain・ok 件数が
    /// 4 節の表の値と一致し、`without_abstention()` の abstain は常に 0。
    #[test]
    fn req22_outcome_counts_match_expected() {
        for config in &CONFIGS {
            let validation_rows = (config.validation)();
            let validation_records = as_records(&validation_rows);
            let calibration = calibrate(&LABELS, &validation_records).unwrap();
            let eval_rows_local = (config.eval)();
            let eval_records = as_records(&eval_rows_local);
            let comparison = compare_abstention(&LABELS, &calibration, &eval_records).unwrap();

            assert_eq!(
                comparison.with_abstention().outcome_counts.abstain,
                config.abstain_count,
                "{}: with_abstention abstain count",
                config.name
            );
            let expected_ok = config.unconditional_total - config.abstain_count;
            assert_eq!(
                comparison.with_abstention().outcome_counts.ok,
                expected_ok,
                "{}: with_abstention ok count",
                config.name
            );
            assert_eq!(
                comparison.without_abstention().outcome_counts.abstain,
                0,
                "{}: without_abstention abstain count must be 0",
                config.name
            );
        }
    }

    /// REQ-22: 各構成の validation 行自身に `decide_abstention` を適用した
    /// 採用件数が `calibration.validation_coverage()` の分子と一致する
    /// （`calibrate` 内部と判定経路がずれていないことの直接の確認）。
    #[test]
    fn req22_decision_on_validation_matches_validation_coverage() {
        for config in &CONFIGS {
            let validation_rows = (config.validation)();
            let validation_records = as_records(&validation_rows);
            let calibration = calibrate(&LABELS, &validation_records).unwrap();

            assert_eq!(
                validation_records.len(),
                config.n_val,
                "{}: validation row count",
                config.name
            );
            assert_eq!(
                calibration.validation_coverage().numerator(),
                config.validation_covered,
                "{}: validation coverage numerator",
                config.name
            );

            let mut covered = 0u64;
            for record in &validation_records {
                let decision = decide_abstention(&calibration, record.logits).unwrap();
                if matches!(decision, AbstentionDecision::Adopt { .. }) {
                    covered += 1;
                }
            }
            assert_eq!(
                covered, config.validation_covered,
                "{}: decide_abstention covered count on validation rows",
                config.name
            );
        }
    }

    /// REQ-22: 同じ入力で 2 回実行した結果が `PartialEq` で一致する（決定的）。
    #[test]
    fn req22_comparison_is_deterministic() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        let eval_rows = c1_eval();
        let eval_records = as_records(&eval_rows);

        let first = compare_abstention(&LABELS, &calibration, &eval_records).unwrap();
        let second = compare_abstention(&LABELS, &calibration, &eval_records).unwrap();
        assert_eq!(first, second);
    }

    /// REQ-27（評価の独立性）: `compare_abstention` の前後で `Calibration` の
    /// 値（threshold・chosen_temperature）が変わらず、eval 側を変えても τ が
    /// validation の値のまま（評価データから τ を選び直さない）。
    #[test]
    fn req27_calibration_not_recomputed_from_eval_data() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        let threshold_before = calibration.threshold();
        let temperature_before = calibration.chosen_temperature();

        // eval 側は validation と全く異なる分布（全件同じ argmax・大きい a）
        // にする。`select_threshold` を誤って呼んでいれば τ が変わるはず。
        let mut skewed_rows: Vec<(String, [f64; 3])> = Vec::new();
        skewed_rows.extend(template_rows(5.0, 0, 40, 0));
        let skewed_records = as_records(&skewed_rows);

        let comparison = compare_abstention(&LABELS, &calibration, &skewed_records).unwrap();
        assert!(approx_eq(calibration.threshold(), threshold_before));
        assert!(approx_eq(
            calibration.chosen_temperature(),
            temperature_before
        ));
        // 全件 a=5.0 で argmax=0・gold=0（正解）のため、保留なしの誤り率は 0。
        assert_eq!(comparison.unconditional_error().numerator(), 0);
    }

    /// 異常系: ロジット長不一致。
    #[test]
    fn req23_compare_abstention_rejects_logit_length_mismatch() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        let bad_logits = [0.0, 0.0];
        let records = vec![CalibrationRecord {
            gold: "l0",
            logits: &bad_logits,
        }];
        let err = compare_abstention(&LABELS, &calibration, &records)
            .expect_err("logit length mismatch must be rejected");
        assert_eq!(
            err,
            CalibrationError::LogitLengthMismatch {
                index: 0,
                expected: 3,
                actual: 2,
            }
        );
    }

    /// 異常系: NaN ロジット。
    #[test]
    fn req23_compare_abstention_rejects_nan_logit() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        let bad_logits = [f64::NAN, 0.0, 0.0];
        let records = vec![CalibrationRecord {
            gold: "l0",
            logits: &bad_logits,
        }];
        let err = compare_abstention(&LABELS, &calibration, &records)
            .expect_err("NaN logit must be rejected");
        assert_eq!(
            err,
            CalibrationError::NonFiniteLogit {
                index: 0,
                label_index: 0,
            }
        );
    }

    /// 異常系: 全要素が `−∞`。
    #[test]
    fn req23_compare_abstention_rejects_no_finite_logit() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        let bad_logits = [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let records = vec![CalibrationRecord {
            gold: "l0",
            logits: &bad_logits,
        }];
        let err = compare_abstention(&LABELS, &calibration, &records)
            .expect_err("all-negative-infinity logits must be rejected");
        assert_eq!(err, CalibrationError::NoFiniteLogit { index: 0 });
    }

    /// 異常系: ラベル数不一致（`calibration` は 3 ラベルで校正済み）。
    #[test]
    fn req17_compare_abstention_rejects_label_count_mismatch() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        let two_labels = ["l0", "l1"];
        let logits = [0.0, 0.0];
        let records = vec![CalibrationRecord {
            gold: "l0",
            logits: &logits,
        }];
        let err = compare_abstention(&two_labels, &calibration, &records)
            .expect_err("label count mismatch must be rejected");
        assert_eq!(
            err,
            CalibrationError::LabelCountMismatch {
                calibrated: 3,
                given: 2,
            }
        );
    }

    /// 異常系: 件数は一致するがラベル ID を並べ替えた集合（REQ-17・REQ-27。
    /// codex/review 指摘対応: 件数一致だけでは、校正時と異なる添字を予測
    /// ラベルとして解釈しうる誤りを見逃す）。
    #[test]
    fn req27_compare_abstention_rejects_permuted_labels() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        let permuted_labels = ["l1", "l0", "l2"];
        let logits = [0.0, 0.0, 0.0];
        let records = vec![CalibrationRecord {
            gold: "l0",
            logits: &logits,
        }];
        let err = compare_abstention(&permuted_labels, &calibration, &records)
            .expect_err("permuted label order must be rejected");
        assert_eq!(
            err,
            CalibrationError::LabelMismatch {
                index: 0,
                calibrated: "l0".to_string(),
                given: "l1".to_string(),
            }
        );
    }

    /// 異常系: 件数は一致するが末尾のラベル ID を別 ID に差し替えた集合
    /// （REQ-17・REQ-27）。
    #[test]
    fn req27_compare_abstention_rejects_substituted_label() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        let substituted_labels = ["l0", "l1", "l9"];
        let logits = [0.0, 0.0, 0.0];
        let records = vec![CalibrationRecord {
            gold: "l0",
            logits: &logits,
        }];
        let err = compare_abstention(&substituted_labels, &calibration, &records)
            .expect_err("substituted label id must be rejected");
        assert_eq!(
            err,
            CalibrationError::LabelMismatch {
                index: 2,
                calibrated: "l2".to_string(),
                given: "l9".to_string(),
            }
        );
    }

    /// 正常系: 校正時と同一のラベル集合（宣言順一致）は通過する
    /// （REQ-17・REQ-27。上記 2 件の異常系と対をなす境界確認）。
    #[test]
    fn req27_compare_abstention_accepts_identical_labels() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        let logits = [1.0, 0.0, 0.0];
        let records = vec![CalibrationRecord {
            gold: "l0",
            logits: &logits,
        }];
        compare_abstention(&LABELS, &calibration, &records)
            .expect("identical label set must be accepted");
    }

    /// 異常系: 評価レコードが 0 件。
    #[test]
    fn req39_compare_abstention_rejects_empty_records() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        let err = compare_abstention(&LABELS, &calibration, &[])
            .expect_err("empty records must be rejected");
        assert_eq!(err, CalibrationError::EmptyRecords);
    }

    /// 異常系: 未知の gold ラベル（評価器〔`evaluate_single_select`〕へ委譲した
    /// エラーが `CalibrationError::Evaluation` として伝わることを確認する）。
    #[test]
    fn req27_compare_abstention_wraps_unknown_gold_label_from_evaluator() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        let logits = [1.0, 0.0, 0.0];
        let records = vec![CalibrationRecord {
            gold: "unknown-label",
            logits: &logits,
        }];
        let err = compare_abstention(&LABELS, &calibration, &records)
            .expect_err("unknown gold label must be rejected");
        assert!(matches!(
            err,
            CalibrationError::Evaluation(metrics::EvalError::UnknownGoldLabel { index: 0 })
        ));
    }

    /// 境界: 全件保留になる eval（全行の `a` が τ 未満）で `adopted_error()`
    /// が `None`（評価契約: 分母 0 の指標は null）。`unconditional_error()`
    /// は具体値のまま返る。
    #[test]
    fn req24_all_abstain_yields_none_adopted_error() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();
        // τ (=0.6 の position) より低い a=0.1 の行だけを用意する。
        let mut low_rows: Vec<(String, [f64; 3])> = Vec::new();
        low_rows.extend(template_rows(0.1, 0, 3, 2));
        let low_records = as_records(&low_rows);

        let comparison = compare_abstention(&LABELS, &calibration, &low_records).unwrap();
        assert_eq!(
            comparison.with_abstention().outcome_counts.abstain,
            low_records.len() as u64
        );
        assert_eq!(comparison.adopted_error(), None);
        // 保留なしの誤り率は具体値のまま（3 正解・2 不正解 = 2/5）。
        let uncond = comparison.unconditional_error();
        assert_eq!(uncond.numerator(), 2);
        assert_eq!(uncond.denominator(), 5);
    }

    /// TASK-22.1-2: `decide_abstention` は τ と等しい confidence を採用する
    /// （厳密な `>=`）。2 ラベルで `a` を τ ちょうどに一致させた合成入力を使う。
    #[test]
    fn req22_decide_adopts_at_exact_threshold() {
        let labels = ["a", "b"];
        // 5 件中 (n-1)/5 = 0 番目（最小値）が τ になるよう 5 件用意する。
        let rows: Vec<(String, [f64; 2])> = vec![
            ("a".to_string(), [0.0, -5.0]),
            ("a".to_string(), [0.0, -4.0]),
            ("a".to_string(), [0.0, -3.0]),
            ("a".to_string(), [0.0, -2.0]),
            ("a".to_string(), [0.0, -1.0]),
        ];
        let records: Vec<CalibrationRecord<'_>> = rows
            .iter()
            .map(|(gold, logits)| CalibrationRecord {
                gold: gold.as_str(),
                logits,
            })
            .collect();
        let calibration = calibrate(&labels, &records).unwrap();
        // `d = [0.0, a]` の非 gold 側が 0 に近いほど（`|a|` が小さいほど）
        // argmax との差が小さく top1（確信度）が低い。5 件中 (n-1)/5 = 0 番目
        // （昇順の最小値）は最も確信度が低い a=-1.0 の行になる。
        let threshold = calibration.threshold();
        let logits_at_threshold = [0.0, -1.0];
        let decision = decide_abstention(&calibration, &logits_at_threshold).unwrap();
        match decision {
            AbstentionDecision::Adopt { confidence, .. } => {
                assert!(approx_eq(confidence, threshold));
            }
            AbstentionDecision::Abstain { .. } => panic!("must adopt at exact threshold"),
        }
    }

    /// TASK-22.1-2: τ 未満は保留。
    #[test]
    fn req22_decide_abstains_below_threshold() {
        let labels = ["a", "b"];
        let rows: Vec<(String, [f64; 2])> = vec![
            ("a".to_string(), [0.0, -5.0]),
            ("a".to_string(), [0.0, -4.0]),
            ("a".to_string(), [0.0, -3.0]),
            ("a".to_string(), [0.0, -2.0]),
            ("a".to_string(), [0.0, -1.0]),
        ];
        let records: Vec<CalibrationRecord<'_>> = rows
            .iter()
            .map(|(gold, logits)| CalibrationRecord {
                gold: gold.as_str(),
                logits,
            })
            .collect();
        let calibration = calibrate(&labels, &records).unwrap();
        // τ は validation 5 件の中で最も確信度が低い行（a=-1.0）の top1。
        // 完全に同値（a=0.0）のロジットは 2 ラベルで理論上最小の
        // top1=0.5（`top1_probability` は `sum_exp <= n_labels` から
        // `>= 1/n_labels` を返す）になり、非同値の a=-1.0 行の
        // 確信度（> 0.5）より明確に低い。τ 未満はしきい値ちょうどの
        // `Adopt` を許容せず `Abstain` のみを期待する（この境界一致の
        // 挙動は `req22_decide_adopts_at_exact_threshold` で別途確認済み）。
        let logits_low = [0.0, 0.0];
        let decision = decide_abstention(&calibration, &logits_low).unwrap();
        match decision {
            AbstentionDecision::Abstain { confidence } => {
                assert!(confidence < calibration.threshold());
            }
            AbstentionDecision::Adopt { confidence, .. } => {
                panic!(
                    "confidence {confidence} below threshold {} must abstain, not adopt",
                    calibration.threshold()
                );
            }
        }
    }

    /// TASK-22.1-2: argmax の同値は宣言順の先頭を採用する。
    #[test]
    fn req22_decide_argmax_tie_takes_first_declared() {
        let labels = ["a", "b", "c"];
        let rows: Vec<(String, [f64; 3])> = vec![
            ("a".to_string(), [1.0, -1.0, -1.0]),
            ("a".to_string(), [1.0, -1.0, -1.0]),
            ("a".to_string(), [1.0, -1.0, -1.0]),
        ];
        let records: Vec<CalibrationRecord<'_>> = rows
            .iter()
            .map(|(gold, logits)| CalibrationRecord {
                gold: gold.as_str(),
                logits,
            })
            .collect();
        let calibration = calibrate(&labels, &records).unwrap();
        // 完全に同値のロジットは `top1` が最小（3 ラベルなら 1/3）になり、
        // 通常は保留（`Abstain`）される。`Abstain` はどの添字を argmax と
        // したかを外部へ返さないため（保留込み評価では argmax を使わない）、
        // argmax の宣言順優先は [`crate::calibration::preprocess_logits`]
        // （`decide_abstention` が内部で使う共通経路）を直接確認する。
        let tied_logits = [0.0, 0.0, 0.0];
        let (_, argmax_index) =
            calibration::preprocess_logits(0, &tied_logits, calibration.n_labels()).unwrap();
        assert_eq!(argmax_index, 0);
    }

    /// `to_outcome` がラベル添字から `Outcome::Label` を正しく組み立てる
    /// （`calibration.labels()` から解決する。REQ-17・REQ-27）。
    #[test]
    fn req22_to_outcome_builds_label_and_abstain() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();

        let adopt = AbstentionDecision::Adopt {
            label_index: 1,
            confidence: 0.9,
        };
        assert_eq!(
            adopt.to_outcome(&calibration).unwrap(),
            Outcome::Label("l1".to_string())
        );
        let abstain = AbstentionDecision::Abstain { confidence: 0.1 };
        assert_eq!(abstain.to_outcome(&calibration).unwrap(), Outcome::Abstain);
    }

    /// `to_outcome` は範囲外のラベル添字を `[]` ではなく `Internal` で拒否する。
    #[test]
    fn req39_to_outcome_rejects_out_of_range_label_index() {
        let validation_rows = c1_validation();
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&LABELS, &validation_records).unwrap();

        let adopt = AbstentionDecision::Adopt {
            label_index: 99,
            confidence: 0.9,
        };
        let err = adopt
            .to_outcome(&calibration)
            .expect_err("out-of-range label index must be rejected");
        assert!(matches!(err, CalibrationError::Internal { .. }));
    }

    /// `to_outcome` は `calibration.labels()` から解決するため、校正時と
    /// 異なるラベル集合を外部から渡す経路自体が存在しない（REQ-17・REQ-27。
    /// codex/review 指摘: 添字が範囲内かだけの確認では校正時のラベル集合との
    /// 不一致を検出できず、誤った予測を正常な結果として評価しうる問題への
    /// 対応）。校正時のラベルが宣言順どおりに解決されることを、並べ替えた
    /// 集合で校正した `Calibration` に対して確認する。
    #[test]
    fn req27_to_outcome_resolves_from_calibration_labels_only() {
        let permuted_labels = ["l2", "l0", "l1"];
        let validation_rows = c1_validation();
        // gold はそのままに、ラベル宣言順だけを並べ替えて校正する。
        let validation_records = as_records(&validation_rows);
        let calibration = calibrate(&permuted_labels, &validation_records).unwrap();

        // label_index=0 は permuted_labels 宣言順の "l2"（LABELS の "l0" では
        // ない）。`to_outcome` は `calibration.labels()`（= permuted_labels）
        // からのみ解決するため、渡しようのない外部 `labels` 引数によって
        // 誤ったラベルへ解決される余地がない。
        let adopt = AbstentionDecision::Adopt {
            label_index: 0,
            confidence: 0.9,
        };
        assert_eq!(
            adopt.to_outcome(&calibration).unwrap(),
            Outcome::Label("l2".to_string())
        );
    }
}
