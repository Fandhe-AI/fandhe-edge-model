//! 行ごとの正誤 → McNemar 検定 → 下限基準（majority）に対する有意性判定。
//!
//! [`crate::mcnemar`]（統計計算コア。TASK-25.1-1・issue #64）と
//! [`crate::baseline`]（下限基準の予測生成）を接続し、CLI の `evaluate`
//! 工程（REQ-33）が「モデルは下限基準（majority）を有意に上回るか」を
//! 判定できるようにする（REQ-25 正常系・TASK-25.1-2・issue #65）。
//!
//! # 対象外（本 issue の範囲外）
//!
//! - Holm 補正（複数候補比較。REQ-26）は [`judge`] に `p` を引数で渡す形に
//!   しておき、TASK-25.3 で補正後の p 値でも呼べるようにする
//! - 件数不足による「判定不能」は未実装（TASK-25.2）。[`BaselineVerdict`]
//!   は `#[non_exhaustive]` にしてあり、後から `Undecidable` を追加できる
//! - CLI の JSON 出力・終了コードへの写像は行わない（TASK-33.x / TASK-18.3）

use crate::baseline::{self, BaselineError};
use crate::mcnemar::{self, McNemarExact, PValue, PairedCounts};
use crate::metrics::{EvalRecord, Outcome};

/// 下限基準（majority）に対する有意性判定の α（評価契約で固定。REQ-25）。
///
/// `.claude/rules/evaluation-contract.md`「有意性・指標」で定めた値であり、
/// 呼び出し側が変更できるよう引数にはしない（変更は評価契約の変更にあたり、
/// main の設計判断とユーザー承認を要する）。
pub const SIGNIFICANCE_ALPHA: f64 = 0.05;

/// 下限基準（majority）に対する判定。
///
/// `#[non_exhaustive]` にしてあるのは、件数不足の「判定不能」
/// （TASK-25.2・`Undecidable`）を後から追加できるようにするため。
/// 現時点では判定不能は未実装（実装済みを装わない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BaselineVerdict {
    /// 候補が下限基準を有意に上回る（`b > c` かつ `p < `[`SIGNIFICANCE_ALPHA`]）。
    SignificantlyBetter,
    /// 有意に上回るとは言えない（`b <= c`、または `p >= `[`SIGNIFICANCE_ALPHA`]）。
    NotSignificantlyBetter,
}

/// `α = `[`SIGNIFICANCE_ALPHA`]`` での有意性判定（純粋関数）。
///
/// 規則: `b > c` かつ `p < α` なら [`BaselineVerdict::SignificantlyBetter`]、
/// それ以外は [`BaselineVerdict::NotSignificantlyBetter`]
/// （PoC-10 `stats_mcnemar.py` の「有意に上回る」の定義から Holm 補正を
/// 除いたもの）。
///
/// 比較は厳密な `<` とし、許容差は付けない。評価契約（REQ-24）が
/// 「p < 0.05」を境界として定めており、判定を緩めない設計判断として
/// 許容差を持ち込まないのが正しい（`.claude/rules/evaluation-contract.md`
/// 「有意性・指標」）。
///
/// 数学的には正確検定の p 値は `k / 2^n` の形の二進有理数で、`0.05`
/// （= 1/20。分母に素因数 5 を含む）とちょうど一致することは原理上ない。
/// ただし [`mcnemar::two_sided_p_value`] の実装は `ln`/`exp`（f64 の
/// libm）による近似計算であり、返る値はその厳密な有理数そのものではなく
/// 浮動小数近似値である（Review 指摘。TASK-25.1-2・issue #65。以前の
/// 記述は実装と不整合だった）。したがって上記の数学的事実だけを根拠に
/// 「境界値問題が実質的に生じない」とは言えない。実際に `0.05` へ
/// 極めて近づく `(b, c)` では近似誤差により判定が理論値と入れ替わる
/// 余地があるが、`<` を許容差なしで使うという設計判断自体は変えない
/// （評価契約の変更にあたり、緩和には main の設計判断とユーザー承認を要する）。
pub fn judge(b: u64, c: u64, p: PValue) -> BaselineVerdict {
    if b > c && p.value() < SIGNIFICANCE_ALPHA {
        BaselineVerdict::SignificantlyBetter
    } else {
        BaselineVerdict::NotSignificantlyBetter
    }
}

/// 1 件の [`Outcome`] が `gold` と一致するかを判定する。
///
/// `Outcome::Label(l)` で `l == gold` の場合のみ正解。未知ラベル・
/// `Invalid`・`Abstain`・`Error` はすべて不正解として扱う（PoC-10:
/// 予測の欠落・`status` が ok 以外・ラベル違いはすべて不正解）。
fn is_correct(gold: &str, outcome: &Outcome) -> bool {
    matches!(outcome, Outcome::Label(l) if l == gold)
}

/// 評価レコード列から、行ごとの正誤（`Outcome` が `gold` と一致するか）を求める。
///
/// - `labels` 自体の検証（空・空 ID・重複・`MAX_LABELS` 超過）は
///   [`baseline::validate_label_order`] に委ね、[`fit_majority`]（下限基準の
///   予測生成）と同じ規則に揃える（Review 指摘。TASK-25.1-2・issue #65。
///   以前は `labels.contains(&record.gold)` の素通し判定のみで、`labels`
///   が空の場合に本来の [`crate::metrics::EvalError::EmptyLabels`] ではなく
///   `UnknownGoldLabel { index: 0 }` を返し、重複ラベルも検出できなかった）
/// - gold がラベル集合に存在しない場合は [`BaselineError::UnknownGoldLabel`]
///   を返す（`evaluate_single_select` と同じ fail-closed。PoC-10 の
///   `stats_mcnemar.py` のように黙って除外しない。正解側の欠陥はデータ契約層
///   の検査で扱う）
/// - `records` が空の場合は [`BaselineError::EmptyRecords`]（評価済みを
///   装わない。件数不足の判定不能は TASK-25.2 の担当）
///
/// 入力は `&` 参照でのみ受け取り、書き換えない（REQ-27）。
///
/// [`fit_majority`]: crate::baseline::fit_majority
pub fn correctness(
    labels: &[&str],
    records: &[EvalRecord<'_>],
) -> Result<Vec<bool>, BaselineError> {
    let index = baseline::validate_label_order(labels)?;

    if records.is_empty() {
        return Err(BaselineError::EmptyRecords);
    }

    let mut result = Vec::with_capacity(records.len());
    for (i, record) in records.iter().enumerate() {
        if !index.contains_key(record.gold) {
            return Err(BaselineError::UnknownGoldLabel { index: i });
        }
        result.push(is_correct(record.gold, record.outcome));
    }
    Ok(result)
}

/// 候補・下限基準を対応させた 1 行分の入力。
///
/// index のずれをスライスの並行走査ではなく型で防ぐため、候補・下限基準の
/// 予測を 1 つの構造体にまとめて渡す。
#[derive(Debug, Clone, Copy)]
pub struct PairedRecord<'a> {
    /// 正解ラベル ID。
    pub gold: &'a str,
    /// 候補モデルの推論結果。
    pub candidate: &'a Outcome,
    /// 下限基準（majority 等）の推論結果。
    pub baseline: &'a Outcome,
}

/// 候補・下限基準の比較結果（McNemar 検定＋有意性判定）。
///
/// フィールドは非公開にし、アクセサ経由で公開する（[`mcnemar::PValue`] を
/// 含む [`McNemarExact`] を保持するため `PartialEq` は derive しない）。
/// PoC `compare()` のキー（n・cand_correct・base_correct・b_cand_only・
/// c_base_only・p_exact_two_sided）と 1 対 1 に対応させ、CLI（TASK-33）が
/// JSON へそのまま写せるようにしている。
#[derive(Debug, Clone, Copy)]
pub struct BaselineComparison {
    counts: PairedCounts,
    test: McNemarExact,
    verdict: BaselineVerdict,
}

impl BaselineComparison {
    /// 対応のある正誤の集計（n・both_correct・b・c・both_wrong）。
    pub fn counts(&self) -> PairedCounts {
        self.counts
    }

    /// 候補モデルの正解件数（`both_correct + b_candidate_only`）。
    pub fn candidate_correct(&self) -> u64 {
        // `paired_counts` が checked 演算で導出した非負値どうしの和で、
        // 生成元の `PairedCounts::n` がオーバーフローしないことを検証済み
        // （`mcnemar::paired_counts`）ため、ここでの再計算も溢れない。
        self.counts.both_correct + self.counts.b_candidate_only
    }

    /// 下限基準の正解件数（`both_correct + c_baseline_only`）。
    pub fn baseline_correct(&self) -> u64 {
        self.counts.both_correct + self.counts.c_baseline_only
    }

    /// McNemar の正確検定（両側）の結果。`p_two_sided()` で p 値を取り出す。
    pub fn test(&self) -> McNemarExact {
        self.test
    }

    /// α = 0.05 での有意性判定。
    pub fn verdict(&self) -> BaselineVerdict {
        self.verdict
    }
}

/// ラベル集合・候補・下限基準の対応データから [`BaselineComparison`] を求める。
///
/// 手順: ラベル検証 → 行ごとの正誤（候補・下限基準それぞれ [`is_correct`]
/// と同じ規則）→ [`mcnemar::paired_counts`] → [`mcnemar::mcnemar_exact_two_sided`]
/// → [`judge`]。
///
/// `labels` 自体の検証（空・空 ID・重複・`MAX_LABELS` 超過）は
/// [`baseline::validate_label_order`] に委ね、[`correctness`]・
/// [`fit_majority`] と同じ規則に揃える（Review 指摘。TASK-25.1-2・issue #65）。
///
/// gold がラベル集合に無い場合は [`BaselineError::UnknownGoldLabel`]、
/// `records` が空の場合は [`BaselineError::EmptyRecords`] を返す
/// （[`correctness`] と同じ規則）。
///
/// [`fit_majority`]: crate::baseline::fit_majority
pub fn compare_with_baseline(
    labels: &[&str],
    records: &[PairedRecord<'_>],
) -> Result<BaselineComparison, BaselineError> {
    let index = baseline::validate_label_order(labels)?;

    if records.is_empty() {
        return Err(BaselineError::EmptyRecords);
    }

    let mut candidate_correct = Vec::with_capacity(records.len());
    let mut baseline_correct = Vec::with_capacity(records.len());
    for (i, record) in records.iter().enumerate() {
        if !index.contains_key(record.gold) {
            return Err(BaselineError::UnknownGoldLabel { index: i });
        }
        candidate_correct.push(is_correct(record.gold, record.candidate));
        baseline_correct.push(is_correct(record.gold, record.baseline));
    }

    let counts = mcnemar::paired_counts(&candidate_correct, &baseline_correct)?;
    let test = mcnemar::mcnemar_exact_two_sided(counts.b_candidate_only, counts.c_baseline_only)?;
    let verdict = judge(
        counts.b_candidate_only,
        counts.c_baseline_only,
        test.p_two_sided(),
    );

    Ok(BaselineComparison {
        counts,
        test,
        verdict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcnemar::mcnemar_exact_two_sided;

    /// (13,4) は α のすぐ下（p = 0.049041748046875 < 0.05）→ Better。
    #[test]
    fn judge_significantly_better_just_under_alpha() {
        let result = mcnemar_exact_two_sided(13, 4).unwrap();
        assert!(result.p_two_sided().value() < SIGNIFICANCE_ALPHA);
        assert_eq!(
            judge(13, 4, result.p_two_sided()),
            BaselineVerdict::SignificantlyBetter
        );
    }

    /// (22,10) は α のすぐ上（p ≈ 0.0501 >= 0.05）→ Not。
    #[test]
    fn judge_not_significantly_better_just_over_alpha() {
        let result = mcnemar_exact_two_sided(22, 10).unwrap();
        assert!(result.p_two_sided().value() >= SIGNIFICANCE_ALPHA);
        assert_eq!(
            judge(22, 10, result.p_two_sided()),
            BaselineVerdict::NotSignificantlyBetter
        );
    }

    /// (6,0) は p = 0.03125 → Better。
    #[test]
    fn judge_significantly_better_small_case() {
        let result = mcnemar_exact_two_sided(6, 0).unwrap();
        assert_eq!(
            judge(6, 0, result.p_two_sided()),
            BaselineVerdict::SignificantlyBetter
        );
    }

    /// (5,0) は p = 0.0625 → Not。
    #[test]
    fn judge_not_significantly_better_small_case() {
        let result = mcnemar_exact_two_sided(5, 0).unwrap();
        assert_eq!(
            judge(5, 0, result.p_two_sided()),
            BaselineVerdict::NotSignificantlyBetter
        );
    }

    /// (9,3) は p = 0.14599609375 → Not。
    #[test]
    fn judge_not_significantly_better_moderate_p() {
        let result = mcnemar_exact_two_sided(9, 3).unwrap();
        assert_eq!(
            judge(9, 3, result.p_two_sided()),
            BaselineVerdict::NotSignificantlyBetter
        );
    }

    /// (3,12) は b < c（下限基準のほうが強い）→ p が小さくても Not（方向の確認）。
    #[test]
    fn judge_not_significantly_better_when_baseline_wins() {
        let result = mcnemar_exact_two_sided(3, 12).unwrap();
        // p = 0.03515625 < 0.05 だが b < c なので Not。
        assert!(result.p_two_sided().value() < SIGNIFICANCE_ALPHA);
        assert_eq!(
            judge(3, 12, result.p_two_sided()),
            BaselineVerdict::NotSignificantlyBetter
        );
    }

    /// (0,0) は p = 1.0 → Not。
    #[test]
    fn judge_not_significantly_better_zero_discordant() {
        let result = mcnemar_exact_two_sided(0, 0).unwrap();
        assert_eq!(
            judge(0, 0, result.p_two_sided()),
            BaselineVerdict::NotSignificantlyBetter
        );
    }

    /// `correctness`: `Label` 一致は正解。
    #[test]
    fn correctness_label_match_is_correct() {
        let labels = ["A", "B"];
        let outcome = Outcome::Label("A".to_string());
        let records = [EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        assert_eq!(correctness(&labels, &records).unwrap(), vec![true]);
    }

    /// `correctness`: `Label` 不一致は不正解。
    #[test]
    fn correctness_label_mismatch_is_incorrect() {
        let labels = ["A", "B"];
        let outcome = Outcome::Label("B".to_string());
        let records = [EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        assert_eq!(correctness(&labels, &records).unwrap(), vec![false]);
    }

    /// `correctness`: ラベル集合外の予測は不正解。
    #[test]
    fn correctness_out_of_set_label_is_incorrect() {
        let labels = ["A", "B"];
        let outcome = Outcome::Label("Z".to_string());
        let records = [EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        assert_eq!(correctness(&labels, &records).unwrap(), vec![false]);
    }

    /// `correctness`: `Invalid`・`Abstain`・`Error` はすべて不正解。
    #[test]
    fn correctness_non_label_outcomes_are_incorrect() {
        let labels = ["A"];
        let invalid = Outcome::Invalid;
        let abstain = Outcome::Abstain;
        let error = Outcome::Error;
        let records = [
            EvalRecord {
                gold: "A",
                outcome: &invalid,
            },
            EvalRecord {
                gold: "A",
                outcome: &abstain,
            },
            EvalRecord {
                gold: "A",
                outcome: &error,
            },
        ];
        assert_eq!(
            correctness(&labels, &records).unwrap(),
            vec![false, false, false]
        );
    }

    /// `correctness`: 未知の gold はエラー。
    #[test]
    fn correctness_unknown_gold_is_error() {
        let labels = ["A", "B"];
        let outcome = Outcome::Label("A".to_string());
        let records = [EvalRecord {
            gold: "Z",
            outcome: &outcome,
        }];
        let err = correctness(&labels, &records).unwrap_err();
        assert_eq!(err, BaselineError::UnknownGoldLabel { index: 0 });
    }

    /// `correctness`: 空の records はエラー。
    #[test]
    fn correctness_empty_records_is_error() {
        let labels = ["A"];
        let err = correctness(&labels, &[]).unwrap_err();
        assert_eq!(err, BaselineError::EmptyRecords);
    }

    /// `compare_with_baseline`: 空の records はエラー。
    #[test]
    fn compare_with_baseline_empty_records_is_error() {
        let labels = ["A", "B"];
        let err = compare_with_baseline(&labels, &[]).unwrap_err();
        assert_eq!(err, BaselineError::EmptyRecords);
    }

    /// `compare_with_baseline`: 未知の gold はエラー。
    #[test]
    fn compare_with_baseline_unknown_gold_is_error() {
        let labels = ["A", "B"];
        let cand = Outcome::Label("A".to_string());
        let base = Outcome::Label("A".to_string());
        let records = [PairedRecord {
            gold: "Z",
            candidate: &cand,
            baseline: &base,
        }];
        let err = compare_with_baseline(&labels, &records).unwrap_err();
        assert_eq!(err, BaselineError::UnknownGoldLabel { index: 0 });
    }

    /// 評価契約（REQ-27）: 入力スライスが評価の前後で変わらないこと
    /// （clone との比較で、書き換えられていないことを確認する）。
    #[test]
    fn compare_with_baseline_does_not_mutate_input() {
        let labels = ["A", "B"];
        let cand_correct = Outcome::Label("A".to_string());
        let cand_wrong = Outcome::Label("B".to_string());
        let base_correct = Outcome::Label("A".to_string());
        let records = [
            PairedRecord {
                gold: "A",
                candidate: &cand_correct,
                baseline: &base_correct,
            },
            PairedRecord {
                gold: "A",
                candidate: &cand_wrong,
                baseline: &base_correct,
            },
        ];
        let before: Vec<(&str, Outcome, Outcome)> = records
            .iter()
            .map(|r| (r.gold, r.candidate.clone(), r.baseline.clone()))
            .collect();
        let _ = compare_with_baseline(&labels, &records).unwrap();
        let after: Vec<(&str, Outcome, Outcome)> = records
            .iter()
            .map(|r| (r.gold, r.candidate.clone(), r.baseline.clone()))
            .collect();
        assert_eq!(before, after);
    }

    /// Review 指摘（issue #65）: `correctness` は `labels` が空の場合
    /// [`fit_majority`] と同じ [`crate::metrics::EvalError::EmptyLabels`]
    /// を返す（以前は `UnknownGoldLabel { index: 0 }` になっていた）。
    ///
    /// [`fit_majority`]: crate::baseline::fit_majority
    #[test]
    fn correctness_empty_labels_is_eval_error() {
        let labels: [&str; 0] = [];
        let outcome = Outcome::Label("A".to_string());
        let records = [EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        let err = correctness(&labels, &records).unwrap_err();
        assert_eq!(
            err,
            BaselineError::Eval(crate::metrics::EvalError::EmptyLabels)
        );
    }

    /// Review 指摘（issue #65）: `correctness` は重複ラベルを検出する
    /// （以前は `labels.contains` の素通し判定のみで重複を検出できなかった）。
    #[test]
    fn correctness_duplicate_labels_is_eval_error() {
        let labels = ["A", "A"];
        let outcome = Outcome::Label("A".to_string());
        let records = [EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        let err = correctness(&labels, &records).unwrap_err();
        assert_eq!(
            err,
            BaselineError::Eval(crate::metrics::EvalError::DuplicateLabel {
                label: "A".to_string()
            })
        );
    }

    /// Review 指摘（issue #65）: `compare_with_baseline` も `labels` が
    /// 空の場合 `EmptyLabels` を返す（`correctness` と同じ規則）。
    #[test]
    fn compare_with_baseline_empty_labels_is_eval_error() {
        let labels: [&str; 0] = [];
        let cand = Outcome::Label("A".to_string());
        let base = Outcome::Label("A".to_string());
        let records = [PairedRecord {
            gold: "A",
            candidate: &cand,
            baseline: &base,
        }];
        let err = compare_with_baseline(&labels, &records).unwrap_err();
        assert_eq!(
            err,
            BaselineError::Eval(crate::metrics::EvalError::EmptyLabels)
        );
    }
}
