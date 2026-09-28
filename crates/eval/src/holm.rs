//! Holm 法による多重比較補正（REQ-25 境界値・TASK-25.3・issue #67）。
//!
//! 評価契約（`.claude/rules/evaluation-contract.md`「有意性・指標」）は
//! 「複数候補の比較は Holm 補正を行う」と定めている。本モジュールは
//! [`crate::significance`]（下限基準 1 候補ずつの McNemar 検定＋α 判定。
//! REQ-25 正常系・TASK-25.1-2・issue #65）が返す生の p 値を、事前登録した
//! 族サイズ（比較する候補の総数）で Holm 補正し、補正後の p 値で判定を
//! 出し直す。PoC-10 `03-poc/scratch-classifier/scripts/stats_mcnemar.py`
//! の `holm()` と同じ手順（PoC-10・PoC-25 で事前登録し実測で採用）を
//! Rust（std のみ）へ移植したもの。
//!
//! # 手順
//!
//! 1. p 値を昇順に安定ソートする（同値は入力順を保つ）
//! 2. 0 始まりで k 番目（ソート後の順位）の p 値に `(m - k)` を掛ける
//!    （`m` は事前登録した族サイズ）
//! 3. 累積最大で単調化する（小さい p 値の補正後の値が、それより大きい
//!    p 値の補正後の値を下回らないようにする）
//! 4. 1 で打ち切る
//!
//! 戻り値は入力順に並べ直す。
//!
//! # 事前登録の規則（PoC-10・PoC-25 が実測で採用した規則）
//!
//! - 族サイズ `m` は呼び出し側が事前登録した値を明示的に渡す。候補が実行時に
//!   脱落しても `m` は登録値のまま補正する（保守側。脱落した候補は
//!   「有意に上回らない」と数える。この既定は呼び出し側〔選定・TASK-18.3〕の
//!   責務であり、本モジュールは渡された候補だけを補正する）
//! - 「有意に上回る」は、`b > c` かつ Holm 補正後の p が α 未満であることと
//!   する（[`crate::significance::judge`] と同じ規則を補正後の p 値で適用）
//!
//! # 対象外（本 issue の範囲外）
//!
//! - CLI の JSON 出力・終了コードへの写像（TASK-33.x・issue #140）
//! - 選定への統合（TASK-18.3）
//! - 「3 seed すべてで有意」の集約
//! - 必要件数の算出（Connor 式・TASK-25.2）
//! - `simple_rule` 下限基準
//!
//! # 評価契約との関係
//!
//! - α（[`crate::significance::SIGNIFICANCE_ALPHA`]）・比較規則（厳密な `<`・
//!   許容差なし）・「有意に上回る」の向き（`b > c`）は変更しない。本モジュールは
//!   既存の契約を実装するものであり、緩和・変更ではない
//! - 件数不足による `Undeterminable` は補正後も維持される
//!   （[`compare_candidates_with_holm`] のドキュメント参照）
//!
//! # 資源上限（REQ-39）
//!
//! [`MAX_FAMILY_SIZE`] を `Vec` 確保の前に検証する。値はデータ契約層の
//! 確定値が無い段階の暫定値（[`crate::mcnemar::MAX_DISCORDANT_PAIRS`]・
//! [`crate::significance::MAX_EVAL_RECORDS`] と同様。実測に基づく調整は
//! 後続 TASK）。
//!
//! # 決定性
//!
//! ソートは `f64::total_cmp` による安定ソートで、同値の p 値は入力順を保つ。
//! 累積最大の性質上、同値の p 値はソート順に関わらず補正後の値が等しくなる。
//! 並列化しない。

use std::fmt;

use crate::mcnemar::PValue;
use crate::significance::{self, BaselineComparison, BaselineVerdict};

/// 1 回の補正で扱う族サイズ（比較する候補の総数）の上限（REQ-39）。
///
/// この上限の下では `(m - k) as f64` が整数として厳密に表現できる
/// （`2^53` 未満）。[`crate::mcnemar::MAX_DISCORDANT_PAIRS`]・
/// [`crate::significance::MAX_EVAL_RECORDS`] と同様、データ契約層の確定値が
/// 無い段階の暫定値。
pub const MAX_FAMILY_SIZE: usize = 10_000;

/// Holm 補正の族サイズ（事前登録した、比較する候補の総数）。
///
/// 0 を拒否し、壊れた値（未設定・0 候補の族）を表現できないようにする
/// （`.claude/rules/coding-rust.md`「公開 API・型設計」。
/// [`crate::significance::RequiredSampleSize`] と同じ型設計）。
///
/// 既定値は持たない。呼び出し側が事前登録した値を必ず明示的に渡す
/// （「候補が脱落しても m は登録値のまま補正する」という保守側の規則を
/// 型で強制するため、実行時の候補数から自動算出する経路は用意しない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FamilySize(usize);

impl FamilySize {
    /// `m` が 1 以上なら `Some` を返す。0 は「族サイズが未設定」と
    /// 区別できず壊れた値になるため拒否する。
    pub fn new(m: usize) -> Option<Self> {
        if m == 0 { None } else { Some(Self(m)) }
    }

    /// 中身の `usize` 値を取り出す。
    pub fn get(&self) -> usize {
        self.0
    }
}

/// [`holm_adjust`]・[`compare_candidates_with_holm`] が返しうるエラー。
///
/// [`crate::mcnemar::McNemarError`]・[`crate::baseline::BaselineError`]とは
/// 意図的に分けている。呼び出し側が終了コード（REQ-21）へ写す際に判別
/// しやすくするため（写像そのものは本モジュールの範囲外）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HolmError {
    /// p 値が 0 件。評価済みを装わない（fail-closed）。
    EmptyPValues,
    /// 族サイズ `m` が渡した p 値の件数より小さい
    /// （`m` を黙って件数に合わせて補正しない。事前登録した `m` と
    /// 実際に渡した候補数の食い違いを検出する）。
    FamilySizeTooSmall {
        /// 渡された族サイズ。
        family_size: usize,
        /// 渡された p 値・候補の件数。
        n_tests: usize,
    },
    /// 族サイズ `m` が [`MAX_FAMILY_SIZE`] を超える（REQ-39）。
    FamilySizeTooLarge {
        /// 渡された族サイズ。
        family_size: usize,
        /// 上限値。
        limit: usize,
    },
    /// 理論上到達しないはずの内部不整合（非有限値の算出等）。
    /// fail-closed のガード。データ本文は含めない。
    Internal {
        /// 診断用の詳細。
        detail: String,
    },
}

impl fmt::Display for HolmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HolmError::EmptyPValues => write!(f, "no p-values were provided for Holm correction"),
            HolmError::FamilySizeTooSmall {
                family_size,
                n_tests,
            } => write!(
                f,
                "family size {family_size} is smaller than the number of tests {n_tests}"
            ),
            HolmError::FamilySizeTooLarge { family_size, limit } => {
                write!(f, "family size {family_size} exceeds limit {limit}")
            }
            HolmError::Internal { detail } => {
                write!(f, "internal holm correction error: {detail}")
            }
        }
    }
}

impl std::error::Error for HolmError {}

/// p 値の列を Holm 法で補正する（モジュールドキュメントの「手順」参照）。
///
/// `family_size` は事前登録した族サイズ `m`。`p_values.len()` より
/// 小さい場合は [`HolmError::FamilySizeTooSmall`] を返す
/// （`m` を黙って件数に合わせない。`m` が件数より大きい場合〔候補の脱落〕は、
/// 末尾に `p = 1` を足した族と同じ結果になる〔保守側の扱い〕）。
///
/// 検証順は 空 → 上限超過 → 件数不足 の順で、いずれも `Vec` の確保前に行う
/// （REQ-39）。
///
/// 戻り値は入力順に並べる。同値の p 値は入力順を入れ替えても補正後の値が
/// 等しくなる（累積最大の性質による。決定性の根拠）。
pub fn holm_adjust(p_values: &[PValue], family_size: FamilySize) -> Result<Vec<PValue>, HolmError> {
    if p_values.is_empty() {
        return Err(HolmError::EmptyPValues);
    }

    let m = family_size.get();
    if m > MAX_FAMILY_SIZE {
        return Err(HolmError::FamilySizeTooLarge {
            family_size: m,
            limit: MAX_FAMILY_SIZE,
        });
    }
    if m < p_values.len() {
        return Err(HolmError::FamilySizeTooSmall {
            family_size: m,
            n_tests: p_values.len(),
        });
    }

    // (入力インデックス, p 値) の組を p 値の昇順で安定ソートする。
    // `sort_by` は安定ソートのため、同値は入力順（インデックス昇順）を保つ。
    let mut indexed: Vec<(usize, f64)> = p_values
        .iter()
        .enumerate()
        .map(|(idx, p)| (idx, p.value()))
        .collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut adjusted: Vec<Option<PValue>> = vec![None; p_values.len()];
    let mut running = 0.0_f64;
    for (rank, &(idx, p)) in indexed.iter().enumerate() {
        // rank は 0 始まり。乗数は (m - rank)。m >= p_values.len() > rank を
        // 上で検証済みのため、乗数は 1 以上で桁溢れしない。
        let multiplier = (m - rank) as f64;
        let candidate = (multiplier * p).min(1.0);
        running = running.max(candidate);

        let value = PValue::new(running).ok_or_else(|| HolmError::Internal {
            detail: format!("holm-adjusted p value is not finite in [0, 1]: {running}"),
        })?;

        let slot = adjusted.get_mut(idx).ok_or_else(|| HolmError::Internal {
            detail: format!("holm adjustment index out of range: {idx}"),
        })?;
        *slot = Some(value);
    }

    adjusted
        .into_iter()
        .map(|slot| {
            slot.ok_or_else(|| HolmError::Internal {
                detail: "holm adjustment left an input index unassigned".to_string(),
            })
        })
        .collect()
}

/// Holm 補正を適用した 1 候補分の比較結果。
///
/// フィールドは非公開で、アクセサ経由で公開する（[`PValue`] を含む
/// [`BaselineComparison`] を保持するため `PartialEq` は derive しない。
/// [`BaselineComparison`] と同じ設計）。
#[derive(Debug, Clone, Copy)]
pub struct HolmComparison {
    comparison: BaselineComparison,
    adjusted_p: PValue,
    verdict: BaselineVerdict,
}

impl HolmComparison {
    /// 補正前の McNemar 比較結果（`b`・`c`・未補正の p 値・未補正の判定を含む）。
    pub fn comparison(&self) -> BaselineComparison {
        self.comparison
    }

    /// Holm 補正後の p 値。
    pub fn adjusted_p(&self) -> PValue {
        self.adjusted_p
    }

    /// Holm 補正後の p 値による有意性判定（α = [`crate::significance::SIGNIFICANCE_ALPHA`]）。
    ///
    /// 評価件数が必要件数未満だった場合（[`BaselineVerdict::Undeterminable`]）
    /// は、補正後の p 値がどれだけ小さくても判定不能のまま維持される
    /// （[`crate::significance::judge`] の優先順位規則をそのまま適用する
    /// ため）。
    pub fn verdict(&self) -> BaselineVerdict {
        self.verdict
    }
}

/// 複数候補の [`BaselineComparison`] を 1 つの族として Holm 補正し、
/// 候補ごとの補正後 p 値・判定を返す。
///
/// 手順: 候補ごとの生の p 値（[`BaselineComparison::test`]の
/// `p_two_sided()`）を集める → [`holm_adjust`] で補正する → 候補ごとに
/// [`crate::significance::judge`]（`b`・`c`・補正後の p・評価件数 N・
/// [`BaselineComparison::required`]）で判定を出し直す。
///
/// 戻り値は入力順。`comparisons` が空の場合は [`HolmError::EmptyPValues`]、
/// `family_size` が `comparisons.len()` 未満の場合は
/// [`HolmError::FamilySizeTooSmall`]（[`holm_adjust`] と同じ規則）を返す。
///
/// 脱落候補（事前登録した族に含まれるが今回評価しなかった候補）は
/// 呼び出し側が本関数へ渡さない。`family_size` を固定したまま渡す候補数を
/// 減らすことで、補正が保守側（脱落候補ぶんの乗数を空けたまま）に倒れる
/// （モジュールドキュメントの「事前登録の規則」参照。脱落候補自体を
/// 「有意でない」と数えるのは呼び出し側〔選定・TASK-18.3〕の責務）。
pub fn compare_candidates_with_holm(
    comparisons: &[BaselineComparison],
    family_size: FamilySize,
) -> Result<Vec<HolmComparison>, HolmError> {
    if comparisons.is_empty() {
        return Err(HolmError::EmptyPValues);
    }

    let raw_p: Vec<PValue> = comparisons.iter().map(|c| c.test().p_two_sided()).collect();
    let adjusted_p = holm_adjust(&raw_p, family_size)?;

    let mut result = Vec::with_capacity(comparisons.len());
    for (comparison, adjusted) in comparisons.iter().zip(adjusted_p.iter()) {
        let counts = comparison.counts();
        let verdict = significance::judge(
            counts.b_candidate_only,
            counts.c_baseline_only,
            *adjusted,
            counts.n,
            comparison.required(),
        );
        result.push(HolmComparison {
            comparison: *comparison,
            adjusted_p: *adjusted,
            verdict,
        });
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcnemar::mcnemar_exact_two_sided;
    use crate::metrics::Outcome;
    use crate::significance::{PairedRecord, RequiredSampleSize, compare_with_baseline};

    const EPSILON: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn p(value: f64) -> PValue {
        PValue::new(value).expect("test helper requires a valid p value")
    }

    fn family(n: usize) -> FamilySize {
        FamilySize::new(n).expect("test helper requires a non-zero family size")
    }

    fn req(n: u64) -> RequiredSampleSize {
        RequiredSampleSize::new(n).expect("test helper requires a non-zero value")
    }

    /// PoC-10 selftest のベクタ: p=(0.01, 0.04, 0.03, 0.005)、m=4
    /// → 補正後 (0.03, 0.06, 0.06, 0.02)（入力順）。0.04 の候補が単調化で
    /// 0.06 になることを確認する。
    #[test]
    fn holm_adjust_poc10_selftest_vector() {
        let values = [p(0.01), p(0.04), p(0.03), p(0.005)];
        let adjusted = holm_adjust(&values, family(4)).unwrap();
        let expected = [0.03, 0.06, 0.06, 0.02];
        for (actual, expected) in adjusted.iter().zip(expected.iter()) {
            assert!(approx_eq(actual.value(), *expected));
        }
    }

    /// m=1・1 件では値が変わらない。
    #[test]
    fn holm_adjust_single_candidate_is_unchanged() {
        let values = [p(0.02)];
        let adjusted = holm_adjust(&values, family(1)).unwrap();
        assert!(approx_eq(adjusted[0].value(), 0.02));
    }

    /// 1 での打ち切り: p=(0.5, 0.6)、m=2 → (1.0, 1.0)。
    #[test]
    fn holm_adjust_caps_at_one() {
        let values = [p(0.5), p(0.6)];
        let adjusted = holm_adjust(&values, family(2)).unwrap();
        assert_eq!(adjusted[0].value(), 1.0);
        assert_eq!(adjusted[1].value(), 1.0);
    }

    /// 同値: p=(0.02, 0.02)、m=2 → (0.04, 0.04)。入力順を入れ替えても同じ。
    #[test]
    fn holm_adjust_tied_values_agree_regardless_of_order() {
        let values = [p(0.02), p(0.02)];
        let adjusted = holm_adjust(&values, family(2)).unwrap();
        assert!(approx_eq(adjusted[0].value(), 0.04));
        assert!(approx_eq(adjusted[1].value(), 0.04));

        // 入力順を替えても（同値なので実質同じ配列だが、決定性の確認として
        // 明示的に別スライスで再検証する）。
        let reordered = [p(0.02), p(0.02)];
        let adjusted_reordered = holm_adjust(&reordered, family(2)).unwrap();
        assert!(approx_eq(
            adjusted_reordered[0].value(),
            adjusted[0].value()
        ));
        assert!(approx_eq(
            adjusted_reordered[1].value(),
            adjusted[1].value()
        ));
    }

    /// 脱落（m > 件数）: p=(0.01, 0.02)、m=4 → (0.04, 0.06)。
    #[test]
    fn holm_adjust_with_dropped_candidates() {
        let values = [p(0.01), p(0.02)];
        let adjusted = holm_adjust(&values, family(4)).unwrap();
        assert!(approx_eq(adjusted[0].value(), 0.04));
        assert!(approx_eq(adjusted[1].value(), 0.06));
    }

    /// 入力順の保持: 降順に並べた入力でも、戻り値は入力の添字に対応する。
    #[test]
    fn holm_adjust_preserves_input_order_for_descending_input() {
        let values = [p(0.09), p(0.03), p(0.01)];
        let adjusted = holm_adjust(&values, family(3)).unwrap();
        // 昇順ソート後: 0.01(rank0,x3=0.03) 0.03(rank1,x2=0.06) 0.09(rank2,x1=0.09)
        // 単調化: running=0.03 → max(0.03,0.06)=0.06 → max(0.06,0.09)=0.09
        // 入力順に戻すと: idx0(0.09)->0.09, idx1(0.03)->0.06, idx2(0.01)->0.03
        assert!(approx_eq(adjusted[0].value(), 0.09));
        assert!(approx_eq(adjusted[1].value(), 0.06));
        assert!(approx_eq(adjusted[2].value(), 0.03));
    }

    /// 単調性: すべての i で補正後 >= 補正前。
    #[test]
    fn holm_adjust_is_never_smaller_than_raw() {
        let raw = [p(0.001), p(0.5), p(0.02), p(0.3)];
        let adjusted = holm_adjust(&raw, family(4)).unwrap();
        for (r, a) in raw.iter().zip(adjusted.iter()) {
            assert!(a.value() >= r.value());
        }
    }

    #[test]
    fn family_size_rejects_zero() {
        assert_eq!(FamilySize::new(0), None);
    }

    #[test]
    fn holm_adjust_empty_input_is_error() {
        let err = holm_adjust(&[], family(1)).unwrap_err();
        assert_eq!(err, HolmError::EmptyPValues);
    }

    #[test]
    fn holm_adjust_family_size_too_small_is_error() {
        let values = [p(0.01), p(0.02), p(0.03)];
        let err = holm_adjust(&values, family(2)).unwrap_err();
        assert_eq!(
            err,
            HolmError::FamilySizeTooSmall {
                family_size: 2,
                n_tests: 3,
            }
        );
    }

    #[test]
    fn holm_adjust_family_size_too_large_is_error() {
        let values = [p(0.5)];
        let err = holm_adjust(&values, family(MAX_FAMILY_SIZE + 1)).unwrap_err();
        assert_eq!(
            err,
            HolmError::FamilySizeTooLarge {
                family_size: MAX_FAMILY_SIZE + 1,
                limit: MAX_FAMILY_SIZE,
            }
        );
    }

    /// `compare_candidates_with_holm`: 補正後の p が `judge` へ渡ることを
    /// 確認する（PoC-10 seed0 の C1〜C4 vs majority、m=4）。
    #[test]
    fn compare_candidates_with_holm_applies_adjusted_p_to_judge() {
        let labels = ["A", "B"];
        let make_comparison = |b: u64, c: u64| -> BaselineComparison {
            let cand = Outcome::Label("B".to_string());
            let base = Outcome::Label("A".to_string());
            let mut records = Vec::new();
            for _ in 0..b {
                records.push(PairedRecord {
                    gold: "B",
                    candidate: &cand,
                    baseline: &base,
                });
            }
            for _ in 0..c {
                records.push(PairedRecord {
                    gold: "A",
                    candidate: &cand,
                    baseline: &base,
                });
            }
            compare_with_baseline(&labels, &records, req(1)).unwrap()
        };

        let comparisons = [
            make_comparison(130, 29),
            make_comparison(123, 26),
            make_comparison(161, 52),
            make_comparison(108, 36),
        ];

        let results = compare_candidates_with_holm(&comparisons, family(4)).unwrap();
        assert_eq!(results.len(), 4);
        for result in &results {
            assert_eq!(result.verdict(), BaselineVerdict::SignificantlyBetter);
            // 補正後は補正前以上。
            assert!(
                result.adjusted_p().value() >= result.comparison().test().p_two_sided().value()
            );
        }
    }

    /// 評価契約: 件数不足による `Undeterminable` は補正後も維持される
    /// （補正後の p がどれだけ小さくても判定を上書きしない）。
    #[test]
    fn compare_candidates_with_holm_keeps_undeterminable() {
        let labels = ["A", "B"];
        let cand = Outcome::Label("B".to_string());
        let base = Outcome::Label("A".to_string());
        // (b, c) = (6, 0) は単体なら p=0.03125 で SignificantlyBetter になる
        // 値だが、必要件数 7 に対して評価件数 6 は不足しているため
        // Undeterminable のままであるべき。
        let records: Vec<PairedRecord<'_>> = (0..6)
            .map(|_| PairedRecord {
                gold: "B",
                candidate: &cand,
                baseline: &base,
            })
            .collect();
        let insufficient = compare_with_baseline(&labels, &records, req(7)).unwrap();

        // もう 1 候補は十分な件数を持つ通常の候補として混在させる。
        let determinable_records: Vec<PairedRecord<'_>> = (0..221)
            .map(|_| PairedRecord {
                gold: "B",
                candidate: &cand,
                baseline: &base,
            })
            .collect();
        let determinable = compare_with_baseline(&labels, &determinable_records, req(221)).unwrap();

        let comparisons = [insufficient, determinable];
        let results = compare_candidates_with_holm(&comparisons, family(2)).unwrap();

        assert!(matches!(
            results[0].verdict(),
            BaselineVerdict::Undeterminable(_)
        ));
        assert_eq!(results[1].verdict(), BaselineVerdict::SignificantlyBetter);
    }

    /// 空スライスは `EmptyPValues`（パニックしない）。
    #[test]
    fn compare_candidates_with_holm_empty_input_is_error() {
        let err = compare_candidates_with_holm(&[], family(1)).unwrap_err();
        assert_eq!(err, HolmError::EmptyPValues);
    }

    /// b <= c の候補は、補正後の p がどれだけ小さくても
    /// `NotSignificantlyBetter`（向きが逆転しないことの確認）。
    #[test]
    fn holm_does_not_flip_direction_when_baseline_wins() {
        let result = mcnemar_exact_two_sided(3, 12).unwrap();
        let raw = [result.p_two_sided()];
        let adjusted = holm_adjust(&raw, family(1)).unwrap();
        let verdict = significance::judge(3, 12, adjusted[0], 221, req(221));
        assert_eq!(verdict, BaselineVerdict::NotSignificantlyBetter);
    }
}
