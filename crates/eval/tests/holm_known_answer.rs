//! 既知値による結合テスト（REQ-25 境界値・TASK-25.3・issue #67）。
//!
//! 参照値は独立実装（Python の `fractions.Fraction`・`math.comb` による
//! 有理数の厳密計算＋有理数のままの Holm 補正）で計画時に算出したもの。
//! `docs/spec` は読み込まず、期待値はこのファイルへ直接書く
//! （`.claude/rules/spec-reference.md` のビルド独立方針）。生成元スクリプトと
//! 再現用のゴールデンベクタは `fixtures/holm/`（`generate_known_values.py`・
//! `known_values.json`・`PROVENANCE.md`）にあり、本ファイルの期待値と全件
//! 一致することを確認済み。このテスト自体は `fixtures/` を実行時に読み込まない
//! （期待値は直書きのまま。ビルド・テストの独立性を保つ）。
//!
//! `b`・`c` の由来（PoC-10・PoC-25 実測値の転記）は `fixtures/holm/
//! PROVENANCE.md` を参照。証拠の種別: テストハーネス（b・c は実測値の転記、
//! p・補正後 p は厳密計算による独立照合）。
//!
//! # 許容差の方針
//!
//! `crates/eval/tests/mcnemar_known_answer.rs` と同じく、絶対誤差 1e-9 **と**
//! 相対誤差 1e-9 の両方を要求する（`.claude/rules/evaluation-contract.md` の
//! 許容差 1e-9 を厳しくする方向の運用であり、緩めるものではない）。

use fandhe_edge_eval::baseline::fit_majority;
use fandhe_edge_eval::holm::{FamilySize, HolmError, compare_candidates_with_holm};
use fandhe_edge_eval::metrics::Outcome;
use fandhe_edge_eval::significance::{
    BaselineComparison, BaselineVerdict, PairedRecord, RequiredSampleSize, compare_with_baseline,
};

const ABS_EPSILON: f64 = 1e-9;
const REL_EPSILON: f64 = 1e-9;

/// 絶対誤差・相対誤差の両方を満たす場合にのみ一致とみなす
/// （`mcnemar_known_answer.rs`・`baseline_significance.rs` と同じ方針）。
fn approx_eq(actual: f64, expected: f64) -> bool {
    let abs_diff = (actual - expected).abs();
    if expected == 0.0 {
        return actual == 0.0;
    }
    let rel_diff = abs_diff / expected.abs();
    abs_diff < ABS_EPSILON && rel_diff < REL_EPSILON
}

const LABELS: [&str; 2] = ["A", "B"];

/// `(b, c)` から [`BaselineComparison`] を決定的に組み立てる。
///
/// gold="B"・候補が正解／下限基準が不正解の行を `b` 件、gold="A"・下限基準が
/// 正解／候補が不正解の行を `c` 件だけ作る単純化された合成データ
/// （`both_correct`・`both_wrong` は 0）。`required` は事前登録の必要件数
/// （[`RequiredSampleSize`]）。
fn comparison_from_bc(b: u64, c: u64, required: RequiredSampleSize) -> BaselineComparison {
    let cand_wins = Outcome::Label("B".to_string());
    let base_wins = Outcome::Label("A".to_string());

    let mut records = Vec::new();
    for _ in 0..b {
        records.push(PairedRecord {
            gold: "B",
            candidate: &cand_wins,
            baseline: &base_wins,
        });
    }
    for _ in 0..c {
        records.push(PairedRecord {
            gold: "A",
            candidate: &cand_wins,
            baseline: &base_wins,
        });
    }

    compare_with_baseline(&LABELS, &records, required).unwrap()
}

fn req(n: u64) -> RequiredSampleSize {
    RequiredSampleSize::new(n).expect("test helper requires a non-zero value")
}

fn family(n: usize) -> FamilySize {
    FamilySize::new(n).expect("test helper requires a non-zero family size")
}

/// PoC-10 `eval_incat` seed0（m=4）。C1〜C4 はすべて `SignificantlyBetter`。
///
/// 既知値: `fixtures/holm/known_values.json`
/// `families[0]`（`poc10_eval_incat_seed0`）。
#[test]
fn poc10_eval_incat_seed0_all_significantly_better() {
    let comparisons = [
        comparison_from_bc(130, 29, req(1)),
        comparison_from_bc(123, 26, req(1)),
        comparison_from_bc(161, 52, req(1)),
        comparison_from_bc(108, 36, req(1)),
    ];
    let results = compare_candidates_with_holm(&comparisons, family(4)).unwrap();

    let expected_adjusted = [
        7.222356750211203e-16,
        8.246207626486645e-16,
        7.329802055010862e-14,
        1.4957587495764933e-9,
    ];
    for (result, expected) in results.iter().zip(expected_adjusted.iter()) {
        assert!(
            approx_eq(result.adjusted_p().value(), *expected),
            "adjusted p mismatch: actual={}, expected={}",
            result.adjusted_p().value(),
            expected
        );
        assert_eq!(result.verdict(), BaselineVerdict::SignificantlyBetter);
    }
}

/// PoC-10 `eval_incat` seed1（m=4）。
///
/// 既知値: `fixtures/holm/known_values.json`
/// `families[1]`（`poc10_eval_incat_seed1`）。
#[test]
fn poc10_eval_incat_seed1_all_significantly_better() {
    let comparisons = [
        comparison_from_bc(130, 29, req(1)),
        comparison_from_bc(123, 26, req(1)),
        comparison_from_bc(143, 43, req(1)),
        comparison_from_bc(125, 58, req(1)),
    ];
    let results = compare_candidates_with_holm(&comparisons, family(4)).unwrap();

    let expected_adjusted = [
        7.222356750211203e-16,
        8.246207626486645e-16,
        1.904732396379798e-13,
        8.13607591316581e-7,
    ];
    for (result, expected) in results.iter().zip(expected_adjusted.iter()) {
        assert!(approx_eq(result.adjusted_p().value(), *expected));
        assert_eq!(result.verdict(), BaselineVerdict::SignificantlyBetter);
    }
}

/// PoC-10 `eval_incat` seed2（m=4）。
///
/// 既知値: `fixtures/holm/known_values.json`
/// `families[2]`（`poc10_eval_incat_seed2`）。
#[test]
fn poc10_eval_incat_seed2_all_significantly_better() {
    let comparisons = [
        comparison_from_bc(130, 29, req(1)),
        comparison_from_bc(123, 26, req(1)),
        comparison_from_bc(144, 40, req(1)),
        comparison_from_bc(134, 58, req(1)),
    ];
    let results = compare_candidates_with_holm(&comparisons, family(4)).unwrap();

    let expected_adjusted = [
        7.222356750211203e-16,
        8.246207626486645e-16,
        1.1036415394112638e-14,
        4.179048819596723e-8,
    ];
    for (result, expected) in results.iter().zip(expected_adjusted.iter()) {
        assert!(approx_eq(result.adjusted_p().value(), *expected));
        assert_eq!(result.verdict(), BaselineVerdict::SignificantlyBetter);
    }
}

/// PoC-10 `ref_test` seed0（m=4）。C1 は `b < c`（下限基準が優勢）で向きの
/// 確認、C4 は補正後 p が α（0.05）以上になる境界の確認。C2・C3 は
/// `SignificantlyBetter`。
///
/// 既知値: `fixtures/holm/known_values.json`
/// `families[3]`（`poc10_ref_test_seed0`）。
#[test]
fn poc10_ref_test_seed0_mixed_verdicts() {
    let comparisons = [
        comparison_from_bc(8, 79, req(1)),   // C1: b < c
        comparison_from_bc(260, 82, req(1)), // C2
        comparison_from_bc(279, 97, req(1)), // C3
        comparison_from_bc(97, 78, req(1)),  // C4: 補正後 p >= alpha
    ];
    let results = compare_candidates_with_holm(&comparisons, family(4)).unwrap();

    let expected_adjusted = [
        1.6755882947460108e-15,
        4.240111977367504e-22,
        4.74291697282034e-21,
        0.1734437279739842,
    ];
    for (result, expected) in results.iter().zip(expected_adjusted.iter()) {
        assert!(approx_eq(result.adjusted_p().value(), *expected));
    }

    // C1: b < c なので、補正後 p が極めて小さくても NotSignificantlyBetter。
    assert_eq!(
        results[0].verdict(),
        BaselineVerdict::NotSignificantlyBetter
    );
    assert_eq!(results[1].verdict(), BaselineVerdict::SignificantlyBetter);
    assert_eq!(results[2].verdict(), BaselineVerdict::SignificantlyBetter);
    // C4: 補正後 p (0.1734...) >= alpha (0.05)。
    assert_eq!(
        results[3].verdict(),
        BaselineVerdict::NotSignificantlyBetter
    );
}

/// PoC-25 `fresh-eval-recheck` primary seed0（m=4）。C1 の補正前 p に family
/// size を掛けた値が C4 の補正後値（`4 * raw_p(C4)`）をわずかに上回るため、
/// 累積最大により C1 の補正後値は C4 の補正後値に巻き取られて一致する
/// （相対差の余裕は `fixtures/holm/PROVENANCE.md`
/// 「`poc25_primary_seed0` の累積最大に関する余裕の確認」参照。実測
/// 約 1.2% で、libm 近似誤差〔相対 1e-14 程度〕を十分上回る）。
///
/// 既知値: `fixtures/holm/known_values.json`
/// `families[4]`（`poc25_primary_seed0`）。
#[test]
fn poc25_primary_seed0_c1_is_capped_by_c4() {
    let comparisons = [
        comparison_from_bc(156, 36, req(1)), // C1
        comparison_from_bc(138, 35, req(1)), // C2
        comparison_from_bc(177, 89, req(1)), // C3
        comparison_from_bc(79, 4, req(1)),   // C4
    ];
    let results = compare_candidates_with_holm(&comparisons, family(4)).unwrap();

    let expected_adjusted = [
        1.5989301978976519e-18, // C1（C4 の補正後値に巻き取られる）
        2.2978495329005742e-15,
        7.363426340076371e-8,
        1.5989301978976519e-18, // C4
    ];
    for (result, expected) in results.iter().zip(expected_adjusted.iter()) {
        assert!(approx_eq(result.adjusted_p().value(), *expected));
        assert_eq!(result.verdict(), BaselineVerdict::SignificantlyBetter);
    }

    // C1 と C4 の補正後値がちょうど一致すること（累積最大の直接確認）。
    assert_eq!(
        results[0].adjusted_p().value(),
        results[3].adjusted_p().value()
    );
}

/// α 境界（合成）: `(13, 4)` を単独（m=1）で補正すると
/// `p = 0.049041748046875 < 0.05` で `SignificantlyBetter`。
///
/// 既知値: `fixtures/holm/known_values.json`
/// `families[5]`（`alpha_boundary_solo`）。
#[test]
fn alpha_boundary_solo_is_significantly_better() {
    let comparisons = [comparison_from_bc(13, 4, req(1))];
    let results = compare_candidates_with_holm(&comparisons, family(1)).unwrap();
    assert!(approx_eq(
        results[0].adjusted_p().value(),
        0.049041748046875
    ));
    assert_eq!(results[0].verdict(), BaselineVerdict::SignificantlyBetter);
}

/// α 境界（合成）: `(13, 4)` と `(22, 10)`（p ≈ 0.0501）を m=2 の族にすると、
/// `(13, 4)` が最小のため ×2 され、累積最大により両者とも `≈0.098` に
/// なって `NotSignificantlyBetter` に反転する（補正で判定が変わることの
/// 確認）。
///
/// 既知値: `fixtures/holm/known_values.json`
/// `families[6]`（`alpha_boundary_pair_flips`）。
#[test]
fn alpha_boundary_pair_flips_verdict() {
    let comparisons = [
        comparison_from_bc(13, 4, req(1)),
        comparison_from_bc(22, 10, req(1)),
    ];
    let results = compare_candidates_with_holm(&comparisons, family(2)).unwrap();
    assert!(approx_eq(results[0].adjusted_p().value(), 0.09808349609375));
    assert!(approx_eq(results[1].adjusted_p().value(), 0.09808349609375));
    assert_eq!(
        results[0].verdict(),
        BaselineVerdict::NotSignificantlyBetter
    );
    assert_eq!(
        results[1].verdict(),
        BaselineVerdict::NotSignificantlyBetter
    );
}

/// α 境界（合成）: `(13, 4)` を、十分小さい p を持つ `(130, 29)` と同じ
/// m=2 の族にした場合、`(13, 4)` は最大順位（乗数 ×1）のままなので
/// `0.049041748046875` を保ち `SignificantlyBetter` のまま。
///
/// 既知値: `fixtures/holm/known_values.json`
/// `families[7]`（`alpha_boundary_pair_holds`）。
#[test]
fn alpha_boundary_pair_holds_verdict() {
    let comparisons = [
        comparison_from_bc(13, 4, req(1)),
        comparison_from_bc(130, 29, req(1)),
    ];
    let results = compare_candidates_with_holm(&comparisons, family(2)).unwrap();
    assert!(approx_eq(
        results[0].adjusted_p().value(),
        0.049041748046875
    ));
    assert_eq!(results[0].verdict(), BaselineVerdict::SignificantlyBetter);
}

/// 判定不能の維持（REQ-25・TASK-25.2）: `n < required` の
/// `BaselineComparison` は、Holm 補正後の p がどれだけ小さくても
/// `Undeterminable` のまま。もう 1 候補（十分な件数）は
/// `SignificantlyBetter` になる。
#[test]
fn undeterminable_is_kept_after_holm_correction() {
    // (b, c) = (6, 0) は単体なら p=0.03125 で SignificantlyBetter になる値だが、
    // 必要件数 7 に対して評価件数 6 は不足している。
    let insufficient = comparison_from_bc(6, 0, req(7));
    let determinable = comparison_from_bc(130, 29, req(1));

    let comparisons = [insufficient, determinable];
    let results = compare_candidates_with_holm(&comparisons, family(2)).unwrap();

    assert!(matches!(
        results[0].verdict(),
        BaselineVerdict::Undeterminable(_)
    ));
    assert_eq!(results[1].verdict(), BaselineVerdict::SignificantlyBetter);
}

/// 脱落（m > 渡した候補数）: PoC-10 seed0 の 4 候補のうち 3 候補
/// （C1〜C3）だけを m=4 で渡す。C4（最大の raw p）を除いても、C1〜C3 の
/// 順位（ランク）と乗数は変わらないため、補正後の値は 4 候補すべてを
/// 渡したときと厳密に一致する（保守側の扱い。族サイズを固定したまま
/// 渡す候補数を減らすことで、脱落分の乗数を空けたまま補正する）。
///
/// 既知値: `fixtures/holm/known_values.json`
/// `families[8]`（`dropout_3_of_4`）。
#[test]
fn dropout_keeps_family_size_and_matches_full_family_values() {
    let comparisons = [
        comparison_from_bc(130, 29, req(1)),
        comparison_from_bc(123, 26, req(1)),
        comparison_from_bc(161, 52, req(1)),
    ];
    let results = compare_candidates_with_holm(&comparisons, family(4)).unwrap();

    let expected_adjusted = [
        7.222356750211203e-16,
        8.246207626486645e-16,
        7.329802055010862e-14,
    ];
    for (result, expected) in results.iter().zip(expected_adjusted.iter()) {
        assert!(approx_eq(result.adjusted_p().value(), *expected));
        assert_eq!(result.verdict(), BaselineVerdict::SignificantlyBetter);
    }
}

/// 空スライスは `HolmError::EmptyPValues`（パニックしない）。
#[test]
fn compare_candidates_with_holm_empty_is_error() {
    let err = compare_candidates_with_holm(&[], family(1)).unwrap_err();
    assert_eq!(err, HolmError::EmptyPValues);
}

/// 族サイズが渡した候補数より小さい場合は `FamilySizeTooSmall`
/// （`m` を黙って件数に合わせない）。
#[test]
fn compare_candidates_with_holm_family_too_small_is_error() {
    let comparisons = [
        comparison_from_bc(130, 29, req(1)),
        comparison_from_bc(123, 26, req(1)),
    ];
    let err = compare_candidates_with_holm(&comparisons, family(1)).unwrap_err();
    assert_eq!(
        err,
        HolmError::FamilySizeTooSmall {
            family_size: 1,
            n_tests: 2,
        }
    );
}

/// [`fit_majority`] を使い、下限基準予測の生成から一気通貫で Holm 補正まで
/// 通す統合的なスモークテスト（評価契約: 入力を書き換えないことの確認も兼ねる）。
#[test]
fn end_to_end_with_fit_majority_baseline() {
    let train_gold = ["A", "A", "A", "B"];
    let majority = fit_majority(&LABELS, &train_gold).unwrap();
    assert_eq!(majority, "A");

    let cand_wins = Outcome::Label("B".to_string());
    let base_wins = Outcome::Label(majority.to_string());

    let mut records = Vec::new();
    for _ in 0..13 {
        records.push(PairedRecord {
            gold: "B",
            candidate: &cand_wins,
            baseline: &base_wins,
        });
    }
    for _ in 0..4 {
        records.push(PairedRecord {
            gold: "A",
            candidate: &cand_wins,
            baseline: &base_wins,
        });
    }
    let comparison = compare_with_baseline(&LABELS, &records, req(1)).unwrap();

    let results = compare_candidates_with_holm(&[comparison], family(1)).unwrap();
    assert_eq!(results.len(), 1);
    assert!(approx_eq(
        results[0].adjusted_p().value(),
        0.049041748046875
    ));
    assert_eq!(results[0].verdict(), BaselineVerdict::SignificantlyBetter);
}
