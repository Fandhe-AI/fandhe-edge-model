//! 既知値による結合テスト（REQ-25 正常系・TASK-25.1-1・issue #64）。
//!
//! 参照値は独立実装（Python の `fractions.Fraction` と `math.comb` による
//! 有理数の厳密計算）で計画時に算出したもの。`docs/spec` は読み込まず、
//! 期待値はこのファイルへ直接書く（`.claude/rules/spec-reference.md` の
//! ビルド独立方針）。
//!
//! 証拠の種別: テストハーネス（有理数の厳密計算による独立実装との照合）。
//!
//! # 許容差の方針
//!
//! `ln`・`exp` はプラットフォームの libm に依存するため `==` では比較しない。
//! 期待値が正のケースは、絶対誤差 1e-9 **と**相対誤差 1e-9 の両方を要求する
//! （どちらか一方だけでは、p 値が小さいケース〔1e-9 を大きく下回る値〕で
//! 絶対誤差判定が `0.0` を誤って許してしまい、意味のある照合にならない）。
//! これは `.claude/rules/evaluation-contract.md` の許容差 1e-9 を
//! **厳しくする方向**の運用であり、契約の許容差を緩めるものではない。
//! `f64` のアンダーフローにより真値が計算結果 `0.0` になるケース
//! （[`underflow_to_zero_is_not_an_error`]）は、`approx_eq` を使わず
//! `assert_eq!(..., 0.0)` で厳密に照合する。

use fandhe_edge_eval::mcnemar::mcnemar_exact_two_sided;

const ABS_EPSILON: f64 = 1e-9;
const REL_EPSILON: f64 = 1e-9;

/// 絶対誤差・相対誤差の両方を満たす場合にのみ一致とみなす。
///
/// 期待値が厳密に 0.0 の場合（アンダーフローの対称性チェック等）は
/// 相対誤差の分母が 0 になり定義できないため、絶対誤差のみで判定する
/// （`actual` も厳密に 0.0 であることを要求するので許容差を弱めない）。
fn approx_eq(actual: f64, expected: f64) -> bool {
    let abs_diff = (actual - expected).abs();
    if expected == 0.0 {
        return abs_diff < ABS_EPSILON;
    }
    let rel_diff = abs_diff / expected.abs();
    abs_diff < ABS_EPSILON && rel_diff < REL_EPSILON
}

/// 小さい既知値ケース（手計算・有理数の厳密計算で照合済み）。
#[test]
fn known_small_values() {
    let cases: &[(u64, u64, f64)] = &[
        (0, 0, 1.0),
        (1, 0, 1.0),
        (3, 1, 0.625),
        (6, 0, 0.03125),
        (10, 0, 0.001953125),
        (5, 5, 1.0),
        (30, 29, 1.0),
        (12, 3, 0.03515625),
        (9, 3, 0.14599609375),
    ];

    for &(b, c, expected) in cases {
        let result = mcnemar_exact_two_sided(b, c).unwrap();
        assert!(
            approx_eq(result.p_two_sided().value(), expected),
            "b={b}, c={c}: expected {expected}, got {}",
            result.p_two_sided().value()
        );
        assert_eq!(result.b(), b);
        assert_eq!(result.c(), c);
        assert_eq!(result.n_discordant(), b + c);
    }
}

/// 中〜大きい n の既知値ケース（有理数の厳密計算。相対誤差で照合）。
#[test]
fn known_large_values() {
    let cases: &[(u64, u64, f64)] = &[
        (25, 10, 0.01667384780012071),
        (60, 40, 0.05688793364098079),
        (400, 250, 4.368816826256423e-09),
        (5000, 4700, 0.0023966033625776252),
        (1000, 0, 1.8665272370064378e-301),
    ];

    for &(b, c, expected) in cases {
        let result = mcnemar_exact_two_sided(b, c).unwrap();
        assert!(
            approx_eq(result.p_two_sided().value(), expected),
            "b={b}, c={c}: expected {expected}, got {}",
            result.p_two_sided().value()
        );
    }
}

/// `f64` のアンダーフローにより、真値が正でも計算結果が `0.0` になる
/// ケース（真値は `2^-1099` で `f64` の最小正規化数を大きく下回る）。
/// これは仕様どおりの挙動であり、エラーにしないことを確認する。
#[test]
fn underflow_to_zero_is_not_an_error() {
    let result = mcnemar_exact_two_sided(1100, 0).unwrap();
    assert_eq!(result.p_two_sided().value(), 0.0);
}

/// PoC-10 の実データ（`logs/eval/summary.json` seed0・n=650・majority との比較）。
///
/// `summary.json` の値（lgamma による計算）とは末尾の桁がわずかに異なる
/// （相対誤差 1e-14 程度）。参照値はここでは有理数の厳密値を使い、
/// 相対誤差 1e-9 で照合する。
#[test]
fn poc10_seed0_majority_comparisons() {
    let cases: &[(&str, u64, u64, f64)] = &[
        ("C1", 130, 29, 1.8055891875528008e-16),
        ("C2", 123, 26, 2.748735875495548e-16),
        ("C3", 161, 52, 3.664901027505431e-14),
        ("C4", 108, 36, 1.4957587495764933e-09),
    ];

    for &(name, b, c, expected) in cases {
        let result = mcnemar_exact_two_sided(b, c).unwrap();
        assert!(
            approx_eq(result.p_two_sided().value(), expected),
            "{name}: b={b}, c={c}: expected {expected}, got {}",
            result.p_two_sided().value()
        );
    }
}

/// 対称性: `p(b, c) == p(c, b)`（許容差つき）。既知値ケース全件で確認する。
#[test]
fn symmetric_in_b_and_c_for_all_known_cases() {
    let pairs: &[(u64, u64)] = &[
        (0, 0),
        (1, 0),
        (3, 1),
        (6, 0),
        (10, 0),
        (5, 5),
        (30, 29),
        (12, 3),
        (9, 3),
        (25, 10),
        (60, 40),
        (400, 250),
        (5000, 4700),
        (1000, 0),
        (1100, 0),
        (130, 29),
        (123, 26),
        (161, 52),
        (108, 36),
    ];

    for &(b, c) in pairs {
        let forward = mcnemar_exact_two_sided(b, c).unwrap();
        let backward = mcnemar_exact_two_sided(c, b).unwrap();
        assert!(
            approx_eq(
                forward.p_two_sided().value(),
                backward.p_two_sided().value()
            ),
            "b={b}, c={c}: p(b,c)={}, p(c,b)={}",
            forward.p_two_sided().value(),
            backward.p_two_sided().value()
        );
    }
}
