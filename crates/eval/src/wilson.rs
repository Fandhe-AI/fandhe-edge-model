//! Wilson スコア法による 95% 信頼区間の算出（REQ-24・TASK-24.1-2・issue #60）。
//!
//! 再現性の判定（REQ-26: 3 seed 以上の Wilson 95% 信頼区間の重なりで再現性を
//! 示す）・評価レポート（REQ-29）から再利用される想定で、[`crate::metrics`]
//! モジュールの [`crate::metrics::Ratio`] とは独立に、`(correct, n)` の組から
//! 直接計算できる関数として提供する。
//!
//! 式の出典は PoC-9 manifest（`wilson_confidence_interval.formula`。
//! z=1.96 で両側 95%）で、center／margin 形式の演算順を Rust 側でも踏襲する:
//!
//! ```text
//! p      = correct / n
//! denom  = 1 + z^2 / n
//! center = (p + z^2 / (2n)) / denom
//! margin = (z / denom) * sqrt(p(1-p)/n + z^2/(4n^2))
//! lo, hi = center ∓ margin
//! ```
//!
//! **sklearn 照合の対象外**: scikit-learn は Wilson 区間を提供しないため、
//! `tests/sklearn_check.rs` の sklearn 照合 8 ケースには Wilson の値を含めない。
//! Wilson の検証は (a) PoC-9 manifest 追補 A-8 の手計算値と (b) PoC-9 評価器
//! （Python 独立実装）が出力した `ci95` 値との照合で行う（本モジュール末尾の
//! 単体テスト、および `tests/sklearn_check.rs` 内の別建て結合テスト）。
//!
//! p=0 のとき上記の式は `center - margin` が `-2.8e-17` のような極小の負値に
//! なりうる（PoC-9 でも同じ現象を確認済み）。評価契約の許容差 1e-9 を超える
//! 負の区間を外部へ見せないよう、`[0.0, 1.0]` へクランプする（PoC の生値との
//! 差は 1e-16 未満で、クランプによって信頼区間の意味は変わらない）。

use crate::metrics::Ratio;

/// 95% 信頼区間で用いる標準正規分布の臨界値（両側）。
pub const WILSON_Z_95: f64 = 1.96;

/// Wilson スコア区間。壊れた値（`lo > hi`・範囲外・NaN 混入）を外部から
/// 構築できないよう、フィールドを非公開にしアクセサのみ公開する
/// （`.claude/rules/coding-rust.md`「判定結果...は壊れた値を表現できない型にする」）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WilsonInterval {
    lo: f64,
    hi: f64,
    z: f64,
}

impl WilsonInterval {
    /// 区間の下限（`[0.0, 1.0]` にクランプ済み）。
    pub fn lo(&self) -> f64 {
        self.lo
    }

    /// 区間の上限（`[0.0, 1.0]` にクランプ済み）。
    pub fn hi(&self) -> f64 {
        self.hi
    }

    /// 使用した z 値（[`wilson_ci95`] 経由では常に [`WILSON_Z_95`]）。
    pub fn z(&self) -> f64 {
        self.z
    }
}

/// Wilson スコア区間を計算する。
///
/// `n == 0`（分母 0。評価契約: 分母 0 の指標は null）・`correct > n`（不正な
/// 入力。`p(1-p)` が負になり `sqrt` が NaN になるのを fail-closed で防ぐ）・
/// `z` が有限な正の数でない場合は `None` を返し、panic・NaN・負の区間を
/// 外部へ見せない。
///
/// `z` 自体が有限でも `z * z` を経由する各中間値（`z_sq`・`denom`・`center`・
/// `margin`）はオーバーフローで無限大・NaN になりうる（例: `z = 1e308` は
/// `z.is_finite()` を通過するが `z * z` が `f64::INFINITY` になる）。最終的な
/// `lo`・`hi` を含め、いずれかが有限でない、または `lo > hi` になった場合も
/// `None` を返し、`WilsonInterval` が「壊れた値を表現できない型」という不変
/// 条件を保つ（issue #60 PR #211 codex レビュー指摘。
/// `.claude/rules/coding-rust.md`「判定結果は壊れた値を表現できない型にする」
/// 「外部入力の経路では明示的に処理する」）。
///
/// `u64 → f64` の変換は `n` が `2^53` を超えると丸め誤差が生じうるが、
/// レコード件数の上限検証は呼び出し側（データ検査層・REQ-39）の責務であり、
/// 実用上の評価件数はこの範囲を大きく下回る。
pub fn wilson_ci(correct: u64, n: u64, z: f64) -> Option<WilsonInterval> {
    if n == 0 || correct > n {
        return None;
    }
    if !z.is_finite() || z <= 0.0 {
        return None;
    }

    let n_f = n as f64;
    let correct_f = correct as f64;
    let p = correct_f / n_f;
    let z_sq = z * z;
    if !z_sq.is_finite() {
        return None;
    }
    let denom = 1.0 + z_sq / n_f;
    let center = (p + z_sq / (2.0 * n_f)) / denom;
    let margin = (z / denom) * (p * (1.0 - p) / n_f + z_sq / (4.0 * n_f * n_f)).sqrt();
    if !denom.is_finite() || !center.is_finite() || !margin.is_finite() {
        return None;
    }

    let lo = (center - margin).max(0.0);
    let hi = (center + margin).min(1.0);
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return None;
    }

    Some(WilsonInterval { lo, hi, z })
}

/// [`wilson_ci`] を `z = `[`WILSON_Z_95`]` で呼ぶ薄いラッパー。
pub fn wilson_ci95(correct: u64, n: u64) -> Option<WilsonInterval> {
    wilson_ci(correct, n, WILSON_Z_95)
}

/// [`Ratio`] から 95% Wilson 信頼区間を求める（[`metrics::Ratio::ci95`] の実体）。
///
/// `Ratio` の不変条件（`0 < denominator`・`numerator <= denominator`。
/// [`metrics::Ratio::new`]）の下では常に `Some` を返すが、`Ratio` の不変条件
/// が将来変わった場合でも panic しないよう `Option` のまま返す。
pub(crate) fn ratio_ci95(ratio: &Ratio) -> Option<WilsonInterval> {
    wilson_ci95(ratio.numerator(), ratio.denominator())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;
    const EPSILON_HAND_CALC: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    /// REQ-24・TASK-24.1-2: PoC-9 manifest 追補 A-8 の手計算値（53/61）と一致する。
    /// 証拠の種別: 手計算（6 桁丸め。許容差 1e-6）。
    #[test]
    fn matches_hand_calculated_value_53_of_61() {
        let interval = wilson_ci95(53, 61).expect("n > 0");
        assert!(approx_eq(interval.lo(), 0.761979, EPSILON_HAND_CALC));
        assert!(approx_eq(interval.hi(), 0.932020, EPSILON_HAND_CALC));
    }

    /// REQ-24・TASK-24.1-2: PoC-9 manifest 追補 A-8 の手計算値（57/61）と一致する。
    #[test]
    fn matches_hand_calculated_value_57_of_61() {
        let interval = wilson_ci95(57, 61).expect("n > 0");
        assert!(approx_eq(interval.lo(), 0.843170, EPSILON_HAND_CALC));
        assert!(approx_eq(interval.hi(), 0.974207, EPSILON_HAND_CALC));
    }

    /// REQ-24・TASK-24.1-2: PoC-9 の Python 実装（独立実装。sklearn ではない）
    /// が出力した全桁値と 1e-9 で一致する（53/61）。
    #[test]
    fn matches_poc9_python_full_precision_53_of_61() {
        let interval = wilson_ci95(53, 61).expect("n > 0");
        assert!(approx_eq(interval.lo(), 0.761978875735935, EPSILON));
        assert!(approx_eq(interval.hi(), 0.932020038541319, EPSILON));
    }

    /// REQ-24・TASK-24.1-2: PoC-9 の Python 実装が出力した全桁値と一致する（57/61）。
    #[test]
    fn matches_poc9_python_full_precision_57_of_61() {
        let interval = wilson_ci95(57, 61).expect("n > 0");
        assert!(approx_eq(interval.lo(), 0.8431697859953278, EPSILON));
        assert!(approx_eq(interval.hi(), 0.9742067130423271, EPSILON));
    }

    /// REQ-24・TASK-24.1-2: 分母 0 は評価契約（分母 0 は null）に従い `None`。
    #[test]
    fn zero_denominator_is_none() {
        assert_eq!(wilson_ci95(0, 0), None);
        assert_eq!(wilson_ci95(5, 0), None);
    }

    /// REQ-24・TASK-24.1-2: `correct > n` は不正な入力として `None`
    /// （`p(1-p)` が負になり `sqrt` が NaN になるのを fail-closed で防ぐ）。
    #[test]
    fn correct_greater_than_n_is_none() {
        assert_eq!(wilson_ci95(11, 10), None);
    }

    /// REQ-24・TASK-24.1-2: `z` が 0・負・NaN・無限大のいずれかなら `None`。
    #[test]
    fn invalid_z_is_none() {
        assert_eq!(wilson_ci(5, 10, 0.0), None);
        assert_eq!(wilson_ci(5, 10, -1.96), None);
        assert_eq!(wilson_ci(5, 10, f64::NAN), None);
        assert_eq!(wilson_ci(5, 10, f64::INFINITY), None);
    }

    /// REQ-24・TASK-24.1-2: `z` は `is_finite()` を通過する有限値でも、
    /// `z * z` がオーバーフローして無限大になりうる（例: `z = 1e308`）。
    /// その場合に `center`・`margin` が NaN になった壊れた `WilsonInterval` を
    /// 外部へ返さず `None` になることを確認する（issue #60 PR #211 codex 指摘）。
    #[test]
    fn huge_finite_z_overflowing_square_is_none() {
        assert_eq!(wilson_ci(5, 10, 1e308), None);
        assert_eq!(wilson_ci(5, 10, f64::MAX), None);
    }

    /// REQ-24・TASK-24.1-2: `p=0` のときクランプ前の生値は負の極小値
    /// （PoC-9 でも確認済み）になるが、`lo` は `0.0` に完全一致する。
    #[test]
    fn zero_correct_clamps_lo_to_zero() {
        let interval = wilson_ci95(0, 10).expect("n > 0");
        assert_eq!(interval.lo(), 0.0);
        assert!(approx_eq(interval.hi(), 0.2775401687666166, EPSILON));
    }

    /// REQ-24・TASK-24.1-2: `p=1` のとき `hi` は `1.0` に完全一致する。
    #[test]
    fn all_correct_clamps_hi_to_one() {
        let interval = wilson_ci95(10, 10).expect("n > 0");
        assert!(approx_eq(interval.lo(), 0.7224598312333834, EPSILON));
        assert_eq!(interval.hi(), 1.0);
    }

    /// REQ-24・TASK-24.1-2: `n=1, correct=0` の境界値。
    #[test]
    fn single_record_zero_correct() {
        let interval = wilson_ci95(0, 1).expect("n > 0");
        assert_eq!(interval.lo(), 0.0);
        assert!(approx_eq(interval.hi(), 0.7934567085261071, EPSILON));
    }

    /// REQ-24・TASK-24.1-2: `n=1, correct=1` の境界値。
    #[test]
    fn single_record_all_correct() {
        let interval = wilson_ci95(1, 1).expect("n > 0");
        assert!(approx_eq(interval.lo(), 0.2065432914738929, EPSILON));
        assert_eq!(interval.hi(), 1.0);
    }

    /// REQ-24・TASK-24.1-2: `wilson_ci95` は常に `z = WILSON_Z_95`（1.96）を使う。
    #[test]
    fn wilson_ci95_uses_z_1_96() {
        let interval = wilson_ci95(5, 10).expect("n > 0");
        assert_eq!(interval.z(), WILSON_Z_95);
        assert_eq!(WILSON_Z_95, 1.96);
    }
}
