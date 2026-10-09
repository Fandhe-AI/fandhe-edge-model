//! 待ち時間の p95 算出と参考値付きレポート（REQ-31・TASK-31.1-2・#128）。
//!
//! # 役割
//!
//! [`crate::latency::measure_latency`]（#127）が返す [`LatencySamples`] から p95 を算出し、
//! 既定の目安 250ms を**参考値**として明記したレポート [`LatencyReport`] を作る。
//! 呼び出し元: CLI の `package` 工程（`stages::package`。#338・#340）。合否とは分離する。
//!
//! # p95 の定義
//!
//! PoC-14 README の warm 計測・`analyze_stage2.py` と同じ方法（全計測値・外れ値を除かない・
//! `numpy.percentile` 既定の線形補間）。REQ-31 の根拠値はこの方法で出ている。PoC-16 の
//! `bench_p95`（`ceil(n*0.95)` 添字方式）は PoC-14 と値が一致しないため採用しない。
//! 計算は整数（u128 中間値）で厳密に行い、浮動小数を経由しない。
//!
//! # u64 ns への丸めは切り上げ
//!
//! 上限 L（整数 ns）に対し「厳密 p95 > L ⇔ 切り上げ p95 > L」が成り立つ。切り上げた値を
//! [`crate::package_outcome::LimitBreach::latency_if_exceeded`]（`>` で超過、`==` は合格）へ
//! 渡せば照合は厳密かつ fail-closed になる（floor・四捨五入だと厳密値 L+0.5 が合格になる）。
//!
//! # 合否との分離（範囲外）
//!
//! 250ms は目安であり合否条件ではない（TASK-31.1）。本モジュールは上限照合を呼ばず、
//! 合否フィールド・終了コードを持たない。利用者設定の上限との照合と `limit_exceeded` への
//! 結線は [`crate::latency_limit::check_latency_limit`]（TASK-31.2・#129）、上限ちょうどの境界判定は #130。JSON 直列化は CLI 側
//! （`stage_output` の `infer_p95`。#340）、配線は `stages::package` で済み（#338）。実機（静かな Mac）での実計測は人間の作業で、
//! 本モジュールのテストの証拠種別はテストハーネス（偽の時計・模擬バックエンド）のみ。

use crate::latency::LatencySamples;
use std::fmt;

/// 既定の目安 p95（ns。250ms。REQ-31。PC・推論のみ・最終判定の完了まで）。
///
/// **参考値**であり合否のしきい値ではない。合否は利用者が設定した上限で判定する。
pub const REFERENCE_P95_NS: u64 = 250_000_000;

/// p95 の算出方法の名前（レポートに出す。PoC-14 の `percentile_method` 相当）。
pub const P95_METHOD: &str =
    "linear interpolation (numpy.percentile default), all samples, no outlier exclusion";

/// 厳密な p95（ns）。厳密値 = `floor_ns` + `frac_hundredths` / 100。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct P95Value {
    floor_ns: u64,
    frac_hundredths: u8,
}

impl P95Value {
    /// 厳密値の整数部（ns）。
    pub fn floor_ns(&self) -> u64 {
        self.floor_ns
    }

    /// 厳密値の小数部（0〜99、1/100 ns 単位）。
    pub fn frac_hundredths(&self) -> u8 {
        self.frac_hundredths
    }

    /// 切り上げた u64 ns。上限照合に渡す値（モジュール doc の根拠を参照）。
    pub fn ceil_ns(&self) -> u64 {
        if self.frac_hundredths > 0 {
            self.floor_ns.saturating_add(1)
        } else {
            self.floor_ns
        }
    }
}

/// 線形補間による p95。0 件、または内部計算の失敗は `None`（fail-closed）。
///
/// 入力の順序は問わない（内部でコピーを昇順に整列する）。添字アクセス・`unwrap` を使わない。
pub fn percentile_95_ns(samples: &[u64]) -> Option<P95Value> {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    percentile_95_sorted(&sorted)
}

fn percentile_95_sorted(sorted: &[u64]) -> Option<P95Value> {
    let n = u128::try_from(sorted.len()).ok()?;
    let pos = n.checked_sub(1)?.checked_mul(95)?;
    let idx = usize::try_from(pos / 100).ok()?;
    let rem = pos % 100;
    let lo = *sorted.get(idx)?;
    let hi = match sorted.get(idx.checked_add(1)?) {
        Some(&h) if rem != 0 => h,
        _ => lo,
    };
    let num = u128::from(hi.checked_sub(lo)?).checked_mul(rem)?;
    let floor_ns = u64::try_from(u128::from(lo).checked_add(num / 100)?).ok()?;
    let frac_hundredths = u8::try_from(num % 100).ok()?;
    Some(P95Value {
        floor_ns,
        frac_hundredths,
    })
}

/// 参考値との比較（情報用。合否に使わない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceComparison {
    /// 厳密な p95 が参考値（250ms）未満（REQ-31 正常系の「未満」）。
    Below,
    /// 厳密な p95 が参考値以上。上限照合（`==` は合格）の境界とは別物。
    AtOrAbove,
}

/// レポート作成のエラー（入力本文を保持しない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LatencyReportError {
    /// 計測値が 0 件。
    Empty,
    /// 内部計算の失敗（fail-closed）。
    Internal,
}

impl LatencyReportError {
    /// 機械可読なエラーコード（英語）。
    pub fn code(&self) -> &'static str {
        match self {
            Self::Empty => "empty_samples",
            Self::Internal => "internal_error",
        }
    }
}

impl fmt::Display for LatencyReportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "latency report failed: {}", self.code())
    }
}

impl std::error::Error for LatencyReportError {}

/// p95 と参考値を持つレポート。合否・終了コードは持たない（上限照合は [`crate::latency_limit::check_latency_limit`]）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatencyReport {
    iters: usize,
    warmup: usize,
    p95: P95Value,
    min_ns: u64,
    max_ns: u64,
    reference_comparison: ReferenceComparison,
}

impl LatencyReport {
    /// 計測回数。
    pub fn iters(&self) -> usize {
        self.iters
    }

    /// warmup 回数。
    pub fn warmup(&self) -> usize {
        self.warmup
    }

    /// 切り上げ済みの p95（ns）。`check_latency_limit`（#129）が `latency_if_exceeded` へ渡す値。
    pub fn p95_ns(&self) -> u64 {
        self.p95.ceil_ns()
    }

    /// 厳密な p95。
    pub fn p95(&self) -> P95Value {
        self.p95
    }

    /// 最小の計測値（ns）。
    pub fn min_ns(&self) -> u64 {
        self.min_ns
    }

    /// 最大の計測値（ns）。
    pub fn max_ns(&self) -> u64 {
        self.max_ns
    }

    /// 参考値（[`REFERENCE_P95_NS`]）。
    pub fn reference_p95_ns(&self) -> u64 {
        REFERENCE_P95_NS
    }

    /// 参考値との比較（情報用）。
    pub fn reference_comparison(&self) -> ReferenceComparison {
        self.reference_comparison
    }

    /// p95 の算出方法の名前。
    pub fn method(&self) -> &'static str {
        P95_METHOD
    }
}

/// 計測値から [`LatencyReport`] を作る（構築の唯一の入口）。
pub fn summarize_latency(samples: &LatencySamples) -> Result<LatencyReport, LatencyReportError> {
    let raw = samples.samples_ns();
    if raw.is_empty() {
        return Err(LatencyReportError::Empty);
    }
    let mut sorted = raw.to_vec();
    sorted.sort_unstable();
    let p95 = percentile_95_sorted(&sorted).ok_or(LatencyReportError::Internal)?;
    let min_ns = *sorted.first().ok_or(LatencyReportError::Internal)?;
    let max_ns = *sorted.last().ok_or(LatencyReportError::Internal)?;
    // 厳密値 = floor + frac/100（frac < 1）なので、厳密値 < L ⇔ floor < L。
    let reference_comparison = if p95.floor_ns() < REFERENCE_P95_NS {
        ReferenceComparison::Below
    } else {
        ReferenceComparison::AtOrAbove
    };
    Ok(LatencyReport {
        iters: samples.iters(),
        warmup: samples.warmup(),
        p95,
        min_ns,
        max_ns,
        reference_comparison,
    })
}

impl fmt::Display for LatencyReport {
    /// 英語 1 行。数値と固定文字列のみで、入力本文・パスを含めない。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // ms 表記は表示専用（小数 2 桁に切り捨て。比較には使わない）。
        let ms_int = self.p95.floor_ns() / 1_000_000;
        let ms_frac = (self.p95.floor_ns() % 1_000_000) / 10_000;
        let cmp = match self.reference_comparison {
            ReferenceComparison::Below => "below",
            ReferenceComparison::AtOrAbove => "at or above",
        };
        write!(
            f,
            "latency p95 {ms_int}.{ms_frac:02} ms ({} ns; n={}, warmup={}; method: {}); \
             reference {} ms ({cmp}; informational only, not a pass/fail criterion; \
             pass/fail uses the user-configured limit)",
            self.p95_ns(),
            self.iters,
            self.warmup,
            P95_METHOD,
            REFERENCE_P95_NS / 1_000_000,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package_outcome::LimitBreach;

    fn p(v: &[u64]) -> (u64, u8, u64) {
        let x = percentile_95_ns(v).unwrap();
        (x.floor_ns(), x.frac_hundredths(), x.ceil_ns())
    }

    #[test]
    fn req31_p95_known_distributions() {
        let r100: Vec<u64> = (1..=100).collect();
        assert_eq!(p(&r100), (95, 5, 96));
        let r20: Vec<u64> = (1..=20).collect();
        assert_eq!(p(&r20), (19, 5, 20));
        assert_eq!(p(&[42]), (42, 0, 42));
        assert_eq!(p(&[0, 100]), (95, 0, 95));
        let r21: Vec<u64> = (0..=20).map(|x| x * 10).collect();
        assert_eq!(p(&r21), (190, 0, 190));
        assert_eq!(p(&[7; 50]), (7, 0, 7));
    }

    #[test]
    fn req31_p95_is_order_independent() {
        assert_eq!(p(&[5, 1, 4, 2, 3]), p(&[1, 2, 3, 4, 5]));
        assert_eq!(p(&[1, 2, 3, 4, 5]), (4, 80, 5));
    }

    #[test]
    fn req31_p95_near_u64_max_does_not_overflow() {
        let (f, fr, c) = p(&[u64::MAX - 1, u64::MAX]);
        assert_eq!((f, fr), (u64::MAX - 1, 95));
        assert_eq!(c, u64::MAX);
    }

    #[test]
    fn req31_p95_empty_is_none() {
        assert_eq!(percentile_95_ns(&[]), None);
    }

    #[test]
    fn req31_ceil_is_fail_closed_at_limit_boundary() {
        let l = REFERENCE_P95_NS;
        // 厳密値 L+9.5 は切り上げで L+10。
        let v = percentile_95_ns(&[l, l + 10]).unwrap();
        assert_eq!((v.floor_ns(), v.frac_hundredths()), (l + 9, 50));
        assert_eq!(v.ceil_ns(), l + 10);
        // 厳密値が L 直上（L+0.95）でも切り上げで超過扱いになる。
        let v = percentile_95_ns(&[l, l + 1]).unwrap();
        assert_eq!((v.floor_ns(), v.frac_hundredths()), (l, 95));
        assert_eq!(v.ceil_ns(), l + 1);
        assert!(LimitBreach::latency_if_exceeded(v.ceil_ns(), l).is_some());
        // ちょうど L は合格側。
        let v = percentile_95_ns(&[l, l]).unwrap();
        assert_eq!(v.ceil_ns(), l);
        assert!(LimitBreach::latency_if_exceeded(v.ceil_ns(), l).is_none());
    }

    fn cmp(v: &[u64]) -> ReferenceComparison {
        let s = crate::latency::test_samples(v);
        summarize_latency(&s).unwrap().reference_comparison()
    }

    #[test]
    fn req31_reference_comparison_boundary() {
        let l = REFERENCE_P95_NS;
        assert_eq!(cmp(&[l - 1]), ReferenceComparison::Below);
        assert_eq!(cmp(&[l]), ReferenceComparison::AtOrAbove);
        assert_eq!(cmp(&[l + 1]), ReferenceComparison::AtOrAbove);
        // 厳密値 L-0.05（floor = L-1, frac 95）は Below。
        assert_eq!(cmp(&[l - 1, l]), ReferenceComparison::Below);
    }

    #[test]
    fn req31_empty_samples_error_code() {
        let s = crate::latency::test_samples(&[]);
        let e = summarize_latency(&s).unwrap_err();
        assert_eq!(e, LatencyReportError::Empty);
        assert_eq!(e.code(), "empty_samples");
        assert_eq!(LatencyReportError::Internal.code(), "internal_error");
    }

    #[test]
    fn req31_report_display_marks_reference_as_informational() {
        let s = crate::latency::test_samples(&[650_000; 10]);
        let r = summarize_latency(&s).unwrap();
        assert_eq!(
            (r.min_ns(), r.max_ns(), r.p95_ns()),
            (650_000, 650_000, 650_000)
        );
        let text = r.to_string();
        assert!(text.starts_with("latency p95 0.65 ms (650000 ns; n=10, warmup=0;"));
        assert!(text.contains("reference 250 ms (below; informational only"));
        assert!(text.contains("not a pass/fail criterion"));
    }
}
