//! 利用者設定の待ち時間上限と p95 の照合（REQ-31 異常系・REQ-21・TASK-31.2・#129）。
//!
//! 計測（`latency`。#127）と p95 算出（`latency_report`。#128）の結果を、利用者が設定した上限と
//! 突き合わせて [`LimitBreach::Latency`] を作る「上限超過ハンドラ」。CLI の `package`／bench 工程
//! （TASK-33.x）が、ここで得た [`LatencyLimitCheck::breach`] を
//! [`crate::package_outcome::resolve_package_outcome`] へ渡して終了コード 20（`limit_exceeded`）を得る想定。
//!
//! # 設計上の不変条件
//!
//! - 判定は利用者の設定値に対して行う。250ms（`REFERENCE_P95_NS`）は初期値の参考であり、
//!   `Default` もコンストラクタからの参照も持たない（事実上の閾値にしない）
//! - 境界規則（`>` で超過・`==` は合格）は [`LimitBreach::latency_if_exceeded`] の 1 箇所に集約されており、
//!   本モジュールは再実装せず 1 回だけ呼ぶ
//! - 終了コードの決定は `resolve_package_outcome` に一本化する。本モジュールは終了コードを持たず、
//!   優先順（上限超過 > 不合格 > 判定不能 > ok）を迂回する経路を作らない
//! - 上限値は外部入力として 0・[`MAX_LATENCY_TIMEOUT_NS`] 超・ms 換算のオーバーフローを拒否する（REQ-39）
//!
//! # 範囲外（実装済みを装わない）
//!
//! - 上限値の取り込み: 定義ファイルの `limits.max_infer_p95_us`（µs）から CLI の `package` 工程が
//!   取り込み・照合する（#338）。CLI 引数からの取り込みは未実装。p95 値の JSON 出力は #340
//! - 上限ちょうどの境界判定は確認済み（`tests/latency_limit_boundary.rs`。TASK-31.3・#130）
//! - [`crate::latency::LatencyError`]（推論 1 回・計測全体の時間上限。REQ-39）は REQ-31 の p95 照合とは別物
//! - 実機（静かな Mac）での実計測は人間の作業。テストの証拠種別はテストハーネス（偽の時計・模擬バックエンド）

use crate::latency::MAX_LATENCY_TIMEOUT_NS;
use crate::latency_report::LatencyReport;
use crate::package_outcome::LimitBreach;
use fandhe_edge_core::exitcode::ExitCode;
use std::fmt;

/// 検証済みの待ち時間上限（ns。1 以上 [`MAX_LATENCY_TIMEOUT_NS`] 以下）。
///
/// 利用者の明示設定のみで生成する。250ms の参考値を既定値として持たない（REQ-31・TASK-31.2）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyLimit(u64);

impl LatencyLimit {
    /// ns 単位の上限を検証して作る。
    pub fn from_ns(ns: u64) -> Result<Self, LatencyLimitError> {
        if ns == 0 {
            return Err(LatencyLimitError::Zero);
        }
        if ns > MAX_LATENCY_TIMEOUT_NS {
            return Err(LatencyLimitError::AboveMax {
                value_ns: ns,
                max_ns: MAX_LATENCY_TIMEOUT_NS,
            });
        }
        Ok(Self(ns))
    }

    /// ms 単位の上限を ns へ換算して検証して作る。換算のオーバーフローは飽和させずエラーにする。
    pub fn from_millis(ms: u64) -> Result<Self, LatencyLimitError> {
        let ns = ms
            .checked_mul(1_000_000)
            .ok_or(LatencyLimitError::MillisOverflow { value_ms: ms })?;
        Self::from_ns(ns)
    }

    /// ns 単位の上限。
    pub fn as_ns(&self) -> u64 {
        self.0
    }
}

/// 待ち時間上限の検証エラー。CLI では `invalid_input`（64）へ写す。値は数値のみで入力本文を持たない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LatencyLimitError {
    /// 上限が 0。
    Zero,
    /// 上限が [`MAX_LATENCY_TIMEOUT_NS`] を超える。
    AboveMax {
        /// 指定値（ns）。
        value_ns: u64,
        /// 許容する最大値（ns）。
        max_ns: u64,
    },
    /// ms から ns への換算が u64 に収まらない。
    MillisOverflow {
        /// 指定値（ms）。
        value_ms: u64,
    },
}

impl LatencyLimitError {
    /// 機械可読な英語コード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::Zero => "latency_limit_zero",
            Self::AboveMax { .. } => "latency_limit_above_max",
            Self::MillisOverflow { .. } => "latency_limit_overflow",
        }
    }

    /// 対応する終了コード（常に `invalid_input`=64）。
    pub fn exit_code(&self) -> ExitCode {
        ExitCode::InvalidInput
    }
}

impl fmt::Display for LatencyLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "latency limit must be at least 1 ns"),
            Self::AboveMax { value_ns, max_ns } => {
                write!(f, "latency limit {value_ns} ns exceeds maximum {max_ns} ns")
            }
            Self::MillisOverflow { value_ms } => {
                write!(
                    f,
                    "latency limit {value_ms} ms overflows nanosecond conversion"
                )
            }
        }
    }
}

impl std::error::Error for LatencyLimitError {}

/// 上限との照合結果。上限未設定を合格（[`Self::Within`]）と区別し、合格を装わない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LatencyLimitCheck {
    /// 上限が設定されていない（照合していない。超過は生じず、合否は品質判定側に委ねる）。
    NotConfigured {
        /// 切り上げ済みの p95（ns）。
        p95_ns: u64,
    },
    /// p95 が上限以下（ちょうど等しいを含む）。
    Within {
        /// 切り上げ済みの p95（ns）。
        p95_ns: u64,
        /// 上限（ns）。
        limit_ns: u64,
    },
    /// p95 が上限を超えた（中身は [`LimitBreach::Latency`]）。
    Exceeded(LimitBreach),
}

impl LatencyLimitCheck {
    /// `resolve_package_outcome` へ渡す上限超過（超過時のみ 1 件）。終了コード決定への唯一の出口。
    pub fn breach(&self) -> Option<LimitBreach> {
        match self {
            Self::Exceeded(b) => Some(*b),
            _ => None,
        }
    }
}

/// レポートの p95 を利用者設定の上限と照合する。
///
/// 比較は [`LimitBreach::latency_if_exceeded`] に委ねる（切り上げ済み p95 を渡すため fail-closed）。
/// 入力は成功した [`LatencyReport`] のみで、計測エラーは扱わない。
pub fn check_latency_limit(
    report: &LatencyReport,
    limit: Option<LatencyLimit>,
) -> LatencyLimitCheck {
    let p95_ns = report.p95_ns();
    let Some(limit) = limit else {
        return LatencyLimitCheck::NotConfigured { p95_ns };
    };
    match LimitBreach::latency_if_exceeded(p95_ns, limit.as_ns()) {
        Some(b) => LatencyLimitCheck::Exceeded(b),
        None => LatencyLimitCheck::Within {
            p95_ns,
            limit_ns: limit.as_ns(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::latency_report::summarize_latency;

    #[test]
    fn req31_limit_from_ns_rejects_zero() {
        let e = LatencyLimit::from_ns(0).unwrap_err();
        assert_eq!(e, LatencyLimitError::Zero);
        assert_eq!(e.exit_code().code(), 64);
        assert_eq!(e.code(), "latency_limit_zero");
    }

    #[test]
    fn req31_limit_from_ns_rejects_above_max() {
        assert_eq!(
            LatencyLimit::from_ns(MAX_LATENCY_TIMEOUT_NS + 1),
            Err(LatencyLimitError::AboveMax {
                value_ns: MAX_LATENCY_TIMEOUT_NS + 1,
                max_ns: MAX_LATENCY_TIMEOUT_NS
            })
        );
        assert_eq!(
            LatencyLimit::from_ns(MAX_LATENCY_TIMEOUT_NS)
                .unwrap()
                .as_ns(),
            MAX_LATENCY_TIMEOUT_NS
        );
    }

    #[test]
    fn req31_limit_from_millis_overflow_is_error() {
        assert_eq!(
            LatencyLimit::from_millis(u64::MAX),
            Err(LatencyLimitError::MillisOverflow { value_ms: u64::MAX })
        );
    }

    #[test]
    fn req31_limit_from_millis_converts_exactly() {
        assert_eq!(LatencyLimit::from_millis(500).unwrap().as_ns(), 500_000_000);
    }

    #[test]
    fn req31_not_configured_yields_no_breach() {
        let r = summarize_latency(&crate::latency::test_samples(&[300_000_000; 10])).unwrap();
        let c = check_latency_limit(&r, None);
        assert_eq!(
            c,
            LatencyLimitCheck::NotConfigured {
                p95_ns: 300_000_000
            }
        );
        assert_eq!(c.breach(), None);
    }

    #[test]
    fn req31_within_and_exceeded_use_user_limit() {
        let r = summarize_latency(&crate::latency::test_samples(&[100; 10])).unwrap();
        let ok = check_latency_limit(&r, Some(LatencyLimit::from_ns(100).unwrap()));
        assert_eq!(
            ok,
            LatencyLimitCheck::Within {
                p95_ns: 100,
                limit_ns: 100
            }
        );
        let ng = check_latency_limit(&r, Some(LatencyLimit::from_ns(99).unwrap()));
        assert_eq!(
            ng.breach(),
            Some(LimitBreach::Latency {
                measured_p95_ns: 100,
                limit_ns: 99
            })
        );
    }

    #[test]
    fn req31_error_display_has_numbers_only() {
        assert_eq!(
            LatencyLimitError::Zero.to_string(),
            "latency limit must be at least 1 ns"
        );
        assert_eq!(
            LatencyLimitError::MillisOverflow { value_ms: 7 }.to_string(),
            "latency limit 7 ms overflows nanosecond conversion"
        );
    }
}
