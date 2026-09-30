//! 利用者設定の容量上限と合計容量の照合（REQ-30 異常系・REQ-21・TASK-30.2・#124）。
//!
//! 容量計測コア（`capacity`。#122）が得た非圧縮合計バイト数を、利用者が設定した上限と
//! 突き合わせて [`LimitBreach::Capacity`] を作る「上限超過ハンドラ」。CLI の `package` 工程が、
//! ここで得た [`CapacityLimitCheck::breach`] を
//! [`crate::package_outcome::resolve_package_outcome`] へ渡して終了コード 20（`limit_exceeded`）を得る。
//!
//! # 設計上の不変条件
//!
//! - 判定は利用者の設定値に対して行う。40MB（[`REFERENCE_CAPACITY_BYTES`]）は目安の参考値であり、
//!   `Default` もコンストラクタからの参照も持たない（事実上の閾値にしない）
//! - 境界規則（`>` で超過・`==` は超過でない）は [`LimitBreach::capacity_if_exceeded`] の
//!   1 箇所に集約されており、本モジュールは再実装せず 1 回だけ呼ぶ
//! - 終了コードの決定は `resolve_package_outcome` に一本化する。本モジュールは終了コードを持たず、
//!   優先順（上限超過 > 不合格 > 判定不能 > ok）を迂回する経路を作らない
//! - 上限値は外部入力として 0 を拒否する（`invalid_input`）。最大値は設けない（u64 の比較のみで
//!   算術をせず overflow しない。spec にも最大値の根拠が無い）
//! - [`crate::capacity::CapacityError`]（1 ファイル上限・合計あふれ。REQ-39 の資源上限）とは別物で、
//!   本モジュールは `CapacityError` を生成しない
//!
//! # 範囲外（実装済みを装わない）
//!
//! - 上限値の定義ファイル・CLI 引数からの取り込み: 入出力契約の変更を伴い、ユーザー承認が必要なため
//!   未実装。CLI は当面暫定固定値を [`CapacityLimit`] に包んで渡す
//! - 語彙ファイル超過構成の除外記録（TASK-30.3）・JSON 出力（CLI 側）
//! - 証拠種別: テストハーネス。本番データでの `limit_exceeded` 再実演は未実施

use crate::capacity::CapacityBreakdown;
use crate::package_outcome::LimitBreach;
use fandhe_edge_core::exitcode::ExitCode;
use std::fmt;

/// 容量の目安（バイト。REQ-30 の 40MB）。参考値であり合否のしきい値ではない。
pub const REFERENCE_CAPACITY_BYTES: u64 = 40_000_000;

/// 検証済みの容量上限（バイト。1 以上）。
///
/// 利用者の明示設定のみで生成する。40MB の参考値を既定値として持たない（REQ-30・TASK-30.2）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityLimit(u64);

impl CapacityLimit {
    /// バイト単位の上限を検証して作る。
    pub fn from_bytes(bytes: u64) -> Result<Self, CapacityLimitError> {
        if bytes == 0 {
            return Err(CapacityLimitError::Zero);
        }
        Ok(Self(bytes))
    }

    /// バイト単位の上限。
    pub fn as_bytes(&self) -> u64 {
        self.0
    }
}

/// 容量上限の検証エラー。CLI では `invalid_input`（64）へ写す。値は数値のみで入力本文を持たない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CapacityLimitError {
    /// 上限が 0。
    Zero,
}

impl CapacityLimitError {
    /// 機械可読な英語コード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::Zero => "capacity_limit_zero",
        }
    }

    /// 対応する終了コード（常に `invalid_input`=64）。
    pub fn exit_code(&self) -> ExitCode {
        ExitCode::InvalidInput
    }
}

impl fmt::Display for CapacityLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "capacity limit must be at least 1 byte"),
        }
    }
}

impl std::error::Error for CapacityLimitError {}

/// 上限との照合結果。上限未設定を合格（[`Self::Within`]）と区別し、合格を装わない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CapacityLimitCheck {
    /// 上限が設定されていない（照合していない。超過は生じず、合否は品質判定側に委ねる）。
    NotConfigured {
        /// 非圧縮合計バイト数。
        total_bytes: u64,
    },
    /// 合計が上限以下（ちょうど等しいを含む）。
    Within {
        /// 非圧縮合計バイト数。
        total_bytes: u64,
        /// 上限バイト数。
        limit_bytes: u64,
    },
    /// 合計が上限を超えた（中身は [`LimitBreach::Capacity`]）。
    Exceeded(LimitBreach),
}

impl CapacityLimitCheck {
    /// `resolve_package_outcome` へ渡す上限超過（超過時のみ 1 件）。終了コード決定への唯一の出口。
    pub fn breach(&self) -> Option<LimitBreach> {
        match self {
            Self::Exceeded(b) => Some(*b),
            _ => None,
        }
    }
}

/// 計測結果の合計容量を利用者設定の上限と照合する。
pub fn check_capacity_limit(
    breakdown: &CapacityBreakdown,
    limit: Option<CapacityLimit>,
) -> CapacityLimitCheck {
    let total_bytes = breakdown.total_bytes();
    let Some(limit) = limit else {
        return CapacityLimitCheck::NotConfigured { total_bytes };
    };
    match LimitBreach::capacity_if_exceeded(total_bytes, limit.as_bytes()) {
        Some(b) => CapacityLimitCheck::Exceeded(b),
        None => CapacityLimitCheck::Within {
            total_bytes,
            limit_bytes: limit.as_bytes(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::{PackageComponent, PackageFile, measure_package};

    fn breakdown_of(len: usize) -> CapacityBreakdown {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-capacity-limit-unit-{}-{len}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.bin");
        std::fs::write(&path, vec![0u8; len]).unwrap();
        let b = measure_package(&[PackageFile {
            component: PackageComponent::Weights,
            path,
        }])
        .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        b
    }

    #[test]
    fn req30_zero_limit_is_rejected_as_invalid_input() {
        let e = CapacityLimit::from_bytes(0).unwrap_err();
        assert_eq!(e, CapacityLimitError::Zero);
        assert_eq!(e.code(), "capacity_limit_zero");
        assert_eq!(e.exit_code().code(), 64);
        assert_eq!(e.to_string(), "capacity limit must be at least 1 byte");
    }

    #[test]
    fn req30_limit_accepts_one_and_u64_max() {
        assert_eq!(CapacityLimit::from_bytes(1).unwrap().as_bytes(), 1);
        assert_eq!(
            CapacityLimit::from_bytes(u64::MAX).unwrap().as_bytes(),
            u64::MAX
        );
    }

    #[test]
    fn req30_check_three_branches() {
        let b = breakdown_of(100);
        let none = check_capacity_limit(&b, None);
        assert_eq!(none, CapacityLimitCheck::NotConfigured { total_bytes: 100 });
        assert_eq!(none.breach(), None);

        let at = check_capacity_limit(&b, Some(CapacityLimit::from_bytes(100).unwrap()));
        assert_eq!(
            at,
            CapacityLimitCheck::Within {
                total_bytes: 100,
                limit_bytes: 100
            }
        );
        assert_eq!(at.breach(), None);

        let over = check_capacity_limit(&b, Some(CapacityLimit::from_bytes(99).unwrap()));
        assert_eq!(
            over.breach(),
            Some(LimitBreach::Capacity {
                measured_bytes: 100,
                limit_bytes: 99
            })
        );
    }

    #[test]
    fn req30_reference_value_is_not_a_threshold() {
        assert_eq!(REFERENCE_CAPACITY_BYTES, 40_000_000);
        // 合計が参考値未満でも、利用者上限を下回れば超過（参考値は判定に関与しない）。
        let b = breakdown_of(1000);
        let c = check_capacity_limit(&b, Some(CapacityLimit::from_bytes(999).unwrap()));
        assert!(matches!(c, CapacityLimitCheck::Exceeded(_)));
    }
}
