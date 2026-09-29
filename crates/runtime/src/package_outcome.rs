//! package 工程の終了コード決定（REQ-21・REQ-30・TASK-21.3-1・#132）。
//!
//! 資源上限（容量。将来は待ち時間も。#133）の超過を、合否判定より優先して
//! `limit_exceeded`（終了コード 20）へ写す純粋関数を持つ。CLI の `package` 工程
//! （TASK-33.x・#123）が、上限の照合結果と合否判定をここへ渡して終了コードを得る想定。
//!
//! # 優先順（PoC-16 の package 工程と同じ）
//!
//! 1. 上限超過が 1 件でもあれば `LimitExceeded`（20）。合否判定は見ない
//! 2. それ以外で合否判定が不合格なら `JudgedFail`（10）
//! 3. それ以外で合否判定が判定不能（件数不足等）なら `Pending`（12。合格扱いにしない）
//! 4. それ以外は `Ok`（0）
//!
//! 超過を合格扱いにする経路は作らない（fail-closed。`quality` が `Pass` でも 20）。
//!
//! # 範囲外（実装済みを装わない）
//!
//! - 超過の生成元となる上限との照合（`total_bytes > limit_bytes`。上限ちょうどは超過でない）は
//!   TASK-30.2（#124）の責務で、本モジュールは [`LimitBreach`] を入力として受け取る。
//!   #124 のマージ後に `measure_package` からの結線確認を行う
//! - 利用者が設定する容量上限の定義ファイルへの取り込み（読み込みと範囲検証）は未実装
//! - 評価器の判定から [`PackageQualityJudgment`] への変換（評価器の判定不能を
//!   `Undeterminable` へ渡す変換を含む）、CLI・JSON 出力への配線は TASK-33.x の責務
//! - 待ち時間の上限超過（[`LimitBreach`] への追加）は #133 の責務
//!
//! 証拠の種別: テストハーネス。本番データでの `limit_exceeded` の再実演は未実施
//! （PoC-16 で本番データ確認したのは `ok`・`judged_fail` のみ）。

use fandhe_edge_core::exitcode::ExitCode;

/// package 工程の合否判定の入力（評価器側の判定から呼び出し側が変換する）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageQualityJudgment {
    /// 合否基準を満たす。
    Pass,
    /// 合否基準を満たさない（`judged_fail`）。
    Fail,
    /// 判定不能（評価器の `Undeterminable`。件数不足等。合格扱いにせず `pending` へ写す。REQ-24・REQ-21）。
    Undeterminable,
    /// 合否基準が未設定（PoC-16 の acceptance なしに相当。`ok`）。
    NotDefined,
}

/// 上限超過の種類。値はバイト数などの数値のみで、パスやデータ本文は持たない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LimitBreach {
    /// 容量上限の超過（REQ-30）。
    Capacity {
        /// 計測した非圧縮合計バイト数。
        measured_bytes: u64,
        /// 上限バイト数。
        limit_bytes: u64,
    },
}

/// package 工程の判定結果の区分。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackageVerdict {
    /// 合格。
    Pass,
    /// 不合格。
    Fail,
    /// 上限超過（合否判定に優先する）。
    LimitExceeded,
    /// 判定不能（終了コード 12）。
    Undeterminable,
    /// 合否基準が未設定。
    NotDefined,
}

/// package 工程の終了コード決定の結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageOutcome {
    /// REQ-21 の終了コード。
    pub exit_code: ExitCode,
    /// 判定結果の区分。
    pub verdict: PackageVerdict,
    /// 超過した上限（呼び出し側が渡した順）。
    pub breaches: Vec<LimitBreach>,
}

/// 上限超過と合否判定から終了コードを決める（純粋関数。panic しない）。
pub fn resolve_package_outcome(
    breaches: &[LimitBreach],
    quality: PackageQualityJudgment,
) -> PackageOutcome {
    let (exit_code, verdict) = if !breaches.is_empty() {
        (ExitCode::LimitExceeded, PackageVerdict::LimitExceeded)
    } else {
        match quality {
            PackageQualityJudgment::Fail => (ExitCode::JudgedFail, PackageVerdict::Fail),
            PackageQualityJudgment::Undeterminable => {
                (ExitCode::Pending, PackageVerdict::Undeterminable)
            }
            PackageQualityJudgment::Pass => (ExitCode::Ok, PackageVerdict::Pass),
            PackageQualityJudgment::NotDefined => (ExitCode::Ok, PackageVerdict::NotDefined),
        }
    };
    PackageOutcome {
        exit_code,
        verdict,
        breaches: breaches.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn over() -> Vec<LimitBreach> {
        vec![LimitBreach::Capacity {
            measured_bytes: 40_000_001,
            limit_bytes: 40_000_000,
        }]
    }

    #[test]
    fn req21_capacity_breach_beats_every_quality() {
        for q in [
            PackageQualityJudgment::Pass,
            PackageQualityJudgment::Fail,
            PackageQualityJudgment::Undeterminable,
            PackageQualityJudgment::NotDefined,
        ] {
            let o = resolve_package_outcome(&over(), q);
            assert_eq!(o.exit_code, ExitCode::LimitExceeded);
            assert_eq!(o.exit_code.code(), 20);
            assert_eq!(o.verdict, PackageVerdict::LimitExceeded);
            assert_eq!(o.breaches, over());
        }
    }

    #[test]
    fn req21_no_breach_fail_is_judged_fail_10() {
        let o = resolve_package_outcome(&[], PackageQualityJudgment::Fail);
        assert_eq!(o.exit_code.code(), 10);
        assert_eq!(o.verdict, PackageVerdict::Fail);
        assert!(o.breaches.is_empty());
    }

    #[test]
    fn req21_no_breach_undeterminable_is_pending_12() {
        let o = resolve_package_outcome(&[], PackageQualityJudgment::Undeterminable);
        assert_eq!(o.exit_code, ExitCode::Pending);
        assert_eq!(o.exit_code.code(), 12);
        assert_eq!(o.verdict, PackageVerdict::Undeterminable);
        assert!(o.breaches.is_empty());
    }

    #[test]
    fn req21_no_breach_pass_or_undefined_is_ok_0() {
        let p = resolve_package_outcome(&[], PackageQualityJudgment::Pass);
        assert_eq!((p.exit_code.code(), p.verdict), (0, PackageVerdict::Pass));
        let n = resolve_package_outcome(&[], PackageQualityJudgment::NotDefined);
        assert_eq!(
            (n.exit_code.code(), n.verdict),
            (0, PackageVerdict::NotDefined)
        );
    }
}
