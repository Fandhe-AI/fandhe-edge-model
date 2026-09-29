//! package 工程の終了コード決定（REQ-21・REQ-30・REQ-31・TASK-21.3-1・#132・TASK-21.3-2・#133）。
//!
//! 資源上限（容量・待ち時間。REQ-30・REQ-31・#133）の超過を、合否判定より優先して
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
//! - 待ち時間上限の検証済み型 `LatencyLimit` と照合 `check_latency_limit` は #129 で追加済み
//!   （`latency_limit` モジュール）。定義ファイルへの取り込みは未実装
//! - 利用者が設定する容量上限の定義ファイルへの取り込み（読み込みと範囲検証）は未実装
//! - 評価器の判定から [`PackageQualityJudgment`] への変換（評価器の判定不能を
//!   `Undeterminable` へ渡す変換を含む）、CLI・JSON 出力への配線は TASK-33.x の責務
//! - 待ち時間の p95 は `latency_report::LatencyReport::p95_ns()`（切り上げ済み。#128）から得る。
//!   計測ハーネスは #127。利用者が設定する待ち時間上限の定義ファイルへの取り込み（範囲検証を含む）・CLI の `package` / `infer` への配線と
//!   JSON 出力（TASK-33.x）は未実装。本モジュールは p95 と上限を数値（ナノ秒）で受け取る
//!
//! # 待ち時間（REQ-31）の境界規則
//!
//! p95 が上限を超える（`>`）ときだけ [`LimitBreach::Latency`] とし、上限ちょうど（`==`）は
//! 合格とする。規則は [`LimitBreach::latency_if_exceeded`] の 1 箇所に集約し、
//! TASK-31.2（#129）・TASK-31.3（#130）は再実装せずこれを再利用する。250ms は目安の
//! 参考値であり、合否のしきい値としてここへ定数化しない。上限ちょうどの扱いは、計測経路を通した
//! 結合テスト（`tests/latency_limit_boundary.rs`。TASK-31.3・#130）で確認済み。
//!
//! infer 1 回ごとの処理時間上限（REQ-39 の資源上限。core の `infer_input` 等が
//! `LimitExceeded` へ写す）は別の仕組みで、REQ-31 の p95 照合とは別物である。
//!
//! 証拠の種別: テストハーネス。本番データでの `limit_exceeded` の再実演は未実施
//! （PoC-16 で本番データ確認したのは `ok`・`judged_fail` のみ）。上限ちょうどを合格とする
//! 扱いは PoC-14 で未確認で、テストハーネスでのみ確認している。

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
    /// 待ち時間（p95）上限の超過（REQ-31）。
    Latency {
        /// 計測した p95（ナノ秒）。
        measured_p95_ns: u64,
        /// 利用者が設定した上限（ナノ秒）。
        limit_ns: u64,
    },
}

impl LimitBreach {
    /// p95 が上限を超える（`>`）ときだけ待ち時間超過を返す。上限ちょうどは超過でない
    /// （REQ-31 の境界値・TASK-31.3）。計測経路を通した境界確認は
    /// `tests/latency_limit_boundary.rs`（#130。証拠種別はテストハーネス）。
    ///
    /// 整数の比較のみで算術をしないため overflow・panic しない。上限値の範囲検証は
    /// 上限を取り込む側の責務。
    pub fn latency_if_exceeded(measured_p95_ns: u64, limit_ns: u64) -> Option<LimitBreach> {
        (measured_p95_ns > limit_ns).then_some(LimitBreach::Latency {
            measured_p95_ns,
            limit_ns,
        })
    }
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

    const LIMIT_NS: u64 = 250_000_000;

    #[test]
    fn req31_latency_equal_to_limit_is_not_breach() {
        assert_eq!(LimitBreach::latency_if_exceeded(LIMIT_NS, LIMIT_NS), None);
    }

    #[test]
    fn req31_latency_one_ns_over_limit_is_breach() {
        assert_eq!(
            LimitBreach::latency_if_exceeded(LIMIT_NS + 1, LIMIT_NS),
            Some(LimitBreach::Latency {
                measured_p95_ns: 250_000_001,
                limit_ns: 250_000_000
            })
        );
    }

    #[test]
    fn req31_latency_below_limit_is_not_breach() {
        assert_eq!(
            LimitBreach::latency_if_exceeded(249_999_999, LIMIT_NS),
            None
        );
    }

    #[test]
    fn req31_latency_u64_extremes() {
        assert_eq!(LimitBreach::latency_if_exceeded(u64::MAX, u64::MAX), None);
        assert!(LimitBreach::latency_if_exceeded(u64::MAX, u64::MAX - 1).is_some());
        assert_eq!(LimitBreach::latency_if_exceeded(0, 0), None);
    }

    fn latency_over() -> Vec<LimitBreach> {
        vec![LimitBreach::Latency {
            measured_p95_ns: 250_000_001,
            limit_ns: 250_000_000,
        }]
    }

    #[test]
    fn req21_latency_breach_beats_every_quality() {
        for q in [
            PackageQualityJudgment::Pass,
            PackageQualityJudgment::Fail,
            PackageQualityJudgment::Undeterminable,
            PackageQualityJudgment::NotDefined,
        ] {
            let o = resolve_package_outcome(&latency_over(), q);
            assert_eq!(o.exit_code.code(), 20);
            assert_eq!(o.verdict, PackageVerdict::LimitExceeded);
            assert_eq!(o.breaches, latency_over());
        }
    }

    #[test]
    fn req21_latency_equal_to_limit_keeps_quality_code() {
        let breaches: Vec<LimitBreach> = LimitBreach::latency_if_exceeded(LIMIT_NS, LIMIT_NS)
            .into_iter()
            .collect();
        let code = |q| resolve_package_outcome(&breaches, q).exit_code.code();
        assert_eq!(code(PackageQualityJudgment::Fail), 10);
        assert_eq!(code(PackageQualityJudgment::Pass), 0);
        assert_eq!(code(PackageQualityJudgment::Undeterminable), 12);
        assert_eq!(code(PackageQualityJudgment::NotDefined), 0);
    }

    #[test]
    fn req21_capacity_and_latency_breaches_preserved_in_order() {
        let mut b = over();
        b.extend(latency_over());
        let o = resolve_package_outcome(&b, PackageQualityJudgment::Pass);
        assert_eq!(o.exit_code.code(), 20);
        assert_eq!(o.breaches, b);
    }
}
