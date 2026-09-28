//! 3 seed 以上の Wilson 95% 信頼区間の重なり判定（REQ-26 境界値・
//! TASK-26.3-1・issue #104）。
//!
//! 評価契約（`.claude/rules/evaluation-contract.md`「有意性・指標」）は
//! 「再現性は 3 seed 以上の Wilson 95% 信頼区間の重なりで示す」と定めている。
//! 本モジュールはその判定関数を提供する。出典は PoC-19
//! （`03-poc/model-lifecycle/scripts/exp_threeseed.py`）で、各 seed の
//! Wilson 区間を算出し、全ペアの重なり（`all_pairs_overlap`）を判定する
//! ロジックを Rust（std のみ）へ移植したもの。
//!
//! # 重なりの定義
//!
//! 区間は閉区間として扱う。2 区間が「重ならない」のは、一方の上限が
//! 他方の下限を許容差 [`OVERLAP_TOLERANCE`]（1e-9。
//! `.claude/rules/coding-rust.md`「浮動小数の比較は 1e-9 を明示し `==` で
//! 比較しない」）を超えて下回るときだけとする（PoC-19 の規則
//! `hi1 < lo2 or hi2 < lo1` と同じ）。端点が接する場合・許容差の範囲内の
//! 隙間は重なりとして扱う。
//!
//! 全体の判定は全ペアの重なりで行う。1 次元の区間では「全ペアが重なる ⇔
//! 区間の共通部分が空でない（`max(lo) <= min(hi)`）」と同値だが、本実装は
//! 診断のため重ならないペアの添字をすべて列挙する（全ペア走査。
//! [`OverlapReport::disjoint_pairs`]）。
//!
//! # z の検証・公開 API の限定（issue #104 レビュー指摘・PR #243）
//!
//! 評価契約は「Wilson **95%** 信頼区間の重なり」を要求するため、
//! 区間だけを受け取る内部関数 `judge_overlap` は各区間の z が
//! [`crate::wilson::WILSON_Z_95`]（1.96）と [`OVERLAP_TOLERANCE`] を超えて
//! 異ならないかを検証する（異なれば [`ReproducibilityError::NonWilson95Z`]）。
//! [`wilson::wilson_ci`] は任意の正の z を受け付けるため、この検証が無いと
//! z=0.1 のような 95% 以外の区間でも `AllPairsOverlap` を返し得てしまう。
//! [`judge_reproducibility`] は常に `wilson::wilson_ci95`（z が常に
//! [`crate::wilson::WILSON_Z_95`]）で区間を算出するため、この検証を
//! 経由しても通常は失敗しない。
//!
//! `judge_overlap` は区間の由来（seed）を検証できず、同一区間を 3 回渡しても
//! 件数・z の条件は満たしてしまうため、[`judge_reproducibility`] が行う
//! seed の重複検出・評価総数の一致検証を迂回する経路になり得る
//! （レビュー指摘）。そのため `judge_overlap` は crate 内部専用（非公開）とし、
//! 本モジュールの公開 API は seed を検証できる [`judge_reproducibility`] の
//! みに限定する。
//!
//! # 資源上限（REQ-39）
//!
//! [`MAX_REPRODUCIBILITY_RUNS`] を `Vec` の確保・O(k²) のペア走査より前に
//! 検証する。[`crate::holm::MAX_FAMILY_SIZE`]・
//! [`crate::significance::MAX_EVAL_RECORDS`] と同様、データ契約層の確定値が
//! 無い段階の暫定値で、実測に基づく調整は後続 TASK とする。
//!
//! # 決定性
//!
//! ペアの走査は入力順（インデックス昇順）で行い、並列化しない。
//! [`judge_reproducibility`] は `wilson::wilson_ci95` を使うため、
//! `z = `[`crate::wilson::WILSON_Z_95`] に固定される。
//!
//! # 実測の限界（実装済みを装わない）
//!
//! この判定ロジックは CPU で得た PoC-19・PoC-17 の実測値との照合で検証して
//! いるが、事前登録条件（Mac・GPU での 3 seed 以上の実再学習）における
//! 同じ結論の再確認は本 issue の範囲外（issue #104 の「人の対応予定」・
//! 後続 issue #105 で扱う）。
//!
//! # 対象外（本 issue の範囲外）
//!
//! - CPU 決定性テスト・GPU 未確認の限界の詳しい記述（issue #105）
//! - 旧モデル比較・回帰件数（TASK-26.1・issue #99。本モジュールとは独立に
//!   実装する。#99 完了後にモジュール統合を検討する余地はあるが本 issue
//!   では判断しない）
//! - CLI `evaluate` への配線・JSON・終了コードへの写像（TASK-33.x・issue #140）
//! - `trainer/` 側の 3 seed 再学習ジョブ管理（REQ-34）

use std::fmt;

use crate::wilson::{self, WilsonInterval};

/// 評価契約が要求する再現性判定の最小 seed 数（3 seed 以上）。
pub const MIN_REPRODUCIBILITY_RUNS: usize = 3;

/// 1 回の判定で扱う run 数の上限（REQ-39）。ペアの走査が O(k²) のため、
/// `Vec` の確保・走査より前に検証する。データ契約層の確定値が無い段階の
/// 暫定値で、実測に基づく調整は後続 TASK とする（[`crate::holm::MAX_FAMILY_SIZE`]
/// と同様の位置づけ）。
pub const MAX_REPRODUCIBILITY_RUNS: usize = 100;

/// 区間の重なり判定で使う許容差（`.claude/rules/coding-rust.md`「浮動小数の
/// 比較は許容差 1e-9 を明示する」）。
pub const OVERLAP_TOLERANCE: f64 = 1e-9;

/// 2 区間が重ならないと判定する規則（PoC-19 と同じ）を、許容差付きで
/// 判定する内部関数。接する場合・許容差内の隙間は「重なる」扱いにする。
///
/// 境界値・対称性はユニットテストで具体値により確認する。
fn intervals_overlap(a_lo: f64, a_hi: f64, b_lo: f64, b_hi: f64) -> bool {
    let disjoint = a_hi < b_lo - OVERLAP_TOLERANCE || b_hi < a_lo - OVERLAP_TOLERANCE;
    !disjoint
}

/// 1 seed 分の評価件数（正解数・評価総数）。[`judge_reproducibility`] の
/// 入力単位。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeedRun {
    /// この run を識別する seed 値（重複検出・診断用）。
    pub seed: u64,
    /// 正解件数。
    pub correct: u64,
    /// 評価総数（分母）。
    pub total: u64,
}

/// 重ならなかった区間の組（入力順の添字。`first < second`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisjointPair {
    first: usize,
    second: usize,
}

impl DisjointPair {
    /// 組の 1 つ目の添字（`first < second`）。フィールドは非公開で、
    /// crate 内（[`judge_overlap`]）でのみ構造体リテラルから直接組み立てる。
    /// 外部からは [`OverlapReport`] が返す値を経由してしか [`DisjointPair`]
    /// を得られないため、`first < second` の前提が壊れた値を利用側が
    /// 組み立てることはできない
    /// （coding-rust.md「壊れた値を表現できない型にする」）。結合テストは
    /// [`DisjointPair::first`]・[`DisjointPair::second`] のタプル比較で
    /// 期待値を確認する。
    pub fn first(&self) -> usize {
        self.first
    }

    /// 組の 2 つ目の添字（`first < second`）。
    pub fn second(&self) -> usize {
        self.second
    }
}

/// 全ペアの重なり判定結果。[`OverlapReport::disjoint_pairs`] が空であること
/// と本 enum の値は必ず対応する（[`OverlapReport`] のコンストラクタが
/// crate 内に限られているため、外部から不整合な組み合わせを作れない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OverlapVerdict {
    /// すべてのペアが重なる（再現性ありと判定できる）。
    AllPairsOverlap,
    /// 重ならないペアが 1 組以上ある。
    SomePairsDisjoint,
}

/// 信頼区間の重なり判定レポート。フィールドは非公開にし、判定
/// （[`OverlapReport::verdict`]）と重ならないペアの一覧
/// （[`OverlapReport::disjoint_pairs`]）の整合を型で保証する
/// （一覧が空 ⇔ `AllPairsOverlap`。`.claude/rules/coding-rust.md`「判定結果は
/// 壊れた値を表現できない型にする」）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlapReport {
    verdict: OverlapVerdict,
    disjoint_pairs: Vec<DisjointPair>,
    run_count: usize,
}

impl OverlapReport {
    /// 全体の重なり判定。
    pub fn verdict(&self) -> OverlapVerdict {
        self.verdict
    }

    /// 重ならなかったペアの一覧（入力順の添字。診断用）。
    /// `verdict()` が [`OverlapVerdict::AllPairsOverlap`] のときは常に空。
    pub fn disjoint_pairs(&self) -> &[DisjointPair] {
        &self.disjoint_pairs
    }

    /// 判定に使った run（区間）の件数。
    pub fn run_count(&self) -> usize {
        self.run_count
    }
}

/// [`judge_overlap`]・[`judge_reproducibility`] が返しうるエラー。
///
/// [`crate::holm::HolmError`]と同様、呼び出し側が終了コード（REQ-21）へ
/// 写す際に判別しやすいよう独立した enum にする（写像そのものは本モジュール
/// の範囲外）。エラー文字列にはデータ本文を含めない。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ReproducibilityError {
    /// run 数が [`MIN_REPRODUCIBILITY_RUNS`] 未満。評価契約が要求する
    /// 「3 seed 以上」を満たさないため合格扱いにしない。
    TooFewRuns {
        /// 実際に渡された run 数。
        got: usize,
        /// 要求される最小 run 数（[`MIN_REPRODUCIBILITY_RUNS`]）。
        min: usize,
    },
    /// run 数が [`MAX_REPRODUCIBILITY_RUNS`] を超える（REQ-39）。
    TooManyRuns {
        /// 実際に渡された run 数。
        got: usize,
        /// 上限値（[`MAX_REPRODUCIBILITY_RUNS`]）。
        limit: usize,
    },
    /// `index` 番目の区間の z が [`crate::wilson::WILSON_Z_95`]（1.96）と
    /// [`OVERLAP_TOLERANCE`] を超えて異なる。評価契約が要求する
    /// 「Wilson 95% 信頼区間」以外（例: z=0.1・90%・99% 等）を再現性判定に
    /// 使わせないための検査（issue #104 レビュー指摘・PR #243）。
    NonWilson95Z {
        /// 不一致が見つかった区間の添字。
        index: usize,
        /// 実際の z 値。
        z: f64,
    },
    /// `index` 番目の run から Wilson 区間が算出できない
    /// （`total == 0` または `correct > total`）。「重なる」扱いにせず
    /// fail-closed で拒否する。
    UndefinedInterval {
        /// 区間が算出できなかった run の添字。
        index: usize,
    },
    /// `seed` が複数の run で重複している。異なる 3 seed 以上という
    /// 評価契約の前提を満たさない。
    DuplicateSeed {
        /// 重複していた seed 値。
        seed: u64,
    },
    /// `index` 番目の run の評価総数 `total` が、先頭（0 番目）の run と
    /// 異なる。同じ凍結評価データ（REQ-17）へ適用した 3 seed 以上の結果
    /// なら件数は一致するはずで、不一致は別データへ適用した兆候として
    /// 拒否する。
    MismatchedTotals {
        /// 不一致が見つかった run の添字。
        index: usize,
    },
}

impl fmt::Display for ReproducibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReproducibilityError::TooFewRuns { got, min } => {
                write!(
                    f,
                    "too few runs for reproducibility check: got {got}, need at least {min}"
                )
            }
            ReproducibilityError::TooManyRuns { got, limit } => {
                write!(f, "run count {got} exceeds limit {limit}")
            }
            ReproducibilityError::NonWilson95Z { index, z } => {
                write!(
                    f,
                    "interval at index {index} uses z={z}, which is not the required Wilson 95% z (WILSON_Z_95)"
                )
            }
            ReproducibilityError::UndefinedInterval { index } => {
                write!(
                    f,
                    "run at index {index} has an undefined Wilson interval (total=0 or correct>total)"
                )
            }
            ReproducibilityError::DuplicateSeed { seed } => {
                write!(f, "seed {seed} is duplicated across runs")
            }
            ReproducibilityError::MismatchedTotals { index } => {
                write!(f, "run at index {index} has a different total than index 0")
            }
        }
    }
}

impl std::error::Error for ReproducibilityError {}

/// run 数を上限・下限の順で検証する（`Vec` の確保・O(k²) 走査より前に
/// 呼ぶ。REQ-39）。
fn validate_run_count(len: usize) -> Result<(), ReproducibilityError> {
    if len > MAX_REPRODUCIBILITY_RUNS {
        return Err(ReproducibilityError::TooManyRuns {
            got: len,
            limit: MAX_REPRODUCIBILITY_RUNS,
        });
    }
    if len < MIN_REPRODUCIBILITY_RUNS {
        return Err(ReproducibilityError::TooFewRuns {
            got: len,
            min: MIN_REPRODUCIBILITY_RUNS,
        });
    }
    Ok(())
}

/// 全ペアの重なりを走査する（z の検証は行わない内部関数）。
/// [`judge_overlap`]・PoC-19 実測値との照合ユニットテストの双方から使う
/// （REQ-39。`Vec` の確保・走査は呼び出し側が件数検証を終えてから行う）。
fn pairwise_overlap_report(intervals: &[WilsonInterval]) -> OverlapReport {
    let mut disjoint_pairs = Vec::new();
    for i in 0..intervals.len() {
        for j in (i + 1)..intervals.len() {
            let (Some(a), Some(b)) = (intervals.get(i), intervals.get(j)) else {
                continue;
            };
            if !intervals_overlap(a.lo(), a.hi(), b.lo(), b.hi()) {
                disjoint_pairs.push(DisjointPair {
                    first: i,
                    second: j,
                });
            }
        }
    }

    let verdict = if disjoint_pairs.is_empty() {
        OverlapVerdict::AllPairsOverlap
    } else {
        OverlapVerdict::SomePairsDisjoint
    };

    OverlapReport {
        verdict,
        disjoint_pairs,
        run_count: intervals.len(),
    }
}

/// 3 件以上の Wilson **95%** 信頼区間を受け取り、全ペアの重なりを判定する
/// （評価契約「再現性は 3 seed 以上の Wilson 95% 信頼区間の重なりで示す」）。
///
/// 検証順は 上限 → 下限 → z が [`crate::wilson::WILSON_Z_95`] と一致するか
/// → 全ペア走査（REQ-39。`Vec` の確保前に件数を検証する）。
///
/// `wilson::wilson_ci` は任意の正の z を受け付けるため、区間の z が
/// [`crate::wilson::WILSON_Z_95`]（1.96）と [`OVERLAP_TOLERANCE`] を超えて
/// 異なる場合は [`ReproducibilityError::NonWilson95Z`] を返し、95% 以外の
/// 区間（例: z=0.1・90%・99% 等）を再現性ありと誤判定させない
/// （issue #104 レビュー指摘・PR #243）。任意の z を扱う純粋な重なり判定は
/// crate 内部専用の [`pairwise_overlap_report`] に分離してある。
///
/// # crate 内部限定にしている理由（issue #104 レビュー指摘・PR #243）
///
/// 区間（[`WilsonInterval`]）だけを受け取る本関数は、区間の由来（どの seed
/// の run から算出したか）を検証できない。同一の区間を 3 回渡しても
/// 「3 件」「z が Wilson 95%」という条件は満たしてしまうため、
/// [`judge_reproducibility`] が行う seed の重複検出
/// （[`ReproducibilityError::DuplicateSeed`]）・評価総数の一致検証
/// （[`ReproducibilityError::MismatchedTotals`]）を経由せずに
/// `AllPairsOverlap` を得られてしまい、評価契約が要求する「3 seed 以上」
/// （REQ-26・`.claude/rules/evaluation-contract.md`「再現性は 3 seed 以上の
/// Wilson 95% 信頼区間の重なりで示す」）を実質的に迂回できる。
/// そのため公開 API は seed を検証できる [`judge_reproducibility`] のみとし、
/// 本関数は crate 内部（[`judge_reproducibility`] からの呼び出しとユニット
/// テスト）に限定する。
fn judge_overlap(intervals: &[WilsonInterval]) -> Result<OverlapReport, ReproducibilityError> {
    validate_run_count(intervals.len())?;

    for (index, interval) in intervals.iter().enumerate() {
        if (interval.z() - wilson::WILSON_Z_95).abs() > OVERLAP_TOLERANCE {
            return Err(ReproducibilityError::NonWilson95Z {
                index,
                z: interval.z(),
            });
        }
    }

    Ok(pairwise_overlap_report(intervals))
}

/// 3 件以上の seed run（正解数・評価総数の組）から Wilson 95% 信頼区間
/// （`z = `[`crate::wilson::WILSON_Z_95`]）を算出し、[`judge_overlap`] で
/// 重なりを判定する。
///
/// 検証順は 上限 → 下限 → (seed の重複・評価総数の一致を 1 走査で要素ごとに
/// 検証) → 区間の算出 → 全ペア走査（REQ-39。`Vec` の確保前に件数を検証する）。
/// seed の重複と評価総数の不一致は同じループの同じ添字で検証するため、
/// 入力によってはどちらが先に成立するかは添字の並びに依存する
/// （例: `[(0,_,5,10), (1,_,6,20), (1,_,7,10)]` は index=1 で
/// `MismatchedTotals` が先に返り、その後方にある seed 重複は検出されない）。
/// どちらのエラーも fail-closed に判定を止める点は変わらない。
///
/// `total == 0` または `correct > total` の run があると Wilson 区間が
/// 算出できない（`wilson::wilson_ci95` が `None`）。この場合「重なる」
/// 扱いにせず [`ReproducibilityError::UndefinedInterval`] を返す
/// （fail-closed）。
pub fn judge_reproducibility(runs: &[SeedRun]) -> Result<OverlapReport, ReproducibilityError> {
    validate_run_count(runs.len())?;

    let first_total = match runs.first() {
        Some(run) => run.total,
        None => {
            return Err(ReproducibilityError::TooFewRuns {
                got: 0,
                min: MIN_REPRODUCIBILITY_RUNS,
            });
        }
    };

    let mut seen_seeds: Vec<u64> = Vec::with_capacity(runs.len());
    for (index, run) in runs.iter().enumerate() {
        if seen_seeds.contains(&run.seed) {
            return Err(ReproducibilityError::DuplicateSeed { seed: run.seed });
        }
        seen_seeds.push(run.seed);

        if run.total != first_total {
            return Err(ReproducibilityError::MismatchedTotals { index });
        }
    }

    let mut intervals = Vec::with_capacity(runs.len());
    for (index, run) in runs.iter().enumerate() {
        let interval = wilson::wilson_ci95(run.correct, run.total)
            .ok_or(ReproducibilityError::UndefinedInterval { index })?;
        intervals.push(interval);
    }

    judge_overlap(&intervals)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    /// REQ-26・TASK-26.3-1: 接する場合（隙間 0）は重なりとして扱う。
    #[test]
    fn intervals_overlap_touching_bounds_overlap() {
        assert!(intervals_overlap(0.1, 0.3, 0.3, 0.5));
    }

    /// REQ-26・TASK-26.3-1: 許容差内の隙間（5e-10 < 1e-9）は重なりとして扱う。
    #[test]
    fn intervals_overlap_within_tolerance_overlap() {
        let hi = 0.3;
        let lo = 0.3 + 5e-10;
        assert!(intervals_overlap(0.1, hi, lo, 0.5));
    }

    /// REQ-26・TASK-26.3-1: 許容差を超える隙間（2e-9 > 1e-9）は重ならない。
    #[test]
    fn intervals_overlap_beyond_tolerance_disjoint() {
        let hi = 0.3;
        let lo = 0.3 + 2e-9;
        assert!(!intervals_overlap(0.1, hi, lo, 0.5));
    }

    /// REQ-26・TASK-26.3-1: 包含関係にある区間は重なる。
    #[test]
    fn intervals_overlap_containment_overlaps() {
        assert!(intervals_overlap(0.1, 0.9, 0.4, 0.5));
    }

    /// REQ-26・TASK-26.3-1: 引数の順序を入れ替えても結果は変わらない（対称性）。
    #[test]
    fn intervals_overlap_is_symmetric() {
        let forward = intervals_overlap(0.1, 0.3, 0.35, 0.5);
        let backward = intervals_overlap(0.35, 0.5, 0.1, 0.3);
        assert_eq!(forward, backward);
    }

    fn run(seed: u64, correct: u64, total: u64) -> SeedRun {
        SeedRun {
            seed,
            correct,
            total,
        }
    }

    /// REQ-26・TASK-26.3-1: PoC-19 の実機結果（`jobs/threeseed/result.json`。
    /// `wilson_ci(correct, 650, 1.959964)`）と 1e-9 で一致する。
    /// 全ペアの重なり判定そのものは、z の検証を行わない内部関数
    /// [`pairwise_overlap_report`] で確認する（PoC-19 は `1.96` を丸めない
    /// より精度の高い z を使っており、公開 API [`judge_overlap`] は
    /// [`ReproducibilityError::NonWilson95Z`] を返す前提のため。
    /// `judge_overlap_rejects_non_wilson95_z` 参照）。
    /// 証拠の種別: テストハーネス（数値の出典は PoC-19 の実機結果）。
    #[test]
    fn pairwise_overlap_report_matches_poc19_intervals() {
        let z = 1.959964_f64;
        let a = wilson::wilson_ci(214, 650, z).expect("valid interval");
        let b = wilson::wilson_ci(210, 650, z).expect("valid interval");
        let c = wilson::wilson_ci(212, 650, z).expect("valid interval");

        assert!(approx_eq(a.lo(), 0.29419969571312854));
        assert!(approx_eq(a.hi(), 0.3662684545020082));
        assert!(approx_eq(b.lo(), 0.28825582805163136));
        assert!(approx_eq(b.hi(), 0.3599769401892761));
        assert!(approx_eq(c.lo(), 0.2912268876983092));
        assert!(approx_eq(c.hi(), 0.36312357152971286));

        let report = pairwise_overlap_report(&[a, b, c]);
        assert_eq!(report.verdict(), OverlapVerdict::AllPairsOverlap);
        assert!(report.disjoint_pairs().is_empty());
        assert_eq!(report.run_count(), 3);
    }

    /// REQ-26・TASK-26.3-1: 同じ件数を `judge_reproducibility`
    /// （`z = WILSON_Z_95` = 1.96 の経路）に通しても全ペア重なりになる。
    #[test]
    fn judge_reproducibility_matches_poc19_counts() {
        let runs = [run(0, 214, 650), run(1, 210, 650), run(2, 212, 650)];
        let report = judge_reproducibility(&runs).expect("3 valid runs");
        assert_eq!(report.verdict(), OverlapVerdict::AllPairsOverlap);
        assert!(report.disjoint_pairs().is_empty());
    }

    /// REQ-26・TASK-26.3-1: 重ならない例（20/100・50/100・80/100）。
    /// 全ペアが離れることを事前の試算（wilson の式からの手計算）で確認済み。
    #[test]
    fn judge_reproducibility_all_pairs_disjoint() {
        let runs = [run(0, 20, 100), run(1, 50, 100), run(2, 80, 100)];
        let report = judge_reproducibility(&runs).expect("3 valid runs");
        assert_eq!(report.verdict(), OverlapVerdict::SomePairsDisjoint);
        assert_eq!(
            report.disjoint_pairs(),
            &[
                DisjointPair {
                    first: 0,
                    second: 1
                },
                DisjointPair {
                    first: 0,
                    second: 2
                },
                DisjointPair {
                    first: 1,
                    second: 2
                },
            ]
        );
    }

    /// REQ-26・TASK-26.3-1: 3 件中 1 ペアだけ離れる例（10/100・15/100・
    /// 25/100）。事前の試算で (0,1)・(1,2) は重なり、(0,2) のみ離れることを
    /// 確認済み。
    #[test]
    fn judge_reproducibility_one_pair_disjoint() {
        let runs = [run(0, 10, 100), run(1, 15, 100), run(2, 25, 100)];
        let report = judge_reproducibility(&runs).expect("3 valid runs");
        assert_eq!(report.verdict(), OverlapVerdict::SomePairsDisjoint);
        assert_eq!(
            report.disjoint_pairs(),
            &[DisjointPair {
                first: 0,
                second: 2
            }]
        );
    }

    /// REQ-26・TASK-26.3-1: 4 seed 以上（5 件）でも動作する。
    #[test]
    fn judge_reproducibility_five_runs() {
        let runs = [
            run(0, 214, 650),
            run(1, 210, 650),
            run(2, 212, 650),
            run(3, 213, 650),
            run(4, 211, 650),
        ];
        let report = judge_reproducibility(&runs).expect("5 valid runs");
        assert_eq!(report.verdict(), OverlapVerdict::AllPairsOverlap);
        assert_eq!(report.run_count(), 5);
    }

    /// REQ-26・TASK-26.3-1: 0 件・2 件は `TooFewRuns`。
    #[test]
    fn judge_overlap_too_few_runs() {
        assert_eq!(
            judge_overlap(&[]).unwrap_err(),
            ReproducibilityError::TooFewRuns {
                got: 0,
                min: MIN_REPRODUCIBILITY_RUNS,
            }
        );
        let a = wilson::wilson_ci95(5, 10).expect("valid interval");
        let b = wilson::wilson_ci95(6, 10).expect("valid interval");
        assert_eq!(
            judge_overlap(&[a, b]).unwrap_err(),
            ReproducibilityError::TooFewRuns { got: 2, min: 3 }
        );
    }

    /// REQ-26・TASK-26.3-1・REQ-39: 上限（101 件）超過は `TooManyRuns`。
    #[test]
    fn judge_overlap_too_many_runs() {
        let interval = wilson::wilson_ci95(5, 10).expect("valid interval");
        let intervals = vec![interval; MAX_REPRODUCIBILITY_RUNS + 1];
        assert_eq!(
            judge_overlap(&intervals).unwrap_err(),
            ReproducibilityError::TooManyRuns {
                got: MAX_REPRODUCIBILITY_RUNS + 1,
                limit: MAX_REPRODUCIBILITY_RUNS,
            }
        );
    }

    /// REQ-26・TASK-26.3-1・issue #104 レビュー指摘（PR #243）:
    /// `WILSON_Z_95`（1.96）から外れた z（1.959964。より精度の高い表現だが
    /// [`OVERLAP_TOLERANCE`] を超えて異なる）は `NonWilson95Z`。
    #[test]
    fn judge_overlap_non_wilson95_z_is_error() {
        let a = wilson::wilson_ci(5, 10, 1.96).expect("valid interval");
        let b = wilson::wilson_ci(5, 10, 1.959964).expect("valid interval");
        let c = wilson::wilson_ci(5, 10, 1.96).expect("valid interval");
        assert_eq!(
            judge_overlap(&[a, b, c]).unwrap_err(),
            ReproducibilityError::NonWilson95Z {
                index: 1,
                z: 1.959964
            }
        );
    }

    /// REQ-26・TASK-26.3-1・issue #104 レビュー指摘（PR #243）: レビューが
    /// 挙げた具体例そのもの。z=0.1 で統一された（区間同士は互いに一致する）
    /// 3 区間を渡しても `AllPairsOverlap` を返さず `NonWilson95Z` で拒否する
    /// （`wilson_ci` は任意の正の z を受け付けるため、この検証が無いと
    /// 95% 以外の区間を再現性ありと誤判定できてしまっていた）。
    #[test]
    fn judge_overlap_rejects_uniform_non_wilson95_z() {
        let a = wilson::wilson_ci(5, 10, 0.1).expect("valid interval");
        let b = wilson::wilson_ci(5, 10, 0.1).expect("valid interval");
        let c = wilson::wilson_ci(5, 10, 0.1).expect("valid interval");
        assert_eq!(
            judge_overlap(&[a, b, c]).unwrap_err(),
            ReproducibilityError::NonWilson95Z { index: 0, z: 0.1 }
        );
    }

    /// REQ-26・TASK-26.3-1: z の差が [`OVERLAP_TOLERANCE`]（1e-9）未満
    /// （5e-10）なら `NonWilson95Z` にならず判定が進む（`intervals_overlap`
    /// 側の許容差テストと対称の正常系。`WILSON_Z_95` との許容差判定にも
    /// 境界を持たせる）。
    #[test]
    fn judge_overlap_z_within_tolerance_is_accepted() {
        let a = wilson::wilson_ci(5, 10, 1.96).expect("valid interval");
        let b = wilson::wilson_ci(5, 10, 1.96 + 5e-10).expect("valid interval");
        let c = wilson::wilson_ci(5, 10, 1.96).expect("valid interval");
        let report = judge_overlap(&[a, b, c]).expect("z 差が許容差未満のため成功する");
        assert_eq!(report.verdict(), OverlapVerdict::AllPairsOverlap);
    }

    /// REQ-26・TASK-26.3-1: `total=0` は `UndefinedInterval`
    /// （`wilson_ci95` が `None` になるケースを fail-closed で拒否する）。
    #[test]
    fn judge_reproducibility_zero_total_is_error() {
        // 3 件とも total=0 で揃え、`MismatchedTotals` ではなく
        // `UndefinedInterval` を検出することを確認する。
        let runs = [run(0, 0, 0), run(1, 0, 0), run(2, 0, 0)];
        assert_eq!(
            judge_reproducibility(&runs).unwrap_err(),
            ReproducibilityError::UndefinedInterval { index: 0 }
        );
    }

    /// REQ-26・TASK-26.3-1: `correct > total` は `UndefinedInterval`。
    #[test]
    fn judge_reproducibility_correct_exceeds_total_is_error() {
        let runs = [run(0, 11, 10), run(1, 5, 10), run(2, 6, 10)];
        assert_eq!(
            judge_reproducibility(&runs).unwrap_err(),
            ReproducibilityError::UndefinedInterval { index: 0 }
        );
    }

    /// REQ-26・TASK-26.3-1: seed の重複は `DuplicateSeed`。
    #[test]
    fn judge_reproducibility_duplicate_seed_is_error() {
        let runs = [run(0, 5, 10), run(0, 6, 10), run(1, 7, 10)];
        assert_eq!(
            judge_reproducibility(&runs).unwrap_err(),
            ReproducibilityError::DuplicateSeed { seed: 0 }
        );
    }

    /// REQ-26・TASK-26.3-1: 評価総数の不一致は `MismatchedTotals`
    /// （同じ凍結評価データへの適用なら件数は一致するはず。REQ-17）。
    #[test]
    fn judge_reproducibility_mismatched_totals_is_error() {
        let runs = [run(0, 5, 10), run(1, 6, 20), run(2, 7, 10)];
        assert_eq!(
            judge_reproducibility(&runs).unwrap_err(),
            ReproducibilityError::MismatchedTotals { index: 1 }
        );
    }
}
