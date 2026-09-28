//! 信頼区間の重なり判定の結合テスト（REQ-26 境界値・TASK-26.3-1・issue #104）。
//!
//! 証拠の種別: テストハーネス（Linux x86_64・CPU）。数値の出典は PoC-19 の
//! 実機結果（`jobs/threeseed/result.json`。CPU device）で、GPU（Metal / MLX）
//! での同じ結論の再確認は本テストの範囲外（issue #105）。

use fandhe_edge_eval::reproducibility::{
    DisjointPair, OverlapVerdict, SeedRun, judge_overlap, judge_reproducibility,
};
use fandhe_edge_eval::wilson;

const EPSILON: f64 = 1e-9;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

fn run(seed: u64, correct: u64, total: u64) -> SeedRun {
    SeedRun {
        seed,
        correct,
        total,
    }
}

/// (a) PoC-19 の再現: `wilson_ci(correct, 650, 1.959964)` が PoC-19 実機結果の
/// lo/hi と 1e-9 で一致し、`judge_overlap` が `AllPairsOverlap` になる。
#[test]
fn poc19_intervals_all_overlap() {
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

    let report = judge_overlap(&[a, b, c]).expect("3 valid intervals");
    assert_eq!(report.verdict(), OverlapVerdict::AllPairsOverlap);
    assert!(report.disjoint_pairs().is_empty());
    assert_eq!(report.run_count(), 3);
}

/// (b) 同じ件数を `judge_reproducibility`（`z = WILSON_Z_95` = 1.96 の経路）に
/// 通しても `AllPairsOverlap` になる（区間の値は PoC の値と照合しない。
/// z が異なるため）。
#[test]
fn judge_reproducibility_matches_poc19_run_counts() {
    let runs = [run(0, 214, 650), run(1, 210, 650), run(2, 212, 650)];
    let report = judge_reproducibility(&runs).expect("3 valid runs");
    assert_eq!(report.verdict(), OverlapVerdict::AllPairsOverlap);
    assert!(report.disjoint_pairs().is_empty());
}

/// (c) 重ならない例: (20,100)・(50,100)・(80,100) → 全ペア重ならない。
/// 事前の試算（wilson の式からの手計算）で 3 ペアすべて離れることを確認済み。
#[test]
fn all_pairs_disjoint_when_far_apart() {
    let runs = [run(0, 20, 100), run(1, 50, 100), run(2, 80, 100)];
    let report = judge_reproducibility(&runs).expect("3 valid runs");
    assert_eq!(report.verdict(), OverlapVerdict::SomePairsDisjoint);
    assert_eq!(
        report.disjoint_pairs(),
        &[
            DisjointPair::new(0, 1),
            DisjointPair::new(0, 2),
            DisjointPair::new(1, 2),
        ]
    );
}

/// (d) 一部だけ重ならない例: (10,100)・(15,100)・(25,100) → (0,2) の 1 ペアのみ
/// 重ならない。事前の試算で (0,1)・(1,2) は重なり、(0,2) のみ離れることを
/// 確認済み。
#[test]
fn only_one_pair_disjoint() {
    let runs = [run(0, 10, 100), run(1, 15, 100), run(2, 25, 100)];
    let report = judge_reproducibility(&runs).expect("3 valid runs");
    assert_eq!(report.verdict(), OverlapVerdict::SomePairsDisjoint);
    assert_eq!(report.disjoint_pairs(), &[DisjointPair::new(0, 2)]);
}

/// (e) 4 seed 以上（5 件）でも動作する。
#[test]
fn five_runs_all_overlap() {
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
