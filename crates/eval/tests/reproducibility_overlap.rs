//! 信頼区間の重なり判定の結合テスト（REQ-26 境界値・TASK-26.3-1・issue #104）。
//!
//! 証拠の種別: テストハーネス（Linux x86_64・CPU）。数値の出典は PoC-19 の
//! 実機結果（`jobs/threeseed/result.json`。CPU device）で、GPU（Metal / MLX）
//! での同じ結論の再確認は本テストの範囲外（issue #105）。

use fandhe_edge_eval::reproducibility::{
    OverlapVerdict, ReproducibilityError, SeedRun, judge_overlap, judge_reproducibility,
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

/// `OverlapReport::disjoint_pairs()` の期待値確認用ヘルパー。
/// `DisjointPair` はコンストラクタを crate 内部限定にしている
/// （壊れた値を外部から組み立てられないようにするため）ため、
/// 結合テスト側は `(first(), second())` のタプルへ写してから比較する。
fn pair_tuples(report: &fandhe_edge_eval::reproducibility::OverlapReport) -> Vec<(usize, usize)> {
    report
        .disjoint_pairs()
        .iter()
        .map(|pair| (pair.first(), pair.second()))
        .collect()
}

/// (a) PoC-19 の再現: `wilson_ci(correct, 650, 1.959964)` が PoC-19 実機結果の
/// lo/hi と 1e-9 で一致する。PoC-19 はより精度の高い z（1.959964）を使って
/// おり、評価契約が要求する Wilson **95%**（`WILSON_Z_95` = 1.96）とは
/// [`fandhe_edge_eval::reproducibility::OVERLAP_TOLERANCE`] を超えて異なる
/// ため、公開 API `judge_overlap` はこれを 95% 区間として受け付けず
/// `NonWilson95Z` を返す（issue #104 レビュー指摘・PR #243。
/// `wilson_ci` が任意の正の z を受け付けるため、この検証が無いと
/// 95% 以外の区間を再現性ありと誤判定できてしまっていた）。
/// 同じ件数を `judge_reproducibility`（z = `WILSON_Z_95` 経路）に通した
/// 重なり判定は `judge_reproducibility_matches_poc19_run_counts` で確認する。
#[test]
fn poc19_intervals_use_non_wilson95_z_and_are_rejected() {
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

    assert_eq!(
        judge_overlap(&[a, b, c]).unwrap_err(),
        ReproducibilityError::NonWilson95Z { index: 0, z }
    );
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
    assert_eq!(pair_tuples(&report), vec![(0, 1), (0, 2), (1, 2)]);
}

/// (d) 一部だけ重ならない例: (10,100)・(15,100)・(25,100) → (0,2) の 1 ペアのみ
/// 重ならない。事前の試算で (0,1)・(1,2) は重なり、(0,2) のみ離れることを
/// 確認済み。
#[test]
fn only_one_pair_disjoint() {
    let runs = [run(0, 10, 100), run(1, 15, 100), run(2, 25, 100)];
    let report = judge_reproducibility(&runs).expect("3 valid runs");
    assert_eq!(report.verdict(), OverlapVerdict::SomePairsDisjoint);
    assert_eq!(pair_tuples(&report), vec![(0, 2)]);
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
