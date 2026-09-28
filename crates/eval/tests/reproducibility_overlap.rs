//! 信頼区間の重なり判定の結合テスト（REQ-26 境界値・TASK-26.3-1・issue #104）。
//!
//! 証拠の種別: テストハーネス（Linux x86_64・CPU）。数値の出典は PoC-19 の
//! 実機結果（`jobs/threeseed/result.json`。CPU device）で、GPU（Metal / MLX）
//! での同じ結論の再確認は本テストの範囲外（issue #105）。

use fandhe_edge_eval::reproducibility::{OverlapVerdict, SeedRun, judge_reproducibility};

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

/// (a) 公開 API は seed を検証できる `judge_reproducibility` のみで、区間
/// だけを受け取る内部関数 `judge_overlap` は crate 外から呼べない
/// （issue #104 レビュー指摘・PR #243）。同一 seed を複数の run で渡すと、
/// 全 run が同一の (correct, total) で区間も同一（重なりは必ず成立する）
/// であっても `DuplicateSeed` で fail-closed に拒否され、区間だけを見て
/// `AllPairsOverlap`（3 seed 以上の再現性ありと誤判定）を返す経路が無い
/// ことを確認する。修正前は区間単位の公開関数へ同一区間を 3 回渡すことで
/// この検証を迂回できていた。
#[test]
fn duplicate_seed_cannot_bypass_reproducibility_check_via_identical_runs() {
    let runs = [run(0, 214, 650), run(0, 214, 650), run(0, 214, 650)];
    assert_eq!(
        judge_reproducibility(&runs).unwrap_err(),
        fandhe_edge_eval::reproducibility::ReproducibilityError::DuplicateSeed { seed: 0 }
    );
}

/// PoC-19 の実機結果（`jobs/threeseed/result.json`）と同じ run 数（correct・
/// total）を `judge_reproducibility`（`z = WILSON_Z_95` = 1.96 の経路）に
/// 通すと `AllPairsOverlap` になる（区間の値は PoC の値と照合しない。
/// PoC-19 はより精度の高い z（1.959964）を使っており、評価契約が要求する
/// Wilson **95%** とは異なる区間になるため。この非 95% z の拒否
/// 〔`NonWilson95Z`〕は crate 内部のユニットテスト
/// `judge_overlap_non_wilson95_z_is_error`・`judge_overlap_rejects_uniform_non_wilson95_z`
/// で確認する）。
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
