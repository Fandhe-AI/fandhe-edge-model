//! ラベル集合相違時の前提明記の結合テスト（REQ-26 異常系・TASK-26.2・
//! issue #102）。
//!
//! [`fandhe_edge_eval::regression::regression_report`] を、PoC-19
//! （`03-poc/model-lifecycle/`。`definitions/label_order_*.json` と
//! `jobs/rebuild/result.json`）のラベル順・件数の転記で確認する。
//!
//! 証拠の種別: テストハーネス。ラベル ID と件数は PoC-19 からの転記であり、
//! 本テストによる新しい実測ではない（`docs/spec` は読まない）。正誤列は乱数を
//! 使わず決定的に組み立てる。

use fandhe_edge_eval::regression::{ComparisonPremise, regression_counts, regression_report};

/// PoC-19 の 9 ラベル（旧モデル）。
const M9: [&str; 9] = [
    "tier-s__low",
    "tier-m__low",
    "tier-m__medium",
    "tier-m__high",
    "tier-l__low",
    "tier-l__medium",
    "tier-l__high",
    "tier-xl__medium",
    "tier-xl__high",
];

/// M9 から `tier-xl__high` を除いた 8 ラベル（削除パターン）。
const M8_RM: [&str; 8] = [
    "tier-s__low",
    "tier-m__low",
    "tier-m__medium",
    "tier-m__high",
    "tier-l__low",
    "tier-l__medium",
    "tier-l__high",
    "tier-xl__medium",
];

/// `tier-l__medium`・`tier-l__high` を `tier-l__midhigh` に統合した 8 ラベル。
const M8_MERGE: [&str; 8] = [
    "tier-s__low",
    "tier-m__low",
    "tier-m__medium",
    "tier-m__high",
    "tier-l__low",
    "tier-l__midhigh",
    "tier-xl__medium",
    "tier-xl__high",
];

/// 「両方正解 / 回帰 / 改善 / 両方不正解」の件数から正誤列を組み立てる。
fn build(both: usize, c2i: usize, i2c: usize, wrong: usize) -> (Vec<bool>, Vec<bool>) {
    let mut prev = Vec::new();
    let mut cur = Vec::new();
    for _ in 0..both {
        prev.push(true);
        cur.push(true);
    }
    for _ in 0..c2i {
        prev.push(true);
        cur.push(false);
    }
    for _ in 0..i2c {
        prev.push(false);
        cur.push(true);
    }
    for _ in 0..wrong {
        prev.push(false);
        cur.push(false);
    }
    (prev, cur)
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

/// P3（統合）: 旧 9 ラベル → 新 8 ラベル。件数 170/44/45/391（n=650）に
/// 前提相違が付く。
#[test]
fn p3_merge_reports_label_set_differs() {
    let (prev, cur) = build(170, 44, 45, 391);
    let report = regression_report(&M9, &M8_MERGE, &prev, &cur).unwrap();
    assert_eq!(report.counts().n(), 650);
    assert_eq!(report.counts().correct_to_incorrect(), 44);
    assert_eq!(report.counts().incorrect_to_correct(), 45);
    assert_eq!(
        report.premise(),
        &ComparisonPremise::LabelSetDiffers {
            removed: strings(&["tier-l__medium", "tier-l__high"]),
            added: strings(&["tier-l__midhigh"]),
        }
    );
    assert_eq!(report.premise().as_str(), "label_set_differs");
    assert!(report.premise().note().is_some());
}

/// P1（追加）: 旧 8 ラベル → 新 9 ラベル。件数 100/58/22/351（n=531）。
#[test]
fn p1_addition_reports_added_label() {
    let (prev, cur) = build(100, 58, 22, 351);
    let report = regression_report(&M8_RM, &M9, &prev, &cur).unwrap();
    assert_eq!(report.counts().n(), 531);
    assert_eq!(
        report.premise(),
        &ComparisonPremise::LabelSetDiffers {
            removed: vec![],
            added: strings(&["tier-xl__high"]),
        }
    );
}

/// P2（削除）: 旧 9 ラベル → 新 8 ラベル。件数 100/22/58/351。
#[test]
fn p2_removal_reports_removed_label() {
    let (prev, cur) = build(100, 22, 58, 351);
    let report = regression_report(&M9, &M8_RM, &prev, &cur).unwrap();
    assert_eq!(report.counts().correct_to_incorrect(), 22);
    assert_eq!(report.counts().incorrect_to_correct(), 58);
    assert_eq!(
        report.premise(),
        &ComparisonPremise::LabelSetDiffers {
            removed: strings(&["tier-xl__high"]),
            added: vec![],
        }
    );
}

/// 同一集合（TASK-26.1 の受入データ 2/2/1/2・n=7）: 前提付きにしても件数は
/// `regression_counts` と完全一致し、前提は `SameLabelSet`。
#[test]
fn same_label_set_keeps_counts_identical() {
    let (prev, cur) = build(2, 2, 1, 2);
    let report = regression_report(&M9, &M9, &prev, &cur).unwrap();
    assert_eq!(report.counts(), &regression_counts(&prev, &cur).unwrap());
    assert_eq!(report.counts().both_correct(), 2);
    assert_eq!(report.counts().correct_to_incorrect(), 2);
    assert_eq!(report.counts().incorrect_to_correct(), 1);
    assert_eq!(report.counts().both_wrong(), 2);
    assert_eq!(report.premise(), &ComparisonPremise::SameLabelSet);
    assert_eq!(report.premise().note(), None);
}
