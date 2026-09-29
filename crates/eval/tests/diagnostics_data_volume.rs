//! データ量水準別の効果報告の結合テスト（REQ-29 境界値・TASK-29.3・issue #110。
//! 証拠種別: テストハーネス。PoC-11 の傾向の定義確認であり、精度の期待値の出典ではない）。

use fandhe_edge_eval::diagnostics::{
    DataVolumeLevel, DiagnosticReport, InputKey, RowCountEffect, StatsRow, basic_stats,
    diagnostic_report, note_label_count_change,
};
use fandhe_edge_eval::metrics::{EvalRecord, Outcome, evaluate_single_select};

const LABELS: [&str; 2] = ["secret-label", "B"];

/// 学習 `train_n` 行・評価 10 行の実レポートを作る。
fn report_with(train_n: usize, reverse: bool) -> DiagnosticReport {
    let inputs: Vec<String> = (0..train_n).map(|i| format!("t{i}")).collect();
    let mut train_rows: Vec<StatsRow<'_>> = inputs
        .iter()
        .enumerate()
        .map(|(i, s)| StatsRow {
            input: s,
            label: LABELS[i % 2],
        })
        .collect();
    if reverse {
        train_rows.reverse();
    }
    let eval_inputs: Vec<String> = (0..10).map(|i| format!("e{i}")).collect();
    let eval_rows: Vec<StatsRow<'_>> = eval_inputs
        .iter()
        .enumerate()
        .map(|(i, s)| StatsRow {
            input: s,
            label: LABELS[i % 2],
        })
        .collect();
    let train = basic_stats(&LABELS, &train_rows, InputKey::ByteExact).unwrap();
    let eval = basic_stats(&LABELS, &eval_rows, InputKey::ByteExact).unwrap();
    let outcomes: Vec<Outcome> = (0..10).map(|_| Outcome::Label("B".to_string())).collect();
    let records: Vec<EvalRecord<'_>> = eval_rows
        .iter()
        .zip(outcomes.iter())
        .map(|(r, o)| EvalRecord {
            gold: r.label,
            outcome: o,
        })
        .collect();
    let m = evaluate_single_select(&LABELS, &records).unwrap();
    diagnostic_report(train, eval, &m, 5).unwrap()
}

#[test]
fn levels_follow_train_row_count() {
    let cases = [
        (
            99,
            DataVolumeLevel::Below100,
            "below_100",
            RowCountEffect::Large,
            "large",
        ),
        (
            100,
            DataVolumeLevel::From100To3000,
            "from_100_to_3000",
            RowCountEffect::Large,
            "large",
        ),
        (
            2999,
            DataVolumeLevel::From100To3000,
            "from_100_to_3000",
            RowCountEffect::Large,
            "large",
        ),
        (
            3000,
            DataVolumeLevel::From3000,
            "from_3000",
            RowCountEffect::Plateau,
            "plateau",
        ),
    ];
    for (n, level, level_id, effect, effect_id) in cases {
        let report = report_with(n, false);
        let v = report.data_volume();
        assert_eq!(v.train_rows(), n as u64);
        assert_eq!(v.level(), level, "n={n}");
        assert_eq!(v.level().as_str(), level_id);
        assert_eq!(v.effect(), effect);
        assert_eq!(v.effect().as_str(), effect_id);
    }
}

#[test]
fn classification_uses_train_rows_not_eval_rows() {
    let report = report_with(3000, false);
    assert_eq!(report.eval().n_rows, 10);
    assert_eq!(report.data_volume().level(), DataVolumeLevel::From3000);
}

#[test]
fn note_has_no_label_values() {
    for n in [99, 100, 3000] {
        let report = report_with(n, false);
        assert!(!report.data_volume().note().contains("secret"));
    }
}

#[test]
fn label_count_note_does_not_change_data_volume() {
    let report = report_with(100, false);
    let before = report.data_volume().clone();
    let noted = note_label_count_change(report, &["A", "B", "C"]).unwrap();
    assert_eq!(noted.limitations().len(), 1);
    assert_eq!(noted.data_volume(), &before);
}

#[test]
fn reversing_train_rows_gives_same_result() {
    assert_eq!(
        report_with(150, false).data_volume(),
        report_with(150, true).data_volume()
    );
}
