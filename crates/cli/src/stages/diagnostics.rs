//! `evaluate` の診断レポート（stdout の `diagnostics`。REQ-29・REQ-27・REQ-39・TASK-29.1〜29.3・#492）。
//!
//! 計算はすべて評価器 `fandhe_edge_eval::diagnostics` に委ね、ここでは呼び出しと出力型への写しだけを行う
//! （再実装しない）。
//!
//! - [`prepare_stats`]: train 分割（validation・test を含まない。p95 計測・下限基準と同じ範囲）と凍結評価データの
//!   全行の基礎統計（`InputKey::ByteExact`）。`evaluate` が**適用権を取る前**に呼ぶ。上限超過は
//!   `limit_exceeded`（20）、それ以外の失敗は `runtime_error`（70）で、どちらも適用権を使わない
//! - [`build`]: 同じ 1 回の適用の評価結果（混同行列）から上位 [`DEFAULT_CONFUSABLE_PAIRS_TOP_K`] 件の
//!   混同しやすい組を求め、`--previous-project-dir` のときだけ旧定義の選択肢とのラベル数の変化を注記する。
//!   評価記録の書き込みより前に呼び、整合検査の失敗は `runtime_error`
//!
//! 出すのはラベル ID・件数・評価器の固定語彙と固定英文だけで、入力本文は出さない（security.md）。
//! 評価記録には入れず、終了コード・合否・`package` の照合に使わない。

use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::{
    EvaluateBasicStats, EvaluateConfusablePair, EvaluateDataVolume, EvaluateDiagnostics,
    EvaluateLabelCount, EvaluateLimitation,
};
use fandhe_edge_data::inspect::ValidRecord;
use fandhe_edge_eval::diagnostics::{
    BasicStats, DiagnosticLimitation, DiagnosticsError, InputKey, StatsRow, basic_stats,
    diagnostic_report, note_label_count_change,
};
use fandhe_edge_eval::metrics::SingleSelectMetrics;

use crate::project::{fail, runtime};

/// 混同しやすい組の件数（PoC-11 と同じ上位 10。入出力契約。#492）。
pub(super) const DEFAULT_CONFUSABLE_PAIRS_TOP_K: usize = 10;

/// 適用前に求めた基礎統計（train 分割・凍結評価データ）。
pub(super) struct PreparedStats {
    train: BasicStats,
    eval: BasicStats,
}

/// 基礎統計を求める（適用権を取る前に呼ぶ）。`labels` は定義の宣言順の選択肢 ID。
///
/// # Errors
/// 行数・入力長・総バイト数の上限超過は `limit_exceeded`、その他は `runtime_error`。
pub(super) fn prepare_stats<'a>(
    labels: &[&str],
    train: impl Iterator<Item = &'a ValidRecord>,
    eval: &'a [ValidRecord],
) -> Result<PreparedStats, ErrorReport> {
    let stats = |rows: Vec<StatsRow<'_>>| {
        basic_stats(labels, &rows, InputKey::ByteExact).map_err(|e| match e {
            DiagnosticsError::TooManyRows { .. }
            | DiagnosticsError::InputTooLong { .. }
            | DiagnosticsError::TotalInputTooLarge { .. } => {
                fail(ExitCode::LimitExceeded, "data exceeds diagnostics limits")
            }
            _ => runtime("cannot compute diagnostics"),
        })
    };
    let row = |r: &'a ValidRecord| StatsRow {
        input: &r.input,
        label: &r.label_id,
    };
    Ok(PreparedStats {
        train: stats(train.map(row).collect())?,
        eval: stats(eval.iter().map(row).collect())?,
    })
}

/// 適用結果の評価指標から診断レポートを組み立てる（評価記録の書き込みより前に呼ぶ）。
///
/// `previous_labels` は `--previous-project-dir` の旧定義の選択肢 ID（宣言順）。無ければ `limitations` は空。
///
/// # Errors
/// 評価器の整合検査（ラベル並び・評価件数・support の一致）の失敗は `runtime_error`。
pub(super) fn build(
    stats: PreparedStats,
    metrics: &SingleSelectMetrics,
    previous_labels: Option<&[&str]>,
) -> Result<EvaluateDiagnostics, ErrorReport> {
    let failed = |_| runtime("cannot build diagnostics");
    let mut report = diagnostic_report(
        stats.train,
        stats.eval,
        metrics,
        DEFAULT_CONFUSABLE_PAIRS_TOP_K,
    )
    .map_err(failed)?;
    if let Some(previous) = previous_labels {
        report = note_label_count_change(report, previous).map_err(failed)?;
    }
    let limitations = report
        .limitations()
        .iter()
        .map(|l| match l {
            DiagnosticLimitation::LabelCountChanged { previous, current } => {
                Ok(EvaluateLimitation {
                    kind: l.as_str(),
                    previous: *previous,
                    current: *current,
                    note: l.note(),
                })
            }
            // 評価器に限界の種類が増えたら黙って落とさず止める（fail-closed）。
            _ => Err(runtime("cannot build diagnostics")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let volume = report.data_volume();
    Ok(EvaluateDiagnostics {
        train: stats_output(report.train()),
        eval: stats_output(report.eval()),
        confusable_pairs: report
            .confusable_pairs()
            .iter()
            .map(|p| EvaluateConfusablePair {
                gold: p.gold.clone(),
                predicted: p.predicted.clone(),
                count: p.count,
                gold_support: p.gold_support,
            })
            .collect(),
        limitations,
        data_volume: EvaluateDataVolume {
            train_rows: volume.train_rows(),
            level: volume.level().as_str(),
            effect: volume.effect().as_str(),
            note: volume.note(),
        },
    })
}

fn stats_output(s: &BasicStats) -> EvaluateBasicStats {
    EvaluateBasicStats {
        n_rows: s.n_rows,
        unique_inputs: s.unique_inputs,
        unique_labels: s.unique_labels,
        label_counts: s
            .label_counts
            .iter()
            .map(|c| EvaluateLabelCount {
                label: c.label.clone(),
                count: c.count,
            })
            .collect(),
        min_label_count: s.min_label_count,
        min_labels: s.min_labels.clone(),
        unobserved_labels: s.unobserved_labels.clone(),
        input_key_rule: s.input_key_rule,
    }
}

#[cfg(test)]
mod tests {
    use fandhe_edge_eval::diagnostics::{MAX_STATS_INPUT_BYTES, data_volume_report};
    use fandhe_edge_eval::metrics::{EvalRecord, Outcome, evaluate_single_select};

    use super::*;

    const LABELS: [&str; 3] = ["a", "b", "c"];

    fn record(id: &str, input: &str, label: &str) -> ValidRecord {
        ValidRecord {
            line: 1,
            id: id.to_string(),
            input: input.to_string(),
            label_id: label.to_string(),
            output_key: String::new(),
            output_original: String::new(),
            tags: None,
            group_id: None,
        }
    }

    fn counts(c: [u64; 3]) -> Vec<EvaluateLabelCount> {
        LABELS
            .iter()
            .zip(c)
            .map(|(l, count)| EvaluateLabelCount {
                label: l.to_string(),
                count,
            })
            .collect()
    }

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// 1 行だけの評価（a を a と予測）の評価指標。
    fn one_correct_metrics() -> SingleSelectMetrics {
        let outcome = Outcome::Label("a".to_string());
        evaluate_single_select(
            &LABELS,
            &[EvalRecord {
                gold: "a",
                outcome: &outcome,
            }],
        )
        .expect("metrics")
    }

    /// 固定 fixture（train 4 行・評価 5 行、本文は `secret-*`）を 1 回適用した評価指標から診断を作る。
    /// 予測: e1・e2 は a→b、e3 は b→a、e4 は正解、e5 は c→a。
    fn fixture(previous_labels: Option<&[&str]>) -> Result<EvaluateDiagnostics, ErrorReport> {
        let train = [
            record("t1", "secret-x", "a"),
            record("t2", "secret-x", "a"),
            record("t3", "secret-y", "a"),
            record("t4", "secret-z", "b"),
        ];
        let eval = [
            record("e1", "secret-p", "a"),
            record("e2", "secret-q", "a"),
            record("e3", "secret-r", "b"),
            record("e4", "secret-s", "b"),
            record("e5", "secret-s", "c"),
        ];
        let stats = prepare_stats(&LABELS, train.iter(), &eval)?;
        let outcomes: Vec<Outcome> = ["b", "b", "a", "b", "a"]
            .iter()
            .map(|p| Outcome::Label(p.to_string()))
            .collect();
        let records: Vec<EvalRecord<'_>> = eval
            .iter()
            .zip(&outcomes)
            .map(|(r, outcome)| EvalRecord {
                gold: &r.label_id,
                outcome,
            })
            .collect();
        let metrics = evaluate_single_select(&LABELS, &records).expect("metrics");
        build(stats, &metrics, previous_labels)
    }

    /// REQ-29・TASK-29.1・29.3・#492: 基礎統計・混同しやすい組・データ量水準が具体値で一致する。
    /// `min_label_count` は観測ラベルだけの最小（未出現 c の 0 にならない）。組は件数降順、同数は gold の
    /// 宣言順（b→a が c→a より先）。`--previous-project-dir` が無ければ `limitations` は空。
    #[test]
    fn req29_issue492_diagnostics_values_are_exact() {
        let d = fixture(None).expect("diagnostics");
        let expected = EvaluateDiagnostics {
            train: EvaluateBasicStats {
                n_rows: 4,
                unique_inputs: 3,
                unique_labels: 2,
                label_counts: counts([3, 1, 0]),
                min_label_count: Some(1),
                min_labels: strings(&["b"]),
                unobserved_labels: strings(&["c"]),
                input_key_rule: "byte_exact",
            },
            eval: EvaluateBasicStats {
                n_rows: 5,
                unique_inputs: 4,
                unique_labels: 3,
                label_counts: counts([2, 2, 1]),
                min_label_count: Some(1),
                min_labels: strings(&["c"]),
                unobserved_labels: vec![],
                input_key_rule: "byte_exact",
            },
            confusable_pairs: [("a", "b", 2, 2), ("b", "a", 1, 2), ("c", "a", 1, 1)]
                .iter()
                .map(
                    |&(gold, predicted, count, gold_support)| EvaluateConfusablePair {
                        gold: gold.to_string(),
                        predicted: predicted.to_string(),
                        count,
                        gold_support,
                    },
                )
                .collect(),
            limitations: vec![],
            data_volume: EvaluateDataVolume {
                train_rows: 4,
                level: "below_100",
                effect: "large",
                note: "training row count is below 100; row count had a large effect on accuracy in the PoC-11 learning curve (tendency only, not a guarantee; not used for pass/fail)",
            },
        };
        assert_eq!(d, expected);
        // 入力本文は出力型のどこにも入らない（security.md）。
        assert!(!format!("{d:?}").contains("secret"));
    }

    /// REQ-29・TASK-29.2・#492: 旧定義の選択肢数が異なるときだけ `label_count_changed` を注記し、同数なら空。
    #[test]
    fn req29_issue492_limitations_only_when_label_count_differs() {
        let changed = fixture(Some(&["a", "b"])).expect("diagnostics");
        assert_eq!(
            changed.limitations,
            vec![EvaluateLimitation {
                kind: "label_count_changed",
                previous: 2,
                current: 3,
                note: "label count differs between previous and current label sets; accuracy is not directly comparable because the output definition changed",
            }]
        );
        let same = fixture(Some(&["x", "y", "z"])).expect("diagnostics");
        assert_eq!(same.limitations, vec![]);
    }

    /// REQ-29・TASK-29.3・#492: train 行数 99/100/2999/3000 の境界で水準と傾向が切り替わる。
    #[test]
    fn req29_issue492_data_volume_boundaries() {
        for (n, level, effect) in [
            (99, "below_100", "large"),
            (100, "from_100_to_3000", "large"),
            (2999, "from_100_to_3000", "large"),
            (3000, "from_3000", "plateau"),
        ] {
            let train: Vec<ValidRecord> = (0..n)
                .map(|i| record(&format!("t{i}"), &format!("i{i}"), "a"))
                .collect();
            let eval = [record("e1", "p", "a")];
            let stats = prepare_stats(&LABELS, train.iter(), &eval).expect("stats");
            let d = build(stats, &one_correct_metrics(), None).expect("diagnostics");
            assert_eq!(
                (
                    d.data_volume.train_rows,
                    d.data_volume.level,
                    d.data_volume.effect
                ),
                (n, level, effect)
            );
            assert_eq!(d.data_volume.note, data_volume_report(n).note());
            assert_eq!(d.confusable_pairs, vec![]);
        }
    }

    /// REQ-39・#492: 基礎統計の上限超過（1 行の入力長）は `limit_exceeded`、その他の失敗（train 分割が空）は
    /// `runtime_error`。どちらも適用権を取る前に呼ばれる。
    #[test]
    fn req39_issue492_stats_failures_map_to_exit_codes() {
        let eval = [record("e1", "p", "a")];
        let long = [record("t1", &"x".repeat(MAX_STATS_INPUT_BYTES + 1), "a")];
        let err = prepare_stats(&LABELS, long.iter(), &eval)
            .err()
            .expect("too long");
        assert_eq!(
            (err.code, err.message.as_str()),
            (ExitCode::LimitExceeded, "data exceeds diagnostics limits")
        );
        let err = prepare_stats(&LABELS, [].iter(), &eval)
            .err()
            .expect("empty");
        assert_eq!(
            (err.code, err.message.as_str()),
            (ExitCode::RuntimeError, "cannot compute diagnostics")
        );
    }

    /// REQ-27・#492: 評価結果と基礎統計が食い違う（別の評価データの統計）なら記録前に `runtime_error`。
    #[test]
    fn req27_issue492_inconsistent_metrics_is_runtime_error() {
        let train = [record("t1", "x", "a")];
        let eval = [record("e1", "p", "a"), record("e2", "q", "b")];
        let stats = prepare_stats(&LABELS, train.iter(), &eval).expect("stats");
        let err = build(stats, &one_correct_metrics(), None).expect_err("mismatch");
        assert_eq!(
            (err.code, err.message.as_str()),
            (ExitCode::RuntimeError, "cannot build diagnostics")
        );
    }
}
