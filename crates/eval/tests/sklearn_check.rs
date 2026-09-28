//! sklearn 照合 8 ケース + Wilson の結合テスト（REQ-24 正常系・TASK-24.1-2・issue #60）。
//!
//! 出典: PoC-9 `03-poc/evaluation-contract/compare/sklearn_check.py`
//! （対象定義。TARGETS）と `logs/compare/sklearn_check.json`（scikit-learn
//! 1.9.1・numpy 2.5.3・2026-09-23 実行の出力値）。混同行列のセル値は
//! PoC-9 評価器の出力（`logs/compare/eval_*.json`・`logs/score_*_test.json`）
//! から転記し、sklearn の `confusion_matrix` がセル単位で一致することを
//! PoC-9 側で確認済み。
//!
//! **証拠の種別: テストハーネス**（PoC-9 で記録した sklearn の出力値を
//! 固定値として照合する。CI で sklearn は実行しない。新規の Python 依存は
//! 追加しない）。許容差は 1e-9。`None` は sklearn の `zero_division=nan` の
//! `nan` に対応する。
//!
//! sklearn_check は行ラベルを `sorted()` 順で扱うため、本ファイルでも
//! ラベルをアルファベット順で宣言し、`per_label[i]` が sklearn 記録の
//! i 行目と対応するようにする。
//!
//! **Wilson 95% 信頼区間は sklearn の照合対象外**（scikit-learn に実装が
//! 無いため）。本ファイル末尾の結合テストは PoC-9 評価器（Python・独立実装）
//! が出力した `ci95` 値との照合であり、sklearn との照合ではない
//! （`crates/eval/src/wilson.rs` のドキュメントコメント参照）。
//!
//! **照合しない項目**: multi モード（複数項目選択）の
//! `full_item_match_accuracy`・`intent_only_accuracy` は混同行列から
//! 再構成できず、Rust 評価器に multi-item モードも無いため対象外
//! （`known/multi-item`・`v1.1/extra-key` の 2 対象は `intent_only` の
//! 混同行列のみを `evaluate_single_select` に展開して照合し、`adopted_decision`
//! は検証しない。sklearn の記録が無いため）。

use fandhe_edge_eval::metrics::{ConfusionColumn, EvalRecord, Outcome, evaluate_single_select};
use fandhe_edge_eval::wilson::wilson_ci95;

const EPSILON: f64 = 1e-9;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

/// `Option<f64>` 同士を許容差 1e-9 で比較する。`None` 同士は一致、片方だけ
/// `None` なら失敗（sklearn の `zero_division=nan` を `None` として扱う）。
fn assert_opt_close(actual: Option<f64>, expected: Option<f64>, context: &str) {
    match (actual, expected) {
        (None, None) => {}
        (Some(a), Some(e)) => {
            assert!(approx_eq(a, e), "{context}: actual {a} != expected {e}")
        }
        _ => panic!("{context}: actual {actual:?} != expected {expected:?}"),
    }
}

/// 1 ラベル分の期待値（sklearn_check.json の出力値）。
struct ExpectedLabel {
    label: &'static str,
    support: u64,
    precision: Option<f64>,
    recall: Option<f64>,
    f1: Option<f64>,
}

/// 混同行列（行: 宣言順のラベル、列: 宣言順のラベル + invalid + abstain + error）
/// を `EvalRecord` の列へ展開する。`Outcome::Label` は列ごとに 1 つだけ作って
/// 参照し、レコード件数分（最大 1508 件）の `String` 確保を避ける。
///
/// テスト専用のヘルパーのため、添字アクセスは `get()` ではなく `expect` で
/// 書く（呼び出し元がテストデータの行数・列数を保証する）。
fn expand<'a>(
    labels: &'a [&'a str],
    rows: &[&[u64]],
    label_outcomes: &'a [Outcome],
    invalid: &'a Outcome,
    abstain: &'a Outcome,
    error: &'a Outcome,
) -> Vec<EvalRecord<'a>> {
    let n_labels = labels.len();
    let mut records = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        let gold = labels.get(r).expect("row count matches labels count");
        for (c, &count) in row.iter().enumerate() {
            let outcome = if c < n_labels {
                label_outcomes.get(c).expect("column within n_labels")
            } else if c == n_labels {
                invalid
            } else if c == n_labels + 1 {
                abstain
            } else if c == n_labels + 2 {
                error
            } else {
                panic!("row {r} has more columns than n_labels + 3");
            };
            for _ in 0..count {
                records.push(EvalRecord { gold, outcome });
            }
        }
    }
    records
}

/// sklearn 照合 1 対象を検証する共通処理（3.3 節の `check_case` 相当）。
///
/// `expected_adopted` を `None` にすると adopted_decision の照合を省略する
/// （multi モードの intent_only 展開では sklearn の記録が無いため）。
/// `ci95` を `Some((lo, hi))` にすると `accuracy.overall.ci95()` も照合する
/// （PoC-9 評価器の独立実装との照合。sklearn との照合ではない）。
#[allow(clippy::too_many_arguments)]
fn check_case(
    labels: &[&str],
    rows: &[&[u64]],
    expected_labels: &[ExpectedLabel],
    expected_macro_f1: Option<f64>,
    expected_overall: f64,
    expected_adopted: Option<f64>,
    expected_overall_ci95: Option<(f64, f64)>,
) {
    let label_outcomes: Vec<Outcome> = labels
        .iter()
        .map(|&label| Outcome::Label(label.to_string()))
        .collect();
    let invalid = Outcome::Invalid;
    let abstain = Outcome::Abstain;
    let error = Outcome::Error;
    let records = expand(labels, rows, &label_outcomes, &invalid, &abstain, &error);

    let metrics = evaluate_single_select(labels, &records).expect("valid sklearn-check input");

    let expected_n_total: u64 = rows.iter().flat_map(|row| row.iter()).sum();
    assert_eq!(metrics.n_total, expected_n_total);

    let n_labels = labels.len();
    for (r, row) in rows.iter().enumerate() {
        for (c, &expected_count) in row.iter().enumerate() {
            let column = if c < n_labels {
                ConfusionColumn::Label(c)
            } else if c == n_labels {
                ConfusionColumn::Invalid
            } else if c == n_labels + 1 {
                ConfusionColumn::Abstain
            } else {
                ConfusionColumn::Error
            };
            let actual = metrics
                .confusion
                .get(r, column)
                .unwrap_or_else(|| panic!("confusion[{r}][{c}] out of range"));
            assert_eq!(actual, expected_count, "confusion[{r}][{c}] mismatch");
        }
    }

    assert_eq!(metrics.per_label.len(), expected_labels.len());
    for (actual, expected) in metrics.per_label.iter().zip(expected_labels.iter()) {
        assert_eq!(actual.label, expected.label);
        assert_eq!(
            actual.support, expected.support,
            "support mismatch for {}",
            expected.label
        );
        assert_opt_close(
            actual.precision,
            expected.precision,
            &format!("precision[{}]", expected.label),
        );
        assert_opt_close(
            actual.recall,
            expected.recall,
            &format!("recall[{}]", expected.label),
        );
        assert_opt_close(actual.f1, expected.f1, &format!("f1[{}]", expected.label));
    }

    assert_opt_close(metrics.macro_f1.value(), expected_macro_f1, "macro_f1");
    assert!(
        approx_eq(metrics.accuracy.overall.value(), expected_overall),
        "overall accuracy: {} != {}",
        metrics.accuracy.overall.value(),
        expected_overall
    );

    if let Some(expected_adopted_value) = expected_adopted {
        let adopted = metrics
            .accuracy
            .adopted_decision
            .expect("adopted_decision defined (no all-abstain case in sklearn_check targets)");
        assert!(
            approx_eq(adopted.value(), expected_adopted_value),
            "adopted_decision accuracy: {} != {}",
            adopted.value(),
            expected_adopted_value
        );
    }

    if let Some((expected_lo, expected_hi)) = expected_overall_ci95 {
        let interval = metrics
            .accuracy
            .overall
            .ci95()
            .expect("Ratio invariant guarantees Some");
        assert!(
            approx_eq(interval.lo(), expected_lo),
            "overall ci95 lo: {} != {}",
            interval.lo(),
            expected_lo
        );
        assert!(
            approx_eq(interval.hi(), expected_hi),
            "overall ci95 hi: {} != {}",
            interval.hi(),
            expected_hi
        );
    }
}

/// PoC-9 sklearn_check 対象 1/8: `known/single-select`（24 件。
/// `tests/known_answer.rs` と同じ既知解データを流用する。重複する対象が
/// あるのは意図的〔issue #60 実装計画〕）。
///
/// sklearn の `precision_recall_fscore_support(zero_division=nan)`・
/// `f1_score(average='macro', zero_division=nan)`・`accuracy_score`・
/// `confusion_matrix` の出力値と 1e-9 で一致する。
#[test]
fn sklearn_known_single_select() {
    let labels = ["A", "B", "C", "D"];
    let rows: [&[u64]; 4] = [
        &[3, 1, 0, 0, 0, 1, 1],
        &[1, 4, 0, 0, 1, 0, 0],
        &[0, 0, 5, 0, 1, 0, 0],
        &[2, 2, 2, 0, 0, 0, 0],
    ];
    let expected = [
        ExpectedLabel {
            label: "A",
            support: 6,
            precision: Some(0.5),
            recall: Some(0.5),
            f1: Some(0.5),
        },
        ExpectedLabel {
            label: "B",
            support: 6,
            precision: Some(4.0 / 7.0),
            recall: Some(4.0 / 6.0),
            f1: Some(8.0 / 13.0),
        },
        ExpectedLabel {
            label: "C",
            support: 6,
            precision: Some(5.0 / 7.0),
            recall: Some(5.0 / 6.0),
            f1: Some(10.0 / 13.0),
        },
        ExpectedLabel {
            label: "D",
            support: 6,
            precision: None,
            recall: Some(0.0),
            f1: Some(0.0),
        },
    ];
    check_case(
        &labels,
        &rows,
        &expected,
        Some(0.47115384615384615),
        0.5,
        Some(12.0 / 23.0),
        Some((0.31427131627763083, 0.6857286837223692)),
    );
}

/// PoC-9 sklearn_check 対象 2/8: `known/multi-item`（intent_only の混同行列を
/// 単一選択として展開）。`full_item_match_accuracy`・`intent_only_accuracy`
/// は照合対象外（multi-item モードが Rust 評価器に無いため）。
#[test]
fn sklearn_known_multi_item_intent_only() {
    let labels = ["X", "Y"];
    let rows: [&[u64]; 2] = [&[4, 0, 1, 0, 0], &[1, 1, 1, 1, 1]];
    let expected = [
        ExpectedLabel {
            label: "X",
            support: 5,
            precision: Some(0.8),
            recall: Some(0.8),
            f1: Some(0.8),
        },
        ExpectedLabel {
            label: "Y",
            support: 5,
            precision: Some(1.0),
            recall: Some(0.2),
            f1: Some(1.0 / 3.0),
        },
    ];
    check_case(
        &labels,
        &rows,
        &expected,
        Some(0.5666666666666667),
        0.5,
        None,
        None,
    );
}

/// PoC-9 sklearn_check 対象 3/8: `known/baselines (majority)`。
#[test]
fn sklearn_known_baselines_majority() {
    let labels = ["A", "B", "C", "D"];
    let row: &[u64] = &[6, 0, 0, 0, 0, 0, 0];
    let rows: [&[u64]; 4] = [row, row, row, row];
    let expected = [
        ExpectedLabel {
            label: "A",
            support: 6,
            precision: Some(0.25),
            recall: Some(1.0),
            f1: Some(0.4),
        },
        ExpectedLabel {
            label: "B",
            support: 6,
            precision: None,
            recall: Some(0.0),
            f1: Some(0.0),
        },
        ExpectedLabel {
            label: "C",
            support: 6,
            precision: None,
            recall: Some(0.0),
            f1: Some(0.0),
        },
        ExpectedLabel {
            label: "D",
            support: 6,
            precision: None,
            recall: Some(0.0),
            f1: Some(0.0),
        },
    ];
    check_case(&labels, &rows, &expected, Some(0.1), 0.25, Some(0.25), None);
}

/// PoC-9 sklearn_check 対象 4/8: `known/baselines (oracle)`。
#[test]
fn sklearn_known_baselines_oracle() {
    let labels = ["A", "B", "C", "D"];
    let rows: [&[u64]; 4] = [
        &[6, 0, 0, 0, 0, 0, 0],
        &[0, 6, 0, 0, 0, 0, 0],
        &[0, 0, 6, 0, 0, 0, 0],
        &[0, 0, 0, 6, 0, 0, 0],
    ];
    let expected = ["A", "B", "C", "D"].map(|label| ExpectedLabel {
        label,
        support: 6,
        precision: Some(1.0),
        recall: Some(1.0),
        f1: Some(1.0),
    });
    check_case(&labels, &rows, &expected, Some(1.0), 1.0, Some(1.0), None);
}

/// PoC-9 sklearn_check 対象 5/8: `v1.1/prediction-defects`。
#[test]
fn sklearn_v1_1_prediction_defects() {
    let labels = ["A", "B"];
    let rows: [&[u64]; 2] = [&[2, 0, 0, 0, 1], &[1, 0, 1, 0, 1]];
    let expected = [
        ExpectedLabel {
            label: "A",
            support: 3,
            precision: Some(2.0 / 3.0),
            recall: Some(2.0 / 3.0),
            f1: Some(2.0 / 3.0),
        },
        ExpectedLabel {
            label: "B",
            support: 3,
            precision: None,
            recall: Some(0.0),
            f1: Some(0.0),
        },
    ];
    check_case(
        &labels,
        &rows,
        &expected,
        Some(1.0 / 3.0),
        1.0 / 3.0,
        Some(1.0 / 3.0),
        Some((0.09676933255921683, 0.7000116786584712)),
    );
}

/// PoC-9 sklearn_check 対象 6/8: `v1.1/extra-key`（intent_only の混同行列を
/// 単一選択として展開）。`full_item_match_accuracy` は偶然 0.5 で一致するが、
/// 検証したことにはしない（issue #60 実装計画）。
#[test]
fn sklearn_v1_1_extra_key_intent_only() {
    let labels = ["X", "Y"];
    let rows: [&[u64]; 2] = [&[0, 0, 1, 0, 0], &[0, 1, 0, 0, 0]];
    let expected = [
        ExpectedLabel {
            label: "X",
            support: 1,
            precision: None,
            recall: Some(0.0),
            f1: Some(0.0),
        },
        ExpectedLabel {
            label: "Y",
            support: 1,
            precision: Some(1.0),
            recall: Some(1.0),
            f1: Some(1.0),
        },
    ];
    check_case(&labels, &rows, &expected, Some(0.5), 0.5, None, None);
}

/// PoC-9 sklearn_check 対象 7/8: 参照ログ `preds_majority_test`
/// （9 intent・1508 件）。全件を set_reminder と予測する下限基準。
#[test]
fn sklearn_reference_preds_majority_test() {
    let labels = [
        "complete_task",
        "create_note",
        "create_task",
        "delete_task",
        "list_tasks",
        "none",
        "schedule_event",
        "search_notes",
        "set_reminder",
    ];
    // 列順は labels と同じ 9 列 + invalid + abstain + error。
    // 常に set_reminder（列 8）へ予測するため、各行の support が列 8 に集まる。
    let complete_task: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 168, 0, 0, 0];
    let create_note: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 108, 0, 0, 0];
    let create_task: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0];
    let delete_task: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 170, 0, 0, 0];
    let list_tasks: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 281, 0, 0, 0];
    let none: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 164, 0, 0, 0];
    let schedule_event: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 137, 0, 0, 0];
    let search_notes: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 246, 0, 0, 0];
    let set_reminder: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 106, 0, 0, 0];
    let rows: [&[u64]; 9] = [
        complete_task,
        create_note,
        create_task,
        delete_task,
        list_tasks,
        none,
        schedule_event,
        search_notes,
        set_reminder,
    ];
    let supports: [u64; 9] = [168, 108, 128, 170, 281, 164, 137, 246, 106];
    let expected: Vec<ExpectedLabel> = labels
        .iter()
        .zip(supports.iter())
        .map(|(&label, &support)| {
            if label == "set_reminder" {
                ExpectedLabel {
                    label,
                    support,
                    precision: Some(0.07029177718832891),
                    recall: Some(1.0),
                    f1: Some(0.13135068153655513),
                }
            } else {
                ExpectedLabel {
                    label,
                    support,
                    precision: None,
                    recall: Some(0.0),
                    f1: Some(0.0),
                }
            }
        })
        .collect();
    check_case(
        &labels,
        &rows,
        &expected,
        Some(0.014594520170728348),
        0.07029177718832891,
        Some(0.07029177718832891),
        Some((0.058451183561754626, 0.0843161539688438)),
    );
}

/// PoC-9 sklearn_check 対象 8/8: 参照ログ `preds_tfidf_linear_test`
/// （9 intent・1508 件。候補 C1 の推論結果）。
///
/// 混同行列は sklearn_check.json の集計値（各ラベルの precision・recall・
/// support）から一意に再構成できる（各セルの導出は issue #60 実装計画
/// 「3.3」節を参照。列和・行和が全ラベルの predicted_count・support と
/// 一致することを確認済み）。
#[test]
fn sklearn_reference_preds_tfidf_linear_test() {
    let labels = [
        "complete_task",
        "create_note",
        "create_task",
        "delete_task",
        "list_tasks",
        "none",
        "schedule_event",
        "search_notes",
        "set_reminder",
    ];
    let complete_task: &[u64] = &[90, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let create_note: &[u64] = &[0, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let create_task: &[u64] = &[0, 0, 127, 1, 0, 0, 0, 0, 0, 0, 0, 0];
    let delete_task: &[u64] = &[0, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 0];
    let list_tasks: &[u64] = &[0, 270, 2, 0, 9, 0, 0, 0, 0, 0, 0, 0];
    let none: &[u64] = &[0, 164, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let schedule_event: &[u64] = &[0, 0, 0, 0, 0, 0, 137, 0, 0, 0, 0, 0];
    let search_notes: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 246, 0, 0, 0, 0];
    let set_reminder: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 106, 0, 0, 0];
    let rows: [&[u64]; 9] = [
        complete_task,
        create_note,
        create_task,
        delete_task,
        list_tasks,
        none,
        schedule_event,
        search_notes,
        set_reminder,
    ];
    let expected = [
        ExpectedLabel {
            label: "complete_task",
            support: 168,
            precision: Some(1.0),
            recall: Some(0.5357142857142857),
            f1: Some(0.6976744186046512),
        },
        ExpectedLabel {
            label: "create_note",
            support: 108,
            precision: Some(0.17419354838709677),
            recall: Some(1.0),
            f1: Some(0.2967032967032967),
        },
        ExpectedLabel {
            label: "create_task",
            support: 128,
            precision: Some(0.9844961240310077),
            recall: Some(0.9921875),
            f1: Some(0.9883268482490273),
        },
        ExpectedLabel {
            label: "delete_task",
            support: 170,
            precision: Some(0.9941520467836257),
            recall: Some(1.0),
            f1: Some(0.9970674486803519),
        },
        ExpectedLabel {
            label: "list_tasks",
            support: 281,
            precision: Some(1.0),
            recall: Some(0.03202846975088968),
            f1: Some(0.06206896551724138),
        },
        ExpectedLabel {
            label: "none",
            support: 164,
            precision: None,
            recall: Some(0.0),
            f1: Some(0.0),
        },
        ExpectedLabel {
            label: "schedule_event",
            support: 137,
            precision: Some(1.0),
            recall: Some(1.0),
            f1: Some(1.0),
        },
        ExpectedLabel {
            label: "search_notes",
            support: 246,
            precision: Some(1.0),
            recall: Some(1.0),
            f1: Some(1.0),
        },
        ExpectedLabel {
            label: "set_reminder",
            support: 106,
            precision: Some(1.0),
            recall: Some(1.0),
            f1: Some(1.0),
        },
    ];
    check_case(
        &labels,
        &rows,
        &expected,
        Some(0.671315664194952),
        0.6584880636604774,
        Some(0.6584880636604774),
        Some((0.6341774435857807, 0.681993245195372)),
    );
}

/// REQ-24・REQ-26・TASK-24.1-2: Wilson 95% 信頼区間が PoC-9 評価器
/// （Python・独立実装）の `ci95` 出力値と 1e-9 で一致する。
///
/// **sklearn との照合ではない**（scikit-learn は Wilson 区間を実装しない）。
/// 上記 8 ケースのうち `accuracy.overall.ci95()` は `check_case` 内でも
/// 照合済みだが、本テストでは PoC-9 の全対象（`adopted_decision`・multi の
/// `full_item_match` を含む）をまとめて `wilson_ci95` へ直接照合する。
#[test]
fn wilson_ci_matches_poc9_independent_implementation() {
    let cases: [(u64, u64, f64, f64); 9] = [
        // known/single-select: overall 12/24, adopted 12/23。
        (12, 24, 0.31427131627763083, 0.6857286837223692),
        (12, 23, 0.3296242986421849, 0.7076313046005427),
        // known/baselines (majority): 6/24。
        (6, 24, 0.1199920347934233, 0.4489982531210644),
        // known/baselines (oracle): 24/24。
        (24, 24, 0.8620194241710247, 1.0),
        // v1.1/prediction-defects: 2/6。
        (2, 6, 0.09676933255921683, 0.7000116786584712),
        // 参照ログ preds_majority_test: 106/1508。
        (106, 1508, 0.058451183561754626, 0.0843161539688438),
        // 参照ログ preds_tfidf_linear_test: 993/1508。
        (993, 1508, 0.6341774435857807, 0.681993245195372),
        // multi の full_item_match（multi-item モード自体は対象外だが、
        // Wilson の式の検証値としては使える。issue #60 実装計画）。
        (3, 10, 0.10778928748621183, 0.6032267800204347),
        (1, 2, 0.09452865480086614, 0.9054713451991339),
    ];
    for (correct, n, expected_lo, expected_hi) in cases {
        let interval = wilson_ci95(correct, n).expect("n > 0 in all sklearn_check targets");
        assert!(
            approx_eq(interval.lo(), expected_lo),
            "wilson_ci95({correct}, {n}).lo(): {} != {}",
            interval.lo(),
            expected_lo
        );
        assert!(
            approx_eq(interval.hi(), expected_hi),
            "wilson_ci95({correct}, {n}).hi(): {} != {}",
            interval.hi(),
            expected_hi
        );
    }
}
