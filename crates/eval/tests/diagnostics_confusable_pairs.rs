//! 混同しやすいラベルの組とレポート統合の結合テスト（REQ-29 正常系・TASK-29.1-2・
//! issue #108。証拠種別: テストハーネス。手組みの合成データの具体値で検証する）。

use fandhe_edge_eval::diagnostics::{
    ConfusablePair, DatasetSide, DiagnosticsError, InputKey, MAX_CONFUSABLE_PAIRS, StatsRow,
    basic_stats, confusable_pairs, diagnostic_report,
};
use fandhe_edge_eval::metrics::{EvalRecord, Outcome, SingleSelectMetrics, evaluate_single_select};

const LABELS: [&str; 3] = ["A", "B", "C"];

fn lab(s: &str) -> Outcome {
    Outcome::Label(s.to_string())
}

/// (gold, outcome) の並びから評価結果を作る。
fn metrics_of(pairs: &[(&str, Outcome)]) -> SingleSelectMetrics {
    let records: Vec<EvalRecord<'_>> = pairs
        .iter()
        .map(|(g, o)| EvalRecord {
            gold: g,
            outcome: o,
        })
        .collect();
    evaluate_single_select(&LABELS, &records).unwrap()
}

/// A の support は 5（誤り 3 + 正解 2）。A→B=3, C→B=2, B→A=1。
fn sample_pairs() -> Vec<(&'static str, Outcome)> {
    vec![
        ("A", lab("B")),
        ("A", lab("B")),
        ("A", lab("B")),
        ("A", lab("A")),
        ("A", lab("A")),
        ("B", lab("A")),
        ("B", lab("B")),
        ("C", lab("B")),
        ("C", lab("B")),
        ("C", lab("C")),
    ]
}

fn pair(gold: &str, predicted: &str, count: u64, gold_support: u64) -> ConfusablePair {
    ConfusablePair {
        gold: gold.to_string(),
        predicted: predicted.to_string(),
        count,
        gold_support,
    }
}

/// 受け入れ条件: 最も混同しやすい組が具体値で抽出される。
#[test]
fn extracts_top_confusable_pairs_with_exact_values() {
    let m = metrics_of(&sample_pairs());
    let got = confusable_pairs(&m, 10).unwrap();
    assert_eq!(
        got,
        vec![
            pair("A", "B", 3, 5),
            pair("C", "B", 2, 3),
            pair("B", "A", 1, 2)
        ]
    );
}

/// 同数は gold・predicted の宣言順。
#[test]
fn ties_are_broken_by_declaration_order() {
    let m = metrics_of(&[
        ("C", lab("A")),
        ("B", lab("C")),
        ("B", lab("A")),
        ("A", lab("C")),
        ("A", lab("B")),
    ]);
    let got = confusable_pairs(&m, 10).unwrap();
    let keys: Vec<(&str, &str)> = got
        .iter()
        .map(|p| (p.gold.as_str(), p.predicted.as_str()))
        .collect();
    assert_eq!(
        keys,
        vec![("A", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("C", "A")]
    );
    assert!(got.iter().all(|p| p.count == 1));
}

/// Invalid・未知ラベル・Abstain・Error は組に現れない。
#[test]
fn non_label_columns_are_excluded() {
    let m = metrics_of(&[
        ("A", Outcome::Invalid),
        ("A", lab("unknown")),
        ("A", Outcome::Abstain),
        ("B", Outcome::Error),
        ("B", Outcome::Error),
        ("B", lab("B")),
    ]);
    assert_eq!(confusable_pairs(&m, 10).unwrap(), Vec::new());
}

/// 誤りが無ければ空（エラーではない）。
#[test]
fn perfect_predictions_yield_empty() {
    let m = metrics_of(&[("A", lab("A")), ("B", lab("B")), ("C", lab("C"))]);
    assert!(confusable_pairs(&m, 1).unwrap().is_empty());
}

#[test]
fn top_k_truncates_and_validates() {
    let m = metrics_of(&sample_pairs());
    assert_eq!(confusable_pairs(&m, 1).unwrap(), vec![pair("A", "B", 3, 5)]);
    assert_eq!(confusable_pairs(&m, 100).unwrap().len(), 3);
    assert_eq!(
        confusable_pairs(&m, 0).unwrap_err(),
        DiagnosticsError::InvalidTopK {
            top_k: 0,
            limit: MAX_CONFUSABLE_PAIRS
        }
    );
    assert_eq!(
        confusable_pairs(&m, MAX_CONFUSABLE_PAIRS + 1).unwrap_err(),
        DiagnosticsError::InvalidTopK {
            top_k: MAX_CONFUSABLE_PAIRS + 1,
            limit: MAX_CONFUSABLE_PAIRS
        }
    );
}

/// per_label と混同行列のラベル数が食い違う評価結果は fail-closed。
#[test]
fn inconsistent_metrics_fail_closed() {
    let mut m = metrics_of(&sample_pairs());
    m.per_label.pop();
    assert_eq!(
        confusable_pairs(&m, 10).unwrap_err(),
        DiagnosticsError::LabelCountMismatch {
            per_label: 2,
            confusion: 3
        }
    );
}

/// 決定性: レコードの並びを反転しても同じ結果。
#[test]
fn result_is_independent_of_record_order() {
    let mut rev = sample_pairs();
    rev.reverse();
    assert_eq!(
        confusable_pairs(&metrics_of(&sample_pairs()), 10).unwrap(),
        confusable_pairs(&metrics_of(&rev), 10).unwrap()
    );
}

fn stats_of(labels: &[&str], rows: &[(&str, &str)]) -> fandhe_edge_eval::diagnostics::BasicStats {
    let rows: Vec<StatsRow<'_>> = rows
        .iter()
        .map(|(input, label)| StatsRow { input, label })
        .collect();
    basic_stats(labels, &rows, InputKey::ByteExact).unwrap()
}

fn eval_rows() -> Vec<(&'static str, &'static str)> {
    // 評価レコードと同じ 10 行（gold の並びは sample_pairs と同じ）。
    let golds: Vec<&str> = sample_pairs().iter().map(|(g, _)| *g).collect();
    let inputs = ["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"];
    inputs.iter().copied().zip(golds).collect()
}

fn train_stats() -> fandhe_edge_eval::diagnostics::BasicStats {
    stats_of(
        &LABELS,
        &[("t0", "A"), ("t1", "A"), ("t2", "B"), ("t3", "C")],
    )
}

/// レポート統合: 各フィールドが具体値で一致する。
#[test]
fn report_integrates_stats_and_pairs() {
    let m = metrics_of(&sample_pairs());
    let train = train_stats();
    let eval = stats_of(&LABELS, &eval_rows());
    let report = diagnostic_report(train.clone(), eval.clone(), &m, 2).unwrap();
    assert_eq!(report.train(), &train);
    assert_eq!(report.eval(), &eval);
    assert_eq!(report.train().n_rows, 4);
    assert_eq!(report.eval().n_rows, 10);
    assert_eq!(
        report.confusable_pairs(),
        &[pair("A", "B", 3, 5), pair("C", "B", 2, 3)]
    );
}

#[test]
fn report_rejects_label_order_mismatch() {
    let m = metrics_of(&sample_pairs());
    let swapped = ["B", "A", "C"];
    let bad_train = stats_of(&swapped, &[("t0", "A")]);
    let eval = stats_of(&LABELS, &eval_rows());
    assert_eq!(
        diagnostic_report(bad_train, eval.clone(), &m, 5).unwrap_err(),
        DiagnosticsError::LabelOrderMismatch {
            side: DatasetSide::Train,
            index: 0
        }
    );
    let bad_eval = stats_of(&["A", "B"], &[("e0", "A")]);
    assert_eq!(
        diagnostic_report(train_stats(), bad_eval, &m, 5).unwrap_err(),
        DiagnosticsError::LabelOrderMismatch {
            side: DatasetSide::Eval,
            index: 2
        }
    );
}

#[test]
fn report_rejects_eval_row_count_mismatch() {
    let m = metrics_of(&sample_pairs());
    let eval = stats_of(&LABELS, &[("e0", "A")]);
    assert_eq!(
        diagnostic_report(train_stats(), eval, &m, 5).unwrap_err(),
        DiagnosticsError::EvalRowCountMismatch {
            eval_rows: 1,
            metrics_total: 10
        }
    );
}

/// 総行数が同じでもラベル別件数が評価結果の support と異なれば拒否する。
#[test]
fn report_rejects_label_support_mismatch() {
    let m = metrics_of(&sample_pairs());
    // 総数 10 だが A=4, B=3, C=3（評価結果は A=5, B=2, C=3）。
    let mut rows: Vec<(String, &str)> = Vec::new();
    for (i, l) in ["A", "A", "A", "A", "B", "B", "B", "C", "C", "C"]
        .iter()
        .enumerate()
    {
        rows.push((format!("e{i}"), l));
    }
    let refs: Vec<(&str, &str)> = rows.iter().map(|(k, l)| (k.as_str(), *l)).collect();
    let eval = stats_of(&LABELS, &refs);
    assert_eq!(
        diagnostic_report(train_stats(), eval, &m, 5).unwrap_err(),
        DiagnosticsError::EvalLabelSupportMismatch { index: 0 }
    );
}

/// 全非対角セルが正でも top_k 件だけが件数降順・宣言順で返る（有界ヒープ）。
#[test]
fn bounded_selection_keeps_ordering_with_ties() {
    let labels = ["A", "B", "C", "D"];
    let mut pairs: Vec<(&str, Outcome)> = Vec::new();
    for g in labels {
        for p in labels {
            if g != p {
                pairs.push((g, lab(p)));
            }
        }
    }
    pairs.push(("D", lab("A")));
    let records: Vec<EvalRecord<'_>> = pairs
        .iter()
        .map(|(g, o)| EvalRecord {
            gold: g,
            outcome: o,
        })
        .collect();
    let m = evaluate_single_select(&labels, &records).unwrap();
    let got = confusable_pairs(&m, 3).unwrap();
    let seq: Vec<(&str, &str, u64)> = got
        .iter()
        .map(|p| (p.gold.as_str(), p.predicted.as_str(), p.count))
        .collect();
    assert_eq!(seq, vec![("D", "A", 2), ("A", "B", 1), ("A", "C", 1)]);
}

/// エラーメッセージにラベル値を含めない。
#[test]
fn error_messages_hide_label_values() {
    let m = metrics_of(&sample_pairs());
    let bad = stats_of(&["secret-x", "B", "C"], &[("t0", "secret-x")]);
    let err = diagnostic_report(bad, stats_of(&LABELS, &eval_rows()), &m, 5).unwrap_err();
    let msg = err.to_string();
    assert_eq!(msg, "label order mismatch on train data at label index 0");
    assert!(!msg.contains("secret"));
}
