//! 推論関数へ `input` だけを渡すことの機械照合テスト。
//!
//! REQ-27 異常系・TASK-27.2・issue #71。PoC-9 `InvarianceTest.test_argument_recording_and_hash`
//! の `no_arg_violations` の移植。証拠の種別: テストハーネス（合成データ）。実機ではない。
//! `docs/spec` は読み込まない。

use fandhe_edge_eval::input_only::{
    ArgumentRecorder, EvalItem, InputIsolationError, InputIsolationViolation,
    run_inference_input_only, verify_input_only,
};
use fandhe_edge_eval::metrics::{EvalRecord, Outcome, evaluate_single_select};

struct Fixture {
    ids: Vec<String>,
    inputs: Vec<String>,
    golds: Vec<String>,
    tags: Vec<Vec<String>>,
}

fn fixture(n: usize) -> Fixture {
    Fixture {
        ids: (0..n).map(|i| format!("rec-{i}")).collect(),
        inputs: (0..n).map(|i| format!("text number {i}")).collect(),
        golds: (0..n)
            .map(|i| if i % 5 < 3 { "pos" } else { "neg" }.to_string())
            .collect(),
        tags: (0..n).map(|i| vec![format!("tag-{i}")]).collect(),
    }
}

fn items(f: &Fixture) -> Vec<EvalItem<'_>> {
    (0..f.ids.len())
        .map(|i| EvalItem {
            id: &f.ids[i],
            input: &f.inputs[i],
            gold: &f.golds[i],
            tags: &f.tags[i],
        })
        .collect()
}

/// 呼び出し側ループで `mk(i)` を引数として記録器へ渡し、照合結果を返す。
fn drive(
    f: &Fixture,
    calls: usize,
    mk: impl Fn(usize) -> String,
) -> Result<(), InputIsolationViolation> {
    let it = items(f);
    let mut rec = ArgumentRecorder::new(|_: &str| 0u8);
    for i in 0..calls {
        rec.call(&mk(i));
    }
    verify_input_only(&it, &rec)
}

#[test]
fn req27_input_only_has_no_violation() {
    let f = fixture(20);
    let it = items(&f);
    let preds = run_inference_input_only(&it, |_| "pos").unwrap();
    assert_eq!(preds.len(), 20);
    assert_eq!(drive(&f, 20, |i| f.inputs[i].clone()), Ok(()));
}

#[test]
fn req27_gold_concatenated_input_is_detected_for_all() {
    let f = fixture(20);
    let r = drive(&f, 20, |i| format!("{}{}", f.inputs[i], f.golds[i]));
    assert_eq!(
        r,
        Err(InputIsolationViolation::ArgumentMismatch {
            indices: (0..20).collect()
        })
    );
}

#[test]
fn req27_id_or_tags_instead_of_input_is_detected() {
    let f = fixture(20);
    let r = drive(&f, 20, |i| f.ids[i].clone());
    assert_eq!(
        r,
        Err(InputIsolationViolation::ArgumentMismatch {
            indices: (0..20).collect()
        })
    );
    let r = drive(&f, 20, |i| {
        format!("{} {}", f.inputs[i], f.tags[i].join(","))
    });
    assert_eq!(
        r,
        Err(InputIsolationViolation::ArgumentMismatch {
            indices: (0..20).collect()
        })
    );
}

#[test]
fn req27_partial_leak_reports_exact_indices() {
    let f = fixture(20);
    let r = drive(&f, 20, |i| {
        if i == 3 || i == 7 {
            format!("{}{}", f.inputs[i], f.golds[i])
        } else {
            f.inputs[i].clone()
        }
    });
    assert_eq!(
        r,
        Err(InputIsolationViolation::ArgumentMismatch {
            indices: vec![3, 7]
        })
    );
}

#[test]
fn req27_call_count_mismatch() {
    let f = fixture(20);
    assert_eq!(
        drive(&f, 19, |i| f.inputs[i].clone()),
        Err(InputIsolationViolation::CallCountMismatch {
            expected: 20,
            actual: 19
        })
    );
    assert_eq!(
        drive(&f, 21, |i| f.inputs[i % 20].clone()),
        Err(InputIsolationViolation::CallCountMismatch {
            expected: 20,
            actual: 21
        })
    );
}

#[test]
fn req27_no_false_positive_for_gold_substring_duplicates_and_empty() {
    let tags: Vec<String> = vec![];
    let raw = ["this is pos text", "same", "same", ""];
    let it: Vec<EvalItem<'_>> = raw
        .iter()
        .map(|s| EvalItem {
            id: "x",
            input: s,
            gold: "pos",
            tags: &tags,
        })
        .collect();
    let mut seen = Vec::new();
    let preds = run_inference_input_only(&it, |s| {
        seen.push(s.to_string());
        s.len()
    })
    .unwrap();
    assert_eq!(preds, vec![16, 4, 4, 0]);
    assert_eq!(seen, raw);
}

#[test]
fn req27_run_returns_error_type_and_no_values_in_display() {
    let e = InputIsolationError::Violation(InputIsolationViolation::ArgumentMismatch {
        indices: vec![0],
    });
    assert_eq!(
        e.to_string(),
        "input isolation violated: 1 inference call(s) received an argument other than the record input"
    );
}

#[test]
fn req27_gold_only_used_in_aggregation() {
    let f = fixture(20);
    let it = items(&f);
    // 推論クロージャは gold を一切参照しない。予測は受け取った input（"text number N"）
    // から決定的に導く: N < 12 は fixture の規則（N % 5 < 3 なら pos）どおりに、
    // N >= 12 はその反対のラベルを返す。gold は集計側でのみ使う。
    let outcomes: Vec<Outcome> = run_inference_input_only(&it, |input| {
        let n: usize = input
            .strip_prefix("text number ")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let rule_pos = n % 5 < 3;
        let pos = if n < 12 { rule_pos } else { !rule_pos };
        Outcome::Label(if pos { "pos" } else { "neg" }.to_string())
    })
    .unwrap();
    let recs: Vec<EvalRecord<'_>> = it
        .iter()
        .zip(outcomes.iter())
        .map(|(i, o)| EvalRecord {
            gold: i.gold,
            outcome: o,
        })
        .collect();
    let m = evaluate_single_select(&["pos", "neg"], &recs).unwrap();
    assert_eq!(m.accuracy.overall.numerator(), 12);
    assert_eq!(m.accuracy.overall.denominator(), 20);
    assert!((m.accuracy.overall.value() - 0.6).abs() < 1e-9);
}
