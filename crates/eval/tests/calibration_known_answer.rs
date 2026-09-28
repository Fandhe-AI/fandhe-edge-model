//! [`fandhe_edge_eval::calibration`] の既知解結合テスト（TASK-22.1-1・issue #95）。
//!
//! 証拠の種別: テストハーネス（閉形式の式・手計算値との照合）。
//! `docs/spec` は参照しない（`.claude/rules/spec-reference.md`「運用」）。
//! 浮動小数の許容差は 1e-9（`.claude/rules/evaluation-contract.md`「決定性」）。
//! 期待値はテスト対象の出力をそのまま貼り付けるのではなく、閉形式・手計算の
//! 式から独立に導く（`calibration.rs` モジュールコメント 1 節参照。PoC-12 の
//! ログにある `T*` の数値との一致は検証しない。scipy 由来の探索誤差を含む
//! ためで、本実装は閉形式の解との一致で検証する）。

use fandhe_edge_eval::calibration::{
    CalibrationError, CalibrationRecord, calibrate, select_threshold,
};

const FLOAT_EPSILON: f64 = 1e-9;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < FLOAT_EPSILON
}

/// 2 ラベル・全行 `[a, 0]` の合成 validation を作る。`gold_is_label0` の
/// 個数が `n_label0`、残りが label1。
fn synthetic_records(a: f64, n_label0: usize, n_label1: usize) -> Vec<(String, Vec<f64>)> {
    let mut records = Vec::with_capacity(n_label0 + n_label1);
    for _ in 0..n_label0 {
        records.push(("label0".to_string(), vec![a, 0.0]));
    }
    for _ in 0..n_label1 {
        records.push(("label1".to_string(), vec![a, 0.0]));
    }
    records
}

fn as_calibration_records(rows: &[(String, Vec<f64>)]) -> Vec<CalibrationRecord<'_>> {
    rows.iter()
        .map(|(gold, logits)| CalibrationRecord {
            gold: gold.as_str(),
            logits: logits.as_slice(),
        })
        .collect()
}

/// REQ-22・TASK-22.1-1: 2 ラベル・ロジット `[a, 0]`・gold=label0 の割合 q の
/// とき、閉形式 `T* = a / ln(q/(1-q))` と 1e-9 で一致する。
///
/// 導出: label0 の確率は `p0(T) = 1/(1+exp(-a/T))`。NLL の導関数を 0 と
/// 置くと `q = p0(T*)` になる（Bernoulli の最尤推定は経験比率に一致する
/// ため）。これを T について解くと `T* = a / ln(q/(1-q))`。
#[test]
fn req22_closed_form_temperature_star_case_a() {
    let a = 2.0;
    let rows = synthetic_records(a, 3, 1); // q = 3/4 = 0.75
    let records = as_calibration_records(&rows);
    let calibration = calibrate(&["label0", "label1"], &records).expect("valid input");

    let q: f64 = 0.75;
    let expected_t_star = a / (q / (1.0 - q)).ln();
    assert!(
        approx_eq(calibration.temperature_star(), expected_t_star),
        "T*={} expected={}",
        calibration.temperature_star(),
        expected_t_star
    );
}

/// REQ-22・TASK-22.1-1: 別組（a=1.5, q=0.6）でも閉形式と一致する。
#[test]
fn req22_closed_form_temperature_star_case_b() {
    let a = 1.5;
    let rows = synthetic_records(a, 3, 2); // q = 3/5 = 0.6
    let records = as_calibration_records(&rows);
    let calibration = calibrate(&["label0", "label1"], &records).expect("valid input");

    let q: f64 = 0.6;
    let expected_t_star = a / (q / (1.0 - q)).ln();
    assert!(
        approx_eq(calibration.temperature_star(), expected_t_star),
        "T*={} expected={}",
        calibration.temperature_star(),
        expected_t_star
    );
}

/// REQ-22・TASK-22.1-1: a=2, q=0.75 の例で ECE(1) は `|sigmoid(2) - 0.75|`
/// になり（全行が同じビンに入るため）、ECE(T*) は 1e-9 未満（T* は NLL を
/// 最小化する温度であり、この合成例では ECE も 0 にする）。したがって
/// `adopted=true`・`chosen_temperature=T*`。τ=0.75（全行が同じ信頼度のため）・
/// coverage=4/4。
#[test]
fn req22_adoption_ece_and_threshold_case_a() {
    let a = 2.0;
    let rows = synthetic_records(a, 3, 1); // q = 0.75
    let records = as_calibration_records(&rows);
    let calibration = calibrate(&["label0", "label1"], &records).expect("valid input");

    // T=1 での label0 の確率（top1 は必ず label0 側、a=2>0 のため argmax は
    // 常に label0）。sigmoid(2) = 1/(1+exp(-2))。
    let sigmoid_2 = 1.0 / (1.0 + (-2.0f64).exp());
    let expected_ece_t1 = (sigmoid_2 - 0.75f64).abs();
    assert!(
        approx_eq(calibration.ece_t1(), expected_ece_t1),
        "ece_t1={} expected={}",
        calibration.ece_t1(),
        expected_ece_t1
    );
    assert!(
        calibration.ece_t_star() < 1e-9,
        "ece_t_star={} should be ~0",
        calibration.ece_t_star()
    );
    assert!(calibration.adopted());
    assert!(approx_eq(
        calibration.chosen_temperature(),
        calibration.temperature_star()
    ));
    assert!(approx_eq(calibration.threshold(), 0.75));
    assert_eq!(calibration.validation_coverage().numerator(), 4);
    assert_eq!(calibration.validation_coverage().denominator(), 4);
    assert_eq!(calibration.n_validation(), 4);
}

/// REQ-22・TASK-22.1-1: 全行が正解方向に大きなロジットを持つ場合、
/// h(β_hi) <= 0 となり T* は範囲の下端 [`TEMPERATURE_MIN`] ちょうどになる。
#[test]
fn req22_all_correct_direction_hits_lower_bound() {
    // a を大きくすることで、探索範囲の上端 β_hi=20 でも h < 0（まだ
    // 「もっと自信を持ってよい」方向）になるようにする。
    let rows = synthetic_records(100.0, 4, 0);
    let records = as_calibration_records(&rows);
    let calibration = calibrate(&["label0", "label1"], &records).expect("valid input");
    assert!(
        approx_eq(
            calibration.temperature_star(),
            fandhe_edge_eval::calibration::TEMPERATURE_MIN
        ),
        "T*={}",
        calibration.temperature_star()
    );
}

/// REQ-22・TASK-22.1-1: 全行が不正解方向のとき、T* は範囲の上端
/// [`TEMPERATURE_MAX`] ちょうどになる。
#[test]
fn req22_all_wrong_direction_hits_upper_bound() {
    let rows = synthetic_records(-100.0, 4, 0);
    let records = as_calibration_records(&rows);
    let calibration = calibrate(&["label0", "label1"], &records).expect("valid input");
    assert!(
        approx_eq(
            calibration.temperature_star(),
            fandhe_edge_eval::calibration::TEMPERATURE_MAX
        ),
        "T*={}",
        calibration.temperature_star()
    );
}

/// REQ-22・TASK-22.1-1: 全ロジットが等しい（`[0,0]`）行だけの validation は
/// 目的関数が完全に平らで T*=1.0。ECE(1)=ECE(T*) のため `adopted=false`。
#[test]
fn req22_flat_logits_yield_temperature_one_and_not_adopted() {
    let rows = vec![
        ("label0".to_string(), vec![0.0, 0.0]),
        ("label1".to_string(), vec![0.0, 0.0]),
    ];
    let records = as_calibration_records(&rows);
    let calibration = calibrate(&["label0", "label1"], &records).expect("valid input");
    assert!(approx_eq(calibration.temperature_star(), 1.0));
    assert!(!calibration.adopted());
    assert!(approx_eq(calibration.chosen_temperature(), 1.0));
    assert!(approx_eq(calibration.ece_t1(), calibration.ece_t_star()));
}

/// [`select_threshold`] 単体: 重複の無い 6 件で `(n-1)/5 = 1` 番目
/// （0 始まり・昇順）の値が τ になる。
#[test]
fn req22_select_threshold_six_values_no_ties() {
    let top1 = vec![0.10, 0.20, 0.30, 0.40, 0.50, 0.60];
    let threshold = select_threshold(&top1).expect("non-empty");
    // idx = (6-1)/5 = 1 -> sorted[1] = 0.20
    assert!(approx_eq(threshold, 0.20));
    let coverage = top1.iter().filter(|&&v| v >= threshold).count();
    assert_eq!(coverage, 5);
    assert!(coverage as f64 / top1.len() as f64 >= 0.8);
}

/// [`select_threshold`] 単体: 11 件で `(11-1)/5 = 2` 番目の値が τ になる。
#[test]
fn req22_select_threshold_eleven_values_no_ties() {
    let top1: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect(); // 0.0..=1.0
    let threshold = select_threshold(&top1).expect("non-empty");
    // idx = (11-1)/5 = 2 -> sorted[2] = 0.2
    assert!(approx_eq(threshold, 0.2));
    let coverage = top1.iter().filter(|&&v| v >= threshold).count();
    assert_eq!(coverage, 9);
    assert!(coverage as f64 / top1.len() as f64 >= 0.8);
}

/// [`select_threshold`] 単体: τ と同じ値の行はすべて coverage に含まれる
/// （同値が τ の周辺に複数ある場合）。
#[test]
fn req22_select_threshold_with_ties_includes_all_equal_values() {
    let top1 = vec![0.1, 0.2, 0.2, 0.2, 0.5, 0.9];
    let threshold = select_threshold(&top1).expect("non-empty");
    // idx = (6-1)/5 = 1 -> sorted = [0.1,0.2,0.2,0.2,0.5,0.9] -> sorted[1]=0.2
    assert!(approx_eq(threshold, 0.2));
    let coverage = top1.iter().filter(|&&v| v >= threshold).count();
    assert_eq!(coverage, 5); // 0.2 が 3 件 + 0.5, 0.9
}

/// REQ-22・TASK-22.1-1: gold 以外に `−∞` を含む行があっても、有限値の部分
/// だけで T* が閉形式と一致する（3 ラベルのうち 3 列目が常に `−∞`）。
#[test]
fn req22_negative_infinity_non_gold_column_matches_closed_form() {
    let a = 2.0;
    let mut rows: Vec<(String, Vec<f64>)> = Vec::new();
    for _ in 0..3 {
        rows.push(("label0".to_string(), vec![a, 0.0, f64::NEG_INFINITY]));
    }
    rows.push(("label1".to_string(), vec![a, 0.0, f64::NEG_INFINITY]));
    let records = as_calibration_records(&rows);
    let calibration = calibrate(&["label0", "label1", "label2"], &records).expect("valid input");

    let q: f64 = 0.75;
    let expected_t_star = a / (q / (1.0 - q)).ln();
    assert!(approx_eq(calibration.temperature_star(), expected_t_star));
}

/// REQ-22・TASK-22.1-1: gold が `−∞` の行を追加しても T* は変わらない
/// （導関数の計算から除外されるため）。
#[test]
fn req22_negative_infinity_gold_row_does_not_change_temperature_star() {
    let a = 2.0;
    let baseline_rows = synthetic_records(a, 3, 1);
    let baseline_records = as_calibration_records(&baseline_rows);
    let baseline = calibrate(&["label0", "label1"], &baseline_records).expect("valid input");

    let mut with_abstained_gold_rows = baseline_rows.clone();
    // gold=label1 だが label1 のロジットが -inf という、通常あり得ないが
    // fail-closed に処理できるべき行を追加する。
    with_abstained_gold_rows.push(("label1".to_string(), vec![a, f64::NEG_INFINITY]));
    let with_abstained_gold_records = as_calibration_records(&with_abstained_gold_rows);
    let with_abstained_gold =
        calibrate(&["label0", "label1"], &with_abstained_gold_records).expect("valid input");

    assert!(approx_eq(
        baseline.temperature_star(),
        with_abstained_gold.temperature_star()
    ));
    assert_eq!(with_abstained_gold.n_validation(), 5);
}

/// REQ-22 異常系: NaN ロジットは拒否される。
#[test]
fn req22_nan_logit_is_rejected() {
    let rows = vec![("label0".to_string(), vec![f64::NAN, 0.0])];
    let records = as_calibration_records(&rows);
    let result = calibrate(&["label0", "label1"], &records);
    assert_eq!(
        result,
        Err(CalibrationError::NonFiniteLogit {
            index: 0,
            label_index: 0,
        })
    );
}

/// REQ-22 異常系: `+∞` ロジットは拒否される。
#[test]
fn req22_positive_infinity_logit_is_rejected() {
    let rows = vec![("label0".to_string(), vec![f64::INFINITY, 0.0])];
    let records = as_calibration_records(&rows);
    let result = calibrate(&["label0", "label1"], &records);
    assert_eq!(
        result,
        Err(CalibrationError::NonFiniteLogit {
            index: 0,
            label_index: 0,
        })
    );
}

/// REQ-22 異常系: 1 行の全要素が `−∞` は `NoFiniteLogit`。
#[test]
fn req22_all_negative_infinity_row_is_rejected() {
    let rows = vec![(
        "label0".to_string(),
        vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
    )];
    let records = as_calibration_records(&rows);
    let result = calibrate(&["label0", "label1"], &records);
    assert_eq!(result, Err(CalibrationError::NoFiniteLogit { index: 0 }));
}

/// REQ-22 異常系: ロジット長がラベル数と不一致。
#[test]
fn req22_logit_length_mismatch_is_rejected() {
    let rows = vec![("label0".to_string(), vec![1.0])];
    let records = as_calibration_records(&rows);
    let result = calibrate(&["label0", "label1"], &records);
    assert_eq!(
        result,
        Err(CalibrationError::LogitLengthMismatch {
            index: 0,
            expected: 2,
            actual: 1,
        })
    );
}

/// REQ-22 異常系: 未知の gold ラベル。
#[test]
fn req22_unknown_gold_label_is_rejected() {
    let rows = vec![("unknown".to_string(), vec![1.0, 0.0])];
    let records = as_calibration_records(&rows);
    let result = calibrate(&["label0", "label1"], &records);
    assert_eq!(result, Err(CalibrationError::UnknownGoldLabel { index: 0 }));
}

/// REQ-22 異常系: レコードが空。
#[test]
fn req22_empty_records_is_rejected() {
    let records: Vec<CalibrationRecord> = vec![];
    let result = calibrate(&["label0", "label1"], &records);
    assert_eq!(result, Err(CalibrationError::EmptyRecords));
}

/// REQ-22 異常系: ラベル集合が空（[`crate::metrics::build_label_index`] の
/// `EmptyLabels` を包む）。
#[test]
fn req22_empty_labels_is_rejected() {
    let rows = vec![("label0".to_string(), vec![1.0])];
    let records = as_calibration_records(&rows);
    let result = calibrate(&[], &records);
    assert!(matches!(result, Err(CalibrationError::InvalidLabels(_))));
}

/// REQ-22 異常系: 重複ラベル。
#[test]
fn req22_duplicate_labels_is_rejected() {
    let rows = vec![("label0".to_string(), vec![1.0, 0.0, 0.5])];
    let records = as_calibration_records(&rows);
    let result = calibrate(&["label0", "label1", "label0"], &records);
    assert!(matches!(result, Err(CalibrationError::InvalidLabels(_))));
}

/// REQ-39・TASK-22.1-1: `n_records * n_labels` が上限を超える入力は、
/// 行ごとの検証・確保をせずに `TooManyCells` で拒否する（cell 数の判定は
/// `n_labels.checked_mul(records.len())` の値だけで行われ、行内容には
/// 依存しないため、全行同一の最小構成で境界値のみを確認する）。
#[test]
fn req22_too_many_cells_is_rejected_before_computation() {
    use fandhe_edge_eval::calibration::MAX_CALIBRATION_CELLS;

    // ラベル数 100・レコード数を MAX_CALIBRATION_CELLS を僅かに超える件数
    // にする（MAX_EVAL_RECORDS=1_000_000 は超えない値に収める）。
    const N_LABELS: usize = 100;
    let n_records = MAX_CALIBRATION_CELLS / N_LABELS + 10;
    assert!(n_records < 1_000_000, "MAX_EVAL_RECORDS 未満に収める");

    let owned_labels: Vec<String> = (0..N_LABELS).map(|i| format!("L{i}")).collect();
    let labels: Vec<&str> = owned_labels.iter().map(String::as_str).collect();
    let logits = vec![0.0f64; N_LABELS];
    let record = CalibrationRecord {
        gold: "L0",
        logits: &logits,
    };
    let records = vec![record; n_records];

    let result = calibrate(&labels, &records);
    assert_eq!(
        result,
        Err(CalibrationError::TooManyCells {
            n_records,
            n_labels: N_LABELS,
            limit: MAX_CALIBRATION_CELLS,
        })
    );
}

/// REQ-22・TASK-22.1-1: 決定性。同じ入力を 2 回呼び出し、全フィールドが
/// ビット単位で一致する。
#[test]
fn req22_calibrate_is_deterministic() {
    let rows = synthetic_records(2.0, 3, 1);
    let records = as_calibration_records(&rows);
    let first = calibrate(&["label0", "label1"], &records).expect("valid input");
    let second = calibrate(&["label0", "label1"], &records).expect("valid input");

    assert_eq!(
        first.temperature_star().to_bits(),
        second.temperature_star().to_bits()
    );
    assert_eq!(
        first.chosen_temperature().to_bits(),
        second.chosen_temperature().to_bits()
    );
    assert_eq!(first.adopted(), second.adopted());
    assert_eq!(first.threshold().to_bits(), second.threshold().to_bits());
    assert_eq!(
        first.validation_coverage().numerator(),
        second.validation_coverage().numerator()
    );
    assert_eq!(
        first.validation_coverage().denominator(),
        second.validation_coverage().denominator()
    );
    assert_eq!(first.nll_t1().to_bits(), second.nll_t1().to_bits());
    assert_eq!(first.nll_t_star().to_bits(), second.nll_t_star().to_bits());
    assert_eq!(first.ece_t1().to_bits(), second.ece_t1().to_bits());
    assert_eq!(first.ece_t_star().to_bits(), second.ece_t_star().to_bits());
    assert_eq!(first.n_validation(), second.n_validation());
}
