//! 評価入力の正常系完走確認と異常系ケース 1〜6 の結合テスト
//! （REQ-23・TASK-23.1-1・issue #55）。
//!
//! フィクスチャは PoC-9（`docs/spec/03-poc/evaluation-contract/`。private
//! submodule）から移植した `fixtures/evaluation_contract/`（出典は同ディレクトリの
//! `PROVENANCE.md`）を使う。証拠種別: テストハーネス（実機測定なし）。

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use fandhe_edge_data::eval_input::{
    ActiveRow, ErrorOrigin, EvalInputStop, InvalidPredictionReason, PredictionOutcome, Side,
    WarningAction, WarningCode, prepare_evaluation_input,
};

/// `fixtures/evaluation_contract/<case>` への絶対パスを組み立てる。
fn fixture_dir(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("evaluation_contract")
        .join(case)
}

fn read_fixture(case: &str, file: &str) -> String {
    let path = fixture_dir(case).join(file);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// `{"labels": [...]}` 形式の labels.json を `BTreeSet<String>` に変換する。
fn read_labels(case: &str) -> BTreeSet<String> {
    let content = read_fixture(case, "labels.json");
    let value: serde_json::Value =
        serde_json::from_str(&content).expect("labels.json must be valid JSON");
    value
        .get("labels")
        .and_then(|v| v.as_array())
        .expect("labels.json must have a `labels` array")
        .iter()
        .map(|v| v.as_str().expect("label must be a string").to_string())
        .collect()
}

/// labels.json を持たないケース（01・02・05・06）は `known/single-select` の
/// ラベル定義（A/B/C/D）を共通のラベル定義として使う
/// （`fixtures/evaluation_contract/PROVENANCE.md` の規約）。
fn shared_labels() -> BTreeSet<String> {
    read_labels("known/single-select")
}

fn find_active(active: &[ActiveRow], gold_line: usize) -> &ActiveRow {
    active
        .iter()
        .find(|row| row.gold_line == gold_line)
        .unwrap_or_else(|| panic!("no active row for gold_line={gold_line}"))
}

/// REQ-23 正常系: 型・enum・必須項目に合う入力が、追加の警告なしに最後まで処理される
/// （PoC-9 の既知解 `known/single-select`）。
#[test]
fn req23_normal_known_single_select_completes_without_warnings() {
    let gold = read_fixture("known/single-select", "gold.jsonl");
    let pred = read_fixture("known/single-select", "pred.jsonl");
    let labels = read_labels("known/single-select");

    let outcome =
        prepare_evaluation_input(&gold, &pred, &labels).expect("known/single-select must not stop");

    assert!(outcome.warnings.is_empty());
    assert_eq!(outcome.active.len(), 24);

    let mut label_count = 0usize;
    let mut invalid_unknown_label_count = 0usize;
    let mut abstain_count = 0usize;
    let mut error_reported_count = 0usize;
    for row in &outcome.active {
        match &row.prediction {
            PredictionOutcome::Label(_) => label_count += 1,
            PredictionOutcome::Invalid(InvalidPredictionReason::UnknownLabel) => {
                invalid_unknown_label_count += 1
            }
            PredictionOutcome::Abstain => abstain_count += 1,
            PredictionOutcome::Error(ErrorOrigin::Reported) => error_reported_count += 1,
            other => panic!("unexpected prediction outcome: {other:?}"),
        }
    }
    // expected.json: status_counts.ok=20 は「型も値も正しい ok」の件数を指し
    // （PoC-9 の集計上の区分）、predicted_label が型不正・未知ラベルの
    // 2 件（ss-12・ss-18。type_invalid_count=2）はここに含まれない。
    // 24 = 20（Label）+ 2（Invalid）+ 1（Abstain）+ 1（Error）。
    assert_eq!(label_count, 20);
    assert_eq!(invalid_unknown_label_count, 2);
    assert_eq!(abstain_count, 1);
    assert_eq!(error_reported_count, 1);

    // ss-12 (gold_line=12) は predicted_label="E"、ss-18 (gold_line=18) は
    // predicted_label="" で、いずれもラベル集合 {A,B,C,D} に含まれない。
    assert_eq!(
        find_active(&outcome.active, 12).prediction,
        PredictionOutcome::Invalid(InvalidPredictionReason::UnknownLabel)
    );
    assert_eq!(
        find_active(&outcome.active, 18).prediction,
        PredictionOutcome::Invalid(InvalidPredictionReason::UnknownLabel)
    );
    // ss-05 (gold_line=5) は abstain、ss-06 (gold_line=6) は error。
    assert_eq!(
        find_active(&outcome.active, 5).prediction,
        PredictionOutcome::Abstain
    );
    assert_eq!(
        find_active(&outcome.active, 6).prediction,
        PredictionOutcome::Error(ErrorOrigin::Reported)
    );
}

/// ケース 1: 空データは停止する（`empty_data`）。
#[test]
fn req23_case01_empty_data_stops() {
    let gold = read_fixture("anomaly/01-empty-data", "gold.jsonl");
    let pred = read_fixture("anomaly/01-empty-data", "pred.jsonl");
    let labels = shared_labels();

    let result = prepare_evaluation_input(&gold, &pred, &labels);

    assert_eq!(
        result,
        Err(EvalInputStop::EmptyData {
            gold_rows: 0,
            pred_rows: 0,
        })
    );
    assert_eq!(result.unwrap_err().code(), "empty_data");
}

/// ケース 2: gold の欠落（`label: null`）は警告して除外する。
#[test]
fn req23_case02_missing_gold_warn_exclude() {
    let gold = read_fixture("anomaly/02-missing-gold", "gold.jsonl");
    let pred = read_fixture("anomaly/02-missing-gold", "pred.jsonl");
    let labels = shared_labels();

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 02 must not stop");

    assert_eq!(outcome.warnings.len(), 1);
    let warning = &outcome.warnings[0];
    assert_eq!(warning.code, WarningCode::MissingGold);
    assert_eq!(warning.action, WarningAction::Exclude);
    assert_eq!(warning.side, Side::Gold);
    assert_eq!(warning.lines, vec![1]);

    let mut gold_lines: Vec<usize> = outcome.active.iter().map(|row| row.gold_line).collect();
    gold_lines.sort_unstable();
    assert_eq!(gold_lines, vec![2, 3]);
    // an02-2 (gold_line=2, label=A) は予測も A、an02-3 (gold_line=3, label=B) は予測も B。
    assert_eq!(find_active(&outcome.active, 2).gold_label, "A");
    assert_eq!(
        find_active(&outcome.active, 2).prediction,
        PredictionOutcome::Label("A".to_string())
    );
    assert_eq!(find_active(&outcome.active, 3).gold_label, "B");
    assert_eq!(
        find_active(&outcome.active, 3).prediction,
        PredictionOutcome::Label("B".to_string())
    );
}

/// ケース 3: 未知ラベル。gold 側は警告して除外し、pred 側は invalid として含める。
#[test]
fn req23_case03_unknown_label() {
    let gold = read_fixture("anomaly/03-unknown-label", "gold.jsonl");
    let pred = read_fixture("anomaly/03-unknown-label", "pred.jsonl");
    let labels = read_labels("anomaly/03-unknown-label");

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 03 must not stop");

    assert_eq!(outcome.warnings.len(), 1);
    let warning = &outcome.warnings[0];
    assert_eq!(warning.code, WarningCode::UnknownGoldLabel);
    assert_eq!(warning.action, WarningAction::Exclude);
    assert_eq!(warning.side, Side::Gold);
    assert_eq!(warning.lines, vec![1]);

    assert_eq!(outcome.active.len(), 1);
    let row = find_active(&outcome.active, 2);
    assert_eq!(row.gold_label, "A");
    assert_eq!(
        row.prediction,
        PredictionOutcome::Invalid(InvalidPredictionReason::UnknownLabel)
    );
}

/// ケース 4: 型不正。gold 側は警告して除外し、pred 側は invalid として含める。
#[test]
fn req23_case04_type_invalid() {
    let gold = read_fixture("anomaly/04-type-invalid", "gold.jsonl");
    let pred = read_fixture("anomaly/04-type-invalid", "pred.jsonl");
    let labels = read_labels("anomaly/04-type-invalid");

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 04 must not stop");

    assert_eq!(outcome.warnings.len(), 1);
    let warning = &outcome.warnings[0];
    assert_eq!(warning.code, WarningCode::MalformedGold);
    assert_eq!(warning.action, WarningAction::Exclude);
    assert_eq!(warning.side, Side::Gold);
    assert_eq!(warning.lines, vec![1]);

    assert_eq!(outcome.active.len(), 1);
    let row = find_active(&outcome.active, 2);
    assert_eq!(row.gold_label, "A");
    assert_eq!(
        row.prediction,
        PredictionOutcome::Invalid(InvalidPredictionReason::NotString)
    );
}

/// ケース 5: id の重複は停止する（`duplicate_id`）。
#[test]
fn req23_case05_duplicate_id_stops() {
    let gold = read_fixture("anomaly/05-duplicate-id", "gold.jsonl");
    let pred = read_fixture("anomaly/05-duplicate-id", "pred.jsonl");
    let labels = shared_labels();

    let result = prepare_evaluation_input(&gold, &pred, &labels);

    assert_eq!(
        result,
        Err(EvalInputStop::DuplicateId {
            side: Side::Gold,
            lines: vec![1, 2],
        })
    );
    assert_eq!(result.unwrap_err().code(), "duplicate_id");
}

/// ケース 6: 正規化 input の重複は警告するが除外しない（分母に含める）。
#[test]
fn req23_case06_duplicate_input_warn_include() {
    let gold = read_fixture("anomaly/06-duplicate-input", "gold.jsonl");
    let pred = read_fixture("anomaly/06-duplicate-input", "pred.jsonl");
    let labels = shared_labels();

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 06 must not stop");

    assert_eq!(outcome.warnings.len(), 1);
    let warning = &outcome.warnings[0];
    assert_eq!(warning.code, WarningCode::DuplicateInputWithinSplit);
    assert_eq!(warning.action, WarningAction::WarnInclude);
    assert_eq!(warning.side, Side::Gold);
    assert_eq!(warning.lines, vec![1, 2]);

    // 除外しないため 3 件すべてが対象として残る。
    assert_eq!(outcome.active.len(), 3);
    assert_eq!(
        find_active(&outcome.active, 1).prediction,
        PredictionOutcome::Label("A".to_string())
    );
    assert_eq!(
        find_active(&outcome.active, 2).prediction,
        PredictionOutcome::Label("A".to_string())
    );
    assert_eq!(
        find_active(&outcome.active, 3).prediction,
        PredictionOutcome::Label("B".to_string())
    );
}
