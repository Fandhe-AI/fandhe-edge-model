//! 評価入力の正常系完走確認と異常系ケース 1〜12 の結合テスト
//! （REQ-23・TASK-23.1-1（issue #55・ケース 1〜6）・TASK-23.1-2
//! （issue #56・ケース 7〜12）。
//!
//! フィクスチャは PoC-9（`docs/spec/03-poc/evaluation-contract/`。private
//! submodule）から移植した `fixtures/evaluation_contract/`（出典は同ディレクトリの
//! `PROVENANCE.md`）を使う。証拠種別: テストハーネス（PoC-9 v1.1 の実測ログ
//! との照合。実機測定なし）。

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use fandhe_edge_data::eval_input::{
    ActiveRow, ErrorOrigin, EvalInputOutcome, EvalInputStop, InvalidPredictionReason,
    PredictionOutcome, Side, WarningAction, WarningCode, prepare_evaluation_input,
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

/// `outcome` の最後の警告が [`WarningCode::UnseenClass`] で、`labels` が
/// 期待どおりであることを確認する（TASK-23.1-2・ケース 9 と、02/03/04/06 が
/// PoC-9 v1.1 の実測（stderr ログ）どおり追加で出す `UnseenClass` の
/// 両方で使う共通アサーション）。
fn assert_last_warning_is_unseen_class(outcome: &EvalInputOutcome, expected_labels: &[&str]) {
    let warning = outcome
        .warnings
        .last()
        .expect("outcome.warnings must not be empty");
    assert_eq!(warning.code, WarningCode::UnseenClass);
    assert_eq!(warning.action, WarningAction::WarnInclude);
    assert_eq!(warning.side, Side::Gold);
    assert!(warning.lines.is_empty());
    let expected: Vec<String> = expected_labels.iter().map(|s| s.to_string()).collect();
    assert_eq!(warning.labels, expected);
    assert_eq!(outcome.unseen_labels, expected);
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
///
/// active な gold ラベルは A・B のみで、共有ラベル定義（A/B/C/D。ケース
/// 01・02・05・06 が使う `known/single-select` 由来）の C・D は 1 件も
/// 出現しないため、TASK-23.1-2（issue #56）で追加した [`WarningCode::UnseenClass`]
/// が PoC-9 v1.1 の stderr ログどおり追加で出る（`unseen_labels=[C,D]`）。
#[test]
fn req23_case02_missing_gold_warn_exclude() {
    let gold = read_fixture("anomaly/02-missing-gold", "gold.jsonl");
    let pred = read_fixture("anomaly/02-missing-gold", "pred.jsonl");
    let labels = shared_labels();

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 02 must not stop");

    assert_eq!(outcome.warnings.len(), 2);
    let warning = &outcome.warnings[0];
    assert_eq!(warning.code, WarningCode::MissingGold);
    assert_eq!(warning.action, WarningAction::Exclude);
    assert_eq!(warning.side, Side::Gold);
    assert_eq!(warning.lines, vec![1]);
    assert_last_warning_is_unseen_class(&outcome, &["C", "D"]);

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
///
/// ラベル定義は A・B の 2 つのみで、active な gold ラベルは A だけのため、
/// TASK-23.1-2（issue #56）の [`WarningCode::UnseenClass`] が
/// `unseen_labels=[B]` として追加で出る（PoC-9 v1.1 の実測どおり）。
#[test]
fn req23_case03_unknown_label() {
    let gold = read_fixture("anomaly/03-unknown-label", "gold.jsonl");
    let pred = read_fixture("anomaly/03-unknown-label", "pred.jsonl");
    let labels = read_labels("anomaly/03-unknown-label");

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 03 must not stop");

    assert_eq!(outcome.warnings.len(), 2);
    let warning = &outcome.warnings[0];
    assert_eq!(warning.code, WarningCode::UnknownGoldLabel);
    assert_eq!(warning.action, WarningAction::Exclude);
    assert_eq!(warning.side, Side::Gold);
    assert_eq!(warning.lines, vec![1]);
    assert_last_warning_is_unseen_class(&outcome, &["B"]);

    assert_eq!(outcome.active.len(), 1);
    let row = find_active(&outcome.active, 2);
    assert_eq!(row.gold_label, "A");
    assert_eq!(
        row.prediction,
        PredictionOutcome::Invalid(InvalidPredictionReason::UnknownLabel)
    );
}

/// ケース 4: 型不正。gold 側は警告して除外し、pred 側は invalid として含める。
///
/// ケース 3 と同じラベル定義（A・B）で active な gold ラベルは A だけのため、
/// [`WarningCode::UnseenClass`]（`unseen_labels=[B]`）が追加で出る
/// （TASK-23.1-2・issue #56）。
#[test]
fn req23_case04_type_invalid() {
    let gold = read_fixture("anomaly/04-type-invalid", "gold.jsonl");
    let pred = read_fixture("anomaly/04-type-invalid", "pred.jsonl");
    let labels = read_labels("anomaly/04-type-invalid");

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 04 must not stop");

    assert_eq!(outcome.warnings.len(), 2);
    let warning = &outcome.warnings[0];
    assert_eq!(warning.code, WarningCode::MalformedGold);
    assert_eq!(warning.action, WarningAction::Exclude);
    assert_eq!(warning.side, Side::Gold);
    assert_eq!(warning.lines, vec![1]);
    assert_last_warning_is_unseen_class(&outcome, &["B"]);

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
///
/// 共有ラベル定義（A/B/C/D）のうち active な gold ラベルは A・B のみのため、
/// [`WarningCode::UnseenClass`]（`unseen_labels=[C,D]`）が追加で出る
/// （TASK-23.1-2・issue #56。ケース 02 と同じ理由）。
#[test]
fn req23_case06_duplicate_input_warn_include() {
    let gold = read_fixture("anomaly/06-duplicate-input", "gold.jsonl");
    let pred = read_fixture("anomaly/06-duplicate-input", "pred.jsonl");
    let labels = shared_labels();

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 06 must not stop");

    assert_eq!(outcome.warnings.len(), 2);
    let warning = &outcome.warnings[0];
    assert_eq!(warning.code, WarningCode::DuplicateInputWithinSplit);
    assert_eq!(warning.action, WarningAction::WarnInclude);
    assert_eq!(warning.side, Side::Gold);
    assert_eq!(warning.lines, vec![1, 2]);
    assert_last_warning_is_unseen_class(&outcome, &["C", "D"]);

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

// --- TASK-23.1-2（issue #56）: ケース 7〜12 -------------------------------

/// ケース 7: 矛盾（同じ正規化 input に異なる gold ラベル）は警告して除外する。
///
/// `anomaly/07-contradiction` には `labels.json` が無いため、
/// `fixtures/evaluation_contract/PROVENANCE.md` の規約に従い
/// `known/single-select` のラベル定義（A/B/C/D）を使う。an07-1・an07-2 が
/// 除外されると active な gold ラベルは A のみになるため、
/// [`WarningCode::UnseenClass`]（`unseen_labels=[B,C,D]`）も出る
/// （PoC-9 v1.1 の実測）。PoC-9 の `expected.json` は v1.0 ログ
/// （`excluded_ids`）だが、除外対象の行番号は v1.0・v1.1 で変わらない。
#[test]
fn req23_case07_contradiction_warn_exclude() {
    let gold = read_fixture("anomaly/07-contradiction", "gold.jsonl");
    let pred = read_fixture("anomaly/07-contradiction", "pred.jsonl");
    let labels = shared_labels();

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 07 must not stop");

    assert_eq!(outcome.warnings.len(), 2);
    let contradiction = &outcome.warnings[0];
    assert_eq!(contradiction.code, WarningCode::ContradictoryInput);
    assert_eq!(contradiction.action, WarningAction::Exclude);
    assert_eq!(contradiction.side, Side::Gold);
    assert_eq!(contradiction.lines, vec![1, 2]);
    assert!(contradiction.labels.is_empty());
    assert_last_warning_is_unseen_class(&outcome, &["B", "C", "D"]);

    assert_eq!(outcome.active.len(), 1);
    let row = find_active(&outcome.active, 3);
    assert_eq!(row.gold_label, "A");
    assert_eq!(row.prediction, PredictionOutcome::Label("A".to_string()));
}

/// ケース 8: ラベル集合の順序が違っても（`labels.json` と
/// `labels_reordered.json`）、`known/single-select` と完全に同じ結果になる
/// （不変性テスト）。まず 2 つのラベルファイルの並びが実際に異なることを
/// 確かめてから、結果の一致を確認する（そうしないと同じファイルを 2 回
/// 読んだだけの空疎な比較になる）。
#[test]
fn req23_case08_label_order_invariant() {
    let ordered_content = read_fixture("anomaly/08-label-order", "labels.json");
    let reordered_content = read_fixture("anomaly/08-label-order", "labels_reordered.json");
    let ordered_value: serde_json::Value = serde_json::from_str(&ordered_content).unwrap();
    let reordered_value: serde_json::Value = serde_json::from_str(&reordered_content).unwrap();
    assert_ne!(
        ordered_value["labels"], reordered_value["labels"],
        "labels.json と labels_reordered.json は並びが異なるはず"
    );

    let gold = read_fixture("anomaly/08-label-order", "gold.jsonl");
    let pred = read_fixture("anomaly/08-label-order", "pred.jsonl");
    let ordered_labels = read_labels("anomaly/08-label-order");
    // labels_reordered.json も同じラベル ID 集合（BTreeSet に変換すれば順序は
    // 無視される）を持つため、`read_labels` と同じ変換をここでも行う。
    let reordered_labels: BTreeSet<String> = reordered_value
        .get("labels")
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    let outcome_ordered =
        prepare_evaluation_input(&gold, &pred, &ordered_labels).expect("case 08 must not stop");
    let outcome_reordered = prepare_evaluation_input(&gold, &pred, &reordered_labels)
        .expect("case 08 (reordered) must not stop");
    assert_eq!(outcome_ordered, outcome_reordered);

    // known/single-select と同一データのため、結果も完全一致する（警告なし）。
    let known_gold = read_fixture("known/single-select", "gold.jsonl");
    let known_pred = read_fixture("known/single-select", "pred.jsonl");
    let known_labels = read_labels("known/single-select");
    let outcome_known = prepare_evaluation_input(&known_gold, &known_pred, &known_labels)
        .expect("known/single-select must not stop");
    assert_eq!(outcome_ordered, outcome_known);
    assert!(outcome_ordered.warnings.is_empty());
    assert_eq!(outcome_ordered.active.len(), 24);
}

/// ケース 9: 未出現クラス。定義にあるが gold・pred のいずれにも出現しない
/// ラベル（C）を警告するが、件数・予測分類には影響しない。
#[test]
fn req23_case09_unseen_class_warn_include() {
    let gold = read_fixture("anomaly/09-unseen-class", "gold.jsonl");
    let pred = read_fixture("anomaly/09-unseen-class", "pred.jsonl");
    let labels = read_labels("anomaly/09-unseen-class");

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 09 must not stop");

    assert_eq!(outcome.warnings.len(), 1);
    assert_last_warning_is_unseen_class(&outcome, &["C"]);

    assert_eq!(outcome.active.len(), 4);
    assert_eq!(
        find_active(&outcome.active, 1).prediction,
        PredictionOutcome::Label("A".to_string())
    );
    assert_eq!(
        find_active(&outcome.active, 2).prediction,
        PredictionOutcome::Label("B".to_string())
    );
    assert_eq!(
        find_active(&outcome.active, 3).prediction,
        PredictionOutcome::Label("B".to_string())
    );
    assert_eq!(
        find_active(&outcome.active, 4).prediction,
        PredictionOutcome::Label("A".to_string())
    );
}

/// ケース 10: 不正なスコア（NaN・無限大・負値・合計が 1 から外れる）は
/// 除外せず [`ErrorOrigin::InvalidScore`] として含め、不正解として数える
/// （v1.1 addendum A-2。モジュール doc「PoC-9 との差分」参照。
/// `expected.json` は v1.0 の `warn_exclude` のままだが、本テストは v1.1 の
/// 挙動を照合する）。
#[test]
fn req23_case10_invalid_score_include_as_error() {
    let gold = read_fixture("anomaly/10-invalid-score", "gold.jsonl");
    let pred = read_fixture("anomaly/10-invalid-score", "pred.jsonl");
    let labels = read_labels("anomaly/10-invalid-score");

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 10 must not stop");

    assert_eq!(outcome.warnings.len(), 1);
    let warning = &outcome.warnings[0];
    assert_eq!(warning.code, WarningCode::InvalidScore);
    assert_eq!(warning.action, WarningAction::IncludeAsError);
    assert_eq!(warning.side, Side::Prediction);
    assert_eq!(warning.lines, vec![2, 3, 4, 5]);

    assert_eq!(outcome.active.len(), 5);
    assert_eq!(
        find_active(&outcome.active, 1).prediction,
        PredictionOutcome::Label("A".to_string())
    );
    for gold_line in [2, 3, 4, 5] {
        assert_eq!(
            find_active(&outcome.active, gold_line).prediction,
            PredictionOutcome::Error(ErrorOrigin::InvalidScore),
            "gold_line={gold_line} must be Error(InvalidScore)"
        );
    }
}

/// ケース 11: 全件保留。評価器は停止せず、全行が Abstain として分母に残る。
#[test]
fn req23_case11_all_abstain_warn_include() {
    let gold = read_fixture("anomaly/11-all-abstain", "gold.jsonl");
    let pred = read_fixture("anomaly/11-all-abstain", "pred.jsonl");
    let labels = read_labels("anomaly/11-all-abstain");

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 11 must not stop");

    assert_eq!(outcome.warnings.len(), 1);
    let warning = &outcome.warnings[0];
    assert_eq!(warning.code, WarningCode::AllAbstain);
    assert_eq!(warning.action, WarningAction::WarnInclude);
    assert_eq!(warning.side, Side::Prediction);
    assert_eq!(warning.lines, vec![1, 2, 3]);

    assert_eq!(outcome.active.len(), 3);
    assert!(
        outcome
            .active
            .iter()
            .all(|row| row.prediction == PredictionOutcome::Abstain)
    );
}

/// ケース 12: 全件失敗。評価器は停止せず、全行が Error として分母に残る
/// （不正解として数える）。
#[test]
fn req23_case12_all_error_warn_include() {
    let gold = read_fixture("anomaly/12-all-error", "gold.jsonl");
    let pred = read_fixture("anomaly/12-all-error", "pred.jsonl");
    let labels = read_labels("anomaly/12-all-error");

    let outcome = prepare_evaluation_input(&gold, &pred, &labels).expect("case 12 must not stop");

    assert_eq!(outcome.warnings.len(), 1);
    let warning = &outcome.warnings[0];
    assert_eq!(warning.code, WarningCode::AllError);
    assert_eq!(warning.action, WarningAction::WarnInclude);
    assert_eq!(warning.side, Side::Prediction);
    assert_eq!(warning.lines, vec![1, 2, 3]);

    assert_eq!(outcome.active.len(), 3);
    assert!(
        outcome
            .active
            .iter()
            .all(|row| row.prediction == PredictionOutcome::Error(ErrorOrigin::Reported))
    );
}
