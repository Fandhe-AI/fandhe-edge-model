//! 検査レポート（件数・ラベル別集計）の生成（REQ-16・TASK-16.1-2）。
//!
//! [`crate::inspect::inspect_records`] から呼ばれ、検査を通過した
//! [`crate::inspect::ValidRecord`] の列とその時点で使ったラベル ID 集合から
//! [`InspectReport`] を組み立てる。集計対象・呼び出し元は
//! [`crate::inspect`] のモジュール doc・[`crate::inspect::InspectOutcome::report`] を参照。
//!
//! # PoC-9 との対応・差異（実装済みを装わない）
//!
//! 本モジュールは PoC-9（`docs/spec/03-poc/evaluation-contract/evaluator/inspect.py`
//! の `inspect_split`）を出典とし、`unique_outputs`・`label_counts`（PoC-9 の
//! `intent_counts`）・`min_label_count`（PoC-9 の `min_intent_count`）は
//! PoC-9 と同じ値になるよう実装している（受け入れテストは
//! `crates/data/tests/inspect_report_clean.rs`）。以下の点のみ挙動が異なる。
//!
//! - `total_rows`: PoC-9 は読み込み済みレコード数（`rows`）だが、本実装は
//!   [`inspect_records`](crate::inspect::inspect_records) が空行をスキップする
//!   仕様に合わせ「空行を除く行数（異常行を含む）」とする。妥当なレコード数は
//!   別途 `valid_rows` として持つ（PoC-9 の `rows` に相当）
//! - `unique_inputs`: PoC-9 は NFKC 正規化＋空白圧縮後の異なり数だが、
//!   `unicode-normalization` は未承認（`.claude/rules/dependency-policy.md`）
//!   のため、本実装は入力の**完全一致**による異なり数とする。TASK-16.2-2
//!   （issue #42）の `InputNormalizer` がマージされた後に差し替える候補
//! - `unique_outputs`: [`crate::inspect::ValidRecord::output_key`]（`output`
//!   オブジェクト全体をキー整列した JSON 文字列）の異なり数とする。PoC-9 の
//!   `json.dumps(out, sort_keys=True)` と同じ意味（`ValidRecord::output_key`
//!   の doc 参照）
//! - 集計対象: [`crate::inspect::ValidRecord`]（検査を通過したレコード）
//!   のみを数える。異常を出した行は集計から除外する（誤検出 0 件の契約を
//!   崩さないため）
//! - `labels_without_records`: PoC-9 に対応物は無い。定義済みラベル集合
//!   （`valid_label_ids`）のうち観測 0 件のラベル（REQ-16 の「ラベル欠落」の
//!   兆候）を可視化するために本実装が追加したフィールド。`label_counts`・
//!   `min_label_count` の意味（観測されたラベルのみで集計。PoC-9 と同じ）は
//!   変えない
//! - 重複入力の群数・行数（PoC-9 の `duplicate_input_groups`/`rows`）は
//!   含めない（TASK-16.2・issue #40〜#42 の範囲）

use std::collections::{BTreeMap, BTreeSet};

use crate::inspect::ValidRecord;

/// [`crate::inspect::inspect_records`] が返す件数・ラベル別集計レポート。
///
/// データ本文（`id`・`input`・`output` の実際の値）は保持しない
/// （`.claude/rules/security.md`「データ本文をログ・エラーメッセージへ転記しない」。
/// [`crate::inspect`] モジュール doc の「セキュリティ上の注意」も参照）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectReport {
    /// 空行を除く行数（異常行を含む。空行はそもそも 1 レコードとして
    /// 数えられないため、[`InspectOutcome::anomalies`](crate::inspect::InspectOutcome::anomalies)
    /// にも [`InspectOutcome::valid_records`](crate::inspect::InspectOutcome::valid_records)
    /// にも現れず、`total_rows` からも除く）。
    pub total_rows: usize,
    /// 検査を通過した妥当なレコードの数
    /// （`valid_records.len()` と一致する。PoC-9 の `rows` に相当）。
    pub valid_rows: usize,
    /// 1 件以上の異常を出した行の数（異常の**件数**ではなく行数。
    /// 1 行から複数の異常が出ることがあるため、
    /// `anomalies.len()` とは一致しない場合がある）。
    pub anomalous_rows: usize,
    /// 妥当なレコードの `input` の完全一致による異なり数。
    /// 正規化は行わない（上記モジュール doc「PoC-9 との差異」参照）。
    pub unique_inputs: usize,
    /// 妥当なレコードの `output` オブジェクト全体
    /// （[`crate::inspect::ValidRecord::output_key`]）の異なり数。
    /// `output.intent` の異なり数（＝ラベル種別数）ではない点に注意
    /// （PoC-9 `inspect_split` の `unique_outputs` と同じ意味）。
    pub unique_outputs: usize,
    /// 観測された（1 件以上出現した）ラベル ID をキーとする件数
    /// （PoC-9 の `intent_counts` と同じ。`BTreeMap` により出力順が決定的）。
    /// 観測 0 件のラベルはここに含まれない（[`Self::labels_without_records`] 参照）。
    pub label_counts: BTreeMap<String, usize>,
    /// `label_counts` の最小値（観測されたラベルのみが対象。PoC-9 の
    /// `min_intent_count` と同じ）。妥当なレコードが 0 件で `label_counts` が
    /// 空のときは分母 0 を意味のある値で埋めない評価契約の方針に合わせ
    /// `None` とする（`.claude/rules/evaluation-contract.md`「分母が 0 の
    /// 指標は `null`」）。
    pub min_label_count: Option<usize>,
    /// 定義済みラベル集合（`valid_label_ids`）のうち観測が 0 件だったラベル
    /// ID（`BTreeSet` 由来のため昇順）。REQ-16 の「ラベル欠落」を利用者が
    /// 見られるようにする追加フィールド（上記モジュール doc 参照）。
    pub labels_without_records: Vec<String>,
}

/// 検査を通過したレコード列からレポートを組み立てる。
///
/// `total_rows`（空行を除く行数）は呼び出し元（[`crate::inspect::inspect_records`]）
/// が検査ループ中に数えた値をそのまま受け取る。`anomalous_rows`（1 件以上の
/// 異常を出した行数）はここでは受け取らず、`total_rows - valid_records.len()`
/// から導出する。非空行は検査ループの末尾で必ず「`valid_records` に残る
/// （異常 0 件）」か「1 件以上の異常を出す」のいずれか一方に分類される
/// （[`crate::inspect::inspect_records`] の実装上の不変条件）ため、この引き算で
/// 過不足なく求まる（1 行が複数の異常を出しても行数としては 1 回だけ数える）。
///
/// `valid_label_ids` は空でないことを呼び出し元が保証する
/// （[`crate::inspect::inspect_records`] は空集合を [`crate::inspect::EmptyLabelSet`]
/// として検査前に拒否する）。crate 内部関数のため `pub(crate)` とする。
pub(crate) fn summarize(
    total_rows: usize,
    valid_records: &[ValidRecord],
    valid_label_ids: &BTreeSet<String>,
) -> InspectReport {
    let mut label_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut unique_inputs: BTreeSet<&str> = BTreeSet::new();
    let mut unique_outputs: BTreeSet<&str> = BTreeSet::new();

    for record in valid_records {
        unique_inputs.insert(record.input.as_str());
        unique_outputs.insert(record.output_key.as_str());
        // 観測されたラベルだけを加算する（PoC-9 `intent_counts` と同じ）。
        // オーバーフローで panic させないため飽和加算とする
        // （`.claude/rules/coding-rust.md`）。
        let count = label_counts.entry(record.label_id.clone()).or_insert(0);
        *count = count.saturating_add(1);
    }

    // 分母が 0 の指標を 0 で埋めない評価契約の方針（`evaluation-contract.md`）
    // に合わせ、観測されたラベルが無ければ `None` とする。
    let min_label_count = label_counts.values().copied().min();

    // 定義済みラベル集合のうち観測 0 件だったものを昇順で列挙する。
    let labels_without_records: Vec<String> = valid_label_ids
        .iter()
        .filter(|label_id| !label_counts.contains_key(label_id.as_str()))
        .cloned()
        .collect();

    // 不変条件（本関数 doc 参照）による導出。万一の前提崩れ（総行数を
    // 下回るはずの妥当行数がそれを上回る等）でも panic させず `0` に倒す
    // （`.claude/rules/coding-rust.md`）。
    let anomalous_rows = total_rows.saturating_sub(valid_records.len());

    InspectReport {
        total_rows,
        valid_rows: valid_records.len(),
        anomalous_rows,
        unique_inputs: unique_inputs.len(),
        unique_outputs: unique_outputs.len(),
        label_counts,
        min_label_count,
        labels_without_records,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(line: usize, input: &str, label_id: &str, output_key: &str) -> ValidRecord {
        ValidRecord {
            line,
            id: format!("id-{line}"),
            input: input.to_string(),
            label_id: label_id.to_string(),
            output_key: output_key.to_string(),
            output_original: output_key.to_string(),
            tags: None,
            group_id: None,
        }
    }

    fn labels(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    /// REQ-16・TASK-16.1-2: 定義ラベル `{a, b, c}` のうち観測されたのは
    /// a×2・b×1 で、c は 0 件（`label_counts` には現れず
    /// `labels_without_records` に残る）こと。
    #[test]
    fn summarize_counts_only_observed_labels_and_lists_missing_ones() {
        let valid_label_ids = labels(&["a", "b", "c"]);
        let records = vec![
            record(1, "in1", "a", "{\"intent\":\"a\"}"),
            record(2, "in2", "a", "{\"intent\":\"a\"}"),
            record(3, "in3", "b", "{\"intent\":\"b\"}"),
        ];

        let report = summarize(3, &records, &valid_label_ids);

        let mut expected = BTreeMap::new();
        expected.insert("a".to_string(), 2);
        expected.insert("b".to_string(), 1);
        assert_eq!(report.label_counts, expected);
        assert_eq!(report.min_label_count, Some(1));
        assert_eq!(report.labels_without_records, vec!["c".to_string()]);
        assert_eq!(report.unique_outputs, 2);
        assert_eq!(report.unique_inputs, 3);
        assert_eq!(report.valid_rows, 3);
        assert_eq!(report.total_rows, 3);
        assert_eq!(report.anomalous_rows, 0);
    }

    /// `input` が完全一致する 2 件は 1 件として数えられること
    /// （正規化しない契約の確認）。
    #[test]
    fn summarize_counts_exact_duplicate_input_once() {
        let valid_label_ids = labels(&["a"]);
        let records = vec![
            record(1, "same", "a", "{\"intent\":\"a\"}"),
            record(2, "same", "a", "{\"intent\":\"a\"}"),
        ];

        let report = summarize(2, &records, &valid_label_ids);

        assert_eq!(report.unique_inputs, 1);
        assert_eq!(report.valid_rows, 2);
    }

    /// `output.intent` は同じでも `output` 全体（`arguments` 等）が異なれば
    /// `unique_outputs` は別々に数えること（`label_id` の異なり数ではない
    /// ことの固定。PoC-9 `inspect_split` の `unique_outputs` と同じ意味）。
    #[test]
    fn summarize_unique_outputs_counts_full_output_not_label_id() {
        let valid_label_ids = labels(&["x"]);
        let records = vec![
            record(1, "in1", "x", "{\"arguments\":{\"p\":1},\"intent\":\"x\"}"),
            record(2, "in2", "x", "{\"arguments\":{\"p\":2},\"intent\":\"x\"}"),
        ];

        let report = summarize(2, &records, &valid_label_ids);

        assert_eq!(report.unique_outputs, 2);
        assert_eq!(report.label_counts.get("x").copied(), Some(2));
    }

    /// 妥当なレコードが 0 件のとき、`min_label_count` を `0` で埋めず
    /// `None` とすること（評価契約「分母が 0 の指標は `null`」）。
    #[test]
    fn summarize_empty_records_yield_none_min_label_count() {
        let valid_label_ids = labels(&["a", "b"]);

        let report = summarize(0, &[], &valid_label_ids);

        assert_eq!(report.min_label_count, None);
        assert!(report.label_counts.is_empty());
        assert_eq!(
            report.labels_without_records,
            vec!["a".to_string(), "b".to_string()]
        );
    }
}
