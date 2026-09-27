//! 検査レポート（件数・ラベル別集計）の生成（REQ-16・TASK-16.1-2）。
//!
//! [`crate::inspect::inspect_records`] から呼ばれ、検査を通過した
//! [`crate::inspect::ValidRecord`] の列とその時点で使ったラベル ID 集合から
//! [`InspectReport`] を組み立てる。集計対象・呼び出し元は
//! [`crate::inspect`] のモジュール doc・[`InspectOutcome::report`] を参照。
//!
//! # PoC-9 との差異（実装済みを装わない）
//!
//! 本モジュールは PoC-9（`docs/spec/03-poc/evaluation-contract/evaluator/inspect.py`
//! の `inspect_split`）を出典とするが、以下の点で挙動が異なる。
//!
//! - `total_rows`: PoC-9 は読み込み済みレコード数だが、本実装は
//!   [`inspect_records`](crate::inspect::inspect_records) が空行をスキップする
//!   仕様に合わせ「空行を除く行数（異常行を含む）」とする
//! - `unique_inputs`: PoC-9 は NFKC 正規化＋空白圧縮後の異なり数だが、
//!   `unicode-normalization` は未承認（`.claude/rules/dependency-policy.md`）
//!   のため、本実装は入力の**完全一致**による異なり数とする。TASK-16.2-2
//!   （issue #42）の `InputNormalizer` がマージされた後に差し替える候補
//! - `unique_outputs`: PoC-9 は `output` 全体を正準化して数えるが、
//!   [`crate::inspect::ValidRecord`] は `label_id`（`output.intent`）のみを
//!   保持するため、本実装は `label_id` の異なり数とする。構造化出力
//!   （`arguments` 等）は `ValidRecord` が保持するようになるまで対象外
//! - `label_counts` / `min_label_count`: PoC-9 は観測されたラベルのみで
//!   集計するが、本実装は定義済みラベル集合（`valid_label_ids`）全体を
//!   0 件で初期化してから集計する。0 件ラベル（REQ-16 の「ラベル欠落」の
//!   兆候）を可視化するため
//! - 集計対象: [`crate::inspect::ValidRecord`]（検査を通過したレコード）
//!   のみを数える。異常を出した行は集計から除外する（誤検出 0 件の契約を
//!   崩さないため）
//! - 重複入力の群数・行数（PoC-9 の `duplicate_input_groups`/`rows`）は
//!   含めない（TASK-16.2・issue #40〜#42 の範囲）

use std::collections::{BTreeMap, BTreeSet};

use crate::inspect::ValidRecord;

/// [`crate::inspect::inspect_records`] が返す件数・ラベル別集計レポート。
///
/// データ本文（`id`・`input` の実際の値）は保持しない
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
    /// （`valid_records.len()` と一致する）。
    pub valid_rows: usize,
    /// 1 件以上の異常を出した行の数（異常の**件数**ではなく行数。
    /// 1 行から複数の異常が出ることがあるため、
    /// `anomalies.len()` とは一致しない場合がある）。
    pub anomalous_rows: usize,
    /// 妥当なレコードの `input` の完全一致による異なり数。
    /// 正規化は行わない（上記モジュール doc「PoC-9 との差異」参照）。
    pub unique_inputs: usize,
    /// 妥当なレコードの `output.intent`（`label_id`）の異なり数。
    pub unique_outputs: usize,
    /// 定義された全ラベル ID をキーとする件数（観測 0 件のラベルも
    /// `0` として含む。`BTreeMap` により出力順が決定的）。
    pub label_counts: BTreeMap<String, usize>,
    /// `label_counts` の最小値。呼び出し側は `valid_label_ids` が空でない
    /// ことを事前に保証する（[`crate::inspect::EmptyLabelSet`] 参照）ため、
    /// `label_counts` は必ず 1 件以上のキーを持ち、本フィールドは
    /// 常に定義される（`Option` にしない）。
    pub min_label_count: usize,
}

/// 検査を通過したレコード列からレポートを組み立てる。
///
/// `total_rows`・`anomalous_rows` は呼び出し元（[`crate::inspect::inspect_records`]）
/// が検査ループ中に数えた値をそのまま受け取る（1 行が複数の異常を出す・
/// 空行は数えないため、`valid_records` や異常一覧の長さだけからは
/// 復元できない。crate 内部関数のため `pub(crate)` とする）。
///
/// `valid_label_ids` は空でないことを呼び出し元が保証する
/// （[`crate::inspect::inspect_records`] は空集合を [`crate::inspect::EmptyLabelSet`]
/// として検査前に拒否する）。
pub(crate) fn summarize(
    total_rows: usize,
    anomalous_rows: usize,
    valid_records: &[ValidRecord],
    valid_label_ids: &BTreeSet<String>,
) -> InspectReport {
    let mut label_counts: BTreeMap<String, usize> = valid_label_ids
        .iter()
        .map(|label_id| (label_id.clone(), 0usize))
        .collect();

    let mut unique_inputs: BTreeSet<&str> = BTreeSet::new();
    let mut unique_outputs: BTreeSet<&str> = BTreeSet::new();

    for record in valid_records {
        unique_inputs.insert(record.input.as_str());
        unique_outputs.insert(record.label_id.as_str());
        // `label_id` は `inspect_records` が `valid_label_ids` への所属を
        // 保証済み（未知ラベルは `UnknownLabel` として弾かれ `valid_records`
        // に混入しない）ため、既存キーのみを加算する。万一この前提が
        // 崩れても新規キーを追加しない実装にすることで、データ本文由来の
        // 文字列が `label_counts` のキーへ混入することを防ぐ。
        if let Some(count) = label_counts.get_mut(record.label_id.as_str()) {
            *count = count.saturating_add(1);
        }
    }

    // `valid_label_ids` が空でないことは呼び出し元が保証しているため、
    // `label_counts` は必ず 1 件以上のキーを持つ。`unwrap_or(0)` は
    // その保証が崩れた場合の安全側フォールバックであり、外部入力の
    // 経路で panic させないための措置（`.claude/rules/coding-rust.md`）。
    let min_label_count = label_counts.values().copied().min().unwrap_or(0);

    InspectReport {
        total_rows,
        valid_rows: valid_records.len(),
        anomalous_rows,
        unique_inputs: unique_inputs.len(),
        unique_outputs: unique_outputs.len(),
        label_counts,
        min_label_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(line: usize, input: &str, label_id: &str) -> ValidRecord {
        ValidRecord {
            line,
            id: format!("id-{line}"),
            input: input.to_string(),
            label_id: label_id.to_string(),
            tags: None,
            group_id: None,
        }
    }

    fn labels(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    /// REQ-16・TASK-16.1-2: 定義ラベル `{a, b, c}` のうち観測されたのは
    /// a×2・b×1 で、c は 0 件のまま `label_counts` に残ること。
    #[test]
    fn summarize_counts_records_per_defined_label_including_zero() {
        let valid_label_ids = labels(&["a", "b", "c"]);
        let records = vec![
            record(1, "in1", "a"),
            record(2, "in2", "a"),
            record(3, "in3", "b"),
        ];

        let report = summarize(3, 0, &records, &valid_label_ids);

        let mut expected = BTreeMap::new();
        expected.insert("a".to_string(), 2);
        expected.insert("b".to_string(), 1);
        expected.insert("c".to_string(), 0);
        assert_eq!(report.label_counts, expected);
        assert_eq!(report.min_label_count, 0);
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
        let records = vec![record(1, "same", "a"), record(2, "same", "a")];

        let report = summarize(2, 0, &records, &valid_label_ids);

        assert_eq!(report.unique_inputs, 1);
        assert_eq!(report.valid_rows, 2);
    }
}
