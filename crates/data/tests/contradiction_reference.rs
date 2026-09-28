//! 矛盾レコード検出の境界値テスト（REQ-16・TASK-16.2-2）。
//!
//! # 手書きの合成データのみを使う
//!
//! 本ファイルは [`find_contradictions`] の挙動を、本ファイルにインラインで
//! 書いた合成レコード（13 件）だけで検証する。PoC-9 の実データ本文は同梱
//! しない（PR #188 レビュー指摘 P0。security.md「学習・評価データ本文を
//! ログ・エラーメッセージ・Issue・PR へ転記しない」に反するため、従来
//! 同梱していた `fixtures/data_contract/reference/{train,validation,test}.jsonl`
//! を削除した。詳細は `fixtures/data_contract/reference/PROVENANCE.md`）。
//!
//! 以前は PoC-9 参照データを `FANDHE_EDGE_POC9_DATA_DIR` 環境変数経由で読む
//! `#[ignore]` の opt-in テストも用意していたが、PR #188 レビュー指摘 P0
//! （reviewThread PRRT_kwDOUq-SxM6mg8Cy・PRRT_kwDOUq-SxM6mg8Ct）で
//! spec-reference.md「テストから docs/spec 配下を参照しない」に反し、
//! かつ既定の検証集合から回帰検出が外れると指摘されたため削除した。
//! 参照データでの照合が必要になった場合は、本リポ内の許可された合成
//! fixture を新設する（docs/spec を入力経路にしない）。

use std::collections::BTreeSet;

use fandhe_edge_data::consistency::find_contradictions;
use fandhe_edge_data::normalize::PyWhitespaceNormalizer;

/// テスト専用の [`fandhe_edge_data::consistency::ContradictionRecord`] 実装
/// （`gold_key` は呼び出し側が正準化済みの文字列を渡す契約なので、ここでは
/// 正解ラベルの文字列をそのまま使う）。
struct SyntheticRecord {
    id: &'static str,
    input: &'static str,
    gold: &'static str,
    group_id: Option<&'static str>,
}

impl fandhe_edge_data::consistency::ContradictionRecord for SyntheticRecord {
    fn id(&self) -> &str {
        self.id
    }
    fn input(&self) -> &str {
        self.input
    }
    fn gold_key(&self) -> &str {
        self.gold
    }
    fn group_id(&self) -> Option<&str> {
        self.group_id
    }
}

/// 手書きの合成プール（13 件）。
///
/// - `s1`/`s2`: 正規化後の入力 `"hello world"` が同じで正解が異なる（矛盾）。
///   前後・内部の空白の揺れが正規化で吸収されることも兼ねて確認する。
/// - `s3`/`s4`/`s5`: 正規化後の入力 `"a b c"` が同じで正解が 2 種類に分かれる
///   （3 件・distinct_gold_count=2 の矛盾。`s3`/`s5` は同じ正解）。
/// - `s6`: 単独の入力（矛盾なし）。
/// - `s7`/`s8`: 入力は同じだが正解も同じ（矛盾ではない。重複の取り扱い確認）。
/// - `s9`: group_id を持たない矛盾行（`s10` と矛盾するが group_id が無い）。
/// - `s10`: `s9` と正規化後の入力が同じで正解が異なり、group_id を持つ。
/// - `s11`/`s12`/`s13`: 正規化後の入力が同じで正解が 3 種類に分かれる
///   （distinct_gold_count=3 の境界値）。`s11`/`s13` は同じ group_id、`s12` は
///   別の group_id を持ち、1 エントリの `group_ids` が複数 group にまたがる
///   ケースを確認する。
fn synthetic_pool() -> Vec<SyntheticRecord> {
    vec![
        SyntheticRecord {
            id: "s1",
            input: "  hello   world  ",
            gold: "yes",
            group_id: Some("g-a"),
        },
        SyntheticRecord {
            id: "s2",
            input: "hello world",
            gold: "no",
            group_id: Some("g-a"),
        },
        SyntheticRecord {
            id: "s3",
            input: "a b c",
            gold: "x",
            group_id: Some("g-b"),
        },
        SyntheticRecord {
            id: "s4",
            input: "a  b  c",
            gold: "y",
            group_id: Some("g-b"),
        },
        SyntheticRecord {
            id: "s5",
            input: "a b c",
            gold: "x",
            group_id: Some("g-b"),
        },
        SyntheticRecord {
            id: "s6",
            input: "unique input",
            gold: "z",
            group_id: Some("g-c"),
        },
        SyntheticRecord {
            id: "s7",
            input: "same both",
            gold: "same",
            group_id: Some("g-d"),
        },
        SyntheticRecord {
            id: "s8",
            input: "same both",
            gold: "same",
            group_id: Some("g-d"),
        },
        SyntheticRecord {
            id: "s9",
            input: "no group here",
            gold: "p",
            group_id: None,
        },
        SyntheticRecord {
            id: "s10",
            input: "no group here",
            gold: "q",
            group_id: Some("g-e"),
        },
        SyntheticRecord {
            id: "s11",
            input: "same three way",
            gold: "p1",
            group_id: Some("g-f"),
        },
        SyntheticRecord {
            id: "s12",
            input: "same three way",
            gold: "p2",
            group_id: Some("g-g"),
        },
        SyntheticRecord {
            id: "s13",
            input: "same three way",
            gold: "p3",
            group_id: Some("g-f"),
        },
    ]
}

/// REQ-16: 手書きの合成プールで矛盾 4 個（distinct 入力）・10 行・5 group を
/// 検出する（`s1`/`s2` の 2 行、`s3`/`s4`/`s5` の 3 行、`s9`/`s10` の 2 行、
/// `s11`/`s12`/`s13` の 3 行。`s6`・`s7`/`s8` は矛盾に含まれない）。
#[test]
fn req16_synthetic_pool_counts() {
    let records = synthetic_pool();
    let report =
        find_contradictions(&records, &PyWhitespaceNormalizer).expect("id は重複・空のはず無し");

    assert_eq!(report.distinct_inputs, 4);
    assert_eq!(report.rows, 10);
    assert_eq!(report.rows_without_group_id, 1);
    assert_eq!(report.normalizer_rule_id, "py-whitespace-v1/no-nfkc");

    let expected_group_ids: BTreeSet<String> = ["g-a", "g-b", "g-e", "g-f", "g-g"]
        .into_iter()
        .map(str::to_string)
        .collect();
    assert_eq!(report.group_ids, expected_group_ids);

    let mut row_counts: Vec<usize> = report.entries.iter().map(|e| e.ids.len()).collect();
    row_counts.sort_unstable();
    assert_eq!(row_counts, vec![2, 2, 3, 3]);

    let mut distinct_gold_counts: Vec<usize> = report
        .entries
        .iter()
        .map(|e| e.distinct_gold_count)
        .collect();
    distinct_gold_counts.sort_unstable();
    assert_eq!(distinct_gold_counts, vec![2, 2, 2, 3]);
}

/// REQ-16: `s1`/`s2` の矛盾エントリの id・group_ids が具体値で一致する
/// （前後・内部の空白の揺れが正規化で吸収されていることの確認）。
#[test]
fn req16_synthetic_pool_whitespace_variant_entry_matches() {
    let records = synthetic_pool();
    let report =
        find_contradictions(&records, &PyWhitespaceNormalizer).expect("id は重複・空のはず無し");

    let matched = report
        .entries
        .iter()
        .find(|e| e.ids.iter().any(|id| id == "s1"))
        .expect("s1 を含むエントリが見つかるはず");
    assert_eq!(matched.ids, vec!["s1".to_string(), "s2".to_string()]);
    assert_eq!(matched.group_ids, BTreeSet::from(["g-a".to_string()]));
    assert_eq!(matched.distinct_gold_count, 2);
}

/// REQ-16: `s11`/`s12`/`s13` の矛盾エントリで distinct_gold_count が 3
/// （境界値。3 種類以上の正解が分かれるケース）になり、`group_ids` が
/// `g-f`・`g-g` の 2 つにまたがることを確認する。
#[test]
fn req16_synthetic_pool_three_way_gold_entry_matches() {
    let records = synthetic_pool();
    let report =
        find_contradictions(&records, &PyWhitespaceNormalizer).expect("id は重複・空のはず無し");

    let matched = report
        .entries
        .iter()
        .find(|e| e.ids.iter().any(|id| id == "s11"))
        .expect("s11 を含むエントリが見つかるはず");
    assert_eq!(
        matched.ids,
        vec!["s11".to_string(), "s12".to_string(), "s13".to_string()]
    );
    assert_eq!(
        matched.group_ids,
        BTreeSet::from(["g-f".to_string(), "g-g".to_string()])
    );
    assert_eq!(matched.distinct_gold_count, 3);
}

/// REQ-16: 同じテストを 2 回実行しても結果が一致する（決定性の確認）。
#[test]
fn req16_synthetic_pool_is_deterministic_across_runs() {
    let records = synthetic_pool();
    let report_a = find_contradictions(&records, &PyWhitespaceNormalizer).unwrap();
    let report_b = find_contradictions(&records, &PyWhitespaceNormalizer).unwrap();
    assert_eq!(report_a, report_b);
}
