//! 矛盾レコード検出・メタデータ混入検出（REQ-16・TASK-16.2-2・PoC-9 追補 A-10）。
//!
//! データ検査層（REQ-16）の異常系のうち、次の 2 つを担う純粋な検出ロジック。
//!
//! 1. **矛盾レコード検出**（[`find_contradictions`]）: 正規化した入力が同じ
//!    なのに正解が異なるレコードを検出する。
//! 2. **メタデータ混入検出**（[`find_metadata_mixed`]）: レコードの `input`
//!    に、そのレコード自身の id・正解ラベル・正解の直列化表現が部分文字列
//!    として含まれる状態を検出する。推論側へ正解が漏れるため、評価を
//!    過大評価させる（PoC-9 追補 A-10・E4）。
//!
//! # 層の境界
//!
//! [`crate::split`] と同じ方針で、本モジュールはファイル I/O・JSONL パーサを
//! 持たない。呼び出し側（TASK-16.1 のデータ検査ローダ。REQ-39 のガード層）が
//! ファイルの読み込み・サイズ上限・型検証を担い、本モジュールは最小の trait
//! （[`ContradictionRecord`]・[`MetadataRecord`]）を実装したレコードの
//! スライスを受け取る純関数として提供する。1 件・1 ファイルあたりの資源上限
//! （REQ-39）も呼び出し側の責務とする。
//!
//! # レポートにデータ本文を含めない
//!
//! セキュリティ規約（`.claude/rules/security.md`）は学習・評価データに個人
//! 情報・機密情報が含まれうることを前提に、データ本文をログ・エラー
//! メッセージへ転記しないことを求める。そのため [`ContradictionReport`]・
//! [`MetadataMixedReport`] は id・件数・group_id・理由の enum のみを持ち、
//! 正規化前後の入力本文は含めない。

use std::collections::{BTreeMap, BTreeSet};

/// 矛盾検出の対象になるレコードが満たす最小の契約。
pub trait ContradictionRecord {
    /// レコード ID。
    fn id(&self) -> &str;
    /// 正規化前の入力本文。
    fn input(&self) -> &str;
    /// 呼び出し側が正準化した正解の文字列（例: キー順を整列した JSON）。
    /// 同じ意味の正解は必ず同じ文字列になるよう、呼び出し側が責務を持つ。
    fn gold_key(&self) -> &str;
    /// group ID（無ければ `None`）。
    fn group_id(&self) -> Option<&str>;
}

/// メタデータ混入検出の対象になるレコードが満たす最小の契約。
pub trait MetadataRecord {
    /// レコード ID。
    fn id(&self) -> &str;
    /// 正規化前の入力本文（生の文字列に対して部分文字列一致を取る）。
    fn input(&self) -> &str;
    /// 正解ラベル（intent 名・選択肢 ID 等）。無ければ `None`。
    fn gold_label(&self) -> Option<&str>;
    /// 正解が漏れうる直列化表現（`json.dumps` の複数パターン等）。
    /// 呼び出し側が用意する。空文字列・`"{}"`・`"null"` は判定から除外される。
    fn gold_serializations(&self) -> &[String];
}

/// 矛盾検出・メタデータ混入検出で共通のエラー。
///
/// レコード ID の検証は集計の前提であり、空 id・重複 id を許すと
/// 集計結果が黙って上書きされる（[`crate::split`] の
/// `SplitError::DuplicateRecordId` と同じ方針）。将来 REQ-21 の終了コード
/// （`invalid_input`=64）へ写像するのは CLI 側の責務であり、本層では
/// `Result::Err` を返すところまでを担う。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConsistencyError {
    /// `index` 番目のレコードの id が空文字列だった。
    EmptyRecordId { index: usize },
    /// id が重複していた。
    DuplicateRecordId(String),
}

impl std::fmt::Display for ConsistencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConsistencyError::EmptyRecordId { index } => {
                write!(f, "record id is empty (index={index})")
            }
            ConsistencyError::DuplicateRecordId(id) => {
                write!(f, "duplicate record id: {id}")
            }
        }
    }
}

impl std::error::Error for ConsistencyError {}

/// id の検証（空文字列・重複の禁止）を行う。
///
/// [`find_contradictions`]・[`find_metadata_mixed`] の前処理として共有する。
fn validate_ids<'a, I: Iterator<Item = &'a str>>(ids: I) -> Result<(), ConsistencyError> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for (index, id) in ids.enumerate() {
        if id.is_empty() {
            return Err(ConsistencyError::EmptyRecordId { index });
        }
        if !seen.insert(id) {
            return Err(ConsistencyError::DuplicateRecordId(id.to_string()));
        }
    }
    Ok(())
}

/// 矛盾検出（1 正規化入力に対する）1 件のエントリ。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContradictionEntry {
    /// 矛盾に属するレコード ID（昇順）。
    pub ids: Vec<String>,
    /// 正解の異なり数（2 以上）。
    pub distinct_gold_count: usize,
    /// このエントリに属するレコードの group_id の和集合。
    pub group_ids: BTreeSet<String>,
}

/// [`find_contradictions`] の結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContradictionReport {
    /// 正規化に使った規則の ID（[`crate::normalize::InputNormalizer::rule_id`]）。
    pub normalizer_rule_id: &'static str,
    /// 矛盾が見つかった正規化入力の異なり数。
    pub distinct_inputs: usize,
    /// 矛盾する入力に属するレコードの総数。
    pub rows: usize,
    /// 矛盾行が持つ group_id の和集合（レポート全体）。
    pub group_ids: BTreeSet<String>,
    /// 矛盾行のうち group_id を持たない行数（黙って除外せず件数を報告する）。
    pub rows_without_group_id: usize,
    /// 矛盾エントリ（各エントリの最小 id の昇順）。
    pub entries: Vec<ContradictionEntry>,
}

/// 正規化した入力が同じなのに正解が異なるレコードを検出する（REQ-16）。
///
/// # 決定性
///
/// 集約に `BTreeMap`／`BTreeSet` のみを用い、`records` の並び順に
/// 依存しない結果を返す（`.claude/rules/coding-rust.md` の決定性要件）。
///
/// # エラー
///
/// `records` に空 id・重複 id があれば `Err` を返す。
pub fn find_contradictions<R, N>(
    records: &[R],
    normalizer: &N,
) -> Result<ContradictionReport, ConsistencyError>
where
    R: ContradictionRecord,
    N: crate::normalize::InputNormalizer,
{
    validate_ids(records.iter().map(ContradictionRecord::id))?;

    // 正規化入力 -> 正解 -> このレコードたちの (id, group_id) 一覧、の集約。
    // レコード単位の group_id（`None` かどうか）を保持したまま束ねることで、
    // 「group_id を持たない行数」をレコード単位で正確に数えられるようにする
    // （group_id の和集合だけでは、同じ正解の中に group 有り・無しが混在する
    // 場合に個々の行の有無を区別できない）。
    type GoldBucket = Vec<(String, Option<String>)>;
    let mut by_input: BTreeMap<String, BTreeMap<String, GoldBucket>> = BTreeMap::new();

    for record in records {
        let normalized = normalizer.normalize(record.input()).into_owned();
        let by_gold = by_input.entry(normalized).or_default();
        let bucket = by_gold.entry(record.gold_key().to_string()).or_default();
        bucket.push((
            record.id().to_string(),
            record.group_id().map(str::to_string),
        ));
    }

    let mut entries: Vec<ContradictionEntry> = Vec::new();
    let mut rows = 0usize;
    let mut all_group_ids: BTreeSet<String> = BTreeSet::new();
    let mut rows_without_group_id = 0usize;

    for by_gold in by_input.into_values() {
        if by_gold.len() < 2 {
            // 正解が 1 種類のみ（同じ正解の重複も含む）は矛盾にしない。
            continue;
        }
        let mut ids: Vec<String> = Vec::new();
        let mut group_ids: BTreeSet<String> = BTreeSet::new();
        for bucket in by_gold.values() {
            for (id, group_id) in bucket {
                ids.push(id.clone());
                match group_id {
                    Some(g) => {
                        group_ids.insert(g.clone());
                    }
                    None => rows_without_group_id += 1,
                }
            }
        }

        ids.sort();
        rows += ids.len();
        all_group_ids.extend(group_ids.iter().cloned());

        entries.push(ContradictionEntry {
            ids,
            distinct_gold_count: by_gold.len(),
            group_ids,
        });
    }

    // 各エントリの最小 id の昇順に並べる（`ids` は既にソート済みなので
    // 先頭要素を比較すればよい）。
    entries.sort_by(|a, b| a.ids.first().cmp(&b.ids.first()));

    Ok(ContradictionReport {
        normalizer_rule_id: normalizer.rule_id(),
        distinct_inputs: entries.len(),
        rows,
        group_ids: all_group_ids,
        rows_without_group_id,
        entries,
    })
}

/// メタデータ混入の理由。
///
/// `input` にどの種類の正解情報が混入していたかを区別する
/// （PoC-9 追補 A-10 の分類に対応）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetadataMixReason {
    /// `input` が自身のレコード ID を含む。
    IdInInput,
    /// `input` が正解ラベル（intent 名等）を含む。
    GoldLabelInInput,
    /// `input` が正解の直列化表現（引数の JSON 等）を含む。
    GoldSerializationInInput,
}

/// [`find_metadata_mixed`] の結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataMixedReport {
    /// メタデータ混入が検出されたレコード数。
    pub count: usize,
    /// レコード ID -> 検出された理由の集合（複数の理由が同時に付きうる）。
    pub hits: BTreeMap<String, BTreeSet<MetadataMixReason>>,
}

/// 直列化表現として判定対象にするかどうか（空・`"{}"`・`"null"` を除外する）。
fn is_meaningful_serialization(s: &str) -> bool {
    !s.is_empty() && s != "{}" && s != "null"
}

/// レコードの `input` に、そのレコード自身の id・正解ラベル・正解の直列化
/// 表現が部分文字列として含まれていないかを検出する（REQ-16・PoC-9 A-10）。
///
/// 判定は生の `input` に対する `str::contains` のみを用いる（線形時間。
/// 独自の JSON 断片走査・スタックは持たない）。`gold_label` の判定は正解の
/// 有無に関係なく行う（`IdInInput` も同様、正解が無い行でも id の混入だけは
/// 検出できる）。
///
/// # 引数単体の値は対象外
///
/// 呼び出し側が渡す `gold_serializations` に含める値は「直列化表現全体」を
/// 想定しており、引数の値単体（例: タイトル文字列そのもの）は対象に含めない
/// 設計とする。正当な入力文に値単体が自然に現れることがあり、それを対象に
/// 含めると clean なデータでも誤検出になるため（PoC-9 A-10・E4 の判断）。
///
/// # 計算量・資源上限
///
/// 計算量は O(Σ(needle 数 × input 長)) で、1 件あたりの上限（REQ-39）は
/// 呼び出し側（TASK-16.1 のローダ）の責務とする（[`crate::split`] と同じ方針）。
///
/// # エラー
///
/// `records` に空 id・重複 id があれば `Err` を返す。
pub fn find_metadata_mixed<R>(records: &[R]) -> Result<MetadataMixedReport, ConsistencyError>
where
    R: MetadataRecord,
{
    validate_ids(records.iter().map(MetadataRecord::id))?;

    let mut hits: BTreeMap<String, BTreeSet<MetadataMixReason>> = BTreeMap::new();

    for record in records {
        let input = record.input();
        let mut reasons: BTreeSet<MetadataMixReason> = BTreeSet::new();

        if input.contains(record.id()) {
            reasons.insert(MetadataMixReason::IdInInput);
        }
        if let Some(label) = record.gold_label()
            && !label.is_empty()
            && input.contains(label)
        {
            reasons.insert(MetadataMixReason::GoldLabelInInput);
        }
        for serialization in record.gold_serializations() {
            if is_meaningful_serialization(serialization) && input.contains(serialization.as_str())
            {
                reasons.insert(MetadataMixReason::GoldSerializationInInput);
                break;
            }
        }

        if !reasons.is_empty() {
            hits.insert(record.id().to_string(), reasons);
        }
    }

    Ok(MetadataMixedReport {
        count: hits.len(),
        hits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::NfkcWhitespaceNormalizer;

    struct TestRecord {
        id: &'static str,
        input: &'static str,
        gold_key: &'static str,
        group_id: Option<&'static str>,
    }

    impl ContradictionRecord for TestRecord {
        fn id(&self) -> &str {
            self.id
        }
        fn input(&self) -> &str {
            self.input
        }
        fn gold_key(&self) -> &str {
            self.gold_key
        }
        fn group_id(&self) -> Option<&str> {
            self.group_id
        }
    }

    /// 空白だけが違う入力に異なる正解があれば矛盾 1 件。
    /// 同じ正解の重複は矛盾にしない（REQ-16）。
    #[test]
    fn req16_contradiction_detects_whitespace_variant_with_different_gold() {
        let records = vec![
            TestRecord {
                id: "a",
                input: "hello world",
                gold_key: "g1",
                group_id: Some("grp"),
            },
            TestRecord {
                id: "b",
                input: "hello   world",
                gold_key: "g2",
                group_id: Some("grp"),
            },
            TestRecord {
                id: "c",
                input: "unrelated",
                gold_key: "g3",
                group_id: Some("grp"),
            },
            // 同じ正解の重複は矛盾に数えない対照群。
            TestRecord {
                id: "d",
                input: "same same",
                gold_key: "g4",
                group_id: None,
            },
            TestRecord {
                id: "e",
                input: "same  same",
                gold_key: "g4",
                group_id: None,
            },
        ];
        let report = find_contradictions(&records, &NfkcWhitespaceNormalizer).unwrap();
        assert_eq!(report.distinct_inputs, 1);
        assert_eq!(report.rows, 2);
        assert_eq!(report.entries.len(), 1);
        assert_eq!(
            report.entries[0].ids,
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(report.entries[0].distinct_gold_count, 2);
        assert_eq!(report.rows_without_group_id, 0);
    }

    /// 正解が 3 種の場合の `distinct_gold_count == 3`。
    #[test]
    fn req16_contradiction_distinct_gold_count_three() {
        let records = vec![
            TestRecord {
                id: "a",
                input: "x",
                gold_key: "g1",
                group_id: Some("grp"),
            },
            TestRecord {
                id: "b",
                input: "x",
                gold_key: "g2",
                group_id: Some("grp"),
            },
            TestRecord {
                id: "c",
                input: "x",
                gold_key: "g3",
                group_id: Some("grp"),
            },
        ];
        let report = find_contradictions(&records, &NfkcWhitespaceNormalizer).unwrap();
        assert_eq!(report.entries.len(), 1);
        assert_eq!(report.entries[0].distinct_gold_count, 3);
        assert_eq!(report.rows, 3);
    }

    /// group_id の無い矛盾行は rows_without_group_id に数える。
    #[test]
    fn req16_contradiction_counts_rows_without_group_id() {
        let records = vec![
            TestRecord {
                id: "a",
                input: "x",
                gold_key: "g1",
                group_id: None,
            },
            TestRecord {
                id: "b",
                input: "x",
                gold_key: "g2",
                group_id: Some("grp"),
            },
        ];
        let report = find_contradictions(&records, &NfkcWhitespaceNormalizer).unwrap();
        assert_eq!(report.rows, 2);
        assert_eq!(report.rows_without_group_id, 1);
        assert_eq!(report.group_ids, BTreeSet::from(["grp".to_string()]));
    }

    /// 重複 id は Err で、値まで確認する。
    #[test]
    fn req16_contradiction_duplicate_id_is_err() {
        let records = vec![
            TestRecord {
                id: "dup",
                input: "x",
                gold_key: "g1",
                group_id: None,
            },
            TestRecord {
                id: "dup",
                input: "y",
                gold_key: "g2",
                group_id: None,
            },
        ];
        let err = find_contradictions(&records, &NfkcWhitespaceNormalizer).unwrap_err();
        assert_eq!(err, ConsistencyError::DuplicateRecordId("dup".to_string()));
    }

    /// 空 id は Err で、値（index）まで確認する。
    #[test]
    fn req16_contradiction_empty_id_is_err() {
        let records = vec![TestRecord {
            id: "",
            input: "x",
            gold_key: "g1",
            group_id: None,
        }];
        let err = find_contradictions(&records, &NfkcWhitespaceNormalizer).unwrap_err();
        assert_eq!(err, ConsistencyError::EmptyRecordId { index: 0 });
    }

    /// 既定の正規化（NFKC）で「Ａ１」（全角）と「A1」（半角）が同一入力として
    /// 扱われ、正解が異なれば矛盾として検出されることを確認する（REQ-16。
    /// `fixtures/preprocess/byte_encoding_vectors.json` の
    /// `fullwidth_alnum_nfkc` ベクタと同じ入力）。
    #[test]
    fn req16_contradiction_default_normalizer_detects_fullwidth_variant() {
        let records = vec![
            TestRecord {
                id: "a",
                input: "\u{ff21}\u{ff11}",
                gold_key: "g1",
                group_id: None,
            },
            TestRecord {
                id: "b",
                input: "A1",
                gold_key: "g2",
                group_id: None,
            },
        ];
        let report = find_contradictions(&records, &NfkcWhitespaceNormalizer).unwrap();
        assert_eq!(report.distinct_inputs, 1);
        assert_eq!(report.rows, 2);
        assert_eq!(report.normalizer_rule_id, "nfkc-whitespace-v1");
        assert_eq!(
            report.entries[0].ids,
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(report.entries[0].distinct_gold_count, 2);
    }

    /// 独自の正規化（大文字・小文字を無視）を渡すと、NFKC だけでは統合され
    /// ない揺れ（大文字・小文字）も矛盾として検出できることを示す契約テスト
    /// （既定の [`NfkcWhitespaceNormalizer`] は大文字・小文字を区別するため、
    /// 独自実装との違いが確認できる）。
    struct CaseInsensitiveNormalizer;
    impl crate::normalize::InputNormalizer for CaseInsensitiveNormalizer {
        fn rule_id(&self) -> &'static str {
            "test-case-insensitive"
        }
        fn normalize<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
            std::borrow::Cow::Owned(text.to_ascii_lowercase())
        }
    }

    #[test]
    fn req16_contradiction_custom_normalizer_detects_case_variant() {
        let records = vec![
            TestRecord {
                id: "a",
                input: "HELLO",
                gold_key: "g1",
                group_id: None,
            },
            TestRecord {
                id: "b",
                input: "hello",
                gold_key: "g2",
                group_id: None,
            },
        ];
        let with_custom = find_contradictions(&records, &CaseInsensitiveNormalizer).unwrap();
        assert_eq!(with_custom.distinct_inputs, 1);
        assert_eq!(with_custom.normalizer_rule_id, "test-case-insensitive");

        let with_default = find_contradictions(&records, &NfkcWhitespaceNormalizer).unwrap();
        assert_eq!(with_default.distinct_inputs, 0);
    }

    /// 結果の順序が入力の並べ替えに左右されない（決定性）。
    #[test]
    fn req16_contradiction_result_order_independent_of_input_order() {
        let forward = vec![
            TestRecord {
                id: "a",
                input: "x",
                gold_key: "g1",
                group_id: None,
            },
            TestRecord {
                id: "b",
                input: "x",
                gold_key: "g2",
                group_id: None,
            },
            TestRecord {
                id: "c",
                input: "y",
                gold_key: "g3",
                group_id: None,
            },
            TestRecord {
                id: "d",
                input: "y",
                gold_key: "g4",
                group_id: None,
            },
        ];
        let reversed = vec![
            TestRecord {
                id: "d",
                input: "y",
                gold_key: "g4",
                group_id: None,
            },
            TestRecord {
                id: "c",
                input: "y",
                gold_key: "g3",
                group_id: None,
            },
            TestRecord {
                id: "b",
                input: "x",
                gold_key: "g2",
                group_id: None,
            },
            TestRecord {
                id: "a",
                input: "x",
                gold_key: "g1",
                group_id: None,
            },
        ];
        let report_forward = find_contradictions(&forward, &NfkcWhitespaceNormalizer).unwrap();
        let report_reversed = find_contradictions(&reversed, &NfkcWhitespaceNormalizer).unwrap();
        assert_eq!(report_forward, report_reversed);
    }

    struct TestMetadataRecord {
        id: &'static str,
        input: &'static str,
        gold_label: Option<&'static str>,
        gold_serializations: Vec<String>,
    }

    impl MetadataRecord for TestMetadataRecord {
        fn id(&self) -> &str {
            self.id
        }
        fn input(&self) -> &str {
            self.input
        }
        fn gold_label(&self) -> Option<&str> {
            self.gold_label
        }
        fn gold_serializations(&self) -> &[String] {
            &self.gold_serializations
        }
    }

    /// IdInInput は正解ラベル・直列化が空でも検出する。
    #[test]
    fn req16_metadata_id_in_input_detected_without_gold() {
        let records = vec![TestMetadataRecord {
            id: "rec-1",
            input: "これは [rec-1] を含む入力",
            gold_label: None,
            gold_serializations: vec![],
        }];
        let report = find_metadata_mixed(&records).unwrap();
        assert_eq!(report.count, 1);
        assert_eq!(
            report.hits.get("rec-1"),
            Some(&BTreeSet::from([MetadataMixReason::IdInInput]))
        );
    }

    /// 空のラベル・"{}"・"null" は無視する。
    #[test]
    fn req16_metadata_ignores_empty_and_placeholder_serializations() {
        let records = vec![TestMetadataRecord {
            id: "rec-2",
            input: "{} null であっても混入とみなさない",
            gold_label: Some(""),
            gold_serializations: vec!["".to_string(), "{}".to_string(), "null".to_string()],
        }];
        let report = find_metadata_mixed(&records).unwrap();
        assert_eq!(report.count, 0);
    }

    /// 1 レコードに複数の理由がある場合、BTreeSet に全部入る。
    #[test]
    fn req16_metadata_multiple_reasons_all_recorded() {
        let records = vec![TestMetadataRecord {
            id: "rec-3",
            input: "rec-3 の意図は create_task で引数は {\"title\":\"x\"}",
            gold_label: Some("create_task"),
            gold_serializations: vec!["{\"title\":\"x\"}".to_string()],
        }];
        let report = find_metadata_mixed(&records).unwrap();
        assert_eq!(
            report.hits.get("rec-3"),
            Some(&BTreeSet::from([
                MetadataMixReason::IdInInput,
                MetadataMixReason::GoldLabelInInput,
                MetadataMixReason::GoldSerializationInInput,
            ]))
        );
    }

    /// 重複 id・空 id は Err で、値まで確認する。
    #[test]
    fn req16_metadata_duplicate_id_is_err() {
        let records = vec![
            TestMetadataRecord {
                id: "dup",
                input: "x",
                gold_label: None,
                gold_serializations: vec![],
            },
            TestMetadataRecord {
                id: "dup",
                input: "y",
                gold_label: None,
                gold_serializations: vec![],
            },
        ];
        let err = find_metadata_mixed(&records).unwrap_err();
        assert_eq!(err, ConsistencyError::DuplicateRecordId("dup".to_string()));
    }

    #[test]
    fn req16_metadata_empty_id_is_err() {
        let records = vec![TestMetadataRecord {
            id: "",
            input: "x",
            gold_label: None,
            gold_serializations: vec![],
        }];
        let err = find_metadata_mixed(&records).unwrap_err();
        assert_eq!(err, ConsistencyError::EmptyRecordId { index: 0 });
    }

    /// clean（正当な入力）では誤検出が起きないことを示す対照群。
    #[test]
    fn req16_metadata_clean_record_has_no_hits() {
        let records = vec![TestMetadataRecord {
            id: "rec-4",
            input: "明日の会議の予定を教えて",
            gold_label: Some("list_events"),
            gold_serializations: vec!["{\"intent\":\"list_events\",\"arguments\":{}}".to_string()],
        }];
        let report = find_metadata_mixed(&records).unwrap();
        assert_eq!(report.count, 0);
    }
}
