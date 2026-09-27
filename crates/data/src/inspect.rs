//! JSONL 形式の学習・評価データを検査する（REQ-16・TASK-16.1-1）。
//!
//! CLI の `inspect` 工程（TASK-33.x で配線予定）から、ガード層を通過済みの
//! JSONL 本文を受け取って呼ばれることを想定する。件数集計（TASK-16.1-2）は
//! 本モジュールの対象外で、[`inspect_records`] は「型・必須項目・ラベル enum の
//! 検査」と「妥当なレコードの抽出」のみを行う。
//!
//! # セキュリティ上の注意（データ本文の非転記）
//!
//! [`RecordAnomaly`] はレコード本文（`input`・`output` の実際の値）を
//! 一切保持しない。位置特定は行番号・`id`・フィールド名・JSON 型名のみで行う
//! （`.claude/rules/security.md`「データ本文をログ・エラーメッセージへ転記しない」）。
//! `id`（`RecordAnomaly.id`）とラベル ID（`UnknownLabel` が保持する値）は
//! 自由記述本文ではなく構造化された識別子であるため、例外として含めてよいと
//! 判断している。ただし両者とも外部データの生値であり、検証に失敗した
//! 場合（型不正・未知のラベル）はその生値自体が異常の原因であるため、
//! 長さ・文字種の担保は無い。CLI/MCP が本構造体をそのままログ・Issue へ
//! 出力する将来の用途を考え、[`RecordAnomaly`] に格納する `id`・`label_id` は
//! [`cap_diagnostic_value`] で長さ上限（[`MAX_DIAGNOSTIC_VALUE_CHARS`]）に
//! 切り詰めてから保持する。切り詰めは診断用の複製にのみ適用し、
//! [`ValidRecord`] が保持する実際の値（分割・ハッシュ・突き合わせのキーとして
//! 使われる）は切り詰めない。
//!
//! # 既知の残存リスク（ガード層の責務）
//!
//! `serde_json::from_str::<Value>` は深いネストの JSON を再帰的に処理する。
//! serde_json の既定再帰上限（128）により、通常は極端なネストより先に
//! [`AnomalyCode::MalformedJson`] としてパースエラーで止まることを
//! `deeply_nested_json_is_rejected_before_stack_overflow` で確認している
//! （200 段ネストでの実測。証拠種別: テストハーネス）。この上限は serde_json の
//! 実装詳細であり本 crate が保証するものではないため、1 件あたりのサイズ・
//! ネスト深さの明示的な上限（REQ-39）は引き続きガード層（パス未確定）の
//! 責務とし、本関数はガード層を通過済みの入力を受け取る前提で実装している。

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

/// [`RecordAnomaly`] に格納する診断用文字列（`id`・`label_id`）の上限文字数。
///
/// ログ・Issue への転記時に無制限の任意文字列が紛れ込むのを防ぐための上限で、
/// 実データの `id`/ラベル ID の規約上の長さとは無関係の安全弁である
/// （`.claude/rules/security.md`「データ本文をログ・エラーメッセージへ転記しない」）。
const MAX_DIAGNOSTIC_VALUE_CHARS: usize = 200;

/// 診断（[`RecordAnomaly`]）にのみ使う複製を、文字数上限で切り詰める。
///
/// UTF-8 の文字境界で安全に切り詰め、切り詰めが発生した場合は末尾に
/// マーカーを付けて「値が省略されている」ことを分かるようにする。
/// [`ValidRecord`] に格納する実際の値には適用しない（上記モジュール doc 参照）。
fn cap_diagnostic_value(s: &str) -> String {
    if s.chars().count() <= MAX_DIAGNOSTIC_VALUE_CHARS {
        return s.to_string();
    }
    let truncated: String = s.chars().take(MAX_DIAGNOSTIC_VALUE_CHARS).collect();
    format!("{truncated}...(truncated)")
}

/// `inspect_records` の異常種別（本 crate の内部語彙）。
///
/// CLI が最終的に出力する `code`/`message`（REQ-21・TASK-21.1・TASK-33.2）とは
/// 別に決まる、検査ロジック内部の分類である。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyCode {
    /// 行が JSON として解釈できなかった（PoC-9 `evaluator/records.py` の
    /// `InputFileError("malformed_json", ...)` を踏襲）。
    MalformedJson,
    /// JSON としては解釈できたが object（レコード）ではなかった
    /// （PoC-9 `InputFileError("malformed_record", ...)` を踏襲）。
    MalformedRecord,
    /// 必須フィールドが存在しない。
    MissingField,
    /// フィールドは存在するが期待した JSON 型と異なる。
    TypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    /// `output.intent` が有効なラベル ID 集合に含まれない。
    /// `label_id` は [`cap_diagnostic_value`] で長さ上限に切り詰めた
    /// 診断用の複製（上記モジュール doc 参照）。
    UnknownLabel { label_id: String },
    /// `id` が既出の行と重複している。
    DuplicateId { first_line: usize },
}

impl AnomalyCode {
    /// CLI 等の上位層が機械判定に使う内部コード文字列を返す。
    pub fn code(&self) -> &'static str {
        match self {
            AnomalyCode::MalformedJson => "malformed_json",
            AnomalyCode::MalformedRecord => "malformed_record",
            AnomalyCode::MissingField => "missing_field",
            AnomalyCode::TypeMismatch { .. } => "type_mismatch",
            AnomalyCode::UnknownLabel { .. } => "unknown_label",
            AnomalyCode::DuplicateId { .. } => "duplicate_id",
        }
    }
}

/// 1 レコードで検出された 1 件の異常。
///
/// レコード本文（`input`・`output` の実値）は保持しない（上記モジュール doc 参照）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordAnomaly {
    /// 1 始まりの行番号。
    pub line: usize,
    /// 取得できた場合のみの `id`（型不正等で取得できないこともある）。
    /// [`cap_diagnostic_value`] で長さ上限に切り詰めた診断用の複製であり、
    /// [`ValidRecord::id`] の実値とは別物（上記モジュール doc 参照）。
    pub id: Option<String>,
    /// 異常が生じたフィールド名（`"id"`・`"input"`・`"output"`・`"output.intent"`・
    /// `"tags"`・`"tags[]"`・`"group_id"`・レコード自体を指す `"<record>"`）。
    pub field: &'static str,
    pub code: AnomalyCode,
}

/// 型・必須項目・ラベル enum の検査を通過した 1 レコード。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidRecord {
    /// 1 始まりの行番号。
    pub line: usize,
    pub id: String,
    pub input: String,
    /// `output.intent` の値。
    pub label_id: String,
    pub tags: Option<Vec<String>>,
    pub group_id: Option<String>,
}

/// [`inspect_records`] の結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectOutcome {
    /// 行番号順（入力順）に並んだ異常一覧。
    pub anomalies: Vec<RecordAnomaly>,
    /// 行番号順（入力順）に並んだ妥当なレコード一覧。
    pub valid_records: Vec<ValidRecord>,
}

/// 有効なラベル ID の集合が空であることを示すエラー。
///
/// ラベル定義が欠けている場合の `missing_labels` の扱いは TASK-15.4（#35）が
/// 定義するため、本関数はここでは判定せず、空集合が渡された場合は
/// 検査自体を行わずに呼び出し側へエラーとして返す。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyLabelSet;

/// JSON の値の型名（`"null"`|`"bool"`|`"number"`|`"string"`|`"array"`|`"object"`）を返す。
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// 1 行 1 JSON（JSONL）の本文を検査する。
///
/// ファイル読み込み・サイズ上限（REQ-39）はガード層／CLI 側の責務であり、
/// 本関数は既に読み込み済みの本文だけを受け取る（モジュール doc 参照）。
///
/// # 挙動
///
/// - `valid_label_ids` が空集合の場合は検査を行わず [`EmptyLabelSet`] を返す
/// - 前後空白を除去して空になった行は「レコードなし」として異常にせず読み飛ばす
///   （行番号のカウント自体は空行も含めて進める）
/// - 1 行の JSON パースに失敗したら [`AnomalyCode::MalformedJson`]、
///   パースできたが object でなければ [`AnomalyCode::MalformedRecord`] を記録し、
///   その行はそこで打ち切って次の行へ進む
/// - 1 レコードにつき複数の異常をまとめて報告する（先頭の異常で打ち切らない）
/// - 同一 `id` が複数行に現れた場合、2 回目以降の出現に
///   [`AnomalyCode::DuplicateId`] を記録する。型検査に違反がなければ
///   重複した行も含めて両方とも `valid_records` に残す（取捨選択はしない）
/// - `anomalies`・`valid_records` はいずれも入力の行順で返す
///   （`HashMap`/`HashSet` を使わず決定的な順序を保つ）
pub fn inspect_records(
    content: &str,
    valid_label_ids: &BTreeSet<String>,
) -> Result<InspectOutcome, EmptyLabelSet> {
    if valid_label_ids.is_empty() {
        return Err(EmptyLabelSet);
    }

    let mut anomalies = Vec::new();
    let mut valid_records = Vec::new();
    // id の初出行を記録する。並列化されないため BTreeMap で決定的な順序を保つ。
    let mut seen_ids: BTreeMap<String, usize> = BTreeMap::new();

    for (idx, raw_line) in content.lines().enumerate() {
        let line = idx + 1;
        if raw_line.trim().is_empty() {
            continue;
        }

        let value: Value = match serde_json::from_str(raw_line) {
            Ok(value) => value,
            Err(_) => {
                anomalies.push(RecordAnomaly {
                    line,
                    id: None,
                    field: "<record>",
                    code: AnomalyCode::MalformedJson,
                });
                continue;
            }
        };

        let Some(record) = value.as_object() else {
            anomalies.push(RecordAnomaly {
                line,
                id: None,
                field: "<record>",
                code: AnomalyCode::MalformedRecord,
            });
            continue;
        };

        let mut record_has_error = false;

        // id: 必須・string。
        let id_opt: Option<String> = match record.get("id") {
            None => {
                anomalies.push(RecordAnomaly {
                    line,
                    id: None,
                    field: "id",
                    code: AnomalyCode::MissingField,
                });
                record_has_error = true;
                None
            }
            Some(v) => match v.as_str() {
                Some(s) => Some(s.to_string()),
                None => {
                    anomalies.push(RecordAnomaly {
                        line,
                        id: None,
                        field: "id",
                        code: AnomalyCode::TypeMismatch {
                            expected: "string",
                            actual: json_type_name(v),
                        },
                    });
                    record_has_error = true;
                    None
                }
            },
        };

        // input: 必須・string（本文＝データそのもの。ログ等へ転記しない）。
        let input_opt: Option<String> = match record.get("input") {
            None => {
                anomalies.push(RecordAnomaly {
                    line,
                    id: id_opt.as_deref().map(cap_diagnostic_value),
                    field: "input",
                    code: AnomalyCode::MissingField,
                });
                record_has_error = true;
                None
            }
            Some(v) => match v.as_str() {
                Some(s) => Some(s.to_string()),
                None => {
                    anomalies.push(RecordAnomaly {
                        line,
                        id: id_opt.as_deref().map(cap_diagnostic_value),
                        field: "input",
                        code: AnomalyCode::TypeMismatch {
                            expected: "string",
                            actual: json_type_name(v),
                        },
                    });
                    record_has_error = true;
                    None
                }
            },
        };

        // output: 必須・object。output.intent: 必須・string・有効ラベル集合に含まれること。
        let label_id_opt: Option<String> = match record.get("output") {
            None => {
                anomalies.push(RecordAnomaly {
                    line,
                    id: id_opt.as_deref().map(cap_diagnostic_value),
                    field: "output",
                    code: AnomalyCode::MissingField,
                });
                record_has_error = true;
                None
            }
            Some(v) => match v.as_object() {
                None => {
                    anomalies.push(RecordAnomaly {
                        line,
                        id: id_opt.as_deref().map(cap_diagnostic_value),
                        field: "output",
                        code: AnomalyCode::TypeMismatch {
                            expected: "object",
                            actual: json_type_name(v),
                        },
                    });
                    record_has_error = true;
                    None
                }
                Some(output) => match output.get("intent") {
                    None => {
                        anomalies.push(RecordAnomaly {
                            line,
                            id: id_opt.as_deref().map(cap_diagnostic_value),
                            field: "output.intent",
                            code: AnomalyCode::MissingField,
                        });
                        record_has_error = true;
                        None
                    }
                    Some(intent_value) => match intent_value.as_str() {
                        None => {
                            anomalies.push(RecordAnomaly {
                                line,
                                id: id_opt.as_deref().map(cap_diagnostic_value),
                                field: "output.intent",
                                code: AnomalyCode::TypeMismatch {
                                    expected: "string",
                                    actual: json_type_name(intent_value),
                                },
                            });
                            record_has_error = true;
                            None
                        }
                        Some(intent) => {
                            if valid_label_ids.contains(intent) {
                                Some(intent.to_string())
                            } else {
                                anomalies.push(RecordAnomaly {
                                    line,
                                    id: id_opt.as_deref().map(cap_diagnostic_value),
                                    field: "output.intent",
                                    code: AnomalyCode::UnknownLabel {
                                        label_id: cap_diagnostic_value(intent),
                                    },
                                });
                                record_has_error = true;
                                None
                            }
                        }
                    },
                },
            },
        };

        // tags: 任意。存在する場合のみ array であること・各要素が string であることを検査する。
        let tags_opt: Option<Vec<String>> = match record.get("tags") {
            None => None,
            Some(v) => match v.as_array() {
                None => {
                    anomalies.push(RecordAnomaly {
                        line,
                        id: id_opt.as_deref().map(cap_diagnostic_value),
                        field: "tags",
                        code: AnomalyCode::TypeMismatch {
                            expected: "array",
                            actual: json_type_name(v),
                        },
                    });
                    record_has_error = true;
                    None
                }
                Some(arr) => {
                    let mut tags = Vec::with_capacity(arr.len());
                    let mut ok = true;
                    for element in arr {
                        match element.as_str() {
                            Some(s) => tags.push(s.to_string()),
                            None => {
                                anomalies.push(RecordAnomaly {
                                    line,
                                    id: id_opt.as_deref().map(cap_diagnostic_value),
                                    field: "tags[]",
                                    code: AnomalyCode::TypeMismatch {
                                        expected: "string",
                                        actual: json_type_name(element),
                                    },
                                });
                                ok = false;
                            }
                        }
                    }
                    record_has_error = record_has_error || !ok;
                    if ok { Some(tags) } else { None }
                }
            },
        };

        // group_id: 任意。存在する場合のみ string であることを検査する。
        // 欠如を異常としない根拠は PoC-9 evaluator/inspect.py の find_cross_group
        // （group_id が無い行を素通りする）。必須化・自動算出は TASK-17.1 の検討事項。
        let group_id_opt: Option<String> = match record.get("group_id") {
            None => None,
            Some(v) => match v.as_str() {
                Some(s) => Some(s.to_string()),
                None => {
                    anomalies.push(RecordAnomaly {
                        line,
                        id: id_opt.as_deref().map(cap_diagnostic_value),
                        field: "group_id",
                        code: AnomalyCode::TypeMismatch {
                            expected: "string",
                            actual: json_type_name(v),
                        },
                    });
                    record_has_error = true;
                    None
                }
            },
        };

        // 重複 id の検出。id の一意性は後続処理（分割・ハッシュ・突き合わせ）の
        // キーとしての契約の一部とみなし、コストが低いため本検査に含める。
        // DuplicateId の有無は valid_records への追加可否に影響させない
        // （値の取捨選択は TASK-16.2／分割側の責務）。
        if let Some(ref id) = id_opt {
            match seen_ids.get(id) {
                Some(&first_line) => {
                    anomalies.push(RecordAnomaly {
                        line,
                        id: Some(cap_diagnostic_value(id)),
                        field: "id",
                        code: AnomalyCode::DuplicateId { first_line },
                    });
                }
                None => {
                    seen_ids.insert(id.clone(), line);
                }
            }
        }

        if !record_has_error
            && let (Some(id), Some(input), Some(label_id)) = (id_opt, input_opt, label_id_opt)
        {
            valid_records.push(ValidRecord {
                line,
                id,
                input,
                label_id,
                tags: tags_opt,
                group_id: group_id_opt,
            });
        }
    }

    Ok(InspectOutcome {
        anomalies,
        valid_records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    /// REQ-16 正常系: clean データで誤検出が 0 件であること。
    #[test]
    fn clean_records_produce_no_anomalies() {
        let content = "\
{\"id\":\"r1\",\"input\":\"hello\",\"output\":{\"intent\":\"tier-s__low\"}}
{\"id\":\"r2\",\"input\":\"world\",\"output\":{\"intent\":\"tier-s__high\"},\"tags\":[\"a\",\"b\"],\"group_id\":\"g1\"}";
        let valid = labels(&["tier-s__low", "tier-s__high"]);

        let outcome = inspect_records(content, &valid).expect("valid_label_ids は空でない");

        assert!(outcome.anomalies.is_empty());
        assert_eq!(outcome.valid_records.len(), 2);
        assert_eq!(
            outcome.valid_records[0],
            ValidRecord {
                line: 1,
                id: "r1".to_string(),
                input: "hello".to_string(),
                label_id: "tier-s__low".to_string(),
                tags: None,
                group_id: None,
            }
        );
        assert_eq!(
            outcome.valid_records[1],
            ValidRecord {
                line: 2,
                id: "r2".to_string(),
                input: "world".to_string(),
                label_id: "tier-s__high".to_string(),
                tags: Some(vec!["a".to_string(), "b".to_string()]),
                group_id: Some("g1".to_string()),
            }
        );
    }

    #[test]
    fn missing_id_is_reported() {
        let content = "{\"input\":\"x\",\"output\":{\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: None,
                field: "id",
                code: AnomalyCode::MissingField,
            }]
        );
        assert!(outcome.valid_records.is_empty());
    }

    #[test]
    fn numeric_id_is_type_mismatch() {
        let content = "{\"id\":1,\"input\":\"x\",\"output\":{\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: None,
                field: "id",
                code: AnomalyCode::TypeMismatch {
                    expected: "string",
                    actual: "number",
                },
            }]
        );
    }

    #[test]
    fn missing_input_is_reported() {
        let content = "{\"id\":\"r1\",\"output\":{\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: Some("r1".to_string()),
                field: "input",
                code: AnomalyCode::MissingField,
            }]
        );
    }

    #[test]
    fn output_not_object_skips_intent_check() {
        let content = "{\"id\":\"r1\",\"input\":\"x\",\"output\":\"not-an-object\"}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        // output.intent 由来の異常が重複して出ないこと（1 件のみ）。
        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: Some("r1".to_string()),
                field: "output",
                code: AnomalyCode::TypeMismatch {
                    expected: "object",
                    actual: "string",
                },
            }]
        );
    }

    #[test]
    fn missing_intent_is_reported() {
        let content = "{\"id\":\"r1\",\"input\":\"x\",\"output\":{}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: Some("r1".to_string()),
                field: "output.intent",
                code: AnomalyCode::MissingField,
            }]
        );
    }

    #[test]
    fn unknown_label_is_reported() {
        let content = "{\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"nope\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: Some("r1".to_string()),
                field: "output.intent",
                code: AnomalyCode::UnknownLabel {
                    label_id: "nope".to_string(),
                },
            }]
        );
    }

    #[test]
    fn tags_not_array_is_type_mismatch() {
        let content =
            "{\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"ok\"},\"tags\":\"x\"}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: Some("r1".to_string()),
                field: "tags",
                code: AnomalyCode::TypeMismatch {
                    expected: "array",
                    actual: "string",
                },
            }]
        );
    }

    #[test]
    fn tags_element_type_mismatch() {
        let content =
            "{\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"ok\"},\"tags\":[\"a\",1]}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: Some("r1".to_string()),
                field: "tags[]",
                code: AnomalyCode::TypeMismatch {
                    expected: "string",
                    actual: "number",
                },
            }]
        );
        assert!(outcome.valid_records.is_empty());
    }

    #[test]
    fn group_id_type_mismatch() {
        let content =
            "{\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"ok\"},\"group_id\":1}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: Some("r1".to_string()),
                field: "group_id",
                code: AnomalyCode::TypeMismatch {
                    expected: "string",
                    actual: "number",
                },
            }]
        );
    }

    #[test]
    fn missing_group_id_is_not_an_anomaly() {
        let content = "{\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert!(outcome.anomalies.is_empty());
        assert_eq!(outcome.valid_records.len(), 1);
        assert_eq!(outcome.valid_records[0].group_id, None);
    }

    #[test]
    fn duplicate_id_is_reported_on_second_occurrence() {
        let content = "\
{\"id\":\"r1\",\"input\":\"a\",\"output\":{\"intent\":\"ok\"}}
{\"id\":\"r1\",\"input\":\"b\",\"output\":{\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 2,
                id: Some("r1".to_string()),
                field: "id",
                code: AnomalyCode::DuplicateId { first_line: 1 },
            }]
        );
        // 型が正しければ両方とも valid_records に残す（取捨選択はしない）。
        assert_eq!(outcome.valid_records.len(), 2);
    }

    #[test]
    fn malformed_json_line_is_reported() {
        let content = "{not json";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: None,
                field: "<record>",
                code: AnomalyCode::MalformedJson,
            }]
        );
    }

    #[test]
    fn malformed_record_non_object_is_reported() {
        let content = "[1,2,3]";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: None,
                field: "<record>",
                code: AnomalyCode::MalformedRecord,
            }]
        );
    }

    #[test]
    fn blank_lines_are_skipped_silently() {
        let content = "\n   \n{\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"ok\"}}\n\n";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert!(outcome.anomalies.is_empty());
        assert_eq!(outcome.valid_records.len(), 1);
        // 行番号のカウントは空行も含めて進む（3 行目が有効レコード）。
        assert_eq!(outcome.valid_records[0].line, 3);
    }

    #[test]
    fn empty_label_set_is_rejected() {
        let content = "{\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"ok\"}}";
        let empty: BTreeSet<String> = BTreeSet::new();

        let result = inspect_records(content, &empty);

        assert_eq!(result, Err(EmptyLabelSet));
    }

    /// 検証に失敗した `output.intent` の生値が無制限に `RecordAnomaly` へ
    /// 伝播しないこと（長さ上限で切り詰められること）。
    #[test]
    fn unknown_label_id_is_capped_for_diagnostics() {
        let long_label = "x".repeat(MAX_DIAGNOSTIC_VALUE_CHARS + 50);
        let content =
            format!("{{\"id\":\"r1\",\"input\":\"x\",\"output\":{{\"intent\":\"{long_label}\"}}}}");
        let valid = labels(&["ok"]);

        let outcome = inspect_records(&content, &valid).unwrap();

        assert_eq!(outcome.anomalies.len(), 1);
        let AnomalyCode::UnknownLabel { label_id } = &outcome.anomalies[0].code else {
            panic!("expected UnknownLabel");
        };
        assert_eq!(
            label_id.chars().count(),
            MAX_DIAGNOSTIC_VALUE_CHARS + "...(truncated)".chars().count()
        );
        assert!(label_id.ends_with("...(truncated)"));
    }

    /// 検証に失敗した `id` の生値（型不正時ではなく重複検出時の複製）も
    /// 同じ上限で切り詰められること。
    #[test]
    fn duplicate_id_value_is_capped_for_diagnostics() {
        let long_id = "y".repeat(MAX_DIAGNOSTIC_VALUE_CHARS + 50);
        let content = format!(
            "{{\"id\":\"{long_id}\",\"input\":\"a\",\"output\":{{\"intent\":\"ok\"}}}}\n\
             {{\"id\":\"{long_id}\",\"input\":\"b\",\"output\":{{\"intent\":\"ok\"}}}}"
        );
        let valid = labels(&["ok"]);

        let outcome = inspect_records(&content, &valid).unwrap();

        assert_eq!(outcome.anomalies.len(), 1);
        let anomaly_id = outcome.anomalies[0]
            .id
            .as_ref()
            .expect("DuplicateId は id を保持する");
        assert!(anomaly_id.ends_with("...(truncated)"));
        // ValidRecord 側は切り詰めない実値を保持する（分割・突き合わせのキーのため）。
        assert_eq!(outcome.valid_records[0].id, long_id);
        assert_eq!(outcome.valid_records[1].id, long_id);
    }

    /// 200 段ネストした JSON は、スタックオーバーフローより先に
    /// serde_json のパースエラーとして `MalformedJson` になること
    /// （証拠種別: テストハーネス。モジュール doc の「既知の残存リスク」参照）。
    #[test]
    fn deeply_nested_json_is_rejected_before_stack_overflow() {
        const DEPTH: usize = 200;
        let opens: String = "[".repeat(DEPTH);
        let closes: String = "]".repeat(DEPTH);
        let content = format!("{opens}1{closes}");
        let valid = labels(&["ok"]);

        let outcome = inspect_records(&content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                id: None,
                field: "<record>",
                code: AnomalyCode::MalformedJson,
            }]
        );
    }

    /// REQ-16: 1 レコードで複数フィールドが同時に不正な場合、先頭の異常で
    /// 打ち切らずすべて報告すること（モジュール doc の「挙動」節）。
    #[test]
    fn multiple_anomalies_in_one_record_are_all_reported() {
        let content = "{\"id\":1,\"output\":{\"intent\":\"nope\"},\"group_id\":2}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![
                RecordAnomaly {
                    line: 1,
                    id: None,
                    field: "id",
                    code: AnomalyCode::TypeMismatch {
                        expected: "string",
                        actual: "number",
                    },
                },
                RecordAnomaly {
                    line: 1,
                    id: None,
                    field: "input",
                    code: AnomalyCode::MissingField,
                },
                RecordAnomaly {
                    line: 1,
                    id: None,
                    field: "output.intent",
                    code: AnomalyCode::UnknownLabel {
                        label_id: "nope".to_string(),
                    },
                },
                RecordAnomaly {
                    line: 1,
                    id: None,
                    field: "group_id",
                    code: AnomalyCode::TypeMismatch {
                        expected: "string",
                        actual: "number",
                    },
                },
            ]
        );
        assert!(outcome.valid_records.is_empty());
    }

    #[test]
    fn anomaly_code_strings_match_internal_vocabulary() {
        assert_eq!(AnomalyCode::MalformedJson.code(), "malformed_json");
        assert_eq!(AnomalyCode::MalformedRecord.code(), "malformed_record");
        assert_eq!(AnomalyCode::MissingField.code(), "missing_field");
        assert_eq!(
            AnomalyCode::TypeMismatch {
                expected: "string",
                actual: "number"
            }
            .code(),
            "type_mismatch"
        );
        assert_eq!(
            AnomalyCode::UnknownLabel {
                label_id: "x".to_string()
            }
            .code(),
            "unknown_label"
        );
        assert_eq!(
            AnomalyCode::DuplicateId { first_line: 1 }.code(),
            "duplicate_id"
        );
    }
}
