//! JSONL 形式の学習・評価データを検査する（REQ-16・TASK-16.1-1・TASK-16.1-2）。
//!
//! CLI の `inspect` 工程（TASK-33.x で配線予定）から、ガード層を通過済みの
//! JSONL 本文を受け取って呼ばれることを想定する。[`inspect_records`] は
//! 「型・必須項目・ラベル enum の検査」「妥当なレコードの抽出」に加え、
//! 件数集計は [`crate::report`] が行い、結果を [`InspectOutcome::report`] に
//! 格納する（TASK-16.1-2・issue #39）。
//!
//! # セキュリティ上の注意（データ本文・識別子の非転記）
//!
//! [`RecordAnomaly`] はレコード本文（`id`・`input`・`output` の実際の値）を
//! 一切保持しない。位置特定は行番号・フィールド名・JSON 型名のみで行う
//! （`.claude/rules/security.md`「データ本文をログ・エラーメッセージへ転記しない」）。
//! 学習・評価データの `id`・`output.intent` は利用者が自由に設定できる値で
//! あり、個人情報・機密情報が混入しうる。検証に失敗した場合（型不正・未知の
//! ラベル）であっても、その生値・生値の一部（先頭 N 文字等）を
//! [`RecordAnomaly`]・[`AnomalyCode`] へ格納しない。CLI/MCP が本構造体を
//! そのままログ・Issue へ出力する将来の用途を想定した安全側の設計であり、
//! 同一 `id` の対応付けは値そのものではなく行番号
//! （[`AnomalyCode::DuplicateId`] の `first_line`）で行う。
//! [`ValidRecord`] が保持する実際の値（分割・ハッシュ・突き合わせのキーとして
//! 使われる）は本節の対象外で、検証を通過した値をそのまま保持する。
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
//!
//! # 重複 JSON キーの検出（REQ-16）
//!
//! `serde_json::Value` のパースはオブジェクトを `serde_json::Map` へ順に
//! `insert` するため、同一キーが複数回出現する行（トップレベルの `id` の
//! 重複・`output.intent` の重複を含む）は後勝ちの値のみが残り、パース結果
//! だけでは重複の事実が分からない。[`inspect_records`] は
//! パース成功後にもう一度、文字列リテラル外に現れる `:` の個数（生テキスト上の
//! key/value ペア数）とパース後の木に残ったエントリ総数を突き合わせ、両者が
//! 一致しない場合（=重複キーで木のエントリが後勝ちに潰れている場合）に
//! [`AnomalyCode::DuplicateKey`] を記録してその行を `valid_records` から
//! 除外する（`count_raw_key_value_separators`・`count_tree_entries`。
//! ネスト先を含め任意の深さの重複を検出できる。再帰は成功済みパースの木を
//! たどるだけのため、上記の serde_json 再帰上限に既に収まっている）。

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::report::{self, InspectReport};

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
    /// 文字列フィールドが空文字列（`""`）だった（`id`・`input`・`group_id`
    /// （存在する場合）が対象。空文字列は分割・ハッシュ・突き合わせの
    /// キーとして機能しないため、型は正しくても不正な値として扱う。
    /// 空文字列の `id` はこのレコード自体を無効にする（`valid_records` から
    /// 除外し、`id` の重複検出（[`AnomalyCode::DuplicateId`]）の対象にも
    /// しない。空 `id` はそもそも突き合わせのキーになり得ないため）。
    EmptyValue,
    /// `output.intent` が有効なラベル ID 集合に含まれない。
    /// 未知のラベル値そのものは保持しない（上記モジュール doc「セキュリティ上の
    /// 注意」参照）。位置特定は [`RecordAnomaly::line`]・[`RecordAnomaly::field`]
    /// （`"output.intent"`）で行う。
    UnknownLabel,
    /// `id` が既出の行と重複している。初出行（`first_line`）は `valid_records`
    /// に残る場合があるが、2 回目以降の出現は `valid_records` から除外される
    /// （`id` の一意性は分割・ハッシュ・突き合わせのキーとしての契約のため）。
    DuplicateId { first_line: usize },
    /// レコード内（トップレベルまたは `output` 等のネスト先）に同一キーが
    /// 複数回出現した（例: `{"id":1,"id":"r1",...}`）。`serde_json::Value` への
    /// パースはオブジェクトを `Map` へ挿入する際に後勝ちで上書きするため、
    /// 重複キー自体は検出せず素通りする。これを個別に検出しないと、
    /// 不正な重複キーを持つ行が型・enum 検査をすり抜けて `valid_records` へ
    /// 混入する（REQ-16 のレビュー指摘。issue #38 PR #191）。
    DuplicateKey,
}

impl AnomalyCode {
    /// CLI 等の上位層が機械判定に使う内部コード文字列を返す。
    pub fn code(&self) -> &'static str {
        match self {
            AnomalyCode::MalformedJson => "malformed_json",
            AnomalyCode::MalformedRecord => "malformed_record",
            AnomalyCode::MissingField => "missing_field",
            AnomalyCode::TypeMismatch { .. } => "type_mismatch",
            AnomalyCode::EmptyValue => "empty_value",
            AnomalyCode::UnknownLabel => "unknown_label",
            AnomalyCode::DuplicateId { .. } => "duplicate_id",
            AnomalyCode::DuplicateKey => "duplicate_key",
        }
    }
}

/// 1 レコードで検出された 1 件の異常。
///
/// レコード本文（`input`・`output` の実値）は保持しない（上記モジュール doc 参照）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordAnomaly {
    /// 1 始まりの行番号。異常のあったレコードの特定はこの行番号のみで行う
    /// （`id` 等の生値は保持しない。上記モジュール doc 参照）。
    pub line: usize,
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
    /// 件数・ラベル別集計レポート（REQ-16・TASK-16.1-2）。
    /// `valid_records` のみを集計対象とし、異常を出した行は含めない
    /// （[`crate::report`] モジュール doc 参照）。
    pub report: InspectReport,
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

/// 生の JSON テキストのうち、文字列リテラルの外に現れる `:` の個数を数える。
///
/// 妥当な JSON では `:` はオブジェクトの key/value 区切りとしてのみ現れる
/// （数値・`true`/`false`/`null` に `:` は含まれない）ため、この個数は
/// 生テキスト上に書かれた key/value ペアの総数に一致する。文字列内の `:` は
/// 引用符の開閉状態（エスケープを考慮）を追跡して除外する。バイト単位で
/// 走査するため UTF-8 の継続バイト（0x80〜0xBF）が `"`・`\`・`:` と
/// 衝突することはなく、多バイト文字境界を壊さない。
fn count_raw_key_value_separators(raw_line: &str) -> usize {
    let mut in_string = false;
    let mut escaped = false;
    let mut count = 0usize;
    for byte in raw_line.bytes() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
        } else if byte == b'"' {
            in_string = true;
        } else if byte == b':' {
            count += 1;
        }
    }
    count
}

/// パース済みの JSON 木に残ったオブジェクトのエントリ総数を数える。
///
/// 同一キーが複数回出現していた場合、`serde_json::Value` のパース時点で
/// 後勝ちの 1 エントリへ潰れているため、この値は生テキスト上のキー数より
/// 小さくなる（[`count_raw_key_value_separators`] との差分が重複検出の根拠）。
fn count_tree_entries(value: &Value) -> usize {
    match value {
        Value::Object(map) => map.len() + map.values().map(count_tree_entries).sum::<usize>(),
        Value::Array(items) => items.iter().map(count_tree_entries).sum(),
        _ => 0,
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
/// - object であっても、トップレベルまたはネスト先（`output` 等）に同一キーが
///   複数回出現していた場合は [`AnomalyCode::DuplicateKey`] を記録し、
///   その行は個々のフィールド検査を行わずに次の行へ進む（モジュール doc
///   「重複 JSON キーの検出」参照。REQ-16）
/// - `id`・`input`・`group_id`（存在する場合）が空文字列（`""`）の場合は
///   [`AnomalyCode::EmptyValue`] を記録する。空文字列の `id` はそもそも
///   突き合わせのキーになり得ないため、[`AnomalyCode::DuplicateId`] の
///   判定対象（後述）にもしない
/// - 1 レコードにつき複数の異常をまとめて報告する（先頭の異常で打ち切らない）
/// - 同一 `id`（空文字列を除く）が複数行に現れた場合、2 回目以降の出現に
///   [`AnomalyCode::DuplicateId`] を記録し、その行は `valid_records` から
///   除外する（`id` は分割・ハッシュ・突き合わせのキーであり、
///   `valid_records` 内で一意であることを保証する）。初出の行は他に
///   異常が無ければ `valid_records` に残る
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
    // 件数集計（TASK-16.1-2）向け。空行を除く行数と、1 件以上の異常を出した
    // 行の数（異常の件数ではなく行数）をループ内で数える。1 行が複数の異常を
    // 出し、かつ空行は `continue` で読み飛ばすため、`anomalies.len()` と
    // `valid_records.len()` からは事後に復元できない（[`crate::report`] 参照）。
    let mut total_rows = 0usize;
    let mut anomalous_rows = 0usize;

    for (idx, raw_line) in content.lines().enumerate() {
        let line = idx + 1;
        if raw_line.trim().is_empty() {
            continue;
        }
        total_rows += 1;
        // この行の処理開始時点の異常件数。行末で比較し、1 件でも増えていれば
        // この行を「異常を出した行」として 1 回だけ数える。
        let anomalies_before_this_line = anomalies.len();

        let value: Value = match serde_json::from_str(raw_line) {
            Ok(value) => value,
            Err(_) => {
                anomalies.push(RecordAnomaly {
                    line,
                    field: "<record>",
                    code: AnomalyCode::MalformedJson,
                });
                anomalous_rows += 1;
                continue;
            }
        };

        let Some(record) = value.as_object() else {
            anomalies.push(RecordAnomaly {
                line,
                field: "<record>",
                code: AnomalyCode::MalformedRecord,
            });
            anomalous_rows += 1;
            continue;
        };

        // 重複 JSON キーの検出（モジュール doc「重複 JSON キーの検出」参照）。
        // パース後の木では後勝ちで潰れているため、生テキストの key/value 区切り数と
        // 突き合わせて初めて検出できる。検出した行は個々のフィールド検査に進まず、
        // レコード全体を無効として次の行へ進む（MalformedRecord と同じ扱い）。
        if count_raw_key_value_separators(raw_line) != count_tree_entries(&value) {
            anomalies.push(RecordAnomaly {
                line,
                field: "<record>",
                code: AnomalyCode::DuplicateKey,
            });
            anomalous_rows += 1;
            continue;
        }

        let mut record_has_error = false;

        // id: 必須・string。
        let id_opt: Option<String> = match record.get("id") {
            None => {
                anomalies.push(RecordAnomaly {
                    line,
                    field: "id",
                    code: AnomalyCode::MissingField,
                });
                record_has_error = true;
                None
            }
            Some(v) => match v.as_str() {
                Some("") => {
                    anomalies.push(RecordAnomaly {
                        line,
                        field: "id",
                        code: AnomalyCode::EmptyValue,
                    });
                    record_has_error = true;
                    None
                }
                Some(s) => Some(s.to_string()),
                None => {
                    anomalies.push(RecordAnomaly {
                        line,
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
                    field: "input",
                    code: AnomalyCode::MissingField,
                });
                record_has_error = true;
                None
            }
            Some(v) => match v.as_str() {
                Some("") => {
                    anomalies.push(RecordAnomaly {
                        line,
                        field: "input",
                        code: AnomalyCode::EmptyValue,
                    });
                    record_has_error = true;
                    None
                }
                Some(s) => Some(s.to_string()),
                None => {
                    anomalies.push(RecordAnomaly {
                        line,
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
                                    field: "output.intent",
                                    code: AnomalyCode::UnknownLabel,
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
                Some("") => {
                    anomalies.push(RecordAnomaly {
                        line,
                        field: "group_id",
                        code: AnomalyCode::EmptyValue,
                    });
                    record_has_error = true;
                    None
                }
                Some(s) => Some(s.to_string()),
                None => {
                    anomalies.push(RecordAnomaly {
                        line,
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
        // キーとしての契約の一部であるため、2 回目以降の出現は record_has_error を
        // 立てて valid_records から除外する（初出の行は他に異常が無ければ残る）。
        // これにより valid_records 内で id が重複することはない。
        //
        // seen_ids へ登録するのはこの時点まで他に異常が無いレコード（other_fields_ok）
        // に限る。無効な初出レコード（input 欠落等で record_has_error が既に立っている）
        // の id を登録すると、後続の同一 id を持つ妥当なレコードまで DuplicateId として
        // valid_records から誤って除外してしまうため（レビュー指摘。REQ-16）。
        let other_fields_ok = !record_has_error;
        if let Some(ref id) = id_opt {
            match seen_ids.get(id) {
                Some(&first_line) => {
                    anomalies.push(RecordAnomaly {
                        line,
                        field: "id",
                        code: AnomalyCode::DuplicateId { first_line },
                    });
                    record_has_error = true;
                }
                None => {
                    if other_fields_ok {
                        seen_ids.insert(id.clone(), line);
                    }
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

        // フィールド検査（id・input・output・tags・group_id・重複 id）で
        // 1 件以上の異常が積まれていれば、この行を「異常を出した行」として
        // 1 回だけ数える（`MalformedJson`・`MalformedRecord`・`DuplicateKey` は
        // 上の早期 `continue` 側で既に数えているため、ここには到達しない）。
        if anomalies.len() != anomalies_before_this_line {
            anomalous_rows += 1;
        }
    }

    let report = report::summarize(total_rows, anomalous_rows, &valid_records, valid_label_ids);

    Ok(InspectOutcome {
        anomalies,
        valid_records,
        report,
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
                field: "output.intent",
                code: AnomalyCode::MissingField,
            }]
        );
    }

    /// P0 修正の回帰確認（レビュー指摘）: 未知のラベル値そのものは
    /// `RecordAnomaly`/`AnomalyCode` のどこにも保持されないこと。
    #[test]
    fn unknown_label_is_reported_without_raw_value() {
        let content = "{\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"nope\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                field: "output.intent",
                code: AnomalyCode::UnknownLabel,
            }]
        );
        assert!(outcome.valid_records.is_empty());
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

    /// P1 修正の回帰確認（レビュー指摘）: 重複 `id` の 2 回目以降の出現は
    /// `valid_records` から除外され、`id` の一意性が保たれること。
    #[test]
    fn duplicate_id_is_reported_and_excluded_from_valid_records() {
        let content = "\
{\"id\":\"r1\",\"input\":\"a\",\"output\":{\"intent\":\"ok\"}}
{\"id\":\"r1\",\"input\":\"b\",\"output\":{\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 2,
                field: "id",
                code: AnomalyCode::DuplicateId { first_line: 1 },
            }]
        );
        // 初出（1 行目）のみ valid_records に残り、重複（2 行目）は除外される。
        assert_eq!(outcome.valid_records.len(), 1);
        assert_eq!(outcome.valid_records[0].line, 1);
        assert_eq!(outcome.valid_records[0].input, "a".to_string());
    }

    /// P1 修正の回帰確認（PR #191 レビュー指摘・codex/review）: `input` 欠落等で
    /// 無効な初出レコードの `id` を `seen_ids` へ登録してはならない。登録すると
    /// 後続の同一 `id` を持つ妥当なレコードまで `DuplicateId` として誤って
    /// `valid_records` から除外され、REQ-16 の妥当なレコード抽出を壊す。
    #[test]
    fn invalid_first_record_id_does_not_block_later_valid_record_with_same_id() {
        let content = "\
{\"id\":\"r1\",\"output\":{\"intent\":\"ok\"}}
{\"id\":\"r1\",\"input\":\"b\",\"output\":{\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        // 1 行目は input 欠落で無効（MissingField 相当）であり、DuplicateId は
        // 発生しない（2 行目は妥当なレコードとして残るべきである）。
        assert!(
            outcome
                .anomalies
                .iter()
                .all(|a| !matches!(a.code, AnomalyCode::DuplicateId { .. })),
            "無効な初出レコードの id が重複判定に使われてはならない: {:?}",
            outcome.anomalies
        );
        assert_eq!(outcome.valid_records.len(), 1);
        assert_eq!(outcome.valid_records[0].line, 2);
        assert_eq!(outcome.valid_records[0].id, "r1".to_string());
        assert_eq!(outcome.valid_records[0].input, "b".to_string());
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

    /// 検証に失敗した `output.intent` の生値（極端に長い文字列）が
    /// `RecordAnomaly`/`AnomalyCode` のどこにも伝播しないこと（P0 修正の回帰確認）。
    #[test]
    fn unknown_label_raw_value_never_reaches_diagnostics() {
        let long_label = "x".repeat(500);
        let content =
            format!("{{\"id\":\"r1\",\"input\":\"x\",\"output\":{{\"intent\":\"{long_label}\"}}}}");
        let valid = labels(&["ok"]);

        let outcome = inspect_records(&content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                field: "output.intent",
                code: AnomalyCode::UnknownLabel,
            }]
        );
    }

    /// 長い `id`（学習・評価データ利用者が自由に設定する値）が重複しても、
    /// その生値が `RecordAnomaly` に伝播しないこと（P0 修正の回帰確認）。
    /// `ValidRecord` 側は検証を通過した実値をそのまま保持する。
    #[test]
    fn duplicate_id_raw_value_never_reaches_diagnostics() {
        let long_id = "y".repeat(500);
        let content = format!(
            "{{\"id\":\"{long_id}\",\"input\":\"a\",\"output\":{{\"intent\":\"ok\"}}}}\n\
             {{\"id\":\"{long_id}\",\"input\":\"b\",\"output\":{{\"intent\":\"ok\"}}}}"
        );
        let valid = labels(&["ok"]);

        let outcome = inspect_records(&content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 2,
                field: "id",
                code: AnomalyCode::DuplicateId { first_line: 1 },
            }]
        );
        // 初出のみ valid_records に残り、実値（分割・突き合わせのキー）を保持する。
        assert_eq!(outcome.valid_records.len(), 1);
        assert_eq!(outcome.valid_records[0].id, long_id);
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
                    field: "id",
                    code: AnomalyCode::TypeMismatch {
                        expected: "string",
                        actual: "number",
                    },
                },
                RecordAnomaly {
                    line: 1,
                    field: "input",
                    code: AnomalyCode::MissingField,
                },
                RecordAnomaly {
                    line: 1,
                    field: "output.intent",
                    code: AnomalyCode::UnknownLabel,
                },
                RecordAnomaly {
                    line: 1,
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
        assert_eq!(AnomalyCode::EmptyValue.code(), "empty_value");
        assert_eq!(AnomalyCode::UnknownLabel.code(), "unknown_label");
        assert_eq!(
            AnomalyCode::DuplicateId { first_line: 1 }.code(),
            "duplicate_id"
        );
        assert_eq!(AnomalyCode::DuplicateKey.code(), "duplicate_key");
    }

    /// REQ-16・TASK-16.1-1: `id`・`input`・`group_id` が空文字列の場合は
    /// それぞれ `EmptyValue` として検出され、`valid_records` から除外される。
    #[test]
    fn empty_id_input_group_id_are_reported() {
        let valid = labels(&["ok"]);

        let id_outcome = inspect_records(
            "{\"id\":\"\",\"input\":\"x\",\"output\":{\"intent\":\"ok\"}}",
            &valid,
        )
        .unwrap();
        assert_eq!(
            id_outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                field: "id",
                code: AnomalyCode::EmptyValue,
            }]
        );
        assert!(id_outcome.valid_records.is_empty());

        let input_outcome = inspect_records(
            "{\"id\":\"r1\",\"input\":\"\",\"output\":{\"intent\":\"ok\"}}",
            &valid,
        )
        .unwrap();
        assert_eq!(
            input_outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                field: "input",
                code: AnomalyCode::EmptyValue,
            }]
        );
        assert!(input_outcome.valid_records.is_empty());

        let group_id_outcome = inspect_records(
            "{\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"ok\"},\"group_id\":\"\"}",
            &valid,
        )
        .unwrap();
        assert_eq!(
            group_id_outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                field: "group_id",
                code: AnomalyCode::EmptyValue,
            }]
        );
        assert!(group_id_outcome.valid_records.is_empty());
    }

    /// REQ-16・TASK-16.1-1: 空文字列の `id` は突き合わせのキーになり得ないため、
    /// 複数行に現れても `DuplicateId` にはせず、行ごとに独立して `EmptyValue`
    /// を報告する（挙動節の規則そのものの確認）。
    #[test]
    fn empty_id_across_multiple_lines_is_not_duplicate_id() {
        let content = "\
{\"id\":\"\",\"input\":\"a\",\"output\":{\"intent\":\"ok\"}}
{\"id\":\"\",\"input\":\"b\",\"output\":{\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![
                RecordAnomaly {
                    line: 1,
                    field: "id",
                    code: AnomalyCode::EmptyValue,
                },
                RecordAnomaly {
                    line: 2,
                    field: "id",
                    code: AnomalyCode::EmptyValue,
                },
            ]
        );
        assert!(outcome.valid_records.is_empty());
    }

    /// P1 修正の回帰確認（レビュー指摘。PR #191）: トップレベルの `id` が
    /// 重複しているレコードは、`serde_json::Value` パース時点の後勝ちで
    /// 妥当なレコードとして混入せず、`DuplicateKey` として検出されること。
    #[test]
    fn duplicate_top_level_key_is_reported_and_excluded_from_valid_records() {
        let content = "{\"id\":1,\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                field: "<record>",
                code: AnomalyCode::DuplicateKey,
            }]
        );
        assert!(outcome.valid_records.is_empty());
    }

    /// P1 修正の回帰確認: ネスト先（`output.intent`）の重複キーも検出されること。
    #[test]
    fn duplicate_nested_key_is_reported_and_excluded_from_valid_records() {
        let content =
            "{\"id\":\"r1\",\"input\":\"x\",\"output\":{\"intent\":\"nope\",\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                field: "<record>",
                code: AnomalyCode::DuplicateKey,
            }]
        );
        assert!(outcome.valid_records.is_empty());
    }

    /// P1 修正の回帰確認: Unicode エスケープで書かれた重複キー
    /// （`"id"` と `"\u0069d"` はいずれも文字列としては `"id"`）も検出されること。
    #[test]
    fn duplicate_key_via_unicode_escape_is_reported() {
        let content =
            "{\"id\":\"a\",\"\\u0069d\":\"b\",\"input\":\"x\",\"output\":{\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert_eq!(
            outcome.anomalies,
            vec![RecordAnomaly {
                line: 1,
                field: "<record>",
                code: AnomalyCode::DuplicateKey,
            }]
        );
        assert!(outcome.valid_records.is_empty());
    }

    /// 誤検出防止の対照実験: 文字列値の中に `:`・`"` を含むレコードは
    /// （重複キーが実際には無いため）異常として検出されないこと
    /// （`count_raw_key_value_separators` の文字列スキップが正しく働く確認）。
    #[test]
    fn colon_inside_string_values_is_not_a_false_positive() {
        let content =
            "{\"id\":\"a:b\\\"c:d\",\"input\":\"{\\\"x\\\":1}\",\"output\":{\"intent\":\"ok\"}}";
        let valid = labels(&["ok"]);

        let outcome = inspect_records(content, &valid).unwrap();

        assert!(outcome.anomalies.is_empty());
        assert_eq!(outcome.valid_records.len(), 1);
        assert_eq!(outcome.valid_records[0].id, "a:b\"c:d");
        assert_eq!(outcome.valid_records[0].input, "{\"x\":1}");
    }
}
