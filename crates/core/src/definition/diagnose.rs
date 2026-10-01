//! `Definition::parse` の失敗経路専用の診断（TASK-15.3-2・REQ-15）。
//!
//! `serde_json::from_str::<RawDefinition>` が `Category::Data`（構文エラー
//! ではなく型・必須項目の不整合）で失敗したときにだけ呼ばれる。`Value` へ
//! 読み直して固定順（`super::FieldPath` の doc を参照）で走査し、最初に
//! 見つかった構造エラーを型付きの [`super::DefinitionError`] で返す。
//! 何も見つからない場合（重複キー等、`Value` 側では検出できない不整合）は
//! `None` を返し、呼び出し元（`Definition::parse`）が元の `Parse { source }`
//! を返す（fail-closed。`Value` へ丸ごと寄せて重複キーを後勝ちで受理する
//! 回帰を避ける設計判断は `Definition::parse` の doc を参照）。
//!
//! 資源上限（REQ-39）: 呼び出し元でサイズ検査（`MAX_DEFINITION_FILE_BYTES`）
//! 済みの入力にのみ適用され、`serde_json` の既定の再帰上限（128）がネスト
//! 攻撃を抑える。`Value` の走査は `get()`/`as_*()` のみで行い、添字 `[]`・
//! `unwrap`・`expect` は使わない（`.claude/rules/coding-rust.md`）。

use super::{ChoiceField, DefinitionError, ExpectedType, FieldPath, JsonType};
use serde_json::Value;

/// `judgment_type` の許可値（`super::JudgmentType` の serde 表現と一致する
/// ことをユニットテストで固定する）。
const JUDGMENT_TYPE_SINGLE_SELECT: &str = "single_select";

/// `io.input` の許可値（`super::InputRepresentation` の serde 表現と一致
/// することをユニットテストで固定する）。
const INPUT_REPRESENTATION_BYTES: &str = "bytes";

/// `text` を `Value` として読み直し、最初に見つかった構造エラーを返す。
/// `text` の構文自体が不正な場合（`Value` へのパースにも失敗する場合）は
/// 呼び出し元が既に `Category::Syntax`/`Eof` として弾いているため、ここには
/// 到達しない想定だが、防御的に `None` を返す（fail-closed で `Parse` に
/// 委ねる）。
pub(super) fn diagnose(text: &str) -> Option<DefinitionError> {
    let value: Value = serde_json::from_str(text).ok()?;

    // 1. ルートが object であること（本関数内の走査順のステップ 1）。
    // 非 object の場合はここで確定的に `TypeMismatch(Root, Object, actual)`
    // を返す（`?` で無言の `None` に落とさない）。
    if let Some(err) = check_root_is_object(&value) {
        return Some(err);
    }
    let root = value.as_object()?;

    // 2. schema（誤誘導を避けるため最優先）
    let schema_value = root.get("schema");
    if let Some(err) = check_required_string(schema_value, FieldPath::Schema) {
        return Some(err);
    }
    if let Some(Value::String(schema)) = schema_value
        && schema != super::SCHEMA_ID
    {
        return Some(DefinitionError::UnsupportedSchema {
            schema: schema.clone(),
        });
    }

    // 3. トップレベル必須フィールド（宣言順）
    if let Some(err) = check_required_string(root.get("name"), FieldPath::Name) {
        return Some(err);
    }

    match root.get("version") {
        None => {
            return Some(DefinitionError::MissingField {
                field: FieldPath::Version,
            });
        }
        Some(version_value) => {
            if version_value
                .as_u64()
                .is_none_or(|v| v > u64::from(u32::MAX))
            {
                return Some(DefinitionError::TypeMismatch {
                    field: FieldPath::Version,
                    expected: ExpectedType::UnsignedInt32,
                    actual: json_type(version_value),
                });
            }
        }
    }

    match root.get("judgment_type") {
        None => {
            return Some(DefinitionError::MissingField {
                field: FieldPath::JudgmentType,
            });
        }
        Some(Value::String(s)) => {
            if s != JUDGMENT_TYPE_SINGLE_SELECT {
                return Some(DefinitionError::UnsupportedValue {
                    field: FieldPath::JudgmentType,
                });
            }
        }
        Some(other) => {
            return Some(DefinitionError::TypeMismatch {
                field: FieldPath::JudgmentType,
                expected: ExpectedType::String,
                actual: json_type(other),
            });
        }
    }

    // `options` キー自体の欠落は「ラベル定義が同梱されていない」
    // （TASK-15.4・REQ-15 異常系。PoC-9 追補 v1.1 A-6）として `MissingLabels`
    // を返す。`schema`/`name`/`version`/`judgment_type` の検査を通過した
    // 後に判定するため、これらの不整合が優先される（本関数の固定走査順）。
    // `options` キーはあるが値が空配列・不正な型のケースは区別し、後続の
    // `as_array()` チェックと `check_option_entry` 呼び出し後の経路
    // （`Definition::parse` 側の `EmptyOptions` 判定）に委ねる。
    let options_value = match root.get("options") {
        None => {
            return Some(DefinitionError::MissingLabels);
        }
        Some(v) => v,
    };
    let options = match options_value.as_array() {
        None => {
            return Some(DefinitionError::TypeMismatch {
                field: FieldPath::Options,
                expected: ExpectedType::Array,
                actual: json_type(options_value),
            });
        }
        Some(a) => a,
    };

    // 4. options の各要素（index 昇順）
    for (index, entry) in options.iter().enumerate() {
        if let Some(err) = check_option_entry(index, entry) {
            return Some(err);
        }
    }

    // 5. io
    let io_value = match root.get("io") {
        None => {
            return Some(DefinitionError::MissingField {
                field: FieldPath::Io,
            });
        }
        Some(v) => v,
    };
    let io_object = match io_value.as_object() {
        None => {
            return Some(DefinitionError::TypeMismatch {
                field: FieldPath::Io,
                expected: ExpectedType::Object,
                actual: json_type(io_value),
            });
        }
        Some(o) => o,
    };
    match io_object.get("input") {
        None => {
            return Some(DefinitionError::MissingField {
                field: FieldPath::IoInput,
            });
        }
        Some(Value::String(s)) => {
            if s != INPUT_REPRESENTATION_BYTES {
                return Some(DefinitionError::UnsupportedValue {
                    field: FieldPath::IoInput,
                });
            }
        }
        Some(other) => {
            return Some(DefinitionError::TypeMismatch {
                field: FieldPath::IoInput,
                expected: ExpectedType::String,
                actual: json_type(other),
            });
        }
    }
    for key in io_object.keys() {
        if key != "input" {
            return Some(DefinitionError::UnknownField {
                parent: FieldPath::Io,
                name: key.clone(),
            });
        }
    }

    // 5b. acceptance（省略可能。#328）。キーがあれば object で、
    // `min_accuracy_bp` が必須の u32。`null` は未定義扱いにせず型エラーにする。
    if let Some(acceptance_value) = root.get("acceptance") {
        let acceptance_object = match acceptance_value.as_object() {
            None => {
                return Some(DefinitionError::TypeMismatch {
                    field: FieldPath::Acceptance,
                    expected: ExpectedType::Object,
                    actual: json_type(acceptance_value),
                });
            }
            Some(o) => o,
        };
        match acceptance_object.get("min_accuracy_bp") {
            None => {
                return Some(DefinitionError::MissingField {
                    field: FieldPath::AcceptanceMinAccuracyBp,
                });
            }
            Some(bp_value) => {
                if bp_value.as_u64().is_none_or(|v| v > u64::from(u32::MAX)) {
                    return Some(DefinitionError::TypeMismatch {
                        field: FieldPath::AcceptanceMinAccuracyBp,
                        expected: ExpectedType::UnsignedInt32,
                        actual: json_type(bp_value),
                    });
                }
            }
        }
        for key in acceptance_object.keys() {
            if key != "min_accuracy_bp" {
                return Some(DefinitionError::UnknownField {
                    parent: FieldPath::Acceptance,
                    name: key.clone(),
                });
            }
        }
    }

    // 6. トップレベルの未知キー（`serde_json::Map` は既定で `BTreeMap` の
    // ためキー順走査は決定的）。
    const KNOWN_TOP_LEVEL_KEYS: [&str; 7] = [
        "schema",
        "name",
        "version",
        "judgment_type",
        "options",
        "io",
        "acceptance",
    ];
    for key in root.keys() {
        if !KNOWN_TOP_LEVEL_KEYS.contains(&key.as_str()) {
            return Some(DefinitionError::UnknownField {
                parent: FieldPath::Root,
                name: key.clone(),
            });
        }
    }

    None
}

/// ルートが object であることを確認する。object でなければ
/// `TypeMismatch { Root, Object, actual }` を返す。
fn check_root_is_object(value: &Value) -> Option<DefinitionError> {
    if value.as_object().is_some() {
        None
    } else {
        Some(DefinitionError::TypeMismatch {
            field: FieldPath::Root,
            expected: ExpectedType::Object,
            actual: json_type(value),
        })
    }
}

/// `field` が欠落していれば `MissingField`、文字列でなければ
/// `TypeMismatch { expected: String }` を返す。文字列であれば `None`
/// （空文字列の判定は呼び出し元〔`name` は `EmptyName`〕に委ねる）。
fn check_required_string(value: Option<&Value>, field: FieldPath) -> Option<DefinitionError> {
    match value {
        None => Some(DefinitionError::MissingField { field }),
        Some(Value::String(_)) => None,
        Some(other) => Some(DefinitionError::TypeMismatch {
            field,
            expected: ExpectedType::String,
            actual: json_type(other),
        }),
    }
}

/// `options[index]` の 1 要素を検査する。
///
/// `options` キー自体の欠落は呼び出し元（`diagnose` 内、本関数の呼び出しより
/// 前）で `DefinitionError::MissingLabels`（TASK-15.4）として検出済みで、
/// ここには到達しない。本関数は `options` が配列として存在する場合の各要素の
/// 必須項目のみを見る。
fn check_option_entry(index: usize, entry: &Value) -> Option<DefinitionError> {
    let object = match entry.as_object() {
        None => {
            return Some(DefinitionError::TypeMismatch {
                field: FieldPath::OptionEntry { index },
                expected: ExpectedType::Object,
                actual: json_type(entry),
            });
        }
        Some(o) => o,
    };

    for (key, choice_field) in [
        ("id", ChoiceField::Id),
        ("display_name", ChoiceField::DisplayName),
        ("description", ChoiceField::Description),
    ] {
        if let Some(err) = check_required_string(
            object.get(key),
            FieldPath::OptionField {
                index,
                field: choice_field,
            },
        ) {
            return Some(err);
        }
    }

    const KNOWN_OPTION_KEYS: [&str; 3] = ["id", "display_name", "description"];
    for key in object.keys() {
        if !KNOWN_OPTION_KEYS.contains(&key.as_str()) {
            return Some(DefinitionError::UnknownField {
                parent: FieldPath::OptionEntry { index },
                name: key.clone(),
            });
        }
    }

    None
}

/// `Value` の実行時の型を [`JsonType`] へ写す。
fn json_type(value: &Value) -> JsonType {
    match value {
        Value::Null => JsonType::Null,
        Value::Bool(_) => JsonType::Bool,
        Value::Number(_) => JsonType::Number,
        Value::String(_) => JsonType::String,
        Value::Array(_) => JsonType::Array,
        Value::Object(_) => JsonType::Object,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Definition::parse` の正常系テスト（`TWO_OPTIONS_JSON`）と同一構成の
    /// 定義。ここに `KNOWN_TOP_LEVEL_KEYS`/`KNOWN_OPTION_KEYS` が
    /// `RawDefinition`/`Choice` のフィールドと食い違っていないかを
    /// 固定するための正常系専用の複製を持つ（フィールド追加時に
    /// これらの一覧を更新し忘れると本テストが失敗して顕在化する）。
    const VALID_DEFINITION_JSON: &str = r#"{
        "schema": "fandhe-edge-model-definition/v1",
        "name": "sample_topic",
        "version": 1,
        "judgment_type": "single_select",
        "options": [
            { "id": "yes", "display_name": "Yes", "description": "肯定" },
            { "id": "no", "display_name": "No", "description": "否定" }
        ],
        "io": { "input": "bytes" }
    }"#;

    /// diagnose の許可値定数が `JudgmentType`/`InputRepresentation` の serde
    /// 表現と食い違わないことを固定する（本ファイル冒頭の doc）。
    #[test]
    fn allowed_values_match_serde_representation() {
        let judgment_type_json =
            serde_json::to_string(&super::super::JudgmentType::SingleSelect).unwrap();
        assert_eq!(
            judgment_type_json,
            format!("{JUDGMENT_TYPE_SINGLE_SELECT:?}")
        );

        let input_json = serde_json::to_string(&super::super::InputRepresentation::Bytes).unwrap();
        assert_eq!(input_json, format!("{INPUT_REPRESENTATION_BYTES:?}"));
    }

    /// 正常系定義に対して `diagnose` が `None` を返すことを直接検証する。
    /// `KNOWN_TOP_LEVEL_KEYS`/`KNOWN_OPTION_KEYS` のドリフト（フィールド
    /// 追加の反映漏れ）を、他のエラー経路を介さず本テスト単体で検出する。
    #[test]
    fn diagnose_returns_none_for_valid_definition() {
        assert!(diagnose(VALID_DEFINITION_JSON).is_none());
    }

    #[test]
    fn diagnose_reports_type_mismatch_for_non_string_schema() {
        let mut value: Value = serde_json::from_str(VALID_DEFINITION_JSON).unwrap();
        value["schema"] = serde_json::json!(1);
        let err = diagnose(&value.to_string()).expect("診断結果があるはず");
        assert!(matches!(
            err,
            DefinitionError::TypeMismatch {
                field: FieldPath::Schema,
                expected: ExpectedType::String,
                actual: JsonType::Number,
            }
        ));
    }

    #[test]
    fn diagnose_reports_type_mismatch_for_non_string_judgment_type() {
        let mut value: Value = serde_json::from_str(VALID_DEFINITION_JSON).unwrap();
        value["judgment_type"] = serde_json::json!(true);
        let err = diagnose(&value.to_string()).expect("診断結果があるはず");
        assert!(matches!(
            err,
            DefinitionError::TypeMismatch {
                field: FieldPath::JudgmentType,
                expected: ExpectedType::String,
                actual: JsonType::Bool,
            }
        ));
    }

    #[test]
    fn diagnose_reports_type_mismatch_for_non_object_io() {
        let mut value: Value = serde_json::from_str(VALID_DEFINITION_JSON).unwrap();
        value["io"] = serde_json::json!("bytes");
        let err = diagnose(&value.to_string()).expect("診断結果があるはず");
        assert!(matches!(
            err,
            DefinitionError::TypeMismatch {
                field: FieldPath::Io,
                expected: ExpectedType::Object,
                actual: JsonType::String,
            }
        ));
    }

    #[test]
    fn diagnose_reports_type_mismatch_for_non_string_io_input() {
        let mut value: Value = serde_json::from_str(VALID_DEFINITION_JSON).unwrap();
        value["io"]["input"] = serde_json::json!(1);
        let err = diagnose(&value.to_string()).expect("診断結果があるはず");
        assert!(matches!(
            err,
            DefinitionError::TypeMismatch {
                field: FieldPath::IoInput,
                expected: ExpectedType::String,
                actual: JsonType::Number,
            }
        ));
    }

    #[test]
    fn diagnose_reports_unknown_field_for_io_extra_key() {
        let mut value: Value = serde_json::from_str(VALID_DEFINITION_JSON).unwrap();
        value["io"]["extra"] = serde_json::json!("unexpected");
        let err = diagnose(&value.to_string()).expect("診断結果があるはず");
        match err {
            DefinitionError::UnknownField { parent, name } => {
                assert_eq!(parent, FieldPath::Io);
                assert_eq!(name, "extra");
            }
            other => panic!("UnknownField を期待したが {other:?} だった"),
        }
    }
}
