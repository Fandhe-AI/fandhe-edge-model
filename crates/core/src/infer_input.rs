//! 推論入力レコードの検証（REQ-21 異常系・REQ-39・TASK-21.2）。
//!
//! # 呼び出し文脈
//!
//! 呼び出し元（想定・TASK-33.1 で配線）: CLI の `infer` サブコマンドが
//! `--text`／`--input-file`（1 行 1 JSON）から読んだ 1 件分の文字列を
//! [`InferInput::parse`] へ渡す。本モジュール自体はファイル I/O・argv 解析
//! を行わない（層の境界を保つ。`.claude/rules/coding-rust.md`「操作アダプ
//! ターは薄く保ち、業務ロジックは下位層に置く」）。argv の非 UTF-8
//! （`OsString`）の扱いは引数解析と一体のため TASK-33.1 の対象とし、本
//! モジュールでは扱わない。
//!
//! 呼び出し先（想定）: 検証を通った [`InferInput`] は推論ランタイム
//! （TASK-30.x/31.x。未実装）へ `input()` のみを渡す想定（評価契約
//! 「推論関数には `input` だけを渡す」。`.claude/rules/evaluation-contract.md`）。
//!
//! # 「未知の入力形式・型不正」の解釈
//!
//! 入力表現は byte のみ（README「実装方針（要点）」）で、バイトエンコーダ
//! は 0〜255 の全値を扱うためバイト語彙の未知（OOV）は発生しない。した
//! がって本モジュールが検出する「未知の入力形式」はレコード／リクエスト
//! の水準（JSON として不正・ルートが object でない・必須フィールドの欠
//! 落・未知フィールド・型不一致）で解釈する。
//!
//! 空・空白のみの `input` は**有効な入力**として扱う（拒否しない）。
//! TASK-23.2（#212）で正規化後に空になる `input` はバイト列 `[0]` へエン
//! コードされる契約が既に確定しているため、本モジュールがそれを矛盾させ
//! ない。
//!
//! # PoC-16 との違い（回帰防止）
//!
//! PoC-16 の CLI は `row["input"].as_str().unwrap_or("")` で入力を読んで
//! いたため、型が不正な入力（数値・配列・object 等）が黙って空文字列へ
//! 変換されて推論されていた。これは評価契約「型の正しさと意味の正しさは
//! 別々に数える」（`type_meaning_quadrant`）に反する。本モジュールは型不
//! 正を [`InferInputError::TypeMismatch`] として明示的に拒否し、黙った変
//! 換をしない。
//!
//! # 検証（fail-closed）
//!
//! [`InferInput`] は `Deserialize` を実装せず、フィールドを非公開にし、
//! 唯一の構築経路を [`InferInput::parse`] に限定する（`definition.rs` の
//! `Definition`／`RawDefinition` と同じ設計。security.md「ガード層の迂
//! 回」）。

use crate::definition::{InputRepresentation, IoSchema};
use crate::judgment::MAX_INPUT_ID_BYTES;
use serde::Deserialize;
use std::fmt;

/// 推論入力 1 件分（1 行の JSON）の読み込み時サイズ上限（暫定値。REQ-39
/// の資源上限が正式に決まり次第、値を見直す）。`serde_json::from_str` へ
/// 渡す前に検査し、上限超過の入力に対して事前にアロケーションしない
/// （coding-rust.md「サイズ・件数を上限検証してからアロケーションに使
/// う」。`definition.rs` の `MAX_DEFINITION_FILE_BYTES` と同じ理由）。
pub const MAX_INFER_INPUT_BYTES: usize = 1_048_576;

/// 推論入力レコード内のフィールドの位置を表すパス。自由文字列ではなく
/// enum にすることで、でっち上げのパスを表現できない型にする
/// （coding-rust.md「公開 API・型設計」。`definition.rs` の `FieldPath` と
/// 同じ方針）。値そのものは保持しない（security.md「秘密情報の混入防
/// 止」）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InferInputField {
    /// JSON ドキュメントのルート。`InferInputError::NotObject` はルート非
    /// object を `actual: JsonType` のみで表すため現状は未使用だが、
    /// `definition.rs` の `FieldPath::Root` と語彙を揃えるため enum に含め
    /// ておく（将来ルート直下以外の位置を指すエラーを追加する際の予約）。
    Root,
    Id,
    Input,
}

impl fmt::Display for InferInputField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferInputField::Root => write!(f, "$"),
            InferInputField::Id => write!(f, "id"),
            InferInputField::Input => write!(f, "input"),
        }
    }
}

/// 検証済みの推論入力 1 件（識別子・入力本文）。
///
/// フィールドは非公開。検証を通る唯一の構築経路は [`InferInput::parse`]
/// に限定する（モジュール冒頭のドキュメント参照）。
///
/// `Debug` は `#[derive]` せず手書きする。`input`（推論データ本文）・`id`
/// はいずれも利用者入力で個人情報・機密情報を含みうるため、`{:?}` による
/// 診断表示（ログ等）でそのまま転記されないよう値を伏せる
/// （security.md「秘密情報の混入防止（P0）」。PR #217 レビュー指摘）。
#[derive(Clone, PartialEq, Eq)]
pub struct InferInput {
    id: String,
    input: String,
}

impl fmt::Debug for InferInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferInput")
            .field("id", &"<redacted>")
            .field("input", &"<redacted>")
            .finish()
    }
}

/// `InferInput` の未検証の中間表現（デシリアライズ専用）。
///
/// `InferInput` 自体には `Deserialize` を実装しない。実装すると
/// `serde_json::from_str::<InferInput>` のように `parse` を経由しない直接
/// デシリアライズが可能になり、`parse` が担う検証を丸ごと迂回できてしま
/// う（security.md「ガード層の迂回」。`definition.rs` の `RawDefinition`
/// と同じ理由）。`#[serde(deny_unknown_fields)]` により未知フィールドは
/// serde の時点で拒否される（`未知の入力形式`）。serde の struct デシリア
/// ライズは重複フィールドを `duplicate field` エラーとして拒否するため
/// （`serde_json::Error::classify() == Data`）、後勝ちで受理することはな
/// い。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInferInput {
    id: String,
    input: String,
}

/// [`InferInput::parse`] が拒否する入力の種類。
///
/// 入力本文・`id` の値・未知フィールドのキー名を保持しない（security.md
/// 「秘密情報の混入防止」: 学習・評価・推論データの本文をエラーへ転記し
/// ない）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InferInputError {
    /// 入力レコード（1 行）が [`MAX_INFER_INPUT_BYTES`] を超える（REQ-39
    /// 資源の上限）。`serde_json::from_str` へ渡す前に検査する。
    TooLarge { len: usize, limit: usize },
    /// JSON として構文不正（`serde_json::Error::classify()` が
    /// `Syntax`/`Eof`/`Io`）。
    MalformedJson,
    /// ルートが object でない（配列・文字列・数値・真偽値・`null`）。
    NotObject { actual: crate::definition::JsonType },
    /// 必須フィールド（`id`／`input`）が欠落している。
    MissingField { field: InferInputField },
    /// フィールドの型が期待と異なる（値そのものは保持しない）。
    TypeMismatch {
        field: InferInputField,
        expected: crate::definition::ExpectedType,
        actual: crate::definition::JsonType,
    },
    /// スキーマが許可しない未知のキーを含む（`deny_unknown_fields`）。キー
    /// 名は保持しない（`definition.rs` の `UnknownField` と異なり、本レコ
    /// ードは利用者が直接投入する推論入力で、キー名自体が入力本文の一部
    /// になりうるため。security.md「秘密情報の混入防止」）。
    UnknownField,
    /// 上記のいずれにも分類できない構造的な不正（例: 重複フィールド。
    /// `serde_json::Value` への再走査でも判別できない不整合）。fail-closed
    /// でここに落とす（`definition.rs` の `Parse { .. }` フォールバックと
    /// 同じ方針）。
    Malformed,
    /// `id` が空文字列。
    EmptyId,
    /// `id` が [`crate::judgment::MAX_INPUT_ID_BYTES`] を超える（REQ-39
    /// 資源の上限超過そのものだが、終了コードは
    /// `judgment::JudgmentError::InputIdTooLong` に揃え `InvalidInput`
    /// （64）とする。`exit_code()` のドキュメント参照）。
    IdTooLong { len: usize, limit: usize },
}

impl fmt::Display for InferInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferInputError::TooLarge { len, limit } => {
                write!(f, "infer input too large: {len} bytes (limit: {limit})")
            }
            InferInputError::MalformedJson => write!(f, "infer input is not valid JSON"),
            InferInputError::NotObject { actual } => {
                write!(f, "infer input root must be an object, got {actual}")
            }
            InferInputError::MissingField { field } => {
                write!(f, "missing required field: {field}")
            }
            InferInputError::TypeMismatch {
                field,
                expected,
                actual,
            } => {
                write!(f, "field {field} must be {expected}, got {actual}")
            }
            InferInputError::UnknownField => write!(f, "infer input contains an unknown field"),
            InferInputError::Malformed => write!(f, "infer input is malformed"),
            InferInputError::EmptyId => write!(f, "id must not be empty"),
            InferInputError::IdTooLong { len, limit } => {
                write!(f, "id too long: {len} bytes (limit: {limit})")
            }
        }
    }
}

impl std::error::Error for InferInputError {}

impl InferInputError {
    /// REQ-21 の終了コードへの写像。網羅 `match`（`_ =>` 禁止）で書き、新
    /// variant 追加時にコンパイルエラーで分類漏れに気付けるようにする
    /// （`exitcode.rs`・`judgment.rs` の `exit_code()` と同じ方針）。
    ///
    /// - [`InferInputError::TooLarge`] → `LimitExceeded`（20。REQ-39 資源
    ///   の上限超過そのもの。`serde_json::from_str` へ渡す前段の入力バイ
    ///   ト数の上限）
    /// - [`InferInputError::IdTooLong`] は
    ///   `judgment::JudgmentError::InputIdTooLong`（`id` の長さ上限超過を
    ///   `InvalidInput` に分類する既存の先例）に一貫性を揃え、
    ///   `InvalidInput`（64）とする
    /// - それ以外はすべて `InvalidInput`（64）
    #[must_use]
    pub const fn exit_code(&self) -> crate::exitcode::ExitCode {
        match self {
            InferInputError::TooLarge { .. } => crate::exitcode::ExitCode::LimitExceeded,
            InferInputError::MalformedJson
            | InferInputError::NotObject { .. }
            | InferInputError::MissingField { .. }
            | InferInputError::TypeMismatch { .. }
            | InferInputError::UnknownField
            | InferInputError::Malformed
            | InferInputError::EmptyId
            | InferInputError::IdTooLong { .. } => crate::exitcode::ExitCode::InvalidInput,
        }
    }

    /// JSON `code`／ログ用の機械可読な理由コード（`definition.rs` の
    /// `reason_code()` と同じ語彙の付け方）。
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            InferInputError::TooLarge { .. } => "too_large",
            InferInputError::MalformedJson => "malformed_json",
            InferInputError::NotObject { .. } => "not_object",
            InferInputError::MissingField { .. } => "missing_field",
            InferInputError::TypeMismatch { .. } => "type_mismatch",
            InferInputError::UnknownField => "unknown_field",
            InferInputError::Malformed => "malformed",
            InferInputError::EmptyId => "empty_id",
            InferInputError::IdTooLong { .. } => "id_too_long",
        }
    }
}

impl InferInput {
    /// JSON 1 行（1 件分の推論入力）をパースし検証する。
    ///
    /// `io.input` は現状 [`InputRepresentation::Bytes`] の 1 variant のみ
    /// のため網羅 `match` は自明だが、将来 variant が追加されたら
    /// コンパイルエラーで気付けるよう `_ =>` を使わずに書く。将来 byte 以
    /// 外の表現を追加する場合は、ここで未対応表現を検出して
    /// `InferInputError`（例: 追加予定の `UnsupportedInputRepresentation`
    /// → `InvalidInput`）として拒否する想定だが、その variant は現状の
    /// スキーマには存在しないため本 TASK では追加しない（実装済みを装わ
    /// ない。coding-rust.md「未実装・簡易実装の箇所は実装済みを装わな
    /// い」）。
    ///
    /// # Errors
    /// [`InferInputError`] の各 variant を参照。
    pub fn parse(line: &str, io: &IoSchema) -> Result<Self, InferInputError> {
        // 資源の上限（REQ-39）: `serde_json::from_str` によるアロケーショ
        // ン・パースより前に、入力バイト数を検証する（`definition.rs` の
        // `Definition::parse` と同じ順序）。
        let len = line.len();
        if len > MAX_INFER_INPUT_BYTES {
            return Err(InferInputError::TooLarge {
                len,
                limit: MAX_INFER_INPUT_BYTES,
            });
        }

        match io.input {
            InputRepresentation::Bytes => {}
        }

        let raw: RawInferInput = match serde_json::from_str(line) {
            Ok(raw) => raw,
            Err(source) => {
                return Err(match source.classify() {
                    serde_json::error::Category::Syntax
                    | serde_json::error::Category::Eof
                    | serde_json::error::Category::Io => InferInputError::MalformedJson,
                    serde_json::error::Category::Data => Self::diagnose_data_error(line),
                });
            }
        };

        if raw.id.is_empty() {
            return Err(InferInputError::EmptyId);
        }
        if raw.id.len() > MAX_INPUT_ID_BYTES {
            return Err(InferInputError::IdTooLong {
                len: raw.id.len(),
                limit: MAX_INPUT_ID_BYTES,
            });
        }

        // `input` の空・空白のみは有効な入力として受理する（モジュール冒
        // 頭ドキュメント「TASK-23.2 との整合」）。

        Ok(Self {
            id: raw.id,
            input: raw.input,
        })
    }

    /// `serde_json::Error::classify() == Data`（構文エラーではない）の失
    /// 敗を `serde_json::Value` として再走査し、型付きの `InferInputError`
    /// へ分類する（`definition.rs` の `diagnose` モジュールと同じ考え方。
    /// 本モジュールはフィールド数が少ないため専用サブモジュールへ分割せ
    /// ず、この関数 1 つに留める）。
    ///
    /// `serde_json::Value` は重複キーを後勝ちで黙って受理するため、重複
    /// フィールド等の `Value` 側では検出できない不整合は
    /// [`InferInputError::Malformed`] へ fail-closed で落とす
    /// （`definition.rs::Definition::parse` の doc と同じ方針）。
    fn diagnose_data_error(line: &str) -> InferInputError {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            return InferInputError::Malformed;
        };

        let Some(object) = value.as_object() else {
            return InferInputError::NotObject {
                actual: json_type_of(&value),
            };
        };

        // 未知フィールド（許可されるのは `id`・`input` のみ）。
        for key in object.keys() {
            if key != "id" && key != "input" {
                return InferInputError::UnknownField;
            }
        }

        match object.get("id") {
            None => {
                return InferInputError::MissingField {
                    field: InferInputField::Id,
                };
            }
            Some(id_value) => {
                if id_value.as_str().is_none() {
                    return InferInputError::TypeMismatch {
                        field: InferInputField::Id,
                        expected: crate::definition::ExpectedType::String,
                        actual: json_type_of(id_value),
                    };
                }
            }
        }

        match object.get("input") {
            None => InferInputError::MissingField {
                field: InferInputField::Input,
            },
            Some(input_value) => {
                if input_value.as_str().is_none() {
                    InferInputError::TypeMismatch {
                        field: InferInputField::Input,
                        expected: crate::definition::ExpectedType::String,
                        actual: json_type_of(input_value),
                    }
                } else {
                    // ここに到達するのは `id`／`input` がともに文字列であ
                    // るにもかかわらず型付きデシリアライズが失敗したケー
                    // ス（重複フィールド等）のみ。fail-closed で `Malformed`
                    // へ落とす。
                    InferInputError::Malformed
                }
            }
        }
    }

    /// 入力の識別子。入力本文は含まない（security.md）。
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// 入力本文（UTF-8 文字列）。byte 化は推論ランタイム／前処理
    /// （Chore #10・REQ-28）の責務で、本型は行わない。
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }
}

/// `serde_json::Value` の実行時の型を [`crate::definition::JsonType`] へ
/// 写す（`definition.rs` の `diagnose` モジュールと同じ語彙を再利用す
/// る）。
fn json_type_of(value: &serde_json::Value) -> crate::definition::JsonType {
    match value {
        serde_json::Value::Null => crate::definition::JsonType::Null,
        serde_json::Value::Bool(_) => crate::definition::JsonType::Bool,
        serde_json::Value::Number(_) => crate::definition::JsonType::Number,
        serde_json::Value::String(_) => crate::definition::JsonType::String,
        serde_json::Value::Array(_) => crate::definition::JsonType::Array,
        serde_json::Value::Object(_) => crate::definition::JsonType::Object,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn io_bytes() -> IoSchema {
        IoSchema {
            input: InputRepresentation::Bytes,
        }
    }

    /// REQ-21: 正常系。
    #[test]
    fn req21_parses_valid_input() {
        let result = InferInput::parse(r#"{"id":"a","input":"hello"}"#, &io_bytes()).unwrap();
        assert_eq!(result.id(), "a");
        assert_eq!(result.input(), "hello");
    }

    /// REQ-21・TASK-23.2 整合: 空・空白のみの `input` は有効な入力として
    /// 受理する（拒否しない）。
    #[test]
    fn req21_accepts_empty_and_whitespace_only_input() {
        let empty = InferInput::parse(r#"{"id":"x","input":""}"#, &io_bytes()).unwrap();
        assert_eq!(empty.input(), "");

        let whitespace =
            InferInput::parse("{\"id\":\"x\",\"input\":\"  \\t\\n\"}", &io_bytes()).unwrap();
        assert_eq!(whitespace.input(), "  \t\n");
    }

    /// REQ-21: 不正な JSON 構文を `MalformedJson`（64）として拒否する。
    #[test]
    fn req21_rejects_malformed_json() {
        let err = InferInput::parse(r#"{"id":"a","#, &io_bytes()).unwrap_err();
        assert_eq!(err, InferInputError::MalformedJson);
        assert_eq!(err.exit_code(), crate::exitcode::ExitCode::InvalidInput);
        assert_eq!(err.exit_code().code(), 64);
    }

    /// REQ-21: ルートが object でない入力（配列・文字列・数値・真偽値・
    /// `null`）を `NotObject` として拒否する。
    #[test]
    fn req21_rejects_non_object_root() {
        for (text, expected_type) in [
            ("[]", crate::definition::JsonType::Array),
            (r#""text""#, crate::definition::JsonType::String),
            ("42", crate::definition::JsonType::Number),
            ("null", crate::definition::JsonType::Null),
            ("true", crate::definition::JsonType::Bool),
        ] {
            let err = InferInput::parse(text, &io_bytes()).unwrap_err();
            assert_eq!(
                err,
                InferInputError::NotObject {
                    actual: expected_type
                }
            );
            assert_eq!(err.exit_code().code(), 64);
        }
    }

    /// REQ-21: `input`・`id` それぞれの欠落を検出する。
    #[test]
    fn req21_rejects_missing_fields() {
        let missing_input = InferInput::parse(r#"{"id":"a"}"#, &io_bytes()).unwrap_err();
        assert_eq!(
            missing_input,
            InferInputError::MissingField {
                field: InferInputField::Input
            }
        );

        let missing_id = InferInput::parse(r#"{"input":"x"}"#, &io_bytes()).unwrap_err();
        assert_eq!(
            missing_id,
            InferInputError::MissingField {
                field: InferInputField::Id
            }
        );
    }

    /// REQ-21: `input` が数値・配列・object・真偽値・`null` の場合、
    /// PoC-16 の `unwrap_or("")` のように黙って空文字列化せず
    /// `TypeMismatch` として拒否する（回帰防止）。
    #[test]
    fn req21_rejects_type_mismatch_for_input_field() {
        for (text, actual) in [
            (
                r#"{"id":"a","input":42}"#,
                crate::definition::JsonType::Number,
            ),
            (
                r#"{"id":"a","input":[]}"#,
                crate::definition::JsonType::Array,
            ),
            (
                r#"{"id":"a","input":{}}"#,
                crate::definition::JsonType::Object,
            ),
            (
                r#"{"id":"a","input":true}"#,
                crate::definition::JsonType::Bool,
            ),
            (
                r#"{"id":"a","input":null}"#,
                crate::definition::JsonType::Null,
            ),
        ] {
            let err = InferInput::parse(text, &io_bytes()).unwrap_err();
            assert_eq!(
                err,
                InferInputError::TypeMismatch {
                    field: InferInputField::Input,
                    expected: crate::definition::ExpectedType::String,
                    actual,
                }
            );
            assert_eq!(err.exit_code().code(), 64);
        }
    }

    /// REQ-21: `id` が数値の場合も `TypeMismatch` として拒否する。
    #[test]
    fn req21_rejects_type_mismatch_for_id_field() {
        let err = InferInput::parse(r#"{"id":42,"input":"x"}"#, &io_bytes()).unwrap_err();
        assert_eq!(
            err,
            InferInputError::TypeMismatch {
                field: InferInputField::Id,
                expected: crate::definition::ExpectedType::String,
                actual: crate::definition::JsonType::Number,
            }
        );
    }

    /// REQ-21: 未知フィールド（未対応の入力種別キー）を拒否する。
    /// `Display`／`{:?}` にキー名・値が含まれないことを確認する
    /// （security.md「秘密情報の混入防止」）。
    #[test]
    fn req21_rejects_unknown_field_without_leaking_key_or_value() {
        let secret_marker = "super-secret-image-payload-marker";
        let text = format!(r#"{{"id":"a","input":"x","image":"{secret_marker}"}}"#);
        let err = InferInput::parse(&text, &io_bytes()).unwrap_err();
        assert_eq!(err, InferInputError::UnknownField);

        let display = err.to_string();
        let debug = format!("{err:?}");
        assert!(!display.contains("image"));
        assert!(!display.contains(secret_marker));
        assert!(!debug.contains("image"));
        assert!(!debug.contains(secret_marker));
    }

    /// REQ-21: 重複キー（後勝ちで受理しない。fail-closed）。
    #[test]
    fn req21_rejects_duplicate_keys() {
        let err =
            InferInput::parse(r#"{"id":"a","input":"x","input":"y"}"#, &io_bytes()).unwrap_err();
        assert_eq!(err, InferInputError::Malformed);
        assert_eq!(err.exit_code().code(), 64);
    }

    /// REQ-21: 空の `id` を拒否する。
    #[test]
    fn req21_rejects_empty_id() {
        let err = InferInput::parse(r#"{"id":"","input":"x"}"#, &io_bytes()).unwrap_err();
        assert_eq!(err, InferInputError::EmptyId);
    }

    /// REQ-21・REQ-39: `id` の長さ境界値（`MAX_INPUT_ID_BYTES` は受理、
    /// 1 バイト超過は `IdTooLong`（64）として拒否する）。
    #[test]
    fn req21_id_length_boundary() {
        let at_limit = "x".repeat(MAX_INPUT_ID_BYTES);
        let text_ok = format!(r#"{{"id":"{at_limit}","input":"x"}}"#);
        assert!(InferInput::parse(&text_ok, &io_bytes()).is_ok());

        let over_limit = "x".repeat(MAX_INPUT_ID_BYTES + 1);
        let text_over = format!(r#"{{"id":"{over_limit}","input":"x"}}"#);
        let err = InferInput::parse(&text_over, &io_bytes()).unwrap_err();
        assert_eq!(
            err,
            InferInputError::IdTooLong {
                len: MAX_INPUT_ID_BYTES + 1,
                limit: MAX_INPUT_ID_BYTES,
            }
        );
        assert_eq!(err.exit_code(), crate::exitcode::ExitCode::InvalidInput);
        assert_eq!(err.exit_code().code(), 64);
    }

    /// REQ-39: サイズ超過（`MAX_INFER_INPUT_BYTES + 1` バイト）は
    /// `TooLarge` として `LimitExceeded`（20）を返す（64 ではない）。
    #[test]
    fn req39_rejects_oversized_input_as_limit_exceeded_not_invalid_input() {
        // ちょうど上限を 1 バイト超える JSON 文字列を作る。`input` の値を
        // 十分長い文字列で埋め、全体のバイト長が `MAX_INFER_INPUT_BYTES` を
        // 1 バイト超えるよう調整する。
        let prefix = r#"{"id":"a","input":""#;
        let suffix = r#""}"#;
        let overhead = prefix.len() + suffix.len();
        let filler_len = MAX_INFER_INPUT_BYTES + 1 - overhead;
        let filler = "x".repeat(filler_len);
        let text = format!("{prefix}{filler}{suffix}");
        assert_eq!(text.len(), MAX_INFER_INPUT_BYTES + 1);

        let err = InferInput::parse(&text, &io_bytes()).unwrap_err();
        assert_eq!(
            err,
            InferInputError::TooLarge {
                len: MAX_INFER_INPUT_BYTES + 1,
                limit: MAX_INFER_INPUT_BYTES,
            }
        );
        assert_eq!(err.exit_code(), crate::exitcode::ExitCode::LimitExceeded);
        assert_eq!(err.exit_code().code(), 20);
        assert_ne!(err.exit_code().code(), 64);
    }

    /// `reason_code()` の全 variant の固定値。
    #[test]
    fn req21_reason_code_is_fixed_for_all_variants() {
        assert_eq!(
            InferInputError::TooLarge { len: 1, limit: 1 }.reason_code(),
            "too_large"
        );
        assert_eq!(
            InferInputError::MalformedJson.reason_code(),
            "malformed_json"
        );
        assert_eq!(
            InferInputError::NotObject {
                actual: crate::definition::JsonType::Array
            }
            .reason_code(),
            "not_object"
        );
        assert_eq!(
            InferInputError::MissingField {
                field: InferInputField::Id
            }
            .reason_code(),
            "missing_field"
        );
        assert_eq!(
            InferInputError::TypeMismatch {
                field: InferInputField::Input,
                expected: crate::definition::ExpectedType::String,
                actual: crate::definition::JsonType::Number,
            }
            .reason_code(),
            "type_mismatch"
        );
        assert_eq!(InferInputError::UnknownField.reason_code(), "unknown_field");
        assert_eq!(InferInputError::Malformed.reason_code(), "malformed");
        assert_eq!(InferInputError::EmptyId.reason_code(), "empty_id");
        assert_eq!(
            InferInputError::IdTooLong { len: 1, limit: 1 }.reason_code(),
            "id_too_long"
        );
    }

    /// REQ-21・security.md「秘密情報の混入防止」: `Display` に入力本文が
    /// 含まれないこと（秘密風ダミー文字列で確認）。
    #[test]
    fn req21_display_never_contains_input_body() {
        let secret = "sk-test-dummy-not-a-real-secret-0123456789";
        let text = format!(r#"{{"id":"a","input":"{secret}"}}"#);
        // 正常にパースできる入力だが、エラー系（型不正）でも本文が漏れな
        // いことを別途確認する。
        assert!(InferInput::parse(&text, &io_bytes()).is_ok());

        let bad_text = format!(r#"{{"id":"a","input":{{"nested":"{secret}"}}}}"#);
        let err = InferInput::parse(&bad_text, &io_bytes()).unwrap_err();
        assert!(!err.to_string().contains(secret));
    }

    /// security.md「秘密情報の混入防止（P0）」: `InferInput` の `Debug`
    /// （`{:?}`）に `id`・`input` の実値が含まれないこと（PR #217 レビュー
    /// 指摘。`#[derive(Debug)]` は非公開フィールドをそのまま出力してしま
    /// い、診断表示〔ログ等〕で推論データ本文が転記されうる）。
    #[test]
    fn req21_debug_never_contains_id_or_input_body() {
        let secret_id = "row-secret-id-marker";
        let secret_input = "sk-test-dummy-not-a-real-secret-0123456789";
        let text = format!(r#"{{"id":"{secret_id}","input":"{secret_input}"}}"#);
        let parsed = InferInput::parse(&text, &io_bytes()).unwrap();

        let debug = format!("{parsed:?}");
        assert!(!debug.contains(secret_id));
        assert!(!debug.contains(secret_input));
        assert_eq!(
            debug,
            r#"InferInput { id: "<redacted>", input: "<redacted>" }"#
        );
    }
}
