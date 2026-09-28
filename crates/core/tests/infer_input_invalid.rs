//! 結合テスト: 定義ファイルのパース（`Definition::parse`）から推論入力の
//! 検証（`InferInput::parse`）までを通し、異常系が終了コード 64
//! （`invalid_input`）へ写ることを確認する（REQ-21・TASK-21.2）。
//!
//! 定義ファイル JSON はテスト内に直接埋め込み、`docs/spec` は読み込まな
//! い（`.claude/rules/spec-reference.md`「ビルド・テストは `docs/spec`
//! 抜きで成立させる」）。

use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::infer_input::{InferInput, InferInputError};

/// テスト用の最小の定義ファイル JSON（入力表現は byte のみ）。
const VALID_DEFINITION_JSON: &str = r#"{
  "schema": "fandhe-edge-model-definition/v1",
  "name": "test-definition",
  "version": 1,
  "judgment_type": "single_select",
  "options": [
    {"id": "yes", "display_name": "Yes", "description": ""},
    {"id": "no", "display_name": "No", "description": ""}
  ],
  "io": {"input": "bytes"}
}"#;

/// REQ-21・TASK-21.2: 正常な定義ファイル・正常な推論入力の組み合わせで
/// `InferInput::parse` が成功し、`id`／`input` を復元できること。
#[test]
fn req21_valid_definition_and_input_round_trip() {
    let definition = Definition::parse(VALID_DEFINITION_JSON).expect("valid definition");
    let input = InferInput::parse(r#"{"id":"row-1","input":"hello"}"#, definition.io())
        .expect("valid infer input");

    assert_eq!(input.id(), "row-1");
    assert_eq!(input.input(), "hello");
}

/// REQ-21・TASK-21.2: 型不正の推論入力（`input` が数値）を、正常な定義
/// ファイルの `io` と組み合わせて検証すると `ExitCode::InvalidInput`
/// （64）になること。`std::process::ExitCode` への変換も確認する。
#[test]
fn req21_type_mismatch_infer_input_maps_to_invalid_input_exit_code() {
    let definition = Definition::parse(VALID_DEFINITION_JSON).expect("valid definition");
    let err = InferInput::parse(r#"{"id":"row-1","input":42}"#, definition.io())
        .expect_err("type mismatch must be rejected");

    assert_eq!(err.exit_code(), ExitCode::InvalidInput);
    assert_eq!(err.exit_code().code(), 64);
    assert_eq!(
        std::process::ExitCode::from(err.exit_code()),
        std::process::ExitCode::from(64u8)
    );
}

/// REQ-21・TASK-21.2: 未知の入力形式（未対応の入力種別キー・JSON 構文
/// 不正・ルート非 object・必須フィールド欠落）を一括で確認し、すべて
/// `ExitCode::InvalidInput`（64）になること。
#[test]
fn req21_unknown_input_formats_map_to_invalid_input_exit_code() {
    let definition = Definition::parse(VALID_DEFINITION_JSON).expect("valid definition");

    let cases: [&str; 5] = [
        r#"{"id":"a","input":"x","image":"unsupported-kind"}"#, // 未対応の入力種別キー
        r#"{"id":"a","#,                                        // JSON 構文不正
        r#"[]"#,                                                // ルート非 object
        r#"{"id":"a"}"#,                                        // input 欠落
        r#"{"input":"x"}"#,                                     // id 欠落
    ];

    for case in cases {
        let err = InferInput::parse(case, definition.io())
            .expect_err("must be rejected as invalid input");
        assert_eq!(
            err.exit_code(),
            ExitCode::InvalidInput,
            "case {case:?} must map to InvalidInput"
        );
        assert_eq!(err.exit_code().code(), 64);
    }
}

/// REQ-21・PoC-16 相当（`invalid_input_missing_kind`）: 必須項目（選択肢
/// 一覧）が欠けた定義ファイルは `DefinitionError::exit_code()` が 64 にな
/// ること。
#[test]
fn req21_definition_missing_labels_maps_to_invalid_input_exit_code() {
    const MISSING_OPTIONS_JSON: &str = r#"{
      "schema": "fandhe-edge-model-definition/v1",
      "name": "test-definition",
      "version": 1,
      "judgment_type": "single_select",
      "io": {"input": "bytes"}
    }"#;

    let err = Definition::parse(MISSING_OPTIONS_JSON).expect_err("missing options must fail");
    assert_eq!(err.exit_code(), ExitCode::InvalidInput);
    assert_eq!(err.exit_code().code(), 64);
}

/// TASK-23.2 整合: 空・空白のみの `input` は、正常な定義ファイルと組み合
/// わせても拒否されないこと（有効な入力として受理する）。
#[test]
fn req21_empty_input_is_not_rejected_as_invalid() {
    let definition = Definition::parse(VALID_DEFINITION_JSON).expect("valid definition");

    let empty = InferInput::parse(r#"{"id":"row-1","input":""}"#, definition.io())
        .expect("empty input must be accepted");
    assert_eq!(empty.input(), "");

    let whitespace =
        InferInput::parse("{\"id\":\"row-2\",\"input\":\"  \\t\\n\"}", definition.io())
            .expect("whitespace-only input must be accepted");
    assert_eq!(whitespace.input(), "  \t\n");
}

/// TASK-21.2: `InferInputError` の `Display` に秘密情報が含まれないこと
/// を、未知フィールドのキー名・値で確認する（結合テストとしても固定す
/// る）。
#[test]
fn req_security_unknown_field_error_does_not_leak_key_or_value() {
    let definition = Definition::parse(VALID_DEFINITION_JSON).expect("valid definition");
    let secret = "dummy-secret-marker-not-a-real-credential";
    let text = format!(r#"{{"id":"a","input":"x","image":"{secret}"}}"#);

    let err =
        InferInput::parse(&text, definition.io()).expect_err("unknown field must be rejected");
    assert_eq!(err, InferInputError::UnknownField);
    assert!(!err.to_string().contains("image"));
    assert!(!err.to_string().contains(secret));
}
