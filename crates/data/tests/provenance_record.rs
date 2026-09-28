//! 来歴レコード型（[`fandhe_edge_data::provenance`]）の結合テスト
//! （REQ-40・TASK-40.1-1・issue #74）。公開 API 経由でのみ検証する。
//!
//! 具体値は PoC-11（`docs/spec/03-poc/`。private submodule）の生成ログ
//! （`meta.json` の `model_requested`・`prompt_sha256`・`started_utc`・
//! `usage_observed`）からリテラルとして移植した（`docs/spec` は参照しない。
//! 証拠種別: テストハーネス）。

use fandhe_edge_data::provenance::{
    GeneratedAt, ModelName, ProvenanceError, ProvenanceRecord, TokenCount, TokenUsage,
};

const SAMPLE_MODEL_NAME: &str = "gpt-5.6-sol";
const SAMPLE_PROMPT_HASH_HEX: &str =
    "30382f17d2a33e6e40c9a6ce38563083ab4c788c83785fd70bff6b7e9c03f19c";
const SAMPLE_GENERATED_AT: &str = "2026-09-24T00:14:33.672492+00:00";

#[test]
fn req40_record_holds_four_fields_with_concrete_values() {
    let model_name = ModelName::new(SAMPLE_MODEL_NAME).expect("valid model name");
    let prompt_hash = fandhe_edge_data::provenance::PromptHash::from_hex(SAMPLE_PROMPT_HASH_HEX)
        .expect("valid 64-hex prompt hash");
    let generated_at =
        GeneratedAt::parse_rfc3339_utc(SAMPLE_GENERATED_AT).expect("valid rfc3339 utc timestamp");
    let token_count = TokenCount::Observed(TokenUsage::new(14095, 6795));

    let record = ProvenanceRecord::new(model_name, prompt_hash, generated_at, token_count);

    assert_eq!(record.model_name().as_str(), SAMPLE_MODEL_NAME);
    assert_eq!(record.prompt_hash().to_hex(), SAMPLE_PROMPT_HASH_HEX);
    assert_eq!(
        record.generated_at().to_rfc3339_utc(),
        "2026-09-24T00:14:33.672492000Z"
    );
    match record.token_count() {
        TokenCount::Observed(usage) => assert_eq!(usage.total(), Some(20890)),
        other => panic!("token count must be observed in this fixture, got {other:?}"),
    }
}

#[test]
fn req40_prompt_hash_hex_round_trip() {
    let hash = fandhe_edge_data::provenance::PromptHash::from_hex(SAMPLE_PROMPT_HASH_HEX)
        .expect("valid 64-hex prompt hash");
    assert_eq!(hash.to_hex(), SAMPLE_PROMPT_HASH_HEX);
    assert_eq!(&hash.as_bytes()[..2], &[0x30, 0x38]);
}

#[test]
fn req40_generated_at_z_and_plus_zero_are_equal() {
    let a = GeneratedAt::parse_rfc3339_utc("2026-09-24T00:14:33Z").expect("valid");
    let b = GeneratedAt::parse_rfc3339_utc("2026-09-24T00:14:33+00:00").expect("valid");
    assert_eq!(a, b);
    assert_eq!(a.to_rfc3339_utc(), "2026-09-24T00:14:33Z");
}

#[test]
fn req40_generated_at_orders_chronologically() {
    // 桁数が異なる小数秒でも数値としての大小で比較できることを確認する
    // （.40 秒 = 400ms、.5 秒 = 500ms）。
    let earlier = GeneratedAt::parse_rfc3339_utc("2026-09-24T00:14:33.40Z").expect("valid");
    let later = GeneratedAt::parse_rfc3339_utc("2026-09-24T00:14:33.5Z").expect("valid");
    assert!(earlier < later);
}

#[test]
fn req40_generated_at_accepts_leap_day() {
    assert!(GeneratedAt::parse_rfc3339_utc("2024-02-29T00:00:00Z").is_ok());
    assert!(GeneratedAt::parse_rfc3339_utc("2000-02-29T00:00:00Z").is_ok());
    assert!(GeneratedAt::parse_rfc3339_utc("1900-02-29T00:00:00Z").is_err());
    assert!(GeneratedAt::parse_rfc3339_utc("2026-02-29T00:00:00Z").is_err());
}

#[test]
fn req40_token_count_unobserved_is_not_zero() {
    assert_ne!(
        TokenCount::Unobserved,
        TokenCount::Observed(TokenUsage::new(0, 0))
    );
}

#[test]
fn req40_model_name_boundary_and_invalid_cases() {
    assert_eq!(ModelName::new(""), Err(ProvenanceError::EmptyModelName));

    let max_len = "a".repeat(256);
    assert!(ModelName::new(&max_len).is_ok());
    let too_long = "a".repeat(257);
    assert_eq!(
        ModelName::new(&too_long),
        Err(ProvenanceError::ModelNameTooLong)
    );

    assert_eq!(
        ModelName::new("gpt\n5"),
        Err(ProvenanceError::ModelNameHasControlChar)
    );
    assert_eq!(
        ModelName::new(" gpt-5"),
        Err(ProvenanceError::ModelNameHasSurroundingWhitespace)
    );
}

#[test]
fn req40_prompt_hash_length_and_hex_errors() {
    use fandhe_edge_data::provenance::PromptHash;

    let short = "a".repeat(63);
    assert_eq!(
        PromptHash::from_hex(&short),
        Err(ProvenanceError::PromptHashInvalidLength)
    );
    let long = "a".repeat(65);
    assert_eq!(
        PromptHash::from_hex(&long),
        Err(ProvenanceError::PromptHashInvalidLength)
    );

    let uppercase = format!("A{}", "a".repeat(63));
    assert_eq!(
        PromptHash::from_hex(&uppercase),
        Err(ProvenanceError::PromptHashInvalidHex)
    );
    let non_hex = format!("g{}", "a".repeat(63));
    assert_eq!(
        PromptHash::from_hex(&non_hex),
        Err(ProvenanceError::PromptHashInvalidHex)
    );
}

#[test]
fn req40_generated_at_invalid_cases() {
    let invalid_inputs = [
        "2026-13-01T00:00:00Z",            // 月 13
        "2026-02-30T00:00:00Z",            // 2 月 30 日
        "2026-02-29T00:00:00Z",            // 平年のうるう日
        "1900-02-29T00:00:00Z",            // 100 で割り切れ 400 で割り切れない年のうるう日
        "2026-09-24T24:00:00Z",            // 時 24
        "2026-09-24T00:00:60Z",            // うるう秒
        "2026-09-24T00:14:33+09:00",       // UTC 以外のオフセット
        "2026-09-24T00:14:33",             // オフセット無し
        "2026-09-24 00:14:33Z",            // T 以外の区切り
        "2026-09-24T00:14:33.Z",           // 小数点のみ
        "2026-09-24T00:14:33.1234567890Z", // 小数 10 桁
        "2026-09-24T00:14:33.１２３Z",     // 全角数字
    ];
    for input in invalid_inputs {
        assert_eq!(
            GeneratedAt::parse_rfc3339_utc(input),
            Err(ProvenanceError::InvalidGeneratedAt),
            "input {input:?} should be rejected"
        );
    }

    let too_long_offset = format!("2026-09-24T00:14:33.{}Z", "1".repeat(60));
    assert_eq!(
        GeneratedAt::parse_rfc3339_utc(&too_long_offset),
        Err(ProvenanceError::InvalidGeneratedAt)
    );
}

#[test]
fn req40_token_usage_total_does_not_panic_on_overflow() {
    let usage = TokenUsage::new(u64::MAX, 1);
    assert_eq!(usage.total(), None);
}

#[test]
fn req40_model_name_error_display_does_not_leak_input_value() {
    let marker = format!("MARKER_{}", "x".repeat(300));
    let err = ModelName::new(&marker).expect_err("must be rejected as too long");
    assert!(!err.to_string().contains("MARKER_"));
}
