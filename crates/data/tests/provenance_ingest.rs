//! 来歴の取り込み記録とデータ検査との接続（REQ-40・TASK-40.1-2・issue #75）
//! の結合テスト。公開 API 経由でのみ検証する。
//!
//! 具体値は `crates/data/tests/provenance_record.rs`（TASK-40.1-1）と同じ
//! PoC-11（`docs/spec/03-poc/`。private submodule）の生成ログ由来のリテラル
//! （`docs/spec` は参照しない。証拠種別: テストハーネス。実機での生成ログ
//! 取り込みは未実施）。`meta.json` 形の入力には、実際の生成ログに含まれる
//! が本モジュールが無視すべき余分なキー（`model_observed`・`ended_utc`・
//! `returncode`・`cached_input_tokens` 等）もあえて含め、無視されることを
//! 確認する。絶対パス等が入りうる `cmd`・`cwd`・`prompt_file` は
//! リテラルに含めない。

use std::collections::BTreeSet;

use fandhe_edge_data::ingest::{IngestError, ingest_records};
use fandhe_edge_data::inspect::inspect_records;
use fandhe_edge_data::provenance::ProvenanceError;
use fandhe_edge_data::provenance::ingest::{
    ProvenanceIngestError, parse_provenance_json, provenance_to_json,
};

const SAMPLE_MODEL_NAME: &str = "gpt-5.6-sol";
const SAMPLE_PROMPT_HASH_HEX: &str =
    "30382f17d2a33e6e40c9a6ce38563083ab4c788c83785fd70bff6b7e9c03f19c";
const SAMPLE_STARTED_UTC: &str = "2026-09-24T00:14:33.672492+00:00";
const SAMPLE_GENERATED_AT_CANONICAL: &str = "2026-09-24T00:14:33.672492000Z";

/// `meta.json` 形のリテラル（余分なキーを含む。絶対パス等は含めない）。
fn sample_meta_json() -> String {
    format!(
        r#"{{
            "model_requested": "{SAMPLE_MODEL_NAME}",
            "model_observed": null,
            "prompt_sha256": "{SAMPLE_PROMPT_HASH_HEX}",
            "started_utc": "{SAMPLE_STARTED_UTC}",
            "ended_utc": "2026-09-24T00:15:01.000000+00:00",
            "returncode": 0,
            "usage_observed": {{
                "input_tokens": 14095,
                "output_tokens": 6795,
                "cached_input_tokens": 512,
                "cache_write_input_tokens": 0,
                "reasoning_output_tokens": 128
            }}
        }}"#
    )
}

fn labels(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|s| s.to_string()).collect()
}

/// 受入基準: `meta.json` 形の入力から 4 項目が具体値で記録されること。
#[test]
fn req40_parse_provenance_json_extracts_four_fields_from_meta_json_shape() {
    let record = parse_provenance_json(&sample_meta_json()).expect("valid provenance JSON");

    assert_eq!(record.model_name().as_str(), SAMPLE_MODEL_NAME);
    assert_eq!(record.prompt_hash().to_hex(), SAMPLE_PROMPT_HASH_HEX);
    assert_eq!(
        record.generated_at().to_rfc3339_utc(),
        SAMPLE_GENERATED_AT_CANONICAL
    );
    match record.token_count() {
        fandhe_edge_data::provenance::TokenCount::Observed(usage) => {
            assert_eq!(usage.input_tokens(), 14095);
            assert_eq!(usage.output_tokens(), 6795);
        }
        other => panic!("expected observed token count, got {other:?}"),
    }
}

/// 記録 JSON が期待する正準形（キー整列済み・決定的）と完全一致すること。
#[test]
fn req40_provenance_to_json_matches_canonical_form() {
    let record = parse_provenance_json(&sample_meta_json()).expect("valid provenance JSON");
    let json = provenance_to_json(&record);

    let expected = format!(
        "{{\"generated_at\":\"{SAMPLE_GENERATED_AT_CANONICAL}\",\"model_name\":\"{SAMPLE_MODEL_NAME}\",\"prompt_sha256\":\"{SAMPLE_PROMPT_HASH_HEX}\",\"source\":\"unspecified\",\"token_count\":{{\"input_tokens\":14095,\"output_tokens\":6795,\"status\":\"observed\"}}}}"
    );
    assert_eq!(json, expected);
}

/// `usage_observed: null` は `Unobserved` になり、記録 JSON も
/// `{"status":"unobserved"}` になること（0 で埋めない）。
#[test]
fn req40_usage_observed_null_is_unobserved_and_not_zero_filled() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null}}"#
    );
    let record = parse_provenance_json(&content).expect("valid provenance JSON");
    assert_eq!(
        record.token_count(),
        &fandhe_edge_data::provenance::TokenCount::Unobserved
    );

    let json = provenance_to_json(&record);
    assert!(json.contains("\"token_count\":{\"status\":\"unobserved\"}"));
}

/// `ingest_records`: clean な JSONL 数行＋来歴 → `provenance` が期待レコードと
/// 一致し、`inspect` は `inspect_records` 単体の結果と完全に一致すること。
#[test]
fn req40_ingest_records_with_provenance_matches_standalone_inspect() {
    let content = "\
{\"id\":\"r1\",\"input\":\"hello\",\"output\":{\"intent\":\"ok\"}}
{\"id\":\"r2\",\"input\":\"world\",\"output\":{\"intent\":\"ok\"}}";
    let valid = labels(&["ok"]);
    let provenance_json = sample_meta_json();

    let outcome = ingest_records(content, &valid, Some(&provenance_json))
        .expect("clean content and valid provenance must succeed");

    let standalone = inspect_records(content, &valid).expect("valid_label_ids は空でない");
    assert_eq!(outcome.inspect, standalone);
    assert!(outcome.inspect.anomalies.is_empty());

    let provenance = outcome.provenance.expect("provenance must be recorded");
    assert_eq!(provenance.model_name().as_str(), SAMPLE_MODEL_NAME);
    assert_eq!(provenance.prompt_hash().to_hex(), SAMPLE_PROMPT_HASH_HEX);
    assert_eq!(
        provenance.generated_at().to_rfc3339_utc(),
        SAMPLE_GENERATED_AT_CANONICAL
    );
}

/// `provenance_json = None` の場合、`provenance` は `None` になり、
/// `inspect` は同一であること（来歴が無いデータは TASK-40.2 で許可と判断
/// 済み。issue #76）。
#[test]
fn req40_ingest_records_without_provenance_json_is_not_rejected() {
    let content = "{\"id\":\"r1\",\"input\":\"hello\",\"output\":{\"intent\":\"ok\"}}";
    let valid = labels(&["ok"]);

    let outcome = ingest_records(content, &valid, None).expect("no provenance is allowed");

    assert!(outcome.provenance.is_none());
    let standalone = inspect_records(content, &valid).expect("valid_label_ids は空でない");
    assert_eq!(outcome.inspect, standalone);
}

/// 来歴 JSON が不正な場合、データ検査を行わずに
/// `IngestError::Provenance` を返すこと（fail-closed。検査結果を返さない）。
#[test]
fn req40_invalid_provenance_json_stops_before_inspecting() {
    let content = "{\"id\":\"r1\",\"input\":\"hello\",\"output\":{\"intent\":\"ok\"}}";
    let valid = labels(&["ok"]);

    let result = ingest_records(content, &valid, Some("not json"));

    assert!(matches!(
        result,
        Err(IngestError::Provenance(
            ProvenanceIngestError::MalformedJson
        ))
    ));
}

/// 空のラベル集合は `IngestError::EmptyLabelSet` になること。
#[test]
fn req40_empty_label_set_is_propagated() {
    let content = "{\"id\":\"r1\",\"input\":\"hello\",\"output\":{\"intent\":\"ok\"}}";
    let empty: BTreeSet<String> = BTreeSet::new();

    let result = ingest_records(content, &empty, None);

    assert!(matches!(result, Err(IngestError::EmptyLabelSet(_))));
}

// --- 異常系（parse_provenance_json 単体） ---

#[test]
fn req40_malformed_json_is_rejected() {
    assert_eq!(
        parse_provenance_json("not json"),
        Err(ProvenanceIngestError::MalformedJson)
    );
}

#[test]
fn req40_array_top_level_is_not_an_object() {
    assert_eq!(
        parse_provenance_json("[1,2,3]"),
        Err(ProvenanceIngestError::NotAnObject)
    );
}

#[test]
fn req40_duplicate_top_level_key_is_rejected() {
    let content = format!(
        r#"{{"model_requested":"a","model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::DuplicateKey)
    );
}

#[test]
fn req40_duplicate_nested_key_in_usage_observed_is_rejected() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":{{"input_tokens":1,"input_tokens":2,"output_tokens":3}}}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::DuplicateKey)
    );
}

#[test]
fn req40_missing_model_requested_is_reported() {
    let content = format!(
        r#"{{"prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::MissingField("model_requested"))
    );
}

#[test]
fn req40_missing_prompt_sha256_is_reported() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::MissingField("prompt_sha256"))
    );
}

#[test]
fn req40_missing_started_utc_is_reported() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","usage_observed":null}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::MissingField("started_utc"))
    );
}

/// `usage_observed` キー自体の欠落は「欠落」エラー（明示的な `null` とは
/// 区別する）。
#[test]
fn req40_missing_usage_observed_key_is_reported() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}"}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::MissingField("usage_observed"))
    );
}

#[test]
fn req40_missing_input_tokens_in_usage_observed_is_reported() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":{{"output_tokens":1}}}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::MissingField(
            "usage_observed.input_tokens"
        ))
    );
}

#[test]
fn req40_missing_output_tokens_in_usage_observed_is_reported() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":{{"input_tokens":1}}}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::MissingField(
            "usage_observed.output_tokens"
        ))
    );
}

#[test]
fn req40_negative_token_count_is_invalid_field_type() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":{{"input_tokens":-1,"output_tokens":1}}}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::InvalidFieldType(
            "usage_observed.input_tokens"
        ))
    );
}

#[test]
fn req40_fractional_token_count_is_invalid_field_type() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":{{"input_tokens":1.5,"output_tokens":1}}}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::InvalidFieldType(
            "usage_observed.input_tokens"
        ))
    );
}

#[test]
fn req40_string_token_count_is_invalid_field_type() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":{{"input_tokens":"1","output_tokens":1}}}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::InvalidFieldType(
            "usage_observed.input_tokens"
        ))
    );
}

#[test]
fn req40_uppercase_prompt_hash_is_invalid_field() {
    let uppercase_hash = "A".repeat(64);
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{uppercase_hash}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::InvalidField {
            field: "prompt_sha256",
            source: ProvenanceError::PromptHashInvalidHex,
        })
    );
}

#[test]
fn req40_non_utc_offset_started_utc_is_invalid_field() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"2026-09-24T00:14:33+09:00","usage_observed":null}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::InvalidField {
            field: "started_utc",
            source: ProvenanceError::InvalidGeneratedAt,
        })
    );
}

#[test]
fn req40_empty_model_name_is_invalid_field() {
    let content = format!(
        r#"{{"model_requested":"","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(ProvenanceIngestError::InvalidField {
            field: "model_requested",
            source: ProvenanceError::EmptyModelName,
        })
    );
}

/// エラーの `to_string()` に入力値のマーカー文字列が含まれないこと。
#[test]
fn req40_error_display_never_leaks_input_marker() {
    let marker_model_name = "MARKER_MODEL_NAME_SHOULD_NOT_LEAK";
    let content = format!(
        r#"{{"model_requested":"","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null,"cmd":"{marker_model_name}"}}"#
    );
    let err = parse_provenance_json(&content).expect_err("empty model name must be rejected");
    assert!(!err.to_string().contains(marker_model_name));
}
