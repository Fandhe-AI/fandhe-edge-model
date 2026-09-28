//! Jev 出力を学習データの既定生成元にしない制約の結合テスト
//! （REQ-40 受け入れ基準 2・TASK-40.2・NR-29・issue #76）。
//!
//! 証拠種別: テストハーネス（`cargo test`）。実機・実 Jev 連携は未実施
//! （spec-reference の証拠種別の区別を崩さない）。
//!
//! 具体値は `crates/data/tests/provenance_ingest.rs`（TASK-40.1-2）と同じ
//! PoC-11 由来のリテラルを流用する（`docs/spec` は参照しない）。

use std::collections::BTreeSet;

use fandhe_edge_data::ingest::{IngestError, ingest_records};
use fandhe_edge_data::inspect::inspect_records;
use fandhe_edge_data::provenance::ingest::{parse_provenance_json, provenance_to_json};
use fandhe_edge_data::provenance::{
    DisallowedSourceError, GenerationSource, ProvenanceError, ProvenanceRecord,
    check_default_training_source,
};

const SAMPLE_MODEL_NAME: &str = "gpt-5.6-sol";
const SAMPLE_PROMPT_HASH_HEX: &str =
    "30382f17d2a33e6e40c9a6ce38563083ab4c788c83785fd70bff6b7e9c03f19c";
const SAMPLE_STARTED_UTC: &str = "2026-09-24T00:14:33.672492+00:00";

fn labels(ids: &[&str]) -> BTreeSet<String> {
    ids.iter().map(|s| s.to_string()).collect()
}

fn meta_json_with_source(source: &str) -> String {
    format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null,"source":"{source}"}}"#
    )
}

fn meta_json_without_source() -> String {
    format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null}}"#
    )
}

/// REQ-40 受け入れ基準 2（異常系の中心）: 生成元が `jev_output` の来歴を
/// 伴う取り込みは拒否される。
#[test]
fn req40_ingest_records_rejects_jev_output_source() {
    let content = "{\"id\":\"r1\",\"input\":\"hello\",\"output\":{\"intent\":\"ok\"}}";
    let valid = labels(&["ok"]);
    let provenance_json = meta_json_with_source("jev_output");

    let result = ingest_records(content, &valid, Some(&provenance_json));

    assert_eq!(
        result,
        Err(IngestError::DisallowedSource(
            DisallowedSourceError::JevOutputAsTrainingSource
        ))
    );
}

/// 順序の確認: 生成元が `jev_output` の場合、データ検査（ラベル集合が空）
/// より前に拒否される（データ検査は行われず `EmptyLabelSet` にはならない。
/// 検査前拒否の機械照合）。
#[test]
fn req40_disallowed_source_is_checked_before_inspecting() {
    let content = "{\"id\":\"r1\",\"input\":\"hello\",\"output\":{\"intent\":\"ok\"}}";
    let empty_valid_label_ids: BTreeSet<String> = BTreeSet::new();
    let provenance_json = meta_json_with_source("jev_output");

    let result = ingest_records(content, &empty_valid_label_ids, Some(&provenance_json));

    assert_eq!(
        result,
        Err(IngestError::DisallowedSource(
            DisallowedSourceError::JevOutputAsTrainingSource
        )),
        "EmptyLabelSet ではなく DisallowedSource が先に返るはず"
    );
}

/// 正常系: 生成元が `self` の来歴は許可され、`inspect` は単体呼び出しと
/// 一致する。
#[test]
fn req40_ingest_records_allows_self_prepared_source() {
    let content = "{\"id\":\"r1\",\"input\":\"hello\",\"output\":{\"intent\":\"ok\"}}";
    let valid = labels(&["ok"]);
    let provenance_json = meta_json_with_source("self");

    let outcome =
        ingest_records(content, &valid, Some(&provenance_json)).expect("self は許可されるはず");

    let provenance = outcome.provenance.expect("来歴は解析できているはず");
    assert_eq!(provenance.source(), GenerationSource::SelfPrepared);

    let standalone = inspect_records(content, &valid).expect("valid_label_ids は空でない");
    assert_eq!(outcome.inspect, standalone);
}

/// 既定値: `source` キー欠落の来歴（PoC-10／PoC-11 の `meta.json` 形）は
/// 許可され、`Unspecified`（非 Jev）になる。
#[test]
fn req40_ingest_records_defaults_missing_source_to_unspecified() {
    let content = "{\"id\":\"r1\",\"input\":\"hello\",\"output\":{\"intent\":\"ok\"}}";
    let valid = labels(&["ok"]);
    let provenance_json = meta_json_without_source();

    let outcome = ingest_records(content, &valid, Some(&provenance_json))
        .expect("source 欠落は許可されるはず");

    let provenance = outcome.provenance.expect("来歴は解析できているはず");
    assert_eq!(provenance.source(), GenerationSource::Unspecified);
    assert_eq!(check_default_training_source(&provenance), Ok(()));
}

/// `ProvenanceRecord::new` の既定生成元も `Unspecified` であり、
/// `check_default_training_source` は許可する
/// （REQ-40 受け入れ基準 2「既定の生成元が Jev にならない」）。
#[test]
fn req40_provenance_record_new_default_source_is_allowed() {
    let record = parse_provenance_json(&meta_json_without_source()).expect("valid provenance");
    assert_eq!(record.source(), GenerationSource::Unspecified);
    assert_eq!(check_default_training_source(&record), Ok(()));
}

/// 許可リスト外の値（表記揺れ・空文字）は `InvalidField` として拒否される。
#[test]
fn req40_unknown_source_values_are_rejected() {
    for input in ["Jev_output", "jev", "distilled", ""] {
        let provenance_json = meta_json_with_source(input);
        let result = parse_provenance_json(&provenance_json);
        assert_eq!(
            result,
            Err(
                fandhe_edge_data::provenance::ingest::ProvenanceIngestError::InvalidField {
                    field: "source",
                    source: ProvenanceError::UnknownGenerationSource,
                }
            ),
            "input={input:?} は拒否されるはず"
        );
    }
}

/// 型不正: `source` が `null` または数値の場合は `InvalidFieldType` になる。
#[test]
fn req40_source_type_errors_are_rejected() {
    let null_source = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null,"source":null}}"#
    );
    assert_eq!(
        parse_provenance_json(&null_source),
        Err(
            fandhe_edge_data::provenance::ingest::ProvenanceIngestError::InvalidFieldType("source")
        )
    );

    let numeric_source = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null,"source":1}}"#
    );
    assert_eq!(
        parse_provenance_json(&numeric_source),
        Err(
            fandhe_edge_data::provenance::ingest::ProvenanceIngestError::InvalidFieldType("source")
        )
    );
}

/// 重複キー: `source` キーが 2 回出現する場合は既存の `DuplicateKey` に
/// なる（後勝ちで Jev がすり抜けないことの確認）。
#[test]
fn req40_duplicate_source_key_is_rejected() {
    let content = format!(
        r#"{{"model_requested":"{SAMPLE_MODEL_NAME}","prompt_sha256":"{SAMPLE_PROMPT_HASH_HEX}","started_utc":"{SAMPLE_STARTED_UTC}","usage_observed":null,"source":"self","source":"jev_output"}}"#
    );
    assert_eq!(
        parse_provenance_json(&content),
        Err(fandhe_edge_data::provenance::ingest::ProvenanceIngestError::DuplicateKey)
    );
}

/// 記録 JSON: `self`・欠落（`unspecified`）の各場合の `provenance_to_json`
/// 出力キー `source` を完全一致で確認する。
#[test]
fn req40_provenance_to_json_includes_source_key() {
    let self_record = parse_provenance_json(&meta_json_with_source("self")).expect("valid");
    assert!(provenance_to_json(&self_record).contains("\"source\":\"self\""));

    let unspecified_record = parse_provenance_json(&meta_json_without_source()).expect("valid");
    assert!(provenance_to_json(&unspecified_record).contains("\"source\":\"unspecified\""));
}

/// エラー文言: `DisallowedSourceError`・`UnknownGenerationSource` の
/// `to_string()` に入力値（マーカー文字列）が含まれないこと。
/// `code()` の値も固定する。
#[test]
fn req40_disallowed_source_error_does_not_leak_input_and_has_stable_code() {
    let err = DisallowedSourceError::JevOutputAsTrainingSource;
    assert!(!err.to_string().contains("MARKER_SOURCE_SHOULD_NOT_LEAK"));
    assert_eq!(err.code(), "provenance_disallowed_source");

    let unknown_source_err = ProvenanceError::UnknownGenerationSource;
    assert!(
        !unknown_source_err
            .to_string()
            .contains("MARKER_SOURCE_SHOULD_NOT_LEAK")
    );
}

/// 来歴なし（`provenance_json = None`）は従来どおり許可される
/// （TASK-40.2 で判断済み。PoC-20 の模擬台帳は来歴なしも拒否していたが、
/// それは模擬の仕様であり、本実装は主経路〔利用者自身のデータ〕を壊さない
/// 判断を採る）。
#[test]
fn req40_ingest_records_without_provenance_still_allowed() {
    let content = "{\"id\":\"r1\",\"input\":\"hello\",\"output\":{\"intent\":\"ok\"}}";
    let valid = labels(&["ok"]);

    let outcome = ingest_records(content, &valid, None).expect("来歴なしは許可されるはず");
    assert!(outcome.provenance.is_none());
}

/// `with_source` の builder 挙動そのものを直接確認する
/// （`ProvenanceRecord::new` は据え置きで `Unspecified` を返す）。
#[test]
fn req40_provenance_record_with_source_builder() {
    let record = ProvenanceRecord::new(
        fandhe_edge_data::provenance::ModelName::new(SAMPLE_MODEL_NAME).expect("valid model name"),
        fandhe_edge_data::provenance::PromptHash::from_hex(SAMPLE_PROMPT_HASH_HEX)
            .expect("valid prompt hash"),
        fandhe_edge_data::provenance::GeneratedAt::parse_rfc3339_utc(SAMPLE_STARTED_UTC)
            .expect("valid timestamp"),
        fandhe_edge_data::provenance::TokenCount::Unobserved,
    );
    assert_eq!(record.source(), GenerationSource::Unspecified);

    let jev_record = record.with_source(GenerationSource::JevOutput);
    assert_eq!(jev_record.source(), GenerationSource::JevOutput);
    assert_eq!(
        check_default_training_source(&jev_record),
        Err(DisallowedSourceError::JevOutputAsTrainingSource)
    );
}
