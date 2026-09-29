//! 書き出し不能構成の除外・理由記録の結合テスト（REQ-32 異常系・TASK-32.2・#114）。
//!
//! 証拠種別: テストハーネス。既知制約の表の根拠（tract × C3 × INT8 の型不整合）は PoC-14 実測
//! （実機）で、本テストは `tract-onnx` を実行せず、表による除外判定と記録の内容を固定する。
//! 自作ランタイムでの読み込み拒否は、コミット済み C3 を `ConvInteger` へ改変して実際に再現する。
//! 失敗メッセージはケース名と件数のみで入力本文を出さない（security.md）。

#[allow(dead_code)] // 共有の編集器のうち本テストで使わない関数がある
#[path = "common/proto_builder.rs"]
mod proto_builder;

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_runtime::export_exclusion::{
    CandidateDecision, EvidenceKind, ExclusionReason, ExportConfig, InferenceRuntime,
    MAX_EXPORT_CANDIDATES, NumericFormat, ParityCase, ScreeningError, screen_candidate,
    screen_candidate_from_path, screen_candidates,
};
use fandhe_edge_runtime::onnx::{ModelKind, OnnxBackend, OnnxLoadError};
use fandhe_edge_runtime::pipeline::InferencePipeline;
use fandhe_edge_runtime::preprocess::ByteEncodingPreprocessor;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("onnx_parity")
}

struct Fixture {
    path: PathBuf,
    sha: Sha256Digest,
    max_bytes: usize,
    inputs: Vec<String>,
    labels: Vec<usize>,
}

fn fixture(kind: ModelKind) -> Fixture {
    let path = fixture_dir().join("cases.json");
    let doc: Value = serde_json::from_str(&fs::read_to_string(path).expect("read")).expect("json");
    let entry = &doc["kinds"][kind.as_str()];
    let cases = entry["cases"].as_array().expect("cases");
    Fixture {
        path: fixture_dir().join(entry["onnx"].as_str().expect("onnx")),
        sha: entry["onnx_sha256"]
            .as_str()
            .expect("sha")
            .parse()
            .expect("parse"),
        max_bytes: usize::try_from(entry["max_bytes"].as_u64().expect("mb")).expect("usize"),
        inputs: cases
            .iter()
            .map(|c| c["input"].as_str().expect("input").to_string())
            .collect(),
        labels: cases
            .iter()
            .map(|c| usize::try_from(c["mlx_label_index"].as_u64().expect("l")).expect("usize"))
            .collect(),
    }
}

fn parity_cases<'a>(f: &'a Fixture, labels: &[usize]) -> Vec<ParityCase<'a>> {
    f.inputs
        .iter()
        .zip(labels)
        .map(|(input, &reference_label)| ParityCase {
            input,
            reference_label,
        })
        .collect()
}

fn own_f32(kind: ModelKind) -> ExportConfig {
    ExportConfig {
        kind,
        runtime: InferenceRuntime::Own,
        format: NumericFormat::F32,
    }
}

type Pipeline = InferencePipeline<ByteEncodingPreprocessor, OnnxBackend>;

fn from_bytes(bytes: &[u8], kind: ModelKind, max_bytes: usize) -> Result<Pipeline, OnnxLoadError> {
    Ok(InferencePipeline::new(
        ByteEncodingPreprocessor::new(max_bytes),
        OnnxBackend::from_bytes(bytes, kind)?,
    ))
}

fn never_load(_: ModelKind) -> Result<Pipeline, OnnxLoadError> {
    panic!("load must not be called");
}

/// REQ-32: tract × C3 × INT8 は型の不整合を理由に除外され、モデルは読まれない。
#[test]
fn req32_c3_tract_int8_excluded_type_unification() {
    let config = ExportConfig {
        kind: ModelKind::C3,
        runtime: InferenceRuntime::Tract,
        format: NumericFormat::Int8Dynamic,
    };
    let d = screen_candidate(config, never_load, &[]).expect("decision");
    let CandidateDecision::Excluded(r) = d else {
        panic!("must be excluded");
    };
    assert_eq!(r.code(), "export_infeasible");
    assert_eq!(r.detail_code(), Some("type_unification_failed"));
    assert_eq!(r.evidence, EvidenceKind::Measured);
    assert_eq!(r.source(), Some("PoC-14"));
    assert_eq!(r.config, config);
}

/// REQ-32: 自作 F32 の C1・C3 は参照ラベルと全件一致すれば候補に残る。
#[test]
fn req32_own_f32_c1_c3_eligible_when_labels_match() {
    for kind in [ModelKind::C1, ModelKind::C3] {
        let f = fixture(kind);
        let cases = parity_cases(&f, &f.labels);
        let d = screen_candidate_from_path(own_f32(kind), f.max_bytes, &f.path, &f.sha, &cases)
            .expect("decision");
        assert_eq!(d, CandidateDecision::Eligible(own_f32(kind)), "{kind:?}");
    }
}

/// REQ-32: 参照ラベルが 1 件だけ違えば予測ずれとして除外される。
#[test]
fn req32_prediction_mismatch_excluded() {
    let f = fixture(ModelKind::C1);
    let mut labels = f.labels.clone();
    let first = labels.first_mut().expect("first");
    *first = (*first + 1) % 2;
    let cases = parity_cases(&f, &labels);
    let d =
        screen_candidate_from_path(own_f32(ModelKind::C1), f.max_bytes, &f.path, &f.sha, &cases)
            .expect("decision");
    let CandidateDecision::Excluded(r) = d else {
        panic!("must be excluded");
    };
    assert_eq!(
        r.reason,
        ExclusionReason::PredictionMismatch {
            mismatched: 1,
            total: f.inputs.len()
        }
    );
    assert_eq!(r.evidence, EvidenceKind::TestHarness);
}

/// REQ-32: 動的量子化の畳み込み（ConvInteger）は自作ランタイムでも成立せず除外される。
#[test]
fn req32_quantized_conv_rejected_by_own_runtime() {
    let f = fixture(ModelKind::C3);
    let bytes = fs::read(&f.path).expect("read");
    let broken = proto_builder::set_op_type(&bytes, 7, "ConvInteger");
    let cases = parity_cases(&f, &f.labels);
    let d = screen_candidate(
        own_f32(ModelKind::C3),
        |k| from_bytes(&broken, k, f.max_bytes),
        &cases,
    )
    .expect("decision");
    let CandidateDecision::Excluded(r) = d else {
        panic!("must be excluded");
    };
    assert_eq!(
        r.reason,
        ExclusionReason::RuntimeRejectedModel {
            load_code: "unsupported_graph"
        }
    );
    assert_eq!(r.detail_code(), Some("unsupported_graph"));
}

/// REQ-39: sha256 不一致・不正な protobuf・存在しないファイルは除外にせずエラーで止める。
#[test]
fn req39_integrity_and_malformed_propagate_not_excluded() {
    let f = fixture(ModelKind::C1);
    let cases = parity_cases(&f, &f.labels);
    let wrong = Sha256Digest::of_bytes(b"other");
    let err =
        screen_candidate_from_path(own_f32(ModelKind::C1), f.max_bytes, &f.path, &wrong, &cases)
            .expect_err("integrity");
    assert_eq!(err.code(), "integrity_mismatch");
    assert!(matches!(err, ScreeningError::Load(_)));

    let err = screen_candidate(
        own_f32(ModelKind::C1),
        |k| from_bytes(b"not an onnx model", k, f.max_bytes),
        &cases,
    )
    .expect_err("malformed");
    assert_eq!(err.code(), "malformed_protobuf");

    let missing = fixture_dir().join("export_exclusion_missing.onnx");
    let err = screen_candidate_from_path(
        own_f32(ModelKind::C1),
        f.max_bytes,
        &missing,
        &f.sha,
        &cases,
    )
    .expect_err("missing");
    assert_eq!(err.code(), "model_file_error");
}

/// REQ-32: 未対応のランタイム・数値形式とケース 0 件はモデルを読まずに除外される。
#[test]
fn req32_runtime_not_available_and_parity_unverified() {
    let f = fixture(ModelKind::C1);
    let cases = parity_cases(&f, &f.labels);
    let ort = ExportConfig {
        kind: ModelKind::C1,
        runtime: InferenceRuntime::OnnxRuntime,
        format: NumericFormat::F32,
    };
    let CandidateDecision::Excluded(r) = screen_candidate(ort, never_load, &cases).expect("d")
    else {
        panic!("must be excluded");
    };
    assert_eq!(r.code(), "runtime_not_available");

    let CandidateDecision::Excluded(r) =
        screen_candidate(own_f32(ModelKind::C1), never_load, &[]).expect("d")
    else {
        panic!("must be excluded");
    };
    assert_eq!(r.code(), "parity_unverified");
}

/// REQ-32: 複数候補は入力順を保ち、重複・上限超過・範囲外の参照ラベルはエラーになる。
#[test]
fn req32_screen_candidates_order_limits_duplicates() {
    let f = fixture(ModelKind::C1);
    let cases = parity_cases(&f, &f.labels);
    let tract_c3 = ExportConfig {
        kind: ModelKind::C3,
        runtime: InferenceRuntime::Tract,
        format: NumericFormat::Int8Dynamic,
    };
    let ort = ExportConfig {
        kind: ModelKind::C1,
        runtime: InferenceRuntime::OnnxRuntime,
        format: NumericFormat::F16,
    };
    let own = own_f32(ModelKind::C1);
    let decide =
        |c: ExportConfig| screen_candidate_from_path(c, f.max_bytes, &f.path, &f.sha, &cases);
    let s = screen_candidates(&[tract_c3, own, ort], decide).expect("screening");
    assert_eq!(s.eligible, vec![own]);
    let codes: Vec<&str> = s.excluded.iter().map(|r| r.code()).collect();
    assert_eq!(codes, vec!["export_infeasible", "runtime_not_available"]);
    assert_eq!(s.excluded.first().map(|r| r.config), Some(tract_c3));

    let err = screen_candidates(&[own, own], decide).expect_err("dup");
    assert!(matches!(err, ScreeningError::DuplicateCandidate));

    let many = vec![own; MAX_EXPORT_CANDIDATES + 1];
    let err = screen_candidates(&many, decide).expect_err("limit");
    assert!(matches!(err, ScreeningError::TooManyCandidates));

    let out_of_range: Vec<usize> = vec![99; f.inputs.len()];
    let bad = parity_cases(&f, &out_of_range);
    let err =
        screen_candidate_from_path(own, f.max_bytes, &f.path, &f.sha, &bad).expect_err("range");
    assert!(matches!(err, ScreeningError::ReferenceOutOfRange));
}

/// REQ-39: 既知制約・未対応ランタイム・ケース 0 件でも、sha256 不一致・ファイル欠落は
/// 除外にならず全体停止（fail-closed）になる。
#[test]
fn req39_from_path_verifies_model_before_exclusion() {
    let f = fixture(ModelKind::C1);
    let wrong: Sha256Digest = "0000000000000000000000000000000000000000000000000000000000000000"
        .parse()
        .expect("digest");
    let missing = f.path.with_file_name("does_not_exist.onnx");
    let tract_int8 = ExportConfig {
        kind: ModelKind::C3,
        runtime: InferenceRuntime::Tract,
        format: NumericFormat::Int8Dynamic,
    };
    let not_available = ExportConfig {
        kind: ModelKind::C1,
        runtime: InferenceRuntime::OnnxRuntime,
        format: NumericFormat::F16,
    };
    let cases = parity_cases(&f, &f.labels);
    for config in [tract_int8, not_available, own_f32(ModelKind::C1)] {
        for case_set in [&cases[..], &[][..]] {
            let e = screen_candidate_from_path(config, f.max_bytes, &f.path, &wrong, case_set)
                .expect_err("sha mismatch must stop");
            assert_eq!(e.code(), "integrity_mismatch");
            let e = screen_candidate_from_path(config, f.max_bytes, &missing, &f.sha, case_set)
                .expect_err("missing file must stop");
            assert!(matches!(e, ScreeningError::Load(_)));
        }
    }
    // 正しいファイルなら既知制約の除外は従来どおり成立する
    let d = screen_candidate_from_path(tract_int8, f.max_bytes, &f.path, &f.sha, &[])
        .expect("decision");
    assert!(matches!(d, CandidateDecision::Excluded(_)));
}

/// REQ-39: 照合ケースの総入力バイト数が上限（64 MiB）を超えれば推論前に `limit_exceeded`。
#[test]
fn req39_parity_total_input_bytes_limited() {
    let f = fixture(ModelKind::C1);
    let big = "a".repeat(1024 * 1024);
    let cases: Vec<ParityCase<'_>> = (0..65)
        .map(|_| ParityCase {
            input: &big,
            reference_label: 0,
        })
        .collect();
    let e =
        screen_candidate_from_path(own_f32(ModelKind::C1), f.max_bytes, &f.path, &f.sha, &cases)
            .expect_err("total bytes limit");
    assert!(matches!(e, ScreeningError::TotalInputTooLarge));
    assert_eq!(e.code(), "limit_exceeded");
}
