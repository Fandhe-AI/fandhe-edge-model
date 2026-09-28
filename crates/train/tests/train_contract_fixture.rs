//! REQ-18・REQ-19・REQ-34・REQ-39（issue #177）: 学習リクエスト・結果 JSON の
//! Rust ⇔ 学習ワーカー（Python）一致を共有 fixture（`fixtures/train_contract/`）
//! で照合する。
//!
//! 対になるテストは `trainer/tests/test_train_contract_fixture.py`
//! （pytest）。`fixtures/train_contract/` というパスが唯一の結合点であり、
//! このパスを変えると両テストの結合が切れる。本テストは `fandhe-edge-train`・
//! `fandhe-edge-core`（定義ファイル投影のため）の範囲に閉じており、`docs/spec`
//! は参照しない（本リポの CI は `docs/spec` 抜きで成立させる方針のため）。

use std::fs;
use std::path::PathBuf;

use fandhe_edge_core::definition::Definition;
use fandhe_edge_train::error::TrainRequestError;
use fandhe_edge_train::limits;
use fandhe_edge_train::request::{
    Device, TrainRequest, TrainRequestParams, label_order_from_definition,
};
use fandhe_edge_train::result::TrainOutcome;
use serde_json::Value;

/// 外部入力と同じ作法で扱うための読み込み前サイズ上限（REQ-39）。
const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("train_contract")
}

fn load_fixture_value(name: &str) -> Value {
    let path = fixture_dir().join(name);
    let metadata =
        fs::metadata(&path).unwrap_or_else(|e| panic!("failed to stat fixture {path:?}: {e}"));
    assert!(
        metadata.len() <= MAX_FIXTURE_BYTES,
        "fixture {path:?} exceeds size limit ({} bytes > {MAX_FIXTURE_BYTES})",
        metadata.len()
    );
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("failed to read fixture {path:?}: {e}"));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse fixture {path:?} as JSON: {e}"))
}

fn load_fixture_bytes(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    let metadata =
        fs::metadata(&path).unwrap_or_else(|e| panic!("failed to stat fixture {path:?}: {e}"));
    assert!(
        metadata.len() <= MAX_FIXTURE_BYTES,
        "fixture {path:?} exceeds size limit"
    );
    fs::read(&path).unwrap_or_else(|e| panic!("failed to read fixture {path:?}: {e}"))
}

/// REQ-39(a): fixture の上限値が `crate::limits` の定数と一致する。
#[test]
fn req39_limits_fixture_matches_rust_constants() {
    let fixture = load_fixture_value("limits.json");
    assert_eq!(
        fixture["request_schema_version"].as_u64(),
        Some(u64::from(limits::REQUEST_SCHEMA_VERSION))
    );
    assert_eq!(
        fixture["max_request_bytes"].as_u64(),
        Some(limits::MAX_REQUEST_BYTES as u64)
    );
    assert_eq!(
        fixture["min_labels"].as_u64(),
        Some(limits::MIN_LABELS as u64)
    );
    assert_eq!(
        fixture["max_labels"].as_u64(),
        Some(limits::MAX_LABELS as u64)
    );
    assert_eq!(
        fixture["max_label_bytes"].as_u64(),
        Some(limits::MAX_LABEL_BYTES as u64)
    );
    assert_eq!(
        fixture["min_max_bytes"].as_u64(),
        Some(u64::from(limits::MIN_MAX_BYTES))
    );
    assert_eq!(
        fixture["max_max_bytes"].as_u64(),
        Some(u64::from(limits::MAX_MAX_BYTES))
    );
    assert_eq!(
        fixture["min_seed"].as_u64(),
        Some(u64::from(limits::MIN_SEED))
    );
    assert_eq!(
        fixture["max_seed"].as_u64(),
        Some(u64::from(limits::MAX_SEED))
    );
    assert_eq!(
        fixture["max_train_wall_seconds"].as_u64(),
        Some(u64::from(limits::MAX_TRAIN_WALL_SECONDS))
    );
    assert_eq!(
        fixture["max_train_rss_bytes"].as_u64(),
        Some(limits::MAX_TRAIN_RSS_BYTES)
    );
    assert_eq!(
        fixture["max_result_bytes"].as_u64(),
        Some(limits::MAX_RESULT_BYTES as u64)
    );
    let devices: Vec<String> = fixture["allowed_devices"]
        .as_array()
        .expect("allowed_devices must be an array")
        .iter()
        .map(|v| v.as_str().expect("device must be a string").to_string())
        .collect();
    assert_eq!(devices, limits::ALLOWED_DEVICES.to_vec());
}

/// REQ-18(b): `definition.json` からの投影 + `TrainRequest::new` の直列化が
/// `request_full.json` と一致する（`root` はプレースホルダーのまま）。
#[test]
fn req18_projected_label_order_round_trips_to_request_full_fixture() {
    let definition_text = String::from_utf8(load_fixture_bytes("definition.json"))
        .expect("definition.json must be utf-8");
    let definition = Definition::parse(&definition_text).expect("definition.json must parse");
    let label_order =
        label_order_from_definition(&definition).expect("definition.json must project");

    let expected = load_fixture_value("request_full.json");
    let params = TrainRequestParams {
        kind: expected["kind"].as_str().unwrap().to_string(),
        kind_version: expected["kind_version"].as_u64().unwrap() as u32,
        config: expected["config"].as_object().unwrap().clone(),
        label_order: label_order.clone().into_vec(),
        max_bytes: expected["max_bytes"].as_u64().unwrap() as u32,
        seed: expected["seed"].as_u64().unwrap() as u32,
        device: Device::Cpu,
        root: expected["root"].as_str().unwrap().to_string(),
        train_path: expected["train_path"].as_str().unwrap().to_string(),
        out_dir: expected["out_dir"].as_str().unwrap().to_string(),
        time_limit_seconds: expected["time_limit_seconds"].as_u64().map(|v| v as u32),
        rss_limit_bytes: expected["rss_limit_bytes"].as_u64(),
    };
    let req = TrainRequest::new(params).expect("request_full.json fields must be valid");
    let actual: Value =
        serde_json::from_slice(&req.to_json_vec().expect("serialize")).expect("valid json");
    assert_eq!(actual, expected);
    assert_eq!(
        label_order.as_slice(),
        expected["label_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>()
            .as_slice()
    );
}

/// REQ-18(c): `request_full.json`／`request_minimal.json` を `from_json_slice`
/// で読んだ値のアクセサが fixture の具体値と一致する（往復の一致）。
/// minimal 側は省略した任意項目が既定値へ解決され、直列化時に省略されること
/// も確認する。
#[test]
fn req18_from_json_slice_round_trips_request_full_and_minimal() {
    let full_bytes = load_fixture_bytes("request_full.json");
    let full = TrainRequest::from_json_slice(&full_bytes).expect("request_full.json must parse");
    assert_eq!(full.kind(), "c3");
    assert_eq!(full.kind_version(), 1);
    assert_eq!(full.max_bytes(), 512);
    assert_eq!(full.seed(), 42);
    assert_eq!(full.device(), Device::Cpu);
    assert_eq!(full.root(), "/fandhe-edge-fixture-root");
    assert_eq!(full.train_path(), "train.jsonl");
    assert_eq!(full.out_dir(), "out");
    assert_eq!(full.time_limit_seconds(), 600);
    assert_eq!(full.rss_limit_bytes(), 1_073_741_824);
    let full_expected = load_fixture_value("request_full.json");
    let full_actual: Value =
        serde_json::from_slice(&full.to_json_vec().expect("serialize")).expect("valid json");
    assert_eq!(full_actual, full_expected);

    let minimal_bytes = load_fixture_bytes("request_minimal.json");
    let minimal =
        TrainRequest::from_json_slice(&minimal_bytes).expect("request_minimal.json must parse");
    // 既定値解決（省略時は上限そのもの。`contract.py` の既定値と同じ）。
    assert_eq!(minimal.time_limit_seconds(), limits::MAX_TRAIN_WALL_SECONDS);
    assert_eq!(minimal.rss_limit_bytes(), limits::MAX_TRAIN_RSS_BYTES);
    let minimal_roundtrip: Value =
        serde_json::from_slice(&minimal.to_json_vec().expect("serialize")).expect("valid json");
    // `config` は省略時も `{}` として出力され、`time_limit_seconds`／
    // `rss_limit_bytes` は既定値と等しいため出力されない
    // （`request_minimal.json` 自体にキーが無いことと一致する）。
    let minimal_expected = load_fixture_value("request_minimal.json");
    let mut expected_with_config = minimal_expected.clone();
    expected_with_config["config"] = serde_json::json!({});
    assert_eq!(minimal_roundtrip, expected_with_config);
    assert!(minimal_expected.get("time_limit_seconds").is_none());
    assert!(minimal_expected.get("rss_limit_bytes").is_none());
}

/// REQ-39(d): `request_reject_cases.json` の全ケースで `from_json_slice` が
/// `Err` を返し、`reason_code()`／`exit_code().code()` が fixture の値と
/// 一致する。
#[test]
fn req39_reject_cases_match_expected_code_and_exit() {
    let fixture = load_fixture_value("request_reject_cases.json");
    let base = fixture["base"].clone();
    let cases = fixture["cases"].as_array().expect("cases must be an array");
    assert!(!cases.is_empty(), "reject cases fixture must not be empty");

    for case in cases {
        let name = case["name"].as_str().expect("case name must be a string");
        let expected_code = case["expected_code"]
            .as_str()
            .expect("expected_code must be a string");
        let expected_exit = case["expected_exit"]
            .as_u64()
            .expect("expected_exit must be a number");

        let bytes: Vec<u8> = if let Some(raw_text) = case.get("raw_text").and_then(Value::as_str) {
            raw_text.as_bytes().to_vec()
        } else {
            let mut applied = base.clone();
            let patch = case["patch"].as_object().expect("patch must be an object");
            let obj = applied.as_object_mut().expect("base must be an object");
            for (key, value) in patch {
                obj.insert(key.clone(), value.clone());
            }
            serde_json::to_vec(&applied).expect("serialize patched request")
        };

        let err = TrainRequest::from_json_slice(&bytes)
            .expect_err(&format!("case {name} must be rejected"));
        assert_eq!(err.reason_code(), expected_code, "case: {name}");
        assert_eq!(
            u64::from(err.exit_code().code()),
            expected_exit,
            "case: {name}"
        );
    }
}

/// REQ-39(e): `MAX_REQUEST_BYTES` を超えるリクエストは解析前に拒否され、
/// `limit_exceeded`／exit 20 になる（fixture には巨大な文字列を置かず、
/// テスト内で生成する）。
#[test]
fn req39_oversized_request_is_rejected_before_parsing() {
    let huge = vec![b'a'; limits::MAX_REQUEST_BYTES + 1];
    let err = TrainRequest::from_json_slice(&huge).unwrap_err();
    assert_eq!(err.reason_code(), "limit_exceeded");
    assert_eq!(err.exit_code().code(), 20);
    assert!(matches!(err, TrainRequestError::TooLarge { .. }));
}

/// REQ-39(e): `config` が巨大で `to_json_vec` がサイズ超過を検出する
/// （組み立て段階での再検査）。
#[test]
fn req39_to_json_vec_rejects_oversized_config() {
    let mut config = serde_json::Map::new();
    config.insert(
        "huge".to_string(),
        Value::String("a".repeat(limits::MAX_REQUEST_BYTES)),
    );
    let params = TrainRequestParams {
        kind: "c3".to_string(),
        kind_version: 1,
        config,
        label_order: vec!["a".to_string(), "b".to_string()],
        max_bytes: 512,
        seed: 0,
        device: Device::Cpu,
        root: "/fandhe-edge-fixture-root".to_string(),
        train_path: "train.jsonl".to_string(),
        out_dir: "out".to_string(),
        time_limit_seconds: None,
        rss_limit_bytes: None,
    };
    let req = TrainRequest::new(params).expect("oversized config is not rejected by new()");
    let err = req.to_json_vec().unwrap_err();
    assert_eq!(err.reason_code(), "limit_exceeded");
    assert_eq!(err.exit_code().code(), 20);
}

/// REQ-21(g): `result_ok.json`／`result_error.json` を `from_worker_stdout`
/// で読むと各項目が具体値と一致し、Serialize し直した Value が fixture と
/// 一致する。`result_ok.json` の `artifact` の各値（`kind`・`kind_version`・
/// `config`・`label_order`・`max_bytes`）と `artifact_dir`（`root`＋`out_dir`
/// の結合）は `request_full.json` と揃えてある（REQ-39・P0/P1。PR #220
/// レビュー対応。`crates/train/src/result.rs` の `from_worker_stdout` doc
/// 参照）ため、ここでは `request_full.json` から組み立てた `TrainRequest`
/// を渡して検証する。
#[test]
fn req21_result_ok_and_error_round_trip_fixture() {
    let request_full_bytes = load_fixture_bytes("request_full.json");
    let request = TrainRequest::from_json_slice(&request_full_bytes)
        .expect("request_full.json must parse into a TrainRequest");

    let ok_bytes = load_fixture_bytes("result_ok.json");
    let ok_line = String::from_utf8(ok_bytes).unwrap().replace('\n', "");
    let outcome = TrainOutcome::from_worker_stdout(ok_line.as_bytes(), &request)
        .expect("result_ok.json must parse and match request_full.json");
    match &outcome {
        TrainOutcome::Ok(success) => {
            assert_eq!(success.artifact_dir(), "/fandhe-edge-fixture-root/out");
            assert_eq!(success.artifact().kind(), "c3");
            assert_eq!(success.artifact().onnx_file(), "model.onnx");
            assert_eq!(
                success.artifact().onnx_sha256().as_str(),
                "ae2c3277fb02c187294c01f4e19e3eca57815d28e9ebc05f4d013a05a02ecfbe"
            );
        }
        TrainOutcome::Error(_) => panic!("expected Ok"),
    }
    let expected_ok = load_fixture_value("result_ok.json");
    let actual_ok = serde_json::to_value(&outcome).expect("serialize outcome");
    assert_eq!(actual_ok, expected_ok);

    let error_bytes = load_fixture_bytes("result_error.json");
    let error_line = String::from_utf8(error_bytes).unwrap().replace('\n', "");
    let error_outcome = TrainOutcome::from_worker_stdout(error_line.as_bytes(), &request)
        .expect("result_error.json must parse");
    match &error_outcome {
        TrainOutcome::Error(failure) => {
            assert_eq!(failure.code(), "invalid_request");
            assert_eq!(failure.message(), "file not readable: FileNotFoundError");
        }
        TrainOutcome::Ok(_) => panic!("expected Error"),
    }
    let expected_error = load_fixture_value("result_error.json");
    let actual_error = serde_json::to_value(&error_outcome).expect("serialize outcome");
    assert_eq!(actual_error, expected_error);
}

/// REQ-18・REQ-19・REQ-21・REQ-39・P1（codex 指摘 PR #220「成果物の追加
/// config 値を検証せず成功扱いにしている」）: `result_ok.json` の
/// `artifact.config` は「`kind_defaults.json` の `c3` に `request_full.json`
/// の `config`（`{"epochs":2}`）を上書きした実効 config」と完全一致する。
/// `crates/train/src/kind_defaults.rs` が埋め込む fixture の内容が
/// `result_ok.json`（実際のワーカー出力を書き起こした値）と食い違っていない
/// ことを、`from_worker_stdout` 経由（上のテスト）とは独立に、fixture 同士の
/// 突き合わせでも確認する。
#[test]
fn req19_result_ok_config_matches_kind_defaults_fixture_merged_with_request_full() {
    let kind_defaults = load_fixture_value("kind_defaults.json");
    let request_full = load_fixture_value("request_full.json");
    let result_ok = load_fixture_value("result_ok.json");

    let kind = request_full["kind"]
        .as_str()
        .expect("kind must be a string");
    let mut expected_config = kind_defaults[kind]
        .as_object()
        .unwrap_or_else(|| panic!("kind_defaults.json must have defaults for kind {kind:?}"))
        .clone();
    let request_config = request_full["config"]
        .as_object()
        .expect("config must be an object");
    for (key, value) in request_config {
        expected_config.insert(key.clone(), value.clone());
    }

    assert_eq!(
        result_ok["artifact"]["config"],
        Value::Object(expected_config)
    );
}

/// issue #178 PR #233 レビュー再々々指摘 P0「単独起動時の防御を弱めている」:
/// Rust 側 `SUPERVISOR_GROUP_MANAGED_ENV`／`SUPERVISOR_GROUP_MANAGED_VALUE`
/// が、共有 fixture（`fixtures/train_contract/supervisor_group_managed_env.json`）
/// 経由で Python 側 `supervisor.py::SUPERVISOR_GROUP_MANAGED_ENV` と一致する
/// こと。対になるテストは `trainer/tests/test_train_contract_fixture.py`。
/// unix 限定（当該定数が unix 限定のため。`fandhe_edge_train::process` 参照）。
#[cfg(unix)]
#[test]
fn req39_supervisor_group_managed_env_matches_python_constant() {
    let fixture = load_fixture_value("supervisor_group_managed_env.json");
    let expected_env_var = fixture["env_var"]
        .as_str()
        .expect("env_var must be a string");
    let expected_managed_value = fixture["managed_value"]
        .as_str()
        .expect("managed_value must be a string");

    assert_eq!(
        fandhe_edge_train::process::SUPERVISOR_GROUP_MANAGED_ENV,
        expected_env_var
    );
    assert_eq!(
        fandhe_edge_train::process::SUPERVISOR_GROUP_MANAGED_VALUE,
        expected_managed_value
    );
}
