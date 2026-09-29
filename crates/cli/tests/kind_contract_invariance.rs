//! 種類の差し替えで `train`／`infer` の stdout JSON と終了コードが変わらない
//! ことの確認（REQ-19 受け入れ基準「境界値」・TASK-19.4・#142。REQ-21・REQ-33）。
//!
//! 学習するモデルの種類（`c1`・`c3`・`autoregressive`）は選択口（REQ-19）で
//! 差し替えられる。種類固有の情報や形が CLI の出力契約へ漏れていないことを、
//! `fandhe-edge-cli` の lib が公開する出力・変換関数（#136 が各工程の結線で
//! 使うのと同じ部品）へ種類ごとの入力を流して固定する。
//!
//! # 証拠の種別
//!
//! テストハーネス（lib 経由）。バイナリ（`fandhe-edge`）での完走は #136 の
//! 範囲で、現状の `main.rs` はどの種類でも `runtime_error`（exit 70）を返す
//! ため、本ファイルはバイナリを起動しない。#136 の結線後にバイナリレベルへ
//! 広げる想定。期待値の出典は PoC-16（`core-cli-vertical-slice`）の終了コード表・
//! `{"code","message"}`・推論出力のキー構成で、値はリテラルで埋め込む
//! （`docs/spec` は読まない）。
//!
//! # 未確定の範囲
//!
//! `train` の exit 0 の stdout JSON 型は未定義（#136・TASK-33.x で決まる）。
//! 成功経路で確かめるのは「exit 0 になり異常系 JSON へ流れないこと」までで、
//! 型が決まったら具体値のアサーションを加える。

use fandhe_edge_cli::error_report::{emit_error, emit_error_report, train_outcome_error_report};
use fandhe_edge_cli::output::write_ok_judgment;
use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::judgment::JudgmentResult;
use fandhe_edge_runtime::pipeline::{
    BackendError, InferencePipeline, PreprocessError, Preprocessor, ScoringBackend, TokenIds,
};
use fandhe_edge_train::error::TrainResultError;
use fandhe_edge_train::request::TrainRequest;
use fandhe_edge_train::result::{FailureCode, TrainOutcome};

const KINDS: [&str; 3] = ["c1", "c3", "autoregressive"];

/// 実在しない絶対パス（`crates/train/tests/worker_process.rs` と同じ方式）。
const ROOT: &str = "/fandhe-edge-kind-contract-fixture-root";

/// `fixtures/train_contract/request_minimal.json` と同じ項目で、種類だけを差し替える。
fn request_json(kind: &str) -> String {
    format!(
        r#"{{"schema_version":1,"kind":"{kind}","kind_version":1,"label_order":["positive","negative","neutral"],"max_bytes":512,"seed":42,"device":"cpu","root":"{ROOT}","train_path":"train.jsonl","out_dir":"out"}}"#
    )
}

fn request(kind: &str) -> TrainRequest {
    TrainRequest::from_json_slice(request_json(kind).as_bytes()).expect("valid request")
}

/// `fixtures/train_contract/kind_defaults.json` と数値の書き方まで同じ config。
fn default_config(kind: &str) -> &'static str {
    match kind {
        "c1" => {
            r#"{"ngram_min":1,"ngram_max":4,"min_df":2,"max_features":200000,"C":1.0,"epochs":30,"batch_size":64,"lr":0.5}"#
        }
        "c3" => {
            r#"{"lr":0.001,"weight_decay":0.0001,"epochs":40,"batch_size":64,"emb":64,"filters":128,"widths":[3,5,7],"dropout":0.3}"#
        }
        "autoregressive" => {
            r#"{"layers":2,"dims":128,"heads":4,"dropout":0.1,"lr":0.0005,"warmup_steps":100,"weight_decay":0.0001,"batch_size":32,"epochs":40}"#
        }
        other => panic!("unknown kind in test: {other}"),
    }
}

fn ok_worker_json(kind: &str) -> String {
    format!(
        r#"{{"status":"ok","artifact_dir":"{ROOT}/out","artifact":{{"kind":"{kind}","kind_version":1,"selector_version":"0.1","config":{config},"label_order":["positive","negative","neutral"],"output_type":"choice","max_bytes":512,"onnx_file":"model.onnx","onnx_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","created_utc":"2026-09-28T00:00:00Z","candidate_label":"{kind}"}}}}"#,
        config = default_config(kind)
    )
}

fn error_worker_json(code: &str, message: &str) -> String {
    format!(r#"{{"status":"error","code":"{code}","message":"{message}"}}"#)
}

fn emit_to_string(report: &ErrorReport) -> (ExitCode, String) {
    let mut buffer: Vec<u8> = Vec::new();
    let code = emit_error_report(&mut buffer, report).expect("write must succeed");
    (code, String::from_utf8(buffer).expect("UTF-8 output"))
}

/// REQ-19・REQ-21・REQ-33・TASK-19.4: ワーカーの失敗 12 種すべてについて、
/// 種類を差し替えても stdout の 1 行と終了コードがバイト単位で同じで、
/// ワーカー由来の message・種類名を含まないこと。
#[test]
fn req19_train_failure_output_is_kind_invariant() {
    // (ワーカーの code, 期待する終了コード, 期待する stdout の 1 行)
    let expected: [(&str, ExitCode, &str); 12] = [
        (
            "invalid_request",
            ExitCode::InvalidInput,
            "{\"code\":\"invalid_input\",\"message\":\"train worker failed: invalid_request\"}\n",
        ),
        (
            "invalid_data",
            ExitCode::InvalidInput,
            "{\"code\":\"invalid_input\",\"message\":\"train worker failed: invalid_data\"}\n",
        ),
        (
            "limit_exceeded",
            ExitCode::LimitExceeded,
            "{\"code\":\"limit_exceeded\",\"message\":\"train worker failed: limit_exceeded\"}\n",
        ),
        (
            "invalid_path",
            ExitCode::InvalidInput,
            "{\"code\":\"invalid_input\",\"message\":\"train worker failed: invalid_path\"}\n",
        ),
        (
            "symlink_not_allowed",
            ExitCode::InvalidInput,
            "{\"code\":\"invalid_input\",\"message\":\"train worker failed: symlink_not_allowed\"}\n",
        ),
        (
            "integrity_check_failed",
            ExitCode::RuntimeError,
            "{\"code\":\"runtime_error\",\"message\":\"train worker failed: integrity_check_failed\"}\n",
        ),
        (
            "output_conflict",
            ExitCode::InvalidInput,
            "{\"code\":\"invalid_input\",\"message\":\"train worker failed: output_conflict\"}\n",
        ),
        (
            "unsupported_kind",
            ExitCode::InvalidInput,
            "{\"code\":\"invalid_input\",\"message\":\"train worker failed: unsupported_kind\"}\n",
        ),
        (
            "unsupported_kind_version",
            ExitCode::InvalidInput,
            "{\"code\":\"invalid_input\",\"message\":\"train worker failed: unsupported_kind_version\"}\n",
        ),
        (
            "invalid_config",
            ExitCode::InvalidInput,
            "{\"code\":\"invalid_input\",\"message\":\"train worker failed: invalid_config\"}\n",
        ),
        (
            "training_diverged",
            ExitCode::Pending,
            "{\"code\":\"pending\",\"message\":\"train worker failed: training_diverged\"}\n",
        ),
        (
            "runtime_error",
            ExitCode::RuntimeError,
            "{\"code\":\"runtime_error\",\"message\":\"train worker failed: runtime_error\"}\n",
        ),
    ];
    // FailureCode が増減したら、この表の更新を促すために失敗させる。
    assert_eq!(expected.len(), FailureCode::all().len());

    for (code, exit_expected, line_expected) in expected {
        let mut lines: Vec<(ExitCode, String)> = Vec::new();
        for kind in KINDS {
            let detail = format!("worker detail for {kind}");
            let stdout = error_worker_json(code, &detail);
            let outcome = TrainOutcome::from_worker_stdout(stdout.as_bytes(), &request(kind))
                .expect("worker error JSON must parse");
            let report =
                train_outcome_error_report(&outcome).expect("failure must yield an error report");
            let (exit, line) = emit_to_string(&report);

            assert_eq!(exit, outcome.exit_code(), "outcome exit for {code}/{kind}");
            assert_eq!(line.matches('\n').count(), 1, "single newline for {code}");
            assert!(line.ends_with('\n'));
            assert!(!line.contains(&detail), "worker message leaked: {code}");
            assert!(
                !line.contains(kind),
                "kind leaked into output: {code}/{kind}"
            );
            lines.push((exit, line));
        }
        assert!(
            lines.windows(2).all(|w| w[0] == w[1]),
            "output differs across kinds for {code}"
        );
        assert_eq!(lines[0].0, exit_expected, "exit code for {code}");
        assert_eq!(lines[0].1, line_expected, "stdout line for {code}");
    }
}

/// REQ-19・REQ-21: 成功結果は種類によらず exit 0 で、異常系 JSON へ流れず、
/// 成果物の kind が依頼の kind と一致すること（`autoregressive` の既定 config の
/// 照合が成立することも兼ねる）。exit 0 の stdout JSON 型は未定義（#136）。
#[test]
fn req19_train_success_is_exit_ok_for_every_kind() {
    for kind in KINDS {
        let outcome =
            TrainOutcome::from_worker_stdout(ok_worker_json(kind).as_bytes(), &request(kind))
                .expect("ok worker JSON must be accepted");
        assert_eq!(outcome.exit_code(), ExitCode::Ok, "{kind}");
        assert_eq!(outcome.exit_code().code(), 0, "{kind}");
        assert!(train_outcome_error_report(&outcome).is_none(), "{kind}");
        match &outcome {
            TrainOutcome::Ok(success) => assert_eq!(success.artifact().kind(), kind),
            TrainOutcome::Error(_) => panic!("unexpected error outcome for {kind}"),
        }
    }
}

/// REQ-19・REQ-21・REQ-39: 依頼と成果物の種類の取り違えは、どの組でも同じ
/// `runtime_error`（exit 70）の 1 行になること。
#[test]
fn req19_train_kind_mismatch_output_is_pair_invariant() {
    let pairs = [
        ("c1", "c3"),
        ("c3", "autoregressive"),
        ("autoregressive", "c1"),
    ];
    for (requested, produced) in pairs {
        let err = TrainOutcome::from_worker_stdout(
            ok_worker_json(produced).as_bytes(),
            &request(requested),
        )
        .expect_err("kind mismatch must be rejected");
        assert!(
            matches!(err, TrainResultError::ArtifactMismatch { field: "kind" }),
            "{requested}/{produced}"
        );
        let mut buffer: Vec<u8> = Vec::new();
        let exit = emit_error(&mut buffer, &err).expect("write must succeed");
        assert_eq!(exit, ExitCode::RuntimeError);
        assert_eq!(exit.code(), 70);
        assert_eq!(
            String::from_utf8(buffer).expect("UTF-8"),
            "{\"code\":\"runtime_error\",\"message\":\"train result artifact field does not match the request: kind\"}\n",
            "{requested}/{produced}"
        );
    }
}

/// REQ-19・REQ-21: 未知の種類に対するワーカーの `unsupported_kind` も、既知の
/// 種類が受けたときと同じ形（exit 64・固定 message）になること。
#[test]
fn req19_unsupported_kind_uses_same_error_shape() {
    let unknown = "unknown_kind_for_test";
    let outcome = TrainOutcome::from_worker_stdout(
        error_worker_json("unsupported_kind", "worker detail").as_bytes(),
        &request(unknown),
    )
    .expect("worker error JSON must parse");
    let report = train_outcome_error_report(&outcome).expect("error report");
    let (exit, line) = emit_to_string(&report);
    assert_eq!(exit, ExitCode::InvalidInput);
    assert_eq!(exit.code(), 64);
    assert_eq!(
        line,
        "{\"code\":\"invalid_input\",\"message\":\"train worker failed: unsupported_kind\"}\n"
    );
    assert!(!line.contains(unknown));
}

/// 種類ごとに異なるトークン列を作る前処理のスタブ。
///
/// 本物の種類別 ONNX 推論経路は未実装（`crates/runtime` は前処理・ONNX 推論が
/// 未着手）のため、本テストは「種類が違えば前処理と backend の経路の中身が違う」
/// 状況をスタブで作り、その差が CLI の出力契約へ漏れないことだけを確認する。
struct KindPreprocessor {
    kind: &'static str,
}

impl Preprocessor for KindPreprocessor {
    fn preprocess(&self, input: &str) -> Result<TokenIds, PreprocessError> {
        let mut ids: Vec<i64> = input.bytes().map(i64::from).collect();
        match self.kind {
            // 固定長 16 へ 0 埋めする（C3 の畳み込み入力を模す）。
            "c3" => ids.resize(16, 0),
            // 逆順にする（自己回帰 decoder の系列処理を模す）。
            "autoregressive" => ids.reverse(),
            _ => {}
        }
        Ok(TokenIds::new(ids))
    }
}

/// トークン列の形から種類ごとに異なる確率を返すスタブ。
///
/// 前処理が自分の種類の形（c1 は先頭 `s`・長さ 12、c3 は長さ 16 の 0 埋め、
/// autoregressive は先頭 `t` の逆順）を作っていなければ `BackendError::Failed` を
/// 返す。種類ごとの前処理と backend の組が噛み合わないと本テストが失敗する。
struct KindStubBackend {
    kind: &'static str,
}

impl ScoringBackend for KindStubBackend {
    /// テスト用スタブ: 計算は即時で打ち切り対象の反復を持たないため、`scores` と同じ結果を返す。
    fn scores_limited(
        &self,
        ids: &fandhe_edge_runtime::pipeline::TokenIds,
        _limit: std::time::Duration,
    ) -> Result<Vec<f64>, fandhe_edge_runtime::pipeline::BackendError> {
        self.scores(ids)
    }
    fn scores(&self, ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        let slice = ids.as_slice();
        match self.kind {
            "c1" if slice.len() == 12 && slice.first() == Some(&i64::from(b's')) => {
                Ok(vec![0.75, 0.125, 0.125])
            }
            "c3" if slice.len() == 16 && slice.last() == Some(&0) => Ok(vec![0.125, 0.75, 0.125]),
            "autoregressive" if slice.len() == 12 && slice.first() == Some(&i64::from(b't')) => {
                Ok(vec![0.25, 0.25, 0.5])
            }
            _ => Err(BackendError::Failed),
        }
    }
}

const DEFINITION_JSON: &str = r#"{
  "schema": "fandhe-edge-model-definition/v1",
  "name": "kind-contract-invariance-test",
  "version": 1,
  "judgment_type": "single_select",
  "options": [
    {"id": "positive", "display_name": "Positive", "description": "positive input"},
    {"id": "negative", "display_name": "Negative", "description": "negative input"},
    {"id": "neutral", "display_name": "Neutral", "description": "neutral input"}
  ],
  "io": {"input": "bytes"}
}"#;

/// REQ-19・REQ-21・REQ-33・TASK-19.4: 種類ごとに前処理・backend の経路が異なり
/// （スタブ。本物の種類別 ONNX 推論は未実装）スコアも違っても、`infer` の
/// stdout は同じ骨格（キー順 `id`→`status`→`predicted_label`→`scores`、`scores` は
/// 選択肢の宣言順）の 1 行で、exit 0、種類名を含まないこと。
#[test]
fn req19_infer_output_schema_is_kind_invariant() {
    let definition = Definition::parse(DEFINITION_JSON).expect("valid definition");
    let options = definition.options();
    // (種類, スコア, 期待ラベル, 期待する scores の JSON 本体)。値は 2 進で正確に表せるものにする。
    let cases: [(&str, [f64; 3], &str, &str); 3] = [
        (
            "c1",
            [0.75, 0.125, 0.125],
            "positive",
            "\"positive\":0.75,\"negative\":0.125,\"neutral\":0.125",
        ),
        (
            "c3",
            [0.125, 0.75, 0.125],
            "negative",
            "\"positive\":0.125,\"negative\":0.75,\"neutral\":0.125",
        ),
        (
            "autoregressive",
            [0.25, 0.25, 0.5],
            "neutral",
            "\"positive\":0.25,\"negative\":0.25,\"neutral\":0.5",
        ),
    ];
    for (kind, scores, label, score_body) in cases {
        let pipeline = InferencePipeline::new(KindPreprocessor { kind }, KindStubBackend { kind });
        let prediction = pipeline
            .infer_one("sample input")
            .expect("inference succeeds");
        let predicted = options
            .get(prediction.label_index())
            .map(|choice| choice.id.as_str())
            .expect("label index within options");
        assert_eq!(predicted, label, "{kind}");
        // 種類ごとの経路が想定どおりのスコアを出すこと（経路が種類で分岐している確認）。
        // 浮動小数は == で比較せず、許容差 1e-9 を要素ごとの差で判定する。
        assert_eq!(prediction.scores().len(), scores.len(), "{kind}");
        for (actual, expected) in prediction.scores().iter().zip(scores.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-9,
                "{kind}: {actual} vs {expected}"
            );
        }

        let result = JudgmentResult::new(options, "input-001", predicted, prediction.scores())
            .expect("valid judgment result");
        let mut buffer: Vec<u8> = Vec::new();
        let exit = write_ok_judgment(&mut buffer, &result).expect("write must succeed");
        assert_eq!(exit, ExitCode::Ok, "{kind}");
        let text = String::from_utf8(buffer).expect("UTF-8");

        let expected = format!(
            "{{\"id\":\"input-001\",\"status\":\"ok\",\"predicted_label\":\"{label}\",\"scores\":{{{score_body}}}}}\n"
        );
        assert_eq!(text, expected, "{kind}");
        assert_eq!(text.matches('\n').count(), 1, "{kind}");
        assert!(!text.contains(kind), "kind leaked into output: {kind}");
    }
}
