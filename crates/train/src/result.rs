//! 学習ワーカーの標準出力（結果 JSON）の検証済み型（REQ-21・REQ-39）。
//!
//! `trainer/src/fandhe_edge_trainer/cli.py::main`・`supervisor.py` が
//! `print(json.dumps(...))` で出す JSON 1 行を、信頼しない外部入力として
//! 解析する（学習ワーカーは Rust 側から見て子プロセス。#178 で子プロセス
//! 起動へ配線予定）。成功時は `artifact.py::build_artifact` と同じ 11 項目の
//! 成果物記録、失敗時は `{"status":"error","code":...,"message":...}` を扱う。

use serde::{Deserialize, Serialize};

use crate::error::{TrainResultError, sanitize_serde_error};
use crate::limits::MAX_RESULT_BYTES;
use crate::request::LabelOrder;

/// `artifact.py::build_artifact` の `output_type` フィールド。現状は
/// `"choice"`（固定選択肢からの単一選択の出力）のみが有効。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputType {
    Choice,
}

/// `model.onnx` の SHA-256（小文字 16 進数 64 桁）。
///
/// `artifact.py` のドキュメントが明記するとおり、本値は「ワーカーが書いた
/// バイト列」と「実際に読めるバイト列」の自己整合性の記録に過ぎず、
/// 配布パッケージ読み込み時の完全性検証（REQ-39「完全性と版」）は推論
/// ランタイム・パッケージ層（TASK-28・TASK-30.x）の責務。本型は形式
/// （小文字 16 進数 64 桁）だけを検査し、実ファイルとの照合はしない
/// （実装済みを装わない）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxSha256(String);

impl OnnxSha256 {
    fn parse(value: String) -> Result<Self, TrainResultError> {
        let is_valid = value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !is_valid {
            return Err(TrainResultError::MalformedArtifact {
                field: "onnx_sha256",
            });
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for OnnxSha256 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

/// 成功時の成果物記録（11 項目。`artifact.py::build_artifact` と同じ名前・型）。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArtifactRecord {
    kind: String,
    kind_version: u32,
    selector_version: String,
    config: serde_json::Map<String, serde_json::Value>,
    label_order: LabelOrder,
    output_type: OutputType,
    max_bytes: u32,
    onnx_file: String,
    onnx_sha256: OnnxSha256,
    created_utc: String,
    candidate_label: String,
}

impl ArtifactRecord {
    pub fn kind(&self) -> &str {
        &self.kind
    }
    pub fn kind_version(&self) -> u32 {
        self.kind_version
    }
    pub fn selector_version(&self) -> &str {
        &self.selector_version
    }
    pub fn config(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.config
    }
    pub fn label_order(&self) -> &LabelOrder {
        &self.label_order
    }
    pub fn output_type(&self) -> OutputType {
        self.output_type
    }
    pub fn max_bytes(&self) -> u32 {
        self.max_bytes
    }
    pub fn onnx_file(&self) -> &str {
        &self.onnx_file
    }
    pub fn onnx_sha256(&self) -> &OnnxSha256 {
        &self.onnx_sha256
    }
    /// `YYYY-MM-DDTHH:MM:SSZ` 形式（`artifact.py::now_utc`）の構文検査のみ
    /// 行う。時刻として解釈・検証はしない。
    pub fn created_utc(&self) -> &str {
        &self.created_utc
    }
    pub fn candidate_label(&self) -> &str {
        &self.candidate_label
    }
}

/// 失敗時のエラー（`{"status":"error","code":...,"message":...}`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerFailure {
    code: String,
    message: String,
}

/// `code` の上限バイト数（Rust 側だけの防御。共有スキーマに定義はないが、
/// リクエストの `MAX_LABEL_BYTES` と同水準に抑える）。
const MAX_FAILURE_CODE_BYTES: usize = 64;
/// `message` の上限バイト数（Rust 側だけの防御。`errors.py` はメッセージへ
/// データ本文を含めない方針だが、想定外の長大化に備える）。
const MAX_FAILURE_MESSAGE_BYTES: usize = 4 * 1024;

impl WorkerFailure {
    fn parse(code: String, message: String) -> Result<Self, TrainResultError> {
        let code_is_valid = !code.is_empty()
            && code.len() <= MAX_FAILURE_CODE_BYTES
            && code.bytes().all(|b| b == b'_' || b.is_ascii_lowercase());
        if !code_is_valid {
            return Err(TrainResultError::InvalidFailureCode);
        }
        if message.len() > MAX_FAILURE_MESSAGE_BYTES {
            return Err(TrainResultError::FailureMessageTooLong);
        }
        Ok(Self { code, message })
    }

    pub fn code(&self) -> &str {
        &self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// 学習ワーカーの結果（成功／失敗の 2 択）。
#[derive(Debug, Clone, PartialEq)]
pub enum TrainOutcome {
    Ok {
        artifact_dir: String,
        artifact: Box<ArtifactRecord>,
    },
    Error(WorkerFailure),
}

/// `from_worker_stdout` の内部専用中間表現。`status` と各フィールドの組み
/// 合わせはここでは検査せず、`TrainOutcome::from_worker_stdout` 側で
/// 明示的に判定する（serde の internally-tagged enum + `deny_unknown_fields`
/// の相互作用に頼らないため。issue #177 実装計画）。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOutcome {
    status: String,
    artifact_dir: Option<String>,
    artifact: Option<RawArtifact>,
    code: Option<String>,
    message: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArtifact {
    kind: String,
    kind_version: u32,
    selector_version: String,
    #[serde(default)]
    config: serde_json::Map<String, serde_json::Value>,
    label_order: Vec<String>,
    output_type: OutputType,
    max_bytes: u32,
    onnx_file: String,
    onnx_sha256: String,
    created_utc: String,
    candidate_label: String,
}

/// `created_utc` の構文検査（`YYYY-MM-DDTHH:MM:SSZ`。20 文字固定）。時刻と
/// して解釈しない（`artifact.py::now_utc` の出力形式のみを確認する）。
fn is_syntactically_valid_created_utc(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 20 {
        return false;
    }
    let is_digit = |i: usize| bytes.get(i).is_some_and(u8::is_ascii_digit);
    let is_byte = |i: usize, expected: u8| bytes.get(i) == Some(&expected);
    (0..4).all(is_digit)
        && is_byte(4, b'-')
        && (5..7).all(is_digit)
        && is_byte(7, b'-')
        && (8..10).all(is_digit)
        && is_byte(10, b'T')
        && (11..13).all(is_digit)
        && is_byte(13, b':')
        && (14..16).all(is_digit)
        && is_byte(16, b':')
        && (17..19).all(is_digit)
        && is_byte(19, b'Z')
}

impl TrainOutcome {
    /// 学習ワーカーの標準出力（信頼しない外部入力）から結果を解析する。
    ///
    /// `supervisor.py`（子プロセスが自分で終了した場合の検査）と同じ規則:
    /// (1) バイト長を [`MAX_RESULT_BYTES`] と照合 → (2) UTF-8 として読める →
    /// (3) 空行を除いてちょうど 1 行 → (4) JSON として解析可能 →
    /// (5) `status`／各フィールドの組み合わせが妥当。いずれかを満たさない
    /// 場合は `TrainResultError`（呼び出し元は `reason_code()=="runtime_error"`・
    /// `exit_code()==RuntimeError` として扱う。#178 の対象）。
    pub fn from_worker_stdout(bytes: &[u8]) -> Result<Self, TrainResultError> {
        if bytes.len() > MAX_RESULT_BYTES {
            return Err(TrainResultError::TooLarge {
                size: bytes.len(),
                limit: MAX_RESULT_BYTES,
            });
        }
        let text = std::str::from_utf8(bytes).map_err(|_| TrainResultError::NotUtf8)?;
        let non_empty_lines: Vec<&str> = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect();
        if non_empty_lines.len() != 1 {
            return Err(TrainResultError::NotExactlyOneLine {
                lines: non_empty_lines.len(),
            });
        }
        let Some(only_line) = non_empty_lines.first() else {
            // 直前の長さ検査（`non_empty_lines.len() != 1`）により到達しない
            // 防御的な分岐。添字アクセス `[0]` を避け、外部入力の経路で
            // `unwrap`／`expect`／添字アクセスを使わない方針を保つ
            // （`.claude/rules/coding-rust.md`）。
            return Err(TrainResultError::NotExactlyOneLine { lines: 0 });
        };
        let raw: RawOutcome =
            serde_json::from_str(only_line).map_err(|e| TrainResultError::NotJson {
                message: sanitize_serde_error(&e),
            })?;

        match raw.status.as_str() {
            "ok" => {
                let (Some(artifact_dir), Some(raw_artifact), None, None) =
                    (raw.artifact_dir, raw.artifact, raw.code, raw.message)
                else {
                    return Err(TrainResultError::MalformedOutcome);
                };
                let label_order = LabelOrder::new(raw_artifact.label_order).map_err(|_| {
                    TrainResultError::MalformedArtifact {
                        field: "label_order",
                    }
                })?;
                if !(crate::limits::MIN_MAX_BYTES..=crate::limits::MAX_MAX_BYTES)
                    .contains(&raw_artifact.max_bytes)
                {
                    return Err(TrainResultError::MalformedArtifact { field: "max_bytes" });
                }
                if raw_artifact.onnx_file != "model.onnx" {
                    return Err(TrainResultError::MalformedArtifact { field: "onnx_file" });
                }
                if raw_artifact.candidate_label.is_empty() {
                    return Err(TrainResultError::MalformedArtifact {
                        field: "candidate_label",
                    });
                }
                if !is_syntactically_valid_created_utc(&raw_artifact.created_utc) {
                    return Err(TrainResultError::MalformedArtifact {
                        field: "created_utc",
                    });
                }
                let onnx_sha256 = OnnxSha256::parse(raw_artifact.onnx_sha256)?;
                Ok(TrainOutcome::Ok {
                    artifact_dir,
                    artifact: Box::new(ArtifactRecord {
                        kind: raw_artifact.kind,
                        kind_version: raw_artifact.kind_version,
                        selector_version: raw_artifact.selector_version,
                        config: raw_artifact.config,
                        label_order,
                        output_type: raw_artifact.output_type,
                        max_bytes: raw_artifact.max_bytes,
                        onnx_file: raw_artifact.onnx_file,
                        onnx_sha256,
                        created_utc: raw_artifact.created_utc,
                        candidate_label: raw_artifact.candidate_label,
                    }),
                })
            }
            "error" => {
                let (None, None, Some(code), Some(message)) =
                    (raw.artifact_dir, raw.artifact, raw.code, raw.message)
                else {
                    return Err(TrainResultError::MalformedOutcome);
                };
                Ok(TrainOutcome::Error(WorkerFailure::parse(code, message)?))
            }
            _ => Err(TrainResultError::MalformedOutcome),
        }
    }
}

impl Serialize for TrainOutcome {
    /// 往復検証用（`crates/train/tests/train_contract_fixture.rs`）に実装
    /// する。`supervisor.py`／`cli.py` が実際に出す JSON と同じキー名・
    /// 構造にする。
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            TrainOutcome::Ok {
                artifact_dir,
                artifact,
            } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("status", "ok")?;
                map.serialize_entry("artifact_dir", artifact_dir)?;
                map.serialize_entry("artifact", artifact.as_ref())?;
                map.end()
            }
            TrainOutcome::Error(failure) => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("status", "error")?;
                map.serialize_entry("code", failure.code())?;
                map.serialize_entry("message", failure.message())?;
                map.end()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_OK_JSON: &str = r#"{"status":"ok","artifact_dir":"out","artifact":{"kind":"c3","kind_version":1,"selector_version":"0.1","config":{},"label_order":["a","b"],"output_type":"choice","max_bytes":512,"onnx_file":"model.onnx","onnx_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","created_utc":"2026-09-28T00:00:00Z","candidate_label":"c3"}}"#;

    #[test]
    fn req21_parses_valid_ok_outcome() {
        let outcome =
            TrainOutcome::from_worker_stdout(VALID_OK_JSON.as_bytes()).expect("valid outcome");
        match outcome {
            TrainOutcome::Ok {
                artifact_dir,
                artifact,
            } => {
                assert_eq!(artifact_dir, "out");
                assert_eq!(artifact.kind(), "c3");
                assert_eq!(
                    artifact.label_order().as_slice(),
                    &["a".to_string(), "b".to_string()]
                );
            }
            TrainOutcome::Error(_) => panic!("expected Ok"),
        }
    }

    #[test]
    fn req21_parses_valid_error_outcome() {
        let json = r#"{"status":"error","code":"invalid_request","message":"file not readable: FileNotFoundError"}"#;
        let outcome = TrainOutcome::from_worker_stdout(json.as_bytes()).expect("valid outcome");
        match outcome {
            TrainOutcome::Error(failure) => {
                assert_eq!(failure.code(), "invalid_request");
                assert_eq!(failure.message(), "file not readable: FileNotFoundError");
            }
            TrainOutcome::Ok { .. } => panic!("expected Error"),
        }
    }

    /// REQ-39: 2 行出力は「壊れたワーカー出力」として `runtime_error` になる。
    #[test]
    fn req39_rejects_two_line_output() {
        let json = format!("{VALID_OK_JSON}\n{VALID_OK_JSON}");
        let err = TrainOutcome::from_worker_stdout(json.as_bytes()).unwrap_err();
        assert_eq!(err.reason_code(), "runtime_error");
        assert!(matches!(
            err,
            TrainResultError::NotExactlyOneLine { lines: 2 }
        ));
    }

    /// REQ-39: 未知フィールドを含む出力は拒否する（fail-closed）。
    #[test]
    fn req39_rejects_unknown_top_level_field() {
        let json = r#"{"status":"ok","artifact_dir":"out","artifact":{},"extra":1}"#;
        let err = TrainOutcome::from_worker_stdout(json.as_bytes()).unwrap_err();
        assert_eq!(err.reason_code(), "runtime_error");
    }

    /// REQ-39: `onnx_sha256` の形式不正（短すぎる・大文字を含む）は拒否する。
    #[test]
    fn req39_rejects_malformed_onnx_sha256() {
        let json = VALID_OK_JSON.replace(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
            "DEADBEEF",
        );
        let err = TrainOutcome::from_worker_stdout(json.as_bytes()).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::MalformedArtifact {
                field: "onnx_sha256"
            }
        ));
    }

    /// REQ-39: `label_order` が上限外の成果物は拒否する。
    #[test]
    fn req39_rejects_artifact_with_invalid_label_order() {
        let json = VALID_OK_JSON.replace(r#""label_order":["a","b"]"#, r#""label_order":["only"]"#);
        let err = TrainOutcome::from_worker_stdout(json.as_bytes()).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::MalformedArtifact {
                field: "label_order"
            }
        ));
    }

    /// REQ-39: `status` が不正な値は拒否する。
    #[test]
    fn req39_rejects_invalid_status() {
        let json = r#"{"status":"pending"}"#;
        let err = TrainOutcome::from_worker_stdout(json.as_bytes()).unwrap_err();
        assert!(matches!(err, TrainResultError::MalformedOutcome));
    }

    /// REQ-39: サイズ超過（`MAX_RESULT_BYTES` 超）は `limit_exceeded` 相当
    /// ではなく `runtime_error`（壊れたワーカー出力として扱う。
    /// `WorkerFailure` を返す `error` ステータスとは別の、監視側の検査）。
    #[test]
    fn req39_rejects_output_over_size_limit() {
        let huge = "a".repeat(MAX_RESULT_BYTES + 1);
        let json = format!(r#"{{"status":"error","code":"x","message":"{huge}"}}"#);
        let err = TrainOutcome::from_worker_stdout(json.as_bytes()).unwrap_err();
        assert!(matches!(err, TrainResultError::TooLarge { .. }));
        assert_eq!(err.reason_code(), "runtime_error");
        assert_eq!(
            err.exit_code(),
            fandhe_edge_core::exitcode::ExitCode::RuntimeError
        );
    }
}
