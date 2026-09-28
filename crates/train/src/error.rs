//! 学習リクエスト・結果 JSON の検証エラー（REQ-21・REQ-39）。
//!
//! [`crate::request::TrainRequest`]・[`crate::result::TrainOutcome`] の組み
//! 立て・解析経路で発生するエラーを表す。外部入力（学習リクエスト・ワーカー
//! 出力）の経路のため `panic`・`unwrap` はせず、すべて `Result` で返す
//! （`.claude/rules/coding-rust.md`）。メッセージは英語で、ラベル本文・
//! `config` の値・パス文字列などデータ本文を含めない
//! （`.claude/rules/security.md`）。

use fandhe_edge_core::exitcode::ExitCode;

/// 学習リクエストの組み立て（[`crate::request::TrainRequest::new`]・
/// [`crate::request::TrainRequest::from_json_slice`]）で発生するエラー。
///
/// `reason_code()` は学習ワーカー（`contract.py`・`guard.py`）が同じ入力に
/// 対して返す `code` と揃える（`invalid_path`・`limit_exceeded`・
/// `invalid_request`）。パス以外の検証は Python 側の型検査・範囲検査より
/// 先に構文だけを見るため、Python が `invalid_path` を返すケースと Rust が
/// `invalid_request` を返すケースが一致しないことがある（本 crate はそれを
/// 「Rust 側の方が厳しい」区別として単体テストで確認し、共有 fixture には
/// 両者が一致するケースだけを載せる。issue #177 実装計画）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrainRequestError {
    /// リクエストの生バイト列が [`crate::limits::MAX_REQUEST_BYTES`] を超える
    /// （解析前・直列化後のいずれの検査でも到達しうる）。
    TooLarge { size: usize, limit: usize },
    /// UTF-8 として読めない。
    NotUtf8,
    /// JSON として解析できない（構文エラー・重複キー等）。
    NotJson { message: String },
    /// トップレベルが JSON オブジェクトでない。
    NotObject,
    /// `schema_version` が存在しない・整数でない・`1` 以外。
    UnsupportedSchemaVersion,
    /// 未知のフィールドを含む。
    UnknownField { field: &'static str },
    /// `kind` が空文字列、または存在しない。
    EmptyKind,
    /// `kind_version` が非負整数として読めない。
    InvalidKindVersion,
    /// `config` が JSON オブジェクトでない。
    ConfigNotObject,
    /// `label_order` が範囲外の件数。
    LabelOrderCount { actual: usize },
    /// `label_order[index]` が空文字列。
    LabelOrderEmptyLabel { index: usize },
    /// `label_order[index]` が [`crate::limits::MAX_LABEL_BYTES`] を超える。
    LabelOrderLabelTooLong { index: usize },
    /// `label_order` 内に重複するラベルがある。
    LabelOrderDuplicateLabel { index: usize },
    /// `label_order` がリストでない。
    LabelOrderNotArray,
    /// `max_bytes` が整数として読めない、または範囲外。
    InvalidMaxBytes,
    /// `seed` が整数として読めない、または範囲外。
    InvalidSeed,
    /// `device` が `"cpu"`／`"gpu"` のいずれでもない。
    InvalidDevice,
    /// `time_limit_seconds` が範囲外（指定された場合のみ検査）。
    InvalidTimeLimitSeconds,
    /// `rss_limit_bytes` が範囲外（指定された場合のみ検査）。
    InvalidRssLimitBytes,
    /// `root`・`train_path`・`out_dir` の経路の構文が不正
    /// （空・NUL・絶対 / 相対の取り違え・`..` 構成要素・`.` のみ等）。
    InvalidPath { field: &'static str },
    /// `serde_json` による直列化に失敗した（通常到達しない防御的な分岐）。
    SerializeFailed,
}

impl std::fmt::Display for TrainRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainRequestError::TooLarge { size, limit } => {
                write!(f, "train request exceeds {limit} bytes limit (got {size})")
            }
            TrainRequestError::NotUtf8 => write!(f, "train request is not valid utf-8"),
            TrainRequestError::NotJson { message } => {
                write!(f, "train request is not valid JSON: {message}")
            }
            TrainRequestError::NotObject => write!(f, "train request must be a JSON object"),
            TrainRequestError::UnsupportedSchemaVersion => {
                write!(
                    f,
                    "train request schema_version must be integer {}",
                    crate::limits::REQUEST_SCHEMA_VERSION
                )
            }
            TrainRequestError::UnknownField { field } => {
                write!(f, "train request has an unknown field: {field}")
            }
            TrainRequestError::EmptyKind => write!(f, "kind must be a non-empty string"),
            TrainRequestError::InvalidKindVersion => {
                write!(f, "kind_version must be a non-negative integer")
            }
            TrainRequestError::ConfigNotObject => write!(f, "config must be a JSON object"),
            TrainRequestError::LabelOrderCount { actual } => write!(
                f,
                "label_order length out of range [{}, {}] (got {actual})",
                crate::limits::MIN_LABELS,
                crate::limits::MAX_LABELS
            ),
            TrainRequestError::LabelOrderEmptyLabel { index } => {
                write!(f, "label_order[{index}] must be a non-empty string")
            }
            TrainRequestError::LabelOrderLabelTooLong { index } => write!(
                f,
                "label_order[{index}] exceeds {} utf-8 bytes",
                crate::limits::MAX_LABEL_BYTES
            ),
            TrainRequestError::LabelOrderDuplicateLabel { index } => {
                write!(f, "label_order[{index}] is a duplicate")
            }
            TrainRequestError::LabelOrderNotArray => write!(f, "label_order must be an array"),
            TrainRequestError::InvalidMaxBytes => write!(
                f,
                "max_bytes must be an integer in [{}, {}]",
                crate::limits::MIN_MAX_BYTES,
                crate::limits::MAX_MAX_BYTES
            ),
            TrainRequestError::InvalidSeed => write!(
                f,
                "seed must be an integer in [{}, {}]",
                crate::limits::MIN_SEED,
                crate::limits::MAX_SEED
            ),
            TrainRequestError::InvalidDevice => write!(
                f,
                "device must be one of {:?}",
                crate::limits::ALLOWED_DEVICES
            ),
            TrainRequestError::InvalidTimeLimitSeconds => write!(
                f,
                "time_limit_seconds must be an integer in [1, {}]",
                crate::limits::MAX_TRAIN_WALL_SECONDS
            ),
            TrainRequestError::InvalidRssLimitBytes => write!(
                f,
                "rss_limit_bytes must be an integer in [1, {}]",
                crate::limits::MAX_TRAIN_RSS_BYTES
            ),
            TrainRequestError::InvalidPath { field } => {
                write!(f, "{field} has an invalid path")
            }
            TrainRequestError::SerializeFailed => {
                write!(f, "failed to serialize train request")
            }
        }
    }
}

impl std::error::Error for TrainRequestError {}

impl TrainRequestError {
    /// 学習ワーカー（`contract.py`・`guard.py`）と同じ語彙の機械可読コード。
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            TrainRequestError::InvalidPath { .. } => "invalid_path",
            TrainRequestError::TooLarge { .. } => "limit_exceeded",
            TrainRequestError::NotUtf8
            | TrainRequestError::NotJson { .. }
            | TrainRequestError::NotObject
            | TrainRequestError::UnsupportedSchemaVersion
            | TrainRequestError::UnknownField { .. }
            | TrainRequestError::EmptyKind
            | TrainRequestError::InvalidKindVersion
            | TrainRequestError::ConfigNotObject
            | TrainRequestError::LabelOrderCount { .. }
            | TrainRequestError::LabelOrderEmptyLabel { .. }
            | TrainRequestError::LabelOrderLabelTooLong { .. }
            | TrainRequestError::LabelOrderDuplicateLabel { .. }
            | TrainRequestError::LabelOrderNotArray
            | TrainRequestError::InvalidMaxBytes
            | TrainRequestError::InvalidSeed
            | TrainRequestError::InvalidDevice
            | TrainRequestError::InvalidTimeLimitSeconds
            | TrainRequestError::InvalidRssLimitBytes
            | TrainRequestError::SerializeFailed => "invalid_request",
        }
    }

    /// REQ-21 の終了コードへの対応づけ。`TooLarge` のみ `LimitExceeded`（20）、
    /// それ以外はすべて `InvalidInput`（64）。
    #[must_use]
    pub const fn exit_code(&self) -> ExitCode {
        match self {
            TrainRequestError::TooLarge { .. } => ExitCode::LimitExceeded,
            _ => ExitCode::InvalidInput,
        }
    }
}

/// 学習ワーカーの標準出力（結果 JSON）の解析（[`crate::result::TrainOutcome`]）
/// で発生するエラー。ワーカー出力は信頼しない外部入力として扱う（fail-closed。
/// `.claude/rules/security.md`）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrainResultError {
    /// 標準出力が [`crate::limits::MAX_RESULT_BYTES`] を超える。
    TooLarge { size: usize, limit: usize },
    /// UTF-8 として読めない。
    NotUtf8,
    /// 空行を除いた行数がちょうど 1 行でない（0 行または 2 行以上）。
    NotExactlyOneLine { lines: usize },
    /// JSON として解析できない。
    NotJson { message: String },
    /// トップレベルが JSON オブジェクトでない。
    NotObject,
    /// 未知のフィールドを含む。
    UnknownField { field: &'static str },
    /// `status` が `"ok"`／`"error"` のいずれでもない、または各 status に
    /// 必要なフィールドの組み合わせを満たさない。
    MalformedOutcome,
    /// `artifact` 内のフィールドが不正（型・範囲・許可値のいずれか）。
    MalformedArtifact { field: &'static str },
    /// `code`（[`crate::result::WorkerFailure::code`]）が
    /// `[a-z_]+`・非空・64 バイト以下の規則を満たさない。
    InvalidFailureCode,
    /// `message` が上限（4 KiB）を超える。
    FailureMessageTooLong,
}

impl std::fmt::Display for TrainResultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainResultError::TooLarge { size, limit } => {
                write!(f, "train result exceeds {limit} bytes limit (got {size})")
            }
            TrainResultError::NotUtf8 => write!(f, "train result is not valid utf-8"),
            TrainResultError::NotExactlyOneLine { lines } => write!(
                f,
                "train result must be exactly one JSON line (got {lines})"
            ),
            TrainResultError::NotJson { message } => {
                write!(f, "train result is not valid JSON: {message}")
            }
            TrainResultError::NotObject => write!(f, "train result must be a JSON object"),
            TrainResultError::UnknownField { field } => {
                write!(f, "train result has an unknown field: {field}")
            }
            TrainResultError::MalformedOutcome => {
                write!(f, "train result has a malformed status/field combination")
            }
            TrainResultError::MalformedArtifact { field } => {
                write!(f, "train result artifact field is malformed: {field}")
            }
            TrainResultError::InvalidFailureCode => {
                write!(
                    f,
                    "train result code must match [a-z_]+ and be 1..=64 bytes"
                )
            }
            TrainResultError::FailureMessageTooLong => {
                write!(f, "train result message exceeds size limit")
            }
        }
    }
}

impl std::error::Error for TrainResultError {}

/// `serde_json` のエラーメッセージをそのまま埋め込むと、失敗した値の一部が
/// 引用符付きで含まれることがある（例: 型不一致の実測値）。データ本文では
/// ないが念のため長さを上限で切り詰める（`errors.py::truncate_for_message`
/// と同じ考え方。`.claude/rules/security.md`）。学習リクエスト
/// （[`crate::request::TrainRequest::from_json_slice`]）・学習ワーカー標準
/// 出力（[`crate::result::TrainOutcome::from_worker_stdout`]）の両方が同じ
/// リスク（信頼しない外部入力を解析する serde_json のエラー）を持つため、
/// ここに集約して両者から呼ぶ。
const MAX_EMBEDDED_ERROR_MESSAGE_CHARS: usize = 200;

/// [`MAX_EMBEDDED_ERROR_MESSAGE_CHARS`] を超える分を切り詰めた
/// `serde_json::Error` の表示用メッセージを返す。
pub(crate) fn sanitize_serde_error(err: &serde_json::Error) -> String {
    let message = err.to_string();
    if message.chars().count() <= MAX_EMBEDDED_ERROR_MESSAGE_CHARS {
        message
    } else {
        let truncated: String = message
            .chars()
            .take(MAX_EMBEDDED_ERROR_MESSAGE_CHARS)
            .collect();
        format!("{truncated}...(truncated)")
    }
}

impl TrainResultError {
    /// 壊れたワーカー出力はすべて `runtime_error`（supervisor.py が
    /// 妥当な JSON 1 個・既知の終了コードでない出力を扱う場合と同じ扱い）。
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        "runtime_error"
    }

    /// REQ-21 の終了コード。壊れたワーカー出力は常に `RuntimeError`（70）。
    #[must_use]
    pub const fn exit_code(&self) -> ExitCode {
        ExitCode::RuntimeError
    }
}
