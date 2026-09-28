//! 来歴 JSON（PoC-10／PoC-11／PoC-25 の生成ログ `meta.json` と同じ形）の
//! 検証・変換（REQ-40・TASK-40.1-2・issue #75）。
//!
//! 呼び出し文脈: [`crate::ingest::ingest_records`]（本 crate のデータ検査
//! （[`crate::inspect::inspect_records`]）との接続点）から呼ばれる。将来は
//! CLI の `inspect` 工程（TASK-33.x・パス未確定）から、ガード層を通過済みの
//! 来歴 JSON 文字列を受け取って呼ばれる想定。ファイル読み込み・サイズ上限
//! （REQ-39）は本モジュールの責務ではなく、呼び出し側（ガード層）が担う
//! （[`crate`] クレート doc「前提条件」節と同じ方針）。
//!
//! # 粒度（1 回の生成につき 1 レコード）
//!
//! 来歴は **1 回の生成（1 データセット・1 JSONL ファイル）につき 1 レコード**
//! であり、JSONL の行ごとではない（PoC の `meta.json` と同じ粒度）。
//! [`parse_provenance_json`] は来歴 JSON 全体を 1 回だけ解析する。
//!
//! # 受け付けるキーと無視するキー
//!
//! PoC-10／PoC-11／PoC-25 の生成ログ（`meta.json`）と同じキー名を受け付け、
//! 既存の生成ログをそのまま取り込めるようにする。
//!
//! | 入力キー | 変換先 |
//! | -------- | ------ |
//! | `model_requested` | [`crate::provenance::ModelName`]（`model_observed` は使わない） |
//! | `prompt_sha256` | [`crate::provenance::PromptHash`]（計算済みハッシュを受け取るのみ） |
//! | `started_utc` | [`crate::provenance::GeneratedAt`]（`Z`／`+00:00` のみ） |
//! | `usage_observed` | `null` → [`crate::provenance::TokenCount::Unobserved`]、object → `Observed`（`input_tokens`・`output_tokens` のみ採用） |
//!
//! 上記以外のキー（`cmd`・`cwd`・`ended_utc`・`returncode`・`prompt_file`
//! 等。`cmd`・`cwd` には絶対パス等が入りうる）は読み取らず、記録にも
//! 残さない。`usage_observed` 内の `cached_input_tokens`・
//! `cache_write_input_tokens`・`reasoning_output_tokens` も同様に無視する
//! （[`crate::provenance::TokenUsage`] の doc の方針）。
//!
//! `usage_observed` キー自体が欠落している場合は
//! [`ProvenanceIngestError::MissingField`] とする（明示的な `null` =
//! 観測できなかった、と区別するため）。
//!
//! # 範囲外（本モジュールでは実装しない）
//!
//! - 指示文本文から sha256 を計算する処理（`sha2` の data 層配置・
//!   `data → core` 依存はユーザー承認事項。本モジュールは計算済みハッシュ
//!   〔`prompt_sha256`〕の取り込みのみ行う）
//! - `source`（生成元）フィールド・来歴が無いデータの拒否・外部 LLM 出力の
//!   既定拒否（TASK-40.2 の範囲。PoC-20 ケース 9）
//! - 記録 JSON（[`provenance_to_json`] の出力）を読み戻す関数
//!   （必要になった時点で追加する。YAGNI）

use std::fmt;

use serde_json::{Map, Value};

use crate::json_keys::has_duplicate_key;
use crate::provenance::{
    GeneratedAt, ModelName, ProvenanceError, ProvenanceRecord, TokenCount, TokenUsage,
};

/// 来歴 JSON の検証・構築時のエラー。
///
/// `Display` は固定の英語文言のみを返し、入力値（モデル名・日時文字列・
/// token 数等）を一切含めない（[`ProvenanceError`] と同じ方針。
/// `.claude/rules/security.md`「データ本文をログ・エラーメッセージへ
/// 転記しない」）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProvenanceIngestError {
    /// 来歴 JSON 全体が JSON として解釈できなかった。
    MalformedJson,
    /// JSON としては解釈できたがトップレベルが object ではなかった。
    NotAnObject,
    /// トップレベルまたはネスト先（`usage_observed` 等）に同一キーが
    /// 複数回出現していた（[`crate::json_keys::has_duplicate_key`] で検出。
    /// 複数行の JSON 全体に対しても機能する）。
    DuplicateKey,
    /// 必須フィールドが存在しない。値はフィールド名（例:
    /// `"model_requested"`・`"usage_observed.input_tokens"`）のみ。
    MissingField(&'static str),
    /// フィールドは存在するが期待した JSON 型と異なる（負数・小数の
    /// token 数を含む。`serde_json::Value::as_u64` は非負整数以外で
    /// `None` を返すため、このエラーが両方をまとめて捕捉する）。
    InvalidFieldType(&'static str),
    /// フィールドの値は正しい JSON 型だったが、既存の検証型
    /// （[`ModelName`]・[`crate::provenance::PromptHash`]・[`GeneratedAt`]）
    /// の検証に失敗した。
    InvalidField {
        field: &'static str,
        source: ProvenanceError,
    },
}

impl fmt::Display for ProvenanceIngestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProvenanceIngestError::MalformedJson => {
                f.write_str("provenance JSON could not be parsed")
            }
            ProvenanceIngestError::NotAnObject => {
                f.write_str("provenance JSON top level must be an object")
            }
            ProvenanceIngestError::DuplicateKey => {
                f.write_str("provenance JSON contains a duplicate key")
            }
            ProvenanceIngestError::MissingField(field) => {
                write!(f, "provenance JSON is missing required field: {field}")
            }
            ProvenanceIngestError::InvalidFieldType(field) => {
                write!(f, "provenance JSON field has an unexpected type: {field}")
            }
            ProvenanceIngestError::InvalidField { field, source } => {
                write!(f, "provenance JSON field {field} is invalid: {source}")
            }
        }
    }
}

impl std::error::Error for ProvenanceIngestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProvenanceIngestError::InvalidField { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl ProvenanceIngestError {
    /// CLI 等の上位層が機械判定に使う内部コード文字列を返す
    /// （[`crate::inspect::AnomalyCode::code`] と同じ流儀）。CLI 接続時は
    /// `invalid_input`（終了コード 64・REQ-21）へ写す想定（写像自体は
    /// TASK-33.x の範囲。本モジュールでは行わない）。
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            ProvenanceIngestError::MalformedJson => "provenance_malformed_json",
            ProvenanceIngestError::NotAnObject => "provenance_not_an_object",
            ProvenanceIngestError::DuplicateKey => "provenance_duplicate_key",
            ProvenanceIngestError::MissingField(_) => "provenance_missing_field",
            ProvenanceIngestError::InvalidFieldType(_) => "provenance_invalid_field_type",
            ProvenanceIngestError::InvalidField { .. } => "provenance_invalid_field",
        }
    }
}

/// 文字列フィールドを取得する（欠落・型不一致を判別して返す）。
fn get_required_str<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, ProvenanceIngestError> {
    match object.get(field) {
        None => Err(ProvenanceIngestError::MissingField(field)),
        Some(value) => value
            .as_str()
            .ok_or(ProvenanceIngestError::InvalidFieldType(field)),
    }
}

/// PoC の生成ログ（`meta.json`）形の来歴 JSON を検証し、[`ProvenanceRecord`]
/// へ変換する。
///
/// 上記モジュール doc の「受け付けるキーと無視するキー」に従い、
/// `model_requested`・`prompt_sha256`・`started_utc`・`usage_observed` の
/// 4 キーのみを読み取る。`unwrap`／`expect`／添字アクセスは使わず、
/// `get`／`as_str`／`as_u64` と `Option` の分岐で明示的に処理する
/// （[`crate::inspect::inspect_records`] と同じ方針）。
pub fn parse_provenance_json(content: &str) -> Result<ProvenanceRecord, ProvenanceIngestError> {
    let value: Value =
        serde_json::from_str(content).map_err(|_| ProvenanceIngestError::MalformedJson)?;

    let object = value
        .as_object()
        .ok_or(ProvenanceIngestError::NotAnObject)?;

    // 重複 JSON キーの検出は木全体（ネスト先の usage_observed を含む）に
    // 対して 1 回で行える（[`has_duplicate_key`]・[`crate::json_keys`] の
    // モジュール doc「任意の深さの重複を検出できる」を参照）。
    if has_duplicate_key(content, &value) {
        return Err(ProvenanceIngestError::DuplicateKey);
    }

    let model_name_str = get_required_str(object, "model_requested")?;
    let model_name =
        ModelName::new(model_name_str).map_err(|source| ProvenanceIngestError::InvalidField {
            field: "model_requested",
            source,
        })?;

    let prompt_hash_str = get_required_str(object, "prompt_sha256")?;
    let prompt_hash =
        crate::provenance::PromptHash::from_hex(prompt_hash_str).map_err(|source| {
            ProvenanceIngestError::InvalidField {
                field: "prompt_sha256",
                source,
            }
        })?;

    let started_utc_str = get_required_str(object, "started_utc")?;
    let generated_at = GeneratedAt::parse_rfc3339_utc(started_utc_str).map_err(|source| {
        ProvenanceIngestError::InvalidField {
            field: "started_utc",
            source,
        }
    })?;

    // usage_observed: キー自体の欠落（MissingField）と、明示的な null
    // （Unobserved）を区別する（モジュール doc「受け付けるキー」参照）。
    let token_count = match object.get("usage_observed") {
        None => return Err(ProvenanceIngestError::MissingField("usage_observed")),
        Some(Value::Null) => TokenCount::Unobserved,
        Some(usage_value) => {
            let usage_object = usage_value
                .as_object()
                .ok_or(ProvenanceIngestError::InvalidFieldType("usage_observed"))?;
            let input_tokens =
                get_usage_field(usage_object, "input_tokens", "usage_observed.input_tokens")?;
            let output_tokens = get_usage_field(
                usage_object,
                "output_tokens",
                "usage_observed.output_tokens",
            )?;
            TokenCount::Observed(TokenUsage::new(input_tokens, output_tokens))
        }
    };

    Ok(ProvenanceRecord::new(
        model_name,
        prompt_hash,
        generated_at,
        token_count,
    ))
}

/// `usage_observed` オブジェクトから 1 フィールド（`input_tokens`・
/// `output_tokens`）を取得する。エラー報告に使うフィールド名
/// （`error_field`）はドット区切りの合成名（例:
/// `"usage_observed.input_tokens"`）で、実際の JSON 上のキー
/// （`raw_field`。例: `"input_tokens"`）とは別に受け取る。
fn get_usage_field(
    usage_object: &Map<String, Value>,
    raw_field: &str,
    error_field: &'static str,
) -> Result<u64, ProvenanceIngestError> {
    match usage_object.get(raw_field) {
        None => Err(ProvenanceIngestError::MissingField(error_field)),
        Some(value) => value
            .as_u64()
            .ok_or(ProvenanceIngestError::InvalidFieldType(error_field)),
    }
}

/// [`ProvenanceRecord`] を記録用の正準 JSON 文字列（1 行）に変換する。
///
/// キーは英語。出力キー名（`model_name`／`generated_at`）は入力キー名
/// （`model_requested`／`started_utc`）とは意図的に異なる（本ツールの
/// 記録形式と PoC ログ形式を分ける）。workspace の `serde_json` は
/// `preserve_order` feature が無効（[`crate::inspect`] doc で確認済み）
/// のため、`Value::Object(..).to_string()` はキーが整列済みで決定的に
/// なる。
#[must_use]
pub fn provenance_to_json(record: &ProvenanceRecord) -> String {
    let mut root = Map::new();
    root.insert(
        "generated_at".to_string(),
        Value::String(record.generated_at().to_rfc3339_utc()),
    );
    root.insert(
        "model_name".to_string(),
        Value::String(record.model_name().as_str().to_string()),
    );
    root.insert(
        "prompt_sha256".to_string(),
        Value::String(record.prompt_hash().to_hex()),
    );

    let mut token_count = Map::new();
    match record.token_count() {
        TokenCount::Observed(usage) => {
            token_count.insert(
                "input_tokens".to_string(),
                Value::Number(usage.input_tokens().into()),
            );
            token_count.insert(
                "output_tokens".to_string(),
                Value::Number(usage.output_tokens().into()),
            );
            token_count.insert("status".to_string(), Value::String("observed".to_string()));
        }
        TokenCount::Unobserved => {
            token_count.insert(
                "status".to_string(),
                Value::String("unobserved".to_string()),
            );
        }
    }
    root.insert("token_count".to_string(), Value::Object(token_count));

    Value::Object(root).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-40: `code()` が返す機械可読な値の一覧を固定する。
    #[test]
    fn req40_code_values_are_stable() {
        assert_eq!(
            ProvenanceIngestError::MalformedJson.code(),
            "provenance_malformed_json"
        );
        assert_eq!(
            ProvenanceIngestError::NotAnObject.code(),
            "provenance_not_an_object"
        );
        assert_eq!(
            ProvenanceIngestError::DuplicateKey.code(),
            "provenance_duplicate_key"
        );
        assert_eq!(
            ProvenanceIngestError::MissingField("x").code(),
            "provenance_missing_field"
        );
        assert_eq!(
            ProvenanceIngestError::InvalidFieldType("x").code(),
            "provenance_invalid_field_type"
        );
        assert_eq!(
            ProvenanceIngestError::InvalidField {
                field: "x",
                source: ProvenanceError::EmptyModelName,
            }
            .code(),
            "provenance_invalid_field"
        );
    }

    /// REQ-40: エラー文言に入力値（モデル名等）が含まれないこと。
    #[test]
    fn req40_error_display_does_not_leak_input_value() {
        let marker = "MARKER_SECRET_VALUE";
        let err = ProvenanceIngestError::InvalidField {
            field: "model_requested",
            source: ProvenanceError::EmptyModelName,
        };
        assert!(!err.to_string().contains(marker));
    }
}
