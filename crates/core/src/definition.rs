//! 定義ファイル（選択肢〔ラベル〕一覧・判定型・入出力構造）のスキーマとパーサ（REQ-15）。
//!
//! # 対応 TASK
//! - TASK-15.3-1（本ファイル）: 型定義と正常系パーサ
//! - TASK-15.3-2: 必須項目欠落・不整合の検証、エラー型の拡張
//! - TASK-15.4: ラベル定義非同梱の `missing_labels` 判定
//! - TASK-15.5: 正準化ハッシュ（`options`・`judgment_type` を用いた作り直し要否判定。
//!   `docs/spec/03-poc/model-lifecycle/scripts/catalog.py` の `canon_hash`/`need_rebuild` を踏襲）
//!
//! # 出典
//! フィールド構成は PoC-19（`03-poc/model-lifecycle/definitions/catalog_*.json`）の
//! カタログ形式に合わせる（TASK-15.5 での再利用のため）。PoC-16 の
//! `core-cli-vertical-slice/core/src/definition.rs` は選択口（学習・選定）の
//! 入口契約であり `data`・`selection`・`acceptance` を含むが、それらはデータ契約・
//! 学習ワーカー層（TASK-16.x 以降）の関心事のためここには持ち込まない。

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// 定義ファイルのスキーマ識別子（版が上がったら値も変える）。
pub const SCHEMA_ID: &str = "fandhe-edge-model-definition/v1";

/// 定義ファイル読み込み時のサイズ上限（暫定値。REQ-39 の資源上限が正式に
/// 決まり次第、値を見直す）。
pub const MAX_DEFINITION_FILE_BYTES: u64 = 1_048_576;

/// 判定型。REQ-15 は「固定選択肢から1つを選ぶ判定」を起点とする（NR-1）。
/// 複数選択（NR-2）・複数項目同時判定（NR-3）・順序値（NR-4）は
/// REQ-15 の対象外（未検証の条件）のため、バリアントを増やさない。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgmentType {
    SingleSelect,
}

/// 選択肢1件（不変 ID・表示名・説明。REQ-15）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Choice {
    pub id: String,
    pub display_name: String,
    pub description: String,
}

/// 入力の表現。README「実装方針（要点）」により byte のみに確定済みで、
/// 現状は唯一のバリアントを持つ（将来の拡張点として enum にしている）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputRepresentation {
    Bytes,
}

/// 入出力のデータ構造（REQ-15）。出力側は `judgment_type`／`options` で
/// 表現済みのため、ここでは入力表現のみを持つ。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IoSchema {
    pub input: InputRepresentation,
}

/// 定義ファイル本体（選択肢一覧・判定型・入出力構造。REQ-15）。
///
/// `options` は選択肢一覧の宣言順を保持する `Vec` である（`HashMap`/`HashSet` にしない）。
/// PoC-9 追補 A-10（`docs/spec/03-poc/evaluation-contract/README.md`）により、
/// 下限基準（majority）のタイブレークはラベル定義の宣言順で解決するため、
/// 順序の破壊は評価契約に影響する。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    pub schema: String,
    pub name: String,
    pub version: u32,
    pub judgment_type: JudgmentType,
    pub options: Vec<Choice>,
    pub io: IoSchema,
}

/// 定義ファイルの読み込み・パース時のエラー。
///
/// 本 TASK（TASK-15.3-1）は正常系のみを対象とするため、ここでは
/// 読み込み失敗・サイズ超過・JSON/スキーマ不整合の3種類に留める。
/// 必須項目欠落・重複 ID 等の詳細な検証バリアントは TASK-15.3-2 が追加し、
/// `missing_labels` 相当の判定は TASK-15.4 が追加する（`#[non_exhaustive]`）。
#[derive(Debug)]
#[non_exhaustive]
pub enum DefinitionError {
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    TooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    Parse {
        source: serde_json::Error,
    },
}

impl std::fmt::Display for DefinitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefinitionError::Read { path, source } => {
                write!(f, "failed to read definition file {path:?}: {source}")
            }
            DefinitionError::TooLarge { path, size, limit } => write!(
                f,
                "definition file {path:?} is too large: {size} bytes (limit: {limit} bytes)"
            ),
            DefinitionError::Parse { source } => {
                write!(f, "failed to parse definition file: {source}")
            }
        }
    }
}

impl std::error::Error for DefinitionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DefinitionError::Read { source, .. } => Some(source),
            DefinitionError::TooLarge { .. } => None,
            DefinitionError::Parse { source } => Some(source),
        }
    }
}

impl Definition {
    /// JSON 文字列から定義ファイルをパースする（正常系。REQ-15）。
    /// 選択肢0件の拒否・必須項目の詳細検証は TASK-15.3-2 で追加する。
    pub fn parse(text: &str) -> Result<Self, DefinitionError> {
        serde_json::from_str(text).map_err(|source| DefinitionError::Parse { source })
    }

    /// パスから定義ファイルを読み込む。読み込み前にファイルサイズを確認し、
    /// 上限（`MAX_DEFINITION_FILE_BYTES`）を超える場合は内容を読まずに拒否する
    /// （REQ-39。無制限アロケーションの防止）。
    /// 経路の閉じ込め（`../` 等の拒否）は操作アダプターのガード層（TASK-39.x）の
    /// 責務であり、ここでは行わない。
    pub fn load(path: &Path) -> Result<Self, DefinitionError> {
        let metadata = std::fs::metadata(path).map_err(|source| DefinitionError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let size = metadata.len();
        if size > MAX_DEFINITION_FILE_BYTES {
            return Err(DefinitionError::TooLarge {
                path: path.to_path_buf(),
                size,
                limit: MAX_DEFINITION_FILE_BYTES,
            });
        }
        let text = std::fs::read_to_string(path).map_err(|source| DefinitionError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_OPTIONS_JSON: &str = r#"{
        "schema": "fandhe-edge-model-definition/v1",
        "name": "sample_topic",
        "version": 1,
        "judgment_type": "single_select",
        "options": [
            { "id": "yes", "display_name": "Yes", "description": "肯定" },
            { "id": "no", "display_name": "No", "description": "否定" }
        ],
        "io": { "input": "bytes" }
    }"#;

    const SINGLE_OPTION_JSON: &str = r#"{
        "schema": "fandhe-edge-model-definition/v1",
        "name": "sample_topic_single",
        "version": 1,
        "judgment_type": "single_select",
        "options": [
            { "id": "only", "display_name": "Only", "description": "唯一の選択肢" }
        ],
        "io": { "input": "bytes" }
    }"#;

    #[test]
    fn req15_parses_definition_with_two_or_more_options() {
        let def = Definition::parse(TWO_OPTIONS_JSON).expect("parse すべき");
        assert_eq!(def.schema, SCHEMA_ID);
        assert_eq!(def.judgment_type, JudgmentType::SingleSelect);
        assert_eq!(def.options.len(), 2);
        assert_eq!(def.options[0].id, "yes");
        assert_eq!(def.options[0].display_name, "Yes");
        assert_eq!(def.options[1].id, "no");
        assert_eq!(def.io.input, InputRepresentation::Bytes);
    }

    #[test]
    fn req15_accepts_definition_with_single_option_boundary() {
        let def = Definition::parse(SINGLE_OPTION_JSON).expect("1件でも受理すべき");
        assert_eq!(def.options.len(), 1);
        assert_eq!(def.options[0].id, "only");
    }

    #[test]
    fn req15_rejects_definition_that_is_not_valid_json() {
        let err = Definition::parse("{not json").unwrap_err();
        assert!(matches!(err, DefinitionError::Parse { .. }));
    }
}
