//! 定義ファイル（選択肢〔ラベル〕一覧・判定型・入出力構造）のスキーマとパーサ（REQ-15）。
//!
//! # 対応 TASK
//! - TASK-15.3-1（本ファイル）: 型定義と正常系パーサ。加えて PR #187 のセキュリティ
//!   レビュー（P0/P1）指摘に基づき、ガード層としての完全性検証（`schema`/`version`
//!   の照合・選択肢の整合性検証）とサイズ上限の TOCTOU 対策を先行実装する
//!   （security.md「ガード層」）
//! - TASK-15.3-2: 上記以外の必須項目欠落・不整合の詳細検証、エラー型の拡張
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

/// `SCHEMA_ID` に対応する定義ファイルの版（`Definition::version` フィールドの
/// 期待値）。異なる版・異なるスキーマ形式の定義ファイルを、検証なしに正常な
/// `Definition` として後続処理（学習ワーカー・CLI）へ渡さないための照合値
/// （security.md「ガード層: 完全性と版」）。
pub const DEFINITION_SCHEMA_VERSION: u32 = 1;

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
/// 本 TASK（TASK-15.3-1）は正常系のみを対象とするが、PR #187 のセキュリティ
/// レビュー（P0/P1）指摘に基づき、ガード層としての完全性検証
/// （`UnsupportedSchema`・`UnsupportedVersion`・選択肢の整合性）を先行実装する。
/// これ以外の必須項目欠落等の詳細な検証バリアントは TASK-15.3-2 が追加し、
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
    /// `schema` フィールドが `SCHEMA_ID` と一致しない（未対応の形式。
    /// security.md「ガード層: 完全性と版」）。
    UnsupportedSchema {
        schema: String,
    },
    /// `version` フィールドが `DEFINITION_SCHEMA_VERSION` と一致しない
    /// （未対応の版。同上）。
    UnsupportedVersion {
        version: u32,
    },
    /// `options` が空で、固定選択肢からの判定が成立しない（REQ-15）。
    EmptyOptions,
    /// `options` 内に空文字列の `id` を持つ選択肢がある（ラベル照合が成立しない）。
    EmptyOptionId,
    /// `options` 内で `id` が重複している（ラベル照合が一意に定まらない）。
    DuplicateOptionId {
        id: String,
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
            DefinitionError::UnsupportedSchema { schema } => {
                write!(
                    f,
                    "unsupported definition schema {schema:?} (expected {SCHEMA_ID:?})"
                )
            }
            DefinitionError::UnsupportedVersion { version } => {
                write!(
                    f,
                    "unsupported definition version {version} (expected {DEFINITION_SCHEMA_VERSION})"
                )
            }
            DefinitionError::EmptyOptions => {
                write!(f, "definition options must not be empty")
            }
            DefinitionError::EmptyOptionId => {
                write!(f, "definition option id must not be empty")
            }
            DefinitionError::DuplicateOptionId { id } => {
                write!(f, "definition option id {id:?} is duplicated")
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
            DefinitionError::UnsupportedSchema { .. } => None,
            DefinitionError::UnsupportedVersion { .. } => None,
            DefinitionError::EmptyOptions => None,
            DefinitionError::EmptyOptionId => None,
            DefinitionError::DuplicateOptionId { .. } => None,
        }
    }
}

impl Definition {
    /// JSON 文字列から定義ファイルをパースし、ガード層としての完全性検証
    /// （`schema`/`version` の照合・選択肢の整合性）を行う（REQ-15）。
    /// 必須項目欠落等のこれ以外の詳細検証は TASK-15.3-2 で追加する。
    pub fn parse(text: &str) -> Result<Self, DefinitionError> {
        let definition: Definition =
            serde_json::from_str(text).map_err(|source| DefinitionError::Parse { source })?;

        // 完全性と版（security.md「ガード層: 完全性と版」）: 異なるスキーマ形式・
        // 将来版の定義をそのまま正常な `Definition` として後続処理へ渡さない。
        if definition.schema != SCHEMA_ID {
            return Err(DefinitionError::UnsupportedSchema {
                schema: definition.schema,
            });
        }
        if definition.version != DEFINITION_SCHEMA_VERSION {
            return Err(DefinitionError::UnsupportedVersion {
                version: definition.version,
            });
        }

        // 選択肢の整合性: 固定選択肢からの選択・ラベル照合が成立する状態
        // （空でない・id が空でない・id が重複しない）であることを検証する
        // （REQ-15 の入出力契約）。
        if definition.options.is_empty() {
            return Err(DefinitionError::EmptyOptions);
        }
        let mut seen_ids = std::collections::HashSet::with_capacity(definition.options.len());
        for choice in &definition.options {
            if choice.id.is_empty() {
                return Err(DefinitionError::EmptyOptionId);
            }
            if !seen_ids.insert(choice.id.as_str()) {
                return Err(DefinitionError::DuplicateOptionId {
                    id: choice.id.clone(),
                });
            }
        }

        Ok(definition)
    }

    /// パスから定義ファイルを読み込む。サイズ確認と内容読み込みを同一の
    /// ファイルハンドルに対して行い、`MAX_DEFINITION_FILE_BYTES` を超える
    /// 場合は上限+1バイトを超えた時点で打ち切って拒否する（REQ-39）。
    ///
    /// `std::fs::metadata` でサイズを確認した後に `read_to_string` がパスを
    /// 再度開く実装は、確認後にファイルが拡大・差し替えられると上限を超えて
    /// 無制限にメモリへ読み込みうる（TOCTOU。security.md「ガード層: 資源の上限」）。
    /// ここでは 1 つの `File` から metadata 取得・`take` による打ち切り読み込みまで
    /// 行い、その間の再オープンを避けることでこの窓を閉じる。
    ///
    /// 経路の閉じ込め（`../` 等の拒否）は操作アダプターのガード層（TASK-39.x）の
    /// 責務であり、ここでは行わない。
    pub fn load(path: &Path) -> Result<Self, DefinitionError> {
        use std::io::Read as _;

        let mut file = std::fs::File::open(path).map_err(|source| DefinitionError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let metadata = file.metadata().map_err(|source| DefinitionError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let reported_size = metadata.len();
        if reported_size > MAX_DEFINITION_FILE_BYTES {
            return Err(DefinitionError::TooLarge {
                path: path.to_path_buf(),
                size: reported_size,
                limit: MAX_DEFINITION_FILE_BYTES,
            });
        }

        // 上限+1バイトまでしか読まない。開いた後にファイルが上限超へ拡大・
        // 差し替えられていても、読み込み量自体を上限近傍で頭打ちにできる。
        let mut buf = Vec::new();
        (&mut file)
            .take(MAX_DEFINITION_FILE_BYTES.saturating_add(1))
            .read_to_end(&mut buf)
            .map_err(|source| DefinitionError::Read {
                path: path.to_path_buf(),
                source,
            })?;
        let actual_size = buf.len() as u64;
        if actual_size > MAX_DEFINITION_FILE_BYTES {
            return Err(DefinitionError::TooLarge {
                path: path.to_path_buf(),
                size: actual_size,
                limit: MAX_DEFINITION_FILE_BYTES,
            });
        }

        let text = String::from_utf8(buf).map_err(|err| DefinitionError::Read {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, err),
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

    /// security.md「ガード層: 完全性と版」: 未対応の `schema` は拒否する。
    #[test]
    fn req15_rejects_unsupported_schema() {
        let json = r#"{
            "schema": "other-schema/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "Yes", "description": "肯定" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let err = Definition::parse(json).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::UnsupportedSchema { schema } if schema == "other-schema/v1"
        ));
    }

    /// security.md「ガード層: 完全性と版」: `DEFINITION_SCHEMA_VERSION` と異なる
    /// `version`（未対応の将来版を含む）は拒否する。
    #[test]
    fn req15_rejects_unsupported_version() {
        let json = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 2,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "Yes", "description": "肯定" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let err = Definition::parse(json).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::UnsupportedVersion { version: 2 }
        ));
    }

    /// REQ-15: `options` が空だと固定選択肢からの判定が成立しないため拒否する。
    #[test]
    fn req15_rejects_empty_options() {
        let json = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [],
            "io": { "input": "bytes" }
        }"#;
        let err = Definition::parse(json).unwrap_err();
        assert!(matches!(err, DefinitionError::EmptyOptions));
    }

    /// REQ-15: `id` が空文字列の選択肢はラベル照合が成立しないため拒否する。
    #[test]
    fn req15_rejects_empty_option_id() {
        let json = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "", "display_name": "Yes", "description": "肯定" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let err = Definition::parse(json).unwrap_err();
        assert!(matches!(err, DefinitionError::EmptyOptionId));
    }

    /// REQ-15: `id` が重複するとラベル照合が一意に定まらないため拒否する。
    #[test]
    fn req15_rejects_duplicate_option_id() {
        let json = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "dup", "display_name": "A", "description": "1件目" },
                { "id": "dup", "display_name": "B", "description": "2件目" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let err = Definition::parse(json).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::DuplicateOptionId { id } if id == "dup"
        ));
    }

    /// REQ-39: サイズ確認と読み込みを同一ファイルハンドルに対して行い、
    /// 上限（`MAX_DEFINITION_FILE_BYTES`）超のファイルを内容の全量読み込み
    /// なしに拒否することを確認する（TOCTOU 対策の回帰）。
    #[test]
    fn req39_load_rejects_file_over_size_limit_without_full_read() {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-core-definition-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .expect("system clock should be after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir を作成できるはず");
        let path = dir.join("oversized.json");
        // 上限を確実に超える内容（JSON としては不正でもよい。全量読み込み前に
        // サイズで拒否されることを確認するため）。
        let oversized = vec![b'a'; usize::try_from(MAX_DEFINITION_FILE_BYTES).unwrap() + 1];
        std::fs::write(&path, &oversized).expect("テスト用ファイルを書き込めるはず");

        let err = Definition::load(&path).unwrap_err();
        assert!(
            matches!(err, DefinitionError::TooLarge { limit, .. } if limit == MAX_DEFINITION_FILE_BYTES)
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// REQ-15/REQ-39: 上限以下の正常な定義ファイルは `load` で読み込める
    /// （`parse` と同じ検証を通る）ことを確認する。
    #[test]
    fn req15_load_reads_and_validates_definition_from_path() {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-core-definition-test-ok-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .expect("system clock should be after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir を作成できるはず");
        let path = dir.join("definition.json");
        std::fs::write(&path, TWO_OPTIONS_JSON).expect("テスト用ファイルを書き込めるはず");

        let def = Definition::load(&path).expect("上限以下の正常な定義は読み込めるはず");
        assert_eq!(def.options.len(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }
}
