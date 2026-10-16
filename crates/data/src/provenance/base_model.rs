//! ベースモデルの来歴記録（REQ-41・REQ-40・TASK-41.2-3・issue #399）。
//!
//! 追加学習（REQ-41）で使うベースモデルの「モデル名・ライセンス表記・取得元・
//! ファイルごとの sha256」を、来歴と配布パッケージのメタデータ（`artifact.json`
//! の `base_model` 欄）で共用する 1 つの JSON 表現にまとめる。
//!
//! # 呼び出し文脈と分担
//!
//! - 形式の許可・経路の閉じ込め・サイズ確認・shard ごとの sha256 計算は
//!   guard 層（`base_model`・`split_files`。#397・#398・#580）の責務。本モジュールは
//!   検査済みの値（相対パス・サイズ・[`Sha256Digest`]）を受け取るところから始まり、
//!   guard への依存は持たない。guard の型からの変換は CLI の配線（TASK-41.5・#425）が行う
//! - ライセンス表記は記録するだけで、可否は判定しない（ライセンス判断は人が行う）
//! - 取得元は通信せず文字列として記録する（REQ-38）。ローカル絶対パス
//!   や資格情報入り URL は配布メタデータへ漏らすため、`https://` 始まりのみ許可し
//!   `@`・`?`・`#` を含む値を拒否する
//! - 読み戻し（JSON → 型）は呼び出し側が無いため未実装

use std::collections::BTreeSet;
use std::fmt;

use fandhe_edge_core::hash::Sha256Digest;
use serde_json::{Map, Value};

use super::ModelName;

/// ライセンス表記の上限バイト長。
const LICENSE_MAX_BYTES: usize = 256;
/// 取得元の上限バイト長。
const ORIGIN_MAX_BYTES: usize = 1024;
/// ファイルパスの上限バイト長。
const PATH_MAX_BYTES: usize = 512;
/// ファイル件数の上限（guard の `MAX_SPLIT_FILES` と同じ値。data は guard に依存しないため複製）。
const MAX_FILES: usize = 256;

/// 記録の構築エラー。`Display` は固定の英語文で、入力値を含めない。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BaseModelProvenanceError {
    /// ライセンス表記が不正。
    InvalidLicense,
    /// 取得元が不正（ローカル絶対パスを含む）。
    InvalidOrigin,
    /// ファイルが 0 件。
    NoFiles,
    /// ファイルが上限件数を超えた。
    TooManyFiles,
    /// `index` 番目のファイルパスが不正。
    InvalidFilePath {
        /// 渡された順の位置。
        index: usize,
    },
    /// `index` 番目のファイルパスが重複。
    DuplicateFilePath {
        /// 渡された順の位置。
        index: usize,
    },
}

impl fmt::Display for BaseModelProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLicense => f.write_str("invalid base model license"),
            Self::InvalidOrigin => f.write_str("invalid base model origin"),
            Self::NoFiles => f.write_str("base model has no files"),
            Self::TooManyFiles => f.write_str("base model has too many files"),
            Self::InvalidFilePath { .. } => f.write_str("invalid base model file path"),
            Self::DuplicateFilePath { .. } => f.write_str("duplicate base model file path"),
        }
    }
}

impl std::error::Error for BaseModelProvenanceError {}

/// ベースモデルの 1 ファイル分の記録（相対パス・サイズ・sha256）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseModelFile {
    path: String,
    size_bytes: u64,
    sha256: Sha256Digest,
}

impl BaseModelFile {
    /// 検査済みの値から構築する。パスの検証は [`BaseModelProvenance::new`] が行う。
    #[must_use]
    pub fn new(path: &str, size_bytes: u64, sha256: Sha256Digest) -> Self {
        Self {
            path: path.to_string(),
            size_bytes,
            sha256,
        }
    }

    /// `/` 区切りの相対パス。
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// バイト長。
    #[must_use]
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// 生バイト列の sha256。
    #[must_use]
    pub fn sha256(&self) -> &Sha256Digest {
        &self.sha256
    }
}

/// ベースモデル 1 つ分の来歴（名前・ライセンス表記・取得元・ファイル群）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseModelProvenance {
    model_name: ModelName,
    license: String,
    origin: String,
    files: Vec<BaseModelFile>,
}

/// 空でなく上限内で、制御文字・前後空白を含まない。
fn text_ok(value: &str, max_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_bytes
        && !value.chars().any(char::is_control)
        && value.trim().len() == value.len()
}

/// 相対パスとして妥当か（空要素・`.`・`..`・先頭 `/`・`\`・制御文字を拒否）。
fn path_ok(path: &str) -> bool {
    text_ok(path, PATH_MAX_BYTES)
        && !path.contains('\\')
        && path
            .split('/')
            .all(|c| !c.is_empty() && c != "." && c != "..")
}

impl BaseModelProvenance {
    /// 検証して構築する。ファイルは `path` のバイト順に整列する（出力の決定性）。
    ///
    /// # Errors
    /// 各項目の不正・ファイル 0 件・上限超過・パス重複。
    pub fn new(
        model_name: ModelName,
        license: &str,
        origin: &str,
        mut files: Vec<BaseModelFile>,
    ) -> Result<Self, BaseModelProvenanceError> {
        use BaseModelProvenanceError as E;
        if !text_ok(license, LICENSE_MAX_BYTES) {
            return Err(E::InvalidLicense);
        }
        // https:// 始まりのみ許可し、資格情報（userinfo の `@`）・クエリ・フラグメントを拒否する。
        // ローカルパス（絶対・相対・Windows ドライブ・UNC）は https:// で始まらないため落ちる。
        // URL としての正しさは通信しない方針（REQ-38）のため検証しない
        if !text_ok(origin, ORIGIN_MAX_BYTES)
            || !origin.starts_with("https://")
            || origin.contains(['@', '?', '#', '\\'])
        {
            return Err(E::InvalidOrigin);
        }
        if files.is_empty() {
            return Err(E::NoFiles);
        }
        if files.len() > MAX_FILES {
            return Err(E::TooManyFiles);
        }
        let mut seen = BTreeSet::new();
        for (index, f) in files.iter().enumerate() {
            if !path_ok(&f.path) {
                return Err(E::InvalidFilePath { index });
            }
            if !seen.insert(f.path.as_str()) {
                return Err(E::DuplicateFilePath { index });
            }
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(Self {
            model_name,
            license: license.to_string(),
            origin: origin.to_string(),
            files,
        })
    }

    /// モデル名。
    #[must_use]
    pub fn model_name(&self) -> &ModelName {
        &self.model_name
    }

    /// ライセンス表記。
    #[must_use]
    pub fn license(&self) -> &str {
        &self.license
    }

    /// 取得元。
    #[must_use]
    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// `path` 順のファイル群。
    #[must_use]
    pub fn files(&self) -> &[BaseModelFile] {
        &self.files
    }
}

/// 来歴・`artifact.json` の `base_model` 欄に共通の JSON 値。キー順は決定的。
#[must_use]
pub fn base_model_to_json_value(p: &BaseModelProvenance) -> Value {
    let files = p
        .files
        .iter()
        .map(|f| {
            let mut m = Map::new();
            m.insert("path".into(), Value::String(f.path.clone()));
            m.insert("sha256".into(), Value::String(f.sha256.to_hex()));
            m.insert("size_bytes".into(), Value::Number(f.size_bytes.into()));
            Value::Object(m)
        })
        .collect();
    let mut root = Map::new();
    root.insert("files".into(), Value::Array(files));
    root.insert("license".into(), Value::String(p.license.clone()));
    root.insert(
        "model_name".into(),
        Value::String(p.model_name.as_str().to_string()),
    );
    root.insert("origin".into(), Value::String(p.origin.clone()));
    Value::Object(root)
}

/// [`base_model_to_json_value`] の 1 行文字列表現。
#[must_use]
pub fn base_model_to_json(p: &BaseModelProvenance) -> String {
    base_model_to_json_value(p).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(p: &str) -> BaseModelFile {
        BaseModelFile::new(p, 1, Sha256Digest::of_bytes(b""))
    }

    fn make(
        license: &str,
        origin: &str,
        files: Vec<BaseModelFile>,
    ) -> Result<(), BaseModelProvenanceError> {
        let name = ModelName::new("m").unwrap();
        BaseModelProvenance::new(name, license, origin, files).map(|_| ())
    }

    #[test]
    fn req41_license_boundaries() {
        let ok = |l: &str| make(l, "https://x", vec![file("a")]);
        assert!(ok(&"a".repeat(256)).is_ok());
        for bad in [
            String::new(),
            "a".repeat(257),
            "a\nb".into(),
            " a".into(),
            "a ".into(),
        ] {
            assert_eq!(ok(&bad), Err(BaseModelProvenanceError::InvalidLicense));
        }
    }

    #[test]
    fn req41_origin_boundaries() {
        let ok = |o: &str| make("MIT", o, vec![file("a")]);
        assert!(ok(&format!("https://{}", "a".repeat(1016))).is_ok());
        assert!(ok("https://huggingface.co/org/model").is_ok());
        for bad in [
            String::new(),
            format!("https://{}", "a".repeat(1017)),
            "https://a\0b".into(),
            " https://a".into(),
            "a".into(),
            "/home/u/m".into(),
            "~/m".into(),
            "file:///m".into(),
            "http://x".into(),
            "https://user:token@host/m".into(),
            "https://host/m?token=1".into(),
            "https://host/m#frag".into(),
            "C:/m".into(),
            "//host/m".into(),
            "./m".into(),
            "../m".into(),
        ] {
            assert_eq!(ok(&bad), Err(BaseModelProvenanceError::InvalidOrigin));
        }
    }

    /// guard の `MAX_SPLIT_FILES` との値の一致を機械照合する（data は guard に依存しないため
    /// ソース文字列で確認する）。
    #[test]
    fn req41_max_files_matches_guard() {
        let src = include_str!("../../../guard/src/split_files.rs");
        let want = format!("pub const MAX_SPLIT_FILES: usize = {MAX_FILES};");
        assert!(src.contains(&want), "{want}");
    }

    #[test]
    fn req41_file_path_rules() {
        let long = "a".repeat(513);
        for bad in [
            "",
            "..",
            "a/../b",
            "/abs",
            "a//b",
            "./a",
            "a\\b",
            "a\0",
            long.as_str(),
        ] {
            assert_eq!(
                make("MIT", "https://x", vec![file(bad)]),
                Err(BaseModelProvenanceError::InvalidFilePath { index: 0 }),
                "{bad:?}"
            );
        }
        let two = vec![
            file("model-00001-of-00002.safetensors"),
            file("sub/w.safetensors"),
        ];
        assert!(make("MIT", "https://x", two).is_ok());
    }

    #[test]
    fn req41_file_count_and_duplicates() {
        let many = |n: usize| (0..n).map(|i| file(&format!("f{i}"))).collect::<Vec<_>>();
        assert_eq!(
            make("MIT", "https://x", vec![]),
            Err(BaseModelProvenanceError::NoFiles)
        );
        assert!(make("MIT", "https://x", many(256)).is_ok());
        assert_eq!(
            make("MIT", "https://x", many(257)),
            Err(BaseModelProvenanceError::TooManyFiles)
        );
        assert_eq!(
            make("MIT", "https://x", vec![file("a"), file("a")]),
            Err(BaseModelProvenanceError::DuplicateFilePath { index: 1 })
        );
    }
}
