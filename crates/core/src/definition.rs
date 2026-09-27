//! 定義ファイル（選択肢〔ラベル〕一覧・判定型・入出力構造）のスキーマとパーサ（REQ-15）。
//!
//! # 対応 TASK
//! - TASK-15.3-1（本ファイル）: 型定義と正常系パーサ。加えて PR #187 のセキュリティ
//!   レビュー（P0/P1）指摘に基づき、ガード層としての完全性検証（`schema` の照合・
//!   選択肢の整合性検証）とサイズ上限の TOCTOU 対策を先行実装する
//!   （security.md「ガード層」）。`version` フィールドはカタログ形式のドキュメント
//!   バージョン（`name` と並ぶ台帳上のメタ情報）であり、形式・版の識別は既に
//!   `schema`/`SCHEMA_ID`（例: `.../v1` の版数を含む）が担うため、`version` を
//!   第二のスキーマ固定値として拒否条件には使わない（PR #187 レビュー指摘）。
//!   さらに同 PR の追加レビュー指摘に基づき、`Definition` を `Deserialize` させず
//!   `parse` 限定の未検証中間型（`RawDefinition`）を経由させることでガード層の
//!   迂回を防ぎ、`load` では FIFO 等の特殊ファイルに対する `File::open` の無期限
//!   停止を避ける（`NotRegularFile`。Linux・macOS では `O_NONBLOCK` で開く）
//! - TASK-15.3-2（本ファイル + `diagnose` サブモジュール）: `RawDefinition` への
//!   型付きデシリアライズが `serde_json::Error::classify()` で `Data`（構文
//!   エラーではない）に分類される失敗を、`diagnose` モジュールで
//!   `serde_json::Value` として再走査し、必須項目欠落・型不整合・未知キー・
//!   enum 外の値を型付きの `DefinitionError` バリアントで返す。`name` の
//!   空文字列は `EmptyName` として拒否する
//! - TASK-15.4: ラベル定義非同梱の `missing_labels` 判定
//! - TASK-15.5（本ファイル + `canonical` モジュール）: 定義の同一性
//!   （`DefinitionIdentity`。選択肢 ID の集合＋`judgment_type`）と、定義全体の
//!   正準化ハッシュ（`DefinitionHash`）を提供する。正準化の規則・sha256 計算は
//!   `canonical` モジュールに集約し（REQ-15「正準化の規則を1箇所に集約する」）、
//!   本ファイルには薄いアクセサ（[`Definition::identity`]・
//!   [`Definition::canonical_json`]・[`Definition::canonical_hash`]）のみを置く。
//!   作り直し要否の判定そのもの（TASK-20.1〜20.3）は本 TASK の対象外
//!
//! # 出典
//! フィールド構成は PoC-19（`03-poc/model-lifecycle/definitions/catalog_*.json`）の
//! カタログ形式に合わせる（TASK-15.5 での再利用のため）。PoC-16 の
//! `core-cli-vertical-slice/core/src/definition.rs` は選択口（学習・選定）の
//! 入口契約であり `data`・`selection`・`acceptance` を含むが、それらはデータ契約・
//! 学習ワーカー層（TASK-16.x 以降）の関心事のためここには持ち込まない。

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// `diagnose` は `Definition::parse` の失敗経路（`Category::Data`）でのみ
// 呼ばれる内部専用モジュール（TASK-15.3-2）。`super::` 経由で `FieldPath` 等の
// 補助型・`SCHEMA_ID`・`JudgmentType`/`InputRepresentation` を参照する。
mod diagnose;

/// 定義ファイルのスキーマ識別子（版が上がったら値も変える）。
pub const SCHEMA_ID: &str = "fandhe-edge-model-definition/v1";

/// `SCHEMA_ID` が表す形式・版の参考値（`SCHEMA_ID` 末尾の `/v1` と対応）。
/// スキーマ形式・版の識別は `schema`/`SCHEMA_ID` の一致検査で行い、この定数は
/// `Definition::version`（カタログ上のドキュメントバージョン。`name` と並ぶ
/// メタ情報であり、スキーマ固定値ではない）とは照合しない
/// （security.md「ガード層: 完全性と版」。PR #187 レビュー指摘で `version` の
/// 拒否条件化を撤回）。
#[allow(dead_code)]
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
///
/// フィールドは非公開にしている。全フィールドが `pub` だと、別 crate が
/// `schema` の不一致・空や重複した `options` を持つ値を `Definition { .. }`
/// で直接構築でき、`parse`/`load` が行うガード層の検証（`schema` 照合・
/// 選択肢の整合性・サイズ上限）を丸ごと迂回できてしまう（security.md
/// 「ガード層の迂回」。PR #187 レビュー指摘）。検証済みの値を作る経路は
/// `parse`（`load` も内部で `parse` を呼ぶ）に限定し、値の参照は下記の
/// 読み取り専用アクセサ経由に限る（フィールドへの代入による事後改変も防ぐ）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Definition {
    schema: String,
    name: String,
    version: u32,
    judgment_type: JudgmentType,
    options: Vec<Choice>,
    io: IoSchema,
}

/// `Definition` の未検証の中間表現（デシリアライズ専用）。
///
/// `Definition` 自体には `Deserialize` を実装しない。もし実装すると
/// `serde_json::from_str::<Definition>(text)` のように `Definition::parse` を
/// 経由しない直接デシリアライズが可能になり、`parse` が担うガード層の検証
/// （サイズ上限・`schema` 照合・選択肢の整合性）を丸ごと迂回できてしまう
/// （security.md「ガード層の迂回」。PR #187 レビュー指摘）。
/// この中間型は `parse` の内部でのみ使い、検証を経ないまま外部へ返さない。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDefinition {
    schema: String,
    name: String,
    version: u32,
    judgment_type: JudgmentType,
    options: Vec<Choice>,
    io: IoSchema,
}

/// 定義ファイル内のフィールドの位置を表すパス（TASK-15.3-2）。
///
/// 自由文字列ではなく enum にすることで、壊れた・でっち上げのパスを
/// 表現できない型にする（`.claude/rules/coding-rust.md`「公開 API・型設計」）。
/// `Display` は `diagnose` の走査順（本ファイル冒頭の doc）に対応する
/// `"$"`・`"schema"`・`"options[2]"`・`"options[2].id"`・`"io.input"` の
/// ようなドット記法のパス文字列を返す。値そのものは保持しない
/// （security.md「秘密情報の混入防止」: 利用者データをエラー文へ漏らさない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FieldPath {
    /// JSON ドキュメントのルート。
    Root,
    Schema,
    Name,
    Version,
    JudgmentType,
    Options,
    /// `options` 配列の `index` 番目の要素全体。
    OptionEntry {
        index: usize,
    },
    /// `options[index]` の中の特定フィールド。
    OptionField {
        index: usize,
        field: ChoiceField,
    },
    Io,
    /// `io.input`。
    IoInput,
}

impl std::fmt::Display for FieldPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldPath::Root => write!(f, "$"),
            FieldPath::Schema => write!(f, "schema"),
            FieldPath::Name => write!(f, "name"),
            FieldPath::Version => write!(f, "version"),
            FieldPath::JudgmentType => write!(f, "judgment_type"),
            FieldPath::Options => write!(f, "options"),
            FieldPath::OptionEntry { index } => write!(f, "options[{index}]"),
            FieldPath::OptionField { index, field } => write!(f, "options[{index}].{field}"),
            FieldPath::Io => write!(f, "io"),
            FieldPath::IoInput => write!(f, "io.input"),
        }
    }
}

/// `options` の要素が持つフィールドの種別（[`FieldPath::OptionField`] で使う）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChoiceField {
    Id,
    DisplayName,
    Description,
}

impl std::fmt::Display for ChoiceField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChoiceField::Id => write!(f, "id"),
            ChoiceField::DisplayName => write!(f, "display_name"),
            ChoiceField::Description => write!(f, "description"),
        }
    }
}

/// [`DefinitionError::TypeMismatch`] が期待していた型（診断側の許可規則の
/// 語彙。実測された型は [`JsonType`] で表す）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExpectedType {
    Object,
    Array,
    String,
    /// `version` フィールドの許容範囲（0 以上 `u32::MAX` 以下の整数）。
    UnsignedInt32,
}

impl std::fmt::Display for ExpectedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpectedType::Object => write!(f, "object"),
            ExpectedType::Array => write!(f, "array"),
            ExpectedType::String => write!(f, "string"),
            ExpectedType::UnsignedInt32 => write!(f, "unsigned integer (u32)"),
        }
    }
}

/// JSON の実行時の型（PR #191 の `null`/`bool`/`number`/`string`/`array`/
/// `object` と同じ語彙。データ契約層〔`crates/data`〕には依存せず、
/// 語彙だけを合わせる）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JsonType {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
}

impl std::fmt::Display for JsonType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonType::Null => write!(f, "null"),
            JsonType::Bool => write!(f, "bool"),
            JsonType::Number => write!(f, "number"),
            JsonType::String => write!(f, "string"),
            JsonType::Array => write!(f, "array"),
            JsonType::Object => write!(f, "object"),
        }
    }
}

/// 定義ファイルの読み込み・パース時のエラー。
///
/// 本 TASK（TASK-15.3-1）は正常系のみを対象とするが、PR #187 のセキュリティ
/// レビュー（P0/P1）指摘に基づき、ガード層としての完全性検証
/// （`UnsupportedSchema`・選択肢の整合性）とサイズ上限（`TooLarge`。
/// `Definition::parse` からも到達する。同レビュー指摘）を先行実装する。
/// これ以外の必須項目欠落等の詳細な検証バリアントは TASK-15.3-2 が追加し、
/// `missing_labels` 相当の判定は TASK-15.4 が追加する（`#[non_exhaustive]`）。
#[non_exhaustive]
pub enum DefinitionError {
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    /// サイズ上限超過（REQ-39）。`Definition::load` 経由ではファイルパスを
    /// 持つが、`Definition::parse` に文字列を直接渡す呼び出し（パスを
    /// 持たない公開 API 経由）でも同じ上限を適用するため `path` は `None` を
    /// 取りうる（PR #187 レビュー指摘: 公開パース API にも入力サイズ上限を
    /// 適用する）。
    TooLarge {
        path: Option<PathBuf>,
        size: u64,
        limit: u64,
    },
    /// JSON 構文エラー（`serde_json::Error::classify()` が `Syntax`/`Eof`/`Io`）、
    /// または `diagnose`（TASK-15.3-2）が分類できなかったデータエラー
    /// （例: 重複フィールド。`serde_json::Value` は重複キーを後勝ちで黙って
    /// 受理するため `diagnose` の対象外とし、fail-closed でここに落とす。
    /// `Definition::parse` の doc を参照）。
    Parse { source: serde_json::Error },
    /// `schema` フィールドが `SCHEMA_ID` と一致しない（未対応の形式。
    /// security.md「ガード層: 完全性と版」）。
    UnsupportedSchema { schema: String },
    /// `options` が空で、固定選択肢からの判定が成立しない（REQ-15）。
    EmptyOptions,
    /// `options` 内に空文字列の `id` を持つ選択肢がある（ラベル照合が成立しない）。
    EmptyOptionId,
    /// `options` 内で `id` が重複している（ラベル照合が一意に定まらない）。
    DuplicateOptionId { id: String },
    /// パス先が通常ファイルではない（FIFO・ソケット・キャラクタデバイス等）。
    /// これらを許すと `File::open`/`read_to_end` が書き手を待って無期限に
    /// 停止しうる（security.md「ガード層: 資源の上限」。PR #187 レビュー指摘）。
    NotRegularFile { path: PathBuf },
    /// 必須フィールドが欠落している（TASK-15.3-2）。`options` の欠落も
    /// ここに含む。TASK-15.4 が `options` 欠落と「`options` はあるが
    /// ラベル定義が非同梱」を区別する `MissingLabels` を追加する接続点は
    /// `diagnose` モジュール内の該当箇所に注記する。
    MissingField { field: FieldPath },
    /// フィールドの型が期待と異なる（TASK-15.3-2）。値そのものは保持せず、
    /// 期待した型と実測した型のみを保持する（security.md「秘密情報の混入
    /// 防止」: 利用者データをエラー文へ漏らさない）。
    TypeMismatch {
        field: FieldPath,
        expected: ExpectedType,
        actual: JsonType,
    },
    /// 定義ファイルのスキーマが許可しない未知のキーを含む（TASK-15.3-2）。
    /// `name`（利用者が任意に指定できるキー名）は秘密情報やデータ本文を
    /// 含みうるため `Display` にも `Debug`（`{:?}` によるログ出力を含む）
    /// にも一切出さず、親フィールドのパスと `reason_code()`（`unknown_field`）
    /// のみで表す（security.md「秘密情報の混入防止」。PR #197 レビュー指摘。
    /// `DefinitionError` は `#[derive(Debug)]` を使わず本ファイル末尾で
    /// `Debug` を手書き実装し、`name` を redacted 表示に固定している）。
    /// `name` はプログラム的な照合（テスト・診断）のためにのみ保持する。
    UnknownField { parent: FieldPath, name: String },
    /// フィールドは正しい型だが、許可された値の集合に含まれない
    /// （例: `judgment_type: "multi_select"`。TASK-15.3-2）。値そのものは
    /// 保持せず、許可値の一覧は `Display` に固定文字列で出す。
    UnsupportedValue { field: FieldPath },
    /// `name`（カタログ上の識別子。PoC-19 の `catalog_id` 相当）が空文字列
    /// （TASK-15.3-2）。空文字列は実質的な欠落として扱う（REQ-15）。
    EmptyName,
}

impl std::fmt::Display for DefinitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefinitionError::Read { path, source } => {
                write!(f, "failed to read definition file {path:?}: {source}")
            }
            DefinitionError::TooLarge { path, size, limit } => match path {
                Some(path) => write!(
                    f,
                    "definition file {path:?} is too large: {size} bytes (limit: {limit} bytes)"
                ),
                None => write!(
                    f,
                    "definition input is too large: {size} bytes (limit: {limit} bytes)"
                ),
            },
            DefinitionError::Parse { source } => {
                write!(f, "failed to parse definition file: {source}")
            }
            DefinitionError::UnsupportedSchema { schema } => {
                write!(
                    f,
                    "unsupported definition schema {schema:?} (expected {SCHEMA_ID:?})"
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
            DefinitionError::NotRegularFile { path } => {
                write!(f, "definition path {path:?} is not a regular file")
            }
            DefinitionError::MissingField { field } => {
                write!(f, "definition field {field} is missing")
            }
            DefinitionError::TypeMismatch {
                field,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "definition field {field} has wrong type: expected {expected}, found {actual}"
                )
            }
            DefinitionError::UnknownField { parent, .. } => {
                write!(f, "definition field {parent} has an unknown key")
            }
            DefinitionError::UnsupportedValue { field } => {
                write!(f, "definition field {field} has an unsupported value")
            }
            DefinitionError::EmptyName => {
                write!(f, "definition name must not be empty")
            }
        }
    }
}

/// `Debug`（`{:?}`）を手書きする。`#[derive(Debug)]` は `UnknownField.name`
/// （利用者が任意に指定できる未知キー名。秘密情報やデータ本文を含みうる）を
/// そのまま出力してしまい、`Display` 側でキー名を伏せていてもログ等で
/// 露出しうる（security.md「秘密情報の混入防止」。PR #197 レビュー指摘）。
/// 他のバリアントは `derive(Debug)` と同等の出力形式にし、`UnknownField` の
/// `name` だけ固定の redacted 表示に置き換える。
impl std::fmt::Debug for DefinitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefinitionError::Read { path, source } => f
                .debug_struct("Read")
                .field("path", path)
                .field("source", source)
                .finish(),
            DefinitionError::TooLarge { path, size, limit } => f
                .debug_struct("TooLarge")
                .field("path", path)
                .field("size", size)
                .field("limit", limit)
                .finish(),
            DefinitionError::Parse { source } => {
                f.debug_struct("Parse").field("source", source).finish()
            }
            DefinitionError::UnsupportedSchema { schema } => f
                .debug_struct("UnsupportedSchema")
                .field("schema", schema)
                .finish(),
            DefinitionError::EmptyOptions => f.write_str("EmptyOptions"),
            DefinitionError::EmptyOptionId => f.write_str("EmptyOptionId"),
            DefinitionError::DuplicateOptionId { id } => {
                f.debug_struct("DuplicateOptionId").field("id", id).finish()
            }
            DefinitionError::NotRegularFile { path } => f
                .debug_struct("NotRegularFile")
                .field("path", path)
                .finish(),
            DefinitionError::MissingField { field } => f
                .debug_struct("MissingField")
                .field("field", field)
                .finish(),
            DefinitionError::TypeMismatch {
                field,
                expected,
                actual,
            } => f
                .debug_struct("TypeMismatch")
                .field("field", field)
                .field("expected", expected)
                .field("actual", actual)
                .finish(),
            DefinitionError::UnknownField { parent, name: _ } => f
                .debug_struct("UnknownField")
                .field("parent", parent)
                // `name` は秘密情報・データ本文を含みうるため常に redacted 表示に
                // 固定する（実際の値は保持しているが Debug には出さない）。
                .field("name", &"<redacted>")
                .finish(),
            DefinitionError::UnsupportedValue { field } => f
                .debug_struct("UnsupportedValue")
                .field("field", field)
                .finish(),
            DefinitionError::EmptyName => f.write_str("EmptyName"),
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
            DefinitionError::EmptyOptions => None,
            DefinitionError::EmptyOptionId => None,
            DefinitionError::DuplicateOptionId { .. } => None,
            DefinitionError::NotRegularFile { .. } => None,
            DefinitionError::MissingField { .. } => None,
            DefinitionError::TypeMismatch { .. } => None,
            DefinitionError::UnknownField { .. } => None,
            DefinitionError::UnsupportedValue { .. } => None,
            DefinitionError::EmptyName => None,
        }
    }
}

impl DefinitionError {
    /// 機械可読な snake_case のエラーコード（CLI・MCP の JSON 出力契約
    /// （REQ-21・REQ-33）へ配線する際の接続点。本 TASK では配線しない）。
    ///
    /// `missing_field`/`type_mismatch` は PR #191（データ契約層）の語彙と
    /// 揃える。TASK-15.4 は新しいバリアント（例: `MissingLabels`）を足して
    /// `missing_labels` を返すだけで済む。ワイルドカードなしの網羅 `match`
    /// にすることで、バリアント追加時のコード漏れをコンパイルエラーで
    /// 検出する。
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            DefinitionError::Read { .. } => "read_error",
            DefinitionError::TooLarge { .. } => "too_large",
            DefinitionError::Parse { .. } => "parse_error",
            DefinitionError::UnsupportedSchema { .. } => "unsupported_schema",
            DefinitionError::EmptyOptions => "empty_options",
            DefinitionError::EmptyOptionId => "empty_option_id",
            DefinitionError::DuplicateOptionId { .. } => "duplicate_option_id",
            DefinitionError::NotRegularFile { .. } => "not_regular_file",
            DefinitionError::MissingField { .. } => "missing_field",
            DefinitionError::TypeMismatch { .. } => "type_mismatch",
            DefinitionError::UnknownField { .. } => "unknown_field",
            DefinitionError::UnsupportedValue { .. } => "unsupported_value",
            DefinitionError::EmptyName => "empty_name",
        }
    }

    /// REQ-21 の 7 種の終了コードへの対応づけ（CLI への配線〔TASK-33.x〕は
    /// 本 TASK の対象外。`crate::exitcode::ExitCode` を参照）。
    ///
    /// `TooLarge` のみ資源上限超過（`LimitExceeded`）とし、それ以外
    /// （`Read`・`NotRegularFile` を含む）はすべて外部入力の不正
    /// （`InvalidInput`）として扱う（PoC-16 でも読めない定義は
    /// invalid_input だった）。非対応の `judgment_type` を `OutOfScope`
    /// ではなく `InvalidInput` にする点は PoC-16 と異なる判断で、定義ファイル
    /// 自体が不正である（判定対象の範囲外の入力とは別の事象である）ことを
    /// 理由とする。
    #[must_use]
    pub const fn exit_code(&self) -> crate::exitcode::ExitCode {
        match self {
            DefinitionError::TooLarge { .. } => crate::exitcode::ExitCode::LimitExceeded,
            DefinitionError::Read { .. }
            | DefinitionError::Parse { .. }
            | DefinitionError::UnsupportedSchema { .. }
            | DefinitionError::EmptyOptions
            | DefinitionError::EmptyOptionId
            | DefinitionError::DuplicateOptionId { .. }
            | DefinitionError::NotRegularFile { .. }
            | DefinitionError::MissingField { .. }
            | DefinitionError::TypeMismatch { .. }
            | DefinitionError::UnknownField { .. }
            | DefinitionError::UnsupportedValue { .. }
            | DefinitionError::EmptyName => crate::exitcode::ExitCode::InvalidInput,
        }
    }
}

impl Definition {
    /// JSON 文字列から定義ファイルをパースし、ガード層としての完全性検証
    /// （`schema` の照合・選択肢の整合性）を行う（REQ-15）。本関数は公開 API で
    /// あり、`Definition::load` が担うファイル経由のサイズ検証（REQ-39）を
    /// 経由しない呼び出し（文字列を直接渡す呼び出し）もありうるため、
    /// `serde_json::from_str` へ渡す前に入力バイト数を `MAX_DEFINITION_FILE_BYTES`
    /// と照合する（PR #187 レビュー指摘: 公開パース API にも入力サイズ上限を
    /// 適用する。無制限のメモリ消費を防ぐ）。
    /// 必須項目欠落等のこれ以外の詳細検証は TASK-15.3-2 で追加する。
    pub fn parse(text: &str) -> Result<Self, DefinitionError> {
        let size = text.len() as u64;
        if size > MAX_DEFINITION_FILE_BYTES {
            return Err(DefinitionError::TooLarge {
                path: None,
                size,
                limit: MAX_DEFINITION_FILE_BYTES,
            });
        }

        // `Definition` は `Deserialize` を実装しない（上記 `RawDefinition` の
        // ドキュメンテーションコメントを参照）。ここで未検証の中間型へ
        // デシリアライズしたうえで、以下の検証をすべて経てから初めて
        // `Definition` を構築する。
        //
        // 型付きデシリアライズが失敗した場合（TASK-15.3-2）、`classify()` が
        // `Syntax`/`Eof`/`Io` なら構文エラーとしてそのまま `Parse` を返す。
        // `Data`（必須項目欠落・型不整合等）の場合のみ `diagnose` で
        // `serde_json::Value` として再走査し、最初に見つかった構造エラーを
        // 型付きバリアントで返す。`diagnose` が何も分類できなかった場合
        // （重複フィールド等、`Value` 側では検出できない不整合）は元の
        // `Parse { source }` を返す（fail-closed。`diagnose.rs` の doc を参照）。
        let raw: RawDefinition = match serde_json::from_str(text) {
            Ok(raw) => raw,
            Err(source) => {
                if source.classify() == serde_json::error::Category::Data
                    && let Some(diagnosed) = diagnose::diagnose(text)
                {
                    return Err(diagnosed);
                }
                return Err(DefinitionError::Parse { source });
            }
        };

        // 完全性と版（security.md「ガード層: 完全性と版」）: 異なるスキーマ形式の
        // 定義をそのまま正常な `Definition` として後続処理へ渡さない。`version`
        // フィールドはカタログ上のドキュメントバージョン（`name` と並ぶメタ情報）
        // であり、形式・版の識別を担う第二のスキーマ固定値としては扱わない
        // （PR #187 レビュー指摘。有効な定義の後続版を誤って拒否しないため）。
        if raw.schema != SCHEMA_ID {
            return Err(DefinitionError::UnsupportedSchema { schema: raw.schema });
        }

        // `name` はカタログ上の識別子（PoC-19 の `catalog_id` 相当）で、
        // 空文字列は実質的な欠落として扱う（REQ-15。TASK-15.3-2）。
        if raw.name.is_empty() {
            return Err(DefinitionError::EmptyName);
        }

        // 選択肢の整合性: 固定選択肢からの選択・ラベル照合が成立する状態
        // （空でない・id が空でない・id が重複しない）であることを検証する
        // （REQ-15 の入出力契約）。
        if raw.options.is_empty() {
            return Err(DefinitionError::EmptyOptions);
        }
        let mut seen_ids = std::collections::HashSet::with_capacity(raw.options.len());
        for choice in &raw.options {
            if choice.id.is_empty() {
                return Err(DefinitionError::EmptyOptionId);
            }
            if !seen_ids.insert(choice.id.as_str()) {
                return Err(DefinitionError::DuplicateOptionId {
                    id: choice.id.clone(),
                });
            }
        }

        Ok(Definition {
            schema: raw.schema,
            name: raw.name,
            version: raw.version,
            judgment_type: raw.judgment_type,
            options: raw.options,
            io: raw.io,
        })
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
    /// FIFO・ソケット・キャラクタデバイス等を指すパスを渡されると、通常の
    /// `File::open` は書き手が現れるまで無期限に停止しうる（security.md
    /// 「ガード層: 資源の上限」。PR #187 レビュー指摘）。Linux・macOS では
    /// `open(2)` に `O_NONBLOCK` を付けて開くことでこの停止を避け（POSIX:
    /// `O_NONBLOCK` は通常ファイルの読み込み完了には影響しない）、開いた後に
    /// 種別を確認して通常ファイル以外を拒否する。両 OS 以外（Windows 等。
    /// 検証環境は Mac のみで Windows は M10 時点で対象外）では通常の
    /// `File::open` にフォールバックする（coding-rust.md「クロスプラット
    /// フォーム」）。
    ///
    /// 経路の閉じ込め（`../` 等の拒否）は操作アダプターのガード層（TASK-39.x）の
    /// 責務であり、ここでは行わない。
    pub fn load(path: &Path) -> Result<Self, DefinitionError> {
        use std::io::Read as _;

        let mut file = Self::open_for_read(path)?;
        let metadata = file.metadata().map_err(|source| DefinitionError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        if !metadata.file_type().is_file() {
            return Err(DefinitionError::NotRegularFile {
                path: path.to_path_buf(),
            });
        }
        let reported_size = metadata.len();
        if reported_size > MAX_DEFINITION_FILE_BYTES {
            return Err(DefinitionError::TooLarge {
                path: Some(path.to_path_buf()),
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
                path: Some(path.to_path_buf()),
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

    /// `load` の内部専用: FIFO 等での無期限停止を避けるため `O_NONBLOCK` 付きで
    /// 開く（Linux・macOS）。`file_type().is_file()` の検査は `load` 側で行う。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn open_for_read(path: &Path) -> Result<std::fs::File, DefinitionError> {
        use std::os::unix::fs::OpenOptionsExt as _;

        // `O_NONBLOCK` はカーネル ABI の安定値（Linux は全対応アーキテクチャで
        // 8進 0o4000、macOS（BSD 系）は 0x0004）。`libc` 等の新規依存を追加
        // せず（dependency-policy.md）標準ライブラリの `custom_flags` のみで
        // 実現するため、対応 OS を限定してハードコードする。
        #[cfg(target_os = "linux")]
        const O_NONBLOCK: i32 = 0o4000;
        #[cfg(target_os = "macos")]
        const O_NONBLOCK: i32 = 0x0004;

        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(O_NONBLOCK)
            .open(path)
            .map_err(|source| DefinitionError::Read {
                path: path.to_path_buf(),
                source,
            })
    }

    /// 検証済みの `schema` フィールドを返す（`SCHEMA_ID` と一致することが
    /// `parse` により保証済み）。フィールドが非公開のため、値の参照は
    /// このアクセサ経由に限る（上記 `Definition` のドキュメンテーション
    /// コメントを参照）。
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// カタログ上のドキュメントバージョン（`name` と並ぶメタ情報。
    /// `schema`/`SCHEMA_ID` とは独立で、形式・版の識別には使わない）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// カタログ上のドキュメントバージョン番号。
    pub fn version(&self) -> u32 {
        self.version
    }

    /// 判定型（REQ-15 時点では `SingleSelect` のみ）。
    pub fn judgment_type(&self) -> JudgmentType {
        self.judgment_type
    }

    /// 検証済みの選択肢一覧（空でない・`id` が非空かつ重複しないことが
    /// `parse` により保証済み。宣言順を保持）。
    pub fn options(&self) -> &[Choice] {
        &self.options
    }

    /// 入出力のデータ構造。
    pub fn io(&self) -> &IoSchema {
        &self.io
    }

    /// 定義の同一性（選択肢 ID の集合＋`judgment_type`。表示名・説明・`name`・
    /// `version`・選択肢の宣言順は含めない。TASK-15.5・境界値。REQ-15）。
    /// 失敗しない（既に検証済みのフィールドから組み立てるだけのため）。
    #[must_use]
    pub fn identity(&self) -> crate::canonical::DefinitionIdentity {
        crate::canonical::DefinitionIdentity::from_definition(self)
    }

    /// 定義全体（表示名・説明を含む）を正準化した JSON 文字列。記録・デバッグ
    /// 用（TASK-15.5）。正準化の規則は `canonical` モジュールに集約している。
    pub fn canonical_json(&self) -> Result<String, crate::canonical::CanonicalError> {
        crate::canonical::canonical_json(self)
    }

    /// 定義全体を正準化した JSON の sha256（TASK-15.5）。TASK-20.3（#93）の
    /// 「新旧定義の正準化ハッシュが完全一致したら変更なし」判定に使う。
    pub fn canonical_hash(
        &self,
    ) -> Result<crate::canonical::DefinitionHash, crate::canonical::CanonicalError> {
        let json = self.canonical_json()?;
        Ok(crate::canonical::DefinitionHash::from_json_bytes(
            json.as_bytes(),
        ))
    }

    /// `load` の内部専用: Linux・macOS 以外（Windows 等）向けのフォールバック。
    /// `O_NONBLOCK` 相当の対策は持たないが、Windows は M10 時点で対象外
    /// （coding-rust.md「クロスプラットフォーム」）であり、`file_type().is_file()`
    /// による通常ファイル以外の拒否は `load` 側で引き続き行う。
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn open_for_read(path: &Path) -> Result<std::fs::File, DefinitionError> {
        std::fs::File::open(path).map_err(|source| DefinitionError::Read {
            path: path.to_path_buf(),
            source,
        })
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

    /// PR #187 レビュー指摘: `version` はカタログ上のドキュメントバージョン
    /// （`name` と並ぶメタ情報）であり、`schema`/`SCHEMA_ID` とは独立した
    /// 第二のスキーマ固定値として扱わない。`schema` が一致する限り、
    /// `DEFINITION_SCHEMA_VERSION`（1）と異なる `version`（将来のドキュメント
    /// 版）でも定義として受理する。
    #[test]
    fn req15_accepts_definition_with_catalog_version_other_than_schema_version() {
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
        let def =
            Definition::parse(json).expect("version はカタログのメタ情報であり拒否対象ではない");
        assert_eq!(def.version, 2);
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

    /// PR #187 レビュー指摘（P0）: 公開パース API `Definition::parse` は
    /// `Definition::load` のファイル経由サイズ検証を経由しない呼び出し
    /// （文字列を直接渡す呼び出し）でも、`serde_json::from_str` へ渡す前に
    /// `MAX_DEFINITION_FILE_BYTES` 超の入力を拒否することを確認する（REQ-39）。
    /// `path` を持たない呼び出しのため `TooLarge.path` は `None` になる。
    #[test]
    fn req39_parse_rejects_oversized_input_without_path() {
        let oversized_json = "a".repeat(usize::try_from(MAX_DEFINITION_FILE_BYTES).unwrap() + 1);
        let err = Definition::parse(&oversized_json).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::TooLarge { path: None, limit, .. } if limit == MAX_DEFINITION_FILE_BYTES
        ));
    }

    /// REQ-39: サイズ確認と読み込みを同一ファイルハンドルに対して行い、
    /// 上限（`MAX_DEFINITION_FILE_BYTES`）超のファイルを内容の全量読み込み
    /// なしに拒否することを確認する（TOCTOU 対策の回帰）。`load` 経由では
    /// ファイルパスを保持するため `TooLarge.path` は `Some` になる。
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
        assert!(matches!(
            err,
            DefinitionError::TooLarge { path: Some(_), limit, .. } if limit == MAX_DEFINITION_FILE_BYTES
        ));

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

    // ------------------------------------------------------------------
    // TASK-15.3-2: 必須項目欠落・型不整合・未知キー・enum 外の値の検証。
    // ------------------------------------------------------------------

    /// 正常な定義（`TWO_OPTIONS_JSON` 相当）をトップレベルの `Value` として
    /// 返す。各テストはこれを変形して欠落・型不整合を作る。
    fn valid_definition_value() -> serde_json::Value {
        serde_json::from_str(TWO_OPTIONS_JSON).expect("固定 fixture は valid JSON のはず")
    }

    fn parse_value(value: &serde_json::Value) -> Result<Definition, DefinitionError> {
        Definition::parse(&value.to_string())
    }

    #[test]
    fn req15_rejects_definition_missing_each_required_top_level_field() {
        let cases: [(&str, FieldPath); 6] = [
            ("schema", FieldPath::Schema),
            ("name", FieldPath::Name),
            ("version", FieldPath::Version),
            ("judgment_type", FieldPath::JudgmentType),
            ("options", FieldPath::Options),
            ("io", FieldPath::Io),
        ];
        for (key, expected_field) in cases {
            let mut value = valid_definition_value();
            value
                .as_object_mut()
                .expect("object のはず")
                .remove(key)
                .expect("既存キーのはず");
            let err = parse_value(&value).unwrap_err();
            assert!(
                matches!(&err, DefinitionError::MissingField { field } if *field == expected_field),
                "key {key} を削除した場合に MissingField({expected_field}) を期待したが {err:?} だった"
            );
            assert_eq!(err.reason_code(), "missing_field");
        }
    }

    #[test]
    fn req15_rejects_option_entry_missing_each_required_field() {
        let cases: [(&str, ChoiceField); 3] = [
            ("id", ChoiceField::Id),
            ("display_name", ChoiceField::DisplayName),
            ("description", ChoiceField::Description),
        ];
        for (key, expected_field) in cases {
            let mut value = valid_definition_value();
            let options = value
                .get_mut("options")
                .and_then(serde_json::Value::as_array_mut)
                .expect("options は配列のはず");
            let second = options
                .get_mut(1)
                .and_then(serde_json::Value::as_object_mut)
                .expect("options[1] は object のはず");
            second.remove(key).expect("既存キーのはず");

            let err = parse_value(&value).unwrap_err();
            match err {
                DefinitionError::MissingField {
                    field: FieldPath::OptionField { index, field },
                } => {
                    assert_eq!(index, 1);
                    assert_eq!(field, expected_field);
                    assert_eq!(
                        FieldPath::OptionField { index, field }.to_string(),
                        format!("options[1].{key}")
                    );
                }
                other => panic!("MissingField(OptionField) を期待したが {other:?} だった"),
            }
        }
    }

    #[test]
    fn req15_rejects_io_missing_input() {
        let mut value = valid_definition_value();
        value["io"] = serde_json::json!({});
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::MissingField {
                field: FieldPath::IoInput
            }
        ));
        assert_eq!(FieldPath::IoInput.to_string(), "io.input");
    }

    /// TASK-15.4 の前提: `options` キーが無い場合はフォールバック（既定の
    /// ラベル集合への補完）にならず、`MissingField { Options }` になる。
    #[test]
    fn req15_missing_options_does_not_fall_back_to_default_labels() {
        let mut value = valid_definition_value();
        value
            .as_object_mut()
            .expect("object のはず")
            .remove("options")
            .expect("既存キーのはず");
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::MissingField {
                field: FieldPath::Options
            }
        ));
    }

    #[test]
    fn req15_rejects_version_with_wrong_type() {
        let mut value = valid_definition_value();
        value["version"] = serde_json::json!("1");
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::TypeMismatch {
                field: FieldPath::Version,
                expected: ExpectedType::UnsignedInt32,
                actual: JsonType::String,
            }
        ));
        assert_eq!(err.reason_code(), "type_mismatch");
    }

    #[test]
    fn req15_rejects_version_out_of_u32_range() {
        for version in [
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!(4_294_967_296i64),
        ] {
            let mut value = valid_definition_value();
            value["version"] = version.clone();
            let err = parse_value(&value).unwrap_err();
            assert!(
                matches!(
                    err,
                    DefinitionError::TypeMismatch {
                        field: FieldPath::Version,
                        expected: ExpectedType::UnsignedInt32,
                        actual: JsonType::Number,
                    }
                ),
                "version={version} で TypeMismatch(Version, UnsignedInt32, Number) を期待した"
            );
        }
    }

    #[test]
    fn req15_rejects_options_with_wrong_type() {
        let mut value = valid_definition_value();
        value["options"] = serde_json::json!({});
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::TypeMismatch {
                field: FieldPath::Options,
                expected: ExpectedType::Array,
                actual: JsonType::Object,
            }
        ));
    }

    #[test]
    fn req15_rejects_option_entry_with_wrong_type() {
        let mut value = valid_definition_value();
        value["options"] = serde_json::json!(["yes"]);
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::TypeMismatch {
                field: FieldPath::OptionEntry { index: 0 },
                expected: ExpectedType::Object,
                actual: JsonType::String,
            }
        ));
    }

    #[test]
    fn req15_rejects_null_name() {
        let mut value = valid_definition_value();
        value["name"] = serde_json::Value::Null;
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::TypeMismatch {
                field: FieldPath::Name,
                expected: ExpectedType::String,
                actual: JsonType::Null,
            }
        ));
    }

    #[test]
    fn req15_rejects_non_object_root() {
        let err = Definition::parse("[]").unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::TypeMismatch {
                field: FieldPath::Root,
                expected: ExpectedType::Object,
                actual: JsonType::Array,
            }
        ));
        assert_eq!(FieldPath::Root.to_string(), "$");
    }

    #[test]
    fn req15_rejects_unsupported_judgment_type_value() {
        let mut value = valid_definition_value();
        value["judgment_type"] = serde_json::json!("multi_select");
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::UnsupportedValue {
                field: FieldPath::JudgmentType
            }
        ));
        assert_eq!(err.reason_code(), "unsupported_value");
    }

    #[test]
    fn req15_rejects_unsupported_io_input_value() {
        let mut value = valid_definition_value();
        value["io"]["input"] = serde_json::json!("text");
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::UnsupportedValue {
                field: FieldPath::IoInput
            }
        ));
    }

    #[test]
    fn req15_rejects_unknown_top_level_field() {
        let mut value = valid_definition_value();
        value["extra"] = serde_json::json!(1);
        let err = parse_value(&value).unwrap_err();
        match err {
            DefinitionError::UnknownField { parent, name } => {
                assert_eq!(parent, FieldPath::Root);
                assert_eq!(name, "extra");
            }
            other => panic!("UnknownField を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req15_rejects_unknown_option_field() {
        let mut value = valid_definition_value();
        value["options"][0]["color"] = serde_json::json!("red");
        let err = parse_value(&value).unwrap_err();
        match err {
            DefinitionError::UnknownField { parent, name } => {
                assert_eq!(parent, FieldPath::OptionEntry { index: 0 });
                assert_eq!(name, "color");
            }
            other => panic!("UnknownField を期待したが {other:?} だった"),
        }
    }

    /// 走査順（`diagnose::diagnose` 関数内のステップ 2〔schema〕を
    /// ステップ 5〔io〕より先に評価する設計）: `schema` が別形式で `io` も
    /// 欠落している場合、誤誘導の `MissingField(Io)` ではなく
    /// `UnsupportedSchema` を返す。
    #[test]
    fn req15_prioritizes_unsupported_schema_over_missing_field() {
        let mut value = valid_definition_value();
        value["schema"] = serde_json::json!("other-schema/v1");
        value
            .as_object_mut()
            .expect("object のはず")
            .remove("io")
            .expect("既存キーのはず");
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(
            err,
            DefinitionError::UnsupportedSchema { schema } if schema == "other-schema/v1"
        ));
    }

    /// 回帰防止: `serde_json::Value` は重複キーを後勝ちで黙って受理するため
    /// `diagnose` の対象にせず、型付きデシリアライズ側の `duplicate field`
    /// 拒否をそのまま活かす（`Definition::parse` の doc・`diagnose.rs` の doc）。
    #[test]
    fn req15_rejects_duplicate_top_level_key_as_parse_error() {
        let json = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "name": "sample_topic_2",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "Yes", "description": "肯定" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let err = Definition::parse(json).unwrap_err();
        assert!(matches!(err, DefinitionError::Parse { .. }));
    }

    #[test]
    fn req15_rejects_empty_name() {
        let mut value = valid_definition_value();
        value["name"] = serde_json::json!("");
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(err, DefinitionError::EmptyName));
        assert_eq!(err.reason_code(), "empty_name");
    }

    /// security.md「秘密情報の混入防止」: `Display` は値を漏らさない。
    #[test]
    fn req15_display_does_not_leak_field_values() {
        let mut value = valid_definition_value();
        value["description_typo_marker"] = serde_json::json!("s3cr3t-value");
        let err = parse_value(&value).unwrap_err();
        assert!(!err.to_string().contains("s3cr3t-value"));

        let mut type_mismatch_value = valid_definition_value();
        type_mismatch_value["options"][0]["description"] = serde_json::json!(12345);
        let err = parse_value(&type_mismatch_value).unwrap_err();
        assert!(!err.to_string().contains("12345"));
    }

    /// PR #197 レビュー指摘（P0）: 未知キーの「名前」自体も利用者が任意に
    /// 指定できる入力であり、秘密情報やデータ本文が入りうるため、
    /// `Display` に一切出さない（security.md「秘密情報の混入防止」）。
    #[test]
    fn req15_display_does_not_leak_unknown_field_name() {
        let mut value = valid_definition_value();
        value["s3cr3t-api-key-should-not-leak"] = serde_json::json!(1);
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(err, DefinitionError::UnknownField { .. }));
        assert!(!err.to_string().contains("s3cr3t-api-key-should-not-leak"));
        assert_eq!(err.reason_code(), "unknown_field");
    }

    /// PR #197 レビュー指摘（P0・codex/review 再指摘）: `Display` だけでなく
    /// `{:?}`（`Debug`）でも未知キー名を出さないことを確認する。
    /// `DefinitionError` は `#[derive(Debug)]` を使わず手書きの `Debug` 実装
    /// （本ファイルの `impl std::fmt::Debug for DefinitionError`）を持つため、
    /// ログ等が誤って `{:?}` でエラーを出力してもキー名は露出しない。
    #[test]
    fn req15_debug_does_not_leak_unknown_field_name() {
        let mut value = valid_definition_value();
        value["s3cr3t-api-key-should-not-leak"] = serde_json::json!(1);
        let err = parse_value(&value).unwrap_err();
        assert!(matches!(err, DefinitionError::UnknownField { .. }));
        let debug_output = format!("{err:?}");
        assert!(!debug_output.contains("s3cr3t-api-key-should-not-leak"));
        assert!(debug_output.contains("<redacted>"));
    }

    #[test]
    fn req15_reason_code_and_exit_code_cover_all_variants() {
        use crate::exitcode::ExitCode;

        let cases: [(DefinitionError, &str, ExitCode); 13] = [
            (
                DefinitionError::Read {
                    path: PathBuf::from("x"),
                    source: std::io::Error::other("boom"),
                },
                "read_error",
                ExitCode::InvalidInput,
            ),
            (
                DefinitionError::TooLarge {
                    path: None,
                    size: 2,
                    limit: 1,
                },
                "too_large",
                ExitCode::LimitExceeded,
            ),
            (
                DefinitionError::UnsupportedSchema {
                    schema: "x".to_string(),
                },
                "unsupported_schema",
                ExitCode::InvalidInput,
            ),
            (
                DefinitionError::EmptyOptions,
                "empty_options",
                ExitCode::InvalidInput,
            ),
            (
                DefinitionError::EmptyOptionId,
                "empty_option_id",
                ExitCode::InvalidInput,
            ),
            (
                DefinitionError::DuplicateOptionId {
                    id: "x".to_string(),
                },
                "duplicate_option_id",
                ExitCode::InvalidInput,
            ),
            (
                DefinitionError::NotRegularFile {
                    path: PathBuf::from("x"),
                },
                "not_regular_file",
                ExitCode::InvalidInput,
            ),
            (
                DefinitionError::MissingField {
                    field: FieldPath::Name,
                },
                "missing_field",
                ExitCode::InvalidInput,
            ),
            (
                DefinitionError::TypeMismatch {
                    field: FieldPath::Name,
                    expected: ExpectedType::String,
                    actual: JsonType::Null,
                },
                "type_mismatch",
                ExitCode::InvalidInput,
            ),
            (
                DefinitionError::UnknownField {
                    parent: FieldPath::Root,
                    name: "x".to_string(),
                },
                "unknown_field",
                ExitCode::InvalidInput,
            ),
            (
                DefinitionError::UnsupportedValue {
                    field: FieldPath::JudgmentType,
                },
                "unsupported_value",
                ExitCode::InvalidInput,
            ),
            (
                DefinitionError::EmptyName,
                "empty_name",
                ExitCode::InvalidInput,
            ),
            (
                DefinitionError::Parse {
                    source: serde_json::from_str::<()>("{not json").unwrap_err(),
                },
                "parse_error",
                ExitCode::InvalidInput,
            ),
        ];

        for (err, expected_reason, expected_exit) in cases {
            assert_eq!(err.reason_code(), expected_reason);
            assert_eq!(err.exit_code(), expected_exit);
        }
    }
}
