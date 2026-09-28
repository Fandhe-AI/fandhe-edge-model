//! 学習リクエスト JSON（Rust 側 CLI が学習ワーカーへ渡す入力。#178 で子プロセス
//! へ配線予定）の検証済み型（REQ-18・REQ-19・REQ-39）。
//!
//! スキーマは `trainer/src/fandhe_edge_trainer/contract.py`（schema_version 1）
//! と同じ 13 項目を持つ。値の検証は同モジュール・`limits.py`・`guard.py` の
//! 実装を書き起こしたもので、Python 側が返す `code`／終了コードと一致する
//! ことを共有 fixture（`fixtures/train_contract/`）で照合する
//! （`crates/train/tests/train_contract_fixture.rs`）。
//!
//! # 経路の閉じ込めについて
//!
//! `root`・`train_path`・`out_dir` はここでは**文字列としての構文検査のみ**を
//! 行う（空・NUL・絶対/相対の取り違え・`..` 構成要素・`.` のみの拒否）。
//! ファイルシステムへの実際の閉じ込め（存在確認・dir_fd による symlink 対策）
//! は行わない。これは学習ワーカー自身が多層防御として `guard.py::confine` で
//! 検証する設計（`contract.py` のモジュール docstring 参照）であり、Rust 側
//! ガード層（TASK-39.x）が別途 fs 上の検証を担う計画のため、本 crate では
//! 二重実装しない。

use std::collections::HashSet;

use fandhe_edge_core::definition::{Definition, JudgmentType};
use serde::{Deserialize, Serialize};

use crate::error::TrainRequestError;
use crate::limits::{
    MAX_LABEL_BYTES, MAX_LABELS, MAX_MAX_BYTES, MAX_REQUEST_BYTES, MAX_SEED, MAX_TRAIN_RSS_BYTES,
    MAX_TRAIN_WALL_SECONDS, MIN_LABELS, MIN_MAX_BYTES, MIN_SEED, REQUEST_SCHEMA_VERSION,
};

/// 学習の実行デバイス。`contract.py::_ALLOWED_DEVICES`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    Cpu,
    Gpu,
}

/// 検証済みのラベル順（`label_order`）。
///
/// [`LabelOrder::new`] を経由した値だけが構築でき、件数（[`MIN_LABELS`]..=
/// [`MAX_LABELS`]）・各要素の非空性・UTF-8 バイト長上限（[`MAX_LABEL_BYTES`]）・
/// 重複なしを満たすことが型で保証される。宣言順を保持する（評価契約の
/// タイブレーク規則に関わる。`crates/core/src/definition.rs` と同じ理由）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct LabelOrder(Vec<String>);

impl LabelOrder {
    /// `contract.py::_validate_label_order` と同じ規則で検証する。
    pub fn new(labels: Vec<String>) -> Result<Self, TrainRequestError> {
        if !(MIN_LABELS..=MAX_LABELS).contains(&labels.len()) {
            return Err(TrainRequestError::LabelOrderCount {
                actual: labels.len(),
            });
        }
        let mut seen: HashSet<&str> = HashSet::with_capacity(labels.len());
        for (index, label) in labels.iter().enumerate() {
            if label.is_empty() {
                return Err(TrainRequestError::LabelOrderEmptyLabel { index });
            }
            if label.len() > MAX_LABEL_BYTES {
                return Err(TrainRequestError::LabelOrderLabelTooLong { index });
            }
            if !seen.insert(label.as_str()) {
                return Err(TrainRequestError::LabelOrderDuplicateLabel { index });
            }
        }
        Ok(Self(labels))
    }

    /// 宣言順のラベル一覧。
    #[must_use]
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }

    /// 内部の `Vec<String>` を取り出す。
    #[must_use]
    pub fn into_vec(self) -> Vec<String> {
        self.0
    }
}

/// 未検証の学習リクエストの構成要素（呼び出し元が組み立てる入力）。
///
/// [`TrainRequest::new`] へ渡す前段の値で、フィールドはすべて公開だが
/// `Deserialize` は実装しない（検証済みの `TrainRequest` を作る経路は
/// `TrainRequest::new`／`TrainRequest::from_json_slice` の 2 つに限る。
/// `crates/core/src/definition.rs` の `RawDefinition` と同じ設計）。
#[derive(Debug, Clone)]
pub struct TrainRequestParams {
    pub kind: String,
    pub kind_version: u32,
    pub config: serde_json::Map<String, serde_json::Value>,
    pub label_order: Vec<String>,
    pub max_bytes: u32,
    pub seed: u32,
    pub device: Device,
    pub root: String,
    pub train_path: String,
    pub out_dir: String,
    /// 省略時は [`MAX_TRAIN_WALL_SECONDS`]（`contract.py` の既定値と同じ）。
    pub time_limit_seconds: Option<u32>,
    /// 省略時は [`MAX_TRAIN_RSS_BYTES`]（`contract.py` の既定値と同じ）。
    pub rss_limit_bytes: Option<u64>,
}

/// 検証済みの学習リクエスト（schema_version 1）。
///
/// フィールドは非公開にし、読み取り専用アクセサのみを公開する
/// （`definition.rs` と同じ理由: 別 crate が未検証の値を直接構築して
/// ガード層を迂回できないようにする。`.claude/rules/security.md`）。
/// `Deserialize` は実装しない。外部 JSON から作る経路は
/// [`TrainRequest::from_json_slice`] のみ。
#[derive(Debug, Clone, PartialEq)]
pub struct TrainRequest {
    kind: String,
    kind_version: u32,
    config: serde_json::Map<String, serde_json::Value>,
    label_order: LabelOrder,
    max_bytes: u32,
    seed: u32,
    device: Device,
    root: String,
    train_path: String,
    out_dir: String,
    time_limit_seconds: u32,
    rss_limit_bytes: u64,
}

/// `TrainRequest::from_json_slice` の内部専用中間表現（デシリアライズ専用）。
///
/// `#[serde(deny_unknown_fields)]` により未知フィールドを拒否する。数値
/// フィールドは Rust の型で範囲・型を検査するため（例: `bool` は `u32` へ
/// デシリアライズできない。Python 側が `bool` を `int` のサブクラスとして
/// 誤受理する既知の落とし穴〔`contract.py` コメント参照〕を、型自体で回避
/// する）、Python の `type(v) is int` チェックに相当する検査を別途書く必要
/// がない。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTrainRequest {
    schema_version: u32,
    kind: String,
    kind_version: u32,
    #[serde(default)]
    config: serde_json::Map<String, serde_json::Value>,
    label_order: Vec<String>,
    max_bytes: u32,
    seed: u32,
    device: Device,
    root: String,
    train_path: String,
    out_dir: String,
    time_limit_seconds: Option<u32>,
    rss_limit_bytes: Option<u64>,
}

/// [`TrainRequest::to_json_vec`] の内部専用の直列化表現。フィールド順は
/// `contract.py` モジュール docstring のリクエスト例と揃える。
#[derive(Serialize)]
struct WireTrainRequest<'a> {
    schema_version: u32,
    kind: &'a str,
    kind_version: u32,
    config: &'a serde_json::Map<String, serde_json::Value>,
    label_order: &'a LabelOrder,
    max_bytes: u32,
    seed: u32,
    device: Device,
    root: &'a str,
    train_path: &'a str,
    out_dir: &'a str,
    #[serde(skip_serializing_if = "is_default_wall_seconds")]
    time_limit_seconds: u32,
    #[serde(skip_serializing_if = "is_default_rss_bytes")]
    rss_limit_bytes: u64,
}

fn is_default_wall_seconds(value: &u32) -> bool {
    *value == MAX_TRAIN_WALL_SECONDS
}

fn is_default_rss_bytes(value: &u64) -> bool {
    *value == MAX_TRAIN_RSS_BYTES
}

/// `serde_json` のエラーメッセージをそのまま埋め込むと、失敗した値の一部が
/// 引用符付きで含まれることがある（例: 型不一致の実測値）。データ本文では
/// ないが念のため長さを上限で切り詰める（`errors.py::truncate_for_message`
/// と同じ考え方。`.claude/rules/security.md`）。
const MAX_EMBEDDED_ERROR_MESSAGE_CHARS: usize = 200;

fn sanitize_serde_error(err: &serde_json::Error) -> String {
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

/// `root`（絶対パス）の構文検査。`guard.py::resolve_root` の文字列レベルの
/// 規則（存在確認・open は行わない）。
fn check_root_syntax(value: &str) -> Result<(), TrainRequestError> {
    if value.is_empty() || value.contains('\0') || !value.starts_with('/') {
        return Err(TrainRequestError::InvalidPath { field: "root" });
    }
    Ok(())
}

/// `train_path`／`out_dir`（`root` からの相対パス）の構文検査。
/// `guard.py::_check_syntax` と同じ規則: 空でない・NUL を含まない・
/// 絶対パスでない・少なくとも 1 つの通常の構成要素を持つ・`..` 構成要素を
/// 含まない。`PurePosixPath` と同様に `.` 単体の構成要素・連続するスラッシュ
/// は無視して構成要素を数える。
fn check_relative_path_syntax(value: &str, field: &'static str) -> Result<(), TrainRequestError> {
    if value.is_empty() || value.contains('\0') || value.starts_with('/') {
        return Err(TrainRequestError::InvalidPath { field });
    }
    let mut has_component = false;
    for part in value.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return Err(TrainRequestError::InvalidPath { field });
        }
        has_component = true;
    }
    if !has_component {
        return Err(TrainRequestError::InvalidPath { field });
    }
    Ok(())
}

impl TrainRequest {
    /// [`TrainRequestParams`] を検証し、[`TrainRequest`] を組み立てる。
    ///
    /// `contract.py::validate_request` と同じ検証内容（経路の閉じ込め自体を
    /// 除く。本モジュールの doc 参照）を、fail-closed に適用する。
    pub fn new(params: TrainRequestParams) -> Result<Self, TrainRequestError> {
        if params.kind.is_empty() {
            return Err(TrainRequestError::EmptyKind);
        }
        let label_order = LabelOrder::new(params.label_order)?;
        if !(MIN_MAX_BYTES..=MAX_MAX_BYTES).contains(&params.max_bytes) {
            return Err(TrainRequestError::InvalidMaxBytes);
        }
        if !(MIN_SEED..=MAX_SEED).contains(&params.seed) {
            return Err(TrainRequestError::InvalidSeed);
        }
        let time_limit_seconds = params.time_limit_seconds.unwrap_or(MAX_TRAIN_WALL_SECONDS);
        if !(1..=MAX_TRAIN_WALL_SECONDS).contains(&time_limit_seconds) {
            return Err(TrainRequestError::InvalidTimeLimitSeconds);
        }
        let rss_limit_bytes = params.rss_limit_bytes.unwrap_or(MAX_TRAIN_RSS_BYTES);
        if !(1..=MAX_TRAIN_RSS_BYTES).contains(&rss_limit_bytes) {
            return Err(TrainRequestError::InvalidRssLimitBytes);
        }
        check_root_syntax(&params.root)?;
        check_relative_path_syntax(&params.train_path, "train_path")?;
        check_relative_path_syntax(&params.out_dir, "out_dir")?;

        Ok(Self {
            kind: params.kind,
            kind_version: params.kind_version,
            config: params.config,
            label_order,
            max_bytes: params.max_bytes,
            seed: params.seed,
            device: params.device,
            root: params.root,
            train_path: params.train_path,
            out_dir: params.out_dir,
            time_limit_seconds,
            rss_limit_bytes,
        })
    }

    /// 生バイト列（学習リクエスト JSON ファイルの中身）から検証済みの
    /// [`TrainRequest`] を組み立てる。`contract.py::parse_request_bytes` +
    /// `validate_request` に相当する 2 段検証を 1 呼び出しにまとめる。
    ///
    /// 検査順: (1) バイト長（解析前）→ (2) UTF-8 → (3) JSON 構文・型
    /// （`serde_json` の型システムにより、Python の `bool` を `int` の
    /// サブクラスとして誤受理する落とし穴を型レベルで回避する）→
    /// (4) `schema_version` → (5) [`TrainRequest::new`] の各検証。
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, TrainRequestError> {
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err(TrainRequestError::TooLarge {
                size: bytes.len(),
                limit: MAX_REQUEST_BYTES,
            });
        }
        let text = std::str::from_utf8(bytes).map_err(|_| TrainRequestError::NotUtf8)?;
        // 構文エラー（NaN 等の非標準トークンを含む）・型不一致（未知フィール
        // ド・`bool`/文字列を数値フィールドへ渡す等）はいずれも `serde_json`
        // の型付きデシリアライズが一括して検出する。両者を区別する必要は
        // ない（どちらも `code=invalid_request`／`exit=64` に写す。
        // `contract.py::validate_request` も型不一致を区別せず同じ
        // `invalid_request` にしている）。
        let raw: RawTrainRequest =
            serde_json::from_str(text).map_err(|e| TrainRequestError::NotJson {
                message: sanitize_serde_error(&e),
            })?;
        // トップレベルが JSON オブジェクトであることは `RawTrainRequest` への
        // デシリアライズが成功した時点で保証されている（`serde` はオブジェクト
        // 以外からの構造体デシリアライズを許さない）。
        if raw.schema_version != REQUEST_SCHEMA_VERSION {
            return Err(TrainRequestError::UnsupportedSchemaVersion);
        }
        let params = TrainRequestParams {
            kind: raw.kind,
            kind_version: raw.kind_version,
            config: raw.config,
            label_order: raw.label_order,
            max_bytes: raw.max_bytes,
            seed: raw.seed,
            device: raw.device,
            root: raw.root,
            train_path: raw.train_path,
            out_dir: raw.out_dir,
            time_limit_seconds: raw.time_limit_seconds,
            rss_limit_bytes: raw.rss_limit_bytes,
        };
        Self::new(params)
    }

    /// 検証済みの [`TrainRequest`] を、`contract.py` と同じスキーマの JSON
    /// バイト列へ直列化する。直列化後のサイズも [`MAX_REQUEST_BYTES`] を
    /// 超えないことを確認する（`config` は任意の JSON を保持しうるため、
    /// 組み立て段階での再検査が必要。本モジュールの doc 参照）。
    pub fn to_json_vec(&self) -> Result<Vec<u8>, TrainRequestError> {
        let wire = WireTrainRequest {
            schema_version: REQUEST_SCHEMA_VERSION,
            kind: &self.kind,
            kind_version: self.kind_version,
            config: &self.config,
            label_order: &self.label_order,
            max_bytes: self.max_bytes,
            seed: self.seed,
            device: self.device,
            root: &self.root,
            train_path: &self.train_path,
            out_dir: &self.out_dir,
            time_limit_seconds: self.time_limit_seconds,
            rss_limit_bytes: self.rss_limit_bytes,
        };
        let bytes = serde_json::to_vec(&wire).map_err(|_| TrainRequestError::SerializeFailed)?;
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err(TrainRequestError::TooLarge {
                size: bytes.len(),
                limit: MAX_REQUEST_BYTES,
            });
        }
        Ok(bytes)
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn kind_version(&self) -> u32 {
        self.kind_version
    }

    pub fn config(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.config
    }

    pub fn label_order(&self) -> &LabelOrder {
        &self.label_order
    }

    pub fn max_bytes(&self) -> u32 {
        self.max_bytes
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn root(&self) -> &str {
        &self.root
    }

    pub fn train_path(&self) -> &str {
        &self.train_path
    }

    pub fn out_dir(&self) -> &str {
        &self.out_dir
    }

    /// 既定値解決後の壁時計上限（秒）。
    pub fn time_limit_seconds(&self) -> u32 {
        self.time_limit_seconds
    }

    /// 既定値解決後の RSS 上限（バイト）。
    pub fn rss_limit_bytes(&self) -> u64 {
        self.rss_limit_bytes
    }
}

/// 共通コアの定義ファイル（[`Definition`]）から `label_order` を投影する
/// （REQ-15 の選択肢一覧を、学習リクエストのラベル順として使う）。
///
/// `Definition` は「空でない・`id` が非空・重複しない」ことまでしか保証しない
/// （`crates/core/src/definition.rs`）。学習リクエストの上限（件数 2〜1024・
/// 1 ラベルあたり 256 UTF-8 バイト）はここで別途検査する（選択肢が 1 件の
/// 定義・257 バイトの `id` を持つ定義は拒否される）。
///
/// `judgment_type` は網羅的な `match` で `SingleSelect` のみを受け入れる。
/// 将来バリアントが増えた場合はこの関数がコンパイルエラーで気づけるように
/// する（`crates/core/src/definition.rs::JudgmentType` に `#[non_exhaustive]`
/// が付いていないことに依存する）。
pub fn label_order_from_definition(def: &Definition) -> Result<LabelOrder, TrainRequestError> {
    match def.judgment_type() {
        JudgmentType::SingleSelect => {}
    }
    let labels: Vec<String> = def
        .options()
        .iter()
        .map(|choice| choice.id.clone())
        .collect();
    LabelOrder::new(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_params() -> TrainRequestParams {
        TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 42,
            device: Device::Cpu,
            root: "/tmp/fandhe-edge-train-test".to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: None,
            rss_limit_bytes: None,
        }
    }

    /// REQ-39: 既定値解決（`time_limit_seconds`／`rss_limit_bytes` 省略時）が
    /// `contract.py` の既定値（上限そのもの）と一致すること。
    #[test]
    fn req39_defaults_resolve_to_the_upper_limits() {
        let req = TrainRequest::new(valid_params()).expect("valid params must be accepted");
        assert_eq!(req.time_limit_seconds(), MAX_TRAIN_WALL_SECONDS);
        assert_eq!(req.rss_limit_bytes(), MAX_TRAIN_RSS_BYTES);
    }

    /// REQ-39・境界値: `label_order` が 1 件（`MIN_LABELS` 未満）だと拒否する。
    #[test]
    fn req39_rejects_label_order_below_min_labels() {
        let mut params = valid_params();
        params.label_order = vec!["only".to_string()];
        let err = TrainRequest::new(params).unwrap_err();
        assert_eq!(err.reason_code(), "invalid_request");
        assert!(matches!(
            err,
            TrainRequestError::LabelOrderCount { actual: 1 }
        ));
    }

    /// REQ-39・境界値: `label_order` が `MAX_LABELS` を 1 件超えると拒否する。
    #[test]
    fn req39_rejects_label_order_above_max_labels() {
        let mut params = valid_params();
        params.label_order = (0..=MAX_LABELS).map(|i| format!("l{i}")).collect();
        let err = TrainRequest::new(params).unwrap_err();
        assert!(matches!(
            err,
            TrainRequestError::LabelOrderCount { actual } if actual == MAX_LABELS + 1
        ));
    }

    /// Rust の方が厳しいケース: `kind_version` に負の値を渡すと、Python の
    /// `isinstance(v, int)` 検査（`bool` 除き通す）とは異なり、`serde` の
    /// `u32` デシリアライズ自体が失敗する（`NotJson`。共有 fixture には
    /// 含めず、Rust 単体テストとして残す。issue #177 実装計画）。
    #[test]
    fn rust_is_stricter_than_python_for_negative_kind_version() {
        let json = br#"{
            "schema_version": 1,
            "kind": "c3",
            "kind_version": -1,
            "label_order": ["a", "b"],
            "max_bytes": 512,
            "seed": 0,
            "device": "cpu",
            "root": "/tmp/fandhe-edge-train-test",
            "train_path": "train.jsonl",
            "out_dir": "out"
        }"#;
        let err = TrainRequest::from_json_slice(json).unwrap_err();
        assert_eq!(err.reason_code(), "invalid_request");
        assert!(matches!(err, TrainRequestError::NotJson { .. }));
    }

    /// Rust の方が厳しいケース: `crates/core::Definition::parse` が扱う
    /// `serde_json::Value` は重複フィールドを後勝ちで黙って受理するが、
    /// 本 crate の型付きデシリアライズ（`serde_derive`）は重複フィールドを
    /// 構文エラーとして拒否する。Python 側 JSON パーサー（`json.loads`）も
    /// 後勝ちで受理するため、この差異は fixture には含めず Rust 単体
    /// テストとして残す（issue #177 実装計画「Rust の方が厳しいケース」）。
    #[test]
    fn rust_rejects_duplicate_fields_unlike_json_loads() {
        let json = br#"{
            "schema_version": 1,
            "kind": "c3",
            "kind": "c1",
            "kind_version": 1,
            "label_order": ["a", "b"],
            "max_bytes": 512,
            "seed": 0,
            "device": "cpu",
            "root": "/tmp/fandhe-edge-train-test",
            "train_path": "train.jsonl",
            "out_dir": "out"
        }"#;
        let err = TrainRequest::from_json_slice(json).unwrap_err();
        assert_eq!(err.reason_code(), "invalid_request");
        assert!(matches!(err, TrainRequestError::NotJson { .. }));
    }

    /// path 構文検査: `.`・`..`・空文字列・絶対パス・NUL を個別に確認する。
    #[test]
    fn req39_rejects_malformed_relative_paths() {
        for bad in ["", ".", "..", "/abs", "a/../b", "a/..", "with\0nul"] {
            let mut params = valid_params();
            params.train_path = bad.to_string();
            let err = TrainRequest::new(params).unwrap_err();
            assert_eq!(err.reason_code(), "invalid_path", "case: {bad:?}");
            assert_eq!(
                err.exit_code(),
                fandhe_edge_core::exitcode::ExitCode::InvalidInput
            );
        }
    }

    /// `./train.jsonl` のような先頭の `.` 構成要素は無視され（`PurePosixPath`
    /// と同じ挙動）、残りの構成要素があれば受理する。
    #[test]
    fn req39_accepts_relative_path_with_leading_dot_component() {
        let mut params = valid_params();
        params.train_path = "./train.jsonl".to_string();
        TrainRequest::new(params)
            .expect("leading dot component should be ignored like PurePosixPath");
    }

    /// `root` の構文検査: 絶対パスでない・空・NUL を拒否する。
    #[test]
    fn req39_rejects_malformed_root() {
        for bad in ["", "relative/root", "with\0nul"] {
            let mut params = valid_params();
            params.root = bad.to_string();
            let err = TrainRequest::new(params).unwrap_err();
            assert_eq!(err.reason_code(), "invalid_path", "case: {bad:?}");
        }
    }

    /// `to_json_vec` は `config` を省略時も `{}` として出力する
    /// （`contract.py` の既定値と同じ）。
    #[test]
    fn req18_to_json_vec_always_emits_config_object() {
        let req = TrainRequest::new(valid_params()).expect("valid params must be accepted");
        let bytes = req.to_json_vec().expect("serialize");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("valid json");
        assert_eq!(value.get("config"), Some(&serde_json::json!({})));
        assert!(value.get("time_limit_seconds").is_none());
        assert!(value.get("rss_limit_bytes").is_none());
    }

    /// `label_order_from_definition`: 選択肢 1 件の定義は投影時に拒否される
    /// （学習リクエストの `MIN_LABELS` は 2 件のため）。
    #[test]
    fn req18_label_order_from_definition_rejects_single_option_definition() {
        let json = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "single",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "only", "display_name": "Only", "description": "唯一" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def = Definition::parse(json).expect("valid definition");
        let err = label_order_from_definition(&def).unwrap_err();
        assert!(matches!(
            err,
            TrainRequestError::LabelOrderCount { actual: 1 }
        ));
    }

    /// `label_order_from_definition`: 257 バイトの `id` を持つ定義は拒否される。
    #[test]
    fn req18_label_order_from_definition_rejects_oversized_label_byte() {
        let long_id = "a".repeat(MAX_LABEL_BYTES + 1);
        let json = format!(
            r#"{{
                "schema": "fandhe-edge-model-definition/v1",
                "name": "oversized",
                "version": 1,
                "judgment_type": "single_select",
                "options": [
                    {{ "id": "{long_id}", "display_name": "A", "description": "A" }},
                    {{ "id": "b", "display_name": "B", "description": "B" }}
                ],
                "io": {{ "input": "bytes" }}
            }}"#
        );
        let def = Definition::parse(&json).expect("valid definition");
        let err = label_order_from_definition(&def).unwrap_err();
        assert!(matches!(
            err,
            TrainRequestError::LabelOrderLabelTooLong { index: 0 }
        ));
    }

    /// `label_order_from_definition`: 2 件以上・上限以下の定義は宣言順を
    /// 保った `label_order` になる。
    #[test]
    fn req18_label_order_from_definition_preserves_declaration_order() {
        let json = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "ok",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "positive", "display_name": "Positive", "description": "P" },
                { "id": "negative", "display_name": "Negative", "description": "N" },
                { "id": "neutral", "display_name": "Neutral", "description": "U" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def = Definition::parse(json).expect("valid definition");
        let label_order =
            label_order_from_definition(&def).expect("valid definition should project");
        assert_eq!(
            label_order.as_slice(),
            &[
                "positive".to_string(),
                "negative".to_string(),
                "neutral".to_string()
            ]
        );
    }
}
