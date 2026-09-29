//! 学習リクエスト JSON（Rust 側 CLI が学習ワーカーへ渡す入力。子プロセスへの
//! 配線は `crate::process::run_train`〔issue #178〕）の検証済み型
//! （REQ-18・REQ-19・REQ-39）。
//!
//! スキーマは `trainer/src/fandhe_edge_trainer/contract.py`（schema_version 1）
//! と同じ 13 項目に加え、任意項目 `validation_inputs`（学習ジョブ内での採点用
//! validation 入力。`{id,input}` のみで正解ラベルは持たない。[`ValidationInput`]・
//! REQ-27）を持つ。値の検証は同モジュール・`limits.py`・`guard.py` の
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
use fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES;
use fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES;
use serde::{Deserialize, Serialize};

use crate::error::TrainRequestError;
use crate::limits::{
    MAX_LABEL_BYTES, MAX_LABELS, MAX_MAX_BYTES, MAX_REQUEST_BYTES, MAX_RESULT_BYTES,
    MAX_RESULT_BYTES_WITH_VALIDATION, MAX_SEED, MAX_TRAIN_RSS_BYTES, MAX_TRAIN_WALL_SECONDS,
    MAX_VALIDATION_INPUT_TOTAL_BYTES, MIN_LABELS, MIN_MAX_BYTES, MIN_SEED, REQUEST_SCHEMA_VERSION,
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

/// 学習ジョブ内での採点用に、学習ワーカーへ渡す validation 入力 1 件
/// （`{"id","input"}`。REQ-27・REQ-18。issue #84 PR #238・選択肢 2）。
///
/// **正解ラベルは持たない**（REQ-27「推論関数には `input` だけを渡す」。
/// 学習ワーカーへ gold を渡さないことを型で保証する）。`id` は
/// validation レコードの識別子で、結果の `validation_predictions` の
/// 突き合わせに使う。`input` は `String`（UTF-8 保証）で、UTF-8 でない
/// バイト列はこの型を作る前に呼び出し元が拒否する
/// （`crate::search` の `SearchError::ValidationInputNotUtf8`。
/// 子プロセスを起動する前に `invalid_input` で拒否する）。
///
/// `Debug` は件数・長さのみで、`id`・`input` の内容（学習・評価データ本文に
/// なりうる）を出力しない（`.claude/rules/security.md`）。
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ValidationInput {
    id: String,
    input: String,
}

impl ValidationInput {
    /// 構築のみ。検証は [`TrainRequest::with_validation_inputs`]・
    /// [`TrainRequest::from_json_slice`] が行う。
    #[must_use]
    pub fn new(id: String, input: String) -> Self {
        Self { id, input }
    }

    /// validation レコードの識別子。
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// 推論関数へ渡す入力本文。
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }
}

impl std::fmt::Debug for ValidationInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidationInput")
            .field("id_len", &self.id.len())
            .field("input_len", &self.input.len())
            .finish()
    }
}

/// `validation_inputs` の検証（`contract.py::_validate_validation_inputs` と
/// 同じ規則）。件数は 1 件以上、`id` は非空・[`MAX_INPUT_ID_BYTES`] 以下・
/// 重複なし、`input` は [`MAX_INFER_INPUT_BYTES`] 以下、`id`＋`input` の
/// 合計は [`MAX_VALIDATION_INPUT_TOTAL_BYTES`] 以下。リクエスト JSON の内側を
/// 通るため実効上限は [`MAX_REQUEST_BYTES`] が決める
/// （[`MAX_VALIDATION_INPUT_TOTAL_BYTES`] の doc 参照）。
fn validate_validation_inputs(inputs: &[ValidationInput]) -> Result<(), TrainRequestError> {
    if inputs.is_empty() {
        return Err(TrainRequestError::ValidationInputsEmpty);
    }
    let mut seen: HashSet<&str> = HashSet::with_capacity(inputs.len().min(4096));
    let mut total: usize = 0;
    for (index, item) in inputs.iter().enumerate() {
        if item.id.is_empty() || item.id.len() > MAX_INPUT_ID_BYTES {
            return Err(TrainRequestError::ValidationInputInvalidId { index });
        }
        if item.input.len() > MAX_INFER_INPUT_BYTES {
            return Err(TrainRequestError::ValidationInputTooLarge { index });
        }
        total = total
            .saturating_add(item.id.len())
            .saturating_add(item.input.len());
        if total > MAX_VALIDATION_INPUT_TOTAL_BYTES {
            return Err(TrainRequestError::ValidationInputsTotalBytesExceeded);
        }
        if !seen.insert(item.id.as_str()) {
            return Err(TrainRequestError::ValidationInputDuplicateId { index });
        }
    }
    Ok(())
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
    /// 学習ジョブ内での採点用 validation 入力（任意。gold は持たない）。
    validation_inputs: Option<Vec<ValidationInput>>,
}

/// [`ValidationInput`] の JSON 読み込み用中間表現（未知フィールドを拒否する）。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawValidationInput {
    id: String,
    input: String,
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
    /// キー省略時は `None`（既定値に解決）だが、明示的な JSON `null` は
    /// 拒否する（`deserialize_present` 参照）。`contract.py::validate_request`
    /// が `raw.get(field, DEFAULT)` を使うため、キーが存在して値が `null` の
    /// 場合は `DEFAULT` へ解決されず `isinstance(None, int)` が `False` に
    /// なって `invalid_request` を返すことと揃える（REQ-39・Bugbot 指摘。
    /// PR #220）。
    #[serde(default, deserialize_with = "deserialize_present")]
    time_limit_seconds: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_present")]
    rss_limit_bytes: Option<u64>,
    /// キー省略時は `None`。明示的な `null` は拒否する（他の任意項目と同じ）。
    #[serde(default, deserialize_with = "deserialize_present")]
    validation_inputs: Option<Vec<RawValidationInput>>,
}

/// `Option<T>` フィールド用の `deserialize_with`。キー省略時は
/// `#[serde(default)]`（`None`）に任せ、キーが存在する場合は必ず `T` として
/// 解析する（`T::deserialize` は JSON `null` を受け付けないため、明示的な
/// `null` はここでエラーになる）。`Option<T>::deserialize` を素の型で使うと
/// 「省略」と「明示的な `null`」を区別できず、どちらも `None` になってしまう
/// ため必要（REQ-39・Bugbot 指摘。PR #220。標準的な serde イディオム）。
fn deserialize_present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
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
    #[serde(skip_serializing_if = "Option::is_none")]
    validation_inputs: Option<&'a [ValidationInput]>,
}

fn is_default_wall_seconds(value: &u32) -> bool {
    *value == MAX_TRAIN_WALL_SECONDS
}

fn is_default_rss_bytes(value: &u64) -> bool {
    *value == MAX_TRAIN_RSS_BYTES
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

/// [`TrainRequest::to_json_vec`] 専用の `Vec<u8>` ライター。書き込み総量が
/// `limit` を超えた時点で `Err` を返し、以降の書き込み（`serde_json` の
/// 直列化）を打ち切る。`config` に外部由来の巨大な値が入っていても、
/// [`MAX_REQUEST_BYTES`] を大幅に超えるメモリを先に確保しないための資源上限
/// （REQ-39「資源の上限」・P1。codex review PR #220）。
struct LimitedVecWriter {
    buf: Vec<u8>,
    limit: usize,
    limit_exceeded: bool,
}

impl LimitedVecWriter {
    fn new(limit: usize) -> Self {
        Self {
            buf: Vec::new(),
            limit,
            limit_exceeded: false,
        }
    }

    fn into_inner(self) -> Vec<u8> {
        self.buf
    }

    fn limit_exceeded(&self) -> bool {
        self.limit_exceeded
    }
}

impl std::io::Write for LimitedVecWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        if self.buf.len().saturating_add(data.len()) > self.limit {
            self.limit_exceeded = true;
            return Err(std::io::Error::other(
                "train request serialization exceeds size limit",
            ));
        }
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
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
            validation_inputs: None,
        })
    }

    /// 学習ジョブ内での採点用 validation 入力を付ける（REQ-18・REQ-27。
    /// issue #84 PR #238・選択肢 2）。正解ラベルは受け取らない。検証は
    /// [`validate_validation_inputs`] と同じ規則で fail-closed に行い、
    /// 入力は 1 件以上でなければならない（付けない場合は
    /// `validation_inputs` キー自体を出力しない）。
    pub fn with_validation_inputs(
        mut self,
        inputs: Vec<ValidationInput>,
    ) -> Result<Self, TrainRequestError> {
        validate_validation_inputs(&inputs)?;
        self.validation_inputs = Some(inputs);
        Ok(self)
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
                line: e.line(),
                column: e.column(),
                category: e.classify(),
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
        let request = Self::new(params)?;
        match raw.validation_inputs {
            None => Ok(request),
            Some(items) => request.with_validation_inputs(
                items
                    .into_iter()
                    .map(|item| ValidationInput::new(item.id, item.input))
                    .collect(),
            ),
        }
    }

    /// 検証済みの [`TrainRequest`] を、`contract.py` と同じスキーマの JSON
    /// バイト列へ直列化する。`config` は [`TrainRequest::new`] の時点では
    /// サイズを検査しない（任意の JSON を保持しうる。本モジュールの doc
    /// 参照）ため、ここで検査する。[`LimitedVecWriter`] へ書き込みながら
    /// [`MAX_REQUEST_BYTES`] 超過を検出した時点で直列化を打ち切ることで、
    /// 外部由来の巨大な `config` を渡された場合でも上限を超える確保を
    /// 先に行わない（REQ-39 ガード層「資源の上限」・P1。codex review
    /// PR #220: 旧実装は `serde_json::to_vec` で全体を確保してから長さを
    /// 検査していた）。
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
            validation_inputs: self.validation_inputs.as_deref(),
        };
        let mut writer = LimitedVecWriter::new(MAX_REQUEST_BYTES);
        match serde_json::to_writer(&mut writer, &wire) {
            Ok(()) => Ok(writer.into_inner()),
            // 上限超過による打ち切り（本 wire 型の値はいずれも JSON へ直列化
            // 可能なため、他の直列化失敗要因では到達しない）。実際の最終
            // サイズは打ち切りのため未確定だが、上限超過を示す下限値として
            // 上限 + 1 を報告する。
            Err(_) if writer.limit_exceeded() => Err(TrainRequestError::TooLarge {
                size: MAX_REQUEST_BYTES + 1,
                limit: MAX_REQUEST_BYTES,
            }),
            Err(_) => Err(TrainRequestError::SerializeFailed),
        }
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

    /// 学習ジョブ内での採点用 validation 入力（付けていなければ `None`）。
    #[must_use]
    pub fn validation_inputs(&self) -> Option<&[ValidationInput]> {
        self.validation_inputs.as_deref()
    }

    /// このリクエストに対する結果 JSON（ワーカーの標準出力）の読み込み上限
    /// （バイト）。`validation_inputs` を持つ場合だけ、予測列を含むぶん緩い
    /// [`MAX_RESULT_BYTES_WITH_VALIDATION`]、それ以外は [`MAX_RESULT_BYTES`]。
    #[must_use]
    pub fn max_result_bytes(&self) -> usize {
        if self.validation_inputs.is_some() {
            MAX_RESULT_BYTES_WITH_VALIDATION
        } else {
            MAX_RESULT_BYTES
        }
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

    /// security.md「秘密情報の混入防止」・REQ-39: 未知フィールド名に秘密
    /// 風の文字列を含めても、`Display`／`Debug` のどちらにも現れない
    /// （`serde_json::Error::to_string()` を経由すると
    /// `unknown field \`secret_key_XYZ\`, expected ...` のように入力由来の
    /// キー名がそのまま出てしまうため、位置情報のみを保持する。PR #220
    /// レビュー指摘 P0）。
    #[test]
    fn req39_unknown_field_secret_does_not_leak_into_error_text() {
        let json = br#"{
            "schema_version": 1,
            "kind": "c3",
            "kind_version": 1,
            "label_order": ["a", "b"],
            "max_bytes": 512,
            "seed": 0,
            "device": "cpu",
            "root": "/tmp/fandhe-edge-train-test",
            "train_path": "train.jsonl",
            "out_dir": "out",
            "secret_key_XYZ": "leak-me"
        }"#;
        let err = TrainRequest::from_json_slice(json).unwrap_err();
        assert!(matches!(err, TrainRequestError::NotJson { .. }));
        let display = err.to_string();
        let debug = format!("{err:?}");
        assert!(!display.contains("secret_key_XYZ"));
        assert!(!display.contains("leak-me"));
        assert!(!debug.contains("secret_key_XYZ"));
        assert!(!debug.contains("leak-me"));
    }

    /// REQ-39: `config` の値に JSON 以外の数値トークン（`NaN`・`Infinity`・
    /// `-Infinity`）を含むリクエストは、`serde_json::Value` へのデシリアライズ
    /// 自体が構文エラーとして拒否する（学習ワーカーの
    /// `contract.py::parse_request_bytes` が同じトークンを全体として拒否する
    /// ことと揃える。PR #220 Bugbot 指摘の確認）。
    #[test]
    fn req39_rejects_non_json_number_tokens_in_config() {
        for token in ["NaN", "Infinity", "-Infinity"] {
            let json = format!(
                r#"{{
                    "schema_version": 1,
                    "kind": "c3",
                    "kind_version": 1,
                    "config": {{"lr": {token}}},
                    "label_order": ["a", "b"],
                    "max_bytes": 512,
                    "seed": 0,
                    "device": "cpu",
                    "root": "/tmp/fandhe-edge-train-test",
                    "train_path": "train.jsonl",
                    "out_dir": "out"
                }}"#
            );
            let err = TrainRequest::from_json_slice(json.as_bytes()).unwrap_err();
            assert!(
                matches!(
                    err,
                    TrainRequestError::NotJson {
                        category: serde_json::error::Category::Syntax,
                        ..
                    }
                ),
                "token {token} must be rejected as a JSON syntax error"
            );
        }
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

    // ---- validation_inputs（REQ-18・REQ-27・REQ-39。issue #84 PR #238・選択肢 2）----

    fn vi(id: &str, input: &str) -> ValidationInput {
        ValidationInput::new(id.to_string(), input.to_string())
    }

    /// REQ-27: `validation_inputs` は `{id,input}` のみを運び（gold は持てない）、
    /// JSON 往復で値が保たれる。付けていないリクエストにはキー自体が現れない。
    #[test]
    fn req27_validation_inputs_round_trip_and_key_omitted_when_absent() {
        let plain = TrainRequest::new(valid_params()).expect("valid");
        let plain_json = String::from_utf8(plain.to_json_vec().expect("serialize")).expect("utf8");
        assert!(!plain_json.contains("validation_inputs"));
        assert_eq!(plain.validation_inputs(), None);
        assert_eq!(plain.max_result_bytes(), MAX_RESULT_BYTES);

        let request = plain
            .with_validation_inputs(vec![vi("r1", "alpha"), vi("r2", "beta")])
            .expect("valid validation inputs");
        assert_eq!(request.max_result_bytes(), MAX_RESULT_BYTES_WITH_VALIDATION);
        let bytes = request.to_json_vec().expect("serialize");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(
            value["validation_inputs"],
            serde_json::json!([
                {"id": "r1", "input": "alpha"},
                {"id": "r2", "input": "beta"}
            ])
        );
        let parsed = TrainRequest::from_json_slice(&bytes).expect("round trip");
        assert_eq!(parsed, request);
    }

    /// REQ-39: `validation_inputs` の各規則（空配列・空 id・長すぎる id・
    /// 重複 id・1 件の巨大入力・合計超過）は fail-closed で拒否される。
    #[test]
    fn req39_validation_inputs_rejections() {
        let base = || TrainRequest::new(valid_params()).expect("valid");
        assert_eq!(
            base().with_validation_inputs(vec![]).unwrap_err(),
            TrainRequestError::ValidationInputsEmpty
        );
        assert_eq!(
            base()
                .with_validation_inputs(vec![vi("ok", "a"), vi("", "b")])
                .unwrap_err(),
            TrainRequestError::ValidationInputInvalidId { index: 1 }
        );
        let long_id = "x".repeat(MAX_INPUT_ID_BYTES + 1);
        assert_eq!(
            base()
                .with_validation_inputs(vec![vi(&long_id, "a")])
                .unwrap_err(),
            TrainRequestError::ValidationInputInvalidId { index: 0 }
        );
        assert_eq!(
            base()
                .with_validation_inputs(vec![vi("a", "1"), vi("a", "2")])
                .unwrap_err(),
            TrainRequestError::ValidationInputDuplicateId { index: 1 }
        );
        let huge = "y".repeat(MAX_INFER_INPUT_BYTES + 1);
        assert_eq!(
            base()
                .with_validation_inputs(vec![vi("a", &huge)])
                .unwrap_err(),
            TrainRequestError::ValidationInputTooLarge { index: 0 }
        );
        // 合計超過: 1 件あたり上限ちょうどの入力を、合計上限を超える件数だけ並べる。
        let max_each = "z".repeat(MAX_INFER_INPUT_BYTES);
        let count = MAX_VALIDATION_INPUT_TOTAL_BYTES / MAX_INFER_INPUT_BYTES + 1;
        let many: Vec<ValidationInput> = (0..count)
            .map(|i| ValidationInput::new(format!("id{i}"), max_each.clone()))
            .collect();
        assert_eq!(
            base().with_validation_inputs(many).unwrap_err(),
            TrainRequestError::ValidationInputsTotalBytesExceeded
        );
    }

    /// REQ-39: 実効上限は `MAX_REQUEST_BYTES`（リクエスト全体）。`with_validation_inputs`
    /// は通る 1 MiB ちょうどの入力でも、直列化で `TooLarge` になる。
    #[test]
    fn req39_validation_inputs_effective_cap_is_max_request_bytes() {
        let max_each = "z".repeat(MAX_INFER_INPUT_BYTES);
        let request = TrainRequest::new(valid_params())
            .expect("valid")
            .with_validation_inputs(vec![vi("a", &max_each)])
            .expect("per-record limit satisfied");
        assert!(matches!(
            request.to_json_vec(),
            Err(TrainRequestError::TooLarge { .. })
        ));
    }

    /// REQ-39: JSON 経由でも要素の未知フィールド（gold を紛れ込ませる等）・
    /// 明示的な `null`・型違いは拒否され、重複 id は検証で拒否される。
    #[test]
    fn req27_validation_inputs_json_rejects_gold_and_null() {
        let base = |extra: &str| {
            format!(
                r#"{{"schema_version":1,"kind":"c3","kind_version":1,"label_order":["a","b"],"max_bytes":512,"seed":0,"device":"cpu","root":"/r","train_path":"t.jsonl","out_dir":"o",{extra}}}"#
            )
        };
        let gold = base(r#""validation_inputs":[{"id":"r1","input":"x","label":"a"}]"#);
        assert!(matches!(
            TrainRequest::from_json_slice(gold.as_bytes()),
            Err(TrainRequestError::NotJson { .. })
        ));
        let null = base(r#""validation_inputs":null"#);
        assert!(matches!(
            TrainRequest::from_json_slice(null.as_bytes()),
            Err(TrainRequestError::NotJson { .. })
        ));
        let dup = base(r#""validation_inputs":[{"id":"r","input":"x"},{"id":"r","input":"y"}]"#);
        assert_eq!(
            TrainRequest::from_json_slice(dup.as_bytes()).unwrap_err(),
            TrainRequestError::ValidationInputDuplicateId { index: 1 }
        );
    }

    /// セキュリティ: `Debug` は `id`・`input` の内容を出力しない。
    #[test]
    fn req39_validation_input_debug_does_not_leak_content() {
        let request = TrainRequest::new(valid_params())
            .expect("valid")
            .with_validation_inputs(vec![vi("secret-id-MARKER", "secret-input-MARKER")])
            .expect("valid");
        let debug = format!(
            "{request:?} {:?}",
            vi("secret-id-MARKER", "secret-input-MARKER")
        );
        assert!(!debug.contains("MARKER"), "{debug}");
        assert!(debug.contains("input_len"));
    }
}
