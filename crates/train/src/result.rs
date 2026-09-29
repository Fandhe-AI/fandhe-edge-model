//! 学習ワーカーの標準出力（結果 JSON）の検証済み型（REQ-21・REQ-39）。
//!
//! `trainer/src/fandhe_edge_trainer/cli.py::main`・`supervisor.py` が
//! `print(json.dumps(...))` で出す JSON 1 行を、信頼しない外部入力として
//! 解析する（学習ワーカーは Rust 側から見て子プロセス。起動そのものは
//! `crate::process::run_train`〔issue #178〕が担う）。成功時は
//! `artifact.py::build_artifact` と同じ 11 項目の
//! 成果物記録、失敗時は `{"status":"error","code":...,"message":...}` を扱う。
//!
//! **ガード層（REQ-39「経路の閉じ込め」「完全性と版」）との関係**: 唯一の
//! 公開の組み立て経路 [`TrainOutcome::from_worker_stdout`] は、この結果に
//! 対応する [`crate::request::TrainRequest`] を必須で受け取り、`artifact_dir`
//! がその `root`／`out_dir` 配下に閉じ込められていること、`kind`・
//! `kind_version`・`label_order`・`max_bytes`・`candidate_label` が依頼内容と
//! 一致することを検査する（ワーカーが依頼と異なる種類・未対応の版・root 外
//! のパス・別種類を名乗る成果物を返しても成功扱いにしない）。`config` は
//! 「`kind` ごとの既定値（共有 fixture `fixtures/train_contract/
//! kind_defaults.json`）に `request.config` を上書きした実効 config」との
//! 完全一致で検査する（[`crate::kind_defaults::effective_config`] 参照。
//! `kinds/c1.py`・`kinds/c3.py::train` の `{**DEFAULT_CONFIG,
//! **request.config}` と同じ上書き規則）。fixture に登録の無い `kind` は
//! 実効 config を算出できないため拒否する。`root` の symlink 解決
//! （`std::fs::canonicalize`）を挟む点を除き照合は文字列レベルの検査に留まり、
//! 実際の FS 上の閉じ込め（dir_fd 等の多層防御）は学習ワーカー自身
//! （`guard.py::confine`）が担う（`crates/train/src/request.rs` のモジュール
//! doc と同じ設計。詳細は [`expected_artifact_dir`]・
//! [`canonicalize_root_best_effort`] 参照）。

use fandhe_edge_core::exitcode::ExitCode;
use serde::{Deserialize, Deserializer, Serialize};

use crate::error::TrainResultError;
use fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES;

use crate::request::{LabelOrder, TrainRequest};

/// `#[serde(default, deserialize_with = "deserialize_present")]` と組み合わせ、
/// 「キーが無い」（既定値 `None`）と「キーが `null`」（`Some(None)`）を区別する
/// ための deserializer（いわゆる double-Option イディオム）。
///
/// [`RawOutcome`] の各フィールドは素の `Option<T>` のままだと、キー欠落時の
/// `None` とキーはあるが値が `null` の場合の `None` を区別できない。結果
/// `"status":"ok"` に余分な `"code":null` を足した出力まで正当な成功結果として
/// 受理してしまっていた（REQ-21・REQ-39・P1。codex 指摘 PR #220）。
fn deserialize_present<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    T::deserialize(deserializer).map(Some)
}

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

/// 学習ワーカーが返しうる既知の失敗コード（REQ-21・REQ-39）。
///
/// `trainer/src/fandhe_edge_trainer/` 配下（`kinds/` サブディレクトリを含む）
/// の `WorkerError(code, ...)` 呼び出し箇所を全数 grep して洗い出した 12 種類
/// に限定する（`errors.py`・`contract.py`・`guard.py`・`artifact.py`・
/// `cli.py`・`supervisor.py`・`budget.py`・`kinds/__init__.py`・`kinds/c1.py`・
/// `kinds/c3.py`）。任意の `[a-z_]+` 文字列を検証済みとして受理すると、
/// ワーカーが本来返さないコード（例: `"success"`）を許してしまう（PR #220
/// レビュー指摘 P1）ため、許可リストで検証し、未知のコードは
/// [`TrainResultError::InvalidFailureCode`]（呼び出し元は
/// `runtime_error`／`RuntimeError`〔70〕として扱う）として拒否する
/// （fail-closed）。この一致は `crates/train/tests/failure_code_matches_trainer.rs`
/// で Python 側の grep 結果（`kinds/` を含む再帰走査）と機械照合する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FailureCode {
    /// `contract.py::_invalid_request` 等。リクエストの構文・型・範囲が不正。
    InvalidRequest,
    /// `contract.py::_invalid_data`・`kinds/c1.py`。学習データの構造が不正。
    InvalidData,
    /// `contract.py::_limit_exceeded`・`budget.py`・`kinds/c1.py`。資源上限
    /// 超過（REQ-39）。
    LimitExceeded,
    /// `guard.py`。経路の構文検証・閉じ込め検証の失敗（REQ-39「経路の閉じ込め」）。
    InvalidPath,
    /// `guard.py`。閉じ込め対象の経路に symlink を許さない検証の失敗。
    SymlinkNotAllowed,
    /// `artifact.py`。成果物 ONNX の sha256 検証・読み込み失敗
    /// （REQ-39「完全性と版」）。
    IntegrityCheckFailed,
    /// `contract.py`。`out_dir` の予約・確定処理での競合・不整合。
    OutputConflict,
    /// `kinds/__init__.py`。未対応の `kind`（学習ワーカー側のレジストリに
    /// 存在しない種類）。
    UnsupportedKind,
    /// `kinds/__init__.py`。指定 `kind` に対応する `kind_version` が未対応。
    UnsupportedKindVersion,
    /// `kinds/c1.py`・`kinds/c3.py`。`config` の値が当該 `kind` の許可範囲・
    /// 許可フィールドを満たさない（本 crate の `TrainRequest` は「JSON
    /// オブジェクトであること」だけを検査し、`kind` ごとの `config` 検証は
    /// 再実装しない。モジュール doc「スコープ外」参照）。
    InvalidConfig,
    /// `kinds/c1.py`・`kinds/c3.py`。学習ループの損失が非有限値になった
    /// （作り直し判定・再学習の対象。`ExitCode::Pending`〔12〕に対応）。
    TrainingDiverged,
    /// 上記以外のワーカー内部エラー（`supervisor.py` が子プロセスの異常終了・
    /// 不正な標準出力を検出した場合を含む）。
    RuntimeError,
}

impl FailureCode {
    /// ワーカー出力の `code` 文字列（`errors.py::WorkerError.code`）と同じ
    /// 語彙。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            FailureCode::InvalidRequest => "invalid_request",
            FailureCode::InvalidData => "invalid_data",
            FailureCode::LimitExceeded => "limit_exceeded",
            FailureCode::InvalidPath => "invalid_path",
            FailureCode::SymlinkNotAllowed => "symlink_not_allowed",
            FailureCode::IntegrityCheckFailed => "integrity_check_failed",
            FailureCode::OutputConflict => "output_conflict",
            FailureCode::UnsupportedKind => "unsupported_kind",
            FailureCode::UnsupportedKindVersion => "unsupported_kind_version",
            FailureCode::InvalidConfig => "invalid_config",
            FailureCode::TrainingDiverged => "training_diverged",
            FailureCode::RuntimeError => "runtime_error",
        }
    }

    /// 許可リストに一致する場合だけ `Some` を返す（fail-closed。未知の
    /// コードは `None`）。
    #[must_use]
    fn parse(code: &str) -> Option<Self> {
        Some(match code {
            "invalid_request" => FailureCode::InvalidRequest,
            "invalid_data" => FailureCode::InvalidData,
            "limit_exceeded" => FailureCode::LimitExceeded,
            "invalid_path" => FailureCode::InvalidPath,
            "symlink_not_allowed" => FailureCode::SymlinkNotAllowed,
            "integrity_check_failed" => FailureCode::IntegrityCheckFailed,
            "output_conflict" => FailureCode::OutputConflict,
            "unsupported_kind" => FailureCode::UnsupportedKind,
            "unsupported_kind_version" => FailureCode::UnsupportedKindVersion,
            "invalid_config" => FailureCode::InvalidConfig,
            "training_diverged" => FailureCode::TrainingDiverged,
            "runtime_error" => FailureCode::RuntimeError,
            _ => return None,
        })
    }

    /// 本 crate・`crates/train/tests/failure_code_matches_trainer.rs` が共有
    /// する全許可コードの一覧（宣言順を安定させ、機械照合テストで走査する）。
    #[must_use]
    pub const fn all() -> &'static [FailureCode] {
        &[
            FailureCode::InvalidRequest,
            FailureCode::InvalidData,
            FailureCode::LimitExceeded,
            FailureCode::InvalidPath,
            FailureCode::SymlinkNotAllowed,
            FailureCode::IntegrityCheckFailed,
            FailureCode::OutputConflict,
            FailureCode::UnsupportedKind,
            FailureCode::UnsupportedKindVersion,
            FailureCode::InvalidConfig,
            FailureCode::TrainingDiverged,
            FailureCode::RuntimeError,
        ]
    }

    /// 対応する終了コード（REQ-21）。`trainer/src/fandhe_edge_trainer/`
    /// 配下の `WorkerError(code, ..., ExitCode.<NAME>)` 呼び出しを全数走査
    /// して洗い出した対応表（issue #178 実装計画）。学習ワーカーが同じ
    /// `code` に対して複数の終了コードを返すことはない（this の全 12
    /// 種は 1 対 1）。この一致は
    /// `crates/train/tests/failure_code_matches_trainer.rs` で Python 側の
    /// `ExitCode.<NAME>` と機械照合する。
    #[must_use]
    pub const fn exit_code(self) -> ExitCode {
        match self {
            FailureCode::InvalidRequest
            | FailureCode::InvalidData
            | FailureCode::InvalidPath
            | FailureCode::SymlinkNotAllowed
            | FailureCode::OutputConflict
            | FailureCode::UnsupportedKind
            | FailureCode::UnsupportedKindVersion
            | FailureCode::InvalidConfig => ExitCode::InvalidInput,
            FailureCode::LimitExceeded => ExitCode::LimitExceeded,
            FailureCode::TrainingDiverged => ExitCode::Pending,
            FailureCode::IntegrityCheckFailed | FailureCode::RuntimeError => ExitCode::RuntimeError,
        }
    }
}

impl Serialize for FailureCode {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

/// 失敗時のエラー（`{"status":"error","code":...,"message":...}`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerFailure {
    code: FailureCode,
    message: String,
}

/// `message` の上限バイト数（Rust 側だけの防御。`errors.py` はメッセージへ
/// データ本文を含めない方針だが、想定外の長大化に備える）。
const MAX_FAILURE_MESSAGE_BYTES: usize = 4 * 1024;

impl WorkerFailure {
    fn parse(code: String, message: String) -> Result<Self, TrainResultError> {
        let Some(code) = FailureCode::parse(&code) else {
            return Err(TrainResultError::InvalidFailureCode);
        };
        if message.len() > MAX_FAILURE_MESSAGE_BYTES {
            return Err(TrainResultError::FailureMessageTooLong);
        }
        Ok(Self { code, message })
    }

    pub fn code(&self) -> &str {
        self.code.as_str()
    }

    /// 型付きの失敗コード（[`FailureCode`]）。呼び出し元（`crates/train::time_allotment`
    /// の `run_candidate`。TASK-18.1-1）が `code() == "limit_exceeded"` の
    /// ような文字列比較を書かずに、許可リストの列挙値で分岐できるようにする。
    /// `crates/train/src/process.rs`（REQ-21・REQ-39・#178）も終了コード写像の
    /// 突き合わせ〔ワーカーの終了コードと JSON の `code` が対応する終了コードで
    /// 一致するか〕にこの値を使う。
    #[must_use]
    pub fn failure_code(&self) -> FailureCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

/// 学習ワーカーの結果（成功／失敗の 2 択）。
///
/// `Ok` バリアントは [`SuccessOutcome`]（フィールド非公開・アクセサのみ公開）
/// を保持する。フィールドを直接公開しないのは、公開されたバリアントの
/// フィールドから別 crate が検証済み [`ArtifactRecord`] を使って任意の
/// `artifact_dir` を持つ成功結果を直接組み立て、`from_worker_stdout` の経路
/// 検証（ガード層「経路の閉じ込め」）を迂回できてしまうため
/// （REQ-39・P0。codex review PR #220）。`enum` の `pub` バリアントは
/// フィールド単位で非公開にできない（フィールドが public な struct-like
/// variant になる）ため、フィールド非公開の struct を経由させて型で防ぐ
/// （`crates/core/src/definition.rs`・本 crate の `TrainRequest` と同じ設計）。
#[derive(Debug, Clone, PartialEq)]
pub enum TrainOutcome {
    Ok(SuccessOutcome),
    Error(WorkerFailure),
}

/// 学習ジョブ内で採点した validation の予測 1 件の `status`
/// （`kinds/autoregressive.py::build_prediction_record` の語彙のうち、
/// `crates/data/src/eval_input.rs` が受理する `ok`／`abstain`／`error`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationPredictionStatus {
    Ok,
    Abstain,
    Error,
}

/// 学習ジョブ内で採点した validation の予測 1 件（`{id,status,predicted_label}`。
/// 既存の推論出力〔`prediction_record_to_json_line`〕と同じ形から `scores` を
/// 除いたもの。REQ-27・issue #84 PR #238・選択肢 2）。
///
/// `Debug` は `status` のみで、`id`・`predicted_label`（学習・評価データ由来の
/// 予測ラベル）を出力しない（`.claude/rules/security.md`）。構築できるのは
/// 本モジュール（[`TrainOutcome::from_worker_stdout`]）だけ。
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ValidationPrediction {
    id: String,
    status: ValidationPredictionStatus,
    predicted_label: Option<String>,
}

impl ValidationPrediction {
    /// validation レコードの識別子（リクエストの `validation_inputs[].id` と対応）。
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// 予測の状態。
    #[must_use]
    pub fn status(&self) -> ValidationPredictionStatus {
        self.status
    }

    /// 予測ラベル（`status` が `Ok` のときだけ `Some`）。
    #[must_use]
    pub fn predicted_label(&self) -> Option<&str> {
        self.predicted_label.as_deref()
    }
}

impl std::fmt::Debug for ValidationPrediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidationPrediction")
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

/// [`TrainOutcome::Ok`] の中身（成果物の配置先ディレクトリ・成果物記録・
/// 任意の validation 予測列）。
///
/// フィールドは非公開。この型を構築できるのは本モジュール内
/// （[`TrainOutcome::from_worker_stdout`]）だけで、別 crate は読み取り専用
/// アクセサしか使えない。`Debug` は予測列の件数のみを出力し、内容
/// （予測ラベル）を出力しない。
#[derive(Clone, PartialEq)]
pub struct SuccessOutcome {
    artifact_dir: String,
    artifact: Box<ArtifactRecord>,
    validation_predictions: Option<Vec<ValidationPrediction>>,
}

impl std::fmt::Debug for SuccessOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SuccessOutcome")
            .field("artifact_dir", &self.artifact_dir)
            .field("artifact", &self.artifact)
            .field(
                "validation_prediction_count",
                &self.validation_predictions.as_ref().map(Vec::len),
            )
            .finish()
    }
}

impl SuccessOutcome {
    /// 検証済みの成果物配置先ディレクトリ（`request` の `root`／`out_dir`
    /// 配下に閉じ込められていることを `from_worker_stdout` が保証済み）。
    #[must_use]
    pub fn artifact_dir(&self) -> &str {
        &self.artifact_dir
    }

    /// 検証済みの成果物記録。
    #[must_use]
    pub fn artifact(&self) -> &ArtifactRecord {
        &self.artifact
    }

    /// 学習ジョブ内で採点した validation の予測列。リクエストが
    /// `validation_inputs` を持っていた場合だけ `Some`（件数・`id` の順序が入力と
    /// 一致するかは呼び出し元〔`crate::search`〕が照合する。入力より多くても
    /// ここでは拒否しない）。
    #[must_use]
    pub fn validation_predictions(&self) -> Option<&[ValidationPrediction]> {
        self.validation_predictions.as_deref()
    }
}

/// `from_worker_stdout` の内部専用中間表現。`status` と各フィールドの組み
/// 合わせはここでは検査せず、`TrainOutcome::from_worker_stdout` 側で
/// 明示的に判定する（serde の internally-tagged enum + `deny_unknown_fields`
/// の相互作用に頼らないため。issue #177 実装計画）。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOutcome {
    status: String,
    // `Option<Option<T>>`（キーの有無と `null` を区別する double-Option。
    // [`deserialize_present`] 参照）: `None` はキー欠落、`Some(None)` は
    // `"key":null` が明示された場合。`TrainOutcome::from_worker_stdout` は
    // どちらも「値が確定していない」として拒否するが、フィールドの組み合わせ
    // 検査自体は「キーが無いこと」だけを許可し、「値が null であること」は
    // 許可しない（REQ-21・REQ-39・P1。codex 指摘 PR #220）。
    #[serde(default, deserialize_with = "deserialize_present")]
    artifact_dir: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    artifact: Option<Option<RawArtifact>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    code: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    message: Option<Option<String>>,
    /// `status:"ok"` のとき、リクエストが `validation_inputs` を持つ場合に
    /// 限り現れる。`null` は拒否する（キー欠落と区別する double-Option）。
    #[serde(default, deserialize_with = "deserialize_present")]
    validation_predictions: Option<Option<Vec<RawValidationPrediction>>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawValidationPrediction {
    id: String,
    status: ValidationPredictionStatus,
    predicted_label: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArtifact {
    kind: String,
    kind_version: u32,
    selector_version: String,
    // `#[serde(default)]` を付けない: 11 項目の成果物契約（REQ-21）で
    // `config` は必須フィールド。ワーカー出力に `config` が欠落した場合、
    // 空オブジェクトへ暗黙補完すると契約を満たさない出力を成功として
    // 受理してしまう（REQ-39「完全性と版」・P1。codex review PR #220）。
    // 欠落時は `serde_json::from_str` がここでエラーになり、呼び出し元は
    // `TrainResultError::NotJson` として拒否する（`kind`・`label_order` 等
    // 他の必須フィールドと同じ扱い）。
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

/// 絶対パス文字列を構成要素単位で正規化する（`os.path.normpath` 相当）。
/// `.` 構成要素を除去し、`..` 構成要素は直前の構成要素を取り除く（ルートを
/// 超える `..` は無視する）。連続する `/` は 1 つにまとめる。**symlink の解決
/// は行わない**（ファイルシステムへ触れない純粋な文字列操作。symlink 対策は
/// 学習ワーカー自身が `guard.py::confine` で多層防御として担う設計を変えない。
/// 本モジュール doc 参照）。
///
/// `guard.py::resolve_root` は `root` に `os.path.realpath` を適用してから
/// `artifact_dir` を組み立てるため、`root` が `.`・連続スラッシュを
/// 含む正当な絶対パスの場合（`..` は #256 以降 `TrainRequest` が拒否するが、
/// 本関数の `..` 処理は多層防御として残す）、正規化しない文字列比較では正常な結果まで
/// `runtime_error` にしてしまっていた（REQ-39・P1。codex review・cursor[bot]
/// 重複指摘。PR #220）。
///
/// 本関数自体は `..`／`.`／連続スラッシュの除去のみを行い、symlink の解決は
/// しない（ファイルシステムへ触れない純粋な文字列操作）。symlink を含む
/// `root` の解決は [`expected_artifact_dir`] が `std::fs::canonicalize` で
/// 別途行う（REQ-39・P1。codex 指摘 PR #220）。
fn normalize_absolute_path(path: &str) -> String {
    let mut components: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => continue,
            ".." => {
                components.pop();
            }
            _ => components.push(part),
        }
    }
    let mut normalized = String::from("/");
    normalized.push_str(&components.join("/"));
    normalized
}

/// `request.root()` を起点に `request.out_dir()` を連結し、[`normalize_absolute_path`]
/// で正規化した絶対パス文字列を組み立てる。`cli.py::run_worker_train` が
/// `artifact_dir` として出す `str(request.out_dir.display)`
/// （`guard.confine` の `display = root_handle.root_real.joinpath(*rel.parts)`。
/// `root_real` は `os.path.realpath(root)`）と一致させるため、`root` 部分は
/// [`canonicalize_root_best_effort`] で symlink 解決を試みる（`rel.parts`
/// 側は `..` を含まない構文検査済みの構成要素のみなので、`out_dir` 自体を
/// realpath する必要はない。`guard.py::confine` と同じ非対称性）。
/// `root`／`out_dir` はいずれも [`TrainRequest`] が構文検査済み（空・NUL・
/// 絶対/相対の取り違えを含まず、`..` 構成要素も拒否済み。#256）だが、`root` の
/// `.`・連続スラッシュは残りうる（[`TrainRequest`] のモジュール doc「経路の
/// 閉じ込めについて」参照）ため、結合後に [`normalize_absolute_path`] でも
/// 正規化する。
fn expected_artifact_dir(request: &TrainRequest) -> String {
    let mut joined = canonicalize_root_best_effort(request.root());
    for part in request.out_dir().split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        joined.push('/');
        joined.push_str(part);
    }
    normalize_absolute_path(&joined)
}

/// `root` の symlink 解決を試みる（`guard.py::resolve_root` の
/// `os.path.realpath(root)` に相当）。
///
/// `root` は学習リクエストの一部として呼び出し元（本 crate の利用者）が
/// 用意した値であり、ワーカーが返す `artifact_dir`（信頼しない外部入力）
/// とは異なる（ここで実ファイルシステムへ触れるのは呼び出し元が既に
/// 把握しているパスのみで、信頼しない入力からの経路トラバーサルにはならない）。
/// `root` が存在しない・権限がない等で `canonicalize` が失敗した場合は
/// 元の文字列のまま返す（symlink を含まない `root` や、実在しない `root`
/// を使うテスト・開発時のフィクスチャでは従来どおり文字列正規化のみで
/// 比較する。フォールバックは検証を緩めない: 実在する `root` の symlink を
/// 解決できない場合に限られ、その場合でも実際のワーカー出力との不一致は
/// 通常どおり `runtime_error` として拒否される）。
fn canonicalize_root_best_effort(root: &str) -> String {
    match std::fs::canonicalize(root) {
        Ok(resolved) => match resolved.into_os_string().into_string() {
            Ok(s) => s,
            Err(_) => root.to_string(),
        },
        Err(_) => root.to_string(),
    }
}

/// `validation_predictions` の検証（REQ-27・REQ-39）。`id` は非空・[`MAX_INPUT_ID_BYTES`] 以下、`status:"ok"`
/// は `predicted_label` が文字列、`abstain`／`error` は `null` でなければならない。
/// `id` の順序・件数の一致は検査しない（呼び出し元の `crate::search` が
/// 既存の record_id 照合で候補単位に処理する）。件数が入力より多い場合も
/// ここでは拒否しない（Cursor 指摘対応。issue #84 PR #238 レビュー: 結果全体の
/// 拒否は探索全体の失敗になる。件数は結果のバイト長上限〔リクエストごとに
/// 正確に計算〕で有界）。
fn parse_validation_predictions(
    items: Vec<RawValidationPrediction>,
) -> Result<Vec<ValidationPrediction>, TrainResultError> {
    let malformed = || TrainResultError::MalformedArtifact {
        field: "validation_predictions",
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if item.id.is_empty() || item.id.len() > MAX_INPUT_ID_BYTES {
            return Err(malformed());
        }
        match (item.status, item.predicted_label.as_ref()) {
            (ValidationPredictionStatus::Ok, Some(_))
            | (ValidationPredictionStatus::Abstain | ValidationPredictionStatus::Error, None) => {}
            _ => return Err(malformed()),
        }
        out.push(ValidationPrediction {
            id: item.id,
            status: item.status,
            predicted_label: item.predicted_label,
        });
    }
    Ok(out)
}

impl TrainOutcome {
    /// 学習ワーカーの標準出力（信頼しない外部入力）から結果を解析する。
    ///
    /// `supervisor.py`（子プロセスが自分で終了した場合の検査）と同じ規則:
    /// (1) バイト長を [`TrainRequest::max_result_bytes`]（既定 [`crate::limits::MAX_RESULT_BYTES`]。`validation_inputs` 付きはラベル・id から計算した値）と照合 → (2) UTF-8 として読める →
    /// (3) 空行を除いてちょうど 1 行 → (4) JSON として解析可能 →
    /// (5) `status`／各フィールドの組み合わせが妥当 → (6) `status:"ok"` の
    /// 場合に限り、`artifact_dir` が `request` の `root`／`out_dir` 配下に
    /// 閉じ込められていること、`kind`・`kind_version`・`label_order`・
    /// `max_bytes`・`candidate_label` が `request` と一致すること、`config`
    /// が「`kind` ごとの既定値（[`crate::kind_defaults`]）に `request` の
    /// `config` を上書きした実効 config」と完全一致することを検査する（REQ-39
    /// ガード層「経路の閉じ込め」「完全性と版」。ワーカーが依頼と異なる
    /// 種類・未対応の版・root 外のパス・別種類を名乗る成果物を返しても
    /// 成功扱いにしない）。いずれかを満たさない場合は `TrainResultError`
    /// （呼び出し元は `reason_code()=="runtime_error"`・
    /// `exit_code()==RuntimeError` として扱う。`crate::process::run_train`
    /// 〔issue #178〕がこの経路を子プロセスの実終了コードと突き合わせる）。
    ///
    /// `request` はこの標準出力を生成した学習ワーカーへ実際に渡したリクエスト
    /// でなければならない（呼び出し元の責務。本関数は同一性を検証しない）。
    pub fn from_worker_stdout(
        bytes: &[u8],
        request: &TrainRequest,
    ) -> Result<Self, TrainResultError> {
        let max_result_bytes = request.max_result_bytes();
        if bytes.len() > max_result_bytes {
            return Err(TrainResultError::TooLarge {
                size: bytes.len(),
                limit: max_result_bytes,
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
                line: e.line(),
                column: e.column(),
                category: e.classify(),
            })?;

        match raw.status.as_str() {
            "ok" => {
                // `artifact_dir`／`artifact` はキーがあり値も入っていること
                // （`Some(Some(_))`）、`code`／`message` はキー自体が無いこと
                // （`None`）を要求する。`"code":null` のようにキーは在るが値が
                // `null` の場合（`Some(None)`）は、キー欠落と区別してここで
                // 拒否する（REQ-21・REQ-39・P1。codex 指摘 PR #220）。
                let (Some(Some(artifact_dir)), Some(Some(raw_artifact)), None, None) =
                    (raw.artifact_dir, raw.artifact, raw.code, raw.message)
                else {
                    return Err(TrainResultError::MalformedOutcome);
                };
                // `validation_predictions` は、リクエストが `validation_inputs`
                // を持つときは必須、持たないときは存在してはならない
                // （REQ-27・REQ-39。要求していない採点結果を受理しない・
                // 要求した採点結果の欠落を成功扱いにしない。fail-closed）。
                let validation_predictions =
                    match (request.validation_inputs(), raw.validation_predictions) {
                        (None, None) => None,
                        (Some(_), Some(Some(items))) => Some(parse_validation_predictions(items)?),
                        _ => return Err(TrainResultError::MalformedOutcome),
                    };
                // 経路の閉じ込め（REQ-39・P0）: 空文字・絶対パスの取り違え・
                // `..` を含む値はもちろん、`request` の `root`／`out_dir` と
                // 無関係な絶対パスもここで拒否する。
                if artifact_dir.is_empty()
                    || artifact_dir.contains('\0')
                    || artifact_dir != expected_artifact_dir(request)
                {
                    return Err(TrainResultError::MalformedArtifact {
                        field: "artifact_dir",
                    });
                }
                // 依頼内容との一致（REQ-39・P1）: `kind` ごとの許可版一覧は
                // 本 crate では確定していない（PR #220）ため、ここでは
                // 「依頼した内容がそのまま返ってきたか」だけを検査する。
                if raw_artifact.kind != request.kind() {
                    return Err(TrainResultError::ArtifactMismatch { field: "kind" });
                }
                if raw_artifact.kind_version != request.kind_version() {
                    return Err(TrainResultError::ArtifactMismatch {
                        field: "kind_version",
                    });
                }
                // `config` は完全一致で検査する（REQ-19・REQ-21・REQ-39・P1。
                // codex 指摘 PR #220「成果物の追加 config 値を検証せず成功
                // 扱いにしている」）: `kinds/c1.py`・`kinds/c3.py::train` は
                // `{**DEFAULT_CONFIG, **request.config}` で `kind` ごとの既定値を
                // 補完した実効 config を `trained.config` として返し、
                // `cli.py::run_worker_train` はその値を成果物へ記録する。
                // `kind` ごとの既定値そのものは学習ワーカー側が正本（層の境界。
                // `.claude/rules/dependency-policy.md`）だが、その値を共有
                // fixture（`fixtures/train_contract/kind_defaults.json`。
                // `trainer/tests/test_kind_defaults_fixture.py` が Python 側の
                // `DEFAULT_CONFIG` との一致を機械照合する）経由で取り込み、
                // 「defaults(kind) に request.config を上書きした実効 config」
                // を Rust 側でも独立に算出して完全一致を要求する。これにより
                // 未知のキーの追加・既定値の書き換え・明示値の書き換えを
                // すべて拒否できる（旧 `config_matches_explicit_keys` の
                // 部分一致では、`request.config` が空の場合に成果物の
                // `config` を無検査で受理してしまっていた）。
                let Some(expected_config) =
                    crate::kind_defaults::effective_config(request.kind(), request.config())
                else {
                    return Err(TrainResultError::UnsupportedKindForConfigDefaults);
                };
                if raw_artifact.config != expected_config {
                    return Err(TrainResultError::ArtifactMismatch { field: "config" });
                }
                if raw_artifact.max_bytes != request.max_bytes() {
                    return Err(TrainResultError::ArtifactMismatch { field: "max_bytes" });
                }
                let label_order = LabelOrder::new(raw_artifact.label_order).map_err(|_| {
                    TrainResultError::MalformedArtifact {
                        field: "label_order",
                    }
                })?;
                if label_order.as_slice() != request.label_order().as_slice() {
                    return Err(TrainResultError::ArtifactMismatch {
                        field: "label_order",
                    });
                }
                if !(crate::limits::MIN_MAX_BYTES..=crate::limits::MAX_MAX_BYTES)
                    .contains(&raw_artifact.max_bytes)
                {
                    return Err(TrainResultError::MalformedArtifact { field: "max_bytes" });
                }
                if raw_artifact.onnx_file != "model.onnx" {
                    return Err(TrainResultError::MalformedArtifact { field: "onnx_file" });
                }
                // 完全性と版（REQ-39・P1）: `selector_version` は対応版の許可
                // リストで検証する。空文字列・未対応版の成功結果を検証済みと
                // して受理しない（codex review PR #220）。
                if !crate::limits::ALLOWED_SELECTOR_VERSIONS
                    .contains(&raw_artifact.selector_version.as_str())
                {
                    return Err(TrainResultError::MalformedArtifact {
                        field: "selector_version",
                    });
                }
                // 依頼内容との一致（REQ-21・REQ-39・P1。codex 指摘 PR #220）:
                // `cli.py::run_worker_train` は `candidate_label=request.kind`
                // を記録する（`artifact.py` モジュール doc「複数候補からの
                // 選定は行わないため `candidate_label` は `kind` と同じ」）。
                // 非空性だけの検査では、依頼と異なる種類を名乗る成果物まで
                // 成功扱いにしてしまう。
                if raw_artifact.candidate_label != request.kind() {
                    return Err(TrainResultError::ArtifactMismatch {
                        field: "candidate_label",
                    });
                }
                if !is_syntactically_valid_created_utc(&raw_artifact.created_utc) {
                    return Err(TrainResultError::MalformedArtifact {
                        field: "created_utc",
                    });
                }
                let onnx_sha256 = OnnxSha256::parse(raw_artifact.onnx_sha256)?;
                Ok(TrainOutcome::Ok(SuccessOutcome {
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
                    validation_predictions,
                }))
            }
            "error" => {
                // 対称的に、`code`／`message` はキーがあり値も入っていること、
                // `artifact_dir`／`artifact` はキー自体が無いことを要求する。
                let (None, None, Some(Some(code)), Some(Some(message)), None) = (
                    raw.artifact_dir,
                    raw.artifact,
                    raw.code,
                    raw.message,
                    raw.validation_predictions,
                ) else {
                    return Err(TrainResultError::MalformedOutcome);
                };
                Ok(TrainOutcome::Error(WorkerFailure::parse(code, message)?))
            }
            _ => Err(TrainResultError::MalformedOutcome),
        }
    }

    /// この結果に対応する終了コード（REQ-21）。`Ok` は `ExitCode::Ok`
    /// （プロセスの終了コード 0）、`Error` はワーカーの `code` に対応する
    /// 終了コード（[`FailureCode::exit_code`]）。`crates/train/src/process.rs`
    /// が、ワーカーの実プロセス終了コードとこの値の一致を検証する
    /// （REQ-39「完全性と版」・#178）。
    #[must_use]
    pub fn exit_code(&self) -> ExitCode {
        match self {
            TrainOutcome::Ok(_) => ExitCode::Ok,
            TrainOutcome::Error(failure) => failure.failure_code().exit_code(),
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
            TrainOutcome::Ok(success) => {
                let entries = if success.validation_predictions().is_some() {
                    4
                } else {
                    3
                };
                let mut map = serializer.serialize_map(Some(entries))?;
                map.serialize_entry("status", "ok")?;
                map.serialize_entry("artifact_dir", success.artifact_dir())?;
                map.serialize_entry("artifact", success.artifact())?;
                if let Some(predictions) = success.validation_predictions() {
                    map.serialize_entry("validation_predictions", predictions)?;
                }
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
    use crate::limits::MAX_RESULT_BYTES;
    use crate::request::{Device, TrainRequestParams};

    /// `VALID_OK_JSON` に対応するリクエスト（`root`＋`out_dir` の結合が
    /// `artifact_dir` と一致する。本モジュール doc の `expected_artifact_dir`
    /// 参照）。
    fn test_request() -> TrainRequest {
        TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: "/fandhe-edge-fixture-root".to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: None,
            rss_limit_bytes: None,
        })
        .expect("test request params must be valid")
    }

    // `config` は `test_request()` の `config`（空オブジェクト）に対応する
    // 実効 config、すなわち `c3.DEFAULT_CONFIG` そのもの
    // （`fixtures/train_contract/kind_defaults.json` の `c3` と同じ値。
    // REQ-19・REQ-21・REQ-39・P1。codex 指摘 PR #220「成果物の追加 config
    // 値を検証せず成功扱いにしている」対応で完全一致検査に変更したため、
    // 空オブジェクトのままでは拒否されてしまう）。
    const VALID_OK_JSON: &str = r#"{"status":"ok","artifact_dir":"/fandhe-edge-fixture-root/out","artifact":{"kind":"c3","kind_version":1,"selector_version":"0.1","config":{"lr":0.001,"weight_decay":0.0001,"epochs":40,"batch_size":64,"emb":64,"filters":128,"widths":[3,5,7],"dropout":0.3},"label_order":["a","b"],"output_type":"choice","max_bytes":512,"onnx_file":"model.onnx","onnx_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","created_utc":"2026-09-28T00:00:00Z","candidate_label":"c3"}}"#;

    /// `VALID_OK_JSON` 内の `config` フィールド全体（キーと値）。他のテストが
    /// `config` だけを差し替える際、旧・部分一致検査の時代に使っていた
    /// `"config":{}"` という置換対象がもう存在しない（完全一致検査に伴い
    /// `test_request()`〔空の `config`〕に対応する実効 config を書いたため）
    /// ので、この定数を置換対象として使う。
    const VALID_OK_CONFIG_FRAGMENT: &str = r#""config":{"lr":0.001,"weight_decay":0.0001,"epochs":40,"batch_size":64,"emb":64,"filters":128,"widths":[3,5,7],"dropout":0.3}"#;

    #[test]
    fn req21_parses_valid_ok_outcome() {
        let outcome = TrainOutcome::from_worker_stdout(VALID_OK_JSON.as_bytes(), &test_request())
            .expect("valid outcome");
        match outcome {
            TrainOutcome::Ok(success) => {
                assert_eq!(success.artifact_dir(), "/fandhe-edge-fixture-root/out");
                assert_eq!(success.artifact().kind(), "c3");
                assert_eq!(
                    success.artifact().label_order().as_slice(),
                    &["a".to_string(), "b".to_string()]
                );
            }
            TrainOutcome::Error(_) => panic!("expected Ok"),
        }
    }

    #[test]
    fn req21_parses_valid_error_outcome() {
        let json = r#"{"status":"error","code":"invalid_request","message":"file not readable: FileNotFoundError"}"#;
        let outcome = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request())
            .expect("valid outcome");
        match outcome {
            TrainOutcome::Error(failure) => {
                assert_eq!(failure.code(), "invalid_request");
                assert_eq!(failure.message(), "file not readable: FileNotFoundError");
            }
            TrainOutcome::Ok(_) => panic!("expected Error"),
        }
    }

    /// REQ-21・REQ-39・P1（codex 指摘 PR #220）: `code` は許可リスト
    /// （[`FailureCode`]）に一致する値のみ受理する。ワーカーが返しえない
    /// コード（`"[a-z_]+"` には一致するが許可リスト外。例: `"success"`）は
    /// `InvalidFailureCode` として拒否し、検証済みとして受理しない。
    #[test]
    fn req39_rejects_unknown_failure_code_not_in_allowlist() {
        let json = r#"{"status":"error","code":"success","message":"m"}"#;
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(err, TrainResultError::InvalidFailureCode));
        assert_eq!(err.reason_code(), "runtime_error");
    }

    /// [`FailureCode::all`] の全件が [`FailureCode::as_str`] →
    /// [`WorkerFailure::parse`] を往復できる（許可リストとパース規則の整合）。
    #[test]
    fn req21_all_known_failure_codes_round_trip() {
        for code in FailureCode::all() {
            let json = format!(
                r#"{{"status":"error","code":"{}","message":"m"}}"#,
                code.as_str()
            );
            let outcome = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request())
                .expect("known failure code must be accepted");
            match outcome {
                TrainOutcome::Error(failure) => assert_eq!(failure.code(), code.as_str()),
                TrainOutcome::Ok(_) => panic!("expected Error"),
            }
        }
    }

    /// REQ-39: 2 行出力は「壊れたワーカー出力」として `runtime_error` になる。
    #[test]
    fn req39_rejects_two_line_output() {
        let json = format!("{VALID_OK_JSON}\n{VALID_OK_JSON}");
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
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
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert_eq!(err.reason_code(), "runtime_error");
    }

    /// REQ-39: `onnx_sha256` の形式不正（短すぎる・大文字を含む）は拒否する。
    #[test]
    fn req39_rejects_malformed_onnx_sha256() {
        let json = VALID_OK_JSON.replace(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
            "DEADBEEF",
        );
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
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
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
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
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(err, TrainResultError::MalformedOutcome));
    }

    /// REQ-39: サイズ超過（`MAX_RESULT_BYTES` 超）は `limit_exceeded` 相当
    /// ではなく `runtime_error`（壊れたワーカー出力として扱う。
    /// `WorkerFailure` を返す `error` ステータスとは別の、監視側の検査）。
    #[test]
    fn req39_rejects_output_over_size_limit() {
        let huge = "a".repeat(MAX_RESULT_BYTES + 1);
        let json = format!(r#"{{"status":"error","code":"x","message":"{huge}"}}"#);
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(err, TrainResultError::TooLarge { .. }));
        assert_eq!(err.reason_code(), "runtime_error");
        assert_eq!(
            err.exit_code(),
            fandhe_edge_core::exitcode::ExitCode::RuntimeError
        );
    }

    /// REQ-39・P0（codex review PR #220）: `artifact_dir` が `request` の
    /// `root`／`out_dir` と無関係な絶対パスの場合は拒否する。
    #[test]
    fn req39_rejects_artifact_dir_outside_request_root() {
        let json =
            VALID_OK_JSON.replace("/fandhe-edge-fixture-root/out", "/somewhere-else/evil/out");
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::MalformedArtifact {
                field: "artifact_dir"
            }
        ));
    }

    /// REQ-39・P0: `artifact_dir` が空文字列の場合は拒否する。
    #[test]
    fn req39_rejects_empty_artifact_dir() {
        let json = VALID_OK_JSON.replace(r#""/fandhe-edge-fixture-root/out""#, r#""""#);
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::MalformedArtifact {
                field: "artifact_dir"
            }
        ));
    }

    /// REQ-39・P0: `artifact_dir` が `root` 自体（`out_dir` を含まない `..`
    /// 相当のより浅い階層）の場合も拒否する（`request.out_dir()` と一致しない）。
    #[test]
    fn req39_rejects_artifact_dir_escaping_via_dot_dot() {
        let json = VALID_OK_JSON.replace(
            "/fandhe-edge-fixture-root/out",
            "/fandhe-edge-fixture-root/out/../../etc",
        );
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::MalformedArtifact {
                field: "artifact_dir"
            }
        ));
    }

    /// REQ-39・P1（codex review PR #220）: `kind` が依頼内容と異なる場合は
    /// 成功扱いにしない。
    #[test]
    fn req39_rejects_artifact_kind_mismatching_request() {
        let json = VALID_OK_JSON.replace(r#""kind":"c3""#, r#""kind":"c1""#);
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::ArtifactMismatch { field: "kind" }
        ));
    }

    /// REQ-39・P1: `kind_version` が依頼内容と異なる（未対応の版を返す）場合
    /// は成功扱いにしない。
    #[test]
    fn req39_rejects_artifact_kind_version_mismatching_request() {
        let json = VALID_OK_JSON.replace(r#""kind_version":1"#, r#""kind_version":2"#);
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::ArtifactMismatch {
                field: "kind_version"
            }
        ));
    }

    /// REQ-39・P1: `label_order` が依頼内容と異なる場合は成功扱いにしない。
    #[test]
    fn req39_rejects_artifact_label_order_mismatching_request() {
        let json =
            VALID_OK_JSON.replace(r#""label_order":["a","b"]"#, r#""label_order":["b","a"]"#);
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::ArtifactMismatch {
                field: "label_order"
            }
        ));
    }

    /// REQ-39・P1: `max_bytes` が依頼内容と異なる場合は成功扱いにしない。
    #[test]
    fn req39_rejects_artifact_max_bytes_mismatching_request() {
        let json = VALID_OK_JSON.replace(r#""max_bytes":512"#, r#""max_bytes":1024"#);
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::ArtifactMismatch { field: "max_bytes" }
        ));
    }

    /// `test_request()` と同じ内容だが `config` を差し替えたリクエストを作る
    /// （`config` 部分一致検査の検証用）。
    fn test_request_with_config(
        config: serde_json::Map<String, serde_json::Value>,
    ) -> TrainRequest {
        TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config,
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: "/fandhe-edge-fixture-root".to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: None,
            rss_limit_bytes: None,
        })
        .expect("test request params must be valid")
    }

    /// REQ-19・REQ-21・REQ-39・P1（codex 指摘 PR #220）: リクエストが明示した
    /// `config` のキーが成果物で書き換わっている場合は成功扱いにしない
    /// （`kind` ごとの既定値補完の陰に、依頼したハイパーパラメータの書き換え
    /// を隠せてはいけない）。
    #[test]
    fn req39_rejects_artifact_config_mismatching_explicit_request_key() {
        let mut requested = serde_json::Map::new();
        requested.insert("epochs".to_string(), serde_json::json!(2));
        let request = test_request_with_config(requested);

        // 明示値（`epochs`）の書き換えだけを検出することを確認するため、
        // それ以外のキーは既定値のまま（`c3_effective_config_json` が
        // `epochs` の値だけを 99 に差し替える）にする。
        let config_json = c3_effective_config_json(&[("epochs", serde_json::json!(99))]);
        let json = VALID_OK_JSON.replace(
            VALID_OK_CONFIG_FRAGMENT,
            &format!(r#""config":{config_json}"#),
        );
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &request).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::ArtifactMismatch { field: "config" }
        ));
    }

    /// c3 の `DEFAULT_CONFIG`（`fixtures/train_contract/kind_defaults.json`・
    /// `trainer/src/fandhe_edge_trainer/kinds/c3.py` と一致する具体値）を
    /// JSON の `config` フラグメント文字列として返す。`overrides` の各キーで
    /// 値を上書きする（`{**DEFAULT_CONFIG, **request.config}` を模す）。
    fn c3_effective_config_json(overrides: &[(&str, serde_json::Value)]) -> String {
        let mut config = serde_json::json!({
            "lr": 0.001,
            "weight_decay": 0.0001,
            "epochs": 40,
            "batch_size": 64,
            "emb": 64,
            "filters": 128,
            "widths": [3, 5, 7],
            "dropout": 0.3,
        });
        let obj = config.as_object_mut().expect("object");
        for (key, value) in overrides {
            obj.insert((*key).to_string(), value.clone());
        }
        serde_json::to_string(&config).expect("serialize config fragment")
    }

    /// REQ-18・REQ-19・REQ-21・REQ-39・P1（codex 指摘 PR #220）: `config` を
    /// 省略した・一部だけ指定したリクエストでは、`kinds/c1.py`・
    /// `kinds/c3.py::train` が `kind` ごとの既定値で補完した実効 config を
    /// 成果物へ記録する。「defaults(kind) に request.config を上書きした
    /// 実効 config」と完全一致する成果物は受理する。
    #[test]
    fn req39_accepts_artifact_config_with_kind_default_filled_keys() {
        let mut requested = serde_json::Map::new();
        requested.insert("epochs".to_string(), serde_json::json!(2));
        let request = test_request_with_config(requested);

        let config_json = c3_effective_config_json(&[("epochs", serde_json::json!(2))]);
        let json = VALID_OK_JSON.replace(
            VALID_OK_CONFIG_FRAGMENT,
            &format!(r#""config":{config_json}"#),
        );
        let outcome = TrainOutcome::from_worker_stdout(json.as_bytes(), &request)
            .expect("full effective config must be accepted");
        match outcome {
            TrainOutcome::Ok(success) => {
                assert_eq!(
                    success.artifact().config().get("lr"),
                    Some(&serde_json::json!(0.001))
                );
                assert_eq!(
                    success.artifact().config().get("epochs"),
                    Some(&serde_json::json!(2))
                );
            }
            TrainOutcome::Error(_) => panic!("expected Ok"),
        }
    }

    /// REQ-19・REQ-21・REQ-39・P1: `config` を省略したリクエスト（空
    /// オブジェクト）でも、成果物の `config` が `kind` の既定値そのものと
    /// 完全一致すれば受理する。
    #[test]
    fn req39_accepts_artifact_config_matching_defaults_when_request_omits_config() {
        let request = test_request_with_config(serde_json::Map::new());
        let config_json = c3_effective_config_json(&[]);
        let json = VALID_OK_JSON.replace(
            VALID_OK_CONFIG_FRAGMENT,
            &format!(r#""config":{config_json}"#),
        );
        let outcome = TrainOutcome::from_worker_stdout(json.as_bytes(), &request)
            .expect("defaults-only config must be accepted when request omits config");
        assert!(matches!(outcome, TrainOutcome::Ok(_)));
    }

    /// REQ-19・REQ-21・REQ-39・P1: `request.config` が全キーを明示指定した
    /// （既定値と異なる値を含む）場合でも、成果物の `config` がその実効
    /// config と完全一致すれば受理する（accept: 全指定）。
    #[test]
    fn req39_accepts_artifact_config_when_request_specifies_all_keys() {
        let overrides: &[(&str, serde_json::Value)] = &[
            ("lr", serde_json::json!(0.01)),
            ("weight_decay", serde_json::json!(0.001)),
            ("epochs", serde_json::json!(5)),
            ("batch_size", serde_json::json!(32)),
            ("emb", serde_json::json!(16)),
            ("filters", serde_json::json!(32)),
            ("widths", serde_json::json!([3, 5])),
            ("dropout", serde_json::json!(0.1)),
        ];
        let mut requested = serde_json::Map::new();
        for (key, value) in overrides {
            requested.insert((*key).to_string(), value.clone());
        }
        let request = test_request_with_config(requested);

        let config_json = c3_effective_config_json(overrides);
        let json = VALID_OK_JSON.replace(
            VALID_OK_CONFIG_FRAGMENT,
            &format!(r#""config":{config_json}"#),
        );
        let outcome = TrainOutcome::from_worker_stdout(json.as_bytes(), &request)
            .expect("fully explicit config must be accepted when it matches the request");
        match outcome {
            TrainOutcome::Ok(success) => {
                assert_eq!(
                    success.artifact().config().get("epochs"),
                    Some(&serde_json::json!(5))
                );
                assert_eq!(
                    success.artifact().config().get("widths"),
                    Some(&serde_json::json!([3, 5]))
                );
            }
            TrainOutcome::Error(_) => panic!("expected Ok"),
        }
    }

    /// REQ-19・REQ-21・REQ-39・P1（codex 指摘 PR #220 P1「成果物の追加
    /// config 値を検証せず成功扱いにしている」）: `request.config` が空でも、
    /// 成果物の `config` が `kind` の既定値と異なれば拒否する（空リクエスト
    /// に対して任意の `config` を成功扱いにしない）。
    #[test]
    fn req39_rejects_artifact_config_arbitrary_when_request_omits_config() {
        let request = test_request_with_config(serde_json::Map::new());
        let json = VALID_OK_JSON.replace(VALID_OK_CONFIG_FRAGMENT, r#""config":{"anything":1}"#);
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &request).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::ArtifactMismatch { field: "config" }
        ));
    }

    /// REQ-19・REQ-21・REQ-39・P1: 既定値・明示値をすべて満たしていても、
    /// 未知のキーが 1 つ増えていれば拒否する。
    #[test]
    fn req39_rejects_artifact_config_with_unknown_extra_key() {
        let mut requested = serde_json::Map::new();
        requested.insert("epochs".to_string(), serde_json::json!(2));
        let request = test_request_with_config(requested);

        let config_json = c3_effective_config_json(&[
            ("epochs", serde_json::json!(2)),
            ("extra", serde_json::json!(1)),
        ]);
        let json = VALID_OK_JSON.replace(
            VALID_OK_CONFIG_FRAGMENT,
            &format!(r#""config":{config_json}"#),
        );
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &request).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::ArtifactMismatch { field: "config" }
        ));
    }

    /// REQ-19・REQ-21・REQ-39・P1: リクエストが明示していないキー（既定値
    /// 側）が書き換えられている場合も拒否する（旧・部分一致では検出できな
    /// かった。codex 指摘の中心ケース）。
    #[test]
    fn req39_rejects_artifact_config_with_altered_default_value() {
        let mut requested = serde_json::Map::new();
        requested.insert("epochs".to_string(), serde_json::json!(2));
        let request = test_request_with_config(requested);

        // `lr` はリクエストが明示していない既定値キー。ここを書き換える。
        let config_json = c3_effective_config_json(&[
            ("epochs", serde_json::json!(2)),
            ("lr", serde_json::json!(999.0)),
        ]);
        let json = VALID_OK_JSON.replace(
            VALID_OK_CONFIG_FRAGMENT,
            &format!(r#""config":{config_json}"#),
        );
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &request).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::ArtifactMismatch { field: "config" }
        ));
    }

    /// REQ-19・REQ-21・REQ-39・P1: fixture に無い `kind` は実効 config を
    /// 算出できないため、成功結果でも拒否する（`kind`・`candidate_label` は
    /// リクエストと一致させ、`config` 検査だけに到達させる）。
    #[test]
    fn req39_rejects_success_outcome_with_kind_unknown_to_config_defaults_fixture() {
        let request = TrainRequest::new(TrainRequestParams {
            kind: "unknown-kind".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: "/fandhe-edge-fixture-root".to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: None,
            rss_limit_bytes: None,
        })
        .expect("test request params must be valid");
        let json = VALID_OK_JSON
            .replace(r#""kind":"c3""#, r#""kind":"unknown-kind""#)
            .replace(
                r#""candidate_label":"c3""#,
                r#""candidate_label":"unknown-kind""#,
            );
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &request).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::UnsupportedKindForConfigDefaults
        ));
    }

    /// REQ-21・REQ-39・P1（codex 指摘 PR #220）: `candidate_label` が依頼した
    /// `kind` と異なる場合は成功扱いにしない
    /// （`cli.py::run_worker_train` は `candidate_label=request.kind` を記録する）。
    #[test]
    fn req39_rejects_artifact_candidate_label_mismatching_request_kind() {
        let json = VALID_OK_JSON.replace(r#""candidate_label":"c3""#, r#""candidate_label":"c1""#);
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(
            err,
            TrainResultError::ArtifactMismatch {
                field: "candidate_label"
            }
        ));
    }

    /// `test_request()` と同じ内容だが `root` を差し替えたリクエストを作る
    /// （正規化の検証用）。
    fn test_request_with_root(root: &str) -> TrainRequest {
        TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: root.to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: None,
            rss_limit_bytes: None,
        })
        .expect("test request params must be valid")
    }

    /// REQ-39・P1（codex review・cursor[bot] 重複指摘。PR #220）: `root` が
    /// `.`・連続スラッシュ・末尾スラッシュを含む正当な絶対パスでも、正規化後に一致する
    /// `artifact_dir` は拒否しない。
    #[test]
    fn req39_accepts_artifact_dir_when_root_needs_normalization() {
        let request = test_request_with_root("/fandhe-edge-fixture-root/./sub//");
        let json = VALID_OK_JSON.replace(
            "/fandhe-edge-fixture-root/out",
            "/fandhe-edge-fixture-root/sub/out",
        );
        let outcome = TrainOutcome::from_worker_stdout(json.as_bytes(), &request)
            .expect("normalized root must match worker's realpath-based artifact_dir");
        match outcome {
            TrainOutcome::Ok(success) => {
                assert_eq!(success.artifact_dir(), "/fandhe-edge-fixture-root/sub/out");
            }
            TrainOutcome::Error(_) => panic!("expected Ok"),
        }
    }

    /// REQ-39・P1（完全性と版。codex review PR #220）: `selector_version` が
    /// 対応版の許可リストに無い場合は成功扱いにしない（空文字列を含む）。
    #[test]
    fn req39_rejects_unsupported_selector_version() {
        for bad in ["", "0.2", "9.9"] {
            let json = VALID_OK_JSON.replace(
                r#""selector_version":"0.1""#,
                &format!(r#""selector_version":"{bad}""#),
            );
            let err =
                TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
            assert!(
                matches!(
                    err,
                    TrainResultError::MalformedArtifact {
                        field: "selector_version"
                    }
                ),
                "case: {bad:?}"
            );
        }
    }

    /// REQ-39・P1（codex 指摘 PR #220）: `root` の途中に symlink を含む正当な
    /// リクエストで、`root` を `os.path.realpath` した先を `artifact_dir` と
    /// して返すワーカー出力を拒否しない（symlink 解決なしの文字列正規化だけ
    /// では実在パスの相違により `runtime_error` を誤検出していた）。
    /// 実ファイルシステム上に symlink を作るため Unix 専用（本 crate の
    /// クロスプラットフォーム方針は `.claude/rules/coding-rust.md` 参照。
    /// 検証環境は Mac のみで、Windows 対応は対象外）。
    #[cfg(unix)]
    #[test]
    fn req39_accepts_artifact_dir_when_root_contains_symlink() {
        use std::os::unix::fs::symlink;

        let base = std::env::temp_dir().join(format!(
            "fandhe-edge-train-symlink-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let real_root = base.join("real-root");
        let link_root = base.join("link-root");
        std::fs::create_dir_all(&real_root).expect("create real root");
        // `base` 自体が symlink 経由（macOS の `TMPDIR` は `/var/...` が
        // `/private/var/...` への symlink）の場合があるため、`real_root` も
        // 事前に `canonicalize` してから `link_root` の期待値を組み立てる。
        // ワーカー（`os.path.realpath`）は経路上の全 symlink を解決するため、
        // `link-root` 側だけでなく比較対象の実体パスも完全に解決しておかないと
        // macOS で `/var` と `/private/var` の食い違いにより誤って
        // `runtime_error` になる（REQ-39・P1。CI #220 で実際に検出）。
        let real_root_canonical =
            std::fs::canonicalize(&real_root).expect("canonicalize real root");
        symlink(&real_root, &link_root).expect("create symlink root");

        let request = test_request_with_root(link_root.to_str().expect("utf-8 path"));
        let real_artifact_dir = real_root_canonical.join("out");
        let json = VALID_OK_JSON.replace(
            "/fandhe-edge-fixture-root/out",
            real_artifact_dir.to_str().expect("utf-8 path"),
        );

        let result = TrainOutcome::from_worker_stdout(json.as_bytes(), &request);
        std::fs::remove_dir_all(&base).expect("cleanup temp dirs");

        match result.expect("symlink-resolved root must match worker's realpath artifact_dir") {
            TrainOutcome::Ok(success) => {
                assert_eq!(
                    success.artifact_dir(),
                    real_artifact_dir.to_str().expect("utf-8 path")
                );
            }
            TrainOutcome::Error(_) => panic!("expected Ok"),
        }
    }

    /// REQ-21・REQ-39・P1（codex 指摘 PR #220）: 成功結果で `config` が欠落
    /// している場合、空オブジェクトへ補完せず拒否する（11 項目の成果物契約を
    /// 満たさない出力を成功扱いにしない）。
    #[test]
    fn req39_rejects_success_outcome_missing_config() {
        let json = VALID_OK_JSON.replace(&format!("{VALID_OK_CONFIG_FRAGMENT},"), "");
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(err, TrainResultError::NotJson { .. }));
    }

    /// REQ-21・REQ-39・P1（codex 指摘 PR #220「成功結果の余分な null フィール
    /// ドを拒否する」）: `status:"ok"` に `"code":null`／`"message":null` を
    /// 加えた出力を成功扱いにしない。`Option<String>` のままだとキー欠落と
    /// 区別できず誤って受理していた（[`deserialize_present`] 参照）。
    #[test]
    fn req39_rejects_success_outcome_with_null_code_or_message() {
        // 末尾の `}`（トップレベルオブジェクトの閉じ括弧）の直前へ追加のキーを
        // 挿入する。`str::replace('}', ...)` だと `"config":{}` 等ネストした
        // `}` にもマッチしてしまうため使わない。
        let without_trailing_brace = &VALID_OK_JSON[..VALID_OK_JSON.len() - 1];
        for extra in [r#","code":null"#, r#","message":null"#] {
            let json = format!("{without_trailing_brace}{extra}}}");
            let err =
                TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
            assert!(
                matches!(err, TrainResultError::MalformedOutcome),
                "extra: {extra:?}, err: {err:?}"
            );
        }
    }

    /// 上と対称: `status:"error"` に `"artifact_dir":null`／`"artifact":null`
    /// を加えた出力も成功結果同様に拒否する。
    #[test]
    fn req39_rejects_error_outcome_with_null_artifact_dir_or_artifact() {
        let base = r#"{"status":"error","code":"invalid_request","message":"m"}"#;
        let without_trailing_brace = &base[..base.len() - 1];
        for extra in [r#","artifact_dir":null"#, r#","artifact":null"#] {
            let json = format!("{without_trailing_brace}{extra}}}");
            let err =
                TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
            assert!(
                matches!(err, TrainResultError::MalformedOutcome),
                "extra: {extra:?}, err: {err:?}"
            );
        }
    }

    /// security.md「秘密情報の混入防止」・REQ-39: ワーカー出力に含まれる
    /// 未知フィールド名（秘密風の文字列を想定）が、`Display`／`Debug` の
    /// どちらにも現れない。`serde_json::Error::to_string()` を経由すると
    /// `unknown field \`secret_key_XYZ\`, expected ...` のように出力の
    /// キー名がそのまま出てしまうため、位置情報のみを保持する（PR #220
    /// レビュー指摘 P0）。
    #[test]
    fn req39_unknown_field_secret_does_not_leak_into_error_text() {
        let without_trailing_brace = &VALID_OK_JSON[..VALID_OK_JSON.len() - 1];
        let json = format!(r#"{without_trailing_brace},"secret_key_XYZ":"leak-me"}}"#);
        let err = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err();
        assert!(matches!(err, TrainResultError::NotJson { .. }));
        let display = err.to_string();
        let debug = format!("{err:?}");
        assert!(!display.contains("secret_key_XYZ"));
        assert!(!display.contains("leak-me"));
        assert!(!debug.contains("secret_key_XYZ"));
        assert!(!debug.contains("leak-me"));
    }

    /// REQ-18・TASK-18.1-1: `WorkerFailure::failure_code()` は `code()` が
    /// 表す文字列と同じ [`FailureCode`] を返す（`crates/train::time_allotment`
    /// が `limit_exceeded` を型で判定するために使う）。
    #[test]
    fn task18_1_1_failure_code_returns_typed_variant_for_limit_exceeded() {
        let json = r#"{"status":"error","code":"limit_exceeded","message":"training exceeded wall-clock budget of 2 seconds"}"#;
        let outcome = TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request())
            .expect("valid error outcome");
        match outcome {
            TrainOutcome::Error(failure) => {
                assert_eq!(failure.failure_code(), FailureCode::LimitExceeded);
            }
            TrainOutcome::Ok(_) => panic!("expected Error"),
        }
    }

    // ---- validation_predictions（REQ-18・REQ-27・REQ-39。issue #84 PR #238・選択肢 2）----

    fn request_with_validation(n: usize) -> TrainRequest {
        let inputs = (0..n)
            .map(|i| crate::request::ValidationInput::new(format!("r{i}"), format!("input {i}")))
            .collect();
        test_request()
            .with_validation_inputs(inputs)
            .expect("valid validation inputs")
    }

    fn ok_json_with_predictions(predictions: &str) -> String {
        let without_trailing_brace = &VALID_OK_JSON[..VALID_OK_JSON.len() - 1];
        format!(r#"{without_trailing_brace},"validation_predictions":{predictions}}}"#)
    }

    /// 要求どおりの予測列は受理され、内容が保たれる。`Debug` は予測ラベル
    /// （`LABEL_MARKER`）を出力せず件数だけを示し、直列化は往復できる。
    #[test]
    fn req27_parses_validation_predictions_and_debug_hides_labels() {
        let request = request_with_validation(3);
        let json = ok_json_with_predictions(
            r#"[{"id":"r0","status":"ok","predicted_label":"a"},{"id":"r1","status":"abstain","predicted_label":null},{"id":"r2","status":"error","predicted_label":null}]"#,
        );
        let outcome =
            TrainOutcome::from_worker_stdout(json.as_bytes(), &request).expect("valid outcome");
        let TrainOutcome::Ok(success) = &outcome else {
            panic!("expected Ok");
        };
        let predictions = success.validation_predictions().expect("present");
        assert_eq!(predictions.len(), 3);
        assert_eq!(predictions[0].id(), "r0");
        assert_eq!(predictions[0].status(), ValidationPredictionStatus::Ok);
        assert_eq!(predictions[0].predicted_label(), Some("a"));
        assert_eq!(predictions[1].status(), ValidationPredictionStatus::Abstain);
        assert_eq!(predictions[2].predicted_label(), None);
        // `config` のキー順は正規化されるため、文字列ではなく値で比較する。
        let round_trip = serde_json::to_value(&outcome).expect("serialize");
        let expected: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(round_trip, expected);

        let marker_json = ok_json_with_predictions(
            r#"[{"id":"r0","status":"ok","predicted_label":"LABEL_MARKER"}]"#,
        );
        let marker =
            TrainOutcome::from_worker_stdout(marker_json.as_bytes(), &request_with_validation(1))
                .expect("valid outcome");
        let debug = format!("{marker:?}");
        assert!(!debug.contains("LABEL_MARKER"), "{debug}");
        assert!(debug.contains("validation_prediction_count"));
    }

    /// 要求したのに無い・要求していないのに有る・`null` は拒否する（fail-closed）。
    #[test]
    fn req39_validation_predictions_presence_must_match_request() {
        let requested = request_with_validation(1);
        assert_eq!(
            TrainOutcome::from_worker_stdout(VALID_OK_JSON.as_bytes(), &requested).unwrap_err(),
            TrainResultError::MalformedOutcome
        );
        let json = ok_json_with_predictions(r#"[{"id":"r0","status":"ok","predicted_label":"a"}]"#);
        assert_eq!(
            TrainOutcome::from_worker_stdout(json.as_bytes(), &test_request()).unwrap_err(),
            TrainResultError::MalformedOutcome
        );
        let null_json = ok_json_with_predictions("null");
        assert_eq!(
            TrainOutcome::from_worker_stdout(null_json.as_bytes(), &requested).unwrap_err(),
            TrainResultError::MalformedOutcome
        );
    }

    /// 件数超過・空 id・status とラベルの不整合・未知フィールドは拒否する。
    #[test]
    fn req39_validation_predictions_shape_rejections() {
        let malformed = TrainResultError::MalformedArtifact {
            field: "validation_predictions",
        };
        let cases = [
            r#"[{"id":"","status":"ok","predicted_label":"a"}]"#,
            r#"[{"id":"r0","status":"ok","predicted_label":null}]"#,
            r#"[{"id":"r0","status":"error","predicted_label":"a"}]"#,
        ];
        for predictions in cases {
            let json = ok_json_with_predictions(predictions);
            assert_eq!(
                TrainOutcome::from_worker_stdout(json.as_bytes(), &request_with_validation(1))
                    .unwrap_err(),
                malformed,
                "{predictions}"
            );
        }
        let unknown = ok_json_with_predictions(
            r#"[{"id":"r0","status":"ok","predicted_label":"a","scores":{"a":1.0}}]"#,
        );
        assert!(matches!(
            TrainOutcome::from_worker_stdout(unknown.as_bytes(), &request_with_validation(1)),
            Err(TrainResultError::NotJson { .. })
        ));
        let bad_status =
            ok_json_with_predictions(r#"[{"id":"r0","status":"nope","predicted_label":null}]"#);
        assert!(matches!(
            TrainOutcome::from_worker_stdout(bad_status.as_bytes(), &request_with_validation(1)),
            Err(TrainResultError::NotJson { .. })
        ));
    }

    /// 結果の読み込み上限は、`validation_inputs` 付きのときだけ緩む。
    #[test]
    fn req39_result_size_limit_depends_on_validation_inputs() {
        let over_default = " ".repeat(MAX_RESULT_BYTES + 1);
        let plain_err =
            TrainOutcome::from_worker_stdout(over_default.as_bytes(), &test_request()).unwrap_err();
        assert_eq!(
            plain_err,
            TrainResultError::TooLarge {
                size: MAX_RESULT_BYTES + 1,
                limit: MAX_RESULT_BYTES
            }
        );
        // 検証付きでは同じサイズは TooLarge にならない（空白のみ → 行数エラー）。
        assert!(matches!(
            TrainOutcome::from_worker_stdout(over_default.as_bytes(), &request_with_validation(1)),
            Err(TrainResultError::NotExactlyOneLine { .. })
        ));
        let over_validation = " ".repeat(crate::limits::MAX_RESULT_BYTES_WITH_VALIDATION + 1);
        assert!(matches!(
            TrainOutcome::from_worker_stdout(
                over_validation.as_bytes(),
                &request_with_validation(1)
            ),
            Err(TrainResultError::TooLarge { .. })
        ));
    }

    /// (Cursor 指摘対応・issue #84 PR #238 レビュー) 予測列が入力より多くても、
    /// 結果の解析では拒否しない（結果全体の拒否は探索全体の失敗になるため）。
    /// 件数の不一致は呼び出し元（`crate::search`）が候補単位の `ScoringFailed`
    /// にする。
    #[test]
    fn req27_extra_prediction_rows_are_parsed_and_left_to_the_caller() {
        let json = ok_json_with_predictions(
            r#"[{"id":"r0","status":"ok","predicted_label":"a"},{"id":"r1","status":"ok","predicted_label":"a"}]"#,
        );
        let outcome =
            TrainOutcome::from_worker_stdout(json.as_bytes(), &request_with_validation(1))
                .expect("extra rows are not a malformed result");
        let TrainOutcome::Ok(success) = outcome else {
            panic!("expected Ok");
        };
        assert_eq!(success.validation_predictions().map(<[_]>::len), Some(2));
    }
}
