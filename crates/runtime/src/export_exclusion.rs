//! 書き出し不能・予測ずれの構成を配布候補から外し、理由を記録する（REQ-32 異常系・TASK-32.2・#114）。
//!
//! # 役割と呼び出し文脈
//!
//! 成果物・推論 SDK 層の一部。CLI の `package`・`select` 工程から呼ぶ想定の部品で、CLI からは呼んでいない。
//! 現行の CLI は自作バックエンドだけを対象に ONNX の読み込み可否を検証し、失敗は除外でなく終了コードで
//! 止める（fail-closed）ため、除外対象の構成が無い（2026-10-12 オーナー判断で未配線のまま。REQ-41 の
//! TASK-41.6 で再利用予定）。呼ぶ場合は、候補構成（種類 × 推論ランタイム × 数値形式）ごとに
//! [`screen_candidate`] を呼び、[`ExportScreening`] の
//! `eligible` だけを配布候補に残し、`excluded` を機械可読な [`ExclusionRecord`] として出力する。
//! JSON への直列化は CLI 側（`fandhe-edge-cli` の `output.rs`。runtime は serde を持たない）。
//! 合否判定（[`crate::package_outcome`]）・容量（[`crate::capacity`]）は変更しない。
//!
//! # 除外の根拠と証拠種別
//!
//! - 既知制約の表 [`KNOWN_INFEASIBLE`]: PoC-14 実測（証拠種別: 実機）。`tract-onnx` 0.21.18 は
//!   C3 の動的量子化 INT8（`DynamicQuantizeLinear` の U8 出力と `ConvInteger` の I8 重み）を
//!   `Impossible to unify U8 with I8` で読み込めなかった。TRACT-i8 × C1・TRACT-f32 × C3 は成功
//! - `tract-onnx`・`ort` は未承認の依存のため本 crate では実行しない。除外は表で宣言的に判定し、
//!   判定と記録をテストで固定する（証拠種別: テストハーネス）
//! - 実行時に検出した除外（予測ずれ・読み込み拒否など）の証拠種別は [`EvidenceKind::TestHarness`]
//!   （本ツールの照合による検出で、実機計測ではない）
//!
//! # 判定基準
//!
//! 予測ラベルは全件一致を基準にする（REQ-32 正常系・`tests/onnx_parity.rs` に合わせる）。
//! 量子化形式に許容幅（McNemar 等）を設けるかは評価契約の判断事項で、本 TASK の範囲外。
//!
//! # 呼び出し側への契約
//!
//! - 照合ケースに凍結済みの最終 test を使わない（REQ-27「最終 test は 1 回限り」）。参照ラベルは
//!   学習フレームワーク内の推論の予測ラベルで、正解ラベルではない。推論には `input` だけを渡す
//! - モデルの読み込み経路の閉じ込め（`safe_join` 相当）は呼び出し側（ガード層。REQ-39）の責務
//! - sha256 不一致・破損・上限超過・I/O 失敗は「除外」にせず [`ScreeningError`] で全体を止める
//!   （fail-closed。REQ-39）
//! - 語彙ファイル超過構成の除外（TASK-30.3・#125）は [`crate::vocab_exclusion`] が持つ。
//!   [`ExportConfig`] は Qwen 語彙流用の構成を表せないため、本モジュールへは相乗りしない
//!
//! # 追加学習の種類 `qwen2_lora` の対応範囲外（REQ-41・TASK-41.6-4・#433）
//!
//! [`screen_scoring_candidate`] が、書き出した ONNX の演算（演算名 × domain。許可リストは
//! [`SUPPORTED_SCORING_OPS`]、設計書 `docs/design/qwen2-scoring-onnx-ops.md` §3）とトークナイザー
//! JSON（[`TokenizerSpec`] の対応範囲）を判定し、範囲外を [`ExclusionReason::UnsupportedOps`]・
//! [`ExclusionReason::UnsupportedTokenizer`] で除外する。属性・dtype・rank の範囲外は実行器の
//! 読み込み（#434）で判定する。範囲内でも実行器が未接続で予測照合ができないため
//! [`ExclusionReason::ParityUnverified`] で除外する（#434 で推論照合へ差し替え）。
//! 7B 非 tie の lm_head が 1 GiB を超える場合は学習ワーカーが `limit_exceeded` で停止する現行の
//! 契約のままで、本モジュールでは扱わない。

use crate::onnx::{
    FINETUNE_MODEL_LIMITS, ModelKind, NodeOp, OnnxBackend, OnnxLoadError, load_pipeline,
    scan_node_ops_path,
};
use crate::pipeline::{
    BackendError, InferError, InferencePipeline, MAX_INFER_BATCH_LEN, MAX_INFER_BATCH_TOTAL_BYTES,
};
use crate::preprocess::ByteEncodingPreprocessor;
use crate::tokenizer::{TokenizerLoadError, TokenizerSpec};
use fandhe_edge_core::hash::Sha256Digest;
use std::fmt;
use std::path::Path;
use std::time::{Duration, Instant};

/// 1 回の [`screen_candidates`] が受け付ける候補数の上限（現実の組み合わせは 2 × 3 × 3 = 18）。
pub const MAX_EXPORT_CANDIDATES: usize = 64;
/// 1 候補あたりの照合ケース数の上限（バッチ推論の上限と同値。REQ-39）。
pub const MAX_PARITY_CASES: usize = MAX_INFER_BATCH_LEN;
/// 1 候補の予測照合全体の時間上限（REQ-39。暫定値。ケースごとの推論の合計に対する上限。
/// 各推論には残り時間を打ち切り上限として渡すため、1 件の推論中にも期限を強制する）。
pub const MAX_PARITY_DURATION: Duration = Duration::from_secs(60);

/// 書き出し先の推論ランタイム。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InferenceRuntime {
    /// 本 crate の自作バックエンド（[`crate::onnx`]）。
    Own,
    /// ONNX Runtime（`ort`。未承認のため本リポでは実行しない）。
    OnnxRuntime,
    /// `tract-onnx`（未承認のため本リポでは実行しない）。
    Tract,
}

impl InferenceRuntime {
    /// 機械可読な名前（ASCII snake_case）。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Own => "own",
            Self::OnnxRuntime => "onnx_runtime",
            Self::Tract => "tract",
        }
    }
}

/// 数値形式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NumericFormat {
    /// 32 ビット浮動小数。
    F32,
    /// 16 ビット浮動小数。
    F16,
    /// 動的量子化 INT8（PoC-14 の方式）。
    Int8Dynamic,
}

impl NumericFormat {
    /// 機械可読な名前（ASCII snake_case）。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Int8Dynamic => "int8_dynamic",
        }
    }
}

/// 配布候補の種類。推論側の [`ModelKind`] に無い `qwen2_lora` も表す（実行器の無い種類を推論の
/// 許可リストへ載せないため、除外判定側にだけ持つ。REQ-41・#433）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExportKind {
    /// バイト n-gram TF-IDF + ロジスティック回帰。
    C1,
    /// バイト CNN。
    C3,
    /// 追加学習（Qwen2 の LoRA 統合・採点グラフ）。推論の実行器は #434 で接続する。
    Qwen2Lora,
}

impl ExportKind {
    /// `artifact.json` の `kind` 文字列と同じ名前（ASCII snake_case）。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::C1 => "c1",
            Self::C3 => "c3",
            Self::Qwen2Lora => "qwen2_lora",
        }
    }

    /// 推論側の種類。実行器の無い種類は `None`。
    pub fn model_kind(self) -> Option<ModelKind> {
        match self {
            Self::C1 => Some(ModelKind::C1),
            Self::C3 => Some(ModelKind::C3),
            Self::Qwen2Lora => None,
        }
    }
}

impl From<ModelKind> for ExportKind {
    fn from(k: ModelKind) -> Self {
        match k {
            ModelKind::C1 => Self::C1,
            ModelKind::C3 => Self::C3,
        }
    }
}

/// 配布候補の構成（種類 × ランタイム × 数値形式）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportConfig {
    /// モデルの種類（`autoregressive` は未対応。REQ-19b の後続 TASK）。
    pub kind: ExportKind,
    /// 推論ランタイム。
    pub runtime: InferenceRuntime,
    /// 数値形式。
    pub format: NumericFormat,
}

/// 証拠の種別（spec の 4 種。記録の際に必ず明記する）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceKind {
    /// テストハーネス。
    TestHarness,
    /// 模擬。
    Simulated,
    /// 推定。
    Estimated,
    /// 実機。
    Measured,
}

impl EvidenceKind {
    /// 機械可読な名前（ASCII snake_case）。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TestHarness => "test_harness",
            Self::Simulated => "simulated",
            Self::Estimated => "estimated",
            Self::Measured => "measured",
        }
    }
}

/// 書き出しが技術的に成立しない理由の詳細。利用者のデータ・パスは含まない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExportInfeasibility {
    /// 動的量子化の出力型（U8）と畳み込み整数演算の重み型（I8）を統一できない。
    QuantizedConvTypeMismatch,
}

impl ExportInfeasibility {
    /// 機械可読な詳細コード（ASCII snake_case）。
    pub fn code(self) -> &'static str {
        match self {
            Self::QuantizedConvTypeMismatch => "type_unification_failed",
        }
    }

    /// 固定の英語説明（PoC-14 で観測した型不整合の要約）。
    pub fn description(self) -> &'static str {
        match self {
            Self::QuantizedConvTypeMismatch => {
                "DynamicQuantizeLinear (U8) and ConvInteger (I8) types cannot be unified"
            }
        }
    }
}

/// 既知の書き出し不能構成 1 件（宣言的な表の要素）。
#[derive(Debug, PartialEq, Eq)]
pub struct KnownInfeasibility {
    /// 対象の構成。
    pub config: ExportConfig,
    /// 理由の詳細。
    pub reason: ExportInfeasibility,
    /// 根拠の証拠種別。
    pub evidence: EvidenceKind,
    /// 根拠の出典（PoC-n）。
    pub source: &'static str,
}

/// PoC-14 実測に基づく既知の書き出し不能構成（証拠種別: 実機）。
pub static KNOWN_INFEASIBLE: &[KnownInfeasibility] = &[KnownInfeasibility {
    config: ExportConfig {
        kind: ExportKind::C3,
        runtime: InferenceRuntime::Tract,
        format: NumericFormat::Int8Dynamic,
    },
    reason: ExportInfeasibility::QuantizedConvTypeMismatch,
    evidence: EvidenceKind::Measured,
    source: "PoC-14",
}];

/// 構成が既知の書き出し不能構成に該当すればその項目を返す。
pub fn known_infeasibility(config: &ExportConfig) -> Option<&'static KnownInfeasibility> {
    KNOWN_INFEASIBLE.iter().find(|k| k.config == *config)
}

/// 除外の理由。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExclusionReason {
    /// 既知制約により書き出しが成立しない。
    ExportInfeasible(&'static KnownInfeasibility),
    /// 本リポで実行できないランタイム・数値形式の組（未検証の組を通さない）。
    RuntimeNotAvailable,
    /// 自作ランタイムが書き出し物の形式を受け付けない。`load_code` は [`OnnxLoadError::code`]。
    RuntimeRejectedModel {
        /// 読み込みエラーのコード。
        load_code: &'static str,
    },
    /// 予測ラベルが参照とずれる。
    PredictionMismatch {
        /// 不一致の件数。
        mismatched: usize,
        /// 照合した件数。
        total: usize,
    },
    /// 照合ケースが 0 件、または実行器が未接続で一致を確かめられない。
    ParityUnverified,
    /// 採点グラフに許可リスト外の演算がある（REQ-41・#433）。`ops` は整形済みの演算名
    /// （ソート・重複除去・先頭 [`MAX_REPORTED_OPS`] 件）、`total` は相異なる未対応演算の総数。
    UnsupportedOps {
        /// 未対応の演算名（整形済み）。
        ops: Vec<String>,
        /// 相異なる未対応演算の総数。
        total: usize,
    },
    /// トークナイザー JSON が対応範囲外（REQ-41・#433）。`section` は
    /// [`TokenizerLoadError::Unsupported`] の固定の節名。
    UnsupportedTokenizer {
        /// 範囲外だった節名。
        section: &'static str,
    },
}

impl ExclusionReason {
    /// 機械可読な除外コード（ASCII snake_case）。
    pub fn code(&self) -> &'static str {
        match self {
            Self::ExportInfeasible(_) => "export_infeasible",
            Self::RuntimeNotAvailable => "runtime_not_available",
            Self::RuntimeRejectedModel { .. } => "runtime_rejected_model",
            Self::PredictionMismatch { .. } => "prediction_mismatch",
            Self::ParityUnverified => "parity_unverified",
            Self::UnsupportedOps { .. } => "unsupported_ops",
            Self::UnsupportedTokenizer { .. } => "unsupported_tokenizer",
        }
    }
}

/// 除外記録。配布候補から外した構成・理由・証拠種別を持つ。入力本文・パス・重みは持たない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExclusionRecord {
    /// 除外した構成。
    pub config: ExportConfig,
    /// 理由。
    pub reason: ExclusionReason,
    /// 証拠種別（既知制約は表の値、実行時検出は `TestHarness`）。
    pub evidence: EvidenceKind,
}

impl ExclusionRecord {
    /// 除外コード。
    pub fn code(&self) -> &'static str {
        self.reason.code()
    }

    /// 詳細コード（既知制約の理由・読み込み拒否のコード）。
    pub fn detail_code(&self) -> Option<&'static str> {
        match &self.reason {
            ExclusionReason::ExportInfeasible(k) => Some(k.reason.code()),
            ExclusionReason::RuntimeRejectedModel { load_code } => Some(load_code),
            ExclusionReason::UnsupportedTokenizer { section } => Some(section),
            _ => None,
        }
    }

    /// 出典（既知制約の場合のみ）。
    pub fn source(&self) -> Option<&'static str> {
        match &self.reason {
            ExclusionReason::ExportInfeasible(k) => Some(k.source),
            _ => None,
        }
    }

    /// 公開してよい英語の固定文（件数とコードのみ。入力本文・パスを含めない。security.md P0）。
    pub fn public_message(&self) -> String {
        let head = format!(
            "excluded {}/{}/{}",
            self.config.kind.as_str(),
            self.config.runtime.as_str(),
            self.config.format.as_str()
        );
        match &self.reason {
            ExclusionReason::ExportInfeasible(k) => {
                format!("{head}: export infeasible: {}", k.reason.description())
            }
            ExclusionReason::RuntimeNotAvailable => {
                format!("{head}: runtime and format combination is not available")
            }
            ExclusionReason::RuntimeRejectedModel { load_code } => {
                format!("{head}: runtime rejected the exported model ({load_code})")
            }
            ExclusionReason::PredictionMismatch { mismatched, total } => {
                format!("{head}: {mismatched} of {total} predictions differ from the reference")
            }
            ExclusionReason::ParityUnverified => {
                format!("{head}: predictions could not be verified against the reference")
            }
            ExclusionReason::UnsupportedOps { total, .. } => {
                format!("{head}: {total} unsupported operator type(s) in the graph")
            }
            ExclusionReason::UnsupportedTokenizer { section } => {
                format!("{head}: unsupported tokenizer section ({section})")
            }
        }
    }
}

/// 1 候補の判定結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateDecision {
    /// 配布候補に残す。
    Eligible(ExportConfig),
    /// 配布候補から外す。
    Excluded(ExclusionRecord),
}

/// 複数候補の選別結果。入力順を保つ（決定性）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExportScreening {
    /// 配布候補に残した構成。
    pub eligible: Vec<ExportConfig>,
    /// 除外した構成と理由。
    pub excluded: Vec<ExclusionRecord>,
}

/// 処理全体を止めるエラー（除外ではない。fail-closed）。
#[derive(Debug)]
#[non_exhaustive]
pub enum ScreeningError {
    /// 完全性・I/O・破損・上限の読み込みエラー。
    Load(OnnxLoadError),
    /// トークナイザー JSON の完全性・破損・上限・内容不正（対応範囲外は除外で、ここには来ない）。
    Tokenizer(TokenizerLoadError),
    /// 照合中の推論失敗。
    Infer(InferError),
    /// 候補数が [`MAX_EXPORT_CANDIDATES`] を超えた。
    TooManyCandidates,
    /// 照合ケース数が [`MAX_PARITY_CASES`] を超えた。
    TooManyCases,
    /// 照合ケースの総入力バイト数が [`MAX_INFER_BATCH_TOTAL_BYTES`] を超えた（推論前に検査。REQ-39）。
    TotalInputTooLarge,
    /// 照合全体の時間が [`MAX_PARITY_DURATION`] を超えた（REQ-39。処理を停止する）。
    ParityTimeExceeded,
    /// 参照ラベルがクラス数の範囲外。
    ReferenceOutOfRange,
    /// 同じ構成が重複して渡された。
    DuplicateCandidate,
}

impl ScreeningError {
    /// 機械可読なエラーコード（英語）。写像先の終了コードは CLI 側（入力不正 64・上限 20・実行時 70）。
    pub fn code(&self) -> &'static str {
        match self {
            Self::Load(e) => e.code(),
            Self::Tokenizer(e) => e.code(),
            Self::Infer(e) => e.code(),
            Self::TooManyCandidates
            | Self::TooManyCases
            | Self::TotalInputTooLarge
            | Self::ParityTimeExceeded => "limit_exceeded",
            Self::ReferenceOutOfRange => "reference_out_of_range",
            Self::DuplicateCandidate => "duplicate_candidate",
        }
    }
}

impl fmt::Display for ScreeningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "export screening failed: {}", self.code())
    }
}

impl std::error::Error for ScreeningError {}

/// 読み込みエラーの分類結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadErrorClass {
    /// 書き出し物の形式が成立しないものとして除外する（コードを記録）。
    Exclude(&'static str),
    /// 除外にせずエラーとして返す。
    Propagate,
}

/// 読み込みエラーを「除外」か「全体停止」に振り分ける。未知のバリアントは必ず `Propagate`
/// （改ざん・破損を除外として黙って通さない。REQ-39 fail-closed）。
pub fn classify_load_error(err: &OnnxLoadError) -> LoadErrorClass {
    match err {
        OnnxLoadError::UnsupportedGraph
        | OnnxLoadError::UnsupportedTensor
        | OnnxLoadError::UnsupportedOpset
        | OnnxLoadError::UnsupportedIrVersion
        | OnnxLoadError::UnsupportedKind => LoadErrorClass::Exclude(err.code()),
        _ => LoadErrorClass::Propagate,
    }
}

/// 照合ケース 1 件。`reference_label` は学習フレームワーク内の推論の予測ラベル（正解ではない）。
#[derive(Debug, Clone, Copy)]
pub struct ParityCase<'a> {
    /// 推論へ渡す入力（推論関数へ渡してよい唯一の情報。REQ-27）。
    pub input: &'a str,
    /// 参照の予測ラベル index。
    pub reference_label: usize,
}

/// 照合結果。件数のみで入力本文・添字は持たない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParityReport {
    /// 照合した件数。
    pub total: usize,
    /// 不一致の件数。
    pub mismatched: usize,
}

/// 照合ケースの件数と総入力バイト数を推論前に検査する（REQ-39）。除外判定より先に呼び、
/// 構成によって上限エラーと除外記録に分かれないようにする。`infer_batch` の総量上限の迂回も防ぐ。
fn validate_parity_cases(cases: &[ParityCase<'_>]) -> Result<(), ScreeningError> {
    if cases.len() > MAX_PARITY_CASES {
        return Err(ScreeningError::TooManyCases);
    }
    let mut total_bytes = 0usize;
    for case in cases {
        total_bytes = total_bytes
            .checked_add(case.input.len())
            .ok_or(ScreeningError::TotalInputTooLarge)?;
        if total_bytes > MAX_INFER_BATCH_TOTAL_BYTES {
            return Err(ScreeningError::TotalInputTooLarge);
        }
    }
    Ok(())
}

/// 予測ラベルを参照と照合する。クラス数は予測のスコア列の長さから取る。
/// 照合全体の時間上限は [`MAX_PARITY_DURATION`]。
///
/// # Errors
/// 参照ラベルがクラス数以上（`ReferenceOutOfRange`）、推論失敗（`Infer`）、ケース数・総量・時間の超過。
pub fn check_prediction_parity(
    pipeline: &InferencePipeline<ByteEncodingPreprocessor, OnnxBackend>,
    cases: &[ParityCase<'_>],
) -> Result<ParityReport, ScreeningError> {
    check_prediction_parity_within(pipeline, cases, MAX_PARITY_DURATION)
}

/// [`check_prediction_parity`] の時間上限を引数で受ける版（テストで上限を小さくするため）。
///
/// # Errors
/// [`check_prediction_parity`] と同じ。
pub fn check_prediction_parity_within(
    pipeline: &InferencePipeline<ByteEncodingPreprocessor, OnnxBackend>,
    cases: &[ParityCase<'_>],
    max_duration: Duration,
) -> Result<ParityReport, ScreeningError> {
    let started = Instant::now();
    check_prediction_parity_with_clock(pipeline, cases, max_duration, || started.elapsed())
}

/// 経過時間の取得を `elapsed` で差し替えられる版（最後の推論中に期限を超える境界を決定的にテストするため）。
/// 各推論の前後で経過時間を確認し、超過なら結果を確定せず停止する（fail-closed。REQ-39）。
///
/// # Errors
/// [`check_prediction_parity`] と同じ。
pub fn check_prediction_parity_with_clock<C>(
    pipeline: &InferencePipeline<ByteEncodingPreprocessor, OnnxBackend>,
    cases: &[ParityCase<'_>],
    max_duration: Duration,
    mut elapsed: C,
) -> Result<ParityReport, ScreeningError>
where
    C: FnMut() -> Duration,
{
    validate_parity_cases(cases)?;
    let mut mismatched = 0usize;
    for case in cases {
        // 各推論の前に照合全体の残り時間を確認し、尽きていれば停止する（fail-closed。REQ-39）
        let now = elapsed();
        if now >= max_duration {
            return Err(ScreeningError::ParityTimeExceeded);
        }
        // 残り時間を推論の打ち切り上限として渡し、1 件の推論中にも期限を強制する（REQ-39）
        let pred = match pipeline.infer_one_within(case.input, max_duration.saturating_sub(now)) {
            Ok(p) => p,
            Err(InferError::Backend(BackendError::TimeLimitExceeded)) => {
                return Err(ScreeningError::ParityTimeExceeded);
            }
            Err(e) => return Err(ScreeningError::Infer(e)),
        };
        // 打ち切り検査の粒度を超えて期限を過ぎた場合も結果を確定させず停止する
        if elapsed() >= max_duration {
            return Err(ScreeningError::ParityTimeExceeded);
        }
        if case.reference_label >= pred.scores().len() {
            return Err(ScreeningError::ReferenceOutOfRange);
        }
        if pred.label_index() != case.reference_label {
            mismatched = mismatched.saturating_add(1);
        }
    }
    Ok(ParityReport {
        total: cases.len(),
        mismatched,
    })
}

fn excluded(
    config: ExportConfig,
    reason: ExclusionReason,
    evidence: EvidenceKind,
) -> CandidateDecision {
    CandidateDecision::Excluded(ExclusionRecord {
        config,
        reason,
        evidence,
    })
}

/// 1 構成を判定する。ケースの件数・総量は除外判定より先に検査する。`load` は遅延実行され、既知制約・未対応ランタイム・ケース 0 件では呼ばれない。
///
/// # Errors
/// [`ScreeningError`]（除外ではなく処理全体を止める失敗）。
pub fn screen_candidate<F>(
    config: ExportConfig,
    load: F,
    cases: &[ParityCase<'_>],
) -> Result<CandidateDecision, ScreeningError>
where
    F: FnOnce(
        ModelKind,
    )
        -> Result<InferencePipeline<ByteEncodingPreprocessor, OnnxBackend>, OnnxLoadError>,
{
    // 早期除外より先にケースの件数・総量を検査する（構成によって結果が変わらないように。REQ-39）
    validate_parity_cases(cases)?;
    if let Some(known) = known_infeasibility(&config) {
        return Ok(excluded(
            config,
            ExclusionReason::ExportInfeasible(known),
            known.evidence,
        ));
    }
    if !matches!(
        (config.runtime, config.format),
        (InferenceRuntime::Own, NumericFormat::F32)
    ) {
        return Ok(excluded(
            config,
            ExclusionReason::RuntimeNotAvailable,
            EvidenceKind::TestHarness,
        ));
    }
    // 推論側に実行器の無い種類（qwen2_lora）はこの経路では照合できない（screen_scoring_candidate を使う）
    let Some(model_kind) = config.kind.model_kind() else {
        return Ok(excluded(
            config,
            ExclusionReason::RuntimeRejectedModel {
                load_code: OnnxLoadError::UnsupportedKind.code(),
            },
            EvidenceKind::TestHarness,
        ));
    };
    if cases.is_empty() {
        return Ok(excluded(
            config,
            ExclusionReason::ParityUnverified,
            EvidenceKind::TestHarness,
        ));
    }
    let pipeline = match load(model_kind) {
        Ok(p) => p,
        Err(e) => {
            return match classify_load_error(&e) {
                LoadErrorClass::Exclude(load_code) => Ok(excluded(
                    config,
                    ExclusionReason::RuntimeRejectedModel { load_code },
                    EvidenceKind::TestHarness,
                )),
                LoadErrorClass::Propagate => Err(ScreeningError::Load(e)),
            };
        }
    };
    let report = check_prediction_parity(&pipeline, cases)?;
    if report.mismatched > 0 {
        return Ok(excluded(
            config,
            ExclusionReason::PredictionMismatch {
                mismatched: report.mismatched,
                total: report.total,
            },
            EvidenceKind::TestHarness,
        ));
    }
    Ok(CandidateDecision::Eligible(config))
}

/// パスと sha256 からモデルを読んで判定する薄い関数（内部で [`load_pipeline`]。サイズ上限と
/// sha256 照合を通る。検証は除外判定より先に行い、既知制約で除外する構成でもモデルの欠落・破損・
/// 不一致は [`ScreeningError::Load`] になる）。パスの閉じ込めは呼び出し側（ガード層）の責務。
///
/// # Errors
/// [`screen_candidate`] と同じ。
pub fn screen_candidate_from_path(
    config: ExportConfig,
    max_bytes: usize,
    path: &Path,
    expected_sha256: &Sha256Digest,
    cases: &[ParityCase<'_>],
) -> Result<CandidateDecision, ScreeningError> {
    // 除外判定（既知制約・未対応ランタイム・ケース 0 件）より先にファイルの存在・サイズ・sha256 を
    // 検証する。欠落・破損・改ざんは除外に化けさせず全体停止にする（REQ-39 fail-closed）。
    // 形式不成立（`Exclude` 分類）だけは除外判定へ回すため、結果を保持して渡す。
    validate_parity_cases(cases)?;
    let Some(model_kind) = config.kind.model_kind() else {
        // 実行器の無い種類でも、上限付き読み込みと sha256 照合を除外判定より先に行う（REQ-39）
        if let Err(e) = scan_node_ops_path(path, &FINETUNE_MODEL_LIMITS, expected_sha256)
            && classify_load_error(&e) == LoadErrorClass::Propagate
        {
            return Err(ScreeningError::Load(e));
        }
        return screen_candidate(config, |_| Err(OnnxLoadError::UnsupportedKind), cases);
    };
    let loaded = load_pipeline(model_kind, max_bytes, path, expected_sha256);
    let loaded = match loaded {
        Err(e) if classify_load_error(&e) == LoadErrorClass::Propagate => {
            return Err(ScreeningError::Load(e));
        }
        other => other,
    };
    screen_candidate(config, |_| loaded, cases)
}

/// 採点グラフが使ってよい演算（既定 domain。`docs/design/qwen2-scoring-onnx-ops.md` §3 の 28 種。
/// ソート済み。変更は設計書の更新を先に行う）。
pub const SUPPORTED_SCORING_OPS: [&str; 28] = [
    "Add",
    "Cast",
    "Concat",
    "Cos",
    "Expand",
    "Gather",
    "GatherElements",
    "Gemm",
    "Greater",
    "LogSoftmax",
    "MatMul",
    "Mul",
    "Range",
    "Reciprocal",
    "ReduceMean",
    "ReduceSum",
    "Reshape",
    "Shape",
    "Sigmoid",
    "Sin",
    "Slice",
    "Softmax",
    "Split",
    "Sqrt",
    "Sub",
    "Transpose",
    "Unsqueeze",
    "Where",
];

/// 除外記録に載せる未対応演算名の件数の上限。
pub const MAX_REPORTED_OPS: usize = 16;
/// 除外記録に載せる演算名 1 件の文字数の上限。
const MAX_REPORTED_OP_CHARS: usize = 64;

/// 許可リスト外の演算を、記録用に整形して返す（相異なる名前の総数も返す）。
///
/// 演算名はファイル由来の untrusted な文字列のため、`[A-Za-z0-9_.]` 以外は `_` に置換し
/// [`MAX_REPORTED_OP_CHARS`] 文字で切り詰め（総数は整形前の domain と op_type の組で数える）、ソート・重複除去して先頭 [`MAX_REPORTED_OPS`] 件にする。
/// 既定 domain 以外の演算はすべて未対応（設計書 §5）。
pub fn unsupported_scoring_ops(ops: &[NodeOp]) -> (Vec<String>, usize) {
    let mut raw: Vec<(&str, &str)> = ops
        .iter()
        .filter(|o| !o.domain.is_empty() || !SUPPORTED_SCORING_OPS.contains(&o.op_type.as_str()))
        .map(|o| (o.domain.as_str(), o.op_type.as_str()))
        .collect();
    // 総数は整形前の (domain, op_type) の組で数える（整形後の衝突で過少にしない）
    raw.sort();
    raw.dedup();
    let total = raw.len();
    let mut names: Vec<String> = raw
        .iter()
        .map(|(_, op)| {
            op.chars()
                .take(MAX_REPORTED_OP_CHARS)
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '.' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect()
        })
        .collect();
    names.sort();
    names.dedup();
    names.truncate(MAX_REPORTED_OPS);
    (names, total)
}

/// トークナイザーの読み込みエラーを「除外」か「全体停止」に振り分ける。対応範囲外
/// （`Unsupported`）だけを除外にし、完全性・破損・上限・内容不正と未知のバリアントは
/// `Propagate`（REQ-39 fail-closed）。
pub fn classify_tokenizer_error(err: &TokenizerLoadError) -> LoadErrorClass {
    match err {
        TokenizerLoadError::Unsupported { section, .. } => LoadErrorClass::Exclude(section),
        _ => LoadErrorClass::Propagate,
    }
}

/// 追加学習の種類（`qwen2_lora`）の書き出し物 1 構成を判定する（REQ-41・TASK-41.6-4・#433）。
///
/// 順序: 両ファイルの完全性系エラーを先に全体停止 → 既知制約 → ランタイム・形式の門
/// （`Own` × `F32` のみ）→ トークナイザー範囲外 → ONNX の読み込み拒否 → 未対応演算 →
/// 実行器未接続（[`ExclusionReason::ParityUnverified`]。#434 で推論照合へ差し替える）。
/// 経路の閉じ込めは呼び出し側（ガード層）の責務。
///
/// # Errors
/// [`ScreeningError`]（sha256 不一致・破損・上限超過・I/O 失敗・トークナイザーの内容不正）。
pub fn screen_scoring_candidate(
    config: ExportConfig,
    onnx_path: &Path,
    onnx_sha256: &Sha256Digest,
    tokenizer_path: &Path,
    tokenizer_sha256: &Sha256Digest,
) -> Result<CandidateDecision, ScreeningError> {
    // 完全性・破損・上限は除外に化けさせず全体停止にする（どの除外判定よりも先。REQ-39）
    let tokenizer_section = match TokenizerSpec::load_path(tokenizer_path, tokenizer_sha256) {
        Ok(_) => None,
        Err(e) => match classify_tokenizer_error(&e) {
            LoadErrorClass::Exclude(section) => Some(section),
            LoadErrorClass::Propagate => return Err(ScreeningError::Tokenizer(e)),
        },
    };
    let ops = match scan_node_ops_path(onnx_path, &FINETUNE_MODEL_LIMITS, onnx_sha256) {
        Err(e) if classify_load_error(&e) == LoadErrorClass::Propagate => {
            return Err(ScreeningError::Load(e));
        }
        other => other,
    };
    let harness = EvidenceKind::TestHarness;
    if let Some(known) = known_infeasibility(&config) {
        return Ok(excluded(
            config,
            ExclusionReason::ExportInfeasible(known),
            known.evidence,
        ));
    }
    if !matches!(
        (config.runtime, config.format),
        (InferenceRuntime::Own, NumericFormat::F32)
    ) {
        return Ok(excluded(
            config,
            ExclusionReason::RuntimeNotAvailable,
            harness,
        ));
    }
    if let Some(section) = tokenizer_section {
        return Ok(excluded(
            config,
            ExclusionReason::UnsupportedTokenizer { section },
            harness,
        ));
    }
    let ops = match ops {
        Ok(o) => o,
        Err(e) => {
            // 上で Propagate は返し済み。残りは形式の不成立（Exclude）
            let load_code = match classify_load_error(&e) {
                LoadErrorClass::Exclude(c) => c,
                LoadErrorClass::Propagate => return Err(ScreeningError::Load(e)),
            };
            return Ok(excluded(
                config,
                ExclusionReason::RuntimeRejectedModel { load_code },
                harness,
            ));
        }
    };
    let (names, total) = unsupported_scoring_ops(&ops);
    if total > 0 {
        return Ok(excluded(
            config,
            ExclusionReason::UnsupportedOps { ops: names, total },
            harness,
        ));
    }
    Ok(excluded(config, ExclusionReason::ParityUnverified, harness))
}

/// 複数構成を入力順に判定する。候補数上限・重複検査・順序保持をここに集約する。
///
/// `decide` は構成ごとの判定（通常は [`screen_candidate_from_path`] を包む）。
///
/// # Errors
/// 候補数超過・重複、または `decide` が返したエラー。
pub fn screen_candidates<F>(
    configs: &[ExportConfig],
    mut decide: F,
) -> Result<ExportScreening, ScreeningError>
where
    F: FnMut(ExportConfig) -> Result<CandidateDecision, ScreeningError>,
{
    if configs.len() > MAX_EXPORT_CANDIDATES {
        return Err(ScreeningError::TooManyCandidates);
    }
    for (i, c) in configs.iter().enumerate() {
        if configs.iter().take(i).any(|p| p == c) {
            return Err(ScreeningError::DuplicateCandidate);
        }
    }
    let mut out = ExportScreening::default();
    for &config in configs {
        match decide(config)? {
            CandidateDecision::Eligible(c) => out.eligible.push(c),
            CandidateDecision::Excluded(r) => out.excluded.push(r),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::BackendError;

    fn cfg(kind: ModelKind, runtime: InferenceRuntime, format: NumericFormat) -> ExportConfig {
        ExportConfig {
            kind: kind.into(),
            runtime,
            format,
        }
    }

    /// REQ-32: PoC-14 で失敗した組だけが表に該当する。
    #[test]
    fn req32_known_table_positive_and_negative() {
        let hit = known_infeasibility(&cfg(
            ModelKind::C3,
            InferenceRuntime::Tract,
            NumericFormat::Int8Dynamic,
        ))
        .expect("hit");
        assert_eq!(hit.reason.code(), "type_unification_failed");
        assert_eq!(hit.evidence, EvidenceKind::Measured);
        assert_eq!(hit.source, "PoC-14");
        for c in [
            cfg(
                ModelKind::C1,
                InferenceRuntime::Tract,
                NumericFormat::Int8Dynamic,
            ),
            cfg(ModelKind::C3, InferenceRuntime::Tract, NumericFormat::F32),
            cfg(
                ModelKind::C3,
                InferenceRuntime::OnnxRuntime,
                NumericFormat::Int8Dynamic,
            ),
        ] {
            assert!(known_infeasibility(&c).is_none());
        }
    }

    /// REQ-32・REQ-39: 形式の不成立だけを除外にし、完全性・破損・上限・I/O は全体停止にする。
    #[test]
    fn req32_classify_load_error_mapping() {
        use fandhe_edge_core::fs::FsError;
        for e in [
            OnnxLoadError::UnsupportedGraph,
            OnnxLoadError::UnsupportedTensor,
            OnnxLoadError::UnsupportedOpset,
            OnnxLoadError::UnsupportedIrVersion,
            OnnxLoadError::UnsupportedKind,
        ] {
            assert_eq!(classify_load_error(&e), LoadErrorClass::Exclude(e.code()));
        }
        for e in [
            OnnxLoadError::File(FsError::NotRegularFile {
                path: std::path::PathBuf::new(),
            }),
            OnnxLoadError::IntegrityMismatch,
            OnnxLoadError::MalformedProtobuf,
            OnnxLoadError::LimitExceeded,
            OnnxLoadError::MaxBytesOutOfRange,
        ] {
            assert_eq!(classify_load_error(&e), LoadErrorClass::Propagate);
        }
    }

    /// REQ-32: コード・名前は ASCII snake_case の固定値（JSON 直列化でエスケープ不要の前提）。
    #[test]
    fn req32_codes_are_fixed_ascii_snake_case() {
        let names = [
            InferenceRuntime::Own.as_str(),
            InferenceRuntime::OnnxRuntime.as_str(),
            InferenceRuntime::Tract.as_str(),
            NumericFormat::F32.as_str(),
            NumericFormat::F16.as_str(),
            NumericFormat::Int8Dynamic.as_str(),
            EvidenceKind::TestHarness.as_str(),
            EvidenceKind::Simulated.as_str(),
            EvidenceKind::Estimated.as_str(),
            EvidenceKind::Measured.as_str(),
            ExportInfeasibility::QuantizedConvTypeMismatch.code(),
            ExclusionReason::RuntimeNotAvailable.code(),
            ExclusionReason::ParityUnverified.code(),
            ExclusionReason::RuntimeRejectedModel { load_code: "x" }.code(),
            ExclusionReason::PredictionMismatch {
                mismatched: 1,
                total: 2,
            }
            .code(),
        ];
        assert_eq!(names[0], "own");
        assert_eq!(names[1], "onnx_runtime");
        assert_eq!(names[5], "int8_dynamic");
        assert_eq!(names[9], "measured");
        assert_eq!(names[10], "type_unification_failed");
        assert_eq!(names[11], "runtime_not_available");
        assert_eq!(names[12], "parity_unverified");
        assert_eq!(names[13], "runtime_rejected_model");
        assert_eq!(names[14], "prediction_mismatch");
        for n in names {
            assert!(
                n.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{n}"
            );
        }
    }

    /// REQ-32・security.md: 公開メッセージは固定文と件数だけ。
    #[test]
    fn req32_public_message_is_fixed_text() {
        let r = ExclusionRecord {
            config: cfg(ModelKind::C1, InferenceRuntime::Own, NumericFormat::F32),
            reason: ExclusionReason::PredictionMismatch {
                mismatched: 3,
                total: 10,
            },
            evidence: EvidenceKind::TestHarness,
        };
        assert_eq!(
            r.public_message(),
            "excluded c1/own/f32: 3 of 10 predictions differ from the reference"
        );
        assert_eq!(r.detail_code(), None);
        assert_eq!(r.source(), None);
    }

    /// REQ-41・#433: 許可リストは 28 件・ソート済み・重複なし。
    #[test]
    fn req41_supported_scoring_ops_sorted_unique_28() {
        assert_eq!(SUPPORTED_SCORING_OPS.len(), 28);
        assert!(SUPPORTED_SCORING_OPS.windows(2).all(|w| w[0] < w[1]));
    }

    /// REQ-41・#433: 演算名は整形・切り詰め・件数制限され、既定 domain 以外は未対応になる。
    #[test]
    fn req41_op_name_sanitized_and_capped() {
        let op = |domain: &str, name: &str| NodeOp {
            domain: domain.into(),
            op_type: name.into(),
        };
        let long = "A".repeat(100);
        let mut ops = vec![
            op("", "Add"),
            op("", "bad\"name\n"),
            op("", "bad\"name\n"),
            op("", &long),
            op("com.microsoft", "Add"),
        ];
        let (names, total) = unsupported_scoring_ops(&ops);
        assert_eq!(total, 3);
        assert_eq!(
            names,
            vec!["A".repeat(64), "Add".to_string(), "bad_name_".to_string()]
        );

        let collide = [op("", "Foo-Bar"), op("", "Foo_Bar")];
        assert_eq!(unsupported_scoring_ops(&collide).1, 2);

        ops = (0..20).map(|i| op("", &format!("X{i:02}"))).collect();
        let (names, total) = unsupported_scoring_ops(&ops);
        assert_eq!((names.len(), total), (16, 20));
        assert_eq!(names.first().map(String::as_str), Some("X00"));
        assert_eq!(names.last().map(String::as_str), Some("X15"));
    }

    /// REQ-41・REQ-39: トークナイザーは対応範囲外だけが除外で、他は全体停止。
    #[test]
    fn req41_classify_tokenizer_error_mapping() {
        let unsupported = TokenizerLoadError::Unsupported {
            section: "model",
            supported: "x",
        };
        assert_eq!(
            classify_tokenizer_error(&unsupported),
            LoadErrorClass::Exclude("model")
        );
        for e in [
            TokenizerLoadError::IntegrityMismatch,
            TokenizerLoadError::MalformedJson,
            TokenizerLoadError::LimitExceeded { what: "x" },
            TokenizerLoadError::Invalid { what: "x" },
        ] {
            assert_eq!(classify_tokenizer_error(&e), LoadErrorClass::Propagate);
        }
    }

    /// REQ-41・#433: 新しい除外コード・種類名は固定の ASCII snake_case。
    #[test]
    fn req41_new_codes_fixed() {
        assert_eq!(
            ExclusionReason::UnsupportedOps {
                ops: vec![],
                total: 0
            }
            .code(),
            "unsupported_ops"
        );
        assert_eq!(
            ExclusionReason::UnsupportedTokenizer { section: "model" }.code(),
            "unsupported_tokenizer"
        );
        assert_eq!(ExportKind::Qwen2Lora.as_str(), "qwen2_lora");
        assert_eq!(ExportKind::Qwen2Lora.model_kind(), None);
        assert_eq!(ExportKind::from(ModelKind::C3), ExportKind::C3);
    }

    /// REQ-32: 推論エラーのコードは ScreeningError にそのまま出る。
    #[test]
    fn req32_screening_error_codes() {
        assert_eq!(ScreeningError::TooManyCases.code(), "limit_exceeded");
        assert_eq!(
            ScreeningError::Infer(InferError::Backend(BackendError::Failed)).code(),
            InferError::Backend(BackendError::Failed).code()
        );
        assert_eq!(
            ScreeningError::Load(OnnxLoadError::IntegrityMismatch).code(),
            "integrity_mismatch"
        );
    }
}
