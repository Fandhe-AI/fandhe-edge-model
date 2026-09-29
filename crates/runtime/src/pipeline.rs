//! 単体推論・バッチ推論が共通で通る 1 系列の推論経路（REQ-28・TASK-28.1-1・#117）。
//!
//! # 役割
//!
//! PoC-16 では、バッチ内の最大長へ合わせたパディング位置の埋め込みが畳み込みに混入し、
//! バッチ推論が 1 件ずつの推論と 650 件中 2 件で予測ラベルを食い違えた。本モジュールは
//! 「バッチ推論を、単体推論と同じ 1 系列経路の繰り返しとして実装する」ことで、系列間で
//! パディング・状態を共有できない構造にする。
//!
//! - [`Preprocessor`]・[`ScoringBackend`] は 1 系列だけを受け取る（バッチ次元を型に持たない）
//! - [`InferencePipeline::infer_one`]・[`InferencePipeline::infer_batch`] はともに
//!   private な `run_single` だけを呼ぶ（バッチ専用の前処理・スコア計算コードを持たない）
//!
//! # 継ぎ目（スタブ）
//!
//! - 前処理: #112（TASK-32.1 配下）のバイト前処理（NFKC 正規化＋バイトエンコード）が
//!   [`Preprocessor`] を実装して差し込む想定。本 crate は未実装（REQ-32）
//! - スコア計算: #113 の ONNX 推論が [`ScoringBackend`] を実装する想定。ONNX Runtime の
//!   セッション API が `&mut` を要求する場合の内部可変性の扱いは #113 で判断する
//! - 実装は呼び出し間で結果に影響する状態を持たないこと（REQ-28）。trait は `&self` だが
//!   内部可変性は型で塞げないため契約として明記する
//!
//! # 終了コードとの関係
//!
//! エラーの 7 種終了コードへの写像は CLI 側（TASK-33.x）の責務で、本モジュールは
//! 機械可読な `code()` 文字列のみ返す。エラー・`Debug` は入力本文やトークン値を保持しない
//! （security.md）。

use fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES;
use std::fmt;

/// 1 回のバッチ推論で受け付ける件数の上限（暫定値。REQ-39 の資源上限が正式に決まるまで暫定）。
/// `Vec` の確保前に検査する。
pub const MAX_INFER_BATCH_LEN: usize = 100_000;

/// 前処理後のトークン列（1 系列分。バッチ次元を持たない）。
#[derive(Clone, PartialEq, Eq)]
pub struct TokenIds(Vec<i64>);

impl TokenIds {
    /// トークン列から作る。
    pub fn new(ids: Vec<i64>) -> Self {
        Self(ids)
    }

    /// トークン列を参照する。
    pub fn as_slice(&self) -> &[i64] {
        &self.0
    }
}

impl fmt::Debug for TokenIds {
    /// トークン値を出さず長さのみ表示する（security.md）。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TokenIds(len={})", self.0.len())
    }
}

/// 前処理の継ぎ目（#112 が実装する想定。本 crate では未実装）。
///
/// 実装は呼び出し間で結果に影響する状態を持たないこと（REQ-28）。
pub trait Preprocessor {
    /// 入力 1 件をトークン列へ変換する。
    fn preprocess(&self, input: &str) -> Result<TokenIds, PreprocessError>;
}

/// スコア計算の継ぎ目（#113 の ONNX 推論が実装する想定）。
///
/// 引数は 1 系列のみで、系列間パディングを型の上で表現できない。実装は呼び出し間で
/// 結果に影響する状態を持たないこと（REQ-28）。
pub trait ScoringBackend {
    /// 1 系列のスコア列（選択肢の宣言順）を返す。
    fn scores(&self, ids: &TokenIds) -> Result<Vec<f64>, BackendError>;
}

/// 前処理の失敗（本文を保持しない）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PreprocessError {
    /// 前処理を実行できない。
    Failed,
}

impl PreprocessError {
    /// 機械可読なエラーコード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::Failed => "preprocess_failed",
        }
    }
}

/// スコア計算の失敗（本文を保持しない）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackendError {
    /// スコア計算を実行できない。
    Failed,
}

impl BackendError {
    /// 機械可読なエラーコード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::Failed => "backend_failed",
        }
    }
}

/// 1 件の推論の失敗。長さ・上限のみ保持する。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InferError {
    /// 入力が [`MAX_INFER_INPUT_BYTES`] を超える（前処理前に検査。REQ-39）。
    InputTooLarge {
        /// 入力のバイト長。
        len: usize,
        /// 上限。
        limit: usize,
    },
    /// 前処理の失敗。
    Preprocess(PreprocessError),
    /// スコア計算の失敗。
    Backend(BackendError),
    /// スコアが空、または非有限値を含む。
    InvalidScores,
}

impl InferError {
    /// 機械可読なエラーコード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::InputTooLarge { .. } => "input_too_large",
            Self::Preprocess(e) => e.code(),
            Self::Backend(e) => e.code(),
            Self::InvalidScores => "invalid_scores",
        }
    }
}

/// バッチ全体の失敗。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchError {
    /// 件数が [`MAX_INFER_BATCH_LEN`] を超える。
    TooManyInputs {
        /// 渡された件数。
        len: usize,
        /// 上限。
        limit: usize,
    },
}

impl BatchError {
    /// 機械可読なエラーコード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::TooManyInputs { .. } => "too_many_inputs",
        }
    }
}

/// 1 件の予測。ラベル ID への写像は CLI 側の責務で、ここでは宣言順の index を返す。
#[derive(Clone, PartialEq)]
pub struct Prediction {
    label_index: usize,
    scores: Vec<f64>,
}

impl Prediction {
    /// 最高スコアの選択肢の index（宣言順。同点は先頭）。
    pub fn label_index(&self) -> usize {
        self.label_index
    }

    /// スコア列。
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }
}

impl fmt::Debug for Prediction {
    /// 値を出さず件数のみ表示する。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Prediction(scores_len={})", self.scores.len())
    }
}

/// バッチ推論の戻り値。要素ごとの成否を持つ。
pub type BatchResult = Vec<Result<Prediction, InferError>>;

/// 前処理とスコア計算を束ねた推論パイプライン。呼び出し間で状態を持たない。
pub struct InferencePipeline<P, B> {
    preprocessor: P,
    backend: B,
}

impl<P: Preprocessor, B: ScoringBackend> InferencePipeline<P, B> {
    /// 前処理とバックエンドから作る。
    pub fn new(preprocessor: P, backend: B) -> Self {
        Self {
            preprocessor,
            backend,
        }
    }

    /// 単体推論。`run_single` を呼ぶだけ。
    pub fn infer_one(&self, input: &str) -> Result<Prediction, InferError> {
        self.run_single(input)
    }

    /// バッチ推論。件数上限を確保前に検査し、1 件ずつ `run_single` を呼ぶだけ。
    /// 1 件の失敗は他要素へ波及しない。
    pub fn infer_batch(&self, inputs: &[&str]) -> Result<BatchResult, BatchError> {
        if inputs.len() > MAX_INFER_BATCH_LEN {
            return Err(BatchError::TooManyInputs {
                len: inputs.len(),
                limit: MAX_INFER_BATCH_LEN,
            });
        }
        Ok(inputs.iter().map(|x| self.run_single(x)).collect())
    }

    /// `run_single` と同義の関数（評価器の推論関数へ渡す用。#118）。
    pub fn as_predict_fn(&self) -> impl Fn(&str) -> Result<Prediction, InferError> + '_ {
        move |input| self.run_single(input)
    }

    /// 単体・バッチ共通の唯一の経路。
    fn run_single(&self, input: &str) -> Result<Prediction, InferError> {
        if input.len() > MAX_INFER_INPUT_BYTES {
            return Err(InferError::InputTooLarge {
                len: input.len(),
                limit: MAX_INFER_INPUT_BYTES,
            });
        }
        let ids = self
            .preprocessor
            .preprocess(input)
            .map_err(InferError::Preprocess)?;
        let scores = self.backend.scores(&ids).map_err(InferError::Backend)?;
        let label_index = argmax(&scores).ok_or(InferError::InvalidScores)?;
        Ok(Prediction {
            label_index,
            scores,
        })
    }
}

/// 最大値の index。厳密比較・同点は先頭優先（`JudgmentResult::new` と同じ規則）。
/// 空・非有限値を含む場合は `None`。
fn argmax(scores: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &s) in scores.iter().enumerate() {
        if !s.is_finite() {
            return None;
        }
        match best {
            Some((_, b)) if s <= b => {}
            _ => best = Some((i, s)),
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-28: 同点は先頭優先。
    #[test]
    fn req28_argmax_tie_prefers_first() {
        assert_eq!(argmax(&[0.5, 0.5, 0.1]), Some(0));
        assert_eq!(argmax(&[0.1, 0.7, 0.7]), Some(1));
    }

    /// REQ-28: 空・NaN・無限大は拒否。
    #[test]
    fn req28_argmax_rejects_invalid() {
        assert_eq!(argmax(&[]), None);
        assert_eq!(argmax(&[0.1, f64::NAN]), None);
        assert_eq!(argmax(&[f64::INFINITY, 0.1]), None);
    }

    /// エラーコードは英語の機械可読文字列。
    #[test]
    fn error_codes() {
        assert_eq!(InferError::InvalidScores.code(), "invalid_scores");
        assert_eq!(
            InferError::InputTooLarge { len: 2, limit: 1 }.code(),
            "input_too_large"
        );
        assert_eq!(
            BatchError::TooManyInputs { len: 2, limit: 1 }.code(),
            "too_many_inputs"
        );
    }

    /// Debug は値を出さない。
    #[test]
    fn debug_hides_values() {
        assert_eq!(
            format!("{:?}", TokenIds::new(vec![7, 8])),
            "TokenIds(len=2)"
        );
    }
}
