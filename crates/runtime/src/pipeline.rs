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
//! - 前処理: #112（TASK-32.1 配下）のバイト前処理（NFKC 正規化＋バイトエンコード）は
//!   [`crate::preprocess::ByteEncodingPreprocessor`] が [`Preprocessor`] を実装する（REQ-32）
//! - スコア計算: [`crate::onnx::OnnxBackend`]（#113。自作の ONNX 推論）が [`ScoringBackend`] を
//!   実装する。`&self` の純粋計算で内部可変性を持たない
//! - 実装は呼び出し間で結果に影響する状態を持たないこと（REQ-28）。trait は `&self` だが
//!   内部可変性は型で塞げないため契約として明記する
//!
//! # 終了コードとの関係
//!
//! エラーの 7 種終了コードへの写像は CLI 側（TASK-33.x）の責務で、本モジュールは
//! 機械可読な `code()` 文字列のみ返す。エラー・`Debug` は入力本文やトークン値を保持しない
//! （security.md）。

use fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES;
use fandhe_edge_core::judgment::{MAX_OPTIONS, SCORE_SUM_TOLERANCE};
use std::fmt;

/// 1 回のバッチ推論で受け付ける件数の上限（暫定値。REQ-39 の資源上限が正式に決まるまで暫定）。
/// `Vec` の確保前に検査する。
pub const MAX_INFER_BATCH_LEN: usize = 100_000;

/// 1 回のバッチ推論の総入力バイト数の上限（暫定値。REQ-39）。
/// 件数上限だけでは 100,000 件 × 1 MiB で約 100 GB になるため、処理前に総量を検査する。
pub const MAX_INFER_BATCH_TOTAL_BYTES: usize = 64 * 1024 * 1024;

/// 前処理後のトークン数の上限（暫定値。REQ-39）。NFKC 正規化による膨張を見込み、
/// 入力上限 [`MAX_INFER_INPUT_BYTES`] の 4 倍とする。バックエンド呼び出し前に検査する。
pub const MAX_INFER_TOKEN_IDS: usize = 4 * MAX_INFER_INPUT_BYTES;

/// 1 件の予測が持てるスコア数の上限（選択肢数の上限 `MAX_OPTIONS` と同じ。REQ-39）。
pub const MAX_INFER_SCORES: usize = MAX_OPTIONS;

/// 1 回のバッチ推論で保持する結果スコアの総数の上限（暫定値。REQ-39。f64 で約 64 MiB）。
pub const MAX_INFER_BATCH_TOTAL_SCORES: usize = 8 * 1024 * 1024;

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

/// 前処理の継ぎ目（#112 の [`crate::preprocess::ByteEncodingPreprocessor`] が実装する）。
///
/// 実装は呼び出し間で結果に影響する状態を持たないこと（REQ-28）。
pub trait Preprocessor {
    /// 入力 1 件をトークン列へ変換する。
    fn preprocess(&self, input: &str) -> Result<TokenIds, PreprocessError>;
}

/// スコア計算の継ぎ目（[`crate::onnx::OnnxBackend`] が実装する。#113）。
///
/// 引数は 1 系列のみで、系列間パディングを型の上で表現できない。実装は呼び出し間で
/// 結果に影響する状態を持たないこと（REQ-28）。
pub trait ScoringBackend {
    /// 1 系列のスコア列（選択肢の宣言順）を返す。
    fn scores(&self, ids: &TokenIds) -> Result<Vec<f64>, BackendError>;

    /// 計算時間を `limit` 以内に打ち切る版（超過は [`BackendError::TimeLimitExceeded`]。REQ-39）。
    ///
    /// 既定実装は置かない。時間上限を強制できないバックエンドが `scores` へ委譲して
    /// `InferencePipeline::infer_one_within` の上限を黙って無効化することを防ぐため、
    /// 各実装が打ち切りを実装するか、保証できない場合は明示的にエラーを返す。
    fn scores_limited(
        &self,
        ids: &TokenIds,
        limit: std::time::Duration,
    ) -> Result<Vec<f64>, BackendError>;
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
    /// トークン列が空、または系列長がモデルの上限を超える（ONNX バックエンド。REQ-39）。
    InvalidSequenceLength,
    /// トークン値が語彙の範囲（`0..257`）外（ONNX バックエンド。REQ-39）。
    InvalidTokenId,
    /// 1 件の計算時間が上限を超えたため打ち切った（ONNX バックエンド。REQ-39）。
    TimeLimitExceeded,
}

impl BackendError {
    /// 機械可読なエラーコード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::Failed => "backend_failed",
            Self::InvalidSequenceLength => "invalid_sequence_length",
            Self::InvalidTokenId => "invalid_token_id",
            Self::TimeLimitExceeded => "time_limit_exceeded",
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
    /// 前処理後のトークン数が [`MAX_INFER_TOKEN_IDS`] を超える（バックエンド呼び出し前に検査。REQ-39）。
    TooManyTokens {
        /// トークン数。
        len: usize,
        /// 上限。
        limit: usize,
    },
    /// バックエンドが返したスコア数が [`MAX_INFER_SCORES`] を超える（REQ-39）。
    TooManyScores {
        /// スコア数。
        len: usize,
        /// 上限。
        limit: usize,
    },
    /// スコアが確率として不正（空・非有限・`[0, 1]` 外・合計が 1 から許容差超のずれ）。
    InvalidScores,
}

impl InferError {
    /// 機械可読なエラーコード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::InputTooLarge { .. } => "input_too_large",
            Self::Preprocess(e) => e.code(),
            Self::Backend(e) => e.code(),
            Self::TooManyTokens { .. } => "too_many_tokens",
            Self::TooManyScores { .. } => "too_many_scores",
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
    /// 総入力バイト数が [`MAX_INFER_BATCH_TOTAL_BYTES`] を超える（処理前に検査）。
    TotalInputTooLarge {
        /// 総入力バイト数（飽和加算）。
        total: usize,
        /// 上限。
        limit: usize,
    },
    /// 結果として保持するスコアの総数が [`MAX_INFER_BATCH_TOTAL_SCORES`] を超えた。
    /// 部分結果は返さず、バッチ全体を失敗とする。
    ResultTooLarge {
        /// 上限。
        limit: usize,
    },
}

impl BatchError {
    /// 機械可読なエラーコード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::TooManyInputs { .. } => "too_many_inputs",
            Self::TotalInputTooLarge { .. } => "total_input_too_large",
            Self::ResultTooLarge { .. } => "result_too_large",
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
        self.run_single(input, None)
    }

    /// 単体推論。スコア計算を `limit` 以内に打ち切る（超過は `Backend(TimeLimitExceeded)`。REQ-39）。
    pub fn infer_one_within(
        &self,
        input: &str,
        limit: std::time::Duration,
    ) -> Result<Prediction, InferError> {
        self.run_single(input, Some(limit))
    }

    /// バッチ推論。件数・総入力バイト数を処理前に検査し、1 件ずつ `run_single` を呼ぶだけ。
    /// 結果スコアの総数が上限を超えたら、以降を処理せずバッチ全体を失敗とする。
    /// 1 件の失敗は他要素へ波及しない。
    pub fn infer_batch(&self, inputs: &[&str]) -> Result<BatchResult, BatchError> {
        if inputs.len() > MAX_INFER_BATCH_LEN {
            return Err(BatchError::TooManyInputs {
                len: inputs.len(),
                limit: MAX_INFER_BATCH_LEN,
            });
        }
        let total = inputs
            .iter()
            .fold(0usize, |acc, x| acc.saturating_add(x.len()));
        if total > MAX_INFER_BATCH_TOTAL_BYTES {
            return Err(BatchError::TotalInputTooLarge {
                total,
                limit: MAX_INFER_BATCH_TOTAL_BYTES,
            });
        }
        let mut results = Vec::with_capacity(inputs.len());
        let mut retained_scores = 0usize;
        for input in inputs {
            let result = self.run_single(input, None);
            if let Ok(p) = &result {
                retained_scores = retained_scores.saturating_add(p.scores.len());
                if retained_scores > MAX_INFER_BATCH_TOTAL_SCORES {
                    return Err(BatchError::ResultTooLarge {
                        limit: MAX_INFER_BATCH_TOTAL_SCORES,
                    });
                }
            }
            results.push(result);
        }
        Ok(results)
    }

    /// `run_single` と同義の関数（評価器の推論関数へ渡す用。#118）。
    pub fn as_predict_fn(&self) -> impl Fn(&str) -> Result<Prediction, InferError> + '_ {
        move |input| self.run_single(input, None)
    }

    /// 単体・バッチ共通の唯一の経路。
    fn run_single(
        &self,
        input: &str,
        limit: Option<std::time::Duration>,
    ) -> Result<Prediction, InferError> {
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
        if ids.as_slice().len() > MAX_INFER_TOKEN_IDS {
            return Err(InferError::TooManyTokens {
                len: ids.as_slice().len(),
                limit: MAX_INFER_TOKEN_IDS,
            });
        }
        let scores = match limit {
            Some(l) => self.backend.scores_limited(&ids, l),
            None => self.backend.scores(&ids),
        }
        .map_err(InferError::Backend)?;
        if scores.len() > MAX_INFER_SCORES {
            return Err(InferError::TooManyScores {
                len: scores.len(),
                limit: MAX_INFER_SCORES,
            });
        }
        let label_index = argmax(&scores).ok_or(InferError::InvalidScores)?;
        Ok(Prediction {
            label_index,
            scores,
        })
    }
}

/// 最大値の index。厳密比較・同点は先頭優先（`JudgmentResult::new` と同じ規則）。
/// 空・非有限値・`[0, 1]` 外を含む場合、または合計が 1 から `SCORE_SUM_TOLERANCE` を
/// 超えてずれる場合は `None`（`JudgmentResult::new` と同じ確率条件）。
fn argmax(scores: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    let mut sum = 0.0f64;
    for (i, &s) in scores.iter().enumerate() {
        if !s.is_finite() || !(0.0..=1.0).contains(&s) {
            return None;
        }
        sum += s;
        match best {
            Some((_, b)) if s <= b => {}
            _ => best = Some((i, s)),
        }
    }
    if (sum - 1.0).abs() > SCORE_SUM_TOLERANCE {
        return None;
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-28: 同点は先頭優先。
    #[test]
    fn req28_argmax_tie_prefers_first() {
        assert_eq!(argmax(&[0.4, 0.4, 0.2]), Some(0));
        assert_eq!(argmax(&[0.1, 0.45, 0.45]), Some(1));
    }

    /// REQ-28: 空・NaN・無限大は拒否。
    #[test]
    fn req28_argmax_rejects_invalid() {
        assert_eq!(argmax(&[]), None);
        assert_eq!(argmax(&[0.1, f64::NAN]), None);
        assert_eq!(argmax(&[f64::INFINITY, 0.1]), None);
    }

    /// REQ-28: 範囲外・合計不正は拒否し、許容差内の合計は受理する。
    #[test]
    fn req28_argmax_rejects_non_probability() {
        assert_eq!(argmax(&[-0.5, 1.5]), None);
        assert_eq!(argmax(&[0.9, 0.9]), None);
        assert_eq!(argmax(&[0.2, 0.2]), None);
        assert_eq!(argmax(&[0.5 + 5e-7, 0.5]), Some(0));
        assert_eq!(argmax(&[0.5 + 2e-6, 0.5]), None);
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
        assert_eq!(
            InferError::TooManyTokens { len: 2, limit: 1 }.code(),
            "too_many_tokens"
        );
        assert_eq!(
            BatchError::TotalInputTooLarge { total: 2, limit: 1 }.code(),
            "total_input_too_large"
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
