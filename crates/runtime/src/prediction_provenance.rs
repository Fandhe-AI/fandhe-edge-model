//! 予測の出どころ（判定用か参考測定用か・バッチ API か）の記録ルール
//! （REQ-28 境界値・TASK-28.3・#120。根拠: PoC-12 `abstention-calibration` のレビュー運用）。
//!
//! # 役割
//!
//! 判定（合否・評価器の指標）に使う予測は、TASK-28.1（#117・#118）で単体とバッチの全件一致を
//! 保証した経路を通る。一方、判定に使わない参考測定でバッチ予測 API を使った場合は、
//! 「バッチ予測である」ことを記録に明記する。PoC-12 では、判定は 1 件ずつの予測で行い、
//! 参考測定（ood_mix）のみバッチ API を使って、それを記録に明記することで判定への影響が
//! ないことを示した。本モジュールはその規則を型で固定する。
//!
//! - 明記が必要な組み合わせの規則は [`PredictionProvenance::batch_prediction_notice`] の 1 箇所に集約する
//! - [`PredictionProvenance::Judgment`] は mode を持たない。判定経路にバッチのフラグを付けること
//!   自体を型で表現できない（判定経路にフラグは不要）
//! - [`ReferenceBatchPredictions`] は結果と注記を分離できない形で束ねる
//!
//! # 呼び出し元と範囲外
//!
//! - 呼び出し元（想定）: 参考測定を行う評価・診断側、CLI の JSON 出力（TASK-33.x）。
//!   JSON のキー・注記は本モジュールの定数を SSOT とし、直列化は CLI 側で行う
//!   （`latency_report::P95_METHOD` と同じ分担。runtime は serde に依存しない）
//! - 範囲外（未実装。実装済みを装わない）: CLI 出力への `batch_prediction` の配線（TASK-33.x）、
//!   評価器・診断側の参考測定そのもの、不一致発見時の原因特定プロセス（TASK-28.2・#119）
//!
//! 証拠種別: テストハーネス（`tests/batch_prediction_notice.rs`）。

use crate::pipeline::{BatchResult, InferError, Prediction};
use std::fmt;

/// JSON へ出す際のフラグのキー名（CLI が直列化に使う。TASK-33.x）。
pub const BATCH_PREDICTION_FIELD: &str = "batch_prediction";

/// バッチ予測を参考測定に使った記録へ付ける注記（機械処理向けに英語）。
pub const BATCH_PREDICTION_NOTICE: &str = "predictions produced by the batch prediction API for reference measurement; not used for judgment (REQ-28)";

/// 予測 API の呼び方。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionMode {
    /// 1 件ずつの予測。
    Single,
    /// バッチ予測 API。
    Batch,
}

impl PredictionMode {
    /// 機械可読なコード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Batch => "batch",
        }
    }
}

/// 予測の用途。網羅 match を呼び出し側に強制するため `non_exhaustive` にしない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionProvenance {
    /// 判定（合否・評価器の指標）に使う予測。TASK-28.1 で単体とバッチの一致を保証した経路で、
    /// バッチのフラグは不要。
    Judgment,
    /// 判定に使わない参考測定の予測。
    Reference {
        /// 予測 API の呼び方。
        mode: PredictionMode,
    },
}

impl PredictionProvenance {
    /// バッチ予測の明記が必要なときだけ注記を返す（規則の唯一の定義）。
    /// 参考測定でバッチ API を使った場合のみ `Some`。
    pub fn batch_prediction_notice(&self) -> Option<&'static str> {
        match self {
            Self::Reference {
                mode: PredictionMode::Batch,
            } => Some(BATCH_PREDICTION_NOTICE),
            Self::Reference {
                mode: PredictionMode::Single,
            }
            | Self::Judgment => None,
        }
    }

    /// `batch_prediction` フラグの値。注記の有無と同じ条件。
    pub fn is_batch_prediction(&self) -> bool {
        self.batch_prediction_notice().is_some()
    }
}

/// 参考測定でバッチ API を使った予測。結果と注記を分離できない。
pub struct ReferenceBatchPredictions {
    results: BatchResult,
}

impl ReferenceBatchPredictions {
    /// `InferencePipeline::infer_batch_for_reference` からのみ作る。
    pub(crate) fn new(results: BatchResult) -> Self {
        Self { results }
    }

    /// 要素ごとの結果。
    pub fn results(&self) -> &[Result<Prediction, InferError>] {
        &self.results
    }

    /// 結果を出どころ・注記つきで取り出す。結果だけを注記なしで取り出す API は提供しない
    /// （記録から `batch_prediction` の明記が落ちるのを防ぐ。REQ-28・TASK-28.3）。
    pub fn into_annotated_results(self) -> AnnotatedBatchResults {
        AnnotatedBatchResults {
            results: self.results,
            provenance: PredictionProvenance::Reference {
                mode: PredictionMode::Batch,
            },
            notice: BATCH_PREDICTION_NOTICE,
        }
    }

    /// 常に参考測定・バッチ。
    pub fn provenance(&self) -> PredictionProvenance {
        PredictionProvenance::Reference {
            mode: PredictionMode::Batch,
        }
    }

    /// 記録へ付ける注記（常に [`BATCH_PREDICTION_NOTICE`]）。
    pub fn notice(&self) -> &'static str {
        BATCH_PREDICTION_NOTICE
    }
}

/// [`ReferenceBatchPredictions::into_annotated_results`] の戻り値。
/// 取り出した結果に、バッチ API を使った出どころと注記が常に付く。
pub struct AnnotatedBatchResults {
    /// 要素ごとの結果。
    pub results: BatchResult,
    /// 常に参考測定・バッチ。
    pub provenance: PredictionProvenance,
    /// 記録へ付ける注記（常に [`BATCH_PREDICTION_NOTICE`]）。
    pub notice: &'static str,
}

/// スコア・ラベル・入力本文を出さず、件数と用途のみ出す（security.md）。
impl fmt::Debug for ReferenceBatchPredictions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ReferenceBatchPredictions(len={}, provenance={:?})",
            self.results.len(),
            self.provenance()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn req28_reference_batch_has_notice() {
        let p = PredictionProvenance::Reference {
            mode: PredictionMode::Batch,
        };
        assert_eq!(p.batch_prediction_notice(), Some(BATCH_PREDICTION_NOTICE));
        assert!(p.is_batch_prediction());
    }

    #[test]
    fn req28_reference_single_has_no_notice() {
        let p = PredictionProvenance::Reference {
            mode: PredictionMode::Single,
        };
        assert_eq!(p.batch_prediction_notice(), None);
        assert!(!p.is_batch_prediction());
    }

    #[test]
    fn req28_judgment_has_no_notice() {
        let p = PredictionProvenance::Judgment;
        assert_eq!(p.batch_prediction_notice(), None);
        assert!(!p.is_batch_prediction());
    }

    #[test]
    fn req28_mode_codes_and_field() {
        assert_eq!(PredictionMode::Single.code(), "single");
        assert_eq!(PredictionMode::Batch.code(), "batch");
        assert_eq!(BATCH_PREDICTION_FIELD, "batch_prediction");
    }

    #[test]
    fn req28_into_annotated_results_keeps_provenance() {
        let batch: BatchResult = vec![Err(InferError::InputTooLarge { len: 2, limit: 1 })];
        let out = ReferenceBatchPredictions::new(batch).into_annotated_results();
        assert_eq!(out.results.len(), 1);
        assert_eq!(out.notice, BATCH_PREDICTION_NOTICE);
        assert_eq!(
            out.provenance,
            PredictionProvenance::Reference {
                mode: PredictionMode::Batch
            }
        );
        assert!(out.provenance.is_batch_prediction());
    }

    #[test]
    fn req28_notice_wording() {
        assert!(BATCH_PREDICTION_NOTICE.contains("REQ-28"));
        assert!(BATCH_PREDICTION_NOTICE.contains("not used for judgment"));
    }
}
