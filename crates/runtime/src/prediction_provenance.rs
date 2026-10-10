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
//! - [`ReferenceBatchPredictions`] は結果と注記を分離できない形で束ねる。結果は要素ごとに注記と一体の
//!   [`AnnotatedPrediction`] としてのみ読め、結果だけを返す公開アクセサは持たない
//! - 注記なしの `InferencePipeline::infer_batch` は判定経路と共通の入口（TASK-28.1）で、
//!   参考測定の記録を作る側は `infer_batch_for_reference` を使う（記録生成の入口はこちらに一本化）
//!
//! # 呼び出し元と範囲外
//!
//! - CLI に参考測定なし。規則は型とテストで固定（#493）。CLI のバッチ API 呼び出しは判定経路
//!   （`infer --input-file`）だけで、`infer_batch_for_reference` の呼び出しは 0 件。これを
//!   `crates/cli/tests/batch_prediction_sites.rs` が機械照合する（明記すべき記録が存在しないことの照合）
//! - 今後 CLI が参考測定にバッチ API を使うときは `infer_batch_for_reference` を通し、その記録 JSON に
//!   `"batch_prediction":true`（キーは [`BATCH_PREDICTION_FIELD`]）と `"batch_prediction_notice"`
//!   （値は [`BATCH_PREDICTION_NOTICE`]）を出す。キー・注記は本モジュールの定数を SSOT とし、直列化は
//!   CLI 側で行う（`latency_report::P95_METHOD` と同じ分担。runtime は serde に依存しない）
//! - 範囲外（未実装。実装済みを装わない）: 評価器・診断側の参考測定そのもの、
//!   不一致発見時の原因特定プロセス（TASK-28.2・#119）
//!
//! 証拠種別: テストハーネス（`tests/batch_prediction_notice.rs`）。

use crate::pipeline::{BatchResult, InferError, Prediction};
use std::fmt;

/// JSON へ出す際のフラグのキー名（CLI が直列化に使う。#493）。
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

    /// 件数（結果本体は出さない。結果は [`Self::into_annotated_results`] 経由でのみ読める）。
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// 件数が 0 か。
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
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
///
/// フィールドは非公開。呼び出し側が `results` だけを取り出したり、出どころ・注記を
/// 書き換えたりできないよう、生成は本モジュール内（`into_annotated_results`）に限り、
/// 読み出しはアクセサ経由にする（REQ-28・TASK-28.3）。
pub struct AnnotatedBatchResults {
    results: BatchResult,
    provenance: PredictionProvenance,
    notice: &'static str,
}

impl AnnotatedBatchResults {
    /// 件数。
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// 件数が 0 か。
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// 要素ごとの記録。各要素が結果・出どころ・注記を一体で持つ。
    /// 結果だけを返す公開アクセサは持たない（REQ-28・TASK-28.3）。
    pub fn iter(&self) -> impl Iterator<Item = AnnotatedPrediction<'_>> + '_ {
        self.results.iter().map(|result| AnnotatedPrediction {
            result,
            provenance: self.provenance,
            notice: self.notice,
        })
    }

    /// 常に参考測定・バッチ。
    pub fn provenance(&self) -> PredictionProvenance {
        self.provenance
    }

    /// 記録へ付ける注記（常に [`BATCH_PREDICTION_NOTICE`]）。
    pub fn notice(&self) -> &'static str {
        self.notice
    }
}

/// 1 件分の記録。結果・出どころ・注記を一体にした借用で、本モジュール外からは作れない。
/// 結果を読む [`Self::result`] と同じ値から、必ず注記も取り出せる（REQ-28・TASK-28.3）。
#[derive(Clone, Copy)]
pub struct AnnotatedPrediction<'a> {
    result: &'a Result<Prediction, InferError>,
    provenance: PredictionProvenance,
    notice: &'static str,
}

impl<'a> AnnotatedPrediction<'a> {
    /// この要素の予測結果。
    pub fn result(&self) -> &'a Result<Prediction, InferError> {
        self.result
    }

    /// 常に参考測定・バッチ。
    pub fn provenance(&self) -> PredictionProvenance {
        self.provenance
    }

    /// 記録へ付ける注記（常に [`BATCH_PREDICTION_NOTICE`]）。
    pub fn notice(&self) -> &'static str {
        self.notice
    }

    /// `batch_prediction` フラグの値（常に true）。
    pub fn is_batch_prediction(&self) -> bool {
        self.provenance.is_batch_prediction()
    }
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
        assert_eq!(out.len(), 1);
        assert!(
            out.iter()
                .all(|r| r.notice() == BATCH_PREDICTION_NOTICE && r.is_batch_prediction())
        );
        assert_eq!(out.notice(), BATCH_PREDICTION_NOTICE);
        assert_eq!(
            out.provenance(),
            PredictionProvenance::Reference {
                mode: PredictionMode::Batch
            }
        );
        assert!(out.provenance().is_batch_prediction());
    }

    #[test]
    fn req28_notice_wording() {
        assert!(BATCH_PREDICTION_NOTICE.contains("REQ-28"));
        assert!(BATCH_PREDICTION_NOTICE.contains("not used for judgment"));
    }
}
