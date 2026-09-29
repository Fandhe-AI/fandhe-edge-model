//! バッチ予測利用時の明記ルールの結合テスト（REQ-28 境界値・TASK-28.3・#120）。
//!
//! 証拠種別: テストハーネス（模擬の前処理・模擬バックエンド）。実機・E2E ではない。

use fandhe_edge_eval::input_only::{EvalItem, run_inference_input_only};
use fandhe_edge_runtime::pipeline::*;
use fandhe_edge_runtime::prediction_provenance::*;
use std::rc::Rc;

const TOL: f64 = 1e-9;

struct MockPre;
impl Preprocessor for MockPre {
    fn preprocess(&self, input: &str) -> Result<TokenIds, PreprocessError> {
        if input.is_empty() {
            return Ok(TokenIds::new(vec![0]));
        }
        Ok(TokenIds::new(
            input.bytes().map(|b| i64::from(b) + 1).collect(),
        ))
    }
}

struct MockBackend;
impl ScoringBackend for MockBackend {
    fn scores(&self, ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        let s = ids.as_slice();
        let n = s.len() as f64;
        let sum: i64 = s.iter().sum();
        let a = ((sum as f64) / n).sin().exp();
        let b = n.cos().exp();
        let c = 1.0;
        let t = a + b + c;
        Ok(vec![a / t, b / t, c / t])
    }
}

fn pipeline() -> InferencePipeline<MockPre, MockBackend> {
    InferencePipeline::new(MockPre, MockBackend)
}

fn inputs() -> Vec<String> {
    let mut v = vec![String::new()];
    for i in 0..40 {
        v.push(format!("sample-{i}-{}", "x".repeat(i % 7)));
    }
    v
}

fn pred(r: &Result<Prediction, InferError>, i: usize) -> &Prediction {
    match r {
        Ok(p) => p,
        Err(e) => panic!("inference failed at index {i}: code={}", e.code()),
    }
}

#[test]
fn req28_reference_batch_record_carries_notice() {
    let data = inputs();
    let refs: Vec<&str> = data.iter().map(String::as_str).collect();
    let out = pipeline()
        .infer_batch_for_reference(&refs)
        .expect("within limits");
    assert_eq!(
        out.provenance(),
        PredictionProvenance::Reference {
            mode: PredictionMode::Batch
        }
    );
    assert_eq!(out.notice(), BATCH_PREDICTION_NOTICE);
    assert!(out.provenance().is_batch_prediction());
}

#[test]
fn req28_reference_batch_predictions_equal_infer_batch() {
    let data = inputs();
    let refs: Vec<&str> = data.iter().map(String::as_str).collect();
    let pl = pipeline();
    let plain = pl.infer_batch(&refs).expect("within limits");
    let wrapped = pl.infer_batch_for_reference(&refs).expect("within limits");
    assert_eq!(plain.len(), 41);
    assert_eq!(wrapped.results().len(), 41);
    for (i, (a, b)) in plain.iter().zip(wrapped.results()).enumerate() {
        let (a, b) = (pred(a, i), pred(b, i));
        assert_eq!(a.label_index(), b.label_index(), "label differs at {i}");
        assert_eq!(a.scores().len(), b.scores().len());
        for (p, q) in a.scores().iter().zip(b.scores()) {
            assert!((p - q).abs() <= TOL, "score differs at {i}");
        }
    }
}

#[test]
fn req28_judgment_path_needs_no_batch_flag() {
    let data = inputs();
    let pl = Rc::new(pipeline());
    let q = Rc::clone(&pl);
    let tags: Vec<String> = Vec::new();
    let ids: Vec<String> = (0..data.len()).map(|i| format!("r{i}")).collect();
    let items: Vec<EvalItem<'_>> = data
        .iter()
        .zip(&ids)
        .map(|(x, id)| EvalItem {
            id,
            input: x,
            gold: "dummy",
            tags: &tags,
        })
        .collect();
    let preds = run_inference_input_only(&items, move |s: &str| (q.as_predict_fn())(s))
        .unwrap_or_else(|e| panic!("evaluator path aborted: {e:?}"));
    assert_eq!(preds.len(), data.len());
    // 判定経路の出どころは Judgment で、バッチの注記は付かない。
    let provenance = PredictionProvenance::Judgment;
    assert_eq!(provenance.batch_prediction_notice(), None);
    assert!(!provenance.is_batch_prediction());
    // 判定経路のラベルは単体推論と全件一致する（フラグ不要の根拠。TASK-28.1）。
    for (i, (x, r)) in data.iter().zip(&preds).enumerate() {
        let single = pl.infer_one(x).expect("single");
        assert_eq!(single.label_index(), pred(r, i).label_index(), "index {i}");
    }
}

#[test]
fn req28_reference_batch_debug_hides_values() {
    let refs = ["abc", "def"];
    let out = pipeline()
        .infer_batch_for_reference(&refs)
        .expect("within limits");
    assert_eq!(
        format!("{out:?}"),
        "ReferenceBatchPredictions(len=2, provenance=Reference { mode: Batch })"
    );
}

#[test]
fn req28_reference_batch_propagates_batch_error() {
    let refs: Vec<&str> = vec![""; MAX_INFER_BATCH_LEN + 1];
    let err = pipeline()
        .infer_batch_for_reference(&refs)
        .expect_err("too many inputs");
    assert_eq!(
        err,
        BatchError::TooManyInputs {
            len: MAX_INFER_BATCH_LEN + 1,
            limit: MAX_INFER_BATCH_LEN
        }
    );
}
