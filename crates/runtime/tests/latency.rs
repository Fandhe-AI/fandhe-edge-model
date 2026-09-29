//! 待ち時間計測ハーネスの結合テスト（REQ-31・REQ-39・TASK-31.1-1・#127）。
//! 証拠種別: テストハーネス（偽の時計・模擬の前処理 / バックエンド）。実機計測ではない。

use fandhe_edge_runtime::latency::*;
use fandhe_edge_runtime::pipeline::*;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[derive(Clone)]
struct FakeClock(Rc<Cell<u64>>);
impl Clock for FakeClock {
    fn now_ns(&self) -> u64 {
        self.0.get()
    }
}

struct Pre(Rc<RefCell<Vec<String>>>);
impl Preprocessor for Pre {
    fn preprocess(&self, input: &str) -> Result<TokenIds, PreprocessError> {
        self.0.borrow_mut().push(input.to_string());
        Ok(TokenIds::new(vec![1]))
    }
}

/// 呼び出し番号 k ごとに時計を `100 + k` 進める。`fail_at` の呼び出しは失敗させる。
struct TickBackend {
    clock: Rc<Cell<u64>>,
    calls: Rc<Cell<u64>>,
    fail_at: Option<u64>,
}
impl ScoringBackend for TickBackend {
    fn scores(&self, _ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        let k = self.calls.get();
        self.calls.set(k + 1);
        self.clock.set(self.clock.get() + 100 + k);
        if self.fail_at == Some(k) {
            return Err(BackendError::Failed);
        }
        Ok(vec![0.75, 0.25])
    }
}

struct Rig {
    pipeline: InferencePipeline<Pre, TickBackend>,
    clock: FakeClock,
    calls: Rc<Cell<u64>>,
    seen: Rc<RefCell<Vec<String>>>,
}

fn rig(fail_at: Option<u64>) -> Rig {
    // モデルロードの代用: 構築時に時計を大きく進める（計測区間外であることを示す）。
    let cell = Rc::new(Cell::new(1_000_000_000));
    let calls = Rc::new(Cell::new(0));
    let seen = Rc::new(RefCell::new(Vec::new()));
    let pipeline = InferencePipeline::new(
        Pre(seen.clone()),
        TickBackend {
            clock: cell.clone(),
            calls: calls.clone(),
            fail_at,
        },
    );
    Rig {
        pipeline,
        clock: FakeClock(cell),
        calls,
        seen,
    }
}

#[test]
fn req31_samples_count_equals_iters_and_values_are_exact() {
    let r = rig(None);
    let cfg = LatencyConfig::new(3, 5).unwrap();
    let s = measure_latency(&r.pipeline, &["a"], &cfg, &r.clock).unwrap();
    // warmup は呼び出し番号 0..3、計測は 3..8 なので所要時間は 100 + k。
    assert_eq!(s.samples_ns(), &[103, 104, 105, 106, 107]);
    assert_eq!((s.warmup(), s.iters()), (3, 5));
}

#[test]
fn req31_model_load_time_is_excluded() {
    let r = rig(None);
    let cfg = LatencyConfig::new(0, 2).unwrap();
    let s = measure_latency(&r.pipeline, &["a"], &cfg, &r.clock).unwrap();
    assert_eq!(s.samples_ns(), &[100, 101]);
    assert!(s.samples_ns().iter().all(|&x| x < 1_000_000_000));
}

#[test]
fn req31_warmup_is_not_recorded() {
    let r = rig(None);
    let cfg = LatencyConfig::new(4, 6).unwrap();
    let s = measure_latency(&r.pipeline, &["a", "b"], &cfg, &r.clock).unwrap();
    assert_eq!(r.calls.get(), 10);
    assert_eq!(s.samples_ns().len(), 6);
}

#[test]
fn req31_inputs_are_cycled_in_order() {
    let r = rig(None);
    let cfg = LatencyConfig::new(2, 5).unwrap();
    measure_latency(&r.pipeline, &["x", "y", "z"], &cfg, &r.clock).unwrap();
    let seen: Vec<String> = r.seen.borrow().clone();
    // warmup と計測はそれぞれ先頭から循環する。
    assert_eq!(seen, ["x", "y", "x", "y", "z", "x", "y"]);
}

#[test]
fn req31_inference_failure_aborts_fail_closed() {
    let cfg = LatencyConfig::new(2, 5).unwrap();
    let r = rig(Some(4));
    let e = measure_latency(&r.pipeline, &["a"], &cfg, &r.clock).unwrap_err();
    assert_eq!(
        e,
        LatencyError::Inference {
            phase: LatencyPhase::Measure,
            iteration: 2,
            code: "backend_failed"
        }
    );
    let r = rig(Some(1));
    let e = measure_latency(&r.pipeline, &["a"], &cfg, &r.clock).unwrap_err();
    assert_eq!(
        e,
        LatencyError::Inference {
            phase: LatencyPhase::Warmup,
            iteration: 1,
            code: "backend_failed"
        }
    );
}

#[test]
fn req39_rejects_empty_inputs() {
    let r = rig(None);
    let e = measure_latency(&r.pipeline, &[], &LatencyConfig::default(), &r.clock).unwrap_err();
    assert_eq!(e, LatencyError::NoInputs);
    assert_eq!(r.calls.get(), 0);
}

#[test]
fn req39_rejects_too_many_inputs() {
    let r = rig(None);
    let inputs = vec!["a"; MAX_INFER_BATCH_LEN + 1];
    let e = measure_latency(&r.pipeline, &inputs, &LatencyConfig::default(), &r.clock).unwrap_err();
    assert_eq!(
        e,
        LatencyError::TooManyInputs {
            len: MAX_INFER_BATCH_LEN + 1,
            limit: MAX_INFER_BATCH_LEN
        }
    );
    assert_eq!(r.calls.get(), 0);
}

#[test]
fn non_monotonic_clock_is_rejected() {
    // 推論のたびに時計を巻き戻す。
    struct Rewind(Rc<Cell<u64>>);
    impl ScoringBackend for Rewind {
        fn scores(&self, _ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
            self.0.set(self.0.get().saturating_sub(10));
            Ok(vec![1.0])
        }
    }
    let cell = Rc::new(Cell::new(1000));
    let p = InferencePipeline::new(Pre(Rc::default()), Rewind(cell.clone()));
    let cfg = LatencyConfig::new(0, 3).unwrap();
    let e = measure_latency(&p, &["a"], &cfg, &FakeClock(cell)).unwrap_err();
    assert_eq!(e, LatencyError::NonMonotonicClock { iteration: 0 });
}

#[test]
fn req31_smoke_with_monotonic_clock() {
    // 実時計での動作確認のみ。時間の閾値は置かない（実機計測ではない）。
    let r = rig(None);
    let cfg = LatencyConfig::new(1, 10).unwrap();
    let s = measure_latency(&r.pipeline, &["a"], &cfg, &MonotonicClock::new()).unwrap();
    assert_eq!(s.samples_ns().len(), 10);
    assert_eq!(r.calls.get(), 11);
}

#[test]
fn debug_output_does_not_contain_input_text() {
    let r = rig(Some(0));
    let cfg = LatencyConfig::new(0, 1).unwrap();
    let e = measure_latency(&r.pipeline, &["secret-body-xyz"], &cfg, &r.clock).unwrap_err();
    assert!(!format!("{e:?}").contains("secret-body-xyz"));
}
