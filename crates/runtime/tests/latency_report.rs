//! p95 レポートの結合テスト（REQ-31・TASK-31.1-2・#128）。
//! 証拠種別: テストハーネス（偽の時計・模擬バックエンド）。実機計測ではない。

use fandhe_edge_runtime::latency::*;
use fandhe_edge_runtime::latency_report::*;
use fandhe_edge_runtime::pipeline::*;
use std::cell::Cell;
use std::rc::Rc;

#[derive(Clone)]
struct FakeClock(Rc<Cell<u64>>);
impl Clock for FakeClock {
    fn now_ns(&self) -> u64 {
        self.0.get()
    }
}

struct Pre;
impl Preprocessor for Pre {
    fn preprocess(&self, _input: &str) -> Result<TokenIds, PreprocessError> {
        Ok(TokenIds::new(vec![1]))
    }
}

/// 呼び出し番号 k ごとに時計を `base + step * k` 進める。
struct TickBackend {
    clock: Rc<Cell<u64>>,
    calls: Cell<u64>,
    base: u64,
    step: u64,
}
impl ScoringBackend for TickBackend {
    fn scores(&self, _ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        let k = self.calls.get();
        self.calls.set(k + 1);
        self.clock.set(self.clock.get() + self.base + self.step * k);
        Ok(vec![0.75, 0.25])
    }
}

fn report(base: u64, step: u64, input: &str) -> LatencyReport {
    let cell = Rc::new(Cell::new(0));
    let pipeline = InferencePipeline::new(
        Pre,
        TickBackend {
            clock: cell.clone(),
            calls: Cell::new(0),
            base,
            step,
        },
    );
    let cfg = LatencyConfig::new(0, 100).unwrap();
    let s = measure_latency(&pipeline, &[input], &cfg, &FakeClock(cell)).unwrap();
    summarize_latency(&s).unwrap()
}

#[test]
fn req31_measured_samples_yield_exact_p95() {
    // サンプルは 100..=199。厳密 p95 は 194.05 なので切り上げて 195。
    let r = report(100, 1, "a");
    assert_eq!((r.min_ns(), r.max_ns()), (100, 199));
    assert_eq!((r.p95().floor_ns(), r.p95().frac_hundredths()), (194, 5));
    assert_eq!(r.p95_ns(), 195);
    assert_eq!(r.reference_comparison(), ReferenceComparison::Below);
}

#[test]
fn req31_p95_over_reference_is_informational_only() {
    // 300ms ずつ進める分布でも、レポートは情報を出すだけで合否を持たない。
    let r = report(300_000_000, 0, "a");
    assert_eq!(r.p95_ns(), 300_000_000);
    assert_eq!(r.reference_comparison(), ReferenceComparison::AtOrAbove);
    let text = r.to_string();
    assert!(text.contains("informational only"));
    assert!(text.contains("not a pass/fail criterion"));
    assert!(text.contains("reference 250 ms (at or above;"));
}

#[test]
fn req31_display_excludes_input_text() {
    let r = report(100, 1, "secret-input");
    assert!(!r.to_string().contains("secret-input"));
    assert!(r.to_string().contains("linear interpolation"));
}
