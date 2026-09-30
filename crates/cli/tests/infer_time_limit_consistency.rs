//! 推論 1 件の時間上限（暫定 10 秒）の定数一致テスト（REQ-39・TASK-39.5-1・#170）。
//!
//! ガード層（プロセス境界の強制）と推論ランタイム（プロセス内の協調的な期限）は
//! 互いに依存しないため別々に値を持つ。ずれると片方だけが先に効くので、CLI 層で照合する。

use fandhe_edge_guard::resource::INFER_TIME_LIMIT;
use fandhe_edge_runtime::latency::DEFAULT_LATENCY_PER_INFER_TIMEOUT_NS;

/// REQ-39: guard の強制上限と runtime の 1 件あたり期限の既定値が一致する（10 秒）。
#[test]
fn req39_guard_and_runtime_per_infer_limits_agree() {
    assert_eq!(
        INFER_TIME_LIMIT.as_nanos(),
        u128::from(DEFAULT_LATENCY_PER_INFER_TIMEOUT_NS)
    );
    assert_eq!(INFER_TIME_LIMIT.as_secs(), 10);
}
