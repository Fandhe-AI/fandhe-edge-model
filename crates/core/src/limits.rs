//! REQ-39 の資源上限（暫定値）の単一の出所（共通コア。TASK-39.5-1・#170）。
//!
//! ガード層（子プロセス境界の強制）・推論ランタイム（プロセス内の協調的な期限）・CLI が
//! 同じ契約値を別々に持つとずれて片方だけが先に効くため、全層が依存する最下層に 1 箇所で
//! 定義し、各層はここを参照・導出する。値はいずれも暫定（PoC-20 の C-16 に基づく。
//! 確定値は実機測定後に spec 側で決まる）。ここへ集約するのは「同じ REQ-39 の同じ上限」を
//! 表す値だけで、名前・値が偶然一致するだけの別契約の値（バッチ全体の期限・
//! レイテンシ計測全体の期限など）は置かない。

use std::time::Duration;

/// 推論 1 件の実行時間の暫定上限（REQ-39）。
pub const INFER_TIME_LIMIT: Duration = Duration::from_secs(10);
/// [`INFER_TIME_LIMIT`] の ns 表現（`INFER_TIME_LIMIT` から導出。別の値を持たない）。
pub const INFER_TIME_LIMIT_NS: u64 = INFER_TIME_LIMIT.as_secs() * 1_000_000_000;

/// 呼び出し側が指定できる時間上限の最大値（暫定 1 時間。REQ-39）。
pub const MAX_TIME_LIMIT: Duration = Duration::from_secs(3600);
/// [`MAX_TIME_LIMIT`] の ns 表現（`MAX_TIME_LIMIT` から導出）。
pub const MAX_TIME_LIMIT_NS: u64 = MAX_TIME_LIMIT.as_secs() * 1_000_000_000;

/// 出力量（バッチ全体の総出力、または子プロセス出力の読み取り）の上限の最大値
/// （暫定 256 MiB。REQ-39）。
pub const MAX_OUTPUT_BYTES: usize = 256 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39: 暫定値の具体値と、ns 表現が Duration と一致すること。
    #[test]
    fn req39_limits_have_expected_values() {
        assert_eq!(INFER_TIME_LIMIT, Duration::from_secs(10));
        assert_eq!(INFER_TIME_LIMIT_NS, 10_000_000_000);
        assert_eq!(MAX_TIME_LIMIT, Duration::from_secs(3600));
        assert_eq!(MAX_TIME_LIMIT_NS, 3_600_000_000_000);
        assert_eq!(MAX_OUTPUT_BYTES, 268_435_456);
        assert_eq!(u128::from(INFER_TIME_LIMIT_NS), INFER_TIME_LIMIT.as_nanos());
        assert_eq!(u128::from(MAX_TIME_LIMIT_NS), MAX_TIME_LIMIT.as_nanos());
    }
}
