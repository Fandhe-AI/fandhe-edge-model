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
///
/// 性能の 3 層（REQ-30・REQ-31・REQ-39 の 2026-10-16 注記、TASK-41.9・#579）のうち**安全上限**
/// （暴走止め。超えたら停止して終了コード 20）。目標（p95 250ms。警告のみ）・利用者設定の上限
/// （`limits.max_infer_p95_us`）とは別の契約。
pub const INFER_TIME_LIMIT: Duration = Duration::from_secs(10);
/// [`INFER_TIME_LIMIT`] の ns 表現（`INFER_TIME_LIMIT` から導出。別の値を持たない）。
pub const INFER_TIME_LIMIT_NS: u64 = INFER_TIME_LIMIT.as_secs() * 1_000_000_000;

/// 推論プロセスの RSS（常駐メモリ）の暫定上限（2 GiB。REQ-39・PoC-20 の `INFER_MAX_RSS_BYTES`・C-16）。
///
/// 3 層のうち**安全上限**（超えたら停止して終了コード 20。TASK-41.9・#579）。目標の RSS 2 GiB（警告のみ）とは
/// 別の契約で、RSS には利用者設定の上限は無い。現状は目標と値が同じで、目標超過の警告が出る区間
/// （目標 < 観測 <= 安全上限）は無い。警告の出力は安全上限の種類別化（#578）の後に検討する。
///
/// 推論経路向けの値で、学習ワーカーには適用しない。ガード層が RSS ポーリングで強制する（模擬。
/// 確実な上限機構への置き換えは別課題。TASK-39.5-2・#171）。
pub const INFER_RSS_LIMIT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// 呼び出し側が指定できる時間上限の最大値（暫定 1 時間。REQ-39）。
pub const MAX_TIME_LIMIT: Duration = Duration::from_secs(3600);
/// [`MAX_TIME_LIMIT`] の ns 表現（`MAX_TIME_LIMIT` から導出）。
pub const MAX_TIME_LIMIT_NS: u64 = MAX_TIME_LIMIT.as_secs() * 1_000_000_000;

/// 出力量（バッチ全体の総出力、または子プロセス出力の読み取り）の上限の最大値
/// （暫定 256 MiB。REQ-39）。
pub const MAX_OUTPUT_BYTES: usize = 256 * 1024 * 1024;

/// プロジェクトへ取り込む学習・評価データ 1 ファイルの読み込み上限（64 MiB。暫定。REQ-39）。
/// CLI の各工程の読み込み（`MAX_PROJECT_FILE_BYTES`）の出所で、下の学習用 JSONL の上限もここから導出する。
pub const MAX_PROJECT_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// 学習ワーカーへ渡す trainer 形式 JSONL の生成量の上限（暫定。REQ-39）。
///
/// 入力データの読み込み上限 [`MAX_PROJECT_FILE_BYTES`] の 4 倍。`{"input","label"}` の行への変換は、
/// JSON のキー・ラベルの付与と、入力文字列のエスケープ（制御文字・非 ASCII が `\uXXXX` で最大 6 倍に
/// 膨らむ）により入力より大きくなりうるため、入力上限そのものでは正常な入力まで拒否しうる。一方で
/// 無制限に生成すると資源を使い切るので、上限を超えた時点で生成をやめる。4 倍は暫定値で、確定値は
/// 実機測定後に spec 側で決まる。
pub const TRAINER_JSONL_MAX_BYTES: u64 = 4 * MAX_PROJECT_FILE_BYTES;

/// 分割ファイル（外部データの shard）の最大件数（REQ-39・TASK-41.7-8）。
///
/// ガード層の `MAX_SPLIT_FILES`・推論ランタイムの `FINETUNE_MODEL_LIMITS.external.max_shards`・
/// `artifact.json` の `external_data` 件数上限が同じ値を参照する（上限変更時に受理範囲をずらさない）。
pub const MAX_EXTERNAL_DATA_SHARDS: usize = 256;

/// `qwen2_lora` の `config.max_seq_length` の下限（REQ-41・REQ-39）。
///
/// 学習ワーカーの `limits.py::MIN_LORA_SEQ_LENGTH` とは言語が違い参照できないため、共有 fixture
/// `fixtures/lora_limits/lora_seq_length.json` を介して双方のテストで照合する。
pub const MIN_LORA_SEQ_LENGTH: u64 = 8;

/// `qwen2_lora` の `config.max_seq_length` の上限（`limits.py::MAX_LORA_SEQ_LENGTH` と同値）。
pub const MAX_LORA_SEQ_LENGTH: u64 = 4096;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39: 暫定値の具体値と、ns 表現が Duration と一致すること。
    #[test]
    fn req39_limits_have_expected_values() {
        assert_eq!(INFER_TIME_LIMIT, Duration::from_secs(10));
        assert_eq!(INFER_TIME_LIMIT_NS, 10_000_000_000);
        assert_eq!(INFER_RSS_LIMIT_BYTES, 2_147_483_648);
        assert_eq!(MAX_TIME_LIMIT, Duration::from_secs(3600));
        assert_eq!(MAX_TIME_LIMIT_NS, 3_600_000_000_000);
        assert_eq!(MAX_OUTPUT_BYTES, 268_435_456);
        assert_eq!(MAX_PROJECT_FILE_BYTES, 67_108_864);
        assert_eq!(TRAINER_JSONL_MAX_BYTES, 268_435_456);
        assert_eq!(u128::from(INFER_TIME_LIMIT_NS), INFER_TIME_LIMIT.as_nanos());
        assert_eq!(u128::from(MAX_TIME_LIMIT_NS), MAX_TIME_LIMIT.as_nanos());
    }
}
