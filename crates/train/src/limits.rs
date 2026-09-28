//! 学習リクエスト・結果 JSON の資源上限（REQ-39）。
//!
//! 値は学習ワーカー側の単一真実源
//! （`trainer/src/fandhe_edge_trainer/limits.py`・`contract.py`・
//! `supervisor.py`）をそのまま写したもので、オーナー確認済み
//! （2026-09-27・issue #11）。本ファイルは値を緩めず、変更する場合は
//! Python 側・共有 fixture（`fixtures/train_contract/limits.json`）と
//! 同時に見直す（`crates/train/tests/train_contract_fixture.rs` が
//! fixture との一致を機械照合する）。
//!
//! [`crate::limits`] の各定数は `fandhe_edge_core::judgment` の
//! `MAX_OPTIONS`・`MAX_CHOICE_ID_BYTES` と値がたまたま同じでも、
//! 契約としては別物であり結合しない（学習リクエストの `label_order` は
//! 学習ワーカー固有の契約であり、推論 1 件あたりの判定結果契約とは独立に
//! 変更されうる）。

/// 学習リクエスト JSON の `schema_version`（`contract.py::SCHEMA_VERSION`）。
pub const REQUEST_SCHEMA_VERSION: u32 = 1;

/// 学習リクエスト JSON ファイルの読み込み上限（バイト）。
/// `contract.py::limits.MAX_REQUEST_BYTES`。
pub const MAX_REQUEST_BYTES: usize = 1024 * 1024;

/// `label_order` の最小・最大件数。`limits.py::MIN_LABELS`・`MAX_LABELS`。
pub const MIN_LABELS: usize = 2;
pub const MAX_LABELS: usize = 1024;

/// 1 ラベルあたりの最大 UTF-8 バイト数。`limits.py::MAX_LABEL_BYTES`。
pub const MAX_LABEL_BYTES: usize = 256;

/// `max_bytes`（バイト入力の最大長）の許容範囲。
/// `limits.py::MIN_MAX_BYTES`・`MAX_MAX_BYTES`。
pub const MIN_MAX_BYTES: u32 = 1;
pub const MAX_MAX_BYTES: u32 = 4096;

/// `seed` の許容範囲。`limits.py::MIN_SEED`・`MAX_SEED`
/// （`mx.random.seed`・`numpy.random.default_rng` が受け付ける非負 32bit
/// 符号なし整数の範囲）。
pub const MIN_SEED: u32 = 0;
pub const MAX_SEED: u32 = u32::MAX;

/// 1 学習ジョブあたりの壁時計上限（秒）。`limits.py::MAX_TRAIN_WALL_SECONDS`。
pub const MAX_TRAIN_WALL_SECONDS: u32 = 3600;

/// 1 学習ジョブあたりの RSS 上限（バイト）。`limits.py::MAX_TRAIN_RSS_BYTES`。
pub const MAX_TRAIN_RSS_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// 学習結果（ワーカーの標準出力）の読み込み上限（バイト）。
/// `supervisor.py::_MAX_WORKER_STDOUT_BYTES`。
pub const MAX_RESULT_BYTES: usize = 1024 * 1024;

/// `device` に許可される値。`contract.py::_ALLOWED_DEVICES`。
pub const ALLOWED_DEVICES: [&str; 2] = ["cpu", "gpu"];
