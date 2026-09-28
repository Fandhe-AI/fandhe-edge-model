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
//! 例外: [`MAX_WORKER_STDERR_BYTES`]・[`SUPERVISOR_SHUTDOWN_GRACE_SECONDS`]
//! （issue #178）は Rust 側（子プロセスの起動・監視）固有の値で、
//! 学習ワーカー側に対応する単一真実源を持たない（各定数の doc に根拠を
//! 記す）。
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

/// 成果物の `selector_version` に許可される値（`artifact.py::SELECTOR_VERSION`）。
/// 空文字列・未対応版の成功結果を検証済みとして受理しないための許可リスト
/// （REQ-39「完全性と版」・P1。codex review PR #220）。値を追加する場合は
/// `artifact.py::SELECTOR_VERSION` の版付け方針と合わせて見直す。
pub const ALLOWED_SELECTOR_VERSIONS: [&str; 1] = ["0.1"];

/// 学習ワーカーの標準エラー出力（stderr）の保持上限（バイト）。REQ-39
/// 「資源の上限」（#178）。`_worker` は supervisor の stderr を継承する
/// （`supervisor.py::_spawn_worker_and_finalize`）ため、Rust 側 stderr
/// パイプの書き手は supervisor 自身と `_worker` の両方になりうる。stdout
/// （[`MAX_RESULT_BYTES`]。契約上の結果 JSON）とは別に、診断用途の stderr
/// にも無制限の保持を許さないための独立した上限を設ける。先頭から保持し、
/// 超過分は読み捨てる（`crates/train/src/process.rs`）。
pub const MAX_WORKER_STDERR_BYTES: usize = 64 * 1024;

/// Rust 側の壁時計締め切りに足す猶予（秒）。REQ-34・REQ-39（#178）。
///
/// `supervisor.py`（唯一の内側監視者）は `_worker` を別セッション
/// （`start_new_session=True`）で起動するため、Rust 側が supervisor
/// プロセスだけを kill しても、別セッションの `_worker` には SIGKILL が
/// 届かない。そのため Rust 側の外側締め切りは、supervisor 自身が
/// `time_limit_seconds` 超過を検出してから後始末を終えるまでの時間を
/// 確実に上回る必要がある（さもないと `_worker` が孤児として残る）。
///
/// 内訳（`supervisor.py` の定数から算出。根拠を明示し、緩めずに保つ）:
/// 内側の猶予 `_TIME_LIMIT_GRACE_SECONDS`（5 秒）＋ `ps` 呼び出しの
/// タイムアウト（5 秒）＋ kill 後の `proc.wait()` 上限（10 秒）＋ reader
/// スレッドの join 上限（10 秒＋5 秒）＝ 35 秒に、Python プロセス起動・
/// OS のプロセス後始末のための余裕を加えて 60 秒とする。
///
/// この結果、既定値（`time_limit_seconds` = [`MAX_TRAIN_WALL_SECONDS`] =
/// 3600 秒）では Rust 側の外側締め切りは 3660 秒となり、字義どおりの
/// 「3600 秒を超えない」からは猶予分だけ外れる（issue #178 実装計画・
/// オーナー確認事項 1。孤児プロセスを残さないことを優先する設計判断）。
pub const SUPERVISOR_SHUTDOWN_GRACE_SECONDS: u32 = 60;
