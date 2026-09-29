//! 成果物・推論 SDK 層（`fandhe-edge-runtime`。REQ-28/30〜32）。
//!
//! # 層の境界
//!
//! - 依存は共通コア（`fandhe-edge-core`）のみ。評価器・データ契約・学習ワーカーには依存しない
//!   （推論は学習に依存しない。REQ-32）
//! - 呼び出し元（想定）: CLI の `infer` 工程（TASK-33.x）・評価器の推論関数（TASK-27.2 経由。#118）
//!
//! # 現状
//!
//! - [`pipeline`][]: 単体推論とバッチ推論が同一の 1 系列経路を通る構造の継ぎ目（TASK-28.1-1・#117）。
//!   実前処理（NFKC＋バイトエンコード。#112）と ONNX 推論（#113）は未実装（実装済みを装わない）
//! - [`capacity`][]: 容量計測コア（REQ-30・TASK-30.1-1・#122）。実装済み。上限照合
//!   （TASK-30.2）・配布パッケージ形式は後続 TASK
//! - [`latency`][]: 推論のみの待ち時間の反復計測ハーネス（REQ-31・TASK-31.1-1・#127）。
//!   実モデルでの計測は #113 後
//! - [`latency_report`][]: p95 の厳密な整数算出と、250ms を参考値として明記したレポート
//!   （REQ-31・TASK-31.1-2・#128）。合否・上限照合は持たない（#129）
//! - [`package_outcome`][]: 上限超過を合否判定より優先して `limit_exceeded`（20）へ写す
//!   終了コード決定（REQ-21・TASK-21.3-1・#132）。待ち時間（p95）の上限超過
//!   （`LimitBreach::Latency`・境界規則 `latency_if_exceeded`。REQ-31・TASK-21.3-2・#133）も
//!   扱う。容量の照合・CLI 配線は未実装
//!
//! TASK-28.1-2（#118）で、650 件の入力に対する単体・バッチ・評価器経路の予測ラベル全件一致
//! テスト（`tests/full_match.rs`。証拠種別: テストハーネス）を追加した。実前処理・ONNX での
//! 再実行は #112・#113 の実装後に行う。

pub mod capacity;
pub mod latency;
pub mod latency_report;
pub mod package_outcome;
pub mod pipeline;
