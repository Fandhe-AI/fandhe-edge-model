//! 成果物・推論 SDK 層（`fandhe-edge-runtime`。REQ-28/30〜32）。
//!
//! 配布パッケージと学習に依存しない推論ランタイムを担う層で、学習側
//! （`fandhe-edge-train`・Python・MLX）にも評価器（`fandhe-edge-eval`）にも
//! 依存しない（REQ-32）。依存は共通コア（`fandhe-edge-core`）のみ。
//!
//! # 実装状況
//!
//! - [`capacity`][]: 容量計測コア（REQ-30・TASK-30.1-1・#122）。実装済み
//! - 推論ランタイム・配布パッケージ形式・上限照合（TASK-30.2）などは後続 TASK で追加する

pub mod capacity;
