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
//!   ONNX 推論の実装は [`onnx`][]（#113）
//! - [`preprocess`][]: バイト前処理（NFKC 正規化＋バイトエンコード。REQ-32・TASK-32.1-1・#112）。
//!   実装済み。共有ゴールデンベクタで学習ワーカーの出力と機械照合する
//! - [`onnx`][]: C1・C3 の ONNX を読む自作の推論バックエンド（`ScoringBackend` 実装。REQ-32・REQ-28・
//!   TASK-32.1-2・#113）。std と承認済み依存のみで、書き出し器のグラフとの完全一致を照合する許可制。
//!   `ort`・`tract-onnx` は未承認のため使わない（承認事項）。`artifact.json` の読み込み・CLI 配線は
//!   後続（TASK-33.x）。`autoregressive` は未対応（REQ-19b の後続 TASK）。共有 fixture
//!   `fixtures/onnx_parity/` との全件一致は `tests/onnx_parity.rs`（証拠種別: テストハーネス）
//! - [`capacity`][]: 容量計測コア（REQ-30・TASK-30.1-1・#122）。実装済み。上限照合
//!   （TASK-30.2）・配布パッケージ形式は後続 TASK。エラーの終了コード・公開メッセージへの
//!   写像（TASK-30.1-2・#123）を持ち、JSON 直列化は CLI 側
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
//! テスト（`tests/full_match.rs`。証拠種別: テストハーネス）を追加した。650 件規模の実モデルでの再実行と
//! 実モデルでの待ち時間計測は未実施（#113 の実装で可能になった。`tests/onnx_parity.rs` は fixture の
//! 全件で単体・バッチ・評価器用関数の一致まで確認する）。
//!
//! 不一致を検出した場合の原因特定・記録の手順は `docs/design/runtime-batch-mismatch-procedure.md`
//! （REQ-28 異常系・TASK-28.2・#119）に従う。
//!
//! TASK-32.3（#115）で、環境変数を空にした（`env -i` 相当）子プロセスでの推論成功テスト
//! （`tests/env_isolation.rs`。証拠種別: テストハーネス）を追加した。CLI `infer` 経由の確認は
//! 工程の接続（#136）後に追加する。

pub mod capacity;
pub mod latency;
pub mod latency_report;
pub mod onnx;
pub mod package_outcome;
pub mod pipeline;
pub mod preprocess;
