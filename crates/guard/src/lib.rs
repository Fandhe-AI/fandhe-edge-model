//! ガード層（REQ-39・PoC-20）。CLI・MCP・TUI から渡される入力を、処理の手前で検査する。
//!
//! PoC-20 で「ガード層の無い CLI には防御がない」ことを実測済み。CLI・MCP・TUI が
//! 同じ検査を通れるよう、操作アダプターとは別 crate に置く（REQ-33・REQ-36）。
//! 依存は共通コア（`fandhe-edge-core`）のみで、推論ランタイムからも参照できる（REQ-32）。
//!
//! # 現状
//!
//! - [`format`][]: 許可リストによるファイル形式判定（TASK-39.2-1・#153）。実装済み
//! - [`model_file`]: モデルファイルの拡張子（`.onnx`）と内容（ONNX）の照合による pickle 偽装・非 ONNX の拒否
//!   （TASK-39.2-2・#154）。実装済み
//! - [`path`]: 経路の閉じ込め（`safe_join` 相当・検証と open を一体化した `open_confined`。TASK-39.4-1・#158）。実装済み。
//!   CLI 引数への組み込みは TASK-39.4-2（#159）で実装済み
//! - [`package`]: パッケージ単位の閉じ込め（`--package` は workspace 配下、`onnx_file` 等の
//!   メンバーはパッケージ配下かつ workspace 配下。TASK-39.4-2・#159）。実装済み
//! - [`kind`]: `kind` 値の許可リスト判定（TASK-39.2-3・#155）。実装済み。CLI への統合は #156 で未着手
//! - [`version_ledger`]: 版管理台帳（版 ID・sha256・作成時刻の記録と取得。TASK-39.3-1・#167）。実装済み（メモリ上のみ）。前版へのロールバックと復元後のハッシュ一致検証も実装済み（TASK-39.3-2・#168）。
//!   ロールバックと復元後のハッシュ一致検証は TASK-39.3-2（#168）、永続化は未着手
//! - 未着手（後続 TASK）: CLI への統合（TASK-39.2-4・#156）・資源の上限（TASK-39.5）・
//!   完全性と版（`kind_version` の許可リストを含む。TASK-39.6）

pub mod format;
pub mod kind;
pub mod model_file;
pub mod package;
pub mod path;
pub mod version_ledger;
