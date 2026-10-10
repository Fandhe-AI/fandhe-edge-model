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
//! - [`kind`]: `kind` 値の許可リスト判定（TASK-39.2-3・#155）。実装済み。`infer` の `artifact.json` への CLI 統合は #156 で済み（`train` への統合は未了。#136）
//! - [`version_ledger`]: 版管理台帳（版 ID・sha256・作成時刻の記録と取得。TASK-39.3-1・#167）。実装済み。前版へのロールバック対象の検証（ハッシュ一致検証。復元操作そのものは未実装）も実装済み（TASK-39.3-2・#168）。永続化（`version_ledger.json`）は #491 で実装し、CLI の `package` が記録し `infer` が照合する（#518）
//! - [`resource`]: 実行時間の上限（推論 1 件あたり暫定 10 秒）を子プロセス境界で強制し、超過を記録する
//!   （TASK-39.5-1・#170）。メモリ（RSS）上限（暫定 2 GiB。RSS ポーリングによる模擬。TASK-39.5-2・#171）も実装済み。
//!   CLI 等への配線は #136
//! - [`file_size`]: ファイルサイズ上限（暫定 1 GiB）の読み込み前検証（fstat。TASK-39.5-3・#172）。実装済み
//! - [`kind_version`]: `kind` ごとの `kind_version` 許可リスト判定（TASK-39.6-1・#174）。実装済み。`infer` の `artifact.json` へ CLI 統合済み
//! - 未統合（#136）: `train` への `kind` 検査（[`kind`] の許可リスト）の統合。`register` は経路の閉じ込め
//!   （[`path`]）まで接続済み。破損パッケージの確認（TASK-39.6-2・#175）は未着手

pub mod file_size;
pub mod format;
pub mod kind;
pub mod kind_version;
pub mod model_file;
pub mod package;
pub mod path;
pub mod resource;
pub mod version_ledger;
