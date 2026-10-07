//! 操作アダプター層の CLI（`fandhe-edge-cli`）ライブラリターゲット。
//!
//! PoC-16（`core-cli-vertical-slice`）と同様に lib/bin を分離し、業務ロジ
//! ックを持たない薄いアダプター関数（出力配線等）をここに置いて、テスト
//! ハーネスから直接呼べるようにする（`main.rs` の `[[bin]]` は
//! `CARGO_BIN_EXE_fandhe-edge` としてしか結合テストから叩けないため）。
//!
//! # 現状（TASK-21.1-2・TASK-21.2・TASK-33.1-1）
//!
//! [`output`] は `JudgmentResult` / `ErrorReport` を JSON 1 行として書き出す
//! （TASK-21.1-2・TASK-21.2）。あわせて容量内訳を `package` 出力用 JSON にする
//! `output::package_capacity_json`・`write_package_capacity` と
//! `capacity_error_report`（TASK-30.1-2・#123。`package` 工程への配線は
//! TASK-33.1-2）も実装済み。[`args`] は 7 工程サブコマンドの引数定義と
//! パーサ・help 生成（TASK-33.1-1）。各工程の下位層への接続は
//! TASK-33.1-2（#136）で [`stages`] に実装し、`main.rs` は解析成功後に [`stages::run`] を呼ぶ
//! （プロジェクトディレクトリの規約は [`project`]。未接続の区間は [`stages`] の doc を参照）。
//!
//! TASK-33.2-2（#139）で、`package` の結果を stdout の JSON 1 つへ写す
//! [`stage_output`] と、stderr へテキストログを出す [`log`] を追加した。
//! stdout（結果 JSON）と stderr（ログ）は別の書き込み先として扱う。`package` の
//! 実処理への接続は #136 で [`stages::package`] に実装した。
//!
//! TASK-33.4（#141）で、`infer --input-file` の一括推論と 1 行 1 JSON 出力（REQ-33 の唯一の
//! 例外）を [`infer_batch`] に追加した。それ以外のコマンドの出力は 1 呼び出し 1 JSON のまま
//! （[`infer_batch::output_mode`] で型として固定）。`infer` 工程への配線は #136 で [`stages::infer`] に実装した。
//!
//! TASK-39.4-2（#159）で、`infer` の `--package` と `artifact.json` の `onnx_file` を
//! ガード層の経路検証へ通す [`infer_guard`] を追加し、`main.rs` の `infer` 分岐へ接続した。
//! TASK-39.2-4（#156）で、同じ経路に `artifact.json` の `kind` の許可リスト検査も接続した。
//! 拒否は `invalid_input`（64）の JSON 1 行。通過後の推論本体は #136 で [`stages::infer`] に接続した（`kind_version` は許可リストで検証済み。外部台帳による sha256 検証と版管理台帳による前版への復帰は未検証のまま。#168・#174）。
//!
//! TASK-33.3（#140）で、評価データ未定義の `evaluate` を `status:"skipped"`・exit 0 で終える
//! `stage_output::evaluate_start`・`emit_evaluate_skipped` を追加した。バイナリへの結線は #136 で [`stages::evaluate`] に実装した。

pub mod args;
pub mod error_report;
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod frozen_dir;
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod held_ledger_dir;
pub mod infer_batch;
pub mod infer_guard;
pub mod log;
pub mod output;
pub(crate) mod prediction_lines;
pub mod project;
pub mod stage_output;
pub mod stages;
