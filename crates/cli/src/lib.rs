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
//! パーサ・help 生成（TASK-33.1-1）。各工程の下位層への接続と完走は
//! TASK-33.1-2（#136）、入出力契約全体の統合は TASK-33.2 の対象で、
//! `main.rs` は解析成功後も工程を実行せず `runtime_error` を返す。
//!
//! TASK-33.2-2（#139）で、`package` の結果を stdout の JSON 1 つへ写す
//! [`stage_output`] と、stderr へテキストログを出す [`log`] を追加した。
//! stdout（結果 JSON）と stderr（ログ）は別の書き込み先として扱う。`package` の
//! 実処理への接続は #136 の範囲で、現状はテストハーネスでのみ確認している。

pub mod args;
pub mod error_report;
pub mod log;
pub mod output;
pub mod stage_output;
