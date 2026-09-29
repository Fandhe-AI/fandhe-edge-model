//! 操作アダプター層の CLI（`fandhe-edge-cli`）ライブラリターゲット。
//!
//! PoC-16（`core-cli-vertical-slice`）と同様に lib/bin を分離し、業務ロジ
//! ックを持たない薄いアダプター関数（出力配線等）をここに置いて、テスト
//! ハーネスから直接呼べるようにする（`main.rs` の `[[bin]]` は
//! `CARGO_BIN_EXE_fandhe-edge` としてしか結合テストから叩けないため）。
//!
//! # 現状（TASK-21.1-2・TASK-21.2）
//!
//! [`output`] モジュールに、`fandhe-edge-core` の
//! `judgment::JudgmentResult` を JSON 1 行として書き出す
//! [`output::write_ok_judgment`]（TASK-21.1-2）と、`ErrorReport` を JSON 1
//! 行として書き出す [`output::write_error_report`]・各層のエラー型を
//! `ErrorReport` へ変換する薄い関数（`infer_input_error_report`・
//! `judgment_error_report`・`definition_error_report`。TASK-21.2）を実装
//! 済み。あわせて容量内訳を `package` 出力用 JSON にする
//! `output::package_capacity_json`・`write_package_capacity` と `capacity_error_report`
//! （TASK-30.1-2・#123。`package` 工程への配線は TASK-33.1）も実装済み。CLI バイナリ（`src/main.rs`）の 7 工程ディスパッチ（`register →
//! inspect → train → evaluate → select → package → infer`）への実配線・
//! 推論入力の受け取り（`--text`／`--input-file` の引数解析）は TASK-33.1、
//! stdout・stderr・終了コードの入出力契約全体の統合は TASK-33.2 の対象
//! で、いずれも本 crate ではまだ行っていない。`main.rs` は引き続き exit
//! 70・stdout 空のスタブ契約のまま（`.claude/rules/coding-rust.md`「操作
//! アダプターは薄く保つ」）。

pub mod output;
