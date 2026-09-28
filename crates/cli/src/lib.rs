//! 操作アダプター層の CLI（`fandhe-edge-cli`）ライブラリターゲット。
//!
//! PoC-16（`core-cli-vertical-slice`）と同様に lib/bin を分離し、業務ロジ
//! ックを持たない薄いアダプター関数（出力配線等）をここに置いて、テスト
//! ハーネスから直接呼べるようにする（`main.rs` の `[[bin]]` は
//! `CARGO_BIN_EXE_fandhe-edge` としてしか結合テストから叩けないため）。
//!
//! # 現状（TASK-21.1-2）
//!
//! [`output`] モジュールに、`fandhe-edge-core` の
//! `judgment::JudgmentResult` を JSON 1 行として書き出す
//! [`output::write_ok_judgment`] のみを実装済み。CLI バイナリ
//! （`src/main.rs`）の 7 工程ディスパッチ（`register → inspect → train →
//! evaluate → select → package → infer`）への実配線は TASK-33.1、stdout・
//! stderr・終了コードの入出力契約全体の統合は TASK-33.2 の対象で、いずれ
//! も本 crate ではまだ行っていない。`main.rs` は引き続き exit 70・stdout
//! 空のスタブ契約のまま（`.claude/rules/coding-rust.md`「操作アダプター
//! は薄く保つ」）。

pub mod output;
