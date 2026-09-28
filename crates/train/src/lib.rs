//! 学習ワーカー層（Rust 側呼び出し元）の JSON 境界契約（`fandhe-edge-train`）。
//!
//! Rust 側 CLI／ジョブ管理（REQ-34。#178 で子プロセス起動・タイムアウト・
//! 終了コード写像を実装予定）が、学習ワーカー（`trainer/`・Python）を子プロセ
//! スとして起動して JSON で学習を依頼する経路のうち、本 crate は「学習リクエ
//! スト」と「学習結果（成功時の成果物記録・失敗時のエラー）」の 2 つの JSON
//! を検証付きの型で表す（REQ-18・REQ-19・REQ-34。issue #177。子プロセスの
//! 起動そのもの・終了コード写像は #178 の対象で本 crate には含まない）。
//!
//! # 単一真実源
//!
//! `coding-python.md`「入出力の JSON スキーマは Rust 側の定義を正とし、
//! Python 側で独自のフィールドを増やさない」の方針に基づき、本 crate が
//! スキーマの正となる。ただし実装（検証順序・フィールド名・上限値）は
//! 既存の Python 実装（`trainer/src/fandhe_edge_trainer/contract.py`・
//! `limits.py`・`artifact.py`）から書き起こしたものであり、両者の解釈が
//! 一致することを共有 fixture（`fixtures/train_contract/`）で機械照合する
//! （`crates/train/tests/train_contract_fixture.rs` と
//! `trainer/tests/test_train_contract_fixture.py`）。
//!
//! # 層の境界
//!
//! 学習ワーカー層（`trainer/` の呼び出し元）に位置し、共通コア
//! （`fandhe-edge-core`）にのみ依存する（`label_order` を定義ファイル
//! （`Definition`）から投影するため）。操作アダプター（CLI・TUI・MCP）・
//! 推論ランタイムはこの crate に依存しない（推論経路に学習側の型を持ち込ま
//! ない。REQ-32・`.claude/rules/coding-rust.md`）。
//!
//! # スコープ外（#178 以降）
//!
//! - 子プロセスの起動・タイムアウト・終了コード写像（ワーカーの `code` から
//!   [`fandhe_edge_core::exitcode::ExitCode`] への対応づけ）
//! - `kind` ごとの `config` 検証（`config` は「JSON オブジェクトであること」
//!   だけを検査する。`kinds/c1.py`・`c3.py` の `_validate_config` は再実装
//!   しない）
//! - ファイルシステムへの経路の閉じ込め（`root`・`train_path`・`out_dir` は
//!   文字列としての構文検査に留め、存在確認・dir_fd による閉じ込めは学習
//!   ワーカー自身の多層防御（`trainer/src/fandhe_edge_trainer/guard.py`）と
//!   将来のガード層（TASK-39.x）が担う）

pub mod error;
pub mod limits;
pub mod request;
pub mod result;
