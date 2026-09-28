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
//! （`fandhe-edge-core`）に加え、評価器（`fandhe-edge-eval`）にも依存する
//! （[`selection_significance`] が選定結果への McNemar・Holm 有意性判定の
//! 付与に評価器を呼ぶ。評価ロジックは再実装しない。REQ-18・TASK-18.3-1・
//! #87）。操作アダプター（CLI・TUI・MCP）・推論ランタイムはこの crate に
//! 依存しない（推論経路に学習側の型を持ち込まない。REQ-32・
//! `.claude/rules/coding-rust.md`）。
//!
//! [`time_allotment`] は探索予算（REQ-18。既定 1 時間）のうち候補 1 件へ
//! 配分する持ち時間の算出・実行記録を担う（TASK-18.1-1・issue #83）。
//! [`search`] はそれを繰り返し呼び、探索予算全体の消費を追跡し、
//! validation 正解率が最も高い候補を選定する（TASK-18.1-2・issue #84）。
//! 正解率の算出は評価器 `fandhe-edge-eval`
//! （[`fandhe_edge_eval::metrics::evaluate_single_select`]）に委譲する
//! （評価器は 1 つに集約し、他の層で再実装しない。
//! `.claude/rules/coding-rust.md`「crate 構成と層の境界」）。
//!
//! # スコープ外（#178 以降）
//!
//! - 子プロセスの起動・タイムアウト・終了コード写像（ワーカーの `code` から
//!   [`fandhe_edge_core::exitcode::ExitCode`] への対応づけ）。
//!   [`time_allotment::CandidateRunner`]・[`search::ValidationScorer`] は
//!   この実行器・推論器を差し込むための接合点（trait）のみを用意する
//! - `kind` ごとの `config` 検証（`config` は「JSON オブジェクトであること」
//!   だけを検査する。`kinds/c1.py`・`c3.py` の `_validate_config` は再実装
//!   しない）
//! - ファイルシステムへの経路の閉じ込め（`root`・`train_path`・`out_dir` は
//!   文字列としての構文検査に留め、存在確認・dir_fd による閉じ込めは学習
//!   ワーカー自身の多層防御（`trainer/src/fandhe_edge_trainer/guard.py`）と
//!   将来のガード層（TASK-39.x）が担う）
//! - 「予算到達」を合格扱いしない判定（TASK-18.2・issue #85）
//! - McNemar・Holm による有意性判定の選定記録への統合（TASK-18.3-1・issue #87）
//! - 探索記録のファイルへの永続化・CLI `select` 工程の JSON 出力・終了コード
//!   への写像（TASK-33.x）

pub mod error;
mod kind_defaults;
pub mod limits;
pub mod request;
pub mod result;
pub mod search;
// 選定結果への McNemar・Holm 有意性判定の付与（評価器
// `fandhe-edge-eval` を呼ぶ。REQ-18・REQ-25・TASK-18.3-1・#87）。
pub mod selection_significance;
pub mod time_allotment;
