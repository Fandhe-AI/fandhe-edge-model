//! 操作アダプター層の CLI（`fandhe-edge` バイナリ）。
//!
//! 最終的には `register → inspect → train → evaluate → select → package → infer`
//! の 7 工程（REQ-33）を TASK-33.1 で実装し、stdout の JSON 出力・stderr・
//! 終了コードの入出力契約を TASK-33.2 で確定させる。TUI・MCP / Codex 連携
//! （REQ-35〜37）もここで確定する契約を再利用する想定で、CLI 以外の口に
//! 別の契約を作らない。
//!
//! 現状は TASK-15.2 の crate 雛形であり、どの引数を渡しても工程は一切実行
//! しない（実装済みを装わない）。crate 構成は PoC-16
//! （`core-cli-vertical-slice`）の lib/bin 分離を踏襲しつつ、共通コアと
//! アダプターを別 crate に分けている（`.claude/rules/coding-rust.md`「アダ
//! プターは薄く保ち、業務ロジックは下位層に置く」）。業務ロジックは
//! `fandhe-edge-core` 側に置き、本 crate には持ち込まない。

use fandhe_edge_core::exitcode::ExitCode;

fn main() -> std::process::ExitCode {
    // 引数は読まない（工程を一切実行しないため std::env::args を使わない）。
    // stdout は空のまま終える。CLI の出力契約（1 呼び出しにつき JSON 1 つ。
    // TASK-33.2）をここで先取りしないためで、stdout に何か出すこと自体が
    // 契約の先取りになる。
    eprintln!("fandhe-edge: not implemented yet (TASK-33.1)");
    ExitCode::RuntimeError.into()
}
