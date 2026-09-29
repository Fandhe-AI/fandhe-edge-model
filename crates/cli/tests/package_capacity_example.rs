//! `examples/package_capacity.rs` の単体テストを既定のテスト集合（`cargo test --workspace`）で
//! 実行するための取り込み（REQ-39・REQ-21・TASK-30.1-2・#123）。
//!
//! example 内の `#[cfg(test)]` モジュールは `cargo test` では実行されないため、同じソースを
//! モジュールとして取り込み、経路の閉じ込め・引数件数上限・ファイル系エラー写像のテストを
//! CI で実際に走らせる。`main` はこのテストからは使わないので dead_code を許可する。

#[allow(dead_code)]
#[path = "../examples/package_capacity.rs"]
mod package_capacity;
