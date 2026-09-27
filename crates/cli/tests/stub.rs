//! CLI 雛形（TASK-15.2）の結合テスト。
//!
//! 本 crate はまだ 7 工程（REQ-33）を実装しておらず、バイナリを起動すると
//! stdout を空のまま終了コード 70 で終える契約になっている（`main.rs` 参
//! 照）。TASK-33.1 で CLI を実装したら本テストは実際の工程を検証するテス
//! トへ置き換える。

use std::process::Command;

/// TASK-15.2 の雛形の挙動を確認する。終了コード 70 は REQ-21 の
/// `runtime_error`。TASK-33.1 で CLI を実装したらこのテストは置き換える。
#[test]
fn stub_binary_exits_with_runtime_error_and_empty_stdout() {
    let output = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .output()
        .expect("failed to spawn fandhe-edge stub binary");

    assert_eq!(
        output.status.code(),
        Some(70),
        "stub binary must exit with REQ-21 runtime_error (70)"
    );
    assert_eq!(
        output.stdout.len(),
        0,
        "stub binary must not write to stdout (JSON output contract is not yet implemented)"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not implemented"),
        "stub binary stderr must mention 'not implemented', got: {stderr}"
    );
}
