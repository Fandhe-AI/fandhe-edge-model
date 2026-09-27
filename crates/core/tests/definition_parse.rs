//! `Definition::load`（ファイル読み込み経路）の結合テスト（REQ-15・REQ-39）。
//!
//! ユニットテスト（`definition.rs` 内）は文字列パースのみを検証するため、
//! ここではファイルシステム経由の読み込み（サイズ確認 → 読み込み → パース）を
//! 一気通貫で確認する。一時ファイルはリポジトリ外（`std::env::temp_dir()`）に
//! 作成し、テスト終了時に削除する。

use fandhe_edge_core::definition::{
    Definition, DefinitionError, FieldPath, MAX_DEFINITION_FILE_BYTES,
};
use std::io::Write;
use std::path::PathBuf;

/// テスト専用の一時ファイルパスを発行する（プロセス ID とテスト名で衝突を避ける）。
fn temp_file_path(test_name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "fandhe-edge-core-definition-{test_name}-{}.json",
        std::process::id()
    ))
}

#[test]
fn req15_load_reads_definition_file_from_disk() {
    let path = temp_file_path("load-ok");
    let json = r#"{
        "schema": "fandhe-edge-model-definition/v1",
        "name": "sample_topic",
        "version": 1,
        "judgment_type": "single_select",
        "options": [
            { "id": "yes", "display_name": "Yes", "description": "肯定" },
            { "id": "no", "display_name": "No", "description": "否定" }
        ],
        "io": { "input": "bytes" }
    }"#;
    std::fs::write(&path, json).expect("一時ファイルを書き込めるはず");

    let result = Definition::load(&path);
    std::fs::remove_file(&path).expect("一時ファイルを削除できるはず");

    let def = result.expect("ファイルから正常にパースできるはず");
    assert_eq!(def.options().len(), 2);
    assert_eq!(def.options()[0].id, "yes");
    assert_eq!(def.options()[1].id, "no");
}

#[test]
fn req39_load_rejects_file_larger_than_size_limit() {
    let path = temp_file_path("load-too-large");
    let mut file = std::fs::File::create(&path).expect("一時ファイルを作成できるはず");
    // 中身は JSON として不正だが、サイズ上限チェックは内容を読む前に働くため
    // パース可否には到達しない（意図的にダミーバイト列を書く）。
    let oversized_len = MAX_DEFINITION_FILE_BYTES + 1;
    let chunk = vec![b'a'; 8192];
    let mut written: u64 = 0;
    while written < oversized_len {
        let remaining = oversized_len - written;
        let to_write = remaining.min(chunk.len() as u64) as usize;
        file.write_all(&chunk[..to_write])
            .expect("一時ファイルへ書き込めるはず");
        written += to_write as u64;
    }
    drop(file);

    let result = Definition::load(&path);
    std::fs::remove_file(&path).expect("一時ファイルを削除できるはず");

    match result.expect_err("サイズ上限超過は拒否されるはず") {
        DefinitionError::TooLarge { size, limit, .. } => {
            assert_eq!(limit, MAX_DEFINITION_FILE_BYTES);
            assert!(size > limit);
        }
        other => panic!("TooLarge を期待したが {other:?} だった"),
    }
}

/// FIFO（名前付きパイプ）を指すパスを `Definition::load` に渡すと、書き手が
/// 現れなくても即座に拒否されること（無期限に停止しない）を確認する
/// （REQ-39・security.md「ガード層: 資源の上限」。PR #187 レビュー指摘）。
/// テストが実際に無期限停止した場合は harness のタイムアウトで検出される。
/// `O_NONBLOCK` で開くのは Linux・macOS だけ（`Definition::open_for_read`）で、
/// 他の Unix では書き手のいない FIFO を開く時点で停止するため、対象 OS を揃える。
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn req39_load_rejects_fifo_without_blocking() {
    let path = temp_file_path("load-fifo");
    // std に mkfifo 相当の API が無いため、テスト専用に `mkfifo` コマンドで
    // FIFO を作成する（本体コードでは子プロセスを起動しない）。
    let status = std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("mkfifo コマンドを起動できるはず");
    assert!(status.success(), "mkfifo が成功するはず");

    // 書き手が存在しない FIFO に対して呼ぶ。以前の実装（無条件の
    // `File::open`）はここでプロセスごと無期限に停止しうる。
    let result = Definition::load(&path);
    std::fs::remove_file(&path).expect("FIFO を削除できるはず");

    match result.expect_err("FIFO は通常ファイルではないため拒否されるはず") {
        DefinitionError::NotRegularFile { path: rejected } => {
            assert_eq!(rejected, path);
        }
        other => panic!("NotRegularFile を期待したが {other:?} だった"),
    }
}

/// TASK-15.3-2: `Definition::load`（ファイル経由）でも `version` の欠落が
/// 型付きエラー `MissingField { field: FieldPath::Version }` として返る
/// ことを確認する（`parse` 単体のユニットテストとは別に、ファイル読み込み
/// 経路まで通しで確認する結合テスト）。
#[test]
fn req15_load_rejects_definition_file_missing_required_field() {
    let path = temp_file_path("load-missing-version");
    // `version` を欠いた定義ファイル。
    let json = r#"{
        "schema": "fandhe-edge-model-definition/v1",
        "name": "sample_topic",
        "judgment_type": "single_select",
        "options": [
            { "id": "yes", "display_name": "Yes", "description": "肯定" }
        ],
        "io": { "input": "bytes" }
    }"#;
    std::fs::write(&path, json).expect("一時ファイルを書き込めるはず");

    let result = Definition::load(&path);
    std::fs::remove_file(&path).expect("一時ファイルを削除できるはず");

    match result.expect_err("必須項目欠落は拒否されるはず") {
        DefinitionError::MissingField { field } => {
            assert_eq!(field, FieldPath::Version);
            assert_eq!(field.to_string(), "version");
        }
        other => panic!("MissingField を期待したが {other:?} だった"),
    }
}
