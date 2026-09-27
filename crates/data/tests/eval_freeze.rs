//! `freeze_eval_data` の結合テスト（REQ-17）。
//!
//! 証拠種別: テストハーネス（Mac 実機・GPU 不要。`.claude/rules/ci.md` の
//! 実機前提テストには該当しない）。
//!
//! 一時ファイルは `tempfile` crate（新規依存の追加は本 issue の範囲外）を使わず、
//! `std::env::temp_dir()` 配下にプロセス ID とテスト名を含む一意なサブディレクトリを
//! 作成し、テスト終了時に削除する。

use std::path::{Path, PathBuf};

use fandhe_edge_core::eval_data::EvalDataStatus;
use fandhe_edge_data::eval_freeze::{FreezeError, freeze_eval_data};
use fandhe_edge_data::limits::MAX_EVAL_DATA_BYTES;

/// テストごとに一意な一時ディレクトリを作り、後始末までを面倒みるガード。
struct TempDirGuard {
    dir: PathBuf,
}

impl TempDirGuard {
    fn new(test_name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-data-eval-freeze-{}-{}-{}",
            std::process::id(),
            test_name,
            // 同一プロセス内でテストが並行実行されても衝突しないよう、
            // アドレス値（実行のたびに変わり得る値）も混ぜる。
            &format!("{:p}", &test_name) as &str
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir for test");
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// REQ-17 正常系（具体値）: 固定フィクスチャを凍結すると、事前計算済みの
/// sha256 具体値・ファイルサイズと一致する `FreezeRecord` が返る。
/// `root` はフィクスチャの親ディレクトリ、`path` はそこからの相対パスとして渡す
/// （REQ-39 経路の閉じ込め。`root` 配下であることを検証してから開く）。
#[test]
fn freeze_eval_data_returns_frozen_with_known_hash_for_fixture() {
    // リポジトリルートからの相対パス（`cargo test` は crate ディレクトリを
    // カレントにするため、CARGO_MANIFEST_DIR 経由でリポジトリルートへ辿る）。
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/eval_freeze");
    let relative = Path::new("sample.jsonl");
    let canonical_fixture =
        std::fs::canonicalize(root.join(relative)).expect("fixture must exist and canonicalize");

    let status = freeze_eval_data(&root, Some(relative)).expect("fixture must freeze successfully");

    match status {
        EvalDataStatus::Frozen(record) => {
            assert_eq!(
                record.sha256,
                "c7fd9e6fd863a5de11eef802552013905ba7ce328e22c8f5fcd03c456dfb45a2"
            );
            assert_eq!(record.byte_len, 66);
            // 記録されるのは正規化後の実体パス（経路の閉じ込め検証の結果）であり、
            // 呼び出し元が渡した相対パスそのものではない。
            assert_eq!(record.path, canonical_fixture);
        }
        EvalDataStatus::NotProvided => panic!("expected Frozen, got NotProvided"),
        _ => panic!("unexpected EvalDataStatus variant"),
    }
}

/// REQ-17 境界値: 評価データが与えられない場合、I/O を発生させず `NotProvided` を
/// 返す（PoC-16 縦断 2 の `evaluate` `status:"skipped"`・exit 0 に対応する境界。
/// ライブラリレベルのテストハーネス証跡であり、CLI 配線自体は TASK-33.x の範囲）。
#[test]
fn freeze_eval_data_returns_not_provided_when_path_is_none() {
    let guard = TempDirGuard::new("none-path");
    let status = freeze_eval_data(guard.path(), None).expect("None path must never fail");
    assert_eq!(status, EvalDataStatus::NotProvided);
}

/// REQ-39 異常系: 存在しないパスを渡すと `FreezeError::NotFound` を返し、panic しない。
#[test]
fn freeze_eval_data_returns_not_found_for_missing_path() {
    let guard = TempDirGuard::new("missing-path");
    let missing = Path::new("does-not-exist.jsonl");

    let result = freeze_eval_data(guard.path(), Some(missing));

    match result {
        Err(FreezeError::NotFound(_)) => {}
        Err(other) => panic!("expected NotFound, got {other}"),
        Ok(_) => panic!("expected NotFound error, got Ok"),
    }
}

/// REQ-39 境界値（サイズ超過）: `MAX_EVAL_DATA_BYTES` を超えるファイルを渡すと
/// `FreezeError::TooLarge` を返す（`File::open` 前の `metadata()` で拒否するため、
/// ファイルは疎ファイルとして作成し実データを書き込まずにサイズだけ確保する）。
#[test]
fn freeze_eval_data_returns_too_large_for_oversized_file() {
    let guard = TempDirGuard::new("oversized-file");
    let relative = Path::new("oversized.bin");
    let oversized = guard.path().join(relative);

    let file = std::fs::File::create(&oversized).expect("create oversized fixture file");
    file.set_len(MAX_EVAL_DATA_BYTES + 1)
        .expect("set_len must succeed to create a sparse file");
    drop(file);

    let result = freeze_eval_data(guard.path(), Some(relative));

    match result {
        Err(FreezeError::TooLarge { limit, actual }) => {
            assert_eq!(limit, MAX_EVAL_DATA_BYTES);
            assert_eq!(actual, MAX_EVAL_DATA_BYTES + 1);
        }
        Err(other) => panic!("expected TooLarge, got {other}"),
        Ok(_) => panic!("expected TooLarge error, got Ok"),
    }
}

/// REQ-39 異常系（非通常ファイル）: `stat` 上のサイズが 0 のキャラクタデバイス
/// （`/dev/zero`）は、ファイル種別の検証が無いと `TooLarge` 検査を素通りしたうえ
/// `sha256_hex_of_reader` が EOF に到達せず無限に読み続ける。`FreezeError::NotAFile`
/// で拒否することを確認する（Linux/macOS 実機・テストハーネス）。
/// `/dev/zero` は Unix 系では常に存在するため `#[cfg(unix)]` で対象環境を絞り、
/// 実行時の存在チェックによる非対応環境での暗黙 skip（false pass）を避ける
/// （`.claude/rules/coding-rust.md`「テストの skip・ignore…で CI を通さない」）。
/// `root` に `/dev` を渡し、経路の閉じ込め検証自体は通過させたうえで
/// ファイル種別の検証に到達することを確認する。
#[test]
#[cfg(unix)]
fn freeze_eval_data_returns_not_a_file_for_character_device() {
    let root = Path::new("/dev");
    let relative = Path::new("zero");

    let result = freeze_eval_data(root, Some(relative));

    match result {
        Err(FreezeError::NotAFile) => {}
        Err(other) => panic!("expected NotAFile, got {other}"),
        Ok(_) => panic!("expected NotAFile error, got Ok"),
    }
}

/// REQ-39 異常系（TOCTOU / FIFO）: 書き手の無い FIFO を渡しても無限にブロックせず
/// `FreezeError::NotAFile` で拒否することを確認する（codex レビュー P0 指摘の
/// 回帰テスト）。
///
/// 本テストが検証するのは `freeze_eval_data` 冒頭のパスレベルの `is_file()`
/// 事前チェック（`metadata(path)` の時点で FIFO と判定して拒否する経路）であり、
/// `open_without_blocking` 自体のノンブロッキング挙動（`metadata()` と `open()` の
/// 間でパスが差し替えられる本来の TOCTOU）はここでは経由しない
/// （このテストのパスは最初から FIFO のため、`open_without_blocking` に到達する前に
/// 事前チェックで `NotAFile` が返る。Cursor Bugbot 指摘）。TOCTOU window 自体の
/// ロックインは `crates/data/src/eval_freeze.rs` の
/// `open_without_blocking_tests::open_without_blocking_returns_promptly_for_writerless_fifo`
/// （`open_without_blocking` を直接呼ぶ単体テスト）が担う。
/// 本テストは無限ブロックしないことの結合レベルでの確認として残す
/// （万一ブロッキング実装へ回帰した場合にテストスイート自体が無期限にハングしない
/// よう、判定は別スレッド＋タイムアウトで行う。Linux/macOS 実機・テストハーネス）。
#[test]
#[cfg(unix)]
fn freeze_eval_data_returns_not_a_file_for_fifo_without_blocking() {
    let guard = TempDirGuard::new("fifo-path");
    let relative = Path::new("eval.fifo");
    let fifo_path = guard.path().join(relative);

    let status = std::process::Command::new("mkfifo")
        .arg(&fifo_path)
        .status()
        .expect("mkfifo command must be available on unix test runners");
    assert!(status.success(), "mkfifo must exit successfully");

    let (tx, rx) = std::sync::mpsc::channel();
    let root = guard.path().to_path_buf();
    let relative = relative.to_path_buf();
    std::thread::spawn(move || {
        let result = freeze_eval_data(&root, Some(&relative));
        // メインスレッドがタイムアウトで抜けた後に送信が失敗しても
        // （受信側が既に drop 済み）テストの成否には影響しないため無視する。
        let _ = tx.send(result);
    });

    let result = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("freeze_eval_data must not block indefinitely on a writerless FIFO");

    match result {
        Err(FreezeError::NotAFile) => {}
        Err(other) => panic!("expected NotAFile, got {other}"),
        Ok(_) => panic!("expected NotAFile error, got Ok"),
    }
}

/// REQ-39 異常系（非通常ファイル）: ディレクトリを渡しても `NotAFile` で拒否する
/// （`metadata()` 自体は成功するため、`is_file()` による種別検証が効いていることの
/// 確認）。
#[test]
fn freeze_eval_data_returns_not_a_file_for_directory() {
    let guard = TempDirGuard::new("directory-path");
    let relative = Path::new("subdir");
    std::fs::create_dir_all(guard.path().join(relative)).expect("create subdirectory");

    let result = freeze_eval_data(guard.path(), Some(relative));

    match result {
        Err(FreezeError::NotAFile) => {}
        Err(other) => panic!("expected NotAFile, got {other}"),
        Ok(_) => panic!("expected NotAFile error, got Ok"),
    }
}

/// REQ-39 異常系（経路の閉じ込め・codex P0 回帰テスト）: 絶対パスを渡すと
/// ファイルシステムへ触れる前に `FreezeError::OutsideRoot` で拒否する
/// （`root.join(absolute_path)` は `root` を無視して絶対パスを返してしまうため、
/// `join` の前に構文検証で弾く必要がある）。
#[test]
fn freeze_eval_data_returns_outside_root_for_absolute_path() {
    let guard = TempDirGuard::new("outside-root-absolute");
    let outside = guard.path().join("elsewhere.jsonl");
    std::fs::write(&outside, b"outside root").expect("write file outside root");

    let result = freeze_eval_data(guard.path(), Some(outside.as_path()));

    match result {
        Err(FreezeError::OutsideRoot) => {}
        Err(other) => panic!("expected OutsideRoot, got {other}"),
        Ok(_) => panic!("expected OutsideRoot error, got Ok"),
    }
}

/// REQ-39 異常系（経路の閉じ込め・codex P0 回帰テスト）: `..` で親ディレクトリへ
/// 脱出しようとするパスを `FreezeError::OutsideRoot` で拒否する。
#[test]
fn freeze_eval_data_returns_outside_root_for_parent_dir_component() {
    let guard = TempDirGuard::new("outside-root-parent");
    let escaping = Path::new("../escape.jsonl");

    let result = freeze_eval_data(guard.path(), Some(escaping));

    match result {
        Err(FreezeError::OutsideRoot) => {}
        Err(other) => panic!("expected OutsideRoot, got {other}"),
        Ok(_) => panic!("expected OutsideRoot error, got Ok"),
    }
}

/// REQ-39 異常系（経路の閉じ込め・codex P0 回帰テスト）: `path` 自体は `root` 配下の
/// 表記でも、symlink の解決先が `root` の外にある場合は `FreezeError::OutsideRoot`
/// で拒否する（`canonicalize` 後の実体パスで確認する検証の対象）。
#[test]
#[cfg(unix)]
fn freeze_eval_data_returns_outside_root_for_symlink_escaping_root() {
    let root_guard = TempDirGuard::new("symlink-root");
    let outside_guard = TempDirGuard::new("symlink-outside-target");
    let outside_file = outside_guard.path().join("secret.jsonl");
    std::fs::write(&outside_file, b"secret").expect("write outside target file");

    let relative = Path::new("link.jsonl");
    std::os::unix::fs::symlink(&outside_file, root_guard.path().join(relative))
        .expect("create symlink escaping root");

    let result = freeze_eval_data(root_guard.path(), Some(relative));

    match result {
        Err(FreezeError::OutsideRoot) => {}
        Err(other) => panic!("expected OutsideRoot, got {other}"),
        Ok(_) => panic!("expected OutsideRoot error, got Ok"),
    }
}
