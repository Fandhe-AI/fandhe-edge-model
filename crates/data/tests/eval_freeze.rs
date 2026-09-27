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
#[test]
fn freeze_eval_data_returns_frozen_with_known_hash_for_fixture() {
    // リポジトリルートからの相対パス（`cargo test` は crate ディレクトリを
    // カレントにするため、CARGO_MANIFEST_DIR 経由でリポジトリルートへ辿る）。
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/eval_freeze/sample.jsonl");

    let status = freeze_eval_data(Some(&fixture)).expect("fixture must freeze successfully");

    match status {
        EvalDataStatus::Frozen(record) => {
            assert_eq!(
                record.sha256,
                "c7fd9e6fd863a5de11eef802552013905ba7ce328e22c8f5fcd03c456dfb45a2"
            );
            assert_eq!(record.byte_len, 66);
            assert_eq!(record.path, fixture);
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
    let status = freeze_eval_data(None).expect("None path must never fail");
    assert_eq!(status, EvalDataStatus::NotProvided);
}

/// REQ-39 異常系: 存在しないパスを渡すと `FreezeError::NotFound` を返し、panic しない。
#[test]
fn freeze_eval_data_returns_not_found_for_missing_path() {
    let guard = TempDirGuard::new("missing-path");
    let missing = guard.path().join("does-not-exist.jsonl");

    let result = freeze_eval_data(Some(&missing));

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
    let oversized = guard.path().join("oversized.bin");

    let file = std::fs::File::create(&oversized).expect("create oversized fixture file");
    file.set_len(MAX_EVAL_DATA_BYTES + 1)
        .expect("set_len must succeed to create a sparse file");
    drop(file);

    let result = freeze_eval_data(Some(&oversized));

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
/// で拒否することを確認する（Linux 実機・テストハーネス。`/dev/zero` が存在しない
/// 環境では検証不能のため skip する）。
#[test]
fn freeze_eval_data_returns_not_a_file_for_character_device() {
    let dev_zero = Path::new("/dev/zero");
    if !dev_zero.exists() {
        eprintln!("skip: /dev/zero not available on this platform");
        return;
    }

    let result = freeze_eval_data(Some(dev_zero));

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

    let result = freeze_eval_data(Some(guard.path()));

    match result {
        Err(FreezeError::NotAFile) => {}
        Err(other) => panic!("expected NotAFile, got {other}"),
        Ok(_) => panic!("expected NotAFile error, got Ok"),
    }
}
