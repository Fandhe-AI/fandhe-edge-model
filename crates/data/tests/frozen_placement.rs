//! 評価データの読み取り専用配置（[`fandhe_edge_data::frozen_placement`]）の
//! 結合テスト（REQ-39・REQ-17・TASK-17.2-2・issue #48）。
//!
//! 凍結（[`fandhe_edge_data::eval_freeze::freeze_eval_data`]）→ 配置
//! （[`fandhe_edge_data::frozen_placement::place_read_only`]。利用者の
//! `src` には触れず、本ツールが管理する `dest_dir` の中へ `hard_link` で
//! 公開する）→ 評価時の再計算（[`fandhe_edge_data::eval_freeze::evaluate_gate`]）
//! という呼び出し順序を、実データ相当の JSONL を使って一貫して確認する
//! （PoC-20 ケース 5
//! 〔`docs/spec/03-poc/safety-hardening/scripts/case5_eval_integrity.py`〕の
//! 再現。証拠の種別: テストハーネス）。
//!
//! 本ファイルのテストはすべて unix の permission bit（`0o400`）に依存する
//! ため `#![cfg(unix)]` でファイル全体を unix 限定にする（windows では
//! `File::set_permissions` の挙動が異なるため。モジュール doc「設計」）。
//! 個々のテストに `#[cfg(unix)]` を付けるだけでは、windows ビルドで
//! import・ヘルパー関数が未使用になり clippy `-D warnings` で fail する
//! （issue #227 CI 指摘: rust-ci (windows-latest) / cargo clippy）。

#![cfg(unix)]

use fandhe_edge_data::eval_freeze::{EvalDataState, EvaluateGate, evaluate_gate, freeze_eval_data};
use fandhe_edge_data::frozen_placement::{PlacementError, place_read_only};
use std::io::ErrorKind;
use std::path::PathBuf;

/// サンプル評価データ本体（`tests/eval_freeze.rs` と同形の JSONL・2 行）。
const SAMPLE: &[u8] =
    b"{\"id\":\"e1\",\"input\":\"hello\",\"output\":{\"intent\":\"greet\"}}\n{\"id\":\"e2\",\"input\":\"bye\",\"output\":{\"intent\":\"farewell\"}}\n";

/// テスト用の一時ファイルを、成否に関わらず削除するガード（RAII）。本テスト
/// では `src` を `place_read_only` が一切書き換えないため、削除前に権限を
/// 戻す処理は不要（POSIX の unlink は対象ファイル自身の書き込み権限を
/// 要求しない）。
struct TempFileGuard(PathBuf);

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn write_unique_temp_file(label: &str, bytes: &[u8]) -> TempFileGuard {
    let pid = std::process::id();
    for attempt in 0..1000u32 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let candidate = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-it-{pid}-{label}-{attempt}-{nanos}"
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                use std::io::Write as _;
                file.write_all(bytes).expect("書き込みに失敗しないはず");
                return TempFileGuard(candidate);
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => panic!("一時ファイルの作成に失敗しないはず: {err}"),
        }
    }
    panic!("一意な一時ファイルを作成できなかった");
}

/// テスト用の一時ディレクトリ（`dest_dir` 役）を、成否に関わらず削除する
/// ガード（RAII）。
struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `dest_dir` は呼び出し側が `0700` で用意する契約
/// （[`fandhe_edge_data::frozen_placement`] モジュール doc「`dest_dir` の
/// 機密性」）。テストヘルパーも同じ契約に従い、作成直後に明示的へ `0700`
/// を設定する。
fn make_temp_dir(label: &str) -> TempDirGuard {
    let pid = std::process::id();
    for attempt in 0..1000u32 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let candidate = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-it-dir-{pid}-{label}-{attempt}-{nanos}"
        ));
        match std::fs::create_dir(&candidate) {
            Ok(()) => {
                use std::os::unix::fs::PermissionsExt as _;
                std::fs::set_permissions(&candidate, std::fs::Permissions::from_mode(0o700))
                    .expect("テスト用ディレクトリの権限設定に失敗しないはず");
                return TempDirGuard(candidate);
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => panic!("テスト用ディレクトリの作成に失敗しないはず: {err}"),
        }
    }
    panic!("一意なテスト用ディレクトリを作成できなかった");
}

fn running_as_root(path: &std::path::Path) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    std::fs::metadata(path)
        .map(|meta| meta.uid() == 0)
        .unwrap_or(false)
}

/// REQ-39・REQ-17・TASK-17.2-2: 凍結 → 配置 → 評価ゲートの一連の流れで、
/// (a) append 書き込みが拒否される、(b) truncate 書き込みが拒否される、
/// (c) 読み戻したバイト列が元の凍結記録と一致し `evaluate_gate` が
/// `Proceed` になる、(d) unix では mode が `0o400` と完全一致する、
/// (e) 利用者の `src` は変わらず書き込み可能なまま残る、ことを確認する
/// （PoC-20 ケース 5 相当。証拠の種別: テストハーネス）。
#[test]
fn req39_req17_freeze_then_place_read_only_then_evaluate_gate_proceeds() {
    let src = write_unique_temp_file("acceptance", SAMPLE);
    if running_as_root(&src.0) {
        // root 分岐は `req39_place_read_only_fails_closed_when_root_via_integration` で扱う。
        return;
    }
    let dest_dir = make_temp_dir("acceptance-dest");

    let record = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");

    let placement = place_read_only(
        &src.0,
        &dest_dir.0,
        "frozen.jsonl",
        &record,
        SAMPLE.len() as u64,
    )
    .expect("非 root では配置が成功するはず");
    assert_eq!(
        placement.mode(),
        0o400,
        "(d) mode が 0o400 と完全一致すること"
    );

    // (a) append 書き込みは拒否される。
    match std::fs::OpenOptions::new()
        .append(true)
        .open(placement.path())
    {
        Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
        Ok(_) => panic!("append open は拒否されるはず"),
    }

    // (b) truncate 書き込みも拒否される。
    match std::fs::write(placement.path(), b"tampered") {
        Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
        Ok(()) => panic!("truncate write は拒否されるはず"),
    }

    // (c) 読み戻したバイト列は変わっておらず、`evaluate_gate` が `Proceed` になる。
    let actual_bytes = std::fs::read(placement.path()).expect("読み取りは成功するはず");
    assert_eq!(actual_bytes, SAMPLE);
    let state = EvalDataState::Frozen(record.clone());
    assert_eq!(
        evaluate_gate(&state, &actual_bytes),
        Ok(EvaluateGate::Proceed(record))
    );

    // (e) 利用者の `src` は変わらず書き込み可能なまま残る（本モジュールは
    // `src` を chmod・rename・unlink しない）。
    use std::os::unix::fs::PermissionsExt as _;
    let src_mode = std::fs::metadata(&src.0)
        .expect("src のメタデータを取得できるはず")
        .permissions()
        .mode()
        & 0o7777;
    assert_ne!(src_mode, 0o400, "src の権限は変更されないはず");
    let src_content = std::fs::read(&src.0).expect("src の読み取りは成功するはず");
    assert_eq!(src_content, SAMPLE, "src の内容は変更されないはず");
}

/// REQ-39・TASK-17.2-2: root 実行下では `place_read_only` が
/// `WriteNotRejected` で fail-closed に失敗し、`dest_dir` には何も公開
/// されない（配置済みを装わない）。
#[test]
fn req39_place_read_only_fails_closed_when_root_via_integration() {
    let src = write_unique_temp_file("root-acceptance", SAMPLE);
    if !running_as_root(&src.0) {
        return;
    }
    let dest_dir = make_temp_dir("root-acceptance-dest");

    let record = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");
    match place_read_only(
        &src.0,
        &dest_dir.0,
        "frozen.jsonl",
        &record,
        SAMPLE.len() as u64,
    ) {
        Err(PlacementError::WriteNotRejected { .. }) => {}
        other => panic!("root では WriteNotRejected を期待したが {other:?} だった"),
    }
    assert!(!dest_dir.0.join("frozen.jsonl").exists());
}

/// REQ-17・REQ-39: `src` の内容が凍結記録と食い違っている場合（差し替え・
/// 改変）、`place_read_only` は `HashMismatch` で拒否し、`evaluate_gate` を
/// 呼ぶまでもなく配置の時点で fail-closed に止まることを確認する
/// （issue #227 P0 指摘の結合テスト版）。
#[test]
fn req17_place_read_only_rejects_tampered_src_before_evaluate_gate() {
    // 元データと同じ長さのまま改変した内容を `src` に置く。
    let tampered = {
        let mut bytes = SAMPLE.to_vec();
        if let Some(first) = bytes.first_mut() {
            *first = b'X';
        }
        bytes
    };
    let src = write_unique_temp_file("tampered-src", &tampered);
    let dest_dir = make_temp_dir("tampered-dest");

    let record = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");
    match place_read_only(
        &src.0,
        &dest_dir.0,
        "frozen.jsonl",
        &record,
        SAMPLE.len() as u64,
    ) {
        Err(PlacementError::HashMismatch) => {}
        other => panic!("HashMismatch を期待したが {other:?} だった"),
    }
    assert!(!dest_dir.0.join("frozen.jsonl").exists());
}
