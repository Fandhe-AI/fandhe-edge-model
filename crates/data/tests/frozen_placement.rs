//! 評価データの読み取り専用配置（[`fandhe_edge_data::frozen_placement`]）の
//! 結合テスト（REQ-39・REQ-17・TASK-17.2-2・issue #48）。
//!
//! 凍結（[`fandhe_edge_data::eval_freeze::freeze_eval_data`]）→ 配置
//! （[`fandhe_edge_data::frozen_placement::place_read_only`]）→ 評価時の
//! 再計算（[`fandhe_edge_data::eval_freeze::evaluate_gate`]）という呼び出し
//! 順序を、実データ相当の JSONL を使って一貫して確認する（PoC-20 ケース 5
//! 〔`docs/spec/03-poc/safety-hardening/scripts/case5_eval_integrity.py`〕の
//! 再現。証拠の種別: テストハーネス）。
//!
//! 本ファイルのテストはすべて unix の permission bit（`0o444`）に依存する
//! ため `#![cfg(unix)]` でファイル全体を unix 限定にする（windows では
//! `File::set_permissions` の挙動が異なり読み取り専用ハンドル経由の
//! chmod が成立しないため。モジュール doc「手順」）。個々のテストに
//! `#[cfg(unix)]` を付けるだけでは、windows ビルドで import・ヘルパー
//! 関数が未使用になり clippy `-D warnings` で fail する（issue #227 CI
//! 指摘: rust-ci (windows-latest) / cargo clippy）。

#![cfg(unix)]

use fandhe_edge_data::eval_freeze::{EvalDataState, EvaluateGate, evaluate_gate, freeze_eval_data};
use fandhe_edge_data::frozen_placement::{
    PlacementError, place_read_only, verify_direct_write_rejected,
};
use std::io::ErrorKind;
use std::path::PathBuf;

/// サンプル評価データ本体（`tests/eval_freeze.rs` と同形の JSONL・2 行）。
const SAMPLE: &[u8] =
    b"{\"id\":\"e1\",\"input\":\"hello\",\"output\":{\"intent\":\"greet\"}}\n{\"id\":\"e2\",\"input\":\"bye\",\"output\":{\"intent\":\"farewell\"}}\n";

/// テスト用の一時ファイルを、成否に関わらず削除するガード（RAII）。
/// 削除の前に権限を書き込み可へ戻す。
struct TempFileGuard(PathBuf);

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o644));
        }
        #[cfg(not(unix))]
        {
            if let Ok(meta) = std::fs::metadata(&self.0) {
                let mut perms = meta.permissions();
                #[allow(
                    clippy::permissions_set_readonly_false,
                    reason = "テスト後片付けで Windows の読み取り専用属性を解除するため"
                )]
                perms.set_readonly(false);
                let _ = std::fs::set_permissions(&self.0, perms);
            }
        }
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

#[cfg(unix)]
fn running_as_root(path: &std::path::Path) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    std::fs::metadata(path)
        .map(|meta| meta.uid() == 0)
        .unwrap_or(false)
}

/// REQ-39・REQ-17・TASK-17.2-2: 凍結 → 配置 → 評価ゲートの一連の流れで、
/// (a) append 書き込みが拒否される、(b) truncate 書き込みが拒否される、
/// (c) 読み戻したバイト列が元の凍結記録と一致し `evaluate_gate` が
/// `Proceed` になる、(d) unix では mode が `0o444` と完全一致する、
/// ことを確認する（PoC-20 ケース 5 相当。証拠の種別: テストハーネス）。
#[cfg(unix)]
#[test]
fn req39_req17_freeze_then_place_read_only_then_evaluate_gate_proceeds() {
    let guard = write_unique_temp_file("acceptance", SAMPLE);
    if running_as_root(&guard.0) {
        // root 分岐は `req39_place_read_only_fails_closed_when_root_via_integration` で扱う。
        return;
    }

    let record = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");

    let placement = place_read_only(&guard.0).expect("非 root では配置が成功するはず");
    assert_eq!(
        placement.mode(),
        0o444,
        "(d) mode が 0o444 と完全一致すること"
    );

    // (a) append 書き込みは拒否される。
    match std::fs::OpenOptions::new().append(true).open(&guard.0) {
        Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
        Ok(_) => panic!("append open は拒否されるはず"),
    }

    // (b) truncate 書き込みも拒否される。
    match std::fs::write(&guard.0, b"tampered") {
        Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
        Ok(()) => panic!("truncate write は拒否されるはず"),
    }

    // (c) 読み戻したバイト列は変わっておらず、`evaluate_gate` が `Proceed` になる。
    let actual_bytes = std::fs::read(&guard.0).expect("読み取りは成功するはず");
    assert_eq!(actual_bytes, SAMPLE);
    let state = EvalDataState::Frozen(record.clone());
    assert_eq!(
        evaluate_gate(&state, &actual_bytes),
        Ok(EvaluateGate::Proceed(record))
    );
}

/// REQ-39・TASK-17.2-2: root 実行下では `place_read_only` が
/// `WriteNotRejected` で fail-closed に失敗する（配置済みを装わない）。
#[cfg(unix)]
#[test]
fn req39_place_read_only_fails_closed_when_root_via_integration() {
    let guard = write_unique_temp_file("root-acceptance", SAMPLE);
    if !running_as_root(&guard.0) {
        return;
    }

    match place_read_only(&guard.0) {
        Err(PlacementError::WriteNotRejected { .. }) => {}
        other => panic!("root では WriteNotRejected を期待したが {other:?} だった"),
    }
}

/// REQ-39: `verify_direct_write_rejected` を直接呼んでも、書き込み可能な
/// ファイルに対しては非 root で `WriteNotRejected` を返す（配置を経由しない
/// 単独呼び出しでも決定的に判定される）。
#[cfg(unix)]
#[test]
fn req39_verify_direct_write_rejected_standalone_on_writable_file() {
    let guard = write_unique_temp_file("standalone-probe", SAMPLE);
    if running_as_root(&guard.0) {
        return;
    }

    match verify_direct_write_rejected(&guard.0) {
        Err(PlacementError::WriteNotRejected { .. }) => {}
        other => panic!("WriteNotRejected を期待したが {other:?} だった"),
    }
    // 非破壊プローブ: 内容は変わらない。
    assert_eq!(
        std::fs::read(&guard.0).expect("読み取りに失敗しないはず"),
        SAMPLE
    );
}
