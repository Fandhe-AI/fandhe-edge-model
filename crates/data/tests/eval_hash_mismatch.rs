//! 評価データのハッシュ不一致で評価を停止する分岐の検知テスト
//! （REQ-17 異常系・TASK-17.3・issue #49。証拠の種別: テストハーネス）。
//!
//! 凍結記録 JSON をファイルへ保存して読み戻し、評価データファイルを改変して
//! 読み直し、[`evaluate_gate`] が評価を進めず不一致を報告して非ゼロの終了コードで
//! 停止することを往復シナリオで固定する
//! （`.claude/rules/evaluation-contract.md`「データの分割と凍結」の fail-closed）。
//! ゴールデン sha256 は `printf '%s' '<bytes>' | sha256sum` で独立に計算した値。
//! 評価工程は [`simulated_evaluate_step`] で模擬する（実際の評価器は
//! `fandhe-edge-eval`。CLI の `evaluate` 工程への接続は #314 で済み、本テストは模擬のまま）。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_data::eval_freeze::{
    EvalDataState, EvaluateGate, FreezeError, FreezeRecord, evaluate_gate, freeze_eval_data,
};
use std::path::{Path, PathBuf};

const SAMPLE: &[u8] =
    b"{\"id\":\"e1\",\"input\":\"hello\",\"output\":{\"intent\":\"greet\"}}\n{\"id\":\"e2\",\"input\":\"bye\",\"output\":{\"intent\":\"farewell\"}}\n";
const SAMPLE_SHA256: &str = "edb189cef6cd5da4187a462786a777e93d58920566931e32d5dfb5d77ed39f62";
const SAMPLE_ONE_BYTE_CHANGED_SHA256: &str =
    "306b04c17f66a94946eccceb1b4fdd0ceb1ec7dbcf2dc021be8adab8b56aa53a";
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// 一時ファイルを成否に関わらず削除するガード（RAII）。
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn write_unique_temp_file(label: &str, bytes: &[u8]) -> TempFile {
    let pid = std::process::id();
    for attempt in 0..1000u32 {
        let path =
            std::env::temp_dir().join(format!("fandhe-edge-hash-mismatch-{pid}-{label}-{attempt}"));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                use std::io::Write as _;
                file.write_all(bytes).expect("書き込みは成功するはず");
                return TempFile(path);
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => panic!("一時ファイルの作成に失敗しないはず: {err}"),
        }
    }
    panic!("一意な一時ファイルを作成できなかった");
}

/// 呼び出し側（CLI `evaluate` 工程）の分岐の形を模擬する。記録 JSON と評価
/// データを読み直し、ゲートが `Proceed` のときだけ評価（`evaluated = true`）へ
/// 進む。`Err` のときは評価せず、そのまま返す。
fn simulated_evaluate_step(
    record_path: &Path,
    data_path: &Path,
    evaluated: &mut bool,
) -> Result<ExitCode, FreezeError> {
    let record = FreezeRecord::load(record_path).expect("記録 JSON は読み込めるはず");
    let bytes = std::fs::read(data_path).expect("評価データは読めるはず");
    match evaluate_gate(&EvalDataState::Frozen(record), &bytes)? {
        EvaluateGate::Proceed(_) => {
            *evaluated = true;
            Ok(ExitCode::Ok)
        }
        EvaluateGate::Skip => Ok(ExitCode::Ok),
    }
}

/// 凍結して記録 JSON と評価データをそれぞれ一時ファイルへ保存する。
fn freeze_and_persist(label: &str) -> (TempFile, TempFile) {
    let record = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");
    let json = serde_json::to_string(&record).expect("serialize は成功するはず");
    (
        write_unique_temp_file(&format!("{label}-record"), json.as_bytes()),
        write_unique_temp_file(&format!("{label}-data"), SAMPLE),
    )
}

/// 不一致で停止し、終了コードが非ゼロで固定であることを確認する共通部。
fn expect_stop(record: &TempFile, data: &TempFile) -> FreezeError {
    let mut evaluated = false;
    let err = simulated_evaluate_step(&record.0, &data.0, &mut evaluated)
        .expect_err("不一致なら停止するはず");
    assert!(!evaluated, "停止時に評価が実行されてはならない");
    assert_eq!(err.exit_code(), ExitCode::InvalidInput);
    assert_ne!(err.exit_code(), ExitCode::Ok);
    err
}

/// 対照群: 未改変なら評価へ進む。
#[test]
fn req17_task17_3_unchanged_data_proceeds() {
    let (record, data) = freeze_and_persist("unchanged");
    let mut evaluated = false;
    let code = simulated_evaluate_step(&record.0, &data.0, &mut evaluated)
        .expect("未改変なら通過するはず");
    assert_eq!(code, ExitCode::Ok);
    assert!(evaluated);
}

/// 同じ長さのまま 1 バイト書き換えると停止し、両側のハッシュを報告する。
#[test]
fn req17_task17_3_single_byte_change_stops_and_reports() {
    let (record, data) = freeze_and_persist("onebyte");
    let mut changed = SAMPLE.to_vec();
    // `farewell` の最終文字 `l` を `L` に変える（末尾から数えて `l"}}\n` の位置）。
    let pos = changed.len() - 5;
    *changed.get_mut(pos).expect("範囲内") = b'L';
    std::fs::write(&data.0, &changed).expect("改変は成功するはず");

    match expect_stop(&record, &data) {
        FreezeError::HashMismatch {
            expected_sha256,
            expected_byte_len,
            actual_sha256,
            actual_byte_len,
        } => {
            assert_eq!(expected_sha256.to_hex(), SAMPLE_SHA256);
            assert_eq!(actual_sha256.to_hex(), SAMPLE_ONE_BYTE_CHANGED_SHA256);
            assert_eq!(expected_byte_len, 113);
            assert_eq!(actual_byte_len, 113);
        }
        other => panic!("HashMismatch を期待したが {other:?} だった"),
    }
}

/// 末尾への追記で停止する。
#[test]
fn req17_task17_3_appended_data_stops() {
    let (record, data) = freeze_and_persist("append");
    let mut bytes = SAMPLE.to_vec();
    bytes.extend_from_slice(b"{\"id\":\"e3\"}\n");
    std::fs::write(&data.0, &bytes).expect("改変は成功するはず");

    match expect_stop(&record, &data) {
        FreezeError::HashMismatch {
            expected_byte_len,
            actual_byte_len,
            ..
        } => {
            assert_eq!(expected_byte_len, 113);
            assert_eq!(actual_byte_len, 125);
        }
        other => panic!("HashMismatch を期待したが {other:?} だった"),
    }
}

/// 末尾の切り詰めで停止する。
#[test]
fn req17_task17_3_truncated_data_stops() {
    let (record, data) = freeze_and_persist("truncate");
    let cut = SAMPLE.get(..SAMPLE.len() - 10).expect("範囲内");
    std::fs::write(&data.0, cut).expect("改変は成功するはず");

    match expect_stop(&record, &data) {
        FreezeError::HashMismatch {
            expected_byte_len,
            actual_byte_len,
            ..
        } => {
            assert_eq!(expected_byte_len, 113);
            assert_eq!(actual_byte_len, 103);
        }
        other => panic!("HashMismatch を期待したが {other:?} だった"),
    }
}

/// 空にされたデータは「評価データなし」（skipped）へすり替わらず停止する。
#[test]
fn req17_task17_3_emptied_data_stops_not_skipped() {
    let (record, data) = freeze_and_persist("emptied");
    std::fs::write(&data.0, b"").expect("改変は成功するはず");

    match expect_stop(&record, &data) {
        FreezeError::HashMismatch {
            actual_sha256,
            actual_byte_len,
            ..
        } => {
            assert_eq!(actual_byte_len, 0);
            assert_eq!(actual_sha256.to_hex(), EMPTY_SHA256);
        }
        other => panic!("HashMismatch を期待したが {other:?} だった"),
    }
}

/// データは不変でも、保存済み記録 JSON 側が改変されていれば停止する。
#[test]
fn req17_task17_3_tampered_record_json_stops() {
    let (record, data) = freeze_and_persist("tamperedrec");
    let json = std::fs::read_to_string(&record.0).expect("読めるはず");
    let forged = json.replace(SAMPLE_SHA256, SAMPLE_ONE_BYTE_CHANGED_SHA256);
    assert_ne!(json, forged);
    std::fs::write(&record.0, forged).expect("改変は成功するはず");

    match expect_stop(&record, &data) {
        FreezeError::HashMismatch {
            expected_sha256,
            actual_sha256,
            ..
        } => {
            assert_eq!(expected_sha256.to_hex(), SAMPLE_ONE_BYTE_CHANGED_SHA256);
            assert_eq!(actual_sha256.to_hex(), SAMPLE_SHA256);
        }
        other => panic!("HashMismatch を期待したが {other:?} だった"),
    }
}
