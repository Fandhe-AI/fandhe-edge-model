//! 評価データの評価前後ハッシュ比較・凍結記録との接続の結合テスト
//! （REQ-27「評価の独立性」・REQ-17「データの分割と凍結」・TASK-27.1-2・issue #70）。
//!
//! PoC-9 `InvarianceTest` の評価データ側手順（凍結 → 実評価経路 → 評価前後の
//! ディスク再読み込み比較）を移植する（`docs/spec` は読まず、手順の再現のみを
//! 行う。`.claude/rules/spec-reference.md` のビルド独立方針）。
//!
//! 証拠の種別: テストハーネス（合成データによる結合テスト）。実機ではない。
//!
//! 本ファイルは `fandhe-edge-data`（`[dev-dependencies]` としてのみ依存。
//! `crates/eval/Cargo.toml`・`crates/eval/src/lib.rs`「層の境界」参照）の
//! `eval_freeze::{freeze_eval_data, evaluate_gate, EvalDataState, FreezeError}` を
//! 使い、本 crate の評価データ不変性チェックが data 層の凍結記録・停止判定
//! （TASK-17.3）と同じ条件で停止することを、両者を並べて assert することで
//! 確認する（「停止分岐への接続確認」の証拠）。

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_data::eval_freeze::{EvalDataState, FreezeError, evaluate_gate, freeze_eval_data};
use fandhe_edge_eval::eval_data_invariance::{
    EvalDataInvarianceError, FrozenEvalData, evaluate_with_eval_data_invariance,
};
use fandhe_edge_eval::invariance::{
    EvaluationInvarianceError, ModelPackageBytes, ModelPackagePaths, evaluate_with_invariance,
};
use fandhe_edge_eval::metrics::{EvalRecord, Outcome, evaluate_single_select};
use std::fmt;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

/// テスト用の一時ファイルを、成否に関わらず削除するガード（RAII）。
struct TempFileGuard(PathBuf);

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// 排他的に一意な一時ファイルパスを作り、`bytes` を書き込む
/// （`crates/core/src/fs.rs` のテストヘルパーと同じ方針。外部 crate〔tempfile
/// 等〕は追加しない）。
fn write_unique_temp_file(label: &str, bytes: &[u8]) -> TempFileGuard {
    let pid = std::process::id();
    for attempt in 0..1000u32 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let candidate = std::env::temp_dir().join(format!(
            "fandhe-edge-eval-eval-data-invariance-test-{pid}-{label}-{attempt}-{nanos}"
        ));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                file.write_all(bytes)
                    .expect("一時ファイルへの書き込みに失敗しないはず");
                return TempFileGuard(candidate);
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => panic!("一時ファイルの作成に失敗しないはず: {err}"),
        }
    }
    panic!("一意な一時ファイルを作成できなかった");
}

/// 合成評価データ本体: 1 行 1 gold ラベルの 6 行（A×4・B×2）。
/// パイプ区切りの単純な行フォーマットで、実際のデータ形式（TASK-16.x）を
/// 先取りしない（本テストは評価データ「本体のハッシュ」だけを扱う）。
const SYNTHETIC_EVAL_DATA: &[u8] = b"A\nA\nA\nA\nB\nB\n";

/// 評価データ本体から gold ラベルを復元し、majority スタブの予測と
/// `evaluate_single_select` まで通す（PoC-9 相当の「実評価経路」）。
///
/// 実運用の評価器と同じく、`bytes`（凍結・ハッシュ照合済みの評価データ本体）
/// から直接 gold を読み取る点が、固定値を返すだけのスタブと異なる
/// （`invariance.rs` の `predict_using_package` と同じ考え方を評価データ側に
/// 適用したもの）。
fn run_real_evaluation_path(bytes: &[u8]) -> Result<(u64, u64), EvalStepError> {
    let text = std::str::from_utf8(bytes).map_err(|err| EvalStepError(err.to_string()))?;
    let golds: Vec<&str> = text.lines().collect();
    if golds.is_empty() {
        return Err(EvalStepError("empty eval data".to_string()));
    }
    // majority スタブ: 常に "A" と予測する。
    let outcomes: Vec<Outcome> = vec![Outcome::Label("A".to_string()); golds.len()];
    let records: Vec<EvalRecord<'_>> = golds
        .into_iter()
        .zip(outcomes.iter())
        .map(|(gold, outcome)| EvalRecord { gold, outcome })
        .collect();
    let labels = ["A", "B"];
    let metrics =
        evaluate_single_select(&labels, &records).map_err(|err| EvalStepError(err.to_string()))?;
    let overall = metrics.accuracy.overall;
    Ok((overall.numerator(), overall.denominator()))
}

/// テストの評価クロージャのエラー型。
#[derive(Debug)]
struct EvalStepError(String);

impl fmt::Display for EvalStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for EvalStepError {}

#[test]
fn req27_eval_data_hash_unchanged_across_real_evaluation_path() {
    // PoC-9 相当の正常系（受入基準）: 合成評価データを凍結 → data 層の
    // evaluate_gate が Proceed → 本 API で実評価経路を通す → 正解率が
    // 具体値（4/6。gold は A×4・B×2、majority は常に "A"）で一致する。
    let record = freeze_eval_data(SYNTHETIC_EVAL_DATA).expect("凍結は失敗しないはず");
    let state = EvalDataState::Frozen(record.clone());
    let gate = evaluate_gate(&state, SYNTHETIC_EVAL_DATA).expect("Proceed のはず");
    let fandhe_edge_data::eval_freeze::EvaluateGate::Proceed(gated_record) = gate else {
        panic!("Proceed を期待した");
    };
    assert_eq!(gated_record, record);

    let guard = write_unique_temp_file("happy-path", SYNTHETIC_EVAL_DATA);
    let frozen = FrozenEvalData {
        path: &guard.0,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };

    // 評価前後ダイジェストの hex を独立計算したゴールデン値と比較する。
    let expected_hex = Sha256Digest::of_bytes(SYNTHETIC_EVAL_DATA).to_hex();
    assert_eq!(record.sha256().to_hex(), expected_hex);

    let (correct, total) = evaluate_with_eval_data_invariance(&frozen, run_real_evaluation_path)
        .expect("成功するはず");
    assert_eq!((correct, total), (4, 6));
}

#[test]
fn req17_req27_stale_frozen_record_stops_before_evaluation() {
    // 凍結後・評価前にファイルを改変 → FrozenRecordMismatch で停止し、
    // クロージャが呼ばれていないことを確認する。同じ改変後バイト列に対し
    // data の evaluate_gate も HashMismatch を返すこと（停止分岐への接続確認）。
    let record = freeze_eval_data(SYNTHETIC_EVAL_DATA).expect("凍結は失敗しないはず");
    let guard = write_unique_temp_file("stale-record", SYNTHETIC_EVAL_DATA);
    let tampered: &[u8] = b"A\nA\nA\nA\nB\nB\nC\n";
    fs::write(&guard.0, tampered).expect("改変の書き込みに失敗しないはず");

    let frozen = FrozenEvalData {
        path: &guard.0,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };
    let mut called = false;
    let result = evaluate_with_eval_data_invariance(&frozen, |_bytes| {
        called = true;
        Ok::<(), EvalStepError>(())
    });
    assert!(!called, "凍結記録と不一致なら eval を呼ばないはず");
    match result {
        Err(EvalDataInvarianceError::FrozenRecordMismatch { .. }) => {}
        other => panic!("FrozenRecordMismatch を期待したが {other:?} だった"),
    }

    // data 層の evaluate_gate も同じ改変後バイト列に対して HashMismatch を返す。
    let state = EvalDataState::Frozen(record);
    let gate_result = evaluate_gate(&state, tampered);
    assert_eq!(gate_result, Err(FreezeError::HashMismatch));
}

#[test]
fn req17_req27_tampering_by_evaluation_step_is_detected() {
    // クロージャ内でファイルに追記（評価処理自身による改変）→ クロージャは
    // Ok を返すが API は ChangedDuringEvaluation（before/after の hex を
    // 具体値で確認）。読み直したバイト列で evaluate_gate も HashMismatch。
    let record = freeze_eval_data(SYNTHETIC_EVAL_DATA).expect("凍結は失敗しないはず");
    let guard = write_unique_temp_file("tamper-during-eval", SYNTHETIC_EVAL_DATA);
    let path = guard.0.clone();
    let frozen = FrozenEvalData {
        path: &path,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };

    let result = evaluate_with_eval_data_invariance(&frozen, |_bytes| {
        // 評価処理自身が評価データを書き換える回帰を模擬する。
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&guard.0)
            .expect("追記のために開けるはず");
        file.write_all(b"C\n")
            .expect("追記の書き込みに失敗しないはず");
        Ok::<(), EvalStepError>(())
    });

    let tampered_bytes = fs::read(&path).expect("改変後のファイルを読めるはず");
    let expected_after = Sha256Digest::of_bytes(&tampered_bytes);
    match result {
        Err(EvalDataInvarianceError::ChangedDuringEvaluation { before, after }) => {
            assert_eq!(before.to_hex(), record.sha256().to_hex());
            assert_eq!(after.to_hex(), expected_after.to_hex());
        }
        other => panic!("ChangedDuringEvaluation を期待したが {other:?} だった"),
    }

    let state = EvalDataState::Frozen(record);
    let gate_result = evaluate_gate(&state, &tampered_bytes);
    assert_eq!(gate_result, Err(FreezeError::HashMismatch));
}

#[test]
fn req27_tampering_takes_precedence_over_evaluation_error() {
    // クロージャが Err を返しつつ改変 → ChangedDuringEvaluation が優先される
    // （fail-closed。改変検出のほうが評価失敗より重大な不変条件違反のため）。
    let record = freeze_eval_data(SYNTHETIC_EVAL_DATA).expect("凍結は失敗しないはず");
    let guard = write_unique_temp_file("tamper-and-error", SYNTHETIC_EVAL_DATA);
    let path = guard.0.clone();
    let frozen = FrozenEvalData {
        path: &path,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };

    let result = evaluate_with_eval_data_invariance(&frozen, |_bytes| {
        fs::write(&guard.0, b"tampered").expect("改変の書き込みに失敗しないはず");
        Err::<(), EvalStepError>(EvalStepError("boom".to_string()))
    });

    match result {
        Err(EvalDataInvarianceError::ChangedDuringEvaluation { .. }) => {}
        other => panic!("ChangedDuringEvaluation を期待したが {other:?} だった"),
    }
}

#[test]
fn req27_evaluation_error_is_wrapped_when_unchanged() {
    // 改変なしでクロージャが Err → Evaluation(e) として包まれる。
    let record = freeze_eval_data(SYNTHETIC_EVAL_DATA).expect("凍結は失敗しないはず");
    let guard = write_unique_temp_file("unchanged-error", SYNTHETIC_EVAL_DATA);
    let frozen = FrozenEvalData {
        path: &guard.0,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };

    let result = evaluate_with_eval_data_invariance(&frozen, |_bytes| {
        Err::<(), EvalStepError>(EvalStepError("boom".to_string()))
    });

    match result {
        Err(EvalDataInvarianceError::Evaluation(EvalStepError(message))) => {
            assert_eq!(message, "boom");
        }
        other => panic!("Evaluation を期待したが {other:?} だった"),
    }
}

#[test]
fn req39_missing_path_maps_to_read() {
    let missing = std::env::temp_dir().join(format!(
        "fandhe-edge-eval-eval-data-invariance-test-missing-{}",
        std::process::id()
    ));
    let frozen = FrozenEvalData {
        path: &missing,
        sha256: Sha256Digest::of_bytes(b""),
        byte_len: 0,
    };
    let result = evaluate_with_eval_data_invariance(&frozen, |_bytes| Ok::<(), EvalStepError>(()));
    match result {
        Err(EvalDataInvarianceError::Read { .. }) => {}
        other => panic!("Read を期待したが {other:?} だった"),
    }
}

#[test]
fn req39_directory_maps_to_not_regular_file() {
    let dir = std::env::temp_dir().join(format!(
        "fandhe-edge-eval-eval-data-invariance-test-dir-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    fs::create_dir(&dir).expect("テスト用ディレクトリを作成できるはず");

    let frozen = FrozenEvalData {
        path: &dir,
        sha256: Sha256Digest::of_bytes(b""),
        byte_len: 0,
    };
    let result = evaluate_with_eval_data_invariance(&frozen, |_bytes| Ok::<(), EvalStepError>(()));
    fs::remove_dir(&dir).expect("テスト用ディレクトリを削除できるはず");

    match result {
        Err(EvalDataInvarianceError::NotRegularFile { .. }) => {}
        other => panic!("NotRegularFile を期待したが {other:?} だった"),
    }
}

#[test]
fn req27_display_does_not_leak_eval_data_content() {
    // 識別しやすいダミー文字列を含む評価データで、各エラーの to_string() に
    // 本文が出ず、hex は含まれることを確認する。
    let secret: &[u8] = b"SECRET-VALUE-should-not-leak";
    let record = freeze_eval_data(secret).expect("凍結は失敗しないはず");
    let guard = write_unique_temp_file("display-leak", secret);

    // FrozenRecordMismatch の Display 確認（期待値を偽装して不一致にする）。
    let frozen_mismatch = FrozenEvalData {
        path: &guard.0,
        sha256: Sha256Digest::of_bytes(b"different"),
        byte_len: 999,
    };
    let mismatch_result =
        evaluate_with_eval_data_invariance(&frozen_mismatch, |_bytes| Ok::<(), EvalStepError>(()));
    let mismatch_message = mismatch_result.unwrap_err().to_string();
    assert!(!mismatch_message.contains("SECRET-VALUE-should-not-leak"));
    assert!(mismatch_message.contains(&record.sha256().to_hex()));

    // ChangedDuringEvaluation の Display 確認。
    let path = guard.0.clone();
    let frozen_ok = FrozenEvalData {
        path: &path,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };
    let changed_result = evaluate_with_eval_data_invariance(&frozen_ok, |_bytes| {
        fs::write(&guard.0, b"SECRET-VALUE-changed-should-not-leak-either")
            .expect("書き込みに失敗しないはず");
        Ok::<(), EvalStepError>(())
    });
    let changed_message = changed_result.unwrap_err().to_string();
    assert!(!changed_message.contains("SECRET-VALUE"));
}

#[test]
fn req27_nested_model_and_eval_data_invariance_happy_path() {
    // evaluate_with_invariance（モデル側）と evaluate_with_eval_data_invariance
    // （評価データ側）を入れ子にした場合に、両方不変なら指標が返ることを確認する。
    let package = ModelPackageBytes {
        weights: Some(b"synthetic-weights-bytes-v1".as_slice()),
        vocab: None,
        calibration: None,
        thresholds: None,
    };
    let model_dir_label = "nested-model";
    let model_guard = write_unique_temp_file(model_dir_label, package.weights.expect("値あり"));
    let model_paths = ModelPackagePaths {
        weights: &model_guard.0,
        vocab: None,
        calibration: None,
        thresholds: None,
    };

    let record = freeze_eval_data(SYNTHETIC_EVAL_DATA).expect("凍結は失敗しないはず");
    let eval_guard = write_unique_temp_file("nested-eval-data", SYNTHETIC_EVAL_DATA);
    let eval_path = eval_guard.0.clone();

    type InnerResult = Result<(u64, u64), EvalDataInvarianceError<EvalStepError>>;
    type OuterResult =
        Result<InnerResult, EvaluationInvarianceError<EvalDataInvarianceError<EvalStepError>>>;

    let outcome: OuterResult = evaluate_with_invariance(&model_paths, |_paths| {
        let frozen = FrozenEvalData {
            path: &eval_path,
            sha256: record.sha256(),
            byte_len: record.byte_len(),
        };
        Ok(evaluate_with_eval_data_invariance(
            &frozen,
            run_real_evaluation_path,
        ))
    });

    let inner = outcome.expect("モデル側は不変のはず");
    let (correct, total) = inner.expect("評価データ側も不変のはず");
    assert_eq!((correct, total), (4, 6));
}

#[test]
fn req17_empty_eval_data_is_frozen_not_skipped() {
    // 空ファイル（byte_len: 0、空文字列の sha256）でも評価が進むこと
    // （#47 の「空は評価データなしではない」という境界と整合）。
    let empty: &[u8] = b"";
    let record = freeze_eval_data(empty).expect("空データも凍結できるはず");
    assert_eq!(record.byte_len(), 0);
    assert_eq!(
        record.sha256().to_hex(),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );

    let guard = write_unique_temp_file("empty-eval-data", empty);
    let frozen = FrozenEvalData {
        path: &guard.0,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };
    let mut called = false;
    let result = evaluate_with_eval_data_invariance(&frozen, |bytes| {
        called = true;
        assert!(bytes.is_empty());
        Ok::<(), EvalStepError>(())
    });
    assert!(called, "空データでも eval は呼ばれるはず（skip ではない）");
    assert!(result.is_ok());
}
