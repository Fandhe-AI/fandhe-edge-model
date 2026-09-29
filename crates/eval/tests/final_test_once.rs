//! 凍結した最終 test への 1 回限り適用の強制の結合テスト
//! （REQ-27 境界値・TASK-27.3・issue #72）。
//!
//! PoC-10 の事前登録（方式 c1/c2/c3 × seed0..2 を各 1 回）と PoC-25 のロック
//! ファイル方式の手順を再現する。証拠の種別: テストハーネス（合成データ）。
//! 公開経路は `apply_once` のみ（ロックのキーは凍結記録と照合済みの値からしか
//! 作れない）ため、全テストがこの経路を通す。

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_data::eval_freeze::freeze_eval_data;
use fandhe_edge_eval::eval_data_invariance::{EvalDataInvarianceError, FrozenEvalData};
use fandhe_edge_eval::final_test_once::{
    AcquireError, AppliedBy, ApplyOnceError, FinalTestLedger, RepresentativeConfigId, apply_once,
};
use fandhe_edge_eval::invariance::{ModelPackageBytes, ModelPackageSnapshot};
use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

/// 一時ディレクトリを成否に関わらず削除するガード。
struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let pid = std::process::id();
        for attempt in 0..1000u32 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let p = std::env::temp_dir().join(format!(
                "fandhe-edge-eval-final-test-once-{pid}-{label}-{attempt}-{nanos}"
            ));
            if fs::create_dir(&p).is_ok() {
                return TempDir(p);
            }
        }
        panic!("failed to create temp dir");
    }
    fn path(&self) -> &Path {
        &self.0
    }
    fn count(&self) -> usize {
        fs::read_dir(&self.0).unwrap().count()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn snapshot(weights: &[u8], thresholds: Option<&[u8]>) -> ModelPackageSnapshot {
    ModelPackageSnapshot::capture(&ModelPackageBytes {
        weights: Some(weights),
        thresholds,
        ..Default::default()
    })
    .unwrap()
}

const DATA: &[u8] = b"A\nA\nB\n";

type Outcome = Result<(), EvalDataInvarianceError<ApplyOnceError<String>>>;

fn id(raw: &str) -> RepresentativeConfigId {
    RepresentativeConfigId::parse(raw).unwrap()
}

/// 評価データを一時ファイルに置き、その正しい凍結記録で `apply_once` を 1 回試みる。
fn run(
    ledger: &FinalTestLedger,
    data: &[u8],
    config: &str,
    weights: &[u8],
    thresholds: Option<&[u8]>,
    calls: &Cell<u32>,
) -> Outcome {
    let data_dir = TempDir::new("data");
    let path = data_dir.path().join("eval.bin");
    fs::write(&path, data).unwrap();
    let record = freeze_eval_data(data).unwrap();
    let frozen = FrozenEvalData {
        path: &path,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };
    apply_once(
        ledger,
        &frozen,
        id(config),
        &snapshot(weights, thresholds),
        |_t, _bytes| {
            calls.set(calls.get() + 1);
            Ok::<_, String>(())
        },
    )
}

fn is_already(r: &Outcome, want: AppliedBy) -> bool {
    matches!(
        r,
        Err(EvalDataInvarianceError::Evaluation(ApplyOnceError::Acquire(
            AcquireError::AlreadyApplied { by, .. }
        ))) if *by == want
    )
}

#[test]
fn req27_second_application_same_config_is_rejected() {
    let dir = TempDir::new("second");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let calls = Cell::new(0u32);
    assert!(run(&ledger, DATA, "c1:seed0", b"w1", None, &calls).is_ok());
    let r = run(&ledger, DATA, "c1:seed0", b"w1", None, &calls);
    assert!(is_already(&r, AppliedBy::RepresentativeConfig), "{r:?}");
    assert_eq!(calls.get(), 1);
}

#[test]
fn req27_rejection_persists_across_ledger_instances() {
    let dir = TempDir::new("persist");
    let calls = Cell::new(0u32);
    let l1 = FinalTestLedger::open(dir.path()).unwrap();
    assert!(run(&l1, DATA, "c1:seed0", b"w1", None, &calls).is_ok());
    let l2 = FinalTestLedger::open(dir.path()).unwrap();
    let r = run(&l2, DATA, "c1:seed0", b"w1", None, &calls);
    assert!(is_already(&r, AppliedBy::RepresentativeConfig));
    assert_eq!(calls.get(), 1);
}

#[test]
fn req27_concurrent_apply_exactly_one_wins() {
    let dir = TempDir::new("concurrent");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let data_dir = TempDir::new("concurrent-data");
    let path = data_dir.path().join("eval.bin");
    fs::write(&path, DATA).unwrap();
    let record = freeze_eval_data(DATA).unwrap();
    let hits = AtomicU32::new(0);
    let oks: Vec<bool> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..8)
            .map(|_| {
                s.spawn(|| {
                    let frozen = FrozenEvalData {
                        path: &path,
                        sha256: record.sha256(),
                        byte_len: record.byte_len(),
                    };
                    apply_once(
                        &ledger,
                        &frozen,
                        id("c1:seed0"),
                        &snapshot(b"w1", None),
                        |_t, _b| {
                            hits.fetch_add(1, Ordering::SeqCst);
                            Ok::<_, String>(())
                        },
                    )
                    .is_ok()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(oks.iter().filter(|ok| **ok).count(), 1);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[test]
fn req27_prediction_error_still_consumes_application() {
    let dir = TempDir::new("prederr");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let data_dir = TempDir::new("prederr-data");
    let path = data_dir.path().join("eval.bin");
    fs::write(&path, DATA).unwrap();
    let record = freeze_eval_data(DATA).unwrap();
    let frozen = FrozenEvalData {
        path: &path,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };
    let r = apply_once(
        &ledger,
        &frozen,
        id("c1:seed0"),
        &snapshot(b"w1", None),
        |_t, _b| Err::<(), _>("boom".to_string()),
    );
    assert!(matches!(
        r,
        Err(EvalDataInvarianceError::Evaluation(
            ApplyOnceError::Prediction(ref m)
        )) if m == "boom"
    ));
    let calls = Cell::new(0u32);
    let r = run(&ledger, DATA, "c1:seed0", b"w1", None, &calls);
    assert!(is_already(&r, AppliedBy::RepresentativeConfig));
    assert_eq!(calls.get(), 0);
}

#[test]
fn req27_invalid_config_id_creates_no_lock() {
    let dir = TempDir::new("badid");
    for bad in ["", "../x", "a/b", "a b", "日本"] {
        assert!(matches!(
            RepresentativeConfigId::parse(bad),
            Err(AcquireError::InvalidConfigId { .. })
        ));
    }
    assert!(RepresentativeConfigId::parse(&"x".repeat(129)).is_err());
    assert!(RepresentativeConfigId::parse(&"x".repeat(128)).is_ok());
    assert_eq!(dir.count(), 0);
}

#[test]
fn req27_frozen_hash_mismatch_does_not_consume_application() {
    let dir = TempDir::new("frozen");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let record = freeze_eval_data(DATA).unwrap();

    let data_dir = TempDir::new("frozen-data");
    let path = data_dir.path().join("eval.bin");
    let mut tampered = DATA.to_vec();
    tampered[0] = b'B';
    fs::write(&path, &tampered).unwrap();
    let frozen = FrozenEvalData {
        path: &path,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };
    let hits = Cell::new(0u32);
    let attempt = || {
        apply_once(
            &ledger,
            &frozen,
            id("c1:seed0"),
            &snapshot(b"w1", None),
            |_t, _b| {
                hits.set(hits.get() + 1);
                Ok::<_, String>(())
            },
        )
    };
    assert!(matches!(
        attempt(),
        Err(EvalDataInvarianceError::FrozenRecordMismatch { .. })
    ));
    assert_eq!(dir.count(), 0);
    assert_eq!(hits.get(), 0);

    fs::write(&path, DATA).unwrap();
    assert!(attempt().is_ok());
    assert_eq!(dir.count(), 2);
    assert_eq!(hits.get(), 1);
}

/// P0 回帰: 実データと一致しない別ダイジェストを主張してもキーは作れず、
/// 別ロックを取得して 2 回目の予測を通すことはできない。
#[test]
fn req27_cannot_bypass_lock_with_different_digest() {
    let dir = TempDir::new("bypass");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let calls = Cell::new(0u32);
    assert!(run(&ledger, DATA, "c1:seed0", b"w1", None, &calls).is_ok());

    let data_dir = TempDir::new("bypass-data");
    let path = data_dir.path().join("eval.bin");
    fs::write(&path, DATA).unwrap();
    let other = freeze_eval_data(b"other").unwrap();
    let frozen = FrozenEvalData {
        path: &path,
        sha256: other.sha256(),
        byte_len: other.byte_len(),
    };
    let hits = Cell::new(0u32);
    let r = apply_once(
        &ledger,
        &frozen,
        id("c1:seed0"),
        &snapshot(b"w1", None),
        |_t, _b| {
            hits.set(hits.get() + 1);
            Ok::<_, String>(())
        },
    );
    assert!(matches!(
        r,
        Err(EvalDataInvarianceError::FrozenRecordMismatch { .. })
    ));
    assert_eq!(hits.get(), 0);
    assert_eq!(dir.count(), 2);
}

#[test]
fn req27_different_config_or_dataset_is_allowed() {
    let dir = TempDir::new("different");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let calls = Cell::new(0u32);
    assert!(run(&ledger, DATA, "c1:seed0", b"w1", None, &calls).is_ok());
    assert!(run(&ledger, DATA, "c2:seed0", b"w2", None, &calls).is_ok());
    assert!(run(&ledger, b"other", "c1:seed0", b"w1", None, &calls).is_ok());
    assert_eq!(calls.get(), 3);
}

#[test]
fn req27_same_weights_under_renamed_config_is_rejected() {
    let dir = TempDir::new("renamed");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let calls = Cell::new(0u32);
    assert!(run(&ledger, DATA, "c1:seed0", b"w1", None, &calls).is_ok());
    let renamed = run(&ledger, DATA, "c1:renamed", b"w1", None, &calls);
    assert!(is_already(&renamed, AppliedBy::ModelWeights));
    // 重み同一でしきい値だけ変えても拒否される。
    let tuned = run(&ledger, DATA, "c1:tuned", b"w1", Some(b"t2"), &calls);
    assert!(is_already(&tuned, AppliedBy::ModelWeights));
    // 別 ID の代表構成ロックは残る（fail-closed）: 代表 1 + 重み 1 + 残り 2。
    assert_eq!(dir.count(), 4);
    assert_eq!(calls.get(), 1);
}

#[test]
fn req27_poc10_methods_by_seeds_each_once() {
    let dir = TempDir::new("poc10");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let calls = Cell::new(0u32);
    for method in ["c1", "c2", "c3"] {
        for seed in 0..3 {
            let cid = format!("{method}:seed{seed}");
            let w = format!("weights-{cid}");
            assert!(run(&ledger, DATA, &cid, w.as_bytes(), None, &calls).is_ok());
        }
    }
    assert_eq!(dir.count(), 18);
    let again = run(&ledger, DATA, "c3:seed1", b"weights-c3:seed1", None, &calls);
    assert!(is_already(&again, AppliedBy::RepresentativeConfig));
    assert_eq!(dir.count(), 18);
    assert_eq!(calls.get(), 9);
}

#[test]
fn req27_ledger_dir_must_be_real_directory() {
    let dir = TempDir::new("ledgerdir");
    let missing = dir.path().join("missing");
    assert!(matches!(
        FinalTestLedger::open(&missing),
        Err(AcquireError::LedgerDirInvalid { .. })
    ));
    let file = dir.path().join("file");
    fs::write(&file, b"x").unwrap();
    assert!(matches!(
        FinalTestLedger::open(&file),
        Err(AcquireError::LedgerDirInvalid { .. })
    ));
    #[cfg(unix)]
    {
        let real = dir.path().join("real");
        fs::create_dir(&real).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(matches!(
            FinalTestLedger::open(&link),
            Err(AcquireError::LedgerDirInvalid { .. })
        ));
    }
}

/// 台帳ディレクトリが失われた場合、ロックを取得できず予測を呼ばない（fail-closed）。
/// 永続化（fsync）失敗そのものは環境依存で再現できないため、実装側の`?` 伝播で担保する
/// （Unix 限定。非 Unix は `DurabilityUnsupported` で常に拒否）。
#[cfg(unix)]
#[test]
fn req27_ledger_dir_removed_does_not_call_prediction() {
    let dir = TempDir::new("durability");
    let ledger_dir = dir.path().join("ledger");
    fs::create_dir(&ledger_dir).unwrap();
    let ledger = FinalTestLedger::open(&ledger_dir).unwrap();
    fs::remove_dir(&ledger_dir).unwrap();
    let calls = Cell::new(0u32);
    let r = run(&ledger, DATA, "c1:seed0", b"w1", None, &calls);
    assert!(matches!(
        r,
        Err(EvalDataInvarianceError::Evaluation(
            ApplyOnceError::Acquire(AcquireError::Io { .. })
        ))
    ));
    assert_eq!(calls.get(), 0);
}

#[test]
fn req27_record_contains_digests_and_id_only() {
    let dir = TempDir::new("record");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let data_dir = TempDir::new("record-data");
    let path = data_dir.path().join("eval.bin");
    fs::write(&path, DATA).unwrap();
    let record = freeze_eval_data(DATA).unwrap();
    let frozen = FrozenEvalData {
        path: &path,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };
    let lock_path = apply_once(
        &ledger,
        &frozen,
        id("c1:seed0"),
        &snapshot(b"w1", None),
        |t, _b| Ok::<_, String>(t.config_lock_path().to_path_buf()),
    )
    .unwrap();
    let body = fs::read_to_string(&lock_path).unwrap();
    assert!(body.starts_with("fandhe-edge-final-test-application v1\nlock=config\n"));
    assert!(body.contains(&format!(
        "eval_data_sha256={}\n",
        Sha256Digest::of_bytes(DATA).to_hex()
    )));
    assert!(body.contains("representative_config_id=c1:seed0\n"));
    assert!(!body.contains("A\nA\nB"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = fs::metadata(&lock_path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
