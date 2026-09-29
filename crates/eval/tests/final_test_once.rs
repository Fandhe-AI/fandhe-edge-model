//! 凍結した最終 test への 1 回限り適用の強制の結合テスト
//! （REQ-27 境界値・TASK-27.3・issue #72）。
//!
//! PoC-10 の事前登録（方式 c1/c2/c3 × seed0..2 を各 1 回）と PoC-25 のロック
//! ファイル方式の手順を再現する。証拠の種別: テストハーネス（合成データ）。

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_data::eval_freeze::freeze_eval_data;
use fandhe_edge_eval::eval_data_invariance::{
    EvalDataInvarianceError, FrozenEvalData, evaluate_with_eval_data_invariance,
};
use fandhe_edge_eval::final_test_once::{
    AcquireError, AppliedBy, ApplyOnceError, FinalTestKey, FinalTestLedger, RepresentativeConfigId,
    apply_once,
};
use fandhe_edge_eval::invariance::{ModelPackageBytes, ModelPackageSnapshot};
use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};

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

fn key(data: &[u8], id: &str, weights: &[u8]) -> FinalTestKey {
    FinalTestKey::new(
        Sha256Digest::of_bytes(data),
        RepresentativeConfigId::parse(id).unwrap(),
        &snapshot(weights, None),
    )
    .unwrap()
}

const DATA: &[u8] = b"A\nA\nB\n";

#[test]
fn req27_second_application_same_config_is_rejected() {
    let dir = TempDir::new("second");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let k = key(DATA, "c1:seed0", b"w1");
    let calls = Cell::new(0u32);
    let run = || {
        apply_once(&ledger, &k, |_t| {
            calls.set(calls.get() + 1);
            Ok::<_, String>(())
        })
    };
    assert!(run().is_ok());
    match run() {
        Err(ApplyOnceError::Acquire(AcquireError::AlreadyApplied { by, .. })) => {
            assert_eq!(by, AppliedBy::RepresentativeConfig)
        }
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(calls.get(), 1);
}

#[test]
fn req27_rejection_persists_across_ledger_instances() {
    let dir = TempDir::new("persist");
    let k = key(DATA, "c1:seed0", b"w1");
    FinalTestLedger::open(dir.path())
        .unwrap()
        .acquire(&k)
        .unwrap();
    let again = FinalTestLedger::open(dir.path()).unwrap().acquire(&k);
    assert!(matches!(again, Err(AcquireError::AlreadyApplied { .. })));
}

#[test]
fn req27_concurrent_acquire_exactly_one_wins() {
    let dir = TempDir::new("concurrent");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let k = key(DATA, "c1:seed0", b"w1");
    let results: Vec<bool> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..8)
            .map(|_| s.spawn(|| ledger.acquire(&k).is_ok()))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(results.iter().filter(|ok| **ok).count(), 1);
}

#[test]
fn req27_prediction_error_still_consumes_application() {
    let dir = TempDir::new("prederr");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let k = key(DATA, "c1:seed0", b"w1");
    let r = apply_once(&ledger, &k, |_t| Err::<(), _>("boom".to_string()));
    assert!(matches!(r, Err(ApplyOnceError::Prediction(ref m)) if m == "boom"));
    let r = apply_once(&ledger, &k, |_t| Ok::<_, String>(()));
    assert!(matches!(
        r,
        Err(ApplyOnceError::Acquire(AcquireError::AlreadyApplied { .. }))
    ));
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
    let k = key(DATA, "c1:seed0", b"w1");

    // 評価データは台帳とは別の一時ディレクトリに置く。
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
    let r = evaluate_with_eval_data_invariance(&frozen, |_bytes| {
        apply_once(&ledger, &k, |_t| Ok::<_, String>(())).map_err(|e| e.to_string())
    });
    assert!(matches!(
        r,
        Err(EvalDataInvarianceError::FrozenRecordMismatch { .. })
    ));
    assert_eq!(dir.count(), 0);

    fs::write(&path, DATA).unwrap();
    let r = evaluate_with_eval_data_invariance(&frozen, |_bytes| {
        apply_once(&ledger, &k, |_t| Ok::<_, String>(())).map_err(|e| e.to_string())
    });
    assert!(r.is_ok());
    assert_eq!(dir.count(), 2);
}

#[test]
fn req27_different_config_or_dataset_is_allowed() {
    let dir = TempDir::new("different");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    ledger.acquire(&key(DATA, "c1:seed0", b"w1")).unwrap();
    ledger.acquire(&key(DATA, "c2:seed0", b"w2")).unwrap();
    ledger.acquire(&key(b"other", "c1:seed0", b"w1")).unwrap();
}

#[test]
fn req27_same_weights_under_renamed_config_is_rejected() {
    let dir = TempDir::new("renamed");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    ledger.acquire(&key(DATA, "c1:seed0", b"w1")).unwrap();
    let renamed = ledger.acquire(&key(DATA, "c1:renamed", b"w1"));
    assert!(matches!(
        renamed,
        Err(AcquireError::AlreadyApplied {
            by: AppliedBy::ModelWeights,
            ..
        })
    ));
    // 重み同一でしきい値だけ変えても拒否される。
    let tuned = FinalTestKey::new(
        Sha256Digest::of_bytes(DATA),
        RepresentativeConfigId::parse("c1:tuned").unwrap(),
        &snapshot(b"w1", Some(b"t2")),
    )
    .unwrap();
    assert!(matches!(
        ledger.acquire(&tuned),
        Err(AcquireError::AlreadyApplied {
            by: AppliedBy::ModelWeights,
            ..
        })
    ));
    // 別 ID の代表構成ロックは残る（fail-closed）: 代表 1 + 重み 1 + 残り 2。
    assert_eq!(dir.count(), 4);
}

#[test]
fn req27_poc10_methods_by_seeds_each_once() {
    let dir = TempDir::new("poc10");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    for method in ["c1", "c2", "c3"] {
        for seed in 0..3 {
            let id = format!("{method}:seed{seed}");
            let w = format!("weights-{id}");
            ledger.acquire(&key(DATA, &id, w.as_bytes())).unwrap();
        }
    }
    assert_eq!(dir.count(), 18);
    let again = ledger.acquire(&key(DATA, "c3:seed1", b"weights-c3:seed1"));
    assert!(matches!(
        again,
        Err(AcquireError::AlreadyApplied {
            by: AppliedBy::RepresentativeConfig,
            ..
        })
    ));
    assert_eq!(dir.count(), 18);
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

#[test]
fn req27_record_contains_digests_and_id_only() {
    let dir = TempDir::new("record");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let k = key(DATA, "c1:seed0", b"w1");
    let ticket = ledger.acquire(&k).unwrap();
    let body = fs::read_to_string(ticket.config_lock_path()).unwrap();
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
        let mode = fs::metadata(ticket.config_lock_path())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
