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
    AcquireError, AppliedBy, AppliedOnce, ApplyOnceError, DecodeFailed, FinalTestLedger,
    LabeledInput, RegisteredConfig, RepresentativeConfigId, apply_once,
};
use fandhe_edge_eval::invariance::{EvaluationInvarianceError, ModelPackagePaths};
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

/// 重み（と任意のしきい値）をファイルに置いたモデルパッケージ。`apply_once` は
/// パスから重みを自らハッシュするため、テストも実ファイルを使う。
struct Model {
    _dir: TempDir,
    weights: PathBuf,
    thresholds: Option<PathBuf>,
}

impl Model {
    fn new(weights: &[u8], thresholds: Option<&[u8]>) -> Self {
        let dir = TempDir::new("model");
        let w = dir.path().join("weights.bin");
        fs::write(&w, weights).unwrap();
        let t = thresholds.map(|bytes| {
            let t = dir.path().join("thresholds.json");
            fs::write(&t, bytes).unwrap();
            t
        });
        Model {
            _dir: dir,
            weights: w,
            thresholds: t,
        }
    }

    fn paths(&self) -> ModelPackagePaths<'_> {
        ModelPackagePaths {
            weights: &self.weights,
            vocab: None,
            calibration: None,
            thresholds: self.thresholds.as_deref(),
        }
    }
}

const DATA: &[u8] = b"A\nA\nB\n";

type Outcome = Result<
    AppliedOnce<()>,
    EvalDataInvarianceError<EvaluationInvarianceError<ApplyOnceError<String>>>,
>;

/// 評価データを 1 行 1 件として分解する: `input` は行そのもの、正解は `gold-<行>`。
fn dec(bytes: &[u8]) -> Result<Vec<LabeledInput>, DecodeFailed> {
    let text = std::str::from_utf8(bytes).map_err(|_| DecodeFailed)?;
    Ok(text
        .lines()
        .map(|l| LabeledInput {
            input: l.to_string(),
            gold: format!("gold-{l}"),
        })
        .collect())
}

fn id(raw: &str) -> RepresentativeConfigId {
    RepresentativeConfigId::parse(raw).unwrap()
}

fn reg(id_str: &str, weights: &[u8]) -> RegisteredConfig {
    RegisteredConfig::new(id(id_str), Sha256Digest::of_bytes(weights))
}

/// 評価データの事前登録（評価前の凍結）。ID ごとに当てる重みの内容も結び付ける。
fn register(ledger: &FinalTestLedger, data: &[u8], entries: &[(&str, &[u8])]) {
    let entries: Vec<_> = entries.iter().map(|(i, w)| reg(i, w)).collect();
    ledger
        .register_configs(&freeze_eval_data(data).unwrap().sha256(), &entries)
        .unwrap();
}

/// 評価データを一時ファイルに置き、その正しい凍結記録で `apply_once` を 1 回試みる。
/// 代表構成 ID は呼び出し前に `register` で事前登録しておくこと。
fn run(
    ledger: &FinalTestLedger,
    data: &[u8],
    config: &str,
    model: &Model,
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
        &model.paths(),
        dec,
        |_t, _inputs, _paths| {
            calls.set(calls.get() + 1);
            Ok::<_, String>(())
        },
    )
}

fn acquire_err(r: &Outcome) -> Option<&AcquireError> {
    match r {
        Err(EvalDataInvarianceError::Evaluation(EvaluationInvarianceError::Evaluation(
            ApplyOnceError::Acquire(e),
        ))) => Some(e),
        _ => None,
    }
}

fn is_already(r: &Outcome, want: AppliedBy) -> bool {
    matches!(acquire_err(r), Some(AcquireError::AlreadyApplied { by, .. }) if *by == want)
}

#[test]
fn req27_second_application_same_config_is_rejected() {
    let dir = TempDir::new("second");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let m = Model::new(b"w1", None);
    let calls = Cell::new(0u32);
    assert!(run(&ledger, DATA, "c1:seed0", &m, &calls).is_ok());
    let r = run(&ledger, DATA, "c1:seed0", &m, &calls);
    assert!(is_already(&r, AppliedBy::RepresentativeConfig), "{r:?}");
    assert_eq!(calls.get(), 1);
}

#[test]
fn req27_rejection_persists_across_ledger_instances() {
    let dir = TempDir::new("persist");
    let calls = Cell::new(0u32);
    let m = Model::new(b"w1", None);
    let l1 = FinalTestLedger::open(dir.path()).unwrap();
    register(&l1, DATA, &[("c1:seed0", b"w1")]);
    assert!(run(&l1, DATA, "c1:seed0", &m, &calls).is_ok());
    let l2 = FinalTestLedger::open(dir.path()).unwrap();
    let r = run(&l2, DATA, "c1:seed0", &m, &calls);
    assert!(is_already(&r, AppliedBy::RepresentativeConfig));
    assert_eq!(calls.get(), 1);
}

#[test]
fn req27_concurrent_apply_exactly_one_wins() {
    let dir = TempDir::new("concurrent");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let data_dir = TempDir::new("concurrent-data");
    let path = data_dir.path().join("eval.bin");
    fs::write(&path, DATA).unwrap();
    let record = freeze_eval_data(DATA).unwrap();
    let m = Model::new(b"w1", None);
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
                        &m.paths(),
                        dec,
                        |_t, _b, _p| {
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
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let data_dir = TempDir::new("prederr-data");
    let path = data_dir.path().join("eval.bin");
    fs::write(&path, DATA).unwrap();
    let record = freeze_eval_data(DATA).unwrap();
    let frozen = FrozenEvalData {
        path: &path,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };
    let m = Model::new(b"w1", None);
    let r = apply_once(
        &ledger,
        &frozen,
        id("c1:seed0"),
        &m.paths(),
        dec,
        |_t, _b, _p| Err::<(), _>("boom".to_string()),
    );
    assert!(matches!(
        r,
        Err(EvalDataInvarianceError::Evaluation(
            EvaluationInvarianceError::Evaluation(ApplyOnceError::Prediction(ref msg))
        )) if msg == "boom"
    ));
    let calls = Cell::new(0u32);
    let r = run(&ledger, DATA, "c1:seed0", &m, &calls);
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
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
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
    let m = Model::new(b"w1", None);
    let hits = Cell::new(0u32);
    let attempt = || {
        apply_once(
            &ledger,
            &frozen,
            id("c1:seed0"),
            &m.paths(),
            dec,
            |_t, _b, _p| {
                hits.set(hits.get() + 1);
                Ok::<_, String>(())
            },
        )
    };
    assert!(matches!(
        attempt(),
        Err(EvalDataInvarianceError::FrozenRecordMismatch { .. })
    ));
    // 事前登録ファイル 1 つだけ。ロックは作られていない。
    assert_eq!(dir.count(), 1);
    assert_eq!(hits.get(), 0);

    fs::write(&path, DATA).unwrap();
    assert!(attempt().is_ok());
    assert_eq!(dir.count(), 3);
    assert_eq!(hits.get(), 1);
}

/// P0 回帰: 実データと一致しない別ダイジェストを主張してもキーは作れず、
/// 別ロックを取得して 2 回目の予測を通すことはできない。
#[test]
fn req27_cannot_bypass_lock_with_different_digest() {
    let dir = TempDir::new("bypass");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let m = Model::new(b"w1", None);
    let calls = Cell::new(0u32);
    assert!(run(&ledger, DATA, "c1:seed0", &m, &calls).is_ok());

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
        &m.paths(),
        dec,
        |_t, _b, _p| {
            hits.set(hits.get() + 1);
            Ok::<_, String>(())
        },
    );
    assert!(matches!(
        r,
        Err(EvalDataInvarianceError::FrozenRecordMismatch { .. })
    ));
    assert_eq!(hits.get(), 0);
    // 登録 1 + 適用 2（代表構成・重み）。
    assert_eq!(dir.count(), 3);
}

#[test]
fn req27_different_config_or_dataset_is_allowed() {
    let dir = TempDir::new("different");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1"), ("c2:seed0", b"w2")]);
    register(&ledger, b"other", &[("c1:seed0", b"w1")]);
    let calls = Cell::new(0u32);
    let w1 = Model::new(b"w1", None);
    let w2 = Model::new(b"w2", None);
    assert!(run(&ledger, DATA, "c1:seed0", &w1, &calls).is_ok());
    assert!(run(&ledger, DATA, "c2:seed0", &w2, &calls).is_ok());
    assert!(run(&ledger, b"other", "c1:seed0", &w1, &calls).is_ok());
    assert_eq!(calls.get(), 3);
}

#[test]
fn req27_same_weights_under_renamed_config_is_rejected() {
    let dir = TempDir::new("renamed");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    ledger
        .register_configs(
            &freeze_eval_data(DATA).unwrap().sha256(),
            &[
                reg("c1:seed0", b"w1"),
                reg("c1:renamed", b"w1"),
                reg("c1:tuned", b"w1").with_thresholds(Sha256Digest::of_bytes(b"t2")),
            ],
        )
        .unwrap();
    let calls = Cell::new(0u32);
    let m = Model::new(b"w1", None);
    assert!(run(&ledger, DATA, "c1:seed0", &m, &calls).is_ok());
    let renamed = run(&ledger, DATA, "c1:renamed", &m, &calls);
    assert!(is_already(&renamed, AppliedBy::ModelWeights));
    // 重み同一でしきい値だけ変えても拒否される。
    let tuned_model = Model::new(b"w1", Some(b"t2"));
    let tuned = run(&ledger, DATA, "c1:tuned", &tuned_model, &calls);
    assert!(is_already(&tuned, AppliedBy::ModelWeights));
    // 登録 1 + 代表 1 + 重み 1 + 別 ID の代表構成ロック 2（fail-closed で残る）。
    assert_eq!(dir.count(), 5);
    assert_eq!(calls.get(), 1);
}

/// P0 回帰（レビュー指摘）: 事前登録に無い ID・別重みでの最終 test 再適用は
/// 拒否され、ロックも作られず予測も呼ばれない。
#[test]
fn req27_unregistered_config_is_rejected_without_lock() {
    let dir = TempDir::new("unregistered");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let calls = Cell::new(0u32);
    assert!(run(&ledger, DATA, "c1:seed0", &Model::new(b"w1", None), &calls).is_ok());
    assert_eq!(dir.count(), 3);
    let r = run(&ledger, DATA, "c9:other", &Model::new(b"w9", None), &calls);
    assert!(
        matches!(acquire_err(&r), Some(AcquireError::UnregisteredConfig)),
        "{r:?}"
    );
    assert_eq!(dir.count(), 3);
    assert_eq!(calls.get(), 1);
}

#[test]
fn req27_apply_without_registration_is_rejected_without_lock() {
    let dir = TempDir::new("noreg");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let calls = Cell::new(0u32);
    let r = run(&ledger, DATA, "c1:seed0", &Model::new(b"w1", None), &calls);
    assert!(matches!(acquire_err(&r), Some(AcquireError::NotRegistered)));
    assert_eq!(dir.count(), 0);
    assert_eq!(calls.get(), 0);
}

#[test]
fn req27_registration_is_frozen_once() {
    let dir = TempDir::new("regonce");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let digest = freeze_eval_data(DATA).unwrap().sha256();
    ledger
        .register_configs(&digest, &[reg("b", b"1"), reg("a", b"1"), reg("a", b"1")])
        .unwrap();
    assert!(matches!(
        ledger.register_configs(&digest, &[reg("c", b"1")]),
        Err(AcquireError::AlreadyRegistered { .. })
    ));
    assert!(matches!(
        ledger.register_configs(&freeze_eval_data(b"x").unwrap().sha256(), &[]),
        Err(AcquireError::RegistryInvalid { .. })
    ));
    assert_eq!(dir.count(), 1);
}

#[test]
fn req27_registration_after_application_is_rejected() {
    let dir = TempDir::new("latereg");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let calls = Cell::new(0u32);
    assert!(run(&ledger, DATA, "c1:seed0", &Model::new(b"w1", None), &calls).is_ok());
    // 適用済みの代表構成ロックだけを別の台帳へ持ち込んでも、事後登録は拒否される。
    let dir2 = TempDir::new("latereg2");
    let digest = freeze_eval_data(DATA).unwrap().sha256();
    for entry in fs::read_dir(dir.path()).unwrap() {
        let e = entry.unwrap();
        if e.file_name().to_string_lossy().starts_with("config-") {
            fs::copy(e.path(), dir2.path().join(e.file_name())).unwrap();
        }
    }
    let ledger2 = FinalTestLedger::open(dir2.path()).unwrap();
    assert!(matches!(
        ledger2.register_configs(&digest, &[reg("c1:seed0", b"w1")]),
        Err(AcquireError::AlreadyRegistered { .. })
    ));
}

/// 予測に使うモデルからキーを導出する: `predict` に渡るパスは `apply_once` へ
/// 渡したものと同一で、評価中に重みを書き換えると不変性違反になる。
#[test]
fn req27_model_mutation_during_prediction_is_rejected() {
    let dir = TempDir::new("mutation");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let data_dir = TempDir::new("mutation-data");
    let path = data_dir.path().join("eval.bin");
    fs::write(&path, DATA).unwrap();
    let record = freeze_eval_data(DATA).unwrap();
    let frozen = FrozenEvalData {
        path: &path,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };
    let m = Model::new(b"w1", None);
    let r = apply_once(
        &ledger,
        &frozen,
        id("c1:seed0"),
        &m.paths(),
        dec,
        |_t, _b, p| {
            assert_eq!(p.weights, m.weights.as_path());
            fs::write(p.weights, b"swapped").unwrap();
            Ok::<_, String>(())
        },
    );
    assert!(matches!(
        r,
        Err(EvalDataInvarianceError::Evaluation(
            EvaluationInvarianceError::Changed(_)
        ))
    ));
}

#[test]
fn req27_missing_weights_file_does_not_create_lock() {
    let dir = TempDir::new("noweights");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let m = Model::new(b"w1", None);
    fs::remove_file(&m.weights).unwrap();
    let calls = Cell::new(0u32);
    let r = run(&ledger, DATA, "c1:seed0", &m, &calls);
    assert!(r.is_err());
    assert_eq!(dir.count(), 1);
    assert_eq!(calls.get(), 0);
}

#[test]
fn req27_poc10_methods_by_seeds_each_once() {
    let dir = TempDir::new("poc10");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let ids: Vec<String> = ["c1", "c2", "c3"]
        .iter()
        .flat_map(|m| (0..3).map(move |s| format!("{m}:seed{s}")))
        .collect();
    let weights: Vec<Vec<u8>> = ids
        .iter()
        .map(|cid| format!("weights-{cid}").into_bytes())
        .collect();
    let entries: Vec<(&str, &[u8])> = ids
        .iter()
        .zip(&weights)
        .map(|(cid, w)| (cid.as_str(), w.as_slice()))
        .collect();
    register(&ledger, DATA, &entries);
    let calls = Cell::new(0u32);
    for cid in &ids {
        let m = Model::new(format!("weights-{cid}").as_bytes(), None);
        assert!(run(&ledger, DATA, cid, &m, &calls).is_ok());
    }
    assert_eq!(dir.count(), 19);
    let again = Model::new(b"weights-c3:seed1", None);
    let r = run(&ledger, DATA, "c3:seed1", &again, &calls);
    assert!(is_already(&r, AppliedBy::RepresentativeConfig));
    assert_eq!(dir.count(), 19);
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

/// 台帳ディレクトリが失われた場合、事前登録を読めず予測を呼ばない（fail-closed）。
/// 永続化（fsync）失敗そのものは環境依存で再現できないため、実装側の `?` 伝播で担保する。
#[cfg(unix)]
#[test]
fn req27_ledger_dir_removed_does_not_call_prediction() {
    let dir = TempDir::new("durability");
    let ledger_dir = dir.path().join("ledger");
    fs::create_dir(&ledger_dir).unwrap();
    let ledger = FinalTestLedger::open(&ledger_dir).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    fs::remove_dir_all(&ledger_dir).unwrap();
    let calls = Cell::new(0u32);
    let r = run(&ledger, DATA, "c1:seed0", &Model::new(b"w1", None), &calls);
    assert!(matches!(
        acquire_err(&r),
        Some(AcquireError::NotRegistered | AcquireError::Io { .. })
    ));
    assert_eq!(calls.get(), 0);
}

#[test]
fn req27_record_contains_digests_and_id_only() {
    let dir = TempDir::new("record");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let data_dir = TempDir::new("record-data");
    let path = data_dir.path().join("eval.bin");
    fs::write(&path, DATA).unwrap();
    let record = freeze_eval_data(DATA).unwrap();
    let frozen = FrozenEvalData {
        path: &path,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };
    let m = Model::new(b"w1", None);
    let lock_path = apply_once(
        &ledger,
        &frozen,
        id("c1:seed0"),
        &m.paths(),
        dec,
        |t, _b, _p| Ok::<_, String>(t.config_lock_path().to_path_buf()),
    )
    .unwrap()
    .output;
    let body = fs::read_to_string(&lock_path).unwrap();
    assert!(body.starts_with("fandhe-edge-final-test-application v1\nlock=config\n"));
    assert!(body.contains(&format!(
        "eval_data_sha256={}\n",
        Sha256Digest::of_bytes(DATA).to_hex()
    )));
    assert!(body.contains("representative_config_id=c1:seed0\n"));
    assert!(body.contains(&format!(
        "weights_sha256={}\n",
        Sha256Digest::of_bytes(b"w1").to_hex()
    )));
    assert!(!body.contains("A\nA\nB"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = fs::metadata(&lock_path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
/// P0 回帰（レビュー指摘）: 登録済みだが未使用の ID に、登録と異なる重みを当てて
/// 最終 test を再適用することはできない。ロックも作られず予測も呼ばれない。
#[test]
fn req27_registered_id_with_different_weights_is_rejected() {
    let dir = TempDir::new("otherweights");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1"), ("c2:seed0", b"w2")]);
    let calls = Cell::new(0u32);
    assert!(run(&ledger, DATA, "c1:seed0", &Model::new(b"w1", None), &calls).is_ok());
    assert_eq!(dir.count(), 3);
    // 初回の結果を見てから、未使用 ID に登録外の重みを当てる。
    let r = run(
        &ledger,
        DATA,
        "c2:seed0",
        &Model::new(b"w-tuned", None),
        &calls,
    );
    assert!(
        matches!(acquire_err(&r), Some(AcquireError::WeightsNotRegistered)),
        "{r:?}"
    );
    assert_eq!(dir.count(), 3);
    assert_eq!(calls.get(), 1);
}

/// 同じ ID に異なる重みを登録することはできない。
#[test]
fn req27_registration_rejects_conflicting_weights_for_same_id() {
    let dir = TempDir::new("conflictreg");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let digest = freeze_eval_data(DATA).unwrap().sha256();
    let r = ledger.register_configs(&digest, &[reg("a", b"1"), reg("a", b"2")]);
    assert!(
        matches!(r, Err(AcquireError::RegistryInvalid { .. })),
        "{r:?}"
    );
    assert_eq!(dir.count(), 0);
}

/// P0 回帰（REQ-39）: 登録件数の上限超過は、複製・ソートの前に拒否される。
#[test]
fn req39_registration_over_limit_is_rejected() {
    let dir = TempDir::new("overlimit");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let digest = freeze_eval_data(DATA).unwrap().sha256();
    let entries: Vec<_> = (0..=fandhe_edge_eval::final_test_once::MAX_REGISTERED_CONFIGS)
        .map(|i| reg(&format!("c{i}"), b"w"))
        .collect();
    let r = ledger.register_configs(&digest, &entries);
    assert!(
        matches!(r, Err(AcquireError::RegistryInvalid { .. })),
        "{r:?}"
    );
    assert_eq!(dir.count(), 0);
}

/// P0 回帰（REQ-39）: 事前登録ファイルが FIFO に差し替えられていても、開く前に
/// 通常ファイルでないと判定して拒否し、無期限に待たない。
#[cfg(unix)]
#[test]
fn req39_registry_fifo_is_rejected_without_blocking() {
    let dir = TempDir::new("fifo");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let mut reg_path = None;
    for entry in fs::read_dir(dir.path()).unwrap() {
        let e = entry.unwrap();
        if e.file_name().to_string_lossy().starts_with("registry-") {
            reg_path = Some(e.path());
        }
    }
    let reg_path = reg_path.unwrap();
    fs::remove_file(&reg_path).unwrap();
    let status = std::process::Command::new("mkfifo")
        .arg(&reg_path)
        .status()
        .unwrap();
    assert!(status.success());
    let calls = Cell::new(0u32);
    let r = run(&ledger, DATA, "c1:seed0", &Model::new(b"w1", None), &calls);
    assert!(
        matches!(acquire_err(&r), Some(AcquireError::RegistryInvalid { .. })),
        "{r:?}"
    );
    assert_eq!(calls.get(), 0);
}

/// P0 回帰（レビュー指摘）: 重みが登録と一致しても、しきい値が登録と異なる（または
/// 登録に無いのに付いている・登録にあるのに無い）場合は拒否され、ロックも作られず
/// 予測も呼ばれない。未使用構成 B のしきい値を A の結果を見てから調整する迂回の拒否。
#[test]
fn req27_registered_id_with_different_thresholds_is_rejected() {
    let dir = TempDir::new("otherthresholds");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    ledger
        .register_configs(
            &freeze_eval_data(DATA).unwrap().sha256(),
            &[
                reg("a", b"wa"),
                reg("b", b"wb").with_thresholds(Sha256Digest::of_bytes(b"t-registered")),
                reg("c", b"wc"),
            ],
        )
        .unwrap();
    let calls = Cell::new(0u32);
    assert!(run(&ledger, DATA, "a", &Model::new(b"wa", None), &calls).is_ok());
    let locks = dir.count();
    // B: 重みは登録どおり、しきい値だけ調整済み。
    let tuned = run(
        &ledger,
        DATA,
        "b",
        &Model::new(b"wb", Some(b"t-tuned")),
        &calls,
    );
    assert!(
        matches!(
            acquire_err(&tuned),
            Some(AcquireError::ComponentNotRegistered {
                component: "thresholds"
            })
        ),
        "{tuned:?}"
    );
    // B: 登録にあるしきい値が無い。
    let missing = run(&ledger, DATA, "b", &Model::new(b"wb", None), &calls);
    assert!(matches!(
        acquire_err(&missing),
        Some(AcquireError::ComponentNotRegistered {
            component: "thresholds"
        })
    ));
    // C: 登録に無いしきい値が付いている。
    let extra = run(&ledger, DATA, "c", &Model::new(b"wc", Some(b"t")), &calls);
    assert!(matches!(
        acquire_err(&extra),
        Some(AcquireError::ComponentNotRegistered {
            component: "thresholds"
        })
    ));
    assert_eq!(dir.count(), locks);
    assert_eq!(calls.get(), 1);
    // 登録どおりなら適用できる。
    assert!(
        run(
            &ledger,
            DATA,
            "b",
            &Model::new(b"wb", Some(b"t-registered")),
            &calls
        )
        .is_ok()
    );
}

/// `from_package` は実ファイルから全構成要素のダイジェストを作り、登録どおりに通る。
#[test]
fn req27_from_package_binds_all_components() {
    let dir = TempDir::new("frompackage");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    let model = Model::new(b"w1", Some(b"t1"));
    let entry = RegisteredConfig::from_package(id("a"), &model.paths()).unwrap();
    assert_eq!(
        entry,
        reg("a", b"w1").with_thresholds(Sha256Digest::of_bytes(b"t1"))
    );
    ledger
        .register_configs(&freeze_eval_data(DATA).unwrap().sha256(), &[entry])
        .unwrap();
    let calls = Cell::new(0u32);
    assert!(run(&ledger, DATA, "a", &model, &calls).is_ok());
}

/// REQ-27: 予測関数へは各件の `input` だけが渡り、正解ラベルは評価器側が返す。
#[test]
fn req27_predict_receives_inputs_only_and_golds_stay_with_evaluator() {
    let dir = TempDir::new("ledger");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let m = Model::new(b"w1", None);
    let data_dir = TempDir::new("data");
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
        &m.paths(),
        dec,
        |_t, inputs, _p| Ok::<_, String>(inputs.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
    )
    .unwrap();
    assert_eq!(r.output, vec!["A", "A", "B"]);
    assert_eq!(r.golds, vec!["gold-A", "gold-A", "gold-B"]);
    assert!(r.output.iter().all(|i| !i.contains("gold-")));
}

/// REQ-27・REQ-39: 分解に失敗しても適用は消費され（本文を見てからの呼び直しを拒否）、
/// 予測は呼ばれず、エラー文言に理由（本文）を含まない。
#[test]
fn req27_decode_failure_consumes_application() {
    let dir = TempDir::new("ledger");
    let ledger = FinalTestLedger::open(dir.path()).unwrap();
    register(&ledger, DATA, &[("c1:seed0", b"w1")]);
    let m = Model::new(b"w1", None);
    let data_dir = TempDir::new("data");
    let path = data_dir.path().join("eval.bin");
    fs::write(&path, DATA).unwrap();
    let record = freeze_eval_data(DATA).unwrap();
    let frozen = FrozenEvalData {
        path: &path,
        sha256: record.sha256(),
        byte_len: record.byte_len(),
    };
    let calls = Cell::new(0u32);
    let r = apply_once(
        &ledger,
        &frozen,
        id("c1:seed0"),
        &m.paths(),
        |_b| Err(DecodeFailed),
        |_t, _i, _p| {
            calls.set(calls.get() + 1);
            Ok::<_, String>(())
        },
    );
    assert!(matches!(
        r,
        Err(EvalDataInvarianceError::Evaluation(
            EvaluationInvarianceError::Evaluation(ApplyOnceError::Decode)
        ))
    ));
    assert_eq!(calls.get(), 0);
    assert_eq!(
        format!("{}", ApplyOnceError::<String>::Decode),
        "failed to decode eval data"
    );
    // ロックは消費済みなので、正しい分解での再適用も拒否される。
    let again = run(&ledger, DATA, "c1:seed0", &m, &calls);
    assert!(matches!(
        again,
        Err(EvalDataInvarianceError::Evaluation(
            EvaluationInvarianceError::Evaluation(ApplyOnceError::Acquire(
                AcquireError::AlreadyApplied { .. }
            ))
        ))
    ));
    assert_eq!(calls.get(), 0);
}
