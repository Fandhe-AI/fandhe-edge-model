//! 台帳ディレクトリ操作の抽象（`LedgerDir`）越しでも台帳ロジックが変わらないことの結合テスト
//! （REQ-27・REQ-39・#314）。
//!
//! `StdLedgerDir` へ委譲しつつ全操作を記録するモック実装を `FinalTestLedger::with_dir` へ渡し、
//! (1) 事前登録・適用・完了記録・`is_applied` の結果がパス版と同じ、(2) 台帳のファイル操作が
//! すべて trait 経由で、名前が台帳直下からの相対（`..`・絶対パスなし・入れ子 1 段まで）、
//! (3) 永続化の失敗は予測を呼ばず適用を消費扱いにする（fail-closed）ことを確かめる。
//! 証拠の種別: テストハーネス（合成データ）。

use std::cell::Cell;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use fandhe_edge_core::fs::FsError;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_data::eval_freeze::freeze_eval_data;
use fandhe_edge_eval::eval_data_invariance::{EvalDataInvarianceError, FrozenEvalData};
use fandhe_edge_eval::final_test_once::{
    AcquireError, ApplyOnceError, DecodeFailed, FinalTestLedger, LabeledInput, RegisteredConfig,
    RepresentativeConfigId, apply_once,
};
use fandhe_edge_eval::invariance::{EvaluationInvarianceError, ModelPackagePaths};
use fandhe_edge_eval::ledger_dir::{EntryKind, LedgerDir, StdLedgerDir};

const DATA: &[u8] = b"A\nA\nB\n";

/// 作業ディレクトリ。Drop で片付ける。
struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "fandhe-edge-eval-ledger-dir-{}-{label}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        Scratch(p)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// 操作を記録して `StdLedgerDir` へ委譲するモック。`fail_sync` が真なら `sync_dir` を失敗させる。
#[derive(Clone)]
struct RecordingDir {
    inner: Arc<StdLedgerDir>,
    log: Arc<Mutex<Vec<(&'static str, String)>>>,
    fail_sync: Arc<AtomicBool>,
}

impl RecordingDir {
    fn new(root: &Path) -> Self {
        Self {
            inner: Arc::new(StdLedgerDir::new(root.to_path_buf())),
            log: Arc::default(),
            fail_sync: Arc::default(),
        }
    }

    fn note(&self, op: &'static str, rel: &str) {
        self.log.lock().unwrap().push((op, rel.to_string()));
    }
}

impl LedgerDir for RecordingDir {
    fn create_dir(&self, rel: &str) -> io::Result<()> {
        self.note("create_dir", rel);
        self.inner.create_dir(rel)
    }
    fn entry_kind(&self, rel: &str) -> io::Result<EntryKind> {
        self.note("entry_kind", rel);
        self.inner.entry_kind(rel)
    }
    fn create_new_file(&self, rel: &str) -> io::Result<File> {
        self.note("create_new_file", rel);
        self.inner.create_new_file(rel)
    }
    fn read_bounded(&self, rel: &str, limit: u64) -> Result<Vec<u8>, FsError> {
        self.note("read_bounded", rel);
        self.inner.read_bounded(rel, limit)
    }
    fn is_read_only_file(&self, rel: &str) -> bool {
        self.note("is_read_only_file", rel);
        self.inner.is_read_only_file(rel)
    }
    fn make_read_only(&self, rel: &str) -> io::Result<()> {
        self.note("make_read_only", rel);
        self.inner.make_read_only(rel)
    }
    fn list_names(&self, rel: &str, limit: usize) -> io::Result<Vec<Option<String>>> {
        self.note("list_names", rel);
        self.inner.list_names(rel, limit)
    }
    fn sync_dir(&self, rel: &str) -> io::Result<()> {
        self.note("sync_dir", rel);
        if self.fail_sync.load(Ordering::SeqCst) {
            return Err(io::Error::other("injected sync failure"));
        }
        self.inner.sync_dir(rel)
    }
    fn display_path(&self, rel: &str) -> PathBuf {
        self.inner.display_path(rel)
    }
}

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

type Applied = Result<
    fandhe_edge_eval::final_test_once::AppliedOnce<()>,
    EvalDataInvarianceError<EvaluationInvarianceError<ApplyOnceError<String>>>,
>;

struct Fixture {
    _scratch: Scratch,
    weights: PathBuf,
    data_path: PathBuf,
    ledger_root: PathBuf,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let scratch = Scratch::new(label);
        let weights = scratch.0.join("weights.bin");
        fs::write(&weights, b"w1").unwrap();
        let data_path = scratch.0.join("eval.bin");
        fs::write(&data_path, DATA).unwrap();
        let ledger_root = scratch.0.join("ledger");
        fs::create_dir(&ledger_root).unwrap();
        Self {
            _scratch: scratch,
            weights,
            data_path,
            ledger_root,
        }
    }

    fn id() -> RepresentativeConfigId {
        RepresentativeConfigId::parse("c1:seed0").unwrap()
    }

    fn register(&self, ledger: &FinalTestLedger) {
        let entry = RegisteredConfig::new(Self::id(), Sha256Digest::of_bytes(b"w1"));
        ledger
            .register_configs(&freeze_eval_data(DATA).unwrap().sha256(), &[entry])
            .unwrap();
    }

    fn apply(&self, ledger: &FinalTestLedger, calls: &Cell<u32>) -> Applied {
        let record = freeze_eval_data(DATA).unwrap();
        let frozen = FrozenEvalData {
            path: &self.data_path,
            sha256: record.sha256(),
            byte_len: record.byte_len(),
        };
        let model = ModelPackagePaths {
            weights: &self.weights,
            vocab: None,
            calibration: None,
            thresholds: None,
        };
        apply_once(ledger, &frozen, Self::id(), &model, dec, |_t, _i, _p| {
            calls.set(calls.get() + 1);
            Ok::<_, String>(())
        })
    }

    fn is_applied(&self, ledger: &FinalTestLedger) -> bool {
        ledger
            .is_applied(
                &freeze_eval_data(DATA).unwrap().sha256(),
                &Self::id(),
                &Sha256Digest::of_bytes(b"w1"),
            )
            .unwrap()
    }
}

/// REQ-27・REQ-39: モック経由でも、登録 → 適用 → 完了記録 → `is_applied` の結果と 1 回限りの
/// 拒否がパス版と同じで、台帳の全操作が trait 経由の台帳相対名（`..`・絶対パスなし）で行われる。
#[test]
fn req27_req39_ledger_logic_is_unchanged_through_the_trait() {
    let fx = Fixture::new("logic");
    let mock = RecordingDir::new(&fx.ledger_root);
    let ledger = FinalTestLedger::with_dir(mock.clone());
    fx.register(&ledger);
    assert!(!fx.is_applied(&ledger));
    let calls = Cell::new(0);
    assert!(fx.apply(&ledger, &calls).is_ok());
    assert_eq!(calls.get(), 1);
    assert!(fx.is_applied(&ledger));

    // 2 回目は拒否され、予測は呼ばれない（適用は 1 回限り）。
    let second = fx.apply(&ledger, &calls);
    assert!(matches!(
        second,
        Err(EvalDataInvarianceError::Evaluation(
            EvaluationInvarianceError::Evaluation(ApplyOnceError::Acquire(
                AcquireError::AlreadyApplied { .. }
            ))
        ))
    ));
    assert_eq!(calls.get(), 1);

    // パス版の台帳で同じディレクトリを開いても同じ状態（同一のファイル形式）。
    let plain = FinalTestLedger::open(&fx.ledger_root).unwrap();
    assert!(fx.is_applied(&plain));

    let log = mock.log.lock().unwrap();
    assert!(!log.is_empty());
    for (op, rel) in log.iter() {
        let p = Path::new(rel);
        assert!(
            rel.is_empty()
                || (p.is_relative()
                    && p.components().count() <= 2
                    && p.components()
                        .all(|c| matches!(c, std::path::Component::Normal(_)))),
            "{op}: {rel}"
        );
    }
    let ops: Vec<&str> = log.iter().map(|(op, _)| *op).collect();
    for want in [
        "create_dir",
        "entry_kind",
        "create_new_file",
        "read_bounded",
        "is_read_only_file",
        "make_read_only",
        "list_names",
        "sync_dir",
    ] {
        assert!(ops.contains(&want), "operation {want} must go via trait");
    }
}

/// REQ-27・REQ-39: 永続化（`sync_dir`）に失敗したら予測を呼ばず `DurabilityFailed` を返し、
/// 適用権は消費済みのまま（再適用は拒否）。fail-closed の挙動が trait 越しでも変わらない。
#[test]
fn req27_sync_failure_through_trait_is_fail_closed() {
    let fx = Fixture::new("syncfail");
    let mock = RecordingDir::new(&fx.ledger_root);
    let ledger = FinalTestLedger::with_dir(mock.clone());
    fx.register(&ledger);
    mock.fail_sync.store(true, Ordering::SeqCst);
    let calls = Cell::new(0);
    let first = fx.apply(&ledger, &calls);
    assert!(matches!(
        first,
        Err(EvalDataInvarianceError::Evaluation(
            EvaluationInvarianceError::Evaluation(ApplyOnceError::Acquire(
                AcquireError::DurabilityFailed { .. }
            ))
        ))
    ));
    assert_eq!(calls.get(), 0);
    mock.fail_sync.store(false, Ordering::SeqCst);
    assert!(fx.apply(&ledger, &calls).is_err());
    assert_eq!(calls.get(), 0);
    assert!(!fx.is_applied(&ledger));
}

/// REQ-27: 選定（代表構成 ID と選定記録のダイジェスト）は最初の適用の前に 1 回だけ固定でき、
/// 同じ選定なら冪等、異なる ID・ダイジェストは `SelectionChanged`、未固定の照合は `SelectionNotPinned`。
/// 固定は `LedgerDir` 越しに作られ、ロックは作られない。
#[test]
fn req27_selection_pin_is_fixed_once_and_rejects_changes() {
    let fx = Fixture::new("selpin");
    let mock = RecordingDir::new(&fx.ledger_root);
    let ledger = FinalTestLedger::with_dir(mock.clone());
    let eval = freeze_eval_data(DATA).unwrap().sha256();
    let a = RepresentativeConfigId::parse("c1:seed0").unwrap();
    let b = RepresentativeConfigId::parse("c3:seed0").unwrap();
    let (sel_a, sel_b) = (Sha256Digest::of_bytes(b"a"), Sha256Digest::of_bytes(b"b"));

    assert!(matches!(
        ledger.verify_selection_pin(&eval, &a, &sel_a),
        Err(AcquireError::SelectionNotPinned)
    ));
    ledger.pin_selection(&eval, &a, &sel_a).unwrap();
    ledger.pin_selection(&eval, &a, &sel_a).unwrap();
    ledger.verify_selection_pin(&eval, &a, &sel_a).unwrap();
    for (id, sel) in [(&b, &sel_a), (&a, &sel_b), (&b, &sel_b)] {
        assert!(matches!(
            ledger.pin_selection(&eval, id, sel),
            Err(AcquireError::SelectionChanged)
        ));
        assert!(matches!(
            ledger.verify_selection_pin(&eval, id, sel),
            Err(AcquireError::SelectionChanged)
        ));
    }
    let ops: Vec<&str> = mock.log.lock().unwrap().iter().map(|(o, _)| *o).collect();
    assert!(ops.contains(&"create_new_file") && ops.contains(&"make_read_only"));
}
