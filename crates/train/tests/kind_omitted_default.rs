//! REQ-19・TASK-19.2（issue #77）: `kind` を省略した経路の受け入れテスト
//! （証拠種別: テストハーネス。実際の学習・重みの取得は行わない）。
//!
//! 現行の定義ファイルには `kind` フィールドが無いため、共有 fixture
//! `definition.json` から作った学習パラメータは「kind を省略した定義」に
//! あたる。`resolve_kind_candidates(None, ..)` で既定候補（c1・c3）が探索候補に
//! なり、偽実行器で `run_search` を通すと、各成果物の `kind` が解決済みの
//! 既定候補と一致すること、選定が既定候補の一方になることを確認する。

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use fandhe_edge_core::definition::Definition;
use fandhe_edge_data::split::{Groupable, Split, SplitRatios};
use fandhe_edge_data::split_record::{SplitRecord, split_and_record};
use fandhe_edge_train::kind_resolution::{
    CommonTrainParams, ExplicitKind, KindSource, resolve_kind_candidates,
};
use fandhe_edge_train::request::{Device, TrainRequest, label_order_from_definition};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::search::{
    SearchBudget, SearchInput, SearchRecord, SelectionDecision, run_search,
};
use fandhe_edge_train::time_allotment::{CandidateRunner, Clock, PerCandidatePolicy};

const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;
const N: usize = 10;
const IDS: [&str; N] = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"];
const INPUTS: [&[u8]; N] = [
    b"i0", b"i1", b"i2", b"i3", b"i4", b"i5", b"i6", b"i7", b"i8", b"i9",
];

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("train_contract")
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    let len = fs::metadata(&path).expect("stat fixture").len();
    assert!(len <= MAX_FIXTURE_BYTES, "fixture too large");
    fs::read(&path).expect("read fixture")
}

struct Rec {
    id: String,
    group_id: String,
    label: String,
    input: Vec<u8>,
}

impl Groupable for Rec {
    fn id(&self) -> &str {
        &self.id
    }
    fn group_id(&self) -> &str {
        &self.group_id
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn input(&self) -> &[u8] {
        &self.input
    }
}

/// 全件が validation split に入る凍結記録（gold はすべて `positive`）。
fn split_record() -> SplitRecord {
    let records: Vec<Rec> = IDS
        .iter()
        .zip(INPUTS.iter())
        .enumerate()
        .map(|(i, (id, input))| Rec {
            id: (*id).to_string(),
            group_id: format!("g{i}"),
            label: "positive".to_string(),
            input: input.to_vec(),
        })
        .collect();
    let ratios = SplitRatios {
        train: 0.0,
        validation: 1.0,
        test: 0.0,
    };
    let recorded = split_and_record(&records, 0, &ratios).expect("valid ratios");
    let record = recorded.record().clone();
    assert_eq!(record.digest(Split::Validation).record_ids(), IDS);
    record
}

struct FakeClock(Mutex<Duration>);

impl Clock for FakeClock {
    fn monotonic(&self) -> Duration {
        *self.0.lock().expect("clock mutex")
    }
    fn unix_millis(&self) -> Result<u64, fandhe_edge_train::time_allotment::TimeAllotmentError> {
        Ok(1_700_000_000_000)
    }
}

/// リクエストの `kind` から成功 JSON を組み立てる偽実行器。`correct[kind]` 件を
/// 正解にする。受け取ったリクエスト JSON の `kind` と成果物の `kind` を記録する。
struct KindAwareRunner {
    correct: Vec<(&'static str, usize)>,
    wire_kinds: Vec<String>,
    artifact_kinds: Vec<String>,
}

impl CandidateRunner for KindAwareRunner {
    type Error = ();

    fn run(&mut self, request: &TrainRequest) -> Result<TrainOutcome, ()> {
        let wire: serde_json::Value =
            serde_json::from_slice(&request.to_json_vec().map_err(|_| ())?).map_err(|_| ())?;
        self.wire_kinds
            .push(wire["kind"].as_str().unwrap_or_default().to_string());
        let defaults: serde_json::Value =
            serde_json::from_slice(&read_fixture("kind_defaults.json")).map_err(|_| ())?;
        let mut config = defaults[request.kind()].as_object().cloned().ok_or(())?;
        for (k, v) in request.config() {
            config.insert(k.clone(), v.clone());
        }
        let mut value: serde_json::Value =
            serde_json::from_slice(&read_fixture("result_ok.json")).map_err(|_| ())?;
        value["artifact_dir"] =
            serde_json::Value::from(format!("{}/{}", request.root(), request.out_dir()));
        value["artifact"]["kind"] = request.kind().into();
        value["artifact"]["kind_version"] = request.kind_version().into();
        value["artifact"]["candidate_label"] = request.kind().into();
        value["artifact"]["config"] = serde_json::Value::Object(config);
        value["artifact"]["label_order"] =
            serde_json::to_value(request.label_order().as_slice()).map_err(|_| ())?;
        value["artifact"]["max_bytes"] = request.max_bytes().into();
        let correct = self
            .correct
            .iter()
            .find(|(k, _)| *k == request.kind())
            .map_or(0, |(_, n)| *n);
        let preds: Vec<serde_json::Value> = request
            .validation_inputs()
            .ok_or(())?
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let label = if i < correct { "positive" } else { "negative" };
                serde_json::json!({"id": v.id(), "status": "ok", "predicted_label": label})
            })
            .collect();
        value["validation_predictions"] = serde_json::Value::Array(preds);
        let stdout = serde_json::to_vec(&value).map_err(|_| ())?;
        let outcome = TrainOutcome::from_worker_stdout(&stdout, request).map_err(|_| ())?;
        if let TrainOutcome::Ok(ok) = &outcome {
            self.artifact_kinds.push(ok.artifact().kind().to_string());
        }
        Ok(outcome)
    }

    fn is_wall_timeout(_error: &()) -> bool {
        false
    }
}

fn common() -> CommonTrainParams {
    let def = Definition::parse(
        std::str::from_utf8(&read_fixture("definition.json")).expect("utf8 definition"),
    )
    .expect("valid definition");
    let labels = label_order_from_definition(&def).expect("label order");
    CommonTrainParams {
        label_order: labels.into_vec(),
        max_bytes: 512,
        seed: 1,
        device: Device::Cpu,
        root: "/fandhe-edge-fixture-root".to_string(),
        train_path: "train.jsonl".to_string(),
        out_dir: "out".to_string(),
        time_limit_seconds: None,
        rss_limit_bytes: None,
    }
}

fn search(
    explicit: Option<ExplicitKind>,
    correct: Vec<(&'static str, usize)>,
) -> (SearchRecord, KindAwareRunner) {
    let common = common();
    let label_order: Vec<String> = common.label_order.clone();
    let label_refs: Vec<&str> = label_order.iter().map(String::as_str).collect();
    let resolution = resolve_kind_candidates(explicit, common).expect("resolves");
    let split = split_record();
    let gold = ["positive"; N];
    let clock = FakeClock(Mutex::new(Duration::ZERO));
    let mut runner = KindAwareRunner {
        correct,
        wire_kinds: Vec::new(),
        artifact_kinds: Vec::new(),
    };
    let record = run_search(
        &mut runner,
        &clock,
        SearchInput {
            label_order: &label_refs,
            validation_gold: &gold,
            validation_record_ids: &IDS,
            validation_inputs: &INPUTS,
            validation_split_record: &split,
            candidates: resolution.into_candidates(),
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        },
    )
    .expect("search succeeds");
    (record, runner)
}

fn selected(record: &SearchRecord) -> (&str, &[String]) {
    match &record.selection {
        SelectionDecision::Selected {
            candidate_id,
            tied_candidate_ids,
            ..
        } => (candidate_id, tied_candidate_ids),
        other => panic!("expected Selected, got {other:?}"),
    }
}

/// REQ-19・TASK-19.2: kind を省いた定義で学習すると、既定候補の成果物
/// （`kind` の記録）が出て、選定が既定候補の一方になる。
#[test]
fn req19_train_without_kind_records_default_candidate_kind_in_artifact() {
    let (record, runner) = search(None, vec![("c1", 6), ("c3", 9)]);
    assert_eq!(runner.wire_kinds, ["c1", "c3"]);
    assert_eq!(runner.artifact_kinds, ["c1", "c3"]);
    let (id, _) = selected(&record);
    assert_eq!(id, "c3");
    assert_eq!(runner.artifact_kinds.get(1).map(String::as_str), Some("c3"));
}

/// REQ-19・TASK-19.2: 同率のときは宣言順（c1 → c3。暫定）で c1 が選ばれる。
#[test]
fn req19_train_without_kind_tie_prefers_declaration_order() {
    let (record, _) = search(None, vec![("c1", 8), ("c3", 8)]);
    let (id, tied) = selected(&record);
    assert_eq!(id, "c1");
    assert_eq!(tied, ["c1", "c3"]);
}

/// REQ-19・TASK-19.3: 明示した kind の経路は従来どおり 1 候補。
#[test]
fn req19_explicit_kind_path_is_unchanged() {
    let mut config = serde_json::Map::new();
    config.insert("epochs".to_string(), serde_json::Value::from(2));
    let explicit = ExplicitKind {
        kind: "c3".to_string(),
        kind_version: 1,
        config,
    };
    let resolution = resolve_kind_candidates(Some(explicit.clone()), common()).expect("resolves");
    assert_eq!(resolution.source(), KindSource::Explicit);
    assert_eq!(resolution.candidates().len(), 1);
    assert_eq!(resolution.candidates()[0].params.out_dir, "out");
    let (record, runner) = search(Some(explicit), vec![("c3", 7)]);
    assert_eq!(runner.artifact_kinds, ["c3"]);
    assert_eq!(selected(&record).0, "c3");
}

/// REQ-19: 解決記録の JSON（CLI が「既定候補が選ばれた」ことを出す形）。
#[test]
fn req19_resolution_record_json() {
    let resolution = resolve_kind_candidates(None, common()).expect("resolves");
    assert_eq!(
        serde_json::to_value(resolution.record()).expect("serialize"),
        serde_json::json!({
            "kind_source": "default",
            "candidates": [
                {"candidate_id": "c1", "kind": "c1", "kind_version": 1},
                {"candidate_id": "c3", "kind": "c3", "kind_version": 1}
            ]
        })
    );
}
