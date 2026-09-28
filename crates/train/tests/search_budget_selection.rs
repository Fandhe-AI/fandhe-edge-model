//! REQ-18・REQ-27・REQ-39（TASK-18.1-2・issue #84）: 探索予算全体の管理・
//! 複数候補の比較・選定の記録の受け入れ条件を確認する結合テスト
//! （証拠種別: テストハーネス）。
//!
//! `FakeClock`（`Mutex` で内部可変性を持つ仮想時計）・`FakeRunner`（呼び出し
//! ごとの所要 ms と結果種別を事前登録した実行器）・`FakeScorer`
//! （candidate_id ごとの validation 予測を返す推論器）で
//! `fandhe_edge_train::search::run_search` を検証する。`ValidationScorer:
//! Send + 'static`（P0 指摘対応・REQ-39。issue #84 PR #238 レビュー）に
//! なったことに伴い、`run_search` は scorer を締め切り付きの専用スレッドへ
//! 渡すため、`FakeClock`・`FakeScorer` はスレッド境界を越えられるよう
//! `candidate_time_limit.rs` と異なり `Cell` ではなく `Mutex`／`Arc` を使う
//! （モジュール内の各型 doc 参照）。ほとんどのテストは `thread::sleep`・
//! 実時間に依存しない（3 OS の CI での決定性のため。`.claude/rules/ci.md`）。
//! **例外は 1 件**（締め切りを守らない scorer を確認するテスト。
//! `task18_1_2_scorer_exceeding_deadline_is_recorded_as_timed_out`）で、
//! 締め切り超過を実際に発生させるため意図的に実時間の `thread::sleep` を
//! 使う（当該テストの doc コメント参照）。本テストは `crates/train`・
//! `crates/data`・`crates/eval`・`fandhe-edge-core` の範囲に閉じており、
//! `docs/spec` は参照しない。

use std::collections::BTreeMap;
use std::fs;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use fandhe_edge_data::split::{Groupable, Split, SplitRatios};
use fandhe_edge_data::split_record::{SplitRecord, split_and_record};
use fandhe_edge_eval::metrics::Outcome;
use fandhe_edge_train::request::{Device, TrainRequest, TrainRequestParams};
use fandhe_edge_train::result::{SuccessOutcome, TrainOutcome};
use fandhe_edge_train::search::{
    CandidateSearchResult, NotStartedReason, ScoredOutcome, SearchBudget, SearchCandidate,
    SearchError, SearchInput, SelectionDecision, ValidationInputRecord, ValidationScorer,
    run_search,
};
use fandhe_edge_train::time_allotment::{CandidateRunner, Clock, PerCandidatePolicy};

/// 外部入力と同じ作法で扱うための読み込み前サイズ上限（REQ-39）。
const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

const LABEL_ORDER: [&str; 3] = ["positive", "negative", "neutral"];
const VALIDATION_LEN: usize = 10;

/// `validation_gold()` と同じ順・同じ件数の validation レコード識別子
/// （REQ-27・P0 指摘対応。issue #84 PR #238 レビュー）。ほとんどのテストは
/// `run_search` がこれを [`ValidationScorer::predict_validation`] へそのまま
/// 渡すことのみ確認し、record_id 検証自体（不一致の拒否）は専用テストで
/// 確認する。
const VALIDATION_RECORD_IDS: [&str; VALIDATION_LEN] =
    ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"];

/// `validation_gold()`・`VALIDATION_RECORD_IDS` と同じ件数・同じ順の byte
/// 入力。各要素を区別できる値にしておき、`run_search` が
/// `SearchInput::validation_inputs` の要素を正しく
/// `ValidationScorer::predict_validation` へ転送すること（`b""` 等の
/// プレースホルダで置き換えても検出できない偽陰性を防ぐ）を機械照合できる
/// ようにする（P0・REQ-27 指摘対応。issue #84 PR #238 レビュー）。
const VALIDATION_INPUTS: [&[u8]; VALIDATION_LEN] = [
    b"i0", b"i1", b"i2", b"i3", b"i4", b"i5", b"i6", b"i7", b"i8", b"i9",
];

/// [`Groupable`] の最小実装（`fandhe_edge_data::split_record` のテストと
/// 同じ設計）。
struct SplitGroupable {
    id: String,
    group_id: String,
    label: String,
}

impl Groupable for SplitGroupable {
    fn id(&self) -> &str {
        &self.id
    }
    fn group_id(&self) -> &str {
        &self.group_id
    }
    fn label(&self) -> &str {
        &self.label
    }
}

/// `VALIDATION_RECORD_IDS` を validation split の record_ids として持つ
/// [`SplitRecord`] を組み立てる（P0 指摘対応・REQ-17・REQ-27。issue #84
/// PR #238 レビュー）。比率 `validation: 1.0`（train・test は `0.0`）を
/// 指定すると、`fandhe_edge_data::split::alloc_counts` の契約
/// （比率が厳密に `0.0` の split には一切割り付けない）により、全レコード
/// が validation split に入る。各レコードを別 group にして group 単位分割の
/// 影響を受けないようにする。
fn validation_split_record_fixture() -> SplitRecord {
    let records: Vec<SplitGroupable> = VALIDATION_RECORD_IDS
        .iter()
        .enumerate()
        .map(|(i, &id)| SplitGroupable {
            id: id.to_string(),
            group_id: format!("g{i}"),
            label: "positive".to_string(),
        })
        .collect();
    let ratios = SplitRatios {
        train: 0.0,
        validation: 1.0,
        test: 0.0,
    };
    let recorded = split_and_record(&records, 0, &ratios).expect("valid ratios");
    let record = recorded.record().clone();
    assert_eq!(
        record.digest(Split::Validation).record_ids(),
        VALIDATION_RECORD_IDS
    );
    record
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("train_contract")
}

fn load_fixture_value(name: &str) -> serde_json::Value {
    let path = fixture_dir().join(name);
    let metadata =
        fs::metadata(&path).unwrap_or_else(|e| panic!("failed to stat fixture {path:?}: {e}"));
    assert!(
        metadata.len() <= MAX_FIXTURE_BYTES,
        "fixture {path:?} exceeds size limit"
    );
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("failed to read fixture {path:?}: {e}"));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse fixture {path:?} as JSON: {e}"))
}

fn validation_gold() -> [&'static str; VALIDATION_LEN] {
    ["positive"; VALIDATION_LEN]
}

/// `correct` 件を gold と一致する `"positive"`、残りを不一致の `"negative"`
/// にした validation 予測列を作る（正解率を任意の値に制御するための
/// テスト専用ヘルパー）。
fn outcomes_with_correct(correct: usize) -> Vec<Outcome> {
    (0..VALIDATION_LEN)
        .map(|i| {
            if i < correct {
                Outcome::Label("positive".to_string())
            } else {
                Outcome::Label("negative".to_string())
            }
        })
        .collect()
}

fn candidate_params(out_dir: &str, seed: u32) -> TrainRequestParams {
    TrainRequestParams {
        kind: "c3".to_string(),
        kind_version: 1,
        config: serde_json::Map::from_iter([("epochs".to_string(), serde_json::Value::from(2))]),
        label_order: LABEL_ORDER.iter().map(|s| s.to_string()).collect(),
        max_bytes: 512,
        seed,
        device: Device::Cpu,
        root: "/fandhe-edge-fixture-root".to_string(),
        train_path: "train.jsonl".to_string(),
        out_dir: out_dir.to_string(),
        time_limit_seconds: None,
        rss_limit_bytes: None,
    }
}

/// テストが自由に進められる仮想時計。
///
/// `ValidationScorer: Send + 'static`（P0 指摘対応・REQ-39。issue #84
/// PR #238 レビュー）になったことに伴い、`FakeScorer` は `Arc<FakeClock>`
/// で本時計を共有する（`predict_validation` が専用スレッドで実行されるため、
/// `Cell`〔`candidate_time_limit.rs` と同じ従来設計〕のような `Sync` でない
/// 内部可変性は使えない）。`Mutex` で保護する。
struct FakeClock {
    elapsed: Mutex<Duration>,
    unix_ms: Mutex<u64>,
}

impl FakeClock {
    fn new(start_unix_ms: u64) -> Self {
        Self {
            elapsed: Mutex::new(Duration::ZERO),
            unix_ms: Mutex::new(start_unix_ms),
        }
    }

    fn advance_ms(&self, ms: u64) {
        let mut elapsed = self.elapsed.lock().expect("fake clock mutex poisoned");
        *elapsed += Duration::from_millis(ms);
        let mut unix_ms = self.unix_ms.lock().expect("fake clock mutex poisoned");
        *unix_ms += ms;
    }
}

impl Clock for FakeClock {
    fn monotonic(&self) -> Duration {
        *self.elapsed.lock().expect("fake clock mutex poisoned")
    }

    fn unix_millis(&self) -> Result<u64, fandhe_edge_train::time_allotment::TimeAllotmentError> {
        Ok(*self.unix_ms.lock().expect("fake clock mutex poisoned"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FakeRunnerError(&'static str);

impl std::fmt::Display for FakeRunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fake runner error: {}", self.0)
    }
}

/// 呼び出し 1 回分の学習ワーカーの振る舞い（宣言順の候補と 1 対 1 対応）。
#[derive(Debug, Clone)]
enum RunnerBehavior {
    /// 成功。`artifact_dir` は `request` から動的に組み立てる。
    Ok { advance_ms: u64 },
    /// `limit_exceeded` で打ち切り。
    LimitExceeded { advance_ms: u64 },
    /// 実行器自体が失敗する（学習ワーカーの起動失敗等）。
    RunnerError,
}

/// 事前登録した振る舞いを順番に返す実行器。呼び出し回数を記録する。
struct FakeRunner<'a> {
    clock: &'a FakeClock,
    behaviors: Vec<RunnerBehavior>,
    calls: usize,
}

impl<'a> FakeRunner<'a> {
    fn new(clock: &'a FakeClock, behaviors: Vec<RunnerBehavior>) -> Self {
        Self {
            clock,
            behaviors,
            calls: 0,
        }
    }
}

/// `result_ok.json` fixture の `artifact_dir` を `request` の `root`／
/// `out_dir` から組み立て直した 1 行 JSON にする（`request` 間で `out_dir`
/// だけが異なる本テストの構成に合わせる）。
fn ok_stdout_for_request(request: &TrainRequest) -> String {
    let mut value = load_fixture_value("result_ok.json");
    let artifact_dir = format!("{}/{}", request.root(), request.out_dir());
    value["artifact_dir"] = serde_json::Value::from(artifact_dir);
    serde_json::to_string(&value).expect("serialize fixture-derived stdout")
}

impl CandidateRunner for FakeRunner<'_> {
    type Error = FakeRunnerError;

    fn run(&mut self, request: &TrainRequest) -> Result<TrainOutcome, Self::Error> {
        let index = self.calls;
        self.calls += 1;
        let behavior = self
            .behaviors
            .get(index)
            .cloned()
            .unwrap_or_else(|| panic!("unexpected extra runner call at index {index}"));
        match behavior {
            RunnerBehavior::Ok { advance_ms } => {
                self.clock.advance_ms(advance_ms);
                let stdout = ok_stdout_for_request(request);
                TrainOutcome::from_worker_stdout(stdout.as_bytes(), request)
                    .map_err(|_| FakeRunnerError("worker stdout did not match request"))
            }
            RunnerBehavior::LimitExceeded { advance_ms } => {
                self.clock.advance_ms(advance_ms);
                let stdout = r#"{"status":"error","code":"limit_exceeded","message":"training exceeded wall-clock budget"}"#;
                TrainOutcome::from_worker_stdout(stdout.as_bytes(), request)
                    .map_err(|_| FakeRunnerError("worker stdout did not match request"))
            }
            RunnerBehavior::RunnerError => Err(FakeRunnerError("spawn failed")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FakeScorerError;

impl std::fmt::Display for FakeScorerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fake scorer error")
    }
}

/// [`FakeScorer`] の呼び出し履歴（P0 指摘対応・REQ-39。issue #84 PR #238
/// レビュー）。
///
/// `ValidationScorer: Send + 'static` になったことに伴い、`run_search` は
/// `scorer`（[`FakeScorer`]）の所有権を締め切り付きの専用スレッドへ渡す
/// （`call_predict_validation_with_deadline` 参照）。呼び出し後に
/// `FakeScorer` 自身を直接検査できなくなる（所有権が `run_search` 側へ
/// 移り、スレッドを介して返るか、締め切り超過時は戻ってこない）ため、
/// `Arc<Mutex<_>>` で共有する別データとして呼び出し履歴を保持し、
/// `run_search` へ `scorer` を渡す前に [`FakeScorer::probe`] で複製した
/// ハンドルをテスト側に残しておく。
#[derive(Debug, Default)]
struct FakeScorerProbe {
    calls: Vec<String>,
    time_limits: Vec<Duration>,
    /// 呼び出しごとに受け取った `(record_id, input)` の組（P0・REQ-27:
    /// `run_search` が `SearchInput::validation_inputs` を `record_id` と
    /// 揃えて転送していることを機械照合するための記録。`input` を
    /// プレースホルダ〔例: 全件 `b""`〕にすると転送漏れを検出できないため、
    /// `VALIDATION_INPUTS` は要素ごとに異なる値にする）。
    received_records: Vec<Vec<(String, Vec<u8>)>>,
}

/// candidate_id ごとに事前登録した validation 予測（または失敗）を返す
/// 推論器。呼び出し順・件数・受け取った `time_limit`・`records`
/// （record_id・byte 入力の組）を [`FakeScorerProbe`] へ記録する（REQ-27:
/// `validation_gold`〔正解ラベル〕を渡さないことは trait の署名で構造的に
/// 保証される）。
///
/// `advance_ms`（candidate_id ごと）を指定すると、`predict_validation` の
/// 呼び出し中に `clock`（[`Arc<FakeClock>`]。全候補・`run_search` 本体と
/// 共有する）を進める（P0・REQ-39: 採点が探索予算を超過するケースを模擬
/// する。呼び出し自体を打ち切れないことをテストでも示す）。`sleep_ms`
/// （candidate_id ごと）を指定すると、`predict_validation` の呼び出し中に
/// 実時間で `thread::sleep` する（P0 指摘対応・REQ-39: 締め切りを守らない
/// 実装が `ScoringTimedOut` として打ち切られることを確認するテスト専用。
/// 本テストのみ実時間に依存する）。
///
/// 既定では受け取った `records` の `record_id` をそのまま echo する
/// （正直な実装を模擬）。`record_id_override`（candidate_id ごと）を指定
/// すると、代わりに別の record_id 列を返す（P0・REQ-27: `run_search` が
/// record_id の不一致を検出することを確認するテスト専用）。
struct FakeScorer {
    clock: Arc<FakeClock>,
    responses: BTreeMap<String, Result<Vec<Outcome>, FakeScorerError>>,
    advance_ms: BTreeMap<String, u64>,
    sleep_ms: BTreeMap<String, u64>,
    record_id_override: BTreeMap<String, Vec<String>>,
    probe: Arc<Mutex<FakeScorerProbe>>,
}

impl FakeScorer {
    fn new(
        clock: Arc<FakeClock>,
        responses: BTreeMap<String, Result<Vec<Outcome>, FakeScorerError>>,
    ) -> Self {
        Self {
            clock,
            responses,
            advance_ms: BTreeMap::new(),
            sleep_ms: BTreeMap::new(),
            record_id_override: BTreeMap::new(),
            probe: Arc::new(Mutex::new(FakeScorerProbe::default())),
        }
    }

    /// 呼び出し履歴を検査するための共有ハンドルを複製する。`run_search`
    /// （所有権を取る）へ `self` を渡す**前**に呼ぶこと。
    fn probe(&self) -> Arc<Mutex<FakeScorerProbe>> {
        Arc::clone(&self.probe)
    }

    /// `candidate_id` の採点呼び出し中に `clock` を `ms` だけ進めるよう
    /// 登録する（P0 テスト専用）。
    fn with_advance(mut self, candidate_id: &str, ms: u64) -> Self {
        self.advance_ms.insert(candidate_id.to_string(), ms);
        self
    }

    /// `candidate_id` の採点呼び出し中に実時間で `ms` ミリ秒 `thread::sleep`
    /// するよう登録する（P0 指摘対応・REQ-39 テスト専用: 締め切りを守らない
    /// 実装を模擬する）。
    fn with_sleep(mut self, candidate_id: &str, ms: u64) -> Self {
        self.sleep_ms.insert(candidate_id.to_string(), ms);
        self
    }

    /// `candidate_id` の戻り値の `record_id` 列を、受け取った `records` の
    /// `record_id` とは無関係な `ids` へ差し替える（P0・REQ-27 テスト専用:
    /// 件数は同じだが順序・内容が異なる予測を模擬する）。
    fn with_record_id_override(mut self, candidate_id: &str, ids: Vec<&str>) -> Self {
        self.record_id_override.insert(
            candidate_id.to_string(),
            ids.into_iter().map(str::to_string).collect(),
        );
        self
    }
}

impl ValidationScorer for FakeScorer {
    type Error = FakeScorerError;

    fn predict_validation(
        &mut self,
        candidate_id: &str,
        _artifact: &SuccessOutcome,
        records: &[ValidationInputRecord<'_>],
        time_limit: Duration,
    ) -> Result<Vec<ScoredOutcome>, Self::Error> {
        {
            let mut probe = self.probe.lock().expect("fake scorer probe poisoned");
            probe.calls.push(candidate_id.to_string());
            probe.time_limits.push(time_limit);
            probe.received_records.push(
                records
                    .iter()
                    .map(|record| (record.record_id.to_string(), record.input.to_vec()))
                    .collect(),
            );
        }
        if let Some(&ms) = self.sleep_ms.get(candidate_id) {
            // P0 指摘対応（REQ-39）: 締め切りを守らない scorer を模擬する
            // ため、本テストに限り実時間で待つ（`call_predict_validation_with_deadline`
            // が別スレッドで本メソッドを呼ぶため、ここで実時間 sleep しても
            // 呼び出し元（`recv_timeout`）は待たされない）。
            thread::sleep(Duration::from_millis(ms));
        }
        if let Some(&ms) = self.advance_ms.get(candidate_id) {
            self.clock.advance_ms(ms);
        }
        let outcomes = match self.responses.get(candidate_id).cloned() {
            Some(Ok(outcomes)) => outcomes,
            Some(Err(e)) => return Err(e),
            None => panic!("unexpected scorer call for {candidate_id}"),
        };
        let ids: Vec<String> = match self.record_id_override.get(candidate_id) {
            Some(overridden) => overridden.clone(),
            None => records
                .iter()
                .map(|record| record.record_id.to_string())
                .collect(),
        };
        Ok(ids
            .into_iter()
            .zip(outcomes)
            .map(|(record_id, outcome)| ScoredOutcome { record_id, outcome })
            .collect())
    }
}

fn fixed_policy(seconds: u32) -> PerCandidatePolicy {
    PerCandidatePolicy::Fixed(NonZeroU32::new(seconds).expect("non-zero"))
}

/// (T1・主受入) 既定予算・`EvenSplit`・3 候補。正解数 7/10・9/10・8/10 の
/// うち最高正解率の候補（9/10）が選ばれる。
#[test]
fn task18_1_2_default_budget_selects_highest_accuracy_candidate() {
    let clock = Arc::new(FakeClock::new(1_700_000_000_000));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 1_000 },
            RunnerBehavior::Ok { advance_ms: 1_000 },
            RunnerBehavior::Ok { advance_ms: 1_000 },
        ],
    );
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([
            ("c3-a".to_string(), Ok(outcomes_with_correct(7))),
            ("c3-b".to_string(), Ok(outcomes_with_correct(9))),
            ("c3-c".to_string(), Ok(outcomes_with_correct(8))),
        ]),
    );

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
        SearchCandidate {
            candidate_id: "c3-c".to_string(),
            params: candidate_params("out/c3-c", 3),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");

    match &record.selection {
        SelectionDecision::Selected {
            candidate_id,
            validation_accuracy,
            rule,
            ..
        } => {
            assert_eq!(candidate_id, "c3-b");
            assert_eq!(validation_accuracy.correct, 9);
            assert_eq!(validation_accuracy.total, 10);
            assert_eq!(rule, "validation_accuracy_desc_then_candidate_order");
        }
        SelectionDecision::NoEligibleCandidate => panic!("expected Selected"),
        _ => panic!("unexpected selection decision"),
    }

    let expected_time_limits = [1200u32, 1799, 3598];
    for (entry, expected) in record.candidates.iter().zip(expected_time_limits) {
        let time = entry.time.as_ref().expect("candidate must have run");
        assert_eq!(
            time.time_limit_seconds(),
            expected,
            "candidate {}",
            entry.candidate_id
        );
    }
    assert_eq!(runner.calls, 3);
    assert_eq!(
        probe.lock().expect("probe poisoned").calls,
        vec!["c3-a", "c3-b", "c3-c"]
    );
}

/// (T2・予算の消費) 予算 100 秒・`Fixed(50)`・4 候補（各 50,000ms）。
/// c3-a は評価済み。c3-b は学習だけでちょうど探索予算全体を使い切るため
/// 採点を呼び出さず `scoring_skipped_budget_exhausted` になる（P0 指摘
/// 対応・REQ-39・issue #84 PR #238 レビュー）。c3-c・c3-d は c3-b の時点で
/// 実行順が一度も回ってこず、どちらも `elapsed_at_start_ms: None`・
/// `not_started/budget_exhausted` になる（P1 指摘対応・REQ-18
/// 「候補ごとの選定記録」。4 候補目を追加したのは「予算切れ候補の直後だけ
/// でなく、さらにその後ろの候補」も記録されることを検証するため）。
#[test]
fn task18_1_2_budget_exhaustion_marks_remaining_candidates_not_started() {
    let clock = Arc::new(FakeClock::new(1_700_000_000_000));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 50_000 },
            RunnerBehavior::Ok { advance_ms: 50_000 },
        ],
    );
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([
            ("c3-a".to_string(), Ok(outcomes_with_correct(6))),
            ("c3-b".to_string(), Ok(outcomes_with_correct(7))),
        ]),
    );

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
        SearchCandidate {
            candidate_id: "c3-c".to_string(),
            params: candidate_params("out/c3-c", 3),
        },
        SearchCandidate {
            candidate_id: "c3-d".to_string(),
            params: candidate_params("out/c3-d", 4),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::new(100).expect("non-zero"),
        policy: fixed_policy(50),
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");

    assert_eq!(runner.calls, 2);
    assert_eq!(record.total_elapsed_ms, 100_000);
    assert_eq!(record.candidates.len(), 4);
    assert!(matches!(
        record.candidates[0].result,
        CandidateSearchResult::Evaluated { .. }
    ));
    // c3-b は学習だけでちょうど探索予算全体（100_000ms）を使い切るため、
    // 採点（`predict_validation`）を呼び出さずに打ち切る（P0 指摘対応・
    // issue #84 PR #238 レビュー。旧実装はここで `Evaluated` になり選定され
    // 得た）。
    assert_eq!(
        record.candidates[1].result,
        CandidateSearchResult::ScoringSkippedBudgetExhausted
    );
    assert!(record.candidates[1].time.is_some());
    assert_eq!(
        probe.lock().expect("probe poisoned").calls,
        vec!["c3-a".to_string()]
    );
    // c3-b の時点で探索予算全体を使い切ったため、c3-c・c3-d はどちらも
    // 実行順が回ってこず（`drain_remaining_as_not_started`）、
    // `elapsed_at_start_ms` は `None` になる。
    assert_eq!(record.candidates[2].candidate_id, "c3-c");
    assert_eq!(record.candidates[2].elapsed_at_start_ms, None);
    assert_eq!(record.candidates[2].time, None);
    assert_eq!(
        record.candidates[2].result,
        CandidateSearchResult::NotStarted {
            reason: NotStartedReason::BudgetExhausted
        }
    );
    assert_eq!(record.candidates[3].candidate_id, "c3-d");
    assert_eq!(record.candidates[3].elapsed_at_start_ms, None);
    assert_eq!(record.candidates[3].time, None);
    assert_eq!(
        record.candidates[3].result,
        CandidateSearchResult::NotStarted {
            reason: NotStartedReason::BudgetExhausted
        }
    );
    match &record.selection {
        // c3-b が選定対象から除外されるため、唯一評価済みの c3-a が選ばれる。
        SelectionDecision::Selected { candidate_id, .. } => assert_eq!(candidate_id, "c3-a"),
        SelectionDecision::NoEligibleCandidate => panic!("expected Selected"),
        _ => panic!("unexpected selection decision"),
    }
}

/// (T3・同率) 8/10 と 8/10 のとき先の候補を選び、`tied_candidate_ids` に
/// 両方が入る。
#[test]
fn task18_1_2_tie_breaks_to_first_declared_candidate() {
    let clock = Arc::new(FakeClock::new(1_700_000_000_000));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 100 },
            RunnerBehavior::Ok { advance_ms: 100 },
        ],
    );
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([
            ("c3-a".to_string(), Ok(outcomes_with_correct(8))),
            ("c3-b".to_string(), Ok(outcomes_with_correct(8))),
        ]),
    );

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    match &record.selection {
        SelectionDecision::Selected {
            candidate_id,
            tied_candidate_ids,
            ..
        } => {
            assert_eq!(candidate_id, "c3-a");
            assert_eq!(
                tied_candidate_ids,
                &vec!["c3-a".to_string(), "c3-b".to_string()]
            );
        }
        SelectionDecision::NoEligibleCandidate => panic!("expected Selected"),
        _ => panic!("unexpected selection decision"),
    }
}

/// (T4・対象なし) 全候補が `limit_exceeded` のとき `NoEligibleCandidate`。
/// scorer は 1 回も呼ばれない。
#[test]
fn task18_1_2_no_eligible_candidate_when_all_candidates_fail() {
    let clock = Arc::new(FakeClock::new(1_700_000_000_000));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::LimitExceeded { advance_ms: 10 },
            RunnerBehavior::LimitExceeded { advance_ms: 10 },
        ],
    );
    let scorer = FakeScorer::new(Arc::clone(&clock), BTreeMap::new());

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
    assert!(probe.lock().expect("probe poisoned").calls.is_empty());
    for entry in &record.candidates {
        let time = entry.time.as_ref().expect("candidate must have run");
        assert!(matches!(
            time.status(),
            fandhe_edge_train::time_allotment::CandidateTimeStatus::LimitExceeded { .. }
        ));
    }
}

/// (T5・採点失敗) 1 候補の scorer が Err を返すとその候補は
/// `scoring_failed` になり、残りの候補から選定される。
#[test]
fn task18_1_2_scoring_failure_is_recorded_and_search_continues() {
    let clock = Arc::new(FakeClock::new(1_700_000_000_000));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 100 },
            RunnerBehavior::Ok { advance_ms: 100 },
        ],
    );
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([
            ("c3-a".to_string(), Err(FakeScorerError)),
            ("c3-b".to_string(), Ok(outcomes_with_correct(6))),
        ]),
    );

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringFailed
    );
    match &record.selection {
        SelectionDecision::Selected { candidate_id, .. } => assert_eq!(candidate_id, "c3-b"),
        SelectionDecision::NoEligibleCandidate => panic!("expected Selected"),
        _ => panic!("unexpected selection decision"),
    }
}

/// (P1・REQ-18・REQ-39・issue #84 PR #238 レビュー) 採点
/// （[`ValidationScorer::predict_validation`]）が `Err` を返した場合も、
/// 成功時に `predict_validation` から戻った直後へ移した判定
/// （`ScoringSkippedBudgetExhausted`）と同じく呼び出し後の経過時間を確認
/// する。採点呼び出し中に探索予算全体を使い切っていれば、`ScoringFailed` の候補は
/// 記録しつつ残り候補を実行せず `NotStarted { reason: BudgetExhausted }` として
/// 一括で未着手にする（`drain_remaining_as_not_started`）。
#[test]
fn task18_1_2_scoring_failure_after_budget_exhausted_stops_remaining_candidates() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::Ok { advance_ms: 10 }]);
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([("c3-a".to_string(), Err(FakeScorerError))]),
    )
    .with_advance("c3-a", 4_000_000);

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(
        runner.calls, 1,
        "c3-b は採点失敗後の予算超過により実行されない"
    );
    assert_eq!(record.candidates.len(), 2);
    assert_eq!(record.candidates[0].candidate_id, "c3-a");
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringFailed
    );
    assert_eq!(record.candidates[1].candidate_id, "c3-b");
    assert_eq!(record.candidates[1].elapsed_at_start_ms, None);
    assert_eq!(
        record.candidates[1].result,
        CandidateSearchResult::NotStarted {
            reason: NotStartedReason::BudgetExhausted
        }
    );
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// `VALIDATION_RECORD_IDS` とは異なる record_ids（"x0".."x9"）を validation
/// split として持つ [`SplitRecord`] を組み立てる（P0 指摘対応・REQ-17・
/// REQ-27 テスト専用。issue #84 PR #238 レビュー: `validation_split_record_fixture`
/// との不一致を確認するための対照）。
fn mismatched_split_record_fixture() -> SplitRecord {
    let records: Vec<SplitGroupable> = (0..VALIDATION_LEN)
        .map(|i| SplitGroupable {
            id: format!("x{i}"),
            group_id: format!("g{i}"),
            label: "positive".to_string(),
        })
        .collect();
    let ratios = SplitRatios {
        train: 0.0,
        validation: 1.0,
        test: 0.0,
    };
    let recorded = split_and_record(&records, 0, &ratios).expect("valid ratios");
    recorded.record().clone()
}

/// (T5b・P0・REQ-17・REQ-27。issue #84 PR #238 レビュー) `validation_record_ids`
/// から再計算したハッシュが、`validation_split_record`（凍結済み validation
/// split の記録）の `validation` split のハッシュと一致しない場合、
/// `ValidationSplitHashMismatch` として拒否され、runner・scorer のいずれも
/// 呼び出されない（採点を呼び出す前の fail-closed な事前検証）。
#[test]
fn task18_1_2_validation_split_hash_mismatch_is_rejected_before_scoring() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let scorer = FakeScorer::new(Arc::clone(&clock), BTreeMap::new());

    let gold = validation_gold();
    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let mismatched_record = mismatched_split_record_fixture();
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &mismatched_record,
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let err = run_search(&mut runner, scorer, &*clock, input).unwrap_err();
    assert_eq!(err, SearchError::ValidationSplitHashMismatch);
    assert_eq!(runner.calls, 0);
    assert!(probe.lock().expect("probe poisoned").calls.is_empty());
}

/// (T5c・P0・REQ-39。issue #84 PR #238 レビュー) 採点（`predict_validation`）
/// が締め切り（`time_limit`）以内に戻らない scorer は `ScoringTimedOut` と
/// して打ち切られ、戻り値は使われない。締め切りを守らない実装を
/// `thread::sleep` で模擬する（本テストのみ実時間に依存する。trait doc
/// 「時間上限」参照）。以降の候補も採点できないため未着手になる。
#[test]
fn task18_1_2_scorer_exceeding_deadline_is_recorded_as_timed_out() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 0 },
            RunnerBehavior::Ok { advance_ms: 0 },
        ],
    );
    // 予算 1 秒（1000ms）に対し、scorer は実時間で 2 秒 sleep する
    // （締め切りを大きく超える。CI が多少遅くても誤判定しないよう十分な
    // 余裕を持たせる）。
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([
            ("c3-a".to_string(), Ok(outcomes_with_correct(9))),
            ("c3-b".to_string(), Ok(outcomes_with_correct(9))),
        ]),
    )
    .with_sleep("c3-a", 2_000);

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::new(1).expect("non-zero"),
        // `EvenSplit` だと 1 秒を候補 2 件で割って 0 秒（`Allotment::Exhausted`）
        // になり、採点まで到達せず本テストの意図（締め切り超過）を検証
        // できない。`Fixed(1)` は `allot` が残り予算（1 秒）で頭打ちにしつつ
        // 候補ごとに 1 秒を配分するため、学習自体は完了して採点まで進む。
        policy: fixed_policy(1),
    };

    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(record.candidates.len(), 2);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringTimedOut
    );
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    // scorer の所有権を取り戻せないため、c3-b は実行順が回ってこない
    // （既存の予算切れと同じ「残り候補を未着手にする」扱い）。
    assert_eq!(record.candidates[1].candidate_id, "c3-b");
    assert_eq!(record.candidates[1].elapsed_at_start_ms, None);
    assert_eq!(
        record.candidates[1].result,
        CandidateSearchResult::NotStarted {
            reason: NotStartedReason::BudgetExhausted
        }
    );
    assert_eq!(runner.calls, 1, "c3-b は締め切り超過後に実行されない");
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T6・事前検証) ID 重複・`label_order` の不一致・gold の未知ラベル・
/// `out_dir` の重複はそれぞれ Err になり、runner の呼び出しは 0 回。
#[test]
fn task18_1_2_precondition_violations_do_not_consume_budget() {
    let gold = validation_gold();

    // ID 重複。
    {
        let clock = Arc::new(FakeClock::new(0));
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let scorer = FakeScorer::new(Arc::clone(&clock), BTreeMap::new());
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: candidate_params("out/c3-a", 1),
            },
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: candidate_params("out/c3-b", 2),
            },
        ];
        let input = SearchInput {
            label_order: &LABEL_ORDER,
            validation_gold: &gold,
            validation_record_ids: &VALIDATION_RECORD_IDS,
            validation_inputs: &VALIDATION_INPUTS,
            validation_split_record: &validation_split_record_fixture(),
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = run_search(&mut runner, scorer, &*clock, input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateCandidateId { index: 1 });
        assert_eq!(runner.calls, 0);
    }

    // label_order の不一致。
    {
        let clock = Arc::new(FakeClock::new(0));
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let scorer = FakeScorer::new(Arc::clone(&clock), BTreeMap::new());
        let mut params = candidate_params("out/c3-a", 1);
        params.label_order = vec!["negative".to_string(), "positive".to_string()];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params,
        }];
        let input = SearchInput {
            label_order: &LABEL_ORDER,
            validation_gold: &gold,
            validation_record_ids: &VALIDATION_RECORD_IDS,
            validation_inputs: &VALIDATION_INPUTS,
            validation_split_record: &validation_split_record_fixture(),
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = run_search(&mut runner, scorer, &*clock, input).unwrap_err();
        assert_eq!(err, SearchError::LabelOrderMismatch { index: 0 });
        assert_eq!(runner.calls, 0);
    }

    // gold の未知ラベル。
    {
        let clock = Arc::new(FakeClock::new(0));
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let scorer = FakeScorer::new(Arc::clone(&clock), BTreeMap::new());
        let bad_gold = ["unknown_label"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        }];
        let input = SearchInput {
            label_order: &LABEL_ORDER,
            validation_gold: &bad_gold,
            validation_record_ids: &VALIDATION_RECORD_IDS[..1],
            validation_inputs: &VALIDATION_INPUTS[..1],
            validation_split_record: &validation_split_record_fixture(),
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = run_search(&mut runner, scorer, &*clock, input).unwrap_err();
        assert_eq!(err, SearchError::UnknownValidationGold { index: 0 });
        assert_eq!(runner.calls, 0);
    }

    // out_dir の重複。
    {
        let clock = Arc::new(FakeClock::new(0));
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let scorer = FakeScorer::new(Arc::clone(&clock), BTreeMap::new());
        let candidates = vec![
            SearchCandidate {
                candidate_id: "c3-a".to_string(),
                params: candidate_params("out/same", 1),
            },
            SearchCandidate {
                candidate_id: "c3-b".to_string(),
                params: candidate_params("out/same", 2),
            },
        ];
        let input = SearchInput {
            label_order: &LABEL_ORDER,
            validation_gold: &gold,
            validation_record_ids: &VALIDATION_RECORD_IDS,
            validation_inputs: &VALIDATION_INPUTS,
            validation_split_record: &validation_split_record_fixture(),
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = run_search(&mut runner, scorer, &*clock, input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateOutDir { index: 1 });
        assert_eq!(runner.calls, 0);
    }
}

/// (T6b・P0・REQ-39。issue #84 PR #238 レビュー) `validation_inputs` の
/// 1 件が [`fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES`] を超える
/// と `ValidationInputTooLarge` として拒否され、runner・scorer のいずれも
/// 呼び出されない（`SearchInput` はデータ契約層を経由しない呼び出し元も
/// 直接組み立てられる公開 API のため、`run_search` 自身が予算・runner・
/// scorer を消費する前に検証する）。
#[test]
fn task18_1_2_validation_input_exceeding_per_record_limit_is_rejected_before_scoring() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let scorer = FakeScorer::new(Arc::clone(&clock), BTreeMap::new());

    let over_limit_len = fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES + 1;
    let oversized_input = vec![0u8; over_limit_len];
    let mut validation_inputs: Vec<&[u8]> = VALIDATION_INPUTS.to_vec();
    validation_inputs[0] = oversized_input.as_slice();

    let gold = validation_gold();
    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &validation_inputs,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let err = run_search(&mut runner, scorer, &*clock, input).unwrap_err();
    assert_eq!(
        err,
        SearchError::ValidationInputTooLarge {
            index: 0,
            size: over_limit_len,
            limit: fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES,
        }
    );
    assert_eq!(runner.calls, 0);
    assert_eq!(probe.lock().expect("probe poisoned").calls.len(), 0);
}

/// (T6c・P0・REQ-39。issue #84 PR #238 レビュー) `validation_inputs` の合計
/// バイト数が [`fandhe_edge_train::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES`]
/// を超えると `ValidationInputTotalBytesExceeded` として拒否され、
/// runner・scorer のいずれも呼び出されない。個々の要素は
/// `MAX_INFER_INPUT_BYTES`（1 件あたりの上限）ちょうどに収まっているため、
/// 1 件あたりの上限チェックだけでは検出できず合計チェックが必要なことを
/// 示す。
#[test]
fn task18_1_2_validation_inputs_exceeding_total_limit_is_rejected_before_scoring() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let scorer = FakeScorer::new(Arc::clone(&clock), BTreeMap::new());

    // 1 件あたりはちょうど上限（超過しない）だが、件数を増やして合計が
    // 上限を超えるようにする。
    let per_record_bytes = fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES;
    let n_records =
        fandhe_edge_train::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES / per_record_bytes + 1;
    let buffer = vec![0u8; per_record_bytes * n_records];
    let validation_inputs: Vec<&[u8]> = buffer.chunks(per_record_bytes).collect();
    let gold: Vec<&str> = (0..n_records).map(|_| "positive").collect();
    let record_ids: Vec<String> = (0..n_records).map(|i| format!("r{i}")).collect();
    let record_id_refs: Vec<&str> = record_ids.iter().map(String::as_str).collect();

    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &record_id_refs,
        validation_inputs: &validation_inputs,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let err = run_search(&mut runner, scorer, &*clock, input).unwrap_err();
    assert_eq!(
        err,
        SearchError::ValidationInputTotalBytesExceeded {
            total: per_record_bytes * n_records,
            limit: fandhe_edge_train::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES,
        }
    );
    assert_eq!(runner.calls, 0);
    assert_eq!(probe.lock().expect("probe poisoned").calls.len(), 0);
}

/// (T7・P1・REQ-27。issue #84 PR #238 レビュー) scorer が gold と異なる件数
/// を返した場合、以前は `SearchError::ScorerOutputMismatch` で探索全体を
/// 打ち切り、それまでの候補の記録を失っていた（非対称の指摘）。record_id
/// 不一致と同じ `ScoringFailed`（候補単位の失敗）として記録し、探索を継続
/// して後続候補を実行することを確認する。
#[test]
fn task18_1_2_scorer_output_length_mismatch_is_recorded_and_search_continues() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 10 },
            RunnerBehavior::Ok { advance_ms: 10 },
        ],
    );
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([
            (
                "c3-a".to_string(),
                Ok(outcomes_with_correct(9)[..9].to_vec()),
            ),
            ("c3-b".to_string(), Ok(outcomes_with_correct(7))),
        ]),
    );

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(record.candidates.len(), 2);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringFailed
    );
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    // 件数不一致の候補で探索が打ち切られず、後続候補（c3-b）が実行・評価
    // されて選定されることを確認する（P1 指摘対応の核心: それまでの候補の
    // 記録を失わない）。
    match &record.candidates[1].result {
        CandidateSearchResult::Evaluated {
            validation_accuracy,
        } => {
            assert_eq!(validation_accuracy.correct, 7);
            assert_eq!(validation_accuracy.total, 10);
        }
        other => panic!("expected Evaluated for c3-b, got {other:?}"),
    }
    assert_eq!(
        record.selection,
        SelectionDecision::Selected {
            candidate_id: "c3-b".to_string(),
            validation_accuracy: fandhe_edge_train::search::ValidationAccuracy {
                correct: 7,
                total: 10,
                value: 0.7,
            },
            rule: "validation_accuracy_desc_then_candidate_order".to_string(),
            tied_candidate_ids: vec!["c3-b".to_string()],
        }
    );
}

/// (T8b・P0・REQ-27・評価の独立性。issue #84 PR #238 レビュー) scorer が
/// 件数は一致するが `record_id` の順序が異なる予測を返した場合、
/// `run_search` は `validation_gold` と誤って突き合わせて正解率を算出せず、
/// `ScoringFailed`（採点エラーと同じ扱い）として選定対象から除外する。
/// 探索全体は中断しない（次候補が宣言されていれば実行される）。
#[test]
fn task18_1_2_scorer_record_id_order_mismatch_excludes_candidate_from_selection() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::Ok { advance_ms: 10 }]);
    // record_id 列を逆順にして返す（件数は validation_record_ids と同じ
    // だが順序が異なる。`run_search` が渡した `record_ids` を無視した実装を
    // 模擬する）。
    let mut reversed_ids: Vec<&str> = VALIDATION_RECORD_IDS.to_vec();
    reversed_ids.reverse();
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([("c3-a".to_string(), Ok(outcomes_with_correct(10)))]),
    )
    .with_record_id_override("c3-a", reversed_ids);

    let gold = validation_gold();
    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    // scorer 自体は `run_search` が渡した正しい `(record_id, input)` の組を
    // 受け取った（呼び出しは行われた。P0・REQ-27: `validation_inputs` が
    // `record_id` と揃った状態で転送されていることの機械照合）。
    let probe = probe.lock().expect("probe poisoned");
    assert_eq!(probe.received_records.len(), 1);
    let expected_records: Vec<(String, Vec<u8>)> = VALIDATION_RECORD_IDS
        .iter()
        .zip(VALIDATION_INPUTS.iter())
        .map(|(&id, &input)| (id.to_string(), input.to_vec()))
        .collect();
    assert_eq!(probe.received_records[0], expected_records);
    assert_eq!(record.candidates.len(), 1);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringFailed
    );
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T8c・P0・REQ-27・評価の独立性。issue #84 PR #238 レビュー) scorer が
/// 件数は一致するが全く別の（`validation_record_ids` に含まれない）
/// `record_id` を返した場合も、`ScoringFailed` として選定対象から除外する。
#[test]
fn task18_1_2_scorer_record_id_foreign_ids_excludes_candidate_from_selection() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::Ok { advance_ms: 10 }]);
    let foreign_ids: Vec<&str> = (0..VALIDATION_LEN).map(|_| "unrelated-record").collect();
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([("c3-a".to_string(), Ok(outcomes_with_correct(10)))]),
    )
    .with_record_id_override("c3-a", foreign_ids);

    let gold = validation_gold();
    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringFailed
    );
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T8・runner のエラー) runner が Err を返すと `SearchError::Candidate`。
#[test]
fn task18_1_2_runner_failure_aborts_search() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::RunnerError]);
    let scorer = FakeScorer::new(Arc::clone(&clock), BTreeMap::new());

    let gold = validation_gold();
    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };
    let err = run_search(&mut runner, scorer, &*clock, input).unwrap_err();
    assert!(matches!(err, SearchError::Candidate { index: 0, .. }));
}

/// (T9・記録の JSON) 選定・候補ごとの記録が期待どおりのキー・値で直列化
/// される。
#[test]
fn task18_1_2_record_serializes_expected_json_shape() {
    let clock = Arc::new(FakeClock::new(1_700_000_000_000));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 10 },
            RunnerBehavior::LimitExceeded {
                advance_ms: 5_000_000,
            },
        ],
    );
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([("c3-a".to_string(), Ok(outcomes_with_correct(9)))]),
    );

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::new(3600).expect("non-zero"),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    let json = serde_json::to_value(&record).expect("serialize record");

    assert_eq!(json["budget_seconds"], serde_json::json!(3600));
    assert_eq!(
        json["per_candidate_policy"],
        serde_json::json!("even_split")
    );
    assert_eq!(json["selection"]["decision"], serde_json::json!("selected"));
    assert_eq!(json["selection"]["candidate_id"], serde_json::json!("c3-a"));
    assert_eq!(
        json["selection"]["validation_accuracy"]["correct"],
        serde_json::json!(9)
    );
    assert_eq!(
        json["selection"]["validation_accuracy"]["total"],
        serde_json::json!(10)
    );
    assert_eq!(
        json["candidates"][0]["result"],
        serde_json::json!("evaluated")
    );
    assert_eq!(
        json["candidates"][1]["result"],
        serde_json::json!("training_not_completed")
    );
    assert_eq!(
        json["candidates"][1]["time"]["status"],
        serde_json::json!("limit_exceeded")
    );
    assert!(json["candidates"][0].get("validation_outcomes").is_none());
}

/// (T10・REQ-27) 評価済み候補の validation 出力アクセサが使え、scorer は
/// 評価済み候補の ID だけを宣言順に受け取る。`validation_gold` を
/// `ValidationScorer` へ渡さないことは trait の署名で構造的に保証される。
#[test]
fn task18_1_2_evaluated_candidate_exposes_validation_outcomes_without_leaking_gold() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 10 },
            RunnerBehavior::LimitExceeded {
                advance_ms: 5_000_000,
            },
        ],
    );
    let expected_outcomes = outcomes_with_correct(7);
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([("c3-a".to_string(), Ok(expected_outcomes.clone()))]),
    );

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(probe.lock().expect("probe poisoned").calls, vec!["c3-a"]);
    assert_eq!(
        record.candidates[0].validation_outcomes(),
        Some(expected_outcomes.as_slice())
    );
    assert_eq!(record.candidates[1].validation_outcomes(), None);
}

/// (T11・P0/P1・REQ-39・issue #84 PR #238 レビュー) 採点（`predict_validation`）
/// の呼び出し中に探索予算全体を使い切った場合、`predict_validation` から
/// 戻った直後（`EvalRecord` の構築・評価器 `evaluate_single_select` の
/// 呼び出しより前）に打ち切り、その候補は `scoring_skipped_budget_exhausted`
/// として記録される（正解率は算出しない。P1 指摘対応: 期限後に評価器という
/// 重い処理を呼び出さない）。選定対象（`evaluated_owned`）から除外され、
/// 1 候補しかない場合 `NoEligibleCandidate` になる（超過後も最後の候補なら
/// `Selected` を返してしまう不具合の再現・修正確認）。`validation_outcomes()`
/// も `None`（#87 の McNemar 検定に使えないことを保証する）。
#[test]
fn task18_1_2_scoring_exceeding_budget_is_excluded_from_selection() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::Ok { advance_ms: 10 }]);
    // 採点呼び出し中に予算（3600 秒 = 3_600_000ms）を大きく超えて時計を
    // 進める（採点自体は完了するが、期限を守れなかった想定。戻り値
    // 自体はあるが、戻った直後の予算確認で評価器を呼ばずに打ち切る）。
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([("c3-a".to_string(), Ok(outcomes_with_correct(9)))]),
    )
    .with_advance("c3-a", 4_000_000);

    let gold = validation_gold();
    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(record.candidates.len(), 1);
    // 採点自体は呼ばれている（`predict_validation` から戻ってきた）が、
    // 評価器（`evaluate_single_select`）は呼ばれず正解率が存在しない
    // ことを `ScoringSkippedBudgetExhausted`（accuracy を持たない）で示す。
    assert_eq!(
        probe.lock().expect("probe poisoned").calls,
        vec!["c3-a".to_string()]
    );
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringSkippedBudgetExhausted
    );
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T12・P0・REQ-39) 採点が予算を超過した候補より後ろに宣言されていた候補は
/// 実行されず、`not_started/budget_exhausted` として記録される（P1 の
/// `drain_remaining_as_not_started` が P0 の超過経路でも呼ばれることの確認）。
#[test]
fn task18_1_2_scoring_exceeding_budget_stops_remaining_candidates() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::Ok { advance_ms: 10 }]);
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([("c3-a".to_string(), Ok(outcomes_with_correct(9)))]),
    )
    .with_advance("c3-a", 4_000_000);

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(runner.calls, 1, "c3-b はスコアリング超過後に実行されない");
    assert_eq!(record.candidates.len(), 2);
    assert_eq!(record.candidates[1].candidate_id, "c3-b");
    assert_eq!(record.candidates[1].elapsed_at_start_ms, None);
    assert_eq!(
        record.candidates[1].result,
        CandidateSearchResult::NotStarted {
            reason: NotStartedReason::BudgetExhausted
        }
    );
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T13・P0・REQ-39) scorer が受け取る `time_limit` は、呼び出し時点で
/// 残っている探索予算全体（`budget - elapsed`）と一致する。
#[test]
fn task18_1_2_scorer_receives_remaining_budget_as_time_limit() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 1_000 },
            RunnerBehavior::Ok { advance_ms: 2_000 },
        ],
    );
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([
            ("c3-a".to_string(), Ok(outcomes_with_correct(6))),
            ("c3-b".to_string(), Ok(outcomes_with_correct(7))),
        ]),
    );

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::new(3_600).expect("non-zero"),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    let probe = probe.lock().expect("probe poisoned");
    assert_eq!(probe.time_limits.len(), 2);
    // budget = 3_600_000ms。c3-a の採点呼び出し時点では 1_000ms 経過。
    assert_eq!(
        probe.time_limits[0],
        Duration::from_millis(3_600_000 - 1_000)
    );
    // c3-b の採点呼び出し時点では c3-a の学習（1_000ms）＋ c3-b の学習
    // （2_000ms）で 3_000ms 経過。
    assert_eq!(
        probe.time_limits[1],
        Duration::from_millis(3_600_000 - 3_000)
    );
}

/// 事前検証: `SearchBudget::new(0)` は `None`（0 秒を表現できない）。
#[test]
fn task18_1_2_search_budget_rejects_zero() {
    assert_eq!(SearchBudget::new(0), None);
}

/// `SearchBudget::default()` が [`fandhe_edge_train::search::DEFAULT_SEARCH_BUDGET_SECONDS`]
/// と一致する。
#[test]
fn task18_1_2_search_budget_default_matches_constant() {
    assert_eq!(
        SearchBudget::default().get(),
        fandhe_edge_train::search::DEFAULT_SEARCH_BUDGET_SECONDS
    );
}

/// REQ-18・TASK-18.1-2・REQ-39（P0 指摘対応。issue #84 PR #238 レビュー）:
/// `SearchBudget::new` は [`fandhe_edge_train::search::MAX_SEARCH_BUDGET_SECONDS`]
/// ちょうどは受理し、1 秒でも超えると `None` を返す（探索全体を極端に長く
/// 実行できる経路を閉じる）。
#[test]
fn task18_1_2_search_budget_rejects_over_max() {
    let max = fandhe_edge_train::search::MAX_SEARCH_BUDGET_SECONDS;
    assert_eq!(SearchBudget::new(max).map(SearchBudget::get), Some(max));
    assert_eq!(SearchBudget::new(max + 1), None);
    assert_eq!(SearchBudget::new(u64::MAX), None);
}

/// (T14・P0/P1・REQ-39・issue #84 PR #238 レビュー) 採点呼び出し中に経過時間が
/// ちょうど探索予算全体（3_600_000ms）に達した場合も「予算到達を合格扱いに
/// しない」（evaluation-contract）に含める。`predict_validation` から戻った
/// 直後の判定を `>`（旧実装）のままにすると本テストは失敗し（`Evaluated`・
/// `Selected` になる）、`>=` への修正で `ScoringSkippedBudgetExhausted`・
/// `NoEligibleCandidate` になることを確認する（P1 指摘対応で判定が
/// 評価器呼び出し前へ移動したため、境界到達時に評価器を呼ばず正解率も
/// 算出しないことも合わせて確認する）。
#[test]
fn task18_1_2_scoring_exceeding_budget_exactly_at_boundary_is_excluded() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::Ok { advance_ms: 10 }]);
    // 学習で 10ms 経過済み。採点呼び出し中に残り全体（3_600_000 - 10ms）を
    // 使い切り、採点終了時点でちょうど budget_ms（3_600_000ms）に一致させる。
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([("c3-a".to_string(), Ok(outcomes_with_correct(9)))]),
    )
    .with_advance("c3-a", 3_600_000 - 10);

    let gold = validation_gold();
    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(record.candidates.len(), 1);
    assert_eq!(
        probe.lock().expect("probe poisoned").calls,
        vec!["c3-a".to_string()]
    );
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringSkippedBudgetExhausted
    );
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T16・P1・REQ-39・issue #84 PR #238 レビュー) `predict_validation` が
/// 成功して戻ってきても、戻った直後の時点で探索予算全体を使い切っていれば、
/// `EvalRecord` の構築・評価器（`evaluate_single_select`）の呼び出しを一切
/// 行わずに打ち切る（採点結果を最後まで評価してから事後に打ち切っていた
/// 旧実装は、最大 `MAX_SEARCH_OUTCOME_CELLS` 件分の評価処理を期限後も
/// 実行してしまい REQ-39「資源の上限」に反していた）。`scorer.calls` に
/// candidate_id が記録されている（採点自体は呼ばれた）一方で、結果に
/// 正解率を持たないことにより評価器が呼ばれていないことを確認する。
#[test]
fn task18_1_2_scoring_returns_but_budget_exhausted_skips_evaluator() {
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 10 },
            RunnerBehavior::Ok { advance_ms: 10 },
        ],
    );
    // c3-a の採点で予算を使い切る。c3-b は宣言順で後ろのため、予算超過後
    // 実行されず not_started になる（`drain_remaining_as_not_started`）。
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([("c3-a".to_string(), Ok(outcomes_with_correct(10)))]),
    )
    .with_advance("c3-a", 3_600_000);

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(
        probe.lock().expect("probe poisoned").calls,
        vec!["c3-a".to_string()]
    );
    assert_eq!(record.candidates.len(), 2);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringSkippedBudgetExhausted
    );
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    assert_eq!(
        record.candidates[1].result,
        CandidateSearchResult::NotStarted {
            reason: NotStartedReason::BudgetExhausted
        }
    );
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T15・P0・REQ-39・issue #84 PR #238 レビュー) 学習だけで探索予算全体を
/// 使い切っていた場合、採点（`predict_validation`）を一切呼び出さずに
/// `ScoringSkippedBudgetExhausted` として記録し、以降の候補は未着手にする。
/// （呼び出し前に残り時間がないと分かっている場合に呼び出し自体を避ける、
/// という P0 指摘の核心部分の確認。post-check の `>=` 修正だけでは
/// 「呼び出さない」ことまでは保証されないため、`scorer.calls` が空である
/// ことを直接検証する）。
#[test]
fn task18_1_2_training_exhausts_budget_skips_scoring_call() {
    let clock = Arc::new(FakeClock::new(0));
    // 学習だけでちょうど探索予算全体（3_600_000ms）を使い切る。
    let mut runner = FakeRunner::new(
        &clock,
        vec![RunnerBehavior::Ok {
            advance_ms: 3_600_000,
        }],
    );
    let scorer = FakeScorer::new(Arc::clone(&clock), BTreeMap::new());

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert!(
        probe.lock().expect("probe poisoned").calls.is_empty(),
        "budget exhausted before scoring must not call predict_validation"
    );
    assert_eq!(
        runner.calls, 1,
        "c3-b はスコアリング前に予算切れのため実行されない"
    );
    assert_eq!(record.candidates.len(), 2);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringSkippedBudgetExhausted
    );
    assert!(record.candidates[0].time.is_some());
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    assert_eq!(record.candidates[1].candidate_id, "c3-b");
    assert_eq!(record.candidates[1].elapsed_at_start_ms, None);
    assert_eq!(
        record.candidates[1].result,
        CandidateSearchResult::NotStarted {
            reason: NotStartedReason::BudgetExhausted
        }
    );
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}
/// (P0・REQ-39・issue #84 PR #238 レビュー) 候補の学習が成功
/// （`TrainOutcome::Ok`）しても、実測時間（`elapsed_ms`）がその候補へ配分
/// した持ち時間（`time_limit_seconds`）以上だった場合は、採点
/// （`predict_validation`）を呼び出さずに `TrainingExceededTimeLimit` として
/// 選定対象から除外する。探索予算 3600 秒を 2 候補へ均等配分（各 1800 秒）
/// し、最初の候補が割当を超えて 2000 秒で成功した場合の再現（指摘本文の
/// シナリオそのもの）。探索全体の予算はまだ残っている（2000 秒 <
/// 3600 秒）ため、2 番目の候補は通常どおり実行・評価され選定される。
#[test]
fn task18_1_2_training_exceeds_own_time_limit_is_excluded_from_selection() {
    let clock = Arc::new(FakeClock::new(0));
    // c3-a: 割当 1800 秒に対し 2000 秒かけて成功する（超過）。
    // c3-b: 残り予算（3600 - 2000 = 1600 秒）内に収まり、評価まで完了する。
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok {
                advance_ms: 2_000_000,
            },
            RunnerBehavior::Ok { advance_ms: 10 },
        ],
    );
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([("c3-b".to_string(), Ok(outcomes_with_correct(9)))]),
    );

    let gold = validation_gold();
    let candidates = vec![
        SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        },
        SearchCandidate {
            candidate_id: "c3-b".to_string(),
            params: candidate_params("out/c3-b", 2),
        },
    ];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(
        probe.lock().expect("probe poisoned").calls,
        vec!["c3-b".to_string()],
        "c3-a is excluded before scoring; c3-b is scored normally"
    );
    assert_eq!(record.candidates.len(), 2);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::TrainingExceededTimeLimit
    );
    assert!(record.candidates[0].time.is_some());
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    match &record.candidates[1].result {
        CandidateSearchResult::Evaluated {
            validation_accuracy,
        } => {
            assert_eq!(validation_accuracy.correct, 9);
            assert_eq!(validation_accuracy.total, 10);
        }
        other => panic!("expected Evaluated for c3-b, got {other:?}"),
    }
    match &record.selection {
        SelectionDecision::Selected {
            candidate_id,
            validation_accuracy,
            tied_candidate_ids,
            ..
        } => {
            assert_eq!(candidate_id, "c3-b");
            assert_eq!(validation_accuracy.correct, 9);
            assert_eq!(validation_accuracy.total, 10);
            assert_eq!(tied_candidate_ids, &vec!["c3-b".to_string()]);
        }
        other => panic!("expected Selected(c3-b), got {other:?}"),
    }
}

/// (P0・REQ-39・issue #84 PR #238 レビュー) 候補の実測時間が割当持ち時間に
/// ちょうど一致した場合（`==`）は超過扱いにしない（単調時計はミリ秒単位で
/// 丸まり、割当時間ぴったりで完了する候補は珍しくないため。`run_search`
/// 実装のコメント参照）。1ms でも超えれば `TrainingExceededTimeLimit` に
/// なることを対比で確認する。
#[test]
fn task18_1_2_training_exactly_at_own_time_limit_is_not_excluded_but_one_ms_over_is() {
    // ちょうど一致（100_000ms）: 超過扱いにせず採点まで進む。
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(
        &clock,
        vec![RunnerBehavior::Ok {
            advance_ms: 100_000,
        }],
    );
    let scorer = FakeScorer::new(
        Arc::clone(&clock),
        BTreeMap::from([("c3-a".to_string(), Ok(outcomes_with_correct(9)))]),
    );

    let gold = validation_gold();
    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: fixed_policy(100),
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert_eq!(
        probe.lock().expect("probe poisoned").calls,
        vec!["c3-a".to_string()],
        "exactly-at-limit training must still be scored"
    );
    assert_eq!(record.candidates.len(), 1);
    assert!(matches!(
        record.candidates[0].result,
        CandidateSearchResult::Evaluated { .. }
    ));

    // 1ms でも超過（100_001ms）: 採点せず `TrainingExceededTimeLimit` になる。
    let clock = Arc::new(FakeClock::new(0));
    let mut runner = FakeRunner::new(
        &clock,
        vec![RunnerBehavior::Ok {
            advance_ms: 100_001,
        }],
    );
    let scorer = FakeScorer::new(Arc::clone(&clock), BTreeMap::new());

    let gold = validation_gold();
    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: &validation_split_record_fixture(),
        candidates,
        budget: SearchBudget::default(),
        policy: fixed_policy(100),
    };

    let probe = scorer.probe();
    let record = run_search(&mut runner, scorer, &*clock, input).expect("search succeeds");
    assert!(
        probe.lock().expect("probe poisoned").calls.is_empty(),
        "over-limit training must not be scored"
    );
    assert_eq!(record.candidates.len(), 1);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::TrainingExceededTimeLimit
    );
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}
