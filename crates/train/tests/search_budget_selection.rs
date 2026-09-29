//! REQ-18・REQ-27・REQ-39（TASK-18.1-2・issue #84。`task18_2_` で始まるテストは
//! TASK-18.2・issue #85 の予算到達の記録・非合格扱い）: 探索予算全体の管理・
//! 複数候補の比較・選定の記録の受け入れ条件を確認する結合テスト
//! （証拠種別: テストハーネス）。
//!
//! `FakeClock`（`Mutex` で内部可変性を持つ仮想時計）・`FakeRunner`
//! （呼び出しごとの所要 ms と結果種別を事前登録した実行器。学習ジョブが学習直後に
//! 返す validation 予測列〔`validation_predictions`〕も、リクエストの
//! `validation_inputs` の `id` から組み立てて返す）で
//! `fandhe_edge_train::search::run_search` を検証する（issue #84 PR #238・
//! 選択肢 2: 採点は学習ジョブの中で行うため、採点用の trait・スレッド・実時間の
//! `sleep` は無い。すべて実時間に依存せず決定的。`.claude/rules/ci.md`）。
//! 実際の子プロセス経路〔`WorkerCandidateRunner`＋`run_train`〕を通す結合テストは
//! `worker_process.rs`。本テストは `crates/train`・`crates/data`・
//! `crates/eval`・`fandhe-edge-core` の範囲に閉じており、`docs/spec` は参照しない。

use std::fs;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use fandhe_edge_data::split::{Groupable, Split, SplitRatios};
use fandhe_edge_data::split_record::{SplitRecord, split_and_record};
use fandhe_edge_eval::metrics::Outcome;
use fandhe_edge_train::request::{Device, TrainRequest, TrainRequestParams};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::search::{
    BudgetReachedScope, CandidateSearchResult, NotStartedReason, SearchBudget, SearchCandidate,
    SearchError, SearchInput, SelectionDecision, run_search,
};
use fandhe_edge_train::time_allotment::{CandidateRunner, Clock, PerCandidatePolicy};

/// 外部入力と同じ作法で扱うための読み込み前サイズ上限（REQ-39）。
const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

const LABEL_ORDER: [&str; 3] = ["positive", "negative", "neutral"];
const VALIDATION_LEN: usize = 10;

/// `validation_gold()` と同じ順・同じ件数の validation レコード識別子
/// （REQ-27。issue #84 PR #238 レビュー）。`run_search` がこれを各候補の
/// 学習リクエストの `validation_inputs` へ `input` と組にして渡すこと、
/// 結果の予測列の `id` 列との突き合わせを専用テストで確認する。
const VALIDATION_RECORD_IDS: [&str; VALIDATION_LEN] =
    ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"];

/// `validation_gold()`・`VALIDATION_RECORD_IDS` と同じ件数・同じ順の byte
/// 入力。各要素を区別できる値にしておき、`run_search` が
/// `SearchInput::validation_inputs` の要素を正しく学習リクエストへ転送する
/// こと（`b""` 等のプレースホルダで置き換えても検出できない偽陰性を防ぐ）を
/// 機械照合できるようにする（P0・REQ-27 指摘対応）。
const VALIDATION_INPUTS: [&[u8]; VALIDATION_LEN] = [
    b"i0", b"i1", b"i2", b"i3", b"i4", b"i5", b"i6", b"i7", b"i8", b"i9",
];

/// [`Groupable`] の最小実装（`fandhe_edge_data::split_record` のテストと
/// 同じ設計）。
struct SplitGroupable {
    id: String,
    group_id: String,
    label: String,
    input: Vec<u8>,
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
    fn input(&self) -> &[u8] {
        &self.input
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
    split_record_for(
        &VALIDATION_RECORD_IDS,
        &VALIDATION_INPUTS,
        &validation_gold(),
    )
}

/// `ids`・`inputs`・`labels`（正解ラベル）を validation split として持つ凍結記録
/// （中身のハッシュを含む）を組み立てる。ID・input・ラベルの一部だけを差し替えた
/// 記録を作って、中身のハッシュの照合を確認するのにも使う。
fn split_record_for(ids: &[&str], inputs: &[&[u8]], labels: &[&str]) -> SplitRecord {
    let records: Vec<SplitGroupable> = ids
        .iter()
        .zip(inputs.iter())
        .zip(labels.iter())
        .enumerate()
        .map(|(i, ((&id, &input), &label))| SplitGroupable {
            id: id.to_string(),
            group_id: format!("g{i}"),
            label: label.to_string(),
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
            input: Vec::new(),
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

/// テストが自由に進められる仮想時計。
///
/// `advance_at_monotonic_call`（[`FakeClock::with_advance_at_monotonic_call`]）を
/// 指定すると、`monotonic()` の n 回目の呼び出しで時計を進める（評価器の呼び出し
/// 「後」に探索予算を使い切るケースを模擬するため。評価器自体は仮想時計を
/// 進めないので、`run_search` の呼び出し順序に合わせて呼び出し回数で指定する。
/// 数え方は使うテストの doc を参照）。
struct FakeClock {
    elapsed: Mutex<Duration>,
    unix_ms: Mutex<u64>,
    monotonic_calls: Mutex<u64>,
    advance_at_monotonic_call: Mutex<Option<(u64, u64)>>,
}

impl FakeClock {
    fn new(start_unix_ms: u64) -> Self {
        Self {
            elapsed: Mutex::new(Duration::ZERO),
            unix_ms: Mutex::new(start_unix_ms),
            monotonic_calls: Mutex::new(0),
            advance_at_monotonic_call: Mutex::new(None),
        }
    }

    /// `monotonic()` の `call_number` 回目（1 始まり）の呼び出しで `ms` だけ進める。
    fn with_advance_at_monotonic_call(self, call_number: u64, ms: u64) -> Self {
        *self
            .advance_at_monotonic_call
            .lock()
            .expect("fake clock mutex poisoned") = Some((call_number, ms));
        self
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
        let call_number = {
            let mut calls = self
                .monotonic_calls
                .lock()
                .expect("fake clock mutex poisoned");
            *calls += 1;
            *calls
        };
        let trigger = *self
            .advance_at_monotonic_call
            .lock()
            .expect("fake clock mutex poisoned");
        if let Some((n, ms)) = trigger
            && n == call_number
        {
            self.advance_ms(ms);
        }
        *self.elapsed.lock().expect("fake clock mutex poisoned")
    }

    fn unix_millis(&self) -> Result<u64, fandhe_edge_train::time_allotment::TimeAllotmentError> {
        Ok(*self.unix_ms.lock().expect("fake clock mutex poisoned"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FakeRunnerError {
    Spawn,
    WallTimeout,
}

impl std::fmt::Display for FakeRunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fake runner error: {self:?}")
    }
}

/// 学習ジョブが学習直後に返す validation 予測列の作り方。
#[derive(Debug, Clone)]
enum Prediction {
    /// リクエストの `validation_inputs` の `id` 順に、この予測を付ける
    /// （件数が少なければ短い列になる）。
    Outcomes(Vec<Outcome>),
    /// `id` 列だけを差し替える（件数は同じでも順序・内容が異なる予測を模擬）。
    WithIds {
        ids: Vec<String>,
        outcomes: Vec<Outcome>,
    },
}

/// 呼び出し 1 回分の学習ジョブの振る舞い（宣言順の候補と 1 対 1 対応）。
#[derive(Debug, Clone)]
enum RunnerBehavior {
    /// 成功。`artifact_dir` は `request` から動的に組み立てる。
    Ok {
        advance_ms: u64,
        prediction: Prediction,
    },
    /// 学習ワーカーが `code` のエラー結果を返す（例: `limit_exceeded`）。
    WorkerError { code: &'static str, advance_ms: u64 },
    /// 実行器が壁時計の締め切りで子プロセスを強制終了した
    /// （`CandidateRunner::is_wall_timeout` が `true`）。
    WallTimeout { advance_ms: u64 },
    /// 実行器自体が失敗する（学習ワーカーの起動失敗等）。
    RunnerError,
}

fn ok(advance_ms: u64, outcomes: Vec<Outcome>) -> RunnerBehavior {
    RunnerBehavior::Ok {
        advance_ms,
        prediction: Prediction::Outcomes(outcomes),
    }
}

/// 実行器が受け取ったリクエストの記録（REQ-27・REQ-39 の照合用）。
#[derive(Debug, Clone)]
struct ReceivedRequest {
    time_limit_seconds: u32,
    /// `validation_inputs` の `(id, input)`。
    validation_inputs: Vec<(String, String)>,
    /// 実際に学習ワーカーへ渡る JSON の `validation_inputs`。
    wire_validation_inputs: serde_json::Value,
}

/// 事前登録した振る舞いを順番に返す実行器。呼び出し回数と受け取ったリクエストを
/// 記録する。
struct FakeRunner<'a> {
    clock: &'a FakeClock,
    behaviors: Vec<RunnerBehavior>,
    calls: usize,
    received: Vec<ReceivedRequest>,
}

impl<'a> FakeRunner<'a> {
    fn new(clock: &'a FakeClock, behaviors: Vec<RunnerBehavior>) -> Self {
        Self {
            clock,
            behaviors,
            calls: 0,
            received: Vec::new(),
        }
    }
}

fn outcome_to_json(id: &str, outcome: &Outcome) -> serde_json::Value {
    match outcome {
        Outcome::Label(label) => {
            serde_json::json!({"id": id, "status": "ok", "predicted_label": label})
        }
        Outcome::Abstain => {
            serde_json::json!({"id": id, "status": "abstain", "predicted_label": null})
        }
        Outcome::Invalid | Outcome::Error => {
            serde_json::json!({"id": id, "status": "error", "predicted_label": null})
        }
    }
}

/// `result_ok.json` fixture の `artifact_dir` を `request` の `root`／
/// `out_dir` から組み立て直し、`prediction` から `validation_predictions` を
/// 付けた 1 行 JSON にする。
fn ok_stdout_for_request(request: &TrainRequest, prediction: &Prediction) -> String {
    let mut value = load_fixture_value("result_ok.json");
    let artifact_dir = format!("{}/{}", request.root(), request.out_dir());
    value["artifact_dir"] = serde_json::Value::from(artifact_dir);
    let request_ids: Vec<&str> = request
        .validation_inputs()
        .expect("search must attach validation inputs")
        .iter()
        .map(|v| v.id())
        .collect();
    let predictions: Vec<serde_json::Value> = match prediction {
        Prediction::Outcomes(outcomes) => request_ids
            .iter()
            .zip(outcomes.iter())
            .map(|(id, outcome)| outcome_to_json(id, outcome))
            .collect(),
        Prediction::WithIds { ids, outcomes } => ids
            .iter()
            .zip(outcomes.iter())
            .map(|(id, outcome)| outcome_to_json(id, outcome))
            .collect(),
    };
    value["validation_predictions"] = serde_json::Value::Array(predictions);
    serde_json::to_string(&value).expect("serialize fixture-derived stdout")
}

impl CandidateRunner for FakeRunner<'_> {
    type Error = FakeRunnerError;

    fn run(&mut self, request: &TrainRequest) -> Result<TrainOutcome, Self::Error> {
        let index = self.calls;
        self.calls += 1;
        let wire: serde_json::Value =
            serde_json::from_slice(&request.to_json_vec().expect("request must serialize"))
                .expect("request json");
        self.received.push(ReceivedRequest {
            time_limit_seconds: request.time_limit_seconds(),
            validation_inputs: request
                .validation_inputs()
                .expect("search must attach validation inputs")
                .iter()
                .map(|v| (v.id().to_string(), v.input().to_string()))
                .collect(),
            wire_validation_inputs: wire["validation_inputs"].clone(),
        });
        let behavior = self
            .behaviors
            .get(index)
            .cloned()
            .unwrap_or_else(|| panic!("unexpected extra runner call at index {index}"));
        match behavior {
            RunnerBehavior::Ok {
                advance_ms,
                prediction,
            } => {
                self.clock.advance_ms(advance_ms);
                let stdout = ok_stdout_for_request(request, &prediction);
                TrainOutcome::from_worker_stdout(stdout.as_bytes(), request)
                    .map_err(|_| FakeRunnerError::Spawn)
            }
            RunnerBehavior::WorkerError { code, advance_ms } => {
                self.clock.advance_ms(advance_ms);
                let stdout = format!(r#"{{"status":"error","code":"{code}","message":"m"}}"#);
                TrainOutcome::from_worker_stdout(stdout.as_bytes(), request)
                    .map_err(|_| FakeRunnerError::Spawn)
            }
            RunnerBehavior::WallTimeout { advance_ms } => {
                self.clock.advance_ms(advance_ms);
                Err(FakeRunnerError::WallTimeout)
            }
            RunnerBehavior::RunnerError => Err(FakeRunnerError::Spawn),
        }
    }

    fn is_wall_timeout(error: &Self::Error) -> bool {
        matches!(error, FakeRunnerError::WallTimeout)
    }
}

fn fixed_policy(seconds: u32) -> PerCandidatePolicy {
    PerCandidatePolicy::Fixed(NonZeroU32::new(seconds).expect("non-zero"))
}

/// 宣言順の候補を作る（`out_dir` は `out/<id>`、seed は 1 始まりの連番）。
fn candidates(ids: &[&str]) -> Vec<SearchCandidate> {
    ids.iter()
        .enumerate()
        .map(|(i, id)| SearchCandidate {
            candidate_id: (*id).to_string(),
            params: candidate_params(&format!("out/{id}"), (i + 1) as u32),
        })
        .collect()
}

fn input_for<'a>(
    gold: &'a [&'a str],
    split: &'a SplitRecord,
    candidates: Vec<SearchCandidate>,
    budget: SearchBudget,
    policy: PerCandidatePolicy,
) -> SearchInput<'a> {
    SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: gold,
        validation_record_ids: &VALIDATION_RECORD_IDS,
        validation_inputs: &VALIDATION_INPUTS,
        validation_split_record: split,
        candidates,
        budget,
        policy,
    }
}

/// 既定予算・`EvenSplit` で `ids` の候補を探索する入力。
fn default_input<'a>(gold: &'a [&'a str], split: &'a SplitRecord, ids: &[&str]) -> SearchInput<'a> {
    input_for(
        gold,
        split,
        candidates(ids),
        SearchBudget::default(),
        PerCandidatePolicy::EvenSplit,
    )
}

fn selected_id(record: &fandhe_edge_train::search::SearchRecord) -> &str {
    match &record.selection {
        SelectionDecision::Selected { candidate_id, .. } => candidate_id,
        other => panic!("expected Selected, got {other:?}"),
    }
}

/// (T1・主受入) 既定予算・`EvenSplit`・3 候補。正解数 7/10・9/10・8/10 の
/// うち最高正解率の候補（9/10）が選ばれる。あわせて、学習ワーカーへ渡る
/// リクエストの `validation_inputs` が `id`・`input` だけ（正解ラベルを持たない。
/// REQ-27）で、`SearchInput` の値を全候補へ順序どおり転送していることを確認する。
#[test]
fn task18_1_2_default_budget_selects_highest_accuracy_candidate() {
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            ok(1_000, outcomes_with_correct(7)),
            ok(1_000, outcomes_with_correct(9)),
            ok(1_000, outcomes_with_correct(8)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let input = default_input(&gold, &split, &["c3-a", "c3-b", "c3-c"]);

    let record = run_search(&mut runner, &clock, input).expect("search succeeds");

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
    let expected_inputs: Vec<(String, String)> = VALIDATION_RECORD_IDS
        .iter()
        .zip(VALIDATION_INPUTS.iter())
        .map(|(&id, &input)| {
            (
                id.to_string(),
                String::from_utf8(input.to_vec()).expect("utf8"),
            )
        })
        .collect();
    let expected_wire: Vec<serde_json::Value> = VALIDATION_RECORD_IDS
        .iter()
        .zip(VALIDATION_INPUTS.iter())
        .map(|(&id, &input)| {
            serde_json::json!({"id": id, "input": String::from_utf8(input.to_vec()).expect("utf8")})
        })
        .collect();
    for (received, expected_limit) in runner.received.iter().zip(expected_time_limits) {
        assert_eq!(received.time_limit_seconds, expected_limit);
        assert_eq!(received.validation_inputs, expected_inputs);
        // 学習ワーカーへ渡る JSON の要素は `id`・`input` の 2 キーだけ。
        assert_eq!(
            received.wire_validation_inputs,
            serde_json::Value::Array(expected_wire.clone())
        );
    }
}

/// (T2・予算の消費) 予算 100 秒・`Fixed(50)`・4 候補（各 50,000ms）。
/// c3-a は評価済み。c3-b は学習ジョブだけでちょうど探索予算全体を使い切るため
/// 評価器を呼ばず `scoring_skipped_budget_exhausted` になる（P0 指摘対応・
/// REQ-39・issue #84 PR #238 レビュー）。c3-c・c3-d は c3-b の時点で実行順が
/// 一度も回ってこず、どちらも `elapsed_at_start_ms: None`・
/// `not_started/budget_exhausted` になる（P1 指摘対応・REQ-18「候補ごとの選定
/// 記録」。4 候補目は「予算切れ候補の直後だけでなく、さらにその後ろの候補」も
/// 記録されることを検証するため）。
#[test]
fn task18_1_2_budget_exhaustion_marks_remaining_candidates_not_started() {
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            ok(50_000, outcomes_with_correct(6)),
            ok(50_000, outcomes_with_correct(7)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let input = input_for(
        &gold,
        &split,
        candidates(&["c3-a", "c3-b", "c3-c", "c3-d"]),
        SearchBudget::new(100).expect("non-zero"),
        fixed_policy(50),
    );

    let record = run_search(&mut runner, &clock, input).expect("search succeeds");

    assert_eq!(runner.calls, 2);
    assert_eq!(record.total_elapsed_ms, 100_000);
    assert_eq!(record.candidates.len(), 4);
    assert!(matches!(
        record.candidates[0].result,
        CandidateSearchResult::Evaluated { .. }
    ));
    assert_eq!(
        record.candidates[1].result,
        CandidateSearchResult::ScoringSkippedBudgetExhausted
    );
    assert!(record.candidates[1].time.is_some());
    assert_eq!(record.candidates[1].validation_outcomes(), None);
    for (index, id) in [(2, "c3-c"), (3, "c3-d")] {
        assert_eq!(record.candidates[index].candidate_id, id);
        assert_eq!(record.candidates[index].elapsed_at_start_ms, None);
        assert_eq!(record.candidates[index].time, None);
        assert_eq!(
            record.candidates[index].result,
            CandidateSearchResult::NotStarted {
                reason: NotStartedReason::BudgetExhausted
            }
        );
    }
    // c3-b が選定対象から除外されるため、唯一評価済みの c3-a が選ばれる。
    assert_eq!(selected_id(&record), "c3-a");
}

/// (T3・同率) 8/10 と 8/10 のとき先の候補を選び、`tied_candidate_ids` に
/// 両方が入る。
#[test]
fn task18_1_2_tie_breaks_to_first_declared_candidate() {
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            ok(100, outcomes_with_correct(8)),
            ok(100, outcomes_with_correct(8)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
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

/// (T4・対象なし) 全候補が `limit_exceeded`（経過時間は持ち時間に届かない
/// ため時間切れ扱いにならない）のとき `NoEligibleCandidate`。探索は次候補へ
/// 進み、両方とも `training_not_completed` になる。
#[test]
fn task18_1_2_no_eligible_candidate_when_all_candidates_fail() {
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::WorkerError {
                code: "limit_exceeded",
                advance_ms: 10,
            },
            RunnerBehavior::WorkerError {
                code: "limit_exceeded",
                advance_ms: 10,
            },
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
    assert_eq!(runner.calls, 2);
    for entry in &record.candidates {
        assert_eq!(entry.result, CandidateSearchResult::TrainingNotCompleted);
        let time = entry.time.as_ref().expect("candidate must have run");
        assert!(matches!(
            time.status(),
            fandhe_edge_train::time_allotment::CandidateTimeStatus::LimitExceeded { .. }
        ));
    }
}

/// (T5・学習失敗) 1 候補が学習ワーカーのエラー（`training_diverged`）で終わると
/// その候補は `training_not_completed` になり、探索は続いて残りの候補から
/// 選定される。
#[test]
fn task18_1_2_training_failure_is_recorded_and_search_continues() {
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::WorkerError {
                code: "training_diverged",
                advance_ms: 100,
            },
            ok(100, outcomes_with_correct(6)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::TrainingNotCompleted
    );
    assert_eq!(selected_id(&record), "c3-b");
}

/// (T5a・P1・REQ-18・REQ-39) 学習失敗で探索予算全体を使い切った場合、次の候補は
/// 実行されず `not_started/budget_exhausted` として記録される
/// （予算超過後の学習を防ぐ。`allot` が `Exhausted` を返す）。
#[test]
fn task18_1_2_training_failure_that_exhausts_budget_stops_remaining_candidates() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![RunnerBehavior::WorkerError {
            code: "training_diverged",
            advance_ms: 4_000_000,
        }],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert_eq!(runner.calls, 1, "c3-b は予算超過後のため実行されない");
    assert_eq!(record.candidates.len(), 2);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::TrainingNotCompleted
    );
    assert_eq!(record.candidates[1].candidate_id, "c3-b");
    assert_eq!(
        record.candidates[1].result,
        CandidateSearchResult::NotStarted {
            reason: NotStartedReason::BudgetExhausted
        }
    );
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T5b・P0・REQ-17・REQ-27) `validation_record_ids` から再計算したハッシュが、
/// `validation_split_record`（凍結済み validation split の記録）の
/// `validation` split のハッシュと一致しない場合、`ValidationSplitHashMismatch`
/// として拒否され、runner は 1 回も呼び出されない（学習を始める前の
/// fail-closed な事前検証。採点は学習ジョブの中で行うため、この検査が
/// 学習ジョブ全体の前提になる）。
#[test]
fn task18_1_2_validation_split_hash_mismatch_is_rejected_before_training() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let gold = validation_gold();
    let mismatched_record = mismatched_split_record_fixture();
    let err = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &mismatched_record, &["c3-a"]),
    )
    .unwrap_err();
    assert_eq!(err, SearchError::ValidationSplitHashMismatch);
    assert_eq!(err.exit_code().code(), 64);
    assert_eq!(runner.calls, 0);
}

/// (T5c・P0/P1・REQ-39・issue #84 PR #238) 実行器が壁時計の締め切りで子プロセスを
/// 強制終了した場合（`CandidateRunner::is_wall_timeout`）、探索全体の失敗
/// （`SearchError`）ではなく、その候補を `training_timed_out` として記録する。
/// **候補単位の時間切れと探索全体の予算切れは別**: 探索予算 3600 秒・2 候補
/// （各 1800 秒）で、1 つ目が 1800 秒で時間切れになっても、残り 1800 秒で
/// 2 つ目が実行・評価され選定される（P1 指摘の回帰テスト）。
#[test]
fn task18_1_2_candidate_timeout_does_not_stop_remaining_candidates_while_budget_remains() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::WallTimeout {
                advance_ms: 1_800_000,
            },
            ok(1_000, outcomes_with_correct(9)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        input_for(
            &gold,
            &split,
            candidates(&["c3-a", "c3-b"]),
            SearchBudget::new(3600).expect("non-zero"),
            PerCandidatePolicy::EvenSplit,
        ),
    )
    .expect("search succeeds even though a candidate timed out");
    assert_eq!(runner.calls, 2, "c3-b は予算が残っているため実行される");
    assert_eq!(record.candidates.len(), 2);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::TrainingTimedOut
    );
    assert_eq!(record.candidates[0].time, None);
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    assert!(matches!(
        record.candidates[1].result,
        CandidateSearchResult::Evaluated { .. }
    ));
    // c3-b の持ち時間は、時間切れで消費した 1800 秒を除いた残り予算で決まる。
    assert_eq!(runner.received[1].time_limit_seconds, 1800);
    assert_eq!(selected_id(&record), "c3-b");
}

/// (T5c2・P1・REQ-39) 時間切れで探索予算全体を使い切った場合だけ、残りの候補は
/// `not_started/budget_exhausted` になる（候補単位の時間切れではなく予算切れが
/// 理由）。
#[test]
fn task18_1_2_candidate_timeout_that_exhausts_budget_marks_remaining_not_started() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            ok(100, outcomes_with_correct(9)),
            RunnerBehavior::WallTimeout {
                advance_ms: 4_000_000,
            },
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b", "c3-c"]),
    )
    .expect("search succeeds even though a candidate timed out");
    assert_eq!(runner.calls, 2, "c3-c は予算切れ後に実行されない");
    assert!(matches!(
        record.candidates[0].result,
        CandidateSearchResult::Evaluated { .. }
    ));
    assert_eq!(
        record.candidates[1].result,
        CandidateSearchResult::TrainingTimedOut
    );
    assert_eq!(record.candidates[2].candidate_id, "c3-c");
    assert_eq!(
        record.candidates[2].result,
        CandidateSearchResult::NotStarted {
            reason: NotStartedReason::BudgetExhausted
        }
    );
    // 時間切れの候補は選定対象外。評価済みの c3-a が選ばれる。
    assert_eq!(selected_id(&record), "c3-a");
}

/// (T5d・P0/P1・REQ-39・選択肢 2) 学習ワーカー自身が `limit_exceeded` を報告し、
/// Rust 側で測った経過時間が持ち時間に達していた場合（学習後・予測の最初の
/// 資源検査で持ち時間超過が検出された場合など）は、実行器の強制終了と同じ
/// `training_timed_out` として扱うが、探索予算が残っていれば次の候補へ進む
/// （`Fixed(10)` の持ち時間 10 秒に対し経過 12 秒）。
#[test]
fn task18_1_2_worker_limit_exceeded_after_time_limit_is_candidate_timeout_and_search_continues() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::WorkerError {
                code: "limit_exceeded",
                advance_ms: 12_000,
            },
            ok(100, outcomes_with_correct(8)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        input_for(
            &gold,
            &split,
            candidates(&["c3-a", "c3-b"]),
            SearchBudget::default(),
            fixed_policy(10),
        ),
    )
    .expect("search succeeds");
    assert_eq!(runner.calls, 2);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::TrainingTimedOut
    );
    assert!(record.candidates[0].time.is_some());
    assert!(matches!(
        record.candidates[1].result,
        CandidateSearchResult::Evaluated { .. }
    ));
    assert_eq!(selected_id(&record), "c3-b");
}

/// (T5e・REQ-39・選択肢 2) `limit_exceeded` でも経過時間が持ち時間に達して
/// いなければ（RSS 等の他の資源上限）時間切れ扱いにせず、通常の学習失敗
/// （`training_not_completed`）として次候補へ進む（1 候補の資源超過で探索全体を
/// 打ち切らない）。
#[test]
fn task18_1_2_worker_limit_exceeded_before_time_limit_is_plain_failure() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::WorkerError {
                code: "limit_exceeded",
                advance_ms: 2_000,
            },
            ok(100, outcomes_with_correct(8)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        input_for(
            &gold,
            &split,
            candidates(&["c3-a", "c3-b"]),
            SearchBudget::default(),
            fixed_policy(10),
        ),
    )
    .expect("search succeeds");
    assert_eq!(runner.calls, 2);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::TrainingNotCompleted
    );
    assert_eq!(selected_id(&record), "c3-b");
}

/// (T6・事前検証) ID 重複・`label_order` の不一致・gold の未知ラベル・
/// `out_dir` の重複はそれぞれ Err になり、runner の呼び出しは 0 回。
#[test]
fn task18_1_2_precondition_violations_do_not_consume_budget() {
    let gold = validation_gold();
    let split = validation_split_record_fixture();

    // ID 重複。
    {
        let clock = FakeClock::new(0);
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let mut cands = candidates(&["c3-a", "c3-a"]);
        cands[1].params = candidate_params("out/c3-b", 2);
        let input = input_for(
            &gold,
            &split,
            cands,
            SearchBudget::default(),
            PerCandidatePolicy::EvenSplit,
        );
        let err = run_search(&mut runner, &clock, input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateCandidateId { index: 1 });
        assert_eq!(runner.calls, 0);
    }

    // label_order の不一致。
    {
        let clock = FakeClock::new(0);
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let mut cands = candidates(&["c3-a"]);
        cands[0].params.label_order = vec!["negative".to_string(), "positive".to_string()];
        let input = input_for(
            &gold,
            &split,
            cands,
            SearchBudget::default(),
            PerCandidatePolicy::EvenSplit,
        );
        let err = run_search(&mut runner, &clock, input).unwrap_err();
        assert_eq!(err, SearchError::LabelOrderMismatch { index: 0 });
        assert_eq!(runner.calls, 0);
    }

    // gold の未知ラベル。
    {
        let clock = FakeClock::new(0);
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let bad_gold = ["unknown_label"];
        let input = SearchInput {
            label_order: &LABEL_ORDER,
            validation_gold: &bad_gold,
            validation_record_ids: &VALIDATION_RECORD_IDS[..1],
            validation_inputs: &VALIDATION_INPUTS[..1],
            validation_split_record: &split,
            candidates: candidates(&["c3-a"]),
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = run_search(&mut runner, &clock, input).unwrap_err();
        assert_eq!(err, SearchError::UnknownValidationGold { index: 0 });
        assert_eq!(runner.calls, 0);
    }

    // out_dir の重複。
    {
        let clock = FakeClock::new(0);
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let mut cands = candidates(&["c3-a", "c3-b"]);
        cands[0].params = candidate_params("out/same", 1);
        cands[1].params = candidate_params("out/same", 2);
        let input = input_for(
            &gold,
            &split,
            cands,
            SearchBudget::default(),
            PerCandidatePolicy::EvenSplit,
        );
        let err = run_search(&mut runner, &clock, input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateOutDir { index: 1 });
        assert_eq!(runner.calls, 0);
    }
}

/// (T6b・P0・REQ-39) `validation_inputs` の 1 件が
/// [`fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES`] を超えると
/// `ValidationInputTooLarge` として拒否され、runner は呼び出されない
/// （`SearchInput` は公開 API のため、`run_search` 自身が学習を始める前に検証する）。
#[test]
fn task18_1_2_validation_input_exceeding_per_record_limit_is_rejected_before_training() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let over_limit_len = fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES + 1;
    let oversized_input = vec![b'a'; over_limit_len];
    let mut validation_inputs: Vec<&[u8]> = VALIDATION_INPUTS.to_vec();
    validation_inputs[0] = oversized_input.as_slice();

    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let mut input = default_input(&gold, &split, &["c3-a"]);
    input.validation_inputs = &validation_inputs;
    let err = run_search(&mut runner, &clock, input).unwrap_err();
    assert_eq!(
        err,
        SearchError::ValidationInputTooLarge {
            index: 0,
            size: over_limit_len,
            limit: fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES,
        }
    );
    assert_eq!(runner.calls, 0);
}

/// (T6c・P0・REQ-39) `validation_inputs` の合計バイト数が
/// [`fandhe_edge_train::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES`] を超えると
/// `ValidationInputTotalBytesExceeded` として拒否され、runner は呼び出されない。
/// 個々の要素は `MAX_INFER_INPUT_BYTES` ちょうどに収まっているため、1 件あたりの
/// 上限チェックだけでは検出できず合計チェックが必要なことを示す。
#[test]
fn task18_1_2_validation_inputs_exceeding_total_limit_is_rejected_before_training() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let per_record_bytes = fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES;
    let n_records =
        fandhe_edge_train::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES / per_record_bytes + 1;
    let buffer = vec![b'a'; per_record_bytes * n_records];
    let validation_inputs: Vec<&[u8]> = buffer.chunks(per_record_bytes).collect();
    let gold: Vec<&str> = (0..n_records).map(|_| "positive").collect();
    let record_ids: Vec<String> = (0..n_records).map(|i| format!("r{i}")).collect();
    let record_id_refs: Vec<&str> = record_ids.iter().map(String::as_str).collect();

    let split = validation_split_record_fixture();
    let mut input = default_input(&gold, &split, &["c3-a"]);
    input.validation_record_ids = &record_id_refs;
    input.validation_inputs = &validation_inputs;
    let err = run_search(&mut runner, &clock, input).unwrap_err();
    assert_eq!(
        err,
        SearchError::ValidationInputTotalBytesExceeded {
            total: per_record_bytes * n_records,
            limit: fandhe_edge_train::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES,
        }
    );
    assert_eq!(runner.calls, 0);
}

/// (T6d・REQ-27・REQ-39・選択肢 2) UTF-8 でない validation 入力は、子プロセスを
/// 起動する前に `ValidationInputNotUtf8`（`invalid_input`）で拒否される
/// （学習リクエスト JSON の文字列に載せられないため）。
#[test]
fn task18_1_2_non_utf8_validation_input_is_rejected_before_training() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let invalid_utf8: &[u8] = &[0xff, 0xfe, 0x00];
    let mut validation_inputs: Vec<&[u8]> = VALIDATION_INPUTS.to_vec();
    validation_inputs[3] = invalid_utf8;

    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let mut input = default_input(&gold, &split, &["c3-a"]);
    input.validation_inputs = &validation_inputs;
    let err = run_search(&mut runner, &clock, input).unwrap_err();
    assert_eq!(err, SearchError::ValidationInputNotUtf8 { index: 3 });
    assert_eq!(runner.calls, 0);
    // `invalid_input`（REQ-21 の 64）へ写る。
    assert_eq!(err.reason_code(), "invalid_input");
    assert_eq!(err.exit_code().code(), 64);
}

/// (T6e・REQ-39・選択肢 2) 1 件あたりの上限（`MAX_INFER_INPUT_BYTES` = 1 MiB）
/// ちょうどの入力は、学習リクエスト全体の上限（`MAX_REQUEST_BYTES` = 1 MiB）を
/// 超えるため、学習を始める前に `InvalidRequest`（`TooLarge`）で拒否される
/// （実効上限は `MAX_REQUEST_BYTES`。`limits::MAX_VALIDATION_INPUT_TOTAL_BYTES` の
/// doc 参照）。
#[test]
fn task18_1_2_validation_inputs_exceeding_request_size_are_rejected_before_training() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let max_each = vec![b'a'; fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES];
    let mut validation_inputs: Vec<&[u8]> = VALIDATION_INPUTS.to_vec();
    validation_inputs[0] = max_each.as_slice();

    let gold = validation_gold();
    // 凍結記録も同じ（巨大な）入力で作り、中身のハッシュは一致させる。
    let split = split_record_for(
        &VALIDATION_RECORD_IDS,
        &validation_inputs,
        &validation_gold(),
    );
    let mut input = default_input(&gold, &split, &["c3-a"]);
    input.validation_inputs = &validation_inputs;
    let err = run_search(&mut runner, &clock, input).unwrap_err();
    assert!(
        matches!(
            err,
            SearchError::InvalidRequest {
                index: 0,
                source: fandhe_edge_train::error::TrainRequestError::TooLarge { .. }
            }
        ),
        "unexpected error: {err:?}"
    );
    assert_eq!(err.reason_code(), "limit_exceeded");
    assert_eq!(err.exit_code().code(), 20);
    assert_eq!(runner.calls, 0);
}

/// (issue #255・REQ-39) 候補の `config` が直列化後 `MAX_REQUEST_BYTES` を超える場合、
/// 子プロセスを起動する前に `ConfigTooLarge`（`limit_exceeded`・20）で拒否される。
#[test]
fn issue255_oversized_candidate_config_is_rejected_before_training() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let mut input = default_input(&gold, &split, &["c3-a"]);
    let mut config = serde_json::Map::new();
    config.insert(
        "k".to_string(),
        serde_json::Value::String("a".repeat(fandhe_edge_train::limits::MAX_REQUEST_BYTES - 7)),
    );
    input.candidates[0].params.config = config;
    let err = run_search(&mut runner, &clock, input).unwrap_err();
    assert!(
        matches!(
            err,
            SearchError::InvalidRequest {
                index: 0,
                source: fandhe_edge_train::error::TrainRequestError::ConfigTooLarge { .. }
            }
        ),
        "unexpected error: {err:?}"
    );
    assert_eq!(err.reason_code(), "limit_exceeded");
    assert_eq!(err.exit_code().code(), 20);
    assert_eq!(runner.calls, 0);
}

/// (T7・P1・REQ-27) 予測列の件数が gold と異なる場合、その候補は `scoring_failed`
/// （候補単位の失敗）として記録され、探索は継続して後続候補を実行・選定する
/// （それまでの候補の記録を失わない）。
#[test]
fn task18_1_2_prediction_length_mismatch_is_recorded_and_search_continues() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            ok(10, outcomes_with_correct(9)[..9].to_vec()),
            ok(10, outcomes_with_correct(7)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert_eq!(record.candidates.len(), 2);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringFailed
    );
    assert_eq!(record.candidates[0].validation_outcomes(), None);
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

/// (T7b・Cursor 指摘対応・REQ-27。issue #84 PR #238 レビュー) 予測列が入力より
/// **多い**場合も、少ない場合（T7）・`id` が合わない場合（T8b・T8c）と同じく候補単位の
/// `scoring_failed` になり、他の候補の記録が残って探索が続く（以前は結果の解析が
/// 結果全体を拒否し、探索全体が `SearchError` で止まっていた）。
#[test]
fn task18_1_2_extra_prediction_rows_fail_only_that_candidate_and_search_continues() {
    let clock = FakeClock::new(0);
    let mut ids: Vec<String> = VALIDATION_RECORD_IDS
        .iter()
        .map(|s| s.to_string())
        .collect();
    ids.push("extra-row".to_string());
    let mut outcomes = outcomes_with_correct(10);
    outcomes.push(Outcome::Label("positive".to_string()));
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok {
                advance_ms: 10,
                prediction: Prediction::WithIds { ids, outcomes },
            },
            ok(10, outcomes_with_correct(7)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("extra rows must not abort the whole search");
    assert_eq!(runner.calls, 2, "c3-b は c3-a の後も実行される");
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringFailed
    );
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    assert!(matches!(
        record.candidates[1].result,
        CandidateSearchResult::Evaluated { .. }
    ));
    assert_eq!(selected_id(&record), "c3-b");
}

/// (T8b・P0・REQ-27・評価の独立性) 予測列が件数は一致するが `id` の順序が異なる
/// 場合、`run_search` は `validation_gold` と誤って突き合わせて正解率を算出せず、
/// `scoring_failed` として選定対象から除外する。探索全体は中断しない。
#[test]
fn task18_1_2_prediction_id_order_mismatch_excludes_candidate_from_selection() {
    let clock = FakeClock::new(0);
    let mut reversed_ids: Vec<String> = VALIDATION_RECORD_IDS
        .iter()
        .map(|s| s.to_string())
        .collect();
    reversed_ids.reverse();
    let mut runner = FakeRunner::new(
        &clock,
        vec![RunnerBehavior::Ok {
            advance_ms: 10,
            prediction: Prediction::WithIds {
                ids: reversed_ids,
                outcomes: outcomes_with_correct(10),
            },
        }],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(&mut runner, &clock, default_input(&gold, &split, &["c3-a"]))
        .expect("search succeeds");
    assert_eq!(record.candidates.len(), 1);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringFailed
    );
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T8c・P0・REQ-27・評価の独立性) 予測列が件数は一致するが全く別の
/// （`validation_record_ids` に含まれない）`id` を返した場合も、`scoring_failed`
/// として選定対象から除外する。
#[test]
fn task18_1_2_prediction_foreign_ids_exclude_candidate_from_selection() {
    let clock = FakeClock::new(0);
    let foreign_ids: Vec<String> = (0..VALIDATION_LEN)
        .map(|i| format!("unrelated-record-{i}"))
        .collect();
    let mut runner = FakeRunner::new(
        &clock,
        vec![RunnerBehavior::Ok {
            advance_ms: 10,
            prediction: Prediction::WithIds {
                ids: foreign_ids,
                outcomes: outcomes_with_correct(10),
            },
        }],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(&mut runner, &clock, default_input(&gold, &split, &["c3-a"]))
        .expect("search succeeds");
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringFailed
    );
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T8・runner のエラー) runner が時間切れ以外のエラーを返すと
/// `SearchError::Candidate`。
#[test]
fn task18_1_2_runner_failure_aborts_search() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::RunnerError]);
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let err = run_search(&mut runner, &clock, default_input(&gold, &split, &["c3-a"])).unwrap_err();
    assert!(matches!(err, SearchError::Candidate { index: 0, .. }));
    assert_eq!(err.exit_code().code(), 70);
}

/// (T9・記録の JSON) 選定・候補ごとの記録が期待どおりのキー・値で直列化される
/// （`limit_exceeded` の経過時間が持ち時間に届かない候補は
/// `training_not_completed`）。
#[test]
fn task18_1_2_record_serializes_expected_json_shape() {
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            ok(10, outcomes_with_correct(9)),
            RunnerBehavior::WorkerError {
                code: "limit_exceeded",
                advance_ms: 5_000,
            },
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let input = input_for(
        &gold,
        &split,
        candidates(&["c3-a", "c3-b"]),
        SearchBudget::new(3600).expect("non-zero"),
        PerCandidatePolicy::EvenSplit,
    );
    let record = run_search(&mut runner, &clock, input).expect("search succeeds");
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

/// (T9b・選択肢 2) 時間切れの候補の記録は `training_timed_out`（`time` は
/// 実行器の強制終了のため `null`）として直列化され、探索予算が残っていれば次の
/// 候補が実行される。
#[test]
fn task18_1_2_timed_out_candidate_serializes_as_training_timed_out() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::WallTimeout { advance_ms: 1_000 },
            ok(10, outcomes_with_correct(9)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    let json = serde_json::to_value(&record).expect("serialize record");
    assert_eq!(
        json["candidates"][0]["result"],
        serde_json::json!("training_timed_out")
    );
    assert_eq!(json["candidates"][0]["time"], serde_json::Value::Null);
    assert_eq!(
        json["candidates"][1]["result"],
        serde_json::json!("evaluated")
    );
    assert_eq!(json["selection"]["candidate_id"], serde_json::json!("c3-b"));
}

/// (T10・REQ-27) 評価済み候補の validation 出力アクセサが使え、失敗した候補では
/// `None`。
#[test]
fn task18_1_2_evaluated_candidate_exposes_validation_outcomes() {
    let clock = FakeClock::new(0);
    let expected_outcomes = outcomes_with_correct(7);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            ok(10, expected_outcomes.clone()),
            RunnerBehavior::WorkerError {
                code: "limit_exceeded",
                advance_ms: 5_000,
            },
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert_eq!(
        record.candidates[0].validation_outcomes(),
        Some(expected_outcomes.as_slice())
    );
    assert_eq!(record.candidates[1].validation_outcomes(), None);
}

/// (T10c・REQ-27) 予測の `abstain`・`error` は対応する `Outcome` に写り、正解率の
/// 分母には含まれ分子には含まれない。
#[test]
fn task18_1_2_abstain_and_error_predictions_map_to_outcomes() {
    let clock = FakeClock::new(0);
    let mut outcomes = outcomes_with_correct(8);
    outcomes[8] = Outcome::Abstain;
    outcomes[9] = Outcome::Error;
    let mut runner = FakeRunner::new(&clock, vec![ok(10, outcomes.clone())]);
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(&mut runner, &clock, default_input(&gold, &split, &["c3-a"]))
        .expect("search succeeds");
    assert_eq!(
        record.candidates[0].validation_outcomes(),
        Some(outcomes.as_slice())
    );
    match &record.candidates[0].result {
        CandidateSearchResult::Evaluated {
            validation_accuracy,
        } => {
            assert_eq!(validation_accuracy.correct, 8);
            assert_eq!(validation_accuracy.total, 10);
        }
        other => panic!("expected Evaluated, got {other:?}"),
    }
}

/// (T10b・P0・security.md「秘密情報の混入防止」) `CandidateSearchEntry`・
/// `SearchRecord` の `{:?}`（Debug）出力に、学習ジョブが返した予測ラベルの文字列が
/// 含まれない。
#[test]
fn task18_1_2_candidate_search_entry_debug_does_not_leak_predicted_labels() {
    const MARKER_LABEL: &str = "UNIQUE_PREDICTED_LABEL_MARKER_ZZQX";
    let clock = FakeClock::new(0);
    let outcomes: Vec<Outcome> = (0..VALIDATION_LEN)
        .map(|_| Outcome::Label(MARKER_LABEL.to_string()))
        .collect();
    let mut runner = FakeRunner::new(&clock, vec![ok(10, outcomes)]);
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(&mut runner, &clock, default_input(&gold, &split, &["c3-a"]))
        .expect("search succeeds");
    // 前提: この候補は実際に評価済みで、予測ラベルを保持している。
    assert!(record.candidates[0].validation_outcomes().is_some());

    let entry_debug = format!("{:?}", record.candidates[0]);
    assert!(
        !entry_debug.contains(MARKER_LABEL),
        "CandidateSearchEntry Debug output must not leak predicted labels: {entry_debug}"
    );
    assert!(
        entry_debug.contains("redacted"),
        "CandidateSearchEntry Debug output should indicate redaction: {entry_debug}"
    );
    let record_debug = format!("{record:?}");
    assert!(
        !record_debug.contains(MARKER_LABEL),
        "SearchRecord Debug output must not leak predicted labels: {record_debug}"
    );
}

/// (T11・P0/P1・REQ-39) 学習ジョブ（学習＋予測）が探索予算全体を超えて終わった
/// 場合、評価器を呼ばずに `scoring_skipped_budget_exhausted` として打ち切る
/// （正解率は算出しない。P1 指摘対応: 期限後に評価器という重い処理を呼び出さ
/// ない）。1 候補しかないため `NoEligibleCandidate`（超過後も最後の候補なら
/// `Selected` を返してしまう不具合の再現・修正確認）。
#[test]
fn task18_1_2_job_exceeding_budget_skips_evaluator_and_is_excluded_from_selection() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, vec![ok(4_000_000, outcomes_with_correct(9))]);
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(&mut runner, &clock, default_input(&gold, &split, &["c3-a"]))
        .expect("search succeeds");
    assert_eq!(record.candidates.len(), 1);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringSkippedBudgetExhausted
    );
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T12・P0・REQ-39) 学習ジョブが予算を使い切った候補より後ろに宣言されていた
/// 候補は実行されず、`not_started/budget_exhausted` として記録される。
#[test]
fn task18_1_2_job_exceeding_budget_stops_remaining_candidates() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, vec![ok(3_600_000, outcomes_with_correct(10))]);
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert_eq!(runner.calls, 1, "c3-b は予算超過後に実行されない");
    assert_eq!(record.candidates.len(), 2);
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringSkippedBudgetExhausted
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

/// REQ-18・TASK-18.1-2・REQ-39（P0 指摘対応）: `SearchBudget::new` は
/// [`fandhe_edge_train::search::MAX_SEARCH_BUDGET_SECONDS`] ちょうどは受理し、
/// 1 秒でも超えると `None` を返す。
#[test]
fn task18_1_2_search_budget_rejects_over_max() {
    let max = fandhe_edge_train::search::MAX_SEARCH_BUDGET_SECONDS;
    assert_eq!(SearchBudget::new(max).map(SearchBudget::get), Some(max));
    assert_eq!(SearchBudget::new(max + 1), None);
    assert_eq!(SearchBudget::new(u64::MAX), None);
}

/// (T14・P0/P1・REQ-39) 学習ジョブが終わった時点の経過時間がちょうど探索予算全体
/// （3_600_000ms）に達した場合も「予算到達を合格扱いにしない」
/// （evaluation-contract）に含める。判定を `>` にすると本テストは失敗し
/// （`Evaluated`・`Selected` になる）、`>=` で `scoring_skipped_budget_exhausted`・
/// `NoEligibleCandidate` になる。
#[test]
fn task18_1_2_job_ending_exactly_at_budget_is_excluded() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, vec![ok(3_600_000, outcomes_with_correct(9))]);
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(&mut runner, &clock, default_input(&gold, &split, &["c3-a"]))
        .expect("search succeeds");
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringSkippedBudgetExhausted
    );
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T16・P0・REQ-39) 評価器の呼び出し「後」に探索予算全体を使い切った場合は、
/// 正解率を参考値として記録するが選定対象から除外し（`scoring_exceeded_budget`）、
/// 以降の候補を未着手にする。評価器自体は仮想時計を進めないため、`run_search` の
/// `monotonic()` 呼び出し順（開始 1・候補ループ先頭 2・`run_candidate` の開始 3・
/// 終了 4・学習ジョブ後の予算確認 5・評価器後の予算確認 6）の 6 回目で時計を進める。
#[test]
fn task18_1_2_budget_exhausted_after_evaluator_is_excluded_and_stops_remaining() {
    let clock = FakeClock::new(0).with_advance_at_monotonic_call(6, 4_000_000);
    let mut runner = FakeRunner::new(&clock, vec![ok(10, outcomes_with_correct(9))]);
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert_eq!(runner.calls, 1, "c3-b は予算超過後に実行されない");
    match &record.candidates[0].result {
        CandidateSearchResult::ScoringExceededBudget {
            validation_accuracy,
        } => {
            assert_eq!(validation_accuracy.correct, 9);
            assert_eq!(validation_accuracy.total, 10);
        }
        other => panic!("expected ScoringExceededBudget, got {other:?}"),
    }
    assert_eq!(record.candidates[0].validation_outcomes(), None);
    assert_eq!(
        record.candidates[1].result,
        CandidateSearchResult::NotStarted {
            reason: NotStartedReason::BudgetExhausted
        }
    );
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T16b・P0・REQ-39) 評価器の呼び出し後の経過時間がちょうど探索予算全体に達した
/// 場合も選定対象から除外する（`>=`。T16 と同じ 6 回目の呼び出しで、学習ジョブ後の
/// 経過 10ms との合計がちょうど 3_600_000ms になる分だけ進める）。
#[test]
fn task18_1_2_budget_reached_exactly_after_evaluator_is_excluded() {
    let clock = FakeClock::new(0).with_advance_at_monotonic_call(6, 3_600_000 - 10);
    let mut runner = FakeRunner::new(&clock, vec![ok(10, outcomes_with_correct(9))]);
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(&mut runner, &clock, default_input(&gold, &split, &["c3-a"]))
        .expect("search succeeds");
    assert!(matches!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringExceededBudget { .. }
    ));
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (P0・REQ-39・issue #84 PR #238 レビュー) 候補の学習ジョブが成功しても、実測時間
/// （`elapsed_ms`）がその候補へ配分した持ち時間を超えていた場合は、正解率を算出
/// せず `TrainingExceededTimeLimit` として選定対象から除外する。探索予算 3600 秒を
/// 2 候補へ均等配分（各 1800 秒）し、最初の候補が割当を超えて 2000 秒で成功した
/// 場合の再現。探索全体の予算はまだ残っている（2000 秒 < 3600 秒）ため、2 番目の
/// 候補は通常どおり実行・評価され選定される。
#[test]
fn task18_1_2_training_exceeds_own_time_limit_is_excluded_from_selection() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            ok(2_000_000, outcomes_with_correct(10)),
            ok(10, outcomes_with_correct(9)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert_eq!(runner.calls, 2);
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

/// (P0・REQ-39) 候補の実測時間が割当持ち時間にちょうど一致した場合（`==`）は
/// 超過扱いにしない。1ms でも超えれば `TrainingExceededTimeLimit` になることを
/// 対比で確認する。
#[test]
fn task18_1_2_training_exactly_at_own_time_limit_is_not_excluded_but_one_ms_over_is() {
    for (advance_ms, expect_evaluated) in [(100_000u64, true), (100_001, false)] {
        let clock = FakeClock::new(0);
        let mut runner = FakeRunner::new(&clock, vec![ok(advance_ms, outcomes_with_correct(9))]);
        let gold = validation_gold();
        let split = validation_split_record_fixture();
        let input = input_for(
            &gold,
            &split,
            candidates(&["c3-a"]),
            SearchBudget::default(),
            fixed_policy(100),
        );
        let record = run_search(&mut runner, &clock, input).expect("search succeeds");
        assert_eq!(record.candidates.len(), 1);
        if expect_evaluated {
            assert!(matches!(
                record.candidates[0].result,
                CandidateSearchResult::Evaluated { .. }
            ));
        } else {
            assert_eq!(
                record.candidates[0].result,
                CandidateSearchResult::TrainingExceededTimeLimit
            );
            assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
        }
    }
}

/// (P0・REQ-17・REQ-27・issue #84 PR #238 レビュー) ID を変えずに `input` だけを
/// 差し替えた入力は、ID ハッシュは一致しても中身のハッシュが一致せず、学習を始める
/// 前に `ValidationContentHashMismatch` で拒否される。
#[test]
fn task18_1_2_swapped_input_with_same_ids_is_rejected_before_training() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let gold = validation_gold();
    // 凍結記録は元の入力、実際に渡す入力は 1 件だけ差し替える。
    let split = validation_split_record_fixture();
    let mut swapped: Vec<&[u8]> = VALIDATION_INPUTS.to_vec();
    swapped[4] = b"tampered input";
    let mut input = default_input(&gold, &split, &["c3-a"]);
    input.validation_inputs = &swapped;
    let err = run_search(&mut runner, &clock, input).unwrap_err();
    assert_eq!(err, SearchError::ValidationContentHashMismatch);
    assert_eq!(err.exit_code().code(), 64);
    assert_eq!(runner.calls, 0);
}

/// (P0・REQ-17・REQ-27) ID を変えずに正解ラベルだけを差し替えた場合も拒否される
/// （凍結記録の validation は `negative` のラベルを含み、渡された gold は全件
/// `positive`）。
#[test]
fn task18_1_2_swapped_gold_with_same_ids_is_rejected_before_training() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let mut frozen_labels = validation_gold();
    frozen_labels[7] = "negative";
    let split = split_record_for(&VALIDATION_RECORD_IDS, &VALIDATION_INPUTS, &frozen_labels);
    let gold = validation_gold();
    let err = run_search(&mut runner, &clock, default_input(&gold, &split, &["c3-a"])).unwrap_err();
    assert_eq!(err, SearchError::ValidationContentHashMismatch);
    assert_eq!(runner.calls, 0);
}

/// (P0・REQ-17・REQ-27) 中身のハッシュを持たない古い分割記録は、中身の同一性を
/// 保証できないため拒否される（fail-closed）。ID ハッシュは一致していても止まる。
#[test]
fn task18_1_2_split_record_without_content_hash_is_rejected() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, Vec::new());
    let mut value: serde_json::Value = serde_json::from_str(
        &validation_split_record_fixture()
            .to_json()
            .expect("to_json"),
    )
    .expect("json");
    for split in ["train", "validation", "test"] {
        value["splits"][split]
            .as_object_mut()
            .expect("object")
            .remove("content_sha256");
    }
    let legacy = SplitRecord::from_json_str(&value.to_string()).expect("legacy record loads");
    assert_eq!(legacy.digest(Split::Validation).content_sha256(), None);
    let gold = validation_gold();
    let err = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &legacy, &["c3-a"]),
    )
    .unwrap_err();
    assert_eq!(err, SearchError::ValidationContentHashMismatch);
    assert_eq!(runner.calls, 0);
}

/// 予算到達の記録と選定の照合（TASK-18.2・issue #85・REQ-18 異常系。証拠種別:
/// テストハーネス）。選ばれた候補が `Evaluated` かつ予算到達でないことを確かめる。
fn assert_selected_is_evaluated_and_not_budget_reached(
    record: &fandhe_edge_train::search::SearchRecord,
) {
    let id = selected_id(record);
    let entry = record
        .candidates
        .iter()
        .find(|e| e.candidate_id == id)
        .expect("selected candidate is recorded");
    assert!(matches!(
        entry.result,
        CandidateSearchResult::Evaluated { .. }
    ));
    assert_eq!(entry.budget_reached(), None);
}

/// (TASK-18.2・REQ-18 異常系) 全体予算が学習ジョブで尽きた場合、その候補と
/// 残りの候補が全体予算の予算到達として記録され、選定されない。
#[test]
fn task18_2_search_budget_exhaustion_is_recorded_as_budget_reached_and_not_selected() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, vec![ok(3_600_000, outcomes_with_correct(10))]);
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let input = input_for(
        &gold,
        &split,
        candidates(&["c3-a", "c3-b"]),
        SearchBudget::default(),
        fixed_policy(3600),
    );
    let record = run_search(&mut runner, &clock, input).expect("search succeeds");
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::ScoringSkippedBudgetExhausted
    );
    assert_eq!(
        record.candidates[1].result,
        CandidateSearchResult::NotStarted {
            reason: NotStartedReason::BudgetExhausted
        }
    );
    for entry in &record.candidates {
        assert_eq!(
            entry.budget_reached(),
            Some(BudgetReachedScope::SearchBudget)
        );
        assert_eq!(entry.validation_outcomes(), None);
    }
    assert!(record.budget_reached);
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (TASK-18.2) 評価器の後で予算を越えた候補は正解率を参考値として残すが、
/// 全体予算の予算到達で選定されない。`monotonic()` の 6 回目は T16 と同じ。
#[test]
fn task18_2_scoring_exceeded_budget_keeps_reference_accuracy_but_is_not_selected() {
    let clock = FakeClock::new(0).with_advance_at_monotonic_call(6, 4_000_000);
    let mut runner = FakeRunner::new(&clock, vec![ok(10, outcomes_with_correct(9))]);
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(&mut runner, &clock, default_input(&gold, &split, &["c3-a"]))
        .expect("search succeeds");
    match &record.candidates[0].result {
        CandidateSearchResult::ScoringExceededBudget {
            validation_accuracy,
        } => assert_eq!(validation_accuracy.correct, 9),
        other => panic!("expected ScoringExceededBudget, got {other:?}"),
    }
    assert_eq!(
        record.candidates[0].budget_reached(),
        Some(BudgetReachedScope::SearchBudget)
    );
    assert!(record.budget_reached);
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (TASK-18.2) 壁時計の強制終了は持ち時間の予算到達。次の候補は選定できる。
#[test]
fn task18_2_candidate_wall_timeout_is_budget_reached_and_next_candidate_selected() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::WallTimeout {
                advance_ms: 1_800_000,
            },
            ok(10, outcomes_with_correct(9)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::TrainingTimedOut
    );
    assert_eq!(
        record.candidates[0].budget_reached(),
        Some(BudgetReachedScope::CandidateTimeLimit)
    );
    assert_eq!(record.candidates[1].budget_reached(), None);
    assert_eq!(selected_id(&record), "c3-b");
    assert!(record.budget_reached);
    assert_selected_is_evaluated_and_not_budget_reached(&record);
}

/// (TASK-18.2) ワーカーの `limit_exceeded` で経過時間が持ち時間以上なら
/// 持ち時間の予算到達。
#[test]
fn task18_2_worker_limit_exceeded_after_time_limit_is_candidate_time_limit() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::WorkerError {
                code: "limit_exceeded",
                advance_ms: 1_800_000,
            },
            ok(10, outcomes_with_correct(9)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert_eq!(
        record.candidates[0].result,
        CandidateSearchResult::TrainingTimedOut
    );
    assert_eq!(
        record.candidates[0].budget_reached(),
        Some(BudgetReachedScope::CandidateTimeLimit)
    );
    assert!(record.budget_reached);
    assert_selected_is_evaluated_and_not_budget_reached(&record);
}

/// (TASK-18.2) 持ち時間を超えて成功した候補は、最高正解率でも合格にしない。
#[test]
fn task18_2_training_over_own_time_limit_is_budget_reached_even_with_best_accuracy() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            ok(2_000_000, outcomes_with_correct(10)),
            ok(10, outcomes_with_correct(9)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert_eq!(
        record.candidates[0].budget_reached(),
        Some(BudgetReachedScope::CandidateTimeLimit)
    );
    assert_eq!(selected_id(&record), "c3-b");
    assert!(record.budget_reached);
    assert_selected_is_evaluated_and_not_budget_reached(&record);
}

/// (TASK-18.2) 予算内で全候補が完了すれば予算到達なし。
#[test]
fn task18_2_normal_run_has_no_budget_reached() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            ok(10, outcomes_with_correct(7)),
            ok(10, outcomes_with_correct(9)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    assert!(
        record
            .candidates
            .iter()
            .all(|e| e.budget_reached().is_none())
    );
    assert!(!record.budget_reached);
    assert_selected_is_evaluated_and_not_budget_reached(&record);
}

/// (TASK-18.2) 持ち時間未満の資源上限・その他のワーカー失敗は不合格であって
/// 予算到達ではない（混同しない）。
#[test]
fn task18_2_training_failure_is_not_budget_reached() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::WorkerError {
                code: "limit_exceeded",
                advance_ms: 10,
            },
            RunnerBehavior::WorkerError {
                code: "runtime_error",
                advance_ms: 10,
            },
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    for entry in &record.candidates {
        assert_eq!(entry.result, CandidateSearchResult::TrainingNotCompleted);
        assert_eq!(entry.budget_reached(), None);
    }
    assert!(!record.budget_reached);
}

/// (TASK-18.2) JSON に予算到達のフィールドが出て、`validation_outcomes` は出ない。
#[test]
fn task18_2_record_json_exposes_budget_reached_fields() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            ok(2_000_000, outcomes_with_correct(10)),
            ok(10, outcomes_with_correct(9)),
        ],
    );
    let gold = validation_gold();
    let split = validation_split_record_fixture();
    let record = run_search(
        &mut runner,
        &clock,
        default_input(&gold, &split, &["c3-a", "c3-b"]),
    )
    .expect("search succeeds");
    let json = serde_json::to_value(&record).expect("serialize");
    assert_eq!(json["budget_reached"], serde_json::json!(true));
    assert_eq!(
        json["candidates"][0]["budget_reached"],
        serde_json::json!("candidate_time_limit")
    );
    assert_eq!(
        json["candidates"][0]["result"],
        "training_exceeded_time_limit"
    );
    assert_eq!(
        json["candidates"][1]["budget_reached"],
        serde_json::Value::Null
    );
    assert_eq!(json["candidates"][1]["result"], "evaluated");
    for c in json["candidates"].as_array().expect("array") {
        assert!(c.get("validation_outcomes").is_none());
    }
}
