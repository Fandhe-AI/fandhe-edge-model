//! REQ-18・REQ-27・REQ-39（TASK-18.1-2・issue #84）: 探索予算全体の管理・
//! 複数候補の比較・選定の記録の受け入れ条件を確認する結合テスト
//! （証拠種別: テストハーネス）。
//!
//! `FakeClock`（`crates/train/tests/candidate_time_limit.rs` と同じ `Cell`
//! 方式）・`FakeRunner`（呼び出しごとの所要 ms と結果種別を事前登録した
//! 実行器）・`FakeScorer`（candidate_id ごとの validation 予測を返す推論器）
//! で `fandhe_edge_train::search::run_search` を検証する。`thread::sleep`・
//! 実時間には依存しない（3 OS の CI での決定性のため。`.claude/rules/ci.md`）。
//! 本テストは `crates/train`・`crates/eval`・`fandhe-edge-core` の範囲に
//! 閉じており、`docs/spec` は参照しない。

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fs;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;

use fandhe_edge_eval::metrics::Outcome;
use fandhe_edge_train::request::{Device, TrainRequest, TrainRequestParams};
use fandhe_edge_train::result::{SuccessOutcome, TrainOutcome};
use fandhe_edge_train::search::{
    CandidateSearchResult, NotStartedReason, SearchBudget, SearchCandidate, SearchError,
    SearchInput, SelectionDecision, ValidationScorer, run_search,
};
use fandhe_edge_train::time_allotment::{CandidateRunner, Clock, PerCandidatePolicy};

/// 外部入力と同じ作法で扱うための読み込み前サイズ上限（REQ-39）。
const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

const LABEL_ORDER: [&str; 3] = ["positive", "negative", "neutral"];
const VALIDATION_LEN: usize = 10;

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

/// テストが自由に進められる仮想時計（`candidate_time_limit.rs` と同じ設計）。
struct FakeClock {
    elapsed: Cell<Duration>,
    unix_ms: Cell<u64>,
}

impl FakeClock {
    fn new(start_unix_ms: u64) -> Self {
        Self {
            elapsed: Cell::new(Duration::ZERO),
            unix_ms: Cell::new(start_unix_ms),
        }
    }

    fn advance_ms(&self, ms: u64) {
        self.elapsed
            .set(self.elapsed.get() + Duration::from_millis(ms));
        self.unix_ms.set(self.unix_ms.get() + ms);
    }
}

impl Clock for FakeClock {
    fn monotonic(&self) -> Duration {
        self.elapsed.get()
    }

    fn unix_millis(&self) -> Result<u64, fandhe_edge_train::time_allotment::TimeAllotmentError> {
        Ok(self.unix_ms.get())
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

/// candidate_id ごとに事前登録した validation 予測（または失敗）を返す
/// 推論器。呼び出し順・件数・受け取った `time_limit` を記録する
/// （REQ-27: `validation_gold` を渡さないことは trait の署名で構造的に
/// 保証されるため、ここでは呼び出し順のみ確認する）。
///
/// `advance_ms`（candidate_id ごと）を指定すると、`predict_validation` の
/// 呼び出し中に `clock` を進める（P0・REQ-39: 採点が探索予算を超過する
/// ケースを模擬する。呼び出し自体を打ち切れないことをテストでも示す）。
struct FakeScorer<'a> {
    clock: &'a FakeClock,
    responses: BTreeMap<String, Result<Vec<Outcome>, FakeScorerError>>,
    advance_ms: BTreeMap<String, u64>,
    calls: Vec<String>,
    time_limits: Vec<Duration>,
}

impl<'a> FakeScorer<'a> {
    fn new(
        clock: &'a FakeClock,
        responses: BTreeMap<String, Result<Vec<Outcome>, FakeScorerError>>,
    ) -> Self {
        Self {
            clock,
            responses,
            advance_ms: BTreeMap::new(),
            calls: Vec::new(),
            time_limits: Vec::new(),
        }
    }

    /// `candidate_id` の採点呼び出し中に `clock` を `ms` だけ進めるよう
    /// 登録する（P0 テスト専用）。
    fn with_advance(mut self, candidate_id: &str, ms: u64) -> Self {
        self.advance_ms.insert(candidate_id.to_string(), ms);
        self
    }
}

impl ValidationScorer for FakeScorer<'_> {
    type Error = FakeScorerError;

    fn predict_validation(
        &mut self,
        candidate_id: &str,
        _artifact: &SuccessOutcome,
        time_limit: Duration,
    ) -> Result<Vec<Outcome>, Self::Error> {
        self.calls.push(candidate_id.to_string());
        self.time_limits.push(time_limit);
        if let Some(&ms) = self.advance_ms.get(candidate_id) {
            self.clock.advance_ms(ms);
        }
        self.responses
            .get(candidate_id)
            .cloned()
            .unwrap_or_else(|| panic!("unexpected scorer call for {candidate_id}"))
    }
}

fn fixed_policy(seconds: u32) -> PerCandidatePolicy {
    PerCandidatePolicy::Fixed(NonZeroU32::new(seconds).expect("non-zero"))
}

/// (T1・主受入) 既定予算・`EvenSplit`・3 候補。正解数 7/10・9/10・8/10 の
/// うち最高正解率の候補（9/10）が選ばれる。
#[test]
fn task18_1_2_default_budget_selects_highest_accuracy_candidate() {
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 1_000 },
            RunnerBehavior::Ok { advance_ms: 1_000 },
            RunnerBehavior::Ok { advance_ms: 1_000 },
        ],
    );
    let mut scorer = FakeScorer::new(
        &clock,
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
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, &mut scorer, &clock, input).expect("search succeeds");

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
    assert_eq!(scorer.calls, vec!["c3-a", "c3-b", "c3-c"]);
}

/// (T2・予算の消費) 予算 100 秒・`Fixed(50)`・4 候補（各 50,000ms）。
/// c3-a・c3-b は評価済み、c3-c は予算が尽きた時点の候補として
/// `elapsed_at_start_ms: Some`・`not_started/budget_exhausted`、c3-c より
/// 後に宣言されていた c3-d も（実行順が一度も回ってこなくても）記録に残り
/// `elapsed_at_start_ms: None`・`not_started/budget_exhausted` になる
/// （P1 指摘対応・REQ-18「候補ごとの選定記録」。4 候補目を追加したのは
/// 「予算切れ候補の直後だけでなく、さらにその後ろの候補」も記録される
/// ことを検証するため）。
#[test]
fn task18_1_2_budget_exhaustion_marks_remaining_candidates_not_started() {
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 50_000 },
            RunnerBehavior::Ok { advance_ms: 50_000 },
        ],
    );
    let mut scorer = FakeScorer::new(
        &clock,
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
        candidates,
        budget: SearchBudget::new(100).expect("non-zero"),
        policy: fixed_policy(50),
    };

    let record = run_search(&mut runner, &mut scorer, &clock, input).expect("search succeeds");

    assert_eq!(runner.calls, 2);
    assert_eq!(record.total_elapsed_ms, 100_000);
    assert_eq!(record.candidates.len(), 4);
    assert!(matches!(
        record.candidates[0].result,
        CandidateSearchResult::Evaluated { .. }
    ));
    assert!(matches!(
        record.candidates[1].result,
        CandidateSearchResult::Evaluated { .. }
    ));
    assert_eq!(record.candidates[2].candidate_id, "c3-c");
    assert_eq!(record.candidates[2].elapsed_at_start_ms, Some(100_000));
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
        SelectionDecision::Selected { candidate_id, .. } => assert_eq!(candidate_id, "c3-b"),
        SelectionDecision::NoEligibleCandidate => panic!("expected Selected"),
        _ => panic!("unexpected selection decision"),
    }
}

/// (T3・同率) 8/10 と 8/10 のとき先の候補を選び、`tied_candidate_ids` に
/// 両方が入る。
#[test]
fn task18_1_2_tie_breaks_to_first_declared_candidate() {
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 100 },
            RunnerBehavior::Ok { advance_ms: 100 },
        ],
    );
    let mut scorer = FakeScorer::new(
        &clock,
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
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, &mut scorer, &clock, input).expect("search succeeds");
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
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::LimitExceeded { advance_ms: 10 },
            RunnerBehavior::LimitExceeded { advance_ms: 10 },
        ],
    );
    let mut scorer = FakeScorer::new(&clock, BTreeMap::new());

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
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, &mut scorer, &clock, input).expect("search succeeds");
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
    assert!(scorer.calls.is_empty());
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
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 100 },
            RunnerBehavior::Ok { advance_ms: 100 },
        ],
    );
    let mut scorer = FakeScorer::new(
        &clock,
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
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, &mut scorer, &clock, input).expect("search succeeds");
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

/// (T6・事前検証) ID 重複・`label_order` の不一致・gold の未知ラベル・
/// `out_dir` の重複はそれぞれ Err になり、runner の呼び出しは 0 回。
#[test]
fn task18_1_2_precondition_violations_do_not_consume_budget() {
    let gold = validation_gold();

    // ID 重複。
    {
        let clock = FakeClock::new(0);
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let mut scorer = FakeScorer::new(&clock, BTreeMap::new());
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
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = run_search(&mut runner, &mut scorer, &clock, input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateCandidateId { index: 1 });
        assert_eq!(runner.calls, 0);
    }

    // label_order の不一致。
    {
        let clock = FakeClock::new(0);
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let mut scorer = FakeScorer::new(&clock, BTreeMap::new());
        let mut params = candidate_params("out/c3-a", 1);
        params.label_order = vec!["negative".to_string(), "positive".to_string()];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params,
        }];
        let input = SearchInput {
            label_order: &LABEL_ORDER,
            validation_gold: &gold,
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = run_search(&mut runner, &mut scorer, &clock, input).unwrap_err();
        assert_eq!(err, SearchError::LabelOrderMismatch { index: 0 });
        assert_eq!(runner.calls, 0);
    }

    // gold の未知ラベル。
    {
        let clock = FakeClock::new(0);
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let mut scorer = FakeScorer::new(&clock, BTreeMap::new());
        let bad_gold = ["unknown_label"];
        let candidates = vec![SearchCandidate {
            candidate_id: "c3-a".to_string(),
            params: candidate_params("out/c3-a", 1),
        }];
        let input = SearchInput {
            label_order: &LABEL_ORDER,
            validation_gold: &bad_gold,
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = run_search(&mut runner, &mut scorer, &clock, input).unwrap_err();
        assert_eq!(err, SearchError::UnknownValidationGold { index: 0 });
        assert_eq!(runner.calls, 0);
    }

    // out_dir の重複。
    {
        let clock = FakeClock::new(0);
        let mut runner = FakeRunner::new(&clock, Vec::new());
        let mut scorer = FakeScorer::new(&clock, BTreeMap::new());
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
            candidates,
            budget: SearchBudget::default(),
            policy: PerCandidatePolicy::EvenSplit,
        };
        let err = run_search(&mut runner, &mut scorer, &clock, input).unwrap_err();
        assert_eq!(err, SearchError::DuplicateOutDir { index: 1 });
        assert_eq!(runner.calls, 0);
    }
}

/// (T7・契約違反) scorer が gold と異なる件数を返すと
/// `ScorerOutputMismatch`。
#[test]
fn task18_1_2_scorer_output_length_mismatch_aborts_search() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::Ok { advance_ms: 10 }]);
    let mut scorer = FakeScorer::new(
        &clock,
        BTreeMap::from([(
            "c3-a".to_string(),
            Ok(outcomes_with_correct(9)[..9].to_vec()),
        )]),
    );

    let gold = validation_gold();
    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };
    let err = run_search(&mut runner, &mut scorer, &clock, input).unwrap_err();
    assert_eq!(
        err,
        SearchError::ScorerOutputMismatch {
            index: 0,
            expected: 10,
            actual: 9,
        }
    );
}

/// (T8・runner のエラー) runner が Err を返すと `SearchError::Candidate`。
#[test]
fn task18_1_2_runner_failure_aborts_search() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::RunnerError]);
    let mut scorer = FakeScorer::new(&clock, BTreeMap::new());

    let gold = validation_gold();
    let candidates = vec![SearchCandidate {
        candidate_id: "c3-a".to_string(),
        params: candidate_params("out/c3-a", 1),
    }];
    let input = SearchInput {
        label_order: &LABEL_ORDER,
        validation_gold: &gold,
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };
    let err = run_search(&mut runner, &mut scorer, &clock, input).unwrap_err();
    assert!(matches!(err, SearchError::Candidate { index: 0, .. }));
}

/// (T9・記録の JSON) 選定・候補ごとの記録が期待どおりのキー・値で直列化
/// される。
#[test]
fn task18_1_2_record_serializes_expected_json_shape() {
    let clock = FakeClock::new(1_700_000_000_000);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 10 },
            RunnerBehavior::LimitExceeded {
                advance_ms: 5_000_000,
            },
        ],
    );
    let mut scorer = FakeScorer::new(
        &clock,
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
        candidates,
        budget: SearchBudget::new(3600).expect("non-zero"),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, &mut scorer, &clock, input).expect("search succeeds");
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
    let clock = FakeClock::new(0);
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
    let mut scorer = FakeScorer::new(
        &clock,
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
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, &mut scorer, &clock, input).expect("search succeeds");
    assert_eq!(scorer.calls, vec!["c3-a"]);
    assert_eq!(
        record.candidates[0].validation_outcomes(),
        Some(expected_outcomes.as_slice())
    );
    assert_eq!(record.candidates[1].validation_outcomes(), None);
}

/// (T11・P0・REQ-39) 採点（`predict_validation`）の呼び出し中に探索予算
/// 全体を使い切った場合、その候補は `scoring_exceeded_budget` として記録
/// され、選定対象（`evaluated_owned`）から除外される。1 候補しかない場合
/// `NoEligibleCandidate` になる（超過後も最後の候補なら `Selected` を返して
/// しまう不具合の再現・修正確認）。`validation_outcomes()` も `None`（#87 の
/// McNemar 検定に使えないことを保証する）。
#[test]
fn task18_1_2_scoring_exceeding_budget_is_excluded_from_selection() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::Ok { advance_ms: 10 }]);
    // 採点呼び出し中に予算（3600 秒 = 3_600_000ms）を大きく超えて時計を
    // 進める（採点自体は正解率を計算できるが、期限を守れなかった想定）。
    let mut scorer = FakeScorer::new(
        &clock,
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
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, &mut scorer, &clock, input).expect("search succeeds");
    assert_eq!(record.candidates.len(), 1);
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
    assert_eq!(record.selection, SelectionDecision::NoEligibleCandidate);
}

/// (T12・P0・REQ-39) 採点が予算を超過した候補より後ろに宣言されていた候補は
/// 実行されず、`not_started/budget_exhausted` として記録される（P1 の
/// `drain_remaining_as_not_started` が P0 の超過経路でも呼ばれることの確認）。
#[test]
fn task18_1_2_scoring_exceeding_budget_stops_remaining_candidates() {
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(&clock, vec![RunnerBehavior::Ok { advance_ms: 10 }]);
    let mut scorer = FakeScorer::new(
        &clock,
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
        candidates,
        budget: SearchBudget::default(),
        policy: PerCandidatePolicy::EvenSplit,
    };

    let record = run_search(&mut runner, &mut scorer, &clock, input).expect("search succeeds");
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
    let clock = FakeClock::new(0);
    let mut runner = FakeRunner::new(
        &clock,
        vec![
            RunnerBehavior::Ok { advance_ms: 1_000 },
            RunnerBehavior::Ok { advance_ms: 2_000 },
        ],
    );
    let mut scorer = FakeScorer::new(
        &clock,
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
        candidates,
        budget: SearchBudget::new(3_600).expect("non-zero"),
        policy: PerCandidatePolicy::EvenSplit,
    };

    run_search(&mut runner, &mut scorer, &clock, input).expect("search succeeds");
    assert_eq!(scorer.time_limits.len(), 2);
    // budget = 3_600_000ms。c3-a の採点呼び出し時点では 1_000ms 経過。
    assert_eq!(
        scorer.time_limits[0],
        Duration::from_millis(3_600_000 - 1_000)
    );
    // c3-b の採点呼び出し時点では c3-a の学習（1_000ms）＋ c3-b の学習
    // （2_000ms）で 3_000ms 経過。
    assert_eq!(
        scorer.time_limits[1],
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
