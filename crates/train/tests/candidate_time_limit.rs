//! REQ-18・REQ-39（TASK-18.1-1・issue #83）: 候補ごとの持ち時間管理の受け入れ
//! 条件を確認する結合テスト（証拠種別: テストハーネス）。
//!
//! `FakeClock`・`FakeRunner` で時計と学習ワーカーの実行を差し替え、
//! `crates/train::time_allotment::run_candidate` が
//! - 持ち時間を明示的に短くした候補が `limit_exceeded` で打ち切られ、
//!   経過時間が持ち時間以上だった観測値が記録されること
//!   （`LimitExceeded { elapsed_reached_time_limit: true }`。打ち切り原因
//!   〔持ち時間か他の資源上限か〕は断定しない）
//! - 経過時間が持ち時間未満で打ち切られた場合は観測値が `false` になる
//!   こと（`LimitExceeded { elapsed_reached_time_limit: false }`）
//! - 成功・その他の失敗・実行器のエラーを正しく扱うこと
//!
//! を検証する。`thread::sleep` や実時間には依存しない（3 OS の CI での
//! 決定性のため。`.claude/rules/ci.md`）。本テストは `crates/train`・
//! `fandhe-edge-core` の範囲に閉じており、`docs/spec` は参照しない。

use std::cell::Cell;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use fandhe_edge_train::request::{TrainRequest, TrainRequestParams};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::time_allotment::{
    AllottedSeconds, CandidateRunner, CandidateTimeError, CandidateTimeStatus, Clock,
    PerCandidatePolicy, allot,
};

/// 外部入力と同じ作法で扱うための読み込み前サイズ上限（REQ-39）。
const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("train_contract")
}

fn load_fixture_bytes(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    let metadata =
        fs::metadata(&path).unwrap_or_else(|e| panic!("failed to stat fixture {path:?}: {e}"));
    assert!(
        metadata.len() <= MAX_FIXTURE_BYTES,
        "fixture {path:?} exceeds size limit"
    );
    fs::read(&path).unwrap_or_else(|e| panic!("failed to read fixture {path:?}: {e}"))
}

/// `TrainRequestParams` を、fixture（`request_full.json`。`root` を含む
/// 構文的な値のみで実在しないパス）の内容から再構築する。`TrainRequest`
/// は `Deserialize` を実装しないため（`crates/train/src/request.rs`
/// モジュール doc）、いったん `from_json_slice` で検証済みの値を読み、
/// アクセサから `TrainRequestParams` を組み立て直す。
fn params_from_request_full_fixture() -> TrainRequestParams {
    let bytes = load_fixture_bytes("request_full.json");
    let parsed = TrainRequest::from_json_slice(&bytes).expect("request_full.json must parse");
    TrainRequestParams {
        kind: parsed.kind().to_string(),
        kind_version: parsed.kind_version(),
        config: parsed.config().clone(),
        label_order: parsed.label_order().as_slice().to_vec(),
        max_bytes: parsed.max_bytes(),
        seed: parsed.seed(),
        device: parsed.device(),
        root: parsed.root().to_string(),
        train_path: parsed.train_path().to_string(),
        out_dir: parsed.out_dir().to_string(),
        // `time_limit_seconds`／`rss_limit_bytes` は各テストが上書きする。
        time_limit_seconds: None,
        rss_limit_bytes: None,
    }
}

/// テストが自由に進められる仮想時計。`Cell` で内部可変にし、`&FakeClock`
/// を共有借用したまま [`FakeClock::advance_ms`] で経過時間を進める。
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

    /// 単調時計・壁時計の両方を同じミリ秒数だけ進める。
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

/// 固定の学習ワーカー標準出力を返す実行器。呼び出し中に `clock` を
/// `advance_ms` だけ進めることで、学習ワーカーの実行に要した時間を
/// 模擬する。
struct FakeRunner<'a> {
    clock: &'a FakeClock,
    advance_ms: u64,
    stdout: &'static str,
}

/// `FakeRunner` 用のエラー型（実行器自体が失敗するケースの検証用）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct FakeRunnerError(&'static str);

impl std::fmt::Display for FakeRunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fake runner error: {}", self.0)
    }
}

impl CandidateRunner for FakeRunner<'_> {
    type Error = FakeRunnerError;

    fn run(&mut self, request: &TrainRequest) -> Result<TrainOutcome, Self::Error> {
        self.clock.advance_ms(self.advance_ms);
        TrainOutcome::from_worker_stdout(self.stdout.as_bytes(), request)
            .map_err(|_| FakeRunnerError("worker stdout did not match request"))
    }
}

/// 実行器そのものが失敗する（学習ワーカーの起動に失敗した等）ケース専用の
/// 実行器。
struct AlwaysFailingRunner;

impl CandidateRunner for AlwaysFailingRunner {
    type Error = FakeRunnerError;

    fn run(&mut self, _request: &TrainRequest) -> Result<TrainOutcome, Self::Error> {
        Err(FakeRunnerError("spawn failed"))
    }
}

/// 受け入れ条件（本 Issue の主テスト）: `Fixed(2)` で 2 秒の持ち時間を
/// 配分した候補が、7 秒経過後に `limit_exceeded` を返した場合、
/// `LimitExceeded { elapsed_reached_time_limit: true }` として記録される
/// （経過時間が持ち時間以上だった観測値。打ち切り原因は断定しない）。
#[test]
fn task18_1_1_candidate_exceeding_time_limit_is_recorded_as_limit_exceeded() {
    const STARTED_AT_UNIX_MS: u64 = 1_700_000_000_000;
    let clock = FakeClock::new(STARTED_AT_UNIX_MS);
    let fixed = std::num::NonZeroU32::new(2).expect("non-zero");
    let allotment = allot(
        3600,
        std::num::NonZeroUsize::new(1).expect("non-zero"),
        PerCandidatePolicy::Fixed(fixed),
    )
    .expect("valid allotment");
    let allotted = match allotment {
        fandhe_edge_train::time_allotment::Allotment::Granted(a) => a,
        _ => panic!("expected granted allotment"),
    };
    assert_eq!(allotted.get(), 2);

    let stdout = r#"{"status":"error","code":"limit_exceeded","message":"training exceeded wall-clock budget of 2 seconds"}"#;
    let mut runner = FakeRunner {
        clock: &clock,
        advance_ms: 7000,
        stdout,
    };

    let run = fandhe_edge_train::time_allotment::run_candidate(
        &mut runner,
        &clock,
        params_from_request_full_fixture(),
        allotted,
    )
    .expect("run_candidate must succeed");

    assert_eq!(run.record().time_limit_seconds(), 2);
    assert_eq!(run.record().started_at_unix_ms(), STARTED_AT_UNIX_MS);
    assert_eq!(
        run.record().deadline_at_unix_ms(),
        STARTED_AT_UNIX_MS + 2_000
    );
    assert_eq!(run.record().elapsed_ms(), 7000);
    assert_eq!(
        run.record().status(),
        CandidateTimeStatus::LimitExceeded {
            elapsed_reached_time_limit: true
        }
    );

    let serialized = serde_json::to_value(run.record()).expect("serialize record");
    assert_eq!(
        serialized.get("status"),
        Some(&serde_json::json!("limit_exceeded"))
    );
    assert_eq!(
        serialized.get("elapsed_reached_time_limit"),
        Some(&serde_json::json!(true))
    );
}

/// 判別のテスト: `limit_exceeded` でも、経過時間が持ち時間未満なら
/// `elapsed_reached_time_limit: false`（RSS 等、持ち時間以外の資源上限
/// による打ち切りの可能性が高いが、ここでは断定しない）。
#[test]
fn task18_1_1_limit_exceeded_before_deadline_has_elapsed_reached_time_limit_false() {
    let clock = FakeClock::new(1_700_000_000_000);
    let fixed = std::num::NonZeroU32::new(2).expect("non-zero");
    let allotment = allot(
        3600,
        std::num::NonZeroUsize::new(1).expect("non-zero"),
        PerCandidatePolicy::Fixed(fixed),
    )
    .expect("valid allotment");
    let allotted = match allotment {
        fandhe_edge_train::time_allotment::Allotment::Granted(a) => a,
        _ => panic!("expected granted allotment"),
    };

    let stdout = r#"{"status":"error","code":"limit_exceeded","message":"rss limit exceeded"}"#;
    let mut runner = FakeRunner {
        clock: &clock,
        advance_ms: 1000,
        stdout,
    };

    let run = fandhe_edge_train::time_allotment::run_candidate(
        &mut runner,
        &clock,
        params_from_request_full_fixture(),
        allotted,
    )
    .expect("run_candidate must succeed");

    assert_eq!(
        run.record().status(),
        CandidateTimeStatus::LimitExceeded {
            elapsed_reached_time_limit: false
        }
    );
}

/// 成功のテスト: `result_ok.json` 相当の出力を返した場合は `Completed`。
#[test]
fn task18_1_1_successful_outcome_is_recorded_as_completed() {
    let clock = FakeClock::new(1_700_000_000_000);
    let allotted = allotted_seconds_for_test(120);

    // `result_ok.json` は複数行の整形済み JSON（fixture の可読性のため）。
    // 学習ワーカー出力の契約（`from_worker_stdout`）は改行を含まない 1 行の
    // JSON を要求するため、改行を取り除いて 1 行にする
    // （`crates/train/tests/train_contract_fixture.rs` と同じ作法）。
    let stdout_bytes = load_fixture_bytes("result_ok.json");
    let stdout_owned = String::from_utf8(stdout_bytes)
        .expect("result_ok.json must be utf-8")
        .replace('\n', "");
    let stdout: &'static str = Box::leak(stdout_owned.into_boxed_str());
    let mut runner = FakeRunner {
        clock: &clock,
        advance_ms: 500,
        stdout,
    };

    let run = fandhe_edge_train::time_allotment::run_candidate(
        &mut runner,
        &clock,
        params_from_request_full_fixture(),
        allotted,
    )
    .expect("run_candidate must succeed");

    assert_eq!(run.record().status(), CandidateTimeStatus::Completed);
    assert_eq!(run.record().elapsed_ms(), 500);
    assert!(matches!(run.outcome(), TrainOutcome::Ok(_)));
}

/// その他の失敗: `limit_exceeded` 以外の失敗コードは `Failed { code }`。
#[test]
fn task18_1_1_other_failure_code_is_recorded_as_failed() {
    let clock = FakeClock::new(1_700_000_000_000);
    let allotted = allotted_seconds_for_test(120);

    let stdout = r#"{"status":"error","code":"invalid_data","message":"malformed training data"}"#;
    let mut runner = FakeRunner {
        clock: &clock,
        advance_ms: 10,
        stdout,
    };

    let run = fandhe_edge_train::time_allotment::run_candidate(
        &mut runner,
        &clock,
        params_from_request_full_fixture(),
        allotted,
    )
    .expect("run_candidate must succeed");

    assert_eq!(
        run.record().status(),
        CandidateTimeStatus::Failed {
            code: fandhe_edge_train::result::FailureCode::InvalidData
        }
    );
}

/// 実行器のエラー: `CandidateRunner::run` 自体が失敗した場合は
/// `CandidateTimeError::Runner` を返し、記録は作らない。
#[test]
fn task18_1_1_runner_error_propagates_without_producing_a_record() {
    let clock = FakeClock::new(1_700_000_000_000);
    let allotted = allotted_seconds_for_test(120);
    let mut runner = AlwaysFailingRunner;

    let err = fandhe_edge_train::time_allotment::run_candidate(
        &mut runner,
        &clock,
        params_from_request_full_fixture(),
        allotted,
    )
    .unwrap_err();

    assert_eq!(
        err,
        CandidateTimeError::Runner(FakeRunnerError("spawn failed"))
    );
}

/// テスト内で妥当な [`AllottedSeconds`] を作るヘルパー（`allot` の
/// `Fixed` 経路を借りて、指定した秒数をそのまま [`AllottedSeconds`] に
/// する）。
fn allotted_seconds_for_test(seconds: u32) -> AllottedSeconds {
    match allot(
        u64::from(seconds),
        std::num::NonZeroUsize::new(1).expect("non-zero"),
        PerCandidatePolicy::Fixed(std::num::NonZeroU32::new(seconds).expect("non-zero")),
    )
    .expect("valid allotment")
    {
        fandhe_edge_train::time_allotment::Allotment::Granted(a) => a,
        _ => panic!("expected granted allotment"),
    }
}
