//! `infer`・`evaluate`・`package` の時間・RSS の安全上限を子プロセス境界で強制する監督層（REQ-39・TASK-39.5-3・#577。
//! 親 issue #576 の二層化のうち「安全上限」側。`evaluate`・`package` は追加学習の GiB 級 shard を読むため、#662 で対象に加えた）。
//!
//! # 役割と呼び出し文脈
//!
//! `main.rs` が `infer` を受けると、本モジュールの [`run`] が `fandhe-edge` 自身を worker
//! （環境変数 [`WORKER_ENV`] 付き）として子プロセスで起動し、ガード層の
//! [`fandhe_edge_guard::resource::run_with_limits`] で時間と RSS を監視する。worker は従来どおりの
//! in-process 経路（`stages::run` → `stages::infer`）を実行するので、経路の閉じ込め・形式・sha256・
//! `kind_version` のガードと、runtime の協調的な期限・出力の見張りは worker の中で今までどおり効く。
//! 親は worker の stdout を中継するだけで、上限内の出力（バイト列・終了コード）は変えない。
//!
//! # 3 層の区別（REQ-30・REQ-31・REQ-39 の 2026-10-16 注記、TASK-41.9・#579）
//!
//! ここで強制するのは**安全上限**（暴走止め。超えたら停止して終了コード 20）だけ。目標（p95 250ms・RSS 2 GiB）は
//! 警告のみで、p95 は `package` の `infer_p95.over_guideline` に出る。RSS の目標超過の警告は未出力
//! （現状は目標と安全上限が同じ値で警告区間が無く、`infer` の出力は 1 呼び出し 1 JSON のまま中継する契約のため）。
//!
//! # 結果の写像
//!
//! - 7 種の終了コードで終了し stdout が空でなく切り詰めもない: stdout をそのまま中継し同じコードで終える
//! - 7 種の終了コードだが stdout が空: [`ErrorReport`] を 1 行補う（1 呼び出し 1 JSON）
//! - 切り詰め・シグナル終了・未知の終了コード: 子の出力を捨てて `runtime_error`（70。fail-closed）
//! - 時間・RSS 超過: `limit_exceeded`（20）。message で 2 つを区別する
//! - 監督自体の失敗（起動・待機・計測）: `runtime_error`（70）。message は固定語彙でパス・pid・入力本文・
//!   観測 RSS を含めない
//!
//! # 限界（証拠種別: テストハーネス・模擬）
//!
//! - RSS は約 50 ms 間隔のポーリングで、間隔の間は上限を超えうる（オーバーシュート）
//! - 監視するのは直接の子だけで、子孫プロセスは見ない。macOS の `pti_resident_size` は圧縮メモリを含まない。
//!   終了処理中は観測できない（`fandhe_edge_guard::resource` の「制限」）
//! - [`WORKER_ENV`] を外から付けると監視を外せる。安全上限は暴走止めで、呼び出し元を信頼しない境界ではない
//!   （worker 内の協調的な期限と見張りはその場合も効く）
//! - 親の stdout 中継は最大 `MAX_OUTPUT_CAP` のバッファを持つ。親の書き込み停止は `OutputWatchdog` で回収する
//! - 親が kill したとき、`--out` の OUT が途中まで書かれて残りうる（worker 内の見張りの exit 20 と同じ）
//!
//! 上限値は [`supervisor_config`] に集約する。種類ごとの安全上限（`safety_limits_for_kind`。#578・#435）は
//! `kind` を [`peek_checked_kind`] で先読みして渡す。
//! - 先読みと worker の読み取りの間にパッケージが差し替わると、変わるのは上限の選択だけ（緩い側へ誘導されうる）。
//!   worker のガード検査と協調的な期限は実際の `kind` で効く
//! - `kind` を読めない・許可外のときは厳しい上限（10 秒・2 GiB）に落とす（本来のエラーは worker が出す）

use crate::args::{EvaluateArgs, InferArgs, InferSource, PackageArgs};
use crate::error_report::default_message;
use crate::infer_batch::{
    MAX_INFER_BATCH_OUTPUT_DURATION, OutputWatchdog, SINGLE_HARD_LIMIT_GRACE,
};
use crate::infer_guard::peek_checked_kind;
use crate::output::write_error_report;
use crate::project::{PACKAGE_DIR, PACKAGE_STAGING_DIR, Project, SELECTION_FILE, invalid};
use crate::stages::train::load_trained;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::version_ledger_record::VERSION_LEDGER_FILE;
use fandhe_edge_guard::resource::{
    DEFAULT_STDERR_CAP, GuardRunError, GuardedCommand, GuardedRunOutcome, MAX_OUTPUT_CAP,
    ResourceConfigError, ResourceKind, ResourceLimitExceeded, RunConfig, SafetyLimits, TimeLimit,
    run_with_limits, safety_limits_for_kind,
};
use fandhe_edge_runtime::latency::DEFAULT_LATENCY_TOTAL_TIMEOUT_NS;
use fandhe_edge_runtime::pipeline::MAX_INFER_BATCH_DURATION;
use fandhe_edge_train::stage_files::SelectionRecord;
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;

/// `package` の親がプロジェクト単位の排他に使うロックファイル（中身は空。[`PackageRemnants`]）。
const PACKAGE_LOCK_FILE: &str = ".package.lock";

/// worker として起動されたことを示す環境変数（値が `"1"` と完全一致するときだけ worker）。
pub const WORKER_ENV: &str = "FANDHE_EDGE_INFER_WORKER";

/// 親の時間上限を worker 内の上限より長くする猶予（暫定）。パッケージの読み込みと起動の時間も
/// 親の時間に入るため、worker 自身の `limit_exceeded` JSON が先に出る余地を残す。
// ponytail: 暫定 2 秒。種類ごとの時間上限（#578・#435）に上乗せする固定値で、大きいパッケージの起動が長引くなら見直す。
pub const SUPERVISOR_GRACE: Duration = Duration::from_secs(2);

/// 現在のプロセスが worker か（`infer`・`evaluate`・`package` 共通）。
#[must_use]
pub fn is_worker() -> bool {
    std::env::var_os(WORKER_ENV).is_some_and(|v| v == "1")
}

/// 入力源と種類別の安全上限 `limits` からの監督設定（RSS は `limits.memory()`。迂回しない）。
///
/// 親の時間上限は worker 内の上限（協調的な期限＋見張りまでの猶予）より必ず長い。
/// バッチの時間（全体 600 秒＋出力 60 秒）は種類に依らず追加学習の 60 秒より長いので据え置く。
///
/// # Errors
/// 上限が範囲外（定数から作るため通常は起きない）。
pub fn supervisor_config(
    source: &InferSource,
    limits: SafetyLimits,
) -> Result<RunConfig, ResourceConfigError> {
    let inner = match source {
        InferSource::Text { .. } => limits.time().get().saturating_add(SINGLE_HARD_LIMIT_GRACE),
        InferSource::InputFile { .. } => {
            MAX_INFER_BATCH_DURATION.saturating_add(MAX_INFER_BATCH_OUTPUT_DURATION)
        }
    };
    config_with(inner, limits)
}

/// `evaluate` の監督設定（REQ-39・REQ-41・#662。RSS は `limits.memory()`）。
///
/// 親の時間上限は
/// `2 × limits.time() + 2 × MAX_INFER_BATCH_DURATION + MAX_INFER_BATCH_OUTPUT_DURATION + SUPERVISOR_GRACE`。内訳:
/// - `2 × limits.time()`: パイプライン構築が 2 回ある（適用権を取る前の試し組み立てと、取得後の再構築。
///   どちらも `infer` の 1 件上限と同じく「読み込み込み」の上限を 1 回ぶんずつ見込む）
/// - `2 × MAX_INFER_BATCH_DURATION`: 1 件ずつの推論の期限が 2 回ある（validation の校正と、凍結 test への適用）
/// - `MAX_INFER_BATCH_OUTPUT_DURATION`: 推論の後の結果の書き出し(指標・予測 JSONL・評価記録の `finish`・stdout)。
///   `infer --input-file` の出力段と同じ定数を使う
///
/// `--previous-project-dir`・`--seed-run-project` は保存済みの記録を読むだけで推論しないので数えない。
/// 値は種類が 10 秒の既存 3 種で 1282 秒、追加学習（60 秒）で 1382 秒（どちらも `MAX_TIME_LIMIT` 3600 秒以内）。
///
/// # Errors
/// 上限が範囲外（定数から作るため通常は起きない）。
pub fn evaluate_config(limits: SafetyLimits) -> Result<RunConfig, ResourceConfigError> {
    let inner = limits
        .time()
        .get()
        .saturating_mul(2)
        .saturating_add(MAX_INFER_BATCH_DURATION.saturating_mul(2))
        .saturating_add(MAX_INFER_BATCH_OUTPUT_DURATION);
    config_with(inner, limits)
}

/// `package` の監督設定（REQ-39・REQ-41・#662。RSS は `limits.memory()`）。
///
/// 親の時間上限は
/// `2 × limits.time() + DEFAULT_LATENCY_TOTAL_TIMEOUT + MAX_INFER_BATCH_OUTPUT_DURATION + SUPERVISOR_GRACE`。内訳:
/// - `1 × limits.time()`: パイプライン構築（shard の全読込と sha256 照合を含む。`infer` の 1 件上限と同じく
///   「読み込み込み」の上限を 1 回ぶん見込む）
/// - `DEFAULT_LATENCY_TOTAL_TIMEOUT`（600 秒）: p95 計測（`limits.max_infer_p95_us` があるときだけ。warmup と計測の合計の
///   協調的な期限。`MAX_INFER_BATCH_DURATION` と同じ値）
/// - `1 × limits.time()`: 成果物のステージングへの複写（GiB 級 shard）・複写後の sha256 再照合・容量計測
/// - `MAX_INFER_BATCH_OUTPUT_DURATION`（60 秒）: 版管理台帳・`package/` の公開（fsync・rename）と stdout の書き出し
///
/// 値は種類が 10 秒の既存 3 種で 682 秒、追加学習（60 秒）で 782 秒（どちらも `MAX_TIME_LIMIT` 3600 秒以内）。
///
/// # Errors
/// 上限が範囲外（定数から作るため通常は起きない）。
// ponytail: 複写と再照合の時間を表す既存定数が無いため、暫定で構築の上限 `limits.time()` をもう 1 回足している。
// 実機で GiB 級 shard の複写＋再照合がこれを超えるなら、複写専用の定数を設計して置き換える。
pub fn package_config(limits: SafetyLimits) -> Result<RunConfig, ResourceConfigError> {
    let inner = limits
        .time()
        .get()
        .saturating_mul(2)
        .saturating_add(Duration::from_nanos(DEFAULT_LATENCY_TOTAL_TIMEOUT_NS))
        .saturating_add(MAX_INFER_BATCH_OUTPUT_DURATION);
    config_with(inner, limits)
}

fn config_with(inner: Duration, limits: SafetyLimits) -> Result<RunConfig, ResourceConfigError> {
    RunConfig::new(
        TimeLimit::new(inner.saturating_add(SUPERVISOR_GRACE))?,
        MAX_OUTPUT_CAP,
        DEFAULT_STDERR_CAP,
    )
    .map(|c| c.with_memory_limit(limits.memory()))
}

/// 監督する処理の種類（上限超過・監督失敗の message の語彙を分ける。固定語彙）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    /// `infer`。
    Inference,
    /// `evaluate`（REQ-39・#662）。
    Evaluation,
    /// `package`（REQ-39・#662）。
    Packaging,
}

impl Workload {
    fn noun(self) -> &'static str {
        match self {
            Self::Inference => "inference",
            Self::Evaluation => "evaluation",
            Self::Packaging => "packaging",
        }
    }
}

fn runtime_error() -> ErrorReport {
    ErrorReport::new(
        ExitCode::RuntimeError,
        default_message(ExitCode::RuntimeError),
    )
}

fn supervisor_failed(w: Workload, e: GuardRunError) -> ErrorReport {
    ErrorReport::new(
        ExitCode::RuntimeError,
        format!("{} supervisor failed: {}", w.noun(), e.code()),
    )
}

fn limit_report(w: Workload, e: &ResourceLimitExceeded) -> ErrorReport {
    let what = match e.kind() {
        ResourceKind::Memory => "memory",
        _ => "time",
    };
    ErrorReport::new(
        ExitCode::LimitExceeded,
        format!("{} {what} limit exceeded", w.noun()),
    )
}

/// `infer` を worker として起動し、時間・RSS を監視して結果を `out` へ中継する。
///
/// # Errors
/// `out` への書き込み失敗（呼び出し側は exit 70 に写す）。
pub fn run<W: Write>(
    out: &mut W,
    args: &InferArgs,
    raw_args: impl IntoIterator<Item = OsString>,
    cwd: &Path,
) -> io::Result<ExitCode> {
    let limits = peek_checked_kind(cwd, &args.package)
        .map_or(SafetyLimits::STRICT, |k| safety_limits_for_kind(k.as_str()));
    launch(
        out,
        Workload::Inference,
        supervisor_config(&args.source, limits),
        raw_args,
        cwd,
        || {},
    )
}

/// `evaluate` を worker として起動し、時間・RSS を監視して結果を `out` へ中継する（REQ-39・REQ-41・#662）。
///
/// RSS・時間の上限は候補の `kind`（`result.json` の再検証済みリクエストから読む）で選ぶ。読めない・
/// 許可外なら厳しい上限に落とし、本来のエラーは worker が出す（`infer` と同じ扱い）。
///
/// worker が kill されたとき、最終 test の台帳に適用ロックが残っていれば完了記録（`done-*.lock`）が無く
/// 評価は未成功のまま。再評価は `AlreadyApplied` で拒否され、`package` も配布を許さない（既存の
/// 適用権消費後の失敗と同じ fail-closed）。私用のステージング（`$TMPDIR/fandhe-edge-eval-*`）は
/// best effort の Drop 頼みなので kill では残りうる（0700。評価データと ONNX の複製を含む）。
///
/// # Errors
/// `out` への書き込み失敗（呼び出し側は exit 70 に写す）。
pub fn run_evaluate<W: Write>(
    out: &mut W,
    args: &EvaluateArgs,
    raw_args: impl IntoIterator<Item = OsString>,
    cwd: &Path,
) -> io::Result<ExitCode> {
    let limits =
        peek_evaluate_kind(cwd, args).map_or(SafetyLimits::STRICT, |k| safety_limits_for_kind(&k));
    launch(
        out,
        Workload::Evaluation,
        evaluate_config(limits),
        raw_args,
        cwd,
        || {},
    )
}

/// `package` を worker として起動し、時間・RSS を監視して結果を `out` へ中継する（REQ-39・REQ-41・#662）。
///
/// RSS・時間の上限は選定候補の `kind`（`selection_record.json` の添字の `result.json` から読む。`evaluate` と同じ
/// 読み方）で選ぶ。読めない・許可外なら厳しい上限に落とし、本来のエラーは worker が出す。
///
/// worker が kill されたとき（時間・RSS 超過、または監督失敗）の残り方と後始末:
/// - `package/` は原子的な公開（rename）まで作られないので、kill では半端な `package/` は残らない
/// - 組み立て中の kill では `package.staging/` が、台帳の公開と `package/` の公開の間（ごく短い窓）の kill では
///   `version_ledger.json` も残りうる。残ると再実行が `directory already exists`・`version ledger already exists`
///   （64）で拒否されて利用者が復旧できないため、親が launch の前に両者の有無を記録し（[`PackageRemnants`]）、
///   kill で戻ったときに **launch 前に無かったものだけ** を閉じ込めつきで片付ける。記録から片付けまでは
///   プロジェクト単位の排他ロック（`.package.lock` の `flock`）を保持するので、launch 後に現れたものは自分の
///   worker が作ったものに限られる（同じプロジェクトの並行 `package` は `package already running`（64）で拒否）。`package/` 本体には触れず、
///   `package/` が出来ていれば台帳も消さない。片付けに失敗しても出力は `limit_exceeded` のまま（best effort）
/// - worker が自分で終えた場合（exit 0・10・12・20 等）は親は何も消さない（worker が自分で片付ける）。7 種に写らない
///   終了（panic・シグナル死）は worker が片付けられていないので、kill と同じく親が片付ける
///
/// # Errors
/// `out` への書き込み失敗（呼び出し側は exit 70 に写す）。
pub fn run_package<W: Write>(
    out: &mut W,
    args: &PackageArgs,
    raw_args: impl IntoIterator<Item = OsString>,
    cwd: &Path,
) -> io::Result<ExitCode> {
    let limits =
        peek_package_kind(cwd, args).map_or(SafetyLimits::STRICT, |k| safety_limits_for_kind(&k));
    let remnants = match PackageRemnants::snapshot(cwd, &args.project_dir) {
        Ok(r) => r,
        Err(report) => return write_error_report(out, &report),
    };
    launch(
        out,
        Workload::Packaging,
        package_config(limits),
        raw_args,
        cwd,
        || {
            if let Some(r) = &remnants {
                r.discard_created();
            }
        },
    )
}

/// `package` の launch 前に存在していた `package.staging/`・`version_ledger.json` の記録（kill 後の後始末用。
/// REQ-39・#662）。launch 前から有ったものは利用者の物として消さない。保持している間はプロジェクト単位の
/// 排他ロックを持ち、他の `package` の親を締め出す（drop・プロセス終了で解放）。
pub struct PackageRemnants {
    project: Project,
    _lock: std::fs::File,
    staging_before: bool,
    ledger_before: bool,
}

impl PackageRemnants {
    /// プロジェクトを閉じ込めつきで開いて排他ロックを取り、launch 前の有無を記録する。プロジェクトを開けない
    /// ときは `Ok(None)`（何も消さない。開けない理由は worker が報告する）。
    ///
    /// # Errors
    /// 他の `package` がロックを持っている、またはロックを取れないときは `invalid_input`（64）。ロック後に
    /// 有無を確かめられないときはその経路のエラー（worker は起動しない）。
    pub fn snapshot(cwd: &Path, project_dir: &Path) -> Result<Option<Self>, ErrorReport> {
        let Ok(project) = Project::open(cwd, project_dir) else {
            return Ok(None);
        };
        // 既存なら作成が失敗するだけ（無視）。開けなければ次の open_file が拒否する。
        let _ = project.write_new(PACKAGE_LOCK_FILE, b"");
        let busy = || invalid("package already running");
        let (lock, _) = project.open_file(PACKAGE_LOCK_FILE).map_err(|_| busy())?;
        // 待たない（相手は最大で package の時間上限まで走るため）。
        lock.try_lock().map_err(|_| busy())?;
        // ロックを持ったまま有無を確かめる。確かめられなければロックを手放して起動しない（fail-closed）。
        let staging_before = project.exists(PACKAGE_STAGING_DIR)?;
        let ledger_before = project.exists(VERSION_LEDGER_FILE)?;
        Ok(Some(Self {
            project,
            _lock: lock,
            staging_before,
            ledger_before,
        }))
    }

    /// launch 前に無かった `package.staging/` と（`package/` が無いときの）`version_ledger.json`、
    /// 台帳の書きかけ `.tmp-version_ledger.json-*` を片付ける（best effort。失敗は無視する）。ステージングは
    /// 保持 fd で同一性を確かめて消す。台帳の書き手は `package` だけで、ロックで排他している。
    pub fn discard_created(&self) {
        let _ = self.project.remove_tmp_files_of("", VERSION_LEDGER_FILE);
        if !self.staging_before
            && let Ok(held) = self.project.track_existing_dir(PACKAGE_STAGING_DIR)
        {
            let _ = self.project.remove_created_dir(&held);
        }
        if !self.ledger_before && matches!(self.project.exists(PACKAGE_DIR), Ok(false)) {
            let _ = self.project.remove_file_if_exists(VERSION_LEDGER_FILE);
        }
    }
}

/// 親が種類別の上限を選ぶために、選定候補の `kind` だけを読む（失敗は `None`）。
fn peek_package_kind(cwd: &Path, args: &PackageArgs) -> Option<String> {
    let project = Project::open(cwd, &args.project_dir).ok()?;
    let bytes = project.read(SELECTION_FILE, 64 * 1024).ok()?;
    let selection = SelectionRecord::from_json_slice(&bytes).ok()?;
    let (request, _) = load_trained(&project, selection.candidate_index).ok()??;
    Some(request.kind().to_string())
}

/// 親が種類別の上限を選ぶために、対象候補の `kind` だけを読む（失敗は `None`）。
fn peek_evaluate_kind(cwd: &Path, args: &EvaluateArgs) -> Option<String> {
    let project = Project::open(cwd, &args.project_dir).ok()?;
    let (request, _) = load_trained(&project, args.candidate).ok()??;
    Some(request.kind().to_string())
}

/// 自分自身を worker として起動し、`config` で監視して中継する。
fn launch<W: Write>(
    out: &mut W,
    workload: Workload,
    config: Result<RunConfig, ResourceConfigError>,
    raw_args: impl IntoIterator<Item = OsString>,
    cwd: &Path,
    on_killed: impl FnOnce(),
) -> io::Result<ExitCode> {
    let prepared = config
        .map_err(|_| GuardRunError::InvalidConfig)
        .and_then(|config| {
            let exe = std::env::current_exe().map_err(|_| GuardRunError::Spawn)?;
            let mut cmd = GuardedCommand::new(exe)?
                .current_dir(cwd)
                .env(WORKER_ENV, "1");
            for a in raw_args {
                cmd = cmd.arg(a);
            }
            Ok((cmd, config))
        });
    match prepared {
        Ok((cmd, config)) => run_supervised_cleaning(
            workload,
            out,
            &mut io::stderr().lock(),
            &cmd,
            &config,
            on_killed,
        ),
        Err(e) => write_error_report(out, &supervisor_failed(workload, e)),
    }
}

/// 起動・監視・中継の本体（テストで偽の子を渡せるよう分けている）。
///
/// # Errors
/// `out` への書き込み失敗。
pub fn run_supervised<W: Write, E: Write>(
    out: &mut W,
    err: &mut E,
    cmd: &GuardedCommand,
    config: &RunConfig,
) -> io::Result<ExitCode> {
    run_supervised_as(Workload::Inference, out, err, cmd, config)
}

/// [`run_supervised`] の処理種別つき（上限超過・監督失敗の message の語彙だけが変わる）。
///
/// # Errors
/// `out` への書き込み失敗。
pub fn run_supervised_as<W: Write, E: Write>(
    workload: Workload,
    out: &mut W,
    err: &mut E,
    cmd: &GuardedCommand,
    config: &RunConfig,
) -> io::Result<ExitCode> {
    run_supervised_cleaning(workload, out, err, cmd, config, || {})
}

/// [`run_supervised_as`] に、worker が監督側に kill された（時間・RSS 超過、または監督失敗）ときだけ呼ぶ
/// 後始末 `on_killed` を足したもの（`package` のステージング掃除用。REQ-39・#662）。7 種に写らない終了（panic・
/// シグナル死）でも呼ぶ。worker が 7 種のどれかで自分で終えた場合は呼ばない。出力の写像は [`run_supervised_as`] と同じ。
///
/// # Errors
/// `out` への書き込み失敗。
pub fn run_supervised_cleaning<W: Write, E: Write>(
    workload: Workload,
    out: &mut W,
    err: &mut E,
    cmd: &GuardedCommand,
    config: &RunConfig,
    on_killed: impl FnOnce(),
) -> io::Result<ExitCode> {
    let outcome = run_with_limits(cmd, config);
    // 親の stdout が詰まっても止まらないよう、既存の見張りで守る（書き込み停止は exit 20）。
    let _watchdog = OutputWatchdog::arm(true, MAX_INFER_BATCH_OUTPUT_DURATION)?;
    match outcome {
        Ok(GuardedRunOutcome::Exited { status, output, .. }) => {
            // stderr は契約外のため中継の失敗は無視する。
            let _ = err.write_all(output.stderr());
            // 7 種に写らない終了（panic・シグナル死）は worker が後始末できていないので親が片付ける。
            if status
                .code()
                .and_then(|c| ExitCode::try_from(c).ok())
                .is_none()
            {
                on_killed();
            }
            relay_exited(
                out,
                status.code(),
                output.stdout(),
                output.stdout_truncated(),
            )
        }
        Ok(GuardedRunOutcome::LimitExceeded(e)) => {
            on_killed();
            write_error_report(out, &limit_report(workload, &e))
        }
        Ok(_) => {
            on_killed();
            write_error_report(out, &runtime_error())
        }
        Err(e) => {
            on_killed();
            write_error_report(out, &supervisor_failed(workload, e))
        }
    }
}

/// 子が終了したときの中継（写像表は module doc）。
///
/// # Errors
/// `out` への書き込み失敗。
pub fn relay_exited<W: Write>(
    out: &mut W,
    code: Option<i32>,
    stdout: &[u8],
    truncated: bool,
) -> io::Result<ExitCode> {
    let Some(exit) = code.and_then(|c| ExitCode::try_from(c).ok()) else {
        return write_error_report(out, &runtime_error());
    };
    if truncated {
        return write_error_report(out, &runtime_error());
    }
    if stdout.is_empty() {
        return write_error_report(out, &ErrorReport::new(exit, default_message(exit)));
    }
    out.write_all(stdout)?;
    out.flush()?;
    Ok(exit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn text() -> InferSource {
        InferSource::Text {
            text: "x".to_string(),
            id: None,
        }
    }

    fn file() -> InferSource {
        InferSource::InputFile {
            path: PathBuf::from("a"),
            out: None,
        }
    }

    fn relay(code: Option<i32>, stdout: &[u8], truncated: bool) -> (ExitCode, String) {
        let mut buf = Vec::new();
        let c = relay_exited(&mut buf, code, stdout, truncated).unwrap();
        (c, String::from_utf8(buf).unwrap())
    }

    #[test]
    fn req39_config_text_is_12_5_seconds_and_file_is_662_seconds() {
        // 既存の 3 種と表に無い kind は厳しい側（10 秒・2 GiB）。
        for kind in ["c1", "c3", "autoregressive", "unknown"] {
            let l = safety_limits_for_kind(kind);
            let t = supervisor_config(&text(), l).unwrap();
            assert_eq!(t.time_limit(), Duration::from_millis(12_500));
            assert_eq!(t.memory_limit(), Some(2_147_483_648));
            let f = supervisor_config(&file(), l).unwrap();
            assert_eq!(f.time_limit(), Duration::from_secs(662));
            assert_eq!(f.memory_limit(), Some(2_147_483_648));
        }
    }

    /// REQ-39・REQ-41・TASK-41.8・#435: 追加学習の種類は 60 秒・24 GiB（親は 62.5 秒）。
    #[test]
    fn req39_config_qwen2_lora_is_62_5_seconds_and_24_gib() {
        let l = safety_limits_for_kind("qwen2_lora");
        let t = supervisor_config(&text(), l).unwrap();
        assert_eq!(t.time_limit(), Duration::from_millis(62_500));
        assert_eq!(t.memory_limit(), Some(25_769_803_776));
        let f = supervisor_config(&file(), l).unwrap();
        assert_eq!(f.time_limit(), Duration::from_secs(662));
        assert_eq!(f.memory_limit(), Some(25_769_803_776));
    }

    /// REQ-39・REQ-41・#662: `evaluate` の親の時間は `2 × limits.time() + 2 × 600 秒 + 60 秒 + 2 秒`。
    /// 既存 3 種と未知の種類は 1282 秒・2 GiB、追加学習は 1382 秒・24 GiB。
    #[test]
    fn req39_evaluate_config_is_1282_seconds_for_byte_kinds_and_1382_for_qwen2_lora() {
        for kind in ["c1", "c3", "autoregressive", "unknown"] {
            let c = evaluate_config(safety_limits_for_kind(kind)).unwrap();
            assert_eq!(c.time_limit(), Duration::from_secs(1282));
            assert_eq!(c.memory_limit(), Some(2_147_483_648));
        }
        let c = evaluate_config(safety_limits_for_kind("qwen2_lora")).unwrap();
        assert_eq!(c.time_limit(), Duration::from_secs(1382));
        assert_eq!(c.memory_limit(), Some(25_769_803_776));
        // kind を読めなかったときの厳しい側は byte kind と同じ値になる。
        let s = evaluate_config(SafetyLimits::STRICT).unwrap();
        assert_eq!(s.time_limit(), Duration::from_secs(1282));
        assert_eq!(s.memory_limit(), Some(2_147_483_648));
    }

    /// REQ-39・#662: `evaluate` の上限超過・監督失敗の message は `evaluation` の固定語彙。
    #[test]
    fn req39_evaluate_messages_use_evaluation_vocabulary() {
        let t = limit_report(
            Workload::Evaluation,
            &ResourceLimitExceeded::time(Duration::from_secs(1), Duration::from_secs(2), true),
        );
        assert_eq!(t.code, ExitCode::LimitExceeded);
        assert_eq!(t.message, "evaluation time limit exceeded");
        let m = limit_report(
            Workload::Evaluation,
            &ResourceLimitExceeded::memory(
                Duration::from_secs(1),
                1,
                2,
                Duration::from_secs(1),
                true,
            ),
        );
        assert_eq!(m.message, "evaluation memory limit exceeded");
        let f = supervisor_failed(Workload::Evaluation, GuardRunError::Spawn);
        assert_eq!(f.code, ExitCode::RuntimeError);
        assert_eq!(
            f.message,
            format!(
                "evaluation supervisor failed: {}",
                GuardRunError::Spawn.code()
            )
        );
    }

    /// REQ-39・REQ-41・#662: `package` の親の時間は `2 × limits.time() + 600 秒 + 60 秒 + 2 秒`。
    /// 既存 3 種と未知の種類・読めないときの厳しい側は 682 秒・2 GiB、追加学習は 782 秒・24 GiB。
    #[test]
    fn req39_package_config_is_682_seconds_for_byte_kinds_and_782_for_qwen2_lora() {
        for kind in ["c1", "c3", "autoregressive", "unknown"] {
            let c = package_config(safety_limits_for_kind(kind)).unwrap();
            assert_eq!(c.time_limit(), Duration::from_secs(682));
            assert_eq!(c.memory_limit(), Some(2_147_483_648));
        }
        let c = package_config(safety_limits_for_kind("qwen2_lora")).unwrap();
        assert_eq!(c.time_limit(), Duration::from_secs(782));
        assert_eq!(c.memory_limit(), Some(25_769_803_776));
        let s = package_config(SafetyLimits::STRICT).unwrap();
        assert_eq!(s.time_limit(), Duration::from_secs(682));
        assert_eq!(s.memory_limit(), Some(2_147_483_648));
    }

    /// REQ-39・#662: `package` の上限超過・監督失敗の message は `packaging` の固定語彙。
    #[test]
    fn req39_package_messages_use_packaging_vocabulary() {
        let t = limit_report(
            Workload::Packaging,
            &ResourceLimitExceeded::time(Duration::from_secs(1), Duration::from_secs(2), true),
        );
        assert_eq!(t.code, ExitCode::LimitExceeded);
        assert_eq!(t.message, "packaging time limit exceeded");
        let m = limit_report(
            Workload::Packaging,
            &ResourceLimitExceeded::memory(
                Duration::from_secs(1),
                1,
                2,
                Duration::from_secs(1),
                true,
            ),
        );
        assert_eq!(m.message, "packaging memory limit exceeded");
        let f = supervisor_failed(Workload::Packaging, GuardRunError::Spawn);
        assert_eq!(f.code, ExitCode::RuntimeError);
        assert_eq!(
            f.message,
            format!(
                "packaging supervisor failed: {}",
                GuardRunError::Spawn.code()
            )
        );
    }

    #[test]
    fn req39_relay_passes_through_every_exit_code_bytes_unchanged() {
        for c in ExitCode::ALL {
            let body = b"{\"x\":1}\n";
            let (got, text) = relay(Some(i32::from(c.code())), body, false);
            assert_eq!(got, c);
            assert_eq!(text.as_bytes(), body);
        }
    }

    #[test]
    fn req21_relay_fills_error_report_when_stdout_empty() {
        let (c, text) = relay(Some(20), b"", false);
        assert_eq!(c, ExitCode::LimitExceeded);
        assert_eq!(
            text,
            "{\"code\":\"limit_exceeded\",\"message\":\"resource limit exceeded\"}\n"
        );
    }

    #[test]
    fn req39_relay_fails_closed_on_truncation_signal_and_unknown_code() {
        for (code, trunc) in [(Some(0), true), (None, false), (Some(3), false)] {
            let (c, text) = relay(code, b"{\"ok\":1}\n", trunc);
            assert_eq!(c, ExitCode::RuntimeError);
            assert_eq!(
                text,
                "{\"code\":\"runtime_error\",\"message\":\"runtime error\"}\n"
            );
        }
    }

    #[test]
    fn req39_limit_report_distinguishes_time_and_memory() {
        let t = limit_report(
            Workload::Inference,
            &ResourceLimitExceeded::time(Duration::from_secs(1), Duration::from_secs(2), true),
        );
        assert_eq!(t.code, ExitCode::LimitExceeded);
        assert_eq!(t.message, "inference time limit exceeded");
        let m = limit_report(
            Workload::Inference,
            &ResourceLimitExceeded::memory(
                Duration::from_secs(1),
                1,
                2,
                Duration::from_secs(1),
                true,
            ),
        );
        assert_eq!(m.code, ExitCode::LimitExceeded);
        assert_eq!(m.message, "inference memory limit exceeded");
    }

    #[test]
    fn req21_supervisor_errors_map_to_runtime_error_without_details() {
        for e in [
            GuardRunError::InvalidProgram,
            GuardRunError::Spawn,
            GuardRunError::MemoryProbe,
        ] {
            let r = supervisor_failed(Workload::Inference, e);
            assert_eq!(r.code, ExitCode::RuntimeError);
            assert_eq!(
                r.message,
                format!("inference supervisor failed: {}", e.code())
            );
        }
    }

    /// 候補 2 に `qwen2_lora` の学習結果（共有 fixture の request・result）を置いた小さなプロジェクトを作る。
    #[cfg(unix)] // Project は Linux・macOS 限定（Windows は unsupported_platform）。
    fn qwen2_project(case: &str) -> (std::path::PathBuf, Project) {
        use crate::project::{REQUEST_FILE, RESULT_FILE};
        let cwd = std::fs::canonicalize(std::env::temp_dir())
            .expect("temp")
            .join(format!("fandhe-sup-peek-{case}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cwd);
        std::fs::create_dir_all(&cwd).expect("cwd");
        let project = Project::create(&cwd, Path::new("proj")).expect("project");
        project.create_dir("candidates").expect("candidates");
        project.create_dir("candidates/2").expect("candidate");
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/train_contract");
        for (name, dest) in [
            ("request_qwen2_lora.json", REQUEST_FILE),
            ("result_ok_qwen2_lora.json", RESULT_FILE),
        ] {
            // 結果 JSON は 1 行形式（共有 fixture は整形済みなので改行を除く）。
            let text = std::fs::read_to_string(fixtures.join(name)).expect("fixture");
            let bytes = text.replace('\n', "").into_bytes();
            project
                .write_new(Path::new("candidates/2").join(dest), &bytes)
                .expect("write");
        }
        (cwd, project)
    }

    /// REQ-39・REQ-41・#662: 親が上限を選ぶための kind の先読みは、`evaluate` は `--candidate`、`package` は
    /// `selection_record.json` の添字の `result.json` から `qwen2_lora` を読む。読めなければ `None`（厳しい上限へ）。
    #[cfg(unix)]
    #[test]
    fn req39_issue662_peek_kind_reads_qwen2_lora_for_evaluate_and_package() {
        let (cwd, project) = qwen2_project("kind");
        project
            .write_new(
                SELECTION_FILE,
                br#"{"candidate_index":2,"candidate_id":"qwen2_lora","rule":"r","validation_correct":1,"validation_total":2}"#,
            )
            .expect("selection");
        let evaluate = |candidate| EvaluateArgs {
            project_dir: "proj".into(),
            candidate,
            previous_project_dir: None,
            seed_run_projects: Vec::new(),
        };
        let package = PackageArgs {
            project_dir: "proj".into(),
            allow_smoke: false,
            previous_project_dir: None,
        };
        assert_eq!(
            peek_evaluate_kind(&cwd, &evaluate(2)).as_deref(),
            Some("qwen2_lora")
        );
        assert_eq!(
            peek_package_kind(&cwd, &package).as_deref(),
            Some("qwen2_lora")
        );
        // 学習結果の無い候補・存在しないプロジェクトは `None`。
        assert_eq!(peek_evaluate_kind(&cwd, &evaluate(0)), None);
        let missing = PackageArgs {
            project_dir: "nope".into(),
            ..package
        };
        assert_eq!(peek_package_kind(&cwd, &missing), None);
    }
}
