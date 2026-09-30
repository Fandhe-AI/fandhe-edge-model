//! 資源の上限のうち実行時間（推論 1 件あたり暫定 10 秒）とメモリ（RSS。暫定 2 GiB・模擬）を
//! プロセス境界で強制する（REQ-39・TASK-39.5-1・#170・TASK-39.5-2・#171）。
//!
//! REQ-39 異常系は、推論 1 件の実行時間が暫定上限（10 秒。推論経路向けで、学習ワーカーの
//! 上限とは別）を超えたらプロセスを強制終了し、資源上限の超過として区別できる形で記録する
//! ことを求める。PoC-20 ケース 3 の実測は `subprocess.run(timeout=...)` による外側からの
//! 強制終了だった。`fandhe-edge-runtime` の期限（`infer_one_within` 等）はプロセス内の協調的な
//! もので止まった 1 件を中断できないため、強制的な上限は本モジュールが子プロセスの境界で担う。
//!
//! メモリ（RSS）は PoC-20 ケース 3 と同じく、子の RSS を 50 ms 間隔でポーリングし、上限を厳密に
//! 超えたら kill して回収する方式（模擬）。`setrlimit` 等の確実な上限機構ではない。証拠の種別は
//! テストハーネス（実プロセス・実割り当て）で、実 CLI の推論が 2 GiB を超える実測ではない。
//!
//! # 使われ方
//!
//! 将来の CLI・MCP・TUI の配線（#136・REQ-36・REQ-37）が、`fandhe-edge infer` を子プロセスとして
//! [`run_with_limits`] で起動する。起動対象の経路検査（`path`・`package` の閉じ込め）は行わない。
//! 呼び出し側が既存のガードを通した経路だけを渡す前提とする。
//!
//! # 不変条件
//!
//! - シェルを経由せず、program は絶対パス必須（PATH 探索をしない）。stdin は null
//! - 環境変数は `env_clear` のうえ許可リスト（REQ-38）と呼び出し側が明示した値だけを渡す
//! - stdout・stderr は上限つきで読み、超過分は読み捨てて drain する（子のパイプ詰まり防止）。
//!   Linux・macOS はパイプを非ブロッキングにして監視ループ内で読み、スレッドを残さない
//! - 時間は子の起動前から測る（起動に要した時間も上限に含む）
//! - kill は直前の `try_wait` が未終了を返した直接の子にだけ送る（回収済み pid への誤送出を防ぐ）。
//!   期限後に完了を観測した場合は kill せず時間超過として扱う（fail-closed。成功扱いにしない）
//! - kill 後の回収待ち・読み取りスレッドの待ちは有界。回収を確認できなければエラー
//! - 記録・エラーに入力本文・子の出力・パス・pid を含めない（固定語彙と数値のみ）
//! - RSS の計測は「直前の `try_wait` が未終了を返した直接の子」にだけ行う。未回収の間は pid が
//!   再利用されないため、別プロセスの RSS を測る・誤って kill することがない
//! - RSS の計測失敗は成功扱いにせず、子を kill・回収して `MemoryProbe`（70）を返す（fail-closed。
//!   PoC-20 は計測失敗を無視していた）。計測手段の無い OS でメモリ上限が指定されたら起動前に
//!   `MemoryLimitUnsupported`（70）を返す。時間とメモリが同時に該当した場合は時間を優先する
//! - 計測の待ち（macOS の `ps`）・読み取り（`/proc`・`ps` の出力）は有界
//!
//! # 制限
//!
//! - 対象は直接の子だけで、子孫プロセスは kill しない。`fandhe-edge infer` は子プロセスを
//!   起動しない（REQ-32）ため受け入れる。子孫の管理にはプロセスグループ（`unsafe`・新規依存）が
//!   要り、ユーザー承認事項のため行わない
//! - Linux・macOS 以外の OS は読み取りスレッドで代替し、期限超過時は子孫がパイプを離すまで
//!   スレッドが残りうる
//! - 実装は全 OS でビルドされるが、検証環境は Mac のみ（Windows は実機検証の対象外）
//! - メモリ上限は模擬（RSS ポーリング）で、ポーリングの間（約 50 ms と計測の遅延）は上限を超えうる
//!   （オーバーシュート）。子孫の RSS は監視しない。macOS の RSS は圧縮メモリを含まない指標。
//!   cgroup・コンテナのメモリ制限などの確実な上限機構への置き換えは別課題（TASK-39.5 の備考）
//! - ファイルサイズ上限は [`crate::file_size`]（#172）の範囲

use fandhe_edge_core::exitcode::ExitCode;
use std::ffi::OsString;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// 推論プロセスの RSS の暫定上限（2 GiB。REQ-39）。値の出所は共通コアの `limits`。
pub use fandhe_edge_core::limits::INFER_RSS_LIMIT_BYTES;
/// 推論 1 件の実行時間の暫定上限（REQ-39）。値の出所は共通コアの `limits`（runtime・cli と共有）。
pub use fandhe_edge_core::limits::INFER_TIME_LIMIT;
/// 指定できる時間上限の最大値（暫定 1 時間。REQ-39）。値の出所は共通コアの `limits`。
pub use fandhe_edge_core::limits::MAX_TIME_LIMIT;
/// stdout の既定の読み取り上限（推論 1 件の JSON は小さい）。
pub const DEFAULT_STDOUT_CAP: usize = 1024 * 1024;
/// stderr の既定の読み取り上限。
pub const DEFAULT_STDERR_CAP: usize = 64 * 1024;
/// 読み取り上限の最大値（暫定 256 MiB。REQ-39）。値の出所は共通コアの `limits`（cli と共有）。
pub const MAX_OUTPUT_CAP: usize = fandhe_edge_core::limits::MAX_OUTPUT_BYTES;

/// 監視のポーリング間隔（busy-spin を避ける）。
const POLL_INTERVAL: Duration = Duration::from_millis(10);
/// RSS の計測間隔（PoC-20 の 0.05 秒。`POLL_INTERVAL` とは別に `Clock` で管理する）。
const MEMORY_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// kill 後に回収を待つ上限。
const KILL_WAIT_TIMEOUT: Duration = Duration::from_secs(2);
/// 終了後に読み取りスレッドを待つ上限（孫がパイプを保持すると EOF が来ないため）。
const READER_WAIT_TIMEOUT: Duration = Duration::from_secs(1);

/// 設定値の検証エラー（`invalid_input`=64）。値そのものは含めない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceConfigError {
    /// 時間上限が 0 または [`MAX_TIME_LIMIT`] 超。
    TimeLimitOutOfRange,
    /// 読み取り上限が 0 または [`MAX_OUTPUT_CAP`] 超。
    OutputCapOutOfRange,
    /// メモリ上限が 0 または [`INFER_RSS_LIMIT_BYTES`] 超。
    MemoryLimitOutOfRange,
}

impl ResourceConfigError {
    /// 機械可読な理由コード。
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::TimeLimitOutOfRange => "time_limit_out_of_range",
            Self::OutputCapOutOfRange => "output_cap_out_of_range",
            Self::MemoryLimitOutOfRange => "memory_limit_out_of_range",
        }
    }

    /// 対応する終了コード（REQ-21）。
    #[must_use]
    pub const fn exit_code(self) -> ExitCode {
        ExitCode::InvalidInput
    }
}

impl fmt::Display for ResourceConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid resource config: {}", self.code())
    }
}

impl std::error::Error for ResourceConfigError {}

/// 検証済みの時間上限（1 ns 以上 [`MAX_TIME_LIMIT`] 以下）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeLimit(Duration);

impl TimeLimit {
    /// 上限を検証して作る。
    ///
    /// # Errors
    /// 0 または [`MAX_TIME_LIMIT`] 超は [`ResourceConfigError::TimeLimitOutOfRange`]。
    pub fn new(limit: Duration) -> Result<Self, ResourceConfigError> {
        if limit.is_zero() || limit > MAX_TIME_LIMIT {
            return Err(ResourceConfigError::TimeLimitOutOfRange);
        }
        Ok(Self(limit))
    }

    /// 推論 1 件の暫定上限（[`INFER_TIME_LIMIT`]）。
    #[must_use]
    pub const fn infer_default() -> Self {
        Self(INFER_TIME_LIMIT)
    }

    /// 上限値。
    #[must_use]
    pub const fn get(self) -> Duration {
        self.0
    }
}

/// 検証済みのメモリ（RSS）上限（1 バイト以上 [`INFER_RSS_LIMIT_BYTES`] 以下。REQ-39）。
///
/// 暫定上限より緩い値は渡せない（`file_size` の上限と同じ方針）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryLimit(u64);

impl MemoryLimit {
    /// 上限を検証して作る。
    ///
    /// # Errors
    /// 0 または [`INFER_RSS_LIMIT_BYTES`] 超は [`ResourceConfigError::MemoryLimitOutOfRange`]。
    pub fn new(bytes: u64) -> Result<Self, ResourceConfigError> {
        if bytes == 0 || bytes > INFER_RSS_LIMIT_BYTES {
            return Err(ResourceConfigError::MemoryLimitOutOfRange);
        }
        Ok(Self(bytes))
    }

    /// 推論プロセスの暫定上限（[`INFER_RSS_LIMIT_BYTES`]）。
    #[must_use]
    pub const fn infer_default() -> Self {
        Self(INFER_RSS_LIMIT_BYTES)
    }

    /// 上限値（バイト）。
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// 超過した資源の種類。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResourceKind {
    /// 実行時間（壁時計）。
    Time,
    /// メモリ（RSS。ポーリングによる模擬。TASK-39.5-2・#171）。
    Memory,
}

/// 資源上限の超過の記録。正常終了・ほかのエラーと区別できる（REQ-39）。
///
/// 入力本文と子の出力は保持しない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLimitExceeded {
    kind: ResourceKind,
    limit: Duration,
    elapsed: Duration,
    child_reaped: bool,
    /// メモリ超過のときだけ `Some((適用した上限, 観測した RSS))`（バイト）。
    memory: Option<(u64, u64)>,
}

impl ResourceLimitExceeded {
    /// 時間超過の記録を作る（上位層の写像テスト・模擬用。runner 自身も同じ値を作る）。
    #[must_use]
    pub const fn time(limit: Duration, elapsed: Duration, child_reaped: bool) -> Self {
        Self {
            kind: ResourceKind::Time,
            limit,
            elapsed,
            child_reaped,
            memory: None,
        }
    }

    /// メモリ超過の記録を作る（`time_limit` は同時に適用していた時間上限）。
    #[must_use]
    pub const fn memory(
        time_limit: Duration,
        memory_limit_bytes: u64,
        observed_rss_bytes: u64,
        elapsed: Duration,
        child_reaped: bool,
    ) -> Self {
        Self {
            kind: ResourceKind::Memory,
            limit: time_limit,
            elapsed,
            child_reaped,
            memory: Some((memory_limit_bytes, observed_rss_bytes)),
        }
    }

    /// 適用したメモリ上限（バイト）。メモリ超過のときだけ `Some`。
    #[must_use]
    pub const fn memory_limit_bytes(&self) -> Option<u64> {
        match self.memory {
            Some((limit, _)) => Some(limit),
            None => None,
        }
    }

    /// 超過を検出した時の RSS（バイト）。メモリ超過のときだけ `Some`。
    #[must_use]
    pub const fn observed_rss_bytes(&self) -> Option<u64> {
        match self.memory {
            Some((_, observed)) => Some(observed),
            None => None,
        }
    }

    /// 超過した資源の種類。
    #[must_use]
    pub const fn kind(&self) -> ResourceKind {
        self.kind
    }

    /// 適用した時間上限（メモリ超過の記録でも、同時に適用していた時間上限）。
    #[must_use]
    pub const fn limit(&self) -> Duration {
        self.limit
    }

    /// 超過を観測するまでの経過時間。
    #[must_use]
    pub const fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// 子プロセスを回収済みか。
    #[must_use]
    pub const fn child_reaped(&self) -> bool {
        self.child_reaped
    }

    /// 機械可読な理由コード。
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self.kind {
            ResourceKind::Time => "time_limit_exceeded",
            ResourceKind::Memory => "memory_limit_exceeded",
        }
    }

    /// 対応する終了コード（常に `limit_exceeded`=20。REQ-21）。
    #[must_use]
    pub const fn exit_code(&self) -> ExitCode {
        ExitCode::LimitExceeded
    }
}

/// 子の出力（読み取り上限つき）。`Debug` は長さだけを出し、本文を出さない。
#[derive(Clone, PartialEq, Eq)]
pub struct ChildOutput {
    stdout: Vec<u8>,
    stdout_truncated: bool,
    stderr: Vec<u8>,
    stderr_truncated: bool,
}

impl ChildOutput {
    /// stdout（上限まで）。
    #[must_use]
    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }

    /// stdout が上限で切られたか。
    #[must_use]
    pub const fn stdout_truncated(&self) -> bool {
        self.stdout_truncated
    }

    /// stderr（上限まで）。
    #[must_use]
    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }

    /// stderr が上限で切られたか。
    #[must_use]
    pub const fn stderr_truncated(&self) -> bool {
        self.stderr_truncated
    }
}

impl fmt::Debug for ChildOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChildOutput")
            .field("stdout_len", &self.stdout.len())
            .field("stdout_truncated", &self.stdout_truncated)
            .field("stderr_len", &self.stderr.len())
            .field("stderr_truncated", &self.stderr_truncated)
            .finish()
    }
}

/// 実行の結果。
#[derive(Debug)]
#[non_exhaustive]
pub enum GuardedRunOutcome {
    /// 期限内に子が終了した（終了コードが非 0 でもこちら）。
    Exited {
        /// 子の終了状態。
        status: ExitStatus,
        /// 子の出力。
        output: ChildOutput,
        /// 経過時間。
        elapsed: Duration,
    },
    /// 時間上限またはメモリ上限を超えた（kill・回収済み、または期限後の完了を観測）。
    LimitExceeded(ResourceLimitExceeded),
}

/// runner 自体の失敗。`Ok` に丸めない。パス・本文を含まない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GuardRunError {
    /// program が絶対パスでない。
    InvalidProgram,
    /// 時刻計算のオーバーフローなど設定が不正。
    InvalidConfig,
    /// 子の起動に失敗。
    Spawn,
    /// 子の状態取得に失敗（子は kill・回収済み）。
    Wait,
    /// 期限後の kill に失敗。
    KillFailed,
    /// kill 後の回収を確認できなかった。
    ReapTimeout,
    /// 子の出力の読み取りに失敗した（欠けた出力を正常終了として返さない）。
    ReadOutput,
    /// RSS の計測に失敗した（子は kill・回収済み。計測できないまま成功扱いにしない）。
    MemoryProbe,
    /// RSS の計測手段が無い OS でメモリ上限が指定された（起動前に拒否する）。
    MemoryLimitUnsupported,
}

impl GuardRunError {
    /// 機械可読な理由コード。
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::InvalidProgram => "invalid_program",
            Self::InvalidConfig => "invalid_config",
            Self::Spawn => "spawn_failed",
            Self::Wait => "wait_failed",
            Self::KillFailed => "kill_failed",
            Self::ReapTimeout => "reap_timeout",
            Self::ReadOutput => "read_output_failed",
            Self::MemoryProbe => "memory_probe_failed",
            Self::MemoryLimitUnsupported => "memory_limit_unsupported",
        }
    }

    /// 対応する終了コード（REQ-21）。
    #[must_use]
    pub const fn exit_code(self) -> ExitCode {
        match self {
            Self::InvalidProgram | Self::InvalidConfig => ExitCode::InvalidInput,
            Self::Spawn
            | Self::Wait
            | Self::KillFailed
            | Self::ReapTimeout
            | Self::ReadOutput
            | Self::MemoryProbe
            | Self::MemoryLimitUnsupported => ExitCode::RuntimeError,
        }
    }
}

impl fmt::Display for GuardRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "guarded run failed: {}", self.code())
    }
}

impl std::error::Error for GuardRunError {}

/// 起動対象。program は絶対パス必須。
#[derive(Clone, PartialEq, Eq)]
pub struct GuardedCommand {
    program: PathBuf,
    args: Vec<OsString>,
    current_dir: Option<PathBuf>,
    envs: Vec<(OsString, OsString)>,
}

impl GuardedCommand {
    /// 絶対パスの program から作る（PATH 探索をしない）。
    ///
    /// # Errors
    /// 相対パスは [`GuardRunError::InvalidProgram`]。
    pub fn new(program: impl Into<PathBuf>) -> Result<Self, GuardRunError> {
        let program = program.into();
        if !program.is_absolute() {
            return Err(GuardRunError::InvalidProgram);
        }
        Ok(Self {
            program,
            args: Vec::new(),
            current_dir: None,
            envs: Vec::new(),
        })
    }

    /// 引数を 1 つ追加する（シェルを経由しない）。
    #[must_use]
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// 作業ディレクトリを指定する。
    #[must_use]
    pub fn current_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.current_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// 呼び出し側が明示する環境変数を追加する（許可リスト以外はこれだけが渡る）。
    #[must_use]
    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.envs.push((key.into(), value.into()));
        self
    }
}

impl fmt::Debug for GuardedCommand {
    // 引数・パス・環境変数にデータが含まれうるため個数だけを出す。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuardedCommand")
            .field("args_len", &self.args.len())
            .finish_non_exhaustive()
    }
}

/// 実行設定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunConfig {
    time_limit: TimeLimit,
    stdout_cap: usize,
    stderr_cap: usize,
    memory_limit: Option<MemoryLimit>,
}

impl RunConfig {
    /// 設定を検証して作る。メモリ上限は既定の暫定 2 GiB を有効にする（REQ-39。上限を迂回させない）。
    /// 上限なしは [`RunConfig::without_memory_limit`] による明示的な経路だけで得られる。
    ///
    /// # Errors
    /// cap が 0 または [`MAX_OUTPUT_CAP`] 超なら [`ResourceConfigError::OutputCapOutOfRange`]。
    pub fn new(
        time_limit: TimeLimit,
        stdout_cap: usize,
        stderr_cap: usize,
    ) -> Result<Self, ResourceConfigError> {
        for cap in [stdout_cap, stderr_cap] {
            if cap == 0 || cap > MAX_OUTPUT_CAP {
                return Err(ResourceConfigError::OutputCapOutOfRange);
            }
        }
        Ok(Self {
            time_limit,
            stdout_cap,
            stderr_cap,
            memory_limit: Some(MemoryLimit::infer_default()),
        })
    }

    /// メモリ（RSS）上限を無効にする明示的な経路（時間上限だけの検証用途）。通常の経路では使わない。
    /// 計測手段の無い OS でも起動前拒否を受けずに実行できるが、RSS は制限されない。
    #[must_use]
    pub const fn without_memory_limit(mut self) -> Self {
        self.memory_limit = None;
        self
    }

    /// メモリ（RSS）上限を付ける。計測手段の無い OS では [`run_with_limits`] が
    /// [`GuardRunError::MemoryLimitUnsupported`] を返す（fail-closed）。
    #[must_use]
    pub const fn with_memory_limit(mut self, limit: MemoryLimit) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    /// 時間上限。
    #[must_use]
    pub const fn time_limit(&self) -> Duration {
        self.time_limit.get()
    }

    /// メモリ上限（バイト）。無効なら `None`。
    #[must_use]
    pub const fn memory_limit(&self) -> Option<u64> {
        match self.memory_limit {
            Some(l) => Some(l.get()),
            None => None,
        }
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            time_limit: TimeLimit::infer_default(),
            stdout_cap: DEFAULT_STDOUT_CAP,
            stderr_cap: DEFAULT_STDERR_CAP,
            // 全 OS でメモリ上限を保持する。計測手段の無い OS では `run_with_limits` が
            // 起動前に `MemoryLimitUnsupported` で拒否する（上限なしで子を起動しない。fail-closed）。
            memory_limit: Some(MemoryLimit::infer_default()),
        }
    }
}

/// 子に引き継ぐ環境変数の許可リスト（REQ-38。train の `ENV_ALLOWLIST` と同方針）。
#[cfg(unix)]
const ENV_ALLOWLIST: &[&str] = &["TMPDIR"];
#[cfg(windows)]
const ENV_ALLOWLIST: &[&str] = &["SystemRoot", "TEMP", "TMP"];
#[cfg(not(any(unix, windows)))]
const ENV_ALLOWLIST: &[&str] = &[];

/// 監視ロジックの対象（実プロセスと偽物を差し替える継ぎ目）。
pub(crate) trait ChildControl {
    type Status;
    /// OS のプロセス ID（RSS 計測用。未回収の間だけ有効）。
    fn id(&self) -> u32;
    fn try_wait(&mut self) -> io::Result<Option<Self::Status>>;
    fn kill(&mut self) -> io::Result<()>;
}

/// 時刻の継ぎ目。
pub(crate) trait Clock {
    fn now(&self) -> Instant;
    fn sleep(&self, d: Duration);
}

struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
    fn sleep(&self, d: Duration) {
        thread::sleep(d);
    }
}

impl ChildControl for std::process::Child {
    type Status = ExitStatus;
    fn id(&self) -> u32 {
        std::process::Child::id(self)
    }
    fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        std::process::Child::try_wait(self)
    }
    fn kill(&mut self) -> io::Result<()> {
        std::process::Child::kill(self)
    }
}

/// 監視の結果。
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum MonitorEnd<S> {
    Exited {
        status: S,
        elapsed: Duration,
    },
    TimedOut {
        elapsed: Duration,
        reaped: bool,
    },
    MemoryExceeded {
        elapsed: Duration,
        observed_rss_bytes: u64,
        reaped: bool,
    },
}

/// RSS 計測の失敗（理由は持たない。パス・pid を記録に出さないため）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProbeError;

/// 子プロセスの RSS 計測の継ぎ目（実計測と偽物を差し替える）。
///
/// 呼び出し側（`monitor`）は「直前の `try_wait` が未終了を返した直接の子」にだけ呼ぶ。
pub(crate) trait RssProbe {
    /// `Ok(Some(n))` は RSS（バイト）、`Ok(None)` はプロセス不在（次周回の `try_wait` に委ねる）。
    fn rss_bytes(&mut self, pid: u32) -> Result<Option<u64>, ProbeError>;
}

/// `/proc/<pid>/status` の `VmRSS:` 行（kB）をバイトにする。行が無ければ `Ok(None)`（zombie 等）。
#[cfg_attr(not(any(target_os = "linux", test)), allow(dead_code))]
pub(crate) fn parse_proc_status_vmrss(text: &str) -> Result<Option<u64>, ProbeError> {
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("VmRSS:") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let (Some(num), Some("kB"), None) = (parts.next(), parts.next(), parts.next()) else {
            return Err(ProbeError);
        };
        let kb: u64 = num.parse().map_err(|_| ProbeError)?;
        return kb.checked_mul(1024).map(Some).ok_or(ProbeError);
    }
    Ok(None)
}

/// `ps -o rss=` の出力（KiB）をバイトにする。空なら `Ok(None)`（プロセス不在）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
pub(crate) fn parse_ps_rss(text: &str) -> Result<Option<u64>, ProbeError> {
    let t = text.trim();
    if t.is_empty() {
        return Ok(None);
    }
    if !t.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ProbeError);
    }
    let kb: u64 = t.parse().map_err(|_| ProbeError)?;
    kb.checked_mul(1024).map(Some).ok_or(ProbeError)
}

/// `ps` の終了状態と出力から RSS 計測の結果を決める。
///
/// 成功終了は出力を解釈する（空ならプロセス不在）。`ps -p` は該当プロセスが無いとき
/// 空出力で終了コード 1 を返すため、その組だけを不在（`Ok(None)`）として扱う。
/// それ以外の非 0 終了・シグナル終了は計測失敗で、不在と区別して fail-closed にする。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
pub(crate) fn classify_ps_result(
    exit_code: Option<i32>,
    text: &str,
) -> Result<Option<u64>, ProbeError> {
    match exit_code {
        Some(0) => parse_ps_rss(text),
        Some(1) if text.trim().is_empty() => Ok(None),
        _ => Err(ProbeError),
    }
}

/// 実プロセスの RSS 計測。Linux は `/proc`、macOS は `/bin/ps`（PoC-20 と同じ）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
struct SystemRssProbe;

#[cfg(target_os = "linux")]
impl RssProbe for SystemRssProbe {
    fn rss_bytes(&mut self, pid: u32) -> Result<Option<u64>, ProbeError> {
        use std::io::Read;
        /// `/proc` のファイルは size 0 を報告するため fstat で検査できない。読み取り量で上限を課す。
        const STATUS_READ_CAP: u64 = 16 * 1024;
        let path = PathBuf::from("/proc").join(pid.to_string()).join("status");
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err(ProbeError),
        };
        let mut bytes = Vec::new();
        match file.take(STATUS_READ_CAP).read_to_end(&mut bytes) {
            Ok(_) => {}
            // 読み取り中にプロセスが消えた（ESRCH）。
            Err(e) if e.raw_os_error() == Some(3) => return Ok(None),
            Err(_) => return Err(ProbeError),
        }
        let text = std::str::from_utf8(&bytes).map_err(|_| ProbeError)?;
        parse_proc_status_vmrss(text)
    }
}

#[cfg(target_os = "macos")]
impl RssProbe for SystemRssProbe {
    fn rss_bytes(&mut self, pid: u32) -> Result<Option<u64>, ProbeError> {
        use std::io::Read;
        /// `ps` の待ちの上限。
        const PROBE_TIMEOUT: Duration = Duration::from_secs(1);
        /// `ps` の出力の読み取り上限（数字 1 つと改行だけのため十分）。
        const PS_READ_CAP: u64 = 64;
        let mut child = Command::new("/bin/ps")
            .args(["-o", "rss=", "-p"])
            .arg(pid.to_string())
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| ProbeError)?;
        let give_up = Instant::now() + PROBE_TIMEOUT;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {}
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(ProbeError);
                }
            }
            if Instant::now() >= give_up {
                let _ = child.kill();
                let _ = child.wait();
                return Err(ProbeError);
            }
            thread::sleep(Duration::from_millis(2));
        };
        let mut bytes = Vec::new();
        if let Some(out) = child.stdout.take() {
            out.take(PS_READ_CAP)
                .read_to_end(&mut bytes)
                .map_err(|_| ProbeError)?;
        }
        let text = std::str::from_utf8(&bytes).map_err(|_| ProbeError)?;
        classify_ps_result(status.code(), text)
    }
}

/// kill を送り、有界の間だけ回収を待つ。
fn kill_and_reap<C: ChildControl, K: Clock>(child: &mut C, clock: &K) -> Result<(), GuardRunError> {
    let give_up = clock
        .now()
        .checked_add(KILL_WAIT_TIMEOUT)
        .ok_or(GuardRunError::InvalidConfig)?;
    if child.kill().is_err() {
        // 直前に子が終了した競合では kill が失敗する。回収済みなら時間超過として扱う
        // （`limit_exceeded`=20 と `runtime_error`=70 を取り違えない。REQ-39・REQ-21）。
        // EINTR の再試行にも期限を設け、回収できなければ `KillFailed` を返す（無限ループを作らない）。
        return loop {
            match child.try_wait() {
                Ok(Some(_)) => break Ok(()),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {
                    if clock.now() >= give_up {
                        break Err(GuardRunError::KillFailed);
                    }
                }
                _ => break Err(GuardRunError::KillFailed),
            }
        };
    }
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Ok(()),
            Ok(None) => {}
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => return Err(GuardRunError::ReapTimeout),
        }
        if clock.now() >= give_up {
            return Err(GuardRunError::ReapTimeout);
        }
        clock.sleep(POLL_INTERVAL);
    }
}

/// 期限まで子を監視し、超過したら kill して回収する（メモリ監視なし。時間上限の単体テスト用）。
#[cfg(test)]
pub(crate) fn monitor<C: ChildControl, K: Clock>(
    child: &mut C,
    clock: &K,
    start: Instant,
    limit: Duration,
    pump: &mut dyn FnMut() -> bool,
) -> Result<MonitorEnd<C::Status>, GuardRunError> {
    monitor_with_memory(child, clock, start, limit, pump, None)
}

/// 期限まで子を監視し、時間またはメモリ（RSS）が超過したら kill して回収する。
///
/// `start` は子の起動前に取った時刻で、起動に要した時間も上限に含める（REQ-39）。
/// `pump` は毎周回で呼ばれ、パイプを読み進めて進捗があれば true を返す（進捗があれば
/// sleep を省き、子の書き込み詰まりを避ける）。`memory` が `Some` なら、未終了と確認した
/// 周回で `MEMORY_POLL_INTERVAL` ごとに RSS を計測し、上限を厳密に超えたら kill する。
pub(crate) fn monitor_with_memory<C: ChildControl, K: Clock>(
    child: &mut C,
    clock: &K,
    start: Instant,
    limit: Duration,
    pump: &mut dyn FnMut() -> bool,
    mut memory: Option<(&mut dyn RssProbe, MemoryLimit)>,
) -> Result<MonitorEnd<C::Status>, GuardRunError> {
    let deadline = start
        .checked_add(limit)
        .ok_or(GuardRunError::InvalidConfig)?;
    let mut last_probe: Option<Instant> = None;
    loop {
        let progressed = pump();
        match child.try_wait() {
            Ok(Some(status)) => {
                let now = clock.now();
                let elapsed = now.saturating_duration_since(start);
                if now > deadline {
                    // 期限後の完了は成功扱いにしない。回収済みのため kill は送らない。
                    return Ok(MonitorEnd::TimedOut {
                        elapsed,
                        reaped: true,
                    });
                }
                return Ok(MonitorEnd::Exited { status, elapsed });
            }
            Ok(None) => {
                let now = clock.now();
                if now >= deadline {
                    kill_and_reap(child, clock)?;
                    return Ok(MonitorEnd::TimedOut {
                        elapsed: clock.now().saturating_duration_since(start),
                        reaped: true,
                    });
                }
                if let Some((probe, mem_limit)) = memory.as_mut() {
                    let due = last_probe
                        .is_none_or(|t| now.saturating_duration_since(t) >= MEMORY_POLL_INTERVAL);
                    if due {
                        last_probe = Some(now);
                        // 直前の try_wait が未終了を返した直接の子だけに計測する（pid 再利用の防止）。
                        let probed = probe.rss_bytes(child.id());
                        // 計測（macOS の `ps` は最大 1 秒かかる）の間に期限へ達していたら、
                        // 計測結果によらず時間超過を優先する（REQ-39）。
                        if clock.now() >= deadline {
                            kill_and_reap(child, clock)?;
                            return Ok(MonitorEnd::TimedOut {
                                elapsed: clock.now().saturating_duration_since(start),
                                reaped: true,
                            });
                        }
                        match probed {
                            Ok(Some(rss)) if rss > mem_limit.get() => {
                                kill_and_reap(child, clock)?;
                                return Ok(MonitorEnd::MemoryExceeded {
                                    elapsed: clock.now().saturating_duration_since(start),
                                    observed_rss_bytes: rss,
                                    reaped: true,
                                });
                            }
                            Ok(_) => {}
                            Err(ProbeError) => {
                                kill_and_reap(child, clock)?;
                                return Err(GuardRunError::MemoryProbe);
                            }
                        }
                    }
                }
                if !progressed {
                    clock.sleep(POLL_INTERVAL.min(deadline.saturating_duration_since(now)));
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => {
                kill_and_reap(child, clock)?;
                return Err(GuardRunError::Wait);
            }
        }
    }
}

/// 上限つきで読む reader。Linux・macOS ではパイプを非ブロッキングにして監視ループの中で
/// 読む（スレッドを持たないため、子孫がパイプを保持していても期限超過時に何も残らない）。
/// 他 OS は読み取りスレッドで代替する（期限超過時は子孫がパイプを離すまでスレッドが残りうる）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod reader {
    use std::fs::File;
    use std::io::{self, Read};
    use std::os::fd::OwnedFd;
    use std::thread;
    use std::time::{Duration, Instant};

    /// 非ブロッキング化できる読み取り元。
    pub trait Source: Into<OwnedFd> {}
    impl<T: Into<OwnedFd>> Source for T {}

    /// `pump` 1 回あたりに読む最大バイト数。
    const PUMP_BUDGET_BYTES: usize = 64 * 1024;

    pub struct Reader {
        file: File,
        buf: Vec<u8>,
        cap: usize,
        truncated: bool,
        eof: bool,
        failed: bool,
    }

    fn errno_io(e: rustix::io::Errno) -> io::Error {
        io::Error::from_raw_os_error(e.raw_os_error())
    }

    impl Reader {
        pub fn new<R: Source>(src: R, cap: usize) -> io::Result<Self> {
            let fd: OwnedFd = src.into();
            let flags = rustix::fs::fcntl_getfl(&fd).map_err(errno_io)?;
            rustix::fs::fcntl_setfl(&fd, flags | rustix::fs::OFlags::NONBLOCK).map_err(errno_io)?;
            Ok(Self {
                file: File::from(fd),
                buf: Vec::new(),
                cap,
                truncated: false,
                eof: false,
                failed: false,
            })
        }

        /// 読める分だけ読む。進捗（1 バイト以上）があれば true。超過分は読み捨てて drain する。
        ///
        /// 1 回の呼び出しで読む量は `PUMP_BUDGET_BYTES` までに制限する。子が出力し続けても
        /// 必ず監視ループへ戻り、期限判定と kill が走るようにするため（REQ-39）。
        pub fn pump(&mut self) -> bool {
            let mut progressed = false;
            let mut scratch = [0u8; 8192];
            let mut budget = PUMP_BUDGET_BYTES;
            while !self.eof && budget > 0 {
                let n = match self.file.read(&mut scratch) {
                    Ok(0) => {
                        self.eof = true;
                        break;
                    }
                    Ok(n) => n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(_) => {
                        // 読み取り失敗は正常な EOF と区別して保持する（欠けた出力を成功扱いにしない）。
                        self.eof = true;
                        self.failed = true;
                        break;
                    }
                };
                progressed = true;
                budget = budget.saturating_sub(n);
                let take = self.cap.saturating_sub(self.buf.len()).min(n);
                if let Some(chunk) = scratch.get(..take) {
                    self.buf.extend_from_slice(chunk);
                }
                if take < n {
                    self.truncated = true;
                }
            }
            progressed
        }

        /// `deadline` まで EOF を待ち、出力・切り詰め・EOF 到達・読み取り失敗を返す。
        pub fn finish(mut self, deadline: Instant) -> super::Finished {
            loop {
                self.pump();
                let now = Instant::now();
                if self.eof || now >= deadline {
                    break;
                }
                thread::sleep(
                    Duration::from_millis(5).min(deadline.saturating_duration_since(now)),
                );
            }
            super::Finished {
                buf: self.buf,
                truncated: self.truncated || !self.eof,
                done: self.eof,
                failed: self.failed,
            }
        }
    }
    /// REQ-39: 読み取りエラーは正常な EOF と区別され、`failed` として保持される。
    #[cfg(test)]
    #[test]
    fn req39_read_error_is_reported_as_failed_not_eof() {
        // ディレクトリ fd への read は EISDIR で失敗する。
        let dir = File::open("/").unwrap();
        let reader = Reader::new(OwnedFd::from(dir), 1024).unwrap();
        let fin = reader.finish(Instant::now() + Duration::from_secs(1));
        assert!(fin.failed);
        assert!(fin.buf.is_empty());
    }

    /// REQ-39: 書き手が出力し続けても 1 回の `pump` は上限量で返る（監視ループへ制御を戻す）。
    #[cfg(test)]
    #[test]
    fn req39_pump_returns_after_bounded_read_under_continuous_output() {
        use std::io::Write;
        use std::os::unix::net::UnixStream;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let (mut writer, read_end) = UnixStream::pair().unwrap();
        writer.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_w = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let chunk = [b'x'; 8192];
            while !stop_w.load(Ordering::Relaxed) {
                let _ = writer.write(&chunk);
            }
        });
        // 書き手がソケットバッファを満たすまで待ち、1 回で読み切れない状態にする。
        thread::sleep(Duration::from_millis(200));
        let mut reader = Reader::new(read_end, usize::MAX).unwrap();
        assert!(reader.pump());
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
        // 上限量に達した時点で返る（最後の 1 読み分の超過のみ許容）。下限は環境（ソケット
        // バッファ長）に依存するため検証しない。
        assert!(
            reader.buf.len() < PUMP_BUDGET_BYTES + 8192,
            "read {}",
            reader.buf.len()
        );
        assert!(!reader.eof);
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod reader {
    use std::io::{self, Read};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Instant;

    pub trait Source: Read + Send + 'static {}
    impl<T: Read + Send + 'static> Source for T {}

    /// (出力, 切り詰め, 読み取り失敗)。
    type Captured = Arc<Mutex<(Vec<u8>, bool, bool)>>;

    pub struct Reader {
        captured: Captured,
        done: mpsc::Receiver<()>,
    }

    impl Reader {
        pub fn new<R: Source>(mut src: R, cap: usize) -> io::Result<Self> {
            let captured: Captured = Arc::new(Mutex::new((Vec::new(), false, false)));
            let shared = Arc::clone(&captured);
            let (tx, rx) = mpsc::channel();
            thread::spawn(move || {
                let mut scratch = [0u8; 8192];
                loop {
                    let n = match src.read(&mut scratch) {
                        Ok(0) => break,
                        Ok(n) => n,
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                        Err(_) => {
                            let mut guard = shared.lock().unwrap_or_else(|e| e.into_inner());
                            guard.2 = true;
                            break;
                        }
                    };
                    let mut guard = shared.lock().unwrap_or_else(|e| e.into_inner());
                    let take = cap.saturating_sub(guard.0.len()).min(n);
                    if let Some(chunk) = scratch.get(..take) {
                        guard.0.extend_from_slice(chunk);
                    }
                    if take < n {
                        guard.1 = true;
                    }
                }
                let _ = tx.send(());
            });
            Ok(Self { captured, done: rx })
        }

        pub fn pump(&mut self) -> bool {
            false
        }

        pub fn finish(self, deadline: Instant) -> super::Finished {
            let wait = deadline.saturating_duration_since(Instant::now());
            let finished = self.done.recv_timeout(wait).is_ok();
            let guard = self.captured.lock().unwrap_or_else(|e| e.into_inner());
            super::Finished {
                buf: guard.0.clone(),
                truncated: guard.1 || !finished,
                done: finished,
                failed: guard.2,
            }
        }
    }
}

use reader::Reader;

/// reader の終了結果。`failed` は読み取りエラー（正常な EOF ではない）。
pub(crate) struct Finished {
    pub buf: Vec<u8>,
    pub truncated: bool,
    pub done: bool,
    pub failed: bool,
}

impl Finished {
    /// reader が無い（パイプ未取得）場合の空の結果。
    fn empty() -> Self {
        Self {
            buf: Vec::new(),
            truncated: false,
            done: true,
            failed: false,
        }
    }
}

/// reader を作れなかった場合に、起動済みの子を止めて回収を試みてからエラーを返す。
///
/// 回収待ちは `KILL_WAIT_TIMEOUT` までに制限する（kill が効かない子でブロックしない。REQ-39）。
/// 停止・回収に失敗した場合は元の起動失敗ではなく [`GuardRunError::KillFailed`]・
/// [`GuardRunError::ReapTimeout`] を返し、子が残りうることを呼び出し側が判別できるようにする。
fn abort_child(child: &mut std::process::Child) -> Result<GuardedRunOutcome, GuardRunError> {
    abort_with(child, &SystemClock)
}

/// [`abort_child`] の本体。子の制御と時計を差し替えて単体テストできるよう分けている。
fn abort_with<C: ChildControl, K: Clock>(
    child: &mut C,
    clock: &K,
) -> Result<GuardedRunOutcome, GuardRunError> {
    kill_and_reap(child, clock)?;
    Err(GuardRunError::Spawn)
}

/// 子を起動し、時間上限とメモリ（RSS）上限を強制して実行する。
///
/// # Errors
/// 起動・待機・kill・回収・RSS 計測の失敗は [`GuardRunError`]。計測手段の無い OS でメモリ上限が
/// 有効なら、起動前に [`GuardRunError::MemoryLimitUnsupported`]。時間超過・メモリ超過は
/// エラーではなく [`GuardedRunOutcome::LimitExceeded`]。
pub fn run_with_limits(
    cmd: &GuardedCommand,
    config: &RunConfig,
) -> Result<GuardedRunOutcome, GuardRunError> {
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    if config.memory_limit.is_some() {
        return Err(GuardRunError::MemoryLimitUnsupported);
    }
    let mut command = Command::new(&cmd.program);
    command
        .args(&cmd.args)
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for key in ENV_ALLOWLIST {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    for (k, v) in &cmd.envs {
        command.env(k, v);
    }
    if let Some(dir) = &cmd.current_dir {
        command.current_dir(dir);
    }
    // 起動に要する時間も上限に含めるため、spawn の前から測る（REQ-39）。
    let start = Instant::now();
    let mut child = command.spawn().map_err(|_| GuardRunError::Spawn)?;
    let limit = config.time_limit.get();
    let spawn_deadline = start.checked_add(limit).ok_or(GuardRunError::InvalidConfig);
    let spawn_deadline = match spawn_deadline {
        Ok(d) => d,
        Err(e) => {
            let _ = kill_and_reap(&mut child, &SystemClock);
            return Err(e);
        }
    };
    let mut stdout = None;
    let mut stderr = None;
    // 起動後の reader 準備にも期限を適用する。超過したら子を止めて回収し時間超過として返す（REQ-39）。
    let expired =
        |child: &mut std::process::Child| -> Option<Result<GuardedRunOutcome, GuardRunError>> {
            let now = Instant::now();
            if now < spawn_deadline {
                return None;
            }
            Some(kill_and_reap(child, &SystemClock).map(|()| {
                GuardedRunOutcome::LimitExceeded(ResourceLimitExceeded::time(
                    limit,
                    now.saturating_duration_since(start),
                    true,
                ))
            }))
        };
    if let Some(r) = expired(&mut child) {
        return r;
    }
    if let Some(pipe) = child.stdout.take() {
        stdout = Reader::new(pipe, config.stdout_cap).ok();
        if stdout.is_none() {
            return abort_child(&mut child);
        }
        if let Some(r) = expired(&mut child) {
            return r;
        }
    }
    if let Some(pipe) = child.stderr.take() {
        stderr = Reader::new(pipe, config.stderr_cap).ok();
        if stderr.is_none() {
            return abort_child(&mut child);
        }
        if let Some(r) = expired(&mut child) {
            return r;
        }
    }

    let mut pump = || {
        let a = stdout.as_mut().is_some_and(Reader::pump);
        let b = stderr.as_mut().is_some_and(Reader::pump);
        a || b
    };
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    let mut probe = SystemRssProbe;
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    let memory = config
        .memory_limit
        .map(|l| (&mut probe as &mut dyn RssProbe, l));
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let memory: Option<(&mut dyn RssProbe, MemoryLimit)> = None;
    match monitor_with_memory(&mut child, &SystemClock, start, limit, &mut pump, memory)? {
        MonitorEnd::Exited { status, elapsed } => {
            // 終了後の出力待機も実行時間の上限に含める。両 reader で 1 つの期限を共有し、
            // 上限直前に終了した子の出力待ちで上限を超えたら時間超過として扱う（REQ-39）。
            let deadline = start
                .checked_add(limit)
                .ok_or(GuardRunError::InvalidConfig)?;
            let reader_deadline = Instant::now()
                .checked_add(READER_WAIT_TIMEOUT)
                .map_or(deadline, |d| d.min(deadline));
            let out = stdout.map_or_else(Finished::empty, |r| r.finish(reader_deadline));
            let err = stderr.map_or_else(Finished::empty, |r| r.finish(reader_deadline));
            let (stdout_done, stderr_done) = (out.done, err.done);
            let now = Instant::now();
            if now > deadline || (!(stdout_done && stderr_done) && now >= deadline) {
                return Ok(GuardedRunOutcome::LimitExceeded(
                    ResourceLimitExceeded::time(limit, now.saturating_duration_since(start), true),
                ));
            }
            // 判定の順序（REQ-39）: 1) 期限到達は時間超過を優先する。2) 期限前でも、読み取りエラー、
            // または子孫がパイプを保持して EOF に達しなかった（done が false）場合は出力が欠けている
            // ため、正常終了とせず `ReadOutput` を返す。3) それ以外だけを `Exited` とする。
            if out.failed || err.failed || !(stdout_done && stderr_done) {
                return Err(GuardRunError::ReadOutput);
            }
            Ok(GuardedRunOutcome::Exited {
                status,
                output: ChildOutput {
                    stdout: out.buf,
                    stdout_truncated: out.truncated,
                    stderr: err.buf,
                    stderr_truncated: err.truncated,
                },
                elapsed,
            })
        }
        MonitorEnd::TimedOut { elapsed, reaped } => {
            // reader は破棄する。Linux・macOS はスレッドを持たず fd を閉じるだけで何も残らない。
            // 出力は記録に含めない。
            Ok(GuardedRunOutcome::LimitExceeded(
                ResourceLimitExceeded::time(limit, elapsed, reaped),
            ))
        }
        MonitorEnd::MemoryExceeded {
            elapsed,
            observed_rss_bytes,
            reaped,
        } => {
            // 出力は記録に含めない。MemoryExceeded は memory_limit が Some の経路でのみ返る。
            let mem_limit = config
                .memory_limit
                .map_or(INFER_RSS_LIMIT_BYTES, MemoryLimit::get);
            Ok(GuardedRunOutcome::LimitExceeded(
                ResourceLimitExceeded::memory(
                    limit,
                    mem_limit,
                    observed_rss_bytes,
                    elapsed,
                    reaped,
                ),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;

    struct FakeClock {
        base: Instant,
        offset: Cell<Duration>,
    }

    impl FakeClock {
        fn new() -> Self {
            Self {
                base: Instant::now(),
                offset: Cell::new(Duration::ZERO),
            }
        }
    }

    impl Clock for FakeClock {
        fn now(&self) -> Instant {
            self.base + self.offset.get()
        }
        fn sleep(&self, d: Duration) {
            self.offset.set(self.offset.get() + d);
        }
    }

    struct FakeChild<'a> {
        clock: &'a FakeClock,
        script: RefCell<VecDeque<io::Result<Option<i32>>>>,
        // try_wait 1 回ごとに進める時間。
        step: Duration,
        kill_ok: bool,
        kill_calls: Cell<u32>,
    }

    impl ChildControl for FakeChild<'_> {
        type Status = i32;
        fn id(&self) -> u32 {
            1
        }
        fn try_wait(&mut self) -> io::Result<Option<i32>> {
            self.clock.sleep(self.step);
            self.script.borrow_mut().pop_front().unwrap_or(Ok(None))
        }
        fn kill(&mut self) -> io::Result<()> {
            self.kill_calls.set(self.kill_calls.get() + 1);
            if self.kill_ok {
                Ok(())
            } else {
                Err(io::Error::from(io::ErrorKind::PermissionDenied))
            }
        }
    }

    fn fake(
        clock: &FakeClock,
        script: Vec<io::Result<Option<i32>>>,
        step_ms: u64,
    ) -> FakeChild<'_> {
        FakeChild {
            clock,
            script: RefCell::new(script.into()),
            step: Duration::from_millis(step_ms),
            kill_ok: true,
            kill_calls: Cell::new(0),
        }
    }

    /// REQ-39: 期限後に完了を観測した場合は kill を送らず時間超過にする。
    #[test]
    fn req39_late_exit_is_time_limit_without_kill() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![Ok(Some(0))], 11_000);
        let end = monitor(
            &mut child,
            &clock,
            clock.now(),
            INFER_TIME_LIMIT,
            &mut || false,
        )
        .unwrap();
        assert!(matches!(end, MonitorEnd::TimedOut { reaped: true, .. }));
        assert_eq!(child.kill_calls.get(), 0);
    }

    /// REQ-39: 期限内の完了は Exited。
    #[test]
    fn req39_exit_within_limit_is_exited() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![Ok(None), Ok(Some(7))], 10);
        let end = monitor(
            &mut child,
            &clock,
            clock.now(),
            INFER_TIME_LIMIT,
            &mut || false,
        )
        .unwrap();
        assert!(matches!(end, MonitorEnd::Exited { status: 7, .. }));
    }

    /// REQ-39: EINTR は再試行する。
    #[test]
    fn req39_interrupted_try_wait_is_retried() {
        let clock = FakeClock::new();
        let mut child = fake(
            &clock,
            vec![
                Err(io::Error::from(io::ErrorKind::Interrupted)),
                Ok(Some(0)),
            ],
            1,
        );
        let end = monitor(
            &mut child,
            &clock,
            clock.now(),
            INFER_TIME_LIMIT,
            &mut || false,
        )
        .unwrap();
        assert!(matches!(end, MonitorEnd::Exited { status: 0, .. }));
    }

    /// REQ-39: 未終了のまま期限に達したら kill して回収する（kill は 1 回だけ）。
    #[test]
    fn req39_running_past_deadline_is_killed_and_reaped() {
        let clock = FakeClock::new();
        // 期限（300ms）まで Ok(None) が続き、kill 後の最初の try_wait で回収される。
        let mut script: Vec<io::Result<Option<i32>>> = (0..31).map(|_| Ok(None)).collect();
        script.push(Ok(Some(9)));
        let mut child = fake(&clock, script, 0);
        let end = monitor(
            &mut child,
            &clock,
            clock.now(),
            Duration::from_millis(300),
            &mut || false,
        )
        .unwrap();
        assert!(matches!(end, MonitorEnd::TimedOut { reaped: true, .. }));
        assert_eq!(child.kill_calls.get(), 1);
    }

    /// REQ-39: reader 初期化失敗後の停止に失敗したら、Spawn ではなく KillFailed を返す。
    #[test]
    fn req39_abort_reports_kill_failed_not_spawn() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![Ok(None)], 10);
        child.kill_ok = false;
        let err = abort_with(&mut child, &clock).unwrap_err();
        assert_eq!(err, GuardRunError::KillFailed);
    }

    /// REQ-39: reader 初期化失敗後に回収できなければ ReapTimeout を返す。
    #[test]
    fn req39_abort_reports_reap_timeout_not_spawn() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![], 10);
        let err = abort_with(&mut child, &clock).unwrap_err();
        assert_eq!(err, GuardRunError::ReapTimeout);
    }

    /// REQ-39: 回収できた場合のみ元の起動失敗（Spawn）を返す。
    #[test]
    fn req39_abort_returns_spawn_when_reaped() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![Ok(Some(0))], 10);
        let err = abort_with(&mut child, &clock).unwrap_err();
        assert_eq!(err, GuardRunError::Spawn);
        assert_eq!(child.kill_calls.get(), 1);
    }

    /// REQ-39: kill 失敗は成功を装わず KillFailed。
    #[test]
    fn req39_kill_failure_is_reported() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![], 20_000);
        child.kill_ok = false;
        let err = monitor(
            &mut child,
            &clock,
            clock.now(),
            INFER_TIME_LIMIT,
            &mut || false,
        )
        .unwrap_err();
        assert_eq!(err, GuardRunError::KillFailed);
        assert_eq!(err.exit_code(), ExitCode::RuntimeError);
    }

    /// REQ-39: kill 失敗後の try_wait が Interrupted を返し続けても、期限で打ち切って KillFailed。
    #[test]
    fn req39_kill_failure_with_endless_eintr_is_bounded() {
        let clock = FakeClock::new();
        let script = (0..10_000)
            .map(|_| Err(io::Error::from(io::ErrorKind::Interrupted)))
            .collect();
        let mut child = fake(&clock, script, 100);
        child.kill_ok = false;
        let err = kill_and_reap(&mut child, &clock).unwrap_err();
        assert_eq!(err, GuardRunError::KillFailed);
    }

    /// REQ-39・REQ-21: kill が競合で失敗しても、直後に回収済みなら時間超過（70 にしない）。
    #[test]
    fn req39_kill_race_with_exit_is_time_limit() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![Ok(None), Ok(Some(0))], 20_000);
        child.kill_ok = false;
        let end = monitor(
            &mut child,
            &clock,
            clock.now(),
            INFER_TIME_LIMIT,
            &mut || false,
        )
        .unwrap();
        assert!(matches!(end, MonitorEnd::TimedOut { reaped: true, .. }));
        assert_eq!(child.kill_calls.get(), 1);
    }

    /// REQ-39: 起動前から測った開始時刻を渡すと、起動に要した時間も上限に含まれる。
    #[test]
    fn req39_spawn_time_counts_toward_limit() {
        let clock = FakeClock::new();
        let start = clock.now();
        clock.sleep(Duration::from_secs(9)); // 起動に 9 秒かかった想定
        let mut child = fake(&clock, vec![], 0);
        let end = monitor(&mut child, &clock, start, INFER_TIME_LIMIT, &mut || false);
        // 残り 1 秒で kill を送るが、回収できない fake なので ReapTimeout になる。
        assert_eq!(end.unwrap_err(), GuardRunError::ReapTimeout);
        assert_eq!(child.kill_calls.get(), 1);
        // 起動後から 10 秒待たず、開始から約 10 秒（+ 回収待ち 2 秒）で見切る。
        assert!(clock.now().duration_since(start) < Duration::from_millis(12_500));
    }

    /// REQ-39: kill 後に回収できなければ ReapTimeout。
    #[test]
    fn req39_unreapable_child_is_reap_timeout() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![], 0);
        let err = monitor(
            &mut child,
            &clock,
            clock.now(),
            Duration::from_millis(50),
            &mut || false,
        )
        .unwrap_err();
        assert_eq!(err, GuardRunError::ReapTimeout);
    }

    /// REQ-39: try_wait の致命的失敗は kill・回収のうえ Wait。
    #[test]
    fn req39_wait_error_kills_and_reports_wait() {
        let clock = FakeClock::new();
        let mut child = fake(
            &clock,
            vec![
                Err(io::Error::from(io::ErrorKind::PermissionDenied)),
                Ok(Some(1)),
            ],
            1,
        );
        let err = monitor(
            &mut child,
            &clock,
            clock.now(),
            INFER_TIME_LIMIT,
            &mut || false,
        )
        .unwrap_err();
        assert_eq!(err, GuardRunError::Wait);
        assert_eq!(child.kill_calls.get(), 1);
    }

    /// REQ-39: 時間上限は 0 と最大値超を拒否し、最大値ちょうどは受理する。
    #[test]
    fn req39_time_limit_bounds() {
        assert_eq!(
            TimeLimit::new(Duration::ZERO),
            Err(ResourceConfigError::TimeLimitOutOfRange)
        );
        assert_eq!(
            TimeLimit::new(MAX_TIME_LIMIT + Duration::from_nanos(1)),
            Err(ResourceConfigError::TimeLimitOutOfRange)
        );
        assert_eq!(
            TimeLimit::new(MAX_TIME_LIMIT).unwrap().get(),
            MAX_TIME_LIMIT
        );
        assert_eq!(
            ResourceConfigError::TimeLimitOutOfRange.exit_code().code(),
            64
        );
    }

    /// REQ-39: cap は 0 と最大値超を拒否する。
    #[test]
    fn req39_output_cap_bounds() {
        let t = TimeLimit::infer_default();
        assert!(RunConfig::new(t, 0, 1).is_err());
        assert!(RunConfig::new(t, 1, MAX_OUTPUT_CAP + 1).is_err());
        assert!(RunConfig::new(t, MAX_OUTPUT_CAP, 1).is_ok());
    }

    /// REQ-39: 相対パスの program は InvalidProgram（64）。
    #[test]
    fn req39_relative_program_is_rejected() {
        let err = GuardedCommand::new("sleep").unwrap_err();
        assert_eq!(err, GuardRunError::InvalidProgram);
        assert_eq!(err.exit_code().code(), 64);
    }

    /// REQ-39・REQ-21: 超過の記録は `time_limit_exceeded`・終了コード 20。
    #[test]
    fn req39_limit_exceeded_record_is_time_and_20() {
        let rec = ResourceLimitExceeded::time(INFER_TIME_LIMIT, Duration::from_secs(10), true);
        assert_eq!(rec.code(), "time_limit_exceeded");
        assert_eq!(rec.exit_code().code(), 20);
    }

    /// RSS 計測の偽物。呼び出しごとに script の先頭を返し、尽きたら最後の値を返し続ける。
    struct FakeProbe {
        script: VecDeque<Result<Option<u64>, ProbeError>>,
        calls: u32,
    }

    impl FakeProbe {
        fn new(script: Vec<Result<Option<u64>, ProbeError>>) -> Self {
            Self {
                script: script.into(),
                calls: 0,
            }
        }
    }

    impl RssProbe for FakeProbe {
        fn rss_bytes(&mut self, _pid: u32) -> Result<Option<u64>, ProbeError> {
            self.calls += 1;
            if self.script.len() > 1 {
                self.script.pop_front().unwrap_or(Ok(None))
            } else {
                self.script.front().copied().unwrap_or(Ok(None))
            }
        }
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    fn run_mem(
        clock: &FakeClock,
        child: &mut FakeChild<'_>,
        probe: &mut FakeProbe,
        limit: Duration,
    ) -> Result<MonitorEnd<i32>, GuardRunError> {
        monitor_with_memory(
            child,
            clock,
            clock.now(),
            limit,
            &mut || false,
            Some((probe, MemoryLimit::infer_default())),
        )
    }

    /// REQ-39・TASK-39.5-2: 上限 + 1 バイトで kill が 1 回、観測 RSS が記録される。
    #[test]
    fn req39_rss_over_limit_kills_once_and_records_memory() {
        let clock = FakeClock::new();
        let mut script: Vec<io::Result<Option<i32>>> = (0..10).map(|_| Ok(None)).collect();
        script.push(Ok(Some(9)));
        let mut child = fake(&clock, script, 0);
        let mut probe = FakeProbe::new(vec![Ok(Some(GIB)), Ok(Some(2 * GIB + 1))]);
        let end = run_mem(&clock, &mut child, &mut probe, INFER_TIME_LIMIT).unwrap();
        assert!(matches!(
            end,
            MonitorEnd::MemoryExceeded {
                observed_rss_bytes: 2_147_483_649,
                reaped: true,
                ..
            }
        ));
        assert_eq!(child.kill_calls.get(), 1);
    }

    /// REQ-39: 上限ちょうどは超過ではない（厳密な「超」。PoC-20）。
    #[test]
    fn req39_rss_equal_to_limit_is_not_exceeded() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![Ok(None), Ok(None), Ok(Some(0))], 60);
        let mut probe = FakeProbe::new(vec![Ok(Some(2 * GIB))]);
        let end = run_mem(&clock, &mut child, &mut probe, INFER_TIME_LIMIT).unwrap();
        assert!(matches!(end, MonitorEnd::Exited { status: 0, .. }));
        assert_eq!(child.kill_calls.get(), 0);
        assert!(probe.calls >= 1);
    }

    /// REQ-39: プロセス不在（`Ok(None)`）は継続し、次周回の終了観測に委ねる。
    #[test]
    fn req39_rss_probe_none_continues() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![Ok(None), Ok(None), Ok(Some(3))], 60);
        let mut probe = FakeProbe::new(vec![Ok(None)]);
        let end = run_mem(&clock, &mut child, &mut probe, INFER_TIME_LIMIT).unwrap();
        assert!(matches!(end, MonitorEnd::Exited { status: 3, .. }));
    }

    /// REQ-39: 計測失敗は fail-closed（kill・回収のうえ `MemoryProbe`=70）。
    #[test]
    fn req39_rss_probe_error_kills_and_reports_memory_probe() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![Ok(None), Ok(Some(9))], 1);
        let mut probe = FakeProbe::new(vec![Err(ProbeError)]);
        let err = run_mem(&clock, &mut child, &mut probe, INFER_TIME_LIMIT).unwrap_err();
        assert_eq!(err, GuardRunError::MemoryProbe);
        assert_eq!(err.exit_code().code(), 70);
        assert_eq!(err.code(), "memory_probe_failed");
        assert_eq!(child.kill_calls.get(), 1);
    }

    /// REQ-39: 計測間隔は 50 ms（経過時間 / 50 ms + 1 回以下）。
    #[test]
    fn req39_rss_poll_interval_is_50ms() {
        let clock = FakeClock::new();
        // 約 1 秒（10 ms 刻み 100 周）経過してから終了する。
        let mut script: Vec<io::Result<Option<i32>>> = (0..99).map(|_| Ok(None)).collect();
        script.push(Ok(Some(0)));
        let mut child = fake(&clock, script, 0);
        let mut probe = FakeProbe::new(vec![Ok(Some(GIB))]);
        let end = run_mem(&clock, &mut child, &mut probe, INFER_TIME_LIMIT).unwrap();
        assert!(matches!(end, MonitorEnd::Exited { .. }));
        assert!(probe.calls >= 2, "calls {}", probe.calls);
        assert!(probe.calls <= 1000 / 50 + 1, "calls {}", probe.calls);
    }

    /// REQ-39: 期限到達と同時に RSS が超過しても時間超過を優先する。
    #[test]
    fn req39_time_limit_takes_precedence_over_memory() {
        let clock = FakeClock::new();
        let mut child = fake(&clock, vec![Ok(None), Ok(Some(9))], 20_000);
        let mut probe = FakeProbe::new(vec![Ok(Some(3 * GIB))]);
        let end = run_mem(&clock, &mut child, &mut probe, INFER_TIME_LIMIT).unwrap();
        assert!(matches!(end, MonitorEnd::TimedOut { reaped: true, .. }));
        assert_eq!(probe.calls, 0);
    }

    /// 計測中に時計を進める RSS 計測の偽物（`ps` の遅延を模す）。
    struct SlowProbe<'a> {
        clock: &'a FakeClock,
        delay: Duration,
        result: Result<Option<u64>, ProbeError>,
    }

    impl RssProbe for SlowProbe<'_> {
        fn rss_bytes(&mut self, _pid: u32) -> Result<Option<u64>, ProbeError> {
            self.clock.sleep(self.delay);
            self.result
        }
    }

    /// REQ-39: 計測の最中に期限へ達し、RSS も超過していても時間超過を優先する。
    #[test]
    fn req39_deadline_reached_during_probe_is_time_limit() {
        for result in [Ok(Some(3 * GIB)), Err(ProbeError)] {
            let clock = FakeClock::new();
            let mut child = fake(&clock, vec![Ok(None), Ok(Some(9))], 0);
            let mut probe = SlowProbe {
                clock: &clock,
                delay: INFER_TIME_LIMIT + Duration::from_secs(1),
                result,
            };
            let end = monitor_with_memory(
                &mut child,
                &clock,
                clock.now(),
                INFER_TIME_LIMIT,
                &mut || false,
                Some((&mut probe, MemoryLimit::infer_default())),
            )
            .unwrap();
            assert!(matches!(end, MonitorEnd::TimedOut { reaped: true, .. }));
            assert_eq!(child.kill_calls.get(), 1);
        }
    }

    /// REQ-39: `ps` の終了状態を見て、不在（空出力・終了コード 1）と計測失敗を区別する。
    #[test]
    fn req39_classify_ps_result() {
        assert_eq!(classify_ps_result(Some(0), "  2048\n"), Ok(Some(2_097_152)));
        assert_eq!(classify_ps_result(Some(0), ""), Ok(None));
        assert_eq!(classify_ps_result(Some(1), ""), Ok(None));
        assert_eq!(classify_ps_result(Some(1), "2048\n"), Err(ProbeError));
        assert_eq!(classify_ps_result(Some(2), ""), Err(ProbeError));
        assert_eq!(classify_ps_result(None, ""), Err(ProbeError));
    }

    /// REQ-39: メモリ上限は 0 と暫定上限超を拒否し（64）、ちょうどは受理する。
    #[test]
    fn req39_memory_limit_bounds() {
        assert_eq!(
            MemoryLimit::new(0),
            Err(ResourceConfigError::MemoryLimitOutOfRange)
        );
        assert_eq!(
            MemoryLimit::new(INFER_RSS_LIMIT_BYTES + 1),
            Err(ResourceConfigError::MemoryLimitOutOfRange)
        );
        assert_eq!(
            MemoryLimit::new(INFER_RSS_LIMIT_BYTES).unwrap().get(),
            2_147_483_648
        );
        assert_eq!(MemoryLimit::new(1).unwrap().get(), 1);
        assert_eq!(
            ResourceConfigError::MemoryLimitOutOfRange.code(),
            "memory_limit_out_of_range"
        );
        assert_eq!(
            ResourceConfigError::MemoryLimitOutOfRange
                .exit_code()
                .code(),
            64
        );
    }

    /// REQ-39: 既定設定・`RunConfig::new` は全 OS で 2 GiB のメモリ上限を持つ。上限なしは
    /// `without_memory_limit` の明示的な経路だけ（計測不能 OS では起動前拒否に倒れる）。
    #[test]
    fn req39_default_config_has_2gib_memory_limit() {
        assert_eq!(RunConfig::default().memory_limit(), Some(2_147_483_648));
        let t = TimeLimit::infer_default();
        assert_eq!(
            RunConfig::new(t, 1, 1).unwrap().memory_limit(),
            Some(2_147_483_648)
        );
        assert_eq!(
            RunConfig::new(t, 1, 1)
                .unwrap()
                .without_memory_limit()
                .memory_limit(),
            None
        );
        let with = RunConfig::new(t, 1, 1)
            .unwrap()
            .with_memory_limit(MemoryLimit::new(64).unwrap());
        assert_eq!(with.memory_limit(), Some(64));
    }

    /// REQ-39: `/proc/<pid>/status` の VmRSS 行（kB）をバイトにする。
    #[test]
    fn req39_parse_proc_status_vmrss() {
        assert_eq!(
            parse_proc_status_vmrss("Name:\tx\nVmRSS:\t  2048 kB\nThreads:\t1\n"),
            Ok(Some(2_097_152))
        );
        assert_eq!(parse_proc_status_vmrss("Name:\tx\nState:\tZ\n"), Ok(None));
        assert_eq!(parse_proc_status_vmrss("VmRSS:\tabc kB\n"), Err(ProbeError));
        assert_eq!(parse_proc_status_vmrss("VmRSS:\t10 MB\n"), Err(ProbeError));
        assert_eq!(parse_proc_status_vmrss("VmRSS:\t10\n"), Err(ProbeError));
        // u64 * 1024 がオーバーフローする値。
        assert_eq!(
            parse_proc_status_vmrss("VmRSS:\t18446744073709551615 kB\n"),
            Err(ProbeError)
        );
    }

    /// REQ-39: `ps -o rss=` の出力（KiB）をバイトにする。
    #[test]
    fn req39_parse_ps_rss() {
        assert_eq!(parse_ps_rss("  2048\n"), Ok(Some(2_097_152)));
        assert_eq!(parse_ps_rss(""), Ok(None));
        assert_eq!(parse_ps_rss("\n"), Ok(None));
        assert_eq!(parse_ps_rss("abc"), Err(ProbeError));
        assert_eq!(parse_ps_rss("12 34"), Err(ProbeError));
        assert_eq!(parse_ps_rss("18446744073709551615"), Err(ProbeError));
    }

    /// REQ-39・REQ-21: メモリ超過の記録は `memory_limit_exceeded`・終了コード 20で、数値を保持する。
    #[test]
    fn req39_memory_record_is_memory_and_20() {
        let rec = ResourceLimitExceeded::memory(
            INFER_TIME_LIMIT,
            INFER_RSS_LIMIT_BYTES,
            2_190_000_000,
            Duration::from_secs(1),
            true,
        );
        assert_eq!(rec.kind(), ResourceKind::Memory);
        assert_eq!(rec.code(), "memory_limit_exceeded");
        assert_eq!(rec.exit_code().code(), 20);
        assert_eq!(rec.memory_limit_bytes(), Some(2_147_483_648));
        assert_eq!(rec.observed_rss_bytes(), Some(2_190_000_000));
        assert_eq!(rec.limit(), INFER_TIME_LIMIT);
        let time = ResourceLimitExceeded::time(INFER_TIME_LIMIT, Duration::from_secs(10), true);
        assert_eq!(time.memory_limit_bytes(), None);
        assert_eq!(time.observed_rss_bytes(), None);
    }

    /// REQ-39: 新しい runner の失敗コードは 70 の固定語彙。
    #[test]
    fn req39_memory_run_errors_are_runtime_error() {
        assert_eq!(
            GuardRunError::MemoryProbe.exit_code(),
            ExitCode::RuntimeError
        );
        assert_eq!(
            GuardRunError::MemoryLimitUnsupported.code(),
            "memory_limit_unsupported"
        );
        assert_eq!(
            GuardRunError::MemoryLimitUnsupported.exit_code(),
            ExitCode::RuntimeError
        );
    }
}
