//! 資源の上限のうち実行時間（推論 1 件あたり暫定 10 秒）をプロセス境界で強制する（REQ-39・TASK-39.5-1・#170）。
//!
//! REQ-39 異常系は、推論 1 件の実行時間が暫定上限（10 秒。推論経路向けで、学習ワーカーの
//! 上限とは別）を超えたらプロセスを強制終了し、資源上限の超過として区別できる形で記録する
//! ことを求める。PoC-20 ケース 3 の実測は `subprocess.run(timeout=...)` による外側からの
//! 強制終了だった。`fandhe-edge-runtime` の期限（`infer_one_within` 等）はプロセス内の協調的な
//! もので止まった 1 件を中断できないため、強制的な上限は本モジュールが子プロセスの境界で担う。
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
//! - 記録・エラーに入力本文・子の出力・パスを含めない（固定語彙のみ）
//!
//! # 制限
//!
//! - 対象は直接の子だけで、子孫プロセスは kill しない。`fandhe-edge infer` は子プロセスを
//!   起動しない（REQ-32）ため受け入れる。子孫の管理にはプロセスグループ（`unsafe`・新規依存）が
//!   要り、ユーザー承認事項のため行わない
//! - Linux・macOS 以外の OS は読み取りスレッドで代替し、期限超過時は子孫がパイプを離すまで
//!   スレッドが残りうる
//! - 実装は全 OS でビルドされるが、検証環境は Mac のみ（Windows は実機検証の対象外）
//! - メモリ（RSS）上限は #171、ファイルサイズ上限は #172 の範囲。[`ResourceKind`] へ追加する

use fandhe_edge_core::exitcode::ExitCode;
use std::ffi::OsString;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// 推論 1 件の実行時間の暫定上限（REQ-39。PoC-20 の C-16・`guard.py` の `INFER_TIMEOUT_SEC`）。
pub const INFER_TIME_LIMIT: Duration = Duration::from_secs(10);
/// 指定できる時間上限の最大値（1 時間。runtime の `MAX_LATENCY_TIMEOUT_NS` と同値）。
pub const MAX_TIME_LIMIT: Duration = Duration::from_secs(3600);
/// stdout の既定の読み取り上限（推論 1 件の JSON は小さい）。
pub const DEFAULT_STDOUT_CAP: usize = 1024 * 1024;
/// stderr の既定の読み取り上限。
pub const DEFAULT_STDERR_CAP: usize = 64 * 1024;
/// 読み取り上限の最大値（CLI の `MAX_INFER_BATCH_OUTPUT_BYTES` と同値）。
pub const MAX_OUTPUT_CAP: usize = 256 * 1024 * 1024;

/// 監視のポーリング間隔（busy-spin を避ける）。
const POLL_INTERVAL: Duration = Duration::from_millis(10);
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
}

impl ResourceConfigError {
    /// 機械可読な理由コード。
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::TimeLimitOutOfRange => "time_limit_out_of_range",
            Self::OutputCapOutOfRange => "output_cap_out_of_range",
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

/// 超過した資源の種類。#171 で `Memory` を追加する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResourceKind {
    /// 実行時間（壁時計）。
    Time,
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
        }
    }

    /// 超過した資源の種類。
    #[must_use]
    pub const fn kind(&self) -> ResourceKind {
        self.kind
    }

    /// 適用した上限。
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
    /// 時間上限を超えた（kill・回収済み、または期限後の完了を観測）。
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
        }
    }

    /// 対応する終了コード（REQ-21）。
    #[must_use]
    pub const fn exit_code(self) -> ExitCode {
        match self {
            Self::InvalidProgram | Self::InvalidConfig => ExitCode::InvalidInput,
            Self::Spawn | Self::Wait | Self::KillFailed | Self::ReapTimeout => {
                ExitCode::RuntimeError
            }
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
}

impl RunConfig {
    /// 設定を検証して作る。
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
        })
    }

    /// 時間上限。
    #[must_use]
    pub const fn time_limit(&self) -> Duration {
        self.time_limit.get()
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            time_limit: TimeLimit::infer_default(),
            stdout_cap: DEFAULT_STDOUT_CAP,
            stderr_cap: DEFAULT_STDERR_CAP,
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
    Exited { status: S, elapsed: Duration },
    TimedOut { elapsed: Duration, reaped: bool },
}

/// kill を送り、有界の間だけ回収を待つ。
fn kill_and_reap<C: ChildControl, K: Clock>(child: &mut C, clock: &K) -> Result<(), GuardRunError> {
    if child.kill().is_err() {
        // 直前に子が終了した競合では kill が失敗する。回収済みなら時間超過として扱う
        // （`limit_exceeded`=20 と `runtime_error`=70 を取り違えない。REQ-39・REQ-21）。
        return loop {
            match child.try_wait() {
                Ok(Some(_)) => break Ok(()),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                _ => break Err(GuardRunError::KillFailed),
            }
        };
    }
    let give_up = clock
        .now()
        .checked_add(KILL_WAIT_TIMEOUT)
        .ok_or(GuardRunError::InvalidConfig)?;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Ok(()),
            Ok(None) => {}
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(GuardRunError::ReapTimeout),
        }
        if clock.now() >= give_up {
            return Err(GuardRunError::ReapTimeout);
        }
        clock.sleep(POLL_INTERVAL);
    }
}

/// 期限まで子を監視し、超過したら kill して回収する。
///
/// `start` は子の起動前に取った時刻で、起動に要した時間も上限に含める（REQ-39）。
/// `pump` は毎周回で呼ばれ、パイプを読み進めて進捗があれば true を返す（進捗があれば
/// sleep を省き、子の書き込み詰まりを避ける）。
pub(crate) fn monitor<C: ChildControl, K: Clock>(
    child: &mut C,
    clock: &K,
    start: Instant,
    limit: Duration,
    pump: &mut dyn FnMut() -> bool,
) -> Result<MonitorEnd<C::Status>, GuardRunError> {
    let deadline = start
        .checked_add(limit)
        .ok_or(GuardRunError::InvalidConfig)?;
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

    pub struct Reader {
        file: File,
        buf: Vec<u8>,
        cap: usize,
        truncated: bool,
        eof: bool,
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
            })
        }

        /// 読める分だけ読む。進捗（1 バイト以上）があれば true。超過分は読み捨てて drain する。
        pub fn pump(&mut self) -> bool {
            let mut progressed = false;
            let mut scratch = [0u8; 8192];
            while !self.eof {
                let n = match self.file.read(&mut scratch) {
                    Ok(0) => {
                        self.eof = true;
                        break;
                    }
                    Ok(n) => n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                    Err(_) => {
                        self.eof = true;
                        break;
                    }
                };
                progressed = true;
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

        /// 有界の間だけ EOF を待ち、それまでの分を返す。EOF が来なければ切り詰め扱い。
        pub fn finish(mut self, wait: Duration) -> (Vec<u8>, bool) {
            let give_up = Instant::now().checked_add(wait);
            loop {
                self.pump();
                if self.eof || give_up.is_none_or(|g| Instant::now() >= g) {
                    break;
                }
                thread::sleep(Duration::from_millis(5));
            }
            (self.buf, self.truncated || !self.eof)
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod reader {
    use std::io::{self, Read};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    pub trait Source: Read + Send + 'static {}
    impl<T: Read + Send + 'static> Source for T {}

    type Captured = Arc<Mutex<(Vec<u8>, bool)>>;

    pub struct Reader {
        captured: Captured,
        done: mpsc::Receiver<()>,
    }

    impl Reader {
        pub fn new<R: Source>(mut src: R, cap: usize) -> io::Result<Self> {
            let captured: Captured = Arc::new(Mutex::new((Vec::new(), false)));
            let shared = Arc::clone(&captured);
            let (tx, rx) = mpsc::channel();
            thread::spawn(move || {
                let mut scratch = [0u8; 8192];
                loop {
                    let n = match src.read(&mut scratch) {
                        Ok(0) => break,
                        Ok(n) => n,
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
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

        pub fn finish(self, wait: Duration) -> (Vec<u8>, bool) {
            let finished = self.done.recv_timeout(wait).is_ok();
            let guard = self.captured.lock().unwrap_or_else(|e| e.into_inner());
            (guard.0.clone(), guard.1 || !finished)
        }
    }
}

use reader::Reader;

/// reader を作れなかった場合に、起動済みの子を止めて回収してからエラーを返す。
fn abort_child(child: &mut std::process::Child) -> Result<GuardedRunOutcome, GuardRunError> {
    let _ = child.kill();
    let _ = child.wait();
    Err(GuardRunError::Spawn)
}

/// 子を起動し、時間上限を強制して実行する。
///
/// # Errors
/// 起動・待機・kill・回収の失敗は [`GuardRunError`]。時間超過はエラーではなく
/// [`GuardedRunOutcome::LimitExceeded`]。
pub fn run_with_limits(
    cmd: &GuardedCommand,
    config: &RunConfig,
) -> Result<GuardedRunOutcome, GuardRunError> {
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
    let mut stdout = None;
    let mut stderr = None;
    if let Some(pipe) = child.stdout.take() {
        stdout = Reader::new(pipe, config.stdout_cap).ok();
        if stdout.is_none() {
            return abort_child(&mut child);
        }
    }
    if let Some(pipe) = child.stderr.take() {
        stderr = Reader::new(pipe, config.stderr_cap).ok();
        if stderr.is_none() {
            return abort_child(&mut child);
        }
    }

    let limit = config.time_limit.get();
    let mut pump = || {
        let a = stdout.as_mut().is_some_and(Reader::pump);
        let b = stderr.as_mut().is_some_and(Reader::pump);
        a || b
    };
    match monitor(&mut child, &SystemClock, start, limit, &mut pump)? {
        MonitorEnd::Exited { status, elapsed } => {
            let (stdout, stdout_truncated) =
                stdout.map_or((Vec::new(), false), |r| r.finish(READER_WAIT_TIMEOUT));
            let (stderr, stderr_truncated) =
                stderr.map_or((Vec::new(), false), |r| r.finish(READER_WAIT_TIMEOUT));
            Ok(GuardedRunOutcome::Exited {
                status,
                output: ChildOutput {
                    stdout,
                    stdout_truncated,
                    stderr,
                    stderr_truncated,
                },
                elapsed,
            })
        }
        MonitorEnd::TimedOut { elapsed, reaped } => {
            // reader は破棄する。Linux・macOS はスレッドを持たず fd を閉じるだけで何も残らない。
            // 出力は記録に含めない。
            Ok(GuardedRunOutcome::LimitExceeded(ResourceLimitExceeded {
                kind: ResourceKind::Time,
                limit,
                elapsed,
                child_reaped: reaped,
            }))
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
        let rec = ResourceLimitExceeded {
            kind: ResourceKind::Time,
            limit: INFER_TIME_LIMIT,
            elapsed: Duration::from_secs(10),
            child_reaped: true,
        };
        assert_eq!(rec.code(), "time_limit_exceeded");
        assert_eq!(rec.exit_code().code(), 20);
    }
}
