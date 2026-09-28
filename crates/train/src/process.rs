//! 学習ワーカーの子プロセス起動・タイムアウト・終了コード写像（REQ-21・
//! REQ-34・REQ-39。issue #178）。
//!
//! Rust 側 CLI・ジョブ管理（未配線。TASK-33.x）が、学習ワーカー
//! （`trainer/launch.py`）を引数リストで子プロセス起動し、壁時計タイムアウト
//! と標準出力／標準エラー出力の読み取り上限を掛けたうえで、
//! [`crate::result::TrainOutcome::from_worker_stdout`] へ渡して結果を得る
//! ための型・関数を提供する（[`run_train`] が公開入口）。
//!
//! # supervisor との二重監視の関係
//!
//! `trainer/src/fandhe_edge_trainer/supervisor.py::run_supervised_train` は
//! 既に内側の監視者として、`_worker` を新しいセッション
//! （`start_new_session=True`）で起動し、`time_limit_seconds` 超過時に
//! プロセスグループ全体を `SIGKILL` する。本モジュールの [`RunLimits`] は
//! それとは独立した **外側** の壁時計締め切りで、supervisor 自身が固まった
//! 場合の保険として働く（[`crate::limits::SUPERVISOR_SHUTDOWN_GRACE_SECONDS`]
//! のドキュメントに根拠を記す）。
//!
//! # 孤児化の限界
//!
//! supervisor は `_worker` を別セッションで起動するため、本モジュールが
//! supervisor プロセス（`python -I launch.py train ...`）を kill しても、
//! 別セッションの `_worker` には SIGKILL が届かない。supervisor が Rust の
//! 外側締め切りより先に自分の内側締め切りで `_worker` を kill・回収する
//! 設計（[`crate::limits::SUPERVISOR_SHUTDOWN_GRACE_SECONDS`]）で緩和する
//! が、supervisor プロセス自体が応答不能になった極端なケースでは `_worker`
//! が孤児として残りうる。Rust 側だけでプロセスグループ単位の kill を行う
//! には依存追加（`nix`／`libc` 等）が要り、本 issue の対象外とする
//! （issue #178 実装計画 8 章）。
//!
//! # 推論ランタイムとの境界
//!
//! 本モジュールは学習ワーカー層（`crates/train`）に閉じる。推論ランタイム・
//! CLI コアはこの crate に依存しない（REQ-32。`.claude/rules/coding-rust.md`）。

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use fandhe_edge_core::exitcode::ExitCode;

use crate::error::TrainProcessError;
use crate::limits::{MAX_RESULT_BYTES, MAX_WORKER_STDERR_BYTES, SUPERVISOR_SHUTDOWN_GRACE_SECONDS};
use crate::request::TrainRequest;
use crate::result::TrainOutcome;

/// 出力読み取りスレッドが完了を待つ上限（秒）。子プロセスの終了後、
/// パイプが閉じてスレッドが `recv` から戻るまでの猶予（REQ-39「資源の
/// 上限」）。`_worker` が supervisor の標準エラー出力を継承しているため、
/// 孤児が残っているとパイプが閉じず読み取りが終わらないことへの対策
/// （issue #178 実装計画 3.5）。
const READER_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// `Child::try_wait()` のポーリング間隔（ミリ秒）。std に `wait_timeout`
/// 相当が無いため、短い間隔で非同期にポーリングする。
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// 子プロセスの環境変数の許可リスト（OS ごと）。`env_clear()` のうえで
/// これらのキーだけを、親プロセスに存在する場合に限り引き継ぐ。
///
/// unix: `TMPDIR` のみ（一時ファイルの配置先解決に使われうる）。
/// windows: 完全に空の環境は CRT・Python の DLL 探索を壊しうるため、
/// `SystemRoot`・`TEMP`・`TMP` を許可する（issue #178 実装計画 3.3 を
/// Windows でも安全に動くよう拡張。根拠: Python ランタイムは
/// Windows で `SystemRoot` が無いと DLL 探索に失敗しうる）。
#[cfg(unix)]
pub const ENV_ALLOWLIST: &[&str] = &["TMPDIR"];
#[cfg(windows)]
pub const ENV_ALLOWLIST: &[&str] = &["SystemRoot", "TEMP", "TMP"];

/// 学習ワーカーの起動口（`<python> -I <launch.py> train --request <path>`）
/// を表す検証済みの型（issue #178 実装計画 3.1）。
///
/// フィールドは非公開。[`WorkerLauncher::new`]／[`WorkerLauncher::from_trainer_dir`]
/// を経由した値だけが構築でき、絶対パスの通常ファイルであることが型で
/// 保証される（PATH 検索はしない。PATH 乗っ取り対策。
/// `.claude/rules/security.md`「インジェクション」）。
#[derive(Debug, Clone)]
pub struct WorkerLauncher {
    python: PathBuf,
    launch_script: PathBuf,
}

impl WorkerLauncher {
    /// `python`・`launch_script` を検証して [`WorkerLauncher`] を組み立てる。
    ///
    /// 検証: 両方とも絶対パスであること、`launch_script` のファイル名が
    /// `launch.py` であること（`trainer/launch.py` のモジュール doc「唯一の
    /// 起動口」契約を Rust 側でも守る）、`std::fs::metadata` でいずれも
    /// 通常ファイルであること。
    pub fn new(python: PathBuf, launch_script: PathBuf) -> Result<Self, TrainProcessError> {
        if !python.is_absolute() {
            return Err(TrainProcessError::InvalidLauncher { field: "python" });
        }
        if !launch_script.is_absolute() {
            return Err(TrainProcessError::InvalidLauncher {
                field: "launch_script",
            });
        }
        if launch_script.file_name().and_then(|n| n.to_str()) != Some("launch.py") {
            return Err(TrainProcessError::InvalidLauncher {
                field: "launch_script",
            });
        }
        let python_meta = std::fs::metadata(&python)
            .map_err(|_| TrainProcessError::InvalidLauncher { field: "python" })?;
        if !python_meta.is_file() {
            return Err(TrainProcessError::InvalidLauncher { field: "python" });
        }
        let launch_meta =
            std::fs::metadata(&launch_script).map_err(|_| TrainProcessError::InvalidLauncher {
                field: "launch_script",
            })?;
        if !launch_meta.is_file() {
            return Err(TrainProcessError::InvalidLauncher {
                field: "launch_script",
            });
        }
        Ok(Self {
            python,
            launch_script,
        })
    }

    /// `trainer_dir`（絶対パス）から venv の python と `launch.py` を導く
    /// 補助コンストラクタ。`trainer_dir` の発見（CLI 引数・環境変数からの
    /// 解決）自体は本 issue の対象外で、CLI 配線（TASK-33.x）が担う
    /// （issue #178 実装計画 3.1）。
    pub fn from_trainer_dir(trainer_dir: &Path) -> Result<Self, TrainProcessError> {
        if !trainer_dir.is_absolute() {
            return Err(TrainProcessError::InvalidLauncher {
                field: "trainer_dir",
            });
        }
        #[cfg(unix)]
        let python = trainer_dir.join(".venv").join("bin").join("python3");
        #[cfg(windows)]
        let python = trainer_dir.join(".venv").join("Scripts").join("python.exe");
        let launch_script = trainer_dir.join("launch.py");
        Self::new(python, launch_script)
    }

    /// このランチャーで起動する固定 argv（`Command::args` へそのまま渡す。
    /// シェルを介さない。`.claude/rules/security.md`「インジェクション」）。
    fn argv(&self, request_path: &Path) -> Vec<std::ffi::OsString> {
        vec![
            "-I".into(),
            self.launch_script.clone().into_os_string(),
            "train".into(),
            "--request".into(),
            request_path.to_path_buf().into_os_string(),
        ]
    }
}

/// 学習ジョブ 1 件あたりの壁時計締め切り（REQ-39「資源の上限」）。
///
/// 既定は [`TrainRequest::time_limit_seconds`] に
/// [`SUPERVISOR_SHUTDOWN_GRACE_SECONDS`] を足した値（[`RunLimits::for_request`]）。
/// [`RunLimits::with_wall_timeout`] は **締める方向だけ** を許す型で、
/// 呼び出し元が誤って上限を緩めることができないようにする（テストで
/// タイムアウトを短くする用途に使う）。
#[derive(Debug, Clone, Copy)]
pub struct RunLimits {
    wall_timeout: Duration,
}

impl RunLimits {
    /// `request.time_limit_seconds()` に基づく既定の外側締め切りを返す。
    #[must_use]
    pub fn for_request(request: &TrainRequest) -> Self {
        let seconds = u64::from(request.time_limit_seconds())
            .saturating_add(u64::from(SUPERVISOR_SHUTDOWN_GRACE_SECONDS));
        Self {
            wall_timeout: Duration::from_secs(seconds),
        }
    }

    /// 既定の締め切りより **短い** 値へ上書きする（テスト用途）。
    /// `timeout` が 0、または既定の締め切り以上の場合は `Err`（上限を緩める
    /// ことはできない）。
    pub fn with_wall_timeout(self, timeout: Duration) -> Result<Self, TrainProcessError> {
        if timeout.is_zero() || timeout >= self.wall_timeout {
            return Err(TrainProcessError::InvalidRunLimits);
        }
        Ok(Self {
            wall_timeout: timeout,
        })
    }

    #[must_use]
    pub fn wall_timeout(&self) -> Duration {
        self.wall_timeout
    }
}

/// `job_dir/request.json` を新規作成し、成功・失敗・タイムアウトの全経路で
/// 削除する Drop ガード（issue #178 実装計画 3.2）。
///
/// 削除失敗はエラーにしない（学習結果を失わせないための方針をコメントで
/// 明示する。`.claude/rules/code-comment-style.md`「非自明な前提・副作用」）。
#[derive(Debug)]
struct RequestFileGuard {
    path: PathBuf,
}

impl Drop for RequestFileGuard {
    fn drop(&mut self) {
        // 削除に失敗しても学習結果（`run_train` が既に得た `TrainRun`／
        // エラー）を失わせないため、ここでのエラーは無視する。ジョブ
        // ディレクトリの残置ファイルは呼び出し元の責務（次回同じ `job_dir`
        // を使う場合は本ガードの新規作成〔`create_new`〕が衝突を検出する）。
        let _ = std::fs::remove_file(&self.path);
    }
}

/// `job_dir`（既存の絶対パスディレクトリ）へ検証済みリクエストを書き込む。
/// 既存の `request.json` は上書きしない（`OpenOptions::create_new`）。
fn write_request_file(
    job_dir: &Path,
    request: &TrainRequest,
) -> Result<RequestFileGuard, TrainProcessError> {
    if !job_dir.is_absolute() {
        return Err(TrainProcessError::InvalidJobDir);
    }
    let metadata = std::fs::metadata(job_dir).map_err(|_| TrainProcessError::InvalidJobDir)?;
    if !metadata.is_dir() {
        return Err(TrainProcessError::InvalidJobDir);
    }
    let path = job_dir.join("request.json");
    // `TrainRequest::new` は `config` の値そのもののサイズを検査しない
    // （`config` は任意の JSON を保持しうる。`request.rs` のモジュール
    // doc 参照）ため、巨大な `config` を持つリクエストでは
    // `to_json_vec` が直列化上限超過（`TrainRequestError::TooLarge`）を
    // 返しうる（防御的な分岐ではなく到達しうる経路）。`TrainRequestError`
    // の終了コード（`TooLarge` は `LimitExceeded`）をそのまま尊重するため
    // `TrainProcessError::Request` へ委譲する（`?` は
    // `From<TrainRequestError>` 経由）。
    let bytes = request.to_json_vec()?;

    let mut open_options = std::fs::OpenOptions::new();
    open_options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        open_options.mode(0o600);
    }
    let mut file = open_options
        .open(&path)
        .map_err(|e| TrainProcessError::RequestWrite { kind: e.kind() })?;
    file.write_all(&bytes)
        .map_err(|e| TrainProcessError::RequestWrite { kind: e.kind() })?;
    file.sync_all()
        .map_err(|e| TrainProcessError::RequestWrite { kind: e.kind() })?;
    Ok(RequestFileGuard { path })
}

/// 上限つきで読み進める出力読み取りスレッドの結果。
struct DrainedOutput {
    /// 保持したバイト列（上限超過分は含まない）。
    kept: Vec<u8>,
    /// 上限を超えて読み捨てた分があるか。
    truncated: bool,
}

/// パイプから上限 `cap` バイトまで保持しつつ読み進める（超過分は読み捨てる
/// が、読み取り自体は EOF まで続ける。子プロセスがパイプ書き込みで
/// ブロックしないようにするため。`supervisor.py::_drain_stdout` と同じ
/// 理由）。別スレッドで実行する想定（[`spawn_reader`]）。
fn drain_capped<R: Read>(mut reader: R, cap: usize) -> DrainedOutput {
    let mut kept = Vec::new();
    let mut truncated = false;
    let mut buf = [0u8; 8192];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let chunk = &buf[..n];
                if kept.len() < cap {
                    let remaining = cap - kept.len();
                    let take = remaining.min(chunk.len());
                    kept.extend_from_slice(&chunk[..take]);
                    if take < chunk.len() {
                        truncated = true;
                    }
                } else {
                    truncated = true;
                }
            }
            Err(_) => break,
        }
    }
    DrainedOutput { kept, truncated }
}

/// パイプ読み取りを専用スレッドへ切り出し、`mpsc::Receiver` を返す
/// （呼び出し元は `recv_timeout` で上限付きに待つ）。
fn spawn_reader<R>(reader: R, cap: usize) -> mpsc::Receiver<DrainedOutput>
where
    R: Read + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let drained = drain_capped(reader, cap);
        // 受信側が既に諦めて `Receiver` を drop していても送信失敗は
        // 無視してよい（スレッドは自然に終了する）。
        let _ = tx.send(drained);
    });
    rx
}

/// 学習ワーカーの実行結果（[`run_train`] の成功時の戻り値）。
///
/// フィールドは非公開。ワーカーの標準エラー出力はアクセサ
/// [`TrainRun::worker_stderr`] 経由でのみ公開し、データ由来の文字列を
/// 含みうるためログ・エラーメッセージへ転記しないこと
/// （`.claude/rules/security.md`「秘密情報の混入防止」）。
#[derive(Debug)]
pub struct TrainRun {
    outcome: TrainOutcome,
    exit_code: ExitCode,
    elapsed: Duration,
    worker_stderr: Vec<u8>,
    stderr_truncated: bool,
}

impl TrainRun {
    #[must_use]
    pub fn outcome(&self) -> &TrainOutcome {
        &self.outcome
    }

    #[must_use]
    pub fn exit_code(&self) -> ExitCode {
        self.exit_code
    }

    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// 学習ワーカーの標準エラー出力（先頭から [`MAX_WORKER_STDERR_BYTES`]
    /// まで保持）。**データ由来の文字列を含みうるため、ログ・エラー
    /// メッセージへ転記しないこと**（呼び出し元の責務）。
    #[must_use]
    pub fn worker_stderr(&self) -> &[u8] {
        &self.worker_stderr
    }

    #[must_use]
    pub fn stderr_truncated(&self) -> bool {
        self.stderr_truncated
    }
}

/// [`Child::wait()`] の失敗を [`TrainProcessError::Wait`] へ写す。
fn wait_child(child: &mut Child) -> Result<ExitStatus, TrainProcessError> {
    child
        .wait()
        .map_err(|e| TrainProcessError::Wait { kind: e.kind() })
}

/// 子プロセスの終了コード（unix はシグナル終了で `None` になりうる）を
/// [`ExitCode`] へ写す。JSON 側の判定（`classify_result_exit_code`）とは
/// 独立に、まず「プロセスがどう終了したか」だけを判定する。
fn classify_process_exit_status(status: ExitStatus) -> Result<ExitCode, TrainProcessError> {
    let Some(code) = status.code() else {
        return Err(TrainProcessError::TerminatedBySignal);
    };
    ExitCode::try_from(code).map_err(|e| TrainProcessError::UnknownExitCode(e.0))
}

/// 標準出力（信頼しない外部入力）を検証済みの [`TrainOutcome`] へ解析し、
/// その結果が示す終了コードと、子プロセスの実終了コード（`process_exit`）
/// が一致することを確認する（issue #178 実装計画 3.6・表の 6〜8 行目）。
fn classify_exit(
    process_exit: ExitCode,
    stdout: &[u8],
    request: &TrainRequest,
) -> Result<TrainOutcome, TrainProcessError> {
    let outcome = TrainOutcome::from_worker_stdout(stdout, request)?;
    let expected = outcome.exit_code();
    if process_exit != expected {
        return Err(TrainProcessError::ExitCodeMismatch {
            process: process_exit,
            expected,
        });
    }
    Ok(outcome)
}

/// 学習ワーカーを子プロセスとして起動し、壁時計タイムアウトと出力の読み
/// 取り上限を掛けたうえで、検証済みの結果 [`TrainRun`] を得る（REQ-21・
/// REQ-34・REQ-39。issue #178 の公開入口）。
///
/// 手順: (1) `job_dir/request.json` を新規作成 → (2) 固定 argv・
/// `env_clear()`＋許可リストの環境・`current_dir(job_dir)` で子プロセスを
/// 起動 → (3) [`RunLimits`] の締め切りまで `try_wait()` をポーリング、
/// 超過時は kill して回収 → (4) 標準出力／標準エラー出力を上限つきで
/// 読み取り → (5) [`classify_exit`] で結果を検証。すべての経路
/// （成功・失敗・タイムアウト）で `request.json` を削除する
/// （[`RequestFileGuard`]）。
pub fn run_train(
    launcher: &WorkerLauncher,
    request: &TrainRequest,
    job_dir: &Path,
    limits: &RunLimits,
) -> Result<TrainRun, TrainProcessError> {
    let guard = write_request_file(job_dir, request)?;
    let started = Instant::now();

    let mut command = Command::new(&launcher.python);
    command
        .args(launcher.argv(&guard.path))
        .env_clear()
        .current_dir(job_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for key in ENV_ALLOWLIST {
        if let Ok(value) = std::env::var(key) {
            command.env(key, value);
        }
    }

    let mut child = command
        .spawn()
        .map_err(|e| TrainProcessError::Spawn { kind: e.kind() })?;

    // stdout・stderr の読み取りスレッドを起動する前にパイプを取り出す
    // （書き込み側を親プロセスが握ったままにしない。取り出し忘れると
    // `Child` が drop される際にパイプが閉じられず readers が終わらない
    // ことがある）。
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let stdout_rx = stdout_pipe.map(|pipe| spawn_reader(pipe, MAX_RESULT_BYTES + 1));
    let stderr_rx = stderr_pipe.map(|pipe| spawn_reader(pipe, MAX_WORKER_STDERR_BYTES));

    // 壁時計タイムアウトまで `try_wait()` をポーリングする（std に
    // `wait_timeout` 相当が無いため）。
    let deadline = started + limits.wall_timeout();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    break None;
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(e) => return Err(TrainProcessError::Wait { kind: e.kind() }),
        }
    };

    let status = match status {
        Some(status) => status,
        None => {
            // 締め切り超過: kill してから必ず `wait()` で回収する
            // （ゾンビを残さない）。kill 自体の失敗（既に終了済み等）は
            // 無視してよい。
            let _ = child.kill();
            wait_child(&mut child)?;
            return Err(TrainProcessError::WallTimeout {
                limit_ms: u64::try_from(limits.wall_timeout().as_millis()).unwrap_or(u64::MAX),
            });
        }
    };

    let elapsed = started.elapsed();

    // 子は既に終了しているため、パイプは EOF に達し reader スレッドは
    // 有限時間で終わるはずである。それでも孤児プロセス（`_worker` が別
    // セッションで生き残った場合）がパイプの書き手を握り続けるケースに
    // 備え、`recv_timeout` で上限を掛ける（issue #178 実装計画 3.5）。
    let stdout_drain = match stdout_rx {
        Some(rx) => rx.recv_timeout(READER_DRAIN_TIMEOUT).ok(),
        None => None,
    };
    let stderr_drain = match stderr_rx {
        Some(rx) => rx.recv_timeout(READER_DRAIN_TIMEOUT).ok(),
        None => None,
    };

    let Some(stdout_drain) = stdout_drain else {
        return Err(TrainProcessError::StdoutIncomplete);
    };
    let (worker_stderr, stderr_truncated) = match stderr_drain {
        Some(drained) => (drained.kept, drained.truncated),
        None => (Vec::new(), true),
    };

    let process_exit = classify_process_exit_status(status)?;
    let outcome = classify_exit(process_exit, &stdout_drain.kept, request)?;

    Ok(TrainRun {
        outcome,
        exit_code: process_exit,
        elapsed,
        worker_stderr,
        stderr_truncated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::MAX_TRAIN_WALL_SECONDS;
    use crate::request::{Device, TrainRequestParams};

    fn test_request(time_limit_seconds: Option<u32>) -> TrainRequest {
        TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: "/fandhe-edge-fixture-root".to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds,
            rss_limit_bytes: None,
        })
        .expect("valid request params")
    }

    /// REQ-39・issue #178 オーナー確認事項 1: 既定の外側締め切りは
    /// `time_limit_seconds + SUPERVISOR_SHUTDOWN_GRACE_SECONDS`（具体値）。
    #[test]
    fn req39_for_request_adds_shutdown_grace_to_time_limit() {
        let request = test_request(Some(1));
        let limits = RunLimits::for_request(&request);
        assert_eq!(
            limits.wall_timeout(),
            Duration::from_secs(1 + u64::from(SUPERVISOR_SHUTDOWN_GRACE_SECONDS))
        );
    }

    /// 既定値（3600 秒）でも外側締め切りは
    /// `MAX_TRAIN_WALL_SECONDS + SUPERVISOR_SHUTDOWN_GRACE_SECONDS`
    /// （= 3660 秒）を超えない。
    #[test]
    fn req39_for_request_never_exceeds_wall_plus_grace() {
        let request = test_request(None);
        let limits = RunLimits::for_request(&request);
        assert_eq!(
            limits.wall_timeout(),
            Duration::from_secs(u64::from(
                MAX_TRAIN_WALL_SECONDS + SUPERVISOR_SHUTDOWN_GRACE_SECONDS
            ))
        );
    }

    /// `with_wall_timeout` は締める方向だけを許可し、既定以上・0 の指定は
    /// 拒否する（REQ-39「資源の上限」を緩めない）。
    #[test]
    fn req39_with_wall_timeout_rejects_loosening() {
        let request = test_request(Some(10));
        let limits = RunLimits::for_request(&request);
        assert!(matches!(
            limits.with_wall_timeout(Duration::ZERO).unwrap_err(),
            TrainProcessError::InvalidRunLimits
        ));
        assert!(matches!(
            limits.with_wall_timeout(limits.wall_timeout()).unwrap_err(),
            TrainProcessError::InvalidRunLimits
        ));
        assert!(matches!(
            limits
                .with_wall_timeout(limits.wall_timeout() + Duration::from_secs(1))
                .unwrap_err(),
            TrainProcessError::InvalidRunLimits
        ));
        let tightened = limits
            .with_wall_timeout(Duration::from_millis(500))
            .expect("tightening must be accepted");
        assert_eq!(tightened.wall_timeout(), Duration::from_millis(500));
    }

    /// REQ-39・issue #178 レビュー指摘: `config` が巨大なリクエストは
    /// `write_request_file`（`run_train` 内部）が `TrainRequestError::TooLarge`
    /// を `TrainProcessError::Request` として伝播し、`exit_code()` は
    /// `LimitExceeded`（20）になる（`RequestWrite`〔`RuntimeError`〕へ
    /// 丸めない）。
    #[test]
    fn req39_write_request_file_propagates_too_large_config_as_limit_exceeded() {
        use crate::limits::MAX_REQUEST_BYTES;
        use crate::request::TrainRequestParams;

        let mut huge_config = serde_json::Map::new();
        huge_config.insert(
            "huge".to_string(),
            serde_json::Value::String("x".repeat(MAX_REQUEST_BYTES * 2)),
        );
        let request = TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: huge_config,
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: "/fandhe-edge-fixture-root".to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: None,
            rss_limit_bytes: None,
        })
        .expect("config size is not checked at construction time");

        let job_dir = std::env::temp_dir().join(format!(
            "fandhe-edge-train-test-huge-config-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&job_dir).expect("create job dir");
        let err = write_request_file(&job_dir, &request).unwrap_err();
        let _ = std::fs::remove_dir_all(&job_dir);
        assert!(matches!(err, TrainProcessError::Request(_)));
        assert_eq!(err.exit_code(), ExitCode::LimitExceeded);
    }

    /// `WorkerLauncher::new`: 相対パスの `python` は拒否する。
    #[test]
    fn req39_launcher_rejects_relative_python() {
        let err = WorkerLauncher::new(PathBuf::from("python3"), PathBuf::from("/abs/launch.py"))
            .unwrap_err();
        assert!(matches!(
            err,
            TrainProcessError::InvalidLauncher { field: "python" }
        ));
    }

    /// `WorkerLauncher::new`: `launch.py` 以外のファイル名は拒否する。
    #[test]
    fn req39_launcher_rejects_wrong_launch_script_name() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let python = dir.join(format!("fandhe-edge-train-test-python-{pid}"));
        let wrong = dir.join(format!("fandhe-edge-train-test-not-launch-{pid}.py"));
        std::fs::write(&python, b"").expect("write stub python");
        std::fs::write(&wrong, b"").expect("write stub script");
        let err = WorkerLauncher::new(python.clone(), wrong.clone()).unwrap_err();
        let _ = std::fs::remove_file(&python);
        let _ = std::fs::remove_file(&wrong);
        assert!(matches!(
            err,
            TrainProcessError::InvalidLauncher {
                field: "launch_script"
            }
        ));
    }

    /// `WorkerLauncher::new`: 存在しないファイルは拒否する。
    #[test]
    fn req39_launcher_rejects_missing_files() {
        let err = WorkerLauncher::new(
            PathBuf::from("/nonexistent/fandhe-edge-train/python3"),
            PathBuf::from("/nonexistent/fandhe-edge-train/launch.py"),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            TrainProcessError::InvalidLauncher { field: "python" }
        ));
    }

    /// `WorkerLauncher::new`: ディレクトリを渡すと拒否する（通常ファイルで
    /// ないため）。
    #[test]
    fn req39_launcher_rejects_directory() {
        let dir = std::env::temp_dir();
        // `python` にディレクトリを渡す（通常ファイルではないため拒否される）。
        // 他テストと衝突しないよう固有名の一時ディレクトリを使う。
        let fake_python_dir = dir.join(format!(
            "fandhe-edge-train-test-launcher-dir-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&fake_python_dir).expect("create test dir");
        let launch_py_dir = fake_python_dir.join("launch.py");
        std::fs::create_dir_all(&launch_py_dir).expect("create launch.py dir");
        let err = WorkerLauncher::new(fake_python_dir.clone(), launch_py_dir).unwrap_err();
        let _ = std::fs::remove_dir_all(&fake_python_dir);
        assert!(matches!(err, TrainProcessError::InvalidLauncher { .. }));
    }

    /// `classify_exit`: プロセス終了コードと結果 JSON が示す終了コードが
    /// 一致しない場合は `ExitCodeMismatch` を返す。
    #[test]
    fn req39_classify_exit_rejects_process_result_mismatch() {
        let request = test_request(None);
        let json = r#"{"status":"error","code":"invalid_request","message":"m"}"#;
        let err = classify_exit(ExitCode::Ok, json.as_bytes(), &request).unwrap_err();
        assert!(matches!(
            err,
            TrainProcessError::ExitCodeMismatch {
                process: ExitCode::Ok,
                expected: ExitCode::InvalidInput,
            }
        ));
    }

    /// `classify_exit`: プロセス終了コードと結果 JSON が一致する場合は
    /// 受理する。
    #[test]
    fn req39_classify_exit_accepts_matching_error_result() {
        let request = test_request(None);
        let json = r#"{"status":"error","code":"invalid_request","message":"m"}"#;
        let outcome = classify_exit(ExitCode::InvalidInput, json.as_bytes(), &request)
            .expect("matching exit code must be accepted");
        assert_eq!(outcome.exit_code(), ExitCode::InvalidInput);
    }
}
