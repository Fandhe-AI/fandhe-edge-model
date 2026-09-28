//! 学習ワーカーの子プロセス起動・タイムアウト・終了コード写像（REQ-21・
//! REQ-34・REQ-39。issue #178）。
//!
//! Rust 側 CLI・ジョブ管理（未配線。TASK-33.x）が、学習ワーカー
//! （`trainer/launch.py`）を引数リストで子プロセス起動し、壁時計タイムアウト
//! と標準出力／標準エラー出力の読み取り上限を掛けたうえで、
//! [`crate::result::TrainOutcome::from_worker_stdout`] へ渡して結果を得る
//! ための型・関数を提供する（[`run_train`] が公開入口）。
//!
//! # プロセスグループによる一括終了（issue #178 PR #233 レビュー再々指摘。
//! `ps` スナップショット方式からの全面移行）
//!
//! 以前の実装は、壁時計タイムアウト検出後に `/bin/ps` でプロセスツリーを
//! 走査してから子孫を 1 件ずつ `kill` する方式だった。この方式には
//! 構造的な欠陥が複数あった: (1) `ps` の出力サイズに固定上限
//! （`MAX_RESULT_BYTES`）を流用しており、プロセス数が多い環境では常に
//! 失敗しうる、(2) `ps` が失敗した際に記録済み PID の同一性を確認できず、
//! PID 再利用時に無関係なプロセスを誤って終了させかねない、(3) supervisor
//! が別セッションで `_worker` を起動した直後、最初の `ps` 実行より前に
//! supervisor 自身が終了すると、子孫 PID を一度も記録できないまま
//! `_worker` を見失う。
//!
//! 本モジュールはこれを**プロセスグループへの一括シグナル送出**へ置き換えた:
//!
//! 1. [`run_train`] は supervisor（`trainer/launch.py train ...`）を
//!    `std::os::unix::process::CommandExt::process_group(0)` で起動する。
//!    これにより supervisor は新しいプロセスグループの先頭（`pgid` =
//!    supervisor 自身の pid）になる。`unsafe`・新規依存は不要（`process_group`
//!    は safe API）。
//! 2. `trainer/src/fandhe_edge_trainer/supervisor.py` は `_worker` をもはや
//!    別セッション（`start_new_session=True`）で起動しない。`_worker` は
//!    fork した瞬間からこの同じプロセスグループに属し続ける（同モジュールの
//!    docstring 参照）。したがって供給元が別セッションへ抜けて発見できなく
//!    なる窓は存在しない。
//! 3. [`kill_process_group_best_effort`] が `/bin/kill -KILL -- -<pgid>`
//!    （固定 argv・絶対パス・シェル不使用）を 1 回呼ぶだけで、supervisor・
//!    `_worker`・その子孫のすべてへ `SIGKILL` が届く（カーネルが
//!    プロセスグループ単位でシグナルを配送するため、`ps` による個別の
//!    走査・再送ラウンドは不要になった。これに伴い `MAX_RESULT_BYTES` の
//!    サイズ上限流用問題も解消する）。
//!
//! # 不変条件: グループ kill は reap（回収）より必ず先に行う
//!
//! `Child::try_wait()`／`Child::wait()` は `waitpid(2)` 相当で、
//! `Ok(Some(status))` を返した時点で supervisor（= プロセスグループの
//! リーダー）は reap 済みになり、その pid（= pgid）は OS に返却されて
//! 再利用されうる。reap 済みの pgid へ後から `kill(-pgid)` を送ると、
//! 無関係な新しいプロセスグループを巻き込みかねない（codex 指摘 P0 x2。
//! issue #178 PR #233 レビュー再々指摘）。
//!
//! そのため [`run_train`] は、supervisor の完了検知に `try_wait()` の
//! ポーリングを一切使わない。代わりに、標準出力の読み取りスレッド
//! （[`spawn_reader`]）が EOF・読み取り上限超過・エラーのいずれかで完了
//! するのを、残りの壁時計予算だけ [`std::sync::mpsc::Receiver::recv_timeout`]
//! で待つ。この待機が時間切れになった場合も、`_worker` が標準出力を握り
//! 続けているだけで供給元自体は既に正常終了している場合も、区別せず同じ
//! 経路で扱う: **正常系・異常系を問わず、reap の前に必ず
//! [`kill_process_group_best_effort`] を呼び、その後で初めて
//! [`wait_after_kill`] により reap する**（[`kill_group_then_reap`] が
//! この順序を関数として固定する）。壁時計予算内に読み取りが完了しな
//! かった場合は [`TrainProcessError::WallTimeout`] を返す（`try_wait()` を
//! 使わないため、以前あった「観測時刻が締め切り後でも成功扱いしてしまう」
//! という種類の競合は構造的に発生しない）。
//!
//! # windows（対象外・fail-closed）
//!
//! windows で同等の恒久対応を行うにはジョブオブジェクト
//! （`CreateJobObject`／`AssignProcessToJobObject`／`TerminateJobObject`。
//! 将来仕様。REQ-39）が要り、`windows` crate 等の新規依存を追加する必要が
//! ある（`.claude/rules/dependency-policy.md` のユーザー承認事項）。承認を
//! 経ずに追加しないため、本 issue では windows を資源上限（REQ-39）を
//! 保証できない環境として扱い、[`run_train`]（`cfg(not(unix))` 版）は
//! 子プロセスを一切起動せず即座に [`TrainProcessError::UnsupportedPlatform`]
//! を返す（fail-closed）。
//!
//! # 推論ランタイムとの境界
//!
//! 本モジュールは学習ワーカー層（`crates/train`）に閉じる。推論ランタイム・
//! CLI コアはこの crate に依存しない（REQ-32。`.claude/rules/coding-rust.md`）。

#[cfg(unix)]
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::{Child, Command, ExitStatus, Stdio};
#[cfg(unix)]
use std::sync::mpsc;
use std::time::Duration;
#[cfg(unix)]
use std::time::Instant;

use fandhe_edge_core::exitcode::ExitCode;

use crate::error::TrainProcessError;
use crate::limits::SUPERVISOR_SHUTDOWN_GRACE_SECONDS;
#[cfg(unix)]
use crate::limits::{MAX_RESULT_BYTES, MAX_WORKER_STDERR_BYTES};
use crate::request::TrainRequest;
use crate::result::TrainOutcome;

/// 出力読み取りスレッドが完了を待つ上限（秒）。子プロセスの終了後、
/// パイプが閉じてスレッドが `recv` から戻るまでの猶予（REQ-39「資源の
/// 上限」）。`_worker` が supervisor の標準エラー出力を継承しているため、
/// 孤児が残っているとパイプが閉じず読み取りが終わらないことへの対策
/// （issue #178 実装計画 3.5）。
#[cfg(unix)]
const READER_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// `Child::try_wait()` のポーリング間隔（ミリ秒）。std に `wait_timeout`
/// 相当が無いため、短い間隔で非同期にポーリングする。
#[cfg(unix)]
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// 子プロセスの環境変数の許可リスト（OS ごと）。`env_clear()` のうえで
/// これらのキーだけを、親プロセスに存在する場合に限り引き継ぐ。
///
/// unix: `TMPDIR` のみ（一時ファイルの配置先解決に使われうる）。
/// windows: 完全に空の環境は CRT・Python の DLL 探索を壊しうるため、
/// `SystemRoot`・`TEMP`・`TMP` を許可する（issue #178 実装計画 3.3 を
/// Windows でも安全に動くよう拡張。根拠: Python ランタイムは
/// Windows で `SystemRoot` が無いと DLL 探索に失敗しうる）。現時点では
/// windows 版 [`run_train`] が子プロセスを起動しない（`UnsupportedPlatform`
/// を即座に返す。モジュール doc「windows（対象外・fail-closed）」参照）
/// ため、この許可リストは実際には使われていない。将来ジョブオブジェクトで
/// windows 対応する際に使う想定で残す（REQ-39。実装済みを装わないための
/// 明記）。
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
// unix 以外では [`run_train`] がこれらのフィールドを一切読まない
// （`UnsupportedPlatform` を即座に返す fail-closed 版のため。モジュール doc
// 「windows（対象外・fail-closed）」参照）が、`WorkerLauncher::new` に
// よる起動口の検証自体はプラットフォームによらず意味を持つ（将来 CLI 層が
// 事前検証に使いうる）ため、型自体は cfg で分けない。
#[cfg_attr(not(unix), allow(dead_code))]
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
    /// unix 限定（[`run_train`] 参照）。
    #[cfg(unix)]
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
///
/// 本型自体は組み立てに使った [`TrainRequest`] を覚えていない（型として
/// 「どの `request` に対応する上限か」を保持しない）ため、[`run_train`] へ
/// 渡す `request` と `limits` の対応は呼び出し元の責務になる。誤って
/// 別リクエスト（長い `time_limit_seconds`）から作った `RunLimits` を
/// 短い `request` に渡すと、外側の壁時計上限を実質的に緩めてしまいうる
/// （codex/review 指摘 P0。issue #178 PR #233 レビュー）。[`run_train`] は
/// 呼び出し時に `limits.wall_timeout()` が `RunLimits::for_request(request)`
/// の値を超えないことを検証し、超える場合は
/// [`TrainProcessError::InvalidRunLimits`] を返す（fail-closed）。
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
#[cfg(unix)]
#[derive(Debug)]
struct RequestFileGuard {
    path: PathBuf,
}

#[cfg(unix)]
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
#[cfg(unix)]
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
    // `create_new` の成功直後（`write_all`／`sync_all` より前）にガードを
    // 確立する。書き込み・同期の失敗時にも作成済みファイルを必ず削除し、
    // 残置による次回同一 `job_dir` での `AlreadyExists` 衝突を防ぐ
    // （codex/Cursor Bugbot 指摘。issue #178 PR #233 レビュー）。
    let guard = RequestFileGuard { path };
    file.write_all(&bytes)
        .map_err(|e| TrainProcessError::RequestWrite { kind: e.kind() })?;
    file.sync_all()
        .map_err(|e| TrainProcessError::RequestWrite { kind: e.kind() })?;
    Ok(guard)
}

/// 上限つきで読み進める出力読み取りスレッドの結果。
#[cfg(unix)]
struct DrainedOutput {
    /// 保持したバイト列（上限超過分は含まない）。
    kept: Vec<u8>,
    /// 上限を超えて読み捨てた分があるか。
    truncated: bool,
    /// `read()` 自体がエラーを返して読み取りを終えたか（EOF による正常終了
    /// ではない）。`true` の場合、`kept` は最後まで読み切れていない可能性が
    /// あり、たまたま有効な結果 JSON に見えても呼び出し元は成功として扱っては
    /// ならない（codex/review 指摘 P1「パイプ読み取りエラーを EOF として
    /// 扱う」。issue #178 PR #233 レビュー。REQ-39「資源の上限」・エラー
    /// ハンドリングの基準）。
    read_error: bool,
}

/// パイプから上限 `cap` バイトまで保持しつつ読み進める（超過分は読み捨てる
/// が、読み取り自体は EOF まで続ける。子プロセスがパイプ書き込みで
/// ブロックしないようにするため。`supervisor.py::_drain_stdout` と同じ
/// 理由）。別スレッドで実行する想定（[`spawn_reader`]）。
///
/// `read()` のエラーは EOF（`Ok(0)`）と区別し、`DrainedOutput::read_error` へ
/// 記録する（`kept` を最後まで読み切れなかった可能性があるため。codex/review
/// 指摘 P1。issue #178 PR #233 レビュー）。ただし `ErrorKind::Interrupted`
/// （`EINTR`。シグナル配送等で発生しうる retryable なエラー）は読み取り
/// 未完了の証拠にならないため打ち切らず、同じ `read()` をやり直す（Cursor
/// Bugbot 指摘 Medium「Pipe reads fail on interrupt」。issue #178 PR #233
/// レビュー: 修正前は `EINTR` も他の `read()` エラーと同列に扱い、ワーカーが
/// 実際には正常終了していても `StdoutIncomplete`／`StderrIncomplete` に
/// 分類してしまっていた）。
#[cfg(unix)]
fn drain_capped<R: Read>(mut reader: R, cap: usize) -> DrainedOutput {
    let mut kept = Vec::new();
    let mut truncated = false;
    let mut read_error = false;
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
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                // retryable: シグナル配送等による一時的な中断。読み取り
                // 未完了として扱わず、同じ read() をやり直す。
            }
            Err(_) => {
                read_error = true;
                break;
            }
        }
    }
    DrainedOutput {
        kept,
        truncated,
        read_error,
    }
}

/// パイプ読み取りを専用スレッドへ切り出し、`mpsc::Receiver` を返す
/// （呼び出し元は `recv_timeout` で上限付きに待つ）。
#[cfg(unix)]
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
///
/// `Debug` は手書きする（`#[derive(Debug)]` を使わない）: `worker_stderr` の
/// バイト列と、失敗時は `outcome` 内の `message`（[`crate::result::WorkerFailure::message`]）
/// が学習データ由来の内容を含みうるため、呼び出し元が本型をそのまま
/// デバッグログへ出すだけでデータが記録されてしまう（codex/review 指摘 P0
/// 「TrainRun の Debug 表示でワーカー出力が漏れる」。issue #178 PR #233
/// レビュー。`.claude/rules/security.md`「秘密情報の混入防止」）。
pub struct TrainRun {
    outcome: TrainOutcome,
    exit_code: ExitCode,
    elapsed: Duration,
    worker_stderr: Vec<u8>,
    stderr_truncated: bool,
}

impl std::fmt::Debug for TrainRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainRun")
            .field("outcome", &RedactedOutcome(&self.outcome))
            .field("exit_code", &self.exit_code)
            .field("elapsed", &self.elapsed)
            .field("worker_stderr_len", &self.worker_stderr.len())
            .field("stderr_truncated", &self.stderr_truncated)
            .finish()
    }
}

/// [`TrainRun`] の手書き `Debug` 実装が使う補助型。`outcome` の中身を、
/// データ由来の値（`message`）を伏せた形で整形する。
struct RedactedOutcome<'a>(&'a TrainOutcome);

impl std::fmt::Debug for RedactedOutcome<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            TrainOutcome::Ok(_) => write!(f, "Ok(<redacted: worker-provided artifact record>)"),
            TrainOutcome::Error(failure) => write!(
                f,
                "Error {{ code: {:?}, message: <redacted: worker-provided, {} bytes> }}",
                failure.code(),
                failure.message().len()
            ),
        }
    }
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

/// `/bin/kill`（プロセスグループへの `SIGKILL` 送出）の絶対パス。
/// `supervisor.py::_PS_BIN` と同じ方針（シェル不使用・絶対パス固定。
/// `.claude/rules/security.md`「インジェクション」）。
#[cfg(unix)]
const KILL_BIN: &str = "/bin/kill";

/// `/bin/kill` の起動・終了待ちあたりの上限（REQ-39「資源の上限」）。
/// `SIGKILL` は通常即座に効くため、この上限に達するのは割り込み不可能な
/// OS 側の待ち（D state）等の極めて稀なケースに限られる（Cursor Bugbot
/// 指摘 Medium「Timeout wait can block forever」。issue #178 PR #233
/// レビュー。[`wait_after_kill`] 参照）。
#[cfg(unix)]
const KILL_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

/// `pgid`（[`run_train`] が起動した supervisor のプロセスグループ ID。
/// `process_group(0)` により supervisor 自身の pid と一致する）へ
/// `SIGKILL` を送る（`/bin/kill -KILL -- -<pgid>`。固定 argv・絶対パス、
/// シェル不使用）。
///
/// `_worker` が supervisor と同じプロセスグループに留まる設計
/// （`trainer/src/fandhe_edge_trainer/supervisor.py` が `start_new_session`
/// を使わない。モジュール doc「プロセスグループによる一括終了」参照）の
/// ため、この 1 回の呼び出しで supervisor 自身・`_worker`・その子孫の
/// すべてへ `SIGKILL` が届く。カーネルがプロセスグループ単位でシグナルを
/// 配送するため、`ps` による個別の走査・再送ラウンドは不要である
/// （旧実装との違い。issue #178 PR #233 レビュー再々指摘）。
///
/// **呼び出し元が守るべき不変条件**: 本関数は、対象の `pgid`（= supervisor
/// の pid）をまだ `try_wait()`／`wait()` で回収（reap）していない間に
/// だけ呼んでよい。reap 済みの pid は OS に返却されて別プロセスに
/// 再利用されうるため、reap 後に同じ pgid へ送ると無関係なプロセス
/// グループを巻き込みかねない（codex 指摘 P0 x2。issue #178 PR #233
/// レビュー再々指摘）。[`run_train`] は本関数を [`kill_group_then_reap`]
/// 経由でのみ呼び、`wait_after_kill`（reap）より必ず先に実行する。
///
/// `/bin/kill` の終了コード 0（送出成功）・1（多くの実装で ESRCH＝対象が
/// 既に存在しない、を含む一般エラー）のいずれも「掃除を試みて問題
/// なかった」として `true` を返す（ESRCH は「既に何も残っていない」ことを
/// 意味し正常。他の一般エラー〔EPERM・引数不正等〕も同じ終了コード 1 の
/// ため区別できないが、ベストエフォートの範囲として許容する）。起動失敗・
/// タイムアウト・その他の終了コードは `false`（確認できなかった）。
#[cfg(unix)]
fn kill_process_group_best_effort(pgid: u32) -> bool {
    let pgid_arg = format!("-{pgid}");
    let mut command = Command::new(KILL_BIN);
    command
        .args(["-KILL", "--", pgid_arg.as_str()])
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let Ok(mut child) = command.spawn() else {
        return false;
    };
    let deadline = Instant::now() + KILL_WAIT_TIMEOUT;
    match poll_wait_bounded(&mut child, deadline) {
        Ok(Some(status)) => matches!(status.code(), Some(0) | Some(1)),
        Ok(None) | Err(_) => {
            // タイムアウト・`try_wait()` 自体のエラーのいずれでも、`kill()`
            // 後の回収を無期限に待たない（Cursor Bugbot 指摘 Medium「kill 後
            // wait が無期限ブロックしうる」。issue #178 PR #233 レビュー）。
            let _ = child.kill();
            let kill_deadline = Instant::now() + KILL_WAIT_TIMEOUT;
            let _ = poll_wait_bounded(&mut child, kill_deadline);
            false
        }
    }
}

/// [`run_train`] の異常系共通処理: プロセスグループへの一括 `SIGKILL`
/// （`kill_group`）を**必ず**直接の子（supervisor）の reap（`reap`）より
/// 先に呼ぶ、という順序を関数として固定する（モジュール doc「不変条件」
/// 参照。issue #178 PR #233 レビュー再々指摘 6「kill がリーダー回収前に
/// 行われる順序を確認する」）。`kill_group` の戻り値（掃除を確認できたか）
/// を `reap`（`descendants_confirmed_clean` として使う）へそのまま渡す。
///
/// 実プロセスを使わずに呼び出し順序を検証できるよう、`kill_group`・`reap`
/// を引数として受け取る形に切り出してある
/// （`tests::req39_kill_group_then_reap_calls_kill_before_reap` 参照）。
#[cfg(unix)]
fn kill_group_then_reap<T>(
    mut kill_group: impl FnMut() -> bool,
    mut reap: impl FnMut(bool) -> Result<T, TrainProcessError>,
) -> (bool, Result<T, TrainProcessError>) {
    let confirmed = kill_group();
    let reaped = reap(confirmed);
    (confirmed, reaped)
}

/// `Child::try_wait()` を `deadline` までポーリングする共通ヘルパー。
///
/// `try_wait()` 自体が `ErrorKind::Interrupted`（`EINTR`。シグナル配送等で
/// 発生しうる retryable なエラー）を返した場合は打ち切らず再試行する
/// （[`drain_capped`] の `EINTR` 扱いと同じ理由。Cursor Bugbot 指摘 Medium
/// 「Pipe reads fail on interrupt」と同種の欠陥が [`wait_after_kill`] にも
/// あった：修正前は `EINTR` を他の `try_wait()` エラーと同列に扱い、
/// `SIGKILL` 送出後で実際にはまだ生きているだけのプロセスを、回収に失敗した
/// ものとして即座にエラー化していた。issue #178 PR #233 レビュー）。
///
/// 戻り値: `Ok(Some(status))` は終了を確認できた、`Ok(None)` は `deadline`
/// まで待っても終了しなかった（呼び出し元は上限超過として扱う）、
/// `Err(e)`（`EINTR` 以外）は `try_wait()` 自体が失敗したことを示す。
#[cfg(unix)]
fn poll_wait_bounded(child: &mut Child, deadline: Instant) -> std::io::Result<Option<ExitStatus>> {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) => {
                if Instant::now() >= deadline {
                    return Ok(None);
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                if Instant::now() >= deadline {
                    return Ok(None);
                }
            }
            Err(e) => return Err(e),
        }
    }
}

/// `child.kill()`（`SIGKILL` 送出）の直後に呼び、終了を最大
/// [`KILL_WAIT_TIMEOUT`] までポーリングして待つ。
///
/// `SIGKILL` は通常即座に効くが、ディスク I/O 等での割り込み不可能な待ち
/// （D state）に入っているプロセスは、シグナル配送後もしばらく（理論上は
/// 無期限に）終了しないことがある。素朴な `Child::wait()` はこの間ブロック
/// し続け、呼び出しスレッド自体が資源の上限なく固まってしまう（Cursor
/// Bugbot 指摘 Medium「Timeout wait can block forever」。issue #178 PR #233
/// レビュー）。[`poll_wait_bounded`] で上限つきでポーリングし、上限に達しても
/// 終了しない場合は [`TrainProcessError::KillWaitTimedOut`] を返して呼び出し
/// 元へ制御を返す（この場合プロセスは OS 上にゾンビとして残り続ける可能性が
/// あり、確実な後始末を主張しない。fail-closed。REQ-39「資源の上限」）。
///
/// `descendants_confirmed_clean` は、この直前に実行した
/// [`kill_process_group_best_effort`] の戻り値をそのまま
/// [`TrainProcessError::Wait`] へ伝播するために受け取る（`Child::wait()`
/// 自体が失敗した場合のみ使う。[`kill_group_then_reap`] がこの受け渡しを
/// 固定する。issue #178 PR #233 レビュー）。
#[cfg(unix)]
fn wait_after_kill(
    child: &mut Child,
    descendants_confirmed_clean: bool,
) -> Result<ExitStatus, TrainProcessError> {
    let deadline = Instant::now() + KILL_WAIT_TIMEOUT;
    match poll_wait_bounded(child, deadline) {
        Ok(Some(status)) => Ok(status),
        Ok(None) => Err(TrainProcessError::KillWaitTimedOut),
        Err(e) => Err(TrainProcessError::Wait {
            kind: e.kind(),
            descendants_confirmed_clean,
        }),
    }
}

/// 子プロセスの終了コード（unix はシグナル終了で `None` になりうる）を
/// [`ExitCode`] へ写す。JSON 側の判定（`classify_result_exit_code`）とは
/// 独立に、まず「プロセスがどう終了したか」だけを判定する。
#[cfg(unix)]
fn classify_process_exit_status(status: ExitStatus) -> Result<ExitCode, TrainProcessError> {
    let Some(code) = status.code() else {
        return Err(TrainProcessError::TerminatedBySignal);
    };
    ExitCode::try_from(code).map_err(|e| TrainProcessError::UnknownExitCode(e.0))
}

/// 標準出力（信頼しない外部入力）を検証済みの [`TrainOutcome`] へ解析し、
/// その結果が示す終了コードと、子プロセスの実終了コード（`process_exit`）
/// が一致することを確認する（issue #178 実装計画 3.6・表の 6〜8 行目）。
#[cfg(unix)]
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

/// unix 以外（windows 等）では [`TrainProcessError::UnsupportedPlatform`]
/// を返す（fail-closed。モジュール doc「windows（対象外・fail-closed）」
/// 参照。issue #178 PR #233 レビュー再指摘 P0）。子プロセスは一切起動せず、
/// `job_dir` への書き込みも行わない。
#[cfg(not(unix))]
pub fn run_train(
    _launcher: &WorkerLauncher,
    _request: &TrainRequest,
    _job_dir: &Path,
    _limits: &RunLimits,
) -> Result<TrainRun, TrainProcessError> {
    Err(TrainProcessError::UnsupportedPlatform)
}

/// 学習ワーカーを子プロセスとして起動し、壁時計タイムアウトと出力の読み
/// 取り上限を掛けたうえで、検証済みの結果 [`TrainRun`] を得る（REQ-21・
/// REQ-34・REQ-39。issue #178 の公開入口）。
///
/// 手順: (1) `job_dir/request.json` を新規作成 → (2) supervisor を新しい
/// プロセスグループの先頭として起動（固定 argv・`env_clear()`＋許可リストの
/// 環境・`current_dir(job_dir)`）→ (3) 標準出力が読み取り上限まで完了する
/// （EOF・エラー）のを、残りの壁時計予算だけ待つ → (4) 正常系・異常系を
/// 問わず、プロセスグループへ一括 `SIGKILL` してから直接の子を reap する
/// （[`kill_group_then_reap`]。モジュール doc「不変条件」参照）→
/// (5) [`classify_exit`] で結果を検証。すべての経路（成功・失敗・
/// タイムアウト）で `request.json` を削除する（[`RequestFileGuard`]）。
/// unix 限定（windows 版は上記の `#[cfg(not(unix))]` 版を参照）。
#[cfg(unix)]
pub fn run_train(
    launcher: &WorkerLauncher,
    request: &TrainRequest,
    job_dir: &Path,
    limits: &RunLimits,
) -> Result<TrainRun, TrainProcessError> {
    // `request` と `limits` は呼び出し元が別々の引数として渡すため、型では
    // 対応関係を強制できない。検証なしに `limits.wall_timeout()` を採用すると、
    // 短い `time_limit_seconds` の `request` に、別の（長い）リクエストから
    // 作った `RunLimits` を渡すことで外側の壁時計上限を実質的に緩められて
    // しまう（codex/review 指摘 P0「別のリクエスト用 RunLimits で壁時計上限を
    // 緩められる」。issue #178 PR #233 レビュー。REQ-39「資源の上限」）。
    // `RunLimits::for_request(request)` から導かれる上限を超える `limits` は
    // 拒否する（`RunLimits::with_wall_timeout` が保証する「締める方向だけ」の
    // 不変条件を、`request` との対応についても同様に守る）。
    if limits.wall_timeout() > RunLimits::for_request(request).wall_timeout() {
        return Err(TrainProcessError::InvalidRunLimits);
    }
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
    // supervisor を新しいプロセスグループの先頭として起動する（`pgid` =
    // supervisor 自身の pid）。`_worker` は python 側で `start_new_session`
    // を使わない設計に変えたため（`trainer/src/fandhe_edge_trainer/
    // supervisor.py` のモジュール docstring 参照）、fork した瞬間から
    // このグループに属し続ける。`process_group` は safe API のため
    // `unsafe` は不要（モジュール doc「プロセスグループによる一括終了」）。
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }

    let mut child = command
        .spawn()
        .map_err(|e| TrainProcessError::Spawn { kind: e.kind() })?;
    // `process_group(0)` により pgid == 自身の pid になる。
    let pgid = child.id();

    // stdout・stderr の読み取りスレッドを起動する前にパイプを取り出す
    // （書き込み側を親プロセスが握ったままにしない。取り出し忘れると
    // `Child` が drop される際にパイプが閉じられず readers が終わらない
    // ことがある）。
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let stdout_rx = stdout_pipe.map(|pipe| spawn_reader(pipe, MAX_RESULT_BYTES + 1));
    let stderr_rx = stderr_pipe.map(|pipe| spawn_reader(pipe, MAX_WORKER_STDERR_BYTES));

    // 完了検知は「標準出力の読み取りスレッドが完了する（EOF・上限超過・
    // エラーのいずれか）まで、残りの壁時計予算だけ待つ」の 1 本にする。
    // `try_wait()` によるポーリングは使わない（モジュール doc「不変条件:
    // グループ kill は reap（回収）より必ず先に行う」参照。`try_wait()` が
    // `Ok(Some(status))` を返した時点で pgid が再利用されうるため、それより
    // 前に完了を検知する手段が必要）。
    let deadline = started + limits.wall_timeout();
    let stdout_wait = stdout_rx.map(|rx| {
        let remaining = deadline.saturating_duration_since(Instant::now());
        rx.recv_timeout(remaining)
    });
    // `stdout_wait` が `None` になるのは `stdout_pipe` を取得できなかった
    // 場合だが、本関数は常に `stdout(Stdio::piped())` を指定するため
    // 到達しない防御的分岐。到達したとしても後続の `match` が
    // `StdoutIncomplete` として安全側に倒す。
    let timed_out = matches!(
        stdout_wait,
        Some(Err(mpsc::RecvTimeoutError::Timeout)) | None
    );

    // 正常系・異常系（タイムアウトを含む）を問わず、reap の前に必ず
    // プロセスグループ全体へ `SIGKILL` を送る。`_worker` が同じグループに
    // 残っていれば道連れに終了させる（ESRCH 相当＝既に何も残っていない、は
    // 正常な結果として扱う。issue #178 PR #233 レビュー再々指摘「ps 由来の
    // スナップショット方式そのものの欠陥」への対応として、`ps` による
    // プロセスツリー走査を全廃し、プロセスグループへの一括シグナルへ
    // 置き換えた）。
    let (descendants_confirmed_clean, reap_result) = kill_group_then_reap(
        || kill_process_group_best_effort(pgid),
        |confirmed| wait_after_kill(&mut child, confirmed),
    );

    if timed_out {
        return Err(TrainProcessError::WallTimeout {
            limit_ms: u64::try_from(limits.wall_timeout().as_millis()).unwrap_or(u64::MAX),
            descendants_confirmed_clean,
            child_reaped: reap_result.is_ok(),
        });
    }

    let status = reap_result?;
    let elapsed = started.elapsed();

    // `timed_out` が `false` の場合、`stdout_wait` は `Some(Ok(_))` か
    // `Some(Err(RecvTimeoutError::Disconnected))`（reader スレッドが
    // 送信せず終了した。極めて稀なバグ経路）のいずれか。後者・`None` は
    // 読み取り未完了として扱う（fail-closed）。
    let stdout_drain = match stdout_wait {
        Some(Ok(drained)) => drained,
        _ => return Err(TrainProcessError::StdoutIncomplete),
    };
    // `recv_timeout` が期限内に値を返した（EOF に達した）場合でも、
    // `DrainedOutput::read_error`（`read()` 自体のエラーで打ち切られ、
    // `kept` を最後まで読み切れていない可能性がある。codex/review 指摘 P1
    // 「パイプ読み取りエラーを EOF として扱う」。issue #178 PR #233
    // レビュー）は読み取り未完了として扱う。たまたま `kept` が有効な結果
    // JSON に見えても、読み切れていない出力を成功として受理しない
    // （fail-closed。REQ-39「資源の上限」）。
    if stdout_drain.read_error {
        return Err(TrainProcessError::StdoutIncomplete);
    }

    // プロセスグループは既に `SIGKILL` 済みのため、標準エラー出力は速やかに
    // EOF へ達するはずである。念のため短い上限（[`READER_DRAIN_TIMEOUT`]）を
    // 掛ける（issue #178 実装計画 3.5）。
    let stderr_drain = match stderr_rx {
        Some(rx) => rx.recv_timeout(READER_DRAIN_TIMEOUT).ok(),
        None => None,
    };
    let Some(stderr_drain) = stderr_drain.filter(|d| !d.read_error) else {
        return Err(TrainProcessError::StderrIncomplete);
    };
    let (worker_stderr, stderr_truncated) = (stderr_drain.kept, stderr_drain.truncated);

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
    /// 丸めない）。`write_request_file` は unix 限定（[`run_train`] 参照）。
    #[cfg(unix)]
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
    /// 一致しない場合は `ExitCodeMismatch` を返す。`classify_exit` は unix
    /// 限定（[`run_train`] 参照）。
    #[cfg(unix)]
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
    #[cfg(unix)]
    #[test]
    fn req39_classify_exit_accepts_matching_error_result() {
        let request = test_request(None);
        let json = r#"{"status":"error","code":"invalid_request","message":"m"}"#;
        let outcome = classify_exit(ExitCode::InvalidInput, json.as_bytes(), &request)
            .expect("matching exit code must be accepted");
        assert_eq!(outcome.exit_code(), ExitCode::InvalidInput);
    }

    /// REQ-39・issue #178 PR #233 レビュー再指摘 P0「Windows で正常終了後の
    /// 孤児ワーカーを停止できない」: unix 以外では `run_train` が子プロセス
    /// を一切起動せず `UnsupportedPlatform`（`RuntimeError`＝70）を返す
    /// （fail-closed）。
    #[cfg(not(unix))]
    #[test]
    fn req39_run_train_rejects_unsupported_platform() {
        // `WorkerLauncher::new` はファイル名がちょうど `launch.py` である
        // ことを要求する（`req39_launcher_rejects_wrong_launch_script_name`
        // 参照）ため、他テストと衝突しない専用ディレクトリの下に
        // `launch.py` という名前で置く。
        let case_dir = std::env::temp_dir().join(format!(
            "fandhe-edge-train-test-unsupported-platform-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&case_dir).expect("create case dir");
        let launch_script = case_dir.join("launch.py");
        std::fs::write(&launch_script, b"").expect("write stub launch.py");
        let python = std::env::current_exe().expect("resolve current_exe");
        let launcher =
            WorkerLauncher::new(python, launch_script.clone()).expect("valid launcher fields");
        let request = test_request(Some(30));
        let limits = RunLimits::for_request(&request);
        let err = run_train(&launcher, &request, &case_dir, &limits).unwrap_err();
        assert!(matches!(err, TrainProcessError::UnsupportedPlatform));
        assert_eq!(err.exit_code(), ExitCode::RuntimeError);
        assert!(
            !case_dir.join("request.json").exists(),
            "request.json must not be written when platform is unsupported"
        );
        let _ = std::fs::remove_dir_all(&case_dir);
    }

    /// `TrainProcessError::UnsupportedPlatform` の終了コードは常に
    /// `RuntimeError`（70）（プラットフォームに依存せず確認できる契約）。
    #[test]
    fn req39_unsupported_platform_maps_to_runtime_error() {
        assert_eq!(
            TrainProcessError::UnsupportedPlatform.exit_code(),
            ExitCode::RuntimeError
        );
    }

    /// issue #178 PR #233 レビュー再々指摘 6「kill がリーダー回収前に行われる
    /// 順序を確認する」: [`kill_group_then_reap`] が `kill_group` を必ず
    /// `reap` より先に呼び、`kill_group` の戻り値（掃除を確認できたか）を
    /// `reap` へそのまま渡すこと。呼び出し順序を実プロセスを使わずに検証する。
    #[cfg(unix)]
    #[test]
    fn req39_kill_group_then_reap_calls_kill_before_reap() {
        let log = std::cell::RefCell::new(Vec::<&'static str>::new());
        let (confirmed, reaped) = kill_group_then_reap(
            || {
                log.borrow_mut().push("kill");
                true
            },
            |passed_confirmed| {
                log.borrow_mut().push("reap");
                assert!(passed_confirmed, "reap must receive kill_group's result");
                Ok::<(), TrainProcessError>(())
            },
        );
        assert!(confirmed);
        assert!(reaped.is_ok());
        assert_eq!(*log.borrow(), vec!["kill", "reap"]);
    }

    /// `kill_group_then_reap`: `kill_group` が `false`（掃除を確認できな
    /// かった）を返した場合も、その値がそのまま `reap` へ渡ること。
    #[cfg(unix)]
    #[test]
    fn req39_kill_group_then_reap_propagates_unconfirmed_kill() {
        let (confirmed, reaped) = kill_group_then_reap(
            || false,
            |passed_confirmed| {
                assert!(!passed_confirmed);
                Ok::<(), TrainProcessError>(())
            },
        );
        assert!(!confirmed);
        assert!(reaped.is_ok());
    }
}
