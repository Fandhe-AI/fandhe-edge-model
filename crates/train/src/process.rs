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
//! # 孤児化の限界（codex/review 指摘。issue #178 PR #233 レビュー P0 対応）
//!
//! supervisor は `_worker` を別セッション（`start_new_session=True`）で
//! 起動するため、supervisor プロセス（`python -I launch.py train ...`）を
//! 直接 kill しただけでは、別セッションの `_worker` には SIGKILL が届かない。
//! 本モジュールは外側の壁時計締め切り超過・`try_wait()` 自体のエラーの
//! いずれの経路でも、[`kill_process_tree_best_effort`] で supervisor を
//! 根とするプロセスツリー（`_worker` を含む子孫）を `/bin/ps`・`/bin/kill`
//! （固定 argv・絶対パス。シェル不使用）で走査・強制終了してから、直接の
//! 子（supervisor 自身）を `Child::kill()`／`wait()` で確実に回収する。
//! native `kill(2)` を直接呼ぶには `libc` クレートの追加（新規依存。
//! `.claude/rules/dependency-policy.md`。ユーザー承認事項）か `unsafe` な
//! FFI（`.claude/rules/coding-rust.md`「unsafe・FFI」。同じくユーザー承認
//! 事項）が要るため、いずれも承認を経ずに追加しない。代わりに
//! `supervisor.py::_current_child_rss_bytes` が既に採用している「固定
//! argv・絶対パスの外部コマンドを呼ぶ」方針に揃える
//! （[`kill_process_tree_best_effort`] のドキュメントコメント参照）。
//!
//! [`kill_process_tree_best_effort`] は「掃除を確認できたか」（`bool`）を
//! 呼び出し元へ返し、[`TrainProcessError::Wait`]・[`TrainProcessError::WallTimeout`]
//! の `descendants_confirmed_clean` フィールドとして伝わる（codex/review
//! 指摘 P0「プロセスツリーの掃除に失敗しても子孫が動き続ける」。issue #178
//! PR #233 レビュー。以前は戻り値を持たず、確認できなくても常に成功した
//! ものとして扱っていた）。
//!
//! 壁時計締め切り超過を検出した後、[`wait_after_kill`] による直接の子の
//! 回収が失敗しても（`KillWaitTimedOut`／`Wait`）、必ず
//! [`TrainProcessError::WallTimeout`]（`LimitExceeded`＝20）を返す。回収の
//! 失敗を理由に代わりの回収エラー（`RuntimeError`＝70）を返すと、締め切り
//! 超過という確定した事実が REQ-21 の終了コード契約に反する形で消えてしまう
//! （codex/review 指摘 P1「wall timeout 超過後 wait_after_kill(...)? 失敗で
//! WallTimeout が KillWaitTimedOut／Wait に化ける」。issue #178 PR #233
//! レビュー。Cursor Bugbot 指摘 Medium も同一事象）。回収できたかどうかは
//! `WallTimeout::child_reaped` で別途伝える。[`wait_after_kill`]・
//! [`poll_wait_bounded`] は `try_wait()` 自体の `EINTR`（`ErrorKind::Interrupted`）
//! も再試行する（Cursor Bugbot 指摘 Medium「Pipe reads fail on interrupt」と
//! 同種の欠陥が `wait_after_kill` にもあった。issue #178 PR #233 レビュー）。
//!
//! supervisor 自体が既に正常終了した後（`try_wait()` が `Ok(Some(status))`
//! を返す通常の終了経路）に `_worker` だけが孤児として残っているケースは、
//! `_worker` の `ppid` チェーンが supervisor の終了と同時に切れてしまう
//! （最も近い subreaper／init へ reparent 済みになる）ため、
//! `kill_process_tree_best_effort` と同じ「`ppid` を辿り直す」方法では
//! 見つけられない。[`run_train`] は壁時計タイムアウトのポーリング中
//! （supervisor がまだ生きている間）に `ps` で子孫 pid のスナップショットを
//! 定期的に記録しておき（[`DESCENDANT_SNAPSHOT_INTERVAL`]）、パイプ読み取り
//! が期限内に完了しなかった場合はそのスナップショットへ
//! [`kill_pids_best_effort`] で直接 `SIGKILL` を送る（codex/review 指摘 P0
//! 「supervisor が先に終了すると孤児ワーカーを停止できない」。issue #178
//! PR #233 レビュー。結合テスト `case_exit_with_orphan_kills_orphan` 参照）。
//!
//! [`run_train`] は壁時計ポーリングループに入る前（最初の `try_wait()` を
//! 呼ぶ前）にも 1 回スナップショットを取る。最初の `try_wait()` が
//! `Ok(Some(status))`（supervisor がポーリング開始前に既に正常終了して
//! いた）を返すと、ループ内のスナップショット処理が一度も実行されず
//! `last_known_descendants` が空のまま孤児化を検出できなくなるため
//! （codex/review 指摘 P0「supervisor 早期終了で子孫 PID スナップショット
//! 未取得のまま抜け、`_worker` 孤児化を検出・停止できない」。issue #178
//! PR #233 レビュー）。
//!
//! `kill_pids_best_effort` が使うスナップショットは `(pid, lstart)`（起動
//! 時刻の絶対文字列）で識別する。`ppid` ではなく `lstart` を使うのは、
//! `_worker` が reparent 済み（`ppid` が変化済み）の状態で呼ばれるためで、
//! `ppid` 一致を条件にすると正規に孤児化しただけのケースまで「PID 再利用の
//! 疑いあり」として除外してしまう（codex/review 指摘 P0「PID 再利用時に
//! 無関係なプロセスを誤って強制終了しうる」。issue #178 PR #233 レビュー）。
//! `ps` 自体が失敗した場合は識別による絞り込みができないため、記録済みの
//! pid を安全側でそのまま kill 対象とする（同レビューの別 P0「`ps` 失敗時に
//! 既知孤児 PID を一度も停止しない」への対応）。
//!
//! [`kill_process_tree_best_effort`]（`ppid` チェーン走査）自体は、`ps`
//! （[`ps_pid_ppid_pairs`]）が失敗しても最初のラウンドで即座に諦めない
//! （codex/review 指摘 P0「`ps_pid_ppid_pairs()` 失敗時、関数が即 `false` を
//! 返し、呼び出し元は直接の子だけを kill する」。issue #178 PR #233 レビュー
//! 再指摘）。判断ロジックは [`sweep_with`] へ切り出しており、`ps` に相当する
//! 問い合わせ（`query`）が失敗する各ラウンドでも、[`run_train`] が保持する
//! 直近のスナップショット（`last_known_descendants`）へ
//! [`kill_pids_best_effort`] で直接 `SIGKILL` を送る（fallback。Cursor
//! Bugbot 指摘 Medium「壁時計タイムアウト時の掃除にポーリング中の
//! `last_known_descendants` が適用されない」と同じ fallback を、`ppid`
//! チェーン走査自体の失敗時にも用いる）。`descendants_confirmed_clean` は
//! `ppid` チェーン側・スナップショット側の両方が掃除を確認できた場合のみ
//! `true` になる（片方だけの確認を「確認できた」と誤って伝えない）。
//!
//! `try_wait()` が `Ok(Some(status))` を返しても、それを観測した時刻が既に
//! 締め切りを過ぎていれば成功として受理しない（codex/review 指摘 P1
//! 「try_wait() の Ok(Some(status)) を締め切り判定より先に受理するため、
//! 期限超過後に終了したプロセスを成功扱いしうる」。issue #178 PR #233
//! レビュー再指摘）。この場合 `try_wait()` は既に子（supervisor）を
//! 回収済みのため、その pid は OS に返却され再利用されうる。同じ pid を対象に
//! `kill_process_tree_best_effort`（`ppid` で再検索する）や `Child::kill()`
//! を再度呼ぶと、無関係なプロセスを誤って終了させる恐れがあるため、
//! [`WaitOutcome::LateExit`] 経路では `pid` に触れず、生存確認済みの子孫
//! スナップショットだけを対象に `kill_pids_best_effort` を呼ぶ。
//!
//! # windows（対象外・fail-closed。issue #178 PR #233 レビュー再指摘 P0）
//!
//! 以前の実装は `taskkill /T /F /PID <root_pid>` でサブツリーごと終了を
//! 試みていたが、supervisor が正常終了した「後」に `_worker` だけが別
//! セッションの孤児として残るケース（本モジュール doc の
//! `kill_pids_best_effort` の節を参照）には対応できていなかった。unix 側は
//! ポーリング中に記録した `(pid, lstart)` スナップショットへ直接
//! `SIGKILL` を送ることでこれを検出・停止するが、windows で同等の恒久対応を
//! 行うにはジョブオブジェクト（`CreateJobObject`／`AssignProcessToJobObject`／
//! `TerminateJobObject`。将来仕様。REQ-39）が要り、`windows` crate 等の新規
//! 依存を追加する必要がある（`.claude/rules/dependency-policy.md` のユーザー
//! 承認事項）。承認を経ずに追加しないため、本 issue では windows を資源上限
//! （REQ-39）を保証できない環境として扱い、[`run_train`]（`cfg(not(unix))`
//! 版）は子プロセスを一切起動せず即座に
//! [`TrainProcessError::UnsupportedPlatform`] を返す（fail-closed。
//! codex/review 指摘 P0「Windows で正常終了後の孤児ワーカーを停止できない」）。
//!
//! 残る既知の限界（unix）: supervisor が `spawn()` 直後（最初のスナップ
//! ショットの `ps` 実行が完了する前）に正常終了した場合は、依然として
//! `_worker` の pid を一度も記録できず対象にできない（現実的な学習ワーカー
//! の起動・初期化時間に対しては極めて狭い窓だが、理論上は残る）。
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

/// `/bin/ps`（プロセス一覧の取得）の絶対パス。`supervisor.py::_PS_BIN` と
/// 同じ方針（シェル不使用・絶対パス固定。`.claude/rules/security.md`
/// 「インジェクション」）。
#[cfg(unix)]
const PS_BIN: &str = "/bin/ps";

/// `/bin/kill`（個々の子孫プロセスへの `SIGKILL` 送出）の絶対パス。
#[cfg(unix)]
const KILL_BIN: &str = "/bin/kill";

/// プロセスツリー走査・強制終了の 1 ステップ（`ps` の実行・`kill` の実行）
/// あたりの上限（REQ-39「資源の上限」）。`ps`／`kill` 自体が固まった場合に
/// 無限待ちにしない。
#[cfg(unix)]
const ORPHAN_SWEEP_STEP_TIMEOUT: Duration = Duration::from_secs(3);

/// [`kill_process_tree_best_effort`] が `ps` による子孫確認・`kill` 送出を
/// 繰り返す最大ラウンド数（REQ-39「資源の上限」。codex/review 指摘 P0
/// 「プロセスツリーの掃除に失敗しても孫プロセスが動き続ける」。issue #178
/// PR #233 レビュー）。単発の `kill` 送出だけでは、送出直後にまだ終了処理中
/// だった子孫や、`kill` コマンド自体が一時的に失敗したケースを取りこぼす
/// ため、`ps` で子孫が消えたことを確認できるまで（最大この回数まで）
/// 再送する。全ラウンドを終えても子孫が残っている場合は、これ以上待たずに
/// 呼び出し元（直接の子＝supervisor の回収）へ制御を返す（ベストエフォート。
/// モジュール doc「孤児化の限界」参照）。
#[cfg(unix)]
const ORPHAN_SWEEP_MAX_ROUNDS: u32 = 5;

/// [`kill_process_tree_best_effort`] の各ラウンドの間隔。`kill` 送出直後は
/// プロセスの終了処理が完了していないことがあるため、次の `ps` 確認まで
/// 短い猶予を置く。
#[cfg(unix)]
const ORPHAN_SWEEP_RETRY_DELAY: Duration = Duration::from_millis(100);

/// [`run_train`] が壁時計タイムアウトのポーリング中に子孫プロセスの pid
/// スナップショット（`ps` の `ppid` チェーン走査）を取り直す最小間隔。
/// supervisor が正常終了した後にパイプ読み取りが完了しない場合
/// （[`kill_pids_best_effort`] が使う経路）に備え、supervisor がまだ生きて
/// いる間の子孫 pid を記録しておく（供給元の `ppid` チェーンは supervisor
/// の終了と同時に切れる〔reparent 済みになる〕ため、終了後には辿れない。
/// codex/review 指摘 P0「supervisor が先に終了すると孤児ワーカーを停止
/// できない」。issue #178 PR #233 レビュー）。`POLL_INTERVAL`（50ms）ごとに
/// 毎回 `ps` を起動すると学習時間全体（最大 3660 秒）にわたって大量の
/// 子プロセスを起動し続けることになるため、この間隔で間引く（REQ-39
/// 「資源の上限」。実際の学習ワーカーの起動・初期化に要する時間に対して
/// 十分短く、この間隔内に終了する supervisor は現実的な想定の外とする）。
#[cfg(unix)]
const DESCENDANT_SNAPSHOT_INTERVAL: Duration = Duration::from_millis(200);

/// `SIGKILL` 送出後、直接の子プロセスの終了を待つ上限（REQ-39「資源の
/// 上限」）。通常 `SIGKILL` は即座に効くため、この上限に達するのは
/// 割り込み不可能な OS 側の待ち（D state）等の極めて稀なケースに限られる
/// （Cursor Bugbot 指摘 Medium「Timeout wait can block forever」。issue
/// #178 PR #233 レビュー。[`wait_after_kill`] 参照）。
#[cfg(unix)]
const KILL_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

/// `program`（絶対パス）を `args` で子プロセスとして起動し、標準出力を
/// 上限 `cap` バイトまで読み取りつつ、`timeout` を超えたら強制終了する。
/// [`kill_process_tree_best_effort`] が `ps` の出力を取得するために使う
/// 内部ヘルパー（REQ-39「資源の上限」。無制限の待ち・読み取りを作らない）。
///
/// 起動・読み取り・終了待ちのいずれかに失敗した、`timeout` を超過した、
/// または終了コードが非 0 の場合は `None`（呼び出し元は「取得できなかった」
/// として安全側〔fail-closed。プロセスツリーの走査を諦めて直接の子だけを
/// 回収する〕に倒す）。
#[cfg(unix)]
fn run_bounded_capture(
    program: &str,
    args: &[&str],
    timeout: Duration,
    cap: usize,
) -> Option<Vec<u8>> {
    let mut command = Command::new(program);
    command
        .args(args)
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = command.spawn().ok()?;
    let rx = child.stdout.take().map(|pipe| spawn_reader(pipe, cap));

    let deadline = Instant::now() + timeout;
    let status = poll_wait_bounded(&mut child, deadline).unwrap_or_default();
    let Some(status) = status else {
        // タイムアウト・`try_wait()` 自体のエラーのいずれでも、`kill()` 後の
        // 回収を無期限に待たない（Cursor Bugbot 指摘 Medium「kill 後 wait が
        // 無期限ブロックしうる」。issue #178 PR #233 レビュー）。
        let _ = child.kill();
        let kill_deadline = Instant::now() + KILL_WAIT_TIMEOUT;
        let _ = poll_wait_bounded(&mut child, kill_deadline);
        return None;
    };
    if !status.success() {
        return None;
    }
    let rx = rx?;
    let drained = rx.recv_timeout(timeout).ok()?;
    if drained.truncated || drained.read_error {
        return None;
    }
    Some(drained.kept)
}

/// `program`（絶対パス）を `args` で起動し、`timeout` の範囲でベストエフォート
/// に完了を待つ（結果は捨てる）。[`kill_process_tree_best_effort`] が
/// `/bin/kill` の起動に使う。起動失敗・タイムアウトは無視する（呼び出し元は
/// 本関数の成否によらず直接の子を [`Child::kill`] で確実に回収するため、
/// ここでの失敗は許容できる）。
#[cfg(unix)]
fn run_bounded_fire_and_forget(program: &str, args: &[&str], timeout: Duration) {
    let mut command = Command::new(program);
    command
        .args(args)
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let Ok(mut child) = command.spawn() else {
        return;
    };
    let deadline = Instant::now() + timeout;
    match poll_wait_bounded(&mut child, deadline) {
        Ok(Some(_)) => {}
        Ok(None) | Err(_) => {
            // タイムアウト・`try_wait()` 自体のエラーのいずれでも、`kill()` 後
            // の回収を無期限に待たない（Cursor Bugbot 指摘 Medium「kill 後
            // wait が無期限ブロックしうる」。issue #178 PR #233 レビュー）。
            let _ = child.kill();
            let kill_deadline = Instant::now() + KILL_WAIT_TIMEOUT;
            let _ = poll_wait_bounded(&mut child, kill_deadline);
        }
    }
}

/// `/bin/ps -eo pid=,ppid=` の出力を `(pid, ppid)` の一覧へ解析する
/// （OS 全体のプロセス一覧。`supervisor.py::_current_child_rss_bytes` と
/// 同じ `ps` 呼び出し方針）。解析できない行（ヘッダ・空行・想定外の書式）は
/// 読み飛ばす（1 行の解析失敗でプロセスツリー走査全体を諦めない）。
#[cfg(unix)]
fn ps_pid_ppid_pairs() -> Option<Vec<(u32, u32)>> {
    let raw = run_bounded_capture(
        PS_BIN,
        &["-eo", "pid=,ppid="],
        ORPHAN_SWEEP_STEP_TIMEOUT,
        MAX_RESULT_BYTES,
    )?;
    let text = String::from_utf8(raw).ok()?;
    let mut pairs = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let (Some(pid_str), Some(ppid_str)) = (parts.next(), parts.next()) else {
            continue;
        };
        let (Ok(pid), Ok(ppid)) = (pid_str.parse::<u32>(), ppid_str.parse::<u32>()) else {
            continue;
        };
        pairs.push((pid, ppid));
    }
    Some(pairs)
}

/// `root`（supervisor の pid）を起点に、`pairs`（OS 全体の `(pid, ppid)`）
/// から推移的な子孫（`_worker` を含む）の pid 一覧を求める。
#[cfg(unix)]
fn transitive_descendants(root: u32, pairs: &[(u32, u32)]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut frontier = vec![root];
    while let Some(current) = frontier.pop() {
        for &(pid, ppid) in pairs {
            if ppid == current && pid != root && !result.contains(&pid) {
                result.push(pid);
                frontier.push(pid);
            }
        }
    }
    result
}

/// `/bin/ps -eo pid=,lstart=` の出力を `(pid, lstart)` の一覧へ解析する。
/// `lstart`（プロセス開始時刻の絶対文字列。`Www Mmm dd hh:mm:ss yyyy` 形式。
/// GNU procps・BSD/macOS の `ps` いずれも対応）は、`ppid` と異なり親の終了に
/// 伴う reparent の影響を受けず、プロセスの生存期間中は不変であるため、
/// [`kill_pids_best_effort`] が「記録した pid が今も記録時と同一プロセスか」
/// を確認する識別子として使う（`ppid` を識別に使うと、reparent そのものが
/// `ppid` を変えてしまい、`_worker` が正規に孤児化しただけのケースまで
/// 「PID が再利用された」と誤判定して kill 対象から除外してしまう欠陥が
/// あった。issue #178 PR #233 レビュー。`ppid` で識別する案は
/// `case_exit_with_orphan_kills_orphan`（reparent 済みの孤児を kill する
/// テスト）で矛盾が顕在化するため採用しなかった）。解析できない行は
/// 読み飛ばす。
#[cfg(unix)]
fn ps_pid_start_pairs() -> Option<Vec<(u32, String)>> {
    let raw = run_bounded_capture(
        PS_BIN,
        &["-eo", "pid=,lstart="],
        ORPHAN_SWEEP_STEP_TIMEOUT,
        MAX_RESULT_BYTES,
    )?;
    let text = String::from_utf8(raw).ok()?;
    let mut pairs = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(pid_str) = parts.next() else {
            continue;
        };
        let Ok(pid) = pid_str.parse::<u32>() else {
            continue;
        };
        let lstart: Vec<&str> = parts.collect();
        if lstart.is_empty() {
            continue;
        }
        pairs.push((pid, lstart.join(" ")));
    }
    Some(pairs)
}

/// [`transitive_descendants`] と同じ走査だが、各子孫の起動時刻
/// （[`ps_pid_start_pairs`] の `lstart`）も併せて返す（`(pid, lstart)`）。
/// [`run_train`] が壁時計ポーリング中に記録するスナップショット
/// （[`kill_pids_best_effort`] が後で使う）は、supervisor 終了後は `ppid`
/// チェーンを辿り直せないため、記録した時点の起動時刻を「その pid が記録時と
/// 同一プロセスであること」の確認材料として保持しておく（codex/review 指摘
/// P0「PID 再利用時に無関係なプロセスを誤って強制終了しうる」。issue #178
/// PR #233 レビュー）。`lstart` の解像度（多くの実装で秒単位）の範囲内で
/// 偶然同じ起動時刻の別プロセスが同じ pid を得る理論上の窓はゼロにできないが
/// 極めて狭い。ベストエフォートの範囲（`.claude/rules/security.md`）。
#[cfg(unix)]
fn transitive_descendants_with_start(root: u32, pairs: &[(u32, u32)]) -> Vec<(u32, String)> {
    let descendants = transitive_descendants(root, pairs);
    if descendants.is_empty() {
        return Vec::new();
    }
    let Some(start_pairs) = ps_pid_start_pairs() else {
        return Vec::new();
    };
    descendants
        .into_iter()
        .filter_map(|pid| {
            start_pairs
                .iter()
                .find(|&&(p, _)| p == pid)
                .map(|(p, s)| (*p, s.clone()))
        })
        .collect()
}

/// 外側の壁時計締め切り超過・`try_wait()` 自体のエラーの経路で、supervisor
/// （直接の子プロセス）だけでなくその子孫（`_worker` が
/// `start_new_session=True` で作る別セッション・別プロセスグループの
/// プロセスを含む）もまとめて `SIGKILL` する（REQ-39「資源の上限」。
/// codex/review 指摘 P0「外側のタイムアウト時に学習プロセスを停止できない」。
/// issue #178 PR #233 レビュー。モジュール doc「孤児化の限界」参照）。
///
/// `root_pid`（= supervisor の pid）がまだ生きている前提で `ps` を実行し、
/// 子孫の一覧を取得してから 1 件ずつ `/bin/kill -s KILL <pid>` を送る。
/// 呼び出し元は本関数の後に必ず supervisor 自身を `Child::kill()`／
/// `Child::wait()` で回収すること（本関数は supervisor 自身の回収を行わず、
/// 子孫だけを対象にする）。
///
/// 単発の `kill` 送出では、送出直後に終了処理中だった子孫や `kill`
/// コマンド自体の一時的な失敗を取りこぼしうるため、`ps` で子孫が消えたことを
/// 確認できるまで最大 [`ORPHAN_SWEEP_MAX_ROUNDS`] 回、`ps`→`kill` を繰り返す
/// （codex/review 指摘 P0「プロセスツリーの掃除に失敗しても孫プロセスが
/// 動き続ける」。issue #178 PR #233 レビュー。REQ-39「資源の上限」）。
///
/// `ps`・`kill` 自体の失敗・タイムアウトは無視する（本関数はベストエフォート
/// の多層防御であり、これが失敗しても呼び出し元による直接の子の回収は
/// 妨げない。fail-open だが、直接の子の回収という主要な不変条件〔ゾンビを
/// 残さない〕は本関数の成否と独立に保たれる）。
///
/// 戻り値 `true` は、最終確認の `ps` で子孫が 1 件も見つからず（`ppid` チェ
/// ーン経由の掃除を確認できた）、かつ `fallback` スナップショット（`(pid,
/// lstart)`。[`run_train`] がポーリング中に記録した直近の子孫一覧）についても
/// [`kill_pids_best_effort`] で生存を確認できなかったことを示す。`false` は
/// いずれかが「確認できなかった」ことを示し、呼び出し元は「別セッションの
/// `_worker` 等が生き残っている可能性がある」ことを認識したうえで後続の判断
/// （エラー型への反映等）に用いること。
///
/// `ps_pid_ppid_pairs()` が失敗しても即座に諦めない: `ppid` チェーンでの
/// 確認ができない各ラウンドでも `fallback` へ直接 `SIGKILL` を送ることで、
/// 直接の子（supervisor）だけでなく子孫も止められるようにする（codex/review
/// 指摘 P0「`ps_pid_ppid_pairs()` 失敗時、関数が即 `false` を返し、呼び出し
/// 元は直接の子だけを kill する」。issue #178 PR #233 レビュー再指摘。
/// 修正前は最初のラウンドで `ps` が失敗すると即 `return false` し、後続
/// ラウンド・fallback のいずれも試みなかった）。
///
/// 純粋なリトライ・fallback 呼び出しの判断ロジックは [`sweep_with`] へ切り
/// 出し、実プロセスを使わずに単体テストできるようにしてある
/// （`tests::req39_sweep_with_retries_and_falls_back_when_ps_fails` 参照）。
#[cfg(unix)]
fn kill_process_tree_best_effort(root_pid: u32, fallback: &[(u32, String)]) -> bool {
    sweep_with(
        || ps_pid_ppid_pairs().map(|pairs| transitive_descendants(root_pid, &pairs)),
        |descendants| {
            for pid in descendants {
                run_bounded_fire_and_forget(
                    KILL_BIN,
                    &["-s", "KILL", &pid.to_string()],
                    ORPHAN_SWEEP_STEP_TIMEOUT,
                );
            }
        },
        fallback,
    )
}

/// [`kill_process_tree_best_effort`] のリトライ・fallback 判断ロジック
/// （`ps` 呼び出し・`kill` 送出そのものは引数のクロージャへ委譲する）。
///
/// `query`: 現在の子孫一覧を取得する（`ps` 失敗時は `None`）。
/// `kill_descendants`: `query` が返した子孫一覧へ `SIGKILL` を送る。
/// `fallback`: `ppid` チェーンでは辿れない孤児（supervisor 終了後に
/// reparent 済みの `_worker` 等）に備えた `(pid, lstart)` スナップショット。
///
/// `query` が失敗するラウンドでも `fallback` への
/// [`kill_pids_best_effort`] 呼び出しは必ず行う（fallback が空なら
/// 呼び出しても副作用は無く `true` を返す）。全ラウンドを終えても `query`
/// による確認が一度も取れなかった場合は `false`（確認できなかった）を返す。
#[cfg(unix)]
fn sweep_with(
    mut query: impl FnMut() -> Option<Vec<u32>>,
    mut kill_descendants: impl FnMut(&[u32]),
    fallback: &[(u32, String)],
) -> bool {
    let mut tree_clean = false;
    for round in 0..ORPHAN_SWEEP_MAX_ROUNDS {
        match query() {
            Some(descendants) if descendants.is_empty() => {
                tree_clean = true;
            }
            Some(descendants) => {
                kill_descendants(&descendants);
            }
            None => {
                // `ps` 自体が失敗: `ppid` チェーンでの確認はできないが、
                // 直近まで判明していた子孫スナップショットへは引き続き
                // `SIGKILL` を送る（fallback。ここで諦めない）。
            }
        }
        if tree_clean {
            break;
        }
        if !fallback.is_empty() {
            kill_pids_best_effort(fallback);
        }
        // 最終ラウンドの後に確認待ちしても無意味なので、最後の反復では
        // 待たずに抜ける（呼び出し元をこれ以上待たせない）。
        if round + 1 < ORPHAN_SWEEP_MAX_ROUNDS {
            std::thread::sleep(ORPHAN_SWEEP_RETRY_DELAY);
        }
    }
    if !tree_clean {
        // 全ラウンドを終えた後の最終確認。`kill` 送出直後の終了処理中
        // だった子孫が、この時点までに実際に消えている可能性があるため、
        // 諦める前にもう一度だけ確認する。
        tree_clean = matches!(query(), Some(descendants) if descendants.is_empty());
    }
    let fallback_clean = fallback.is_empty() || kill_pids_best_effort(fallback);
    tree_clean && fallback_clean
}

/// 既知の `(pid, lstart)`（[`run_train`] がポーリング中に記録した直近の
/// スナップショット。[`transitive_descendants_with_start`] 参照）へ直接
/// `SIGKILL` を送る。
///
/// [`kill_process_tree_best_effort`] は `root_pid`（supervisor）がまだ
/// 生きている前提で `ppid` チェーンを辿るが、supervisor が既に終了・
/// 回収済みの場合はチェーンが切れており（子孫は最も近い subreaper／init
/// へ reparent 済み）同じ方法では見つけられない。代わりに、supervisor が
/// まだ生きていた時点で確認できていた子孫の `(pid, lstart)` を直接指定して
/// kill する（codex/review 指摘 P0「supervisor が先に終了すると孤児ワーカー
/// を停止できない」。issue #178 PR #233 レビュー。REQ-39「資源の上限」）。
///
/// 各ラウンドで `ps` を再実行し、`pids` のうち **記録した起動時刻（`lstart`）
/// も一致する** ものだけを「まだ同一プロセスとして存在する」とみなして
/// 再送する（bare pid の一致だけで判定すると、極めて短時間に同じ pid が別
/// プロセスへ再利用された場合に無関係なプロセスを誤って `SIGKILL` して
/// しまいうる。codex/review 指摘 P0「kill_pids_best_effort が保存 PID の
/// 再利用確認をせず SIGKILL 送出、PID 再利用時に無関係プロセスを誤 kill
/// しうる」。issue #178 PR #233 レビュー）。`ppid` ではなく `lstart` で
/// 識別するのは、対象プロセスが supervisor の終了に伴い reparent 済みで
/// `ppid` が変化しているため（`ppid` 一致を条件にすると、reparent された
/// 正規の孤児まで「PID 再利用の疑いあり」として除外してしまう）。`ps` 自体が
/// 失敗した場合は識別による絞り込みができないため、記録済みの `pids` を
/// 安全側でそのまま kill 対象とする（fail-closed。`ps` 失敗時に何もしないと、
/// 実在する孤児ワーカーを一度も kill できず取りこぼす。codex/review 指摘 P0
/// 「ps_pid_ppid_pairs() 失敗時に既知孤児 PID を停止しない」。issue #178
/// PR #233 レビュー。誤 kill のリスクは `ps` 障害時に限定されるベスト
/// エフォートの範囲として許容する）。
///
/// 戻り値 `true` は、最終確認で `pids` のいずれも生存を確認できなかった
/// （掃除を確認できた）ことを示す。`pids` が空なら対象が無いため無条件に
/// `true`（[`sweep_with`] がこの戻り値をそのまま「fallback 側の掃除の成否」
/// として合成できるようにするための取り決め）。`false` は `ps` 自体の失敗が
/// 続いた・全ラウンドを終えても生存が確認されたことを示す（codex/review
/// 指摘 P0「`descendants_confirmed_clean: false` は通知だけで上限を強制
/// しない」。issue #178 PR #233 レビュー再指摘: 修正前は戻り値を持たず、
/// 呼び出し元は本関数の掃除が実際に効いたかを一切確認できなかった）。
#[cfg(unix)]
fn kill_pids_best_effort(pids: &[(u32, String)]) -> bool {
    if pids.is_empty() {
        return true;
    }
    for round in 0..ORPHAN_SWEEP_MAX_ROUNDS {
        let (alive, confirmed): (Vec<u32>, bool) = match ps_pid_start_pairs() {
            Some(current) => (
                pids.iter()
                    .filter(|(pid, start)| current.iter().any(|(p, s)| p == pid && s == start))
                    .map(|(pid, _)| *pid)
                    .collect(),
                true,
            ),
            None => (pids.iter().map(|(pid, _)| *pid).collect(), false),
        };
        if alive.is_empty() && confirmed {
            return true;
        }
        for pid in &alive {
            run_bounded_fire_and_forget(
                KILL_BIN,
                &["-s", "KILL", &pid.to_string()],
                ORPHAN_SWEEP_STEP_TIMEOUT,
            );
        }
        if round + 1 < ORPHAN_SWEEP_MAX_ROUNDS {
            std::thread::sleep(ORPHAN_SWEEP_RETRY_DELAY);
        }
    }
    // 全ラウンドを終えた後の最終確認。
    matches!(ps_pid_start_pairs(), Some(current)
        if !pids.iter().any(|(pid, start)| current.iter().any(|(p, s)| p == pid && s == start)))
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
/// [`kill_process_tree_best_effort`] の戻り値をそのまま
/// [`TrainProcessError::Wait`] へ伝播するために受け取る（`Child::wait()`
/// 自体が失敗した場合のみ使う。issue #178 PR #233 レビュー）。
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

/// `try_wait()` を 1 回ポーリングした結果を、壁時計締め切りとの関係で
/// 分類したもの（[`run_train`] のポーリングループが使う）。
#[cfg(unix)]
enum WaitOutcome {
    /// 締め切り内に終了を観測できた。
    Exited(ExitStatus),
    /// 終了はしていたが、観測できた時刻が既に締め切りを過ぎていた
    /// （codex/review 指摘 P1「try_wait() の Ok(Some(status)) を締め切り
    /// 判定より先に受理するため、期限超過後に終了したプロセスを成功扱い
    /// しうる」。issue #178 PR #233 レビュー再指摘）。`try_wait()` は既に
    /// `waitpid` 相当で子を回収済みのため、この時点で `root_pid`（＝
    /// `child.id()`）を再利用した無関係なプロセスが存在しうる。以後
    /// `kill_process_tree_best_effort`（`root_pid` を `ps` の `ppid` で
    /// 再検索する）や `Child::kill()` を呼んではならない。
    LateExit,
    /// 締め切りまでに終了を確認できなかった。
    TimedOut,
}

/// `observed_at`（`try_wait()` が `Ok(Some(status))` を返した時点の時刻）が
/// `deadline` より前かどうかを判定する。[`run_train`] のポーリングループ
/// から純粋関数として切り出し、実プロセスを使わずに単体テストできるように
/// する（issue #178 PR #233 レビュー再指摘 P1）。
#[cfg(unix)]
fn observed_within_deadline(observed_at: Instant, deadline: Instant) -> bool {
    observed_at < deadline
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
/// （[`RequestFileGuard`]）。unix 限定（windows 版は上記の
/// `#[cfg(not(unix))]` 版を参照）。
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
    // `wait_timeout` 相当が無いため）。supervisor が生きている間、`ps` の
    // `ppid` チェーンで確認できた子孫 pid の直近スナップショットを記録して
    // おく。supervisor が正常終了した後にパイプ読み取りが完了しない場合、
    // `_worker` の `ppid` チェーンは supervisor の終了と同時に切れてしまう
    // （reparent 済みになる）ため、終了後に辿り直すことができない。生きて
    // いる間の最後のスナップショットを使って直接 pid を kill する
    // （codex/review 指摘 P0「supervisor が先に終了すると孤児ワーカーを
    // 停止できない」。issue #178 PR #233 レビュー。[`kill_pids_best_effort`]
    // 参照）。以下のすべての異常系（`Err`・締め切り超過・期限超過後の
    // 終了観測）で、このスナップショットを [`kill_process_tree_best_effort`]
    // の fallback として渡す（Cursor Bugbot 指摘 Medium「壁時計タイムアウト
    // 時の掃除にポーリング中の last_known_descendants が適用されない」。
    // issue #178 PR #233 レビュー再指摘）。`(pid, lstart)` で記録するのは、
    // 後で `kill_pids_best_effort` が「記録時と同一プロセスか」を `lstart`
    // 一致で確認するため（`ppid` は reparent で変わるため使えない。
    // [`transitive_descendants_with_start`] のドキュメント参照）。
    let mut last_known_descendants: Vec<(u32, String)> = Vec::new();
    let mut last_descendant_snapshot_at: Instant;

    // `try_wait()` を一度も呼ぶ前に最初のスナップショットを取る。ポーリング
    // ループ内の最初の `try_wait()` が `Ok(Some(status))`（supervisor が
    // ポーリング開始前に既に正常終了していた）を返すと、ループ内の
    // スナップショット処理が一度も実行されず `last_known_descendants` が
    // 空のまま `_worker` の孤児化を検出・停止できなくなる（codex/review
    // 指摘 P0「supervisor 早期終了で子孫 PID スナップショット未取得のまま
    // 抜け、`_worker` 孤児化を検出・停止できない」。issue #178 PR #233
    // レビュー）。残る既知の限界: `spawn()` 直後から本スナップショットの
    // `ps` 実行までの間に supervisor が終了した場合は、この対策でも
    // 取りこぼす（モジュール doc「孤児化の限界」参照）。
    if let Some(pairs) = ps_pid_ppid_pairs() {
        let descendants = transitive_descendants_with_start(child.id(), &pairs);
        if !descendants.is_empty() {
            last_known_descendants = descendants;
        }
    }
    last_descendant_snapshot_at = Instant::now();

    let deadline = started + limits.wall_timeout();
    let outcome = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if observed_within_deadline(Instant::now(), deadline) {
                    break WaitOutcome::Exited(status);
                }
                break WaitOutcome::LateExit;
            }
            Ok(None) => {
                let now = Instant::now();
                if now >= deadline {
                    break WaitOutcome::TimedOut;
                }
                let need_snapshot =
                    now.duration_since(last_descendant_snapshot_at) >= DESCENDANT_SNAPSHOT_INTERVAL;
                if need_snapshot {
                    if let Some(pairs) = ps_pid_ppid_pairs() {
                        let descendants = transitive_descendants_with_start(child.id(), &pairs);
                        if !descendants.is_empty() {
                            last_known_descendants = descendants;
                        }
                    }
                    last_descendant_snapshot_at = now;
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                // `EINTR`（シグナル配送等による retryable なエラー）は
                // 健全な子プロセスの終了待ちでも到達しうるため、恒久的な
                // 監視失敗として扱わず単に次のポーリングへ進める
                // （[`poll_wait_bounded`] と同じ理由。以前はこの分岐が無く、
                // `EINTR` 発生時に生きている supervisor を誤って kill して
                // いた。issue #178 PR #233 レビュー「wait_after_kill が
                // EINTR を恒久エラー扱いする」と同種の欠陥がこのループにも
                // あった）。
                if Instant::now() >= deadline {
                    break WaitOutcome::TimedOut;
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(e) => {
                // `try_wait()` 自体のエラー（`EINTR` 以外。到達は稀）で即座に
                // 返すと `child` が drop され、実際にはまだ生き
                // ている子プロセスを kill／wait せず放置してゾンビ化・孤児化
                // させる（Cursor Bugbot 指摘「Child leaked on wait error」。
                // issue #178 PR #233 レビュー）。エラーを返す前に必ず回収を
                // 試みる。supervisor がまだ生きている前提でプロセスツリー
                // （`_worker` を含む子孫）も先に掃除してから、直接の子
                // （supervisor 自身）を回収する（codex/review 指摘 P0。
                // モジュール doc「孤児化の限界」参照）。kill 自体の失敗
                // （既に終了済み等）は元の `try_wait` エラーの報告を妨げない
                // よう無視する。掃除の成否（子孫が残っていないことを
                // 確認できたか）は `descendants_confirmed_clean` として
                // 呼び出し元へ伝える（codex/review 指摘 P0「プロセス
                // ツリーの掃除に失敗しても子孫が動き続ける」。issue #178
                // PR #233 レビュー）。直接の子の最終回収も無期限に待たず
                // `wait_after_kill` で上限を掛ける（Cursor Bugbot 指摘
                // Medium「Timeout wait can block forever」）。
                let descendants_confirmed_clean =
                    kill_process_tree_best_effort(child.id(), &last_known_descendants);
                let _ = child.kill();
                let _ = wait_after_kill(&mut child, descendants_confirmed_clean);
                return Err(TrainProcessError::Wait {
                    kind: e.kind(),
                    descendants_confirmed_clean,
                });
            }
        }
    };

    let status = match outcome {
        WaitOutcome::Exited(status) => status,
        WaitOutcome::LateExit => {
            // `try_wait()` が既に子（supervisor）を回収済みのため、`pid`
            // （= `child.id()`）は OS に返却され再利用されうる。この pid を
            // 対象に `kill_process_tree_best_effort`（`ps` の `ppid` で
            // 再検索する）や `Child::kill()` を再度呼ぶと、無関係な
            // プロセスを誤って終了させる恐れがある（codex/review 指摘 P1
            // 「try_wait() の成功を締め切り判定より先に受理する」。issue
            // #178 PR #233 レビュー再指摘）。生存確認済みの子孫スナップ
            // ショット（`lstart` で識別する [`kill_pids_best_effort`]）
            // だけを対象に直接 `SIGKILL` を送る。
            let descendants_confirmed_clean = kill_pids_best_effort(&last_known_descendants);
            return Err(TrainProcessError::WallTimeout {
                limit_ms: u64::try_from(limits.wall_timeout().as_millis()).unwrap_or(u64::MAX),
                descendants_confirmed_clean,
                // supervisor は `try_wait()` で既に回収済み。
                child_reaped: true,
            });
        }
        WaitOutcome::TimedOut => {
            // 締め切り超過: supervisor がまだ生きている前提でプロセス
            // ツリー（`_worker` を含む子孫）を先に掃除してから、supervisor
            // 自身を kill してから必ず `wait()` で回収する（ゾンビを残さ
            // ない。codex/review 指摘 P0「外側のタイムアウト時に学習
            // プロセスを停止できない」。issue #178 PR #233 レビュー。
            // モジュール doc「孤児化の限界」参照）。kill 自体の失敗
            // （既に終了済み等）は無視してよい。掃除の成否は
            // `descendants_confirmed_clean` として呼び出し元へ伝える
            // （codex/review 指摘 P0「プロセスツリーの掃除に失敗しても
            // 子孫が動き続ける」。issue #178 PR #233 レビュー）。
            let descendants_confirmed_clean =
                kill_process_tree_best_effort(child.id(), &last_known_descendants);
            let _ = child.kill();
            // `wait_after_kill` の失敗（`KillWaitTimedOut`／`Wait`）を `?` で
            // そのまま伝播すると、壁時計タイムアウトが検出された事実が
            // `WallTimeout`（`LimitExceeded`＝20）ではなく回収エラー
            // （`RuntimeError`＝70）として返ってしまい、REQ-21 の終了コード
            // 契約に違反する（codex/review 指摘 P1「wall timeout 超過後
            // wait_after_kill(...)? 失敗で WallTimeout が KillWaitTimedOut／
            // Wait に化ける」。issue #178 PR #233 レビュー。Cursor Bugbot 指摘
            // Medium も同一事象）。締め切り超過は `wait_after_kill` の成否に
            // かかわらず確定した事実のため、常に `WallTimeout` を返し、
            // 子プロセスを実際に回収できたか（ゾンビとして残っていないか）は
            // `child_reaped` フィールドで別途伝える。
            let child_reaped = wait_after_kill(&mut child, descendants_confirmed_clean).is_ok();
            return Err(TrainProcessError::WallTimeout {
                limit_ms: u64::try_from(limits.wall_timeout().as_millis()).unwrap_or(u64::MAX),
                descendants_confirmed_clean,
                child_reaped,
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

    // `recv_timeout` の失敗（期限内に届かない）だけでなく、届いた
    // `DrainedOutput::read_error`（`read()` 自体のエラーで打ち切られ、
    // `kept` を最後まで読み切れていない可能性がある。codex/review 指摘 P1
    // 「パイプ読み取りエラーを EOF として扱う」。issue #178 PR #233
    // レビュー）も同じ「読み取り未完了」として扱う。たまたま `kept` が
    // 有効な結果 JSON に見えても、読み切れていない出力を成功として
    // 受理しない（fail-closed。REQ-39「資源の上限」）。
    //
    // supervisor は既に正常終了しているため（`status` を取得済み）、読み
    // 取り未完了は「`_worker` が別セッションで生き残ってパイプの書き手を
    // 握り続けている」ケースを主に想定する。`kill_process_tree_best_effort`
    // が使う `ppid` チェーンは supervisor の終了と同時に切れてしまい
    // 使えないため、ポーリング中に記録しておいた最後の子孫スナップショット
    // （`last_known_descendants`）へ直接 kill を送ってから、読み取り未完了
    // として返す（codex/review 指摘 P0「supervisor が先に終了すると孤児
    // ワーカーを停止できない」。issue #178 PR #233 レビュー）。
    let Some(stdout_drain) = stdout_drain.filter(|d| !d.read_error) else {
        let _ = kill_pids_best_effort(&last_known_descendants);
        return Err(TrainProcessError::StdoutIncomplete);
    };
    // 標準エラー出力の読み取りタイムアウトも標準出力と同様にエラーとして
    // 扱う。`stderr_truncated: true` のまま `Ok(TrainRun)` を返すと、孤児化
    // した `_worker`（「孤児化の限界」節）がパイプを握り続けて学習が実際
    // には継続中でも成功と区別できなくなる（codex/review 指摘。issue #178
    // PR #233 レビュー。REQ-39「資源の上限」）。
    let Some(stderr_drain) = stderr_drain.filter(|d| !d.read_error) else {
        let _ = kill_pids_best_effort(&last_known_descendants);
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
        let dir = std::env::temp_dir();
        let launch_script = dir.join(format!(
            "fandhe-edge-train-test-unsupported-platform-{}.py",
            std::process::id()
        ));
        std::fs::write(&launch_script, b"").expect("write stub launch.py");
        let python = std::env::current_exe().expect("resolve current_exe");
        let launcher =
            WorkerLauncher::new(python, launch_script.clone()).expect("valid launcher fields");
        let request = test_request(Some(30));
        let limits = RunLimits::for_request(&request);
        let err = run_train(&launcher, &request, &dir, &limits).unwrap_err();
        let _ = std::fs::remove_file(&launch_script);
        assert!(matches!(err, TrainProcessError::UnsupportedPlatform));
        assert_eq!(err.exit_code(), ExitCode::RuntimeError);
        assert!(
            !dir.join("request.json").exists(),
            "request.json must not be written when platform is unsupported"
        );
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

    /// issue #178 PR #233 レビュー再指摘 P1「try_wait() の Ok(Some(status))
    /// を締め切り判定より先に受理するため、期限超過後に終了したプロセスを
    /// 成功扱いしうる」: 観測時刻が締め切りより前なら受理してよいが、
    /// 締め切りちょうど・締め切り後は受理してはならない。
    #[cfg(unix)]
    #[test]
    fn req39_observed_within_deadline_rejects_late_observation() {
        let deadline = Instant::now();
        let before = deadline - Duration::from_millis(1);
        let after = deadline + Duration::from_millis(1);
        assert!(observed_within_deadline(before, deadline));
        assert!(!observed_within_deadline(deadline, deadline));
        assert!(!observed_within_deadline(after, deadline));
    }

    /// `kill_pids_best_effort`: 対象が空なら（掃除すべき子孫が無いため）
    /// 無条件に「確認できた」（`true`）を返す。
    #[cfg(unix)]
    #[test]
    fn req39_kill_pids_best_effort_empty_is_confirmed_clean() {
        assert!(kill_pids_best_effort(&[]));
    }

    /// issue #178 PR #233 レビュー再指摘 P0「`ps_pid_ppid_pairs()` 失敗時、
    /// 関数が即 `false` を返し、呼び出し元は直接の子だけを kill する」:
    /// `ps` 相当の問い合わせ（`query`）が全ラウンドにわたって失敗し続けても、
    /// `sweep_with` は 1 回で諦めず、`fallback` への掃除（`kill_fallback`）を
    /// 複数回試みたうえで「確認できなかった」（`false`）を返す。
    #[cfg(unix)]
    #[test]
    fn req39_sweep_with_retries_and_falls_back_when_ps_fails() {
        // 実在しない pid・起動時刻の組み合わせ（実際の `ps` 出力には決して
        // 現れない）。`kill_pids_best_effort` が内部で呼ぶ実際の `/bin/ps` は
        // これを「生存していない」と判定するため、無関係なプロセスを誤って
        // `SIGKILL` する心配はない。
        let fallback = vec![(u32::MAX, "unreachable-lstart".to_string())];
        let query_calls = std::cell::Cell::new(0_u32);
        let cleaned = sweep_with(
            || {
                query_calls.set(query_calls.get() + 1);
                None
            },
            |_descendants: &[u32]| unreachable!("query always fails; must not report descendants"),
            &fallback,
        );
        // 修正前は最初の `query()` 失敗で即 `return false` していたため
        // `query_calls` は高々 1 回しか増えなかった。全ラウンド
        // （`ORPHAN_SWEEP_MAX_ROUNDS`）＋最終確認の 1 回まで retry しつつ
        // fallback を試みることを、呼び出し回数で確認する。
        assert_eq!(query_calls.get(), ORPHAN_SWEEP_MAX_ROUNDS + 1);
        assert!(!cleaned);
    }

    /// `sweep_with`: `query` が最初から空の子孫一覧を返す場合は、
    /// `fallback` が空であれば直ちに「確認できた」（`true`）を返す。
    #[cfg(unix)]
    #[test]
    fn req39_sweep_with_confirms_clean_when_tree_already_empty() {
        let cleaned = sweep_with(
            || Some(Vec::new()),
            |_descendants: &[u32]| unreachable!("no descendants to kill"),
            &[],
        );
        assert!(cleaned);
    }
}
