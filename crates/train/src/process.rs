//! 学習ワーカーの子プロセス起動・タイムアウト・終了コード写像（REQ-21・
//! REQ-34・REQ-39。issue #178）。
//!
//! Rust 側 CLI の `train` 工程（`stages::train`。#136）が、学習ワーカー
//! （`trainer/launch.py`。`train --request <path> --cancel-on-stdin-eof`）を
//! 引数リストで子プロセス起動し、壁時計タイムアウト
//! と標準出力／標準エラー出力の読み取り上限を掛けたうえで、
//! [`crate::result::TrainOutcome::from_worker_stdout`] へ渡して結果を得る
//! ための型・関数を提供する（[`run_train`] が公開入口）。
//!
//! # 子孫プロセスの掃除は lifeline（学習ワーカー側）に委ねる（issue #178
//! PR #233 レビュー: Rust 側でのプロセスグループ管理から全面移行）
//!
//! 以前の実装は、[`run_train`] が supervisor を新しいプロセスグループの
//! 先頭として起動し（`process_group(0)`）、壁時計タイムアウト時に
//! `/bin/kill -KILL -- -<pgid>` でそのグループ全体（`_worker` を含む）へ
//! 一括で `SIGKILL` を送る設計だった。この設計は「Rust 側が外部から
//! worker のプロセスグループを観測・操作する」という前提そのものに起因する
//! 構造的な欠陥が複数収束しなかった: 回収済み（reap 済み）の pgid が
//! 再利用されうるのに `kill(-pgid)` を送ってしまう・環境変数の設定だけで
//! 単独起動時の防御を無効化できてしまう・回収前のゾンビがグループに残る
//! ため `kill -0` による確認が常に「生存している」と誤判定する・
//! `WallTimeout` が別のエラーに化ける、等。
//!
//! そこで、**子孫プロセスの掃除を Rust 側から完全に切り離し、学習ワーカー
//! （`_worker`）自身が supervisor（本モジュールが起動する直接の子）の死を
//! 検知して自己終了する「lifeline」方式**へ全面移行した
//! （`trainer/src/fandhe_edge_trainer/supervisor.py` のモジュール docstring
//! 「lifeline」節参照）。概要: supervisor は `os.pipe()` の読み取り端だけを
//! `_worker` へ渡し、書き込み端を握り続ける。`_worker` は起動直後から
//! 読み取り端を block read する daemon スレッドを持ち、supervisor が
//! どのような形で終了しても（正常終了・内部タイムアウト・**本モジュールが
//! 送る `SIGKILL` を含む**）カーネルが書き込み端を自動的に閉じるため、
//! 必ず EOF を観測して自分自身とその子孫を `killpg` で終了させる。
//!
//! **これにより Rust 側は直接の子（supervisor）だけを把握すればよい**:
//! `_worker` を含む子孫プロセスへは一切触れない（`process_group`・
//! `/bin/kill` 呼び出し・`kill -0` による確認はすべて撤去した）。孫プロセス
//! がさらに `setsid` 等で lifeline の読み取り端を引き継がない別プロセスを
//! 作った場合は、この方式でも対象外である（限界として記録する。
//! `supervisor.py` のモジュール docstring 参照）。
//!
//! # 完了検知・タイムアウト時の回収順序
//!
//! [`run_train`] は、直接の子（supervisor）の標準出力が読み取り上限まで
//! 完了する（EOF・エラー）のを、既存の上限・期限つきで待ったうえで、
//! **残りの壁時計予算の範囲で** `Child::try_wait()` をポーリングし、実際に
//! 終了するのを待つ（[`poll_wait_bounded`]）。期限内に終了すれば、その
//! 終了状態をそのまま使う。
//!
//! 期限内に終了しなかった、または `try_wait()` 自体が失敗した場合は
//! `Child::kill()` で直接の子を `SIGKILL` してから [`wait_after_kill_until`] で
//! 回収し、[`TrainProcessError::WallTimeout`]（`LimitExceeded`＝20）として
//! 分類する（この分類は変えない）。直接の子は Rust 自身の未回収の子である
//! ため、`wait()` するまで pid が OS に返却されない（＝再利用されない）。
//! したがって `Child::kill()` を呼ぶ前に必ず「まだ reap していないか」を
//! 確認する必要がある: `try_wait()` が `Ok(Some(status))` を返した時点で
//! 既に reap 済みであり、それより後に同じ pid へ `kill()` を送ると、
//! 理論上は再利用された無関係なプロセスを誤って終了させうる（codex 指摘
//! P1。以前の「観測時刻が締め切り後でも成功扱いしてしまう」種類の競合と
//! 同根）。[`run_train`] はこの観測を [`WaitOutcome::LateExit`] として
//! 区別し、`Child::kill()` を呼ばずに `WallTimeout` として扱う。
//!
//! # キャンセル（REQ-34・TASK-34.1-1・#144・TASK-34.1-2・#145）
//!
//! [`run_train_cancellable`] は [`CancelToken`] が立つと、まず**協調キャンセル**を
//! 試み、猶予内に止まらなければ `SIGKILL` にフォールバックする。
//!
//! ## 協調キャンセル（std のみ。`SIGTERM` は使わない）
//!
//! supervisor は `stdin(Stdio::piped())` で起動し、[`std::process::ChildStdin`]
//! を実行中ずっと `Child` に持たせる（**途中で drop すると、それだけで
//! キャンセルになる**）。キャンセル時は stdin を閉じる（EOF）。supervisor は
//! EOF を受けて自ら worker を止め（`killpg` → 回収）、保持中の fd だけで
//! `out_dir` の予約を解放して終了する（`supervisor.py` モジュール docstring
//! 「協調キャンセル」）。猶予は [`RunLimits::cooperative_grace`]（既定
//! [`crate::limits::COOPERATIVE_CANCEL_GRACE_SECONDS`]）で、常に壁時計の期限より
//! 内側（`SIGKILL` 経路の回収待ち [`KILL_WAIT_TIMEOUT`] を残す）に切り詰める。
//! 壁時計の期限が優先する規則は従来どおり。`SIGTERM` による graceful cancel は
//! `libc`／`unsafe` が要りユーザー承認事項のため使わない。
//!
//! ## 整合の要（公開されている ⇔ supervisor が exit 0 と ok JSON を返した）
//!
//! supervisor は確定（`rename`）の直前までキャンセルを確認し、確定後に届いた
//! キャンセルは無視して成功を報告する。Rust 側は stdin を閉じた後も supervisor の
//! 終了状態と結果 JSON に従って分類する: 成功 JSON なら [`TrainRunEnd::Completed`]
//! （確定済み。`Cancelling` → `Completed`）、所定のキャンセル応答（exit 70・
//! `runtime_error`・固定メッセージ）は [`CancelStop::Cooperative`]、別のエラー JSON
//! （`invalid_request` 等の入力エラー）はキャンセル完了と断定せずエラー結果のまま
//! [`TrainRunEnd::Completed`] で返す。クラッシュ・終了状態の不整合は
//! [`CancelStop::Unconfirmed`] とし、[`CancelledRun::out_dir_residue`] で残置を
//! 報告する（クラッシュを協調完了として隠さない）。`SIGKILL` 後・応答未確認の
//! いずれでも、`out_dir` が公開済みの可能性（`NonEmpty`）か判定不能（`Unknown`）なら
//! `Cancelled` とせず [`TrainProcessError::CancelOutcomeUnconfirmed`] を返す
//! （「公開されている ⇔ 成功報告」の契約。fail-closed）。
//!
//! ## 責務分担
//!
//! | 項目 | 担当 |
//! | ---- | ---- |
//! | `out_dir` の予約・確定・解放（fd 束縛） | `supervisor.py` のみ |
//! | キャンセル時の worker 停止と予約の解放 | `supervisor.py`（協調経路） |
//! | キャンセル要求の伝達・猶予の管理・猶予超過時の `SIGKILL` | 本モジュール |
//! | `Completed`／`Cancelled` の最終判定 | 本モジュール |
//! | `SIGKILL` フォールバック後の残置の削除 | どちらも行わない（本モジュールは読み取り専用で検査して報告するだけ） |
//!
//! ## `SIGKILL` フォールバック（[`CancelStop::ForcedKill`]）
//!
//! 猶予内に supervisor が終了しなければ、従来どおり直接の子へ `Child::kill()`
//! （`SIGKILL`）を送って回収する。kill は未回収の直接の子にだけ送り、
//! `try_wait()` で既に終了していたら送らず通常経路へ合流する。キャンセルと
//! 壁時計締め切りが同時に成立したら壁時計を優先する規則（
//! [`classify_cancel_outcome`]）。kill の送出失敗で終了も確認できない場合は
//! `Cancelled` とせず監視を続け、壁時計の上限で回収する。回収確認の失敗は
//! エラーで返し、ジョブは `Failed` になる（REQ-39）。子の終了・回収後に届いた
//! キャンセルは結果の分類を完了して `Completed` を返す。
//!
//! この場合だけ supervisor が後始末できず、空の予約済み `out_dir`（
//! [`OutDirResidue::EmptyReservation`]）や書きかけの一時ディレクトリ、
//! 確定直後の窓では公開済みの成果物（[`OutDirResidue::NonEmpty`]）が残りうる。
//! 本モジュールは [`CancelledRun::out_dir_residue`] で読み取り専用の状態だけ
//! 報告し、削除・rename はしない。やり直しの案内は [`crate::restart`]
//! （TASK-34.3・#147。自動掃除はせず案内する）、
//! `job.json` の永続化とクラッシュ検出は [`crate::job_record`]（TASK-34.2・#146。実装済み）。キャンセルの終了コード写像は
//! TASK-33.x の承認事項。ジョブ状態の遷移は [`crate::job`]。
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
#[cfg(unix)]
use crate::limits::MAX_WORKER_STDERR_BYTES;
use crate::limits::{COOPERATIVE_CANCEL_GRACE_SECONDS, SUPERVISOR_SHUTDOWN_GRACE_SECONDS};
use crate::request::TrainRequest;
use crate::result::TrainOutcome;
use crate::time_allotment::CandidateRunner;

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
    /// 補助コンストラクタ。`trainer_dir` の発見は CLI の `stages::train` が
    /// 環境変数で暫定に行う（#136。配布形態は未確定。issue #178 実装計画 3.1）。
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
    /// 末尾に協調キャンセルの `--cancel-on-stdin-eof` を含む。
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
            // 協調キャンセルの opt-in（stdin の EOF で supervisor が自ら後始末
            // する。REQ-34・#145。モジュール doc「キャンセル」）。
            "--cancel-on-stdin-eof".into(),
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
    /// 協調キャンセルで supervisor の自己終了を待つ猶予（REQ-34・#145）。
    cooperative_grace: Duration,
}

impl RunLimits {
    /// `request.time_limit_seconds()` に基づく既定の外側締め切りを返す。
    #[must_use]
    pub fn for_request(request: &TrainRequest) -> Self {
        let seconds = u64::from(request.time_limit_seconds())
            .saturating_add(u64::from(SUPERVISOR_SHUTDOWN_GRACE_SECONDS));
        Self {
            wall_timeout: Duration::from_secs(seconds),
            cooperative_grace: Duration::from_secs(u64::from(COOPERATIVE_CANCEL_GRACE_SECONDS)),
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
            cooperative_grace: self.cooperative_grace,
        })
    }

    /// 協調キャンセルの猶予を既定より **短い** 値へ上書きする（テスト用途。
    /// `SIGKILL` フォールバックの確認を速くする）。0、または現在の猶予以上の
    /// 場合は `Err`（上限を緩めることはできない）。
    pub fn with_cooperative_grace(self, grace: Duration) -> Result<Self, TrainProcessError> {
        if grace.is_zero() || grace >= self.cooperative_grace {
            return Err(TrainProcessError::InvalidRunLimits);
        }
        Ok(Self {
            wall_timeout: self.wall_timeout,
            cooperative_grace: grace,
        })
    }

    #[must_use]
    pub fn wall_timeout(&self) -> Duration {
        self.wall_timeout
    }

    /// 協調キャンセルで supervisor の自己終了を待つ猶予。
    #[must_use]
    pub fn cooperative_grace(&self) -> Duration {
        self.cooperative_grace
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
    // `TrainRequest::new` は `config` 単体の大きさを検査済みだが、リクエスト
    // 全体（`config`＋`validation_inputs` 等）の大きさは `to_json_vec` が
    // 初めて判定するため、`to_json_vec` が直列化上限超過（`TrainRequestError::TooLarge`）を
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
    /// 単体テスト専用の組み立て（`job_record` の分類テストが、子プロセスを起動せずに
    /// 結果を作るため）。製品コードからは使えない。
    #[cfg(test)]
    pub(crate) fn for_test(outcome: TrainOutcome, exit_code: ExitCode) -> Self {
        Self {
            outcome,
            exit_code,
            elapsed: Duration::ZERO,
            worker_stderr: Vec::new(),
            stderr_truncated: false,
        }
    }

    /// テスト用: ワーカーのエラー結果で終わった実行（`restart` のテストが使う）。
    #[cfg(test)]
    pub(crate) fn for_test_error() -> Self {
        Self {
            outcome: TrainOutcome::Error(crate::result::WorkerFailure::for_test("runtime_error")),
            exit_code: ExitCode::RuntimeError,
            elapsed: Duration::ZERO,
            worker_stderr: Vec::new(),
            stderr_truncated: false,
        }
    }

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

/// 学習ジョブのキャンセル要求を運ぶ共有フラグ（REQ-34・TASK-34.1-1・#144）。
///
/// [`crate::job::JobHandle::cancel`] が立て、[`run_train_cancellable`] が
/// 待ちの刻みごとに確認する。クローン間で同じフラグを共有し、`Send + Sync`
/// なのでスレッドをまたいで使える。一度立てたら戻せない。
#[derive(Debug, Clone, Default)]
pub struct CancelToken(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl CancelToken {
    /// 立っていないトークンを作る。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// キャンセルを要求する（冪等）。
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// キャンセルが要求されたか。
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// [`run_train_cancellable`] の終わり方。キャンセルはエラーではない
/// 終わり方として表す。
///
/// キャンセルを REQ-21 の 7 種の終了コードのどれへ写すかは CLI の入出力契約
/// の設計事項で、TASK-33.x での承認事項（本 issue では決めない。そのため
/// [`TrainProcessError`] にはバリアントを足していない）。
#[derive(Debug)]
pub enum TrainRunEnd {
    /// 子が最後まで走り、結果を得た。
    Completed(TrainRun),
    /// キャンセルで止めた。成果物は持たない（成功として扱わない）。
    Cancelled(CancelledRun),
}

/// キャンセルの止め方（[`CancelledRun::stop`]）。REQ-34・#145。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelStop {
    /// 子プロセスの起動前にキャンセルされた（何も起動していない）。
    BeforeStart,
    /// supervisor が stdin の EOF を受けて自ら後始末し、終了した。予約は
    /// supervisor が fd で解放済みで、公開場所に何も残らない。
    Cooperative,
    /// 協調の猶予内に終了せず `SIGKILL` で止めた。supervisor の後始末は走って
    /// いないため、[`CancelledRun::out_dir_residue`] の状態が残りうる。
    ForcedKill,
    /// キャンセル送出後に supervisor が終了したが、所定のキャンセル応答
    /// （exit 70・`runtime_error`・固定メッセージ）を確認できなかった（クラッシュ・
    /// 外部からの終了・出力の不整合。`supervisor.py` が予約を解放した証拠が無い）。
    /// 成果物は確定していないが、[`CancelledRun::out_dir_residue`] に読み取り専用の
    /// 観測結果を持つ（異常を隠さない。削除はしない。REQ-34・#145）。
    Unconfirmed,
}

/// `SIGKILL` フォールバック後の `out_dir` の読み取り専用の観測結果
/// （[`CancelledRun::out_dir_residue`]）。**削除・rename は一切しない**。
/// やり直しの案内は [`crate::restart`]（自動掃除はしない。TASK-34.3・#147）。REQ-34。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutDirResidue {
    /// `out_dir` が存在しない（予約前、または解放済み）。
    Absent,
    /// 空のディレクトリが残っている（予約が解放されなかった。成果物は公開されて
    /// いない証拠）。
    EmptyReservation,
    /// 中身のあるディレクトリがある（確定直後にキャンセルされた窓では、公開済みの
    /// 成果物の可能性）。
    NonEmpty,
    /// symlink・ディレクトリ以外・読み取りエラーなど、判定できない。
    Unknown,
}

/// キャンセルで止めた実行の診断情報。ワーカーの出力は保持しない
/// （データ由来の文字列を持たない。`.claude/rules/security.md`）。
///
/// [`CancelStop::Cooperative`]・[`CancelStop::BeforeStart`] では公開場所に何も
/// 残らない（supervisor が fd で解放済み、または起動前）。[`CancelStop::ForcedKill`]
/// のときだけ supervisor の後始末が走らないため、[`Self::out_dir_residue`] に
/// 観測結果を持つ（削除はしない。案内は [`crate::restart`]。TASK-34.3・#147）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancelledRun {
    elapsed: Duration,
    child_spawned: bool,
    child_reaped: bool,
    signal: Option<i32>,
    stop: CancelStop,
    out_dir_residue: Option<OutDirResidue>,
}

impl CancelledRun {
    /// 子プロセスを起動する前にキャンセル済みだった場合の結果
    /// （`Queued` のまま取り消されたジョブ。REQ-34・#144）。
    pub(crate) fn before_start() -> Self {
        Self {
            elapsed: Duration::ZERO,
            child_spawned: false,
            child_reaped: true,
            signal: None,
            stop: CancelStop::BeforeStart,
            out_dir_residue: None,
        }
    }

    /// supervisor が協調キャンセルで自ら終了した場合の結果（REQ-34・#145）。
    #[cfg(unix)]
    fn cooperative(elapsed: Duration) -> Self {
        Self {
            elapsed,
            child_spawned: true,
            child_reaped: true,
            signal: None,
            stop: CancelStop::Cooperative,
            out_dir_residue: None,
        }
    }

    /// テスト用の任意の組の構築（`restart` の写像の網羅テストが使う）。
    #[cfg(test)]
    pub(crate) fn for_test(
        stop: CancelStop,
        out_dir_residue: Option<OutDirResidue>,
        elapsed: Duration,
    ) -> Self {
        Self {
            elapsed,
            child_spawned: !matches!(stop, CancelStop::BeforeStart),
            child_reaped: true,
            signal: None,
            stop,
            out_dir_residue,
        }
    }

    /// 開始からキャンセル完了までの経過時間。
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// 子プロセスを起動したか（起動前にキャンセルされたら `false`）。
    #[must_use]
    pub fn child_spawned(&self) -> bool {
        self.child_spawned
    }

    /// 直接の子を回収（wait）できたか。回収を確認できない場合は
    /// `Cancelled` ではなくエラーで返すため、`Cancelled` では常に `true`。
    #[must_use]
    pub fn child_reaped(&self) -> bool {
        self.child_reaped
    }

    /// 子を止めたシグナル番号（`SIGKILL` なら `Some(9)`）。協調キャンセル・
    /// 起動前キャンセルでは `None`（supervisor が自ら終了した／起動していない）。
    #[must_use]
    pub fn signal(&self) -> Option<i32> {
        self.signal
    }

    /// どの経路で止めたか。
    #[must_use]
    pub fn stop(&self) -> CancelStop {
        self.stop
    }

    /// `SIGKILL` フォールバック後・応答未確認（[`CancelStop::Unconfirmed`]）の
    /// `out_dir` の観測結果。それ以外では `None`（supervisor が解放済み、または起動前で、何も残らない）。
    #[must_use]
    pub fn out_dir_residue(&self) -> Option<OutDirResidue> {
        self.out_dir_residue
    }

    /// 所定のキャンセル応答を確認できないまま supervisor が終了した場合の結果
    /// （REQ-34・#145）。残置の観測結果を必ず付ける。
    #[cfg(unix)]
    fn unconfirmed(elapsed: Duration, residue: OutDirResidue) -> Self {
        Self {
            elapsed,
            child_spawned: true,
            child_reaped: true,
            signal: None,
            stop: CancelStop::Unconfirmed,
            out_dir_residue: Some(residue),
        }
    }

    /// `ForcedKill` の実行に、`out_dir` の観測結果を付ける。
    #[cfg(unix)]
    fn with_out_dir_residue(mut self, residue: OutDirResidue) -> Self {
        self.out_dir_residue = Some(residue);
        self
    }
}

/// ディレクトリ実体の識別子（`dev`・`ino`）。名前ではなく実体で同一性を判断する
/// ために使う（REQ-34・#145）。
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirId {
    dev: u64,
    ino: u64,
}

#[cfg(unix)]
impl DirId {
    /// `path`（symlink は名前解決と同じく辿る）がディレクトリなら、その識別子。
    fn of(path: &Path) -> Option<Self> {
        use std::os::unix::fs::MetadataExt;
        let meta = std::fs::metadata(path).ok()?;
        meta.is_dir().then(|| Self {
            dev: meta.dev(),
            ino: meta.ino(),
        })
    }
}

/// 子プロセスを起動する**前**に記録する、`root` と `out_dir` の親ディレクトリの
/// 実体（`dev`・`ino`）。supervisor の確定処理は自分が保持する親ディレクトリの
/// fd に束縛されるが、Rust 側の残置観測は名前で再解決するため、実行中に root が
/// 改名・差し替えされると、公開済みの実体を見ずに `Absent` と誤観測しうる。
/// 観測時に再解決した実体がここへ記録した実体と一致する場合だけ、`Absent`・
/// `EmptyReservation` を「未公開の証拠」として採用する（REQ-34・#145。
/// fail-closed）。記録できなかった場合は常に不一致（`Unknown`）になる。
#[cfg(unix)]
#[derive(Debug, Clone, Copy)]
struct OutDirAnchor {
    root: Option<DirId>,
    parent: Option<DirId>,
}

#[cfg(unix)]
impl OutDirAnchor {
    fn paths(request: &TrainRequest) -> (PathBuf, PathBuf) {
        let root = PathBuf::from(request.root());
        let mut parent = root.clone();
        let components = out_dir_components(request);
        // 最後の構成要素（予約されるディレクトリ自身）を除いたものが親。
        for part in components.iter().take(components.len().saturating_sub(1)) {
            parent.push(part);
        }
        (root, parent)
    }

    /// 起動前の実体を記録する（読み取りのみ）。
    fn capture(request: &TrainRequest) -> Self {
        let (root, parent) = Self::paths(request);
        Self {
            root: DirId::of(&root),
            parent: DirId::of(&parent),
        }
    }

    /// 名前を再解決した実体が、記録した実体と一致するか。
    fn still_matches(&self, request: &TrainRequest) -> bool {
        let (root, parent) = Self::paths(request);
        matches!((self.root, DirId::of(&root)), (Some(a), Some(b)) if a == b)
            && matches!((self.parent, DirId::of(&parent)), (Some(a), Some(b)) if a == b)
    }
}

/// 検証（`request.rs`）と同じ規則で正規化した `out_dir` の構成要素。
#[cfg(unix)]
fn out_dir_components(request: &TrainRequest) -> Vec<&str> {
    crate::request::relative_path_components(request.out_dir())
}

/// 正規化した `out_dir` の絶対パス（`root` 配下）。
#[cfg(unix)]
fn out_dir_path(request: &TrainRequest) -> PathBuf {
    let mut path = PathBuf::from(request.root());
    for part in out_dir_components(request) {
        path.push(part);
    }
    path
}

/// `SIGKILL` フォールバック後の `out_dir` を読み取り専用で観測する
/// （[`OutDirResidue`]）。`symlink_metadata` でディレクトリかを確かめ、`read_dir`
/// は最初の 1 件だけ読む（件数上限 1。REQ-39）。symlink は辿らず `Unknown`。
/// **削除・rename はしない**（REQ-34・#145）。
///
/// 名前の再解決だけで「未公開」を断定しない**唯一の場所**: root・親ディレクトリの
/// 実体が起動前の記録（[`OutDirAnchor`]）と一致しなければ `Unknown` を返す。
#[cfg(unix)]
fn inspect_out_dir_residue(request: &TrainRequest, anchor: &OutDirAnchor) -> OutDirResidue {
    if !anchor.still_matches(request) {
        return OutDirResidue::Unknown;
    }
    let path = out_dir_path(request);
    let meta = match std::fs::symlink_metadata(&path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return OutDirResidue::Absent,
        Err(_) => return OutDirResidue::Unknown,
    };
    if !meta.is_dir() {
        return OutDirResidue::Unknown;
    }
    match std::fs::read_dir(&path) {
        Ok(mut entries) => match entries.next() {
            None => OutDirResidue::EmptyReservation,
            Some(Ok(_)) => OutDirResidue::NonEmpty,
            Some(Err(_)) => OutDirResidue::Unknown,
        },
        Err(_) => OutDirResidue::Unknown,
    }
}

/// [`cancel_child`] の結果。
#[cfg(unix)]
enum CancelStep {
    /// 確認した時点で子は既に終了していた（回収済み。kill していない）。
    AlreadyExited(ExitStatus),
    /// キャンセル用パイプを閉じた後、`kill` を送らずに猶予内で自ら終了した
    /// （回収済み。協調キャンセル。REQ-34・#145）。呼び出し元は通常の結果分類へ
    /// 合流し、成功 JSON なら `Completed`、キャンセル応答なら `Cancelled(Cooperative)`、それ以外は `Cancelled(Unconfirmed)`。
    ExitedAfterSignal(ExitStatus),
    /// 協調の猶予内に終了せず `SIGKILL` を送って止めた。
    Cancelled(CancelledRun),
    /// `kill()` の送出に失敗し、直後の `try_wait()` でも終了を確認できなかった
    /// （子はまだ生きている可能性がある）。`Child` を drop しても子は止まらない
    /// ため、呼び出し元は監視を続け、壁時計の上限で改めて回収する
    /// （codex/review 指摘 P0。REQ-39「資源の上限」）。壁時計の期限で kill 後の
    /// 回収待ちを打ち切った場合も同じ扱いで、`WallTimeout` として回収する。
    StillRunning,
    /// kill 後の回収待ちが壁時計の期限に達して打ち切られた（または期限後に
    /// 回収を試みた）。`WallTimeout { child_reaped }` として返す。回収の成否は
    /// `child_reaped` で示す。回収失敗は経路を問わず
    /// `WallTimeout`（limit_exceeded）へ写し、runtime_error にしない
    /// （Cursor Bugbot 指摘 Medium。REQ-21・REQ-39）。
    WallDeadline { child_reaped: bool },
}

/// 子の回収結果（状態・観測時刻・自分の `kill()` が成功したか）。
/// [`classify_cancel_outcome`] の入力（REQ-34・REQ-39）。
#[cfg(unix)]
struct Reaped {
    status: ExitStatus,
    at: Instant,
    /// 回収までに自分の `kill()` が成功したか。失敗した・kill 前に終了していた
    /// 場合の `SIGKILL` 終了は外部由来の可能性があり `Cancelled` にしない。
    kill_delivered: bool,
    /// キャンセル用パイプを閉じ済みか（協調キャンセルを送出したか）。
    cancel_signalled: bool,
}

/// キャンセル結末の分類。キャンセルと壁時計期限の優先規則を持つ**唯一**の
/// 関数で、すべてのキャンセル経路が回収結果をここへ渡す（経路ごとに分岐を
/// 散らさない。codex/review 指摘 P1。REQ-34・REQ-39）。
///
/// 規則は「子の回収を観測した時刻が壁時計の期限より前か後か」だけで決める。
/// - 回収を確認できなかった: `WallDeadline { child_reaped: false }`
/// - 期限以後に回収した（キャンセル要求が先でも）: `WallDeadline { child_reaped: true }`
/// - 期限前に回収し、自分の kill が成功して `SIGKILL` 終了だった: `Cancelled`
///   （[`CancelStop::ForcedKill`]）
/// - 期限前に回収し、上記でなく、キャンセル用パイプを閉じた後に終了した:
///   `ExitedAfterSignal`（協調）
/// - 期限前に回収したが、パイプを閉じる前の終了・外部由来の `SIGKILL`（kill 失敗・
///   kill 前に終了）だった: `AlreadyExited`（通常の結果分類へ戻す。成功した学習を
///   `Cancelled` と誤記録しない）
///
/// `Cancelled` は自分の `kill()` の成功（`kill_delivered`）を条件にする。
#[cfg(unix)]
fn classify_cancel_outcome(
    reaped: Option<Reaped>,
    wall_deadline: Instant,
    started: Instant,
) -> CancelStep {
    use std::os::unix::process::ExitStatusExt;
    match reaped {
        None => CancelStep::WallDeadline {
            child_reaped: false,
        },
        Some(r) if r.at >= wall_deadline => CancelStep::WallDeadline { child_reaped: true },
        Some(r) if r.kill_delivered && r.status.signal() == Some(SIGKILL) => {
            CancelStep::Cancelled(CancelledRun {
                elapsed: started.elapsed(),
                child_spawned: true,
                child_reaped: true,
                signal: r.status.signal(),
                stop: CancelStop::ForcedKill,
                out_dir_residue: None,
            })
        }
        Some(r) if r.cancel_signalled => CancelStep::ExitedAfterSignal(r.status),
        Some(r) => CancelStep::AlreadyExited(r.status),
    }
}

/// キャンセル分岐。まず `try_wait()` で未回収か確かめ、未回収なら**協調
/// キャンセル**（キャンセル用パイプを閉じ、`min(coop_grace, 壁時計の期限から
/// [`KILL_WAIT_TIMEOUT`] を遡った時刻)` まで supervisor の自己終了を待つ。
/// REQ-34・#145）を試み、終了しなければ kill（`SIGKILL`）を送って回収する。
/// 既に回収済みの子へ kill を送らない（pid 再利用で無関係なプロセスを止めない。
/// モジュール doc「完了検知・タイムアウト時の回収順序」と同じ不変条件）。子孫は
/// lifeline に委ねる。待機エラー・回収失敗でも `Child` を手放さず壁時計の期限まで
/// 再試行し、結末は [`classify_cancel_outcome`] で 1 か所で決める。
/// `final_grace` は期限後の最後の回収待ち（通常は [`KILL_WAIT_TIMEOUT`]）。
/// `cancel_signalled` は呼び出しをまたいで持ち回る単調な状態（一度パイプを閉じたら
/// 戻さず、再入時は協調を繰り返さず kill へ進む）。
#[cfg(unix)]
fn cancel_child<C: ChildControl>(
    child: &mut C,
    started: Instant,
    wall_deadline: Instant,
    final_grace: Duration,
    coop_grace: Duration,
    cancel_signalled: &mut bool,
) -> CancelStep {
    // キャンセル処理全体の待機を `min(KILL_WAIT_TIMEOUT, 壁時計の残り)` で
    // 抑える（REQ-39「資源の上限」）。
    let mut deadline = bounded_deadline(Instant::now(), wall_deadline);
    // kill 成功の記録。全経路・全再試行で持ち回る単調な状態（`ensure_reaped` へ
    // `&mut` で渡し、戻り値で上書きしない）。
    let mut kill_delivered = false;
    match try_wait_interrupt_bounded(|| child.try_wait(), deadline) {
        Ok(Some(status)) => {
            let reaped = Some(Reaped {
                status,
                at: Instant::now(),
                kill_delivered: false,
                cancel_signalled: *cancel_signalled,
            });
            return classify_cancel_outcome(reaped, wall_deadline, started);
        }
        Ok(None) => {}
        Err(_) => {
            let reaped =
                ensure_reaped(child, wall_deadline, final_grace, &mut kill_delivered).map(|r| {
                    Reaped {
                        cancel_signalled: *cancel_signalled,
                        ..r
                    }
                });
            return classify_cancel_outcome(reaped, wall_deadline, started);
        }
    }
    // 協調キャンセル。待ちの上限は壁時計の期限から `KILL_WAIT_TIMEOUT` を遡った
    // 時刻で切り詰める（フォールバックの `SIGKILL` 経路の回収待ちを壁時計の内側に
    // 残す。さもないと、協調しない supervisor では常に期限以後の回収になり
    // `Cancelled` でなく `WallTimeout` になってしまう。壁時計優先の規則は不変）。
    if !*cancel_signalled && child.close_cancel_channel() {
        *cancel_signalled = true;
        let now = Instant::now();
        let coop_deadline =
            (now + coop_grace).min(wall_deadline.checked_sub(KILL_WAIT_TIMEOUT).unwrap_or(now));
        if let Ok(Some(status)) = poll_wait_bounded(child, coop_deadline) {
            let reaped = Some(Reaped {
                status,
                at: Instant::now(),
                kill_delivered: false,
                cancel_signalled: true,
            });
            return classify_cancel_outcome(reaped, wall_deadline, started);
        }
        // 猶予内に終了しなかった（または `try_wait()` が失敗した）: `SIGKILL` へ。
        deadline = bounded_deadline(Instant::now(), wall_deadline);
    }
    // `kill()` の送出に失敗して終了も確認できないときは `StillRunning` を返し、
    // 呼び出し元が監視を続けて壁時計の上限で回収する。
    if child.kill().is_err() {
        // kill 送出に失敗しても、直前に既に終了していた可能性を再確認する。
        return match child.try_wait() {
            Ok(Some(status)) => {
                let reaped = Some(Reaped {
                    status,
                    at: Instant::now(),
                    kill_delivered: false,
                    cancel_signalled: *cancel_signalled,
                });
                classify_cancel_outcome(reaped, wall_deadline, started)
            }
            _ => CancelStep::StillRunning,
        };
    }
    kill_delivered = true;
    let reaped = match wait_after_kill_until(child, deadline) {
        Some(status) => Some(Reaped {
            status,
            at: Instant::now(),
            kill_delivered,
            cancel_signalled: *cancel_signalled,
        }),
        None => {
            ensure_reaped(child, wall_deadline, final_grace, &mut kill_delivered).map(|r| Reaped {
                cancel_signalled: *cancel_signalled,
                ..r
            })
        }
    };
    classify_cancel_outcome(reaped, wall_deadline, started)
}

/// `try_wait` を 1 回確認する。`Interrupted`（`EINTR`）は再試行するが、毎回
/// `deadline` を確認し、到達したら `Interrupted` のエラーで返す（無限ループ
/// 防止。REQ-39）。`Ok(None)` は「まだ生きている」。
#[cfg(unix)]
fn try_wait_interrupt_bounded(
    mut try_wait: impl FnMut() -> std::io::Result<Option<ExitStatus>>,
    deadline: Instant,
) -> std::io::Result<Option<ExitStatus>> {
    loop {
        match try_wait() {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                if Instant::now() >= deadline {
                    return Err(e);
                }
            }
            other => return other,
        }
    }
}

/// 壁時計超過のエラー（`limit_exceeded`）を作る。
#[cfg(unix)]
fn wall_timeout_error(limits: &RunLimits, child_reaped: bool) -> TrainProcessError {
    TrainProcessError::WallTimeout {
        limit_ms: u64::try_from(limits.wall_timeout().as_millis()).unwrap_or(u64::MAX),
        child_reaped,
    }
}

/// キャンセルを採用するか。壁時計の期限（`deadline`）より前に観測した
/// キャンセルだけを採用し、期限後は `WallTimeout` の分類を優先する
/// （codex/review 指摘 P1。REQ-34・REQ-39）。
#[cfg(unix)]
fn cancel_applies(now: Instant, deadline: Instant, cancel: &CancelToken) -> bool {
    now < deadline && cancel.is_cancelled()
}

/// 子プロセスの操作（`Child` の kill・非ブロッキング待ち）。エラー注入
/// テストのための差し替え点で、実運用は [`Child`] の実装のみ。
#[cfg(unix)]
trait ChildControl {
    fn kill(&mut self) -> std::io::Result<()>;
    fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>>;
    /// キャンセル用パイプ（supervisor の stdin）を閉じる（協調キャンセルの
    /// 要求。REQ-34・#145）。閉じられたら `true`、既に閉じ済み・無い場合は
    /// `false`（呼び出し元は協調を諦めて `SIGKILL` へ進む）。
    fn close_cancel_channel(&mut self) -> bool;
}

#[cfg(unix)]
impl ChildControl for Child {
    fn kill(&mut self) -> std::io::Result<()> {
        Child::kill(self)
    }
    fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        Child::try_wait(self)
    }
    fn close_cancel_channel(&mut self) -> bool {
        // `ChildStdin` を drop すると書き込み端が閉じ、supervisor は EOF を観測する。
        self.stdin.take().is_some()
    }
}

/// 待機エラー後に、子が回収されたと確認できるまで `Child` を手放さない。
/// `retry_until`（壁時計の期限）まで `kill()` と `try_wait()` を毎周回
/// 再試行し、期限後は最後の `kill()` と `final_grace` 以内の有界な回収待ちを
/// する。回収を確認できなければ `None`、確認できれば状態と観測時刻。`None` は常に
/// 壁時計の期限を過ぎた後なので、呼び出し側は経路を問わず
/// `WallTimeout { child_reaped: false }`（limit_exceeded）で返す。観測時刻が
/// 期限以後なら `child_reaped: true` の `WallTimeout`（REQ-21・REQ-39）。
#[cfg(unix)]
fn ensure_reaped<C: ChildControl>(
    child: &mut C,
    retry_until: Instant,
    final_grace: Duration,
    kill_delivered: &mut bool,
) -> Option<Reaped> {
    // `kill_delivered` は単調（一度 true になったら戻さない）。再試行の kill が
    // 「既に終了」で失敗しても、それ以前に成功した kill の記録を消さない
    // （codex/review 指摘 P1・Cursor Bugbot 指摘。REQ-34）。
    loop {
        *kill_delivered |= child.kill().is_ok();
        if let Ok(Some(status)) = child.try_wait() {
            return Some(Reaped {
                status,
                at: Instant::now(),
                kill_delivered: *kill_delivered,
                cancel_signalled: false,
            });
        }
        if Instant::now() >= retry_until {
            break;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    *kill_delivered |= child.kill().is_ok();
    let grace_deadline = Instant::now() + final_grace;
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return Some(Reaped {
                status,
                at: Instant::now(),
                kill_delivered: *kill_delivered,
                cancel_signalled: false,
            });
        }
        if Instant::now() >= grace_deadline {
            return None;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// 壁時計超過後の回収。期限は既に過ぎているため、`ensure_reaped` の最後の
/// kill と `final_grace` 以内の有界な回収待ちに当たる。回収できたかを返し、
/// できなくても呼び出し側は `WallTimeout { child_reaped: false }` を返す
/// （超過を `runtime_error` に化けさせない。REQ-21・REQ-39）。
#[cfg(unix)]
fn reap_after_wall_timeout<C: ChildControl>(child: &mut C, final_grace: Duration) -> bool {
    ensure_reaped(child, Instant::now(), final_grace, &mut false).is_some()
}

/// `SIGKILL` のシグナル番号（`Child::kill()` が送る値。unix で共通）。
#[cfg(unix)]
const SIGKILL: i32 = 9;

/// `SIGKILL` 送出後、直接の子プロセスの終了待ちあたりの上限（REQ-39
/// 「資源の上限」）。`SIGKILL` は通常即座に効くため、この上限に達するのは
/// 割り込み不可能な OS 側の待ち（D state）等の極めて稀なケースに限られる
/// （Cursor Bugbot 指摘 Medium「Timeout wait can block forever」。issue
/// #178 PR #233 レビュー。[`wait_after_kill_until`] 参照）。
#[cfg(unix)]
const KILL_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

/// `Child::try_wait()` を `deadline` までポーリングする共通ヘルパー。
///
/// `try_wait()` 自体が `ErrorKind::Interrupted`（`EINTR`。シグナル配送等で
/// 発生しうる retryable なエラー）を返した場合は打ち切らず再試行する
/// （[`drain_capped`] の `EINTR` 扱いと同じ理由。Cursor Bugbot 指摘 Medium
/// 「Pipe reads fail on interrupt」と同種の欠陥が [`wait_after_kill_until`] にも
/// あった：修正前は `EINTR` を他の `try_wait()` エラーと同列に扱い、
/// `SIGKILL` 送出後で実際にはまだ生きているだけのプロセスを、回収に失敗した
/// ものとして即座にエラー化していた。issue #178 PR #233 レビュー）。
///
/// 戻り値: `Ok(Some(status))` は終了を確認できた、`Ok(None)` は `deadline`
/// まで待っても終了しなかった（呼び出し元は上限超過として扱う）、
/// `Err(e)`（`EINTR` 以外）は `try_wait()` 自体が失敗したことを示す。
#[cfg(unix)]
fn poll_wait_bounded<C: ChildControl>(
    child: &mut C,
    deadline: Instant,
) -> std::io::Result<Option<ExitStatus>> {
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

/// キャンセル待機の期限。`now + KILL_WAIT_TIMEOUT` と壁時計の期限の早い方
/// （壁時計の上限を超えて待たない。REQ-39）。
#[cfg(unix)]
fn bounded_deadline(now: Instant, wall_deadline: Instant) -> Instant {
    (now + KILL_WAIT_TIMEOUT).min(wall_deadline)
}

/// kill（`SIGKILL` 送出）の直後に呼び、終了を `deadline` までポーリングして
/// 待つ。素朴な `wait()` は D state 等で無期限にブロックしうるため上限つきに
/// する（Cursor Bugbot 指摘 Medium「Timeout wait can block forever」。issue
/// #178 PR #233 レビュー。REQ-39「資源の上限」）。回収できれば `Some(status)`、
/// 期限に達した・`try_wait()` が失敗した場合は `None`（呼び出し元は
/// [`ensure_reaped`] で再試行し、回収を主張しない）。子孫の掃除は学習ワーカー
/// 側の lifeline に委ねる（モジュール doc 参照）。
#[cfg(unix)]
fn wait_after_kill_until<C: ChildControl>(child: &mut C, deadline: Instant) -> Option<ExitStatus> {
    poll_wait_bounded(child, deadline).ok().flatten()
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

/// unix 以外では子を起動せず、キャンセル済みなら `Cancelled`（起動前キャンセル）、
/// そうでなければ [`TrainProcessError::UnsupportedPlatform`] を返す（fail-closed）。
/// 起動前キャンセルを未対応プラットフォームより優先し、unix 版と同じ結果に
/// そろえる（`Start` 直後にキャンセルが届いても戻り値とジョブ状態が OS に
/// よらず一致する。codex/review 指摘 P1。REQ-34）。
#[cfg(not(unix))]
pub fn run_train_cancellable(
    _launcher: &WorkerLauncher,
    _request: &TrainRequest,
    _job_dir: &Path,
    _limits: &RunLimits,
    cancel: &CancelToken,
) -> Result<TrainRunEnd, TrainProcessError> {
    if cancel.is_cancelled() {
        return Ok(TrainRunEnd::Cancelled(CancelledRun::before_start()));
    }
    Err(TrainProcessError::UnsupportedPlatform)
}

/// [`run_train`] を [`CandidateRunner`]（[`crate::search::run_search`] が候補を
/// 実行する接合点）として使うアダプター（REQ-18・REQ-34・REQ-39。issue #84
/// PR #238・選択肢 2）。
///
/// 各候補を、その `request.time_limit_seconds()` に基づく既定の壁時計締め切り
/// （[`RunLimits::for_request`]）で子プロセス実行する。リクエストが
/// `validation_inputs` を持つ場合、学習直後の validation 予測も同じ子プロセス・
/// 同じ締め切りの中で行われる（予測時間も締め切りに含まれる）。壁時計超過は
/// [`TrainProcessError::WallTimeout`] で、[`CandidateRunner::is_wall_timeout`]
/// がそれを候補単位の時間切れとして見分ける。`job_dir` はジョブ管理側が
/// 用意した専用ディレクトリで、候補は逐次実行されるため共有できる
/// （`request.json` は実行のたびに新規作成・削除される。[`run_train`] 参照）。
#[derive(Debug)]
pub struct WorkerCandidateRunner<'a> {
    launcher: &'a WorkerLauncher,
    job_dir: &'a Path,
    wall_timeout_override: Option<Duration>,
}

impl<'a> WorkerCandidateRunner<'a> {
    /// 既定の締め切り（各リクエストの `time_limit_seconds` ＋
    /// [`SUPERVISOR_SHUTDOWN_GRACE_SECONDS`]）で実行するアダプターを作る。
    #[must_use]
    pub fn new(launcher: &'a WorkerLauncher, job_dir: &'a Path) -> Self {
        Self {
            launcher,
            job_dir,
            wall_timeout_override: None,
        }
    }

    /// 外側の壁時計締め切りを、既定より **短い** 固定値へ上書きする
    /// （テスト用途。[`RunLimits::with_wall_timeout`] と同じく、既定以上の
    /// 値は実行時に [`TrainProcessError::InvalidRunLimits`] になる）。
    #[must_use]
    pub fn with_wall_timeout(mut self, timeout: Duration) -> Self {
        self.wall_timeout_override = Some(timeout);
        self
    }
}

impl CandidateRunner for WorkerCandidateRunner<'_> {
    type Error = TrainProcessError;

    fn run(&mut self, request: &TrainRequest) -> Result<TrainOutcome, Self::Error> {
        let mut limits = RunLimits::for_request(request);
        if let Some(timeout) = self.wall_timeout_override {
            limits = limits.with_wall_timeout(timeout)?;
        }
        let run = run_train(self.launcher, request, self.job_dir, &limits)?;
        Ok(run.outcome().clone())
    }

    fn is_wall_timeout(error: &Self::Error) -> bool {
        matches!(error, TrainProcessError::WallTimeout { .. })
    }
}

/// `try_wait()` を 1 回ポーリングした結果を、壁時計締め切りとの関係で
/// 分類したもの（[`run_train`] のポーリングループが使う）。
#[cfg(unix)]
enum WaitOutcome {
    /// 締め切り内に終了を観測できた。
    Exited(ExitStatus),
    /// 終了はしていたが、観測できた時刻が既に締め切りを過ぎていた
    /// （codex 指摘 P1「try_wait() の Ok(Some(status)) を締め切り判定より
    /// 先に受理するため、期限超過後に終了したプロセスを成功扱いしうる」）。
    /// `try_wait()` は既に `waitpid` 相当で子を回収済みのため、この時点で
    /// pid を再利用した無関係なプロセスが存在しうる。以後 `Child::kill()`
    /// を呼んではならない。
    LateExit,
    /// 締め切りまでに終了を確認できなかった。
    TimedOut,
}

/// `observed_at`（`try_wait()` が `Ok(Some(status))` を返した時点の時刻）が
/// `deadline` より前かどうかを判定する。[`run_train`] のポーリングループ
/// から純粋関数として切り出し、実プロセスを使わずに単体テストできるように
/// する（issue #178 PR #233 レビュー）。
#[cfg(unix)]
fn observed_within_deadline(observed_at: Instant, deadline: Instant) -> bool {
    observed_at < deadline
}

/// 学習ワーカーを子プロセスとして起動し、壁時計タイムアウトと出力の読み
/// 取り上限を掛けたうえで、検証済みの結果 [`TrainRun`] を得る（REQ-21・
/// REQ-34・REQ-39。issue #178 の公開入口）。
///
/// 手順: (1) `job_dir/request.json` を新規作成 → (2) supervisor を固定
/// argv・`env_clear()`＋許可リストの環境・`current_dir(job_dir)` で起動
/// → (3) 標準出力が読み取り上限まで完了する（EOF・エラー）のを、残りの
/// 壁時計予算だけ待つ → (4) 残りの壁時計予算の範囲で `try_wait()` を
/// ポーリングし、直接の子（supervisor）が実際に終了するのを待つ（期限内に
/// 終了すればその終了状態を使う。期限超過・異常系では `Child::kill()` で
/// 強制終了してから回収する）→ (5) [`classify_exit`] で結果を検証。
/// すべての経路（成功・失敗・タイムアウト）で `request.json` を削除する
/// （[`RequestFileGuard`]）。**子孫プロセス（`_worker` を含む）の掃除には
/// 関与しない**（学習ワーカー側の lifeline に委ねる設計。モジュール doc
/// 参照）。unix 限定（windows 版は上記の `#[cfg(not(unix))]` 版を参照）。
#[cfg(unix)]
pub fn run_train_cancellable(
    launcher: &WorkerLauncher,
    request: &TrainRequest,
    job_dir: &Path,
    limits: &RunLimits,
    cancel: &CancelToken,
) -> Result<TrainRunEnd, TrainProcessError> {
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
    // 起動前にキャンセル済みなら、子プロセスも `request.json` も作らずに
    // 返す（`Queued` のまま取り消されたジョブ。REQ-34・#144）。
    if cancel.is_cancelled() {
        return Ok(TrainRunEnd::Cancelled(CancelledRun::before_start()));
    }
    let guard = write_request_file(job_dir, request)?;
    let started = Instant::now();

    let mut command = Command::new(&launcher.python);
    command
        .args(launcher.argv(&guard.path))
        .env_clear()
        .current_dir(job_dir)
        // キャンセル用パイプ（協調キャンセル。REQ-34・#145）。書き込み端
        // （`ChildStdin`）は `Child` が実行中ずっと保持し、`cancel_child` だけが
        // 閉じる。途中で drop すると、それだけで supervisor がキャンセル扱いに
        // なるため、`Child::stdin` を取り出して別の場所へ動かさない。
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for key in ENV_ALLOWLIST {
        if let Ok(value) = std::env::var(key) {
            command.env(key, value);
        }
    }

    // 最初の確認から `write_request_file`・環境構築を経る間にキャンセルが
    // 要求された場合も、子プロセスを起動しない。`guard` は return で drop され
    // `request.json` が削除される（REQ-34・#144。codex/review 指摘 P1）。
    // 本確認以降の要求は、起動後の監視ループが子の kill として処理する。
    if cancel.is_cancelled() {
        return Ok(TrainRunEnd::Cancelled(CancelledRun::before_start()));
    }
    // 残置観測の同一性確認用に、起動前の root・親ディレクトリの実体を記録する
    // （REQ-34・#145）。
    let anchor = OutDirAnchor::capture(request);
    let mut child = command
        .spawn()
        .map_err(|e| TrainProcessError::Spawn { kind: e.kind() })?;

    // stdout・stderr の読み取りスレッドを起動する前にパイプを取り出す
    // （書き込み側を親プロセスが握ったままにしない。取り出し忘れると
    // `Child` が drop される際にパイプが閉じられず readers が終わらない
    // ことがある）。
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let stdout_rx =
        stdout_pipe.map(|pipe| spawn_reader(pipe, request.max_result_bytes().saturating_add(1)));
    let stderr_rx = stderr_pipe.map(|pipe| spawn_reader(pipe, MAX_WORKER_STDERR_BYTES));

    let deadline = started + limits.wall_timeout();

    // 標準出力の読み取りスレッドが完了する（EOF・上限超過・エラーの
    // いずれか）のを、残りの壁時計予算だけ待つ。hang 中でもキャンセルに
    // 反応できるよう、待ちを [`POLL_INTERVAL`] 刻みに分割し、各刻みで
    // トークンを確認する（壁時計の上限そのものは緩めない）。
    let mut pre_exited: Option<ExitStatus> = None;
    // キャンセル用パイプを閉じたか（協調キャンセルを送出したか）。閉じた後は、
    // supervisor の終了状態と結果 JSON に従って `Completed`／`Cancelled` を決める。
    let mut cancel_signalled = false;
    let cooperative_grace = limits.cooperative_grace();
    let mut stdout_wait = None;
    if let Some(rx) = stdout_rx.as_ref() {
        // キャンセルは各刻みで確認する。`StillRunning`（kill 送出失敗で停止を
        // 確認できない）の場合も待ちを一括せず、次の刻みで kill を再試行する
        // （codex/review 指摘 P1。REQ-34・REQ-39）。壁時計の上限は緩めない。
        let first = loop {
            // 壁時計の期限後に観測したキャンセルは採用しない（期限超過を
            // `Cancelled` で隠さない。codex/review 指摘 P1。REQ-39）。
            if pre_exited.is_none() && cancel_applies(Instant::now(), deadline, cancel) {
                match cancel_child(
                    &mut child,
                    started,
                    deadline,
                    KILL_WAIT_TIMEOUT,
                    cooperative_grace,
                    &mut cancel_signalled,
                ) {
                    CancelStep::Cancelled(run) => {
                        return forced_kill_run(run, request, &anchor).map(TrainRunEnd::Cancelled);
                    }
                    // キャンセルが間に合わず子は既に終了していた（kill しない）、
                    // または協調キャンセルで自ら終了した。通常経路へ合流し、
                    // stdout を締め切りまで待つ。
                    CancelStep::AlreadyExited(status) | CancelStep::ExitedAfterSignal(status) => {
                        pre_exited = Some(status);
                    }
                    CancelStep::StillRunning => {}
                    CancelStep::WallDeadline { child_reaped } => {
                        return Err(wall_timeout_error(limits, child_reaped));
                    }
                }
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(remaining.min(POLL_INTERVAL)) {
                Err(mpsc::RecvTimeoutError::Timeout) if Instant::now() < deadline => continue,
                other => break other,
            }
        };
        stdout_wait = Some(first);
    }
    // `stdout_wait` が `None` になるのは `stdout_pipe` を取得できなかった
    // 場合だが、本関数は常に `stdout(Stdio::piped())` を指定するため
    // 到達しない防御的分岐。到達したとしても後続の `match` が
    // `StdoutIncomplete` として安全側に倒す。
    let stdout_timed_out = matches!(
        stdout_wait,
        Some(Err(mpsc::RecvTimeoutError::Timeout)) | None
    );

    // 直接の子（supervisor）が実際に終了するのを、残りの壁時計予算だけ
    // `try_wait()` でポーリングする。`Ok(Some(status))` の観測が締め切り後
    // になった場合は `WaitOutcome::LateExit` として区別し、既に reap 済みの
    // pid（再利用されうる）へ `Child::kill()` を送らない（codex 指摘 P1。
    // モジュール doc「完了検知・タイムアウト時の回収順序」参照）。
    let wait_deadline = deadline;
    let outcome = loop {
        if let Some(status) = pre_exited.take() {
            if observed_within_deadline(Instant::now(), wait_deadline) {
                break WaitOutcome::Exited(status);
            }
            break WaitOutcome::LateExit;
        }
        // キャンセルは期限内に観測した場合だけ採用する。期限後に観測した
        // キャンセルより壁時計超過（`WallTimeout`）を優先し、超過を
        // `Cancelled` で隠さない（codex/review 指摘 P1。REQ-39）。
        if cancel_applies(Instant::now(), wait_deadline, cancel) {
            match cancel_child(
                &mut child,
                started,
                wait_deadline,
                KILL_WAIT_TIMEOUT,
                cooperative_grace,
                &mut cancel_signalled,
            ) {
                CancelStep::Cancelled(run) => {
                    return forced_kill_run(run, request, &anchor).map(TrainRunEnd::Cancelled);
                }
                CancelStep::AlreadyExited(status) | CancelStep::ExitedAfterSignal(status) => {
                    pre_exited = Some(status);
                    continue;
                }
                CancelStep::WallDeadline { child_reaped } => {
                    return Err(wall_timeout_error(limits, child_reaped));
                }
                CancelStep::StillRunning => {
                    // 生存中の子の監視を続ける（次の周回で kill を再試行し、
                    // 壁時計の上限に達すれば `TimedOut` として回収する）。
                    if Instant::now() >= wait_deadline {
                        break WaitOutcome::TimedOut;
                    }
                    std::thread::sleep(POLL_INTERVAL);
                    continue;
                }
            }
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                if observed_within_deadline(Instant::now(), wait_deadline) {
                    break WaitOutcome::Exited(status);
                }
                break WaitOutcome::LateExit;
            }
            Ok(None) => {
                if Instant::now() >= wait_deadline {
                    break WaitOutcome::TimedOut;
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                // `EINTR`（シグナル配送等による retryable なエラー）は
                // 健全な子プロセスの終了待ちでも到達しうるため、恒久的な
                // 監視失敗として扱わず単に次のポーリングへ進める
                // （[`poll_wait_bounded`] と同じ理由）。
                if Instant::now() >= wait_deadline {
                    break WaitOutcome::TimedOut;
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(e) => {
                // `try_wait()` 自体のエラー（`EINTR` 以外。到達は稀）で
                // 即座に返すと `child` が drop され、実際にはまだ生きている
                // 子プロセスを kill／wait せず放置してゾンビ化させる
                // （Cursor Bugbot 指摘「Child leaked on wait error」）。
                // エラーを返す前に必ず回収を試みる。
                // 回収を確認できるまで `Child` を手放さず、壁時計の期限まで
                // kill と回収を再試行する。確認できなければ
                // `WallTimeout { child_reaped: false }`（codex/review 指摘 P0。REQ-39）。
                return match ensure_reaped(&mut child, wait_deadline, KILL_WAIT_TIMEOUT, &mut false)
                {
                    // 回収を観測したのが期限以後なら壁時計超過（キャンセルと
                    // 同じ規則。[`classify_cancel_outcome`]）。
                    Some(r) if r.at >= wait_deadline => Err(wall_timeout_error(limits, true)),
                    Some(_) => Err(TrainProcessError::Wait { kind: e.kind() }),
                    None => Err(wall_timeout_error(limits, false)),
                };
            }
        }
    };

    let (status, timed_out_by_wait, child_reaped) = match outcome {
        WaitOutcome::Exited(status) => (Some(status), false, true),
        WaitOutcome::LateExit => {
            // 既に `try_wait()` で reap 済み。pid は OS に返却され再利用され
            // うるため、`Child::kill()` を再度呼んではならない
            // （モジュール doc 参照）。
            (None, true, true)
        }
        WaitOutcome::TimedOut => {
            // 締め切り超過: 直接の子（supervisor）を `SIGKILL` してから
            // 必ず `wait()` で回収する（ゾンビを残さない）。子孫プロセス
            // （`_worker` を含む）の掃除は学習ワーカー側の lifeline に
            // 委ねる（モジュール doc 参照）。
            let child_reaped = reap_after_wall_timeout(&mut child, KILL_WAIT_TIMEOUT);
            (None, true, child_reaped)
        }
    };

    // 元のタイムアウト（stdout 読み取りが壁時計予算内に完了しなかった）・
    // `try_wait()` 側のタイムアウト・`LateExit` のいずれかがあれば、既存の
    // 語彙のとおり `WallTimeout` として分類する。
    let timed_out = stdout_timed_out || timed_out_by_wait;
    if timed_out {
        return Err(TrainProcessError::WallTimeout {
            limit_ms: u64::try_from(limits.wall_timeout().as_millis()).unwrap_or(u64::MAX),
            child_reaped,
        });
    }

    let finished = finish_run(FinishInput {
        status,
        started,
        deadline,
        stdout_wait,
        stderr_rx,
        child_reaped,
        limits,
        request,
    });
    conclude_run(finished, cancel_signalled, started, request, &anchor)
}

/// `SIGKILL` フォールバックの `Cancelled` に `out_dir` の読み取り専用の観測結果を
/// 付ける（削除はしない。REQ-34・#145）。
///
/// 「公開されている ⇔ 成功報告」の契約を守るため、公開済みの可能性がある
/// （[`OutDirResidue::NonEmpty`]）か公開状態を判定できない（[`OutDirResidue::Unknown`]）
/// 場合は `Cancelled` と断定せず [`TrainProcessError::CancelOutcomeUnconfirmed`] で返す
/// （成功応答を受け取る前に強制停止したため成功とも言えない。fail-closed。
/// codex/review 指摘 P1）。
#[cfg(unix)]
fn forced_kill_run(
    run: CancelledRun,
    request: &TrainRequest,
    anchor: &OutDirAnchor,
) -> Result<CancelledRun, TrainProcessError> {
    let residue = inspect_out_dir_residue(request, anchor);
    ensure_not_published(residue)?;
    Ok(run.with_out_dir_residue(residue))
}

/// 残置の観測結果が「未公開」（`Absent`・`EmptyReservation`）と確認できる場合だけ
/// `Ok`。公開済みの可能性・判定不能ならキャンセル完了と断定しない。
#[cfg(unix)]
fn ensure_not_published(residue: OutDirResidue) -> Result<(), TrainProcessError> {
    match residue {
        OutDirResidue::Absent | OutDirResidue::EmptyReservation => Ok(()),
        OutDirResidue::NonEmpty | OutDirResidue::Unknown => {
            Err(TrainProcessError::CancelOutcomeUnconfirmed { residue })
        }
    }
}

/// 子の終了・回収後の結果分類（[`finish_run`]）の入力。
#[cfg(unix)]
struct FinishInput<'a> {
    status: Option<ExitStatus>,
    started: Instant,
    deadline: Instant,
    stdout_wait: Option<Result<DrainedOutput, mpsc::RecvTimeoutError>>,
    stderr_rx: Option<mpsc::Receiver<DrainedOutput>>,
    child_reaped: bool,
    limits: &'a RunLimits,
    request: &'a TrainRequest,
}

/// 協調キャンセル後の最終判定（`Completed`／`Cancelled` を決める唯一の場所。
/// REQ-34・#145）。
///
/// パイプを閉じていない（`cancel_signalled == false`）なら従来どおり。閉じた後は、
/// supervisor の報告に従う: 成功 JSON は確定済みなので `Completed`、所定の
/// キャンセル応答（[`is_cancel_ack`]）は `Cancelled(Cooperative)`、別のエラー JSON
/// （`invalid_request` 等）は `Completed`（エラー結果のまま保持。キャンセル完了と
/// 断定しない）、クラッシュ・終了状態の不整合は `Cancelled(Unconfirmed)` で
/// 残置を観測して報告する（予約解放の証拠が無い）。残置が公開済み・判定不能なら
/// `Cancelled` とせず [`TrainProcessError::CancelOutcomeUnconfirmed`]。壁時計超過（`WallTimeout`）はキャンセルより優先し、
/// 従来どおりエラーで返す。
#[cfg(unix)]
fn conclude_run(
    finished: Result<TrainRun, TrainProcessError>,
    cancel_signalled: bool,
    started: Instant,
    request: &TrainRequest,
    anchor: &OutDirAnchor,
) -> Result<TrainRunEnd, TrainProcessError> {
    // 所定のキャンセル応答でない終了は `Unconfirmed` とし、残置を観測して報告する
    // （supervisor のクラッシュ等を協調キャンセル完了として扱わない）。
    let unconfirmed = || {
        let residue = inspect_out_dir_residue(request, anchor);
        ensure_not_published(residue)?;
        Ok(TrainRunEnd::Cancelled(CancelledRun::unconfirmed(
            started.elapsed(),
            residue,
        )))
    };
    match finished {
        Ok(run) if !cancel_signalled || matches!(run.outcome, TrainOutcome::Ok(_)) => {
            Ok(TrainRunEnd::Completed(run))
        }
        // 応答の文面だけでは supervisor 自身の応答と、worker が転送したエラー JSON を
        // 区別できない（JSON 契約は変えない）。そのため文面が一致しても、公開場所に
        // 何も残っていない（`Absent`）ことを観測できたときだけ `Cooperative` とする。
        // 残置があれば通常の `Unconfirmed` 経路（公開済み・判定不能ならエラー）へ回す
        // （fail-closed。REQ-34・#145）。
        Ok(run) if is_cancel_ack(&run) => {
            if inspect_out_dir_residue(request, anchor) == OutDirResidue::Absent {
                Ok(TrainRunEnd::Cancelled(CancelledRun::cooperative(
                    started.elapsed(),
                )))
            } else {
                unconfirmed()
            }
        }
        // supervisor が予約の解放を確認できなかった旨の報告は、通常の失敗として
        // 隠さず `Unconfirmed` 経路で残置を観測する（公開済み・判定不能なら
        // `CancelOutcomeUnconfirmed`。REQ-34・#145）。
        Ok(run) if is_cleanup_incomplete(&run) => unconfirmed(),
        // supervisor が返したキャンセル応答以外のエラー結果（`invalid_request` 等の
        // 入力エラーを含む）は隠さず、そのまま呼び出し元へ返す（キャンセル要求後に
        // 検証が先に失敗した場合の入力エラーを保持する。codex/review 指摘 P1）。
        Ok(run) => Ok(TrainRunEnd::Completed(run)),
        Err(e @ TrainProcessError::WallTimeout { .. }) => Err(e),
        Err(_) if cancel_signalled => unconfirmed(),
        Err(e) => Err(e),
    }
}

/// `supervisor.py::_report_cancelled` の所定のキャンセル応答（exit 70・
/// `runtime_error`・固定メッセージ）か。これだけが「supervisor が予約を解放して
/// 終了した」証拠になる（REQ-34・#145）。
#[cfg(unix)]
fn is_cancel_ack(run: &TrainRun) -> bool {
    run.exit_code == ExitCode::RuntimeError
        && matches!(
            &run.outcome,
            TrainOutcome::Error(f)
                if f.failure_code() == crate::result::FailureCode::RuntimeError
                    && f.message() == CANCEL_ACK_MESSAGE
        )
}

/// `supervisor.py::_report_cancelled` が出す固定メッセージ（両側で一致させる）。
#[cfg(unix)]
const CANCEL_ACK_MESSAGE: &str = "training cancelled by caller";

/// `supervisor.py::_report_cancelled` が予約の解放を確認できなかったときに出す
/// 固定メッセージ（両側で一致させる）。
#[cfg(unix)]
const CANCEL_CLEANUP_INCOMPLETE_MESSAGE: &str = "training cancelled but cleanup incomplete";

/// supervisor が「キャンセルしたが予約の解放を確認できなかった」と報告したか。
#[cfg(unix)]
fn is_cleanup_incomplete(run: &TrainRun) -> bool {
    run.exit_code == ExitCode::RuntimeError
        && matches!(
            &run.outcome,
            TrainOutcome::Error(f)
                if f.failure_code() == crate::result::FailureCode::RuntimeError
                    && f.message() == CANCEL_CLEANUP_INCOMPLETE_MESSAGE
        )
}

/// 子（supervisor）の終了・回収後に、標準出力・標準エラー出力・終了コードを
/// 検証して [`TrainRun`] を得る（[`run_train_cancellable`] の後半。協調キャンセルの
/// 最終判定 [`conclude_run`] から切り離すために関数化した。挙動は #144 から不変）。
#[cfg(unix)]
fn finish_run(input: FinishInput<'_>) -> Result<TrainRun, TrainProcessError> {
    let FinishInput {
        status,
        started,
        deadline,
        stdout_wait,
        stderr_rx,
        child_reaped,
        limits,
        request,
    } = input;

    let status = status.expect("status must be Some when timed_out is false");
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

    // supervisor は既に正常終了しているため、標準エラー出力は速やかに
    // EOF へ達するはずである。念のため短い上限（[`READER_DRAIN_TIMEOUT`]）を
    // 掛ける（issue #178 実装計画 3.5）。ただし壁時計の残り予算を無視して
    // 独自に最大 `READER_DRAIN_TIMEOUT`（5 秒）待ってしまうと、子孫が
    // stderr を開いたままの場合に締め切りを超過してから `StderrIncomplete`
    // （runtime_error=70）を返してしまい、本来返すべき上限超過
    // （`WallTimeout`・limit_exceeded=20）を誤分類する（issue #178 PR #233
    // レビュー再々指摘 P1）。待つ時間は「残りの壁時計予算」と
    // `READER_DRAIN_TIMEOUT` の短い方にし、締め切りを過ぎてもなお読み
    // 切れなかった場合は `WallTimeout` として分類する。締め切り前に
    // `READER_DRAIN_TIMEOUT` だけが尽きた場合は、従来どおり
    // `StderrIncomplete` とする。
    let stderr_wait_budget = deadline
        .saturating_duration_since(Instant::now())
        .min(READER_DRAIN_TIMEOUT);
    // 直接の子は既に終了・回収済みで、キャンセルで止めるものが無い。この時点
    // 以降にキャンセルが要求されても結果の分類を完了して `Completed` を返す
    // （codex/review 指摘 P1・Bugbot High。正常終了したジョブを `Cancelled` と
    // 記録しない。`Cancelling` → `Completed` は有効遷移。REQ-34・#144）。
    let stderr_wait_deadline = Instant::now() + stderr_wait_budget;
    let stderr_drain = match stderr_rx {
        Some(rx) => loop {
            let slice = stderr_wait_deadline
                .saturating_duration_since(Instant::now())
                .min(POLL_INTERVAL);
            match rx.recv_timeout(slice) {
                Ok(drained) => break Some(drained),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if Instant::now() >= deadline {
                        // 直接の子（supervisor）は既に reap 済み（`timed_out`
                        // が `false` でここへ到達した以上、`outcome` は
                        // `WaitOutcome::Exited` であり `child_reaped == true`）。
                        return Err(TrainProcessError::WallTimeout {
                            limit_ms: u64::try_from(limits.wall_timeout().as_millis())
                                .unwrap_or(u64::MAX),
                            child_reaped,
                        });
                    }
                    if Instant::now() >= stderr_wait_deadline {
                        break None;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break None,
            }
        },
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

/// [`run_train_cancellable`] を、キャンセルされない前提で呼ぶ従来の入口
/// （挙動は #178 から変えない）。
///
/// 立てられることのないトークンを渡すため `Cancelled` には到達しない。
/// それでもライブラリで panic しない方針（`.claude/rules/coding-rust.md`）
/// により、到達時は `Wait { kind: Interrupted }` を返して安全側に倒す。
#[cfg(unix)]
pub fn run_train(
    launcher: &WorkerLauncher,
    request: &TrainRequest,
    job_dir: &Path,
    limits: &RunLimits,
) -> Result<TrainRun, TrainProcessError> {
    match run_train_cancellable(launcher, request, job_dir, limits, &CancelToken::new())? {
        TrainRunEnd::Completed(run) => Ok(run),
        TrainRunEnd::Cancelled(_) => Err(TrainProcessError::Wait {
            kind: std::io::ErrorKind::Interrupted,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::MAX_TRAIN_WALL_SECONDS;
    use crate::request::{Device, TrainRequestParams};

    /// REQ-34・REQ-39: `try_wait()` が `Interrupted` を返し続けても、期限で
    /// 抜けて `Interrupted` のエラーを返す（無限ループしない）。
    #[cfg(unix)]
    #[test]
    fn try_wait_interrupt_bounded_gives_up_at_deadline() {
        let started = Instant::now();
        let mut calls = 0u64;
        let r = try_wait_interrupt_bounded(
            || {
                calls += 1;
                Err(std::io::Error::from(std::io::ErrorKind::Interrupted))
            },
            Instant::now() + Duration::from_millis(50),
        );
        assert_eq!(r.unwrap_err().kind(), std::io::ErrorKind::Interrupted);
        assert!(calls >= 1);
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// REQ-34: 期限前の `Interrupted` は再試行し、その後の結果を返す。
    #[cfg(unix)]
    #[test]
    fn try_wait_interrupt_bounded_retries_before_deadline() {
        let mut calls = 0;
        let r = try_wait_interrupt_bounded(
            || {
                calls += 1;
                if calls < 3 {
                    Err(std::io::Error::from(std::io::ErrorKind::Interrupted))
                } else {
                    Ok(None)
                }
            },
            Instant::now() + Duration::from_secs(30),
        );
        assert!(matches!(r, Ok(None)));
        assert_eq!(calls, 3);
    }

    /// REQ-34・REQ-39: 期限後に観測したキャンセルは採用せず、壁時計超過を
    /// 優先する。期限前なら採用する。
    #[cfg(unix)]
    #[test]
    fn cancel_applies_only_before_wall_deadline() {
        let token = CancelToken::new();
        let now = Instant::now();
        assert!(!cancel_applies(now, now + Duration::from_secs(1), &token));
        token.cancel();
        assert!(cancel_applies(now, now + Duration::from_secs(1), &token));
        assert!(!cancel_applies(now, now, &token));
        assert!(!cancel_applies(now + Duration::from_secs(1), now, &token));
    }

    /// REQ-39: キャンセル待機の期限は壁時計の残りで抑える。
    #[cfg(unix)]
    #[test]
    fn bounded_deadline_is_min_of_kill_wait_and_wall_remaining() {
        let now = Instant::now();
        let short = now + Duration::from_millis(100);
        assert_eq!(bounded_deadline(now, short), short);
        let long = now + KILL_WAIT_TIMEOUT + Duration::from_secs(60);
        assert_eq!(bounded_deadline(now, long), now + KILL_WAIT_TIMEOUT);
        assert_eq!(bounded_deadline(now, now), now);
    }

    /// REQ-34・REQ-39: 壁時計の残りが短いとき、SIGKILL を受けない子に対する
    /// キャンセルの回収待ちが `KILL_WAIT_TIMEOUT` ではなく残り時間内で終わる。
    #[cfg(unix)]
    #[test]
    fn cancel_child_wait_is_bounded_by_wall_remaining() {
        // SIGKILL は無視できないため、子が終了しない状況は作れない。代わりに
        // 期限切れの壁時計でも、生きている子の kill 後の回収が
        // 5 秒より十分早く（即時に）終わることと、結果が有限であることを確かめる。
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        let started = Instant::now();
        let r = cancel_plain(
            &mut child,
            started,
            started + Duration::from_millis(200),
            KILL_WAIT_TIMEOUT,
        );
        assert!(started.elapsed() < Duration::from_secs(4));
        match r {
            CancelStep::Cancelled(run) => assert_eq!(run.signal(), Some(SIGKILL)),
            // 壁時計の期限で打ち切った場合は実行時エラーではなく `StillRunning`
            // （呼び出し元が `WallTimeout` として回収する）。
            CancelStep::StillRunning | CancelStep::WallDeadline { .. } => {}
            _ => panic!("unexpected cancel step"),
        }
        let _ = child.kill();
        let _ = child.wait();
    }

    #[cfg(unix)]
    #[derive(Default)]
    struct FakeChild {
        kill_results: Vec<std::io::Result<()>>,
        wait_results: Vec<std::io::Result<Option<ExitStatus>>>,
        kills: usize,
        always_err: bool,
        /// この時刻以後の `try_wait()` は `SIGKILL` 終了を返す。
        reap_at: Option<Instant>,
        /// キャンセル用パイプを持つか（`true` のとき最初の
        /// `close_cancel_channel()` が `true` を返す）。
        has_cancel_channel: bool,
        /// `close_cancel_channel()` が成功した回数。
        channel_closes: usize,
        /// この時刻以後の `try_wait()` は、パイプを閉じていれば正常終了
        /// （`exit(0)`）を返す（協調キャンセルで自ら終了する子を模す）。
        exit_after_close_at: Option<Instant>,
        /// `kill()` が 1 回以上呼ばれた後の `try_wait()` は `SIGKILL` 終了を返す
        /// （時刻に依存せず、協調しない子を決定的に模す）。
        reap_on_kill: bool,
    }

    #[cfg(unix)]
    impl ChildControl for FakeChild {
        fn close_cancel_channel(&mut self) -> bool {
            if self.has_cancel_channel {
                self.has_cancel_channel = false;
                self.channel_closes += 1;
                true
            } else {
                false
            }
        }
        fn kill(&mut self) -> std::io::Result<()> {
            self.kills += 1;
            if self.kill_results.is_empty() {
                Ok(())
            } else {
                self.kill_results.remove(0)
            }
        }
        fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
            if self.always_err {
                Err(io_err())
            } else if !self.wait_results.is_empty() {
                self.wait_results.remove(0)
            } else if self.channel_closes > 0
                && self
                    .exit_after_close_at
                    .is_some_and(|t| Instant::now() >= t)
            {
                use std::os::unix::process::ExitStatusExt;
                Ok(Some(ExitStatus::from_raw(0)))
            } else if (self.reap_on_kill && self.kills > 0)
                || self.reap_at.is_some_and(|t| Instant::now() >= t)
            {
                use std::os::unix::process::ExitStatusExt;
                Ok(Some(ExitStatus::from_raw(SIGKILL)))
            } else {
                Ok(None)
            }
        }
    }

    /// 協調キャンセルを使わない従来の呼び出し（パイプ無し）。
    #[cfg(unix)]
    fn cancel_plain<C: ChildControl>(
        child: &mut C,
        started: Instant,
        wall_deadline: Instant,
        final_grace: Duration,
    ) -> CancelStep {
        cancel_child(
            child,
            started,
            wall_deadline,
            final_grace,
            Duration::ZERO,
            &mut false,
        )
    }

    #[cfg(unix)]
    fn io_err() -> std::io::Error {
        std::io::Error::from(std::io::ErrorKind::PermissionDenied)
    }

    /// REQ-39: kill が失敗し try_wait がエラーでも、再試行して回収できたら
    /// 成功する（Child を手放さない）。
    #[cfg(unix)]
    #[test]
    fn ensure_reaped_retries_after_kill_and_wait_errors() {
        use std::os::unix::process::ExitStatusExt;
        let mut c = FakeChild {
            kill_results: vec![Err(io_err()), Err(io_err())],
            wait_results: vec![Err(io_err()), Ok(None), Ok(Some(ExitStatus::from_raw(9)))],
            kills: 0,
            always_err: false,
            reap_at: None,
            ..FakeChild::default()
        };
        let r = ensure_reaped(
            &mut c,
            Instant::now() + Duration::from_secs(5),
            Duration::from_millis(50),
            &mut false,
        );
        assert_eq!(r.and_then(|r| r.status.signal()), Some(9));
        assert!(c.kills >= 3);
    }

    /// REQ-39: 回収を確認できない場合は `None`（呼び出し側が `WallTimeout` にする）。
    #[cfg(unix)]
    #[test]
    fn ensure_reaped_returns_none_when_never_reaped() {
        let mut c = FakeChild {
            kill_results: vec![],
            wait_results: vec![],
            kills: 0,
            always_err: false,
            reap_at: None,
            ..FakeChild::default()
        };
        let started = Instant::now();
        let r = ensure_reaped(
            &mut c,
            Instant::now() + Duration::from_millis(100),
            Duration::from_millis(100),
            &mut false,
        );
        assert!(r.is_none());
        assert!(c.kills >= 2);
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    /// REQ-21・REQ-39: 壁時計超過後の回収は kill 失敗を再試行して回収でき、
    /// 回収できなければ `false`（呼び出し側は `WallTimeout` を維持する）。
    #[cfg(unix)]
    #[test]
    fn reap_after_wall_timeout_reports_reaped_flag() {
        use std::os::unix::process::ExitStatusExt;
        let mut ok = FakeChild {
            kill_results: vec![Err(io_err())],
            wait_results: vec![Err(io_err()), Ok(Some(ExitStatus::from_raw(9)))],
            kills: 0,
            always_err: false,
            reap_at: None,
            ..FakeChild::default()
        };
        assert!(reap_after_wall_timeout(&mut ok, Duration::from_millis(500)));
        assert!(ok.kills >= 2);
        let mut never = FakeChild {
            kill_results: vec![],
            wait_results: vec![],
            kills: 0,
            always_err: false,
            reap_at: None,
            ..FakeChild::default()
        };
        assert!(!reap_after_wall_timeout(
            &mut never,
            Duration::from_millis(100)
        ));
    }

    /// REQ-21・REQ-34・REQ-39: 期限より前に回収が完了したキャンセルだけが
    /// `Cancelled`（唯一の規則 `classify_cancel_outcome`）。
    #[cfg(unix)]
    #[test]
    fn cancel_reaped_before_deadline_is_cancelled() {
        let started = Instant::now();
        let mut c = FakeChild {
            wait_results: vec![Ok(None)],
            reap_at: Some(started),
            ..FakeChild::default()
        };
        let step = cancel_plain(
            &mut c,
            started,
            started + Duration::from_secs(30),
            Duration::from_millis(100),
        );
        assert!(matches!(step, CancelStep::Cancelled(_)));
    }

    /// REQ-21・REQ-39: キャンセル要求が先でも、回収を期限後に観測したら
    /// `WallDeadline { child_reaped: true }`（`WallTimeout`）で `Cancelled` にしない。
    #[cfg(unix)]
    #[test]
    fn cancel_reaped_after_deadline_is_wall_timeout_reaped() {
        let started = Instant::now();
        let mut c = FakeChild {
            wait_results: vec![Ok(None)],
            reap_at: Some(started + Duration::from_millis(250)),
            ..FakeChild::default()
        };
        let step = cancel_plain(
            &mut c,
            started,
            started + Duration::from_millis(100),
            Duration::from_secs(5),
        );
        assert!(matches!(
            step,
            CancelStep::WallDeadline { child_reaped: true }
        ));
    }

    /// REQ-21・REQ-39: キャンセル中の `try_wait()` エラーが続き期限までに
    /// 回収を確定できなければ `WallDeadline { child_reaped: false }`。
    #[cfg(unix)]
    #[test]
    fn cancel_persistent_wait_errors_are_wall_timeout_not_reaped() {
        let started = Instant::now();
        let mut c = FakeChild {
            always_err: true,
            ..FakeChild::default()
        };
        let step = cancel_plain(
            &mut c,
            started,
            started + Duration::from_millis(100),
            Duration::from_millis(100),
        );
        assert!(matches!(
            step,
            CancelStep::WallDeadline {
                child_reaped: false
            }
        ));
        assert!(c.kills >= 2);
    }

    /// REQ-21・REQ-39: 待機エラー後に期限後に回収できた場合は
    /// `WallDeadline { child_reaped: true }`（`runtime_error` にしない）。
    #[cfg(unix)]
    #[test]
    fn cancel_wait_error_then_late_reap_is_wall_timeout_reaped() {
        let started = Instant::now();
        let mut c = FakeChild {
            wait_results: vec![Err(io_err())],
            reap_at: Some(started + Duration::from_millis(250)),
            ..FakeChild::default()
        };
        let step = cancel_plain(
            &mut c,
            started,
            started + Duration::from_millis(100),
            Duration::from_secs(5),
        );
        assert!(matches!(
            step,
            CancelStep::WallDeadline { child_reaped: true }
        ));
    }

    /// REQ-34: 自分の kill が成功して `SIGKILL` 終了なら `Cancelled`。
    #[cfg(unix)]
    #[test]
    fn cancel_with_delivered_kill_is_cancelled() {
        let started = Instant::now();
        let mut c = FakeChild {
            wait_results: vec![Ok(None)],
            reap_at: Some(started),
            ..FakeChild::default()
        };
        let step = cancel_plain(
            &mut c,
            started,
            started + Duration::from_secs(30),
            Duration::from_millis(100),
        );
        assert!(matches!(step, CancelStep::Cancelled(_)));
    }

    /// REQ-34: kill が失敗したのに `SIGKILL` で終わった子（外部由来）は
    /// `Cancelled` にせず `AlreadyExited`（通常の結果分類へ戻す）。
    #[cfg(unix)]
    #[test]
    fn cancel_with_failed_kill_and_external_sigkill_is_not_cancelled() {
        let started = Instant::now();
        let mut c = FakeChild {
            kill_results: vec![Err(io_err())],
            wait_results: vec![Ok(None)],
            reap_at: Some(started),
            ..FakeChild::default()
        };
        let step = cancel_plain(
            &mut c,
            started,
            started + Duration::from_secs(30),
            Duration::from_millis(100),
        );
        assert!(matches!(step, CancelStep::AlreadyExited(_)));
    }

    /// REQ-34: 1 回目の kill が成功し、回収が一時エラーになり、再試行の kill が
    /// 「既に終了」で失敗して `SIGKILL` 終了として回収された場合も、送信済みの
    /// kill を記録しているので `Cancelled`。
    #[cfg(unix)]
    #[test]
    fn cancel_keeps_earlier_kill_record_across_retries() {
        let started = Instant::now();
        let mut c = FakeChild {
            kill_results: vec![Ok(()), Err(io_err()), Err(io_err())],
            wait_results: vec![Ok(None), Err(io_err())],
            reap_at: Some(started),
            ..FakeChild::default()
        };
        let step = cancel_plain(
            &mut c,
            started,
            started + Duration::from_secs(30),
            Duration::from_millis(100),
        );
        assert!(matches!(step, CancelStep::Cancelled(_)));
    }

    /// REQ-34・#145: `classify_cancel_outcome` の全セル
    /// （回収なし・期限前後 × kill 成功 × パイプを閉じたか × `SIGKILL` 終了か）。
    #[cfg(unix)]
    #[test]
    fn classify_cancel_outcome_table() {
        use std::os::unix::process::ExitStatusExt;
        #[derive(Debug, PartialEq)]
        enum K {
            WallReaped,
            WallNotReaped,
            Forced,
            Coop,
            Already,
        }
        fn kind(step: CancelStep) -> K {
            match step {
                CancelStep::WallDeadline { child_reaped: true } => K::WallReaped,
                CancelStep::WallDeadline {
                    child_reaped: false,
                } => K::WallNotReaped,
                CancelStep::Cancelled(run) => {
                    assert_eq!(run.stop(), CancelStop::ForcedKill);
                    assert_eq!(run.signal(), Some(SIGKILL));
                    K::Forced
                }
                CancelStep::ExitedAfterSignal(_) => K::Coop,
                CancelStep::AlreadyExited(_) => K::Already,
                CancelStep::StillRunning => panic!("StillRunning is not a classification"),
            }
        }
        let started = Instant::now();
        let wall = started + Duration::from_secs(30);
        let before = started;
        let after = wall + Duration::from_secs(1);
        assert_eq!(
            kind(classify_cancel_outcome(None, wall, started)),
            K::WallNotReaped
        );
        for (raw, kill_delivered, signalled, at, expected) in [
            // 期限以後の回収は常に壁時計超過。
            (9, true, true, after, K::WallReaped),
            (0, false, true, after, K::WallReaped),
            // 自分の kill が成功して SIGKILL 終了: ForcedKill（パイプの有無によらず）。
            (9, true, true, before, K::Forced),
            (9, true, false, before, K::Forced),
            // パイプを閉じた後に kill を送らず終了: 協調。
            (0, false, true, before, K::Coop),
            // kill 成功後に SIGKILL 以外で終了（kill と自己終了の競合）: 協調。
            (0, true, true, before, K::Coop),
            // パイプを閉じる前の終了: 従来どおり AlreadyExited。
            (0, false, false, before, K::Already),
            // 外部由来の SIGKILL（kill 未送出）: パイプを閉じていなければ AlreadyExited。
            (9, false, false, before, K::Already),
            // kill 成功でも SIGKILL 終了でなくパイプ未送出: AlreadyExited。
            (0, true, false, before, K::Already),
            // 外部由来の SIGKILL でもパイプを閉じ済みなら協調側（Cancelled にしない）。
            (9, false, true, before, K::Coop),
        ] {
            let reaped = Reaped {
                status: ExitStatus::from_raw(raw),
                at,
                kill_delivered,
                cancel_signalled: signalled,
            };
            assert_eq!(
                kind(classify_cancel_outcome(Some(reaped), wall, started)),
                expected,
                "raw={raw} kill_delivered={kill_delivered} signalled={signalled}"
            );
        }
    }

    /// REQ-34・#145: パイプを閉じた後、猶予内に子が自ら終了したら kill を送らず
    /// `ExitedAfterSignal`（協調キャンセル）。
    #[cfg(unix)]
    #[test]
    fn cancel_child_cooperative_exit_sends_no_kill() {
        let started = Instant::now();
        let mut c = FakeChild {
            has_cancel_channel: true,
            exit_after_close_at: Some(started),
            ..FakeChild::default()
        };
        let mut signalled = false;
        let step = cancel_child(
            &mut c,
            started,
            started + Duration::from_secs(60),
            Duration::from_millis(100),
            Duration::from_secs(10),
            &mut signalled,
        );
        assert!(matches!(step, CancelStep::ExitedAfterSignal(_)));
        assert!(signalled);
        assert_eq!(c.channel_closes, 1);
        assert_eq!(c.kills, 0);
    }

    /// REQ-34・#145: 猶予内に終了しない（協調しない）子は `SIGKILL` へ
    /// フォールバックし、`ForcedKill` になる（壁時計の内側に収まる場合）。
    #[cfg(unix)]
    #[test]
    fn cancel_child_falls_back_to_sigkill_after_grace() {
        let started = Instant::now();
        let mut c = FakeChild {
            has_cancel_channel: true,
            reap_on_kill: true,
            ..FakeChild::default()
        };
        // kill 送出後にのみ SIGKILL 終了を返す。時刻依存にすると、macOS 等で
        // 猶予内の poll が遅延したとき協調終了と誤判定され不安定になる。
        let mut signalled = false;
        let step = cancel_child(
            &mut c,
            started,
            started + Duration::from_secs(60),
            Duration::from_millis(100),
            Duration::from_millis(100),
            &mut signalled,
        );
        match step {
            CancelStep::Cancelled(run) => assert_eq!(run.stop(), CancelStop::ForcedKill),
            _ => panic!("expected ForcedKill"),
        }
        assert!(signalled);
        assert!(c.kills >= 1);
    }

    /// REQ-34・REQ-39・#145: 協調の待ちは壁時計の期限から `KILL_WAIT_TIMEOUT` を
    /// 遡った時刻で切り詰める（フォールバックの回収待ちを壁時計の内側に残す）。
    /// 壁時計の残りが `KILL_WAIT_TIMEOUT` 未満なら協調の待ちは実質ゼロ。
    #[cfg(unix)]
    #[test]
    fn cancel_child_cooperative_wait_is_clamped_by_wall_deadline() {
        let started = Instant::now();
        let mut c = FakeChild {
            has_cancel_channel: true,
            ..FakeChild::default()
        };
        let mut signalled = false;
        // 壁時計の残り 200ms・猶予 30 秒: 猶予をそのまま待てば壁時計を超える。
        let step = cancel_child(
            &mut c,
            started,
            started + Duration::from_millis(200),
            Duration::from_millis(100),
            Duration::from_secs(30),
            &mut signalled,
        );
        assert!(started.elapsed() < Duration::from_secs(4));
        assert!(signalled);
        assert!(matches!(
            step,
            CancelStep::StillRunning | CancelStep::WallDeadline { .. }
        ));
    }

    /// REQ-34・#145: 一度パイプを閉じた後の再入では協調を繰り返さず kill へ進む。
    #[cfg(unix)]
    #[test]
    fn cancel_child_reentry_skips_cooperative_wait() {
        let started = Instant::now();
        let mut c = FakeChild {
            has_cancel_channel: true,
            wait_results: vec![Ok(None)],
            reap_at: Some(started),
            ..FakeChild::default()
        };
        let mut signalled = true; // 前回の呼び出しで閉じ済み
        let step = cancel_child(
            &mut c,
            started,
            started + Duration::from_secs(60),
            Duration::from_millis(100),
            Duration::from_secs(30),
            &mut signalled,
        );
        assert!(matches!(step, CancelStep::Cancelled(_)));
        assert_eq!(c.channel_closes, 0);
    }

    /// REQ-34・#145: `RunLimits::with_cooperative_grace` は締める方向だけを許す。
    #[test]
    fn req34_with_cooperative_grace_rejects_loosening() {
        let request = test_request(Some(30));
        let limits = RunLimits::for_request(&request);
        assert_eq!(
            limits.cooperative_grace(),
            Duration::from_secs(u64::from(COOPERATIVE_CANCEL_GRACE_SECONDS))
        );
        let tight = limits
            .with_cooperative_grace(Duration::from_secs(1))
            .expect("tighter grace");
        assert_eq!(tight.cooperative_grace(), Duration::from_secs(1));
        assert!(limits.with_cooperative_grace(Duration::ZERO).is_err());
        assert!(
            limits
                .with_cooperative_grace(limits.cooperative_grace())
                .is_err()
        );
        assert!(
            limits
                .with_cooperative_grace(Duration::from_secs(16))
                .is_err()
        );
    }

    /// REQ-34・#145: `out_dir` の読み取り専用の観測（削除しない）。
    #[cfg(unix)]
    #[test]
    fn inspect_out_dir_residue_reports_state_without_deleting() {
        let root = std::env::temp_dir().join(format!("fandhe-residue-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create root");
        let request = TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: root.to_string_lossy().to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: Some(30),
            rss_limit_bytes: None,
        })
        .expect("valid request");
        assert_eq!(
            inspect_out_dir_residue(&request, &OutDirAnchor::capture(&request)),
            OutDirResidue::Absent
        );
        std::fs::create_dir(root.join("out")).expect("create out");
        assert_eq!(
            inspect_out_dir_residue(&request, &OutDirAnchor::capture(&request)),
            OutDirResidue::EmptyReservation
        );
        std::fs::write(root.join("out").join("model.onnx"), b"x").expect("write");
        assert_eq!(
            inspect_out_dir_residue(&request, &OutDirAnchor::capture(&request)),
            OutDirResidue::NonEmpty
        );
        // 観測は何も削除しない。
        assert!(root.join("out").join("model.onnx").exists());
        std::fs::remove_dir_all(root.join("out")).expect("rm out");
        std::os::unix::fs::symlink(&root, root.join("out")).expect("symlink");
        assert_eq!(
            inspect_out_dir_residue(&request, &OutDirAnchor::capture(&request)),
            OutDirResidue::Unknown
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// REQ-34・#145: 協調キャンセル後の最終判定（成功 JSON は `Completed`、
    /// それ以外は `Cancelled(Cooperative)`、壁時計超過はエラーのまま）。
    #[cfg(unix)]
    #[test]
    fn conclude_run_maps_outcomes_after_cancel_signal() {
        let started = Instant::now();
        let wall = || TrainProcessError::WallTimeout {
            limit_ms: 1,
            child_reaped: true,
        };
        let request = test_request(None);
        // 送出前は従来どおり（エラーはエラー）。
        assert!(matches!(
            conclude_run_t(
                Err(TrainProcessError::StdoutIncomplete),
                false,
                started,
                &request
            ),
            Err(TrainProcessError::StdoutIncomplete)
        ));
        // 送出後の別の失敗（クラッシュ等）は `Unconfirmed`（残置を観測して報告する）。
        match conclude_run_t(
            Err(TrainProcessError::StdoutIncomplete),
            true,
            started,
            &request,
        ) {
            Ok(TrainRunEnd::Cancelled(run)) => {
                assert_eq!(run.stop(), CancelStop::Unconfirmed);
                assert_eq!(run.signal(), None);
                assert_eq!(run.out_dir_residue(), Some(OutDirResidue::Absent));
                assert!(run.child_spawned() && run.child_reaped());
            }
            _ => panic!("expected Cancelled(Unconfirmed)"),
        }
        // 壁時計超過は送出後でもキャンセルより優先する。
        assert!(matches!(
            conclude_run_t(Err(wall()), true, started, &request),
            Err(TrainProcessError::WallTimeout { .. })
        ));
    }

    /// REQ-34・#145: 所定のキャンセル応答（exit 70・`runtime_error`・固定メッセージ）
    /// だけが `Cooperative`。別のエラー JSON（exit 70 でもメッセージ違い）は
    /// `Unconfirmed` で、残置を観測して報告する。
    #[cfg(unix)]
    #[test]
    fn conclude_run_requires_cancel_ack_for_cooperative() {
        let started = Instant::now();
        let request = test_request(None);
        let run_of = |message: &str| {
            let json =
                format!(r#"{{"status":"error","code":"runtime_error","message":"{message}"}}"#);
            let outcome =
                classify_exit(ExitCode::RuntimeError, json.as_bytes(), &request).expect("classify");
            TrainRun {
                outcome,
                exit_code: ExitCode::RuntimeError,
                elapsed: Duration::ZERO,
                worker_stderr: Vec::new(),
                stderr_truncated: false,
            }
        };
        match conclude_run_t(Ok(run_of(CANCEL_ACK_MESSAGE)), true, started, &request) {
            Ok(TrainRunEnd::Cancelled(run)) => {
                assert_eq!(run.stop(), CancelStop::Cooperative);
                assert_eq!(run.out_dir_residue(), None);
            }
            _ => panic!("expected Cancelled(Cooperative)"),
        }
        // 別のエラー JSON は隠さず `Completed`（エラー結果）のまま保持する。
        match conclude_run_t(Ok(run_of("worker crashed")), true, started, &request) {
            Ok(TrainRunEnd::Completed(run)) => {
                assert!(matches!(run.outcome, TrainOutcome::Error(_)));
                assert_eq!(run.exit_code, ExitCode::RuntimeError);
            }
            _ => panic!("expected Completed(Error)"),
        }
    }

    /// REQ-34・#145 回帰: 文面がキャンセル応答と一致する転送された worker エラーでも、
    /// 予約が残っていれば `Cooperative` としない（空の予約が残る場合は `Unconfirmed`、
    /// 中身がある場合は `CancelOutcomeUnconfirmed`）。残置が無ければ従来どおり
    /// `Cooperative`（本物の応答）。
    #[cfg(unix)]
    #[test]
    fn conclude_run_does_not_trust_ack_lookalike_when_reservation_remains() {
        let root = std::env::temp_dir().join(format!("fandhe-ack-fwd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create root");
        let request = TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: root.to_string_lossy().to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: Some(30),
            rss_limit_bytes: None,
        })
        .expect("valid request");
        let json = format!(
            r#"{{"status":"error","code":"runtime_error","message":"{CANCEL_ACK_MESSAGE}"}}"#
        );
        let make_run = || {
            let outcome =
                classify_exit(ExitCode::RuntimeError, json.as_bytes(), &request).expect("classify");
            TrainRun {
                outcome,
                exit_code: ExitCode::RuntimeError,
                elapsed: Duration::ZERO,
                worker_stderr: Vec::new(),
                stderr_truncated: false,
            }
        };
        // 残置なし: 本物の応答として Cooperative。
        assert!(matches!(
            conclude_run_t(Ok(make_run()), true, Instant::now(), &request),
            Ok(TrainRunEnd::Cancelled(run)) if run.stop() == CancelStop::Cooperative
        ));
        // 空の予約が残る: Cooperative ではなく Unconfirmed（残置を報告）。
        std::fs::create_dir(root.join("out")).expect("create out");
        match conclude_run_t(Ok(make_run()), true, Instant::now(), &request) {
            Ok(TrainRunEnd::Cancelled(run)) => {
                assert_eq!(run.stop(), CancelStop::Unconfirmed);
                assert_eq!(run.out_dir_residue(), Some(OutDirResidue::EmptyReservation));
            }
            _ => panic!("expected Cancelled(Unconfirmed)"),
        }
        // 中身が残る: 公開済みの可能性があるため Cancelled としない。
        std::fs::write(root.join("out").join("model.onnx"), b"x").expect("write");
        assert!(matches!(
            conclude_run_t(Ok(make_run()), true, Instant::now(), &request),
            Err(TrainProcessError::CancelOutcomeUnconfirmed {
                residue: OutDirResidue::NonEmpty
            })
        ));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// REQ-34・#145 回帰: 実行中に root が改名され、同じ名前に空ディレクトリが
    /// 置き直された場合、名前の再解決では `Absent` に見えても実体が起動前の記録と
    /// 違うため `Unknown` 扱いになり、`Cancelled` にならない（元の実体には成果物が
    /// 公開済みでありうる）。
    #[cfg(unix)]
    #[test]
    fn req34_swapped_root_is_not_observed_as_unpublished() {
        let root = unique_root();
        let request = TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: root.to_string_lossy().to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: Some(30),
            rss_limit_bytes: None,
        })
        .expect("valid request");
        let anchor = OutDirAnchor::capture(&request);
        assert_eq!(
            inspect_out_dir_residue(&request, &anchor),
            OutDirResidue::Absent
        );
        // 実行中に root を改名し、同じ名前に空ディレクトリを置き直す。
        let moved = root.with_extension("moved");
        std::fs::rename(&root, &moved).expect("rename root");
        std::fs::create_dir(&root).expect("recreate root");
        std::fs::create_dir(moved.join("out")).expect("published in original");
        std::fs::write(moved.join("out").join("model.onnx"), b"x").expect("write");
        assert_eq!(
            inspect_out_dir_residue(&request, &anchor),
            OutDirResidue::Unknown
        );
        // SIGKILL 経路・協調キャンセル経路のどちらも Cancelled としない。
        let run = CancelledRun::before_start();
        assert!(matches!(
            forced_kill_run(run, &request, &anchor),
            Err(TrainProcessError::CancelOutcomeUnconfirmed {
                residue: OutDirResidue::Unknown
            })
        ));
        let json = format!(
            r#"{{"status":"error","code":"runtime_error","message":"{CANCEL_ACK_MESSAGE}"}}"#
        );
        let outcome =
            classify_exit(ExitCode::RuntimeError, json.as_bytes(), &request).expect("classify");
        let ack = TrainRun {
            outcome,
            exit_code: ExitCode::RuntimeError,
            elapsed: Duration::ZERO,
            worker_stderr: Vec::new(),
            stderr_truncated: false,
        };
        assert!(matches!(
            conclude_run(Ok(ack), true, Instant::now(), &request, &anchor),
            Err(TrainProcessError::CancelOutcomeUnconfirmed {
                residue: OutDirResidue::Unknown
            })
        ));
        // 起動前に root を記録できなかった場合も fail-closed。
        let missing = TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: root.join("nonexistent").to_string_lossy().to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: Some(30),
            rss_limit_bytes: None,
        })
        .expect("valid request");
        assert_eq!(
            inspect_out_dir_residue(&missing, &OutDirAnchor::capture(&missing)),
            OutDirResidue::Unknown
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&moved);
    }

    /// REQ-34・#145 回帰: 末尾スラッシュ・`.` を含む `out_dir` でも、検証と同じ
    /// 規則で正規化した親を求めるため、実体の記録が成立し（`Unknown` に固定されず）
    /// `Absent`／`EmptyReservation` を観測でき、協調キャンセルは `Cooperative`
    /// になる。
    #[cfg(unix)]
    #[test]
    fn req34_out_dir_spellings_are_normalized_for_anchor() {
        for (out_dir, parent) in [
            ("out", ""),
            ("out/", ""),
            ("./out", ""),
            ("a/./b", "a"),
            ("a//b/", "a"),
        ] {
            let root = unique_root();
            std::fs::create_dir_all(root.join(parent)).expect("create parent");
            let request = TrainRequest::new(TrainRequestParams {
                kind: "c3".to_string(),
                kind_version: 1,
                config: serde_json::Map::new(),
                label_order: vec!["a".to_string(), "b".to_string()],
                max_bytes: 512,
                seed: 0,
                device: Device::Cpu,
                root: root.to_string_lossy().to_string(),
                train_path: "train.jsonl".to_string(),
                out_dir: out_dir.to_string(),
                time_limit_seconds: Some(30),
                rss_limit_bytes: None,
            })
            .expect("valid request");
            let anchor = OutDirAnchor::capture(&request);
            assert_eq!(
                inspect_out_dir_residue(&request, &anchor),
                OutDirResidue::Absent,
                "{out_dir}"
            );
            std::fs::create_dir(out_dir_path(&request)).expect("reserve");
            assert_eq!(
                inspect_out_dir_residue(&request, &anchor),
                OutDirResidue::EmptyReservation,
                "{out_dir}"
            );
            let json = format!(
                r#"{{"status":"error","code":"runtime_error","message":"{CANCEL_ACK_MESSAGE}"}}"#
            );
            std::fs::remove_dir(out_dir_path(&request)).expect("release");
            let outcome =
                classify_exit(ExitCode::RuntimeError, json.as_bytes(), &request).expect("classify");
            let ack = TrainRun {
                outcome,
                exit_code: ExitCode::RuntimeError,
                elapsed: Duration::ZERO,
                worker_stderr: Vec::new(),
                stderr_truncated: false,
            };
            assert!(
                matches!(
                    conclude_run(Ok(ack), true, Instant::now(), &request, &anchor),
                    Ok(TrainRunEnd::Cancelled(run)) if run.stop() == CancelStop::Cooperative
                ),
                "{out_dir}"
            );
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    /// REQ-34・#145: supervisor が予約の解放を確認できなかった旨（固定メッセージ）を
    /// 報告した場合は、通常の失敗（`Completed`）にせず `Unconfirmed` 経路に入る。
    #[cfg(unix)]
    #[test]
    fn conclude_run_treats_cleanup_incomplete_as_unconfirmed() {
        let request = test_request(None);
        let json = format!(
            r#"{{"status":"error","code":"runtime_error","message":"{CANCEL_CLEANUP_INCOMPLETE_MESSAGE}"}}"#
        );
        let outcome =
            classify_exit(ExitCode::RuntimeError, json.as_bytes(), &request).expect("classify");
        let run = TrainRun {
            outcome,
            exit_code: ExitCode::RuntimeError,
            elapsed: Duration::ZERO,
            worker_stderr: Vec::new(),
            stderr_truncated: false,
        };
        assert!(is_cleanup_incomplete(&run));
        match conclude_run_t(Ok(run), true, Instant::now(), &request) {
            Ok(TrainRunEnd::Cancelled(run)) => assert_eq!(run.stop(), CancelStop::Unconfirmed),
            Err(TrainProcessError::CancelOutcomeUnconfirmed { .. }) => {}
            _ => panic!("expected Unconfirmed handling, not Completed"),
        }
    }

    /// REQ-34・#145: 入力エラー（`invalid_request`）がキャンセル送出後でも
    /// 呼び出し元へ伝わる。
    #[cfg(unix)]
    #[test]
    fn conclude_run_preserves_invalid_request_after_cancel_signal() {
        let request = test_request(None);
        let json = r#"{"status":"error","code":"invalid_request","message":"bad field"}"#;
        let outcome =
            classify_exit(ExitCode::InvalidInput, json.as_bytes(), &request).expect("classify");
        let run = TrainRun {
            outcome,
            exit_code: ExitCode::InvalidInput,
            elapsed: Duration::ZERO,
            worker_stderr: Vec::new(),
            stderr_truncated: false,
        };
        match conclude_run_t(Ok(run), true, Instant::now(), &request) {
            Ok(TrainRunEnd::Completed(run)) => {
                assert_eq!(run.exit_code, ExitCode::InvalidInput);
                assert!(matches!(run.outcome, TrainOutcome::Error(_)));
            }
            _ => panic!("expected Completed(Error invalid_request)"),
        }
    }

    /// REQ-34・#145: 公開済みの可能性・判定不能な残置は `Cancelled` と断定しない。
    #[cfg(unix)]
    #[test]
    fn ensure_not_published_rejects_published_or_unknown_residue() {
        assert!(ensure_not_published(OutDirResidue::Absent).is_ok());
        assert!(ensure_not_published(OutDirResidue::EmptyReservation).is_ok());
        for residue in [OutDirResidue::NonEmpty, OutDirResidue::Unknown] {
            assert!(matches!(
                ensure_not_published(residue),
                Err(TrainProcessError::CancelOutcomeUnconfirmed { residue: r }) if r == residue
            ));
        }
    }

    /// 呼び出し時点の実体を記録して [`conclude_run`] を呼ぶ（実体の差し替えを
    /// 伴わないテスト用）。
    #[cfg(unix)]
    fn conclude_run_t(
        finished: Result<TrainRun, TrainProcessError>,
        cancel_signalled: bool,
        started: Instant,
        request: &TrainRequest,
    ) -> Result<TrainRunEnd, TrainProcessError> {
        conclude_run(
            finished,
            cancel_signalled,
            started,
            request,
            &OutDirAnchor::capture(request),
        )
    }

    /// テスト用リクエストの root 文字列。unix では実在する一意なディレクトリ
    /// （残置観測が実体を記録できるようにする）。root は `/` 始まりの構文検査が
    /// あるため、windows ではこれを使えず固定の架空 root にする（残置観測は
    /// unix 限定）。
    fn test_root() -> String {
        #[cfg(unix)]
        {
            unique_root().to_string_lossy().to_string()
        }
        #[cfg(not(unix))]
        {
            "/fandhe-edge-fixture-root".to_string()
        }
    }

    #[cfg(unix)]
    fn unique_root() -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "fandhe-train-root-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).expect("create root");
        root
    }

    fn test_request(time_limit_seconds: Option<u32>) -> TrainRequest {
        TrainRequest::new(TrainRequestParams {
            kind: "c3".to_string(),
            kind_version: 1,
            config: serde_json::Map::new(),
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 0,
            device: Device::Cpu,
            root: test_root(),
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

        // 直列化後ちょうど `MAX_REQUEST_BYTES` の `config`（`{"k":"` 6 バイト＋
        // 本文＋`"}` 2 バイト）。`new` は受理し、リクエスト全体では超過する。
        let mut huge_config = serde_json::Map::new();
        huge_config.insert(
            "k".to_string(),
            serde_json::Value::String("x".repeat(MAX_REQUEST_BYTES - 8)),
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
        .expect("config exactly at the limit is accepted by new()");

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

    /// issue #178 PR #233 レビュー P1「try_wait() の Ok(Some(status)) を
    /// 締め切り判定より先に受理するため、期限超過後に終了したプロセスを
    /// 成功扱いしうる」: 観測時刻が締め切りより前なら受理してよいが、
    /// 締め切りちょうど・締め切り後は受理してはならない（`WaitOutcome::
    /// LateExit` として扱う）。
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
}
