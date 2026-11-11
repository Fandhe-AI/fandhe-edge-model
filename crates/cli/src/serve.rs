//! 常駐プロセス `fandhe-edge serve`（#708。親 #707 の (A)。REQ-21・REQ-28・REQ-33・REQ-38・REQ-39）。
//!
//! # 役割と呼び出し文脈
//!
//! `main.rs` が `serve --package <dir>` を受けると [`run`] が前景で動き続ける。起動時に `infer` と同じ
//! `prepare`（経路の閉じ込め・形式・sha256 の全量検証・モデルの読み込み）を 1 回だけ通し、得た推論
//! パイプラインを保持して、`infer --text` の委ねる要求（[`crate::serve_client`]）を Unix ソケットで受ける。
//! 判定はメモリ上の検証済みの内容で行い、読み込み後にファイルが書き換わった・消えた場合は、委ねる側が配置の
//! 指紋（[`crate::serve_layout`]）の不一致で検出して単独へ戻る。
//!
//! # 受け付けと判定
//!
//! - 要求・応答は 1 行 JSON（`fandhe_edge_core::serve_protocol`）。応答の `stdout` は単独の `infer` と
//!   同じバイト列（同じ組み立てを `Vec<u8>` へ書く）で、終了コードも単独と同じ
//! - 接続ごとにスレッドを立て（同時 [`MAX_CONNECTIONS`] 本まで）、読み書きに期限を付ける。要求 1 行は
//!   `MAX_REQUEST_LINE_BYTES` まで。判定は上限 [`QUEUE_DEPTH`] の `sync_channel` を介して 1 本のスレッドが
//!   順に行う。満杯なら即座に単独と同じ形の `limit_exceeded`（exit 20）を返す
//! - 期限超過はプロセスを終えずに `limit_exceeded` として返す（`StallPolicy::Resident`）。協調的に
//!   打ち切れない停止の見張りと常駐のメモリ上限は #709
//! - 資源の上限: 起動から自分の RSS と起動（`prepare`）の時間を、単独の監督と同じ種類別の上限で見張る
//!   （[`spawn_guard_thread`]）。ハード期限で計算スレッドを切り離したら、その要求に `limit_exceeded` を返して終える
//! - ログ（stderr）に入力本文・判定結果は出さない（受け付け時刻・id・終了コードのみ）。書くのは専用スレッドで、
//!   有界の待ち行列に `try_send` する（満杯なら捨てる）ため、stderr の詰まりで判定・応答が止まらない
//!
//! # 置き場所と権限（REQ-39）
//!
//! 既定は `$HOME/Library/Application Support/fandhe-edge/`、`FANDHE_EDGE_SERVE_DIR` で変えられる。
//! ディレクトリは 0700、ソケット `serve.sock` は作成後 0600、二重起動防止は `serve.lock` の
//! `File::try_lock`。`unsafe` を使わないため umask の設定はせず、0700 のディレクトリの中に作る。
//! SIGTERM は既定の動き（即終了）で、残ったソケットは次の起動が `serve.lock` を取れたら消して作り直す。
//! 待ち受けは Unix ソケットのみで、ネットワークには公開しない（REQ-38）。

use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::serve_protocol::{
    MAX_REQUEST_LINE_BYTES, PackageIdentity, ServeResponse, decode_request, encode_response,
};
use fandhe_edge_guard::resource::{
    SafetyLimits, current_process_rss_bytes, safety_limits_for_kind,
};

use crate::args::{InferArgs, InferSource, ServeArgs};
use crate::error_report::{default_message, emit_error_report};
use crate::infer_batch::{SINGLE_HARD_LIMIT_GRACE, SingleCall, infer_single_to_bytes};
use crate::infer_guard::peek_checked_kind;
use crate::output::write_error_report;
use crate::project::{invalid, runtime};
use crate::stages::infer::{DEFAULT_TEXT_ID, Prepared, prepare};

/// 置き場所を変える環境変数（試験用。未設定なら `$HOME/Library/Application Support/fandhe-edge/`）。
pub const SERVE_DIR_ENV: &str = "FANDHE_EDGE_SERVE_DIR";
/// 設定されていれば `infer` が常駐へ委ねない環境変数。
pub const NO_SERVE_ENV: &str = "FANDHE_EDGE_NO_SERVE";
/// ソケットのファイル名。
pub const SOCKET_FILE: &str = "serve.sock";
const LOCK_FILE: &str = "serve.lock";

/// 判定待ちの上限（判定中の 1 件を除く）。超えたら `limit_exceeded`。
const QUEUE_DEPTH: usize = 8;
/// 同時に扱う接続（スレッド）の上限。超えた接続は読まずに `limit_exceeded` を返して閉じる。
const MAX_CONNECTIONS: usize = 64;
/// 要求 1 行を読み切るまでの合計の期限（遅送りで接続を占有されないため）。
const REQUEST_READ_TIMEOUT: Duration = Duration::from_secs(10);
/// 判定スレッドの応答を待つ上限（待ち 8 件 + 判定中 1 件が最長のハード期限〔追加学習 60.5 秒〕で並んだ場合を覆う）。
const REPLY_WAIT: Duration = Duration::from_secs(10 * 60);
/// 起動失敗の JSON を見張りのスレッドが書く場合に、main が終了を待つ上限。
const STARTUP_ERROR_WAIT: Duration = Duration::from_secs(5);
/// 資源の見張りの間隔（監督の RSS ポーリングと同程度）。
const GUARD_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// 応答の書き込みの期限。
const RESPONSE_WRITE_TIMEOUT: Duration = Duration::from_secs(10);

/// 判定を試験で遅らせる環境変数（ms）。デバッグビルド限定でリリースビルドには入らない（待ちの上限の試験用）。
#[cfg(debug_assertions)]
const TEST_DELAY_ENV: &str = "FANDHE_EDGE_SERVE_TEST_DELAY_MS";

/// 置き場所のディレクトリ。既定は macOS の慣習をそのまま使う（Linux は検証対象外）。環境変数があればそれ（絶対パスのみ）、無ければ `$HOME` 配下の既定。
#[must_use]
pub fn serve_dir() -> Option<PathBuf> {
    let dir = match std::env::var_os(SERVE_DIR_ENV) {
        Some(d) => PathBuf::from(d),
        None => {
            Path::new(&std::env::var_os("HOME")?).join("Library/Application Support/fandhe-edge")
        }
    };
    dir.is_absolute().then_some(dir)
}

/// 置き場所として許せるディレクトリか（symlink でない実ディレクトリで、権限がちょうど 0700）。
pub(crate) fn is_private_dir(meta: &fs::Metadata) -> bool {
    meta.is_dir() && meta.mode() & 0o777 == 0o700
}

/// 改行までの 1 行を `deadline` までに、`limit` バイト（本文）を超えない範囲で読む。
/// 改行より後ろは捨てる。改行なしで EOF なら読めた分を返す（空なら `UnexpectedEof`）。
pub(crate) fn read_line_bounded(
    stream: &mut UnixStream,
    limit: usize,
    deadline: Instant,
) -> io::Result<Vec<u8>> {
    let mut line = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::ErrorKind::TimedOut.into());
        }
        stream.set_read_timeout(Some(remaining))?;
        let n = stream.read(&mut chunk)?;
        let Some(read) = chunk.get(..n) else {
            return Err(io::ErrorKind::InvalidData.into());
        };
        if n == 0 {
            return if line.is_empty() {
                Err(io::ErrorKind::UnexpectedEof.into())
            } else {
                Ok(line)
            };
        }
        let end = read.iter().position(|b| *b == b'\n');
        line.extend_from_slice(read.get(..end.map_or(n, |i| i + 1)).unwrap_or(read));
        if line.len() > limit.saturating_add(1) {
            return Err(io::ErrorKind::InvalidData.into());
        }
        if end.is_some() {
            return Ok(line);
        }
    }
}

/// 置き場所を作る（無ければ 0700）。あれば symlink でない 0700 のディレクトリであることを確かめる。
fn ensure_dir(dir: &Path) -> Result<fs::Metadata, ErrorReport> {
    let create = match fs::symlink_metadata(dir) {
        Ok(_) => false,
        Err(e) if e.kind() == io::ErrorKind::NotFound => true,
        Err(_) => return Err(runtime("cannot inspect serve directory")),
    };
    if create {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
            .and_then(|()| fs::set_permissions(dir, fs::Permissions::from_mode(0o700)))
            .map_err(|_| runtime("cannot create serve directory"))?;
    }
    let meta = fs::symlink_metadata(dir).map_err(|_| runtime("cannot inspect serve directory"))?;
    if is_private_dir(&meta) {
        Ok(meta)
    } else {
        Err(invalid(
            "serve directory must be a real directory with mode 0700",
        ))
    }
}

/// `serve.lock` を作って排他ロックを取る（戻り値を持っている間が常駐の生存期間）。
/// ディレクトリと同じ所有者（自分が作ったファイルの所有者）であることも確かめる。
fn acquire_lock(dir: &Path, dir_meta: &fs::Metadata) -> Result<fs::File, ErrorReport> {
    let lock = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .mode(0o600)
        .open(dir.join(LOCK_FILE))
        .map_err(|_| runtime("cannot open serve lock"))?;
    let owner = lock
        .metadata()
        .map_err(|_| runtime("cannot inspect serve lock"))?
        .uid();
    if owner != dir_meta.uid() {
        return Err(invalid("serve directory is not owned by the current user"));
    }
    lock.try_lock()
        .map_err(|_| invalid("serve already running"))?;
    Ok(lock)
}

/// 残ったソケットを消す（ロックを持っているので、持ち主は存在しない）。ソケット以外は消さずに失敗する。
fn remove_stale_socket(sock: &Path) -> Result<(), ErrorReport> {
    match fs::symlink_metadata(sock) {
        Ok(m) if m.file_type().is_socket() => {
            fs::remove_file(sock).map_err(|_| runtime("cannot remove stale serve socket"))
        }
        Ok(_) => Err(invalid("serve socket path is occupied")),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err(runtime("cannot inspect serve socket")),
    }
}

/// ログ行の待ち行列（有界）。stderr が詰まっても判定・応答・終了処理が止まらないよう、書くのは専用スレッドだけで、
/// 渡すときは `try_send` で満杯なら捨てる（ログの欠落は許し、常駐の動作を止めない）。
static LOG_TX: OnceLock<SyncSender<String>> = OnceLock::new();
/// 渡したがまだ書かれていない行の数（終了前の流し切りの待ちに使う）。
static LOG_PENDING: AtomicUsize = AtomicUsize::new(0);
/// ログ待ち行列の長さ。
const LOG_QUEUE: usize = 64;
/// 終了直前に、ログの流し切りを待つ上限。
const LOG_DRAIN_WAIT: Duration = Duration::from_millis(200);

/// ログ専用スレッドを起動する（`run` の最初）。起動できなければログは捨てる。
fn init_logger() {
    let (tx, rx) = mpsc::sync_channel::<String>(LOG_QUEUE);
    let spawned = thread::Builder::new()
        .name("serve-log".to_string())
        .spawn(move || {
            for line in rx {
                let _ = writeln!(io::stderr(), "serve: {line}");
                LOG_PENDING.fetch_sub(1, Ordering::SeqCst);
            }
        });
    if spawned.is_ok() {
        let _ = LOG_TX.set(tx);
    }
}

/// stderr へ 1 行（本文・結果は渡さない）。待ち行列が満杯なら捨てる。
fn log_line(msg: std::fmt::Arguments<'_>) {
    if let Some(tx) = LOG_TX.get() {
        LOG_PENDING.fetch_add(1, Ordering::SeqCst);
        if tx.try_send(msg.to_string()).is_err() {
            LOG_PENDING.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// ログを（上限つきで）流し切ってから終了する。stderr が詰まっていても `LOG_DRAIN_WAIT` を超えて待たない。
fn exit_with(code: ExitCode) -> ! {
    let deadline = Instant::now() + LOG_DRAIN_WAIT;
    while LOG_PENDING.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    std::process::exit(i32::from(code.code()))
}

/// 起動中に stdout へエラー JSON を書く権利（見張りのスレッドと main が両方 JSON を出さないよう、
/// 先に取った側だけが書く）。
static STDOUT_CLAIMED: AtomicBool = AtomicBool::new(false);

/// stdout への書き込みで止まっても、この時間でプロセスを終える保険（REQ-39）。
const EXIT_WRITE_DEADLINE: Duration = Duration::from_secs(3);

/// 終了直前に stdout へ同期で書く前に張る、上限つきの終了の保険。`EXIT_WRITE_DEADLINE` 後に同じ終了コードで
/// プロセスを終える（出力先が詰まっていても有界時間内に終わる）。張れなければ `false`（書かずに終える）。
fn arm_exit_deadline(code: ExitCode) -> bool {
    thread::Builder::new()
        .name("serve-exit".to_string())
        .spawn(move || {
            thread::sleep(EXIT_WRITE_DEADLINE);
            std::process::exit(i32::from(code.code()));
        })
        .is_ok()
}

fn claim_stdout() -> bool {
    !STDOUT_CLAIMED.swap(true, Ordering::SeqCst)
}

/// 起動失敗のエラーを stdout へ 1 回だけ書く。先に見張りのスレッドが書く（終了する）側なら、その終了を待つ。
fn emit_startup_error<W: Write>(out: &mut W, report: &ErrorReport) -> io::Result<ExitCode> {
    if claim_stdout() {
        // main が返ればプロセスは終わるが、書き込みで止まっても終われるよう保険を張る。張れなければ書かない。
        if !arm_exit_deadline(report.code) {
            return Ok(report.code);
        }
        emit_error_report(out, report)
    } else {
        // 見張りのスレッドが終了させるまで待つ。stdout で詰まっても残り続けないよう、期限つき。
        let deadline = Instant::now() + STARTUP_ERROR_WAIT;
        while Instant::now() < deadline {
            thread::park_timeout(Duration::from_millis(100));
        }
        exit_with(ExitCode::RuntimeError)
    }
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// 判定スレッドへ渡す 1 件。
struct Job {
    id: String,
    text: String,
    accepted_unix: u64,
    /// 受け付けた時刻（待っている間に委ねる側の期限が過ぎた要求を判定しないため）。
    accepted_at: Instant,
    reply: SyncSender<ServeResponse>,
}

/// `ErrorReport` を単独と同じ 1 行 JSON の応答にする。
fn error_response(report: &ErrorReport) -> ServeResponse {
    let mut buf = Vec::new();
    match write_error_report(&mut buf, report) {
        Ok(code) => ServeResponse::Ok {
            exit_code: code.code(),
            stdout: String::from_utf8(buf).unwrap_or_default(),
        },
        Err(_) => ServeResponse::Ok {
            exit_code: ExitCode::RuntimeError.code(),
            stdout: String::new(),
        },
    }
}

fn limit_response() -> ServeResponse {
    error_response(&ErrorReport::new(
        ExitCode::LimitExceeded,
        default_message(ExitCode::LimitExceeded),
    ))
}

/// 常駐を起動して要求を受け付け続ける。起動に失敗したときだけ、単独の `infer` と同じ形のエラーを
/// `out` へ書いて戻る（ソケットは作らない）。成功時は戻らない。
///
/// # Errors
/// `out` への書き込み失敗。
pub fn run<W: Write>(out: &mut W, args: &ServeArgs, cwd: &Path) -> io::Result<ExitCode> {
    init_logger();
    // 起動時（`prepare` の前）から、単独の監督と同じ種類別の時間・RSS の上限で自分を見張る。
    let limits = peek_checked_kind(cwd, &args.package)
        .map_or(SafetyLimits::STRICT, |k| safety_limits_for_kind(k.as_str()));
    let loaded = Arc::new(AtomicBool::new(false));
    if spawn_guard_thread(limits, Arc::clone(&loaded)).is_err() {
        return emit_startup_error(out, &runtime("cannot start serve guard thread"));
    }
    let (listener, prepared, _lock) = match start(args, cwd) {
        Ok(started) => started,
        Err(report) => return emit_startup_error(out, &report),
    };
    loaded.store(true, Ordering::SeqCst);
    let identity = Arc::new(prepared.identity.clone());
    let (tx, rx) = mpsc::sync_channel::<Job>(QUEUE_DEPTH);
    if thread::Builder::new()
        .name("serve-judge".to_string())
        .spawn(move || judge_loop(&rx, prepared))
        .is_err()
    {
        return emit_startup_error(out, &runtime("cannot start serve judge thread"));
    }
    log_line(format_args!("ready"));
    let active = Arc::new(AtomicUsize::new(0));
    loop {
        let Ok((stream, _)) = listener.accept() else {
            thread::sleep(Duration::from_millis(50));
            continue;
        };
        if active.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
            active.fetch_sub(1, Ordering::SeqCst);
            reply(stream, &limit_response());
            continue;
        }
        let guard = ConnGuard(Arc::clone(&active));
        let (identity, tx) = (Arc::clone(&identity), tx.clone());
        // 起動に失敗したらクロージャごと guard が落ち、接続も閉じる。
        let _ = thread::Builder::new()
            .name("serve-conn".to_string())
            .spawn(move || {
                let _guard = guard;
                handle(stream, &identity, &tx);
            });
    }
}

/// 同時接続数の計数を Drop で戻す。
struct ConnGuard(Arc<AtomicUsize>);

impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// 起動順（置き場所 → ロック → 全量検証と読み込み → 残ったソケットの除去 → bind → 0600）。
/// 全量検証に失敗したらソケットを作らない。
fn start(args: &ServeArgs, cwd: &Path) -> Result<(UnixListener, Prepared, fs::File), ErrorReport> {
    let dir = serve_dir().ok_or_else(|| invalid("serve directory is not configured"))?;
    let dir_meta = ensure_dir(&dir)?;
    let lock = acquire_lock(&dir, &dir_meta)?;
    let infer_args = InferArgs {
        package: args.package.clone(),
        source: InferSource::Text {
            text: String::new(),
            id: None,
        },
        version: None,
    };
    let prepared = prepare(cwd, &infer_args)?;
    let sock = dir.join(SOCKET_FILE);
    remove_stale_socket(&sock)?;
    let listener = UnixListener::bind(&sock).map_err(|_| runtime("cannot bind serve socket"))?;
    if fs::set_permissions(&sock, fs::Permissions::from_mode(0o600)).is_err() {
        let _ = fs::remove_file(&sock);
        return Err(runtime("cannot restrict serve socket"));
    }
    Ok((listener, prepared, lock))
}

/// 応答を書いて閉じる（失敗は無視。相手が既にいない）。
fn reply(mut stream: UnixStream, response: &ServeResponse) {
    let _ = stream.set_write_timeout(Some(RESPONSE_WRITE_TIMEOUT));
    let _ = stream.write_all(&encode_response(response));
    let _ = stream.flush();
}

/// 1 接続: 要求を読み、パッケージを照合し、判定スレッドへ渡して応答を返す。
fn handle(mut stream: UnixStream, identity: &PackageIdentity, tx: &SyncSender<Job>) {
    let accepted_unix = unix_secs();
    let response = respond(&mut stream, identity, tx, accepted_unix);
    reply(stream, &response);
}

fn respond(
    stream: &mut UnixStream,
    identity: &PackageIdentity,
    tx: &SyncSender<Job>,
    accepted_unix: u64,
) -> ServeResponse {
    let deadline = Instant::now() + REQUEST_READ_TIMEOUT;
    let request = read_line_bounded(stream, MAX_REQUEST_LINE_BYTES, deadline)
        .ok()
        .and_then(|line| decode_request(&line).ok());
    let Some(request) = request else {
        return error_response(&invalid("invalid serve request"));
    };
    if request.package != *identity {
        return ServeResponse::PackageMismatch;
    }
    let (reply_tx, reply_rx) = mpsc::sync_channel(1);
    let job = Job {
        id: request.id,
        text: request.text,
        accepted_unix,
        accepted_at: Instant::now(),
        reply: reply_tx,
    };
    match tx.try_send(job) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => return limit_response(),
        Err(TrySendError::Disconnected(_)) => {
            // 判定スレッドが死んだ常駐は続けても全件失敗するため、落として再起動に任せる（fail-closed）。
            log_line(format_args!("judge thread stopped; exiting"));
            exit_with(ExitCode::RuntimeError);
        }
    }
    // 判定スレッドは必ず（ハード期限内に）返すが、無期限には待たない。
    match reply_rx.recv_timeout(REPLY_WAIT) {
        Ok(response) => response,
        Err(mpsc::RecvTimeoutError::Timeout) => limit_response(),
        Err(mpsc::RecvTimeoutError::Disconnected) => error_response(&runtime("serve judge failed")),
    }
}

/// 判定スレッド: 保持した `Prepared` で 1 件ずつ順に判定する（`Pipeline` は呼び出し間で状態を持たない）。
fn judge_loop(rx: &mpsc::Receiver<Job>, prepared: Prepared) {
    let Prepared {
        definition,
        pipeline,
        scope,
        single_time,
        ..
    } = prepared;
    let io_schema = definition.io().clone();
    let pipeline = Arc::new(pipeline);
    for job in rx {
        #[cfg(debug_assertions)]
        if let Some(ms) = std::env::var(TEST_DELAY_ENV)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            thread::sleep(Duration::from_millis(ms));
        }
        // 委ねる側の読みの期限（`time + SINGLE_HARD_LIMIT_GRACE`）と同じ式。過ぎた要求は相手がもう
        // 諦めているので判定せず次へ。
        let budget = single_time.saturating_add(SINGLE_HARD_LIMIT_GRACE);
        if job.accepted_at.elapsed() > budget {
            let _ = job.reply.send(limit_response());
            continue;
        }
        let call = SingleCall {
            io: &io_schema,
            options: definition.options(),
            scope: &scope,
            id: DEFAULT_TEXT_ID,
            text: &job.text,
            duration: single_time,
        };
        let mut detached = false;
        let response = match infer_single_to_bytes(call, Arc::clone(&pipeline)) {
            Ok((buf, code, was_detached)) => match String::from_utf8(buf) {
                Ok(stdout) => {
                    detached = was_detached;
                    ServeResponse::Ok {
                        exit_code: code.code(),
                        stdout,
                    }
                }
                Err(_) => error_response(&runtime("serve judge failed")),
            },
            Err(_) => error_response(&runtime("serve judge failed")),
        };
        if let ServeResponse::Ok { exit_code, .. } = &response {
            // 本文・判定結果は出さない。id は英数字と `-_` だけに絞る。
            let id: String = job
                .id
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
                .collect();
            log_line(format_args!(
                "accepted_at={} id={id} exit_code={exit_code}",
                job.accepted_unix
            ));
        }
        let _ = job.reply.send(response);
        if detached {
            // ハード期限で計算スレッドを切り離した。積み上げないよう、この要求に `limit_exceeded` を返した後に
            // 常駐を終える（待っていた要求は EOF で単独へ戻る）。応答の書き出しを待つために少しだけ待つ。
            log_line(format_args!("judge stalled past the hard limit; exiting"));
            thread::sleep(Duration::from_millis(500));
            exit_with(ExitCode::LimitExceeded);
        }
    }
}

/// RSS を読めずに終わる。起動中なら `runtime_error` の JSON を stdout に 1 回だけ出してから exit 70。
fn runtime_exit(msg: &'static str, startup: bool) -> ! {
    log_line(format_args!("{msg}"));
    if startup && claim_stdout() && arm_exit_deadline(ExitCode::RuntimeError) {
        let report = ErrorReport::new(
            ExitCode::RuntimeError,
            default_message(ExitCode::RuntimeError),
        );
        let _ = write_error_report(&mut io::stdout().lock(), &report);
    }
    exit_with(ExitCode::RuntimeError)
}

/// 上限超過で終わる。起動中（ソケットを作る前）なら `limit_exceeded` の JSON を stdout に 1 回だけ出す
/// （先に取った側だけが書く。`STDOUT_CLAIMED`）。
///
/// 限界: 見張りが `claim_stdout` に負けた（main が起動失敗の JSON を書いている）場合、終了コードは
/// 20 になり、main 側のコードとは一致しないことがある。競合は起動失敗と上限超過が同じ 50 ms 内に重なる
/// ときだけで窓は狭く、どちらの場合も stdout の JSON は 1 つだけである。
fn limit_exit(msg: &'static str, startup: bool) -> ! {
    log_line(format_args!("{msg}"));
    if startup && claim_stdout() && arm_exit_deadline(ExitCode::LimitExceeded) {
        let report = ErrorReport::new(
            ExitCode::LimitExceeded,
            default_message(ExitCode::LimitExceeded),
        );
        let _ = write_error_report(&mut io::stdout().lock(), &report);
    }
    exit_with(ExitCode::LimitExceeded)
}

/// 常駐自身の資源の見張り（約 50 ms ごと。REQ-39・#708）。上限は単独の監督と同じ種類別の値。
/// - 起動（`prepare`）が `limits.time()` を過ぎても終わらなければ、ソケットを作る前に `limit_exceeded`
///   （exit 20）の JSON を出して終わる
/// - RSS が `limits.memory()` を超えたら stderr に 1 行出して exit 20（判定中の要求のクライアントは EOF で単独へ戻る）。
///   RSS を読めなければ fail-closed で exit 70
fn spawn_guard_thread(limits: SafetyLimits, loaded: Arc<AtomicBool>) -> io::Result<()> {
    let started = Instant::now();
    thread::Builder::new()
        .name("serve-guard".to_string())
        .spawn(move || {
            loop {
                thread::sleep(GUARD_POLL_INTERVAL);
                let startup = !loaded.load(Ordering::SeqCst);
                if startup && started.elapsed() > limits.time().get() {
                    limit_exit("startup time limit exceeded; exiting", true);
                }
                match current_process_rss_bytes() {
                    Ok(Some(rss)) if rss > limits.memory().get() => {
                        limit_exit("memory limit exceeded; exiting", startup);
                    }
                    Ok(_) => {}
                    Err(_) => runtime_exit("memory probe failed; exiting", startup),
                }
            }
        })
        .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39: 改行までの 1 行を返し、上限を超える行は拒否する。
    #[test]
    fn read_line_bounded_respects_limit() {
        let deadline = Instant::now() + Duration::from_secs(5);
        let (mut a, mut b) = UnixStream::pair().expect("pair");
        b.write_all(b"abc\nrest").expect("write");
        assert_eq!(
            read_line_bounded(&mut a, 16, deadline).expect("line"),
            b"abc\n"
        );
        b.write_all(&[b'x'; 64]).expect("write");
        let err = read_line_bounded(&mut a, 16, deadline).expect_err("too long");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    /// REQ-39: 期限内に改行が来なければ時間切れにする。
    #[test]
    fn read_line_bounded_times_out() {
        let (mut a, _b) = UnixStream::pair().expect("pair");
        let deadline = Instant::now() + Duration::from_millis(100);
        let err = read_line_bounded(&mut a, 16, deadline).expect_err("timeout");
        assert!(matches!(
            err.kind(),
            io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
        ));
    }
}
