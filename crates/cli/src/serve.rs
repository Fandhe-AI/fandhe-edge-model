//! 常駐プロセス `fandhe-edge serve`（#708・#709。親 #707 の (A)。REQ-21・REQ-28・REQ-33・REQ-38・REQ-39）。
//!
//! # 役割と呼び出し文脈
//!
//! `main.rs` が `serve --package <dir>` を受けると [`run`] が前景で動き続ける。プロセスは 2 つに分かれる（#709）。
//!
//! - **親**: 置き場所・ロック・Unix ソケットの受け付け・待ち行列・見張りを持つ。モデルは持たない
//! - **worker 子**: 親が自分自身を [`SERVE_WORKER_ENV`]`=1` つきで起動する（stdin・stdout はパイプ、stderr は継承）。
//!   `infer` と同じ `prepare`（経路の閉じ込め・形式・sha256 の全量検証・モデルの読み込み）を 1 回だけ通し、
//!   読み込んだ内容の同一性を `ready` の 1 行で親へ名乗ってから、要求 1 行ごとに判定して応答 1 行を返す
//!
//! `infer --text` の委ねる要求（[`crate::serve_client`]）は親がソケットで受け、版とパッケージを照合してから
//! worker へ渡す。判定は worker のメモリ上の検証済みの内容で行い、読み込み後にファイルが書き換わった・消えた
//! 場合は、委ねる側が配置の指紋（[`crate::serve_layout`]）の不一致で検出して単独へ戻る。
//!
//! # 受け付けと判定
//!
//! - 要求・応答は 1 行 JSON（`fandhe_edge_core::serve_protocol`）。応答の `stdout` は単独の `infer` と
//!   同じバイト列（同じ組み立てを `Vec<u8>` へ書く）で、終了コードも単独と同じ
//! - 要求の `protocol`・`cli_version` が自分と違えば、判定せず `version_mismatch` を返す（委ねる側は単独へ戻る。
//!   古い常駐が新しい CLI の要求を古い規則で判定しないため。#709）
//! - 接続ごとにスレッドを立て（同時 [`MAX_CONNECTIONS`] 本まで。超えたら読まずに `limit_exceeded` を返して
//!   閉じる）、読み書きに期限を付ける。要求 1 行は `MAX_REQUEST_LINE_BYTES` まで。判定は上限 [`QUEUE_DEPTH`] の
//!   `sync_channel` を介して 1 本の判定スレッドが順に worker へ渡す。満杯なら即座に単独と同じ形の
//!   `limit_exceeded`（exit 20）を返す
//! - 待っている間に 1 件の期限（`limits.time() + SINGLE_HARD_LIMIT_GRACE`。委ねる側の期限と同じ式）を過ぎた
//!   要求は、判定せず `limit_exceeded` を返す
//! - ログ（stderr）に入力本文・判定結果は出さない（worker の受け付け時刻・id・終了コードと、親の起動・停止・
//!   作り直しの固定の文だけ）。書くのは専用スレッドで、有界の待ち行列に `try_send` する（満杯なら捨てる）ため、
//!   stderr の詰まりで判定・応答が止まらない
//!
//! # 見張りと作り直し（REQ-39・#709）
//!
//! 上限は単独の監督と同じ種類別の値（`safety_limits_for_kind`）。1 件の時間の上限の値は変えていない
//! （kind の値のまま。常駐向けに短くするかは実機の計測 #710 の後に決める）。
//!
//! - **1 件の時間**: worker の中の協調的な期限とハード期限（`StallPolicy::Resident`）は `limit_exceeded` の応答に
//!   なり、worker は続く。ハード期限で計算スレッドを切り離した worker は、応答の後に自分で終わる。応答が
//!   「1 件の期限 + 1 秒」までに来なければ、親が worker を止めて `limit_exceeded` を返す（協調的に打ち切れない
//!   停止の回収）
//! - **worker の RSS**（約 50 ms ごと）: 上限を超えたら worker を止める。`ready` の直後に 1 回計測して上限内と
//!   確かめた後の超過は止めるだけで、判定中の要求には `limit_exceeded` を返す。確かめる前（起動中）の超過は、
//!   作り直しても同じなので親ごと exit 20（[`worker_rss_action`]）
//! - **worker の回収**: 止めた worker の回収は有界（kill の後 [`WORKER_REAP_TIMEOUT`] まで）。回収できなければ
//!   親ごと exit 70（止められない worker を抱えたまま判定スレッドを止めない）
//! - **親自身の RSS**: 上限を超えたら exit 20。RSS を読めなければ（親・worker とも）fail-closed で exit 70
//! - **作り直し**: worker が止まったら（親が止めた・自分で終わった・外から kill された）、判定スレッドが次の
//!   要求を待つ前に起動し直す。常駐は続き、次の要求は読み込み時間を払わない。ハード期限で計算スレッドを
//!   切り離した worker は最後の応答を `ok_retiring` で返し、親は次の要求を渡さずに作り直す。worker が要求を
//!   読む前に止まっていた場合（書き込みの失敗・渡す前の見張りによる停止）は、作り直して 1 回だけ渡し直す
//!   （応答待ちの間の切断は渡し直さない）
//! - **親ごと終える場合**（理由をログに出す。次の要求は単独へ戻る）: 作り直しの起動の失敗（検証の失敗・
//!   時間切れ）、worker の版が親と違う（稼働中のバイナリの差し替え。exit 70）、作り直した worker の同一性が
//!   起動時と違う（置き場所の中身の変化。exit 70）、1 件も判定しないまま worker が終わる作り直しが
//!   [`MAX_BARREN_RESTARTS`] 回続いた（exit 70）
//! - **起動**: worker の読み込みの失敗は、単独の `infer` と同じ stdout・終了コードで終わる。`ready` が
//!   `limits.time()` 以内に来なければ `limit_exceeded`（exit 20）。どちらもソケットは作らない
//! - **起動の全体**: 親の見張りは、置き場所・ロックに触る前から待ち受けの開始までの時間も見る。`limits.time()` を
//!   過ぎても待ち受けに至らなければ `limit_exceeded`（exit 20）の JSON を 1 つ出して終わる（ロックファイルの
//!   open が止まる場合なども覆う）
//! - **worker 自身の見張り**（約 50 ms ごと。親の見張りと二重）: 親がいなくなったらすぐ終わる（exit 70）。
//!   読み込みが `limits.time()` を過ぎた・自分の RSS が `limits.memory()` を超えたら exit 20（RSS を読めなければ
//!   exit 70）。親は worker の終了コード 20 を回収時に見て、判定中の要求を `limit_exceeded` にする
//! - 親が終わる経路では worker を止めてから終わる。親が SIGTERM・SIGKILL された場合は、worker が自分の見張りで
//!   親の不在に気づいて終わる（待機中なら stdin の EOF でも終わる）。終了処理に入った後は、親の見張りは
//!   何もしない（終了コードを上書きしない）
//!
//! # 置き場所と権限（REQ-39）
//!
//! 既定は `$HOME/Library/Application Support/fandhe-edge/`、`FANDHE_EDGE_SERVE_DIR` で変えられる。
//! ディレクトリは 0700、ソケット `serve.v1.sock` は作成後 0600、二重起動防止は `serve.lock` の
//! `File::try_lock`。`unsafe` を使わないため umask の設定はせず、0700 のディレクトリの中に作る。
//! SIGTERM は既定の動き（即終了）で、残ったソケットは次の起動が `serve.lock` を取れたら消して作り直す。
//! 待ち受けは Unix ソケットのみで、ネットワークには公開しない（REQ-38）。
//!
//! ソケットのファイル名は 1 行 JSON の形の版（`PROTOCOL_VERSION`）ごとに分ける（[`SOCKET_FILE`]。プロトコルを
//! 上げるときはファイル名も上げる）。版の違う常駐と CLI は互いのソケットを見ないので、古い常駐が新しい要求を
//! 「不正な要求」（exit 64）と答えたものを CLI が判定結果として採用することがない（CLI からは常駐が無い扱いで、
//! 何も言わず単独で判定する）。ロックのファイル名は版で分けないので、常駐は版をまたいで同時に 1 つだけで、
//! 起動時にロックを取れたら、残っている古い名前のソケット（[`LEGACY_SOCKET_FILES`]）も消す。
//!
//! # 限界（証拠種別: テストハーネス）
//!
//! - 接続してきた相手の uid の照合（`getpeereid`）はしない（std で書けない）。0700 のディレクトリと 0600 の
//!   ソケットで、他の uid は接続できない
//! - RSS は約 50 ms 間隔のポーリングで、間隔の間は上限を超えうる。見るのは親と worker だけ
//! - worker が親の不在に気づくのは約 50 ms ごとの確認で、その間は動き続ける。worker の子孫プロセスは見ない
//! - `accept` から close-on-exec の設定までの間に作り直しが走ると、その接続の fd を worker が継承しうる
//!   （worker は使わない。接続が worker の終了まで閉じ切らないことがある）
//! - worker の応答と要求の id は照合しない（要求は 1 件ずつ渡し、頼んでいない行が来た worker は作り直す）
//! - [`SERVE_WORKER_ENV`] を外から付けると、見張りの無い worker として動く（ソケットは作らない）。安全上限は
//!   暴走止めで、呼び出し元を信頼しない境界ではない

use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::serve_protocol::{
    MAX_REQUEST_LINE_BYTES, MAX_RESPONSE_LINE_BYTES, PackageIdentity, ProtocolError, ServeResponse,
    decode_request, decode_response, encode_response,
};
use fandhe_edge_guard::resource::{
    SafetyLimits, current_process_rss_bytes, process_rss_bytes, safety_limits_for_kind,
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
///
/// 名前の数字は `PROTOCOL_VERSION` と一致させる（単体テストで照合）。プロトコルを上げるときは、ここも上げて
/// 今の名前を [`LEGACY_SOCKET_FILES`] へ移す。
pub const SOCKET_FILE: &str = "serve.v1.sock";
/// 以前の版のソケットのファイル名。起動時にロックを取れたら消す（持ち主はもういない。古い CLI が死んだ
/// ソケットへつなぎに行かないように）。
const LEGACY_SOCKET_FILES: [&str; 1] = ["serve.sock"];
/// worker を kill してから回収できるまで待つ上限。ガード層の監督（`resource::KILL_WAIT_TIMEOUT`）と同じ 2 秒。
/// SIGKILL の後の回収は通常数 ms で済み、これを超えるのは終了が完了しない異常なので、親ごと終える。
const WORKER_REAP_TIMEOUT: Duration = Duration::from_secs(2);
/// 親が終わるときに worker の回収を待つ上限（終了を遅らせすぎない。回収できなくても終える）。
const EXIT_REAP_TIMEOUT: Duration = Duration::from_millis(200);
/// 回収待ちの間隔。
const REAP_POLL_INTERVAL: Duration = Duration::from_millis(5);
const LOCK_FILE: &str = "serve.lock";

/// 判定待ちの上限（判定中の 1 件を除く）。超えたら `limit_exceeded`。
const QUEUE_DEPTH: usize = 8;
/// 同時に扱う接続（スレッド）の上限。超えた接続は読まずに `limit_exceeded` を返して閉じる。
const MAX_CONNECTIONS: usize = 64;
/// 要求 1 行を読み切るまでの合計の期限（遅送りで接続を占有されないため）。
const REQUEST_READ_TIMEOUT: Duration = Duration::from_secs(10);
/// 判定スレッドの応答を待つ上限（待ち 8 件 + 判定中 1 件が最長の期限〔追加学習 60.5 秒 + 余裕 1 秒〕で並んだ場合を覆う。
/// 間に worker の作り直しが入っても、待っている要求は 1 件の期限で `limit_exceeded` になる）。
const REPLY_WAIT: Duration = Duration::from_secs(10 * 60);
/// 起動失敗の JSON を見張りのスレッドが書く場合に、main が終了を待つ上限。
const STARTUP_ERROR_WAIT: Duration = Duration::from_secs(5);
/// 資源の見張りの間隔（監督の RSS ポーリングと同程度）。
const GUARD_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// 応答の書き込みの期限。
const RESPONSE_WRITE_TIMEOUT: Duration = Duration::from_secs(10);

/// worker 子として起動されたことを示す環境変数（値が `"1"` と完全一致するときだけ worker。#709）。
pub const SERVE_WORKER_ENV: &str = "FANDHE_EDGE_SERVE_WORKER";
/// 親が worker 子へ自分の pid を渡す環境変数。worker は親がこの pid でなくなったら終わる（起動時に自分で
/// 読む親の pid だと、読む前に親が終わっていた場合に気づけないため）。
const SERVE_PARENT_PID_ENV: &str = "FANDHE_EDGE_SERVE_PARENT_PID";
/// 要求の `cli_version` と照合する自分の版。
const CLI_VERSION: &str = env!("CARGO_PKG_VERSION");
/// 判定スレッドが、要求を待つ間に worker の生死を確かめる間隔。
const IDLE_CHECK_INTERVAL: Duration = Duration::from_millis(200);
/// worker の応答を待つ時間に、1 件の期限へ足す余裕（worker 自身のハード期限の応答を先に受け取るため）。
const WORKER_REPLY_SLACK: Duration = Duration::from_secs(1);

/// 1 件も判定しないまま worker が終わる作り直しが、この回数続いたら親ごと終える（外からの kill の繰り返し・
/// OOM・読み込み直後のクラッシュで回り続けない）。
const MAX_BARREN_RESTARTS: u32 = 3;

/// worker の判定を試験で遅らせる環境変数（ms）。デバッグビルド限定でリリースビルドには入らない（待ちの上限の試験用）。
#[cfg(debug_assertions)]
const TEST_DELAY_ENV: &str = "FANDHE_EDGE_SERVE_TEST_DELAY_MS";
/// 入力がこの値と一致する要求で、worker が応答せず止まる（親が止めて作り直す試験用。デバッグビルド限定）。
#[cfg(debug_assertions)]
const TEST_HANG_ENV: &str = "FANDHE_EDGE_SERVE_TEST_HANG_TEXT";
/// [`TEST_HANG_ENV`] で止まる長さ（親が SIGKILL された試験でも worker が残り続けないよう有界）。
#[cfg(debug_assertions)]
const TEST_HANG_DURATION: Duration = Duration::from_secs(60);
/// 親の 1 件の期限をこの値（ms）に縮める（デバッグビルド限定）。
#[cfg(debug_assertions)]
const TEST_TIME_ENV: &str = "FANDHE_EDGE_SERVE_TEST_TIME_MS";
/// 親の起動全体の期限をこの値（ms）に縮める（デバッグビルド限定。worker の `ready` 待ちと worker 自身の
/// 読み込みの期限には効かない）。
#[cfg(debug_assertions)]
const TEST_STARTUP_ENV: &str = "FANDHE_EDGE_SERVE_TEST_STARTUP_MS";
/// 親が worker に使う RSS の上限をこの値（バイト）に縮める（デバッグビルド限定）。
#[cfg(debug_assertions)]
const TEST_RSS_LIMIT_ENV: &str = "FANDHE_EDGE_SERVE_TEST_RSS_LIMIT_BYTES";

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
        match stream.set_read_timeout(Some(remaining)) {
            Ok(()) => {}
            // macOS は、相手が閉じた後のソケットへの期限の設定を EINVAL で断る。期限なしで読まないよう
            // ノンブロッキングにして読む（閉じた接続の読みは残りのデータか EOF をすぐ返す。閉じていないのに
            // EINVAL だった場合も、下で `WouldBlock` を時間切れとして返すので止まらない。#709）。
            Err(e) if e.kind() == io::ErrorKind::InvalidInput => stream.set_nonblocking(true)?,
            Err(e) => return Err(e),
        }
        let n = match stream.read(&mut chunk) {
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                return Err(io::ErrorKind::TimedOut.into());
            }
            Err(e) => return Err(e),
        };
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

/// 終了処理に入ったか。入った後は見張りが何もしない（終了中に別の理由で終了コードを上書きしない）。
static EXITING: AtomicBool = AtomicBool::new(false);

/// 終了処理に入る。最初に入ったスレッドだけが真を受け取り、終了コードを決める。
fn begin_exit() -> bool {
    !EXITING.swap(true, Ordering::SeqCst)
}

/// 終了する。先に別のスレッドが終了処理に入っていたら、そちらの終了を待つ（終了コードを上書きしない）。
/// worker のロックを持ったまま呼ばないこと。
fn exit_with(code: ExitCode) -> ! {
    if !begin_exit() {
        wait_for_exit();
    }
    finish_exit(code)
}

/// worker を止め、ログを（上限つきで）流し切ってから終了する。stderr が詰まっていても `LOG_DRAIN_WAIT` を
/// 超えて待たない。[`begin_exit`] で終了処理に入ったスレッドだけが呼ぶ。
fn finish_exit(code: ExitCode) -> ! {
    kill_worker();
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
            kill_worker();
            std::process::exit(i32::from(code.code()));
        })
        .is_ok()
}

fn claim_stdout() -> bool {
    !STDOUT_CLAIMED.swap(true, Ordering::SeqCst)
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// 終了を決めた別のスレッド（stdout へ JSON を書いている側・終了処理に入った側）がプロセスを終えるのを待つ。
/// 相手が詰まっても残り続けないよう、期限つき。
fn wait_for_exit() -> ! {
    let deadline = Instant::now() + STARTUP_ERROR_WAIT;
    while Instant::now() < deadline {
        thread::park_timeout(Duration::from_millis(100));
    }
    finish_exit(ExitCode::RuntimeError)
}

/// 起動失敗の 1 行を stdout へ 1 回だけ書く。先に見張りのスレッドが書く（終了する）側なら、その終了を待つ。
fn emit_startup<W: Write>(
    out: &mut W,
    code: ExitCode,
    write: impl FnOnce(&mut W) -> io::Result<()>,
) -> io::Result<ExitCode> {
    if !claim_stdout() {
        wait_for_exit();
    }
    // main が返ればプロセスは終わるが、書き込みで止まっても終われるよう保険を張る。張れなければ書かない。
    if arm_exit_deadline(code) {
        write(out)?;
    }
    Ok(code)
}

fn emit_startup_error<W: Write>(out: &mut W, report: &ErrorReport) -> io::Result<ExitCode> {
    emit_startup(out, report.code, |out| {
        emit_error_report(out, report).map(|_| ())
    })
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

fn limit_report() -> ErrorReport {
    ErrorReport::new(
        ExitCode::LimitExceeded,
        default_message(ExitCode::LimitExceeded),
    )
}

fn limit_response() -> ServeResponse {
    error_response(&limit_report())
}

/// 現在のプロセスが常駐の worker 子か。
fn is_worker() -> bool {
    std::env::var_os(SERVE_WORKER_ENV).is_some_and(|v| v == "1")
}

#[cfg(debug_assertions)]
fn test_env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// 親が 1 件に許す時間（委ねる側の読みの期限 `time + SINGLE_HARD_LIMIT_GRACE` と同じ式）。
fn job_budget(limits: SafetyLimits) -> Duration {
    #[cfg(debug_assertions)]
    if let Some(ms) = test_env_u64(TEST_TIME_ENV) {
        return Duration::from_millis(ms);
    }
    limits.time().get().saturating_add(SINGLE_HARD_LIMIT_GRACE)
}

/// 親の起動（待ち受けの開始まで）に許す時間（種類別の時間の上限と同じ値）。
fn startup_budget(limits: SafetyLimits) -> Duration {
    #[cfg(debug_assertions)]
    if let Some(ms) = test_env_u64(TEST_STARTUP_ENV) {
        return Duration::from_millis(ms);
    }
    limits.time().get()
}

/// 親が worker 子に許す RSS（種類別の安全上限と同じ値）。
fn worker_rss_limit(limits: SafetyLimits) -> u64 {
    #[cfg(debug_assertions)]
    if let Some(bytes) = test_env_u64(TEST_RSS_LIMIT_ENV) {
        return bytes;
    }
    limits.memory().get()
}

/// 改行までの 1 行をパイプから読む（本文は `limit` バイトまで。改行込みで返す）。`Ok(None)` は EOF。
/// 上限を超えた行・改行の前で切れた行は `InvalidData`。パイプには読みの期限が無いため、期限は呼び出し側が持つ。
fn read_pipe_line<R: BufRead>(reader: &mut R, limit: usize) -> io::Result<Option<Vec<u8>>> {
    let cap = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
    let mut line = Vec::new();
    if reader.by_ref().take(cap).read_until(b'\n', &mut line)? == 0 {
        return Ok(None);
    }
    if line.last() == Some(&b'\n') {
        Ok(Some(line))
    } else {
        Err(io::ErrorKind::InvalidData.into())
    }
}

// ───────────────────────── worker 子 ─────────────────────────

/// worker 子: パッケージを 1 回読み込み、stdin の要求 1 行ごとに判定して stdout へ応答 1 行を返す（#709）。
///
/// 最初の 1 行は、読み込みに成功すれば `ready`、失敗すれば単独の `infer` と同じエラーの `ok`（その終了コードで
/// 終わる）。ソケット・ロック・置き場所には触らず、自分の資源の見張りも持たない（親が見張って止める）。
/// stdin が EOF（親が終わった）なら exit 0。ハード期限で計算スレッドを切り離したら、応答を書いた後に exit 20
/// （切り離したスレッドとメモリを積み上げない。親が作り直す）。
fn run_worker<W: Write>(out: &mut W, args: &ServeArgs, cwd: &Path) -> io::Result<ExitCode> {
    let loaded = Arc::new(AtomicBool::new(false));
    if spawn_worker_guard(args, cwd, Arc::clone(&loaded)).is_err() {
        let report = runtime("cannot start serve worker guard thread");
        out.write_all(&encode_response(&error_response(&report)))?;
        out.flush()?;
        return Ok(report.code);
    }
    let infer_args = InferArgs {
        package: args.package.clone(),
        source: InferSource::Text {
            text: String::new(),
            id: None,
        },
        version: None,
    };
    let mut send = |response: &ServeResponse| {
        out.write_all(&encode_response(response))
            .and_then(|()| out.flush())
    };
    let prepared = match prepare(cwd, &infer_args) {
        Ok(prepared) => prepared,
        Err(report) => {
            send(&error_response(&report))?;
            return Ok(report.code);
        }
    };
    let Prepared {
        definition,
        pipeline,
        scope,
        single_time,
        identity,
        ..
    } = prepared;
    let io_schema = definition.io().clone();
    let pipeline = Arc::new(pipeline);
    loaded.store(true, Ordering::SeqCst);
    send(&ServeResponse::Ready {
        package: identity,
        cli_version: CLI_VERSION.to_string(),
    })?;
    let mut input = io::stdin().lock();
    loop {
        let line = match read_pipe_line(&mut input, MAX_REQUEST_LINE_BYTES) {
            Ok(Some(line)) => line,
            Ok(None) => return Ok(ExitCode::Ok),
            Err(_) => return Ok(ExitCode::RuntimeError),
        };
        let accepted_unix = unix_secs();
        let Ok(request) = decode_request(&line, CLI_VERSION) else {
            send(&error_response(&invalid("invalid serve request")))?;
            continue;
        };
        #[cfg(debug_assertions)]
        {
            if let Some(ms) = test_env_u64(TEST_DELAY_ENV) {
                thread::sleep(Duration::from_millis(ms));
            }
            if std::env::var(TEST_HANG_ENV).is_ok_and(|v| v == request.text) {
                // 応答せずに止まる（親が止めて作り直す試験用）。親が先に消えても残り続けないよう有界にする。
                thread::sleep(TEST_HANG_DURATION);
                return Ok(ExitCode::RuntimeError);
            }
        }
        let call = SingleCall {
            io: &io_schema,
            options: definition.options(),
            scope: &scope,
            id: DEFAULT_TEXT_ID,
            text: &request.text,
            duration: single_time,
        };
        let (response, detached) = match infer_single_to_bytes(call, Arc::clone(&pipeline)) {
            Ok((buf, code, detached)) => match String::from_utf8(buf) {
                Ok(stdout) => (
                    ServeResponse::Ok {
                        exit_code: code.code(),
                        stdout,
                    },
                    detached,
                ),
                Err(_) => (error_response(&runtime("serve judge failed")), false),
            },
            Err(_) => (error_response(&runtime("serve judge failed")), false),
        };
        if let ServeResponse::Ok { exit_code, .. } = &response {
            // 本文・判定結果は出さない。id は英数字と `-_` だけに絞る。
            let id: String = request
                .id
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
                .collect();
            log_line(format_args!(
                "accepted_at={accepted_unix} id={id} exit_code={exit_code}"
            ));
        }
        if detached {
            // この応答を最後に終わると親へ伝える（親は次の要求を渡さずに作り直す）。
            if let ServeResponse::Ok { exit_code, stdout } = response {
                send(&ServeResponse::OkRetiring { exit_code, stdout })?;
            }
            log_line(format_args!(
                "judge stalled past the hard limit; worker exiting"
            ));
            exit_with(ExitCode::LimitExceeded);
        }
        send(&response)?;
    }
}

/// worker 自身の見張りが終了を決めたときの終了コードとログの文。`None` は続行。
///
/// - 親が `expected_parent` でなくなった: 親の見張りを失ったまま動き続けないよう、すぐ終わる（exit 70）
/// - 読み込み中（`startup`）に `time_limit` を過ぎた: exit 20
/// - 自分の RSS が `memory_limit` を超えた: exit 20。読めなければ fail-closed で exit 70
fn worker_guard_verdict(
    parent: u32,
    expected_parent: u32,
    startup_elapsed: Option<Duration>,
    time_limit: Duration,
    rss: io::Result<Option<u64>>,
    memory_limit: u64,
) -> Option<(ExitCode, &'static str)> {
    if parent != expected_parent {
        return Some((ExitCode::RuntimeError, "parent is gone; worker exiting"));
    }
    if startup_elapsed.is_some_and(|elapsed| elapsed > time_limit) {
        return Some((
            ExitCode::LimitExceeded,
            "worker startup time limit exceeded; worker exiting",
        ));
    }
    match rss {
        Ok(Some(rss)) if rss > memory_limit => Some((
            ExitCode::LimitExceeded,
            "worker memory limit exceeded; worker exiting",
        )),
        Ok(_) => None,
        Err(_) => Some((
            ExitCode::RuntimeError,
            "worker memory probe failed; worker exiting",
        )),
    }
}

/// worker 自身の見張りのスレッド（約 50 ms ごと。REQ-39・#709）。親の見張りと二重で、先に気づいた側が止める。
/// 親が SIGTERM・SIGKILL で消えた後も、読み込み・判定の途中の worker が上限なしで動き続けないためのもの。
///
/// 上限は親と同じ求め方（`kind` の先読み）。読み込み中（`loaded` が偽）に上限で終わるときは、親が起動の失敗
/// として同じ終了コードを出せるよう、エラーの 1 行を stdout に書いてから終わる（読み込み中は main スレッドが
/// stdout に書かない）。読み込み後は何も書かずに終わり、親は終了コード 20 を回収時に見て `limit_exceeded` にする。
fn spawn_worker_guard(args: &ServeArgs, cwd: &Path, loaded: Arc<AtomicBool>) -> io::Result<()> {
    let limits = peek_checked_kind(cwd, &args.package)
        .map_or(SafetyLimits::STRICT, |k| safety_limits_for_kind(k.as_str()));
    let expected_parent = std::env::var(SERVE_PARENT_PID_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(std::os::unix::process::parent_id);
    let started = Instant::now();
    thread::Builder::new()
        .name("serve-worker-guard".to_string())
        .spawn(move || {
            loop {
                thread::sleep(GUARD_POLL_INTERVAL);
                let startup = !loaded.load(Ordering::SeqCst);
                let Some((code, msg)) = worker_guard_verdict(
                    std::os::unix::process::parent_id(),
                    expected_parent,
                    startup.then(|| started.elapsed()),
                    limits.time().get(),
                    current_process_rss_bytes(),
                    limits.memory().get(),
                ) else {
                    continue;
                };
                if !begin_exit() {
                    wait_for_exit();
                }
                log_line(format_args!("{msg}"));
                if startup && arm_exit_deadline(code) {
                    let report = ErrorReport::new(code, default_message(code));
                    let mut stdout = io::stdout().lock();
                    let _ = stdout.write_all(&encode_response(&error_response(&report)));
                    let _ = stdout.flush();
                }
                finish_exit(code);
            }
        })
        .map(|_| ())
}

// ───────────────────────── 親: worker 子の持ち方 ─────────────────────────

/// 親が持つ worker 子の状態。見張り（計測・kill）・判定スレッド（回収・作り直し）・終了処理（kill）が同じ
/// ロックの下で触る。`child` が `Some` の間は未回収なので、pid が別のプロセスへ再利用されることはない
/// （回収するときは、ロックの下で `take` してから `wait` する。回収済みの pid を計測しない）。
struct WorkerSlot {
    child: Option<Child>,
    /// `ready` の直後の計測で、RSS が上限内と確かめたか。確かめる前（起動中）の超過は、作り直しても同じ
    /// なので親ごと終える。確かめた後の超過は worker を止めるだけ。
    settled: bool,
    /// この worker が 1 件以上の判定を終えたか（判定しないまま終わる作り直しの連続を数えるため）。
    judged: bool,
    /// 見張りが上限（RSS・1 件の期限）で止めたか。判定スレッドが応答を `limit_exceeded` にする。
    limit_killed: bool,
    /// 判定中の 1 件の期限。過ぎたら見張りが止める（要求の書き込みで止まった判定スレッドも外れる）。
    kill_at: Option<Instant>,
}

static WORKER: Mutex<WorkerSlot> = Mutex::new(WorkerSlot {
    child: None,
    settled: false,
    judged: false,
    limit_killed: false,
    kill_at: None,
});

/// ソケットを作って待ち受けを始めたか。偽の間（起動中）に親が終わるときは、stdout にエラー JSON を出す。
static SERVING: AtomicBool = AtomicBool::new(false);

/// ロックの中では長く待つ処理をしない（kill・計測・フラグの読み書きだけ）。
fn worker_slot() -> MutexGuard<'static, WorkerSlot> {
    WORKER.lock().unwrap_or_else(PoisonError::into_inner)
}

/// kill してから、`timeout` まで短い間隔で回収を試みる（無期限の `wait` をしない。REQ-39）。回収できたら真。
/// kill の失敗は、既に終わっていて回収できた場合だけ成功として扱う。`try_wait` は回収できたら `Ok(true)`。
fn kill_and_reap(
    kill: impl FnOnce() -> io::Result<()>,
    mut try_wait: impl FnMut() -> io::Result<bool>,
    timeout: Duration,
) -> bool {
    let kill_failed = kill().is_err();
    let deadline = Instant::now() + timeout;
    loop {
        match try_wait() {
            Ok(true) => return true,
            Ok(false) => {}
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => return false,
        }
        // kill できず、まだ終わってもいない worker は待っても終わらない。
        if kill_failed || Instant::now() >= deadline {
            return false;
        }
        thread::sleep(REAP_POLL_INTERVAL);
    }
}

fn kill_and_reap_child(child: &mut Child, timeout: Duration) -> bool {
    // `kill` と `try_wait` が同じ `child` を可変で借りるため、内部可変性で 1 つずつ使う。
    let child = std::cell::RefCell::new(child);
    kill_and_reap(
        || child.borrow_mut().kill(),
        || child.borrow_mut().try_wait().map(|status| status.is_some()),
        timeout,
    )
}

/// 親が終わる前に worker を止める（7B の読み込みを続ける孤児を残さない）。回収は [`EXIT_REAP_TIMEOUT`] まで
/// 待つだけで、できなくても終了を続ける（残りは親の終了で init が回収する）。
fn kill_worker() {
    let mut slot = worker_slot();
    if let Some(child) = slot.child.as_mut()
        && kill_and_reap_child(child, EXIT_REAP_TIMEOUT)
    {
        // 回収済みの pid を残さない（見張りが消えた pid を計測しない）。
        slot.child = None;
    }
}

/// worker を止めて回収する（[`WORKER_REAP_TIMEOUT`] まで）。戻り値は、上限で止まったか（親の見張りが止めた、
/// または worker が自分の見張りで exit 20 で終わっていた）。判定中の要求を `limit_exceeded` にするのに使う。
/// 回収できなければ、止められない worker を抱えたまま続けず、理由をログに出して親ごと exit 70 で終える。
fn reap_worker() -> bool {
    let (child, limit_killed) = {
        let mut slot = worker_slot();
        (slot.child.take(), slot.limit_killed)
    };
    let Some(mut child) = child else {
        return limit_killed;
    };
    if !kill_and_reap_child(&mut child, WORKER_REAP_TIMEOUT) {
        // 終了処理が最後にもう一度だけ kill を試みられるよう、戻してから終える。
        worker_slot().child = Some(child);
        fatal_exit(
            ExitCode::RuntimeError,
            "cannot stop worker; exiting",
            !SERVING.load(Ordering::SeqCst),
        );
    }
    // 回収済みなので、保持されている終了状態を読むだけ（kill が効く前に自分で終わっていれば終了コードが残る）。
    let self_limited = child.try_wait().ok().flatten().and_then(|s| s.code())
        == Some(i32::from(ExitCode::LimitExceeded.code()));
    limit_killed || self_limited
}

/// 判定スレッドが持つ worker との口（stdin と、stdout の行を渡す受け口）。
struct WorkerIo {
    stdin: ChildStdin,
    /// stdout を読む専用スレッドが 1 行ずつ渡す。worker が終わる・行が不正なら切れる。
    lines: mpsc::Receiver<Vec<u8>>,
}

impl WorkerIo {
    /// worker が生きていて、頼んでいない行も来ていない。
    fn is_idle(&self) -> bool {
        matches!(self.lines.try_recv(), Err(mpsc::TryRecvError::Empty))
    }
}

/// 自分自身を worker 子として起動する（`serve --package <同じ値>`・[`SERVE_WORKER_ENV`]`=1`・stderr は継承）。
fn spawn_worker(args: &ServeArgs, cwd: &Path) -> Result<WorkerIo, ErrorReport> {
    let failed = || runtime("cannot start serve worker");
    let exe = std::env::current_exe().map_err(|_| failed())?;
    let mut child = Command::new(exe)
        .arg("serve")
        .arg("--package")
        .arg(&args.package)
        .env(SERVE_WORKER_ENV, "1")
        .env(SERVE_PARENT_PID_ENV, std::process::id().to_string())
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|_| failed())?;
    let (stdin, stdout) = (child.stdin.take(), child.stdout.take());
    *worker_slot() = WorkerSlot {
        child: Some(child),
        settled: false,
        judged: false,
        limit_killed: false,
        kill_at: None,
    };
    let (Some(stdin), Some(stdout)) = (stdin, stdout) else {
        reap_worker();
        return Err(failed());
    };
    let (tx, lines) = mpsc::sync_channel(1);
    let reader = thread::Builder::new()
        .name("serve-worker-out".to_string())
        .spawn(move || {
            let mut stdout = BufReader::new(stdout);
            while let Ok(Some(line)) = read_pipe_line(&mut stdout, MAX_RESPONSE_LINE_BYTES) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
    if reader.is_err() {
        reap_worker();
        return Err(failed());
    }
    Ok(WorkerIo { stdin, lines })
}

/// worker の最初の 1 行の結果。
#[derive(Debug, PartialEq, Eq)]
enum Startup {
    Ready(PackageIdentity),
    /// 読み込みに失敗した（単独の `infer` と同じ終了コードと stdout）。
    Failed(ExitCode, String),
    TimedOut,
    /// worker の版が親と違う（稼働中にバイナリが差し替わった）。
    VersionChanged,
    /// 作り直した worker の読み込んだ内容が、起動時と違う（置き場所の中身が変わった）。
    PackageChanged,
    /// 最初の 1 行が不正、または書く前に終わった。
    Broken,
}

/// worker の最初の 1 行を読み分ける。`own_version` は親の版、`expected` は起動時の同一性（最初の起動では `None`）。
/// 版・同一性が違う worker は使わない（親は理由をログに出して終わる。別の版は要求を読めず、別の中身は
/// 起動時に決めた上限・照合と食い違うため）。
fn classify_first_line(
    line: &[u8],
    own_version: &str,
    expected: Option<&PackageIdentity>,
) -> Startup {
    match decode_response(line) {
        Ok(ServeResponse::Ready {
            package,
            cli_version,
        }) => {
            if cli_version != own_version {
                Startup::VersionChanged
            } else if expected.is_some_and(|e| *e != package) {
                Startup::PackageChanged
            } else {
                Startup::Ready(package)
            }
        }
        Ok(ServeResponse::Ok { exit_code, stdout }) => {
            match ExitCode::try_from(i32::from(exit_code)) {
                Ok(code) if code != ExitCode::Ok => Startup::Failed(code, stdout),
                _ => Startup::Broken,
            }
        }
        _ => Startup::Broken,
    }
}

fn await_ready(io: &WorkerIo, wait: Duration, expected: Option<&PackageIdentity>) -> Startup {
    match io.lines.recv_timeout(wait) {
        Ok(line) => classify_first_line(&line, CLI_VERSION, expected),
        Err(mpsc::RecvTimeoutError::Timeout) => Startup::TimedOut,
        Err(mpsc::RecvTimeoutError::Disconnected) => Startup::Broken,
    }
}

/// `ready` の直後に worker の RSS を 1 回確かめ、上限内ならそれ以降の超過を「worker を止めるだけ」に切り替える。
/// 超えていたら [`guard_tick`] が親ごと終える（見張りの間隔より速く読み込みが済んでも、上限を超える
/// パッケージで待ち受けを始めない・作り直しを繰り返さない）。
fn settle_worker(limits: SafetyLimits, startup: bool) {
    guard_tick(limits, startup);
    worker_slot().settled = true;
}

/// worker の応答 1 行を読み分ける。戻り値は（クライアントへ返す応答, worker がこの応答を最後に終わるか）。
/// `ok_retiring` はクライアントへは `ok` として返す。それ以外の行は不正（`None`）。
fn classify_answer(line: &[u8]) -> Option<(ServeResponse, bool)> {
    match decode_response(line) {
        Ok(response @ ServeResponse::Ok { .. }) => Some((response, false)),
        Ok(ServeResponse::OkRetiring { exit_code, stdout }) => {
            Some((ServeResponse::Ok { exit_code, stdout }, true))
        }
        _ => None,
    }
}

/// 1 件も判定しないまま worker が終わった回数の連続を数える（1 件でも判定していれば数え直す）。
/// [`MAX_BARREN_RESTARTS`] に達したら、呼び出し側は親ごと終える。
fn barren_streak(streak: u32, judged_any: bool) -> u32 {
    if judged_any {
        0
    } else {
        streak.saturating_add(1)
    }
}

// ───────────────────────── 親 ─────────────────────────

/// 判定スレッドへ渡す 1 件。
struct Job {
    /// 検証済みの要求の 1 行（改行つき）。worker へそのまま渡す。
    line: Vec<u8>,
    /// 受け付けた時刻（待っている間に委ねる側の期限が過ぎた要求を判定しないため）。
    accepted_at: Instant,
    reply: SyncSender<ServeResponse>,
}

/// 常駐を起動して要求を受け付け続ける。起動に失敗したときだけ、単独の `infer` と同じ形のエラーを
/// `out` へ書いて戻る（ソケットは作らない）。成功時は戻らない。
/// [`SERVE_WORKER_ENV`] つきで起動されたときは worker 子として動く（[`run_worker`]）。
///
/// # Errors
/// `out` への書き込み失敗。
pub fn run<W: Write>(out: &mut W, args: &ServeArgs, cwd: &Path) -> io::Result<ExitCode> {
    init_logger();
    if is_worker() {
        return run_worker(out, args, cwd);
    }
    let result = run_parent(out, args, cwd);
    // 起動失敗で戻る。見張りに終了コードを上書きさせず、worker を止めてから終わる。
    let _ = begin_exit();
    kill_worker();
    result
}

fn run_parent<W: Write>(out: &mut W, args: &ServeArgs, cwd: &Path) -> io::Result<ExitCode> {
    // 起動時から、単独の監督と同じ種類別の RSS の上限で自分と worker を見張る。
    let limits = peek_checked_kind(cwd, &args.package)
        .map_or(SafetyLimits::STRICT, |k| safety_limits_for_kind(k.as_str()));
    if spawn_guard_thread(limits).is_err() {
        return emit_startup_error(out, &runtime("cannot start serve guard thread"));
    }
    let (listener, io, identity, _lock) = match start(args, cwd, limits) {
        Ok(started) => started,
        Err(StartFailure::Report(report)) => return emit_startup_error(out, &report),
        Err(StartFailure::Worker(code, stdout)) => {
            return emit_startup(out, code, |out| {
                out.write_all(stdout.as_bytes()).and_then(|()| out.flush())
            });
        }
    };
    SERVING.store(true, Ordering::SeqCst);
    let identity = Arc::new(identity);
    let (tx, rx) = mpsc::sync_channel::<Job>(QUEUE_DEPTH);
    let ctx = JudgeCtx {
        args: args.clone(),
        cwd: cwd.to_path_buf(),
        limits,
        identity: Arc::clone(&identity),
    };
    if thread::Builder::new()
        .name("serve-judge".to_string())
        .spawn(move || judge_loop(&rx, io, &ctx))
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

/// 起動の失敗。
enum StartFailure {
    /// 親が見つけた失敗。
    Report(ErrorReport),
    /// worker が読み込みに失敗した。単独の `infer` と同じ終了コードと stdout をそのまま出す。
    Worker(ExitCode, String),
}

impl From<ErrorReport> for StartFailure {
    fn from(report: ErrorReport) -> Self {
        Self::Report(report)
    }
}

/// 起動順（置き場所 → ロック → worker の起動と全量検証・読み込み → 残ったソケットの除去 → bind → 0600）。
/// 全量検証に失敗したらソケットを作らない。
fn start(
    args: &ServeArgs,
    cwd: &Path,
    limits: SafetyLimits,
) -> Result<(UnixListener, WorkerIo, PackageIdentity, fs::File), StartFailure> {
    let dir = serve_dir().ok_or_else(|| invalid("serve directory is not configured"))?;
    let dir_meta = ensure_dir(&dir)?;
    let lock = acquire_lock(&dir, &dir_meta)?;
    // 以前の版の名前で残っているソケットを消す（ロックを取れたので持ち主はいない。ソケット以外は触らない）。
    for name in LEGACY_SOCKET_FILES {
        let legacy = dir.join(name);
        if fs::symlink_metadata(&legacy).is_ok_and(|m| m.file_type().is_socket()) {
            let _ = fs::remove_file(&legacy);
        }
    }
    let io = spawn_worker(args, cwd)?;
    let identity = match await_ready(&io, limits.time().get(), None) {
        Startup::Ready(identity) => identity,
        Startup::Failed(code, stdout) => {
            reap_worker();
            return Err(StartFailure::Worker(code, stdout));
        }
        Startup::TimedOut => {
            log_line(format_args!("startup time limit exceeded; exiting"));
            reap_worker();
            return Err(limit_report().into());
        }
        Startup::VersionChanged => {
            log_line(format_args!("worker version differs; exiting"));
            return Err(runtime("serve worker version differs").into());
        }
        // 見張りが RSS 超過で終了を決めた場合は、見張りが先に stdout を取っている（`emit_startup` が待つ）。
        Startup::Broken | Startup::PackageChanged => {
            return Err(runtime("serve worker failed to start").into());
        }
    };
    settle_worker(limits, true);
    let sock = dir.join(SOCKET_FILE);
    remove_stale_socket(&sock)?;
    let listener = UnixListener::bind(&sock).map_err(|_| runtime("cannot bind serve socket"))?;
    if fs::set_permissions(&sock, fs::Permissions::from_mode(0o600)).is_err() {
        let _ = fs::remove_file(&sock);
        return Err(runtime("cannot restrict serve socket").into());
    }
    Ok((listener, io, identity, lock))
}

/// 応答を書いて閉じる（失敗は無視。相手が既にいない）。
fn reply(mut stream: UnixStream, response: &ServeResponse) {
    let _ = stream.set_write_timeout(Some(RESPONSE_WRITE_TIMEOUT));
    let _ = stream.write_all(&encode_response(response));
    let _ = stream.flush();
}

/// 1 接続: 要求を読み、版とパッケージを照合し、判定スレッドへ渡して応答を返す。
fn handle(mut stream: UnixStream, identity: &PackageIdentity, tx: &SyncSender<Job>) {
    let response = respond(&mut stream, identity, tx);
    reply(stream, &response);
}

fn respond(
    stream: &mut UnixStream,
    identity: &PackageIdentity,
    tx: &SyncSender<Job>,
) -> ServeResponse {
    let deadline = Instant::now() + REQUEST_READ_TIMEOUT;
    let Ok(mut line) = read_line_bounded(stream, MAX_REQUEST_LINE_BYTES, deadline) else {
        return error_response(&invalid("invalid serve request"));
    };
    let request = match decode_request(&line, CLI_VERSION) {
        Ok(request) => request,
        // 版の違う CLI には判定せず、単独へ戻らせる（#709）。
        Err(ProtocolError::VersionMismatch) => return ServeResponse::VersionMismatch,
        Err(_) => return error_response(&invalid("invalid serve request")),
    };
    if request.package != *identity {
        return ServeResponse::PackageMismatch;
    }
    drop(request);
    if line.last() != Some(&b'\n') {
        line.push(b'\n');
    }
    let (reply_tx, reply_rx) = mpsc::sync_channel(1);
    let job = Job {
        line,
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
    // 判定スレッドは必ず（期限内に）返すが、無期限には待たない。
    match reply_rx.recv_timeout(REPLY_WAIT) {
        Ok(response) => response,
        Err(mpsc::RecvTimeoutError::Timeout) => limit_response(),
        Err(mpsc::RecvTimeoutError::Disconnected) => error_response(&runtime("serve judge failed")),
    }
}

/// 判定スレッドが作り直しに使うもの。
struct JudgeCtx {
    args: ServeArgs,
    cwd: PathBuf,
    limits: SafetyLimits,
    /// 起動時の同一性。作り直した worker が違う値を名乗ったら親ごと終える。
    identity: Arc<PackageIdentity>,
}

/// 判定スレッド: worker の持ち主。要求を 1 件ずつ worker へ渡して応答を返し、worker が止まったら作り直す
/// （作り直しはここだけが行う）。
fn judge_loop(rx: &mpsc::Receiver<Job>, mut io: WorkerIo, ctx: &JudgeCtx) {
    let budget = job_budget(ctx.limits);
    let mut streak = 0;
    loop {
        let job = match rx.recv_timeout(IDLE_CHECK_INTERVAL) {
            Ok(job) => Some(job),
            Err(mpsc::RecvTimeoutError::Timeout) => None,
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        };
        // 待っている間に worker が止まっていたら（外からの kill・見張りによる停止）、次の要求が読み込み時間を
        // 払わずに済むよう先に作り直す。
        if !io.is_idle() {
            io = respawn(ctx, &mut streak);
        }
        let Some(job) = job else { continue };
        // 待っている間に期限を過ぎた要求は、相手がもう諦めているので判定せず次へ。
        if job.accepted_at.elapsed() > budget {
            let _ = job.reply.send(limit_response());
            continue;
        }
        let mut asked = ask_worker(&mut io, &job.line, budget);
        if matches!(asked, Asked::NotDelivered) {
            // worker は要求を読んでいない（渡す前に止まっていた）。判定していない要求を失敗にしないよう、
            // 作り直して 1 回だけ渡し直す。応答待ちの間の切断は渡し直さない（毒入力で回り続けない）。
            io = respawn(ctx, &mut streak);
            asked = if job.accepted_at.elapsed() > budget {
                Asked::Answer(limit_response())
            } else {
                ask_worker(&mut io, &job.line, budget)
            };
        }
        let response = match asked {
            Asked::Answer(response) => response,
            Asked::NotDelivered => {
                reap_worker();
                error_response(&runtime("serve judge failed"))
            }
        };
        let _ = job.reply.send(response);
        if worker_slot().child.is_none() {
            io = respawn(ctx, &mut streak);
        }
    }
}

/// worker へ 1 件渡した結果。
enum Asked {
    /// クライアントへ返す応答（判定結果、または渡した後の失敗）。
    Answer(ServeResponse),
    /// worker が要求を読んでいない（渡す前に見張りが止めていた・書き込みに失敗した）。
    NotDelivered,
}

/// 要求の 1 行を worker へ渡し、応答の 1 行を `budget + WORKER_REPLY_SLACK` まで待つ。
/// 渡した後の期限切れ・見張りによる停止は `limit_exceeded`、それ以外で応答が得られなければ `runtime_error`
/// （どちらも worker を止めて回収する。作り直しは呼び出し側）。`ok_retiring` を受けたら worker を回収する。
fn ask_worker(io: &mut WorkerIo, line: &[u8], budget: Duration) -> Asked {
    let wait = budget.saturating_add(WORKER_REPLY_SLACK);
    {
        let mut slot = worker_slot();
        // この要求を渡す前に見張りが止めていた worker には渡さない（判定していない要求に上限超過を付けない）。
        if slot.limit_killed {
            return Asked::NotDelivered;
        }
        // パイプの書き込みには期限が無いので、見張りにも止めさせる（要求の書き込みで止まった場合の保険。
        // 応答待ちの期限切れは下の `recv_timeout` が先に拾うよう、少し遅らせる）。
        slot.kill_at = Instant::now().checked_add(wait.saturating_add(WORKER_REPLY_SLACK));
    }
    if io
        .stdin
        .write_all(line)
        .and_then(|()| io.stdin.flush())
        .is_err()
    {
        return Asked::NotDelivered;
    }
    match io.lines.recv_timeout(wait) {
        Ok(answer) => {
            if let Some((response, retiring)) = classify_answer(&answer) {
                {
                    let mut slot = worker_slot();
                    slot.judged = true;
                    slot.kill_at = None;
                }
                if retiring {
                    log_line(format_args!("worker retired after a stalled judgement"));
                    reap_worker();
                }
                return Asked::Answer(response);
            }
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            log_line(format_args!("judge time limit exceeded; stopping worker"));
            reap_worker();
            return Asked::Answer(limit_response());
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {}
    }
    Asked::Answer(if reap_worker() {
        limit_response()
    } else {
        log_line(format_args!("worker stopped while judging"));
        error_response(&runtime("serve judge failed"))
    })
}

/// worker を作り直す。次のどれかなら理由をログに出して親ごと終える（壊れた状態で回り続けない）:
/// 起動の失敗（検証の失敗・時間切れ）・worker の版や同一性が起動時と違う・`ready` 直後の RSS が上限超・
/// 1 件も判定しないまま worker が終わる作り直しが [`MAX_BARREN_RESTARTS`] 回続いた。
fn respawn(ctx: &JudgeCtx, streak: &mut u32) -> WorkerIo {
    reap_worker();
    *streak = barren_streak(*streak, worker_slot().judged);
    if *streak >= MAX_BARREN_RESTARTS {
        log_line(format_args!(
            "worker stopped without judging {MAX_BARREN_RESTARTS} times in a row; exiting"
        ));
        exit_with(ExitCode::RuntimeError);
    }
    log_line(format_args!("worker stopped; restarting"));
    let wait = ctx.limits.time().get();
    let (reason, code) = match spawn_worker(&ctx.args, &ctx.cwd) {
        Ok(io) => match await_ready(&io, wait, Some(&ctx.identity)) {
            Startup::Ready(_) => {
                settle_worker(ctx.limits, false);
                log_line(format_args!("worker ready"));
                return io;
            }
            Startup::Failed(code, _) => ("worker failed to load the package", code),
            Startup::TimedOut => (
                "worker startup time limit exceeded",
                ExitCode::LimitExceeded,
            ),
            Startup::VersionChanged => ("worker version differs", ExitCode::RuntimeError),
            Startup::PackageChanged => ("package changed since startup", ExitCode::RuntimeError),
            Startup::Broken => ("worker failed to start", ExitCode::RuntimeError),
        },
        Err(_) => ("cannot start worker", ExitCode::RuntimeError),
    };
    log_line(format_args!("{reason}; exiting"));
    exit_with(code)
}

// ───────────────────────── 見張り ─────────────────────────

/// 上限超過・計測失敗で親を終える。起動中（ソケットを作る前）なら、その終了コードの既定のエラー JSON を
/// stdout に 1 回だけ出す（先に取った側だけが書く。`STDOUT_CLAIMED`）。先に別のスレッドが取っていたら、
/// そちらが終えるのを待つ（JSON は 1 つだけ。終了コードは先に取った側のもの）。
fn fatal_exit(code: ExitCode, msg: &'static str, startup: bool) -> ! {
    if !begin_exit() {
        wait_for_exit();
    }
    log_line(format_args!("{msg}"));
    if startup {
        if !claim_stdout() {
            wait_for_exit();
        }
        if arm_exit_deadline(code) {
            let report = ErrorReport::new(code, default_message(code));
            let _ = write_error_report(&mut io::stdout().lock(), &report);
        }
    }
    finish_exit(code)
}

/// worker の RSS を見た後の動き。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerAction {
    Keep,
    /// worker だけ止める（判定スレッドが作り直す）。
    Kill,
    /// worker を止めて親も終える（exit 20）。
    KillAndExit,
}

/// worker の RSS が上限を超えたときの分岐（REQ-39・#709）。`ready` の直後の計測で上限内と確かめた
/// （`settled`）後の超過は、使ううちに膨らんだとみて worker だけ止める。確かめる前（起動中）の超過は、
/// 作り直しても同じなので親ごと終える。`rss` の `None` は観測できない（終了済みで未回収など）で、何もしない。
fn worker_rss_action(rss: Option<u64>, limit: u64, settled: bool) -> WorkerAction {
    match rss {
        Some(rss) if rss > limit => {
            if settled {
                WorkerAction::Kill
            } else {
                WorkerAction::KillAndExit
            }
        }
        _ => WorkerAction::Keep,
    }
}

/// 見張りの 1 回分。親自身の RSS → worker の RSS と 1 件の期限、の順に見る。
/// - 親の RSS が `limits.memory()` 超: exit 20。読めなければ fail-closed で exit 70
/// - worker の RSS が上限超: [`worker_rss_action`]。読めなければ fail-closed で exit 70
/// - 判定中の 1 件が期限を過ぎた: worker を止める（判定スレッドが `limit_exceeded` を返して作り直す）
///
/// worker の計測は回収と同じロックの下で行う（回収済みの pid を計測しない）。親を終えるときは
/// ロックを離してから終える（終了処理が worker を止める）。
fn guard_tick(limits: SafetyLimits, startup: bool) {
    if EXITING.load(Ordering::SeqCst) {
        return;
    }
    match current_process_rss_bytes() {
        Ok(Some(rss)) if rss > limits.memory().get() => {
            fatal_exit(
                ExitCode::LimitExceeded,
                "memory limit exceeded; exiting",
                startup,
            );
        }
        Ok(_) => {}
        Err(_) => fatal_exit(
            ExitCode::RuntimeError,
            "memory probe failed; exiting",
            startup,
        ),
    }
    let mut slot = worker_slot();
    // ロックを待つ間に終了処理が始まっていたら（worker を回収している）、何もしない。
    if EXITING.load(Ordering::SeqCst) {
        return;
    }
    let Some(pid) = slot.child.as_ref().map(Child::id) else {
        return;
    };
    let action = match process_rss_bytes(pid) {
        Ok(rss) => worker_rss_action(rss, worker_rss_limit(limits), slot.settled),
        Err(_) => {
            drop(slot);
            fatal_exit(
                ExitCode::RuntimeError,
                "worker memory probe failed; exiting",
                startup,
            );
        }
    };
    let reason = match action {
        WorkerAction::KillAndExit => {
            drop(slot);
            fatal_exit(
                ExitCode::LimitExceeded,
                "worker memory limit exceeded right after loading; exiting",
                startup,
            );
        }
        WorkerAction::Kill => "worker memory limit exceeded; stopping worker",
        WorkerAction::Keep if slot.kill_at.is_some_and(|at| Instant::now() >= at) => {
            "worker did not answer in time; stopping worker"
        }
        WorkerAction::Keep => return,
    };
    if !slot.limit_killed {
        log_line(format_args!("{reason}"));
    }
    slot.limit_killed = true;
    if let Some(child) = slot.child.as_mut() {
        let _ = child.kill();
    }
}

/// 資源の見張りのスレッド（約 50 ms ごとに [`guard_tick`]。REQ-39・#708・#709）。[`SERVING`] が偽の間は起動中。
///
/// 起動の全体（このスレッドの起動＝置き場所・ロックに触る前から、待ち受けの開始まで）にも期限を掛ける。
/// `startup_budget` を過ぎても待ち受けに至らなければ、worker がいれば止めて、`limit_exceeded` の JSON を
/// stdout に 1 つ出して exit 20（ロックファイルの open など、worker の `ready` 待ち以外で止まる場合も覆う）。
fn spawn_guard_thread(limits: SafetyLimits) -> io::Result<()> {
    let started = Instant::now();
    let budget = startup_budget(limits);
    thread::Builder::new()
        .name("serve-guard".to_string())
        .spawn(move || {
            loop {
                thread::sleep(GUARD_POLL_INTERVAL);
                let startup = !SERVING.load(Ordering::SeqCst);
                if startup && started.elapsed() > budget {
                    fatal_exit(
                        ExitCode::LimitExceeded,
                        "startup time limit exceeded; exiting",
                        true,
                    );
                }
                guard_tick(limits, startup);
            }
        })
        .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39・#709: worker 自身の見張り。親の不在は最優先で exit 70、読み込み中の時間切れと RSS 超過は
    /// exit 20、RSS を読めなければ exit 70。上限ちょうど・読み込み後の経過時間では止めない。
    #[test]
    fn req39_worker_guard_verdict_branches() {
        let t = Duration::from_secs(10);
        let code = |v: Option<(ExitCode, &str)>| v.map(|(c, _)| c);
        let over = Some(Duration::from_secs(11));
        assert_eq!(
            worker_guard_verdict(7, 7, Some(t), t, Ok(Some(100)), 100),
            None
        );
        assert_eq!(worker_guard_verdict(7, 7, None, t, Ok(None), 100), None);
        assert_eq!(
            worker_guard_verdict(1, 7, over, t, Ok(Some(101)), 100),
            Some((ExitCode::RuntimeError, "parent is gone; worker exiting"))
        );
        assert_eq!(
            code(worker_guard_verdict(7, 7, over, t, Ok(Some(1)), 100)),
            Some(ExitCode::LimitExceeded)
        );
        assert_eq!(
            code(worker_guard_verdict(7, 7, None, t, Ok(Some(101)), 100)),
            Some(ExitCode::LimitExceeded)
        );
        assert_eq!(
            code(worker_guard_verdict(
                7,
                7,
                None,
                t,
                Err(io::Error::other("probe")),
                100
            )),
            Some(ExitCode::RuntimeError)
        );
    }

    /// #709: ソケットのファイル名の数字は `PROTOCOL_VERSION` と一致し、以前の名前とは重ならない。
    #[test]
    fn socket_file_name_carries_protocol_version() {
        use fandhe_edge_core::serve_protocol::PROTOCOL_VERSION;
        assert_eq!(SOCKET_FILE, format!("serve.v{PROTOCOL_VERSION}.sock"));
        assert_eq!(SOCKET_FILE, "serve.v1.sock");
        assert_eq!(LEGACY_SOCKET_FILES, ["serve.sock"]);
        assert!(!LEGACY_SOCKET_FILES.contains(&SOCKET_FILE));
    }

    /// REQ-39・#709: worker の回収は有界。kill の後に終われば成功、期限まで終わらなければ失敗、
    /// kill に失敗しても既に終わっていれば成功、kill に失敗して終わってもいなければ待たずに失敗。
    #[test]
    fn req39_kill_and_reap_is_bounded() {
        let ok = || Ok(());
        let failed = || Err(io::Error::other("kill failed"));
        let long = Duration::from_secs(5);
        // 3 回目の確認で回収できる。
        let mut polls = 0;
        let reaped = kill_and_reap(
            ok,
            || {
                polls += 1;
                Ok(polls >= 3)
            },
            long,
        );
        assert!(reaped);
        assert_eq!(polls, 3);
        // 終わらない worker は期限で諦める（無期限に待たない）。
        let started = Instant::now();
        assert!(!kill_and_reap(ok, || Ok(false), Duration::from_millis(50)));
        assert!(started.elapsed() >= Duration::from_millis(50));
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
        // kill の失敗: 既に終わっていれば成功、終わっていなければ待たずに失敗。
        assert!(kill_and_reap(failed, || Ok(true), long));
        let started = Instant::now();
        assert!(!kill_and_reap(failed, || Ok(false), long));
        assert!(started.elapsed() < Duration::from_secs(1));
        // 回収の確認自体の失敗は失敗。EINTR は続ける。
        assert!(!kill_and_reap(
            ok,
            || Err(io::Error::other("wait failed")),
            long
        ));
        let mut first = true;
        let reaped = kill_and_reap(
            ok,
            || {
                if std::mem::take(&mut first) {
                    Err(io::ErrorKind::Interrupted.into())
                } else {
                    Ok(true)
                }
            },
            long,
        );
        assert!(reaped);
    }

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

    /// REQ-39: パイプの 1 行読みは、改行込みで返し、EOF は `None`、上限超過・改行なしの切れ端は拒否する。
    #[test]
    fn read_pipe_line_respects_limit() {
        let mut input: &[u8] = b"abc\nde\n";
        assert_eq!(
            read_pipe_line(&mut input, 3).expect("line"),
            Some(b"abc\n".to_vec())
        );
        assert_eq!(
            read_pipe_line(&mut input, 3).expect("line"),
            Some(b"de\n".to_vec())
        );
        assert_eq!(read_pipe_line(&mut input, 3).expect("eof"), None);
        let mut long: &[u8] = b"abcd\n";
        let err = read_pipe_line(&mut long, 3).expect_err("too long");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let mut cut: &[u8] = b"ab";
        let err = read_pipe_line(&mut cut, 3).expect_err("no newline");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    /// REQ-39・#709: worker の RSS 超過は、`ready` 直後の計測で上限内と確かめた後なら worker だけ止め、
    /// 確かめる前（起動中）なら親ごと終える。上限ちょうど・観測できない（未回収の zombie）ときは何もしない。
    #[test]
    fn req39_worker_rss_action_branches() {
        assert_eq!(worker_rss_action(Some(101), 100, true), WorkerAction::Kill);
        assert_eq!(
            worker_rss_action(Some(101), 100, false),
            WorkerAction::KillAndExit
        );
        assert_eq!(worker_rss_action(Some(100), 100, true), WorkerAction::Keep);
        assert_eq!(worker_rss_action(Some(100), 100, false), WorkerAction::Keep);
        assert_eq!(worker_rss_action(None, 100, true), WorkerAction::Keep);
        assert_eq!(worker_rss_action(None, 0, false), WorkerAction::Keep);
    }

    fn identity(c: char) -> PackageIdentity {
        let h = c.to_string().repeat(64);
        PackageIdentity {
            onnx_sha256: h.clone(),
            shard_sha256: vec![],
            artifact_sha256: h.clone(),
            definition_sha256: h.clone(),
            calibration_sha256: None,
            layout_sha256: h,
        }
    }

    /// REQ-39・#709: worker の最初の 1 行。版の違う `ready`（バイナリの差し替え）と、起動時と同一性の違う
    /// `ready`（置き場所の中身の変化）は使わない。読み込みの失敗は終了コードと stdout をそのまま運ぶ。
    #[test]
    fn req39_first_line_rejects_other_version_and_changed_package() {
        let ready = |c: char, v: &str| {
            encode_response(&ServeResponse::Ready {
                package: identity(c),
                cli_version: v.into(),
            })
        };
        let a = identity('a');
        assert_eq!(
            classify_first_line(&ready('a', "1.0.0"), "1.0.0", None),
            Startup::Ready(identity('a'))
        );
        assert_eq!(
            classify_first_line(&ready('a', "1.0.0"), "1.0.0", Some(&a)),
            Startup::Ready(identity('a'))
        );
        assert_eq!(
            classify_first_line(&ready('a', "1.0.1"), "1.0.0", Some(&a)),
            Startup::VersionChanged
        );
        assert_eq!(
            classify_first_line(&ready('a', "1.0.1"), "1.0.0", None),
            Startup::VersionChanged
        );
        assert_eq!(
            classify_first_line(&ready('b', "1.0.0"), "1.0.0", Some(&a)),
            Startup::PackageChanged
        );
        let failed = encode_response(&ServeResponse::Ok {
            exit_code: 64,
            stdout: "e\n".into(),
        });
        assert_eq!(
            classify_first_line(&failed, "1.0.0", None),
            Startup::Failed(ExitCode::InvalidInput, "e\n".into())
        );
        let ok0 = encode_response(&ServeResponse::Ok {
            exit_code: 0,
            stdout: String::new(),
        });
        assert_eq!(classify_first_line(&ok0, "1.0.0", None), Startup::Broken);
        assert_eq!(
            classify_first_line(b"junk\n", "1.0.0", None),
            Startup::Broken
        );
        let retiring = encode_response(&ServeResponse::OkRetiring {
            exit_code: 20,
            stdout: String::new(),
        });
        assert_eq!(
            classify_first_line(&retiring, "1.0.0", None),
            Startup::Broken
        );
    }

    /// REQ-21・#709: worker の応答。`ok_retiring` はクライアントへ `ok` として返し、worker の終了を伝える。
    /// `ok`・`ok_retiring` 以外の行は不正。
    #[test]
    fn req21_answer_maps_retiring_to_ok() {
        let ok = ServeResponse::Ok {
            exit_code: 20,
            stdout: "x\n".into(),
        };
        assert_eq!(
            classify_answer(&encode_response(&ok)),
            Some((ok.clone(), false))
        );
        let retiring = ServeResponse::OkRetiring {
            exit_code: 20,
            stdout: "x\n".into(),
        };
        assert_eq!(
            classify_answer(&encode_response(&retiring)),
            Some((ok, true))
        );
        for other in [
            ServeResponse::PackageMismatch,
            ServeResponse::VersionMismatch,
            ServeResponse::Ready {
                package: identity('a'),
                cli_version: "1.0.0".into(),
            },
        ] {
            assert_eq!(classify_answer(&encode_response(&other)), None);
        }
        assert_eq!(classify_answer(b"junk\n"), None);
    }

    /// REQ-39・#709: 1 件も判定しないまま worker が終わる作り直しが 3 回続いたら上限に達する。
    /// 1 件でも判定していれば数え直す。
    #[test]
    fn req39_barren_streak_counts_consecutive_restarts() {
        assert_eq!(MAX_BARREN_RESTARTS, 3);
        assert_eq!(barren_streak(0, false), 1);
        assert_eq!(barren_streak(1, false), 2);
        assert_eq!(barren_streak(2, false), 3);
        assert_eq!(barren_streak(2, true), 0);
        assert_eq!(barren_streak(0, true), 0);
        assert_eq!(barren_streak(u32::MAX, false), u32::MAX);
    }
}
