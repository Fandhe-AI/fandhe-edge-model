//! `serve --stop`: 動いている常駐（[`crate::serve`]）を止める側（#728。親 #726。REQ-21・REQ-33・REQ-38・REQ-39）。
//!
//! # 役割と呼び出し文脈
//!
//! `main.rs` が `serve --stop` を受けると [`run_stop`] を呼ぶ。置き場所のソケットへ停止要求
//! （`fandhe_edge_core::serve_protocol::encode_stop_request`）を 1 行送り、常駐がプロセスごと終わる（接続が
//! 閉じる）のを最大 [`STOP_WAIT`] 待つ。利用側のセッション終了 hook などから呼ばれることを想定する。
//!
//! # 出力と終了コード（1 呼び出し 1 JSON）
//!
//! - 止まった: `{"status":"stopped"}`・exit 0
//! - 常駐が居ない（置き場所・ソケットが無い、ソケットが残っているだけで接続を断られた）:
//!   `{"status":"not_running"}`・exit 0
//! - 間に合わなかった: `limit_exceeded`・exit 20
//! - 置き場所が決まらない: `invalid_input`・exit 64。要求を受け付けなかった（停止要求を知らない旧い常駐など）・
//!   不正な応答・I/O 失敗: `runtime_error`・exit 70（メッセージは固定文。パス・応答の値は含めない）
//!
//! # 安全性（REQ-39）
//!
//! 停止は 0700 の置き場所の中の 0600 のソケット経由に限り、pid を使った kill はしない。接続の前に、置き場所が
//! 0700 の実ディレクトリ（symlink でない）で、ソケットが実ソケットであることを確かめる（検証してから操作する）。
//! 置き場所は作らない。待ちはすべて期限つき。
//!
//! # 限界
//!
//! 版の違う常駐はソケット名が違う（`serve.v<版>.sock`）ため見えず、`not_running` になる。停止要求を知らない
//! 旧い常駐は `runtime_error` になるので、手で止める。

use std::convert::Infallible;
use std::io::{self, Write};
use std::os::unix::fs::FileTypeExt;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::serve_protocol::{
    MAX_RESPONSE_LINE_BYTES, ServeResponse, decode_response, encode_stop_request,
};

use crate::error_report::{default_message, emit_error_report};
use crate::output::write_stage_line;
use crate::serve::{SOCKET_FILE, is_private_dir, read_line_bounded, serve_dir};

/// 停止要求を送ってから、常駐が終わるのを待つ上限（#728）。
pub const STOP_WAIT: Duration = Duration::from_secs(1);

/// 停止要求の結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Stopped,
    NotRunning,
    TimedOut,
    Failed,
}

/// `serve --stop` を実行して結果の JSON を `out` へ書く。
///
/// # Errors
/// `out` への書き込み失敗（呼び出し側は exit 70 に写す）。
pub fn run_stop<W: Write>(out: &mut W) -> io::Result<ExitCode> {
    let Some(dir) = serve_dir() else {
        return emit_error_report(
            out,
            &ErrorReport::new(ExitCode::InvalidInput, "serve directory is not configured"),
        );
    };
    write_outcome(out, request_stop(&dir))
}

fn write_outcome<W: Write>(out: &mut W, outcome: Outcome) -> io::Result<ExitCode> {
    let status = match outcome {
        Outcome::Stopped => "stopped",
        Outcome::NotRunning => "not_running",
        Outcome::TimedOut => {
            return emit_error_report(
                out,
                &ErrorReport::new(
                    ExitCode::LimitExceeded,
                    default_message(ExitCode::LimitExceeded),
                ),
            );
        }
        Outcome::Failed => {
            return emit_error_report(
                out,
                &ErrorReport::new(
                    ExitCode::RuntimeError,
                    "serve did not accept the stop request",
                ),
            );
        }
    };
    write_stage_line(
        out,
        Ok::<_, Infallible>(format!("{{\"status\":\"{status}\"}}")),
    )
}

fn is_timeout(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    )
}

/// ブロッキング処理 `f` を別スレッドで走らせ、`deadline` までに終わらなければ `None`（REQ-39）。
///
/// `UnixStream::connect` は接続キュー満杯時に期限なく待ちうるため、この形で期限を保証する。
/// 超過した処理のスレッドは放置する（`serve --stop` は直後にプロセスが終わる）。
fn within<T: Send + 'static>(
    deadline: Instant,
    f: impl FnOnce() -> T + Send + 'static,
) -> Option<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .ok()
}

/// 置き場所 `dir` の常駐へ停止要求を送り、終わるのを待つ。
fn request_stop(dir: &Path) -> Outcome {
    let sock = dir.join(SOCKET_FILE);
    if !std::fs::symlink_metadata(dir).is_ok_and(|m| is_private_dir(&m))
        || !std::fs::symlink_metadata(&sock).is_ok_and(|m| m.file_type().is_socket())
    {
        return Outcome::NotRunning;
    }
    let deadline = Instant::now() + STOP_WAIT;
    let sock_owned = sock.clone();
    let mut stream = match within(deadline, move || UnixStream::connect(sock_owned)) {
        None => return Outcome::TimedOut,
        Some(Ok(s)) => s,
        // 持ち主の居ない残ったソケット。
        Some(Err(e))
            if matches!(
                e.kind(),
                io::ErrorKind::ConnectionRefused | io::ErrorKind::NotFound
            ) =>
        {
            return Outcome::NotRunning;
        }
        Some(Err(_)) => return Outcome::Failed,
    };
    let remaining = deadline.saturating_duration_since(Instant::now());
    if stream.set_write_timeout(Some(remaining)).is_err() {
        return Outcome::Failed;
    }
    if let Err(e) = stream.write_all(&encode_stop_request()) {
        return if is_timeout(&e) {
            Outcome::TimedOut
        } else {
            Outcome::Failed
        };
    }
    // 応答 1 行（`stopped`）。旧い常駐の `version_mismatch` などはここで失敗として扱う。
    match read_line_bounded(&mut stream, MAX_RESPONSE_LINE_BYTES, deadline) {
        Ok(line) if decode_response(&line) == Ok(ServeResponse::Stopped) => {}
        Err(e) if is_timeout(&e) => return Outcome::TimedOut,
        _ => return Outcome::Failed,
    }
    // 常駐のプロセスが終わると接続が閉じる（EOF）。
    match read_line_bounded(&mut stream, MAX_RESPONSE_LINE_BYTES, deadline) {
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::UnexpectedEof | io::ErrorKind::ConnectionReset
            ) =>
        {
            Outcome::Stopped
        }
        Err(e) if is_timeout(&e) => Outcome::TimedOut,
        _ => Outcome::Failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixListener;

    /// 0700 の短い置き場所（ソケットのパス長の上限を避けるため `/tmp` 直下）。
    fn private_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::path::PathBuf::from(format!("/tmp/fe-stop-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        dir
    }

    /// REQ-33・#728: 置き場所が無い・ソケットが無い・持ち主の居ない残ったソケットは `not_running`。
    #[test]
    fn req33_no_serve_is_not_running() {
        let missing =
            std::path::PathBuf::from(format!("/tmp/fe-stop-{}-missing", std::process::id()));
        assert_eq!(request_stop(&missing), Outcome::NotRunning);
        assert!(!missing.exists());
        let dir = private_dir("none");
        assert_eq!(request_stop(&dir), Outcome::NotRunning);
        drop(UnixListener::bind(dir.join(SOCKET_FILE)).unwrap());
        assert_eq!(request_stop(&dir), Outcome::NotRunning);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-39・#728: 接続が停滞しても期限で打ち切る（`connect` は `within` 経由で、超過は exit 20 の `TimedOut`）。
    #[test]
    fn req39_stalled_blocking_call_times_out() {
        let t = Instant::now();
        let r = within(t + Duration::from_millis(100), || {
            std::thread::sleep(Duration::from_secs(5));
        });
        assert!(r.is_none());
        assert!(t.elapsed() < Duration::from_secs(2));
        assert_eq!(within(Instant::now() + STOP_WAIT, || 7), Some(7));
    }

    /// REQ-33・#728: 出力は固定の 1 行 JSON で、終了コードは 0・20・70。
    #[test]
    fn req33_outcome_output_and_exit_codes() {
        let run = |o| {
            let mut buf = Vec::new();
            let code = write_outcome(&mut buf, o).unwrap();
            (code, String::from_utf8(buf).unwrap())
        };
        assert_eq!(
            run(Outcome::Stopped),
            (ExitCode::Ok, "{\"status\":\"stopped\"}\n".to_string())
        );
        assert_eq!(
            run(Outcome::NotRunning),
            (ExitCode::Ok, "{\"status\":\"not_running\"}\n".to_string())
        );
        assert_eq!(run(Outcome::TimedOut).0, ExitCode::LimitExceeded);
        assert_eq!(run(Outcome::Failed).0, ExitCode::RuntimeError);
    }
}
