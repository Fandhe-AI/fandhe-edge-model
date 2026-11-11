//! `infer --text` を常駐プロセス（[`crate::serve`]）へ委ねる側（#708・#709。REQ-21・REQ-33・REQ-38・REQ-39）。
//!
//! # 呼び出し文脈
//!
//! 監督下の worker の中で、`stages::infer::run` が `prepare` の前に [`try_delegate`] を呼ぶ。軽い検査・ハッシュ・
//! ソケットの往復・stdout の書き込みが、監督の時間・RSS の上限と出力の見張りの下に入る。
//! 委ねられなければ [`Delegation::Standalone`]（常駐がいる環境。理由を stderr に出す）または
//! [`Delegation::NotInUse`]（ソケットが無い。無言）を返し、呼び出し側が今までの経路（監督つきの子 worker）へ進む。
//!
//! # 委ねる条件
//!
//! ソケットのファイル名は 1 行 JSON の形の版ごとに分かれている（`serve::SOCKET_FILE`）。版の違う常駐の
//! ソケットは見ない（ソケットが無い扱いで、何も言わず単独で判定する。#709）。
//!
//! 置き場所が 0700 でソケットがある・`--text` のみ（`--id`・`--input-file`・`--out`・`--version-ledger` なし）・`FANDHE_EDGE_NO_SERVE` 未設定・
//! ソケットにつながり、軽い検査（[`check_infer_path_and_format`]。閉じ込め・`artifact.json`・
//! 形式・`kind_version`。重い shard の検証は常駐の起動時に済み）を通り、`artifact.json` の記録が実体の
//! ONNX と一致する。同一性は `artifact.json`・`definition.json`・校正ファイルのバイト列の sha256 も含む。軽い検査の失敗は委ねずに単独へ戻す（エラーの出し方を単独と同じに保つ）。
//!
//! # つながった後
//!
//! 要求には 1 行 JSON の形の版（`PROTOCOL_VERSION`）と自分の版（`CARGO_PKG_VERSION`）を入れる（#709）。
//! 常駐が判定した後は単独へ戻らない（7B の読み込み直しで負荷が二重になるため）。時間切れは `limit_exceeded`（20）、
//! 応答の不正・切断は `runtime_error`（70）を単独と同じ形のエラー JSON で出す。時間は監督の `infer` の
//! 親の上限より `SUPERVISOR_GRACE` 短い期限。応答の `stdout` は加工せず流し、`exit_code` で終える。
//!
//! 単独へ戻るのは、常駐が判定していないと分かる場合だけ: `package_mismatch`・`version_mismatch`（版の違う
//! 常駐）・応答の前に常駐が終わった（EOF・切断）・要求を送り切れず応答も読めなかった。要求の書き込みが
//! 時間切れ以外で失敗しても、応答の 1 行が読めればそれを使う（常駐が接続数の上限で、要求を読まずに
//! `limit_exceeded` を返して閉じた場合。#709）。

use std::io::{self, Write};
use std::os::unix::fs::FileTypeExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fandhe_edge_core::artifact_meta::ArtifactMeta;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES;
use fandhe_edge_core::serve_protocol::{
    MAX_REQUEST_LINE_BYTES, MAX_RESPONSE_LINE_BYTES, PROTOCOL_VERSION, ServeRequest, ServeResponse,
    decode_response, encode_request,
};
use fandhe_edge_guard::resource::safety_limits_for_kind;

use crate::args::{InferArgs, InferSource};
use crate::error_report::{default_message, emit_error_report};
use crate::infer_batch::{
    MAX_INFER_BATCH_OUTPUT_DURATION, OutputWatchdog, SINGLE_HARD_LIMIT_GRACE,
};
use crate::infer_guard::check_infer_path_and_format;
use crate::serve::{NO_SERVE_ENV, SOCKET_FILE, is_private_dir, read_line_bounded, serve_dir};
use crate::stages::infer::light_package_identity;

/// 委ねた結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delegation {
    /// 常駐が応えた（または、つながった後の失敗を単独と同じ形で出した）。この終了コードで終える。
    Done(ExitCode),
    /// 常駐がいる環境で、委ねられなかった。理由（固定語彙）を stderr に出して今の経路へ進む。
    Standalone(&'static str),
    /// 置き場所にソケットが無い（常駐を使っていない）。何も出さず今の経路へ進む。
    NotInUse,
}

/// 委ねる条件を満たすなら常駐へ要求を送り、応答を `out` へ中継する。
///
/// # Errors
/// `out` への書き込み失敗（呼び出し側は exit 70 に写す）。
pub fn try_delegate<W: Write>(out: &mut W, args: &InferArgs, cwd: &Path) -> io::Result<Delegation> {
    try_delegate_in(out, args, cwd, serve_dir())
}

/// [`try_delegate`] の置き場所を指定できる版（環境変数を書き換えずに試験するため）。
///
/// # Errors
/// [`try_delegate`] と同じ。
pub fn try_delegate_in<W: Write>(
    out: &mut W,
    args: &InferArgs,
    cwd: &Path,
    dir: Option<PathBuf>,
) -> io::Result<Delegation> {
    // 常駐を使っていない利用者の stderr・挙動を変えないよう、ソケットが無ければ何も言わず単独へ進む
    // （ONNX を読む軽い検査より先に見て、常駐がいない普通の場合の追加コストも避ける）。
    let Some(dir) = dir else {
        return Ok(Delegation::NotInUse);
    };
    let sock = dir.join(SOCKET_FILE);
    if !std::fs::symlink_metadata(&dir).is_ok_and(|m| is_private_dir(&m))
        || !std::fs::symlink_metadata(&sock).is_ok_and(|m| m.file_type().is_socket())
    {
        return Ok(Delegation::NotInUse);
    }
    let text = match (&args.source, &args.version) {
        (InferSource::Text { id: Some(_), .. }, _) => {
            return Ok(Delegation::Standalone("--id is given"));
        }
        (InferSource::Text { text, id: None }, None) => text,
        (InferSource::Text { .. }, Some(_)) => {
            return Ok(Delegation::Standalone("--version-ledger is given"));
        }
        (InferSource::InputFile { .. }, _) => {
            return Ok(Delegation::Standalone("input is not --text"));
        }
    };
    if std::env::var_os(NO_SERVE_ENV).is_some() {
        return Ok(Delegation::Standalone("disabled by FANDHE_EDGE_NO_SERVE"));
    }

    // (3) 入力の検査は委ねる前に。単独と同じエラー（上限超過など）を単独の経路に出させる。
    if text.len() > MAX_INFER_INPUT_BYTES {
        return Ok(Delegation::Standalone("input is too large"));
    }
    let started = Instant::now();

    // (a) 試しの接続。つながらなければ ONNX を読まずに単独へ進む。
    if UnixStream::connect(&sock).is_err() {
        return Ok(Delegation::Standalone("cannot connect"));
    }

    // (b) 軽い検査。失敗は単独へ戻して、単独と同じエラーを出させる。
    let Ok(checked) = check_infer_path_and_format(cwd, args) else {
        return Ok(Delegation::Standalone("package check failed"));
    };
    let Ok(meta) = ArtifactMeta::parse(&checked.meta_bytes) else {
        return Ok(Delegation::Standalone("package check failed"));
    };
    if meta.onnx_sha256() != Sha256Digest::of_bytes(checked.onnx.as_bytes()).to_hex() {
        return Ok(Delegation::Standalone("package check failed"));
    }
    let Ok(identity) = light_package_identity(&checked, &meta) else {
        return Ok(Delegation::Standalone("package check failed"));
    };
    // 期限は監督の上限（`time + SINGLE_HARD_LIMIT_GRACE + SUPERVISOR_GRACE`）より SUPERVISOR_GRACE だけ短い。
    // 監督に kill される前に、自分で `limit_exceeded` を出せる。検査に使った時間も含める。
    let deadline = started
        + safety_limits_for_kind(checked.kind.as_str())
            .time()
            .get()
            .saturating_add(SINGLE_HARD_LIMIT_GRACE);

    let request = encode_request(&ServeRequest {
        protocol: PROTOCOL_VERSION,
        cli_version: env!("CARGO_PKG_VERSION").to_string(),
        id: request_id(),
        text: text.clone(),
        package: identity,
    });
    if request.len() > MAX_REQUEST_LINE_BYTES.saturating_add(1) {
        return Ok(Delegation::Standalone("request is too large"));
    }

    // (c) 要求用に接続し直す。ここでつながらなければまだ何も送っていないので単独へ進める。
    let Ok(mut stream) = UnixStream::connect(&sock) else {
        return Ok(Delegation::Standalone("cannot connect"));
    };

    // 送った後は、時間切れ以外では単独へ戻らない。ただし応答の前に常駐が終わった（EOF・切断）なら
    // 読み込みが二重になる心配がないので単独へ戻す。
    let remaining = deadline.saturating_duration_since(Instant::now());
    let armed = if remaining.is_zero() {
        Err(io::ErrorKind::TimedOut.into())
    } else {
        stream.set_write_timeout(Some(remaining))
    };
    let written = armed.and_then(|()| stream.write_all(&request));
    let read = read_line_bounded(&mut stream, MAX_RESPONSE_LINE_BYTES, deadline);
    let line = match (written, read) {
        (Err(e), _) if is_timeout(&e) => return failed(out, &e),
        // 書き込みが時間切れ以外で失敗しても、常駐が要求を読まずに応答して閉じた（接続数の上限の
        // `limit_exceeded`）場合があるので、応答が読めればそれを使う（#709）。
        (_, Ok(line)) => line,
        // 要求を送り切れていないので常駐は判定しておらず、単独へ戻しても読み込みは二重にならない。
        (Err(_), Err(_)) => return Ok(Delegation::Standalone("serve closed the connection")),
        (Ok(()), Err(e)) => return failed(out, &e),
    };
    match decode_response(&line) {
        Ok(ServeResponse::PackageMismatch) => Ok(Delegation::Standalone("package mismatch")),
        // 版の違う常駐には委ねない（#709）。常駐は判定していないので単独へ戻れる。
        Ok(ServeResponse::VersionMismatch) => Ok(Delegation::Standalone("version mismatch")),
        Ok(ServeResponse::Ok { exit_code, stdout }) => {
            match ExitCode::try_from(i32::from(exit_code)) {
                Ok(code) => {
                    // 出力の書き込みの見張りは、応答を受け取った後・書く直前に始める（stdout が詰まっても
                    // 止まらない。書き込み停止は exit 20）。待っている間の時間は数えない。
                    let _watchdog = OutputWatchdog::arm(true, MAX_INFER_BATCH_OUTPUT_DURATION)?;
                    out.write_all(stdout.as_bytes())?;
                    out.flush()?;
                    Ok(Delegation::Done(code))
                }
                Err(_) => runtime_failed(out),
            }
        }
        // `ready`・`ok_retiring` は常駐の親と worker 子の間専用で、ソケットに流れてきたら不正な応答。
        Ok(ServeResponse::Ready { .. } | ServeResponse::OkRetiring { .. }) | Err(_) => {
            runtime_failed(out)
        }
    }
}

/// つながった後の I/O 失敗を単独と同じ形のエラー JSON にする（時間切れは `limit_exceeded`）。
fn failed<W: Write>(out: &mut W, e: &io::Error) -> io::Result<Delegation> {
    // 応答の前に常駐が終わった（EOF・切断）。単独へ戻す。
    if matches!(
        e.kind(),
        io::ErrorKind::UnexpectedEof
            | io::ErrorKind::BrokenPipe
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
    ) {
        return Ok(Delegation::Standalone("serve closed the connection"));
    }
    if is_timeout(e) {
        let report = ErrorReport::new(
            ExitCode::LimitExceeded,
            default_message(ExitCode::LimitExceeded),
        );
        let _watchdog = OutputWatchdog::arm(true, MAX_INFER_BATCH_OUTPUT_DURATION)?;
        return emit_error_report(out, &report).map(Delegation::Done);
    }
    runtime_failed(out)
}

fn is_timeout(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    )
}

fn runtime_failed<W: Write>(out: &mut W) -> io::Result<Delegation> {
    let report = ErrorReport::new(
        ExitCode::RuntimeError,
        default_message(ExitCode::RuntimeError),
    );
    let _watchdog = OutputWatchdog::arm(true, MAX_INFER_BATCH_OUTPUT_DURATION)?;
    emit_error_report(out, &report).map(Delegation::Done)
}

/// pid と時刻から作る要求 id（英数字と `-` のみ。64 バイト以内）。
fn request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{}-{nanos}", std::process::id())
}
