//! `infer --text` を常駐プロセス（[`crate::serve`]）へ委ねる側（#708。REQ-21・REQ-33・REQ-38・REQ-39）。
//!
//! # 呼び出し文脈
//!
//! `main.rs` が `infer` を受け、監督（[`crate::infer_supervisor`]）に入る前に [`try_delegate`] を呼ぶ。
//! 委ねられなければ [`Delegation::Standalone`]（常駐がいる環境。理由を stderr に出す）または
//! [`Delegation::NotInUse`]（ソケットが無い。無言）を返し、呼び出し側が今までの経路（監督つきの子 worker）へ進む。
//!
//! # 委ねる条件
//!
//! 置き場所が 0700 でソケットがある・`--text` のみ（`--id`・`--input-file`・`--out`・`--version-ledger` なし）・`FANDHE_EDGE_NO_SERVE` 未設定・
//! ソケットにつながり、軽い検査（[`check_infer_path_and_format`]。閉じ込め・`artifact.json`・
//! 形式・`kind_version`。重い shard の検証は常駐の起動時に済み）を通り、`artifact.json` の記録が実体の
//! ONNX と一致する。同一性は `artifact.json`・`definition.json`・校正ファイルのバイト列の sha256 も含む。軽い検査の失敗は委ねずに単独へ戻す（エラーの出し方を単独と同じに保つ）。
//!
//! # つながった後
//!
//! 単独へは戻らない（7B の読み込み直しで負荷が二重になるため）。時間切れは `limit_exceeded`（20）、
//! 応答の不正・切断は `runtime_error`（70）を単独と同じ形のエラー JSON で出す。時間は監督の `infer` の
//! 親の上限と同じ値。`package_mismatch` のときだけ単独へ戻る。応答の `stdout` は加工せず流し、
//! `exit_code` で終える。

use std::io::{self, Write};
use std::os::unix::fs::FileTypeExt;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fandhe_edge_core::artifact_meta::ArtifactMeta;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::serve_protocol::{
    MAX_REQUEST_LINE_BYTES, MAX_RESPONSE_LINE_BYTES, ServeRequest, ServeResponse, decode_response,
    encode_request,
};
use fandhe_edge_guard::resource::safety_limits_for_kind;

use crate::args::{InferArgs, InferSource};
use crate::error_report::{default_message, emit_error_report};
use crate::infer_batch::SINGLE_HARD_LIMIT_GRACE;
use crate::infer_guard::check_infer_path_and_format;
use crate::infer_supervisor::SUPERVISOR_GRACE;
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
    // 常駐を使っていない利用者の stderr・挙動を変えないよう、ソケットが無ければ何も言わず単独へ進む
    // （ONNX を読む軽い検査より先に見て、常駐がいない普通の場合の追加コストも避ける）。
    let Some(dir) = serve_dir() else {
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

    // 先につなぐ。つながらなければ ONNX を読まずに単独へ進む（つなぐのは軽いので、検査の失敗時は
    // 要求を送らずに stream を drop するだけ）。
    let Ok(mut stream) = UnixStream::connect(&sock) else {
        return Ok(Delegation::Standalone("cannot connect"));
    };

    // 軽い検査。失敗は単独へ戻して、単独と同じエラーを出させる。
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
    let limits = safety_limits_for_kind(checked.kind.as_str());
    let timeout = limits
        .time()
        .get()
        .saturating_add(SINGLE_HARD_LIMIT_GRACE)
        .saturating_add(SUPERVISOR_GRACE);

    let request = encode_request(&ServeRequest {
        id: request_id(),
        text: text.clone(),
        package: identity,
    });
    if request.len() > MAX_REQUEST_LINE_BYTES.saturating_add(1) {
        return Ok(Delegation::Standalone("request is too large"));
    }

    // つながった後は単独へ戻らない。
    let deadline = Instant::now() + timeout;
    let line = stream
        .set_write_timeout(Some(timeout))
        .and_then(|()| stream.write_all(&request))
        .and_then(|()| read_line_bounded(&mut stream, MAX_RESPONSE_LINE_BYTES, deadline));
    let line = match line {
        Ok(line) => line,
        Err(e) => return failed(out, &e),
    };
    match decode_response(&line) {
        Ok(ServeResponse::PackageMismatch) => Ok(Delegation::Standalone("package mismatch")),
        Ok(ServeResponse::Ok { exit_code, stdout }) => {
            match ExitCode::try_from(i32::from(exit_code)) {
                Ok(code) => {
                    out.write_all(stdout.as_bytes())?;
                    out.flush()?;
                    Ok(Delegation::Done(code))
                }
                Err(_) => runtime_failed(out),
            }
        }
        Err(_) => runtime_failed(out),
    }
}

/// つながった後の I/O 失敗を単独と同じ形のエラー JSON にする（時間切れは `limit_exceeded`）。
fn failed<W: Write>(out: &mut W, e: &io::Error) -> io::Result<Delegation> {
    if matches!(
        e.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    ) {
        let report = ErrorReport::new(
            ExitCode::LimitExceeded,
            default_message(ExitCode::LimitExceeded),
        );
        return emit_error_report(out, &report).map(Delegation::Done);
    }
    runtime_failed(out)
}

fn runtime_failed<W: Write>(out: &mut W) -> io::Result<Delegation> {
    let report = ErrorReport::new(
        ExitCode::RuntimeError,
        default_message(ExitCode::RuntimeError),
    );
    emit_error_report(out, &report).map(Delegation::Done)
}

/// pid と時刻から作る要求 id（英数字と `-` のみ。64 バイト以内）。
fn request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{}-{nanos}", std::process::id())
}
