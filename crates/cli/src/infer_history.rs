//! `infer --text` の判定履歴を手元の `history.jsonl` へ追記する（issue #711。REQ-33・REQ-38・REQ-39）。
//!
//! # 役割と呼び出し文脈
//!
//! `stages::infer::run` が薄く包んで呼ぶ。書く場所は `infer` のプロセス（常駐へ委ねる経路でも単独でも同じ
//! worker 内の `run`）の 1 か所だけで、常駐プロセス（`serve`）側では書かない。
//!
//! - 既定は無効。環境変数 `FANDHE_EDGE_HISTORY` が `"1"` と完全一致するときだけ有効で、それ以外は
//!   ファイルもディレクトリも作らない。**有効にすると入力の全文が手元のファイルに残る**
//! - 対象は `infer --text` のみ（`--id` 付きを含む）。`--input-file` は件数が多いため対象外
//! - 置き場所は `serve::serve_dir()`（既定 `$HOME/Library/Application Support/fandhe-edge/`、
//!   `FANDHE_EDGE_SERVE_DIR` で変わる）。無ければ 0700 で作り、あれば symlink でない 0700 の実ディレクトリの
//!   ときだけ使う。ファイルは 0600。`history.jsonl` が通常ファイルでない（symlink 等）なら書かない
//! - 1 行は [`fandhe_edge_core::infer_history`] の形。`output` は stdout へ出した JSON の先頭 64 KiB までの控え
//! - 追記は `history.lock` の排他（待ち続けず数回だけ試す）の下で、上限 [`MAX_HISTORY_BYTES`] を超えていたら
//!   `history.jsonl.1` へ回してから `O_APPEND` で 1 回の `write_all`
//! - 書き込みが途中で失敗したら、欠けた行が残らないよう元の長さへ戻す。回し（rename）の失敗は捨てて追記を続ける
//! - 開く処理は guard の `private_file`（`O_NOFOLLOW | O_NONBLOCK`・fd の `fstat` で通常ファイルかつ
//!   `mode & 0o077 == 0`）。検査と open の間に symlink・FIFO へ差し替えても追わず、止まらない
//! - `output` は読み直した JSON なので、キーの順・数値の表記は stdout のバイト列と一致しないことがある
//! - 限界: 置き場所の所有者が自分かどうかは確かめていない（std で自分の uid を取れない。
//!   `FANDHE_EDGE_SERVE_DIR` を他人の 0700 のディレクトリへ向けて root で実行した場合に書きうる）
//! - 限界: 監督が worker を止めた場合（時間・RSS の超過）は履歴が残らない
//! - 履歴に関する失敗は全部黙って捨てる。stdout・stderr・終了コードは有効でも無効でも変えない（REQ-33）
//! - 通信はしない（REQ-38）。消すには `history.jsonl*` を消すだけ

use std::fs::{self, File};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::infer_history::{HistoryRecord, encode, output_value};

use crate::args::{InferArgs, InferSource};
use fandhe_edge_guard::private_file::open_private_append;

use crate::serve::serve_dir;

/// 履歴を有効にする環境変数（値が `"1"` と完全一致するときだけ有効）。
pub const HISTORY_ENV: &str = "FANDHE_EDGE_HISTORY";
const HISTORY_FILE: &str = "history.jsonl";
const HISTORY_ROTATED: &str = "history.jsonl.1";
const LOCK_FILE: &str = "history.lock";
/// `output` 欄のために控える stdout の最大バイト数。
pub(crate) const CAPTURE_LIMIT: usize = 64 * 1024;
/// 履歴ファイルの上限。超えたら `.1` へ回す。
// ponytail: 1 世代だけの持ち回し。細かい削除が要れば見直す
const MAX_HISTORY_BYTES: u64 = 64 * 1024 * 1024;
const LOCK_TRIES: u32 = 5;
const LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(20);

/// 書き込みをそのまま `inner` へ通しつつ、先頭 [`CAPTURE_LIMIT`] バイトまでを控える。
pub(crate) struct Tee<'a, W: Write> {
    inner: &'a mut W,
    pub(crate) captured: Vec<u8>,
}

impl<'a, W: Write> Tee<'a, W> {
    pub(crate) fn new(inner: &'a mut W) -> Self {
        Self {
            inner,
            captured: Vec::new(),
        }
    }
}

impl<W: Write> Write for Tee<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        let room = CAPTURE_LIMIT.saturating_sub(self.captured.len());
        if let Some(head) = buf.get(..n.min(room)) {
            self.captured.extend_from_slice(head);
        }
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// 有効かつ `--text` のときだけ開始時刻を返す。
pub(crate) fn begin(args: &InferArgs) -> Option<Instant> {
    let enabled = std::env::var_os(HISTORY_ENV).is_some_and(|v| v == "1");
    (enabled && matches!(args.source, InferSource::Text { .. })).then(Instant::now)
}

/// 1 行を追記する。失敗は全部捨てる。
pub(crate) fn record(args: &InferArgs, started: Instant, captured: &[u8], code: ExitCode) {
    let InferSource::Text { text, .. } = &args.source else {
        return;
    };
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let rec = HistoryRecord {
        time,
        text: text.clone(),
        output: output_value(captured),
        exit_code: code.code(),
        elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        package: args.package.to_string_lossy().into_owned(),
        cli_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    if let Some(line) = encode(&rec) {
        let _ = append(&line);
    }
}

fn append(line: &[u8]) -> io::Result<()> {
    let dir = serve_dir().ok_or(io::ErrorKind::NotFound)?;
    // 無ければ 0700 で作る。あれば symlink でない 0700 の実ディレクトリかは `open_private_append` が fd で確かめる。
    if matches!(fs::symlink_metadata(&dir), Err(e) if e.kind() == io::ErrorKind::NotFound) {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&dir)?;
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
    }
    append_line(&dir, line, MAX_HISTORY_BYTES)
}

/// `dir` の `history.lock` の排他の下で、`history.jsonl` が `max` バイトを超えていれば `.1` へ回してから
/// `line` を 1 回の `write_all` で追記する。ロックは待ち続けず、取れなければ `WouldBlock`。
/// 回し（rename）の失敗は捨てて追記を続ける。
fn append_line(dir: &Path, line: &[u8], max: u64) -> io::Result<()> {
    let lock = open_private_append(dir, LOCK_FILE)?;
    let mut locked = false;
    for _ in 0..LOCK_TRIES {
        if lock.try_lock().is_ok() {
            locked = true;
            break;
        }
        std::thread::sleep(LOCK_RETRY_INTERVAL);
    }
    if !locked {
        return Err(io::ErrorKind::WouldBlock.into());
    }
    if fs::metadata(dir.join(HISTORY_FILE)).is_ok_and(|m| m.len() > max) {
        let _ = fs::rename(dir.join(HISTORY_FILE), dir.join(HISTORY_ROTATED));
    }
    let mut file = open_private_append(dir, HISTORY_FILE)?;
    let len = file.metadata()?.len();
    let written = file.write_all(line);
    restore_on_err(&file, len, written)
}

/// 書き込みが失敗したら、欠けた行が次の行とつながらないよう元の長さへ戻す（戻しの失敗は捨てる）。
fn restore_on_err(file: &File, len: u64, written: io::Result<()>) -> io::Result<()> {
    if written.is_err() {
        let _ = file.set_len(len);
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("fe-ih-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("mkdir");
        fs::set_permissions(&d, fs::Permissions::from_mode(0o700)).expect("chmod");
        d
    }

    fn read(d: &Path, n: &str) -> String {
        fs::read_to_string(d.join(n)).unwrap_or_default()
    }

    /// REQ-39・#711: 上限を超えたときだけ `.1` へ回り（境界は「超えた」）、3 回目で前の `.1` が置き換わって最古が消える。
    #[test]
    fn rotation_boundary_and_replaces_oldest() {
        let d = tmp("rot");
        append_line(&d, b"aaaa\n", 5).expect("1");
        // 長さ 5 は上限ちょうどなので回らない。
        append_line(&d, b"bbbb\n", 5).expect("2");
        assert_eq!(read(&d, HISTORY_FILE), "aaaa\nbbbb\n");
        assert_eq!(read(&d, HISTORY_ROTATED), "");
        // 10 > 5 で回る。
        append_line(&d, b"cccc\n", 5).expect("3");
        assert_eq!(read(&d, HISTORY_FILE), "cccc\n");
        assert_eq!(read(&d, HISTORY_ROTATED), "aaaa\nbbbb\n");
        append_line(&d, b"dddd\n", 5).expect("4");
        append_line(&d, b"eeee\n", 5).expect("5");
        assert_eq!(read(&d, HISTORY_FILE), "eeee\n");
        assert_eq!(read(&d, HISTORY_ROTATED), "cccc\ndddd\n");
        let _ = fs::remove_dir_all(&d);
    }

    /// #711: 書き込み失敗では、途中まで書かれた分を元の長さへ戻す。成功なら触らない。
    #[test]
    fn failed_write_restores_length() {
        let d = tmp("restore");
        let mut f = open_private_append(&d, HISTORY_FILE).expect("open");
        f.write_all(b"keep\n").expect("write");
        f.write_all(b"par").expect("partial");
        let err = io::Error::from(io::ErrorKind::WriteZero);
        assert!(restore_on_err(&f, 5, Err(err)).is_err());
        assert_eq!(read(&d, HISTORY_FILE), "keep\n");
        assert!(restore_on_err(&f, 0, Ok(())).is_ok());
        assert_eq!(read(&d, HISTORY_FILE), "keep\n");
        let _ = fs::remove_dir_all(&d);
    }

    /// #711: 控えは 64 KiB で打ち切るが、内側への書き込みは全部通る。
    #[test]
    fn tee_caps_capture_but_passes_everything() {
        let mut sink = Vec::new();
        let mut tee = Tee::new(&mut sink);
        let data = vec![b'x'; CAPTURE_LIMIT + 10];
        tee.write_all(&data).expect("write");
        tee.write_all(b"tail").expect("write");
        assert_eq!(tee.captured.len(), CAPTURE_LIMIT);
        assert_eq!(sink.len(), CAPTURE_LIMIT + 14);
    }
}
