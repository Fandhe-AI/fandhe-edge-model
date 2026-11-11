//! `infer --text` の判定履歴を手元の `history.jsonl` へ追記する（issue #711。REQ-33・REQ-38・REQ-39）。
//!
//! # 役割と呼び出し文脈
//!
//! `infer_supervisor::run`（利用者が起動した一番外側の `infer` プロセス）が薄く包んで呼ぶ。書く場所はここ 1 か所
//! だけで、監督下の worker・常駐プロセス（`serve`）側では書かない（worker には有効化の環境変数を渡さない）。
//! 結果を stdout へ出し終えた後に書くので、結果の出どころ（常駐へ委ねた・単独・監督が worker を止めた）に
//! 関わらず 1 件 1 行になる。
//!
//! - 保存は監督の時間・RSS の上限の外で行い、独立した上限 [`SAVE_LIMIT`] を持つ。別スレッドで書き、間に合わなければ
//!   待たずに戻る（プロセスの終了でスレッドは捨てる）。保存が停滞しても、出し終えた判定と終了コードは変わらない
//!
//! - 既定は無効。環境変数 `FANDHE_EDGE_HISTORY` が `"1"` と完全一致するときだけ有効で、それ以外は
//!   ファイルもディレクトリも作らない。**有効にすると入力の全文が手元のファイルに残る**
//! - 対象は `infer --text` のみ（`--id` 付きを含む）。`--input-file` は件数が多いため対象外
//! - 置き場所は `serve::serve_dir()`（既定 `$HOME/Library/Application Support/fandhe-edge/`、
//!   `FANDHE_EDGE_SERVE_DIR` で変わる）。無ければ 0700 で作り（作った後にパス指定で権限を直さない）、
//!   symlink でない 0700 の実ディレクトリで所有者が自分のときだけ使う。ファイルは 0600。`history.jsonl` が
//!   通常ファイルでない（symlink 等）・group / other の権限がある・所有者が自分でないなら書かない
//! - 1 行は [`fandhe_edge_core::infer_history`] の形。`output` は stdout へ出した JSON の先頭 64 KiB までの控え
//! - 追記は `history.lock` の排他（待ち続けず数回だけ試す）の下で、`history.jsonl` を開いて検査し、その fd の
//!   長さが上限 [`MAX_HISTORY_BYTES`] を超えていたら `history.jsonl.1` へ回して開き直してから `O_APPEND` で
//!   1 回の `write_all`。検査に落ちたファイル（symlink・権限が緩い等）は回さず、触らない
//! - 書き込みが途中で失敗したら、欠けた行が残らないよう元の長さへ戻す。回し（rename）に失敗したら書かない
//! - 開く・回す処理は guard の `private_file`（置き場所の fd 起点の `openat`・`renameat`。`O_NOFOLLOW | O_NONBLOCK`・
//!   fd の `fstat` で通常ファイルかつ `mode & 0o077 == 0` かつ所有者が自分）。検査と open の間に symlink・FIFO へ
//!   差し替えても追わず、止まらない。検査の後はパスを引き直さない
//! - `output` は読み直した JSON なので、キーの順・数値の表記は stdout のバイト列と一致しないことがある
//! - 所有者が他人の置き場所（`FANDHE_EDGE_SERVE_DIR` を他人の 0700 のディレクトリへ向けて root で実行した場合）
//!   には書かない。自分の uid は、置き場所に新規作成してすぐ消す空ファイルの所有者から得る（guard 側の doc）
//! - 監督が worker を止めた場合（時間・RSS の超過）も、外側が出した `limit_exceeded` の行が残る
//! - 限界: 保存を上限で打ち切った場合、その 1 行は残らないことがある。追記は 1 行 1 回の `write` なので通常は
//!   行が欠けて残らないが、`write` が一部だけ書いて戻った（ディスクの空きが尽きた等）直後に打ち切ると、
//!   長さを戻す前に終わって欠けた行が残りうる
//! - 履歴に関する失敗は全部黙って捨てる。stdout・stderr・終了コードは有効でも無効でも変えない（REQ-33）
//! - 通信はしない（REQ-38）。消すには `history.jsonl*` を消すだけ

use std::fs::{self, File};
use std::io::{self, Write};
use std::os::unix::fs::DirBuilderExt;
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::infer_history::{HistoryRecord, encode, output_value};

use crate::args::{InferArgs, InferSource};
use fandhe_edge_guard::private_file::PrivateDir;

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
/// 保存を待つ上限。通常の保存は数 ms、ロック待ちは最大 `LOCK_TRIES × LOCK_RETRY_INTERVAL` = 100 ms なので、
/// その数倍を取る。超えるのは置き場所が応答しないとき（止まったネットワーク上の置き場所等）だけで、
/// そのときは保存を捨てて `infer` を終える（REQ-33・REQ-39・#711）。
const SAVE_LIMIT: Duration = Duration::from_millis(500);

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

/// 1 行を追記する。失敗は全部捨て、[`SAVE_LIMIT`] を超えたら待たずに戻る。
/// 呼び出し側は、結果を stdout へ出し終えてから呼ぶ。
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
        bounded(SAVE_LIMIT, move || {
            let _ = append(&line);
        });
    }
}

/// `work` を別スレッドで走らせ、終わるか `limit` が過ぎるまでだけ待つ。過ぎたら `work` を置いて戻る
/// （スレッドはプロセスの終了で捨てる）。スレッドを起こせなければ何もしない。
fn bounded(limit: Duration, work: impl FnOnce() + Send + 'static) {
    let (done, wait) = mpsc::channel::<()>();
    let spawned = std::thread::Builder::new().spawn(move || {
        work();
        let _ = done.send(());
    });
    if spawned.is_ok() {
        // `work` が panic した場合は送り手が落ちて、すぐ戻る。
        let _ = wait.recv_timeout(limit);
    }
}

fn append(line: &[u8]) -> io::Result<()> {
    let dir = serve_dir().ok_or(io::ErrorKind::NotFound)?;
    // 無ければ 0700 で作る。作った後にパス指定で権限を直さない（検査と chmod の間に symlink へ差し替えられると
    // リンク先の権限を変えてしまう。REQ-39・#711）。umask で 0700 にならなければ、下の検査で書かない側に倒れる。
    // symlink でない 0700 の実ディレクトリで所有者が自分かは `PrivateDir::open` が fd で確かめる。
    if matches!(fs::symlink_metadata(&dir), Err(e) if e.kind() == io::ErrorKind::NotFound) {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&dir)?;
    }
    append_line(&dir, line, MAX_HISTORY_BYTES)
}

/// `dir` の `history.lock` の排他の下で、`history.jsonl` が `max` バイトを超えていれば `.1` へ回してから
/// `line` を 1 回の `write_all` で追記する。ロックは待ち続けず、取れなければ `WouldBlock`。
/// 回し（rename）に失敗したら書かない。
///
/// 回すかどうかは、検査に通って開けた fd の長さで決める（REQ-39・#711）。検査に落ちた `history.jsonl`
/// （symlink・権限が緩い・所有者が違う）は回さず、触らずに `Err` で戻る。
fn append_line(dir: &Path, line: &[u8], max: u64) -> io::Result<()> {
    let dir = PrivateDir::open(dir)?;
    let lock = dir.open_append(LOCK_FILE)?;
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
    let mut file = dir.open_append(HISTORY_FILE)?;
    let mut len = file.metadata()?.len();
    if len > max {
        // 回せないとき（`.1` がディレクトリなど）は書かない。追記を続けると上限なしに増える（REQ-39）。
        dir.rename(HISTORY_FILE, HISTORY_ROTATED)?;
        file = dir.open_append(HISTORY_FILE)?;
        len = file.metadata()?.len();
    }
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
    use std::os::unix::fs::{PermissionsExt, symlink};

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

    /// REQ-39・#711: 上限を超えた先を指す symlink の `history.jsonl` は回さない。リンクもリンク先もそのままで、
    /// `.1` はできない（検査より先にパス指定で長さを見て rename していた頃は、リンクが `.1` へ移っていた）。
    #[test]
    fn oversized_symlink_is_not_rotated() {
        let d = tmp("rot-link");
        let target = tmp("rot-link-target").join("big");
        fs::write(&target, "0123456789").expect("write");
        symlink(&target, d.join(HISTORY_FILE)).expect("symlink");
        assert!(append_line(&d, b"x\n", 5).is_err());
        assert_eq!(fs::read_link(d.join(HISTORY_FILE)).expect("link"), target);
        assert_eq!(fs::read_to_string(&target).expect("read"), "0123456789");
        assert!(fs::symlink_metadata(d.join(HISTORY_ROTATED)).is_err());
        let _ = fs::remove_dir_all(&d);
        let _ = fs::remove_dir_all(target.parent().expect("parent"));
    }

    /// REQ-39・#711: 上限を超えた、権限の緩い（0644）`history.jsonl` は回さず、中身も変えない。
    #[test]
    fn oversized_loose_file_is_not_rotated() {
        let d = tmp("rot-loose");
        fs::write(d.join(HISTORY_FILE), "0123456789").expect("write");
        fs::set_permissions(d.join(HISTORY_FILE), fs::Permissions::from_mode(0o644))
            .expect("chmod");
        assert_eq!(
            append_line(&d, b"x\n", 5).expect_err("loose").kind(),
            io::ErrorKind::PermissionDenied
        );
        assert_eq!(read(&d, HISTORY_FILE), "0123456789");
        assert!(fs::symlink_metadata(d.join(HISTORY_ROTATED)).is_err());
        let _ = fs::remove_dir_all(&d);
    }

    /// REQ-39・#711: 回せない（`.1` が空でないディレクトリ）ときは書かない。上限を超えたまま増やさない。
    #[test]
    fn failed_rotation_does_not_append() {
        let d = tmp("rot-fail");
        append_line(&d, b"0123456789\n", 100).expect("first");
        fs::create_dir(d.join(HISTORY_ROTATED)).expect("mkdir");
        fs::write(d.join(HISTORY_ROTATED).join("x"), "x").expect("fill");
        assert!(append_line(&d, b"y\n", 5).is_err());
        assert_eq!(read(&d, HISTORY_FILE), "0123456789\n");
        let _ = fs::remove_dir_all(&d);
    }

    /// #711: 書き込み失敗では、途中まで書かれた分を元の長さへ戻す。成功なら触らない。
    #[test]
    fn failed_write_restores_length() {
        let d = tmp("restore");
        let mut f = PrivateDir::open(&d)
            .and_then(|p| p.open_append(HISTORY_FILE))
            .expect("open");
        f.write_all(b"keep\n").expect("write");
        f.write_all(b"par").expect("partial");
        let err = io::Error::from(io::ErrorKind::WriteZero);
        assert!(restore_on_err(&f, 5, Err(err)).is_err());
        assert_eq!(read(&d, HISTORY_FILE), "keep\n");
        assert!(restore_on_err(&f, 0, Ok(())).is_ok());
        assert_eq!(read(&d, HISTORY_FILE), "keep\n");
        let _ = fs::remove_dir_all(&d);
    }

    /// REQ-33・#711: 保存が停滞しても、上限だけ待って戻る（停滞した処理の終わりを待たない）。
    /// 終わる処理は上限まで待たずに戻り、最後まで走っている。
    #[test]
    fn bounded_returns_at_limit_without_waiting_for_stalled_work() {
        let started = Instant::now();
        bounded(Duration::from_millis(50), || {
            std::thread::sleep(Duration::from_secs(30));
        });
        let waited = started.elapsed();
        assert!(waited >= Duration::from_millis(50), "{waited:?}");
        assert!(waited < Duration::from_secs(10), "{waited:?}");

        let (tx, rx) = mpsc::channel();
        let started = Instant::now();
        bounded(Duration::from_secs(30), move || {
            let _ = tx.send(7);
        });
        assert!(started.elapsed() < Duration::from_secs(10));
        assert_eq!(rx.try_recv(), Ok(7));
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
