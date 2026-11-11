//! `infer --text` の判定履歴（issue #711。REQ-33・REQ-38・REQ-39。証拠種別: テストハーネス〔実バイナリ・合成パッケージ〕）。
//!
//! 共有 fixture の C1 で組んだパッケージを使う。置き場所は `FANDHE_EDGE_SERVE_DIR` で `/tmp` 直下の短い
//! 一時ディレクトリへ向ける（macOS の `sun_path` は 104 バイトまで）。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::fs;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt, symlink};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const C1_DEFINITION: &str = r#"{"schema":"fandhe-edge-model-definition/v1","name":"kv","version":1,"judgment_type":"single_select","options":[{"id":"alpha","display_name":"a","description":"d"},{"id":"beta","display_name":"b","description":"d"},{"id":"gamma","display_name":"g","description":"d"}],"io":{"input":"bytes"}}"#;

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn unique(tag: &str) -> String {
    format!(
        "{}-{}-{tag}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::SeqCst)
    )
}

fn workspace(tag: &str) -> PathBuf {
    let ws = std::env::temp_dir().join(format!("fe-hist-ws-{}", unique(tag)));
    let _ = fs::remove_dir_all(&ws);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let c = ws.join("c");
    fs::create_dir_all(&c).expect("mkdir");
    let onnx = fs::read(root.join("onnx_parity/c1.onnx")).expect("fixture onnx");
    let sha = fandhe_edge_core::hash::Sha256Digest::of_bytes(&onnx).to_hex();
    fs::write(c.join("model.onnx"), &onnx).expect("write onnx");
    fs::write(
        c.join("artifact.json"),
        format!(
            r#"{{"kind":"c1","kind_version":1,"max_bytes":48,"label_order":["alpha","beta","gamma"],"onnx_file":"model.onnx","onnx_sha256":"{sha}"}}"#
        ),
    )
    .expect("write meta");
    fs::write(c.join("definition.json"), C1_DEFINITION).expect("write definition");
    ws
}

/// `/tmp` 直下の短い置き場所（ディレクトリ自体は作らない）。
fn dir_for(tag: &str) -> PathBuf {
    let dir = PathBuf::from(format!("/tmp/fe-hs-{}", unique(tag)));
    let _ = fs::remove_dir_all(&dir);
    dir
}

struct Server {
    child: Option<Child>,
    dir: PathBuf,
}

impl Server {
    fn start(ws: &Path, dir: &Path) -> Self {
        let child = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
            .args(["serve", "--package", "c"])
            .current_dir(ws)
            .env("FANDHE_EDGE_SERVE_DIR", dir)
            .env_remove("FANDHE_EDGE_NO_SERVE")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn serve");
        let server = Self {
            child: Some(child),
            dir: dir.to_path_buf(),
        };
        let sock = dir.join(fandhe_edge_cli::serve::SOCKET_FILE);
        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            if fs::metadata(&sock).is_ok_and(|m| m.permissions().mode() & 0o777 == 0o600)
                && UnixStream::connect(&sock).is_ok()
            {
                return server;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!("serve did not become ready");
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = fs::remove_dir_all(&self.dir);
    }
}

struct Run {
    code: Option<i32>,
    stdout: Vec<u8>,
    stderr: String,
}

fn infer_args(ws: &Path, dir: &Path, history: Option<&str>, no_serve: bool, args: &[&str]) -> Run {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"));
    cmd.arg("infer")
        .args(args)
        .current_dir(ws)
        .env("FANDHE_EDGE_SERVE_DIR", dir)
        .env_remove("FANDHE_EDGE_HISTORY")
        .stdin(Stdio::null());
    if let Some(h) = history {
        cmd.env("FANDHE_EDGE_HISTORY", h);
    }
    if no_serve {
        cmd.env("FANDHE_EDGE_NO_SERVE", "1");
    } else {
        cmd.env_remove("FANDHE_EDGE_NO_SERVE");
    }
    let out = cmd.output().expect("run");
    Run {
        code: out.status.code(),
        stdout: out.stdout,
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn infer(ws: &Path, dir: &Path, history: Option<&str>, no_serve: bool, package: &str) -> Run {
    infer_args(
        ws,
        dir,
        history,
        no_serve,
        &["--package", package, "--text", "hello \"world\""],
    )
}

fn lines(dir: &Path, name: &str) -> Vec<String> {
    fs::read_to_string(dir.join(name))
        .expect("read history")
        .lines()
        .map(str::to_string)
        .collect()
}

fn field<'a>(line: &'a str, key: &str) -> &'a str {
    // 欄の値の先頭（次の `,"` まで）を返す簡易取り出し。output は呼び出し側で別に比べる。
    let pat = format!("\"{key}\":");
    let start = line.find(&pat).expect("field") + pat.len();
    let rest = &line[start..];
    let end = rest.find(",\"").unwrap_or(rest.len() - 1);
    &rest[..end]
}

/// REQ-33・#711: 環境変数なし・`"1"` 以外では置き場所に何も作られない。
#[test]
fn history_disabled_creates_nothing() {
    let (ws, dir) = (workspace("off"), dir_for("off"));
    for h in [None, Some("0"), Some("true"), Some("")] {
        let run = infer(&ws, &dir, h, true, "c");
        assert_eq!(run.code, Some(0));
        assert!(!dir.exists(), "{h:?}");
    }
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-33・#711: 単独で 1 件 1 行。欄が期待どおりで、stdout・終了コードは無効のときと同一。
#[test]
fn history_standalone_one_line_and_output_unchanged() {
    let (ws, dir) = (workspace("solo"), dir_for("solo"));
    let off = infer(&ws, &dir, None, true, "c");
    let on = infer(&ws, &dir, Some("1"), true, "c");
    assert_eq!(on.code, off.code);
    assert_eq!(on.stdout, off.stdout);
    assert_eq!(on.stderr, off.stderr);
    let l = lines(&dir, "history.jsonl");
    assert_eq!(l.len(), 1);
    let line = &l[0];
    let stdout = String::from_utf8(on.stdout).expect("utf8");
    // output は JSON として読み直した値（キーは辞書順）で入るため、バイト列ではなく中身で照合する。
    for needle in [r#""predicted_label":"beta""#, r#""status":"ok""#] {
        assert!(stdout.contains(needle) && line.contains(needle), "{line}");
    }
    assert!(
        line.contains(r#""output":{"id":"input","predicted_label":"beta""#),
        "{line}"
    );
    assert!(line.contains(r#""text":"hello \"world\"""#), "{line}");
    assert_eq!(field(line, "exit_code"), "0");
    assert!(line.contains(r#""package":"c""#), "{line}");
    assert!(
        line.contains(&format!(
            "\"cli_version\":\"{}\"",
            env!("CARGO_PKG_VERSION")
        )),
        "{line}"
    );
    let mode = |p: PathBuf| fs::metadata(p).expect("meta").permissions().mode() & 0o777;
    assert_eq!(mode(dir.clone()), 0o700);
    assert_eq!(mode(dir.join("history.jsonl")), 0o600);
    let _ = fs::remove_dir_all(&ws);
    let _ = fs::remove_dir_all(&dir);
}

/// REQ-33・#711: 常駐経由（stderr が `serve: delegated`）でも 1 行。
#[test]
fn history_via_serve_one_line() {
    let (ws, dir) = (workspace("srv"), dir_for("srv"));
    let _server = Server::start(&ws, &dir);
    let run = infer(&ws, &dir, Some("1"), false, "c");
    assert_eq!(run.stderr.trim_end(), "serve: delegated");
    assert_eq!(run.code, Some(0));
    let l = lines(&dir, "history.jsonl");
    assert_eq!(l.len(), 1);
    assert_eq!(field(&l[0], "exit_code"), "0");
    assert!(l[0].contains(r#""text":"hello \"world\"""#), "{}", l[0]);
    assert!(
        l[0].contains(r#""output":{"id":"input","predicted_label":"beta""#),
        "{}",
        l[0]
    );
    assert!(l[0].contains(r#""status":"ok""#), "{}", l[0]);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-21・#711: エラーで終わる infer でも 1 行残り、exit_code は実際の値。
#[test]
fn history_records_error_exit() {
    let (ws, dir) = (workspace("err"), dir_for("err"));
    let on = infer(&ws, &dir, Some("1"), true, "nope");
    let off = infer(&ws, &dir_for("err2"), None, true, "nope");
    assert_eq!(on.code, Some(64));
    assert_eq!(on.stdout, off.stdout);
    let l = lines(&dir, "history.jsonl");
    assert_eq!(l.len(), 1);
    assert_eq!(field(&l[0], "exit_code"), "64");
    assert!(
        l[0].contains(r#""output":{"code":"invalid_input""#),
        "{}",
        l[0]
    );
    let _ = fs::remove_dir_all(&ws);
    let _ = fs::remove_dir_all(&dir);
}

/// REQ-39・#711: 置き場所が 0755・history.jsonl が symlink なら書かれず、stdout・終了コードは変わらない。
#[test]
fn history_refuses_unsafe_locations() {
    let (ws, dir) = (workspace("bad"), dir_for("bad"));
    let off = infer(&ws, &dir, None, true, "c");
    fs::create_dir_all(&dir).expect("mkdir");
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).expect("chmod");
    let on = infer(&ws, &dir, Some("1"), true, "c");
    assert_eq!((on.code, &on.stdout), (off.code, &off.stdout));
    assert_eq!(fs::read_dir(&dir).expect("ls").count(), 0);

    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).expect("chmod");
    let target = ws.join("target.txt");
    fs::write(&target, "keep").expect("write");
    symlink(&target, dir.join("history.jsonl")).expect("symlink");
    let on = infer(&ws, &dir, Some("1"), true, "c");
    assert_eq!((on.code, &on.stdout), (off.code, &off.stdout));
    assert_eq!(fs::read_to_string(&target).expect("read"), "keep");
    let _ = fs::remove_dir_all(&ws);
    let _ = fs::remove_dir_all(&dir);
}

/// REQ-39・#711: 置き場所を作った後にパス指定で権限を直さない。umask で 0700 にならなかった置き場所（0500）は
/// そのまま残り、履歴は書かれず、stdout・終了コードは変わらない。
#[test]
fn history_does_not_chmod_created_dir() {
    let (ws, dir) = (workspace("umask"), dir_for("umask"));
    let off = baseline(&ws);
    let out = Command::new("/bin/sh")
        .args(["-c", "umask 277; exec \"$0\" \"$@\""])
        .arg(env!("CARGO_BIN_EXE_fandhe-edge"))
        .args(["infer", "--package", "c", "--text", "hello \"world\""])
        .current_dir(&ws)
        .env("FANDHE_EDGE_SERVE_DIR", &dir)
        .env("FANDHE_EDGE_HISTORY", "1")
        .env("FANDHE_EDGE_NO_SERVE", "1")
        .stdin(Stdio::null())
        .output()
        .expect("run");
    assert_eq!((out.status.code(), &out.stdout), (off.code, &off.stdout));
    let mode = fs::metadata(&dir).expect("dir").permissions().mode() & 0o777;
    assert_eq!(mode, 0o500);
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).expect("chmod");
    assert_eq!(fs::read_dir(&dir).expect("ls").count(), 0);
    let _ = fs::remove_dir_all(&ws);
    let _ = fs::remove_dir_all(&dir);
}

/// 無効のときの結果（stdout・終了コード）。
fn baseline(ws: &Path) -> Run {
    infer(ws, &dir_for("base"), None, true, "c")
}

fn make_private_dir(dir: &Path) {
    fs::create_dir_all(dir).expect("mkdir");
    fs::set_permissions(dir, fs::Permissions::from_mode(0o700)).expect("chmod");
}

/// REQ-33・#711: 8 プロセス並列でも 8 行すべてが 1 行 1 JSON で読め、入力がそれぞれ 1 回ずつ現れる。
#[test]
fn history_concurrent_appends_keep_every_line() {
    let (ws, dir) = (workspace("par"), dir_for("par"));
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let (ws, dir) = (ws.clone(), dir.clone());
            thread::spawn(move || {
                let text = format!("input-{i}");
                let run = infer_args(
                    &ws,
                    &dir,
                    Some("1"),
                    true,
                    &["--package", "c", "--text", &text],
                );
                assert_eq!(run.code, Some(0), "{}", run.stderr);
            })
        })
        .collect();
    for h in handles {
        h.join().expect("join");
    }
    let l = lines(&dir, "history.jsonl");
    assert_eq!(l.len(), 8);
    for i in 0..8 {
        let needle = format!(r#""text":"input-{i}""#);
        assert_eq!(l.iter().filter(|x| x.contains(&needle)).count(), 1, "{i}");
    }
    for x in &l {
        assert!(x.starts_with("{\"time\":") && x.ends_with('}'), "{x}");
        assert!(x.contains(r#""output":{"id":"input""#), "{x}");
    }
    let _ = fs::remove_dir_all(&ws);
    let _ = fs::remove_dir_all(&dir);
}

/// REQ-33・REQ-39・#711: `history.lock` が取られたままでも infer は待たずに戻り、出力・終了コードは同じで履歴は増えない。
#[test]
fn history_lock_held_skips_without_changing_output() {
    let (ws, dir) = (workspace("lock"), dir_for("lock"));
    let off = baseline(&ws);
    make_private_dir(&dir);
    let lock = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(dir.join("history.lock"))
        .expect("lock file");
    lock.lock().expect("lock");
    let started = Instant::now();
    let on = infer(&ws, &dir, Some("1"), true, "c");
    assert!(started.elapsed() < Duration::from_secs(30));
    assert_eq!((on.code, &on.stdout), (off.code, &off.stdout));
    assert!(!dir.join("history.jsonl").exists());
    drop(lock);
    let _ = fs::remove_dir_all(&ws);
    let _ = fs::remove_dir_all(&dir);
}

/// REQ-39・#711: `history.jsonl` が FIFO でも止まらず、出力・終了コードは同じ。
#[test]
fn history_fifo_does_not_block() {
    let (ws, dir) = (workspace("fifo"), dir_for("fifo"));
    let off = baseline(&ws);
    make_private_dir(&dir);
    let ok = Command::new("mkfifo")
        .arg(dir.join("history.jsonl"))
        .status()
        .expect("mkfifo")
        .success();
    assert!(ok);
    let started = Instant::now();
    let on = infer(&ws, &dir, Some("1"), true, "c");
    assert!(started.elapsed() < Duration::from_secs(30));
    assert_eq!((on.code, &on.stdout), (off.code, &off.stdout));
    let _ = fs::remove_dir_all(&ws);
    let _ = fs::remove_dir_all(&dir);
}

/// REQ-39・#711: 権限の緩い（0644）既存の `history.jsonl` には書かれない。
#[test]
fn history_refuses_loose_existing_file() {
    let (ws, dir) = (workspace("loose"), dir_for("loose"));
    let off = baseline(&ws);
    make_private_dir(&dir);
    fs::write(dir.join("history.jsonl"), "old\n").expect("write");
    fs::set_permissions(dir.join("history.jsonl"), fs::Permissions::from_mode(0o644))
        .expect("chmod");
    let on = infer(&ws, &dir, Some("1"), true, "c");
    assert_eq!((on.code, &on.stdout), (off.code, &off.stdout));
    assert_eq!(
        fs::read_to_string(dir.join("history.jsonl")).expect("read"),
        "old\n"
    );
    let _ = fs::remove_dir_all(&ws);
    let _ = fs::remove_dir_all(&dir);
}

/// REQ-33・#711: `--input-file` では書かれない。
#[test]
fn history_skips_input_file() {
    let (ws, dir) = (workspace("file"), dir_for("file"));
    fs::write(ws.join("in.jsonl"), "{\"id\":\"a\",\"input\":\"hello\"}\n").expect("write");
    let run = infer_args(
        &ws,
        &dir,
        Some("1"),
        true,
        &["--package", "c", "--input-file", "in.jsonl"],
    );
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert!(!dir.exists());
    let _ = fs::remove_dir_all(&ws);
}
