//! 常駐プロセス `fandhe-edge serve` と、`infer --text` から委ねる経路（#708・#709。REQ-21・REQ-28・REQ-33・
//! REQ-38・REQ-39。証拠種別: テストハーネス〔実バイナリ・合成パッケージ〕）。
//!
//! 共有 fixture の C1（`fixtures/onnx_parity`）と合成 `qwen2_lora`（`fixtures/qwen2_parity`）で組んだ
//! パッケージを使う。常駐は子プロセスで起動し、置き場所は `FANDHE_EDGE_SERVE_DIR` で `/tmp` 直下の
//! 短い一時ディレクトリへ向ける（macOS の `sun_path` は 104 バイトまで）。テストの終わりに必ず kill する。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use fandhe_edge_core::serve_protocol::{
    PROTOCOL_VERSION, PackageIdentity, ServeRequest, ServeResponse, decode_response, encode_request,
};

use fandhe_edge_cli::serve::SOCKET_FILE;

/// 以前の版（#708）のソケットのファイル名。
const LEGACY_SOCKET_FILE: &str = "serve.sock";

/// 常駐と同じ CLI の版（このテストは cli crate の一部なので同じ値になる）。
const CLI_VERSION: &str = env!("CARGO_PKG_VERSION");

const C1_DEFINITION: &str = r#"{"schema":"fandhe-edge-model-definition/v1","name":"kv","version":1,"judgment_type":"single_select","options":[{"id":"alpha","display_name":"a","description":"d"},{"id":"beta","display_name":"b","description":"d"},{"id":"gamma","display_name":"g","description":"d"}],"io":{"input":"bytes"}}"#;
const C1_INPUTS: [&str; 3] = ["hello world", "foo bar", "zzz"];
const QWEN2_INPUTS: [&str; 2] = [
    "red3 red0 sky3 red1 red2 red0 red1 red2 red0 red3",
    "tan3 sky1 sky0 sky1 sky0 sky1 sky3",
];

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn unique(tag: &str) -> String {
    format!(
        "{}-{}-{tag}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::SeqCst)
    )
}

/// cwd 相当の作業ディレクトリ。`c`（C1）と `q`（`qwen2_lora`）のパッケージを置く。
fn workspace(tag: &str) -> PathBuf {
    let ws = std::env::temp_dir().join(format!("fe-serve-ws-{}", unique(tag)));
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

    let q = ws.join("q");
    fs::create_dir_all(&q).expect("mkdir");
    for f in [
        "artifact.json",
        "model.onnx",
        "tokenizer.json",
        "weights-00000.bin",
    ] {
        fs::copy(root.join("qwen2_parity").join(f), q.join(f)).expect("copy");
    }
    let options: Vec<String> = ["red", "tan", "sky", "map", "fox", "owl", "gem", "ink"]
        .iter()
        .map(|l| format!(r#"{{"id":"{l}","display_name":"{l}","description":"d"}}"#))
        .collect();
    fs::write(
        q.join("definition.json"),
        format!(
            r#"{{"schema":"fandhe-edge-model-definition/v1","name":"q","version":1,"judgment_type":"single_select","options":[{}],"io":{{"input":"bytes"}}}}"#,
            options.join(",")
        ),
    )
    .expect("definition");
    ws
}

/// `/tmp` 直下の短い置き場所（ディレクトリ自体は作らない。`serve` が 0700 で作る）。
fn serve_dir(tag: &str) -> PathBuf {
    let dir = PathBuf::from(format!("/tmp/fe-sv-{}", unique(tag)));
    let _ = fs::remove_dir_all(&dir);
    dir
}

/// 起動した常駐。Drop で kill して置き場所を消す。
struct Server {
    child: Option<Child>,
    dir: PathBuf,
}

impl Server {
    fn start(ws: &Path, dir: &Path, package: &str, env: &[(&str, &str)]) -> Self {
        Self::start_with_args(ws, dir, package, &[], env)
    }

    /// `serve` に追加の引数（`--idle-exit-secs` など）を渡して起動する。
    fn start_with_args(
        ws: &Path,
        dir: &Path,
        package: &str,
        extra: &[&str],
        env: &[(&str, &str)],
    ) -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"));
        cmd.args(["serve", "--package", package])
            .args(extra)
            .current_dir(ws)
            .env("FANDHE_EDGE_SERVE_DIR", dir)
            .env_remove("FANDHE_EDGE_NO_SERVE")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, v) in env {
            cmd.env(k, v);
        }
        let mut server = Self {
            child: Some(cmd.spawn().expect("spawn serve")),
            dir: dir.to_path_buf(),
        };
        server.wait_ready();
        server
    }

    /// ソケットが 0600 で現れ、接続できるまで待つ（chmod が bind の後のため）。
    fn wait_ready(&mut self) {
        let sock = self.dir.join(SOCKET_FILE);
        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            let child = self.child.as_mut().expect("child");
            if let Some(status) = child.try_wait().expect("try_wait") {
                panic!("serve exited early: {status:?}");
            }
            // 残ったソケットを除いて作り直す起動では 0600 の古いソケットが先に見えるため、接続できることも要る。
            if fs::metadata(&sock).is_ok_and(|m| m.permissions().mode() & 0o777 == 0o600)
                && UnixStream::connect(&sock).is_ok()
            {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!("serve did not become ready");
    }

    /// `lsof` の確認（macOS のみ）で使う。
    #[cfg(target_os = "macos")]
    fn pid(&self) -> u32 {
        self.child.as_ref().expect("child").id()
    }

    /// kill して stderr を返す。
    fn stop(mut self) -> String {
        let mut child = self.child.take().expect("child");
        let _ = child.kill();
        let out = child.wait_with_output().expect("wait");
        String::from_utf8_lossy(&out.stderr).into_owned()
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

fn infer(ws: &Path, dir: &Path, package: &str, no_serve: bool, text: &str) -> Run {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"));
    cmd.args(["infer", "--package", package, "--text", text])
        .current_dir(ws)
        .env("FANDHE_EDGE_SERVE_DIR", dir)
        .stdin(Stdio::null());
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

/// 常駐経由と単独で、stdout のバイト列と終了コードが一致する。
fn assert_parity(ws: &Path, dir: &Path, package: &str, text: &str) {
    let served = infer(ws, dir, package, false, text);
    let alone = infer(ws, dir, package, true, text);
    assert_eq!(served.stderr.trim_end(), "serve: delegated", "{package}");
    assert_eq!(
        alone.stderr.trim_end(),
        "serve: standalone (disabled by FANDHE_EDGE_NO_SERVE)"
    );
    assert_eq!(served.code, alone.code, "{package}");
    assert_eq!(served.stdout, alone.stdout, "{package}");
}

/// REQ-21・REQ-28・REQ-33: C1 の常駐経由と単独で stdout のバイト列と終了コードが一致する。
#[test]
fn req33_c1_served_output_identical_to_standalone() {
    let (ws, dir) = (workspace("par-c"), serve_dir("par-c"));
    let _server = Server::start(&ws, &dir, "c", &[]);
    for t in C1_INPUTS {
        assert_parity(&ws, &dir, "c", t);
    }
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-21・REQ-28・REQ-33: `qwen2_lora` で正常・入力エラー（系列長超過で非 0 の終了）とも一致する。
#[test]
fn req33_qwen2_served_output_identical_to_standalone() {
    let (ws, dir) = (workspace("par-q"), serve_dir("par-q"));
    let _server = Server::start(&ws, &dir, "q", &[]);
    for t in QWEN2_INPUTS {
        assert_parity(&ws, &dir, "q", t);
    }
    let long = "red3 ".repeat(1500);
    assert_parity(&ws, &dir, "q", &long);
    let alone = infer(&ws, &dir, "q", true, &long);
    assert_ne!(alone.code, Some(0), "the long input must be an error case");
    let _ = fs::remove_dir_all(&ws);
}

/// 常駐が動いていないとき、`infer` は単独で判定して exit 0。
#[test]
fn infer_without_serve_runs_standalone() {
    let (ws, dir) = (workspace("none"), serve_dir("none"));
    let run = infer(&ws, &dir, "c", false, "hello world");
    assert_eq!(run.code, Some(0));
    assert!(run.stdout.starts_with(b"{"), "stdout is a JSON line");
    // 常駐を使っていない環境では stderr も従来のまま（無言）。
    assert_eq!(run.stderr, "");
    let _ = fs::remove_dir_all(&ws);
}

/// 別のパッケージで動く常駐には委ねず、単独で判定する。
#[test]
fn infer_with_other_package_runs_standalone() {
    let (ws, dir) = (workspace("other"), serve_dir("other"));
    let _server = Server::start(&ws, &dir, "c", &[]);
    let run = infer(&ws, &dir, "q", false, QWEN2_INPUTS[0]);
    assert_eq!(run.code, Some(0));
    assert_eq!(
        run.stderr.trim_end(),
        "serve: standalone (package mismatch)"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// `c` パッケージの正しい同一性（生ソケットの要求用。委ねる側と同じ規則）。
fn c_identity(ws: &Path) -> PackageIdentity {
    use fandhe_edge_core::hash::Sha256Digest;
    let sha = |name: &str| {
        Sha256Digest::of_bytes(&fs::read(ws.join("c").join(name)).expect("read")).to_hex()
    };
    PackageIdentity {
        onnx_sha256: sha("model.onnx"),
        shard_sha256: vec![],
        artifact_sha256: sha("artifact.json"),
        definition_sha256: sha("definition.json"),
        calibration_sha256: None,
        layout_sha256: {
            use fandhe_edge_cli::serve_layout::{FileStat, layout_sha256};
            let stats = ["artifact.json", "model.onnx", "definition.json"]
                .iter()
                .map(|n| {
                    let f = fs::File::open(ws.join("c").join(n)).expect("open");
                    FileStat::of(n, &f).expect("stat")
                })
                .collect();
            layout_sha256(stats)
        },
    }
}

/// REQ-39: 待ちが上限（判定中 1 件 + 待ち 8 件）に達した状態で `infer` を走らせると `limit_exceeded`・exit 20
/// （生ソケットで 9 本をつないだまま保持して状態を決定的に作る）。
#[test]
fn req39_queue_overflow_returns_limit_exceeded() {
    let (ws, dir) = (workspace("queue"), serve_dir("queue"));
    let _server = Server::start(
        &ws,
        &dir,
        "c",
        &[("FANDHE_EDGE_SERVE_TEST_DELAY_MS", "3000")],
    );
    let request = encode_request(&ServeRequest {
        protocol: PROTOCOL_VERSION,
        cli_version: CLI_VERSION.into(),
        id: "held".into(),
        text: "hello world".into(),
        package: c_identity(&ws),
    });
    let mut held = Vec::new();
    for i in 0..9 {
        let mut s = UnixStream::connect(dir.join(SOCKET_FILE)).expect("connect");
        s.write_all(&request).expect("write");
        held.push(s);
        // 1 本目は判定スレッドが取り出して待ちに入るまで、以降は待ち行列に入るまで待つ。
        thread::sleep(Duration::from_millis(if i == 0 { 500 } else { 100 }));
    }
    let run = infer(&ws, &dir, "c", false, "hello world");
    assert_eq!(run.code, Some(20));
    assert!(
        run.stdout.starts_with(b"{\"code\":\"limit_exceeded\""),
        "{}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert_eq!(run.stderr.trim_end(), "serve: delegated");
    drop(held);
    let _ = fs::remove_dir_all(&ws);
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("mkdir");
    for e in fs::read_dir(from).expect("read_dir") {
        let e = e.expect("entry");
        fs::copy(e.path(), to.join(e.file_name())).expect("copy");
    }
}

/// REQ-39: ONNX が同じで `definition.json` だけ違うパッケージには委ねず、単独で判定する。
#[test]
fn req39_different_definition_runs_standalone() {
    let (ws, dir) = (workspace("def"), serve_dir("def"));
    let _server = Server::start(&ws, &dir, "c", &[]);
    copy_dir(&ws.join("c"), &ws.join("d"));
    let def = fs::read_to_string(ws.join("d/definition.json")).expect("read");
    fs::write(ws.join("d/definition.json"), def.replace("\"a\"", "\"A2\"")).expect("write");
    let run = infer(&ws, &dir, "d", false, "hello world");
    assert_eq!(run.code, Some(0));
    assert_eq!(
        run.stderr.trim_end(),
        "serve: standalone (package mismatch)"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: 校正ファイルがあるだけ（ONNX・定義は同じ）のパッケージにも委ねず、単独で判定する。
#[test]
fn req39_different_calibration_runs_standalone() {
    use fandhe_edge_core::hash::Sha256Digest;
    let (ws, dir) = (workspace("cal"), serve_dir("cal"));
    let _server = Server::start(&ws, &dir, "c", &[]);
    copy_dir(&ws.join("c"), &ws.join("k"));
    let onnx_sha =
        Sha256Digest::of_bytes(&fs::read(ws.join("k/model.onnx")).expect("read")).to_hex();
    let cal = format!(
        r#"{{"onnx_sha256":"{onnx_sha}","label_order":["alpha","beta","gamma"],"temperature":1.0,"threshold":0.0}}"#
    );
    fs::write(ws.join("k/calibration.json"), &cal).expect("write");
    let cal_sha = Sha256Digest::of_bytes(cal.as_bytes()).to_hex();
    let meta = fs::read_to_string(ws.join("k/artifact.json")).expect("read");
    let meta = meta.replacen('{', &format!("{{\"calibration_sha256\":\"{cal_sha}\","), 1);
    fs::write(ws.join("k/artifact.json"), meta).expect("write");
    let run = infer(&ws, &dir, "k", false, "hello world");
    assert_eq!(
        run.code,
        Some(0),
        "{}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert_eq!(
        run.stderr.trim_end(),
        "serve: standalone (package mismatch)"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: 読み込んだ後に shard を書き換えたら、配置の指紋が変わるので委ねず単独へ戻り、単独が全量検証して
/// `invalid_input`（64）で拒否する。書き換え前は委ねて成功する。
#[test]
fn req39_shard_rewrite_after_load_falls_back_to_standalone() {
    let (ws, dir) = (workspace("shard"), serve_dir("shard"));
    let _server = Server::start(&ws, &dir, "q", &[]);
    let before = infer(&ws, &dir, "q", false, QWEN2_INPUTS[0]);
    assert_eq!(before.code, Some(0));
    assert_eq!(before.stderr.trim_end(), "serve: delegated");
    let shard = ws.join("q/weights-00000.bin");
    let len = fs::metadata(&shard).expect("meta").len();
    fs::write(&shard, vec![0u8; usize::try_from(len).expect("len")]).expect("rewrite");
    let after = infer(&ws, &dir, "q", false, QWEN2_INPUTS[0]);
    assert_eq!(
        after.stderr.trim_end(),
        "serve: standalone (package mismatch)"
    );
    assert_eq!(after.code, Some(64));
    let alone = infer(&ws, &dir, "q", true, QWEN2_INPUTS[0]);
    assert_eq!(after.stdout, alone.stdout);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: `tokenizer.json` を消したら、軽い検査で開けず委ねずに単独へ戻り、単独と同じエラーになる。
#[test]
fn req39_missing_tokenizer_falls_back_to_standalone() {
    let (ws, dir) = (workspace("tok"), serve_dir("tok"));
    let _server = Server::start(&ws, &dir, "q", &[]);
    fs::remove_file(ws.join("q/tokenizer.json")).expect("remove");
    let run = infer(&ws, &dir, "q", false, QWEN2_INPUTS[0]);
    assert_eq!(
        run.stderr.trim_end(),
        "serve: standalone (package check failed)"
    );
    let alone = infer(&ws, &dir, "q", true, QWEN2_INPUTS[0]);
    assert_ne!(run.code, Some(0));
    assert_eq!(run.code, alone.code);
    assert_eq!(run.stdout, alone.stdout);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: 起動時の検証に失敗したら、ソケットを作らず単独の `infer` と同じエラー JSON・終了コードで終わる。
#[test]
fn req39_startup_verification_failure_creates_no_socket() {
    let (ws, dir) = (workspace("bad"), serve_dir("bad"));
    let shard = ws.join("q/weights-00000.bin");
    let mut bytes = fs::read(&shard).expect("read");
    bytes.extend_from_slice(b"x");
    fs::write(&shard, bytes).expect("tamper");
    let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .args(["serve", "--package", "q"])
        .current_dir(&ws)
        .env("FANDHE_EDGE_SERVE_DIR", &dir)
        .stdin(Stdio::null())
        .output()
        .expect("run");
    let alone = infer(&ws, &dir, "q", true, "x");
    assert_eq!(out.status.code(), Some(64));
    assert_eq!(out.status.code(), alone.code);
    assert_eq!(out.stdout, alone.stdout);
    assert!(!dir.join(SOCKET_FILE).exists());
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・#723: 置き場所を作った後に、パス指定で権限を変えない。umask で 0700 にならなければ起動しない。
#[test]
fn req39_created_dir_is_not_chmoded() {
    let (ws, dir) = (workspace("umask"), serve_dir("umask"));
    let mut child = Command::new("/bin/sh")
        .args(["-c", "umask 277; exec \"$0\" \"$@\""])
        .arg(env!("CARGO_BIN_EXE_fandhe-edge"))
        .args(["serve", "--package", "c"])
        .current_dir(&ws)
        .env("FANDHE_EDGE_SERVE_DIR", &dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn serve");
    // 権限を直してしまう退行では常駐が起動して終わらないため、期限つきで待つ。
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("serve started despite a non-0700 directory");
        }
        thread::sleep(Duration::from_millis(20));
    };
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .expect("stdout")
        .read_to_string(&mut stdout)
        .expect("read stdout");
    assert_eq!(status.code(), Some(64));
    assert_eq!(
        stdout,
        "{\"code\":\"invalid_input\",\"message\":\"serve directory must be a real directory with mode 0700\"}\n"
    );
    let mode = fs::metadata(&dir).expect("dir").permissions().mode() & 0o777;
    assert_eq!(mode, 0o500);
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).expect("chmod");
    assert_eq!(fs::read_dir(&dir).expect("ls").count(), 0);
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: ディレクトリは 0700・ソケットは 0600。既存のディレクトリが 0700 でなければ起動しない。
#[test]
fn req39_dir_and_socket_permissions() {
    let (ws, dir) = (workspace("perm"), serve_dir("perm"));
    {
        let _server = Server::start(&ws, &dir, "c", &[]);
        let mode = |p: &Path| fs::metadata(p).expect("meta").permissions().mode() & 0o777;
        assert_eq!(mode(&dir), 0o700);
        assert_eq!(mode(&dir.join(SOCKET_FILE)), 0o600);
    }
    fs::create_dir_all(&dir).expect("mkdir");
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).expect("chmod");
    let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .args(["serve", "--package", "c"])
        .current_dir(&ws)
        .env("FANDHE_EDGE_SERVE_DIR", &dir)
        .stdin(Stdio::null())
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(64));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("mode 0700"),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(!dir.join(SOCKET_FILE).exists());
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: 二重起動は 2 つ目が失敗する。SIGKILL で残ったソケットは次の起動で作り直される。
#[test]
fn req39_second_start_fails_and_stale_socket_is_replaced() {
    let (ws, dir) = (workspace("dup"), serve_dir("dup"));
    let first = Server::start(&ws, &dir, "c", &[]);
    let second = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .args(["serve", "--package", "c"])
        .current_dir(&ws)
        .env("FANDHE_EDGE_SERVE_DIR", &dir)
        .stdin(Stdio::null())
        .output()
        .expect("run");
    assert_eq!(second.status.code(), Some(64));
    assert!(
        String::from_utf8_lossy(&second.stdout).contains("serve already running"),
        "{}",
        String::from_utf8_lossy(&second.stdout)
    );
    // 1 つ目は影響を受けず応答する。
    assert_eq!(
        infer(&ws, &dir, "c", false, "zzz").stderr.trim_end(),
        "serve: delegated"
    );

    // SIGKILL ではソケットが残る。そこへつなぐ `infer` は単独へ戻る。
    let sock = dir.join(SOCKET_FILE);
    let mut first = first;
    let mut c = first.child.take().expect("child");
    c.kill().expect("kill");
    c.wait().expect("wait");
    assert!(sock.exists(), "the socket should remain after SIGKILL");
    let stale = infer(&ws, &dir, "c", false, "zzz");
    assert_eq!(stale.code, Some(0));
    assert_eq!(
        stale.stderr.trim_end(),
        "serve: standalone (cannot connect)"
    );

    let _again = Server::start(&ws, &dir, "c", &[]);
    assert_eq!(
        infer(&ws, &dir, "c", false, "zzz").stderr.trim_end(),
        "serve: delegated"
    );
    drop(first);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-38: 常駐は TCP などで待ち受けない（`lsof` でネットワークのソケットが 0 件）。macOS で実行する。
#[cfg(target_os = "macos")]
#[test]
fn req38_serve_does_not_listen_on_network() {
    let (ws, dir) = (workspace("net"), serve_dir("net"));
    let server = Server::start(&ws, &dir, "c", &[]);
    let out = Command::new("lsof")
        .args(["-a", "-p", &server.pid().to_string(), "-i"])
        .output()
        .expect("lsof");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "",
        "serve must not hold network sockets"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・security.md: 常駐のログ（stderr）に入力の本文が出ない（id・時刻・終了コードのみ）。
#[test]
fn req39_serve_log_has_no_input_text() {
    let (ws, dir) = (workspace("log"), serve_dir("log"));
    let server = Server::start(&ws, &dir, "c", &[]);
    let marker = "ZQXMARKER-body-7731";
    let run = infer(&ws, &dir, "c", false, marker);
    assert_eq!(run.code, Some(0));
    let log = server.stop();
    assert!(log.contains("exit_code=0"), "{log}");
    assert!(!log.contains(marker), "{log}");
    assert!(!log.contains("predicted_label"), "{log}");
    let _ = fs::remove_dir_all(&ws);
}

/// `FANDHE_EDGE_NO_SERVE` があれば常駐が動いていても委ねない。
#[test]
fn no_serve_env_disables_delegation() {
    let (ws, dir) = (workspace("nosv"), serve_dir("nosv"));
    let server = Server::start(&ws, &dir, "c", &[]);
    let run = infer(&ws, &dir, "c", true, "zzz");
    assert_eq!(run.code, Some(0));
    assert_eq!(
        run.stderr.trim_end(),
        "serve: standalone (disabled by FANDHE_EDGE_NO_SERVE)"
    );
    assert!(!server.stop().contains("accepted_at"));
    let _ = fs::remove_dir_all(&ws);
}

/// `--input-file` は委ねず単独で処理する（初版の範囲）。
#[test]
fn input_file_is_not_delegated() {
    let (ws, dir) = (workspace("file"), serve_dir("file"));
    let _server = Server::start(&ws, &dir, "c", &[]);
    fs::write(ws.join("in.jsonl"), "{\"id\":\"a\",\"input\":\"zzz\"}\n").expect("write");
    let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .args(["infer", "--package", "c", "--input-file", "in.jsonl"])
        .current_dir(&ws)
        .env("FANDHE_EDGE_SERVE_DIR", &dir)
        .env_remove("FANDHE_EDGE_NO_SERVE")
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&out.stderr).trim_end(),
        "serve: standalone (input is not --text)"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: 不正な要求行は `invalid_input`（64）の JSON で返り、常駐は落ちない。
#[test]
fn req39_malformed_request_is_rejected_and_serve_survives() {
    let (ws, dir) = (workspace("raw"), serve_dir("raw"));
    let _server = Server::start(&ws, &dir, "c", &[]);
    let call = |line: &[u8]| {
        let mut s = UnixStream::connect(dir.join(SOCKET_FILE)).expect("connect");
        s.write_all(line).expect("write");
        let mut buf = Vec::new();
        s.read_to_end(&mut buf).expect("read");
        decode_response(&buf).expect("decode")
    };
    match call(b"not json\n") {
        ServeResponse::Ok { exit_code, stdout } => {
            assert_eq!(exit_code, 64);
            assert!(
                stdout.starts_with("{\"code\":\"invalid_input\""),
                "{stdout}"
            );
        }
        other => panic!("unexpected: {other:?}"),
    }
    // 別のパッケージを名乗る要求は package_mismatch。
    let request = encode_request(&ServeRequest {
        protocol: PROTOCOL_VERSION,
        cli_version: CLI_VERSION.into(),
        id: "t1".into(),
        text: "x".into(),
        package: PackageIdentity {
            onnx_sha256: "0".repeat(64),
            shard_sha256: vec![],
            artifact_sha256: "0".repeat(64),
            definition_sha256: "0".repeat(64),
            calibration_sha256: None,
            layout_sha256: "0".repeat(64),
        },
    });
    assert_eq!(call(&request), ServeResponse::PackageMismatch);
    assert_eq!(
        infer(&ws, &dir, "c", false, "zzz").stderr.trim_end(),
        "serve: delegated"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// 1 MiB を超える `--text` は委ねずに単独へ戻す（単独と同じ上限エラーを単独の経路に出させる。REQ-39・REQ-33）。
/// OS の引数長の上限で実バイナリには渡せないため、委ねる関数を直接呼ぶ。
#[test]
fn req39_oversized_text_is_not_delegated() {
    use fandhe_edge_cli::args::{InferArgs, InferSource};
    use fandhe_edge_cli::serve_client::{Delegation, try_delegate_in};
    let (ws, dir) = (workspace("big"), serve_dir("big"));
    let _server = Server::start(&ws, &dir, "c", &[]);
    let mk = |len: usize| InferArgs {
        package: PathBuf::from("c"),
        source: InferSource::Text {
            text: "a".repeat(len),
            id: None,
        },
        version: None,
    };
    let mut out = Vec::new();
    let big = try_delegate_in(&mut out, &mk(1_048_577), &ws, Some(dir.clone())).expect("io");
    assert_eq!(big, Delegation::Standalone("input is too large"));
    assert!(out.is_empty());
    // 上限ちょうどは委ねる（応答が返る）。
    let ok = try_delegate_in(&mut out, &mk(1_048_576), &ws, Some(dir.clone())).expect("io");
    assert!(matches!(ok, Delegation::Done(_)), "{ok:?}");
    let _ = fs::remove_dir_all(&ws);
}

/// 応答の前に常駐が終わった（判定中に kill）ときは、クライアントが単独で判定して exit 0。
///
/// REQ-39・#709: 親が SIGKILL されたら、判定中の worker も自分の見張りで親の不在に気づいてすぐ終わる
/// （判定の遅延 10 秒の終わりを待たない。親の見張りを失ったまま動き続けない）。
#[test]
fn serve_killed_while_judging_falls_back_to_standalone() {
    let (ws, dir) = (workspace("kill"), serve_dir("kill"));
    let server = Server::start(
        &ws,
        &dir,
        "c",
        &[("FANDHE_EDGE_SERVE_TEST_DELAY_MS", "10000")],
    );
    let workers = child_pids(server.child.as_ref().expect("child").id());
    assert_eq!(workers.len(), 1, "one worker: {workers:?}");
    let (run, stop_took, log) = thread::scope(|s| {
        let h = s.spawn(|| infer(&ws, &dir, "c", false, "hello world"));
        thread::sleep(Duration::from_millis(1500));
        // `stop` は親を kill し、stderr が閉じる（worker も終わる）まで待つ。
        let stopping = Instant::now();
        let log = server.stop();
        (h.join().expect("join"), stopping.elapsed(), log)
    });
    assert_eq!(
        run.code,
        Some(0),
        "{}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert_eq!(
        run.stderr.trim_end(),
        "serve: standalone (serve closed the connection)"
    );
    assert!(run.stdout.starts_with(b"{"));
    // 遅延の残り（8.5 秒）を待たずに worker が終わっている。
    assert!(stop_took < Duration::from_secs(4), "{stop_took:?}");
    assert!(gone_within(workers[0], Duration::from_secs(1)));
    assert!(log.contains("parent is gone; worker exiting"), "{log}");
    assert!(!log.contains("exit_code="), "{log}");
    let _ = fs::remove_dir_all(&ws);
}

/// 試しの接続ができない（残ったソケット）ときは、ONNX を読む検査より先に単独へ戻る
/// （パッケージを壊しても理由は `cannot connect` で、`package check failed` にならない）。
#[test]
fn stale_socket_returns_before_reading_the_package() {
    let (ws, dir) = (workspace("stale"), serve_dir("stale"));
    let mut server = Server::start(&ws, &dir, "c", &[]);
    let mut c = server.child.take().expect("child");
    c.kill().expect("kill");
    c.wait().expect("wait");
    fs::write(ws.join("c/artifact.json"), "{").expect("corrupt");
    let run = infer(&ws, &dir, "c", false, "x");
    assert_eq!(run.code, Some(64));
    assert_eq!(run.stderr.trim_end(), "serve: standalone (cannot connect)");
    let _ = fs::remove_dir_all(&ws);
}

/// 生のソケットへ 1 行送り、応答の 1 行を読む。
fn raw_call(dir: &Path, line: &[u8]) -> ServeResponse {
    let mut s = UnixStream::connect(dir.join(SOCKET_FILE)).expect("connect");
    s.write_all(line).expect("write");
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).expect("read");
    decode_response(&buf).expect("decode")
}

/// `c` パッケージ宛ての正しい要求（生ソケット用）。
fn c_request(ws: &Path, text: &str) -> ServeRequest {
    ServeRequest {
        protocol: PROTOCOL_VERSION,
        cli_version: CLI_VERSION.into(),
        id: "raw".into(),
        text: text.into(),
        package: c_identity(ws),
    }
}

/// `pid` のプロセスが `wait` 以内に消えるか。
fn gone_within(pid: u32, wait: Duration) -> bool {
    let alive = || {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stderr(Stdio::null())
            .status()
            .expect("kill -0")
            .success()
    };
    let deadline = Instant::now() + wait;
    while alive() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    !alive()
}

/// `parent` の直接の子（worker）の pid。
fn child_pids(parent: u32) -> Vec<u32> {
    let out = Command::new("pgrep")
        .args(["-P", &parent.to_string()])
        .output()
        .expect("pgrep");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect()
}

/// コマンドラインに `needle` を含むプロセスの数。
fn count_processes(needle: &str) -> usize {
    let out = Command::new("pgrep")
        .args(["-f", needle])
        .output()
        .expect("pgrep");
    String::from_utf8_lossy(&out.stdout).lines().count()
}

fn assert_limit_exceeded(run: &Run) {
    assert_eq!(run.code, Some(20), "{}", run.stderr);
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "{\"code\":\"limit_exceeded\",\"message\":\"resource limit exceeded\"}\n"
    );
    assert_eq!(run.stderr.trim_end(), "serve: delegated");
}

/// REQ-39・REQ-21・#709: worker が固まったら親が期限で止めて `limit_exceeded`（exit 20）を返し、worker を
/// 作り直す。その後の要求は委ねられて単独と同じ結果になる（常駐は続く）。
#[test]
fn req39_hung_worker_is_killed_and_restarted() {
    let (ws, dir) = (workspace("hang"), serve_dir("hang"));
    let server = Server::start(
        &ws,
        &dir,
        "c",
        &[
            ("FANDHE_EDGE_SERVE_TEST_HANG_TEXT", "hang-here"),
            ("FANDHE_EDGE_SERVE_TEST_TIME_MS", "1000"),
        ],
    );
    let before = child_pids(server.child.as_ref().expect("child").id());
    assert_eq!(before.len(), 1, "one worker: {before:?}");
    let started = Instant::now();
    assert_limit_exceeded(&infer(&ws, &dir, "c", false, "hang-here"));
    // 親の期限（1 秒 + 余裕 1 秒）で返る。worker 自身の時間上限（10 秒）までは待たない。
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
    for t in C1_INPUTS {
        assert_parity(&ws, &dir, "c", t);
    }
    let after = child_pids(server.child.as_ref().expect("child").id());
    assert_eq!(after.len(), 1, "one worker: {after:?}");
    assert_ne!(before, after, "the worker must have been replaced");
    let log = server.stop();
    assert!(
        log.contains("judge time limit exceeded; stopping worker"),
        "{log}"
    );
    assert!(log.contains("worker ready"), "{log}");
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・#709: worker を外から kill しても親が作り直し、その後の要求は委ねられて単独と同じ結果になる。
#[test]
fn req39_externally_killed_worker_is_restarted() {
    let (ws, dir) = (workspace("wkill"), serve_dir("wkill"));
    let server = Server::start(&ws, &dir, "c", &[]);
    assert_parity(&ws, &dir, "c", C1_INPUTS[0]);
    let workers = child_pids(server.child.as_ref().expect("child").id());
    assert_eq!(workers.len(), 1, "one worker: {workers:?}");
    let killed = Command::new("kill")
        .args(["-9", &workers[0].to_string()])
        .status()
        .expect("kill");
    assert!(killed.success());
    // 作り直しの前でも後でも、要求は委ねられる（途中なら作り直しを待つ）。
    for t in C1_INPUTS {
        assert_parity(&ws, &dir, "c", t);
    }
    let after = child_pids(server.child.as_ref().expect("child").id());
    assert_eq!(after.len(), 1, "one worker: {after:?}");
    assert_ne!(workers, after);
    // 親を止めたら、作り直した worker も（stdin の EOF で）消える。
    let _ = server.stop();
    assert!(
        gone_within(after[0], Duration::from_secs(10)),
        "the worker must exit after its parent"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・#709: `protocol`・`cli_version` の違う要求には判定せず `version_mismatch` を返し、常駐は続く。
/// 将来の版が欄を増やした要求も、不正な要求ではなく版の食い違いとして断る。
#[test]
fn req39_version_mismatch_is_answered_and_serve_survives() {
    let (ws, dir) = (workspace("ver"), serve_dir("ver"));
    let server = Server::start(&ws, &dir, "c", &[]);
    let mut other_protocol = c_request(&ws, "zzz");
    other_protocol.protocol = PROTOCOL_VERSION + 1;
    let mut other_cli = c_request(&ws, "zzz");
    other_cli.cli_version = format!("{CLI_VERSION}-other");
    let future = String::from_utf8(encode_request(&other_protocol))
        .expect("utf8")
        .replacen('{', "{\"future_field\":1,", 1);
    for line in [
        encode_request(&other_protocol),
        encode_request(&other_cli),
        future.into_bytes(),
    ] {
        assert_eq!(
            encode_response_line(&raw_call(&dir, &line)),
            "{\"status\":\"version_mismatch\"}\n"
        );
    }
    // 版が合う要求は判定される。
    assert!(matches!(
        raw_call(&dir, &encode_request(&c_request(&ws, "zzz"))),
        ServeResponse::Ok { exit_code: 0, .. }
    ));
    assert_parity(&ws, &dir, "c", "zzz");
    // 版の違う要求は判定されていない（ログの判定行は、生の 1 件と委ねた 1 件だけ）。
    let log = server.stop();
    assert_eq!(log.matches("exit_code=").count(), 2, "{log}");
    let _ = fs::remove_dir_all(&ws);
}

fn encode_response_line(response: &ServeResponse) -> String {
    String::from_utf8(fandhe_edge_core::serve_protocol::encode_response(response)).expect("utf8")
}

/// REQ-21・REQ-33・#709: 常駐が `version_mismatch` を返したら、`infer` は単独で判定して単独と同じ stdout・
/// 終了コードで終わる（偽の常駐: 何を受けても `version_mismatch` を返す）。
#[test]
fn req33_version_mismatch_response_falls_back_to_standalone() {
    let (ws, dir) = (workspace("fake"), serve_dir("fake"));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).expect("chmod");
    let listener = UnixListener::bind(dir.join(SOCKET_FILE)).expect("bind");
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            // 試しの接続は何も送らずに閉じる。要求の接続は 1 行送ってくる。
            let mut buf = [0u8; 65536];
            let mut seen = false;
            while let Ok(n) = stream.read(&mut buf) {
                if n == 0 {
                    break;
                }
                seen = true;
                if buf[..n].contains(&b'\n') {
                    break;
                }
            }
            if seen {
                let _ = stream.write_all(b"{\"status\":\"version_mismatch\"}\n");
            }
        }
    });
    for t in C1_INPUTS {
        let run = infer(&ws, &dir, "c", false, t);
        let alone = infer(&ws, &dir, "c", true, t);
        assert_eq!(
            run.stderr.trim_end(),
            "serve: standalone (version mismatch)"
        );
        assert_eq!(run.code, Some(0));
        assert_eq!(run.code, alone.code);
        assert_eq!(run.stdout, alone.stdout);
    }
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・REQ-21・#709: 読み込んだ worker の RSS が上限を超えていたら、ソケットを作らず `limit_exceeded` の
/// JSON 1 つ・exit 20 で終わり、worker も残さない。
#[test]
fn req39_worker_rss_over_limit_at_startup_exits_without_socket() {
    let (ws, dir) = (workspace("rss"), serve_dir("rss"));
    // worker をコマンドラインで見分けられるよう、パッケージ名を一意にする。
    let package = format!("pk-{}", unique("rss"));
    copy_dir(&ws.join("c"), &ws.join(&package));
    let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .args(["serve", "--package", &package])
        .current_dir(&ws)
        .env("FANDHE_EDGE_SERVE_DIR", &dir)
        .env("FANDHE_EDGE_SERVE_TEST_RSS_LIMIT_BYTES", "1")
        .stdin(Stdio::null())
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(20));
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "{\"code\":\"limit_exceeded\",\"message\":\"resource limit exceeded\"}\n"
    );
    assert!(!dir.join(SOCKET_FILE).exists());
    // kill は送信済み。プロセスが消えるまで少し待つ。
    let deadline = Instant::now() + Duration::from_secs(5);
    while count_processes(&package) > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(count_processes(&package), 0, "no worker may remain");
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・REQ-21・#709: 接続数が上限（64 本）のとき、常駐は要求を読まずに `limit_exceeded` を返して閉じる。
/// 大きい要求を書いている途中で閉じられた `infer` も、その応答を受け取って exit 20 で終わる（70 にしない）。
#[test]
fn req39_connection_limit_returns_limit_exceeded_for_large_request() {
    let (ws, dir) = (workspace("conn"), serve_dir("conn"));
    let _server = Server::start(&ws, &dir, "c", &[]);
    // 引数 1 つの長さの上限（Linux は 128 KiB）に収まり、ソケットのバッファ（macOS は 8 KiB）より大きい入力。
    let text = "a".repeat(120_000);
    // 何も送らない接続は、常駐が読み取りの期限（10 秒）で閉じて枠を空ける。負荷で `infer` の起動が遅れると
    // 枠が空いた後に届いて委ねられてしまうので、塞ぎ直して数回まで試す（上限に達した状態で届いた 1 回を見る）。
    let mut run = None;
    let mut held = Vec::new();
    for _ in 0..3 {
        held = (0..64)
            .map(|_| UnixStream::connect(dir.join(SOCKET_FILE)).expect("connect"))
            .collect::<Vec<UnixStream>>();
        // 常駐が 64 本とも受け付けて数えるまで待つ: 何も送らない追加の接続に、読まずに `limit_exceeded` が
        // 返ってくれば上限に達している。
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            let mut probe = UnixStream::connect(dir.join(SOCKET_FILE)).expect("connect");
            probe
                .set_read_timeout(Some(Duration::from_millis(200)))
                .expect("timeout");
            let mut buf = Vec::new();
            let _ = probe.read_to_end(&mut buf);
            if buf.starts_with(b"{\"status\":\"ok\",\"exit_code\":20,") {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the connection limit was not reached"
            );
        }
        let filled = Instant::now();
        let got = infer(&ws, &dir, "c", false, &text);
        // 塞いでから 5 秒以内に終わっていれば、枠はまだ空いていない（読み取りの期限 10 秒の半分）。
        let in_time = filled.elapsed() < Duration::from_secs(5);
        let limited = got.code == Some(20);
        run = Some(got);
        if in_time || limited {
            break;
        }
    }
    assert_limit_exceeded(&run.expect("at least one attempt"));
    drop(held);
    // 接続が空けば委ねられる（接続スレッドが抜けるまでの間は `limit_exceeded` が返りうるので、上限つきで待つ）。
    let deadline = Instant::now() + Duration::from_secs(10);
    while infer(&ws, &dir, "c", false, "zzz").code != Some(0) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(100));
    }
    assert_parity(&ws, &dir, "c", "zzz");
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・REQ-21・#709: 待っている間に期限を過ぎた要求は判定せず `limit_exceeded` を返す。
///
/// 1 件の判定を 3 秒に遅らせ、親の 1 件の期限を 4 秒にして、生のソケットで 0.1 秒おきに 3 件送る
/// （受け付けの順と間隔を決定的にする）。
/// - 1 件目: すぐ渡され、3 秒後に応答（親の待ちの上限 5 秒まで 2 秒の余裕）
/// - 2 件目: 3 秒後に取り出され、待ちは 2.9 秒（期限 4 秒まで 1.1 秒の余裕）で判定される
/// - 3 件目: 6 秒後に取り出され、待ちは 5.8 秒（期限 4 秒を 1.8 秒超過）で判定されない
#[test]
fn req39_request_expired_while_waiting_is_not_judged() {
    let (ws, dir) = (workspace("wait"), serve_dir("wait"));
    let server = Server::start(
        &ws,
        &dir,
        "c",
        &[
            ("FANDHE_EDGE_SERVE_TEST_DELAY_MS", "3000"),
            ("FANDHE_EDGE_SERVE_TEST_TIME_MS", "4000"),
        ],
    );
    let responses: Vec<ServeResponse> = thread::scope(|s| {
        let handles: Vec<_> = C1_INPUTS
            .iter()
            .map(|text| {
                let h = s.spawn(|| raw_call(&dir, &encode_request(&c_request(&ws, text))));
                thread::sleep(Duration::from_millis(100));
                h
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("join"))
            .collect()
    });
    for judged in &responses[..2] {
        assert!(
            matches!(judged, ServeResponse::Ok { exit_code: 0, .. }),
            "{judged:?}"
        );
    }
    assert_eq!(
        responses[2],
        ServeResponse::Ok {
            exit_code: 20,
            stdout: "{\"code\":\"limit_exceeded\",\"message\":\"resource limit exceeded\"}\n"
                .into(),
        }
    );
    let log = server.stop();
    assert_eq!(log.matches("exit_code=").count(), 2, "{log}");
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・#709: 1 件も判定しないまま worker が終わる作り直しが 3 回続いたら、親は exit 70 で終わる
/// （外からの kill の繰り返しで回り続けない）。その後の `infer` は単独で判定する。
#[test]
fn req39_three_barren_restarts_stop_the_parent() {
    let (ws, dir) = (workspace("barren"), serve_dir("barren"));
    let mut server = Server::start(&ws, &dir, "c", &[]);
    let parent = server.child.as_ref().expect("child").id();
    let mut last = 0;
    for round in 0..3 {
        // 作り直された（前回と違う pid の）worker が現れるまで待ってから kill する。
        let deadline = Instant::now() + Duration::from_secs(10);
        let worker = loop {
            match child_pids(parent).first() {
                Some(&pid) if pid != last => break pid,
                _ => assert!(Instant::now() < deadline, "no new worker in round {round}"),
            }
            thread::sleep(Duration::from_millis(50));
        };
        // `ready` の後（作り直しの完了後）に kill するよう少し待つ。
        thread::sleep(Duration::from_millis(500));
        let killed = Command::new("kill")
            .args(["-9", &worker.to_string()])
            .status()
            .expect("kill");
        assert!(killed.success());
        last = worker;
    }
    let mut child = server.child.take().expect("child");
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            break status;
        }
        assert!(Instant::now() < deadline, "the parent must exit");
        thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(status.code(), Some(70));
    let out = child.wait_with_output().expect("output");
    let log = String::from_utf8_lossy(&out.stderr);
    assert!(
        log.contains("worker stopped without judging 3 times in a row; exiting"),
        "{log}"
    );
    let run = infer(&ws, &dir, "c", false, "zzz");
    assert_eq!(run.code, Some(0));
    assert_eq!(run.stderr.trim_end(), "serve: standalone (cannot connect)");
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-21・REQ-33・#709: 以前の版（#708）の常駐が古い名前のソケットで動いていても、`infer` はそれを見ずに
/// 単独で判定する（古い常駐が新しい要求に返す `invalid serve request`・exit 64 を判定結果として採用しない）。
/// 常駐を使っていない環境と同じく、stderr に `serve:` の行は出ない。
#[test]
fn req33_legacy_socket_is_ignored() {
    let (ws, dir) = (workspace("legacy"), serve_dir("legacy"));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).expect("chmod");
    let listener = UnixListener::bind(dir.join(LEGACY_SOCKET_FILE)).expect("bind");
    let connections = std::sync::Arc::new(AtomicUsize::new(0));
    let seen = std::sync::Arc::clone(&connections);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            seen.fetch_add(1, Ordering::SeqCst);
            let _ = stream.write_all(
                b"{\"status\":\"ok\",\"exit_code\":64,\"stdout\":\"{\\\"code\\\":\\\"invalid_input\\\",\\\"message\\\":\\\"invalid serve request\\\"}\\n\"}\n",
            );
        }
    });
    for t in C1_INPUTS {
        let run = infer(&ws, &dir, "c", false, t);
        let alone = infer(&ws, &dir, "c", true, t);
        assert_eq!(
            run.code,
            Some(0),
            "{}",
            String::from_utf8_lossy(&run.stdout)
        );
        assert_eq!(run.stdout, alone.stdout);
        assert_eq!(run.stderr, "");
    }
    assert_eq!(
        connections.load(Ordering::SeqCst),
        0,
        "the legacy socket must not be used"
    );
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・#709: 新しい `serve` の起動で、残っていた古い名前のソケットが消える（ソケット以外は消さない）。
/// 古い常駐が動いている（ロックを持っている）間は、新しい `serve` は起動せず、古いソケットも消さない。
#[test]
fn req39_startup_removes_legacy_socket() {
    let (ws, dir) = (workspace("legrm"), serve_dir("legrm"));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).expect("chmod");
    let legacy = dir.join(LEGACY_SOCKET_FILE);
    // 持ち主のいない古いソケット（listener を落としてもファイルは残る）。
    drop(UnixListener::bind(&legacy).expect("bind"));
    assert!(legacy.exists());

    // 古い常駐に見立てて `serve.lock` を持っている間は、起動が断られ、古いソケットは残る。
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join("serve.lock"))
        .expect("lock file");
    lock.try_lock().expect("lock");
    let refused = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .args(["serve", "--package", "c"])
        .current_dir(&ws)
        .env("FANDHE_EDGE_SERVE_DIR", &dir)
        .stdin(Stdio::null())
        .output()
        .expect("run");
    assert_eq!(refused.status.code(), Some(64));
    assert!(
        String::from_utf8_lossy(&refused.stdout).contains("serve already running"),
        "{}",
        String::from_utf8_lossy(&refused.stdout)
    );
    assert!(legacy.exists());
    assert!(!dir.join(SOCKET_FILE).exists());
    drop(lock);

    let server = Server::start(&ws, &dir, "c", &[]);
    assert!(!legacy.exists(), "the legacy socket must be removed");
    assert!(dir.join(SOCKET_FILE).exists());
    assert_parity(&ws, &dir, "c", "zzz");
    // Drop が置き場所を消すので、先に常駐を止める。
    drop(server);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・REQ-21・#709: 起動が worker の `ready` 待ちより前で止まっても（`serve.lock` が読み手のいない FIFO で、
/// 書き込み専用の open が返らない）、起動全体の期限で `limit_exceeded` の JSON 1 つ・exit 20 で終わり、
/// ソケットを作らない。
#[test]
fn req39_startup_stuck_before_worker_hits_the_startup_limit() {
    let (ws, dir) = (workspace("fifo"), serve_dir("fifo"));
    fs::create_dir_all(&dir).expect("mkdir");
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).expect("chmod");
    let made = Command::new("mkfifo")
        .arg(dir.join("serve.lock"))
        .status()
        .expect("mkfifo");
    assert!(made.success());
    let started = Instant::now();
    let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .args(["serve", "--package", "c"])
        .current_dir(&ws)
        .env("FANDHE_EDGE_SERVE_DIR", &dir)
        .env("FANDHE_EDGE_SERVE_TEST_STARTUP_MS", "500")
        .stdin(Stdio::null())
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(20));
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "{\"code\":\"limit_exceeded\",\"message\":\"resource limit exceeded\"}\n"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("startup time limit exceeded; exiting"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // 種類別の上限（10 秒）ではなく、縮めた起動の期限で終わっている。
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
    assert!(!dir.join(SOCKET_FILE).exists());
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&ws);
}

/// 親が `wait` 以内に自分で終わるのを待ち、終了状態と stdout を返す。
fn wait_exit(server: &mut Server, wait: Duration) -> (std::process::ExitStatus, Vec<u8>) {
    let child = server.child.as_mut().expect("child");
    let deadline = Instant::now() + wait;
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            let mut out = Vec::new();
            child
                .stdout
                .take()
                .expect("stdout")
                .read_to_end(&mut out)
                .expect("read");
            return (status, out);
        }
        assert!(Instant::now() < deadline, "serve did not exit by itself");
        thread::sleep(Duration::from_millis(50));
    }
}

/// REQ-39・#727: `--idle-exit-secs` で要求が無ければ、親も worker も消え（exit 0・stdout 空）、
/// ソケットは消えるが `serve.lock` は残る。次の起動はすぐ取れる。
#[test]
fn req39_idle_exit_stops_parent_and_worker() {
    let (ws, dir) = (workspace("idle"), serve_dir("idle"));
    let mut server = Server::start_with_args(&ws, &dir, "c", &["--idle-exit-secs", "2"], &[]);
    let parent = server.child.as_ref().expect("child").id();
    let workers = child_pids(parent);
    assert_eq!(workers.len(), 1, "{workers:?}");
    let (status, stdout) = wait_exit(&mut server, Duration::from_secs(15));
    assert_eq!(status.code(), Some(0));
    assert!(stdout.is_empty(), "{stdout:?}");
    assert!(gone_within(parent, Duration::from_secs(5)));
    assert!(gone_within(workers[0], Duration::from_secs(5)));
    assert!(!dir.join(SOCKET_FILE).exists());
    assert!(dir.join("serve.lock").exists());
    // ロックとソケットがすぐ取り直せる。
    let again = Server::start(&ws, &dir, "c", &[]);
    let run = infer(&ws, &dir, "c", false, "hello world");
    assert!(run.stderr.contains("serve: delegated"), "{}", run.stderr);
    let log = again.stop();
    assert!(!log.contains("hello world"), "{log}");
    drop(server);
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・#727: 要求が来るたびに終了までの時間が延び、最後の要求から N 秒より前には終わらない。
#[test]
fn req39_idle_timer_extends_on_each_request() {
    let (ws, dir) = (workspace("idleext"), serve_dir("idleext"));
    let mut server = Server::start_with_args(&ws, &dir, "c", &["--idle-exit-secs", "4"], &[]);
    let line = encode_request(&c_request(&ws, "hello world"));
    let mut last = Instant::now();
    for _ in 0..4 {
        thread::sleep(Duration::from_millis(1500));
        let resp = raw_call(&dir, &line);
        last = Instant::now();
        assert!(
            matches!(resp, ServeResponse::Ok { exit_code: 0, .. }),
            "{resp:?}"
        );
        assert!(
            server
                .child
                .as_mut()
                .expect("child")
                .try_wait()
                .expect("try_wait")
                .is_none()
        );
    }
    let (status, _) = wait_exit(&mut server, Duration::from_secs(15));
    assert_eq!(status.code(), Some(0));
    assert!(
        last.elapsed() >= Duration::from_secs(4),
        "{:?}",
        last.elapsed()
    );
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・#727: 処理中の要求がある間はアイドルとみなさず、応答してから数え直す。
#[test]
fn req39_idle_exit_waits_for_inflight_request() {
    let (ws, dir) = (workspace("idleinfl"), serve_dir("idleinfl"));
    let mut server = Server::start_with_args(
        &ws,
        &dir,
        "c",
        &["--idle-exit-secs", "1"],
        &[("FANDHE_EDGE_SERVE_TEST_DELAY_MS", "3000")],
    );
    let line = encode_request(&c_request(&ws, "hello world"));
    let resp = raw_call(&dir, &line);
    assert!(
        matches!(resp, ServeResponse::Ok { exit_code: 0, .. }),
        "{resp:?}"
    );
    let (status, _) = wait_exit(&mut server, Duration::from_secs(15));
    assert_eq!(status.code(), Some(0));
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39・#727: 未指定なら終わらない。
#[test]
fn req39_without_idle_option_keeps_running() {
    let (ws, dir) = (workspace("noidle"), serve_dir("noidle"));
    let mut server = Server::start(&ws, &dir, "c", &[]);
    thread::sleep(Duration::from_secs(3));
    assert!(
        server
            .child
            .as_mut()
            .expect("child")
            .try_wait()
            .expect("try_wait")
            .is_none()
    );
    let run = infer(&ws, &dir, "c", false, "hello world");
    assert!(run.stderr.contains("serve: delegated"), "{}", run.stderr);
    let _ = fs::remove_dir_all(&ws);
}
