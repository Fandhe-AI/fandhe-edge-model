//! 常駐プロセス `fandhe-edge serve` と、`infer --text` から委ねる経路（#708。REQ-21・REQ-28・REQ-33・
//! REQ-38・REQ-39。証拠種別: テストハーネス〔実バイナリ・合成パッケージ〕）。
//!
//! 共有 fixture の C1（`fixtures/onnx_parity`）と合成 `qwen2_lora`（`fixtures/qwen2_parity`）で組んだ
//! パッケージを使う。常駐は子プロセスで起動し、置き場所は `FANDHE_EDGE_SERVE_DIR` で `/tmp` 直下の
//! 短い一時ディレクトリへ向ける（macOS の `sun_path` は 104 バイトまで）。テストの終わりに必ず kill する。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use fandhe_edge_core::serve_protocol::{
    PackageIdentity, ServeRequest, ServeResponse, decode_response, encode_request,
};

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
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"));
        cmd.args(["serve", "--package", package])
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
        let sock = self.dir.join("serve.sock");
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
        id: "held".into(),
        text: "hello world".into(),
        package: c_identity(&ws),
    });
    let mut held = Vec::new();
    for i in 0..9 {
        let mut s = UnixStream::connect(dir.join("serve.sock")).expect("connect");
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

/// REQ-39: 読み込んだ後に shard を書き換えても、常駐の結果は変わらない（メモリ上の検証済みの内容で判定）。
#[test]
fn req39_shard_rewrite_after_load_does_not_change_result() {
    let (ws, dir) = (workspace("shard"), serve_dir("shard"));
    let _server = Server::start(&ws, &dir, "q", &[]);
    let before = infer(&ws, &dir, "q", false, QWEN2_INPUTS[0]);
    assert_eq!(before.code, Some(0));
    let shard = ws.join("q/weights-00000.bin");
    let len = fs::metadata(&shard).expect("meta").len();
    fs::write(&shard, vec![0u8; usize::try_from(len).expect("len")]).expect("rewrite");
    let after = infer(&ws, &dir, "q", false, QWEN2_INPUTS[0]);
    assert_eq!(after.stderr.trim_end(), "serve: delegated");
    assert_eq!(after.code, before.code);
    assert_eq!(after.stdout, before.stdout);
    // 単独なら書き換えを検出して拒否する（比較用。検証が常駐の起動時に移っていることの確認）。
    let alone = infer(&ws, &dir, "q", true, QWEN2_INPUTS[0]);
    assert_eq!(alone.code, Some(64));
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
    assert!(!dir.join("serve.sock").exists());
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
        assert_eq!(mode(&dir.join("serve.sock")), 0o600);
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
    assert!(!dir.join("serve.sock").exists());
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
    let sock = dir.join("serve.sock");
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
        let mut s = UnixStream::connect(dir.join("serve.sock")).expect("connect");
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
        id: "t1".into(),
        text: "x".into(),
        package: PackageIdentity {
            onnx_sha256: "0".repeat(64),
            shard_sha256: vec![],
            artifact_sha256: "0".repeat(64),
            definition_sha256: "0".repeat(64),
            calibration_sha256: None,
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
#[test]
fn serve_killed_while_judging_falls_back_to_standalone() {
    let (ws, dir) = (workspace("kill"), serve_dir("kill"));
    let server = Server::start(
        &ws,
        &dir,
        "c",
        &[("FANDHE_EDGE_SERVE_TEST_DELAY_MS", "5000")],
    );
    let run = thread::scope(|s| {
        let h = s.spawn(|| infer(&ws, &dir, "c", false, "hello world"));
        thread::sleep(Duration::from_millis(1500));
        let _ = server.stop();
        h.join().expect("join")
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
