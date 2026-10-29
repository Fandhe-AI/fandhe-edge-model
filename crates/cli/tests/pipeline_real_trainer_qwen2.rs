#![cfg(unix)]
//! 実バイナリ `fandhe-edge` と実 trainer（MLX CPU）で、`qwen2_lora` を CLI の `train --base-model-dir` から
//! 学習し、`select`・`evaluate`・`package`・`infer` まで通す結合テスト（REQ-41 正常系・REQ-33・REQ-39・REQ-28・
//! #628・#662）。`infer` の予測が学習直後の validation 予測と一致することで、学習側（MLX）と推論ランタイムの
//! 連鎖を確かめる。
//!
//! # 証拠の種別
//!
//! テストハーネス。ベースモデルは `trainer/tools/gen_qwen2_tiny_base_model.py` が書く合成の極小モデル
//! （hidden 16・2 層・vocab 300・seed 固定の乱数重み。`fixtures/qwen2_parity` の生成元と同じ）で、学習データも
//! 本ファイルが生成する合成データのみ（外部データ・事前学習済み重み・通信なし）。実重み（Qwen2.5 系）での
//! 学習・実機の容量・RSS・p95 は TASK-41.11（#442。人の担当）で、本テストはその証拠にならない。
//!
//! # 既定のテスト集合から分離する理由（`.claude/rules/ci.md`）
//!
//! 必要な環境（`make py-sync` で同期した `trainer/.venv` と MLX CPU）が `rust-ci` の runner に無い新規
//! テストで、既定集合から移したものではない。`make test-trainer-integration` が `--ignored --exact` で実行する。
//! テスト名を変えるときは `scripts/test-trainer-integration.sh` も更新すること。

use std::fs;
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// 子プロセス 1 つの上限時間（資源上限。REQ-39）。極小モデルの既定 config（iters 300）の学習を含む。
const TIMEOUT: Duration = Duration::from_secs(900);
const LABELS: [&str; 8] = ["red", "tan", "sky", "map", "fox", "owl", "gem", "ink"];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

/// テスト固有の作業ディレクトリ（Drop で削除）。CLI の cwd（経路の閉じ込め。REQ-39）。
struct Work(PathBuf);

impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// 子を独立プロセスグループで起動し、上限時間内に終了コードと stdout を回収する。超過時はグループごと KILL。
fn run(mut cmd: Command) -> (Option<i32>, String) {
    cmd.process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = cmd.spawn().expect("spawn");
    let mut so = child.stdout.take().expect("stdout");
    let pgid = child.id();
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        so.read_to_string(&mut s).ok();
        s
    });
    let start = Instant::now();
    let status = loop {
        if let Some(st) = child.try_wait().expect("try_wait") {
            break st;
        }
        if start.elapsed() > TIMEOUT {
            Command::new("kill")
                .args(["-KILL", &format!("-{pgid}")])
                .status()
                .ok();
            child.kill().ok();
            child.wait().ok();
            reader.join().ok();
            panic!("child timed out");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    (status.code(), reader.join().expect("join"))
}

/// `fandhe-edge <args>` を cwd = `work` で実行し、exit 0 を確かめて stdout を返す。
fn cli_ok(work: &Path, args: &[&str]) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"));
    cmd.args(args).current_dir(work);
    let (code, stdout) = run(cmd);
    assert_eq!(code, Some(0), "args={args:?} stdout={stdout}");
    stdout
}

/// `"key":"<value>"` の値を `from` 以降から取り出す（キーと値は英数字と記号のみの合成データ）。
fn string_after<'a>(json: &'a str, from: &str, key: &str) -> &'a str {
    let rest = json.split_once(from).expect("anchor").1;
    let marker = format!("\"{key}\":\"");
    let rest = rest.split_once(&marker).expect("key").1;
    rest.split_once('"').expect("value end").0
}

/// REQ-41・REQ-33・REQ-39・#628: 実 trainer で `train --kind qwen2_lora --base-model-dir` が完走し（cwd 配下の
/// ベースモデルを候補ディレクトリの外の絶対パスとして読む）、`select`・`evaluate`・`package` が通り、
/// `infer --text` の予測ラベルが学習直後の validation 予測と一致する（REQ-28）。
#[test]
#[ignore = "requires trainer/.venv (make py-sync) and MLX CPU; run via make test-trainer-integration"]
fn req41_issue628_real_trainer_qwen2_lora_trains_via_cli() {
    let root = repo_root();
    let work = Work(
        PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("real-qwen2-{}", std::process::id())),
    );
    let _ = fs::remove_dir_all(&work.0);
    fs::create_dir_all(work.0.join("def")).expect("work");
    let work_dir = work.0.canonicalize().expect("canonical work");

    // 合成の極小ベースモデルを cwd 配下へ書き、sha256 の写像（1 行の JSON）を定義へそのまま埋める。
    let mut gen_cmd = Command::new(root.join("trainer/.venv/bin/python"));
    gen_cmd
        .arg("-I")
        .arg(root.join("trainer/tools/gen_qwen2_tiny_base_model.py"))
        .arg("--out")
        .arg(work_dir.join("bm"))
        .env_clear();
    let (code, sha256) = run(gen_cmd);
    assert_eq!(code, Some(0), "gen_qwen2_tiny_base_model.py: {sha256}");
    let options: Vec<String> = LABELS
        .iter()
        .map(|l| format!(r#"{{"id":"{l}","display_name":"{l}","description":"dummy"}}"#))
        .collect();
    fs::write(
        work_dir.join("def/definition.json"),
        format!(
            r#"{{"schema":"fandhe-edge-model-definition/v1","name":"real_qwen2","version":1,"judgment_type":"single_select","options":[{}],"io":{{"input":"bytes"}},"base_model":{{"name":"tiny-synthetic","sha256":{},"license":"MIT","origin":"https://example.com/tiny"}}}}"#,
            options.join(","),
            sha256.trim()
        ),
    )
    .expect("definition");
    let rows = |prefix: &str, n: usize| {
        let mut out = String::new();
        for i in 0..n {
            for l in LABELS {
                out.push_str(&format!(
                    r#"{{"id":"{prefix}{l}-{i}","input":"{l}{} {l}{} {prefix}x{i}","output":{{"intent":"{l}"}},"group_id":"{prefix}g-{l}-{i}"}}"#,
                    i % 4,
                    (i + 1) % 4
                ));
                out.push('\n');
            }
        }
        out
    };
    fs::write(work_dir.join("def/train.jsonl"), rows("", 10)).expect("train");
    fs::write(work_dir.join("def/evaluation.jsonl"), rows("e-", 2)).expect("evaluation");

    cli_ok(
        &work_dir,
        &[
            "register",
            "--definition",
            "def/definition.json",
            "--project-dir",
            "proj",
        ],
    );
    cli_ok(&work_dir, &["inspect", "--project-dir", "proj"]);
    assert_eq!(
        cli_ok(
            &work_dir,
            &[
                "train",
                "--project-dir",
                "proj",
                "--candidate",
                "2",
                "--kind",
                "qwen2_lora",
                "--base-model-dir",
                "bm",
            ],
        ),
        "{\"step\":\"train\",\"status\":\"ok\",\"candidate\":2,\"kind\":\"qwen2_lora\"}\n"
    );
    // 実 trainer が絶対パスのベースモデルを読み（sha256 照合済み）、validation 予測まで返した。
    let result =
        fs::read_to_string(work_dir.join("proj/candidates/2/result.json")).expect("result");
    let label = string_after(&result, "\"validation_predictions\":[", "predicted_label");
    assert!(LABELS.contains(&label), "{label}");
    assert!(
        work_dir
            .join("proj/candidates/2/model-qwen2_lora/model.onnx")
            .is_file()
    );

    // `select`・`evaluate` は tokenizer 入力表現（`qwen2_lora`）に接続済み（#662-a）。実 trainer の自己完結
    // パッケージ（tokenizer.json・外部データ shard）で、構造検査・校正・凍結 test への適用が通る。
    assert_eq!(
        cli_ok(
            &work_dir,
            &["select", "--project-dir", "proj", "--kind", "qwen2_lora"],
        ),
        "{\"step\":\"select\",\"status\":\"ok\",\"candidate\":2,\"kind\":\"qwen2_lora\",\"significance\":null}\n"
    );
    let evaluated = cli_ok(
        &work_dir,
        &["evaluate", "--project-dir", "proj", "--candidate", "2"],
    );
    assert!(
        evaluated.starts_with("{\"step\":\"evaluate\",\"status\":\"ok\""),
        "{evaluated}"
    );
    let packaged = cli_ok(&work_dir, &["package", "--project-dir", "proj"]);
    assert!(
        packaged.starts_with("{\"step\":\"package\",\"status\":\"ok\""),
        "{packaged}"
    );
    for f in ["tokenizer.json", "model.onnx", "artifact.json"] {
        assert!(work_dir.join("proj/package").join(f).is_file(), "{f}");
    }
    // validation 予測の先頭 1 件（id から合成データの入力を復元できる）を配布パッケージへ `infer` し、
    // 学習直後の予測ラベルと一致することを確かめる（学習側 MLX ＝ 推論ランタイム。REQ-28・REQ-41）。
    // 評価で校正が同梱されると棄権の行は exit 12 になるため、終了コードではなく出力行のラベルを見る。
    let anchor = "\"validation_predictions\":[";
    let id = string_after(&result, anchor, "id");
    let (label_name, index) = id.split_once('-').expect("synthetic id");
    let i: usize = index.parse().expect("row index");
    let text = format!("{label_name}{} {label_name}{} x{i}", i % 4, (i + 1) % 4);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"));
    cmd.args(["infer", "--package", "proj/package", "--text", &text])
        .current_dir(&work_dir);
    let (code, inferred) = run(cmd);
    assert!(matches!(code, Some(0 | 12)), "{code:?} {inferred}");
    assert_eq!(string_after(&inferred, "", "predicted_label"), label);
}
