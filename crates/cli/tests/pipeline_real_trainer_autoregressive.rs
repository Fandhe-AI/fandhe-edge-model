#![cfg(unix)]
//! 実バイナリ `fandhe-edge` と実 trainer（MLX CPU）で、`autoregressive` を CLI の `train --kind autoregressive`
//! （候補番号 3）から学習し、`select`・`evaluate`・`package`・`infer` まで通す結合テスト（REQ-19b・REQ-33・
//! REQ-28・#697）。`infer` の予測が学習直後の validation 予測と一致することで、学習側（MLX）と推論ランタイム
//! （autoregressive の ONNX）の連鎖を確かめる。
//!
//! # 証拠の種別
//!
//! テストハーネス。学習データは本ファイルが生成する合成データのみ（3 ラベル・外部データ・通信なし）。CLI は
//! 設定を渡せないため既定設定（CPU）で学習するが、データが小さいので短時間で終わる。既定の設定での実機の
//! 速度・メモリ測定と推論上限の見直しは対象外（#697）。
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
const LABELS: [&str; 3] = ["alpha", "beta", "gamma"];

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

/// REQ-19b・REQ-33・#697: 実 trainer で `train --kind autoregressive`（候補 3）が完走し、`select`・`evaluate`・
/// `package` が通り、`infer --text` の予測ラベルが学習直後の validation 予測と一致する（REQ-28）。
#[test]
#[ignore = "requires trainer/.venv (make py-sync) and MLX CPU; run via make test-trainer-integration"]
fn req19b_issue697_real_trainer_autoregressive_trains_via_cli() {
    let work = Work(
        PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("real-autoregressive-{}", std::process::id())),
    );
    let _ = fs::remove_dir_all(&work.0);
    fs::create_dir_all(work.0.join("def")).expect("work");
    let work_dir = work.0.canonicalize().expect("canonical work");

    let options: Vec<String> = LABELS
        .iter()
        .map(|l| format!(r#"{{"id":"{l}","display_name":"{l}","description":"dummy"}}"#))
        .collect();
    fs::write(
        work_dir.join("def/definition.json"),
        format!(
            r#"{{"schema":"fandhe-edge-model-definition/v1","name":"real_ar","version":1,"judgment_type":"single_select","options":[{}],"io":{{"input":"bytes"}}}}"#,
            options.join(",")
        ),
    )
    .expect("definition");
    let rows = |prefix: &str, n: usize| {
        let mut out = String::new();
        for i in 0..n {
            for l in LABELS {
                out.push_str(&format!(
                    r#"{{"id":"{prefix}{l}-{i}","input":"{l} {l} {prefix}x{i}","output":{{"intent":"{l}"}},"group_id":"{prefix}g-{l}-{i}"}}"#
                ));
                out.push('\n');
            }
        }
        out
    };
    fs::write(work_dir.join("def/train.jsonl"), rows("", 20)).expect("train");
    fs::write(work_dir.join("def/evaluation.jsonl"), rows("e-", 4)).expect("evaluation");

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
                "3",
                "--kind",
                "autoregressive",
                "--device",
                "cpu",
            ],
        ),
        "{\"step\":\"train\",\"status\":\"ok\",\"candidate\":3,\"kind\":\"autoregressive\"}\n"
    );
    let result =
        fs::read_to_string(work_dir.join("proj/candidates/3/result.json")).expect("result");
    let label = string_after(&result, "\"validation_predictions\":[", "predicted_label");
    assert!(LABELS.contains(&label), "{label}");

    let selected = cli_ok(
        &work_dir,
        &[
            "select",
            "--project-dir",
            "proj",
            "--kind",
            "autoregressive",
        ],
    );
    assert!(
        selected.starts_with("{\"step\":\"select\",\"status\":\"ok\",\"candidate\":3,"),
        "{selected}"
    );
    let evaluated = cli_ok(
        &work_dir,
        &["evaluate", "--project-dir", "proj", "--candidate", "3"],
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
    // validation 予測の先頭 1 件（id から合成データの入力を復元できる）を `infer` し、学習直後の予測と比べる。
    let id = string_after(&result, "\"validation_predictions\":[", "id");
    let (label_name, index) = id.split_once('-').expect("synthetic id");
    let text = format!("{label_name} {label_name} x{index}");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"));
    cmd.args(["infer", "--package", "proj/package", "--text", &text])
        .current_dir(&work_dir);
    let (code, inferred) = run(cmd);
    assert!(matches!(code, Some(0 | 12)), "{code:?} {inferred}");
    assert_eq!(string_after(&inferred, "", "predicted_label"), label);
}
