#![cfg(unix)]
//! 実際の `trainer/`（MLX・C1/C3・ONNX 書き出し）を `WorkerLauncher`・
//! `run_train` 経由で起動し、学習ジョブ 1 件を完走させて結果 JSON を
//! [`TrainOutcome`] として受け取れることを確認する結合テスト
//! （issue #258・親 #176 受け入れ条件 1。REQ-18・REQ-19・REQ-34・REQ-39）。
//!
//! # 偽ワーカー（`worker_process.rs`）との役割分担
//!
//! `worker_process.rs`（#178）は MLX 不要の偽ワーカーで子プロセス制御を
//! 検証する。本ファイルは実 trainer を最後まで動かす唯一の Rust 側テストで、
//! 成功時の結果 JSON が実際の学習・ONNX 書き出しから得られることを確認する。
//!
//! # 既定のテスト集合から分離する理由（`.claude/rules/ci.md`）
//!
//! - 既定集合から移したテストではなく新規テスト。必要な環境（`make py-sync`
//!   で同期した `trainer/.venv` と MLX）が `rust-ci` の 3 OS runner に無い。
//! - `#[ignore]` は CI を通すためのものではない。`make test-trainer-integration`
//!   が `--ignored` で実行し、`python-ci`（macos-14 arm64）とローカルの
//!   `make ci` で実際に走らせる。silent skip はせず、libtest が `ignored` と
//!   明示的に報告する。
//! - GPU・実機測定を伴わない（`Device::Cpu`・極小設定）ため実機前提テストではない。
//!
//! # 証拠の種別
//!
//! テストハーネス（合成データ・CPU）。性能・容量の実機測定ではない。
//!
//! # 入力データ
//!
//! 学習データ・validation 入力は本ファイルが生成する合成データのみで、個人情報・
//! 機密を含まない。失敗診断のため trainer の stderr を表示するのは、この合成
//! データ由来に限った意図的な例外（`security.md` のデータ本文転記禁止は実データ対象）。
//!
//! # 環境変数
//!
//! `run_train` は子の環境を `env_clear()` する（`ENV_ALLOWLIST` のみ通す。REQ-39）。
//! 本テストは許可リストの拡張・環境の迂回をしない。
//!
//! # テスト名と Makefile の同期
//!
//! `Makefile` の `test-trainer-integration` は `--exact` でテスト名を列挙し、
//! 実行件数を検査する。テスト名を変えるときは Makefile も更新すること。

use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_train::process::{RunLimits, WorkerLauncher, run_train};
use fandhe_edge_train::request::{Device, TrainRequest, TrainRequestParams, ValidationInput};
use fandhe_edge_train::result::{
    OutputType, SuccessOutcome, TrainOutcome, ValidationPredictionStatus,
};

/// 合成データのラベル順（`label_order`）。
const LABELS: [&str; 2] = ["cat_a", "cat_b"];
/// ONNX ファイルの sha256 再計算時の読み込み上限（REQ-39。極小モデルには十分）。
const ONNX_READ_LIMIT: u64 = 64 * 1024 * 1024;

/// 実 trainer を指すランチャー。`.venv` が無ければ `make py-sync` を案内して
/// panic する（明示的に実行を要求されたテストなので fail-closed）。
fn trainer_launcher() -> WorkerLauncher {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("trainer");
    let dir = std::fs::canonicalize(&dir)
        .unwrap_or_else(|e| panic!("trainer dir not found ({e}); run from the repository"));
    WorkerLauncher::from_trainer_dir(&dir).unwrap_or_else(|e| {
        panic!("trainer venv is not ready ({e:?}); run `make py-sync` first (issue #258)")
    })
}

/// テストごとの作業ディレクトリ（Drop で削除）。`root/` は trainer の
/// `out_dir` 親ディレクトリ検査（所有者のみ書き込み可）を満たすため mode 0700
/// で作る（umask に依存させない）。
struct CaseDir {
    base: PathBuf,
}

impl CaseDir {
    fn new(case: &str) -> Self {
        let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("real-trainer-{case}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("create case dir");
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(base.join("root"))
            .expect("create root");
        std::fs::set_permissions(base.join("root"), std::fs::Permissions::from_mode(0o700))
            .expect("chmod root");
        std::fs::create_dir(base.join("job")).expect("create job dir");
        Self { base }
    }

    fn root(&self) -> PathBuf {
        self.base.join("root")
    }

    fn job(&self) -> PathBuf {
        self.base.join("job")
    }
}

impl Drop for CaseDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

/// 合成 2 クラスの学習データ（`trainer/tests/conftest.py` と同じ形。24 行）。
fn write_train_jsonl(root: &Path) {
    let mut out = String::new();
    for i in 0..12 {
        for (label, text) in [
            ("cat_a", format!("alpha alpha beta gamma {i}")),
            ("cat_b", format!("delta delta epsilon zeta {i}")),
        ] {
            out.push_str(&serde_json::json!({"input": text, "label": label}).to_string());
            out.push('\n');
        }
    }
    std::fs::write(root.join("train.jsonl"), out).expect("write train.jsonl");
}

fn make_request(kind: &str, config: serde_json::Value, root: &Path) -> TrainRequest {
    let serde_json::Value::Object(config) = config else {
        panic!("config must be an object");
    };
    TrainRequest::new(TrainRequestParams {
        kind: kind.to_string(),
        kind_version: 1,
        config,
        label_order: LABELS.iter().map(|s| s.to_string()).collect(),
        max_bytes: 64,
        seed: 0,
        device: Device::Cpu,
        root: root.to_str().expect("utf-8 root").to_string(),
        train_path: "train.jsonl".to_string(),
        out_dir: "out".to_string(),
        time_limit_seconds: Some(120),
        rss_limit_bytes: None,
    })
    .expect("valid train request")
}

/// 失敗時に trainer の stderr（合成データ由来のみ）を添えて panic する。
fn run_ok(case: &CaseDir, request: &TrainRequest) -> SuccessOutcome {
    let launcher = trainer_launcher();
    let limits = RunLimits::for_request(request);
    let run = run_train(&launcher, request, &case.job(), &limits)
        .unwrap_or_else(|e| panic!("run_train failed: {e:?}"));
    let stderr = run.worker_stderr();
    let stderr = String::from_utf8_lossy(&stderr[..stderr.len().min(4096)]).into_owned();
    assert_eq!(run.exit_code(), ExitCode::Ok, "worker stderr: {stderr}");
    match run.outcome() {
        TrainOutcome::Ok(success) => success.clone(),
        TrainOutcome::Error(f) => panic!("worker error {}: {} / {stderr}", f.code(), f.message()),
    }
}

/// C1・C3 共通の成功結果の検証。
fn assert_common(case: &CaseDir, success: &SuccessOutcome, kind: &str, epochs: u64) {
    let artifact = success.artifact();
    assert_eq!(artifact.kind(), kind);
    assert_eq!(artifact.candidate_label(), kind);
    assert_eq!(artifact.kind_version(), 1);
    assert_eq!(
        artifact.config().get("epochs"),
        Some(&serde_json::json!(epochs))
    );
    assert_eq!(artifact.label_order().as_slice(), &LABELS);
    assert_eq!(artifact.output_type(), OutputType::Choice);
    assert_eq!(artifact.max_bytes(), 64);
    assert_eq!(artifact.onnx_file(), "model.onnx");

    let root = std::fs::canonicalize(case.root()).expect("canonicalize root");
    let out = root.join("out");
    assert_eq!(Path::new(success.artifact_dir()), out.as_path());
    let onnx = out.join("model.onnx");
    assert!(out.join("artifact.json").is_file());
    assert!(onnx.is_file());
    // 完全性（REQ-39）: 結果 JSON の sha256 と実ファイルの再計算値が一致する。
    let digest = fandhe_edge_core::fs::sha256_file_bounded(&onnx, ONNX_READ_LIMIT)
        .expect("sha256 of model.onnx");
    assert_eq!(digest.to_hex(), artifact.onnx_sha256().as_str());

    assert!(!case.job().join("request.json").exists());
    let leftovers: Vec<_> = std::fs::read_dir(&root)
        .expect("read root")
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with(".out.tmp-"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temporary out dirs left: {leftovers:?}"
    );
}

#[test]
#[ignore = "requires trainer/.venv (make py-sync) and MLX CPU; run via make test-trainer-integration (issue #258, REQ-18/19/34/39)"]
fn req18_real_trainer_c1_job_completes_with_typed_outcome() {
    let case = CaseDir::new("c1");
    write_train_jsonl(&case.root());
    // trainer/tests/conftest.py の TINY_C1_CONFIG と同じ極小設定。
    let config = serde_json::json!({
        "ngram_min": 1, "ngram_max": 3, "min_df": 1, "max_features": 1000,
        "C": 1.0, "epochs": 40, "batch_size": 8, "lr": 0.5
    });
    let request = make_request("c1", config, &case.root());
    let success = run_ok(&case, &request);
    assert_common(&case, &success, "c1", 40);
    assert!(success.validation_predictions().is_none());
}

#[test]
#[ignore = "requires trainer/.venv (make py-sync) and MLX CPU; run via make test-trainer-integration (issue #258, REQ-18/19/34/39)"]
fn req18_real_trainer_c3_job_completes_with_validation_predictions() {
    let case = CaseDir::new("c3");
    write_train_jsonl(&case.root());
    // trainer/tests/conftest.py の TINY_CONFIG と同じ極小設定。
    let config = serde_json::json!({
        "epochs": 5, "batch_size": 8, "emb": 8, "filters": 8,
        "widths": [3, 5, 7], "dropout": 0.0
    });
    // gold（正解ラベル）は渡さない（REQ-27）。
    let request = make_request("c3", config, &case.root())
        .with_validation_inputs(vec![
            ValidationInput::new("v1".to_string(), "alpha alpha beta gamma 99".to_string()),
            ValidationInput::new("v2".to_string(), "delta delta epsilon zeta 99".to_string()),
        ])
        .expect("valid validation inputs");
    let success = run_ok(&case, &request);
    assert_common(&case, &success, "c3", 5);

    let preds = success
        .validation_predictions()
        .expect("validation predictions");
    let ids: Vec<&str> = preds.iter().map(|p| p.id()).collect();
    assert_eq!(ids, ["v1", "v2"]);
    for p in preds {
        assert_eq!(p.status(), ValidationPredictionStatus::Ok);
        // 予測の正誤は精度の主張になるので断言せず、ラベル集合への所属だけ確認する。
        let label = p.predicted_label().expect("predicted label");
        assert!(LABELS.contains(&label), "unexpected label {label}");
    }
}
