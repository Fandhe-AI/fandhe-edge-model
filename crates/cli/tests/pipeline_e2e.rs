//! 7 工程（`register → inspect → train → evaluate → select → package → infer`）の完走確認
//! （REQ-33 正常系・TASK-33.1-2・#136）。実バイナリ `fandhe-edge` を非対話で順に実行する。
//!
//! # 証拠の種別
//!
//! テストハーネス。学習ワーカーは MLX 不要の**偽ワーカー**（本ファイル自身。下記）で、ONNX は
//! 共有 fixture（`fixtures/onnx_parity/{c1,c3}.onnx`。生成元は同 `PROVENANCE.md`）をそのまま
//! 配置する。学習・validation 予測の中身は実 trainer のものではない（実 trainer での完走は
//! `pipeline_real_trainer.rs`〔`#[ignore]`・`make test-trainer-integration`〕）。学習データは
//! 本ファイルが生成する合成データのみで、個人情報・機密を含まない。
//!
//! # なぜ `harness = false` の単一バイナリか
//!
//! `crates/train/tests/worker_process.rs` と同じ方式。本ファイル自身を学習ワーカーの `python`
//! として渡し（`<trainer>/.venv/bin/python3` を本バイナリへの symlink にする）、`main()` が
//! argv で「偽ワーカー」か「テストランナー」かを分岐する。`run_train` は子の環境を
//! `env_clear()` するため、環境変数では挙動を渡せない。
//!
//! # 偽ワーカーの挙動
//!
//! 学習リクエストを検証つきで読み、`<root>/<out_dir>` に kind ごとの fixture ONNX と `artifact.json`
//! を置き、結果 JSON を 1 行出す。validation 予測は c1 が常に `alpha`、c3 が入力の先頭の語
//! （合成データの正解ラベルと一致）とする（正解率に差を付けて選定を確認するため）。

#[cfg(unix)]
mod suite {
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use fandhe_edge_core::definition::Definition;
    use fandhe_edge_core::hash::Sha256Digest;
    use fandhe_edge_train::request::TrainRequest;

    const LABELS: [&str; 3] = ["alpha", "beta", "gamma"];

    /// `c1`・`c3` の既定設定（`fixtures/train_contract/kind_defaults.json` と同じ具体値。
    /// 結果の `config` は「既定値を要求の設定で上書きしたもの」と完全一致する必要がある）。
    const C1_CONFIG: &str = r#"{"ngram_min":1,"ngram_max":4,"min_df":2,"max_features":200000,"C":1.0,"epochs":30,"batch_size":64,"lr":0.5}"#;
    const C3_CONFIG: &str = r#"{"lr":0.001,"weight_decay":0.0001,"epochs":40,"batch_size":64,"emb":64,"filters":128,"widths":[3,5,7],"dropout":0.3}"#;

    fn definition_text() -> String {
        let options: Vec<String> = LABELS
            .iter()
            .map(|l| format!(r#"{{"id":"{l}","display_name":"{l}","description":"dummy"}}"#))
            .collect();
        format!(
            r#"{{"schema":"fandhe-edge-model-definition/v1","name":"pipeline_e2e","version":1,"judgment_type":"single_select","options":[{}],"io":{{"input":"bytes"}}}}"#,
            options.join(",")
        )
    }

    /// 合成データ（ラベルごとに 30 件。group_id・input はすべて異なる）。
    fn train_jsonl() -> String {
        let mut out = String::new();
        for i in 0..30 {
            for l in LABELS {
                out.push_str(&format!(
                    r#"{{"id":"{l}-{i}","input":"{l} sample {i}","output":{{"intent":"{l}"}},"group_id":"g-{l}-{i}"}}"#
                ));
                out.push('\n');
            }
        }
        out
    }

    /// 独立した評価データ（train と input・group_id が重ならない合成データ）。
    fn evaluation_jsonl() -> String {
        let mut out = String::new();
        for i in 0..4 {
            for l in LABELS {
                out.push_str(&format!(
                    r#"{{"id":"e-{l}-{i}","input":"{l} evaluation {i}","output":{{"intent":"{l}"}},"group_id":"eg-{l}-{i}"}}"#
                ));
                out.push('\n');
            }
        }
        out
    }

    /// テストごとの作業ディレクトリ（Drop で削除）。`work/` が CLI の cwd。
    pub struct Env {
        base: PathBuf,
        work: PathBuf,
        trainer: PathBuf,
    }

    impl Env {
        fn new(case: &str, with_evaluation: bool) -> Self {
            let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
                .join(format!("pipeline-e2e-{case}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&base);
            let work = base.join("work");
            let trainer = base.join("trainer");
            std::fs::create_dir_all(work.join("def")).expect("work dir");
            std::fs::create_dir_all(trainer.join(".venv").join("bin")).expect("trainer dir");
            std::fs::write(trainer.join("launch.py"), "fake").expect("launch.py");
            let exe = std::env::current_exe().expect("current exe");
            std::os::unix::fs::symlink(exe, trainer.join(".venv").join("bin").join("python3"))
                .expect("symlink python3");
            std::fs::write(work.join("def").join("definition.json"), definition_text())
                .expect("definition");
            std::fs::write(work.join("def").join("train.jsonl"), train_jsonl()).expect("train");
            if with_evaluation {
                std::fs::write(
                    work.join("def").join("evaluation.jsonl"),
                    evaluation_jsonl(),
                )
                .expect("evaluation");
            }
            Self {
                base,
                work,
                trainer,
            }
        }

        /// CLI を cwd = `work/` で実行する（学習ワーカーは偽の trainer ディレクトリを指す）。
        pub fn cli(&self, args: &[&str]) -> Output {
            Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
                .args(args)
                .current_dir(&self.work)
                .env("FANDHE_EDGE_TRAINER_DIR", &self.trainer)
                .output()
                .expect("run fandhe-edge")
        }

        /// 実行して終了コードと stdout（1 呼び出し 1 JSON のため末尾改行つきの 1 行）を返す。
        fn run(&self, args: &[&str]) -> (i32, String) {
            let out = self.cli(args);
            (
                out.status.code().expect("exit code"),
                String::from_utf8(out.stdout).expect("utf8 stdout"),
            )
        }

        /// 成功（exit 0）を期待し、stdout を返す。
        fn ok(&self, args: &[&str]) -> String {
            let (code, stdout) = self.run(args);
            assert_eq!(code, 0, "args={args:?} stdout={stdout}");
            assert_eq!(stdout.matches('\n').count(), 1, "one JSON line: {stdout}");
            stdout
        }

        /// 失敗を期待し、終了コードと `code` 値の JSON 先頭を確認する。
        fn fails(&self, args: &[&str], exit: i32, code_name: &str) -> String {
            let (code, stdout) = self.run(args);
            assert_eq!(code, exit, "args={args:?} stdout={stdout}");
            assert!(
                stdout.starts_with(&format!(r#"{{"code":"{code_name}","#)),
                "stdout={stdout}"
            );
            stdout
        }

        fn project_file(&self, rel: &str) -> PathBuf {
            self.work.join("proj").join(rel)
        }
    }

    impl Drop for Env {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.base);
        }
    }

    const DEF: &str = "def/definition.json";

    fn registered(case: &str, with_evaluation: bool) -> Env {
        let env = Env::new(case, with_evaluation);
        env.ok(&["register", "--definition", DEF, "--project-dir", "proj"]);
        env
    }

    fn inspected(case: &str) -> Env {
        let env = registered(case, false);
        env.ok(&["inspect", "--project-dir", "proj"]);
        env
    }

    /// REQ-33: 評価データなしで 7 工程がすべて exit 0 で完走し、各出力が JSON 1 つ。
    pub fn full_pipeline_completes_without_evaluation_data() {
        let env = Env::new("full", false);
        let def = Definition::parse(&definition_text()).expect("definition");
        let hash = def.canonical_hash().expect("hash").to_hex();

        assert_eq!(
            env.ok(&["register", "--definition", DEF, "--project-dir", "proj"]),
            format!(
                "{{\"step\":\"register\",\"status\":\"ok\",\"definition_sha256\":\"{hash}\",\"options\":3,\"evaluation_defined\":false}}\n"
            )
        );
        let inspect = env.ok(&["inspect", "--project-dir", "proj"]);
        assert_eq!(
            inspect,
            "{\"step\":\"inspect\",\"status\":\"ok\",\"valid_records\":90,\"split\":{\"train\":72,\"validation\":9,\"test\":9}}\n"
        );
        assert_eq!(
            env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]),
            "{\"step\":\"train\",\"status\":\"ok\",\"candidate\":0,\"kind\":\"c1\"}\n"
        );
        assert_eq!(
            env.ok(&["train", "--project-dir", "proj", "--candidate", "1"]),
            "{\"step\":\"train\",\"status\":\"ok\",\"candidate\":1,\"kind\":\"c3\"}\n"
        );
        // 評価データ未定義: skipped・exit 0（評価済みを装わない。REQ-17）。
        assert_eq!(
            env.ok(&["evaluate", "--project-dir", "proj", "--candidate", "0"]),
            "{\"step\":\"evaluate\",\"status\":\"skipped\",\"reason\":\"evaluation_data_not_defined\"}\n"
        );
        // c3（先頭の語で予測する偽ワーカー）が c1（常に alpha）より validation 正解率が高い。
        assert_eq!(
            env.ok(&["select", "--project-dir", "proj"]),
            "{\"step\":\"select\",\"status\":\"ok\",\"candidate\":1,\"kind\":\"c3\"}\n"
        );
        assert_eq!(
            env.ok(&["package", "--project-dir", "proj"]),
            "{\"step\":\"package\",\"status\":\"ok\",\"judgment\":null,\"acceptance_defined\":false}\n"
        );

        let text = env.ok(&[
            "infer",
            "--package",
            "proj/package",
            "--text",
            "alpha sample 1",
        ]);
        assert!(
            text.starts_with("{\"id\":\"input\",\"status\":\"ok\",\"predicted_label\":\""),
            "{text}"
        );
        let predicted = text
            .split("\"predicted_label\":\"")
            .nth(1)
            .and_then(|r| r.split('"').next())
            .expect("predicted label");
        assert!(LABELS.contains(&predicted), "{predicted}");

        std::fs::write(
            env.work.join("batch.jsonl"),
            "{\"id\":\"a\",\"input\":\"alpha sample 1\"}\n{\"id\":\"b\",\"input\":\"gamma sample 2\"}\n",
        )
        .expect("batch");
        let (code, batch) = env.run(&[
            "infer",
            "--package",
            "proj/package",
            "--input-file",
            "batch.jsonl",
        ]);
        assert_eq!(code, 0, "{batch}");
        let lines: Vec<&str> = batch.lines().collect();
        assert_eq!(lines.len(), 2, "{batch}");
        assert!(
            lines[0].starts_with("{\"id\":\"a\",\"status\":\"ok\""),
            "{batch}"
        );
        assert!(
            lines[1].starts_with("{\"id\":\"b\",\"status\":\"ok\""),
            "{batch}"
        );
        // 単体推論とバッチ推論は同じ入力で同じ判定になる（REQ-28）。
        let single = env.ok(&[
            "infer",
            "--package",
            "proj/package",
            "--text",
            "alpha sample 1",
            "--id",
            "a",
        ]);
        assert_eq!(single.trim_end(), lines[0]);
    }

    /// REQ-17・REQ-33: 評価データがあると凍結・読み取り専用配置され、`evaluate` は本体が未実装のため
    /// `runtime_error`（評価済みを装わない）。
    pub fn evaluation_data_is_frozen_and_evaluate_is_not_faked() {
        let env = registered("eval", true);
        assert!(env.project_file("eval_freeze.json").is_file());
        let eval_file = env.project_file("data/evaluation.jsonl");
        let mode = std::os::unix::fs::PermissionsExt::mode(
            &std::fs::metadata(&eval_file)
                .expect("eval meta")
                .permissions(),
        );
        assert_eq!(
            mode & 0o222,
            0,
            "evaluation data must be read-only: {mode:o}"
        );
        env.ok(&["inspect", "--project-dir", "proj"]);
        let out = env.fails(
            &["evaluate", "--project-dir", "proj", "--candidate", "0"],
            70,
            "runtime_error",
        );
        assert_eq!(
            out,
            "{\"code\":\"runtime_error\",\"message\":\"evaluation on frozen data is not implemented yet\"}\n"
        );
    }

    /// REQ-17: 凍結記録と評価データが食い違うと `evaluate` は停止する（fail-closed）。
    pub fn evaluate_stops_on_frozen_hash_mismatch() {
        let env = registered("evalhash", true);
        // 読み取り専用配置を外して差し替える（実運用では書き込みは拒否される。ここは改ざんの模擬）。
        let eval_file = env.project_file("data/evaluation.jsonl");
        let mut perm = std::fs::metadata(&eval_file).expect("meta").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o600);
        std::fs::set_permissions(&eval_file, perm).expect("chmod");
        std::fs::write(&eval_file, evaluation_jsonl().replace("alpha", "gamma")).expect("tamper");
        env.fails(
            &["evaluate", "--project-dir", "proj", "--candidate", "0"],
            64,
            "invalid_input",
        );
    }

    /// REQ-17: 凍結記録が無いまま空の評価データがあっても `inspect` は拒否する（fail-closed）。
    pub fn inspect_rejects_empty_evaluation_data_without_freeze_record() {
        let env = registered("evalempty", false);
        std::fs::write(env.project_file("data/evaluation.jsonl"), "").expect("write empty");
        env.fails(&["inspect", "--project-dir", "proj"], 64, "invalid_input");
    }

    /// REQ-33: 前工程が済んでいない場合は次工程が失敗する（固定メッセージ・パスを含まない）。
    pub fn stages_require_their_predecessors() {
        let env = registered("order", false);
        env.fails(
            &["train", "--project-dir", "proj", "--candidate", "0"],
            64,
            "invalid_input",
        );
        env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input");
        env.ok(&["inspect", "--project-dir", "proj"]);
        // 学習済みの候補が無い select は pending（12）。
        env.fails(&["select", "--project-dir", "proj"], 12, "pending");
        env.fails(
            &["train", "--project-dir", "proj", "--candidate", "9"],
            64,
            "invalid_input",
        );
    }

    /// REQ-39: 既存の `--project-dir` への register・cwd 外の参照は拒否する。
    pub fn register_rejects_existing_and_escaping_paths() {
        let env = registered("guard", false);
        env.fails(
            &["register", "--definition", DEF, "--project-dir", "proj"],
            64,
            "invalid_input",
        );
        env.fails(
            &[
                "register",
                "--definition",
                DEF,
                "--project-dir",
                "../escaped",
            ],
            64,
            "invalid_input",
        );
        env.fails(
            &[
                "register",
                "--definition",
                "/etc/hostname",
                "--project-dir",
                "p2",
            ],
            64,
            "invalid_input",
        );
        env.fails(
            &["inspect", "--project-dir", "../proj"],
            64,
            "invalid_input",
        );
    }

    /// REQ-39: データに異常（重複 id 等）があると inspect が停止し、分割記録を残さない。
    pub fn inspect_rejects_invalid_records() {
        let env = Env::new("badrecords", false);
        std::fs::write(
            env.work.join("def").join("train.jsonl"),
            "{\"id\":\"a\",\"input\":\"x\",\"output\":{\"intent\":\"alpha\"},\"group_id\":\"g\"}\nnot json\n",
        )
        .expect("bad data");
        env.ok(&["register", "--definition", DEF, "--project-dir", "proj"]);
        let out = env.fails(&["inspect", "--project-dir", "proj"], 64, "invalid_input");
        assert!(!out.contains("not json"), "{out}");
        assert!(!env.project_file("split.json").exists());
    }

    fn packaged(case: &str) -> Env {
        let env = inspected(case);
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        env.ok(&["select", "--project-dir", "proj"]);
        env.ok(&["package", "--project-dir", "proj"]);
        env
    }

    /// REQ-32・REQ-39: パッケージの記載（sha256・label_order）と実体が食い違うと infer は拒否する。
    pub fn infer_rejects_tampered_package() {
        let env = packaged("tamper");
        let meta = env.project_file("package/artifact.json");
        let original = std::fs::read_to_string(&meta).expect("artifact.json");
        let onnx_hash = Sha256Digest::of_bytes(
            &std::fs::read(env.project_file("package/model.onnx")).expect("onnx"),
        )
        .to_hex();
        let text_args = ["infer", "--package", "proj/package", "--text", "alpha"];

        std::fs::write(&meta, original.replace(&onnx_hash, &"0".repeat(64))).expect("tamper sha");
        assert_eq!(
            env.fails(&text_args, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"package model does not match its recorded hash\"}\n"
        );
        std::fs::write(
            &meta,
            original.replace(
                "\"label_order\":[\"alpha\",\"beta\",\"gamma\"]",
                "\"label_order\":[\"beta\",\"alpha\",\"gamma\"]",
            ),
        )
        .expect("tamper labels");
        assert_eq!(
            env.fails(&text_args, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"package label order does not match definition\"}\n"
        );
        std::fs::write(&meta, &original).expect("restore");
        env.ok(&text_args);
    }

    /// REQ-33: 未実装の `infer --out` は成功を装わず `runtime_error`。
    pub fn infer_out_option_is_not_faked() {
        let env = packaged("out");
        std::fs::write(
            env.work.join("in.jsonl"),
            "{\"id\":\"a\",\"input\":\"x\"}\n",
        )
        .expect("in");
        assert_eq!(
            env.fails(
                &[
                    "infer",
                    "--package",
                    "proj/package",
                    "--input-file",
                    "in.jsonl",
                    "--out",
                    "out.jsonl"
                ],
                70,
                "runtime_error"
            ),
            "{\"code\":\"runtime_error\",\"message\":\"infer --out is not implemented yet\"}\n"
        );
    }

    /// 偽ワーカー本体。`launch_script` の中身は使わず、学習リクエストの内容だけで動く。
    pub fn run_fake_worker(request_path: &str) -> ! {
        let bytes = std::fs::read(request_path).expect("read request");
        let request = TrainRequest::from_json_slice(&bytes).expect("valid request");
        let out_dir = format!("{}/{}", request.root(), request.out_dir());
        std::fs::create_dir(&out_dir).expect("create out dir");
        let onnx_src = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/onnx_parity")
            .join(format!("{}.onnx", request.kind()));
        let onnx = std::fs::read(&onnx_src).expect("fixture onnx");
        std::fs::write(format!("{out_dir}/model.onnx"), &onnx).expect("write onnx");
        let sha = Sha256Digest::of_bytes(&onnx).to_hex();
        let labels: Vec<String> = request
            .label_order()
            .as_slice()
            .iter()
            .map(|l| format!("\"{l}\""))
            .collect();
        let labels = labels.join(",");
        let config = if request.kind() == "c1" {
            C1_CONFIG
        } else {
            C3_CONFIG
        };
        let (kind, version, max_bytes) =
            (request.kind(), request.kind_version(), request.max_bytes());
        std::fs::write(
            format!("{out_dir}/artifact.json"),
            format!(
                r#"{{"kind":"{kind}","kind_version":{version},"max_bytes":{max_bytes},"label_order":[{labels}],"onnx_file":"model.onnx","onnx_sha256":"{sha}"}}"#
            ),
        )
        .expect("write artifact.json");
        let predictions: Vec<String> = request
            .validation_inputs()
            .unwrap_or_default()
            .iter()
            .map(|v| {
                let label = if kind == "c3" {
                    v.input().split(' ').next().unwrap_or("alpha")
                } else {
                    "alpha"
                };
                format!(
                    r#"{{"id":"{}","status":"ok","predicted_label":"{label}"}}"#,
                    v.id()
                )
            })
            .collect();
        println!(
            r#"{{"status":"ok","artifact_dir":"{out_dir}","artifact":{{"kind":"{kind}","kind_version":{version},"selector_version":"0.1","config":{config},"label_order":[{labels}],"output_type":"choice","max_bytes":{max_bytes},"onnx_file":"model.onnx","onnx_sha256":"{sha}","created_utc":"2026-09-30T00:00:00Z","candidate_label":"{kind}"}},"validation_predictions":[{}]}}"#,
            predictions.join(",")
        );
        std::process::exit(0);
    }
}

#[cfg(unix)]
fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().collect();
    // `run_train` の固定 argv: [program, "-I", launch_script, "train", "--request", path, ...]
    if args.get(1).map(String::as_str) == Some("-I")
        && args.get(3).map(String::as_str) == Some("train")
        && args.get(4).map(String::as_str) == Some("--request")
    {
        let request = args.get(5).expect("request path");
        suite::run_fake_worker(request);
    }
    let tests: &[(&str, fn())] = &[
        (
            "full_pipeline_completes_without_evaluation_data",
            suite::full_pipeline_completes_without_evaluation_data,
        ),
        (
            "evaluation_data_is_frozen_and_evaluate_is_not_faked",
            suite::evaluation_data_is_frozen_and_evaluate_is_not_faked,
        ),
        (
            "evaluate_stops_on_frozen_hash_mismatch",
            suite::evaluate_stops_on_frozen_hash_mismatch,
        ),
        (
            "inspect_rejects_empty_evaluation_data_without_freeze_record",
            suite::inspect_rejects_empty_evaluation_data_without_freeze_record,
        ),
        (
            "stages_require_their_predecessors",
            suite::stages_require_their_predecessors,
        ),
        (
            "register_rejects_existing_and_escaping_paths",
            suite::register_rejects_existing_and_escaping_paths,
        ),
        (
            "inspect_rejects_invalid_records",
            suite::inspect_rejects_invalid_records,
        ),
        (
            "infer_rejects_tampered_package",
            suite::infer_rejects_tampered_package,
        ),
        (
            "infer_out_option_is_not_faked",
            suite::infer_out_option_is_not_faked,
        ),
    ];
    let mut failed = 0;
    for (name, test) in tests {
        match std::panic::catch_unwind(test) {
            Ok(()) => println!("test {name} ... ok"),
            Err(_) => {
                println!("test {name} ... FAILED");
                failed += 1;
            }
        }
    }
    println!(
        "\ntest result: {} passed; {failed} failed",
        tests.len() - failed
    );
    if failed == 0 {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
}

#[cfg(not(unix))]
fn main() {}
