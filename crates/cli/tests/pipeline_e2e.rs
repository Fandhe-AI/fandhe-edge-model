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

    /// REQ-17: 評価データがあると `register` が凍結記録を作り、評価データを読み取り専用（0400）で配置する。
    pub fn evaluation_data_is_frozen_and_read_only() {
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

    /// 評価データを 1 バイトだけ書き換える（読み取り専用配置を外して差し替える改ざんの模擬。
    /// 実運用では書き込みは拒否される）。長さは変えない。
    fn tamper_evaluation_one_byte(env: &Env) {
        let eval_file = env.project_file("data/evaluation.jsonl");
        let mut perm = std::fs::metadata(&eval_file).expect("meta").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o600);
        std::fs::set_permissions(&eval_file, perm).expect("chmod");
        let mut bytes = std::fs::read(&eval_file).expect("read");
        let at = bytes
            .windows(5)
            .position(|w| w == b"alpha")
            .expect("alpha exists");
        bytes[at + 4] = b'b';
        std::fs::write(&eval_file, bytes).expect("tamper");
    }

    /// 凍結ハッシュ不一致で工程が停止したときの、`evaluate` と同じ終了コードと `code`・固定 message。
    const FROZEN_MISMATCH_STDOUT: &str = "{\"code\":\"invalid_input\",\"message\":\"eval data hash mismatch: frozen record sha256=de442855ffe8258cc3f2d6db32e769393420740014b4ba9f0026cf94066b47ec byte_len=1184, actual sha256=e74473414f0868e0362f32368f9d860dc4dd3cead3b98005d282e859ff5c5d18 byte_len=1184\"}\n";

    fn eval_env_until(case: &str, stages: &[&[&str]]) -> Env {
        let env = registered(case, true);
        env.ok(&["inspect", "--project-dir", "proj"]);
        for args in stages {
            env.ok(args);
        }
        env
    }

    /// REQ-17: `train` は開始時に凍結ハッシュを確認し、不一致なら学習の出力物を作らず停止する。
    pub fn train_stops_on_frozen_hash_mismatch_without_outputs() {
        let env = eval_env_until("trainfrz", &[]);
        tamper_evaluation_one_byte(&env);
        let out = env.fails(
            &["train", "--project-dir", "proj", "--candidate", "0"],
            64,
            "invalid_input",
        );
        assert_eq!(out, FROZEN_MISMATCH_STDOUT);
        assert!(!env.project_file("candidates").exists());
    }

    /// REQ-17: `select` は開始時に凍結ハッシュを確認し、不一致なら選定記録を作らず停止する。
    pub fn select_stops_on_frozen_hash_mismatch_without_outputs() {
        let env = eval_env_until(
            "selfrz",
            &[&["train", "--project-dir", "proj", "--candidate", "0"]],
        );
        tamper_evaluation_one_byte(&env);
        let out = env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input");
        assert_eq!(out, FROZEN_MISMATCH_STDOUT);
        assert!(!env.project_file("selection_record.json").exists());
    }

    /// REQ-17: `package` は開始時に凍結ハッシュを確認し、不一致なら `package/` もステージングも作らず停止する。
    pub fn package_stops_on_frozen_hash_mismatch_without_outputs() {
        let env = eval_env_until(
            "pkgfrz",
            &[
                &["train", "--project-dir", "proj", "--candidate", "0"],
                &["select", "--project-dir", "proj"],
            ],
        );
        tamper_evaluation_one_byte(&env);
        let out = env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input");
        assert_eq!(out, FROZEN_MISMATCH_STDOUT);
        assert!(!env.project_file("package").exists());
        assert!(!env.project_file("package.staging").exists());
    }

    /// REQ-17・REQ-27: 評価データがあるプロジェクトは、評価の完了記録が無い（#314 で追加予定）ため、
    /// register → inspect → train → select の後でも `package` は `invalid_input`（64）で拒否し、
    /// `package/` もステージングも作らない（評価が未完了のまま配布パッケージを公開しない）。
    pub fn package_is_rejected_when_evaluation_data_exists_but_not_completed() {
        let env = eval_env_until(
            "pkgnoeval",
            &[
                &["train", "--project-dir", "proj", "--candidate", "0"],
                &["select", "--project-dir", "proj"],
            ],
        );
        assert_eq!(
            env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"evaluation has not been completed\"}\n"
        );
        assert!(!env.project_file("package").exists());
        assert!(!env.project_file("package.staging").exists());
    }

    /// REQ-27・REQ-33: `train --smoke`（epochs=1 の短縮学習）の候補は `select` できるが、`package` は
    /// 既定で `invalid_input`（64）・固定 message で拒否し `package/` もステージングも作らない。検証専用の
    /// `--allow-smoke` を付けたときだけ成功する。
    pub fn package_rejects_smoke_trained_candidate_unless_allowed() {
        let env = inspected("smokepkg");
        env.ok(&[
            "train",
            "--project-dir",
            "proj",
            "--candidate",
            "0",
            "--smoke",
        ]);
        env.ok(&["select", "--project-dir", "proj"]);
        assert_eq!(
            env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"smoke-trained candidate cannot be packaged\"}\n"
        );
        assert!(!env.project_file("package").exists());
        assert!(!env.project_file("package.staging").exists());
        let out = env.ok(&["package", "--project-dir", "proj", "--allow-smoke"]);
        assert!(out.contains("\"status\":\"ok\""), "{out}");
        assert!(env.project_file("package/artifact.json").is_file());
    }

    /// REQ-27・REQ-39: 保存済みの `request.json` の `device`・時間制限・`train_path`（結果 `result.json` とは
    /// 整合したまま）を書き換えると、`select` も `package` も、期待するリクエストを丸ごと組み立てた比較で
    /// `invalid_input`（64）・固定 message で止まり、選定記録・`package/` を作らない。`epochs` だけが違う
    /// smoke の結果はこの照合を通る（別テストで確認済み）。
    pub fn select_and_package_reject_request_fields_beyond_kind_and_seed() {
        let env = inspected("reqfields");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        let request = env.project_file("candidates/0/request.json");
        let original = std::fs::read_to_string(&request).expect("request.json");
        let tampers = [
            ("device", "\"device\":\"cpu\"", "\"device\":\"gpu\""),
            // 既定の時間制限は JSON に出ないため、非既定の値（1 秒）を先頭に加える。
            ("time limit", "{", "{\"time_limit_seconds\":1,"),
            (
                "train_path",
                "\"train_path\":\"train_input.jsonl\"",
                "\"train_path\":\"other_input.jsonl\"",
            ),
        ];
        let mismatch = "{\"code\":\"invalid_input\",\"message\":\"train request does not match the candidate\"}\n";
        for (label, from, to) in &tampers {
            assert!(original.contains(from), "{label}: {original}");
            std::fs::write(&request, original.replacen(from, to, 1)).expect("tamper");
            assert_eq!(
                env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input"),
                mismatch,
                "select {label}"
            );
            assert!(
                !env.project_file("selection_record.json").exists(),
                "{label}"
            );
        }
        std::fs::write(&request, &original).expect("restore");
        env.ok(&["select", "--project-dir", "proj"]);
        for (label, from, to) in &tampers {
            std::fs::write(&request, original.replacen(from, to, 1)).expect("tamper");
            assert_eq!(
                env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input"),
                mismatch,
                "package {label}"
            );
            assert!(!env.project_file("package").exists(), "{label}");
            assert!(!env.project_file("package.staging").exists(), "{label}");
        }
        std::fs::write(&request, original).expect("restore");
        env.ok(&["package", "--project-dir", "proj"]);
    }

    /// REQ-17: 凍結記録が欠落（評価データだけ残る）していても、後続工程は停止する（fail-closed）。
    pub fn later_stages_stop_when_freeze_record_is_missing() {
        let env = eval_env_until("nofrz", &[]);
        std::fs::remove_file(env.project_file("eval_freeze.json")).expect("remove record");
        let out = env.fails(
            &["train", "--project-dir", "proj", "--candidate", "0"],
            64,
            "invalid_input",
        );
        assert_eq!(
            out,
            "{\"code\":\"invalid_input\",\"message\":\"eval data state is not_provided but actual_bytes is non-empty\"}\n"
        );
        assert!(!env.project_file("candidates").exists());
    }

    /// REQ-17・REQ-39: 評価データの読み取り専用配置は data 層の `place_read_only` を通り、0400・
    /// 書き込み拒否（追記オープンが `PermissionDenied`）・作業用ディレクトリの残骸なし。
    /// root 実行では書き込みを拒否できず配置は fail-closed で失敗する（`runtime_error`・
    /// project は残らない）ため、その場合は失敗側を確認する（本環境の実行者が root でない場合のみ
    /// 成功側が検証される。ACL を実際に作って書き込みを通す模擬は行っていない）。
    pub fn register_places_evaluation_data_via_write_probe() {
        let env = Env::new("probe", true);
        let args = ["register", "--definition", DEF, "--project-dir", "proj"];
        let (code, stdout) = env.run(&args);
        if running_as_root() {
            assert_eq!(code, 70, "{stdout}");
            assert_eq!(
                stdout,
                "{\"code\":\"runtime_error\",\"message\":\"cannot place evaluation data read-only\"}\n"
            );
            assert!(!env.project_file("").exists());
            return;
        }
        assert_eq!(code, 0, "{stdout}");
        let eval_file = env.project_file("data/evaluation.jsonl");
        let mode = std::os::unix::fs::PermissionsExt::mode(
            &std::fs::metadata(&eval_file).expect("meta").permissions(),
        );
        assert_eq!(mode & 0o7777, 0o400);
        let append = std::fs::OpenOptions::new().append(true).open(&eval_file);
        assert_eq!(
            append.expect_err("write must be rejected").kind(),
            std::io::ErrorKind::PermissionDenied
        );
        let mut entries: Vec<String> = std::fs::read_dir(env.project_file("data"))
            .expect("read_dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        entries.sort();
        assert_eq!(entries, ["evaluation.jsonl", "train.jsonl"]);
    }

    fn running_as_root() -> bool {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("Uid:").map(str::to_string))
            })
            .is_some_and(|l| l.split_whitespace().nth(1) == Some("0"))
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

    /// REQ-17: group が 1 件だけで validation 分割が空になるデータは、`inspect` が `invalid_input`（64）で
    /// 拒否し、分割記録（split.json）を保存しない（後続の train が必ず失敗する状態を ok にしない）。
    pub fn inspect_rejects_empty_validation_split_without_split_record() {
        let env = Env::new("emptyval", false);
        let mut data = String::new();
        for i in 0..6 {
            data.push_str(&format!(
                "{{\"id\":\"r{i}\",\"input\":\"alpha sample {i}\",\"output\":{{\"intent\":\"alpha\"}},\"group_id\":\"only\"}}\n"
            ));
        }
        std::fs::write(env.work.join("def").join("train.jsonl"), data).expect("data");
        env.ok(&["register", "--definition", DEF, "--project-dir", "proj"]);
        let out = env.run(&["inspect", "--project-dir", "proj"]);
        assert_eq!(out.0, 64, "{}", out.1);
        assert_eq!(
            out.1,
            "{\"code\":\"invalid_input\",\"message\":\"validation split is empty\"}\n"
        );
        assert!(!env.project_file("split.json").exists());
    }

    /// REQ-34: 学習ワーカーが失敗（異常終了・残骸あり）すると `candidates/<N>/` は片付けられ、
    /// 同じ `--candidate N` を成功するワーカーで再実行すると成功する（再開ではなく新規のやり直し）。
    pub fn train_failure_cleans_candidate_dir_and_allows_retry() {
        let env = inspected("trainretry");
        let marker = env.project_file("fail_worker");
        std::fs::write(&marker, "").expect("marker");
        let (code, stdout) = env.run(&["train", "--project-dir", "proj", "--candidate", "0"]);
        assert_eq!(code, 70, "{stdout}");
        assert_eq!(
            stdout,
            "{\"code\":\"runtime_error\",\"message\":\"worker process exited with unknown exit code 1\"}\n"
        );
        assert!(!env.project_file("candidates/0").exists());
        // 別の候補の領域には触れない（親の `candidates/` は残る）。
        assert!(env.project_file("candidates").is_dir());

        std::fs::remove_file(&marker).expect("remove marker");
        assert_eq!(
            env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]),
            "{\"step\":\"train\",\"status\":\"ok\",\"candidate\":0,\"kind\":\"c1\"}\n"
        );
        assert!(env.project_file("candidates/0/result.json").is_file());
    }

    /// REQ-17: `inspect --seed 7` は分割記録の seed を 7 にし、`train` の学習リクエストの seed も 7 になる
    /// （分割と学習で seed がずれない）。範囲外（u32 超）の `--seed` は引数エラーの `invalid_input`。
    pub fn inspect_seed_is_recorded_and_used_by_train() {
        let env = registered("seed7", false);
        env.fails(
            &["inspect", "--project-dir", "proj", "--seed", "4294967296"],
            64,
            "invalid_input",
        );
        assert!(!env.project_file("split.json").exists());
        env.ok(&["inspect", "--project-dir", "proj", "--seed", "7"]);
        let split = std::fs::read_to_string(env.project_file("split.json")).expect("split.json");
        assert!(split.contains("\"seed\":7"), "{split}");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        let request = std::fs::read_to_string(env.project_file("candidates/0/request.json"))
            .expect("request");
        assert!(request.contains("\"seed\":7"), "{request}");
        // select・package も記録値（7）と照合して通る。
        env.ok(&["select", "--project-dir", "proj"]);
        env.ok(&["package", "--project-dir", "proj"]);
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

    /// REQ-27: `select` は保存済みの学習リクエストの validation が分割記録の validation 全体と
    /// 一致しなければ、候補を選ばず拒否する（記録の改変による部分集合での選定を防ぐ）。
    pub fn select_rejects_request_validation_not_matching_split() {
        let env = inspected("selsplit");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        let request = env.project_file("candidates/0/request.json");
        let original = std::fs::read_to_string(&request).expect("request.json");
        let start = original
            .find("\"validation_inputs\":[")
            .expect("validation_inputs")
            + "\"validation_inputs\":[".len();
        let end = start + original[start..].find("},").expect("first element") + 2;
        let mut tampered = original.clone();
        tampered.replace_range(start..end, "");
        std::fs::write(&request, tampered).expect("tamper request");
        let out = env.run(&["select", "--project-dir", "proj"]);
        assert_eq!(out.0, 64, "{}", out.1);
        assert!(!env.project_file("selection_record.json").exists());
        std::fs::write(&request, original).expect("restore");
        env.ok(&["select", "--project-dir", "proj"]);
    }

    /// REQ-27: `select` は保存済みの学習リクエストが既定候補 N の種類・構成と一致しなければ、
    /// 採点せず `invalid_input` で止める（別の構成の学習結果を候補 N として選ばない）。
    pub fn select_rejects_request_not_matching_default_candidate() {
        let env = inspected("selcand");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        let request = env.project_file("candidates/0/request.json");
        let original = std::fs::read_to_string(&request).expect("request.json");
        assert!(original.contains("\"seed\":42"), "{original}");
        std::fs::write(&request, original.replace("\"seed\":42", "\"seed\":43"))
            .expect("tamper request");
        assert_eq!(
            env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"train request does not match the candidate\"}\n"
        );
        assert!(!env.project_file("selection_record.json").exists());
        std::fs::write(&request, original).expect("restore");
        env.ok(&["select", "--project-dir", "proj"]);
    }

    /// REQ-27・REQ-39: `package` は選定記録を信じず選定をやり直し、記録の候補 ID・添字を別の学習済み
    /// 候補へ書き換えると `invalid_input`（64）で止まり、`package/` もステージングも作らない。
    pub fn package_rejects_selection_record_rewritten_to_other_candidate() {
        let env = inspected("pkgrewrite");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        env.ok(&["select", "--project-dir", "proj"]);
        let record = env.project_file("selection_record.json");
        let c1_record = std::fs::read_to_string(&record).expect("record of c1");
        assert!(c1_record.contains("\"candidate_index\":0"), "{c1_record}");
        std::fs::remove_file(&record).expect("remove record");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "1"]);
        env.ok(&["select", "--project-dir", "proj"]);
        let c3_record = std::fs::read_to_string(&record).expect("record of c3");
        assert!(c3_record.contains("\"candidate_index\":1"), "{c3_record}");
        // 実際の選定は候補 1（c3）。記録を候補 0（c1、学習済み）へ書き換える。
        std::fs::write(&record, &c1_record).expect("rewrite record");
        assert_eq!(
            env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"selection record does not match the candidate\"}\n"
        );
        assert!(!env.project_file("package").exists());
        assert!(!env.project_file("package.staging").exists());
        std::fs::write(&record, &c3_record).expect("restore record");
        env.ok(&["package", "--project-dir", "proj"]);
    }

    /// REQ-39: ハッシュは `artifact.json` と一致するが ONNX ではないファイル（pickle 先頭バイトの偽装・
    /// ONNX 以外の中身）は、`infer` と同じ `invalid_input`（64）で公開前に止まり、`package/` を作らない。
    pub fn package_rejects_non_onnx_model_with_matching_hash() {
        let env = inspected("pkgnononnx");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        env.ok(&["select", "--project-dir", "proj"]);
        let dir = env.project_file("candidates/0/model-c1");
        let onnx = dir.join("model.onnx");
        let meta = dir.join("artifact.json");
        let meta_original = std::fs::read_to_string(&meta).expect("artifact.json");
        let onnx_original = std::fs::read(&onnx).expect("onnx");
        let old_hash = Sha256Digest::of_bytes(&onnx_original).to_hex();
        for (label, bytes, expected) in [
            (
                "not onnx at all",
                b"this is not an onnx model".to_vec(),
                "{\"code\":\"invalid_input\",\"message\":\"format rejected: format_not_allowed\"}\n",
            ),
            (
                "pickle disguise",
                b"\x80\x04\x95 pickle".to_vec(),
                "{\"code\":\"invalid_input\",\"message\":\"format rejected: format_not_allowed\"}\n",
            ),
        ] {
            let new_hash = Sha256Digest::of_bytes(&bytes).to_hex();
            std::fs::write(&onnx, &bytes).expect("replace onnx");
            std::fs::write(&meta, meta_original.replace(&old_hash, &new_hash)).expect("meta");
            assert_eq!(
                env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input"),
                expected,
                "{label}"
            );
            assert!(!env.project_file("package").exists(), "{label}");
            assert!(!env.project_file("package.staging").exists(), "{label}");
        }
        std::fs::write(&onnx, onnx_original).expect("restore onnx");
        std::fs::write(&meta, meta_original).expect("restore meta");
        env.ok(&["package", "--project-dir", "proj"]);
    }

    /// 候補 0 の成果物をプロジェクト内の別の場所（`other/`）へ移した状態を作る。
    /// `request.json` の `root` と `result.json` の `artifact_dir` を、移動先に合わせて書き換える。
    fn relocate_candidate_zero_artifacts(env: &Env) {
        std::fs::create_dir_all(env.project_file("other")).expect("other");
        std::fs::rename(
            env.project_file("candidates/0/model-c1"),
            env.project_file("other/model-c1"),
        )
        .expect("move artifacts");
        let project = env.project_file("");
        let project = project.to_str().expect("utf8").trim_end_matches('/');
        let old_root = format!("{project}/candidates/0");
        let new_root = format!("{project}/other");
        for file in ["request.json", "result.json"] {
            let path = env.project_file(&format!("candidates/0/{file}"));
            let text = std::fs::read_to_string(&path).expect(file);
            assert!(text.contains(&old_root), "{file}: {text}");
            std::fs::write(&path, text.replace(&old_root, &new_root)).expect("rewrite");
        }
    }

    /// REQ-39: 候補 0 の `request.json` の `root` と `result.json` の `artifact_dir` を、プロジェクト内の
    /// 別の場所へ揃えて書き換えても（リクエストと結果の整合は保たれる）、`select` は候補 0 の期待する
    /// `root` と一致しないため `invalid_input`（64）で止まり、選定記録を作らない。
    pub fn select_rejects_request_root_outside_candidate_dir() {
        let env = inspected("selroot");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        relocate_candidate_zero_artifacts(&env);
        assert_eq!(
            env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"train request does not match the candidate\"}\n"
        );
        assert!(!env.project_file("selection_record.json").exists());
    }

    /// REQ-39: 同じ差し替えを選定の後に行うと、`package` も `invalid_input`（64）で止まり、
    /// `package/` もステージングも作らない。
    pub fn package_rejects_request_root_outside_candidate_dir() {
        let env = inspected("pkgroot");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        env.ok(&["select", "--project-dir", "proj"]);
        relocate_candidate_zero_artifacts(&env);
        assert_eq!(
            env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"train request does not match the candidate\"}\n"
        );
        assert!(!env.project_file("package").exists());
        assert!(!env.project_file("package.staging").exists());
    }

    /// REQ-39: `request.json` はそのままで、成果物のディレクトリ `candidates/0/model-c1` を
    /// プロジェクト内の別の場所（`other/`）への symlink にして `artifact_dir` が候補の外を指す場合、
    /// 保存済みの結果の再検証（期待する `artifact_dir` と不一致）で `invalid_input`（64）として止まり、
    /// `package/` を作らない。
    pub fn package_rejects_artifact_dir_outside_candidate_dir() {
        let env = inspected("pkgoutside");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        env.ok(&["select", "--project-dir", "proj"]);
        std::fs::create_dir_all(env.project_file("other")).expect("other");
        let link = env.project_file("candidates/0/model-c1");
        std::fs::rename(&link, env.project_file("other/model-c1")).expect("move");
        std::os::unix::fs::symlink("../../other/model-c1", &link).expect("symlink");
        let result = env.project_file("candidates/0/result.json");
        let text = std::fs::read_to_string(&result).expect("result.json");
        std::fs::write(
            &result,
            text.replace("/candidates/0/model-c1", "/other/model-c1"),
        )
        .expect("rewrite result");
        assert_eq!(
            env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"stored train result is invalid\"}\n"
        );
        assert!(!env.project_file("package").exists());
        assert!(!env.project_file("package.staging").exists());
    }

    /// REQ-39: `artifact.json` の `kind_version` が選定候補の学習リクエストと食い違うと、公開前に
    /// `invalid_input`（64）で止まり `package/` を作らない。
    pub fn package_rejects_artifact_kind_version_mismatch() {
        let env = inspected("pkgkv");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        env.ok(&["select", "--project-dir", "proj"]);
        let meta = env.project_file("candidates/0/model-c1/artifact.json");
        let original = std::fs::read_to_string(&meta).expect("artifact.json");
        assert!(original.contains("\"kind_version\":1"), "{original}");
        std::fs::write(
            &meta,
            original.replace("\"kind_version\":1", "\"kind_version\":2"),
        )
        .expect("tamper");
        assert_eq!(
            env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"artifact kind_version does not match the selected candidate\"}\n"
        );
        assert!(!env.project_file("package").exists());
        assert!(!env.project_file("package.staging").exists());
        std::fs::write(&meta, original).expect("restore");
        env.ok(&["package", "--project-dir", "proj"]);
    }

    /// REQ-27・REQ-32・REQ-39: `package` は選定記録の候補 ID、`artifact.json` の `label_order`・`kind`
    /// が選定候補・定義と食い違うと、`package/` を残さず拒否する。
    pub fn package_rejects_mismatched_selection_and_metadata() {
        let env = inspected("pkgmismatch");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        env.ok(&["select", "--project-dir", "proj"]);
        let record = env.project_file("selection_record.json");
        let original = std::fs::read_to_string(&record).expect("selection record");
        std::fs::write(
            &record,
            original.replace("\"candidate_id\":\"", "\"candidate_id\":\"other-"),
        )
        .expect("tamper record");
        assert_eq!(
            env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"selection record does not match the candidate\"}\n"
        );
        std::fs::write(&record, &original).expect("restore record");

        let meta = env.project_file("candidates/0/model-c1/artifact.json");
        let meta_original = std::fs::read_to_string(&meta).expect("artifact.json");
        std::fs::write(
            &meta,
            meta_original.replace(
                "\"label_order\":[\"alpha\",\"beta\",\"gamma\"]",
                "\"label_order\":[\"beta\",\"alpha\",\"gamma\"]",
            ),
        )
        .expect("tamper labels");
        assert_eq!(
            env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"package label order does not match definition\"}\n"
        );
        std::fs::write(
            &meta,
            meta_original.replace("\"kind\":\"c1\"", "\"kind\":\"c3\""),
        )
        .expect("tamper kind");
        env.fails(&["package", "--project-dir", "proj"], 64, "invalid_input");
        assert!(!env.project_file("package").exists());
        std::fs::write(&meta, &meta_original).expect("restore meta");
        env.ok(&["package", "--project-dir", "proj"]);
    }

    /// 評価データありで `register → inspect → train 0 → train 1` まで進める（評価はまだ）。
    fn eval_trained(case: &str) -> Env {
        eval_env_until(
            case,
            &[
                &["train", "--project-dir", "proj", "--candidate", "0"],
                &["train", "--project-dir", "proj", "--candidate", "1"],
            ],
        )
    }

    const EVALUATE_0: [&str; 5] = ["evaluate", "--project-dir", "proj", "--candidate", "0"];
    const EVALUATE_1: [&str; 5] = ["evaluate", "--project-dir", "proj", "--candidate", "1"];
    const PACKAGE: [&str; 3] = ["package", "--project-dir", "proj"];
    const SELECT: [&str; 3] = ["select", "--project-dir", "proj"];

    /// `"key":<number>` の数値（次の `,` または `}` まで）を取り出す。
    fn number_field(json: &str, key: &str) -> f64 {
        let marker = format!("\"{key}\":");
        let rest = json.split(&marker).nth(1).expect("key exists");
        let end = rest.find([',', '}']).expect("value end");
        rest[..end].parse().expect("number")
    }

    fn file_sha256(env: &Env, rel: &str) -> String {
        Sha256Digest::of_bytes(&std::fs::read(env.project_file(rel)).expect("read")).to_hex()
    }

    /// REQ-33・REQ-24・REQ-27: 評価データありで 7 工程がすべて非対話で exit 0 まで完走し、
    /// `evaluate` は JSON 1 つ（正解率・Macro-F1）と評価完了の記録を残す。選定（`select`）が先で、
    /// 選定前・選定候補以外の `evaluate` は拒否される（REQ-27）。
    pub fn full_pipeline_completes_with_evaluation_data() {
        let env = eval_trained("fullevl");
        // 選定前は、どの候補も最終 test に適用できない。
        assert_eq!(
            env.fails(&EVALUATE_1, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"candidate selection has not been recorded\"}\n"
        );
        assert!(!env.project_file("final_test_ledger").exists());
        assert_eq!(
            env.ok(&["select", "--project-dir", "proj"]),
            "{\"step\":\"select\",\"status\":\"ok\",\"candidate\":1,\"kind\":\"c3\"}\n"
        );
        // 選定されていない候補は評価できない（台帳にも触れない）。
        assert_eq!(
            env.fails(&EVALUATE_0, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"candidate is not the selected candidate\"}\n"
        );
        assert!(!env.project_file("final_test_ledger").exists());
        let out = env.ok(&EVALUATE_1);
        let prefix = "{\"step\":\"evaluate\",\"status\":\"ok\",\"candidate\":1,\"kind\":\"c3\",\"n_total\":12,\"correct\":";
        assert!(out.starts_with(prefix), "{out}");
        let correct = number_field(&out, "correct");
        assert!((0.0..=12.0).contains(&correct), "{out}");
        assert!(
            (number_field(&out, "accuracy") - correct / 12.0).abs() < 1e-9,
            "{out}"
        );
        assert!(out.contains("\"macro_f1\":"), "{out}");
        assert!(
            env.project_file("candidates/1/evaluation_record.json")
                .is_file()
        );
        assert_eq!(
            env.ok(&PACKAGE),
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
            text.starts_with("{\"id\":\"input\",\"status\":\"ok\""),
            "{text}"
        );
        std::fs::write(
            env.work.join("batch.jsonl"),
            "{\"id\":\"a\",\"input\":\"alpha sample 1\"}\n",
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
    }

    /// REQ-27: `evaluate` の前後で、モデル（ONNX・`artifact.json`）と評価データの sha256 が一致し、
    /// 評価完了記録の値が実ファイルのハッシュと一致する。
    pub fn evaluate_keeps_model_and_evaluation_hashes() {
        let env = eval_trained("evalhashes");
        env.ok(&SELECT);
        let files = [
            "candidates/1/model-c3/model.onnx",
            "candidates/1/model-c3/artifact.json",
            "data/evaluation.jsonl",
        ];
        let before: Vec<String> = files.iter().map(|f| file_sha256(&env, f)).collect();
        env.ok(&EVALUATE_1);
        let after: Vec<String> = files.iter().map(|f| file_sha256(&env, f)).collect();
        assert_eq!(before, after);
        let record =
            std::fs::read_to_string(env.project_file("candidates/1/evaluation_record.json"))
                .expect("record");
        for (key, hash) in [
            ("onnx_sha256", &before[0]),
            ("artifact_meta_sha256", &before[1]),
            ("evaluation_sha256", &before[2]),
        ] {
            assert!(
                record.contains(&format!("\"{key}\":\"{hash}\"")),
                "{key}: {record}"
            );
        }
        assert!(record.contains("\"config_id\":\"c3:seed42\""), "{record}");
    }

    /// REQ-28・REQ-27: 評価の `correct` は、同じ入力を配布パッケージへ `infer --input-file` した
    /// 予測と正解の一致件数に等しい（評価経路と推論経路の全件一致）。
    pub fn evaluate_correct_matches_infer_on_package() {
        let env = eval_trained("evalinfer");
        env.ok(&SELECT);
        let out = env.ok(&EVALUATE_1);
        env.ok(&PACKAGE);
        let mut batch = String::new();
        let mut golds = Vec::new();
        for i in 0..4 {
            for l in LABELS {
                batch.push_str(&format!(
                    "{{\"id\":\"e-{l}-{i}\",\"input\":\"{l} evaluation {i}\"}}\n"
                ));
                golds.push(l);
            }
        }
        std::fs::write(env.work.join("batch.jsonl"), batch).expect("batch");
        let (code, lines) = env.run(&[
            "infer",
            "--package",
            "proj/package",
            "--input-file",
            "batch.jsonl",
        ]);
        assert_eq!(code, 0, "{lines}");
        assert_eq!(lines.lines().count(), 12, "{lines}");
        let matched = lines
            .lines()
            .zip(&golds)
            .filter(|(line, gold)| line.contains(&format!("\"predicted_label\":\"{gold}\"")))
            .count();
        assert_eq!(
            number_field(&out, "correct").to_bits(),
            (matched as f64).to_bits(),
            "{out} matched={matched}"
        );
    }

    /// REQ-27: 同じ候補への 2 回目の `evaluate` は固定 message の `invalid_input` で、記録は変わらない。
    /// 記録を消しても、台帳の適用ロックが再適用を拒否する。
    pub fn evaluate_twice_is_rejected() {
        let env = eval_trained("evaltwice");
        env.ok(&SELECT);
        env.ok(&EVALUATE_1);
        let record_path = env.project_file("candidates/1/evaluation_record.json");
        let original = std::fs::read(&record_path).expect("record");
        let expected = "{\"code\":\"invalid_input\",\"message\":\"candidate has already been evaluated on the frozen data\"}\n";
        assert_eq!(env.fails(&EVALUATE_1, 64, "invalid_input"), expected);
        assert_eq!(std::fs::read(&record_path).expect("record"), original);
        // 記録ファイルを消しても、台帳のロックが 2 回目の適用を拒否する（記録の削除で回避できない）。
        std::fs::remove_file(&record_path).expect("remove record");
        assert_eq!(env.fails(&EVALUATE_1, 64, "invalid_input"), expected);
        assert!(!record_path.exists());
    }

    /// REQ-27: 選定後に学習した候補が現れて選定が変わる場合、選定記録と再計算が食い違うため、
    /// `evaluate` は拒否する（最終 test の適用前に選定を確定させる）。
    pub fn evaluate_rejects_selection_record_out_of_date() {
        let env = eval_env_until(
            "evalafter",
            &[&["train", "--project-dir", "proj", "--candidate", "0"]],
        );
        env.ok(&SELECT);
        env.ok(&["train", "--project-dir", "proj", "--candidate", "1"]);
        assert_eq!(
            env.fails(&EVALUATE_0, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"selection record does not match the candidate\"}\n"
        );
        assert!(!env.project_file("final_test_ledger").exists());
        assert!(
            !env.project_file("candidates/0/evaluation_record.json")
                .exists()
        );
    }

    /// REQ-27: 選定記録が別の候補へ書き換えられても、`evaluate` は再計算と不一致で拒否する。
    pub fn evaluate_rejects_rewritten_selection_record() {
        let env = eval_trained("evalrewrite");
        env.ok(&SELECT);
        let record = env.project_file("selection_record.json");
        let original = std::fs::read_to_string(&record).expect("selection record");
        std::fs::write(
            &record,
            original
                .replacen("\"candidate_index\":1", "\"candidate_index\":0", 1)
                .replacen("\"candidate_id\":\"c3\"", "\"candidate_id\":\"c1\"", 1),
        )
        .expect("tamper");
        assert_eq!(
            env.fails(&EVALUATE_0, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"selection record does not match the candidate\"}\n"
        );
        assert!(!env.project_file("final_test_ledger").exists());
    }

    /// REQ-27: 未学習の候補は選定できず（`pending`）評価もできない。`train --smoke` の候補は選定できても
    /// 評価できず、台帳にロックを作らない。
    pub fn evaluate_rejects_untrained_and_smoke_trained_candidates() {
        let env = eval_env_until("evalsmoke", &[]);
        assert_eq!(
            env.fails(&EVALUATE_0, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"candidate selection has not been recorded\"}\n"
        );
        env.ok(&[
            "train",
            "--project-dir",
            "proj",
            "--candidate",
            "0",
            "--smoke",
        ]);
        env.ok(&SELECT);
        assert_eq!(
            env.fails(&EVALUATE_0, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"smoke-trained candidate cannot be evaluated\"}\n"
        );
        assert!(!env.project_file("final_test_ledger").exists());
        assert!(
            !env.project_file("candidates/0/evaluation_record.json")
                .exists()
        );
    }

    /// REQ-27: 選定候補が評価済みでなければ `package` は拒否し、`package/`・ステージングを作らない
    /// （選定されていない候補は評価できないため、評価済みの別候補で代替できない）。
    pub fn package_rejects_when_selected_candidate_not_evaluated() {
        let env = eval_trained("pkgselnoeval");
        // select は validation だけで選ぶため candidate 1（c3）を選ぶ。
        assert_eq!(
            env.ok(&SELECT),
            "{\"step\":\"select\",\"status\":\"ok\",\"candidate\":1,\"kind\":\"c3\"}\n"
        );
        assert_eq!(
            env.fails(&PACKAGE, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"evaluation has not been completed\"}\n"
        );
        assert!(!env.project_file("package").exists());
        assert!(!env.project_file("package.staging").exists());
    }

    /// REQ-27: 評価完了記録が実体と一致していても、最終 test の台帳に適用完了が無ければ（台帳を消した・
    /// 記録だけを用意した）`package` は拒否し、`package/` を作らない（記録の偽造では公開できない）。
    pub fn package_rejects_record_without_ledger_completion() {
        let env = eval_trained("pkgnoledger");
        env.ok(&SELECT);
        env.ok(&EVALUATE_1);
        std::fs::remove_dir_all(env.project_file("final_test_ledger")).expect("remove ledger");
        assert_eq!(
            env.fails(&PACKAGE, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"evaluation has not been completed\"}\n"
        );
        assert!(!env.project_file("package").exists());
        assert!(!env.project_file("package.staging").exists());
    }

    /// REQ-27: 評価完了記録のハッシュ・候補の値が実体と食い違う、または未知キーを含む場合、
    /// `package` は拒否し `package/` を作らない。元に戻せば成功する。
    pub fn package_rejects_tampered_evaluation_record() {
        let env = eval_trained("pkgtamper");
        env.ok(&SELECT);
        env.ok(&EVALUATE_1);
        let record_path = env.project_file("candidates/1/evaluation_record.json");
        let original = std::fs::read_to_string(&record_path).expect("record");
        let mismatch = "{\"code\":\"invalid_input\",\"message\":\"evaluation record does not match the package\"}\n";
        for key in [
            "onnx_sha256",
            "evaluation_sha256",
            "definition_sha256",
            "artifact_meta_sha256",
        ] {
            let marker = format!("\"{key}\":\"");
            let at = original.find(&marker).expect("key") + marker.len();
            let mut tampered = original.clone();
            let flipped = if original.as_bytes().get(at) == Some(&b'0') {
                "1"
            } else {
                "0"
            };
            tampered.replace_range(at..=at, flipped);
            std::fs::write(&record_path, tampered).expect("tamper");
            assert_eq!(env.fails(&PACKAGE, 64, "invalid_input"), mismatch, "{key}");
            assert!(!env.project_file("package").exists(), "{key}");
            assert!(!env.project_file("package.staging").exists(), "{key}");
        }
        let tampered = original.replacen("\"candidate_id\":\"c3\"", "\"candidate_id\":\"c1\"", 1);
        std::fs::write(&record_path, tampered).expect("tamper");
        assert_eq!(env.fails(&PACKAGE, 64, "invalid_input"), mismatch);
        // 構成 ID・評価件数・正解数の改変（他のハッシュが合っていても公開できない。REQ-27）。
        let config_marker = "\"config_id\":\"";
        let at = original.find(config_marker).expect("config_id") + config_marker.len();
        let mut other_config = original.clone();
        other_config.insert(at, 'x');
        std::fs::write(&record_path, other_config).expect("tamper");
        assert_eq!(
            env.fails(&PACKAGE, 64, "invalid_input"),
            mismatch,
            "config_id"
        );
        for key in ["total", "correct"] {
            let marker = format!("\"{key}\":");
            let at = original.find(&marker).expect("key") + marker.len();
            let end = at
                + original[at..]
                    .find(|c: char| !c.is_ascii_digit())
                    .expect("number end");
            let mut bumped = original.clone();
            // 評価件数・正解数を桁違いの値へ（`correct` は `total` を超え、`total` は評価件数と食い違う）。
            bumped.replace_range(at..end, "99999");
            std::fs::write(&record_path, bumped).expect("tamper");
            assert_eq!(env.fails(&PACKAGE, 64, "invalid_input"), mismatch, "{key}");
        }
        assert!(!env.project_file("package").exists());
        let unknown = original.trim_end().trim_end_matches('}').to_string() + ",\"extra\":1}\n";
        std::fs::write(&record_path, unknown).expect("tamper");
        assert_eq!(
            env.fails(&PACKAGE, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"evaluation record is invalid\"}\n"
        );
        assert!(!env.project_file("package").exists());
        std::fs::write(&record_path, original).expect("restore");
        env.ok(&PACKAGE);
    }

    /// REQ-27: 評価データがあるプロジェクトでは、smoke 学習の候補は `--allow-smoke` を付けても
    /// `package` できない（`--allow-smoke` が緩めるのは smoke 候補の拒否だけ。評価完了の確認は緩めない）。
    pub fn package_rejects_smoke_candidate_with_allow_smoke_when_evaluation_data_exists() {
        let env = eval_env_until("smokeevalpkg", &[]);
        env.ok(&[
            "train",
            "--project-dir",
            "proj",
            "--candidate",
            "0",
            "--smoke",
        ]);
        env.ok(&SELECT);
        assert_eq!(
            env.fails(&PACKAGE, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"smoke-trained candidate cannot be packaged\"}\n"
        );
        assert_eq!(
            env.fails(
                &["package", "--project-dir", "proj", "--allow-smoke"],
                64,
                "invalid_input"
            ),
            "{\"code\":\"invalid_input\",\"message\":\"evaluation has not been completed\"}\n"
        );
        assert!(!env.project_file("package").exists());
        assert!(!env.project_file("package.staging").exists());
    }

    /// 候補 `index` の `result.json` の validation 予測を、すべて正解ラベル（id の先頭語）へ書き換える。
    /// `validation_predictions` の各要素は `{"id":"<label>-<n>","status":"ok","predicted_label":"<x>"}`。
    fn set_validation_predictions(env: &Env, index: usize, correct: bool) {
        let path = env.project_file(&format!("candidates/{index}/result.json"));
        let text = std::fs::read_to_string(&path).expect("result.json");
        let mut out = String::new();
        let mut rest = text.as_str();
        let key = "\"predicted_label\":\"";
        let mut rewritten = 0;
        while let Some(at) = rest.find(key) {
            let (head, tail) = rest.split_at(at + key.len());
            let id_start = head.rfind("\"id\":\"").expect("id before label") + 6;
            let id = &head[id_start..];
            let gold = id.split('-').next().expect("label prefix").to_string();
            let end = tail.find('"').expect("label end");
            let label = if correct { gold } else { "zzz".to_string() };
            out.push_str(head);
            out.push_str(&label);
            rest = &tail[end..];
            rewritten += 1;
        }
        out.push_str(rest);
        assert!(rewritten > 0, "no validation predictions rewritten");
        std::fs::write(&path, out).expect("write result.json");
    }

    /// REQ-27: 最初の `evaluate` で固定した選定と異なる選定（A を評価した後に validation 結果と選定記録を
    /// 書き換えて B を選ぶ）では、B の `evaluate` は台帳が拒否し最終 test へ適用させない。`package` も拒否する。
    pub fn evaluate_and_package_reject_selection_switched_after_first_evaluation() {
        let env = eval_trained("selswitch");
        // 既定の偽ワーカーでは candidate 1（c3）が選ばれる。A = candidate 1 を評価する。
        env.ok(&SELECT);
        env.ok(&EVALUATE_1);
        // validation 結果を書き換えて candidate 0（c1）が選ばれるようにし、選定記録も作り直す。
        set_validation_predictions(&env, 0, true);
        set_validation_predictions(&env, 1, false);
        std::fs::remove_file(env.project_file("selection_record.json")).expect("remove selection");
        assert_eq!(
            env.ok(&SELECT),
            "{\"step\":\"select\",\"status\":\"ok\",\"candidate\":0,\"kind\":\"c1\"}\n"
        );
        assert_eq!(
            env.fails(&EVALUATE_0, 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"selection differs from the one fixed at the first evaluation\"}\n"
        );
        assert!(
            !env.project_file("candidates/0/evaluation_record.json")
                .exists()
        );
        assert_eq!(
            env.fails(&PACKAGE, 64, "invalid_input")
                .matches('\n')
                .count(),
            1
        );
        assert!(!env.project_file("package").exists());
    }

    /// REQ-27: 選定記録の内容（ダイジェスト）が最初の `evaluate` の時点から変わると（同じ候補でも）、
    /// 再評価は台帳が選定の不一致として拒否し、`package` も固定した選定と一致しないため拒否する。
    /// 元に戻せば `package` は成功する。
    pub fn evaluate_and_package_reject_selection_digest_changed_after_first_evaluation() {
        let env = eval_trained("seldigest");
        env.ok(&SELECT);
        env.ok(&EVALUATE_1);
        let selection_path = env.project_file("selection_record.json");
        let original = std::fs::read(&selection_path).expect("selection record");
        // 意味は同じ（同じ候補・同じ値）だが、バイト列（ダイジェスト）が異なる記録。
        let mut altered = original.clone();
        altered.insert(0, b' ');
        std::fs::write(&selection_path, &altered).expect("alter selection");
        let differs = "{\"code\":\"invalid_input\",\"message\":\"selection differs from the one fixed at the first evaluation\"}\n";
        assert_eq!(env.fails(&PACKAGE, 64, "invalid_input"), differs);
        assert!(!env.project_file("package").exists());
        // 記録を消しても、台帳が固定した選定との不一致で拒否する（適用ロックより前の判定）。
        std::fs::remove_file(env.project_file("candidates/1/evaluation_record.json"))
            .expect("remove record");
        assert_eq!(env.fails(&EVALUATE_1, 64, "invalid_input"), differs);
        std::fs::write(&selection_path, &original).expect("restore selection");
    }

    /// 候補 0 の成果物へ `vocab.json` を置き、`recorded` が `Some` なら `artifact.json` にその sha256 を記録する。
    fn place_vocab(env: &Env, bytes: &[u8], recorded: Option<&[u8]>) {
        std::fs::write(env.project_file("candidates/0/model-c1/vocab.json"), bytes).expect("vocab");
        if let Some(hashed) = recorded {
            let meta = env.project_file("candidates/0/model-c1/artifact.json");
            let text = std::fs::read_to_string(&meta).expect("artifact.json");
            let hex = Sha256Digest::of_bytes(hashed).to_hex();
            let patched = text.replacen('{', &format!("{{\"vocab_sha256\":\"{hex}\","), 1);
            std::fs::write(&meta, patched).expect("patch artifact.json");
        }
    }

    /// 配布容量の目安（40MB）を超える、許可形式の語彙ファイル（約 45MB）。
    fn oversized_vocab() -> Vec<u8> {
        let mut out = String::from("{");
        for i in 0..3_000_000u32 {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("\"t{i}\":{i}"));
        }
        out.push('}');
        out.into_bytes()
    }

    /// REQ-39・REQ-30: 語彙ファイルがあるのに記録ハッシュが無い・記録と不一致の候補は、選定時にも
    /// `invalid_input`（64）で止まり、選定記録を作らない（容量判定より前に完全性を確認する）。
    pub fn select_rejects_vocab_file_without_matching_recorded_hash() {
        let env = inspected("selvocabhash");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        place_vocab(&env, br#"{"a":0}"#, None);
        env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input");
        assert!(!env.project_file("selection_record.json").exists());
        // 記録ハッシュが別内容のもの。
        place_vocab(&env, br#"{"a":0}"#, Some(br#"{"b":1}"#));
        env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input");
        assert!(!env.project_file("selection_record.json").exists());
    }

    /// REQ-30・REQ-39: 容量超過で除外される候補にも整合性の確認が先に適用される。改ざんのない
    /// 超過候補は除外（全件除外で `pending`）になるが、validation 予測を改ざんした超過候補は
    /// 「除外」として扱わず `runtime_error`（70）で止まる。
    pub fn select_validates_candidate_before_capacity_exclusion() {
        let env = inspected("selexclorder");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        let vocab = oversized_vocab();
        place_vocab(&env, &vocab, Some(&vocab));
        let out = env.fails(&["select", "--project-dir", "proj"], 12, "pending");
        assert!(out.contains("excluded"), "{out}");

        let result = env.project_file("candidates/0/result.json");
        let text = std::fs::read_to_string(&result).expect("result.json");
        let marker = "\"validation_predictions\":[{\"id\":\"";
        assert!(text.contains(marker), "{text}");
        std::fs::write(
            &result,
            text.replacen(marker, &format!("{marker}tampered-"), 1),
        )
        .expect("tamper");
        let (code, stdout) = env.run(&["select", "--project-dir", "proj"]);
        assert_eq!(code, 70, "{stdout}");
        assert!(stdout.contains("cannot score"), "{stdout}");
        assert!(!env.project_file("selection_record.json").exists());
    }

    /// REQ-30・REQ-39: 容量の目安を超える語彙も同じストリーミング検証を通す。ハッシュだけ合わせた
    /// 45MB の非 JSON は除外にせず `invalid_input`（64）で止まる。形式が正しい超過は除外（`pending`）、
    /// ハッシュ不一致・末尾に余分なデータがある場合も 64。
    pub fn select_oversized_vocab_is_validated_not_just_excluded() {
        let env = inspected("selbigvocab");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        let garbage = vec![b'x'; 45_000_000];
        place_vocab(&env, &garbage, Some(&garbage));
        env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input");
        assert!(!env.project_file("selection_record.json").exists());

        let meta = env.project_file("candidates/0/model-c1/artifact.json");
        let vocab_path = env.project_file("candidates/0/model-c1/vocab.json");
        let replace_hash = |old: &[u8], new: &[u8]| {
            let text = std::fs::read_to_string(&meta).expect("artifact.json");
            let (old, new) = (
                Sha256Digest::of_bytes(old).to_hex(),
                Sha256Digest::of_bytes(new).to_hex(),
            );
            assert!(text.contains(&old), "{text}");
            std::fs::write(&meta, text.replace(&old, &new)).expect("patch");
        };
        // 正しい語彙（45MB 超）は除外される。
        let valid = oversized_vocab();
        std::fs::write(&vocab_path, &valid).expect("vocab");
        replace_hash(&garbage, &valid);
        let out = env.fails(&["select", "--project-dir", "proj"], 12, "pending");
        assert!(out.contains("vocab_package_over_guideline"), "{out}");
        // 末尾に余分なデータ。
        let mut trailing = valid.clone();
        trailing.extend_from_slice(b" x");
        std::fs::write(&vocab_path, &trailing).expect("vocab");
        replace_hash(&valid, &trailing);
        env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input");
        // ハッシュ不一致。
        replace_hash(&trailing, b"other");
        env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input");
        assert!(!env.project_file("selection_record.json").exists());
    }

    /// REQ-30・REQ-39: `model-c1/` 配下に置いた実配置の小さな語彙は、select・package・infer の
    /// 3 経路が同じ保持 fd 検証で通り、パッケージへ複写される。
    pub fn vocab_in_artifact_dir_passes_select_package_and_infer() {
        let env = inspected("vocabpipe");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        let vocab = br#"{"a":0,"b":1}"#;
        place_vocab(&env, vocab, Some(vocab));
        env.ok(&["select", "--project-dir", "proj"]);
        env.ok(&["package", "--project-dir", "proj"]);
        assert_eq!(
            std::fs::read(env.project_file("package/vocab.json")).expect("packaged vocab"),
            vocab
        );
        env.ok(&[
            "infer",
            "--package",
            "proj/package",
            "--text",
            "alpha sample",
        ]);
    }

    /// REQ-30・REQ-39: 容量超過の候補にも ONNX の完全性確認が先に適用される。ONNX を改ざんした
    /// 超過候補（記録 sha256 と不一致・非 ONNX で記録ハッシュを合わせた場合）は「除外」にならず
    /// `invalid_input`（64）で止まる。
    pub fn select_verifies_onnx_before_capacity_exclusion() {
        let env = inspected("selonnxexcl");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        let vocab = oversized_vocab();
        place_vocab(&env, &vocab, Some(&vocab));
        let onnx = env.project_file("candidates/0/model-c1/model.onnx");
        let original = std::fs::read(&onnx).expect("onnx");
        // 1 バイト追記（記録 sha256 と不一致）。
        let mut tampered = original.clone();
        tampered.push(0);
        std::fs::write(&onnx, &tampered).expect("tamper");
        env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input");
        assert!(!env.project_file("selection_record.json").exists());
        // 非 ONNX の内容へ差し替え、artifact.json の記録ハッシュも合わせる（形式の許可リストで拒否）。
        let junk = b"not an onnx model at all".to_vec();
        std::fs::write(&onnx, &junk).expect("junk");
        let meta = env.project_file("candidates/0/model-c1/artifact.json");
        let text = std::fs::read_to_string(&meta).expect("artifact.json");
        let old = Sha256Digest::of_bytes(&original).to_hex();
        let new = Sha256Digest::of_bytes(&junk).to_hex();
        assert!(text.contains(&old), "{text}");
        std::fs::write(&meta, text.replace(&old, &new)).expect("patch");
        env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input");
        assert!(!env.project_file("selection_record.json").exists());
    }

    /// REQ-30・REQ-39: 容量超過の候補にも `package` と同じ `load_backend` の確認が先に適用される。
    /// 形式は ONNX でハッシュも一致するが読み込めないモデルと、出力クラス数が定義の選択肢数と
    /// 食い違うモデル（2 択の定義に 3 クラスの fixture）は、除外にならず `invalid_input`（64）で止まる。
    pub fn select_loads_onnx_before_capacity_exclusion() {
        let env = inspected("selloadexcl");
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        let vocab = oversized_vocab();
        place_vocab(&env, &vocab, Some(&vocab));
        let onnx = env.project_file("candidates/0/model-c1/model.onnx");
        let original = std::fs::read(&onnx).expect("onnx");
        // ONNX の最小の形（形式判定は通る）だが、モデルとしては読み込めない。
        let broken = vec![0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78];
        std::fs::write(&onnx, &broken).expect("broken");
        let meta = env.project_file("candidates/0/model-c1/artifact.json");
        let text = std::fs::read_to_string(&meta).expect("artifact.json");
        let old = Sha256Digest::of_bytes(&original).to_hex();
        let new = Sha256Digest::of_bytes(&broken).to_hex();
        std::fs::write(&meta, text.replace(&old, &new)).expect("patch");
        assert_eq!(
            env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"model file cannot be loaded\"}\n"
        );
        assert!(!env.project_file("selection_record.json").exists());

        // 2 択の定義（fixture は 3 クラス）。
        let env = Env::new("selclasses", false);
        let two = definition_text().replace(
            r#",{"id":"gamma","display_name":"gamma","description":"dummy"}"#,
            "",
        );
        std::fs::write(env.work.join("def").join("definition.json"), two).expect("definition");
        let data: String = train_jsonl()
            .lines()
            .filter(|l| !l.contains("gamma"))
            .map(|l| format!("{l}\n"))
            .collect();
        std::fs::write(env.work.join("def").join("train.jsonl"), data).expect("data");
        env.ok(&["register", "--definition", DEF, "--project-dir", "proj"]);
        env.ok(&["inspect", "--project-dir", "proj"]);
        env.ok(&["train", "--project-dir", "proj", "--candidate", "0"]);
        let vocab = oversized_vocab();
        place_vocab(&env, &vocab, Some(&vocab));
        assert_eq!(
            env.fails(&["select", "--project-dir", "proj"], 64, "invalid_input"),
            "{\"code\":\"invalid_input\",\"message\":\"model output size does not match definition\"}\n"
        );
        assert!(!env.project_file("selection_record.json").exists());
    }

    /// 偽ワーカー本体。`launch_script` の中身は使わず、学習リクエストの内容だけで動く。
    pub fn run_fake_worker(request_path: &str) -> ! {
        let bytes = std::fs::read(request_path).expect("read request");
        let request = TrainRequest::from_json_slice(&bytes).expect("valid request");
        let out_dir = format!("{}/{}", request.root(), request.out_dir());
        std::fs::create_dir(&out_dir).expect("create out dir");
        // 失敗の模擬: プロジェクト直下に目印があれば、出力の残骸を残して異常終了する
        // （`root` は `<project>/candidates/<N>`）。
        if Path::new(request.root()).join("../../fail_worker").exists() {
            std::fs::write(format!("{out_dir}/partial.bin"), b"debris").expect("debris");
            std::process::exit(1);
        }
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
        let default_config = if request.kind() == "c1" {
            C1_CONFIG
        } else {
            C3_CONFIG
        };
        // 結果の `config` は「既定値を要求の設定で上書きしたもの」。`train --smoke` は `epochs` を 1 にする。
        let config = if request.config().get("epochs").is_some_and(|e| e == 1) {
            default_config
                .replace("\"epochs\":30", "\"epochs\":1")
                .replace("\"epochs\":40", "\"epochs\":1")
        } else {
            default_config.to_string()
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
            "evaluation_data_is_frozen_and_read_only",
            suite::evaluation_data_is_frozen_and_read_only,
        ),
        (
            "full_pipeline_completes_with_evaluation_data",
            suite::full_pipeline_completes_with_evaluation_data,
        ),
        (
            "evaluate_keeps_model_and_evaluation_hashes",
            suite::evaluate_keeps_model_and_evaluation_hashes,
        ),
        (
            "evaluate_correct_matches_infer_on_package",
            suite::evaluate_correct_matches_infer_on_package,
        ),
        (
            "evaluate_twice_is_rejected",
            suite::evaluate_twice_is_rejected,
        ),
        (
            "evaluate_rejects_selection_record_out_of_date",
            suite::evaluate_rejects_selection_record_out_of_date,
        ),
        (
            "evaluate_rejects_rewritten_selection_record",
            suite::evaluate_rejects_rewritten_selection_record,
        ),
        (
            "evaluate_rejects_untrained_and_smoke_trained_candidates",
            suite::evaluate_rejects_untrained_and_smoke_trained_candidates,
        ),
        (
            "package_rejects_when_selected_candidate_not_evaluated",
            suite::package_rejects_when_selected_candidate_not_evaluated,
        ),
        (
            "package_rejects_record_without_ledger_completion",
            suite::package_rejects_record_without_ledger_completion,
        ),
        (
            "package_rejects_smoke_candidate_with_allow_smoke_when_evaluation_data_exists",
            suite::package_rejects_smoke_candidate_with_allow_smoke_when_evaluation_data_exists,
        ),
        (
            "evaluate_and_package_reject_selection_switched_after_first_evaluation",
            suite::evaluate_and_package_reject_selection_switched_after_first_evaluation,
        ),
        (
            "evaluate_and_package_reject_selection_digest_changed_after_first_evaluation",
            suite::evaluate_and_package_reject_selection_digest_changed_after_first_evaluation,
        ),
        (
            "package_rejects_tampered_evaluation_record",
            suite::package_rejects_tampered_evaluation_record,
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
            "inspect_rejects_empty_validation_split_without_split_record",
            suite::inspect_rejects_empty_validation_split_without_split_record,
        ),
        (
            "train_failure_cleans_candidate_dir_and_allows_retry",
            suite::train_failure_cleans_candidate_dir_and_allows_retry,
        ),
        (
            "inspect_seed_is_recorded_and_used_by_train",
            suite::inspect_seed_is_recorded_and_used_by_train,
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
            "select_rejects_request_validation_not_matching_split",
            suite::select_rejects_request_validation_not_matching_split,
        ),
        (
            "package_rejects_mismatched_selection_and_metadata",
            suite::package_rejects_mismatched_selection_and_metadata,
        ),
        (
            "train_stops_on_frozen_hash_mismatch_without_outputs",
            suite::train_stops_on_frozen_hash_mismatch_without_outputs,
        ),
        (
            "select_stops_on_frozen_hash_mismatch_without_outputs",
            suite::select_stops_on_frozen_hash_mismatch_without_outputs,
        ),
        (
            "package_stops_on_frozen_hash_mismatch_without_outputs",
            suite::package_stops_on_frozen_hash_mismatch_without_outputs,
        ),
        (
            "later_stages_stop_when_freeze_record_is_missing",
            suite::later_stages_stop_when_freeze_record_is_missing,
        ),
        (
            "register_places_evaluation_data_via_write_probe",
            suite::register_places_evaluation_data_via_write_probe,
        ),
        (
            "select_rejects_request_not_matching_default_candidate",
            suite::select_rejects_request_not_matching_default_candidate,
        ),
        (
            "package_rejects_selection_record_rewritten_to_other_candidate",
            suite::package_rejects_selection_record_rewritten_to_other_candidate,
        ),
        (
            "package_rejects_non_onnx_model_with_matching_hash",
            suite::package_rejects_non_onnx_model_with_matching_hash,
        ),
        (
            "package_rejects_artifact_kind_version_mismatch",
            suite::package_rejects_artifact_kind_version_mismatch,
        ),
        (
            "select_rejects_request_root_outside_candidate_dir",
            suite::select_rejects_request_root_outside_candidate_dir,
        ),
        (
            "package_rejects_request_root_outside_candidate_dir",
            suite::package_rejects_request_root_outside_candidate_dir,
        ),
        (
            "package_rejects_artifact_dir_outside_candidate_dir",
            suite::package_rejects_artifact_dir_outside_candidate_dir,
        ),
        (
            "package_is_rejected_when_evaluation_data_exists_but_not_completed",
            suite::package_is_rejected_when_evaluation_data_exists_but_not_completed,
        ),
        (
            "package_rejects_smoke_trained_candidate_unless_allowed",
            suite::package_rejects_smoke_trained_candidate_unless_allowed,
        ),
        (
            "select_and_package_reject_request_fields_beyond_kind_and_seed",
            suite::select_and_package_reject_request_fields_beyond_kind_and_seed,
        ),
        (
            "select_rejects_vocab_file_without_matching_recorded_hash",
            suite::select_rejects_vocab_file_without_matching_recorded_hash,
        ),
        (
            "select_validates_candidate_before_capacity_exclusion",
            suite::select_validates_candidate_before_capacity_exclusion,
        ),
        (
            "select_oversized_vocab_is_validated_not_just_excluded",
            suite::select_oversized_vocab_is_validated_not_just_excluded,
        ),
        (
            "select_verifies_onnx_before_capacity_exclusion",
            suite::select_verifies_onnx_before_capacity_exclusion,
        ),
        (
            "vocab_in_artifact_dir_passes_select_package_and_infer",
            suite::vocab_in_artifact_dir_passes_select_package_and_infer,
        ),
        (
            "select_loads_onnx_before_capacity_exclusion",
            suite::select_loads_onnx_before_capacity_exclusion,
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
