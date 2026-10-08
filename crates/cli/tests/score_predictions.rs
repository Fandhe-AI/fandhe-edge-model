//! PoC-26 用の採点入口 `fandhe-edge-score` の結合テスト（REQ-41・TASK-41.1・#445。REQ-17・REQ-24・REQ-25・
//! REQ-27・REQ-39 の不変条件を機械照合する）。
//!
//! 証拠種別: テストハーネス。データはすべて本ファイルが生成する合成データ（個人情報・機密なし）で、
//! 学習は行わない（`register → inspect` までで majority・凍結 test が揃う）。予測ファイルは本ファイルが
//! 書く（候補 P の `pred.jsonl` の代わり）。

#![cfg(unix)]

use std::path::PathBuf;
use std::process::{Command, Output};

const LABELS: [&str; 3] = ["alpha", "beta", "gamma"];
/// 評価データの件数（alpha 150・beta 75・gamma 75）。
const N_EVAL: usize = 300;

fn gold_of(i: usize) -> &'static str {
    match i {
        0..150 => "alpha",
        150..225 => "beta",
        _ => "gamma",
    }
}

fn definition_text() -> String {
    let options: Vec<String> = LABELS
        .iter()
        .map(|l| format!(r#"{{"id":"{l}","display_name":"{l}","description":"dummy"}}"#))
        .collect();
    format!(
        r#"{{"schema":"fandhe-edge-model-definition/v1","name":"score_predictions","version":1,"judgment_type":"single_select","options":[{}],"io":{{"input":"bytes"}}}}"#,
        options.join(",")
    )
}

/// 学習データ（alpha 40・beta 10・gamma 10。majority は alpha）。
fn train_jsonl() -> String {
    let mut out = String::new();
    for (label, n) in [("alpha", 40), ("beta", 10), ("gamma", 10)] {
        for i in 0..n {
            out.push_str(&format!(
                r#"{{"id":"{label}-{i}","input":"{label} sample {i}","output":{{"intent":"{label}"}},"group_id":"g-{label}-{i}"}}"#
            ));
            out.push('\n');
        }
    }
    out
}

fn eval_id(i: usize) -> String {
    format!("e-{i}")
}

fn evaluation_jsonl() -> String {
    (0..N_EVAL)
        .map(|i| {
            format!(
                r#"{{"id":"{}","input":"evaluation {i}","output":{{"intent":"{}"}},"group_id":"eg-{i}"}}"#,
                eval_id(i),
                gold_of(i)
            ) + "\n"
        })
        .collect()
}

/// 予測ファイルの本文。`label_of(i)` が予測ラベル（`None` は `status:"error"`）。
fn pred_jsonl(label_of: impl Fn(usize) -> Option<&'static str>) -> String {
    (0..N_EVAL)
        .map(|i| match label_of(i) {
            Some(l) => format!(
                r#"{{"id":"{}","status":"ok","predicted_label":"{l}","scores":{{"{l}":1.0}}}}"#,
                eval_id(i)
            ),
            None => format!(
                r#"{{"id":"{}","status":"error","predicted_label":null}}"#,
                eval_id(i)
            ),
        } + "\n")
        .collect()
}

/// 候補 P: 先頭 30 件（alpha）だけ beta と誤る（正解 270）。
fn pred_p(i: usize) -> Option<&'static str> {
    Some(if i < 30 { "beta" } else { gold_of(i) })
}

/// 比較相手 A: 先頭 60 件（alpha）だけ gamma と誤る（正解 240）。
fn pred_a(i: usize) -> Option<&'static str> {
    Some(if i < 60 { "gamma" } else { gold_of(i) })
}

/// テスト用の sha256（`shasum -a 256` を使い、cli の実装を再利用しない独立の計算にする）。
fn sha256_hex(bytes: &[u8]) -> String {
    use std::io::Write;
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("shasum");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(bytes)
        .expect("write");
    let out = child.wait_with_output().expect("wait");
    String::from_utf8(out.stdout).expect("utf8")[..64].to_string()
}

/// 合成の評価記録の来歴欄。
struct Rec {
    candidate_id: String,
    index: usize,
    evaluation_sha256: String,
    evaluation_bytes: u64,
    definition_sha256: String,
}

struct Env {
    base: PathBuf,
    work: PathBuf,
}

impl Env {
    /// `register → inspect` まで済ませた作業ディレクトリ（cwd = `work/`、プロジェクトは `proj`）。
    fn new(case: &str) -> Self {
        let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("score-predictions-{case}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let work = base.join("work");
        std::fs::create_dir_all(work.join("def")).expect("work dir");
        std::fs::write(work.join("def/definition.json"), definition_text()).expect("definition");
        std::fs::write(work.join("def/train.jsonl"), train_jsonl()).expect("train");
        std::fs::write(work.join("def/evaluation.jsonl"), evaluation_jsonl()).expect("evaluation");
        let env = Self { base, work };
        for args in [
            &[
                "register",
                "--definition",
                "def/definition.json",
                "--project-dir",
                "proj",
            ][..],
            &["inspect", "--project-dir", "proj"][..],
        ] {
            let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
                .args(args)
                .current_dir(&env.work)
                .output()
                .expect("run fandhe-edge");
            assert_eq!(out.status.code(), Some(0), "{args:?}");
        }
        env
    }

    fn write(&self, name: &str, body: &str) {
        std::fs::write(self.work.join(name), body).expect("write pred");
    }

    /// `proj` の現在の状態（凍結記録・定義）に正しく束縛した合成の評価記録の材料。
    fn rec(&self, proj: &str, candidate_id: &str, index: usize) -> Rec {
        let eval = std::fs::read(self.work.join(proj).join("data/evaluation.jsonl")).expect("eval");
        let def = std::fs::read_to_string(self.work.join(proj).join("definition.json"))
            .expect("definition");
        Rec {
            candidate_id: candidate_id.to_string(),
            index,
            evaluation_sha256: sha256_hex(&eval),
            evaluation_bytes: eval.len() as u64,
            definition_sha256: fandhe_edge_core::definition::Definition::parse(&def)
                .expect("parse")
                .canonical_hash()
                .expect("hash")
                .to_hex(),
        }
    }

    /// `evaluate` が保存した予測の代わり（C1・C3・AR 用）。`proj/candidates/<index>/` に予測ファイルと、
    /// その sha256 と現在のプロジェクトの来歴を束縛した合成の評価記録を置き、予測ファイルの相対パスを返す。
    fn bound(&self, candidate_id: &str, index: usize, body: &str) -> String {
        self.bound_rec("proj", &self.rec("proj", candidate_id, index), body)
    }

    fn bound_rec(&self, proj: &str, rec: &Rec, body: &str) -> String {
        let dir = format!("{proj}/candidates/{}", rec.index);
        self.write_in(&dir, "evaluation_predictions.jsonl", body);
        self.write_in(
            &dir,
            "evaluation_record.json",
            &format!(
                r#"{{"candidate_index":{},"candidate_id":"{}","config_id":"{}:seed0","evaluation_sha256":"{}","evaluation_bytes":{},"onnx_sha256":"{z}","artifact_meta_sha256":"{z}","definition_sha256":"{}","correct":1,"total":1,"predictions_sha256":"{}"}}"#,
                rec.index,
                rec.candidate_id,
                rec.candidate_id,
                rec.evaluation_sha256,
                rec.evaluation_bytes,
                rec.definition_sha256,
                sha256_hex(body.as_bytes()),
                z = "0".repeat(64)
            ),
        );
        format!("{dir}/evaluation_predictions.jsonl")
    }

    fn write_in(&self, dir: &str, name: &str, body: &str) {
        std::fs::create_dir_all(self.work.join(dir)).expect("dir");
        self.write(&format!("{dir}/{name}"), body);
    }

    fn score(&self, args: &[&str]) -> (i32, String) {
        let out: Output = Command::new(env!("CARGO_BIN_EXE_fandhe-edge-score"))
            .args(args)
            .current_dir(&self.work)
            .output()
            .expect("run fandhe-edge-score");
        let stdout = String::from_utf8(out.stdout).expect("utf8");
        assert_eq!(stdout.matches('\n').count(), 1, "one JSON line: {stdout}");
        (out.status.code().expect("exit code"), stdout)
    }

    fn ok(&self, args: &[&str]) -> String {
        let (code, stdout) = self.score(args);
        assert_eq!(code, 0, "{stdout}");
        stdout
    }

    fn fails(&self, args: &[&str], message: &str) {
        let (code, stdout) = self.score(args);
        assert_eq!(code, 64, "{stdout}");
        assert_eq!(
            stdout,
            format!("{{\"code\":\"invalid_input\",\"message\":\"{message}\"}}\n")
        );
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.base);
    }
}

const BASE: [&str; 4] = ["--project-dir", "proj", "--seed", "0"];

fn args<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    BASE.iter().chain(extra).copied().collect()
}

/// REQ-41・REQ-24・REQ-25: 正常系。正解数・正解率・対 majority の b・c と、Holm の族の大きさ（m=3）が
/// 具体値で一致する。majority は train のラベルだけから作る alpha（正解 150）。
#[test]
fn req41_scores_candidate_with_expected_counts() {
    let env = Env::new("ok");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    let a = env.bound("c1", 1, &pred_jsonl(pred_a));
    let out = env.ok(&args(&[
        "--candidate",
        "P=p.jsonl",
        "--compare",
        &format!("C1={a}"),
    ]));
    assert!(
        out.starts_with("{\"step\":\"score_predictions\",\"status\":\"ok\",\"seed\":0,"),
        "{out}"
    );
    assert!(
        out.contains("\"n_total\":300,\"required_sample_size\":215,"),
        "{out}"
    );
    assert!(
        out.contains("\"name\":\"P\",\"role\":\"candidate\","),
        "{out}"
    );
    assert!(out.contains("\"correct\":270,\"accuracy\":0.9,"), "{out}");
    assert!(
        out.contains("\"vs_majority\":{\"b\":150,\"c\":30,"),
        "{out}"
    );
    // 比較相手 A の正解は 240。
    assert!(
        out.contains("\"name\":\"C1\",\"role\":\"compare\","),
        "{out}"
    );
    assert!(out.contains("\"correct\":240,\"accuracy\":0.8,"), "{out}");
    // Holm: 対 majority と対 A。b・c は P を基準にした値（P のみ正解 30・A のみ正解 0）。
    assert!(
        out.contains("\"holm\":{\"candidate\":\"P\",\"m\":3,\"comparisons\":["),
        "{out}"
    );
    assert!(
        out.contains("{\"against\":\"C1\",\"b\":30,\"c\":0,\"p_raw\":"),
        "{out}"
    );
    // 混同行列の P 行（alpha の正解 150 件のうち 30 件を beta、120 件を alpha と予測）。
    assert!(
        out.contains("\"rows\":[[120,30,0,0,0,0],[0,75,0,0,0,0],[0,0,75,0,0,0]]"),
        "{out}"
    );
    assert!(out.contains("\"references\":[]}"), "{out}");
}

/// REQ-25: 比較相手が 0 でも 1 でも、Holm の族の大きさは 3 固定（m は相手の数に依存しない）。
#[test]
fn req25_holm_family_size_is_three_even_without_compares() {
    let env = Env::new("holm_m");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    let out = env.ok(&args(&["--candidate", "P=p.jsonl"]));
    assert!(
        out.contains(
            "\"holm\":{\"candidate\":\"P\",\"m\":3,\"comparisons\":[{\"against\":\"majority\","
        ),
        "{out}"
    );
    assert_eq!(out.matches("\"against\":").count(), 1, "{out}");
}

/// REQ-25: `--reference` は族に入れず、McNemar の生の b・c・p だけを返す。
#[test]
fn req25_reference_is_outside_the_holm_family() {
    let env = Env::new("reference");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    let a = env.bound("c3", 2, &pred_jsonl(pred_a));
    let out = env.ok(&args(&[
        "--candidate",
        "P=p.jsonl",
        "--reference",
        &format!("C3={a}"),
    ]));
    assert!(
        out.contains(
            "\"references\":[{\"candidate\":\"P\",\"against\":\"C3\",\"b\":30,\"c\":0,\"p_raw\":"
        ),
        "{out}"
    );
    // Holm の比較は対 majority の 1 件のみ。
    assert!(
        out.contains("\"comparisons\":[{\"against\":\"majority\","),
        "{out}"
    );
    assert_eq!(out.matches("\"against\":").count(), 2, "{out}");
}

/// REQ-17: 凍結 test のハッシュが記録と一致しなければ停止する（fail-closed）。
#[test]
fn req17_stops_on_frozen_hash_mismatch() {
    let env = Env::new("hash");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    let eval_file = env.work.join("proj/data/evaluation.jsonl");
    let mut perm = std::fs::metadata(&eval_file).expect("meta").permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o600);
    std::fs::set_permissions(&eval_file, perm).expect("chmod");
    let mut bytes = std::fs::read(&eval_file).expect("read");
    let at = bytes.windows(5).position(|w| w == b"alpha").expect("alpha");
    bytes[at + 4] = b'b';
    std::fs::write(&eval_file, bytes).expect("tamper");
    let (code, stdout) = env.score(&args(&["--candidate", "P=p.jsonl"]));
    assert_eq!(code, 64, "{stdout}");
    assert!(
        stdout.starts_with("{\"code\":\"invalid_input\",\"message\":\"eval data hash mismatch"),
        "{stdout}"
    );
    // 停止したので台帳は作られない。
    assert!(!env.work.join("proj/poc26_score_ledger").exists());
}

/// REQ-23: 予測の id 重複・欠落（空ファイルは全件欠落＝全件エラーではなく不正 JSON 等の停止）で停止する。
#[test]
fn req23_stops_on_duplicate_id_and_malformed_json() {
    let env = Env::new("stop");
    let mut dup = pred_jsonl(pred_p);
    dup.push_str(&format!(
        "{{\"id\":\"{}\",\"status\":\"ok\",\"predicted_label\":\"alpha\"}}\n",
        eval_id(0)
    ));
    env.write("dup.jsonl", &dup);
    env.fails(
        &args(&["--candidate", "P=dup.jsonl"]),
        "prediction input rejected for P: duplicate_id",
    );
    env.write("bad.jsonl", "not json\n");
    env.fails(
        &args(&["--candidate", "P=bad.jsonl"]),
        "prediction input rejected for P: malformed_json",
    );
    assert!(!env.work.join("proj/poc26_score_ledger").exists());
}

/// REQ-23: pred に行が無い id は不正解（エラー）として分母に数える。
#[test]
fn req23_missing_prediction_counts_as_wrong() {
    let env = Env::new("missing");
    // 最後の 10 件（gamma）の行を欠落させる。
    let body: String = pred_jsonl(|i| Some(gold_of(i)))
        .lines()
        .take(N_EVAL - 10)
        .map(|l| format!("{l}\n"))
        .collect();
    env.write("p.jsonl", &body);
    let out = env.ok(&args(&["--candidate", "P=p.jsonl"]));
    assert!(out.contains("\"n_total\":300,"), "{out}");
    assert!(out.contains("\"correct\":290,"), "{out}");
}

/// REQ-23: 不正な scores の行は不正解に数える（status・predicted_label が正しくても）。
#[test]
fn req23_invalid_scores_count_as_wrong() {
    let env = Env::new("scores");
    let body: String = pred_jsonl(|i| Some(gold_of(i)))
        .lines()
        .enumerate()
        .map(|(i, l)| {
            if i < 5 {
                // 合計が 1 でない scores。
                l.replace("\"scores\":{", "\"scores\":{\"zzz\":5.0,") + "\n"
            } else {
                format!("{l}\n")
            }
        })
        .collect();
    env.write("p.jsonl", &body);
    let out = env.ok(&args(&["--candidate", "P=p.jsonl"]));
    assert!(out.contains("\"correct\":295,"), "{out}");
}

/// 台帳ファイルのパス（凍結 test の sha256 のディレクトリ配下。唯一のサブディレクトリを探す）。
fn ledger_file(env: &Env, seed: u32, name: &str) -> PathBuf {
    let root = env.work.join("proj/poc26_score_ledger");
    let sub: Vec<_> = std::fs::read_dir(&root)
        .expect("ledger root")
        .map(|e| e.expect("entry").path())
        .collect();
    assert_eq!(sub.len(), 1, "one evaluation sha256 dir");
    assert_eq!(sub[0].file_name().expect("name").len(), 64);
    sub[0].join(format!("seed-{seed}/{name}.sha256"))
}

/// REQ-27: 台帳。同一 candidate の 2 回目は（同じ sha256 でも）拒否し、compare は同じ sha256 の
/// 再読込を許可し、別の sha256 は拒否する。別 seed の台帳は独立。
#[test]
fn req27_ledger_enforces_single_application() {
    let env = Env::new("ledger");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    let a = env.bound("c1", 1, &pred_jsonl(pred_a));
    let c1 = format!("C1={a}");
    env.ok(&args(&["--candidate", &c1]));
    assert!(ledger_file(&env, 0, "C1").is_file());
    env.fails(
        &args(&["--candidate", &c1]),
        "candidate has already been scored",
    );
    env.ok(&args(&["--candidate", "P=p.jsonl", "--compare", &c1]));
    let a2 = env.bound(
        "c1",
        4,
        &pred_jsonl(|i| Some(if i < 61 { "gamma" } else { gold_of(i) })),
    );
    let ar = env.bound(
        "autoregressive",
        3,
        &pred_jsonl(|i| Some(if i < 10 { "gamma" } else { gold_of(i) })),
    );
    env.fails(
        &args(&[
            "--candidate",
            &format!("AR={ar}"),
            "--compare",
            &format!("C1={a2}"),
        ]),
        "prediction file differs from the one recorded for this name",
    );
    assert!(!ledger_file(&env, 0, "AR").exists());
    env.ok(&["--project-dir", "proj", "--seed", "1", "--candidate", &c1]);
}

/// REQ-27: 同じ seed で、同じ sha256 の予測ファイルを別 NAME の採点対象として出すと拒否する。
#[test]
fn req27_same_sha256_under_another_name_is_rejected() {
    let env = Env::new("othername");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    let c3 = format!("C3={}", env.bound("c3", 2, &pred_jsonl(pred_p)));
    env.ok(&args(&["--candidate", "P=p.jsonl"]));
    env.fails(
        &args(&["--candidate", &c3]),
        "prediction file has already been scored under another name",
    );
    // 別 seed なら独立。
    env.ok(&["--project-dir", "proj", "--seed", "2", "--candidate", &c3]);
}

/// REQ-27: 事前登録外の NAME・seed と、reference 2 個を拒否する（適用は最大 4 候補 × 3 seed）。
#[test]
fn req27_rejects_names_seeds_and_references_outside_preregistration() {
    let env = Env::new("prereg");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    for name in ["X", "p", "c1", "ar", "C2"] {
        env.fails(
            &args(&["--candidate", &format!("{name}=p.jsonl")]),
            "name is not in the preregistered list",
        );
    }
    env.fails(
        &[
            "--project-dir",
            "proj",
            "--seed",
            "3",
            "--candidate",
            "P=p.jsonl",
        ],
        "seed is not in the preregistered list",
    );
    env.fails(
        &args(&[
            "--candidate",
            "P=p.jsonl",
            "--reference",
            "C1=p.jsonl",
            "--reference",
            "C3=p.jsonl",
        ]),
        "too many --reference options",
    );
    assert!(!env.work.join("proj/poc26_score_ledger").exists());
}

/// REQ-39: 経路の閉じ込め（`../`・絶対パス）と NAME の文字種を拒否する。
#[test]
fn req39_rejects_escaping_paths_and_bad_names() {
    let env = Env::new("guard");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    let (code, stdout) = env.score(&args(&["--candidate", "P=../p.jsonl"]));
    assert_eq!(code, 64, "{stdout}");
    assert!(
        stdout.starts_with("{\"code\":\"invalid_input\",\"message\":\"path rejected"),
        "{stdout}"
    );
    // cwd の外にある絶対パスと、外を指す symlink は拒否する（cwd 内の絶対パスは許可）。
    let outside = env.base.join("outside.jsonl");
    std::fs::write(&outside, pred_jsonl(pred_p)).expect("outside");
    let (code, stdout) = env.score(&args(&["--candidate", &format!("P={}", outside.display())]));
    assert_eq!(code, 64, "{stdout}");
    std::os::unix::fs::symlink(&outside, env.work.join("link.jsonl")).expect("symlink");
    let (code, stdout) = env.score(&args(&["--candidate", "P=link.jsonl"]));
    assert_eq!(code, 64, "{stdout}");
    env.fails(
        &args(&["--candidate", "P/x=p.jsonl"]),
        "name must match [A-Za-z0-9_-]{1,32}",
    );
    env.fails(
        &args(&["--candidate", "P=p.jsonl", "--compare", "P=p.jsonl"]),
        "duplicate NAME",
    );
    assert!(!env.work.join("proj/poc26_score_ledger").exists());
}

/// REQ-27・#445: 予測ファイルを 1 バイト変える（0400 を外して書き換え）と、評価記録の sha256 と合わず
/// 拒否される。拒否は台帳の書き込みより前で、台帳は作られない。
#[test]
fn req27_issue445_tampered_prediction_is_rejected() {
    let env = Env::new("tamper");
    let c1 = env.bound("c1", 1, &pred_jsonl(pred_a));
    let file = env.work.join(&c1);
    let mut body = std::fs::read(&file).expect("read");
    let at = body.windows(5).position(|w| w == b"alpha").expect("alpha");
    body[at + 4] = b'b';
    std::fs::write(&file, body).expect("tamper");
    env.fails(
        &args(&["--candidate", &format!("C1={c1}")]),
        "prediction file does not match the evaluation record",
    );
    assert!(!env.work.join("proj/poc26_score_ledger").exists());
}

/// REQ-27・#445: 評価記録が無い・`predictions_sha256` の欄が無い C1 は拒否する。P は評価記録なしで通る
/// （`p.jsonl` は上の各テストが評価記録なしで採点している）。
#[test]
fn req27_issue445_missing_record_or_field_is_rejected() {
    let env = Env::new("norecord");
    env.write_in(
        "proj/candidates/1",
        "evaluation_predictions.jsonl",
        &pred_jsonl(pred_a),
    );
    env.fails(
        &args(&[
            "--candidate",
            "C1=proj/candidates/1/evaluation_predictions.jsonl",
        ]),
        "evaluation record is missing for the prediction file",
    );
    env.write(
        "proj/candidates/1/evaluation_record.json",
        &format!(
            r#"{{"candidate_index":1,"candidate_id":"c1","config_id":"c1:seed0","evaluation_sha256":"{z}","evaluation_bytes":1,"onnx_sha256":"{z}","artifact_meta_sha256":"{z}","definition_sha256":"{z}","correct":1,"total":1}}"#,
            z = "0".repeat(64)
        ),
    );
    env.fails(
        &args(&[
            "--candidate",
            "C1=proj/candidates/1/evaluation_predictions.jsonl",
        ]),
        "prediction file does not match the evaluation record",
    );
    assert!(!env.work.join("proj/poc26_score_ledger").exists());
}

/// 来歴の不一致（別プロジェクト・凍結記録・定義・候補・配置）は、sha256 が合っていても拒否し、
/// 台帳を作らない（#445・REQ-27）。
#[test]
fn req27_issue445_provenance_mismatch_is_rejected() {
    let env = Env::new("prov");
    let body = pred_jsonl(pred_a);
    let no_ledger = || assert!(!env.work.join("proj/poc26_score_ledger").exists());
    let check = |path: &str, name: &str, message: &str| {
        env.fails(&args(&["--candidate", &format!("{name}={path}")]), message);
        no_ledger();
    };
    let mismatch = "evaluation record does not belong to this project state";

    // 別プロジェクト（凍結 test が違う）の記録と予測。
    std::fs::create_dir_all(env.work.join("other/data")).expect("other");
    std::fs::write(env.work.join("other/data/evaluation.jsonl"), "x\n").expect("eval");
    std::fs::write(env.work.join("other/definition.json"), definition_text()).expect("def");
    let rec = env.rec("other", "c1", 1);
    check(&env.bound_rec("proj", &rec, &body), "C1", mismatch);

    // evaluation_sha256・evaluation_bytes・definition_sha256 の個別の改変。
    let mut rec = env.rec("proj", "c1", 1);
    rec.evaluation_sha256 = "1".repeat(64);
    check(&env.bound_rec("proj", &rec, &body), "C1", mismatch);
    let mut rec = env.rec("proj", "c1", 1);
    rec.evaluation_bytes += 1;
    check(&env.bound_rec("proj", &rec, &body), "C1", mismatch);
    let mut rec = env.rec("proj", "c1", 1);
    rec.definition_sha256 = "2".repeat(64);
    check(&env.bound_rec("proj", &rec, &body), "C1", mismatch);

    // C1 に c3 の記録、AR に c1 の記録。
    let c3 = env.bound("c3", 1, &body);
    check(
        &c3,
        "C1",
        "evaluation record does not match the candidate name",
    );
    let c1 = env.bound("c1", 1, &body);
    check(
        &c1,
        "AR",
        "evaluation record does not match the candidate name",
    );

    // 記録の candidate_index と置き場所の N の不一致。
    let path = env.bound("c1", 9, &body);
    let record9 = env.work.join("proj/candidates/9/evaluation_record.json");
    let text = std::fs::read_to_string(&record9).expect("record");
    std::fs::write(
        &record9,
        text.replace("\"candidate_index\":9", "\"candidate_index\":1"),
    )
    .expect("rewrite");
    check(
        &path,
        "C1",
        "evaluation record does not match the candidate name",
    );

    // candidates の外（プロジェクト直下・別ディレクトリ）に置いた予測と記録。
    env.write_in("proj/elsewhere", "evaluation_predictions.jsonl", &body);
    std::fs::copy(
        env.work.join("proj/candidates/1/evaluation_record.json"),
        env.work.join("proj/elsewhere/evaluation_record.json"),
    )
    .expect("copy");
    let outside = "prediction file is not under the project candidates directory";
    check("proj/elsewhere/evaluation_predictions.jsonl", "C1", outside);
    let bad_name = env.bound("c1", 1, &body);
    let copy = env.work.join("proj/candidates/1/copy.jsonl");
    std::fs::copy(env.work.join(&bad_name), &copy).expect("copy");
    check("proj/candidates/1/copy.jsonl", "C1", outside);
}

/// REQ-27・#445: 公開 API から `ScoreArgs` を直接組んでも、`run` が事前登録の制限（seed・NAME）を
/// 検証し、拒否したときは台帳を作らない。
#[test]
fn req27_issue445_run_revalidates_public_args() {
    use fandhe_edge_cli::score_predictions::{NamedPath, ScoreArgs, run};
    let env = Env::new("api");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    let named = |name: &str| NamedPath {
        name: name.to_string(),
        path: PathBuf::from("p.jsonl"),
    };
    let build = |seed: u32, name: &str| ScoreArgs {
        project_dir: PathBuf::from("proj"),
        seed,
        candidate: named(name),
        compares: vec![],
        references: vec![],
    };
    for (args, message) in [
        (build(3, "P"), "seed is not in the preregistered list"),
        (build(0, "X"), "name is not in the preregistered list"),
    ] {
        let err = run(&args, &env.work).expect_err("rejected");
        assert_eq!(
            err.to_json_line().expect("json"),
            format!("{{\"code\":\"invalid_input\",\"message\":\"{message}\"}}")
        );
    }
    let mut many = build(0, "P");
    many.compares = vec![named("C1"), named("C3"), named("AR")];
    assert!(run(&many, &env.work).is_err());
    let mut dup = build(0, "P");
    dup.references = vec![named("P")];
    assert!(run(&dup, &env.work).is_err());
    assert!(!env.work.join("proj/poc26_score_ledger").exists());
}

/// REQ-27・#445: 未記録の相手（`--compare`）と同じバイト列を別 NAME の採点対象にすると、台帳の書き込み前に
/// 拒否され、台帳が作られない。
#[test]
fn req27_issue445_same_bytes_in_one_invocation_under_two_names_is_rejected() {
    let env = Env::new("samebytes");
    let c1 = env.bound("c1", 1, &pred_jsonl(pred_a));
    env.write("p.jsonl", &pred_jsonl(pred_a));
    env.fails(
        &args(&["--candidate", "P=p.jsonl", "--compare", &format!("C1={c1}")]),
        "the same prediction file is used under more than one name",
    );
    assert!(!env.work.join("proj/poc26_score_ledger").exists());
}

/// REQ-27・#445: 台帳ディレクトリが既にある状態でも採点できる。symlink が先にあれば拒否する。
#[test]
fn req27_issue445_existing_ledger_dir_is_tolerated_but_symlink_is_rejected() {
    let env = Env::new("ledgerdir");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    std::fs::create_dir(env.work.join("proj/poc26_score_ledger")).expect("mkdir");
    env.ok(&args(&["--candidate", "P=p.jsonl"]));
    let c1 = env.bound("c1", 1, &pred_jsonl(pred_a));
    env.ok(&args(&["--candidate", &format!("C1={c1}")]));

    let env = Env::new("ledgersym");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    std::fs::create_dir(env.base.join("elsewhere")).expect("mkdir");
    std::os::unix::fs::symlink(
        env.base.join("elsewhere"),
        env.work.join("proj/poc26_score_ledger"),
    )
    .expect("symlink");
    let (code, stdout) = env.score(&args(&["--candidate", "P=p.jsonl"]));
    assert_eq!(code, 64, "{stdout}");
    assert_eq!(
        std::fs::read_dir(env.base.join("elsewhere"))
            .expect("read")
            .count(),
        0
    );
}
