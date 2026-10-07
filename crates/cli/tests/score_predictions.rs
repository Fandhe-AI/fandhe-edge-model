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
    env.write("a.jsonl", &pred_jsonl(pred_a));
    let out = env.ok(&args(&[
        "--candidate",
        "P=p.jsonl",
        "--compare",
        "C1=a.jsonl",
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
    env.write("a.jsonl", &pred_jsonl(pred_a));
    let out = env.ok(&args(&[
        "--candidate",
        "P=p.jsonl",
        "--reference",
        "C3=a.jsonl",
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
    env.write("a.jsonl", &pred_jsonl(pred_a));
    env.ok(&args(&["--candidate", "C1=a.jsonl"]));
    assert!(ledger_file(&env, 0, "C1").is_file());
    env.fails(
        &args(&["--candidate", "C1=a.jsonl"]),
        "candidate has already been scored",
    );
    env.ok(&args(&[
        "--candidate",
        "P=p.jsonl",
        "--compare",
        "C1=a.jsonl",
    ]));
    env.write(
        "a2.jsonl",
        &pred_jsonl(|i| Some(if i < 61 { "gamma" } else { gold_of(i) })),
    );
    env.write(
        "c.jsonl",
        &pred_jsonl(|i| Some(if i < 10 { "gamma" } else { gold_of(i) })),
    );
    env.fails(
        &args(&["--candidate", "AR=c.jsonl", "--compare", "C1=a2.jsonl"]),
        "prediction file differs from the one recorded for this name",
    );
    assert!(!ledger_file(&env, 0, "AR").exists());
    env.ok(&[
        "--project-dir",
        "proj",
        "--seed",
        "1",
        "--candidate",
        "C1=a.jsonl",
    ]);
}

/// REQ-27: 同じ seed で、同じ sha256 の予測ファイルを別 NAME の採点対象として出すと拒否する。
#[test]
fn req27_same_sha256_under_another_name_is_rejected() {
    let env = Env::new("othername");
    env.write("p.jsonl", &pred_jsonl(pred_p));
    env.ok(&args(&["--candidate", "P=p.jsonl"]));
    env.fails(
        &args(&["--candidate", "C3=p.jsonl"]),
        "prediction file has already been scored under another name",
    );
    // 別 seed なら独立。
    env.ok(&[
        "--project-dir",
        "proj",
        "--seed",
        "2",
        "--candidate",
        "C3=p.jsonl",
    ]);
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
