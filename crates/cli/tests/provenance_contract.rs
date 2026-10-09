//! `register`・`inspect` の来歴の記録・検査（REQ-40・TASK-40.1・TASK-40.2・#475）。
//!
//! 証拠種別はテストハーネス。データはすべて合成で、実バイナリ `fandhe-edge` を cwd 固定で実行する。

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const LABELS: [&str; 3] = ["alpha", "beta", "gamma"];
const PROV_BASE: &str = r#""model_requested":"m","prompt_sha256":"30382f17d2a33e6e40c9a6ce38563083ab4c788c83785fd70bff6b7e9c03f19c","started_utc":"2026-09-24T00:14:33+00:00","usage_observed":null"#;

struct Env(PathBuf);

impl Env {
    fn new(case: &str) -> Self {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("prov-{case}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let options: Vec<String> = LABELS
            .iter()
            .map(|l| format!(r#"{{"id":"{l}","display_name":"{l}","description":"d"}}"#))
            .collect();
        let def = format!(
            r#"{{"schema":"fandhe-edge-model-definition/v1","name":"prov","version":1,"judgment_type":"single_select","options":[{}],"io":{{"input":"bytes"}}}}"#,
            options.join(",")
        );
        let mut train = String::new();
        for i in 0..30 {
            for l in LABELS {
                train.push_str(&format!(
                    "{{\"id\":\"{l}-{i}\",\"input\":\"{l} sample {i}\",\"output\":{{\"intent\":\"{l}\"}},\"group_id\":\"g-{l}-{i}\"}}\n"
                ));
            }
        }
        std::fs::write(dir.join("def.json"), def).expect("def");
        std::fs::write(dir.join("train.jsonl"), train).expect("train");
        Self(dir)
    }

    fn put(&self, name: &str, text: &str) {
        std::fs::write(self.0.join(name), text).expect("write");
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
            .current_dir(&self.0)
            .args(args)
            .output()
            .expect("spawn")
    }

    fn register(&self) -> Output {
        self.run(&[
            "register",
            "--definition",
            "def.json",
            "--project-dir",
            "proj",
        ])
    }

    fn inspect(&self) -> Output {
        self.run(&["inspect", "--project-dir", "proj"])
    }

    fn proj(&self) -> PathBuf {
        self.0.join("proj")
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn exists(p: &Path) -> bool {
    std::fs::symlink_metadata(p).is_ok()
}

/// REQ-40: 来歴なしは従来どおり成功し、`provenance_record.json` は作らない。
#[test]
fn req40_no_provenance_succeeds_without_record() {
    let e = Env::new("none");
    assert_eq!(e.register().status.code(), Some(0));
    assert_eq!(e.inspect().status.code(), Some(0));
    assert!(!exists(&e.proj().join("provenance_record.json")));
}

/// REQ-40: 正しい来歴は `data/` へ写され、`inspect` が正準 JSON を記録する。
#[test]
fn req40_valid_provenance_is_copied_and_recorded() {
    let e = Env::new("valid");
    e.put(
        "train.provenance.json",
        &format!(r#"{{{PROV_BASE},"source":"self"}}"#),
    );
    assert_eq!(e.register().status.code(), Some(0));
    assert!(exists(&e.proj().join("data/train.provenance.json")));
    assert_eq!(e.inspect().status.code(), Some(0));
    let rec = std::fs::read_to_string(e.proj().join("provenance_record.json")).expect("record");
    assert!(rec.starts_with(r#"{"train":{"#), "{rec}");
    assert!(rec.contains(r#""model_name":"m""#), "{rec}");
}

/// REQ-40: Jev 出力を生成元とする来歴は 64 で、プロジェクトを残さない。
#[test]
fn req40_jev_source_is_rejected_and_project_removed() {
    let e = Env::new("jev");
    e.put(
        "train.provenance.json",
        &format!(r#"{{{PROV_BASE},"source":"jev_output"}}"#),
    );
    let o = e.register();
    assert_eq!(o.status.code(), Some(64));
    assert!(stdout(&o).contains("training data source is not allowed"));
    assert!(!exists(&e.proj()));
}

/// REQ-40: 壊れた来歴 JSON は 64。
#[test]
fn req40_malformed_provenance_is_rejected() {
    let e = Env::new("broken");
    e.put("train.provenance.json", "{not json");
    let o = e.register();
    assert_eq!(o.status.code(), Some(64));
    assert!(stdout(&o).contains("provenance record is invalid"));
    assert!(!exists(&e.proj()));
}

/// REQ-40: 取り込み後に `data/*.provenance.json` を改変すると `inspect` は 64（fail-closed）。
#[test]
fn req40_tampered_provenance_fails_inspect() {
    let e = Env::new("tamper");
    e.put("train.provenance.json", &format!(r#"{{{PROV_BASE}}}"#));
    assert_eq!(e.register().status.code(), Some(0));
    let p = e.proj().join("data/train.provenance.json");
    std::fs::remove_file(&p).expect("rm");
    std::fs::write(&p, format!(r#"{{{PROV_BASE},"source":"jev_output"}}"#)).expect("tamper");
    let o = e.inspect();
    assert_eq!(o.status.code(), Some(64));
    assert!(!exists(&e.proj().join("split.json")));
}

fn put_eval(e: &Env) {
    let mut s = String::new();
    for (i, l) in LABELS.iter().enumerate() {
        s.push_str(&format!(
            "{{\"id\":\"ev-{i}\",\"input\":\"held out {l} case\",\"output\":{{\"intent\":\"{l}\"}},\"group_id\":\"ge-{i}\"}}\n"
        ));
    }
    e.put("evaluation.jsonl", &s);
}

/// REQ-40・REQ-39: ルート（cwd）外を指す symlink の来歴は拒否し、プロジェクトを残さない。
#[test]
fn req39_symlinked_provenance_is_rejected() {
    let e = Env::new("symlink");
    let outside = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("prov-outside.json");
    std::fs::write(&outside, format!(r#"{{{PROV_BASE}}}"#)).expect("outside");
    std::os::unix::fs::symlink(&outside, e.0.join("train.provenance.json")).expect("symlink");
    let o = e.register();
    assert_eq!(o.status.code(), Some(64), "{}", stdout(&o));
    assert!(!exists(&e.proj()));
}

/// REQ-39: 64 KiB を超える来歴は exit 20（`limit_exceeded`）。
#[test]
fn req39_oversized_provenance_is_limit_exceeded() {
    let e = Env::new("big");
    let pad = " ".repeat(64 * 1024 + 1);
    e.put("train.provenance.json", &format!(r#"{{{PROV_BASE}}}{pad}"#));
    let o = e.register();
    assert_eq!(o.status.code(), Some(20), "{}", stdout(&o));
    assert!(!exists(&e.proj()));
}

/// REQ-40: 不正 UTF-8・重複キー・`source: null` は 64。
#[test]
fn req40_invalid_utf8_duplicate_key_and_null_source_are_rejected() {
    let e = Env::new("shapes");
    std::fs::write(e.0.join("train.provenance.json"), [0xff, 0xfe, b'{', b'}']).expect("w");
    assert_eq!(e.register().status.code(), Some(64));
    e.put(
        "train.provenance.json",
        &format!(r#"{{{PROV_BASE},"source":"self","source":"self"}}"#),
    );
    assert_eq!(e.register().status.code(), Some(64));
    e.put(
        "train.provenance.json",
        &format!(r#"{{{PROV_BASE},"source":null}}"#),
    );
    let o = e.register();
    assert_eq!(o.status.code(), Some(64), "{}", stdout(&o));
    assert!(stdout(&o).contains("provenance record is invalid"));
}

/// REQ-17・REQ-40: 評価データ無しの評価来歴は 64（評価済みを装う記録を作らない）。
#[test]
fn req40_orphan_evaluation_provenance_is_rejected() {
    let e = Env::new("orphan");
    e.put("evaluation.provenance.json", &format!(r#"{{{PROV_BASE}}}"#));
    let o = e.register();
    assert_eq!(o.status.code(), Some(64));
    assert!(stdout(&o).contains("evaluation provenance without evaluation data"));
    assert!(!exists(&e.proj()));
}

/// REQ-40: 評価データなしで取り込んだ後に評価来歴を `data/` へ足しても、`inspect` は 64 で記録しない。
#[test]
fn req40_orphan_evaluation_provenance_added_after_register_fails_inspect() {
    let e = Env::new("orphan-late");
    assert_eq!(e.register().status.code(), Some(0));
    let p = e.proj().join("data/evaluation.provenance.json");
    std::fs::write(&p, format!(r#"{{{PROV_BASE}}}"#)).expect("add");
    let o = e.inspect();
    assert_eq!(o.status.code(), Some(64));
    assert!(stdout(&o).contains("evaluation provenance without evaluation data"));
    assert!(!exists(&e.proj().join("split.json")));
    assert!(!exists(&e.proj().join("provenance_record.json")));
}

/// REQ-40: 評価来歴は評価データとともに取り込まれ、`inspect` が `evaluation` キーで記録する。
#[test]
fn req40_evaluation_provenance_is_recorded() {
    let e = Env::new("evalprov");
    put_eval(&e);
    e.put("evaluation.provenance.json", &format!(r#"{{{PROV_BASE}}}"#));
    assert_eq!(e.register().status.code(), Some(0));
    assert!(exists(&e.proj().join("data/evaluation.provenance.json")));
    let o = e.inspect();
    assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    let rec = std::fs::read_to_string(e.proj().join("provenance_record.json")).expect("record");
    assert!(rec.starts_with(r#"{"evaluation":{"#), "{rec}");
}

/// REQ-40: Jev 不使用は学習データに限る（spec 受け入れ基準 2）。評価来歴は形式のみ検証。
#[test]
fn req40_jev_source_in_evaluation_provenance_is_format_checked_only() {
    let e = Env::new("evaljev");
    put_eval(&e);
    e.put(
        "evaluation.provenance.json",
        &format!(r#"{{{PROV_BASE},"source":"jev_output"}}"#),
    );
    assert_eq!(e.register().status.code(), Some(0));
    assert_eq!(e.inspect().status.code(), Some(0));
}
