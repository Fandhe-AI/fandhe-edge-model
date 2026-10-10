//! `register --previous-project-dir` の作り直し判定（REQ-20・TASK-20.1〜20.3・REQ-21・REQ-33・REQ-39・#487）。
//!
//! PoC-19 の 5 パターン（`fixtures/rebuild/poc19/`）を実バイナリ `fandhe-edge` で通し、stdout の JSON を
//! 完全一致で照合する。証拠種別はテストハーネス（データはすべて合成）。

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use fandhe_edge_core::definition::Definition;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/rebuild/poc19");
const TRAIN_A: &str = "{\"id\":\"a\",\"input\":\"x\",\"output\":{\"intent\":\"tier-xs__low\"}}\n";
const TRAIN_B: &str = "{\"id\":\"b\",\"input\":\"y\",\"output\":{\"intent\":\"tier-xs__low\"}}\n";
const EVAL_A: &str = "{\"id\":\"e1\",\"input\":\"z\",\"output\":{\"intent\":\"tier-xs__low\"}}\n";
const EVAL_B: &str = "{\"id\":\"e2\",\"input\":\"w\",\"output\":{\"intent\":\"tier-xs__low\"}}\n";

/// cwd に使う作業ディレクトリ（テストごとに独立）。
struct Env(PathBuf);

impl Env {
    fn new(case: &str) -> Self {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("rebuild-{case}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        Self(dir)
    }

    /// `src/` に定義（fixture 名）・学習データ・任意の評価データを置く。
    fn source(&self, src: &str, fixture: &str, train: &str, eval: Option<&str>) {
        let d = self.0.join(src);
        std::fs::create_dir_all(&d).expect("mkdir src");
        std::fs::copy(Path::new(FIXTURES).join(fixture), d.join("def.json")).expect("copy def");
        std::fs::write(d.join("train.jsonl"), train).expect("train");
        if let Some(eval) = eval {
            std::fs::write(d.join("evaluation.jsonl"), eval).expect("eval");
        }
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
            .current_dir(&self.0)
            .args(args)
            .output()
            .expect("spawn")
    }

    /// 旧プロジェクト `oldproj` を作る。
    fn register_old(&self, fixture: &str, train: &str, eval: Option<&str>) {
        self.source("old", fixture, train, eval);
        let o = self.run(&[
            "register",
            "--definition",
            "old/def.json",
            "--project-dir",
            "oldproj",
        ]);
        assert_eq!(o.status.code(), Some(0), "{}", stdout(&o));
    }

    /// `new/` の定義で `proj` を作り、`OLD` と比べる。
    fn register_new(&self, project_dir: &str, previous: &str) -> Output {
        self.run(&[
            "register",
            "--definition",
            "new/def.json",
            "--project-dir",
            project_dir,
            "--previous-project-dir",
            previous,
        ])
    }

    fn exists(&self, rel: &str) -> bool {
        std::fs::symlink_metadata(self.0.join(rel)).is_ok()
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

/// fixture の定義の正準化ハッシュ（16 進）。
fn hash(fixture: &str) -> String {
    let text = std::fs::read_to_string(Path::new(FIXTURES).join(fixture)).expect("fixture");
    Definition::parse(&text)
        .expect("definition")
        .canonical_hash()
        .expect("hash")
        .to_hex()
}

/// 期待する stdout（1 行 JSON＋改行）。`rebuild` は JSON 断片で渡す。
fn expected(fixture: &str, options: usize, eval: bool, rebuild: &str) -> String {
    format!(
        "{{\"step\":\"register\",\"status\":\"ok\",\"definition_sha256\":\"{}\",\"options\":{options},\"evaluation_defined\":{eval},\"rebuild\":{rebuild}}}\n",
        hash(fixture)
    )
}

/// 64 の拒否で、`proj` を作っていないこと。
fn assert_rejected(e: &Env, o: &Output, message: &str) {
    assert_eq!(o.status.code(), Some(64), "{}", stdout(o));
    assert!(stdout(o).contains(message), "{}", stdout(o));
    assert!(!e.exists("proj"));
}

/// REQ-20 P2: 選択肢の削除は `required`。学習データ同一は `false`、評価データが両側に無ければ `null`。
#[test]
fn req20_removed_option_is_required() {
    let e = Env::new("rm");
    e.register_old("v1_9.json", TRAIN_A, None);
    e.source("new", "v2_8rm.json", TRAIN_A, None);
    let o = e.register_new("proj", "oldproj");
    assert_eq!(o.status.code(), Some(0));
    let rebuild = format!(
        "{{\"decision\":\"required\",\"previous_definition_sha256\":\"{}\",\"reasons\":[{{\"kind\":\"option_ids_changed\",\"added\":[],\"removed\":[\"tier-xl__high\"]}}],\"display_name_changed\":[],\"description_changed\":[],\"training_data_changed\":false,\"evaluation_data_changed\":null}}",
        hash("v1_9.json")
    );
    assert_eq!(stdout(&o), expected("v2_8rm.json", 8, false, &rebuild));
    assert!(e.exists("proj/definition.json"));
}

/// REQ-20 P3: 統合は `added`・`removed` が両方非空の `required`（ID は辞書順）。データ差は true / false。
#[test]
fn req20_merged_options_are_required_with_added_and_removed() {
    let e = Env::new("merge");
    e.register_old("v1_9.json", TRAIN_A, Some(EVAL_A));
    e.source("new", "v3_8merge.json", TRAIN_B, Some(EVAL_A));
    let o = e.register_new("proj", "oldproj");
    assert_eq!(o.status.code(), Some(0));
    let rebuild = format!(
        "{{\"decision\":\"required\",\"previous_definition_sha256\":\"{}\",\"reasons\":[{{\"kind\":\"option_ids_changed\",\"added\":[\"tier-l__midhigh\"],\"removed\":[\"tier-l__high\",\"tier-l__medium\"]}}],\"display_name_changed\":[],\"description_changed\":[],\"training_data_changed\":true,\"evaluation_data_changed\":false}}",
        hash("v1_9.json")
    );
    assert_eq!(stdout(&o), expected("v3_8merge.json", 8, true, &rebuild));
}

/// REQ-20 P4: 表示名・説明だけの差は `not_required`。評価データが新側に無ければ `null`。
#[test]
fn req20_rename_is_not_required() {
    let e = Env::new("rename");
    e.register_old("v1_9.json", TRAIN_A, Some(EVAL_A));
    e.source("new", "v4_rename.json", TRAIN_A, None);
    let o = e.register_new("proj", "oldproj");
    assert_eq!(o.status.code(), Some(0));
    let rebuild = format!(
        "{{\"decision\":\"not_required\",\"previous_definition_sha256\":\"{}\",\"reasons\":[],\"display_name_changed\":[\"tier-xs__low\"],\"description_changed\":[\"tier-xl__high\"],\"training_data_changed\":false,\"evaluation_data_changed\":null}}",
        hash("v1_9.json")
    );
    assert_eq!(stdout(&o), expected("v4_rename.json", 9, false, &rebuild));
}

/// REQ-20 境界値: 同一の定義は `unchanged`（exit 0 でプロジェクトを作る）。評価データの差は `true`。
#[test]
fn req20_same_definition_is_unchanged() {
    let e = Env::new("same");
    e.register_old("v1_9.json", TRAIN_A, Some(EVAL_A));
    e.source("new", "v1_9.json", TRAIN_A, Some(EVAL_B));
    let o = e.register_new("proj", "oldproj");
    assert_eq!(o.status.code(), Some(0));
    let rebuild = format!(
        "{{\"decision\":\"unchanged\",\"previous_definition_sha256\":\"{}\",\"reasons\":[],\"display_name_changed\":[],\"description_changed\":[],\"training_data_changed\":false,\"evaluation_data_changed\":true}}",
        hash("v1_9.json")
    );
    assert_eq!(stdout(&o), expected("v1_9.json", 9, true, &rebuild));
    assert!(e.exists("proj/definition.json"));
}

/// REQ-20 P5: 判定型 `multi_select` の新定義は読み込みで 64（判定しない）。
#[test]
fn req20_multi_select_new_definition_is_invalid() {
    let e = Env::new("multi-new");
    e.register_old("v1_9.json", TRAIN_A, None);
    e.source("new", "v5_multi.json", TRAIN_A, None);
    let o = e.register_new("proj", "oldproj");
    assert_rejected(&e, &o, "\"code\":\"invalid_input\"");
}

/// REQ-20・REQ-21: 旧プロジェクトの定義が読めない（`multi_select`・欠落）は 64。
#[test]
fn req20_unreadable_previous_definition_is_invalid() {
    let e = Env::new("old-bad");
    e.source("new", "v1_9.json", TRAIN_A, None);
    std::fs::create_dir_all(e.0.join("oldproj")).expect("mkdir");
    let o = e.register_new("proj", "oldproj");
    assert_rejected(&e, &o, "\"code\":\"invalid_input\"");
    std::fs::copy(
        Path::new(FIXTURES).join("v5_multi.json"),
        e.0.join("oldproj/definition.json"),
    )
    .expect("copy");
    let o = e.register_new("proj", "oldproj");
    assert_rejected(&e, &o, "\"code\":\"invalid_input\"");
}

/// REQ-39: 旧プロジェクトが存在しない・cwd 外は 64。
#[test]
fn req39_missing_or_outside_previous_project_is_invalid() {
    let e = Env::new("old-path");
    e.source("new", "v1_9.json", TRAIN_A, None);
    let o = e.register_new("proj", "nope");
    assert_rejected(&e, &o, "path rejected");
    let o = e.register_new("proj", "..");
    assert_rejected(&e, &o, "path rejected");
}

/// REQ-39: `--project-dir` が旧プロジェクトと同じ・その配下は 64 で、旧プロジェクトへ書かない。
#[test]
fn req39_project_dir_same_as_or_inside_previous_is_invalid() {
    let e = Env::new("nested");
    e.register_old("v1_9.json", TRAIN_A, None);
    e.source("new", "v2_8rm.json", TRAIN_A, None);
    let message = "project directory must not be the previous project directory or inside it";
    for target in ["oldproj", "./oldproj", "oldproj/sub"] {
        let o = e.register_new(target, "oldproj");
        assert_rejected(&e, &o, message);
    }
    assert!(!e.exists("oldproj/sub"));
}

/// REQ-33: フラグが無ければ `rebuild` は `null`。
#[test]
fn req33_without_flag_rebuild_is_null() {
    let e = Env::new("noflag");
    e.source("new", "v1_9.json", TRAIN_A, None);
    let o = e.run(&[
        "register",
        "--definition",
        "new/def.json",
        "--project-dir",
        "proj",
    ]);
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(stdout(&o), expected("v1_9.json", 9, false, "null"));
}
