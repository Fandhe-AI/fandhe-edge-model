//! 最終 test の台帳ディレクトリを、プロジェクトの保持 fd 起点で開いて使う（REQ-27・REQ-39・#314・#168）。
//! `evaluate`（事前登録と適用）と `package`（適用完了の照会）が共有する。
//!
//! 台帳（`fandhe-edge-eval` の `FinalTestLedger`）のファイル操作は `LedgerDir` 越しで、本工程は
//! [`Project::open_subdir`]（保持 fd 起点・`O_NOFOLLOW`）で開いた台帳ディレクトリを
//! [`crate::held_ledger_dir::HeldLedgerDir`] に包んで渡す。台帳の読み・作成・一覧・読み取り専用化・
//! 永続化はすべて保持 fd 起点の `openat` で、パスへ戻る経路を持たない。検証後に台帳ディレクトリ
//! やメンバーが symlink へ差し替えられても、プロジェクトの外を読み書きしない（差し替えは
//! ガード層が fail-closed で拒否する）。台帳のファイル自体を書き換えられる主体による偽造の
//! 検出は外部台帳（#168・TASK-39.3-2）の範囲で、本モジュールは検証済みとしない。
//!
//! Linux・macOS 限定（fd 操作の target 依存。それ以外は `runtime_error` で拒否する）。

use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_eval::final_test_once::FinalTestLedger;

use crate::project::{FINAL_TEST_LEDGER_DIR, Project};

/// 保持 fd つきで開いた台帳。
pub(super) struct HeldLedger {
    ledger: FinalTestLedger,
}

impl HeldLedger {
    /// 台帳ディレクトリを開く。無いとき、`create` が真なら 0700 で作り、偽なら `None`。
    ///
    /// # Errors
    /// 経路の拒否・台帳ディレクトリが不正は `invalid_input`、I/O 失敗・非対応 OS は `runtime_error`。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub(super) fn open(project: &Project, create: bool) -> Result<Option<Self>, ErrorReport> {
        if !project.exists(FINAL_TEST_LEDGER_DIR)? {
            if !create {
                return Ok(None);
            }
            // 最初の `evaluate` が同時に走ると後発の作成は「既存」で失敗する。作成に失敗しても
            // 既にディレクトリがあれば（先行プロセスが作った）開き直す。
            if let Err(e) = project.create_dir(FINAL_TEST_LEDGER_DIR)
                && !project.exists(FINAL_TEST_LEDGER_DIR)?
            {
                return Err(e);
            }
        }
        let held = project.open_subdir(FINAL_TEST_LEDGER_DIR)?;
        let ledger = FinalTestLedger::with_dir(crate::held_ledger_dir::HeldLedgerDir::new(held));
        Ok(Some(Self { ledger }))
    }

    /// 非対応 OS では保持 fd を使えないため、台帳を使わない（fail-closed）。
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub(super) fn open(project: &Project, create: bool) -> Result<Option<Self>, ErrorReport> {
        let _ = (project, create, FINAL_TEST_LEDGER_DIR);
        Err(crate::project::runtime("unsupported platform"))
    }

    /// 台帳を返す。
    pub(super) fn ledger(&self) -> &FinalTestLedger {
        &self.ledger
    }
}

/// 台帳・評価記録の差し替え拒否の回帰テスト（REQ-39・REQ-27・#314・#168）。
///
/// 検証後にディレクトリやメンバーを外部への symlink へ差し替えても、保持 fd 起点の操作が
/// プロジェクトの外を読み書きしないことを確かめる。
#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use fandhe_edge_core::exitcode::ExitCode;
    use fandhe_edge_core::hash::Sha256Digest;
    use fandhe_edge_eval::final_test_once::{RegisteredConfig, RepresentativeConfigId};

    use super::*;
    use crate::project::{CANDIDATES_DIR, EVALUATION_RECORD_FILE};

    /// テスト用の作業ディレクトリ（cwd 役）。Drop で片付ける。
    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "fandhe-edge-ledger-test-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create scratch");
            Self(std::fs::canonicalize(&dir).expect("canonicalize scratch"))
        }

        fn path(&self) -> &Path {
            &self.0
        }

        /// 外部（プロジェクト外）の空ディレクトリを作る。
        fn outside(&self) -> PathBuf {
            let outside = self.0.join("outside");
            std::fs::create_dir(&outside).expect("outside dir");
            outside
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn entry() -> RegisteredConfig {
        RegisteredConfig::new(
            RepresentativeConfigId::parse("c1:seed0").expect("id"),
            Sha256Digest::of_bytes(b"w"),
        )
    }

    fn entry_count(dir: &Path) -> usize {
        std::fs::read_dir(dir).expect("read dir").count()
    }

    /// REQ-39: 台帳ディレクトリを開いた後に外部への symlink へ差し替えても、台帳の操作は
    /// 保持 fd 起点で失敗し（差し替え先を辿らない）、外部ディレクトリには何も作られない。
    #[test]
    fn req39_swapped_ledger_dir_symlink_is_not_followed() {
        let scratch = Scratch::new();
        let outside = scratch.outside();
        let project = Project::create(scratch.path(), Path::new("proj")).expect("create project");
        let held = HeldLedger::open(&project, true)
            .expect("open ledger")
            .expect("ledger is created");

        let ledger_path = scratch.path().join("proj").join(FINAL_TEST_LEDGER_DIR);
        std::fs::rename(&ledger_path, scratch.path().join("moved")).expect("move away");
        std::os::unix::fs::symlink(&outside, &ledger_path).expect("symlink swap");

        let sha = Sha256Digest::of_bytes(b"eval");
        let result = held.ledger().register_configs(&sha, &[entry()]);
        assert!(
            result.is_err(),
            "operation on a swapped ledger must fail closed"
        );
        assert_eq!(entry_count(&outside), 0);
    }

    /// REQ-39: 台帳ディレクトリが最初から外部への symlink なら、開く段階で拒否する。
    #[test]
    fn req39_ledger_dir_that_is_a_symlink_is_rejected_on_open() {
        let scratch = Scratch::new();
        let outside = scratch.outside();
        let project = Project::create(scratch.path(), Path::new("proj")).expect("create project");
        std::os::unix::fs::symlink(
            &outside,
            scratch.path().join("proj").join(FINAL_TEST_LEDGER_DIR),
        )
        .expect("symlink");

        let Err(err) = HeldLedger::open(&project, true) else {
            panic!("symlinked ledger dir must be rejected");
        };
        assert_ne!(err.code, ExitCode::Ok);
        assert_eq!(entry_count(&outside), 0);
    }

    /// REQ-39・REQ-27: 事前登録後に、評価データごとのサブディレクトリを外部への symlink へ
    /// 差し替えても、台帳は外を読まず・書かない（`is_applied` は誤って完了扱いにならない）。
    #[test]
    fn req39_swapped_scope_dir_symlink_is_not_followed() {
        let scratch = Scratch::new();
        let outside = scratch.outside();
        let project = Project::create(scratch.path(), Path::new("proj")).expect("create project");
        let held = HeldLedger::open(&project, true)
            .expect("open ledger")
            .expect("ledger is created");
        let sha = Sha256Digest::of_bytes(b"eval");
        held.ledger()
            .register_configs(&sha, &[entry()])
            .expect("register");

        let ledger_path = scratch.path().join("proj").join(FINAL_TEST_LEDGER_DIR);
        let scope = std::fs::read_dir(&ledger_path)
            .expect("list ledger")
            .next()
            .expect("scope dir")
            .expect("entry")
            .path();
        std::fs::rename(&scope, scratch.path().join("moved-scope")).expect("move away");
        std::os::unix::fs::symlink(&outside, &scope).expect("symlink swap");

        let id = RepresentativeConfigId::parse("c1:seed0").expect("id");
        let applied = held
            .ledger()
            .is_applied(&sha, &id, &Sha256Digest::of_bytes(b"w"));
        assert!(applied.is_err(), "swapped scope must not be trusted");
        assert!(
            held.ledger().register_configs(&sha, &[entry()]).is_err(),
            "registering through a swapped scope must fail"
        );
        assert_eq!(entry_count(&outside), 0);
    }

    /// REQ-39・REQ-27: 事前登録ファイルを外部の偽造ファイルへの symlink へ差し替えても読まない
    /// （`is_applied` は誤って `Ok` にならず、外部ファイルの内容を採用しない）。
    #[test]
    fn req39_swapped_registry_member_symlink_is_not_read() {
        let scratch = Scratch::new();
        let outside = scratch.outside();
        let project = Project::create(scratch.path(), Path::new("proj")).expect("create project");
        let held = HeldLedger::open(&project, true)
            .expect("open ledger")
            .expect("ledger is created");
        let sha = Sha256Digest::of_bytes(b"eval");
        held.ledger()
            .register_configs(&sha, &[entry()])
            .expect("register");

        let ledger_path = scratch.path().join("proj").join(FINAL_TEST_LEDGER_DIR);
        let scope = std::fs::read_dir(&ledger_path)
            .expect("list ledger")
            .next()
            .expect("scope dir")
            .expect("entry")
            .path();
        for e in std::fs::read_dir(&scope).expect("list scope") {
            let path = e.expect("entry").path();
            let name = path.file_name().expect("name").to_owned();
            let body = std::fs::read(&path).expect("read member");
            // 同じ内容の複製を外部に置き、メンバーをそちらへの symlink にする。
            std::fs::write(outside.join(&name), body).expect("copy outside");
            let mut perms = std::fs::metadata(&path).expect("meta").permissions();
            #[allow(clippy::permissions_set_readonly_false)]
            perms.set_readonly(false);
            std::fs::set_permissions(&path, perms).expect("chmod");
            std::fs::remove_file(&path).expect("remove member");
            std::os::unix::fs::symlink(outside.join(&name), &path).expect("symlink swap");
        }
        let before = entry_count(&outside);

        let id = RepresentativeConfigId::parse("c1:seed0").expect("id");
        let applied = held
            .ledger()
            .is_applied(&sha, &id, &Sha256Digest::of_bytes(b"w"));
        assert!(applied.is_err(), "symlinked members must not be trusted");
        assert_eq!(
            entry_count(&outside),
            before,
            "nothing may be written outside"
        );
    }

    /// REQ-39・REQ-27: 評価完了記録の書き込み先（`candidates/<N>`）を外部への symlink へ差し替えても、
    /// fd 起点の作成が拒否し、外部ディレクトリへ `evaluation_record.json` を書かない。
    #[test]
    fn req39_swapped_candidate_dir_symlink_rejects_record_write() {
        let scratch = Scratch::new();
        let outside = scratch.outside();
        let project = Project::create(scratch.path(), Path::new("proj")).expect("create project");
        project.create_dir(CANDIDATES_DIR).expect("candidates dir");
        project
            .create_dir(format!("{CANDIDATES_DIR}/0"))
            .expect("candidate dir");

        let cand = scratch.path().join("proj").join(CANDIDATES_DIR).join("0");
        std::fs::remove_dir_all(&cand).expect("remove candidate");
        std::os::unix::fs::symlink(&outside, &cand).expect("symlink swap");

        let rel = format!("{CANDIDATES_DIR}/0/{EVALUATION_RECORD_FILE}");
        let err = project
            .write_new(&rel, b"{}")
            .expect_err("write through symlink must be rejected");
        assert_ne!(err.code, ExitCode::Ok);
        assert_eq!(entry_count(&outside), 0);
    }
}
