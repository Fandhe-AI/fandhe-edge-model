//! 最終 test の台帳ディレクトリを、プロジェクトの保持 fd 起点で開いて同一性を確認しながら使う
//! （REQ-27・REQ-39・#314）。`evaluate`（事前登録と適用）と `package`（適用完了の照会）が共有する。
//!
//! 台帳（`fandhe-edge-eval` の `FinalTestLedger`）はパスで操作する API のため、プロジェクトの
//! パスを連結して開くと、確認後の差し替え（ディレクトリを symlink へ置換）で閉じ込めが崩れる。
//! そこで (1) 台帳ディレクトリを [`Project::open_subdir`]（保持 fd 起点・`O_NOFOLLOW`）で開き、
//! (2) 台帳へ渡すパスは保持 fd から得た実パスだけにし、(3) 台帳を使う直前ごとに保持 fd と
//! そのパスの実体（dev・ino）が一致することを確認する。
//!
//! **限界（実装済みを装わない）**: 評価器の台帳 API がパスを受け取るため、(3) の確認と台帳内の
//! ファイル操作の間の差し替えは原理的に残る。fd 相対の台帳 API と外部台帳は #168・TASK-39.3-2
//! の範囲で、本モジュールは競合窓を狭めて差し替えを検出するに留まる。

use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_eval::final_test_once::FinalTestLedger;
use fandhe_edge_guard::package::ConfinedPackage;

use crate::error_report::acquire_error_report;
use crate::project::{FINAL_TEST_LEDGER_DIR, Project, invalid, runtime};

/// 保持 fd つきで開いた台帳。
pub(super) struct HeldLedger {
    held: ConfinedPackage,
    ledger: FinalTestLedger,
}

impl HeldLedger {
    /// 台帳ディレクトリを開く。無いとき、`create` が真なら 0700 で作り、偽なら `None`。
    ///
    /// # Errors
    /// 経路の拒否・台帳ディレクトリが不正・同一性の不一致は `invalid_input`、I/O 失敗は `runtime_error`。
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
        let ledger = FinalTestLedger::open(held.dir()).map_err(|e| acquire_error_report(&e))?;
        let this = Self { held, ledger };
        this.verify_same_dir()?;
        Ok(Some(this))
    }

    /// 台帳を使う直前に同一性を確認して返す。
    ///
    /// # Errors
    /// 保持 fd の実体とパスの実体が食い違う（差し替え）場合は `invalid_input`。
    pub(super) fn ledger(&self) -> Result<&FinalTestLedger, ErrorReport> {
        self.verify_same_dir()?;
        Ok(&self.ledger)
    }

    #[cfg(unix)]
    fn verify_same_dir(&self) -> Result<(), ErrorReport> {
        use std::os::unix::fs::MetadataExt as _;
        let held = self
            .held
            .metadata()
            .map_err(|_| runtime("cannot inspect ledger directory"))?;
        let by_path = std::fs::symlink_metadata(self.held.dir())
            .map_err(|_| invalid("final test ledger is invalid"))?;
        if !by_path.is_dir() || held.dev() != by_path.dev() || held.ino() != by_path.ino() {
            return Err(invalid("final test ledger is invalid"));
        }
        Ok(())
    }

    /// 非 Unix では保持 fd を使えないため、台帳を使わない（fail-closed）。
    #[cfg(not(unix))]
    fn verify_same_dir(&self) -> Result<(), ErrorReport> {
        let _ = (&self.held, &self.ledger, invalid);
        Err(runtime("unsupported platform"))
    }
}
