//! `--project-dir`（プロジェクトディレクトリ）の規約と、閉じ込めつきの読み書き
//! （REQ-17・REQ-33・REQ-39・TASK-33.1-2・#136）。
//!
//! # 位置づけ
//!
//! 7 工程（`register → inspect → train → evaluate → select → package → infer`）は、
//! 工程間の受け渡しを `--project-dir` 配下のファイルで行う（PoC-16 の「project_dir を自己完結に
//! する」方針。工程ごとの結果 JSON は stdout の 1 つだけ）。レイアウトの名前・既定値は本モジュールの
//! 定数に集約し、各工程（`crate::stages`）は名前を直書きしない。
//!
//! ```text
//! <project-dir>/                 （register が新規作成。既存なら invalid_input）
//! ├── definition.json            登録した定義ファイルのコピー（バイト列そのまま）
//! ├── data/train.jsonl           取り込んだ学習データ
//! ├── data/evaluation.jsonl      任意。凍結・読み取り専用配置（REQ-17）
//! ├── eval_freeze.json           evaluation.jsonl がある場合のみ（凍結記録）
//! ├── split.json                 inspect が作る分割記録（seed・規則・各分割のハッシュ）
//! ├── candidates/<N>/            train が候補ごとに作る（request.json・train_input.jsonl・
//! │                              job/・result.json と、学習ワーカーの出力先）。evaluate が
//! │                              evaluation_record.json（評価完了の記録）を足す
//! ├── final_test_ledger/         最初の evaluate が作る最終 test の台帳（評価器が管理。0700）
//! ├── selection_record.json      select の記録
//! ├── package.staging/           package の組み立て・容量計測用（公開後・失敗時は残らない）
//! └── package/                   package が作る配布パッケージ（容量が上限内のときだけ公開）
//! ```
//!
//! # 入力元の規約（暫定・オーナー確認事項）
//!
//! 定義ファイルに `data`（データパス）の欄は無い（スキーマ変更は承認事項）ため、`register` は
//! **定義ファイルと同じディレクトリの固定名** `train.jsonl`（必須）・`evaluation.jsonl`（任意）を
//! 取り込む。
//!
//! # 閉じ込め（REQ-39）
//!
//! プロジェクト内の読み取り・書き込みはガード層の [`ConfinedPackage`]（ディレクトリ fd 起点の
//! `openat`＋`O_NOFOLLOW`。書き込みは `O_EXCL`・`mkdirat`）を通す（既存の上書き・symlink の追従・
//! 検証後の親の差し替えによる外部への書き込みをしない）。プロジェクトの作成は cwd 配下へ閉じ込めた
//! 親の下だけで、既存の `--project-dir` は拒否する。ディレクトリは所有者のみ（0700）で作る
//! （学習ワーカーが出力先の親の権限を検査するため）。
//!
//! エラーの message は固定の英語語彙で、パス・データ本文・利用者の値を含めない（`security.md`）。

use std::fs::File;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

use fandhe_edge_core::definition::{Definition, MAX_DEFINITION_FILE_BYTES};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::{FsError, read_bounded_open_file};
use fandhe_edge_data::inspect::{ValidRecord, inspect_records};
use fandhe_edge_guard::package::{ConfinedPackage, confine_package};
use fandhe_edge_guard::path::PathRejection;

use crate::error_report::ToErrorReport;

/// 登録済み定義のファイル名。
pub const DEFINITION_FILE: &str = "definition.json";
/// 取り込んだデータのディレクトリ名。
pub const DATA_DIR: &str = "data";
/// 学習データのファイル名（入力元・取り込み先で共通）。
pub const TRAIN_DATA_FILE: &str = "train.jsonl";
/// 独立した評価データのファイル名（入力元・取り込み先で共通）。
pub const EVALUATION_DATA_FILE: &str = "evaluation.jsonl";
/// 評価データの凍結記録のファイル名。
pub const FREEZE_FILE: &str = "eval_freeze.json";
/// 分割記録のファイル名。
pub const SPLIT_FILE: &str = "split.json";
/// 候補ディレクトリの親。
pub const CANDIDATES_DIR: &str = "candidates";
/// 最終 test の台帳（評価器の `FinalTestLedger`）のディレクトリ名。最初の `evaluate` が作り、
/// 学習済みの候補をまとめて事前登録する（REQ-27。#314）。
pub const FINAL_TEST_LEDGER_DIR: &str = "final_test_ledger";
/// 候補ディレクトリ内の評価完了記録のファイル名（`evaluate` が新規に書き、`package` が確認する。#314）。
pub const EVALUATION_RECORD_FILE: &str = "evaluation_record.json";
/// 評価データの 1 件ごとの予測（`evaluate` が評価記録と同じ候補ディレクトリへ保存する JSONL。REQ-27・#445）。
pub const EVALUATION_PREDICTIONS_FILE: &str = "evaluation_predictions.jsonl";
/// 選定記録のファイル名。
pub const SELECTION_FILE: &str = "selection_record.json";
/// 全候補が容量超過で除外された `select` の除外結果（内部記録。他工程は読まない。#125）。
pub const SELECTION_EXCLUSIONS_FILE: &str = "selection_exclusions.json";
/// 配布パッケージのディレクトリ名（`sandbox-run.sh` の出力先と同じ）。
pub const PACKAGE_DIR: &str = "package";
/// `package` 工程の組み立て・容量計測用のステージング。上限内のときだけ [`PACKAGE_DIR`] へ
/// 原子的に名前替えして公開する（推論可能な場所に超過したパッケージを残さない。REQ-30・REQ-39）。
pub const PACKAGE_STAGING_DIR: &str = "package.staging";
/// 候補ディレクトリ内の学習リクエスト（`select` が結果の再検証に使う）。
pub const REQUEST_FILE: &str = "request.json";
/// `train --train-seed` で学習 seed を上書きした候補だけに置く、実際に使った seed の記録
/// （10 進数の u32。省略時は作らず、下流は `split.json` の seed を使う。REQ-17・REQ-41）。
pub const TRAIN_SEED_FILE: &str = "train_seed.txt";
/// 候補ディレクトリ内の学習結果。
pub const RESULT_FILE: &str = "result.json";
/// 候補ディレクトリ内の trainer 形式の学習データ。
pub const TRAIN_INPUT_FILE: &str = "train_input.jsonl";
/// 候補ディレクトリ内の学習ジョブ作業ディレクトリ。
pub const JOB_DIR: &str = "job";
/// 候補ディレクトリ内の学習ワーカーの出力先（種類ごとに `-<kind>` が付く。REQ-19）。
pub const MODEL_DIR: &str = "model";

/// プロジェクトの seed の既定値（`inspect --seed` の省略時。暫定の固定値）。分割と学習で共通で、
/// `inspect` が `split.json` に記録し、`train`・`select`・`package` はその記録値を使う（REQ-17）。
pub const DEFAULT_SEED: u32 = 42;
/// 前処理の最大バイト長（暫定の固定値。`fixtures/train_contract/request_minimal.json` と同じ）。
pub const DEFAULT_MAX_BYTES: u32 = 512;

/// データ・記録ファイルの読み込み上限（バイト。データ検査の合計上限と同じ。REQ-39）。
pub const MAX_PROJECT_FILE_BYTES: u64 = fandhe_edge_core::limits::MAX_PROJECT_FILE_BYTES;

/// 工程内のエラーを [`ErrorReport`] にする（message は固定語彙）。
#[must_use]
pub fn fail(code: ExitCode, message: &str) -> ErrorReport {
    ErrorReport::new(code, message)
}

/// `invalid_input`（64）の [`ErrorReport`]。
#[must_use]
pub fn invalid(message: &str) -> ErrorReport {
    fail(ExitCode::InvalidInput, message)
}

/// `runtime_error`（70）の [`ErrorReport`]。
#[must_use]
pub fn runtime(message: &str) -> ErrorReport {
    fail(ExitCode::RuntimeError, message)
}

/// [`FsError`] を写す（サイズ超過は `limit_exceeded`、それ以外は `runtime_error`）。
#[must_use]
pub fn fs_report(error: &FsError) -> ErrorReport {
    match error {
        FsError::TooLarge { .. } => fail(ExitCode::LimitExceeded, "input file exceeds size limit"),
        _ => runtime("cannot read project file"),
    }
}

/// 開いたプロジェクトディレクトリ（cwd 配下へ閉じ込め済み）。
#[derive(Debug)]
pub struct Project {
    package: ConfinedPackage,
    /// [`Project::create`] が作った場合の、親（保持 fd）と末尾の名前。失敗時の後始末
    /// （[`Project::remove_created`]）が、作ったディレクトリと同一の実体だけを消すために使う。
    created_in: Option<(ConfinedPackage, std::ffi::OsString)>,
}

/// [`Project::create_dir_tracked`] が作ったディレクトリ（fd を保持し、後始末の同一性確認に使う）。
#[derive(Debug)]
pub struct CreatedDir {
    rel: PathBuf,
    handle: ConfinedPackage,
}

impl Project {
    /// 既存の `--project-dir` を cwd 配下へ閉じ込めて開く。
    ///
    /// # Errors
    /// 経路の拒否（存在しない・cwd 外・ディレクトリでない）は `invalid_input`（64）等。
    pub fn open(cwd: &Path, project_dir: &Path) -> Result<Self, ErrorReport> {
        let package = confine_package(cwd, project_dir).map_err(|e| e.to_error_report())?;
        Ok(Self {
            package,
            created_in: None,
        })
    }

    /// `register` 用に、未作成の `--project-dir` を cwd 配下の既存の親の下へ新規作成して開く。
    ///
    /// # Errors
    /// 親の経路の拒否、末尾が通常の名前でない、既に存在する場合は `invalid_input`（64）。
    pub fn create(cwd: &Path, project_dir: &Path) -> Result<Self, ErrorReport> {
        let Some(leaf) = project_dir.file_name() else {
            return Err(invalid("project directory path is invalid"));
        };
        let parent = project_dir
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        // 親も保持した fd 起点で開き、`mkdirat` で作る（検証後の親の差し替えで cwd 外へ作らない。REQ-39）。
        let parent = confine_package(cwd, parent).map_err(|e| e.to_error_report())?;
        parent.create_dir_member(Path::new(leaf)).map_err(|e| {
            write_rejection(&e, "directory already exists", "cannot create directory")
        })?;
        // 作った直後に fd を保持して開く（パスを開き直さない。後始末で同一性を確認するため）。
        let package = parent
            .open_subdir(Path::new(leaf))
            .map_err(|e| e.to_error_report())?;
        Ok(Self {
            package,
            created_in: Some((parent, leaf.to_os_string())),
        })
    }

    /// [`Project::create`] が作ったプロジェクトを片付ける（失敗時の後始末。best effort）。
    ///
    /// 中身は保持した fd 起点で消し（パスを再解決しない）、ディレクトリ自身は親の下の名前が作成時と
    /// 同一の実体のときだけ削除する。差し替えられていれば他所のディレクトリには触れない（REQ-39）。
    /// [`Project::create`] で作っていないプロジェクトには何もしない。
    pub fn remove_created(&self) {
        let Some((parent, leaf)) = &self.created_in else {
            return;
        };
        let _ = self.package.clear_contents();
        let _ = parent.remove_empty_dir_member_if_same(Path::new(leaf), &self.package);
    }

    /// 正準化済みのプロジェクトディレクトリ（学習ワーカーへ渡す `root` の組み立て用。
    /// ファイルを開き直すために使わない）。
    #[must_use]
    pub fn dir(&self) -> &Path {
        self.package.dir()
    }

    /// プロジェクト内の相対パスを絶対パスにする（存在確認・書き込み先の組み立て用）。
    #[must_use]
    pub fn path(&self, rel: impl AsRef<Path>) -> PathBuf {
        self.dir().join(rel)
    }

    /// 相対パスの対象が存在するか（symlink は追従しない）。
    ///
    /// `NotFound` だけを「無い」とし、権限拒否・I/O 失敗は `runtime_error` にする（失敗を
    /// 「無い」と読み替えて後続工程が既存の記録を無視しない。fail-closed）。
    ///
    /// # Errors
    /// `NotFound` 以外のメタデータ取得失敗は `runtime_error`（70）。
    pub fn exists(&self, rel: impl AsRef<Path>) -> Result<bool, ErrorReport> {
        match std::fs::symlink_metadata(self.path(rel)) {
            Ok(_) => Ok(true),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
            Err(_) => Err(runtime("cannot inspect project file")),
        }
    }

    /// 閉じ込めつきで開いて上限付きで読む（パスを開き直さない）。
    ///
    /// # Errors
    /// 経路の拒否は `invalid_input`、サイズ超過は `limit_exceeded`、I/O 失敗は `runtime_error`。
    pub fn read(&self, rel: impl AsRef<Path>, limit: u64) -> Result<Vec<u8>, ErrorReport> {
        let (file, real) = self
            .package
            .open_member(rel.as_ref())
            .map_err(|e| e.to_error_report())?;
        read_bounded_open_file(file, real.as_path(), limit).map_err(|e| fs_report(&e))
    }

    /// 閉じ込めつきで開く（容量計測など、ハンドルを保持したい用途）。
    ///
    /// # Errors
    /// [`Project::read`] と同じ経路の拒否。
    pub fn open_file(&self, rel: impl AsRef<Path>) -> Result<(File, PathBuf), ErrorReport> {
        let (file, real) = self
            .package
            .open_member(rel.as_ref())
            .map_err(|e| e.to_error_report())?;
        Ok((file, real.into_path_buf()))
    }

    /// 相対パスが存在するときだけ読む（存在しなければ `None`）。
    ///
    /// # Errors
    /// 存在するのに読めない場合は [`Project::read`] と同じ。
    pub fn read_optional(
        &self,
        rel: impl AsRef<Path>,
        limit: u64,
    ) -> Result<Option<Vec<u8>>, ErrorReport> {
        // 開いた結果で「無い」を判定する（`NotFound` のみ `None`。権限拒否・I/O 失敗は
        // 「無い」と読み替えず失敗として返す。評価データが「無い」扱いで evaluate が
        // skipped になるのを防ぐ。REQ-17）。
        let (file, real) = match self.package.open_member(rel.as_ref()) {
            Ok(opened) => opened,
            Err(PathRejection::Unresolvable { source, .. })
                if source.kind() == ErrorKind::NotFound =>
            {
                return Ok(None);
            }
            Err(e) => return Err(e.to_error_report()),
        };
        read_bounded_open_file(file, real.as_path(), limit)
            .map(Some)
            .map_err(|e| fs_report(&e))
    }

    /// 新規ファイルを書く（既存なら拒否。symlink は追従しない）。親ディレクトリは既存であること。
    ///
    /// 親は保持したディレクトリ fd 起点で成分ごとに `O_NOFOLLOW` で辿り、`O_EXCL` で作る
    /// （`Project::open` 後に親が symlink へ差し替えられても外へ書かない。REQ-39）。書き込みに
    /// 失敗したら作りかけのファイルを消す（再実行が「既存」で恒久的に拒否されないように）。
    ///
    /// # Errors
    /// 既存は `invalid_input`、書き込み失敗は `runtime_error`。
    pub fn write_new(&self, rel: impl AsRef<Path>, bytes: &[u8]) -> Result<(), ErrorReport> {
        let rel = rel.as_ref();
        let mut file = self
            .package
            .create_new_member(rel)
            .map_err(|e| write_rejection(&e, "file already exists", "cannot write project file"))?;
        if file.write_all(bytes).and_then(|()| file.flush()).is_err() {
            drop(file);
            // best effort（消せなくても元の失敗を返す）。
            let _ = self.package.remove_file_member(rel);
            return Err(runtime("cannot write project file"));
        }
        Ok(())
    }

    /// 既存ファイルを読み取り専用（0400）にする（開いた fd への `fchmod`。パスを開き直さない）。
    /// 評価の予測ファイルなど、書いた後に差し替えさせたくない記録に使う（非 unix では何もしない）。
    ///
    /// # Errors
    /// 開けない場合は経路の拒否、`fchmod` 失敗は `runtime_error`。
    pub fn set_read_only(&self, rel: impl AsRef<Path>) -> Result<(), ErrorReport> {
        let (file, _) = self.open_file(rel)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(std::fs::Permissions::from_mode(0o400))
                .map_err(|_| runtime("cannot protect project file"))?;
        }
        #[cfg(not(unix))]
        drop(file);
        Ok(())
    }

    /// `rel` の通常ファイルを保持 fd 起点で削除する。無ければ何もしない（削除に失敗したら
    /// `runtime_error`。古い記録を残さないための fail-closed）。
    ///
    /// # Errors
    /// 削除失敗（存在しない場合を除く）は `runtime_error`。
    pub fn remove_file_if_exists(&self, rel: impl AsRef<Path>) -> Result<(), ErrorReport> {
        match self.package.remove_file_member(rel.as_ref()) {
            Ok(()) => Ok(()),
            Err(PathRejection::Unresolvable { source, .. })
                if source.kind() == ErrorKind::NotFound =>
            {
                Ok(())
            }
            Err(_) => Err(runtime("cannot remove project file")),
        }
    }

    /// `rel` を新しい内容で置き換える（既存があってもよい）。同じディレクトリの一時名へ新規に書いて
    /// から、既存を消して保持 fd 起点の名前替えで置くため、`rel` が書きかけの状態で見えることはない
    /// （失敗時は一時ファイルを片付ける。名前替えは既存を置き換えない `NOREPLACE` のため、
    /// 消去と名前替えの間に落ちると `rel` は無い状態になる）。
    ///
    /// # Errors
    /// 書き込み・名前替えの失敗は `runtime_error`。
    pub fn replace_file(&self, rel: impl AsRef<Path>, bytes: &[u8]) -> Result<(), ErrorReport> {
        let rel = rel.as_ref();
        let mut tmp_name = rel.as_os_str().to_os_string();
        tmp_name.push(".tmp");
        let tmp = PathBuf::from(tmp_name);
        self.remove_file_if_exists(&tmp)?;
        self.write_new(&tmp, bytes)?;
        let published = self.remove_file_if_exists(rel).and_then(|()| {
            self.package
                .rename_member(&tmp, rel)
                .map_err(|_| runtime("cannot write project file"))
        });
        if published.is_err() {
            let _ = self.package.remove_file_member(&tmp);
        }
        published
    }

    /// [`Project::write_new`] のストリーミング版。`reader` から最大 `limit` バイトを固定長バッファで
    /// 複写し、全体をメモリへ載せない（語彙ファイルの配布物への複写。REQ-30・REQ-39）。
    /// `limit` を超えて続くデータがあれば拒否し、書きかけのファイルは消す。
    ///
    /// # Errors
    /// 既存は `invalid_input`、上限超過は `invalid_input`、書き込み・読み込み失敗は `runtime_error`。
    pub fn write_new_from_reader(
        &self,
        rel: impl AsRef<Path>,
        reader: &mut impl std::io::Read,
        limit: u64,
    ) -> Result<(), ErrorReport> {
        use std::io::Read as _;
        let rel = rel.as_ref();
        let mut file = self
            .package
            .create_new_member(rel)
            .map_err(|e| write_rejection(&e, "file already exists", "cannot write project file"))?;
        let copied = std::io::copy(&mut reader.take(limit.saturating_add(1)), &mut file)
            .and_then(|n| file.flush().map(|()| n));
        let result = match copied {
            Ok(n) if n <= limit => return Ok(()),
            Ok(_) => invalid("file exceeds the size limit"),
            Err(_) => runtime("cannot write project file"),
        };
        drop(file);
        let _ = self.package.remove_file_member(rel);
        Err(result)
    }

    /// ディレクトリを所有者のみ（0700）で新規作成する（既存なら拒否）。
    ///
    /// # Errors
    /// 既存は `invalid_input`、作成失敗は `runtime_error`。
    pub fn create_dir(&self, rel: impl AsRef<Path>) -> Result<PathBuf, ErrorReport> {
        let rel = rel.as_ref();
        self.package.create_dir_member(rel).map_err(|e| {
            write_rejection(&e, "directory already exists", "cannot create directory")
        })?;
        Ok(self.path(rel))
    }

    /// [`Project::create_dir`] と同じくディレクトリを新規作成し、fd を保持したハンドルも返す
    /// （失敗時に [`Project::remove_created_dir`] で同一の実体だけを片付けるため）。
    ///
    /// # Errors
    /// [`Project::create_dir`] と同じ。
    pub fn create_dir_tracked(&self, rel: impl AsRef<Path>) -> Result<CreatedDir, ErrorReport> {
        let rel = rel.as_ref();
        self.create_dir(rel)?;
        let handle = self
            .package
            .open_subdir(rel)
            .map_err(|e| e.to_error_report())?;
        Ok(CreatedDir {
            rel: rel.to_path_buf(),
            handle,
        })
    }

    /// プロジェクト内のディレクトリ `rel` を、保持 fd 起点の相対オープン（`O_NOFOLLOW`・
    /// `O_DIRECTORY`）で開き、新しい閉じ込めルートとして返す（パスの正規化・再解決をしない。
    /// 検証後の差し替えでも別のディレクトリを開かない。REQ-39）。
    ///
    /// # Errors
    /// 存在しない・ディレクトリでない・symlink・プロジェクトの外を指す場合は `invalid_input` 等。
    pub fn open_subdir(&self, rel: impl AsRef<Path>) -> Result<ConfinedPackage, ErrorReport> {
        self.package
            .open_subdir(rel.as_ref())
            .map_err(|e| e.to_error_report())
    }

    /// [`Project::create_dir_tracked`] が作ったディレクトリを片付ける（best effort）。
    /// 中身は保持 fd 起点で消し（symlink は追従せず、プロジェクトの外へは出ない）、名前が作成時と
    /// 同一の実体のときだけディレクトリ自身を消す。差し替えられていれば他所には触れない。
    ///
    /// 戻り値は、ディレクトリを完全に片付けられたか（`false` は残骸が残りうる）。
    #[must_use]
    pub fn remove_created_dir(&self, created: &CreatedDir) -> bool {
        let cleared = created.handle.clear_contents().is_ok();
        let removed = matches!(
            self.package
                .remove_empty_dir_member_if_same(&created.rel, &created.handle),
            Ok(true)
        );
        cleared && removed
    }

    /// プロジェクト内のディレクトリ `from` を、存在しない `to` へ原子的に名前替えする
    /// （ステージングの公開用。保持 fd 起点。Linux は `RENAME_NOREPLACE`）。
    ///
    /// # Errors
    /// `to` が既存は `invalid_input`、それ以外の失敗は `runtime_error`。
    pub fn publish_dir(
        &self,
        from: impl AsRef<Path>,
        to: impl AsRef<Path>,
    ) -> Result<(), ErrorReport> {
        self.package
            .rename_member(from.as_ref(), to.as_ref())
            .map_err(|e| {
                write_rejection(&e, "directory already exists", "cannot publish directory")
            })
    }

    /// 登録済みの定義を読む。
    ///
    /// # Errors
    /// 読み込み・解析の失敗（`register` 済みでなければ `invalid_input`）。
    pub fn load_definition(&self) -> Result<Definition, ErrorReport> {
        let bytes = self.read(DEFINITION_FILE, MAX_DEFINITION_FILE_BYTES)?;
        parse_definition(&bytes)
    }

    /// 取り込んだ学習データを検査して妥当なレコードを返す。異常が 1 件でもあれば
    /// `invalid_input`（固定 message。行番号・本文は出さない）。
    ///
    /// # Errors
    /// 読み込み失敗・異常レコードあり・ラベル集合が空。
    pub fn load_records(&self, definition: &Definition) -> Result<Vec<ValidRecord>, ErrorReport> {
        let bytes = self.read(
            Path::new(DATA_DIR).join(TRAIN_DATA_FILE),
            MAX_PROJECT_FILE_BYTES,
        )?;
        inspect_bytes(&bytes, definition)
    }
}

/// 定義ファイルのバイト列を UTF-8 として解析する。
///
/// # Errors
/// UTF-8 でない・定義として不正な場合は定義層の写像（`invalid_input`）。
pub fn parse_definition(bytes: &[u8]) -> Result<Definition, ErrorReport> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| invalid("definition file is not valid UTF-8"))?;
    Definition::parse(text).map_err(|e| e.to_error_report())
}

/// JSONL のバイト列を検査し、異常がなければ妥当なレコードを返す（`inspect_records` の薄い写像）。
///
/// # Errors
/// UTF-8 でない・異常レコードあり・ラベル集合が空は `invalid_input`。
pub fn inspect_bytes(
    bytes: &[u8],
    definition: &Definition,
) -> Result<Vec<ValidRecord>, ErrorReport> {
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("data file is not valid UTF-8"))?;
    let labels: std::collections::BTreeSet<String> =
        definition.options().iter().map(|c| c.id.clone()).collect();
    let outcome =
        inspect_records(text, &labels).map_err(|_| invalid("definition has no options"))?;
    if !outcome.anomalies.is_empty() {
        return Err(invalid("data file has invalid records"));
    }
    Ok(outcome.valid_records)
}

/// 書き込み系の閉じ込め拒否を [`ErrorReport`] にする（message は固定語彙）。
///
/// 既存は `invalid_input`、経路の拒否・未対応 OS は下位層の写像、それ以外（権限・容量等）は
/// `runtime_error`。
fn write_rejection(e: &PathRejection, exists_message: &str, other_message: &str) -> ErrorReport {
    match e {
        PathRejection::Unresolvable { source, .. } if source.kind() == ErrorKind::AlreadyExists => {
            invalid(exists_message)
        }
        PathRejection::Escapes { .. }
        | PathRejection::EmptyPath
        | PathRejection::UnsupportedPlatform => e.to_error_report(),
        _ => runtime(other_message),
    }
}
