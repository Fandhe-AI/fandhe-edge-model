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
//! │                              job/・result.json と、学習ワーカーの出力先）
//! ├── selection_record.json      select の記録
//! └── package/                   package が作る配布パッケージ
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
//! プロジェクト内の読み取りはガード層の [`ConfinedPackage`]（ディレクトリ fd 起点の `openat`＋
//! `O_NOFOLLOW`）を通す。書き込みは正準化済みのプロジェクトパス配下へ `create_new`（既存ファイルの
//! 上書き・symlink の追従をしない）で行う。プロジェクトの作成は cwd 配下へ閉じ込めた親の下だけで、
//! 既存の `--project-dir` は拒否する。ディレクトリは所有者のみ（0700）で作る
//! （学習ワーカーが出力先の親の権限を検査するため）。
//!
//! エラーの message は固定の英語語彙で、パス・データ本文・利用者の値を含めない（`security.md`）。

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use fandhe_edge_core::definition::{Definition, MAX_DEFINITION_FILE_BYTES};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::{FsError, read_bounded_open_file};
use fandhe_edge_data::inspect::{ValidRecord, inspect_records};
use fandhe_edge_guard::package::{ConfinedPackage, confine_package};
use fandhe_edge_guard::path::safe_join;

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
/// 選定記録のファイル名。
pub const SELECTION_FILE: &str = "selection_record.json";
/// 配布パッケージのディレクトリ名（`sandbox-run.sh` の出力先と同じ）。
pub const PACKAGE_DIR: &str = "package";
/// 候補ディレクトリ内の学習リクエスト（`select` が結果の再検証に使う）。
pub const REQUEST_FILE: &str = "request.json";
/// 候補ディレクトリ内の学習結果。
pub const RESULT_FILE: &str = "result.json";
/// 候補ディレクトリ内の trainer 形式の学習データ。
pub const TRAIN_INPUT_FILE: &str = "train_input.jsonl";
/// 候補ディレクトリ内の学習ジョブ作業ディレクトリ。
pub const JOB_DIR: &str = "job";
/// 候補ディレクトリ内の学習ワーカーの出力先（種類ごとに `-<kind>` が付く。REQ-19）。
pub const MODEL_DIR: &str = "model";

/// 分割の seed（暫定の固定値。定義ファイルに欄が無いため。オーナー確認事項）。
pub const SPLIT_SEED: u64 = 42;
/// 学習の seed（暫定の固定値）。
pub const TRAIN_SEED: u32 = 42;
/// 前処理の最大バイト長（暫定の固定値。`fixtures/train_contract/request_minimal.json` と同じ）。
pub const DEFAULT_MAX_BYTES: u32 = 512;

/// データ・記録ファイルの読み込み上限（バイト。データ検査の合計上限と同じ。REQ-39）。
pub const MAX_PROJECT_FILE_BYTES: u64 = 64 * 1024 * 1024;

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
}

impl Project {
    /// 既存の `--project-dir` を cwd 配下へ閉じ込めて開く。
    ///
    /// # Errors
    /// 経路の拒否（存在しない・cwd 外・ディレクトリでない）は `invalid_input`（64）等。
    pub fn open(cwd: &Path, project_dir: &Path) -> Result<Self, ErrorReport> {
        let package = confine_package(cwd, project_dir).map_err(|e| e.to_error_report())?;
        Ok(Self { package })
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
        let parent = safe_join(cwd, parent).map_err(|e| e.to_error_report())?;
        let target = parent.as_path().join(leaf);
        make_dir(&target)?;
        Self::open(cwd, &target)
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
    #[must_use]
    pub fn exists(&self, rel: impl AsRef<Path>) -> bool {
        std::fs::symlink_metadata(self.path(rel)).is_ok()
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
        if self.exists(rel.as_ref()) {
            self.read(rel, limit).map(Some)
        } else {
            Ok(None)
        }
    }

    /// 新規ファイルを書く（既存なら拒否。symlink は追従しない）。親ディレクトリは既存であること。
    ///
    /// # Errors
    /// 既存は `invalid_input`、書き込み失敗は `runtime_error`。
    pub fn write_new(&self, rel: impl AsRef<Path>, bytes: &[u8]) -> Result<(), ErrorReport> {
        write_new_file(&self.path(rel), bytes)
    }

    /// ディレクトリを所有者のみ（0700）で新規作成する（既存なら拒否）。
    ///
    /// # Errors
    /// 既存は `invalid_input`、作成失敗は `runtime_error`。
    pub fn create_dir(&self, rel: impl AsRef<Path>) -> Result<PathBuf, ErrorReport> {
        let path = self.path(rel);
        make_dir(&path)?;
        Ok(path)
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

#[cfg(unix)]
fn make_dir(path: &Path) -> Result<(), ErrorReport> {
    use std::os::unix::fs::DirBuilderExt as _;
    match std::fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            Err(invalid("directory already exists"))
        }
        Err(_) => Err(runtime("cannot create directory")),
    }
}

#[cfg(not(unix))]
fn make_dir(path: &Path) -> Result<(), ErrorReport> {
    match std::fs::create_dir(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            Err(invalid("directory already exists"))
        }
        Err(_) => Err(runtime("cannot create directory")),
    }
}

/// 新規ファイルを `create_new`（`O_EXCL`。既存・symlink は拒否）で書く。
///
/// # Errors
/// 既存は `invalid_input`、書き込み失敗は `runtime_error`。
pub fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), ErrorReport> {
    let mut file = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(invalid("file already exists"));
        }
        Err(_) => return Err(runtime("cannot write project file")),
    };
    file.write_all(bytes)
        .and_then(|()| file.flush())
        .map_err(|_| runtime("cannot write project file"))
}
