//! `register` 工程: 定義ファイルと学習・評価データをプロジェクトディレクトリへ取り込む
//! （REQ-15・REQ-16・REQ-17・REQ-33・TASK-33.1-2・#136）。
//!
//! # 入出力
//!
//! - 入力: `--definition`（定義ファイル。cwd 配下）と、その**同じディレクトリ**の
//!   `train.jsonl`（必須）・`evaluation.jsonl`（任意）。定義ファイルにデータパスの欄は無い
//!   ため固定名で取り込む（暫定・オーナー確認事項。[`crate::project`]）
//! - 出力: `--project-dir`（未作成であること。cwd 配下の既存の親の下へ新規作成）
//!
//! # 評価契約
//!
//! 評価データは取り込み時に凍結記録（sha256・バイト長。[`freeze_eval_data`]）を作り、
//! プロジェクトへ読み取り専用配置する（data 層の [`place_read_only_bytes`]。書き込みプローブで拒否を
//! 確認できない環境〔root・ACL 等〕では配置しない。凍結確認の単一の出所。REQ-17・REQ-39）。以後の
//! `inspect`・`evaluate` は記録とのハッシュ一致を確認し、不一致なら停止する（fail-closed）。
//!
//! 取り込みの途中で失敗した場合、本工程が作った `--project-dir` は削除する（半端な状態の
//! プロジェクトを残さない。既存の `--project-dir` は最初に拒否するため、削除対象は
//! 本工程が作ったものに限る）。

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use fandhe_edge_core::definition::MAX_DEFINITION_FILE_BYTES;
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_core::fs::read_bounded_open_file;
use fandhe_edge_core::stage_report::RegisterReport;
use fandhe_edge_data::eval_freeze::{FreezeRecord, freeze_eval_data};
use fandhe_edge_guard::path::{PathRejection, open_confined, safe_join};

use crate::args::RegisterArgs;
use crate::error_report::ToErrorReport;
use crate::project::{
    DATA_DIR, DEFINITION_FILE, EVALUATION_DATA_FILE, FREEZE_FILE, MAX_PROJECT_FILE_BYTES, Project,
    TRAIN_DATA_FILE, fs_report, invalid, parse_definition, runtime,
};

/// 「対象が存在しない」を表す拒否か（`NotFound` のみ。権限拒否・I/O 失敗は含めない）。
fn is_not_found(e: &PathRejection) -> bool {
    matches!(e, PathRejection::Unresolvable { source, .. } if source.kind() == ErrorKind::NotFound)
}

/// cwd 配下へ閉じ込めて開き、上限付きで読む（実体パスも返す）。存在しなければ `None`。
///
/// `None` は `NotFound` のときだけ。権限拒否・I/O 失敗・閉じ込め違反は失敗として返す
/// （失敗を「無い」と読み替えない。評価データの取り込み漏れを防ぐ。REQ-17）。
fn read_confined_optional(
    cwd: &Path,
    path: &Path,
    limit: u64,
) -> Result<Option<(Vec<u8>, PathBuf)>, ErrorReport> {
    match open_confined(cwd, path) {
        Ok((file, real)) => {
            let bytes =
                read_bounded_open_file(file, real.as_path(), limit).map_err(|e| fs_report(&e))?;
            Ok(Some((bytes, real.into_path_buf())))
        }
        Err(e) if is_not_found(&e) => Ok(None),
        // 閉じ込めつきの open が使えない OS では、存在しない・cwd 外のパスの拒否だけを
        // 他 OS と同じ結果にそろえ、それ以外は未対応として拒否する（fail-closed）。
        Err(PathRejection::UnsupportedPlatform) => {
            let root = std::fs::canonicalize(cwd).map_err(|_| runtime("cannot resolve cwd"))?;
            match safe_join(&root, path) {
                Err(e) if is_not_found(&e) => Ok(None),
                Err(e) => Err(e.to_error_report()),
                Ok(_) => Err(PathRejection::UnsupportedPlatform.to_error_report()),
            }
        }
        Err(e) => Err(e.to_error_report()),
    }
}

/// [`read_confined_optional`] で、存在しないことも `invalid_input`（経路の拒否）として返す。
fn read_confined(cwd: &Path, path: &Path, limit: u64) -> Result<(Vec<u8>, PathBuf), ErrorReport> {
    read_confined_optional(cwd, path, limit)?.ok_or_else(|| {
        PathRejection::Unresolvable {
            candidate: path.to_path_buf(),
            source: std::io::Error::from(ErrorKind::NotFound),
        }
        .to_error_report()
    })
}

/// `register` を実行する。
///
/// # Errors
/// 経路の拒否・定義の不正・データ欠落・既存の `--project-dir` は `invalid_input`（64）、
/// サイズ超過は `limit_exceeded`（20）、I/O 失敗は `runtime_error`（70）。
pub fn run(args: &RegisterArgs, cwd: &Path) -> Result<RegisterReport, ErrorReport> {
    let (def_bytes, def_real) = read_confined(cwd, &args.definition, MAX_DEFINITION_FILE_BYTES)?;
    let definition = parse_definition(&def_bytes)?;
    let hash = definition
        .canonical_hash()
        .map_err(|_| runtime("cannot compute definition hash"))?;
    let Some(src_dir) = def_real.parent() else {
        return Err(invalid("definition path is invalid"));
    };

    // 学習データが無い場合は利用者の入力不備（`invalid_input`）。権限拒否・容量超過・I/O 失敗は
    // それぞれの終了コードのまま返す。
    let (train_bytes, _) =
        read_confined_optional(cwd, &src_dir.join(TRAIN_DATA_FILE), MAX_PROJECT_FILE_BYTES)?
            .ok_or_else(|| invalid("training data file is missing"))?;
    // 評価データは任意。`NotFound` のときだけ「提供なし」とし、それ以外の失敗は停止する
    // （失敗を「無い」と読み替えると凍結記録が作られず、evaluate が skipped・exit 0 になる。REQ-17）。
    let eval_path = src_dir.join(EVALUATION_DATA_FILE);
    let evaluation = match std::fs::symlink_metadata(&eval_path) {
        Ok(_) => {
            let (bytes, _) = read_confined(cwd, &eval_path, MAX_PROJECT_FILE_BYTES)?;
            let record = freeze_eval_data(&bytes).map_err(|e| e.to_error_report())?;
            Some((bytes, record))
        }
        Err(e) if e.kind() == ErrorKind::NotFound => None,
        Err(_) => return Err(runtime("cannot inspect evaluation data file")),
    };

    let project = Project::create(cwd, &args.project_dir)?;
    let placed = place_project(&project, &def_bytes, &train_bytes, evaluation.as_ref());
    if let Err(report) = placed {
        // 本工程が作ったディレクトリだけを、保持した fd 起点で片付ける（best effort。REQ-39）。
        project.remove_created();
        return Err(report);
    }
    Ok(RegisterReport::new(
        hash.to_hex(),
        definition.options().len(),
        evaluation.is_some(),
    ))
}

/// 取り込んだファイルをプロジェクトへ置く（評価データは凍結記録つきで読み取り専用配置）。
fn place_project(
    project: &Project,
    def_bytes: &[u8],
    train_bytes: &[u8],
    evaluation: Option<&(Vec<u8>, FreezeRecord)>,
) -> Result<(), ErrorReport> {
    project.write_new(DEFINITION_FILE, def_bytes)?;
    project.create_dir(DATA_DIR)?;
    project.write_new(Path::new(DATA_DIR).join(TRAIN_DATA_FILE), train_bytes)?;
    if let Some((eval_bytes, record)) = evaluation {
        place_evaluation(project, eval_bytes, record)?;
        let json = record
            .to_json()
            .map_err(|_| runtime("cannot serialize freeze record"))?;
        project.write_new(FREEZE_FILE, json.as_bytes())?;
    }
    Ok(())
}

/// 評価データを `data/` へ読み取り専用で配置する（Linux・macOS）。
///
/// 凍結記録との照合・0400 化・書き込み拒否のプローブ・原子的な公開の手順は data 層の
/// `place_read_only_bytes` に一本化する（モード 0400 だけで凍結済みとみなさない。REQ-17・REQ-39）。
/// コピー元は `read_confined` で検証・読み込み済みのバイト列（パスを開き直さない）、配置先は
/// `data/` を保持 fd から開いたハンドル（パスを再解決しない）。`data/` は 0700 の管理ディレクトリ。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn place_evaluation(
    project: &Project,
    eval_bytes: &[u8],
    record: &FreezeRecord,
) -> Result<(), ErrorReport> {
    use crate::frozen_dir::HeldPlacementDir;
    use fandhe_edge_core::exitcode::ExitCode;
    use fandhe_edge_data::frozen_placement::{PlacementError, place_read_only_bytes};

    let data_dir = HeldPlacementDir::new(project.open_subdir(DATA_DIR)?);
    place_read_only_bytes(
        eval_bytes,
        &data_dir,
        EVALUATION_DATA_FILE,
        record,
        MAX_PROJECT_FILE_BYTES,
    )
    .map(|_| ())
    .map_err(|e| match &e {
        // 入力起因（記録との不一致・既存）は `invalid_input`、サイズ超過は `limit_exceeded`、
        // 書き込み拒否を確認できない環境（root・ACL 等）や I/O 失敗は `runtime_error`（fail-closed）。
        // message は固定語彙でパスを含めない。
        PlacementError::HashMismatch => invalid("evaluation data does not match freeze record"),
        PlacementError::AlreadyExists { .. } => invalid("file already exists"),
        PlacementError::TooLarge { .. } => crate::project::fail(
            ExitCode::LimitExceeded,
            "evaluation data exceeds size limit",
        ),
        PlacementError::Fs(fs) => fs_report(fs),
        _ => runtime("cannot place evaluation data read-only"),
    })
}

/// 保持 fd 起点の配置は Linux・macOS のみ。それ以外の OS では配置せず拒否する（fail-closed。
/// Windows は M10 時点で対象外。従来の `UnsupportedPlatform` と同じく `runtime_error`）。
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn place_evaluation(
    _project: &Project,
    _eval_bytes: &[u8],
    _record: &FreezeRecord,
) -> Result<(), ErrorReport> {
    Err(runtime("cannot place evaluation data read-only"))
}
