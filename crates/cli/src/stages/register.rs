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
//! プロジェクトへ読み取り専用配置する（[`place_read_only`]。REQ-17・REQ-39）。以後の
//! `inspect`・`evaluate` は記録とのハッシュ一致を確認し、不一致なら停止する（fail-closed）。
//!
//! 取り込みの途中で失敗した場合、本工程が作った `--project-dir` は削除する（半端な状態の
//! プロジェクトを残さない。既存の `--project-dir` は最初に拒否するため、削除対象は
//! 本工程が作ったものに限る）。

use std::path::{Path, PathBuf};

use fandhe_edge_core::definition::MAX_DEFINITION_FILE_BYTES;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::read_bounded_open_file;
use fandhe_edge_core::stage_report::RegisterReport;
use fandhe_edge_data::eval_freeze::{FreezeRecord, freeze_eval_data};
use fandhe_edge_data::frozen_placement::place_read_only;
use fandhe_edge_guard::path::open_confined;

use crate::args::RegisterArgs;
use crate::error_report::ToErrorReport;
use crate::project::{
    DATA_DIR, DEFINITION_FILE, EVALUATION_DATA_FILE, FREEZE_FILE, MAX_PROJECT_FILE_BYTES, Project,
    TRAIN_DATA_FILE, fail, fs_report, invalid, parse_definition, runtime,
};

/// cwd 配下へ閉じ込めて開き、上限付きで読む（実体パスも返す）。
fn read_confined(cwd: &Path, path: &Path, limit: u64) -> Result<(Vec<u8>, PathBuf), ErrorReport> {
    let (file, real) = open_confined(cwd, path).map_err(|e| e.to_error_report())?;
    let bytes = read_bounded_open_file(file, real.as_path(), limit).map_err(|e| fs_report(&e))?;
    Ok((bytes, real.into_path_buf()))
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

    let (train_bytes, _) =
        read_confined(cwd, &src_dir.join(TRAIN_DATA_FILE), MAX_PROJECT_FILE_BYTES).map_err(
            |e| {
                // 学習データが無い場合は利用者の入力不備として扱う。
                if e.code == ExitCode::RuntimeError {
                    invalid("training data file is missing")
                } else {
                    e
                }
            },
        )?;
    let eval_path = src_dir.join(EVALUATION_DATA_FILE);
    let evaluation = if std::fs::symlink_metadata(&eval_path).is_ok() {
        let (bytes, real) = read_confined(cwd, &eval_path, MAX_PROJECT_FILE_BYTES)?;
        let record = freeze_eval_data(&bytes).map_err(|e| e.to_error_report())?;
        Some((real, record))
    } else {
        None
    };

    let project = Project::create(cwd, &args.project_dir)?;
    let placed = place_project(&project, &def_bytes, &train_bytes, evaluation.as_ref());
    if let Err(report) = placed {
        // 本工程が作ったディレクトリだけを片付ける（best effort）。
        let _ = std::fs::remove_dir_all(project.dir());
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
    evaluation: Option<&(PathBuf, FreezeRecord)>,
) -> Result<(), ErrorReport> {
    project.write_new(DEFINITION_FILE, def_bytes)?;
    let data_dir = project.create_dir(DATA_DIR)?;
    project.write_new(Path::new(DATA_DIR).join(TRAIN_DATA_FILE), train_bytes)?;
    if let Some((src, record)) = evaluation {
        place_read_only(
            src,
            &data_dir,
            EVALUATION_DATA_FILE,
            record,
            MAX_PROJECT_FILE_BYTES,
        )
        .map_err(|_| {
            fail(
                ExitCode::RuntimeError,
                "cannot place evaluation data read-only",
            )
        })?;
        let json = record
            .to_json()
            .map_err(|_| runtime("cannot serialize freeze record"))?;
        project.write_new(FREEZE_FILE, json.as_bytes())?;
    }
    Ok(())
}
