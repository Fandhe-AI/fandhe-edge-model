//! 学習済み候補の成果物（`artifact.json`・ONNX）を、閉じ込めつきで読み込む共有処理
//! （REQ-27・REQ-32・REQ-39・TASK-33.1-2・#136・#314）。
//!
//! # 呼び出し文脈
//!
//! `package`（選定候補の公開）と `evaluate`（評価する候補の重みの取得）が同じ規則で成果物を読む
//! ために、読み込みと整合性の検査をここに 1 つだけ置く。候補ディレクトリは [`Project`] の保持 fd
//! からの相対オープン（`O_NOFOLLOW`・`O_DIRECTORY`）で得て、`artifact_dir` は文字列の前方一致でなく
//! 候補ディレクトリからの相対パスとして求める（学習結果の差し替えで候補の外を読まない。REQ-39）。

use std::path::{Component, Path, PathBuf};

use fandhe_edge_core::artifact_meta::{ArtifactMeta, MAX_ARTIFACT_META_BYTES};
use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_core::fs::read_bounded_open_file;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_guard::kind::KindAllowlist;
use fandhe_edge_runtime::onnx::{MAX_MAX_BYTES, MAX_MODEL_FILE_BYTES, MIN_MAX_BYTES};
use fandhe_edge_train::result::SuccessOutcome;

use crate::error_report::ToErrorReport;
use crate::project::{Project, fs_report, invalid};

use super::train::candidate_rel;

/// 配布パッケージ・成果物内のメタデータのファイル名（`infer_guard` と同じ）。
pub(crate) const ARTIFACT_META_FILE: &str = "artifact.json";

/// 候補ディレクトリから読み込んだ成果物一式。
pub(crate) struct CandidateArtifact {
    /// `artifact.json` のバイト列そのまま（ハッシュ・コピーに使う）。
    pub meta_bytes: Vec<u8>,
    /// 解析済みのメタデータ（`onnx_sha256` と ONNX のバイト列の一致を確認済み）。
    pub meta: ArtifactMeta,
    /// ONNX のバイト列。
    pub onnx_bytes: Vec<u8>,
    /// ONNX のファイル名（単一の通常の名前であることを確認済み）。
    pub onnx_file: String,
    /// ONNX のプロジェクト内の相対パス（`Project::path` へ渡す。評価器のパスベース API 用）。
    pub onnx_rel: PathBuf,
}

/// 候補 `index` の成果物を閉じ込めつきで読み、メタデータと ONNX の自己整合性を確認する。
///
/// # Errors
/// 成果物ディレクトリが候補ディレクトリの外・ONNX ファイル名が不正・メタデータと ONNX の不一致は
/// `invalid_input`、サイズ超過は `limit_exceeded`、I/O 失敗は `runtime_error`。
pub(crate) fn load_candidate_artifact(
    project: &Project,
    index: usize,
    success: &SuccessOutcome,
) -> Result<CandidateArtifact, ErrorReport> {
    let candidate_dir = project.open_subdir(candidate_rel(index))?;
    let artifact_rel = Path::new(success.artifact_dir())
        .strip_prefix(candidate_dir.dir())
        .map_err(|_| invalid("artifact directory is outside the candidate directory"))?
        .to_path_buf();
    let onnx_file = success.artifact().onnx_file();
    if !is_single_component(onnx_file) {
        return Err(invalid("onnx file name is invalid"));
    }
    let read_member = |name: &str, limit: u64| -> Result<Vec<u8>, ErrorReport> {
        let (file, real) = candidate_dir
            .open_member(&artifact_rel.join(name))
            .map_err(|e| e.to_error_report())?;
        read_bounded_open_file(file, real.as_path(), limit).map_err(|e| fs_report(&e))
    };
    let meta_bytes = read_member(ARTIFACT_META_FILE, MAX_ARTIFACT_META_BYTES)?;
    let onnx_bytes = read_member(onnx_file, MAX_MODEL_FILE_BYTES)?;
    let meta = ArtifactMeta::parse(&meta_bytes).map_err(|e| e.to_error_report())?;
    if meta.onnx_file() != onnx_file
        || meta.onnx_sha256() != Sha256Digest::of_bytes(&onnx_bytes).to_hex()
    {
        return Err(invalid("artifact metadata does not match the model file"));
    }
    Ok(CandidateArtifact {
        meta_bytes,
        meta,
        onnx_bytes,
        onnx_file: onnx_file.to_string(),
        onnx_rel: candidate_rel(index).join(&artifact_rel).join(onnx_file),
    })
}

/// `artifact.json` の `kind`・`label_order`・`max_bytes` を、学習リクエストと定義に照合する
/// （`infer` が読み込み時に行う検査と同じ観点。食い違う成果物を公開・評価しない。REQ-32・REQ-39）。
pub(crate) fn check_meta_consistency(
    meta: &ArtifactMeta,
    definition: &Definition,
    trained_kind: &str,
    trained_kind_version: u32,
    trained_max_bytes: u32,
) -> Result<(), ErrorReport> {
    KindAllowlist::supported()
        .check(meta.kind())
        .map_err(|e| e.to_error_report())?;
    if meta.kind() != trained_kind {
        return Err(invalid(
            "artifact kind does not match the selected candidate",
        ));
    }
    if meta.kind_version() != trained_kind_version {
        return Err(invalid(
            "artifact kind_version does not match the selected candidate",
        ));
    }
    let option_ids = definition.options().iter().map(|c| c.id.as_str());
    if !meta.label_order().iter().map(String::as_str).eq(option_ids) {
        return Err(invalid("package label order does not match definition"));
    }
    if meta.max_bytes() != trained_max_bytes {
        return Err(invalid(
            "artifact max_bytes does not match the selected candidate",
        ));
    }
    let in_range = usize::try_from(meta.max_bytes())
        .is_ok_and(|n| (MIN_MAX_BYTES..=MAX_MAX_BYTES).contains(&n));
    if !in_range {
        return Err(invalid("package max_bytes is out of range"));
    }
    Ok(())
}

/// 単一の通常の名前（区切り・`..`・絶対パスを含まない）か。
pub(crate) fn is_single_component(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    )
}
