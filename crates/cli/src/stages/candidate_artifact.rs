//! 学習済み候補の成果物（`artifact.json`・ONNX）を、閉じ込めつきで読み込む共有処理
//! （REQ-27・REQ-32・REQ-39・TASK-33.1-2・#136・#314）。
//!
//! # 呼び出し文脈
//!
//! `package`（選定候補の公開）と `evaluate`（評価する候補の重みの取得）が同じ規則で成果物を読む
//! ために、読み込みと整合性の検査をここに 1 つだけ置く。候補ディレクトリは [`Project`] の保持 fd
//! からの相対オープン（`O_NOFOLLOW`・`O_DIRECTORY`）で得て、`artifact_dir` は文字列の前方一致でなく
//! 候補ディレクトリからの相対パスとして求める（学習結果の差し替えで候補の外を読まない。REQ-39）。

use std::fs::File;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use fandhe_edge_core::artifact_meta::{
    ArtifactMeta, ArtifactMetaError, MAX_ARTIFACT_META_BYTES, VocabStreamError, verify_vocab_stream,
};
use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_core::fs::{FsError, read_bounded_open_file};
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_guard::kind::KindAllowlist;
use fandhe_edge_guard::path::PathRejection;
use fandhe_edge_runtime::capacity::MAX_FILE_BYTES;
use fandhe_edge_runtime::onnx::{MAX_MAX_BYTES, MAX_MODEL_FILE_BYTES, MIN_MAX_BYTES};
use fandhe_edge_runtime::vocab_exclusion::VOCAB_FILE_NAME;
use fandhe_edge_train::result::SuccessOutcome;

use crate::error_report::ToErrorReport;
use crate::project::{Project, fs_report, invalid, runtime};

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
    /// 検証に使った保持 fd（容量計測・語彙の複写に使い回し、パスで開き直さない）。
    pub handles: ArtifactHandles,
}

/// [`load_candidate_artifact`] が読んだ成果物の保持 fd（`(fd, 閉じ込め検証済みの実パス)`）。
///
/// fd のオフセットは読み込み後の位置にあるため、内容を読み直す側は先頭へ戻す。
pub(crate) struct ArtifactHandles {
    /// `artifact.json`。
    pub meta: (File, PathBuf),
    /// ONNX ファイル。
    pub onnx: (File, PathBuf),
    /// 語彙ファイル（`vocab.json`。あるときだけ。検証済み）。
    pub vocab: Option<(File, PathBuf)>,
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
    // 各メンバーは 1 回だけ開き、読み込み用に複製した fd で読む（元の fd は容量計測用に保持する）。
    let open_member = |name: &str| candidate_dir.open_member(&artifact_rel.join(name));
    let read_member = |name: &str, limit: u64| -> Result<(Vec<u8>, File, PathBuf), ErrorReport> {
        let (file, real) = open_member(name).map_err(|e| e.to_error_report())?;
        let reader = file
            .try_clone()
            .map_err(|_| runtime("cannot read candidate artifact"))?;
        let bytes =
            read_bounded_open_file(reader, real.as_path(), limit).map_err(|e| fs_report(&e))?;
        Ok((bytes, file, real.into_path_buf()))
    };
    let (meta_bytes, meta_file, meta_path) =
        read_member(ARTIFACT_META_FILE, MAX_ARTIFACT_META_BYTES)?;
    let (onnx_bytes, onnx_handle, onnx_path) = read_member(onnx_file, MAX_MODEL_FILE_BYTES)?;
    let meta = ArtifactMeta::parse(&meta_bytes).map_err(|e| e.to_error_report())?;
    // `calibration_sha256`・`definition_sha256` は `package` が配布用の `artifact.json` にだけ書く欄で、候補側に
    // あれば改変として扱う（`package` が追記するときに重複キーを作らない。REQ-39・#497・#491）。
    if meta.onnx_file() != onnx_file
        || meta.onnx_sha256() != Sha256Digest::of_bytes(&onnx_bytes).to_hex()
        || meta.calibration_sha256().is_some()
        || meta.definition_sha256().is_some()
    {
        return Err(invalid("artifact metadata does not match the model file"));
    }
    // 語彙ファイル（あれば）: 無い（NotFound）ときだけ省略し、それ以外の失敗は止める。保持 fd 1 本を
    // ストリーミング 1 パスで検証し、以降の計測・複写にも同じ fd を使う（REQ-30・REQ-39・#125）。
    let vocab = match open_member(VOCAB_FILE_NAME) {
        Ok((file, real)) => Some((file, real.into_path_buf())),
        Err(PathRejection::Unresolvable { source, .. }) if source.kind() == ErrorKind::NotFound => {
            None
        }
        Err(e) => return Err(e.to_error_report()),
    };
    verify_vocab_file(&meta, vocab.as_ref().map(|(f, p)| (f, p.as_path())))?;
    Ok(CandidateArtifact {
        meta_bytes,
        meta,
        onnx_bytes,
        onnx_file: onnx_file.to_string(),
        handles: ArtifactHandles {
            meta: (meta_file, meta_path),
            onnx: (onnx_handle, onnx_path),
            vocab,
        },
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

/// 語彙ファイルの形式（許可制）と `artifact.json` 記載の sha256 との一致を、開いた保持 fd からの
/// ストリーミング 1 パスで検証する（`select`・`package`・`evaluate`・`infer` が共有。REQ-30・REQ-39）。
///
/// 全体をメモリへ読まず、読みながら sha256 と形式（件数・トークン長の上限つき）を確認する
/// （[`verify_vocab_stream`]）。保持 fd はオフセット 0 から読むこと（呼び出し側で開き直さない）。
/// 語彙ファイルがあるのに sha256 の記録が無い・不一致・形式不正・サイズ超過、または記録があるのに
/// ファイルが無い場合は `invalid_input`（fail-closed）、読み込み失敗は `runtime_error`。語彙ファイルも
/// 記録も無ければ `Ok(None)`。あれば fstat のサイズを返す。
pub(crate) fn verify_vocab_file(
    meta: &ArtifactMeta,
    vocab: Option<(&File, &Path)>,
) -> Result<Option<u64>, ErrorReport> {
    match (vocab, meta.vocab_sha256()) {
        (None, None) => Ok(None),
        (None, Some(_)) => Err(invalid("vocab file is missing but its hash is recorded")),
        (Some(_), None) => Err(invalid("vocab file has no recorded hash")),
        (Some((file, path)), Some(recorded)) => {
            let read_err = |source| {
                fs_report(&FsError::Read {
                    path: path.to_path_buf(),
                    source,
                })
            };
            let metadata = file.metadata().map_err(read_err)?;
            if !metadata.is_file() {
                return Err(fs_report(&FsError::NotRegularFile {
                    path: path.to_path_buf(),
                }));
            }
            let size = metadata.len();
            if size > MAX_FILE_BYTES {
                return Err(fs_report(&FsError::TooLarge {
                    path: path.to_path_buf(),
                    size,
                    limit: MAX_FILE_BYTES,
                }));
            }
            let digest = verify_vocab_stream(file, MAX_FILE_BYTES).map_err(|e| match e {
                VocabStreamError::Format => ArtifactMetaError::InvalidVocab.to_error_report(),
                VocabStreamError::Read => runtime("cannot read vocab file"),
            })?;
            if digest.to_hex() != recorded {
                return Err(invalid("vocab file does not match its recorded hash"));
            }
            Ok(Some(size))
        }
    }
}
