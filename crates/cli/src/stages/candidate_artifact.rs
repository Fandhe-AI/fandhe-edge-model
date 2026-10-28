//! 学習済み候補の成果物（`artifact.json`・ONNX）を、閉じ込めつきで読み込む共有処理
//! （REQ-27・REQ-32・REQ-39・TASK-33.1-2・#136・#314）。
//!
//! # 呼び出し文脈
//!
//! `package`（選定候補の公開）と `evaluate`（評価する候補の重みの取得）が同じ規則で成果物を読む
//! ために、読み込みと整合性の検査をここに 1 つだけ置く。候補ディレクトリは [`Project`] の保持 fd
//! からの相対オープン（`O_NOFOLLOW`・`O_DIRECTORY`）で得て、`artifact_dir` は文字列の前方一致でなく
//! 候補ディレクトリからの相対パスとして求める（学習結果の差し替えで候補の外を読まない。REQ-39）。
//!
//! # kind 別の分岐の集約（REQ-41・#662）
//!
//! 入力表現が tokenizer の `qwen2_lora` だけは、成果物に `tokenizer.json` と外部データの shard が付く。
//! `select`・`evaluate`（・`package`）が kind ごとに同じ分岐を持たないよう、形式許可リスト
//! （[`crate::infer_guard::format_allowlist_for`]）・パイプライン組立（[`build_candidate_pipeline`]）・
//! 計測対象の列挙（[`measured_members`]）をここへ置く。読み込み時の構造検査では `tokenizer.json` を
//! 上限つきで読んで sha256 を照合し、shard は開いて fstat のサイズ上限だけを確認する
//! （中身は読まない。sha256 の照合は推論のパイプライン組立時にランタイムが読みながら行う）。

use std::fs::File;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use fandhe_edge_core::artifact_meta::{
    ArtifactMeta, ArtifactMetaError, MAX_ARTIFACT_META_BYTES, TokenizerPackageRefs,
    VocabStreamError, verify_vocab_stream,
};
use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_core::fs::{FsError, read_bounded_open_file};
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_guard::kind::KindAllowlist;
use fandhe_edge_guard::package::ConfinedPackage;
use fandhe_edge_guard::path::PathRejection;
use fandhe_edge_guard::split_files::MAX_SHARD_FILE_BYTES;
use fandhe_edge_runtime::capacity::{MAX_FILE_BYTES, PackageComponent};
use fandhe_edge_runtime::onnx::{MAX_MAX_BYTES, MAX_MODEL_FILE_BYTES, MIN_MAX_BYTES};
use fandhe_edge_runtime::tokenizer::MAX_TOKENIZER_JSON_BYTES;
use fandhe_edge_runtime::vocab_exclusion::VOCAB_FILE_NAME;
use fandhe_edge_train::result::SuccessOutcome;

use crate::error_report::ToErrorReport;
use crate::project::{Project, fs_report, invalid, runtime};

use super::infer::{
    Pipeline, build_pipeline, build_qwen2_pipeline, check_input_representation, is_tokenizer_qwen2,
};
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
    /// 成果物ディレクトリ（保持 fd 起点。tokenizer 表現の shard を推論時に開く口。REQ-39）。
    pub dir: ConfinedPackage,
    /// `tokenizer.json` のバイト列（tokenizer 表現のときだけ。`MAX_TOKENIZER_JSON_BYTES` 以下で、
    /// `artifact.json` 記載の sha256 と照合済み。`evaluate` が前後ハッシュ照合のために複製する）。
    pub tokenizer_bytes: Option<Vec<u8>>,
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
    /// `tokenizer.json`（tokenizer 表現のときだけ。sha256 照合済み）。
    pub tokenizer: Option<(File, PathBuf)>,
    /// 外部データの shard（tokenizer 表現のときだけ。`(location, fd, 実パス)`。fstat のサイズ上限だけ確認済みで、
    /// 中身の sha256 は未照合）。
    pub shards: Vec<(String, File, PathBuf)>,
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
    // 成果物ディレクトリは保持 fd で 1 回だけ開き、メンバーはすべてそこから相対で開く（ディレクトリを
    // 辿るのは 1 回。tokenizer 表現の shard を推論時に開き直す口にもなる。REQ-39）。
    let dir = candidate_dir
        .open_subdir(&artifact_rel)
        .map_err(|e| e.to_error_report())?;
    // 各メンバーは 1 回だけ開き、読み込み用に複製した fd で読む（元の fd は容量計測用に保持する）。
    let open_member = |name: &str| dir.open_member(Path::new(name));
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
    // tokenizer 表現は `qwen2_lora` だけが推論できる。それ以外の宣言は構造検査（tokenizer.json・shard の
    // 読み込み）へ進めず先に拒否する（REQ-41・#662）。
    check_input_representation(&meta)?;
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
    let (tokenizer, tokenizer_bytes, shards) = if is_tokenizer_qwen2(&meta) {
        let members = open_tokenizer_members(&dir, &meta_bytes)?;
        (
            Some(members.tokenizer),
            Some(members.tokenizer_bytes),
            members.shards,
        )
    } else {
        (None, None, Vec::new())
    };
    Ok(CandidateArtifact {
        meta_bytes,
        meta,
        onnx_bytes,
        onnx_file: onnx_file.to_string(),
        handles: ArtifactHandles {
            meta: (meta_file, meta_path),
            onnx: (onnx_handle, onnx_path),
            vocab,
            tokenizer,
            shards,
        },
        dir,
        tokenizer_bytes,
    })
}

/// [`open_tokenizer_members`] の戻り値。
struct TokenizerMembers {
    tokenizer: (File, PathBuf),
    tokenizer_bytes: Vec<u8>,
    shards: Vec<(String, File, PathBuf)>,
}

/// tokenizer 表現の成果物の構造検査（REQ-41・REQ-39・#662）。
///
/// `artifact.json` の参照欄（[`TokenizerPackageRefs`]）から `tokenizer.json` を閉じ込めつきで開いて
/// `MAX_TOKENIZER_JSON_BYTES` 以下で読み、記録の sha256 と照合する。各 shard は開いて fstat で通常ファイルか・
/// `MAX_SHARD_FILE_BYTES` 以下かだけを確認する（中身は読まない。`select` が GiB を読まないため。sha256 の照合は
/// [`build_candidate_pipeline`] が組むランタイムが読みながら行い、不一致は fail-closed）。
fn open_tokenizer_members(
    dir: &ConfinedPackage,
    meta_bytes: &[u8],
) -> Result<TokenizerMembers, ErrorReport> {
    let refs = TokenizerPackageRefs::parse(meta_bytes).map_err(|e| e.to_error_report())?;
    let (file, real) = dir
        .open_member(Path::new(&refs.tokenizer_file))
        .map_err(|e| e.to_error_report())?;
    let reader = file
        .try_clone()
        .map_err(|_| runtime("cannot read candidate artifact"))?;
    let tokenizer_bytes = read_bounded_open_file(reader, real.as_path(), MAX_TOKENIZER_JSON_BYTES)
        .map_err(|e| fs_report(&e))?;
    if Sha256Digest::of_bytes(&tokenizer_bytes).to_hex() != refs.tokenizer_sha256 {
        return Err(invalid(
            "package tokenizer does not match its recorded hash",
        ));
    }
    let mut shards = Vec::with_capacity(refs.external_data.len());
    for e in &refs.external_data {
        let (shard, shard_real) = dir
            .open_member(Path::new(&e.location))
            .map_err(|err| err.to_error_report())?;
        check_shard_size(&shard, shard_real.as_path())?;
        shards.push((e.location.clone(), shard, shard_real.into_path_buf()));
    }
    Ok(TokenizerMembers {
        tokenizer: (file, real.into_path_buf()),
        tokenizer_bytes,
        shards,
    })
}

/// 開いた shard が通常ファイルで `MAX_SHARD_FILE_BYTES` 以下かを fstat で確認する（1 バイトも読まない）。
fn check_shard_size(file: &File, path: &Path) -> Result<(), ErrorReport> {
    let metadata = file.metadata().map_err(|source| {
        fs_report(&FsError::Read {
            path: path.to_path_buf(),
            source,
        })
    })?;
    if !metadata.is_file() {
        return Err(fs_report(&FsError::NotRegularFile {
            path: path.to_path_buf(),
        }));
    }
    if metadata.len() > MAX_SHARD_FILE_BYTES {
        return Err(fs_report(&FsError::TooLarge {
            path: path.to_path_buf(),
            size: metadata.len(),
            limit: MAX_SHARD_FILE_BYTES,
        }));
    }
    Ok(())
}

/// 候補の入力表現に合う推論パイプラインを組む（byte は [`build_pipeline`]、tokenizer の `qwen2_lora` は
/// [`build_qwen2_pipeline`]。`evaluate` が校正と適用で共有する。REQ-32・REQ-41・REQ-39）。
///
/// `onnx` は検証済みのバイト列（評価ではステージングから読み直して sha256 照合した方を渡す）。
/// tokenizer 表現では shard を成果物ディレクトリの保持 fd から開き、ランタイムが sha256 を照合する。
pub(crate) fn build_candidate_pipeline(
    onnx: &[u8],
    artifact: &CandidateArtifact,
    n_options: usize,
) -> Result<Pipeline, ErrorReport> {
    if is_tokenizer_qwen2(&artifact.meta) {
        build_qwen2_pipeline(
            onnx,
            &artifact.dir,
            &artifact.meta_bytes,
            &artifact.meta,
            n_options,
        )
    } else {
        build_pipeline(onnx, &artifact.meta, n_options)
    }
}

/// 評価台帳が候補を見分ける識別（重みの識別）の正準入力の領域名。
const WEIGHTS_IDENTITY_DOMAIN: &[u8] = b"fandhe-edge/evaluate/weights-identity/v1\0";

/// ONNX 本体の sha256 と `artifact.json`（shard の sha256 を含む）の sha256 から、台帳の識別を求める
/// （正準化は本関数の 1 か所。固定長 32 バイトの 2 値を領域名の後ろへ連結して sha256。REQ-27・#662）。
fn weights_identity_digest(onnx: &Sha256Digest, artifact_meta: &Sha256Digest) -> Sha256Digest {
    let mut buf = Vec::with_capacity(WEIGHTS_IDENTITY_DOMAIN.len() + 64);
    buf.extend_from_slice(WEIGHTS_IDENTITY_DOMAIN);
    buf.extend_from_slice(onnx.as_bytes());
    buf.extend_from_slice(artifact_meta.as_bytes());
    Sha256Digest::of_bytes(&buf)
}

/// 評価台帳（`final_test_once`）が候補を見分けるときの重みの識別を返す（REQ-27・#662）。
///
/// byte 表現（c1・c3）は `None` で、識別は従来どおり ONNX ファイルの sha256（評価器がファイルから求める。
/// 記録・挙動は変えない）。tokenizer 表現（`qwen2_lora`）は `Some(ONNX の sha256 と `artifact.json` の sha256 から
/// 求めた値)`。ONNX 本体は shard の中身が違っても同じバイト列になりうる（重みの大半は外部データで、ONNX には
/// 場所・オフセット・長さしか入らない）ため、ONNX だけでは shard だけが違う 2 候補が同じ識別になり、
/// 2 つ目が重みロックで `already evaluated` と誤って断られる。`artifact.json` は各 shard の sha256 を持つので、
/// 識別へ含めれば shard の違いが識別に出る。ファイルの照合（ステージングした ONNX の `ModelChanged` など）は
/// 識別とは別に従来どおり行う。
pub(crate) fn ledger_weights_identity(artifact: &CandidateArtifact) -> Option<Sha256Digest> {
    is_tokenizer_qwen2(&artifact.meta).then(|| {
        weights_identity_digest(
            &Sha256Digest::of_bytes(&artifact.onnx_bytes),
            &Sha256Digest::of_bytes(&artifact.meta_bytes),
        )
    })
}

/// 評価台帳へ事前登録する重みの識別（byte は ONNX ファイルの sha256。[`ledger_weights_identity`]）。
pub(crate) fn registered_weights_digest(artifact: &CandidateArtifact) -> Sha256Digest {
    ledger_weights_identity(artifact)
        .unwrap_or_else(|| Sha256Digest::of_bytes(&artifact.onnx_bytes))
}

/// 容量計測の対象（`(区分, 実パス, fd)`）を列挙する。ONNX と shard は重み、`artifact.json` はメタデータ、
/// `vocab.json` と `tokenizer.json` は語彙・特徴量変換に割り当てる（`capacity` の 5 区分は据え置き。REQ-30・#662）。
/// 選択肢表（`definition.json`）は成果物の外にあるため呼び出し側が足す。
///
/// 保持 fd は `try_clone` で複製する（`fstat` で測るだけで、読み位置は共有しても影響しない）。
pub(crate) fn measured_members(
    handles: &ArtifactHandles,
) -> Result<Vec<(PackageComponent, PathBuf, File)>, ErrorReport> {
    let clone = |file: &File| {
        file.try_clone()
            .map_err(|_| runtime("cannot read candidate artifact"))
    };
    let mut members = vec![
        (
            PackageComponent::Weights,
            handles.onnx.1.clone(),
            clone(&handles.onnx.0)?,
        ),
        (
            PackageComponent::Metadata,
            handles.meta.1.clone(),
            clone(&handles.meta.0)?,
        ),
    ];
    for (_, file, path) in &handles.shards {
        members.push((PackageComponent::Weights, path.clone(), clone(file)?));
    }
    for (file, path) in handles.vocab.iter().chain(handles.tokenizer.iter()) {
        members.push((
            PackageComponent::VocabOrFeatureTransform,
            path.clone(),
            clone(file)?,
        ));
    }
    Ok(members)
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
    // 推論できない入力表現の候補は受け付けない（REQ-41・TASK-41.3-1・#402）。tokenizer は `qwen2_lora` のみ
    // （`infer` は #626、`select`・`evaluate` は #662-a で接続。`package` は #662-b）。
    super::infer::check_input_representation(meta)?;
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
// 閉じ込めつきの読み込みは Linux・macOS 前提のため、unix に限る（保持 fd の計測は unix の dev/ino を使う）。
#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// テスト用の一時ディレクトリ。
    fn workdir(name: &str) -> PathBuf {
        let dir = std::fs::canonicalize(std::env::temp_dir())
            .expect("temp dir")
            .join(format!(
                "fandhe-candidate-artifact-{name}-{}",
                std::process::id()
            ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    fn meta(kind: &str, extra: &str) -> ArtifactMeta {
        let z = "0".repeat(64);
        ArtifactMeta::parse(
            format!(
                r#"{{"kind":"{kind}","kind_version":1,"max_bytes":64,"label_order":["a","b"],"onnx_file":"model.onnx","onnx_sha256":"{z}"{extra}}}"#
            )
            .as_bytes(),
        )
        .expect("meta")
    }

    /// REQ-27・#662: 識別は ONNX の sha256 と `artifact.json` の sha256 の両方で決まる。ONNX が同じで
    /// `artifact.json`（shard の sha256）だけが違えば別の識別になり、同じ入力なら同じ識別になる。
    #[test]
    fn req27_issue662_weights_identity_depends_on_onnx_and_artifact_meta() {
        let onnx = Sha256Digest::of_bytes(b"onnx");
        let meta_a = Sha256Digest::of_bytes(b"meta-a");
        let meta_b = Sha256Digest::of_bytes(b"meta-b");
        assert_eq!(
            weights_identity_digest(&onnx, &meta_a),
            weights_identity_digest(&onnx, &meta_a)
        );
        assert_ne!(
            weights_identity_digest(&onnx, &meta_a),
            weights_identity_digest(&onnx, &meta_b)
        );
        assert_ne!(
            weights_identity_digest(&onnx, &meta_a),
            weights_identity_digest(&meta_a, &onnx)
        );
        assert_ne!(weights_identity_digest(&onnx, &meta_a), onnx);
    }

    /// REQ-27・#662: fixture の `qwen2_lora` 成果物で、`model.onnx` が同じまま shard の sha256（`artifact.json`）
    /// だけが違う候補は、ONNX の sha256 が同じでも台帳の識別が別になる。
    #[test]
    fn req27_issue662_fixture_candidates_differing_only_in_shard_get_distinct_identities() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/qwen2_parity");
        let onnx = Sha256Digest::of_bytes(&std::fs::read(dir.join("model.onnx")).expect("onnx"));
        let meta = std::fs::read_to_string(dir.join("artifact.json")).expect("meta");
        let shard =
            Sha256Digest::of_bytes(&std::fs::read(dir.join("weights-00000.bin")).expect("shard"))
                .to_hex();
        assert!(meta.contains(&shard), "shard sha256 is recorded");
        let other = meta.replace(&shard, &"1".repeat(64));
        assert_ne!(meta, other);
        let a = weights_identity_digest(&onnx, &Sha256Digest::of_bytes(meta.as_bytes()));
        let b = weights_identity_digest(&onnx, &Sha256Digest::of_bytes(other.as_bytes()));
        assert_ne!(a, b);
        assert_ne!(a, onnx);
    }

    /// REQ-41・#662: tokenizer 表現を通すのは `qwen2_lora` だけ。c1 など他の kind の tokenizer 宣言は
    /// `unsupported input representation`（64）で止まり、byte 前処理の組み立ては byte 表現にしか行わない。
    #[test]
    fn req41_only_qwen2_lora_may_declare_tokenizer_representation() {
        let tokenizer = r#","input_representation":"tokenizer""#;
        assert!(check_input_representation(&meta("c1", "")).is_ok());
        assert!(check_input_representation(&meta("qwen2_lora", tokenizer)).is_ok());
        let err = check_input_representation(&meta("c1", tokenizer)).expect_err("c1 tokenizer");
        assert_eq!(err.code, fandhe_edge_core::exitcode::ExitCode::InvalidInput);
        assert_eq!(err.message, "unsupported input representation");
        let err = match build_pipeline(b"", &meta("qwen2_lora", tokenizer), 2) {
            Ok(_) => panic!("must reject"),
            Err(e) => e,
        };
        assert_eq!(err.message, "unsupported input representation");
    }

    /// REQ-39・#662: shard の fstat 検査は通常ファイルだけを通し、ディレクトリを拒否する。上限は 2 GiB
    /// （巨大ファイルを作らずに済むよう、定数の値そのものを確認する）。
    #[test]
    fn req39_shard_size_check_accepts_regular_file_and_rejects_directory() {
        let dir = workdir("shard");
        let file_path = dir.join("weights-00000.bin");
        std::fs::write(&file_path, b"0123").expect("write");
        let file = File::open(&file_path).expect("open");
        assert!(check_shard_size(&file, &file_path).is_ok());
        let as_dir = File::open(&dir).expect("open dir");
        let err = check_shard_size(&as_dir, &dir).expect_err("directory");
        assert_eq!(err.code, fandhe_edge_core::exitcode::ExitCode::RuntimeError);
        assert_eq!(MAX_SHARD_FILE_BYTES, 2 * 1024 * 1024 * 1024);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-30・#662: 計測対象は ONNX と shard が重み、`artifact.json` がメタデータ、`tokenizer.json` が語彙・
    /// 特徴量変換に割り当たる（区分は据え置き）。byte 表現（shard・tokenizer なし）は従来の 2 件だけ。
    #[test]
    fn req30_measured_members_assign_components() {
        let dir = workdir("members");
        let open = |name: &str| {
            let path = dir.join(name);
            std::fs::write(&path, name.as_bytes()).expect("write");
            (File::open(&path).expect("open"), path)
        };
        let handles = ArtifactHandles {
            meta: open("artifact.json"),
            onnx: open("model.onnx"),
            vocab: None,
            tokenizer: Some(open("tokenizer.json")),
            shards: {
                let (file, path) = open("weights-00000.bin");
                vec![("weights-00000.bin".to_string(), file, path)]
            },
        };
        let members = measured_members(&handles).expect("members");
        let summary: Vec<(PackageComponent, String)> = members
            .iter()
            .map(|(c, p, _)| {
                (
                    *c,
                    p.file_name().expect("name").to_string_lossy().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                (PackageComponent::Weights, "model.onnx".to_string()),
                (PackageComponent::Metadata, "artifact.json".to_string()),
                (PackageComponent::Weights, "weights-00000.bin".to_string()),
                (
                    PackageComponent::VocabOrFeatureTransform,
                    "tokenizer.json".to_string()
                ),
            ]
        );
        let bytes_only = ArtifactHandles {
            tokenizer: None,
            shards: Vec::new(),
            ..handles
        };
        assert_eq!(measured_members(&bytes_only).expect("members").len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
