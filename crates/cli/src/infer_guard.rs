//! `infer` の `--package` と `onnx_file` を経路の閉じ込めへ通す統合（REQ-39・PoC-20 ケース 1・
//! TASK-39.4-2・#159・`kind` の許可リスト検査は TASK-39.2-4・#156・`kind_version` は TASK-39.6-1・#174）。
//!
//! # 呼び出し文脈
//!
//! `main.rs` が `infer` の引数解析に成功した直後に [`check_infer_path_and_format`] を呼ぶ。拒否は
//! [`ErrorReport`]（`invalid_input`=64 等。message は `path rejected: <reason_code>` の固定語彙で、
//! パス・`onnx_file` の値・`artifact.json` の本文を含めない）として呼び出し側が JSON 1 行で出力する。
//!
//! # 検査の順序（fail-closed）
//!
//! 1. `--package` を workspace（カレントディレクトリ）配下へ閉じ込め、ディレクトリであることを確認
//! 2. パッケージ配下の `artifact.json` を、開いた fd から上限付きで読む
//! 3. `onnx_file`・`kind`・`kind_version` を取り出す（`onnx_file`・`kind` の欠落・型違い、`kind_version` の型違いは `artifact metadata is invalid`・64。`kind_version` の欠落は版 1 として扱う）
//! 4. `kind` を許可リスト（`KindAllowlist::supported()`）で検査する。拒否は
//!    `kind rejected: <reason_code>`・64。**モデルのバイト列に触れる前**に行う（fail-closed）
//! 5. `kind_version` を `kind` ごとの許可リスト（`KindVersionAllowlist::supported()`）で検査する。
//!    拒否は `kind_version rejected: <reason_code>`・64。**モデルのバイト列に触れる前**に行う
//! 6. `--version-ledger L` があるときだけ、2 で読んだ `artifact.json` のバイト列の sha256 を台帳の
//!    `(model, V)` の記録と照合する（[`verify_against_ledger`]。前版復帰。REQ-39・TASK-39.6・#491）。
//!    `V` は `--version-id`、省略時は台帳の最新 model 版。`L` は cwd 配下へ閉じ込めて上限
//!    （`MAX_VERSION_LEDGER_BYTES`）つきで読む。未記録は `version is not recorded`、不一致は
//!    `package does not match version ledger`（いずれも 64）。ONNX を開く前に行う
//! 7. `onnx_file` を、パッケージ配下（かつ workspace 配下）の ONNX ファイルを開き、
//!    拡張子（`.onnx`）を確認し、保持した fd を上限（`MAX_MODEL_FILE_BYTES`）付きで読み切って
//!    許可制の形式検査（ONNX のみ許可。pickle 偽装・非 ONNX は拒否）を通す
//!    （形式不許可は `invalid_input`=64、超過は `limit_exceeded`=20。REQ-39）
//!
//! # 未検証の項目（「検証済み」ではない）
//!
//! 本モジュールが行うのは経路の閉じ込め・形式の許可制と、指定時の版管理台帳との照合のみ。次は**まだ検査していない**ため、
//! 戻り値を「完全性・版まで検証済み」と扱ってはならない。
//!
//! - 完全性（モデルの sha256 照合）: `--version-ledger` を指定したときだけ、`artifact.json` を台帳と照合する
//!   （ONNX・語彙・校正は `artifact.json` の記録との一致を `stages::infer` が確認するので、台帳の 1 件で
//!   束縛される。#491）。指定しなければ外部台帳による照合は行わない。台帳ファイル自体の改変は検出しない
//! - 読み込み前のサイズ上限の正式値: #172（TASK-39.5-3）
//!
//! パッケージ形式のうち sha256 の欄は未定義で、形式の確定は TASK-28・TASK-32 で行う。`kind_version` の許可リスト検査は TASK-39.6-1（#174）で実装済み。
//! `--input-file` と `--out` の閉じ込めは本関数の範囲外で、`stages::infer` が個別に行う
//! （`--input-file` は `open_confined`、`--out` は `OutTarget::preflight`。#459）。
//!
//! # 推論への接続（#136。`stages::infer`）
//!
//! - 返す [`PathFormatCheckedInputs::kind`] は許可リスト検査済みの `kind`。ランタイムの
//!   `ModelKind::parse` へ写すこと。ガードの許可集合（`c1`・`c3`・`autoregressive`）はランタイムの
//!   対応（`c1`・`c3`）より広く、`autoregressive` はガードを通ってもランタイムで
//!   `unsupported_kind` になる（別の層の検査。autoregressive の ONNX 推論は未着手）
//! - `train`・`register` への `kind` 検査の統合は未了（#136）。両工程は下位層へ接続済みだが、
//!   ガードの `KindAllowlist` は通していない（候補の `artifact.json` の `kind` 検査は
//!   `stages::candidate_artifact` が担う）。`train` へ統合する際は学習ワーカー起動前に通すこと
//! - 返す [`PathFormatCheckedInputs::onnx`] は閉じ込めつきで開いて形式検査のみを通した [`CheckedFile`]。推論への接続では、この
//!   バイト列（`as_bytes`・`Read`）をランタイムへ渡す（`stages::infer` はそうしている）。パスから開き直すと検証後の差し替え
//!   （TOCTOU）が残るため、パスを受け取って自前で開く読み込み API は使わない。

use std::path::Path;

use fandhe_edge_core::artifact_meta::{ArtifactOnnxRef, MAX_ARTIFACT_META_BYTES};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::read_bounded_open_file;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::version_ledger_record::MAX_VERSION_LEDGER_BYTES;
use fandhe_edge_guard::format::{CheckedFile, FormatAllowlist, FormatRejection, check_open_file};
use fandhe_edge_guard::kind::{CheckedKind, KindAllowlist};
use fandhe_edge_guard::kind_version::{CheckedKindVersion, KindVersionAllowlist};
use fandhe_edge_guard::model_file::MODEL_FILE_EXTENSION;
use fandhe_edge_guard::package::{ConfinedPackage, confine_package};
use fandhe_edge_guard::path::{ConfinedPath, open_confined};
use fandhe_edge_guard::version_ledger::{ArtifactKind, VersionId, VersionLedger};
use fandhe_edge_runtime::onnx::MAX_MODEL_FILE_BYTES;

use crate::args::{InferArgs, InferVersion};
use crate::error_report::ToErrorReport;

/// パッケージ内のメタデータのファイル名（パッケージ形式の確定は TASK-28/32）。
const ARTIFACT_META_FILE: &str = "artifact.json";

/// 経路の閉じ込めと形式の許可制のみを通過した `infer` の入力。
///
/// **完全性（sha256）は `--version-ledger` 指定時の `artifact.json` の照合のみ**（#491。モジュール doc 参照）。
/// 台帳を指定しなければ、改変されたモデルでも経路・形式・版が正しければ返る。`kind_version` は許可リスト検査済み。
#[derive(Debug)]
pub struct PathFormatCheckedInputs {
    /// 許可リストで検査済みの `kind`（`&'static str`。入力の文字列は流れない）。
    pub kind: CheckedKind,
    /// 許可リストで検査済みの `kind_version`（`kind` との組で合格した証明。TASK-39.6-1・#174）。
    pub kind_version: CheckedKindVersion,
    /// workspace 配下へ閉じ込め済みのパッケージ。
    pub package: ConfinedPackage,
    /// 閉じ込めつきで開き、形式検査のみを通した ONNX のバイト列（sha256 未照合）。読むときはこれだけを使う。
    pub onnx: CheckedFile,
    /// `onnx` の実パス（表示・診断用。開き直さない）。
    pub onnx_path: ConfinedPath,
    /// 検査に使った `artifact.json` のバイト列（台帳と照合済みならその内容。`stages::infer` は開き直さず
    /// これを解析する。#491）。
    pub meta_bytes: Vec<u8>,
}

/// `args.package` と `artifact.json` の `onnx_file` の経路を閉じ込め、ONNX ファイルを開いて形式を検査する。
/// `kind_version` は許可リストで検査する（#174）。`args.version`（`--version-ledger`）があるときだけ、ONNX を
/// 開く前に `artifact.json` のバイト列の sha256 を版管理台帳と照合する（#491）。ONNX・定義などパッケージ内の
/// sha256 の自己整合は `stages::infer` が `artifact.json` の記録と照合する。
///
/// `workspace` はカレントディレクトリ（CLI の規約。容量計測 example と同じ）。
///
/// # Errors
/// 経路の拒否は `invalid_input`（64）、`artifact.json` の上限超過は `limit_exceeded`（20）、
/// 不正な内容は `invalid_input`、I/O 起因・非対応 OS は `runtime_error`（70）。
pub fn check_infer_path_and_format(
    workspace: &Path,
    args: &InferArgs,
) -> Result<PathFormatCheckedInputs, ErrorReport> {
    let package = confine_package(workspace, &args.package).map_err(|e| e.to_error_report())?;
    let (meta_file, _) = package
        .open_member(Path::new(ARTIFACT_META_FILE))
        .map_err(|e| e.to_error_report())?;
    let bytes = read_bounded_open_file(
        meta_file,
        Path::new(ARTIFACT_META_FILE),
        MAX_ARTIFACT_META_BYTES,
    )
    .map_err(|e| e.to_error_report())?;
    let onnx_ref = ArtifactOnnxRef::parse(&bytes).map_err(|e| e.to_error_report())?;
    // モデルのバイト列を開く前に kind を拒否する（REQ-39・PoC-20）。
    let kind = KindAllowlist::supported()
        .check(onnx_ref.kind())
        .map_err(|e| e.to_error_report())?;
    // 版の検査も同様にモデルのバイト列を開く前に行う（REQ-39・PoC-20 ケース 7・TASK-39.6-1）。
    let kind_version = KindVersionAllowlist::supported()
        .check(kind, onnx_ref.kind_version())
        .map_err(|e| e.to_error_report())?;
    // 前版復帰の照合も ONNX を開く前に行う（REQ-39・TASK-39.6・#491）。
    if let Some(version) = &args.version {
        verify_against_ledger(workspace, version, &bytes)?;
    }
    let (onnx, onnx_path) = package
        .open_member(Path::new(onnx_ref.onnx_file()))
        .map_err(|e| e.to_error_report())?;
    // 実体パスの拡張子を確認してから、保持した fd を上限付きで読み切り形式を検査する
    // （パスから開き直さない。読み込み前の上限確認は check_open_file 内。REQ-39）。
    if onnx_path
        .as_path()
        .extension()
        .is_none_or(|e| e != MODEL_FILE_EXTENSION)
    {
        return Err(FormatRejection::ExtensionNotAllowed {
            expected: MODEL_FILE_EXTENSION,
        }
        .to_error_report());
    }
    let onnx = check_open_file(
        onnx,
        onnx_path.as_path(),
        &FormatAllowlist::onnx_only(),
        MAX_MODEL_FILE_BYTES,
    )
    .map_err(|e| e.to_error_report())?;
    Ok(PathFormatCheckedInputs {
        kind,
        kind_version,
        package,
        onnx,
        onnx_path,
        meta_bytes: bytes,
    })
}

/// `artifact.json` のバイト列（`meta_bytes`）の sha256 が、版管理台帳の `(model, V)` の記録と一致するか確認する
/// （REQ-39・TASK-39.6・#491）。`V` は `--version-id`、省略時は台帳の最新 model 版。`package/` には何も書かない。
///
/// # Errors
/// 台帳の経路の拒否（cwd 外・cwd 外を指す symlink 等）は経路の写像、上限超過は `limit_exceeded`、形の不正は
/// `version ledger is invalid`（[`fandhe_edge_guard::version_ledger::LedgerError::exit_code`] の写像）、
/// 不正な版 ID はその検証の固定文、未記録は `version is not recorded`、不一致は
/// `package does not match version ledger`（いずれも 64）。メッセージはパスを含めない。
fn verify_against_ledger(
    workspace: &Path,
    version: &InferVersion,
    meta_bytes: &[u8],
) -> Result<(), ErrorReport> {
    let (file, path) =
        open_confined(workspace, &version.ledger).map_err(|e| e.to_error_report())?;
    let bytes = read_bounded_open_file(file, path.as_path(), MAX_VERSION_LEDGER_BYTES)
        .map_err(|e| crate::project::fs_report(&e))?;
    let ledger = VersionLedger::from_file(&bytes)
        .map_err(|e| ErrorReport::new(e.exit_code(), "version ledger is invalid"))?;
    let entry = match &version.id {
        Some(raw) => {
            let id =
                VersionId::new(raw).map_err(|e| ErrorReport::new(e.exit_code(), e.to_string()))?;
            ledger.get(ArtifactKind::Model, &id)
        }
        None => ledger.versions_of(ArtifactKind::Model).last(),
    }
    .ok_or_else(|| ErrorReport::new(ExitCode::InvalidInput, "version is not recorded"))?;
    if *entry.sha256() != Sha256Digest::of_bytes(meta_bytes) {
        return Err(ErrorReport::new(
            ExitCode::InvalidInput,
            "package does not match version ledger",
        ));
    }
    Ok(())
}
