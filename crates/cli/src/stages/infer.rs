//! `infer` 工程: 配布パッケージから推論する（REQ-28・REQ-32・REQ-33・REQ-39・
//! TASK-33.1-2・#136）。
//!
//! # 呼び出し文脈と検査の順序（fail-closed）
//!
//! 1. ガード層（[`check_infer_path_and_format`]）が `--package` を cwd 配下へ閉じ込め、
//!    `artifact.json` の `onnx_file` の ONNX を開いて形式を検査する（バイト列は保持済みで、
//!    以降パスから開き直さない。TOCTOU 対策）
//! 2. 同じ閉じ込め済みパッケージから `artifact.json`（拡張メタ）と `definition.json` を上限付きで
//!    読み、`onnx_sha256` と保持した ONNX のバイト列の sha256 の一致（パッケージの自己整合性）・
//!    `label_order` と定義の選択肢の宣言順の一致・`max_bytes` の範囲を確認する
//! 3. 保持したバイト列から ONNX バックエンドを組み立て、前処理と束ねて推論する
//!    （推論経路は学習側〔`fandhe-edge-train`・Python〕に依存しない。REQ-32）
//!
//! 推論関数へ渡すのは `input` のみ（REQ-27）。`--text` は 1 件・`--input-file` は 1 行 1 JSON
//! （[`crate::infer_batch`]。REQ-33 の唯一の例外）。`--out` は未実装で `runtime_error`。
//!
//! # 未検証の項目（「検証済み」ではない）
//!
//! 外部台帳による sha256 完全性（#168）・`kind_version` の許可リスト（#174）は未検証。ここで
//! 行う sha256 照合は、パッケージ自身が記す値との一致だけを確認する。

use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

use fandhe_edge_core::artifact_meta::{ArtifactMeta, MAX_ARTIFACT_META_BYTES};
use fandhe_edge_core::definition::{Definition, MAX_DEFINITION_FILE_BYTES};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::read_bounded_open_file;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::infer_input::InferInput;
use fandhe_edge_guard::path::open_confined;
use fandhe_edge_runtime::onnx::{MAX_MAX_BYTES, MIN_MAX_BYTES, ModelKind, OnnxBackend};
use fandhe_edge_runtime::pipeline::InferencePipeline;
use fandhe_edge_runtime::preprocess::ByteEncodingPreprocessor;

use crate::args::{InferArgs, InferSource};
use crate::error_report::{ToErrorReport, emit_error_report};
use crate::infer_batch::{emit_infer_batch, judgment_from_prediction};
use crate::infer_guard::check_infer_path_and_format;
use crate::output::write_ok_judgment;
use crate::project::{DEFINITION_FILE, fs_report, invalid, parse_definition, runtime};

/// パッケージ内のメタデータのファイル名。
const ARTIFACT_META_FILE: &str = "artifact.json";

/// `--id` を省略した `--text` の入力 ID。
const DEFAULT_TEXT_ID: &str = "input";

type Pipeline = InferencePipeline<ByteEncodingPreprocessor, OnnxBackend>;

/// 検査を通過したパッケージから組み立てた推論の準備。
struct Prepared {
    definition: Definition,
    pipeline: Pipeline,
}

/// `infer` を実行し、結果（または `ErrorReport`）を `out` へ書く。
///
/// # Errors
/// `out` への書き込み失敗（呼び出し側は exit 70 に写し、追加の出力をしない）。
pub fn run<W: Write>(out: &mut W, args: &InferArgs, cwd: &Path) -> io::Result<ExitCode> {
    let prepared = match prepare(cwd, args) {
        Ok(p) => p,
        Err(report) => return emit_error_report(out, &report),
    };
    match &args.source {
        InferSource::Text { text, id } => {
            match infer_text(&prepared, id.as_deref().unwrap_or(DEFAULT_TEXT_ID), text) {
                Ok(result) => write_ok_judgment(out, &result),
                Err(report) => emit_error_report(out, &report),
            }
        }
        InferSource::InputFile {
            path,
            out: out_path,
        } => {
            if out_path.is_some() {
                // TODO(後続): `--out` への書き出し。未実装のため実装済みを装わず拒否する。
                return emit_error_report(out, &runtime("infer --out is not implemented yet"));
            }
            let file = match open_confined(cwd, path) {
                Ok((file, _)) => file,
                Err(rejection) => return emit_error_report(out, &rejection.to_error_report()),
            };
            emit_infer_batch(
                out,
                file,
                &prepared.definition.io().clone(),
                prepared.definition.options(),
                Arc::new(prepared.pipeline),
            )
        }
    }
}

/// `--text` の 1 件を推論して判定結果へ写す。
fn infer_text(
    prepared: &Prepared,
    id: &str,
    text: &str,
) -> Result<fandhe_edge_core::judgment::JudgmentResult, ErrorReport> {
    let input = InferInput::from_text(id, text, prepared.definition.io())
        .map_err(|e| crate::output::infer_input_error_report(&e))?;
    let prediction = prepared
        .pipeline
        .infer_one(input.input())
        .map_err(|e| e.to_error_report())?;
    judgment_from_prediction(prepared.definition.options(), input.id(), &prediction)
}

/// 経路・形式・自己整合性を検査し、推論パイプラインを組み立てる。
fn prepare(cwd: &Path, args: &InferArgs) -> Result<Prepared, ErrorReport> {
    let checked = check_infer_path_and_format(cwd, args)?;

    let (meta_file, meta_path) = checked
        .package
        .open_member(Path::new(ARTIFACT_META_FILE))
        .map_err(|e| e.to_error_report())?;
    let meta_bytes =
        read_bounded_open_file(meta_file, meta_path.as_path(), MAX_ARTIFACT_META_BYTES)
            .map_err(|e| fs_report(&e))?;
    let meta = ArtifactMeta::parse(&meta_bytes).map_err(|e| e.to_error_report())?;

    let (def_file, def_path) = checked
        .package
        .open_member(Path::new(DEFINITION_FILE))
        .map_err(|e| e.to_error_report())?;
    let def_bytes = read_bounded_open_file(def_file, def_path.as_path(), MAX_DEFINITION_FILE_BYTES)
        .map_err(|e| fs_report(&e))?;
    let definition = parse_definition(&def_bytes)?;

    // ガードが開いた ONNX の実体パスと、ここで読み直したメタデータの `onnx_file` が同じ対象を指す
    // ことを確認する（`artifact.json` を 2 回読むため、間の差し替えを検出する。バイト列は
    // 次の sha256 照合でも束縛される）。
    if !checked.onnx_path.as_path().ends_with(meta.onnx_file()) {
        return Err(invalid("artifact metadata does not match the model file"));
    }
    let onnx = checked.onnx.as_bytes();
    if meta.onnx_sha256() != Sha256Digest::of_bytes(onnx).to_hex() {
        return Err(invalid("package model does not match its recorded hash"));
    }
    let option_ids = definition.options().iter().map(|c| c.id.as_str());
    if !meta.label_order().iter().map(String::as_str).eq(option_ids) {
        return Err(invalid("package label order does not match definition"));
    }
    let max_bytes = usize::try_from(meta.max_bytes())
        .ok()
        .filter(|n| (MIN_MAX_BYTES..=MAX_MAX_BYTES).contains(n))
        .ok_or_else(|| invalid("package max_bytes is out of range"))?;
    let kind = ModelKind::parse(meta.kind()).map_err(|_| invalid("unsupported model kind"))?;
    let backend =
        OnnxBackend::from_bytes(onnx, kind).map_err(|_| invalid("model file cannot be loaded"))?;
    if backend.n_classes() != definition.options().len() {
        return Err(invalid("model output size does not match definition"));
    }
    Ok(Prepared {
        definition,
        pipeline: InferencePipeline::new(ByteEncodingPreprocessor::new(max_bytes), backend),
    })
}
