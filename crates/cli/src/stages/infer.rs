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
//! （[`crate::infer_batch`]。REQ-33 の唯一の例外）。`--text` も上限つき（`INFER_TIME_LIMIT`。runtime の
//! 協調的な期限＋バッチと共通の見張りで `limit_exceeded`・exit 20。REQ-39）。
//! `--out` は結果行をファイルへ書き stdout に要約 JSON（[`InferBatchReport`]）を出す（#459。
//! 契約は 2026-10-09 オーナー承認）。OUT は推論の計算前に cwd 配下の既存の親・非存在の名前で
//! あることを確認し（違反は `invalid_input`）、計算成功が確定してから `O_EXCL` で作る。上書きはしない。
//!
//! # 未検証の項目（「検証済み」ではない）
//!
//! 外部台帳による sha256 完全性（#168）・版管理台帳による前版への復帰（#174）は未検証。ここで
//! 行う sha256 照合は、パッケージ自身が記す値との一致だけを確認する。`kind_version` は
//! 許可リスト（[`ALLOWED_KIND_VERSIONS`]）で検証し、未許可の版は `invalid_input` で拒否する（REQ-39）。

use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

use fandhe_edge_core::artifact_meta::{ArtifactMeta, MAX_ARTIFACT_META_BYTES};
use fandhe_edge_core::definition::{Definition, MAX_DEFINITION_FILE_BYTES};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::read_bounded_open_file;
use fandhe_edge_core::hash::{Sha256Digest, Sha256Stream};
use fandhe_edge_core::stage_report::InferBatchReport;
use fandhe_edge_guard::package::{ConfinedPackage, confine_package};
use fandhe_edge_guard::path::{PathRejection, open_confined};
use fandhe_edge_runtime::onnx::{MAX_MAX_BYTES, MIN_MAX_BYTES, ModelKind, OnnxBackend};
use fandhe_edge_runtime::pipeline::InferencePipeline;
use fandhe_edge_runtime::preprocess::ByteEncodingPreprocessor;
use fandhe_edge_runtime::vocab_exclusion::VOCAB_FILE_NAME;

use crate::args::{InferArgs, InferSource};
use crate::error_report::{ToErrorReport, emit_error_report};
use crate::infer_batch::{
    BatchResults, WriteFailure, emit_infer_batch, emit_infer_batch_split, emit_infer_single,
};
use crate::infer_guard::check_infer_path_and_format;
use crate::output::write_stage_line;
use crate::project::{
    DEFINITION_FILE, fs_report, invalid, parse_definition, runtime, write_rejection,
};

/// パッケージ内のメタデータのファイル名。
const ARTIFACT_META_FILE: &str = "artifact.json";

/// 推論を許可する `kind_version` の許可リスト（REQ-39。`kind` ごとに列挙する）。
///
/// 学習ワーカーの登録簿（`trainer` の `resolve_kind`）と同じ版だけを許可し、未知の版のパッケージを
/// 推論へ進めない。版管理台帳による前版への復帰（#174・TASK-39.6）は未実装。
const ALLOWED_KIND_VERSIONS: &[(ModelKind, &[u32])] =
    &[(ModelKind::C1, &[1]), (ModelKind::C3, &[1])];

/// `kind_version` が許可リストにあるか。
pub(crate) fn kind_version_allowed(kind: ModelKind, version: u32) -> bool {
    ALLOWED_KIND_VERSIONS
        .iter()
        .any(|(k, versions)| *k == kind && versions.contains(&version))
}

/// `--id` を省略した `--text` の入力 ID。
const DEFAULT_TEXT_ID: &str = "input";

pub(crate) type Pipeline = InferencePipeline<ByteEncodingPreprocessor, OnnxBackend>;

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
    // OUT の検査は推論の計算（パッケージの読み込みを含む）より前に行う（#459）。
    let target = match &args.source {
        InferSource::InputFile {
            out: Some(out_path),
            ..
        } => match OutTarget::preflight(cwd, out_path) {
            Ok(target) => Some(target),
            Err(report) => return emit_error_report(out, &report),
        },
        _ => None,
    };
    let prepared = match prepare(cwd, args) {
        Ok(p) => p,
        Err(report) => return emit_error_report(out, &report),
    };
    match &args.source {
        // 単件推論も上限つきで、止まったらバッチと同じ見張りで `limit_exceeded`・exit 20 に終える
        // （runtime の協調的な期限＋CLI のプロセス境界。REQ-39）。
        InferSource::Text { text, id } => emit_infer_single(
            out,
            &prepared.definition.io().clone(),
            prepared.definition.options(),
            Arc::new(prepared.pipeline),
            id.as_deref().unwrap_or(DEFAULT_TEXT_ID),
            text,
        ),
        InferSource::InputFile { path, .. } => {
            let file = match open_confined(cwd, path) {
                Ok((file, _)) => file,
                Err(rejection) => return emit_error_report(out, &rejection.to_error_report()),
            };
            let io_schema = prepared.definition.io().clone();
            let pipeline = Arc::new(prepared.pipeline);
            let Some(target) = target else {
                return emit_infer_batch(
                    out,
                    file,
                    &io_schema,
                    prepared.definition.options(),
                    pipeline,
                );
            };
            let mut sink = LazyOut::new(&target);
            emit_infer_batch_split(
                out,
                &mut sink,
                file,
                &io_schema,
                prepared.definition.options(),
                pipeline,
            )
        }
    }
}

/// 事前検査済みの OUT（保持した親ディレクトリ fd と末尾の名前。REQ-39・#459）。
struct OutTarget {
    parent: ConfinedPackage,
    leaf: OsString,
}

impl OutTarget {
    /// OUT が cwd 配下の既存の親の下の、まだ無い通常の名前であることを確認する。
    ///
    /// # Errors
    /// 絶対パス・`..` 終わり・親の経路拒否（cwd 外・symlink・親なし）・既存（symlink を含む）は
    /// `invalid_input`。
    fn preflight(cwd: &Path, out_path: &Path) -> Result<Self, ErrorReport> {
        let leaf = out_path
            .file_name()
            .filter(|_| !out_path.is_absolute())
            .ok_or_else(|| invalid("output path is invalid"))?;
        let parent = out_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        // 閉じ込めは cwd 内を指す symlink を通すため、親の各成分が symlink でないことも確認する。
        let mut walked = cwd.to_path_buf();
        for component in parent.components() {
            walked.push(component);
            if std::fs::symlink_metadata(&walked).is_ok_and(|m| m.file_type().is_symlink()) {
                return Err(invalid("output path is invalid"));
            }
        }
        let parent = confine_package(cwd, parent).map_err(|e| e.to_error_report())?;
        // リンク先のない symlink は `open_member` では NotFound に見えるため、末尾も lstat で確認する。
        if std::fs::symlink_metadata(parent.dir().join(leaf)).is_ok() {
            return Err(invalid("output file already exists"));
        }
        match parent.open_member(Path::new(leaf)) {
            Err(PathRejection::Unresolvable { source, .. })
                if source.kind() == io::ErrorKind::NotFound => {}
            Ok(_) => return Err(invalid("output file already exists")),
            // 非対応 OS（`UnsupportedPlatform`）や閉じ込め違反を「既存」と偽らず、他工程と同じ写像にする。
            Err(e) => return Err(e.to_error_report()),
        }
        Ok(Self {
            parent,
            leaf: leaf.to_os_string(),
        })
    }
}

/// OUT の出力先。最初の書き込みで同じ親の一時名（`.tmp-<leaf>-<pid>`。`Project::publish_new_file`
/// と同じ命名）を `O_EXCL` 作成して書き、成功確定後に [`LazyOut::publish`] が名前替え
/// （上書きしない）で OUT を公開する。書いたバイト列から行数と sha256 を数える（要約と書いた
/// 内容が必ず一致する）。結果行の書き込みは計算成功の確定後にだけ始まるため、失敗時は何も作られない。
///
/// 書き出し中のウォッチドッグ終了（exit 20）や SIGKILL で残りうるのは一時名だけで、OUT に
/// 半端な内容は現れない。そのときの一時名の残骸は片付けない（best effort。REQ-33・REQ-39・#459）。
struct LazyOut<'a> {
    target: &'a OutTarget,
    tmp: std::path::PathBuf,
    file: Option<File>,
    hash: Sha256Stream,
    lines: usize,
    create_error: Option<ErrorReport>,
}

impl<'a> LazyOut<'a> {
    fn new(target: &'a OutTarget) -> Self {
        let mut tmp_name = OsString::from(".tmp-");
        tmp_name.push(&target.leaf);
        tmp_name.push(format!("-{}", std::process::id()));
        Self {
            target,
            tmp: tmp_name.into(),
            file: None,
            hash: Sha256Stream::new(),
            lines: 0,
            create_error: None,
        }
    }

    /// 作りかけの一時ファイルを消す（best effort）。
    fn cleanup(&mut self) {
        if self.file.take().is_some() {
            let _ = self.target.parent.remove_file_member(&self.tmp);
        }
    }

    /// 結果ファイル側の書き込み失敗後の後始末。一時ファイルを消し、返す報告を決める。
    fn abort(&mut self) -> ErrorReport {
        self.cleanup();
        self.create_error
            .take()
            .unwrap_or_else(|| runtime("cannot write output file"))
    }

    /// 一時ファイルを OUT へ名前替えで公開し、要約を返す。OUT が先に作られていれば `invalid_input`、
    /// それ以外の失敗は `runtime_error`。どちらも一時ファイルを消す。
    fn publish(&mut self) -> Result<InferBatchReport, ErrorReport> {
        let synced = match self.file.take() {
            Some(file) => file.sync_all().is_ok(),
            None => false,
        };
        if !synced {
            let _ = self.target.parent.remove_file_member(&self.tmp);
            return Err(runtime("cannot write output file"));
        }
        let renamed = self
            .target
            .parent
            .rename_member(&self.tmp, Path::new(&self.target.leaf));
        if let Err(e) = renamed {
            let _ = self.target.parent.remove_file_member(&self.tmp);
            return Err(write_rejection(
                &e,
                "output file already exists",
                "cannot write output file",
            ));
        }
        Ok(InferBatchReport::new(
            self.lines,
            std::mem::take(&mut self.hash).finish(),
        ))
    }
}

impl BatchResults for LazyOut<'_> {
    /// 成功なら公開して要約を stdout へ出す（要約を返せなければ公開済みの OUT を best effort で
    /// 消す）。書き込み失敗なら一時ファイルを消して報告する。計算失敗の `ErrorReport` は出力済みで、
    /// OUT は作られていない。ウォッチドッグ解除前に呼ばれる（REQ-39）。
    fn finish(
        &mut self,
        mut out: &mut dyn Write,
        result: Result<ExitCode, WriteFailure>,
    ) -> io::Result<ExitCode> {
        match result {
            Ok(ExitCode::Ok) => {
                let report = match self.publish() {
                    Ok(report) => report,
                    Err(error) => return emit_error_report(&mut out, &error),
                };
                let written = write_stage_line(&mut out, report.to_json_line());
                if written.is_err() {
                    let _ = self
                        .target
                        .parent
                        .remove_file_member(Path::new(&self.target.leaf));
                }
                written
            }
            Ok(code) => Ok(code),
            // stdout の失敗は部分書き込みの可能性があり、追加の出力はしない（元のエラーを返す）。
            Err(WriteFailure::Stdout(error)) => {
                self.cleanup();
                Err(error)
            }
            Err(WriteFailure::Results(_)) => {
                let report = self.abort();
                emit_error_report(&mut out, &report)
            }
        }
    }
}

impl Write for LazyOut<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.file.is_none() {
            let file = self
                .target
                .parent
                .create_new_member(&self.tmp)
                .map_err(|e| {
                    self.create_error = Some(write_rejection(
                        &e,
                        "output file already exists",
                        "cannot write output file",
                    ));
                    io::Error::other("cannot create output file")
                })?;
            self.file = Some(file);
        }
        let Some(file) = self.file.as_mut() else {
            return Err(io::Error::other("output file is not open"));
        };
        let n = file.write(buf)?;
        let written = buf.get(..n).unwrap_or_default();
        self.hash.update(written);
        self.lines += written.iter().filter(|b| **b == b'\n').count();
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.as_mut().map_or(Ok(()), Write::flush)
    }
}

/// `kind_version` の許可リスト検査・ONNX の読み込み・出力サイズと選択肢数の一致を確認して
/// バックエンドを組み立てる（`infer` の `prepare` と、`package` の公開前検証が共有する。REQ-39）。
///
/// # Errors
/// 未許可の版・読み込めない ONNX・出力サイズの不一致は `invalid_input`（64）。
pub(crate) fn load_backend(
    onnx: &[u8],
    kind: ModelKind,
    kind_version: u32,
    n_options: usize,
) -> Result<OnnxBackend, ErrorReport> {
    if !kind_version_allowed(kind, kind_version) {
        return Err(invalid("unsupported kind_version"));
    }
    let backend =
        OnnxBackend::from_bytes(onnx, kind).map_err(|_| invalid("model file cannot be loaded"))?;
    if backend.n_classes() != n_options {
        return Err(invalid("model output size does not match definition"));
    }
    Ok(backend)
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
    // 再読込したメタデータの `kind` が、ガードが検査した `kind` と一致することを確認する
    // （2 回の読み込みの間に `artifact.json` を差し替えられても、検査した `kind` と別の `kind` で
    // バックエンドを組み立てない。REQ-39）。
    if meta.kind() != checked.kind.as_str() {
        return Err(invalid("artifact metadata does not match the model file"));
    }
    let onnx = checked.onnx.as_bytes();
    if meta.onnx_sha256() != Sha256Digest::of_bytes(onnx).to_hex() {
        return Err(invalid("package model does not match its recorded hash"));
    }
    // 語彙ファイル（あれば）も配布物の一部として、記録されたハッシュと形式を検証する（REQ-39）。
    let vocab_file = match checked.package.open_member(Path::new(VOCAB_FILE_NAME)) {
        Ok((file, real)) => Some((file, real.into_path_buf())),
        Err(PathRejection::Unresolvable { source, .. })
            if source.kind() == io::ErrorKind::NotFound =>
        {
            None
        }
        Err(e) => return Err(e.to_error_report()),
    };
    super::candidate_artifact::verify_vocab_file(
        &meta,
        vocab_file.as_ref().map(|(f, p)| (f, p.as_path())),
    )?;
    let option_ids = definition.options().iter().map(|c| c.id.as_str());
    if !meta.label_order().iter().map(String::as_str).eq(option_ids) {
        return Err(invalid("package label order does not match definition"));
    }
    let pipeline = build_pipeline(onnx, &meta, definition.options().len())?;
    Ok(Prepared {
        definition,
        pipeline,
    })
}

/// `max_bytes` の範囲検査・`kind` の解析・[`load_backend`] を行い、前処理と束ねた推論パイプラインを
/// 組み立てる（`infer` の `prepare` と、`package` の公開前検証・p95 計測が共有する。REQ-32・REQ-39）。
///
/// `package` は公開するのと同じ `onnx` のバイト列から組み立て、計測対象と配布物を一致させる。
///
/// # Errors
/// `max_bytes` が範囲外・未対応の `kind`・未許可の版・読み込めない ONNX・出力サイズの不一致は
/// `invalid_input`（64）。
pub(crate) fn build_pipeline(
    onnx: &[u8],
    meta: &ArtifactMeta,
    n_options: usize,
) -> Result<Pipeline, ErrorReport> {
    let max_bytes = usize::try_from(meta.max_bytes())
        .ok()
        .filter(|n| (MIN_MAX_BYTES..=MAX_MAX_BYTES).contains(n))
        .ok_or_else(|| invalid("package max_bytes is out of range"))?;
    let kind = ModelKind::parse(meta.kind()).map_err(|_| invalid("unsupported model kind"))?;
    let backend = load_backend(onnx, kind, meta.kind_version(), n_options)?;
    Ok(InferencePipeline::new(
        ByteEncodingPreprocessor::new(max_bytes),
        backend,
    ))
}
// 閉じ込めは Linux・macOS 前提のため、guard を使う他のテスト（e2e）と同じく unix に限る。
#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// テスト用の作業ディレクトリ（cwd 相当）。
    fn workdir(name: &str) -> std::path::PathBuf {
        let dir = std::fs::canonicalize(std::env::temp_dir())
            .expect("temp dir")
            .join(format!("fandhe-infer-out-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .expect("read_dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// REQ-33: 公開は名前替えで行い、OUT には書いた全体だけが現れ、一時名は残らない。
    #[test]
    fn req33_publish_renames_tmp_to_out_with_summary() {
        let cwd = workdir("ok");
        let target = OutTarget::preflight(&cwd, Path::new("out.jsonl")).expect("preflight");
        let mut sink = LazyOut::new(&target);
        sink.write_all(b"a\nb\n").expect("write");
        assert_eq!(entries(&cwd), vec![sink.tmp.to_string_lossy().into_owned()]);
        let report = sink.publish().expect("publish");
        assert_eq!(
            report.to_json_line().expect("json"),
            format!(
                r#"{{"step":"infer","status":"ok","count":2,"sha256":"{}"}}"#,
                Sha256Digest::of_bytes(b"a\nb\n").to_hex()
            )
        );
        assert_eq!(entries(&cwd), vec!["out.jsonl".to_string()]);
        assert_eq!(
            std::fs::read(cwd.join("out.jsonl")).expect("out"),
            b"a\nb\n"
        );
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-39・REQ-21: 書き込み後に OUT が先に作られていた場合は `invalid_input`（64）で、
    /// 先客の中身は変わらず、一時ファイルは消える。
    #[test]
    fn req39_publish_refuses_when_out_appeared_first() {
        let cwd = workdir("race");
        let target = OutTarget::preflight(&cwd, Path::new("out.jsonl")).expect("preflight");
        let mut sink = LazyOut::new(&target);
        sink.write_all(b"new\n").expect("write");
        std::fs::write(cwd.join("out.jsonl"), "first").expect("racer");
        let error = sink.publish().expect_err("must refuse");
        assert_eq!(error.code, ExitCode::InvalidInput);
        assert_eq!(std::fs::read(cwd.join("out.jsonl")).expect("out"), b"first");
        assert_eq!(entries(&cwd), vec!["out.jsonl".to_string()]);
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-21・REQ-39: 書き込み失敗後の `abort` は一時ファイルを消して `runtime_error`（70）。
    /// 何も書かなかった出力先は何も作らない。
    #[test]
    fn req21_abort_removes_tmp_and_reports_runtime_error() {
        let cwd = workdir("abort");
        let target = OutTarget::preflight(&cwd, Path::new("out.jsonl")).expect("preflight");
        let mut untouched = LazyOut::new(&target);
        assert_eq!(untouched.abort().code, ExitCode::RuntimeError);
        assert!(entries(&cwd).is_empty());
        let mut sink = LazyOut::new(&target);
        sink.write_all(b"partial").expect("write");
        let error = sink.abort();
        assert_eq!(error.code, ExitCode::RuntimeError);
        assert!(entries(&cwd).is_empty());
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-21・REQ-39: 一時名が既に塞がれていて作れない場合は、書き込みが失敗し `abort` が
    /// 作成失敗の報告（既存は 64）を返す。
    #[test]
    fn req39_tmp_name_collision_is_invalid_input() {
        let cwd = workdir("tmpcol");
        let target = OutTarget::preflight(&cwd, Path::new("out.jsonl")).expect("preflight");
        let mut sink = LazyOut::new(&target);
        std::fs::write(cwd.join(&sink.tmp), "squatter").expect("squat");
        assert!(sink.write_all(b"x").is_err());
        assert_eq!(sink.abort().code, ExitCode::InvalidInput);
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// 常に失敗し、書き込みの試行回数を数える stdout。
    struct FailingOut {
        attempts: usize,
    }

    impl Write for FailingOut {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            self.attempts += 1;
            Err(io::Error::other("stdout closed"))
        }
        fn flush(&mut self) -> io::Result<()> {
            self.attempts += 1;
            Err(io::Error::other("stdout closed"))
        }
    }

    /// REQ-21・REQ-33: stdout 側の失敗では追加の出力をせず元のエラーを返し、一時ファイルは消す。
    #[test]
    fn req33_stdout_failure_adds_no_output_and_removes_tmp() {
        let cwd = workdir("stdoutfail");
        let target = OutTarget::preflight(&cwd, Path::new("out.jsonl")).expect("preflight");
        let mut sink = LazyOut::new(&target);
        sink.write_all(b"a\n").expect("write");
        let mut stdout = FailingOut { attempts: 0 };
        let result = sink.finish(
            &mut stdout,
            Err(WriteFailure::Stdout(io::Error::other("stdout closed"))),
        );
        assert_eq!(
            result.expect_err("original error").to_string(),
            "stdout closed"
        );
        assert_eq!(stdout.attempts, 0);
        assert!(entries(&cwd).is_empty());
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-21: 要約の出力に失敗したら（公開後でも）追加の出力をせず、OUT も残さない。
    #[test]
    fn req33_summary_write_failure_removes_published_out() {
        let cwd = workdir("summaryfail");
        let target = OutTarget::preflight(&cwd, Path::new("out.jsonl")).expect("preflight");
        let mut sink = LazyOut::new(&target);
        sink.write_all(b"a\n").expect("write");
        let mut stdout = FailingOut { attempts: 0 };
        assert!(sink.finish(&mut stdout, Ok(ExitCode::Ok)).is_err());
        assert_eq!(stdout.attempts, 1);
        assert!(entries(&cwd).is_empty());
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-21: 結果ファイル側の失敗だけが `runtime_error` の `ErrorReport` を 1 つ出し、一時ファイルを消す。
    #[test]
    fn req21_results_failure_emits_one_runtime_error() {
        let cwd = workdir("resultsfail");
        let target = OutTarget::preflight(&cwd, Path::new("out.jsonl")).expect("preflight");
        let mut sink = LazyOut::new(&target);
        sink.write_all(b"a\n").expect("write");
        let mut stdout: Vec<u8> = Vec::new();
        let code = sink
            .finish(
                &mut stdout,
                Err(WriteFailure::Results(io::Error::other("disk full"))),
            )
            .expect("report written");
        assert_eq!(code, ExitCode::RuntimeError);
        let text = String::from_utf8(stdout).expect("utf8");
        assert_eq!(text.matches('\n').count(), 1);
        assert!(text.starts_with("{\"code\":\"runtime_error\""));
        assert!(entries(&cwd).is_empty());
        let _ = std::fs::remove_dir_all(&cwd);
    }
}
