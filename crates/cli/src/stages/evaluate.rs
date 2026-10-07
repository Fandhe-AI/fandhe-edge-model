//! `evaluate` 工程（REQ-17・REQ-21・REQ-24・REQ-27・REQ-33・TASK-33.1-2・#136・#314）。
//!
//! # 手順
//!
//! 最初に評価データの状態を判定する（[`load_frozen_evaluation`]。候補が学習済みかの確認より前。
//! #140 の申し送り）。
//!
//! - 評価データ未定義: `status:"skipped"`・exit 0（評価済みを装わない。REQ-17）
//! - 評価データあり: 凍結記録とのハッシュ一致を確認し（不一致は `invalid_input`。fail-closed）、
//!   評価器（`fandhe-edge-eval`）へ接続して凍結データへ **1 回だけ** 適用する
//!   （[`fandhe_edge_eval::final_test_once::apply_once`]）。指標（正解率・Macro-F1）は評価器の
//!   [`fandhe_edge_eval::metrics::evaluate_single_select`] で求め、CLI では再実装しない
//!
//! # 評価の独立性（REQ-27）
//!
//! - 推論関数（`predict`）へ渡すのは `input` の列だけ（正解ラベルは評価器側に残る）。評価データの
//!   結果は `select` に使わない（選定は validation のみ）
//! - 評価の前後でモデル（重み）と評価データのハッシュが一致することは `apply_once` が検証する
//! - 最終 test の台帳は `final_test_ledger/` 1 つで、**最初の `evaluate` がその時点の学習済み候補を
//!   すべて事前登録する**（代表構成 ID は `"<candidate_id>:seed<seed>"`。暫定・オーナー確認事項）。
//!   そのため最初の `evaluate` の後に学習した候補は `invalid_input`
//!   （`candidate is not registered for evaluation`）になる。**全候補を学習し、`select` で選定してから選定候補だけを評価すること**
//! - `train --smoke` の候補は評価できない（検証専用モデルが本番候補の構成ロックを使い切らないため）
//!
//! # 適用権を消費した後はやり直せない
//!
//! `apply_once` がロックを取った後の失敗（推論の時間超過・記録の書き込み失敗など）では、その候補は
//! 二度と評価できない（評価器の fail-closed の契約）。そのため適用権が要らない失敗しうる処理
//! （成果物の検証・バックエンドの試し組み立て・評価データの事前検査・下限基準 majority と必要件数の確定・
//! 台帳の用意と事前登録・記録ファイルの不在確認）はすべて `apply_once` の前に済ませる。
//!
//! # 推論失敗の扱い
//!
//! 1 件でも推論または判定への変換に失敗したら、評価全体を失敗（`runtime_error` / `limit_exceeded`）
//! とし、完了記録を書かない（成功 JSON・配布許可を作らない。REQ-27）。適用権は消費済みのため、
//! その候補は再評価できない（fail-closed）。
//!
//! # 評価完了の記録
//!
//! 成功時は `candidates/<N>/evaluation_record.json`（[`EvaluationRecord`]）を新規に書く。`package` は
//! 記録とモデル・`artifact.json`・評価データ・定義のハッシュの一致と、最終 test の台帳での適用完了を確認してから公開する。
//! 記録ファイル自体はプロジェクトに書き込める主体なら作り直せる（外部台帳は #168・TASK-39.3-2。
//! 本工程は検証済みとしない）。評価器のモデル・評価データ入力はパスを受け取り
//! `O_NOFOLLOW` の成分走査をしないため、閉じ込めつきで読み検証したバイト列（評価データ・重み）を
//! 私用の一時ディレクトリ（0700）へ複製し、そのパスだけを評価器へ渡す（プロジェクト内のパスを
//! 渡さない。REQ-39）。最終 test の台帳は、保持 fd 起点で開いた台帳ディレクトリを
//! `LedgerDir` の fd 実装（[`crate::held_ledger_dir::HeldLedgerDir`]。[`super::ledger::HeldLedger`]）
//! 越しに評価器へ渡し、台帳の読み書き・一覧・永続化はすべて保持 fd 起点の `openat` で行う
//! （パスへ戻る経路なし。検証後の差し替えでプロジェクト外を読み書きしない。#168）。
//!
//! 指標の算出・結果の検証・記録ファイルの書き込みは、台帳へ完了を記録する前
//! （[`fandhe_edge_eval::final_test_once::apply_once_then`] の `finish`）に行う。これらが失敗しても
//! 台帳は完了状態にならず（適用権のロックのみが残る）、記録の無い完了を作らない（REQ-27）。
//!
//! # 1 件ごとの予測の保存（REQ-27・REQ-41・#445）
//!
//! 評価記録と同じ候補ディレクトリへ `evaluation_predictions.jsonl`（評価データの行順。
//! `{id,status,predicted_label,scores}`。[`crate::prediction_lines`]）を新規に書く（既存なら適用権を取る前に
//! `invalid_input`）。PoC-26 の採点入口（`fandhe-edge-score`）が、他候補の予測と同じ形式で読む。
//! stdout の JSON・[`EvaluationRecord`] のスキーマは変えない。書き込みは評価記録と同じ位置
//! （台帳への完了記録の前）で行い、失敗時の扱いも同じ。
//!
//! # 選定との順序（REQ-27）
//!
//! `select` の記録（`selection_record.json`）が無い・再計算と不一致・対象が選定候補でない場合は
//! `invalid_input`。最終 test の結果を見てから候補を選べない。
//!
//! # 未接続（実装済みを装わない）
//!
//! Wilson 区間・診断レポート（REQ-29）・校正と棄権（REQ-22）は結線していない。
//! 結果 JSON は正解率と Macro-F1 のみ（スキーマは 2026-09-30 オーナー承認済み）。
//!
//! # 下限基準との比較（REQ-25・REQ-27・#339）
//!
//! 定義に `baseline_comparison`（事前登録した仮定）があるときだけ、majority との McNemar 比較
//! （Holm の族の大きさは 1 で恒等）の結果を評価記録の `baseline_comparison` へ残す（出力 JSON は変えない）。
//! majority は train 分割のラベルだけから作り、必要件数は定義の仮定から求める。どちらも適用権を取る前に
//! 確定する（[`super::baseline::prepare_baseline`]）。欄が無い定義では比較しない。

use std::cell::RefCell;
use std::path::Path;
use std::time::Instant;

use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::evaluation_record::{EvaluationRecord, MAX_EVALUATION_RECORD_BYTES};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::read_bounded;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::limits::INFER_TIME_LIMIT;
use fandhe_edge_core::stage_report::{EvaluateCompletedReport, EvaluateReport};
use fandhe_edge_data::eval_freeze::{EvalDataState, FreezeRecord};
use fandhe_edge_eval::eval_data_invariance::FrozenEvalData;
use fandhe_edge_eval::final_test_once::{
    AcquireError, AppliedOnce, DecodeFailed, FinalTestLedger, LabeledInput, RegisteredConfig,
    RepresentativeConfigId, apply_once_then,
};
use fandhe_edge_eval::invariance::ModelPackagePaths;
use fandhe_edge_eval::metrics::{self, EvalRecord, Outcome};
use fandhe_edge_guard::format::{FormatAllowlist, check_bytes};
use fandhe_edge_runtime::onnx::{MAX_MODEL_FILE_BYTES, ModelKind};
use fandhe_edge_runtime::pipeline::{
    BackendError, InferError, InferencePipeline, MAX_INFER_BATCH_DURATION, MAX_INFER_BATCH_LEN,
};
use fandhe_edge_runtime::preprocess::ByteEncodingPreprocessor;
use fandhe_edge_train::request::TrainRequest;
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::stage_files::SelectionRecord;

use crate::args::EvaluateArgs;
use crate::error_report::{
    EvalPredictFailure, ToErrorReport, acquire_error_report, apply_once_error_report,
};
use crate::infer_batch::judgment_from_prediction;
use crate::prediction_lines::prediction_line;
use crate::project::{
    EVALUATION_PREDICTIONS_FILE, EVALUATION_RECORD_FILE, Project, SELECTION_FILE, fail,
    inspect_bytes, invalid, runtime,
};
use crate::stage_output::{EvaluateStart, evaluate_start};

use super::baseline::{PreparedBaseline, compare, prepare_baseline};
use super::candidate_artifact::{
    CandidateArtifact, check_meta_consistency, load_candidate_artifact,
};
use super::infer::load_backend;
use super::inspect::load_frozen_evaluation;
use super::ledger::HeldLedger;
use super::select::compute_selection;
use super::train::{
    candidate_rel, effective_train_seed, load_trained, request_is_smoke_trained,
    request_matches_candidate, resolve_candidates, verified_split,
};

/// `evaluate` の成功結果（stdout の JSON 1 つへ写す）。
#[derive(Debug)]
pub enum EvaluateOutcome {
    /// 評価データ未定義（exit 0。評価済みを装わない）。
    Skipped(EvaluateReport),
    /// 凍結した評価データへの適用が完了した。
    Completed(EvaluateCompletedReport),
}

/// 推論に使う、事前検証済みの候補（評価する 1 候補分）。
struct PreparedCandidate {
    artifact: CandidateArtifact,
    kind: ModelKind,
    max_bytes: usize,
    config_id: RepresentativeConfigId,
    candidate_id: String,
    kind_name: String,
}

/// `evaluate` を実行する。
///
/// # Errors
/// 凍結記録との不一致・候補が未学習 / smoke 学習・既に評価済み・未登録は `invalid_input`（64）、
/// 推論・読み込みの上限超過は `limit_exceeded`（20）、I/O 失敗は `runtime_error`（70）。
/// 適用権の消費後の失敗では、その候補は再評価できない（モジュール doc）。
pub fn run(args: &EvaluateArgs, cwd: &Path) -> Result<EvaluateOutcome, ErrorReport> {
    let project = Project::open(cwd, &args.project_dir)?;
    // 評価データの有無・凍結記録とのハッシュ一致を先に判定する。
    let Some((freeze, eval_bytes)) = load_frozen_evaluation(&project)? else {
        return match evaluate_start(&EvalDataState::NotProvided, b"")? {
            EvaluateStart::Skipped(report) => Ok(EvaluateOutcome::Skipped(report)),
            EvaluateStart::Proceed(_) => Err(runtime("unexpected evaluation state")),
        };
    };
    let definition = project.load_definition()?;
    // 最終 test の結果を見て候補を選び直せないよう、validation による選定（`select`）を先に確定させ、
    // 選定された候補だけを評価可能にする（REQ-27）。
    let selection_sha256 = ensure_selected(&project, &definition, args.candidate)?;
    let records = project.load_records(&definition)?;
    let (split, seed) = verified_split(&project, &records)?;

    let mut prepared_target: Option<PreparedCandidate> = None;
    let mut entries: Vec<RegisteredConfig> = Vec::new();
    let n_candidates = resolve_candidates(&project, &definition, args.candidate, seed)?.len();
    if args.candidate >= n_candidates {
        return Err(invalid("candidate index is out of range"));
    }
    // 対象の候補を先に検証し（固有の message を返す）、続いて全候補を事前登録用に検証する。
    for index in
        std::iter::once(args.candidate).chain((0..n_candidates).filter(|i| *i != args.candidate))
    {
        let is_target = index == args.candidate;
        let Some(prepared) = prepare_candidate(
            &project,
            &definition,
            &records,
            &split,
            seed,
            index,
            is_target,
        )?
        else {
            continue;
        };
        entries.push(RegisteredConfig::new(
            prepared.config_id.clone(),
            Sha256Digest::of_bytes(&prepared.artifact.onnx_bytes),
        ));
        if is_target {
            prepared_target = Some(prepared);
        }
    }
    let target = prepared_target.ok_or_else(|| invalid("candidate is not trained"))?;

    // ロックを取る前に弾けるものはすべてここで弾く（適用権は消費したら戻らない）。
    let record_rel = candidate_rel(args.candidate).join(EVALUATION_RECORD_FILE);
    if project.exists(&record_rel)? {
        return Err(invalid(
            "candidate has already been evaluated on the frozen data",
        ));
    }
    // バックエンドを試しに組み立てて、読めない ONNX で適用権を使わない。
    check_bytes(
        target.artifact.onnx_bytes.clone(),
        &FormatAllowlist::onnx_only(),
    )
    .map_err(|e| e.to_error_report())?;
    load_backend(
        &target.artifact.onnx_bytes,
        target.kind,
        target.artifact.meta.kind_version(),
        definition.options().len(),
    )?;
    // 評価データも事前に検査する（ロック取得後の分解失敗で適用権を失わない）。
    let decoded = decode_evaluation(&eval_bytes, &definition).map_err(|e| e.to_error_report())?;
    // 1 件ごとの予測の保存用に id を控える（`decode_evaluation` と同じ検査・同じ行順。推論側には渡さない）。
    let eval_ids: Vec<String> = inspect_bytes(&eval_bytes, &definition)?
        .into_iter()
        .map(|r| r.id)
        .collect();
    // 予測行の件数照合は適用権を取る前に済ませる（保存の失敗で適用権を失わない）。
    if eval_ids.len() != decoded.len() {
        return Err(runtime("evaluation record count mismatch"));
    }
    // 推論ごとのスコア（保存用。推論関数の戻り値の型は変えず、横で受ける）。
    let scores_log: RefCell<Vec<Vec<f64>>> = RefCell::new(Vec::new());
    // 下限基準（majority）と必要件数は適用権を取る前に確定する（失敗しても適用権を使い切らない）。
    // 引数は train 側の入力だけで、評価データを渡せない（REQ-27・#339）。
    let baseline = prepare_baseline(&definition, &records, &split)?;
    let definition_sha256 = definition
        .canonical_hash()
        .map_err(|_| runtime("cannot hash definition"))?
        .to_hex();
    let onnx_digest = Sha256Digest::of_bytes(&target.artifact.onnx_bytes);

    // 台帳は保持 fd 起点で開き、以降の操作もすべて fd 相対で行う（REQ-39。[`HeldLedger`]）。
    let held_ledger = HeldLedger::open(&project, true)?
        .ok_or_else(|| runtime("cannot open final test ledger"))?;
    match held_ledger
        .ledger()
        .register_configs(&freeze.sha256(), &entries)
    {
        Ok(()) | Err(AcquireError::AlreadyRegistered { .. }) => {}
        Err(e) => return Err(acquire_error_report(&e)),
    }
    // 最初の適用の前に、選定（候補 ID と、validation 結果を含む選定記録のダイジェスト）を台帳へ
    // 固定する。以後は同じ選定のときだけ評価でき、A を評価した後に validation 結果と選定記録を
    // 書き換えて B を選んでも、B は最終 test に適用できない（REQ-27）。
    held_ledger
        .ledger()
        .pin_selection(&freeze.sha256(), &target.config_id, &selection_sha256)
        .map_err(|e| acquire_error_report(&e))?;
    // 1 件ごとの予測ファイルも、適用権を取る前に不在を確認する（書けないことで適用権を失わない）。
    // 台帳の選定固定より後に置く（選定不一致の拒否を優先する既存の挙動を変えない）。
    let predictions_rel = candidate_rel(args.candidate).join(EVALUATION_PREDICTIONS_FILE);
    if project.exists(&predictions_rel)? {
        return Err(invalid(
            "candidate has already been evaluated on the frozen data",
        ));
    }

    // 指標の算出・評価記録の書き込みは、台帳へ完了を記録する前（`finish`）に済ませる。失敗しても
    // 台帳は完了状態にならず、記録の無い完了（台帳だけが完了）を作らない（REQ-27）。
    let mut finish_error: Option<ErrorReport> = None;
    let finish = |applied: &AppliedOnce<Vec<Outcome>>| -> Result<EvaluateCompletedReport, EvalPredictFailure> {
        finalize_evaluation(
            &project,
            &definition,
            &freeze,
            &target,
            args.candidate,
            &record_rel,
            &PredictionsSink {
                rel: &predictions_rel,
                ids: &eval_ids,
                scores: &scores_log.borrow(),
            },
            definition_sha256,
            onnx_digest,
            baseline.as_ref(),
            applied,
        )
        .map_err(|e| {
            finish_error = Some(e);
            EvalPredictFailure::Failed
        })
    };
    let (_applied, report) = match apply_to_frozen_data(
        held_ledger.ledger(),
        &freeze,
        &definition,
        &eval_bytes,
        &target,
        onnx_digest,
        &scores_log,
        finish,
    ) {
        Ok(v) => v,
        Err(report) => return Err(finish_error.take().unwrap_or(report)),
    };

    Ok(EvaluateOutcome::Completed(report))
}

/// `select` の記録があり、保存済みの結果から再計算した選定と一致し、対象候補が選定された候補で
/// あることを確認する（`package` と同じ再検証。選定ロジックを複製しない。REQ-27）。
///
/// 成功時は選定記録のバイト列の sha256 を返す（台帳へ固定する選定のダイジェスト）。
///
/// # Errors
/// 選定記録が無い・不正・再計算と不一致・対象候補が選定された候補でない場合は `invalid_input`。
fn ensure_selected(
    project: &Project,
    definition: &Definition,
    candidate: usize,
) -> Result<Sha256Digest, ErrorReport> {
    let Some(bytes) = project.read_optional(SELECTION_FILE, 64 * 1024)? else {
        return Err(invalid("candidate selection has not been recorded"));
    };
    let selection = SelectionRecord::from_json_slice(&bytes)
        .map_err(|_| invalid("selection record is invalid"))?;
    if compute_selection(project, definition)?.as_ref() != Some(&selection) {
        return Err(invalid("selection record does not match the candidate"));
    }
    if selection.candidate_index != candidate {
        return Err(invalid("candidate is not the selected candidate"));
    }
    // 選定記録には選定の入力になった validation 結果（正解数・件数）が含まれる。
    Ok(Sha256Digest::of_bytes(&bytes))
}

/// 候補 `index` の学習結果を検証して、評価に使える形にする（評価できない候補は `None`）。
///
/// 対象候補（`is_target`）は、未学習・smoke 学習・リクエスト不一致を `invalid_input` で拒否する。
/// 対象以外の候補は、未学習・smoke 学習・失敗結果なら登録しないだけ（`None`）で、保存済みの結果が
/// 壊れている（リクエスト・成果物の不一致）場合は停止する（fail-closed）。
fn prepare_candidate(
    project: &Project,
    definition: &Definition,
    records: &[fandhe_edge_data::inspect::ValidRecord],
    split: &fandhe_edge_data::split::SplitResult,
    seed: u32,
    index: usize,
    is_target: bool,
) -> Result<Option<PreparedCandidate>, ErrorReport> {
    let seed = effective_train_seed(project, index, seed)?;
    let candidates = resolve_candidates(project, definition, index, seed)?;
    let candidate = candidates
        .get(index)
        .ok_or_else(|| invalid("candidate index is out of range"))?;
    let loaded = load_trained(project, index)?;
    let Some((request, TrainOutcome::Ok(success))) = loaded else {
        return if is_target {
            Err(invalid("candidate is not trained"))
        } else {
            Ok(None)
        };
    };
    if !request_matches_candidate(&request, &candidate.params, records, split) {
        return Err(invalid("train request does not match the candidate"));
    }
    if request_is_smoke_trained(&request, &candidate.params) {
        return if is_target {
            Err(invalid("smoke-trained candidate cannot be evaluated"))
        } else {
            Ok(None)
        };
    }
    let artifact = load_candidate_artifact(project, index, &success)?;
    check_candidate_artifact(&artifact, definition, &request)?;
    let kind =
        ModelKind::parse(artifact.meta.kind()).map_err(|_| invalid("unsupported model kind"))?;
    let max_bytes = usize::try_from(artifact.meta.max_bytes())
        .map_err(|_| invalid("package max_bytes is out of range"))?;
    let config_id =
        RepresentativeConfigId::parse(&format!("{}:seed{}", candidate.candidate_id, seed))
            .map_err(|e| acquire_error_report(&e))?;
    Ok(Some(PreparedCandidate {
        kind_name: artifact.meta.kind().to_string(),
        artifact,
        kind,
        max_bytes,
        config_id,
        candidate_id: candidate.candidate_id.clone(),
    }))
}

/// 成果物のメタデータを、学習リクエストと定義に照合する（`package` と同じ検査）。
fn check_candidate_artifact(
    artifact: &CandidateArtifact,
    definition: &Definition,
    request: &TrainRequest,
) -> Result<(), ErrorReport> {
    check_meta_consistency(
        &artifact.meta,
        definition,
        request.kind(),
        request.kind_version(),
        request.max_bytes(),
    )
}

/// 評価データの分解の失敗（本文・行番号を含まない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalDecodeError {
    /// 検査で異常があった・0 件。
    Invalid,
    /// 件数が上限を超えた。
    TooMany,
}

impl ToErrorReport for EvalDecodeError {
    fn to_error_report(&self) -> ErrorReport {
        match self {
            EvalDecodeError::Invalid => invalid("evaluation data is invalid"),
            EvalDecodeError::TooMany => fail(
                ExitCode::LimitExceeded,
                "evaluation data has too many records",
            ),
        }
    }
}

/// 照合済みの評価データのバイト列を、`input` と正解ラベルへ分ける。
///
/// 件数は推論バッチの上限（[`MAX_INFER_BATCH_LEN`]）までとし、0 件は拒否する
/// （`evaluate_single_select` が適用後に失敗しないよう、適用前の事前検査にも使う。REQ-39）。
fn decode_evaluation(
    bytes: &[u8],
    definition: &Definition,
) -> Result<Vec<LabeledInput>, EvalDecodeError> {
    let records = inspect_bytes(bytes, definition).map_err(|_| EvalDecodeError::Invalid)?;
    if records.is_empty() {
        return Err(EvalDecodeError::Invalid);
    }
    if records.len() > MAX_INFER_BATCH_LEN {
        return Err(EvalDecodeError::TooMany);
    }
    Ok(records
        .into_iter()
        .map(|r| LabeledInput {
            input: r.input,
            gold: r.label_id,
        })
        .collect())
}

/// 評価結果を確定する（指標の算出・完了報告の構築・評価記録の書き込み。台帳への完了記録の前に呼ぶ）。
///
/// 指標は評価器（`fandhe-edge-eval`）で求め、CLI では再実装しない（REQ-24）。記録の書き込みは最後に行い、
/// それ以前の失敗では記録を残さない。
#[allow(clippy::too_many_arguments)]
fn finalize_evaluation(
    project: &Project,
    definition: &Definition,
    freeze: &FreezeRecord,
    target: &PreparedCandidate,
    candidate: usize,
    record_rel: &Path,
    predictions: &PredictionsSink<'_>,
    definition_sha256: String,
    onnx_digest: Sha256Digest,
    baseline: Option<&PreparedBaseline>,
    applied: &AppliedOnce<Vec<Outcome>>,
) -> Result<EvaluateCompletedReport, ErrorReport> {
    let labels: Vec<&str> = definition.options().iter().map(|c| c.id.as_str()).collect();
    let eval_records: Vec<EvalRecord<'_>> = applied
        .golds
        .iter()
        .zip(&applied.output)
        .map(|(gold, outcome)| EvalRecord { gold, outcome })
        .collect();
    let computed = metrics::evaluate_single_select(&labels, &eval_records)
        .map_err(|_| runtime("cannot compute evaluation metrics"))?;
    let correct = computed.accuracy.overall.numerator();
    let total = computed.accuracy.overall.denominator();
    // 下限基準との比較（定義に `baseline_comparison` があるときだけ。出力 JSON には出さない。#339）。
    let baseline_comparison = match baseline {
        None => None,
        Some(prepared) => {
            let (record, candidate_correct) =
                compare(prepared, &labels, &applied.golds, &applied.output)?;
            // 正誤の規則を 2 重化した結果のずれは、記録する前に止める。
            if candidate_correct != correct {
                return Err(runtime("baseline comparison disagrees with metrics"));
            }
            Some(record)
        }
    };
    let report = EvaluateCompletedReport::completed(
        candidate,
        target.kind_name.clone(),
        correct,
        total,
        computed.macro_f1.value(),
    )
    .ok_or_else(|| runtime("cannot build evaluation report"))?;

    let record = EvaluationRecord {
        candidate_index: candidate,
        candidate_id: target.candidate_id.clone(),
        config_id: target.config_id.as_str().to_string(),
        evaluation_sha256: freeze.sha256().to_hex(),
        evaluation_bytes: freeze.byte_len(),
        onnx_sha256: onnx_digest.to_hex(),
        artifact_meta_sha256: Sha256Digest::of_bytes(&target.artifact.meta_bytes).to_hex(),
        definition_sha256,
        correct,
        total,
        baseline_comparison,
    };
    let record_json = record
        .to_json_vec()
        .map_err(|_| runtime("cannot serialize evaluation record"))?;
    // `package` は記録を `MAX_EVALUATION_RECORD_BYTES` までしか読まない。読めない記録を書かない。
    if u64::try_from(record_json.len()).map_or(true, |n| n > MAX_EVALUATION_RECORD_BYTES) {
        return Err(runtime("evaluation record is too large"));
    }
    // 1 件ごとの予測を先に書く（記録があるのに予測が無い状態を作らない）。失敗は記録の失敗と同じ扱い。
    project.write_new(
        predictions.rel,
        predictions
            .to_jsonl(&labels, &applied.output)
            .ok_or_else(|| runtime("cannot build evaluation predictions"))?
            .as_bytes(),
    )?;
    // 書いた後は読み取り専用にする（凍結データの配置と同じ扱い。改ざんの抑止であり、
    // 記録の封印は外部台帳〔#168〕の範囲）。
    project.set_read_only(predictions.rel)?;
    project.write_new(record_rel, &record_json)?;
    Ok(report)
}

/// 1 件ごとの予測の保存に必要な材料（書き込み先・評価データの id 列・推論ごとのスコア）。
///
/// 行形式は [`crate::prediction_lines`]。stdout の JSON・[`EvaluationRecord`] のスキーマには影響しない
/// （REQ-27。PoC-26 の採点入口 `fandhe-edge-score` が読む。REQ-41・#445）。
struct PredictionsSink<'a> {
    rel: &'a Path,
    ids: &'a [String],
    scores: &'a [Vec<f64>],
}

impl PredictionsSink<'_> {
    /// 評価データの行順の JSONL を作る。id・予測の件数が合わなければ `None`。
    fn to_jsonl(&self, labels: &[&str], outcomes: &[Outcome]) -> Option<String> {
        if self.ids.len() != outcomes.len() {
            return None;
        }
        let mut out = String::new();
        for (i, (id, outcome)) in self.ids.iter().zip(outcomes).enumerate() {
            let scores = self.scores.get(i).map(|s| (labels, s.as_slice()));
            out.push_str(&prediction_line(id, outcome, scores));
            out.push('\n');
        }
        Some(out)
    }
}

/// 台帳で適用権を取り、凍結した評価データへ 1 回だけ推論を当てる。
#[allow(clippy::too_many_arguments)]
fn apply_to_frozen_data<R>(
    ledger: &FinalTestLedger,
    freeze: &FreezeRecord,
    definition: &Definition,
    eval_bytes: &[u8],
    target: &PreparedCandidate,
    onnx_digest: Sha256Digest,
    scores_log: &RefCell<Vec<Vec<f64>>>,
    finish: impl FnOnce(&AppliedOnce<Vec<Outcome>>) -> Result<R, EvalPredictFailure>,
) -> Result<(AppliedOnce<Vec<Outcome>>, R), ErrorReport> {
    // 評価器はパスから開き直すため、閉じ込めつきで読み検証済みのバイト列を、この実行だけの
    // 私用ディレクトリ（0700・新規作成）へ複製し、そのパスを渡す。プロジェクト内のパスを渡すと、
    // 検証後に symlink へ差し替えられてプロジェクト外を読みうる（REQ-39）。
    let staged = StagedFiles::create()?;
    let eval_path = staged.write("evaluation.jsonl", eval_bytes)?;
    let weights_path = staged.write("model.onnx", &target.artifact.onnx_bytes)?;
    let frozen = FrozenEvalData {
        path: &eval_path,
        sha256: freeze.sha256(),
        byte_len: freeze.byte_len(),
    };
    let model = ModelPackagePaths {
        weights: &weights_path,
        vocab: None,
        calibration: None,
        thresholds: None,
    };
    let options = definition.options();
    apply_once_then(
        ledger,
        &frozen,
        target.config_id.clone(),
        &model,
        |bytes| decode_evaluation(bytes, definition).map_err(|_| DecodeFailed),
        // 推論関数へは `input` の列だけが渡る（正解ラベルは評価器側に残る。REQ-27）。
        |_ticket, inputs, paths| {
            // 評価器のパスベースの読み込みと、閉じ込めつきで検証した重みが同一であることを確認する。
            let weights = read_bounded(paths.weights, MAX_MODEL_FILE_BYTES)
                .map_err(|_| EvalPredictFailure::Failed)?;
            if Sha256Digest::of_bytes(&weights) != onnx_digest {
                return Err(EvalPredictFailure::ModelChanged);
            }
            let backend = load_backend(
                &weights,
                target.kind,
                target.artifact.meta.kind_version(),
                options.len(),
            )
            .map_err(|_| EvalPredictFailure::Failed)?;
            let pipeline =
                InferencePipeline::new(ByteEncodingPreprocessor::new(target.max_bytes), backend);
            let deadline = Instant::now().checked_add(MAX_INFER_BATCH_DURATION);
            let mut outcomes = Vec::with_capacity(inputs.len());
            for input in inputs {
                if deadline.is_some_and(|d| Instant::now() > d) {
                    return Err(EvalPredictFailure::TimeLimit);
                }
                // 推論・判定への変換の失敗は 1 件ごとの `Outcome::Error` にせず、評価全体の失敗として
                // 伝える。`Ok` で返すと、全件が失敗しても完了記録と配布許可が作られてしまう（REQ-27）。
                // 失敗時は評価器が成功を記録しないため、`package` は配布を許さない。
                let prediction = match pipeline.infer_one_within(input, INFER_TIME_LIMIT) {
                    Ok(prediction) => prediction,
                    Err(InferError::Backend(BackendError::TimeLimitExceeded)) => {
                        return Err(EvalPredictFailure::TimeLimit);
                    }
                    Err(_) => return Err(EvalPredictFailure::Failed),
                };
                let judgment = judgment_from_prediction(options, PLACEHOLDER_ID, &prediction)
                    .map_err(|_| EvalPredictFailure::Failed)?;
                outcomes.push(Outcome::Label(judgment.predicted_choice_id().to_string()));
                scores_log.borrow_mut().push(prediction.scores().to_vec());
            }
            Ok(outcomes)
        },
        finish,
    )
    .map_err(|e| apply_once_error_report(&e))
}

/// 評価器へ渡すファイルの私用置き場（OS の一時ディレクトリ配下に 0700 で新規作成し、破棄時に消す）。
///
/// 名前は重複しない限り新規作成のみ（既存なら作り直しを試す）で、他者が事前に用意した
/// ディレクトリを使わない。
struct StagedFiles {
    dir: std::path::PathBuf,
}

impl StagedFiles {
    fn create() -> Result<Self, ErrorReport> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir();
        for _ in 0..16 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let name = format!(
                "fandhe-edge-eval-{}-{}-{}",
                std::process::id(),
                nanos,
                COUNTER.fetch_add(1, Ordering::Relaxed)
            );
            let dir = base.join(name);
            #[cfg_attr(not(unix), allow(unused_mut))]
            let mut builder = std::fs::DirBuilder::new();
            #[cfg(unix)]
            std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
            match builder.create(&dir) {
                Ok(()) => return Ok(Self { dir }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(_) => return Err(runtime("cannot create staging directory")),
            }
        }
        Err(runtime("cannot create staging directory"))
    }

    fn write(&self, name: &str, bytes: &[u8]) -> Result<std::path::PathBuf, ErrorReport> {
        use std::io::Write;
        let path = self.dir.join(name);
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options
            .open(&path)
            .map_err(|_| runtime("cannot stage evaluation file"))?;
        file.write_all(bytes)
            .and_then(|()| file.flush())
            .map_err(|_| runtime("cannot stage evaluation file"))?;
        Ok(path)
    }
}

impl Drop for StagedFiles {
    fn drop(&mut self) {
        // best effort（自分が作った私用ディレクトリだけを消す）。
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// 判定型の `id`（評価ではレコード ID を推論側へ渡さないため固定の占位値を使う。REQ-27）。
const PLACEHOLDER_ID: &str = "evaluation";

#[cfg(test)]
mod tests {
    use super::*;

    const DEFINITION: &str = r#"{"schema":"fandhe-edge-model-definition/v1","name":"t","version":1,"judgment_type":"single_select","options":[{"id":"a","display_name":"a","description":"d"},{"id":"b","display_name":"b","description":"d"}],"io":{"input":"bytes"}}"#;

    fn definition() -> Definition {
        Definition::parse(DEFINITION).expect("definition")
    }

    /// REQ-27: 評価データは `input` と正解ラベルへ分かれ、レコード順が保たれる。
    #[test]
    fn req27_decode_splits_input_and_gold_in_order() {
        let data = b"{\"id\":\"1\",\"input\":\"x\",\"output\":{\"intent\":\"b\"}}\n{\"id\":\"2\",\"input\":\"y\",\"output\":{\"intent\":\"a\"}}\n";
        let decoded = decode_evaluation(data, &definition()).expect("decode");
        assert_eq!(
            decoded,
            vec![
                LabeledInput {
                    input: "x".to_string(),
                    gold: "b".to_string()
                },
                LabeledInput {
                    input: "y".to_string(),
                    gold: "a".to_string()
                },
            ]
        );
    }

    /// REQ-27・REQ-39: 空・異常なデータは適用前に拒否する（固定 message。本文を含まない）。
    #[test]
    fn req39_decode_rejects_empty_and_invalid_data() {
        assert_eq!(
            decode_evaluation(b"", &definition()),
            Err(EvalDecodeError::Invalid)
        );
        assert_eq!(
            decode_evaluation(b"not json\n", &definition()),
            Err(EvalDecodeError::Invalid)
        );
        let report = EvalDecodeError::Invalid.to_error_report();
        assert_eq!(report.code, ExitCode::InvalidInput);
        assert_eq!(report.message, "evaluation data is invalid");
        let report = EvalDecodeError::TooMany.to_error_report();
        assert_eq!(report.code, ExitCode::LimitExceeded);
        assert_eq!(report.message, "evaluation data has too many records");
    }
}
