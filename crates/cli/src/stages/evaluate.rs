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
//! `{id,status,predicted_label,scores}`。[`PredictionLine`]）を新規に書く（既存なら適用権を取る前に
//! `invalid_input`）。PoC-26 の採点入口（`fandhe-edge-score`）が、他候補の予測と同じ形式で読む。
//! 予測ファイルは stdout の JSON・[`EvaluationRecord`] のスキーマに影響しない。書き込みは評価記録と同じ位置
//! （台帳への完了記録の前）で行い、失敗時の扱いも同じ。
//!
//! # 選定との順序（REQ-27）
//!
//! `select` の記録（`selection_record.json`）が無い・再計算と不一致・対象が選定候補でない場合は
//! `invalid_input`。最終 test の結果を見てから候補を選べない。
//!
//! # 正解率の Wilson 区間
//!
//! 正解率の Wilson 区間は評価記録・結果 JSON には含めず、合否基準の照合で `package` 工程が使う
//! （#328。再現性の seed ごとの区間だけは `reproducibility` に出す。#490）。
//!
//! # 診断レポート（REQ-29・TASK-29.1〜29.3・#492）
//!
//! stdout の末尾の `diagnostics` に、train 分割と凍結評価データの基礎統計・混同しやすい組（上位 10）・
//! ラベル数の変化の注記（`--previous-project-dir` のときだけ）・データ量水準を出す（[`super::diagnostics`]）。
//! 基礎統計は適用権を取る前に求め、混同しやすい組は同じ 1 回の適用の混同行列から求める（REQ-27）。
//! 評価記録には入れず、終了コード・合否・`package` の照合に使わない。
//!
//! # 校正（REQ-22・REQ-27・#477）
//!
//! 温度 T・保留しきい値 τ は **validation 分割だけ** から決める（[`calibrate_on_validation`]）。候補の
//! ONNX を validation の `input` だけに当て（正解ラベルは `calibrate` にだけ渡る）、確率 p から
//! ロジット `ln(p)` を得て評価器の `calibrate` に渡す。凍結 test の結果は T・τ に影響せず、適用権を取る
//! 前に確定する。`calibration:null` は validation が 0 件のときだけ（0 や 1 で埋めない）。校正の計算が
//! 失敗したら握りつぶさず `runtime_error`（`calibration failed`）で止める（適用権は消費しない。
//! オーナー決定 2026-10-09）。最小件数は設けず、少件数でも T・τ が確定する。件数は `n_validation` で
//! 利用者に示す（REQ-22 に基準なし。オーナー判断 2026-10-09）。T・τ を決める口は定義にも CLI 引数にも無い。
//!
//! # 保留と対象外（REQ-22・REQ-27・#479・#478）
//!
//! 校正があるとき、凍結 test の各行のスコア `ln(p)` に validation で決めた T・τ を適用して保留・対象外を
//! 数え、`abstention` と評価記録へ出す（[`count_abstention`]。評価器の
//! `compare_abstention_with_out_of_scope` を呼ぶだけ）。T・τ を test から決め直さない。対象外は τ ではなく
//! 定義の `out_of_scope_label` で判定する。対象外は「答えた」側で、`out_of_scope` は `answered` の内数
//! （`answered + abstained` が評価件数、`coverage = answered / total`）（80% は参考値で合否条件にしない）。校正が `null` なら `abstention` も `null`。

//! # 旧モデルとの比較（REQ-26・REQ-27・#488・#489）
//!
//! `--previous-project-dir` があるときだけ、旧プロジェクトが保存した予測（再推論しない）と新の適用結果の
//! 正誤の遷移（2×2・遷移率の Wilson 95% 区間・ラベル集合の前提）を stdout の `comparison` と評価記録の
//! `previous_comparison` へ出す（[`super::previous_comparison`]）。旧側の読み込み・照合はすべて台帳を開く
//! 前（適用権を取る前）に済ませ、違反は `invalid_input`（台帳を作らない・触れない）。終了コードに影響しない。
//! 無ければ `comparison:null`。

//! # 下限基準との比較（REQ-25・REQ-27・#339）
//!
//! 定義に `baseline_comparison`（事前登録した仮定）があるときだけ、majority との McNemar 比較
//! （Holm の族の大きさは 1 で恒等）の結果を評価記録の `baseline_comparison` へ残す（出力 JSON は変えない）。
//! majority は train 分割のラベルだけから作り、必要件数は定義の仮定から求める。どちらも適用権を取る前に
//! 確定する（[`super::baseline::prepare_baseline`]）。欄が無い定義では比較しない。
//!
//! # 再現性（REQ-26・REQ-27・TASK-26.3・#490）
//!
//! `--seed-run-project R`（反復可。自分を含め 3 run 以上）を指定すると、seed ごとの複製プロジェクトの
//! 評価記録を適用権を取る前に読んで照合し（違反は `invalid_input`。適用権を失わない）、自分の正解数の
//! 確定後に Wilson 95% 区間の重なりを判定して stdout・評価記録の `reproducibility` へ出す
//! （[`super::reproducibility`]）。終了コードには影響しない。

use std::cell::RefCell;
use std::path::Path;
use std::time::Instant;

use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::evaluation_record::{
    AbstentionRecord, CalibrationRecord, EvaluationRecord, MAX_EVALUATION_RECORD_BYTES,
    TypeMeaningQuadrantRecord,
};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::read_bounded;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::limits::INFER_TIME_LIMIT;
use fandhe_edge_core::stage_report::{
    EvaluateAbstention, EvaluateCalibration, EvaluateCompletedReport, EvaluateDetails,
    EvaluateLabelMetrics, EvaluateReport, PredictionLine, PredictionLineOutcome,
};
use fandhe_edge_data::eval_freeze::{EvalDataState, FreezeRecord};
use fandhe_edge_data::inspect::ValidRecord;
use fandhe_edge_data::split::Split;
use fandhe_edge_eval::abstention::{OutOfScopeLabel, compare_abstention_with_out_of_scope};
use fandhe_edge_eval::calibration::{
    Calibration, CalibrationRecord as CalibrationInput, MAX_CALIBRATION_CELLS, calibrate,
};
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
    MAX_INFER_BATCH_TOTAL_SCORES,
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
use crate::project::{
    EVALUATION_PREDICTIONS_FILE, EVALUATION_RECORD_FILE, MAX_PROJECT_FILE_BYTES, Project,
    SELECTION_FILE, fail, inspect_bytes, invalid, runtime,
};
use crate::stage_output::{EvaluateStart, evaluate_start};

use super::baseline::{PreparedBaseline, compare, prepare_baseline};
use super::candidate_artifact::{
    CandidateArtifact, check_meta_consistency, load_candidate_artifact,
};
use super::diagnostics::{PreparedStats, prepare_stats};
use super::infer::load_backend;
use super::inspect::load_frozen_evaluation;
use super::ledger::HeldLedger;
use super::previous_comparison::{PreparedPrevious, current_correctness, prepare_previous};
use super::reproducibility::{OwnRun, SeedRuns, check_run_count, load_seed_runs};
use super::select::compute_selection;
use super::train::{
    allotted_time_limit, candidate_rel, effective_train_seed, load_trained,
    request_is_smoke_trained, request_matches_candidate, resolve_candidates, train_rows,
    verified_split,
};

/// `evaluate` の成功結果（stdout の JSON 1 つへ写す）。
#[derive(Debug)]
pub enum EvaluateOutcome {
    /// 評価データ未定義（exit 0。評価済みを装わない）。
    Skipped(EvaluateReport),
    /// 凍結した評価データへの適用が完了した。
    Completed(Box<EvaluateCompletedReport>),
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
    check_run_count(args.seed_run_projects.len())?;
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
    // 旧モデルとの比較の共通レコードの照合にも使う（#488）。
    let current_records = inspect_bytes(&eval_bytes, &definition)?;
    let eval_ids: Vec<String> = current_records.iter().map(|r| r.id.clone()).collect();
    // 予測行の件数照合は適用権を取る前に済ませる（保存の失敗で適用権を失わない）。
    if eval_ids.len() != decoded.len() {
        return Err(runtime("evaluation record count mismatch"));
    }
    // 予測ファイルの大きさの上界も適用権を取る前に確認する（採点側の読み込み上限と同じ定数。#445）。
    let option_ids: Vec<&str> = definition.options().iter().map(|c| c.id.as_str()).collect();
    check_predictions_limit(&eval_ids, &option_ids, MAX_PROJECT_FILE_BYTES)?;
    // 推論ごとのスコア（保存用。推論関数の戻り値の型は変えず、横で受ける）。
    let scores_log: RefCell<Vec<Vec<f64>>> = RefCell::new(Vec::new());
    // 下限基準（majority）と必要件数は適用権を取る前に確定する（失敗しても適用権を使い切らない）。
    // 引数は train 側の入力だけで、評価データを渡せない（REQ-27・#339）。
    let baseline = prepare_baseline(&definition, &records, &split)?;
    // 校正は validation だけから、適用権を取る前に確定する（REQ-22・REQ-27・#477）。
    let calibration = calibrate_on_validation(&target, &definition, &records, &split)?;
    let definition_sha256 = definition
        .canonical_hash()
        .map_err(|_| runtime("cannot hash definition"))?
        .to_hex();
    let onnx_digest = Sha256Digest::of_bytes(&target.artifact.onnx_bytes);
    // 旧モデルとの比較の材料は、台帳を開く前（適用権を取る前）に読み・照合して確定する（#488・REQ-27）。
    let previous = args
        .previous_project_dir
        .as_deref()
        .map(|dir| prepare_previous(cwd, dir, &definition, &freeze, &current_records))
        .transpose()?;

    // 再現性の複製プロジェクトの読み込みと照合も適用権を取る前（台帳を開く前）に済ませる（#490）。
    let seed_runs = if args.seed_run_projects.is_empty() {
        None
    } else {
        let total =
            u64::try_from(decoded.len()).map_err(|_| runtime("evaluation count overflow"))?;
        Some(load_seed_runs(
            cwd,
            &args.seed_run_projects,
            &OwnRun {
                project: &project,
                candidate: args.candidate,
                freeze: &freeze,
                definition_sha256: &definition_sha256,
                candidate_id: &target.candidate_id,
                config_id: target.config_id.as_str(),
                total,
            },
        )?)
    };

    // 診断の基礎統計も適用権を取る前（台帳を開く前）に求める（失敗しても適用権を使わない。#492）。
    let diagnostics_stats =
        prepare_stats(&option_ids, train_rows(&records, &split), &current_records)?;

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
            FinalizeContext {
                project: &project,
                definition: &definition,
                freeze: &freeze,
                target: &target,
                candidate: args.candidate,
                record_rel: &record_rel,
                definition_sha256,
                onnx_digest,
                baseline: baseline.as_ref(),
                calibration: calibration.as_ref(),
                previous: previous.as_ref(),
                seed_runs: seed_runs.as_ref(),
                diagnostics_stats,
            },
            &PredictionsSink {
                rel: &predictions_rel,
                ids: &eval_ids,
                scores: &scores_log.borrow(),
            },
            applied,
        )
        .map_err(|e| {
            finish_error = Some(e);
            EvalPredictFailure::Failed
        })
    };
    let (_applied, report) = match apply_to_frozen_data(
        held_ledger.ledger(),
        FrozenApplication {
            freeze: &freeze,
            definition: &definition,
            eval_bytes: &eval_bytes,
            target: &target,
            onnx_digest,
            scores_log: &scores_log,
        },
        finish,
    ) {
        Ok(v) => v,
        Err(report) => return Err(finish_error.take().unwrap_or(report)),
    };

    Ok(EvaluateOutcome::Completed(Box::new(report)))
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
    let allotted = allotted_time_limit(project, index, &candidate.candidate_id)?;
    if !request_matches_candidate(&request, &candidate.params, allotted, records, split) {
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

/// validation 分割だけから校正（温度 T・保留しきい値 τ）を決める（REQ-22・REQ-27・#477）。
///
/// 推論には `input` だけを渡し（`evaluate` の推論関数と同じ経路）、確率 p から `ln(p)` をロジットとして
/// 評価器の [`calibrate`] へ渡す。正解ラベルは評価器にだけ渡り、凍結 test のデータ・結果は使わない。
/// 適用権（`apply_once`）より前に呼ぶ。validation が 0 件のときだけ `None`（`calibration:null`。
/// 0 や 1 で埋めない）。最小件数は設けず、少件数でも T・τ が確定する（件数は `n_validation` で示す。
/// REQ-22 に基準なし。オーナー判断 2026-10-09）。校正の計算の失敗は `runtime_error`、
/// 上限（件数・時間）超過は `limit_exceeded`。
fn calibrate_on_validation(
    target: &PreparedCandidate,
    definition: &Definition,
    records: &[ValidRecord],
    split: &fandhe_edge_data::split::SplitResult,
) -> Result<Option<Calibration>, ErrorReport> {
    let validation: Vec<&ValidRecord> = records
        .iter()
        .filter(|r| split.by_record.get(&r.id) == Some(&Split::Validation))
        .collect();
    if validation.is_empty() {
        return Ok(None);
    }
    if validation.len() > MAX_INFER_BATCH_LEN {
        return Err(fail(
            ExitCode::LimitExceeded,
            "validation data has too many records",
        ));
    }
    let options = definition.options();
    // 推論とロジットの確保の前に「件数 × 選択肢数」を、バッチ推論の保持スコア上限と
    // 校正の計算量上限の小さい方で検証する（REQ-39 資源の上限）。
    if !score_cells_within_limit(validation.len(), options.len()) {
        return Err(fail(
            ExitCode::LimitExceeded,
            "validation data has too many scores",
        ));
    }
    let backend = load_backend(
        &target.artifact.onnx_bytes,
        target.kind,
        target.artifact.meta.kind_version(),
        options.len(),
    )?;
    let pipeline = InferencePipeline::new(ByteEncodingPreprocessor::new(target.max_bytes), backend);
    let deadline = Instant::now().checked_add(MAX_INFER_BATCH_DURATION);
    let mut logits: Vec<Vec<f64>> = Vec::with_capacity(validation.len());
    for record in &validation {
        if deadline.is_some_and(|d| Instant::now() > d) {
            return Err(fail(
                ExitCode::LimitExceeded,
                "validation inference exceeded the time limit",
            ));
        }
        let prediction = pipeline
            .infer_one_within(&record.input, INFER_TIME_LIMIT)
            .map_err(|e| e.to_error_report())?;
        logits.push(prediction.scores().iter().map(|p| p.ln()).collect());
    }
    let labels: Vec<&str> = options.iter().map(|c| c.id.as_str()).collect();
    let inputs: Vec<CalibrationInput<'_>> = validation
        .iter()
        .zip(&logits)
        .map(|(r, l)| CalibrationInput {
            gold: &r.label_id,
            logits: l,
        })
        .collect();
    calibrate_checked(&labels, &inputs).map(Some)
}

/// 校正・保留の計算用に保持するロジットの総数（件数 × 選択肢数）が上限内か。桁あふれは上限超過として扱う。
fn score_cells_within_limit(records: usize, options: usize) -> bool {
    records
        .checked_mul(options)
        .is_some_and(|n| n <= MAX_INFER_BATCH_TOTAL_SCORES.min(MAX_CALIBRATION_CELLS))
}

/// 評価器の [`calibrate`] を呼び、失敗を握りつぶさず `runtime_error`（固定 message）へ写す。
fn calibrate_checked(
    labels: &[&str],
    inputs: &[CalibrationInput<'_>],
) -> Result<Calibration, ErrorReport> {
    calibrate(labels, inputs).map_err(|_| runtime("calibration failed"))
}

/// 評価データの分解の失敗（本文・行番号を含まない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvalDecodeError {
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

/// 照合済みの評価データのバイト列を、評価対象のレコード列へ検査して分ける（`evaluate` と採点入口
/// `fandhe-edge-score` が共有する唯一のデコード経路。評価対象の件数・順序・ID が両者で食い違わない。
/// REQ-27・#445）。
///
/// 件数は推論バッチの上限（[`MAX_INFER_BATCH_LEN`]）までとし、0 件は拒否する
/// （`evaluate_single_select` が適用後に失敗しないよう、適用前の事前検査にも使う。REQ-39）。
pub(crate) fn evaluation_records(
    bytes: &[u8],
    definition: &Definition,
) -> Result<Vec<ValidRecord>, EvalDecodeError> {
    let records = inspect_bytes(bytes, definition).map_err(|_| EvalDecodeError::Invalid)?;
    if records.is_empty() {
        return Err(EvalDecodeError::Invalid);
    }
    if records.len() > MAX_INFER_BATCH_LEN {
        return Err(EvalDecodeError::TooMany);
    }
    Ok(records)
}

/// 照合済みの評価データのバイト列を、`input` と正解ラベルへ分ける（[`evaluation_records`] の薄い写像）。
fn decode_evaluation(
    bytes: &[u8],
    definition: &Definition,
) -> Result<Vec<LabeledInput>, EvalDecodeError> {
    Ok(evaluation_records(bytes, definition)?
        .into_iter()
        .map(|r| LabeledInput {
            input: r.input,
            gold: r.label_id,
        })
        .collect())
}

/// [`finalize_evaluation`] へ渡す、評価の確定に要るプロジェクト・候補・凍結データの情報。
struct FinalizeContext<'a> {
    project: &'a Project,
    definition: &'a Definition,
    freeze: &'a FreezeRecord,
    target: &'a PreparedCandidate,
    candidate: usize,
    record_rel: &'a Path,
    definition_sha256: String,
    onnx_digest: Sha256Digest,
    baseline: Option<&'a PreparedBaseline>,
    calibration: Option<&'a Calibration>,
    previous: Option<&'a PreparedPrevious>,
    seed_runs: Option<&'a SeedRuns>,
    diagnostics_stats: PreparedStats,
}

/// 評価結果を確定する（指標の算出・完了報告の構築・評価記録の書き込み。台帳への完了記録の前に呼ぶ）。
///
/// 指標は評価器（`fandhe-edge-eval`）で求め、CLI では再実装しない（REQ-24）。記録の書き込みは最後に行い、
/// それ以前の失敗では記録を残さない。
fn finalize_evaluation(
    ctx: FinalizeContext<'_>,
    predictions: &PredictionsSink<'_>,
    applied: &AppliedOnce<Vec<Outcome>>,
) -> Result<EvaluateCompletedReport, ErrorReport> {
    let FinalizeContext {
        project,
        definition,
        freeze,
        target,
        candidate,
        record_rel,
        definition_sha256,
        onnx_digest,
        baseline,
        calibration,
        previous,
        seed_runs,
        diagnostics_stats,
    } = ctx;
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
    // 診断は同じ 1 回の適用の混同行列から求める（記録の書き込みより前。REQ-27・#492）。
    let previous_labels = previous.map(PreparedPrevious::previous_labels);
    let diagnostics =
        super::diagnostics::build(diagnostics_stats, &computed, previous_labels.as_deref())?;
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
    // 保留・対象外の件数。T・τ は適用権を取る前に validation から確定した値をそのまま使う
    // （凍結 test の結果で決め直さない。REQ-22・REQ-27・#479）。
    let abstention = match calibration {
        None => None,
        Some(c) => Some(count_abstention(
            c,
            &labels,
            definition.out_of_scope_label(),
            &applied.golds,
            predictions.scores,
            correct,
        )?),
    };
    // 旧モデルとの比較（`--previous-project-dir` のときだけ。新側の正誤は適用済みの予測から求める。#488）。
    let comparison = match previous {
        None => None,
        Some(p) => {
            let current = current_correctness(&labels, &applied.golds, &applied.output)?;
            Some(p.compare(&labels, &current)?)
        }
    };
    // 再現性（自分の正解数の確定後・記録の前。判定は評価器に委ねる。#490）。
    let reproducibility = seed_runs
        .map(|r| r.judge(correct, total, freeze.sha256()))
        .transpose()?;
    let q = &computed.type_meaning_quadrant;
    let quadrant = TypeMeaningQuadrantRecord {
        type_ok_meaning_ok: q.type_ok_meaning_ok(),
        type_ok_meaning_ng: q.type_ok_meaning_ng(),
        type_ng_count: q.type_ng_count(),
        abstain: q.abstain(),
        error: q.error(),
    };
    let details = EvaluateDetails {
        macro_f1_excluded_labels: computed.macro_f1.excluded_labels().to_vec(),
        per_label: computed
            .per_label
            .iter()
            .map(|m| EvaluateLabelMetrics {
                label: m.label.clone(),
                support: m.support,
                predicted: m.predicted_count,
                precision: m.precision,
                recall: m.recall,
                f1: m.f1,
            })
            .collect(),
        type_meaning_quadrant: quadrant,
        calibration: calibration.map(|c| EvaluateCalibration {
            temperature: c.chosen_temperature(),
            adopted: c.adopted(),
            threshold: c.threshold(),
            n_validation: c.n_validation(),
            validation_coverage: c.validation_coverage().value(),
        }),
        out_of_scope_label: definition.out_of_scope_label().map(str::to_string),
        abstention: abstention.map(|(report, _)| report),
        comparison: comparison.as_ref().map(|(report, _)| report.clone()),
        reproducibility: reproducibility.as_ref().map(|(report, _)| report.clone()),
        diagnostics,
    };
    let report = EvaluateCompletedReport::completed(
        candidate,
        target.kind_name.clone(),
        correct,
        total,
        computed.macro_f1.value(),
        details,
    )
    .ok_or_else(|| runtime("cannot build evaluation report"))?;

    // 予測ファイルの本文を先に作り、そのバイト列の sha256 を評価記録へ束縛する（#445・REQ-27）。
    let predictions_jsonl =
        predictions.to_jsonl(&labels, &applied.output, MAX_PROJECT_FILE_BYTES)?;
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
        predictions_sha256: Some(Sha256Digest::of_bytes(predictions_jsonl.as_bytes()).to_hex()),
        type_meaning_quadrant: Some(quadrant),
        calibration: calibration.map(|c| CalibrationRecord {
            temperature: c.chosen_temperature(),
            adopted: c.adopted(),
            threshold: c.threshold(),
            n_validation: c.n_validation(),
            validation_answered: c.validation_coverage().numerator(),
        }),
        out_of_scope_label: definition.out_of_scope_label().map(str::to_string),
        abstention: abstention.map(|(_, record)| record),
        previous_comparison: comparison.map(|(_, record)| record),
        reproducibility: reproducibility.map(|(_, record)| record),
    };
    let record_json = record
        .to_json_vec()
        .map_err(|_| runtime("cannot serialize evaluation record"))?;
    // `package` は記録を `MAX_EVALUATION_RECORD_BYTES` までしか読まない。読めない記録を書かない。
    if u64::try_from(record_json.len()).map_or(true, |n| n > MAX_EVALUATION_RECORD_BYTES) {
        return Err(runtime("evaluation record is too large"));
    }
    // 1 件ごとの予測を先に書く（記録があるのに予測が無い状態を作らない）。失敗は記録の失敗と同じ扱い。
    // この時点で適用権は取得済み（ロックは消費され、`finish` が失敗しても残る）で、再実行は
    // `AlreadyApplied` で拒否される（fail-closed）。以降の失敗で予測ファイルだけが残っても、
    // 凍結 test に適用した痕跡として消さない（REQ-27。eval の `apply_once_then` の doc も参照）。
    project.write_new(predictions.rel, predictions_jsonl.as_bytes())?;
    // 書いた後は読み取り専用にする（凍結データの配置と同じ扱い。改ざんの抑止であり、
    // 記録の封印は外部台帳〔#168〕の範囲）。
    project.set_read_only(predictions.rel)?;
    project.write_new(record_rel, &record_json)?;
    Ok(report)
}

/// 凍結 test の各行のスコア（確率）から `ln(p)` をロジットとして、validation で決めた T・τ を適用し、
/// 保留・対象外の件数を数える（REQ-22・REQ-27・#479・#478）。評価ロジックは評価器の
/// [`compare_abstention_with_out_of_scope`] に委ね、CLI では再実装しない。
///
/// 対象外は τ ではなく定義の `out_of_scope_label` で判定する（REQ-22 異常系）。対象外は「答えた」側で、
/// `out_of_scope` は `answered` の内数（`answered + abstained` が評価件数、`coverage = answered / total`）。
/// `correct_answered`・`adopted_error`（分母 = 全件 − 保留 = `answered`）も対象外の行を採用判定に含める
/// （評価器の定義どおり）。計算に失敗したら握りつぶさず `runtime_error`。
fn count_abstention(
    calibration: &Calibration,
    labels: &[&str],
    out_of_scope_label: Option<&str>,
    golds: &[String],
    scores: &[Vec<f64>],
    expected_correct: u64,
) -> Result<(EvaluateAbstention, AbstentionRecord), ErrorReport> {
    let fail_abstention = || runtime("abstention failed");
    if golds.len() != scores.len() {
        return Err(fail_abstention());
    }
    // ロジットの複製を確保する前に総数を上限で検証する（REQ-39 資源の上限）。
    if !score_cells_within_limit(scores.len(), labels.len()) {
        return Err(fail(
            ExitCode::LimitExceeded,
            "evaluation data has too many scores",
        ));
    }
    let logits: Vec<Vec<f64>> = scores
        .iter()
        .map(|s| s.iter().map(|p| p.ln()).collect())
        .collect();
    let inputs: Vec<CalibrationInput<'_>> = golds
        .iter()
        .zip(&logits)
        .map(|(g, l)| CalibrationInput { gold: g, logits: l })
        .collect();
    let oos = out_of_scope_label
        .map(|id| OutOfScopeLabel::new(calibration, id))
        .transpose()
        .map_err(|_| fail_abstention())?;
    let cmp = compare_abstention_with_out_of_scope(labels, calibration, oos.as_ref(), &inputs)
        .map_err(|_| fail_abstention())?;
    // 保留なしの正解数は評価指標の `correct` と別経路（評価器の argmax）で求まる。ずれは記録前に止める。
    if cmp.without_abstention().accuracy.overall.numerator() != expected_correct {
        return Err(runtime("abstention disagrees with metrics"));
    }
    let cov = cmp.coverage();
    let with = cmp.with_abstention();
    let out_of_scope = cmp.out_of_scope();
    let answered = cov.answered();
    let abstained = cov.abstained();
    let correct_answered = with.accuracy.overall.numerator();
    Ok((
        EvaluateAbstention {
            answered,
            abstained,
            out_of_scope,
            coverage: cov.coverage().value(),
            correct_answered,
            adopted_error: cmp.adopted_error().map(|r| r.value()),
            unconditional_error: cmp.unconditional_error().value(),
        },
        AbstentionRecord {
            answered,
            abstained,
            out_of_scope,
            correct_answered,
        },
    ))
}

/// 1 件ごとの予測の保存に必要な材料（書き込み先・評価データの id 列・推論ごとのスコア）。
///
/// 行形式は [`PredictionLine`]。stdout の JSON・[`EvaluationRecord`] のスキーマには影響しない
/// （REQ-27。PoC-26 の採点入口 `fandhe-edge-score` が読む。REQ-41・#445）。
struct PredictionsSink<'a> {
    rel: &'a Path,
    ids: &'a [String],
    scores: &'a [Vec<f64>],
}

impl PredictionsSink<'_> {
    /// 評価データの行順の JSONL を作る。id・予測の件数が合わなければ `runtime_error`、構築中に `limit`
    /// バイトを超えたら打ち切って `limit_exceeded`。
    fn to_jsonl(
        &self,
        labels: &[&str],
        outcomes: &[Outcome],
        limit: u64,
    ) -> Result<String, ErrorReport> {
        if self.ids.len() != outcomes.len() {
            return Err(runtime("cannot build evaluation predictions"));
        }
        let mut out = String::new();
        for (i, (id, outcome)) in self.ids.iter().zip(outcomes).enumerate() {
            let scores = self.scores.get(i).map(|s| (labels, s.as_slice()));
            let line_outcome = match outcome {
                Outcome::Label(l) => PredictionLineOutcome::Label(l.clone()),
                Outcome::Invalid => PredictionLineOutcome::Invalid,
                Outcome::Abstain => PredictionLineOutcome::Abstain,
                Outcome::Error => PredictionLineOutcome::Error,
            };
            let line = PredictionLine::new(id, line_outcome, scores)
                .to_json_line()
                .map_err(|_| runtime("cannot serialize evaluation predictions"))?;
            out.push_str(&line);
            out.push('\n');
            if u64::try_from(out.len()).map_or(true, |n| n > limit) {
                return Err(predictions_limit_error());
            }
        }
        Ok(out)
    }
}

fn predictions_limit_error() -> ErrorReport {
    fail(
        ExitCode::LimitExceeded,
        "evaluation predictions exceed size limit",
    )
}

/// 予測ファイルの大きさの上界（件数 × (固定部 + id の最大長 + 全ラベル ID 長の合計 + ラベルごとの固定部。
/// JSON エスケープで最大 6 倍）が `limit` 以下であることを確認する。超えれば `limit_exceeded`
/// （適用権を取る前に呼ぶ。#445・REQ-39）。
fn check_predictions_limit(ids: &[String], labels: &[&str], limit: u64) -> Result<(), ErrorReport> {
    let escaped = |n: usize| (n as u64).saturating_mul(6);
    let max_id = ids.iter().map(String::len).max().unwrap_or(0);
    let max_label = labels.iter().map(|l| l.len()).max().unwrap_or(0);
    let per_label_sum: u64 = labels
        .iter()
        .map(|l| escaped(l.len()).saturating_add(40))
        .fold(0, u64::saturating_add);
    let per_line = 128u64
        .saturating_add(escaped(max_id))
        .saturating_add(escaped(max_label))
        .saturating_add(per_label_sum);
    match (ids.len() as u64).checked_mul(per_line) {
        Some(bound) if bound <= limit => Ok(()),
        _ => Err(predictions_limit_error()),
    }
}

/// [`apply_to_frozen_data`] へ渡す、凍結データへの適用に要る情報。
struct FrozenApplication<'a> {
    freeze: &'a FreezeRecord,
    definition: &'a Definition,
    eval_bytes: &'a [u8],
    target: &'a PreparedCandidate,
    onnx_digest: Sha256Digest,
    scores_log: &'a RefCell<Vec<Vec<f64>>>,
}

/// 台帳で適用権を取り、凍結した評価データへ 1 回だけ推論を当てる。
fn apply_to_frozen_data<R>(
    ledger: &FinalTestLedger,
    application: FrozenApplication<'_>,
    finish: impl FnOnce(&AppliedOnce<Vec<Outcome>>) -> Result<R, EvalPredictFailure>,
) -> Result<(AppliedOnce<Vec<Outcome>>, R), ErrorReport> {
    let FrozenApplication {
        freeze,
        definition,
        eval_bytes,
        target,
        onnx_digest,
        scores_log,
    } = application;
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

    /// REQ-39・#445: 予測ファイルの上界（2 件 × (128 + 6×2 + 6×1 + 2×(6+40)) = 476）が上限を超えると、
    /// 適用権を取る前の確認が `limit_exceeded` になる。ちょうど上限なら通る。
    #[test]
    fn req39_issue445_predictions_upper_bound_is_checked_against_limit() {
        let ids = vec!["e1".to_string(), "e2".to_string()];
        let labels = ["a", "b"];
        assert!(check_predictions_limit(&ids, &labels, 476).is_ok());
        let err = check_predictions_limit(&ids, &labels, 475).expect_err("over");
        assert_eq!(err.code, ExitCode::LimitExceeded);
    }

    /// REQ-39・#445: 予測行の構築中に上限を超えたら打ち切って `limit_exceeded`。
    #[test]
    fn req39_issue445_predictions_building_stops_at_limit() {
        let ids = vec!["e1".to_string(), "e2".to_string()];
        let sink = PredictionsSink {
            rel: Path::new("x"),
            ids: &ids,
            scores: &[],
        };
        let outcomes = [Outcome::Label("a".into()), Outcome::Label("b".into())];
        let one_line = sink.to_jsonl(&["a", "b"], &outcomes, 1 << 20).expect("ok");
        let first_len = one_line.find('\n').expect("nl") as u64 + 1;
        assert!(sink.to_jsonl(&["a", "b"], &outcomes, first_len * 2).is_ok());
        let err = sink
            .to_jsonl(&["a", "b"], &outcomes, first_len * 2 - 1)
            .expect_err("over");
        assert_eq!(err.code, ExitCode::LimitExceeded);
    }

    /// REQ-22・REQ-27・#477: 校正の計算の失敗は握りつぶさず `runtime_error`（exit 70）になる。
    /// REQ-39: 校正用ロジットの総数は確保前に上限で拒否する（桁あふれも拒否）。
    #[test]
    fn req39_validation_cells_are_limited_before_inference() {
        let limit = MAX_INFER_BATCH_TOTAL_SCORES.min(MAX_CALIBRATION_CELLS);
        assert!(score_cells_within_limit(limit, 1));
        assert!(!score_cells_within_limit(limit + 1, 1));
        assert!(!score_cells_within_limit(100_000, 1024));
        assert!(!score_cells_within_limit(usize::MAX, 2));
    }

    #[test]
    fn req22_issue477_calibration_failure_is_runtime_error() {
        let bad = [CalibrationInput {
            gold: "a",
            logits: &[0.0],
        }];
        let err = calibrate_checked(&["a", "b"], &bad).expect_err("length mismatch");
        assert_eq!(err.code, ExitCode::RuntimeError);
        assert_eq!(err.message, "calibration failed");
    }

    /// REQ-22・REQ-27・#479・#478: validation の T・τ を test のスコアに適用すると、確信度の低い誤りが
    /// 保留になって `adopted_error < unconditional_error`、argmax が対象外ラベルの行は τ に関係なく
    /// `out_of_scope` に数えられ（`answered` の内数）、保留にならない（`answered + abstained` は評価件数）。
    #[test]
    fn req22_issue479_count_abstention_counts_out_of_scope_within_answered() {
        let labels = ["a", "b", "c"];
        let confident = |i: usize| {
            let mut p = [0.05_f64; 3];
            p[i] = 0.9;
            p.to_vec()
        };
        let unsure = vec![0.45, 0.40, 0.15];
        // validation: 確信度の高い正解 9 件と、確信度の低い誤り 2 件（τ は 11 件の下から 3 番目の確信度）。
        let mut validation: Vec<(&str, Vec<f64>)> = Vec::new();
        for _ in 0..4 {
            validation.push(("a", confident(0)));
            validation.push(("b", confident(1)));
        }
        validation.push(("b", confident(1)));
        validation.push(("b", unsure.clone()));
        validation.push(("b", vec![0.5, 0.4, 0.1]));
        let logits = |p: &[f64]| p.iter().map(|v| v.ln()).collect::<Vec<f64>>();
        let val_logits: Vec<Vec<f64>> = validation.iter().map(|(_, p)| logits(p)).collect();
        let inputs: Vec<CalibrationInput<'_>> = validation
            .iter()
            .zip(&val_logits)
            .map(|((g, _), l)| CalibrationInput { gold: g, logits: l })
            .collect();
        let calibration = calibrate_checked(&labels, &inputs).expect("calibrate");
        // test: 確信度の高い正解 4 件、確信度の低い誤り 2 件、argmax が対象外 c の行 2 件（正解 1・誤り 1）。
        let cases: [(&str, Vec<f64>); 8] = [
            ("a", confident(0)),
            ("a", confident(0)),
            ("b", confident(1)),
            ("b", confident(1)),
            ("b", unsure.clone()),
            ("b", unsure),
            ("c", vec![0.2, 0.1, 0.7]),
            ("a", vec![0.3, 0.2, 0.5]),
        ];
        let golds: Vec<String> = cases.iter().map(|(g, _)| g.to_string()).collect();
        let scores: Vec<Vec<f64>> = cases.iter().map(|(_, p)| p.clone()).collect();
        let (plain, _) =
            count_abstention(&calibration, &labels, None, &golds, &scores, 5).expect("plain");
        let (report, record) =
            count_abstention(&calibration, &labels, Some("c"), &golds, &scores, 5).expect("oos");
        // 保留なしの正解数が評価指標とずれたら記録前に止める。
        assert_eq!(
            count_abstention(&calibration, &labels, None, &golds, &scores, 4)
                .expect_err("mismatch")
                .message,
            "abstention disagrees with metrics"
        );
        // 対象外なし: 確信度の高い 4 件だけが答え、残り 4 件（対象外ラベルの 2 件を含む）は保留。
        assert_eq!(
            (
                plain.answered,
                plain.abstained,
                plain.out_of_scope,
                plain.correct_answered
            ),
            (4, 4, 0, 4)
        );
        assert_eq!(plain.adopted_error, Some(0.0));
        // 対象外あり: argmax が c の 2 件は保留でなく answered に入り、うち 2 件が out_of_scope（内数）。
        // 答えた行は 4 + 対象外 2 の 6 件で coverage 6/8、正解は 4 + gold が c の 1 件、誤りは 1/6。
        assert_eq!(
            (
                report.answered,
                report.abstained,
                report.out_of_scope,
                report.correct_answered
            ),
            (6, 2, 2, 5)
        );
        assert_eq!(record.answered + record.abstained, 8);
        assert!(record.out_of_scope <= record.answered);
        assert!((report.coverage - report.answered as f64 / 8.0).abs() < 1e-9);
        assert!((report.coverage - 0.75).abs() < 1e-9);
        assert!((report.adopted_error.expect("answered") - 1.0 / 6.0).abs() < 1e-9);
        assert!((report.unconditional_error - 0.375).abs() < 1e-9);
        assert!(report.adopted_error.expect("answered") < report.unconditional_error);
    }

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

    /// REQ-27・#445: 同じ `input` に異なるラベルが付いた矛盾入力も評価対象から除外されない
    /// （採点入口と共有するデコード経路。件数は全レコード）。
    #[test]
    fn req27_issue445_evaluation_records_keep_contradictory_inputs() {
        let data = b"{\"id\":\"1\",\"input\":\"x\",\"output\":{\"intent\":\"a\"}}\n{\"id\":\"2\",\"input\":\"x\",\"output\":{\"intent\":\"b\"}}\n";
        let records = evaluation_records(data, &definition()).expect("records");
        assert_eq!(records.len(), 2);
        assert_eq!(
            decode_evaluation(data, &definition())
                .expect("decode")
                .len(),
            2
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
