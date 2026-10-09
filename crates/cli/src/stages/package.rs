//! `package` 工程: 選定した候補から配布パッケージを組み立て、容量を計測する
//! （REQ-30・REQ-32・REQ-33・REQ-39・TASK-33.1-2・#136）。
//!
//! # 手順
//!
//! 0. 開始時（ステージングを作る前）に評価データの凍結ハッシュを確認する（不一致・凍結記録の欠落は
//!    `invalid_input` で停止。REQ-17）。評価データがあるプロジェクトは、選定候補の評価完了記録
//!    （`candidates/<N>/evaluation_record.json`。`evaluate` が書く。#314）が無い限り
//!    `invalid_input`（`evaluation has not been completed`）で拒否し、記録の ONNX・`artifact.json`・
//!    評価データ・定義のハッシュが今の実体と一致しなければ
//!    `evaluation record does not match the package` で拒否し、最終 test の台帳に適用完了が
//!    記録されていなければ `evaluation has not been completed` で拒否する（成果物を読んだ後・ステージングを
//!    作る前に確認する。[`verify_evaluation_record`]。REQ-27）
//! 1. `selection_record.json`（`select` の記録）から選定候補を読み、`request.json`・`result.json`
//!    を再検証つきで読み戻す（[`super::train::load_trained`]）。記録の `candidate_id`・添字の既定候補・
//!    学習リクエストの `kind` の一致も確認する
//! 2. 選定候補の学習ワーカー出力（`artifact.json`・ONNX ファイル）と登録済みの `definition.json`
//!    （選択肢表）をステージングへ新規コピーする（既存の `package/` は拒否。上書きしない）
//! 3. `artifact.json` の `onnx_sha256` とコピーした ONNX の sha256 の一致、および `kind`・
//!    `label_order`・`max_bytes` の定義・選定候補との一致を確認する
//!    （パッケージの自己整合性。**外部台帳による完全性検証〔#168〕の代替ではない**）
//!    あわせて、公開前に `infer` と同じ検証（ガード層の形式許可リスト・ONNX の読み込み。
//!    [`super::infer::load_backend`]）を通す
//!    `train --smoke` の結果は `--allow-smoke`（検証専用）が無ければ拒否する（REQ-27）。
//!    `--allow-smoke` の検証専用パッケージは最終 test を適用できないため、評価完了の確認を行わない
//! 4. 定義の `limits.max_infer_p95_us` があるときだけ、`train` 分割の入力（[`latency_inputs`]）で
//!    推論待ち時間 p95 を計測し（warmup 20・iters 1000 固定）、上限と照合する（REQ-31。#338）。
//!    ステージングを作る前に行う。計測の失敗（推論失敗・時計の逆行・タイムアウト）は `runtime_error`（70）
//! 5. 容量を計測し（[`measure_opened_files_with_limit`]。REQ-30）、利用者が設定した上限
//!    （`limits.max_package_bytes`）を超えたら `limit_exceeded`。上限が無ければ照合しない
//!    （既定の強制上限は無い。2026-10-06 オーナー判断）
//!
//! 2・3・5 は `package.staging/` で行い、容量と p95 がともに上限内のときだけ `package/` へ原子的に名前替えして
//! 公開する。途中の失敗・容量または p95 の上限超過ではステージングを片付け、`package/` を作らない
//! （推論可能な場所に半端・超過のパッケージを残さない。既存の `package/` は事前に拒否し、
//! 置き換えも削除もしない）。容量は計測して上限照合（`limit_exceeded` の判定）に使う。
//!
//! # 合否基準（#328・REQ-24・REQ-33）
//!
//! 定義ファイルの省略可能な `acceptance.min_accuracy_bp`（1 万分率）を、評価記録の
//! `correct`/`total` と評価器の [`judge_min_accuracy`]（Wilson 95% 区間。pass / fail / undeterminable）
//! で照合する（[`quality_from_acceptance`]）。基準が無ければ従来どおり `judgment:null`・
//! `acceptance_defined:false`・exit 0。基準があり評価データが無ければ判定不能（exit 12）。
//! 下限基準との比較（`baseline_comparison`。#339）の `verdict` は合否に使わず、評価記録の照合
//! （改変検出）にだけ使う。`baseline_comparison` だけの定義は基準なしと同じ出力になる（#344）。
//! `fail`（exit 10）・判定不能でも公開の関門は容量だけで、`package/` は公開する（合否は終了コードと
//! JSON で伝える）。基準は定義の正準化ハッシュに含まれるため、`evaluate` の後に書き換えると
//! 評価記録の `definition_sha256` 照合で `invalid_input` になる（REQ-27）。
//!
//! # 上限（#338・REQ-30・REQ-31・REQ-21）
//!
//! 定義ファイルの省略可能な `limits`（`max_infer_p95_us`・`max_package_bytes`）を、設定されたものだけ照合する。合否
//! （`acceptance`）とは別で、超過は exit 20 とし合否より優先する（`resolve_package_outcome`）。
//! p95 の計測入力は `train` 分割の `input` だけで（validation・test と凍結した評価データは使わない。
//! REQ-27）、件数・総バイト数の上限で先頭から切り詰める。計測回数は利用者設定にしない。
//! 証拠種別はテストハーネス（偽の時計・固定 fixture の ONNX）で、実機での p95 計測は人の担当である。
//!
//! # 計測値の出力（#340・REQ-30・REQ-31・REQ-33）
//!
//! exit 0・10・12・20 の stdout JSON の末尾に `capacity`（常に。5 構成要素の内訳・合計・上限〔未設定は
//! `null`〕・`exceeded`・目安 `guideline_bytes`〔40MB〕・`over_guideline`）と
//! `infer_p95`（`limits.max_infer_p95_us` があるときだけ。µs へ切り上げ。無ければ `null`）を載せる。
//! 値は [`PackageRunResult::metrics`] で返し、`exceeded` は runtime の照合結果から写す（[`capacity_report`]・
//! [`infer_p95_report`]。写像の不整合は公開前に `runtime_error`）。p95 は上限が無いとき計測しない。
//! 目安超過（`over_guideline`）は警告で、終了コード・`package/` の公開に影響しない（REQ-30）。
//!
//! 評価記録の `correct` は
//! 外部台帳に記録されておらず、範囲内の書き換えは検出できない（#168 の完全性検証が対象）。

use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::path::Path;

use fandhe_edge_core::artifact_meta::ArtifactMeta;
use fandhe_edge_core::definition::{Definition, Limits, MAX_DEFINITION_FILE_BYTES};
use fandhe_edge_core::evaluation_record::{EvaluationRecord, MAX_EVALUATION_RECORD_BYTES};
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::stage_report::{
    InferP95, PackageCapacity, PackageCapacityComponents, PackageComponentSize, PackageMetrics,
};
use fandhe_edge_data::eval_freeze::FreezeRecord;
use fandhe_edge_data::inspect::ValidRecord;
use fandhe_edge_data::split::SplitResult;
use fandhe_edge_eval::acceptance::{AcceptanceVerdict, judge_min_accuracy};
use fandhe_edge_eval::final_test_once::RepresentativeConfigId;
use fandhe_edge_guard::format::{FormatAllowlist, check_bytes};
use fandhe_edge_runtime::capacity::{
    CapacityBreakdown, MAX_FILE_BYTES, PackageComponent, measure_opened_files_with_limit,
};
use fandhe_edge_runtime::capacity_limit::{
    CapacityLimit, CapacityLimitCheck, REFERENCE_CAPACITY_BYTES, check_capacity_limit,
    exceeds_reference,
};
use fandhe_edge_runtime::latency::{
    Clock, LatencyConfig, LatencyError, MonotonicClock, measure_latency,
};
use fandhe_edge_runtime::latency_limit::{LatencyLimit, LatencyLimitCheck, check_latency_limit};
use fandhe_edge_runtime::latency_report::summarize_latency;
use fandhe_edge_runtime::package_outcome::{
    LimitBreach, PackageOutcome, PackageQualityJudgment, resolve_package_outcome,
};
use fandhe_edge_runtime::pipeline::{
    InferencePipeline, MAX_INFER_BATCH_LEN, MAX_INFER_BATCH_TOTAL_BYTES, Preprocessor,
    ScoringBackend,
};
use fandhe_edge_runtime::vocab_exclusion::VOCAB_FILE_NAME;
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::stage_files::SelectionRecord;

use crate::args::PackageArgs;
use crate::error_report::{ToErrorReport, acquire_error_report};
use crate::project::{
    CreatedDir, DEFINITION_FILE, EVALUATION_RECORD_FILE, PACKAGE_DIR, PACKAGE_STAGING_DIR, Project,
    SELECTION_FILE, inspect_bytes, invalid, parse_definition, runtime,
};

use super::baseline::{PreparedBaseline, prepare_baseline, record_matches};
use super::candidate_artifact::{
    ARTIFACT_META_FILE, CandidateArtifact, check_meta_consistency, load_candidate_artifact,
    verify_vocab_file,
};

use super::infer::build_pipeline;
use super::ledger::HeldLedger;
use super::select::compute_selection;
use super::train::{
    candidate_rel, effective_train_seed, load_trained, request_is_smoke_trained,
    request_matches_candidate, resolve_candidates, train_rows, verified_split,
};

/// `package` の結果（終了コードの決定と、stdout へ載せる計測値。#340）。
///
/// `metrics` の各 `exceeded` は runtime の照合結果から写したもので、`outcome.breaches` と一致する
/// （`stage_output::package_outcome_report` が fail-closed で再確認する）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageRunResult {
    /// 終了コード・合否・上限超過の一覧。
    pub outcome: PackageOutcome,
    /// 容量内訳と p95（REQ-30・REQ-31）。
    pub metrics: PackageMetrics,
}

/// `package` を実行する。
///
/// # Errors
/// 選定記録が無い・既存の `package/`・自己整合性の不一致は `invalid_input`（64）、
/// I/O 失敗は `runtime_error`（70）。容量・p95 の上限超過は [`PackageOutcome`]（`limit_exceeded`）。
pub fn run(args: &PackageArgs, cwd: &Path) -> Result<PackageRunResult, ErrorReport> {
    let project = Project::open(cwd, &args.project_dir)?;
    // 副作用（ステージングの作成など）の前に、評価データが凍結記録どおりか確認する（REQ-17）。
    // 評価データがあるプロジェクトは、選定候補の評価完了記録を確認できるまで公開しない
    // （下の [`verify_evaluation_record`]。fail-closed。REQ-27）。
    let frozen = super::inspect::load_frozen_evaluation(&project)?;
    let selection_bytes = project.read(SELECTION_FILE, 64 * 1024)?;
    let selection = SelectionRecord::from_json_slice(&selection_bytes)
        .map_err(|_| invalid("selection record is invalid"))?;
    let definition_bytes = project.read(DEFINITION_FILE, MAX_DEFINITION_FILE_BYTES)?;
    let definition = parse_definition(&definition_bytes)?;
    // 選定記録の `candidate_index` だけで学習結果を読まず、その添字の既定候補の ID・kind が
    // 記録と一致することを確認する（別の候補を配布しない。REQ-27・REQ-39）。
    // 記録された選定結果を信じず、`select` と同じ関数で保存済みの候補結果・分割データから
    // 選定をやり直し、記録と完全一致することを確認する（記録の改変対策。validation のみを使い、
    // 凍結 test には触れない。REQ-27）。
    if compute_selection(&project, &definition)?.as_ref() != Some(&selection) {
        return Err(invalid("selection record does not match the candidate"));
    }
    // 期待する seed・validation 入力は固定値ではなく `split.json` の記録とデータから求める
    // （`compute_selection` が分割を検証済みだが、期待値の組み立てのため同じ検証をもう一度通す）。
    let records = project.load_records(&definition)?;
    let (split, split_seed) = verified_split(&project, &records)?;
    let seed = effective_train_seed(&project, selection.candidate_index, split_seed)?;
    let candidates = resolve_candidates(&project, &definition, selection.candidate_index, seed)?;
    let candidate = candidates
        .get(selection.candidate_index)
        .filter(|c| c.candidate_id == selection.candidate_id)
        .ok_or_else(|| invalid("selection record does not match the candidate"))?;
    let (request, outcome) = load_trained(&project, selection.candidate_index)?
        .ok_or_else(|| invalid("selected candidate is not trained"))?;
    if !request_matches_candidate(&request, &candidate.params, &records, &split) {
        return Err(invalid("selection record does not match the candidate"));
    }
    // 短縮学習（`train --smoke`）の結果は、検証専用の `--allow-smoke` を明示しない限り配布しない
    // （`select` は smoke の結果も選定できるため、ここが配布の関門。REQ-27）。`--allow-smoke` が
    // 緩めるのはこの確認だけで、評価完了の確認（下の [`verify_evaluation_record`]）は緩めない。
    if request_is_smoke_trained(&request, &candidate.params) && !args.allow_smoke {
        return Err(invalid("smoke-trained candidate cannot be packaged"));
    }
    let TrainOutcome::Ok(success) = &outcome else {
        return Err(invalid("selected candidate has no artifact"));
    };
    // 成果物は選定候補のディレクトリ（`candidates/<N>/`）配下から閉じ込めつきで読む
    // （`evaluate` と共有する [`load_candidate_artifact`]。語彙ファイルの保持 fd 1 本によるストリーミング
    // 検証もここで行い、以降の複写・計測は同じ fd を使う。REQ-30・REQ-39）。
    let CandidateArtifact {
        meta_bytes,
        meta,
        onnx_bytes,
        onnx_file,
        handles,
    } = load_candidate_artifact(&project, selection.candidate_index, success)?;
    let onnx_file = onnx_file.as_str();
    let vocab_file = handles.vocab;
    check_meta_consistency(
        &meta,
        &definition,
        request.kind(),
        request.kind_version(),
        request.max_bytes(),
    )?;
    // 公開前に、`infer` が同じパッケージを読むときと同じ検証を通す（ハッシュが一致しても ONNX として
    // 読めないファイルを公開しない）。形式の許可リスト（ガード層）→ `kind_version` の許可リスト・
    // ONNX の読み込み・出力サイズの一致（`infer` の `load_backend` と共通。REQ-39）。
    check_bytes(onnx_bytes.clone(), &FormatAllowlist::onnx_only())
        .map_err(|e| e.to_error_report())?;
    // 組み立てたパイプラインは p95 の計測にも使う（公開するのと同じ `onnx_bytes` から作る。#338）。
    let pipeline = build_pipeline(&onnx_bytes, &meta, definition.options().len())?;

    // 評価データがあるなら、選定候補が評価済みで、記録とモデル・評価データ・定義が一致することを
    // 公開（ステージングの作成）より前に確認する（評価していないモデルを配布しない。REQ-27）。
    // smoke 学習の候補かどうか・`--allow-smoke` の有無に関係なく確認する（smoke の候補は
    // `evaluate` が拒否するため、評価データがあるプロジェクトでは公開できない。fail-closed。REQ-27）。
    let verified_record = match frozen.as_ref() {
        Some((freeze, eval_bytes)) => Some(verify_evaluation_record(
            &project,
            (&selection, &Sha256Digest::of_bytes(&selection_bytes)),
            (freeze, eval_bytes),
            (
                &definition,
                // majority・必要件数は train 分割と定義から計算し直す（評価データは渡さない。#339）。
                prepare_baseline(&definition, &records, &split)?,
            ),
            &meta_bytes,
            &onnx_bytes,
            &format!("{}:seed{}", candidate.candidate_id, seed),
        )?),
        None => None,
    };
    // 合否判定は公開（ステージングの作成）より前に確定する（半端な状態を残さない。#328）。
    let quality = quality_from_acceptance(&definition, verified_record.as_ref())?;

    if project.exists(PACKAGE_DIR)? {
        return Err(invalid("package directory already exists"));
    }
    // p95 の計測・照合（`limits.max_infer_p95_us` があるときだけ。REQ-31・#338）。ステージングを作る前に
    // 行うので、計測の失敗で片付ける分岐が要らず、既存の `package/` がある場合は推論を回す前に失敗する。
    let infer_p95 = match definition.limits().and_then(Limits::max_infer_p95_us) {
        None => None,
        Some(us) => {
            // 定義の検証（`1..=MAX_LIMIT_INFER_P95_US`）で到達しない防御的な変換。
            let limit = us
                .checked_mul(1_000)
                .ok_or_else(|| invalid("latency limit is out of range"))
                .and_then(|ns| {
                    LatencyLimit::from_ns(ns).map_err(|_| invalid("latency limit is out of range"))
                })?;
            let inputs = latency_inputs(&records, &split);
            let check =
                check_latency_with_clock(&pipeline, &inputs, limit, &MonotonicClock::new())?;
            // 写像の失敗もステージングを作る前に返す（公開後に runtime_error を返す経路を作らない）。
            Some((infer_p95_report(&check, us)?, check.breach()))
        }
    };

    // 組み立て・容量計測はステージングで行い、上限内のときだけ `package/` へ原子的に公開する
    // （容量超過のパッケージを `infer --package` で使える場所に残さない。既存の `package/` は
    // 事前に拒否済みで、置き換えも削除もしない。REQ-30・REQ-39）。
    let staging = project.create_dir_tracked(PACKAGE_STAGING_DIR)?;
    // 組み立て・容量計測のどこかで失敗したら、本工程が作ったステージングを片付ける（best effort）。
    // 半端なステージングが残ると再実行が「既存」で拒否されるため。
    let breakdown = match assemble_and_measure(
        &project,
        onnx_file,
        &meta_bytes,
        &onnx_bytes,
        &definition_bytes,
        &meta,
        vocab_file.as_ref().map(|(f, p)| (f, p.as_path())),
    ) {
        Ok(breakdown) => breakdown,
        Err(report) => {
            let _ = project.remove_created_dir(&staging);
            return Err(report);
        }
    };
    // 利用者が設定した上限だけを検証済み型へ変換し、runtime の照合（`check_capacity_limit`）の結果を
    // 渡す。未設定なら `None`（照合しない）。変換失敗（0）は `invalid_input`。ステージングは片付けてから返す。
    let limit = match definition
        .limits()
        .and_then(Limits::max_package_bytes)
        .map(CapacityLimit::from_bytes)
        .transpose()
    {
        Ok(limit) => limit,
        Err(e) => {
            let _ = project.remove_created_dir(&staging);
            return Err(invalid(&e.to_string()));
        }
    };
    let check = check_capacity_limit(&breakdown, limit);
    // 写像（fail-closed）は公開（`finalize_staging`）より前に行い、失敗したらステージングを片付ける。
    let capacity = match capacity_report(&breakdown, &check) {
        Ok(capacity) => capacity,
        Err(report) => {
            let _ = project.remove_created_dir(&staging);
            return Err(report);
        }
    };
    // p95 が超過しても組み立てと容量計測は行い、両方の超過を載せる。
    let mut breaches: Vec<LimitBreach> = check.breach().into_iter().collect();
    breaches.extend(infer_p95.as_ref().and_then(|(_, breach)| *breach));
    let breaches = finalize_staging(&project, &staging, &breaches)?;
    let metrics = PackageMetrics {
        capacity,
        infer_p95: infer_p95.map(|(p95, _)| p95),
    };
    // 上限超過（20）が合否より優先される規則は runtime の `resolve_package_outcome` に任せる。
    // Fail・Undeterminable でも公開の関門は容量・p95 だけ（`finalize_staging` は合否を見ない。#328）。
    Ok(PackageRunResult {
        outcome: resolve_package_outcome(&breaches, quality),
        metrics,
    })
}

/// runtime の容量照合結果と内訳を、stdout 用の値型へ写す（REQ-30・#340）。
///
/// `exceeded` は照合結果（`Within`・`Exceeded`）から決め、`>` を再計算しない。上限未設定
/// （`NotConfigured`）は `limit_bytes:None`・`exceeded:false`。合計が内訳と食い違う・未知の区分は
/// `runtime_error`（fail-closed）。目安 40MB の超過（`over_guideline`）は警告のみ（REQ-30）。
fn capacity_report(
    breakdown: &CapacityBreakdown,
    check: &CapacityLimitCheck,
) -> Result<PackageCapacity, ErrorReport> {
    let (total_bytes, limit_bytes, exceeded) = match check {
        CapacityLimitCheck::Within {
            total_bytes,
            limit_bytes,
        } => (*total_bytes, Some(*limit_bytes), false),
        CapacityLimitCheck::NotConfigured { total_bytes } => (*total_bytes, None, false),
        CapacityLimitCheck::Exceeded(LimitBreach::Capacity {
            measured_bytes,
            limit_bytes,
        }) => (*measured_bytes, Some(*limit_bytes), true),
        _ => return Err(runtime("capacity check is inconsistent")),
    };
    if total_bytes != breakdown.total_bytes() {
        return Err(runtime("capacity check is inconsistent"));
    }
    let size = |c: PackageComponent| {
        let b = breakdown.component(c);
        PackageComponentSize::new(b.bytes, b.file_count)
    };
    Ok(PackageCapacity::new(
        total_bytes,
        limit_bytes,
        exceeded,
        (REFERENCE_CAPACITY_BYTES, exceeds_reference(total_bytes)),
        PackageCapacityComponents::new(
            size(PackageComponent::Weights),
            size(PackageComponent::VocabOrFeatureTransform),
            size(PackageComponent::LabelTable),
            size(PackageComponent::Calibration),
            size(PackageComponent::Metadata),
        ),
    ))
}

/// runtime の p95 照合結果を stdout 用の値型へ写す（REQ-31・#340）。
///
/// `p95_us` は ns を切り上げた値（`limit_ns = limit_us * 1000` なので、`p95_ns > limit_ns` と
/// `p95_us > limit_us` は同値で、切り上げが偽陽性を生まない）。`exceeded` は照合結果から決める。
/// `limit_us` と照合に使った `limit_ns` が食い違う・照合していない・未知の区分は `runtime_error`。
fn infer_p95_report(check: &LatencyLimitCheck, limit_us: u64) -> Result<InferP95, ErrorReport> {
    let (p95_ns, limit_ns, exceeded) = match check {
        LatencyLimitCheck::Within { p95_ns, limit_ns } => (*p95_ns, *limit_ns, false),
        LatencyLimitCheck::Exceeded(LimitBreach::Latency {
            measured_p95_ns,
            limit_ns,
        }) => (*measured_p95_ns, *limit_ns, true),
        _ => return Err(runtime("latency check is inconsistent")),
    };
    if limit_us.checked_mul(1_000) != Some(limit_ns) {
        return Err(runtime("latency check is inconsistent"));
    }
    Ok(InferP95::new(p95_ns.div_ceil(1_000), limit_us, exceeded))
}

/// p95 計測の入力を `train` 分割の `input` から集める（REQ-27・REQ-39。#338）。
///
/// validation・test 分割と凍結した評価データは使わない。件数が [`MAX_INFER_BATCH_LEN`] に達するか、
/// 総バイト数が [`MAX_INFER_BATCH_TOTAL_BYTES`] を超える手前で先頭から切り詰める。空になったときは
/// `measure_latency` が `LatencyError::NoInputs` を返し、`package` は `runtime_error`（70）になる。
fn latency_inputs<'a>(records: &'a [ValidRecord], split: &'a SplitResult) -> Vec<&'a str> {
    let mut inputs = Vec::new();
    let mut total_bytes = 0_usize;
    for row in train_rows(records, split) {
        if inputs.len() >= MAX_INFER_BATCH_LEN {
            break;
        }
        match total_bytes.checked_add(row.input.len()) {
            Some(next) if next <= MAX_INFER_BATCH_TOTAL_BYTES => total_bytes = next,
            _ => break,
        }
        inputs.push(row.input.as_str());
    }
    inputs
}

/// 既定の計測回数（warmup 20・iters 1000）で p95 を計測し、上限との照合結果を返す
/// （超過なら [`LimitBreach::Latency`] を含む。上限内でも p95 の値を残す。REQ-31・REQ-21。#338・#340）。
///
/// 境界規則（`>` で超過・ちょうどは超過でない）は runtime の `check_latency_limit` に任せ、再実装しない。
/// 計測・レポートの失敗は `runtime_error`（`message` は固定語彙の `LatencyError::code()` のみ）。
fn check_latency_with_clock<P: Preprocessor, B: ScoringBackend, C: Clock>(
    pipeline: &InferencePipeline<P, B>,
    inputs: &[&str],
    limit: LatencyLimit,
    clock: &C,
) -> Result<LatencyLimitCheck, ErrorReport> {
    let samples = measure_latency(pipeline, inputs, &LatencyConfig::default(), clock)
        .map_err(|e: LatencyError| runtime(&format!("latency measurement failed: {}", e.code())))?;
    let report = summarize_latency(&samples).map_err(|_| runtime("latency measurement failed"))?;
    Ok(check_latency_limit(&report, Some(limit)))
}

/// 定義の合否基準と照合済みの評価記録から、`package` の合否判定を決める（REQ-24・REQ-33・#328）。
///
/// - 基準が無い: [`PackageQualityJudgment::NotDefined`]（`judgment:null`・exit 0）
/// - 基準があり評価記録が無い（評価データ無し。`--allow-smoke` を含む）: `Undeterminable`
///   （評価していないモデルを合格扱いにしない。REQ-17・REQ-24）
/// - 基準があり照合済みの記録がある: 評価器の [`judge_min_accuracy`]（Wilson 95% 区間）の結果
/// - `baseline_comparison` だけの定義（`acceptance` なし）も基準なし（`NotDefined`。#344）。`verdict` は
///   合否に使わない（承認範囲外。照合は [`verify_evaluation_record`]）
///
/// 記録の `correct` は最終 test の台帳に記録されておらず、プロジェクトへ書き込める主体が
/// `correct <= total` の範囲で書き換えれば判定を変えられる。これは外部台帳による完全性検証
/// （#168）の既知の限界で、本関数はその代替ではない。
fn quality_from_acceptance(
    definition: &Definition,
    record: Option<&EvaluationRecord>,
) -> Result<PackageQualityJudgment, ErrorReport> {
    let Some(acceptance) = definition.acceptance() else {
        return Ok(PackageQualityJudgment::NotDefined);
    };
    let Some(record) = record else {
        return Ok(PackageQualityJudgment::Undeterminable);
    };
    let verdict = judge_min_accuracy(record.correct, record.total, acceptance.min_accuracy_bp())
        .map_err(|_| runtime("cannot judge acceptance"))?;
    Ok(match verdict {
        AcceptanceVerdict::Pass => PackageQualityJudgment::Pass,
        AcceptanceVerdict::Fail => PackageQualityJudgment::Fail,
        AcceptanceVerdict::Undeterminable => PackageQualityJudgment::Undeterminable,
    })
}

/// 評価記録の `baseline_comparison` が定義と一致するか（#339・REQ-25）。
///
/// 定義に欄があるのに記録に無い（欄の削除）、定義に欄が無いのに記録にある（欄の追加）は不一致。
/// 両方にあるときは [`record_matches`] が majority・必要件数・判定を計算し直して照合する。
fn baseline_matches(
    baseline: Option<&PreparedBaseline>,
    record: &EvaluationRecord,
    golds: &[ValidRecord],
) -> bool {
    match (baseline, record.baseline_comparison.as_ref()) {
        (None, None) => true,
        (Some(prepared), Some(rec)) => {
            // `baseline_correct` は凍結した評価データの正解ラベルから数え直した値と照合する。
            let expected = prepared.baseline_correct_on(golds.iter().map(|r| r.label_id.as_str()));
            record_matches(prepared, rec, (record.correct, record.total), expected)
        }
        _ => false,
    }
}

/// 選定候補の評価完了記録を読み、公開する成果物・評価データ・定義と一致することを確認する
/// （REQ-27・#314）。
///
/// 記録が無ければ `evaluation has not been completed`、解析できなければ
/// `evaluation record is invalid`、いずれかの値が一致しなければ
/// `evaluation record does not match the package`（いずれも `invalid_input`）。
///
/// 記録の `config_id`（期待する代表構成 ID）・`total`（凍結した評価データの件数）・`correct <= total`
/// も照合する（他の構成の記録・件数や正解数の改変を公開に使わせない）。
///
/// 記録ファイルに加えて、最終 test の台帳（`final_test_ledger/`。適用ロックと封印つき事前登録）が
/// その代表構成・重みの適用完了を記録していることを照合する（台帳に完了が無ければ
/// `evaluation has not been completed`。記録の偽造だけでは公開できない）。
///
/// 下限基準との比較欄（#339）は、定義と記録で欄の有無が一致することと、majority・必要件数・判定を
/// train 分割と定義から、`baseline_correct` を凍結した評価データから計算し直した値との一致を確認する。
/// `b`・`c` の同量ずらしは検出できない（[`record_matches`] の doc。#168）。判定（`verdict`）は
/// 照合するだけで合否には使わない（#339・#344）。
///
/// **台帳ファイル自体もプロジェクトへ書き込める主体なら丸ごと作り直せる**ため、本確認は外部台帳による
/// 完全性の検証（#168・TASK-39.3-2）の代替ではない。
fn verify_evaluation_record(
    project: &Project,
    (selection, selection_sha256): (&SelectionRecord, &Sha256Digest),
    (freeze, eval_bytes): (&FreezeRecord, &[u8]),
    (definition, baseline): (&Definition, Option<PreparedBaseline>),
    meta_bytes: &[u8],
    onnx_bytes: &[u8],
    config_id: &str,
) -> Result<EvaluationRecord, ErrorReport> {
    let rel = candidate_rel(selection.candidate_index).join(EVALUATION_RECORD_FILE);
    let Some(bytes) = project.read_optional(&rel, MAX_EVALUATION_RECORD_BYTES)? else {
        return Err(invalid("evaluation has not been completed"));
    };
    let record = EvaluationRecord::from_json_slice(&bytes)
        .map_err(|_| invalid("evaluation record is invalid"))?;
    let definition_sha256 = definition
        .canonical_hash()
        .map_err(|_| runtime("cannot hash definition"))?
        .to_hex();
    // 評価件数は凍結した評価データ（照合済みのバイト列）から数え直し、記録の `total` と照合する。
    let eval_records =
        inspect_bytes(eval_bytes, definition).map_err(|_| invalid("evaluation data is invalid"))?;
    let expected_total = eval_records.len();
    let matches = record.candidate_index == selection.candidate_index
        && record.config_id == config_id
        && record.correct <= record.total
        && record
            .type_meaning_quadrant
            .is_none_or(|q| q.total() == Some(record.total))
        && usize::try_from(record.total).is_ok_and(|t| t == expected_total)
        && record.candidate_id == selection.candidate_id
        && record.evaluation_sha256 == freeze.sha256().to_hex()
        && record.evaluation_bytes == freeze.byte_len()
        && record.onnx_sha256 == Sha256Digest::of_bytes(onnx_bytes).to_hex()
        && record.artifact_meta_sha256 == Sha256Digest::of_bytes(meta_bytes).to_hex()
        && record.definition_sha256 == definition_sha256
        && baseline_matches(baseline.as_ref(), &record, &eval_records);
    if !matches {
        return Err(invalid("evaluation record does not match the package"));
    }
    // 記録ファイルだけでは偽造できてしまうため、最終 test の台帳（ロックと封印つき事前登録）が
    // この評価データ × 代表構成 × 重みの適用完了を記録していることを必ず照合する（REQ-27）。
    let config_id =
        RepresentativeConfigId::parse(config_id).map_err(|e| acquire_error_report(&e))?;
    let onnx_digest = Sha256Digest::of_bytes(onnx_bytes);
    let Some(held) = HeldLedger::open(project, false)? else {
        return Err(invalid("evaluation has not been completed"));
    };
    let applied = held
        .ledger()
        .is_applied(&freeze.sha256(), &config_id, &onnx_digest)
        .map_err(|e| acquire_error_report(&e))?;
    if !applied {
        return Err(invalid("evaluation has not been completed"));
    }
    // 最初の適用の前に台帳へ固定した選定が、現在の選定（候補 ID と選定記録のダイジェスト）と
    // 一致することを確認する（選定を書き換えた別候補を配布しない。REQ-27）。
    held.ledger()
        .verify_selection_pin(&freeze.sha256(), &config_id, selection_sha256)
        .map_err(|e| acquire_error_report(&e))?;
    Ok(record)
}

/// 容量・p95 の超過（`breaches`）が無ければステージングを `package/` へ原子的に公開し、あれば公開せず片付ける
/// （REQ-30・REQ-39）。公開に失敗した場合もステージングを片付けてエラーを返す。
///
/// 超過があった場合はその一覧（[`LimitBreach::Capacity`]・[`LimitBreach::Latency`]）を返し、`package/` は作らない。既存の `package/` は
/// 触らない（呼び出し元が事前に不在を確認済み。公開は `RENAME_NOREPLACE` 相当で置き換えない）。
fn finalize_staging(
    project: &Project,
    staging: &CreatedDir,
    breaches: &[LimitBreach],
) -> Result<Vec<LimitBreach>, ErrorReport> {
    // 境界規則（`>` で超過・`==` は超過でない）は runtime の `check_capacity_limit`・`check_latency_limit`
    // （TASK-30.2・#124、TASK-31.x・#338）が決めて `breaches` に載せる。ここでは再実装しない。
    if !breaches.is_empty() {
        let _ = project.remove_created_dir(staging);
        return Ok(breaches.to_vec());
    }
    if let Err(report) = project.publish_dir(PACKAGE_STAGING_DIR, PACKAGE_DIR) {
        let _ = project.remove_created_dir(staging);
        return Err(report);
    }
    Ok(Vec::new())
}

/// ステージングへ 3 ファイル（語彙ファイルがあれば 4）を新規に書き、閉じ込めつきで開いたハンドルで容量を計測する（REQ-30）。
///
/// 呼び出し元（[`run`]）はステージングを作成済みで、失敗時の後始末は呼び出し元が行う。
fn assemble_and_measure(
    project: &Project,
    onnx_file: &str,
    meta_bytes: &[u8],
    onnx_bytes: &[u8],
    definition_bytes: &[u8],
    meta: &ArtifactMeta,
    vocab: Option<(&File, &Path)>,
) -> Result<CapacityBreakdown, ErrorReport> {
    let pkg = Path::new(PACKAGE_STAGING_DIR);
    project.write_new(pkg.join(ARTIFACT_META_FILE), meta_bytes)?;
    project.write_new(pkg.join(onnx_file), onnx_bytes)?;
    project.write_new(pkg.join(DEFINITION_FILE), definition_bytes)?;
    let mut members = vec![
        (PackageComponent::Weights, onnx_file),
        (PackageComponent::LabelTable, DEFINITION_FILE),
        (PackageComponent::Metadata, ARTIFACT_META_FILE),
    ];
    if let Some((mut file, _)) = vocab {
        // 保持 fd を先頭へ戻し、固定長バッファで複写する（全体をメモリへ読まない）。
        file.seek(SeekFrom::Start(0))
            .map_err(|_| runtime("cannot read vocab file"))?;
        project.write_new_from_reader(pkg.join(VOCAB_FILE_NAME), &mut file, MAX_FILE_BYTES)?;
        members.push((PackageComponent::VocabOrFeatureTransform, VOCAB_FILE_NAME));
    }

    let mut files = Vec::new();
    for (component, name) in members {
        let (file, path) = project.open_file(pkg.join(name))?;
        files.push((component, path, file));
    }
    // 公開するのはステージングへ複写したファイルなので、複写後の実体も記録どおりか再検証する
    // （検証から複写までの間の差し替え対策。REQ-39）。
    if vocab.is_some() {
        let (file, path) = project.open_file(pkg.join(VOCAB_FILE_NAME))?;
        verify_vocab_file(meta, Some((&file, path.as_path())))?;
    }
    measure_opened_files_with_limit(&files, MAX_FILE_BYTES)
        .map_err(|e| crate::output::capacity_error_report(&e))
}
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::stages::train::test_support::records_and_split;
    use fandhe_edge_core::evaluation_record::BaselineComparisonVerdict;
    use fandhe_edge_core::stage_report::{
        InferP95, PackageCapacity, PackageCapacityComponents, PackageComponentSize,
    };
    use fandhe_edge_data::split::Split;
    use fandhe_edge_runtime::capacity_limit::CapacityLimitCheck;
    use fandhe_edge_runtime::pipeline::{BackendError, PreprocessError, TokenIds};
    use std::cell::Cell;
    use std::rc::Rc;

    /// 呼び出しごとに共有カウンタを一定量進める偽の時計（`ScoringBackend` と共有する）。
    struct FakeClock(Rc<Cell<u64>>);
    impl Clock for FakeClock {
        fn now_ns(&self) -> u64 {
            self.0.get()
        }
    }

    struct Pre;
    impl Preprocessor for Pre {
        fn preprocess(&self, _input: &str) -> Result<TokenIds, PreprocessError> {
            Ok(TokenIds::new(vec![1]))
        }
    }

    /// 1 回の推論ごとに時計を `step` ns 進める（全サンプルが同値で p95 == `step`）。`fail` なら推論失敗。
    struct TickBackend {
        clock: Rc<Cell<u64>>,
        step: u64,
        fail: bool,
        back: bool,
    }
    impl ScoringBackend for TickBackend {
        fn scores(&self, _ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
            if self.fail {
                return Err(BackendError::Failed);
            }
            if self.back {
                self.clock.set(self.clock.get().saturating_sub(self.step));
            } else {
                self.clock.set(self.clock.get() + self.step);
            }
            Ok(vec![0.75, 0.25])
        }

        fn scores_limited(
            &self,
            ids: &TokenIds,
            _limit: std::time::Duration,
        ) -> Result<Vec<f64>, BackendError> {
            self.scores(ids)
        }
    }

    fn fake_pipeline(
        step: u64,
        fail: bool,
        back: bool,
    ) -> (InferencePipeline<Pre, TickBackend>, FakeClock) {
        let cell = Rc::new(Cell::new(1_000_000_000_000));
        let pipeline = InferencePipeline::new(
            Pre,
            TickBackend {
                clock: cell.clone(),
                step,
                fail,
                back,
            },
        );
        (pipeline, FakeClock(cell))
    }

    /// REQ-31・REQ-21・#338・#340: p95 が上限ちょうどなら超過でなく、1 ns 超えると `LimitBreach::Latency`。
    #[test]
    fn req31_issue338_latency_equal_is_within_and_one_ns_over_is_breach() {
        let limit = LatencyLimit::from_ns(5_000 * 1_000).expect("limit");
        let (pipeline, clock) = fake_pipeline(5_000_000, false, false);
        assert_eq!(
            check_latency_with_clock(&pipeline, &["a"], limit, &clock).expect("measure"),
            LatencyLimitCheck::Within {
                p95_ns: 5_000_000,
                limit_ns: 5_000_000
            }
        );
        let (pipeline, clock) = fake_pipeline(5_000_001, false, false);
        assert_eq!(
            check_latency_with_clock(&pipeline, &["a"], limit, &clock).expect("measure"),
            LatencyLimitCheck::Exceeded(LimitBreach::Latency {
                measured_p95_ns: 5_000_001,
                limit_ns: 5_000_000
            })
        );
    }

    /// 上限 5000 µs で計測し、`infer_p95_report` へ通した結果を返す。
    fn p95_for_step(step: u64) -> InferP95 {
        let limit = LatencyLimit::from_ns(5_000 * 1_000).expect("limit");
        let (pipeline, clock) = fake_pipeline(step, false, false);
        let check = check_latency_with_clock(&pipeline, &["a"], limit, &clock).expect("measure");
        infer_p95_report(&check, 5_000).expect("report")
    }

    /// REQ-31・#340: µs への切り上げの境界。ちょうど・1 ns 超・1000 の倍数でない上限内の 3 点。
    #[test]
    fn req31_issue340_p95_us_ceil_boundaries() {
        assert_eq!(p95_for_step(5_000_000), InferP95::new(5_000, 5_000, false));
        assert_eq!(p95_for_step(5_000_001), InferP95::new(5_001, 5_000, true));
        // 1000 の倍数でない上限内の値は切り上げても超過にならない（偽陽性なし）。
        assert_eq!(p95_for_step(4_999_999), InferP95::new(5_000, 5_000, false));
        assert_eq!(p95_for_step(4_999_001), InferP95::new(5_000, 5_000, false));
    }

    /// REQ-31・#340: `div_ceil` の境界と写像の fail-closed（上限の食い違い・未設定・未知）。
    #[test]
    fn req31_issue340_infer_p95_report_mapping_and_fail_closed() {
        let within = |p95_ns, limit_ns| LatencyLimitCheck::Within { p95_ns, limit_ns };
        for (ns, us) in [(0, 0), (1, 1), (999, 1), (1000, 1), (1001, 2)] {
            assert_eq!(
                infer_p95_report(&within(ns, 5_000_000), 5_000).expect("ok"),
                InferP95::new(us, 5_000, false)
            );
        }
        // u64::MAX でも panic せず切り上げる（超過として写す）。
        let over = LatencyLimitCheck::Exceeded(LimitBreach::Latency {
            measured_p95_ns: u64::MAX,
            limit_ns: 5_000_000,
        });
        assert_eq!(
            infer_p95_report(&over, 5_000).expect("ok"),
            InferP95::new(u64::MAX / 1_000 + 1, 5_000, true)
        );
        // limit_us と limit_ns の不一致・未設定・オーバーフローは runtime_error。
        for bad in [
            infer_p95_report(&within(1, 5_000_000), 4_999),
            infer_p95_report(&LatencyLimitCheck::NotConfigured { p95_ns: 1 }, 5_000),
            infer_p95_report(&within(1, 5_000_000), u64::MAX),
        ] {
            let e = bad.expect_err("fail closed");
            assert_eq!(e.code, fandhe_edge_core::exitcode::ExitCode::RuntimeError);
        }
    }

    /// REQ-30・#340: 容量の写像。上限ちょうどは `exceeded:false`、超過は true、合計不一致・未設定は runtime_error。
    #[test]
    fn req30_issue340_capacity_report_mapping_and_fail_closed() {
        let breakdown = CapacityBreakdown::from_sizes([
            (PackageComponent::Weights, 100),
            (PackageComponent::LabelTable, 20),
            (PackageComponent::Metadata, 5),
        ])
        .expect("breakdown");
        let ok = capacity_report(&breakdown, &check_of(125, 125)).expect("within");
        assert_eq!(ok, capacity_of(125, 125, false));
        let over = capacity_report(&breakdown, &check_of(125, 124)).expect("exceeded");
        assert_eq!(over, capacity_of(125, 124, true));
        // 合計が内訳と食い違う結果は fail-closed。
        assert!(capacity_report(&breakdown, &check_of(126, 200)).is_err());
        assert!(
            capacity_report(
                &breakdown,
                &CapacityLimitCheck::NotConfigured { total_bytes: 126 }
            )
            .is_err()
        );
        // 上限未設定は limit_bytes:None・exceeded:false（目安以下なら over_guideline も false）。
        let unset = capacity_report(
            &breakdown,
            &CapacityLimitCheck::NotConfigured { total_bytes: 125 },
        )
        .expect("not configured");
        assert_eq!(unset.limit_bytes(), None);
        assert!(!unset.exceeded());
        assert!(!unset.over_guideline());
    }

    /// REQ-30: 上限未設定で目安 40MB を超えても警告（`over_guideline:true`）のみで、`exceeded:false`・
    /// breach なし（exit 20 にならない）。目安ちょうどは超過でない。疎ファイルで測る（実データは書かない）。
    #[test]
    fn req30_not_configured_over_guideline_is_warning_only() {
        let dir = std::env::temp_dir().join(format!("fandhe-pkg-guideline-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let measure = |len: u64| {
            let path = dir.join(format!("w{len}.bin"));
            let f = File::create(&path).expect("create");
            f.set_len(len).expect("sparse");
            let f = File::open(&path).expect("open");
            measure_opened_files_with_limit(&[(PackageComponent::Weights, path, f)], MAX_FILE_BYTES)
                .expect("measure")
        };
        for (len, over) in [
            (REFERENCE_CAPACITY_BYTES, false),
            (REFERENCE_CAPACITY_BYTES + 1, true),
        ] {
            let b = measure(len);
            let check = check_capacity_limit(&b, None);
            let cap = capacity_report(&b, &check).expect("report");
            assert_eq!(cap.limit_bytes(), None);
            assert!(!cap.exceeded());
            assert_eq!(cap.over_guideline(), over);
            assert_eq!(check.breach(), None);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn capacity_of(total: u64, limit: u64, exceeded: bool) -> PackageCapacity {
        let c = PackageComponentSize::new;
        PackageCapacity::new(
            total,
            Some(limit),
            exceeded,
            (40_000_000, false),
            PackageCapacityComponents::new(c(100, 1), c(0, 0), c(20, 1), c(0, 0), c(5, 1)),
        )
    }

    /// REQ-30・#123・#340: stdout 用の `components` は `package_capacity_json`（#123）と同じ文字列になる。
    #[test]
    fn req30_issue340_components_match_package_capacity_json() {
        let breakdown = CapacityBreakdown::from_sizes([
            (PackageComponent::Weights, 100),
            (PackageComponent::LabelTable, 20),
            (PackageComponent::Metadata, 5),
        ])
        .expect("breakdown");
        let mapped = capacity_report(&breakdown, &check_of(125, 200)).expect("within");
        let line = fandhe_edge_core::stage_report::PackageReport::pass(PackageMetrics {
            capacity: mapped,
            infer_p95: None,
        })
        .to_json_line()
        .expect("json");
        let legacy = crate::output::package_capacity_json(&breakdown);
        let components = |s: &str| {
            let start = s.find("\"components\":").expect("components");
            let rest = &s[start..];
            // `components` オブジェクトは入れ子 1 段の `}}` までで閉じる。
            let end = rest.find("}}}").expect("end") + 3;
            rest[..end].to_string()
        };
        assert_eq!(components(&line), components(&legacy));
    }

    /// REQ-31・REQ-21・#338: 推論失敗は `runtime_error`（固定語彙の message）。
    #[test]
    fn req31_issue338_latency_inference_failure_is_runtime_error() {
        let limit = LatencyLimit::from_ns(1_000).expect("limit");
        let (pipeline, clock) = fake_pipeline(1, true, false);
        let err = check_latency_with_clock(&pipeline, &["a"], limit, &clock).expect_err("fail");
        assert_eq!(err.code, fandhe_edge_core::exitcode::ExitCode::RuntimeError);
        assert_eq!(err.message, "latency measurement failed: inference_failed");
    }

    /// REQ-31・REQ-21・#338: 時計の逆行は `runtime_error`。
    #[test]
    fn req31_issue338_latency_non_monotonic_clock_is_runtime_error() {
        let limit = LatencyLimit::from_ns(1_000).expect("limit");
        let (pipeline, clock) = fake_pipeline(10, false, true);
        let err = check_latency_with_clock(&pipeline, &["a"], limit, &clock).expect_err("fail");
        assert_eq!(err.code, fandhe_edge_core::exitcode::ExitCode::RuntimeError);
        assert_eq!(
            err.message,
            "latency measurement failed: non_monotonic_clock"
        );
    }

    /// REQ-27・#338: 計測入力は Train の `input` だけで、Validation・Test は含まれない。
    #[test]
    fn req27_issue338_latency_inputs_use_train_split_only() {
        let (records, split) = records_and_split(&[
            ("a", Split::Train),
            ("b", Split::Validation),
            ("c", Split::Test),
            ("d", Split::Train),
        ]);
        let inputs = latency_inputs(&records, &split);
        assert_eq!(inputs, vec!["input-a", "input-d"]);
        assert!(!inputs.contains(&"input-b"));
        assert!(!inputs.contains(&"input-c"));
    }

    /// REQ-39・#338: 件数の上限を超える Train は先頭から切り詰められ、長さがちょうど上限になる。
    #[test]
    fn req39_issue338_latency_inputs_truncate_to_batch_len_limit() {
        let ids: Vec<String> = (0..=MAX_INFER_BATCH_LEN).map(|i| i.to_string()).collect();
        let rows: Vec<(&str, Split)> = ids.iter().map(|id| (id.as_str(), Split::Train)).collect();
        let (records, split) = records_and_split(&rows);
        let inputs = latency_inputs(&records, &split);
        assert_eq!(inputs.len(), MAX_INFER_BATCH_LEN);
        assert_eq!(inputs.first().copied(), Some("input-0"));
    }

    /// REQ-31・REQ-30・#338: 容量は上限内でも p95 の超過があれば公開せず、ステージングも残さない。
    #[test]
    fn req31_issue338_latency_breach_leaves_no_package_and_no_staging() {
        let (cwd, project, staging) = setup("latency");
        let breach = LimitBreach::Latency {
            measured_p95_ns: 2,
            limit_ns: 1,
        };
        let breaches = finalize_staging(&project, &staging, &[breach]).expect("finalize");
        assert_eq!(breaches, vec![breach]);
        assert!(!project.exists(PACKAGE_DIR).expect("exists"));
        assert!(!project.exists(PACKAGE_STAGING_DIR).expect("exists"));
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// 測定値と上限から `check_capacity_limit` 相当の照合結果を作る（境界規則は runtime 側の 1 箇所）。
    fn check_of(measured_bytes: u64, limit_bytes: u64) -> CapacityLimitCheck {
        match LimitBreach::capacity_if_exceeded(measured_bytes, limit_bytes) {
            Some(b) => CapacityLimitCheck::Exceeded(b),
            None => CapacityLimitCheck::Within {
                total_bytes: measured_bytes,
                limit_bytes,
            },
        }
    }

    /// 一時 cwd の下に `proj/` を新規作成し、ステージングにファイルを 1 つ置いて返す。
    fn setup(case: &str) -> (std::path::PathBuf, Project, CreatedDir) {
        let cwd =
            std::env::temp_dir().join(format!("package-staging-{case}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cwd);
        std::fs::create_dir_all(&cwd).expect("cwd");
        let project = Project::create(&cwd, Path::new("proj")).expect("project");
        let staging = project
            .create_dir_tracked(PACKAGE_STAGING_DIR)
            .expect("staging");
        project
            .write_new(Path::new(PACKAGE_STAGING_DIR).join("artifact.json"), b"new")
            .expect("write");
        (cwd, project, staging)
    }

    /// REQ-30・REQ-39: 上限超過では `package/` を作らず、ステージングも残さない。
    #[test]
    fn req30_over_limit_leaves_no_package_and_no_staging() {
        let (cwd, project, staging) = setup("over");
        let breaches = finalize_staging(
            &project,
            &staging,
            &check_of(41, 40).breach().into_iter().collect::<Vec<_>>(),
        )
        .expect("finalize");
        assert_eq!(
            breaches,
            vec![LimitBreach::Capacity {
                measured_bytes: 41,
                limit_bytes: 40
            }]
        );
        assert!(!project.exists(PACKAGE_DIR).expect("exists"));
        assert!(!project.exists(PACKAGE_STAGING_DIR).expect("exists"));
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-30・REQ-39: 上限ちょうど（超過でない）なら `package/` として公開され、ステージングは消える。
    #[test]
    fn req30_within_limit_publishes_package() {
        let (cwd, project, staging) = setup("within");
        let breaches = finalize_staging(
            &project,
            &staging,
            &check_of(40, 40).breach().into_iter().collect::<Vec<_>>(),
        )
        .expect("finalize");
        assert_eq!(breaches, Vec::new());
        assert!(!project.exists(PACKAGE_STAGING_DIR).expect("exists"));
        let bytes = project
            .read(Path::new(PACKAGE_DIR).join("artifact.json"), 16)
            .expect("read published");
        assert_eq!(bytes, b"new");
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-30・REQ-39: 超過時に以前公開済みの `package/` があっても消さず・置き換えない。
    #[test]
    fn req30_over_limit_keeps_previously_published_package() {
        let (cwd, project, staging) = setup("keep");
        // 以前の公開物（ステージングの名前替えより前に存在した状態を模擬）。
        project.create_dir(PACKAGE_DIR).expect("old package");
        project
            .write_new(Path::new(PACKAGE_DIR).join("artifact.json"), b"old")
            .expect("old file");
        let breaches = finalize_staging(
            &project,
            &staging,
            &check_of(41, 40).breach().into_iter().collect::<Vec<_>>(),
        )
        .expect("finalize");
        assert_eq!(breaches.len(), 1);
        let bytes = project
            .read(Path::new(PACKAGE_DIR).join("artifact.json"), 16)
            .expect("read old");
        assert_eq!(bytes, b"old");
        assert!(!project.exists(PACKAGE_STAGING_DIR).expect("exists"));
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// REQ-39: 公開先が既にあれば（空でも）置き換えず、ステージングを片付けて `invalid_input`。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn req39_publish_does_not_replace_existing_package() {
        let (cwd, project, staging) = setup("noreplace");
        project.create_dir(PACKAGE_DIR).expect("empty package");
        let err = finalize_staging(
            &project,
            &staging,
            &check_of(1, 40).breach().into_iter().collect::<Vec<_>>(),
        )
        .expect_err("must not replace");
        assert_eq!(err.code, fandhe_edge_core::exitcode::ExitCode::InvalidInput);
        assert!(!project.exists(PACKAGE_STAGING_DIR).expect("exists"));
        let _ = std::fs::remove_dir_all(&cwd);
    }

    fn definition_with(acceptance: Option<u32>) -> Definition {
        definition_with_baseline(acceptance, false)
    }

    fn definition_with_baseline(acceptance: Option<u32>, baseline: bool) -> Definition {
        let mut extra = acceptance
            .map(|bp| format!(r#","acceptance":{{"min_accuracy_bp":{bp}}}"#))
            .unwrap_or_default();
        if baseline {
            extra.push_str(
                r#","baseline_comparison":{"assumed_p_b_bp":9500,"assumed_p_c_bp":0,"power_bp":8000}"#,
            );
        }
        let json = format!(
            r#"{{"schema":"fandhe-edge-model-definition/v1","name":"t","version":1,"judgment_type":"single_select","options":[{{"id":"a","display_name":"A","description":"a"}}],"io":{{"input":"bytes"}}{extra}}}"#
        );
        Definition::parse(&json).expect("definition")
    }

    fn record_with(correct: u64, total: u64) -> EvaluationRecord {
        EvaluationRecord {
            candidate_index: 0,
            candidate_id: "c".to_string(),
            config_id: "c:seed1".to_string(),
            evaluation_sha256: "0".repeat(64),
            evaluation_bytes: 1,
            onnx_sha256: "0".repeat(64),
            artifact_meta_sha256: "0".repeat(64),
            definition_sha256: "0".repeat(64),
            correct,
            total,
            baseline_comparison: None,
            predictions_sha256: None,
            calibration: None,
            type_meaning_quadrant: None,
        }
    }

    /// REQ-24・REQ-33・#328: 基準なしは `NotDefined`、基準ありで記録なしは `Undeterminable`、
    /// 基準ありで記録ありは評価器の判定（n=12 の Wilson 区間。手計算値）を写す。
    #[test]
    fn req24_issue328_quality_from_acceptance_maps_all_cases() {
        let record = record_with(12, 12);
        assert_eq!(
            quality_from_acceptance(&definition_with(None), Some(&record)),
            Ok(PackageQualityJudgment::NotDefined)
        );
        assert_eq!(
            quality_from_acceptance(&definition_with(Some(7500)), None),
            Ok(PackageQualityJudgment::Undeterminable)
        );
        assert_eq!(
            quality_from_acceptance(&definition_with(Some(7500)), Some(&record)),
            Ok(PackageQualityJudgment::Pass)
        );
        assert_eq!(
            quality_from_acceptance(&definition_with(Some(2500)), Some(&record_with(0, 12))),
            Ok(PackageQualityJudgment::Fail)
        );
        assert_eq!(
            quality_from_acceptance(&definition_with(Some(5000)), Some(&record_with(6, 12))),
            Ok(PackageQualityJudgment::Undeterminable)
        );
    }

    fn with_verdict(verdict: Option<BaselineComparisonVerdict>) -> EvaluationRecord {
        let mut r = record_with(12, 12);
        r.baseline_comparison =
            verdict.map(
                |verdict| fandhe_edge_core::evaluation_record::BaselineComparisonRecord {
                    majority_label: "a".to_string(),
                    baseline_correct: 6,
                    b: 6,
                    c: 0,
                    required_n: 7,
                    verdict,
                },
            );
        r
    }

    /// REQ-24・REQ-25・REQ-33・#344: 下限基準の `verdict` は合否に使わない。`acceptance` と併記しても
    /// 結果は正解率だけで決まり、下限基準だけの定義は基準なし（`NotDefined`）になる。
    #[test]
    fn req25_issue344_quality_ignores_baseline_verdict() {
        use BaselineComparisonVerdict as V;
        use PackageQualityJudgment as Q;
        let verdicts = [
            Some(V::SignificantlyBetter),
            Some(V::NotSignificantlyBetter),
            Some(V::Undeterminable),
            None,
        ];
        let only_baseline = definition_with_baseline(None, true);
        for (bp, correct, expected) in [
            (7500, 12, Q::Pass),
            (2500, 0, Q::Fail),
            (5000, 6, Q::Undeterminable),
        ] {
            let both = definition_with_baseline(Some(bp), true);
            for verdict in verdicts {
                let mut record = with_verdict(verdict);
                record.correct = correct;
                assert_eq!(
                    quality_from_acceptance(&both, Some(&record)),
                    Ok(expected),
                    "{bp} {correct} {verdict:?}"
                );
            }
        }
        for verdict in verdicts {
            assert_eq!(
                quality_from_acceptance(&only_baseline, Some(&with_verdict(verdict))),
                Ok(Q::NotDefined),
                "{verdict:?}"
            );
        }
        assert_eq!(
            quality_from_acceptance(&only_baseline, None),
            Ok(Q::NotDefined)
        );
    }

    fn meta_with_vocab(vocab_sha: Option<&str>) -> ArtifactMeta {
        let extra = vocab_sha
            .map(|h| format!(r#","vocab_sha256":"{h}""#))
            .unwrap_or_default();
        let json = format!(
            r#"{{"onnx_file":"m.onnx","kind":"c1","kind_version":1,"max_bytes":48,"label_order":["a"],"onnx_sha256":"{}"{extra}}}"#,
            "0".repeat(64)
        );
        ArtifactMeta::parse(json.as_bytes()).expect("meta")
    }

    fn temp_vocab(tag: &str, bytes: &[u8]) -> (std::path::PathBuf, File) {
        let path = std::env::temp_dir().join(format!("fandhe-vocab-{tag}-{}", std::process::id()));
        std::fs::write(&path, bytes).expect("write");
        let file = File::open(&path).expect("open");
        (path, file)
    }

    /// REQ-39: 語彙ファイルは記録ハッシュとの一致と許可形式の両方を満たすときだけ通す
    /// （保持 fd からのストリーミング検証。select・package・infer で共有）。
    #[test]
    fn req39_verify_vocab_file_checks_hash_and_format() {
        let good = br#"{"a":0}"#;
        let good_hex = Sha256Digest::of_bytes(good).to_hex();
        let (p1, f1) = temp_vocab("good", good);
        assert_eq!(verify_vocab_file(&meta_with_vocab(None), None), Ok(None));
        assert_eq!(
            verify_vocab_file(&meta_with_vocab(Some(&good_hex)), Some((&f1, &p1))),
            Ok(Some(7))
        );
        // 記録なし・ファイルなし・ハッシュ不一致は拒否。
        assert!(verify_vocab_file(&meta_with_vocab(None), Some((&f1, &p1))).is_err());
        assert!(verify_vocab_file(&meta_with_vocab(Some(&good_hex)), None).is_err());
        let wrong = meta_with_vocab(Some(&"1".repeat(64)));
        assert!(verify_vocab_file(&wrong, Some((&f1, &p1))).is_err());
        // ハッシュが一致しても形式が許可外・末尾に余分なデータなら拒否。
        for (tag, bad) in [("bad", &b"not json"[..]), ("trail", br#"{"a":0} x"#)] {
            let hex = Sha256Digest::of_bytes(bad).to_hex();
            let (p, f) = temp_vocab(tag, bad);
            let err = verify_vocab_file(&meta_with_vocab(Some(&hex)), Some((&f, &p))).unwrap_err();
            assert_eq!(err.code, fandhe_edge_core::exitcode::ExitCode::InvalidInput);
            let _ = std::fs::remove_file(&p);
        }
        let _ = std::fs::remove_file(&p1);
    }
}
