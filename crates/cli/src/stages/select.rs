//! `select` 工程: 学習済みの候補から validation 正解率が最も高い候補を選ぶ
//! （REQ-18・REQ-27・REQ-33・TASK-33.1-2・#136）。
//!
//! # 手順
//!
//! 各候補の `request.json`・`result.json` を再検証つきで読み戻し（[`super::train::load_trained`]）、
//! 学習ジョブが返した validation 予測を、`train.jsonl` の正解ラベルと突き合わせて正解率を出す
//! （評価器を呼ぶだけで再実装しない。[`fandhe_edge_train::stage_files::validation_accuracy`]）。
//! 最高正解率の候補を [`select_best`]（同率は宣言順で先。乱数なし）で選び、`selection_record.json`
//! へ記録する。学習済みの候補が 1 件も無ければ `pending`（12）。
//!
//! 採点の前に `split.json` をデータから再現して照合し（[`super::train::verified_split`]）、各候補の
//! 保存済み `request.json` の validation（id・input）が分割記録の validation 全体と一致することを
//! 確認する（不一致は `invalid_input`。記録の改変による部分集合での選定を防ぐ。REQ-27）。
//!
//! 開始時（記録の書き込み前）に評価データの凍結ハッシュを確認し、不一致・凍結記録の欠落は
//! `invalid_input` で停止する（[`super::inspect::ensure_evaluation_frozen`]。REQ-17）。
//!
//! # 未接続
//!
//! McNemar・Holm による選定結果の有意性判定（[`fandhe_edge_train::selection_significance`]）は
//! 本工程に**未接続**で、記録にも含めない（合否には使わない。REQ-25）。最終 test は使わない
//! （REQ-27。validation のみで選ぶ）。

use std::collections::BTreeMap;
use std::path::Path;

use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::SelectReport;
use fandhe_edge_guard::format::{FormatAllowlist, check_bytes};
use fandhe_edge_runtime::capacity::{
    MAX_FILE_BYTES, PackageComponent, measure_opened_files_with_limit,
};
use fandhe_edge_runtime::onnx::ModelKind;
use fandhe_edge_runtime::vocab_exclusion::{VOCAB_GUIDELINE_BYTES, screen_vocab_candidates};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::search::{EvaluatedCandidate, SelectionDecision, select_best};
use fandhe_edge_train::stage_files::{
    ExcludedCandidate, SelectionExclusions, SelectionRecord, validation_accuracy,
};

use crate::args::SelectArgs;
use crate::error_report::{ToErrorReport, default_message};
use crate::project::{
    DEFINITION_FILE, Project, SELECTION_EXCLUSIONS_FILE, SELECTION_FILE, fail, invalid, runtime,
};

use super::candidate_artifact::{
    CandidateArtifact, check_meta_consistency, load_candidate_artifact,
};
use super::infer::load_backend;
use super::train::{load_trained, request_matches_candidate, resolve_candidates, verified_split};

/// `select` を実行する。
///
/// # Errors
/// 学習済みの候補が無ければ `pending`（12）、既存の選定記録があれば `invalid_input`（64）、
/// 予測と入力の不一致・採点失敗は `runtime_error`（70）。
pub fn run(args: &SelectArgs, cwd: &Path) -> Result<SelectReport, ErrorReport> {
    let project = Project::open(cwd, &args.project_dir)?;
    // 副作用（学習・選定・書き出し）の前に、評価データが凍結記録どおりか確認する（REQ-17）。
    super::inspect::ensure_evaluation_frozen(&project)?;
    if project.exists(SELECTION_FILE)? {
        return Err(crate::project::invalid("selection record already exists"));
    }
    let definition = project.load_definition()?;
    let (record, excluded) = compute_selection_with_exclusions(&project, &definition)?;
    let Some(record) = record else {
        // 全候補が容量の目安超過で除外された場合は、除外理由をメッセージに残し、内部記録
        // `selection_exclusions.json` にも保存する（`selection_record.json` は作らない。他工程は
        // 読まない。REQ-30・TASK-30.3・#125）。
        if excluded.is_empty() {
            return Err(fail(ExitCode::Pending, default_message(ExitCode::Pending)));
        }
        let json = SelectionExclusions {
            excluded_candidates: excluded.clone(),
        }
        .to_json_vec()
        .map_err(|_| runtime("cannot serialize selection exclusions"))?;
        project.replace_file(SELECTION_EXCLUSIONS_FILE, &json)?;
        return Err(fail(ExitCode::Pending, &all_excluded_message(&excluded)));
    };
    let json = record
        .to_json_vec()
        .map_err(|_| runtime("cannot serialize selection record"))?;
    // 選定できたので、以前の全件除外で残った内部記録は消す（古い除外結果を残さない。削除に失敗したら
    // 選定記録を書かずに止める。fail-closed）。
    project.remove_file_if_exists(SELECTION_EXCLUSIONS_FILE)?;
    project.write_new(SELECTION_FILE, &json)?;
    Ok(SelectReport::new(
        record.candidate_index,
        record.candidate_id,
    ))
}

/// 保存済みの候補結果と分割データから、選定結果を計算する（`select` の記録と、`package` の
/// 記録の再検証が同じ関数を使う。選定ロジックを複製しない。REQ-27）。
///
/// 選定は validation のみで行い、凍結した最終 test・評価データは使わない。学習済みの候補が無ければ
/// `None`（`select` は `pending`）。
///
/// # Errors
/// 分割記録・保存済みリクエストの不整合は `invalid_input`、採点失敗は `runtime_error`。
pub fn compute_selection(
    project: &Project,
    definition: &Definition,
) -> Result<Option<SelectionRecord>, ErrorReport> {
    compute_selection_with_exclusions(project, definition).map(|(record, _)| record)
}

/// 全候補が容量の目安超過で除外されたときの `pending` メッセージ（英語。除外理由を残す）。
fn all_excluded_message(excluded: &[ExcludedCandidate]) -> String {
    let detail: Vec<String> = excluded
        .iter()
        .map(|e| {
            format!(
                "{}:{} {} ({} > {} bytes)",
                e.candidate_index, e.candidate_id, e.reason, e.total_bytes, e.guideline_bytes
            )
        })
        .collect();
    format!(
        "no eligible candidate: all trained candidates were excluded ({})",
        detail.join(", ")
    )
}

/// [`compute_selection`] と同じ選定に加え、容量の目安超過で除外した候補も返す
/// （選定候補が無いときも除外理由を呼び出し元へ渡すため。REQ-30・TASK-30.3・#125）。
///
/// # Errors
/// [`compute_selection`] と同じ。
pub fn compute_selection_with_exclusions(
    project: &Project,
    definition: &Definition,
) -> Result<(Option<SelectionRecord>, Vec<ExcludedCandidate>), ErrorReport> {
    let records = project.load_records(definition)?;
    let gold: BTreeMap<&str, &str> = records
        .iter()
        .map(|r| (r.id.as_str(), r.label_id.as_str()))
        .collect();
    // 分割記録をデータから再現して照合する。保存済みの `request.json` の validation を信用すると、
    // 記録の改変で任意の部分集合の正解率により候補を選べてしまう（REQ-27）。
    let (split, seed) = verified_split(project, &records)?;
    let labels: Vec<&str> = definition.options().iter().map(|c| c.id.as_str()).collect();

    let mut evaluated = Vec::new();
    let mut excluded = Vec::new();
    // 候補 N ごとに `train` と同じ関数で `root`・`out_dir` を組み立てた既定候補を用意する
    // （保存済みリクエストの `root`・`out_dir` の照合に使う。REQ-39）。
    let candidate_count = resolve_candidates(project, definition, 0, seed)?.len();
    let candidates = (0..candidate_count)
        .map(|i| {
            let mut all = resolve_candidates(project, definition, i, seed)?;
            if i < all.len() {
                Ok(all.swap_remove(i))
            } else {
                Err(runtime("candidate is not available"))
            }
        })
        .collect::<Result<Vec<_>, ErrorReport>>()?;
    for (index, candidate) in candidates.iter().enumerate() {
        let Some((request, outcome)) = load_trained(project, index)? else {
            continue;
        };
        let TrainOutcome::Ok(success) = &outcome else {
            continue;
        };
        // 保存済みのリクエストが既定候補 N の種類・構成と一致すること（別の種類の学習結果を
        // 候補 N として採点しない。記録の差し替え対策。REQ-27）。
        if !request_matches_candidate(&request, &candidate.params, &records, &split) {
            return Err(invalid("train request does not match the candidate"));
        }
        let inputs = request
            .validation_inputs()
            .ok_or_else(|| runtime("train result has no validation inputs"))?;
        let predictions = success
            .validation_predictions()
            .ok_or_else(|| runtime("train result has no validation predictions"))?;
        // request の validation 入力は、分割記録の validation 全体と一致済み（上の照合が期待する
        // リクエストを丸ごと組み立てて比べる。記録の改変による部分集合での選定を防ぐ。REQ-27）。
        let ids: Vec<&str> = inputs.iter().map(|v| v.id()).collect();
        let gold_labels = ids
            .iter()
            .map(|id| gold.get(id).copied())
            .collect::<Option<Vec<&str>>>()
            .ok_or_else(|| runtime("validation record is missing"))?;
        let accuracy = validation_accuracy(&labels, &ids, &gold_labels, predictions)
            .map_err(|_| runtime("cannot score validation predictions"))?;
        // 整合性の確認（上の validation 入力・予測・正解率）を通した候補にだけ、語彙ファイルの
        // 検証と容量による除外を適用する（改ざん候補を「容量超過で除外」として正常扱いしない。
        // 語彙ファイルは記録ハッシュ・形式も照合する。REQ-30・REQ-39・TASK-30.3・#125。
        // 失敗時は除外せず処理全体を止める）。
        if let Some(entry) =
            vocab_exclusion_of(project, definition, index, candidate, success, &request)?
        {
            excluded.push(entry);
            continue;
        }
        evaluated.push((index, candidate.candidate_id.as_str(), accuracy));
    }

    let inputs: Vec<EvaluatedCandidate<'_>> = evaluated
        .iter()
        .map(|(_, id, accuracy)| EvaluatedCandidate {
            candidate_id: id,
            accuracy: *accuracy,
        })
        .collect();
    let decision = select_best(&inputs).map_err(|_| runtime("cannot select a candidate"))?;
    let SelectionDecision::Selected {
        candidate_id,
        validation_accuracy: accuracy,
        rule,
        ..
    } = decision
    else {
        return Ok((None, excluded));
    };
    let Some((index, _, _)) = evaluated.iter().find(|(_, id, _)| *id == candidate_id) else {
        return Err(runtime("selected candidate is not evaluated"));
    };
    Ok((
        Some(SelectionRecord {
            candidate_index: *index,
            candidate_id,
            rule,
            validation_correct: accuracy.correct,
            validation_total: accuracy.total,
            excluded_candidates: excluded.clone(),
        }),
        excluded,
    ))
}

/// 学習済みの候補の成果物を検証したうえで、パッケージ相当の容量（ONNX・語彙ファイル〔あれば〕・
/// 選択肢表 `definition.json`・`artifact.json`）を計測し、語彙超過構成なら除外記録を返す
/// （REQ-30・REQ-39・TASK-30.3・#125）。
///
/// 検証は除外判定より前に、除外されない候補と同じ内容を行う。
/// - ONNX: `artifact.json` 記載の sha256 とのストリーミング照合・形式の許可リスト・`kind`・
///   `kind_version` の許可リストと学習リクエストとの一致（`package` の公開前検証と同じ観点）。
/// - 語彙ファイル: 記録 sha256 とのストリーミング照合（固定長バッファ。全体をメモリへ読まない）。
///   同じストリームで JSON 形式も検証する（[`verify_vocab_file`]。`package`・`infer` と共通）。容量を
///   超える語彙も同じ検証を通し、形式不正なら除外にせず `invalid_input`、形式が正しい超過だけを除外する。
///   語彙は保持 fd 1 本を検証と計測で使い回し（開き直さない）、メモリ使用量はファイルサイズに比例しない。
///
/// 構成要素は `package` 工程の組み立て（`assemble_and_measure`）と揃える。語彙ファイルの有無は
/// 成果物ディレクトリの [`VOCAB_FILE_NAME`] で判定する（現行の既定候補 c1・c3 は語彙を ONNX グラフ内に
/// 持ち、このファイルを作らない）。判定は runtime の [`screen_vocab_candidates`] に集約し再実装
/// しない。検証・計測の失敗は除外にせずエラーで返す（fail-closed。REQ-39）。
fn vocab_exclusion_of(
    project: &Project,
    definition: &Definition,
    index: usize,
    candidate: &fandhe_edge_train::search::SearchCandidate,
    success: &fandhe_edge_train::result::SuccessOutcome,
    request: &fandhe_edge_train::request::TrainRequest,
) -> Result<Option<ExcludedCandidate>, ErrorReport> {
    // 成果物の読み込み（記録 sha256 との ONNX 照合・語彙の保持 fd によるストリーミング検証）は
    // `package`・`evaluate` と共有する [`load_candidate_artifact`]。除外判定より前に、除外されない
    // 候補と同じ整合性確認（kind・kind_version・label_order・max_bytes・ONNX 形式）も通す。
    let CandidateArtifact {
        meta,
        onnx_bytes,
        handles,
        ..
    } = load_candidate_artifact(project, index, success)?;
    check_meta_consistency(
        &meta,
        definition,
        request.kind(),
        request.kind_version(),
        request.max_bytes(),
    )?;
    // `package`・`infer` と共有する `load_backend` で、ONNX として読み込めること・出力クラス数が
    // 定義の選択肢数と一致すること・`kind_version` の許可も確認する（容量超過の候補も同じ。
    // 失敗は除外にせず、`package` と同じ終了コードで止める。REQ-30・REQ-39）。
    let checked =
        check_bytes(onnx_bytes, &FormatAllowlist::onnx_only()).map_err(|e| e.to_error_report())?;
    let kind = ModelKind::parse(meta.kind()).map_err(|_| invalid("unsupported model kind"))?;
    load_backend(
        checked.as_bytes(),
        kind,
        meta.kind_version(),
        definition.options().len(),
    )?;
    // 語彙の保持 fd 1 本を検証（ローダー内）と容量計測（fstat）の両方で使い、開き直さない。
    // 容量を超える語彙も同じ形式検証を通し済みで、不正なら除外にせず `invalid_input` で止まる。
    let has_vocab_file = handles.vocab.is_some();
    let mut files = vec![
        (PackageComponent::Weights, handles.onnx.1, handles.onnx.0),
        (PackageComponent::Metadata, handles.meta.1, handles.meta.0),
    ];
    if let Some((file, real)) = handles.vocab {
        files.push((PackageComponent::VocabOrFeatureTransform, real, file));
    }
    // 選択肢表（`package` は `definition.json` を LabelTable として合計に含める）。
    let (file, path) = project.open_file(DEFINITION_FILE)?;
    files.push((PackageComponent::LabelTable, path, file));
    let breakdown = measure_opened_files_with_limit(&files, MAX_FILE_BYTES)
        .map_err(|e| crate::output::capacity_error_report(&e))?;
    let screening = screen_vocab_candidates(&[(index, breakdown, has_vocab_file)])
        .map_err(|_| runtime("cannot screen candidate capacity"))?;
    Ok(screening.excluded().first().map(|r| ExcludedCandidate {
        candidate_index: index,
        candidate_id: candidate.candidate_id.clone(),
        reason: r.code().unwrap_or_default().to_string(),
        total_bytes: r.total_bytes,
        guideline_bytes: VOCAB_GUIDELINE_BYTES,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-30・TASK-30.3: 全候補が除外されたときの `pending` メッセージに除外理由が残る。
    #[test]
    fn req30_all_excluded_message_keeps_reason_and_sizes() {
        let excluded = vec![ExcludedCandidate {
            candidate_index: 1,
            candidate_id: "c9".to_string(),
            reason: "vocab_package_over_guideline".to_string(),
            total_bytes: 41_000_000,
            guideline_bytes: 40_000_000,
        }];
        assert_eq!(
            all_excluded_message(&excluded),
            "no eligible candidate: all trained candidates were excluded \
             (1:c9 vocab_package_over_guideline (41000000 > 40000000 bytes))"
        );
    }
}
