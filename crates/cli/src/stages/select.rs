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
use std::io::ErrorKind;
use std::path::Path;

use fandhe_edge_core::artifact_meta::{ArtifactMeta, MAX_ARTIFACT_META_BYTES};
use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::read_bounded_open_file;
use fandhe_edge_core::stage_report::SelectReport;
use fandhe_edge_guard::path::PathRejection;
use fandhe_edge_runtime::capacity::{
    MAX_FILE_BYTES, PackageComponent, measure_opened_files_with_limit,
};
use fandhe_edge_runtime::vocab_exclusion::{VOCAB_FILE_NAME, screen_vocab_candidates};
use fandhe_edge_train::result::TrainOutcome;
use fandhe_edge_train::search::{EvaluatedCandidate, SelectionDecision, select_best};
use fandhe_edge_train::stage_files::{ExcludedCandidate, SelectionRecord, validation_accuracy};

use crate::args::SelectArgs;
use crate::error_report::{ToErrorReport, default_message};
use crate::project::{DEFINITION_FILE, Project, SELECTION_FILE, fail, fs_report, invalid, runtime};

use super::package::verify_vocab_member;
use super::train::{
    candidate_rel, load_trained, request_matches_candidate, resolve_candidates, verified_split,
};

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
        // 全候補が容量の目安超過で除外された場合は、除外理由を失わないようメッセージに残す
        // （記録は作らない。REQ-30・TASK-30.3・#125）。
        if excluded.is_empty() {
            return Err(fail(ExitCode::Pending, default_message(ExitCode::Pending)));
        }
        return Err(fail(ExitCode::Pending, &all_excluded_message(&excluded)));
    };
    let json = record
        .to_json_vec()
        .map_err(|_| runtime("cannot serialize selection record"))?;
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
        if let Some(entry) = vocab_exclusion_of(project, index, candidate, success)? {
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

/// 学習済みの候補のパッケージ相当の容量（ONNX・語彙ファイル〔あれば〕・選択肢表
/// `definition.json`・`artifact.json`）を計測し、語彙超過構成なら除外記録を返す
/// （REQ-30・TASK-30.3・#125）。構成要素は `package` 工程の組み立て（`assemble_and_measure`）と
/// 揃え、選定時に 40MB 以下でも配布時に超過する候補を見逃さない。語彙ファイルの有無は成果物
/// ディレクトリの [`VOCAB_FILE_NAME`] で判定する（現行の既定候補 c1・c3 は語彙を ONNX グラフ内に
/// 持ち、このファイルを作らない）。判定は runtime の [`screen_vocab_candidates`] に集約し再実装
/// しない。計測の失敗は除外にせずエラーで返す（fail-closed。REQ-39）。
fn vocab_exclusion_of(
    project: &Project,
    index: usize,
    candidate: &fandhe_edge_train::search::SearchCandidate,
    success: &fandhe_edge_train::result::SuccessOutcome,
) -> Result<Option<ExcludedCandidate>, ErrorReport> {
    let candidate_dir = project.open_subdir(candidate_rel(index))?;
    let artifact_rel = Path::new(success.artifact_dir())
        .strip_prefix(candidate_dir.dir())
        .map_err(|_| invalid("artifact directory is outside the candidate directory"))?
        .to_path_buf();
    let onnx_file = success.artifact().onnx_file();
    let mut files = Vec::new();
    let meta_bytes = {
        let (file, real) = candidate_dir
            .open_member(&artifact_rel.join("artifact.json"))
            .map_err(|e| e.to_error_report())?;
        read_bounded_open_file(file, real.as_path(), MAX_ARTIFACT_META_BYTES)
            .map_err(|e| fs_report(&e))?
    };
    for (component, name) in [
        (PackageComponent::Weights, onnx_file),
        (PackageComponent::Metadata, "artifact.json"),
    ] {
        if Path::new(name).components().count() != 1 {
            return Err(invalid("onnx file name is invalid"));
        }
        let (file, real) = candidate_dir
            .open_member(&artifact_rel.join(name))
            .map_err(|e| e.to_error_report())?;
        files.push((component, real.into_path_buf(), file));
    }
    // 語彙ファイル: 無い（NotFound）ときだけ `has_vocab_file = false`。それ以外の失敗は止める。
    // あれば `artifact.json` 記載の sha256・許可形式と照合し（`package`・`infer` と同じ
    // `verify_vocab_member`）、不一致・形式不正は `invalid_input` で止める（REQ-39）。
    let vocab_rel = artifact_rel.join(VOCAB_FILE_NAME);
    let vocab_bytes = match candidate_dir.open_member(&vocab_rel) {
        Ok((file, real)) => Some(
            read_bounded_open_file(file, real.as_path(), MAX_FILE_BYTES)
                .map_err(|e| fs_report(&e))?,
        ),
        Err(PathRejection::Unresolvable { source, .. }) if source.kind() == ErrorKind::NotFound => {
            None
        }
        Err(e) => return Err(e.to_error_report()),
    };
    let meta = ArtifactMeta::parse(&meta_bytes).map_err(|e| e.to_error_report())?;
    verify_vocab_member(&meta, vocab_bytes.as_deref())?;
    let has_vocab_file = vocab_bytes.is_some();
    if has_vocab_file {
        let (file, real) = candidate_dir
            .open_member(&vocab_rel)
            .map_err(|e| e.to_error_report())?;
        files.push((
            PackageComponent::VocabOrFeatureTransform,
            real.into_path_buf(),
            file,
        ));
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
        guideline_bytes: fandhe_edge_runtime::vocab_exclusion::VOCAB_GUIDELINE_BYTES,
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
