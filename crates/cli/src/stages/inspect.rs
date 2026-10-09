//! `inspect` 工程: 取り込んだデータの検査・group 単位の分割・分割記録の保存
//! （REQ-16・REQ-17・REQ-33・TASK-33.1-2・#136）。
//!
//! # 手順（fail-closed）
//!
//! 1. 登録済みの定義の選択肢 ID を有効ラベルとして、`data/train.jsonl` を検査する
//!    （[`fandhe_edge_data::inspect::inspect_records`]。異常が 1 件でもあれば `invalid_input`）
//! 2. 評価データがあれば、凍結記録とのハッシュ一致を確認し（不一致は停止）、同じ検査を通す
//! 3. group 単位で分割し、seed・規則・各分割のハッシュを `split.json` へ記録する
//!    （[`fandhe_edge_data::split_record::split_and_record`]。seed は暫定の固定値）
//! 4. train・validation・test（と評価データ）の入力漏洩・group 跨ぎを検査し、あれば `invalid_input`
//!
//! 異常・漏洩の message は固定語彙で、行番号・ID・本文を出さない（`security.md`）。
//!
//! # 決定事項・未接続
//!
//! - `group_id` は必須とする（欠損は `invalid_input`）。group を推測して割り付けると
//!   同一 group の跨ぎを見逃しうるため（REQ-17）。暫定の判断でオーナー確認事項
//! - 来歴（REQ-40）: `data/*.provenance.json` があれば再検証し（改変対策・fail-closed）、正準 JSON を
//!   `provenance_record.json` として `split.json` と並べて書く。来歴なしは許可
//! - 矛盾（正規化後同一・ラベル違い）とメタデータ混入（`data::consistency`・REQ-16・TASK-16.2）は
//!   **報告のみで止めない**（spec は「検出して報告」でオーナー判断 2026-10-09）。stderr に理由別の
//!   件数だけを固定文で出し、終了コード・stdout JSON は変えない。学習・評価データを別々に検査する
//!   （学習と評価を跨ぐ矛盾は対象外）

use std::path::Path;

use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::{InspectStageReport, SplitCounts};
use fandhe_edge_data::consistency::{
    ConsistencyError, ContradictionRecord, MetadataMixReason, MetadataRecord, find_contradictions,
    find_metadata_mixed, gold_serializations,
};
use fandhe_edge_data::eval_freeze::{EvalDataState, FreezeRecord};
use fandhe_edge_data::inspect::ValidRecord;
use fandhe_edge_data::leak::{LeakCheckable, Partitions, inspect_leakage};
use fandhe_edge_data::normalize::NfkcWhitespaceNormalizer;
use fandhe_edge_data::split::{Groupable, Split, SplitRatios};
use fandhe_edge_data::split_record::split_and_record;

use crate::args::{InspectArgs, Subcommand};
use crate::log::StderrLog;
use crate::project::{
    DATA_DIR, EVALUATION_DATA_FILE, EVALUATION_PROVENANCE_FILE, FREEZE_FILE,
    MAX_PROJECT_FILE_BYTES, MAX_PROVENANCE_FILE_BYTES, PROVENANCE_RECORD_FILE, Project, SPLIT_FILE,
    TRAIN_PROVENANCE_FILE, check_provenance, fail, inspect_bytes, invalid, runtime,
};
use crate::stage_output::{EvaluateStart, evaluate_start};

/// 分割・漏洩検査の対象になる 1 行（`ValidRecord` の借用）。
///
/// `group` は `None` を表現できる（評価データは group を必須にしないため）。分割に使う行は
/// [`split_rows`] が `group_id` の存在を確認してから作る。
pub(crate) struct Row<'a> {
    id: &'a str,
    input: &'a str,
    label: &'a str,
    group: Option<&'a str>,
}

impl Groupable for Row<'_> {
    fn id(&self) -> &str {
        self.id
    }
    fn group_id(&self) -> &str {
        // `split_rows` が存在を確認済みの行だけが分割に渡る。
        self.group.unwrap_or_default()
    }
    fn label(&self) -> &str {
        self.label
    }
    fn input(&self) -> &[u8] {
        self.input.as_bytes()
    }
}

impl LeakCheckable for Row<'_> {
    fn id(&self) -> &str {
        self.id
    }
    fn input(&self) -> &[u8] {
        self.input.as_bytes()
    }
    fn group_id(&self) -> Option<&str> {
        self.group
    }
}

/// レコードを行へ写す（group は `None` を許す。漏洩検査の評価データ用）。
pub(crate) fn rows(records: &[ValidRecord]) -> Vec<Row<'_>> {
    records
        .iter()
        .map(|r| Row {
            id: &r.id,
            input: &r.input,
            label: &r.label_id,
            group: r.group_id.as_deref(),
        })
        .collect()
}

/// 分割に使う行を作る。`group_id` が 1 件でも欠けていれば `invalid_input`。
pub(crate) fn split_rows(records: &[ValidRecord]) -> Result<Vec<Row<'_>>, ErrorReport> {
    if records.iter().any(|r| r.group_id.is_none()) {
        return Err(invalid("records must have group_id"));
    }
    Ok(rows(records))
}

/// 登録済みの凍結記録と評価データ本体を読み、記録とのハッシュ一致を確認する
/// （不一致は停止。データが無ければ `None`）。
///
/// # Errors
/// 記録の読み込み失敗・ハッシュ不一致（`invalid_input`）。
pub(crate) fn load_evaluation_bytes(project: &Project) -> Result<Option<Vec<u8>>, ErrorReport> {
    Ok(load_frozen_evaluation(project)?.map(|(_, bytes)| bytes))
}

/// [`load_evaluation_bytes`] と同じ照合を行い、一致した凍結記録（sha256・バイト長）も返す
/// （`evaluate` が評価器へ渡す期待値と、評価完了記録の内容に使う。REQ-17・REQ-27・#314）。
///
/// # Errors
/// [`load_evaluation_bytes`] と同じ。
pub(crate) fn load_frozen_evaluation(
    project: &Project,
) -> Result<Option<(FreezeRecord, Vec<u8>)>, ErrorReport> {
    let data_rel = Path::new(DATA_DIR).join(EVALUATION_DATA_FILE);
    let Some(record_bytes) = project.read_optional(FREEZE_FILE, 1024)? else {
        // 記録が無いのに評価データがあれば、`evaluate_start` と同じく fail-closed で拒否する。
        let bytes = project.read_optional(&data_rel, MAX_PROJECT_FILE_BYTES)?;
        return match bytes {
            None => Ok(None),
            Some(b) => {
                // 凍結記録が無い状態の評価データは、空ファイルを含め評価データありとして扱わない。
                evaluate_start(&EvalDataState::NotProvided, &b)?;
                Err(invalid("freeze record is missing"))
            }
        };
    };
    let text =
        std::str::from_utf8(&record_bytes).map_err(|_| invalid("freeze record is invalid"))?;
    let record = FreezeRecord::parse(text).map_err(|_| invalid("freeze record is invalid"))?;
    let bytes = project.read(&data_rel, MAX_PROJECT_FILE_BYTES)?;
    match evaluate_start(&EvalDataState::Frozen(record), &bytes)? {
        EvaluateStart::Proceed(record) => Ok(Some((record, bytes))),
        EvaluateStart::Skipped(_) => Err(runtime("unexpected evaluation state")),
    }
}

/// 評価データが凍結記録どおりであることを確認する（`train`・`select`・`package` が副作用の前に呼ぶ。
/// REQ-17・REQ-27）。
///
/// 評価データが無い（`evaluate` が `skipped` になる）プロジェクトは何もせず通す。ある場合は
/// [`load_evaluation_bytes`]（`inspect`・`evaluate` と同じ照合の単一の出所）で、凍結記録の欠落・破損・
/// ハッシュ不一致を `invalid_input`（64）で停止する（fail-closed。学習・選定・書き出しで評価データを
/// 差し替えたまま進めない）。
///
/// # Errors
/// [`load_evaluation_bytes`] と同じ。
pub(crate) fn ensure_evaluation_frozen(project: &Project) -> Result<(), ErrorReport> {
    load_evaluation_bytes(project).map(|_| ())
}

/// train・validation・test のいずれかが 0 件の分割を拒否する（`invalid_input`。固定 message
/// `train split is empty`・`validation split is empty`・`test split is empty`）。
///
/// 空のまま `status:"ok"` で記録すると、後続の `train`（train・validation）や凍結 test での評価
/// （test）が必ず成り立たない。`split.json` などの書き込みより前に呼ぶ（分割規則・seed は変えない。
/// REQ-17・REQ-27）。group の件数が少ないと（目安として数件未満）起こるため、利用者には group の件数を
/// 増やしてもらう（message は固定語彙でデータ本文・件数を含めない）。
fn ensure_splits_non_empty(
    by_record: &std::collections::BTreeMap<String, Split>,
) -> Result<(), ErrorReport> {
    for (split, message) in [
        (Split::Train, "train split is empty"),
        (Split::Validation, "validation split is empty"),
        (Split::Test, "test split is empty"),
    ] {
        if !by_record.values().any(|s| *s == split) {
            return Err(invalid(message));
        }
    }
    Ok(())
}

/// 矛盾・メタデータ混入の検査用に `ValidRecord` を借用する（正解は `output_key` / `label_id`）。
struct ConsistencyRow<'a> {
    record: &'a ValidRecord,
    serializations: Vec<String>,
}

impl<'a> ConsistencyRow<'a> {
    fn new(record: &'a ValidRecord) -> Self {
        Self {
            record,
            serializations: gold_serializations(&record.output_key, &record.output_original),
        }
    }
}

impl ContradictionRecord for ConsistencyRow<'_> {
    fn id(&self) -> &str {
        &self.record.id
    }
    fn input(&self) -> &str {
        &self.record.input
    }
    fn gold_key(&self) -> &str {
        &self.record.output_key
    }
    fn group_id(&self) -> Option<&str> {
        self.record.group_id.as_deref()
    }
}

impl MetadataRecord for ConsistencyRow<'_> {
    fn id(&self) -> &str {
        &self.record.id
    }
    fn input(&self) -> &str {
        &self.record.input
    }
    fn gold_label(&self) -> Option<&str> {
        Some(&self.record.label_id)
    }
    fn gold_serializations(&self) -> &[String] {
        &self.serializations
    }
}

/// 矛盾とメタデータ混入を検出し、理由別の件数だけを stderr へ報告する（止めない。REQ-16・TASK-16.2）。
///
/// 0 件の理由は出さない。ID・本文・ラベルは出さない（`msgs` は固定語彙）。
///
/// # Errors
/// 検査済みデータでは起きない id の空・重複のみ `runtime_error`（fail-closed）。
fn report_consistency<W: std::io::Write>(
    log: &mut StderrLog<W>,
    records: &[ValidRecord],
    msgs: [&'static str; 4],
) -> Result<(), ErrorReport> {
    let rows: Vec<ConsistencyRow<'_>> = records.iter().map(ConsistencyRow::new).collect();
    let map_err = |_: ConsistencyError| runtime("unexpected inconsistent record ids");
    let contradictions = find_contradictions(&rows, &NfkcWhitespaceNormalizer).map_err(map_err)?;
    let mixed = find_metadata_mixed(&rows).map_err(map_err)?;
    let count_of = |reason: MetadataMixReason| {
        mixed
            .hits
            .values()
            .filter(|set| set.contains(&reason))
            .count()
    };
    for (msg, n) in [
        (msgs[0], contradictions.entries.len()),
        (msgs[1], count_of(MetadataMixReason::IdInInput)),
        (msgs[2], count_of(MetadataMixReason::GoldLabelInInput)),
        (
            msgs[3],
            count_of(MetadataMixReason::GoldSerializationInInput),
        ),
    ] {
        if n > 0 {
            log.info_count(Subcommand::Inspect, msg, n);
        }
    }
    Ok(())
}

const TRAIN_MSGS: [&str; 4] = [
    "train contradictory inputs",
    "train metadata id in input",
    "train metadata gold label in input",
    "train metadata gold serialization in input",
];
const EVAL_MSGS: [&str; 4] = [
    "evaluation contradictory inputs",
    "evaluation metadata id in input",
    "evaluation metadata gold label in input",
    "evaluation metadata gold serialization in input",
];

/// 取り込み済みの来歴を再検証し、`provenance_record.json` の内容（あるものだけの `evaluation`・`train`
/// キーの正準 JSON）を返す。来歴が 1 つも無ければ `None`（REQ-40）。
///
/// 取り込み後の改変・差し替えは `register` と同じ検査（[`check_provenance`]）で `invalid_input`。
fn load_provenance_json(project: &Project) -> Result<Option<String>, ErrorReport> {
    let mut parts = Vec::new();
    // キー順（evaluation < train）に固定して決定的にする。
    for (key, name) in [
        ("evaluation", EVALUATION_PROVENANCE_FILE),
        ("train", TRAIN_PROVENANCE_FILE),
    ] {
        let rel = Path::new(DATA_DIR).join(name);
        if let Some(bytes) = project.read_optional(rel, MAX_PROVENANCE_FILE_BYTES)? {
            parts.push(format!("\"{key}\":{}", check_provenance(&bytes)?));
        }
    }
    Ok((!parts.is_empty()).then(|| format!("{{{}}}", parts.join(","))))
}

/// `inspect` を実行する（検出の報告は実プロセスの stderr へ出す）。
///
/// # Errors
/// [`run_with_log`] と同じ。
pub fn run(args: &InspectArgs, cwd: &Path) -> Result<InspectStageReport, ErrorReport> {
    run_with_log(args, cwd, &mut StderrLog::new(std::io::stderr().lock()))
}

/// [`run`] の本体。`log` は矛盾・メタデータ混入の件数報告の出力先（REQ-16）。
///
/// # Errors
/// データの異常・漏洩・凍結記録の不一致は `invalid_input`（64）、上限超過は
/// `limit_exceeded`（20）、I/O 失敗は `runtime_error`（70）。
pub fn run_with_log<W: std::io::Write>(
    args: &InspectArgs,
    cwd: &Path,
    log: &mut StderrLog<W>,
) -> Result<InspectStageReport, ErrorReport> {
    let project = Project::open(cwd, &args.project_dir)?;
    let definition = project.load_definition()?;
    let records = project.load_records(&definition)?;
    let provenance_json = load_provenance_json(&project)?;
    report_consistency(log, &records, TRAIN_MSGS)?;
    let train_rows = split_rows(&records)?;

    let eval_bytes = load_evaluation_bytes(&project)?;
    let eval_records = match &eval_bytes {
        Some(bytes) => {
            let eval = inspect_bytes(bytes, &definition)?;
            report_consistency(log, &eval, EVAL_MSGS)?;
            Some(eval)
        }
        None => None,
    };

    let recorded = split_and_record(&train_rows, u64::from(args.seed), &SplitRatios::default())
        .map_err(|_| invalid("cannot split records"))?;
    let by_record = &recorded.result().by_record;
    ensure_splits_non_empty(by_record)?;
    let pick = |split: Split| -> Vec<Row<'_>> {
        train_rows
            .iter()
            .filter(|r| by_record.get(r.id) == Some(&split))
            .map(|r| Row {
                id: r.id,
                input: r.input,
                label: r.label,
                group: r.group,
            })
            .collect()
    };
    let (train, validation, test) = (
        pick(Split::Train),
        pick(Split::Validation),
        pick(Split::Test),
    );
    let eval_rows = eval_records.as_deref().map(rows);
    let leakage = inspect_leakage(&Partitions {
        train: &train,
        validation: Some(&validation),
        test: Some(&test),
        evaluation: eval_rows.as_deref(),
    })
    .map_err(|_| fail(ExitCode::LimitExceeded, "data exceeds inspection limits"))?;
    if leakage.input_leaks.leak_pair_count() > 0 || !leakage.group_straddles.straddles.is_empty() {
        return Err(invalid("data leakage detected between partitions"));
    }

    let json = recorded
        .record()
        .to_json()
        .map_err(|_| runtime("cannot serialize split record"))?;
    project.write_new(SPLIT_FILE, json.as_bytes())?;
    if let Some(provenance_json) = provenance_json {
        project.write_new(PROVENANCE_RECORD_FILE, provenance_json.as_bytes())?;
    }
    Ok(InspectStageReport::new(
        records.len(),
        SplitCounts {
            train: train.len(),
            validation: validation.len(),
            test: test.len(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(splits: &[Split]) -> std::collections::BTreeMap<String, Split> {
        splits
            .iter()
            .enumerate()
            .map(|(i, s)| (format!("r{i}"), *s))
            .collect()
    }

    /// REQ-17・REQ-27: 3 分割がすべて非空なら通り、空の分割ごとに固定 message の `invalid_input`。
    #[test]
    fn req17_empty_split_is_rejected_with_fixed_message() {
        let all = [Split::Train, Split::Validation, Split::Test];
        assert!(ensure_splits_non_empty(&map(&all)).is_ok());
        for (missing, message) in [
            (Split::Train, "train split is empty"),
            (Split::Validation, "validation split is empty"),
            (Split::Test, "test split is empty"),
        ] {
            let rest: Vec<Split> = all.iter().copied().filter(|s| *s != missing).collect();
            let err = ensure_splits_non_empty(&map(&rest)).expect_err("must reject");
            assert_eq!(err.code, ExitCode::InvalidInput);
            assert_eq!(err.message, message);
        }
        assert_eq!(
            ensure_splits_non_empty(&map(&[]))
                .expect_err("empty")
                .message,
            "train split is empty"
        );
    }
}
