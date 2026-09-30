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
//! - 来歴の取り込み（`data::ingest`・REQ-40）・矛盾検出（`data::consistency`）は本工程に
//!   未接続（後続で結線）

use std::path::Path;

use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::stage_report::{InspectStageReport, SplitCounts};
use fandhe_edge_data::eval_freeze::{EvalDataState, FreezeRecord};
use fandhe_edge_data::inspect::ValidRecord;
use fandhe_edge_data::leak::{LeakCheckable, Partitions, inspect_leakage};
use fandhe_edge_data::split::{Groupable, Split, SplitRatios};
use fandhe_edge_data::split_record::split_and_record;

use crate::args::InspectArgs;
use crate::project::{
    DATA_DIR, EVALUATION_DATA_FILE, FREEZE_FILE, MAX_PROJECT_FILE_BYTES, Project, SPLIT_FILE,
    SPLIT_SEED, fail, inspect_bytes, invalid, runtime,
};
use crate::stage_output::evaluate_start;

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
    evaluate_start(&EvalDataState::Frozen(record), &bytes)?;
    Ok(Some(bytes))
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

/// `inspect` を実行する。
///
/// # Errors
/// データの異常・漏洩・凍結記録の不一致は `invalid_input`（64）、上限超過は
/// `limit_exceeded`（20）、I/O 失敗は `runtime_error`（70）。
pub fn run(args: &InspectArgs, cwd: &Path) -> Result<InspectStageReport, ErrorReport> {
    let project = Project::open(cwd, &args.project_dir)?;
    let definition = project.load_definition()?;
    let records = project.load_records(&definition)?;
    let train_rows = split_rows(&records)?;

    let eval_bytes = load_evaluation_bytes(&project)?;
    let eval_records = match &eval_bytes {
        Some(bytes) => Some(inspect_bytes(bytes, &definition)?),
        None => None,
    };

    let recorded = split_and_record(&train_rows, SPLIT_SEED, &SplitRatios::default())
        .map_err(|_| invalid("cannot split records"))?;
    let by_record = &recorded.result().by_record;
    // 分割が空のまま `status:"ok"` で記録すると、後続の `train` が必ず失敗する。書き込みの前に拒否する
    // （分割規則・seed は変えない。REQ-17）。group の件数が少ないと（目安として数件未満）起こるため、
    // 利用者には group の件数を増やしてもらう（message は固定語彙でデータ本文・件数を含めない）。
    for (split, message) in [
        (Split::Train, "train split is empty"),
        (Split::Validation, "validation split is empty"),
    ] {
        if !by_record.values().any(|s| *s == split) {
            return Err(invalid(message));
        }
    }
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
    Ok(InspectStageReport::new(
        records.len(),
        SplitCounts {
            train: train.len(),
            validation: validation.len(),
            test: test.len(),
        },
    ))
}
