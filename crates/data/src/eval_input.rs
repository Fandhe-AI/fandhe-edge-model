//! 評価入力（gold・pred の JSONL）の異常系処理（REQ-23・TASK-23.1-1）。
//!
//! CLI の `evaluate` 工程（TASK-33.x で配線予定）から、ガード層を通過済みの
//! gold（正解）・pred（予測）JSONL 本文を受け取って呼ばれることを想定する。
//! 本モジュールが担うのは「異常な入力・空の入力をどう扱うか」（停止する／
//! 警告して除外する／警告して含める、のいずれか）を固定することだけであり、
//! 正解率・混同行列・Macro-F1・Wilson 信頼区間などの指標計算は行わない
//! （評価器は TASK-24.1（#58/#59/#60）の 1 つに集約する方針のため、本 crate に
//! 指標計算を持ち込まない）。[`prepare_evaluation_input`] が返す
//! [`EvalInputOutcome::active`] を評価器（TASK-24.1）が受け取り、指標を
//! 計算する想定である。
//!
//! # なぜ [`inspect`] を再利用しないか
//!
//! [`inspect::inspect_records`] とはレコードの形も方針も異なる。
//! `inspect` は学習データ（`{id,input,output.intent,tags?,group_id?}`）を
//! 対象に、重複 `id` の 2 行目を除外して処理を続ける。本モジュールが扱う
//! gold（`{id,label}` または `{id,output:{intent}}`）・pred
//! （`{id,status,predicted_label}`）は形が異なるうえ、`id` が重複した場合は
//! gold と pred を一意に突き合わせられないため停止する（[`EvalInputStop::DuplicateId`]）。
//! 方針が正反対のため、共通化せず別モジュールとして実装する。
//!
//! # 出典・PoC-9 との差分
//!
//! 挙動は PoC-9（`docs/spec/03-poc/evaluation-contract/`。private submodule）の
//! `evaluator/metrics.py`・`evaluator/records.py` の v1.1 既定挙動と、
//! `fixtures/known/single-select`・`fixtures/anomaly/01`〜`06`
//! （本リポには `fixtures/evaluation_contract/` として移植済み。出典は
//! `fixtures/evaluation_contract/PROVENANCE.md`）の実測に基づいて固定した。
//! ケース 7〜12（矛盾・ラベル順序・未出現クラス・不正なスコア・全件保留・
//! 全件失敗）は本モジュールの対象外で、兄弟 issue（TASK-23.1-2）が拡張する
//! （[`WarningCode`] の宣言時コメント参照）。PoC-9 との既知の差分:
//!
//! - PoC-9 の評価器は終了コード `2` で停止を表すが、本リポの 7 種終了コード
//!   契約（REQ-21）に `2` は無い。本モジュールは [`EvalInputStop`] という
//!   enum を返すだけにとどめ、`ExitCode`（`invalid_input`=64 等）への対応付けは
//!   CLI 側（TASK-21.2・TASK-33.x）の責務とする。本 crate は終了コードの
//!   数値をハードコードしない
//! - PoC-9 は `excluded_ids` 等をレコードの `id` の値で列挙するが、本モジュールは
//!   診断情報（[`EvalInputStop`]・[`EvalInputWarning`]）にデータの生値
//!   （`id`・`label`・`input`）を一切含めず、行番号だけで位置を示す
//!   （`.claude/rules/security.md`「データ本文をログ・エラーメッセージへ転記しない」。
//!   [`inspect`] モジュールの前例（issue #38・PR #191 のレビュー指摘）に倣う）
//! - 正規化した `input` の重複検出（ケース 6）は「前後空白の除去＋内部の
//!   連続空白を半角スペース 1 つにまとめる」までとし、NFKC 正規化は
//!   適用しない。`unicode-normalization` が未承認の依存のため
//!   （`.claude/rules/dependency-policy.md`）、承認後に別途対応する
//! - `id` が存在しない・文字列でない・空文字列の場合に停止する
//!   （[`EvalInputStop::InvalidId`]）のは PoC-9 に規定が無い、本リポとして
//!   安全側に倒した判断である（`id` で一意に突き合わせられない入力を
//!   評価しないため）
//!
//! # 前提条件（呼び出し元が守るべきこと）
//!
//! [`inspect`] と同様、本モジュールの関数はファイル読み込み・サイズ上限検査
//! （REQ-39）を行わない。既に読み込み済みの JSONL 本文を受け取るところから
//! 始まる。ガード層（パス未確定）を通過済みの入力を渡すことを前提とする。

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

/// gold（正解）側か pred（予測）側かを表す。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Gold,
    Prediction,
}

impl Side {
    pub fn code(&self) -> &'static str {
        match self {
            Side::Gold => "gold",
            Side::Prediction => "prediction",
        }
    }
}

/// [`prepare_evaluation_input`] が処理を停止させる異常（REQ-23）。
///
/// いずれも「評価を続けられない」ことを表し、指標は一切計算しない。
/// `code()` は CLI が `ExitCode`（REQ-21。本 crate はその数値をハード
/// コードしない）へ対応付ける際の機械判定キーとして使う想定。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalInputStop {
    /// gold が 0 レコード（PoC-9 ケース 01）。[`require_non_empty`]
    /// （学習データが 0 件の場合）でも同じ variant を再利用し、その場合は
    /// `pred_rows` を意味を持たない `0` として埋める（学習データには
    /// pred の概念が無いため）。
    EmptyData { gold_rows: usize, pred_rows: usize },
    /// 有効なラベル ID の集合が空（PoC-9 addendum A-6 の `missing_labels` 相当）。
    EmptyLabelSet,
    /// `id` が同じ側（gold または pred）で複数行に重複した（PoC-9 ケース 05）。
    /// `id` で一意に突き合わせられないため、gold の欠陥除外（ケース 2〜4・6）
    /// より前に検出して停止する。`lines` は重複した行番号を昇順ですべて含む
    /// （重複グループが複数あれば、それらすべての行番号を含む）。
    DuplicateId { side: Side, lines: Vec<usize> },
    /// 行が JSON として解釈できなかった（[`inspect::AnomalyCode::MalformedJson`] と
    /// 同じ分類。PoC-9 `records.py` の `InputFileError("malformed_json", ...)`）。
    MalformedJson { side: Side, line: usize },
    /// JSON としては解釈できたが object（レコード）ではなかった。
    MalformedRecord { side: Side, line: usize },
    /// `id` が存在しない・文字列でない・空文字列だった。PoC-9 に規定は無く、
    /// `id` で一意に突き合わせられない入力を安全側に倒して停止する
    /// （モジュール doc「PoC-9 との差分」参照）。
    InvalidId { side: Side, line: usize },
}

impl EvalInputStop {
    pub fn code(&self) -> &'static str {
        match self {
            EvalInputStop::EmptyData { .. } => "empty_data",
            EvalInputStop::EmptyLabelSet => "empty_label_set",
            EvalInputStop::DuplicateId { .. } => "duplicate_id",
            EvalInputStop::MalformedJson { .. } => "malformed_json",
            EvalInputStop::MalformedRecord { .. } => "malformed_record",
            EvalInputStop::InvalidId { .. } => "invalid_id",
        }
    }
}

/// 警告時にとる扱い（除外する／エラーとして含める／警告するが含める）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningAction {
    /// `active` から除外する（正解側の欠陥のみに適用する。評価契約
    /// 「正解側の欠陥だけを除外してよい」という原則）。
    Exclude,
    /// 除外せず [`PredictionOutcome::Error`] として `active` に含める
    /// （不正解として分母に数える。評価を甘く見せないため）。
    IncludeAsError,
    /// 除外せず警告のみ行う（分母に含める）。
    WarnInclude,
}

impl WarningAction {
    pub fn code(&self) -> &'static str {
        match self {
            WarningAction::Exclude => "exclude",
            WarningAction::IncludeAsError => "include_as_error",
            WarningAction::WarnInclude => "warn_include",
        }
    }
}

/// 警告の種別。
///
/// TASK-23.1-2（兄弟 issue）が `ContradictoryInput`（正規化 input が同じで
/// gold ラベルが食い違うグループ。ケース 7）・`InvalidScore`（ケース 10）・
/// `UnseenClass`（ケース 9）・`AllAbstain`（ケース 11）・`AllError`（ケース 12）を
/// 追加する想定。本モジュールはそれらの分岐点（[`find_duplicate_input_lines`]
/// が労合するグループのうちラベルが食い違うものを警告しないことで、追加の
/// 分岐を差し込む余地を残している）だけを用意する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WarningCode {
    /// gold の正解が欠落している（`label` が無い・`null`。ケース 2）。
    MissingGold,
    /// gold の正解が定義された型・enum に合わない（ケース 4）。
    MalformedGold,
    /// gold の正解が有効なラベル ID の集合に含まれない（ケース 3）。
    UnknownGoldLabel,
    /// 正規化した `input` が一致し、gold ラベルも一致する行が複数ある
    /// （ケース 6）。ラベルが食い違うグループ（矛盾）は対象外（TASK-23.1-2 が
    /// `ContradictoryInput` として扱う）。
    DuplicateInputWithinSplit,
    /// `active` な gold 行に対応する pred 行が無い。
    MissingPrediction,
}

impl WarningCode {
    pub fn code(&self) -> &'static str {
        match self {
            WarningCode::MissingGold => "missing_gold",
            WarningCode::MalformedGold => "malformed_gold",
            WarningCode::UnknownGoldLabel => "unknown_gold_label",
            WarningCode::DuplicateInputWithinSplit => "duplicate_input_within_split",
            WarningCode::MissingPrediction => "missing_prediction",
        }
    }

    /// この警告種別に固定で対応する [`WarningAction`]。
    fn action(&self) -> WarningAction {
        match self {
            WarningCode::MissingGold
            | WarningCode::MalformedGold
            | WarningCode::UnknownGoldLabel => WarningAction::Exclude,
            WarningCode::DuplicateInputWithinSplit => WarningAction::WarnInclude,
            WarningCode::MissingPrediction => WarningAction::IncludeAsError,
        }
    }

    /// この警告種別に固定で対応する [`Side`]（診断情報が指す側）。
    fn side(&self) -> Side {
        match self {
            WarningCode::MissingGold
            | WarningCode::MalformedGold
            | WarningCode::UnknownGoldLabel => Side::Gold,
            WarningCode::DuplicateInputWithinSplit => Side::Gold,
            WarningCode::MissingPrediction => Side::Prediction,
        }
    }
}

/// 1 件の警告（同じ [`WarningCode`] に該当した全行を集約する）。
///
/// レコード本文（`id`・`label`・`input` の実値）は保持しない
/// （モジュール doc「PoC-9 との差分」参照）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalInputWarning {
    pub code: WarningCode,
    pub action: WarningAction,
    pub side: Side,
    /// 該当した行番号（昇順・重複なし）。[`WarningCode::MissingPrediction`] の
    /// 場合は対応する pred 行が存在しないため、代わりに gold 側の行番号を指す。
    pub lines: Vec<usize>,
}

/// 予測が無効だった理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidPredictionReason {
    /// `status: "ok"` だが `predicted_label` が文字列でない。
    NotString,
    /// `status: "ok"` で文字列だが、有効なラベル ID の集合に含まれない
    /// （空文字列を含む）。
    UnknownLabel,
    /// `status` が `"ok"` / `"abstain"` / `"error"` のいずれでもない。
    UnknownStatus,
}

impl InvalidPredictionReason {
    pub fn code(&self) -> &'static str {
        match self {
            InvalidPredictionReason::NotString => "not_string",
            InvalidPredictionReason::UnknownLabel => "unknown_label",
            InvalidPredictionReason::UnknownStatus => "unknown_status",
        }
    }
}

/// エラー扱いになった予測の由来。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorOrigin {
    /// pred 側が `status: "error"` と報告した。
    Reported,
    /// 対応する pred 行そのものが無かった（[`WarningCode::MissingPrediction`]）。
    MissingPrediction,
}

impl ErrorOrigin {
    pub fn code(&self) -> &'static str {
        match self {
            ErrorOrigin::Reported => "reported",
            ErrorOrigin::MissingPrediction => "missing_prediction",
        }
    }
}

/// 1 件の gold 行に対する予測の分類結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionOutcome {
    /// `status: "ok"` かつ予測が有効なラベル ID の集合に含まれる文字列。
    Label(String),
    /// 型または値が不正（分母には含めるが、不正解として扱う）。
    Invalid(InvalidPredictionReason),
    /// `status: "abstain"`（判定を保留した）。
    Abstain,
    /// エラー（分母には含めるが、不正解として扱う）。
    Error(ErrorOrigin),
}

/// 評価対象として残った 1 行（gold 行を基準にする）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveRow {
    /// gold 側の 1 始まり行番号。
    pub gold_line: usize,
    pub id: String,
    pub gold_label: String,
    pub prediction: PredictionOutcome,
    /// 対応する pred 行の行番号（見つからなかった場合は `None`）。
    pub pred_line: Option<usize>,
}

/// [`prepare_evaluation_input`] の結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalInputOutcome {
    /// gold の行順に並んだ、評価対象として残った行。
    pub active: Vec<ActiveRow>,
    /// 決定的な順序（モジュール doc・[`prepare_evaluation_input`] 参照）で
    /// 並んだ警告一覧。該当が無い [`WarningCode`] は含まれない。
    pub warnings: Vec<EvalInputWarning>,
}

/// 1 行分の JSON レコード（パース済み・`id` 検証済み）。
struct ParsedRow {
    line: usize,
    id: String,
    fields: Map<String, Value>,
}

/// 1 行 1 JSON（JSONL）の本文をパースする（gold・pred で共通の手順 1）。
///
/// 前後空白を除いて空になる行は読み飛ばす（行番号のカウントは進める）。
/// `id` が存在しない・文字列でない・空文字列の場合は
/// [`EvalInputStop::InvalidId`] で打ち切る。
fn parse_side(content: &str, side: Side) -> Result<Vec<ParsedRow>, EvalInputStop> {
    let mut rows = Vec::new();
    for (idx, raw_line) in content.lines().enumerate() {
        let line = idx + 1;
        if raw_line.trim().is_empty() {
            continue;
        }

        let value: Value = serde_json::from_str(raw_line)
            .map_err(|_| EvalInputStop::MalformedJson { side, line })?;

        let Value::Object(fields) = value else {
            return Err(EvalInputStop::MalformedRecord { side, line });
        };

        let id = match fields.get("id") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return Err(EvalInputStop::InvalidId { side, line }),
        };

        rows.push(ParsedRow { line, id, fields });
    }
    Ok(rows)
}

/// 同じ側（gold または pred）の中で `id` が重複していないか検査する（手順 4）。
///
/// 重複が見つかった場合、重複に関与した全行番号（初出行を含む）を昇順で返す。
fn duplicate_id_lines(rows: &[ParsedRow]) -> Vec<usize> {
    let mut first_seen: BTreeMap<&str, usize> = BTreeMap::new();
    let mut duplicated: BTreeSet<usize> = BTreeSet::new();
    for row in rows {
        match first_seen.get(row.id.as_str()) {
            Some(&first_line) => {
                duplicated.insert(first_line);
                duplicated.insert(row.line);
            }
            None => {
                first_seen.insert(row.id.as_str(), row.line);
            }
        }
    }
    duplicated.into_iter().collect()
}

/// gold 1 行分の正解ラベル抽出結果（手順 5）。
enum GoldExtract {
    Valid(String),
    MissingGold,
    MalformedGold,
    UnknownGoldLabel,
}

/// gold 1 行の正解を抽出・検証する（手順 5）。
///
/// `label` キーを優先し、無ければ `output.intent` を使う
/// （[`inspect`] モジュールが扱うレコード形状（`output.intent`）との
/// 互換のため。PoC-9 のフィクスチャはいずれも `label` キーのみを使う）。
fn extract_gold_label(
    fields: &Map<String, Value>,
    valid_label_ids: &BTreeSet<String>,
) -> GoldExtract {
    if let Some(value) = fields.get("label") {
        return classify_gold_value(value, valid_label_ids);
    }
    if let Some(output) = fields.get("output") {
        let Some(output_fields) = output.as_object() else {
            return GoldExtract::MalformedGold;
        };
        return match output_fields.get("intent") {
            Some(value) => classify_gold_value(value, valid_label_ids),
            None => GoldExtract::MissingGold,
        };
    }
    GoldExtract::MissingGold
}

/// gold の正解値（`label` または `output.intent`）を分類する共通処理。
fn classify_gold_value(value: &Value, valid_label_ids: &BTreeSet<String>) -> GoldExtract {
    if value.is_null() {
        return GoldExtract::MissingGold;
    }
    match value.as_str() {
        Some(s) if valid_label_ids.contains(s) => GoldExtract::Valid(s.to_string()),
        Some(_) => GoldExtract::UnknownGoldLabel,
        None => GoldExtract::MalformedGold,
    }
}

/// 前後空白の除去＋内部の連続空白を半角スペース 1 つにまとめる正規化
/// （モジュール doc「PoC-9 との差分」参照。NFKC は適用しない）。
fn normalize_input(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// 正規化した `input` が一致し、gold ラベルも一致するグループの行番号を集める
/// （手順 6・ケース 6）。ラベルが食い違うグループは対象外とする
/// （[`WarningCode::DuplicateInputWithinSplit`] の doc 参照）。
fn find_duplicate_input_lines(rows: &[(usize, String, &Map<String, Value>)]) -> BTreeSet<usize> {
    // 正規化 input -> [(line, label)]。BTreeMap で決定的な順序を保つ。
    let mut groups: BTreeMap<String, Vec<(usize, &str)>> = BTreeMap::new();
    for (line, label, fields) in rows {
        if let Some(input) = fields.get("input").and_then(Value::as_str) {
            groups
                .entry(normalize_input(input))
                .or_default()
                .push((*line, label.as_str()));
        }
    }

    let mut warned = BTreeSet::new();
    for members in groups.values() {
        if members.len() < 2 {
            continue;
        }
        let first_label = members[0].1;
        if members.iter().all(|(_, label)| *label == first_label) {
            warned.extend(members.iter().map(|(line, _)| *line));
        }
    }
    warned
}

/// pred 1 行の `status`/`predicted_label` を分類する（手順 7）。
fn classify_prediction(
    fields: &Map<String, Value>,
    valid_label_ids: &BTreeSet<String>,
) -> PredictionOutcome {
    match fields.get("status").and_then(Value::as_str) {
        Some("ok") => match fields.get("predicted_label") {
            Some(Value::String(s)) if valid_label_ids.contains(s) => {
                PredictionOutcome::Label(s.clone())
            }
            Some(Value::String(_)) => {
                PredictionOutcome::Invalid(InvalidPredictionReason::UnknownLabel)
            }
            _ => PredictionOutcome::Invalid(InvalidPredictionReason::NotString),
        },
        Some("abstain") => PredictionOutcome::Abstain,
        Some("error") => PredictionOutcome::Error(ErrorOrigin::Reported),
        _ => PredictionOutcome::Invalid(InvalidPredictionReason::UnknownStatus),
    }
}

/// gold（正解）・pred（予測）の JSONL 本文から評価対象の行を組み立てる
/// （REQ-23・TASK-23.1-1）。
///
/// # 手順（モジュール doc・PoC-9 `metrics.py` の手順 1〜5・7 を踏襲）
///
/// 1. gold・pred をそれぞれ行単位でパースする（[`parse_side`]）
/// 2. gold が 0 レコードなら [`EvalInputStop::EmptyData`] で停止する
/// 3. `valid_label_ids` が空集合なら [`EvalInputStop::EmptyLabelSet`] で停止する
/// 4. gold 側・pred 側それぞれで `id` の重複を検査し、あれば
///    [`EvalInputStop::DuplicateId`] で停止する（gold 側を先に判定する）
/// 5. gold の欠陥（欠落・型不正・未知ラベル）を除外する
///    （[`WarningCode::MissingGold`]・[`MalformedGold`](WarningCode::MalformedGold)・
///    [`UnknownGoldLabel`](WarningCode::UnknownGoldLabel)）
/// 6. 手順 5 を通過した行のうち、正規化した `input` が一致し gold ラベルも
///    一致する行を [`WarningCode::DuplicateInputWithinSplit`] として警告する
///    （除外しない）
/// 7. 手順 5 を通過した行それぞれに対応する pred 行を突き合わせ、
///    [`PredictionOutcome`] に分類する。対応する pred 行が無い場合は
///    [`PredictionOutcome::Error`]（[`ErrorOrigin::MissingPrediction`]）とし、
///    [`WarningCode::MissingPrediction`] を警告する。gold に存在しない `id` の
///    pred 行は無視する
/// 8. 警告は Exclude 系 → IncludeAsError 系 → WarnInclude 系の順、
///    各群の中は [`WarningCode`] の宣言順に並べる（`lines` は昇順）
pub fn prepare_evaluation_input(
    gold_content: &str,
    pred_content: &str,
    valid_label_ids: &BTreeSet<String>,
) -> Result<EvalInputOutcome, EvalInputStop> {
    let gold_rows = parse_side(gold_content, Side::Gold)?;
    let pred_rows = parse_side(pred_content, Side::Prediction)?;

    if gold_rows.is_empty() {
        return Err(EvalInputStop::EmptyData {
            gold_rows: gold_rows.len(),
            pred_rows: pred_rows.len(),
        });
    }

    if valid_label_ids.is_empty() {
        return Err(EvalInputStop::EmptyLabelSet);
    }

    let gold_dup_lines = duplicate_id_lines(&gold_rows);
    if !gold_dup_lines.is_empty() {
        return Err(EvalInputStop::DuplicateId {
            side: Side::Gold,
            lines: gold_dup_lines,
        });
    }
    let pred_dup_lines = duplicate_id_lines(&pred_rows);
    if !pred_dup_lines.is_empty() {
        return Err(EvalInputStop::DuplicateId {
            side: Side::Prediction,
            lines: pred_dup_lines,
        });
    }

    // 手順 5: gold の欠陥を除外し、通過した行だけを集める。
    let mut lines_by_code: BTreeMap<WarningCode, BTreeSet<usize>> = BTreeMap::new();
    let mut accepted: Vec<&ParsedRow> = Vec::new();
    let mut accepted_labels: Vec<String> = Vec::new();
    for row in &gold_rows {
        match extract_gold_label(&row.fields, valid_label_ids) {
            GoldExtract::Valid(label) => {
                accepted.push(row);
                accepted_labels.push(label);
            }
            GoldExtract::MissingGold => {
                lines_by_code
                    .entry(WarningCode::MissingGold)
                    .or_default()
                    .insert(row.line);
            }
            GoldExtract::MalformedGold => {
                lines_by_code
                    .entry(WarningCode::MalformedGold)
                    .or_default()
                    .insert(row.line);
            }
            GoldExtract::UnknownGoldLabel => {
                lines_by_code
                    .entry(WarningCode::UnknownGoldLabel)
                    .or_default()
                    .insert(row.line);
            }
        }
    }

    // 手順 6: 正規化 input による重複検出（除外しない）。
    let dedup_input: Vec<(usize, String, &Map<String, Value>)> = accepted
        .iter()
        .zip(accepted_labels.iter())
        .map(|(row, label)| (row.line, label.clone(), &row.fields))
        .collect();
    let duplicate_input_lines = find_duplicate_input_lines(&dedup_input);
    if !duplicate_input_lines.is_empty() {
        lines_by_code
            .entry(WarningCode::DuplicateInputWithinSplit)
            .or_default()
            .extend(duplicate_input_lines);
    }

    // 手順 7: pred との突き合わせ。
    let mut pred_by_id: BTreeMap<&str, &ParsedRow> = BTreeMap::new();
    for row in &pred_rows {
        pred_by_id.insert(row.id.as_str(), row);
    }

    let mut active = Vec::with_capacity(accepted.len());
    for (row, label) in accepted.into_iter().zip(accepted_labels) {
        match pred_by_id.get(row.id.as_str()) {
            Some(pred_row) => {
                active.push(ActiveRow {
                    gold_line: row.line,
                    id: row.id.clone(),
                    gold_label: label,
                    prediction: classify_prediction(&pred_row.fields, valid_label_ids),
                    pred_line: Some(pred_row.line),
                });
            }
            None => {
                lines_by_code
                    .entry(WarningCode::MissingPrediction)
                    .or_default()
                    .insert(row.line);
                active.push(ActiveRow {
                    gold_line: row.line,
                    id: row.id.clone(),
                    gold_label: label,
                    prediction: PredictionOutcome::Error(ErrorOrigin::MissingPrediction),
                    pred_line: None,
                });
            }
        }
    }

    // 手順 8: Exclude 系 → IncludeAsError 系 → WarnInclude 系の順に並べる。
    const ORDER: [WarningCode; 5] = [
        WarningCode::MissingGold,
        WarningCode::MalformedGold,
        WarningCode::UnknownGoldLabel,
        WarningCode::MissingPrediction,
        WarningCode::DuplicateInputWithinSplit,
    ];
    let mut warnings = Vec::new();
    for code in ORDER {
        if let Some(lines) = lines_by_code.remove(&code) {
            warnings.push(EvalInputWarning {
                code,
                action: code.action(),
                side: code.side(),
                lines: lines.into_iter().collect(),
            });
        }
    }

    Ok(EvalInputOutcome { active, warnings })
}

/// 学習データが空（有効レコード 0 件）でないことを確認する（REQ-23 ケース 1 の
/// 学習データ側）。
///
/// [`crate::inspect::inspect_records`] は空の入力（`content == ""` や、
/// 全レコードが検査で除外された場合）でも `Ok`（空のベクタ）を返すため、
/// 呼び出し側が本関数を挟まないと黙って 0 件のまま処理が続いてしまう。
/// CLI の `inspect` / `train` 工程からこの関数を呼ぶ配線は TASK-33.x の
/// 担当であり、本関数はその手前の判定ロジックだけを提供する。
pub fn require_non_empty(outcome: &crate::inspect::InspectOutcome) -> Result<(), EvalInputStop> {
    if outcome.valid_records.is_empty() {
        return Err(EvalInputStop::EmptyData {
            gold_rows: 0,
            pred_rows: 0,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(ids: &[&str]) -> BTreeSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    /// REQ-23: 空の gold と重複 id が同時にある入力は EmptyData を優先する
    /// （手順の優先順位: 空データ判定が id 重複判定より先に走る）。
    #[test]
    fn req23_empty_data_takes_priority_over_duplicate_id() {
        let gold = "";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::EmptyData {
                gold_rows: 0,
                pred_rows: 2,
            })
        );
    }

    /// gold 側・pred 側の両方に重複 id がある場合、gold 側を先に報告する。
    #[test]
    fn req23_duplicate_id_reports_gold_side_first() {
        let gold = "{\"id\":\"dup\",\"label\":\"A\"}\n{\"id\":\"dup\",\"label\":\"A\"}\n";
        let pred = "{\"id\":\"dup\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n{\"id\":\"dup\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::DuplicateId {
                side: Side::Gold,
                lines: vec![1, 2],
            })
        );
    }

    /// JSON として壊れた行は MalformedJson で打ち切る。
    #[test]
    fn req23_malformed_json_stops() {
        let gold = "not json\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::MalformedJson {
                side: Side::Gold,
                line: 1,
            })
        );
        assert_eq!(result.unwrap_err().code(), "malformed_json");
    }

    /// object でない行は MalformedRecord で打ち切る。
    #[test]
    fn req23_malformed_record_stops() {
        let gold = "[1,2,3]\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::MalformedRecord {
                side: Side::Gold,
                line: 1,
            })
        );
    }

    /// id が無い・空文字列の行は InvalidId で打ち切る。
    #[test]
    fn req23_invalid_id_stops() {
        let gold = "{\"id\":\"\",\"label\":\"A\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::InvalidId {
                side: Side::Gold,
                line: 1,
            })
        );
    }

    /// `output.intent` 形式の gold も読めること。
    #[test]
    fn req23_reads_output_intent_shape() {
        let gold = "{\"id\":\"a\",\"output\":{\"intent\":\"A\"}}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();
        assert!(outcome.warnings.is_empty());
        assert_eq!(outcome.active.len(), 1);
        assert_eq!(outcome.active[0].gold_label, "A");
        assert_eq!(
            outcome.active[0].prediction,
            PredictionOutcome::Label("A".to_string())
        );
    }

    /// 予測の分類（Label / Invalid の 2 種 / Abstain / Error の 2 種）。
    #[test]
    fn req23_classifies_predictions() {
        let valid = labels(&["A", "B"]);
        assert_eq!(
            classify_prediction(
                &serde_json::json!({"status":"ok","predicted_label":"A"})
                    .as_object()
                    .unwrap()
                    .clone(),
                &valid,
            ),
            PredictionOutcome::Label("A".to_string())
        );
        assert_eq!(
            classify_prediction(
                &serde_json::json!({"status":"ok","predicted_label":"Z"})
                    .as_object()
                    .unwrap()
                    .clone(),
                &valid,
            ),
            PredictionOutcome::Invalid(InvalidPredictionReason::UnknownLabel)
        );
        assert_eq!(
            classify_prediction(
                &serde_json::json!({"status":"ok","predicted_label":["A"]})
                    .as_object()
                    .unwrap()
                    .clone(),
                &valid,
            ),
            PredictionOutcome::Invalid(InvalidPredictionReason::NotString)
        );
        assert_eq!(
            classify_prediction(
                &serde_json::json!({"status":"abstain","predicted_label":null})
                    .as_object()
                    .unwrap()
                    .clone(),
                &valid,
            ),
            PredictionOutcome::Abstain
        );
        assert_eq!(
            classify_prediction(
                &serde_json::json!({"status":"error","predicted_label":null})
                    .as_object()
                    .unwrap()
                    .clone(),
                &valid,
            ),
            PredictionOutcome::Error(ErrorOrigin::Reported)
        );
        assert_eq!(
            classify_prediction(
                &serde_json::json!({"status":"weird"})
                    .as_object()
                    .unwrap()
                    .clone(),
                &valid,
            ),
            PredictionOutcome::Invalid(InvalidPredictionReason::UnknownStatus)
        );
    }

    /// 正規化で空白差分を吸収して重複と判定すること。
    #[test]
    fn req23_normalizes_whitespace_before_dedup_check() {
        let gold = "{\"id\":\"a\",\"input\":\"foo  bar\",\"label\":\"A\"}\n{\"id\":\"b\",\"input\":\" foo bar \",\"label\":\"A\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n{\"id\":\"b\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();
        assert_eq!(outcome.warnings.len(), 1);
        assert_eq!(
            outcome.warnings[0].code,
            WarningCode::DuplicateInputWithinSplit
        );
        assert_eq!(outcome.warnings[0].lines, vec![1, 2]);
    }

    /// ラベルが食い違うグループ（矛盾）は重複として警告しない。
    #[test]
    fn req23_contradictory_input_is_not_reported_as_duplicate() {
        let gold = "{\"id\":\"a\",\"input\":\"foo\",\"label\":\"A\"}\n{\"id\":\"b\",\"input\":\"foo\",\"label\":\"B\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n{\"id\":\"b\",\"status\":\"ok\",\"predicted_label\":\"B\"}\n";
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A", "B"])).unwrap();
        assert!(outcome.warnings.is_empty());
        assert_eq!(outcome.active.len(), 2);
    }

    /// `code()` の文字列。
    #[test]
    fn req23_code_strings() {
        assert_eq!(WarningCode::MissingGold.code(), "missing_gold");
        assert_eq!(WarningCode::MalformedGold.code(), "malformed_gold");
        assert_eq!(WarningCode::UnknownGoldLabel.code(), "unknown_gold_label");
        assert_eq!(
            WarningCode::DuplicateInputWithinSplit.code(),
            "duplicate_input_within_split"
        );
        assert_eq!(WarningCode::MissingPrediction.code(), "missing_prediction");
        assert_eq!(WarningAction::Exclude.code(), "exclude");
        assert_eq!(WarningAction::IncludeAsError.code(), "include_as_error");
        assert_eq!(WarningAction::WarnInclude.code(), "warn_include");
        assert_eq!(Side::Gold.code(), "gold");
        assert_eq!(Side::Prediction.code(), "prediction");
    }

    /// 警告に生値（`id`・`label`・`input`）が入らないこと（長い値を使った回帰テスト。
    /// PR #191（issue #38）の前例に倣う）。
    #[test]
    fn req23_warnings_never_contain_raw_values() {
        let long_id = "x".repeat(500);
        let gold = format!("{{\"id\":\"{long_id}\",\"label\":null}}\n");
        let pred =
            format!("{{\"id\":\"{long_id}\",\"status\":\"ok\",\"predicted_label\":\"A\"}}\n");
        let outcome = prepare_evaluation_input(&gold, &pred, &labels(&["A"])).unwrap();
        assert_eq!(outcome.warnings.len(), 1);
        assert_eq!(outcome.warnings[0].lines, vec![1]);
        assert!(outcome.active.is_empty());
        // EvalInputWarning のフィールドは code/action/side/lines のみで、
        // 生値を保持するフィールドが型として存在しないことをコンパイル時に保証する。
    }

    /// `require_non_empty`。
    #[test]
    fn req23_require_non_empty() {
        let empty = crate::inspect::InspectOutcome {
            anomalies: Vec::new(),
            valid_records: Vec::new(),
        };
        assert_eq!(
            require_non_empty(&empty),
            Err(EvalInputStop::EmptyData {
                gold_rows: 0,
                pred_rows: 0,
            })
        );

        let non_empty = crate::inspect::InspectOutcome {
            anomalies: Vec::new(),
            valid_records: vec![crate::inspect::ValidRecord {
                line: 1,
                id: "a".to_string(),
                input: "hello".to_string(),
                label_id: "A".to_string(),
                tags: None,
                group_id: None,
            }],
        };
        assert_eq!(require_non_empty(&non_empty), Ok(()));
    }
}
