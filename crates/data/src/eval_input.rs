//! 評価入力（gold・pred の JSONL）の異常系処理（REQ-23・TASK-23.1-1）。
//!
//! 現行の `evaluate` 工程（`stages::evaluate`）は本モジュールを使わず、本モジュールは PoC-26 の採点入口
//! （`fandhe-edge-score`。#445）から呼ばれる。`evaluate` 工程への配線は未実装。ガード層を通過済みの
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
//! `fixtures/known/single-select`・`fixtures/anomaly/01`〜`12`
//! （本リポには `fixtures/evaluation_contract/` として移植済み。出典は
//! `fixtures/evaluation_contract/PROVENANCE.md`）の実測に基づいて固定した。
//! ケース 7〜12（矛盾・ラベル順序・未出現クラス・不正なスコア・全件保留・
//! 全件失敗）は TASK-23.1-2（issue #56）で追加した。PoC-9 との既知の差分:
//!
//! - PoC-9 の評価器は終了コード `2` で停止を表すが、本リポの 7 種終了コード
//!   契約（REQ-21）に `2` は無い。本モジュールは [`EvalInputStop`] という
//!   enum を返すだけにとどめ、`ExitCode`（`invalid_input`=64 等）への対応付けは
//!   CLI 側（TASK-21.2・TASK-33.x）の責務とする。本 crate は終了コードの
//!   数値をハードコードしない
//! - PoC-9 は `excluded_ids` 等をレコードの `id` の値で列挙するが、本モジュールは
//!   診断情報（[`EvalInputStop`]・[`EvalInputWarning`]）にデータの生値
//!   （`id`・`label`・`input`）を一切含めず、行番号（または [`WarningCode::UnseenClass`]
//!   の場合はラベル ID。定義ファイル由来でデータ本文ではない）だけで位置を示す
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
//! - JSON オブジェクトキーの重複検出（Unicode エスケープによるキー重複
//!   smuggling を含む）は PoC-9 に規定が無い。検出プリミティブは
//!   [`crate::json_keys`] に集約し [`inspect`] と共有するが、検出後の扱いは
//!   [`inspect`]（[`inspect::AnomalyCode::DuplicateKey`]。レコードを除外して
//!   処理を継続する。issue #38・PR #191）とは正反対にする。gold・pred は
//!   評価契約（REQ-21・REQ-27）の根幹データであり、`serde_json` が後勝ちで
//!   潰した `id`／`label`／`predicted_label` を気付かず処理し続けることは、
//!   視認できるテキストと異なる値で評価が進む安全性の問題になる。そのため
//!   本モジュールは重複キーを検出した時点で [`EvalInputStop::DuplicateKey`]
//!   として処理全体を停止する（安全側に倒す判断は `InvalidId` と同じ理由）
//! - ケース 10（`anomaly/10-invalid-score`）は PoC-9 **v1.0** 時点の記述
//!   （`warn_exclude`）ではなく、v1.1（addendum A-2）の「除外せず error として
//!   分母に含め、不正解として数える」挙動を固定する
//!   （[`WarningAction::IncludeAsError`]・[`WarningCode::InvalidScore`]）。
//!   `expected.json` もレビュー指摘（PR #204）を受けて v1.1 の
//!   `include_as_error`（分母 5 件）へ更新済みで、v1.0 の記述は残っていない。
//!   addendum A-4 がスコア合計の許容差を `1e-6` と定めている
//!   （[`SCORE_SUM_TOLERANCE`] の doc 参照）
//! - pred 側の `NaN`・`Infinity`・`-Infinity`（JSON 標準外リテラル。PoC-9
//!   の Python `json` は `allow_nan=True` で既定受理する）は、厳密パースが
//!   失敗した場合に限り [`substitute_non_finite_literals`] でトップレベル
//!   `scores` フィールドの値限定で `null` へ置換して再パースし、該当行を
//!   無条件に [`ErrorOrigin::InvalidScore`] へ倒す（gold 側は緩和しない。
//!   `scores` 以外（無関係な metadata 等）に出現した該当トークンは置換
//!   されず、再パース失敗として [`EvalInputStop::MalformedJson`] で停止する。
//!   詳細は同関数の doc 参照）
//! - pred 側の `scores` に `1e400` のような、構文自体は正しいが f64 の
//!   表現範囲を超える数値トークンがあった場合も同じ緩和パース
//!   （[`substitute_non_finite_literals`]。内部で
//!   [`scan_json_number_token`] を使う）で `null` へ置換し、
//!   [`ErrorOrigin::InvalidScore`] へ倒す。`serde_json::from_str` はこの
//!   トークンを構文エラーではなく数値範囲外の `Err` として拒否するため、
//!   `NaN`／`Infinity` リテラルの置換だけでは検出できず、無条件に
//!   [`EvalInputStop::MalformedJson`] へ落ちて評価全体を止めてしまう不具合
//!   だった（レビュー指摘。PR #204）
//! - [`WarningCode::EmptyInput`]（REQ-23 境界値・TASK-23.2・issue #57）は
//!   PoC-9 に規定が無い、本リポで追加した警告である。gold の `input` が
//!   学習ワーカーの正規化後に空となる行は、推論経路と評価経路で前処理結果が
//!   食い違いうる境界（[`crate::preprocess_boundary`] モジュール doc・
//!   PoC-16 の `[0]` vs `[]` 参照）であり、両経路の挙動の統一は「検討中」の
//!   まま対象外のため、除外せず警告のみ行う
//!
//! # 前提条件（呼び出し元が守るべきこと）
//!
//! [`inspect`] と同様、本モジュールの関数はファイル読み込み・サイズ上限検査
//! （REQ-39）を行わない。既に読み込み済みの JSONL 本文を受け取るところから
//! 始まる。ガード層（パス未確定）を通過済みの入力を渡すことを前提とする。

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::json_keys::has_duplicate_key;

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
    /// トップレベルまたはネスト先（`output` 等）に同一 JSON キーが複数回
    /// 出現し、`serde_json` のパース時点で後勝ちの値へ潰れていた
    /// （Unicode エスケープによるキー重複 smuggling を含む）。gold・pred は
    /// 評価契約の根幹データのため、[`inspect`] とは異なり除外せず処理全体を
    /// 停止する（モジュール doc「PoC-9 との差分」参照）。
    DuplicateKey { side: Side, line: usize },
    /// gold に行はあるが、手順 5 の除外（欠落・型不正・未知ラベル。
    /// [`WarningCode::MissingGold`]・[`MalformedGold`](WarningCode::MalformedGold)・
    /// [`UnknownGoldLabel`](WarningCode::UnknownGoldLabel)）ですべて除外され、
    /// 評価対象（[`EvalInputOutcome::active`]）が 0 件になった。
    /// `Ok(EvalInputOutcome { active: vec![], .. })` を返すと呼び出し側が
    /// 評価済みと誤認するため、評価契約の fail-closed
    /// （`.claude/rules/evaluation-contract.md`「評価データが無い場合は
    /// status:"skipped"・exit 0」「判定不能時の fail-closed」。REQ-23・REQ-27）
    /// に従い、`EmptyData`（gold が 0 レコード）とは区別して停止する。
    /// `warnings` は除外理由の内訳（行番号のみ・生値は含めない）。
    NoValidGold {
        gold_rows: usize,
        pred_rows: usize,
        warnings: Vec<EvalInputWarning>,
    },
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
            EvalInputStop::DuplicateKey { .. } => "duplicate_key",
            EvalInputStop::NoValidGold { .. } => "no_valid_gold",
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
/// TASK-23.1-2（issue #56）が `ContradictoryInput`・`InvalidScore`・
/// `UnseenClass`・`AllAbstain`・`AllError`（ケース 7・10・9・11・12）を追加した。
/// [`UnseenClass`](WarningCode::UnseenClass) だけは行番号ではなくラベル ID で
/// 対象を示す（[`EvalInputWarning::labels`] の doc 参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WarningCode {
    /// gold の正解が欠落している（`label` が無い・`null`。ケース 2）。
    MissingGold,
    /// gold の正解が定義された型・enum に合わない（ケース 4）。
    MalformedGold,
    /// gold の正解が有効なラベル ID の集合に含まれない（ケース 3）。
    UnknownGoldLabel,
    /// 正規化した `input` が一致するが、gold ラベルが 2 種以上に分かれる
    /// （矛盾。ケース 7）。[`DuplicateInputWithinSplit`](WarningCode::DuplicateInputWithinSplit)
    /// （ラベルが一致する重複）とは異なり、分母から除外する
    /// （[`find_duplicate_input_lines`] の doc 参照）。
    ContradictoryInput,
    /// 正規化した `input` が一致し、gold ラベルも一致する行が複数ある
    /// （ケース 6）。ラベルが食い違うグループ（矛盾）は
    /// [`ContradictoryInput`](WarningCode::ContradictoryInput) が扱う。
    DuplicateInputWithinSplit,
    /// `active` な gold 行に対応する pred 行が無い。
    MissingPrediction,
    /// pred の `scores` が不正（欠損の型・NaN・無限大・負値・合計が 1 から
    /// [`SCORE_SUM_TOLERANCE`]（`1e-6`）を超えて外れる。ケース 10）。除外せず
    /// [`PredictionOutcome::Error`]（[`ErrorOrigin::InvalidScore`]）として
    /// 含め、不正解として数える（評価を甘く見せないため。addendum A-2）。
    InvalidScore,
    /// `active` な行がすべて [`PredictionOutcome::Abstain`]（ケース 11）。
    AllAbstain,
    /// `active` な行がすべて [`PredictionOutcome::Error`]（
    /// [`PredictionOutcome::Invalid`] は対象外。ケース 12）。
    AllError,
    /// gold の `input` が学習ワーカーの正規化後に空となる行
    /// （[`crate::preprocess_boundary::is_empty_after_normalization`] が真。
    /// REQ-23 境界値・TASK-23.2・issue #57）。推論経路と評価経路で前処理
    /// 結果が食い違いうる境界（PoC-16 で `[0]` vs `[]` の食い違いを実測。
    /// [`crate::preprocess_boundary`] モジュール doc 参照）であり、両経路の
    /// 統一は対象外（「検討中」）のため除外せず警告のみ行う。行番号のみを
    /// 報告し `input` の値は含めない。
    EmptyInput,
    /// 定義済みラベル（`valid_label_ids`）のうち、`active` な gold 行に
    /// 1 件も出現しないもの（ケース 9）。行番号では表せないため
    /// [`EvalInputWarning::labels`]（ラベル ID の昇順・重複なし）で示す
    /// （[`EvalInputWarning::lines`] は常に空にする）。
    UnseenClass,
}

impl WarningCode {
    pub fn code(&self) -> &'static str {
        match self {
            WarningCode::MissingGold => "missing_gold",
            WarningCode::MalformedGold => "malformed_gold",
            WarningCode::UnknownGoldLabel => "unknown_gold_label",
            WarningCode::ContradictoryInput => "contradictory_input",
            WarningCode::DuplicateInputWithinSplit => "duplicate_input_within_split",
            WarningCode::MissingPrediction => "missing_prediction",
            WarningCode::InvalidScore => "invalid_score",
            WarningCode::AllAbstain => "all_abstain",
            WarningCode::AllError => "all_error",
            WarningCode::EmptyInput => "empty_input",
            WarningCode::UnseenClass => "unseen_class",
        }
    }

    /// この警告種別に固定で対応する [`WarningAction`]。
    fn action(&self) -> WarningAction {
        match self {
            WarningCode::MissingGold
            | WarningCode::MalformedGold
            | WarningCode::UnknownGoldLabel
            | WarningCode::ContradictoryInput => WarningAction::Exclude,
            WarningCode::DuplicateInputWithinSplit
            | WarningCode::AllAbstain
            | WarningCode::AllError
            | WarningCode::EmptyInput
            | WarningCode::UnseenClass => WarningAction::WarnInclude,
            WarningCode::MissingPrediction | WarningCode::InvalidScore => {
                WarningAction::IncludeAsError
            }
        }
    }

    /// この警告種別に固定で対応する [`Side`]（診断情報が指す側）。
    fn side(&self) -> Side {
        match self {
            WarningCode::MissingGold
            | WarningCode::MalformedGold
            | WarningCode::UnknownGoldLabel
            | WarningCode::ContradictoryInput
            | WarningCode::DuplicateInputWithinSplit
            | WarningCode::EmptyInput
            | WarningCode::UnseenClass => Side::Gold,
            WarningCode::MissingPrediction
            | WarningCode::InvalidScore
            | WarningCode::AllAbstain
            | WarningCode::AllError => Side::Prediction,
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
    /// 該当した行番号（昇順・重複なし）。[`WarningCode::MissingPrediction`]は
    /// 対応する pred 行が存在しないため、常に gold 側の行番号を指す（対象行を
    /// 特定できない警告を出さないため。レビュー指摘。PR #204）。
    /// [`WarningCode::AllError`] も同じ理由で pred 行が 1 件も無い場合
    /// （全行が `MissingPrediction` 起因）は gold 側の行番号を指すが、
    /// pred 行を持つ行が 1 件でもあれば pred 側の行番号だけを使う（gold・pred
    /// は別々の採番空間のため、両者を同一集合に混在させると偶然同じ数値に
    /// なった場合に 1 件へ潰れて対象行を取りこぼす。Bugbot 指摘。PR #204
    /// threadId PRRT_kwDOUq-SxM6mfP3Y）。いずれの場合も `side` フィールドの
    /// 値（常に [`Side::Prediction`]）自体は行番号がどちらの採番空間かを
    /// 表さないため、行番号の実際の由来は本 doc の規約に従う。
    /// [`WarningCode::UnseenClass`] の場合は行番号で表せないため常に空。
    pub lines: Vec<usize>,
    /// [`WarningCode::UnseenClass`] のときだけ非空（ラベル ID の昇順・
    /// 重複なし）。それ以外の `code` では常に空。ラベル ID は定義ファイル
    /// 由来でデータ本文ではないため、生値を診断に含めない方針
    /// （モジュール doc「PoC-9 との差分」）には抵触しない。
    pub labels: Vec<String>,
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
    /// pred の `scores` が不正だった、または行に `NaN`／`Infinity`／
    /// `-Infinity` の緩和パース痕跡があった（[`WarningCode::InvalidScore`]。
    /// ケース 10）。`status` の値（`ok`／`abstain`／`error`／未知）に関わらず
    /// 優先する。
    InvalidScore,
}

impl ErrorOrigin {
    pub fn code(&self) -> &'static str {
        match self {
            ErrorOrigin::Reported => "reported",
            ErrorOrigin::MissingPrediction => "missing_prediction",
            ErrorOrigin::InvalidScore => "invalid_score",
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
    /// 定義済みラベルのうち `active` な gold 行に 1 件も出現しないものの
    /// 昇順リスト（ケース 9。PoC-9 `result.unseen_labels` 相当）。
    /// [`WarningCode::UnseenClass`] が `warnings` に含まれる場合、同じ内容が
    /// その `labels` にも入る。評価器（TASK-24.1）が Macro-F1 の除外ラベル
    /// 列挙に使う想定で、本 crate はそれ以上の計算をしない。
    pub unseen_labels: Vec<String>,
}

/// 1 行分の JSON レコード（パース済み・`id` 検証済み）。
struct ParsedRow {
    line: usize,
    id: String,
    fields: Map<String, Value>,
    /// pred 行が緩和パース（[`substitute_non_finite_literals`]）を経由した
    /// か（`NaN`・`Infinity`・`-Infinity` のリテラルを含んでいた印）。
    /// gold 側は緩和パースを行わないため常に `false`。
    non_finite_literal: bool,
}

/// 1 行 1 JSON（JSONL）の本文をパースする（gold・pred で共通の手順 1）。
///
/// 前後空白を除いて空になる行は読み飛ばす（行番号のカウントは進める）。
/// pred 側に限り、厳密なパースが失敗した場合だけ
/// [`substitute_non_finite_literals`] による緩和パースを 1 回試す
/// （ケース 10。gold 側は緩和しない。モジュール doc「PoC-9 との差分」参照）。
/// トップレベルまたはネスト先に同一 JSON キーが複数回出現していた場合は
/// [`EvalInputStop::DuplicateKey`] で打ち切る（`id` の抽出より前に検査する。
/// `id` 自体が smuggling の対象になり得るため。緩和パースを経由した場合は
/// 置換後の文字列に対して検査する）。`id` が存在しない・文字列でない・
/// 空文字列の場合は [`EvalInputStop::InvalidId`] で打ち切る。
fn parse_side(content: &str, side: Side) -> Result<Vec<ParsedRow>, EvalInputStop> {
    let mut rows = Vec::new();
    for (idx, raw_line) in content.lines().enumerate() {
        let line = idx + 1;
        if raw_line.trim().is_empty() {
            continue;
        }

        let mut non_finite_literal = false;
        let mut substituted: Option<String> = None;
        let value: Value = match serde_json::from_str(raw_line) {
            Ok(v) => v,
            Err(_) if side == Side::Prediction => {
                let relaxed = substitute_non_finite_literals(raw_line)
                    .ok_or(EvalInputStop::MalformedJson { side, line })?;
                let v = serde_json::from_str(&relaxed)
                    .map_err(|_| EvalInputStop::MalformedJson { side, line })?;
                non_finite_literal = true;
                substituted = Some(relaxed);
                v
            }
            Err(_) => return Err(EvalInputStop::MalformedJson { side, line }),
        };

        if !value.is_object() {
            return Err(EvalInputStop::MalformedRecord { side, line });
        }

        let key_check_text = substituted.as_deref().unwrap_or(raw_line);
        if has_duplicate_key(key_check_text, &value) {
            return Err(EvalInputStop::DuplicateKey { side, line });
        }

        let Value::Object(fields) = value else {
            return Err(EvalInputStop::MalformedRecord { side, line });
        };

        let id = match fields.get("id") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            _ => return Err(EvalInputStop::InvalidId { side, line }),
        };

        rows.push(ParsedRow {
            line,
            id,
            fields,
            non_finite_literal,
        });
    }
    Ok(rows)
}

/// トップレベルの `scores` キーの値の中に現れる `NaN`・`Infinity`・
/// `-Infinity` のトークン（文字列リテラル外）、および `1e400` のような
/// 構文は正しいが f64 の表現範囲を超える（無限大になる）数値トークンを
/// `null` へ置換する（pred 側限定の緩和パース。ケース 10・TASK-23.1-2）。
///
/// `serde_json` は `NaN`・`Infinity` を JSON 標準外リテラルとして拒否し、
/// `1e400` のような範囲外の数値は構文自体は妥当なため字句エラーではなく
/// 数値範囲エラー（"number out of range"）として拒否する。いずれも
/// そのままでは [`EvalInputStop::MalformedJson`] で評価全体を止めてしまう
/// （PoC-9 の Python `json`（`allow_nan=True`）は前者を受理したうえで
/// invalid_score として扱う。範囲外数値は後述のとおり本モジュール独自の
/// 拡張。モジュール doc「PoC-9 との差分」参照）。範囲外数値の判定は
/// [`scan_json_number_token`] で数値トークンの終端まで読み取ってから
/// `str::parse::<f64>` の結果が有限かどうかで行う（有限なら置換せず raw の
/// まま残す）。値の位置（直前の非空白
/// バイトが `:`・`[`・`,`、または行頭）にあり、直後が区切り（空白・`,`・
/// `}`・`]`・行末）であるトークンだけを置換候補にし、さらにトップレベル
/// オブジェクトの現在のキーが `scores` であるときに限り実際に置換する。
/// `scores` 以外のフィールド（無関係な metadata 等）に出現した該当トークンは
/// 置換されず raw のまま残るため、呼び出し元の再パースが
/// [`EvalInputStop::MalformedJson`] として自然に停止する（`scores` 限定で
/// なければ、無関係な値の非有限リテラルまで `invalid_score` として誤って
/// 受理してしまう。レビュー指摘。PR #204）。トップレベルキーの追跡は、
/// 深さ 1（ルートオブジェクト直下）でのみ「直前に閉じた文字列リテラルの
/// 直後が `:` であればそれをキー名とみなす」方式で行い、深さが 1 に戻る
/// 閉じ括弧、またはトップレベルの `,` でリセットする。文字列リテラルの
/// 内外とエスケープをバイト列で追跡する線形スキャナで、再帰・バックトラック
/// を行わない（OWASP 不安全な設計対策。トークンの置換後の文字列は必ず
/// 呼び出し元で再パースし、キー重複検査もその文字列に対して行う）。
/// バイト単位で処理する（`byte as char` 等の再解釈をしない）ため、
/// マルチバイト UTF-8 文字を含む文字列リテラルを壊さない。1 つも置換
/// しなかった場合は `None` を返す。
fn substitute_non_finite_literals(raw_line: &str) -> Option<String> {
    const TOKENS: [&str; 3] = ["-Infinity", "Infinity", "NaN"];
    const SCORES_KEY: &[u8] = b"scores";

    let bytes = raw_line.as_bytes();
    let mut result: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut replaced = false;
    // 値の位置（直前の非空白が `:`・`[`・`,`、または行頭）にいるかどうか。
    let mut in_value_position = true;
    // ルートオブジェクトからの深さ（`{`・`[` で +1、`}`・`]` で -1）。
    let mut depth: i32 = 0;
    // 直近の文字列リテラルの中身（トップレベルキー検出用）。
    let mut string_buf: Vec<u8> = Vec::new();
    // 深さ 1 で値の展開中である、直前に確定したトップレベルキー名。
    let mut current_top_key: Option<Vec<u8>> = None;
    let mut i = 0usize;

    while i < bytes.len() {
        let byte = bytes[i];

        if in_string {
            if escaped {
                // 直前の `\` に続くエスケープ本体。トップレベルキー名の比較を
                // `"scores"`（`scores` の Unicode エスケープ表記）のような
                // 正当な JSON エスケープに対しても正しく行うため、string_buf
                // にはデコード後のバイト列を積む（result は再パース用に元の
                // 生バイト列のまま変更しない。レビュー指摘。PR #204）。
                result.push(byte);
                escaped = false;
                match byte {
                    b'"' => string_buf.push(b'"'),
                    b'\\' => string_buf.push(b'\\'),
                    b'/' => string_buf.push(b'/'),
                    b'b' => string_buf.push(0x08),
                    b'f' => string_buf.push(0x0C),
                    b'n' => string_buf.push(b'\n'),
                    b'r' => string_buf.push(b'\r'),
                    b't' => string_buf.push(b'\t'),
                    b'u' => {
                        // `\uXXXX`（サロゲートペアなら続く `\uXXXX` も）を読み、
                        // 元のバイト列を result へそのまま複写しつつ、
                        // string_buf にはデコードしたコードポイントの UTF-8 を
                        // 積む。不正な形式（桁不足・非16進・孤立サロゲート）は
                        // キー比較を諦めるだけに留め、実際の JSON 妥当性検証は
                        // 呼び出し元の再パースに委ねる。
                        if let Some((code_point, consumed)) = decode_unicode_escape(bytes, i + 1) {
                            result.extend_from_slice(&bytes[i + 1..i + 1 + consumed]);
                            if let Some(decoded_char) = char::from_u32(code_point) {
                                let mut char_buf = [0u8; 4];
                                string_buf.extend_from_slice(
                                    decoded_char.encode_utf8(&mut char_buf).as_bytes(),
                                );
                            }
                            i += consumed;
                        } else {
                            // 比較不能マーカー（"scores" とは一致しない任意の
                            // バイト）を積み、以降このキー名の照合を諦める。
                            string_buf.push(0);
                        }
                    }
                    _ => {
                        // JSON として不正なエスケープ文字。同様に比較不能
                        // マーカーを積む（後続の再パースで自然に停止する）。
                        string_buf.push(0);
                    }
                }
                i += 1;
                continue;
            }
            result.push(byte);
            if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
                if depth == 1 {
                    // 直後（空白を除く）が `:` ならこの文字列はトップレベル
                    // キーなので、以降の値がそのキー配下であることの目印として
                    // 記録する。
                    let mut j = i + 1;
                    while j < bytes.len() && matches!(bytes[j], b' ' | b'\t' | b'\r' | b'\n') {
                        j += 1;
                    }
                    if j < bytes.len() && bytes[j] == b':' {
                        current_top_key = Some(std::mem::take(&mut string_buf));
                    }
                }
                string_buf.clear();
            } else {
                string_buf.push(byte);
            }
            i += 1;
            continue;
        }

        if byte == b'"' {
            in_string = true;
            in_value_position = false;
            string_buf.clear();
            result.push(byte);
            i += 1;
            continue;
        }

        if in_value_position
            && current_top_key.as_deref() == Some(SCORES_KEY)
            && let Some(token) = TOKENS.iter().find(|t| bytes[i..].starts_with(t.as_bytes()))
        {
            let after = i + token.len();
            let boundary_ok = after >= bytes.len()
                || matches!(
                    bytes[after],
                    b' ' | b'\t' | b',' | b'}' | b']' | b'\r' | b'\n'
                );
            if boundary_ok {
                result.extend_from_slice(b"null");
                replaced = true;
                i += token.len();
                in_value_position = false;
                continue;
            }
        }

        // `1e400` のような、JSON 数値としての構文自体は正しいが f64 の
        // 表現範囲を超える（無限大になる）トークンを `null` へ置換する
        // （レビュー指摘 PR #204 threadId PRRT_kwDOUq-SxM6mfO20）。
        // `serde_json::from_str` はこの種のトークンを `Err`（"number out of
        // range"）として拒否し、`NaN`／`Infinity` リテラルの置換（上記
        // TOKENS）は対象外のため素通りして [`EvalInputStop::MalformedJson`]
        // になり、ケース 10 の「不正スコアは error として分母に含める」契約
        // （[`ErrorOrigin::InvalidScore`]）を破っていた。数値本体を
        // [`scan_json_number_token`] で構文どおりに読み取り、`f64::parse`
        // が無限大を返す場合だけ `null` に置換する（有限値はそのまま残し、
        // 元々パースできていた数値の挙動を変えない）。
        if in_value_position
            && current_top_key.as_deref() == Some(SCORES_KEY)
            && (byte == b'-' || byte.is_ascii_digit())
            && let Some(token_len) = scan_json_number_token(&bytes[i..])
        {
            let token_bytes = &bytes[i..i + token_len];
            let overflows = std::str::from_utf8(token_bytes)
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .is_some_and(|v| !v.is_finite());
            if overflows {
                result.extend_from_slice(b"null");
                replaced = true;
            } else {
                result.extend_from_slice(token_bytes);
            }
            i += token_len;
            in_value_position = false;
            continue;
        }

        match byte {
            b'{' | b'[' => {
                depth += 1;
                in_value_position = true;
            }
            b'}' | b']' => {
                depth -= 1;
                if depth <= 1 {
                    current_top_key = None;
                }
                in_value_position = false;
            }
            b':' => in_value_position = true,
            b',' => {
                if depth == 1 {
                    current_top_key = None;
                }
                in_value_position = true;
            }
            b' ' | b'\t' | b'\r' | b'\n' => {} // 空白は位置判定を変えない
            _ => in_value_position = false,
        }
        result.push(byte);
        i += 1;
    }

    if replaced {
        String::from_utf8(result).ok()
    } else {
        None
    }
}

/// `\uXXXX` エスケープ（`start` は `\u` 直後の16進4桁の開始位置）を読み、
/// コードポイントと消費した16進バイト数（4、サロゲートペアなら10）を返す。
/// 高サロゲート（`0xD800`〜`0xDBFF`）の直後に低サロゲート（`0xDC00`〜
/// `0xDFFF`）の `\uXXXX` が続く場合は結合し、続かない場合は孤立サロゲートの
/// コードポイントをそのまま返す（呼び出し元は `char::from_u32` が `None` に
/// なることで無視する）。桁不足・非16進など形式が不正な場合は `None` を返す
/// ([`substitute_non_finite_literals`] のトップレベルキー検出専用の
/// 簡易デコーダで、JSON 全体の妥当性検証は呼び出し元の再パースに委ねる)。
fn decode_unicode_escape(bytes: &[u8], start: usize) -> Option<(u32, usize)> {
    let high = parse_hex4(bytes, start)?;
    if (0xD800..=0xDBFF).contains(&high) {
        let low_start = start + 4;
        if bytes.get(low_start) == Some(&b'\\')
            && bytes.get(low_start + 1) == Some(&b'u')
            && let Some(low) = parse_hex4(bytes, low_start + 2)
            && (0xDC00..=0xDFFF).contains(&low)
        {
            let code_point = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
            return Some((code_point, 10));
        }
        return Some((high, 4));
    }
    Some((high, 4))
}

/// `bytes[start..start + 4]` を16進4桁として解釈する。範囲外・非16進なら
/// `None`。
fn parse_hex4(bytes: &[u8], start: usize) -> Option<u32> {
    let slice = bytes.get(start..start + 4)?;
    let text = std::str::from_utf8(slice).ok()?;
    u32::from_str_radix(text, 16).ok()
}

/// `bytes` の先頭から JSON 数値トークン（[RFC 8259] の `number` 生成規則:
/// 先頭の `-`・整数部・任意の小数部・任意の指数部）を構文どおりに読み取り、
/// 消費したバイト数を返す（[`substitute_non_finite_literals`] が `scores`
/// フィールド内の数値オーバーフロー検出専用に呼ぶ。再帰・バックトラック
/// をしない線形スキャン）。整数部が無い・小数部の `.` の後に数字が無い・
/// 指数部の `e`／`E` の後に数字が無いなど、構文が不正な場合は不完全な
/// 部分（`.`・`e` 自体）を消費せずに手前までの妥当な部分を返す（数値
/// 全体が不正な場合は `None`）。呼び出し元は消費後の文字列を
/// `str::parse::<f64>` で解釈するだけで、ここでは値の意味（有限性等）を
/// 判定しない。
///
/// [RFC 8259]: https://www.rfc-editor.org/rfc/rfc8259
fn scan_json_number_token(bytes: &[u8]) -> Option<usize> {
    let mut i = 0usize;
    if bytes.first() == Some(&b'-') {
        i += 1;
    }

    let int_start = i;
    if bytes.get(i) == Some(&b'0') {
        i += 1;
    } else if bytes.get(i).is_some_and(u8::is_ascii_digit) {
        while bytes.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
    }
    if i == int_start {
        // 整数部の数字が 1 桁もない（`-` のみ等）。数値トークンではない。
        return None;
    }

    if bytes.get(i) == Some(&b'.') {
        let frac_start = i + 1;
        let mut j = frac_start;
        while bytes.get(j).is_some_and(u8::is_ascii_digit) {
            j += 1;
        }
        if j > frac_start {
            i = j;
        }
        // `.` の直後に数字が無ければ小数部を消費しない（整数部までを返す）。
    }

    if matches!(bytes.get(i), Some(b'e') | Some(b'E')) {
        let mut j = i + 1;
        if matches!(bytes.get(j), Some(b'+') | Some(b'-')) {
            j += 1;
        }
        let exp_digits_start = j;
        while bytes.get(j).is_some_and(u8::is_ascii_digit) {
            j += 1;
        }
        if j > exp_digits_start {
            i = j;
        }
        // 指数部に数字が無ければ `e`／符号を消費しない。
    }

    Some(i)
}

/// pred の `scores` フィールドが不正か判定する（ケース 10・TASK-23.1-2）。
///
/// `scores` キーが無い、または `null` の場合は検査しない（正常）。object
/// でない、値が JSON の数値でない（真偽値・文字列・null 等を含む）、
/// 有限でない（NaN・±Infinity）、負値、のいずれかがあれば不正。合計が
/// `1.0` から [`SCORE_SUM_TOLERANCE`] を超えて外れる場合も不正（空の
/// object は合計 0 のため不正になる）。スコアのキーが定義済みラベルと
/// 一致するかは検査しない（PoC-9 と同じ）。不正の理由は細分化せず
/// [`ErrorOrigin::InvalidScore`] の 1 種にまとめ、理由文字列や値は
/// 保持しない（生値を出さない方針）。
fn scores_are_invalid(fields: &Map<String, Value>) -> bool {
    let Some(scores) = fields.get("scores") else {
        return false;
    };
    if scores.is_null() {
        return false;
    }
    let Some(map) = scores.as_object() else {
        return true;
    };

    let mut sum = 0.0f64;
    for value in map.values() {
        let Some(n) = value.as_f64() else {
            return true;
        };
        if !n.is_finite() || n < 0.0 {
            return true;
        }
        sum += n;
    }
    (sum - 1.0).abs() > SCORE_SUM_TOLERANCE
}

/// スコア合計が `1.0` から外れてよい許容差（ケース 10・TASK-23.1-2）。
///
/// PoC-9 addendum A-4 が定めた値であり、評価契約の指標一致判定の許容差
/// `1e-9`（`.claude/rules/evaluation-contract.md`「決定性」）とは別物である。
/// 前者は pred のスコアという外部入力の妥当性検査、後者は本ツールが計算した
/// 指標の再現性検証という別の目的に使う値のため、混同して緩めない。
///
/// `crates/core/src/judgment.rs` の同名定数と値が重複しているが、
/// 本 crate は `fandhe-edge-core` に依存しない設計（`crates/data/src/lib.rs`
/// 「層の境界」参照）であるため、値を 1 箇所の定数へ集約する代わりに共有
/// fixture `fixtures/score_tolerance/score_sum_tolerance.json` を単一真実源
/// とし、両 crate のテスト（本 crate の `tests/score_sum_tolerance_fixture.rs`・
/// `fandhe-edge-core` の `tests/score_sum_tolerance_fixture.rs`）が同じ
/// fixture を読み込んで自身の定数と照合する（`fixtures/exitcode/exit_codes.json`
/// による終了コードの Rust／学習ワーカー間照合〔#179〕と同じパターン）。
/// 片方だけを変更すると fixture との不一致でテストが失敗するため、値の
/// 乖離は機械的に検出される（PR #202 レビュー指摘・P1 対応）。
pub const SCORE_SUM_TOLERANCE: f64 = 1e-6;

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

/// 正規化した `input` でグループ化し、(重複行, 矛盾行) を返す（手順 6・
/// ケース 6・7）。同じグループ内の gold ラベルがすべて一致すれば「重複」
/// （除外しない。[`WarningCode::DuplicateInputWithinSplit`]）、2 種以上の
/// ラベルが混在すれば「矛盾」（除外する。[`WarningCode::ContradictoryInput`]。
/// ケース 7）とする。呼び出し元は矛盾行を `accepted` から取り除き、
/// pred との突き合わせ（手順 7）へ進めない。
///
/// TASK-23.1-1（issue #55・PR #203）時点ではラベルまで見ずに正規化 `input`
/// だけでグループ化しており、`A, A, B` のように同じ `input` にラベルが混在
/// する場合に重複行 `A, A` まで警告から漏れる不具合があった（issue #55
/// レビュー指摘）。TASK-23.1-2（ケース 7）でラベル混在グループを丸ごと
/// 「矛盾」として `accepted` から除外する仕様を追加したことで、混在グループを
/// 部分的に「重複」扱いする経路が無くなり、上記の不具合は再現しなくなった
/// （矛盾グループはそもそも重複警告の対象にしない）。
fn find_duplicate_input_lines(
    rows: &[(usize, String, &Map<String, Value>)],
) -> (BTreeSet<usize>, BTreeSet<usize>) {
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

    let mut duplicated = BTreeSet::new();
    let mut contradictory = BTreeSet::new();
    for members in groups.values() {
        if members.len() < 2 {
            continue;
        }
        let Some(&(_, first_label)) = members.first() else {
            continue;
        };
        if members.iter().all(|(_, label)| *label == first_label) {
            duplicated.extend(members.iter().map(|(line, _)| *line));
        } else {
            contradictory.extend(members.iter().map(|(line, _)| *line));
        }
    }
    (duplicated, contradictory)
}

/// pred 1 行の `status`/`predicted_label` を分類する（手順 8）。
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

/// `lines_by_code` から `order` に列挙された [`WarningCode`] だけを、その順に
/// 取り出して [`EvalInputWarning`] へ組み立てる（該当が無いものは含めない）。
/// 取り出した entry は `lines_by_code` から取り除くため、部分的な `order`
/// （手順 6 の Exclude 系のみ等）を渡しても、残りは後続の呼び出し
/// （手順 9 の全体整列）へそのまま引き継がれる。
fn drain_warnings_in_order(
    lines_by_code: &mut BTreeMap<WarningCode, BTreeSet<usize>>,
    order: &[WarningCode],
) -> Vec<EvalInputWarning> {
    let mut warnings = Vec::new();
    for &code in order {
        if let Some(lines) = lines_by_code.remove(&code) {
            warnings.push(EvalInputWarning {
                code,
                action: code.action(),
                side: code.side(),
                lines: lines.into_iter().collect(),
                labels: Vec::new(),
            });
        }
    }
    warnings
}

/// gold（正解）・pred（予測）の JSONL 本文から評価対象の行を組み立てる
/// （REQ-23・TASK-23.1-1・REQ-23 境界値・TASK-23.2）。
///
/// # 手順（モジュール doc・PoC-9 `metrics.py` v1.1 の手順を踏襲）
///
/// 1. gold・pred をそれぞれ行単位でパースする（[`parse_side`]。pred 側は
///    厳密パース失敗時に限り NaN・Infinity の緩和パースを試みる）
/// 2. gold が 0 レコードなら [`EvalInputStop::EmptyData`] で停止する
/// 3. `valid_label_ids` が空集合なら [`EvalInputStop::EmptyLabelSet`] で停止する
/// 4. gold 側・pred 側それぞれで `id` の重複を検査し、あれば
///    [`EvalInputStop::DuplicateId`] で停止する（gold 側を先に判定する）
/// 5. gold の欠陥（欠落・型不正・未知ラベル）を除外する
///    （[`WarningCode::MissingGold`]・[`MalformedGold`](WarningCode::MalformedGold)・
///    [`UnknownGoldLabel`](WarningCode::UnknownGoldLabel)）
/// 6. 手順 5 の除外で評価対象が 0 件になったら [`EvalInputStop::NoValidGold`]
///    で停止する（評価契約の fail-closed。`Ok` で `active: []` を返さない）
/// 7. 手順 5 を通過した行のうち、`input` が
///    [`crate::preprocess_boundary::is_empty_after_normalization`] を満たす
///    行を [`WarningCode::EmptyInput`] として警告する（除外しない。矛盾除外
///    より前に検出するため、後述の矛盾除外に巻き込まれた行も報告される。
///    REQ-23 境界値・TASK-23.2）。続けて正規化 `input` でグループ化し
///    （[`find_duplicate_input_lines`]）、gold ラベルが一致するグループは
///    [`WarningCode::DuplicateInputWithinSplit`] として警告する（除外しない。
///    ケース 6）。2 種以上に分かれるグループは [`WarningCode::ContradictoryInput`]
///    として警告し、`accepted` から除外する（ケース 7。pred との突き合わせ・
///    スコア検査へは進まない）。この矛盾除外で `accepted` が 0 件になったら、
///    手順 6 と同じ [`EvalInputStop::NoValidGold`] で停止する（評価契約の
///    fail-closed。矛盾除外後も `Ok` で `active: []` を返さない）
/// 8. 手順 7 を通過した行それぞれに対応する pred 行を突き合わせる。対応する
///    pred 行の `scores` が不正、または NaN・Infinity の緩和パースを経由して
///    いた場合は `status` に関わらず [`PredictionOutcome::Error`]
///    （[`ErrorOrigin::InvalidScore`]）とし [`WarningCode::InvalidScore`] を
///    警告する（ケース 10）。それ以外は [`PredictionOutcome`] に分類する。
///    対応する pred 行が無い場合は [`PredictionOutcome::Error`]
///    （[`ErrorOrigin::MissingPrediction`]）とし、[`WarningCode::MissingPrediction`]
///    を警告する。gold に存在しない `id` の pred 行は無視する
/// 9. `active` が空でない場合に限り、データセット単位の警告を追加する
///    （PoC-9 は `active` が 0 件なら早期に return しこれらを出さない）。
///    全行が [`PredictionOutcome::Abstain`] なら [`WarningCode::AllAbstain`]
///    （ケース 11）、全行が [`PredictionOutcome::Error`] なら
///    [`WarningCode::AllError`]（[`PredictionOutcome::Invalid`] は対象外。
///    ケース 12）、`valid_label_ids` のうち `active` な gold 行に出現しない
///    ものがあれば [`WarningCode::UnseenClass`] とし [`EvalInputOutcome::unseen_labels`]
///    にも同じ内容を入れる（ケース 9）
/// 10. 警告は Exclude 系 → IncludeAsError 系 → WarnInclude 系の順、
///     各群の中は [`WarningCode`] の宣言順に並べる（`lines` は昇順。
///     [`WarningCode::EmptyInput`] は WarnInclude 系の末尾から 2 番目、
///     [`WarningCode::UnseenClass`] が WarnInclude 系の最後）
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

    // 手順 6: 手順 5 の除外で評価対象が 0 件になっていないか確認する。
    // gold に行があっても、全行が MissingGold・MalformedGold・UnknownGoldLabel の
    // いずれかで除外されると `accepted` が空になる。ここで停止しないと
    // 呼び出し側が `active: []` を「評価済みで対象 0 件」と区別できず、
    // 評価契約の fail-closed（`.claude/rules/evaluation-contract.md`）に反する。
    if accepted.is_empty() {
        const EXCLUDE_ORDER: [WarningCode; 3] = [
            WarningCode::MissingGold,
            WarningCode::MalformedGold,
            WarningCode::UnknownGoldLabel,
        ];
        let warnings = drain_warnings_in_order(&mut lines_by_code, &EXCLUDE_ORDER);
        return Err(EvalInputStop::NoValidGold {
            gold_rows: gold_rows.len(),
            pred_rows: pred_rows.len(),
            warnings,
        });
    }

    // 手順 7: 正規化 input によるグループ化（重複・矛盾の検出。ケース 6・7）。
    let dedup_input: Vec<(usize, String, &Map<String, Value>)> = accepted
        .iter()
        .zip(accepted_labels.iter())
        .map(|(row, label)| (row.line, label.clone(), &row.fields))
        .collect();

    // 手順 7（REQ-23 境界値・TASK-23.2・issue #57）: gold の `input` が学習
    // ワーカーの正規化後に空となる行（推論経路と評価経路で前処理結果が
    // 食い違いうる境界。`crate::preprocess_boundary` モジュール doc・
    // PoC-16 の `[0]` vs `[]` 参照）を検出する。矛盾除外（本手順の後半）より
    // 前に積むことで、空白のみの `input` が複数あり異なるラベルで矛盾扱いに
    // なった場合も報告が漏れない。`input` フィールドが無い・文字列でない行は
    // 対象外（[`find_duplicate_input_lines`] と同じ扱い）。
    let empty_input_lines: BTreeSet<usize> = dedup_input
        .iter()
        .filter_map(|(line, _, fields)| {
            let input = fields.get("input")?.as_str()?;
            crate::preprocess_boundary::is_empty_after_normalization(input).then_some(*line)
        })
        .collect();
    if !empty_input_lines.is_empty() {
        lines_by_code
            .entry(WarningCode::EmptyInput)
            .or_default()
            .extend(empty_input_lines);
    }

    let (duplicate_input_lines, contradictory_lines) = find_duplicate_input_lines(&dedup_input);
    if !duplicate_input_lines.is_empty() {
        lines_by_code
            .entry(WarningCode::DuplicateInputWithinSplit)
            .or_default()
            .extend(duplicate_input_lines);
    }
    if !contradictory_lines.is_empty() {
        lines_by_code
            .entry(WarningCode::ContradictoryInput)
            .or_default()
            .extend(contradictory_lines.iter().copied());
        // 矛盾したグループは分母から除外し、pred との突き合わせ・スコア検査へ
        // 進めない（モジュール doc・[`WarningCode::ContradictoryInput`] 参照）。
        let (filtered_rows, filtered_labels): (Vec<&ParsedRow>, Vec<String>) = accepted
            .into_iter()
            .zip(accepted_labels)
            .filter(|(row, _)| !contradictory_lines.contains(&row.line))
            .unzip();
        accepted = filtered_rows;
        accepted_labels = filtered_labels;
    }

    // 手順 7b: 矛盾除外（手順 7）で評価対象が 0 件になっていないか確認する。
    // 手順 6 と同じ理由（評価契約の fail-closed。呼び出し側が `active: []` を
    // 「評価済みで対象 0 件」と誤認しない）で、矛盾除外後にも空集合を検査する
    // （レビュー指摘。PR #204）。この時点の `lines_by_code` には手順 5 の
    // Exclude 系警告に加え、DuplicateInputWithinSplit・ContradictoryInput も
    // 積まれているため、[`NoValidGold::warnings`] には Exclude 系だけでなく
    // 矛盾除外の内訳も含める。
    //
    // EmptyInput も含める（codex レビュー指摘。PR #212）: [`drain_warnings_in_order`]
    // は `order` に列挙されなかった code を `lines_by_code` へ残したまま
    // 返るが、この関数はここで早期 return するため、含めなかった code は
    // 呼び出し元へ届かず消える。手順 7 で `accepted` 全行が矛盾除外されると
    // EmptyInput（空白のみの `input`）の検知が報告されないまま停止し、
    // REQ-23 境界値・TASK-23.2 の「検知・報告」を満たさなくなるため、
    // EXCLUDE_ORDER の末尾へ加えて必ず持ち出す。
    if accepted.is_empty() {
        const EXCLUDE_ORDER: [WarningCode; 5] = [
            WarningCode::MissingGold,
            WarningCode::MalformedGold,
            WarningCode::UnknownGoldLabel,
            WarningCode::ContradictoryInput,
            WarningCode::EmptyInput,
        ];
        let warnings = drain_warnings_in_order(&mut lines_by_code, &EXCLUDE_ORDER);
        return Err(EvalInputStop::NoValidGold {
            gold_rows: gold_rows.len(),
            pred_rows: pred_rows.len(),
            warnings,
        });
    }

    // 手順 8: pred との突き合わせ（スコア検査を含む）。
    let mut pred_by_id: BTreeMap<&str, &ParsedRow> = BTreeMap::new();
    for row in &pred_rows {
        pred_by_id.insert(row.id.as_str(), row);
    }

    let mut active = Vec::with_capacity(accepted.len());
    for (row, label) in accepted.into_iter().zip(accepted_labels) {
        match pred_by_id.get(row.id.as_str()) {
            Some(pred_row) => {
                let prediction =
                    if pred_row.non_finite_literal || scores_are_invalid(&pred_row.fields) {
                        lines_by_code
                            .entry(WarningCode::InvalidScore)
                            .or_default()
                            .insert(pred_row.line);
                        PredictionOutcome::Error(ErrorOrigin::InvalidScore)
                    } else {
                        classify_prediction(&pred_row.fields, valid_label_ids)
                    };
                active.push(ActiveRow {
                    gold_line: row.line,
                    id: row.id.clone(),
                    gold_label: label,
                    prediction,
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

    // 手順 9: データセット単位の警告（`active` が空でないときだけ。
    // PoC-9 は `active` が 0 件なら早期に return しこれらを出さない挙動に揃える）。
    let mut unseen_labels: Vec<String> = Vec::new();
    if !active.is_empty() {
        if active
            .iter()
            .all(|row| matches!(row.prediction, PredictionOutcome::Abstain))
        {
            // side: Prediction のため gold_line ではなく pred_line を使う
            // （gold と pred の順序が異なる場合に別の予測行を指してしまう
            // 不具合の修正。レビュー指摘。PR #204）。
            let lines: BTreeSet<usize> = active.iter().filter_map(|row| row.pred_line).collect();
            lines_by_code
                .entry(WarningCode::AllAbstain)
                .or_default()
                .extend(lines);
        }
        if active
            .iter()
            .all(|row| matches!(row.prediction, PredictionOutcome::Error(_)))
        {
            // `side: Prediction` だが、`MissingPrediction` による Error は
            // pred_line が存在しない（予測行が無い）。全行が
            // `MissingPrediction` 起因の場合に `lines` が空になり、警告が
            // 対象行を指せなくなる不具合の修正（レビュー指摘。PR #204）。
            // ただし pred_line を持つ行が 1 件でもあれば、そちらだけを使い
            // gold_line へのフォールバックは行わない（gold 側の行番号と
            // pred 側の行番号は別々の採番空間のため、両者を同じ
            // `BTreeSet<usize>` へ混在させると偶然同じ数値になった場合に
            // 1 件へ潰れてしまい、無関係な pred 行を指したり対象行を
            // 取りこぼしたりする。Bugbot 指摘。PR #204
            // threadId PRRT_kwDOUq-SxM6mfP3Y）。MissingPrediction 自体の
            // 対象行は別途 `WarningCode::MissingPrediction`（常に gold_line
            // 採番）が報告するため、混在させなくても情報は失われない。
            let has_pred_line = active.iter().any(|row| row.pred_line.is_some());
            let lines: BTreeSet<usize> = if has_pred_line {
                active.iter().filter_map(|row| row.pred_line).collect()
            } else {
                active.iter().map(|row| row.gold_line).collect()
            };
            lines_by_code
                .entry(WarningCode::AllError)
                .or_default()
                .extend(lines);
        }

        let present_labels: BTreeSet<&str> =
            active.iter().map(|row| row.gold_label.as_str()).collect();
        unseen_labels = valid_label_ids
            .iter()
            .filter(|label| !present_labels.contains(label.as_str()))
            .cloned()
            .collect();
    }

    // 手順 10: Exclude 系 → IncludeAsError 系 → WarnInclude 系の順に並べる。
    // UnseenClass は行番号ではなくラベルで示すため `lines_by_code` に乗せず、
    // WarnInclude 系の最後に個別に追加する。
    const ORDER: [WarningCode; 10] = [
        WarningCode::MissingGold,
        WarningCode::MalformedGold,
        WarningCode::UnknownGoldLabel,
        WarningCode::ContradictoryInput,
        WarningCode::InvalidScore,
        WarningCode::MissingPrediction,
        WarningCode::DuplicateInputWithinSplit,
        WarningCode::AllAbstain,
        WarningCode::AllError,
        WarningCode::EmptyInput,
    ];
    let mut warnings = drain_warnings_in_order(&mut lines_by_code, &ORDER);
    if !unseen_labels.is_empty() {
        warnings.push(EvalInputWarning {
            code: WarningCode::UnseenClass,
            action: WarningCode::UnseenClass.action(),
            side: WarningCode::UnseenClass.side(),
            lines: Vec::new(),
            labels: unseen_labels.clone(),
        });
    }

    Ok(EvalInputOutcome {
        active,
        warnings,
        unseen_labels,
    })
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

    /// REQ-23: gold のトップレベル重複キーは DuplicateKey で打ち切る。
    #[test]
    fn req23_duplicate_top_level_key_stops() {
        let gold = "{\"id\":\"a\",\"id\":\"b\",\"label\":\"A\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::DuplicateKey {
                side: Side::Gold,
                line: 1,
            })
        );
        assert_eq!(result.unwrap_err().code(), "duplicate_key");
    }

    /// REQ-23: ネスト先（`output.intent`）の重複キーも DuplicateKey で打ち切る。
    #[test]
    fn req23_duplicate_nested_key_stops() {
        let gold = "{\"id\":\"a\",\"output\":{\"intent\":\"A\",\"intent\":\"B\"}}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A", "B"]));
        assert_eq!(
            result,
            Err(EvalInputStop::DuplicateKey {
                side: Side::Gold,
                line: 1,
            })
        );
    }

    /// REQ-23: Unicode エスケープによる `id` キー重複 smuggling
    /// （`"id"` と `"\u0069d"` はいずれも `id` を指す）を検出して打ち切ること。
    /// 視認できるテキストと異なる `id`（ここでは後勝ちの `"b"`）で評価が
    /// 進むことを防ぐ（モジュール doc「PoC-9 との差分」参照）。
    #[test]
    fn req23_duplicate_key_via_unicode_escape_stops() {
        let gold = "{\"id\":\"a\",\"\\u0069d\":\"b\",\"label\":\"A\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::DuplicateKey {
                side: Side::Gold,
                line: 1,
            })
        );
    }

    /// REQ-23: pred 側の重複キーも検出すること（gold は正常）。
    #[test]
    fn req23_duplicate_key_on_prediction_side_stops() {
        let gold = "{\"id\":\"a\",\"label\":\"A\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\",\"predicted_label\":\"B\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A", "B"]));
        assert_eq!(
            result,
            Err(EvalInputStop::DuplicateKey {
                side: Side::Prediction,
                line: 1,
            })
        );
    }

    /// REQ-23: 重複キー検出は `id` の抽出より前に働くため、`id` 自体が
    /// smuggling されて偶然重複した場合でも InvalidId ではなく DuplicateKey で
    /// 打ち切ること（重複 id 判定〔手順 4〕より前の、パース段階〔手順 1〕の
    /// 停止であることの確認）。
    #[test]
    fn req23_duplicate_key_is_detected_before_duplicate_id_check() {
        let gold =
            "{\"id\":\"a\",\"label\":\"A\"}\n{\"id\":\"a\",\"\\u0069d\":\"a\",\"label\":\"B\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A", "B"]));
        assert_eq!(
            result,
            Err(EvalInputStop::DuplicateKey {
                side: Side::Gold,
                line: 2,
            })
        );
    }

    /// 重複キーの診断（[`EvalInputStop::DuplicateKey`]）に生値（`id`・`label`）が
    /// 含まれないこと（PR #191（issue #38）の前例に倣う回帰テスト）。
    #[test]
    fn req23_duplicate_key_diagnostics_never_contain_raw_values() {
        let long_id = "x".repeat(500);
        let gold = format!("{{\"id\":\"{long_id}\",\"id\":\"other-{long_id}\"}}\n");
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(&gold, pred, &labels(&["A"]));
        let err = result.unwrap_err();
        assert_eq!(
            err,
            EvalInputStop::DuplicateKey {
                side: Side::Gold,
                line: 1
            }
        );
        assert!(!format!("{err:?}").contains(&long_id));
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

    /// ラベルが食い違うグループ（矛盾）は重複として警告せず、
    /// [`WarningCode::ContradictoryInput`] として除外する（ケース 7・
    /// TASK-23.1-2。issue #56 で `req23_contradictory_input_is_not_reported_as_duplicate`
    /// から更新。以前は矛盾の専用扱いが未実装で、無警告のまま両方 active に
    /// 残っていた）。この fixture は gold が矛盾グループの 2 行だけのため、
    /// 矛盾除外後に `active` が 0 件になり [`EvalInputStop::NoValidGold`] で
    /// 停止する（レビュー指摘。PR #204。`Ok` で `active: []` を返さない）。
    #[test]
    fn req23_contradictory_input_is_excluded_not_reported_as_duplicate() {
        let gold = "{\"id\":\"a\",\"input\":\"foo\",\"label\":\"A\"}\n{\"id\":\"b\",\"input\":\"foo\",\"label\":\"B\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n{\"id\":\"b\",\"status\":\"ok\",\"predicted_label\":\"B\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A", "B"]));
        match result {
            Err(EvalInputStop::NoValidGold {
                gold_rows,
                pred_rows,
                warnings,
            }) => {
                assert_eq!(gold_rows, 2);
                assert_eq!(pred_rows, 2);
                assert_eq!(warnings.len(), 1);
                assert_eq!(warnings[0].code, WarningCode::ContradictoryInput);
                assert_eq!(warnings[0].action, WarningAction::Exclude);
                assert_eq!(warnings[0].side, Side::Gold);
                assert_eq!(warnings[0].lines, vec![1, 2]);
            }
            other => panic!("expected NoValidGold, got {other:?}"),
        }
    }

    /// [`WarningCode::MissingPrediction`]: gold に対応する pred 行が無い場合、
    /// 除外せず [`PredictionOutcome::Error`]（[`ErrorOrigin::MissingPrediction`]）
    /// として含め、[`WarningAction::IncludeAsError`] で警告すること
    /// （手順 8。レビュー指摘: issue #55 の未カバー分岐）。
    #[test]
    fn req23_missing_prediction_is_included_as_error() {
        let gold = "{\"id\":\"a\",\"label\":\"A\"}\n{\"id\":\"b\",\"label\":\"A\"}\n";
        // b に対応する pred 行が無い。
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();

        assert_eq!(outcome.warnings.len(), 1);
        let warning = &outcome.warnings[0];
        assert_eq!(warning.code, WarningCode::MissingPrediction);
        assert_eq!(warning.action, WarningAction::IncludeAsError);
        assert_eq!(warning.side, Side::Prediction);
        assert_eq!(warning.lines, vec![2]);

        assert_eq!(outcome.active.len(), 2);
        let missing = outcome
            .active
            .iter()
            .find(|row| row.gold_line == 2)
            .expect("gold_line=2 must be present as an active row");
        assert_eq!(
            missing.prediction,
            PredictionOutcome::Error(ErrorOrigin::MissingPrediction)
        );
        assert_eq!(missing.pred_line, None);
    }

    /// 手順 9: 複数種別の警告が同時発生した場合、Exclude 系
    /// （[`WarningCode::MissingGold`]）→ IncludeAsError 系
    /// （[`WarningCode::MissingPrediction`]）→ WarnInclude 系
    /// （[`WarningCode::DuplicateInputWithinSplit`]）の順に並ぶこと。
    /// `ORDER` 定数は手動保守のため、`WarningCode` が追加された際に
    /// 追加漏れがあればこのテストが検出する（レビュー指摘: issue #55）。
    #[test]
    fn req23_multiple_warning_kinds_are_ordered_exclude_then_error_then_warn() {
        let gold = concat!(
            "{\"id\":\"g1\",\"label\":null}\n", // MissingGold（除外）
            "{\"id\":\"g2\",\"input\":\"dup\",\"label\":\"A\"}\n",
            "{\"id\":\"g3\",\"input\":\"dup\",\"label\":\"A\"}\n", // g2 と重複（警告のみ）
            "{\"id\":\"g4\",\"label\":\"A\"}\n",                   // pred 行なし
        );
        let pred = concat!(
            "{\"id\":\"g2\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
            "{\"id\":\"g3\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
        );
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();

        let codes: Vec<WarningCode> = outcome.warnings.iter().map(|w| w.code).collect();
        assert_eq!(
            codes,
            vec![
                WarningCode::MissingGold,
                WarningCode::MissingPrediction,
                WarningCode::DuplicateInputWithinSplit,
            ]
        );
        assert_eq!(outcome.warnings[0].action, WarningAction::Exclude);
        assert_eq!(outcome.warnings[0].lines, vec![1]);
        assert_eq!(outcome.warnings[1].action, WarningAction::IncludeAsError);
        assert_eq!(outcome.warnings[1].lines, vec![4]);
        assert_eq!(outcome.warnings[2].action, WarningAction::WarnInclude);
        assert_eq!(outcome.warnings[2].lines, vec![2, 3]);

        // g1 は除外、g2・g3・g4 は含まれる（g4 は Error として含まれる）。
        assert_eq!(outcome.active.len(), 3);
        let g4 = outcome
            .active
            .iter()
            .find(|row| row.gold_line == 4)
            .expect("gold_line=4 must be present as an active row");
        assert_eq!(
            g4.prediction,
            PredictionOutcome::Error(ErrorOrigin::MissingPrediction)
        );
    }

    /// 同じ正規化 input に対して同一ラベルの重複と、食い違うラベルの行が
    /// 混在する場合（`A, A, B`）、TASK-23.1-1（PR #203）時点ではラベルまで
    /// 見ずグループ化していたため先頭 2 行（`A, A`）の重複警告まで消える
    /// 不具合があり、`(正規化 input, ラベル)` の組でグループ化して修正した
    /// （3 行目の `B` は警告にもどちらの分母除外にも含めない、という判断
    /// だった）。TASK-23.1-2（issue #56）で [`WarningCode::ContradictoryInput`]
    /// （ケース 7）を追加したことにより、正規化 input が同じで gold ラベルが
    /// 食い違う行を「サイレントに許容する」選択肢自体を無くしたため、この
    /// グループは丸ごと矛盾として扱い分母から除外する（[`find_duplicate_input_lines`]
    /// の doc「以前の実装との差分」参照。旧実装の「部分的な重複警告」より、
    /// ラベルの食い違いを常に検出・除外する方が評価契約上安全側に倒せる）。
    #[test]
    fn req23_mixed_labels_in_one_group_are_fully_contradictory() {
        let gold = "{\"id\":\"a\",\"input\":\"foo\",\"label\":\"A\"}\n{\"id\":\"b\",\"input\":\"foo\",\"label\":\"A\"}\n{\"id\":\"c\",\"input\":\"foo\",\"label\":\"B\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n{\"id\":\"b\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n{\"id\":\"c\",\"status\":\"ok\",\"predicted_label\":\"B\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A", "B"]));
        // gold の全 3 行が単一の矛盾グループに属し、矛盾除外後に `active` が
        // 0 件になるため [`EvalInputStop::NoValidGold`] で停止する（`Ok` で
        // `active: []` を返さない。レビュー指摘。PR #204）。
        match result {
            Err(EvalInputStop::NoValidGold {
                gold_rows,
                pred_rows,
                warnings,
            }) => {
                assert_eq!(gold_rows, 3);
                assert_eq!(pred_rows, 3);
                assert_eq!(warnings.len(), 1);
                assert_eq!(warnings[0].code, WarningCode::ContradictoryInput);
                // 3 行すべてが同一正規化 input でラベルが混在するため、丸ごと
                // 矛盾として除外する（部分的な重複警告は行わない）。
                assert_eq!(warnings[0].lines, vec![1, 2, 3]);
            }
            other => panic!("expected NoValidGold, got {other:?}"),
        }
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
    /// PR #191（issue #38）の前例に倣う）。除外対象（1 行目）のほかに有効な
    /// gold 行（2 行目）を含め、[`EvalInputStop::NoValidGold`]（本テストの
    /// 対象外。req23_no_valid_gold_* 系で別途検証する）に落ちないようにする。
    #[test]
    fn req23_warnings_never_contain_raw_values() {
        let long_id = "x".repeat(500);
        let gold =
            format!("{{\"id\":\"{long_id}\",\"label\":null}}\n{{\"id\":\"g2\",\"label\":\"A\"}}\n");
        let pred = format!(
            "{{\"id\":\"{long_id}\",\"status\":\"ok\",\"predicted_label\":\"A\"}}\n{{\"id\":\"g2\",\"status\":\"ok\",\"predicted_label\":\"A\"}}\n"
        );
        let outcome = prepare_evaluation_input(&gold, &pred, &labels(&["A"])).unwrap();
        assert_eq!(outcome.warnings.len(), 1);
        assert_eq!(outcome.warnings[0].lines, vec![1]);
        assert_eq!(outcome.active.len(), 1);
        // EvalInputWarning のフィールドは code/action/side/lines のみで、
        // 生値を保持するフィールドが型として存在しないことをコンパイル時に保証する。
    }

    /// REQ-23・REQ-27: gold の全行が MissingGold で除外されると
    /// [`EvalInputStop::NoValidGold`] で停止する（`Ok` で `active: []` を
    /// 返して評価済みを装わない。評価契約の fail-closed）。
    #[test]
    fn req23_no_valid_gold_when_all_missing_gold() {
        let gold = "{\"id\":\"g1\",\"label\":null}\n{\"id\":\"g2\",\"label\":null}\n";
        let pred = concat!(
            "{\"id\":\"g1\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
            "{\"id\":\"g2\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
        );
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::NoValidGold {
                gold_rows: 2,
                pred_rows: 2,
                warnings: vec![EvalInputWarning {
                    code: WarningCode::MissingGold,
                    action: WarningAction::Exclude,
                    side: Side::Gold,
                    lines: vec![1, 2],
                    labels: Vec::new(),
                }],
            })
        );
    }

    /// REQ-23: gold の全行が UnknownGoldLabel で除外された場合も同様に停止する。
    #[test]
    fn req23_no_valid_gold_when_all_unknown_label() {
        let gold = "{\"id\":\"g1\",\"label\":\"Z\"}\n";
        let pred = "{\"id\":\"g1\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::NoValidGold {
                gold_rows: 1,
                pred_rows: 1,
                warnings: vec![EvalInputWarning {
                    code: WarningCode::UnknownGoldLabel,
                    action: WarningAction::Exclude,
                    side: Side::Gold,
                    lines: vec![1],
                    labels: Vec::new(),
                }],
            })
        );
    }

    /// REQ-23: MissingGold・MalformedGold・UnknownGoldLabel が混在して全行を
    /// 除外した場合、`warnings` は Exclude 系の宣言順（MissingGold →
    /// MalformedGold → UnknownGoldLabel）で並ぶ。
    #[test]
    fn req23_no_valid_gold_warnings_are_ordered_when_mixed() {
        let gold = concat!(
            "{\"id\":\"g1\",\"label\":\"Z\"}\n", // UnknownGoldLabel
            "{\"id\":\"g2\",\"label\":null}\n",  // MissingGold
            "{\"id\":\"g3\",\"label\":123}\n",   // MalformedGold
        );
        let pred = concat!(
            "{\"id\":\"g1\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
            "{\"id\":\"g2\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
            "{\"id\":\"g3\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
        );
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::NoValidGold {
                gold_rows: 3,
                pred_rows: 3,
                warnings: vec![
                    EvalInputWarning {
                        code: WarningCode::MissingGold,
                        action: WarningAction::Exclude,
                        side: Side::Gold,
                        lines: vec![2],
                        labels: Vec::new(),
                    },
                    EvalInputWarning {
                        code: WarningCode::MalformedGold,
                        action: WarningAction::Exclude,
                        side: Side::Gold,
                        lines: vec![3],
                        labels: Vec::new(),
                    },
                    EvalInputWarning {
                        code: WarningCode::UnknownGoldLabel,
                        action: WarningAction::Exclude,
                        side: Side::Gold,
                        lines: vec![1],
                        labels: Vec::new(),
                    },
                ],
            })
        );
    }

    /// REQ-23: 1 件でも有効な gold 行があれば `NoValidGold` にはならず、
    /// 通常どおり `Ok` を返す（正のコントロール）。
    #[test]
    fn req23_no_valid_gold_does_not_trigger_when_one_row_is_valid() {
        let gold = "{\"id\":\"g1\",\"label\":null}\n{\"id\":\"g2\",\"label\":\"A\"}\n";
        let pred = concat!(
            "{\"id\":\"g1\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
            "{\"id\":\"g2\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
        );
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();
        assert_eq!(outcome.active.len(), 1);
        assert_eq!(outcome.warnings.len(), 1);
        assert_eq!(outcome.warnings[0].code, WarningCode::MissingGold);
    }

    /// `code()` に `no_valid_gold` が含まれる。
    #[test]
    fn req23_no_valid_gold_code_string() {
        assert_eq!(
            EvalInputStop::NoValidGold {
                gold_rows: 0,
                pred_rows: 0,
                warnings: Vec::new(),
            }
            .code(),
            "no_valid_gold"
        );
    }

    /// `require_non_empty`。
    #[test]
    fn req23_require_non_empty() {
        let empty = crate::inspect::InspectOutcome {
            anomalies: Vec::new(),
            valid_records: Vec::new(),
            report: crate::report::summarize(0, &[], &labels(&["A"])),
        };
        assert_eq!(
            require_non_empty(&empty),
            Err(EvalInputStop::EmptyData {
                gold_rows: 0,
                pred_rows: 0,
            })
        );

        let valid_records = vec![crate::inspect::ValidRecord {
            line: 1,
            id: "a".to_string(),
            input: "hello".to_string(),
            label_id: "A".to_string(),
            output_key: "{\"intent\":\"A\"}".to_string(),
            tags: None,
            group_id: None,
        }];
        let non_empty = crate::inspect::InspectOutcome {
            anomalies: Vec::new(),
            report: crate::report::summarize(1, &valid_records, &labels(&["A"])),
            valid_records,
        };
        assert_eq!(require_non_empty(&non_empty), Ok(()));
    }

    // --- TASK-23.1-2（issue #56）: ケース 7〜12 の単体テスト -----------------

    /// 文字列リテラル内の `"NaN"`（値ではなく文字列本文）は置換されないこと。
    #[test]
    fn req23_substitute_non_finite_does_not_touch_string_literal_nan() {
        let raw = r#"{"id":"a","input":"NaN","status":"ok","predicted_label":"A"}"#;
        assert_eq!(substitute_non_finite_literals(raw), None);
    }

    /// `-Infinity` が値の位置にあれば置換されること。
    #[test]
    fn req23_substitute_non_finite_replaces_negative_infinity() {
        let raw = r#"{"id":"a","scores":{"A":-Infinity,"B":0.0}}"#;
        let substituted =
            substitute_non_finite_literals(raw).expect("must replace -Infinity token");
        let value: Value = serde_json::from_str(&substituted).expect("must reparse");
        assert_eq!(value["scores"]["A"], Value::Null);
        assert_eq!(value["scores"]["B"], serde_json::json!(0.0));
    }

    /// キー名として `"NaN"` が使われている場合は置換されないこと
    /// （文字列リテラルの内側のため）。
    #[test]
    fn req23_substitute_non_finite_does_not_touch_key_named_nan() {
        let raw = r#"{"id":"a","NaN":1}"#;
        assert_eq!(substitute_non_finite_literals(raw), None);
    }

    /// gold 側は緩和パースを行わないため、NaN を含む行は MalformedJson で停止する
    /// （評価契約の根幹データは安全側に倒す。モジュール doc「PoC-9 との差分」）。
    #[test]
    fn req23_gold_side_nan_stops_with_malformed_json() {
        let gold = "{\"id\":\"a\",\"label\":\"A\",\"scores\":{\"A\":NaN}}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::MalformedJson {
                side: Side::Gold,
                line: 1,
            })
        );
    }

    /// 緩和パースを試みても壊れた行（NaN 置換後も JSON として不正）は
    /// MalformedJson で停止すること。
    #[test]
    fn req23_pred_side_still_malformed_after_substitution_stops() {
        let gold = "{\"id\":\"a\",\"label\":\"A\"}\n";
        // NaN を置換しても波括弧が閉じておらず JSON として不正なまま。
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"scores\":{\"A\":NaN}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::MalformedJson {
                side: Side::Prediction,
                line: 1,
            })
        );
    }

    /// スコア境界: 許容差内（1±5e-7）は正常。
    #[test]
    fn req23_scores_sum_within_tolerance_is_valid() {
        let fields = serde_json::json!({"scores": {"A": 0.50000025, "B": 0.49999975}})
            .as_object()
            .unwrap()
            .clone();
        assert!(!scores_are_invalid(&fields));
    }

    /// スコア境界: 許容差を超える（1+2e-6）は不正。
    #[test]
    fn req23_scores_sum_beyond_tolerance_is_invalid() {
        let fields = serde_json::json!({"scores": {"A": 1.000002, "B": 0.0}})
            .as_object()
            .unwrap()
            .clone();
        assert!(scores_are_invalid(&fields));
    }

    /// スコア境界: 空の object は合計 0 のため不正。
    #[test]
    fn req23_scores_empty_object_is_invalid() {
        let fields = serde_json::json!({"scores": {}})
            .as_object()
            .unwrap()
            .clone();
        assert!(scores_are_invalid(&fields));
    }

    /// スコア境界: `scores: null` は検査しない（正常）。
    #[test]
    fn req23_scores_null_is_not_checked() {
        let fields = serde_json::json!({"scores": null})
            .as_object()
            .unwrap()
            .clone();
        assert!(!scores_are_invalid(&fields));
    }

    /// スコア境界: `scores` が object でない（配列）は不正。
    #[test]
    fn req23_scores_array_is_invalid() {
        let fields = serde_json::json!({"scores": [0.5, 0.5]})
            .as_object()
            .unwrap()
            .clone();
        assert!(scores_are_invalid(&fields));
    }

    /// スコア境界: 値が真偽値（数値でない）は不正。
    #[test]
    fn req23_scores_bool_value_is_invalid() {
        let fields = serde_json::json!({"scores": {"A": true, "B": 0.0}})
            .as_object()
            .unwrap()
            .clone();
        assert!(scores_are_invalid(&fields));
    }

    /// スコア境界: `-0.0` は負値ではなく正常。
    #[test]
    fn req23_scores_negative_zero_is_valid() {
        let fields = serde_json::json!({"scores": {"A": -0.0, "B": 1.0}})
            .as_object()
            .unwrap()
            .clone();
        assert!(!scores_are_invalid(&fields));
    }

    /// `status: "abstain"` でもスコアが不正なら Error(InvalidScore) が優先されること
    /// （手順 8。status の値に関わらず検査する）。
    #[test]
    fn req23_invalid_score_takes_priority_over_abstain_status() {
        let gold = "{\"id\":\"a\",\"label\":\"A\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"abstain\",\"predicted_label\":null,\"scores\":{\"A\":-0.1}}\n";
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();
        assert_eq!(outcome.active.len(), 1);
        assert_eq!(
            outcome.active[0].prediction,
            PredictionOutcome::Error(ErrorOrigin::InvalidScore)
        );
        let invalid_score = outcome
            .warnings
            .iter()
            .find(|w| w.code == WarningCode::InvalidScore)
            .expect("InvalidScore warning must be present");
        assert_eq!(invalid_score.lines, vec![1]);
        // 唯一の active 行が Error のため、AllError も同時に成立する
        // （手順 9。UnseenClass は valid_label_ids が {A} だけなので出ない）。
        assert_eq!(
            outcome.warnings.iter().map(|w| w.code).collect::<Vec<_>>(),
            vec![WarningCode::InvalidScore, WarningCode::AllError]
        );
    }

    /// `scores` 以外の無関係なフィールド（`debug_note` 等）に現れた NaN は
    /// 緩和パースの置換対象にならないため、raw の `NaN` トークンが残ったまま
    /// 再パースに失敗し [`EvalInputStop::MalformedJson`] で停止すること。
    /// 緩和パースの置換をトップレベル `scores` の値に限定する前は、`scores`
    /// と無関係な値の非有限リテラルまで `invalid_score` として誤って受理
    /// していた（レビュー指摘。PR #204。[`substitute_non_finite_literals`]
    /// の doc 参照）。
    #[test]
    fn req23_nan_in_unrelated_field_stops_with_malformed_json() {
        let gold = "{\"id\":\"a\",\"label\":\"A\"}\n";
        let pred =
            "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\",\"debug_note\":NaN}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::MalformedJson {
                side: Side::Prediction,
                line: 1,
            })
        );
    }

    /// `scores` フィールドの値に現れた NaN は、これまでどおり緩和パースで
    /// `null` へ置換され、`scores_are_invalid` により Error(InvalidScore) として
    /// 扱われること（`scores` 限定への変更後も本来のケース 10 の挙動は保つ）。
    #[test]
    fn req23_nan_in_scores_field_still_forces_invalid_score() {
        let gold = "{\"id\":\"a\",\"label\":\"A\"}\n";
        let pred =
            "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\",\"scores\":{\"A\":NaN}}\n";
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();
        assert_eq!(outcome.active.len(), 1);
        assert_eq!(
            outcome.active[0].prediction,
            PredictionOutcome::Error(ErrorOrigin::InvalidScore)
        );
    }

    /// トップレベルキーが `"\u0073cores"`（`scores` の Unicode エスケープ
    /// 表記）で書かれていても、通常の `"scores"` 表記と同じく NaN が
    /// 置換されて Error(InvalidScore) になること。`current_top_key` が
    /// キーの生バイト列をそのまま比較していたため、この表記を `scores` と
    /// 認識できず MalformedJson で全体停止していた（レビュー指摘。PR #204）。
    #[test]
    fn req23_escaped_scores_key_still_forces_invalid_score() {
        let gold = "{\"id\":\"a\",\"label\":\"A\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\",\"\\u0073cores\":{\"A\":NaN}}\n";
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();
        assert_eq!(outcome.active.len(), 1);
        assert_eq!(
            outcome.active[0].prediction,
            PredictionOutcome::Error(ErrorOrigin::InvalidScore)
        );
    }

    /// [`substitute_non_finite_literals`] 単体でも、Unicode エスケープされた
    /// `scores` キー配下の NaN を置換できること。
    #[test]
    fn req23_substitute_non_finite_handles_escaped_scores_key() {
        let raw = r#"{"id":"a","\u0073cores":{"A":NaN}}"#;
        let substituted =
            substitute_non_finite_literals(raw).expect("must replace NaN under escaped key");
        let value: Value = serde_json::from_str(&substituted).expect("must reparse");
        assert_eq!(value["\u{73}cores"]["A"], Value::Null);
    }

    /// `1e400`（構文は正しいが f64 の表現範囲を超え無限大になる数値）は
    /// `serde_json` が `"number out of range"` としてエラーにするが、
    /// [`scan_json_number_token`] による緩和パースで `null` へ置換され、
    /// ケース 10 の契約どおり [`ErrorOrigin::InvalidScore`]（分母に含め
    /// 不正解として扱う）に分類される（レビュー指摘。PR #204
    /// threadId PRRT_kwDOUq-SxM6mfO20。以前は NaN/Infinity トークンでは
    /// ない以上緩和パースの対象にならず MalformedJson で評価全体を止めて
    /// いた）。
    #[test]
    fn req23_out_of_range_number_is_invalid_score_not_malformed_json() {
        let gold = "{\"id\":\"a\",\"label\":\"A\"}\n";
        let pred =
            "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\",\"scores\":{\"A\":1e400}}\n";
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).expect("must succeed");
        assert_eq!(outcome.active.len(), 1);
        assert_eq!(
            outcome.active[0].prediction,
            PredictionOutcome::Error(ErrorOrigin::InvalidScore)
        );
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.code == WarningCode::InvalidScore)
        );
    }

    /// `scores` 以外の無関係なフィールド（`metadata` 等）に `1e400` が
    /// あっても置換対象にならず、`serde_json` の数値範囲エラーとして
    /// 素通りし [`EvalInputStop::MalformedJson`] で停止すること（`scores`
    /// 限定の緩和パース方針。モジュール doc「PoC-9 との差分」参照）。
    #[test]
    fn req23_out_of_range_number_outside_scores_stops_with_malformed_json() {
        let gold = "{\"id\":\"a\",\"label\":\"A\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\",\"metadata\":{\"weight\":1e400}}\n";
        let result = prepare_evaluation_input(gold, pred, &labels(&["A"]));
        assert_eq!(
            result,
            Err(EvalInputStop::MalformedJson {
                side: Side::Prediction,
                line: 1,
            })
        );
    }

    /// `scan_json_number_token` の境界値: 通常の有限な数値（負値・小数・
    /// 指数表記含む）は範囲外判定に巻き込まれず、置換されないこと
    /// （直接ユニットテストで検査。境界値テストは具体値で書く方針）。
    #[test]
    fn req23_scan_json_number_token_handles_finite_values() {
        assert_eq!(scan_json_number_token(b"0}"), Some(1));
        assert_eq!(scan_json_number_token(b"-0.5,"), Some(4));
        assert_eq!(scan_json_number_token(b"1.5e-3]"), Some(6));
        assert_eq!(scan_json_number_token(b"123 "), Some(3));
        // `-` のみは数値トークンとして成立しない。
        assert_eq!(scan_json_number_token(b"-,"), None);
    }

    /// 範囲外の数値（`1e400`）は `scan_json_number_token` でトークン全体を
    /// 読み取れ、`str::parse::<f64>` が非有限値を返すことを直接検査する。
    #[test]
    fn req23_scan_json_number_token_reads_overflowing_exponent() {
        let len = scan_json_number_token(b"1e400}").expect("must scan full token");
        assert_eq!(len, 5);
        let parsed: f64 = std::str::from_utf8(&b"1e400"[..len])
            .unwrap()
            .parse()
            .unwrap();
        assert!(!parsed.is_finite());
    }

    /// 矛盾（ラベル不一致）と重複（ラベル一致）が同時にある入力で、
    /// それぞれ正しい集合（[`WarningCode::ContradictoryInput`]・
    /// [`WarningCode::DuplicateInputWithinSplit`]）に振り分けられること。
    #[test]
    fn req23_contradiction_and_duplicate_are_classified_separately() {
        let gold = concat!(
            "{\"id\":\"a\",\"input\":\"foo\",\"label\":\"A\"}\n", // 矛盾グループ
            "{\"id\":\"b\",\"input\":\"foo\",\"label\":\"B\"}\n",
            "{\"id\":\"c\",\"input\":\"bar\",\"label\":\"A\"}\n", // 重複グループ（ラベル一致）
            "{\"id\":\"d\",\"input\":\"bar\",\"label\":\"A\"}\n",
        );
        let pred = concat!(
            "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
            "{\"id\":\"b\",\"status\":\"ok\",\"predicted_label\":\"B\"}\n",
            "{\"id\":\"c\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
            "{\"id\":\"d\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
        );
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A", "B"])).unwrap();

        let contradiction = outcome
            .warnings
            .iter()
            .find(|w| w.code == WarningCode::ContradictoryInput)
            .expect("ContradictoryInput warning must be present");
        assert_eq!(contradiction.lines, vec![1, 2]);

        let duplicate = outcome
            .warnings
            .iter()
            .find(|w| w.code == WarningCode::DuplicateInputWithinSplit)
            .expect("DuplicateInputWithinSplit warning must be present");
        assert_eq!(duplicate.lines, vec![3, 4]);

        // 矛盾グループ（1・2）は除外、重複グループ（3・4）は残る。
        let gold_lines: std::collections::BTreeSet<usize> =
            outcome.active.iter().map(|row| row.gold_line).collect();
        assert_eq!(
            gold_lines,
            [3, 4]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
        );
    }

    /// gold の全行が単一の矛盾グループに属し、矛盾除外（手順 7）後に
    /// `accepted` が 0 件になる場合、手順 8（pred との突き合わせ）・
    /// 手順 9（データセット単位の警告）へは進まず、手順 7b の
    /// [`EvalInputStop::NoValidGold`] で停止すること（`Ok` で `active: []` を
    /// 返さない。レビュー指摘。PR #204）。この不変条件により、手順 9 の
    /// `if !active.is_empty()` ガード（[`WarningCode::AllAbstain`]・
    /// [`WarningCode::AllError`]・[`WarningCode::UnseenClass`] を `active` が
    /// 空のとき出さない防御）は、`prepare_evaluation_input` 単体では
    /// 到達し得ない防御的コードになる（`active` は手順 6・7b の 2 段の
    /// fail-closed 検査を通過した `accepted`（非空）と 1 対 1 対応するため）。
    #[test]
    fn req23_no_valid_gold_after_contradiction_exclusion() {
        // gold 2 行がともに同じ input で矛盾し、accepted が 0 件になる。
        let gold = concat!(
            "{\"id\":\"a\",\"input\":\"foo\",\"label\":\"A\"}\n",
            "{\"id\":\"b\",\"input\":\"foo\",\"label\":\"B\"}\n",
        );
        let pred = concat!(
            "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
            "{\"id\":\"b\",\"status\":\"ok\",\"predicted_label\":\"B\"}\n",
        );
        let result = prepare_evaluation_input(gold, pred, &labels(&["A", "B", "C"]));
        match result {
            Err(EvalInputStop::NoValidGold {
                gold_rows,
                pred_rows,
                warnings,
            }) => {
                assert_eq!(gold_rows, 2);
                assert_eq!(pred_rows, 2);
                let codes: Vec<WarningCode> = warnings.iter().map(|w| w.code).collect();
                assert_eq!(codes, vec![WarningCode::ContradictoryInput]);
            }
            other => panic!("expected NoValidGold, got {other:?}"),
        }
    }

    /// [`WarningCode::AllError`] は [`ErrorOrigin::MissingPrediction`]・
    /// [`ErrorOrigin::InvalidScore`] が混在していても成立すること
    /// （PoC-9 の `status_counts.error` と同じ数え方。[`PredictionOutcome::Invalid`]
    /// は対象外）。
    #[test]
    fn req23_all_error_holds_across_missing_prediction_and_invalid_score() {
        let gold = concat!(
            "{\"id\":\"a\",\"label\":\"A\"}\n", // pred 行なし -> MissingPrediction
            "{\"id\":\"b\",\"label\":\"A\"}\n", // scores 不正 -> InvalidScore
        );
        let pred =
            "{\"id\":\"b\",\"status\":\"ok\",\"predicted_label\":\"A\",\"scores\":{\"A\":-1.0}}\n";
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();
        assert_eq!(outcome.active.len(), 2);
        assert!(
            outcome
                .active
                .iter()
                .all(|row| matches!(row.prediction, PredictionOutcome::Error(_)))
        );
        let warning = outcome
            .warnings
            .iter()
            .find(|w| w.code == WarningCode::AllError)
            .expect("AllError warning must be present");
        // pred 行を 1 件でも持つ場合は pred 側の行番号だけを使う
        // （"b" の pred 行番号は 1 行目）。gold_line（"a" は 1 行目）への
        // フォールバックとは混在させない（Bugbot 指摘。PR #204
        // threadId PRRT_kwDOUq-SxM6mfP3Y）。
        assert_eq!(warning.lines, vec![1]);
    }

    /// gold 行番号（`MissingPrediction`）と pred 行番号（`InvalidScore`）が
    /// たまたま異なる数値の場合、`AllError.lines` が pred 側の行番号だけを
    /// 正しく列挙し、両者を混在させて誤った行を取りこぼさないこと
    /// （Bugbot 指摘。PR #204 threadId PRRT_kwDOUq-SxM6mfP3Y の直接的な
    /// 回帰テスト。pred 側に十分な行数を用意し、対象行が 2 行目であることを
    /// 明示的に検査する）。
    #[test]
    fn req23_all_error_lines_use_pred_line_space_only_when_any_pred_line_exists() {
        let gold = concat!(
            "{\"id\":\"a\",\"label\":\"A\"}\n", // pred 行なし -> MissingPrediction（gold 1 行目）
            "{\"id\":\"b\",\"label\":\"A\"}\n", // scores 不正 -> InvalidScore（pred 2 行目）
        );
        let pred = concat!(
            "{\"id\":\"z\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n", // "b" とは無関係の pred 1 行目
            "{\"id\":\"b\",\"status\":\"ok\",\"predicted_label\":\"A\",\"scores\":{\"A\":-1.0}}\n",
        );
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();
        let warning = outcome
            .warnings
            .iter()
            .find(|w| w.code == WarningCode::AllError)
            .expect("AllError warning must be present");
        // gold 側の "a"（1 行目）を混在させず、pred 側の "b"（2 行目）だけを
        // 指す。gold_line と pred_line を混在させていれば {1, 2} になるところ
        // {2} だけになることを検査する。
        assert_eq!(warning.lines, vec![2]);
    }

    /// `active` の全行が [`ErrorOrigin::MissingPrediction`] 起因の場合、
    /// pred_line が 1 件も存在しないため、`AllError` の `lines` が空になり
    /// 対象行を特定できなくなる不具合の修正確認（レビュー指摘。PR #204）。
    /// gold 側の行番号にフォールバックし、`lines` が非空で gold の行番号と
    /// 一致することを確認する。
    #[test]
    fn req23_all_error_lines_fall_back_to_gold_line_when_all_missing_prediction() {
        let gold = concat!(
            "{\"id\":\"a\",\"label\":\"A\"}\n", // pred 行なし -> MissingPrediction
            "{\"id\":\"b\",\"label\":\"A\"}\n", // pred 行なし -> MissingPrediction
        );
        let pred = "";
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();
        assert_eq!(outcome.active.len(), 2);
        assert!(outcome.active.iter().all(|row| matches!(
            row.prediction,
            PredictionOutcome::Error(ErrorOrigin::MissingPrediction)
        ) && row.pred_line.is_none()));
        let warning = outcome
            .warnings
            .iter()
            .find(|w| w.code == WarningCode::AllError)
            .expect("AllError warning must be present");
        // gold 側の行番号（1・2 行目）にフォールバックしていること。
        assert_eq!(warning.lines, vec![1, 2]);
    }

    /// [`PredictionOutcome::Invalid`] が 1 件でも混じると [`WarningCode::AllError`]
    /// は成立しないこと。
    #[test]
    fn req23_all_error_does_not_hold_when_invalid_is_mixed_in() {
        let gold = concat!(
            "{\"id\":\"a\",\"label\":\"A\"}\n",
            "{\"id\":\"b\",\"label\":\"A\"}\n",
        );
        let pred = concat!(
            "{\"id\":\"a\",\"status\":\"error\",\"predicted_label\":null}\n",
            // 未知ラベル -> Invalid（Error ではない）。
            "{\"id\":\"b\",\"status\":\"ok\",\"predicted_label\":\"Z\"}\n",
        );
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();
        assert_eq!(outcome.active.len(), 2);
        assert!(
            !outcome
                .warnings
                .iter()
                .any(|w| w.code == WarningCode::AllError)
        );
    }

    /// [`EvalInputWarning::labels`] と [`EvalInputWarning::lines`] の排他:
    /// [`WarningCode::UnseenClass`] のときだけ `labels` が非空で `lines` が空、
    /// それ以外の `code` では `labels` が常に空であること。
    #[test]
    fn req23_labels_and_lines_are_mutually_exclusive() {
        let gold = "{\"id\":\"a\",\"label\":\"A\"}\n";
        let pred = "{\"id\":\"a\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A", "B"])).unwrap();
        assert_eq!(outcome.warnings.len(), 1);
        let warning = &outcome.warnings[0];
        assert_eq!(warning.code, WarningCode::UnseenClass);
        assert_eq!(warning.labels, vec!["B".to_string()]);
        assert!(warning.lines.is_empty());
    }

    /// 診断（矛盾・スコア不正の警告）に生値（`id`・`input`・`label`）が
    /// 含まれないこと（長い id を使った回帰テスト。ケース 7 の input を含む。
    /// PR #191（issue #38）の前例に倣う）。
    #[test]
    fn req23_case7_and_case10_diagnostics_never_contain_raw_values() {
        let long_id_a = "x".repeat(500);
        let long_id_b = "y".repeat(500);
        let long_id_c = "z".repeat(500);
        let secret_input = "散歩の予定を追加したい（この文言が漏れてはならない）";
        // a・b は矛盾（ContradictoryInput で除外）、c は active に残り
        // NaN スコアで InvalidScore になる。両方の警告経路を 1 回で確認する。
        let gold = format!(
            "{{\"id\":\"{long_id_a}\",\"input\":\"{secret_input}\",\"label\":\"A\"}}\n{{\"id\":\"{long_id_b}\",\"input\":\"{secret_input}\",\"label\":\"B\"}}\n{{\"id\":\"{long_id_c}\",\"label\":\"A\"}}\n"
        );
        let pred = format!(
            "{{\"id\":\"{long_id_a}\",\"status\":\"ok\",\"predicted_label\":\"A\"}}\n{{\"id\":\"{long_id_b}\",\"status\":\"ok\",\"predicted_label\":\"B\"}}\n{{\"id\":\"{long_id_c}\",\"status\":\"ok\",\"predicted_label\":\"A\",\"scores\":{{\"A\":NaN}}}}\n"
        );
        let outcome = prepare_evaluation_input(&gold, &pred, &labels(&["A", "B"])).unwrap();
        assert_eq!(outcome.active.len(), 1);
        assert_eq!(
            outcome.active[0].prediction,
            PredictionOutcome::Error(ErrorOrigin::InvalidScore)
        );
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.code == WarningCode::ContradictoryInput)
        );
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.code == WarningCode::InvalidScore)
        );

        // 「生値を診断に含めない」不変条件は EvalInputWarning（診断情報）に
        // 適用される（`ActiveRow.id` は評価結果として `id` を保持する設計の
        // ため対象外。モジュール doc「PoC-9 との差分」参照）。
        let warnings_debug = format!("{:?}", outcome.warnings);
        assert!(!warnings_debug.contains(&long_id_a));
        assert!(!warnings_debug.contains(&long_id_b));
        assert!(!warnings_debug.contains(&long_id_c));
        assert!(!warnings_debug.contains(secret_input));
    }

    /// REQ-23 境界値・TASK-23.2（issue #57）: gold に空白のみの `input` 行が
    /// 複数あると、対応する行番号がすべて [`WarningCode::EmptyInput`]
    /// （`warn_include`・`side: gold`）として報告され、`active` から除外
    /// されない。
    #[test]
    fn req23_empty_input_warns_include_without_excluding() {
        let gold = concat!(
            "{\"id\":\"g1\",\"input\":\"  \",\"label\":\"A\"}\n",
            "{\"id\":\"g2\",\"input\":\"\\u001c\",\"label\":\"A\"}\n",
            "{\"id\":\"g3\",\"input\":\"hello\",\"label\":\"A\"}\n",
        );
        let pred = concat!(
            "{\"id\":\"g1\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
            "{\"id\":\"g2\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
            "{\"id\":\"g3\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
        );
        let outcome = prepare_evaluation_input(gold, pred, &labels(&["A"])).unwrap();

        assert_eq!(outcome.active.len(), 3);
        let warning = outcome
            .warnings
            .iter()
            .find(|w| w.code == WarningCode::EmptyInput)
            .expect("EmptyInput warning must be present");
        assert_eq!(warning.action, WarningAction::WarnInclude);
        assert_eq!(warning.side, Side::Gold);
        assert_eq!(warning.lines, vec![1, 2]);
        assert!(warning.labels.is_empty());

        // EmptyInput は他の WarnInclude 系（DuplicateInputWithinSplit・
        // AllAbstain・AllError）の後、UnseenClass の前に並ぶ（手順 10）。
        let codes: Vec<WarningCode> = outcome.warnings.iter().map(|w| w.code).collect();
        assert_eq!(codes, vec![WarningCode::EmptyInput]);
    }

    /// 空白のみの `input` を持つ行が 2 行あり異なるラベルの場合、
    /// [`WarningCode::ContradictoryInput`] で `active` から除外されつつ、
    /// [`WarningCode::EmptyInput`] にも両行が報告される（矛盾除外より前に
    /// 検出するため報告が漏れない。REQ-23 境界値・TASK-23.2）。
    /// 2 行とも矛盾除外され `active` が 0 件になり
    /// [`EvalInputStop::NoValidGold`] で停止しても、`warnings` に
    /// ContradictoryInput・EmptyInput の両方が含まれること（codex レビュー
    /// 指摘・PR #212。修正前は EXCLUDE_ORDER に EmptyInput が無く、
    /// `drain_warnings_in_order` の対象外として `lines_by_code` に残ったまま
    /// 破棄され、報告が漏れていた）。
    #[test]
    fn req23_empty_input_reported_even_when_contradictory_excludes_rows() {
        let gold = concat!(
            "{\"id\":\"g1\",\"input\":\" \",\"label\":\"A\"}\n",
            "{\"id\":\"g2\",\"input\":\" \",\"label\":\"B\"}\n",
        );
        let pred = concat!(
            "{\"id\":\"g1\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n",
            "{\"id\":\"g2\",\"status\":\"ok\",\"predicted_label\":\"B\"}\n",
        );
        let result = prepare_evaluation_input(gold, pred, &labels(&["A", "B"]));
        match result {
            Err(EvalInputStop::NoValidGold { warnings, .. }) => {
                assert!(
                    warnings
                        .iter()
                        .any(|w| w.code == WarningCode::ContradictoryInput
                            && w.lines == vec![1, 2])
                );
                assert!(
                    warnings
                        .iter()
                        .any(|w| w.code == WarningCode::EmptyInput && w.lines == vec![1, 2])
                );
            }
            other => panic!("expected NoValidGold, got {other:?}"),
        }
    }

    /// 警告に `input` の値そのものは含まれない（診断情報に生値を含めない
    /// 方針。モジュール doc「PoC-9 との差分」）。
    #[test]
    fn req23_empty_input_warning_does_not_leak_raw_input_value() {
        let secret_marker = "\u{3000}\u{3000}";
        let gold = format!("{{\"id\":\"g1\",\"input\":\"{secret_marker}\",\"label\":\"A\"}}\n");
        let pred = "{\"id\":\"g1\",\"status\":\"ok\",\"predicted_label\":\"A\"}\n";
        let outcome = prepare_evaluation_input(&gold, pred, &labels(&["A"])).unwrap();
        let warnings_debug = format!("{:?}", outcome.warnings);
        assert!(!warnings_debug.contains(secret_marker));
        assert!(
            outcome
                .warnings
                .iter()
                .any(|w| w.code == WarningCode::EmptyInput)
        );
    }
}
