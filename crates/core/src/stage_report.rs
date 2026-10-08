//! 正常系（exit 0）の工程結果 JSON の型と直列化（REQ-21・REQ-33・TASK-33.2-2・#139）。
//!
//! # 呼び出し文脈
//!
//! CLI の各工程（`register → … → infer`）が exit 0 で終わるとき、stdout へ出す
//! 「工程ごとの結果フィールドを持つ JSON オブジェクト」の型をここに置く。異常系
//! （exit ≠ 0）の `{"code","message"}` は [`crate::exitcode::ErrorReport`] が担い、
//! 本モジュールはそれを変更しない。cli crate は `serde_json` に依存しないため、
//! 直列化を core 側に閉じる（`ErrorReport::to_json_line` と同じ理由。
//! `.claude/rules/dependency-policy.md`）。I/O は行わない。
//!
//! # 現状（実装済みの範囲）
//!
//! `package` 工程の [`PackageReport`] のほか、TASK-33.1-2（#136）で `register`・`inspect`・
//! `train`・`select` の完了結果（[`RegisterReport`]・[`InspectStageReport`]・[`TrainReport`]・
//! [`SelectReport`]。件数・固定語彙のみでパス・本文を含まない）を追加した。
//!
//! `package` 工程の [`PackageReport`] のフィールドは PoC-16 の package 工程の
//! 出力名（`step`・`status`・`judgment`・`acceptance_defined`）に、容量内訳と p95 の計測値
//! （`capacity`・`infer_p95`。#340・REQ-30・REQ-31）を末尾へ足した集合で、載せるのは整数・
//! bool・固定キーだけ（パス・データ本文は載せない。security.md）。加えて `evaluate` 工程の評価データ
//! 未定義時の [`EvaluateReport`]（`status:"skipped"`。TASK-33.3・#140）と、評価データありで
//! 評価が完了したときの [`EvaluateCompletedReport`]（正解率・Macro-F1。#314）を持つ。
//! 後者の JSON スキーマは 2026-09-30 オーナー承認済み。Wilson 区間・McNemar / Holm・診断（REQ-29）は
//! 出力に含めない（未結線）。
//!
//! [`PackageReport`]（exit 0）は `pass` と基準未定義のみを表す。`fail`（exit 10）・判定不能
//! （exit 12）は合否基準が定義されているときにだけ生じ、判定項目つきの
//! [`PackageJudgedReport`] で返す（#328・REQ-21・REQ-33。exit 0 の型に `fail` を載せられない
//! ことを型で保証する）。`limit_exceeded`（exit 20）は [`PackageLimitExceededReport`]
//! （`{"code","message","step","capacity","infer_p95"}`。どの上限を超えたかは各 `exceeded` で区別。#340）で返す。
//! 計測値の値型（[`PackageCapacity`]・[`InferP95`]）は runtime に依存しない整数・bool の型で、
//! 境界規則（`>`）は runtime の `LimitBreach` が唯一の実装であり、ここでは比較しない。

use serde::Serialize;

use crate::evaluation_record::BaselineComparisonVerdict;

/// CLI の 7 工程（REQ-33。工程順）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// 定義ファイルの登録。
    Register,
    /// データ検査。
    Inspect,
    /// 学習。
    Train,
    /// 評価。
    Evaluate,
    /// 選定。
    Select,
    /// 配布パッケージ化。
    Package,
    /// 推論。
    Infer,
}

/// 工程の状態（`ok`・`skipped`）。`skipped` は評価データ未定義の `evaluate` のみ（REQ-17・TASK-33.3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    /// 工程が完了した。
    Ok,
    /// 工程を実行せず終えた（exit 0。評価済みを装わない）。
    Skipped,
}

/// `evaluate` を skipped で終えた理由（機械可読な英語の固定語彙）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluateSkipReason {
    /// 評価データが定義されていない（REQ-17）。
    EvaluationDataNotDefined,
}

/// `evaluate` 工程が評価データ未定義で返す JSON（REQ-17・REQ-33・TASK-33.3・#140）。
///
/// 指標・合否のフィールドを持たず、コンストラクタは [`Self::skipped`] のみのため
/// 「評価済みを装う」値を作れない。フィールドは宣言順（`step`・`status`・`reason`）に直列化する。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvaluateReport {
    step: Stage,
    status: StageStatus,
    reason: EvaluateSkipReason,
}

impl EvaluateReport {
    /// 評価データ未定義の skipped 結果（exit 0。PoC-16 縦断 2 の `status:"skipped"`）。
    #[must_use]
    pub const fn skipped() -> Self {
        Self {
            step: Stage::Evaluate,
            status: StageStatus::Skipped,
            reason: EvaluateSkipReason::EvaluationDataNotDefined,
        }
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。[`PackageReport::to_json_line`] と対称。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// `evaluate` 工程が評価データありで完了したときの JSON（REQ-24・REQ-27・REQ-33・#314）。
///
/// 凍結した評価データへの 1 回限りの適用が終わり、指標を算出できたときだけ作れる。
/// フィールドは非公開でコンストラクタ [`Self::completed`] のみが作る（`total == 0`・
/// `correct > total`・範囲外の `macro_f1` は `None`。壊れた値を表現できない型にする）。
/// パス・データ本文・ラベルは載せない（security.md）。宣言順（`step`・`status`・`candidate`・
/// `kind`・`n_total`・`correct`・`accuracy`・`macro_f1`）に直列化し、`macro_f1` が未定義なら
/// `null`（`skip_serializing_if` を付けずスキーマを固定する。分母 0 の指標は `null`。REQ-24）。
///
/// この JSON スキーマは 2026-09-30 にオーナー承認済み（入出力契約への加算的な追加）。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvaluateCompletedReport {
    step: Stage,
    status: StageStatus,
    candidate: usize,
    kind: String,
    n_total: u64,
    correct: u64,
    accuracy: f64,
    macro_f1: Option<f64>,
}

impl EvaluateCompletedReport {
    /// 評価完了の結果を作る。`accuracy` は `correct / total` から求める。
    ///
    /// `total == 0`、`correct > total`、有限でない・`[0, 1]` の外の `macro_f1` は `None`。
    #[must_use]
    pub fn completed(
        candidate: usize,
        kind: String,
        correct: u64,
        total: u64,
        macro_f1: Option<f64>,
    ) -> Option<Self> {
        if total == 0 || correct > total {
            return None;
        }
        if macro_f1.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
            return None;
        }
        Some(Self {
            step: Stage::Evaluate,
            status: StageStatus::Ok,
            candidate,
            kind,
            n_total: total,
            correct,
            accuracy: correct as f64 / total as f64,
            macro_f1,
        })
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// 容量内訳の 1 構成要素（`bytes`・`file_count`。REQ-30・#340）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PackageComponentSize {
    bytes: u64,
    file_count: u32,
}

impl PackageComponentSize {
    /// 値をそのまま保持する。
    #[must_use]
    pub const fn new(bytes: u64, file_count: u32) -> Self {
        Self { bytes, file_count }
    }
}

/// 容量内訳の 5 構成要素（REQ-30）。宣言順に直列化し、5 項目を常に出す（#123 の
/// `package_capacity_json` と同じ並び）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PackageCapacityComponents {
    weights: PackageComponentSize,
    vocab_or_feature_transform: PackageComponentSize,
    label_table: PackageComponentSize,
    calibration: PackageComponentSize,
    metadata: PackageComponentSize,
}

impl PackageCapacityComponents {
    /// 5 構成要素を宣言順に受け取る。
    #[must_use]
    pub const fn new(
        weights: PackageComponentSize,
        vocab_or_feature_transform: PackageComponentSize,
        label_table: PackageComponentSize,
        calibration: PackageComponentSize,
        metadata: PackageComponentSize,
    ) -> Self {
        Self {
            weights,
            vocab_or_feature_transform,
            label_table,
            calibration,
            metadata,
        }
    }
}

/// `package` の容量の計測値と上限照合の結果（REQ-30・#340）。
///
/// `exceeded` は呼び出し側（cli）が runtime の照合結果から渡す。ここでは `total_bytes > limit_bytes`
/// を計算しない（境界規則は runtime の `LimitBreach` が唯一の実装）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PackageCapacity {
    total_bytes: u64,
    limit_bytes: u64,
    exceeded: bool,
    components: PackageCapacityComponents,
}

impl PackageCapacity {
    /// 値をそのまま保持する。
    #[must_use]
    pub const fn new(
        total_bytes: u64,
        limit_bytes: u64,
        exceeded: bool,
        components: PackageCapacityComponents,
    ) -> Self {
        Self {
            total_bytes,
            limit_bytes,
            exceeded,
            components,
        }
    }

    /// 容量の上限を超えたか（runtime の照合結果）。
    #[must_use]
    pub const fn exceeded(&self) -> bool {
        self.exceeded
    }
}

/// 推論待ち時間 p95 の計測値と上限照合の結果（µs。REQ-31・#340）。
///
/// `p95_us` は ns の p95 を切り上げた値。`exceeded` は runtime の照合結果をそのまま受け取る。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct InferP95 {
    p95_us: u64,
    limit_us: u64,
    exceeded: bool,
}

impl InferP95 {
    /// 値をそのまま保持する。
    #[must_use]
    pub const fn new(p95_us: u64, limit_us: u64, exceeded: bool) -> Self {
        Self {
            p95_us,
            limit_us,
            exceeded,
        }
    }

    /// p95 の上限を超えたか（runtime の照合結果）。
    #[must_use]
    pub const fn exceeded(&self) -> bool {
        self.exceeded
    }
}

/// `package` の計測値の組（容量は常に、p95 は `limits.max_infer_p95_us` があるときだけ。#340）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackageMetrics {
    /// 容量内訳と上限照合。
    pub capacity: PackageCapacity,
    /// p95 と上限照合（上限未設定なら `None`。値を出すためだけに計測はしない）。
    pub infer_p95: Option<InferP95>,
}

/// `package` 工程が上限超過（exit 20）で返す JSON（REQ-21・REQ-30・REQ-31・#340）。
///
/// `{"code":"limit_exceeded","message","step":"package","capacity","infer_p95"}`。少なくとも一方の
/// `exceeded` が true のときにしか作れない（超過が無いのに exit 20 を出せない）。どちらの上限を
/// 超えたかは各キーの `exceeded` で区別する。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PackageLimitExceededReport {
    code: crate::exitcode::ExitCode,
    message: String,
    step: Stage,
    capacity: PackageCapacity,
    infer_p95: Option<InferP95>,
}

impl PackageLimitExceededReport {
    /// どちらの `exceeded` も false なら `None`（fail-closed）。
    #[must_use]
    pub fn new(message: String, metrics: PackageMetrics) -> Option<Self> {
        let any = metrics.capacity.exceeded() || metrics.infer_p95.is_some_and(|p| p.exceeded());
        any.then_some(Self {
            code: crate::exitcode::ExitCode::LimitExceeded,
            message,
            step: Stage::Package,
            capacity: metrics.capacity,
            infer_p95: metrics.infer_p95,
        })
    }

    /// 終了コード（常に `limit_exceeded`=20）。
    #[must_use]
    pub const fn exit_code(&self) -> crate::exitcode::ExitCode {
        self.code
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// 合否判定。`Pass` は exit 0（[`PackageReport`]）、`Fail`・`Undeterminable` は
/// exit 10・12（[`PackageJudgedReport`]）でのみ使う（#328）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageJudgment {
    /// 合否基準を満たした。
    Pass,
    /// 合否基準を満たさないと有意に言える（exit 10）。
    Fail,
    /// 件数不足などで判定できない。合格扱いにしない（exit 12。REQ-24）。
    Undeterminable,
}

/// `package` 工程が合否判定の結果として exit 10・12 で返す JSON（#328・REQ-21・REQ-33）。
///
/// `{"code","message","step":"package","judgment","acceptance_defined":true}` の形で、
/// `ErrorReport` の `{"code","message"}` に判定項目を足したもの。`Fail`・`Undeterminable`
/// は合否基準が定義されているときにだけ生じるため、`acceptance_defined` は常に `true`（#344）。
/// コンストラクタは [`Self::fail`]・[`Self::undeterminable`] のみで、`code` と `judgment` の
/// 矛盾した組み合わせを作れない。宣言順に直列化し、パス・件数・本文は載せない（security.md）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PackageJudgedReport {
    code: crate::exitcode::ExitCode,
    message: String,
    step: Stage,
    judgment: PackageJudgment,
    acceptance_defined: bool,
    capacity: PackageCapacity,
    /// 上限未設定のときは `null`（スキーマを固定する。#340）。
    infer_p95: Option<InferP95>,
}

impl PackageJudgedReport {
    /// 合否基準を満たさない結果（exit 10・`judged_fail`）。
    #[must_use]
    pub fn fail(message: String, metrics: PackageMetrics) -> Self {
        Self {
            code: crate::exitcode::ExitCode::JudgedFail,
            message,
            step: Stage::Package,
            judgment: PackageJudgment::Fail,
            acceptance_defined: true,
            capacity: metrics.capacity,
            infer_p95: metrics.infer_p95,
        }
    }

    /// 判定不能の結果（exit 12・`pending`。合格扱いにしない）。
    #[must_use]
    pub fn undeterminable(message: String, metrics: PackageMetrics) -> Self {
        Self {
            code: crate::exitcode::ExitCode::Pending,
            message,
            step: Stage::Package,
            judgment: PackageJudgment::Undeterminable,
            acceptance_defined: true,
            capacity: metrics.capacity,
            infer_p95: metrics.infer_p95,
        }
    }

    /// この結果に対応する終了コード（10 または 12）。
    #[must_use]
    pub const fn exit_code(&self) -> crate::exitcode::ExitCode {
        self.code
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// `package` 工程が exit 0 で返す JSON（フィールドは宣言順に直列化する）。
///
/// フィールドは非公開で、コンストラクタ経由でのみ作る。`step`・`status` を固定し、
/// `judgment` と `acceptance_defined` の矛盾した組み合わせを作れないようにする。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PackageReport {
    step: Stage,
    status: StageStatus,
    /// 合否基準が未設定のときは `null`（`skip_serializing_if` を付けずスキーマを固定する）。
    judgment: Option<PackageJudgment>,
    acceptance_defined: bool,
    capacity: PackageCapacity,
    /// 上限未設定のときは `null`（スキーマを固定する。#340）。
    infer_p95: Option<InferP95>,
}

impl PackageReport {
    /// 合否基準を満たした結果（PoC-16 実測の `judgment:"pass"`）。
    #[must_use]
    pub fn pass(metrics: PackageMetrics) -> Self {
        Self {
            step: Stage::Package,
            status: StageStatus::Ok,
            judgment: Some(PackageJudgment::Pass),
            acceptance_defined: true,
            capacity: metrics.capacity,
            infer_p95: metrics.infer_p95,
        }
    }

    /// 合否基準が未設定の結果（`judgment` は `null`。exit 0）。
    #[must_use]
    pub fn acceptance_not_defined(metrics: PackageMetrics) -> Self {
        Self {
            step: Stage::Package,
            status: StageStatus::Ok,
            judgment: None,
            acceptance_defined: false,
            capacity: metrics.capacity,
            infer_p95: metrics.infer_p95,
        }
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// [`crate::exitcode::ErrorReport::to_json_line`] と対称の API。改行は
    /// 呼び出し側（cli の出力関数）が付ける。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// `register` 工程が exit 0 で返す JSON（REQ-15・REQ-17・REQ-33・TASK-33.1-2・#136）。
///
/// パス・データ本文は載せない（security.md）。`definition_sha256` は定義の正準化ハッシュ
/// （[`crate::definition::Definition::canonical_hash`]）、`options` は選択肢数、
/// `evaluation_defined` は独立した評価データを取り込んだか（REQ-17）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RegisterReport {
    step: Stage,
    status: StageStatus,
    definition_sha256: String,
    options: usize,
    evaluation_defined: bool,
}

impl RegisterReport {
    /// `register` の完了結果を作る。
    #[must_use]
    pub fn new(definition_sha256: String, options: usize, evaluation_defined: bool) -> Self {
        Self {
            step: Stage::Register,
            status: StageStatus::Ok,
            definition_sha256,
            options,
            evaluation_defined,
        }
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// `inspect` の分割ごとの件数（REQ-17）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SplitCounts {
    /// train 分割の件数。
    pub train: usize,
    /// validation 分割の件数。
    pub validation: usize,
    /// test 分割の件数（記録のみ。最終 test は 1 回限りの適用まで使わない。REQ-27）。
    pub test: usize,
}

/// `inspect` 工程が exit 0 で返す JSON（REQ-16・REQ-17・REQ-33・TASK-33.1-2・#136）。
///
/// 異常・漏洩が 1 件でもあれば exit 0 にせず `invalid_input` を返すため、本型は
/// 「検査を通過した」結果のみを表す。件数のみを載せ、行番号・本文は載せない。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InspectStageReport {
    step: Stage,
    status: StageStatus,
    valid_records: usize,
    split: SplitCounts,
}

impl InspectStageReport {
    /// `inspect` の完了結果を作る。
    #[must_use]
    pub fn new(valid_records: usize, split: SplitCounts) -> Self {
        Self {
            step: Stage::Inspect,
            status: StageStatus::Ok,
            valid_records,
            split,
        }
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// `train` 工程が exit 0 で返す JSON（REQ-18・REQ-33・TASK-33.1-2・#136）。
///
/// `candidate` は候補の添字（`--candidate`）、`kind` は候補の種類 ID（固定語彙 `c1` 等）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrainReport {
    step: Stage,
    status: StageStatus,
    candidate: usize,
    kind: String,
}

impl TrainReport {
    /// `train` の完了結果を作る。
    #[must_use]
    pub fn new(candidate: usize, kind: String) -> Self {
        Self {
            step: Stage::Train,
            status: StageStatus::Ok,
            candidate,
            kind,
        }
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// `select` 工程が exit 0 で返す JSON（REQ-18・REQ-33・TASK-33.1-2・#136）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SelectReport {
    step: Stage,
    status: StageStatus,
    candidate: usize,
    kind: String,
}

impl SelectReport {
    /// `select` の完了結果を作る（選ばれた候補の添字と種類）。
    #[must_use]
    pub fn new(candidate: usize, kind: String) -> Self {
        Self {
            step: Stage::Select,
            status: StageStatus::Ok,
            candidate,
            kind,
        }
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// 採点入口の `step` 値（固定。7 工程の [`Stage`] には含めない。7 工程の契約外。REQ-41・#445）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ScoreStep {
    ScorePredictions,
}

/// 採点入口での予測ファイルの役割（REQ-41・#445）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreRole {
    /// 採点対象。
    Candidate,
    /// Holm の族に入れる比較相手。
    Compare,
    /// 族に入れず McNemar の生の値だけを出す相手。
    Reference,
}

/// ラベル別指標 1 行（分母 0 の指標は `null`。REQ-24）。`label` は定義の選択肢 ID でデータ本文ではない。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScorePerLabel {
    /// 選択肢 ID。
    pub label: String,
    /// 正解ラベルがこのラベルの件数。
    pub support: u64,
    /// このラベルと予測した件数。
    pub predicted_count: u64,
    /// 真陽性。
    pub tp: u64,
    /// 偽陽性。
    pub fp: u64,
    /// 偽陰性。
    #[serde(rename = "fn")]
    pub fn_: u64,
    /// 適合率。
    pub precision: Option<f64>,
    /// 再現率。
    pub recall: Option<f64>,
    /// F1。
    pub f1: Option<f64>,
}

/// 混同行列（行＝正解ラベル、列＝選択肢 ID ＋ `invalid`・`abstain`・`error`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScoreConfusionMatrix {
    labels: Vec<String>,
    columns: Vec<String>,
    rows: Vec<Vec<u64>>,
}

impl ScoreConfusionMatrix {
    /// 選択肢 ID 列と行から作る。`columns` は選択肢 ID に `invalid`・`abstain`・`error` を足して導く。
    /// 各行の長さが `labels.len() + 3`、行数が `labels.len()` でなければ `None`。
    #[must_use]
    pub fn new(labels: Vec<String>, rows: Vec<Vec<u64>>) -> Option<Self> {
        let width = labels.len() + 3;
        if rows.len() != labels.len() || rows.iter().any(|r| r.len() != width) {
            return None;
        }
        let mut columns = labels.clone();
        columns.extend(["invalid", "abstain", "error"].map(String::from));
        Some(Self {
            labels,
            columns,
            rows,
        })
    }
}

/// 対 majority の McNemar 結果（`b`＝候補のみ正解、`c`＝majority のみ正解）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ScoreVsMajority {
    /// 候補だけが正解した件数。
    pub b: u64,
    /// 下限基準だけが正解した件数。
    pub c: u64,
    /// 両側 p 値。
    pub p: f64,
    /// 判定。
    pub verdict: BaselineComparisonVerdict,
}

/// 予測ファイル 1 つ分の採点結果。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoreCandidate {
    /// 候補名（P・C1・C3・AR）。
    pub name: String,
    /// 役割。
    pub role: ScoreRole,
    /// 予測ファイルの sha256（16 進）。
    pub pred_sha256: String,
    /// 正解数。
    pub correct: u64,
    /// 正解率。
    pub accuracy: f64,
    /// 正解率の Wilson 95% 区間 `[下限, 上限]`。
    pub accuracy_wilson95: [f64; 2],
    /// Macro-F1（未定義なら `null`）。
    pub macro_f1: Option<f64>,
    /// ラベル別指標。
    pub per_label: Vec<ScorePerLabel>,
    /// 混同行列。
    pub confusion_matrix: ScoreConfusionMatrix,
    /// 対 majority。
    pub vs_majority: ScoreVsMajority,
}

/// Holm 補正の比較 1 行。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoreHolmComparison {
    /// 比較相手（`majority` または NAME）。
    pub against: String,
    /// 候補だけが正解した件数。
    pub b: u64,
    /// 相手だけが正解した件数。
    pub c: u64,
    /// 補正前の p 値。
    pub p_raw: f64,
    /// Holm 補正後の p 値。
    pub p_adjusted: f64,
    /// 判定。
    pub verdict: BaselineComparisonVerdict,
}

/// Holm 補正の結果（`m` は族の大きさ）。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoreHolm {
    /// 採点対象の NAME。
    pub candidate: String,
    /// 族の大きさ。
    pub m: usize,
    /// 比較。
    pub comparisons: Vec<ScoreHolmComparison>,
}

/// 族に入れない参照相手との McNemar の生の値。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoreReference {
    /// 採点対象の NAME。
    pub candidate: String,
    /// 参照相手の NAME。
    pub against: String,
    /// 候補だけが正解した件数。
    pub b: u64,
    /// 参照相手だけが正解した件数。
    pub c: u64,
    /// 補正前の p 値。
    pub p_raw: f64,
}

/// PoC-26 の採点入口 `fandhe-edge-score` が exit 0 で返す JSON（REQ-41・REQ-27・#445）。
///
/// 7 工程の入出力契約の外（`step:"score_predictions"`）。cli は `serde_json` に依存しないため
/// 直列化を core に閉じる。ラベル ID は定義の選択肢 ID でありデータ本文ではない
/// （`majority_label` と同格。オーナー判断 2026-10-08）。フィールドは宣言順に直列化し、
/// 分母 0 の指標は `null`、非有限の浮動小数も `null`。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScoreReport {
    step: ScoreStep,
    status: StageStatus,
    seed: u32,
    evaluation_sha256: String,
    n_total: u64,
    required_sample_size: u64,
    candidates: Vec<ScoreCandidate>,
    holm: ScoreHolm,
    references: Vec<ScoreReference>,
}

impl ScoreReport {
    /// 採点結果を組み立てる（`status` は常に `ok`）。
    #[must_use]
    pub fn new(
        seed: u32,
        evaluation_sha256: String,
        n_total: u64,
        required_sample_size: u64,
        candidates: Vec<ScoreCandidate>,
        holm: ScoreHolm,
        references: Vec<ScoreReference>,
    ) -> Self {
        Self {
            step: ScoreStep::ScorePredictions,
            status: StageStatus::Ok,
            seed,
            evaluation_sha256,
            n_total,
            required_sample_size,
            candidates,
            holm,
            references,
        }
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// 予測 1 件の結果（[`PredictionLine`] の入力。評価器の `Outcome` と同じ 4 分類。cli が写す）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PredictionLineOutcome {
    /// 選択肢 ID を返した。
    Label(String),
    /// 型不正（`ok` かつ `predicted_label:null`）。
    Invalid,
    /// 保留。
    Abstain,
    /// 実行エラー。
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum PredictionLineStatus {
    Ok,
    Abstain,
    Error,
}

/// 選択肢 ID 順を保つスコアの JSON オブジェクト。
#[derive(Debug, Clone, PartialEq)]
struct PredictionScores(Vec<(String, f64)>);

impl Serialize for PredictionScores {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

/// `evaluate` が `evaluation_predictions.jsonl` へ書く 1 行（REQ-27・REQ-41・#445）。
///
/// 行形式は `{"id","status","predicted_label","scores"?}`。データ契約層の読み手が評価器の
/// 4 分類へ戻せる形に限る。`id` は評価データのレコード ID、`predicted_label` は定義の選択肢 ID
/// （データ本文ではない）。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PredictionLine {
    id: String,
    status: PredictionLineStatus,
    predicted_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scores: Option<PredictionScores>,
}

impl PredictionLine {
    /// 1 行を作る。`scores` は（選択肢 ID 列, スコア列）で、長さが違う・非有限の値を含むときは
    /// 出さない（読み手が不正なスコアを不正解に数えるため、壊れた値を書かない）。
    #[must_use]
    pub fn new(
        id: &str,
        outcome: PredictionLineOutcome,
        scores: Option<(&[&str], &[f64])>,
    ) -> Self {
        let (status, predicted_label) = match outcome {
            PredictionLineOutcome::Label(l) => (PredictionLineStatus::Ok, Some(l)),
            PredictionLineOutcome::Invalid => (PredictionLineStatus::Ok, None),
            PredictionLineOutcome::Abstain => (PredictionLineStatus::Abstain, None),
            PredictionLineOutcome::Error => (PredictionLineStatus::Error, None),
        };
        let scores = scores
            .filter(|(ids, values)| {
                ids.len() == values.len() && values.iter().all(|v| v.is_finite())
            })
            .map(|(ids, values)| {
                PredictionScores(
                    ids.iter()
                        .map(|k| (*k).to_string())
                        .zip(values.iter().copied())
                        .collect(),
                )
            });
        Self {
            id: id.to_string(),
            status,
            predicted_label,
            scores,
        }
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 5 構成要素が宣言順（重み・語彙/特徴量変換・選択肢表・校正・メタデータ）で常に出る（REQ-30）。
    const COMPONENTS: &str = r#""components":{"weights":{"bytes":100,"file_count":1},"vocab_or_feature_transform":{"bytes":0,"file_count":0},"label_table":{"bytes":20,"file_count":1},"calibration":{"bytes":0,"file_count":0},"metadata":{"bytes":5,"file_count":1}}"#;
    const CAP_OK: &str = r#""capacity":{"total_bytes":125,"limit_bytes":40000000,"exceeded":false,"components":{"weights":{"bytes":100,"file_count":1},"vocab_or_feature_transform":{"bytes":0,"file_count":0},"label_table":{"bytes":20,"file_count":1},"calibration":{"bytes":0,"file_count":0},"metadata":{"bytes":5,"file_count":1}}}"#;
    const CAP_EXCEEDED: &str = r#"{"total_bytes":125,"limit_bytes":100,"exceeded":true,"components":{"weights":{"bytes":100,"file_count":1},"vocab_or_feature_transform":{"bytes":0,"file_count":0},"label_table":{"bytes":20,"file_count":1},"calibration":{"bytes":0,"file_count":0},"metadata":{"bytes":5,"file_count":1}}}"#;

    /// 合成の計測値。`capacity_exceeded` が true のときは上限 100・false のときは 40,000,000。
    fn metrics(capacity_exceeded: bool, p95: Option<(u64, u64, bool)>) -> PackageMetrics {
        let c = PackageComponentSize::new;
        let components =
            PackageCapacityComponents::new(c(100, 1), c(0, 0), c(20, 1), c(0, 0), c(5, 1));
        let limit = if capacity_exceeded { 100 } else { 40_000_000 };
        PackageMetrics {
            capacity: PackageCapacity::new(125, limit, capacity_exceeded, components),
            infer_p95: p95.map(|(p, l, e)| InferP95::new(p, l, e)),
        }
    }

    /// REQ-30・#340: components の JSON 断片は 5 項目をこの順で常に出す。
    #[test]
    fn req30_issue340_components_always_five_in_order() {
        assert!(CAP_OK.contains(COMPONENTS));
    }

    /// REQ-33: Pass の JSON が PoC-16 の名前・値と完全一致する。
    #[test]
    fn req33_pass_report_json_is_exact() {
        assert_eq!(
            PackageReport::pass(metrics(false, None))
                .to_json_line()
                .expect("json"),
            format!(
                r#"{{"step":"package","status":"ok","judgment":"pass","acceptance_defined":true,{CAP_OK},"infer_p95":null}}"#
            )
        );
    }

    /// REQ-33: 合否基準未設定は `judgment:null`・`acceptance_defined:false`。
    #[test]
    fn req33_not_defined_report_json_is_exact() {
        assert_eq!(
            PackageReport::acceptance_not_defined(metrics(false, Some((5000, 6000, false))))
                .to_json_line()
                .expect("json"),
            format!(
                r#"{{"step":"package","status":"ok","judgment":null,"acceptance_defined":false,{CAP_OK},"infer_p95":{{"p95_us":5000,"limit_us":6000,"exceeded":false}}}}"#
            )
        );
    }

    /// REQ-21・REQ-33・#328: exit 10・12 の判定項目つき JSON と終了コード。
    #[test]
    fn req33_issue328_judged_reports_json_and_exit_code_are_exact() {
        use crate::exitcode::ExitCode;
        let fail = PackageJudgedReport::fail("judged as fail".to_string(), metrics(false, None));
        assert_eq!(fail.exit_code(), ExitCode::JudgedFail);
        assert_eq!(
            fail.to_json_line().expect("json"),
            format!(
                r#"{{"code":"judged_fail","message":"judged as fail","step":"package","judgment":"fail","acceptance_defined":true,{CAP_OK},"infer_p95":null}}"#
            )
        );
        let pending = PackageJudgedReport::undeterminable(
            "result is pending".to_string(),
            metrics(false, Some((1, 2, false))),
        );
        assert_eq!(pending.exit_code(), ExitCode::Pending);
        assert_eq!(
            pending.to_json_line().expect("json"),
            format!(
                r#"{{"code":"pending","message":"result is pending","step":"package","judgment":"undeterminable","acceptance_defined":true,{CAP_OK},"infer_p95":{{"p95_us":1,"limit_us":2,"exceeded":false}}}}"#
            )
        );
    }

    /// REQ-30・REQ-31・REQ-21・#340: exit 20 の JSON は超過の種類を各 `exceeded` で区別でき、
    /// どちらも超過していなければ構築できない。
    #[test]
    fn req30_req31_issue340_limit_exceeded_report_json_and_constructor() {
        use crate::exitcode::ExitCode;
        let both = PackageLimitExceededReport::new(
            "resource limit exceeded".to_string(),
            metrics(true, Some((7, 6, true))),
        )
        .expect("exceeded");
        assert_eq!(both.exit_code(), ExitCode::LimitExceeded);
        assert_eq!(
            both.to_json_line().expect("json"),
            format!(
                r#"{{"code":"limit_exceeded","message":"resource limit exceeded","step":"package","capacity":{CAP_EXCEEDED},"infer_p95":{{"p95_us":7,"limit_us":6,"exceeded":true}}}}"#
            )
        );
        let only_p95 =
            PackageLimitExceededReport::new("m".to_string(), metrics(false, Some((7, 6, true))))
                .expect("p95 exceeded");
        assert!(
            only_p95
                .to_json_line()
                .expect("json")
                .contains(r#""infer_p95":{"p95_us":7,"limit_us":6,"exceeded":true}"#)
        );
        assert!(PackageLimitExceededReport::new("m".to_string(), metrics(false, None)).is_none());
        assert!(
            PackageLimitExceededReport::new("m".to_string(), metrics(false, Some((1, 2, false))))
                .is_none()
        );
    }

    /// REQ-33: 7 工程が snake_case の名前で直列化される。
    #[test]
    fn req33_stage_names_are_snake_case() {
        let all = [
            (Stage::Register, "register"),
            (Stage::Inspect, "inspect"),
            (Stage::Train, "train"),
            (Stage::Evaluate, "evaluate"),
            (Stage::Select, "select"),
            (Stage::Package, "package"),
            (Stage::Infer, "infer"),
        ];
        for (stage, name) in all {
            assert_eq!(
                serde_json::to_string(&stage).expect("json"),
                format!("\"{name}\"")
            );
        }
    }

    /// REQ-33・REQ-17: evaluate の skipped JSON が完全一致する。
    #[test]
    fn req33_evaluate_skipped_report_json_is_exact() {
        assert_eq!(
            EvaluateReport::skipped().to_json_line().expect("json"),
            r#"{"step":"evaluate","status":"skipped","reason":"evaluation_data_not_defined"}"#
        );
    }

    /// REQ-33: skipped の JSON は 1 行。
    #[test]
    fn req33_evaluate_skipped_report_is_single_line() {
        assert!(
            !EvaluateReport::skipped()
                .to_json_line()
                .expect("json")
                .contains('\n')
        );
    }

    /// REQ-33・REQ-24: 評価完了の JSON が完全一致する（キーは宣言順。`macro_f1` は数値）。
    #[test]
    fn req33_evaluate_completed_report_json_is_exact() {
        let report = EvaluateCompletedReport::completed(1, "c3".to_string(), 3, 4, Some(0.5))
            .expect("report");
        assert_eq!(
            report.to_json_line().expect("json"),
            r#"{"step":"evaluate","status":"ok","candidate":1,"kind":"c3","n_total":4,"correct":3,"accuracy":0.75,"macro_f1":0.5}"#
        );
    }

    /// REQ-24: `macro_f1` が未定義なら `null`（0 や 1 で埋めない）。
    #[test]
    fn req24_evaluate_completed_report_macro_f1_null() {
        let report =
            EvaluateCompletedReport::completed(0, "c1".to_string(), 0, 2, None).expect("report");
        assert_eq!(
            report.to_json_line().expect("json"),
            r#"{"step":"evaluate","status":"ok","candidate":0,"kind":"c1","n_total":2,"correct":0,"accuracy":0.0,"macro_f1":null}"#
        );
    }

    /// REQ-33: 壊れた値（件数 0・正解数が件数超過・範囲外や非有限の macro_f1）は作れない。
    #[test]
    fn req33_evaluate_completed_report_rejects_broken_values() {
        let make = |c, t, f| EvaluateCompletedReport::completed(0, "c1".to_string(), c, t, f);
        assert_eq!(make(0, 0, None), None);
        assert_eq!(make(5, 4, None), None);
        assert_eq!(make(1, 2, Some(f64::NAN)), None);
        assert_eq!(make(1, 2, Some(f64::INFINITY)), None);
        assert_eq!(make(1, 2, Some(1.5)), None);
        assert_eq!(make(1, 2, Some(-0.1)), None);
        assert!(make(2, 2, Some(1.0)).is_some());
    }

    /// REQ-33: 工程状態は snake_case。
    #[test]
    fn req33_stage_status_names_are_snake_case() {
        assert_eq!(
            serde_json::to_string(&StageStatus::Ok).expect("json"),
            "\"ok\""
        );
        assert_eq!(
            serde_json::to_string(&StageStatus::Skipped).expect("json"),
            "\"skipped\""
        );
    }

    /// REQ-33: 出力は 1 行（改行を含まない）。
    #[test]
    fn req33_report_is_single_line() {
        assert!(
            !PackageReport::pass(metrics(false, None))
                .to_json_line()
                .expect("json")
                .contains('\n')
        );
    }

    /// REQ-33: register の JSON が完全一致する。
    #[test]
    fn req33_register_report_json_is_exact() {
        assert_eq!(
            RegisterReport::new("ab".repeat(32), 3, false)
                .to_json_line()
                .expect("json"),
            format!(
                "{{\"step\":\"register\",\"status\":\"ok\",\"definition_sha256\":\"{}\",\"options\":3,\"evaluation_defined\":false}}",
                "ab".repeat(32)
            )
        );
    }

    /// REQ-33: inspect の JSON が完全一致する。
    #[test]
    fn req33_inspect_report_json_is_exact() {
        let split = SplitCounts {
            train: 8,
            validation: 1,
            test: 1,
        };
        assert_eq!(
            InspectStageReport::new(10, split)
                .to_json_line()
                .expect("json"),
            "{\"step\":\"inspect\",\"status\":\"ok\",\"valid_records\":10,\"split\":{\"train\":8,\"validation\":1,\"test\":1}}"
        );
    }

    /// REQ-33: train・select の JSON が完全一致する。
    #[test]
    fn req33_train_and_select_report_json_are_exact() {
        assert_eq!(
            TrainReport::new(0, "c1".to_string())
                .to_json_line()
                .expect("json"),
            "{\"step\":\"train\",\"status\":\"ok\",\"candidate\":0,\"kind\":\"c1\"}"
        );
        assert_eq!(
            SelectReport::new(1, "c3".to_string())
                .to_json_line()
                .expect("json"),
            "{\"step\":\"select\",\"status\":\"ok\",\"candidate\":1,\"kind\":\"c3\"}"
        );
    }

    /// REQ-27・#445: 予測行の exact JSON（キー順・null・エスケープ・scores 欠落）。
    #[test]
    fn req27_prediction_line_json_is_exact() {
        let line = |id, o, s| PredictionLine::new(id, o, s).to_json_line().expect("json");
        assert_eq!(
            line(
                "a\"1\n",
                PredictionLineOutcome::Label("x".into()),
                Some((&["y", "x"], &[0.25, 0.75]))
            ),
            r#"{"id":"a\"1\n","status":"ok","predicted_label":"x","scores":{"y":0.25,"x":0.75}}"#
        );
        assert_eq!(
            line("b", PredictionLineOutcome::Abstain, None),
            r#"{"id":"b","status":"abstain","predicted_label":null}"#
        );
        assert_eq!(
            line("c", PredictionLineOutcome::Error, None),
            r#"{"id":"c","status":"error","predicted_label":null}"#
        );
        assert_eq!(
            line(
                "d",
                PredictionLineOutcome::Invalid,
                Some((&["x"], &[f64::NAN]))
            ),
            r#"{"id":"d","status":"ok","predicted_label":null}"#
        );
        assert_eq!(
            line(
                "e",
                PredictionLineOutcome::Invalid,
                Some((&["x", "y"], &[0.5]))
            ),
            r#"{"id":"e","status":"ok","predicted_label":null}"#
        );
    }

    /// REQ-41・#445: 採点出力の exact JSON（キー順・`fn` の改名・null・非有限は null）。
    #[test]
    fn req41_score_report_json_is_exact() {
        let matrix = ScoreConfusionMatrix::new(
            vec!["a\"".into(), "b".into()],
            vec![vec![1, 0, 0, 0, 0], vec![0, 2, 0, 0, 1]],
        )
        .expect("matrix");
        assert!(ScoreConfusionMatrix::new(vec!["a".into()], vec![vec![1]]).is_none());
        let candidate = ScoreCandidate {
            name: "P".into(),
            role: ScoreRole::Candidate,
            pred_sha256: "ab".into(),
            correct: 3,
            accuracy: 0.75,
            accuracy_wilson95: [0.5, f64::NAN],
            macro_f1: None,
            per_label: vec![ScorePerLabel {
                label: "a\"".into(),
                support: 1,
                predicted_count: 1,
                tp: 1,
                fp: 0,
                fn_: 0,
                precision: Some(1.5),
                recall: None,
                f1: None,
            }],
            confusion_matrix: matrix,
            vs_majority: ScoreVsMajority {
                b: 2,
                c: 0,
                p: 0.5,
                verdict: BaselineComparisonVerdict::Undeterminable,
            },
        };
        let holm = ScoreHolm {
            candidate: "P".into(),
            m: 3,
            comparisons: vec![ScoreHolmComparison {
                against: "majority".into(),
                b: 2,
                c: 0,
                p_raw: 0.5,
                p_adjusted: 0.25,
                verdict: BaselineComparisonVerdict::SignificantlyBetter,
            }],
        };
        let reference = ScoreReference {
            candidate: "P".into(),
            against: "AR".into(),
            b: 1,
            c: 2,
            p_raw: 0.125,
        };
        let json = ScoreReport::new(
            1,
            "ee".into(),
            4,
            30,
            vec![candidate],
            holm,
            vec![reference],
        )
        .to_json_line()
        .expect("json");
        assert_eq!(
            json,
            concat!(
                r#"{"step":"score_predictions","status":"ok","seed":1,"evaluation_sha256":"ee","n_total":4,"required_sample_size":30,"#,
                r#""candidates":[{"name":"P","role":"candidate","pred_sha256":"ab","correct":3,"accuracy":0.75,"accuracy_wilson95":[0.5,null],"macro_f1":null,"#,
                r#""per_label":[{"label":"a\"","support":1,"predicted_count":1,"tp":1,"fp":0,"fn":0,"precision":1.5,"recall":null,"f1":null}],"#,
                r#""confusion_matrix":{"labels":["a\"","b"],"columns":["a\"","b","invalid","abstain","error"],"rows":[[1,0,0,0,0],[0,2,0,0,1]]},"#,
                r#""vs_majority":{"b":2,"c":0,"p":0.5,"verdict":"undeterminable"}}],"#,
                r#""holm":{"candidate":"P","m":3,"comparisons":[{"against":"majority","b":2,"c":0,"p_raw":0.5,"p_adjusted":0.25,"verdict":"significantly_better"}]},"#,
                r#""references":[{"candidate":"P","against":"AR","b":1,"c":2,"p_raw":0.125}]}"#
            )
        );
    }
}
