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
//! [`SelectReport`]。件数・固定語彙のみでパス・本文を含まない）を追加した。`train --all`（探索予算内の
//! 全候補の学習。#482・#483）の結果は [`TrainAllReport`]。
//!
//! `package` 工程の [`PackageReport`] のフィールドは PoC-16 の package 工程の
//! 出力名（`step`・`status`・`judgment`・`acceptance_defined`）に、容量内訳と p95 の計測値
//! （`capacity`・`infer_p95`。#340・REQ-30・REQ-31）と、exit 0・10・12 では記録した版（`version`。
//! [`PackageVersion`]・#491・REQ-39）を末尾へ足した集合で、載せるのは整数・
//! bool・固定キーだけ（パス・データ本文は載せない。security.md）。加えて `evaluate` 工程の評価データ
//! 未定義時の [`EvaluateReport`]（`status:"skipped"`。TASK-33.3・#140）と、評価データありで
//! 評価が完了したときの [`EvaluateCompletedReport`]（正解率・Macro-F1。#314）を持つ。
//! 後者の JSON スキーマは 2026-09-30 オーナー承認済み。Wilson 区間・McNemar / Holm・診断（REQ-29）は
//! 出力に含めない（McNemar の下限基準比較は評価記録にだけ残す。#339）。
//!
//! [`PackageReport`]（exit 0）は `pass` と基準未定義のみを表す。`fail`（exit 10）・判定不能
//! （exit 12）は合否基準が定義されているときにだけ生じ、判定項目つきの
//! [`PackageJudgedReport`] で返す（#328・REQ-21・REQ-33。exit 0 の型に `fail` を載せられない
//! ことを型で保証する）。`limit_exceeded`（exit 20）は [`PackageLimitExceededReport`]
//! （`{"code","message","step","capacity","infer_p95"}`。どの上限を超えたかは各 `exceeded` で区別。#340）で返す。
//! 計測値の値型（[`PackageCapacity`]・[`InferP95`]）は runtime に依存しない整数・bool の型で、
//! 境界規則（`>`）は runtime の `LimitBreach` が唯一の実装であり、ここでは比較しない。

use std::collections::BTreeSet;

use serde::Serialize;

use crate::definition::JudgmentType;
use crate::evaluation_record::{
    BaselineComparisonVerdict, ComparisonEvaluationData, ComparisonPremiseKind,
    PreviousModelRecord, ReproducibilityVerdict, SelectionSignificanceRecord,
    TypeMeaningQuadrantRecord,
};
use crate::hash::Sha256Digest;
use crate::rebuild::{RebuildDecision, RebuildReason};

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
/// `kind`・`n_total`・`correct`・`accuracy`・`macro_f1`・`macro_f1_excluded_labels`・`per_label`・
/// `type_meaning_quadrant`・`out_of_scope_label`・`calibration`・`abstention`・`comparison`・`reproducibility`）に直列化し、`macro_f1` が未定義なら
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
    macro_f1_excluded_labels: Vec<String>,
    per_label: Vec<EvaluateLabelMetrics>,
    type_meaning_quadrant: TypeMeaningQuadrantRecord,
    out_of_scope_label: Option<String>,
    calibration: Option<EvaluateCalibration>,
    abstention: Option<EvaluateAbstention>,
    comparison: Option<EvaluateComparison>,
    reproducibility: Option<EvaluateReproducibility>,
}

/// 区間（`{"lo","hi"}`。Wilson 95% など）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EvaluateInterval {
    /// 下限。
    pub lo: f64,
    /// 上限。
    pub hi: f64,
}

/// `evaluate` の `comparison.counts`（旧・新の正誤の 2×2 と遷移率の Wilson 95% 区間。REQ-26・#488・#489）。
///
/// 4 区分の合計は `n`。区間は `correct_to_incorrect / n`・`incorrect_to_correct / n` の率に対するもの。
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EvaluateRegressionCounts {
    /// 比較した件数（共通レコード数）。
    pub n: u64,
    /// 旧・新ともに正解。
    pub both_correct: u64,
    /// 旧は正解・新は不正解（回帰）。
    pub correct_to_incorrect: u64,
    /// 旧は不正解・新は正解（改善）。
    pub incorrect_to_correct: u64,
    /// 旧・新ともに不正解。
    pub both_wrong: u64,
    /// 回帰率の Wilson 95% 区間。
    pub correct_to_incorrect_ci95: EvaluateInterval,
    /// 改善率の Wilson 95% 区間。
    pub incorrect_to_correct_ci95: EvaluateInterval,
}

/// `evaluate --previous-project-dir` の `comparison`（旧モデルとの正誤の遷移。REQ-26・TASK-26.1・26.2・
/// #488・#489）。p 値・有意性判定は持たず、終了コードに影響しない。
///
/// `counts` は `n_common == 0` のとき `None`（`null`）。`removed_labels`・`added_labels` は旧のみ・新のみの
/// ラベル（各側の宣言順。同一集合なら空）。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvaluateComparison {
    /// 比較した旧モデル。
    pub previous: PreviousModelRecord,
    /// 比較の前提。
    pub premise: ComparisonPremiseKind,
    /// 旧のみにあるラベル。
    pub removed_labels: Vec<String>,
    /// 新のみにあるラベル。
    pub added_labels: Vec<String>,
    /// 比較に使った評価データの範囲。
    pub evaluation_data: ComparisonEvaluationData,
    /// 比較した共通レコード数。
    pub n_common: u64,
    /// 旧の評価データにだけあるレコード数。
    pub n_previous_only: u64,
    /// 新の評価データにだけあるレコード数。
    pub n_current_only: u64,
    /// 2×2 の件数と区間（`n_common == 0` で `None`）。
    pub counts: Option<EvaluateRegressionCounts>,
}

impl EvaluateComparison {
    /// 件数・区間が整合しているか（`completed` の検査用）。
    fn is_consistent(&self) -> bool {
        let interval_ok = |i: &EvaluateInterval| {
            i.lo.is_finite() && i.hi.is_finite() && 0.0 <= i.lo && i.lo <= i.hi && i.hi <= 1.0
        };
        match &self.counts {
            None => self.n_common == 0,
            Some(c) => {
                c.n == self.n_common
                    && c.both_correct
                        .checked_add(c.correct_to_incorrect)
                        .and_then(|v| v.checked_add(c.incorrect_to_correct))
                        .and_then(|v| v.checked_add(c.both_wrong))
                        == Some(c.n)
                    && interval_ok(&c.correct_to_incorrect_ci95)
                    && interval_ok(&c.incorrect_to_correct_ci95)
            }
        }
    }
}

/// `evaluate` の `reproducibility.runs[]` の 1 要素（1 seed 分。REQ-26・#490）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EvaluateSeedRun {
    /// 学習 seed。
    pub seed: u32,
    /// 凍結 test での正解数。
    pub correct: u64,
    /// 凍結 test の評価件数。
    pub total: u64,
    /// 正解率の Wilson 95% 信頼区間。
    pub ci95: EvaluateInterval,
}

/// `evaluate` の `reproducibility`（3 seed 以上の Wilson 95% 区間の重なり。REQ-26・TASK-26.3・#490）。
///
/// `runs` は seed 昇順、`disjoint_pairs` は区間が重ならなかった seed の組（各組は昇順）。記録・報告のみで
/// 終了コードに影響しない。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvaluateReproducibility {
    /// seed ごとの件数と区間（seed 昇順）。
    pub runs: Vec<EvaluateSeedRun>,
    /// 判定。
    pub verdict: ReproducibilityVerdict,
    /// 区間が重ならなかった seed の組。
    pub disjoint_pairs: Vec<[u32; 2]>,
}

/// 再現性の run 数の下限（評価器 `fandhe_edge_eval::reproducibility::MIN_REPRODUCIBILITY_RUNS` の写し。
/// core は eval に依存できないため値を持ち、CLI のテストで一致を固定する）。
pub const REPRODUCIBILITY_MIN_RUNS: usize = 3;
/// 再現性の run 数の上限（評価器 `MAX_REPRODUCIBILITY_RUNS` の写し。同上）。
pub const REPRODUCIBILITY_MAX_RUNS: usize = 100;

impl EvaluateReproducibility {
    /// 構造的に整合しているか（[`EvaluateCompletedReport::completed`] の検査用。PR #516 指摘）。
    ///
    /// - run 数が [`REPRODUCIBILITY_MIN_RUNS`]`..=`[`REPRODUCIBILITY_MAX_RUNS`]
    /// - seed が狭義の昇順（重複なし）
    /// - 各 run の `total` が評価件数 `total` と一致し `correct <= total`、区間は有限で `0 <= lo <= hi <= 1`
    /// - `disjoint_pairs` の各組は `runs` の seed 2 つの昇順の組で、組は重複しない
    /// - `disjoint_pairs` が空 ⇔ `verdict` が `all_pairs_overlap`
    fn is_consistent(&self, total: u64) -> bool {
        let seeds: Vec<u32> = self.runs.iter().map(|r| r.seed).collect();
        let count_ok = (REPRODUCIBILITY_MIN_RUNS..=REPRODUCIBILITY_MAX_RUNS).contains(&seeds.len());
        let ascending = seeds.windows(2).all(|w| matches!(w, [a, b] if a < b));
        let runs_ok = self.runs.iter().all(|r| {
            let ci = r.ci95;
            r.total == total
                && r.correct <= total
                && ci.lo.is_finite()
                && ci.hi.is_finite()
                && 0.0 <= ci.lo
                && ci.lo <= ci.hi
                && ci.hi <= 1.0
        });
        let pairs_ok = self
            .disjoint_pairs
            .iter()
            .all(|[a, b]| a < b && seeds.contains(a) && seeds.contains(b))
            && self
                .disjoint_pairs
                .iter()
                .enumerate()
                .all(|(i, p)| !self.disjoint_pairs.iter().skip(i + 1).any(|q| q == p));
        let verdict_ok = self.disjoint_pairs.is_empty()
            == (self.verdict == ReproducibilityVerdict::AllPairsOverlap);
        count_ok && ascending && runs_ok && pairs_ok && verdict_ok
    }
}

/// `evaluate` の `abstention`（validation の T・τ を凍結 test に適用した保留・対象外の件数。
/// REQ-22・REQ-27・#479・#478）。
///
/// 対象外は「答えた」側（評価器の coverage の定義）で、`out_of_scope` は `answered` の内数。
/// `answered + abstained` は評価件数、`coverage = answered / total`（80% は参考値で合否条件ではない）。
/// `correct_answered` は答えた行（対象外ラベルの行を含む）のうち正解の件数。`adopted_error` は
/// 分母 = `answered` で、全件保留のとき `null`。
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EvaluateAbstention {
    /// 保留にならず答えた件数（対象外を含む）。
    pub answered: u64,
    /// 保留した件数。
    pub abstained: u64,
    /// `answered` のうち、argmax が定義の対象外ラベルだった件数（内数）。
    pub out_of_scope: u64,
    /// 保留にならず答えた割合（対象外を含む）。
    pub coverage: f64,
    /// 答えた行（対象外を含む）のうち正解の件数。
    pub correct_answered: u64,
    /// 保留込みの誤り率（分母 = 全件 − 保留。全件保留で `None`）。
    pub adopted_error: Option<f64>,
    /// 保留なしの誤り率（分母 = 全件）。
    pub unconditional_error: f64,
}

/// `evaluate` の `calibration`（validation だけで決めた温度・保留しきい値。REQ-22・REQ-27・#477）。
///
/// `temperature` は実際に使う温度（不採用なら 1.0）。`validation_coverage` は validation で
/// 確信度が `threshold` 以上の割合。
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct EvaluateCalibration {
    /// 実際に使う温度。
    pub temperature: f64,
    /// 推定した温度を採用したか。
    pub adopted: bool,
    /// 保留しきい値 τ。
    pub threshold: f64,
    /// 校正に使った validation の件数。
    pub n_validation: u64,
    /// validation で τ 以上の割合。
    pub validation_coverage: f64,
}

/// `evaluate` の `per_label[]` の 1 要素（REQ-24・TASK-24.2・#480）。
///
/// 分母 0 の `precision`・`recall`・`f1` は `null`（0 で埋めない）。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvaluateLabelMetrics {
    /// 選択肢 ID（宣言順に並べる）。
    pub label: String,
    /// 正解がこのラベルの件数。
    pub support: u64,
    /// このラベルと予測した件数。
    pub predicted: u64,
    /// 適合率（`predicted == 0` で `None`）。
    pub precision: Option<f64>,
    /// 再現率（`support == 0` で `None`）。
    pub recall: Option<f64>,
    /// F1（未定義で `None`）。
    pub f1: Option<f64>,
}

/// [`EvaluateCompletedReport::completed`] へ渡す、評価器が求めた詳細指標（#480）。
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluateDetails {
    /// Macro-F1 の平均から除いたラベル（宣言順）。
    pub macro_f1_excluded_labels: Vec<String>,
    /// ラベル別指標（宣言順）。
    pub per_label: Vec<EvaluateLabelMetrics>,
    /// 型と意味の 5 区分。合計は評価件数と一致すること。
    pub type_meaning_quadrant: TypeMeaningQuadrantRecord,
    /// 校正（validation だけから決めたもの。無ければ `None`）。
    pub calibration: Option<EvaluateCalibration>,
    /// 定義の対象外ラベル（無ければ `None`。#478）。
    pub out_of_scope_label: Option<String>,
    /// 保留・対象外の件数（校正が無ければ `None`。#479）。
    pub abstention: Option<EvaluateAbstention>,
    /// 旧モデルとの比較（`--previous-project-dir` が無ければ `None`。#488・#489）。
    pub comparison: Option<EvaluateComparison>,
    /// 再現性（`--seed-run-project` が無ければ `None`。#490）。各 run の `total` は評価件数と一致すること。
    pub reproducibility: Option<EvaluateReproducibility>,
}

impl EvaluateCompletedReport {
    /// 評価完了の結果を作る。`accuracy` は `correct / total` から求める。
    ///
    /// `total == 0`、`correct > total`、有限でない・`[0, 1]` の外の指標、
    /// 合計が `total` と一致しない `type_meaning_quadrant` は `None`。
    #[must_use]
    pub fn completed(
        candidate: usize,
        kind: String,
        correct: u64,
        total: u64,
        macro_f1: Option<f64>,
        details: EvaluateDetails,
    ) -> Option<Self> {
        if total == 0 || correct > total {
            return None;
        }
        let bad = |v: Option<f64>| v.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v));
        if bad(macro_f1)
            || details
                .per_label
                .iter()
                .any(|l| bad(l.precision) || bad(l.recall) || bad(l.f1))
            || details.type_meaning_quadrant.total() != Some(total)
            || details.calibration.is_some_and(|c| {
                !c.temperature.is_finite()
                    || c.temperature <= 0.0
                    || !(0.0..=1.0).contains(&c.threshold)
                    || !(0.0..=1.0).contains(&c.validation_coverage)
            })
            || details.abstention.is_some_and(|a| {
                a.answered.checked_add(a.abstained) != Some(total)
                    || a.out_of_scope > a.answered
                    || a.correct_answered > a.answered
                    || bad(Some(a.coverage))
                    || bad(a.adopted_error)
                    || bad(Some(a.unconditional_error))
            })
            || details
                .comparison
                .as_ref()
                .is_some_and(|c| !c.is_consistent())
            || details
                .reproducibility
                .as_ref()
                .is_some_and(|r| !r.is_consistent(total))
        {
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
            macro_f1_excluded_labels: details.macro_f1_excluded_labels,
            per_label: details.per_label,
            type_meaning_quadrant: details.type_meaning_quadrant,
            out_of_scope_label: details.out_of_scope_label,
            calibration: details.calibration,
            abstention: details.abstention,
            comparison: details.comparison,
            reproducibility: details.reproducibility,
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
/// `limit_bytes` は利用者が `limits.max_package_bytes` を設定したときだけ `Some`（未設定は JSON の
/// `null`・`exceeded:false`）。`exceeded`・`over_guideline` は呼び出し側（cli）が runtime の照合結果から
/// 渡す。ここでは `>` を計算しない（境界規則は runtime の `LimitBreach` が唯一の実装）。
/// `guideline_bytes`（目安 40MB。REQ-30）の超過は警告であり、終了コードに影響しない。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PackageCapacity {
    total_bytes: u64,
    limit_bytes: Option<u64>,
    exceeded: bool,
    guideline_bytes: u64,
    over_guideline: bool,
    components: PackageCapacityComponents,
}

impl PackageCapacity {
    /// 値をそのまま保持する。
    #[must_use]
    pub const fn new(
        total_bytes: u64,
        limit_bytes: Option<u64>,
        exceeded: bool,
        (guideline_bytes, over_guideline): (u64, bool),
        components: PackageCapacityComponents,
    ) -> Self {
        Self {
            total_bytes,
            limit_bytes,
            exceeded,
            guideline_bytes,
            over_guideline,
            components,
        }
    }

    /// 利用者設定の上限（未設定は `None`）。
    #[must_use]
    pub const fn limit_bytes(&self) -> Option<u64> {
        self.limit_bytes
    }

    /// 目安（40MB）を超えたか（警告。終了コードに影響しない）。
    #[must_use]
    pub const fn over_guideline(&self) -> bool {
        self.over_guideline
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

/// `package` が版管理台帳へ記録した版（REQ-39・#491）。exit 0・10・12 の stdout の末尾（`infer_p95` の後ろ）に
/// `"version":{"id","previous"}` として載る。exit 20 は `package/` も台帳も作らないため載せない。
///
/// `id` は今回の model 版（`v<n>`）、`previous` は `--previous-project-dir` の旧台帳の最新 model 版
/// （指定なしは `null`）。値はガード層で検証済みの版 ID の文字列。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PackageVersion {
    id: String,
    previous: Option<String>,
}

impl PackageVersion {
    /// 版の組を作る。
    #[must_use]
    pub fn new(id: String, previous: Option<String>) -> Self {
        Self { id, previous }
    }
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
    /// 記録した版（#491）。
    version: PackageVersion,
}

impl PackageJudgedReport {
    /// 合否基準を満たさない結果（exit 10・`judged_fail`）。
    #[must_use]
    pub fn fail(message: String, metrics: PackageMetrics, version: PackageVersion) -> Self {
        Self {
            code: crate::exitcode::ExitCode::JudgedFail,
            message,
            step: Stage::Package,
            judgment: PackageJudgment::Fail,
            acceptance_defined: true,
            capacity: metrics.capacity,
            infer_p95: metrics.infer_p95,
            version,
        }
    }

    /// 判定不能の結果（exit 12・`pending`。合格扱いにしない）。
    #[must_use]
    pub fn undeterminable(
        message: String,
        metrics: PackageMetrics,
        version: PackageVersion,
    ) -> Self {
        Self {
            code: crate::exitcode::ExitCode::Pending,
            message,
            step: Stage::Package,
            judgment: PackageJudgment::Undeterminable,
            acceptance_defined: true,
            capacity: metrics.capacity,
            infer_p95: metrics.infer_p95,
            version,
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
    /// 記録した版（#491）。
    version: PackageVersion,
}

impl PackageReport {
    /// 合否基準を満たした結果（PoC-16 実測の `judgment:"pass"`）。
    #[must_use]
    pub fn pass(metrics: PackageMetrics, version: PackageVersion) -> Self {
        Self {
            step: Stage::Package,
            status: StageStatus::Ok,
            judgment: Some(PackageJudgment::Pass),
            acceptance_defined: true,
            capacity: metrics.capacity,
            infer_p95: metrics.infer_p95,
            version,
        }
    }

    /// 合否基準が未設定の結果（`judgment` は `null`。exit 0）。
    #[must_use]
    pub fn acceptance_not_defined(metrics: PackageMetrics, version: PackageVersion) -> Self {
        Self {
            step: Stage::Package,
            status: StageStatus::Ok,
            judgment: None,
            acceptance_defined: false,
            capacity: metrics.capacity,
            infer_p95: metrics.infer_p95,
            version,
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
/// `evaluation_defined` は独立した評価データを取り込んだか（REQ-17）。`rebuild` は
/// `--previous-project-dir` を指定したときの作り直し判定（[`RebuildReport`]。REQ-20・#487）で、
/// 指定しなければ `null`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RegisterReport {
    step: Stage,
    status: StageStatus,
    definition_sha256: String,
    options: usize,
    evaluation_defined: bool,
    rebuild: Option<RebuildReport>,
}

impl RegisterReport {
    /// `register` の完了結果を作る。
    #[must_use]
    pub fn new(
        definition_sha256: String,
        options: usize,
        evaluation_defined: bool,
        rebuild: Option<RebuildReport>,
    ) -> Self {
        Self {
            step: Stage::Register,
            status: StageStatus::Ok,
            definition_sha256,
            options,
            evaluation_defined,
            rebuild,
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

/// 作り直し判定の区分（[`RebuildDecision`] の 3 バリアントに対応。REQ-20）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RebuildDecisionKind {
    Required,
    NotRequired,
    Unchanged,
}

/// 作り直しが必要な理由 1 件（[`RebuildReason`] の直列化形。`kind` で区別する）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RebuildReasonReport {
    OptionIdsChanged {
        added: BTreeSet<String>,
        removed: BTreeSet<String>,
    },
    JudgmentTypeChanged {
        old: JudgmentType,
        new: JudgmentType,
    },
}

/// `register --previous-project-dir` の作り直し判定（`RegisterReport` の `rebuild` 欄。
/// REQ-20・TASK-20.1〜20.3・#487。契約は 2026-10-10 オーナー承認）。
///
/// 判定は共通コアの [`crate::rebuild::decide_rebuild`] の結果をそのまま写す（ここで再判定しない）。
/// `reasons` は `required` のときだけ、`display_name_changed`・`description_changed` は
/// `not_required` のときだけ中身を持ち、それ以外は空配列。ID は辞書順（`BTreeSet`）。
/// `training_data_changed`・`evaluation_data_changed` は旧プロジェクトの取り込み済みデータとの
/// sha256 比較（片側が無ければ `null`）で、`decision` には影響しない。パス・本文は載せない。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RebuildReport {
    decision: RebuildDecisionKind,
    previous_definition_sha256: String,
    reasons: Vec<RebuildReasonReport>,
    display_name_changed: BTreeSet<String>,
    description_changed: BTreeSet<String>,
    training_data_changed: Option<bool>,
    evaluation_data_changed: Option<bool>,
}

impl RebuildReport {
    /// 判定結果と旧定義の正準化ハッシュ・データ差から作る。
    #[must_use]
    pub fn new(
        decision: &RebuildDecision,
        previous_definition_sha256: String,
        training_data_changed: Option<bool>,
        evaluation_data_changed: Option<bool>,
    ) -> Self {
        let (kind, reasons, display_name_changed, description_changed) = match decision {
            RebuildDecision::Unchanged { .. } => (
                RebuildDecisionKind::Unchanged,
                Vec::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            ),
            RebuildDecision::NotRequired(n) => (
                RebuildDecisionKind::NotRequired,
                Vec::new(),
                n.display_name_changed().clone(),
                n.description_changed().clone(),
            ),
            RebuildDecision::Required(r) => (
                RebuildDecisionKind::Required,
                r.reasons()
                    .iter()
                    .map(|reason| match reason {
                        RebuildReason::OptionIdsChanged { added, removed } => {
                            RebuildReasonReport::OptionIdsChanged {
                                added: added.clone(),
                                removed: removed.clone(),
                            }
                        }
                        RebuildReason::JudgmentTypeChanged { old, new } => {
                            RebuildReasonReport::JudgmentTypeChanged {
                                old: *old,
                                new: *new,
                            }
                        }
                    })
                    .collect(),
                BTreeSet::new(),
                BTreeSet::new(),
            ),
        };
        Self {
            decision: kind,
            previous_definition_sha256,
            reasons,
            display_name_changed,
            description_changed,
            training_data_changed,
            evaluation_data_changed,
        }
    }
}

/// `infer --input-file ... --out` が exit 0 で stdout へ返す要約 JSON（REQ-33・#459）。
///
/// `count` は OUT へ書いた結果行数、`sha256` は OUT に書いたバイト列の sha256（小文字 16 進）。
/// パスは載せない（security.md）。宣言順（`step`・`status`・`count`・`sha256`）に直列化する。
/// この JSON スキーマは 2026-10-09 にオーナー承認済み。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InferBatchReport {
    step: Stage,
    status: StageStatus,
    count: usize,
    sha256: Sha256Digest,
}

impl InferBatchReport {
    /// 書き出し完了の要約を作る。
    #[must_use]
    pub fn new(count: usize, sha256: Sha256Digest) -> Self {
        Self {
            step: Stage::Infer,
            status: StageStatus::Ok,
            count,
            sha256,
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

/// `train --all` が exit 0 で返す JSON（探索予算内の全候補の学習。REQ-18・REQ-33・TASK-18.1・
/// TASK-18.2・#482・#483）。
///
/// 値は学習ワーカー層の探索記録（`fandhe_edge_train::search::SearchRecord`）から CLI が写す。共通コアは
/// 学習ワーカー層に依存しないため、`result`・`budget_reached` は出力契約用の enum
/// （[`TrainSearchResult`]・[`TrainBudgetScope`]）で受け取り、契約外の値を表せないようにする。
/// パス・データ本文は載せない。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrainAllReport {
    step: Stage,
    status: StageStatus,
    budget_seconds: u64,
    budget_reached: bool,
    total_elapsed_ms: u64,
    candidates: Vec<TrainAllCandidate>,
}

/// [`TrainAllReport`] の候補 1 件（宣言順）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrainAllCandidate {
    /// 候補の添字（`--candidate` と同じ）。
    pub candidate: usize,
    /// 候補の種類 ID（`c1` 等）。
    pub kind: String,
    /// 探索結果の分類。
    pub result: TrainSearchResult,
    /// 予算到達の範囲。該当しなければ `null`。
    pub budget_reached: Option<TrainBudgetScope>,
}

/// `train --all` の候補ごとの探索結果（学習ワーカー層の `CandidateSearchResult` のタグと同じ語彙）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainSearchResult {
    /// 学習・validation 推論・正解率算出まで完了した。
    Evaluated,
    /// 学習が完了しなかった。
    TrainingNotCompleted,
    /// validation の採点に失敗した。
    ScoringFailed,
    /// 採点中に探索予算を超えた。
    ScoringExceededBudget,
    /// 予算切れで採点しなかった。
    ScoringSkippedBudgetExhausted,
    /// 学習が持ち時間を超えた。
    TrainingExceededTimeLimit,
    /// 学習が壁時計の期限で打ち切られた。
    TrainingTimedOut,
    /// 予算切れで開始しなかった。
    NotStarted,
}

/// `train --all` の予算到達の範囲（学習ワーカー層の `BudgetReachedScope` と同じ語彙）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainBudgetScope {
    /// 探索全体の予算に達した。
    SearchBudget,
    /// 候補ごとの持ち時間に達した。
    CandidateTimeLimit,
}

impl TrainAllReport {
    /// `train --all` の完了結果を作る。
    #[must_use]
    pub fn new(
        budget_seconds: u64,
        budget_reached: bool,
        total_elapsed_ms: u64,
        candidates: Vec<TrainAllCandidate>,
    ) -> Self {
        Self {
            step: Stage::Train,
            status: StageStatus::Ok,
            budget_seconds,
            budget_reached,
            total_elapsed_ms,
            candidates,
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
    /// 下限基準に対する有意性判定（定義に `baseline_comparison` が無いときは `null`。#481）。
    significance: Option<SelectionSignificanceRecord>,
}

impl SelectReport {
    /// `select` の完了結果を作る（選ばれた候補の添字・種類・有意性判定。REQ-18・REQ-25・#481）。
    #[must_use]
    pub fn new(
        candidate: usize,
        kind: String,
        significance: Option<SelectionSignificanceRecord>,
    ) -> Self {
        Self {
            step: Stage::Select,
            status: StageStatus::Ok,
            candidate,
            kind,
            significance,
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
    const CAP_OK: &str = r#""capacity":{"total_bytes":125,"limit_bytes":40000000,"exceeded":false,"guideline_bytes":40000000,"over_guideline":false,"components":{"weights":{"bytes":100,"file_count":1},"vocab_or_feature_transform":{"bytes":0,"file_count":0},"label_table":{"bytes":20,"file_count":1},"calibration":{"bytes":0,"file_count":0},"metadata":{"bytes":5,"file_count":1}}}"#;
    const CAP_EXCEEDED: &str = r#"{"total_bytes":125,"limit_bytes":100,"exceeded":true,"guideline_bytes":40000000,"over_guideline":false,"components":{"weights":{"bytes":100,"file_count":1},"vocab_or_feature_transform":{"bytes":0,"file_count":0},"label_table":{"bytes":20,"file_count":1},"calibration":{"bytes":0,"file_count":0},"metadata":{"bytes":5,"file_count":1}}}"#;

    /// 合成の計測値。`capacity_exceeded` が true のときは上限 100・false のときは 40,000,000。
    fn metrics(capacity_exceeded: bool, p95: Option<(u64, u64, bool)>) -> PackageMetrics {
        let c = PackageComponentSize::new;
        let components =
            PackageCapacityComponents::new(c(100, 1), c(0, 0), c(20, 1), c(0, 0), c(5, 1));
        let limit = if capacity_exceeded { 100 } else { 40_000_000 };
        PackageMetrics {
            capacity: PackageCapacity::new(
                125,
                Some(limit),
                capacity_exceeded,
                (40_000_000, false),
                components,
            ),
            infer_p95: p95.map(|(p, l, e)| InferP95::new(p, l, e)),
        }
    }

    /// `version` の期待値（#491）。
    const V2: &str = r#""version":{"id":"v2","previous":"v1"}"#;

    fn v2() -> PackageVersion {
        PackageVersion::new("v2".to_string(), Some("v1".to_string()))
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
            PackageReport::pass(metrics(false, None), v2())
                .to_json_line()
                .expect("json"),
            format!(
                r#"{{"step":"package","status":"ok","judgment":"pass","acceptance_defined":true,{CAP_OK},"infer_p95":null,{V2}}}"#
            )
        );
    }

    /// REQ-33: 合否基準未設定は `judgment:null`・`acceptance_defined:false`。
    #[test]
    fn req33_not_defined_report_json_is_exact() {
        assert_eq!(
            PackageReport::acceptance_not_defined(
                metrics(false, Some((5000, 6000, false))),
                PackageVersion::new("v1".to_string(), None)
            )
            .to_json_line()
            .expect("json"),
            format!(
                r#"{{"step":"package","status":"ok","judgment":null,"acceptance_defined":false,{CAP_OK},"infer_p95":{{"p95_us":5000,"limit_us":6000,"exceeded":false}},"version":{{"id":"v1","previous":null}}}}"#
            )
        );
    }

    /// REQ-21・REQ-33・#328: exit 10・12 の判定項目つき JSON と終了コード。
    #[test]
    fn req33_issue328_judged_reports_json_and_exit_code_are_exact() {
        use crate::exitcode::ExitCode;
        let fail =
            PackageJudgedReport::fail("judged as fail".to_string(), metrics(false, None), v2());
        assert_eq!(fail.exit_code(), ExitCode::JudgedFail);
        assert_eq!(
            fail.to_json_line().expect("json"),
            format!(
                r#"{{"code":"judged_fail","message":"judged as fail","step":"package","judgment":"fail","acceptance_defined":true,{CAP_OK},"infer_p95":null,{V2}}}"#
            )
        );
        let pending = PackageJudgedReport::undeterminable(
            "result is pending".to_string(),
            metrics(false, Some((1, 2, false))),
            v2(),
        );
        assert_eq!(pending.exit_code(), ExitCode::Pending);
        assert_eq!(
            pending.to_json_line().expect("json"),
            format!(
                r#"{{"code":"pending","message":"result is pending","step":"package","judgment":"undeterminable","acceptance_defined":true,{CAP_OK},"infer_p95":{{"p95_us":1,"limit_us":2,"exceeded":false}},{V2}}}"#
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

    fn details(quadrant: [u64; 5]) -> EvaluateDetails {
        EvaluateDetails {
            macro_f1_excluded_labels: vec!["c".to_string()],
            per_label: vec![
                EvaluateLabelMetrics {
                    label: "a".to_string(),
                    support: 2,
                    predicted: 2,
                    precision: Some(0.5),
                    recall: Some(0.5),
                    f1: Some(0.5),
                },
                EvaluateLabelMetrics {
                    label: "c".to_string(),
                    support: 0,
                    predicted: 0,
                    precision: None,
                    recall: None,
                    f1: None,
                },
            ],
            type_meaning_quadrant: TypeMeaningQuadrantRecord {
                type_ok_meaning_ok: quadrant[0],
                type_ok_meaning_ng: quadrant[1],
                type_ng_count: quadrant[2],
                abstain: quadrant[3],
                error: quadrant[4],
            },
            calibration: None,
            out_of_scope_label: None,
            abstention: None,
            comparison: None,
            reproducibility: None,
        }
    }

    /// REQ-22・REQ-27・#479・#478: `abstention`・`out_of_scope_label` つきの JSON が完全一致し、
    /// `answered + abstained` が評価件数と合わないもの・`out_of_scope > answered`・全件保留の `adopted_error` は `null`。
    #[test]
    fn req22_issue479_abstention_json_is_exact() {
        let build = |a: EvaluateAbstention| {
            let mut d = details([3, 1, 0, 0, 0]);
            d.out_of_scope_label = Some("c".to_string());
            d.abstention = Some(a);
            EvaluateCompletedReport::completed(1, "c3".to_string(), 3, 4, Some(0.5), d)
        };
        let ok = EvaluateAbstention {
            answered: 3,
            abstained: 1,
            out_of_scope: 1,
            coverage: 0.75,
            correct_answered: 3,
            adopted_error: Some(0.0),
            unconditional_error: 0.25,
        };
        let line = build(ok).expect("report").to_json_line().expect("json");
        assert!(
            line.contains(r#""out_of_scope_label":"c","calibration":null,"abstention":{"answered":3,"abstained":1,"out_of_scope":1,"coverage":0.75,"correct_answered":3,"adopted_error":0.0,"unconditional_error":0.25},"comparison":null,"reproducibility":null}"#),
            "{line}"
        );
        let all_abstained = EvaluateAbstention {
            answered: 0,
            abstained: 4,
            out_of_scope: 0,
            coverage: 0.0,
            correct_answered: 0,
            adopted_error: None,
            unconditional_error: 0.25,
        };
        let line = build(all_abstained)
            .expect("report")
            .to_json_line()
            .expect("json");
        assert!(line.contains(r#""adopted_error":null,"#), "{line}");
        assert_eq!(build(EvaluateAbstention { answered: 4, ..ok }), None);
        assert_eq!(
            build(EvaluateAbstention {
                out_of_scope: 4,
                ..ok
            }),
            None
        );
    }

    /// REQ-26・#490: `reproducibility` つきの JSON が末尾に完全一致する。構造的に整合しないもの（run 数が
    /// 3 未満・上限超、seed の重複・非昇順、評価件数と合わない run・範囲外の区間、`runs` に無い seed や
    /// 降順・重複した `disjoint_pairs`、`verdict` と `disjoint_pairs` の食い違い）は作れない（PR #516 指摘）。
    #[test]
    fn req26_issue490_reproducibility_json_is_exact() {
        let run = |seed: u32, correct: u64, lo: f64, hi: f64| EvaluateSeedRun {
            seed,
            correct,
            total: 4,
            ci95: EvaluateInterval { lo, hi },
        };
        let ok_runs = || {
            vec![
                run(1, 0, 0.0, 0.5),
                run(7, 4, 0.5, 1.0),
                run(42, 4, 0.5, 1.0),
            ]
        };
        let build = |runs: Vec<EvaluateSeedRun>,
                     verdict: ReproducibilityVerdict,
                     disjoint_pairs: Vec<[u32; 2]>| {
            let mut d = details([3, 1, 0, 0, 0]);
            d.reproducibility = Some(EvaluateReproducibility {
                runs,
                verdict,
                disjoint_pairs,
            });
            EvaluateCompletedReport::completed(1, "c3".to_string(), 3, 4, Some(0.5), d)
        };
        let disjoint = ReproducibilityVerdict::SomePairsDisjoint;
        let overlap = ReproducibilityVerdict::AllPairsOverlap;
        let line = build(ok_runs(), disjoint, vec![[1, 7], [1, 42]])
            .expect("report")
            .to_json_line()
            .expect("json");
        assert!(
            line.ends_with(r#""abstention":null,"comparison":null,"reproducibility":{"runs":[{"seed":1,"correct":0,"total":4,"ci95":{"lo":0.0,"hi":0.5}},{"seed":7,"correct":4,"total":4,"ci95":{"lo":0.5,"hi":1.0}},{"seed":42,"correct":4,"total":4,"ci95":{"lo":0.5,"hi":1.0}}],"verdict":"some_pairs_disjoint","disjoint_pairs":[[1,7],[1,42]]}}"#),
            "{line}"
        );
        let all = vec![run(1, 2, 0.1, 0.9); REPRODUCIBILITY_MAX_RUNS]
            .into_iter()
            .enumerate()
            .map(|(i, mut r)| {
                r.seed = u32::try_from(i).expect("seed");
                r
            })
            .collect::<Vec<_>>();
        assert!(build(all.clone(), overlap, vec![]).is_some());
        // run 数: 2 件・上限超は作れない。
        assert_eq!(build(ok_runs()[..2].to_vec(), overlap, vec![]), None);
        let mut too_many = all;
        too_many.push(run(u32::MAX, 2, 0.1, 0.9));
        assert_eq!(build(too_many, overlap, vec![]), None);
        // seed の重複・非昇順。
        let mut dup = ok_runs();
        dup[1].seed = 1;
        assert_eq!(build(dup, overlap, vec![]), None);
        let mut unsorted = ok_runs();
        unsorted.swap(1, 2);
        assert_eq!(build(unsorted, overlap, vec![]), None);
        // 評価件数と合わない run・範囲外の区間。
        let mut other_total = ok_runs();
        other_total[0].total = 5;
        assert_eq!(build(other_total, overlap, vec![]), None);
        for (lo, hi) in [(0.6, 0.5), (-0.1, 0.5), (0.1, 1.1), (f64::NAN, 0.5)] {
            let mut bad = ok_runs();
            bad[0].ci95 = EvaluateInterval { lo, hi };
            assert_eq!(build(bad, overlap, vec![]), None, "{lo} {hi}");
        }
        // disjoint_pairs: runs に無い seed・降順・同じ seed・重複した組。
        for pairs in [
            vec![[1, 9]],
            vec![[7, 1]],
            vec![[1, 1]],
            vec![[1, 7], [1, 7]],
        ] {
            assert_eq!(build(ok_runs(), disjoint, pairs.clone()), None, "{pairs:?}");
        }
        // verdict と disjoint_pairs の食い違い。
        assert_eq!(build(ok_runs(), overlap, vec![[1, 7]]), None);
        assert_eq!(build(ok_runs(), disjoint, vec![]), None);
    }

    /// REQ-22・REQ-27・#477: 校正つきの JSON は `calibration` が完全一致し、範囲外のしきい値は作れない。
    #[test]
    fn req22_issue477_calibration_json_is_exact() {
        let calibration = |threshold: f64| {
            let mut d = details([3, 1, 0, 0, 0]);
            d.calibration = Some(EvaluateCalibration {
                temperature: 1.23,
                adopted: true,
                threshold,
                n_validation: 120,
                validation_coverage: 0.8,
            });
            EvaluateCompletedReport::completed(1, "c3".to_string(), 3, 4, Some(0.5), d)
        };
        let line = calibration(0.61)
            .expect("report")
            .to_json_line()
            .expect("json");
        assert!(
            line.contains(r#""calibration":{"temperature":1.23,"adopted":true,"threshold":0.61,"n_validation":120,"validation_coverage":0.8},"abstention":null,"comparison":null,"reproducibility":null}"#),
            "{line}"
        );
        assert_eq!(calibration(1.5), None);
        let bad_temperature = |t: f64| {
            let mut d = details([3, 1, 0, 0, 0]);
            d.calibration = Some(EvaluateCalibration {
                temperature: t,
                adopted: false,
                threshold: 0.5,
                n_validation: 1,
                validation_coverage: 1.0,
            });
            EvaluateCompletedReport::completed(1, "c3".to_string(), 3, 4, Some(0.5), d)
        };
        assert_eq!(bad_temperature(0.0), None);
        assert_eq!(bad_temperature(-1.0), None);
        assert_eq!(bad_temperature(f64::NAN), None);
    }

    /// REQ-33・REQ-24・#480: 評価完了の JSON が完全一致する（キーは宣言順・後続 3 キーは `null`・
    /// 分母 0 のラベルは `null`・除外ラベルを列挙）。
    #[test]
    fn req24_evaluate_completed_report_json_is_exact() {
        let report = EvaluateCompletedReport::completed(
            1,
            "c3".to_string(),
            3,
            4,
            Some(0.5),
            details([3, 1, 0, 0, 0]),
        )
        .expect("report");
        assert_eq!(
            report.to_json_line().expect("json"),
            r#"{"step":"evaluate","status":"ok","candidate":1,"kind":"c3","n_total":4,"correct":3,"accuracy":0.75,"macro_f1":0.5,"macro_f1_excluded_labels":["c"],"per_label":[{"label":"a","support":2,"predicted":2,"precision":0.5,"recall":0.5,"f1":0.5},{"label":"c","support":0,"predicted":0,"precision":null,"recall":null,"f1":null}],"type_meaning_quadrant":{"type_ok_meaning_ok":3,"type_ok_meaning_ng":1,"type_ng_count":0,"abstain":0,"error":0},"out_of_scope_label":null,"calibration":null,"abstention":null,"comparison":null,"reproducibility":null}"#
        );
    }

    /// REQ-26・#488・#489: `comparison` つきの JSON が `abstention` の後ろに完全一致で並ぶ。`counts` は
    /// `n_common == 0` で `null`。件数の合計・`n` と `n_common` の食い違い・区間の範囲外は構築できない。
    #[test]
    fn req26_issue488_comparison_json_is_exact_and_checked() {
        let interval = |lo: f64, hi: f64| EvaluateInterval { lo, hi };
        let comparison =
            |counts: Option<EvaluateRegressionCounts>, n_common: u64| EvaluateComparison {
                previous: PreviousModelRecord {
                    candidate_id: "c1".to_string(),
                    onnx_sha256: "1".repeat(64),
                    definition_sha256: "2".repeat(64),
                    evaluation_sha256: "3".repeat(64),
                },
                premise: ComparisonPremiseKind::LabelSetDiffers,
                removed_labels: vec!["c".to_string()],
                added_labels: vec!["d".to_string()],
                evaluation_data: ComparisonEvaluationData::CommonSubset,
                n_common,
                n_previous_only: 2,
                n_current_only: 1,
                counts,
            };
        let build = |c: EvaluateComparison| {
            let mut d = details([3, 1, 0, 0, 0]);
            d.comparison = Some(c);
            EvaluateCompletedReport::completed(1, "c3".to_string(), 3, 4, Some(0.5), d)
        };
        let counts = EvaluateRegressionCounts {
            n: 3,
            both_correct: 1,
            correct_to_incorrect: 1,
            incorrect_to_correct: 1,
            both_wrong: 0,
            correct_to_incorrect_ci95: interval(0.25, 0.5),
            incorrect_to_correct_ci95: interval(0.125, 0.75),
        };
        let line = build(comparison(Some(counts), 3))
            .expect("report")
            .to_json_line()
            .expect("json");
        assert!(
            line.ends_with(&format!(
                r#""abstention":null,"comparison":{{"previous":{{"candidate_id":"c1","onnx_sha256":"{}","definition_sha256":"{}","evaluation_sha256":"{}"}},"premise":"label_set_differs","removed_labels":["c"],"added_labels":["d"],"evaluation_data":"common_subset","n_common":3,"n_previous_only":2,"n_current_only":1,"counts":{{"n":3,"both_correct":1,"correct_to_incorrect":1,"incorrect_to_correct":1,"both_wrong":0,"correct_to_incorrect_ci95":{{"lo":0.25,"hi":0.5}},"incorrect_to_correct_ci95":{{"lo":0.125,"hi":0.75}}}}}},"reproducibility":null}}"#,
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64)
            )),
            "{line}"
        );
        let empty = build(comparison(None, 0))
            .expect("report")
            .to_json_line()
            .expect("json");
        assert!(
            empty.ends_with(
                r#""n_common":0,"n_previous_only":2,"n_current_only":1,"counts":null},"reproducibility":null}"#
            ),
            "{empty}"
        );
        assert_eq!(build(comparison(None, 3)), None);
        assert_eq!(build(comparison(Some(counts), 4)), None);
        let mut bad = counts;
        bad.both_wrong = 1;
        assert_eq!(build(comparison(Some(bad), 3)), None);
        let mut bad = counts;
        bad.incorrect_to_correct_ci95 = interval(0.8, 0.2);
        assert_eq!(build(comparison(Some(bad), 3)), None);
        let mut bad = counts;
        bad.correct_to_incorrect_ci95 = interval(f64::NAN, 0.5);
        assert_eq!(build(comparison(Some(bad), 3)), None);
    }

    /// REQ-24: `macro_f1` が未定義なら `null`（0 や 1 で埋めない）。
    #[test]
    fn req24_evaluate_completed_report_macro_f1_null() {
        let report = EvaluateCompletedReport::completed(
            0,
            "c1".to_string(),
            0,
            2,
            None,
            details([0, 2, 0, 0, 0]),
        )
        .expect("report");
        assert!(
            report
                .to_json_line()
                .expect("json")
                .contains(r#""accuracy":0.0,"macro_f1":null,"macro_f1_excluded_labels""#)
        );
    }

    /// REQ-33: 壊れた値（件数 0・正解数が件数超過・範囲外や非有限の指標・合計が件数と違う quadrant）は作れない。
    #[test]
    fn req33_evaluate_completed_report_rejects_broken_values() {
        let make = |c: u64, t: u64, f| {
            EvaluateCompletedReport::completed(
                0,
                "c1".to_string(),
                c,
                t,
                f,
                details([c, t.saturating_sub(c), 0, 0, 0]),
            )
        };
        assert_eq!(make(5, 4, None), None);
        assert_eq!(make(1, 2, Some(f64::NAN)), None);
        assert_eq!(make(1, 2, Some(f64::INFINITY)), None);
        assert_eq!(make(1, 2, Some(1.5)), None);
        assert_eq!(make(1, 2, Some(-0.1)), None);
        assert!(make(2, 2, Some(1.0)).is_some());
        assert_eq!(make(0, 0, None), None);
        let mismatch = EvaluateCompletedReport::completed(
            0,
            "c1".to_string(),
            1,
            2,
            None,
            details([1, 0, 0, 0, 0]),
        );
        assert_eq!(mismatch, None);
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
            !PackageReport::pass(metrics(false, None), v2())
                .to_json_line()
                .expect("json")
                .contains('\n')
        );
    }

    /// REQ-33: register の JSON が完全一致する。
    #[test]
    fn req33_register_report_json_is_exact() {
        assert_eq!(
            RegisterReport::new("ab".repeat(32), 3, false, None)
                .to_json_line()
                .expect("json"),
            format!(
                "{{\"step\":\"register\",\"status\":\"ok\",\"definition_sha256\":\"{}\",\"options\":3,\"evaluation_defined\":false,\"rebuild\":null}}",
                "ab".repeat(32)
            )
        );
    }

    /// REQ-20・#487: 判定型の変更と選択肢 ID の変更は `kind` で区別した理由として、ID は辞書順で出る。
    #[test]
    fn req20_register_rebuild_required_json_is_exact() {
        let decision = RebuildDecision::Required(
            crate::rebuild::RequiredRebuild::from_reasons(vec![
                RebuildReason::OptionIdsChanged {
                    added: ["d".to_string(), "b".to_string()].into(),
                    removed: ["c".to_string()].into(),
                },
                RebuildReason::JudgmentTypeChanged {
                    old: JudgmentType::SingleSelect,
                    new: JudgmentType::TestOnlyAlternate,
                },
            ])
            .expect("non-empty"),
        );
        let report = RebuildReport::new(&decision, "cd".repeat(32), Some(true), None);
        assert_eq!(
            RegisterReport::new("ab".repeat(32), 3, true, Some(report))
                .to_json_line()
                .expect("json"),
            format!(
                "{{\"step\":\"register\",\"status\":\"ok\",\"definition_sha256\":\"{}\",\"options\":3,\"evaluation_defined\":true,\"rebuild\":{{\"decision\":\"required\",\"previous_definition_sha256\":\"{}\",\"reasons\":[{{\"kind\":\"option_ids_changed\",\"added\":[\"b\",\"d\"],\"removed\":[\"c\"]}},{{\"kind\":\"judgment_type_changed\",\"old\":\"single_select\",\"new\":\"test_only_alternate\"}}],\"display_name_changed\":[],\"description_changed\":[],\"training_data_changed\":true,\"evaluation_data_changed\":null}}}}",
                "ab".repeat(32),
                "cd".repeat(32)
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

    /// REQ-18・#482・#483: `train --all` の JSON が契約どおりのキー順・値で完全一致する。
    #[test]
    fn req18_train_all_report_json_is_exact() {
        let report = TrainAllReport::new(
            3600,
            true,
            4210,
            vec![
                TrainAllCandidate {
                    candidate: 0,
                    kind: "c1".to_string(),
                    result: TrainSearchResult::Evaluated,
                    budget_reached: None,
                },
                TrainAllCandidate {
                    candidate: 1,
                    kind: "c3".to_string(),
                    result: TrainSearchResult::TrainingTimedOut,
                    budget_reached: Some(TrainBudgetScope::CandidateTimeLimit),
                },
            ],
        );
        assert_eq!(
            report.to_json_line().expect("json"),
            "{\"step\":\"train\",\"status\":\"ok\",\"budget_seconds\":3600,\"budget_reached\":true,\"total_elapsed_ms\":4210,\"candidates\":[{\"candidate\":0,\"kind\":\"c1\",\"result\":\"evaluated\",\"budget_reached\":null},{\"candidate\":1,\"kind\":\"c3\",\"result\":\"training_timed_out\",\"budget_reached\":\"candidate_time_limit\"}]}"
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
            SelectReport::new(1, "c3".to_string(), None)
                .to_json_line()
                .expect("json"),
            "{\"step\":\"select\",\"status\":\"ok\",\"candidate\":1,\"kind\":\"c3\",\"significance\":null}"
        );
    }

    /// REQ-18・REQ-25・#481: 有意性判定つきの select の JSON が完全一致する（既存キーの後ろ・p 値なし）。
    #[test]
    fn req25_issue481_select_report_with_significance_json_is_exact() {
        let significance = SelectionSignificanceRecord {
            majority_label: "a".to_string(),
            baseline_correct: 40,
            b: 30,
            c: 5,
            required_n: 168,
            family_size: 2,
            verdict: BaselineComparisonVerdict::SignificantlyBetter,
        };
        assert_eq!(
            SelectReport::new(0, "c1".to_string(), Some(significance))
                .to_json_line()
                .expect("json"),
            "{\"step\":\"select\",\"status\":\"ok\",\"candidate\":0,\"kind\":\"c1\",\"significance\":{\"majority_label\":\"a\",\"baseline_correct\":40,\"b\":30,\"c\":5,\"required_n\":168,\"family_size\":2,\"verdict\":\"significantly_better\"}}"
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

    /// REQ-33: `infer --out` の要約はキー順 `step`・`status`・`count`・`sha256` で出る。
    #[test]
    fn req33_infer_batch_report_json_shape() {
        let report = InferBatchReport::new(2, Sha256Digest::of_bytes(b"abc"));
        assert_eq!(
            report.to_json_line().unwrap(),
            r#"{"step":"infer","status":"ok","count":2,"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}"#
        );
    }
}
