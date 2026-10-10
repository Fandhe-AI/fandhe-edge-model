//! `evaluate` 工程が残す評価完了の記録（REQ-27・REQ-33・TASK-33.1-2・#314）。
//!
//! # 位置づけ
//!
//! CLI の `evaluate` 工程（凍結した評価データへの 1 回限りの適用が成功した候補）が
//! `candidates/<N>/evaluation_record.json` へ新規に書き、`package` 工程が公開の前に読み戻して
//! 「選定候補が評価済みで、記録したモデル・評価データ・定義のハッシュが今の実体と一致すること」を
//! 確認する。記録は CLI 層のファイル規約で、共通コアは形と読み書きの規則だけを持つ
//! （I/O は行わない。`SelectionRecord` と同じ流儀）。
//!
//! # 含めるもの・含めないもの
//!
//! ハッシュ・件数・固定語彙のみ。評価データの本文・入力・正解ラベル・パスは含めない（security.md）。
//!
//! # 限界（検証済みではない）
//!
//! 記録ファイルはプロジェクトへ書き込める主体なら作り直せる。外部台帳による完全性の検証は
//! TASK-39.3-2（#168）の範囲で、本型はそれを代替しない。

use serde::{Deserialize, Serialize};

/// 評価完了記録の読み込み上限（バイト。読み込み前のサイズ確認に使う。REQ-39）。
///
/// 候補 ID・sha256（hex 64 桁）4 個・件数・下限基準比較（#339）・対象外ラベル（#478）・旧モデルとの比較（#488）・再現性（#490）のみを持つ。
/// 通常は 1 KiB 程度で、`majority_label`・`out_of_scope_label`（各最大 256 バイト）が全て JSON の
/// `\u00XX` に膨らむ最悪でもこの上限に収まる（テスト `req39_issue339_record_fits_size_limit`。
/// #478 で 4 KiB から 8 KiB に、#490 で再現性の欄〔最大 `MAX_REPRODUCIBILITY_RUNS` = 100 run。
/// 1 run 最大 約 90 バイト〕のため 16 KiB に引き上げた。旧モデルとの比較〔#488〕と同時でも最悪 約 14.3 KiB）。
pub const MAX_EVALUATION_RECORD_BYTES: u64 = 16 * 1024;

/// 評価完了記録の保存・読み込みの失敗（内容を含まない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationRecordError {
    /// JSON の直列化に失敗した。
    Serialize,
    /// 保存済みの記録を読めない・形が合わない（未知キー・型違いを含む）。
    Malformed,
}

impl std::fmt::Display for EvaluationRecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaluationRecordError::Serialize => write!(f, "failed to serialize evaluation record"),
            EvaluationRecordError::Malformed => write!(f, "evaluation record is malformed"),
        }
    }
}

impl std::error::Error for EvaluationRecordError {}

/// 下限基準（majority）との比較の判定（#339・REQ-25）。評価器の `BaselineVerdict` の写し。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineComparisonVerdict {
    /// 有意水準 0.05 で下限基準を上回る。
    SignificantlyBetter,
    /// 件数は足りているが有意差がない。
    NotSignificantlyBetter,
    /// 事前計算した必要件数に満たず判定できない（合格扱いにしない）。
    Undeterminable,
}

/// `select` が選定候補について残す、下限基準（majority）に対する有意性判定の記録
/// （REQ-18・REQ-25・REQ-27・TASK-18.3・#481）。
///
/// validation 分割だけで求め（凍結した最終 test・評価データは使わない）、`verdict` は既定候補の総数
/// `family_size`（脱落候補を含む）で Holm 補正した後の判定。p 値は `b`・`c`・`family_size` から計算し直せる
/// ため持たない。記録・報告のみで、選定・終了コードには使わない。`select` の stdout の `significance` と
/// `selection_record.json` の同名欄が同じ形を使う。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionSignificanceRecord {
    /// train 分割のラベルだけから決めた多数派ラベルの選択肢 ID。
    pub majority_label: String,
    /// validation での下限基準の正解数。
    pub baseline_correct: u64,
    /// 選定候補だけが正解した件数。
    pub b: u64,
    /// 下限基準だけが正解した件数。
    pub c: u64,
    /// 定義の仮定から事前計算した必要件数。
    pub required_n: u64,
    /// Holm 補正の族サイズ（既定候補の総数）。
    pub family_size: usize,
    /// Holm 補正後の判定。
    pub verdict: BaselineComparisonVerdict,
}

/// 下限基準との比較の記録（#339・REQ-25）。p 値は `b`・`c` から計算し直せるため持たない。
/// 値の整合は `package` が定義と train 分割から計算し直して照合する。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineComparisonRecord {
    /// train 分割のラベルだけから決めた多数派ラベルの選択肢 ID。
    pub majority_label: String,
    /// 下限基準の正解数。
    pub baseline_correct: u64,
    /// 候補だけが正解した件数。
    pub b: u64,
    /// 下限基準だけが正解した件数。
    pub c: u64,
    /// 定義の仮定から事前計算した必要件数。
    pub required_n: u64,
    /// 判定。
    pub verdict: BaselineComparisonVerdict,
}

/// 型と意味の正しさの 5 区分の件数（REQ-24・TASK-24.3・#480）。評価器の `TypeMeaningQuadrant` の写し。
///
/// 合計は評価件数（`total`）と一致する。`evaluate` の stdout の同名キーと同じ形で、`package` が
/// 合計と `total` の構造照合に使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypeMeaningQuadrantRecord {
    /// 型が正しく意味も正しい件数。
    pub type_ok_meaning_ok: u64,
    /// 型は正しいが意味が誤りの件数。
    pub type_ok_meaning_ng: u64,
    /// 型が不正の件数。
    pub type_ng_count: u64,
    /// 判定保留の件数。
    pub abstain: u64,
    /// 推論エラーの件数。
    pub error: u64,
}

impl TypeMeaningQuadrantRecord {
    /// 5 区分の合計。桁あふれ時は `None`。
    #[must_use]
    pub fn total(&self) -> Option<u64> {
        self.type_ok_meaning_ok
            .checked_add(self.type_ok_meaning_ng)?
            .checked_add(self.type_ng_count)?
            .checked_add(self.abstain)?
            .checked_add(self.error)
    }
}

/// 校正（温度スケーリング）と保留しきい値の記録（REQ-22・REQ-27・#477）。
///
/// validation 分割だけから決めた値で、凍結 test の結果は入らない。`temperature` は実際に使う温度
/// （不採用なら 1.0）。`validation_answered` は validation で確信度が `threshold` 以上の件数。
/// 最小件数は設けず、少件数でも T・τ が確定する。件数は `n_validation` で示す
/// （REQ-22 に基準なし。オーナー判断 2026-10-09）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationRecord {
    /// 実際に使う温度。
    pub temperature: f64,
    /// 推定した温度を採用したか（ECE が下がらなければ `false`）。
    pub adopted: bool,
    /// 保留しきい値 τ。
    pub threshold: f64,
    /// 校正に使った validation の件数。
    pub n_validation: u64,
    /// validation で τ 以上（保留されない）の件数。
    pub validation_answered: u64,
}

/// 保留・対象外の件数の記録（REQ-22・#479・#478）。`evaluate` の stdout `abstention` の件数部分と同じ意味。
///
/// 対象外は「答えた」側で、`out_of_scope` は `answered` の内数。`answered + abstained` は評価件数
/// （`total`）と一致する。T・τ の再計算照合はしない。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbstentionRecord {
    /// 保留にならず答えた件数（対象外を含む）。
    pub answered: u64,
    /// 保留した件数。
    pub abstained: u64,
    /// `answered` のうち対象外ラベルと判定した件数（内数）。
    pub out_of_scope: u64,
    /// 答えた行（対象外を含む）のうち正解の件数。
    pub correct_answered: u64,
}

impl AbstentionRecord {
    /// 答えた件数と保留件数の合計（評価件数と一致すべき値）。桁あふれ時は `None`。
    #[must_use]
    pub fn total(&self) -> Option<u64> {
        self.answered.checked_add(self.abstained)
    }
}

/// 比較した旧モデル（旧プロジェクトの評価記録の値。REQ-26・TASK-26.1・#488）。
///
/// `evaluate --previous-project-dir` の stdout `comparison.previous` と評価記録の
/// `previous_comparison.previous` が同じ形を使う。旧モデルの重み・ONNX は読まず、旧の評価記録の値を写すだけ。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviousModelRecord {
    /// 旧の評価記録の候補 ID。
    pub candidate_id: String,
    /// 旧の評価記録の ONNX の sha256（hex）。
    pub onnx_sha256: String,
    /// 旧の評価記録の定義の正準化ハッシュ（hex）。
    pub definition_sha256: String,
    /// 旧の評価記録の評価データの sha256（hex）。
    pub evaluation_sha256: String,
}

/// 旧・新のラベル集合から決まる比較の前提（評価器の `ComparisonPremise` の写し。REQ-26・TASK-26.2・#489）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonPremiseKind {
    /// 旧・新が同一のラベル集合（並び順は問わない）。
    SameLabelSet,
    /// ラベル集合が異なる（件数は同じ問題での差分ではない）。
    LabelSetDiffers,
}

/// 比較に使った評価データの範囲（REQ-26・REQ-17・#488）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonEvaluationData {
    /// 旧の評価データの sha256 が新の凍結 sha256 と同じ（全件で比較）。
    Same,
    /// 評価データが異なり、id・input・正解ラベルがすべて一致する共通レコードだけで比較した。
    CommonSubset,
}

/// 旧・新の正誤の 2×2 の件数（評価器の `RegressionCounts` の写し。CI は持たない。REQ-26・#488・#489）。
///
/// 4 区分の合計は `n`。`correct_to_incorrect` が回帰、`incorrect_to_correct` が改善。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegressionCountsRecord {
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
}

impl RegressionCountsRecord {
    /// 4 区分の合計。桁あふれ時は `None`。
    #[must_use]
    pub fn total(&self) -> Option<u64> {
        self.both_correct
            .checked_add(self.correct_to_incorrect)?
            .checked_add(self.incorrect_to_correct)?
            .checked_add(self.both_wrong)
    }
}

/// 旧モデルとの比較の記録（`evaluate --previous-project-dir` のときだけ。REQ-26・TASK-26.1・26.2・
/// #488・#489）。記録・報告のみで、終了コード・合否・`package` の照合には使わない。
///
/// `counts` は共通レコードが 0 件のとき `None`（`null`。0 で埋めない）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviousComparisonRecord {
    /// 比較した旧モデル。
    pub previous: PreviousModelRecord,
    /// 比較の前提（ラベル集合の一致・相違）。
    pub premise: ComparisonPremiseKind,
    /// 比較に使った評価データの範囲。
    pub evaluation_data: ComparisonEvaluationData,
    /// 比較した共通レコード数。
    pub n_common: u64,
    /// 2×2 の件数（`n_common == 0` で `None`）。
    pub counts: Option<RegressionCountsRecord>,
}

/// 再現性の判定（REQ-26・TASK-26.3・#490）。評価器の `OverlapVerdict` の写し。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReproducibilityVerdict {
    /// すべての seed の組で Wilson 95% 区間が重なる。
    AllPairsOverlap,
    /// 重ならない組が 1 組以上ある。
    SomePairsDisjoint,
}

/// 再現性の判定に使った 1 seed 分の件数（REQ-26・#490）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReproducibilityRunRecord {
    /// 学習 seed。
    pub seed: u32,
    /// 凍結 test での正解数。
    pub correct: u64,
    /// 凍結 test の評価件数。
    pub total: u64,
}

/// 3 seed 以上の Wilson 95% 区間の重なりによる再現性の記録（REQ-26・TASK-26.3・#490）。
///
/// 自分と `--seed-run-project` の評価記録の件数を seed 昇順に並べたもの。記録・報告のみで、
/// 終了コード・`package` の照合には使わない。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReproducibilityRecord {
    /// 判定に使った seed（昇順。`runs[].seed` と同じ並び）。
    pub seeds: Vec<u32>,
    /// seed ごとの件数（seed 昇順）。
    pub runs: Vec<ReproducibilityRunRecord>,
    /// 判定。
    pub verdict: ReproducibilityVerdict,
}

/// 評価完了の記録（1 候補・1 評価データ・1 回の適用に 1 つ）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRecord {
    /// 評価した候補の添字（`--candidate` と同じ）。
    pub candidate_index: usize,
    /// 評価した候補 ID。
    pub candidate_id: String,
    /// 最終 test の台帳へ事前登録した代表構成 ID。
    pub config_id: String,
    /// 評価データの sha256（hex。凍結記録の値）。
    pub evaluation_sha256: String,
    /// 評価データのバイト長（凍結記録の値）。
    pub evaluation_bytes: u64,
    /// 評価したモデル（ONNX）の sha256（hex）。
    pub onnx_sha256: String,
    /// 評価した候補の `artifact.json` の sha256（hex）。
    pub artifact_meta_sha256: String,
    /// 評価時点の定義の正準化ハッシュ（hex）。
    pub definition_sha256: String,
    /// 正解数。
    pub correct: u64,
    /// 評価件数。
    pub total: u64,
    /// 下限基準との比較（定義に `baseline_comparison` があるときだけ。#339）。
    /// 欄の無い古い記録はそのまま読める。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_comparison: Option<BaselineComparisonRecord>,
    /// 型と意味の 5 区分（#480・REQ-24）。欄の無い古い記録はそのまま読める。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub type_meaning_quadrant: Option<TypeMeaningQuadrantRecord>,
    /// 校正と保留しきい値（#477・REQ-22）。欄の無い古い記録はそのまま読める。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationRecord>,
    /// 評価時点の定義の対象外ラベル（#478）。欄の無い古い記録はそのまま読める。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_of_scope_label: Option<String>,
    /// 保留・対象外の件数（#479・REQ-22。校正が無ければ無い）。欄の無い古い記録はそのまま読める。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abstention: Option<AbstentionRecord>,
    /// 同じディレクトリの `evaluation_predictions.jsonl` のバイト列の sha256（hex。#445・REQ-27）。
    /// PoC-26 の採点入口が予測ファイルの手編集を検出するために照合する。欄の無い古い記録は
    /// そのまま読める（その場合、採点入口は拒否する）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predictions_sha256: Option<String>,
    /// 旧モデルとの比較（`--previous-project-dir` のときだけ。REQ-26・#488・#489）。欄の無い古い記録は
    /// そのまま読める。`package` は照合しない。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_comparison: Option<PreviousComparisonRecord>,
    /// 再現性（`--seed-run-project` を指定したときだけ。#490・REQ-26）。欄の無い古い記録はそのまま読める。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reproducibility: Option<ReproducibilityRecord>,
}

impl EvaluationRecord {
    /// JSON 1 行（末尾改行つき）へ直列化する。
    ///
    /// # Errors
    /// 直列化に失敗した場合。
    pub fn to_json_vec(&self) -> Result<Vec<u8>, EvaluationRecordError> {
        let mut bytes = serde_json::to_vec(self).map_err(|_| EvaluationRecordError::Serialize)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// 保存済みの記録を読む（未知キー・型違いは拒否）。
    ///
    /// # Errors
    /// JSON として不正、または形が合わない場合。
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, EvaluationRecordError> {
        serde_json::from_slice(bytes).map_err(|_| EvaluationRecordError::Malformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> EvaluationRecord {
        EvaluationRecord {
            candidate_index: 1,
            candidate_id: "c3".to_string(),
            config_id: "c3:seed42".to_string(),
            evaluation_sha256: "a".repeat(64),
            evaluation_bytes: 1184,
            onnx_sha256: "b".repeat(64),
            artifact_meta_sha256: "c".repeat(64),
            definition_sha256: "d".repeat(64),
            correct: 7,
            total: 12,
            baseline_comparison: None,
            predictions_sha256: None,
            type_meaning_quadrant: None,
            calibration: None,
            out_of_scope_label: None,
            abstention: None,
            previous_comparison: None,
            reproducibility: None,
        }
    }

    fn sample_comparison() -> BaselineComparisonRecord {
        BaselineComparisonRecord {
            majority_label: "alpha".to_string(),
            baseline_correct: 5,
            b: 3,
            c: 1,
            required_n: 221,
            verdict: BaselineComparisonVerdict::Undeterminable,
        }
    }

    /// REQ-27・REQ-33: 評価完了記録は往復でき、JSON のキーは宣言順で完全一致する。
    #[test]
    fn req27_evaluation_record_round_trips_with_exact_json() {
        let record = sample();
        let bytes = record.to_json_vec().expect("json");
        let expected = format!(
            "{{\"candidate_index\":1,\"candidate_id\":\"c3\",\"config_id\":\"c3:seed42\",\"evaluation_sha256\":\"{}\",\"evaluation_bytes\":1184,\"onnx_sha256\":\"{}\",\"artifact_meta_sha256\":\"{}\",\"definition_sha256\":\"{}\",\"correct\":7,\"total\":12}}\n",
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            "d".repeat(64)
        );
        assert_eq!(String::from_utf8(bytes.clone()).expect("utf8"), expected);
        assert_eq!(EvaluationRecord::from_json_slice(&bytes), Ok(record));
    }

    /// REQ-27: 未知キー・型違い・欠落は `Malformed`（壊れた記録を評価完了と扱わない）。
    #[test]
    fn req27_evaluation_record_rejects_unknown_keys_and_wrong_types() {
        let mut value: serde_json::Value =
            serde_json::from_slice(&sample().to_json_vec().expect("json")).expect("value");
        value["extra"] = serde_json::json!(1);
        assert_eq!(
            EvaluationRecord::from_json_slice(value.to_string().as_bytes()),
            Err(EvaluationRecordError::Malformed)
        );
        let mut value: serde_json::Value =
            serde_json::from_slice(&sample().to_json_vec().expect("json")).expect("value");
        value["correct"] = serde_json::json!("7");
        assert_eq!(
            EvaluationRecord::from_json_slice(value.to_string().as_bytes()),
            Err(EvaluationRecordError::Malformed)
        );
        assert_eq!(
            EvaluationRecord::from_json_slice(br#"{"candidate_index":0}"#),
            Err(EvaluationRecordError::Malformed)
        );
    }

    /// REQ-25・#339: 比較欄つきの記録は往復でき、キーは宣言順で完全一致する。
    #[test]
    fn req25_issue339_record_with_comparison_round_trips_with_exact_json() {
        let mut record = sample();
        record.baseline_comparison = Some(sample_comparison());
        let bytes = record.to_json_vec().expect("json");
        let text = String::from_utf8(bytes.clone()).expect("utf8");
        assert!(text.ends_with(
            ",\"baseline_comparison\":{\"majority_label\":\"alpha\",\"baseline_correct\":5,\"b\":3,\"c\":1,\"required_n\":221,\"verdict\":\"undeterminable\"}}\n"
        ));
        assert_eq!(EvaluationRecord::from_json_slice(&bytes), Ok(record));
    }

    /// REQ-27・#445: 予測ファイルの sha256 は末尾に完全一致で直列化され、往復できる。
    #[test]
    fn req27_issue445_predictions_sha256_round_trips_with_exact_json() {
        let mut record = sample();
        record.predictions_sha256 = Some("e".repeat(64));
        let bytes = record.to_json_vec().expect("json");
        let text = String::from_utf8(bytes.clone()).expect("utf8");
        assert!(text.ends_with(&format!(
            ",\"predictions_sha256\":\"{}\"}}\n",
            "e".repeat(64)
        )));
        assert_eq!(EvaluationRecord::from_json_slice(&bytes), Ok(record));
    }

    /// REQ-25・#339: 欄の無い古い記録は `None` で読める。
    #[test]
    fn req25_issue339_old_record_without_comparison_reads_as_none() {
        let bytes = sample().to_json_vec().expect("json");
        assert!(
            !String::from_utf8(bytes.clone())
                .unwrap()
                .contains("baseline_comparison")
        );
        let read = EvaluationRecord::from_json_slice(&bytes).expect("read");
        assert_eq!(read.baseline_comparison, None);
    }

    /// REQ-25・#339: 比較欄の未知キー・型違い・未知の判定語彙は `Malformed`。
    #[test]
    fn req25_issue339_comparison_rejects_malformed_shapes() {
        let mut record = sample();
        record.baseline_comparison = Some(sample_comparison());
        let base: serde_json::Value =
            serde_json::from_slice(&record.to_json_vec().expect("json")).expect("value");
        for mutate in [
            |v: &mut serde_json::Value| v["baseline_comparison"]["extra"] = serde_json::json!(1),
            |v: &mut serde_json::Value| v["baseline_comparison"]["b"] = serde_json::json!("3"),
            |v: &mut serde_json::Value| {
                v["baseline_comparison"]["verdict"] = serde_json::json!("pass")
            },
            |v: &mut serde_json::Value| {
                v["baseline_comparison"]
                    .as_object_mut()
                    .unwrap()
                    .remove("c");
            },
        ] {
            let mut value = base.clone();
            mutate(&mut value);
            assert_eq!(
                EvaluationRecord::from_json_slice(value.to_string().as_bytes()),
                Err(EvaluationRecordError::Malformed)
            );
        }
    }

    /// REQ-39・#339・#488・#490: 最悪の大きさ（256 バイトの制御文字ラベル・全件数が `u64::MAX`・旧モデルとの比較と
    /// 100 run の再現性を同時に持つ。2026-10-10 時点で 14,675 バイト）でも上限内。
    #[test]
    fn req39_issue339_record_fits_size_limit() {
        let record = EvaluationRecord {
            candidate_index: usize::MAX,
            candidate_id: "c".repeat(64),
            config_id: "c".repeat(128),
            evaluation_sha256: "a".repeat(64),
            evaluation_bytes: u64::MAX,
            onnx_sha256: "b".repeat(64),
            artifact_meta_sha256: "c".repeat(64),
            definition_sha256: "d".repeat(64),
            correct: u64::MAX,
            total: u64::MAX,
            baseline_comparison: Some(BaselineComparisonRecord {
                majority_label: "\u{1}".repeat(256),
                baseline_correct: u64::MAX,
                b: u64::MAX,
                c: u64::MAX,
                required_n: u64::MAX,
                verdict: BaselineComparisonVerdict::NotSignificantlyBetter,
            }),
            predictions_sha256: Some("e".repeat(64)),
            type_meaning_quadrant: Some(TypeMeaningQuadrantRecord {
                type_ok_meaning_ok: u64::MAX,
                type_ok_meaning_ng: u64::MAX,
                type_ng_count: u64::MAX,
                abstain: u64::MAX,
                error: u64::MAX,
            }),
            calibration: Some(CalibrationRecord {
                temperature: f64::MIN_POSITIVE,
                adopted: true,
                threshold: -1.234_567_890_123_456_7e-300,
                n_validation: u64::MAX,
                validation_answered: u64::MAX,
            }),
            out_of_scope_label: Some("\u{1}".repeat(256)),
            abstention: Some(AbstentionRecord {
                answered: u64::MAX,
                abstained: u64::MAX,
                out_of_scope: u64::MAX,
                correct_answered: u64::MAX,
            }),
            // 旧の候補 ID は `evaluate` が適用前に 128 バイト以下・制御文字なしに限る（#488）。
            // `"` は JSON で 2 倍に膨らむ。sha256 は適用前に 64 桁の hex であることを確かめる。
            previous_comparison: Some(PreviousComparisonRecord {
                previous: PreviousModelRecord {
                    candidate_id: "\"".repeat(128),
                    onnx_sha256: "f".repeat(64),
                    definition_sha256: "f".repeat(64),
                    evaluation_sha256: "f".repeat(64),
                },
                premise: ComparisonPremiseKind::LabelSetDiffers,
                evaluation_data: ComparisonEvaluationData::CommonSubset,
                n_common: u64::MAX,
                counts: Some(RegressionCountsRecord {
                    n: u64::MAX,
                    both_correct: u64::MAX,
                    correct_to_incorrect: u64::MAX,
                    incorrect_to_correct: u64::MAX,
                    both_wrong: u64::MAX,
                }),
            }),
            reproducibility: Some(ReproducibilityRecord {
                seeds: vec![u32::MAX; 100],
                runs: vec![
                    ReproducibilityRunRecord {
                        seed: u32::MAX,
                        correct: u64::MAX,
                        total: u64::MAX,
                    };
                    100
                ],
                verdict: ReproducibilityVerdict::SomePairsDisjoint,
            }),
        };
        let len = record.to_json_vec().expect("json").len() as u64;
        assert!(len <= MAX_EVALUATION_RECORD_BYTES, "len={len}");
    }

    /// REQ-24・#480: quadrant 欄つきの記録は末尾に完全一致で直列化され、往復できる。
    /// 欄の無い古い記録は `None` で読める。
    #[test]
    fn req24_issue480_quadrant_round_trips_and_old_record_reads() {
        let mut record = sample();
        let old = record.to_json_vec().expect("json");
        assert!(!String::from_utf8(old.clone()).unwrap().contains("quadrant"));
        assert_eq!(
            EvaluationRecord::from_json_slice(&old)
                .expect("old")
                .type_meaning_quadrant,
            None
        );
        let q = TypeMeaningQuadrantRecord {
            type_ok_meaning_ok: 7,
            type_ok_meaning_ng: 5,
            type_ng_count: 0,
            abstain: 0,
            error: 0,
        };
        assert_eq!(q.total(), Some(12));
        record.type_meaning_quadrant = Some(q);
        let bytes = record.to_json_vec().expect("json");
        assert!(String::from_utf8(bytes.clone()).unwrap().ends_with(
            ",\"type_meaning_quadrant\":{\"type_ok_meaning_ok\":7,\"type_ok_meaning_ng\":5,\"type_ng_count\":0,\"abstain\":0,\"error\":0}}\n"
        ));
        assert_eq!(EvaluationRecord::from_json_slice(&bytes), Ok(record));
    }

    /// REQ-22・REQ-27・#477: 校正欄つきの記録は完全一致で直列化され、往復できる。
    /// 欄の無い古い記録は `None` で読める。
    #[test]
    fn req22_issue477_calibration_round_trips_and_old_record_reads() {
        let mut record = sample();
        let old = record.to_json_vec().expect("json");
        assert_eq!(
            EvaluationRecord::from_json_slice(&old)
                .expect("old")
                .calibration,
            None
        );
        record.calibration = Some(CalibrationRecord {
            temperature: 1.25,
            adopted: true,
            threshold: 0.5,
            n_validation: 10,
            validation_answered: 8,
        });
        let bytes = record.to_json_vec().expect("json");
        assert!(String::from_utf8(bytes.clone()).unwrap().ends_with(
            ",\"calibration\":{\"temperature\":1.25,\"adopted\":true,\"threshold\":0.5,\"n_validation\":10,\"validation_answered\":8}}\n"
        ));
        assert_eq!(EvaluationRecord::from_json_slice(&bytes), Ok(record));
    }

    /// REQ-22・#479・#478: 欄つきの記録は往復でき、欄の無い古い記録は `None` で読める。
    #[test]
    fn req22_issue479_abstention_round_trips_and_old_record_reads() {
        let old = sample().to_json_vec().expect("json");
        let read = EvaluationRecord::from_json_slice(&old).expect("old");
        assert_eq!((read.abstention, read.out_of_scope_label), (None, None));
        let mut record = sample();
        record.out_of_scope_label = Some("other".to_string());
        record.abstention = Some(AbstentionRecord {
            answered: 5,
            abstained: 7,
            out_of_scope: 3,
            correct_answered: 4,
        });
        assert_eq!(record.abstention.and_then(|a| a.total()), Some(12));
        let bytes = record.to_json_vec().expect("json");
        assert!(String::from_utf8(bytes.clone()).unwrap().ends_with(
            "\"out_of_scope_label\":\"other\",\"abstention\":{\"answered\":5,\"abstained\":7,\"out_of_scope\":3,\"correct_answered\":4}}\n"
        ));
        assert_eq!(EvaluationRecord::from_json_slice(&bytes), Ok(record));
    }
    fn sample_previous(counts: Option<RegressionCountsRecord>) -> PreviousComparisonRecord {
        PreviousComparisonRecord {
            previous: PreviousModelRecord {
                candidate_id: "c1".to_string(),
                onnx_sha256: "1".repeat(64),
                definition_sha256: "2".repeat(64),
                evaluation_sha256: "3".repeat(64),
            },
            premise: ComparisonPremiseKind::SameLabelSet,
            evaluation_data: ComparisonEvaluationData::Same,
            n_common: 12,
            counts,
        }
    }

    /// REQ-26・#488・#489: 旧モデルとの比較欄つきの記録は末尾に完全一致で直列化され、往復できる。
    /// `counts:null`（共通レコード 0 件）も往復でき、欄の無い古い記録は `None` で読める。
    #[test]
    fn req26_issue488_previous_comparison_round_trips_and_old_record_reads() {
        let old = sample().to_json_vec().expect("json");
        assert!(
            !String::from_utf8(old.clone())
                .unwrap()
                .contains("previous_comparison")
        );
        assert_eq!(
            EvaluationRecord::from_json_slice(&old)
                .expect("old")
                .previous_comparison,
            None
        );
        let counts = RegressionCountsRecord {
            n: 12,
            both_correct: 6,
            correct_to_incorrect: 2,
            incorrect_to_correct: 3,
            both_wrong: 1,
        };
        assert_eq!(counts.total(), Some(12));
        let mut record = sample();
        record.previous_comparison = Some(sample_previous(Some(counts)));
        let bytes = record.to_json_vec().expect("json");
        assert!(String::from_utf8(bytes.clone()).unwrap().ends_with(&format!(
            ",\"previous_comparison\":{{\"previous\":{{\"candidate_id\":\"c1\",\"onnx_sha256\":\"{}\",\"definition_sha256\":\"{}\",\"evaluation_sha256\":\"{}\"}},\"premise\":\"same_label_set\",\"evaluation_data\":\"same\",\"n_common\":12,\"counts\":{{\"n\":12,\"both_correct\":6,\"correct_to_incorrect\":2,\"incorrect_to_correct\":3,\"both_wrong\":1}}}}}}\n",
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64)
        )));
        assert_eq!(
            EvaluationRecord::from_json_slice(&bytes),
            Ok(record.clone())
        );
        let mut empty = sample_previous(None);
        empty.n_common = 0;
        empty.premise = ComparisonPremiseKind::LabelSetDiffers;
        empty.evaluation_data = ComparisonEvaluationData::CommonSubset;
        record.previous_comparison = Some(empty);
        let bytes = record.to_json_vec().expect("json");
        assert!(String::from_utf8(bytes.clone()).unwrap().ends_with(
            "\"premise\":\"label_set_differs\",\"evaluation_data\":\"common_subset\",\"n_common\":0,\"counts\":null}}\n"
        ));
        assert_eq!(EvaluationRecord::from_json_slice(&bytes), Ok(record));
    }

    /// REQ-26・#490: 再現性の欄つきの記録は末尾に完全一致で直列化され、往復できる。
    /// 欄の無い古い記録は `None` で読め、未知キーは拒否する。
    #[test]
    fn req26_issue490_reproducibility_round_trips_and_old_record_reads() {
        let old = sample().to_json_vec().expect("json");
        assert_eq!(
            EvaluationRecord::from_json_slice(&old)
                .expect("old")
                .reproducibility,
            None
        );
        let mut record = sample();
        record.reproducibility = Some(ReproducibilityRecord {
            seeds: vec![1, 42, 77],
            runs: vec![
                ReproducibilityRunRecord {
                    seed: 1,
                    correct: 7,
                    total: 12,
                },
                ReproducibilityRunRecord {
                    seed: 42,
                    correct: 7,
                    total: 12,
                },
                ReproducibilityRunRecord {
                    seed: 77,
                    correct: 8,
                    total: 12,
                },
            ],
            verdict: ReproducibilityVerdict::AllPairsOverlap,
        });
        let bytes = record.to_json_vec().expect("json");
        let text = String::from_utf8(bytes.clone()).expect("utf8");
        assert!(
            text.ends_with(
                ",\"reproducibility\":{\"seeds\":[1,42,77],\"runs\":[{\"seed\":1,\"correct\":7,\"total\":12},{\"seed\":42,\"correct\":7,\"total\":12},{\"seed\":77,\"correct\":8,\"total\":12}],\"verdict\":\"all_pairs_overlap\"}}\n"
            ),
            "{text}"
        );
        assert_eq!(EvaluationRecord::from_json_slice(&bytes), Ok(record));
        let unknown = text.replace("\"verdict\"", "\"extra\":1,\"verdict\"");
        assert_eq!(
            EvaluationRecord::from_json_slice(unknown.as_bytes()),
            Err(EvaluationRecordError::Malformed)
        );
    }
}
