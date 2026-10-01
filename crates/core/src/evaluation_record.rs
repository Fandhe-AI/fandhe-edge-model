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
/// 候補 ID・sha256（hex 64 桁）4 個・件数・下限基準比較（#339）のみを持つ。通常は 1 KiB 程度で、
/// 比較欄の `majority_label`（最大 256 バイト）が全て JSON の `\u00XX` に膨らむ最悪でも
/// この上限に収まる（テスト `req39_issue339_record_fits_size_limit`）。
pub const MAX_EVALUATION_RECORD_BYTES: u64 = 4 * 1024;

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

/// 評価完了の記録（1 候補・1 評価データ・1 回の適用に 1 つ）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

    /// REQ-39・#339: 最悪の大きさ（256 バイトの制御文字ラベル・全件数が `u64::MAX`）でも上限内。
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
        };
        let len = record.to_json_vec().expect("json").len() as u64;
        assert!(len <= MAX_EVALUATION_RECORD_BYTES, "len={len}");
    }
}
