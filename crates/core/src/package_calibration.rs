//! 配布パッケージの `calibration.json`（温度 T・保留しきい値 τ。REQ-22・REQ-30・REQ-32・#497）。
//!
//! CLI の `package` 工程が、選定候補の評価記録に校正（`calibration`）があるときだけ書き、`infer` が
//! 読んで保留（exit 12）の判定に使う。評価データが無い・校正が無いプロジェクトでは書かない
//! （`infer` は保留を返さない）。
//!
//! 形式は 1 行 JSON＋改行で、キー順は `onnx_sha256`・`label_order`・`temperature`・`threshold` に固定
//! （構造体の宣言順で直列化する）。校正の完全性は `onnx_sha256`（配布する ONNX の sha256）と
//! `label_order`（定義の選択肢の宣言順）による自己整合で重み・定義へ束縛する。τ の改変とファイル自体の
//! 削除は、`package` が配布用の `artifact.json` に記す `calibration_sha256`（本ファイル全体の sha256。
//! [`crate::artifact_meta::ArtifactMeta::calibration_sha256`]）との照合で `infer` が検出する（パッケージ内の
//! 自己整合。REQ-39）。`artifact.json` ごとの改変の検出は外部台帳（#168）の範囲。値の範囲（T・τ）の検証は
//! 読み手（`infer`）が評価器の定数で行う（共通コアは評価器に依存しない）。

use serde::{Deserialize, Serialize};

/// パッケージ内のファイル名。
pub const PACKAGE_CALIBRATION_FILE: &str = "calibration.json";

/// `calibration.json` の中身（未知キーは拒否）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageCalibration {
    /// 配布する ONNX の sha256（16 進 64 桁）。
    pub onnx_sha256: String,
    /// 定義の選択肢 ID（宣言順）。
    pub label_order: Vec<String>,
    /// 適用する温度（評価記録の `temperature`。不採用なら 1.0）。
    pub temperature: f64,
    /// 保留しきい値 τ。
    pub threshold: f64,
}

impl PackageCalibration {
    /// 1 行 JSON＋改行へ直列化する。
    ///
    /// # Errors
    /// 直列化に失敗した場合。
    pub fn to_json_line(&self) -> Result<Vec<u8>, serde_json::Error> {
        let mut bytes = serde_json::to_vec(self)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// 読み込む（未知キー・型違い・欠落は拒否）。
    ///
    /// # Errors
    /// JSON として不正、または形が合わない場合。
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-30・#497: キー順は固定で、1 行＋改行。読み戻すと同じ値。未知キーは拒否する。
    #[test]
    fn req30_issue497_calibration_file_has_fixed_key_order_and_rejects_unknown_keys() {
        let c = PackageCalibration {
            onnx_sha256: "ab".repeat(32),
            label_order: vec!["a".to_string(), "b".to_string()],
            temperature: 1.25,
            threshold: 0.625,
        };
        let line = c.to_json_line().unwrap();
        assert_eq!(
            String::from_utf8(line.clone()).unwrap(),
            format!(
                "{{\"onnx_sha256\":\"{}\",\"label_order\":[\"a\",\"b\"],\"temperature\":1.25,\"threshold\":0.625}}\n",
                "ab".repeat(32)
            )
        );
        assert_eq!(PackageCalibration::from_json_slice(&line).unwrap(), c);
        let extra = String::from_utf8(line)
            .unwrap()
            .replace("}\n", ",\"x\":1}\n");
        assert!(PackageCalibration::from_json_slice(extra.as_bytes()).is_err());
    }
}
