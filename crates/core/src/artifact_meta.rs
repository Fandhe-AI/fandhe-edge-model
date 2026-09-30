//! 配布パッケージのメタデータ `artifact.json` から `onnx_file` 参照と `kind` だけを取り出す
//! 暫定の最小リーダー（REQ-39・TASK-39.4-2・#159・TASK-39.2-4・#156）。
//!
//! # 位置づけ
//!
//! CLI の `infer` が、ガード層で閉じ込めて開いた `artifact.json` のバイト列を渡し、
//! ONNX ファイルの相対パス（経路検証の対象）を得るために使う。**配布パッケージ形式の
//! 確定は TASK-28・TASK-32 の担当**であり、本モジュールはスキーマを確定したものではない。
//! `onnx_file` と `kind` 以外のフィールドは解釈せず無視する（将来のフィールド追加を妨げない）。
//! `kind` は**必須**（欠落・文字列以外は拒否）。省略を許すと許可リスト検査の迂回路になるため。
//! `kind` の内容（空・構文違反・許可リスト外）は本モジュールでは判定せず、ガード層の
//! `KindAllowlist` に一本化する（core は guard に依存しない。判定規則の集約）。
//!
//! # 移行（破壊的変更。#156）
//!
//! `kind` を持たない既存の `artifact.json` は `infer` で `invalid_input`（64）になる。
//! 移行手順は、既存の `artifact.json` に `"kind": "<c1|c3|autoregressive>"` を追加すること
//! （学習ワーカーが出力する `artifact.json` は常に `kind` を持つ）。
//!
//! # 信頼境界
//!
//! 入力は信頼できないデータ。値の妥当性（ルート配下か）は本モジュールでは判定せず、
//! 呼び出し側が `fandhe-edge-guard` の経路検証を必ず通す。エラーの `Display` は入力値・
//! 本文を含めない（`.claude/rules/security.md`）。

use serde::Deserialize;
use std::fmt;

/// `artifact.json` の最大バイト数（暫定値。上限の正式値は TASK-39.5 で確定する。REQ-39）。
pub const MAX_ARTIFACT_META_BYTES: u64 = 1024 * 1024;

/// メタデータの解釈エラー。入力値を保持しない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArtifactMetaError {
    /// JSON として不正、または `onnx_file`・`kind` が欠落・文字列でない。
    Malformed,
    /// `onnx_file` が空文字列、または NUL を含む。
    InvalidOnnxFile,
}

impl fmt::Display for ArtifactMetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArtifactMetaError::Malformed => write!(f, "artifact metadata is malformed"),
            ArtifactMetaError::InvalidOnnxFile => write!(f, "onnx_file value is invalid"),
        }
    }
}

impl std::error::Error for ArtifactMetaError {}

#[derive(Deserialize)]
struct Raw {
    onnx_file: String,
    kind: String,
}

/// `artifact.json` から取り出した ONNX ファイルへの参照（未検証の相対パス文字列）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactOnnxRef {
    onnx_file: String,
    kind: String,
}

impl ArtifactOnnxRef {
    /// `artifact.json` のバイト列から `onnx_file` を取り出す。
    ///
    /// # Errors
    /// 不正な JSON・`onnx_file` の欠落や型違い・空文字列・NUL を含む値。
    pub fn parse(bytes: &[u8]) -> Result<Self, ArtifactMetaError> {
        // serde の derive は JSON 配列も位置指定で受理してしまうため、先に値として読み、
        // トップレベルがオブジェクトであることを確認してから型へ写す。
        let value: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|_| ArtifactMetaError::Malformed)?;
        if !value.is_object() {
            return Err(ArtifactMetaError::Malformed);
        }
        let raw: Raw = serde_json::from_value(value).map_err(|_| ArtifactMetaError::Malformed)?;
        if raw.onnx_file.is_empty() || raw.onnx_file.contains('\0') {
            return Err(ArtifactMetaError::InvalidOnnxFile);
        }
        Ok(Self {
            onnx_file: raw.onnx_file,
            kind: raw.kind,
        })
    }

    /// 未検証の `onnx_file` 文字列（経路検証の入力にのみ使う）。
    pub fn onnx_file(&self) -> &str {
        &self.onnx_file
    }

    /// 未検証の `kind` 文字列。ガード層の `KindAllowlist` 検査の入力にのみ使い、
    /// 検査を通す前に下流（ランタイム・出力）へ渡さない。
    pub fn kind(&self) -> &str {
        &self.kind
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39: 正常系。未知フィールドは無視する。
    #[test]
    fn req39_parse_extracts_onnx_file_and_ignores_unknown() {
        let r =
            ArtifactOnnxRef::parse(br#"{"onnx_file":"model.onnx","kind":"c3","n":1}"#).expect("ok");
        assert_eq!(r.onnx_file(), "model.onnx");
        assert_eq!(r.kind(), "c3");
    }

    /// REQ-39: 欠落・型違い・不正 JSON・配列は Malformed。
    #[test]
    fn req39_parse_rejects_malformed() {
        for b in [
            &br#"{}"#[..],
            br#"{"onnx_file":1,"kind":"c3"}"#,
            br#"{"onnx_file":null,"kind":"c3"}"#,
            br#"{"onnx_file":"model.onnx"}"#,
            br#"{"onnx_file":"model.onnx","kind":1}"#,
            br#"{"onnx_file":"model.onnx","kind":null}"#,
            br#"{"onnx_file":"model.onnx","kind":[]}"#,
            b"not json",
            br#"["onnx_file"]"#,
            b"",
        ] {
            assert_eq!(ArtifactOnnxRef::parse(b), Err(ArtifactMetaError::Malformed));
        }
    }

    /// REQ-39: 空文字列・NUL は InvalidOnnxFile。
    #[test]
    fn req39_parse_rejects_empty_and_nul() {
        assert_eq!(
            ArtifactOnnxRef::parse(br#"{"onnx_file":"","kind":"c3"}"#),
            Err(ArtifactMetaError::InvalidOnnxFile)
        );
        assert_eq!(
            ArtifactOnnxRef::parse(br#"{"onnx_file":"a\u0000b","kind":"c3"}"#),
            Err(ArtifactMetaError::InvalidOnnxFile)
        );
    }

    /// REQ-39: `kind` の内容判定は core では行わない（空・構文違反でも Ok。ガード層の責務）。
    #[test]
    fn req39_parse_does_not_judge_kind_content() {
        for k in ["", "c3; rm -rf ~", "PT"] {
            let json = format!(r#"{{"onnx_file":"model.onnx","kind":"{k}"}}"#);
            let r = ArtifactOnnxRef::parse(json.as_bytes()).expect("ok");
            assert_eq!(r.kind(), k);
        }
    }

    /// REQ-39: エラー文言に入力値を含めない。
    #[test]
    fn req39_error_display_has_no_input_value() {
        let e = ArtifactOnnxRef::parse(br#"{"onnx_file":1,"kind":"c3","secret":"../../etc"}"#)
            .unwrap_err();
        assert!(!e.to_string().contains("etc"));
    }
}
