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
//! # 後方互換（#174）
//!
//! `kind_version` は後方互換のため**省略可**。省略時は版管理導入前のパッケージとみなして
//! [`LEGACY_KIND_VERSION`]（1）を返す。`kind_version` を持つ場合は `u32` の範囲の整数のみを受理し、
//! 文字列・負数・小数・`null`・範囲外は拒否する（省略と不正値は区別する）。値の内容（許可リスト外か）は
//! 本モジュールでは判定せず、省略時の 1 も含めてガード層の `KindVersionAllowlist` で検査する
//! （許可リストに 1 が無くなれば、旧形式のパッケージも拒否される）。
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

/// `kind_version` を持たない従来の `artifact.json` に当てはめる版（版管理導入前は全て 1 相当。REQ-39・#174）。
pub const LEGACY_KIND_VERSION: u32 = 1;

/// メタデータの解釈エラー。入力値を保持しない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArtifactMetaError {
    /// JSON として不正、または `onnx_file`・`kind` が欠落・文字列でない。
    Malformed,
    /// `onnx_file` が空文字列、または NUL を含む。
    InvalidOnnxFile,
    /// 語彙ファイルが許可された形式でない。
    InvalidVocab,
}

impl fmt::Display for ArtifactMetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArtifactMetaError::Malformed => write!(f, "artifact metadata is malformed"),
            ArtifactMetaError::InvalidOnnxFile => write!(f, "onnx_file value is invalid"),
            ArtifactMetaError::InvalidVocab => write!(f, "vocab file format is invalid"),
        }
    }
}

impl std::error::Error for ArtifactMetaError {}

#[derive(Deserialize)]
struct Raw {
    onnx_file: String,
    kind: String,
    #[serde(default = "legacy_kind_version")]
    kind_version: u32,
}

fn legacy_kind_version() -> u32 {
    LEGACY_KIND_VERSION
}

/// `artifact.json` から取り出した ONNX ファイルへの参照（未検証の相対パス文字列）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactOnnxRef {
    onnx_file: String,
    kind: String,
    kind_version: u32,
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
            kind_version: raw.kind_version,
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

    /// 未検証の `kind_version`（省略時は [`LEGACY_KIND_VERSION`]）。ガード層の `KindVersionAllowlist` 検査の入力にのみ使い、
    /// 検査を通す前に下流（ランタイム・出力）へ渡さない。
    pub fn kind_version(&self) -> u32 {
        self.kind_version
    }
}

/// 推論に必要な `artifact.json` の項目（`kind`・`kind_version`・`max_bytes`・`label_order`・
/// `onnx_file`・`onnx_sha256`。TASK-33.1-2・#136）。
///
/// 推論経路（`infer`）が学習側の `fandhe-edge-train` に依存しないよう（REQ-32）、学習ワーカーの
/// 成果物記録型を使わず、必要項目だけをここで読む。未知フィールドは無視する。入力は信頼できない
/// データとして、型・範囲を fail-closed で検証する。値の経路検証（`onnx_file` がルート配下か）は
/// 呼び出し側がガード層で行う。`onnx_sha256` の照合（パッケージの自己整合性）は呼び出し側が
/// 開いたバイト列に対して行う。外部台帳による完全性検証（#168）・`kind_version` の許可リスト
/// 検証（#174）の代替ではない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactMeta {
    onnx_file: String,
    kind: String,
    kind_version: u32,
    max_bytes: u32,
    label_order: Vec<String>,
    onnx_sha256: String,
    vocab_sha256: Option<String>,
}

/// `label_order` の最大件数（定義の選択肢数の上限と同じ）。
const MAX_META_LABELS: usize = 1024;
/// `kind` の最大バイト数。
const MAX_META_KIND_BYTES: usize = 64;

#[derive(Deserialize)]
struct RawMeta {
    onnx_file: String,
    kind: String,
    kind_version: u32,
    max_bytes: u32,
    label_order: Vec<String>,
    onnx_sha256: String,
    #[serde(default)]
    vocab_sha256: Option<String>,
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// 語彙ファイル（`vocab.json`）の最大エントリ数（REQ-39。巨大な語彙でのアロケーション上限）。
pub const MAX_VOCAB_ENTRIES: usize = 4_194_304;

/// 語彙ファイルが許可された形式（トークン文字列から非負整数 ID への JSON オブジェクト）かを検証する
/// （REQ-39 形式の許可制・REQ-30・TASK-30.3・#125）。
///
/// # Errors
/// JSON として不正・オブジェクトでない・値が非負整数でない・空・エントリ過多の場合は
/// [`ArtifactMetaError::InvalidVocab`]。
pub fn validate_vocab_bytes(bytes: &[u8]) -> Result<(), ArtifactMetaError> {
    let map: std::collections::BTreeMap<String, u32> =
        serde_json::from_slice(bytes).map_err(|_| ArtifactMetaError::InvalidVocab)?;
    if map.is_empty() || map.len() > MAX_VOCAB_ENTRIES {
        return Err(ArtifactMetaError::InvalidVocab);
    }
    Ok(())
}

impl ArtifactMeta {
    /// `artifact.json` のバイト列を検証つきで読む。
    ///
    /// # Errors
    /// 不正な JSON・欠落・型違い・`onnx_file` の空・NUL・`kind` の空・過長・`max_bytes` が 0・
    /// `label_order` が空・過多・重複・空要素・`onnx_sha256` が小文字 16 進 64 桁でない場合。
    pub fn parse(bytes: &[u8]) -> Result<Self, ArtifactMetaError> {
        let value: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|_| ArtifactMetaError::Malformed)?;
        if !value.is_object() {
            return Err(ArtifactMetaError::Malformed);
        }
        let raw: RawMeta =
            serde_json::from_value(value).map_err(|_| ArtifactMetaError::Malformed)?;
        if raw.onnx_file.is_empty() || raw.onnx_file.contains('\0') {
            return Err(ArtifactMetaError::InvalidOnnxFile);
        }
        let mut seen = std::collections::BTreeSet::new();
        let labels_ok = !raw.label_order.is_empty()
            && raw.label_order.len() <= MAX_META_LABELS
            && raw
                .label_order
                .iter()
                .all(|l| !l.is_empty() && seen.insert(l.as_str()));
        let sha_ok = is_hex64(&raw.onnx_sha256) && raw.vocab_sha256.as_deref().is_none_or(is_hex64);
        if raw.kind.is_empty()
            || raw.kind.len() > MAX_META_KIND_BYTES
            || raw.max_bytes == 0
            || !labels_ok
            || !sha_ok
        {
            return Err(ArtifactMetaError::Malformed);
        }
        Ok(Self {
            onnx_file: raw.onnx_file,
            kind: raw.kind,
            kind_version: raw.kind_version,
            max_bytes: raw.max_bytes,
            label_order: raw.label_order,
            onnx_sha256: raw.onnx_sha256,
            vocab_sha256: raw.vocab_sha256,
        })
    }

    /// 未検証の `onnx_file` 文字列（経路検証の入力にのみ使う）。
    #[must_use]
    pub fn onnx_file(&self) -> &str {
        &self.onnx_file
    }

    /// 種類 ID（`c1`・`c3` 等）。
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// 種類の版（許可リスト検証は #174 で未実装）。
    #[must_use]
    pub const fn kind_version(&self) -> u32 {
        self.kind_version
    }

    /// 前処理の最大バイト長（範囲検証は推論ランタイム側）。
    #[must_use]
    pub const fn max_bytes(&self) -> u32 {
        self.max_bytes
    }

    /// 出力ラベルの並び（定義の選択肢の宣言順と一致する必要がある）。
    #[must_use]
    pub fn label_order(&self) -> &[String] {
        &self.label_order
    }

    /// 記載された ONNX の sha256（小文字 16 進 64 桁）。
    #[must_use]
    pub fn onnx_sha256(&self) -> &str {
        &self.onnx_sha256
    }

    /// 記載された語彙ファイルの sha256（小文字 16 進 64 桁）。語彙ファイルを持たない成果物では `None`。
    /// 語彙ファイルがある場合は記録が必須で、呼び出し側（`package`・`infer`）が照合する（REQ-39）。
    #[must_use]
    pub fn vocab_sha256(&self) -> Option<&str> {
        self.vocab_sha256.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39: 正常系。未知フィールドは無視する。
    #[test]
    fn req39_parse_extracts_onnx_file_and_ignores_unknown() {
        let r = ArtifactOnnxRef::parse(
            br#"{"onnx_file":"model.onnx","kind":"c3","kind_version":1,"n":1}"#,
        )
        .expect("ok");
        assert_eq!(r.onnx_file(), "model.onnx");
        assert_eq!(r.kind(), "c3");
    }

    /// REQ-39: 欠落・型違い・不正 JSON・配列は Malformed。
    #[test]
    fn req39_parse_rejects_malformed() {
        for b in [
            &br#"{}"#[..],
            br#"{"onnx_file":1,"kind":"c3","kind_version":1}"#,
            br#"{"onnx_file":null,"kind":"c3","kind_version":1}"#,
            br#"{"onnx_file":"model.onnx"}"#,
            br#"{"onnx_file":"model.onnx","kind":1,"kind_version":1}"#,
            br#"{"onnx_file":"model.onnx","kind":null,"kind_version":1}"#,
            br#"{"onnx_file":"model.onnx","kind":[],"kind_version":1}"#,
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
            ArtifactOnnxRef::parse(br#"{"onnx_file":"","kind":"c3","kind_version":1}"#),
            Err(ArtifactMetaError::InvalidOnnxFile)
        );
        assert_eq!(
            ArtifactOnnxRef::parse(br#"{"onnx_file":"a\u0000b","kind":"c3","kind_version":1}"#),
            Err(ArtifactMetaError::InvalidOnnxFile)
        );
    }

    /// REQ-39: `kind` の内容判定は core では行わない（空・構文違反でも Ok。ガード層の責務）。
    #[test]
    fn req39_parse_does_not_judge_kind_content() {
        for k in ["", "c3; rm -rf ~", "PT"] {
            let json = format!(r#"{{"onnx_file":"model.onnx","kind":"{k}","kind_version":1}}"#);
            let r = ArtifactOnnxRef::parse(json.as_bytes()).expect("ok");
            assert_eq!(r.kind(), k);
        }
    }

    /// REQ-39・TASK-39.6-1: `kind_version` は必須の `u32`。値の内容（99 など）は判定せず読める。
    #[test]
    fn req39_parse_reads_kind_version_without_judging() {
        for (lit, v) in [("1", 1u32), ("99", 99), ("0", 0), ("4294967295", u32::MAX)] {
            let json = format!(r#"{{"onnx_file":"m.onnx","kind":"c3","kind_version":{lit}}}"#);
            let r = ArtifactOnnxRef::parse(json.as_bytes()).expect("ok");
            assert_eq!(r.kind_version(), v);
        }
    }

    /// REQ-39・TASK-39.6-1: `kind_version` の型違い・範囲外は Malformed（fail-closed）。
    #[test]
    fn req39_parse_rejects_missing_or_invalid_kind_version() {
        for lit in [r#""1""#, "-1", "1.5", "null", "[]", "true", "4294967296"] {
            let json = format!(r#"{{"onnx_file":"m.onnx","kind":"c3","kind_version":{lit}}}"#);
            assert_eq!(
                ArtifactOnnxRef::parse(json.as_bytes()),
                Err(ArtifactMetaError::Malformed),
                "{lit}"
            );
        }
    }

    /// REQ-39・#174: `kind_version` を持たない従来の artifact.json は版 1 として読める（後方互換）。
    #[test]
    fn req39_parse_defaults_missing_kind_version_to_legacy() {
        let r = ArtifactOnnxRef::parse(br#"{"onnx_file":"m.onnx","kind":"c3"}"#).expect("ok");
        assert_eq!(r.kind_version(), LEGACY_KIND_VERSION);
        assert_eq!(r.kind_version(), 1);
    }

    /// REQ-39: エラー文言に入力値を含めない。
    #[test]
    fn req39_error_display_has_no_input_value() {
        let e = ArtifactOnnxRef::parse(
            br#"{"onnx_file":1,"kind":"c3","kind_version":1,"secret":"../../etc"}"#,
        )
        .unwrap_err();
        assert!(!e.to_string().contains("etc"));
    }

    fn full_meta(extra: &str) -> String {
        format!(
            r#"{{"onnx_file":"model.onnx","kind":"c1","kind_version":1,"max_bytes":48,"label_order":["a","b"],"onnx_sha256":"{}"{extra}}}"#,
            "0".repeat(64)
        )
    }

    /// REQ-39: 拡張メタの正常系。未知フィールドは無視する。
    #[test]
    fn req39_meta_parse_reads_fields() {
        let m = ArtifactMeta::parse(full_meta(r#","n":1"#).as_bytes()).expect("ok");
        assert_eq!(m.onnx_file(), "model.onnx");
        assert_eq!(m.kind(), "c1");
        assert_eq!(m.kind_version(), 1);
        assert_eq!(m.max_bytes(), 48);
        assert_eq!(m.label_order(), ["a".to_string(), "b".to_string()]);
        assert_eq!(m.onnx_sha256(), "0".repeat(64));
        assert_eq!(m.vocab_sha256(), None);
    }

    /// REQ-39: 語彙ファイルの sha256 は任意だが、あれば小文字 16 進 64 桁でなければならない。
    #[test]
    fn req39_meta_vocab_sha256_is_optional_and_validated() {
        let ok = format!(r#","vocab_sha256":"{}""#, "a".repeat(64));
        let m = ArtifactMeta::parse(full_meta(&ok).as_bytes()).expect("ok");
        assert_eq!(m.vocab_sha256(), Some("a".repeat(64).as_str()));
        assert_eq!(
            ArtifactMeta::parse(full_meta(r#","vocab_sha256":"zz""#).as_bytes()),
            Err(ArtifactMetaError::Malformed)
        );
    }

    /// REQ-39: 語彙ファイルは「トークン -> 非負整数 ID」の JSON オブジェクトだけを許可する。
    #[test]
    fn req39_validate_vocab_bytes_allows_only_token_id_map() {
        assert_eq!(validate_vocab_bytes(br#"{"a":0,"b":1}"#), Ok(()));
        for bad in [
            &b"{}"[..],
            b"[]",
            b"not json",
            br#"{"a":-1}"#,
            br#"{"a":"x"}"#,
            br#"{"a":1.5}"#,
            b"\x80\x81",
        ] {
            assert_eq!(
                validate_vocab_bytes(bad),
                Err(ArtifactMetaError::InvalidVocab)
            );
        }
    }

    /// REQ-39: 欠落・型違い・不正値は fail-closed。
    #[test]
    fn req39_meta_parse_rejects_invalid() {
        let sha = "0".repeat(64);
        let bad: Vec<String> = vec![
            "{}".to_string(),
            "[]".to_string(),
            "not json".to_string(),
            format!(
                r#"{{"onnx_file":"m.onnx","kind":"c1","kind_version":1,"max_bytes":0,"label_order":["a"],"onnx_sha256":"{sha}"}}"#
            ),
            format!(
                r#"{{"onnx_file":"m.onnx","kind":"c1","kind_version":1,"max_bytes":4,"label_order":[],"onnx_sha256":"{sha}"}}"#
            ),
            format!(
                r#"{{"onnx_file":"m.onnx","kind":"c1","kind_version":1,"max_bytes":4,"label_order":["a","a"],"onnx_sha256":"{sha}"}}"#
            ),
            format!(
                r#"{{"onnx_file":"m.onnx","kind":"c1","kind_version":1,"max_bytes":4,"label_order":["a"],"onnx_sha256":"{}"}}"#,
                "A".repeat(64)
            ),
            format!(
                r#"{{"onnx_file":"m.onnx","kind":"","kind_version":1,"max_bytes":4,"label_order":["a"],"onnx_sha256":"{sha}"}}"#
            ),
        ];
        for b in &bad {
            assert_eq!(
                ArtifactMeta::parse(b.as_bytes()),
                Err(ArtifactMetaError::Malformed),
                "{b}"
            );
        }
        let empty_file = full_meta("").replace("model.onnx", "");
        assert_eq!(
            ArtifactMeta::parse(empty_file.as_bytes()),
            Err(ArtifactMetaError::InvalidOnnxFile)
        );
    }
}
