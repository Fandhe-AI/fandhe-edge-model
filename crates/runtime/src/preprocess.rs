//! バイト前処理（NFKC 正規化＋バイトエンコード。REQ-32・REQ-28・TASK-32.1-1・#112）。
//!
//! # 役割
//!
//! 学習ワーカー（Python `trainer/src/fandhe_edge_trainer/encoding.py` の `normalize_input`・
//! `encode_bytes`）と推論側で入力表現がずれると、学習時と推論時で別の入力を見ることになる。
//! 本モジュールは PoC-16（`03-poc/core-cli-vertical-slice/core/src/preprocess.rs`。PoC-14 から
//! 無変更のコピー）の C3（バイト CNN）用関数だけを移植した推論側の前処理で、
//! [`crate::pipeline::Preprocessor`] の実装 [`ByteEncodingPreprocessor`] として推論経路へ差し込まれる。
//! C1 の TF-IDF は ONNX グラフ内で計算するため移植しない。
//!
//! - 正規化: NFKC → 前後の空白除去 → 内部の連続空白を半角スペース 1 つへ圧縮。空白判定は
//!   Python の `str.split()` 互換で、`char::is_whitespace` に U+001C〜U+001F を加える
//! - エンコード: UTF-8 バイトに 1 を足した値（0 はパディング用に予約）を先頭 `max_bytes`
//!   バイトまで並べる。空なら `[0]`
//!
//! # 二重実装の理由と機械照合
//!
//! runtime は共通コアのみに依存し、データ契約層（`fandhe-edge-data`）の正規化器は使えない
//! （REQ-32。共通コアへの集約は `unicode-normalization` の配置承認が無いため見送り）。結果として
//! 同じ規則が Python・データ契約層・本モジュールの 3 箇所にあり、共有ゴールデンベクタ
//! `fixtures/preprocess/byte_encoding_vectors.json` が一致を保つ唯一の機械照合になる
//! （`tests/preprocess_golden.rs`）。NFKC の結果は Unicode の版に依存するため、
//! `unicode-normalization` は Python 3.12 と同じ Unicode 15.0.0 の版で固定している
//! （`.claude/rules/dependency-policy.md`）。
//!
//! # 契約上の注意
//!
//! - [`encode_bytes`] は **正規化済み** 文字列を受け取る（PoC の署名）。Python の `encode_bytes` は
//!   内部で正規化するため署名が異なる。fixture の `ids` は「正規化 → エンコード」の合成結果
//! - `max_bytes == 0` は `[0]` を返す（fixture `max_bytes_zero`）。実モデルの `max_bytes` の範囲
//!   検証は [`crate::onnx::load_pipeline`]（#113。学習ワーカーの `limits.py` と同値の範囲）が
//!   行い、本モジュールは学習ワーカー層の定数に依存しない
//! - 資源上限: 入力長は [`crate::pipeline`] が前処理前に検査し、出力長は `max_bytes` で有界。
//!   NFKC の膨張は正規化後の中間文字列に現れるが、入力上限で有界
//! - 状態を持たない（呼び出し間で結果が変わらない。REQ-28）。外部入力の経路のため
//!   `unwrap` / 添字アクセスを使わず panic しない

use crate::pipeline::{PreprocessError, Preprocessor, TokenIds};
use unicode_normalization::UnicodeNormalization;

/// Python の `str.split()` 互換の空白判定。`char::is_whitespace` に U+001C〜U+001F
/// （情報区切り文字。Python は空白扱い・Rust は非空白）を加える。
fn is_py_whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// 入力を NFKC 正規化し、前後の空白を除いて内部の連続空白を半角スペース 1 つへ圧縮する。
///
/// 学習ワーカーの `normalize_input` と同一の出力になること（共有ゴールデンベクタで照合。REQ-32）。
pub fn normalize_input(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    nfkc.split(is_py_whitespace)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// 正規化済み文字列を UTF-8 バイト + 1 の列へ変換する（先頭 `max_bytes` バイトで切り詰め、
/// 空なら `[0]`）。マルチバイト文字の途中でも切り詰める（Python 側と同じ）。
pub fn encode_bytes(normalized: &str, max_bytes: usize) -> Vec<i64> {
    let bytes = normalized.as_bytes();
    let mut ids: Vec<i64> = Vec::with_capacity(bytes.len().min(max_bytes));
    ids.extend(bytes.iter().take(max_bytes).map(|&b| i64::from(b) + 1));
    if ids.is_empty() {
        ids.push(0);
    }
    ids
}

/// 正規化とバイトエンコードを行う [`Preprocessor`] 実装。
///
/// 単体推論・バッチ推論の共通経路（[`crate::pipeline::InferencePipeline`]）から 1 件ずつ呼ばれる。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteEncodingPreprocessor {
    max_bytes: usize,
}

impl ByteEncodingPreprocessor {
    /// 切り詰めるバイト数の上限を指定して作る（範囲検証は呼び出し側。モジュール doc 参照）。
    pub fn new(max_bytes: usize) -> Self {
        Self { max_bytes }
    }

    /// 切り詰めるバイト数の上限。
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}

impl Preprocessor for ByteEncodingPreprocessor {
    /// `&str` を受けるため失敗経路は無く、常に `Ok` を返す。
    fn preprocess(&self, input: &str) -> Result<TokenIds, PreprocessError> {
        Ok(TokenIds::new(encode_bytes(
            &normalize_input(input),
            self.max_bytes,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-32: 情報区切り文字 U+001C〜U+001F のみ Rust 標準の空白判定へ加える。
    #[test]
    fn req32_py_whitespace_boundaries() {
        for c in [
            '\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', ' ', '\t', '\u{3000}',
        ] {
            assert!(is_py_whitespace(c), "{:?}", c);
        }
        for c in ['\u{200b}', '\u{feff}', '\u{0}', '\u{1}', 'a'] {
            assert!(!is_py_whitespace(c), "{:?}", c);
        }
    }

    /// REQ-32: `max_bytes` 0・空文字・途中切り詰め・上限超過の具体値。
    #[test]
    fn req32_encode_bytes_concrete_values() {
        assert_eq!(encode_bytes("abc", 0), vec![0]);
        assert_eq!(encode_bytes("", 512), vec![0]);
        assert_eq!(encode_bytes("\u{3042}", 2), vec![228, 130]);
        assert_eq!(encode_bytes("ab", 100), vec![98, 99]);
    }

    /// REQ-32・REQ-28: 前処理器は正規化→エンコードの合成で、繰り返し呼んでも同じ結果。
    #[test]
    fn req32_preprocessor_composes_and_is_stateless() {
        let p = ByteEncodingPreprocessor::new(512);
        assert_eq!(p.max_bytes(), 512);
        let first = p
            .preprocess("  \u{ff21}\u{ff11}  ")
            .map(|t| t.as_slice().to_vec());
        let second = p
            .preprocess("  \u{ff21}\u{ff11}  ")
            .map(|t| t.as_slice().to_vec());
        assert_eq!(first, Ok(vec![66, 50]));
        assert_eq!(first, second);
    }
}
