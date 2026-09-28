//! 入力正規化（REQ-16・TASK-16.2-2）。
//!
//! [`crate::consistency::find_contradictions`]（矛盾検出）は、表記ゆれ（空白の
//! 数・種類の違いなど）を無視して同一入力と判定する必要がある。本モジュールは
//! その正規化規則を差し替え可能な trait として切り出し、既定実装
//! ([`PyWhitespaceNormalizer`]) を提供する。[`crate::consistency::find_metadata_mixed`]
//! （メタデータ混入検出）は正規化を行わず、生の `input` に対する
//! `str::contains` のみで判定するため本モジュールには依存しない
//! （[`crate::consistency`] モジュール doc・`find_metadata_mixed` の
//! ドキュメンテーションコメント参照）。
//!
//! # 既定実装は NFKC を含まない
//!
//! 既定の [`PyWhitespaceNormalizer`] は前後の空白除去・内部の連続空白の圧縮
//! （Python の `str.split()` と同じ空白判定）のみを行い、Unicode 正規化
//! （NFKC）は行わない。`Ａ１` と `A1` のような全角・半角の揺れは統合されない。
//! これは PoC-16 由来の共通正規化（`fixtures/preprocess/byte_encoding_vectors.json`・
//! `trainer/.../encoding.py` が契約の正）が NFKC を含むのに対し、本 crate は
//! `unicode-normalization` の依存承認を得るまでの暫定実装であるため
//! （[dependency-policy](../../../.claude/rules/dependency-policy.md)）。
//! [`InputNormalizer::rule_id`] に `\"py-whitespace-v1/no-nfkc\"` のように
//! NFKC を含まないことが読み取れる名前を付け、検出レポートに埋め込むことで、
//! 出力が NFKC 済みであるかのように読める余地を残さない。
//!
//! NFKC が必要な呼び出し側は、独自の [`InputNormalizer`] 実装を渡せる。
//! `unicode-normalization` が承認された後は、共通の正規化（TASK-15.x・
//! 推論前処理）へ差し替える想定。

use std::borrow::Cow;

/// 入力文字列の正規化規則。
///
/// [`crate::consistency::find_contradictions`] の呼び出し側が、用途に応じた
/// 正規化規則を差し替えられるようにするための trait。
/// [`crate::consistency::find_metadata_mixed`] は正規化を行わないため、
/// この trait を受け取らない。
pub trait InputNormalizer {
    /// 正規化規則を識別する ID。検出レポートに埋め込まれ、
    /// どの規則で正規化した結果かを読み手が判別できるようにする。
    fn rule_id(&self) -> &'static str;

    /// `text` を正規化する。変更が不要な場合は `Cow::Borrowed` を返してよい。
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str>;
}

/// Python の `str.split()` と同じ空白判定で、前後の空白除去・内部の連続空白の
/// 半角スペース 1 つへの圧縮のみを行う正規化規則（NFKC は行わない）。
///
/// 空白の判定は `char::is_whitespace()` に U+001C〜U+001F（情報分離子）を
/// 加えた集合とする（`char::is_whitespace` だけでは不足しており、
/// `fixtures/preprocess/byte_encoding_vectors.json` の
/// `info_separator_1c`〜`info_separator_1f` ベクタが期待する挙動と一致させる）。
/// ZWSP（U+200B）・BOM（U+FEFF）・NUL・U+0001 は空白として扱わない
/// （同ベクタの `zero_width_space_200b`・`bom_feff`・`null_byte`・`control_0001`
/// が期待する「空白扱いされず保持される」という挙動と一致させる）。
#[derive(Debug, Clone, Copy, Default)]
pub struct PyWhitespaceNormalizer;

impl PyWhitespaceNormalizer {
    /// `c` が「Python 互換の空白」として扱われるかを判定する。
    fn is_py_whitespace(c: char) -> bool {
        c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
    }
}

impl InputNormalizer for PyWhitespaceNormalizer {
    fn rule_id(&self) -> &'static str {
        "py-whitespace-v1/no-nfkc"
    }

    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Python の `" ".join(text.split())` と同じ結果になるよう、
        // 空白区切りのトークン列を作ってから半角スペースで結合する。
        let tokens: Vec<&str> = text
            .split(Self::is_py_whitespace)
            .filter(|s| !s.is_empty())
            .collect();
        let joined = tokens.join(" ");
        // 変更が無かった場合（元から正規化済み・非空白のみ等）は借用のまま返し、
        // 不要なアロケーションを避ける。
        if joined == text {
            Cow::Borrowed(text)
        } else {
            Cow::Owned(joined)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `fixtures/preprocess/byte_encoding_vectors.json` の各ベクタ（NFKC を
    /// 伴わないもの）に対し、既定の正規化規則が同じ `normalized` を返すことを
    /// 確認する（REQ-16。ベクタ名をテスト名・コメントに記す）。
    fn normalize(text: &str) -> String {
        PyWhitespaceNormalizer.normalize(text).into_owned()
    }

    #[test]
    fn req16_vector_basic_trim_and_compress() {
        assert_eq!(normalize("  hello   world  "), "hello world");
    }

    #[test]
    fn req16_vector_tabs_and_newlines() {
        assert_eq!(normalize("a\tb\nc"), "a b c");
    }

    #[test]
    fn req16_vector_empty_input() {
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn req16_vector_ascii_basic() {
        assert_eq!(normalize("ab"), "ab");
    }

    #[test]
    fn req16_vector_multibyte_japanese() {
        assert_eq!(normalize("あ"), "あ");
    }

    #[test]
    fn req16_vector_info_separator_1c() {
        assert_eq!(normalize("a\u{1c}b"), "a b");
    }

    #[test]
    fn req16_vector_info_separator_1d() {
        assert_eq!(normalize("a\u{1d}b"), "a b");
    }

    #[test]
    fn req16_vector_info_separator_1e() {
        assert_eq!(normalize("a\u{1e}b"), "a b");
    }

    #[test]
    fn req16_vector_info_separator_1f() {
        assert_eq!(normalize("a\u{1f}b"), "a b");
    }

    #[test]
    fn req16_vector_next_line_0085() {
        assert_eq!(normalize("a\u{0085}b"), "a b");
    }

    #[test]
    fn req16_vector_line_separator_2028() {
        assert_eq!(normalize("a\u{2028}b"), "a b");
    }

    #[test]
    fn req16_vector_ogham_space_1680() {
        assert_eq!(normalize("a\u{1680}b"), "a b");
    }

    #[test]
    fn req16_vector_nbsp_00a0() {
        // NBSP（U+00A0）は `char::is_whitespace()` が真を返すため、
        // NFKC を経由せずとも本規則で空白として扱われる。
        assert_eq!(normalize("a\u{00a0}b"), "a b");
    }

    #[test]
    fn req16_vector_ideographic_space_3000() {
        assert_eq!(normalize("a\u{3000}b"), "a b");
    }

    #[test]
    fn req16_vector_zero_width_space_200b_not_whitespace() {
        // ZWSP は空白として扱わない（ベクタの期待値どおり保持される）。
        assert_eq!(normalize("a\u{200b}b"), "a\u{200b}b");
    }

    #[test]
    fn req16_vector_bom_feff_not_whitespace() {
        assert_eq!(normalize("a\u{feff}b"), "a\u{feff}b");
    }

    #[test]
    fn req16_vector_null_byte_not_whitespace() {
        assert_eq!(normalize("a\u{0000}b"), "a\u{0000}b");
    }

    #[test]
    fn req16_vector_control_0001_not_whitespace() {
        assert_eq!(normalize("a\u{0001}b"), "a\u{0001}b");
    }

    #[test]
    fn req16_vector_whitespace_only_spaces_tabs_newlines() {
        assert_eq!(normalize(" \t\n"), "");
    }

    #[test]
    fn req16_vector_whitespace_only_info_separators() {
        assert_eq!(normalize("\u{1c}\u{1d}"), "");
    }

    /// NFKC を要するベクタ（`fullwidth_alnum_nfkc`・`ligature_fi`・
    /// `compat_hangul_parenthesized` 等）は、既定の正規化では変換されない
    /// ことを明示する（未対応であることをテストで示し、実装済みを装わない）。
    #[test]
    fn req16_vector_fullwidth_alnum_nfkc_not_applied_by_default() {
        assert_eq!(normalize("\u{ff21}\u{ff11}"), "\u{ff21}\u{ff11}");
        assert_ne!(normalize("\u{ff21}\u{ff11}"), "A1");
    }

    #[test]
    fn req16_vector_ligature_fi_not_applied_by_default() {
        assert_eq!(normalize("\u{fb01}le"), "\u{fb01}le");
        assert_ne!(normalize("\u{fb01}le"), "file");
    }

    #[test]
    fn req16_rule_id_is_no_nfkc() {
        assert_eq!(PyWhitespaceNormalizer.rule_id(), "py-whitespace-v1/no-nfkc");
    }
}
