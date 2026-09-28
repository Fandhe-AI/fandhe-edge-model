//! 入力正規化（REQ-16・TASK-16.2-2）。
//!
//! [`crate::consistency::find_contradictions`]（矛盾検出）は、表記ゆれ（全角・
//! 半角の違い・空白の数や種類の違いなど）を無視して同一入力と判定する必要が
//! ある。本モジュールはその正規化規則を差し替え可能な trait として切り出し、
//! 既定実装（[`NfkcWhitespaceNormalizer`]）を提供する。[`crate::consistency::find_metadata_mixed`]
//! （メタデータ混入検出）は正規化を行わず、生の `input` に対する
//! `str::contains` のみで判定するため本モジュールには依存しない
//! （[`crate::consistency`] モジュール doc・`find_metadata_mixed` の
//! ドキュメンテーションコメント参照）。
//!
//! # 既定実装は学習ワーカーと同じ NFKC＋空白規則
//!
//! 既定の [`NfkcWhitespaceNormalizer`] は Unicode NFKC 正規化 → 前後の空白
//! 除去 → 内部の連続空白の半角スペース 1 つへの圧縮、の順で行う。これは
//! 学習ワーカー（Python）`trainer/src/fandhe_edge_trainer/encoding.py` の
//! `normalize_input`（`unicodedata.normalize("NFKC", text)` →
//! `" ".join(nfkc.split())`）と同じ順序・同じ規則であり、
//! `fixtures/preprocess/byte_encoding_vectors.json`（共有ゴールデンベクタ・
//! SSOT）の NFKC を伴うケース（`fullwidth_alnum_nfkc`・`ligature_fi` 等）を
//! 含めて両実装の一致を機械照合する（`tests/nfkc_cross_check.rs`）。
//! `unicode-normalization =0.1.22` は 2026-09-28 オーナー承認済み
//! （`.claude/rules/dependency-policy.md`「承認済みの依存」表。初回導入の
//! 承認記録は PR #224、0.1.22 への版変更の承認記録は PR #225）。
//!
//! # Unicode 版を学習ワーカーと揃える
//!
//! `unicode-normalization` 0.1.22（`UNICODE_VERSION == (15, 0, 0)`）と
//! Python 3.12 の `unicodedata`（UCD 15.0.0。`fixtures/preprocess/
//! byte_encoding_vectors.json` の `_meta.unicode_version` が記録する値）は
//! 同じ Unicode 版のデータテーブルを使う。当初導入した 0.1.25 は
//! Unicode 17.0.0 準拠で版がずれていたため（PR #188 Codex レビュー P1
//! 指摘）、0.1.22 へ固定し直した。版の一致は
//! `tests/nfkc_cross_check.rs`（Rust 側。`UNICODE_VERSION` を assert）と
//! `trainer/tests/test_encoding.py::test_fixture_unicode_version_matches_runtime`
//! （Python 側）で機械照合する。
//!
//! NFKC 以外の規則が必要な呼び出し側は、独自の [`InputNormalizer`] 実装を
//! 渡せる。

use std::borrow::Cow;

use unicode_normalization::UnicodeNormalization;

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

/// Unicode NFKC 正規化 → 前後の空白除去 → 内部の連続空白の半角スペース 1 つ
/// への圧縮を行う既定の正規化規則。
///
/// 学習ワーカー（Python）`encoding.py::normalize_input` と同じ順序・同じ
/// 規則にする（モジュール doc「既定実装は学習ワーカーと同じ NFKC＋空白規則」
/// 参照）。
///
/// 空白の判定は `char::is_whitespace()` に U+001C〜U+001F（情報分離子）を
/// 加えた集合とする（`char::is_whitespace` だけでは不足しており、
/// `fixtures/preprocess/byte_encoding_vectors.json` の
/// `info_separator_1c`〜`info_separator_1f` ベクタが期待する挙動と一致させる）。
/// ZWSP（U+200B）・BOM（U+FEFF）・NUL・U+0001 は空白として扱わない
/// （同ベクタの `zero_width_space_200b`・`bom_feff`・`null_byte`・`control_0001`
/// が期待する「空白扱いされず保持される」という挙動と一致させる）。
#[derive(Debug, Clone, Copy, Default)]
pub struct NfkcWhitespaceNormalizer;

impl NfkcWhitespaceNormalizer {
    /// `c` が「Python 互換の空白」として扱われるかを判定する。
    fn is_py_whitespace(c: char) -> bool {
        c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
    }
}

impl InputNormalizer for NfkcWhitespaceNormalizer {
    fn rule_id(&self) -> &'static str {
        "nfkc-whitespace-v1"
    }

    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Python の `" ".join(unicodedata.normalize("NFKC", text).split())` と
        // 同じ結果になるよう、NFKC 正規化した文字列を空白区切りのトークン列に
        // してから半角スペースで結合する。
        let nfkc: String = text.nfkc().collect();
        let tokens: Vec<&str> = nfkc
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

    /// `fixtures/preprocess/byte_encoding_vectors.json` の各ベクタに対し、
    /// 既定の正規化規則が同じ `normalized` を返すことを確認する（REQ-16。
    /// ベクタ名をテスト名・コメントに記す）。NFKC を伴うベクタの機械照合は
    /// `tests/nfkc_cross_check.rs` に集約する。
    fn normalize(text: &str) -> String {
        NfkcWhitespaceNormalizer.normalize(text).into_owned()
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
        // NBSP（U+00A0）は `char::is_whitespace()` が真を返すため空白として
        // 扱われる（NFKC は U+00A0 を半角スペースへ変換しないため、この
        // ケースでは空白判定側の効果が主）。
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

    /// NFKC を要するベクタ（全角英数字の半角化）を既定の正規化規則が
    /// 変換することを確認する（REQ-16。以前の暫定実装〔NFKC 非対応〕からの
    /// 変更点。`fullwidth_alnum_nfkc` ベクタと同じ入出力）。
    #[test]
    fn req16_vector_fullwidth_alnum_nfkc_applied() {
        assert_eq!(normalize("\u{ff21}\u{ff11}"), "A1");
    }

    /// NFKC を要するベクタ（合字の分解）を既定の正規化規則が変換することを
    /// 確認する（REQ-16。`ligature_fi` ベクタと同じ入出力）。
    #[test]
    fn req16_vector_ligature_fi_applied() {
        assert_eq!(normalize("\u{fb01}le"), "file");
    }

    #[test]
    fn req16_rule_id_is_nfkc_whitespace() {
        assert_eq!(NfkcWhitespaceNormalizer.rule_id(), "nfkc-whitespace-v1");
    }
}
