//! Qwen2 の正規化（NFC）と事前分割（Split Isolated）（REQ-41・REQ-39・TASK-41.4-3・#411。親 #408）。
//!
//! # 役割と境界
//!
//! BPE（#412）の前段。[`super::TokenizerSpec::pre_tokenize`] から呼ばれ、入力本文を piece 列へ
//! 分ける。ByteLevel のバイト写像・BPE・特殊トークンはここでは扱わない（#412）。
//!
//! 正規表現エンジンは持たず（#409 のオーナー判断。依存を増やさない）、`SPLIT_REGEX` の 7 択を
//! 手書きの状態機械で写す。参照実装は `trainer/tools/poc26/qwen2_tokenizer.py` の `pre_tokenize`
//! （Python `re`）で、結果の一致は生成 golden（`fixtures/poc26/pretokenize_golden.json`。証拠種別:
//! テストハーネス）で固定する。後戻り（backtracking）を使わないので ReDoS は起きず、各位置の
//! 走査は空白・文字の連続に沿った定数回で、全体は入力長に線形。
//!
//! # 既知の差（許容。#409 のオーナー判断）
//!
//! - `(?i:...)` の大文字小文字の畳み込みは ASCII と `ſ`（U+017F）→ `s` のみ。Python `re` と同じで、
//!   HF（Oniguruma）とは食い違いうる（学習側との一致が基準）
//! - `\p{L}`・`\p{N}` は Unicode 15.0 固定（`unicode_tables.rs`）。HF 側の版と違う文字では
//!   分割が食い違いうる
//!
//! # 資源上限（REQ-39）
//!
//! 入力は [`MAX_ENCODE_BYTES`] 以下。NFC の前に確かめ、NFC で伸びうるので後にも確かめる。
//! 上限ちょうどの入力は前後とも通るので、学習側との一致は崩れない。

use std::fmt;

use unicode_normalization::UnicodeNormalization;

use super::unicode_tables::{LETTER_RANGES, NUMBER_RANGES, WHITE_SPACE};

/// 1 回の符号化で受け付ける入力のバイト数上限（参照実装の `MAX_ENCODE_BYTES` と同値）。
pub const MAX_ENCODE_BYTES: usize = 262_144;

/// 事前分割の失敗。入力本文は保持しない（security.md）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TokenizeError {
    /// 入力が [`MAX_ENCODE_BYTES`] を超える（NFC の前または後。REQ-39）。
    InputTooLong,
}

impl TokenizeError {
    /// 機械可読なエラーコード（英語）。
    pub fn code(&self) -> &'static str {
        match self {
            Self::InputTooLong => "limit_exceeded",
        }
    }
}

impl fmt::Display for TokenizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLong => f.write_str("input text is too long"),
        }
    }
}

impl std::error::Error for TokenizeError {}

/// `nfc` が true なら NFC 正規化してから `split` した piece 列を返す。
pub fn pre_tokenize(text: &str, nfc: bool) -> Result<Vec<String>, TokenizeError> {
    if text.len() > MAX_ENCODE_BYTES {
        return Err(TokenizeError::InputTooLong);
    }
    if !nfc {
        return Ok(split(text).into_iter().map(str::to_owned).collect());
    }
    let normalized: String = text.nfc().collect();
    if normalized.len() > MAX_ENCODE_BYTES {
        return Err(TokenizeError::InputTooLong);
    }
    Ok(split(&normalized).into_iter().map(str::to_owned).collect())
}

fn in_ranges(ranges: &[(u32, u32)], c: char) -> bool {
    let cp = u32::from(c);
    ranges
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                std::cmp::Ordering::Greater
            } else if cp > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

fn is_l(c: char) -> bool {
    in_ranges(LETTER_RANGES, c)
}
fn is_n(c: char) -> bool {
    in_ranges(NUMBER_RANGES, c)
}
fn is_ws(c: char) -> bool {
    WHITE_SPACE.binary_search(&u32::from(c)).is_ok()
}
fn is_crlf(c: char) -> bool {
    c == '\r' || c == '\n'
}
/// 選択肢 4 の `[^\s\p{L}\p{N}]`。
fn is_p(c: char) -> bool {
    !is_ws(c) && !is_l(c) && !is_n(c)
}

/// 正規化済みの `text` を Qwen2 の Split(Isolated) で分ける。piece を連結すると `text` に戻る。
fn split(text: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let byte_at = |k: usize| cs.get(k).map_or(text.len(), |&(b, _)| b);
    let mut pieces = Vec::new();
    let (mut i, mut gap) = (0, None::<usize>);
    while i < cs.len() {
        match match_at(&cs, i) {
            Some(len) if len > 0 => {
                if let Some(g) = gap.take() {
                    pieces.push(&text[byte_at(g)..byte_at(i)]);
                }
                pieces.push(&text[byte_at(i)..byte_at(i + len)]);
                i += len;
            }
            _ => {
                // どの選択肢にも当たらない文字は、次の一致までまとめて 1 piece（Isolated）
                gap.get_or_insert(i);
                i += 1;
            }
        }
    }
    if let Some(g) = gap {
        pieces.push(&text[byte_at(g)..]);
    }
    pieces
}

/// 位置 `i` から始まる最初に成立する選択肢の文字数。
fn match_at(cs: &[(usize, char)], i: usize) -> Option<usize> {
    let c = |k: usize| cs.get(k).map(|&(_, ch)| ch);
    // 位置 k から pred を満たす文字が続く末尾（排他的）の位置
    let run = |k: usize, pred: fn(char) -> bool| {
        let mut e = k;
        while c(e).is_some_and(pred) {
            e += 1;
        }
        e
    };
    let ci = c(i)?;

    // 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if ci == '\'' {
        let fold = |k: usize| {
            c(k).map(|ch| {
                if ch == '\u{17F}' {
                    's'
                } else {
                    ch.to_ascii_lowercase()
                }
            })
        };
        match (fold(i + 1), fold(i + 2)) {
            (Some('s' | 't' | 'm' | 'd'), _) => return Some(2),
            (Some('r' | 'v'), Some('e')) | (Some('l'), Some('l')) => return Some(3),
            _ => {}
        }
    }
    // 2: [^\r\n\p{L}\p{N}]?\p{L}+
    if is_l(ci) {
        return Some(run(i, is_l) - i);
    }
    if !is_crlf(ci) && !is_n(ci) && c(i + 1).is_some_and(is_l) {
        return Some(run(i + 1, is_l) - i);
    }
    // 3: \p{N}
    if is_n(ci) {
        return Some(1);
    }
    // 4: ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    let start = if ci == ' ' && c(i + 1).is_some_and(is_p) {
        i + 1
    } else {
        i
    };
    let end = run(start, is_p);
    if end > start {
        return Some(run(end, is_crlf) - i);
    }
    // 5〜7: 空白で始まる 3 択
    let e = run(i, is_ws);
    if e == i {
        return None;
    }
    // 5: \s*[\r\n]+ （空白の連続のうち最後の改行まで）
    if let Some(k) = (i..e).rev().find(|&k| c(k).is_some_and(is_crlf)) {
        return Some(k + 1 - i);
    }
    // 6: \s+(?!\S) （直後が非空白なら 1 文字残す）
    if e == cs.len() {
        return Some(e - i);
    }
    if e - 1 > i {
        return Some(e - 1 - i);
    }
    // 7: \s+
    Some(e - i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(text: &str) -> Vec<&str> {
        split(text)
    }

    /// REQ-41: 参照実装の pytest（test_pre_tokenize_pieces）と同じ 20 件。
    #[test]
    fn matches_reference_cases() {
        let cases: &[(&str, &[&str])] = &[
            ("I'm", &["I", "'m"]),
            ("DON'T", &["DON", "'T"]),
            ("a, b", &["a", ",", " b"]),
            ("hi!!\n", &["hi", "!!\n"]),
            ("x  y", &["x", " ", " y"]),
            ("a\n\nb", &["a", "\n\n", "b"]),
            ("a \nb", &["a", " \n", "b"]),
            ("2024", &["2", "0", "2", "4"]),
            ("年1月", &["年", "1", "月"]),
            ("x$5", &["x", "$", "5"]),
            ("ab-cd", &["ab", "-cd"]),
            ("a\u{1c}b", &["a", "\u{1c}b"]),
            ("x\u{3000}\u{3000}y", &["x", "\u{3000}", "\u{3000}y"]),
            ("x\u{a0}\u{a0}y", &["x", "\u{a0}", "\u{a0}y"]),
            ("x\u{85}\u{85}y", &["x", "\u{85}", "\u{85}y"]),
            ("x  ", &["x", "  "]),
            ("a\r\nb", &["a", "\r\n", "b"]),
            ("'S", &["'S"]),
            ("I'LL", &["I", "'LL"]),
            ("x\u{b2}", &["x", "\u{b2}"]),
        ];
        for (text, want) in cases {
            assert_eq!(&s(text), want, "{text:?}");
        }
    }

    #[test]
    fn tables_are_sorted_and_disjoint() {
        for t in [LETTER_RANGES, NUMBER_RANGES] {
            assert!(t.iter().all(|&(lo, hi)| lo <= hi));
            assert!(t.windows(2).all(|w| w[0].1 < w[1].0));
        }
        assert!(WHITE_SPACE.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn pieces_concatenate_to_input() {
        let text = "  Hello, 世界!! 123\r\n\r\n  x'sY \u{3000} é\u{301}";
        assert_eq!(s(text).concat(), text);
    }

    #[test]
    fn size_limit_boundary() {
        let ok = "a".repeat(MAX_ENCODE_BYTES);
        assert_eq!(pre_tokenize(&ok, true).map(|v| v.len()), Ok(1));
        let over = "a".repeat(MAX_ENCODE_BYTES + 1);
        assert_eq!(pre_tokenize(&over, true), Err(TokenizeError::InputTooLong));
        assert_eq!(pre_tokenize(&over, false), Err(TokenizeError::InputTooLong));
    }

    #[test]
    fn nfc_only_when_requested() {
        assert_eq!(
            pre_tokenize("e\u{301}", true),
            Ok(vec!["\u{e9}".to_owned()])
        );
        assert_eq!(
            pre_tokenize("e\u{301}", false),
            Ok(vec!["e".to_owned(), "\u{301}".to_owned()])
        );
    }

    /// REQ-39: 後戻りが無く、長い空白の連続でも線形時間で終わる。
    #[test]
    fn long_whitespace_run_is_linear() {
        let n = MAX_ENCODE_BYTES - 1;
        let text = format!("{}x", " ".repeat(n));
        let got = s(&text);
        assert_eq!(got.len(), 2);
        assert_eq!(got.first().map(|p| p.len()), Some(n - 1));
        assert_eq!(got.get(1), Some(&" x"));
    }
}
