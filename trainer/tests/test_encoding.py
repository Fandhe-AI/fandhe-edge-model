"""バイトエンコードのテスト（Rust 側 preprocess.rs との byte-identical 契約）。

Rust 側テストベクタ（`docs/spec/03-poc/core-cli-vertical-slice/core/src/preprocess.rs`
の `#[cfg(test)]` 群）と同じ入出力を Python 側で確認する。REQ-28 の推論一致契約は
学習・推論の両方が同じトークン化をすることを前提とするため、この一致は不可欠。
"""

from __future__ import annotations

from fandhe_edge_trainer.encoding import encode_bytes, normalize_input


def test_normalize_basic() -> None:
    assert normalize_input("  hello   world  ") == "hello world"


def test_normalize_nfkc_fullwidth() -> None:
    # 全角英数字は NFKC で半角化される（Rust 側 normalize_nfkc_fullwidth と同値）。
    assert normalize_input("Ａ１") == "A1"


def test_normalize_internal_tabs_newlines() -> None:
    assert normalize_input("a\tb\nc") == "a b c"


def test_normalize_information_separator() -> None:
    # U+001C（情報分離子）は Python の str.split() が空白として扱う文字であり、
    # Rust 側 is_py_whitespace も明示的に対象へ含めている。
    assert normalize_input("a\x1cb") == "a b"


def test_encode_bytes_truncates() -> None:
    s = "a" * 600
    v = encode_bytes(s, 512)
    assert len(v) == 512
    assert all(x == ord("a") + 1 for x in v)


def test_encode_bytes_empty_gives_pad_token() -> None:
    assert encode_bytes("", 512) == [0]


def test_encode_bytes_multibyte_japanese() -> None:
    # "あ" は UTF-8 で E3 81 82 の 3 バイト（Rust 側 encode_bytes_multibyte と同値）。
    assert encode_bytes("あ", 512) == [0xE3 + 1, 0x81 + 1, 0x82 + 1]


def test_encode_bytes_ascii() -> None:
    assert encode_bytes("ab", 512) == [ord("a") + 1, ord("b") + 1]


def test_encode_bytes_applies_normalization_before_encoding() -> None:
    # 全角英数字が NFKC 正規化されてから UTF-8 化されることを確認する。
    assert encode_bytes("Ａ", 512) == [ord("A") + 1]
