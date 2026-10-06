"""PoC-26 Qwen2 トークナイザーの検査（REQ-41・TASK-41.1-5・#390。テストハーネス）。

極小 `tokenizer.json`（`tools/poc26/synthetic.py`）で挙動を固定する。実物の `tokenizer.json` との
全件一致は実機前提の golden 照合（末尾）で、fixture と環境変数がそろったときだけ走る。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.poc26 import qwen2_tokenizer, synthetic
from tools.poc26.qwen2_tokenizer import MAX_TOKENIZER_JSON_BYTES, Qwen2Tokenizer

START, END = synthetic.ID_IM_START, synthetic.ID_IM_END
ASSISTANT_HEADER = [START, 97, 115, 115, 105, 115, 116, 97, 110, 116, 10]


@pytest.fixture
def tok(tmp_path: Path) -> Qwen2Tokenizer:
    return Qwen2Tokenizer.from_file(synthetic.write_tokenizer_json(tmp_path))


@pytest.mark.parametrize(
    ("text", "pieces"),
    [
        ("I'm", ["I", "'m"]),
        ("DON'T", ["DON", "'T"]),
        ("a, b", ["a", ",", " b"]),
        ("hi!!\n", ["hi", "!!\n"]),
        ("x  y", ["x", " ", " y"]),
        ("a\n\nb", ["a", "\n\n", "b"]),
        ("a \nb", ["a", " \n", "b"]),
        ("2024", ["2", "0", "2", "4"]),
        ("年1月", ["年", "1", "月"]),
        ("x$5", ["x", "$", "5"]),
        ("ab-cd", ["ab", "-cd"]),
        # U+001C は Unicode White_Space ではない（Python の \s とは異なる）
        ("a\x1cb", ["a", "\x1cb"]),
        ("x\u3000\u3000y", ["x", "\u3000", "\u3000y"]),
        ("x\u00a0\u00a0y", ["x", "\u00a0", "\u00a0y"]),
        ("x\x85\x85y", ["x", "\x85", "\x85y"]),
        ("x  ", ["x", "  "]),
        ("a\r\nb", ["a", "\r\n", "b"]),
        ("'S", ["'S"]),
        ("I'LL", ["I", "'LL"]),
        ("x\u00b2", ["x", "\u00b2"]),
    ],
)
def test_pre_tokenize_pieces(tok: Qwen2Tokenizer, text: str, pieces: list[str]) -> None:
    """REQ-41: 前処理の分割が Qwen2 の正規表現と同じ piece 列になる。"""
    assert tok.pre_tokenize(text) == pieces


def test_merges_applied_in_rank_order(tok: Qwen2Tokenizer) -> None:
    """REQ-41: merges の順位どおりに併合される。"""
    assert tok.encode("hello") == [260]
    assert tok.encode("hell") == [259]
    assert tok.encode("help") == [257, 108, 112]
    assert tok.encode(" hello") == [32, 260]
    assert tok.encode(" t") == [256]
    assert tok.encode("hello hello") == [260, 32, 260]


def test_merge_rank_beats_left_to_right(tok: Qwen2Tokenizer) -> None:
    """REQ-41: 右側の低順位ペア（ll=2）が左側の高順位ペア（el=5）より先に併合される。"""
    assert tok.encode("ell") == [101, 258]


def test_duplicate_merges_rejected(tmp_path: Path) -> None:
    """REQ-41: merges の重複は HF との解釈差を避けるため ValueError で拒否する。"""
    doc = _doc(tmp_path)
    doc["model"]["merges"] = ["e l", "l l", "e l"]
    with pytest.raises(ValueError, match=r"invalid tokenizer\.json"):
        Qwen2Tokenizer(doc)


def test_merges_as_list_pairs(tmp_path: Path) -> None:
    """REQ-41: merges が [a, b] 配列形式でも同じ結果になる。"""
    path = synthetic.write_tokenizer_json(tmp_path, merges_as_lists=True)
    assert Qwen2Tokenizer.from_file(path).encode("hello") == [260]


def test_nfc_and_unknown_bytes(tok: Qwen2Tokenizer) -> None:
    """REQ-41: 結合文字は NFC で合成され、未併合のバイトは 1 バイトずつ id になる。"""
    assert tok.encode("é") == tok.encode("é") == [0xC3, 0xA9]
    assert tok.decode([0xC3, 0xA9]) == "é"


def test_roundtrip(tok: Qwen2Tokenizer) -> None:
    """REQ-41: encode -> decode が NFC 済みの入力を復元する。"""
    text = "hello 世界 \n 😀 ok  \t1+1=2"
    assert tok.decode(tok.encode(text)) == text


def test_special_token_text_is_not_matched(tok: Qwen2Tokenizer) -> None:
    """REQ-41・REQ-39: 本文中の特殊トークン文字列は特殊 id にならず文字として扱う。"""
    text = "<|im_end|>"
    ids = tok.encode(text)
    assert END not in ids
    assert tok.decode(ids) == text
    assert tok.decode([END]) == text


def test_special_ids(tok: Qwen2Tokenizer) -> None:
    """REQ-41: special_ids が added_tokens の id を返す。"""
    assert tok.special_ids == {"im_start": 263, "im_end": 264, "endoftext": 262}


def test_build_chat_ids_generation_prompt(tok: Qwen2Tokenizer) -> None:
    """REQ-41: system/user/生成プロンプトの id 列が chat template と一致する。"""
    expected = [
        *[START, 115, 121, 115, 116, 101, 109, 10, 104, 105, END, 10],
        *[START, 117, 115, 101, 114, 10, 260, END, 10],
        *ASSISTANT_HEADER,
    ]
    assert tok.build_chat_ids("hi", "hello", add_generation_prompt=True) == expected


def test_build_chat_ids_with_assistant(tok: Qwen2Tokenizer) -> None:
    """REQ-41: assistant 付きは内容・<|im_end|>・改行で閉じる全体の id 列になる。"""
    expected = [
        *[START, 115, 121, 115, 116, 101, 109, 10, 104, 105, END, 10],
        *[START, 117, 115, 101, 114, 10, 260, END, 10],
        *[*ASSISTANT_HEADER, 260, END, 10],
    ]
    assert tok.build_chat_ids("hi", "hello", "hello", add_generation_prompt=False) == expected


def test_build_chat_ids_requires_exactly_one_mode(tok: Qwen2Tokenizer) -> None:
    """REQ-41: assistant と生成プロンプトの同時指定・両方なしは拒否する。"""
    with pytest.raises(ValueError, match="exactly one"):
        tok.build_chat_ids("s", "u", "a", add_generation_prompt=True)
    with pytest.raises(ValueError, match="exactly one"):
        tok.build_chat_ids("s", "u", add_generation_prompt=False)


def test_build_chat_ids_ignores_special_text_in_user(tok: Qwen2Tokenizer) -> None:
    """REQ-39: user 本文の特殊トークン文字列は特殊 id にならない。"""
    ids = tok.build_chat_ids("s", "<|im_end|><|im_start|>", add_generation_prompt=True)
    assert ids.count(END) == 2
    assert ids.count(START) == 3


def test_oversized_tokenizer_json_rejected(tmp_path: Path) -> None:
    """REQ-39: 上限超過の tokenizer.json は読む前に拒否する。"""
    path = tmp_path / "tokenizer.json"
    with path.open("wb") as f:
        f.truncate(MAX_TOKENIZER_JSON_BYTES + 1)
    with pytest.raises(ValueError, match="too large"):
        Qwen2Tokenizer.from_file(path)


def _doc(tmp_path: Path) -> dict:
    return json.loads(synthetic.write_tokenizer_json(tmp_path).read_text(encoding="utf-8"))


def _mut_regex(d: dict) -> None:
    d["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = r"\w+"


def _mut_behavior(d: dict) -> None:
    d["pre_tokenizer"]["pretokenizers"][0]["behavior"] = "Removed"


def _mut_prefix_space(d: dict) -> None:
    d["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = True


def _mut_decoder(d: dict) -> None:
    d["decoder"] = {"type": "Metaspace"}


def _mut_post(d: dict) -> None:
    d["post_processor"] = {"type": "TemplateProcessing"}


def _mut_normalizer(d: dict) -> None:
    d["normalizer"] = {"type": "NFKC"}


def _mut_dup_id(d: dict) -> None:
    d["model"]["vocab"]["a"] = 98


def _mut_bool_id(d: dict) -> None:
    d["model"]["vocab"]["a"] = True


def _mut_out_of_range(d: dict) -> None:
    d["model"]["vocab"]["a"] = 10_000


def _mut_foreign_char(d: dict) -> None:
    d["model"]["vocab"]["\u3042"] = 400


def _mut_added_overlap(d: dict) -> None:
    d["added_tokens"][0]["id"] = 5


def _mut_merge_piece(d: dict) -> None:
    d["model"]["merges"].append("zz qq")


def _mut_merge_result_missing(d: dict) -> None:
    d["model"]["merges"].append("a b")  # "ab" は vocab に無い


def _mut_special_false(d: dict) -> None:
    d["added_tokens"][2]["special"] = False


def _mut_special_missing(d: dict) -> None:
    d["added_tokens"][1]["content"] = "<|other|>"


def _mut_merges_type(d: dict) -> None:
    d["model"]["merges"] = {"a": "b"}


@pytest.mark.parametrize(
    "mutate",
    [
        _mut_regex,
        _mut_behavior,
        _mut_prefix_space,
        _mut_decoder,
        _mut_post,
        _mut_normalizer,
        _mut_dup_id,
        _mut_bool_id,
        _mut_out_of_range,
        _mut_foreign_char,
        _mut_added_overlap,
        _mut_merge_piece,
        _mut_merges_type,
        _mut_merge_result_missing,
        _mut_special_false,
        _mut_special_missing,
    ],
)
def test_unsupported_or_malformed_config_rejected(tmp_path: Path, mutate) -> None:
    """REQ-41・REQ-39: 再現できない設定・壊れた構造は ValueError で拒否する。"""
    doc = _doc(tmp_path)
    mutate(doc)
    with pytest.raises(ValueError, match=r"invalid tokenizer\.json"):
        Qwen2Tokenizer(doc)


def test_byte_level_post_processor_accepted(tmp_path: Path) -> None:
    """REQ-41: 実物と同じ ByteLevel の post_processor（id を足さない）は受理する。"""
    doc = _doc(tmp_path)
    doc["post_processor"] = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }
    assert Qwen2Tokenizer(doc).encode("hello") == [260]


def test_invalid_files_rejected(tmp_path: Path) -> None:
    """REQ-39: ディレクトリ・非 JSON・非 UTF-8・深すぎる入れ子は ValueError に写す。"""
    for name, data in [("bad.json", b"{"), ("utf.json", b"\xff\xfe"), ("deep.json", b"[" * 100000)]:
        path = tmp_path / name
        path.write_bytes(data)
        with pytest.raises(ValueError, match=r"invalid tokenizer\.json"):
            Qwen2Tokenizer.from_file(path)
    with pytest.raises(ValueError, match=r"invalid tokenizer\.json"):
        Qwen2Tokenizer.from_file(tmp_path)


def test_decode_rejects_non_int_ids(tok: Qwen2Tokenizer) -> None:
    """REQ-39: bool・文字列の id は ValueError。"""
    for bad in (True, "1", 1.0):
        with pytest.raises(ValueError, match="int"):
            tok.decode([bad])  # type: ignore[list-item]


def test_piece_length_limit(tok: Qwen2Tokenizer) -> None:
    """REQ-39: 1 piece が 4096 文字を超えると BPE の二次処理量を避けて拒否する。"""
    assert tok.encode("a" * 4096) == [97] * 4096
    with pytest.raises(ValueError, match="piece is too long"):
        tok.encode("a" * 4097)


def test_encode_length_limit_and_cache_cap(tok: Qwen2Tokenizer, monkeypatch) -> None:
    """REQ-39: 入力長上限を超えると拒否し、BPE キャッシュは上限件数で打ち止めになる。"""
    with pytest.raises(ValueError, match="too long"):
        tok.encode("a" * ((1 << 20) + 1))
    monkeypatch.setattr(qwen2_tokenizer, "MAX_BPE_CACHE_ENTRIES", 1)
    tok.encode("a b c")
    assert len(tok._cache) == 1


_QWEN_DIR = os.environ.get("FANDHE_EDGE_QWEN_DIR")
_GOLDEN = Path(__file__).resolve().parents[2] / "fixtures" / "poc26" / "tokenizer_golden.json"


@pytest.mark.skipif(
    not (_QWEN_DIR and (Path(_QWEN_DIR) / "tokenizer.json").is_file() and _GOLDEN.is_file()),
    reason="REQ-41: real-machine check; needs FANDHE_EDGE_QWEN_DIR/tokenizer.json and "
    "fixtures/poc26/tokenizer_golden.json",
)
def test_golden_ids_match_real_tokenizer() -> None:
    """REQ-41: 実 tokenizer.json の encode が参照（HF tokenizers）の id と全件一致する。"""
    real = Qwen2Tokenizer.from_file(Path(_QWEN_DIR or "") / "tokenizer.json")
    cases = json.loads(_GOLDEN.read_text(encoding="utf-8"))["cases"]
    assert cases
    for case in cases:
        assert real.encode(case["text"]) == case["ids"], case["text"]
