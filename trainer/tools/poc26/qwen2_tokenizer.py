"""Qwen2 系 `tokenizer.json`（byte-level BPE）の標準ライブラリのみの実装。

REQ-41・TASK-41.1-5・#390。PoC-26（Playwright MCP のツール選択）の学習スクリプトが、
ローカルに置いた Qwen2.5-0.5B-Instruct の `tokenizer.json` を読んで、文字列と token id を
相互変換するために使う。通信・`transformers`・`tokenizers` に依存しない（REQ-38）。
Phase 3 の Rust 実装の参照実装・ゴールデンベクタの生成元も兼ねる。

処理: NFC 正規化（normalizer が NFC のときだけ。null・キー無しなら行わない）
-> 前処理の正規表現で Isolated 分割 -> 各 piece を UTF-8 バイト列の
GPT-2 風文字表（`bytes_to_unicode`）へ写す -> merges の順位で BPE（`ignore_merges=false`）。

HF `tokenizers` との既知の差:

- **特殊トークン文字列を本文で照合しない**: `encode()` は `<|im_start|>` 等を通常の文字
  として分割する。特殊トークンは `build_chat_ids` が id で挿入する。入力本文による
  プロンプト注入を塞ぐため（REQ-39）。
- `\\p{L}`・`\\p{N}` は Python の `unicodedata`（Python 3.12 は Unicode 15.0）のカテゴリに
  依存する。HF（Rust の Oniguruma 相当）の Unicode 版と異なる文字では分割が食い違いうる。
- 空白は Unicode White_Space の明示集合を使う（Python の `\\s` は U+001C〜001F を含み、
  HF と異なるため）。
- 非 special の added_tokens も本文では照合しない（注入防止のための意図的な差）。
- `decode()` は special トークンを除去しない（HF 既定の skip_special_tokens=True と異なる）。
- 前処理の `(?i:...)` の大文字小文字畳み込みは Python と Oniguruma で異なりうる
  （U+017F ſ・U+212A Kelvin 記号など）。
- merges の重複は拒否する（HF は後勝ちの可能性があり、挙動を推測しないため。fail-closed）。

読み込みはガード層を通らない。信頼できるローカルパスだけを渡す前提の PoC ツールである。
それでも `tokenizer.json` は fstat で通常ファイルと確認し、サイズ（16 MiB 上限）を超えて
読まず、構造を検証して不正なら `ValueError`（メッセージに入力値を載せない）にする（REQ-39）。
"""

from __future__ import annotations

import heapq
import json
import os
import re
import stat
import unicodedata
from functools import lru_cache
from pathlib import Path

MAX_TOKENIZER_JSON_BYTES = 16 * 1024 * 1024
# encode の入力上限。PoC の入力は 1 件数文で、1 MiB 文字あれば十分余裕がある。
MAX_ENCODE_CHARS = 1 << 20
# BPE キャッシュの上限。短い piece（64 文字以下）だけを最大 20,000 件まで格納し、超えたら
# 格納しない（長い piece は毎回計算する）。1 件は最悪でも key 約 300 B＋値のリスト約 2 KiB
# （id の int は vocab 側の既存オブジェクトを共有）なので、総量は最悪で約 50 MB に収まる。
MAX_BPE_CACHE_ENTRIES = 20_000
MAX_CACHED_PIECE_CHARS = 64

# Qwen2 の Split 正規表現（tokenizer.json の pre_tokenizer と一致を要求する定数）。
SPLIT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Unicode White_Space（PropList）の符号位置。Python の `\s` は使わない。
_WHITE_SPACE_CODEPOINTS = [
    *[0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680],
    *range(0x2000, 0x200B),
    *[0x2028, 0x2029, 0x202F, 0x205F, 0x3000],
]
_WHITE_SPACE = "".join(chr(c) for c in _WHITE_SPACE_CODEPOINTS)

_INVALID = "invalid tokenizer.json"
# model 節の BPE 設定（実物 Qwen2.5-0.5B-Instruct の値。ignore_merges は実物ではキー無し）。
_MODEL_SETTINGS: dict[str, object] = {
    "dropout": None,
    "unk_token": None,
    "continuing_subword_prefix": "",
    "end_of_word_suffix": "",
    "fuse_unk": False,
    "byte_fallback": False,
    "ignore_merges": False,
}


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 の byte -> 可視 Unicode 文字の対応表（ByteLevel 前処理と同一）。"""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    table = {b: chr(b) for b in keep}
    extra = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + extra)
            extra += 1
    return table


@lru_cache(maxsize=1)
def _category_classes() -> tuple[str, str]:
    """`\\p{L}`・`\\p{N}` を `re` の文字クラス本体（範囲列）へ展開して返す。"""
    out: dict[str, list[tuple[int, int]]] = {"L": [], "N": []}
    for cp in range(0x110000):
        cat = unicodedata.category(chr(cp))[0]
        if cat in out:
            ranges = out[cat]
            if ranges and ranges[-1][1] == cp - 1:
                ranges[-1] = (ranges[-1][0], cp)
            else:
                ranges.append((cp, cp))

    def render(ranges: list[tuple[int, int]]) -> str:
        return "".join(
            f"\\U{lo:08x}" if lo == hi else f"\\U{lo:08x}-\\U{hi:08x}" for lo, hi in ranges
        )

    return render(out["L"]), render(out["N"])


@lru_cache(maxsize=1)
def _pre_tokenize_pattern() -> re.Pattern[str]:
    """Qwen2 の Split 正規表現（`SPLIT_REGEX`）を `re` へ翻訳したもの。"""
    letters, numbers = _category_classes()
    ws = "".join(f"\\U{ord(c):08x}" for c in _WHITE_SPACE)
    pattern = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        rf"|[^\r\n{letters}{numbers}]?[{letters}]+"
        rf"|[{numbers}]"
        rf"| ?[^{ws}{letters}{numbers}]+[\r\n]*"
        rf"|[{ws}]*[\r\n]+"
        rf"|[{ws}]+(?![^{ws}])"
        rf"|[{ws}]+"
    )
    return re.compile(pattern)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_pipeline(doc: dict) -> bool:
    """normalizer / pre_tokenizer / decoder / post_processor が対応範囲かを検証する。

    実装は NFC（または normalizer なし）・固定の Split＋ByteLevel・ByteLevel decode だけを
    再現するため、
    それ以外の設定は黙って誤変換せず拒否する（fail-closed）。
    """
    norm = doc.get("normalizer")
    if norm is not None and not (isinstance(norm, dict) and norm.get("type") == "NFC"):
        raise ValueError(_INVALID)
    nfc = norm is not None
    pre = doc.get("pre_tokenizer")
    parts = pre.get("pretokenizers") if isinstance(pre, dict) else None
    if not (isinstance(pre, dict) and pre.get("type") == "Sequence" and isinstance(parts, list)):
        raise ValueError(_INVALID)
    if len(parts) != 2 or not all(isinstance(x, dict) for x in parts):
        raise ValueError(_INVALID)
    split, byte_level = parts
    pattern = split.get("pattern")
    if not (
        split.get("type") == "Split"
        and isinstance(pattern, dict)
        and pattern.get("Regex") == SPLIT_REGEX
        and split.get("behavior") == "Isolated"
        and split.get("invert") is False
    ):
        raise ValueError(_INVALID)
    if not (
        byte_level.get("type") == "ByteLevel"
        and byte_level.get("add_prefix_space") is False
        and byte_level.get("use_regex") is False
    ):
        raise ValueError(_INVALID)
    decoder = doc.get("decoder")
    if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
        raise ValueError(_INVALID)
    # 実物の post_processor は ByteLevel（id を足さない）。TemplateProcessing 等は BOS/EOS を
    # 足しうるため拒否する。
    post = doc.get("post_processor")
    if post is not None and not (isinstance(post, dict) and post.get("type") == "ByteLevel"):
        raise ValueError(_INVALID)
    return nfc  # 戻り値: NFC 正規化を行うか


class Qwen2Tokenizer:
    """`tokenizer.json` から作る byte-level BPE トークナイザー。"""

    def __init__(self, doc: dict) -> None:
        if not isinstance(doc, dict):
            raise ValueError(_INVALID)
        self._nfc = _check_pipeline(doc)
        model = doc.get("model")
        if not isinstance(model, dict):
            raise ValueError(_INVALID)
        if model.get("type") != "BPE":
            raise ValueError(_INVALID)
        # 実物 Qwen2.5 の値に厳密一致を要求する（欠落は HF の既定値と同じ意味なので受理）。
        for key, expected in _MODEL_SETTINGS.items():
            if model.get(key, expected) != expected or type(model.get(key, expected)) is not type(
                expected
            ):
                raise ValueError(_INVALID)
        vocab = model.get("vocab")
        raw_merges = model.get("merges")
        added = doc.get("added_tokens")
        if not (
            isinstance(vocab, dict) and isinstance(raw_merges, list) and isinstance(added, list)
        ):
            raise ValueError(_INVALID)

        b2u = bytes_to_unicode()
        u2b = {u: b for b, u in b2u.items()}
        id_limit = len(vocab) + len(added)
        seen_ids: set[int] = set()
        for token, i in vocab.items():
            if not (isinstance(token, str) and _is_int(i) and 0 <= i < id_limit):
                raise ValueError(_INVALID)
            if i in seen_ids or any(c not in u2b for c in token):
                raise ValueError(_INVALID)
            seen_ids.add(i)
        if any(u not in vocab for u in b2u.values()):
            raise ValueError(_INVALID)

        added_map: dict[int, str] = {}
        special_ids: dict[str, int] = {}
        seen_contents: set[str] = set()
        for entry in added:
            if not isinstance(entry, dict):
                raise ValueError(_INVALID)
            i, content = entry.get("id"), entry.get("content")
            if not (_is_int(i) and isinstance(content, str) and 0 <= i < id_limit):
                raise ValueError(_INVALID)
            if i in seen_ids or i in added_map:
                raise ValueError(_INVALID)
            if content in seen_contents:  # content の重複は id の取り違えを招くため拒否
                raise ValueError(_INVALID)
            seen_contents.add(content)
            added_map[i] = content
            if entry.get("special") is True:
                special_ids[content] = i

        ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(raw_merges):
            if isinstance(merge, str):
                pair = tuple(merge.split(" "))
            elif isinstance(merge, list):
                pair = tuple(merge)
            else:
                raise ValueError(_INVALID)
            if len(pair) != 2 or not all(isinstance(x, str) and x in vocab for x in pair):
                raise ValueError(_INVALID)
            if (pair[0], pair[1]) in ranks:  # 重複の解釈が HF と一致する保証がないため拒否
                raise ValueError(_INVALID)
            if pair[0] + pair[1] not in vocab:  # HF の BpeBuilder も同条件でエラー
                raise ValueError(_INVALID)
            ranks[(pair[0], pair[1])] = rank

        # build_chat_ids が使う 3 件は special=true で存在することを要求する。
        try:
            self.special_ids = {
                "im_start": special_ids["<|im_start|>"],
                "im_end": special_ids["<|im_end|>"],
                "endoftext": special_ids["<|endoftext|>"],
            }
        except KeyError as exc:
            raise ValueError(_INVALID) from exc

        self._vocab: dict[str, int] = vocab
        self._ranks = ranks
        self._added = added_map
        self._u2b = u2b
        self._b2u = b2u
        self._id_to_token = {i: t for t, i in vocab.items()}
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> Qwen2Tokenizer:
        """通常ファイルかつ上限以下であることを確認して `tokenizer.json` を読む（REQ-39）。"""
        try:
            with Path(path).open("rb") as f:
                st = os.fstat(f.fileno())
                if not stat.S_ISREG(st.st_mode):
                    raise ValueError(_INVALID)
                if st.st_size > MAX_TOKENIZER_JSON_BYTES:
                    raise ValueError("tokenizer.json is too large")
                data = f.read(MAX_TOKENIZER_JSON_BYTES + 1)
            if len(data) > MAX_TOKENIZER_JSON_BYTES:
                raise ValueError("tokenizer.json is too large")
            doc = json.loads(data)
        except ValueError as exc:  # JSONDecodeError・UnicodeDecodeError を含む
            if str(exc) == "tokenizer.json is too large":
                raise
            raise ValueError(_INVALID) from None
        except (OSError, RecursionError):
            raise ValueError(_INVALID) from None
        return cls(doc)

    def pre_tokenize(self, text: str) -> list[str]:
        """NFC 正規化（normalizer が NFC のときのみ）後に Split(Isolated) した piece 列。"""
        if len(text) > MAX_ENCODE_CHARS:
            raise ValueError("input text is too long")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:  # 孤立サロゲート等。入力値はメッセージに載せない
            raise ValueError("input text is not valid unicode") from None
        if self._nfc:
            text = unicodedata.normalize("NFC", text)
        pieces: list[str] = []
        pos = 0
        for m in _pre_tokenize_pattern().finditer(text):
            if m.start() > pos:  # 一致しなかった部分も独立 piece（Isolated）
                pieces.append(text[pos : m.start()])
            pieces.append(m.group())
            pos = m.end()
        if pos < len(text):
            pieces.append(text[pos:])
        return pieces

    def _bpe(self, piece: str) -> list[int]:
        """1 piece を BPE する。優先度付きキュー＋双方向リンクで O(n log n)。

        順位が同じなら左側を先に併合する（「最小順位のペアを左から非重複で併合」と同じ結果）。
        併合で変わったトークンは版番号を上げ、古いキュー項目を読み出し時に捨てる。
        """
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        parts = [self._b2u[b] for b in piece.encode("utf-8")]
        n = len(parts)
        nxt = list(range(1, n + 1))
        prev = list(range(-1, n - 1))
        alive = [True] * n
        ver = [0] * n
        ranks = self._ranks
        heap: list[tuple[int, int, int, int, int]] = []

        def push(i: int, j: int) -> None:
            rank = ranks.get((parts[i], parts[j]))
            if rank is not None:
                heapq.heappush(heap, (rank, i, j, ver[i], ver[j]))

        for i in range(n - 1):
            push(i, i + 1)
        while heap:
            _, i, j, vi, vj = heapq.heappop(heap)
            if not (alive[i] and nxt[i] == j and ver[i] == vi and ver[j] == vj):
                continue
            parts[i] += parts[j]
            ver[i] += 1
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prev[nxt[j]] = i
            if prev[i] >= 0:
                push(prev[i], i)
            if nxt[i] < n:
                push(i, nxt[i])
        try:
            ids: list[int] = []
            i = 0
            while i < n:
                ids.append(self._vocab[parts[i]])
                i = nxt[i]
        except KeyError as exc:
            raise ValueError("BPE result is not in vocab") from exc
        if len(piece) <= MAX_CACHED_PIECE_CHARS and len(self._cache) < MAX_BPE_CACHE_ENTRIES:
            self._cache[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """本文を token id 列へ。特殊トークン文字列は照合しない。"""
        ids: list[int] = []
        for piece in self.pre_tokenize(text):
            ids.extend(self._bpe(piece))
        return ids

    def decode(self, ids: list[int]) -> str:
        """token id 列を文字列へ。added_tokens はその content をそのまま出す。"""
        out = bytearray()
        for i in ids:
            if not _is_int(i):
                raise ValueError("token id must be an int")
            if i in self._added:
                out.extend(self._added[i].encode("utf-8"))
                continue
            token = self._id_to_token.get(i)
            if token is None:
                raise ValueError("unknown token id")
            out.extend(self._u2b[c] for c in token)
        return out.decode("utf-8", errors="replace")

    def build_chat_ids(
        self,
        system: str,
        user: str,
        assistant: str | None = None,
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        """`<|im_start|>{role}\\n{content}<|im_end|>\\n` の連結（tools なし経路）を id 列で返す。

        特殊トークンは id で挿入し、各メッセージの `role\\ncontent` は 1 区間として encode する
        （HF が特殊トークン間の区間を丸ごと分割するのと同じ）。
        """
        if (assistant is None) != add_generation_prompt:
            raise ValueError("exactly one of assistant / add_generation_prompt is required")
        start, end = self.special_ids["im_start"], self.special_ids["im_end"]
        ids: list[int] = []
        messages = [("system", system), ("user", user)]
        if assistant is not None:
            messages.append(("assistant", assistant))
        for role, content in messages:
            ids += [start, *self.encode(f"{role}\n{content}"), end, *self.encode("\n")]
        if add_generation_prompt:
            ids += [start, *self.encode("assistant\n")]
        return ids
