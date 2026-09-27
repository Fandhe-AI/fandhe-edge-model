"""バイト入力へのエンコード（C1・C3 共通のトークン化）。

Rust 側 `preprocess.rs`（PoC-16。`docs/spec/03-poc/core-cli-vertical-slice/core/src/
preprocess.rs`）の `normalize_input` + `encode_bytes` と byte-identical になるよう
Python 側で独立に実装したもの。**この二重実装は意図的な暫定措置であり、正準化・
トークン化の規則を 1 箇所に集約する将来方針（REQ-15 は共通コアのハッシュ正準化が
対象で、本エンコードはそれとは別の推論前処理だが同様に一元化が望ましい）に反する。
学習ワーカー（Python）と推論ランタイム（Rust。REQ-32 により学習側を import できない）
は別プロセス・別言語であるため、当面は本リポ内 SSOT（`fixtures/preprocess/
byte_encoding_vectors.json`。`docs/spec` には依存しない。spec-reference.md）に対する
テスト（`trainer/tests/test_encoding.py`）で乖離を検出する。

将来仕様（Chore #10 Deliverable B。TASK-15.2 以降）: Rust workspace 作成後、推論
ランタイム crate（`preprocess.rs` 由来）にも同じ `fixtures/preprocess/
byte_encoding_vectors.json` を読み込んで一致検証するテストを追加し、両言語の
実装を 1 つの SSOT ベクタで機械照合する。Rust 側で NFKC 正規化を行うには
`unicode-normalization` crate の追加が要るが、これは依存追加であり
dependency-policy.md によりユーザー承認事項（バージョンは実装着手時点の
最新 0.1.x パッチ版を提示して確認する。PoC の `"0.1"` という範囲指定は
本リポの完全固定方針〔`=x.y.z`〕に反するため、そのままでは使えない）。

正規化: Unicode NFKC → 前後の空白除去 → 内部の連続空白を半角スペース 1 つへ圧縮
（Python の `str.split()`/`str.strip()` は U+001C-U+001F の情報分離子も空白として
扱うため、Rust 側もこれに合わせて実装されている。`unicodedata` は標準ライブラリの
空白判定を使うため、素朴な `str.split()` で Rust 側の `is_py_whitespace` と同じ挙動になる）。

トークン化: 正規化後の文字列を UTF-8 バイト列にし、各バイトに 1 を足す
（0 を詰め物用に空ける）。先頭 max_bytes バイトで切り詰める。空文字列は `[0]` を返す
（REF 側の空入力対応。ref_predict.py predict_c3 と合わせる。preprocess.rs 同様）。
"""

from __future__ import annotations

import unicodedata


def normalize_input(text: str) -> str:
    """NFKC 正規化 → 前後空白除去 → 内部の連続空白を半角スペース 1 つへ圧縮する。

    Rust 側 `preprocess::normalize_input` と同じ規則（テスト側で確認する）。
    """
    nfkc = unicodedata.normalize("NFKC", text)
    return " ".join(nfkc.split())


def encode_bytes(text: str, max_bytes: int) -> list[int]:
    """正規化済みでない生の入力文字列を、C3 モデルへ渡すトークン列（バイト+1・0=詰め物）へ変換する。

    Rust 側 `ort_backend::predict` が `normalize_input` → `encode_bytes` の順で
    呼ぶのと同じ経路を、学習側では 1 関数にまとめている（学習・推論の双方で
    同じ経路を通ることが REQ-28 の一致契約の前提になる）。
    """
    normalized = normalize_input(text)
    raw = normalized.encode("utf-8")
    take = raw[:max_bytes]
    ids = [b + 1 for b in take]
    if not ids:
        ids = [0]
    return ids
