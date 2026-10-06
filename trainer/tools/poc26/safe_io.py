"""PoC-26 ツール共通の「上限つきで安全にファイルを読む」処理（REQ-41・TASK-41.1-5・#390）。

役割: `qwen2_tokenizer.py`・`qwen2_model.py`・CLI（`common.py` 経由）が使う、ローカルの信頼できる
パスのファイルを読む処理を 1 箇所にまとめる。標準ライブラリだけで書き、WorkerError・終了コード
には依存しない（呼び出し側が写す。CLI は `common.py` が 64 / 20 へ写す）。

- `open_regular` / `read_limited`: `O_NOFOLLOW | O_NONBLOCK` で開き、fstat で通常ファイルと
  サイズ上限を確認し、`limit + 1` までだけ読む（REQ-39）。`O_NOFOLLOW` は最後の要素だけに効く。
  パス途中のディレクトリの symlink は辿る（P3: PoC・ローカルの信頼できるパスだけを渡す脅威モデルで
  許容。ガード層の `safe_join` 相当は持たない）。
- `loads_json`: NaN・Infinity・重複キーを拒否し、不正な UTF-8・深すぎる入れ子も
  `ValueError` にして返す。
- `sha256_hex`: バイト列の sha256（64 桁の小文字 16 進）。

メッセージには入力値（ファイル内容・本文）を載せない。上限超過は専用の `LimitExceededError`
（`ValueError` の部分型）で表し、メッセージの文言では判別しない。
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


class LimitExceededError(ValueError):
    """サイズ・長さ・件数の上限超過。CLI（`cli.py`）が終了コード 20 へ写す。

    `ValueError` の部分型なので、既存の `except ValueError` と `pytest.raises(ValueError)` は
    そのまま通る。メッセージ文言の一致で上限超過を判別しない（文言の変更で終了コードが
    変わるのを防ぐ）。
    """


def open_regular(path: str | Path, limit: int, what: str) -> tuple[int, os.stat_result]:
    """symlink を辿らず通常ファイルだけを開き、サイズ上限を確認して (fd, stat) を返す。

    開けない・通常ファイルでないときは `ValueError`、`limit` 超過は `LimitExceededError`。
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (OSError, ValueError):  # ValueError: パスに NUL を含む場合
        raise ValueError(f"cannot open {what}") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"{what} is not a regular file")
        if st.st_size > limit:
            raise LimitExceededError(f"{what} too large: {st.st_size} bytes > {limit}")
    except BaseException:
        os.close(fd)
        raise
    return fd, st


def read_limited(path: str | Path, limit: int, what: str) -> bytes:
    """通常ファイルを fd 経由で読む（open 後の fstat で上限確認。読み超過も拒否）。"""
    fd, _ = open_regular(path, limit, what)
    with os.fdopen(fd, "rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise LimitExceededError(f"{what} too large: > {limit} bytes")
    return data


def loads_json(raw: bytes, what: str) -> Any:
    """NaN・Infinity・重複キーを拒否して JSON を読む。失敗は `ValueError(f"invalid {what}")`。"""

    def _no_const(_: str) -> None:
        raise ValueError

    def _no_dup(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [k for k, _ in pairs]
        if len(set(keys)) != len(keys):  # 重複キーは後勝ちで黙って受理せず拒否する
            raise ValueError
        return dict(pairs)

    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_no_const, object_pairs_hook=_no_dup)
    except (ValueError, RecursionError):  # JSONDecodeError・UnicodeDecodeError を含む
        raise ValueError(f"invalid {what}") from None


def sha256_hex(data: bytes) -> str:
    """バイト列の sha256（64 桁の小文字 16 進）。"""
    return hashlib.sha256(data).hexdigest()
