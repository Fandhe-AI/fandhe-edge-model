"""学習ワーカーのエラー型。

Rust 側 CLI（子プロセスの起動元）が機械可読な `code` / `message` の JSON へ
写せる形（REQ-21）でエラーを表現する。`message` は英語・機械向けの短い説明に限り、
学習データ本文（個人情報・機密情報を含みうる。security.md）を一切含めない。
件数・行番号・型名だけを記載する。
"""

from __future__ import annotations

from .exitcode import ExitCode

#: `truncate_for_message` の既定の切り詰め長（文字数）。
_DEFAULT_MESSAGE_TRUNCATE_LIMIT = 64


class WorkerError(Exception):
    """学習ワーカーが検出した、利用者・呼び出し元へ伝えるべきエラー。

    Args:
        code: 機械可読なエラー種別（英語 snake_case。例: "invalid_request"）。
        message: 英語の短い説明。データ本文を含めない。
        exit_code: 対応する終了コード（ExitCode）。
    """

    def __init__(self, code: str, message: str, exit_code: ExitCode) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code


def truncate_for_message(value: str, limit: int = _DEFAULT_MESSAGE_TRUNCATE_LIMIT) -> str:
    """エラーメッセージへ埋め込むリクエスト由来の値（`kind`・config のキー名等）を
    上限の長さに切り詰める。

    `WorkerError.message` はデータ本文（学習データの `input`・`label`）を含めない
    方針（security.md）だが、`kind` や `config` のキー名のようにデータ本文では
    ない値であっても、リクエスト JSON の全体サイズ上限（`limits.py::
    MAX_REQUEST_BYTES`）までは利用者が自由に長くできる。切り詰めずにそのまま
    メッセージへ埋め込むと、エラーメッセージ自体が事実上無制限の長さになりうる
    ため、ここで一律に切り詰める。
    """
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...(truncated, {len(value)} chars total)"


def truncate_list_for_message(
    values: list[str], *, item_limit: int = _DEFAULT_MESSAGE_TRUNCATE_LIMIT, max_items: int = 10
) -> str:
    """文字列のリスト（例: 未知のフィールド名の一覧）をエラーメッセージへ安全に
    埋め込める形に切り詰める（各要素は `item_limit` 文字、件数は `max_items` 件まで。
    それぞれ超過分は省略した旨だけを付記する）。
    """
    shown = [truncate_for_message(v, item_limit) for v in values[:max_items]]
    suffix = "" if len(values) <= max_items else f" (+{len(values) - max_items} more)"
    return f"{shown}{suffix}"
