"""学習ワーカーのエラー型。

Rust 側 CLI（子プロセスの起動元）が機械可読な `code` / `message` の JSON へ
写せる形（REQ-21）でエラーを表現する。`message` は英語・機械向けの短い説明に限り、
学習データ本文（個人情報・機密情報を含みうる。security.md）を一切含めない。
件数・行番号・型名だけを記載する。
"""

from __future__ import annotations

from .exitcode import ExitCode


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
