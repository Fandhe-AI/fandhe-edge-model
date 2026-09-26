"""`guard.py`（経路の閉じ込め・fd ベースの TOCTOU 対策）の単体テスト。

経路の閉じ込めそのもの（`root` の検証・シンボリックリンクの拒否・スワップ攻撃への
耐性）は `test_contract.py`（`contract.load_request`・`prepare_out_dir` 経由）で
検証する。本ファイルは `guard.py` 単体で完結する不変条件（fail-closed・fd の
冪等なクローズ）だけを扱う。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fandhe_edge_trainer import guard
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode


def test_resolve_root_fails_closed_when_dir_fd_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`os.supports_dir_fd` にこのモジュールが要る関数が無い環境では、経路
    ベース呼び出しへ黙ってフォールバックせず `runtime_error`（exit 70）で拒否する。
    """
    monkeypatch.setattr(guard.os, "supports_dir_fd", frozenset())
    with pytest.raises(WorkerError) as exc_info:
        guard.resolve_root(str(tmp_path))
    assert exc_info.value.code == "runtime_error"
    assert exc_info.value.exit_code == ExitCode.RUNTIME_ERROR


def test_resolve_root_fails_closed_when_follow_symlinks_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard.os, "supports_follow_symlinks", frozenset())
    with pytest.raises(WorkerError) as exc_info:
        guard.resolve_root(str(tmp_path))
    assert exc_info.value.code == "runtime_error"
    assert exc_info.value.exit_code == ExitCode.RUNTIME_ERROR


def test_root_handle_close_is_idempotent(tmp_path: Path) -> None:
    root_handle = guard.resolve_root(str(tmp_path))
    root_handle.close()
    root_handle.close()  # 2 回目も例外を送出しない


def test_confined_entry_close_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    root_handle = guard.resolve_root(str(tmp_path))
    entry = guard.confine(root_handle, "sub/train.jsonl", "train_path")
    fd = entry.parent_fd
    entry.close()
    entry.close()  # 2 回目も例外を送出しない（fd は既に閉じている）
    with pytest.raises(OSError, match=r"Bad file descriptor"):
        os.fstat(fd)  # 実際に閉じられていることの確認
    root_handle.close()


def test_confine_rejects_empty_relative_path(tmp_path: Path) -> None:
    root_handle = guard.resolve_root(str(tmp_path))
    try:
        with pytest.raises(WorkerError) as exc_info:
            guard.confine(root_handle, "", "train_path")
        assert exc_info.value.code == "invalid_path"
    finally:
        root_handle.close()


def test_confine_rejects_nul_byte(tmp_path: Path) -> None:
    root_handle = guard.resolve_root(str(tmp_path))
    try:
        with pytest.raises(WorkerError) as exc_info:
            guard.confine(root_handle, "a\x00b", "train_path")
        assert exc_info.value.code == "invalid_path"
    finally:
        root_handle.close()
