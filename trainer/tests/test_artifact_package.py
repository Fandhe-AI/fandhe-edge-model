"""自己完結パッケージの自己整合検査（REQ-41・REQ-39・TASK-41.6-3・#432）。

`artifact.verify_output` が tokenizer・外部データ shard の sha256 も照合し、経路・予約名を
開く前に拒否することと、`open_new_member` の名前規則を確かめる。証拠種別: テストハーネス。
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from fandhe_edge_trainer import artifact, contract, guard
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode

_SHARD = "weights-00000.bin"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_package(tmp_fd: int) -> dict:
    """tokenizer 種類の最小パッケージ（model.onnx・tokenizer.json・shard 1 件）を書く。"""
    files = {"model.onnx": b"onnx", "tokenizer.json": b"{}", _SHARD: b"shard-bytes"}
    for name, data in files.items():
        with artifact.open_new_member_unchecked(tmp_fd, name) as f:
            f.write(data)
    extra = {
        "input_representation": "tokenizer",
        "tokenizer_file": "tokenizer.json",
        "tokenizer_sha256": _sha(b"{}"),
        "external_data": [{"location": _SHARD, "size_bytes": 11, "sha256": _sha(b"shard-bytes")}],
        "scoring": {"system_prompt": "x", "choices": []},
        "base_model": {"files": [], "license": None, "model_name": "m", "origin": None},
    }
    art = artifact.build_artifact(
        kind="qwen2_lora",
        kind_version=1,
        config={},
        label_order=["a", "b"],
        output_type="choice",
        max_bytes=64,
        candidate_label="qwen2_lora",
        onnx_sha256=_sha(b"onnx"),
        extra=extra,
    )
    artifact.write_artifact(tmp_fd, art)
    return art


def _rewrite_artifact(fd: int, art: dict) -> None:
    os.unlink("artifact.json", dir_fd=fd)
    artifact.write_artifact(fd, art)


def _verify(tmp_path: Path, mutate: Callable[[int], None]) -> None:
    root_handle = guard.resolve_root(str(tmp_path))
    reservation = contract.prepare_out_dir(guard.confine(root_handle, "out", "out_dir"))
    try:
        mutate(reservation.tmp_fd)
        artifact.verify_output(reservation.tmp_fd)
    finally:
        reservation.entry.close()
        root_handle.close()


def _expect_integrity_failure(tmp_path: Path, mutate: Callable[[int], None]) -> None:
    with pytest.raises(WorkerError) as e:
        _verify(tmp_path, mutate)
    assert e.value.code == "integrity_check_failed"
    assert e.value.exit_code == ExitCode.RUNTIME_ERROR


def test_req41_verify_output_accepts_complete_tokenizer_package(tmp_path: Path) -> None:
    """tokenizer・shard が揃い sha256 が一致すれば受理する。"""
    _verify(tmp_path, _write_package)


@pytest.mark.parametrize("target", [_SHARD, "tokenizer.json"])
def test_req41_verify_output_rejects_tampered_member(tmp_path: Path, target: str) -> None:
    """shard・tokenizer の同サイズ改ざんは integrity_check_failed。"""

    def mutate(fd: int) -> None:
        _write_package(fd)
        size = os.stat(target, dir_fd=fd).st_size
        with os.fdopen(os.open(target, os.O_WRONLY, dir_fd=fd), "wb") as f:
            f.write(b"x" * size)

    _expect_integrity_failure(tmp_path, mutate)


@pytest.mark.parametrize("target", [_SHARD, "tokenizer.json"])
def test_req41_verify_output_rejects_missing_member(tmp_path: Path, target: str) -> None:
    """記録された shard・tokenizer が無ければ integrity_check_failed。"""

    def mutate(fd: int) -> None:
        _write_package(fd)
        os.unlink(target, dir_fd=fd)

    _expect_integrity_failure(tmp_path, mutate)


@pytest.mark.parametrize("bad", ["../escape", "a/b", ".hidden", "model.onnx", "tokenizer.json", ""])
def test_req39_verify_output_rejects_unsafe_shard_location(tmp_path: Path, bad: str) -> None:
    """記録された shard 名が経路・予約名なら開く前に拒否する。"""

    def mutate(fd: int) -> None:
        art = _write_package(fd)
        art["external_data"][0]["location"] = bad
        _rewrite_artifact(fd, art)

    _expect_integrity_failure(tmp_path, mutate)


def test_req41_verify_output_rejects_tokenizer_fields_on_byte_package(tmp_path: Path) -> None:
    """byte 種類に tokenizer の欄があれば拒否する。"""

    def mutate(fd: int) -> None:
        art = _write_package(fd)
        art["input_representation"] = "byte"
        _rewrite_artifact(fd, art)

    _expect_integrity_failure(tmp_path, mutate)


@pytest.mark.parametrize("bad", ["artifact.json", "model.onnx", "../x", "a/b", ""])
def test_req39_open_new_member_rejects_reserved_and_unsafe_names(tmp_path: Path, bad: str) -> None:
    """種類が作れるファイル名は字句規則内かつ予約名以外。"""
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(WorkerError) as e:
            artifact.open_new_member(fd, bad)
        assert e.value.exit_code == ExitCode.RUNTIME_ERROR
    finally:
        os.close(fd)


def test_req39_open_new_member_is_exclusive(tmp_path: Path) -> None:
    """既存ファイルは上書きしない（O_EXCL）。"""
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with artifact.open_new_member(fd, _SHARD) as f:
            f.write(b"x")
        with pytest.raises(FileExistsError):
            artifact.open_new_member(fd, _SHARD)
    finally:
        os.close(fd)
