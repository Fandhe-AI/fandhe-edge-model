"""`artifact.py` の完全性検証（P0: AGENTS.md ガード層「完全性と版」・REQ-39）のテスト。"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest

from fandhe_edge_trainer import artifact, contract, guard
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode


def test_hashing_writer_computes_sha256_of_written_bytes() -> None:
    buf = io.BytesIO()
    writer = artifact.HashingWriter(buf)
    writer.write(b"hello ")
    writer.write(b"world")
    assert buf.getvalue() == b"hello world"
    assert writer.hexdigest() == hashlib.sha256(b"hello world").hexdigest()


def test_build_artifact_includes_onnx_sha256() -> None:
    art = artifact.build_artifact(
        kind="c3",
        kind_version=1,
        config={},
        label_order=["a", "b"],
        output_type="choice",
        max_bytes=64,
        candidate_label="c3",
        onnx_sha256="ab" * 32,
    )
    assert art["onnx_sha256"] == "ab" * 32


def _make_reservation(tmp_path: Path) -> tuple[guard.RootHandle, contract.OutDirReservation]:
    root_handle = guard.resolve_root(str(tmp_path))
    entry = guard.confine(root_handle, "out", "out_dir")
    reservation = contract.prepare_out_dir(entry)
    return root_handle, reservation


def _write_model_and_artifact(tmp_fd: int, onnx_bytes: bytes, *, onnx_sha256: str | None) -> None:
    fd = os.open(
        artifact.ONNX_FILE_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=tmp_fd
    )
    with os.fdopen(fd, "wb") as f:
        f.write(onnx_bytes)
    if onnx_sha256 is None:
        onnx_sha256 = hashlib.sha256(onnx_bytes).hexdigest()
    art = artifact.build_artifact(
        kind="c3",
        kind_version=1,
        config={},
        label_order=["a", "b"],
        output_type="choice",
        max_bytes=64,
        candidate_label="c3",
        onnx_sha256=onnx_sha256,
    )
    artifact.write_artifact(tmp_fd, art)


def test_verify_output_accepts_matching_hash(tmp_path: Path) -> None:
    root_handle, reservation = _make_reservation(tmp_path)
    try:
        _write_model_and_artifact(reservation.tmp_fd, b"fake onnx bytes", onnx_sha256=None)
        artifact.verify_output(reservation.tmp_fd)  # 例外を送出しないことを確認する
    finally:
        reservation.entry.close()
        root_handle.close()


def test_verify_output_rejects_tampered_model_before_finalize(tmp_path: Path) -> None:
    """P0: 確定前に `model.onnx` が改ざん（＝ `artifact.json` 記録時と異なる
    バイト列に）されていた場合、`integrity_check_failed`（exit 70）で拒否し、
    `out_dir` を確定させない（呼び出し元が `finalize_out_dir` を呼ばないことで
    公開されない。本テストは検証関数自体の拒否を確認する）。
    """
    root_handle, reservation = _make_reservation(tmp_path)
    try:
        _write_model_and_artifact(reservation.tmp_fd, b"original bytes", onnx_sha256=None)

        # 確定前に model.onnx を改ざんする（例: 何らかの理由でバイト列が変わった）。
        fd = os.open("model.onnx", os.O_WRONLY, dir_fd=reservation.tmp_fd)
        with os.fdopen(fd, "wb") as f:
            f.write(b"tampered bytes!!")

        with pytest.raises(WorkerError) as exc_info:
            artifact.verify_output(reservation.tmp_fd)
        assert exc_info.value.code == "integrity_check_failed"
        assert exc_info.value.exit_code == ExitCode.RUNTIME_ERROR

        # out_dir はまだ予約段階（空ディレクトリ）のまま。確定（rename）していない。
        assert not any((tmp_path / "out").iterdir())
    finally:
        contract.cleanup_reservation(reservation)
        root_handle.close()
    assert not (tmp_path / "out").exists()  # 予約も解放され、公開されていない


def test_verify_output_rejects_missing_hash_field(tmp_path: Path) -> None:
    root_handle, reservation = _make_reservation(tmp_path)
    try:
        fd = os.open(
            artifact.ONNX_FILE_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=reservation.tmp_fd,
        )
        with os.fdopen(fd, "wb") as f:
            f.write(b"some bytes")
        art = {
            "kind": "c3",
            "kind_version": 1,
            "selector_version": "0.1",
            "config": {},
            "label_order": ["a", "b"],
            "output_type": "choice",
            "max_bytes": 64,
            "onnx_file": "model.onnx",
            "created_utc": "2026-01-01T00:00:00Z",
            "candidate_label": "c3",
            # onnx_sha256 を意図的に欠落させる。
        }
        artifact.write_artifact(reservation.tmp_fd, art)

        with pytest.raises(WorkerError) as exc_info:
            artifact.verify_output(reservation.tmp_fd)
        assert exc_info.value.code == "integrity_check_failed"
        assert exc_info.value.exit_code == ExitCode.RUNTIME_ERROR
    finally:
        contract.cleanup_reservation(reservation)
        root_handle.close()


def test_verify_output_rejects_missing_model_file(tmp_path: Path) -> None:
    root_handle, reservation = _make_reservation(tmp_path)
    try:
        art = artifact.build_artifact(
            kind="c3",
            kind_version=1,
            config={},
            label_order=["a", "b"],
            output_type="choice",
            max_bytes=64,
            candidate_label="c3",
            onnx_sha256="ab" * 32,
        )
        artifact.write_artifact(reservation.tmp_fd, art)
        # model.onnx を書かない。

        with pytest.raises(WorkerError) as exc_info:
            artifact.verify_output(reservation.tmp_fd)
        assert exc_info.value.code == "integrity_check_failed"
        assert exc_info.value.exit_code == ExitCode.RUNTIME_ERROR
    finally:
        contract.cleanup_reservation(reservation)
        root_handle.close()
