"""学習リクエスト・学習データの検証テスト（TASK-19.1 の入口・REQ-39 の資源上限）。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from fandhe_edge_trainer import contract, limits
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode

_BASE_REQUEST = {
    "schema_version": 1,
    "kind": "c3",
    "kind_version": 1,
    "config": {},
    "label_order": ["a", "b"],
    "max_bytes": 64,
    "seed": 0,
    "device": "cpu",
    "train_path": "train.jsonl",
    "out_dir": "out",
}


def _write(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_load_request_accepts_valid_request(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", _BASE_REQUEST)
    req = contract.load_request(p)
    assert req.kind == "c3"
    assert req.label_order == ["a", "b"]
    assert req.max_bytes == 64
    assert req.seed == 0
    assert req.device == "cpu"


def test_load_request_rejects_unknown_field(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", {**_BASE_REQUEST, "extra_field": 1})
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
    assert "extra_field" in exc_info.value.message


def test_load_request_rejects_wrong_type(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", {**_BASE_REQUEST, "max_bytes": "512"})
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"


def test_load_request_rejects_oversize_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limits, "MAX_REQUEST_BYTES", 8)
    monkeypatch.setattr(contract, "MAX_REQUEST_BYTES", 8)
    p = _write(tmp_path / "req.json", _BASE_REQUEST)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


def test_load_train_examples_reports_line_number_not_label_text(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        '{"input": "x1", "label": "a"}\n{"input": "x2", "label": "does-not-exist-label"}\n',
        encoding="utf-8",
    )
    with pytest.raises(WorkerError) as exc_info:
        contract.load_train_examples(train_path, ["a", "b"])
    assert exc_info.value.code == "invalid_data"
    assert "line 2" in exc_info.value.message
    # データ本文（ラベル文字列そのもの）をメッセージへ含めない（security.md）。
    assert "does-not-exist-label" not in exc_info.value.message


def test_load_train_examples_rejects_invalid_json_line(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text('{"input": "x1", "label": "a"}\n{not json\n', encoding="utf-8")
    with pytest.raises(WorkerError) as exc_info:
        contract.load_train_examples(train_path, ["a", "b"])
    assert exc_info.value.code == "invalid_data"
    assert "line 2" in exc_info.value.message


def test_load_train_examples_rejects_blank_line(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text('{"input": "x1", "label": "a"}\n\n', encoding="utf-8")
    with pytest.raises(WorkerError) as exc_info:
        contract.load_train_examples(train_path, ["a", "b"])
    assert "line 2" in exc_info.value.message


def test_load_train_examples_rejects_oversize_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract, "MAX_TRAIN_DATA_BYTES", 4)
    train_path = tmp_path / "train.jsonl"
    train_path.write_text('{"input": "x1", "label": "a"}\n', encoding="utf-8")
    with pytest.raises(WorkerError) as exc_info:
        contract.load_train_examples(train_path, ["a", "b"])
    assert exc_info.value.code == "limit_exceeded"


def test_load_train_examples_requires_at_least_two_distinct_labels(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        '{"input": "x1", "label": "a"}\n{"input": "x2", "label": "a"}\n', encoding="utf-8"
    )
    with pytest.raises(WorkerError) as exc_info:
        contract.load_train_examples(train_path, ["a", "b"])
    assert exc_info.value.code == "invalid_data"


def test_load_request_rejects_nan(tmp_path: Path) -> None:
    """項目 4: JSON 標準外の数値トークン（NaN 等）は範囲比較をすり抜けるため明示的に拒否する。"""
    p = tmp_path / "req.json"
    p.write_text(
        json.dumps({**_BASE_REQUEST, "max_bytes": 64}).replace(
            '"max_bytes": 64', '"max_bytes": NaN'
        ),
        encoding="utf-8",
    )
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_load_request_rejects_deeply_nested_json(tmp_path: Path) -> None:
    """項目 4: 過度なネストによる RecursionError を invalid_request（exit 64）へ正規化する。"""
    # 20_000 段は CPython の既定再帰上限で確実に RecursionError になる水準（実測確認済み）。
    # 過度に大きくすると C 拡張の再帰実装がクラッシュしうるため、必要最小限に留める。
    nested = "[" * 20_000 + "]" * 20_000
    p = tmp_path / "req.json"
    p.write_text(nested, encoding="utf-8")
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


@pytest.mark.skipif(sys.platform == "win32", reason="os.mkfifo は POSIX 限定")
def test_load_request_rejects_fifo(tmp_path: Path) -> None:
    """項目 5: 通常ファイル以外（FIFO 等）は S_ISREG 検査で拒否し、無制限待ちを防ぐ。"""
    fifo_path = tmp_path / "req.fifo"
    os.mkfifo(fifo_path)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(fifo_path)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_load_request_rejects_seed_out_of_range(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", {**_BASE_REQUEST, "seed": -1})
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"


def test_load_request_rejects_seed_too_large(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", {**_BASE_REQUEST, "seed": 2**32})
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"


def test_load_request_rejects_bool_seed(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", {**_BASE_REQUEST, "seed": True})
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"


def test_load_train_examples_rejects_nan_in_line(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text('{"input": NaN, "label": "a"}\n', encoding="utf-8")
    with pytest.raises(WorkerError) as exc_info:
        contract.load_train_examples(train_path, ["a", "b"])
    assert exc_info.value.code == "invalid_data"
    assert "line 1" in exc_info.value.message


def test_load_train_examples_rejects_whitespace_only_input(tmp_path: Path) -> None:
    """項目 6: 正規化後に空となる入力（空白のみ等）を拒否する。メッセージに入力本文を含めない。"""
    train_path = tmp_path / "train.jsonl"
    train_path.write_text('{"input": "   \\t  ", "label": "a"}\n', encoding="utf-8")
    with pytest.raises(WorkerError) as exc_info:
        contract.load_train_examples(train_path, ["a", "b"])
    assert exc_info.value.code == "invalid_data"
    assert "line 1" in exc_info.value.message


def test_load_train_examples_rejects_oversize_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract, "MAX_TRAIN_LINE_BYTES", 16)
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        '{"input": "much longer than sixteen bytes", "label": "a"}\n', encoding="utf-8"
    )
    with pytest.raises(WorkerError) as exc_info:
        contract.load_train_examples(train_path, ["a", "b"])
    assert exc_info.value.code == "limit_exceeded"
    assert "line 1" in exc_info.value.message


def test_load_train_examples_rejects_too_many_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract, "MAX_TRAIN_EXAMPLES", 1)
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        '{"input": "x1", "label": "a"}\n{"input": "x2", "label": "b"}\n', encoding="utf-8"
    )
    with pytest.raises(WorkerError) as exc_info:
        contract.load_train_examples(train_path, ["a", "b"])
    assert exc_info.value.code == "limit_exceeded"


@pytest.mark.skipif(sys.platform == "win32", reason="os.mkfifo は POSIX 限定")
def test_load_train_examples_rejects_fifo(tmp_path: Path) -> None:
    fifo_path = tmp_path / "train.fifo"
    os.mkfifo(fifo_path)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_train_examples(fifo_path, ["a", "b"])
    assert exc_info.value.code == "invalid_request"


def test_finalize_out_dir_converts_os_error_to_worker_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """項目 8: `os.replace` の失敗を `WorkerError`（output_conflict・exit 64）へ変換する。"""
    out_dir = tmp_path / "out"
    tmp_dir = contract.prepare_out_dir(out_dir)

    def _boom(_src: object, _dst: object) -> None:
        raise OSError("simulated concurrent creation")

    monkeypatch.setattr(contract.os, "replace", _boom)
    with pytest.raises(WorkerError) as exc_info:
        contract.finalize_out_dir(tmp_dir, out_dir)
    assert exc_info.value.code == "output_conflict"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
    assert not tmp_dir.exists()  # 失敗時も一時ディレクトリは掃除される


def test_prepare_out_dir_rejects_existing_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(WorkerError) as exc_info:
        contract.prepare_out_dir(out_dir)
    assert exc_info.value.code == "invalid_request"


def test_prepare_and_finalize_out_dir_roundtrip(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    tmp_dir = contract.prepare_out_dir(out_dir)
    assert tmp_dir.exists()
    assert not out_dir.exists()
    (tmp_dir / "marker.txt").write_text("ok", encoding="utf-8")
    contract.finalize_out_dir(tmp_dir, out_dir)
    assert out_dir.exists()
    assert (out_dir / "marker.txt").read_text(encoding="utf-8") == "ok"
