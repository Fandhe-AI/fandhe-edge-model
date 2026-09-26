"""学習リクエスト・学習データの検証テスト（TASK-19.1 の入口・REQ-39 の資源上限・
経路の閉じ込め〔PoC-20〕）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from fandhe_edge_trainer import contract, limits
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode


def _base_request(root: Path) -> dict:
    """`root` 配下に閉じ込められた最小の妥当なリクエストを組み立てる。

    `root` はテストごとの `tmp_path` を渡す（`train_path="train.jsonl"`・
    `out_dir="out"` は root 直下の相対パス）。
    """
    return {
        "schema_version": 1,
        "kind": "c3",
        "kind_version": 1,
        "config": {},
        "label_order": ["a", "b"],
        "max_bytes": 64,
        "seed": 0,
        "device": "cpu",
        "root": str(root),
        "train_path": "train.jsonl",
        "out_dir": "out",
    }


def _write(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_load_request_accepts_valid_request(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", _base_request(tmp_path))
    req = contract.load_request(p)
    assert req.kind == "c3"
    assert req.label_order == ["a", "b"]
    assert req.max_bytes == 64
    assert req.seed == 0
    assert req.device == "cpu"
    # root 配下（realpath 解決後）に閉じ込められていること。
    root_real = Path(os.path.realpath(tmp_path))
    assert req.root == root_real
    assert req.train_path == root_real / "train.jsonl"
    assert req.out_dir == root_real / "out"


def test_load_request_accepts_nested_relative_path(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    req_dict = {**_base_request(tmp_path), "train_path": "data/train.jsonl"}
    p = _write(tmp_path / "req.json", req_dict)
    req = contract.load_request(p)
    root_real = Path(os.path.realpath(tmp_path))
    assert req.train_path == root_real / "data" / "train.jsonl"


def test_load_request_rejects_unknown_field(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", {**_base_request(tmp_path), "extra_field": 1})
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
    assert "extra_field" in exc_info.value.message


def test_load_request_rejects_wrong_type(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", {**_base_request(tmp_path), "max_bytes": "512"})
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"


def test_load_request_rejects_oversize_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limits, "MAX_REQUEST_BYTES", 8)
    monkeypatch.setattr(contract, "MAX_REQUEST_BYTES", 8)
    p = _write(tmp_path / "req.json", _base_request(tmp_path))
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "limit_exceeded"
    assert exc_info.value.exit_code == ExitCode.LIMIT_EXCEEDED


# --------------------------------------------------------------------------
# P0-1: 経路の閉じ込め（REQ-39 ガード層・PoC-20。多層防御。guard.py 参照）
# --------------------------------------------------------------------------


def test_load_request_rejects_absolute_train_path(tmp_path: Path) -> None:
    req_dict = {**_base_request(tmp_path), "train_path": "/etc/passwd"}
    p = _write(tmp_path / "req.json", req_dict)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_path"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_load_request_rejects_dotdot_train_path(tmp_path: Path) -> None:
    req_dict = {**_base_request(tmp_path), "train_path": "../outside/train.jsonl"}
    p = _write(tmp_path / "req.json", req_dict)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_path"


def test_load_request_rejects_dotdot_out_dir(tmp_path: Path) -> None:
    req_dict = {**_base_request(tmp_path), "out_dir": "../outside_out"}
    p = _write(tmp_path / "req.json", req_dict)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_path"


def test_load_request_rejects_root_not_a_directory(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_text("x", encoding="utf-8")
    req_dict = {**_base_request(tmp_path), "root": str(not_a_dir)}
    p = _write(tmp_path / "req.json", req_dict)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_path"


def test_load_request_rejects_root_relative_path(tmp_path: Path) -> None:
    req_dict = {**_base_request(tmp_path), "root": "relative/path"}
    p = _write(tmp_path / "req.json", req_dict)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_path"


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink は POSIX 限定を前提にテストする")
def test_load_request_rejects_symlink_escape_for_train_path(tmp_path: Path) -> None:
    """root 配下のシンボリックリンクが root の外を指す場合、train_path の親経路として
    使うと path_outside_root で拒否されること（realpath による閉じ込め検証）。
    """
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "train.jsonl").write_text('{"input": "x", "label": "a"}\n', encoding="utf-8")
    (root / "escape").symlink_to(outside)

    req_dict = {**_base_request(root), "train_path": "escape/train.jsonl"}
    p = _write(root / "req.json", req_dict)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "path_outside_root"


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink は POSIX 限定を前提にテストする")
def test_load_request_rejects_symlink_escape_for_out_dir(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside)

    req_dict = {**_base_request(root), "out_dir": "escape/out"}
    p = _write(root / "req.json", req_dict)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "path_outside_root"


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


def test_load_train_examples_rejects_oversize_file_via_stat_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P0-2: `st_size` による判定が実際の行読み取りより先に行われること。

    ファイル本文を不正な JSON にしておき、それでも `limit_exceeded` になる
    （＝ 1 行も parse されていない）ことで、st_size チェックが読み取りより
    先に効いていることを確認する。
    """
    monkeypatch.setattr(contract, "MAX_TRAIN_DATA_BYTES", 1 << 10)
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{not json\n", encoding="utf-8")
    os.truncate(train_path, (1 << 10) + 1)  # スパースファイルで stat 上のサイズだけ超過させる
    with pytest.raises(WorkerError) as exc_info:
        contract.load_train_examples(train_path, ["a", "b"])
    assert exc_info.value.code == "limit_exceeded"
    # invalid_data（JSON パースエラー）ではなく limit_exceeded になっている時点で、
    # パースに到達する前に stat のサイズ判定で弾かれていることの証拠になる。
    assert "line" not in exc_info.value.message


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
        json.dumps({**_base_request(tmp_path), "max_bytes": 64}).replace(
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
    p = _write(tmp_path / "req.json", {**_base_request(tmp_path), "seed": -1})
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"


def test_load_request_rejects_seed_too_large(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", {**_base_request(tmp_path), "seed": 2**32})
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"


def test_load_request_rejects_bool_seed(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", {**_base_request(tmp_path), "seed": True})
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
