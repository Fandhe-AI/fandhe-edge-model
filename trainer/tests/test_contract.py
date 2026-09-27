"""学習リクエスト・学習データの検証テスト（TASK-19.1 の入口・REQ-39 の資源上限・
経路の閉じ込め〔PoC-20〕。fd ベースの TOCTOU 対策を含む）。
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import pytest

from fandhe_edge_trainer import contract, guard, limits
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


def _confine(
    tmp_path: Path, rel: str, field_name: str
) -> tuple[guard.RootHandle, guard.ConfinedEntry]:
    """テスト用に `root=tmp_path` 配下の `rel` を閉じ込める（呼び出し側が戻り値の
    `RootHandle`・`ConfinedEntry` を使い終わったら `close()` すること）。
    """
    root_handle = guard.resolve_root(str(tmp_path))
    entry = guard.confine(root_handle, rel, field_name)
    return root_handle, entry


@contextlib.contextmanager
def _confined_train_path(tmp_path: Path, rel: str = "train.jsonl"):
    root_handle, entry = _confine(tmp_path, rel, "train_path")
    try:
        yield entry
    finally:
        entry.close()
        root_handle.close()


@contextlib.contextmanager
def _confined_out_dir(tmp_path: Path, rel: str = "out"):
    root_handle, entry = _confine(tmp_path, rel, "out_dir")
    try:
        yield entry
    finally:
        entry.close()
        root_handle.close()


def test_load_request_accepts_valid_request(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(
        '{"input": "x", "label": "a"}\n{"input": "y", "label": "b"}\n', encoding="utf-8"
    )
    p = _write(tmp_path / "req.json", _base_request(tmp_path))
    req = contract.load_request(p)
    try:
        assert req.kind == "c3"
        assert req.label_order == ["a", "b"]
        assert req.max_bytes == 64
        assert req.seed == 0
        assert req.device == "cpu"
        assert req.train_path.name == "train.jsonl"
        assert req.out_dir.name == "out"
        # fd が実際に root 配下の実体を指していることを機能的に確認する
        # （train_path の open・読み取りが成功する）。
        examples = contract.load_train_examples(req.train_path, req.label_order)
        assert len(examples) == 2
    finally:
        req.close_resources()


def test_load_request_accepts_nested_relative_path(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.jsonl").write_text(
        '{"input": "x", "label": "a"}\n{"input": "y", "label": "b"}\n', encoding="utf-8"
    )
    req_dict = {**_base_request(tmp_path), "train_path": "data/train.jsonl"}
    p = _write(tmp_path / "req.json", req_dict)
    req = contract.load_request(p)
    try:
        assert req.train_path.name == "train.jsonl"
        examples = contract.load_train_examples(req.train_path, req.label_order)
        assert len(examples) == 2
    finally:
        req.close_resources()


def test_load_request_rejects_unknown_field(tmp_path: Path) -> None:
    p = _write(tmp_path / "req.json", {**_base_request(tmp_path), "extra_field": 1})
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
    assert "extra_field" in exc_info.value.message


def test_load_request_rejects_unknown_field_with_bounded_message_length(tmp_path: Path) -> None:
    """P1-1: 未知のフィールド名（リクエスト JSON の全体サイズ上限まで利用者が
    自由に長くできる）を、切り詰めずにそのままエラーメッセージへ埋め込まない。
    """
    huge_field_name = "x" * 10_000
    p = _write(tmp_path / "req.json", {**_base_request(tmp_path), huge_field_name: 1})
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"
    assert len(exc_info.value.message) < 1000  # 切り詰められていることの目安


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
# 経路の閉じ込め（REQ-39 ガード層・PoC-20。多層防御。guard.py 参照）
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
    使うと拒否されること（`O_NOFOLLOW` による ELOOP 検出。symlink_not_allowed）。
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
    assert exc_info.value.code == "symlink_not_allowed"


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
    assert exc_info.value.code == "symlink_not_allowed"


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink は POSIX 限定を前提にテストする")
def test_load_request_rejects_symlink_pointing_inside_root_too(tmp_path: Path) -> None:
    """経路の途中にシンボリックリンクがあれば、root 配下を指していても拒否する
    （以前の実装より厳格な方針。TOCTOU を完全に塞ぐため、経路の途中の
    シンボリックリンクを一切追跡しない。guard.py のモジュール docstring 参照）。
    """
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "train.jsonl").write_text('{"input": "x", "label": "a"}\n', encoding="utf-8")
    (tmp_path / "link").symlink_to(real_dir)  # root 配下の別の場所を指す（root の外ではない）

    req_dict = {**_base_request(tmp_path), "train_path": "link/train.jsonl"}
    p = _write(tmp_path / "req.json", req_dict)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "symlink_not_allowed"


def test_load_train_examples_reports_line_number_not_label_text(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(
        '{"input": "x1", "label": "a"}\n{"input": "x2", "label": "does-not-exist-label"}\n',
        encoding="utf-8",
    )
    with _confined_train_path(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.load_train_examples(entry, ["a", "b"])
        assert exc_info.value.code == "invalid_data"
        assert "line 2" in exc_info.value.message
        # データ本文（ラベル文字列そのもの）をメッセージへ含めない（security.md）。
        assert "does-not-exist-label" not in exc_info.value.message


def test_load_train_examples_rejects_invalid_json_line(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(
        '{"input": "x1", "label": "a"}\n{not json\n', encoding="utf-8"
    )
    with _confined_train_path(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.load_train_examples(entry, ["a", "b"])
        assert exc_info.value.code == "invalid_data"
        assert "line 2" in exc_info.value.message


def test_load_train_examples_rejects_blank_line(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"input": "x1", "label": "a"}\n\n', encoding="utf-8")
    with _confined_train_path(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.load_train_examples(entry, ["a", "b"])
        assert "line 2" in exc_info.value.message


def test_load_train_examples_rejects_oversize_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract, "MAX_TRAIN_DATA_BYTES", 4)
    (tmp_path / "train.jsonl").write_text('{"input": "x1", "label": "a"}\n', encoding="utf-8")
    with _confined_train_path(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.load_train_examples(entry, ["a", "b"])
        assert exc_info.value.code == "limit_exceeded"


def test_load_train_examples_rejects_oversize_file_via_stat_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`st_size` による判定が実際の行読み取りより先に行われること。

    ファイル本文を不正な JSON にしておき、それでも `limit_exceeded` になる
    （＝ 1 行も parse されていない）ことで、st_size チェックが読み取りより
    先に効いていることを確認する。
    """
    monkeypatch.setattr(contract, "MAX_TRAIN_DATA_BYTES", 1 << 10)
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{not json\n", encoding="utf-8")
    os.truncate(train_path, (1 << 10) + 1)  # スパースファイルで stat 上のサイズだけ超過させる
    with _confined_train_path(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.load_train_examples(entry, ["a", "b"])
        assert exc_info.value.code == "limit_exceeded"
        # invalid_data（JSON パースエラー）ではなく limit_exceeded になっている時点で、
        # パースに到達する前に stat のサイズ判定で弾かれていることの証拠になる。
        assert "line" not in exc_info.value.message


def test_load_train_examples_requires_at_least_two_distinct_labels(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(
        '{"input": "x1", "label": "a"}\n{"input": "x2", "label": "a"}\n', encoding="utf-8"
    )
    with _confined_train_path(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.load_train_examples(entry, ["a", "b"])
        assert exc_info.value.code == "invalid_data"


def test_load_request_rejects_nan(tmp_path: Path) -> None:
    """JSON 標準外の数値トークン（NaN 等）は範囲比較をすり抜けるため明示的に拒否する。"""
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
    """過度なネストによる RecursionError を invalid_request（exit 64）へ正規化する。"""
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
    """通常ファイル以外（FIFO 等）は S_ISREG 検査で拒否し、無制限待ちを防ぐ。"""
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


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "kind_version",
        "max_bytes",
        "seed",
        "time_limit_seconds",
        "rss_limit_bytes",
    ],
)
@pytest.mark.parametrize("value", [True, False])
def test_load_request_rejects_bool_for_integer_fields(
    tmp_path: Path, field: str, value: bool
) -> None:
    """P1: 真偽値は `int` のサブクラス（`True == 1`・`False == 0`）であるため、
    整数フィールドの検証は明示的に `bool` を除外しないと素通りしうる
    （`schema_version` はこの見落としがあった。他のフィールドは既に対応済みだが、
    回帰を防ぐためすべて一括りに確認する）。
    """
    req_dict = {**_base_request(tmp_path), field: value}
    p = _write(tmp_path / "req.json", req_dict)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"
    assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_load_request_accepts_lower_resource_limits(tmp_path: Path) -> None:
    """`time_limit_seconds`・`rss_limit_bytes` は既定の上限を下げるだけ許可する。"""
    req_dict = {**_base_request(tmp_path), "time_limit_seconds": 60, "rss_limit_bytes": 1024}
    p = _write(tmp_path / "req.json", req_dict)
    req = contract.load_request(p)
    try:
        assert req.time_limit_seconds == 60
        assert req.rss_limit_bytes == 1024
    finally:
        req.close_resources()


def test_load_request_rejects_time_limit_above_max(tmp_path: Path) -> None:
    from fandhe_edge_trainer.limits import MAX_TRAIN_WALL_SECONDS

    req_dict = {**_base_request(tmp_path), "time_limit_seconds": MAX_TRAIN_WALL_SECONDS + 1}
    p = _write(tmp_path / "req.json", req_dict)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"


def test_load_request_rejects_rss_limit_above_max(tmp_path: Path) -> None:
    from fandhe_edge_trainer.limits import MAX_TRAIN_RSS_BYTES

    req_dict = {**_base_request(tmp_path), "rss_limit_bytes": MAX_TRAIN_RSS_BYTES + 1}
    p = _write(tmp_path / "req.json", req_dict)
    with pytest.raises(WorkerError) as exc_info:
        contract.load_request(p)
    assert exc_info.value.code == "invalid_request"


def test_load_train_examples_rejects_nan_in_line(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"input": NaN, "label": "a"}\n', encoding="utf-8")
    with _confined_train_path(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.load_train_examples(entry, ["a", "b"])
        assert exc_info.value.code == "invalid_data"
        assert "line 1" in exc_info.value.message


def test_load_train_examples_rejects_whitespace_only_input(tmp_path: Path) -> None:
    """正規化後に空となる入力（空白のみ等）を拒否する。メッセージに入力本文を含めない。"""
    (tmp_path / "train.jsonl").write_text('{"input": "   \\t  ", "label": "a"}\n', encoding="utf-8")
    with _confined_train_path(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.load_train_examples(entry, ["a", "b"])
        assert exc_info.value.code == "invalid_data"
        assert "line 1" in exc_info.value.message


def test_load_train_examples_rejects_oversize_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract, "MAX_TRAIN_LINE_BYTES", 16)
    (tmp_path / "train.jsonl").write_text(
        '{"input": "much longer than sixteen bytes", "label": "a"}\n', encoding="utf-8"
    )
    with _confined_train_path(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.load_train_examples(entry, ["a", "b"])
        assert exc_info.value.code == "limit_exceeded"
        assert "line 1" in exc_info.value.message


def test_load_train_examples_rejects_too_many_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract, "MAX_TRAIN_EXAMPLES", 1)
    (tmp_path / "train.jsonl").write_text(
        '{"input": "x1", "label": "a"}\n{"input": "x2", "label": "b"}\n', encoding="utf-8"
    )
    with _confined_train_path(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.load_train_examples(entry, ["a", "b"])
        assert exc_info.value.code == "limit_exceeded"


@pytest.mark.skipif(sys.platform == "win32", reason="os.mkfifo は POSIX 限定")
def test_load_train_examples_rejects_fifo(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "train.jsonl")
    with _confined_train_path(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.load_train_examples(entry, ["a", "b"])
        assert exc_info.value.code == "invalid_request"


# --------------------------------------------------------------------------
# out_dir の予約・確定（TOCTOU 対策。P0-A/P1）
# --------------------------------------------------------------------------


def test_finalize_out_dir_converts_os_error_to_worker_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`os.rename` の失敗を `WorkerError`（output_conflict・exit 64）へ変換する。"""
    with _confined_out_dir(tmp_path) as entry:
        reservation = contract.prepare_out_dir(entry)

        def _boom(*_a: object, **_k: object) -> None:
            raise OSError("simulated concurrent creation")

        monkeypatch.setattr(contract.os, "rename", _boom)
        with pytest.raises(WorkerError) as exc_info:
            contract.finalize_out_dir(reservation)
        assert exc_info.value.code == "output_conflict"
        assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
        assert (tmp_path / "out").is_dir()  # 予約済み out_dir 自体は触れずに残る
        # 自分の一時ディレクトリ（隠しディレクトリ）は掃除される。
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]


def test_prepare_out_dir_rejects_existing_dir(tmp_path: Path) -> None:
    """`os.mkdir` の `FileExistsError` を検出する（存在確認 → 作成の 2 手順にしない）。"""
    (tmp_path / "out").mkdir()
    with _confined_out_dir(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.prepare_out_dir(entry)
        assert exc_info.value.code == "output_conflict"
        assert exc_info.value.exit_code == ExitCode.INVALID_INPUT


def test_prepare_out_dir_releases_reservation_when_tmp_mkdir_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`os.mkdir(out_dir)`（予約）成功後に、一時ディレクトリの `os.mkdir` が
    失敗した場合、予約済みの `out_dir` を残置せず解放すること（さもないと
    空の予約だけが残る。P1）。
    """
    real_mkdir = contract.os.mkdir
    calls = {"n": 0}

    def _flaky_mkdir(*args: object, **kwargs: object):
        calls["n"] += 1
        if calls["n"] == 2:  # 1 回目 = out_dir の予約（成功させる）、2 回目 = 一時ディレクトリ
            raise OSError("simulated tmp mkdir failure")
        return real_mkdir(*args, **kwargs)

    with _confined_out_dir(tmp_path) as entry:
        monkeypatch.setattr(contract.os, "mkdir", _flaky_mkdir)
        with pytest.raises(WorkerError) as exc_info:
            contract.prepare_out_dir(entry)
        assert exc_info.value.code == "invalid_request"
        assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
        assert not (tmp_path / "out").exists()  # 予約が残置されない


def test_prepare_and_finalize_out_dir_roundtrip(tmp_path: Path) -> None:
    with _confined_out_dir(tmp_path) as entry:
        reservation = contract.prepare_out_dir(entry)
        assert (tmp_path / "out").is_dir()  # 予約済み（空ディレクトリとして先に作成されている）
        assert not any((tmp_path / "out").iterdir())

        fd = os.open(
            "marker.txt",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=reservation.tmp_fd,
        )
        with os.fdopen(fd, "wb") as f:
            f.write(b"ok")
        contract.finalize_out_dir(reservation)
        assert (tmp_path / "out" / "marker.txt").read_text(encoding="utf-8") == "ok"


def test_finalize_out_dir_rejects_when_reserved_dir_became_nonempty(tmp_path: Path) -> None:
    """予約後に別プロセス（を模したテスト側の書き込み）が out_dir へ何かを
    置いた場合、`os.rename` が ENOTEMPTY で失敗し、その内容を消さずに残すこと。
    """
    with _confined_out_dir(tmp_path) as entry:
        reservation = contract.prepare_out_dir(entry)
        fd = os.open(
            "marker.txt",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=reservation.tmp_fd,
        )
        with os.fdopen(fd, "wb") as f:
            f.write(b"ok")

        # 予約と確定の間に、別プロセスが out_dir の中身を書き換えたことを模す。
        (tmp_path / "out" / "foreign.txt").write_text("someone else's data", encoding="utf-8")

        with pytest.raises(WorkerError) as exc_info:
            contract.finalize_out_dir(reservation)
        assert exc_info.value.code == "output_conflict"
        assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
        # 他者が書いた内容は消さない。
        assert (tmp_path / "out" / "foreign.txt").read_text(
            encoding="utf-8"
        ) == "someone else's data"


def test_finalize_out_dir_rejects_when_reserved_dir_replaced_by_different_dir(
    tmp_path: Path,
) -> None:
    """予約後に out_dir が別物（別 inode のディレクトリ）へすり替わっていた場合、
    `os.rename` を呼ぶ前に `(st_dev, st_ino)` の不一致で検出し拒否すること。
    """
    with _confined_out_dir(tmp_path) as entry:
        reservation = contract.prepare_out_dir(entry)

        # 予約済みディレクトリを削除し、同名の別ディレクトリへ差し替える
        # （別プロセスによる置き換えを模す。inode が変わる）。
        (tmp_path / "out").rmdir()
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "someone_elses_file.txt").write_text("x", encoding="utf-8")

        with pytest.raises(WorkerError) as exc_info:
            contract.finalize_out_dir(reservation)
        assert exc_info.value.code == "output_conflict"
        assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
        # すり替わった側には触れない。
        assert (tmp_path / "out" / "someone_elses_file.txt").exists()


def test_cleanup_reserved_out_dir_removes_own_empty_reservation(tmp_path: Path) -> None:
    with _confined_out_dir(tmp_path) as entry:
        reservation = contract.prepare_out_dir(entry)
        contract.cleanup_reserved_out_dir(entry, reservation.reserved_id)
        assert not (tmp_path / "out").exists()


def test_cleanup_reserved_out_dir_leaves_nonempty_reservation(tmp_path: Path) -> None:
    with _confined_out_dir(tmp_path) as entry:
        reservation = contract.prepare_out_dir(entry)
        (tmp_path / "out" / "foreign.txt").write_text("x", encoding="utf-8")
        contract.cleanup_reserved_out_dir(entry, reservation.reserved_id)
        assert (tmp_path / "out").exists()
        assert (tmp_path / "out" / "foreign.txt").exists()


def test_finalize_out_dir_rejects_when_tmp_dir_swapped(tmp_path: Path) -> None:
    """P0-2: 予約後・確定前に、作業用一時ディレクトリの名前が別物（別 inode の
    ディレクトリ）へすり替わっていた場合、`fstat(tmp_fd)` と
    `stat(tmp_name)` の不一致で検出して拒否し、すり替わった側の中身には
    一切触れないこと（`rename` は名前しか受け取らないため、fd を握っている
    だけでは差し替えを防げない。直前の突き合わせが本質的な防御になる）。
    """
    with _confined_out_dir(tmp_path) as entry:
        reservation = contract.prepare_out_dir(entry)
        tmp_path_on_disk = tmp_path / reservation.tmp_name

        # 自分の一時ディレクトリを退避し、同名・別 inode のディレクトリへ
        # 差し替える（別プロセスによる置き換えを模す）。
        real_tmp = tmp_path / f"{reservation.tmp_name}.real"
        tmp_path_on_disk.rename(real_tmp)
        tmp_path_on_disk.mkdir()
        (tmp_path_on_disk / "evidence.txt").write_text("foreign data", encoding="utf-8")

        with pytest.raises(WorkerError) as exc_info:
            contract.finalize_out_dir(reservation)
        assert exc_info.value.code == "output_conflict"
        assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
        # すり替わった側の中身には一切触れられていない。
        assert (tmp_path_on_disk / "evidence.txt").read_text(encoding="utf-8") == "foreign data"
        # out_dir 自体（予約済みの空ディレクトリ）にも触れられていない。
        assert (tmp_path / "out").is_dir()
        assert not any((tmp_path / "out").iterdir())


def test_cleanup_reservation_never_touches_foreign_prefix_siblings(tmp_path: Path) -> None:
    """P0-1: `cleanup_reservation` は自分の予約（`reservation.tmp_name`・
    `reservation.entry.name`）だけを対象にし、たとえ同じ命名規則
    （`.{name}.tmp-*`）に一致する無関係な兄弟ディレクトリがあっても一切
    触れないこと（名前の前方一致だけを根拠にした一括削除は廃止済み。
    以前の `cleanup_orphaned_reservation` はこの前方一致に依存していた）。
    """
    with _confined_out_dir(tmp_path) as entry:
        reservation = contract.prepare_out_dir(entry)

        foreign = tmp_path / ".out.tmp-deadbeefdeadbeef"
        foreign.mkdir()
        (foreign / "not_mine.txt").write_text("do not touch", encoding="utf-8")

        contract.cleanup_reservation(reservation)

        assert not (tmp_path / "out").exists()
        assert not (tmp_path / reservation.tmp_name).exists()
        assert foreign.exists()
        assert (foreign / "not_mine.txt").read_text(encoding="utf-8") == "do not touch"


# --------------------------------------------------------------------------
# P0: 親ディレクトリの所有者・書き込み権限検査（Codex レビュー指摘）。
# 予約前に group/others 書き込み可能・他ユーザー所有の親ディレクトリを拒否する
# ことで、確定直前の名前差し替え（TOCTOU）の余地そのものを狭める（REQ-39）。
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX のパーミッションビットを前提にテストする"
)
def test_prepare_out_dir_rejects_group_writable_parent(tmp_path: Path) -> None:
    """親ディレクトリが group 書き込み可能な場合、`out_dir` を何も作らずに
    `output_conflict` で拒否すること（REQ-39 ガード層）。
    """
    original_mode = tmp_path.stat().st_mode & 0o777
    os.chmod(tmp_path, 0o757)  # noqa: S103 - テスト専用。group 書き込み可能な状態を意図的に作る
    try:
        with _confined_out_dir(tmp_path) as entry:
            with pytest.raises(WorkerError) as exc_info:
                contract.prepare_out_dir(entry)
            assert exc_info.value.code == "output_conflict"
            assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
            assert not (tmp_path / "out").exists()  # 予約前に拒否され、何も作られない
    finally:
        os.chmod(tmp_path, original_mode)


def test_prepare_out_dir_rejects_parent_not_owned_by_current_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """親ディレクトリの所有者が実効ユーザーと異なる場合、`output_conflict` で
    拒否すること（`os.geteuid()` を差し替えて「自分の所有ではない」状態を
    決定的に再現する）。
    """
    real_euid = os.geteuid()
    monkeypatch.setattr(contract.os, "geteuid", lambda: real_euid + 1)
    with _confined_out_dir(tmp_path) as entry:
        with pytest.raises(WorkerError) as exc_info:
            contract.prepare_out_dir(entry)
        assert exc_info.value.code == "output_conflict"
        assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
        assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------
# P0: rename 成功後の実体再確認（Codex レビュー指摘）。`os.rename` が返っても
# 公開された実体が検証済みの tmp_fd と同一かを確かめるまで成功にしない。
# --------------------------------------------------------------------------


def test_finalize_out_dir_rejects_when_rename_publishes_a_different_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`os.rename` 自体が「別の実体」を `out_dir` の位置へ置いてしまった場合
    （名前差し替えの TOCTOU を模す）、rename 呼び出しは成功していても
    post-rename の `(st_dev, st_ino)` 突き合わせで検出し、`output_conflict` と
    して成功を報告しないこと。公開された実体（decoy 側）には一切触れない。
    """
    real_rename = os.rename
    with _confined_out_dir(tmp_path) as entry:
        reservation = contract.prepare_out_dir(entry)

        # 検証対象（tmp_fd の実体）へ marker を書く。
        fd = os.open(
            "marker.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=reservation.tmp_fd
        )
        with os.fdopen(fd, "wb") as f:
            f.write(b"verified content")

        # decoy: 検証していない別ディレクトリ（別 inode）を用意する。
        decoy_name = "decoy-dir"
        os.mkdir(decoy_name, 0o700, dir_fd=entry.parent_fd)
        decoy_fd = os.open(decoy_name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=entry.parent_fd)
        marker_fd = os.open(
            "decoy_marker.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=decoy_fd
        )
        with os.fdopen(marker_fd, "wb") as f:
            f.write(b"decoy content")
        os.close(decoy_fd)

        def _fake_rename(src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
            # 呼び出し元が意図した src（自分の tmp_name）を無視し、decoy を
            # dst（out_dir の名前）へ rename する（rename 自体が偽の実体を
            # 公開してしまう TOCTOU を模す）。
            real_rename(decoy_name, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        monkeypatch.setattr(contract.os, "rename", _fake_rename)
        with pytest.raises(WorkerError) as exc_info:
            contract.finalize_out_dir(reservation)
        assert exc_info.value.code == "output_conflict"
        assert exc_info.value.exit_code == ExitCode.INVALID_INPUT
        # 公開された実体（decoy）には触れられていない。
        assert (tmp_path / "out" / "decoy_marker.txt").read_text(
            encoding="utf-8"
        ) == "decoy content"
        # 自分の一時ディレクトリ（検証済みの実体）は _release_tmp によって
        # 掃除される（残置しない）。
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]


# --------------------------------------------------------------------------
# TOCTOU の核心確認: confine 後に経路をシンボリックリンクへ差し替えても、
# fd に束縛された実体（confine 時点のディレクトリ）だけが使われること。
# --------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink は POSIX 限定を前提にテストする")
def test_prepare_out_dir_is_immune_to_parent_swap_after_confine(tmp_path: Path) -> None:
    root = tmp_path / "root"
    # mode を明示する（umask 依存で group/others 書き込み可になると、P0 の
    # 親ディレクトリ権限検査〔prepare_out_dir〕に本テストの主眼と無関係な
    # 理由で弾かれてしまうため）。
    root.mkdir(mode=0o700)
    (root / "sub").mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir()

    root_handle = guard.resolve_root(str(root))
    entry = guard.confine(root_handle, "sub/out", "out_dir")
    try:
        # confine が完了した直後に、"sub" 自体を別物（outside への symlink）へ
        # 差し替える（攻撃者が経路の途中を差し替える TOCTOU を模す）。
        (root / "sub").rename(root / "sub_real")
        (root / "sub").symlink_to(outside)

        reservation = contract.prepare_out_dir(entry)
        try:
            fd = os.open(
                "marker.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=reservation.tmp_fd,
            )
            with os.fdopen(fd, "wb") as f:
                f.write(b"ok")
            contract.finalize_out_dir(reservation)
        except BaseException:
            contract.cleanup_reservation(reservation)
            raise

        # 実際の書き込み先は confine 時点の実体（sub_real 配下）であり、
        # 後から作られたシンボリックリンク（sub → outside）の先ではない。
        assert (root / "sub_real" / "out" / "marker.txt").read_text(encoding="utf-8") == "ok"
        assert not list(outside.iterdir())  # outside には何も書き込まれない
        assert not (root / "sub" / "out").exists()  # symlink の先には何も無い
    finally:
        entry.close()
        root_handle.close()


def test_cleanup_tmp_contents_via_fd_does_not_leak_fds(tmp_path: Path) -> None:
    """一時ディレクトリの後始末（中身あり・空の両方）で fd を 1 つも漏らさない。

    失敗時クリーンアップのたびに fd が漏れると、長時間のジョブ管理下で
    fd 枯渇を招く（`os.scandir(fd)` は渡した fd を閉じない。P0-2 では
    「既に開いている fd」だけを使い、名前で開き直さない設計にしたため、
    本関数自体は渡された fd を閉じない契約になっている＝呼び出し側の責務）。
    """
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "a.bin").write_bytes(b"x")
    (tmp_path / "empty").mkdir()
    full_fd = os.open(tmp_path / "full", os.O_RDONLY | os.O_DIRECTORY)
    empty_fd = os.open(tmp_path / "empty", os.O_RDONLY | os.O_DIRECTORY)
    try:
        before = len(os.listdir("/dev/fd"))
        contract._cleanup_tmp_contents_via_fd(full_fd)
        contract._cleanup_tmp_contents_via_fd(empty_fd)
        after = len(os.listdir("/dev/fd"))
    finally:
        os.close(full_fd)
        os.close(empty_fd)
    assert after == before
    assert list((tmp_path / "full").iterdir()) == []
