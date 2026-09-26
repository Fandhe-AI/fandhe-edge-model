"""CLI（`python -m fandhe_edge_trainer train --request ...`）の統合テスト。

Rust 側 CLI が想定する子プロセス呼び出し方（`sys.executable` を引数リストで、
`shell=True` を使わずに起動する。security.md）を模して subprocess で実行する。
`trainer/pyproject.toml` は `package = false`（配布パッケージを持たない）ため、
`PYTHONPATH` で `src/` を解決する。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import LABEL_ORDER, TINY_CONFIG

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


def _run_cli(request_path: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": _SRC_DIR}
    return subprocess.run(
        [sys.executable, "-m", "fandhe_edge_trainer", "train", "--request", str(request_path)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )


def _run_cli_argv(argv: list[str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": _SRC_DIR}
    return subprocess.run(
        [sys.executable, "-m", "fandhe_edge_trainer", *argv],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )


def _assert_single_json_error(result: subprocess.CompletedProcess[str]) -> dict:
    # 項目 7: argparse の既定動作（exit 2・usage を stderr へ出力）ではなく、
    # 本ワーカーの 7 種の終了コード契約（invalid_request・exit 64）に従うこと。
    assert result.returncode == 64, (result.returncode, result.stdout, result.stderr)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_request"
    return payload


def test_cli_missing_request_argument_exits_64_with_single_json(tmp_path: Path) -> None:
    result = _run_cli_argv(["train"])
    _assert_single_json_error(result)


def test_cli_unknown_subcommand_exits_64_with_single_json(tmp_path: Path) -> None:
    result = _run_cli_argv(["bogus-command"])
    _assert_single_json_error(result)


def test_cli_no_subcommand_exits_64_with_single_json(tmp_path: Path) -> None:
    result = _run_cli_argv([])
    _assert_single_json_error(result)


def _write_train_data(path: Path) -> None:
    rows = []
    for i in range(12):
        rows.append({"input": f"alpha alpha beta gamma {i}", "label": "cat_a"})
        rows.append({"input": f"delta delta epsilon zeta {i}", "label": "cat_b"})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_cli_train_success_emits_single_json_and_exit_0(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    _write_train_data(train_path)
    out_dir = tmp_path / "out"
    request = {
        "schema_version": 1,
        "kind": "c3",
        "kind_version": 1,
        "config": TINY_CONFIG,
        "label_order": LABEL_ORDER,
        "max_bytes": 64,
        "seed": 0,
        "device": "cpu",
        "train_path": str(train_path),
        "out_dir": str(out_dir),
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    result = _run_cli(request_path)
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "ok"
    assert payload["artifact"]["kind"] == "c3"
    assert (out_dir / "artifact.json").exists()
    assert (out_dir / "model.onnx").exists()


def test_cli_train_invalid_request_exits_64(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({"schema_version": 1, "kind": "c3"}), encoding="utf-8")

    result = _run_cli(request_path)
    assert result.returncode == 64
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_request"
