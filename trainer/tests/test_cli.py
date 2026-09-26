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
    # root は絶対パス、train_path・out_dir は root からの相対パス
    # （経路の閉じ込め。REQ-39・PoC-20・guard.py）。
    request = {
        "schema_version": 1,
        "kind": "c3",
        "kind_version": 1,
        "config": TINY_CONFIG,
        "label_order": LABEL_ORDER,
        "max_bytes": 64,
        "seed": 0,
        "device": "cpu",
        "root": str(tmp_path),
        "train_path": "train.jsonl",
        "out_dir": "out",
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
    # 予約に使った作業用一時ディレクトリ（`.out.tmp-*`）が残置されていない
    # （スーパーバイザーが確定〔rename〕まで正しく完了させたことの確認）。
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]


def test_cli_train_succeeds_with_large_stdout_payload(tmp_path: Path) -> None:
    """`supervisor.py::_drain_stdout` が別スレッドでパイプを溜めずに読み進める
    ことで、ワーカーの標準出力が OS のパイプ容量（一般的に 64KiB 程度）を
    大きく超えても（本テストは `label_order` を 1024 件 × 約 200 バイトにして
    artifact の JSON 応答を肥大化させる）、監視がブロックによる誤検知
    〔`limit_exceeded`〕を起こさず正常終了すること（exit 0）。
    """
    label_order = [f"label-{i:04d}-" + "x" * 190 for i in range(1024)]
    assert all(len(label.encode("utf-8")) <= 256 for label in label_order)

    train_path = tmp_path / "train.jsonl"
    rows = []
    for i in range(12):
        rows.append({"input": f"alpha alpha beta gamma {i}", "label": label_order[0]})
        rows.append({"input": f"delta delta epsilon zeta {i}", "label": label_order[1]})
    train_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    request = {
        "schema_version": 1,
        "kind": "c3",
        "kind_version": 1,
        "config": TINY_CONFIG,
        "label_order": label_order,
        "max_bytes": 64,
        "seed": 0,
        "device": "cpu",
        "root": str(tmp_path),
        "train_path": "train.jsonl",
        "out_dir": "out",
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    result = _run_cli(request_path)
    assert result.returncode == 0, (result.returncode, result.stdout[:500], result.stderr[:2000])
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    # パイプ容量（64KiB 程度）を明確に超える大きさであることの確認。
    assert len(lines[0].encode("utf-8")) > 64 * 1024
    payload = json.loads(lines[0])
    assert payload["status"] == "ok"
    assert payload["artifact"]["label_order"] == label_order
    assert (out_dir / "artifact.json").exists()
    assert (out_dir / "model.onnx").exists()


def test_cli_train_exceeding_rss_limit_is_killed_and_cleaned_up(tmp_path: Path) -> None:
    """P0-2: 実際にワーカー（mlx を import する本物のプロセス）を起動しても、
    極小の `rss_limit_bytes` を与えればスーパーバイザーが強制終了し、
    `limit_exceeded`（exit 20）を返すこと。かつ、出力先には何も残らない
    （予約〔`out_dir`・作業用一時ディレクトリ〕が解放される。証拠種別:
    テストハーネス〔本テスト実行機での実測〕）。

    学習を人為的に遅くする代わりに RSS 上限を極小（10 MiB）にする設計にした
    理由: Python・mlx を import した時点で RSS は 10 MiB を確実に上回るため、
    学習の実際の所要時間に依存せず決定的にキルできる（本番コードにテスト専用の
    遅延フックを仕込む必要が無い）。
    """
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
        "root": str(tmp_path),
        "train_path": "train.jsonl",
        "out_dir": "out",
        "rss_limit_bytes": 10 * 1024 * 1024,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    result = _run_cli(request_path)
    assert result.returncode == 20, (result.returncode, result.stdout, result.stderr)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "error"
    assert payload["code"] == "limit_exceeded"
    # 予約済み out_dir・作業用一時ディレクトリのいずれも残っていない
    # （supervisor.py::cleanup_reservation が正しく解放したことの確認）。
    assert not out_dir.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]


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
