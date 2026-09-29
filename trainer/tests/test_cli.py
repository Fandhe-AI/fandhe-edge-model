"""CLI（`trainer/launch.py train --request ...`）の統合テスト。

Rust 側 CLI が想定する子プロセス呼び出し方（`sys.executable` を引数リストで、
`shell=True` を使わずに起動する。security.md）を模して subprocess で実行する。
`trainer/pyproject.toml` は `package = false`（配布パッケージを持たない）ため、
起動は唯一の起動口 `trainer/launch.py`（`-I` 隔離モード必須。Issue #12）を経由し、
`PYTHONPATH` の手動設定には依存しない（`launch.py` のモジュール docstring 参照）。
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import LABEL_ORDER, TINY_AR_CONFIG, TINY_C1_CONFIG, TINY_CONFIG
from fandhe_edge_trainer import cli, supervisor
from fandhe_edge_trainer.prediction import PREDICTION_FIELDS

_LAUNCH_SCRIPT = str(Path(__file__).resolve().parent.parent / "launch.py")


def _run_cli(
    request_path: Path, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", _LAUNCH_SCRIPT, "train", "--request", str(request_path)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )


def _run_cli_argv(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", _LAUNCH_SCRIPT, *argv],
        capture_output=True,
        text=True,
        timeout=30,
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


def test_cli_train_success_with_autoregressive_kind_emits_single_json_and_exit_0(
    tmp_path: Path,
) -> None:
    """REQ-19b・TASK-19b.1-1・#79 受け入れ条件 1: `kind="autoregressive"` の
    学習リクエストが選択口（`launch.py train` → `resolve_kind`）を経由して
    エラーなく最後まで実行できること（合成データ・CPU）。
    """
    train_path = tmp_path / "train.jsonl"
    _write_train_data(train_path)
    out_dir = tmp_path / "out"
    request = {
        "schema_version": 1,
        "kind": "autoregressive",
        "kind_version": 1,
        "config": TINY_AR_CONFIG,
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
    assert payload["artifact"]["kind"] == "autoregressive"
    assert (out_dir / "artifact.json").exists()
    assert (out_dir / "model.onnx").exists()
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


def test_worker_rejects_non_regular_stdin_without_blocking(tmp_path: Path) -> None:
    """REQ-39: `_worker` は通常契約ではスーパーバイザーから無名一時ファイル
    （通常ファイル）を標準入力として渡されるが、パイプ等の非通常ファイルが
    標準入力に接続された場合は `read()` を試みる前に `invalid_request`
    （exit 64）で拒否し、書き手が現れない標準入力を無期限に待たないこと
    （`cli.py::_read_request_from_stdin` 参照。証拠種別: テストハーネス）。

    **書き手側のパイプを意図的に開いたままにする**（識別力の核心）:
    ここで標準入力をすぐ閉じてしまうと（`Popen.communicate()` は書き込み無しでも
    呼び出し直後に子の標準入力を閉じる）、fstat による事前検査が無い実装でも
    `read()` が EOF ですぐ返り `invalid_request`（exit 64）になってしまい、
    本チェックの有無を区別できない。パイプを開いたまま `proc.wait()` する
    ことで、fstat 検査が無い実装なら `read()` がブロックしたまま
    `TimeoutExpired` になり、本テストが失敗して区別できるようにする。

    **`--lifeline-fd` には検証を通過する本物のパイプの読み取り端を渡す**
    （issue #178 PR #233 レビュー再々指摘 P0 で `_start_lifeline_thread` が
    起動時に `--lifeline-fd` を検証するようになったため、実在しないダミー
    fd を渡すと本チェックへ到達する前に lifeline 側の `invalid_request` で
    拒否されてしまい、本テストが検証したい stdin の S_ISREG 検査を通らなく
    なる。本テストの間は書き込み端を閉じずに開いたままにして lifeline の
    EOF を発生させない）。`--out-fd` は本チェックより前に使われないため、
    実在しないダミー fd 番号（3）のままでよい。
    """
    lifeline_read_fd, lifeline_write_fd = os.pipe()
    # `supervisor.worker_argv` を再利用し、実際の起動経路（`-I` 付き
    # `trainer/launch.py` 経由）と同じ argv で検証する（Issue #12）。
    # `start_new_session=True`: 本番の起動契約（`_worker` は常に別
    # プロセスグループのリーダーとして起動される。`supervisor.py`
    # モジュール docstring「lifeline」節参照）に合わせる。lifeline
    # スレッドは自分自身の pgid へ `killpg` するため、これを指定しないと
    # 万一 lifeline スレッドが誤発火した場合に本テストプロセス（pytest）
    # 自身の pgid を巻き込みうる。
    proc = subprocess.Popen(
        supervisor.worker_argv(3, lifeline_read_fd),
        stdin=subprocess.PIPE,  # 通常ファイルではない（S_ISREG ではない）標準入力
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(lifeline_read_fd,),
        start_new_session=True,
    )
    os.close(lifeline_read_fd)  # 子へ複製済み。親側の複製は不要。
    try:
        # 標準入力（パイプの書き込み側）はここでは閉じない。fstat による
        # 事前検査があれば read() を試みる前に拒否されるため、これだけで
        # 待たずに終了するはずである。
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        raise
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        os.close(lifeline_write_fd)

    stdout = proc.stdout.read() if proc.stdout is not None else ""
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    assert proc.returncode == 64, (proc.returncode, stdout, stderr)
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_request"
    # message の内容まで確認し、（偶然 EOF で早期に invalid_request になる
    # 経路ではなく）fstat による S_ISREG 検査で拒否されたことを裏付ける。
    assert payload["message"] == "stdin must be a regular file"


def test_cli_train_ignores_polluted_pythonpath(tmp_path: Path) -> None:
    """受け入れ条件 3（Issue #12）: 呼び出し元の `PYTHONPATH` に、import されると
    即座に異常終了する偽の `mlx` パッケージを置いた状態でも、`kind="c3"`
    （`kinds/c3.py` が `import mlx.core as mx` を eager import する）の学習が
    exit 0 で成功すること。

    `trainer/launch.py` 経由の起動（`-I` 隔離モード）が呼び出し元の
    `PYTHONPATH` を無視して venv の本物の `mlx` を解決するため、この汚染は
    影響しないはずである。`-I` が外れる退行が起きれば、偽の `mlx`
    （`sys.exit(1)` するだけの `__init__.py`）が先に解決されて異常終了し、
    本テストが検出する（証拠種別: テストハーネス）。

    偽の `mlx` は学習を行う子プロセス（`_worker`）で初めて import されるため、
    それだけでは公開プロセス（`trainer/launch.py train`）側の `-I` 欠落を
    検出できない（PR #16 レビュー）。そこで同じ汚染ディレクトリへ
    `sitecustomize.py` も置く。`site` モジュールは起動時に `sys.path` 上の
    `sitecustomize` を import するため、`-I` の無いプロセスが 1 つでもあれば
    （公開プロセス・`_worker` のどちらでも）マーカーファイルが作られる。
    `-I` 付きでは `PYTHONPATH` が `sys.path` に入らず作られないことを実機で
    確認済み（証拠種別: 実機）。
    """
    fake_mlx_root = tmp_path / "polluted-pythonpath"
    fake_mlx_pkg = fake_mlx_root / "mlx"
    fake_mlx_pkg.mkdir(parents=True)
    (fake_mlx_pkg / "__init__.py").write_text(
        "import sys\nsys.exit('fake mlx package must never be imported')\n",
        encoding="utf-8",
    )
    # `-I` の無いプロセスが起動した時点で痕跡を残す（マーカーのパスは
    # 環境変数経由にせずリテラルで埋め込み、検出経路を 1 つに保つ）。
    sitecustomize_marker = tmp_path / "sitecustomize-imported"
    (fake_mlx_root / "sitecustomize.py").write_text(
        f"with open({str(sitecustomize_marker)!r}, 'a', encoding='utf-8') as f:\n"
        "    f.write('imported\\n')\n",
        encoding="utf-8",
    )

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
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    polluted_env = {**os.environ, "PYTHONPATH": str(fake_mlx_root)}
    result = _run_cli(request_path, env=polluted_env)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    # 公開プロセス・`_worker` のいずれも汚染された `PYTHONPATH` を読んでいない。
    assert not sitecustomize_marker.exists()
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "ok"
    assert (out_dir / "artifact.json").exists()
    assert (out_dir / "model.onnx").exists()


# --------------------------------------------------------------------------
# lifeline fd の起動時検証（issue #178 PR #233 レビュー再々指摘 P0
# 「lifeline fd が無効な場合、`_worker` が監視なしで学習を続けてしまう」）。
#
# `cli.main` を直接（同一プロセス内で）呼び、`run_worker_train` を
# 呼び出しがあれば失敗させるダミーへ差し替えることで、「学習が一切
# 開始されないこと」を無効な lifeline fd の 3 パターン（無効な fd・
# 非パイプ fd・未指定）それぞれについて直接的に確認する。
# --------------------------------------------------------------------------


def _fail_if_called(_out_fd: int) -> int:
    raise AssertionError("run_worker_train must not be called when lifeline fd is invalid")


def test_lifeline_fd_invalid_fails_before_training_starts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "run_worker_train", _fail_if_called)
    # 現在のプロセスでは開かれていないであろう、大きな fd 番号を渡す。
    exit_code = cli.main(["_worker", "--out-fd", "3", "--lifeline-fd", "999999"])
    assert exit_code == 64
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_request"
    assert payload["message"] == "lifeline fd not stat-able: OSError"


def test_lifeline_fd_non_pipe_fails_before_training_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "run_worker_train", _fail_if_called)
    regular_file = tmp_path / "not-a-pipe.txt"
    regular_file.write_text("x", encoding="utf-8")
    fd = os.open(regular_file, os.O_RDONLY)
    try:
        exit_code = cli.main(["_worker", "--out-fd", "3", "--lifeline-fd", str(fd)])
        assert exit_code == 64
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["status"] == "error"
        assert payload["code"] == "invalid_request"
        assert payload["message"] == "lifeline fd must be a pipe"
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def test_lifeline_fd_missing_argument_fails_before_training_starts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "run_worker_train", _fail_if_called)
    exit_code = cli.main(["_worker", "--out-fd", "3"])
    assert exit_code == 64
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "error"
    assert payload["code"] == "invalid_request"


def test_lifeline_watch_loop_read_exception_triggers_group_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_lifeline_watch_loop` は `read` が例外を送出した場合も EOF と同じ
    扱い（自分自身のプロセスグループへ `SIGKILL`）にすること（issue #178
    PR #233 レビュー再々指摘 P0。黙って戻って監視を止めたままにしない）。
    """
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(cli.os, "getpgrp", lambda: 4242)
    monkeypatch.setattr(cli.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))

    class _RaisingPipe:
        def read(self, size: int) -> bytes:
            raise OSError("simulated read failure")

        def __enter__(self) -> _RaisingPipe:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    cli._lifeline_watch_loop(_RaisingPipe())
    assert calls == [(4242, signal.SIGKILL)]


# ---- 学習直後の validation 予測（REQ-18・REQ-27。issue #84 PR #238・選択肢 2）----


def _write_validation_request(tmp_path: Path, kind: str, config: dict, extra: dict) -> Path:
    rows = []
    for i in range(12):
        rows.append({"input": f"alpha alpha beta gamma {i}", "label": "cat_a"})
        rows.append({"input": f"delta delta epsilon zeta {i}", "label": "cat_b"})
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    request = {
        "schema_version": 1,
        "kind": kind,
        "kind_version": 1,
        "config": config,
        "label_order": LABEL_ORDER,
        "max_bytes": 64,
        "seed": 0,
        "device": "cpu",
        "root": str(tmp_path),
        "train_path": "train.jsonl",
        "out_dir": "out",
        **extra,
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("kind", "config"),
    [("c1", TINY_C1_CONFIG), ("c3", TINY_CONFIG), ("autoregressive", TINY_AR_CONFIG)],
)
def test_req18_cli_returns_validation_predictions_in_request_order(
    tmp_path: Path, kind: str, config: dict
) -> None:
    """REQ-18・REQ-27: `launch.py train` がリクエストの `validation_inputs`
    （`{id, input}` のみ）を学習直後に予測し、結果 JSON（1 行）の
    `validation_predictions` として入力と同じ順序・件数で返す。
    """
    validation = [
        {"id": "v-b", "input": "delta delta epsilon zeta 7"},
        {"id": "v-a", "input": "alpha alpha beta gamma 7"},
    ]
    request_path = _write_validation_request(
        tmp_path, kind, config, {"validation_inputs": validation}
    )
    result = _run_cli(request_path)
    assert result.returncode == 0, (result.stdout[:500], result.stderr[:2000])
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "ok"
    predictions = payload["validation_predictions"]
    assert [p["id"] for p in predictions] == ["v-b", "v-a"]
    assert all(list(p) == list(PREDICTION_FIELDS) for p in predictions)
    assert all(p["status"] in {"ok", "error"} for p in predictions)
    assert all(p["predicted_label"] in [*LABEL_ORDER, None] for p in predictions)
    if kind != "autoregressive":
        # c1・c3 は合成データを確実に分離できる。
        assert [p["predicted_label"] for p in predictions] == ["cat_b", "cat_a"]


def test_req18_cli_without_validation_inputs_has_no_predictions_key(tmp_path: Path) -> None:
    """`validation_inputs` を付けなければ結果に `validation_predictions` は現れない
    （既存の消費者を壊さない）。
    """
    result = _run_cli(_write_validation_request(tmp_path, "c3", TINY_CONFIG, {}))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert "validation_predictions" not in payload


def test_req27_cli_rejects_gold_label_in_validation_inputs(tmp_path: Path) -> None:
    """REQ-27: `validation_inputs` の要素に正解ラベルを混ぜたリクエストは、学習に
    入る前に `invalid_request`／64 で拒否される。
    """
    validation = [{"id": "v", "input": "alpha", "label": "cat_a"}]
    request_path = _write_validation_request(
        tmp_path, "c3", TINY_CONFIG, {"validation_inputs": validation}
    )
    result = _run_cli(request_path)
    assert result.returncode == 64, (result.stdout[:500], result.stderr[:500])
    payload = json.loads(result.stdout.strip())
    assert payload["code"] == "invalid_request"
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------
# 協調キャンセル `--cancel-on-stdin-eof`（REQ-34・TASK-34.1-2・issue #145）
# --------------------------------------------------------------------------


def _write_cancel_request(tmp_path: Path, config: dict) -> Path:
    _write_train_data(tmp_path / "train.jsonl")
    request = {
        "schema_version": 1,
        "kind": "c3",
        "kind_version": 1,
        "config": config,
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
    return request_path


def test_req34_cancel_flag_rejects_non_pipe_stdin_before_reserving_out_dir(
    tmp_path: Path,
) -> None:
    """REQ-34・REQ-39: フラグ付きで標準入力がパイプでなければ（通常ファイルは
    即 EOF になり自己キャンセルしてしまう）、`out_dir` を予約せず
    `invalid_request`（exit 64）で拒否する。"""
    request_path = _write_cancel_request(tmp_path, TINY_CONFIG)
    stdin_file = tmp_path / "stdin.txt"
    stdin_file.write_text("", encoding="utf-8")
    with stdin_file.open("rb") as stdin:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                _LAUNCH_SCRIPT,
                "train",
                "--request",
                str(request_path),
                "--cancel-on-stdin-eof",
            ],
            stdin=stdin,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    payload = _assert_single_json_error(result)
    assert payload["message"] == "cancel channel (stdin) must be a pipe"
    assert not (tmp_path / "out").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]


def test_req34_train_without_cancel_flag_passes_no_cancel_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-34: フラグが無ければ標準入力を見ず、`cancel_event` は `None`
    （単独起動の挙動は変えない）。"""
    seen: dict[str, object] = {}

    def _fake_run(request_path: Path, *, cancel_event: object = "unset") -> int:
        seen["cancel_event"] = cancel_event
        return 0

    monkeypatch.setattr(supervisor, "run_supervised_train", _fake_run)
    assert cli.main(["train", "--request", str(tmp_path / "request.json")]) == 0
    assert seen["cancel_event"] is None


def test_req34_cancel_on_stdin_eof_leaves_nothing_published(tmp_path: Path) -> None:
    """REQ-34・TASK-34.1-2: 実 `_worker`（c3・CPU）の学習中に標準入力（キャンセル用
    パイプ）を閉じると、supervisor は worker を止めて予約を解放し、`out_dir`・
    作業用一時ディレクトリのいずれも残らない。学習は完走しない長さ（epochs を
    大きく取る）にして、確定前にキャンセルが届くことを保証する。"""
    request_path = _write_cancel_request(tmp_path, dict(TINY_CONFIG, epochs=1_000_000))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-I",
            _LAUNCH_SCRIPT,
            "train",
            "--request",
            str(request_path),
            "--cancel-on-stdin-eof",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        out_dir = tmp_path / "out"
        deadline = time.monotonic() + 30
        while not out_dir.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert out_dir.is_dir(), "out_dir was not reserved in time"
        assert proc.stdin is not None
        proc.stdin.close()
        assert proc.stdout is not None
        stdout = proc.stdout.read()
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert proc.returncode == 70
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "status": "error",
        "code": "runtime_error",
        "message": "training cancelled by caller",
    }
    assert not out_dir.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out.tmp-")]
