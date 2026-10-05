"""実機での動作確認スクリプト（scripts/real_machine_check_record.py）の単体テスト。

REQ-33（記録・stdout に利用者の値を出さない）・REQ-28（バッチと単体の推論の全件一致の突き合わせ）・
REQ-39（子プロセスの上限時間・出力サイズ上限）・REQ-27（評価データを推論の確認に使わない）。
証拠種別: テストハーネス。実機での測定結果ではない（実機の確定は人が行う）。
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "real_machine_check_record.py"

spec = importlib.util.spec_from_file_location("real_machine_check_record", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules["real_machine_check_record"] = mod
spec.loader.exec_module(mod)


def test_redact_value_is_removed_in_favor_of_the_allowlist() -> None:
    """REQ-33: 拒否リスト方式の `redact_value` は残さない（許可リスト方式へ置き換えた）。"""
    assert not hasattr(mod, "redact_value")
    assert not hasattr(mod, "DENY_KEYS")


def test_sanitize_record_allows_only_system_library_paths() -> None:
    """REQ-33: D のライブラリ名は /usr/lib/・/System/ 始まりだけが通り、他のパスは伏せる。"""
    rec = {
        "items": {
            "D": {
                "direct_libraries": [
                    "/usr/lib/libSystem.B.dylib",
                    "/System/Library/Frameworks/X.framework/X",
                    "/opt/homebrew/lib/libz.dylib",
                    "/usr/lib/../../etc/passwd",
                    "/usr/lib/libx.dylib\n",
                    "/usr/lib/" + "a" * 120,
                ]
            }
        },
        "other": {"p": "/usr/lib/libSystem.B.dylib", "cwd": "/Users/someone/work"},
    }
    out = mod.sanitize_record(rec)
    assert out["items"]["D"]["direct_libraries"] == [
        "/usr/lib/libSystem.B.dylib",
        "/System/Library/Frameworks/X.framework/X",
        "<redacted>",
        "<redacted>",
        "<redacted>",
        "<redacted>",
    ]
    assert out["other"] == {"p": "<redacted>", "cwd": "<redacted>"}


def _capacity(total: int, parts: dict[str, int]) -> dict[str, Any]:
    return {
        "capacity": {
            "total_bytes": total,
            "limit_bytes": 40000000,
            "exceeded": False,
            "components": {k: {"bytes": v, "file_count": 1} for k, v in parts.items()},
        }
    }


def test_capacity_summary_requires_the_five_components() -> None:
    """REQ-30: 容量内訳は 5 項目が揃ったときだけ要約し、揃わなければ None。"""
    parts = {
        "weights": 4568,
        "vocab_or_feature_transform": 0,
        "label_table": 328,
        "calibration": 0,
        "metadata": 531,
    }
    cap = mod.capacity_summary(_capacity(5427, parts))
    assert cap is not None
    assert cap["total_bytes"] == 5427
    assert sum(c["bytes"] for c in cap["components"].values()) == 5427
    missing = dict(parts)
    del missing["metadata"]
    assert mod.capacity_summary(_capacity(4896, missing)) is None
    assert mod.capacity_summary({"capacity": None}) is None


def test_parse_make_ci_log_counts_skip_rust_and_pytest() -> None:
    """A: skip 行・Rust の test result 合計・pytest の最終行を数える。"""
    log = "\n".join(
        [
            "skip: something",
            "test result: ok. 10 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out",
            "test result: ok. 5 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out",
            "skip: another",
            "==== 120 passed, 3 skipped in 4.50s ====",
        ]
    )
    assert mod.parse_make_ci_log(log) == {
        "skip_lines": 2,
        "rust_tests": {"passed": 15, "failed": 1, "ignored": 2},
        "pytest": {"passed": 120, "skipped": 3, "failed": 0},
    }


def test_count_read_output_separates_incomplete_from_plain() -> None:
    """F: `ReadOutputIncomplete` と、それが続かない `ReadOutput` を区別する（#346）。"""
    assert mod.count_read_output("error: ReadOutputIncomplete { .. }") == (1, 0)
    assert mod.count_read_output("error: ReadOutput(Io)") == (0, 1)
    assert mod.count_read_output("ReadOutputIncomplete and ReadOutput(x)") == (1, 1)
    assert mod.count_read_output("all good") == (0, 0)


def test_compare_infer_reports_counts_without_ids_in_fields() -> None:
    """REQ-28: バッチと単体を id で突き合わせ、不一致の id は戻り値の別キーへ分ける。"""
    batch = {
        "a": {"predicted_label": "x", "scores": {"x": 0.5, "y": 0.5}},
        "b": {"predicted_label": "x", "scores": {"x": 0.7, "y": 0.3}},
    }
    single = {
        "a": {"predicted_label": "x", "scores": {"x": 0.5, "y": 0.5}},
        "b": {"predicted_label": "y", "scores": {"x": 0.6, "y": 0.4}},
    }
    cmp = mod.compare_infer(batch, single)
    assert cmp["label_match"] == 1
    assert cmp["label_mismatch"] == 1
    assert cmp["scores_exact_match"] == 1
    assert cmp["mismatch_ids"] == ["b"]
    assert cmp["max_abs_score_diff"] == pytest.approx(0.1, abs=1e-9)


def test_parse_otool_libraries_drops_first_line_and_parenthesized_tail() -> None:
    """D: otool -L の 1 行目と、括弧以降（互換バージョン）を除く。"""
    text = (
        "target/release/fandhe-edge:\n"
        "\t/usr/lib/libiconv.2.dylib (compatibility version 7.0.0, current version 7.0.0)\n"
        "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1351.0.0)\n"
    )
    assert mod.parse_otool_libraries(text) == [
        "/usr/lib/libiconv.2.dylib",
        "/usr/lib/libSystem.B.dylib",
    ]


def test_run_cmd_kills_on_timeout_and_on_output_limit(tmp_path: Path) -> None:
    """REQ-39: 期限切れ・出力サイズ超過の子プロセスはグループごと止め、reason を返す。"""
    r = mod.run_cmd(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        tmp_path / "o",
        tmp_path / "e",
        1,
        1024,
        1024,
    )
    assert (r.exit_code, r.reason) == (None, "timeout")
    flood = "import sys\nwhile True:\n    sys.stdout.write('x' * 4096)\n    sys.stdout.flush()\n"
    r = mod.run_cmd(
        [sys.executable, "-c", flood], tmp_path, tmp_path / "o", tmp_path / "e", 20, 1024, 1024
    )
    assert r.reason == "output_limit"
    r = mod.run_cmd(["/nonexistent/cmd"], tmp_path, tmp_path / "o", tmp_path / "e", 5, 10, 10)
    assert (r.exit_code, r.reason) == (None, "spawn_error")


def _fake_cli(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "fake-cli"
    p.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def _ctx(tmp_path: Path, cli: Path) -> Any:
    return mod.Ctx(
        repo=REPO,
        work=tmp_path,
        bin=cli,
        make_cmd="make",
        cargo_cmd="cargo",
        p95_limit_us=50000,
        package_limit_bytes=1000,
        quiet_machine=False,
        repeat=1,
        harness=True,
        offline_env={"CARGO_NET_OFFLINE": "true"},
    )


def test_failed_step_records_exit_code_and_message_size_and_hash(tmp_path: Path) -> None:
    """REQ-33・REQ-21: 失敗した工程は exit_code・code・message の大きさとハッシュだけを記録する。"""
    cli = _fake_cli(
        tmp_path,
        'echo \'{"code":"invalid_input","message":"cannot read /private/secret/file"}\'\nexit 64\n',
    )
    d = tmp_path / "B"
    assert mod.stage_inputs(_ctx(tmp_path, cli), d, None)
    steps, _pkg, failure = mod.run_pipeline(_ctx(tmp_path, cli), d, "sample", {0})
    assert [s["step"] for s in steps] == ["register"]
    assert failure == {
        "status": "failed",
        "reason": "unexpected_exit_code",
        "step": "register",
        "exit_code": 64,
        "code": "invalid_input",
        "message_bytes": 32,
        "message_sha256": hashlib.sha256(b"cannot read /private/secret/file").hexdigest(),
    }
    assert "message" not in failure


def test_invalid_json_stdout_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    """REQ-21: JSON として読めない stdout は exit 0 でも失敗（fail-closed）。"""
    cli = _fake_cli(tmp_path, "echo not-json\nexit 0\n")
    d = tmp_path / "B"
    assert mod.stage_inputs(_ctx(tmp_path, cli), d, None)
    _steps, _pkg, failure = mod.run_pipeline(_ctx(tmp_path, cli), d, "sample", {0})
    assert failure is not None
    assert failure["reason"] == "invalid_json"


def test_render_markdown_has_no_path_strings_outside_library_names() -> None:
    """REQ-33: record.md の生成元は sanitize 済みの record。`/` を含む語は D のライブラリ名だけ。"""
    items = {n: {"status": "not_run", "reason": "not_selected"} for n in mod.ITEM_ORDER}
    items["D"] = {
        "status": "ok",
        "exit_code": 0,
        "skip_lines": 0,
        "env_i_tests_ok": 3,
        "direct_libraries": ["/usr/lib/libSystem.B.dylib", "/home/x/libbad.dylib"],
    }
    rec = mod.sanitize_record(
        {
            "schema": mod.SCHEMA,
            "evidence_hint": "requires_human_review",
            "environment": {"hw_model": None, "cpu": "cpu|x"},
            "inputs": {},
            "options": {"items": ["D"]},
            "items": items,
        }
    )
    rec["schema"] = mod.SCHEMA  # run() と同様に、固定定数の schema は伏せ処理後に戻す
    md = mod.render_markdown(rec)
    assert "/home/x" not in md
    assert "人が確認して記入" in md
    words = [w.strip('`",[]') for w in md.split() if "/" in w]
    assert sorted(set(words)) == ["/usr/lib/libSystem.B.dylib", "real-machine-check/1"]
    assert json.dumps(rec)  # JSON として直列化できる


def _script(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


CARGO_RUST_LINE = (
    "test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s"
)


@pytest.mark.parametrize("pytest_first", [True, False])
def test_parse_make_ci_log_distinguishes_cargo_and_pytest_lines(pytest_first: bool) -> None:
    """A: cargo の `test result:` 行を pytest として数えない（Rust・pytest の両順で同じ結果）。"""
    pyline = "==================== 126 passed, 2 skipped in 2.45s ===================="
    lines = [pyline, CARGO_RUST_LINE] if pytest_first else [CARGO_RUST_LINE, pyline]
    assert mod.parse_make_ci_log("\n".join(lines)) == {
        "skip_lines": 0,
        "rust_tests": {"passed": 12, "failed": 0, "ignored": 0},
        "pytest": {"passed": 126, "skipped": 2, "failed": 0},
    }
    bare = mod.parse_make_ci_log(CARGO_RUST_LINE + "\n126 passed in 2.45s\n")
    assert bare["pytest"] == {"passed": 126, "skipped": 0, "failed": 0}
    assert mod.parse_make_ci_log(CARGO_RUST_LINE)["pytest"] is None


def test_judge_make_ci_requires_tests_without_skips_or_failures() -> None:
    """A: skip 行・0 件・失敗は ok にしない（skip を検証済みと扱わない）。"""
    ok = mod.parse_make_ci_log(CARGO_RUST_LINE + "\n=== 3 passed, 1 skipped in 0.1s ===\n")
    assert mod.judge_make_ci(ok) is None  # pytest の skipped は記録のみ
    assert mod.judge_make_ci(dict(ok, skip_lines=1)) == "skipped"
    zero = mod.parse_make_ci_log("=== 3 passed in 0.1s ===\n")
    assert mod.judge_make_ci(zero) == "no_test_results"
    no_py = mod.parse_make_ci_log(CARGO_RUST_LINE)
    assert mod.judge_make_ci(no_py) == "no_test_results"
    failed = mod.parse_make_ci_log(CARGO_RUST_LINE + "\n=== 1 failed, 3 passed in 0.1s ===\n")
    assert mod.judge_make_ci(failed) == "test_failures"


def test_item_a_fails_when_make_ci_only_skips(tmp_path: Path) -> None:
    """A: 終了コード 0 でも skip だけなら failed（reason: skipped）。"""
    make = _script(tmp_path, "fake-make", 'echo "skip: nothing ran"\nexit 0\n')
    ctx = _ctx(tmp_path / "w", _fake_cli(tmp_path, "exit 0\n"))
    ctx.work.mkdir()
    ctx.make_cmd = str(make)
    res = mod.item_a(ctx)
    assert (res["status"], res["reason"], res["skip_lines"]) == ("failed", "skipped", 1)


def test_item_d_requires_env_i_tests_and_no_skips(tmp_path: Path) -> None:
    """D: `ok: req32_` が 0 件、または skip 行があれば failed。"""
    ctx = _ctx(tmp_path / "w", _fake_cli(tmp_path, "exit 0\n"))
    ctx.work.mkdir()
    ctx.make_cmd = str(_script(tmp_path, "m0", "echo 'skip: x'\nexit 0\n"))
    assert mod.item_d(ctx)["reason"] == "skipped"
    shutil.rmtree(ctx.work / "D")
    ctx.make_cmd = str(_script(tmp_path, "m1", "echo done\nexit 0\n"))
    assert mod.item_d(ctx)["reason"] == "no_test_results"


def _list_script(n: int = 12) -> str:
    """`cargo test -- --list` に n 件で答える偽 cargo の前置き（他の引数では何もしない）。"""
    echoes = "".join(f"echo 't{i}: test'; " for i in range(n))
    return f'case "$*" in *--list*) {echoes}exit 0;; esac\n'


def test_item_f_rejects_zero_tests_and_failed_prebuild(tmp_path: Path) -> None:
    """F: 0 件実行は pass に数えず no_tests に計上する。`--no-run` の失敗は build_failed。"""
    ctx = _ctx(tmp_path / "w", _fake_cli(tmp_path, "exit 0\n"))
    ctx.work.mkdir()
    ctx.repeat = 2
    zero = "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
    ctx.cargo_cmd = str(_script(tmp_path, "c0", _list_script() + f"echo '{zero}'\nexit 0\n"))
    res = mod.item_f(ctx)
    assert (res["passed"], res["failed"], res["no_tests"]) == (0, 2, 2)
    shutil.rmtree(ctx.work / "F")
    ctx.cargo_cmd = str(
        _script(tmp_path, "c1", 'case "$*" in *--no-run*) exit 101;; esac\nexit 0\n')
    )
    assert mod.item_f(ctx) == {"status": "failed", "reason": "build_failed", "exit_code": 101}
    shutil.rmtree(ctx.work / "F")
    good = CARGO_RUST_LINE
    ctx.cargo_cmd = str(_script(tmp_path, "c2", _list_script() + f"echo '{good}'\nexit 0\n"))
    res = mod.item_f(ctx)
    assert (res["status"], res["passed"], res["no_tests"]) == ("ok", 2, 0)


def test_compare_infer_treats_nonfinite_scores_and_bad_labels_as_mismatch() -> None:
    """REQ-28: NaN・無限大のスコア、str でない label は不一致。max diff は有限の差だけ。"""
    nan, inf = float("nan"), float("inf")
    ok = {"predicted_label": "x", "scores": {"x": 0.5}}
    batch = {
        "a": ok,
        "b": {"predicted_label": "x", "scores": {"x": nan}},
        "c": {"predicted_label": "x", "scores": {"x": inf}},
        "d": {"predicted_label": 3, "scores": {"x": 0.5}},
    }
    single = {
        "a": ok,
        "b": {"predicted_label": "x", "scores": {"x": nan}},
        "c": {"predicted_label": "x", "scores": {"x": inf}},
        "d": {"predicted_label": 3, "scores": {"x": 0.5}},
    }
    cmp = mod.compare_infer(batch, single)
    assert cmp["label_match"] == 3
    assert cmp["label_mismatch"] == 1
    assert cmp["scores_nonfinite"] == 2
    assert sorted(cmp["mismatch_ids"]) == ["b", "c", "d"]
    assert cmp["max_abs_score_diff"] == 0.0


def test_sanitize_record_nulls_nonfinite_floats_and_json_rejects_nan() -> None:
    """記録に NaN・無限大は書かない（null にし、allow_nan=False でも直列化できる）。"""
    out = mod.sanitize_record({"a": float("nan"), "b": [float("inf"), 1.5]})
    assert out == {"a": None, "b": [None, 1.5]}
    assert json.dumps(out, allow_nan=False) == '{"a": null, "b": [null, 1.5]}'


def test_sanitize_record_treats_lookalike_slashes_as_path_chars() -> None:
    """REQ-33: U+2215・U+2044・U+FF0F もパス文字として伏せる。キー側のパス文字も伏せる。"""
    obj = {
        "reason": "x\u2215y",
        "step": "x\u2044y",
        "code": "x\uff0fy",
        "name": "x\\y",
        "status": "ok",
        "k\u2215x": "ok",
    }
    assert mod.sanitize_record(obj) == {
        "reason": "<redacted>",
        "step": "<redacted>",
        "code": "<redacted>",
        "name": "<redacted>",
        "status": "ok",
        "<redacted>": "<redacted>",
    }


def test_cell_escapes_markdown_and_html() -> None:
    """record.md のセルは Markdown・HTML として解釈されない。"""
    got = mod._cell("<b>x</b> `y` | a&b\r\nz")
    assert got == "&lt;b&gt;x&lt;/b&gt; &#96;y&#96; &#124; a&amp;b  z"


def test_render_markdown_shows_harness_and_bin_override() -> None:
    """record.md の冒頭に bin_override・evidence_hint・harness かどうかと、差し替えの注意を出す。"""
    items = {n: {"status": "not_run", "reason": "not_selected"} for n in mod.ITEM_ORDER}
    rec = {
        "schema": mod.SCHEMA,
        "evidence_hint": "test_harness",
        "bin_override": True,
        "environment": {},
        "inputs": {},
        "options": {},
        "items": items,
    }
    head = mod.render_markdown(rec).split("## 環境")[0]
    assert "- bin_override: True" in head
    assert "- evidence_hint: test_harness" in head
    assert "- harness（make・cargo の代役）: はい" in head
    assert "CLI を `FANDHE_EDGE_BIN` で差し替えた" in head


def test_find_built_executable_reads_compiler_artifact(tmp_path: Path) -> None:
    """cargo の JSON の compiler-artifact から bin の executable を取る（決め打ちしない）。"""
    exe = tmp_path / "x" / "fandhe-edge"
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8")
    lines = [
        json.dumps({"reason": "compiler-message"}),
        json.dumps(
            {
                "reason": "compiler-artifact",
                "target": {"name": "fandhe-edge", "kind": ["bin"]},
                "executable": str(exe),
            }
        ),
        "not json",
    ]
    assert mod.find_built_executable("\n".join(lines)) == exe
    assert mod.find_built_executable("") is None


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_interrupt_stops_children_and_writes_record(tmp_path: Path, sig: int) -> None:
    """REQ-39: SIGINT・SIGTERM・SIGHUP で子グループを残さず、record を書き exit 70 で終える。"""
    pidfile = tmp_path / "pids"
    make = _script(
        tmp_path,
        "fake-make",
        f'echo $$ >> "{pidfile}"\nsleep 60 &\necho $! >> "{pidfile}"\nwait\n',
    )
    cli = _fake_cli(tmp_path, "exit 0\n")
    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ, FANDHE_EDGE_MAKE_CMD=str(make))
    proc = subprocess.Popen(  # noqa: S603  テスト用に自リポジトリのスクリプトを引数リストで起動する
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--repo-root",
            str(REPO),
            "--work-dir",
            str(work),
            "--bin",
            str(cli),
            "--bin-override",
            "--items",
            "D,F",
            "--repeat",
            "1",
            "--p95-limit-us",
            "50000",
            "--package-limit-bytes",
            "1000",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if pidfile.exists() and len(pidfile.read_text().split()) >= 2:
            break
        time.sleep(0.05)
    pids = [int(x) for x in pidfile.read_text().split()]
    proc.send_signal(sig)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 70
    assert json.loads(out) == {
        "code": "runtime_error",
        "message": "interrupted",
        "record": "record.json",
    }
    time.sleep(0.2)
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    rec = json.loads((work / "record.json").read_text())
    assert rec["items"]["D"] == {"status": "failed", "reason": "interrupted"}
    assert rec["items"]["F"] == {"status": "not_run", "reason": "interrupted"}
    assert rec["options"]["cargo_offline"] is True
    assert rec["options"]["with_ci"] is False


@pytest.fixture(autouse=True)
def _reset_interrupt_mark() -> Any:
    """モジュールの中断の印を、各テストの前後で下ろす（テスト間で引き継がない）。"""
    mod._interrupt_requested = False
    mod._signal_count = 0
    mod._active_pgid = None
    mod._child_may_remain = False
    yield
    mod._interrupt_requested = False
    mod._signal_count = 0
    mod._active_pgid = None
    mod._child_may_remain = False


def _wait_for(pred: Any, seconds: float = 30.0) -> bool:
    """期限つきのポーリング。固定の sleep に頼らず、条件が成り立つまで待つ。"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return bool(pred())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _pids(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split()] if path.exists() else []


def test_on_signal_only_sets_the_mark_and_run_clears_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-21・REQ-39: ハンドラは例外を投げず印を立てるだけ。`run()` の冒頭で印が下りる。"""
    saved = {s: signal.getsignal(s) for s in mod.INTERRUPT_SIGNALS}
    mod._on_signal(signal.SIGTERM, None)
    assert mod._interrupt_requested is True
    assert {s: signal.getsignal(s) for s in mod.INTERRUPT_SIGNALS} == saved
    with pytest.raises(mod.Interrupted):
        mod.check_interrupt()
    seen: list[bool] = []

    def spy(repo: Path) -> None:
        seen.append(mod._interrupt_requested)

    monkeypatch.setattr(mod, "collect_inputs", spy)
    (tmp_path / "w").mkdir()
    args = _ns(
        repo_root=str(REPO),
        work_dir=str(tmp_path / "w"),
        bin="x",
        bin_override=True,
        items="B",
        repeat=1,
        p95_limit_us=1,
        package_limit_bytes=1,
        quiet_machine=False,
        with_ci=False,
    )
    try:
        rc = mod.run(args)
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
    assert (rc, seen) == (70, [False])
    assert json.loads(capsys.readouterr().out)["message"] == "cannot read fixture inputs"


def test_run_cmd_does_not_start_the_child_when_the_mark_is_set(tmp_path: Path) -> None:
    """REQ-39: 印が立っていれば子を起動せずに `Interrupted`（子が書くはずの印ファイルが無い）。"""
    marker = tmp_path / "started"
    mod._interrupt_requested = True
    with pytest.raises(mod.Interrupted):
        _run(tmp_path, ["/bin/sh", "-c", f'echo x > "{marker}"'])
    assert not marker.exists()


def test_run_cmd_raises_interrupted_after_stopping_child_and_grandchild(tmp_path: Path) -> None:
    """REQ-39: 子の実行中に印が立つと、子と孫を止めて回収してから `Interrupted` を送出する。

    印から送出までの時間（5 秒未満）も確かめる。
    """
    import threading

    pidfile = tmp_path / "pids"
    script = f'echo $$ >> "{pidfile}"; sleep 60 & echo $! >> "{pidfile}"; wait'

    tripped_at: list[float] = []

    def trip() -> None:
        _wait_for(lambda: len(_pids(pidfile)) >= 2)
        tripped_at.append(time.monotonic())
        mod._on_signal(signal.SIGTERM, None)

    t = threading.Thread(target=trip)
    t.start()
    try:
        with pytest.raises(mod.Interrupted):
            _run(tmp_path, ["/bin/sh", "-c", script])
        raised_at = time.monotonic()
    finally:
        t.join(timeout=30)
    pids = _pids(pidfile)
    assert len(pids) == 2
    # 待機ループが印を見て抜けること。_run の期限（20 秒）や子の sleep 60 に頼った停止と区別する
    assert len(tripped_at) == 1
    assert raised_at - tripped_at[0] < 5.0
    assert _wait_for(lambda: not any(_alive(p) for p in pids), 10)


def _run_script(args_extra: list[str], work: Path, env: dict[str, str]) -> Any:
    return subprocess.Popen(  # noqa: S603  テスト用に自リポジトリのスクリプトを引数リストで起動する
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--repo-root",
            str(REPO),
            "--work-dir",
            str(work),
            "--repeat",
            "1",
            "--p95-limit-us",
            "50000",
            "--package-limit-bytes",
            "1000",
            *args_extra,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def test_interrupt_during_cli_build_writes_a_skeleton_record(tmp_path: Path) -> None:
    """REQ-21・REQ-39: CLI のビルド中の SIGTERM でも record を書き、項目は not_run/interrupted。"""
    pidfile = tmp_path / "pids"
    cargo = _script(
        tmp_path,
        "fake-cargo",
        f'echo $$ >> "{pidfile}"\nsleep 60 &\necho $! >> "{pidfile}"\nwait\n',
    )
    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ, FANDHE_EDGE_CARGO_CMD=str(cargo))
    proc = _run_script(["--items", "B,C"], work, env)
    try:
        assert _wait_for(lambda: len(_pids(pidfile)) >= 2)
        pids = _pids(pidfile)
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 70
    assert json.loads(out) == {
        "code": "runtime_error",
        "message": "interrupted",
        "record": "record.json",
    }
    assert _wait_for(lambda: not any(_alive(p) for p in pids), 10)
    rec = json.loads((work / "record.json").read_text())
    assert rec["environment"] is None
    assert rec["inputs"]["train_records"] > 0
    assert rec["schema"] == "real-machine-check/1"
    interrupted = {"status": "not_run", "reason": "interrupted"}
    unselected = {"status": "not_run", "reason": "not_selected"}
    assert rec["items"] == {
        "A": unselected,
        "B": interrupted,
        "C": interrupted,
        "D": unselected,
        "E": unselected,
        "F": unselected,
    }
    assert "(not collected)" in (work / "record.md").read_text()


def _run_in_process(tmp_path: Path, items: str, fake_item: Any, monkeypatch: Any) -> Any:
    saved = {s: signal.getsignal(s) for s in mod.INTERRUPT_SIGNALS}
    monkeypatch.setattr(mod, "run_item", fake_item)
    (tmp_path / "w").mkdir()
    args = _ns(
        repo_root=str(REPO),
        work_dir=str(tmp_path / "w"),
        bin=str(_fake_cli(tmp_path, "exit 0\n")),
        bin_override=True,
        items=items,
        repeat=1,
        p95_limit_us=1,
        package_limit_bytes=1,
        quiet_machine=False,
        with_ci=False,
    )
    try:
        rc = mod.run(args)
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
    return rc, json.loads((tmp_path / "w" / "record.json").read_text())


def test_mark_after_an_item_stops_the_next_item_and_keeps_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-21: 項目の合間に印が立てば、次の項目は not_run/interrupted。終えた項目は残す。"""

    def fake(ctx: Any, name: str, b_ok: bool) -> Any:
        mod._on_signal(signal.SIGTERM, None)
        return {"status": "ok"}, True

    rc, rec = _run_in_process(tmp_path, "B,C", fake, monkeypatch)
    assert rc == 70
    assert json.loads(capsys.readouterr().out)["message"] == "interrupted"
    assert rec["items"]["B"] == {"status": "ok"}
    assert rec["items"]["C"] == {"status": "not_run", "reason": "interrupted"}


def test_mark_after_all_items_still_makes_the_result_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-21: 全項目の完了後・記録の書き出し前に印が立てば、結果は中断（exit 70）。項目は残す。"""

    def fake(ctx: Any, name: str, b_ok: bool) -> Any:
        mod._on_signal(signal.SIGTERM, None)
        return {"status": "ok"}, True

    rc, rec = _run_in_process(tmp_path, "B", fake, monkeypatch)
    assert rc == 70
    assert json.loads(capsys.readouterr().out) == {
        "code": "runtime_error",
        "message": "interrupted",
        "record": "record.json",
    }
    assert rec["items"]["B"] == {"status": "ok"}


def test_sanitize_and_markdown_accept_null_environment_and_inputs() -> None:
    """REQ-21: 未採取（null）の environment・inputs の記録でも落ちず、未採取と分かる文言を出す。"""
    items = {n: {"status": "not_run", "reason": "interrupted"} for n in mod.ITEM_ORDER}
    rec = {
        "schema": mod.SCHEMA,
        "evidence_hint": "requires_human_review",
        "bin_override": False,
        "environment": None,
        "inputs": None,
        "options": {"items": ["B"], "repeat": 1},
        "items": items,
    }
    out = mod.sanitize_record(rec)
    assert out["environment"] is None
    assert out["inputs"] is None
    assert out["items"]["B"] == {"status": "not_run", "reason": "interrupted"}
    assert mod.render_markdown(out).count("(not collected)") == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--items", "Z"],
        ["--items", "B,B"],
        ["--repeat", "0"],
        ["--repeat", "1001"],
        ["--items", "A"],
        ["--items", "E"],
        ["--items", "C,E"],
    ],
)
def test_main_rejects_invalid_arguments_without_traceback(
    extra: list[str], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """REQ-21: Python 側も不正な引数を invalid_input(64) の JSON で拒否する（traceback なし）。"""
    base = {
        "--items": "B",
        "--repeat": "1",
        "--p95-limit-us": "1",
        "--package-limit-bytes": "1",
        "--repo-root": str(REPO),
        "--work-dir": str(tmp_path),
    }
    for k, v in zip(extra[::2], extra[1::2]):  # noqa: B905  Python 3.9 互換のため strict を使わない
        base[k] = v
    argv = ["run"] + [x for kv in base.items() for x in kv]
    assert mod.main(argv) == 64
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "invalid_input"


def _ns(**over: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "items": "B",
        "with_ci": False,
        "repeat": 1,
        "p95_limit_us": 1,
        "package_limit_bytes": 1,
        "bin_override": False,
        "bin": None,
    }
    base.update(over)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    ("over", "want"),
    [
        ({"p95_limit_us": 3_600_000_000}, None),
        ({"p95_limit_us": 3_600_000_001}, "--p95-limit-us must be an integer from 1 to 3600000000"),
        ({"p95_limit_us": 0}, "--p95-limit-us must be an integer from 1 to 3600000000"),
        ({"package_limit_bytes": 0}, "--package-limit-bytes must be a positive integer"),
    ],
)
def test_validate_args_limits_boundaries(over: dict[str, Any], want: str | None) -> None:
    """REQ-21・REQ-31: p95 上限は 1〜3600000000（定義ファイルの上限と対）、package は 1 以上。"""
    assert mod.validate_args(_ns(**over)) == want


@pytest.mark.parametrize(
    ("over", "want"),
    [
        ({"items": "E"}, "item E requires item B"),
        ({"items": "C,D,E,F"}, "item E requires item B"),
        ({"items": "B,E"}, None),
        ({"items": "A,B,E", "with_ci": True}, None),
    ],
)
def test_validate_args_requires_item_b_for_item_e(over: dict[str, Any], want: str | None) -> None:
    """REQ-21・REQ-28・#362: E は B の成果物を使うので、B なしの E は起動前に引数エラー（64）。"""
    assert mod.validate_args(_ns(**over)) == want


def test_validate_args_rejects_an_unusable_bin_override(tmp_path: Path) -> None:
    """REQ-21・#362: `FANDHE_EDGE_BIN` が相対・不在・非実行・ディレクトリなら起動前に 64 の分類。"""
    msg = "FANDHE_EDGE_BIN must be a path to an executable file"
    plain = tmp_path / "plain"
    plain.write_text("x", "utf-8")
    plain.chmod(0o600)
    good = tmp_path / "good"
    good.write_text("#!/bin/sh\n", "utf-8")
    good.chmod(0o700)
    for bad in ("relative/bin", str(tmp_path / "missing"), str(plain), str(tmp_path)):
        assert mod.validate_args(_ns(bin_override=True, bin=bad)) == msg, bad
    assert mod.validate_args(_ns(bin_override=True, bin=str(good))) is None


def _run(tmp_path: Path, argv: list[str], name: str = "o") -> Any:
    return mod.run_cmd(argv, tmp_path, tmp_path / name, tmp_path / (name + ".e"), 20, 4096, 4096)


@pytest.mark.parametrize(
    ("script", "want"),
    [
        ("exit 0", 0),
        ("exit 1", 1),
        ("exit 20", 20),
        ("exit 127", 127),
        ("exit 255", 255),
        ("kill -9 $$", 137),
    ],
)
def test_run_cmd_propagates_exit_code_through_wrapper(
    tmp_path: Path, script: str, want: int
) -> None:
    """REQ-39: ラッパーが rc ファイル経由で終了コード（子が KILL で死んだ 137 を含む）を伝える。"""
    r = _run(tmp_path, ["/bin/sh", "-c", script])
    assert (r.exit_code, r.reason) == (want, None)


def test_run_cmd_kills_term_ignoring_grandchild_after_normal_exit(tmp_path: Path) -> None:
    """REQ-39: コマンドが正常終了しても、TERM を無視してグループに残った孫を KILL で止める。"""
    pidfile = tmp_path / "gpid"
    script = f'trap "" TERM; sleep 60 & echo $! > "{pidfile}"; exit 0'
    r = _run(tmp_path, ["/bin/sh", "-c", script])
    assert (r.exit_code, r.reason) == (0, None)
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_run_cmd_is_not_success_when_rc_file_is_missing(tmp_path: Path) -> None:
    """REQ-39: ラッパーが外から KILL され rc ファイルが無いときは exit_code None（fail-closed）。"""
    r = _run(tmp_path, ["/bin/sh", "-c", "kill -9 $PPID; sleep 0.3"])
    assert (r.exit_code, r.reason) == (None, "killed")


def test_run_cmd_passes_awkward_arguments_as_single_arguments(tmp_path: Path) -> None:
    """REQ-39: `-n`・`$HOME`・`*`・改行を含む引数がシェルに解釈されず 1 引数のまま渡る。"""
    script = 'printf "%s|" "$#" "$1" "$2" "$3" "$4"'
    r = _run(tmp_path, ["/bin/sh", "-c", script, "sh", "-n", "$HOME", "*", "a\nb"])
    assert r.exit_code == 0
    assert (tmp_path / "o").read_text() == "4|-n|$HOME|*|a\nb|"


def _cap(total: int, limit: Any, exceeded: bool) -> dict[str, Any]:
    return {"total_bytes": total, "limit_bytes": limit, "exceeded": exceeded, "components": {}}


def test_judge_capacity_limit_checks_reported_limit_and_boundary() -> None:
    """C-2: 報告された上限が指定値と一致し、total > limit で exceeded のときだけ ok。"""
    ok = mod.judge_capacity_limit
    assert ok(20, "limit_exceeded", _cap(1001, 1000, True), 1000, False) is None
    assert ok(20, "limit_exceeded", _cap(1001, 999, True), 1000, False) == "unexpected_output"
    assert ok(20, "limit_exceeded", _cap(1001, None, True), 1000, False) == "unexpected_output"
    assert ok(20, "limit_exceeded", None, 1000, False) == "unexpected_output"
    # 上限ちょうどは超過でない（CLI の規則は total_bytes > limit_bytes）
    want = "capacity_limit_not_enforced"
    assert ok(20, "limit_exceeded", _cap(1000, 1000, True), 1000, False) == want
    assert ok(20, "limit_exceeded", _cap(1001, 1000, False), 1000, False) == want
    assert ok(0, "limit_exceeded", _cap(1001, 1000, True), 1000, False) == want
    assert ok(20, "invalid_input", _cap(1001, 1000, True), 1000, False) == want
    assert ok(20, "limit_exceeded", _cap(1001, 1000, True), 1000, True) == want


def _j95(rc: int, p95: Any, cap: Any, limit: int, code: str = "limit_exceeded") -> Any:
    """judge_p95 の呼び出し補助。exit 20 は code、exit 0 は status を持つ package JSON を渡す。"""
    pkg = {"code": code} if rc == 20 else {"status": "ok"}
    return mod.judge_p95(rc, pkg, p95, cap, limit)


def test_judge_p95_requires_value_exceeded_exit_code_consistency() -> None:
    """C-1: exceeded は p95_us > limit_us と exit 20 の両方と一致し、容量超過の exit 20 は不可。"""
    j = _j95
    limit = 50000
    over = {"p95_us": 60000, "limit_us": limit, "exceeded": True}
    under = {"p95_us": 4, "limit_us": limit, "exceeded": False}
    good = _cap(5, 40000000, False)
    assert j(20, over, good, limit) is None
    assert j(0, under, good, limit) is None
    # capacity が欠落・要約不能なら、p95 側が整合していても不合格
    assert j(0, under, None, limit) == "unexpected_output"
    assert j(20, over, None, limit) == "unexpected_output"
    # p95 が上限未満なのに exceeded true かつ exit 20
    assert j(20, dict(under, exceeded=True), good, limit) == "unexpected_output"
    # p95 が上限超過なのに exceeded false
    assert j(0, dict(over, exceeded=False), good, limit) == "unexpected_output"
    # p95 は超過していないが容量超過で exit 20（p95 以外の理由）
    assert j(20, under, _cap(9, 5, True), limit) == "unexpected_output"
    assert j(0, under, _cap(9, 5, True), limit) == "unexpected_output"
    assert j(0, dict(under, limit_us=7), good, limit) == "unexpected_output"
    assert j(0, {"p95_us": 1}, good, limit) == "missing_field"


def test_compare_infer_lists_rows_without_exact_score_match_in_mismatch_ids() -> None:
    """REQ-28: ラベルが一致しても、スコアが完全一致でない行の id は mismatch_ids に入る。"""
    batch = {
        "a": {"predicted_label": "x", "scores": {"x": 0.5}},
        "b": {"predicted_label": "x", "scores": {"x": 0.5}},
    }
    single = {
        "a": {"predicted_label": "x", "scores": {"x": 0.5 + 1e-12}},
        "b": {"predicted_label": "x", "scores": {"x": 0.5 + 1e-6}},
    }
    cmp = mod.compare_infer(batch, single)
    assert cmp["mismatch_ids"] == ["a", "b"]
    assert (cmp["label_match"], cmp["label_mismatch"]) == (2, 0)
    assert cmp["scores_exact_match"] == 0


def test_compare_infer_one_row_off_by_1e12_is_the_only_mismatch() -> None:
    """REQ-28: 1 行だけスコアが 1e-12 ずれると scores_exact_match が 1 減り、その id だけが入る。"""
    batch = {
        "a": {"predicted_label": "x", "scores": {"x": 0.75, "y": 0.25}},
        "b": {"predicted_label": "x", "scores": {"x": 0.75, "y": 0.25}},
        "c": {"predicted_label": "x", "scores": {"x": 0.75, "y": 0.25}},
    }
    single = {k: {"predicted_label": "x", "scores": {"x": 0.75, "y": 0.25}} for k in "abc"}
    single["b"] = {"predicted_label": "x", "scores": {"x": 0.75 + 1e-12, "y": 0.25 - 1e-12}}
    cmp = mod.compare_infer(batch, single)
    assert cmp["scores_exact_match"] == 2
    assert cmp["mismatch_ids"] == ["b"]
    assert (cmp["label_mismatch"], cmp["scores_nonfinite"]) == (0, 0)


FAKE_INFER = """#!{python}
import json, sys
a = sys.argv[1:]
ids = [o["id"] for o in json.load(open("definition.json"))["options"]]
n = len(ids)
def scores(delta):
    v = [1.0 / n] * n
    v[0] += delta
    v[1] -= delta
    return dict(zip(ids, v))
mode = {mode!r}
only_id = {only_id!r}
def row(rid, delta, step=False):
    d = {{"id": rid, "status": "ok", "predicted_label": ids[0], "scores": scores(delta)}}
    if step:
        d["step"] = "infer"
    return json.dumps(d)
if "--input-file" in a:
    rows = [json.loads(line)["id"] for line in open(a[a.index("--input-file") + 1])]
    if mode == "short":
        rows = rows[:-1]
    if mode == "dup":
        rows[-1] = rows[0]
    if mode == "swap":
        rows[0], rows[1] = rows[1], rows[0]
    for i, rid in enumerate(rows):
        print(row(rid, 0.0, mode == "batch-step" and i == 1))
else:
    rid = a[a.index("--id") + 1]
    print(row(rid, {delta!r} if only_id in (None, rid) else 0.0, mode == "single-step"))
"""


def _infer_cli(
    tmp_path: Path, mode: str = "ok", delta: float = 0.0, only_id: str | None = None
) -> Path:
    fake = tmp_path / "fake-infer"
    fake.write_text(
        FAKE_INFER.format(python=sys.executable, mode=mode, delta=delta, only_id=only_id), "utf-8"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake


def _e_ctx(tmp_path: Path, fake: Path) -> Any:
    ctx = _ctx(tmp_path / "w", fake)
    ctx.work.mkdir()
    assert mod.stage_inputs(ctx, ctx.work / "B", None)
    return ctx


def test_item_e_writes_nonempty_mismatch_ids_when_scores_differ(tmp_path: Path) -> None:
    """REQ-28: E が failed のとき mismatch-ids.txt は空にならない（スコア差のみの不一致でも）。"""
    ctx = _e_ctx(tmp_path, _infer_cli(tmp_path, delta=1e-7))
    res = mod.item_e(ctx)
    assert (res["status"], res["reason"], res["label_mismatch"]) == ("failed", "mismatch", 0)
    ids = (ctx.work / "E" / "mismatch-ids.txt").read_text().split()
    assert len(ids) == res["records"]
    assert res["records"] == mod.read_facts(ctx.work / "B").train_records


def test_item_e_fails_when_one_row_score_is_off_by_1e12(tmp_path: Path) -> None:
    """REQ-28: ラベルが同じでスコアが 1e-12 ずれた 1 行があれば E は failed（完全一致が条件）。"""
    ctx = _e_ctx(tmp_path, _infer_cli(tmp_path))
    first = mod.read_train_inputs(ctx.work / "B" / "train.jsonl")
    assert first is not None
    target = first[0][0]
    # 同じパスの偽 CLI を、先頭 id の単体推論だけ 1e-12 ずらす版へ書き換える
    _infer_cli(tmp_path, delta=1e-12, only_id=target)
    res = mod.item_e(ctx)
    assert (res["status"], res["reason"], res["label_mismatch"]) == ("failed", "mismatch", 0)
    assert res["scores_exact_match"] == res["records"] - 1
    assert (ctx.work / "E" / "mismatch-ids.txt").read_text() == target + "\n"


def test_item_e_passes_when_batch_and_single_agree(tmp_path: Path) -> None:
    """REQ-28: 行数・id・スコアが一致する正常な出力は ok。"""
    res = mod.item_e(_e_ctx(tmp_path, _infer_cli(tmp_path)))
    assert res["status"] == "ok"
    assert res["label_mismatch"] == 0
    assert res["scores_exact_match"] == res["records"]


@pytest.mark.parametrize("mode", ["short", "dup", "swap"])
def test_item_e_rejects_batch_row_count_and_duplicate_ids(tmp_path: Path, mode: str) -> None:
    """REQ-28: バッチ出力の行数が違う・id が重複する・入力順でない（#362）は unexpected_output。"""
    res = mod.item_e(_e_ctx(tmp_path, _infer_cli(tmp_path, mode=mode)))
    assert (res["status"], res["reason"], res["step"]) == (
        "failed",
        "unexpected_output",
        "infer-batch",
    )


@pytest.mark.parametrize(
    ("mode", "step"),
    [("batch-step", "infer-batch"), ("single-step", "infer-single")],
)
def test_item_e_rejects_infer_output_with_a_step_field(
    tmp_path: Path, mode: str, step: str
) -> None:
    """REQ-28・REQ-33: E の infer 出力に `step` があれば（B の単発と同じく）unexpected_output。"""
    res = mod.item_e(_e_ctx(tmp_path, _infer_cli(tmp_path, mode=mode)))
    assert (res["status"], res["reason"], res["step"]) == ("failed", "unexpected_output", step)


def test_judge_p95_requires_limit_exceeded_code_on_exit_20() -> None:
    """REQ-21・REQ-31: exit 20 は code が limit_exceeded、exit 0 は status が ok のときだけ合格。"""
    over = {"p95_us": 60000, "limit_us": 50000, "exceeded": True}
    under = {"p95_us": 4, "limit_us": 50000, "exceeded": False}
    good = _cap(5, 40000000, False)
    assert _j95(20, over, good, 50000, code="invalid_input") == "unexpected_output"
    assert mod.judge_p95(20, {}, over, good, 50000) == "unexpected_output"
    assert mod.judge_p95(0, {"status": "failed"}, under, good, 50000) == "unexpected_output"
    assert mod.judge_p95(0, None, under, good, 50000) == "unexpected_output"
    assert mod.judge_p95(20, {"code": "limit_exceeded"}, over, good, 50000) is None


def test_judge_linkage_target_fixes_the_three_hashes() -> None:
    """REQ-32: 検査対象・いまの CLI・開始時の CLI が一致したときだけ None。"""
    a, b = "a" * 64, "b" * 64
    j = mod.judge_linkage_target
    assert j(a, a, a) is None
    assert j(a, a, b) == "linkage_target_mismatch"
    assert j(a, b, b) == "cli_changed"
    assert j(a, b, a) == "cli_changed"
    assert j(a, a, None) == "linkage_target_unreadable"
    assert j(None, a, a) == "cli_changed"


def test_parse_linkage_cli_path_reads_exactly_one_absolute_cli_bin_line() -> None:
    """REQ-32: `cli_bin:` 行がちょうど 1 行で絶対パスのときだけ Path を返す。"""
    p = mod.parse_linkage_cli_path
    assert p("cli_bin: /abs/t/release/fandhe-edge\n") == Path("/abs/t/release/fandhe-edge")
    mixed = (
        "== build ==\ncli_bin: /x/y/fandhe-edge\nok: req32_a\nOK: tool=otool evidence=e targets=t\n"
    )
    assert p(mixed) == Path("/x/y/fandhe-edge")
    assert p("== build ==\nok: req32_a\n") is None
    assert p("cli_bin: /a/fandhe-edge\ncli_bin: /b/fandhe-edge\n") is None
    assert p("cli_bin: rel/release/fandhe-edge\n") is None


def _linkage_make_body(
    n_ok: int, tool_line: str = "OK: tool=otool evidence=x targets=y", cli_bin: str = ""
) -> str:
    lines = [f"echo 'cli_bin: {cli_bin}'"] if cli_bin else []
    lines += [f"echo 'ok: req32_t{i}'" for i in range(n_ok)]
    if tool_line:
        lines.append(f"echo '{tool_line}'")
    return "\n".join(lines) + "\nexit 0\n"


def _linkage_ctx(tmp_path: Path, cli_bytes: bytes, target_bytes: bytes) -> Any:
    repo = tmp_path / "repo"
    (repo / "target" / "release").mkdir(parents=True)
    (repo / "target" / "release" / "fandhe-edge").write_bytes(target_bytes)
    cli = tmp_path / "cli"
    cli.write_bytes(cli_bytes)
    body = _linkage_make_body(
        mod.LINKAGE_ENV_I_TESTS, cli_bin=str(repo / "target" / "release" / "fandhe-edge")
    )
    make = _script(tmp_path, "fake-make", body)
    ctx = _ctx(tmp_path / "w", cli)
    ctx.work.mkdir()
    ctx.repo = repo
    ctx.harness = False
    ctx.make_cmd = str(make)
    ctx.offline_env = {}
    ctx.cli_sha256 = mod.sha256_file(cli, 1 << 20)
    return ctx


def test_item_d_requires_linkage_target_to_match_executed_cli(tmp_path: Path) -> None:
    """REQ-32: harness でないとき、検査対象が実行 CLI と違えば D は failed（結合）。"""
    res = mod.item_d(_linkage_ctx(tmp_path / "a", b"same", b"same"))
    assert res.get("reason") in (None, "otool_failed")  # otool は偽バイナリで失敗しうる
    assert res["linkage_tool"] == "otool"
    assert res["linkage_target_matches_cli"] is True
    assert len(res["linkage_target_sha256"]) == 64
    res = mod.item_d(_linkage_ctx(tmp_path / "b", b"one", b"two"))
    assert (res["status"], res["reason"]) == ("failed", "linkage_target_mismatch")
    assert res["linkage_target_matches_cli"] is False
    ctx = _linkage_ctx(tmp_path / "c", b"one", b"one")
    ctx.cli_sha256 = "0" * 64
    assert mod.item_d(ctx)["reason"] == "cli_changed"


def test_item_d_without_cli_bin_line_is_linkage_target_unreadable(tmp_path: Path) -> None:
    """REQ-32: `cli_bin:` 行が無ければ（harness でない）D は linkage_target_unreadable。"""
    ctx = _linkage_ctx(tmp_path, b"same", b"same")
    ctx.make_cmd = str(
        _script(tmp_path, "make-no-line", _linkage_make_body(mod.LINKAGE_ENV_I_TESTS))
    )
    res = mod.item_d(ctx)
    assert (res["status"], res["reason"]) == ("failed", "linkage_target_unreadable")
    assert res["linkage_target_sha256"] is None


def test_harness_skips_linkage_comparison_and_record_has_no_path(tmp_path: Path) -> None:
    """代役の下では照合せず null。linkage の欄に sha256 以外（パス）が出ない。"""
    ctx = _linkage_ctx(tmp_path, b"x", b"y")
    ctx.harness = True
    res = mod.item_d(ctx)
    assert (res["status"], res["linkage_target_matches_cli"], res["linkage_target_sha256"]) == (
        "ok",
        None,
        None,
    )
    ok = mod.sanitize_record({"items": {"D": {"linkage_target_sha256": "a" * 64}}})
    assert ok["items"]["D"]["linkage_target_sha256"] == "a" * 64


# ---- 工程ごとの報告値の照合（REQ-21・REQ-33）。期待値は fixture のコピーから導く ----

FIXTURE_DIR = REPO / "fixtures" / "sandbox_run_eval"


def _facts() -> Any:
    facts = mod.read_facts(FIXTURE_DIR)
    assert facts is not None
    return facts


def _valid_reports(facts: Any) -> dict[str, dict[str, Any]]:
    n = facts.train_records
    return {
        "register": {
            "step": "register",
            "status": "ok",
            "options": len(facts.option_ids),
            "evaluation_defined": facts.eval_records > 0,
            "definition_sha256": "a" * 64,
        },
        "inspect": {
            "step": "inspect",
            "status": "ok",
            "valid_records": n,
            "split": {"train": n - 2, "validation": 1, "test": 1},
        },
        "train": {"step": "train", "status": "ok", "candidate": 0, "kind": "c1"},
        "select": {"step": "select", "status": "ok", "candidate": 0, "kind": "c1"},
        "evaluate": {
            "step": "evaluate",
            "status": "ok",
            "candidate": 0,
            "kind": "c1",
            "n_total": facts.eval_records,
            "correct": facts.eval_records,
            "accuracy": 1.0,
            "macro_f1": 1.0,
        },
    }


def test_read_facts_derives_expectations_from_the_fixture_copy() -> None:
    """期待値（件数・選択肢）は定数でなく、fixture のコピーから導く。"""
    facts = _facts()
    lines = lambda name: len([x for x in (FIXTURE_DIR / name).read_text().splitlines() if x])  # noqa: E731
    assert facts.train_records == lines("train.jsonl")
    assert facts.eval_records == lines("evaluation.jsonl")
    defn = json.loads((FIXTURE_DIR / "definition.json").read_text())
    assert facts.option_ids == [o["id"] for o in defn["options"]]
    assert facts.has_acceptance is False
    assert facts.p95_limit_us is None


@pytest.mark.parametrize("name", ["register", "inspect", "train", "select", "evaluate"])
def test_step_check_accepts_valid_reports(name: str) -> None:
    """正常な工程出力は合格する。"""
    f = _facts()
    assert mod._step_check(name, _valid_reports(f)[name], set(), 0, f, 0, "c1") is True


@pytest.mark.parametrize(
    ("name", "patch"),
    [
        ("register", {"options": 99}),
        ("register", {"evaluation_defined": False}),
        ("register", {"options": True}),
        ("inspect", {"valid_records": 1}),
        ("inspect", {"split": {"train": 1, "validation": 1, "test": 1}}),
        ("inspect", {"split": {"train": -1, "validation": 1, "test": 1}}),
        ("train", {"candidate": 1}),
        ("train", {"kind": 3}),
        ("train", {"kind": "zzz"}),
        ("select", {"kind": "c3"}),
        ("evaluate", {"kind": "c3"}),
        ("select", {"candidate": 1}),
        ("evaluate", {"candidate": 1}),
        ("evaluate", {"n_total": 1}),
        ("evaluate", {"correct": 10**6}),
        ("evaluate", {"correct": -1}),
        ("evaluate", {"correct": True}),
    ],
)
def test_step_check_rejects_reports_that_contradict_the_input(
    name: str, patch: dict[str, Any]
) -> None:
    """REQ-21: 指定値・入力から分かる値と食い違う報告は不合格。"""
    f = _facts()
    bad = dict(_valid_reports(f)[name], **patch)
    assert mod._step_check(name, bad, set(), 0, f, 0, "c1") is False


def _scores(facts: Any, delta: float = 0.0) -> dict[str, float]:
    n = len(facts.option_ids)
    vals = [1.0 / n] * n
    vals[0] += delta
    return dict(zip(facts.option_ids, vals))  # noqa: B905  Python 3.9 互換


def test_check_infer_output_validates_label_keys_finiteness_and_sum() -> None:
    """REQ-21: infer の label は選択肢 ID、scores のキーは選択肢 ID と一致、有限で和が 1 近傍。"""
    f = _facts()
    ids = f.option_ids
    ok = {
        "id": mod.DEFAULT_TEXT_ID,
        "status": "ok",
        "predicted_label": ids[0],
        "scores": _scores(f),
    }
    assert mod.check_infer_output(ok, f) is True
    assert mod._step_check("infer", ok, set(), 0, f, None) is True
    # B の単発 infer は `--id` なしなので、id が既定値以外なら不合格（#362）
    assert mod._step_check("infer", dict(ok, id="x"), set(), 0, f, None) is False
    assert mod.check_infer_output(dict(ok, predicted_label="nope"), f) is False
    assert mod.check_infer_output(dict(ok, scores={ids[0]: 1.0}), f) is False
    assert mod.check_infer_output(dict(ok, scores=dict(_scores(f), extra=0.0)), f) is False
    assert mod.check_infer_output(dict(ok, scores=_scores(f, 1e-3)), f) is False
    nan = dict(_scores(f), **{ids[0]: float("nan")})
    assert mod.check_infer_output(dict(ok, scores=nan), f) is False
    # 許容差（1e-6）以内のずれは合格（CLI の runtime が許す範囲）
    assert mod.check_infer_output(dict(ok, scores=_scores(f, 5e-7)), f) is True


def _cap_json(
    total: int = 5, limit: int = 40000000, exceeded: bool = False, sum_delta: int = 0
) -> dict[str, Any]:
    """5 項目の合計が total になる容量内訳（`sum_delta` で合計をずらして不一致にできる）。"""
    comp = {n: {"bytes": 1, "file_count": 1} for n in mod.CAPACITY_COMPONENTS}
    comp["weights"]["bytes"] = total - 4 + sum_delta
    return {"total_bytes": total, "limit_bytes": limit, "exceeded": exceeded, "components": comp}


def _pkg(facts: Any, **patch: Any) -> dict[str, Any]:
    base = {
        "step": "package",
        "status": "ok",
        "judgment": None,
        "acceptance_defined": False,
        "capacity": _cap_json(),
        "infer_p95": None,
    }
    return dict(base, **patch)


def test_package_report_is_checked_against_definition_and_capacity_rules() -> None:
    """REQ-30・REQ-31: 基準・p95 上限が無い定義の package は judgment・infer_p95 が null。"""
    f = _facts()
    chk = lambda obj, rc=0: mod._step_check("package", obj, {0, 20}, rc, f, 0)  # noqa: E731
    assert chk(_pkg(f)) is True
    assert chk(_pkg(f, judgment="pass")) is False
    assert chk(_pkg(f, acceptance_defined=True)) is False
    assert chk(_pkg(f, infer_p95={"p95_us": 1, "limit_us": 2, "exceeded": False})) is False
    # exit 0 で capacity が超過・不整合・要約不能
    assert chk(_pkg(f, capacity=_cap_json(total=9, limit=5, exceeded=True))) is False
    assert chk(_pkg(f, capacity=_cap_json(total=9, limit=5, exceeded=False))) is False
    assert chk(_pkg(f, capacity=_cap_json(total=5, limit=3, exceeded=True))) is False
    assert chk(_pkg(f, capacity=None)) is False
    # exit 20 で容量超過（整合。上限は定義の既定値）は許容される
    over = {
        "code": "limit_exceeded",
        "step": "package",
        "capacity": _cap_json(40000001, 40000000, True),
        "infer_p95": None,
    }
    assert chk(over, rc=20) is True
    # p95 上限のある定義では infer_p95 が非 null
    with_limit = mod.Facts(f.option_ids, f.train_records, f.eval_records, False, 100)
    assert mod._step_check("package", _pkg(f), {0}, 0, with_limit, 0) is False
    p95 = {"p95_us": 1, "limit_us": 100, "exceeded": False}
    assert mod._step_check("package", _pkg(f, infer_p95=p95), {0}, 0, with_limit, 0) is True


def test_capacity_summary_rejects_negative_bool_and_missing_values() -> None:
    """REQ-30: bytes・file_count・total_bytes・limit_bytes は 0 以上の int に限る。"""
    ok = {"capacity": _cap_json()}
    assert mod.capacity_summary(ok) is not None
    for patch in ({"total_bytes": -1}, {"limit_bytes": None}, {"limit_bytes": True}):
        assert mod.capacity_summary({"capacity": dict(_cap_json(), **patch)}) is None
    neg = _cap_json()
    neg["components"]["weights"]["bytes"] = -1
    assert mod.capacity_summary({"capacity": neg}) is None
    boolc = _cap_json()
    boolc["components"]["weights"]["file_count"] = True
    assert mod.capacity_summary({"capacity": boolc}) is None


# ---- C（C-1・C-2）の結合: 偽 CLI は定義・入力ファイルから値を導いて出力する ----

FAKE_PIPELINE = """#!{python}
import json, os, sys
cfg = {cfg}
d = json.load(open("definition.json"))
lines = lambda n: len([x for x in open(n) if x.strip()])
n_train, n_eval = lines("train.jsonl"), lines("evaluation.jsonl")
limits = d.get("limits", {{}})
c2 = "max_package_bytes" in limits
cmd = sys.argv[1]
names = ["weights", "vocab_or_feature_transform", "label_table", "calibration", "metadata"]
def comps(total):
    # file_count は公開される package/ の実ファイル数（model.onnx の 1 つ）に合わせる
    c = {{k: {{"bytes": 1, "file_count": 0}} for k in names}}
    c["weights"]["bytes"] = total - 4 + cfg["comp_delta"]
    c["weights"]["file_count"] = 1 + cfg["fc_delta"]
    return c
def out(o, rc=0):
    print(json.dumps(o))
    sys.exit(rc)
if cmd == "register":
    out({{"step": "register", "status": "ok", "options": len(d["options"]),
         "evaluation_defined": True, "definition_sha256": "a" * 64}})
if cmd == "inspect":
    out({{"step": "inspect", "status": "ok", "valid_records": n_train,
         "split": {{"train": n_train - 2, "validation": 1, "test": 1}}}})
if cmd in ("train", "select"):
    kind = cfg["train_kind"] if cmd == "train" else cfg["select_kind"]
    out({{"step": cmd, "status": "ok", "candidate": 0, "kind": kind}})
if cmd == "evaluate":
    out({{"step": "evaluate", "status": "ok", "candidate": 0, "kind": cfg["eval_kind"],
         "n_total": n_eval, "correct": n_eval, "accuracy": cfg["accuracy"],
         "macro_f1": cfg["macro_f1"]}})
if cmd == "infer":
    ids = [o["id"] for o in d["options"]]
    scores = {{i: 1.0 / len(ids) for i in ids}}
    rid = cfg["default_id"]
    if "--id" in sys.argv:
        rid = sys.argv[sys.argv.index("--id") + 1]
    out({{"id": rid, "status": "ok", "predicted_label": ids[0], "scores": scores}})
if cmd == "package":
    if cfg["staging"] == ("C2" if c2 else "C1"):
        os.makedirs("project/package.staging", exist_ok=True)
    if c2:
        total = cfg["c2_total"]
        limit = cfg["c2_limit"] or limits["max_package_bytes"]
        out({{"code": cfg["c2_code"], "message": "resource limit exceeded", "step": "package",
             "capacity": {{"total_bytes": total, "limit_bytes": limit,
                          "exceeded": cfg["c2_exceeded"], "components": comps(total)}},
             "infer_p95": None}}, cfg["c2_rc"])
    rc = cfg["rc1"]
    p95 = None
    if "max_infer_p95_us" in limits:
        p95 = {{"p95_us": cfg["p95_us"], "limit_us": limits["max_infer_p95_us"],
               "exceeded": cfg["p95_exceeded"]}}
    total = cfg["total1"]
    cap = {{"total_bytes": total, "limit_bytes": cfg["limit1"], "exceeded": False,
           "components": comps(total)}}
    if cfg["publish"] is None:
        publish = rc == 0
    else:
        publish = cfg["publish"]
    if publish:
        os.makedirs("project/package", exist_ok=True)
        with open("project/package/model.onnx", "wb") as f:
            f.write(b"x" * cfg["actual_bytes"])
        if cfg["package_kind"] == "symlink":
            os.symlink("model.onnx", "project/package/link")
        if cfg["package_kind"] == "dir":
            os.makedirs("project/package/sub")
    if rc == 20:
        out({{"code": cfg["code1"], "message": "m", "step": "package", "capacity": cap,
             "infer_p95": p95}}, 20)
    out({{"step": "package", "status": "ok", "judgment": None, "acceptance_defined": False,
         "capacity": cap, "infer_p95": p95}})
"""

C_DEFAULT = {
    "rc1": 0,
    "p95_us": 4,
    "p95_exceeded": False,
    "code1": "limit_exceeded",
    "publish": None,
    "c2_rc": 20,
    "c2_code": "limit_exceeded",
    "c2_total": 5000,
    "c2_limit": None,
    "c2_exceeded": True,
    "total1": 10,
    "limit1": 40000000,
    "actual_bytes": 10,
    "comp_delta": 0,
    "fc_delta": 0,
    "package_kind": "",
    "staging": "",
    "default_id": "input",
    "train_kind": "c1",
    "select_kind": "c1",
    "eval_kind": "c1",
    "accuracy": 1.0,
    "macro_f1": 1.0,
}


def _c_ctx(tmp_path: Path, **cfg: Any) -> Any:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake = tmp_path / "fake-pipeline"
    fake.write_text(
        FAKE_PIPELINE.format(python=sys.executable, cfg=dict(C_DEFAULT, **cfg)), "utf-8"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    ctx = _ctx(tmp_path / "w", fake)
    ctx.work.mkdir()
    return ctx


def test_item_c_accepts_consistent_outputs_and_records_publication(tmp_path: Path) -> None:
    """REQ-30・REQ-31: p95・容量・公開状態が整合する出力は ok（C-1 は公開状態を記録）。"""
    res = mod.item_c(_c_ctx(tmp_path))
    assert res["status"] == "ok"
    assert res["p95"]["package_published"] is True
    assert res["p95"]["classification"] == "reference_only"
    assert res["capacity_limit"]["package_published"] is False


def test_item_c1_exit_20_for_p95_is_ok_only_when_package_is_not_published(tmp_path: Path) -> None:
    """REQ-30: p95 超過の exit 20 は package/ が無いときだけ合格。exit 0 は package/ があるとき。"""
    over = {"rc1": 20, "p95_us": 60000, "p95_exceeded": True}
    res = mod.item_c(_c_ctx(tmp_path / "a", **over))
    assert res["status"] == "ok"
    assert (res["p95"]["package_exit_code"], res["p95"]["package_published"]) == (20, False)
    res = mod.item_c(_c_ctx(tmp_path / "b", publish=True, **over))
    assert (res["status"], res["reason"], res["case"]) == ("failed", "unexpected_output", "C-1")
    res = mod.item_c(_c_ctx(tmp_path / "c", publish=False))
    assert (res["status"], res["reason"], res["case"]) == ("failed", "unexpected_output", "C-1")


def test_item_c1_rejects_p95_values_that_contradict_exceeded_or_exit(tmp_path: Path) -> None:
    """REQ-31: p95_us < limit_us なのに exceeded true、または exit 0 で超過は failed。"""
    res = mod.item_c(_c_ctx(tmp_path / "a", p95_exceeded=True))
    assert (res["status"], res["reason"]) == ("failed", "unexpected_output")
    res = mod.item_c(_c_ctx(tmp_path / "b", rc1=20, p95_exceeded=True, p95_us=1))
    assert (res["status"], res["reason"]) == ("failed", "unexpected_output")
    res = mod.item_c(_c_ctx(tmp_path / "c", rc1=20, p95_exceeded=True, p95_us=60000, code1="x"))
    assert (res["status"], res["reason"]) == ("failed", "unexpected_output")


def test_item_c2_checks_reported_limit_and_boundary(tmp_path: Path) -> None:
    """REQ-30: 報告された上限が指定値と違う、または total <= limit で exceeded true は failed。"""
    res = mod.item_c(_c_ctx(tmp_path / "a", c2_limit=7))
    assert (res["status"], res["reason"]) == ("failed", "unexpected_output")
    res = mod.item_c(_c_ctx(tmp_path / "b", c2_total=1))
    assert res["status"] == "failed"
    res = mod.item_c(_c_ctx(tmp_path / "c", c2_rc=0))
    assert res["status"] == "failed"


def test_classify_p95_is_reference_only_under_harness_or_bin_override(tmp_path: Path) -> None:
    """evidence_hint と矛盾させない: 静かな状態の申告があっても代役・CLI 差し替えでは参考値。"""
    ctx = _ctx(tmp_path, _fake_cli(tmp_path, "exit 0\n"))
    ctx.quiet_machine, ctx.harness, ctx.bin_override = True, False, False
    assert mod.classify_p95(ctx) == "real_machine"
    ctx.harness = True
    assert mod.classify_p95(ctx) == "reference_only"
    ctx.harness, ctx.bin_override = False, True
    assert mod.classify_p95(ctx) == "reference_only"
    ctx.bin_override, ctx.quiet_machine = False, False
    assert mod.classify_p95(ctx) == "reference_only"


def test_judge_make_ci_rejects_all_skipped_pytest() -> None:
    """A: pytest が全件 skip（passed 0）なら合格にしない。"""
    log = CARGO_RUST_LINE + "\n=== 126 skipped in 2.4s ===\n"
    assert mod.judge_make_ci(mod.parse_make_ci_log(log)) == "no_test_results"


def test_linkage_constants_match_the_script_and_ok_line_is_parsed() -> None:
    """D: ok: req32_ の件数の定数は check-runtime-linkage.sh のテスト名の数と一致する。"""
    text = (REPO / "scripts" / "check-runtime-linkage.sh").read_text(encoding="utf-8")
    block = text.split("for t in", 1)[1].split("; do", 1)[0]
    names = [w for w in block.replace("\\", " ").split() if w.startswith("req32_")]
    assert len(names) == mod.LINKAGE_ENV_I_TESTS
    assert mod.parse_linkage_tool("a\nOK: tool=otool evidence=x targets=y\n") == "otool"
    assert mod.parse_linkage_tool("OK: tool=ldd evidence=x") == "ldd"
    assert mod.parse_linkage_tool("OK: tool=weird") is None
    assert mod.parse_linkage_tool("no ok line") is None


def test_item_d_requires_exact_test_count_and_ok_line_and_otool_on_macos(tmp_path: Path) -> None:
    """D: テスト数が定数と一致し、`OK: tool=` 行があること。macOS では tool が otool であること。"""
    for n_ok, tool, want in (
        (mod.LINKAGE_ENV_I_TESTS - 1, "OK: tool=otool x", "unexpected_output"),
        (mod.LINKAGE_ENV_I_TESTS + 1, "OK: tool=otool x", "unexpected_output"),
        (mod.LINKAGE_ENV_I_TESTS, "", "unexpected_output"),
        (mod.LINKAGE_ENV_I_TESTS, "OK: tool=ldd x", "unexpected_output"),
    ):
        ctx = _linkage_ctx(tmp_path / f"{n_ok}{len(tool)}", b"x", b"x")
        ctx.harness = True
        ctx.make_cmd = str(_script(tmp_path, f"m{n_ok}{len(tool)}", _linkage_make_body(n_ok, tool)))
        res = mod.item_d(ctx)
        if sys.platform != "darwin" and tool == "OK: tool=ldd x" and n_ok == 3:
            assert res["status"] == "ok"  # macOS 以外の ldd は現状維持（記録のみ）
            assert res["linkage_tool"] == "ldd"
        else:
            assert (res["status"], res["reason"]) == ("failed", want)


def test_internal_error_makes_the_script_exit_70_and_still_writes_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-21: 項目の想定外の例外（internal_error）が 1 件でもあれば exit 70。record は書く。"""
    saved = {s: signal.getsignal(s) for s in mod.INTERRUPT_SIGNALS}

    def boom(ctx: Any, name: str, b_ok: bool) -> Any:
        raise RuntimeError("x")

    monkeypatch.setattr(mod, "run_item", boom)
    cli = _fake_cli(tmp_path, "exit 0\n")
    args = mod.argparse.Namespace(
        repo_root=str(REPO),
        work_dir=str(tmp_path / "w"),
        bin=str(cli),
        bin_override=True,
        items="B",
        repeat=1,
        p95_limit_us=1,
        package_limit_bytes=1,
        quiet_machine=False,
        with_ci=False,
    )
    (tmp_path / "w").mkdir()
    try:
        rc = mod.run(args)
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
    assert rc == 70
    assert json.loads(capsys.readouterr().out) == {
        "code": "runtime_error",
        "message": "internal error",
        "record": "record.json",
    }
    rec = json.loads((tmp_path / "w" / "record.json").read_text())
    assert rec["items"]["B"] == {
        "status": "failed",
        "reason": "internal_error",
        "error_type": "RuntimeError",
    }


# ---- 記録の許可リスト方式（型 1。REQ-33・REQ-21） ----

HEX64 = "a" * 64
CAP_SUMMARY = {
    "total_bytes": 5,
    "limit_bytes": 40000000,
    "exceeded": False,
    "components": {n: {"bytes": 1, "file_count": 1} for n in mod.CAPACITY_COMPONENTS},
}
# 許可した欄の外に置く値。データ本文・選択肢 ID・入れ子の dict・list を含む
EXTRA = {
    "text": "private body",
    "note": "plain-looking-secret",
    "label": "beta",
    "nested": {"input": "private body", "deep": ["x"]},
    "rows": [{"id": "row-1"}],
}


@pytest.mark.parametrize(
    ("name", "obj", "want"),
    [
        (
            "register",
            {
                "step": "register",
                "status": "ok",
                "options": 3,
                "evaluation_defined": True,
                "definition_sha256": HEX64,
            },
            {
                "step": "register",
                "status": "ok",
                "options": 3,
                "evaluation_defined": True,
                "definition_sha256": HEX64,
            },
        ),
        (
            "inspect",
            {
                "step": "inspect",
                "status": "ok",
                "valid_records": 9,
                "split": {"train": 7, "validation": 1, "test": 1, "extra": "x"},
            },
            {
                "step": "inspect",
                "status": "ok",
                "valid_records": 9,
                "split": {"train": 7, "validation": 1, "test": 1},
            },
        ),
        (
            "train",
            {"step": "train", "status": "ok", "candidate": 0, "kind": "c1"},
            {"step": "train", "status": "ok", "candidate": 0, "kind": "c1"},
        ),
        (
            "select",
            {"step": "select", "status": "ok", "candidate": 1, "kind": "autoregressive"},
            {"step": "select", "status": "ok", "candidate": 1, "kind": "autoregressive"},
        ),
        (
            "evaluate",
            {
                "step": "evaluate",
                "status": "ok",
                "candidate": 0,
                "kind": "c3",
                "n_total": 10,
                "correct": 9,
                "accuracy": 0.9,
                "macro_f1": None,
            },
            {
                "step": "evaluate",
                "status": "ok",
                "candidate": 0,
                "kind": "c3",
                "n_total": 10,
                "correct": 9,
                "accuracy": 0.9,
                "macro_f1": None,
            },
        ),
        (
            "package",
            {
                "step": "package",
                "status": "ok",
                "judgment": "pass",
                "acceptance_defined": True,
                "capacity": CAP_SUMMARY,
                "infer_p95": {"p95_us": 7, "limit_us": 50, "exceeded": False, "extra": "x"},
            },
            {
                "step": "package",
                "status": "ok",
                "code": None,
                "judgment": "pass",
                "acceptance_defined": True,
                "capacity": CAP_SUMMARY,
                "infer_p95": {"p95_us": 7, "limit_us": 50, "exceeded": False},
            },
        ),
        (
            "infer",
            {
                "id": "train-row-7",
                "status": "ok",
                "predicted_label": "beta",
                "scores": {"alpha": 0.1, "beta": 0.8, "gamma": 0.1},
            },
            {"status": "ok", "scores_keys": 3, "predicted_index": 1},
        ),
    ],
)
def test_summarize_step_keeps_only_allowlisted_fields(
    name: str, obj: dict[str, Any], want: dict[str, Any]
) -> None:
    """REQ-33: 工程ごとの許可した欄だけを組み立て、他の欄（文字列・入れ子）は捨てる。"""
    ids = ["alpha", "beta", "gamma"]
    assert mod.summarize_step(name, obj, ids) == want
    got = mod.summarize_step(name, dict(obj, **EXTRA), ids)
    assert got == want
    text = json.dumps(got)
    for leak in ("private body", "plain-looking-secret", "row-1", "train-row-7"):
        assert leak not in text


def test_summarize_step_replaces_out_of_vocabulary_and_wrongly_typed_values() -> None:
    """REQ-33: 語彙外の文字列は `<unexpected>`、型違い・入れ子・欠落は null（具体値）。"""
    ids = ["alpha"]
    got = mod.summarize_step("train", {"step": "select", "status": "weird", "kind": "rm -rf"}, ids)
    assert got == {
        "step": "<unexpected>",
        "status": "<unexpected>",
        "candidate": None,
        "kind": "<unexpected>",
    }
    got = mod.summarize_step(
        "train", {"step": {"a": 1}, "status": ["ok"], "candidate": True, "kind": {"k": "c1"}}, ids
    )
    assert got == {"step": None, "status": None, "candidate": None, "kind": None}
    got = mod.summarize_step("register", {"options": -1, "evaluation_defined": 1}, ids)
    assert got == {
        "step": None,
        "status": None,
        "options": None,
        "evaluation_defined": None,
        "definition_sha256": None,
    }
    for bad in (HEX64.upper(), HEX64 + "\n", "a" * 63, {"x": HEX64}, 5):
        got = mod.summarize_step("register", {"definition_sha256": bad}, ids)
        assert got["definition_sha256"] is None
    # split は 1 欄でも形が違えば split ごと null
    for bad in ({"train": 1, "validation": 1}, {"train": 1, "validation": 1, "test": "1"}, [1]):
        assert mod.summarize_step("inspect", {"split": bad}, ids)["split"] is None
    # evaluate の指標は有限で 0 以上 1 以下の数だけ
    for bad in (7.0, -0.1, float("nan"), float("inf"), "x", True, {"v": 1}):
        got = mod.summarize_step("evaluate", {"accuracy": bad, "macro_f1": bad}, ids)
        assert (got["accuracy"], got["macro_f1"]) == (None, None)
    got = mod.summarize_step("evaluate", {"accuracy": 1, "macro_f1": 0.0}, ids)
    assert (got["accuracy"], got["macro_f1"]) == (1, 0.0)
    # package: code・judgment は語彙、capacity・infer_p95 は形が違えば null
    got = mod.summarize_step(
        "package",
        {
            "code": "limit_exceeded",
            "judgment": "maybe",
            "acceptance_defined": "yes",
            "capacity": {"total_bytes": 1},
            "infer_p95": {"p95_us": -1, "limit_us": 1, "exceeded": False},
        },
        ids,
    )
    assert got == {
        "step": None,
        "status": None,
        "code": "limit_exceeded",
        "judgment": "<unexpected>",
        "acceptance_defined": None,
        "capacity": None,
        "infer_p95": None,
    }
    for bad_p95 in ({"p95_us": True, "limit_us": 1, "exceeded": False}, {"p95_us": 1}, "x"):
        assert mod.summarize_step("package", {"infer_p95": bad_p95}, ids)["infer_p95"] is None


def test_summarize_infer_gives_index_and_never_the_label_or_nested_values() -> None:
    """REQ-33: infer の要約は選択肢の番号だけ。選択肢に無い・型違いの label は null。"""
    ids = ["alpha", "beta"]
    base = {"status": "ok", "scores": {"alpha": 0.5, "beta": 0.5}}
    assert mod.summarize_infer(dict(base, predicted_label="alpha"), ids)["predicted_index"] == 0
    assert mod.summarize_infer(dict(base, predicted_label="beta"), ids)["predicted_index"] == 1
    for bad in ("gamma", {"x": "alpha"}, ["alpha"], 1, None):
        assert mod.summarize_infer(dict(base, predicted_label=bad), ids)["predicted_index"] is None
    got = mod.summarize_infer({"status": {"a": [1]}, "scores": [1, 2]}, ids)
    assert got == {"status": None, "scores_keys": 0, "predicted_index": None}
    assert mod.summarize_infer({"status": "oops"}, ids)["status"] == "<unexpected>"


def test_error_fields_record_code_vocabulary_and_message_size_and_hash_only() -> None:
    """REQ-21・REQ-33: code は語彙、message は本文でなく UTF-8 のバイト数と sha256。"""
    f = mod.error_fields
    msg = "文字 abc"
    want_hash = hashlib.sha256(msg.encode("utf-8")).hexdigest()
    assert f({"code": "invalid_input", "message": msg}) == {
        "code": "invalid_input",
        "message_bytes": 10,
        "message_sha256": want_hash,
    }
    assert f({"code": "plain-looking-secret", "message": ""}) == {
        "code": "<unexpected>",
        "message_bytes": 0,
        "message_sha256": hashlib.sha256(b"").hexdigest(),
    }
    assert f({"code": {"a": 1}, "message": ["x"]}) == {}
    assert f({"code": 70}) == {}
    assert f(None) == {}
    assert f({"message": "\ud800"})["message_bytes"] == 1  # 孤立サロゲートでも落ちない


def test_package_files_validates_names_and_keeps_sizes(tmp_path: Path) -> None:
    """REQ-33: ファイル名は `[A-Za-z0-9._-]{1,64}` だけを記録し、他は `<unrecognized>`。"""
    d = tmp_path / "package"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"abcd")
    (d / "my secret name.bin").write_bytes(b"xy")
    (d / ("a" * 65)).write_bytes(b"z")
    (d / ("b" * 64)).write_bytes(b"zz")
    files = mod.package_files(d)
    assert files is not None
    names = sorted(f["name"] for f in files)
    assert names == sorted(["model.onnx", "<unrecognized>", "<unrecognized>", "b" * 64])
    assert {f["bytes"] for f in files if f["name"] == "model.onnx"} == {4}
    assert all(len(f["sha256"]) == 64 for f in files)
    assert mod.package_dir_stats(d) == (4, 4 + 2 + 1 + 2)
    # 通常ファイル以外（ディレクトリ・symlink）は黙って飛ばさず、どちらの関数も None（#362）
    (d / "sub").mkdir()
    assert mod.package_entries_regular(d) is False
    assert mod.package_files(d) is None
    assert mod.package_dir_stats(d) is None
    (d / "sub").rmdir()
    (d / "link").symlink_to("model.onnx")
    assert mod.package_entries_regular(d) is False
    assert mod.package_files(d) is None
    assert mod.package_dir_stats(d) is None
    assert mod.package_entries_regular(tmp_path / "missing") is None


def test_environment_text_and_ascii_int_validation() -> None:
    """REQ-33: 環境の文字列欄は安全な文字 64 字以内、数値欄は ASCII の 1〜20 桁だけ。"""
    for ok in ("Mac15,3", "Apple M3 Pro", "macOS", "15.0.1 (a)", "24A335", "a" * 64):
        assert mod.env_text(ok) == ok
    for bad in ("", "a" * 65, "x/y", "x\ny", "ok\n", "日本語", "a|b", "<b>", None):
        assert mod.env_text(bad) is None
    assert mod.ascii_int("12") == 12
    assert mod.ascii_int("1" * 20) == int("1" * 20)
    for bad in ("", "1" * 21, "１２", "²", "-1", "1.5", " 1", "1\n", None):
        assert mod.ascii_int(bad) is None  # `isdigit()` なら "²"・"１２" は通って `int()` が落ちる


def test_sanitize_record_redacts_strings_under_keys_outside_the_set() -> None:
    """REQ-33: 集合に無いキーの下の文字列は（パス文字なしでも）`<redacted>`。list も同様。"""
    rec = {
        "items": {
            "B": {
                "status": "ok",
                "note": "plain-looking-secret",
                "extra": {"deep": {"label": "beta"}},
                "tags": ["plain-looking-secret"],
                "reason": "x" * 201,
            }
        },
        "options": {"items": ["A", "B"], "other": ["A"]},
    }
    out = mod.sanitize_record(rec)
    assert out == {
        "items": {
            "B": {
                "status": "ok",
                "note": "<redacted>",
                "extra": {"deep": {"label": "<redacted>"}},
                "tags": ["<redacted>"],
                "reason": "<redacted>",
            }
        },
        "options": {"items": ["A", "B"], "other": ["<redacted>"]},
    }


def test_sanitize_record_direct_libraries_exception_only_at_its_position() -> None:
    """REQ-33: ライブラリ名の例外は `items.D.direct_libraries` の位置だけで効く。"""
    lib = "/usr/lib/libSystem.B.dylib"
    rec = {
        "items": {"D": {"direct_libraries": [lib]}, "C": {"direct_libraries": [lib]}},
        "direct_libraries": [lib],
        "environment": {"direct_libraries": lib},
        "steps": [{"direct_libraries": [lib]}],
    }
    out = mod.sanitize_record(rec)
    assert out["items"]["D"]["direct_libraries"] == [lib]
    assert out["items"]["C"]["direct_libraries"] == ["<redacted>"]
    assert out["direct_libraries"] == ["<redacted>"]
    assert out["environment"]["direct_libraries"] == "<redacted>"
    assert out["steps"] == [{"direct_libraries": ["<redacted>"]}]


def _representative_record(tmp_path: Path) -> dict[str, Any]:
    """実際の記録に現れる文字列の欄を一通り持つ記録（C は偽 CLI の実出力から作る）。"""
    c_res = mod.item_c(_c_ctx(tmp_path))
    assert c_res["status"] == "ok"
    c_dir = tmp_path / "w" / "C1"
    assert (c_dir / "project" / "package").is_dir()
    b_res = mod.item_b(_c_ctx(tmp_path / "b"))[0]
    assert b_res["status"] == "ok"
    items = {
        "A": {"status": "ok", "exit_code": 0, "skip_lines": 0, "stderr_bytes": 3},
        "B": b_res,
        "C": c_res,
        "D": {
            "status": "ok",
            "exit_code": 0,
            "linkage_tool": "otool",
            "direct_libraries": ["/usr/lib/libSystem.B.dylib"],
            "linkage_target_sha256": HEX64,
            "linkage_target_matches_cli": True,
        },
        "E": {"status": "ok", "records": 4, "input_sha256": HEX64, "max_abs_score_diff": 0.0},
        "F": {"status": "failed", "reason": "test_failures", "load_start": [1.5, 2.0, 3.0]},
    }
    return {
        "schema": "real-machine-check/1",
        "evidence_hint": "requires_human_review",
        "bin_override": False,
        "environment": {
            "hw_model": "Mac15,3",
            "cpu": "Apple M3 Pro",
            "ncpu": 12,
            "memory_bytes": 38654705664,
            "os_name": "macOS",
            "os_version": "15.0.1",
            "os_build": "24A335",
            "commit": "0" * 40,
            "worktree_clean": True,
            "commit_end": "0" * 40,
            "worktree_clean_end": True,
            "cli_end_sha256": HEX64,
            "commit_unchanged": True,
            "worktree_clean_unchanged": True,
            "cli_unchanged": True,
            "stable": True,
            "cli_origin": "built_by_script",
            "trainer_origin": "build_default",
            "started_local": "2026-10-05T10:00:00+0900",
            "ended_local": "2026-10-05T10:05:00+0900",
            "cli_sha256": HEX64,
            "cli_bytes": 3000000,
            "cli_profile": "release",
        },
        "inputs": {"train_records": 8, "definition_sha256": HEX64, "train_sha256": HEX64},
        "options": {"items": ["A", "B", "C", "D", "E", "F"], "repeat": 1, "p95_limit_us": 5},
        "items": items,
    }


def _diff_paths(a: Any, b: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """2 つの JSON 値の違う位置（キー・添字の並び）。"""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b)
        return [q for k in a for q in _diff_paths(a[k], b[k], (*path, k))]
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        return [q for i, (x, y) in enumerate(zip(a, b)) for q in _diff_paths(x, y, (*path, i))]  # noqa: B905
    return [] if a == b else [path]


def test_sanitize_record_keeps_a_normal_record_unchanged(tmp_path: Path) -> None:
    """REQ-33: 正常な記録は `sanitize_record` で変わらない（正当な欄を伏せる退行を止める）。

    infer の `command` は固定の表示名でパス文字を含まないため、変わる欄はない（#361）。
    schema は run() が伏せ処理後に戻す固定定数のため比較から除く。
    """
    rec = _representative_record(tmp_path)
    out = mod.sanitize_record(rec)
    changed = _diff_paths(
        {k: v for k, v in rec.items() if k != "schema"},
        {k: v for k, v in out.items() if k != "schema"},
    )
    assert changed == []
    assert mod.sanitize_record(out) == out
    # 工程の要約と package_files が実際に含まれていること（空の比較にしない）
    assert out["items"]["B"]["steps"][5]["summary"]["capacity"]["total_bytes"] == 10
    assert out["items"]["B"]["package_files"][0]["name"] == "model.onnx"
    assert out["items"]["C"]["capacity_limit"]["code"] == "limit_exceeded"


def test_deeply_nested_stdout_is_invalid_json_not_an_internal_error(tmp_path: Path) -> None:
    """REQ-39・REQ-21: 深すぎる入れ子の JSON は `RecursionError` でなく `invalid_json`。"""
    deep = "[" * 300000
    with pytest.raises(ValueError, match="too deep"):
        mod._loads(deep)
    f = tmp_path / "out.json"
    f.write_text(deep, encoding="utf-8")
    assert mod.parse_json_object(f, 1 << 20) is None
    cli = _fake_cli(tmp_path, f'cat "{f}"\nexit 0\n')
    d = tmp_path / "B"
    assert mod.stage_inputs(_ctx(tmp_path, cli), d, None)
    steps, _pkg, failure = mod.run_pipeline(_ctx(tmp_path, cli), d, "sample", {0})
    assert failure == {
        "status": "failed",
        "reason": "invalid_json",
        "step": "register",
        "exit_code": 0,
    }
    assert "summary" not in steps[0]


def test_unexpected_extra_fields_do_not_reach_step_summaries(tmp_path: Path) -> None:
    """REQ-33: CLI が許可リスト外の欄（本文・入れ子）を返しても、`steps[].summary` に出ない。"""
    cli = _fake_cli(
        tmp_path,
        'echo \'{"step":"register","status":"ok","options":3,'
        '"evaluation_defined":true,"note":"plain-looking-secret",'
        '"rows":[{"text":"private body"}]}\'\nexit 0\n',
    )
    d = tmp_path / "B"
    ctx = _ctx(tmp_path, cli)
    assert mod.stage_inputs(ctx, d, None)
    steps, _pkg, failure = mod.run_pipeline(ctx, d, "sample", {0})
    assert failure is not None
    assert steps[0]["summary"] == {
        "step": "register",
        "status": "ok",
        "options": 3,
        "evaluation_defined": True,
        "definition_sha256": None,
    }
    assert "plain-looking-secret" not in json.dumps(steps)


# ---- 判定の強化（型 2。REQ-21・REQ-28・REQ-30・REQ-31・REQ-33） ----


def _infer_obj(facts: Any, scores: dict[str, Any], label: Any) -> dict[str, Any]:
    return {"id": "x", "status": "ok", "predicted_label": label, "scores": scores}


def test_check_infer_output_requires_the_label_to_be_the_argmax() -> None:
    """REQ-21 §2-3: predicted_label は最大スコアの選択肢（同点は宣言順の先頭）。矛盾は不合格。"""
    f = mod.Facts(["alpha", "beta", "gamma"], 8, 4, False, None)
    chk = mod.check_infer_output
    s = {"alpha": 0.9, "beta": 0.05, "gamma": 0.05}
    # 背景: 最大でない gamma を予測しても、キー・合計が合っていれば通っていた
    assert chk(_infer_obj(f, s, "gamma"), f) is False
    assert chk(_infer_obj(f, s, "beta"), f) is False
    assert chk(_infer_obj(f, s, "alpha"), f) is True
    # 同点は宣言順の先頭（beta と gamma が同点の最大なら beta）
    tie = {"alpha": 0.2, "beta": 0.4, "gamma": 0.4}
    assert chk(_infer_obj(f, tie, "beta"), f) is True
    assert chk(_infer_obj(f, tie, "gamma"), f) is False
    flat = {k: 1 / 3 for k in f.option_ids}
    assert chk(_infer_obj(f, flat, "alpha"), f) is True
    assert chk(_infer_obj(f, flat, "beta"), f) is False
    # 許容差なしの比較: わずかに大きい後ろの選択肢が argmax
    close = {"alpha": 0.5 - 1e-12, "beta": 0.5 + 1e-12, "gamma": 0.0}
    assert chk(_infer_obj(f, close, "beta"), f) is True
    assert chk(_infer_obj(f, close, "alpha"), f) is False
    # label が選択肢 ID でない・文字列でない
    assert chk(_infer_obj(f, s, ["alpha"]), f) is False
    assert chk(_infer_obj(f, s, None), f) is False


def test_check_infer_output_requires_scores_within_zero_and_one() -> None:
    """REQ-21 §2-3: 各スコアは 0 以上 1 以下（合計が 1 でも範囲外は不合格）。境界値は合格。"""
    f = mod.Facts(["alpha", "beta"], 8, 4, False, None)
    chk = mod.check_infer_output
    assert chk(_infer_obj(f, {"alpha": 1.5, "beta": -0.5}, "alpha"), f) is False
    assert chk(_infer_obj(f, {"alpha": 1.0, "beta": 0.0}, "alpha"), f) is True
    assert chk(_infer_obj(f, {"alpha": 0, "beta": 1}, "beta"), f) is True
    assert chk(_infer_obj(f, {"alpha": 1.000001, "beta": 0.0}, "alpha"), f) is False
    assert chk(_infer_obj(f, {"alpha": True, "beta": 0.0}, "alpha"), f) is False


def test_p95_values_must_be_nonnegative_integers() -> None:
    """REQ-31 §2-4: p95_us・limit_us は 0 以上の整数（真偽値・負数・小数を拒否）。0 は合格。"""
    good = _cap(5, 40000000, False)
    j = _j95
    assert j(0, {"p95_us": 0, "limit_us": 0, "exceeded": False}, good, 0) is None
    # 背景: p95_us=-5 が通っていた
    assert j(0, {"p95_us": -5, "limit_us": 50000, "exceeded": False}, good, 50000) == (
        "unexpected_output"
    )
    assert j(0, {"p95_us": 1, "limit_us": -1, "exceeded": False}, good, -1) == "unexpected_output"
    for bad in (True, 1.5, "1"):
        p95 = {"p95_us": bad, "limit_us": 50000, "exceeded": False}
        assert j(0, p95, good, 50000) == "missing_field"
    assert mod.p95_summary({"p95_us": -5, "limit_us": 1, "exceeded": False}) is None
    assert mod.p95_summary({"p95_us": 0, "limit_us": 0, "exceeded": False}) == {
        "p95_us": 0,
        "limit_us": 0,
        "exceeded": False,
    }
    f = mod.Facts(["a"], 8, 4, False, 100)
    pkg = _pkg(f, infer_p95={"p95_us": -5, "limit_us": 100, "exceeded": False})
    assert mod._step_check("package", pkg, {0}, 0, f, 0) is False


@pytest.mark.parametrize(
    ("name", "patch"),
    [
        ("register", {"options": 3.0}),
        ("register", {"options": False}),
        ("train", {"candidate": False}),
        ("train", {"candidate": 0.0}),
        ("select", {"candidate": False}),
        ("select", {"candidate": 0.0}),
        ("evaluate", {"candidate": False}),
        ("evaluate", {"candidate": 0.0}),
        ("evaluate", {"n_total": 4.0}),
        ("evaluate", {"correct": 4.0}),
    ],
)
def test_reported_values_must_be_real_integers(name: str, patch: dict[str, Any]) -> None:
    """REQ-21 §2-5: `False == 0`・`0.0 == 0` のような型違いの一致を通さない。"""
    f = _facts()
    reports = _valid_reports(f)
    if name == "register":
        # 期待値と同じ数値の float でも通さない（fixture の選択肢数は 3）
        assert len(f.option_ids) == 3
    good = reports[name]
    assert mod._step_check(name, good, set(), 0, f, 0, "c1") is True
    assert mod._step_check(name, dict(good, **patch), set(), 0, f, 0, "c1") is False


def test_evaluate_accuracy_and_macro_f1_are_checked() -> None:
    """REQ-24 §2-6: accuracy は correct / n_total と一致（1e-9 以内）、macro_f1 は null か 0〜1。"""
    f = _facts()
    n = f.eval_records
    assert n >= 2
    good = _valid_reports(f)["evaluate"]
    chk = lambda **p: mod._step_check("evaluate", dict(good, **p), set(), 0, f, 0, "c1")  # noqa: E731
    assert chk() is True
    # 背景: accuracy:7.0・macro_f1:"x" が通っていた
    assert chk(accuracy=7.0, macro_f1="x") is False
    assert chk(accuracy=7.0) is False
    assert chk(macro_f1="x") is False
    # accuracy が correct / n_total と食い違う
    assert chk(correct=n - 1) is False
    assert chk(correct=n - 1, accuracy=(n - 1) / n) is True
    assert chk(accuracy=1.0 - 1e-8) is False
    assert chk(accuracy=1.0 - 1e-10) is True
    for bad in (float("nan"), float("inf"), None, True, "1"):
        assert chk(accuracy=bad) is False
    # macro_f1: null・0・1 は合格。範囲外・非有限・型違い・欄の欠落は不合格
    for ok in (None, 0, 0.0, 1, 1.0, 0.5):
        assert chk(macro_f1=ok) is True
    for bad in (-0.1, 1.0000001, float("nan"), float("inf"), True, [1], {"a": 1}):
        assert chk(macro_f1=bad) is False
    missing = dict(good)
    del missing["macro_f1"]
    assert mod._step_check("evaluate", missing, set(), 0, f, 0, "c1") is False


def test_kind_must_be_in_vocabulary_and_consistent_across_train_select_evaluate() -> None:
    """REQ-39・REQ-21 §2-2: train の kind は語彙内、select は train・evaluate は select と同じ。"""
    f = _facts()
    reports = _valid_reports(f)
    for kind in ("c1", "c3", "autoregressive"):
        chk = mod._step_check
        assert chk("train", dict(reports["train"], kind=kind), set(), 0, f, 0) is True
        assert chk("select", dict(reports["select"], kind=kind), set(), 0, f, 0, kind) is True
        assert chk("evaluate", dict(reports["evaluate"], kind=kind), set(), 0, f, 0, kind) is True
    assert mod._step_check("train", dict(reports["train"], kind="c9"), set(), 0, f, 0) is False
    # 語彙内でも前の工程と違えば不合格。前の kind が分からない（None）場合も通さない
    assert mod._step_check("select", reports["select"], set(), 0, f, 0, "c3") is False
    assert mod._step_check("evaluate", reports["evaluate"], set(), 0, f, 0, "c3") is False
    assert mod._step_check("select", reports["select"], set(), 0, f, 0, None) is False


@pytest.mark.parametrize(
    ("cfg", "step"),
    [
        ({"train_kind": "c9"}, "train"),
        ({"select_kind": "c3"}, "select"),
        ({"eval_kind": "c3"}, "evaluate"),
        ({"accuracy": 0.5}, "evaluate"),
        ({"macro_f1": 7}, "evaluate"),
    ],
)
def test_pipeline_stops_at_the_step_that_contradicts_the_previous_one(
    tmp_path: Path, cfg: dict[str, Any], step: str
) -> None:
    """REQ-21 §2-2・§2-6: kind の不一致・語彙外・accuracy の食い違いは、その工程で止まる。"""
    ctx = _c_ctx(tmp_path, **cfg)
    d = ctx.work / "B"
    assert mod.stage_inputs(ctx, d, None)
    steps, _pkg, failure = mod.run_pipeline(ctx, d, None, {0})
    assert failure is not None
    assert (failure["reason"], failure["step"]) == ("unexpected_output", step)
    assert steps[-1]["step"] == step


def test_package_report_requires_judgment_and_infer_p95_keys() -> None:
    """REQ-21 §2-7: exit 0 の package は judgment・infer_p95 のキーが必要（値は null でよい）。"""
    f = _facts()
    chk = lambda obj: mod._step_check("package", obj, {0}, 0, f, 0)  # noqa: E731
    assert chk(_pkg(f)) is True
    for key in ("judgment", "infer_p95"):
        broken = _pkg(f)
        del broken[key]
        assert chk(broken) is False
    only_status = {"step": "package", "status": "ok", "acceptance_defined": False}
    assert chk(only_status) is False
    assert chk(dict(_pkg(f), capacity=None)) is False


def test_package_with_acceptance_requires_pass_on_exit_0() -> None:
    """REQ-21・REQ-33 §2-7: 合否基準のある定義で exit 0 なら judgment は pass だけ。"""
    f = mod.Facts(["a", "b"], 8, 4, True, None)
    chk = lambda **p: mod._step_check("package", _pkg(f, **p), {0}, 0, f, 0)  # noqa: E731
    assert chk(judgment="pass", acceptance_defined=True) is True
    for bad in (None, "fail", "undeterminable", "PASS", True):
        assert chk(judgment=bad, acceptance_defined=True) is False
    assert chk(judgment="pass", acceptance_defined=False) is False


def test_package_exit_20_requires_the_fields_the_real_cli_always_emits() -> None:
    """REQ-21・REQ-30・REQ-31 §2-7: exit 20 は code・step・capacity・infer_p95 と超過の実在。"""
    f = _facts()
    over = {
        "code": "limit_exceeded",
        "message": "m",
        "step": "package",
        "capacity": _cap_json(40000001, 40000000, True),
        "infer_p95": None,
    }
    chk = lambda obj: mod._step_check("package", obj, {20}, 20, f, 0)  # noqa: E731
    assert chk(over) is True
    assert chk(dict(over, code="runtime_error")) is False
    assert chk({k: v for k, v in over.items() if k != "code"}) is False
    assert chk({k: v for k, v in over.items() if k != "infer_p95"}) is False
    assert chk(dict(over, step="train")) is False
    # exit 20 なのにどの上限も超過していない
    assert chk(dict(over, capacity=_cap_json())) is False
    # exit 20 は allowed に無ければ不合格
    assert mod._step_check("package", over, {0}, 20, f, 0) is False


def test_p95_over_with_exit_20_is_accepted_only_with_consistent_exceeded() -> None:
    """REQ-31 §2-4・§2-7: p95 超過の exit 20 は、定義の limit_us・exceeded と整合したときだけ。"""
    f = mod.Facts(["a"], 8, 4, False, 100)
    mk = lambda **p: {  # noqa: E731
        "code": "limit_exceeded",
        "step": "package",
        "capacity": _cap_json(),
        "infer_p95": dict({"p95_us": 101, "limit_us": 100, "exceeded": True}, **p),
    }
    chk = lambda obj: mod._step_check("package", obj, {0, 20}, 20, f, 0)  # noqa: E731
    assert chk(mk()) is True
    assert chk(mk(p95_us=100)) is False  # 境界: p95_us == limit_us は超過でない
    assert chk(mk(limit_us=99, p95_us=100)) is False  # 定義の上限（100）と違う
    assert chk(mk(exceeded=False)) is False


def test_capacity_component_sum_must_match_total_for_every_package_check() -> None:
    """REQ-30 §2-1: 5 項目の合計が total_bytes と一致しなければ、exit 0・20 のどちらも不合格。"""
    f = _facts()
    bad = _cap_json(sum_delta=1)
    assert mod.capacity_sum_matches(mod.capacity_summary({"capacity": _cap_json()})) is True
    assert mod.capacity_sum_matches(mod.capacity_summary({"capacity": bad})) is False
    assert mod._step_check("package", _pkg(f, capacity=bad), {0}, 0, f, 0) is False
    over = {"code": "limit_exceeded", "step": "package", "infer_p95": None}
    ok_cap = _cap_json(40000001, 40000000, True)
    bad_cap = _cap_json(40000001, 40000000, True, sum_delta=-1)
    assert mod._step_check("package", dict(over, capacity=ok_cap), {20}, 20, f, 0) is True
    assert mod._step_check("package", dict(over, capacity=bad_cap), {20}, 20, f, 0) is False
    # B は項目側で `capacity_sum_mismatch` として判定するため、工程側の照合を外せる
    assert mod._step_check("package", _pkg(f, capacity=bad), {0}, 0, f, 0, None, False) is True


def test_item_b_rejects_mismatching_capacity_sum_with_its_own_reason(tmp_path: Path) -> None:
    """REQ-30 §2-1: B は内訳の合計の不一致を `capacity_sum_mismatch` として失敗にする。"""
    res, ok = mod.item_b(_c_ctx(tmp_path, comp_delta=1))
    assert (ok, res["status"], res["reason"]) == (False, "failed", "capacity_sum_mismatch")


def test_item_c1_rejects_mismatching_capacity_sum(tmp_path: Path) -> None:
    """REQ-30 §2-1: C-1 でも内訳の合計の不一致は unexpected_output の失敗。"""
    res = mod.item_c(_c_ctx(tmp_path, comp_delta=1))
    assert (res["status"], res["reason"], res["case"]) == ("failed", "unexpected_output", "C-1")
    assert res["step"] == "package"


def test_item_c2_rejects_mismatching_capacity_sum(tmp_path: Path) -> None:
    """REQ-30 §2-1: C-2（exit 20 の JSON）でも内訳の合計の不一致は失敗。"""
    ctx = _c_ctx(tmp_path)
    d = ctx.work / "C2"
    assert mod.stage_inputs(ctx, d, {"max_package_bytes": ctx.package_limit_bytes})
    _steps, _pkg_obj, failure = mod.run_pipeline(ctx, d, None, {20})
    assert failure is None
    ctx = _c_ctx(tmp_path / "bad", comp_delta=1)
    d = ctx.work / "C2"
    assert mod.stage_inputs(ctx, d, {"max_package_bytes": ctx.package_limit_bytes})
    _steps, _pkg_obj, failure = mod.run_pipeline(ctx, d, None, {20})
    assert failure is not None
    assert (failure["reason"], failure["step"]) == ("unexpected_output", "package")
    assert failure["code"] == "limit_exceeded"


def test_expected_limit_bytes_comes_from_the_definition_or_the_default(tmp_path: Path) -> None:
    """REQ-30 §2-8: 期待する limit_bytes は定義の max_package_bytes、無ければ 40_000_000。"""
    assert mod.DEFAULT_CAPACITY_LIMIT_BYTES == 40_000_000
    assert _facts().limit_bytes == 40_000_000
    ctx = _c_ctx(tmp_path)
    d = tmp_path / "defn"
    assert mod.stage_inputs(ctx, d, {"max_package_bytes": 1234})
    facts = mod.read_facts(d)
    assert facts is not None
    assert facts.limit_bytes == 1234
    # 背景: B で limit_bytes=7 が通っていた
    f = _facts()
    assert mod._step_check("package", _pkg(f, capacity=_cap_json(limit=7)), {0}, 0, f, 0) is False
    assert mod._step_check("package", _pkg(f, capacity=_cap_json(limit=40000000)), {0}, 0, f, 0)
    f2 = mod.Facts(f.option_ids, f.train_records, f.eval_records, False, None, 1234)
    assert mod._step_check("package", _pkg(f2, capacity=_cap_json(limit=1234)), {0}, 0, f2, 0)
    assert not mod._step_check("package", _pkg(f2, capacity=_cap_json()), {0}, 0, f2, 0)


def test_item_b_rejects_limit_bytes_that_differ_from_the_default(tmp_path: Path) -> None:
    """REQ-30 §2-8: B で CLI が報告した limit_bytes が既定値と違えば失敗する。"""
    res, ok = mod.item_b(_c_ctx(tmp_path, limit1=7))
    assert (ok, res["status"], res["reason"], res["step"]) == (
        False,
        "failed",
        "unexpected_output",
        "package",
    )
    res, ok = mod.item_b(_c_ctx(tmp_path / "ok"))
    assert (ok, res["status"]) == (True, "ok")
    assert res["capacity"]["limit_bytes"] == 40000000
    assert res["capacity_sum_matches_total"] is True


def test_item_b_requires_published_files_to_match_total_bytes(tmp_path: Path) -> None:
    """REQ-30 §2-9: package/ 直下の通常ファイルの合計が total_bytes と一致しなければ B は失敗。"""
    res, ok = mod.item_b(_c_ctx(tmp_path / "a", actual_bytes=11))
    assert (ok, res["status"], res["reason"], res["step"]) == (
        False,
        "failed",
        "unexpected_output",
        "package",
    )
    res, ok = mod.item_b(_c_ctx(tmp_path / "b", actual_bytes=9))
    assert (ok, res["reason"]) == (False, "unexpected_output")
    res, ok = mod.item_b(_c_ctx(tmp_path / "c", actual_bytes=10))
    assert ok is True
    assert [f["bytes"] for f in res["package_files"]] == [10]


def test_item_c1_requires_published_files_to_match_total_bytes(tmp_path: Path) -> None:
    """REQ-30 §2-9: C-1 の exit 0 も実ファイルの合計が total_bytes と一致すること。"""
    res = mod.item_c(_c_ctx(tmp_path / "a", actual_bytes=11))
    assert (res["status"], res["reason"], res["case"]) == ("failed", "unexpected_output", "C-1")
    res = mod.item_c(_c_ctx(tmp_path / "b", actual_bytes=0))
    assert (res["status"], res["reason"], res["case"]) == ("failed", "unexpected_output", "C-1")
    res = mod.item_c(_c_ctx(tmp_path / "c", actual_bytes=10))
    assert res["status"] == "ok"
    assert "package_files" not in res
    assert "package_files" not in json.dumps(res)
    # p95 超過の exit 20（package/ が作られない）では実ファイルの照合は行わない
    res = mod.item_c(_c_ctx(tmp_path / "d", rc1=20, p95_us=60000, p95_exceeded=True))
    assert res["status"] == "ok"


def test_item_c2_code_is_recorded_only_from_the_vocabulary(tmp_path: Path) -> None:
    """REQ-21・REQ-33: C-2 の code は語彙内ならその値、語彙外なら `<unexpected>`。"""
    res = mod.item_c(_c_ctx(tmp_path / "a"))
    assert res["capacity_limit"]["code"] == "limit_exceeded"
    res = mod.item_c(_c_ctx(tmp_path / "b", c2_code="plain-looking-secret"))
    # 語彙外の code は exit 20 の JSON として不合格（工程で止まる）で、値は記録に出ない
    assert (res["status"], res["reason"], res["case"]) == ("failed", "unexpected_output", "C-2")
    assert res["code"] == "<unexpected>"
    assert "plain-looking-secret" not in json.dumps(res)


# ---- 定数の機械照合（fixtures・Rust のソースと一致すること） ----


def test_code_vocabulary_matches_the_exit_code_fixture() -> None:
    """REQ-21: `CODE_VOCAB` は終了コード 7 種の fixture の name と一致する。"""
    fx = json.loads((REPO / "fixtures" / "exitcode" / "exit_codes.json").read_text("utf-8"))
    assert {e["name"] for e in fx["exit_codes"]} == set(mod.CODE_VOCAB)
    assert len(mod.CODE_VOCAB) == 7


def test_kind_vocabulary_covers_default_candidates_and_the_guard_allowlist() -> None:
    """REQ-19・REQ-39: `KIND_VOCAB` は既定候補の kind を含み、ガード層の許可リストと一致する。"""
    fx = json.loads(
        (REPO / "fixtures" / "train_contract" / "default_candidates.json").read_text("utf-8")
    )
    assert {c["kind"] for c in fx["default_candidates"]} <= set(mod.KIND_VOCAB)
    src = (REPO / "crates" / "guard" / "src" / "kind.rs").read_text("utf-8")
    m = re.search(r"const SUPPORTED_KINDS: \[&str; \d+\] = \[([^\]]*)\];", src)
    assert m is not None, "SUPPORTED_KINDS が見つからない"
    assert set(re.findall(r'"([a-z0-9_]+)"', m.group(1))) == set(mod.KIND_VOCAB)


def test_score_sum_tolerance_matches_the_shared_fixture() -> None:
    """REQ-21: `SCORE_SUM_TOLERANCE` は共有 fixture の値と一致する。"""
    fx = json.loads(
        (REPO / "fixtures" / "score_tolerance" / "score_sum_tolerance.json").read_text("utf-8")
    )
    assert mod.SCORE_SUM_TOLERANCE == fx["score_sum_tolerance"]
    assert mod.SCORE_SUM_TOLERANCE == 1e-6


def test_default_limit_bytes_matches_the_rust_constant() -> None:
    """REQ-30: 既定の limit_bytes は `package.rs` の `DEFAULT_CAPACITY_LIMIT_BYTES` と一致する。"""
    src = (REPO / "crates" / "cli" / "src" / "stages" / "package.rs").read_text("utf-8")
    m = re.search(r"const DEFAULT_CAPACITY_LIMIT_BYTES: u64 = ([0-9_]+);", src)
    assert m is not None, "DEFAULT_CAPACITY_LIMIT_BYTES が見つからない"
    assert int(m.group(1).replace("_", "")) == mod.DEFAULT_CAPACITY_LIMIT_BYTES


def test_staging_and_default_id_constants_match_the_rust_sources() -> None:
    """REQ-30・REQ-33・#362: 組み立て先ディレクトリ名と既定 id は Rust 側の定数と一致する。"""
    project = (REPO / "crates" / "cli" / "src" / "project.rs").read_text("utf-8")
    m = re.search(r'pub const PACKAGE_STAGING_DIR: &str = "([^"]+)";', project)
    assert m is not None, "PACKAGE_STAGING_DIR が見つからない"
    assert m.group(1) == mod.PACKAGE_STAGING_DIR == "package.staging"
    infer = (REPO / "crates" / "cli" / "src" / "stages" / "infer.rs").read_text("utf-8")
    m = re.search(r'const DEFAULT_TEXT_ID: &str = "([^"]+)";', infer)
    assert m is not None, "DEFAULT_TEXT_ID が見つからない"
    assert m.group(1) == mod.DEFAULT_TEXT_ID == "input"


def test_item_b_rejects_non_regular_entries_and_file_count_mismatch(tmp_path: Path) -> None:
    """REQ-30・#362: B は package/ の通常ファイル以外・file_count の合計との不一致で失敗。"""
    for kind in ("symlink", "dir"):
        res, ok = mod.item_b(_c_ctx(tmp_path / kind, package_kind=kind))
        assert (ok, res["status"], res["reason"]) == (False, "failed", "package_entry_not_regular")
    res, ok = mod.item_b(_c_ctx(tmp_path / "fc", fc_delta=1))
    assert (ok, res["reason"], res["step"]) == (False, "unexpected_output", "package")
    res, ok = mod.item_b(_c_ctx(tmp_path / "ok"))
    assert (ok, res["status"], len(res["package_files"])) == (True, "ok", 1)


def test_item_b_rejects_a_default_id_other_than_input(tmp_path: Path) -> None:
    """REQ-33・#362: B の単発 infer は `--id` なしなので、id が既定値 input でなければ失敗。"""
    res, ok = mod.item_b(_c_ctx(tmp_path, default_id="x"))
    assert (ok, res["reason"], res["step"]) == (False, "unexpected_output", "infer")


def test_item_c1_rejects_non_regular_entries_and_file_count_mismatch(tmp_path: Path) -> None:
    """REQ-30・#362: C-1（exit 0）も package/ の通常ファイル以外・ファイル数の不一致で失敗。"""
    for name, over in (
        ("s", {"package_kind": "symlink"}),
        ("d", {"package_kind": "dir"}),
        ("f", {"fc_delta": 1}),
    ):
        res = mod.item_c(_c_ctx(tmp_path / name, **over))
        assert (res["status"], res["reason"], res["case"]) == ("failed", "unexpected_output", "C-1")


def test_item_c_fails_when_package_staging_is_left(tmp_path: Path) -> None:
    """REQ-30・#362: package.staging/ が残っていれば C は staging_left（exit 0・20 とも）。"""
    res = mod.item_c(_c_ctx(tmp_path / "a", staging="C1"))
    assert (res["status"], res["reason"], res["case"]) == ("failed", "staging_left", "C-1")
    over = {"rc1": 20, "p95_us": 60000, "p95_exceeded": True}
    res = mod.item_c(_c_ctx(tmp_path / "b", staging="C1", **over))
    assert (res["status"], res["reason"], res["case"]) == ("failed", "staging_left", "C-1")
    res = mod.item_c(_c_ctx(tmp_path / "c", staging="C2"))
    assert (res["status"], res["reason"]) == ("failed", "staging_left")
    assert res["capacity_limit"]["package_staging_present"] is True
    res = mod.item_c(_c_ctx(tmp_path / "d"))
    assert res["status"] == "ok"
    assert res["p95"]["package_staging_present"] is False
    assert res["capacity_limit"]["package_staging_present"] is False


def test_item_f_compares_each_run_with_the_listed_test_count(tmp_path: Path) -> None:
    """REQ-39・#362: F は `--list` の件数と各回の passed を照合し、失敗の内訳を欄ごとに数える。"""
    ctx = _ctx(tmp_path / "w", _fake_cli(tmp_path, "exit 0\n"))
    ctx.work.mkdir()
    ctx.repeat = 1

    def run(name: str, body: str, n: int = 12) -> dict[str, Any]:
        # `--no-run` のビルドは常に成功させ、本番の回の挙動だけを `body` で変える
        prefix = 'case "$*" in *--no-run*) exit 0;; esac\n' + _list_script(n)
        ctx.cargo_cmd = str(_script(tmp_path, name, prefix + body))
        shutil.rmtree(ctx.work / "F", ignore_errors=True)
        return mod.item_f(ctx)

    def line(passed: int, failed: int = 0, ignored: int = 0) -> str:
        return (
            f"echo 'test result: ok. {passed} passed; {failed} failed; {ignored} ignored; "
            "0 measured; 0 filtered out'\nexit 0\n"
        )

    res = run("ok", line(12))
    assert (res["status"], res["expected_tests"], res["passed"]) == ("ok", 12, 1)
    res = run("short", line(5))
    assert (res["status"], res["failed"], res["count_mismatch"], res["no_tests"]) == (
        "failed",
        1,
        1,
        0,
    )
    res = run("ignored", line(12, 0, 1))
    assert (res["failed"], res["count_mismatch"]) == (1, 1)
    res = run("follow-list", line(7), n=7)
    assert (res["status"], res["expected_tests"]) == ("ok", 7)
    res = run("zero", line(0))
    assert (res["no_tests"], res["count_mismatch"]) == (1, 0)
    res = run("killed", "kill -9 $PPID\nexit 0\n")
    assert (res["killed"], res["failed"]) == (1, 1)
    res = run("fail-rc", "exit 101\n")
    assert (res["failed"], res["killed"], res["output_limit"], res["spawn_error"]) == (1, 0, 0, 0)
    assert res["no_tests"] == 0
    res = run("empty-list", line(12), n=0)
    assert res == {"status": "failed", "reason": "no_tests_listed"}
    ctx.cargo_cmd = str(
        _script(tmp_path, "list-fail", 'case "$*" in *--list*) exit 101;; esac\nexit 0\n')
    )
    shutil.rmtree(ctx.work / "F")
    assert mod.item_f(ctx) == {"status": "failed", "reason": "list_failed", "exit_code": 101}


def test_sanitize_record_checks_every_direct_library_regardless_of_path_chars() -> None:
    """REQ-33: ライブラリ名の要素はパス文字の有無によらず形式・`..`・長さを満たすものだけ残る。"""
    lib = "/usr/lib/libSystem.B.dylib"
    rec = {
        "items": {
            "D": {
                "direct_libraries": [
                    lib,
                    "SECRETBODY",
                    "libX.dylib\n",
                    "/usr/lib/libx.dylib\n",
                    "/usr/lib/" + "a" * 120,
                    "",
                ]
            }
        }
    }
    got = mod.sanitize_record(rec)["items"]["D"]["direct_libraries"]
    assert got == [lib] + ["<redacted>"] * 5


def test_sanitize_record_redacts_keys_outside_the_key_rule() -> None:
    """REQ-33: dict のキーは `^[A-Za-z0-9_.-]{1,64}$` のときだけ残り、他は `<redacted>`。"""
    got = mod.sanitize_record({"ok_key-1.x": 1, "a b": 2, "k" * 65: 3, "日本語": 4, "": 5})
    assert got == {"ok_key-1.x": 1, "<redacted>": 5}
    assert mod.sanitize_record({"k" * 64: 1}) == {"k" * 64: 1}
    assert mod.sanitize_record({"k\n": 1}) == {"<redacted>": 1}


def test_huge_integers_do_not_raise_and_fail_range_checks() -> None:
    """REQ-21・REQ-33: `float` へ変換できない巨大な整数は例外を出さず、範囲の検査で不合格。"""
    huge = 10**400
    assert mod._is_finite_number(huge) is False
    assert mod._unit_number(huge) is False
    f = _facts()
    ids = f.option_ids
    got = mod.summarize_step("evaluate", {"accuracy": huge, "macro_f1": huge}, ids)
    assert (got["accuracy"], got["macro_f1"]) == (None, None)
    good = _valid_reports(f)["evaluate"]
    assert mod.check_stage_report("evaluate", dict(good, accuracy=huge), f, 0, "c1") is False
    assert mod._step_check("evaluate", dict(good, macro_f1=huge), set(), 0, f, 0, "c1") is False
    ok = {"id": "x", "status": "ok", "predicted_label": ids[0], "scores": _scores(f)}
    assert mod.check_infer_output(dict(ok, scores=dict(_scores(f), **{ids[0]: huge})), f) is False
    # E の突き合わせも例外にならず、非有限の扱いになる
    b = {"a": {"predicted_label": "x", "scores": {"x": huge}}}
    s = {"a": {"predicted_label": "x", "scores": {"x": 0.5}}}
    assert mod.compare_infer(b, s)["scores_nonfinite"] == 1


def test_collect_environment_validates_probe_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-33: `collect_environment` は外部コマンドの出力を `env_text`・`ascii_int` へ通す。"""
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    ctx = _ctx(tmp_path, tmp_path / "no-such-cli")
    names_text = ("hw", "cpu", "osn", "osv", "osb")
    names_int = ("ncpu", "mem")

    def fake(values: dict[str, str]) -> Any:
        return lambda _w, _r, argv, name: values.get(name)

    good = {
        "hw": "Mac16,6",
        "cpu": "Apple M4 Max",
        "osn": "macOS",
        "osv": "27.0",
        "osb": "26A428",
        "ncpu": "16",
        "mem": "68719476736",
    }
    monkeypatch.setattr(mod, "_probe", fake(good))
    env = mod.collect_environment(ctx, None)
    assert (env["hw_model"], env["cpu"], env["os_name"]) == ("Mac16,6", "Apple M4 Max", "macOS")
    assert (env["os_version"], env["os_build"]) == ("27.0", "26A428")
    assert (env["ncpu"], env["memory_bytes"]) == (16, 68719476736)
    for bad in ("a|b", "a" * 65, "ok\nx", "日本語"):
        monkeypatch.setattr(mod, "_probe", fake({n: bad for n in names_text}))
        env = mod.collect_environment(ctx, None)
        got = [env[k] for k in ("hw_model", "cpu", "os_name", "os_version", "os_build")]
        assert got == [None] * 5
    for bad in ("１６", "12abc"):
        monkeypatch.setattr(mod, "_probe", fake({n: bad for n in names_int}))
        env = mod.collect_environment(ctx, None)
        assert (env["ncpu"], env["memory_bytes"]) == (None, None)


# --- #359: 記録の書き込みと rc の読み取りの硬化（REQ-39・REQ-33。テストハーネス） ---


def test_write_atomic_replaces_and_keeps_old_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: record は一時ファイル経由で置換し、失敗しても既存 JSON を壊さない。"""
    p = tmp_path / "record.json"
    mod.write_atomic(p, '{"a": 1}\n')
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}

    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("boom")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="boom"):
        mod.write_atomic(p, '{"a": 2}\n')
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}
    assert [x.name for x in tmp_path.iterdir()] == ["record.json"]


def test_write_atomic_fsync_failure_on_new_file_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: 新規作成の途中失敗で壊れた record.json を残さない。"""

    def boom(_fd: int) -> None:
        raise OSError("x")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="x"):
        mod.write_atomic(tmp_path / "record.json", "{}\n")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["record.json", "record.md"])
def test_write_atomic_refuses_symlink_target(tmp_path: Path, name: str) -> None:
    """REQ-39: record のパスに symlink が置かれていたら辿らず失敗し、標的は不変。"""
    target = tmp_path / "victim"
    target.write_text("keep", encoding="utf-8")
    (tmp_path / name).symlink_to(target)
    with pytest.raises(OSError, match="not a regular file"):
        mod.write_atomic(tmp_path / name, "new")
    assert target.read_text(encoding="utf-8") == "keep"
    assert (tmp_path / name).is_symlink()


def test_write_text_nofollow_refuses_symlink(tmp_path: Path) -> None:
    """REQ-39: symlink へは書かない。通常ファイルへは書ける。"""
    target = tmp_path / "victim"
    target.write_text("keep", encoding="utf-8")
    (tmp_path / "l").symlink_to(target)
    with pytest.raises(OSError, match=r"Too many|symbolic|loop"):
        mod.write_text_nofollow(tmp_path / "l", "x")
    assert target.read_text(encoding="utf-8") == "keep"
    mod.write_text_nofollow(tmp_path / "ok", "y")
    assert (tmp_path / "ok").read_text(encoding="utf-8") == "y"


def test_run_cmd_refuses_symlinked_stdout_file(tmp_path: Path) -> None:
    """REQ-39: stdout ファイルが symlink なら spawn_error で、標的は不変。"""
    target = tmp_path / "victim"
    target.write_text("keep", encoding="utf-8")
    (tmp_path / "o").symlink_to(target)
    r = mod.run_cmd(["/bin/echo", "hi"], tmp_path, tmp_path / "o", tmp_path / "e", 5, 100, 100)
    assert (r.exit_code, r.reason) == (None, "spawn_error")
    assert target.read_text(encoding="utf-8") == "keep"


def test_run_cmd_does_not_block_on_preplaced_fifo_output(tmp_path: Path) -> None:
    """REQ-39: 出力先に FIFO が先置きされても open で固まらず spawn_error になる（#359）。"""
    os.mkfifo(tmp_path / "o")
    t0 = time.monotonic()
    r = mod.run_cmd(["/bin/echo", "hi"], tmp_path, tmp_path / "o", tmp_path / "e", 5, 100, 100)
    assert (r.exit_code, r.reason) == (None, "spawn_error")
    assert time.monotonic() - t0 < 4


def test_write_text_nofollow_refuses_fifo_and_hardlink_without_truncating(tmp_path: Path) -> None:
    """REQ-39: FIFO は即失敗、ハードリンクの標的は切り詰めずに拒否する（#359）。"""
    os.mkfifo(tmp_path / "f")
    with pytest.raises(OSError, match=r"Errno 6") as ei:
        mod.write_text_nofollow(tmp_path / "f", "x")
    assert ei.value.errno == errno.ENXIO
    target = tmp_path / "victim"
    target.write_text("keep", encoding="utf-8")
    os.link(target, tmp_path / "hl")
    with pytest.raises(OSError, match="hard links"):
        mod.write_text_nofollow(tmp_path / "hl", "x")
    assert target.read_text(encoding="utf-8") == "keep"


def test_read_rc_only_accepts_regular_files(tmp_path: Path) -> None:
    """REQ-39: rc は通常ファイルだけを読む。symlink・FIFO・ディレクトリ・超過は None。"""
    ok = tmp_path / "ok"
    ok.write_text("7", encoding="utf-8")
    assert mod._read_rc(ok) == 7
    big = tmp_path / "big"
    big.write_text("1" * 4, encoding="utf-8")
    assert mod._read_rc(big) is None
    link = tmp_path / "link"
    link.symlink_to(ok)
    assert mod._read_rc(link) is None
    d = tmp_path / "dir"
    d.mkdir()
    assert mod._read_rc(d) is None
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    t0 = time.monotonic()
    assert mod._read_rc(fifo) is None
    assert time.monotonic() - t0 < 2


def test_run_cmd_removes_preplaced_rc_symlink(tmp_path: Path) -> None:
    """REQ-39: 起動前に rc が symlink なら除去され、標的へ書かれず成功する。"""
    target = tmp_path / "victim"
    target.write_text("keep", encoding="utf-8")
    (tmp_path / "o.rc").symlink_to(target)
    r = mod.run_cmd(["/bin/echo", "hi"], tmp_path, tmp_path / "o", tmp_path / "e", 5, 100, 100)
    assert (r.exit_code, r.reason) == (0, None)
    assert target.read_text(encoding="utf-8") == "keep"


def test_run_cmd_child_swapping_rc_for_symlink_is_killed_not_ok(tmp_path: Path) -> None:
    """REQ-39: 子が rc を symlink にすり替えても辿らず、結果は killed（成功扱いにしない）。"""
    target = tmp_path / "victim"
    target.write_text("keep", encoding="utf-8")
    script = f"ln -s {target} {tmp_path}/o.rc"
    r = mod.run_cmd(
        ["/bin/sh", "-c", script], tmp_path, tmp_path / "o", tmp_path / "e", 5, 100, 4096
    )
    assert (r.exit_code, r.reason) == (None, "killed")
    assert target.read_text(encoding="utf-8") == "keep"


def test_run_cmd_child_forging_rc_file_with_zero_is_killed_not_ok(tmp_path: Path) -> None:
    """REQ-39: 子が rc に偽の終了コード 0 を置いて失敗しても、成功扱いにせず killed。"""
    script = f"printf 0 > {tmp_path}/o.rc; exit 3"
    r = mod.run_cmd(
        ["/bin/sh", "-c", script], tmp_path, tmp_path / "o", tmp_path / "e", 5, 100, 4096
    )
    assert (r.exit_code, r.reason) == (None, "killed")
    assert not (tmp_path / "o.rc").exists()


def test_run_cmd_child_making_rc_fifo_does_not_block(tmp_path: Path) -> None:
    """REQ-39: 子が rc を FIFO にしても成功扱いにせず、期限内に止まる（timeout）。"""
    script = f"mkfifo {tmp_path}/o.rc"
    t0 = time.monotonic()
    r = mod.run_cmd(
        ["/bin/sh", "-c", script], tmp_path, tmp_path / "o", tmp_path / "e", 1, 100, 100
    )
    assert (r.exit_code, r.reason) == (None, "timeout")
    assert time.monotonic() - t0 < 4


def test_run_cmd_rc_path_is_directory_returns_spawn_error_without_running(
    tmp_path: Path,
) -> None:
    """REQ-39: rc のパスがディレクトリでも例外を出さず spawn_error。子は起動しない。"""
    (tmp_path / "o.rc").mkdir()
    marker = tmp_path / "marker"
    r = mod.run_cmd(
        ["/bin/sh", "-c", f"touch {marker}"],
        tmp_path,
        tmp_path / "o",
        tmp_path / "e",
        5,
        100,
        100,
    )
    assert (r.exit_code, r.reason, r.out_bytes, r.err_bytes) == (
        None,
        "spawn_error",
        0,
        0,
    )
    assert not marker.exists()


def test_read_capped_ex_distinguishes_limit_and_unreadable(tmp_path: Path) -> None:
    """REQ-39: 上限超過（output_limit）と読めない（output_unreadable）を区別する。"""
    f = tmp_path / "f"
    f.write_text("hello", encoding="utf-8")
    assert mod.read_capped_ex(f, 100) == ("hello", None)
    assert mod.read_capped_ex(f, 2) == (None, "output_limit")
    assert mod.read_capped_ex(tmp_path / "none", 100) == (None, "output_unreadable")


@pytest.mark.parametrize("item", ["item_a", "item_d"])
def test_unreadable_log_is_output_unreadable_not_output_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, item: str
) -> None:
    """REQ-39: A・D のログが読めないときは output_unreadable（output_limit に見せない）。"""
    monkeypatch.setattr(mod, "run_cmd", lambda *a, **k: mod.RunResult(0, None, 0, 0))
    res = getattr(mod, item)(_ctx(tmp_path, tmp_path / "cli"))
    assert res["status"] == "failed"
    assert res["reason"] == "output_unreadable"


# --------------------------------------------------------------------------------------
# 終了時の環境の再採取（#360。REQ-21・REQ-32）
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "want"),
    [
        ("a" * 40, "a" * 40, True),
        ("a" * 40, "b" * 40, False),
        ("a" * 40, None, False),
        (None, "a" * 40, False),
        (None, None, None),
        (True, True, True),
        (True, False, False),
        (False, False, True),
        (HEX64, HEX64, True),
        (HEX64, "b" * 64, False),
    ],
)
def test_compare_start_end_is_fail_closed(start: Any, end: Any, want: Any) -> None:
    """REQ-21: 片方だけ取れないのは同一と確認できず False。両方 None だけが None。"""
    assert mod.compare_start_end(start, end) is want


@pytest.mark.parametrize(
    ("flags", "want"),
    [
        ((True, True, True), True),
        ((True, None, False), False),
        ((True, None, None), None),
        ((None, None, None), None),
    ],
)
def test_judge_environment_stable_aggregates(flags: Any, want: Any) -> None:
    """REQ-21: 1 つでも False なら False、全部 True のときだけ True、他は None（未確認）。"""
    env = dict(zip(("commit_unchanged", "worktree_clean_unchanged", "cli_unchanged"), flags))  # noqa: B905
    assert mod.judge_environment_stable(env) is want


def _ok_item(ctx: Any, name: str, b_ok: bool) -> Any:
    return {"status": "ok"}, True


def test_end_environment_unchanged_is_stable_and_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-21: 変化が無ければ stable が true で exit 0。"""
    monkeypatch.setattr(mod, "_probe", lambda *a, **k: "0" * 40)
    monkeypatch.setattr(mod, "_worktree_clean", lambda *a, **k: True)
    rc, rec = _run_in_process(tmp_path, "B", _ok_item, monkeypatch)
    env = rec["environment"]
    assert rc == 0
    assert env["stable"] is True
    assert env["commit_end"] == "0" * 40
    assert env["cli_end_sha256"] == env["cli_sha256"]


def test_cli_rebuilt_during_run_is_judged_fail_and_item_status_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-32: 実行中に CLI が書き換わると、項目が ok でも exit 10・stable false。"""

    def fake(ctx: Any, name: str, b_ok: bool) -> Any:
        ctx.bin.write_bytes(b"#!/bin/sh\nexit 1\n")
        return {"status": "ok"}, True

    rc, rec = _run_in_process(tmp_path, "B", fake, monkeypatch)
    env = rec["environment"]
    assert rc == 10
    assert json.loads(capsys.readouterr().out) == {
        "code": "judged_fail",
        "message": "environment changed during the run",
        "record": "record.json",
    }
    assert env["cli_unchanged"] is False
    assert env["stable"] is False
    assert env["cli_end_sha256"] != env["cli_sha256"]
    assert rec["items"]["B"] == {"status": "ok"}
    assert "実行中に環境が変わった" in (tmp_path / "w" / "record.md").read_text()


def _sequence(monkeypatch: pytest.MonkeyPatch, name: str, values: list[Any]) -> None:
    """name の関数を、呼ばれるたび values を順に返す偽に差し替える（尽きたら最後の値）。"""
    it = iter(values)
    last: list[Any] = [values[-1]]

    def fake(*a: Any, **k: Any) -> Any:
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    monkeypatch.setattr(mod, name, fake)


def test_commit_change_during_run_is_judged_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-21: 開始時と終了時で commit が違えば stable false・exit 10。"""
    _sequence(monkeypatch, "_probe", ["0" * 40, "1" * 40])
    monkeypatch.setattr(mod, "_worktree_clean", lambda *a, **k: True)
    rc, rec = _run_in_process(tmp_path, "B", _ok_item, monkeypatch)
    assert rc == 10
    assert rec["environment"]["commit_unchanged"] is False
    assert rec["environment"]["stable"] is False


def test_worktree_change_during_run_is_judged_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-21: 開始時 clean・終了時 dirty なら stable false・exit 10。"""
    monkeypatch.setattr(mod, "_probe", lambda *a, **k: "0" * 40)
    _sequence(monkeypatch, "_worktree_clean", [True, False])
    rc, rec = _run_in_process(tmp_path, "B", _ok_item, monkeypatch)
    assert rc == 10
    assert rec["environment"]["worktree_clean_end"] is False
    assert rec["environment"]["worktree_clean_unchanged"] is False


def test_end_probe_failure_is_not_treated_as_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-21: 終了時だけ commit が取れなければ、同一と確認できず exit 10。"""
    _sequence(monkeypatch, "_probe", ["0" * 40, None])
    monkeypatch.setattr(mod, "_worktree_clean", lambda *a, **k: True)
    rc, rec = _run_in_process(tmp_path, "B", _ok_item, monkeypatch)
    assert rc == 10
    assert rec["environment"]["commit_end"] is None
    assert rec["environment"]["commit_unchanged"] is False


def test_interrupt_wins_over_environment_change_and_skips_end_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-21: 中断の印があれば終了時の採取をせず null のまま。優先は中断（exit 70）。"""

    def fake(ctx: Any, name: str, b_ok: bool) -> Any:
        mod._on_signal(signal.SIGTERM, None)
        return {"status": "ok"}, True

    rc, rec = _run_in_process(tmp_path, "B", fake, monkeypatch)
    env = rec["environment"]
    assert rc == 70
    assert json.loads(capsys.readouterr().out)["message"] == "interrupted"
    assert env["commit_end"] is None
    assert env["cli_end_sha256"] is None
    assert env["stable"] is None


def test_unverifiable_environment_is_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-21: 開始・終了とも採取できず stable が None なら、成功扱いにせず exit 10。"""
    monkeypatch.setattr(mod, "_probe", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_worktree_clean", lambda *a, **k: None)
    rc, rec = _run_in_process(tmp_path, "B", _ok_item, monkeypatch)
    assert rc == 10
    assert rec["environment"]["commit_unchanged"] is None
    assert rec["environment"]["stable"] is None
    assert json.loads(capsys.readouterr().out)["code"] == "judged_fail"


@pytest.mark.parametrize("override", [True, False])
def test_cli_origin_follows_bin_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: bool
) -> None:
    """REQ-33: cli_origin は差し替えなら env_override、そうでなければ built_by_script。"""
    ctx = _c_ctx(tmp_path)
    ctx.bin_override = override
    env = mod.collect_environment(ctx, None)
    assert env["cli_origin"] == ("env_override" if override else "built_by_script")


def test_record_md_notes_commit_does_not_represent_cli_origin() -> None:
    """REQ-33: bin_override の注意行に、commit が CLI の出所を表さない旨を出す。"""
    rec = {
        "schema": "real-machine-check/1",
        "evidence_hint": "test_harness",
        "bin_override": True,
        "environment": None,
        "inputs": None,
        "options": {},
        "items": {n: {"status": "not_run"} for n in mod.ITEM_ORDER},
    }
    assert "CLI の出所を表さない" in mod.render_markdown(rec)


@pytest.mark.parametrize(
    ("value", "want"),
    [(None, "build_default"), ("", "env"), ("/dummy/trainer-dir-xyz", "env")],
)
def test_trainer_origin_records_only_the_origin_not_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: Any, want: str
) -> None:
    """REQ-39: trainer の出所は閉じた語彙だけ。パスは記録に出ない（空文字も設定扱い）。"""
    ctx = _c_ctx(tmp_path)
    ctx.offline_env = {k: v for k, v in ctx.offline_env.items() if k != "FANDHE_EDGE_TRAINER_DIR"}
    if value is not None:
        ctx.offline_env["FANDHE_EDGE_TRAINER_DIR"] = value
    env = mod.collect_environment(ctx, None)
    assert env["trainer_origin"] == want
    assert "trainer-dir-xyz" not in json.dumps(mod.sanitize_record({"environment": env}))


@pytest.mark.parametrize(("items", "want"), [("A,B", False), ("B,D", True)])
def test_cargo_offline_reflects_whether_item_a_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, items: str, want: bool
) -> None:
    """REQ-38: A（make ci は offline を強制しない）を含む実行では cargo_offline が false。"""
    monkeypatch.setattr(mod, "run_item", lambda ctx, name, b_ok: ({"status": "ok"}, True))
    (tmp_path / "w").mkdir()
    args = _ns(
        repo_root=str(REPO),
        work_dir=str(tmp_path / "w"),
        bin=str(_fake_cli(tmp_path, "exit 0\n")),
        bin_override=True,
        items=items,
        quiet_machine=False,
        with_ci="A" in items,
    )
    saved = {s: signal.getsignal(s) for s in mod.INTERRUPT_SIGNALS}
    try:
        mod.run(args)
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
    rec = json.loads((tmp_path / "w" / "record.json").read_text())
    assert rec["options"]["cargo_offline"] is want


# --- #361: 記録へ入る文字列の扱い（REQ-21・REQ-33・REQ-39） ---


def test_cell_strips_control_and_bidi_chars() -> None:
    """REQ-39: record.md のセルから制御・bidi 制御・ゼロ幅文字を除く。"""
    assert mod._cell("a\u202eb\u2066c\u2069d\x00e\x1bf\u200bg") == "abcdefg"
    assert mod._cell("x\ny\u0085z\u2028w\r\nq\tv") == "x y z w  q v"
    assert mod._cell("<a&b>") == "&lt;a&amp;b&gt;"


def test_render_markdown_has_no_control_chars_or_extra_lines() -> None:
    """REQ-39: U+2028・bidi 制御・制御文字を含む値でも record.md に制御文字が残らない。"""
    evil = "v\u202ex\u2028y\x07z\u2066"
    items = {n: {"status": "not_run", "reason": "not_selected"} for n in mod.ITEM_ORDER}
    items["B"] = {"status": "failed", "reason": "invalid_json", "step": evil}
    rec = {
        "schema": mod.SCHEMA,
        "evidence_hint": evil,
        "options": {"x": evil},
        "environment": {"y": evil},
        "items": items,
    }
    md = mod.render_markdown(rec)
    bad = [c for c in md if c != "\n" and unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp")]
    assert bad == []


def test_infer_command_is_a_fixed_display_name(tmp_path: Path) -> None:
    """REQ-33: infer の command は固定語彙で、`<redacted>` にならず、パス文字を含まない。"""
    assert mod.INFER_COMMAND_DISPLAY == "infer --package <package-dir> --text <fixed-sample>"
    assert not any(c in mod.INFER_COMMAND_DISPLAY for c in mod.PATH_CHARS)
    rec = _representative_record(tmp_path)
    steps = rec["items"]["B"]["steps"]
    assert steps[6]["command"] == mod.INFER_COMMAND_DISPLAY
    out = mod.sanitize_record(rec)
    assert out["items"]["B"]["steps"][6]["command"] == mod.INFER_COMMAND_DISPLAY
    for st in steps:
        assert not any(c in st["command"] for c in mod.PATH_CHARS)


def test_error_type_name_is_a_closed_vocabulary() -> None:
    """REQ-21: error_type は組み込みの語彙だけ。語彙外・名乗りだけの自作クラスは `<unexpected>`。"""
    assert mod.error_type_name(ValueError()) == "ValueError"
    assert mod.error_type_name(StopIteration()) == "<unexpected>"
    assert mod.error_type_name(type("Evil\u202e", (Exception,), {})()) == "<unexpected>"
    assert mod.error_type_name(type("ValueError", (Exception,), {})()) == "<unexpected>"
    assert mod.sanitize_record({"error_type": "Evil\u202e"}) == {"error_type": "<unexpected>"}
    assert mod.sanitize_record({"error_type": "RuntimeError"}) == {"error_type": "RuntimeError"}


def test_split_lines_splits_only_on_lf(tmp_path: Path) -> None:
    """REQ-28: 行分割は LF（と行末 CR）だけ。U+2028・U+0085 では割らない。"""
    assert mod.split_lines("a\nb\r\nc") == ["a", "b", "c"]
    assert mod.split_lines("a\u2028b\u0085c\n") == ["a\u2028b\u0085c"]
    assert mod.split_lines("") == []
    f = tmp_path / "t.jsonl"
    f.write_text('{"a":"x\u2028y"}\n', encoding="utf-8")
    assert mod._count_lines(f) == 1
    assert mod.parse_otool_libraries("hdr\n  lib\u2028x.dylib (c)\n") == ["lib\u2028x.dylib"]


def test_script_has_no_standard_line_splitting() -> None:
    """REQ-28: 標準の行分割の再混入を止める。"""
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert ".splitlines(" not in src


def test_nonzero_exit_with_non_json_reports_unexpected_exit_code(tmp_path: Path) -> None:
    """REQ-21: 非 0 終了かつ非 JSON は invalid_json に加え、終了コードの不一致を明示する。"""
    cli = _fake_cli(tmp_path, "echo not-json\nexit 64\n")
    d = tmp_path / "B"
    assert mod.stage_inputs(_ctx(tmp_path, cli), d, None)
    _steps, _pkg, failure = mod.run_pipeline(_ctx(tmp_path, cli), d, "sample", {0})
    assert failure == {
        "status": "failed",
        "reason": "invalid_json",
        "step": "register",
        "exit_code": 64,
        "exit_code_unexpected": True,
    }


def test_run_cmd_gives_up_waiting_when_the_group_kill_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: KILL が失敗し回収が終わらなくても、上限で諦めて `unreaped` を返す。"""
    pidfile = tmp_path / "pid"
    real_wait = subprocess.Popen.wait
    seen: list[int] = []

    def fake_wait(self: Any, timeout: float | None = None) -> int:
        seen.append(self.pid)
        if timeout is None:
            raise AssertionError("wait without a timeout")
        time.sleep(timeout)
        raise subprocess.TimeoutExpired("x", timeout)

    monkeypatch.setattr(mod, "_kill_group", lambda pid: False)
    monkeypatch.setattr(mod, "REAP_WAIT_LIMIT_SECONDS", 0.5)
    monkeypatch.setattr(subprocess.Popen, "wait", fake_wait)
    mod._interrupt_requested = False
    started = time.monotonic()
    try:
        r = mod.run_cmd(
            ["/bin/sh", "-c", f'echo $$ > "{pidfile}"; exec sleep 60'],
            tmp_path,
            tmp_path / "o",
            tmp_path / "o.e",
            1,
            4096,
            4096,
        )
        elapsed = time.monotonic() - started
    finally:
        monkeypatch.setattr(subprocess.Popen, "wait", real_wait)
        for pid in seen:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                os.waitpid(pid, 0)
            except OSError:
                pass
    assert (r.exit_code, r.reason) == (None, "unreaped")
    assert mod._child_may_remain is True
    assert mod._active_pgid is None
    assert elapsed < 10.0


def test_run_cmd_leaves_child_may_remain_false_on_the_normal_path(tmp_path: Path) -> None:
    """REQ-39: 通常の回収では `child_may_remain` は立たない。"""
    r = _run(tmp_path, ["/bin/sh", "-c", "exit 0"])
    assert (r.exit_code, r.reason) == (0, None)
    assert mod._child_may_remain is False
    assert mod._active_pgid is None


def test_unreaped_child_is_recorded_as_a_failed_item_with_child_may_remain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-39: 回収を諦めたら record に `child_may_remain: true`、項目は failed（10）。"""

    def fake_item(ctx: Any, name: str, b_ok: bool) -> Any:
        mod._child_may_remain = True
        return mod.fail_item("unreaped"), b_ok

    rc, rec = _run_in_process(tmp_path, "B", fake_item, monkeypatch)
    assert rc == 10
    assert rec["child_may_remain"] is True
    assert rec["items"]["B"] == {"status": "failed", "reason": "unreaped"}
    assert (
        "子プロセスの回収が上限時間内に終わらなかった" in (tmp_path / "w" / "record.md").read_text()
    )
    assert "a child process may remain" in capsys.readouterr().err


def test_normal_record_has_child_may_remain_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-39: 既定の record では `child_may_remain` は false。"""
    rc, rec = _run_in_process(
        tmp_path, "B", lambda ctx, name, b_ok: (mod.fail_item("x"), b_ok), monkeypatch
    )
    assert rec["child_may_remain"] is False
    assert rc == 10


def test_second_signal_forces_exit_while_cleanup_is_stuck(tmp_path: Path) -> None:
    """REQ-39: 後始末が固まっても、シグナルの 2 回目で子を KILL し exit 70 で強制終了する。"""
    pidfile = tmp_path / "pids"
    make = _script(
        tmp_path,
        "fake-make",
        f'trap "" TERM\necho $$ >> "{pidfile}"\nsleep 60 &\necho $! >> "{pidfile}"\nwait\n',
    )
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['m'] = m\n"
        "spec.loader.exec_module(m)\n"
        "m._kill_group = lambda pid: True\n"
        "m.REAP_WAIT_LIMIT_SECONDS = 120\n"
        "sys.exit(m.main(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ, FANDHE_EDGE_MAKE_CMD=str(make))
    proc = subprocess.Popen(  # noqa: S603  テスト用に自リポジトリのスクリプトを引数リストで起動する
        [
            sys.executable,
            str(launcher),
            "run",
            "--repo-root",
            str(REPO),
            "--work-dir",
            str(work),
            "--bin",
            str(_fake_cli(tmp_path, "exit 0\n")),
            "--bin-override",
            "--items",
            "D",
            "--repeat",
            "1",
            "--p95-limit-us",
            "50000",
            "--package-limit-bytes",
            "1000",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert _wait_for(lambda: len(_pids(pidfile)) >= 2)
        pids = _pids(pidfile)
        proc.send_signal(signal.SIGTERM)
        # 後始末は固まっている（KILL を送らない模擬）ので、1 回目では終わらない
        time.sleep(1.0)
        assert proc.poll() is None
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for pid in _pids(pidfile):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
    assert proc.returncode == 70
    assert json.loads(out) == {
        "code": "runtime_error",
        "message": "interrupted (forced exit)",
    }
    assert not (work / "record.json").exists()
    assert _wait_for(lambda: not any(_alive(p) for p in pids), 10)


def test_run_cmd_timeout_kills_term_ignoring_grandchild(tmp_path: Path) -> None:
    """REQ-39: 期限超過（timeout）でも、TERM を無視する孫まで止める。"""
    pidfile = tmp_path / "pids"
    script = f'trap "" TERM; sleep 60 & echo $! > "{pidfile}"; wait'
    r = mod.run_cmd(
        ["/bin/sh", "-c", script], tmp_path, tmp_path / "o", tmp_path / "o.e", 1, 4096, 4096
    )
    assert (r.exit_code, r.reason) == (None, "timeout")
    pids = _pids(pidfile)
    assert len(pids) == 1
    assert _wait_for(lambda: not _alive(pids[0]), 10)
