"""実機での動作確認スクリプト（scripts/real_machine_check_record.py）の単体テスト。

REQ-33（記録・stdout に利用者の値を出さない）・REQ-28（バッチと単体の推論の全件一致の突き合わせ）・
REQ-39（子プロセスの上限時間・出力サイズ上限）・REQ-27（評価データを推論の確認に使わない）。
証拠種別: テストハーネス。実機での測定結果ではない（実機の確定は人が行う）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
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


def test_redact_value_replaces_path_strings_and_drops_data_keys() -> None:
    """REQ-33: `/` か `\\` を含む文字列は伏せ、input・text・id のキーは落とす。"""
    obj = {
        "step": "register",
        "message": "cannot open /home/user/secret.json",
        "win": "C:\\data\\x",
        "input": "private body",
        "text": "private body",
        "id": "row-1",
        "nested": {"path": "a/b", "n": 3, "ok": True},
        "items": ["x/y", "plain"],
    }
    assert mod.redact_value(obj) == {
        "step": "register",
        "message": "<redacted>",
        "win": "<redacted>",
        "nested": {"path": "<redacted>", "n": 3, "ok": True},
        "items": ["<redacted>", "plain"],
    }


def test_summarize_infer_keeps_only_status_label_and_score_key_count() -> None:
    """REQ-33: infer の要約は id（データの id）と scores の値を含まない。"""
    obj = {
        "id": "train-row-7",
        "status": "ok",
        "predicted_label": "beta",
        "scores": {"alpha": 0.1, "beta": 0.8, "gamma": 0.1},
    }
    assert mod.summarize_infer(obj) == {
        "status": "ok",
        "predicted_label": "beta",
        "scores_keys": 3,
    }


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


def test_failed_step_records_exit_code_and_redacted_message(tmp_path: Path) -> None:
    """REQ-33・REQ-21: 失敗した工程は exit_code・code・伏せた message を記録して止まる。"""
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
        "message": "<redacted>",
    }


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


def test_item_f_rejects_zero_tests_and_failed_prebuild(tmp_path: Path) -> None:
    """F: 0 件実行は pass に数えず no_tests に計上する。`--no-run` の失敗は build_failed。"""
    ctx = _ctx(tmp_path / "w", _fake_cli(tmp_path, "exit 0\n"))
    ctx.work.mkdir()
    ctx.repeat = 2
    zero = "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
    ctx.cargo_cmd = str(_script(tmp_path, "c0", f"echo '{zero}'\nexit 0\n"))
    res = mod.item_f(ctx)
    assert (res["passed"], res["failed"], res["no_tests"]) == (0, 2, 2)
    shutil.rmtree(ctx.work / "F")
    ctx.cargo_cmd = str(
        _script(tmp_path, "c1", 'case "$*" in *--no-run*) exit 101;; esac\nexit 0\n')
    )
    assert mod.item_f(ctx) == {"status": "failed", "reason": "build_failed", "exit_code": 101}
    shutil.rmtree(ctx.work / "F")
    good = CARGO_RUST_LINE
    ctx.cargo_cmd = str(_script(tmp_path, "c2", f"echo '{good}'\nexit 0\n"))
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


def test_redaction_is_case_insensitive_and_covers_lookalike_slashes() -> None:
    """REQ-33: `Input`・`TEXT` のキーも落とし、U+2215・U+2044・U+FF0F もパス文字として伏せる。"""
    obj = {
        "Input": "body",
        "TEXT": "body",
        "Id": "r1",
        "a": "x\u2215y",
        "b": "x\u2044y",
        "c": "x\uff0fy",
    }
    assert mod.redact_value(obj) == {"a": "<redacted>", "b": "<redacted>", "c": "<redacted>"}


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


def test_interrupt_stops_children_and_writes_record(tmp_path: Path) -> None:
    """REQ-39: SIGTERM で子グループを残さず、中断時点の record を書き exit 70 で終える。"""
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
    proc.send_signal(signal.SIGTERM)
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


@pytest.mark.parametrize(
    "extra",
    [
        ["--items", "Z"],
        ["--items", "B,B"],
        ["--repeat", "0"],
        ["--repeat", "1001"],
        ["--items", "A"],
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


def _run(tmp_path: Path, argv: list[str], name: str = "o") -> Any:
    return mod.run_cmd(argv, tmp_path, tmp_path / name, tmp_path / (name + ".e"), 20, 4096, 4096)


@pytest.mark.parametrize(("script", "want"), [("exit 0", 0), ("exit 20", 20), ("kill -9 $$", 137)])
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


def test_compare_infer_lists_rows_over_tolerance_in_mismatch_ids() -> None:
    """REQ-28: ラベルが一致しても、スコア差が許容差を超えた行の id は mismatch_ids に入る。"""
    batch = {
        "a": {"predicted_label": "x", "scores": {"x": 0.5}},
        "b": {"predicted_label": "x", "scores": {"x": 0.5}},
    }
    single = {
        "a": {"predicted_label": "x", "scores": {"x": 0.5 + 1e-12}},
        "b": {"predicted_label": "x", "scores": {"x": 0.5 + 1e-6}},
    }
    cmp = mod.compare_infer(batch, single)
    assert cmp["mismatch_ids"] == ["b"]
    assert (cmp["label_match"], cmp["label_mismatch"]) == (2, 0)


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
def row(rid, delta):
    return json.dumps({{"id": rid, "status": "ok", "predicted_label": ids[0],
                       "scores": scores(delta)}})
mode = {mode!r}
if "--input-file" in a:
    rows = [json.loads(line)["id"] for line in open(a[a.index("--input-file") + 1])]
    if mode == "short":
        rows = rows[:-1]
    if mode == "dup":
        rows[-1] = rows[0]
    for rid in rows:
        print(row(rid, 0.0))
else:
    print(row(a[a.index("--id") + 1], {delta!r}))
"""


def _infer_cli(tmp_path: Path, mode: str = "ok", delta: float = 0.0) -> Path:
    fake = tmp_path / "fake-infer"
    fake.write_text(FAKE_INFER.format(python=sys.executable, mode=mode, delta=delta), "utf-8")
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


def test_item_e_passes_when_batch_and_single_agree(tmp_path: Path) -> None:
    """REQ-28: 行数・id・スコアが一致する正常な出力は ok。"""
    res = mod.item_e(_e_ctx(tmp_path, _infer_cli(tmp_path)))
    assert res["status"] == "ok"
    assert res["label_mismatch"] == 0
    assert res["scores_exact_match"] == res["records"]


@pytest.mark.parametrize("mode", ["short", "dup"])
def test_item_e_rejects_batch_row_count_and_duplicate_ids(tmp_path: Path, mode: str) -> None:
    """REQ-28: バッチ出力の行数が違う・id が重複する場合は unexpected_output（後勝ちにしない）。"""
    res = mod.item_e(_e_ctx(tmp_path, _infer_cli(tmp_path, mode=mode)))
    assert (res["status"], res["reason"], res["step"]) == (
        "failed",
        "unexpected_output",
        "infer-batch",
    )


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
    assert mod._step_check(name, _valid_reports(f)[name], set(), 0, f, 0) is True


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
    assert mod._step_check(name, bad, set(), 0, f, 0) is False


def _scores(facts: Any, delta: float = 0.0) -> dict[str, float]:
    n = len(facts.option_ids)
    vals = [1.0 / n] * n
    vals[0] += delta
    return dict(zip(facts.option_ids, vals))  # noqa: B905  Python 3.9 互換


def test_check_infer_output_validates_label_keys_finiteness_and_sum() -> None:
    """REQ-21: infer の label は選択肢 ID、scores のキーは選択肢 ID と一致、有限で和が 1 近傍。"""
    f = _facts()
    ids = f.option_ids
    ok = {"id": "x", "status": "ok", "predicted_label": ids[0], "scores": _scores(f)}
    assert mod.check_infer_output(ok, f) is True
    assert mod._step_check("infer", ok, set(), 0, f, None) is True
    assert mod.check_infer_output(dict(ok, predicted_label="nope"), f) is False
    assert mod.check_infer_output(dict(ok, scores={ids[0]: 1.0}), f) is False
    assert mod.check_infer_output(dict(ok, scores=dict(_scores(f), extra=0.0)), f) is False
    assert mod.check_infer_output(dict(ok, scores=_scores(f, 1e-3)), f) is False
    nan = dict(_scores(f), **{ids[0]: float("nan")})
    assert mod.check_infer_output(dict(ok, scores=nan), f) is False
    # 許容差（1e-6）以内のずれは合格（CLI の runtime が許す範囲）
    assert mod.check_infer_output(dict(ok, scores=_scores(f, 5e-7)), f) is True


def _cap_json(total: int = 3, limit: int = 40000000, exceeded: bool = False) -> dict[str, Any]:
    comp = {n: {"bytes": 1, "file_count": 1} for n in mod.CAPACITY_COMPONENTS}
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
    assert chk(_pkg(f, capacity=_cap_json(total=3, limit=5, exceeded=True))) is False
    assert chk(_pkg(f, capacity=None)) is False
    # exit 20 で容量超過（整合）は許容される
    over = {"step": "package", "capacity": _cap_json(9, 5, True), "infer_p95": None}
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
comps = {{k: {{"bytes": 1, "file_count": 1}} for k in
         ["weights", "vocab_or_feature_transform", "label_table", "calibration", "metadata"]}}
def out(o, rc=0):
    print(json.dumps(o))
    sys.exit(rc)
if cmd == "register":
    out({{"step": "register", "status": "ok", "options": len(d["options"]),
         "evaluation_defined": True}})
if cmd == "inspect":
    out({{"step": "inspect", "status": "ok", "valid_records": n_train,
         "split": {{"train": n_train - 2, "validation": 1, "test": 1}}}})
if cmd in ("train", "select"):
    out({{"step": cmd, "status": "ok", "candidate": 0, "kind": "c1"}})
if cmd == "evaluate":
    out({{"step": "evaluate", "status": "ok", "candidate": 0, "kind": "c1",
         "n_total": n_eval, "correct": n_eval}})
if cmd == "package":
    if c2:
        total = cfg["c2_total"]
        limit = cfg["c2_limit"] or limits["max_package_bytes"]
        out({{"code": cfg["c2_code"], "message": "resource limit exceeded", "step": "package",
             "capacity": {{"total_bytes": total, "limit_bytes": limit,
                          "exceeded": cfg["c2_exceeded"], "components": comps}},
             "infer_p95": None}}, cfg["c2_rc"])
    rc = cfg["rc1"]
    p95 = {{"p95_us": cfg["p95_us"], "limit_us": limits["max_infer_p95_us"],
           "exceeded": cfg["p95_exceeded"]}}
    cap = {{"total_bytes": 3, "limit_bytes": 40000000, "exceeded": False, "components": comps}}
    if cfg["publish"] is None:
        publish = rc == 0
    else:
        publish = cfg["publish"]
    if publish:
        os.makedirs("project/package", exist_ok=True)
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
