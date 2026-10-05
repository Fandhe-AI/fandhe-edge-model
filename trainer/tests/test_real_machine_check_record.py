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


def _pipeline_cli(tmp_path: Path, package_line: str, package_rc: int) -> Path:
    body = f"""case "$1" in
register) echo '{{"step":"register","status":"ok"}}';;
inspect) echo '{{"step":"inspect","status":"ok"}}';;
train) echo '{{"step":"train","status":"ok"}}';;
select) echo '{{"step":"select","status":"ok","candidate":0}}';;
evaluate) echo '{{"step":"evaluate","status":"ok"}}';;
package) echo '{package_line}'; exit {package_rc};;
esac
"""
    return _fake_cli(tmp_path, body)


COMPONENTS = (
    '{"weights":{"bytes":1,"file_count":1},"vocab_or_feature_transform":{"bytes":0,"file_count":0},'
    '"label_table":{"bytes":1,"file_count":1},"calibration":{"bytes":0,"file_count":0},'
    '"metadata":{"bytes":1,"file_count":1}}'
)


def test_item_c_checks_p95_consistency_with_exit_code_and_limit(tmp_path: Path) -> None:
    """C-1: exit 0 なのに exceeded が true、または limit_us が指定と違えば unexpected_output。"""
    cap = (
        f'"capacity":{{"total_bytes":3,"limit_bytes":9,"exceeded":false,"components":{COMPONENTS}}}'
    )
    line = (
        '{"step":"package","status":"ok",'
        + cap
        + ',"infer_p95":{"p95_us":5,"limit_us":50000,"exceeded":true}}'
    )
    ctx = _ctx(tmp_path / "w", _pipeline_cli(tmp_path, line, 0))
    ctx.work.mkdir()
    res = mod.item_c(ctx)
    assert (res["status"], res["reason"], res["case"]) == ("failed", "unexpected_output", "C-1")
    shutil.rmtree(ctx.work / "C1")
    line = line.replace('"limit_us":50000,"exceeded":true', '"limit_us":7,"exceeded":false')
    ctx.bin = _pipeline_cli(tmp_path, line, 0)
    assert mod.item_c(ctx)["reason"] == "unexpected_output"


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
