"""PoC-26 の測定と記録（`--warmup`・`measure.py`）の検査（REQ-41・TASK-41.1-8・#393）。

合成モデルを CPU で動かすテストハーネス。実重みでの実測は人が行う（手順書参照）。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.poc26 import measure
from tools.poc26.lora_poc import main
from tools.poc26.score import p95

from test_poc26_lora_poc import _predict_argv, _sha, make_data, make_dir, pin_values, train_args

TIME_L_SAMPLE = """\
        1.23 real         1.00 user         0.10 sys
           123456789  maximum resident set size
                   0  average shared memory size
           100000000  peak memory footprint
"""


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("measure")
    model, data = make_dir(tmp), make_data(tmp)
    out = tmp / "trained"
    assert main(train_args(model, data, out, seed=0)) == 0
    out_bf16 = tmp / "trained_bf16"  # measure は bf16 条件で動かすため bf16 学習のアダプタを使う
    assert main(train_args(model, data, out_bf16, seed=0, extra=["--dtype", "bf16"])) == 0
    return {"tmp": tmp, "model": model, "data": data, "outs": [out], "bf16": out_bf16}


def test_warmup_excludes_leading_records_from_stats(env: dict, tmp_path: Path) -> None:
    """REQ-41: --warmup 1 は 4 件中先頭 1 件を統計から除く。既定 0 は従来どおり 4 件。"""
    adapter = env["outs"][0]
    for extra, count, excluded in (([], 4, 0), (["--warmup", "1"], 3, 1)):
        out = tmp_path / f"w{excluded}"
        assert main(_predict_argv(env, adapter, out, *extra)) == 0
        run = json.loads((out / "run.json").read_text())
        assert run["scoring"]["count"] == count
        assert run["scoring"]["warmup_excluded"] == excluded
        assert len((out / "pred.jsonl").read_text().splitlines()) == 4  # 予測は全件出る
        assert run["mlx_peak_memory_bytes"] > 0
    # 件数以上の warmup は統計が空になるため 64
    assert main(_predict_argv(env, adapter, tmp_path / "wbad", "--warmup", "4")) == 64


def test_p95_nearest_rank() -> None:
    """既存の p95 は nearest-rank（20 件なら 19 番目）。warmup 除外後も同じ関数を使う。"""
    assert p95([float(i) for i in range(1, 21)]) == 19.0


def test_parse_time_l() -> None:
    """time -l の maximum resident set size（byte）と peak memory footprint を取り出す。"""
    assert measure.parse_time_l(TIME_L_SAMPLE) == {
        "max_rss_bytes": 123456789,
        "peak_memory_footprint_bytes": 100000000,
    }
    assert measure.parse_time_l("nothing") == {
        "max_rss_bytes": None,
        "peak_memory_footprint_bytes": None,
    }


def test_capacity_sums_decimal_mb_and_excludes_onnx(tmp_path: Path) -> None:
    """容量は 10 進 MB の合計。40MB 超過は警告フラグ。ONNX は参考で合計に含めない。"""
    m, a, o = tmp_path / "m", tmp_path / "a", tmp_path / "o"
    for d in (m, a, o):
        d.mkdir()
    for n in measure.BASE_FILES:
        (m / n).write_bytes(b"x" * 10_000_000)
    for n in measure.ADAPTER_FILES:
        (a / n).write_bytes(b"y" * 1_000_000)
    for n in measure.ONNX_FILES:
        (o / n).write_bytes(b"z" * 5_000_000)
    cap = measure.capacity(m, a, o)
    assert cap["total_bytes"] == 42_000_000
    assert cap["total_mb"] == 42.0
    assert cap["diff_mb"] == 2.0
    assert cap["exceeds_target"] is True
    assert cap["onnx_reference"]["total_bytes"] == 10_000_000
    assert cap["onnx_reference"]["included_in_total"] is False
    (m / "config.json").unlink()
    (m / "config.json").symlink_to(m / "tokenizer.json")
    with pytest.raises(measure.MeasureError):
        measure.capacity(m, a, None)


def _run_json(p95_seconds: float) -> dict:
    return {
        "device": "gpu",
        "dtype": "bf16",
        "input_data_sha256": "0" * 64,
        "records": 10,
        "scoring": {
            "count": 5,
            "warmup_excluded": 5,
            "mean_seconds": 0.1,
            "p95_seconds": p95_seconds,
            "max_seconds": 0.3,
            "choices_per_prompt": 26,
            "forward_chunk": 8,
        },
        "max_rss_bytes": 1,
        "mlx_peak_memory_bytes": 2,
        "elapsed_seconds": 3.0,
    }


def test_classify_and_thresholds() -> None:
    """申告の無い実機測定は参考扱い。p95 は 250ms 未満、RSS は 2GiB 未満で合否が決まる。"""
    assert measure.classify("real_machine", True) == "real_machine"
    assert measure.classify("real_machine", False) == "reference_only"
    assert measure.classify("test_harness", True) == "test_harness"
    gib2 = 2 * 1024**3
    ng = measure.summarize(
        _run_json(0.25), {"max_rss_bytes": gib2, "peak_memory_footprint_bytes": None}, "x", 1.5
    )
    assert (ng["p95_ms"], ng["p95_ok"], ng["rss_ok"]) == (250.0, False, False)  # 境界は未満
    assert ng["forward_calls_per_prompt"] == 4
    ok = measure.summarize(
        _run_json(0.2499), {"max_rss_bytes": gib2 - 1, "peak_memory_footprint_bytes": None}, "x", 1
    )
    assert (ok["p95_ok"], ok["rss_ok"]) == (True, True)


def _argv(env: dict, out: Path, *extra: str, rel_to: Path | None = None) -> list[str]:
    pv = pin_values(env["model"])
    adapter = env["bf16"]

    def p(x: Path) -> str:
        return os.path.relpath(x, rel_to) if rel_to else str(x)

    return [
        *["--model-dir", p(env["model"]), "--definition", p(env["data"]["definition"])],
        *["--adapter-dir", p(adapter), "--input", p(env["data"]["validation"])],
        *["--adapter-sha256", _sha(adapter / "adapters.safetensors")],
        *["--model-sha256", pv["model"], "--config-sha256", pv["config"]],
        *["--tokenizer-sha256", pv["tokenizer"]],
        *["--tokenizer-config-sha256", pv["tokenizer-config"]],
        *["--out-dir", p(out), "--evidence", "test_harness", "--warmup", "1"],
        *["--max-seq-length", "1024", *extra],
    ]


# time -l の代役: RSS 行を stderr に出して子を exec する。gpu 条件だけ exit 3 にする
FAKE_TIME = [
    "/bin/sh",
    "-c",
    'case "$*" in *"--device gpu"*) exit 3;; esac; '
    'echo "  5000  maximum resident set size" >&2; exec "$@"',
    "x",
]


def test_failed_condition_is_recorded_and_successes_kept(
    env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-2・P2-1: 失敗した条件は error で記録して続行し、成功済みを残す。相対パスも動く。"""
    monkeypatch.chdir(tmp_path)
    argv = _argv(env, tmp_path / "rec", rel_to=tmp_path)
    argv[argv.index("--out-dir") + 1] = "rec"
    assert measure.main(argv, time_cmd=FAKE_TIME) == 70
    conds = json.loads((tmp_path / "rec" / "record.json").read_text())["conditions"]
    assert conds["gpu_bf16"] == {"status": "error", "code": "predict exited with code 3"}
    assert (conds["cpu_bf16"]["status"], conds["cpu_bf16"]["rss_time_l_bytes"]) == ("ok", 5000)
    assert conds["cpu_bf16"]["measured"] == 3
    assert "error: predict exited with code 3" in (tmp_path / "rec" / "record.md").read_text()


def test_missing_rss_is_an_error_condition(env: dict, tmp_path: Path) -> None:
    """P2-3: time -l の RSS が取れない条件は error（成功扱いにしない）。"""
    out = tmp_path / "rec"
    argv = _argv(env, out, "--conditions", "cpu_bf16")
    assert measure.main(argv, time_cmd=["/usr/bin/env"]) == 70
    cond = json.loads((out / "record.json").read_text())["conditions"]["cpu_bf16"]
    assert cond == {
        "status": "error",
        "code": "time -l output has no maximum resident set size",
    }


def test_zero_forward_chunk_is_rejected() -> None:
    """P2-4: forward_chunk が 0 なら割り算の前に MeasureError。"""
    run = _run_json(0.1)
    run["scoring"]["forward_chunk"] = 0
    with pytest.raises(measure.MeasureError):
        measure.summarize(run, {"max_rss_bytes": 1, "peak_memory_footprint_bytes": None}, "x", 0.0)


def test_run_child_kills_group_on_timeout_and_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """P2-2: 上限超過・中断（KeyboardInterrupt）の両方で子を止めてから例外を出す。"""
    with pytest.raises(measure.MeasureError, match="timed out"):
        measure.run_child(["/bin/sleep", "30"], 1)
    pids: list[int] = []

    class Interrupted(subprocess.Popen):
        def wait(self, timeout=None):  # type: ignore[override]
            if timeout is not None:
                pids.append(self.pid)
                raise KeyboardInterrupt
            return super().wait()

    monkeypatch.setattr(measure.subprocess, "Popen", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        measure.run_child(["/bin/sleep", "30"], 60)
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_warmup_is_checked_before_model_load(
    env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2-5: 件数以上の --warmup はベースモデルを読む前に 64。"""
    from tools.poc26 import predict as predict_mod

    def boom(*_a, **_k):
        raise AssertionError("model must not be loaded")

    monkeypatch.setattr(predict_mod, "load_base_model", boom)
    assert main(_predict_argv(env, env["outs"][0], tmp_path / "w", "--warmup", "4")) == 64


@pytest.mark.skipif(sys.platform != "darwin", reason="/usr/bin/time -l は macOS（BSD time）のみ")
def test_measure_end_to_end_on_synthetic_model(env: dict, tmp_path: Path) -> None:
    """合成モデルで 1 条件を実行し、record に本文・パスが出ず 0700/0600 で書かれる。"""
    out = tmp_path / "rec"
    argv = _argv(env, out, "--conditions", "cpu_bf16")
    assert measure.main(argv) == 0
    rec = json.loads((out / "record.json").read_text())
    c = rec["conditions"]["cpu_bf16"]
    assert (c["measured"], c["warmup_excluded"], c["classification"]) == (3, 1, "test_harness")
    assert c["rss_time_l_bytes"] > 0
    assert out.stat().st_mode & 0o777 == 0o700
    assert (out / "record.json").stat().st_mode & 0o777 == 0o600
    text = (out / "record.json").read_text() + (out / "record.md").read_text()
    assert "SECRETBODY" not in text
    assert str(tmp_path) not in text
    assert measure.main(argv) == 70  # 既存の out-dir は上書きしない


def test_argument_errors_map_to_64_without_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """P1-1: 引数エラーは SystemExit(2)＋usage でなく 64 と JSON（値は出さない）。"""
    assert measure.main(["--bogus", "SECRETVALUE"]) == 64
    err = json.loads(capsys.readouterr().err)
    assert err == {
        "code": "invalid_input",
        "message": "invalid arguments: the following arguments are required: "
        "--model-dir, --model-sha256, --config-sha256, --tokenizer-sha256, "
        "--tokenizer-config-sha256, --adapter-sha256, --definition, --adapter-dir, "
        "--input, --out-dir, --evidence",
    }
    assert measure.main(["--model-sha256", "SECRETVALUE"]) == 64
    assert "SECRETVALUE" not in capsys.readouterr().err


def test_duplicate_conditions_rejected_before_any_child(env: dict, tmp_path: Path) -> None:
    """P1-2: 条件の重複は子の起動前に 64。out-dir も作らない。"""
    out = tmp_path / "rec"
    argv = _argv(env, out, "--conditions", "cpu_bf16", "cpu_bf16")
    assert measure.main(argv, time_cmd=["/nonexistent"]) == 64
    assert not out.exists()


def test_record_write_failure_is_runtime_error(
    env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """P1-3: record の書き込み失敗は traceback でなく 70 とパスなしの JSON。"""

    def fail(*_a, **_k):
        raise OSError("/secret/path")

    monkeypatch.setattr(measure, "_write_new", fail)
    argv = _argv(env, tmp_path / "rec", "--conditions", "cpu_bf16")
    assert measure.main(argv, time_cmd=FAKE_TIME) == 70
    assert json.loads(capsys.readouterr().err) == {"code": "runtime_error", "message": "OSError"}


def test_killpg_lookup_error_keeps_original_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bugbot: 子が既に終了して killpg が ProcessLookupError でも元の超過を MeasureError で返す。"""
    real = os.killpg

    def gone(pid: int, sig: int) -> None:
        real(pid, sig)
        raise ProcessLookupError

    monkeypatch.setattr(measure.os, "killpg", gone)
    with pytest.raises(measure.MeasureError, match="timed out"):
        measure.run_child(["/bin/sleep", "30"], 1)


def test_keyboard_interrupt_stops_remaining_conditions_but_keeps_record(
    env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bugbot: 中断は残りの条件を実行せず、成功済みの記録を残して KeyboardInterrupt を再送出。"""
    real = measure.run_child
    calls: list[int] = []

    def flaky(cmd: list[str], timeout: int) -> str:
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(cmd, timeout)

    monkeypatch.setattr(measure, "run_child", flaky)
    out = tmp_path / "rec"
    argv = _argv(env, out, "--conditions", "cpu_bf16", "gpu_bf16")
    with pytest.raises(KeyboardInterrupt):
        measure.main(argv, time_cmd=FAKE_TIME)
    assert len(calls) == 2
    conds = json.loads((out / "record.json").read_text())["conditions"]
    assert list(conds) == ["cpu_bf16"]
    assert conds["cpu_bf16"]["status"] == "ok"


def test_sigterm_kills_child_group_and_restores_handlers() -> None:
    """codex P1: SIGTERM で子グループを止めて _Terminated を出し、ハンドラを元に戻す。"""
    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP))
    pids: list[int] = []
    real = subprocess.Popen

    class Spy(real):  # type: ignore[valid-type, misc]
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            pids.append(self.pid)

    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGHUP))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(measure.subprocess, "Popen", Spy)

        def run() -> None:
            with measure.terminate_as_exception():
                timer.start()
                measure.run_child(["/bin/sleep", "30"], 60)

        with pytest.raises(measure._Terminated) as ei:
            run()
    assert ei.value.signum == signal.SIGHUP
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)) == before


def test_sigterm_stops_remaining_conditions_and_keeps_record(
    env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """codex P1: 2 条件目の最中の SIGTERM は 143 で終え、成功済み（cpu_bf16）の記録を残す。"""
    real = measure.run_child
    calls: list[int] = []

    def term(cmd: list[str], timeout: int) -> str:
        calls.append(1)
        if len(calls) == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return real(cmd, timeout)

    monkeypatch.setattr(measure, "run_child", term)
    out = tmp_path / "rec"
    argv = _argv(env, out, "--conditions", "cpu_bf16", "gpu_bf16")
    assert measure.main(argv, time_cmd=FAKE_TIME) == 143
    assert len(calls) == 2
    conds = json.loads((out / "record.json").read_text())["conditions"]
    assert list(conds) == ["cpu_bf16"]
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


@pytest.mark.parametrize("value", ["0", "-1", "86401", "99999999999999999999"])
def test_timeout_seconds_is_range_checked_before_anything_runs(
    env: dict, tmp_path: Path, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """codex P1: --timeout-seconds は 1..86400 のみ。範囲外は 64 で、out-dir も作らない。"""
    out = tmp_path / "rec"
    argv = _argv(env, out, "--timeout-seconds", value)
    assert measure.main(argv, time_cmd=["/nonexistent"]) == 64
    assert json.loads(capsys.readouterr().err) == {
        "code": "invalid_input",
        "message": "invalid arguments: --timeout-seconds out of range: 1..86400",
    }
    assert not out.exists()
