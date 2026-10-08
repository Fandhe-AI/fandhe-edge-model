"""PoC-26 候補 P の容量・RSS・推論のみ p95 の測定と記録（REQ-41・TASK-41.1-8・#393）。

役割: 2 条件（gpu/bf16 を主、cpu/bf16 を併記）それぞれで `predict --warmup N` を
`/usr/bin/time -l` 下の子プロセスで実行し、run.json と time -l の出力から容量の内訳・RSS・p95 を
集約して `record.json`・`record.md` を書く。測定そのものは人が実機で実行する（手順は
`docs/design/poc26-measure-procedure.md`）。使い捨ての PoC 用で、製品（CLI・推論ランタイム）には
入らない（REQ-32）。通信しない（REQ-38）。評価ロジックは持たない（REQ-24〜27）。

記録の規則: 件数・ハッシュ・サイズ・秒・固定語彙だけを書く。パス・データ本文・id・子の stderr は
書かない（security.md。`real_machine_check_record.py` と同じ考え方）。出力先は新規の 0700
ディレクトリ、各ファイルは排他作成の 0600。子には上限時間を設け、超過時はプロセスグループごと止める
（REQ-39）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tools.poc26.common import MAX_WALL_SECONDS_CAP, SHA_RE

MB = 1_000_000  # 容量は 10 進 MB（オーナー確定 2026-10-08）
CAPACITY_TARGET_MB = 40
P95_LIMIT_MS = 250.0
RSS_LIMIT_BYTES = 2 * 1024**3  # 2 GiB（ガード層の RSS 上限と同じ。REQ-39）
TIME_CMD = ["/usr/bin/time", "-l"]
TRAINER_DIR = Path(__file__).resolve().parents[2]
STDERR_TAIL = 64 * 1024  # time -l の出力は末尾にある
CONDITIONS = {"gpu_bf16": ("gpu", "bf16"), "cpu_bf16": ("cpu", "bf16")}
# 容量に数える MLX 推論の読み込みファイル（固定名。ONNX は参考で合計に含めない）
BASE_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
ADAPTER_FILES = ("adapters.safetensors", "adapter_config.json")
ONNX_FILES = ("model.onnx", "model.onnx.data")
# Jev の公表値（参考。指標が違う: E2E と推論のみ。事前登録の参考値）
JEV_REFERENCE = {"e2e_ms_range": [70, 500], "rss_mb": 571.7}
_RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.M)
_FOOT_RE = re.compile(r"^\s*(\d+)\s+peak memory footprint\s*$", re.M)


class MeasureError(Exception):
    """測定の失敗（終了コード 70。メッセージにパス・本文は入れない）。"""


def parse_time_l(text: str) -> dict[str, int | None]:
    """`time -l` の出力から maximum resident set size（byte）と peak memory footprint を得る。"""
    rss = _RSS_RE.findall(text)
    foot = _FOOT_RE.findall(text)
    return {
        "max_rss_bytes": int(rss[-1]) if rss else None,
        "peak_memory_footprint_bytes": int(foot[-1]) if foot else None,
    }


def file_sizes(directory: Path, names: tuple[str, ...]) -> dict[str, int]:
    """通常ファイルだけのサイズ（symlink・非通常ファイルは拒否）。"""
    out = {}
    for n in names:
        st = os.lstat(directory / n)
        if not stat.S_ISREG(st.st_mode):
            raise MeasureError(f"{n} is not a regular file")
        out[n] = st.st_size
    return out


def capacity(model_dir: Path, adapter_dir: Path, onnx_dir: Path | None) -> dict[str, Any]:
    """MLX 推論が読むファイル一式の非圧縮合計（10 進 MB）と 40MB 目安との差。ONNX は参考で別掲。"""
    files = {**file_sizes(model_dir, BASE_FILES), **file_sizes(adapter_dir, ADAPTER_FILES)}
    total = sum(files.values())
    cap: dict[str, Any] = {
        "files_bytes": files,
        "total_bytes": total,
        "total_mb": round(total / MB, 3),
        "target_mb": CAPACITY_TARGET_MB,
        "diff_mb": round(total / MB - CAPACITY_TARGET_MB, 3),
        "exceeds_target": total > CAPACITY_TARGET_MB * MB,
        "onnx_reference": None,
    }
    if onnx_dir is not None:
        o = file_sizes(onnx_dir, ONNX_FILES)
        cap["onnx_reference"] = {
            "files_bytes": o,
            "total_bytes": sum(o.values()),
            "total_mb": round(sum(o.values()) / MB, 3),
            "included_in_total": False,
        }
    return cap


def classify(evidence: str, quiet_machine: bool) -> str:
    """証拠区分。静かな Mac の申告が無い実機測定は参考扱い（real-machine-check と同じ運用）。"""
    if evidence == "test_harness":
        return "test_harness"
    return "real_machine" if quiet_machine else "reference_only"


def summarize(
    run: dict[str, Any], time_l: dict[str, int | None], classification: str, load_avg: float
) -> dict[str, Any]:
    """1 条件分の run.json と time -l から記録用の数値だけを集約し、基準と比較する。"""
    sc = run["scoring"]
    if sc["forward_chunk"] < 1:
        raise MeasureError("forward_chunk must be positive")
    if time_l["max_rss_bytes"] is None:
        raise MeasureError("time -l output has no maximum resident set size")
    p95_ms = sc["p95_seconds"] * 1000.0
    rss = time_l["max_rss_bytes"]
    k, chunk = sc["choices_per_prompt"], sc["forward_chunk"]
    return {
        "device": run["device"],
        "dtype": run["dtype"],
        "classification": classification,
        "load_avg_1m_before": round(load_avg, 2),
        "input_data_sha256": run["input_data_sha256"],
        "records": run["records"],
        "warmup_excluded": sc["warmup_excluded"],
        "measured": sc["count"],
        "p95_ms": round(p95_ms, 3),
        "mean_ms": round(sc["mean_seconds"] * 1000.0, 3),
        "max_ms": round(sc["max_seconds"] * 1000.0, 3),
        "p95_limit_ms": P95_LIMIT_MS,
        "p95_ok": p95_ms < P95_LIMIT_MS,
        "choices_per_prompt": k,
        "forward_chunk": chunk,
        "forward_calls_per_prompt": math.ceil(k / chunk),
        "rss_time_l_bytes": rss,
        "rss_limit_bytes": RSS_LIMIT_BYTES,
        "rss_ok": rss < RSS_LIMIT_BYTES,
        "peak_memory_footprint_bytes": time_l["peak_memory_footprint_bytes"],
        "internal_max_rss_bytes": run["max_rss_bytes"],  # ru_maxrss と MLX ピークの大きい方
        "mlx_peak_memory_bytes": run["mlx_peak_memory_bytes"],
        "elapsed_seconds": round(run["elapsed_seconds"], 3),
    }


def run_child(cmd: list[str], timeout: int) -> str:
    """子を新しいセッションで起動し、超過・中断（Ctrl-C 等）ではグループごと止める。

    stderr は一時ファイルへ逃がし、末尾 `STDERR_TAIL` だけを読む（メモリに全量を溜めない）。
    """
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(  # noqa: S603 (引数リスト・shell なし)
            cmd,
            cwd=TRAINER_DIR,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=errf,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout)
        except BaseException as exc:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # 子が既に終了。元の例外（中断・超過）を置き換えない
            proc.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                raise MeasureError("predict timed out") from None
            raise
        if proc.returncode != 0:
            raise MeasureError(f"predict exited with code {proc.returncode}")
        size = errf.seek(0, os.SEEK_END)
        errf.seek(max(size - STDERR_TAIL, 0))
        return errf.read().decode("utf-8", "replace")


def render_md(rec: dict[str, Any]) -> str:
    """record.md（数値と固定語彙だけ）。"""
    cap = rec["capacity"]
    L = [
        "# PoC-26 候補 P の容量・RSS・p95 測定記録（REQ-41・TASK-41.1-8・#393）",
        "",
        f"- 容量: {cap['total_mb']} MB（目安 {cap['target_mb']} MB との差 {cap['diff_mb']:+} MB）"
        + (" 警告: 目安超過" if cap["exceeds_target"] else ""),
    ]
    if cap["onnx_reference"]:
        L.append(f"- 参考（合計に含めない）ONNX: {cap['onnx_reference']['total_mb']} MB")
    L += [
        "",
        "| 条件 | 証拠 | p95 ms (<250) | mean ms | 測定/除外 | RSS time -l MB (<2GiB) "
        "| MLX peak MB | 内部 RSS MB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    ok_conds = {n: c for n, c in rec["conditions"].items() if c["status"] == "ok"}
    for name, c in rec["conditions"].items():
        if c["status"] != "ok":
            L.append(f"| {name} | error: {c['code']} | - | - | - | - | - | - |")
            continue
        rss = c["rss_time_l_bytes"]
        L.append(
            f"| {name} | {c['classification']} | {c['p95_ms']} ({'ok' if c['p95_ok'] else 'NG'}) "
            f"| {c['mean_ms']} | {c['measured']}/{c['warmup_excluded']} "
            f"| {None if rss is None else round(rss / MB, 1)} ({'ok' if c['rss_ok'] else 'NG'}) "
            f"| {round(c['mlx_peak_memory_bytes'] / MB, 1)} "
            f"| {round(c['internal_max_rss_bytes'] / MB, 1)} |"
        )
    L.append("")
    for c0 in list(ok_conds.values())[:1]:
        L.append(
            f"- 1 件あたり採点: K={c0['choices_per_prompt']}・forward chunk {c0['forward_chunk']}"
            f"（forward {c0['forward_calls_per_prompt']} 回）"
        )
    L += [
        "- RSS はモデル読み込み〜採点の実行全体のピーク（推論のみは読み込みと分離できない）",
        f"- Jev 参考値（E2E {JEV_REFERENCE['e2e_ms_range']} ms・RSS {JEV_REFERENCE['rss_mb']} MB）"
        "は指標が違う（E2E と推論のみ）",
        "",
    ]
    return "\n".join(L)


def _write_new(directory: Path, name: str, text: str) -> None:
    fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _sha_arg(v: str) -> str:
    if not SHA_RE.fullmatch(v):
        raise argparse.ArgumentTypeError("must be 64 lowercase hex characters")
    return v


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        """引数エラーは引数名だけを JSON に出し（値は出さない）、終了コード 64 にする。

        `tools/poc26/cli.py` の `_Parser` と同じ作法（SystemExit(2)＋usage にしない）。
        """
        m = re.match(r"(argument [^:]+|the following arguments are required: .*)", message)
        raise _ArgError(f"invalid arguments: {m.group(1) if m else 'see usage'}")


class _Terminated(BaseException):
    """SIGTERM・SIGHUP の受信（KeyboardInterrupt と同じく子を止め、成功済みの記録を残す）。"""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


@contextmanager
def terminate_as_exception():
    """測定中だけ SIGTERM・SIGHUP を `_Terminated` に変え、抜けるときハンドラを元に戻す。"""

    def handler(signum: int, _frame: Any) -> None:
        raise _Terminated(signum)

    sigs = (signal.SIGTERM, signal.SIGHUP)
    prev = [signal.signal(sg, handler) for sg in sigs]
    try:
        yield
    finally:
        for sg, h in zip(sigs, prev, strict=True):
            signal.signal(sg, h)


class _ArgError(Exception):
    """引数不正（終了コード 64）。"""


def main(argv: list[str] | None = None, time_cmd: list[str] | None = None) -> int:
    """全条件を直列に測定し、record.json・record.md を書く。失敗は 70、引数不正は 64。"""
    ap = _Parser(prog="measure", allow_abbrev=False)
    ap.add_argument("--model-dir", type=Path, required=True)
    for f in ("model", "config", "tokenizer", "tokenizer-config", "adapter"):
        ap.add_argument(f"--{f}-sha256", type=_sha_arg, required=True)
    ap.add_argument("--definition", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--onnx-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--evidence", choices=["real_machine", "test_harness"], required=True)
    ap.add_argument("--quiet-machine", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--timeout-seconds", type=int, default=5400)
    ap.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    try:
        a = ap.parse_args(argv)
        if len(set(a.conditions)) != len(a.conditions):
            raise _ArgError("invalid arguments: --conditions must not repeat a condition")
        if not 1 <= a.timeout_seconds <= MAX_WALL_SECONDS_CAP:
            raise _ArgError(
                f"invalid arguments: --timeout-seconds out of range: 1..{MAX_WALL_SECONDS_CAP}"
            )
    except _ArgError as exc:
        print(json.dumps({"code": "invalid_input", "message": str(exc)}), file=sys.stderr)
        return 64
    # 相対パスは子（cwd が trainer/）でも同じ場所を指すよう絶対化する
    for f in ("model_dir", "definition", "adapter_dir", "input", "onnx_dir"):
        v = getattr(a, f)
        setattr(a, f, None if v is None else v.resolve())
    a.out_dir = a.out_dir.parent.resolve() / a.out_dir.name
    conds: dict[str, Any] = {}
    failed = False
    try:
        if os.path.lexists(a.out_dir) or not a.out_dir.parent.is_dir():
            raise MeasureError("out-dir must be a new directory under an existing parent")
        cap = capacity(a.model_dir, a.adapter_dir, a.onnx_dir)
        os.mkdir(a.out_dir, 0o700)
        os.chmod(a.out_dir, 0o700)  # umask に依らず 0700
        cls = classify(a.evidence, a.quiet_machine)
    except (MeasureError, OSError) as exc:
        msg = str(exc) if isinstance(exc, MeasureError) else type(exc).__name__
        print(json.dumps({"code": "runtime_error", "message": msg}), file=sys.stderr)
        return 70
    interrupted: BaseException | None = None
    try:
        with terminate_as_exception():
            for name in a.conditions:
                device, dtype = CONDITIONS[name]
                out = a.out_dir / name
                cmd = [
                    *(TIME_CMD if time_cmd is None else time_cmd),
                    sys.executable, "-m", "tools.poc26.lora_poc", "predict",
                    "--model-dir", str(a.model_dir),
                    "--model-sha256", a.model_sha256,
                    "--config-sha256", a.config_sha256,
                    "--tokenizer-sha256", a.tokenizer_sha256,
                    "--tokenizer-config-sha256", a.tokenizer_config_sha256,
                    "--definition", str(a.definition),
                    "--adapter-dir", str(a.adapter_dir),
                    "--adapter-sha256", a.adapter_sha256,
                    "--input", str(a.input),
                    "--out-dir", str(out),
                    "--device", device, "--dtype", dtype,
                    "--evidence", a.evidence,
                    "--max-seq-length", str(a.max_seq_length),
                    "--warmup", str(a.warmup),
                ]  # fmt: skip
                try:
                    load = os.getloadavg()[0]
                    err = run_child(cmd, a.timeout_seconds)
                    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
                    conds[name] = {"status": "ok", **summarize(run, parse_time_l(err), cls, load)}
                except (MeasureError, OSError, KeyError, ValueError) as exc:
                    # 条件ごとに失敗を記録して続行する（成功済みの条件は残す）。型名・固定文言だけ
                    failed = True
                    code = str(exc) if isinstance(exc, MeasureError) else type(exc).__name__
                    conds[name] = {"status": "error", "code": code}
    except BaseException as exc:  # 中断でも成功済みの記録を書いてから再送出する
        interrupted = exc
    rc = 70 if failed else 0
    if conds:
        rec = {
            "schema": "poc26-measure/1",
            "requirements": ["REQ-41", "TASK-41.1-8", "#393"],
            "quiet_machine_declared": a.quiet_machine,
            "capacity": cap,
            "conditions": conds,
            "jev_reference": {**JEV_REFERENCE, "note": "E2E vs inference-only: not comparable"},
        }
        try:
            _write_new(a.out_dir, "record.json", json.dumps(rec, indent=2, allow_nan=False) + "\n")
            _write_new(a.out_dir, "record.md", render_md(rec))
        except (OSError, ValueError, KeyError) as exc:
            print(
                json.dumps({"code": "runtime_error", "message": type(exc).__name__}),
                file=sys.stderr,
            )
            rc = 70
    if isinstance(interrupted, _Terminated):
        return 128 + interrupted.signum  # 記録は書いた。非ゼロで終える（シェル慣習）
    if interrupted is not None:
        raise interrupted
    return rc


if __name__ == "__main__":
    sys.exit(main())
