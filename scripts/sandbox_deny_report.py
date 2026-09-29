"""sandbox 下の実行中に採取した `log stream` の拒否ログを集計し、通信拒否が 0 件かを判定する。

REQ-38・TASK-38.1-2・#163。手法の出典は PoC-16（`log stream` を実行前に開始し実行後に止める
ライブ監視）。PoC-16 では拒否行を人が目視で分類したが、ここでは `eventMessage` の
**操作トークン**（`network*` で始まる操作）で機械的に判定する。部分文字列 `network` で
拾うと `networkserviceproxy`（操作は `system-info`）や `networkextension` を含むパスへの
`file-read*` を誤検出するため。

呼び出し元: `scripts/sandbox-monitor.sh`（監視の終了後に起動する）。人が既存の生ログを
再集計する場合や、TASK-38.2 の陽性対照からも単体で実行できる。製品（CLI・推論経路・配布物）には
入らない検証用スクリプトで、標準ライブラリだけを使う（依存の追加なし）。
`python3 -I` で起動する想定。

入力（すべて引数で明示。環境変数は読まない）:
  --stream            `log stream --style ndjson` の出力（先頭に非 JSON のヘッダ行が 1 行ある）
  --run-meta          `sandbox-run.sh` が書く run.meta.json
  --monitor-started-utc / --monitor-stopped-utc  監視の開始・停止時刻（UTC）
  --warmup-secs / --tail-secs / --log-override / --stream-overflow / --stream-died  監視の条件の記録

出力: stdout に JSON 1 行（REQ-33。固定の文字列と件数だけ。利用者の値は出さない）、
`--report-out` に詳細レポート。終了コードは 7 種（0・10・11・12・20・64・70。REQ-21）。

帰属（本ツール起因か）: プロセス名だけでは断定しない（無関係な python3・sh の拒否で誤って
judged_fail にしないため）。`run.meta.json` の任意キー `process_pids`（本ツールが起動した
プロセスの PID の配列）に PID が含まれ、かつ名前が許可リストにある場合だけ tool とする。
`process_pids` が無い・照合できない場合は帰属不明（unattributed）で、判定は pending(12)
（現状の `sandbox-run.sh` は PID を記録しないため、通信拒否があれば常に 12 になる。
PID 記録は将来仕様。REQ-38・TASK-38.2 で扱う）。

件数の意味: `network_deny_events` 等の通信拒否件数は、元の 1 行 + `N duplicate reports for`
の重複分 N を合算した**拒否の発生回数**。`duplicate_reports` は重複分だけの合計（内訳）。
`deny_events`・`parsed_events` はイベント行数。

資源: ログは 1 行ずつ読み（一括読み込みしない）、保持するレコードは MAX_RECORDS 件まで
（超過分は件数だけ数え、レポートの `network_denials_truncated` を true にする。REQ-39）。

判定（優先順 70 > 10 > 12 > run の終了コード > 0）:
  - 監視が無効・読めない行・時刻の不整合 -> undeterminable(70)（fail-closed）
  - 本ツール起因（PID 照合済み）の通信拒否あり -> judged_fail(10)
  - 帰属不明の通信拒否あり -> pending(12。人が確認する)
  - 全プロセスで通信拒否 0 件 -> zero_network_denials。run の終了コードを伝搬する
陽性対照（検出手段が機能することの確認）は TASK-38.2 の担当で、本スクリプトは実行しない
（出力に `positive_control:"not_run"` を明示する）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable, Iterator

# 1 行の長さの上限（REQ-39）。超えたら判定不能
MAX_LINE_BYTES = 64 * 1024
# ログ全体の上限（sandbox-monitor.sh の head -c と揃える）
MAX_STREAM_BYTES = 256 * 1024 * 1024
MAX_META_BYTES = 1024 * 1024
# レポートに保持する通信拒否レコードの上限（超過分は件数のみ。REQ-39）
MAX_RECORDS = 1000
MAX_PIDS = 4096
MAX_TARGET_BYTES = 256
HEADER_PREFIX = "Filtering the log data"
VALID_EXIT_CODES = {0, 10, 11, 12, 20, 64, 70}
CODE_NAMES = {
    0: "ok",
    10: "judged_fail",
    11: "out_of_scope",
    12: "pending",
    20: "limit_exceeded",
    64: "invalid_input",
    70: "runtime_error",
}
# sandbox の内側で動きうるプロセス名（本ツール起因とみなす固定の許可リスト）。
# sandbox の外で起動される head 等は含めない
TOOL_PROCESSES = frozenset(
    {
        "sandbox-exec",
        "fandhe-edge",
        "python",
        "python3",
        "python3.12",
        "Python",
        "sh",
        "bash",
    }
)
# `Sandbox: <プロセス名>(<pid>) deny(<n>) <操作> [対象]`。プロセス名は括弧を含みうるため
# `(<数字>) deny(` を右側から照合する（貪欲な `.+` の後ろ向き探索。行長は事前に制限済み）
EVENT_RE = re.compile(
    r"^(?:(\d+) duplicate reports? for )?Sandbox: (.+)\((\d+)\) deny\((\d+)\) (\S+)(?: (.*))?$",
    re.DOTALL,
)
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class Undeterminable(Exception):
    """監視結果を判定できない（fail-closed で 70 にする）。理由は固定文字列。"""


def read_bounded(path: str, limit: int) -> bytes:
    """ファイルサイズを読む前に確認してから読む（REQ-39）。"""
    try:
        size = os.stat(path).st_size
        if size > limit:
            raise Undeterminable("input file exceeds size limit")
        with open(path, "rb") as f:
            return f.read(limit + 1)
    except OSError as e:
        raise Undeterminable("cannot read input file") from e


def load_run_meta(path: str) -> dict:
    """run.meta.json の必須キーと型を検証する。欠落・契約外の値は判定不能。"""
    try:
        meta = json.loads(read_bounded(path, MAX_META_BYTES).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise Undeterminable("run record is not valid json") from e
    if not isinstance(meta, dict):
        raise Undeterminable("run record is not an object")
    rc = meta.get("exit_code")
    if isinstance(rc, bool) or not isinstance(rc, int) or rc not in VALID_EXIT_CODES:
        raise Undeterminable("run record exit_code is outside the contract")
    for key in ("started_utc", "ended_utc"):
        v = meta.get(key)
        if not isinstance(v, str) or not UTC_RE.match(v):
            raise Undeterminable("run record timestamp is invalid")
    if not isinstance(meta.get("evidence_hint"), str):
        raise Undeterminable("run record evidence_hint is invalid")
    if not isinstance(meta.get("sandbox_exec_override"), bool):
        raise Undeterminable("run record sandbox_exec_override is invalid")
    pids = meta.get("process_pids", [])
    if (
        not isinstance(pids, list)
        or len(pids) > MAX_PIDS
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in pids)
    ):
        raise Undeterminable("run record process_pids is invalid")
    return meta


def truncate_target(text: str) -> str:
    """対象文字列を 256 バイトで切り詰める（他アプリのパスを長く転記しない）。"""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= MAX_TARGET_BYTES:
        return text
    return raw[:MAX_TARGET_BYTES].decode("utf-8", errors="ignore")


def classify(events_raw: Iterable[str], tool_pids: frozenset[int] = frozenset()) -> dict:
    """`eventMessage` を 1 件ずつ分類して件数とレコード（上限あり）を返す。

    `tool_pids` に含まれる PID かつ許可リストのプロセス名だけを tool とし、それ以外は
    帰属不明にする（名前だけでは断定しない）。通信拒否の件数は重複報告分を合算する。
    """
    counts = {
        "parsed_events": 0,
        "deny_events": 0,
        "duplicate_reports": 0,
        "network_deny_events": 0,
        "tool_network_deny_events": 0,
        "unattributed_network_deny_events": 0,
        "unrecognized_deny_events": 0,
    }
    records: list[dict] = []
    truncated = False

    def keep(rec: dict) -> None:
        nonlocal truncated
        if len(records) < MAX_RECORDS:
            records.append(rec)
        else:
            truncated = True

    for msg in events_raw:
        counts["parsed_events"] += 1
        if "deny" not in msg:
            continue
        counts["deny_events"] += 1
        m = EVENT_RE.match(msg)
        if m is None:
            counts["unrecognized_deny_events"] += 1
            if "network" in msg:
                # 形式外でも network を含む拒否は fail-closed で帰属不明の通信拒否とする
                counts["network_deny_events"] += 1
                counts["unattributed_network_deny_events"] += 1
                keep(
                    {
                        "process": None,
                        "pid": None,
                        "operation": None,
                        "target": truncate_target(msg),
                        "attribution": "unattributed",
                        "recognized": False,
                    }
                )
            continue
        dup, proc, pid, _n, op, target = m.groups()
        dup_n = int(dup) if dup is not None else 0
        counts["duplicate_reports"] += dup_n
        if not op.startswith("network"):
            continue
        occurrences = 1 + dup_n
        counts["network_deny_events"] += occurrences
        tool = proc in TOOL_PROCESSES and int(pid) in tool_pids
        counts["tool_network_deny_events" if tool else "unattributed_network_deny_events"] += (
            occurrences
        )
        keep(
            {
                "process": proc,
                "pid": int(pid),
                "operation": op,
                "target": truncate_target(target or ""),
                "occurrences": occurrences,
                "attribution": "tool" if tool else "unattributed",
                "recognized": True,
            }
        )
    return {"counts": counts, "records": records, "truncated": truncated}


def iter_stream(path: str, stats: dict) -> Iterator[str]:
    """ndjson を 1 行ずつ読んで `eventMessage` を返す。総行数は `stats["lines"]` に入れる。

    ファイル全体を読み込まず、1 行の読み込みも MAX_LINE_BYTES + 1 で打ち切る（REQ-39）。
    読めない行・ヘッダ欠落・空出力は判定不能。
    """
    try:
        if os.stat(path).st_size > MAX_STREAM_BYTES:
            raise Undeterminable("input file exceeds size limit")
        f = open(path, "rb")  # ジェネレータの寿命に合わせて finally で閉じる
    except OSError as e:
        raise Undeterminable("cannot read input file") from e
    total = 0
    try:
        while True:
            try:
                raw = f.readline(MAX_LINE_BYTES + 1)
            except OSError as e:
                raise Undeterminable("cannot read input file") from e
            if not raw:
                break
            total += len(raw)
            if total > MAX_STREAM_BYTES:
                raise Undeterminable("input file exceeds size limit")
            if len(raw) > MAX_LINE_BYTES:
                raise Undeterminable("log stream line exceeds length limit")
            if raw.endswith(b"\n"):
                raw = raw[:-1]
            i = stats["lines"]
            stats["lines"] = i + 1
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                raise Undeterminable("log stream line is not utf-8") from e
            if i == 0:
                # 先頭はヘッダ行がちょうど 1 行。ヘッダが無ければ監視が機能していない
                if not text.startswith(HEADER_PREFIX):
                    raise Undeterminable("log stream header is missing")
                continue
            try:
                obj = json.loads(text)
            except ValueError as e:
                raise Undeterminable("log stream line is not valid json") from e
            msg = obj.get("eventMessage") if isinstance(obj, dict) else None
            if not isinstance(msg, str):
                raise Undeterminable("log stream event has no eventMessage")
            yield msg
    finally:
        f.close()
    if stats["lines"] == 0:
        raise Undeterminable("log stream output is empty")


def build(args: argparse.Namespace) -> tuple[int, dict, dict]:
    """判定して (終了コード, stdout 用, レポート用) を返す。"""
    counts_zero = {
        "stream_lines": 0,
        "parsed_events": 0,
        "deny_events": 0,
        "duplicate_reports": 0,
        "network_deny_events": 0,
        "tool_network_deny_events": 0,
        "unattributed_network_deny_events": 0,
        "unrecognized_deny_events": 0,
    }
    counts = dict(counts_zero)
    records: list[dict] = []
    truncated = False
    run_exit: int | None = None
    hint = "requires_human_review"
    try:
        meta = load_run_meta(args.run_meta)
        run_exit = meta["exit_code"]
        if meta["sandbox_exec_override"] or args.log_override:
            hint = "test_harness"
        if args.warmup_secs != 3 or args.tail_secs != 60:
            hint = "test_harness"
        if args.stream_overflow:
            raise Undeterminable("log stream output exceeded capacity")
        if args.stream_died:
            raise Undeterminable("log stream ended before the monitoring window closed")
        stats = {"lines": 0}
        result = classify(
            iter_stream(args.stream, stats),
            frozenset(int(x) for x in meta.get("process_pids", [])),
        )
        counts.update(result["counts"])
        counts["stream_lines"] = stats["lines"]
        records = result["records"]
        truncated = result["truncated"]
        for key in (args.monitor_started_utc, args.monitor_stopped_utc):
            if not UTC_RE.match(key):
                raise Undeterminable("monitor timestamp is invalid")
        if not (
            args.monitor_started_utc <= meta["started_utc"]
            and meta["ended_utc"] <= args.monitor_stopped_utc
        ):
            raise Undeterminable("run is not covered by the monitoring window")
        if counts["tool_network_deny_events"] > 0:
            verdict, rc = "tool_network_denials_found", 10
            msg_text = "network denials attributed to the tool were found"
        elif counts["unattributed_network_deny_events"] > 0:
            verdict, rc = "unattributed_network_denials", 12
            msg_text = "network denials from unattributed processes need human review"
        else:
            verdict, rc = "zero_network_denials", run_exit
            msg_text = (
                "no network denials were observed"
                if run_exit == 0
                else "no network denials were observed but the run did not complete"
            )
    except Undeterminable as e:
        verdict, rc = "undeterminable", 70
        msg_text = str(e)
    summary = {
        "code": CODE_NAMES.get(rc, "runtime_error"),
        "message": msg_text,
        "network_verdict": verdict,
        "run_exit_code": run_exit,
        "run_code": CODE_NAMES.get(run_exit) if run_exit is not None else None,
        "counts": counts,
        "positive_control": "not_run",
        "evidence_hint": hint,
        "log_stream_override": bool(args.log_override),
        "report": "network_report.json",
    }
    report = {
        **summary,
        "predicate": 'process == "kernel" AND eventMessage CONTAINS "deny"',
        "monitor_started_utc": args.monitor_started_utc,
        "monitor_stopped_utc": args.monitor_stopped_utc,
        "warmup_secs": args.warmup_secs,
        "tail_secs": args.tail_secs,
        "network_denials": records,
        "network_denials_truncated": truncated,
    }
    return rc, summary, report


def main(argv: list[str]) -> int:
    """引数を解釈して集計し、stdout に JSON 1 行を出す。"""
    p = argparse.ArgumentParser(prog="sandbox_deny_report.py")
    p.add_argument("--stream", required=True)
    p.add_argument("--run-meta", required=True)
    p.add_argument("--monitor-started-utc", required=True)
    p.add_argument("--monitor-stopped-utc", required=True)
    p.add_argument("--warmup-secs", type=int, default=3)
    p.add_argument("--tail-secs", type=int, default=60)
    p.add_argument("--log-override", action="store_true")
    p.add_argument("--stream-overflow", action="store_true")
    p.add_argument("--stream-died", action="store_true")
    p.add_argument("--report-out", required=True)
    try:
        args = p.parse_args(argv)
    except SystemExit:
        print(json.dumps({"code": "invalid_input", "message": "invalid arguments"}))
        return 64
    rc, summary, report = build(args)
    try:
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False)
            f.write("\n")
    except OSError:
        print(json.dumps({"code": "runtime_error", "message": "cannot write report"}))
        return 70
    print(json.dumps(summary, ensure_ascii=False))
    return rc if rc in VALID_EXIT_CODES else 70


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
