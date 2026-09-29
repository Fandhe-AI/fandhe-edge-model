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

帰属（本ツール起因か）の根拠: 第一の根拠は PID。拒否行 `Sandbox: <プロセス名>(<pid>) deny(...)` の
PID が `run.meta.json` の `process_pids`（`sandbox-run.sh` が工程のプロセスグループから
0.1 秒間隔で採取した PID。sandbox の内側で本ツールが起動したプロセスの集合）に含まれれば tool。
プロセス名は根拠にしない（Python の版・実行ファイル名は環境で変わり、固定の許可リストでは
取りこぼす）。名前は出力する固定語彙（`process` ラベル）への正規化にだけ使い、
`TOOL_NAME_RE`（`python3` と任意の `.N`・sh 等）に合えばその名前、他は `other` とする。
PID が採取できなかった短命プロセスの拒否は帰属不明（unattributed）で pending(12)
（過大に本ツール起因と断定しない側）。既知の限界: 採取後に PID が再利用され同じ窓内で無関係な
プロセスが同じ PID を得た場合は tool と誤帰属しうる（窓は 1 回の実行に限る）。
出力へ出すのは判定結果（`attribution`）と固定語彙だけで、生のプロセス名は出さない。
実ログの形式: 上の形式は PoC-16 で観測した macOS の `log stream` の出力に基づく。本リポの
fixture は合成データで、実機の出力そのものではない（証拠種別: テストハーネス）。

件数の意味: `network_deny_events` 等の通信拒否件数は**拒否の発生回数**。元の 1 行は 1 回、
`N duplicate reports for` の要約行は、同一イベント（`event_key`）の元の行が先にあれば N 回、
無ければ 1+N 回。`duplicate_reports` は重複分だけの合計（内訳）。
`deny_events`・`parsed_events` はイベント行数。

生文字列の非保存（P0）: 拒否ログの対象（通信先・パス）・許可リスト外のプロセス名・形式外の行の
本文は、レポート・stdout・例外メッセージのどこにも書かない。レポートに載せるのは件数・固定語彙
（許可リストで正規化したプロセス名 `other`・`network*` の操作トークン）・PID・実行ごとの salt
付きダイジェスト（`target_digest`。salt は保存しない）に限る。

資源: ログは 1 行ずつ読み（一括読み込みしない）、保持するレコードは MAX_RECORDS 件まで
（超過分は件数だけ数え、レポートの `network_denials_truncated` を true にする。REQ-39）。

判定（優先順は `decide()` が唯一の定義。判定不能 70 > run の 70 > 10 > 12 > run のその他 > 0）:
  - 監視が無効・読めない行・時刻の不整合 -> undeterminable(70)（fail-closed）
  - 本ツール起因（PID 照合済み）の通信拒否あり -> judged_fail(10)
  - 帰属不明の通信拒否あり -> pending(12。人が確認する)
  - 全プロセスで通信拒否 0 件 -> zero_network_denials。run の終了コードを伝搬する
陽性対照（検出手段が機能することの確認）は TASK-38.2 の担当で、本スクリプトは実行しない
（出力に `positive_control:"not_run"` を明示する）。
"""

from __future__ import annotations

import argparse
import hashlib
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
# 重複報告の突き合わせ用に保持する未照合の元イベントの上限（メモリ上限。REQ-39）。
# 超過時は照合できないものとして 1+N に数える（過小に数えない側へ倒す）
MAX_PENDING_EVENTS = 100_000
DIGEST_HEX_LEN = 12
OPERATION_RE = re.compile(r"^network[a-z0-9*-]{0,40}$")
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
# 出力する `process` ラベルの固定語彙への正規化（帰属の根拠ではない）。`python3` と任意の `.N`
# （`python3.11`・`python3.13` 等）を受け付け、`python3-evil` のような別名は `other` にする
TOOL_NAME_RE = re.compile(r"^(?:sandbox-exec|fandhe-edge|python(?:3(?:\.\d+)?)?|Python|sh|bash)$")
# `Sandbox: <プロセス名>(<pid>) deny(<n>) <操作> [対象]`。プロセス名は括弧を含みうるため
# `(<数字>) deny(` を右側から照合する（貪欲な `.+` の後ろ向き探索。行長は事前に制限済み）
EVENT_RE = re.compile(
    r"^(?:(\d+) duplicate reports? for )?Sandbox: (.+)\((\d+)\) deny\((\d+)\) (\S+)(?: (.*))?$",
    re.DOTALL,
)
# 「拒否行」の判別子（唯一の定義）。語境界の `deny` で、`deny(1)`・`deny` は拒否行、`denied` 等は
# 拒否行ではない。拒否行は EVENT_RE で操作を読み取れなければ判定不能になる
DENY_LINE_RE = re.compile(r"\bdeny\b")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def is_deny_line(msg: str) -> bool:
    """`eventMessage` が拒否行か（判別子は DENY_LINE_RE の 1 か所）。"""
    return DENY_LINE_RE.search(msg) is not None


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


def reconcile_run_exit(recorded: int, actual: int) -> int:
    """run.meta.json の exit_code と実行スクリプトの実際の終了コードを照合する（唯一の規則）。

    一致しなければ、記録後に実行スクリプトが異常終了した等で完走を誤認しうるため
    判定不能（fail-closed）にする。
    """
    if recorded != actual:
        raise Undeterminable("run record exit_code does not match the actual exit status")
    return recorded


def target_digest(salt: bytes, text: str) -> str:
    """対象文字列（通信先・パス・ログ本文）の実行ごとの salt 付きダイジェスト先頭 12 桁。

    生文字列はレポートへ書かない（P0）。salt は実行ごとの乱数で**どこにも保存しない**ため、
    低エントロピーな値（ホスト名・IP）を辞書照合で復元できず、同一レポート内の同一対象の
    識別にだけ使える。
    """
    h = hashlib.sha256(salt + text.encode("utf-8", errors="replace")).hexdigest()
    return h[:DIGEST_HEX_LEN]


def event_key(proc: str, pid: str, deny_n: str, op: str, target: str | None) -> bytes:
    """同一拒否イベントの識別規則（唯一の定義）。

    `Sandbox: <プロセス名>(<pid>) deny(<n>) <操作> <対象>` の全要素が一致する元の行と
    `N duplicate reports for` の要約行を同一イベントとみなす（要約行は元の行より後に出る。
    要約行にはイベント ID・時刻が無く、元の行と時刻も異なるため）。キーはメモリ上だけで使い、
    ダイジェスト化して保持する（対象文字列を保持しない）。
    """
    raw = "\x00".join((proc, pid, deny_n, op, target or ""))
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).digest()


def classify(events_raw: Iterable[str], tool_pids: frozenset[int] = frozenset()) -> dict:
    """`eventMessage` を 1 件ずつ分類して件数とレコード（上限あり）を返す。

    `tool_pids` に含まれる PID だけを tool とし、それ以外は帰属不明にする（名前は根拠にしない）。

    発生回数: 元の行は 1 回。`N duplicate reports for` の要約行は、同じ `event_key` の元の行が
    先に数えられていれば N 回だけ加算し（元の 1 回を二重に数えない）、元の行が無ければ
    元の 1 回を含めて 1+N 回とする。

    レポートに載せるのは件数・固定語彙・salt 付きダイジェストだけで、ログ上の生文字列
    （プロセス名の許可リスト外・対象・形式外の行）は保存しない。
    """
    counts = {
        "parsed_events": 0,
        "deny_events": 0,
        "duplicate_reports": 0,
        "network_deny_events": 0,
        "tool_network_deny_events": 0,
        "unattributed_network_deny_events": 0,
        "ignored_non_deny_events": 0,
    }
    records: list[dict] = []
    truncated = False
    salt = os.urandom(16)
    # event_key -> 要約行にまだ照合されていない元の行のレコード位置（保持できなければ -1）
    pending: dict[bytes, list[int]] = {}
    pending_total = 0

    def keep(rec: dict) -> int | None:
        nonlocal truncated
        if len(records) < MAX_RECORDS:
            records.append(rec)
            return len(records) - 1
        truncated = True
        return None

    for msg in events_raw:
        counts["parsed_events"] += 1
        if not is_deny_line(msg):
            # 拒否行ではない（述語の部分文字列 `deny` に `denied` 等が当たっただけ）。
            # 無視してよい行はここだけで扱い、件数を `ignored_non_deny_events` に残す
            counts["ignored_non_deny_events"] += 1
            continue
        counts["deny_events"] += 1
        m = EVENT_RE.match(msg)
        if m is None:
            # 拒否行なのに操作を読み取れない（ログ形式の変化等）。`network` を含むか否かで分けず、
            # 「0 件」側へ倒さないため判定不能にする（fail-closed。REQ-38）。本文は例外へ含めない
            raise Undeterminable("log stream contains a deny line in an unknown format")
        dup, proc, pid, deny_n, op, target = m.groups()
        dup_n = int(dup) if dup is not None else 0
        counts["duplicate_reports"] += dup_n
        if not op.startswith("network"):
            continue
        key = event_key(proc, pid, deny_n, op, target)
        claimed = False
        if dup is not None:
            waiting = pending.get(key)
            if waiting:
                idx = waiting.pop()
                pending_total -= 1
                claimed = True
                if 0 <= idx < len(records):
                    records[idx]["occurrences"] += dup_n
        occurrences = dup_n if claimed else 1 + dup_n
        counts["network_deny_events"] += occurrences
        tool = int(pid) in tool_pids
        counts["tool_network_deny_events" if tool else "unattributed_network_deny_events"] += (
            occurrences
        )
        if claimed:
            continue
        idx = keep(
            {
                "process": proc if TOOL_NAME_RE.match(proc) else "other",
                "pid": int(pid),
                "operation": op if OPERATION_RE.match(op) else "network-other",
                "target_digest": target_digest(salt, target or ""),
                "occurrences": occurrences,
                "attribution": "tool" if tool else "unattributed",
                "recognized": True,
            }
        )
        if dup is None and pending_total < MAX_PENDING_EVENTS:
            pending.setdefault(key, []).append(-1 if idx is None else idx)
            pending_total += 1
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


def decide(run_exit: int, tool_n: int, unattributed_n: int) -> tuple[str, int, str]:
    """最終の (network_verdict, 終了コード, message) を決める（優先順の唯一の定義）。

    判定不能（監視の異常・読めない行・時刻の不整合）は呼び出し元が 70 で先に打ち切る。ここでは
    優先順 run が 70 > 本ツール起因の拒否 10 > 帰属不明の拒否 12 > run のその他の終了コード。
    run が 70（実行失敗）なら拒否の有無にかかわらず 70 を返す（完走していない実行を、拒否件数の
    判定で 10・12 に上書きしない）。拒否件数と network_verdict は 70 でも常にレポートへ残す。
    """
    if tool_n > 0:
        verdict = "tool_network_denials_found"
        text = "network denials attributed to the tool were found"
        rc = 10
    elif unattributed_n > 0:
        verdict = "unattributed_network_denials"
        text = "network denials from unattributed processes need human review"
        rc = 12
    else:
        verdict = "zero_network_denials"
        text = (
            "no network denials were observed"
            if run_exit == 0
            else "no network denials were observed but the run did not complete"
        )
        rc = run_exit
    if run_exit == 70:
        return verdict, 70, "the sandboxed run failed with runtime_error; " + text
    return verdict, rc, text


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
        "ignored_non_deny_events": 0,
    }
    counts = dict(counts_zero)
    records: list[dict] = []
    truncated = False
    run_exit: int | None = None
    hint = "requires_human_review"
    try:
        meta = load_run_meta(args.run_meta)
        run_exit = reconcile_run_exit(meta["exit_code"], args.run_exit_code)
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
        verdict, rc, msg_text = decide(
            run_exit, counts["tool_network_deny_events"], counts["unattributed_network_deny_events"]
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
    p.add_argument("--run-exit-code", type=int, required=True)
    p.add_argument("--report-out", required=True)
    # argparse 既定のエラーは不正な値を stderr へ複写するため、固定の出力に置き換える
    p.error = lambda _message: (_ for _ in ()).throw(SystemExit(2))  # type: ignore[method-assign]
    try:
        args = p.parse_args(argv)
    except SystemExit:
        print(json.dumps({"code": "invalid_input", "message": "invalid arguments"}))
        return 64
    try:
        rc, summary, report = build(args)
    except Exception:  # 想定外の例外でも本文を含む traceback を出さない
        print(json.dumps({"code": "runtime_error", "message": "unexpected report failure"}))
        return 70
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
