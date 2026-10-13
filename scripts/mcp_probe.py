"""MCP 接続確認用の最小 stdio サーバー（REQ-37・PoC-18 の成功基準 3・#353）。

Claude Code などの実クライアントへ登録し、MCP 経由で `infer` を 1 回呼べるかを確かめる
ための検証用。MCP の採否が決まる前の試作で、採用後の本実装（TASK-37.2）はこれを
契約の参考にして作り直す（このスクリプトを製品へ組み込まない）。

- プロトコル: 改行区切り JSON-RPC 2.0・2025-11-25 の初期化ハンドシェイク方式
  （`initialize` → `notifications/initialized`）。2026-07-28 の `server/discover` には
  -32601 を返し、クライアントに旧方式へ戻ってもらう（MCP 仕様の stdio 後方互換）
- 公開するツールは `infer` 1 つ。中身は `scripts/cli-infer-noninteractive.sh`
  （REQ-36。時間・出力の上限、JSON と終了コードの検証つき）を固定 argv・シェル非経由で
  呼び、その stdout（1 JSON）を返す。経路の閉じ込めなどのガード層は CLI 側が行う（REQ-39）
- stdout にはプロトコルのメッセージ以外を書かない。入力テキストはどこにも記録しない
- 通信しない（stdio のみ。REQ-38）。標準ライブラリのみ
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts" / "cli-infer-noninteractive.sh"
PROTOCOL_VERSION = "2025-11-25"
KNOWN_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26")
# 1 行の上限。`infer --text` の入力に十分な大きさで、無制限の読み込みを作らない（REQ-39）
MAX_LINE_BYTES = 1024 * 1024
MAX_TEXT_BYTES = 64 * 1024
# ラッパー自身の上限（既定 300 秒）より長く待たない
CALL_TIMEOUT_SEC = 320
# CLI の終了コードのうち判定として正常なもの（ok・out_of_scope・pending。REQ-21）
JUDGED_EXIT_CODES = (0, 11, 12)

INFER_TOOL = {
    "name": "infer",
    "description": (
        "Run fandhe-edge infer on one text with a local package directory "
        "(relative to the repository root). Returns the CLI JSON line."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "package": {"type": "string", "description": "package directory"},
            "text": {"type": "string", "description": "input text"},
        },
        "required": ["package", "text"],
        "additionalProperties": False,
    },
}


def call_infer(args: Any) -> dict[str, Any]:
    """`infer` ツールの実行結果（MCP の `tools/call` の result）。"""
    package, text = (
        (args.get("package"), args.get("text")) if isinstance(args, dict) else (None, None)
    )
    if not isinstance(package, str) or not isinstance(text, str) or not package:
        return _text_result(
            '{"code":"invalid_input","message":"package and text are required"}', True
        )
    # NUL は argv に渡せず、単独サロゲートは UTF-8 にできない。どちらも入力の誤りとして返す
    if "\x00" in package or "\x00" in text or not _utf8_ok(package) or not _utf8_ok(text):
        return _text_result('{"code":"invalid_input","message":"invalid characters"}', True)
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        return _text_result('{"code":"invalid_input","message":"text too large"}', True)
    try:
        proc = subprocess.run(  # noqa: S603  固定のラッパーへ引数リストで渡す。shell は使わない
            [str(WRAPPER), "--package", package, "--text", text],
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=CALL_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _text_result('{"code":"runtime_error","message":"infer could not be run"}', True)
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    return _text_result(out, proc.returncode not in JUDGED_EXIT_CODES)


def _utf8_ok(s: str) -> bool:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _text_result(text: str, is_error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def handle(msg: Any) -> dict[str, Any] | None:
    """1 メッセージへの応答。通知（`id` なし）には何も返さない。"""
    if (
        not isinstance(msg, dict)
        or msg.get("jsonrpc") != "2.0"
        or not isinstance(msg.get("method"), str)
    ):
        return _error(None, -32600, "invalid request")
    if "id" not in msg:
        return None
    mid, method, params = msg["id"], msg["method"], msg.get("params")
    if not _valid_id(mid):
        return _error(None, -32600, "invalid request")
    if method == "initialize":
        asked = params.get("protocolVersion") if isinstance(params, dict) else None
        return _ok(
            mid,
            {
                "protocolVersion": asked if asked in KNOWN_VERSIONS else PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fandhe-edge-probe", "version": "0.0.1"},
            },
        )
    if method == "ping":
        return _ok(mid, {})
    if method == "tools/list":
        return _ok(mid, {"tools": [INFER_TOOL]})
    if method == "tools/call":
        name = params.get("name") if isinstance(params, dict) else None
        if name != "infer":
            return _error(mid, -32602, "unknown tool")
        return _ok(mid, call_infer(params.get("arguments")))
    return _error(mid, -32601, "method not found")


def _valid_id(mid: Any) -> bool:
    """JSON-RPC の id は文字列・有限の数値・null。入れ子や Infinity を応答へ写さない。"""
    if mid is None or (isinstance(mid, (str, int)) and not isinstance(mid, bool)):
        return True
    return isinstance(mid, float) and math.isfinite(mid)


def _reject_constant(name: str) -> Any:
    raise ValueError(name)


def _ok(mid: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def main() -> None:
    """stdin の 1 行ごとに応答を stdout へ 1 行で書く。EOF で終わる。"""
    stdin, stdout = sys.stdin.buffer, sys.stdout
    while True:
        line = stdin.readline(MAX_LINE_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_LINE_BYTES:
            # 上限を超えた行は読み捨てて終わる（途中から読み直すと別の要求と誤解する）
            reply: dict[str, Any] | None = _error(None, -32600, "message too large")
        elif not line.strip():
            continue
        else:
            try:
                # NaN・Infinity は JSON の値ではない（1e999 は id の検査で拒否する）
                msg = json.loads(line, parse_constant=_reject_constant)
            except (ValueError, RecursionError):
                # RecursionError は深い入れ子（1 行の上限内でも起こる）。落ちずに拒否する
                reply = _error(None, -32700, "parse error")
            else:
                try:
                    reply = handle(msg)
                except Exception:  # 1 要求の失敗でサーバーを止めない。id を保って返す
                    mid = msg.get("id") if isinstance(msg, dict) else None
                    reply = _error(mid if _valid_id(mid) else None, -32603, "internal error")
        if reply is not None:
            # ensure_ascii（既定）で単独サロゲートも \\uXXXX にエスケープし、書き込みで落とさない
            stdout.write(json.dumps(reply, allow_nan=False) + "\n")
            stdout.flush()
        if len(line) > MAX_LINE_BYTES:
            return


if __name__ == "__main__":
    main()
