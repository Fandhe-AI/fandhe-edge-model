"""MCP 接続確認用サーバー（scripts/mcp_probe.py）の単体テスト（REQ-37・PoC-18・#353）。"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "mcp_probe.py"

spec = importlib.util.spec_from_file_location("mcp_probe", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_req37_stdio_handshake_discover_fallback_and_tools_list() -> None:
    """初期化・`server/discover` の -32601（旧方式への後戻り）・通知への無応答・tools/list。"""
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "server/discover"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope"}},
    ]
    stdin = "".join(json.dumps(m) + "\n" for m in msgs) + "not json\n"
    out = subprocess.run(  # noqa: S603  固定の引数
        [sys.executable, "-I", str(SCRIPT)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    replies = [json.loads(line) for line in out.splitlines()]
    assert [r.get("id") for r in replies] == [1, 2, 3, 4, None]
    assert replies[0]["error"]["code"] == -32601
    assert replies[1]["result"]["protocolVersion"] == "2025-06-18"
    assert replies[1]["result"]["capabilities"] == {"tools": {}}
    assert [t["name"] for t in replies[2]["result"]["tools"]] == ["infer"]
    assert replies[3]["error"]["code"] == -32602
    assert replies[4]["error"]["code"] == -32700


def test_req37_infer_relays_the_cli_json_and_flags_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ラッパーの stdout（1 JSON）を返し、判定以外の終了コードは isError（REQ-21）。"""
    fake = tmp_path / "wrapper.sh"
    fake.write_text(
        '#!/bin/sh\necho \'{"id":"input","status":"ok"}\'\nexit "${FAKE_RC:-0}"\n', encoding="utf-8"
    )
    fake.chmod(0o755)
    monkeypatch.setattr(mod, "WRAPPER", fake)
    ok = mod.call_infer({"package": "p", "text": "t"})
    assert ok == {
        "content": [{"type": "text", "text": '{"id":"input","status":"ok"}'}],
        "isError": False,
    }
    monkeypatch.setenv("FAKE_RC", "12")
    assert mod.call_infer({"package": "p", "text": "t"})["isError"] is False
    monkeypatch.setenv("FAKE_RC", "64")
    assert mod.call_infer({"package": "p", "text": "t"})["isError"] is True
    assert mod.call_infer({"package": "p"})["isError"] is True
    big = "x" * (mod.MAX_TEXT_BYTES + 1)
    assert "too large" in mod.call_infer({"package": "p", "text": big})["content"][0]["text"]


def test_req39_malformed_requests_do_not_stop_the_server() -> None:
    """深い入れ子・単独サロゲートの id・NUL・`null` の行でも落ちずに応答する（REQ-39）。"""
    lines = [
        "[" * 100000 + "]" * 100000,
        '{"jsonrpc":"2.0","id":"\\ud800","method":"ping"}',
        '{"jsonrpc":"2.0","id":{"a":1},"method":"ping"}',
        "null",
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"infer",'
        '"arguments":{"package":"p","text":"a\\u0000b"}}}',
        '{"jsonrpc":"2.0","id":1e999,"method":"ping"}',
        '{"jsonrpc":"2.0","id":NaN,"method":"ping"}',
        '{"jsonrpc":"2.0","id":8,"method":"ping"}',
    ]
    out = subprocess.run(  # noqa: S603  固定の引数
        [sys.executable, "-I", str(SCRIPT)],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    replies = [json.loads(line) for line in out.splitlines()]
    assert [r.get("id") for r in replies] == [None, "\ud800", None, None, 7, None, None, 8]
    assert replies[0]["error"]["code"] == -32700
    assert replies[1]["result"] == {}
    assert replies[2]["error"]["code"] == -32600
    assert replies[3]["error"]["code"] == -32600
    assert replies[4]["result"]["isError"] is True
    assert "invalid characters" in replies[4]["result"]["content"][0]["text"]
    assert replies[5]["error"]["code"] == -32600
    assert replies[6]["error"]["code"] == -32700
    assert replies[7]["result"] == {}
