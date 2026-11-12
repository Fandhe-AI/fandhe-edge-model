"""Claude Code プラグイン（plugin/・.claude-plugin/）の検査（REQ-33・REQ-21・#729）。

版の一致・構造・ラッパーの挙動（スタブの `fandhe-edge` を PATH 先頭に置いて確認）。
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "plugin"
WRAPPER = PLUGIN / "bin" / "fandhe-edge-judge"

# 引数を記録し、serve は標識を書いて終わり、infer は固定 JSON を STUB_RC で返す。
STUB = """#!/bin/sh
echo "$@" >> "$STUB_LOG/$1.args"
if [ "$1" = serve ]; then
    : > "$STUB_LOG/serve.started"
    exit 0
fi
echo '{"id":"input","status":"ok"}'
exit "${STUB_RC:-0}"
"""


def _workspace_version() -> str:
    data = tomllib.loads((REPO / "Cargo.toml").read_text(encoding="utf-8"))
    return data["workspace"]["package"]["version"]


def test_req33_versions_match_workspace() -> None:
    """plugin.json・marketplace.json の版と name が workspace と一致する。"""
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    market = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert plugin["name"] == "fandhe-edge"
    assert plugin["version"] == _workspace_version()
    entry = market["plugins"][0]
    assert entry["name"] == plugin["name"]
    assert entry["version"] == plugin["version"]
    assert entry["source"] == "./plugin"
    assert (REPO / "plugin").is_dir()


def test_req33_structure() -> None:
    """skill の frontmatter とラッパーの実行権限・構文。"""
    skill = (PLUGIN / "skills" / "judge" / "SKILL.md").read_text(encoding="utf-8")
    head = skill.split("---")[1]
    assert "name: judge" in head
    assert "description:" in head
    assert "${CLAUDE_PLUGIN_ROOT}/bin/fandhe-edge-judge" in skill
    assert os.access(WRAPPER, os.X_OK)
    subprocess.run(["sh", "-n", str(WRAPPER)], check=True)  # noqa: S603, S607


def _run(
    tmp_path: Path, env_extra: dict[str, str], *, stub: bool = True
) -> subprocess.CompletedProcess[str]:
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "log"
    log.mkdir(exist_ok=True)
    if stub:
        s = bindir / "fandhe-edge"
        s.write_text(STUB, encoding="utf-8")
        s.chmod(0o755)
    # 本物の fandhe-edge を拾わないよう PATH は stub と最小限だけにする。
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "STUB_LOG": str(log), **env_extra}
    return subprocess.run(  # noqa: S603  固定の引数
        ["/bin/sh", str(WRAPPER), "--text", "hello"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("rc", [0, 11, 12, 20, 70])
def test_req21_infer_output_and_exit_code_pass_through(tmp_path: Path, rc: int) -> None:
    """infer の stdout と終了コードがそのまま返り、引数は --package 付きで渡る。"""
    r = _run(tmp_path, {"FANDHE_EDGE_PACKAGE": "pkg", "STUB_RC": str(rc)})
    assert r.returncode == rc
    assert r.stdout == '{"id":"input","status":"ok"}\n'
    assert (tmp_path / "log" / "infer.args").read_text() == "infer --package pkg --text hello\n"


def test_req33_serve_started_with_idle_exit(tmp_path: Path) -> None:
    """serve が --package と既定 600 秒の --idle-exit-secs で裏起動される。"""
    _run(tmp_path, {"FANDHE_EDGE_PACKAGE": "pkg"})
    marker = tmp_path / "log" / "serve.started"
    for _ in range(50):
        if marker.exists():
            break
        time.sleep(0.1)
    assert marker.exists()
    assert (tmp_path / "log" / "serve.args").read_text() == (
        "serve --package pkg --idle-exit-secs 600\n"
    )


def test_req33_idle_exit_secs_env(tmp_path: Path) -> None:
    """環境変数で待ち時間を変えられ、不正値は exit 64 の JSON。"""
    _run(tmp_path, {"FANDHE_EDGE_PACKAGE": "pkg", "FANDHE_EDGE_IDLE_EXIT_SECS": "30"})
    marker = tmp_path / "log" / "serve.started"
    for _ in range(50):
        if marker.exists():
            break
        time.sleep(0.1)
    assert "--idle-exit-secs 30\n" in (tmp_path / "log" / "serve.args").read_text()
    for bad in ["0", "abc", "-1", "1.5", "18446744073709551616"]:
        env = {"FANDHE_EDGE_PACKAGE": "pkg", "FANDHE_EDGE_IDLE_EXIT_SECS": bad}
        r = _run(tmp_path, env)
        assert r.returncode == 64
        assert json.loads(r.stdout)["code"] == "invalid_input"


def test_req21_missing_package_env_and_missing_binary(tmp_path: Path) -> None:
    """FANDHE_EDGE_PACKAGE 未設定は 64、fandhe-edge 不在は 70（どちらも JSON 1 行）。"""
    r = _run(tmp_path, {})
    assert r.returncode == 64
    assert json.loads(r.stdout)["code"] == "invalid_input"
    r = _run(tmp_path / "nostub", {"FANDHE_EDGE_PACKAGE": "pkg"}, stub=False)
    assert r.returncode == 70
    assert json.loads(r.stdout)["code"] == "runtime_error"
