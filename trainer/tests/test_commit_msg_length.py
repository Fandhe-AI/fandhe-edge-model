"""commit-msg フックの行長検査（scripts/check_commit_msg_length.py）。commitlint と同基準。"""

import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[2] / "scripts" / "check_commit_msg_length.py"
_spec = importlib.util.spec_from_file_location("check_commit_msg_length", _p)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_japanese_100_ok_101_ng():
    assert mod.check("ci: " + "あ" * 96) == []
    assert mod.check("ci: " + "あ" * 97) == [(1, 101)]


def test_comment_lines_ignored():
    assert mod.check("ci: x\n\n# " + "あ" * 200) == []


def test_astral_counts_two():
    assert mod.check("x" * 98 + "😀") == []
    assert mod.check("x" * 99 + "😀") == [(1, 101)]


def test_line_number_reported():
    assert mod.check("ci: x\n\n" + "a" * 101) == [(3, 101)]
