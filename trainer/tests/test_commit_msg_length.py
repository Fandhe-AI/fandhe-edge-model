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


def test_diff_after_cut_line_ignored():
    cut = "# ------------------------ >8 ------------------------"
    assert mod.check("ci: x\n\n" + cut + "\n+" + "a" * 200) == []


def test_url_line_ok_but_long_plain_line_ng():
    assert mod.check("ci: x\n\nsee https://example.com/" + "a" * 200) == []
    assert mod.check("ci: x\n\n" + "a" * 101) == [(3, 101)]


def test_header_url_not_exempt():
    assert mod.check("ci: https://example.com/" + "a" * 100) == [(1, 124)]
