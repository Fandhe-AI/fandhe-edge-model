"""事前分割の判定表・golden が古くなっていないことの検査（REQ-41・TASK-41.4-3・#411）。"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import gen_pretokenize_golden as gen


def test_unicode_tables_are_up_to_date() -> None:
    """REQ-41: コミット済みの判定表が生成器の出力と完全一致する。"""
    assert gen.TABLES_PATH.read_text("utf-8") == gen.render_tables()


def test_pretokenize_golden_is_up_to_date() -> None:
    """REQ-41: コミット済みの golden が参照実装の出力と完全一致する。"""
    assert gen.GOLDEN_PATH.read_text("utf-8") == gen.render_golden()


def test_unicode_version_is_15() -> None:
    """REQ-41: Rust の unicode-normalization（15.0）と表の Unicode 版を揃える。"""
    assert unicodedata.unidata_version == "15.0.0"
