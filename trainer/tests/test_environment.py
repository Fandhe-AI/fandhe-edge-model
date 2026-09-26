"""学習ワーカーの実行環境（Python 版）の固定を検証するテスト。"""

import sys


def test_python_minor_version_is_pinned_3_12() -> None:
    """`.python-version`（3.12）・`pyproject.toml` の `requires-python`
    （`==3.12.*`）が指すマイナーバージョンと実行環境が一致することを保証する。

    学習ワーカーは Python の版を固定して決定性を確保する方針
    （.claude/rules/coding-python.md）のため、意図しない Python バージョンで
    テストが走った場合に検出できるようにする。
    """
    assert sys.version_info[:2] == (3, 12)
