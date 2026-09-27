"""終了コード（REQ-21・TASK-21.1・#179）の Rust ⇔ 学習ワーカー一致テスト。

単一真実源は Rust 側（`crates/core/src/exitcode.rs` の `ExitCode`）。学習ワー
カー側ミラー（`fandhe_edge_trainer.exitcode.ExitCode`）の値が Rust 側と食い
違わないことを、`crates/core/tests/exitcode_fixture.rs` と共有する
`fixtures/exitcode/exit_codes.json` を介して照合する。fixture のパスが
Rust／trainer 間の唯一の結合点であり、このパスを変えると両テストの結合が
切れる。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fandhe_edge_trainer.exitcode import ExitCode

# 読み込み前にファイルサイズの上限を確認する（外部入力読み込みの作法。coding-python.md）。
# ローカル固定 fixture だが、想定外に巨大化した場合に無制限アロケーションへ繋げない。
_MAX_FIXTURE_BYTES = 1024 * 1024

# 本ファイル（trainer/tests/test_exitcode.py）からリポジトリ直下までの階層:
# parents[0]=trainer/tests, parents[1]=trainer, parents[2]=リポジトリ直下。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _REPO_ROOT / "fixtures" / "exitcode" / "exit_codes.json"


def _load_fixture() -> list[dict[str, Any]]:
    if not _FIXTURE_PATH.is_file():
        raise FileNotFoundError(
            "共有 exit code fixture が見つからない: "
            f"{_FIXTURE_PATH}（リポジトリ直下 {_REPO_ROOT} 起点で "
            "fixtures/exitcode/exit_codes.json を解決できるか確認する）"
        )
    size = _FIXTURE_PATH.stat().st_size
    assert size <= _MAX_FIXTURE_BYTES, (
        f"exit_codes.json が上限（{_MAX_FIXTURE_BYTES} バイト）を超えている: {size}"
    )
    raw = _FIXTURE_PATH.read_bytes()
    # fixture は Rust 側実装との照合も見据え ASCII のみで書く（非 ASCII は \uXXXX）。
    raw.decode("ascii")
    data = json.loads(raw)
    exit_codes: list[dict[str, Any]] = data["exit_codes"]
    return exit_codes


_ENTRIES = _load_fixture()


def test_req21_fixture_has_seven_entries() -> None:
    assert len(_ENTRIES) == 7
    assert len(_ENTRIES) == len(ExitCode)
    names = [e["name"] for e in _ENTRIES]
    codes = [e["code"] for e in _ENTRIES]
    assert len(set(names)) == len(names), "fixture has duplicate names"
    assert len(set(codes)) == len(codes), "fixture has duplicate codes"


def test_req21_exitcode_matches_fixture() -> None:
    # 大文字メンバー名（IntEnum）と fixture の snake_case 名を橋渡しする。
    from_python = {member.name.lower(): int(member.value) for member in ExitCode}
    from_fixture = {e["name"]: e["code"] for e in _ENTRIES}
    assert from_python == from_fixture


def test_req21_exitcode_values_are_fixed() -> None:
    # fixture 自体の改変（値の緩和）も検出できるよう、具体値を個別に固定する。
    assert ExitCode.OK == 0
    assert ExitCode.JUDGED_FAIL == 10
    assert ExitCode.OUT_OF_SCOPE == 11
    assert ExitCode.PENDING == 12
    assert ExitCode.LIMIT_EXCEEDED == 20
    assert ExitCode.INVALID_INPUT == 64
    assert ExitCode.RUNTIME_ERROR == 70
