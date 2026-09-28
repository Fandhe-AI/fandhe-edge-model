"""`kind` ごとの config 既定値と共有 fixture の一致テスト（REQ-18・REQ-19・
REQ-19b・REQ-21・REQ-39。codex 指摘 PR #220 P1「成果物の追加 config 値を
検証せず成功扱いにしている」対応）。

Rust 側（`crates/train/src/kind_defaults.rs`）は
`fixtures/train_contract/kind_defaults.json` をビルド時に埋め込み、成果物の
`artifact.config` が「`kind` の既定値に `request.config` を上書きした実効
config」と完全一致することを検査する。本ファイルは、その fixture の値が
学習ワーカー側の正本（`trainer/src/fandhe_edge_trainer/kinds/<kind>.py` の
`DEFAULT_CONFIG`）と食い違っていないことを機械照合する。`kinds/__init__.py::
_registry` に新しい種類を追加した場合、この fixture へも追記しないと本テスト
が落ちる。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from fandhe_edge_trainer import kinds

# 読み込み前にファイルサイズの上限を確認する（外部入力読み込みの作法。
# coding-python.md）。
_MAX_FIXTURE_BYTES = 1 * 1024 * 1024

# 本ファイルからリポジトリ直下までの階層:
# parents[0]=trainer/tests, parents[1]=trainer, parents[2]=リポジトリ直下。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _REPO_ROOT / "fixtures" / "train_contract" / "kind_defaults.json"

#: fixture 内でメタ情報に使う予約キー（`crates/train/src/kind_defaults.rs`
#: の `RESERVED_META_KEY` と同じ）。`kind` としては解決しない。
_RESERVED_META_KEY = "_meta"


def _load_kind_defaults_fixture() -> dict[str, Any]:
    if not _FIXTURE_PATH.is_file():
        raise FileNotFoundError(f"共有 train_contract fixture が見つからない: {_FIXTURE_PATH}")
    size = _FIXTURE_PATH.stat().st_size
    assert size <= _MAX_FIXTURE_BYTES, (
        f"kind_defaults.json が上限（{_MAX_FIXTURE_BYTES} バイト）を超えている: {size}"
    )
    raw = _FIXTURE_PATH.read_bytes()
    raw.decode("ascii")  # fixture は ASCII のみで書く契約（非 ASCII は \uXXXX）。
    data = json.loads(raw)
    data.pop(_RESERVED_META_KEY, None)
    return data


def test_req19_kind_defaults_fixture_key_set_matches_registered_kinds() -> None:
    """REQ-19b・TASK-19.1: fixture のキー集合が選択口の登録済み `kind`
    （`kinds/__init__.py::_registry`）と完全一致すること。種類の追加・削除を
    fixture へ反映し忘れた場合に検出する。
    """
    fixture = _load_kind_defaults_fixture()
    registry = kinds._registry()
    assert set(fixture) == set(registry), (
        "kind_defaults.json のキー集合が選択口の登録済み kind と一致しない: "
        f"fixture={sorted(fixture)} registry={sorted(registry)}"
    )


def test_req19_kind_defaults_fixture_matches_each_kind_default_config() -> None:
    """REQ-18・REQ-19・REQ-21・REQ-39: 登録済みの各 `kind` について、
    `kinds/<kind>.py::DEFAULT_CONFIG` が fixture の値と完全一致すること
    （キーの過不足・値の型（int/float）の相違を含む）。

    Python の `==` は `1 == 1.0` が真になり int/float の型差を見逃す
    （Rust 側 `serde_json::Value` の等価は `PosInt(1)`／`Float(1.0)` を区別
    するため、この差を見逃すと Rust 側の完全一致検査だけが実際のワーカー
    出力〔`1.0`〕を誤って拒否する事故に気づけない）。`json.dumps` した文字列
    同士を比較し、`1` と `1.0` の表記差が残るようにする（辞書のキー順は
    `sort_keys=True` で揃え、順序差を無視する）。
    """
    fixture = _load_kind_defaults_fixture()
    registry = kinds._registry()

    for kind_name, versions in registry.items():
        # `kind` 名とモジュール名が一致する契約（`kinds/__init__.py`
        # モジュール doc「新しい種類を追加する手順」）に依存せず、実装
        # クラスの `__module__` から辿る（将来 kind 名とファイル名が
        # 分離しても壊れない）。
        impl_cls = next(iter(versions.values()))
        module = importlib.import_module(impl_cls.__module__)
        default_config = getattr(module, "DEFAULT_CONFIG", None)
        assert default_config is not None, f"{impl_cls.__module__} に DEFAULT_CONFIG が無い"
        fixture_repr = json.dumps(fixture[kind_name], sort_keys=True)
        actual_repr = json.dumps(default_config, sort_keys=True)
        assert fixture_repr == actual_repr, (
            f"kind_defaults.json の {kind_name!r} が "
            f"{impl_cls.__module__}::DEFAULT_CONFIG と一致しない"
            "（int/float の型差を含む可能性がある）: "
            f"fixture={fixture_repr} actual={actual_repr}"
        )
