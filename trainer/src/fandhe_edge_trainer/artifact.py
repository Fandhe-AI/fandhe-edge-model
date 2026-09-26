"""成果物（モデルパッケージ）の書き出し。

Rust 側 `Artifact` 構造体（PoC-16。`docs/spec/03-poc/core-cli-vertical-slice/core/src/
artifact.rs`）とフィールド名・型を揃える。本ワーカーは C3 の学習・ONNX 書き出しまでを
担い、複数候補（C1・C3 等）からの選定（TASK-18.x）は行わないため、`candidate_label`
には学習した種類の名前（例: "c3"）を入れる（将来 TASK-18.x 側の選定処理が複数候補の
成果物を比較する際、この値を選定結果で上書きする）。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: 選択口（本ワーカー）のバージョン。artifact.json の再現性の記録に使う。
SELECTOR_VERSION = "0.1"

ONNX_FILE_NAME = "model.onnx"
ARTIFACT_FILE_NAME = "artifact.json"


def now_utc() -> str:
    """`%Y-%m-%dT%H:%M:%SZ` 形式の現在時刻（UTC）。"""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_artifact(
    *,
    kind: str,
    kind_version: int,
    config: dict[str, Any],
    label_order: list[str],
    output_type: str,
    max_bytes: int,
    candidate_label: str,
) -> dict[str, Any]:
    """Rust 側 `Artifact` 構造体と同じフィールドを持つ辞書を作る。"""
    return {
        "kind": kind,
        "kind_version": kind_version,
        "selector_version": SELECTOR_VERSION,
        "config": config,
        "label_order": label_order,
        "output_type": output_type,
        "max_bytes": max_bytes,
        "onnx_file": ONNX_FILE_NAME,
        "created_utc": now_utc(),
        "candidate_label": candidate_label,
    }


def write_artifact(pkg_dir: Path, artifact: dict[str, Any]) -> None:
    """`artifact.json` を書き出す（改行は LF 固定。`json.dumps` の既定どおり）。"""
    pkg_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(artifact, ensure_ascii=False, indent=2)
    (pkg_dir / ARTIFACT_FILE_NAME).write_text(text + "\n", encoding="utf-8")
