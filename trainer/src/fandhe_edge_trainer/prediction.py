"""学習ジョブ内での validation 採点（REQ-18・REQ-27）の予測レコードの形。

学習ワーカーは学習直後に同じプロセス内で、メモリ上の学習済みモデルへ validation
入力（`(id, input)` のみ。正解ラベルは受け取らない）を通し、結果の
`validation_predictions`（`{id, status, predicted_label}` の列）として返す
（`crates/train/src/result.rs::ValidationPrediction`。`scores` は含めない。
issue #84 PR #238・選択肢 2）。本モジュールは、その予測レコードの組み立てだけを
持つ軽量な共有部品（mlx 等の重い依存を読み込まない）。kind ごとの予測処理は
`kinds/<kind>.py`、kind の振り分けは `predict.py` にある。
"""

from __future__ import annotations

from typing import Any

#: 予測レコードのキー（順序込み。Rust 側の直列化順・共有 fixture と揃える）。
PREDICTION_FIELDS = ("id", "status", "predicted_label")


def ok_prediction_record(record_id: str, label: str) -> dict[str, Any]:
    """予測に成功した 1 件（`status:"ok"`）のレコードを返す。"""
    return {"id": record_id, "status": "ok", "predicted_label": label}
