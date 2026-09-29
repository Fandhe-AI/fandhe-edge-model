"""学習直後の validation 予測の振り分け（REQ-18・REQ-19・REQ-27）。

`cli.py::run_worker_train` が、リクエストに `validation_inputs` がある場合だけ
学習直後（同じプロセス内・メモリ上の `trained` を使う）に呼ぶ単一の入口。
kind ごとの予測処理を 1 箇所で振り分け、kind の追加時に変更箇所を本モジュール
だけに閉じる（`kinds/__init__.py::Kind` プロトコルには `predict` を含めない。
オーナー判断で振り分け関数を 1 つにまとめる方針。issue #84 PR #238・選択肢 2）。

- `c1`・`c3`: 学習時の順伝播（`TfidfLogReg.__call__`・`ByteCNN.__call__`）の
  argmax（`kinds/c1.py::predict_labels`・`kinds/c3.py::predict_labels`）
- `autoregressive`: 既存の `predict_records`（対応づけ (b)）から `scores` を除く。
  対応づけに失敗したレコードは `status:"error"`（判定不能を含む。REQ-19b・
  TASK-19b.2）とし、全件を消費した後に理由別の件数だけを stderr へ 1 行報告する
  （id・入力本文は出さない。c1・c3 は分類器の argmax で対応づけ失敗が起きない）

**推論関数へ渡すのは `(id, input)` だけ**（REQ-27。正解ラベル・分割情報を
受け取らない）。予測時間もジョブ全体の `resource_budget`（学習の壁時計上限）の
対象になる（Rust 側 `run_train` の締め切りが予測も覆う）。
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from . import budget as budget_mod
from .errors import WorkerError, truncate_for_message
from .exitcode import ExitCode
from .prediction import PREDICTION_FIELDS

#: 学習直後の validation 予測に対応している kind。新しい kind を選択口へ足したら、
#: 予測経路を実装して `predict_validation` の分岐とここへ追加する。
_SUPPORTED_KINDS = frozenset({"c1", "c3", "autoregressive"})


def require_validation_prediction_support(kind: str) -> None:
    """`kind` が学習直後の validation 予測に対応していなければ、学習を始める前に
    `invalid_request`（exit 64）で拒否する（学習時間を無駄にしない。fail-closed）。
    """
    if kind not in _SUPPORTED_KINDS:
        raise WorkerError(
            "invalid_request",
            f"kind {truncate_for_message(kind)!r} does not support validation predictions"
            f" (supported: {sorted(_SUPPORTED_KINDS)})",
            ExitCode.INVALID_INPUT,
        )


def predict_validation(
    kind: str,
    trained: Any,
    rows: Sequence[tuple[str, str]],
    resource_budget: budget_mod.ResourceBudget,
) -> list[dict[str, Any]]:
    """学習済みモデルで `rows`（`(id, input)` の列）を予測し、`{id, status,
    predicted_label}` のレコード列を返す。入力の順序・件数を保つ。
    """
    require_validation_prediction_support(kind)
    if kind == "autoregressive":
        from .kinds.autoregressive import ChoiceMappingTally, predict_records

        tally = ChoiceMappingTally()
        records = [
            {key: record[key] for key in PREDICTION_FIELDS}
            for record in predict_records(
                trained, rows, resource_budget=resource_budget, tally=tally
            )
        ]
        print(tally.to_log_line(), file=sys.stderr)
        return records
    if kind == "c1":
        from .kinds.c1 import predict_labels as predict_c1

        return predict_c1(trained, rows, resource_budget)
    from .kinds.c3 import predict_labels as predict_c3

    return predict_c3(trained, rows, resource_budget)
