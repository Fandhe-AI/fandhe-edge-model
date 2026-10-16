"""追加学習の種類 `qwen2_lora`（ベースモデル Qwen2 系＋LoRA）の骨格（REQ-41・TASK-41.5-1・#424）。

学習ワーカーの選択口（`kinds/__init__.py::_registry`）に種類名を載せるためだけの骨格で、
学習・ONNX 書き出しは未実装。呼ばれたら `runtime_error`（終了コード 70）を返し、
実装済みを装わない。ガード層の `kind`・`kind_version` の許可リストと同じ名前を持つ
（共有 fixture `fixtures/train_contract/kind_versions.json`・`kind_defaults.json` で同期）。

後続: ベースモデルの読み込み #425、LoRA 学習 #426、選択肢の採点と予測 #427、
ONNX 書き出し #431・#432。出力は選択肢の採点だけで、自由生成はしない（REQ-19b の
対応づけ (b) と同じ考え方）。既定候補には入れず、利用者が明示したときだけ使う（REQ-41）。
mlx は import しない（遅延 import の方針を保つ）。
"""

from __future__ import annotations

from typing import IO, Any

from .. import budget
from ..contract import TrainExample, TrainRequest
from ..errors import WorkerError
from ..exitcode import ExitCode

# PoC-26 追補 1・追補 2 の固定値と P15 の代表構成（iters 300・lr 1e-4）。#426 で見直す場合は
# fixtures/train_contract/kind_defaults.json も同時に更新する。
DEFAULT_CONFIG: dict[str, Any] = {
    "lora_rank": 8,
    "lora_scale": 20.0,
    "lora_dropout": 0.0,
    "lora_num_layers": 8,
    "lr": 0.0001,
    "iters": 300,
    "batch_size": 4,
    "max_seq_length": 512,
}

_NOT_IMPLEMENTED = "kind 'qwen2_lora' is registered but training is not implemented yet"


class Qwen2LoraKind:
    """`Kind` プロトコルの骨格実装。データ・経路・ベースモデルには一切触れない。"""

    def train(
        self,
        examples: list[TrainExample],
        request: TrainRequest,
        resource_budget: budget.ResourceBudget,
    ) -> Any:
        """未実装（#426）。常に `runtime_error` を送出する。"""
        raise WorkerError("runtime_error", _NOT_IMPLEMENTED, ExitCode.RUNTIME_ERROR)

    def export_onnx(self, trained: Any, out: IO[bytes]) -> None:
        """未実装（#431・#432）。常に `runtime_error` を送出する。"""
        raise WorkerError("runtime_error", _NOT_IMPLEMENTED, ExitCode.RUNTIME_ERROR)
