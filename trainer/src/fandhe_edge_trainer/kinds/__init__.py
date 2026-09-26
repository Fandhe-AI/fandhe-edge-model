"""モデルの種類の選択口（REQ-19b・TASK-19.1）。

`kind`（判別型・生成型などモデルの種類を示す識別子）→ 実装 の対応づけを
1 箇所の許可リスト（`_REGISTRY`）に閉じる。学習・推論の両方で同じ選択口を使う
という設計（REQ-19b）のうち、本パッケージは学習側（候補の学習・書き出し）だけを
扱う。推論側の選択口は推論ランタイム（runtime-builder 担当。REQ-32 によりここへは
依存しない）。

新しい種類を追加する手順（拡張点。C1 は別 PR）:
1. `kinds/<name>.py` に `Kind` プロトコル（`train`・`export_onnx`・既定 `config`）を実装する
2. `_REGISTRY` へ `{"<name>": {<kind_version>: <クラス>}}` を追記する
既存の種類の実装・共通コア・評価器の改変を要しない。
"""

from __future__ import annotations

from typing import IO, Any, Protocol

from ..contract import TrainExample, TrainRequest
from ..errors import WorkerError
from ..exitcode import ExitCode


class TrainedModel(Protocol):
    """種類ごとの学習結果を表すプロトコル。

    `cli.py::run_train` が種類（kind）に依存せず読む共通フィールドをここで宣言する
    （具体的な学習結果の内部表現は種類ごとに異なってよいが、この 3 つは
    `artifact.json` の書き出しに必須。将来 C1 等を追加する際も同じ形にする）。
    """

    config: dict[str, Any]
    label_order: list[str]
    max_bytes: int


class Kind(Protocol):
    """モデルの種類 1 つ分の実装契約（学習・ONNX 書き出し）。"""

    def train(self, examples: list[TrainExample], request: TrainRequest) -> TrainedModel:
        """train データだけを使って学習する（validation・test は読まない）。"""
        ...

    def export_onnx(self, trained: TrainedModel, out: IO[bytes]) -> None:
        """学習結果を ONNX として `out`（呼び出し元が開いた書き込み用バイナリファイル
        オブジェクト）へ書き出す。**経路は一切扱わない**: どこへ書き出すか
        （root 配下への閉じ込め・アトミックな確定）は呼び出し元（`cli.py`・
        `contract.py`）の責務であり、本メソッドは `out.write(...)` するだけに
        留める（TOCTOU を避けるための dir_fd ベースの経路解決を、種類ごとの
        実装へ複製しない。推論ランタイムとの ONNX 形式の契約は runtime-builder
        と共有する）。
        """
        ...


def _registry() -> dict[str, dict[int, type]]:
    # 遅延 import: 種類ごとに必要な重い依存（mlx 等）を、実際に使う種類だけ読み込む。
    from .c3 import C3Kind

    return {"c3": {1: C3Kind}}


def resolve_kind(kind: str, kind_version: int) -> Kind:
    """`kind`・`kind_version` から実装を選ぶ（TASK-19.1）。未対応は TASK-19.3 のエラーを返す。"""
    registry = _registry()
    versions = registry.get(kind)
    if versions is None:
        raise WorkerError(
            "unsupported_kind",
            f"unsupported kind: {kind!r} (supported: {sorted(registry)})",
            ExitCode.INVALID_INPUT,
        )
    impl = versions.get(kind_version)
    if impl is None:
        raise WorkerError(
            "unsupported_kind_version",
            f"unsupported kind_version {kind_version!r} for kind {kind!r} "
            f"(supported: {sorted(versions)})",
            ExitCode.INVALID_INPUT,
        )
    return impl()
