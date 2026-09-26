"""ジョブ全体の資源上限（REQ-39: 学習時間・メモリ）を検査する再利用可能なヘルパー。

`ResourceBudget` は 1 回の学習ジョブ（学習ループ + ONNX 書き出し）につき 1 つ
生成し、学習ループの各バッチ・ONNX 書き出しの前後で `check()` を呼ぶ。
壁時計時間・プロセスの RSS（常駐メモリ量）・（`device="gpu"` の場合のみ）
MLX（Metal）の active memory のいずれかが上限を超えたら `WorkerError`
（`code="limit_exceeded"`・`ExitCode.LIMIT_EXCEEDED`）を送出する。

C3（`kinds/c3.py`）専用の実装にしない: 将来 C1 等を追加する際の学習ループからも
同じ形で使う想定（「拡張点の閉じ方」。資源監視ロジックを種類ごとの学習ループへ
複製しない）。

サンプルステップ数の上限（`check_sample_steps`）は学習開始前に 1 回だけ検査する
（`examples 件数 × epochs` の見積もりで、実測を待たずに拒否できる）。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from .errors import WorkerError
from .exitcode import ExitCode
from .limits import MAX_TRAIN_SAMPLE_STEPS

try:
    import resource
except ImportError:  # pragma: no cover - resource は POSIX 限定（Windows では None）
    resource = None  # type: ignore[assignment]


def check_sample_steps(n_examples: int, epochs: int) -> None:
    """`examples 件数 × epochs`（学習ループが回す総ステップ数の見積もり）が
    `MAX_TRAIN_SAMPLE_STEPS` を超えないことを、学習開始前に検査する。
    """
    steps = n_examples * epochs
    if steps > MAX_TRAIN_SAMPLE_STEPS:
        raise WorkerError(
            "limit_exceeded",
            f"estimated sample steps {steps} exceeds limit {MAX_TRAIN_SAMPLE_STEPS}"
            " (examples x epochs)",
            ExitCode.LIMIT_EXCEEDED,
        )


def _current_rss_bytes() -> int:
    """現在のプロセスの（ピーク）常駐メモリ量を bytes で返す。

    標準ライブラリには「現在の RSS」を直接取る移植性の高い手段が無いため、
    `resource.getrusage(RUSAGE_SELF).ru_maxrss`（プロセス開始からのピーク RSS。
    現在値ではなく単調非減少の最大値）で代用する。ピークで判定しても
    「一度でも上限を超えたら不合格」という判定としては妥当（ピークは現在値
    以上なので、ピークで弾かなければ現在値でも弾かれない）。

    単位はプラットフォームで異なる（macOS: bytes、Linux: KiB）ため
    `sys.platform` で切り替える（実測: 本開発機の macOS で `ru_maxrss` が
    妥当な bytes 値であることを確認済み。Linux 側は man page の記載に基づく）。
    `resource` が無い環境（Windows 等）では 0 を返し、事実上 RSS 検査を
    無効化する（wall-clock・サンプルステップ数の他の防御と併用する前提）。
    """
    if resource is None:  # pragma: no cover - Windows 等
        return 0
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(ru_maxrss)
    return int(ru_maxrss) * 1024


def _mlx_active_memory_bytes() -> int | None:
    """MLX（Metal・GPU）の active memory を bytes で返す。

    GPU 上の確保量はホスト側の RSS には現れないため、`device="gpu"` の場合は
    別途これも監視する。`mx.get_active_memory()`（mlx==0.32.2 で存在を確認済み）
    を使う。mlx 自体が読み込めない・関数が無い場合は None を返し、呼び出し側は
    その項目の検査をスキップする（wall-clock・RSS の他の防御と併用する前提）。
    """
    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover - mlx 未導入環境
        return None
    get_active_memory = getattr(mx, "get_active_memory", None)
    if get_active_memory is None:  # pragma: no cover - 将来の mlx 版で API が変わる場合
        return None
    return int(get_active_memory())


@dataclass
class ResourceBudget:
    """1 回の学習ジョブの資源上限。生成時刻を起点に壁時計の締切りを計算する。"""

    wall_seconds: float
    rss_bytes: int
    device: str
    _deadline: float = field(init=False)

    def __post_init__(self) -> None:
        self._deadline = time.monotonic() + self.wall_seconds

    def check(self) -> None:
        """壁時計・RSS・（device="gpu" なら）MLX active memory を検査する。

        いずれかが上限を超えていれば `WorkerError`（limit_exceeded・exit 20）
        を送出する。学習ループの各バッチ・ONNX 書き出しの前後で呼ぶ想定。
        """
        if time.monotonic() > self._deadline:
            raise WorkerError(
                "limit_exceeded",
                f"training exceeded wall-clock budget of {self.wall_seconds} seconds",
                ExitCode.LIMIT_EXCEEDED,
            )
        rss = _current_rss_bytes()
        if rss > self.rss_bytes:
            raise WorkerError(
                "limit_exceeded",
                f"training exceeded RSS budget of {self.rss_bytes} bytes",
                ExitCode.LIMIT_EXCEEDED,
            )
        if self.device == "gpu":
            active = _mlx_active_memory_bytes()
            if active is not None and active > self.rss_bytes:
                raise WorkerError(
                    "limit_exceeded",
                    f"training exceeded MLX active-memory budget of {self.rss_bytes} bytes",
                    ExitCode.LIMIT_EXCEEDED,
                )
