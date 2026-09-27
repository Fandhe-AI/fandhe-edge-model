"""ジョブ全体の資源上限（REQ-39: 学習時間・メモリ）を検査する再利用可能なヘルパー。

`ResourceBudget` は 1 回の学習ジョブ（学習ループ + ONNX 書き出し）につき 1 つ
生成し、学習ループの各バッチ・ONNX 書き出しの前後で `check()` を呼ぶ。
壁時計時間・プロセスの RSS（常駐メモリ量）・（`device="gpu"` の場合のみ）
MLX（Metal）の active memory のいずれかが上限を超えたら `WorkerError`
（`code="limit_exceeded"`・`ExitCode.LIMIT_EXCEEDED`）を送出する。

C3（`kinds/c3.py`）専用の実装にしない: 将来 C1 等を追加する際の学習ループからも
同じ形で使う想定（「拡張点の閉じ方」。資源監視ロジックを種類ごとの学習ループへ
複製しない）。

サンプルステップ数の上限（`check_sample_steps`）・総トークン数の上限
（`check_total_tokens`）・モデルサイズの上限（`check_model_bytes`）はいずれも
学習開始前に 1 回だけ検査する（実測を待たずに拒否できる見積もりベースの防御）。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from .errors import WorkerError
from .exitcode import ExitCode
from .limits import MAX_MODEL_BYTES, MAX_TRAIN_SAMPLE_STEPS, MAX_TRAIN_TOTAL_TOKENS

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


def check_total_tokens(n_examples: int, max_bytes: int) -> None:
    """`examples 件数 × max_bytes`（エンコード後の総トークン数の見積もり）が
    `MAX_TRAIN_TOTAL_TOKENS` を超えないことを、エンコード開始前に検査する（P0-1）。

    エンコード結果を Python のリストのリストとして保持すると、この見積もりを
    経ずに際限なくメモリを確保してしまう（`kinds/c3.py` はこの検査を通過した
    後、あらかじめ確保した numpy 配列へ行ごとに書き込む設計にしている）。
    """
    total = n_examples * max_bytes
    if total > MAX_TRAIN_TOTAL_TOKENS:
        raise WorkerError(
            "limit_exceeded",
            f"estimated total tokens {total} exceeds limit {MAX_TRAIN_TOTAL_TOKENS}"
            " (examples x max_bytes)",
            ExitCode.LIMIT_EXCEEDED,
        )


def check_model_bytes(param_count: int) -> None:
    """モデルパラメータ数から見積もった float32 換算の総バイト数が
    `MAX_MODEL_BYTES` を超えないことを、モデルの実体を作る前に検査する（P0-2）。
    """
    model_bytes = param_count * 4  # float32 = 4 bytes/要素
    if model_bytes > MAX_MODEL_BYTES:
        raise WorkerError(
            "limit_exceeded",
            f"estimated model size {model_bytes} bytes exceeds limit {MAX_MODEL_BYTES} bytes"
            f" ({param_count} params x 4 bytes)",
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

        `device="gpu"` の場合、`rss_bytes` は 2 つの異なる資源（ホスト側の
        プロセス RSS と GPU〔Metal〕の active memory）の上限として同じ数値を
        流用する。両者は別々のメモリ空間だが、本ワーカーでは 1 つの上限値
        （`limits.MAX_TRAIN_RSS_BYTES` 由来）で両方を検査する設計になっている
        （この設計の妥当性は `limits.py` の `MAX_TRAIN_RSS_BYTES` に記載の論点
        としてオーナー確認待ち。Issue #11）。
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
