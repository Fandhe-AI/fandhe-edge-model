"""C3 のテスト用共通フィクスチャ（合成データ・極小設定。実データは使わない）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from fandhe_edge_trainer import guard
from fandhe_edge_trainer.contract import TrainExample, TrainRequest
from fandhe_edge_trainer.limits import MAX_TRAIN_RSS_BYTES, MAX_TRAIN_WALL_SECONDS

#: テストのみに使う合成データセット（自明に分離可能な 2 クラス）。実データではない。
LABEL_ORDER = ["cat_a", "cat_b"]
_TEXT_A = "alpha alpha beta gamma"
_TEXT_B = "delta delta epsilon zeta"


def make_examples(n_per_label: int = 12) -> list[TrainExample]:
    examples = []
    for i in range(n_per_label):
        examples.append(TrainExample(input=f"{_TEXT_A} {i}", label="cat_a"))
        examples.append(TrainExample(input=f"{_TEXT_B} {i}", label="cat_b"))
    return examples


#: float32 の丸め誤差に由来する許容差を 1 箇所へ集約する（evaluation-contract.md の
#: 1e-9 はハッシュ等の決定性判定向けの基準であり、ここで扱う ONNX Conv/ReduceMax・
#: MLX の演算順序差による float32 丸め誤差とは性質が異なるため、その基準となる
#: float32 の実務的な許容差を使う。既存の許容差を緩めたものではない）。
#:
#: - ATOL_BATCH_PARITY: 同一 ONNX モデルを単独 vs バッチで実行した際の差（REQ-28。
#:   `tests/test_c3_batch_parity.py`）。同一グラフ・同一浮動小数演算のはずで最も厳しい。
#: - ATOL_MLX_ONNX: MLX（学習時フォワード）と ONNX Reference Evaluator という
#:   異なる実装間の差（`tests/test_c3_mlx_onnx_parity.py`）。演算の実装・順序が
#:   異なるため、より緩い許容差を使う。
ATOL_BATCH_PARITY = 1e-6
ATOL_MLX_ONNX = 1e-5

#: ReferenceEvaluator（純 numpy）でも高速に回る極小アーキテクチャ設定。
TINY_CONFIG = {
    "epochs": 5,
    "batch_size": 8,
    "emb": 8,
    "filters": 8,
    "widths": [3, 5, 7],
    "dropout": 0.0,
}

#: `make_request` が作った `TrainRequest`（fd を保持する）を集め、テストごとに
#: 自動で閉じる（`_close_confined_requests`）。fd リークで `EMFILE` に達するのを防ぐ。
_created_requests: list[TrainRequest] = []


def make_request(
    tmp_path,
    *,
    config: dict | None = None,
    seed: int = 0,
    out_name: str = "out",
    max_bytes: int = 64,
    time_limit_seconds: int = MAX_TRAIN_WALL_SECONDS,
    rss_limit_bytes: int = MAX_TRAIN_RSS_BYTES,
) -> TrainRequest:
    """`TrainRequest` を、`guard.resolve_root`/`guard.confine` を実際に経由して
    構築する（`root`＝`tmp_path`。fd の解放はテスト終了時に自動で行われる。
    `_close_confined_requests` 参照）。経路の閉じ込め違反そのものの検証は
    `test_contract.py` 側で個別に行う。
    """
    Path(tmp_path).mkdir(parents=True, exist_ok=True)  # root は存在するディレクトリが前提
    root_handle = guard.resolve_root(str(tmp_path))
    train_path_entry = guard.confine(root_handle, "train.jsonl", "train_path")
    out_dir_entry = guard.confine(root_handle, out_name, "out_dir")
    root_handle.close()  # train_path_entry/out_dir_entry は独立した fd を持つため不要になる
    req = TrainRequest(
        kind="c3",
        kind_version=1,
        config=dict(config if config is not None else TINY_CONFIG),
        label_order=list(LABEL_ORDER),
        max_bytes=max_bytes,
        seed=seed,
        device="cpu",
        root=root_handle,
        train_path=train_path_entry,
        out_dir=out_dir_entry,
        time_limit_seconds=time_limit_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    _created_requests.append(req)
    return req


def export_onnx_to_path(kind, trained, path) -> None:
    """`Kind.export_onnx`（`IO[bytes]` を受け取る契約）をテストの `Path` 引数から
    呼び出す小さなヘルパー（`kinds/__init__.py::Kind.export_onnx` 参照）。
    """
    with open(path, "wb") as f:
        kind.export_onnx(trained, f)


@pytest.fixture(autouse=True)
def _close_confined_requests():
    """`make_request` が作った `TrainRequest` の fd を、テストごとに解放する。"""
    yield
    while _created_requests:
        _created_requests.pop().close_resources()
