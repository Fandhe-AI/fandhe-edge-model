"""C3 のテスト用共通フィクスチャ（合成データ・極小設定。実データは使わない）。"""

from __future__ import annotations

from fandhe_edge_trainer.contract import TrainExample, TrainRequest

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


def make_request(
    tmp_path, *, config: dict | None = None, seed: int = 0, out_name: str = "out"
) -> TrainRequest:
    """`TrainRequest` を直接構築する（`guard.safe_join` を経由しないテスト専用の近道。
    経路の閉じ込め検証そのものは `test_contract.py` 側で個別に検証する）。
    """
    return TrainRequest(
        kind="c3",
        kind_version=1,
        config=dict(config if config is not None else TINY_CONFIG),
        label_order=list(LABEL_ORDER),
        max_bytes=64,
        seed=seed,
        device="cpu",
        root=tmp_path,
        train_path=tmp_path / "train.jsonl",
        out_dir=tmp_path / out_name,
    )
