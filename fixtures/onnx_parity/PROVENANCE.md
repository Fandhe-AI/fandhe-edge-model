# onnx_parity fixture の出典と生成手順

REQ-32（学習依存の無い推論ランタイムの予測ラベルが、学習フレームワーク内の推論と一致すること）・
TASK-32.1-2（#113）の結合テスト用 fixture。Rust 推論ランタイム（`crates/runtime`）の
`tests/onnx_parity.rs` が読む。

## 内容

| ファイル | 内容 |
| -------- | ---- |
| `c1.onnx` | 極小設定で学習した C1（バイト n-gram TF-IDF + ロジスティック回帰）の書き出し |
| `c3.onnx` | 極小設定で学習した C3（バイト CNN）の書き出し |
| `cases.json` | kind ごとの `max_bytes`・`label_order`・`onnx_sha256`・生成設定・各ケースの MLX 内推論の予測ラベル・確率・上位 2 つの確率差 |

## 出典とライセンス

- 学習データ・評価入力はすべて生成器（`trainer/tools/gen_onnx_parity_fixture.py`）が固定 seed で作る
  合成データで、外部データ・LLM 生成物・実データを含まない。評価入力の一部は共有ゴールデンベクタ
  `fixtures/preprocess/byte_encoding_vectors.json` の入力（本リポ内の手動導出値）
- 事前学習済み重み・外部データセットは使っていない（ライセンス確認の対象なし）。ONNX の重みは
  上記の合成データから本リポの学習ワーカーが学習したもの（`MIT OR Apache-2.0`）

## 生成手順と環境

```bash
uv run --locked --directory trainer python tools/gen_onnx_parity_fixture.py --out ../fixtures/onnx_parity --overwrite
```

- 生成環境: Linux x86_64・CPU（`device="cpu"`）・mlx 0.32.2・onnx 1.23.0・numpy 2.5.3
- seed: 20260929（学習・入力生成とも）。`max_bytes` は 48
- 設定: C1 は `ngram 1..3`・`min_df=1`・`epochs=40`、C3 は `emb=8`・`filters=8`・`widths=[3,5,7]`・`epochs=5`
- 通信なし（REQ-38）

## 証拠種別と既知の制限

- 証拠種別: テストハーネス（PoC-14 の実測を踏襲した結合テスト）。実機計測ではない
- 生成器は、MLX と `onnx.reference.ReferenceEvaluator` の確率差が 1e-5 以内（`ATOL_MLX_ONNX`）であること、
  予測ラベルが 2 種類以上現れることを検査し、満たさなければ失敗する
- 上位 2 つの確率差が 1e-3 未満のケース（僅差。f32 の演算順序差で argmax が割れうる）は記録から
  除外し、`cases.json` の `excluded_near_tie` に件数を記録する（C1 は 1 件・C3 は 0 件）
- **再学習してバイト一致を確かめるテストは置かない**。python-ci は macOS arm64、開発機は Linux x86_64 で、
  MLX CPU の浮動小数の結果が一致する保証がないため。コミット済みファイルの自己整合性
  （`trainer/tests/test_onnx_parity_fixture.py`）と、Rust 側の全件一致のみを検査する
