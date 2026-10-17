# qwen2_parity fixture の出典と生成手順

REQ-41（公開モデルの追加学習）・REQ-32（学習依存の無い推論）・REQ-28（単体・バッチの一致）・
TASK-41.6-5（#434）の結合テスト用 fixture。Rust 推論ランタイム（`crates/runtime`）の
`tests/qwen2_parity.rs`・`tests/env_isolation.rs` が読む。

## 内容

学習ワーカーの CLI（`launch.py train`）が実際に書く自己完結パッケージそのもの（`kind=qwen2_lora`）。

| ファイル | 内容 |
| -------- | ---- |
| `artifact.json` | パッケージのメタデータ（`scoring`・`tokenizer_*`・`external_data[]` を含む） |
| `model.onnx` | LoRA 統合済みの採点グラフ（opset 13。重みは外部データを参照） |
| `weights-00000.bin` | 外部データ shard（bf16 の重み） |
| `tokenizer.json` | 合成ベースモデルの tokenizer |
| `cases.json` | 各ケースの入力・学習側（MLX 内推論）の予測ラベル・ReferenceEvaluator の確率・上位 2 つの確率差・生成設定 |

## 出典とライセンス

- ベースモデルは `trainer/tools/poc26/synthetic.py` の極小 config（hidden 16・2 層・vocab 300）と、
  seed 固定の乱数重み。学習データ・評価入力も生成器が固定 seed で作る合成データで、外部データ・
  LLM 生成物・実データ・事前学習済み重みを含まない（ライセンス確認の対象なし）
- 重みは上記の合成から本リポの学習ワーカーが書き出したもの（`MIT OR Apache-2.0`）

## 生成手順と環境

```bash
uv --no-config run --locked --directory trainer python tools/gen_qwen2_parity_fixture.py --out ../fixtures/qwen2_parity --overwrite
```

- 生成環境: macOS arm64・CPU（`device="cpu"`）・mlx 0.32.2・onnx 1.23.0・numpy 2.5.3
- seed: 学習データ生成 20261017（`data_seed`）、検証入力生成 20261018（`input_seed`。20261017+1）、学習リクエスト 0（`train_seed`）、ベース重み 11（標準偏差 1.0）。ラベルは 3 文字 8 種
- 設定: `lora_rank=4`・`lora_num_layers=2`・`iters=20`・`lr=0.01`・`dtype=float32`
- 通信なし（REQ-38）

## 設計上の注意

- 極小モデルは学習ではほとんど入力に反応しないため（全件が同じラベルになる）、予測ラベルが入力で
  変わる（2 種類が十分な件数で現れる）ベース重みの seed を探索して固定した。ラベルの正しさ
  （学習の質）は検証対象ではなく、学習側と推論側の一致が検証対象
- 統合重みは bf16 に丸められるため、f32 の MLX と順位が入れ替わりうる。生成器は ReferenceEvaluator の
  argmax が MLX のラベルと異なる、または上位 2 つの確率差が `NEAR_TIE_MARGIN`（1e-3）未満のケースを
  除外して件数を `cases.json` に記録する（現在の除外は 0 件）
- 生成器は、記録するケースが 100 件以上・ラベルが 2 種類以上であることを検査し、満たさなければ失敗する

## 証拠種別と既知の制限

- 証拠種別: テストハーネス（合成の極小モデル・CPU）。実重み（Qwen2.5 系）での照合ではなく、実機の
  容量・RSS・p95 の測定でもない。実重みは TASK-41.11（#442）の実機作業
- Rust 側は、確率が ReferenceEvaluator と 1e-5 以内、ラベルが MLX と全件一致、単体・バッチが同一で
  あることを確かめる。再学習してバイト一致を確かめるテストは置かない（MLX CPU の浮動小数の結果が
  OS で一致する保証がないため。`trainer/tests/test_qwen2_parity_fixture.py` は記録済み値との整合のみ）
