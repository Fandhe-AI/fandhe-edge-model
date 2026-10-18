# qwen2_eval_contract fixture の出典と生成手順

REQ-41（公開モデルの追加学習）・REQ-25・REQ-26・REQ-28・TASK-41.10-2（#439）の結合テスト用 fixture。
`crates/train/tests/qwen2_lora_evaluation_contract.rs` が読む。

## 内容

`predictions.json` のみ。学習ワーカーの CLI（`launch.py train`）が返した `validation_predictions` の
予測ラベル（判定不能は `null`）を、run ごとに入力順で記録したもの。

| キー | 内容 |
| ---- | ---- |
| `_meta` | run 一覧・ラベル順・seed・qwen2_lora の設定・生成環境 |
| `train_labels` | 学習データの正解ラベル（160 件。下限基準 majority の導出専用） |
| `cases` | 検証 120 件の名前と正解（`label_order[i % 8]`）。入力本文は `qwen2_parity/cases.json` と同じ生成規則で、ここには持たない |
| `predictions` | `c1:seed0`・`c3:seed0`・`qwen2_lora:seed0`・`qwen2_lora:seed1`・`qwen2_lora:seed2` |

## 出典とライセンス

- ベースモデル・学習データ・検証入力は `fixtures/qwen2_parity` と同じ合成（`trainer/tools/gen_qwen2_parity_fixture.py`
  の関数を再利用）。外部データ・LLM 生成物・実データ・事前学習済み重みを含まない（ライセンス確認の対象なし）

## 生成手順と環境

```bash
uv --no-config run --locked --offline --directory trainer python tools/gen_qwen2_eval_contract_fixture.py --out ../fixtures/qwen2_eval_contract --overwrite
```

- 生成環境: macOS arm64・CPU（`device="cpu"`）。mlx・onnx・numpy の版は `_meta.env`
- seed: 学習データ 20261017、検証入力 20261018、ベース重み 11、学習リクエストは run 名の seed
- 通信なし（REQ-38）
- 生成器は、qwen2_lora seed 0 の予測が `fixtures/qwen2_parity/cases.json` の MLX ラベルと全件一致しなければ失敗する

## 証拠種別と既知の制限

- 証拠種別: テストハーネス（合成の極小モデル・CPU）。実重み（Qwen2.5 系）での測定ではない（TASK-41.11・#442）
- 極小モデルの qwen2_lora は正解率が低く（majority を有意に上回らない）、テストは「有意性を装わない」
  ことも含めて具体値で固定する。再学習してバイト一致を確かめるテストは置かない
