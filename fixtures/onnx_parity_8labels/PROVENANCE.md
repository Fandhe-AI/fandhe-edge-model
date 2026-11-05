# onnx_parity_8labels fixture の出典と生成手順

REQ-19b・REQ-33・#699 の e2e 用 fixture。`fixtures/qwen2_parity` と同じ選択肢 8 つ・`max_bytes` 64 で
c1・c3・autoregressive を作り、偽 trainer で qwen2_lora と混ぜても select が選択肢の数の不一致で
止まらないようにする。内容・形式は `fixtures/onnx_parity/PROVENANCE.md` と同じ（ファイル構成、
`cases.json` の項目、証拠種別: テストハーネス）。`fixtures/onnx_parity` は変えない。

## 出典とライセンス

- 学習データ・評価入力はすべて生成器（`trainer/tools/gen_onnx_parity_fixture.py`）が固定 seed で作る合成データ。
  各選択肢の固有の語は選択肢名から派生（`red`・`reda`・`redbb`・`redccc`・`reddd` など）し、雑音語と共通ゴールデンベクタの入力を加える。
  外部データ・LLM 生成物・実データは含まない
- 事前学習済み重み・外部データセットは使っていない。ONNX の重みは合成データから本リポの学習ワーカーが学習したもの（`MIT OR Apache-2.0`）

## 生成手順と環境

```bash
UV_NO_CONFIG=1 uv run --offline --locked --directory trainer python tools/gen_onnx_parity_fixture.py --out-dir ../fixtures/onnx_parity_8labels --labels red,tan,sky,map,fox,owl,gem,ink --max-bytes 64 --overwrite
```

- 生成環境: macOS arm64・CPU（`device="cpu"`）。versions は `cases.json` の `_meta.versions`
- seed 20260929。選択肢の並びは `fixtures/qwen2_parity/artifact.json` の `label_order` と同じ
- 設定は `fixtures/onnx_parity` と同じ（C1 `ngram 1..3`・`epochs=40`、C3 `emb=8`・`filters=8`・`epochs=5`、autoregressive `layers=1`・`dims=16`・`epochs=30`）
- 生成器の検査は既存と同じ（ReferenceEvaluator との確率差 1e-5 以内・予測ラベル 2 種類以上・僅差 1e-3 未満は除外）
- 通信なし（REQ-38）
- 再学習してバイト一致を確かめるテストは置かない（OS 間で MLX CPU の結果の一致を保証できないため）
