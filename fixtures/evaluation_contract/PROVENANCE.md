# evaluation_contract フィクスチャの出典

## 出典

`docs/spec/03-poc/evaluation-contract/fixtures/`（PoC-9。private submodule）から、
バイト単位でそのまま移植した（内容を加工していない）。

- `known/single-select/{gold.jsonl,pred.jsonl,labels.json,expected.json}`
- `anomaly/01-empty-data/{gold.jsonl,pred.jsonl,expected.json}`
- `anomaly/02-missing-gold/{gold.jsonl,pred.jsonl,expected.json}`
- `anomaly/03-unknown-label/{gold.jsonl,pred.jsonl,labels.json,expected.json}`
- `anomaly/04-type-invalid/{gold.jsonl,pred.jsonl,labels.json,expected.json}`
- `anomaly/05-duplicate-id/{gold.jsonl,pred.jsonl,expected.json}`
- `anomaly/06-duplicate-input/{gold.jsonl,pred.jsonl,expected.json}`

`known/single-select/expected-derivation.md`（手計算の導出過程メモ）は移植していない。

## 移植範囲

本 PR（issue #55・TASK-23.1-1）は `known/single-select` と `anomaly/01`〜`06` のみを
移植する。PoC-9 の `anomaly/07`〜`12`（矛盾・ラベル順序・未出現クラス・不正なスコア・
全件保留・全件失敗）は、兄弟 issue #56（TASK-23.1-2）が本ディレクトリへ追加する。

## 生成元

PoC-9 のフィクスチャは、評価契約（REQ-21〜27・REQ-29）の異常系挙動を固定するために
手作りで作成された合成データである（実データではない。個人情報・機密情報を含まない）。

## labels.json が無いケースの扱い

`anomaly/01-empty-data`・`02-missing-gold`・`05-duplicate-id`・`06-duplicate-input` には
`labels.json` が無い。PoC-9 のテストハーネス（`evaluator/harness.py`）と同じく、
これらのケースでは `known/single-select/labels.json`（ラベル `A`・`B`・`C`・`D`）を
共通のラベル定義として使う（本リポの結合テスト `crates/data/tests/eval_input_anomaly.rs`
も同じ規約に従う）。

## PoC-9 との既知の差分

- `01-empty-data` の `gold.jsonl`・`pred.jsonl` は PoC-9 と同じく 0 バイト
  （空行 1 つではなく完全な空ファイル）。`.editorconfig` の
  `insert_final_newline = true` は 0 バイトファイルには適用されない
  （editorconfig-checker は空ファイルを「改行なし」として指摘しないことを
  `make lint-docs` で確認済み。証拠種別: テストハーネス）。
