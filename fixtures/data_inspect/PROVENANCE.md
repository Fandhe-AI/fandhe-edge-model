# data_inspect フィクスチャの出典

## 出典

`docs/spec/03-poc/evaluation-contract/fixtures/injection/clean/{train,test}.jsonl`
（PoC-9。private submodule）から、バイト単位でそのまま移植した（内容を加工していない）。

- `clean/train.jsonl`（sha256: `5d8d1fc68eae961f33696112d12cd1fa4a2d5dd810cd1ca5c392ed0fab123ae6`）
- `clean/test.jsonl`（sha256: `de485b7870ad844cbaf0c657ef9505f6b30a3f25dbd6de9a9e345dd1d0ba8626`）

いずれも移植元と同一のハッシュであることを確認済み（証拠種別: テストハーネス。
`sha256sum` による突き合わせ）。

`clean/labels.json` は `docs/spec/03-poc/evaluation-contract/evaluator/labels_9intent.json`
の `labels` 配列（9 intent: `create_task`・`complete_task`・`delete_task`・
`list_tasks`・`set_reminder`・`create_note`・`search_notes`・`schedule_event`・`none`）
と同じ内容を、`{"labels": [...]}` の形へ整形して複製した。

## 移植範囲

issue #39（TASK-16.1-2）が検査レポート（件数・ラベル別集計）の受け入れテスト
（`crates/data/tests/inspect_report_clean.rs`）向けに移植した。同テストが読み込むのは
`train.jsonl`・`test.jsonl` のみで、ラベル集合は `nine_intent_labels()` 関数として
コード側にハードコードしている（`labels.json` は読み込まない）。`clean/labels.json`
はラベル集合の参照資料として PoC-9 から複製したもので、将来ラベル集合をファイルから
読み込むテスト・実装を追加する際の移植元として残している。

## 生成元

PoC-1（`e2e-minimal`）のテンプレートで正解ラベル・引数を決め、自然文をローカル
Ollama の `qwen2.5:7b-instruct`（Apache-2.0）で生成した合成データ。PoC-9 がそこから
`fixtures/injection/clean/` として抽出したものを移植する。実データ・個人情報を
含まない。

## PoC-9 との既知の差分

- `clean/labels.json` は元ファイル（`labels_9intent.json`）に末尾改行が無く
  `.editorconfig`（`insert_final_newline = true`）に抵触するため、バイト単位の
  複製ではなく同内容を末尾改行付きで作成した。`source` フィールド（labels の
  由来コメント）は本フィクスチャの用途に不要なため含めていない。

## PoC-9 `inspect_split` による参照値（証拠種別: テストハーネス）

`docs/spec/03-poc/evaluation-contract` を作業ディレクトリとして
`evaluator.inspect.inspect_split` を `train.jsonl`・`test.jsonl` それぞれに
適用した実測値。`crates/data/tests/inspect_report_clean.rs` の期待値の根拠。

| フィクスチャ | rows | unique_inputs | unique_outputs | intent_counts | min_intent_count |
| ------------ | ---- | -------------- | --------------- | -------------- | ------------------ |
| `train.jsonl` | 10 | 10 | 10 | `{"create_task": 10}` | 10 |
| `test.jsonl` | 10 | 10 | 7 | `{"create_task": 2, "complete_task": 3, "delete_task": 3, "list_tasks": 2}` | 2 |

`unique_outputs`（`output` オブジェクト全体の異なり数）が `test.jsonl` で
intent 種類数（4）と一致しない点が、`InspectReport::unique_outputs` を
`output.intent` ではなく `output` 全体から計算する必要がある根拠（`crates/data/src/report.rs`
モジュール doc「PoC-9 との対応・差異」参照）。
