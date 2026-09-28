# 出典（`fixtures/data_contract/injection/`）

- `clean/`・`metadata-mixed/`（各 `train.jsonl`・`test.jsonl`・`answer.json`）は本リポジトリで手書きした合成データ（生成元: 人手。学習・評価データ本文の同梱ではない）
- 従来 PoC-9 の `fixtures/injection/`（`clean/`・`metadata-mixed/`）をバイト単位で移植していたが、PR #188 レビュー指摘（P0。reviewThread PRRT_kwDOUq-SxM6mg8C0）で
  security.md「学習・評価データ本文をログ・エラーメッセージ・Issue・PR へ転記しない」に反すると指摘され、手書きの合成データへ全面差し替えた
- `metadata-mixed/` は `clean/` と同じ 10 件（train 5・test 5）をベースに、3 件へ意図的な混入を加えたもの（`answer.json` の `injections` 参照）。
  混入の種類（`kind`）は PoC-9 A-10 の `detection_rule` に対応する 3 種
  （`output_intent_in_input`・`id_in_input`・`output_arguments_in_input`）を各 1 件ずつ用意した
- `leak-duplicate/`・`group-straddle/` は学習↔評価の漏洩・group 跨ぎの検出（TASK-16.2-1・#41 の範囲）のため用意していない
- 用途と根拠: TASK-16.2-2（メタデータ混入検出。REQ-16）のテストハーネス（異常系）。証拠の種別: テストハーネス（手書き合成データ。実機ではない）
- 期待値の根拠: 各ディレクトリの `answer.json`
  - `clean/answer.json`: 誤検出 0 件（`false_positive_expectation`）
  - `metadata-mixed/answer.json`: 混入 3 件（`syn-tr-1`・`syn-tr-2`・`syn-te-1`）。検出規則は `detection_rule` のとおり
