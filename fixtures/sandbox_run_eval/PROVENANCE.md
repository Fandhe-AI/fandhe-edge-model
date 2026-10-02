# sandbox_run_eval fixture の来歴

- 生成元: 手作りの**合成データ**（実データ・LLM 生成物ではない）。個人情報・機密を含まない。証拠種別: テストハーネス
- 内容: `definition.json`（3 ラベルの single_select 定義。`acceptance`・`baseline_comparison`・`limits` は持たない）・
  `train.jsonl`（`fixtures/sandbox_run/train.jsonl` と同一の 90 件）・`evaluation.jsonl`（ラベルごとに 4 件、計 12 件）。
  評価データの id・input・group_id は train と重ならない
- 用途: 評価データありの経路（`--smoke` を付けない）で 7 工程を完走させる
  `crates/cli/tests/sandbox_pipeline_real_trainer.rs`（テストハーネス）と、人が macOS 実機で
  `scripts/sandbox-monitor.sh` を実行するときの入力（REQ-38・TASK-38.1・#161・#348。手順は
  `docs/design/sandbox-offline-check-procedure.md`）
- 評価データは `register` 時に凍結され、最終 test として 1 回だけ適用される（REQ-17・REQ-27）。
  同じ project-dir では再評価できないため、再実行には新しい project-dir が要る
- 経路の閉じ込め（REQ-39）により、`--definition`・`--project-dir` はカレントディレクトリ配下へ
  コピーして使う（本 fixture 自体は書き換えない）
