# sandbox_run fixture の来歴

- 生成元: 手作りの**合成データ**（実データ・LLM 生成物ではない）。個人情報・機密を含まない。証拠種別: テストハーネス
- 内容: `definition.json`（3 ラベルの single_select 定義）・`train.jsonl`（ラベルごとに 30 件、
  `crates/cli/tests/pipeline_e2e.rs` の合成データと同じ規則。group_id・input はすべて異なる）
- 用途: `crates/cli/tests/sandbox_pipeline_real_trainer.rs`（テストハーネス）と、人が macOS 実機で
  `scripts/sandbox-monitor.sh` を実行するときの共通入力（REQ-38・TASK-38.1・#161。手順は
  `docs/design/sandbox-offline-check-procedure.md`）
- `evaluation.jsonl` を置かない理由: 評価データありの `evaluate` は評価本体の CLI 配線が未実装で
  `runtime_error`(70) で停止する（評価済みを装わない）ため、評価データなしの `status:"skipped"`
  経路（exit 0）で 7 工程を完走させる
- 経路の閉じ込め（REQ-39）により、`--definition`・`--project-dir` はカレントディレクトリ配下へ
  コピーして使う（本 fixture 自体は書き換えない）
