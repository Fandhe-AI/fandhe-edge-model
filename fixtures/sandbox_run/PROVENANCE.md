# sandbox_run fixture の来歴

- 生成元: 手作りの**合成データ**（実データ・LLM 生成物ではない）。個人情報・機密を含まない。証拠種別: テストハーネス
- 内容: `definition.json`（3 ラベルの single_select 定義）・`train.jsonl`（ラベルごとに 30 件、
  `crates/cli/tests/pipeline_e2e.rs` の合成データと同じ規則。group_id・input はすべて異なる）
- 用途: `crates/cli/tests/sandbox_pipeline_real_trainer.rs`（テストハーネス）と、人が macOS 実機で
  `scripts/sandbox-monitor.sh` を実行するときの共通入力（REQ-38・TASK-38.1・#161。手順は
  `docs/design/sandbox-offline-check-procedure.md`）
- `evaluation.jsonl` を置かない理由: 評価本体は配線済み（#314）だが、`--smoke` の候補は最終 test を
  適用できず、評価データがあると `package` が評価未完了として拒否する。本 fixture は smoke 経路
  （評価データなしで `evaluate` が `status:"skipped"`・exit 0）専用とし、評価データありの経路は
  `fixtures/sandbox_run_eval/` を使う
- 経路の閉じ込め（REQ-39）により、`--definition`・`--project-dir` はカレントディレクトリ配下へ
  コピーして使う（本 fixture 自体は書き換えない）
