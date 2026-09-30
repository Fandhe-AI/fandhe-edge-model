# sandbox 下の完走確認・実機での記録手順

対応: REQ-38（ローカル完結。正常系）・TASK-38.1（#161。子: #162・#163）・M10・PoC-14・PoC-16。
spec の内容は要約であり、詳細は spec の REQ-38・PoC-16 を参照する。

## 1. 目的と範囲

通信を遮断した状態（sandbox）で 7 工程が完走し、本ツールの実行プロセスに起因する通信拒否が 0 件であることを、
macOS 実機で**人が**確認して記録する手順を定める。Agent の範囲はスクリプトと手順の準備までで、実機の結果を
Agent が確定させない（`evidence_hint` は `requires_human_review` か `test_harness` のみ）。

## 2. Agent 側で済んでいる検証（証拠種別: テストハーネス）

いずれも実際の通信遮断・実際の `log stream` を使わず、実機の証拠にならない。

| テスト | 内容 |
| ------ | ---- |
| `crates/cli/tests/sandbox_run_script.rs` | 偽の launcher・偽の CLI で `sandbox-run.sh` の制御を検証 |
| `crates/cli/tests/sandbox_monitor_script.rs` | 偽の `log`・合成ログで判定規則と出力契約を検証 |
| `crates/cli/tests/sandbox_pipeline_real_trainer.rs`（`#[ignore]`。`make test-trainer-integration`） | 実 CLI＋実 trainer（CPU・合成データ）で両スクリプトの全チェーンを通し、7 工程の完走と 0 件判定の接続を検証 |

注意: 評価データなしの `evaluate`（`status:"skipped"`）経路のみを通している。評価データありの `evaluate` は
評価本体の CLI 配線が未実装で `runtime_error`(70) で停止するため、**評価の完走は未達**で、配線後に再確認が必要。
陽性対照（TASK-38.2・#164）も未実施のため、0 件を「検出手段が機能した上での 0 件」と書かない。

## 3. 実機での手順（人が実行）

1. 前提（sandbox の外で先に済ませる。承認済みの依存取得の通信は計測対象外）: `cargo build`・`make py-sync`。
2. 入力は `fixtures/sandbox_run/`（合成データ）を作業ディレクトリ配下へコピーして使う。経路の閉じ込め（REQ-39）により
   `--definition`・`--project-dir` はカレントディレクトリ配下に置く。
3. オーバーライド用の環境変数（`FANDHE_EDGE_SANDBOX_EXEC`・`FANDHE_EDGE_LOG_CMD`・`FANDHE_EDGE_BIN` 以外の
   `FANDHE_EDGE_LOG_STREAM_*`）は設定しない。設定すると `evidence_hint` が `test_harness` になる。
4. 実行例（`<out>` は空または未作成、`<project>` は未作成）:

   ```bash
   scripts/sandbox-monitor.sh --definition definition.json --project-dir project \
     --out-dir out --candidates 1 --smoke
   ```

## 4. 判定の読み方

| 終了コード | `network_verdict` | 意味 |
| ---------- | ----------------- | ---- |
| 0 | `zero_network_denials` | run が成功し、通信拒否 0 件 |
| 10 | 本ツール起因あり | PID 照合済みの通信拒否がある（不合格） |
| 12 | `unattributed_network_denials` | 帰属不明の通信拒否がある（要確認） |
| 70 | `undeterminable` など | 監視の無効・run の失敗（判定不能） |

## 5. PR・Issue へ記録する項目

- 証拠種別: 実機（機種・OS 版を併記）
- `out/network_report.json` の `counts`・`network_verdict`・`evidence_hint`・`positive_control`
- `out/run/run.meta.json` の工程ごとの終了コード（評価は `skipped` である旨）
- 生ログ `log_stream.ndjson` は他アプリのイベントを含むため**転記しない**。データ本文・パスも記録しない
