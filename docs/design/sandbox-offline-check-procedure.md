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
評価器へ接続済み（#314）だが、この結合テストは評価データなしの経路だけで、**評価データありの完走は未確認**（実機での再確認が必要）。
陽性対照の実機での実行（TASK-38.2・#164。スクリプトへは組み込み済み）が未実施のため、0 件を「検出手段が機能した上での 0 件」と書かない。

## 3. 実機での手順（人が実行）

1. 前提（sandbox の外で先に済ませる。承認済みの依存取得の通信は計測対象外）: `cargo build`・`make py-sync`。
2. 入力は `fixtures/sandbox_run/`（合成データ）を作業ディレクトリ配下へコピーして使う。経路の閉じ込め（REQ-39）により
   `--definition`・`--project-dir` はカレントディレクトリ配下に置く。
3. テスト用の上書き環境変数は**すべて未設定**にする（`unset` してから実行する）。対象は
   `FANDHE_EDGE_SANDBOX_EXEC`（sandbox を偽物へ差し替える）・`FANDHE_EDGE_LOG_CMD`（log stream を偽物へ差し替える）・
   `FANDHE_EDGE_LOG_STREAM_*`（監視の待ち時間の調整）。`FANDHE_EDGE_BIN` は実 CLI を指す場合に限り使ってよい。
   いずれかを設定すると実機確認（REQ-38）にならず、`evidence_hint` が `test_harness` になる。
4. 実行例（手順 2 の作業ディレクトリへ移動してから、スクリプトはリポジトリの絶対パスで呼ぶ。
   `<REPO>` はリポジトリのルート、`<out>` は空または未作成、`<project>` は未作成）:

   ```bash
   cd <作業ディレクトリ>
   <REPO>/scripts/sandbox-monitor.sh --definition definition.json --project-dir project \
     --out-dir out --candidates 1 --smoke
   ```

## 4. 判定の読み方

| 終了コード | `network_verdict` | 意味 |
| ---------- | ----------------- | ---- |
| 0 | `zero_network_denials` | run が成功し、通信拒否 0 件 |
| 10 | 本ツール起因あり | PID 照合済みの通信拒否がある（不合格） |
| 12 | `unattributed_network_denials` | 帰属不明の通信拒否がある（要確認） |
| 70 | `undeterminable` など | 監視の無効・run の失敗（判定不能） |

拒否行は `Sandbox: <プロセス名>(<pid>) deny(<n>) <操作> [対象]` に加え、同じ構造の
`System Policy: ...` 形式も集計対象とする（Issue #331）。`network*` 以外の操作（他プロセスの
`file-read-data` 等）の拒否は無視し、判定不能にしない。`network*` の拒否は接頭辞によらず
同じ規則で帰属・件数に反映する。上記 2 種の接頭辞以外で `(<pid>) deny(<n>)` の構造を欠く
拒否行は、従来どおり判定不能（70）とする。

## 5. PR・Issue へ記録する項目

- 証拠種別: 実機（機種・OS 版を併記）
- 記録の前に、`out/network_report.json` の `evidence_hint` が `requires_human_review` であること、
  `log_stream_override` が `false` であること、`out/run/run.meta.json` の `sandbox_exec_override` が `false` である
  ことを確認する。1 つでも異なれば実機の証拠として記録せず、環境変数を見直して再実行する
- `out/network_report.json` の `counts`・`network_verdict`・`evidence_hint`・`positive_control`
- `out/run/run.meta.json` の工程ごとの終了コード（評価は `skipped` である旨）
- 生ログ `log_stream.ndjson` は他アプリのイベントを含むため**転記しない**。データ本文・パスも記録しない
