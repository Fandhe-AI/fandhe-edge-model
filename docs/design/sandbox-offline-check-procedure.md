# sandbox 下の完走確認・実機での記録手順

対応: REQ-38（ローカル完結。正常系）・TASK-38.1（#161。子: #162・#163・#348）・M10・PoC-14・PoC-16。
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
| `crates/cli/tests/sandbox_pipeline_real_trainer.rs`（`#[ignore]`。`make test-trainer-integration`） | 実 CLI＋実 trainer（CPU・合成データ）で両スクリプトの全チェーンを通し、7 工程の完走と 0 件判定の接続を検証。評価なし（`--smoke`・`fixtures/sandbox_run/`）と評価あり（smoke なし・`fixtures/sandbox_run_eval/`）の 2 件 |

注意: 評価データありの完走（`evaluate` が `status:"ok"`）はテストハーネスで確認済みだが、実機では未確認で、人が §3-B の手順で確認する。
陽性対照の実機での実行（TASK-38.2・#164。スクリプトへは組み込み済み）が未実施のため、0 件を「検出手段が機能した上での 0 件」と書かない。

## 3. 実機での手順（人が実行）

3-A（評価データなし・smoke）と 3-B（評価データあり・smoke なし）の 2 経路がある。共通の前提は次のとおり。

1. 前提（sandbox の外で先に済ませる。承認済みの依存取得の通信は計測対象外）: `cargo build`・`make py-sync`。
2. 入力は 3-A なら `fixtures/sandbox_run/`、3-B なら `fixtures/sandbox_run_eval/`（いずれも合成データ）を作業ディレクトリ配下へコピーして使う。経路の閉じ込め（REQ-39）により
   `--definition`・`--project-dir` はカレントディレクトリ配下に置く。
3. テスト用の上書き環境変数は**すべて未設定**にする（`unset` してから実行する）。対象は
   `FANDHE_EDGE_SANDBOX_EXEC`（sandbox を偽物へ差し替える）・`FANDHE_EDGE_LOG_CMD`（log stream を偽物へ差し替える）・
   `FANDHE_EDGE_LOG_STREAM_*`（監視の待ち時間の調整）。`FANDHE_EDGE_BIN` は実 CLI を指す場合に限り使ってよい。
   いずれかを設定すると実機確認（REQ-38）にならず、`evidence_hint` が `test_harness` になる。
4. 実行例（3-A。手順 2 の作業ディレクトリへ移動してから、スクリプトはリポジトリの絶対パスで呼ぶ。
   `<REPO>` はリポジトリのルート、`<out>` は空または未作成、`<project>` は未作成）:

   ```bash
   cd <作業ディレクトリ>
   <REPO>/scripts/sandbox-monitor.sh --definition definition.json --project-dir project \
     --out-dir out --candidates 1 --smoke
   ```

### 3-B. 評価データあり（smoke なし）

- 入力は `fixtures/sandbox_run_eval/` の `definition.json`・`train.jsonl`・`evaluation.jsonl` の 3 ファイルをコピーする。
  `--smoke` は付けない（smoke 候補は最終 test を適用できず、`package` も拒否する）。前提・環境変数の扱いは 3-A と同じ
- 学習は CPU 固定（現状 `train` が CPU を指定するため指定は不要）。smoke より所要時間が長い（c1 を 30 epochs で学習する）
- 評価データは凍結され 1 回だけ適用される（REQ-27）。`<project>` は未作成のものを使い、再実行には新しい project-dir と out-dir を使う

```bash
cd <作業ディレクトリ>
<REPO>/scripts/sandbox-monitor.sh --definition definition.json --project-dir project \
  --out-dir out --candidates 1
```

### 3-C. 拡張確認（`--extended`。評価データあり・smoke なし）

REQ-38・#469 の CLI 結線で増えた工程・引数も、通信 0 件で完走することを確かめる任意の経路。3-B と同じ入力
（`fixtures/sandbox_run_eval/`）・前提で、`--extended` を足す。評価データが必要で、`--smoke` とは併用できない
（併用は起動前に 64。評価データなしで evaluate が skipped になった場合は 70 で止まる。評価の省略を通さないため）。

- 7 工程の後に、別プロジェクト `<project-dir>-extended`（未作成であること。`train --all` が既存の候補を拒否するため分ける）で
  `register`・`inspect`・`train --all --smoke --budget-seconds 600`・`train --status`・`train --cancel` を実行し、
  最後に本プロジェクトの package へ `infer --version-ledger <project-dir>/version_ledger.json --version-id v1` を実行する
- 校正・保留（REQ-22）は 7 工程内の `evaluate` が出す。`--extended` では `calibration`・`abstention` がオブジェクトで
  ない evaluate を完了扱いにせず 70 で止める。版管理台帳 `version_ledger.json` は 7 工程内の `package` が作る
- `train --cancel` は対象のジョブ（実行中）が無いため、空の一覧を返して exit 0 になる想定（`cancellations` が空）
- 許容する終了コードは全工程 0（infer は従来どおり判定行の status が一致する 11・12 も可）。1 つでも 0 以外なら
  その工程で停止する
- 実 trainer での学習を含むため 3-B より時間がかかる。実行は人の担当で、Agent は実行しない

```bash
cd <作業ディレクトリ>
<REPO>/scripts/sandbox-monitor.sh --definition definition.json --project-dir project \
  --out-dir out --candidates 1 --extended
```

## 4. 判定の読み方

| 終了コード | `network_verdict` | 意味 |
| ---------- | ----------------- | ---- |
| 0 | `zero_network_denials` | run が成功し、通信拒否 0 件 |
| 10 | 本ツール起因あり | PID 照合済みの通信拒否がある（不合格） |
| 12 | `unattributed_network_denials` | 帰属不明の通信拒否がある（要確認） |
| 70 | `undeterminable` など | 監視の無効・run の失敗（判定不能） |

評価データありで `evaluate` が失敗した場合（評価データのハッシュ不一致は 64 など）は run の終了コードがそのまま返り、判定不能（70）を含めて読む。

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
- `out/run/run.meta.json` の `exit_code` が 0・`failed_step` が `null` で、`steps[]` の全要素の `exit_code` が 0
  （`--candidates 1` なら 7 要素）であること。あわせて `out/network_report.json` の `run_exit_code` が 0・
  `run_code` が `ok` であること
  - 3-A（smoke）の場合: `steps[]` のうち `step` が `evaluate` の要素の `status` が `skipped` である旨
    （smoke は evaluate を起動せず skipped として記録される）
  - 3-B（評価データあり）の場合: `step` が `evaluate` の要素の `status` が `ok` で、`steps[]` に `status` が
    `skipped` の要素が 0 件であること（`status` に値が入るのは evaluate だけで、ほかの工程は `null`）
  - 3-C（`--extended`）の場合: `steps[]` が `--candidates 1` で 13 要素（7 工程 + `register`・`inspect`・`train`×3・
    `infer`）で全要素の `exit_code` が 0、`evaluate` の `status` が `ok`、`<project-dir>-extended` が作られていること。
    証拠種別は実機（実機で未実施の間は「テストハーネスのみ。実機未確認」と明記する）
- 記録はすべて `out/` 配下のファイルから取る。`sandbox-run.sh` の標準出力（集計 message）は
  `sandbox-monitor.sh` が捨てるため残らず、記録項目に含めない
- 生ログ `log_stream.ndjson` は他アプリのイベントを含むため**転記しない**。データ本文・パスも記録しない
