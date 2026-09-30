# Claude Code の許可操作（確認画面）発生時の動作確認手順

対応: REQ-36（境界値「Claude Code の許可操作が出る権限モード」）・TASK-36.3（#160）・M9・PoC-16。前提は TASK-36.1（#148・#149・#150）と TASK-33.1-2（#310）。
spec の内容は要約であり、詳細は spec の REQ-36・TASK-36.3 を参照する。

## 1. 目的・担当・証拠種別

- 目的: コマンド実行の前に確認画面が出る権限モードで `scripts/cli-infer-noninteractive.sh` 経由の `infer` を実行し、承認後に正常終了（exit 0・1 行 JSON）することを 1 回確認する。PoC-16 は確認なしで実行するモードで行ったため、この条件は未検証だった（REQ-36）。
- 担当: **人**（spec 上は「共同」）。権限モードの切り替えと確認画面での承認・記録は人が行う。Agent は本手順書の準備までで、受け入れ条件を満たしたと扱わない。M9 は本確認が済むまで完了扱いにしない。
- 証拠種別: 人が Claude Code で実測した記録は「**実機**」。Agent が一時ディレクトリで行う予行（第 4 節の準備と、確認画面を通さない実行）は「テストハーネス」で、確認画面を通していないため受け入れ条件を満たさない。

## 2. 前提

- 検証環境は Mac（Apple Silicon）。Claude Code がインストール済みで、版を第 6 節の記録に書けること。
- Claude Code を起動するディレクトリ（cwd）は本リポジトリのルートに限る。準備手順の `fixtures/onnx_parity/c1.onnx` と実行例の `scripts/cli-infer-noninteractive.sh` はリポジトリルートからの相対パスであり、配下の作業用ディレクトリを cwd にするとファイルが見つからない。`--package` は **cwd 配下の相対パス**で渡す（`infer` は cwd を workspace として経路を閉じ込める。TASK-39.4-2・#159）。
- 確認は通信を発生させない（REQ-38）。ONNX はリポジトリ同梱の fixture を使い、外部から取得しない。

## 3. 準備（権限モードを切り替える前に、人がターミナルで実行）

1. `cargo build -p fandhe-edge-cli` で `target/debug/fandhe-edge` を作る。ラッパーは環境変数 `FANDHE_EDGE_BIN`、未設定なら `${CARGO_TARGET_DIR:-<repo>/target}/debug/fandhe-edge` を使う。
2. 合成パッケージを作る（構成は `crates/cli/tests/bash_noninteractive.rs` の `make_real_package` と同じ）。作業用ディレクトリはコミットされない場所にする（例: リポジトリルートの `tmp-task36-3/`。`.gitignore` 対象ではないため、作業後に `git status` で未追跡ファイルが残っていないことを確かめ、使用後に削除する）。次のコードブロックは heredoc の終端 `JSON` が行頭に来るようリストの外に置いている。そのまま貼り付けてよい。

```sh
W=tmp-task36-3
mkdir -p "$W/p" "$W/records"
cp fixtures/onnx_parity/c1.onnx "$W/p/model.onnx"
SHA=$(shasum -a 256 "$W/p/model.onnx" | cut -d' ' -f1)
cat > "$W/p/artifact.json" <<JSON
{"kind":"c1","kind_version":1,"max_bytes":48,"label_order":["alpha","beta","gamma"],"onnx_file":"model.onnx","onnx_sha256":"$SHA"}
JSON
cat > "$W/p/definition.json" <<'JSON'
{"schema":"fandhe-edge-model-definition/v1","name":"sh_real","version":1,"judgment_type":"single_select","options":[{"id":"alpha","display_name":"a","description":"d"},{"id":"beta","display_name":"b","description":"d"},{"id":"gamma","display_name":"g","description":"d"}],"io":{"input":"bytes"}}
JSON
```

3. 実行記録（TASK-36.1-2）の保存先 `$W/records` は実在する通常のディレクトリで、symlink は使えない。`onnx_sha256` の検査（完全性。REQ-39）は外さない。

## 4. 権限モードの切り替え（人の操作）

権限モードの名称・フラグは Claude Code の版で変わりうるため、起動前に `claude --help` と `/permissions` で現行の表示を確かめる（参照: 導入済みスキル `anthropic-claude-code` の `cli-reference.md`・`settings.md`）。

- コマンド実行の前に確認画面が出るモードで起動する。`--permission-mode default`（別名 `manual`）または既定。`acceptEdits`・`auto`・`dontAsk`・`bypassPermissions`（`--dangerously-skip-permissions`）のような、確認なしで実行するモードは使わない。
- `/permissions` で、ユーザー設定・`.claude/settings.local.json`・プロジェクト設定の allow ルールが `Bash(scripts/cli-infer-noninteractive.sh:*)` や `Bash(sh:*)` などに一致していないかを確かめる。一致していると確認画面が出ない。本リポジトリの `.claude/settings.json` には `permissions` の設定が無いが、ユーザー側の設定は人が確かめる。
- 本手順書は設定ファイルの編集を指示しない。一致するルールがあった場合は、そのセッションだけ除く方法を人が選ぶ。

## 5. 確認の実行（人の操作）

1. Claude Code に Bash ツールで次を実行させる。`--out` はラッパーが受け付けないので使わない。

   ```sh
   FANDHE_EDGE_RECORD_DIR=tmp-task36-3/records scripts/cli-infer-noninteractive.sh --package tmp-task36-3/p --text "hello world"
   ```

2. 確認画面が出ることを確かめる。表示されたコマンドと、選べる選択肢の種類を控える。
3. 承認する。
4. 期待値:
   - 終了コードが 0。
   - stdout が 1 行の JSON で、`{"id":"input","status":"ok","predicted_label":"` で始まる。
   - stderr の最終行が `exit_code=0`。
   - `tmp-task36-3/records/run-record.<pid>.<乱数>` が 1 件作られる。
5. （任意・境界）同じコマンドで確認画面から拒否する。期待値は、CLI が起動せず、`run-record.*` が増えないこと。

### 5.1 記録の限界

- 実行記録の `started_at` は CLI を起動した時刻、つまり承認の後。実行記録だけでは確認画面が出たことの証拠にならない。確認画面が出たこと（表示されたコマンド・承認か拒否か）は人が別途記録する。これが TASK-36.3 の本来の証拠になる。
- ラッパーの実行時間上限（既定 300 秒。`FANDHE_EDGE_TIMEOUT_SECS`）は起動後から数える。承認待ちの時間は含まない。Claude Code 側のタイムアウトと承認待ちの関係は、観察できたときだけ記録し、推測で書かない。

## 6. 記録テンプレート（Issue #160 または PR のコメントに貼る）

データ本文を転記しない（security.md）。入力テキスト・stdout 本文・stderr 本文は貼らず、件数とハッシュだけにする。スクリーンショットを貼る場合は、表示に入力本文が含まれないことを確かめる。

```text
実施日:
Mac の機種 / OS 版:
Claude Code の版:
権限モード名（起動時の表示）:
allow ルールに一致しないことの確認結果（/permissions）:
確認画面が出たか（はい / いいえ）:
操作（承認 / 拒否）:
ラッパーの exit_code:
実行記録のファイル名:
stdout.bytes / stdout.sha256:
stderr.bytes / stderr.sha256:
証拠種別: 実機
備考（承認待ちとタイムアウトの関係など、観察できたことのみ）:
```

## 7. 後片付け

- 作業用ディレクトリ（`tmp-task36-3/`）を削除する。コミットしない（`git status` で未追跡ファイルが残っていないことを確かめる）。
- 権限モードを元に戻す。
