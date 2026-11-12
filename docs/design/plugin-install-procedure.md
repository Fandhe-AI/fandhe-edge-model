# Claude Code プラグインの導入手順と実機確認の手順

対応: REQ-33・REQ-36・REQ-38・REQ-39・#726・#730。前提は `plugin/`（#729）、`serve --idle-exit-secs`（#727）、`serve --stop`（#728）。
spec の内容は要約であり、詳細は spec の各 REQ を参照する。

## 1. 目的・担当・証拠種別

- 目的: 利用者がプラグインを入れる手順、人が実機で確かめる手順と記録欄、版のタグの切り方を 1 か所にまとめる。
- 担当: 導入手順（第 2 節）は利用者向け。実機確認（第 3 節）は**人**が行う。Agent は手順の準備までに留め、実機の結果を確定させない。
- 証拠種別: 人が実機で行った記録は「**実機**」。Agent が一時ディレクトリで行う予行は「テストハーネス」で、受け入れ条件を満たさない。
- 通信: `cargo install --git` と marketplace の取得は通信を伴うため、実行前に利用者が承認する（REQ-38）。本ツールの実行時（推論・常駐）は通信しない。

## 2. 導入手順（利用者向け）

1. バイナリ
   - `cargo install --git https://github.com/Fandhe-AI/fandhe-edge-model --tag <版> --locked fandhe-edge-cli`
   - `<版>` は導入するプラグインの `plugin.json` の version に `v` を付けたもの。CLI とプラグインの版は一致させる。
   - `command -v fandhe-edge` で PATH にあることを確かめる。
   - 常駐のソケットは版ごとに分かれる（`serve.v<版>.sock`）。更新後の CLI の `serve --stop` では旧版の常駐が見えず `not_running` になる。CLI を更新する前に、旧版の `fandhe-edge serve --stop` で常駐を止める。止め忘れた旧版の常駐は、アイドル終了を待つか手で止める。
2. パッケージ
   - 利用者が用意する。モデルは同梱も取得もしない。完全性は既存の sha256 照合に任せる。
   - 第三者の重みを含む場合の再配布可否は、[licensing](../../.claude/rules/licensing.md) に従い利用者が判断する。
   - Claude Code を起動するプロジェクトの cwd 配下に置く（例: `.fandhe-edge/package/`）。ガード層の閉じ込めにより cwd の外は使えない（REQ-39）。
   - 1 置き場所につき 1 パッケージ。
3. 環境変数
   - 個人用は `.claude/settings.local.json`（git 管理外）の `env` に書く。例: `{"env":{"FANDHE_EDGE_PACKAGE":".fandhe-edge/package"}}`
   - 相対パスは、Claude Code をプロジェクトのルートで起動する前提。
   - `FANDHE_EDGE_IDLE_EXIT_SECS` は任意。既定 600（推定値。#710 の計測後に見直す）。
4. 許可ルール
   - `permissions.allow` に、判定ラッパー `bin/fandhe-edge-judge` の呼び出しだけに一致するルールを 1 行足す。
   - 広い許可（`Bash(*)`・`sh`・`fandhe-edge` 全体）は勧めない。
   - 正確な書式は第 3 節の項目 0 で確定し、本節へ追記する（確定前に書式を断定しない）。
5. marketplace の追加とプラグインの導入
   - `claude plugin marketplace add Fandhe-AI/fandhe-edge-model`（ローカルの clone パスでも可）の後、`claude plugin install fandhe-edge@fandhe-edge-model`。対話では `/plugin` から同じ操作ができる。
   - コマンドの形は Claude Code の版で変わりうるため `claude plugin --help` で確かめる。実際の形と、プラグインがタグで固定されるか否かは項目 0 で確定し、追記する。
6. 動作確認
   - skill `judge` 経由で判定が通り、JSON が返ること。
   - 使い終えたら `fandhe-edge serve --stop` で手で止められること（出力は `{"status":"stopped"}` か `{"status":"not_running"}`）。

## 3. 実機確認の手順と記録欄（人が実行）

### 共通の準備

- 合成パッケージを `tmp-plugin-check/` に作る。作り方は [claude-code-permission-prompt-procedure.md](claude-code-permission-prompt-procedure.md) の第 3 節に従う（`fixtures/onnx_parity/c1.onnx` と heredoc）。
- `cargo install --path crates/cli --locked` で `fandhe-edge` を PATH に載せる。`cargo build -p fandhe-edge-cli` を使う場合、バイナリは `target/debug/` に出るだけで PATH に入らない。`export PATH="$PWD/target/debug:$PATH"` を実行したシェルから Claude Code を起動し、PATH を引き継ぐ。判定ラッパーの `command -v` が失敗すると exit 70 になる。
- `FANDHE_EDGE_SERVE_DIR` を一時ディレクトリへ向ける（普段使いの常駐とぶつけない）。`env` にも同じ値を書く。
- `FANDHE_EDGE_IDLE_EXIT_SECS` は 30 など短い値にする。実測は設定値に対する値として記録する。
- 低電力モードはオフ（[ci](../../.claude/rules/ci.md) の基準）。数値にモードを明記する。
- 常駐の有無は別ターミナルで確かめる。親と worker で 2 行出る。

  ```sh
  ps -axo pid,ppid,etime,rss,command | grep '[f]andhe-edge serve'
  ```

### 項目 0: 前提の確定

- 許可ルールの書式を `/permissions` と実際の確認画面から確定する。
- `claude plugin` の導入コマンドの形と、プラグインの取得元（タグ固定か checkout か）を確定する。
- プラグインの `bin/` が Bash ツールの PATH に載るか観察する。
- 結果を第 2 節へ反映する。

### 項目 1: プラグインから判定を呼べる

- 期待値: skill 経由で判定が通り、exit 0、stdout が JSON 1 つ。

### 項目 2: アイドル終了

- 通常終了: 判定の後に Claude Code を `/exit` で終える。
- 強制終了: 判定の後に Claude Code のプロセスを `kill -9` する。
- どちらも、最後の判定からの経過秒を計り、親と worker の両方が居なくなるまでの時間を記録する。
- 期待値: アイドル時間＋数秒以内に居なくなる。ソケットが残っていない。

### 項目 3: 2 セッションで共有できる

- 同じ cwd で 2 つの Claude Code セッションから続けて判定する。
- 期待値: 常駐の親と worker の組が 1 つだけで pid が変わらない。2 件目以降は読み込み待ちなしで返る。

### 項目 4: 裏で起動した `serve` が Bash ツールの終了後も残る

- 判定の呼び出しが終わった直後に、別ターミナルで `ps` を見る。
- 期待値: 残っている。
- 残らない場合は、#726 の設計どおり `--detach` を足す新規 issue の材料として報告する。本書やプラグインは直さない。

### 項目 5: `env` がラッパーに届く

- `env` に設定したときは判定が通る。`FANDHE_EDGE_PACKAGE` を外すと code `invalid_input`・exit 64 になる。
- `FANDHE_EDGE_IDLE_EXIT_SECS` の値が常駐に効いている（項目 2 の秒数と一致する）。

### 記録テンプレート

本文を転記しない（[security](../../.claude/rules/security.md)）。入力テキスト・stdout 本文は貼らず、終了コードと件数だけにする。記録は #726 へコメントする。

```text
証拠種別: 実機
実施日:
機種・OS:
Claude Code の版:
fandhe-edge の版（タグまたは commit）:
低電力モード:
項目 0（許可ルールの書式・導入コマンド・PATH）:
項目 1（exit・JSON 1 つか）:
項目 2（設定秒 / 通常終了の実測秒 / 強制終了の実測秒 / ソケット残り）:
項目 3（常駐の組の数 / pid 不変か）:
項目 4（判定直後に残っていたか）:
項目 5（env 有り exit / env 無し exit / idle 秒の反映）:
```

### 後片付け

- `fandhe-edge serve --stop` で常駐を止める。
- `tmp-plugin-check/` を削除し、`settings` の変更を戻し、`git status` で未追跡ファイルが無いことを確かめる。
- 使い終えた出力の削除は `trash` コマンドで行う（実行は本人）。

## 4. 版のタグの切り方

版は次の 3 か所を揃える。`trainer/tests/test_plugin.py` が一致を機械照合する。

- ルート `Cargo.toml` の `[workspace.package] version`（`Cargo.lock` も更新される）
- `plugin/.claude-plugin/plugin.json` の `version`
- `.claude-plugin/marketplace.json` の `plugins[].version`

手順:

1. 上の 3 か所を上げる。
2. 機械照合の対象外の旧版文字列を直す。対象は `plugin/README.md` の `cargo install` の行と `plugin/bin/fandhe-edge-judge` のエラー文。`grep -rn 'v<旧版>' plugin README.md` で漏れを探す。
3. `make test` と `make py-ci` を通す。
4. main にマージした commit に注釈付きタグ `v<版>` を切って push する。タグの push は共有状態を変えるため、オーナーが実行するか明示承認を得てから行う。

`release.yml`（crates.io）は無効化中で、本タグとは無関係。ビルド済みバイナリの配布は、Rust を使わない利用者へ広げる段で別に扱う（#726）。

## 5. 制約・限界

- 1 置き場所 1 パッケージ。複数モデルの切り替えは扱わない。
- 常駐に委ねるのは `infer --text` のみ。セッション終了 hook では止めない。
- 常駐は 0700 の置き場所と 0600 の Unix ソケットだけで、ネットワークに公開しない。`serve --stop` も同じソケット経由。
- アイドル時間 600 秒は推定値。
