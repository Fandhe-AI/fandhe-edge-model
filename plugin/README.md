# fandhe-edge プラグイン

fandhe-edge-model で作った判定モデルを、Claude Code から呼ぶプラグイン。
skill `judge` と、常駐 `serve` を裏で起動して `infer` を呼ぶラッパー
`fandhe-edge-judge` を持つ。hook は置かない。モデルとバイナリは同梱しない。
導入手順と実機確認の手順は [plugin-install-procedure](../docs/design/plugin-install-procedure.md) を参照。

## 前提

- `fandhe-edge` が PATH にあること

  ```sh
  cargo install --git https://github.com/Fandhe-AI/fandhe-edge-model --tag v0.1.0 --locked fandhe-edge-cli
  ```

- 環境変数 `FANDHE_EDGE_PACKAGE`: パッケージのパス
- 環境変数 `FANDHE_EDGE_IDLE_EXIT_SECS`: 任意。既定 600。使われない常駐が終わるまでの秒数

## 導入

```sh
claude plugin marketplace add Fandhe-AI/fandhe-edge-model
claude plugin install fandhe-edge@fandhe-edge-model
```

Claude Code 2.1.275 以降は、対話の中で次の 1 行でも入れられる（本リポジトリでは未確認）。

```text
/plugin install fandhe-edge --marketplace Fandhe-AI/fandhe-edge-model
```

許可ルールと動作確認は [plugin-install-procedure](../docs/design/plugin-install-procedure.md) の第 2 節を参照。

## 構成

| パス | 内容 |
| ---- | ---- |
| `.claude-plugin/plugin.json` | プラグイン定義 |
| `skills/judge/SKILL.md` | 判定の呼び方と終了コードの読み方 |
| `bin/fandhe-edge-judge` | 常駐の起動と `infer` 呼び出しのラッパー |

## 制約

- `FANDHE_EDGE_PACKAGE` は絶対パスで書く（`~` は展開されない）。プロジェクトの外に 1 つ置けば、どのプロジェクトからも使える
- 常駐に委ねるのは `infer --text` のみ。初回の判定は常駐の準備を待たず、単独で行う
- 確かめた環境は macOS（Apple Silicon）だけ
- 正答率の目安とそのほかの制約は、ルートの [README](../README.md) の「Claude Code プラグイン」を参照
