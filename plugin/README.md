# fandhe-edge プラグイン

fandhe-edge-model で作った判定モデルを、Claude Code から呼ぶプラグイン。
skill `judge` と、常駐 `serve` を裏で起動して `infer` を呼ぶラッパー
`fandhe-edge-judge` を持つ。hook は置かない。モデルとバイナリは同梱しない。
導入手順の詳細は #730 で文書化する。

## 前提

- `fandhe-edge` が PATH にあること

  ```sh
  cargo install --git https://github.com/Fandhe-AI/fandhe-edge-model --tag v0.1.0 --locked fandhe-edge-cli
  ```

- 環境変数 `FANDHE_EDGE_PACKAGE`: パッケージのパス
- 環境変数 `FANDHE_EDGE_IDLE_EXIT_SECS`: 任意。既定 600。使われない常駐が終わるまでの秒数

## 構成

| パス | 内容 |
| ---- | ---- |
| `.claude-plugin/plugin.json` | プラグイン定義 |
| `skills/judge/SKILL.md` | 判定の呼び方と終了コードの読み方 |
| `bin/fandhe-edge-judge` | 常駐の起動と `infer` 呼び出しのラッパー |

## 制約

- `FANDHE_EDGE_PACKAGE` は、呼び出し時のカレントディレクトリ配下に置く。ガード層の経路の閉じ込めにより、プロジェクト外のパッケージは使えない
- 常駐に委ねるのは `infer --text` のみ。初回の判定は常駐の準備を待たず、単独で行う
