# Conventional Commits 規約

## 形式

```
<type>(<scope>): <日本語の説明>

<本文（任意・日本語）>
```

`make lint-commits`（commitlint・`commitlint.config.mjs`）と CI の lint-docs（PR 時）で検証する。lefthook の commit-msg フックは `lefthook.yml` の追加後に有効になる。

## type

| type | 用途 |
| ---- | ---- |
| feat | 機能追加 |
| fix | バグ修正 |
| refactor | 挙動を変えないコード整理 |
| perf | 性能改善 |
| test | テストの追加・修正 |
| docs | ドキュメントのみの変更 |
| ci | CI 設定の変更 |
| build | ビルド・依存関係の変更 |
| chore | 上記以外の雑務 |

`commitlint.config.mjs` の `type-enum` もこの 9 種類に限定している。

## scope

scope は spec の 6 層に対応する。TASK-15.2（#28）で core・cli の 2 crate、TASK-16.1-1（#38）・TASK-17.1-1（#44）で data crate（検査・group 単位分割ロジック）、TASK-24.1-1（#59）で eval crate（正解率・ラベル別指標・Macro-F1・混同行列）、issue #177 で train crate（学習リクエスト・結果 JSON の型）のパスが確定し、残りの層のパスは crate 追加時に本表へ反映する。

| scope | 対象 | パス |
| ----- | ---- | ---- |
| core | 共通コア（定義ファイル・選択肢・判定型・正準化ハッシュ。REQ-15） | `crates/core/`（`fandhe-edge-core`） |
| data | データ契約（検査・group 分割と凍結・来歴・読み取り専用配置。REQ-16/17/40） | `crates/data/`（`fandhe-edge-data`。検査・group 単位分割・来歴の記録型と取り込み記録・データ検査との接続を実装済み。凍結・読み取り専用配置は未着手） |
| train | 学習ワーカー（候補学習・選定・選択口・作り直し判定・ジョブ管理。REQ-18〜20/34） | `trainer/`（Python）・`crates/train/`（`fandhe-edge-train`。学習リクエスト・結果 JSON の型を実装済み。issue #177） |
| eval | 評価器（指標・有意性・回帰・診断。REQ-21〜27/29） | `crates/eval/`（`fandhe-edge-eval`。正解率・ラベル別指標・Macro-F1・混同行列を実装済み。有意性・回帰・診断は未着手） |
| runtime | 成果物・推論 SDK（学習非依存の推論ランタイム。REQ-28/30〜32） | 未確定 |
| cli | CLI（7 工程・JSON 入出力契約。REQ-33） | `crates/cli/`（`fandhe-edge-cli`） |
| tui | TUI（REQ-35） | 未確定 |
| mcp | MCP / Codex 連携（REQ-36/37） | 未確定 |
| guard | ガード層（経路・形式・資源・版の検査。REQ-39） | 未確定 |
| deps | 依存の追加・更新（`Cargo.toml`・`Cargo.lock`・`pyproject.toml`・lock ファイル） | — |
| spec | `docs/spec` submodule 参照の更新 | — |
| claude | `CLAUDE.md`・`.claude/`（agents・rules・settings・workflows） | — |
| skills | `.claude/skills`・`.agents/skills`・`skills-lock.json` | — |

CI・Makefile 等は type（`ci` / `build`）で表し、scope は省略してよい。複数層に跨る場合は scope を省略するか、主たる対象を選ぶ。

## breaking change

- 破壊的変更は `!` を付け（例: `feat(cli)!: ...`）、本文に `BREAKING CHANGE:` を記載する
- CLI の JSON 出力・終了コード・配布パッケージ形式の変更は利用者（エージェント）側を壊すため、原則 breaking change として扱う（REQ-21・REQ-33）

## 禁止事項

- `git commit --no-verify` の使用（pre-commit / commit-msg フックを必ず通す）
- 複数の関心事を 1 コミットに混在させること（type が 2 つ以上必要なら分割する）
- スコープ外の変更の混入（[out-of-scope-tracking](./out-of-scope-tracking.md)）
