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
| data | データ契約（検査・group 分割と凍結・来歴・読み取り専用配置。REQ-16/17/40） | `crates/data/`（`fandhe-edge-data`。検査・group 単位分割・凍結記録・読み取り専用配置・ハッシュ不一致時の停止・分割記録のレコード内容ハッシュ・来歴の記録型と取り込み記録・データ検査との接続を実装済み） |
| train | 学習ワーカー（候補学習・選定・選択口・作り直し判定・ジョブ管理。REQ-18〜20/34） | `trainer/`（Python）・`crates/train/`（`fandhe-edge-train`。学習リクエスト・結果 JSON の型・子プロセス起動と終了コード写像・探索予算内の候補選定・選定結果の有意性判定を実装済み。CLI 配線は未着手。issue #177） |
| eval | 評価器（指標・有意性・回帰・診断。REQ-21〜27/29） | `crates/eval/`（`fandhe-edge-eval`。指標・McNemar / Holm・回帰・再現性・不変性・校正と棄権などを実装済み。対象外ラベル・coverage・レポート系は未実装） |
| runtime | 成果物・推論 SDK（学習非依存の推論ランタイム。REQ-28/30〜32） | `crates/runtime/`（`fandhe-edge-runtime`。単体/バッチ共通の推論経路の継ぎ目〔#117〕・容量計測コア〔TASK-30.1-1・#122〕・容量内訳の JSON 出力接続〔エラー写像は runtime、直列化は cli の output。TASK-30.1-2・#123〕を実装済み。バイト前処理〔TASK-32.1-1・#112〕・C1・C3 の ONNX を読む自作推論バックエンド〔TASK-32.1-2・#113〕を実装済み。autoregressive の ONNX 推論は未着手） |
| cli | CLI（7 工程・JSON 入出力契約。REQ-33） | `crates/cli/`（`fandhe-edge-cli`） |
| tui | TUI（REQ-35） | 未確定 |
| mcp | MCP / Codex 連携（REQ-36/37） | 未確定 |
| guard | ガード層（経路・形式・資源・版の検査。REQ-39） | `crates/guard/`（`fandhe-edge-guard`。許可リストによる形式判定・経路の閉じ込め〔`safe_join` 相当。TASK-39.4-1・#158〕・`infer` の `--package`・`onnx_file` への統合〔TASK-39.4-2・#159〕を実装済み） |
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
