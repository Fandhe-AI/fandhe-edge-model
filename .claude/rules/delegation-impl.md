# 委譲ルール（作成・編集フェーズ）

## 原則

コードの作成・編集は担当レイヤの builder Agent へ委譲し、main は計画・レビュー・統合に徹する。

## パスベース委譲マッピング（実装）

担当は spec の 6 層（`05-tasks.md`「実装計画の詳細」）で分ける。TASK-15.2（#28）で core・cli の 2 crate、TASK-16.1-1（#38）で data crate（検査ロジックのみ）のパスが確定した。残りの層のパスは crate 追加時に本表へ反映する。

| 層（対象パス） | 委譲先 Agent | model |
| -------------- | ------------ | ----- |
| 共通コア（定義ファイル・選択肢・判定型・正準化ハッシュ。REQ-15）。パス: `crates/core/`（`fandhe-edge-core`） | core-builder | sonnet |
| データ契約（検査・group 分割と凍結・来歴・読み取り専用配置。REQ-16/17/40）。パス: `crates/data/`（`fandhe-edge-data`。検査のみ実装済み。group 分割・凍結・来歴・読み取り専用配置は未着手） | data-builder | sonnet |
| 学習ワーカー（候補学習・選定・選択口・作り直し判定・ジョブ管理。REQ-18〜20/34。Rust / Python）。パス: `trainer/`（Python） | trainer-builder | sonnet |
| 評価器（指標・McNemar / Holm・回帰・診断。REQ-21〜27/29）。パス: 未確定（後続 TASK で追加） | evaluator-builder | sonnet |
| 成果物・推論 SDK（配布パッケージ・学習非依存の推論ランタイム。REQ-28/30〜32）。パス: 未確定（後続 TASK で追加） | runtime-builder | sonnet |
| 操作アダプター - CLI（7 工程・JSON 入出力契約。REQ-33）。パス: `crates/cli/`（`fandhe-edge-cli`） | adapter-builder | sonnet |
| 操作アダプター - TUI・MCP・ガード層（REQ-35/36/37/39）。パス: 未確定（後続 TASK で追加） | adapter-builder | sonnet |
| ルート `Cargo.toml`・`pyproject.toml`・`.github/workflows/`・`deny.toml`・`Makefile`・`lefthook.yml`・lint 設定・`scripts/` | infra-builder | sonnet |
| テスト実行・失敗解析（`make test` / `make lint`） | test-runner | sonnet |
| コードレビュー | reviewer | sonnet |
| セキュリティ監査 | security-auditor | sonnet |
| lint・整形の機械的確認 | linter | haiku |
| README・CLAUDE.md・`AGENTS.md`・`docs/design/`・`.claude/`（agents・rules・settings.json）更新 | docs-writer | haiku |

複数層に跨る変更は層ごとに builder を分けて委譲する（独立していれば並列可）。
層の境界・定義ファイル / 判定型のスキーマ・JSON 入出力契約・終了コード・配布パッケージ形式・評価契約（[evaluation-contract](./evaluation-contract.md)）の設計変更は builder に任せず main（opus / fable）で設計してから委譲する。

## 実装フローの標準形

1. 計画（main。必要に応じて explorer で事前調査）
2. 実装（builder へ委譲）
3. 検証（test-runner → 失敗があれば builder へ差し戻し）
4. レビュー（reviewer / security-auditor。外部入力・ファイル経路・モデルファイル読み込み・子プロセス起動・MCP に触れる変更は security-auditor 必須）
5. コミット（create-commit スキル。Conventional Commits・`--no-verify` 禁止）

## 着手条件（本リポ固有）

- **実装の着手はユーザーの明示指示を経てから行う**（ロードマップ上の着手判定とは別に、個別の開始指示を待つ）
- spec のタスク定義で担当が「人間」「共同」のタスク（実機測定・評価データの用意・技術選定・ライセンス判断・判定等）には Agent から単独で着手しない。準備作業（計測スクリプト作成等）に留め、判断事項はユーザーへ報告する
- 依存（`Cargo.toml`・`pyproject.toml` の dependencies）の追加・更新は builder に委譲せず、必ずユーザー承認を経る（[dependency-policy](./dependency-policy.md)）
- 通信を伴う操作（依存・モデル重みの取得等）はユーザーの明示承認を経てから実行する（REQ-38）
- GPU を長時間占有する学習・計測は、ユーザーの明示指示なしに実行しない（GPU は 1 台で直列運用。M7）
- スコープ外の発見事項は放置せず [out-of-scope-tracking](./out-of-scope-tracking.md) に従い追跡する
