# 委譲ルール（作成・編集フェーズ）

## 原則

コードの作成・編集は担当レイヤの builder Agent へ委譲し、main は計画・レビュー・統合に徹する。

## パスベース委譲マッピング（実装）

担当は spec の 6 層（`05-tasks.md`「実装計画の詳細」）で分ける。TASK-15.2（#28）で core・cli の 2 crate、TASK-16.1-1（#38）・TASK-17.1-1（#44）で data crate（検査・group 単位分割ロジック）、TASK-24.1-1（#59）で eval crate（正解率・ラベル別指標・Macro-F1・混同行列）、issue #177 で train crate（学習リクエスト・結果 JSON の型）のパスが確定した。残りの層のパスは crate 追加時に本表へ反映する。

| 層（対象パス） | 委譲先 Agent | model |
| -------------- | ------------ | ----- |
| 共通コア（定義ファイル・選択肢・判定型・正準化ハッシュ。REQ-15）。パス: `crates/core/`（`fandhe-edge-core`） | core-builder | sonnet |
| データ契約（検査・group 分割と凍結・来歴・読み取り専用配置。REQ-16/17/40）。パス: `crates/data/`（`fandhe-edge-data`。検査・group 単位分割・凍結記録・読み取り専用配置・ハッシュ不一致時の停止・分割記録のレコード内容ハッシュ・来歴の記録型と取り込み記録・データ検査との接続を実装済み） | data-builder | sonnet |
| 学習ワーカー（候補学習・選定・選択口・作り直し判定・ジョブ管理。REQ-18〜20/34。Rust / Python）。パス: `trainer/`（Python）・`crates/train/`（`fandhe-edge-train`。Rust 側呼び出し元の 学習リクエスト・結果 JSON の型・子プロセス起動と終了コード写像〔#178〕・探索予算内の候補選定〔TASK-18.1・#83・#84〕・選定結果の有意性判定〔TASK-18.3-1・#87〕・中断ジョブへの再開非提供とやり直し案内〔TASK-34.3・#147〕を実装済み。CLI `train` 工程への配線〔TASK-33.x〕は未着手。issue #177） | trainer-builder | sonnet |
| 評価器（指標・McNemar / Holm・回帰・診断。REQ-21〜27/29）。パス: `crates/eval/`（`fandhe-edge-eval`。正解率・ラベル別指標・Macro-F1・混同行列・McNemar / Holm・下限基準比較・回帰・Wilson 区間と再現性・不変性・推論関数への input のみ受け渡し・凍結 test の 1 回限り適用・校正と棄権を実装済み。対象外ラベル〔TASK-22.2〕・coverage〔TASK-22.3〕・quadrant の multi-item・レポート系〔REQ-29〕・CLI 配線は未実装。TASK-24.1-1・#59） | evaluator-builder | sonnet |
| 成果物・推論 SDK（配布パッケージ・学習非依存の推論ランタイム。REQ-28/30〜32）。パス: `crates/runtime/`（`fandhe-edge-runtime`。単体/バッチ共通の推論経路の継ぎ目〔#117〕・容量計測コア〔TASK-30.1-1・#122〕・容量内訳の JSON 出力接続〔エラー写像は runtime、直列化は cli の output。TASK-30.1-2・#123〕を実装済み。バイト前処理〔TASK-32.1-1・#112〕・C1・C3 の ONNX を読む自作推論バックエンド〔TASK-32.1-2・#113〕を実装済み。書き出し不能構成の除外・理由記録〔TASK-32.2・#114〕も実装済み。autoregressive の ONNX 推論は未着手） | runtime-builder | sonnet |
| 操作アダプター - CLI（7 工程・JSON 入出力契約。REQ-33）。パス: `crates/cli/`（`fandhe-edge-cli`） | adapter-builder | sonnet |
| 操作アダプター - ガード層（REQ-39）。パス: `crates/guard/`（`fandhe-edge-guard`。許可リストによる形式判定・経路の閉じ込め〔`safe_join` 相当。TASK-39.4-1・#158〕・`infer` の `--package`・`onnx_file` への統合〔TASK-39.4-2・#159〕・実行時間上限〔暫定 10 秒。TASK-39.5-1・#170〕を実装済み。TASK-39.2-1・#153）／TUI・MCP（REQ-35/36/37）。パス: 未確定（後続 TASK で追加） | adapter-builder | sonnet |
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
