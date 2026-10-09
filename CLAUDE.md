# CLAUDE.md

## Overview

用途専用の小型ローカル判定モデルを作成・評価するツールの実装リポジトリ。利用者が用意した選択肢（ラベル）・入出力構造・学習データ・独立した評価データから、数十 MB 以下を志向する判定モデルを作り、Claude Code・Codex などから CLI＋JSON（ローカル MCP を含む）で呼び出せるようにする。

- **本リポは public**。仕様・要件定義の SSOT は private リポ [fandhe-edge-model-spec](https://github.com/Fandhe-AI/fandhe-edge-model-spec)（`docs/spec` submodule の `04-requirements.md`。REQ-15〜40）。spec の内容は本リポに載せてよいが、REQ-n・TASK-n を併記して SSOT へ辿れるようにする（[spec-reference](.claude/rules/spec-reference.md)）
- 実装方針の要点は README「実装方針（要点）」を参照（薄い統合ツール・推論は学習に依存しない・モデルの種類の選択口・評価契約・入力表現は byte のみ・ローカル完結・共通コアは Rust 中心）
- 評価契約（評価データの凍結とハッシュ不変・有意性判定・終了コード 7 種など）はコードが破ってはならない不変条件（[evaluation-contract](.claude/rules/evaluation-contract.md)）
- 依存は最小・完全固定・ユーザー承認制。通信を伴う操作もユーザー承認後（[dependency-policy](.claude/rules/dependency-policy.md)）。ライセンスは `MIT OR Apache-2.0`（[licensing](.claude/rules/licensing.md)）
- **実装の着手はユーザーの明示指示を経てから**行う。タスク定義は spec の `05-tasks.md`（TASK-15.1〜40.x。6 層の対応は「実装計画の詳細」節）、マイルストーンは `06-roadmap.md`（M6〜M11。初期スコープは M6〜M10）。v1・v2（REQ-1〜14）は凍結済みで対象外
- 進捗・ステータスは本ファイルに逐次記録しない（Issue で管理する）

## Repository Structure

Rust の crate は `crates/<name>/` の 1 階層に置く。TASK-15.2（#28）で core・cli の 2 crate、TASK-16.1-1（#38）・TASK-17.1-1（#44）で data crate（検査ロジック・group 単位分割ロジック）、TASK-24.1-1（#59）で eval crate（正解率・ラベル別指標・Macro-F1・混同行列）、issue #177 で train crate（学習リクエスト・結果 JSON の Rust 型と共有 fixture による一致確認）を作成済みで、残りの層の crate は後続 TASK で追加する。

```text
fandhe-edge-model/
├── CLAUDE.md                      # Claude 運用方針（本ファイル）
├── README.md                      # 概要・実装方針（要点）・開発環境構築
├── LICENSE-MIT / LICENSE-APACHE   # MIT OR Apache-2.0 デュアルライセンス
├── rust-toolchain.toml            # stable + rustfmt/clippy（単一真実源）
├── .editorconfig                  # インデント・改行・文字コード規約
├── Makefile                       # 開発タスク集約（setup・doctor・lint-docs・fmt・clippy・test・deny・py-*（ruff・pytest）・test-trainer-integration（実 trainer 結合テスト。#258）・check-dependency-approvals（依存の承認台帳の照合。#165）・check-unsafe-allowlist（`allow(unsafe_code)` の出現と許可リストの照合。#335）・check-runtime-linkage（推論の動的リンク確認。#115）・real-machine-check（Mac 実機での動作確認 A〜F の実行と記録。`make ci` には含まれない。#354）・ci・clean。`make help`）
├── commitlint.config.mjs          # commitlint 設定（type を 9 種に限定）
├── .markdownlint.jsonc / .markdownlintignore / .yamllint / .editorconfig-checker.json  # lint-docs 設定
├── skills-lock.json               # 導入スキルのロックファイル
├── deny.toml                       # cargo-deny 設定（`make deny`。ライセンス・ソース・advisory の検査）
├── lefthook.yml                    # git hooks 定義（`make hooks` で導入。未導入の間は動作しない）
├── Cargo.toml                      # workspace 定義（resolver 3・edition 2024・license `MIT OR Apache-2.0`）
├── Cargo.lock                      # 依存ロック
├── dependency-approvals.json       # 依存の承認台帳（`scripts/check_dependency_approvals.py` が manifest・lock と照合。REQ-38・TASK-38.3・#165）
├── unsafe-allowlist.json           # `allow(unsafe_code)` の許可リスト（`scripts/check_unsafe_allowlist.py` が出現箇所と照合。REQ-39・#335）
├── crates/                         # 層に対応する crate 群（残りの層は後続 TASK で追加）
│   ├── core/                       # `fandhe-edge-core`（lib。共通コア。作り直し判定〔`rebuild`。REQ-20・TASK-20.1〜20.3〕を含む。REQ-15）
│   ├── cli/                        # `fandhe-edge-cli`（lib＋bin `fandhe-edge`。操作アダプター - CLI。REQ-33。PoC-26 用の別 bin `fandhe-edge-score`（外部の予測ファイルを凍結 test で採点。7 工程の契約外。REQ-41・#445）。`evaluate` は候補ディレクトリに 1 件ごとの予測 `evaluation_predictions.jsonl` も保存する（#445））
│   ├── data/                       # `fandhe-edge-data`（lib。データ契約 - 検査・group 単位分割・凍結記録・読み取り専用配置・ハッシュ不一致時の停止・分割記録のレコード内容ハッシュ・来歴の記録型と取り込み記録・データ検査との接続を実装済み。REQ-16/17/40。TASK-16.1-1・#38・TASK-17.1-1・#44・TASK-17.2-2・#227・TASK-17.3・#251・TASK-40.1-1・#74・TASK-40.1-2・#75・TASK-40.2・#76）
│   ├── eval/                       # `fandhe-edge-eval`（lib。評価器 - 正解率・ラベル別指標・Macro-F1・混同行列・McNemar / Holm・下限基準比較・回帰・Wilson 区間と再現性・不変性・推論関数への input のみ受け渡し・凍結 test の 1 回限り適用・校正と棄権を実装済み。対象外ラベル〔TASK-22.2〕・coverage〔TASK-22.3〕・quadrant の multi-item・レポート系〔REQ-29〕・CLI 配線は McNemar の下限基準比較のみ接続済み〔#339〕でそれ以外は未実装。REQ-24〜27。TASK-24.1-1・#59）
│   ├── guard/                      # `fandhe-edge-guard`（lib。ガード層。許可リストによるファイル形式判定〔TASK-39.2-1・#153〕・経路の閉じ込め〔`safe_join` 相当。TASK-39.4-1・#158〕・`infer` の `--package`・`onnx_file` への統合〔TASK-39.4-2・#159〕・`kind` の許可リストの `infer` の `artifact.json` への統合〔TASK-39.2-4・#156〕・`kind_version` の許可リスト〔TASK-39.6-1・#174〕・実行時間上限〔暫定 10 秒。TASK-39.5-1・#170〕・RSS 上限〔暫定 2 GiB・RSS ポーリングによる模擬。TASK-39.5-2・#171〕を実装済み。REQ-39）
│   ├── runtime/                    # `fandhe-edge-runtime`（lib。成果物・推論 SDK。単体/バッチ共通の推論経路の継ぎ目〔REQ-28・#117〕と容量計測コア〔構成要素ごとの内訳集計。REQ-30。TASK-30.1-1・#122〕・容量内訳の JSON 出力接続〔TASK-30.1-2・#123。直列化は cli の output〕を実装済み・容量上限の照合〔TASK-30.2・#124。利用者設定の取り込みは未実装〕を実装済み。単体/バッチ/評価器経路の全件一致テストを追加済み・#118。C1・C3 の ONNX 推論〔自作・std のみ。TASK-32.1-2・#113〕を実装済み）
│   └── train/                      # `fandhe-edge-train`（lib。学習ワーカー層 - 学習リクエスト・結果 JSON の型・子プロセス起動と終了コード写像〔#178〕・探索予算内の候補選定〔TASK-18.1・#83・#84〕・選定結果の有意性判定〔TASK-18.3-1・#87〕・中断ジョブへの再開非提供とやり直し案内〔TASK-34.3・#147〕・ジョブ記録の永続化とクラッシュ検出〔TASK-34.2・#146〕・kind 省略時の既定候補の解決〔TASK-19.2・#77〕を実装済み。CLI `train` 工程への配線〔TASK-33.x〕は未着手。REQ-18/19/34/39。issue #177）
├── trainer/                       # 学習ワーカー（Python。uv プロジェクト: pyproject.toml・uv.lock・.python-version）。選択口（TASK-19.1/19.3）と既定候補 C1（バイト n-gram TF-IDF＋ロジスティック回帰。TF-IDF は ONNX グラフ内で計算）・C3（バイト CNN）・autoregressive（バイト単位の小型自己回帰 decoder。対応づけ (b) を ONNX グラフ内で計算。REQ-19b・TASK-19b.1-1・#79）と対応づけ (b) の選択肢 ID 対応づけ・判定不能扱い（TASK-19b.1-2・#234、TASK-19b.2・#250）を実装済み（MLX 学習・ONNX 書き出し）。lifeline による子孫プロセスの終了（`supervisor.py`）・学習直後の validation 予測（`predict.py`）・資源上限（`budget.py`）も実装済み。選定は Rust 側 `crates/train`、作り直し判定（TASK-20.x）は `crates/core` の `rebuild` で実装済みで、CLI `train` 工程への配線（TASK-33.x）は未着手。`tools/poc26/` は PoC-26 の追加学習候補 P の使い捨てスクリプト（Qwen2 トークナイザー `qwen2_tokenizer.py`・モデルと LoRA `qwen2_model.py`・上限つきで安全に読む `safe_io.py`・学習と採点の CLI〔入口 `lora_poc.py`、実装は `cli.py`・`common.py`・`io_records.py`・`assets.py`・`score.py`・`train.py`・`predict.py`・`probe.py`・`export_onnx.py`。サブコマンドは train / predict〔`--warmup N` で先頭 N 件を採点時間の統計から除外〕/ probe / compare-probe / export-onnx / verify-onnx。容量・RSS・p95 の測定と記録は `measure.py`〔2 条件を `/usr/bin/time -l` 下で実行し record.json・record.md を出す。REQ-41・TASK-41.1-8・#393〕。export-onnx は LoRA 統合済みの ONNX 書き出しと演算集計〔TASK-41.1-7・#392〕〕・テスト用の合成データ `synthetic.py`。推論ランタイム・配布パッケージには入らない。REQ-41・TASK-41.1-5・#390）
├── fixtures/                       # 層をまたいで共有するテストデータ（`docs/spec` を参照しない）。`preprocess/byte_encoding_vectors.json`（バイトエンコードのゴールデンベクタ。将来 Rust 推論ランタイムからも参照する契約。Chore #10）・`exitcode/exit_codes.json`（終了コード 7 種の共有 fixture。Rust と学習ワーカーの一致照合。#179）・`train_contract/`（学習リクエスト・結果 JSON・既定候補 `default_candidates.json` の共有 fixture。`crates/train`・`trainer` の一致照合。issue #177）・`score_tolerance/score_sum_tolerance.json`（`SCORE_SUM_TOLERANCE` の共有 fixture。`crates/core`・`crates/data` が独立に持つ同名定数の一致照合。PR #202）・`sandbox_run/`（sandbox 監視チェーンの完走確認用の合成定義・学習データ。REQ-38・TASK-38.1・#161）・`sandbox_run_eval/`（評価データありの完走確認用の合成定義・学習・評価データ。REQ-38・TASK-38.1・#161・#348）・`onnx_parity/`（C1・C3 の ONNX と MLX 内推論の予測ラベル。`trainer/tools/gen_onnx_parity_fixture.py` が生成し、`crates/runtime` の全件一致テストが読む。REQ-32・#113）・`poc26/`（PoC-26 のトークナイザー golden `tokenizer_golden.json`。人間が実機で HF `tokenizers` から生成して置く。生成前は無く、golden 照合は skip。REQ-41・#390）
├── scripts/                       # `cli-infer-noninteractive.sh`（Bash 経由の非対話実行確認スクリプト。REQ-36・TASK-36.1-1・#149。opt-in の環境変数 `FANDHE_EDGE_RECORD_DIR` で実行記録 JSON を保存する。TASK-36.1-2・#150）・`sandbox-run.sh`（sandbox 下で 7 工程を順に実行するスクリプト。実機確認は人の担当。REQ-38・TASK-38.1-1・#162）・`check_dependency_approvals.py`（依存の承認台帳と manifest・lock の照合。標準ライブラリのみ。REQ-38・TASK-38.3・#165）・`check_unsafe_allowlist.py`（`allow(unsafe_code)` の出現と `unsafe-allowlist.json` の照合。標準ライブラリのみ。REQ-39・#335）・`sandbox-monitor.sh`（`log stream` の拒否ログ監視で包み通信拒否 0 件を自動判定するスクリプト。集計は標準ライブラリのみの `sandbox_deny_report.py`。陽性対照〔curl〕を監視窓へ組み込み済み。実機確認は人の担当。REQ-38・TASK-38.1-2・#163・TASK-38.2・#164）・`real-machine-check.sh`（Mac 実機での動作確認 A〜F を実行するスクリプト。引数で項目・回数・条件を制御。人が実行。REQ-21・REQ-28・REQ-30・REQ-31・REQ-32・REQ-33・REQ-39・#354）・`real_machine_check_record.py`（実行結果の JSON 要約・伏せ処理・`record.json`・`record.md` の生成。標準ライブラリのみ）
├── docs/
│   ├── design/                    # 設計・運用手順（`dependency-approval-flow.md`: 依存追加時の明示承認フローとチェックリスト。REQ-38・TASK-38.3・#165、`sandbox-offline-check-procedure.md`: sandbox 下の完走確認の実機手順と記録項目。REQ-38・TASK-38.1・#161、`runtime-batch-mismatch-procedure.md`: 推論の不一致発見時の原因特定・記録手順。REQ-28・TASK-28.2・#119、`claude-code-permission-prompt-procedure.md`: Claude Code の許可操作〔確認画面〕発生時の動作確認手順。人が実機で実行。REQ-36・TASK-36.3・#160、`real-machine-check-procedure.md`: Mac 実機での動作確認〔A〜F〕の手順と記録簿。人が実行。REQ-21・REQ-28・REQ-30・REQ-31・REQ-32・REQ-33・REQ-39・#354、`poc26-preregistration.md`: PoC-26〔Playwright MCP のツール選択〕の事前登録。Playwright MCP の版・ラベル集合・候補・指標・有意性の判定方法を評価前に固定。確定はオーナー。REQ-41・TASK-41.1・M12・#386、`poc26-preregistration-addendum-1.md`: 事前登録の追補 1。候補 P の LoRA・採点・system プロンプト・chat template 照合の固定値と実重み確認の手順・記録欄。REQ-41・TASK-41.1-5・#390、`poc26-tokenizer-golden-procedure.md`: トークナイザー golden 照合の実機手順。人が実行。REQ-41・TASK-41.1-5・#390、`poc26-measure-procedure.md`: 候補 P の容量・RSS・推論のみ p95 の実機測定手順と記録欄。人が実行。REQ-41・TASK-41.1-8・#393、`poc26-result.md`: PoC-26 の結果と REQ-41 の確定案〔扱い・受け入れ基準・初期アーキテクチャ〕。確定はオーナー。REQ-41・TASK-41.1-9・#394）
│   └── spec/                      # fandhe-edge-model-spec submodule（private・要アクセス権）
├── .github/workflows/             # ai-review・ci・python-ci・update-external（稼働）/ release（発火条件無効化中）
├── .agents/skills/                # npx skills add の導入実体
└── .claude/
    ├── agents/                    # カテゴリ別 subagent 定義（research / implement / testing / quality / docs）
    ├── rules/                     # 運用ルール
    ├── skills/                    # 導入スキル（.agents/skills への symlink）
    ├── workflows/                 # implement-issue-tree.js（相対 symlink）
    └── settings.json              # SessionStart / PostToolUse hooks
```

## 委譲方針（必読）

main セッションはオーケストレーションに徹し、調査・実装・レビューは subagent へ委譲してコンテキスト消費を抑える。詳細は [delegation](.claude/rules/delegation.md)（調査）・[delegation-impl](.claude/rules/delegation-impl.md)（実装）を参照。`docs/spec` は 1 ファイルが数百 KB 規模のため main で通読せず、explorer に REQ-n / TASK-n を指定して抜粋させる。

### パスベース切り替え表

担当は spec の 6 層（`05-tasks.md`「実装計画の詳細」）で分ける。以下は確定済みのパスを示す。未確定の層は後続 TASK で crate 追加時に更新する。

| 対象 | パス | 調査 | 作成・編集 |
| ---- | ---- | ---- | ---------- |
| 共通コア（定義ファイル・選択肢・判定型・正準化ハッシュ。REQ-15） | `crates/core/`（`fandhe-edge-core`） | explorer | core-builder |
| データ契約（検査・group 分割と凍結・来歴・読み取り専用配置。REQ-16/17/40） | `crates/data/`（`fandhe-edge-data`。検査・group 単位分割・凍結記録・読み取り専用配置・ハッシュ不一致時の停止・分割記録のレコード内容ハッシュ・来歴の記録型と取り込み記録・データ検査との接続を実装済み。TASK-16.1-1・#38・TASK-17.1-1・#44・TASK-17.2-2・#227・TASK-17.3・#251・TASK-40.1-1・#74・TASK-40.1-2・#75・TASK-40.2・#76） | explorer | data-builder |
| 学習ワーカー（候補学習・選定・選択口・作り直し判定・ジョブ管理。REQ-18〜20/34） | `trainer/`（Python）・`crates/train/`（`fandhe-edge-train`。Rust 側呼び出し元の 学習リクエスト・結果 JSON の型・子プロセス起動と終了コード写像〔#178〕・探索予算内の候補選定〔TASK-18.1・#83・#84〕・選定結果の有意性判定〔TASK-18.3-1・#87〕・中断ジョブへの再開非提供とやり直し案内〔TASK-34.3・#147〕・ジョブ記録の永続化とクラッシュ検出〔TASK-34.2・#146〕・kind 省略時の既定候補の解決〔TASK-19.2・#77〕を実装済み。CLI `train` 工程への配線〔TASK-33.x〕は未着手。issue #177） | explorer | trainer-builder |
| 評価器（指標・McNemar / Holm・回帰・診断。REQ-21〜27/29） | `crates/eval/`（`fandhe-edge-eval`。正解率・ラベル別指標・Macro-F1・混同行列・McNemar / Holm・下限基準比較・回帰・Wilson 区間と再現性・不変性・推論関数への input のみ受け渡し・凍結 test の 1 回限り適用・校正と棄権を実装済み。対象外ラベル〔TASK-22.2〕・coverage〔TASK-22.3〕・quadrant の multi-item・レポート系〔REQ-29〕・CLI 配線は McNemar の下限基準比較のみ接続済み〔#339〕でそれ以外は未実装。TASK-24.1-1・#59） | explorer | evaluator-builder |
| 成果物・推論 SDK（配布パッケージ・学習非依存の推論ランタイム。REQ-28/30〜32） | `crates/runtime/`（`fandhe-edge-runtime`。単体/バッチ共通の推論経路の継ぎ目〔#117〕・容量計測コア〔TASK-30.1-1・#122〕・容量内訳の JSON 出力接続〔エラー写像は runtime、直列化は cli の output。TASK-30.1-2・#123〕・容量上限の照合〔TASK-30.2・#124。利用者設定の取り込みは未実装〕を実装済み。バイト前処理〔TASK-32.1-1・#112〕・C1・C3 の ONNX を読む自作推論バックエンド〔TASK-32.1-2・#113〕を実装済み。書き出し不能構成の除外・理由記録〔TASK-32.2・#114〕・語彙ファイル超過構成の除外記録〔TASK-30.3・#125〕も実装済み。autoregressive の ONNX 推論は未着手。全件一致テスト #118） | explorer | runtime-builder |
| 操作アダプター - CLI（7 工程・JSON 入出力契約。REQ-33） | `crates/cli/`（`fandhe-edge-cli`） | explorer | adapter-builder |
| 操作アダプター - ガード層（REQ-39） | `crates/guard/`（`fandhe-edge-guard`。許可リストによる形式判定・経路の閉じ込め〔`safe_join` 相当。TASK-39.4-1・#158〕・`infer` の `--package`・`onnx_file` への統合〔TASK-39.4-2・#159〕・`kind` の `infer` への統合〔TASK-39.2-4・#156〕・`kind_version` の許可リスト〔TASK-39.6-1・#174〕・実行時間上限〔暫定 10 秒。TASK-39.5-1・#170〕・RSS 上限〔暫定 2 GiB・RSS ポーリングによる模擬。TASK-39.5-2・#171〕を実装済み。TASK-39.2-1・#153） | explorer | adapter-builder |
| 操作アダプター - TUI・MCP（REQ-35〜37） | 未確定 | explorer | adapter-builder |
| `Cargo.toml`・`trainer/pyproject.toml`・`trainer/uv.lock`・CI・`deny.toml`・`Makefile`・`lefthook.yml`・lint 設定・`scripts/` | — | explorer | infra-builder |
| `docs/spec/`（private） | — | explorer | 変更しない（spec リポ側で管理） |
| 外部仕様（ONNX / `ort`・MLX・candle / burn・MCP・統計手法）・依存候補 | — | reference-researcher | — |
| テスト・lint | — | test-runner / linter | — |
| ドキュメント・`AGENTS.md`・`.claude/`（agents・rules・settings.json） | — | explorer | docs-writer |

### model 配分表

| 用途 | model |
| ---- | ----- |
| 複雑な横断判断・アーキテクチャ設計（層の境界・評価契約・選択口・JSON 入出力契約・配布パッケージ形式） | opus または fable（fable は特に大規模設計・横断判断の最上位 tier） |
| 調査・生成・実装・レビュー | sonnet |
| 機械的集計・lint・ドキュメント更新 | haiku |

## Sub-agents

| カテゴリ | subagent_type | model | 役割 |
| -------- | ------------- | ----- | ---- |
| research | explorer | sonnet | コードベース・spec 横断調査（spec は抜粋参照） |
| research | reference-researcher | sonnet | 外部仕様・依存候補・統計手法の調査 |
| implement | core-builder | sonnet | 共通コア（定義ファイル・判定型・正準化ハッシュ） |
| implement | data-builder | sonnet | データ契約（検査・分割と凍結・来歴・読み取り専用配置） |
| implement | trainer-builder | sonnet | 学習ワーカー・選択口・ジョブ管理（Rust / Python） |
| implement | evaluator-builder | sonnet | 評価器（指標・有意性・再現性・独立性・回帰） |
| implement | runtime-builder | sonnet | 配布パッケージ・学習非依存の推論ランタイム |
| implement | adapter-builder | sonnet | CLI 7 工程・TUI・MCP / Codex・ガード層 |
| implement | infra-builder | sonnet | workspace・CI・deny・Makefile・lefthook・lint 設定・scripts |
| testing | test-runner | sonnet | テスト実行と失敗解析（評価契約・決定性・実機前提の区別） |
| quality | reviewer | sonnet | 層の境界・評価契約・入出力契約・規約準拠のレビュー |
| quality | security-auditor | sonnet | ガード層・逆シリアル化・ローカル完結・MCP・OWASP 監査 |
| quality | linter | haiku | rustfmt / clippy / cargo deny / lint-docs（ruff）の機械的確認 |
| docs | docs-writer | haiku | README・CLAUDE.md・AGENTS.md・docs/design・.claude/ 更新 |

## Rules

| ファイル | 内容 |
| -------- | ---- |
| [delegation.md](.claude/rules/delegation.md) | 調査フェーズの委譲原則・パスベース切り替え・spec の抜粋参照 |
| [delegation-impl.md](.claude/rules/delegation-impl.md) | 実装フェーズの委譲マッピング（6 層）・標準フロー・着手条件（明示指示・人間担当タスク・通信・GPU） |
| [coding-rust.md](.claude/rules/coding-rust.md) | Rust 規約（層の境界・推論の学習非依存・型設計・外部入力・数値と決定性・unsafe・テスト） |
| [coding-python.md](.claude/rules/coding-python.md) | Python 規約（学習ワーカー限定・Rust との JSON 境界・ruff / pytest / uv・安全でない読み込みの禁止） |
| [security.md](.claude/rules/security.md) | 秘密情報・ガード層（REQ-39）・ローカル完結（REQ-38）・MCP・OWASP Top 10 |
| [japanese-style.md](.claude/rules/japanese-style.md) | 日本語出力スタイル |
| [conventional-commits.md](.claude/rules/conventional-commits.md) | Conventional Commits 詳細規約（type / 6 層に対応する scope 一覧） |
| [code-comment-style.md](.claude/rules/code-comment-style.md) | コメント規約（役割・呼び出し文脈・層間契約・評価契約・スタブの将来仕様） |
| [out-of-scope-tracking.md](.claude/rules/out-of-scope-tracking.md) | スコープ外事項の Issue 追跡フロー |
| [spec-reference.md](.claude/rules/spec-reference.md) | **リポ固有**: spec（SSOT）の参照・ID 併記・編集禁止・証拠種別の維持 |
| [evaluation-contract.md](.claude/rules/evaluation-contract.md) | **リポ固有**: 評価契約の不変条件（凍結・ハッシュ・有意性・終了コード・推論一致・決定性） |
| [dependency-policy.md](.claude/rules/dependency-policy.md) | **リポ固有**: 依存最小・推論の学習非依存・完全固定・ユーザー承認制・通信の承認 |
| [licensing.md](.claude/rules/licensing.md) | **リポ固有**: MIT OR Apache-2.0・コピーレフト禁止・重み / データのライセンス確認 |
| [ci.md](.claude/rules/ci.md) | **リポ固有**: ローカルゲート・CI 構成（有効化条件）・実機前提テストの扱い |

## Current Skills

`npx skills add`（Fandhe-AI/agent-cli-skills・Fandhe-AI/agent-reference-skills）で導入済み。ロックは `skills-lock.json`。

- **ワークフロー系**: create-commit / create-pr / create-issue / create-issue-tree / create-plan / implement-issue / implement-issue-tree / implement-review / implement-review-pr / update-issue-tree / update-docs / comment-code
- **メンテ系**: init-claude / update-claude / contribute-skill / setup-repo-guards
- **リファレンス系**: rust / make / github-docs / commitlint / lefthook / editorconfig / anthropic-claude-code / openai-codex / fandhe-ai / fandhe-backend / fandhe-db / fandhe-frontend

## Conventions

- **ローカル検証**: `make fmt-check`・`make lint`・`make test`（学習ワーカーは `make py-ci`。まとめて `make ci`）を通してからコミットする（[ci](.claude/rules/ci.md)）。環境診断は `make doctor`。Makefile の `skip:` を検証済みと扱わない。ビルド・テストは `docs/spec` 抜きで成立させる
- **日本語**: やりとり・報告・コミット説明文・コード内コメントは日本語（プログラム出力文字列・JSON は英語）
- **Conventional Commits**: `--no-verify` 禁止
- **セキュリティレビュー**: PR 作成前に OWASP Top 10＋ガード層（REQ-39）・ローカル完結（REQ-38）を確認
- **ユーザー承認フロー**: 実装の着手 / 依存の追加・更新 / 通信を伴う操作 / 重み・データの取得 / `unsafe` の新規追加 / ライセンス判断 / 評価契約・入出力契約の変更 / GPU を長時間占有する学習 / Issue 起票 / 既存ファイル上書き / implement-issue の実装開始（計画承認後）は必ずユーザー承認を経る
- **spec 参照**: `docs/spec` の内容を引用・要約する際は REQ-n・TASK-n・M-n・PoC-n を併記する。`docs/spec` は本リポから編集しない
- **implement-issue-tree**: `.claude/workflows/implement-issue-tree.js`（相対 symlink）を named workflow として利用できる

## hooks（settings.json）

- **SessionStart**: 日本語・委譲・Conventional Commits・`--no-verify` 禁止・spec 参照（ID 併記・抜粋参照）・評価契約・推論の学習非依存・依存 / 通信の承認制・実装着手条件・GPU 占有のリマインダーを表示
- **PostToolUse**（Edit|Write）: `*.rs` 編集後に rustfmt で自動整形（edition はルート `Cargo.toml` から取得し、未作成時は 2024）。`*.py` 編集後に `ruff format` で自動整形。jq / rustfmt / ruff 未導入時は何もしない。整形失敗で作業を止めない
