# AGENTS.md

## 本書の用途

本書は `.github/workflows/ai-review.yml`（`Fandhe-AI/actions` の `ai-review` reusable workflow を呼ぶ wrapper。check 名 `codex / preflight`・`codex / review`・`codex / post_feedback`）が Codex による PR 自動レビューの基準として読む、リポジトリ固有のレビュー観点集である。

- Codex の既定 prompt は **PR の base コミットの本書** を読む。そのため本書への変更は、当該 PR のレビューには反映されず、**マージ後の次の PR から実効** になる
- 本書は日本語で記述する。プログラムの出力文字列（エラーメッセージ・ログ・CLI 出力・JSON）や識別子・コマンドは原語（英語）のままでよい
- `docs/spec`（`fandhe-edge-model-spec` submodule）は private であり、レビュー実行環境からは読めない前提とする。レビューでは spec 本文との一致を判定材料にせず、**REQ-n・TASK-n・M-n・PoC-n の併記があるか** という、diff だけで確認できる観点に限定する。ID が欠けた spec 由来の変更は「spec 参照規約」観点（P1）として指摘する
- 実装は未着手であり、crate・ディレクトリ名は TASK-15.2（雛形作成）で確定する。本書の層区分は spec の 6 層（共通コア・データ契約・学習ワーカー・評価器・成果物 / 推論 SDK・操作アダプター）に従う
- 本書の各観点の詳細な根拠は `.claude/rules/`（`security.md`・`evaluation-contract.md`・`coding-rust.md`・`coding-python.md`・`dependency-policy.md`・`licensing.md`・`ci.md` 等）にある。本書と `.claude/rules/` が食い違う場合は、本書の優先度判定を正としつつ、食い違い自体を P2 として指摘する

## 優先度定義

| 優先度 | 意味 | 判断基準 | 扱い |
| ---- | ---- | ---- | ---- |
| P0 | マージブロック | セキュリティ・ライセンス・評価契約・アーキテクチャ根幹の違反、または回帰検出の後退（テストの skip・ignore・許容差の拡大等）・実装済みを装う偽装など品質ゲートの破壊 | 修正するまでマージしない |
| P1 | 強く推奨 | 規約違反・入出力契約の無承認変更・保守性の重大な低下 | 未対応のままマージ不可（ai-review の codex ジョブが fail する）。対応不要の場合はスレッドで理由を明示して resolve する |
| P2 | 提案 | 改善提案 | CI を fail させない。次回以降の改善提案として記録すればよい |

## ビルド・テスト・回帰確認コマンド

`make` ターゲットを正とする（括弧内に実行される cargo コマンドを併記する）。

```bash
make fmt-check   # cargo fmt --all --check
make lint        # cargo clippy --workspace --all-targets -- -D warnings（既定 feature）
make test        # cargo test --workspace（既定 feature）
make deny        # cargo deny --locked check advisories bans licenses sources
make ci          # lint-docs + check-workspace-manifest + 上記 4 つを一括実行
make doctor      # 環境診断のみ（何も導入しない）
```

- `Cargo.toml`（workspace）が未作成の間、`fmt`/`lint`/`test` は対象がなく実行できない。`Cargo.toml` と `crates/*/Cargo.toml`（メンバー crate）の両方が揃うまで Makefile 側でこれらは `skip:` を表示してスキップされる。`deny` は加えて `deny.toml` の存在を要する。`skip:` を「検証済み」として扱う記述・変更は P1
- workspace 作成後の PR からは、PR 本文にこれらのコマンドの実行結果が記載されているか（同じ PR で CI 設定を変更する場合はその diff にこれらのコマンドが含まれているか）を確認する。本節の未達は、個別に優先度を明記した項目を除き既定で P1 とする
- clippy 警告は 0 件を維持する。理由コメントなしで `#[allow(...)]` により警告を握りつぶす差分は P1。crate・モジュール全体に及ぶ広範な `#![allow(...)]` や、`unsafe` 関連 lint・外部入力の検証を隠す lint（`clippy::unwrap_used`・`clippy::expect_used`・`clippy::indexing_slicing` 等）の外部入力経路での抑止は、理由コメントの有無を問わず P0
- テストの skip・ignore・アサーション弱体化・浮動小数の許容差の拡大で CI を通す差分は P0（回帰検出の後退を招くため）
- 学習ワーカーを Python で実装する場合、ruff（`ruff check`・`ruff format --check`）・pytest の Makefile ターゲットと CI への組み込みを伴っているか（`.claude/rules/coding-python.md`）。Python コードを追加しながら検証経路が無い差分は P1

### 実機前提テスト

- GPU（Metal / MLX）・実機での性能測定（p95 レイテンシ・容量）・sandbox 下の通信 0 件確認（REQ-38）・長時間学習を要するテストは、GitHub ホステッド runner で実行できないため既定のテスト集合から明示的に分離されているか確認する。分離の仕組み・実行コマンドは該当タスクで決め、本書に追記する
- 分離したテストに理由（必要な環境）と REQ-n が記され、実機での実行結果が **証拠の種別（テストハーネス / 模擬 / 推定 / 実機）付きで** PR に記録されているか確認する
- 既定のテスト集合で動くはずのテストを、CI 通過のために実機前提テストへ移す差分は P0
- 実機での測定・判定が「人間」担当のタスクを、計測スクリプト準備を超えて Agent が単独で完了扱いにしていないか確認する

## レビュー観点

### セキュリティ（P0 中心。`.claude/rules/security.md`・`.claude/rules/dependency-policy.md`・`.claude/rules/licensing.md`）

| 観点 | 確認内容 | 優先度 |
| ---- | ---- | ---- |
| 秘密情報 | 実トークン・API キー・接続資格情報がコード・テスト・fixture・ドキュメント・hooks・`settings.json` に含まれていないか | P0 |
| データの混入 | 学習・評価データの本文（個人情報・機密情報を含みうる）がログ・エラーメッセージ・fixture・PR 本文へ転記されていないか（件数・ハッシュ・行番号で示しているか） | P0 |
| ガード層: 経路の閉じ込め | CLI・MCP・TUI から渡されるパスについて、`../`・絶対パス・symlink によるルート外参照を、正規化後にルート配下であることの確認（`safe_join` 相当）で拒否しているか（REQ-39・PoC-20） | P0 |
| ガード層: 形式の許可制 | 読み込むモデル・データの形式を許可リストで判定しているか。pickle 偽装・非 ONNX ファイル・非対応の `kind` を拒否しているか（REQ-39） | P0 |
| ガード層: 資源の上限 | 1 件あたりの処理時間・RSS・ファイルサイズ（読み込み前に確認）に上限があるか。長さ・件数を上限検証してからアロケーションに使っているか。子プロセス・長時間処理にタイムアウトがあるか（REQ-39） | P0 |
| ガード層: 完全性と版 | 配布パッケージ・モデルを sha256 で検証し、不一致なら拒否しているか。読み取り専用配置への書き込みを拒否しているか。`kind_version` を許可リストで検証しているか（REQ-39） | P0 |
| ガード層の迂回 | TUI・MCP・内部 API から、CLI と同じガード層を通らずに処理へ到達する経路を追加していないか | P0 |
| 安全でない逆シリアル化・コード実行 | `pickle`・`torch.load`（`weights_only=False`）・`eval` / `exec`・非 safe な `yaml.load` を使っていないか。`subprocess` を `shell=True` で呼んでいないか | P0 |
| インジェクション | CLI 引数・定義ファイル・MCP リクエストを子プロセス起動・パス・シェルへ未検証で連結していないか | P0 |
| ローカル完結 | 推論・学習・評価の実行時に本ツール起因の通信（テレメトリ・自動更新・重み / データの自動ダウンロード）を発生させていないか（REQ-38） | P0 |
| MCP の公開範囲 | MCP サーバーをネットワークインターフェースへ公開していないか（ローカル〔stdio 等〕に限る）。学習・削除など副作用の大きい操作を推論・参照系と分けて公開しているか（REQ-36・REQ-37） | P0 |
| 外部入力の検証 | 定義ファイル・学習 / 評価データ・モデルファイル・CLI 引数・MCP リクエスト・学習ワーカーの出力の経路で `unwrap`・`expect`・添字アクセス（`[]`）を使わず、`get()`・`try_into()`・checked 演算で処理しているか | P0 |
| `unsafe`/FFI | `unsafe` の新規追加はユーザー承認済みか（PR 本文に承認の記録があるかで確認）。`unsafe` ブロックに `// SAFETY:` コメント（理由・維持すべき不変条件）があるか | P0 |
| 依存の追加・更新 | Rust 依存が `=x.y.z`、Python 依存が `==x.y.z` の完全固定か（lock ファイルのコミットを含む）。Rust は `[workspace.dependencies]` に集約されているか。ユーザー承認（名前・バージョン・目的・配置する層、ライセンス、メンテナンス状況、推移的依存・ネイティブビルドの有無、配布サイズ・推論レイテンシへの影響）を経ているか（PR 本文に承認の記録があるかで確認）。git 依存・crates.io 以外のレジストリ・バージョン無指定の依存を追加していないか | P0 |
| ライセンス | crate・パッケージのライセンス表記が `MIT OR Apache-2.0` になっているか。GPL / LGPL / AGPL・MPL-2.0 等のコピーレフト系の依存を導入していないか | P0 |
| 重み・データのライセンス | 事前学習済み重み・語彙・データセットの取得・同梱・配布パッケージへの埋め込みについて、ライセンス・再配布可否をユーザーへ確認しているか（PR 本文に記録があるかで確認） | P0 |

### アーキテクチャ・設計整合（`.claude/rules/coding-rust.md`・`CLAUDE.md`）

| 観点 | 確認内容 | 優先度 |
| ---- | ---- | ---- |
| 層構成 | 変更が spec の 6 層（共通コア: 定義ファイル・判定型・正準化ハッシュ／データ契約: 検査・分割と凍結・来歴／学習ワーカー: 候補学習・選定・選択口・ジョブ管理／評価器: 指標・有意性・回帰・診断／成果物・推論 SDK: 配布パッケージ・推論ランタイム／操作アダプター: CLI・TUI・MCP・ガード層）の想定責務に収まっているか | P1 |
| 依存方向 | 層間の一方向依存が保たれ循環がないか。共通コアが他のどの層にも依存していないか。複数 crate が共有する型が下位 crate に置かれているか | P0 |
| 推論の学習非依存 | 推論ランタイム・配布パッケージの経路に、学習側の依存（Python・MLX・学習用 crate）を持ち込んでいないか（REQ-32。Python・MLX が PATH に無い環境でも推論が成功すること） | P0 |
| 評価器の一元化 | 評価ロジック（指標・有意性判定）を評価器（TASK-24.1）以外の層・学習ワーカー側に再実装していないか | P0 |
| アダプターの薄さ | CLI・TUI・MCP に業務ロジックを置いていないか。TUI・MCP が CLI と同じ入出力契約を使っているか（REQ-33・REQ-36・REQ-37） | P1 |
| 正準化ハッシュの一元化 | 正準化・ハッシュ計算の規則が 1 箇所に集約され、他層で独自に計算していないか（REQ-15） | P1 |
| `docs/spec` 非依存ビルド | コード・`build.rs`・テスト・学習スクリプトが `docs/spec` 配下を読み込んでいないか（`docs/spec` 抜きでビルド・テストが成立するか） | P0 |
| spec 参照規約 | spec の内容を引用・要約する箇所に REQ-n・TASK-n・M-n・PoC-n が併記されているか（spec ファイルの丸ごとコピーになっていないか）。`docs/spec` 配下自体を本リポ側で編集していないか | P1 |
| エラーハンドリング | ライブラリコードが `Result` を返し panic させていないか | P0 |

### 評価契約（`.claude/rules/evaluation-contract.md`）

| 観点 | 確認内容 | 優先度 |
| ---- | ---- | ---- |
| 分割と凍結 | 分割が group 単位で、seed・分割規則・各分割のハッシュを記録しているか。評価データのハッシュが記録と一致しない場合に処理を停止しているか（fail-closed）。評価データが無い場合に `status:"skipped"`・exit 0 とし、評価済みを装っていないか（REQ-17） | P0 |
| 評価の独立性 | 評価の前後でモデル（重み・語彙・校正・しきい値）と評価データのハッシュが一致することを確認しているか。推論関数に `input` 以外（正解ラベル・分割情報・評価データの統計）を渡していないか（REQ-27） | P0 |
| 最終 test の 1 回適用 | 凍結した最終 test を 2 回以上適用できる経路、最終 test の結果を見て候補・しきい値を選び直す経路を作っていないか。選定が validation で行われているか（REQ-27） | P0 |
| 有意性判定 | 下限基準（majority）との比較が McNemar 検定（p < 0.05）か。複数候補の比較で Holm 補正を行っているか。件数不足を「判定不能」として返し、合格扱いにしていないか（REQ-25） | P0 |
| 指標の扱い | 分母が 0 の指標を `null` とし平均から除外しているか（0 や 1 で埋めていないか）。型の正しさと意味の正しさを別々に数えているか（REQ-24） | P1 |
| 推論の一致 | 1 件ずつの推論とバッチ推論の結果が全件一致することを機械照合するテストがあるか（REQ-28） | P0 |
| 入出力契約 | 終了コードが 7 種（`ok`=0・`judged_fail`=10・`out_of_scope`=11・`pending`=12・`limit_exceeded`=20・`invalid_input`=64・`runtime_error`=70）から外れていないか。CLI の出力が 1 呼び出し 1 JSON（例外は `infer --input-file` の 1 行 1 JSON のみ）か。エラーが機械可読な `code` / `message` の JSON か（REQ-21・REQ-33） | P1 |
| 契約の破壊的変更 | JSON 出力・終了コード・引数・配布パッケージ形式を変える差分が、`!` と `BREAKING CHANGE:` 付きのコミットで、PR 本文にユーザー承認の記録があるか | P1 |
| 決定性 | 乱数の seed を引数で受け取り、グローバルな乱数状態に依存していないか。浮動小数を `==` で比較せず許容差（1e-9）を明示しているか。GPU 学習の非決定性をテストの許容幅拡大で吸収していないか | P1 |
| 証拠の種別 | 測定値・判定を記録する箇所（ドキュメント・PR 本文・テスト名）に証拠の種別（テストハーネス / 模擬 / 推定 / 実機）が明記されているか | P1 |

### 保守性・自己補修性

| 観点 | 確認内容 | 優先度 |
| ---- | ---- | ---- |
| 構造化された戻り値 | 公開 API の戻り値・判定結果・状態が、将来拡張できる構造を持つ型（enum 等）か（真偽値・フラットな文字列で済ませていないか） | P1 |
| スタブの明示 | 未実装・簡易実装箇所が「実装済みを装って」いないか。ドキュメントコメントに将来仕様と対応する REQ-n が明記されているか | P0 |
| コメント規約 | crate・モジュールの入口に `//!`、公開 API に `///`（Python は docstring）で役割要約があるか。呼び出し元・呼び出し先の文脈、他層との契約が書かれているか。逐語説明や spec 本文の長い引用になっていないか（`.claude/rules/code-comment-style.md`） | P2 |
| テストと REQ-n の対応 | 挙動が REQ-n に対応づけてテストされ、テスト名またはドキュメントコメントに ID が記されているか。期待値が具体値で書かれているか（真偽値のみの assert に頼っていないか）。統計計算に既知の数値例による具体値の assert があるか | P1 |
| スコープ外事項の追跡 | 実装・レビュー中に見つかったスコープ外の事項が、当該 PR に混入せず Issue 追跡へ切り出されているか（`.claude/rules/out-of-scope-tracking.md`） | P1 |

### 規約（表記・コミット）

| 観点 | 確認内容 | 優先度 |
| ---- | ---- | ---- |
| 日本語規約 | コード内コメント・ドキュメントが日本語で書かれているか。エラーメッセージ・ログ・CLI 出力・JSON・MCP レスポンス等プログラムの出力文字列が英語になっているか（`.claude/rules/japanese-style.md`） | P2 |
| Conventional Commits | PR タイトル（および PR 本文に見えるコミットメッセージ）が Conventional Commits 形式か。type/scope が英語、説明文が日本語になっているか（`.claude/rules/conventional-commits.md`） | P2 |

### CI・ワークフロー

| 観点 | 確認内容 | 優先度 |
| ---- | ---- | ---- |
| 第三者 action の固定 | サードパーティ action はコミット SHA で固定されているか | P0 |
| `Fandhe-AI/actions` の例外 | `Fandhe-AI/actions` は組織内（first-party）の上流リポジトリであり、上記「第三者 action の固定」の対象ではない。reusable workflow への参照は組織方針（2026-08-18 オーナー判断）により可変タグ `@latest` の使用が認められている。`@latest` の使用・SHA pin への置き換えを求める指摘をしない | 指摘しない（例外） |
| runner 方針 | public リポジトリのため既定は GitHub ホステッドランナー。self-hosted の使用が許可されるのは `ai-review.yml` の `codex / review` ジョブ（組織承認済み例外）のみで、`codex / preflight`・`codex / post_feedback` を含む他ジョブ・他 workflow は GitHub ホステッドランナーになっているか | P0 |
| permissions | ワークフロー・ジョブの `permissions` が最小権限で明示されているか | P0 |
| secrets の扱い | secrets が `pull_request` イベントのログへ出力されていないか。`${{ }}` を `run` へ直接埋め込まず `env` 経由で渡しているか | P0 |
| `ci.yml` の現状 | `ci.yml` は準備が整うまで発火条件を無効化中（`workflow_dispatch` のみ）であり、workspace（`Cargo.toml`）・メンバー crate・`deny.toml` の作成後に `pull_request` / `push` を有効化し、ruleset の必須チェックへ `ci-complete` を登録する設計であることを踏まえ、この無効化自体を指摘しない | 指摘しない（既知の暫定状態） |
| `ci.yml` への変更 | `ci.yml` を変更する差分では、3 OS matrix（Linux・macOS・Windows）を維持しているか、集約ジョブ `ci-complete` の `needs` に全ジョブが含まれているか、本リポに存在しない `make` ターゲット・`scripts/` を前提にしたジョブが混入していないかを確認する | P1 |
| `release.yml` | `workflow_dispatch` 限定のプレースホルダであり、有効化には公開対象 crate 名・crates.io 公開方針の確定を要する。現状のプレースホルダ状態自体は指摘しない | 指摘しない（既知の暫定状態） |
| `update-external.yml` の `runner-json` | `runner-json: '"ubuntu-latest"'` は YAML の単一引用符内に JSON 文字列 `"ubuntu-latest"` を置く正しい記法（バックスラッシュは含まれない）であり、指摘しない | 指摘しない（正しい記法） |
