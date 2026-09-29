# AGENTS.md

## 本書の用途

本書は `.github/workflows/ai-review.yml`（`Fandhe-AI/actions` の `ai-review` reusable workflow を呼ぶ wrapper。check 名 `codex / preflight`・`codex / review`・`codex / post_feedback`）が Codex による PR 自動レビューの基準として読む、リポジトリ固有のレビュー観点集である。

- Codex の既定 prompt は **PR の base コミットの本書** を読む。そのため本書への変更は、当該 PR のレビューには反映されず、**マージ後の次の PR から実効** になる
- 本書は日本語で記述する。プログラムの出力文字列（エラーメッセージ・ログ・CLI 出力・JSON）や識別子・コマンドは原語（英語）のままでよい
- `docs/spec`（`fandhe-edge-model-spec` submodule）は private であり、レビュー実行環境からは読めない前提とする。レビューでは spec 本文との一致を判定材料にせず、**REQ-n・TASK-n・M-n・PoC-n の併記があるか** という、diff だけで確認できる観点に限定する。ID が欠けた spec 由来の変更は「spec 参照規約」観点（P1）として指摘する
- Rust の crate は `crates/<name>/` の 1 階層に置く。TASK-15.2（#28）で `crates/core`（`fandhe-edge-core`。共通コア）・`crates/cli`（`fandhe-edge-cli`。操作アダプターの CLI）を、TASK-17.1-1（#44）で `crates/data`（`fandhe-edge-data`。データ契約）を、TASK-24.1-1（#59）で `crates/eval`（`fandhe-edge-eval`。評価器）を作成済みで、core・cli・data・eval の 4 crate に加え、#117 で `crates/runtime`（`fandhe-edge-runtime`。成果物・推論 SDK。単体/バッチ共通の推論経路の継ぎ目のみ）を作成済みで、残りの層の crate は後続 TASK で追加する。本書の層区分は spec の 6 層（共通コア・データ契約・学習ワーカー・評価器・成果物 / 推論 SDK・操作アダプター）に従う
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
make py-ci       # 学習ワーカー（trainer/）: ruff format --check・ruff check・pytest（uv run --locked）
make test-trainer-integration  # 実 trainer（MLX CPU）を Rust から起動する結合テスト（#[ignore] 分離分。issue #258）
make check-runtime-linkage  # 推論ランタイムの動的リンク確認（Mac 実機前提。make ci には含まれない。#115）
make ci          # lint-docs + check-workspace-manifest + 上記 6 つを一括実行
make doctor      # 環境診断のみ（何も導入しない）
```

- Makefile の `fmt`/`lint`/`test` は `Cargo.toml`（workspace）と `crates/*/Cargo.toml`（メンバー crate）の両方が揃っている場合に実行され、欠けると `skip:` を表示してスキップする（`deny` は加えて `deny.toml` を要する）。TASK-15.2 以降は両方が揃っているため、これらは実行される。`skip:` を「検証済み」として扱う記述・変更は P1
- workspace 作成後の PR からは、PR 本文にこれらのコマンドの実行結果が記載されているか（同じ PR で CI 設定を変更する場合はその diff にこれらのコマンドが含まれているか）を確認する。本節の未達は、個別に優先度を明記した項目を除き既定で P1 とする
- clippy 警告は 0 件を維持する。理由コメントなしで `#[allow(...)]` により警告を握りつぶす差分は P1。crate・モジュール全体に及ぶ広範な `#![allow(...)]` や、`unsafe` 関連 lint・外部入力の検証を隠す lint（`clippy::unwrap_used`・`clippy::expect_used`・`clippy::indexing_slicing` 等）の外部入力経路での抑止は、理由コメントの有無を問わず P0
- テストの skip・ignore・アサーション弱体化・浮動小数の許容差の拡大で CI を通す差分は P0（回帰検出の後退を招くため）
- 学習ワーカーは `trainer/` の Python（uv プロジェクト）で実装する（オーナー判断 2026-09-27）。`trainer/` のコードは `make py-ci` と `python-ci.yml` で検証される。ruff の設定（特に `S` ルール）の緩和・`# noqa` による理由なしの抑止・`uv.lock` を更新しない依存変更は P1（`.claude/rules/coding-python.md`）

### 実機前提テスト

- GPU（Metal / MLX）・実機での性能測定（p95 レイテンシ・容量）・sandbox 下の通信 0 件確認（REQ-38）・長時間学習を要するテストは、GitHub ホステッド runner で実行できないため既定のテスト集合から明示的に分離されているか確認する。分離の仕組み・実行コマンドは該当タスクで決め、本書に追記する
- 分離したテストに理由（必要な環境）と REQ-n が記され、実機での実行結果が **証拠の種別（テストハーネス / 模擬 / 推定 / 実機）付きで** PR に記録されているか確認する
- 既定のテスト集合で動くはずのテストを、CI 通過のために実機前提テストへ移す差分は P0
- 実機での測定・判定が「人間」担当のタスクを、計測スクリプト準備を超えて Agent が単独で完了扱いにしていないか確認する

#### sandbox 下の完走確認（REQ-38・TASK-38.1-1・#162）

- 実機確認は `scripts/sandbox-run.sh` を macOS 実機（Apple Silicon・`/usr/bin/sandbox-exec`）で**人が手動実行**する。sandbox の外で先に `cargo build` と `make py-sync` を済ませ、定義ファイルとデータを用意する。例: `scripts/sandbox-run.sh --definition <定義> --project-dir <未作成の dir> --out-dir <空の dir> --candidates 1 --smoke`
- 既定の `make test` に含まれる `crates/cli/tests/sandbox_run_script.rs` は偽の launcher を使うテストハーネスで、実際の通信遮断は行わない。実機の証拠にはならない（証拠種別: テストハーネス）
- 実バイナリは TASK-33.1-2（#136）の完了まで `register` で `runtime_error`(70) になる。完走を装っていないことの確認に留まる
- 陽性対照（sandbox 下で curl を実行して拒否の検出を確かめる）は TASK-38.2 の担当

#### 拒否ログの監視・集計（REQ-38・TASK-38.1-2・#163）

- 実機確認は `scripts/sandbox-monitor.sh --definition <定義> --project-dir <未作成の dir> --out-dir <空の dir> [--candidates N] [--smoke]` を macOS 実機で**人が手動実行**する（前提は `sandbox-run.sh` と同じ）。`log stream` を実行の前に開始し後に止め（`log show` では Sandbox の拒否ログが取れない。PoC-16）、`scripts/sandbox_deny_report.py`（標準ライブラリのみ。最低版は Python 3.9。macOS 標準の `/usr/bin/python3`〔Xcode CLT〕が 3.9 のことが多いため。監視スクリプトが前提確認で版を検証し、満たさなければ判定不能(70)にする。集計器は 3.9 の文法で書き、テストで `ast.parse(feature_version=(3, 9))` により文法を検査する）が操作トークン（`network*` で始まる操作）で通信拒否を機械判定する。部分文字列 `network` では判定しない（PoC-16 の誤検出の回避）
- 帰属と件数: 本ツール起因かは PID で判定し（`run.meta.json` の `process_pids` に PID が含まれる場合のみ tool。PID は `pgrep -g` で工程グループを列挙して採取し、ps の列幅に依存しない。名前は根拠にせず出力ラベルの正規化にだけ使う）、最終の終了コードは `decide()` の優先順（判定不能 70 > run の 70 > 10 > 12 > run のその他）で決める。拒否件数は 70 でもレポートに残す（`sandbox-run.sh` が工程グループの PID を 0.1 秒間隔で採取して記録する。採取できなかった短命プロセスの拒否は `pending`(12) になる）。集計器は run.meta.json の `exit_code` と実行スクリプトの実際の終了コードが食い違えば判定不能(70)にする。`log` の stdout は `ulimit -f`（RLIMIT_FSIZE）で書き込み時点の容量上限を掛け、stderr は保存しない。停止は TERM → 上限付き待機 → KILL → wait の順で、終了を確認できなければ判定不能(70)にする。通信拒否の件数は重複報告分を合算した発生回数で、レポートのレコードは 1000 件で切り詰める（`network_denials_truncated`）。ログは 1 行ずつ読む
- 判定と終了コード: 本ツール起因（PID 照合済み）の通信拒否あり `judged_fail`(10)・帰属不明の通信拒否あり `pending`(12)・監視の無効や読めない行・形式外の拒否行（`deny` を含み操作を読み取れない行）・時刻の不整合 `runtime_error`(70。`network_verdict:"undeterminable"`)・拒否 0 件は run の終了コードを伝搬（合格は run も 0 のときのみ）
- 出力先: `<out-dir>/network_report.json`（件数・通信拒否のレコード。通信先・パス等の生文字列は書かず、固定語彙と salt 付きダイジェストだけ）・`log_stream.ndjson`（生ログ。0600）・`monitor.meta.json`・`run/run.meta.json`。生ログには他アプリのイベントが含まれるため、PR・Issue へは転記せず `network_report.json` の件数を記録する
- `crates/cli/tests/sandbox_monitor_script.rs` は偽の `log`・偽の launcher と合成 fixture（`fixtures/sandbox_deny_log/`）を使うテストハーネスで、既定の `make test` で実行される。実機の証拠にはならない（証拠種別: テストハーネス）。`evidence_hint` は `requires_human_review` か `test_harness` のみで、Agent は「実機」と確定させない
- 陽性対照が未実施（`positive_control:"not_run"`）の間は、拒否 0 件の結果で「検出手段が機能する」とは言えない。TASK-38.2 と組み合わせて初めて 0 件の判定が有効になる

### `env -i` 環境での推論（TASK-32.3・#115・REQ-32）

- `crates/runtime/tests/env_isolation.rs` は、環境変数を空にした子プロセスで C1・C3 の fixture 推論が exit 0 になることを確かめ、既定の `make test` で実行される（unix 限定。rust-ci の Linux・macOS runner で実行。証拠種別: テストハーネス）。実機前提へ移す差分は P0
- CLI `fandhe-edge infer` の `env -i` 確認は、工程の接続（#136）完了後に追加する（現時点は推論ランタイム層のみ）
- `make check-runtime-linkage`（`scripts/check-runtime-linkage.sh`）は Mac 実機で人間が実行し、`otool -L` で Python・MLX への動的リンクが無いことと `env -i` 実行の結果を「実機」として PR に記録する。Linux の `ldd` の結果は補助で、Mac 実機の証拠にはならない。未実施の間は「未実施」と書く

### 実行環境（uv venv）を要するテスト（issue #258）

- `crates/train/tests/real_trainer.rs` は実 `trainer/`（MLX・C1/C3・ONNX 書き出し）を Rust の `run_train` から起動して学習ジョブを完走させる（REQ-18/19/34/39。証拠種別: テストハーネス・合成データ・CPU）。`make py-sync` 済みの `trainer/.venv` と MLX CPU が必要で、`rust-ci` の 3 OS runner には無いため `#[ignore]` で既定の `make test` から分離している
- 実行コマンドは `make test-trainer-integration`（`--ignored --exact` で 2 件のテストを個別に起動し、各起動の出力で `1 passed` を検査する）。`python-ci`（macos-14 arm64）とローカルの `make ci` で実際に実行される。GPU・実機測定を伴わないため上の実機前提テストとは別扱いで、CI で実行されない分離は P0
- `rust-ci` では `ignored` と報告され、Windows は `#![cfg(unix)]` で対象外（`run_train` が `UnsupportedPlatform`）

### 学習ワーカーの起動契約（Issue #12）

- 学習ワーカー（`trainer/`）は `package = false`（配布パッケージを持たないスタブ）のため、外部プロセスからの起動は唯一の起動口 `trainer/launch.py` を経由する契約とする: `<venv の python> -I trainer/launch.py train --request <path>`（`-I` 隔離モード必須。`trainer/launch.py`・`trainer/src/fandhe_edge_trainer/cli.py`・`trainer/src/fandhe_edge_trainer/supervisor.py::worker_argv` のモジュール docstring 参照）
- `PYTHONPATH` の手動設定を前提にした呼び出し（`python -m fandhe_edge_trainer ...`）への回帰・新規追加は P1
- Rust 側ジョブ管理（TASK-34.x）からの実際の呼び出しは本 Issue のスコープ外（将来 Rust 側が本契約を採用するかはオーナー確認事項）

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

### 再利用・アセット化

本ツールは利用者ごとの用途に合わせた判定モデルを作る汎用ツールであり、特定用途の前提をコードに持ち込まないことが転用性の前提になる。評価ロジック・正準化ハッシュの一元化は「アーキテクチャ・設計整合」節で扱うため、ここでは重ねて指摘しない。

| 観点 | 確認内容 | 優先度 |
| ---- | ---- | ---- |
| 汎用処理の分離 | 複数の層・工程が使う汎用処理（パスの閉じ込め検証・サイズ上限付きの読み込み・JSON エラー応答の組み立て等）が下位層の 1 箇所に置かれ、CLI・TUI・MCP・学習ワーカーへ複製されていないか | P1 |
| 用途固有値のハードコード | 特定用途のラベル名・入出力構造・データセット名・ファイルパスを、定義ファイルや引数ではなくコードへ埋め込んでいないか。共通コア・推論ランタイムに特定用途の前提を持ち込んでいないか | P1 |
| 契約値・上限値の集約 | 終了コード・資源上限（REQ-39 の暫定値）・有意水準・浮動小数の許容差・`kind_version` の許可リストが、名前付き定数や設定として 1 箇所に集約されているか（同じ値のリテラルが複数箇所に散在していないか） | P1 |
| 拡張点の閉じ方 | 新しいモデルの種類を選択口（学習ワーカー層。REQ-18〜20）へ追加する際に、共通コア・評価器・既存の種類の実装の改変を要しない構造になっているか（種類ごとの分岐を上位層へ散らしていないか） | P1 |
| 推論ランタイムの単体利用性 | 推論ランタイムが配布パッケージだけを入力として、本リポの他の crate（学習ワーカー・操作アダプター）なしに組み込み利用できる API になっているか（REQ-32） | P1 |
| PoC からの移植 | PoC のコード・データを流用する場合に、本リポへ移植したうえで出典の PoC-n が記されているか（`docs/spec` 配下を直接参照していないか） | P1 |
| 利用者向けドキュメント | 公開 API・CLI の JSON 入出力・配布パッケージ形式を追加・変更する差分で、README・利用例・ドキュメントコメントが同じ PR で更新されているか | P2 |

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
| `ci.yml` の現状 | `ci.yml` は稼働中（`workflow_dispatch`・`pull_request`・main への `push` で発火）で、集約ジョブ `ci-complete` と各ジョブの check（`lint-docs / *`・`rust-ci (<os>) / *`・`rust-ci-default-features (<os>)`）が必須チェックに登録されている。発火条件（`pull_request`・main への `push`）を外す・絞める差分がないか（外すと必須チェックが PR に報告されず、全 PR のマージがブロックされる）。ジョブ・check 名の観点は下 2 行で確認する | P1 |
| `ci.yml` への変更 | `ci.yml` を変更する差分では、3 OS matrix（Linux・macOS・Windows）を維持しているか、集約ジョブ `ci-complete` の `needs` に全ジョブが含まれているか、本リポに存在しない `make` ターゲット・`scripts/` を前提にしたジョブが混入していないかを確認する | P1 |
| 必須チェック名の変更 | ジョブ ID・`name:` の変更で check 名（`codex / *`・`python-ci`・`ci-complete`・`lint-docs / *`・`rust-ci (<os>) / *`・`rust-ci-default-features (<os>)`。matrix の OS の追加・削除を含む）が変わる差分に、ruleset の必須チェックの更新手順（マージ前の置換）が PR 本文に記載されているか。旧名のまま残ると全チェック pass でもマージがブロックされる | P1 |
| `release.yml` | `workflow_dispatch` 限定のプレースホルダであり、有効化には公開対象 crate 名・crates.io 公開方針の確定を要する。現状のプレースホルダ状態自体は指摘しない | 指摘しない（既知の暫定状態） |
| `macos-15` ランナーの指定 | GitHub 公式ドキュメント（https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners の標準ランナー一覧）では、`macos-15`・`macos-14`・`macos-latest` は ARM64（Apple Silicon）、Intel（x64）は `macos-15-intel`・`macos-13` 等の別ラベルである。したがって arm64 が必要なジョブの `runs-on: macos-15` は適切なラベルであり、arm64 用ラベルへの変更を求める指摘をしない。Intel 系ラベル（`-intel`・`-large`・`macos-13`）で arm64 を要するジョブを動かす差分は通常どおり P1 とする | 指摘しない（公式ドキュメントに基づく） |
| `update-external.yml` の `runner-json` | `runner-json` は folded block scalar（`>-`）で JSON 文字列 `"ubuntu-latest"`（二重引用符込み、バックスラッシュなし）を渡す正しい記法であり、`fromJSON` で `ubuntu-latest` に解決される。エスケープの誤りとして指摘しない | 指摘しない（正しい記法） |
