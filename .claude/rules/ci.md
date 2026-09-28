# CI・ローカル検証規約（リポ固有）

## ローカルゲート（コミット・PR 前）

```bash
make fmt-check   # cargo fmt --all --check
make lint        # cargo clippy --workspace --all-targets -- -D warnings
make test        # cargo test --workspace
make py-ci       # 学習ワーカー（trainer/）: ruff format --check・ruff check・pytest
make ci          # 上記 + lint-docs + check-workspace-manifest + deny を一括実行
make doctor      # 環境診断のみ（何も導入しない）
```

- テストは変更のたびに全件実行し、失敗・警告を 1 件でも残したまま進めない（fail-closed）
- `Cargo.toml`・メンバー crate・`deny.toml` が欠けている場合、cargo 系ターゲットは Makefile 側で `skip:` を表示してスキップされる（HAS_CARGO / HAS_MEMBERS / HAS_DENY）。TASK-15.2 以降はいずれも揃っているため実行される。スキップを「検証済み」と報告しない
- 学習ワーカー（Python）の ruff / pytest は `make py-ci` として `make ci` に組み込み済み（[coding-python](./coding-python.md)）。`trainer/pyproject.toml` が無い場合は `skip:` になるが、CI（python-ci.yml）は `trainer/` の欠落を失敗として扱う

## CI の構成

- `.github/workflows/ci.yml`: lint-docs（Fandhe-AI/actions）・rust-ci（fmt / clippy / test / deny を 3 OS matrix）・rust-ci-default-features・集約ジョブ `ci-complete`。稼働中（`workflow_dispatch`・`pull_request`・main への `push` で発火）。集約ジョブ `ci-complete` に加え、各ジョブの check も個別に必須チェックへ登録している（下の ruleset の項）
- `.github/workflows/release.yml`: crates.io 公開。**発火条件は無効化中**（公開 crate 名と公開方針の確定後に有効化）
- `.github/workflows/python-ci.yml`: 稼働中。学習ワーカーの `make py-ci` を macos-14（arm64。MLX の wheel と出力先の拡張 ACL 検査が macOS 前提）で実行する（`ci-complete` への統合は検討中）
- `.github/workflows/ai-review.yml`・`update-external.yml`: 稼働中（ai-review は Actions 変数 `CODEX_HOME_DIR` 設定までスキップ。`update-external.yml` が生成する日次同期 PR〔`chore/skills-update-*`・`chore/submodule-update-*` ブランチ〕も `skip-branch-prefixes` でスキップし、指摘は取り込み元の上流リポジトリで扱う）
- ruleset `main-protection` の必須チェック（29 件）: `codex / preflight`・`codex / review`・`codex / post_feedback`・`python-ci`・`ci-complete`・`lint-docs / *`（5 件）・`rust-ci (<os>) / *`（3 OS × 5 件）・`rust-ci-default-features (<os>)`（3 件）（GitHub Actions に束縛）と `Cursor Bugbot`（Cursor App に束縛）。PR HEAD に報告される check をすべて必須にしているのは、implement-issue-tree の自動マージのゲート（required に含まれない check が 1 件でもあればマージを辞退する）を通すため（2026-09-27 オーナー判断）。`lint-docs / *`・`rust-ci (<os>) / *` は Fandhe-AI/actions（`@latest`）の reusable workflow 側のジョブ名に由来するため、上流でジョブ名が変わると旧名の必須チェックが報告されずマージがブロックされる。その場合は ruleset を新しい check 名へ更新する。implement-issue-tree の自動マージ（`autoMerge: true`）は PR HEAD に報告される全チェックの必須化を前提とするため、チェックを追加・改名する workflow 変更では、マージ前に ruleset を新しいチェック名へ更新する

## 実機前提テスト

- GPU（Metal / MLX）・実機での性能測定（p95・容量）・sandbox 下の通信 0 件確認（REQ-38）・長時間学習を要するテストは、GitHub ホステッド runner で実行できないため既定のテスト集合から明示的に分離する（分離の仕組みは該当タスクで決め、`AGENTS.md` に実行コマンドと必要環境を記す）
- 分離したテストには理由（必要な環境）と REQ-n を記し、実機での実行結果を証拠の種別付きで PR に記録する
- 既定のテスト集合で動くはずのテストを、CI 通過のために実機前提テストへ移さない
- 実機での測定・判定が「人間」担当のタスクは、Agent は計測スクリプトの準備までに留める

## ワークフロー変更時の注意

- GitHub Actions のサードパーティ action はコミット SHA で固定する（`Fandhe-AI/actions` のみ `@latest` を許可）
- `permissions` は最小権限で明示し、ジョブには `timeout-minutes` を設定する
- secrets を `pull_request` イベントのログへ出力しない。`${{ }}` を `run` へ直接埋め込まず `env` 経由で渡す
- CI 設定の変更は infra-builder が担当し、reviewer / security-auditor のレビューを経る
