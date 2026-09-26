# CI・ローカル検証規約（リポ固有）

## ローカルゲート（コミット・PR 前）

```bash
make fmt-check   # cargo fmt --all --check
make lint        # cargo clippy --workspace --all-targets -- -D warnings
make test        # cargo test --workspace
make ci          # 上記 + lint-docs + check-workspace-manifest + deny を一括実行
make doctor      # 環境診断のみ（何も導入しない）
```

- テストは変更のたびに全件実行し、失敗・警告を 1 件でも残したまま進めない（fail-closed）
- `Cargo.toml`・メンバー crate・`deny.toml` が未作成の間、cargo 系ターゲットは Makefile 側で `skip:` を表示してスキップされる（HAS_CARGO / HAS_MEMBERS / HAS_DENY）。スキップを「検証済み」と報告しない
- Python の学習ワーカーを追加したら、ruff / pytest のターゲットを `Makefile` と `ci` に組み込む（[coding-python](./coding-python.md)。infra-builder の担当）

## CI の構成

- `.github/workflows/ci.yml`: lint-docs（Fandhe-AI/actions）・rust-ci（fmt / clippy / test / deny を 3 OS matrix）・rust-ci-default-features・集約ジョブ `ci-complete`。**発火条件は無効化中**（workspace・メンバー crate・`deny.toml` の作成後に有効化し、ruleset の必須チェックへ `ci-complete` を登録する）
- `.github/workflows/release.yml`: crates.io 公開。**発火条件は無効化中**（公開 crate 名と公開方針の確定後に有効化）
- `.github/workflows/ai-review.yml`・`update-external.yml`: 稼働中（ai-review は Actions 変数 `CODEX_HOME_DIR` 設定までスキップ）

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
