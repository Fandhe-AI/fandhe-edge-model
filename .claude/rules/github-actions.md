---
paths:
  - ".github/workflows/**"
description: >
  GitHub Actions のワークフローを編集する際の方針。runner 方針（public は GitHub ホステッド）、必須チェック名と
  集約ジョブ、発火条件・concurrency・paths の書き方、ジョブ追加チェックリスト、所要時間の計測方法を規定する。
---

# GitHub Actions 編集ルール（リポ固有）

構成の一覧と必須チェックの現状は [ci](./ci.md)「CI の構成」を正とする。本書は変更時の判断基準を定める。

## runner 方針

- 本リポは **public**。組織方針（Fandhe-AI/actions の `docs/runner-policy.md`）により GitHub ホステッド runner を使う
  （private は self-hosted）。fork PR 経由で self-hosted runner 上に任意コードを実行させないため
- 現行の runner

    | ワークフロー | runner | 理由 |
    | --- | --- | --- |
    | `ci.yml` | `ubuntu-latest`・`macos-latest`・`windows-latest`（3 OS matrix）・`ci-complete` は `ubuntu-latest` | 3 OS 一級対応（OS 要件による例外） |
    | `python-ci.yml` | `macos-14` | MLX の arm64 wheel と拡張 ACL 検査が macOS 前提（OS 要件による例外） |
    | `update-external.yml`・`release.yml` | `ubuntu-latest` | — |
    | `ai-review.yml` | preflight / post_feedback は `ubuntu-latest`、review は reusable workflow 既定の self-hosted `codex` 用途ラベル | 既存の例外。codex の資格情報が self-hosted 側にあり、fork PR は reusable 側が fail-closed で拒否する。新たな self-hosted 利用の追加は禁止 |

- public リポジトリに新たに self-hosted を指定しない。例外を足す場合は fork PR の実行拒否が機械的に保証されることを確認し、
  理由をワークフロー冒頭に書く

## 必須チェック

- ruleset `main-protection` の必須チェック名の一覧は [ci](./ci.md) を参照する（`ci-complete`・`python-ci`・`lint-docs / *`・
  `rust-ci (<os>) / *`・`rust-ci-default-features (<os>)`・`codex / *`・`Cursor Bugbot`）。集約ジョブは `ci-complete`
- **ジョブ名・ジョブ ID・matrix の OS・`name:` を変えない**。変えると旧名の必須チェックが報告されず、PR がマージ不能になる。
  変える場合は先に ruleset を新しい名前へ更新する（変更はユーザー判断。Agent は ruleset を変更しない）
- ジョブを追加したら `ci-complete` の `needs` に加える。必須チェックへの追加は ruleset 側の作業（[ci](./ci.md)）
- reusable workflow を呼ぶジョブ（`uses:`）にジョブレベルの `if:` を付けない。skipped になると内部ジョブ名
  （`rust-ci (...) / cargo test` 等）の必須チェックが報告されずマージ不能になる

## 発火条件

- `push` は `branches: [main]` のみ。PR の検証は `pull_request`。同一内容を push と pull_request で二重に回さない
- 必須チェックを含む workflow に **workflow 単位の `paths` / `paths-ignore` を `pull_request` へ付けない**
  （発火しないと pending のままマージ不能になる）。実行を減らしたいときは、ジョブ内の変更検知でステップを省略し、
  ジョブ自体は成功で終える（例: `python-ci.yml` の `scripts/ci-docs-only.sh`）。`push`（main）は必須チェックが無いため
  `paths-ignore` を使ってよい（`ci.yml` が実施）
- 変更検知の対象に含めてよいのは、テスト・ビルドが読まないことを `grep` で確認したパスのみ。pytest は `docs/design/*.md`・
  `lefthook.yml` を読む
- `concurrency` は `group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}` とし、
  `cancel-in-progress: ${{ github.event_name == 'pull_request' }}`（PR の古い実行のみ打ち切り、main の push は打ち切らない）
- schedule は必要最小の頻度。現状 `update-external.yml` が日次 1 回（`0 0 * * *`）。増やさない
- draft PR のスキップはしない。ジョブレベルの `if` は必須チェックの報告を欠かすため（上記）。`ai-review.yml` の
  `opened / synchronize / reopened` も変更しない
- `merge_group` は使っていない（merge queue 未使用）。有効化する場合は ruleset とあわせて検討する

## 新規ジョブ・ワークフローを追加するときのチェックリスト

1. 既存ワークフローへのジョブ追加を優先し、ワークフロー数を増やさない
2. `ci-complete` の `needs` へ追加し、必須チェック名の ruleset 更新が要るかを PR 本文に書く
3. 不要な `needs` を作らない（独立なジョブは並列に走らせる）。セットアップ（checkout・toolchain・キャッシュ復元）が
   支配的な小さなジョブは分割せず統合する
4. `actions/checkout` は `fetch-depth: 1`・`persist-credentials: false`（履歴が要るときだけ深くする）。submodule は取得しない
   （`docs/spec` は private）
5. キャッシュ: Rust は reusable workflow の `cache` か `Swatinem/rust-cache`（SHA 固定）。`python-ci.yml` は意図的に
   キャッシュ無効（汚染回避）なので変えない
6. サードパーティ action はコミット SHA で固定する（`Fandhe-AI/actions` のみ `@latest`）。`permissions` は最小権限・
   `timeout-minutes` を設定し、`${{ }}` は `run` に直接埋め込まず `env` 経由（[ci](./ci.md)）
7. 処理は Makefile / `scripts/` の共通コマンドを呼ぶ。CI 独自のロジックを `run:` に長く書かない（[makefile](./makefile.md)）
8. lefthook と同じ検査を二重に置かない（[git-hooks](./git-hooks.md)）。重い検査の正は CI 側
9. 変更後に `actionlint .github/workflows/*.yml` を通す（`release.yml` は `inputs` を無効化中のため既存の指摘がある）。
   PR の CI で所要時間を前後比較する

## 所要時間の計測

```bash
gh run list --workflow ci.yml --limit 30 --json databaseId,conclusion,createdAt,updatedAt,event
gh run view <run-id> --json jobs --jq '.jobs[]|[.name,((.completedAt|fromdate)-(.startedAt|fromdate))]|@tsv'
```

### 棚卸し結果（2026-10-11 時点・直近 30 件）

| ワークフロー | 発火 | 中央値 | 最大 | 備考 |
| --- | --- | --- | --- | --- |
| `ci.yml` | pull_request 19 件 / push 11 件 | PR 626 秒 / push 534 秒 | 2261 秒 / 1934 秒 | クリティカルパスは macOS の `rust-ci-default-features`（581 秒）と `cargo test`（470 秒）。26 ジョブ |
| `python-ci.yml` | pull_request 30 件 | 284 秒 | 1304 秒 | `make py-ci` が 272 秒を占める。macOS |
| `ai-review.yml` | pull_request 30 件 | 94 秒 | 178 秒 | 失敗 10 件は `Gate on blocking findings`（指摘による正常な失敗） |
| `update-external.yml` | schedule 15 件 | 126 秒 | 161 秒 | — |
| `release.yml` | 手動のみ | 実績 0 件 | — | 発火条件を無効化中 |
