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
   キャッシュ無効（汚染回避）なので変えない。キーの設計・保存条件は下の「キャッシュ方針」に従い、確認結果を PR 本文に書く
6. サードパーティ action はコミット SHA で固定する（`Fandhe-AI/actions` のみ `@latest`）。`permissions` は最小権限・
   `timeout-minutes` を設定し、`${{ }}` は `run` に直接埋め込まず `env` 経由（[ci](./ci.md)）
7. 処理は Makefile / `scripts/` の共通コマンドを呼ぶ。CI 独自のロジックを `run:` に長く書かない（[makefile](./makefile.md)）
8. lefthook と同じ検査を二重に置かない（[git-hooks](./git-hooks.md)）。重い検査の正は CI 側
9. 変更後に `actionlint .github/workflows/*.yml` を通す（`release.yml` は `inputs` を無効化中のため既存の指摘がある）。
   PR の CI で所要時間を前後比較する

## キャッシュ方針

- 前提: リポジトリのキャッシュ上限は 10GB（Org 設定も 10GB。超過分は LRU で古いものから追い出し、7 日間アクセスの無いものは削除）。
  Org の Actions 予算は $0（超過で停止）のため 10GB を超える上限は使えず（上げると 10GB を超えた時点でキャッシュが読み取り専用になる）、
  上限を超える前提にしない。上限・予算の変更はユーザー判断。**10GB の中でヒット率を最大化する**ことを目的にする
- 入れる前に効果を測る: キャッシュを足す・残す前に、ログ（外部 crate の再ビルド数・`downloaded` 件数・復元にかかった秒数）で
  効果を確かめ、復元時間に見合わないものは入れない（例: 下の実測の `rust-ci-default-features` は正味 1 ジョブ約 10〜15 秒の短縮）。
  本リポのジョブは GitHub ホステッド runner（毎回まっさら）だが、self-hosted runner ではツールチェーン・`CARGO_TARGET_DIR`・
  パッケージストア等がジョブをまたいで残り、キャッシュが無意味になることがある（Org の実例: mirror-ui は pnpm キャッシュの復元に
  1 ジョブ平均 75 秒かけて `downloaded 0`、team-hub は `Swatinem/rust-cache` が registry しか保存しておらず後処理が共有
  `CARGO_HOME` の crate を消していたため、いずれも撤去）
- PR の実行が保存したキャッシュは `refs/pull/<n>/merge` スコープに入り、その PR 以外（main・他 PR）からは復元できない。
  main のスコープに保存したものは全 PR から復元できる。したがって**保存は main への push のみ**とし、PR は復元だけにする
  （`Swatinem/rust-cache` は `save-if: ${{ github.ref == 'refs/heads/main' }}`。reusable `rust-base-ci` は
  `cache-save: ${{ github.ref == 'refs/heads/main' }}`。`actions/cache` は PR で `actions/cache/restore`、main で
  `actions/cache` を使い分ける）。依存を変える PR の 2 回目以降の push は main の旧キーから
  部分復元になるが、依存の変更頻度が低いため複製の容量削減を優先する
- キーに `github.sha`・`github.run_id` 等の実行ごとに一意な値を入れない（毎回新規保存になり再利用されない）。
  `Cargo.lock`・toolchain・OS のハッシュで決まるキーにし、ジョブ間で同一内容なら共有する
- `hashFiles` に `**/Cargo.lock` 等の再帰 glob を使わない。`.agents/skills/make/samples/**` のサンプル lockfile（`.claude/skills` の
  symlink 経由も）が混入し、依存が変わっていないのにスキル更新でキーが変わる。ルートの `Cargo.lock` を直接指定する（再帰が必要なら
  `'!.agents/**'`・`'!.claude/**'` で除外する）。Rust の target をキャッシュするキーには rustc のバージョン（`rustc -vV` の
  commit-hash 等）を含め、runner イメージ・toolchain の更新後に古い target を exact hit で復元し続けないようにする
  （`Swatinem/rust-cache` v2.8.1 は rustc のバージョン・ルートの `Cargo.lock`・workspace メンバーの `Cargo.toml` を自動でキーに含む。
  ただし `**/.cargo/config.toml`・`**/rust-toolchain(.toml)` は再帰 glob で拾い除外の入力も無いため、サンプルの
  `.agents/skills/make/samples/projects/rust-workspace-xtask/.cargo/config.toml` もキーに入る。このファイルがスキル更新で
  変わったときは 1 回だけ miss〔外部 crate の再ビルド約 20 秒〕と main への再保存〔3 OS で約 108MB〕が起きるが、許容する。
  `.claude/skills` の symlink は辿らない）。`actions/cache` を直接使う場合は `restore-keys` で前方一致の部分復元を許し、
  `Cargo.lock` が変わっても cold にしない
- サイズと追い出し: 1 世代の合計（1 件のサイズ × 種別 × OS）が 10GB に余裕をもって収まるよう見積もる。毎回読まれる main の
  最新キャッシュは LRU で最後まで残るため、キー切替で古い世代が追い出されても新しい世代が保存済みならミスにならない。
  この状態を保てていれば上限を上げる必要はない
- キャッシュを追加・変更したら、PR と main の push の両方のログで結果を確認する:

    ```bash
    gh api --paginate "repos/Fandhe-AI/fandhe-edge-model/actions/caches?per_page=100" \
      --jq '.actions_caches[]|[.key,.size_in_bytes,.ref,.last_accessed_at]|@tsv'   # 一覧（ref で PR 由来の複製を確認）
    gh api repos/Fandhe-AI/fandhe-edge-model/actions/cache/usage                   # 使用量
    gh cache list -R Fandhe-AI/fandhe-edge-model -L 100                            # key・サイズ・ref
    gh run view <run-id> --log | grep -E "Restored from cache key|No cache found|Cache restored from key|Cache not found|Cache saved with key"
    gh cache delete <id> -R Fandhe-AI/fandhe-edge-model                            # 不要なキャッシュの削除
    ```

  `Cache restored from key:` が primary key と一致すれば exact、それ以外は partial。`gh cache delete` は全 PR・main で共有する
  状態を消すため、一覧で対象（key・ref・最終アクセス）を確認してから行う
- PR チェックリスト: キャッシュを追加・変更する PR では、上記（効果測定・保存は main のみ・キー設計・サイズ見積もり）を
  確認したことを PR 本文に書く

### 実測（2026-10-11）

- 使用量 0.39GB / 10GB（12 件）。追い出しによる miss は無し（#528 マージ後の同日時点は 0.56GB・19 件）
- reusable `rust-ci`（Linux・macOS の clippy / test / deny）: 直近の PR・main とも全件 exact hit（14〜33MB/件、main スコープのみ）。
  `fmt` ジョブは毎回 `Cache not found`（保存されないキーを復元しようとしている。上流 Fandhe-AI/actions 側の挙動）。
  windows-latest は上流の既知問題で `cache: false`
- reusable `rust-ci` の旧キー（`<prefix>-<OS>-<job>-v3-<hash>`）は `hashFiles('**/Cargo.lock')` で rustc のバージョンを含まず、
  実測のハッシュ `bbff9d30…` はルートの `Cargo.lock` と `.agents/skills/make/samples` のサンプル 3 件の合計だった
  （ルート単独なら `f04a779e…`）。Fandhe-AI/actions#165（`latest` = 3a41f93）でルートの `Cargo.lock` と rustc のハッシュを使う
  `v4` キーに変わり、`cache-save` 入力が追加されたため、本リポは `cache-save: ${{ github.ref == 'refs/heads/main' }}` を渡す
  （キー空間の切り替えで、次の main の push で一度だけ v4 キーが新規保存される）
- `rust-ci-default-features`（`Swatinem/rust-cache`、#527 で追加）: 3 OS で 30〜43MB/件。#527 の PR 実行が PR スコープへ保存した
  複製 3 件（約 108MB）は main から使えず、main の初回 push は 3 OS とも `No cache found` で別途保存した。
  `save-if` 追加後（#528 の PR 実行）は 3 OS とも main のキャッシュから `full match: true` で復元し、PR スコープへの保存は 0 件。
  ヒット時は外部 crate のビルドが 0 件になる（miss 時は 39〜47 crate・約 16〜20 秒）。復元は 3〜4 秒のため、正味の短縮は
  1 ジョブ約 10〜15 秒。ジョブ時間（138〜494 秒）の大半はメンバー crate のビルドとテスト実行で、キャッシュでは縮まない

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
