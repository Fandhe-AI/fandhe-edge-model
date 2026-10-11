---
paths:
  - "lefthook.yml"
description: >
  lefthook.yml を編集する際の方針。フックは staged 限定・glob 限定で一瞬で終わる検査と
  push 後では手遅れの検査に限り、テスト・clippy・ruff check・pytest など CI と重複する
  重い検査は CI（.github/workflows/ci.yml・python-ci.yml）へ置く。時間予算・判断手順・計測方法を規定する。
---

# Git hooks（lefthook）編集ルール（リポ固有）

## 目的

フックはコミットのたびに実行されるため、遅いフックはそのまま開発速度の低下になる。
フックはローカルでの早期検知専用であり、CI の代替ではない。**開発速度を落とさないことを最優先**し、
重い検査・網羅的な検査は CI（`.github/workflows/ci.yml`・`python-ci.yml`）に任せる。

ローカルゲートの規約（[ci](./ci.md)「ローカルゲート」・[coding-rust](./coding-rust.md)）自体は変わらない。
`make lint`・`make test`・`make py-ci` 等はコミット・PR 前に手で実行する前提のまま、フックで強制しないだけである。

## 時間予算

| フック | 予算 | 判定方法 |
| --- | --- | --- |
| pre-commit | 全ジョブ合計 1 秒未満 | `lefthook run pre-commit --all-files` の `summary: (done in N seconds)` |
| commit-msg | 即時（外部コマンド・ネットワーク不可） | 実コミット時に表示される commit-msg の summary（メッセージファイルを引数に取るため `--all-files` では計測しない） |
| pre-push・post-checkout・post-merge 等 | 原則追加しない | 追加する場合は理由と実測値を PR 本文に記載する |

`--all-files` は全追跡ファイルを対象にする最悪ケースであり、glob に一致する staged ファイルが無いジョブは
実際のコミットでは発火しない。一方、`secrets-content-guard` は staged diff の追加行数に比例するため、
`--all-files`（何も stage しない）では計測できない。変更した場合は大きめのファイル（数千行）を stage して
`lefthook run pre-commit --job secrets-content-guard` で別途計測する。

## フックに置かないもの（→ CI へ）

- テストスイート（`make test`・`make py-test`・`make test-trainer-integration`）・ビルド
  → `ci.yml` の `rust-ci (<os>)`（`--all-features`）・`rust-ci-default-features (<os>)`（既定 feature）、`python-ci.yml` の `python-ci`
- clippy（`make lint`）・ruff check（`make py-lint`）などリポジトリ全体を走査する lint
  → `ci.yml` の `rust-ci (<os>)`・`rust-ci-default-features (<os>)`、`python-ci.yml` の `python-ci`（`make py-ci`）
- 依存監査（`make deny`）・ドキュメント系 lint（`make lint-docs`）→ `ci.yml` の `rust-ci (<os>)`・`lint-docs`
- 依存パッケージのインストール・ネットワークアクセス（`uv run` は `--offline` または既存 venv の範囲に限る）
- CI のジョブと同じコマンドを同じ範囲で実行するもの

## フックに置いてよいもの

- `{staged_files}` に限定し、`glob` / `exclude` で対象を絞った 1 秒未満の検査
- CI ではコミット単位で検知できない、または push 後では遅いもの
  - `secrets-guard`・`secrets-content-guard`: 秘密情報は push 前に止める必要がある（[security](./security.md)）
  - commit-msg の `conventional-commits`: CI の commitlint（`lint-docs`）は PR 時のみで、push 済みのコミットは履歴の書き換えなしに直せない

### 明示的な例外

以下は CI と重複するが、発火条件が glob で限定され軽量なため pre-commit に残す（2026-10-11 計測・決定。
計測値は `lefthook run pre-commit --all-files` の各ジョブ時間で、pre-commit 全体の summary は 0.91 秒）。

| ジョブ | 重複する CI | 残す理由 | 計測値 |
| --- | --- | --- | --- |
| `rustfmt-check` | `ci.yml` の `rust-ci (<os>)`（fmt） | staged の `.rs` のみ対象。整形漏れはコミット時点で直す方が安い | 0.91 秒（全 `.rs` 237 件。通常のコミットはこれより速い） |
| `ruff-format-check` | `python-ci.yml` の `python-ci`（`make py-ci` → `py-fmt-check`） | `trainer/**/*.py` が staged のときだけ発火。trainer 全体を走査するが軽量 | 0.03 秒（初回の venv 解決込みで 0.33 秒） |
| `dependency-approvals` | `python-ci.yml` の `python-ci`（pytest の `test_req38_real_repository_is_fully_approved`） | 依存ファイル・台帳が staged のときだけ発火（低頻度）。承認記録の漏れ（REQ-38・[dependency-policy](./dependency-policy.md)）を PR 前に検知する設計ゲート | 0.08 秒 |
| `unsafe-allowlist` | `python-ci.yml` の `python-ci`（pytest の `test_real_repository_matches_allowlist`） | 未承認の `unsafe` 追加（REQ-39・[coding-rust](./coding-rust.md)）を PR 前に検知する設計ゲート | 0.78 秒（index の全 `.rs` を照合） |

- `unsafe-allowlist` は `crates/**/*.rs` が staged のたびに発火し、index の全 `.rs` を照合するため予算に最も近い。
  crate の増加で 1 秒を超えたら、照合スクリプトの高速化か CI（`python-ci`）のみへの移行を検討する
- pre-push は置かない。旧 pre-push の clippy・test・py-lint-test は上表「フックに置かないもの」の CI ジョブが担う

## 書き方の規約

- `pre-commit` の `parallel: true` を維持する（直列化しない）
- `{staged_files}` と `glob` / `exclude` で対象を絞る
- Cargo.toml・`trainer/pyproject.toml` の有無による skip は Makefile の HAS_CARGO / HAS_PY 判定と同一方針に揃える。
  ただし承認台帳の照合（`dependency-approvals`・`unsafe-allowlist`）は前提が欠けたら fail-closed で失敗させる
- 秘密情報の検査は fail-closed を維持する（diff・一覧を取得できない、抽出結果が不完全、grep が 2 以上で終了した場合は失敗させる）
- staged diff の行ごとにプロセスを起動するループは書かない（行数 × パターン数の起動で数千行の stage に数十秒〜数分かかる）。
  `secrets-content-guard` のように awk 1 本で抽出し、パターンごとに grep を 1 回だけ実行する
- 条件付き実行が必要な場合は lefthook の `skip` / `only` を使ってよい（lefthook スキル参照）

## 検査を追加するときの判断手順

1. その検査は CI（`ci.yml`・`python-ci.yml`）で既に実施されているか
    - 実施済みで、staged 限定にしても 1 秒以上かかる → フックに入れない
2. staged ファイルに限定できるか、glob で発火を限定できるか
    - できない → CI へ置く
3. CI へ置く場合は `ci.yml` にジョブを追加し、集約ジョブ `ci-complete` の `needs` にも必ず追加する
    - ruleset `main-protection` の必須チェック名の更新が必要になる（[ci](./ci.md)「CI の構成」）
4. 編集後に以下を実行し、summary の合計時間をユーザーへ報告する（1 秒以上なら構成を見直す）

```bash
time lefthook run pre-commit --all-files
```

## 手元で CI 相当を確認する場合

```bash
make ci          # ci.yml・python-ci.yml と同等のローカルゲートを一括実行
make fmt-check lint test   # Rust のみ（既定 feature）
make py-ci       # 学習ワーカーのみ（ruff format --check・ruff check・pytest）
```

## バイパス禁止

遅いからといって `--no-verify` や `LEFTHOOK=0` で回避しない（[security](./security.md)・[conventional-commits](./conventional-commits.md)）。
遅いフックはフック側を直す（検査を CI へ移す・staged 限定にする・対象 glob を絞る）。
