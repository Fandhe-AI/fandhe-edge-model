# 依存追加時の明示承認フロー

REQ-38・TASK-38.3（#165）。依存の追加・更新・削除は通信を伴う操作（crates.io・PyPI からの取得）で、ユーザーの明示承認を経てから行う（[dependency-policy](../../.claude/rules/dependency-policy.md)）。承認を運用ルールだけに任せず、承認記録を台帳に残し、記録の無い依存変更を機械で止める。根拠は PoC-14・PoC-16 の依存追加運用（すべてユーザー承認済み）。

証拠の種別: 照合ゲートの動作はテストハーネス（合成の変異を加えた pytest）で確認している。承認が実在するかどうかは機械では確かめられず、PR レビューで確認する。

## 構成要素

| 要素 | 役割 |
| ---- | ---- |
| `dependency-approvals.json`（リポジトリルート） | 承認台帳。`direct`（manifest に直接書く依存。承認日・承認者・記録の参照・目的・配置する層）と `locked`（lock にだけ現れる推移的依存。根拠）を、`cargo` と `pypi` ごとに持つ |
| `scripts/check_dependency_approvals.py` | manifest・lock と台帳を照合する（標準ライブラリのみ・読み取り専用・通信なし）。終了コードは 0（全件承認済み）・10（未承認・固定違反・余分な記録）・64（形式不正）・70（予期しない例外） |
| `make check-dependency-approvals` | 上記の実行口。`make ci` の前提に含まれる |
| lefthook の pre-commit `dependency-approvals` | 依存ファイル・台帳が staged のときに照合する |
| `trainer/tests/test_dependency_approvals.py` | 実リポジトリの照合と、未承認・範囲指定・git source・台帳破損などの陰性対照。python-ci（`make py-ci`）で全 PR に効く |

## 照合する内容

- Rust: ルート `[workspace.dependencies]` は `=x.y.z` の完全固定で、`cargo.direct` に記録があること。path 依存は実在するメンバーのパスと package 名に一致すること。メンバー crate とルート manifest 自身の `[dependencies]` 等は `workspace = true` のみ。manifest の直接依存が lock に同じ版で現れること（`missing_in_lock`）。`Cargo.lock` の crates.io パッケージは台帳（direct と locked）に (name, version) があること。git・別レジストリは拒否する。内部 crate（lock に source が無い）の追加は台帳の更新が要らない
- Rust の配置層と機能: メンバー crate（`crates/<層>`）の依存は、台帳 `cargo.direct[].layers` にその層が含まれること（dev-dependencies のみの利用は `<層>(dev)` でも可。記録の無い層での利用は `unapproved_layer`）。ルートの `features`・`default-features` は台帳の `features`・`default_features` と一致すること（`feature_mismatch`）。メンバー側の features・default-features・optional による上書きは拒否する
- Python: `pyproject.toml` の依存は `name[extras]==x.y.z` のみで、`pypi.direct` に記録があること。 `[build-system].requires` も同じ規則で固定と記録を照合する（uv.lock には現れないため lock 照合は対象外）`uv.lock` の PyPI パッケージは台帳に (name, version) があること
- 台帳にだけ残った記録（削除・版の変更の取りこぼし）も失敗とする

## 手順

1. builder は依存の追加・更新・削除の必要が生じたら、実行せずに main へ「承認事項」として報告する（builder は依存も台帳も独断で変更しない）
2. main が承認に必要な 5 項目（名前と版・目的・配置する層 / ライセンス / メンテナンス状況 / 推移的依存とネイティブビルドの有無 / 配布サイズ・レイテンシへの影響）をユーザーに示す
3. ユーザーが明示的に承認する。承認前には `cargo add`・`cargo update`・`uv add`・`uv lock`・`uv sync --upgrade` など通信を伴う操作を実行しない
4. 承認後に操作を実行する（通信は承認の範囲に限る）
5. `Cargo.toml`・`pyproject.toml`・lock の変更と同じコミットで、`dependency-approvals.json`（`direct` は承認日・`record`、推移的依存は `locked` の `basis`）と dependency-policy.md の「承認済みの依存」表を更新し、PR 本文にも承認の記録を残す
6. `make check-dependency-approvals` と `make ci` を通す。PR では python-ci が同じ照合を実行する

## PR チェックリスト（依存に触れる PR の本文に貼る）

```text
- [ ] 承認を得た（名前・版・目的・配置する層・ライセンス・保守状況・推移的依存・サイズ / レイテンシ）。承認の記録: （PR / Issue のリンク）
- [ ] 承認前に通信を伴う操作（cargo update・uv lock 等）を実行していない
- [ ] dependency-approvals.json の direct / locked を更新した（record・basis は実在の記録を指す）
- [ ] Cargo は =x.y.z・Python は ==x.y.z の完全固定で、lock をコミットした
- [ ] make check-dependency-approvals と make ci が通る
- [ ] 証拠の種別（テストハーネス / 実機）を記した
```

## 限界

- 通信なしの契約: 照合は `uv run --no-project --offline`（プロジェクトの同期・解決をしない。Python 本体が未導入ならダウンロードせず失敗）で実行する。`uv run --locked` は環境未構築時に同期（通信）しうるため使わない
- lefthook の pre-commit は index（staged）の内容を一時ディレクトリへ書き出して照合する。作業ツリーだけを直して通すことはできない。最終の強制は CI（python-ci）。Python 側も `pyproject.toml` の直接依存が `uv.lock` に同じ版で現れることを照合する

- 台帳を同じ PR で書き換えれば照合は通る。承認の実在は PR レビュー（AGENTS.md「依存の追加・更新」）で確認する
- Makefile の lint ツール（markdownlint・yamllint・cargo-deny・uv 等）の固定版は照合の対象外
- `.claude/settings.json` の permissions・PreToolUse hook で `cargo add` や `uv add` を実行前に確認させる案、CODEOWNERS による台帳のレビュー必須化は、権限・リポジトリ設定の変更にあたるため本 Issue の範囲外（ユーザー判断）
