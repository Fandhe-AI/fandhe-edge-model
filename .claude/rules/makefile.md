---
paths:
  - "Makefile"
  - "scripts/**"
description: >
  Makefile と scripts/ を編集する際の方針。Makefile は scripts/*.sh か言語ネイティブの 1 行を呼ぶ薄い入口とし、
  複数行・条件分岐は scripts/ に置く。ターゲットの追加・改名・削除の手順、新規スクリプトの規約、
  check / fix / clean の契約を規定する。scripts/hooks/** は git-hooks.md の担当。
---

# Makefile・scripts/ 編集ルール（リポ固有）

詳細な設計論は make スキル（`.claude/skills/make`）を参照する。本書はこのリポジトリで守る要点のみを置く。

## 基本方針

- Makefile は**薄い入口**。レシピは `@scripts/<name>.sh [args]` の 1 行呼び出し、または言語ネイティブの 1 行コマンド
  （`cargo fmt --all --check`・`npx ...`）に限る。複数行のシェル・`for` / `if`・環境依存の処理は `scripts/` へ置く
- Make が持つのは「どのスクリプトをどの条件で呼ぶか」だけ。`HAS_CARGO` / `HAS_MEMBERS` / `HAS_PY` による skip と
  `$(origin ARGS)` 判定（`real-machine-check`）は Make でしか書けないため Makefile に残す
- 固定バージョン（`*_VERSION`）の定義元は Makefile のみ。`PIN_*` として export し scripts/ へ渡す。
  scripts/ 側に既定値を持たせない（二重定義の防止。`scripts/lib.sh` の `need_pin`）
- 空ターゲット（`@true` 等）を置かない。呼び出し元が終了コードを信用してしまう
- GNU Make 3.81（macOS 標準）で動く構文のみ。`.ONESHELL`・`.RECIPEPREFIX` 等の 3.82+ 機能は使わない
- 前提が欠けたときの挙動は 2 種類に分ける。未追加が正常系のもの（Cargo.toml・trainer/ 等）は `skip:` を表示して成功、
  承認台帳の照合・実機確認のように欠落が異常なものは `error:` で fail-closed。`skip:` を「検証済み」と報告しない（[ci](./ci.md)）

## ターゲットを追加・改名・削除するとき

1. `.PHONY` 宣言を置き、レシピ行頭を TAB にする
2. `## 説明` コメントを同じ行に付ける（`make help` はこの行から生成される）
3. `scripts/help.sh` の一覧を同じ変更で更新する（自動同期は無い）
4. 呼び出し元を洗い出して更新する

    ```bash
    grep -rn "make " .github .claude docs README.md CLAUDE.md AGENTS.md lefthook.yml scripts crates trainer
    ```

    特に `.github/workflows/ci.yml`・`python-ci.yml`、`lefthook.yml`、`crates/cli/tests/real_machine_check_script.rs`
    （`make real-machine-check` の ARGS 挙動をテストしている）、`.claude/rules/ci.md` の一覧
5. `make help` と、変更したターゲットの `make -n <target>` で確認する。`make ci` に含めるか判断する
6. 既存ターゲットのリネーム・引数や終了コードの変更は呼び出し元（CI・フック・docs）と同じ PR で行う

## scripts/ の新規スクリプト規約

- 先頭は `#!/usr/bin/env bash` と `set -euo pipefail`。`scripts/lib.sh` を source し（`ROOT_DIR` へ cd・`need_pin`・
  `require_uv`・`reject_args`）、引数を取らないスクリプトは `reject_args` で未知の入力を拒否する
- 実行権限（`chmod +x`）を付ける。`shellcheck -x -P SCRIPTDIR scripts/<name>.sh` を通す
- 出力文字列は [japanese-style](./japanese-style.md) に従い英語が原則。既存ターゲットから移したスクリプトは
  挙動維持のため元の日本語メッセージのまま残している（新規に書くものは英語）
- 既存の `check-runtime-linkage.sh`・`real-machine-check.sh`・`sandbox-*.sh` は POSIX sh 方針。書き換える際はその方針を保つ
- 自動導入（`curl | sh`）を足さない。導入は案内して停止する（[dependency-policy](./dependency-policy.md)）
- `scripts/hooks/**` を足す場合は [git-hooks](./git-hooks.md) の時間予算に従う（本リポは現時点でフックスクリプトを `scripts/` に持たない）

## ターゲットの契約

| 種別 | ターゲット | 契約 |
| --- | --- | --- |
| check（読み取りのみ） | `fmt-check`・`lint`・`test`・`py-fmt-check`・`py-lint`・`py-test`・`lint-*`・`deny`・`check-*`・`doctor` | ソースを書き換えない。失敗は非 0 |
| fix（書き換える） | `fmt`・`py-fmt` | 書き換える唯一のターゲット。`ci` に含めない |
| 環境構築 | `setup`・`hooks`・`rustup`・`submodule`・`py-sync` | 冪等。`rustup` は確認のみで自動導入しない |
| clean | `clean` | 本リポの `./target` のみ。`CARGO_TARGET_DIR` 共有先や symlink の `target/` は消さない（`scripts/clean.sh`） |
| 実機（人が実行） | `check-runtime-linkage`・`real-machine-check` | `ci` に含めない。前提が欠けたら `skip:` にせず非 0 |
| 集約 | `ci`・`lint-docs`・`py-ci` | 上記 check の束。CI の `ci.yml`・`python-ci.yml` と同等の範囲を保つ |

- `doctor` は何も導入・変更しない。必須ツール欠落のみ非 0
- `ci` に新しい check を足すときは、CI 側にも対応するジョブがあるか確認する（[github-actions](./github-actions.md)）
