#!/usr/bin/env bash
# Make を介さない操作一覧。`make help` は Makefile の `## ` コメントから機械的に生成されるのに対し、
# 本ファイルは手動で保守する一覧であり、両者を自動で同期する仕組みは無い。
# Makefile のターゲットを追加・改名・削除したら同じ変更で本ファイルも更新すること。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

cat <<'EOT'
このリポジトリの操作と、Make を使わずに直接実行する方法
（固定バージョン PIN_* の定義元は Makefile。版が要るスクリプトは make 経由で実行する）:

環境
  doctor                    scripts/doctor.sh                       開発環境を診断する（読み取りのみ。PIN_UV_VERSION が必要）
  setup                     make setup                              submodule → rustup → hooks → py-sync を順に実行する
  rustup                    scripts/check-rustup.sh                 rustup の導入を確認する（自動導入はしない）
  submodule                 scripts/submodule.sh                    docs/spec サブモジュールを初期化・更新する
  hooks                     scripts/install-hooks.sh                lefthook の git hooks を導入する（PIN_LEFTHOOK_VERSION）

ドキュメント／設定ファイル系 lint
  lint-md / lint-editorconfig  make <target>                       Makefile に直書き（npx 1 行）
  lint-yaml                 scripts/lint-yaml.sh                    yamllint（PIN_YAMLLINT_VERSION）
  lint-commits              scripts/lint-commits.sh                 commitlint（PIN_COMMITLINT_VERSION・PIN_COMMITLINT_CONFIG_VERSION）
  lint-docs                 make lint-docs                          上記 4 つを一括実行する

Rust
  check-workspace-manifest  scripts/check-workspace-manifest.sh     cargo verify-project
  fmt / fmt-check / lint / test / clean  make <target>             cargo 直書き（clean は scripts/clean.sh）
  deny                      scripts/deny.sh                         cargo deny（PIN_CARGO_DENY_VERSION）
  check-runtime-linkage     scripts/check-runtime-linkage.sh        動的リンク確認（Mac 実機・人間が実行）
  real-machine-check        scripts/real-machine-check.sh [args]    Mac 実機での動作確認 A〜J（make 経由は ARGS='...' をコマンドラインで渡す）

学習ワーカー（trainer/。PIN_UV_VERSION が必要）
  py-sync / py-fmt / py-fmt-check / py-lint / py-test  scripts/py.sh <sync|fmt|fmt-check|lint|test>
  py-ci                     make py-ci                              py-fmt-check + py-lint + py-test
  test-trainer-integration  scripts/test-trainer-integration.sh     実 trainer を Rust から起動する結合テスト
  check-dependency-approvals  scripts/py.sh check-dependency-approvals   依存の承認台帳の照合
  check-unsafe-allowlist    scripts/py.sh check-unsafe-allowlist    unsafe 許可リストの照合

全体
  ci                        make ci                                 ローカルゲート一括実行（CI の ci.yml・python-ci.yml と同等）

`make <target>` は上記と同じ処理を呼ぶ薄い入口であり、処理の定義は scripts/ 側にのみ存在する。
EOT
