#!/usr/bin/env bash
# docs/spec（fandhe-edge-model-spec）は private リポジトリのため、アクセス権のない環境では
# 取得に失敗する。実装コードのビルド・テストは docs/spec 抜きでも成立させる方針
# （README.md）のため、失敗しても警告のみで成功終了する。
#
# `.gitmodules` は docs/spec を `update = none` にしている。spec を読めない利用者が
# `cargo install --git` でバイナリを入れるとき、cargo が submodule を取りに行って
# 失敗するのを避けるため（#758）。開発者はここで `--checkout` を付けて明示的に取得する。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

git submodule update --init --checkout ||
  echo "警告: docs/spec（private）の取得に失敗しました。アクセス権のない環境では想定内です（ビルド・テストは spec 抜きで成立します）"
