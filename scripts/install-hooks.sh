#!/usr/bin/env bash
# lefthook の git hooks を導入する。lefthook（Go 製。crates.io には存在しないため cargo
# フォールバックは置かない）は brew（バージョン固定不可だが常用導線）を優先し、無ければ
# npm 配布版を exact 固定の npx ワンショットで実行する（lefthook が生成する hook スクリプトは
# PATH → npx の順で本体を解決するため、npx 経由の導入でもコミット時にフックが機能する）。
# lefthook.yml の有無による skip は Makefile 側（HAS_LEFTHOOK）で判定する。
# 前提: PIN_LEFTHOOK_VERSION（Makefile が定義元）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"
need_pin PIN_LEFTHOOK_VERSION

if command -v lefthook >/dev/null 2>&1; then
  lefthook install
elif command -v brew >/dev/null 2>&1; then
  echo "lefthook を導入します"
  brew install lefthook && lefthook install
elif command -v npx >/dev/null 2>&1; then
  echo "lefthook（npx 固定版）で hooks を導入します"
  npx --yes "lefthook@${PIN_LEFTHOOK_VERSION}" install
else
  echo "brew / npx が見つかりません。https://lefthook.dev/installation/ を参照してください" >&2
  exit 1
fi
