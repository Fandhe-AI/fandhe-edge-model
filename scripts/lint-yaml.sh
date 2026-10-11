#!/usr/bin/env bash
# yamllint（.yamllint 参照）。yamllint は Python 製のため npx で賄えない。導入済みの実体
# （brew / pip）を優先し、uvx があれば固定版のワンショット実行で代替する。いずれも無ければ
# fail-closed で導入方法を案内して失敗する（silent skip は CI との false-green 乖離になるため行わない）。
# 前提: PIN_YAMLLINT_VERSION（Makefile が定義元）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"
need_pin PIN_YAMLLINT_VERSION

if command -v yamllint >/dev/null 2>&1; then
  yamllint .
elif command -v uvx >/dev/null 2>&1; then
  uvx "yamllint==${PIN_YAMLLINT_VERSION}" .
else
  echo "yamllint 未導入: brew install yamllint / pip install yamllint==${PIN_YAMLLINT_VERSION} で導入してください" >&2
  exit 1
fi
