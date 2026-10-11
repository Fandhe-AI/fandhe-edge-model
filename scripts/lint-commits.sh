#!/usr/bin/env bash
# main からの分岐点以降のコミットを CI（lint-docs の commitlint ジョブ）と同じ extends 構成で
# 検証する。origin/main が未取得の環境では範囲を決められないためスキップする。
# `git rev-parse --verify --quiet` は「参照が存在しない」場合に終了コード 1 を返すため、
# これだけを skip 条件にし、それ以外の非 0（`fatal: detected dubious ownership` 等。
# 典型的には終了コード 128）はエラーメッセージを表示して非 0 終了する（fail-closed）。
# 前提: PIN_COMMITLINT_VERSION・PIN_COMMITLINT_CONFIG_VERSION（Makefile が定義元）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"
need_pin PIN_COMMITLINT_VERSION
need_pin PIN_COMMITLINT_CONFIG_VERSION

st=0
out="$(git rev-parse --verify --quiet refs/remotes/origin/main 2>&1 >/dev/null)" || st=$?
if [ "$st" -eq 1 ]; then
  echo "skip: origin/main が未取得のため commitlint をスキップ"
  exit 0
elif [ "$st" -ne 0 ]; then
  printf '%s\n' "$out" >&2
  echo "NG: origin/main の参照確認に失敗しました（git rev-parse exit=$st ）" >&2
  exit 1
fi
base="$(git merge-base origin/main HEAD)" || {
  echo "NG: git merge-base の実行に失敗しました" >&2
  exit 1
}
npx --yes -p "@commitlint/cli@${PIN_COMMITLINT_VERSION}" -p "@commitlint/config-conventional@${PIN_COMMITLINT_CONFIG_VERSION}" \
  commitlint --extends @commitlint/config-conventional --from "$base" --to HEAD
