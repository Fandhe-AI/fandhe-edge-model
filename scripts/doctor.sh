#!/usr/bin/env bash
# 開発環境の診断（読み取りのみ。何も導入・修復しない）。導入は setup の責務。
# 必須ツールが欠けていれば非 0 で終了し、任意ツールは欠けていても警告に留める。
# 前提: PIN_UV_VERSION（Makefile が定義元。`make doctor` 経由で実行する）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"
need_pin PIN_UV_VERSION

missing=0
for tool in git rustup cargo rustc; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf 'ok    %-16s %s\n' "$tool" "$("$tool" --version 2>/dev/null | head -n 1)"
  else
    printf 'miss  %-16s （必須）\n' "$tool"
    missing=1
  fi
done
for comp in rustfmt cargo-clippy; do
  if command -v "$comp" >/dev/null 2>&1 && "$comp" --version >/dev/null 2>&1; then
    printf 'ok    %-16s %s\n' "$comp" "$("$comp" --version 2>/dev/null | head -n 1)"
  else
    printf 'miss  %-16s （必須。rust-toolchain.toml の components。rustup show で導入）\n' "$comp"
    missing=1
  fi
done
for tool in npx lefthook yamllint uvx cargo-deny; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf 'ok    %-16s\n' "$tool"
  else
    printf 'warn  %-16s （任意。未導入の場合は該当ターゲットが導入案内または自動導入を行う）\n' "$tool"
  fi
done
if command -v uv >/dev/null 2>&1; then
  uv_ver="$(uv --version 2>/dev/null | awk '{print $2}')"
  if [ "$uv_ver" = "$PIN_UV_VERSION" ]; then
    printf 'ok    %-16s %s\n' uv "$uv_ver"
  else
    printf 'warn  %-16s %s （固定版 %s と不一致。trainer/pyproject.toml の required-version が拒否します）\n' uv "$uv_ver" "$PIN_UV_VERSION"
  fi
else
  printf 'warn  %-16s （任意。py-* ターゲットに必要。brew install uv や https://docs.astral.sh/uv/ を参照）\n' uv
fi
if [ -f docs/spec/.git ] || [ -d docs/spec/.git ]; then
  echo 'ok    docs/spec        submodule 取得済み'
else
  echo 'warn  docs/spec        submodule 未取得（private。make submodule で取得を試行）'
fi
if [ "$missing" -ne 0 ]; then
  echo "NG: 必須ツールが不足しています（make setup または公式手順で導入してください）" >&2
  exit 1
fi
