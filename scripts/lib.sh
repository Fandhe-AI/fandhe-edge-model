#!/usr/bin/env bash
# scripts/*.sh が共有するヘルパー。直接実行せず source して使う:
#   DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${DIR}/lib.sh"
#
# source した側は `set -euo pipefail` を引き継ぐ。cwd はリポジトリルートへ移すため、
# `make` 経由でも直接実行でも同じ場所を対象にする。
#
# ツールの固定バージョン（PIN_*）の単一の定義元は Makefile で、Make が export して渡す。
# スクリプトを直接叩く場合は `PIN_UV_VERSION=... scripts/py.sh sync` のように明示するか、
# `make <target>` 経由で実行する（版の二重定義を避けるため既定値は持たない）。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT_DIR
cd "${ROOT_DIR}"

# need_pin <NAME>: 固定バージョンの環境変数が未設定なら、Make 経由で実行するよう案内して失敗する。
need_pin() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "error: ${name} が未設定です（版の定義元は Makefile。make <target> 経由で実行してください）" >&2
    exit 2
  fi
}

# require_uv: uv の有無と固定版（PIN_UV_VERSION）を確認する。無ければ導入方法を案内して停止し、
# 自動導入（curl | sh 等）はしない（rustup と同一のサプライチェーン方針）。
require_uv() {
  need_pin PIN_UV_VERSION
  if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv が見つかりません。brew install uv（バージョン ${PIN_UV_VERSION} 系）または https://docs.astral.sh/uv/getting-started/installation/ の公式手順で導入してから再実行してください" >&2
    exit 1
  fi
  local uv_ver
  uv_ver="$(uv --version 2>/dev/null | awk '{print $2}')"
  if [ "${uv_ver}" != "${PIN_UV_VERSION}" ]; then
    echo "error: uv の版が固定版と不一致です（検出 ${uv_ver}、必要 ${PIN_UV_VERSION}）。--no-config のため pyproject.toml の required-version が効かず、ここで検査します" >&2
    exit 1
  fi
}

# reject_args <script-name> "$@": 引数を取らないスクリプトで未知の入力を黙って無視せず、
# usage を出して終了コード 2（誤用）で止める。
reject_args() {
  local script_name="$1"
  shift
  if [ "$#" -ne 0 ]; then
    echo "usage: ${script_name} (no arguments)" >&2
    exit 2
  fi
}
