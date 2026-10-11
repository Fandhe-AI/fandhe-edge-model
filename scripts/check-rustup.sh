#!/usr/bin/env bash
# rustup は前提条件として確認のみ行い、自動導入はしない。取得したインストーラを検証なしに
# 実行する経路（curl | sh）を作らないため（サプライチェーン対策）。未導入時は
# 公式の導入手順を案内して停止する。toolchain は rust-toolchain.toml が単一真実源。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

if ! command -v rustup >/dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  echo "error: rustup が見つかりません。公式手順（https://rustup.rs/）で導入してから再実行してください" >&2
  exit 1
fi
