#!/usr/bin/env bash
# workspace 仮想 manifest（Cargo.toml）自体の構文・定義としての妥当性を検証する。
# `cargo verify-project` は member crate が 0 件の仮想 workspace でも成功する。
# Cargo.toml の有無による skip は Makefile 側（HAS_CARGO）で判定する。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

out="$(cargo verify-project 2>&1)" || {
  echo "$out" >&2
  echo "NG: Cargo.toml が cargo にとって不正な manifest です" >&2
  exit 1
}
if ! printf '%s\n' "$out" | grep -q '"success"'; then
  echo "$out" >&2
  echo "NG: cargo verify-project が success を返しませんでした" >&2
  exit 1
fi
