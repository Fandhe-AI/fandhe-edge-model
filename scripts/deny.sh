#!/usr/bin/env bash
# 依存監査（cargo deny check advisories bans licenses sources）。cargo-deny 未導入なら固定版を
# 自動導入する。Cargo.toml・deny.toml・メンバー crate の有無による skip は Makefile 側で判定する。
# 前提: PIN_CARGO_DENY_VERSION（Makefile が定義元）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"
need_pin PIN_CARGO_DENY_VERSION

export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo-deny >/dev/null 2>&1 || {
  echo "cargo-deny を導入します"
  cargo install "cargo-deny@${PIN_CARGO_DENY_VERSION}" --locked
}
cargo deny --locked check advisories bans licenses sources
