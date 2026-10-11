#!/usr/bin/env bash
# 本リポの ./target のみ削除する。素の `cargo clean` は CARGO_TARGET_DIR / build.target-dir が
# 他 worktree と共有のディレクトリを指している場合にそれを丸ごと消すため、`--target-dir target` で
# 限定する。target/ が symlink（共有ディレクトリへの参照の可能性）なら辿らずに拒否する。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

if [ -L target ]; then
  echo "error: target/ が symlink です（共有ビルドディレクトリの可能性）。削除を拒否します" >&2
  exit 1
fi
cargo clean --target-dir target
