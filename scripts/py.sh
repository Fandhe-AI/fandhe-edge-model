#!/usr/bin/env bash
# 学習ワーカー（trainer/。Python・uv 管理）系の処理。使い方: py.sh <sub>
#   sync / fmt / fmt-check / lint / test / check-dependency-approvals / check-unsafe-allowlist
# uv 0.12.19（PIN_UV_VERSION）を経由してのみ実行する（uv 未導入時は自動導入しない）。
# `--locked` を必ず付け、lock と pyproject.toml が食い違う場合は fail-closed で止める。
# `--no-config` で開発者のユーザー設定（~/.config/uv/uv.toml の exclude-newer 等）を読まない
# （--locked は lock どおりで再解決しないため exclude-newer が守るものは無く、衝突して失敗するのを
# 避ける。#465）。ミラー等は環境変数で指定する。
# trainer/pyproject.toml の有無による skip は Makefile 側（HAS_PY）で判定する。
# 前提: PIN_UV_VERSION（Makefile が定義元）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

PY_DIR="trainer"

if [ "$#" -ne 1 ]; then
  echo "usage: $(basename "$0") <sync|fmt|fmt-check|lint|test|check-dependency-approvals|check-unsafe-allowlist>" >&2
  exit 2
fi
sub="$1"

case "$sub" in
  sync | fmt | fmt-check | lint | test | check-dependency-approvals | check-unsafe-allowlist) ;;
  *)
    echo "usage: $(basename "$0") <sync|fmt|fmt-check|lint|test|check-dependency-approvals|check-unsafe-allowlist>" >&2
    exit 2
    ;;
esac

require_uv

case "$sub" in
  sync) uv --no-config sync --locked --directory "$PY_DIR" ;;
  fmt) uv --no-config run --locked --directory "$PY_DIR" ruff format . ../scripts ;;
  fmt-check) uv --no-config run --locked --directory "$PY_DIR" ruff format --check . ../scripts ;;
  lint) uv --no-config run --locked --directory "$PY_DIR" ruff check . ../scripts ;;
  test) uv --no-config run --locked --directory "$PY_DIR" pytest ;;
  # tomllib が要るため trainer の Python（3.12。.python-version）で実行する。`uv run --locked` は
  # 環境未構築時に依存の解決・同期（通信）を起こしうるため使わず、`--no-project --offline` で
  # プロジェクト同期と通信を抑止する（Python 本体が未導入ならダウンロードせず失敗する＝fail-closed）。
  # 照合スクリプトは標準ライブラリのみ。
  check-dependency-approvals)
    uv run --no-project --offline --directory "$PY_DIR" python -I ../scripts/check_dependency_approvals.py --root ..
    ;;
  check-unsafe-allowlist)
    uv run --no-project --offline --directory "$PY_DIR" python -I ../scripts/check_unsafe_allowlist.py --root ..
    ;;
esac
