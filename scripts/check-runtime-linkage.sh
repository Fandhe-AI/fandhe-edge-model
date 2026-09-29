#!/bin/sh
# 推論ランタイムが学習フレームワーク（Python・MLX）へ動的リンクしていないことの確認
# （REQ-32・TASK-32.3・#115）。Mac 実機で人間が実行する実機前提の手順で、`make ci` には含めない。
#
# 証拠種別:
#   - Darwin の `otool -L`: 実機（Mac）
#   - Linux の `ldd`: 補助（Mac 実機の証拠ではない）
#
# 確認対象:
#   1. CLI バイナリ `fandhe-edge`（`fandhe-edge-train` を含むが、Python へは子プロセスでしか
#      到達せず動的リンクは無い見込み）
#   2. 推論ランタイムの結合テスト実行ファイル `env_isolation`
# 続けて `env -i PATH=/usr/bin:/bin` で env_isolation を実行し、実機での実行記録を採る。
# CLI `infer` の推論そのものは工程の接続（#136）が未完のため確認対象外（help のみ smoke）。
#
# eval は使わない。ネットワークには接続しない（REQ-38）。

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
TARGET_DIR=${CARGO_TARGET_DIR:-$ROOT/target}
PATTERN='python|libpython|Python\.framework|mlx|libmlx'

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT INT TERM

echo "== build =="
cargo build --release -p fandhe-edge-cli --bin fandhe-edge
CLI_BIN="$TARGET_DIR/release/fandhe-edge"
[ -x "$CLI_BIN" ] || { echo "error: CLI binary not found: $CLI_BIN" >&2; exit 1; }

cargo test -p fandhe-edge-runtime --test env_isolation --no-run 2>"$TMP"
TEST_BIN=$(sed -n 's/^ *Executable .*(\(.*\))$/\1/p' "$TMP" | head -n 1)
[ -n "$TEST_BIN" ] && [ -x "$TEST_BIN" ] || { echo "error: env_isolation executable not found" >&2; exit 1; }

case $(uname -s) in
  Darwin) TOOL=otool; EVIDENCE="real machine (Mac)" ;;
  Linux) TOOL=ldd; EVIDENCE="auxiliary only (Linux ldd; not Mac evidence)" ;;
  *) echo "error: unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac
command -v "$TOOL" >/dev/null 2>&1 || { echo "error: $TOOL not found" >&2; exit 1; }

list_libs() {
  # otool -L は先頭行に検査対象パス自体を出すため除き、ライブラリ行だけを判定対象にする
  # （作業ディレクトリ名に python・mlx が含まれても誤一致させない）
  if [ "$TOOL" = otool ]; then
    out=$(otool -L "$1") || return 1
    printf '%s\n' "$out" | sed 1d
  else
    ldd "$1"
  fi
}

for bin in "$CLI_BIN" "$TEST_BIN"; do
  echo "== $TOOL: $bin =="
  list_libs "$bin" >"$TMP" 2>&1 || { echo "error: $TOOL failed" >&2; exit 1; }
  if grep -i -E "$PATTERN" "$TMP"; then
    echo "FAIL: dynamic link to Python/MLX found in $bin" >&2
    exit 1
  fi
  echo "no Python/MLX dynamic link"
done

echo "== env -i run: env_isolation =="
for t in \
  req32_inference_succeeds_under_env_i_without_path \
  req32_inference_succeeds_under_env_i_with_system_path \
  req32_inference_does_not_invoke_python_from_path; do
  env -i PATH=/usr/bin:/bin "$TEST_BIN" --exact "$t" >"$TMP" 2>&1 || { echo "FAIL: $t" >&2; exit 1; }
  grep -q '1 passed' "$TMP" || { echo "FAIL: $t did not run exactly once" >&2; exit 1; }
  echo "ok: $t"
done

echo "== smoke: env -i fandhe-edge infer --help (not inference; #136 pending) =="
env -i PATH=/usr/bin:/bin "$CLI_BIN" infer --help >/dev/null 2>&1 || { echo "FAIL: infer --help" >&2; exit 1; }

echo "OK: tool=$TOOL evidence=$EVIDENCE targets=fandhe-edge,env_isolation"
