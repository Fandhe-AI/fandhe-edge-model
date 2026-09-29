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
# Darwin では依存ライブラリを再帰的に辿る（直接リンクのみだと間接依存を見逃すため）。
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

# 直接依存のライブラリ名を 1 行 1 件で出す。otool -L は先頭行に検査対象パス自体を出すため除き、
# ライブラリ行（先頭の空白と括弧内の version 情報を除いた名前）だけを判定対象にする
# （作業ディレクトリ名に python・mlx が含まれても誤一致させない）。
direct_deps() {
  out=$(otool -L "$1") || return 1
  printf '%s\n' "$out" | sed 1d | sed 's/^[[:space:]]*//; s/ (.*$//'
}

# Darwin: 直接リンクだけでは「依存ライブラリがさらに Python・MLX へ依存する」経路を見逃すため、
# 絶対パスで実在する依存を再帰的に辿る（REQ-32。上限 MAX_LIBS 件で打ち切り fail-closed）。
# @rpath・@loader_path 等の未解決参照は辿れないが、名前は PATTERN で判定する。
# /usr/lib・/System 配下は OS 提供（dyld shared cache 内）のため再帰しない。
# Linux の ldd は推移的依存を含めて出力するため再帰不要。
MAX_LIBS=500
collect_libs() {
  : >"$2"
  q=$(mktemp)
  printf '%s\n' "$1" >"$q"
  : >"$q.seen"
  seen=0
  while [ -s "$q" ]; do
    cur=$(head -n 1 "$q")
    sed 1d "$q" >"$q.n" && mv "$q.n" "$q"
    direct_deps "$cur" >"$q.d" || { rm -f "$q" "$q.d" "$q.seen"; return 1; }
    while IFS= read -r dep; do
      [ -n "$dep" ] || continue
      printf '%s\n' "$dep" >>"$2"
      case $dep in
        /usr/lib/*|/System/*) continue ;;
        /*) [ -f "$dep" ] || continue ;;
        *) continue ;;
      esac
      if grep -qxF "$dep" "$q.seen"; then continue; fi
      printf '%s\n' "$dep" >>"$q.seen"
      printf '%s\n' "$dep" >>"$q"
      seen=$((seen + 1))
      if [ "$seen" -gt "$MAX_LIBS" ]; then
        echo "error: too many dependent libraries (> $MAX_LIBS)" >&2
        rm -f "$q" "$q.d" "$q.seen"
        return 1
      fi
    done <"$q.d"
  done
  rm -f "$q" "$q.d" "$q.seen" "$q.n"
}

for bin in "$CLI_BIN" "$TEST_BIN"; do
  echo "== $TOOL (transitive on Darwin): $bin =="
  if [ "$TOOL" = otool ]; then
    collect_libs "$bin" "$TMP" || { echo "error: otool failed" >&2; exit 1; }
  else
    ldd "$bin" >"$TMP" 2>&1 || { echo "error: ldd failed" >&2; exit 1; }
  fi
  if grep -i -E "$PATTERN" "$TMP"; then
    echo "FAIL: dynamic link to Python/MLX found in $bin (direct or transitive)" >&2
    exit 1
  fi
  echo "no Python/MLX dynamic link (direct and transitive)"
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
