#!/bin/sh
# 推論ランタイムが学習フレームワーク（Python・MLX）へ動的リンクしていないことの確認
# （REQ-32・TASK-32.3・#115）。Mac 実機で人間が実行する実機前提の手順で、`make ci` には含めない。
#
# 証拠種別:
#   - Darwin の `otool -L`: 実機（Mac）
#   - Linux の `ldd`: 補助（Mac 実機の証拠ではない）
#
# cargo は `--locked` で起動し、Cargo.lock を暗黙に更新しない（REQ-38）。
#
# 確認対象:
#   1. CLI バイナリ `fandhe-edge`（`fandhe-edge-train` を含むが、Python へは子プロセスでしか
#      到達せず動的リンクは無い見込み）
#   2. 推論ランタイムの結合テスト実行ファイル `env_isolation`
# Darwin では依存ライブラリを再帰的に辿る（直接リンクのみだと間接依存を見逃すため）。
# 続けて `env -i PATH=/usr/bin:/bin` で env_isolation を実行し、実機での実行記録を採る。
# CLI `infer` は工程の接続（#136）済みだが、推論そのものの `env -i` 実行 smoke は本スクリプトに
# 未追加（別課題。現状は help のみ smoke）。
#
# CLI の場所は cargo の `--message-format=json` が報告した実物から取る（`build.target-dir`・
# `CARGO_BUILD_TARGET_DIR`・`CARGO_TARGET_DIR` のどれにも追従する。規則を再実装しない）。
# 取り出した絶対パスは `cli_bin: <path>` の 1 行で stdout へ出す（real_machine_check_record.py が
# 読み、B・C・E で実行する CLI との同一性照合に使う。形式を変えない）。
#
# eval は使わない。ネットワークには接続しない（REQ-38）。

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PATTERN='python|libpython|Python\.framework|mlx|libmlx'

TMP=$(mktemp)
trap 'rm -f "$TMP" "$TMP.build" "$TMP.rp"' EXIT INT TERM

echo "== build =="
cargo build --locked --release -p fandhe-edge-cli --bin fandhe-edge \
  --message-format=json-render-diagnostics >"$TMP.build"
CLI_BIN=$(grep '"reason":"compiler-artifact"' "$TMP.build" | grep '"kind":\["bin"\]' \
  | grep '"name":"fandhe-edge"' | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' | tail -n 1)
case $CLI_BIN in
  ''|*\\*|[!/]*) echo "error: CLI binary not found" >&2; exit 1 ;;
esac
[ -x "$CLI_BIN" ] || { echo "error: CLI binary not found" >&2; exit 1; }
echo "cli_bin: $CLI_BIN"

cargo test --locked -p fandhe-edge-runtime --test env_isolation --no-run 2>"$TMP"
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
# 依存を再帰的に辿る（REQ-32。上限 MAX_LIBS 件で打ち切り fail-closed）。
# @rpath・@loader_path・@executable_path は LC_RPATH（辿っている実行ファイル自身と検査対象の
# 最上位バイナリの両方）を展開して解決する。解決できない @ 参照・存在しない絶対パスは、
# 参照先の検査ができないため失敗させる（fail-closed。見逃したまま「リンクなし」と報告しない）。
# /usr/lib・/System 配下は OS 提供（dyld shared cache 内）のため再帰しない。
# Linux の ldd は推移的依存を含めて出力するため再帰不要。
# バイナリの LC_RPATH を 1 行 1 件で出す。
rpaths_of() {
  otool -l "$1" | sed -n '/cmd LC_RPATH/,/^Load command/{s/^ *path //;s/ (offset [0-9]*)$//;t p;b;:p;p;}'
}

# 依存名 $2（辿っている実行ファイル $1、最上位バイナリ $3）を実ファイルのパスへ解決して stdout へ出す。
# 解決できなければ何も出さず非 0 を返す。
resolve_dep() {
  cur_dir=$(dirname "$1")
  top_dir=$(dirname "$3")
  case $2 in
    /*) [ -f "$2" ] && printf '%s\n' "$2"; return ;;
    @loader_path/*) c="$cur_dir/${2#@loader_path/}"; [ -f "$c" ] && printf '%s\n' "$c"; return ;;
    @executable_path/*) c="$top_dir/${2#@executable_path/}"; [ -f "$c" ] && printf '%s\n' "$c"; return ;;
    @rpath/*)
      rel=${2#@rpath/}
      for owner in "$1" "$3"; do
        rpaths_of "$owner" >"$TMP.rp" || return 1
        while IFS= read -r rp; do
          [ -n "$rp" ] || continue
          case $rp in
            @loader_path*) rp="$(dirname "$owner")${rp#@loader_path}" ;;
            @executable_path*) rp="$top_dir${rp#@executable_path}" ;;
          esac
          if [ -f "$rp/$rel" ]; then printf '%s\n' "$rp/$rel"; rm -f "$TMP.rp"; return 0; fi
        done <"$TMP.rp"
      done
      rm -f "$TMP.rp"
      return 1 ;;
    *) return 1 ;;
  esac
}

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
      esac
      real=$(resolve_dep "$cur" "$dep" "$1") || {
        echo "error: cannot resolve dependency '$dep' of $cur" >&2
        rm -f "$q" "$q.d" "$q.seen" "$q.n"
        return 1
      }
      dep=$real
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

echo "== smoke: env -i fandhe-edge infer --help (help only; inference smoke is a separate task) =="
env -i PATH=/usr/bin:/bin "$CLI_BIN" infer --help >/dev/null 2>&1 || { echo "FAIL: infer --help" >&2; exit 1; }

echo "OK: tool=$TOOL evidence=$EVIDENCE targets=fandhe-edge,env_isolation"
