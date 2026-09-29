#!/bin/sh
# Bash ツールなど非対話の呼び出し元から `fandhe-edge infer` を実行し、
# 終了コードと stdout（JSON）を機械的に受け取れることを確認するスクリプト
# （REQ-36・TASK-36.1-1・#149）。
#
# 呼び出し元: Claude Code / Codex の Bash ツール、および
# crates/cli/tests/bash_noninteractive.rs（結合テスト）。
#
# 契約:
#   - 引数はすべて `fandhe-edge infer` へそのまま渡す（${1+"$@"} のみ。eval しない）
#   - stdin は /dev/null に閉じ、入力待ちを作らない（非対話の要点）
#   - stdout は CLI の出力を無加工で中継する（1 呼び出し 1 JSON。REQ-33）
#   - stderr は CLI の stderr に続けて診断 `exit_code=<N>` を 1 行だけ出す
#     （引数値・入力テキストは出さない）
#   - 実行時間（既定 300 秒。FANDHE_EDGE_TIMEOUT_SECS）と stdout 容量（1 MiB）に上限を
#     置き、超過時は子を終了して runtime_error(70) の JSON を返す（REQ-39）
#   - 終了コードは CLI のもの（7 種）をそのまま返す。契約外の値・バイナリ不在・出力なしは
#     runtime_error(70) へ写し、JSON を置き換え / 補う（stdout は常に 1 JSON）
#
# バイナリ: 環境変数 FANDHE_EDGE_BIN、無ければ
#   ${CARGO_TARGET_DIR:-<repo>/target}/debug/fandhe-edge
#
# 現状の制約: infer の実推論は TASK-33.1-2（#136）・前処理（#112）・
# ONNX 推論（#113）が未接続のため exit 70 を返す。exit 0 になるのは
# `--help` のみ。記録の保存形式は #150（TASK-36.1-2）の担当。
set -eu

if [ -n "${FANDHE_EDGE_BIN:-}" ]; then
    bin=$FANDHE_EDGE_BIN
else
    root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
    bin="${CARGO_TARGET_DIR:-$root/target}/debug/fandhe-edge"
fi

# 起動不能（不在・ディレクトリ・実行権限なし）でも契約（stdout 1 JSON・
# stderr に exit_code）を欠けさせず runtime_error を返す
launch_failed() {
    echo "fandhe-edge: binary not found or not executable" >&2
    printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge binary not found or not executable"}'
    echo "exit_code=70" >&2
    exit 70
}

# -x はディレクトリにも成功するため、通常ファイル（-f）であることも確認する
if [ ! -f "$bin" ] || [ ! -x "$bin" ]; then
    launch_failed
fi

# stdout は一時ファイルへ退避し、終了値の確定後に 1 JSON だけを出す
# （実行後に 126/127 を返す実行ファイルが JSON を出していても二重出力しない。REQ-33）
work=$(mktemp -d) || launch_failed
out=$work/out
marker=$work/limit
: >"$out"
trap 'rm -rf "$work"' EXIT

# 期限と出力上限（REQ-39 資源の上限。無期限待ち・無制限のディスク書き込みを作らない）。
# 期限は FANDHE_EDGE_TIMEOUT_SECS（1〜999 の整数秒。不正値は既定 300 へ戻す）、
# 出力上限は 1 MiB。超過時は子プロセスを終了し runtime_error(70) の JSON を返す
# （REQ-21・REQ-33）
timeout_secs=${FANDHE_EDGE_TIMEOUT_SECS:-300}
case "$timeout_secs" in
    '' | *[!0-9]* | 0 | ????*) timeout_secs=300 ;;
esac
max_ticks=$((timeout_secs * 10))
max_out_bytes=1048576
# ulimit -f は 512 バイト単位のブロック数。書き込みの絶対上限（超過は SIGXFSZ で終了）
max_out_blocks=2048

# 子は subshell で ulimit を掛けてから exec する。${1+"$@"}: 引数なしでも
# Bash 3.2 の set -u で abort しない
(
    ulimit -f "$max_out_blocks" 2>/dev/null || true
    exec "$bin" infer ${1+"$@"} </dev/null >"$out"
) &
child=$!

# 監視: 期限超過または出力上限超過で子を TERM（1 秒後に KILL）し、原因を印に残す。
# 呼び出し元のパイプを保持しないよう stdin / stdout / stderr を閉じる
(
    ticks=0
    while [ "$ticks" -lt "$max_ticks" ]; do
        sleep 0.1
        ticks=$((ticks + 1))
        size=$(wc -c <"$out" 2>/dev/null || echo 0)
        if [ "${size:-0}" -gt "$max_out_bytes" ]; then
            echo output_limit >"$marker"
            break
        fi
    done
    [ -e "$marker" ] || echo timeout >"$marker"
    kill -TERM "$child" 2>/dev/null || true
    sleep 1
    kill -KILL "$child" 2>/dev/null || true
) </dev/null >/dev/null 2>&1 &
watchdog=$!

if wait "$child"; then
    rc=0
else
    rc=$?
fi
kill "$watchdog" 2>/dev/null || true

# 上限超過は CLI の出力（途中まで）を捨て、原因を示す JSON へ置き換える
limit_kind=
if [ -e "$marker" ]; then
    limit_kind=$(cat "$marker")
fi
if [ -z "$limit_kind" ] && [ "$(wc -c <"$out")" -ge "$max_out_bytes" ]; then
    # ulimit -f（SIGXFSZ）が先に子を止めた場合
    limit_kind=output_limit
fi
case "$limit_kind" in
    timeout)
        printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge timed out"}' >"$out"
        rc=70
        ;;
    output_limit)
        printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge output exceeded size limit"}' >"$out"
        rc=70
        ;;
esac

# CLI の終了コードは 7 種（0/10/11/12/20/64/70）に固定のため、それ以外
# （126・127・シグナル終了の 128+N 等の契約外）は既存の stdout の有無にかかわらず
# runtime_error(70) の JSON へ置き換える（JSON の code と終了コードを一致させる。REQ-21）。
# 許可された終了コードでも stdout が空なら 1 呼び出し 1 JSON（REQ-33）を満たさないため、
# 同じく runtime_error(70) の JSON を補って 70 を返す
case "$rc" in
    0 | 10 | 11 | 12 | 20 | 64 | 70)
        if [ ! -s "$out" ]; then
            printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge produced no output"}' >"$out"
            rc=70
        fi
        ;;
    *)
        printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge terminated abnormally"}' >"$out"
        rc=70
        ;;
esac
cat "$out"
echo "exit_code=$rc" >&2
exit "$rc"
