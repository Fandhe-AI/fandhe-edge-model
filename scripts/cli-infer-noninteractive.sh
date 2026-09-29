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
#   - stdout は CLI の出力を中継する（1 呼び出し 1 JSON。REQ-33）
#   - stderr は CLI の stderr に続けて診断 `exit_code=<N>` を 1 行だけ出す
#     （引数値・入力テキストは出さない）
#   - 実行時間（既定 300 秒。FANDHE_EDGE_TIMEOUT_SECS）・stdout 容量（1 MiB）・
#     stderr 容量（64 KiB）に上限を置き、超過時は子（子孫プロセスを含む）を終了して
#     runtime_error(70) の JSON を返す（REQ-39）
#   - 許可された終了コードでも stdout が「完結した JSON オブジェクト 1 つ（1 行）」でなければ
#     runtime_error(70) の JSON へ置き換える（複数 JSON・途中切れを中継しない。REQ-33）
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

# シェル自身の診断・ジョブ終了通知（macOS の bash が出す "Terminated: 15" 等）が
# stderr を汚さないよう、契約の出力は fd 3（元の stderr）へだけ出し、fd 2 は後で捨てる
exec 3>&2

if [ -n "${FANDHE_EDGE_BIN:-}" ]; then
    bin=$FANDHE_EDGE_BIN
else
    root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
    bin="${CARGO_TARGET_DIR:-$root/target}/debug/fandhe-edge"
fi

# 起動不能（不在・ディレクトリ・実行権限なし）でも契約（stdout 1 JSON・
# stderr に exit_code）を欠けさせず runtime_error を返す
launch_failed() {
    echo "fandhe-edge: binary not found or not executable" >&3
    printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge binary not found or not executable"}'
    echo "exit_code=70" >&3
    exit 70
}

# -x はディレクトリにも成功するため、通常ファイル（-f）であることも確認する
if [ ! -f "$bin" ] || [ ! -x "$bin" ]; then
    launch_failed
fi

# stdout・stderr は一時ファイルへ退避し、終了値の確定後に 1 JSON だけを出す
# （実行後に 126/127 を返す実行ファイルが JSON を出していても二重出力しない。REQ-33）
work=$(mktemp -d) || launch_failed
out=$work/out
err=$work/err
rcf=$work/rc
: >"$out"
: >"$err"
trap 'rm -rf "$work"' EXIT

# 期限と出力上限（REQ-39 資源の上限。無期限待ち・無制限のディスク書き込みを作らない）。
# 期限は FANDHE_EDGE_TIMEOUT_SECS（先頭 0 なしの 1〜999 の整数秒。不正値と先頭 0
# （算術式で 8 進数解釈される事故を避ける）は既定 300 へ戻す）、stdout 上限は 1 MiB、
# stderr 上限は 64 KiB。超過時は子プロセス木を終了し runtime_error(70) の JSON を返す
# （REQ-21・REQ-33）
timeout_secs=${FANDHE_EDGE_TIMEOUT_SECS:-300}
case "$timeout_secs" in
    '' | *[!0-9]* | 0* | ????*) timeout_secs=300 ;;
esac
max_ticks=$((timeout_secs * 10))
max_out_bytes=1048576
max_err_bytes=65536
# ulimit -f は 512 バイト単位のブロック数。各ファイルへの書き込みの絶対上限
# （超過は SIGXFSZ で終了）
max_out_blocks=2048

# 期限超過時に子孫も終了できるよう、tree_pids で子を根とするプロセス木を辿る
# （ジョブ制御 set -m は tty の無い非対話環境で使えないため使わない）。
# 終了値は rc ファイルへ書き、監視側は kill -0 でなくファイルで終了を判定する
# （終了済みの子へ遅延 KILL を送る競合と PID 再利用を避ける）。
# 子へ元の stderr（fd 3）を継がせない（呼び出し元のパイプを保持させない）。
# ${1+"$@"}: 引数なしでも Bash 3.2 の set -u で abort しない
exec 2>/dev/null
(
    ulimit -f "$max_out_blocks" 2>/dev/null || true
    rc_child=0
    "$bin" infer ${1+"$@"} </dev/null >"$out" 2>"$err" || rc_child=$?
    echo "$rc_child" >"$rcf.tmp"
    mv "$rcf.tmp" "$rcf"
) </dev/null >/dev/null 3>&- &
child=$!

# child を根とする子孫の PID 一覧（ps の pid / ppid から推移閉包を取る。macOS・Linux 共通）
tree_pids() {
    ps -A -o pid= -o ppid= 2>/dev/null | awk -v root="$child" '
        { p[NR] = $1; q[NR] = $2; n = NR }
        END {
            seen[root] = 1
            do {
                grew = 0
                for (i = 1; i <= n; i++)
                    if ((q[i] in seen) && !(p[i] in seen)) { seen[p[i]] = 1; grew = 1 }
            } while (grew)
            for (k in seen) print k
        }'
}

# 監視（前景ループ）: 子の終了・期限超過・容量超過のいずれかで抜ける
ticks=0
limit_kind=
while [ ! -e "$rcf" ]; do
    if [ "$ticks" -ge "$max_ticks" ]; then
        limit_kind=timeout
        break
    fi
    sleep 0.1
    ticks=$((ticks + 1))
    size=$(wc -c <"$out" 2>/dev/null || echo 0)
    if [ "${size:-0}" -gt "$max_out_bytes" ]; then
        limit_kind=output_limit
        break
    fi
    esize=$(wc -c <"$err" 2>/dev/null || echo 0)
    if [ "${esize:-0}" -gt "$max_err_bytes" ]; then
        limit_kind=stderr_limit
        break
    fi
done

# 上限超過時は、未回収の子（PID が再利用されない）から辿ったプロセス木を一度だけ列挙し、
# 同じ一覧へ TERM → 1 秒後に KILL を送る（親を先に落として孫が孤児化しても取りこぼさない）
if [ -n "$limit_kind" ]; then
    victims=$(tree_pids)
    for p in $victims; do kill -s TERM "$p" 2>/dev/null || true; done
    sleep 1
    for p in $victims; do kill -s KILL "$p" 2>/dev/null || true; done
fi
wait "$child" || true

rc=70
if [ -z "$limit_kind" ] && [ -r "$rcf" ]; then
    rc=$(cat "$rcf")
    case "$rc" in
        '' | *[!0-9]*) rc=137 ;;
    esac
fi
if [ -z "$limit_kind" ]; then
    # ulimit -f（SIGXFSZ）が先に子を止めた場合
    if [ "$(wc -c <"$out")" -ge "$max_out_bytes" ]; then
        limit_kind=output_limit
    elif [ "$(wc -c <"$err")" -gt "$max_err_bytes" ]; then
        limit_kind=stderr_limit
    fi
fi

# 上限超過は CLI の出力（途中まで）を捨て、原因を示す JSON へ置き換える
case "$limit_kind" in
    timeout)
        printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge timed out"}' >"$out"
        : >"$err"
        rc=70
        ;;
    output_limit)
        printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge output exceeded size limit"}' >"$out"
        : >"$err"
        rc=70
        ;;
    stderr_limit)
        printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge stderr exceeded size limit"}' >"$out"
        : >"$err"
        rc=70
        ;;
esac

# stdout が「完結した JSON オブジェクト 1 つ（1 行）」かを検査する（awk の状態機械。
# 文字列・エスケープを追跡し、閉じ忘れ・後続の余分な JSON・複数行を不正とする）
is_single_json_object() {
    awk '
        NR > 1 { bad = 1 }
        NR == 1 {
            n = length($0); d = 0; s = 0; e = 0; done = 0
            for (i = 1; i <= n; i++) {
                c = substr($0, i, 1)
                if (done) { if (c != " " && c != "\r") bad = 1; continue }
                if (s) {
                    if (e) e = 0
                    else if (c == "\\") e = 1
                    else if (c == "\"") s = 0
                    continue
                }
                if (c == "\"") { if (d == 0) { bad = 1; break }; s = 1; continue }
                if (c == "{" || c == "[") { if (d == 0 && c != "{") { bad = 1; break }; d++; continue }
                if (c == "}" || c == "]") { d--; if (d < 0) { bad = 1; break }; if (d == 0) done = 1; continue }
                if (d == 0 && c != " ") { bad = 1; break }
            }
        }
        END { exit (bad || !done || NR != 1) ? 1 : 0 }
    ' "$out"
}

# CLI の終了コードは 7 種（0/10/11/12/20/64/70）に固定のため、それ以外
# （126・127・シグナル終了の 128+N 等の契約外）は既存の stdout の有無にかかわらず
# runtime_error(70) の JSON へ置き換える（JSON の code と終了コードを一致させる。REQ-21）。
# 許可された終了コードでも stdout が空なら補い、JSON 1 つ（1 行）でなければ
# （複数 JSON・途中切れ）置き換えて 70 を返す（1 呼び出し 1 JSON。REQ-33）
case "$rc" in
    0 | 10 | 11 | 12 | 20 | 64 | 70)
        if [ ! -s "$out" ]; then
            printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge produced no output"}' >"$out"
            rc=70
        elif ! is_single_json_object; then
            printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge produced invalid output"}' >"$out"
            rc=70
        fi
        ;;
    *)
        printf '%s\n' '{"code":"runtime_error","message":"fandhe-edge terminated abnormally"}' >"$out"
        rc=70
        ;;
esac
cat "$out"
# CLI の stderr（上限 64 KiB 以内）を中継してから診断行を出す
cat "$err" >&3
echo "exit_code=$rc" >&3
exit "$rc"
