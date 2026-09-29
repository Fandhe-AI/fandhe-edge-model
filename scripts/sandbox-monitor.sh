#!/bin/sh
# `scripts/sandbox-run.sh`（sandbox 下の 7 工程実行）を `log stream` の拒否ログ監視で包み、
# 本ツールの実行プロセスに起因する通信拒否が 0 件かを自動判定するスクリプト
# （REQ-38・TASK-38.1-2・#163。手法の出典は PoC-16 の monitored_vertical.sh）。
#
# `log show`（事後検索）では Sandbox の拒否ログが取れないことが PoC-16 で実測されているため、
# 実行の前に `log stream` を開始し、実行の後に止める（ライブ監視）。集計・判定は
# `scripts/sandbox_deny_report.py`（標準ライブラリのみ。`python3 -I` で起動）が行う。
#
# 呼び出し元: 人が macOS 実機で直接実行する（実機での完走確認と拒否ログの記録は人の担当）／
# crates/cli/tests/sandbox_monitor_script.rs（偽の `log`・偽の launcher を使うテストハーネス）。
#
# 使い方:
#   sandbox-monitor.sh --definition PATH --project-dir DIR --out-dir DIR
#                      [--candidates N] [--infer-text TEXT] [--smoke]
#   sandbox-run.sh と同じオプション体系（`--key VALUE` と `--key=VALUE`）。--project-dir は
#   存在しないこと、--out-dir は存在しないか空であること（違反は invalid_input(64)）。
#
# 契約:
#   - `/usr/bin/log stream --style ndjson --predicate 'process == "kernel" AND eventMessage CONTAINS "deny"'`
#     を sandbox-run.sh の実行前に開始し、終了後に停止する。述語・style は定数で、
#     弱める経路（オプション・環境変数）を設けない。log は絶対パスで PATH を探さない。
#     テスト専用の上書きは環境変数 FANDHE_EDGE_LOG_CMD で、上書き時は stdout・レポートに
#     log_stream_override:true・evidence_hint:"test_harness" を記録する
#   - log が実行可能な通常ファイルでなければ sandbox-run.sh を起動せず runtime_error(70)
#     （監視なしで実行して証拠を装わない。fail-closed）
#   - 待機時間: 開始後 3 秒・停止前の余裕 60 秒（PoC-16 と同じ）。環境変数
#     FANDHE_EDGE_LOG_STREAM_WARMUP_SECS（0〜60）・FANDHE_EDGE_LOG_STREAM_TAIL_SECS（0〜600）で
#     上書きでき（不正値は既定へ戻す）、既定と異なる値のときは evidence_hint を test_harness にする
#   - 出力先: <out-dir>/run/（sandbox-run.sh の出力。run.meta.json）・log_stream.ndjson（生ログ。
#     0600）・log_stream.stderr.log・network_report.json（集計レポート）・monitor.meta.json
#   - 生ログには他アプリのイベントが含まれるため PR・Issue へ転記しない（件数は
#     network_report.json を記録する）。工程の stdout・stderr・--infer-text・パスは保存しない
#   - stdout は集計 JSON を 1 行だけ出す（REQ-33。固定の文字列と件数のみ。利用者の値は出さない）。
#     終了コードは 7 種のみ（REQ-21）。判定の意味は sandbox_deny_report.py を参照
#   - 待機の後、log の生存とヘッダ行の出力を確認できなければ sandbox-run.sh を起動せず
#     runtime_error(70)（fail-closed）
#   - 監視が無効になる条件（stream の実行中の終了・容量超過・ヘッダ欠落・集計器の異常終了）は
#     network_verdict:"undeterminable"・runtime_error(70)
#   - 資源上限（REQ-39）: 生ログの容量上限（超過で log を KILL して判定不能）・停止の期限
#     （TERM の後 5 秒で KILL）・独立プロセスグループと監視役（本スクリプトが突然死しても
#     log を KILL する）
#   - 陽性対照（curl を sandbox 下で実行して拒否の検出を確かめる）は TASK-38.2 の担当で
#     本スクリプトは実行しない。出力の positive_control は常に "not_run"
#
# 前提条件: sandbox-run.sh と同じ（cargo build・uv sync は sandbox の外で先に済ませる。
# 本スクリプトは通信せず、それらを実行しない）。
set -eu

# process group 隔離（set -m）のため通常モードの bash で再実行する（1 回だけ。無ければ fail-closed）
if [ -z "${BASH_VERSION:-}" ] || [ -o posix ]; then
    if [ -z "${FANDHE_EDGE_REEXEC:-}" ] && command -v bash >/dev/null 2>&1; then
        FANDHE_EDGE_REEXEC=1
        export FANDHE_EDGE_REEXEC
        exec bash "$0" ${1+"$@"}
    fi
    printf '%s\n' '{"code":"runtime_error","message":"bash is required for process group isolation"}'
    exit 70
fi

# 再実行の目印は子（sandbox-run.sh）へ引き継がない（子も自分で bash へ再実行する）
unset FANDHE_EDGE_REEXEC

umask 077

DEFAULT_LOG_CMD=/usr/bin/log
PREDICATE='process == "kernel" AND eventMessage CONTAINS "deny"'
# 生ログの容量上限（sandbox_deny_report.py の MAX_STREAM_BYTES と揃える）
MAX_STREAM_BYTES=268435456
# `log` の stderr の容量上限（超過は判定不能。REQ-39）
MAX_STREAM_ERR_BYTES=65536
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)

# 固定メッセージの JSON を 1 行出して終了する。$1=終了コード $2=code 名 $3=固定メッセージ
fail() {
    printf '{"code":"%s","message":"%s"}\n' "$2" "$3"
    exit "$1"
}

# ---- 引数の検証（ここで失敗したら何も起動しない） ----
definition=
project_dir=
out_dir=
candidates=
has_candidates=0
infer_text=
has_infer_text=0
smoke=0
seen=
while [ $# -gt 0 ]; do
    key=$1
    val=
    has_val=0
    case "$key" in
        --*=*)
            val=${key#*=}
            key=${key%%=*}
            has_val=1
            ;;
    esac
    case "$key" in
        --help)
            fail 0 ok "usage: sandbox-monitor.sh --definition PATH --project-dir DIR --out-dir DIR [--candidates N] [--infer-text TEXT] [--smoke]"
            ;;
        --smoke)
            [ "$has_val" -eq 0 ] || fail 64 invalid_input "option does not take a value"
            case "$seen" in *" smoke "*) fail 64 invalid_input "duplicate option" ;; esac
            seen="$seen smoke "
            smoke=1
            shift
            continue
            ;;
        --definition | --project-dir | --out-dir | --candidates | --infer-text) ;;
        *) fail 64 invalid_input "unknown option" ;;
    esac
    if [ "$has_val" -eq 0 ]; then
        [ $# -ge 2 ] || fail 64 invalid_input "missing option value"
        val=$2
        shift 2
    else
        shift
    fi
    case "$seen" in *" $key "*) fail 64 invalid_input "duplicate option" ;; esac
    seen="$seen $key "
    case "$key" in
        --definition) definition=$val ;;
        --project-dir) project_dir=$val ;;
        --out-dir) out_dir=$val ;;
        --candidates)
            candidates=$val
            has_candidates=1
            ;;
        --infer-text)
            infer_text=$val
            has_infer_text=1
            ;;
    esac
done

[ -n "$definition" ] || fail 64 invalid_input "--definition is required"
[ -n "$project_dir" ] || fail 64 invalid_input "--project-dir is required"
[ -n "$out_dir" ] || fail 64 invalid_input "--out-dir is required"
# sandbox-run.sh と同じ範囲検証。log stream を開始する前に拒否する（不正値のまま待機して
# 判定不能になるのを避ける。sandbox-run.sh 側の検証との共通化は別課題）
if [ "$has_candidates" -eq 1 ]; then
    case "$candidates" in
        [1-9] | [1-9][0-9]) ;;
        *) fail 64 invalid_input "--candidates must be an integer from 1 to 16" ;;
    esac
    [ "$candidates" -le 16 ] || fail 64 invalid_input "--candidates must be an integer from 1 to 16"
fi
if [ -e "$project_dir" ] || [ -L "$project_dir" ]; then
    fail 64 invalid_input "project directory already exists"
fi
if [ -e "$out_dir" ] || [ -L "$out_dir" ]; then
    if [ ! -d "$out_dir" ] || [ -L "$out_dir" ]; then
        fail 64 invalid_input "output directory is not a directory"
    fi
    [ -z "$(ls -A -- "$out_dir")" ] || fail 64 invalid_input "output directory is not empty"
fi

# パスを物理パスへ正規化する（sandbox-run.sh の canon_path と同じ方式。共通化は別課題）。
canon_out=
canon_path() {
    _p=$1
    case "$_p" in /*) ;; *) _p="$PWD/$_p" ;; esac
    _rest=
    while [ ! -e "$_p" ] && [ ! -L "$_p" ]; do
        _base=${_p##*/}
        _p=${_p%/*}
        [ -n "$_p" ] || _p=/
        case "$_base" in
            '' | .) ;;
            ..) return 1 ;;
            *) _rest="/$_base$_rest" ;;
        esac
    done
    [ -d "$_p" ] || return 1
    _phys=$(cd -P -- "$_p" && pwd -P) || return 1
    canon_out="${_phys%/}$_rest"
}
canon_path "$project_dir" || fail 64 invalid_input "cannot resolve project directory"
canon_project=$canon_out
canon_path "$out_dir" || fail 64 invalid_input "cannot resolve output directory"
case "$canon_out/" in
    "$canon_project/"*) fail 64 invalid_input "output directory must be separate from project directory" ;;
esac
case "$canon_project/" in
    "$canon_out/"*) fail 64 invalid_input "output directory must be separate from project directory" ;;
esac

# ---- 前提（log・sandbox-run.sh・python3）。sandbox-run.sh を起動する前に fail-closed で確認する ----
log_cmd=$DEFAULT_LOG_CMD
log_override=false
if [ -n "${FANDHE_EDGE_LOG_CMD:-}" ]; then
    log_cmd=$FANDHE_EDGE_LOG_CMD
    log_override=true
fi
if [ ! -f "$log_cmd" ] || [ ! -x "$log_cmd" ]; then
    fail 70 runtime_error "log command not found or not executable"
fi
run_script="$here/sandbox-run.sh"
report_script="$here/sandbox_deny_report.py"
if [ ! -f "$run_script" ] || [ ! -f "$report_script" ]; then
    fail 70 runtime_error "companion script not found"
fi
command -v python3 >/dev/null 2>&1 || fail 70 runtime_error "python3 is required for the report"

# 待機時間（先頭 0 は算術式で 8 進解釈されるため 0 単独以外の先頭 0 は不正値として既定へ戻す）
bounded_secs() { # $1=値 $2=既定 $3=上限
    case "$1" in
        '' | *[!0-9]* | ??????) echo "$2" ;;
        0) echo 0 ;;
        0*) echo "$2" ;;
        *) if [ "$1" -le "$3" ]; then echo "$1"; else echo "$2"; fi ;;
    esac
}
warmup=$(bounded_secs "${FANDHE_EDGE_LOG_STREAM_WARMUP_SECS:-3}" 3 60)
tail_secs=$(bounded_secs "${FANDHE_EDGE_LOG_STREAM_TAIL_SECS:-60}" 60 600)

mkdir -p -- "$out_dir" || fail 70 runtime_error "cannot create output directory"
probe="$out_dir/.write-probe.$$"
{ : >"$probe"; } 2>/dev/null || fail 70 runtime_error "cannot write monitor record"
rm -f -- "$probe"

stream_file="$out_dir/log_stream.ndjson"
stream_err="$out_dir/log_stream.stderr.log"
report_file="$out_dir/network_report.json"
meta_file="$out_dir/monitor.meta.json"

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

logpid=
wd=
# 終了時の後始末。log と監視役が残っていれば KILL する
cleanup() {
    if [ -n "$logpid" ]; then
        kill -s KILL -- "-$logpid" 2>/dev/null || true
    fi
    if [ -n "$wd" ]; then
        kill -s KILL -- "-$wd" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 70' TERM INT HUP

# ---- 監視の開始（sandbox-run.sh の実行より前） ----
monitor_started=$(utc_now)
set -m
(
    # stdout・stderr とも head -c で書き込み時点に上限 +1 バイトで打ち切る（REQ-39。
    # sandbox-run.sh・cli-infer-noninteractive.sh と同じ方式）。head も同じグループ。
    # 上限に達すると head が終了して log は書き込み失敗で止まり、超過は下の検査で判定不能になる
    exec "$log_cmd" stream --style ndjson --predicate "$PREDICATE" </dev/null \
        > >(head -c $((MAX_STREAM_BYTES + 1)) >"$stream_file") \
        2> >(head -c $((MAX_STREAM_ERR_BYTES + 1)) >"$stream_err")
) </dev/null >/dev/null 2>&1 &
logpid=$!
# 監視役（独立グループ）: 本スクリプトが突然死しても log を KILL し、容量超過でも KILL する
wrapper_pid=$$
(
    while kill -0 "$wrapper_pid" 2>/dev/null; do
        _sz=$(wc -c <"$stream_file" 2>/dev/null || echo 0)
        _esz=$(wc -c <"$stream_err" 2>/dev/null || echo 0)
        if [ "${_sz:-0}" -gt "$MAX_STREAM_BYTES" ] || [ "${_esz:-0}" -gt "$MAX_STREAM_ERR_BYTES" ]; then
            kill -s KILL -- "-$logpid" 2>/dev/null || true
        fi
        sleep 1
    done
    kill -s KILL -- "-$logpid" 2>/dev/null || true
) </dev/null >/dev/null 2>&1 &
wd=$!
set +m

sleep "$warmup"

# 監視が有効であることを確認してから実行する（fail-closed。REQ-38）。log が生きていて、
# ヘッダ行が生ログの先頭に出るまで最大 5 秒待つ。満たせなければ sandbox-run.sh を起動せず 70
# （後始末は EXIT trap が行う）。
gate_ok=0
i=0
while [ "$i" -le 50 ]; do
    kill -0 "$logpid" 2>/dev/null || break
    if [ "$(head -c 22 -- "$stream_file" 2>/dev/null)" = "Filtering the log data" ]; then
        # ヘッダ出力直後の即終了を拾うため、短い猶予の後にもう一度生存を確認する
        sleep 0.3
        kill -0 "$logpid" 2>/dev/null && gate_ok=1
        break
    fi
    sleep 0.1
    i=$((i + 1))
done
[ "$gate_ok" -eq 1 ] || fail 70 runtime_error "log stream is not active; sandbox run was not started"

# ---- 実行 ----
set -- --definition "$definition" --project-dir "$project_dir" --out-dir "$out_dir/run"
[ "$has_candidates" -eq 0 ] || set -- "$@" --candidates "$candidates"
[ "$has_infer_text" -eq 0 ] || set -- "$@" --infer-text "$infer_text"
[ "$smoke" -eq 0 ] || set -- "$@" --smoke
run_rc=0
"$run_script" "$@" </dev/null >/dev/null 2>&1 || run_rc=$?

sleep "$tail_secs"

# ---- 監視の停止 ----
stream_ok=1
kill -0 "$logpid" 2>/dev/null || stream_ok=0
stopped=$(utc_now)
if [ "$stream_ok" -eq 1 ]; then
    kill -s TERM -- "-$logpid" 2>/dev/null || true
    i=0
    while [ "$i" -lt 50 ] && kill -0 "$logpid" 2>/dev/null; do
        sleep 0.1
        i=$((i + 1))
    done
fi
kill -s KILL -- "-$logpid" 2>/dev/null || true
wait "$logpid" 2>/dev/null || true
logpid=
kill -s KILL -- "-$wd" 2>/dev/null || true
wait "$wd" 2>/dev/null || true
wd=

# ---- 集計 ----
size=$(wc -c <"$stream_file" 2>/dev/null || echo 0)
set -- --stream "$stream_file" --run-meta "$out_dir/run/run.meta.json" \
    --monitor-started-utc "$monitor_started" --monitor-stopped-utc "$stopped" \
    --warmup-secs "$warmup" --tail-secs "$tail_secs" --report-out "$report_file"
[ "$log_override" = false ] || set -- "$@" --log-override
[ "$stream_ok" -eq 1 ] || set -- "$@" --stream-died
esize=$(wc -c <"$stream_err" 2>/dev/null || echo 0)
{ [ "${size:-0}" -le "$MAX_STREAM_BYTES" ] && [ "${esize:-0}" -le "$MAX_STREAM_ERR_BYTES" ]; } \
    || set -- "$@" --stream-overflow
# run の実際の終了コードを渡し、集計器が run.meta.json の exit_code と照合する（不一致は判定不能）
set -- "$@" --run-exit-code "$run_rc"
rep_rc=0
rep_out=$(python3 -I "$report_script" "$@" </dev/null 2>/dev/null) || rep_rc=$?

case "$rep_rc" in 0 | 10 | 11 | 12 | 20 | 64 | 70) ;; *) rep_rc=70 ;; esac
if [ -z "$rep_out" ]; then
    rep_rc=70
    rep_out='{"code":"runtime_error","message":"report generator failed","network_verdict":"undeterminable","positive_control":"not_run"}'
fi

# 述語は PREDICATE から JSON エスケープして埋め込む（定数のため `"` の置換のみで足りる）
predicate_json=$(printf '%s' "$PREDICATE" | sed 's/"/\\"/g')
if ! printf '{"monitor_started_utc":"%s","monitor_stopped_utc":"%s","warmup_secs":%s,"tail_secs":%s,"log_stream_override":%s,"stream_alive_at_stop":%s,"sandbox_run_exit_code":%s,"report_exit_code":%s,"predicate":"%s"}\n' \
    "$monitor_started" "$stopped" "$warmup" "$tail_secs" "$log_override" \
    "$([ "$stream_ok" -eq 1 ] && echo true || echo false)" "$run_rc" "$rep_rc" "$predicate_json" \
    >"$meta_file" 2>/dev/null; then
    fail 70 runtime_error "cannot write monitor record"
fi

printf '%s\n' "$rep_out"
exit "$rep_rc"
