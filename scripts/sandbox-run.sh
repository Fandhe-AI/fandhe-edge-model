#!/bin/sh
# sandbox（通信遮断）下で `fandhe-edge` の 7 工程
# （register → inspect → train → select → evaluate（選定された候補のみ。REQ-27）→ package → infer）を順に実行し、
# 学習・推論・評価が通信なしで完走することを確認するスクリプト
# （REQ-38・TASK-38.1-1・#162。手法の出典は PoC-14・PoC-16 の run_vertical.sh /
# monitored_vertical.sh）。
#
# 呼び出し元: 人が macOS 実機で直接実行する（REQ-38 の実機確認は人の担当）／
# `scripts/sandbox-monitor.sh`（TASK-38.1-2・#163。このスクリプトを `log stream` で包む）／
# crates/cli/tests/sandbox_run_script.rs（偽の launcher を使うテストハーネス）。
# 本スクリプトは拒否ログの監視・0 件の集計をしない（sandbox-monitor.sh と sandbox_deny_report.py の担当）。
#
# 使い方:
#   sandbox-run.sh --definition PATH --project-dir DIR --out-dir DIR
#                  [--candidates N] [--infer-text TEXT] [--smoke] [--extended]
#   値を取るオプションは `--key VALUE` と `--key=VALUE` の両方を受け付ける。
#   --project-dir は存在しないこと、--out-dir は存在しないか空であること
#   （既存の内容を削除・上書きしない。違反は invalid_input(64)）。
#
# 契約:
#   - `--extended`（任意。#469 の CLI 結線で増えた工程・引数も通信 0 件で完走することの確認用）:
#     上の 7 工程に続けて、別プロジェクト <project-dir>-extended（存在しないこと。--project-dir と同じ
#     検証）で register・inspect・`train --all --smoke --budget-seconds 600`・`train --status`・
#     `train --cancel` を実行し、最後に本プロジェクトの package へ
#     `infer --version-ledger <project-dir>/version_ledger.json --version-id v1` を実行する。
#     `train --all` は既存の候補ディレクトリを拒否するため別プロジェクトで行う（実機では実 trainer が要る）。
#     校正・保留（calibration・abstention）は上の evaluate が出し、--extended のときは評価ありの
#     evaluate の結果に calibration・abstention がオブジェクトで含まれることも構造検証する
#     （評価データありかつ smoke なしで使う。`fixtures/sandbox_run_eval`）。版管理台帳は上の package が
#     `<project-dir>/version_ledger.json` として作る。許容する終了コードは全工程 0（infer は従来どおり
#     11・12 も可）。--extended なしの挙動・出力は変えない
#   - 各工程を `<launcher> -p '(version 1)(allow default)(deny network*)' <bin> <工程> ...`
#     で起動する。遮断プロファイルは定数で、弱める経路（オプション・環境変数）を設けない。
#     train が起動する学習ワーカー（Python）の子プロセスも sandbox を継承する前提で、
#     根拠は PoC-16 の実測（子プロセスを含めて拒否 0 件・陽性対照で検出を確認）。陽性対照は
#     sandbox-monitor.sh が監視窓の中で実行する（TASK-38.2・#164。本スクリプトは実行しない）
#   - launcher は絶対パス /usr/bin/sandbox-exec（PATH で探さない）。テスト専用の上書きは
#     環境変数 FANDHE_EDGE_SANDBOX_EXEC で、上書き時は stdout と run.meta.json に
#     sandbox_exec_override:true・evidence_hint:"test_harness" を記録する
#   - launcher が実行可能な通常ファイルでなければ CLI を起動せず runtime_error(70)
#     （遮断なしで実行して「sandbox 下で完走」と誤記録しない。fail-closed）
#   - 工程が 0 以外で終了したらその工程で停止し、以降を起動しない。evaluate が
#     status:"skipped"（exit 0）を返したら続行しつつ工程の status を skipped と記録する
#     （評価済みを装わない。REQ-17）
#   - stdout は集計 JSON を 1 行だけ出す（REQ-33）。JSON には固定値だけを埋め込み、
#     利用者の値（パス・テキスト）を出さない。工程の記録は工程名・終了コード・時刻・
#     出力バイト数だけで、argv の値は記録しない（データ本文を残さない。security.md）
#   - 終了コード: 全工程 0 なら 0。停止時はその工程の終了コード（7 種の値のみ。
#     契約外は 70）。検証エラーは 64、launcher・バイナリの不在・期限切れ・出力超過は 70（REQ-21）
#   - 資源上限（REQ-39）: 工程ごとの期限（FANDHE_EDGE_SANDBOX_STEP_TIMEOUT_SECS。
#     1〜86400 の整数秒、不正値は既定 3600）、stdout 1 MiB・stderr 8 MiB の容量。超過時は
#     工程のプロセスグループごと KILL して 70。stdin は /dev/null。
#     プロセス管理の構造は cli-infer-noninteractive.sh（#149）と同じ（共通化は別課題）
#   - 出力先: <out-dir>/run.meta.json のみ（sandbox 下で実行した時間帯 started_utc/ended_utc
#     と工程ごとのバイト数を sandbox-monitor.sh へ引き渡す）。工程の stdout・stderr の本文は永続化しない
#     （推論結果・エラーに学習・評価データの本文が含まれうるため。security.md）。
#     容量検査のために一時ディレクトリへ受けるが、終了時に必ず削除する
#   - 終了コード 0 の工程は stdout を python3 の json で構造検証する。空でない単一の JSON オブジェクトで
#     ないか、トップレベルに code があるのに "ok" でなければ、工程ごとの契約（step が工程名と一致し
#     status が "ok"。evaluate のみ "skipped" も可。infer は step を持たず status:"ok"・id・predicted_label。
#     infer の exit 11・12 は判定行の status が out_of_scope・abstain なら完了として扱い、元の終了コードを記録する）
#     に合わなければ、空出力・不正 JSON・code 不整合を含めて runtime_error(70) で停止する（fail-closed。REQ-21・REQ-33）。exit 0 の工程結果 JSON
#     （register・inspect・train・evaluate・select・package と infer の判定）は code を持たず
#     step・status 等のフィールドを持つ契約のため、code の欠如は許容する（cli-infer-noninteractive.sh
#     と同じ規則。TASK-33.1-2・#136）。evaluate はさらに
#     トップレベルの status だけを見て skipped を判定する（REQ-17）
#   - --out-dir は --project-dir と同一・包含関係にないこと（両パスを物理パスへ正規化して比較。
#     違反は invalid_input(64)。mkdir が未作成の project-dir を先に作る契約違反を防ぐ）
#
# 前提条件（通信を伴う準備は sandbox の外で先に済ませる。REQ-38。本スクリプトは
# cargo build・uv sync を実行しない）: ビルド済みバイナリ
# （環境変数 FANDHE_EDGE_BIN、無ければ ${CARGO_TARGET_DIR:-<repo>/target}/debug/fandhe-edge）、
# 同期済みの trainer/.venv、定義ファイルと学習・評価データ。
#
# 工程の接続（TASK-33.1-2・#136）: 実バイナリは 7 工程を下位層へ接続済み。データは定義ファイルと
# 同じディレクトリの固定名 train.jsonl（必須）・evaluation.jsonl（任意）から register が取り込む。
# 学習ワーカーは環境変数 FANDHE_EDGE_TRAINER_DIR（絶対パス）で指す（未設定なら開発ツリーの trainer/）。
# 経路の閉じ込め（REQ-39）により、--definition・--project-dir は実行時のカレントディレクトリ配下に
# 置くこと。package の出力先は <project-dir>/package で確定。評価データがあるときの evaluate は
# 選定された候補に凍結データを 1 回だけ適用する（#314）。
set -eu

# 期限監視のプロセスグループ隔離（set -m）と process substitution のため通常モードの
# bash で再実行する（1 回だけ。無ければ fail-closed）
if [ -z "${BASH_VERSION:-}" ] || [ -o posix ]; then
    if [ -z "${FANDHE_EDGE_REEXEC:-}" ] && command -v bash >/dev/null 2>&1; then
        FANDHE_EDGE_REEXEC=1
        export FANDHE_EDGE_REEXEC
        exec bash "$0" ${1+"$@"}
    fi
    printf '%s\n' '{"code":"runtime_error","message":"bash is required for process group isolation"}'
    exit 70
fi

PROFILE='(version 1)(allow default)(deny network*)'
DEFAULT_LAUNCHER=/usr/bin/sandbox-exec
MAX_OUT_BYTES=1048576
MAX_ERR_BYTES=8388608

# 固定メッセージの JSON を 1 行出して終了する。$1=終了コード $2=code 名 $3=固定メッセージ
fail() {
    printf '{"code":"%s","message":"%s"}\n' "$2" "$3"
    exit "$1"
}

# ---- 引数の検証（ここで失敗したら CLI を起動しない） ----
definition=
project_dir=
out_dir=
candidates=1
infer_text='sandbox check 0123456789'
smoke=0
extended=0
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
            fail 0 ok "usage: sandbox-run.sh --definition PATH --project-dir DIR --out-dir DIR [--candidates N] [--infer-text TEXT] [--smoke] [--extended]"
            ;;
        --extended)
            [ "$has_val" -eq 0 ] || fail 64 invalid_input "option does not take a value"
            case "$seen" in *" extended "*) fail 64 invalid_input "duplicate option" ;; esac
            seen="$seen extended "
            extended=1
            shift
            continue
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
        --candidates) candidates=$val ;;
        --infer-text) infer_text=$val ;;
    esac
done

[ -n "$definition" ] || fail 64 invalid_input "--definition is required"
[ -n "$project_dir" ] || fail 64 invalid_input "--project-dir is required"
[ -n "$out_dir" ] || fail 64 invalid_input "--out-dir is required"
case "$candidates" in
    [1-9] | [1-9][0-9]) ;;
    *) fail 64 invalid_input "--candidates must be an integer from 1 to 16" ;;
esac
[ "$candidates" -le 16 ] || fail 64 invalid_input "--candidates must be an integer from 1 to 16"
if [ -e "$project_dir" ] || [ -L "$project_dir" ]; then
    fail 64 invalid_input "project directory already exists"
fi
if [ -e "$out_dir" ] || [ -L "$out_dir" ]; then
    if [ ! -d "$out_dir" ] || [ -L "$out_dir" ]; then
        fail 64 invalid_input "output directory is not a directory"
    fi
    [ -z "$(ls -A -- "$out_dir")" ] || fail 64 invalid_input "output directory is not empty"
fi

# パスを物理パスへ正規化する（未作成の末尾は字句的に連結。macOS に realpath -m が無いため）。
# 未作成部分に `..` を含む・祖先がディレクトリでない場合は 1 を返す。結果は canon_out へ入れる
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
canon_project=
canon_path "$project_dir" || fail 64 invalid_input "cannot resolve project directory"
canon_project=$canon_out
canon_path "$out_dir" || fail 64 invalid_input "cannot resolve output directory"
# --out-dir の作成が未作成の --project-dir を先に作ってしまわないよう、同一・包含を拒否する
case "$canon_out/" in
    "$canon_project/"*) fail 64 invalid_input "output directory must be separate from project directory" ;;
esac
case "$canon_project/" in
    "$canon_out/"*) fail 64 invalid_input "output directory must be separate from project directory" ;;
esac

# --extended の別プロジェクト（train --all は既存の候補ディレクトリを拒否するため分ける）
ext_project="${project_dir}-extended"
if [ "$extended" -eq 1 ]; then
    if [ -e "$ext_project" ] || [ -L "$ext_project" ]; then
        fail 64 invalid_input "extended project directory already exists"
    fi
    canon_path "$ext_project" || fail 64 invalid_input "cannot resolve extended project directory"
    _ext=$canon_out
    canon_path "$out_dir" || fail 64 invalid_input "cannot resolve output directory"
    case "$canon_out/" in
        "$_ext/"*) fail 64 invalid_input "output directory must be separate from project directory" ;;
    esac
    case "$_ext/" in
        "$canon_out/"*) fail 64 invalid_input "output directory must be separate from project directory" ;;
    esac
fi

# ---- 前提（launcher・バイナリ）。CLI を起動する前に fail-closed で確認する ----
override=false
launcher=$DEFAULT_LAUNCHER
if [ -n "${FANDHE_EDGE_SANDBOX_EXEC:-}" ]; then
    launcher=$FANDHE_EDGE_SANDBOX_EXEC
    override=true
fi
if [ ! -f "$launcher" ] || [ ! -x "$launcher" ]; then
    fail 70 runtime_error "sandbox launcher not found or not executable"
fi
if [ -n "${FANDHE_EDGE_BIN:-}" ]; then
    bin=$FANDHE_EDGE_BIN
else
    root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
    bin="${CARGO_TARGET_DIR:-$root/target}/debug/fandhe-edge"
fi
if [ ! -f "$bin" ] || [ ! -x "$bin" ]; then
    fail 70 runtime_error "fandhe-edge binary not found or not executable"
fi

# 工程ごとの期限（先頭 0 は算術式で 8 進解釈されるため不正値として既定へ戻す）
step_timeout=${FANDHE_EDGE_SANDBOX_STEP_TIMEOUT_SECS:-3600}
case "$step_timeout" in
    '' | *[!0-9]* | 0* | ??????) step_timeout=3600 ;;
esac
[ "$step_timeout" -le 86400 ] || step_timeout=3600

mkdir -p -- "$out_dir" || fail 70 runtime_error "cannot create output directory"
# 書き込み不能な既存の空ディレクトリを工程実行の前に検出する（set -e で JSON なしに終了させない。REQ-21・REQ-33）
probe="$out_dir/.write-probe.$$"
{ : >"$probe"; } 2>/dev/null || fail 70 runtime_error "cannot write run record"
rm -f -- "$probe"

work=$(mktemp -d) || fail 70 runtime_error "cannot create temporary directory"
child=
wd=
reaped=1
# 終了時の後始末。未回収の工程グループと監視役が残っていれば KILL する
cleanup() {
    if [ "$reaped" -eq 0 ] && [ -n "$child" ]; then
        kill -s KILL -- "-$child" 2>/dev/null || true
    fi
    if [ -n "$wd" ]; then
        kill -s KILL -- "-$wd" 2>/dev/null || true
    fi
    rm -rf "$work"
}
trap cleanup EXIT
trap 'exit 70' TERM INT HUP

# 工程グループ（pgid = $child）に属するプロセスの PID を run_pids へ集める。拒否ログ
# （sandbox-monitor.sh・sandbox_deny_report.py）が「本ツール起因」を PID で照合するための
# 記録で、run.meta.json の process_pids に書く（REQ-38・TASK-38.1-2）。pgrep の取得は
# ポーリング間隔（0.1 秒）ごとで、それより短命なプロセスは記録できない（その拒否は
# 帰属不明として pending になる。過大に本ツール起因と断定しない側）。上限 MAX_RUN_PIDS 件。
run_pids=
run_pid_n=0
MAX_RUN_PIDS=4096
sample_pids() {
    [ "$run_pid_n" -lt "$MAX_RUN_PIDS" ] || return 0
    # pgrep -g は工程グループの PID を 1 行 1 件で直接列挙する（ps の列幅・桁あふれに依存しない。
    # macOS・Linux の procps 共通）。pgrep が無い・失敗・空の場合は何も記録しない
    # （その拒否は帰属不明 pending になる。fail-closed）
    _new=$(pgrep -g "$child" 2>/dev/null | awk -v seen="$run_pids" '
        BEGIN { n = split(seen, a, " "); for (i = 1; i <= n; i++) s[a[i]] = 1 }
        $1 ~ /^[0-9]+$/ && !($1 in s) { s[$1] = 1; printf "%s ", $1 }') || return 0
    for _p in $_new; do
        [ "$run_pid_n" -lt "$MAX_RUN_PIDS" ] || break
        run_pids="$run_pids$_p "
        run_pid_n=$((run_pid_n + 1))
    done
}

# 工程グループに生存プロセスが残っているか。0=生存 1=なし 2=一覧取得失敗（生存不明）
group_alive() {
    _ps=$(ps -A -o pgid= -o stat= 2>/dev/null) || return 2
    [ -n "$_ps" ] || return 2
    printf '%s\n' "$_ps" | awk -v g="$child" '
        $1 == g && $2 !~ /^Z/ { found = 1 }
        END { exit found ? 0 : 1 }'
}

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# 終了コードから 7 種の code 名へ写す（fixtures/exitcode/exit_codes.json と同一。REQ-21）
code_name() {
    case "$1" in
        0) echo ok ;;
        10) echo judged_fail ;;
        11) echo out_of_scope ;;
        12) echo pending ;;
        20) echo limit_exceeded ;;
        64) echo invalid_input ;;
        *) echo runtime_error ;;
    esac
}

steps_json=
steps_meta=
step_no=0
failed_step=null
final_rc=0
step_rc=0

# 1 工程を sandbox 下で起動し、契約内へ写した終了コードを step_rc に入れる。
# 引数: <ファイル接頭辞> <CLI の argv...>
run_step() {
    prefix=$1
    shift
    so="$work/$prefix.stdout"
    se="$work/$prefix.stderr"
    rcf="$work/rc"
    rm -f "$rcf" "$rcf.tmp"
    : >"$so"
    : >"$se"
    # テスト専用の絞り込み: STEP_TIMEOUT_ONLY が設定されていれば、その工程名だけに短い期限を掛け、
    # 他工程は既定 3600 秒にする（負荷下で前段が短い期限を超えるのを避ける。未設定なら全工程に掛ける）
    this_timeout=$step_timeout
    if [ -n "${FANDHE_EDGE_SANDBOX_STEP_TIMEOUT_ONLY:-}" ] && [ "$name" != "$FANDHE_EDGE_SANDBOX_STEP_TIMEOUT_ONLY" ]; then
        this_timeout=3600
    fi
    deadline=$((SECONDS + this_timeout + 1))
    # 工程は set -m で独立したプロセスグループ（pgid = 子の PID）として起動する
    set -m
    (
        rc_child=0
        # head -c で書き込み時点に上限 +1 バイトで打ち切る（REQ-39）。head も同じグループ
        "$launcher" -p "$PROFILE" "$bin" "$@" </dev/null \
            > >(head -c $((MAX_OUT_BYTES + 1)) >"$so") \
            2> >(head -c $((MAX_ERR_BYTES + 1)) >"$se") || rc_child=$?
        echo "$rc_child" >"$rcf.tmp"
        mv "$rcf.tmp" "$rcf"
    ) </dev/null >/dev/null 2>&1 &
    child=$!
    reaped=0
    # 監視役（独立グループ）: 本スクリプトが突然死しても工程グループを KILL する
    wrapper_pid=$$
    (
        while kill -0 "$wrapper_pid" 2>/dev/null; do sleep 1; done
        kill -s KILL -- "-$child" 2>/dev/null || true
    ) </dev/null >/dev/null 2>&1 &
    wd=$!

    limit_kind=
    while :; do
        sample_pids
        if [ -e "$rcf" ]; then
            alive=0
            group_alive || alive=$?
            if [ "$alive" -eq 1 ]; then
                break
            elif [ "$alive" -eq 2 ]; then
                limit_kind=monitor_error
                break
            fi
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            limit_kind=timeout
            break
        fi
        sleep 0.1
        size=$(wc -c <"$so" 2>/dev/null || echo 0)
        if [ "${size:-0}" -gt "$MAX_OUT_BYTES" ]; then
            limit_kind=output_limit
            break
        fi
        esize=$(wc -c <"$se" 2>/dev/null || echo 0)
        if [ "${esize:-0}" -gt "$MAX_ERR_BYTES" ]; then
            limit_kind=output_limit
            break
        fi
    done
    if [ -n "$limit_kind" ]; then
        kill -s TERM -- "-$child" 2>/dev/null || true
        sleep 1
        kill -s KILL -- "-$child" 2>/dev/null || true
    fi
    wait "$child" 2>/dev/null || true
    reaped=1
    kill -s KILL -- "-$wd" 2>/dev/null || true
    wait "$wd" 2>/dev/null || true
    wd=

    step_rc=70
    if [ -z "$limit_kind" ] && [ -r "$rcf" ]; then
        step_rc=$(cat "$rcf")
        case "$step_rc" in '' | *[!0-9]*) step_rc=70 ;; esac
        # 契約外の値（126・127・シグナル終了の 128+N 等）は runtime_error へ写す（REQ-21）
        case "$step_rc" in 0 | 10 | 11 | 12 | 20 | 64 | 70) ;; *) step_rc=70 ;; esac
    fi
    if [ -z "$limit_kind" ]; then
        # head -c の打ち切りで上限を超えていた場合
        if [ "$(wc -c <"$so")" -gt "$MAX_OUT_BYTES" ] || [ "$(wc -c <"$se")" -gt "$MAX_ERR_BYTES" ]; then
            step_rc=70
        fi
    fi
}

# 工程を実行して記録する。失敗したら failed_step・final_rc を設定して 1 を返す。
# 引数: <工程名> <候補番号|-> <CLI の argv...>
do_step() {
    name=$1
    cand=$2
    shift 2
    step_no=$((step_no + 1))
    nn=$(printf '%02d' "$step_no")
    if [ "$cand" = "-" ]; then
        prefix="${nn}_$name"
        cand_json=null
    else
        prefix="${nn}_${name}_c$cand"
        cand_json=$cand
    fi
    t0=$(utc_now)
    run_step "$prefix" "$@"
    t1=$(utc_now)
    status=null
    # infer の判定行は対象外（11）・保留（12）でも工程としては完了（REQ-21・REQ-22・#478・#497）。
    # 判定行の status が終了コードと一致することを下の構造検証で確かめ、記録には元の終了コードを残す。
    infer_rc=0
    infer_status=ok
    if [ "$name" = infer ] && { [ "$step_rc" -eq 11 ] || [ "$step_rc" -eq 12 ]; }; then
        infer_rc=$step_rc
        if [ "$step_rc" -eq 11 ]; then infer_status=out_of_scope; else infer_status=abstain; fi
        step_rc=0
    fi
    if [ "$step_rc" -eq 0 ]; then
        # 終了コード 0 でも出力を信用しない。stdout が単一の JSON オブジェクトで、
        # トップレベルに code があるなら "ok"（終了コード 0 と整合）であることを検証する。
        # 出力は skipped / ok / invalid のいずれかの固定語（evaluate は status も見る）
        verdict=$(python3 -c '
import json, sys
try:
    v = json.loads(sys.stdin.buffer.read().decode("utf-8"))
except Exception:
    print("invalid")
    sys.exit(0)
name = sys.argv[1]
if not isinstance(v, dict) or not v or ("code" in v and v.get("code") != "ok"):
    print("invalid")
elif name == "infer":
    # infer の判定 JSON は step を持たず、終了コードに対応する status と id・predicted_label を持つ
    if v.get("status") == sys.argv[2] and "step" not in v and isinstance(v.get("id"), str) and isinstance(v.get("predicted_label"), str):
        print("ok")
    else:
        print("invalid")
elif v.get("step") != name:
    print("invalid")
elif name == "evaluate" and v.get("status") == "skipped":
    print("skipped")
elif v.get("status") == "ok":
    if name == "evaluate" and sys.argv[3] == "1" and not (isinstance(v.get("calibration"), dict) and isinstance(v.get("abstention"), dict)):
        # --extended: 校正・保留（REQ-22）が出ていない評価を完了として扱わない
        print("invalid")
    elif name == "select":
        # select の出力の candidate（非負整数）を後段の evaluate の対象にする
        c = v.get("candidate")
        if isinstance(c, int) and not isinstance(c, bool) and c >= 0:
            print("ok " + str(c))
        else:
            print("invalid")
    else:
        print("ok")
else:
    print("invalid")
' "$name" "$infer_status" "$extended" <"$work/$prefix.stdout" 2>/dev/null) || verdict=invalid
        case "$verdict" in
            skipped) status='"skipped"' ;;
            ok) [ "$name" != "evaluate" ] || status='"ok"' ;;
            "ok "*) selected_candidate=${verdict#ok } ;;
            *) step_rc=70 ;;
        esac
    fi
    rec_rc=$step_rc
    if [ "$step_rc" -eq 0 ] && [ "$infer_rc" -ne 0 ]; then rec_rc=$infer_rc; fi
    cn=$(code_name "$rec_rc")
    obytes=$(wc -c <"$work/$prefix.stdout" | tr -d ' ')
    ebytes=$(wc -c <"$work/$prefix.stderr" | tr -d ' ')
    rm -f "$work/$prefix.stdout" "$work/$prefix.stderr"
    entry=$(printf '{"step":"%s","candidate":%s,"exit_code":%s,"code":"%s","status":%s}' \
        "$name" "$cand_json" "$rec_rc" "$cn" "$status")
    steps_json="${steps_json:+$steps_json,}$entry"
    meta_entry=$(printf '{"step":"%s","candidate":%s,"exit_code":%s,"code":"%s","status":%s,"started_utc":"%s","ended_utc":"%s","stdout_bytes":%s,"stderr_bytes":%s}' \
        "$name" "$cand_json" "$rec_rc" "$cn" "$status" "$t0" "$t1" "$obytes" "$ebytes")
    steps_meta="${steps_meta:+$steps_meta,}$meta_entry"
    if [ "$step_rc" -ne 0 ]; then
        failed_step="\"$name\""
        final_rc=$step_rc
        return 1
    fi
    return 0
}

started=$(utc_now)

# smoke 時に実行しない evaluate を、skipped の工程として記録する（工程は起動しない。exit 0）
record_skipped_evaluate() {
    step_no=$((step_no + 1))
    t_skip=$(utc_now)
    entry=$(printf '{"step":"evaluate","candidate":%s,"exit_code":0,"code":"ok","status":"skipped"}' "$1")
    steps_json="${steps_json:+$steps_json,}$entry"
    meta_entry=$(printf '{"step":"evaluate","candidate":%s,"exit_code":0,"code":"ok","status":"skipped","started_utc":"%s","ended_utc":"%s","stdout_bytes":0,"stderr_bytes":0}' "$1" "$t_skip" "$t_skip")
    steps_meta="${steps_meta:+$steps_meta,}$meta_entry"
}

# 7 工程を順に実行する（選定を評価より先に行う。REQ-27）。途中の失敗で停止する（以降の工程は起動しない。fail-closed）
run_all() {
    do_step register - register --definition "$definition" --project-dir "$project_dir" || return 0
    do_step inspect - inspect --project-dir "$project_dir" || return 0
    i=0
    while [ "$i" -lt "$candidates" ]; do
        if [ "$smoke" -eq 1 ]; then
            do_step train "$i" train --project-dir "$project_dir" --candidate "$i" --smoke || return 0
        else
            do_step train "$i" train --project-dir "$project_dir" --candidate "$i" || return 0
        fi
        i=$((i + 1))
    done
    # 最終 test の結果を見て候補を選べないよう、validation による選定（select）を先に確定し、
    # 選定された候補だけを evaluate する（evaluate は選定記録のない候補を拒否する。REQ-27）
    selected_candidate=
    do_step select - select --project-dir "$project_dir" || return 0
    if [ "$smoke" -eq 1 ]; then
        # `--smoke` の短縮学習候補は最終 test を適用できない（evaluate が拒否する。検証専用モデルが
        # 本番候補の構成ロックを使い切らないため）。評価は実行せず、評価未実施を skipped として
        # 記録する（評価済みを装わない。REQ-17・REQ-27）。
        record_skipped_evaluate "$selected_candidate"
    else
        do_step evaluate "$selected_candidate" evaluate --project-dir "$project_dir" --candidate "$selected_candidate" || return 0
    fi
    # `--smoke` で短縮学習した候補は、検証専用の `--allow-smoke` を渡さないと package できない
    # （配布用ではない。REQ-27）。評価データがあるプロジェクトでは、smoke 候補は evaluate できず評価完了
    # 記録が無いため `--allow-smoke` でも package は拒否される（ゲートは緩めない）。smoke ランは評価データの
    # ない fixture（`fixtures/sandbox_run`）で使うこと。評価データありの経路は smoke なしで `fixtures/sandbox_run_eval` を使う。
    if [ "$smoke" -eq 1 ]; then
        do_step package - package --project-dir "$project_dir" --allow-smoke || return 0
    else
        do_step package - package --project-dir "$project_dir" || return 0
    fi
    do_step infer - infer --package "$project_dir/package" --text "$infer_text" || return 0
    [ "$extended" -eq 1 ] || return 0
    run_extended || return 0
}

# --extended の追加工程（REQ-38・#469）。全工程 0 のみ許容し、失敗で停止する
run_extended() {
    do_step register - register --definition "$definition" --project-dir "$ext_project" || return 1
    do_step inspect - inspect --project-dir "$ext_project" || return 1
    do_step train - train --project-dir "$ext_project" --all --smoke --budget-seconds 600 || return 1
    do_step train - train --project-dir "$ext_project" --status || return 1
    do_step train - train --project-dir "$ext_project" --cancel || return 1
    do_step infer - infer --package "$project_dir/package" --text "$infer_text" \
        --version-ledger "$project_dir/version_ledger.json" --version-id v1 || return 1
}
run_all
ended=$(utc_now)

if [ "$final_rc" -eq 0 ]; then
    top_code=ok
    msg="all stages completed under sandbox"
    # 評価が skipped の工程を含むなら、評価未実施を集計の文言にも明示する（REQ-17）
    case "$steps_json" in
        *'"step":"evaluate"'*'"status":"skipped"'*) msg="stages completed under sandbox but evaluation was skipped" ;;
    esac
else
    top_code=$(code_name "$final_rc")
    msg="stopped at a failing stage"
fi
# 実機での判定を Agent が確定させない。上書き（ハーネス）実行だけを test_harness と記す
hint=requires_human_review
launcher_label=$DEFAULT_LAUNCHER
[ "$override" = false ] || launcher_label=override
[ "$override" = false ] || hint=test_harness

pids_json=
for _p in $run_pids; do pids_json="${pids_json:+$pids_json,}$_p"; done
if ! printf '{"started_utc":"%s","ended_utc":"%s","exit_code":%s,"failed_step":%s,"sandbox_exec":"%s","sandbox_profile":"%s","sandbox_exec_override":%s,"evidence_hint":"%s","process_pids":[%s],"steps":[%s]}\n' \
    "$started" "$ended" "$final_rc" "$failed_step" "$launcher_label" "$PROFILE" "$override" "$hint" "$pids_json" "$steps_meta" \
    >"$out_dir/run.meta.json" 2>/dev/null; then
    # 記録の書き込み失敗も契約どおりの JSON（runtime_error・exit 70）で返す
    fail 70 runtime_error "cannot write run record"
fi

printf '{"code":"%s","message":"%s","failed_step":%s,"steps":[%s],"sandbox_profile":"%s","sandbox_exec_override":%s,"run_meta":"run.meta.json"}\n' \
    "$top_code" "$msg" "$failed_step" "$steps_json" "$PROFILE" "$override"
exit "$final_rc"
