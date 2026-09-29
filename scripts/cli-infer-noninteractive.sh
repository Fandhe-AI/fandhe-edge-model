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
#     （引数値・入力テキストは出さない）。ラッパーが 70 へ置き換えた結果では CLI の stderr を
#     中継せず、診断行だけを出す（置き換えは replace_with_error に集約）
#   - 実行時間（既定 300 秒。FANDHE_EDGE_TIMEOUT_SECS）・stdout 容量（1 MiB）・
#     stderr 容量（64 KiB）に上限を置き、超過時は子（子孫プロセスを含む）を終了して
#     runtime_error(70) の JSON を返す（REQ-39）。子は独立したプロセスグループで起動し、
#     直接の子が終了しても、グループの子孫が全員いなくなるまで同じ上限の下で監視する
#     （setsid で自らグループを抜ける子孫は対象外。bash が無い環境は fail-closed で 70）。
#     プロセス一覧（ps）の取得に失敗したら生存不明として fail-closed（グループを KILL して 70）。
#     本スクリプトの突然死（SIGKILL）には独立した監視役が最大 1 秒後に子のグループを KILL する
#     （監視役も同時に殺された場合と、グループが空になった後の pgid 再利用は防げない限界）
#   - 許可された終了コードでも、stdout が JSON の構文（RFC 8259）として正しい
#     オブジェクト 1 つ（1 行）でなければ runtime_error(70) の JSON へ置き換える
#     （複数 JSON・途中切れ・末尾カンマ等を中継しない。REQ-33）。ただし引数に
#     `--input-file` がある呼び出し（バッチ）は 1 行 1 JSON オブジェクトを認め、
#     全行を検証して中継する（REQ-33 の唯一の例外）
#   - JSON の `code` と終了コードの対応（0=ok・10=judged_fail・11=out_of_scope・
#     12=pending・20=limit_exceeded・64=invalid_input・70=runtime_error。
#     fixtures/exitcode/exit_codes.json と同一）を検証し、不一致は runtime_error(70)
#     へ置き換える（REQ-21）。単一 JSON で `code` が無い出力は終了コード 0 のときだけ認める。
#     バッチでは各行の `code`（あれば）が既知の名前であることと、終了コード非 0 のとき
#     最終行の `code` が終了コードと一致することを確認する
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

# プロセスグループの隔離に bash のジョブ制御（set -m）を使う。dash 等は tty が無いと
# set -m が失敗するため、bash で再実行する（Bash ツールの前提は bash。無ければ fail-closed）
# 標準出力・標準エラーの書き込み時点の上限に process substitution を使うため、
# POSIX モードの bash（macOS の /bin/sh 等）も通常モードの bash で再実行する。
# 再実行は 1 回だけ（FANDHE_EDGE_REEXEC。POSIXLY_CORRECT 下での無限ループを避ける）
if [ -z "${BASH_VERSION:-}" ] || [ -o posix ]; then
    if [ -z "${FANDHE_EDGE_REEXEC:-}" ] && command -v bash >/dev/null 2>&1; then
        FANDHE_EDGE_REEXEC=1
        export FANDHE_EDGE_REEXEC
        exec bash "$0" ${1+"$@"}
    fi
    echo "fandhe-edge: bash is required" >&2
    printf '%s\n' '{"code":"runtime_error","message":"bash is required for process group isolation"}'
    echo "exit_code=70" >&2
    exit 70
fi

# バッチ（infer --input-file）は 1 行 1 JSON を認める（REQ-33）。CLI（args.rs の INFER_OPTS）と
# 同じくオプションと値を対応づけて走査し、他オプションの値として現れた `--input-file` は
# バッチ指定とみなさない（infer のオプションはすべて値を取る）
batch=0
skip=0
out_requested=0
for a in ${1+"$@"}; do
    if [ "$skip" -eq 1 ]; then
        skip=0
        continue
    fi
    case "$a" in
        --input-file) batch=1; skip=1 ;;
        --input-file=*) batch=1 ;;
        --out) out_requested=1; skip=1 ;;
        --out=*) out_requested=1 ;;
        --package | --text | --id) skip=1 ;;
    esac
done

# シェル自身の診断・ジョブ終了通知（macOS の bash が出す "Terminated: 15" 等）が
# stderr を汚さないよう、契約の出力は fd 3（元の stderr）へだけ出し、fd 2 は後で捨てる
exec 3>&2

if [ -n "${FANDHE_EDGE_BIN:-}" ]; then
    bin=$FANDHE_EDGE_BIN
else
    root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
    bin="${CARGO_TARGET_DIR:-$root/target}/debug/fandhe-edge"
fi

# 出力は stdout 経由だけに閉じ込める（REQ-39）。infer の `--out`（ファイル出力。`--input-file`
# 専用）を渡すと CLI が出力ファイルへ直接書き、本スクリプトの容量上限を迂回するため、
# CLI を起動せず invalid_input(64) で拒否する（fail-closed。メッセージは固定でパスを含めない。
# TASK-36.1-1 は終了コードと stdout の取得のみを要求し、ファイル出力は対象外）
if [ "$out_requested" -eq 1 ]; then
    echo "fandhe-edge: --out is not supported by this wrapper" >&3
    printf '%s\n' '{"code":"invalid_input","message":"--out is not supported by the non-interactive wrapper"}'
    echo "exit_code=64" >&3
    exit 64
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
child=
wd=
reaped=0
# 終了時の後始末。未回収の子のグループ（と後述の監視役）が残っていれば KILL する
# （wait 後は pgid が再利用されうるため送らない）。TERM/INT/HUP でも EXIT へ流す
# （呼び出し元が本スクリプトを止めてもグループを残さない。REQ-39）
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

# 期限と出力上限（REQ-39 資源の上限。無期限待ち・無制限のディスク書き込みを作らない）。
# 期限は FANDHE_EDGE_TIMEOUT_SECS（先頭 0 なしの 1〜999 の整数秒。不正値と先頭 0
# （算術式で 8 進数解釈される事故を避ける）は既定 300 へ戻す）、stdout 上限は 1 MiB、
# stderr 上限は 64 KiB。超過時は子プロセス木を終了し runtime_error(70) の JSON を返す
# （REQ-21・REQ-33）
timeout_secs=${FANDHE_EDGE_TIMEOUT_SECS:-300}
case "$timeout_secs" in
    '' | *[!0-9]* | 0* | ????*) timeout_secs=300 ;;
esac
max_out_bytes=1048576
max_err_bytes=65536
# 期限は壁時計（bash の SECONDS。整数秒）で判定する。0.1 秒ごとの反復回数で数えると
# ps・wc の実行時間ぶん期限が伸びるため。SECONDS の切り捨てで最大 1 秒遅れる
deadline=$((SECONDS + timeout_secs + 1))

# 子は set -m で独立したプロセスグループ（pgid = 子の PID）として起動し、期限・容量の
# 超過時はグループごと終了する（REQ-39）。直接の子が終了した後も、グループに生存プロセスが
# 残る間は同じ期限・容量の監視を続け、残存する子孫を回収する（孤児化で上限が外れない）。
# 終了値は rc ファイルへ書き、監視側は kill -0 でなくファイルで終了を判定する
# （終了済みの子へ遅延 KILL を送る競合と PID 再利用を避ける。未回収の子がグループの
# リーダーとして残るため pgid は再利用されない）。
# 子へ元の stderr（fd 3）を継がせない（呼び出し元のパイプを保持させない）。
# ${1+"$@"}: 引数なしでも Bash 3.2 の set -u で abort しない
exec 2>/dev/null
set -m
(
    rc_child=0
    # stdout・stderr は head -c で書き込み時点に上限 +1 バイトで打ち切る（REQ-39）。
    # 超過すると子は SIGPIPE/EPIPE で止まり、+1 バイト目の存在で上限超過と判定する。
    # head は同じプロセスグループに属し、監視・後始末の対象になる
    "$bin" infer ${1+"$@"} </dev/null \
        > >(head -c $((max_out_bytes + 1)) >"$out") \
        2> >(head -c $((max_err_bytes + 1)) >"$err") || rc_child=$?
    echo "$rc_child" >"$rcf.tmp"
    mv "$rcf.tmp" "$rcf"
) </dev/null >/dev/null 3>&- &
child=$!

# 監視役（独立したプロセスグループ）: 本スクリプトが SIGKILL 等で突然死しても、
# 生存を確認できなくなった時点で子のグループを KILL する（最大 1 秒の遅れ）。
# 正常終了・捕捉できるシグナルでは cleanup が監視役のグループごと止める
wrapper_pid=$$
(
    while kill -0 "$wrapper_pid" 2>/dev/null; do sleep 1; done
    kill -s KILL -- "-$child" 2>/dev/null || true
) </dev/null >/dev/null 2>&1 3>&- &
wd=$!

# グループに生存プロセス（ゾンビ以外）が残っているか。
# 戻り値: 0=残っている・1=いない・2=プロセス一覧の取得失敗（生存不明。fail-closed に扱う）
group_alive() {
    _ps=$(ps -A -o pgid= -o stat= 2>/dev/null) || return 2
    [ -n "$_ps" ] || return 2
    printf '%s\n' "$_ps" | awk -v g="$child" '
        $1 == g && $2 !~ /^Z/ { found = 1 }
        END { exit found ? 0 : 1 }'
}

# 監視（前景ループ）: グループの全員の終了・期限超過・容量超過のいずれかで抜ける
limit_kind=
while :; do
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

# 上限超過時はグループ全体へ TERM → 1 秒後に KILL を送る
if [ -n "$limit_kind" ]; then
    kill -s TERM -- "-$child" 2>/dev/null || true
    sleep 1
    kill -s KILL -- "-$child" 2>/dev/null || true
fi
wait "$child" || true
reaped=1

rc=70
if [ -z "$limit_kind" ] && [ -r "$rcf" ]; then
    rc=$(cat "$rcf")
    case "$rc" in
        '' | *[!0-9]*) rc=137 ;;
    esac
fi
if [ -z "$limit_kind" ]; then
    # 書き込み時点の打ち切り（head -c）で上限を超えた場合
    if [ "$(wc -c <"$out")" -gt "$max_out_bytes" ]; then
        limit_kind=output_limit
    elif [ "$(wc -c <"$err")" -gt "$max_err_bytes" ]; then
        limit_kind=stderr_limit
    fi
fi

# ラッパーが CLI の結果を 70 へ置き換える経路（上限超過・期限・監視失敗・出力なし・不正な
# JSON・code の不一致・異常終了）は、すべてこの関数を通す。CLI の出力（stdout・stderr の
# どちらも途中まで・信頼できない内容）は捨て、固定文の runtime_error(70) の JSON だけを返す。
# 置き換えない経路（CLI の契約どおりの出力）だけが CLI の stderr を中継する（REQ-21・REQ-33・REQ-39）
replace_with_error() {
    printf '%s\n' "{\"code\":\"runtime_error\",\"message\":\"$1\"}" >"$out"
    : >"$err"
    rc=70
}

case "$limit_kind" in
    timeout) replace_with_error "fandhe-edge timed out" ;;
    output_limit) replace_with_error "fandhe-edge output exceeded size limit" ;;
    monitor_error) replace_with_error "fandhe-edge process monitoring failed" ;;
    stderr_limit) replace_with_error "fandhe-edge stderr exceeded size limit" ;;
esac

# stdout の検証（awk の再帰下降パーサー。RFC 8259 の構文を検査し、既存ツールのみで完結させる）。
# 単一モードは 1 行 1 オブジェクトのみ、バッチは各行がオブジェクト。終了値: 0=正常・
# 1=JSON 不正・2=`code` と終了コードの不一致（REQ-21・REQ-33）。
# 対応表は fixtures/exitcode/exit_codes.json と同一（bash_noninteractive.rs が照合する）。
# top-level のキーと `code` の値は、エスケープ（\uXXXX・サロゲートペア・単純エスケープ）を
# 復号してから `code` との一致・既知の名前かを判定する（生の文字列比較で正当な JSON を
# 誤判定しない。ASCII 印字可能文字以外は "?" に写すため、名前と一致しない。REQ-21）
check_output() {
    # RFC 8259 は JSON テキストを UTF-8 と定める。不正なバイト列（awk は LC_ALL=C で
    # バイトのまま扱うため検出できない）は iconv で検査し、iconv が無い環境も fail-closed で
    # 不正とみなす（REQ-33）
    if ! iconv -f UTF-8 -t UTF-8 <"$out" >/dev/null 2>&1; then
        return 1
    fi
    # 1 行 1 JSON の契約のため、終端が改行でない出力（末尾の改行欠落）は不正とする（REQ-33）。
    # $(...) は末尾の改行を落とすため、最終バイトが改行のときだけ空になる
    if [ -n "$(tail -c 1 "$out")" ]; then
        return 1
    fi
    LC_ALL=C awk -v batch="$batch" -v rc="$rc" '
    function isdig(c) { return c != "" && index("0123456789", c) > 0 }
    function skipws(   c) {
        while (pos <= n) {
            c = substr(s, pos, 1)
            if (c == " " || c == "\t" || c == "\r" || c == "\n") pos++
            else break
        }
    }
    function pstring(   c, k, h, start, v) {
        pos++; start = pos; hasesc = 0
        while (pos <= n) {
            c = substr(s, pos, 1)
            if (c == "\"") { strval = substr(s, start, pos - start); pos++; return 1 }
            if (c < " ") return 0
            if (c == "\\") {
                hasesc = 1
                pos++; c = substr(s, pos, 1)
                if (c != "" && index("\"\\/bfnrt", c) > 0) { pos++; continue }
                if (c == "u") {
                    for (k = 1; k <= 4; k++) {
                        h = substr(s, pos + k, 1)
                        if (h == "" || index("0123456789abcdefABCDEF", h) == 0) return 0
                    }
                    # 単独のサロゲート（high の後に low が続かない・low 単独）は UTF-8 へ復号できず、
                    # 厳格な JSON パーサーも拒否するため不正とする（fail-closed。REQ-33）
                    v = hexval(substr(s, pos + 1, 4))
                    if (v >= 56320 && v <= 57343) return 0
                    if (v >= 55296 && v <= 56319) {
                        if (substr(s, pos + 5, 2) != "\\u") return 0
                        for (k = 7; k <= 10; k++) {
                            h = substr(s, pos + k, 1)
                            if (h == "" || index("0123456789abcdefABCDEF", h) == 0) return 0
                        }
                        v = hexval(substr(s, pos + 7, 4))
                        if (v < 56320 || v > 57343) return 0
                        pos += 6
                    }
                    pos += 5; continue
                }
                return 0
            }
            pos++
        }
        return 0
    }
    function pnumber(   c) {
        if (substr(s, pos, 1) == "-") pos++
        c = substr(s, pos, 1)
        if (c == "0") pos++
        else if (isdig(c)) { while (isdig(substr(s, pos, 1))) pos++ }
        else return 0
        if (substr(s, pos, 1) == ".") {
            pos++
            if (!isdig(substr(s, pos, 1))) return 0
            while (isdig(substr(s, pos, 1))) pos++
        }
        c = substr(s, pos, 1)
        if (c == "e" || c == "E") {
            pos++; c = substr(s, pos, 1)
            if (c == "+" || c == "-") pos++
            if (!isdig(substr(s, pos, 1))) return 0
            while (isdig(substr(s, pos, 1))) pos++
        }
        return 1
    }
    function pliteral(w) {
        if (substr(s, pos, length(w)) == w) { pos += length(w); return 1 }
        return 0
    }
    function pvalue(depth,   c) {
        if (depth > 64) return 0
        skipws()
        c = substr(s, pos, 1)
        if (c == "\"") return pstring()
        if (c == "{") return pobject(depth)
        if (c == "[") return parray(depth)
        if (c == "t") return pliteral("true")
        if (c == "f") return pliteral("false")
        if (c == "n") return pliteral("null")
        return pnumber()
    }
    function hexval(h,   i, n, d) {
        n = 0
        for (i = 1; i <= length(h); i++) {
            d = index("0123456789abcdef", tolower(substr(h, i, 1)))
            n = n * 16 + d - 1
        }
        return n
    }
    # 文字列の生表記を復号する。比較対象（ASCII の名前）に必要な範囲だけを正確に扱い、
    # ASCII 印字可能文字以外（制御文字・非 ASCII・サロゲートペア・単純エスケープ）は "?" に写す
    function decode(raw,   i, n, c, d, out, v) {
        out = ""; i = 1; n = length(raw)
        while (i <= n) {
            c = substr(raw, i, 1)
            if (c != "\\") { out = out c; i++; continue }
            d = substr(raw, i + 1, 1)
            if (d == "u") {
                v = hexval(substr(raw, i + 2, 4))
                out = out ((v >= 32 && v <= 126) ? substr(asc, v - 31, 1) : "?")
                if (v >= 55296 && v <= 56319 && substr(raw, i + 6, 2) == "\\u") i += 12
                else i += 6
            } else { out = out "?"; i += 2 }
        }
        return out
    }
    function pobject(depth,   key, first, c) {
        pos++; skipws()
        if (substr(s, pos, 1) == "}") { pos++; return 1 }
        while (1) {
            skipws()
            if (substr(s, pos, 1) != "\"") return 0
            if (!pstring()) return 0
            key = hasesc ? decode(strval) : strval
            skipws()
            if (substr(s, pos, 1) != ":") return 0
            pos++; skipws()
            first = substr(s, pos, 1)
            if (!pvalue(depth + 1)) return 0
            if (depth == 0 && key == "code") {
                if (hascode || first != "\"") codebad = 1
                else code = hasesc ? decode(strval) : strval
                hascode = 1
            }
            skipws()
            c = substr(s, pos, 1)
            if (c == ",") { pos++; continue }
            if (c == "}") { pos++; return 1 }
            return 0
        }
    }
    function parray(depth,   c) {
        pos++; skipws()
        if (substr(s, pos, 1) == "]") { pos++; return 1 }
        while (1) {
            if (!pvalue(depth + 1)) return 0
            skipws()
            c = substr(s, pos, 1)
            if (c == ",") { pos++; continue }
            if (c == "]") { pos++; return 1 }
            return 0
        }
    }
    BEGIN {
        for (i = 32; i <= 126; i++) asc = asc sprintf("%c", i)
        split("ok=0 judged_fail=10 out_of_scope=11 pending=12 limit_exceeded=20 invalid_input=64 runtime_error=70", pairs, " ")
        for (i in pairs) { split(pairs[i], kv, "="); known[kv[1]] = 1; if (kv[2] == rc) expect = kv[1] }
    }
    {
        lines++
        if (!batch && lines > 1) syntax = 1
        s = $0; n = length(s); pos = 1; code = ""; hascode = 0; codebad = 0
        skipws()
        if (substr(s, pos, 1) != "{" || !pvalue(0)) { syntax = 1; next }
        skipws()
        if (pos <= n) { syntax = 1; next }
        if (batch) {
            if (hascode && (codebad || !(code in known))) mismatch = 1
            lastok = hascode && !codebad && code == expect
        } else if (hascode) {
            if (codebad || code != expect) mismatch = 1
        } else if (rc != 0) {
            mismatch = 1
        }
    }
    END {
        if (lines == 0 || syntax) exit 1
        if (batch && rc != 0 && !lastok) mismatch = 1
        exit mismatch ? 2 : 0
    }' "$out"
}

# CLI の終了コードは 7 種（0/10/11/12/20/64/70）に固定のため、それ以外
# （126・127・シグナル終了の 128+N 等の契約外）は既存の stdout の有無にかかわらず
# runtime_error(70) の JSON へ置き換える（JSON の code と終了コードを一致させる。REQ-21）。
# 許可された終了コードでも stdout が空なら補い、JSON 構文の不正・`code` の不一致は
# 置き換えて 70 を返す（1 呼び出し 1 JSON。REQ-21・REQ-33）
case "$rc" in
    0 | 10 | 11 | 12 | 20 | 64 | 70)
        if [ ! -s "$out" ]; then
            replace_with_error "fandhe-edge produced no output"
        else
            check_rc=0
            check_output || check_rc=$?
            if [ "$check_rc" -eq 1 ]; then
                replace_with_error "fandhe-edge produced invalid output"
            elif [ "$check_rc" -ne 0 ]; then
                replace_with_error "fandhe-edge output code does not match exit code"
            fi
        fi
        ;;
    *)
        replace_with_error "fandhe-edge terminated abnormally"
        ;;
esac
cat "$out"
# CLI の stderr（上限 64 KiB 以内）を中継してから診断行を出す
cat "$err" >&3
echo "exit_code=$rc" >&3
exit "$rc"
