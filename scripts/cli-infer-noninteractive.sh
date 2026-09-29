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
# 実行記録（REQ-36・TASK-36.1-2・#150。opt-in）:
#   環境変数 FANDHE_EDGE_RECORD_DIR（存在する実ディレクトリ。symlink は不可）を設定したときだけ、
#   その直下へ一時名 `.run-record.<pid>.<乱数>.tmp`（noclobber の O_EXCL 作成・0600）で作り、書き込みと検証が済んだときだけ
#   `run-record.<pid>.<乱数>` へ確定する（ハードリンク `ln` で上書きせず原子的に。衝突したら別の乱数で最大 10 回再試行し、
#   それでも確定できなければ 70）（失敗時は一時名・最終名とも残さない。作成・rename・削除は
#   記録先を cd -P で開いたカレントディレクトリへの相対名で行い、記録先の移動・差し替えに影響されない）。
#   JSON オブジェクト 1 つを 1 行（末尾 LF・UTF-8）で保存する。未設定なら出力も副作用も一切変えない。
#   キーはこの順: schema（`fandhe-edge.run-record/1`）・command（`["fandhe-edge","infer",<引数…>]`。
#   実行ファイルのパスは入れない）・started_at（CLI 起動前の UTC 秒精度）・exit_code（呼び出し元へ
#   返す値）・stdout・stderr（それぞれ `{"bytes":<バイト数>,"sha256":"<hex>"}`。
#   stdout は呼び出し元が受け取る正規化後の全体、stderr は中継する CLI の stderr で、診断行
#   exit_code=<N> は含めない）。
#   データ本文の混入防止（security.md）: 利用者由来の値（stdout に含まれる id・入力由来の文字列、
#   CLI の stderr）は本文を保存せず、件数（バイト数）とハッシュだけを残す。command は許可リスト方式で、
#   記録するのは既知オプション名（--package・--input-file・--out・--text・--id・--help）とパス値
#   （UTF-8 として不正なら `<invalid utf-8>`）だけ、--text・--id の値・未知のトークン・位置引数は
#   `<redacted>` に伏せる。入力ファイルの中身は読まず、環境変数は記録しない。
#   command の記録には資源上限がある（引数 64 件・command 部 4096 バイト・1 値 512 文字）。1 値が長いときはその値を
#   `<truncated>` に置き換え、件数か総量が超えたら以降を記録せず末尾に `<truncated>` を 1 つ付ける（推論の結果は変えない）。
#   記録先は起動前に物理パス（`cd -P` + `pwd -P`。途中の symlink を解決）へ正規化し、作成と書き込みは 1 回の open でパスを再解決せず、保存後に
#   記録先の物理パスが不変であること・記録が通常ファイルであることを確認する
#   （不一致は fail-closed。検査から作成までの完全な排他はシェルでは保証できない限界で、
#   差し替えに気づいたら記録を削除して 70 にする）。ハッシュは sha256sum / shasum -a 256 のいずれか
#   （無ければ fail-closed）。記録を要求されたのに保存できなければ runtime_error(70)。
#   CLI 起動前の拒否（--out・不正な記録ディレクトリ・バイナリ不在）は記録しない。
#   証拠種別: テストハーネス（fake bin・help 経路。実クライアントでの記録ではない）。
#
# 現状の制約: infer の実推論は TASK-33.1-2（#136）・前処理（#112）・
# ONNX 推論（#113）が未接続のため exit 70 を返す。exit 0 になるのは
# `--help` のみ。
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

# 実行記録（TASK-36.1-2）用の JSON 文字列エスケープ。stdin をバイト単位（LC_ALL=C）で読み、
# `"`・`\`・制御文字（0x01〜0x1F）をエスケープして引用符なしで出す。引数 $1 は入力が改行で
# 終わるか（1/0）。0x7F 以上は素通し（UTF-8 の正当性は呼び出し側が保証する）。
# 置換は index/substr で組み立て、gsub の置換文字列の `\` 解釈差（awk 実装差）を避ける
json_escape() {
    LC_ALL=C awk -v trail="$1" '
    function rep(l, c, r,   o, p) {
        o = ""
        while ((p = index(l, c)) > 0) { o = o substr(l, 1, p - 1) r; l = substr(l, p + length(c)) }
        return o l
    }
    BEGIN {
        for (i = 1; i <= 31; i++) if (i != 10) { ch[i] = sprintf("%c", i); es[i] = sprintf("\\u%04x", i) }
    }
    {
        l = rep($0, "\\", "\\\\")
        l = rep(l, "\"", "\\\"")
        for (i = 1; i <= 31; i++) if (i != 10 && index(l, ch[i]) > 0) l = rep(l, ch[i], es[i])
        if (NR > 1) printf "\\n"
        printf "%s", l
    }
    END { if (NR > 0 && trail == 1) printf "\\n" }'
}

# ファイルの sha256（16 進 64 桁）を出す。ツールが無い・出力が不正なら非 0（fail-closed）
sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        _h=$(sha256sum <"$1") || return 1
    elif command -v shasum >/dev/null 2>&1; then
        _h=$(shasum -a 256 <"$1") || return 1
    else
        return 1
    fi
    _h=${_h%% *}
    case "$_h" in
        *[!0-9a-f]* | '') return 1 ;;
    esac
    [ "${#_h}" -eq 64 ] || return 1
    printf '%s' "$_h"
}

# 引数 1 つを記録の command 配列へ追加する（記録を要求されたときだけ）。呼び出し側は
# 固定のオプション名か伏せ字だけを渡す（許可リスト方式。入力本文を記録に残さない。security.md）
rec_dir=${FANDHE_EDGE_RECORD_DIR:-}
rec_cmd='"fandhe-edge","infer"'
# command 記録の資源上限（REQ-39）: 引数 64 件・記録の command 部 4096 バイト・1 値 512 文字。
# 1 値が長すぎるときはその値を `<truncated>` に置き換え、件数か総量が上限を超えたら
# 以降の引数を記録せず末尾に固定の印 `<truncated>` を 1 度だけ付ける（記録の切り詰め。
# 推論本体の終了コード・出力には影響しない）
rec_n=0
rec_bytes=0
rec_trunc=0
rec_add() {
    if [ -n "$rec_dir" ] && [ "$rec_trunc" -eq 0 ]; then
        _piece=
        _plen=0
        if [ "$rec_n" -lt 64 ]; then
            if [ "${#1}" -gt 512 ]; then
                _piece=',"<truncated>"'
            else
                # awk は行区切りの末尾改行を取り除くため、値が改行で終わるかを判定して json_escape へ
                # 渡す（末尾改行を含むパスが記録で別の値にならないようにする）
                _trail=0
                case "$1" in
                    *$'\n') _trail=1 ;;
                esac
                _piece=",\"$(printf '%s' "$1" | json_escape "$_trail")\""
            fi
            _plen=$(printf '%s' "$_piece" | wc -c | tr -d ' ')
        fi
        if [ -z "$_piece" ] || [ $((rec_bytes + _plen)) -gt 4096 ]; then
            rec_trunc=1
            rec_cmd="$rec_cmd"',"<truncated>"'
        else
            rec_n=$((rec_n + 1))
            rec_bytes=$((rec_bytes + _plen))
            rec_cmd="$rec_cmd$_piece"
        fi
    fi
}

# パス値（--package・--input-file・--out の値）を記録へ追加する。UTF-8 として不正なら
# 不正な JSON を作らないよう固定値へ置き換える（stderr・stdout の検査と同じ方針）
rec_add_path() {
    if [ -n "$rec_dir" ]; then
        # $2 は `--package=` のような等号形式の接頭辞（空なら値のみ）。UTF-8 の検査は値だけに行う
        if printf '%s' "$1" | iconv -f UTF-8 -t UTF-8 >/dev/null 2>&1; then
            rec_add "${2:-}$1"
        else
            rec_add "${2:-}<invalid utf-8>"
        fi
    fi
}

# バッチ（infer --input-file）は 1 行 1 JSON を認める（REQ-33）。CLI（args.rs の INFER_OPTS）と
# 同じくオプションと値を対応づけて走査し、他オプションの値として現れた `--input-file` は
# バッチ指定とみなさない（infer のオプションはすべて値を取る）
batch=0
skip=0
out_requested=0
redact_next=0
for a in ${1+"$@"}; do
    if [ "$skip" -eq 1 ]; then
        skip=0
        # 記録は許可リスト方式（TASK-36.1-2）。パス値（--package・--input-file・--out）以外の
        # 値（--text・--id の入力本文・利用者の id、未知オプションの値）は伏せる
        case "$redact_next" in
            1) redact_next=0; rec_add_path "$a" ;;
            *) rec_add "<redacted>" ;;
        esac
        continue
    fi
    case "$a" in
        --input-file) batch=1; skip=1; redact_next=1; rec_add "$a" ;;
        --input-file=*) batch=1; rec_add_path "${a#--input-file=}" "--input-file=" ;;
        --out) out_requested=1; skip=1; redact_next=1; rec_add "$a" ;;
        --out=*) out_requested=1; rec_add_path "${a#--out=}" "--out=" ;;
        --package) skip=1; redact_next=1; rec_add "$a" ;;
        --package=*) rec_add_path "${a#--package=}" "--package=" ;;
        --text | --id) skip=1; redact_next=0; rec_add "$a" ;;
        --text=*) rec_add "--text=<redacted>" ;;
        --id=*) rec_add "--id=<redacted>" ;;
        --help | -h) rec_add "$a" ;;
        *) rec_add "<redacted>" ;;
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

# 記録ディレクトリは CLI の起動前に検証する（fail-closed。メッセージは固定でパスを含めない）
# ディレクトリの識別子（デバイス番号:inode）。GNU stat と BSD stat の両方に対応する
dir_identity() {
    stat -c '%d:%i' -- "$1" 2>/dev/null || stat -f '%d:%i' -- "$1" 2>/dev/null
}
# 末尾の / と /. を取り除いた最終要素で symlink を判定する（"link/"・"link/." だと
# -L が偽になり symlink を素通りするため）
rec_trim=$rec_dir
while :; do
    case "$rec_trim" in
        ?*/) rec_trim=${rec_trim%/} ;;
        ?*/.) rec_trim=${rec_trim%/.} ;;
        *) break ;;
    esac
done
rec_real=
rec_id=
if [ -n "$rec_dir" ] && { [ ! -d "$rec_dir" ] || [ -L "$rec_trim" ]; }; then
    rec_real=-
elif [ -n "$rec_dir" ]; then
    # 途中の symlink を含む全経路を物理パスへ正規化し、以降の保存はこの値だけを使う
    rec_real=$(CDPATH='' cd -P -- "$rec_dir" 2>/dev/null && pwd -P) || rec_real=-
    [ -n "$rec_real" ] || rec_real=-
    if [ "$rec_real" != "-" ]; then
        # 起動前のディレクトリ識別子を控える（保存時に同一ディレクトリか照合する）
        rec_id=$(dir_identity "$rec_real") || rec_id=
        [ -n "$rec_id" ] || rec_real=-
    fi
fi
if [ "$rec_real" = "-" ]; then
    printf '%s\n' '{"code":"invalid_input","message":"FANDHE_EDGE_RECORD_DIR must be an existing directory"}'
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
# 記録の started_at は CLI 起動前の UTC 秒精度（取得失敗は記録時に fail-closed）
started_at=
if [ -n "$rec_dir" ]; then
    started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ) || started_at=
fi
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
# 実行記録の保存（opt-in。TASK-36.1-2）。正規化と置き換えが済んだ最終結果を記録する。
# 保存できなければ記録済みを装わず runtime_error(70) にする（再試行しない）
if [ -n "$rec_dir" ]; then
    rec_ok=1
    case "$started_at" in
        [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z) ;;
        *) rec_ok=0 ;;
    esac
    # stdout・stderr は本文を残さずバイト数と sha256 だけを記録する（利用者由来の値の混入防止）
    out_hash=
    err_hash=
    out_bytes=
    err_bytes=
    if [ "$rec_ok" -eq 1 ]; then
        out_hash=$(sha256_file "$out") || rec_ok=0
        err_hash=$(sha256_file "$err") || rec_ok=0
        out_bytes=$(wc -c <"$out" | tr -d ' ') || rec_ok=0
        err_bytes=$(wc -c <"$err" | tr -d ' ') || rec_ok=0
    fi
    # 記録先が起動前の物理パスかつ同一ディレクトリ（デバイス:inode）のままか
    # （検査後の symlink 差し替え・同名の別ディレクトリへの入れ替えの検出）
    rec_dir_unchanged() {
        [ "$(CDPATH='' cd -P -- "$rec_real" 2>/dev/null && pwd -P)" = "$rec_real" ] &&
            [ "$(dir_identity "$rec_real")" = "$rec_id" ]
    }
    if [ "$rec_ok" -eq 1 ]; then
        rec_dir_unchanged || rec_ok=0
    fi
    if [ "$rec_ok" -eq 1 ]; then
        # 記録は一時名で作り、書き込みと検証が済んだときだけ rename で確定する（原子的。
        # 途中の失敗で最終名の不完全な記録を残さない）。作成・rename・削除はすべて子シェルの
        # カレントディレクトリ（起動前に控えた inode を cd -P で開いたもの）に対する相対名で行い、
        # 記録先のパスが移動・差し替えられても、作ったファイルを元のディレクトリから確実に消せる
        # （パスの再解決に依存しない）。名前の衝突は有限回だけ乱数を変えて再試行する
        rec_try=0
        rec_rc=3
        while [ "$rec_try" -lt 10 ]; do
            rec_try=$((rec_try + 1))
            rec_name="run-record.$$.$RANDOM$RANDOM"
            rec_tmp=".$rec_name.tmp"
            # 子シェルの終了値: 0=保存成功・2 または 4=名前の衝突（別の乱数名で再試行する。作成したファイルは
            # 子シェルの中で削除済み）・それ以外=失敗（同じく削除済み）
            rec_rc=0
            # 子シェルの stdout・stderr は捨てる。書き込みが上限超過などで途中失敗すると、printf の stdio
            # バッファに残った記録の断片が、リダイレクトを戻した後に子シェルの stdout（=呼び出し元の
            # stdout）へ吐かれる libc がある（macOS の CI で観測）。契約の出力は最後の cat "$out" だけにする
            (
                set -C
                umask 077
                # 書き込みが上限超過（RLIMIT_FSIZE）でも SIGXFSZ で落とさず、通常の書き込み失敗として扱う
                trap '' XFSZ
                CDPATH='' cd -P -- "$rec_real" 2>/dev/null || exit 3
                [ "$(dir_identity .)" = "$rec_id" ] || exit 3
                exec 4>"$rec_tmp" || exit 2
                if ! printf '{"schema":"fandhe-edge.run-record/1","command":[%s],"started_at":"%s","exit_code":%s,"stdout":{"bytes":%s,"sha256":"%s"},"stderr":{"bytes":%s,"sha256":"%s"}}\n' \
                    "$rec_cmd" "$started_at" "$rc" "$out_bytes" "$out_hash" "$err_bytes" "$err_hash" >&4; then
                    rm -f -- "$rec_tmp"
                    exit 3
                fi
                exec 4>&-
                if [ -L "$rec_tmp" ] || [ ! -f "$rec_tmp" ] || ! rec_dir_unchanged; then
                    rm -f -- "$rec_tmp"
                    exit 3
                fi
                # 確定先の名前が既にあれば（ディレクトリ・symlink を含め）衝突として別の名前で再試行する
                if [ -e "$rec_name" ] || [ -L "$rec_name" ]; then
                    rm -f -- "$rec_tmp"
                    exit 4
                fi
                # 上書きしない原子的な確定: ハードリンクは確定先が既にあれば失敗する（-n は GNU・BSD の
                # どちらの ln にもある no-dereference で、確定先が symlink でも辿らない）。ただし確定先が
                # 検査後にディレクトリへ差し替わると ln はその中へリンクを作るため、ln の後に
                # 「確定先そのものが一時ファイルと同じ実体の通常ファイル」であることを検証し、
                # 満たさなければディレクトリの中にできたリンクを消して衝突として扱う
                ln -n -- "$rec_tmp" "$rec_name" 2>/dev/null || :
                if [ ! -L "$rec_name" ] && [ -f "$rec_name" ] && [ "$rec_name" -ef "$rec_tmp" ]; then
                    rm -f -- "$rec_tmp"
                else
                    if [ -d "$rec_name" ] && [ -f "$rec_name/$rec_tmp" ] && [ "$rec_name/$rec_tmp" -ef "$rec_tmp" ]; then
                        rm -f -- "$rec_name/$rec_tmp"
                    fi
                    rm -f -- "$rec_tmp"
                    exit 4
                fi
                # 確定後も記録先が起動前のままで、記録が通常ファイルであること
                if [ -L "$rec_name" ] || [ ! -f "$rec_name" ] || ! rec_dir_unchanged; then
                    rm -f -- "$rec_name"
                    exit 3
                fi
            ) >/dev/null 2>&1 || rec_rc=$?
            case "$rec_rc" in 2 | 4) ;; *) break ;; esac
        done
        # 記録先が指定されたのに確定できなかった場合は、理由を問わず（一時ファイルの作成失敗・書き込み失敗・
        # 名前の衝突の上限到達・ln の失敗など）ここ 1 か所で 70 にする
        [ "$rec_rc" -eq 0 ] || rec_ok=0
    fi
    if [ "$rec_ok" -ne 1 ]; then
        replace_with_error "failed to save run record"
    fi
fi
cat "$out"
# CLI の stderr（上限 64 KiB 以内）を中継してから診断行を出す
cat "$err" >&3
echo "exit_code=$rc" >&3
exit "$rc"
