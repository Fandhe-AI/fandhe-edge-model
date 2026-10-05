#!/bin/sh
# Mac 実機での動作確認（項目 A〜F）を、PC を変えても同じ手順で再現するための入口スクリプト
# （REQ-21・REQ-28・REQ-30・REQ-31・REQ-32・REQ-33・REQ-39。特定の TASK には対応しない横断の確認ツール。
# REQ-38 の sandbox 下の通信 0 件は対象外で、`sandbox-monitor.sh` の担当）。
#
# 呼び出し元: 人が macOS 実機で直接実行する（`make real-machine-check ARGS='...'`。`make ci` には含めない）／
# crates/cli/tests/real_machine_check_script.rs（偽の make・cargo を使うテストハーネス）。
# このスクリプトは引数の検証・作業ディレクトリの用意・CLI の特定だけを行い、項目の実行・JSON の要約・
# 伏せ処理・record.json と record.md の生成は scripts/real_machine_check_record.py（標準ライブラリのみ。
# jq には依存しない）が行う。実行と記録の整形までが責務で、「実機」の証拠としての確定は人が行う。
#
# 使い方:
#   real-machine-check.sh --work-dir DIR [--items LIST] [--repeat N] [--quiet-machine] [--with-ci]
#                         [--p95-limit-us N] [--package-limit-bytes N]
#   値を取るオプションは `--key VALUE` と `--key=VALUE` の両方を受け付ける。
#   --work-dir: 必須。存在しないか空のディレクトリで、リポジトリ配下でないこと（物理パスで比較）
#   --items:    A,B,C,D,E,F の部分集合（カンマ区切り・大文字・重複不可）。既定は B,C,D,E,F。実行順は常に A→F。
#               E は B の成果物を使うため B と一緒に指定する（B なしの E は invalid_input(64)）
#   --with-ci:  A（make ci）を実行する明示の同意。A は通信を伴いうる（uv sync・advisory DB・npx）。
#               --items に A があり --with-ci が無ければ invalid_input(64) で何も実行しない
#   --repeat:   F の回数（1〜1000。既定 50）
#   --quiet-machine: 他のアプリを閉じた静かな状態という人の申告。あるときだけ p95 の区分が real_machine
#   --p95-limit-us / --package-limit-bytes: C-1・C-2 の limits（既定 50000 / 1000）。
#               --p95-limit-us は 1〜3600000000（定義ファイル側の上限と同じ）
#
# 環境変数:
#   FANDHE_EDGE_BIN        CLI のバイナリ（`/` を含むパス。相対は呼び出し時のカレント基準で絶対化する）。
#                          存在しない・実行できない場合は invalid_input(64)。未設定なら cargo build --locked --release の `compiler-artifact` の executable を使う
#   FANDHE_EDGE_MAKE_CMD   make の代役（テスト専用。絶対パスの実行ファイル）
#   FANDHE_EDGE_CARGO_CMD  cargo の代役（テスト専用。絶対パスの実行ファイル）
#   FANDHE_EDGE_TRAINER_DIR  CLI の train が読む trainer の場所。値は読まず、設定の有無だけを記録する
#   MAKE_CMD・CARGO_CMD のどちらかを設定すると record の evidence_hint は test_harness になる
#
# 契約:
#   - 引数の誤りは CLI・make・cargo を起動する前に {"code":"invalid_input","message":"<固定>"} を 1 行出して exit 64
#   - 終了コード: 全項目 ok なら 0、項目の失敗（または未実行の要求項目）は 10、引数の誤りは 64、
#     実行中に commit・worktree_clean・CLI の sha256 が変わった場合も 10（#360）、
#     スクリプト自身の実行不能（python3 が無い・作業ディレクトリを作れない・CLI のビルド失敗等。
#     起動前に分かる入力の誤りの FANDHE_EDGE_BIN の不在・実行不可は 64）は 70（REQ-21）
#   - stdout は最後に JSON を 1 つだけ出す（REQ-33）: {"code":..,"message":..,"record":"record.json"}。
#     パスは書かない。進行状況は stderr へ出す。record にはパス・データ本文・stderr の内容を書かない
#   - 子プロセスには上限時間と出力サイズ上限を設ける（REQ-39。値は real_machine_check_record.py の定数）。
#     A 以外の子プロセスには CARGO_NET_OFFLINE=true を渡し、cargo は --locked で起動する（REQ-38）。
#     SIGINT・SIGTERM・SIGHUP を受けたら子のグループを止め、その時点までの record を書いて exit 70。
#     後始末が終わらないときのため、シグナルを 2 回受けたら子のグループへ KILL を送って即座に
#     exit 70 する（record なし。stdout は固定 JSON `interrupted (forced exit)`。REQ-39）
set -eu

# 固定メッセージの JSON を 1 行出して終了する。$1=終了コード $2=code 名 $3=固定メッセージ
fail() {
    printf '{"code":"%s","message":"%s"}\n' "$2" "$3"
    exit "$1"
}

# ---- 引数の検証（ここで失敗したら make・cargo・CLI を起動しない） ----
work_dir=
items=B,C,D,E,F
repeat=50
quiet=0
with_ci=0
p95_limit=50000
pkg_limit=1000
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
            fail 0 ok "usage: real-machine-check.sh --work-dir DIR [--items LIST] [--repeat N] [--quiet-machine] [--with-ci] [--p95-limit-us N] [--package-limit-bytes N]"
            ;;
        --quiet-machine | --with-ci)
            [ "$has_val" -eq 0 ] || fail 64 invalid_input "option does not take a value"
            case "$seen" in *" $key "*) fail 64 invalid_input "duplicate option" ;; esac
            seen="$seen $key "
            if [ "$key" = --quiet-machine ]; then quiet=1; else with_ci=1; fi
            shift
            continue
            ;;
        --work-dir | --items | --repeat | --p95-limit-us | --package-limit-bytes) ;;
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
        --work-dir) work_dir=$val ;;
        --items) items=$val ;;
        --repeat) repeat=$val ;;
        --p95-limit-us) p95_limit=$val ;;
        --package-limit-bytes) pkg_limit=$val ;;
    esac
done

[ -n "$work_dir" ] || fail 64 invalid_input "--work-dir is required"

# --items: A〜F の部分集合（大文字・重複なし・空要素なし）
case "$items" in
    '' | ,* | *, | *,,*) fail 64 invalid_input "--items must be a comma-separated subset of A,B,C,D,E,F" ;;
esac
item_seen=
old_ifs=$IFS
IFS=,
for it in $items; do
    case "$it" in
        A | B | C | D | E | F) ;;
        *)
            IFS=$old_ifs
            fail 64 invalid_input "--items must be a comma-separated subset of A,B,C,D,E,F"
            ;;
    esac
    case "$item_seen" in
        *" $it "*)
            IFS=$old_ifs
            fail 64 invalid_input "--items must not contain duplicates"
            ;;
    esac
    item_seen="$item_seen $it "
done
IFS=$old_ifs
case "$item_seen" in
    *" A "*)
        [ "$with_ci" -eq 1 ] || fail 64 invalid_input "item A requires --with-ci (make ci may use the network)"
        ;;
esac
# E は B の成果物（package/ と train.jsonl）を使うため、B なしの指定は起動前に拒否する
case "$item_seen" in
    *" E "*)
        case "$item_seen" in
            *" B "*) ;;
            *) fail 64 invalid_input "item E requires item B" ;;
        esac
        ;;
esac

# 整数オプション（先頭 0 は拒否。桁数を制限して算術の桁あふれを避ける）
case "$repeat" in
    [1-9] | [1-9][0-9] | [1-9][0-9][0-9] | 1000) ;;
    *) fail 64 invalid_input "--repeat must be an integer from 1 to 1000" ;;
esac
case "$p95_limit" in
    [1-9] | [1-9][0-9]* ) ;;
    *) fail 64 invalid_input "--p95-limit-us must be an integer from 1 to 3600000000" ;;
esac
case "$p95_limit" in
    *[!0-9]* | ????????????????*) fail 64 invalid_input "--p95-limit-us must be an integer from 1 to 3600000000" ;;
esac
# 上限は定義ファイルの limits.max_infer_p95_us の上限（crates/core/src/definition.rs の
# MAX_LIMIT_INFER_P95_US = 3,600,000,000 µs）と対。超過を C-1 の register まで持ち越さず、
# ここで引数の誤り（64）にする。15 桁以内に絞った後なので算術の桁あふれは起きない
[ "$p95_limit" -le 3600000000 ] || fail 64 invalid_input "--p95-limit-us must be an integer from 1 to 3600000000"
case "$pkg_limit" in
    [1-9] | [1-9][0-9]*) ;;
    *) fail 64 invalid_input "--package-limit-bytes must be a positive integer" ;;
esac
case "$pkg_limit" in
    *[!0-9]* | ????????????????*) fail 64 invalid_input "--package-limit-bytes must be a positive integer" ;;
esac

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

if [ -e "$work_dir" ] || [ -L "$work_dir" ]; then
    if [ ! -d "$work_dir" ] || [ -L "$work_dir" ]; then
        fail 64 invalid_input "work directory is not a directory"
    fi
    [ -z "$(ls -A -- "$work_dir")" ] || fail 64 invalid_input "work directory is not empty"
fi
root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P) || fail 70 runtime_error "cannot resolve repository root"
canon_path "$work_dir" || fail 64 invalid_input "cannot resolve work directory"
work_canon=$canon_out
# リポジトリには何も書かない。作業ディレクトリがリポジトリ配下（または同一・祖先）なら拒否する
case "$work_canon/" in
    "$root/"*) fail 64 invalid_input "work directory must be outside the repository" ;;
esac
case "$root/" in
    "$work_canon/"*) fail 64 invalid_input "work directory must be outside the repository" ;;
esac

# ---- 前提（python3・CLI・make/cargo の代役）。項目を起動する前に fail-closed で確認する ----
command -v python3 >/dev/null 2>&1 || fail 70 runtime_error "python3 is required"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1 \
    || fail 70 runtime_error "python3 3.9 or newer is required"

# 代役（テスト専用）は絶対パスの実行ファイルに限る（PATH で探さない。誤って別の実行ファイルを起動しない）
check_cmd() {
    [ -n "$2" ] || return 0
    case "$2" in
        /*) ;;
        *) fail 64 invalid_input "$1 must be an absolute path to an executable" ;;
    esac
    if [ ! -f "$2" ] || [ ! -x "$2" ]; then
        fail 64 invalid_input "$1 must be an absolute path to an executable"
    fi
}
check_cmd FANDHE_EDGE_MAKE_CMD "${FANDHE_EDGE_MAKE_CMD:-}"
check_cmd FANDHE_EDGE_CARGO_CMD "${FANDHE_EDGE_CARGO_CMD:-}"

# FANDHE_EDGE_BIN は絶対の物理パスへ正規化してから渡す（ハッシュを取る対象と実行する対象を同一にする）。
# 相対パスは呼び出し時のカレントを基準に解決し、`/` を含まない裸の名前は拒否する
bin_args=
if [ -n "${FANDHE_EDGE_BIN:-}" ]; then
    bin=$FANDHE_EDGE_BIN
    case "$bin" in
        */*) ;;
        *) fail 64 invalid_input "FANDHE_EDGE_BIN must be a path containing a slash" ;;
    esac
    case "$bin" in /*) ;; *) bin="$PWD/$bin" ;; esac
    bin_dir=$(cd -P -- "$(dirname -- "$bin")" 2>/dev/null && pwd -P) \
        || fail 64 invalid_input "FANDHE_EDGE_BIN must be a path to an executable file"
    bin="${bin_dir%/}/$(basename -- "$bin")"
    if [ ! -f "$bin" ] || [ ! -x "$bin" ]; then
        fail 64 invalid_input "FANDHE_EDGE_BIN must be a path to an executable file"
    fi
fi

# 作業ディレクトリ（最上位を含む）は本人だけが読み書きできる権限で作る
umask 077
mkdir -p -- "$work_canon" || fail 70 runtime_error "cannot create work directory"
chmod 700 "$work_canon" || fail 70 runtime_error "cannot create work directory"
# mktemp は O_EXCL で作るため、予測可能な名前への symlink 事前配置を辿らない（#359）
probe=$(mktemp "$work_canon/.write-probe.XXXXXX" 2>/dev/null) || fail 70 runtime_error "cannot write to work directory"
rm -f -- "$probe"

# 以降は python3 が項目の実行と記録を行い、stdout の JSON と終了コードを返す。
# 任意の引数は値があるときだけ展開する。未設定の CLI は python3 が cargo build の成果物から特定する
set -- run \
    --repo-root "$root" \
    --work-dir "$work_canon" \
    --items "$items" \
    --repeat "$repeat" \
    --p95-limit-us "$p95_limit" \
    --package-limit-bytes "$pkg_limit"
[ -z "${bin:-}" ] || set -- "$@" --bin "$bin" --bin-override
[ "$quiet" -eq 0 ] || set -- "$@" --quiet-machine
[ "$with_ci" -eq 0 ] || set -- "$@" --with-ci
exec python3 -I "$root/scripts/real_machine_check_record.py" "$@"
