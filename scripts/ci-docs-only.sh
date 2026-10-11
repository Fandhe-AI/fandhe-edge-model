#!/usr/bin/env bash
# PR の変更ファイルが「実行結果に影響しないファイルのみ」かを判定し、GITHUB_OUTPUT へ
# `docs_only=true|false` を書く。python-ci.yml が macOS runner の重いステップを省略するために使う。
# 必須チェック（python-ci）はジョブ名で登録されているため、workflow 単位の paths フィルタは使えず、
# ジョブ内でこの判定を行う。
#
# 変更一覧は API ではなく、チェックアウト済みの merge commit（pull_request イベントの既定の checkout 対象）
# とその第 1 親（base 側）の差分から取る。検証対象のコミットと判定の入力が必ず一致し（再実行時に
# PR の現在の状態とずれない）、API の 3000 件上限にも影響されない。
#
# fail-closed: pull_request 以外・HEAD が merge commit でない・第 1 親を取得できない・差分が空・
# 1 件でも対象外のファイルがある場合は false（= 全ステップを実行する）。
# 前提: actions/checkout は fetch-depth: 2 以上。環境変数 EVENT_NAME と GITHUB_OUTPUT（Actions が供給）
#
# 対象は「テスト・ビルドが読まない」ことを確認したパスに限る。docs/design/*.md と lefthook.yml は
# pytest が読むため含めない。対象を増やす場合は読まれていないことを grep で確認する。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

emit() {
  echo "docs_only=$1" >>"${GITHUB_OUTPUT:-/dev/stdout}"
  echo "docs_only=$1"
}

if [ "${EVENT_NAME:-}" != "pull_request" ]; then
  emit false
  exit 0
fi

# HEAD が merge commit（親が 2 つ）でなければ base との差分を定義できない
if [ "$(git rev-list --parents -n 1 HEAD | wc -w)" -ne 3 ] ||
  ! files="$(git diff --name-only --no-renames HEAD^1 HEAD)" || [ -z "$files" ]; then
  echo "warn: could not compute the PR diff from the merge commit; running everything" >&2
  emit false
  exit 0
fi

docs_only=true
while IFS= read -r f; do
  case "$f" in
    .claude/* | .agents/* | skills-lock.json | CLAUDE.md | AGENTS.md | README.md | LICENSE-APACHE | LICENSE-MIT) ;;
    *)
      docs_only=false
      break
      ;;
  esac
done <<<"$files"
emit "$docs_only"
