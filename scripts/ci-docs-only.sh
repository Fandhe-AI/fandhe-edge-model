#!/usr/bin/env bash
# PR の変更ファイルが「実行結果に影響しないファイルのみ」かを判定し、GITHUB_OUTPUT へ
# `docs_only=true|false` を書く。python-ci.yml が macOS runner の重いステップを省略するために使う。
# 必須チェック（python-ci）はジョブ名で登録されているため、workflow 単位の paths フィルタは使えず、
# ジョブ内でこの判定を行う。
#
# fail-closed: pull_request 以外・変更一覧を取得できない・一覧が空・1 件でも対象外のファイルが
# ある場合は false（= 全ステップを実行する）。
# 前提の環境変数: EVENT_NAME・PR_NUMBER・REPOSITORY・GH_TOKEN・GITHUB_OUTPUT（Actions が供給）
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

if [ "${EVENT_NAME:-}" != "pull_request" ] || [ -z "${PR_NUMBER:-}" ] || [ -z "${REPOSITORY:-}" ]; then
  emit false
  exit 0
fi

if ! files="$(gh api --paginate "repos/${REPOSITORY}/pulls/${PR_NUMBER}/files" --jq '.[] | .filename, (.previous_filename // empty)')" ||
  [ -z "$files" ]; then
  echo "warn: could not list changed files; running everything" >&2
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
