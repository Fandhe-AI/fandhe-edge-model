#!/bin/sh
# Bash ツールなど非対話の呼び出し元から `fandhe-edge infer` を実行し、
# 終了コードと stdout（JSON）を機械的に受け取れることを確認するスクリプト
# （REQ-36・TASK-36.1-1・#149）。
#
# 呼び出し元: Claude Code / Codex の Bash ツール、および
# crates/cli/tests/bash_noninteractive.rs（結合テスト）。
#
# 契約:
#   - 引数はすべて `fandhe-edge infer` へそのまま渡す（"$@" のみ。eval しない）
#   - stdin は /dev/null に閉じ、入力待ちを作らない（非対話の要点）
#   - stdout は CLI の出力を無加工で中継する（1 呼び出し 1 JSON。REQ-33）
#   - stderr は CLI の stderr に続けて診断 `exit_code=<N>` を 1 行だけ出す
#     （引数値・入力テキストは出さない）
#   - 終了コードは CLI のものをそのまま返す。バイナリが無ければ 70
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

if [ ! -x "$bin" ]; then
    echo "fandhe-edge binary not found or not executable" >&2
    exit 70
fi

if "$bin" infer "$@" </dev/null; then
    rc=0
else
    rc=$?
fi
echo "exit_code=$rc" >&2
exit "$rc"
