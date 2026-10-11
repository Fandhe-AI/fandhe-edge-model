#!/usr/bin/env bash
# 実 trainer（MLX CPU）を Rust の run_train から起動する結合テスト（issue #258。
# REQ-18/19/34/39）。crates/train/tests/real_trainer.rs は #[ignore] で既定の `make test` から
# 分離してあり、ここで --ignored 付きで実行する（python-ci.yml と `make ci` の両方で実行するため、
# CI を通すための skip ではない）。
# libtest のテスト名フィルタは 1 回の起動につき 1 つしか渡せないため、テストごとに個別に起動する
# （「パッケージ:テストバイナリ:テスト名」の組で回す。sandbox_pipeline_real_trainer は sandbox 監視
# チェーンの完走確認〔評価なし・評価あり〕。REQ-38・TASK-38.1・#161）。cargo test 自体の終了状態を
# 保持するためパイプ（tee）は使わず、一時ファイルへ出力してから表示する。--exact のテスト名がずれると
# 0 件実行で成功してしまうため、出力の「1 passed」も検査して fail-closed にする
# （テスト名を変えたらここも更新する。crates/train/tests/real_trainer.rs・
# crates/cli/tests/sandbox_pipeline_real_trainer.rs の冒頭コメント参照）。
# Cargo.toml・メンバー crate・trainer/pyproject.toml・Windows の有無による skip は Makefile 側で判定する。
# 前提: PIN_UV_VERSION（Makefile が定義元）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"
require_uv

specs=(
  fandhe-edge-train:real_trainer:req18_real_trainer_c1_job_completes_with_typed_outcome
  fandhe-edge-train:real_trainer:req18_real_trainer_c3_job_completes_with_validation_predictions
  fandhe-edge-cli:sandbox_pipeline_real_trainer:req38_real_pipeline_completes_under_monitor_with_zero_tool_denials
  fandhe-edge-cli:sandbox_pipeline_real_trainer:req38_real_pipeline_with_evaluation_data_completes_under_monitor_with_zero_tool_denials
)

out="$(mktemp)"
trap 'rm -f "$out"' EXIT
overall=0
for spec in "${specs[@]}"; do
  pkg="${spec%%:*}"
  rest="${spec#*:}"
  bin="${rest%%:*}"
  t="${rest#*:}"
  status=0
  cargo test -p "$pkg" --test "$bin" -- --ignored --exact "$t" >"$out" 2>&1 || status=$?
  cat "$out"
  if [ "$status" -ne 0 ]; then
    echo "error: $bin の $t が失敗しました（終了コード $status）" >&2
    overall=1
  elif ! grep -q "test result: ok. 1 passed" "$out"; then
    echo "error: $bin の $t が実行・成功していません（テスト名の不一致の可能性）" >&2
    overall=1
  fi
done
exit "$overall"
