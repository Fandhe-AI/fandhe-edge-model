"""Mac 実機での動作確認（項目 A〜J）の実行・要約・記録（record.json / record.md）の生成。

REQ-18・REQ-21・REQ-26・REQ-27・REQ-28・REQ-30・REQ-31・REQ-32・REQ-33・REQ-34・REQ-38・
REQ-39。特定の TASK には対応しない横断の確認ツール。REQ-38 のうち sandbox 下の通信 0 件の
判定は対象外（`sandbox-monitor.sh` の担当）。
本スクリプトは通信を起こさない側に倒す（cargo は `--locked`、A 以外の子には `CARGO_NET_OFFLINE`、
A を含む全ての子には rustup のツールチェーン自動取得を止める `RUSTUP_AUTO_INSTALL=0`。#375）。

呼び出し元: `scripts/real-machine-check.sh`（引数検証・作業ディレクトリの用意の後に
`python3 -I` で `run` を起動する）。製品（CLI・推論経路・配布物）には入らない検証用スクリプトで、
標準ライブラリだけを使う（依存の追加なし。Python 3.9 の文法で書く）。

責務:
- 項目 A〜J の子プロセス起動（上限時間・出力サイズ上限つき。中断時は子のグループを止める。REQ-39）
- 実行全体の上限時間（`--overall-timeout-sec`）と、環境採取の子（git・sysctl・sw_vers・otool）の
  固定パス・最小の環境での起動（PATH・GIT_* に左右されない。REQ-38・REQ-39・#364）
- CLI の JSON の要約と、記録へ出してよい値だけへの絞り込み（伏せ処理は `sanitize_record` に集約）
- `record.json`・`record.md` の書き出しと、stdout への固定メッセージ JSON 1 行

記録の規則（最重要）: 記録に書くのは件数・ハッシュ・サイズ・終了コード・固定語彙・検証済みの
数値と真偽値だけ（許可リスト方式）。工程の要約は工程ごとの許可した欄だけを組み立て、失敗時の
`message` は本文でなくバイト数とハッシュだけにする。パス（`/` や `\\` などを含む文字列。例外は D の
ライブラリ名）・データ本文・利用者が決めた文字列（選択肢 ID・`input`・`text`・`id` の値）・
stderr の内容は書かない。最後の関門 `sanitize_record` は、文字列を持ってよいキーの集合に無い
位置の文字列を伏せる。

証拠種別: このスクリプトは測定と整形までを行い、「実機」の証拠としての確定は人が行う
（`evidence_hint` は `requires_human_review` か `test_harness` のみ）。
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "real-machine-check/1"
ITEM_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
# 既定の項目（`scripts/real-machine-check.sh` の `items` 既定値と一致。pytest が機械照合する）。
# A は通信しうる（`--with-ci`）、I は GPU を長時間占有しうる（#103）ため既定に入れない
DEFAULT_ITEMS = ["B", "C", "D", "E", "F", "G", "H", "J"]
ITEMS_MESSAGE = "--items must be a comma-separated subset of A,B,C,D,E,F,G,H,I,J"
# 公開前の組み立て先ディレクトリ名。`package` の実行後（exit 0・20 のどちらも）に残ってはならない
# （crates/cli/src/project.rs の `PACKAGE_STAGING_DIR` と一致。pytest が機械照合する。REQ-30）
PACKAGE_STAGING_DIR = "package.staging"
# `infer --text` で `--id` を省略したときの既定 id（crates/cli/src/stages/infer.rs の
# `DEFAULT_TEXT_ID` と一致。pytest が機械照合する。REQ-33）
DEFAULT_TEXT_ID = "input"

# 終了コード 7 種（REQ-21）のうち本スクリプトが使うもの
EXIT_OK = 0
EXIT_JUDGED_FAIL = 10
EXIT_INVALID_INPUT = 64
EXIT_RUNTIME_ERROR = 70

# 子プロセスの上限時間（秒。REQ-39）
TIMEOUT_CLI_STEP = 600
TIMEOUT_MAKE_CI = 3600
TIMEOUT_LINKAGE = 1800
TIMEOUT_CARGO_TEST = 300
# 契約に定めのない値（自分で決めた点）: ビルド・ビルド済みの確認と環境採取
TIMEOUT_BUILD = 1800
TIMEOUT_PROBE = 30

# 出力サイズ上限（バイト。読み込み前に確認する。REQ-39）
CAP_CLI_STDOUT = 1024 * 1024
CAP_CLI_STDERR = 8 * 1024 * 1024
# 契約に定めのない値（自分で決めた点）: make・cargo のログ
CAP_LOG_STDOUT = 64 * 1024 * 1024
CAP_LOG_STDERR = 64 * 1024 * 1024
CAP_PROBE = 64 * 1024
CAP_INPUT_FILE = 16 * 1024 * 1024
CAP_PACKAGE_FILE = 256 * 1024 * 1024
# CLI 本体のハッシュを取る上限（release ビルドは数 MB。異常に大きいものは読まない）
CAP_CLI_BINARY = 1024 * 1024 * 1024

# 学習する候補の番号（B・C の `train --candidate`。select は同じ候補を返す）
TRAIN_CANDIDATE = 0
# スコア合計が 1 から外れてよい許容差。crates/core/src/judgment.rs の `SCORE_SUM_TOLERANCE` と同じ値
# （実 CLI の runtime は合計がこの範囲に収まらない出力を `RuntimeError` にする）
SCORE_SUM_TOLERANCE = 1e-6
# check-runtime-linkage.sh の `for t in ...` に並ぶ env -i テストの数（pytest で照合する）
LINKAGE_ENV_I_TESTS = 3

# G の `--g-budget-seconds` の上限（crates/train/src/search.rs の `MAX_SEARCH_BUDGET_SECONDS` =
# 3600 × 256。CLI の `--budget-seconds` の上限と同じ）。既定は CLI の既定（3600）に揃える
MAX_G_BUDGET_SECONDS = 921_600
DEFAULT_G_BUDGET_SECONDS = 3600
# G の子プロセスの上限時間は「探索予算 + この余裕」（予算内の学習が終わり次第戻るため）
G_TIMEOUT_MARGIN = 300
# H・I・J が使う固定の学習 seed（I は複製ごとに別の seed。`train --train-seed`。REQ-26・#490）
I_SEEDS = (1, 2, 3)
I_PREVIOUS_SEED = 4
# H のジョブ監視: `job.json` が running になり子孫が現れるまで待つ上限（秒）と、確認の間隔
H_WAIT_RUNNING_SEC = 120.0
H_POLL_SEC = 0.5
# H: 子孫の終了を待つ猶予（キャンセル猶予 15 秒の後の SIGKILL・lifeline を見込む。REQ-39）
H_DESCENDANT_GRACE_SEC = 30.0
# 契約に定めのない値（自分で決めた点）: `ps`・キャンセル要求の子の上限時間
TIMEOUT_AUX = 30
# 子孫の確認に使う `ps`（固定の絶対パス。PATH を探さない。REQ-38・REQ-39）
PS_PATH = "/bin/ps"
# H の `ps`（全プロセスの引数を読む）の出力上限（バイト。契約に定めのない値。REQ-39）
CAP_PS_STDOUT = 8 * 1024 * 1024
CAP_PS_STDERR = 64 * 1024
# `version_ledger.json` の上限（crates/core/src/version_ledger_record.rs の
# `MAX_VERSION_LEDGER_BYTES`）
CAP_VERSION_LEDGER = 2 * 1024 * 1024
# E の件数上限（子プロセスを件数ぶん起動するため）
MAX_E_RECORDS = 1000
MAX_REPEAT = 1000
# --p95-limit-us の上限（µs）。crates/core/src/definition.rs の MAX_LIMIT_INFER_P95_US と対
MAX_P95_LIMIT_US = 3_600_000_000
# --package-limit-bytes の上限（15 桁。シェル側の桁数検査と対。算術の桁あふれを避ける）
MAX_PACKAGE_LIMIT_BYTES = 999_999_999_999_999
# --overall-timeout-sec（実行全体の上限時間。秒。REQ-39）。契約に定めのない値（自分で決めた点）:
# 既定 4 時間は記録簿の通し実行（A〜H・J・`--repeat 50`。G の既定の探索予算 1 時間を含む）を収め、
# F の暴走（最大 1000 × 300 秒）を
# 止める桁。
# 下限 1 はテストハーネスが短い値で発火させるため。既定値の出所はシェル側の 1 箇所だけ
MIN_OVERALL_TIMEOUT_SEC = 1
MAX_OVERALL_TIMEOUT_SEC = 86400
# 浮動小数の許容差（evaluation-contract.md）。evaluate の accuracy の照合に使う。
# E の単体対バッチはスコアも完全一致で、この許容差は使わない
SCORE_TOLERANCE = 1e-9

REDACTED = "<redacted>"
# 閉じた語彙の欄で、文字列だが語彙外の値の代わりに出す固定の文字列
UNEXPECTED = "<unexpected>"
# `package_files[].name` が安全な名前の形に一致しないときの固定の文字列
UNRECOGNIZED = "<unrecognized>"
MAX_STR = 200
# 記録の文字列の欄の語彙（閉じた集合。fixtures・Rust のソースとの一致は pytest で機械照合する）
STATUS_VOCAB = frozenset({"ok", "skipped", "out_of_scope", "abstain"})
# infer の終了コードと判定行の `status` の対応（REQ-21・REQ-22。対象外 11・保留 12。#478・#497）。
# 単発は終了コードと `status` の対応が一致すれば合格、バッチは exit 0 で各行の `status` が値のどれか
INFER_EXIT_STATUS = {0: "ok", 11: "out_of_scope", 12: "abstain"}
# 終了コード 7 種の名前（`fixtures/exitcode/exit_codes.json` の `name`。REQ-21）
CODE_VOCAB = frozenset(
    {
        "ok",
        "judged_fail",
        "out_of_scope",
        "pending",
        "limit_exceeded",
        "invalid_input",
        "runtime_error",
    }
)
# `PackageJudgment`（crates/core/src/stage_report.rs）
JUDGMENT_VOCAB = frozenset({"pass", "fail", "undeterminable"})
# `select` の `significance.verdict`（crates/core/src/evaluation_record.rs の
# `BaselineComparisonVerdict`。REQ-25・#481）。`undeterminable` も正常な値（件数不足は合格にしない）
SIGNIFICANCE_VOCAB = frozenset(
    {"significantly_better", "not_significantly_better", "undeterminable"}
)
# `train --all` の候補ごとの結果と予算到達の範囲（crates/core/src/stage_report.rs の
# `TrainSearchResult`・`TrainBudgetScope`。REQ-18・#482）
TRAIN_RESULT_VOCAB = frozenset(
    {
        "evaluated",
        "training_not_completed",
        "scoring_failed",
        "scoring_exceeded_budget",
        "scoring_skipped_budget_exhausted",
        "training_exceeded_time_limit",
        "training_timed_out",
        "not_started",
    }
)
BUDGET_SCOPE_VOCAB = frozenset({"search_budget", "candidate_time_limit"})
# ジョブ状態・クラッシュの観測点・キャンセル要求の結果（crates/train/src/job_record.rs の
# `JobState`・`CrashCause`、crates/train/src/stage_files.rs の `CancelOutcome`。REQ-34・#484・#485）
JOB_STATE_VOCAB = frozenset({"queued", "running", "cancelling", "cancelled", "succeeded", "failed"})
CRASH_CAUSE_VOCAB = frozenset({"worker_signal", "supervisor_signal", "owner_lost"})
CANCEL_OUTCOME_VOCAB = frozenset({"requested", "already_cancelling", "already_finished"})
# やり直し案内の固定語彙（crates/train/src/restart.rs。再開は提供しない。REQ-34）
RESTART_ACTION = "restart_from_scratch"
RESTART_REASON_CODE = "resume_not_supported"
# キャンセルされた単発 `train` の `message`（crates/cli/src/stages/train.rs の `cancelled_report`）
CANCELLED_MESSAGE = "training cancelled"
# 再現性・旧モデルとの比較の語彙（crates/core/src/evaluation_record.rs。REQ-26・#488〜#490）
REPRODUCIBILITY_VERDICT_VOCAB = frozenset({"all_pairs_overlap", "some_pairs_disjoint"})
COMPARISON_PREMISE_VOCAB = frozenset({"same_label_set", "label_set_differs"})
COMPARISON_DATA_VOCAB = frozenset({"same", "common_subset"})
# I の証拠種別（`--i-device`。CLI の `train` は現状 CPU 固定のため、gpu は人の申告）
I_EVIDENCE = {"cpu": "cpu_real_machine", "gpu": "gpu_real_machine_declared"}
# 版 ID の形（`v<n>`。crates/cli/src/stages/package.rs の台帳の版）
VERSION_ID_RE = re.compile(r"v[1-9][0-9]{0,8}")
# `kind` の許可リスト（crates/guard/src/kind.rs の `SUPPORTED_KINDS`。REQ-39）
KIND_VOCAB = frozenset({"c1", "c3", "autoregressive"})
# `package` の容量の目安（crates/runtime/src/capacity_limit.rs の `REFERENCE_CAPACITY_BYTES`。
# 強制上限ではなく、超過は `over_guideline` の警告のみ。REQ-30・TASK-41.9。
# pytest で Rust のソースと照合する）
REFERENCE_CAPACITY_BYTES = 40_000_000
# パス区切りとして扱う文字（`/`・`\`・U+2215 DIVISION SLASH・U+2044 FRACTION SLASH・
# U+FF0F FULLWIDTH SOLIDUS）
PATH_CHARS = ("/", "\\", "∕", "⁄", "／")
CAPACITY_COMPONENTS = (
    "weights",
    "vocab_or_feature_transform",
    "label_table",
    "calibration",
    "metadata",
)
LIB_RE = re.compile(r"/(usr/lib|System)/[A-Za-z0-9._+/-]+")
# D のライブラリ名 1 件の長さの上限
MAX_LIB_LEN = 128
# record の dict のキーに許す形（規則外のキーは `<redacted>` へ置き換える）
KEY_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PACKAGE_FILE_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
ENV_TEXT_RE = re.compile(r"[A-Za-z0-9 ._,()+-]{1,64}")
ASCII_INT_RE = re.compile(r"[0-9]{1,20}")
# 文字列の値を持ってよいキー（`sanitize_record` の閉じた集合）。実際に記録へ出す文字列の欄だけを
# 並べる。`sha256`・`*_sha256` は関数側で扱う。集合に無いキーの下の文字列は伏せる
STR_KEYS = frozenset(
    {
        "schema",
        "evidence_hint",
        "status",
        "reason",
        "step",
        "command",
        "code",
        "kind",
        "judgment",
        "classification",
        "case",
        "error_type",
        "linkage_tool",
        "name",
        "commit",
        "commit_end",
        "cli_origin",
        "trainer_origin",
        "cli_profile",
        "started_local",
        "ended_local",
        "hw_model",
        "cpu",
        "os_name",
        "os_version",
        "os_build",
        "verdict",
        "result",
        "budget_reached",
        "state",
        "cause",
        "action",
        "evidence",
        "device",
        "premise",
        "evaluation_data",
        "cancel",
        "significance_verdict",
        "outcome",
        "i_device",
        "restart_action",
    }
)
# list の要素として文字列を持ってよい位置（`options.items`・`items.D.direct_libraries`）
LIST_STR_PATHS = frozenset({("options", "items"), ("items", "D", "direct_libraries")})
LIBRARIES_PATH = ("items", "D", "direct_libraries")
FIXTURE_FILES = ("definition.json", "train.jsonl", "evaluation.jsonl")
INTERRUPT_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

# 2 回目の中断シグナルによる強制終了で stdout へ書く固定の 1 行（record は無い）
FORCED_EXIT_LINE = b'{"code":"runtime_error","message":"interrupted (forced exit)"}\n'

# 環境採取（git・sysctl・sw_vers・otool）に使うコマンドの固定パス（PATH を探さない。
# REQ-38・REQ-39）。PATH の先頭に同名の実行ファイルを置かれても別物の出力が記録に入らない
# ようにする。`/usr/local/bin` は
# 利用者が書ける構成があるため候補に入れない。表に無い・実行できない場合は「使えない」として扱い、
# PATH へは戻らない（git が無ければ commit・worktree_clean が null → stable が null → exit 10。
# sysctl・sw_vers が無ければ該当欄が null。macOS で otool が無ければ D は otool_failed）。
# Linux 側の表はテストハーネス用で、実機の証拠にならない（sysctl・sw_vers・otool は使わない）
TOOL_CANDIDATES_DARWIN: dict[str, tuple[str, ...]] = {
    "git": ("/usr/bin/git",),
    "sysctl": ("/usr/sbin/sysctl",),
    "sw_vers": ("/usr/bin/sw_vers",),
    "otool": ("/usr/bin/otool",),
}
TOOL_CANDIDATES_OTHER: dict[str, tuple[str, ...]] = {
    "git": ("/usr/bin/git", "/bin/git"),
}
# 環境採取の子へ渡す PATH（固定。`run_cmd` の外側の /bin/sh ラッパーが rm を呼ぶため /bin も含める）
PROBE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# 子の外側で動く薄い sh。コマンドを子として走らせ、終了コードを rc ファイル（位置引数 $1）へ
# 書いてから、自分を含むグループ全体へ KILL を送る（TERM を無視する孫も残さない。REQ-39）。
# リーダー（この sh）の生存中に行うので、reap 後の killpg による pid 再利用の誤爆がない。
# 文字列は定数で、値は位置引数で渡す
# rc の書き込み直前の `set -C`（noclobber）は、子の実行中に rc のパスへ置かれた
# symlink・既存ファイルをシェルが辿って書かないため（書けなければ rc が無く、
# 呼び出し側は `killed` にする。fail-closed。#359）
# 書けなかったとき（子が置いた通常ファイル・symlink が残っている）は `rm -f` でそれを消し、
# 偽の終了コードを呼び出し側へ読ませない（rc が無ければ `killed`。fail-closed。#359）
GROUP_WRAPPER = (
    'RC=$1; shift; "$@"; rc=$?; set -C; printf "%s" "$rc" > "$RC" || rm -f "$RC"; kill -s KILL 0'
)
# `GROUP_WRAPPER` から最後のグループ KILL を除いたもの。H が「CLI が子孫を自分で止めたか」を確かめる
# ときだけ使う（ラッパーが孫を片付けると、lifeline・killpg が効いていなくても子孫が 0 件に見える）。
# 残った子孫は H が pid を控えて確認し、片付ける（REQ-39）
GROUP_WRAPPER_NO_REAP = (
    'RC=$1; shift; "$@"; rc=$?; set -C; printf "%s" "$rc" > "$RC" || rm -f "$RC"'
)
# rc ファイルの上限（3 桁の整数だけが入る）
CAP_RC_FILE = 16


class OverallTimeout(BaseException):  # `except Exception` に飲まれないよう BaseException にする
    """実行全体の上限時間（`--overall-timeout-sec`）を超えた。`Interrupted` と同じ形で送出する。"""


class Interrupted(BaseException):  # Exception で捕まえられないよう BaseException にする
    """SIGINT・SIGTERM・SIGHUP を受けた（`check_interrupt` の位置でだけ同期的に送出する）。"""


# 中断シグナルを受けた印。ハンドラ（`_on_signal`）が立て、`run()` の冒頭で下ろす（REQ-21・REQ-39）
_interrupt_requested = False
# 中断シグナルを受けた回数。2 回目以降は後始末の完了を待たず強制終了する（`_on_signal`。REQ-39）
_signal_count = 0
# 実行中の子のプロセスグループ id。`Popen` 生成直後に立て、回収の直後に下ろす（`run_cmd`）。
# 2 回目のシグナルのハンドラが KILL を送る宛先。回収から下ろすまでの数命令の窓で pid が
# 再利用される誤爆の余地は既知の限界（確率は無視できる）
_active_pgid: int | None = None
# `Popen` の呼び出しから `_active_pgid` の登録までの間だけ立てる印。この間は宛先の pgid が
# まだ無いため、2 回目のシグナルでも強制終了せず `_force_pending` を立てて戻る（REQ-39）
_spawning = False
# 起動の窓の間に 2 回目のシグナルを受けた印。登録の直後に `run_cmd` が強制終了する
_force_pending = False
# 最終の JSON を stdout へ書き始めた印。以後の 2 回目のシグナルは強制終了の行を書かず無視し、
# stdout の JSON を 1 つに保つ（REQ-21・REQ-33）
_final_emitted = False
# 最終 JSON を stdout へ書けなかった印。終了処理のフラッシュ失敗（exit 120）を防ぐ（REQ-21）
_stdout_failed = False
# 子の回収が上限時間内に終わらず、子（またはその孫）が残っている可能性がある印（REQ-39）。
# record の `child_may_remain` へ出す。`run()` の冒頭で下ろす
_child_may_remain = False
# 回収できなかった子の `Popen`。2 回目のシグナルの強制終了（`_force_exit`）が再度 KILL を
# 試みるため残す（親だけが終わって子が残るのを避ける。REQ-39）。pid の数値ではなく `Popen` を
# 保持するのは、参照が生きている間は GC による暗黙の回収が起きず、リーダーは未回収（zombie を
# 含む）のままなので pid が再利用されず、宛先が無関係なグループにならないため。`_force_exit` は
# `poll()` 等の回収を一切呼ばない。`run()` の冒頭で空にする
_leftover_procs: list[subprocess.Popen[bytes]] = []
# 後始末後の回収（`proc.wait`）を待つ上限秒数。KILL が効けば即時に回収できるため、実際に
# 待つのは KILL が失敗した異常時だけ。超過したら諦めて先へ進む（無限待ちを作らない。REQ-39）。
# モジュール変数にしてあるのはテストが縮めるため
REAP_WAIT_LIMIT_SECONDS = 10.0
# 回収を諦めたときの結果の理由（固定語彙）。項目は `failed` になり、通常の合否判定へ流さない
REASON_UNREAPED = "unreaped"
# 実行全体の上限時間を超えたときの理由（固定語彙。`failed`・`not_run` の両方に使う）
REASON_OVERALL_TIMEOUT = "overall_timeout"


# 全体の期限（`time.monotonic()` 基準）と超過の印。`run()` が設定し、`run_cmd` が子ごとの期限より
# 先に見る。
# 超過を受けた `run()` は両方を下ろし、終了時の再採取と記録の書き出しを期限の外で行う（REQ-39）
_overall_deadline: float | None = None
_overall_timed_out = False


def overall_expired() -> bool:
    """全体の期限を超えたか（超過の印が立っている場合を含む）。"""
    return _overall_timed_out or (
        _overall_deadline is not None and time.monotonic() >= _overall_deadline
    )


def clear_overall() -> None:
    """全体の期限と印を下ろす（超過を受けた後の再採取・記録の書き出しを期限の外で行うため）。"""
    global _overall_deadline, _overall_timed_out
    _overall_deadline = None
    _overall_timed_out = False


def check_overall() -> None:
    """期限を超えていれば `OverallTimeout` を送出する（`run_cmd` の入口から呼ぶ）。"""
    global _overall_timed_out
    if overall_expired():
        _overall_timed_out = True
        raise OverallTimeout


def check_interrupt() -> None:
    """印が立っていれば `Interrupted` を送出する。呼ぶ位置は決まった安全な境目だけ。

    `run_cmd` の入口・後始末の後と、`run()` の項目の開始前から呼ぶ（REQ-21・REQ-39）。
    """
    if _interrupt_requested:
        raise Interrupted


# --------------------------------------------------------------------------------------
# 記録へ出す値の絞り込み（許可リスト方式。最後の関門は `sanitize_record`）
# --------------------------------------------------------------------------------------


def _has_path_char(s: str) -> bool:
    """文字列がパス区切り（半角・全角・類似文字を含む）を含むか。"""
    return any(c in s for c in PATH_CHARS)


def _loads(text: str) -> Any:
    """`json.loads`。深すぎる入れ子の `RecursionError` も `ValueError` として扱う（REQ-39）。"""
    try:
        return json.loads(text)
    except RecursionError as e:
        raise ValueError("json nesting is too deep") from e


def _is_int(v: Any) -> bool:
    """bool を除く整数か。"""
    return isinstance(v, int) and not isinstance(v, bool)


def _nonneg_int(v: Any) -> bool:
    """bool を除く 0 以上の整数か。"""
    return _is_int(v) and v >= 0


def _is_finite_number(v: Any) -> bool:
    """bool を除く有限の数か。

    `float` へ変換できない巨大な整数（`math.isfinite` が `OverflowError` を出す）は、後段の
    `float` 変換・引き算でも例外になるため「有限の数」に数えず False にする（内部エラーにしない）。
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def _unit_number(v: Any) -> bool:
    """bool を除く有限で 0 以上 1 以下の数か。"""
    return _is_finite_number(v) and 0 <= v <= 1


# `error_type` の閉じた語彙。例外の型名は利用者・環境由来の任意文字列になりうるため、
# 組み込みの型名だけを残し、他は `UNEXPECTED` へ寄せる（REQ-21・REQ-39。#361）
ERROR_TYPE_VOCAB = frozenset(
    {
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "OSError",
        "FileNotFoundError",
        "PermissionError",
        "RuntimeError",
        "RecursionError",
        "MemoryError",
        "OverflowError",
        "ZeroDivisionError",
        "AssertionError",
        "UnicodeError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "NotImplementedError",
    }
)
# infer 工程の command の表示名。引数の値・パスは出さない（`/` を含めない。#361）
INFER_COMMAND_DISPLAY = "infer --package <package-dir> --text <fixed-sample>"


def error_type_name(e: BaseException) -> str:
    """例外の型名を閉じた語彙へ写す。語彙外・組み込みを名乗る自作クラスは `<unexpected>`。"""
    t = type(e)
    name = t.__name__
    if name in ERROR_TYPE_VOCAB and getattr(builtins, name, None) is t:
        return name
    return UNEXPECTED


def split_lines(text: str) -> list[str]:
    """LF だけで行に割る（行末の CR は除く）。Rust の `lines()` と JSONL の定義に揃える。

    標準の行分割は U+2028・U+0085・VT・FF でも割れ、JSONL の 1 行 1 JSON と食い違う（#361）。
    """
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    return [p[:-1] if p.endswith("\r") else p for p in parts]


def vocab_value(v: Any, vocab: frozenset[str]) -> str | None:
    """閉じた語彙の文字列欄。語彙内ならその値、文字列だが語彙外なら `<unexpected>`、他は None。"""
    if not isinstance(v, str):
        return None
    return v if v in vocab else UNEXPECTED


def _step_value(v: Any, name: str) -> str | None:
    """`step` 欄。工程名と一致すれば工程名、文字列だが不一致なら `<unexpected>`、他は None。"""
    return vocab_value(v, frozenset({name}))


def _int_or_none(v: Any) -> int | None:
    """0 以上の整数ならそのまま、違えば None（入れ子の dict・list や真偽値を通さない）。"""
    return v if _nonneg_int(v) else None


def _bool_or_none(v: Any) -> bool | None:
    """真偽値ならそのまま、違えば None。"""
    return v if isinstance(v, bool) else None


def _unit_or_none(v: Any) -> int | float | None:
    """有限で 0 以上 1 以下の数ならそのまま、違えば None。"""
    return v if _unit_number(v) else None


def p95_summary(v: Any) -> dict[str, Any] | None:
    """`infer_p95` を固定構造へ写す。`p95_us`・`limit_us` は 0 以上の整数、`exceeded` は真偽値。

    形が違えば None（REQ-31）。
    """
    if not isinstance(v, dict):
        return None
    p95, limit, exceeded = v.get("p95_us"), v.get("limit_us"), v.get("exceeded")
    if not _nonneg_int(p95) or not _nonneg_int(limit) or not isinstance(exceeded, bool):
        return None
    return {"p95_us": p95, "limit_us": limit, "exceeded": exceeded}


def _split_summary(v: Any) -> dict[str, int] | None:
    """`inspect` の `split`。3 欄とも 0 以上の整数のときだけ。形が違えば None。"""
    if not isinstance(v, dict):
        return None
    parts = {k: v.get(k) for k in ("train", "validation", "test")}
    return parts if all(_nonneg_int(x) for x in parts.values()) else None


def summarize_infer(obj: dict[str, Any], option_ids: list[str]) -> dict[str, Any]:
    """`infer` の判定 JSON の要約（REQ-33）。

    `status`・`scores_keys`（`scores` が dict ならキー数、違えば 0）・`predicted_index`
    （`predicted_label` が定義の選択肢 ID の何番目か。0 始まり。選択肢に無ければ None）だけ。
    選択肢 ID・データの id・スコアの値は利用者の値のため記録しない。
    """
    scores = obj.get("scores")
    label = obj.get("predicted_label")
    index = option_ids.index(label) if isinstance(label, str) and label in option_ids else None
    return {
        "status": vocab_value(obj.get("status"), STATUS_VOCAB),
        "scores_keys": len(scores) if isinstance(scores, dict) else 0,
        "predicted_index": index,
    }


def summarize_step(name: str, obj: dict[str, Any], option_ids: list[str]) -> dict[str, Any]:
    """工程の JSON を、工程ごとの許可した欄だけの新しい dict へ組み立てる（REQ-33）。

    元の JSON の他の欄は捨てる。欄は常に出し、欄が無い・型や形が検証に通らなければ None、
    閉じた語彙の欄で語彙外なら `<unexpected>`。入れ子の dict・list は文字列欄・数値欄へ通さない。
    """
    if name == "infer":
        return summarize_infer(obj, option_ids)
    out: dict[str, Any] = {
        "step": _step_value(obj.get("step"), name),
        "status": vocab_value(obj.get("status"), STATUS_VOCAB),
    }
    if name == "register":
        sha = obj.get("definition_sha256")
        out["options"] = _int_or_none(obj.get("options"))
        out["evaluation_defined"] = _bool_or_none(obj.get("evaluation_defined"))
        out["definition_sha256"] = (
            sha if isinstance(sha, str) and SHA256_RE.fullmatch(sha) else None
        )
    elif name == "inspect":
        out["valid_records"] = _int_or_none(obj.get("valid_records"))
        out["split"] = _split_summary(obj.get("split"))
    elif name in ("train", "select", "evaluate"):
        out["candidate"] = _int_or_none(obj.get("candidate"))
        out["kind"] = vocab_value(obj.get("kind"), KIND_VOCAB)
        if name == "evaluate":
            out["n_total"] = _int_or_none(obj.get("n_total"))
            out["correct"] = _int_or_none(obj.get("correct"))
            out["accuracy"] = _unit_or_none(obj.get("accuracy"))
            out["macro_f1"] = _unit_or_none(obj.get("macro_f1"))
    elif name == "package":
        out["code"] = vocab_value(obj.get("code"), CODE_VOCAB)
        out["judgment"] = vocab_value(obj.get("judgment"), JUDGMENT_VOCAB)
        out["acceptance_defined"] = _bool_or_none(obj.get("acceptance_defined"))
        out["capacity"] = capacity_summary(obj)
        out["infer_p95"] = p95_summary(obj.get("infer_p95"))
    return out


def sanitize_record(rec: Any, _path: tuple[str, ...] = (), _in_list: bool = False) -> Any:
    """record 全体の最後の関門（二段構えの 2 段目）。個別の要約で漏れてもここで止まる。

    文字列の値は、`STR_KEYS`（`sha256`・`*_sha256` を含む）に載るキーの下か、`LIST_STR_PATHS`
    の位置の list の要素のときだけ通す。集合に無い位置の文字列と、`MAX_STR` を超える文字列は
    `<redacted>`。パス文字を含む文字列も伏せる。非有限の浮動小数は None にする。
    例外は `items.D.direct_libraries` の位置の要素で、`/usr/lib/`・`/System/` で始まり安全な
    文字だけの `MAX_LIB_LEN` 字以内のものに限る（キー名だけでは効かせない）。
    """
    if isinstance(rec, dict):
        out = {}
        for k, v in rec.items():
            ok = isinstance(k, str) and KEY_RE.fullmatch(k) is not None
            key = k if ok and not _has_path_char(k) else REDACTED
            out[key] = sanitize_record(v, (*_path, key), False)
        return out
    if isinstance(rec, list):
        return [sanitize_record(v, _path, True) for v in rec]
    if isinstance(rec, float) and not math.isfinite(rec):
        return None
    if not isinstance(rec, str):
        return rec
    if _in_list:
        allowed = _path in LIST_STR_PATHS
    else:
        key = _path[-1] if _path else ""
        if key == "error_type":
            return rec if rec in ERROR_TYPE_VOCAB else UNEXPECTED
        allowed = key in STR_KEYS or key == "sha256" or key.endswith("_sha256")
    if not allowed or len(rec) > MAX_STR:
        return REDACTED
    if _in_list and _path == LIBRARIES_PATH:
        # ライブラリ名はパス文字の有無によらず、形式・`..`・長さのすべてを満たすものだけ残す
        if LIB_RE.fullmatch(rec) and ".." not in rec and len(rec) <= MAX_LIB_LEN:
            return rec
        return REDACTED
    if _has_path_char(rec):
        return REDACTED
    return rec


# --------------------------------------------------------------------------------------
# 子プロセス（上限時間・出力サイズ上限・中断時の後始末）
# --------------------------------------------------------------------------------------


@dataclass
class RunResult:
    """子プロセスの結果。

    reason は `timeout`・`output_limit`・`spawn_error`・`killed`・`unreaped`（回収が上限時間内に
    終わらなかった）か None。
    """

    exit_code: int | None
    reason: str | None
    out_bytes: int
    err_bytes: int


def _size(path: Path) -> int:
    """ファイルのバイト数。無ければ 0。"""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _kill_group(pid: int) -> bool:
    """プロセスグループへ KILL を送る。リーダーが未回収（reap 前）のときだけ呼ぶこと。

    戻り値は「止める見込みが立ったか」。グループが既に無い（`ESRCH`）は成功扱い、それ以外の
    `OSError` は False（呼び出し側がリーダー単体の KILL に切り替える）。
    """
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return True


def _resolve_exe(name: str) -> str | None:
    """実行ファイルの絶対パスを返す。`/` を含まない名前は PATH から探す。無ければ None。"""
    exe = name if "/" in name else shutil.which(name)
    if exe is None or not os.path.isfile(exe) or not os.access(exe, os.X_OK):
        return None
    return exe


def resolve_tool(name: str) -> str | None:
    """環境採取用コマンドの固定パスを返す（PATH は探さない）。表に無い・実行できなければ None。

    `sys.platform` は呼び出し時に読む（テストが差し替えるため）。無いときの扱いは
    `TOOL_CANDIDATES_DARWIN` のコメントを参照（REQ-38・REQ-39）。
    """
    table = TOOL_CANDIDATES_DARWIN if sys.platform == "darwin" else TOOL_CANDIDATES_OTHER
    for cand in table.get(name, ()):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def probe_env() -> dict[str, str]:
    """環境採取の子（git・sysctl・sw_vers・otool）へ渡す環境（許可リスト方式）。

    PATH は固定値、LC_ALL=C、HOME は親にあるときだけ通す。`GIT_DIR`・`GIT_WORK_TREE` 等
    （git フックの中から起動すると実際に設定される）や `DEVELOPER_DIR`（xcrun の shim の差し替え）を
    列挙して消す拒否リストにせず、一括で届かなくする。HOME を残すのは、利用者のグローバル設定
    （`safe.directory` 等）が効かないと所有者の違うチェックアウトで commit が取れなくなり、
    `worktree_clean` の意味も変わるため（HOME 配下の設定では git ディレクトリの向き先は
    変えられない）。
    """
    env = {"PATH": PROBE_PATH, "LC_ALL": "C"}
    home = os.environ.get("HOME")
    if home is not None:
        env["HOME"] = home
    return env


def run_cmd(
    argv: list[str],
    cwd: Path,
    out_path: Path,
    err_path: Path,
    timeout: int,
    out_cap: int,
    err_cap: int,
    env: dict[str, str] | None = None,
    during: Callable[[int], None] | None = None,
    reap_group: bool = True,
) -> RunResult:
    """子プロセスを独立したプロセスグループで起動し、期限と出力サイズを監視する（REQ-39）。

    `during` は待機の周ごとに子（外側の sh）の pid を渡して呼ぶコールバック（H のジョブ監視用）。
    例外を外へ出さないこと。`reap_group` が偽なら、正常終了後のグループ KILL を行わない
    （`GROUP_WRAPPER_NO_REAP`。期限・中断・出力超過の後始末のグループ KILL は常に行う）。

    stdin は /dev/null、stdout・stderr はファイルへ書く（呼び出し側が読む前に `out_bytes` を
    上限と照らす）。超過・期限切れ・例外・中断（印）のいずれでも、リーダーが未回収の
    うちにグループごと KILL してから回収する。`shell=True` は使わない。

    中断はシグナルハンドラから例外を投げず、ハンドラが立てた印をこの関数が待機の周ごとに見る。
    例外が `Popen` の内部へ割り込むと、`_waitpid_lock` が解放されず `proc.wait()` が終わらなく
    なる・`Popen` の生成の途中なら子が回収されずに残るため。後始末の後で `Interrupted` を投げる。
    全体の上限時間の超過（`OverallTimeout`）も同じ形で、優先順は `Interrupted` ＞ `OverallTimeout`。
    """
    check_interrupt()  # 印があれば子を起動しない
    check_overall()  # 全体の期限が切れていれば子を起動しない
    exe = _resolve_exe(argv[0])
    if exe is None:
        return RunResult(None, "spawn_error", 0, 0)
    rc_path = out_path.with_name(out_path.name + ".rc")
    try:
        rc_path.unlink(missing_ok=True)
    except OSError:
        # パスがディレクトリ・削除不能のとき。例外を外へ出さず、子を起動しない（#359）
        return RunResult(None, "spawn_error", 0, 0)
    wrapper = GROUP_WRAPPER if reap_group else GROUP_WRAPPER_NO_REAP
    full = ["/bin/sh", "-c", wrapper, "sh", str(rc_path), exe, *argv[1:]]
    global _active_pgid, _child_may_remain, _spawning, _overall_timed_out
    reason = None
    spawn_failed = False
    try:
        with (
            _open_write_nofollow(out_path, False) as fo,
            _open_write_nofollow(err_path, False) as fe,
        ):
            _spawning = True
            try:
                proc = subprocess.Popen(  # noqa: S603  引数リストで起動する。argv は固定語彙と検証済みの値のみ
                    full,
                    cwd=str(cwd),
                    stdin=subprocess.DEVNULL,
                    stdout=fo,
                    stderr=fe,
                    env=env,
                    start_new_session=True,
                )
                _active_pgid = proc.pid
            finally:
                _spawning = False
            if _force_pending:
                _force_exit(_active_pgid)  # 起動の窓で受けた 2 回目のシグナル。戻らない
            try:
                deadline = time.monotonic() + timeout
                while proc.poll() is None:
                    if _interrupt_requested:
                        break  # 例外にせず、下の後始末で止める（reason は None のまま）
                    if _overall_deadline is not None and time.monotonic() >= _overall_deadline:
                        _overall_timed_out = True  # 子ごとの期限より先に全体の期限を見る
                        break
                    if time.monotonic() >= deadline:
                        reason = "timeout"
                        break
                    if _size(out_path) > out_cap or _size(err_path) > err_cap:
                        reason = "output_limit"
                        break
                    if during is not None:
                        during(proc.pid)
                    time.sleep(0.02)
            finally:
                # 例外・中断・期限超過で抜けたら、リーダーが未回収（poll が None）のうちに KILL する
                if proc.poll() is None and not _kill_group(proc.pid):
                    # グループへ送れなかったときは孫が残りうる。回収の成否に関わらず残留の可能性を
                    # 記録し、結果は採用しない（fail-closed。REQ-39）。せめてリーダーだけでも止める
                    # pgid は残さない: リーダーが回収されると pid が再利用されうるため、回収済みの
                    # id への再送は無関係なグループを止めうる。未回収のまま残る場合だけ下で保持する
                    _child_may_remain = True
                    reason = REASON_UNREAPED  # timeout 等より優先（次の回を始めさせない）
                    try:
                        proc.kill()
                    except OSError:
                        pass
                try:
                    proc.wait(timeout=REAP_WAIT_LIMIT_SECONDS)
                except subprocess.TimeoutExpired:
                    # 回収を諦める。子が残っている可能性を記録し、結果は失敗側へ倒す（fail-closed）
                    _child_may_remain = True
                    _leftover_procs.append(proc)
                    reason = REASON_UNREAPED
                finally:
                    _active_pgid = None
    except (OSError, ValueError):
        spawn_failed = True
    # 後始末（グループの KILL と回収）が済んだ後でだけ中断を送出する。子が自然に終わった周に
    # 印が立っていた場合も結果は捨てる
    check_interrupt()
    if _overall_timed_out:
        raise OverallTimeout  # 中断が優先。後始末の後でだけ送出する
    if spawn_failed:
        return RunResult(None, "spawn_error", 0, 0)
    ob, eb = _size(out_path), _size(err_path)
    if reason is None and (ob > out_cap or eb > err_cap):
        reason = "output_limit"
    if reason is not None:
        return RunResult(None, reason, ob, eb)
    code = _read_rc(rc_path)
    # rc ファイルが無い・読めない（ラッパーが外から KILL された等）は成功扱いにしない（fail-closed）
    return RunResult(code, None if code is not None else "killed", ob, eb)


def _read_rc(path: Path) -> int | None:
    """rc ファイルを 0〜255 の整数として読む。無い・上限超過・形が違えば None。

    symlink・FIFO・ディレクトリなど通常ファイルでないものは辿らず None（ブロックもしない）。
    """
    text = _read_regular_capped(path, CAP_RC_FILE)
    if text is None or not re.fullmatch(r"[0-9]{1,3}", text):
        return None
    n = int(text)
    return n if n <= 255 else None


def read_capped_ex(path: Path, cap: int) -> tuple[str | None, str | None]:
    """上限以下のときだけ UTF-8 で読む。戻り値は (本文, 失敗理由)。

    理由は `None`（成功）・`output_limit`（サイズ超過）・`output_unreadable`（読めない）。
    上限超過と読み取り失敗を区別して記録するため（#359）。
    """
    try:
        if path.stat().st_size > cap:
            return None, "output_limit"
        return path.read_bytes().decode("utf-8", errors="replace"), None
    except OSError:
        return None, "output_unreadable"


def read_capped(path: Path, cap: int) -> str | None:
    """サイズが上限以下のときだけファイルを UTF-8 で読む。超過・読めない場合は None。"""
    return read_capped_ex(path, cap)[0]


def _read_regular_capped(path: Path, cap: int) -> str | None:
    """通常ファイルだけを、symlink を辿らず・ブロックせず・上限ぶんだけ読む。それ以外は None。"""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > cap:
            return None
        data = os.read(fd, cap + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(data) > cap:
        return None
    return data.decode("utf-8", errors="replace")


def _open_write_nofollow(path: Path, excl: bool) -> Any:
    """symlink を辿らずに通常ファイルを書き込み用に開く（0600）。通常ファイルでなければ OSError。

    `excl` なら新規作成のみ（O_EXCL）、でなければ既存を切り詰める。
    先に置かれた symlink へ書かない（REQ-39・#359）。

    FIFO は読み手が居ないと open が無期限に止まるため、O_NONBLOCK で開く（読み手が居なければ
    ENXIO で失敗する）。通常ファイルと確認してからブロッキングへ戻す。切り詰めは確認後に行う
    （O_TRUNC を open に付けると、ハードリンクで作られた標的を検査前に切り詰めてしまう）。
    リンク数が 1 でない既存ファイル（ハードリンク）は拒否する。
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    if excl:
        flags |= os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError("not a regular file")
        if st.st_nlink != 1:
            raise OSError("file has multiple hard links")
        os.set_blocking(fd, True)
        if not excl:
            os.ftruncate(fd, 0)
        return os.fdopen(fd, "wb")
    except BaseException:
        os.close(fd)
        raise


def write_text_nofollow(path: Path, text: str, excl: bool = False) -> None:
    """UTF-8 のテキストを symlink を辿らずに書く。"""
    with _open_write_nofollow(path, excl) as f:
        f.write(text.encode("utf-8"))


def write_atomic(path: Path, text: str) -> None:
    """同じディレクトリの一時ファイル経由で原子的に置き換える（壊れた JSON を残さない。#359）。

    既存が通常ファイルでなければ（symlink・ディレクトリ・FIFO）失敗する。失敗時は一時ファイルを消し、
    既存の内容は変えない。
    """
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            raise OSError("existing path is not a regular file")
    except FileNotFoundError:
        pass
    tmp = path.with_name("." + path.name + ".tmp." + str(os.getpid()))
    try:
        with _open_write_nofollow(tmp, True) as f:
            f.write(text.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def sha256_file(path: Path, cap: int, *, deadline_check: bool = False) -> str | None:
    """ファイルの sha256。サイズ上限超過・読めない場合は None。

    `deadline_check` が真なら 1 チャンクごとに全体の期限を見て、超過で `OverallTimeout` を送出する
    （入力採取が期限の外で止まらないようにする。REQ-39）。
    """
    try:
        # FIFO・ディレクトリ等は open で固まる・落ちるため、通常ファイルだけを読む
        st = path.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_size > cap:
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                if deadline_check:
                    check_overall()
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------------------
# 実行文脈
# --------------------------------------------------------------------------------------


@dataclass
class Ctx:
    """項目の実行に共通する値。パスは内部でのみ使い、記録へは出さない。"""

    repo: Path
    work: Path
    bin: Path
    make_cmd: str
    cargo_cmd: str
    p95_limit_us: int
    package_limit_bytes: int
    quiet_machine: bool
    repeat: int
    harness: bool
    # A 以外の子へ渡す環境（CARGO_NET_OFFLINE=true・RUSTUP_AUTO_INSTALL=0。REQ-38）
    offline_env: dict[str, str]
    # A の `make ci` へ渡す環境（RUSTUP_AUTO_INSTALL=0 のみ。`--with-ci` の同意が覆う通信は許す）
    ci_env: dict[str, str]
    # 開始時に記録した CLI の sha256（D が、リンクを確認した対象と同一かを照合する）
    cli_sha256: str | None = None
    # FANDHE_EDGE_BIN で CLI を差し替えたか（差し替えなら p95 は参考値に固定する）
    bin_override: bool = False
    # G の探索予算（秒。`--g-budget-seconds`）と I の証拠種別の申告（`--i-device`）
    g_budget_seconds: int = DEFAULT_G_BUDGET_SECONDS
    i_device: str = "cpu"


# rustup プロキシ（~/.cargo/bin/cargo）は `rust-toolchain.toml` の指すツールチェーンが未導入だと
# 自動取得して通信しうる。`CARGO_NET_OFFLINE` は cargo 自身の設定で rustup を止めないため、
# 別に `RUSTUP_AUTO_INSTALL=0` を渡す（REQ-38・#375）。一次情報: rustup 1.29.1 の文言
# 「you may opt out with RUSTUP_AUTO_INSTALL=0」・rust-lang/rustup#4836。`rustup set auto-install
# disable` は settings.toml へ永続書き込みするため使わない。値は固定リテラルで、
# 親の同名の変数は上書きする。
RUSTUP_AUTO_INSTALL_ENV = "RUSTUP_AUTO_INSTALL"
RUSTUP_AUTO_INSTALL_OFF = "0"


def make_offline_env() -> dict[str, str]:
    """A 以外の子へ渡す環境。`CARGO_NET_OFFLINE=true` と `RUSTUP_AUTO_INSTALL=0` を足したコピー。"""
    env = dict(os.environ)
    env["CARGO_NET_OFFLINE"] = "true"
    env[RUSTUP_AUTO_INSTALL_ENV] = RUSTUP_AUTO_INSTALL_OFF
    return env


def make_ci_env() -> dict[str, str]:
    """A の `make ci` へ渡す環境。`RUSTUP_AUTO_INSTALL=0` だけを足す。"""
    env = dict(os.environ)
    env[RUSTUP_AUTO_INSTALL_ENV] = RUSTUP_AUTO_INSTALL_OFF
    return env


def fail_item(reason: str, **extra: Any) -> dict[str, Any]:
    """失敗した項目の記録。`extra` は固定語彙・数値・伏せ処理済みの値だけを渡す。"""
    d: dict[str, Any] = {"status": "failed", "reason": reason}
    d.update(extra)
    return d


def parse_json_object(path: Path, cap: int) -> dict[str, Any] | None:
    """stdout ファイルを単一の JSON オブジェクトとして読む。違えば None。"""
    text = read_capped(path, cap)
    if text is None:
        return None
    try:
        v = _loads(text)
    except ValueError:
        return None
    return v if isinstance(v, dict) and v else None


def error_fields(obj: dict[str, Any] | None) -> dict[str, Any]:
    """失敗時に記録する `code` と、`message` の大きさ・ハッシュ（REQ-21・REQ-33）。

    `code` は文字列で語彙内ならその値、語彙外なら `<unexpected>`、文字列でなければ欄を出さない。
    `message` は本文を記録せず、文字列のときだけ `message_bytes`（UTF-8 のバイト数）と
    `message_sha256`（UTF-8 の sha256）を出す。本文は `<work-dir>` の stdout のファイルに残る。
    """
    if not isinstance(obj, dict):
        return {}
    out: dict[str, Any] = {}
    code = vocab_value(obj.get("code"), CODE_VOCAB)
    if code is not None:
        out["code"] = code
    msg = obj.get("message")
    if isinstance(msg, str):
        raw = msg.encode("utf-8", errors="replace")
        out["message_bytes"] = len(raw)
        out["message_sha256"] = hashlib.sha256(raw).hexdigest()
    return out


# --------------------------------------------------------------------------------------
# CLI 7 工程のパイプライン（B・C 共通）
# --------------------------------------------------------------------------------------


@dataclass
class Facts:
    """作業ディレクトリへコピーした fixture から導く、報告値の照合用の期待値。"""

    option_ids: list[str]
    train_records: int
    eval_records: int
    has_acceptance: bool
    p95_limit_us: int | None
    # 定義の `limits.max_package_bytes`。無ければ None（package の `limit_bytes` は null。REQ-30）
    limit_bytes: int | None = None


def _count_lines(path: Path) -> int | None:
    """空行を除く行数。読めなければ None。"""
    text = read_capped(path, CAP_INPUT_FILE)
    if text is None:
        return None
    return sum(1 for ln in split_lines(text) if ln.strip())


def read_facts(pdir: Path) -> Facts | None:
    """`pdir` の 3 ファイルから期待値を導く。読めなければ None。"""
    text = read_capped(pdir / "definition.json", CAP_INPUT_FILE)
    train, evaluation = _count_lines(pdir / "train.jsonl"), _count_lines(pdir / "evaluation.jsonl")
    if text is None or train is None or evaluation is None:
        return None
    try:
        d = _loads(text)
    except ValueError:
        return None
    options = d.get("options") if isinstance(d, dict) else None
    if not isinstance(options, list):
        return None
    ids = [o.get("id") for o in options if isinstance(o, dict)]
    if not ids or len(ids) != len(options) or not all(isinstance(i, str) for i in ids):
        return None
    limits = d.get("limits")
    p95 = limits.get("max_infer_p95_us") if isinstance(limits, dict) else None
    max_bytes = limits.get("max_package_bytes") if isinstance(limits, dict) else None
    return Facts(
        option_ids=ids,
        train_records=train,
        eval_records=evaluation,
        has_acceptance="acceptance" in d,
        p95_limit_us=p95 if _is_int(p95) else None,
        limit_bytes=max_bytes if _is_int(max_bytes) else None,
    )


def _eq_int(v: Any, expected: int) -> bool:
    """bool を除く整数で、期待値と等しいか（`False == 0`・`0.0 == 0` を通さない）。"""
    return _is_int(v) and v == expected


def check_infer_output(obj: dict[str, Any], facts: Facts) -> bool:
    """infer の判定が定義と整合するか（REQ-21・REQ-33）。

    `predicted_label` が選択肢 ID のどれか、`scores` のキー集合が選択肢 ID と一致し、各値が有限で
    0 以上 1 以下、合計（宣言順の逐次和で実 CLI と同じ）が 1 から `SCORE_SUM_TOLERANCE` 以内、
    `predicted_label` が最大スコアの選択肢（同点は定義の宣言順の先頭。
    crates/core/src/judgment.rs の規則。比較は許容差なし）。
    実 CLI の判定型（`JudgmentResult`）が検証する範囲のうち、定義と入力から照合できる分を見る。
    """
    scores = obj.get("scores")
    label = obj.get("predicted_label")
    if not isinstance(scores, dict) or not isinstance(label, str):
        return False
    if label not in facts.option_ids:
        return False
    if set(scores) != set(facts.option_ids) or len(scores) != len(facts.option_ids):
        return False
    if not all(_unit_number(v) for v in scores.values()):
        return False
    total = 0.0
    for oid in facts.option_ids:
        total += float(scores[oid])
    if abs(total - 1.0) > SCORE_SUM_TOLERANCE:
        return False
    best = facts.option_ids[0]
    for oid in facts.option_ids:
        if scores[oid] > scores[best]:
            best = oid
    return label == best


def check_stage_report(
    name: str,
    obj: dict[str, Any],
    facts: Facts,
    selected: int | None,
    kind_before: str | None = None,
) -> bool:
    """exit 0 の工程 JSON が、指定値・入力から分かる値と整合するか（REQ-21・REQ-33）。

    欄名・形は crates/core/src/stage_report.rs に合わせる。CLI の計算は再実装しないが、
    `accuracy` は `correct / n_total` との一致だけ照合する。`==` で比べる報告値は真偽値を除く
    整数であることを確かめる。`kind_before` は直前の `kind` を出す工程（`select` なら `train`、
    `evaluate` なら `select`）の `kind` で、`select`・`evaluate` はこれと同じでなければならない。
    """
    if name == "register":
        return _eq_int(obj.get("options"), len(facts.option_ids)) and (
            obj.get("evaluation_defined") is (facts.eval_records > 0)
        )
    if name == "inspect":
        split = obj.get("split")
        vr = obj.get("valid_records")
        if not isinstance(split, dict) or not _nonneg_int(vr) or vr != facts.train_records:
            return False
        parts = [split.get(k) for k in ("train", "validation", "test")]
        return all(_nonneg_int(x) for x in parts) and sum(parts) == vr
    if name in ("train", "select", "evaluate"):
        kind = obj.get("kind")
        if not isinstance(kind, str) or kind not in KIND_VOCAB:
            return False
        if name != "train" and kind != kind_before:
            return False
    if name in ("train", "select"):
        return _eq_int(obj.get("candidate"), TRAIN_CANDIDATE)
    if name == "evaluate":
        n_total, correct = obj.get("n_total"), obj.get("correct")
        if (
            selected is None
            or not _eq_int(obj.get("candidate"), selected)
            or not _eq_int(n_total, facts.eval_records)
            or n_total <= 0
            or not _nonneg_int(correct)
            or correct > n_total
        ):
            return False
        accuracy = obj.get("accuracy")
        if not _is_finite_number(accuracy) or abs(accuracy - correct / n_total) > SCORE_TOLERANCE:
            return False
        # macro_f1 は分母 0 の指標で null になりうる（REQ-24）。欄自体は実 CLI が必ず出す
        return "macro_f1" in obj and (obj["macro_f1"] is None or _unit_number(obj["macro_f1"]))
    if name == "package":
        # 実 CLI は judgment・infer_p95 を値が null のときも必ず出す（欠落は偽の報告）
        if "judgment" not in obj or "infer_p95" not in obj:
            return False
        # 合否基準の無い定義では judgment:null・acceptance_defined:false
        if not facts.has_acceptance:
            return obj["judgment"] is None and obj.get("acceptance_defined") is False
        return obj.get("acceptance_defined") is True and obj["judgment"] == "pass"
    return True


def capacity_sum_matches(cap: dict[str, Any]) -> bool:
    """`capacity_summary` の 5 項目の `bytes` の合計が `total_bytes` と一致するか（REQ-30）。"""
    return sum(c["bytes"] for c in cap["components"].values()) == cap["total_bytes"]


def check_package_metrics(
    obj: dict[str, Any], rc: int, facts: Facts, check_sum: bool = True
) -> bool:
    """package（exit 0・20）の `capacity`・`infer_p95` の整合（REQ-30・REQ-31）。

    capacity は要約でき、`limit_bytes` が定義から導く期待値（`facts.limit_bytes`）と一致し、
    `exceeded == (limit_bytes is not None and total_bytes > limit_bytes)`（上限なしは常に false）、
    `guideline_bytes` が目安（40000000）で `over_guideline == (total_bytes > guideline_bytes)`
    （目安超過は警告のみで exit 0 を妨げない）、exit 0 なら上限超過なし。`check_sum` なら
    5 項目の合計が `total_bytes` と一致すること（B は項目側で `capacity_sum_mismatch` として
    判定するため外す）。
    `infer_p95` は定義に `max_infer_p95_us` があるときだけ非 null で、`limit_us` が定義の値と
    一致し、`exceeded == (p95_us > limit_us)`、exit 0 なら超過なし。exit 20 は `code` が
    `limit_exceeded` で、容量か p95 のどちらかが超過していること。
    """
    cap = capacity_summary(obj)
    if cap is None:
        return False
    lim, total = cap["limit_bytes"], cap["total_bytes"]
    if cap["exceeded"] != (lim is not None and total > lim):
        return False
    if cap["guideline_bytes"] != REFERENCE_CAPACITY_BYTES or cap["over_guideline"] != (
        total > REFERENCE_CAPACITY_BYTES
    ):
        return False
    if cap["limit_bytes"] != facts.limit_bytes or (check_sum and not capacity_sum_matches(cap)):
        return False
    if "infer_p95" not in obj:
        return False
    raw = obj["infer_p95"]
    p95 = p95_summary(raw)
    if facts.p95_limit_us is None:
        if raw is not None:
            return False
    elif (
        p95 is None
        or p95["limit_us"] != facts.p95_limit_us
        or p95["exceeded"] != (p95["p95_us"] > p95["limit_us"])
    ):
        return False
    any_exceeded = cap["exceeded"] or (p95 is not None and p95["exceeded"])
    if rc == 0:
        return not any_exceeded
    return obj.get("code") == "limit_exceeded" and any_exceeded


def _infer_envelope_ok(obj: dict[str, Any], rc: int) -> bool:
    """infer の出力の `status` が終了コード `rc` に対応し、`step` 欄なし・文字列の `id` か。

    B の単発・E で共通（REQ-21・REQ-22）。E のバッチ行は exit 0 のため、行の `status` から引いた
    終了コードを `rc` に渡す（`INFER_EXIT_STATUS` の値のどれかであれば合格）。
    """
    status = INFER_EXIT_STATUS.get(rc)
    return (
        status is not None
        and obj.get("status") == status
        and "step" not in obj
        and isinstance(obj.get("id"), str)
    )


def _step_check(
    name: str,
    obj: dict[str, Any],
    allowed: set[int],
    rc: int,
    facts: Facts,
    selected: int | None,
    kind_before: str | None = None,
    check_sum: bool = True,
) -> bool:
    """工程の stdout JSON が契約と整合するか（exit 0 は報告値まで照合、非 0 は許容と step）。"""
    if name == "infer":
        # B の単発 infer は `--id` を付けないので、id は既定値（`DEFAULT_TEXT_ID`）でなければ不合格
        return (
            _infer_envelope_ok(obj, rc)
            and obj.get("id") == DEFAULT_TEXT_ID
            and check_infer_output(obj, facts)
        )
    if obj.get("step") != name:
        return False
    if rc != 0:
        return rc in allowed and (
            name != "package" or check_package_metrics(obj, rc, facts, check_sum)
        )
    if obj.get("status") != "ok" or not check_stage_report(name, obj, facts, selected, kind_before):
        return False
    return name != "package" or check_package_metrics(obj, rc, facts, check_sum)


def run_pipeline(
    ctx: Ctx,
    pdir: Path,
    sample_text: str | None,
    package_allowed: set[int],
    check_sum: bool = True,
    raw: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """`pdir` で register〜package（と、あれば infer）を実行する。

    `pdir` に definition.json・train.jsonl・evaluation.jsonl が置かれている前提で、そこをカレントに
    する（経路の閉じ込め。REQ-39）。戻り値は (工程記録, package の JSON, 失敗記録)。
    失敗記録が None でなければ以降の工程は実行していない。`kind` は train → select → evaluate の
    一貫性を照合する。`check_sum` は package の容量内訳の合計の照合を工程側で行うか（B は項目側）。
    `raw` を渡すと、検査に通った工程の JSON（exit 0・20）を工程名で入れる（B の追加検査用）。
    """
    facts = read_facts(pdir)
    if facts is None:
        return [], None, fail_item("input_unreadable")
    logs = pdir / "steps"
    logs.mkdir(exist_ok=True)
    steps: list[dict[str, Any]] = []
    package_obj: dict[str, Any] | None = None
    selected: int | None = None
    kind_before: str | None = None

    def go(
        name: str, argv: list[str], command: str, allowed: set[int]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """1 工程を実行して記録する。戻り値は (stdout の JSON, 失敗記録)。"""
        nonlocal package_obj
        n = len(steps) + 1
        so = logs / f"{n:02d}-{name}.stdout"
        se = logs / f"{n:02d}-{name}.stderr"
        r = run_cmd(
            [str(ctx.bin), *argv],
            pdir,
            so,
            se,
            TIMEOUT_CLI_STEP,
            CAP_CLI_STDOUT,
            CAP_CLI_STDERR,
            ctx.offline_env,
        )
        entry: dict[str, Any] = {"step": name, "command": command, "exit_code": r.exit_code}
        entry["stderr_bytes"] = r.err_bytes
        if r.reason is not None:
            steps.append(entry)
            return None, fail_item(r.reason, step=name, exit_code=r.exit_code)
        obj = parse_json_object(so, CAP_CLI_STDOUT)
        if obj is None:
            steps.append(entry)
            rc0 = r.exit_code if r.exit_code is not None else EXIT_RUNTIME_ERROR
            if rc0 != 0 and rc0 not in allowed:
                # 非 0 終了かつ JSON でない: 終了コードが許容外であることも理由へ出す（#361）
                return None, fail_item(
                    "invalid_json",
                    step=name,
                    exit_code=r.exit_code,
                    exit_code_unexpected=True,
                )
            return None, fail_item("invalid_json", step=name, exit_code=r.exit_code)
        rc = r.exit_code if r.exit_code is not None else EXIT_RUNTIME_ERROR
        entry["summary"] = summarize_step(name, obj, facts.option_ids)
        steps.append(entry)
        if rc != 0 and rc not in allowed:
            err = error_fields(obj)
            return None, fail_item("unexpected_exit_code", step=name, exit_code=rc, **err)
        if not _step_check(name, obj, allowed, rc, facts, selected, kind_before, check_sum):
            return None, fail_item(
                "unexpected_output", step=name, exit_code=rc, **error_fields(obj)
            )
        if name == "package":
            package_obj = obj
        if raw is not None:
            raw[name] = obj
        return obj, None

    obj, failure = go(
        "register",
        ["register", "--definition", "definition.json", "--project-dir", "project"],
        "register --definition definition.json --project-dir project",
        set(),
    )
    if failure:
        return steps, package_obj, failure
    obj, failure = go(
        "inspect", ["inspect", "--project-dir", "project"], "inspect --project-dir project", set()
    )
    if failure:
        return steps, package_obj, failure
    obj, failure = go(
        "train",
        ["train", "--project-dir", "project", "--candidate", str(TRAIN_CANDIDATE)],
        f"train --project-dir project --candidate {TRAIN_CANDIDATE}",
        set(),
    )
    if failure:
        return steps, package_obj, failure
    # 直前の `kind`（検証済み）。select は train と、evaluate は select と同じでなければならない
    kind_before = obj.get("kind") if isinstance(obj, dict) else None
    obj, failure = go(
        "select", ["select", "--project-dir", "project"], "select --project-dir project", set()
    )
    if failure:
        return steps, package_obj, failure
    kind_before = obj.get("kind") if isinstance(obj, dict) else None
    c = obj.get("candidate") if isinstance(obj, dict) else None
    if not _is_int(c) or c < 0:
        return steps, package_obj, fail_item("missing_field", step="select", exit_code=0)
    selected = c
    obj, failure = go(
        "evaluate",
        ["evaluate", "--project-dir", "project", "--candidate", str(c)],
        f"evaluate --project-dir project --candidate {c}",
        set(),
    )
    if failure:
        return steps, package_obj, failure
    obj, failure = go(
        "package",
        ["package", "--project-dir", "project"],
        "package --project-dir project",
        package_allowed,
    )
    if failure:
        return steps, package_obj, failure
    if sample_text is not None:
        # 値が `-` で始まっても CLI は値を無条件に消費する（crates/cli/src/args.rs）。
        obj, failure = go(
            "infer",
            ["infer", "--package", "project/package", "--text", sample_text],
            INFER_COMMAND_DISPLAY,
            set(INFER_EXIT_STATUS) - {0},
        )
        if failure:
            return steps, package_obj, failure
    return steps, package_obj, None


def capacity_summary(pkg: dict[str, Any] | None) -> dict[str, Any] | None:
    """package の `capacity` を固定構造へ写す。欠落・負・型違いは None（REQ-30）。"""
    cap = pkg.get("capacity") if isinstance(pkg, dict) else None
    if not isinstance(cap, dict):
        return None
    comps = cap.get("components")
    if not isinstance(comps, dict) or set(comps) != set(CAPACITY_COMPONENTS):
        return None
    out_comps: dict[str, Any] = {}
    for name in CAPACITY_COMPONENTS:
        c = comps[name]
        if not isinstance(c, dict):
            return None
        b, fc = c.get("bytes"), c.get("file_count")
        if not _nonneg_int(b) or not _nonneg_int(fc):
            return None
        out_comps[name] = {"bytes": b, "file_count": fc}
    total, limit, exceeded = cap.get("total_bytes"), cap.get("limit_bytes"), cap.get("exceeded")
    guide, over = cap.get("guideline_bytes"), cap.get("over_guideline")
    # `limit_bytes` は利用者設定の上限で、未設定なら明示的な null（REQ-30）。キーの欠落は受理しない
    if "limit_bytes" not in cap:
        return None
    if not _nonneg_int(total) or not (limit is None or _nonneg_int(limit)):
        return None
    if not isinstance(exceeded, bool) or not _nonneg_int(guide) or not isinstance(over, bool):
        return None
    return {
        "total_bytes": total,
        "limit_bytes": limit,
        "exceeded": exceeded,
        "guideline_bytes": guide,
        "over_guideline": over,
        "components": out_comps,
    }


def stage_inputs(ctx: Ctx, dest: Path, extra_limits: dict[str, int] | None) -> bool:
    """fixtures の 3 ファイルを `dest` へ複製する。`extra_limits` があれば定義へ足す。"""
    src = ctx.repo / "fixtures" / "sandbox_run_eval"
    dest.mkdir(parents=True, exist_ok=False)
    for name in FIXTURE_FILES:
        if name == "definition.json" and extra_limits is not None:
            text = read_capped(src / name, CAP_INPUT_FILE)
            if text is None:
                return False
            d = json.loads(text)
            d["limits"] = extra_limits
            write_text_nofollow(dest / name, json.dumps(d))
        else:
            shutil.copyfile(src / name, dest / name)
    return True


# --------------------------------------------------------------------------------------
# 各項目
# --------------------------------------------------------------------------------------

# pytest の要約行。`=` で囲まれた形（既定）と `-q` の裸の形（`126 passed in 2.45s`）の両方を受ける。
# cargo の `test result: ...` 行は先頭が `test result:` のため、どちらにも一致しない
_PYTEST_COUNTS = r"\d+ (?:passed|failed|skipped|errors?|xfailed|xpassed|deselected|warnings?)"
_PYTEST_BODY = (
    rf"{_PYTEST_COUNTS}(?:, {_PYTEST_COUNTS})*(?:, \d+ rerun)? in [\d.]+s(?: \(\d+:\d+:\d+\))?"
)
PYTEST_LINE_RE = re.compile(rf"^(?:=+ {_PYTEST_BODY} =+|{_PYTEST_BODY})$")
RUST_RESULT_RE = re.compile(r"^test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored;")


def parse_make_ci_log(text: str) -> dict[str, Any]:
    """`make ci` の stdout の `skip:` 行数・Rust の `test result:` 合計・pytest の最終行の件数。"""
    skip_lines = 0
    rust = {"passed": 0, "failed": 0, "ignored": 0}
    pytest_line = None
    for raw in split_lines(text):
        line = raw.strip()
        if raw.startswith("skip:"):
            skip_lines += 1
        m = RUST_RESULT_RE.match(line)
        if m:
            rust["passed"] += int(m.group(1))
            rust["failed"] += int(m.group(2))
            rust["ignored"] += int(m.group(3))
        elif PYTEST_LINE_RE.match(line):
            pytest_line = line
    pytest: dict[str, int] | None = None
    if pytest_line is not None:
        pytest = {"passed": 0, "skipped": 0, "failed": 0}
        for n, word in re.findall(r"(\d+) (passed|skipped|failed|errors?)", pytest_line):
            pytest["failed" if word.startswith("error") else word] += int(n)
    return {"skip_lines": skip_lines, "rust_tests": rust, "pytest": pytest}


def judge_make_ci(counts: dict[str, Any]) -> str | None:
    """A の合格条件。満たさなければ失敗の reason、満たせば None（skip を検証済みと扱わない）。"""
    rust, pytest = counts["rust_tests"], counts["pytest"]
    if counts["skip_lines"] > 0:
        return "skipped"
    if rust["failed"] > 0 or (pytest is not None and pytest["failed"] > 0):
        return "test_failures"
    if rust["passed"] < 1 or pytest is None or pytest["passed"] < 1:
        return "no_test_results"
    return None


def item_a(ctx: Ctx) -> dict[str, Any]:
    """A: `make ci`。通信しうるため `--with-ci` のときだけ呼ばれる。

    `CARGO_NET_OFFLINE` は渡さず、ツールチェーンの自動取得だけ止める（#375）。
    """
    d = ctx.work / "A"
    d.mkdir()
    r = run_cmd(
        [ctx.make_cmd, "ci"],
        ctx.repo,
        d / "make-ci.log",
        d / "make-ci.err",
        TIMEOUT_MAKE_CI,
        CAP_LOG_STDOUT,
        CAP_LOG_STDERR,
        ctx.ci_env,
    )
    if r.reason is not None:
        return fail_item(r.reason, exit_code=r.exit_code)
    text, why = read_capped_ex(d / "make-ci.log", CAP_LOG_STDOUT)
    if text is None:
        return fail_item(why or "output_unreadable", exit_code=r.exit_code)
    counts = parse_make_ci_log(text)
    counts["stderr_bytes"] = r.err_bytes
    if r.exit_code != 0:
        return fail_item("unexpected_exit_code", exit_code=r.exit_code, **counts)
    reason = judge_make_ci(counts)
    if reason is not None:
        return fail_item(reason, exit_code=0, **counts)
    return dict({"status": "ok", "exit_code": 0}, **counts)


def item_b(ctx: Ctx) -> tuple[dict[str, Any], bool]:
    """B: 7 工程を順に実行し、`package` の容量内訳と `package/` の各ファイルを記録する。"""
    bdir = ctx.work / "B"
    if not stage_inputs(ctx, bdir, None):
        return fail_item("input_unreadable"), False
    # 内訳の合計の照合は項目側で行い、不一致を `capacity_sum_mismatch` として記録する
    raw: dict[str, dict[str, Any]] = {}
    steps, pkg, failure = run_pipeline(
        ctx, bdir, "sandbox check 0123456789", {0}, check_sum=False, raw=raw
    )
    if failure:
        return dict(failure, steps=steps), False
    cap = capacity_summary(pkg)
    if cap is None:
        return fail_item("missing_field", step="package", steps=steps), False
    matches = capacity_sum_matches(cap)
    pdir = bdir / "project" / "package"
    regular = package_entries_regular(pdir)
    if regular is None:
        return fail_item("package_unreadable", steps=steps), False
    # 通常ファイル以外（symlink・ディレクトリ・FIFO 等）は計測対象外のまま見逃さず失敗にする
    if not regular:
        return fail_item("package_entry_not_regular", steps=steps), False
    files = package_files(pdir)
    if files is None:
        return fail_item("package_unreadable", steps=steps), False
    # 公開された package/ の実ファイルの合計・件数が、CLI の報告した total_bytes と
    # `file_count` の合計に一致すること（REQ-30。計測対象は公開される集合そのもの）
    if matches and (
        sum(f["bytes"] for f in files) != cap["total_bytes"]
        or len(files) != capacity_file_count(cap)
    ):
        return fail_item("unexpected_output", step="package", steps=steps), False
    # #469 の CLI 結線で増えた出力の整合（校正・保留・診断・有意性・版管理台帳・校正の束縛）
    n_total = raw.get("evaluate", {}).get("n_total")
    reason, extras = check_b_extras(raw, cap, bdir / "project", n_total)
    if reason is not None:
        return fail_item(reason, steps=steps), False
    rec = {
        "status": "ok" if matches else "failed",
        "steps": steps,
        "capacity": cap,
        "capacity_sum_matches_total": matches,
        "package_files": files,
        "contract_checks": extras,
    }
    if not matches:
        rec["reason"] = "capacity_sum_mismatch"
    return rec, matches


def _hex64(v: Any) -> bool:
    """sha256（小文字 hex 64 桁）の文字列か。"""
    return isinstance(v, str) and SHA256_RE.fullmatch(v) is not None


def _json_file(path: Path, cap: int) -> Any:
    """通常ファイルを上限つきで読んで JSON にする。読めない・JSON でなければ None。"""
    text = _read_regular_capped(path, cap)
    if text is None:
        return None
    try:
        return _loads(text)
    except ValueError:
        return None


def check_evaluate_extras(obj: dict[str, Any], n_total: Any) -> tuple[str | None, dict[str, Any]]:
    """`evaluate` の `calibration`・`abstention`・`diagnostics` の整合（REQ-22・REQ-27・REQ-29）。

    欄名・形は crates/core/src/stage_report.rs（`EvaluateCalibration`・`EvaluateAbstention`・
    `EvaluateDiagnostics`）。`abstention` は `answered + abstained == n_total`（`out_of_scope` は
    `answered` の内数。stage_report.rs の `EvaluateAbstention` の doc）で、`coverage` は 0〜1。
    評価データありの B では `calibration` が null にならない（#477）。戻り値は (失敗理由, 要約)。
    """
    cal = obj.get("calibration")
    if (
        not isinstance(cal, dict)
        or not _is_finite_number(cal.get("temperature"))
        or cal["temperature"] <= 0
        or not isinstance(cal.get("adopted"), bool)
        or not _unit_number(cal.get("threshold"))
        or not _nonneg_int(cal.get("n_validation"))
        or cal["n_validation"] <= 0
        or not _unit_number(cal.get("validation_coverage"))
    ):
        return "calibration_invalid", {}
    ab = obj.get("abstention")
    counts = ("answered", "abstained", "out_of_scope", "correct_answered")
    if (
        not isinstance(ab, dict)
        or not all(_nonneg_int(ab.get(k)) for k in counts)
        or not _unit_number(ab.get("coverage"))
        or "adopted_error" not in ab
        or not (ab["adopted_error"] is None or _unit_number(ab["adopted_error"]))
        or not _unit_number(ab.get("unconditional_error"))
        or ab["answered"] + ab["abstained"] != n_total
        or ab["out_of_scope"] > ab["answered"]
        or ab["correct_answered"] > ab["answered"]
    ):
        return "abstention_invalid", {}
    dg = obj.get("diagnostics")
    if (
        not isinstance(dg, dict)
        or not isinstance(dg.get("train"), dict)
        or not isinstance(dg.get("eval"), dict)
        or not isinstance(dg.get("confusable_pairs"), list)
        or not isinstance(dg.get("limitations"), list)
        or not isinstance(dg.get("data_volume"), dict)
        or not _eq_int(dg["eval"].get("n_rows"), n_total)
    ):
        return "diagnostics_invalid", {}
    return None, {
        "calibration": {"n_validation": cal["n_validation"], "adopted": cal["adopted"]},
        "abstention": {k: ab[k] for k in ("answered", "abstained", "out_of_scope", "coverage")},
        "diagnostics_present": True,
    }


def check_select_significance(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    """`select` の `significance`（REQ-25・#481）。キーは必ず出る（定義に `baseline_comparison` が
    無ければ null）。dict なら `verdict` が語彙内（`undeterminable` も正常）。
    戻り値は (失敗理由, verdict)。
    """
    if "significance" not in obj:
        return "significance_invalid", None
    sig = obj["significance"]
    if sig is None:
        return None, None
    verdict = sig.get("verdict") if isinstance(sig, dict) else None
    if not isinstance(verdict, str) or verdict not in SIGNIFICANCE_VOCAB:
        return "significance_invalid", None
    return None, verdict


def check_version_report(v: Any, previous_expected: str | None) -> bool:
    """`package` の `version`（`{"id":"v<n>","previous":null|"v<m>"}`。REQ-39・#491）。"""
    if not isinstance(v, dict) or "previous" not in v:
        return False
    vid, prev = v.get("id"), v["previous"]
    if not isinstance(vid, str) or VERSION_ID_RE.fullmatch(vid) is None:
        return False
    if prev is not None and (not isinstance(prev, str) or VERSION_ID_RE.fullmatch(prev) is None):
        return False
    return prev == previous_expected


def version_number(v: Any) -> int | None:
    """`v<n>` の n。形が違えば None（記録には版 ID の文字列でなく番号を出す）。"""
    if isinstance(v, str) and VERSION_ID_RE.fullmatch(v):
        return int(v[1:])
    return None


def check_version_ledger(path: Path, version_id: str) -> bool:
    """`version_ledger.json` が通常ファイルで、今回の版の model・data・experiment の 3 件を持つか。

    形は crates/core/src/version_ledger_record.rs（`schema_version`・`entries[]` の
    `kind`・`id`・`sha256`）。ファイルの改変検出は範囲外（#491 の限界）。
    """
    led = _json_file(path, CAP_VERSION_LEDGER)
    if not isinstance(led, dict) or not _eq_int(led.get("schema_version"), 1):
        return False
    entries = led.get("entries")
    if not isinstance(entries, list):
        return False
    mine = [e for e in entries if isinstance(e, dict) and e.get("id") == version_id]
    kinds = sorted(e.get("kind") for e in mine if isinstance(e.get("kind"), str))
    return kinds == ["data", "experiment", "model"] and all(_hex64(e.get("sha256")) for e in mine)


def check_calibration_binding(pdir: Path, cap: dict[str, Any]) -> tuple[str | None, bool]:
    """`package/artifact.json` の `calibration_sha256` が `calibration.json` の実 sha256 と一致し、
    容量内訳の `calibration` 枠が `calibration.json` の 1 件・実サイズであること
    （REQ-39・REQ-30・#497）。
    """
    art = _json_file(pdir / "artifact.json", CAP_INPUT_FILE)
    declared = art.get("calibration_sha256") if isinstance(art, dict) else None
    actual = sha256_file(pdir / "calibration.json", CAP_PACKAGE_FILE)
    if not _hex64(declared) or actual is None or declared != actual:
        return "calibration_binding_mismatch", False
    try:
        size = (pdir / "calibration.json").stat().st_size
    except OSError:
        return "calibration_binding_mismatch", False
    if cap["components"]["calibration"] != {"bytes": size, "file_count": 1}:
        return "calibration_capacity_mismatch", False
    return None, True


def check_b_extras(
    raw: dict[str, dict[str, Any]], cap: dict[str, Any], project: Path, n_total: Any
) -> tuple[str | None, dict[str, Any]]:
    """B の追加検査（#469 の CLI 結線で増えた出力。REQ-22・REQ-25・REQ-29・REQ-30・REQ-39）。

    `evaluate` の校正・保留・診断、`select` の有意性、`package` の版（B は `--previous-project-dir`
    なしのため `v1`・previous は null）・`version_ledger.json`・校正の束縛。戻り値は
    (失敗理由, 記録へ出す要約)。
    """
    ev, sel, pkg = raw.get("evaluate"), raw.get("select"), raw.get("package")
    if not isinstance(ev, dict) or not isinstance(sel, dict) or not isinstance(pkg, dict):
        return "missing_field", {}
    reason, extras = check_evaluate_extras(ev, n_total)
    if reason is not None:
        return reason, {}
    reason, verdict = check_select_significance(sel)
    if reason is not None:
        return reason, {}
    version = pkg.get("version")
    if not check_version_report(version, None) or version["id"] != "v1":
        return "version_invalid", {}
    if not check_version_ledger(project / "version_ledger.json", version["id"]):
        return "version_ledger_invalid", {}
    reason, bound = check_calibration_binding(project / "package", cap)
    if reason is not None:
        return reason, {}
    extras["significance_verdict"] = verdict
    extras["version_number"] = version_number(version["id"])
    extras["version_ledger_present"] = True
    extras["calibration_sha256_matches"] = bound
    return None, extras


def capacity_file_count(cap: dict[str, Any]) -> int:
    """`capacity_summary` の 5 項目の `file_count` の合計（REQ-30）。"""
    return sum(c["file_count"] for c in cap["components"].values())


def package_entries_regular(pdir: Path) -> bool | None:
    """`package/` 直下がすべて通常ファイルか。読めなければ None、通常ファイル以外があれば False。"""
    try:
        return all(e.is_file(follow_symlinks=False) for e in os.scandir(pdir))
    except OSError:
        return None


def package_dir_stats(pdir: Path) -> tuple[int, int] | None:
    """`package/` 直下の (通常ファイル数, バイト数の合計)。読めない・通常以外があれば None。"""
    try:
        entries = list(os.scandir(pdir))
        if not all(e.is_file(follow_symlinks=False) for e in entries):
            return None
        return len(entries), sum(e.stat(follow_symlinks=False).st_size for e in entries)
    except OSError:
        return None


def staging_present(project_dir: Path) -> bool:
    """`package.staging` が残っているか（symlink でも真。辿らない）。"""
    return os.path.lexists(project_dir / PACKAGE_STAGING_DIR)


def package_files(pdir: Path) -> list[dict[str, Any]] | None:
    """`package/` 直下の通常ファイルの名前・バイト数・sha256。読めない・通常以外があれば None。

    名前が `PACKAGE_FILE_NAME_RE` に一致しなければ `<unrecognized>` を記録する（利用者が決める
    文字列を記録へ出さない）。
    """
    out: list[dict[str, Any]] = []
    try:
        entries = sorted(os.scandir(pdir), key=lambda e: e.name)
    except OSError:
        return None
    for e in entries:
        if not e.is_file(follow_symlinks=False):
            return None  # 通常ファイル以外は黙って飛ばさない
        digest = sha256_file(Path(e.path), CAP_PACKAGE_FILE)
        if digest is None:
            return None
        name = e.name if PACKAGE_FILE_NAME_RE.fullmatch(e.name) else UNRECOGNIZED
        out.append({"name": name, "bytes": e.stat().st_size, "sha256": digest})
    return out or None


def classify_p95(ctx: Ctx) -> str:
    """p95 の区分。静かな状態の申告があり、代役・CLI 差し替えが無いときだけ real_machine。"""
    if ctx.quiet_machine and not ctx.harness and not ctx.bin_override:
        return "real_machine"
    return "reference_only"


def judge_p95(rc: int, pkg: Any, p95: Any, cap: dict[str, Any] | None, limit_us: int) -> str | None:
    """C-1 の判定。満たさなければ reason、満たせば None。

    CLI の規則は `p95_us > limit_us` で超過（crates/cli/src/stages/package.rs）。exit 20 は p95 超過
    のときだけで、容量超過など p95 以外の理由の exit 20 は ok にしない。
    `p95_us`・`limit_us` は 0 以上の整数・`exceeded` は真偽値でなければならず、型違い・欠落は
    `missing_field`、負数は `unexpected_output`。
    C-1 の capacity が欠落・要約不能・超過のいずれでも `unexpected_output`。
    exit 20 は `code == "limit_exceeded"`、exit 0 は `status == "ok"` の JSON のときだけ合格。
    """
    if (
        not isinstance(p95, dict)
        or not _is_int(p95.get("p95_us"))
        or not _is_int(p95.get("limit_us"))
        or not isinstance(p95.get("exceeded"), bool)
    ):
        return "missing_field"
    if p95["p95_us"] < 0 or p95["limit_us"] < 0:
        return "unexpected_output"
    exceeded = p95["exceeded"]
    if exceeded != (p95["p95_us"] > p95["limit_us"]) or exceeded != (rc == 20):
        return "unexpected_output"
    if not isinstance(pkg, dict):
        return "unexpected_output"
    if rc == 20 and pkg.get("code") != "limit_exceeded":
        return "unexpected_output"
    if rc == 0 and pkg.get("status") != "ok":
        return "unexpected_output"
    if p95["limit_us"] != limit_us or cap is None or cap["exceeded"]:
        return "unexpected_output"
    return None


def judge_capacity_limit(
    rc: int, code: Any, cap: dict[str, Any] | None, limit_bytes: int, published: bool
) -> str | None:
    """C-2 の判定。満たさなければ reason、満たせば None。

    CLI の規則は `total_bytes > limit_bytes` で超過（上限ちょうどは超過でない。
    crates/runtime/src/package_outcome.rs）。報告された上限が指定値と一致しない場合は
    上限が CLI に伝わっていないので `unexpected_output`、それ以外の不成立は
    `capacity_limit_not_enforced`。`code` や超過の不整合は、先に工程の検査
    （`check_package_metrics`）が `unexpected_output` で止めるため、この関数へ届くのは
    「exit 0 で報告値に矛盾が無い」か「exit 20 なのに `package/` が公開されている」場合だけ。
    """
    if cap is None or not _is_int(cap["limit_bytes"]) or cap["limit_bytes"] != limit_bytes:
        return "unexpected_output"
    if (
        rc != 20
        or code != "limit_exceeded"
        or cap["exceeded"] is not True
        or not cap["total_bytes"] > cap["limit_bytes"]
        or published
    ):
        return "capacity_limit_not_enforced"
    return None


def item_c(ctx: Ctx) -> dict[str, Any]:
    """C: 上限つきの定義で package を実行する（C-1 は p95、C-2 は容量上限で exit 20 が期待）。"""
    c1 = ctx.work / "C1"
    if not stage_inputs(ctx, c1, {"max_infer_p95_us": ctx.p95_limit_us}):
        return fail_item("input_unreadable")
    steps1, pkg1, failure = run_pipeline(ctx, c1, None, {0, 20})
    if failure:
        return dict(failure, case="C-1", steps=steps1)
    rc1 = next(s["exit_code"] for s in steps1 if s["step"] == "package")
    p95 = pkg1.get("infer_p95") if isinstance(pkg1, dict) else None
    reason1 = judge_p95(rc1, pkg1, p95, capacity_summary(pkg1), ctx.p95_limit_us)
    # exit 20 なら package/ は作られず、exit 0 なら公開される（crates/cli/src/stages/package.rs）
    published1 = (c1 / "project" / "package").exists()
    if reason1 is None and published1 != (rc1 == 0):
        reason1 = "unexpected_output"
    if reason1 is None and rc1 == 0:
        # 公開された package/ が通常ファイルだけで、実ファイルの合計・件数が total_bytes と
        # `file_count` の合計に一致すること（記録へは足さない）
        stats, cap1 = package_dir_stats(c1 / "project" / "package"), capacity_summary(pkg1)
        if (
            stats is None
            or cap1 is None
            or stats != (capacity_file_count(cap1), cap1["total_bytes"])
        ):
            reason1 = "unexpected_output"
    # exit 0・20 のどちらでも、組み立て先は残らない（crates/cli/src/stages/package.rs）
    staging1 = staging_present(c1 / "project")
    if reason1 is None and staging1:
        reason1 = "staging_left"
    if reason1 is not None:
        return fail_item(reason1, case="C-1", step="package", exit_code=rc1)
    p95_rec = {
        "p95_us": p95["p95_us"],
        "limit_us": p95["limit_us"],
        "exceeded": p95["exceeded"],
        # 静かな状態という人の申告があるときだけ real_machine（`--quiet-machine`）
        "classification": classify_p95(ctx),
        "package_exit_code": rc1,
        "package_published": published1,
        "package_staging_present": staging1,
    }
    c2 = ctx.work / "C2"
    if not stage_inputs(ctx, c2, {"max_package_bytes": ctx.package_limit_bytes}):
        return fail_item("input_unreadable")
    steps2, pkg2, failure = run_pipeline(ctx, c2, None, {20})
    if failure:
        return dict(failure, case="C-2", steps=steps2, p95=p95_rec)
    rc2 = next(s["exit_code"] for s in steps2 if s["step"] == "package")
    cap2 = capacity_summary(pkg2)
    published = (c2 / "project" / "package").exists()
    staging2 = staging_present(c2 / "project")
    p95_2 = pkg2.get("infer_p95") if isinstance(pkg2, dict) else None
    code2 = pkg2.get("code") if isinstance(pkg2, dict) else None
    limit_rec = {
        "package_exit_code": rc2,
        "code": vocab_value(code2, CODE_VOCAB),
        "capacity_exceeded": cap2["exceeded"] if cap2 else None,
        "total_bytes": cap2["total_bytes"] if cap2 else None,
        "limit_bytes": cap2["limit_bytes"] if cap2 else None,
        "infer_p95_exceeded": (
            p95_2.get("exceeded") if isinstance(p95_2, dict) and "exceeded" in p95_2 else None
        ),
        "package_published": published,
        "package_staging_present": staging2,
    }
    reason2 = judge_capacity_limit(rc2, code2, cap2, ctx.package_limit_bytes, published)
    if reason2 is None and staging2:
        reason2 = "staging_left"
    rec = {
        "status": "ok" if reason2 is None else "failed",
        "p95": p95_rec,
        "capacity_limit": limit_rec,
    }
    if reason2 is not None:
        rec["reason"] = reason2
    return rec


def parse_otool_libraries(text: str) -> list[str]:
    """`otool -L` の出力から直接リンクのライブラリ名（1 行目と括弧以降を除く）を取り出す。"""
    libs = []
    for line in split_lines(text)[1:]:
        name = line.strip().split(" (")[0].strip()
        if name:
            libs.append(name)
    return libs


def parse_linkage_tool(text: str) -> str | None:
    """stdout を後ろから見て最初に一致した `OK: tool=<otool|ldd> ...` 行の tool。無ければ None。"""
    for line in reversed(split_lines(text)):
        m = re.match(r"^OK: tool=(\S+)", line)
        if m:
            return m.group(1) if m.group(1) in ("otool", "ldd") else None
    return None


def parse_linkage_cli_path(text: str) -> Path | None:
    """check-runtime-linkage.sh が出す `cli_bin: <絶対パス>` 行から検査対象の Path を得る。

    cargo の報告から取った実物のパスが唯一の出どころ（REQ-32）。該当行がちょうど 1 行で、
    値が絶対パスのときだけ返し、0 行・2 行以上・相対パスは None（照合不能として扱う）。
    """
    found = [m.group(1) for ln in split_lines(text) if (m := re.match(r"^cli_bin: (.+)$", ln))]
    if len(found) != 1 or not found[0].startswith("/"):
        return None
    return Path(found[0])


def judge_linkage_target(
    start_sha: str | None, bin_sha_now: str | None, target_sha: str | None
) -> str | None:
    """D の同一性判定（REQ-32）。リンクを確認した対象・いまの CLI・開始時の CLI が一致すれば None。

    検査対象を読めなければ `linkage_target_unreadable`、いまの CLI が開始時と違う（または読めない）
    なら `cli_changed`、検査対象だけ違えば `linkage_target_mismatch`。
    """
    if target_sha is None:
        return "linkage_target_unreadable"
    if start_sha is None or bin_sha_now is None or bin_sha_now != start_sha:
        return "cli_changed"
    if target_sha != bin_sha_now:
        return "linkage_target_mismatch"
    return None


def item_d(ctx: Ctx) -> dict[str, Any]:
    """D: `make check-runtime-linkage`・検査対象と実行 CLI の同一性・`otool -L`（macOS のみ）。"""
    d = ctx.work / "D"
    d.mkdir()
    r = run_cmd(
        [ctx.make_cmd, "check-runtime-linkage"],
        ctx.repo,
        d / "linkage.log",
        d / "linkage.err",
        TIMEOUT_LINKAGE,
        CAP_LOG_STDOUT,
        CAP_LOG_STDERR,
        ctx.offline_env,
    )
    if r.reason is not None:
        return fail_item(r.reason, exit_code=r.exit_code)
    text, why = read_capped_ex(d / "linkage.log", CAP_LOG_STDOUT)
    if text is None:
        return fail_item(why or "output_unreadable", exit_code=r.exit_code)
    rec: dict[str, Any] = {
        "exit_code": r.exit_code,
        "skip_lines": sum(1 for ln in split_lines(text) if ln.startswith("skip:")),
        "env_i_tests_ok": sum(1 for ln in split_lines(text) if ln.startswith("ok: req32_")),
        "linkage_tool": parse_linkage_tool(text),
        "direct_libraries": None,
        "linkage_target_sha256": None,
        "linkage_target_matches_cli": None,
    }
    if r.exit_code != 0:
        return dict(rec, status="failed", reason="unexpected_exit_code")
    # skip を検証済みと扱わない。env -i のテストが 1 件も ok になっていなければ確認できていない
    if rec["skip_lines"] > 0:
        return dict(rec, status="failed", reason="skipped")
    if rec["env_i_tests_ok"] < 1:
        return dict(rec, status="failed", reason="no_test_results")
    # 成功の印を厳密に: テスト数が定数と一致し、stdout を後ろから見て最初に一致した
    # `OK: tool=...` の行があること。
    # macOS では動的リンクの確認が otool で行われたこと（ldd 等の補助では実機の確認にならない）
    if rec["env_i_tests_ok"] != LINKAGE_ENV_I_TESTS or rec["linkage_tool"] is None:
        return dict(rec, status="failed", reason="unexpected_output")
    if sys.platform == "darwin" and rec["linkage_tool"] != "otool":
        return dict(rec, status="failed", reason="unexpected_output")
    # check-runtime-linkage.sh が検査したバイナリ（`cli_bin:` 行で報告された cargo の実物）が、
    # B・C・E で実行した CLI と同一でなければ、リンク確認は実行した CLI の証拠にならない。
    # 代役の下の偽 make は何も検査していないので照合しない（matches は null のまま）
    if not ctx.harness:
        target = parse_linkage_cli_path(text)
        target_sha = sha256_file(target, CAP_CLI_BINARY) if target is not None else None
        now_sha = sha256_file(ctx.bin, CAP_CLI_BINARY)
        rec["linkage_target_sha256"] = target_sha
        reason = judge_linkage_target(ctx.cli_sha256, now_sha, target_sha)
        rec["linkage_target_matches_cli"] = reason is None
        if reason is not None:
            return dict(rec, status="failed", reason=reason)
    # テストの偽 CLI（スクリプト）には otool が失敗するため、代役の下では取らない
    # macOS で otool が固定パスに無ければ確認できていない（PATH へは戻らず、黙って ok にしない）
    if sys.platform == "darwin" and not ctx.harness:
        otool = resolve_tool("otool")
        if otool is None:
            return dict(rec, status="failed", reason="otool_failed")
        o = run_cmd(
            [otool, "-L", str(ctx.bin)],
            ctx.repo,
            d / "otool.out",
            d / "otool.err",
            TIMEOUT_PROBE,
            CAP_PROBE,
            CAP_PROBE,
            probe_env(),
        )
        body = read_capped(d / "otool.out", CAP_PROBE)
        if o.reason is not None or o.exit_code != 0 or body is None:
            return dict(rec, status="failed", reason="otool_failed")
        rec["direct_libraries"] = parse_otool_libraries(body)
    return dict(rec, status="ok")


def read_train_inputs(path: Path) -> list[tuple[str, str]] | None:
    """train.jsonl から (id, input) だけを取り出す。上限超過・形の違いは None。"""
    text = read_capped(path, CAP_INPUT_FILE)
    if text is None:
        return None
    out: list[tuple[str, str]] = []
    for line in split_lines(text):
        if not line.strip():
            continue
        try:
            v = _loads(line)
        except ValueError:
            return None
        if not isinstance(v, dict) or not isinstance(v.get("id"), str):
            return None
        if not isinstance(v.get("input"), str):
            return None
        out.append((v["id"], v["input"]))
    return out


def compare_infer(
    batch: dict[str, dict[str, Any]], single: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """バッチと単体の `infer` 結果を id で突き合わせる（REQ-28）。id の値は件数欄に含めない。

    `predicted_label` が str でない・両側で異なる行、スコアに NaN・無限大・非数がある行、
    `status`（`ok`・`out_of_scope`・`abstain`。REQ-22）が両側で異なる行は不一致。
    `max_abs_score_diff` は有限の差だけから求める（非有限は `scores_nonfinite` として数える）。
    スコアが完全一致（`==`）でない行の id も `mismatch_ids` に入れる（同一実装の単体とバッチは
    スコアも完全一致という決まり。`SCORE_TOLERANCE` は使わない）。`max_abs_score_diff` は
    参考値で、合否には使わない。
    """
    label_match = 0
    scores_exact = 0
    nonfinite = 0
    max_diff = 0.0
    mismatch_ids: list[str] = []
    for rid, s in single.items():
        b = batch.get(rid)
        if b is None:
            mismatch_ids.append(rid)
            continue
        bl, sl = b.get("predicted_label"), s.get("predicted_label")
        same_label = isinstance(bl, str) and isinstance(sl, str) and bl == sl
        bs, ss = b.get("scores"), s.get("scores")
        finite = isinstance(bs, dict) and isinstance(ss, dict)
        row_diff = 0.0
        if finite:
            for k in set(bs) | set(ss):
                x, y = bs.get(k), ss.get(k)
                if not (_is_finite_number(x) and _is_finite_number(y)):
                    finite = False
                    continue
                row_diff = max(row_diff, abs(float(x) - float(y)))
        max_diff = max(max_diff, row_diff)
        if same_label:
            label_match += 1
        if finite:
            if bs == ss:
                scores_exact += 1
        else:
            nonfinite += 1
        # スコアが完全一致でない行も mismatch_ids へ入れる（件数欄の意味は変えない）
        if not same_label or not finite or bs != ss or b.get("status") != s.get("status"):
            mismatch_ids.append(rid)
    return {
        "label_match": label_match,
        "label_mismatch": len(single) - label_match,
        "scores_exact_match": scores_exact,
        "scores_nonfinite": nonfinite,
        "max_abs_score_diff": max_diff,
        "mismatch_ids": mismatch_ids,
    }


def item_e(ctx: Ctx) -> dict[str, Any]:
    """E: B の `train.jsonl` の入力だけで、バッチ推論と単体推論の全件一致を確かめる（REQ-28）。

    `evaluation.jsonl` は使わない（REQ-27。評価データを推論の確認に再利用しない）。
    """
    bdir = ctx.work / "B"
    edir = ctx.work / "E"
    edir.mkdir()
    recs = read_train_inputs(bdir / "train.jsonl")
    facts = read_facts(bdir)
    if recs is None or facts is None:
        return fail_item("input_unreadable")
    if not recs or len(recs) > MAX_E_RECORDS:
        return fail_item("record_count_out_of_range", records=len(recs))
    if len({rid for rid, _ in recs}) != len(recs):
        return fail_item("duplicate_id", records=len(recs))
    inputs = bdir / "e-inputs.jsonl"
    try:
        write_text_nofollow(
            inputs,
            "".join(
                json.dumps({"id": rid, "input": text}, ensure_ascii=False) + "\n"
                for rid, text in recs
            ),
        )
    except OSError:
        return fail_item("spawn_error")
    digest = sha256_file(inputs, CAP_INPUT_FILE)
    r = run_cmd(
        [str(ctx.bin), "infer", "--package", "project/package", "--input-file", "e-inputs.jsonl"],
        bdir,
        edir / "batch.stdout",
        edir / "batch.stderr",
        TIMEOUT_CLI_STEP,
        CAP_CLI_STDOUT,
        CAP_CLI_STDERR,
        ctx.offline_env,
    )
    if r.reason is not None or r.exit_code != 0:
        reason = r.reason or "unexpected_exit_code"
        return fail_item(reason, step="infer-batch", exit_code=r.exit_code)
    text = read_capped(edir / "batch.stdout", CAP_CLI_STDOUT) or ""
    batch: dict[str, dict[str, Any]] = {}
    lines = split_lines(text)
    if len(lines) != len(recs):
        return fail_item("unexpected_output", step="infer-batch", exit_code=0)
    for (expected_id, _), line in zip(recs, lines):  # noqa: B905  Python 3.9 互換のため strict なし
        try:
            v = _loads(line)
        except ValueError:
            return fail_item("invalid_json", step="infer-batch", exit_code=0)
        # バッチは exit 0 で、各行の `status` は対象外・保留を含む 3 値のどれか（REQ-22）
        if not isinstance(v, dict):
            return fail_item("unexpected_output", step="infer-batch", exit_code=0)
        row_rc = next((c for c, st in INFER_EXIT_STATUS.items() if st == v.get("status")), None)
        if row_rc is None or not _infer_envelope_ok(v, row_rc):
            return fail_item("unexpected_output", step="infer-batch", exit_code=0)
        # 出力は入力順（crates/cli/src/infer_batch.rs）。dict へ入れる前に行ごとに順序を確かめる
        if v["id"] != expected_id:
            return fail_item("unexpected_output", step="infer-batch", exit_code=0)
        # id の重複は後勝ちで潰さない
        if v["id"] in batch or not check_infer_output(v, facts):
            return fail_item("unexpected_output", step="infer-batch", exit_code=0)
        batch[v["id"]] = v
    single: dict[str, dict[str, Any]] = {}
    for rid, inp in recs:
        # `--text` の値が `-` で始まっても値として消費される（crates/cli/src/args.rs）
        argv = [str(ctx.bin), "infer", "--package", "project/package", "--text", inp, "--id", rid]
        r = run_cmd(
            argv,
            bdir,
            edir / "single.stdout",
            edir / "single.stderr",
            TIMEOUT_CLI_STEP,
            CAP_CLI_STDOUT,
            CAP_CLI_STDERR,
            ctx.offline_env,
        )
        # 単発は対象外 11・保留 12 も、`status` が終了コードに対応すれば正常（REQ-21・REQ-22）
        if r.reason is not None or r.exit_code not in INFER_EXIT_STATUS:
            reason = r.reason or "unexpected_exit_code"
            return fail_item(reason, step="infer-single", exit_code=r.exit_code)
        obj = parse_json_object(edir / "single.stdout", CAP_CLI_STDOUT)
        if (
            obj is None
            or not _infer_envelope_ok(obj, r.exit_code)
            or obj.get("id") != rid
            or not check_infer_output(obj, facts)
        ):
            return fail_item("unexpected_output", step="infer-single", exit_code=r.exit_code)
        single[rid] = obj
    cmp = compare_infer(batch, single)
    ids = cmp.pop("mismatch_ids")
    write_text_nofollow(edir / "mismatch-ids.txt", "".join(i + "\n" for i in ids))
    # 合否はラベル・status 全件一致・非有限 0 件・スコア全件完全一致。max_abs_score_diff は参考値
    ok = (
        not ids
        and cmp["label_mismatch"] == 0
        and cmp["scores_nonfinite"] == 0
        and cmp["scores_exact_match"] == len(single)
    )
    rec = dict({"status": "ok" if ok else "failed", "records": len(recs)}, **cmp)
    rec["input_sha256"] = digest
    if not ok:
        rec["reason"] = "mismatch"
    return rec


# --------------------------------------------------------------------------------------
# G〜J（#469 の CLI 結線で増えた機能の実機確認）
# --------------------------------------------------------------------------------------


def go_step(
    ctx: Ctx,
    cwd: Path,
    logs: Path,
    steps: list[dict[str, Any]],
    name: str,
    tag: str,
    argv: list[str],
    command: str,
    *,
    facts: Facts | None = None,
    allowed: frozenset[int] = frozenset(),
    selected: int | None = None,
    kind_before: str | None = None,
    timeout: int = TIMEOUT_CLI_STEP,
    during: Callable[[int], None] | None = None,
    reap_group: bool = True,
) -> tuple[int | None, dict[str, Any] | None, dict[str, Any] | None]:
    """CLI を 1 回実行して `steps` へ記録する。戻り値は (終了コード, stdout の JSON, 失敗記録)。

    `facts` があれば工程の契約検査（`_step_check`。exit 0 は報告値まで、非 0 は `allowed` の中か）を
    行い、`summary` を付ける。無ければ呼び出し側が判定する（終了コードと JSON の取得までを保証）。
    `command` は記録へ出す固定の表示名（パス・利用者の値を含めない）。
    """
    logs.mkdir(exist_ok=True)
    so = logs / f"{len(steps) + 1:02d}-{tag}.stdout"
    se = logs / f"{len(steps) + 1:02d}-{tag}.stderr"
    r = run_cmd(
        [str(ctx.bin), *argv],
        cwd,
        so,
        se,
        timeout,
        CAP_CLI_STDOUT,
        CAP_CLI_STDERR,
        ctx.offline_env,
        during=during,
        reap_group=reap_group,
    )
    entry: dict[str, Any] = {
        "step": name,
        "case": tag,
        "command": command,
        "exit_code": r.exit_code,
        "stderr_bytes": r.err_bytes,
    }
    steps.append(entry)
    if r.reason is not None:
        return (
            None,
            None,
            fail_item(r.reason, step=name, case=tag, exit_code=r.exit_code),
        )
    rc = r.exit_code if r.exit_code is not None else EXIT_RUNTIME_ERROR
    obj = parse_json_object(so, CAP_CLI_STDOUT)
    if obj is None:
        return rc, None, fail_item("invalid_json", step=name, case=tag, exit_code=rc)
    if facts is not None:
        entry["summary"] = summarize_step(name, obj, facts.option_ids)
        if rc != 0 and rc not in allowed:
            err = error_fields(obj)
            return (
                rc,
                obj,
                fail_item("unexpected_exit_code", step=name, case=tag, exit_code=rc, **err),
            )
        if not _step_check(name, obj, set(allowed), rc, facts, selected, kind_before):
            return (
                rc,
                obj,
                fail_item(
                    "unexpected_output",
                    step=name,
                    case=tag,
                    exit_code=rc,
                    **error_fields(obj),
                ),
            )
    return rc, obj, None


def prepare_project(
    ctx: Ctx, d: Path, facts: Facts, logs: Path, steps: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """`d`（定義・train・evaluation を置いたディレクトリ）で register → inspect を実行する。

    プロジェクトは `d/project`。失敗記録（無ければ None）を返す。
    """
    for name, argv in (
        (
            "register",
            ["register", "--definition", "definition.json", "--project-dir", "project"],
        ),
        ("inspect", ["inspect", "--project-dir", "project"]),
    ):
        command = " ".join(argv)
        _rc, _obj, failure = go_step(ctx, d, logs, steps, name, name, argv, command, facts=facts)
        if failure:
            return failure
    return None


def judge_train_all(
    rc: int, obj: dict[str, Any], budget_seconds: int, project: Path
) -> tuple[str | None, dict[str, Any]]:
    """G の `train --all` の判定（REQ-18・REQ-34・#482・#483）。戻り値は (失敗理由, 要約)。

    終了コードは 0（`evaluated` が 1 件以上）か
    20（全件が予算到達。stdout は `code:"limit_exceeded"` の
    エラー JSON）のどちらか。どちらも `search_record.json` が残る。exit 0 は `candidates[].result`・
    `budget_reached` を語彙・型で確かめ、`evaluated` の候補だけ `result.json` が残り、それ以外の
    候補ディレクトリは片付いていること。
    """
    if rc not in (0, 20):
        return "unexpected_exit_code", {}
    record = _json_file(project / "search_record.json", 1024 * 1024)
    if not isinstance(record, dict):
        return "search_record_missing", {}
    if rc == 20:
        if vocab_value(obj.get("code"), CODE_VOCAB) != "limit_exceeded":
            return "unexpected_output", {}
        return None, {"outcome": "budget_exhausted", "search_record_present": True}
    cands = obj.get("candidates")
    if (
        obj.get("step") != "train"
        or obj.get("status") != "ok"
        or not _eq_int(obj.get("budget_seconds"), budget_seconds)
        or not isinstance(obj.get("budget_reached"), bool)
        or not _nonneg_int(obj.get("total_elapsed_ms"))
        or not isinstance(cands, list)
        or not cands
    ):
        return "unexpected_output", {}
    summary: list[dict[str, Any]] = []
    evaluated = 0
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            return "unexpected_output", {}
        result, scope = c.get("result"), c.get("budget_reached")
        if (
            not _eq_int(c.get("candidate"), i)
            or c.get("kind") not in KIND_VOCAB
            or not isinstance(result, str)
            or result not in TRAIN_RESULT_VOCAB
            or not (scope is None or (isinstance(scope, str) and scope in BUDGET_SCOPE_VOCAB))
        ):
            return "unexpected_output", {}
        cdir = project / "candidates" / str(i)
        if result == "evaluated":
            evaluated += 1
            if _read_regular_capped(cdir / "result.json", CAP_INPUT_FILE) is None:
                return "candidate_result_missing", {}
        elif os.path.lexists(cdir):
            return "candidate_dir_not_cleaned", {}
        summary.append(
            {
                "candidate": i,
                "kind": c["kind"],
                "result": result,
                "budget_reached": scope,
            }
        )
    if evaluated < 1:
        return "no_candidate_evaluated", {}
    return None, {
        "outcome": "evaluated",
        "budget_seconds": budget_seconds,
        "budget_reached": obj["budget_reached"],
        "total_elapsed_ms": obj["total_elapsed_ms"],
        "candidates": summary,
        "search_record_present": True,
    }


def item_g(ctx: Ctx) -> dict[str, Any]:
    """G: `train --all --budget-seconds N`（探索予算内の全候補の学習と予算到達の記録）。

    REQ-18・#482。

    学習は CLI 経由（実 trainer）。予算は `--g-budget-seconds`（既定 3600）。
    exit 0 と exit 20（全件が予算到達）のどちらも想定内で、合否の解釈（予算が妥当か）は人が行う。
    """
    gdir = ctx.work / "G"
    if not stage_inputs(ctx, gdir, None):
        return fail_item("input_unreadable")
    facts = read_facts(gdir)
    if facts is None:
        return fail_item("input_unreadable")
    logs, steps = gdir / "steps", []
    failure = prepare_project(ctx, gdir, facts, logs, steps)
    if failure:
        return dict(failure, steps=steps)
    budget = ctx.g_budget_seconds
    rc, obj, failure = go_step(
        ctx,
        gdir,
        logs,
        steps,
        "train",
        "train-all",
        ["train", "--project-dir", "project", "--all", "--budget-seconds", str(budget)],
        f"train --project-dir project --all --budget-seconds {budget}",
        timeout=budget + G_TIMEOUT_MARGIN,
    )
    if failure or rc is None or obj is None:
        return dict(failure or fail_item("invalid_json"), steps=steps)
    reason, summary = judge_train_all(rc, obj, budget, gdir / "project")
    if reason is not None:
        return fail_item(reason, step="train", exit_code=rc, steps=steps, **error_fields(obj))
    return dict({"status": "ok", "exit_code": rc, "steps": steps}, **summary)


# プロセスの識別情報（pid の再利用で無関係なプロセスを止めないため、開始時刻と併せて控える）
@dataclass(frozen=True)
class Proc:
    """`ps` の 1 行。`start` は開始時刻（`lstart`。pid の再利用の見分け）、`command` は引数全体。"""

    ppid: int
    start: str
    command: str


def run_bounded(
    argv: list[str],
    cwd: Path | None,
    env: dict[str, str] | None,
    out: Any,
    err: Any,
    timeout: float,
    out_cap: int,
    err_cap: int,
) -> tuple[int | None, str | None]:
    """上限つきの短い子プロセス実行（`during` コールバックの中から呼べる。REQ-39）。

    `run_cmd` はグローバルな子の追跡（`_active_pgid` 等）を使うため、その待機中のコールバックから
    入れ子では呼べない。ここは状態を持たず、独立したセッションで起動し、`out`・`err`（書き込み用の
    ファイルオブジェクト）のサイズが上限を超える・期限を過ぎたら、回収の前にグループごと KILL
    する。戻り値は (終了コード, 失敗理由)。理由は `timeout`・`output_limit`・`spawn_error`。
    """
    try:
        proc = subprocess.Popen(  # noqa: S603  固定の引数。shell は使わない
            argv,
            cwd=None if cwd is None else str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            env=env,
            start_new_session=True,
        )
    except (OSError, ValueError):
        return None, "spawn_error"
    reason = None
    deadline = time.monotonic() + timeout
    try:
        while proc.poll() is None:
            if time.monotonic() >= deadline:
                reason = "timeout"
                break
            if os.fstat(out.fileno()).st_size > out_cap or os.fstat(err.fileno()).st_size > err_cap:
                reason = "output_limit"
                break
            time.sleep(0.02)
    finally:
        # リーダーが未回収のうちにグループごと KILL してから回収する（pid の再利用を避ける）
        global _child_may_remain
        if proc.poll() is None and not _kill_group(proc.pid):
            # グループへ送れなかった。孫が残りうるので記録し、せめてリーダーだけでも止める
            _child_may_remain = True
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=REAP_WAIT_LIMIT_SECONDS)
        except subprocess.TimeoutExpired:
            # 回収を諦める。子が残りうる印を立て、Popen を保持する（参照が生きている間は pid が
            # 再利用されない。`run_cmd` と同じ扱い）
            _child_may_remain = True
            _leftover_procs.append(proc)
            reason = REASON_UNREAPED
    if reason is None and (
        os.fstat(out.fileno()).st_size > out_cap or os.fstat(err.fileno()).st_size > err_cap
    ):
        reason = "output_limit"
    if reason is not None:
        return None, reason
    return proc.returncode, None


def ps_procs() -> dict[int, Proc] | None:
    """`ps` の (pid → `Proc`) 表。固定パスの `ps` が使えなければ None（REQ-38・REQ-39）。"""
    if not os.path.isfile(PS_PATH) or not os.access(PS_PATH, os.X_OK):
        return None
    # 全プロセスの引数を読むため出力は大きくなりうる。無名の一時ファイルへ書かせ、超えたら止める。
    # macOS の ps は -ww が無いと command を表示幅で切り詰め、末尾の `_worker` が消える
    try:
        with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
            rc, _reason = run_bounded(
                [PS_PATH, "-A", "-ww", "-o", "pid=,ppid=,lstart=,command="],
                None,
                probe_env(),
                fo,
                fe,
                TIMEOUT_AUX,
                CAP_PS_STDOUT,
                CAP_PS_STDERR,
            )
            if rc != 0:
                return None
            fo.seek(0)
            raw = fo.read(CAP_PS_STDOUT + 1)
    except OSError:
        return None
    if len(raw) > CAP_PS_STDOUT:
        return None
    table: dict[int, Proc] = {}
    for line in split_lines(raw.decode("utf-8", errors="replace")):
        parts = line.split(None, 7)
        # pid ppid + lstart（曜日 月 日 時刻 年の 5 語）+ command
        if len(parts) >= 7 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = Proc(
                int(parts[1]), " ".join(parts[2:7]), parts[7] if len(parts) > 7 else ""
            )
    return table


def ps_table() -> dict[int, int] | None:
    """`ps` の (pid → ppid) 表。使えなければ None。"""
    procs = ps_procs()
    return None if procs is None else {pid: pr.ppid for pid, pr in procs.items()}


def descendants_of(table: dict[int, int], root: int) -> set[int]:
    """`root` の子孫の pid（`root` 自身を除く）。"""
    found: set[int] = set()
    frontier = {root}
    while frontier:
        frontier = {p for p, pp in table.items() if pp in frontier and p not in found}
        found |= frontier
    found.discard(root)
    return found


def is_worker(command: str) -> bool:
    """学習ワーカー（`_worker`）か。

    supervisor は `<python> -I <trainer>/launch.py _worker --out-fd N --lifeline-fd N` で
    起動する（trainer/src/fandhe_edge_trainer/supervisor.py の `worker_argv`）。
    """
    tokens = command.split()
    return "_worker" in tokens and any(t.endswith("launch.py") for t in tokens)


def _same_proc(procs: dict[int, Proc] | None, pid: int, start: str) -> bool:
    """`pid` が控えた開始時刻と同じプロセスとして今も存在するか。"""
    return procs is not None and pid in procs and procs[pid].start == start


class JobDriver:
    """H: 実行中の `train` の `job.json` が running になり、学習ワーカーが現れたら操作を行う。

    操作はキャンセル要求か KILL。`run_cmd` の `during` コールバックとして呼ばれる（待機の周
    ごと。実際の確認は `H_POLL_SEC` ごと）。例外は外へ出さず `error` に固定語彙で残す。
    `mode` は `cancel`（`train --cancel` を実行）か `kill`（train 本体の CLI を `SIGKILL`）。
    操作は `_worker`（`is_worker`。CLI と supervisor だけの状態では送らない）の出現後に行う。
    running・ワーカーを待つのは `H_WAIT_RUNNING_SEC` までで、超えたらそのまま実行し
    `timed_out` を立てる（呼び出し側が失敗にする）。操作後も、train の終了まで子孫を観測し続け、
    見つけたプロセスを `seen`（pid → 開始時刻）に残す。
    """

    def __init__(self, ctx: Ctx, cwd: Path, mode: str) -> None:
        self.ctx, self.cwd, self.mode = ctx, cwd, mode
        self.started = time.monotonic()
        self.last = 0.0
        self.done = False
        self.timed_out = False
        self.error: str | None = None
        self.seen: dict[int, str] = {}
        self.worker_seen = False
        self.cancel_rc: int | None = None
        self.killed = False

    def _job_state(self) -> str | None:
        rec = _json_file(self.cwd / "project/candidates/0/job/job.json", CAP_INPUT_FILE)
        state = rec.get("state") if isinstance(rec, dict) else None
        return state if isinstance(state, str) else None

    def __call__(self, root: int) -> None:
        now = time.monotonic()
        if self.error is not None or now - self.last < H_POLL_SEC:
            return
        self.last = now
        try:
            self._tick(root, now)
        except Exception:  # コールバックから例外を出さない（run_cmd の後始末を守る）
            self.error, self.done = "driver_error", True

    def _tick(self, root: int, now: float) -> None:
        procs = ps_procs()
        if procs is None:
            self.error, self.done = "ps_unavailable", True
            return
        table = {pid: pr.ppid for pid, pr in procs.items()}
        desc = descendants_of(table, root)
        for pid in desc:
            self.seen.setdefault(pid, procs[pid].start)
            if is_worker(procs[pid].command):
                self.worker_seen = True
        if self.done:
            return  # 操作の後も観測は続ける（後から起動したワーカーも控える）
        waited_out = now - self.started > H_WAIT_RUNNING_SEC
        if not (self._job_state() == "running" and self.worker_seen) and not waited_out:
            return
        self.timed_out = waited_out
        self.done = True
        if self.mode == "kill":
            cli = sorted(p for p in desc if table.get(p) == root)
            if cli and _same_proc(ps_procs(), cli[0], procs[cli[0]].start):
                os.kill(cli[0], signal.SIGKILL)
                self.killed = True
            return
        so, se = (
            self.cwd / "steps" / "cancel.stdout",
            self.cwd / "steps" / "cancel.stderr",
        )
        with (
            _open_write_nofollow(so, False) as fo,
            _open_write_nofollow(se, False) as fe,
        ):
            rc, reason = run_bounded(
                [str(self.ctx.bin), "train", "--project-dir", "project", "--cancel"],
                self.cwd,
                self.ctx.offline_env,
                fo,
                fe,
                TIMEOUT_AUX,
                CAP_CLI_STDOUT,
                CAP_CLI_STDERR,
            )
        if reason is not None:
            # 出力が上限を超えた・期限切れ・起動失敗（子は止めてある）。cancel_rc は None のまま
            self.error = f"cancel_{reason}"
            return
        self.cancel_rc = rc


def alive_seen(seen: dict[int, str]) -> list[int]:
    """控えたプロセスのうち、同じプロセスとして今も生きているもの（昇順）。

    `ps` が使えなければ全件が残っている扱い（fail-closed）。開始時刻が違えば pid の再利用で
    別のプロセスなので数えない。
    """
    procs = ps_procs()
    if procs is None:
        return sorted(seen)
    return sorted(p for p, start in seen.items() if _same_proc(procs, p, start))


def wait_descendants_gone(seen: dict[int, str]) -> list[int]:
    """観測した子孫が終わるのを `H_DESCENDANT_GRACE_SEC` まで待ち、残った pid を返す（昇順）。"""
    deadline = time.monotonic() + H_DESCENDANT_GRACE_SEC
    while True:
        alive = alive_seen(seen)
        if not alive or time.monotonic() >= deadline:
            return alive
        check_interrupt()
        check_overall()
        time.sleep(H_POLL_SEC)


def kill_seen(seen: dict[int, str]) -> list[int]:
    """控えたプロセスのうち、同じプロセスと確かめられたものだけを KILL し、なお残るものを返す。

    送る直前に `ps` を取り直して pid と開始時刻の一致を確かめる（pid の再利用で無関係なプロセスへ
    送らない。`run_cmd` が回収後の pgid へ送らないのと同じ考え方）。例外・中断の経路でも呼ぶため
    中断検査は行わない。KILL 後も残る（確認できない場合を含む）なら `child_may_remain` を立てる。
    """
    global _child_may_remain
    procs = ps_procs()
    if procs is None:
        _child_may_remain = bool(seen) or _child_may_remain
        return sorted(seen)
    for pid, start in seen.items():
        if _same_proc(procs, pid, start):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    deadline = time.monotonic() + 2.0
    alive = alive_seen(seen)
    while alive and time.monotonic() < deadline:
        time.sleep(0.1)
        alive = alive_seen(seen)
    if alive:
        _child_may_remain = True
    return alive


def settle_descendants(drv: JobDriver, run: Callable[[], Any]) -> tuple[Any, list[int]]:
    """`run()`（train の実行）と子孫の終了待ちを行い、残った子孫は例外の経路でも止める。

    戻り値は (`run()` の結果, 待った後も残っていた子孫の pid)。`Interrupted`・`OverallTimeout` が
    `run()` か待機から出ても、`finally` で確認済みの子孫を KILL し、止められなければ
    `child_may_remain` を立てる（REQ-39）。
    """
    leftover: list[int] = []
    try:
        result = run()
        leftover = wait_descendants_gone(drv.seen)
    finally:
        kill_seen(drv.seen)
    return result, leftover


def _restart_guidance_ok(g: Any) -> bool:
    """やり直し案内が再開なし・最初からのやり直し・固定の理由コードか（REQ-34）。"""
    return (
        isinstance(g, dict)
        and g.get("resumable") is False
        and g.get("action") == RESTART_ACTION
        and g.get("reason_code") == RESTART_REASON_CODE
    )


def judge_cancel_response(rc: int | None, obj: dict[str, Any] | None) -> str | None:
    """`train --cancel` の応答（exit 0・`cancellations[0].cancel == "requested"`。#484）。"""
    if rc != 0 or obj is None or obj.get("step") != "train" or obj.get("status") != "ok":
        return "cancel_response_invalid"
    cs = obj.get("cancellations")
    if not isinstance(cs, list) or len(cs) != 1 or not isinstance(cs[0], dict):
        return "cancel_response_invalid"
    return (
        None
        if _eq_int(cs[0].get("candidate"), 0) and cs[0].get("cancel") == "requested"
        else ("cancel_not_requested")
    )


def judge_cancelled_train(rc: int | None, obj: dict[str, Any] | None) -> str | None:
    """キャンセルされた `train` の出力（exit 70・固定の message・`job.state:"cancelled"`。

    `TrainInterruptedReport`。#484・#486）。
    """
    if rc != 70 or obj is None:
        return "train_exit_code_not_70"
    job = obj.get("job")
    if (
        obj.get("code") != "runtime_error"
        or obj.get("message") != CANCELLED_MESSAGE
        or obj.get("step") != "train"
        or not _eq_int(obj.get("candidate"), 0)
        or not isinstance(job, dict)
        or job.get("state") != "cancelled"
        or not _restart_guidance_ok(obj.get("restart"))
    ):
        return "train_output_invalid"
    return None


def judge_status(
    rc: int | None, obj: dict[str, Any] | None, *, crashed: bool
) -> tuple[str | None, dict[str, Any]]:
    """`train --status --candidate 0` の判定（exit 0・`jobs[0]`。#485）。

    `crashed` が偽なら `state:"cancelled"`
    （`crash_detected:false`・`failure:null`・`record_updated:false`）、
    真なら `state:"failed"`・`crash_detected:true`・`failure:{kind:"crashed",cause:"owner_lost"}`・
    `record_updated:true`（running の残骸を初回に書き戻した）。どちらも `restart` は再開なしの案内。
    """
    if rc != 0 or obj is None or obj.get("step") != "train" or obj.get("status") != "ok":
        return "status_invalid", {}
    jobs = obj.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1 or not isinstance(jobs[0], dict):
        return "status_invalid", {}
    entry = jobs[0]
    job = entry.get("job")
    if not _eq_int(entry.get("candidate"), 0) or not isinstance(job, dict):
        return "status_invalid", {}
    failure = job.get("failure")
    state = job.get("state")
    if crashed:
        ok = (
            state == "failed"
            and job.get("crash_detected") is True
            and job.get("record_updated") is True
            and isinstance(failure, dict)
            and failure.get("kind") == "crashed"
            and failure.get("cause") == "owner_lost"
        )
    else:
        ok = (
            state == "cancelled"
            and job.get("crash_detected") is False
            and job.get("record_updated") is False
            and failure is None
        )
    if not ok or not _restart_guidance_ok(entry.get("restart")):
        return "status_unexpected", {}
    out: dict[str, Any] = {"state": state}
    if crashed:
        out["cause"] = failure["cause"]
    out["restart_action"] = entry["restart"]["action"]
    return None, out


def item_h(ctx: Ctx) -> dict[str, Any]:
    """H: `train --cancel`・`--status` とクラッシュ検出（REQ-34・REQ-39。#484・#485・#486）。

    1) `train --candidate 0` を起動し、running・子孫の出現を待って別プロセスで
       `train --cancel` を送る。train は exit 70・`training cancelled`、`--status` は
       `cancelled`、`package/` は無く、観測した子孫は
       CLI 自身が止めて 0 件になること（ラッパーの後始末に頼らない。`GROUP_WRAPPER_NO_REAP`）。
    2) もう一度 `train --candidate 0`（`cancelled` は丸ごと消して新規に始まる）を起動し、
       同じく running を待って CLI 本体を `SIGKILL` する。`--status` は `failed`・
       `owner_lost`（初回検出）を返すこと。
    実 trainer を使う（CPU）。残った子孫は pid を控えて KILL し、記録に件数だけ残す。
    """
    hdir = ctx.work / "H"
    if not stage_inputs(ctx, hdir, None):
        return fail_item("input_unreadable")
    facts = read_facts(hdir)
    if facts is None:
        return fail_item("input_unreadable")
    logs, steps = hdir / "steps", []
    failure = prepare_project(ctx, hdir, facts, logs, steps)
    if failure:
        return dict(failure, steps=steps)
    train_argv = ["train", "--project-dir", "project", "--candidate", "0"]
    status_argv = ["train", "--project-dir", "project", "--status", "--candidate", "0"]
    status_cmd = "train --project-dir project --status --candidate 0"

    # 1) キャンセル
    drv = JobDriver(ctx, hdir, "cancel")
    (rc, tobj, failure), leftover = settle_descendants(
        drv,
        lambda: go_step(
            ctx, hdir, logs, steps, "train", "train-cancelled", train_argv,
            "train --project-dir project --candidate 0", during=drv, reap_group=False,
        ),
    )  # fmt: skip
    if drv.error is not None or failure is not None:
        return dict(failure or fail_item(drv.error or "driver_error", step="train"), steps=steps)
    if drv.timed_out:
        return fail_item("job_not_running", step="train", steps=steps)
    cancel_obj = parse_json_object(logs / "cancel.stdout", CAP_CLI_STDOUT)
    for reason in (
        judge_cancel_response(drv.cancel_rc, cancel_obj),
        judge_cancelled_train(rc, tobj),
    ):
        if reason is not None:
            return fail_item(reason, step="train", exit_code=rc, steps=steps, **error_fields(tobj))
    rc_s, sobj, failure = go_step(
        ctx, hdir, logs, steps, "train", "status-cancelled", status_argv, status_cmd
    )
    if failure:
        return dict(failure, steps=steps)
    reason, cancel_status = judge_status(rc_s, sobj, crashed=False)
    if reason is not None:
        return fail_item(reason, step="train", exit_code=rc_s, steps=steps)
    if os.path.lexists(hdir / "project" / "package"):
        return fail_item("package_created", step="train", steps=steps)
    if not drv.worker_seen:
        return fail_item("no_descendant_observed", step="train", steps=steps)
    if leftover:
        return fail_item(
            "descendants_remain_after_cancel",
            step="train",
            descendants_observed=len(drv.seen),
            descendants_remaining=len(leftover),
            steps=steps,
        )
    cancel_rec = {
        "cancel": "requested",
        "train_exit_code": rc,
        "descendants_observed": len(drv.seen),
        "descendants_remaining": 0,
        "package_absent": True,
        **cancel_status,
    }

    # 2) クラッシュ検出（train 本体を KILL）
    drv2 = JobDriver(ctx, hdir, "kill")
    (rc2, _obj2, failure), leftover2 = settle_descendants(
        drv2,
        lambda: go_step(
            ctx, hdir, logs, steps, "train", "train-killed", train_argv,
            "train --project-dir project --candidate 0", during=drv2, reap_group=False,
        ),
    )  # fmt: skip
    # KILL された train は JSON を出さない（invalid_json）のが正常。実行できなかった等だけ失敗にする
    if drv2.error is not None or (failure is not None and failure.get("reason") != "invalid_json"):
        return dict(
            failure or fail_item(drv2.error or "driver_error", step="train"),
            steps=steps,
        )
    if drv2.timed_out or not drv2.killed:
        return fail_item("job_not_running", step="train", steps=steps)
    if leftover2:
        return fail_item(
            "descendants_remain_after_crash",
            step="train",
            descendants_observed=len(drv2.seen),
            descendants_remaining=len(leftover2),
            steps=steps,
        )
    rc_c, cobj, failure = go_step(
        ctx, hdir, logs, steps, "train", "status-crashed", status_argv, status_cmd
    )
    if failure:
        return dict(failure, steps=steps)
    reason, crash_status = judge_status(rc_c, cobj, crashed=True)
    if reason is not None:
        return fail_item(reason, step="train", exit_code=rc_c, steps=steps)
    return {
        "status": "ok",
        "steps": steps,
        "cancel_check": cancel_rec,
        "crash_check": {
            "train_exit_code": rc2,
            "descendants_observed": len(drv2.seen),
            "descendants_remaining": 0,
            **crash_status,
        },
    }


def _unit_interval(v: Any) -> bool:
    """`{"lo","hi"}` が有限で 0 ≤ lo ≤ hi ≤ 1 か。"""
    return (
        isinstance(v, dict)
        and _unit_number(v.get("lo"))
        and _unit_number(v.get("hi"))
        and v["lo"] <= v["hi"]
    )


def judge_reproducibility(
    rep: Any, n_total: int, seeds: tuple[int, ...]
) -> tuple[str | None, dict[str, Any]]:
    """`evaluate` の `reproducibility`（3 seed 以上の Wilson 95% 区間の重なり。REQ-26・#490）。

    `runs` は seed 昇順で `seeds` と一致、各 run の `total` は評価件数・
    `correct ≤ total`・区間は 0〜1。
    `disjoint_pairs` は `runs` の seed の組で、空 ⇔ `all_pairs_overlap`。`some_pairs_disjoint` も
    正常な出力（再現性の合否の解釈は人が行う）。戻り値は (失敗理由, 要約)。
    """
    if not isinstance(rep, dict):
        return "reproducibility_invalid", {}
    runs, verdict, pairs = (
        rep.get("runs"),
        rep.get("verdict"),
        rep.get("disjoint_pairs"),
    )
    if (
        not isinstance(runs, list)
        or not isinstance(verdict, str)
        or verdict not in REPRODUCIBILITY_VERDICT_VOCAB
        or not isinstance(pairs, list)
        or [r.get("seed") if isinstance(r, dict) else None for r in runs] != sorted(seeds)
    ):
        return "reproducibility_invalid", {}
    out_runs = []
    for r in runs:
        if (
            not _eq_int(r.get("total"), n_total)
            or not _nonneg_int(r.get("correct"))
            or r["correct"] > n_total
            or not _unit_interval(r.get("ci95"))
        ):
            return "reproducibility_invalid", {}
        out_runs.append({"seed": r["seed"], "correct": r["correct"], "total": r["total"]})
    ok_pairs = all(
        isinstance(p, list)
        and len(p) == 2
        and _is_int(p[0])
        and _is_int(p[1])
        and p[0] < p[1]
        and p[0] in seeds
        and p[1] in seeds
        for p in pairs
    )
    if not ok_pairs or (not pairs) != (verdict == "all_pairs_overlap"):
        return "reproducibility_invalid", {}
    return None, {"verdict": verdict, "runs": out_runs, "disjoint_pairs": pairs}


def judge_comparison(cmp: Any, n_total: int) -> tuple[str | None, dict[str, Any]]:
    """`evaluate --previous-project-dir` の `comparison`（REQ-26・#488・#489）。

    I の旧モデルは同じ定義・同じ凍結 test の別 seed の複製なので、`premise:"same_label_set"`・
    `evaluation_data:"same"`・ラベルの増減なし・共通レコードは全件、`counts` は 4 区分の合計が `n`。
    p 値・有意性は持たない契約のため、回帰の多寡の解釈は人が行う。戻り値は (失敗理由, 要約)。
    """
    if not isinstance(cmp, dict):
        return "comparison_invalid", {}
    counts = cmp.get("counts")
    parts = (
        "both_correct",
        "correct_to_incorrect",
        "incorrect_to_correct",
        "both_wrong",
    )
    if (
        cmp.get("premise") != "same_label_set"
        or cmp.get("evaluation_data") != "same"
        or cmp.get("removed_labels") != []
        or cmp.get("added_labels") != []
        or not _eq_int(cmp.get("n_common"), n_total)
        or not _eq_int(cmp.get("n_previous_only"), 0)
        or not _eq_int(cmp.get("n_current_only"), 0)
        or not isinstance(cmp.get("previous"), dict)
        or not isinstance(counts, dict)
        or not _eq_int(counts.get("n"), n_total)
        or not all(_nonneg_int(counts.get(k)) for k in parts)
        or sum(counts[k] for k in parts) != n_total
        or not _unit_interval(counts.get("correct_to_incorrect_ci95"))
        or not _unit_interval(counts.get("incorrect_to_correct_ci95"))
    ):
        return "comparison_invalid", {}
    return None, {
        "premise": cmp["premise"],
        "evaluation_data": cmp["evaluation_data"],
        "n_common": cmp["n_common"],
        "counts": {k: counts[k] for k in ("n", *parts)},
    }


def item_i(ctx: Ctx) -> dict[str, Any]:
    """I: 3 seed の再現性と、前のモデルとの比較（REQ-26・REQ-27・#488〜#490・#103）。

    register → inspect の後にプロジェクトを複製し（p1・p2・p3・old）、各複製で
    `train --train-seed S → select → evaluate` を実行する（p2・p3・old を先に評価し、最後の p1 の
    `evaluate` に `--seed-run-project p2 --seed-run-project p3`・
    `--previous-project-dir old` を付ける。
    凍結 test への適用は複製ごとに 1 回）。学習を 4 回行うため長時間かかる。既定の項目に入れず、
    `--items I` のときだけ実行する。`--i-device` は証拠種別の申告（cpu → `cpu_real_machine`、
    gpu → `gpu_real_machine_declared`）。CLI の `train` は現状 CPU 固定のため、gpu は人が別の手段で
    GPU 学習へ切り替えたときだけ指定する（切り替えの有無は記録できない）。
    """
    idir = ctx.work / "I"
    if not stage_inputs(ctx, idir, None):
        return fail_item("input_unreadable")
    facts = read_facts(idir)
    if facts is None:
        return fail_item("input_unreadable")
    logs, steps = idir / "steps", []
    failure = prepare_project(ctx, idir, facts, logs, steps)
    if failure:
        return dict(failure, steps=steps)
    seeds = {
        "p1": I_SEEDS[0],
        "p2": I_SEEDS[1],
        "p3": I_SEEDS[2],
        "old": I_PREVIOUS_SEED,
    }
    try:
        for name in seeds:
            shutil.copytree(idir / "project", idir / name, symlinks=True)
    except (OSError, shutil.Error):
        return fail_item("copy_failed", steps=steps)
    final: dict[str, Any] | None = None
    for name in ("p2", "p3", "old", "p1"):
        seed = seeds[name]
        _rc, obj, failure = go_step(
            ctx, idir, logs, steps, "train", f"train-{name}",
            ["train", "--project-dir", name, "--candidate", "0", "--train-seed", str(seed)],
            f"train --project-dir {name} --candidate 0 --train-seed {seed}",
            facts=facts,
        )  # fmt: skip
        if failure or obj is None:
            return dict(failure or fail_item("invalid_json"), steps=steps)
        kind = obj.get("kind")
        _rc, obj, failure = go_step(
            ctx, idir, logs, steps, "select", f"select-{name}",
            ["select", "--project-dir", name], f"select --project-dir {name}",
            facts=facts, kind_before=kind,
        )  # fmt: skip
        if failure or obj is None:
            return dict(failure or fail_item("invalid_json"), steps=steps)
        c = obj.get("candidate")
        if not _is_int(c) or c < 0:
            return fail_item("missing_field", step="select", exit_code=0, steps=steps)
        argv = ["evaluate", "--project-dir", name, "--candidate", str(c)]
        command = f"evaluate --project-dir {name} --candidate {c}"
        if name == "p1":
            argv += ["--previous-project-dir", "old"]
            for other in ("p2", "p3"):
                argv += ["--seed-run-project", other]
            command += " --previous-project-dir old --seed-run-project p2 --seed-run-project p3"
        _rc, obj, failure = go_step(
            ctx, idir, logs, steps, "evaluate", f"evaluate-{name}", argv, command,
            facts=facts, selected=c, kind_before=kind,
        )  # fmt: skip
        if failure or obj is None:
            return dict(failure or fail_item("invalid_json"), steps=steps)
        final = obj
    if final is None:
        return fail_item("missing_field", steps=steps)
    n_total = final.get("n_total")
    reason, repro = judge_reproducibility(final.get("reproducibility"), n_total, I_SEEDS)
    if reason is None:
        reason, comparison = judge_comparison(final.get("comparison"), n_total)
    if reason is not None:
        return fail_item(reason, step="evaluate", exit_code=0, steps=steps)
    return {
        "status": "ok",
        "steps": steps,
        "evidence": I_EVIDENCE[ctx.i_device],
        "device": ctx.i_device,
        "seeds": [*I_SEEDS, I_PREVIOUS_SEED],
        "reproducibility": repro,
        "comparison": comparison,
    }


def item_j(ctx: Ctx) -> dict[str, Any]:
    """J: 前の版への復帰（REQ-39・REQ-27・#491）。B の `project`（v1）を旧プロジェクトに使う。

    新プロジェクト（`J/project`）を package まで実行し、`package --previous-project-dir`
    で v2 を作る。続けて、v1 の `package`（B のもの）を新しい台帳の `--version-id v1` で
    `infer` できること、台帳の最新版（v2）でも `infer` できること、`--version-id` だけの
    指定は exit 64、`artifact.json` を 1 バイト改変した複製は exit 64 になることを確かめる。
    旧・新の `package/` は変更しない（改変は複製）。CLI はカレント配下の相対パスだけを
    受けるため、カレントは作業ディレクトリ（`B/`・`J/` の親）にする。
    """
    bproj = ctx.work / "B" / "project"
    if not (bproj / "package").is_dir():
        return fail_item("previous_package_missing")
    jdir = ctx.work / "J"
    if not stage_inputs(ctx, jdir, None):
        return fail_item("input_unreadable")
    facts = read_facts(jdir)
    if facts is None:
        return fail_item("input_unreadable")
    cwd, logs, steps = ctx.work, jdir / "steps", []
    sample = "sandbox check 0123456789"
    ledger = "J/project/version_ledger.json"
    p = "J/project"

    def run(name: str, tag: str, argv: list[str], command: str, **kw: Any) -> Any:
        return go_step(ctx, cwd, logs, steps, name, tag, argv, command, **kw)

    selected: int | None = None
    kind: str | None = None
    for name, argv, command in (
        ("register", ["register", "--definition", "J/definition.json", "--project-dir", p],
         "register --definition <definition> --project-dir <project>"),
        ("inspect", ["inspect", "--project-dir", p], "inspect --project-dir <project>"),
        ("train", ["train", "--project-dir", p, "--candidate", "0"],
         "train --project-dir <project> --candidate 0"),
        ("select", ["select", "--project-dir", p], "select --project-dir <project>"),
    ):  # fmt: skip
        _rc, obj, failure = run(name, name, argv, command, facts=facts, kind_before=kind)
        if failure or obj is None:
            return dict(failure or fail_item("invalid_json"), steps=steps)
        if name in ("train", "select"):
            kind = obj.get("kind")
        if name == "select":
            selected = obj.get("candidate")
    if not _is_int(selected) or selected < 0:
        return fail_item("missing_field", step="select", exit_code=0, steps=steps)
    _rc, obj, failure = run(
        "evaluate", "evaluate",
        ["evaluate", "--project-dir", p, "--candidate", str(selected)],
        f"evaluate --project-dir <project> --candidate {selected}",
        facts=facts, selected=selected, kind_before=kind,
    )  # fmt: skip
    if failure:
        return dict(failure, steps=steps)
    _rc, pkg, failure = run(
        "package", "package-v2",
        ["package", "--project-dir", p, "--previous-project-dir", "B/project"],
        "package --project-dir <project> --previous-project-dir <previous-project>",
        facts=facts, allowed=frozenset({0}),
    )  # fmt: skip
    if failure or pkg is None:
        return dict(failure or fail_item("invalid_json"), steps=steps)
    version = pkg.get("version")
    if not check_version_report(version, "v1") or version["id"] != "v2":
        return fail_item("version_invalid", step="package", exit_code=0, steps=steps)
    if not check_version_ledger(ctx.work / ledger, "v2") or not check_version_ledger(
        ctx.work / ledger, "v1"
    ):
        return fail_item("version_ledger_invalid", step="package", exit_code=0, steps=steps)

    def infer_ok(tag: str, argv: list[str], command: str) -> dict[str, Any] | None:
        """exit 0・11・12 の infer（status との対応と判定の整合まで）。失敗なら失敗記録を返す。"""
        rc, o, f = run("infer", tag, argv, command)
        if f:
            return f
        # 校正つきでは対象外（11）・保留（12）も正常。終了コードと status の対応を照合する
        if rc not in INFER_EXIT_STATUS or o is None:
            return fail_item("unexpected_exit_code", step="infer", case=tag, exit_code=rc)
        if not (
            _infer_envelope_ok(o, rc)
            and o.get("id") == DEFAULT_TEXT_ID
            and check_infer_output(o, facts)
        ):
            return fail_item("unexpected_output", step="infer", case=tag, exit_code=rc)
        return None

    def infer_rejected(tag: str, argv: list[str], command: str) -> dict[str, Any] | None:
        """exit 64（`invalid_input`）の infer。それ以外は失敗記録を返す。"""
        rc, o, f = run("infer", tag, argv, command)
        if f:
            return f
        if rc != 64 or vocab_value((o or {}).get("code"), CODE_VOCAB) != "invalid_input":
            return fail_item("rejection_not_64", step="infer", case=tag, exit_code=rc)
        return None

    # 旧版（B の package）を新しい台帳の v1 で使う。台帳の最新版（v2）は新しい package で使う
    tampered = ctx.work / "J" / "tampered"
    try:
        shutil.copytree(bproj / "package", tampered, copy_function=shutil.copyfile)
        with open(tampered / "artifact.json", "ab") as f:
            f.write(b" ")
    except (OSError, shutil.Error):
        return fail_item("copy_failed", steps=steps)
    base = ["infer", "--text", sample]
    cmd = "infer --package <package> --version-ledger <ledger> --text <fixed-sample>"
    # 遅延実行（lambda）で順に判定し、最初に失敗した時点で止める（後続の infer は実行しない）
    checks = (
        lambda: infer_ok(
            "infer-rollback-v1",
            [*base, "--package", "B/project/package", "--version-ledger", ledger,
             "--version-id", "v1"],
            cmd + " --version-id v1",
        ),
        lambda: infer_ok(
            "infer-latest",
            [*base, "--package", f"{p}/package", "--version-ledger", ledger],
            cmd,
        ),
        lambda: infer_rejected(
            "infer-version-id-only",
            [*base, "--package", f"{p}/package", "--version-id", "v1"],
            "infer --package <package> --version-id v1 --text <fixed-sample>",
        ),
        lambda: infer_rejected(
            "infer-tampered-artifact",
            [*base, "--package", "J/tampered", "--version-ledger", ledger, "--version-id", "v1"],
            cmd + " --version-id v1",
        ),
    )  # fmt: skip
    for check in checks:
        failure = check()
        if failure:
            return dict(failure, steps=steps)
    return {
        "status": "ok",
        "steps": steps,
        "version_number": 2,
        "previous_version_number": 1,
        "rollback_to_v1_ok": True,
        "latest_ok": True,
        "version_id_only_exit_code": 64,
        "tampered_artifact_exit_code": 64,
    }


def load_average() -> list[float] | None:
    """load average（1・5・15 分）。取れなければ None。"""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        return None


def count_read_output(text: str) -> tuple[int, int]:
    """ログ中の `ReadOutputIncomplete` と、`Incomplete` が続かない `ReadOutput` の有無（#346）。"""
    incomplete = 1 if "ReadOutputIncomplete" in text else 0
    plain = 1 if re.search(r"ReadOutput(?!Incomplete)", text) else 0
    return incomplete, plain


TEST_RESULT_OK_RE = re.compile(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;")
LIST_TEST_LINE_RE = re.compile(r"^\S.*: test$")


def test_result_totals(text: str) -> tuple[int, int, int]:
    """ログの `test result: ok.` 行の (passed, failed, ignored) の合計。該当行が無ければ全部 0。"""
    rows = [tuple(int(n) for n in m) for m in TEST_RESULT_OK_RE.findall(text)]
    return (
        sum(r[0] for r in rows),
        sum(r[1] for r in rows),
        sum(r[2] for r in rows),
    )


def count_listed_tests(text: str) -> int:
    """`cargo test -- --list` の出力のテスト数（`<name>: test` の行。末尾の集計行は数えない）。"""
    return sum(1 for line in split_lines(text) if LIST_TEST_LINE_RE.match(line))


def item_f(ctx: Ctx) -> dict[str, Any]:
    """F: guard の time_limit テストを N 回（直列化しない。#346）。失敗しても全回続ける。

    先に `--no-run` で 1 回ビルドし（回数に数えない）、コールドビルドが 1 回目の上限を食わせない。
    """
    fdir = ctx.work / "F"
    fdir.mkdir()
    test_args = ["test", "--locked", "-p", "fandhe-edge-guard", "--test", "time_limit"]
    b = run_cmd(
        [ctx.cargo_cmd, *test_args, "--no-run"],
        ctx.repo,
        fdir / "build.log",
        fdir / "build.err",
        TIMEOUT_BUILD,
        CAP_LOG_STDOUT,
        CAP_LOG_STDERR,
        ctx.offline_env,
    )
    if b.reason is not None or b.exit_code != 0:
        return fail_item("build_failed", exit_code=b.exit_code)
    # 1 回あたりに実行されるべき件数を `--list` から得る（固定値と比べず、テストの増減に追随する）。
    # 回数には数えず、ビルド済みなので短い上限で足りる
    lst = run_cmd(
        [ctx.cargo_cmd, *test_args, "--", "--list"],
        ctx.repo,
        fdir / "list.log",
        fdir / "list.err",
        TIMEOUT_CARGO_TEST,
        CAP_LOG_STDOUT,
        CAP_LOG_STDERR,
        ctx.offline_env,
    )
    if lst.reason is not None or lst.exit_code != 0:
        return fail_item("list_failed", exit_code=lst.exit_code)
    expected = count_listed_tests(read_capped(fdir / "list.log", CAP_LOG_STDOUT) or "")
    if expected < 1:
        return fail_item("no_tests_listed")
    load_start = load_average()
    passed = failed = timeouts = no_tests = count_mismatch = 0
    output_limit = killed = spawn_error = incomplete = plain = 0
    unreaped = False
    started = 0  # 実際に開始した回数（途中で打ち切ると repeat 未満になる）
    for i in range(1, ctx.repeat + 1):
        started += 1
        log = fdir / f"run-{i:04d}.log"
        r = run_cmd(
            [ctx.cargo_cmd, *test_args],
            ctx.repo,
            log,
            fdir / f"run-{i:04d}.err",
            TIMEOUT_CARGO_TEST,
            CAP_LOG_STDOUT,
            CAP_LOG_STDERR,
            ctx.offline_env,
        )
        if r.reason == "timeout":
            timeouts += 1
        elif r.reason == "output_limit":
            output_limit += 1
        elif r.reason == "killed":
            killed += 1
        elif r.reason == "spawn_error":
            spawn_error += 1
        if r.reason == REASON_UNREAPED:
            # 子が残っている可能性がある。次の回を始めず、この回を失敗として打ち切る（fail-closed）
            failed += 1
            unreaped = True
            break
        text = read_capped(log, CAP_LOG_STDOUT) or ""
        ran, ran_failed, ran_ignored = test_result_totals(text)
        # 合格は「exit 0・失敗なし・passed が `--list` の件数と一致・ignored 0」
        if (
            r.exit_code == 0
            and r.reason is None
            and ran == expected
            and ran_failed == 0
            and ran_ignored == 0
        ):
            passed += 1
            continue
        failed += 1
        if r.exit_code == 0 and r.reason is None:
            if ran == 0 and ran_failed == 0:
                no_tests += 1
            else:
                count_mismatch += 1
        a, p = count_read_output(text)
        incomplete += a
        plain += p
    rec = {
        "status": "ok" if failed == 0 else "failed",
        "runs": started,
        "expected_tests": expected,
        "passed": passed,
        "failed": failed,
        "timeouts": timeouts,
        "no_tests": no_tests,
        "count_mismatch": count_mismatch,
        "output_limit": output_limit,
        "killed": killed,
        "spawn_error": spawn_error,
        "read_output": plain,
        "read_output_incomplete": incomplete,
        "load_start": load_start,
        "load_end": load_average(),
    }
    if failed:
        rec["reason"] = REASON_UNREAPED if unreaped else "test_failures"
    return rec


# --------------------------------------------------------------------------------------
# 環境の採取・入力の記録
# --------------------------------------------------------------------------------------


def _probe(work: Path, repo: Path, argv: list[str], name: str) -> str | None:
    """短い出力のコマンドを実行して先頭を返す。使えない・失敗は None。

    `argv[0]` は論理名で、固定パスへ解決する（PATH は探さない）。子には最小の環境だけを渡す
    （`resolve_tool`・`probe_env`。REQ-38・REQ-39）。
    """
    exe = resolve_tool(argv[0])
    if exe is None:
        return None
    pdir = work / ".probe"
    pdir.mkdir(exist_ok=True)
    r = run_cmd(
        [exe, *argv[1:]],
        repo,
        pdir / (name + ".out"),
        pdir / (name + ".err"),
        TIMEOUT_PROBE,
        CAP_PROBE,
        CAP_PROBE,
        probe_env(),
    )
    if r.reason is not None or r.exit_code != 0:
        return None
    text = read_capped(pdir / (name + ".out"), CAP_PROBE)
    return text.strip() if text is not None else None


def _worktree_clean(work: Path, repo: Path) -> bool | None:
    """`git status --porcelain` が空か。出力が上限を超えれば汚れている扱い。取れなければ None。"""
    git = resolve_tool("git")
    if git is None:
        return None
    pdir = work / ".probe"
    pdir.mkdir(exist_ok=True)
    r = run_cmd(
        [git, "status", "--porcelain"],
        repo,
        pdir / "status.out",
        pdir / "status.err",
        TIMEOUT_PROBE,
        CAP_PROBE,
        CAP_PROBE,
        probe_env(),
    )
    if r.reason == "output_limit":
        return False
    if r.exit_code == 0 and r.reason is None:
        return _size(pdir / "status.out") == 0
    return None


def env_text(s: str | None) -> str | None:
    """環境の文字列欄。`ENV_TEXT_RE`（64 字以内の安全な文字）に一致しなければ None。"""
    return s if s is not None and ENV_TEXT_RE.fullmatch(s) else None


def ascii_int(s: str | None) -> int | None:
    """ASCII の 10 進数字 1〜20 桁だけの文字列を整数へ。違えば None（`isdigit` は使わない）。"""
    return int(s) if s is not None and ASCII_INT_RE.fullmatch(s) else None


def collect_volatile(ctx: Ctx) -> dict[str, Any]:
    """実行中に変わりうる値（commit・worktree_clean・CLI の sha256）の採取。

    開始時（`collect_environment`）と終了時（`run`）で同じ関数を使い、採取の規則を 1 箇所に
    集約する。取れない欄は None（#360）。
    """
    commit = _probe(ctx.work, ctx.repo, ["git", "rev-parse", "HEAD"], "commit")
    if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", commit):
        commit = None
    return {
        "commit": commit,
        "worktree_clean": _worktree_clean(ctx.work, ctx.repo),
        "cli_sha256": sha256_file(ctx.bin, CAP_CLI_BINARY),
    }


def compare_start_end(start: Any, end: Any) -> bool | None:
    """開始時と終了時の値の比較（fail-closed）。

    両方 None は比較不能で None。片方だけ None は同一と確認できないので False。
    両方あれば一致で True、不一致で False（#360）。
    """
    if start is None and end is None:
        return None
    if start is None or end is None:
        return False
    return bool(start == end)


def judge_environment_stable(env: dict[str, Any]) -> bool | None:
    """3 つの比較結果の集約（fail-closed）。1 つでも False なら False、全部 True のときだけ True。

    一部でも None（両端とも取れず確認できない）が残れば None（未確認）。None は成功扱いにしない
    （`run` は stable が True でなければ exit 10 にする。#360）。
    """
    flags = [env.get(k) for k in ("commit_unchanged", "worktree_clean_unchanged", "cli_unchanged")]
    if any(f is False for f in flags):
        return False
    if all(f is True for f in flags):
        return True
    return None


def fill_end_environment(env: dict[str, Any], end: dict[str, Any] | None) -> None:
    """終了時の採取値と比較結果を環境の欄へ入れる。end が None（未採取）なら比較も None。"""
    if end is None:
        env["commit_end"] = None
        env["worktree_clean_end"] = None
        env["cli_end_sha256"] = None
        env["commit_unchanged"] = None
        env["worktree_clean_unchanged"] = None
        env["cli_unchanged"] = None
    else:
        env["commit_end"] = end["commit"]
        env["worktree_clean_end"] = end["worktree_clean"]
        env["cli_end_sha256"] = end["cli_sha256"]
        env["commit_unchanged"] = compare_start_end(env.get("commit"), end["commit"])
        env["worktree_clean_unchanged"] = compare_start_end(
            env.get("worktree_clean"), end["worktree_clean"]
        )
        env["cli_unchanged"] = compare_start_end(env.get("cli_sha256"), end["cli_sha256"])
    env["stable"] = judge_environment_stable(env)


def collect_environment(ctx: Ctx, cli_profile: str | None) -> dict[str, Any]:
    """環境の採取。macOS 以外・コマンド不在で取れない欄は null。"""
    w, rp = ctx.work, ctx.repo
    on_mac = sys.platform == "darwin"

    def sysctl(key: str, name: str) -> str | None:
        """macOS の sysctl 値（macOS 以外は None）。"""
        return _probe(w, rp, ["sysctl", "-n", key], name) if on_mac else None

    def sw_vers(flag: str, name: str) -> str | None:
        """macOS の sw_vers 値（macOS 以外は None）。"""
        return _probe(w, rp, ["sw_vers", flag], name) if on_mac else None

    vol = collect_volatile(ctx)
    try:
        cli_bytes: int | None = ctx.bin.stat().st_size
    except OSError:
        cli_bytes = None
    return {
        "hw_model": env_text(sysctl("hw.model", "hw")),
        "cpu": env_text(sysctl("machdep.cpu.brand_string", "cpu")),
        "ncpu": ascii_int(sysctl("hw.ncpu", "ncpu")),
        "memory_bytes": ascii_int(sysctl("hw.memsize", "mem")),
        "os_name": env_text(sw_vers("-productName", "osn")),
        "os_version": env_text(sw_vers("-productVersion", "osv")),
        "os_build": env_text(sw_vers("-buildVersion", "osb")),
        "commit": vol["commit"],
        "worktree_clean": vol["worktree_clean"],
        "started_local": None,
        "ended_local": None,
        "cli_sha256": vol["cli_sha256"],
        "cli_bytes": cli_bytes,
        "cli_profile": cli_profile,
        # 以下は終了時の再採取と比較（#360）。`run` が埋める。未採取は null
        "commit_end": None,
        "worktree_clean_end": None,
        "cli_end_sha256": None,
        "commit_unchanged": None,
        "worktree_clean_unchanged": None,
        "cli_unchanged": None,
        "stable": None,
        # CLI・trainer の出所（閉じた語彙。パスは記録しない）
        "cli_origin": "env_override" if ctx.bin_override else "built_by_script",
        # Rust 側 `worker_launcher` の判定（var_os の is_some。空文字も設定扱い）と対応させる
        "trainer_origin": (
            "env" if "FANDHE_EDGE_TRAINER_DIR" in ctx.offline_env else "build_default"
        ),
    }


def collect_inputs(repo: Path) -> dict[str, Any] | None:
    """fixtures の 3 ファイルの件数・sha256。読めなければ None。"""
    src = repo / "fixtures" / "sandbox_run_eval"
    out: dict[str, Any] = {}
    for fname, key in (
        ("train.jsonl", "train"),
        ("evaluation.jsonl", "evaluation"),
        ("definition.json", "definition"),
    ):
        p = src / fname
        check_overall()  # 入力採取も全体の期限の内（REQ-39）
        # 通常ファイルだけを先にハッシュし（FIFO 等は None で止まる）、期限を見ながら読む
        digest = sha256_file(p, CAP_INPUT_FILE, deadline_check=True)
        text = _read_regular_capped(p, CAP_INPUT_FILE) if digest is not None else None
        check_overall()
        if text is None or digest is None:
            return None
        if key != "definition":
            out[key + "_records"] = sum(1 for ln in split_lines(text) if ln.strip())
        out[key + "_sha256"] = digest
    return {
        "train_records": out["train_records"],
        "evaluation_records": out["evaluation_records"],
        "definition_sha256": out["definition_sha256"],
        "train_sha256": out["train_sha256"],
        "evaluation_sha256": out["evaluation_sha256"],
    }


# --------------------------------------------------------------------------------------
# record.md
# --------------------------------------------------------------------------------------


_LINE_BREAKS = frozenset("\r\n\v\f\x85\u2028\u2029\t")


def strip_control_chars(s: str) -> str:
    """record.md へ入る文字列から制御文字・bidi 制御・ゼロ幅等（Cc・Cf・Cs）を除く。

    行区切り類とタブは空白 1 個へ置く。表示の欺瞞（Trojan Source 系）と行の増殖を止める（#361）。
    """
    out = []
    for ch in s:
        if ch in _LINE_BREAKS:
            out.append(" ")
        elif unicodedata.category(ch) in ("Cc", "Cf", "Cs"):
            continue
        else:
            out.append(ch)
    return "".join(out)


def _cell(v: Any) -> str:
    """Markdown・HTML として解釈されないよう、セルの値をエスケープする。"""
    if v is None:
        return "-"
    s = strip_control_chars(str(v))
    for a, b in (
        ("&", "&amp;"),
        ("<", "&lt;"),
        (">", "&gt;"),
        ("`", "&#96;"),
        ("|", "&#124;"),
        ("\r", " "),
        ("\n", " "),
    ):
        s = s.replace(a, b)
    return s


def render_markdown(rec: dict[str, Any]) -> str:
    """record.json から貼り付け用の Markdown を作る（`sanitize_record` 済みの値だけを使う）。"""
    env = rec["environment"]
    hint = rec["evidence_hint"]
    lines = [
        "# real-machine-check 記録",
        "",
        f"- schema: {_cell(rec['schema'])}",
        f"- bin_override: {_cell(rec.get('bin_override'))}",
        f"- evidence_hint: {_cell(hint)}",
        f"- harness（make・cargo の代役）: {'はい' if hint == 'test_harness' else 'いいえ'}",
        "- 証拠の種別: 人が確認して記入",
    ]
    if rec.get("bin_override"):
        lines.append(
            "- 注意: CLI を `FANDHE_EDGE_BIN` で差し替えた"
            "（このスクリプトがビルドした CLI ではない。`commit`・`worktree_clean` は"
            " CLI の出所を表さない）"
        )
    if rec.get("child_may_remain") is True:
        lines.append(
            "- 注意: 子プロセスの回収が上限時間内に終わらなかった（子・孫が残っている可能性。"
            "`ps` で確認し、残っていれば手で止める。結果を採用しない）"
        )
    cut_items = any(
        it.get("reason") == REASON_OVERALL_TIMEOUT
        for it in rec["items"].values()
        if isinstance(it, dict)
    )
    if cut_items:
        lines.append(
            "- 注意: 実行全体の上限時間（`options.overall_timeout_sec`）を超えて打ち切った"
            "（実行中の項目は failed、残りは not_run。結果を採用しない）"
        )
    elif rec.get("overall_timeout_exceeded") is True:
        lines.append(
            "- 注意: 実行全体の上限時間（`options.overall_timeout_sec`）を"
            "全項目の完了後に超えていた（各項目の status は変えていない。結果を採用しない）"
        )
    stable = (rec.get("environment") or {}).get("stable")
    if stable is False:
        lines.append(
            "- 注意: 開始時と終了時で commit・worktree_clean・CLI の sha256 のいずれかが"
            "一致しない（実行中に環境が変わった。結果を採用しない）"
        )
    elif rec.get("environment") is not None and stable is None:
        lines.append(
            "- 注意: 開始時と終了時の commit・worktree_clean・CLI の sha256 の一致を"
            "確認できなかった（採取不能。成功扱いにしない）"
        )
    # 未採取の理由は中断（シグナル）と全体の上限時間の超過で書き分ける
    if rec.get("overall_timeout_exceeded") is True or any(
        it.get("reason") == REASON_OVERALL_TIMEOUT
        for it in rec["items"].values()
        if isinstance(it, dict)
    ):
        not_collected = "項目の開始前に全体の上限時間（overall_timeout）を超えたため未採取"
    else:
        not_collected = "項目の開始前に中断されたため未採取"
    lines += ["", "## 環境", "", "| 項目 | 値 |", "| ---- | -- |"]
    if env is None:
        lines.append(f"| (not collected) | {not_collected} |")
    for k, v in (env or {}).items():
        lines.append(f"| {_cell(k)} | {_cell(v)} |")
    lines += ["", "## 入力と指定", "", "| 項目 | 値 |", "| ---- | -- |"]
    if rec.get("inputs") is None:
        lines.append(f"| (not collected) | {not_collected} |")
    for k, v in (rec.get("inputs") or {}).items():
        lines.append(f"| {_cell(k)} | {_cell(v)} |")
    for k, v in rec["options"].items():
        lines.append(f"| {_cell(k)} | {_cell(v)} |")
    lines += ["", "## 項目ごとの結果", "", "| 項目 | status | 要点 |", "| ---- | ------ | ---- |"]
    for name in ITEM_ORDER:
        it = rec["items"][name]
        detail = {
            k: v
            for k, v in it.items()
            if k
            not in (
                "status",
                "steps",
                "package_files",
                "capacity",
                "p95",
                "capacity_limit",
                "message_sha256",
            )
        }
        if name == "B" and "capacity" in it:
            detail["total_bytes"] = it["capacity"]["total_bytes"]
            detail["capacity_sum_matches_total"] = it.get("capacity_sum_matches_total")
        if name == "C":
            if "p95" in it:
                detail["p95"] = it["p95"]
            if "capacity_limit" in it:
                detail["capacity_limit"] = it["capacity_limit"]
        text = json.dumps(detail, ensure_ascii=False, sort_keys=True)
        lines.append(f"| {name} | {_cell(it['status'])} | {_cell(text)} |")
    d_item = rec["items"]["D"]
    lines += [
        "",
        "- D のリンク検査対象と実行した CLI の一致: "
        f"{_cell(d_item.get('linkage_target_matches_cli'))}"
        f"（検査対象 sha256: {_cell(d_item.get('linkage_target_sha256'))}）",
    ]
    lines += [
        "",
        "## 検証済みと扱わない点",
        "",
        "- `not_run` の項目は実行していない。成功扱いにしない",
        "- `reference_only` の p95 は静かな状態の申告が無い参考値",
        "- このスクリプトは測定と整形まで。「実機」の証拠としての確定と合否の判断は人が行う",
        "- 実行環境の通信 0 件（REQ-38）は対象外（`sandbox-monitor.sh` の担当）",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# run エントリ
# --------------------------------------------------------------------------------------


def emit(code: str, message: str, exit_code: int, with_record: bool) -> int:
    """stdout へ固定メッセージの JSON を 1 行だけ出す（パスは書かない）。

    書き込み・フラッシュに失敗した（読み手が閉じたパイプ・書き込めないファイル・
    閉じた stdout）場合は、本来の `exit_code` によらず 70（`runtime_error`）を返す。
    約束した JSON を渡せていないのに 0 を返さず、終了コードを 7 種の内に保つため
    （REQ-21・REQ-33）。`record.json`・`record.md` は `emit` の前に書き終えているので
    失敗の影響を受けない。例外は投げない（`main` の例外経路からの再呼び出しでも
    落ちないため）。
    """
    global _final_emitted, _stdout_failed
    _final_emitted = True  # 以後の 2 回目のシグナルが 2 行目を書かないようにする
    obj: dict[str, Any] = {"code": code, "message": message}
    if with_record:
        obj["record"] = "record.json"
    try:
        sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError, AttributeError):
        # OSError は BrokenPipeError・ENOSPC を含む。ValueError は閉じたファイル、
        # AttributeError は `sys.stdout is None`（fd 1 が閉じた起動）。
        _stdout_failed = True
        return EXIT_RUNTIME_ERROR
    return exit_code


def _silence_stdout_for_exit() -> None:
    """書き込み失敗後に fd 1 を /dev/null へ向ける（プロセスの終了直前にだけ呼ぶ）。

    失敗した行がバッファに残っていると、終了処理のフラッシュが再び失敗して exit 120 と
    `Exception ignored` の出力になる。残りを /dev/null へ流して 70 を保つ。`emit`・`run` の中では
    呼ばない（同じプロセスで `run` を呼ぶ pytest の fd 1 を壊さないため）。REQ-21。
    """
    if not _stdout_failed:
        return
    try:
        fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(fd, 1)
        finally:
            os.close(fd)
    except OSError:
        pass


def _on_signal(signum: int, frame: Any) -> None:
    """中断シグナルのハンドラ。1 回目は印を立てて戻るだけ（例外を投げない。理由は `run_cmd`）。

    2 回目以降は後始末の完了を待たず強制終了する（後始末が終わらないときに SIGKILL 以外で
    止める口。REQ-39）。子のグループへ KILL を送り、固定 JSON を 1 行書いて exit 70 する。
    `Popen` の内部（`_waitpid_lock`）に触れず、例外も投げない。record は書かない。
    """
    global _interrupt_requested, _signal_count, _force_pending
    _interrupt_requested = True
    _signal_count += 1
    if _signal_count < 2:
        return
    if _final_emitted:
        return  # 最終の JSON を書き始めた後。後始末は済んでいるので 2 行目は書かない
    if _spawning:
        _force_pending = True  # pgid が未登録。登録の直後に `run_cmd` が強制終了する
        return
    _force_exit(_active_pgid)


def _force_exit(pgid: int | None) -> None:
    """子のグループへ KILL を送り、固定 JSON を 1 行書いて exit 70 する（戻らない）。"""
    for target in [pgid, *(p.pid for p in _leftover_procs)]:
        if target is None:
            continue
        try:
            os.killpg(target, signal.SIGKILL)
        except OSError:
            pass
    try:
        os.write(1, FORCED_EXIT_LINE)
    except OSError:
        pass
    os._exit(EXIT_RUNTIME_ERROR)


def _warn_child_may_remain() -> None:
    """子が残っている可能性を stderr へ 1 行出す（`_child_may_remain` が立っているときだけ）。"""
    if _child_may_remain:
        sys.stderr.write("a child process may remain\n")
        sys.stderr.flush()


def build_cli(ctx: Ctx) -> Path | None:
    """`cargo build --locked --release` でビルドし、`compiler-artifact` の executable を返す。

    出力先は決め打ちせず、cargo が報告した実際のパスを使う（`CARGO_BUILD_TARGET_DIR` や
    `build.target-dir` に追従する）。取れなければ None。`--locked` と `CARGO_NET_OFFLINE` で
    通信しない（REQ-38）。
    """
    bdir = ctx.work / ".build"
    bdir.mkdir(exist_ok=True)
    log = bdir / "build.json"
    r = run_cmd(
        [
            ctx.cargo_cmd,
            "build",
            "--locked",
            "--release",
            "-p",
            "fandhe-edge-cli",
            "--bin",
            "fandhe-edge",
            "--message-format=json",
        ],
        ctx.repo,
        log,
        bdir / "build.err",
        TIMEOUT_BUILD,
        CAP_LOG_STDOUT,
        CAP_LOG_STDERR,
        ctx.offline_env,
    )
    if r.reason is not None or r.exit_code != 0:
        return None
    return find_built_executable(read_capped(log, CAP_LOG_STDOUT) or "")


def find_built_executable(text: str) -> Path | None:
    """cargo の `--message-format=json` 出力から bin `fandhe-edge` の executable（最後の 1 件）。"""
    exe = None
    for line in split_lines(text):
        try:
            v = _loads(line)
        except ValueError:
            continue
        if not isinstance(v, dict) or v.get("reason") != "compiler-artifact":
            continue
        target = v.get("target")
        if not isinstance(target, dict) or target.get("name") != "fandhe-edge":
            continue
        if "bin" in (target.get("kind") or []) and isinstance(v.get("executable"), str):
            exe = v["executable"]
    if exe is None:
        return None
    p = Path(exe)
    return p if p.is_absolute() and p.is_file() else None


def run_item(ctx: Ctx, name: str, b_ok: bool) -> tuple[dict[str, Any], bool]:
    """項目 1 つを実行する。戻り値は (結果, B の成否)。"""
    if name == "A":
        return item_a(ctx), b_ok
    if name == "B":
        return item_b(ctx)
    if name == "C":
        return item_c(ctx), b_ok
    if name == "D":
        return item_d(ctx), b_ok
    if name == "E":
        return item_e(ctx), b_ok
    if name == "F":
        return item_f(ctx), b_ok
    if name == "G":
        return item_g(ctx), b_ok
    if name == "H":
        return item_h(ctx), b_ok
    if name == "I":
        return item_i(ctx), b_ok
    return item_j(ctx), b_ok


def run(args: argparse.Namespace) -> int:
    """全体の上限時間（`--overall-timeout-sec`）を設けて `_run` を実行する。

    期限は入力の採取・CLI のビルド・開始時の環境採取も含めて数える。戻るとき（例外を含む）に
    期限を必ず下ろし、次の呼び出しへ持ち越さない（REQ-39）。
    """
    global _overall_deadline
    clear_overall()
    _overall_deadline = time.monotonic() + args.overall_timeout_sec
    try:
        return _run(args)
    finally:
        clear_overall()


def _run(args: argparse.Namespace) -> int:
    """項目を A→F の順に実行し、record.json・record.md を書く。

    記録の骨格は入力の採取より前に作る。前半（採取・ビルド）で中断されても、選んだ項目を
    すべて `not_run`（`interrupted`）にした記録を書く（REQ-21・REQ-39）。
    """
    global _interrupt_requested, _signal_count, _active_pgid, _child_may_remain
    global _spawning, _force_pending, _final_emitted
    _interrupt_requested = False  # 前の回の印を引き継がない
    _signal_count = 0
    _spawning = False
    _force_pending = False
    _final_emitted = False
    _active_pgid = None
    _child_may_remain = False
    _leftover_procs.clear()
    os.umask(0o077)
    for s in INTERRUPT_SIGNALS:
        signal.signal(s, _on_signal)
    items = [x for x in ITEM_ORDER if x in args.items.split(",")]
    make_cmd = os.environ.get("FANDHE_EDGE_MAKE_CMD") or "make"
    cargo_cmd = os.environ.get("FANDHE_EDGE_CARGO_CMD") or "cargo"
    harness = bool(
        os.environ.get("FANDHE_EDGE_MAKE_CMD") or os.environ.get("FANDHE_EDGE_CARGO_CMD")
    )
    ctx = Ctx(
        repo=Path(args.repo_root),
        work=Path(args.work_dir),
        bin=Path(args.bin) if args.bin else Path(),
        make_cmd=make_cmd,
        cargo_cmd=cargo_cmd,
        p95_limit_us=args.p95_limit_us,
        package_limit_bytes=args.package_limit_bytes,
        quiet_machine=args.quiet_machine,
        repeat=args.repeat,
        harness=harness,
        offline_env=make_offline_env(),
        ci_env=make_ci_env(),
    )
    ctx.bin_override = bool(args.bin_override)
    ctx.g_budget_seconds = args.g_budget_seconds
    ctx.i_device = args.i_device
    rec: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_hint": "test_harness" if harness else "requires_human_review",
        "bin_override": bool(args.bin_override),
        "environment": None,
        "inputs": None,
        "options": {
            "items": items,
            "repeat": ctx.repeat,
            "quiet_machine": ctx.quiet_machine,
            "p95_limit_us": ctx.p95_limit_us,
            "package_limit_bytes": ctx.package_limit_bytes,
            "overall_timeout_sec": args.overall_timeout_sec,
            "with_ci": bool(args.with_ci),
            "g_budget_seconds": ctx.g_budget_seconds,
            "i_device": ctx.i_device,
            "cargo_offline": "A" not in items,
        },
        "child_may_remain": False,
        "items": {},
    }
    stopped = False
    interrupted = False
    internal_error = False
    exceeded = False  # 全体の上限時間を超えた（超過を受けたら期限を下ろし、以降の再採取は期限の外）
    b_ok = False
    try:
        inputs = collect_inputs(ctx.repo)
        if inputs is None:
            _warn_child_may_remain()
            return emit("runtime_error", "cannot read fixture inputs", EXIT_RUNTIME_ERROR, False)
        rec["inputs"] = inputs
        cli_profile: str | None = None
        if not args.bin_override:
            built = build_cli(ctx)
            if built is None:
                _warn_child_may_remain()  # 回収を諦めたビルドが残っていれば知らせる
                return emit("runtime_error", "cannot build the CLI", EXIT_RUNTIME_ERROR, False)
            ctx.bin = built
            cli_profile = "release"
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        env = collect_environment(ctx, cli_profile)
        env["started_local"] = started
        ctx.cli_sha256 = env["cli_sha256"]
        rec["environment"] = env
    except Interrupted:
        interrupted = True  # 前半の中断。以降の項目はすべて not_run（interrupted）
    except OverallTimeout:
        exceeded = True  # 前半の上限超過。以降の項目はすべて not_run（overall_timeout）
        clear_overall()
    for name in ITEM_ORDER:
        if name in items and _interrupt_requested:
            interrupted = True  # 項目の開始前に印を見る
        if name in items and not exceeded and not interrupted and overall_expired():
            exceeded = True  # 項目の開始前に全体の期限も見る
            clear_overall()
        if name not in items:
            rec["items"][name] = {"status": "not_run", "reason": "not_selected"}
        elif interrupted or exceeded or stopped:
            if interrupted:
                reason = "interrupted"
            elif exceeded:
                reason = REASON_OVERALL_TIMEOUT
            else:
                reason = "previous_item_failed"
            rec["items"][name] = {"status": "not_run", "reason": reason}
        else:
            sys.stderr.write(f"running item {name}\n")
            sys.stderr.flush()
            try:
                res, b_ok = run_item(ctx, name, b_ok)
            except Interrupted:
                res, interrupted = fail_item("interrupted"), True
            except OverallTimeout:
                res, exceeded = fail_item(REASON_OVERALL_TIMEOUT), True
                clear_overall()
            except Exception as e:
                res = fail_item("internal_error", error_type=error_type_name(e))
                internal_error = True
            rec["items"][name] = res
            if res["status"] != "ok":
                stopped = True
    # 記録を書く直前に印を 1 回だけ読む。書き出し中に届いた中断は結果を変えない（印を立てるだけ）
    if _interrupt_requested:
        interrupted = True
    # 期限を下ろす前に超過を判定する。最後の項目の終了後に期限を過ぎていた場合も成功にしない
    # （REQ-39）。項目自身の status は書き換えず、超過は別軸（`overall_timeout_exceeded`）で記録する
    if not exceeded and not interrupted and overall_expired():
        exceeded = True
    if exceeded:
        rec["overall_timeout_exceeded"] = True
    clear_overall()  # 終了時の再採取と記録の書き出しは期限の外で行う（所要は子ごとの上限で頭打ち）
    if rec["environment"] is not None:
        end_vol: dict[str, Any] | None = None
        if not _interrupt_requested:
            try:
                end_vol = collect_volatile(ctx)
            except Interrupted:
                interrupted = True  # 採取中の中断。採取分は捨てる
        fill_end_environment(rec["environment"], end_vol)
        rec["environment"]["ended_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if _child_may_remain:
        rec["child_may_remain"] = True
        _warn_child_may_remain()
    rec = sanitize_record(rec)
    # schema は固定定数（`/` を含む）なので伏せ処理の対象外にして戻す
    rec["schema"] = SCHEMA
    try:
        write_atomic(
            ctx.work / "record.json",
            json.dumps(rec, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        )
        write_atomic(ctx.work / "record.md", render_markdown(rec))
    except (OSError, ValueError):
        return emit("runtime_error", "cannot write the record", EXIT_RUNTIME_ERROR, False)
    if interrupted:
        return emit("runtime_error", "interrupted", EXIT_RUNTIME_ERROR, True)
    if internal_error:
        # 想定外の例外はスクリプト自身の実行不能（70）。record は書いてある
        return emit("runtime_error", "internal error", EXIT_RUNTIME_ERROR, True)
    if _child_may_remain and exceeded:
        # 回収を諦めた子が残りうる場合は、期限超過（10）に隠さず実行不能（70）を返す（REQ-39）。
        # record は書いてある（`overall_timeout_exceeded` も立っている）
        return emit("runtime_error", "a child process may remain", EXIT_RUNTIME_ERROR, True)
    if exceeded:
        return emit("judged_fail", "overall time limit exceeded", EXIT_JUDGED_FAIL, True)
    if (rec["environment"] or {}).get("stable") is not True:
        # 項目の status は書き換えず、環境の食い違い・未確認（採取不能）を別軸の失敗として返す
        # （成功扱いにしない。#360）
        return emit("judged_fail", "environment changed during the run", EXIT_JUDGED_FAIL, True)
    if all(rec["items"][n]["status"] == "ok" for n in items):
        if rec["child_may_remain"]:
            # 子が残りうる実行は採取経路（環境採取の sysctl・sw_vers 等を含む）を問わず成功にしない
            # （資源上限。REQ-39）。record は書いてある
            return emit("runtime_error", "a child process may remain", EXIT_RUNTIME_ERROR, True)
        return emit("ok", "all requested items completed", EXIT_OK, True)
    return emit(
        "judged_fail", "one or more requested items failed or were not run", EXIT_JUDGED_FAIL, True
    )


class _ArgParser(argparse.ArgumentParser):
    """エラーを固定メッセージの JSON（invalid_input・64）で返す。引数の値は出力に含めない。"""

    def error(self, message: str) -> Any:  # message には引数の値が入りうるため出さない
        sys.exit(emit("invalid_input", "invalid arguments", EXIT_INVALID_INPUT, False))


def validate_args(args: argparse.Namespace) -> str | None:
    """引数の検証（シェル側と同じ条件の再確認）。問題があれば固定メッセージ、無ければ None。

    数値の範囲（`--repeat`・`--p95-limit-us`・`--package-limit-bytes` の 15 桁上限・
    `--overall-timeout-sec`）もシェル側と同じ値で確認する。
    """
    parts = args.items.split(",")
    if not parts or any(p not in ITEM_ORDER for p in parts) or len(set(parts)) != len(parts):
        return ITEMS_MESSAGE
    if "A" in parts and not args.with_ci:
        return "item A requires --with-ci"
    # E は B の成果物（package/ と train.jsonl）を使う。B が無ければ起動前に拒否する
    if "E" in parts and "B" not in parts:
        return "item E requires item B"
    # J は B の `project`（package 済み）を旧プロジェクトに使う
    if "J" in parts and "B" not in parts:
        return "item J requires item B"
    if not 1 <= args.repeat <= MAX_REPEAT:
        return "--repeat must be an integer from 1 to 1000"
    if not 1 <= args.p95_limit_us <= MAX_P95_LIMIT_US:
        return "--p95-limit-us must be an integer from 1 to 3600000000"
    if not 1 <= args.package_limit_bytes <= MAX_PACKAGE_LIMIT_BYTES:
        return "--package-limit-bytes must be a positive integer"
    if not MIN_OVERALL_TIMEOUT_SEC <= args.overall_timeout_sec <= MAX_OVERALL_TIMEOUT_SEC:
        return "--overall-timeout-sec must be an integer from 1 to 86400"
    if not 1 <= args.g_budget_seconds <= MAX_G_BUDGET_SECONDS:
        return "--g-budget-seconds must be an integer from 1 to 921600"
    if args.i_device not in I_EVIDENCE:
        return "--i-device must be cpu or gpu"
    if args.bin_override and not args.bin:
        return "--bin-override requires --bin"
    if args.bin_override:
        b = Path(args.bin)
        if not b.is_absolute() or not b.is_file() or not os.access(b, os.X_OK):
            return "FANDHE_EDGE_BIN must be a path to an executable file"
    return None


def main(argv: list[str] | None = None) -> int:
    """エントリポイント。想定外の例外・中断は traceback を出さず固定メッセージにする。"""
    p = _ArgParser(prog="real_machine_check_record.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo-root", required=True)
    r.add_argument("--work-dir", required=True)
    r.add_argument("--bin", default="")
    r.add_argument("--bin-override", action="store_true")
    r.add_argument("--items", required=True)
    r.add_argument("--repeat", type=int, required=True)
    r.add_argument("--p95-limit-us", type=int, required=True)
    r.add_argument("--package-limit-bytes", type=int, required=True)
    r.add_argument("--overall-timeout-sec", type=int, required=True)
    r.add_argument("--g-budget-seconds", type=int, default=DEFAULT_G_BUDGET_SECONDS)
    r.add_argument("--i-device", default="cpu")
    r.add_argument("--quiet-machine", action="store_true")
    r.add_argument("--with-ci", action="store_true")
    ns = p.parse_args(argv)
    problem = validate_args(ns)
    if problem is not None:
        return emit("invalid_input", problem, EXIT_INVALID_INPUT, False)
    try:
        return run(ns)
    except Interrupted:
        return emit("runtime_error", "interrupted", EXIT_RUNTIME_ERROR, False)
    except OverallTimeout:  # 取りこぼしても traceback を出さない
        return emit("runtime_error", "internal error", EXIT_RUNTIME_ERROR, False)
    except Exception:
        return emit("runtime_error", "internal error", EXIT_RUNTIME_ERROR, False)


def _ignore_interrupt_signals() -> None:
    """中断シグナルをすべて無視へ設定する（プロセスの終了直前にだけ呼ぶ）。

    最終の JSON を出した後、インタプリタの終了処理は Python のハンドラを既定の動作へ戻す。
    その窓へ 2 回目のシグナルが届くと exit 70 でなくシグナルで終わるため、明示的に
    `SIG_IGN` にして終了コードを 70 に保つ（`SIG_IGN` は終了処理で既定へ戻されない）。
    `run`・`emit` の中では呼ばない（同じプロセスで `run` を呼ぶ pytest が中断を受けられなくなる）。
    REQ-21・REQ-39。
    """
    for s in INTERRUPT_SIGNALS:
        try:
            signal.signal(s, signal.SIG_IGN)
        except (OSError, ValueError):
            pass


def _entry(argv: list[str] | None = None) -> int:
    """プロセスの入口。`main` から抜けるすべての経路（SystemExit・例外を含む）で無視へ設定する。"""
    try:
        return main(argv)
    finally:
        _ignore_interrupt_signals()
        _silence_stdout_for_exit()


if __name__ == "__main__":
    sys.exit(_entry())
