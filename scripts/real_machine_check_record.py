"""Mac 実機での動作確認（項目 A〜F）の実行・要約・記録（record.json / record.md）の生成。

REQ-21・REQ-28・REQ-30・REQ-31・REQ-32・REQ-33・REQ-38・REQ-39。特定の TASK には対応しない横断の
確認ツール。REQ-38 のうち sandbox 下の通信 0 件の判定は対象外（`sandbox-monitor.sh` の担当）。
本スクリプトは通信を起こさない側に倒す（cargo は `--locked`、A 以外の子には `CARGO_NET_OFFLINE`）。

呼び出し元: `scripts/real-machine-check.sh`（引数検証・作業ディレクトリの用意の後に
`python3 -I` で `run` を起動する）。製品（CLI・推論経路・配布物）には入らない検証用スクリプトで、
標準ライブラリだけを使う（依存の追加なし。Python 3.9 の文法で書く）。

責務:
- 項目 A〜F の子プロセス起動（上限時間・出力サイズ上限つき。中断時は子のグループを止める。REQ-39）
- CLI の JSON の要約と、記録へ出してよい値だけへの絞り込み（伏せ処理は `sanitize_record` に集約）
- `record.json`・`record.md` の書き出しと、stdout への固定メッセージ JSON 1 行

記録の規則（最重要）: 記録に書くのは件数・ハッシュ・サイズ・終了コード・固定語彙・JSON の要約だけ。
パス（`/` や `\\` などを含む文字列。例外は D のライブラリ名）・
データ本文（`input`・`text`・`id` の値）・
stderr の内容は書かない。

証拠種別: このスクリプトは測定と整形までを行い、「実機」の証拠としての確定は人が行う
（`evidence_hint` は `requires_human_review` か `test_harness` のみ）。
"""

from __future__ import annotations

import argparse
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "real-machine-check/1"
ITEM_ORDER = ["A", "B", "C", "D", "E", "F"]

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

# E の件数上限（子プロセスを件数ぶん起動するため）
MAX_E_RECORDS = 1000
MAX_REPEAT = 1000
# --p95-limit-us の上限（µs）。crates/core/src/definition.rs の MAX_LIMIT_INFER_P95_US と対
MAX_P95_LIMIT_US = 3_600_000_000
# 浮動小数の許容差（evaluation-contract.md）
SCORE_TOLERANCE = 1e-9

REDACTED = "<redacted>"
MAX_STR = 200
# 要約から落とす、データ本文を運びうるキー（大文字小文字を区別しない）
DENY_KEYS = frozenset({"input", "text", "id", "output"})
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
LIB_RE = re.compile(r"^/(usr/lib|System)/[A-Za-z0-9._+/-]+$")
FIXTURE_FILES = ("definition.json", "train.jsonl", "evaluation.jsonl")
INTERRUPT_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

# 子の外側で動く薄い sh。コマンドを子として走らせ、終了コードを rc ファイル（位置引数 $1）へ
# 書いてから、自分を含むグループ全体へ KILL を送る（TERM を無視する孫も残さない。REQ-39）。
# リーダー（この sh）の生存中に行うので、reap 後の killpg による pid 再利用の誤爆がない。
# 文字列は定数で、値は位置引数で渡す
GROUP_WRAPPER = 'RC=$1; shift; "$@"; rc=$?; printf "%s" "$rc" > "$RC"; kill -s KILL 0'
# rc ファイルの上限（3 桁の整数だけが入る）
CAP_RC_FILE = 16


class Interrupted(BaseException):  # Exception で捕まえられないよう BaseException にする
    """SIGINT・SIGTERM・SIGHUP を受けた（最上位でだけ捕まえる）。"""


# --------------------------------------------------------------------------------------
# 伏せ処理（記録へ出す値の唯一の関門）
# --------------------------------------------------------------------------------------


def _has_path_char(s: str) -> bool:
    """文字列がパス区切り（半角・全角・類似文字を含む）を含むか。"""
    return any(c in s for c in PATH_CHARS)


def redact_value(value: Any) -> Any:
    """CLI の JSON を再帰的に伏せる。

    文字列値のうちパス文字を含むものは `<redacted>` へ、長い文字列は切り詰める。
    `DENY_KEYS` のキー（大文字小文字不問）は落とし、キー自体にパス文字があれば落とす。
    """
    if isinstance(value, str):
        if _has_path_char(value):
            return REDACTED
        return value[:MAX_STR]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str) or k.lower() in DENY_KEYS or _has_path_char(k):
                continue
            out[k[:MAX_STR]] = redact_value(v)
        return out
    if isinstance(value, list):
        return [redact_value(v) for v in value[:100]]
    return value


def summarize_infer(obj: dict[str, Any]) -> dict[str, Any]:
    """`infer` の判定 JSON の要約。`status`・`predicted_label`・`scores` のキー数だけ。"""
    scores = obj.get("scores")
    return {
        "status": redact_value(obj.get("status")),
        "predicted_label": redact_value(obj.get("predicted_label")),
        "scores_keys": len(scores) if isinstance(scores, dict) else 0,
    }


def sanitize_record(rec: Any, _in_libs: bool = False) -> Any:
    """record 全体の最終関門。パス文字を含む文字列は `<redacted>`、非有限の浮動小数は None にする。

    例外は `items.D.direct_libraries` の要素で、`/usr/lib/`・`/System/` で始まり安全な文字だけの
    ものに限る。個別の要約で伏せ漏れがあっても、ここで必ず止まる。
    """
    if isinstance(rec, dict):
        out = {}
        for k, v in rec.items():
            key = k if isinstance(k, str) and not _has_path_char(k) else REDACTED
            out[key] = sanitize_record(v, key == "direct_libraries")
        return out
    if isinstance(rec, list):
        return [sanitize_record(v, _in_libs) for v in rec]
    if isinstance(rec, float) and not math.isfinite(rec):
        return None
    if isinstance(rec, str) and _has_path_char(rec):
        if _in_libs and LIB_RE.match(rec) and ".." not in rec:
            return rec
        return REDACTED
    return rec


# --------------------------------------------------------------------------------------
# 子プロセス（上限時間・出力サイズ上限・中断時の後始末）
# --------------------------------------------------------------------------------------


@dataclass
class RunResult:
    """子プロセスの結果。reason は `timeout`・`output_limit`・`spawn_error`・`killed` か None。"""

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


def _kill_group(pid: int) -> None:
    """プロセスグループへ KILL を送る。リーダーが未回収（reap 前）のときだけ呼ぶこと。"""
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


def _resolve_exe(name: str) -> str | None:
    """実行ファイルの絶対パスを返す。`/` を含まない名前は PATH から探す。無ければ None。"""
    exe = name if "/" in name else shutil.which(name)
    if exe is None or not os.path.isfile(exe) or not os.access(exe, os.X_OK):
        return None
    return exe


def run_cmd(
    argv: list[str],
    cwd: Path,
    out_path: Path,
    err_path: Path,
    timeout: int,
    out_cap: int,
    err_cap: int,
    env: dict[str, str] | None = None,
) -> RunResult:
    """子プロセスを独立したプロセスグループで起動し、期限と出力サイズを監視する（REQ-39）。

    stdin は /dev/null、stdout・stderr はファイルへ書く（呼び出し側が読む前に `out_bytes` を
    上限と照らす）。超過・期限切れ・例外・中断（`Interrupted`）のいずれでも、リーダーが未回収の
    うちにグループごと KILL してから回収する。`shell=True` は使わない。
    """
    exe = _resolve_exe(argv[0])
    if exe is None:
        return RunResult(None, "spawn_error", 0, 0)
    rc_path = out_path.with_name(out_path.name + ".rc")
    rc_path.unlink(missing_ok=True)
    full = ["/bin/sh", "-c", GROUP_WRAPPER, "sh", str(rc_path), exe, *argv[1:]]
    reason = None
    try:
        with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
            proc = subprocess.Popen(  # noqa: S603  引数リストで起動する。argv は固定語彙と検証済みの値のみ
                full,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=fo,
                stderr=fe,
                env=env,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + timeout
                while proc.poll() is None:
                    if time.monotonic() >= deadline:
                        reason = "timeout"
                        break
                    if _size(out_path) > out_cap or _size(err_path) > err_cap:
                        reason = "output_limit"
                        break
                    time.sleep(0.02)
            finally:
                # 例外・中断・期限超過で抜けたら、リーダーが未回収（poll が None）のうちに KILL する
                if proc.poll() is None:
                    _kill_group(proc.pid)
                proc.wait()
    except (OSError, ValueError):
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
    """rc ファイルを 0〜255 の整数として読む。無い・上限超過・形が違えば None。"""
    text = read_capped(path, CAP_RC_FILE)
    if text is None or not re.fullmatch(r"[0-9]{1,3}", text):
        return None
    n = int(text)
    return n if n <= 255 else None


def read_capped(path: Path, cap: int) -> str | None:
    """サイズが上限以下のときだけファイルを UTF-8 で読む。超過・読めない場合は None。"""
    try:
        if path.stat().st_size > cap:
            return None
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None


def sha256_file(path: Path, cap: int) -> str | None:
    """ファイルの sha256。サイズ上限超過・読めない場合は None。"""
    try:
        # FIFO・ディレクトリ等は open で固まる・落ちるため、通常ファイルだけを読む
        st = path.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_size > cap:
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
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
    # A 以外の子へ渡す環境（CARGO_NET_OFFLINE=true。REQ-38）
    offline_env: dict[str, str]
    # 開始時に記録した CLI の sha256（D が、リンクを確認した対象と同一かを照合する）
    cli_sha256: str | None = None
    # FANDHE_EDGE_BIN で CLI を差し替えたか（差し替えなら p95 は参考値に固定する）
    bin_override: bool = False


def make_offline_env() -> dict[str, str]:
    """現在の環境に `CARGO_NET_OFFLINE=true` を足したコピー（A の `make ci` には渡さない）。"""
    env = dict(os.environ)
    env["CARGO_NET_OFFLINE"] = "true"
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
        v = json.loads(text)
    except ValueError:
        return None
    return v if isinstance(v, dict) and v else None


def error_fields(obj: dict[str, Any] | None) -> dict[str, Any]:
    """失敗時に記録する `code`・`message`（伏せ処理後）。"""
    if not isinstance(obj, dict):
        return {}
    out: dict[str, Any] = {}
    if isinstance(obj.get("code"), str):
        out["code"] = redact_value(obj["code"])
    if isinstance(obj.get("message"), str):
        out["message"] = redact_value(obj["message"])
    return out


def _is_int(v: Any) -> bool:
    """bool を除く整数か。"""
    return isinstance(v, int) and not isinstance(v, bool)


def _is_finite_number(v: Any) -> bool:
    """bool を除く有限の数か。"""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


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


def _count_lines(path: Path) -> int | None:
    """空行を除く行数。読めなければ None。"""
    text = read_capped(path, CAP_INPUT_FILE)
    if text is None:
        return None
    return sum(1 for ln in text.splitlines() if ln.strip())


def read_facts(pdir: Path) -> Facts | None:
    """`pdir` の 3 ファイルから期待値を導く。読めなければ None。"""
    text = read_capped(pdir / "definition.json", CAP_INPUT_FILE)
    train, evaluation = _count_lines(pdir / "train.jsonl"), _count_lines(pdir / "evaluation.jsonl")
    if text is None or train is None or evaluation is None:
        return None
    try:
        d = json.loads(text)
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
    return Facts(
        option_ids=ids,
        train_records=train,
        eval_records=evaluation,
        has_acceptance="acceptance" in d,
        p95_limit_us=p95 if _is_int(p95) else None,
    )


def check_infer_output(obj: dict[str, Any], facts: Facts) -> bool:
    """infer の判定が定義と整合するか（REQ-21・REQ-33）。

    `predicted_label` が選択肢 ID のどれか、`scores` のキー集合が選択肢 ID と一致し、各値が有限で、
    合計が 1 から `SCORE_SUM_TOLERANCE` 以内（実 CLI の runtime が保証する範囲）。
    """
    scores = obj.get("scores")
    if not isinstance(scores, dict) or obj.get("predicted_label") not in facts.option_ids:
        return False
    if set(scores) != set(facts.option_ids) or len(scores) != len(facts.option_ids):
        return False
    if not all(_is_finite_number(v) for v in scores.values()):
        return False
    return abs(sum(float(v) for v in scores.values()) - 1.0) <= SCORE_SUM_TOLERANCE


def _nonneg_int(v: Any) -> bool:
    """bool を除く 0 以上の整数か。"""
    return _is_int(v) and v >= 0


def check_stage_report(name: str, obj: dict[str, Any], facts: Facts, selected: int | None) -> bool:
    """exit 0 の工程 JSON が、指定値・入力から分かる値と整合するか（REQ-21・REQ-33）。

    欄名・形は crates/core/src/stage_report.rs に合わせる。CLI の計算は再実装しない。
    """
    if name == "register":
        return obj.get("options") == len(facts.option_ids) and (
            obj.get("evaluation_defined") is (facts.eval_records > 0)
        )
    if name == "inspect":
        split = obj.get("split")
        vr = obj.get("valid_records")
        if not isinstance(split, dict) or not _nonneg_int(vr) or vr != facts.train_records:
            return False
        parts = [split.get(k) for k in ("train", "validation", "test")]
        return all(_nonneg_int(x) for x in parts) and sum(parts) == vr
    if name in ("train", "select"):
        return obj.get("candidate") == TRAIN_CANDIDATE and isinstance(obj.get("kind"), str)
    if name == "evaluate":
        n_total, correct = obj.get("n_total"), obj.get("correct")
        return (
            selected is not None
            and obj.get("candidate") == selected
            and n_total == facts.eval_records
            and _nonneg_int(correct)
            and correct <= n_total
        )
    if name == "package":
        # 合否基準の無い定義では judgment:null・acceptance_defined:false
        if not facts.has_acceptance:
            return obj.get("judgment") is None and obj.get("acceptance_defined") is False
        return obj.get("acceptance_defined") is True
    return True


def check_package_metrics(obj: dict[str, Any], rc: int, facts: Facts) -> bool:
    """package（exit 0・20）の `capacity`・`infer_p95` の整合（REQ-30・REQ-31）。

    capacity は要約でき、`exceeded == (total_bytes > limit_bytes)`、exit 0 なら超過なし。
    `infer_p95` は定義に `max_infer_p95_us` があるときだけ非 null。
    """
    cap = capacity_summary(obj)
    if cap is None or cap["exceeded"] != (cap["total_bytes"] > cap["limit_bytes"]):
        return False
    if rc == 0 and cap["exceeded"]:
        return False
    return (obj.get("infer_p95") is None) == (facts.p95_limit_us is None)


def _step_check(
    name: str,
    obj: dict[str, Any],
    allowed: set[int],
    rc: int,
    facts: Facts,
    selected: int | None,
) -> bool:
    """工程の stdout JSON が契約と整合するか（exit 0 は報告値まで照合、非 0 は許容と step）。"""
    if name == "infer":
        return (
            rc == 0
            and obj.get("status") == "ok"
            and "step" not in obj
            and isinstance(obj.get("id"), str)
            and check_infer_output(obj, facts)
        )
    if obj.get("step") != name:
        return False
    if rc != 0:
        return rc in allowed and (name != "package" or check_package_metrics(obj, rc, facts))
    if obj.get("status") != "ok" or not check_stage_report(name, obj, facts, selected):
        return False
    return name != "package" or check_package_metrics(obj, rc, facts)


def run_pipeline(
    ctx: Ctx,
    pdir: Path,
    sample_text: str | None,
    package_allowed: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """`pdir` で register〜package（と、あれば infer）を実行する。

    `pdir` に definition.json・train.jsonl・evaluation.jsonl が置かれている前提で、そこをカレントに
    する（経路の閉じ込め。REQ-39）。戻り値は (工程記録, package の JSON, 失敗記録)。
    失敗記録が None でなければ以降の工程は実行していない。
    """
    facts = read_facts(pdir)
    if facts is None:
        return [], None, fail_item("input_unreadable")
    logs = pdir / "steps"
    logs.mkdir(exist_ok=True)
    steps: list[dict[str, Any]] = []
    package_obj: dict[str, Any] | None = None
    selected: int | None = None

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
            return None, fail_item("invalid_json", step=name, exit_code=r.exit_code)
        rc = r.exit_code if r.exit_code is not None else EXIT_RUNTIME_ERROR
        entry["summary"] = summarize_infer(obj) if name == "infer" else redact_value(obj)
        steps.append(entry)
        if rc != 0 and rc not in allowed:
            err = error_fields(obj)
            return None, fail_item("unexpected_exit_code", step=name, exit_code=rc, **err)
        if not _step_check(name, obj, allowed, rc, facts, selected):
            return None, fail_item(
                "unexpected_output", step=name, exit_code=rc, **error_fields(obj)
            )
        if name == "package":
            package_obj = obj
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
    obj, failure = go(
        "select", ["select", "--project-dir", "project"], "select --project-dir project", set()
    )
    if failure:
        return steps, package_obj, failure
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
            "infer --package project/package --text <fixed-sample>",
            set(),
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
    if not _nonneg_int(total) or not _nonneg_int(limit) or not isinstance(exceeded, bool):
        return None
    return {
        "total_bytes": total,
        "limit_bytes": limit,
        "exceeded": exceeded,
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
            (dest / name).write_text(json.dumps(d), encoding="utf-8")
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
    for raw in text.splitlines():
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
    """A: `make ci`。通信しうるため `--with-ci` のときだけ呼ばれる（offline 環境は渡さない）。"""
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
    )
    if r.reason is not None:
        return fail_item(r.reason, exit_code=r.exit_code)
    text = read_capped(d / "make-ci.log", CAP_LOG_STDOUT)
    if text is None:
        return fail_item("output_limit", exit_code=r.exit_code)
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
    steps, pkg, failure = run_pipeline(ctx, bdir, "sandbox check 0123456789", {0})
    if failure:
        return dict(failure, steps=steps), False
    cap = capacity_summary(pkg)
    if cap is None:
        return fail_item("missing_field", step="package", steps=steps), False
    matches = sum(c["bytes"] for c in cap["components"].values()) == cap["total_bytes"]
    files = package_files(bdir / "project" / "package")
    if files is None:
        return fail_item("package_unreadable", steps=steps), False
    rec = {
        "status": "ok" if matches else "failed",
        "steps": steps,
        "capacity": cap,
        "capacity_sum_matches_total": matches,
        "package_files": files,
    }
    if not matches:
        rec["reason"] = "capacity_sum_mismatch"
    return rec, matches


def package_files(pdir: Path) -> list[dict[str, Any]] | None:
    """`package/` 直下の通常ファイルの名前・バイト数・sha256。読めなければ None。"""
    out: list[dict[str, Any]] = []
    try:
        entries = sorted(os.scandir(pdir), key=lambda e: e.name)
    except OSError:
        return None
    for e in entries:
        if not e.is_file(follow_symlinks=False):
            continue
        digest = sha256_file(Path(e.path), CAP_PACKAGE_FILE)
        if digest is None:
            return None
        out.append({"name": e.name, "bytes": e.stat().st_size, "sha256": digest})
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
    `capacity_limit_not_enforced`。
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
    p95_2 = pkg2.get("infer_p95") if isinstance(pkg2, dict) else None
    code2 = pkg2.get("code") if isinstance(pkg2, dict) else None
    limit_rec = {
        "package_exit_code": rc2,
        "code": redact_value(code2),
        "capacity_exceeded": cap2["exceeded"] if cap2 else None,
        "total_bytes": cap2["total_bytes"] if cap2 else None,
        "limit_bytes": cap2["limit_bytes"] if cap2 else None,
        "infer_p95_exceeded": (
            p95_2.get("exceeded") if isinstance(p95_2, dict) and "exceeded" in p95_2 else None
        ),
        "package_published": published,
    }
    reason2 = judge_capacity_limit(rc2, code2, cap2, ctx.package_limit_bytes, published)
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
    for line in text.splitlines()[1:]:
        name = line.strip().split(" (")[0].strip()
        if name:
            libs.append(name)
    return libs


def parse_linkage_tool(text: str) -> str | None:
    """check-runtime-linkage.sh の最終行 `OK: tool=<otool|ldd> ...` の tool。無ければ None。"""
    for line in reversed(text.splitlines()):
        m = re.match(r"^OK: tool=(\S+)", line)
        if m:
            return m.group(1) if m.group(1) in ("otool", "ldd") else None
    return None


def parse_linkage_cli_path(text: str) -> Path | None:
    """check-runtime-linkage.sh が出す `cli_bin: <絶対パス>` 行から検査対象の Path を得る。

    cargo の報告から取った実物のパスが唯一の出どころ（REQ-32）。該当行がちょうど 1 行で、
    値が絶対パスのときだけ返し、0 行・2 行以上・相対パスは None（照合不能として扱う）。
    """
    found = [m.group(1) for ln in text.splitlines() if (m := re.match(r"^cli_bin: (.+)$", ln))]
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
    text = read_capped(d / "linkage.log", CAP_LOG_STDOUT)
    if text is None:
        return fail_item("output_limit", exit_code=r.exit_code)
    rec: dict[str, Any] = {
        "exit_code": r.exit_code,
        "skip_lines": sum(1 for ln in text.splitlines() if ln.startswith("skip:")),
        "env_i_tests_ok": sum(1 for ln in text.splitlines() if ln.startswith("ok: req32_")),
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
    # 成功の印を厳密に: テスト数が定数と一致し、最終の `OK: tool=...` 行があること。
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
    otool = shutil.which("otool")
    if sys.platform == "darwin" and otool and not ctx.harness:
        o = run_cmd(
            [otool, "-L", str(ctx.bin)],
            ctx.repo,
            d / "otool.out",
            d / "otool.err",
            TIMEOUT_PROBE,
            CAP_PROBE,
            CAP_PROBE,
            ctx.offline_env,
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
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            v = json.loads(line)
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

    `predicted_label` が str でない・両側で異なる行、スコアに NaN・無限大・非数がある行は不一致。
    `max_abs_score_diff` は有限の差だけから求める（非有限は `scores_nonfinite` として数える）。
    許容差（`SCORE_TOLERANCE`）を超えた行の id も `mismatch_ids` に入れる。
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
        # 許容差を超えた行も mismatch_ids へ入れる（件数欄の意味は変えない）
        if not same_label or not finite or not (row_diff <= SCORE_TOLERANCE):
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
    with open(inputs, "w", encoding="utf-8") as f:
        for rid, text in recs:
            f.write(json.dumps({"id": rid, "input": text}, ensure_ascii=False) + "\n")
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
    lines = text.splitlines()
    if len(lines) != len(recs):
        return fail_item("unexpected_output", step="infer-batch", exit_code=0)
    for line in lines:
        try:
            v = json.loads(line)
        except ValueError:
            return fail_item("invalid_json", step="infer-batch", exit_code=0)
        if not isinstance(v, dict) or v.get("status") != "ok" or not isinstance(v.get("id"), str):
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
        if r.reason is not None or r.exit_code != 0:
            reason = r.reason or "unexpected_exit_code"
            return fail_item(reason, step="infer-single", exit_code=r.exit_code)
        obj = parse_json_object(edir / "single.stdout", CAP_CLI_STDOUT)
        if (
            obj is None
            or obj.get("status") != "ok"
            or obj.get("id") != rid
            or not check_infer_output(obj, facts)
        ):
            return fail_item("unexpected_output", step="infer-single", exit_code=0)
        single[rid] = obj
    cmp = compare_infer(batch, single)
    ids = cmp.pop("mismatch_ids")
    (edir / "mismatch-ids.txt").write_text("".join(i + "\n" for i in ids), encoding="utf-8")
    ok = (
        cmp["label_mismatch"] == 0
        and cmp["scores_nonfinite"] == 0
        and not (cmp["max_abs_score_diff"] > SCORE_TOLERANCE)
    )
    rec = dict({"status": "ok" if ok else "failed", "records": len(recs)}, **cmp)
    rec["input_sha256"] = digest
    if not ok:
        rec["reason"] = "mismatch"
    return rec


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


def ran_tests(text: str) -> bool:
    """ログに `test result: ok. N passed`（N は 1 以上）の行があるか（0 件実行は pass でない）。"""
    return any(int(n) >= 1 for n in re.findall(r"test result: ok\. (\d+) passed", text))


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
    load_start = load_average()
    passed = failed = timeouts = no_tests = incomplete = plain = 0
    for i in range(1, ctx.repeat + 1):
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
        text = read_capped(log, CAP_LOG_STDOUT) or ""
        if r.exit_code == 0 and r.reason is None and ran_tests(text):
            passed += 1
            continue
        failed += 1
        if r.exit_code == 0 and r.reason is None:
            no_tests += 1
        a, p = count_read_output(text)
        incomplete += a
        plain += p
    rec = {
        "status": "ok" if failed == 0 else "failed",
        "runs": ctx.repeat,
        "passed": passed,
        "failed": failed,
        "timeouts": timeouts,
        "no_tests": no_tests,
        "read_output": plain,
        "read_output_incomplete": incomplete,
        "load_start": load_start,
        "load_end": load_average(),
    }
    if failed:
        rec["reason"] = "test_failures"
    return rec


# --------------------------------------------------------------------------------------
# 環境の採取・入力の記録
# --------------------------------------------------------------------------------------


def _probe(work: Path, repo: Path, argv: list[str], name: str) -> str | None:
    """短い出力のコマンドを実行して先頭を返す。使えない・失敗は None。"""
    pdir = work / ".probe"
    pdir.mkdir(exist_ok=True)
    r = run_cmd(
        argv,
        repo,
        pdir / (name + ".out"),
        pdir / (name + ".err"),
        TIMEOUT_PROBE,
        CAP_PROBE,
        CAP_PROBE,
    )
    if r.reason is not None or r.exit_code != 0:
        return None
    text = read_capped(pdir / (name + ".out"), CAP_PROBE)
    return text.strip() if text is not None else None


def _worktree_clean(work: Path, repo: Path) -> bool | None:
    """`git status --porcelain` が空か。出力が上限を超えれば汚れている扱い。取れなければ None。"""
    pdir = work / ".probe"
    pdir.mkdir(exist_ok=True)
    r = run_cmd(
        ["git", "status", "--porcelain"],
        repo,
        pdir / "status.out",
        pdir / "status.err",
        TIMEOUT_PROBE,
        CAP_PROBE,
        CAP_PROBE,
    )
    if r.reason == "output_limit":
        return False
    if r.exit_code == 0 and r.reason is None:
        return _size(pdir / "status.out") == 0
    return None


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

    def as_int(s: str | None) -> int | None:
        """10 進数字だけの文字列を整数へ。違えば None。"""
        return int(s) if s is not None and s.isdigit() else None

    commit = _probe(w, rp, ["git", "rev-parse", "HEAD"], "commit")
    if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", commit):
        commit = None
    try:
        cli_bytes: int | None = ctx.bin.stat().st_size
    except OSError:
        cli_bytes = None
    return {
        "hw_model": sysctl("hw.model", "hw"),
        "cpu": sysctl("machdep.cpu.brand_string", "cpu"),
        "ncpu": as_int(sysctl("hw.ncpu", "ncpu")),
        "memory_bytes": as_int(sysctl("hw.memsize", "mem")),
        "os_name": sw_vers("-productName", "osn"),
        "os_version": sw_vers("-productVersion", "osv"),
        "os_build": sw_vers("-buildVersion", "osb"),
        "commit": commit,
        "worktree_clean": _worktree_clean(w, rp),
        "started_local": None,
        "ended_local": None,
        "cli_sha256": sha256_file(ctx.bin, CAP_CLI_BINARY),
        "cli_bytes": cli_bytes,
        "cli_profile": cli_profile,
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
        text = read_capped(p, CAP_INPUT_FILE)
        digest = sha256_file(p, CAP_INPUT_FILE)
        if text is None or digest is None:
            return None
        if key != "definition":
            out[key + "_records"] = sum(1 for ln in text.splitlines() if ln.strip())
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


def _cell(v: Any) -> str:
    """Markdown・HTML として解釈されないよう、セルの値をエスケープする。"""
    if v is None:
        return "-"
    s = str(v)
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
            "（このスクリプトがビルドした CLI ではない）"
        )
    lines += ["", "## 環境", "", "| 項目 | 値 |", "| ---- | -- |"]
    for k, v in env.items():
        lines.append(f"| {_cell(k)} | {_cell(v)} |")
    lines += ["", "## 入力と指定", "", "| 項目 | 値 |", "| ---- | -- |"]
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
            if k not in ("status", "steps", "package_files", "capacity", "p95", "capacity_limit")
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
    """stdout へ固定メッセージの JSON を 1 行だけ出す（パスは書かない）。"""
    obj: dict[str, Any] = {"code": code, "message": message}
    if with_record:
        obj["record"] = "record.json"
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    return exit_code


def _on_signal(signum: int, frame: Any) -> None:
    """中断シグナルのハンドラ。以降のシグナルを無視して後始末を終え `Interrupted` を送出する。"""
    for s in INTERRUPT_SIGNALS:
        signal.signal(s, signal.SIG_IGN)
    raise Interrupted


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
    for line in text.splitlines():
        try:
            v = json.loads(line)
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
    return item_f(ctx), b_ok


def run(args: argparse.Namespace) -> int:
    """項目を A→F の順に実行し、record.json・record.md を書く。"""
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
    )
    inputs = collect_inputs(ctx.repo)
    if inputs is None:
        return emit("runtime_error", "cannot read fixture inputs", EXIT_RUNTIME_ERROR, False)
    cli_profile: str | None = None
    if not args.bin_override:
        built = build_cli(ctx)
        if built is None:
            return emit("runtime_error", "cannot build the CLI", EXIT_RUNTIME_ERROR, False)
        ctx.bin = built
        cli_profile = "release"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env = collect_environment(ctx, cli_profile)
    env["started_local"] = started
    ctx.cli_sha256 = env["cli_sha256"]
    ctx.bin_override = bool(args.bin_override)
    rec: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_hint": "test_harness" if harness else "requires_human_review",
        "bin_override": bool(args.bin_override),
        "environment": env,
        "inputs": inputs,
        "options": {
            "items": items,
            "repeat": ctx.repeat,
            "quiet_machine": ctx.quiet_machine,
            "p95_limit_us": ctx.p95_limit_us,
            "package_limit_bytes": ctx.package_limit_bytes,
            "with_ci": bool(args.with_ci),
            "cargo_offline": True,
        },
        "items": {},
    }
    stopped = False
    interrupted = False
    internal_error = False
    b_ok = False
    for name in ITEM_ORDER:
        if name not in items:
            rec["items"][name] = {"status": "not_run", "reason": "not_selected"}
        elif interrupted or stopped:
            reason = "interrupted" if interrupted else "previous_item_failed"
            rec["items"][name] = {"status": "not_run", "reason": reason}
        elif name == "E" and not b_ok:
            rec["items"][name] = {"status": "not_run", "reason": "requires_B"}
        else:
            sys.stderr.write(f"running item {name}\n")
            sys.stderr.flush()
            try:
                res, b_ok = run_item(ctx, name, b_ok)
            except Interrupted:
                res, interrupted = fail_item("interrupted"), True
            except Exception as e:
                res = fail_item("internal_error", error_type=type(e).__name__)
                internal_error = True
            rec["items"][name] = res
            if res["status"] != "ok":
                stopped = True
    # 記録の書き出し中は中断シグナルを無視して、record を必ず完成させる
    for s in INTERRUPT_SIGNALS:
        signal.signal(s, signal.SIG_IGN)
    env["ended_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rec = sanitize_record(rec)
    # schema は固定定数（`/` を含む）なので伏せ処理の対象外にして戻す
    rec["schema"] = SCHEMA
    try:
        (ctx.work / "record.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (ctx.work / "record.md").write_text(render_markdown(rec), encoding="utf-8")
    except (OSError, ValueError):
        return emit("runtime_error", "cannot write the record", EXIT_RUNTIME_ERROR, False)
    if interrupted:
        return emit("runtime_error", "interrupted", EXIT_RUNTIME_ERROR, True)
    if internal_error:
        # 想定外の例外はスクリプト自身の実行不能（70）。record は書いてある
        return emit("runtime_error", "internal error", EXIT_RUNTIME_ERROR, True)
    if all(rec["items"][n]["status"] == "ok" for n in items):
        return emit("ok", "all requested items completed", EXIT_OK, True)
    return emit(
        "judged_fail", "one or more requested items failed or were not run", EXIT_JUDGED_FAIL, True
    )


class _ArgParser(argparse.ArgumentParser):
    """エラーを固定メッセージの JSON（invalid_input・64）で返す。引数の値は出力に含めない。"""

    def error(self, message: str) -> Any:  # message には引数の値が入りうるため出さない
        sys.exit(emit("invalid_input", "invalid arguments", EXIT_INVALID_INPUT, False))


def validate_args(args: argparse.Namespace) -> str | None:
    """引数の検証（シェル側と同じ条件の再確認）。問題があれば固定メッセージ、無ければ None。"""
    parts = args.items.split(",")
    if not parts or any(p not in ITEM_ORDER for p in parts) or len(set(parts)) != len(parts):
        return "--items must be a comma-separated subset of A,B,C,D,E,F"
    if "A" in parts and not args.with_ci:
        return "item A requires --with-ci"
    if not 1 <= args.repeat <= MAX_REPEAT:
        return "--repeat must be an integer from 1 to 1000"
    if not 1 <= args.p95_limit_us <= MAX_P95_LIMIT_US:
        return "--p95-limit-us must be an integer from 1 to 3600000000"
    if args.package_limit_bytes < 1:
        return "--package-limit-bytes must be a positive integer"
    if args.bin_override and not args.bin:
        return "--bin-override requires --bin"
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
    except Exception:
        return emit("runtime_error", "internal error", EXIT_RUNTIME_ERROR, False)


if __name__ == "__main__":
    sys.exit(main())
