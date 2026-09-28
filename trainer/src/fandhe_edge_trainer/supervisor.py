"""学習ワーカーのスーパーバイザー（`train`。公開サブコマンドの実体。P0-2）。

`onnx.checker.check_model`・`SerializeToString`・単一の MLX 演算のような「1 回の
呼び出しが長時間かかりうる同期呼び出し」の最中は、ワーカー内の協調的な資源チェック
（`budget.py::ResourceBudget.check`）は次に Python コードへ制御が戻るまで実行され
ない。そのため、壁時計・RSS の強制打ち切りはプロセス境界の外側から行う必要がある。

**本モジュールは意図的に mlx・onnx・numpy を import しない**（学習の実行に必要な
重い依存が無くても、別プロセス〔`_worker`〕の起動・監視・強制終了・異常終了時の
出力後始末だけなら本モジュールだけで完結する設計）。`tests/test_supervisor.py::
test_supervisor_module_does_not_import_mlx` で検証する。

**`out_dir` の所有権は本モジュールに一元化する**（P0-1・P0-2 の見直し。
`contract.py`・`OutDirReservation` のモジュール/クラス docstring も参照）。
`_worker` は強制終了されうる別プロセスであり、強制終了後に「経路の名前を
頼りに後始末を再構築する」設計は、名前が一致するというだけの根拠で無関係な
ディレクトリを消してしまう TOCTOU を生む。本モジュールは `_worker` を監視する
側であり自身は強制終了されない前提のため、予約に使った fd
（`OutDirReservation.entry.parent_fd`・`tmp_fd`）をジョブの最初から最後まで
手放さずに持ち続けられる。`_worker` には `pass_fds` で一時ディレクトリの fd
番号だけを渡し（`--out-fd <n>`）、`_worker` はその fd への書き込みしかしない
（`out_dir` の名前・予約・確定・後始末のいずれにも関与しない）。

流れ:
1. リクエスト JSON ファイルを `contract.read_request_bytes` で 1 回だけ読み、
   その生バイト列を `contract.parse_request_bytes`・`contract.validate_request`
   で完全に検証する（`_worker` と同じ検証・同じエラーメッセージ。ここで失敗
   すれば子プロセスは起動しない）。検証のうち `train_path`・`root` の fd は
   ここでは使わない（`_worker` が独立に開き直す）ので閉じる。`out_dir` の fd
   （`ConfinedEntry`）だけは、直後の予約のために保持し続ける。**読み取り済みの
   生バイト列（`raw`）は手元に保持し、後続の子プロセスへそのまま渡す**（P1:
   ファイルパスを子へ渡して再読込させると、検証後にファイルが書き換えられた
   場合、スーパーバイザーと子プロセスが異なる内容を見てしまう。検証済みの
   内容を固定して渡すことでこれを防ぐ）。
2. `contract.prepare_out_dir` で `out_dir` を排他的に予約し、作業用の一時
   ディレクトリを作る（`OutDirReservation`。`tmp_fd` を含む）。
3. `<sys.executable> -I <trainer/launch.py の絶対パス> _worker --out-fd <tmp_fd>
   --lifeline-fd <n>`（`worker_argv` が組み立てる argv。Issue #12: `-I`
   〔隔離モード〕で呼び出し元の `PYTHONPATH` 等に依存せず `trainer/src` を
   解決する）を、**常に新しいセッション**（`start_new_session=True`）で
   子プロセスとして起動する。`_worker` はこの新しいプロセスグループ W の
   リーダーになる。`pass_fds=(tmp_fd, lifeline_read_fd)` で一時ディレクトリの
   fd と lifeline パイプの読み取り端だけを引き継がせる（`lifeline_read_fd`
   は下記「lifeline」節参照）。リクエストの内容は `--request <path>` では
   渡さない。代わりに `raw` を `tempfile.TemporaryFile()`（作成直後に
   unlink 済みの無名一時ファイル。stdlib のみで完結し、`supervisor.py` が
   mlx・onnx・numpy を import しない設計を崩さない）へ書き込み、`seek(0)`
   してから子プロセスの標準入力（`stdin=`）として渡す（P1）。
4. 子プロセスの標準出力を、監視と並行して別スレッドで上限
   （`_MAX_WORKER_STDOUT_BYTES`）まで保持しつつ読み進める（P1-1: 監視ループが
   `stdout=PIPE` を読み出さないと、子プロセスがパイプを書き切れずに
   ブロックし、実際には正常に進んでいるのに `limit_exceeded` と誤判定しうる）。
5. 0.1 秒間隔でポーリングする: 壁時計（`time.monotonic`）が
   `time_limit_seconds` ＋ 猶予（`_TIME_LIMIT_GRACE_SECONDS`）を超えたか、
   `ps`（絶対パス `/bin/ps`）で読んだ子プロセスの RSS が `rss_limit_bytes` を
   超えたかを見る。`ps` の実行自体に失敗したら「監視ができない」ことを
   fail-closed に扱い、子プロセスを強制終了して `runtime_error` とする
   （安全側に倒す。上限を検査できないまま野放しにしない）。
6. 超過を検出したら `_terminate_worker` で `_worker` のプロセスグループ W
   全体を `os.killpg` で終了させ（孫プロセスまで含めて掃除する）、予約
   （一時ディレクトリとその中身・空の予約済みディレクトリ）を
   `contract.cleanup_reservation` で解放する（本モジュールが保持し続けている
   fd だけを使う。名前を再解決しない）。
7. `_worker` が自分で終了した場合も、**回収（reap）より必ず先に**
   `_terminate_worker`（`killpg`）を呼ぶ（`_current_child_status`・
   `_terminate_and_reap_after_self_exit` 参照。issue #178 PR #233 レビュー
   再指摘 P0「worker が先に終了すると子孫プロセスが残る」。`_worker` が
   何かの理由で孫プロセスを残したまま正常終了したケースを取りこぼさない
   ため）。回収後、標準出力が「ちょうど 1 つの妥当な JSON オブジェクトで
   ある」ことを確認し、終了コードが 7 種のいずれかであることも確認する。
   いずれかを満たさない、またはシグナルによる終了なら `runtime_error`
   （exit 70）とし、予約を解放する。終了コードが 0 以外（7 種のいずれかの
   エラー）なら、そのコード・JSON をそのまま使い、予約は解放する（出力を
   確定させない）。終了コードが 0（成功）なら、確定の前に
   `artifact.verify_output` で `model.onnx` の SHA-256 が `artifact.json` の
   記録と一致するかを確認し（P0: AGENTS.md ガード層「完全性と版」・REQ-39。
   不一致・欠落は予約を解放して確定させない）、一致すれば
   `contract.finalize_out_dir` で確定させたうえで、そのまま出力する。

Rust 側ジョブ管理（REQ-34）が最終的にはこの「外側のスーパーバイザー」の役割を
担う計画であり、本モジュールは Rust 側が無い・本ワーカーが単独プロセスとして
起動される場合の防御として存在する。本モジュール自身が SIGKILL 等で道連れに
終了した場合の後始末は、本モジュールの責務ではなく Rust 側ジョブ管理
（TASK-34.x）に委ねる。

**lifeline（issue #178 PR #233 レビュー: Rust 側でのプロセスグループ管理
〔`process_group(0)`・`/bin/kill` 呼び出し・`kill -0` 確認〕からの全面移行）**:
Rust 側 `run_train` が `_worker`（＝ Rust から見た孫プロセス）を直接把握・
終了させる設計（`process_group(0)`・`kill(-pgid)`）は、次のような構造的な
欠陥が収束しなかった: 回収済み（reap 済み）pgid への誤送出・環境変数だけで
単独起動時の掃除を無効化できてしまう・回収前のゾンビがグループに残るため
`kill -0` が常に「生存している」と誤判定する・`WallTimeout` が別のエラー
（`GroupCleanupUnconfirmed`）に化ける、等。これらは「Rust 側が worker の
プロセスグループを外部から観測・操作する」という設計そのものに起因する
ため、代わりに **worker 自身が supervisor（本プロセス）の死を検知して
自己終了する** 方式（lifeline）へ全面移行した。

- 本プロセスは `os.pipe()` を作り、**読み取り端だけ**を `pass_fds` で
  `_worker` へ渡す。**書き込み端は本プロセスが保持し続け、`_worker` には
  一切渡さない**（`_spawn_worker_and_finalize` 参照）。
- `_worker`（`cli.py::_start_lifeline_thread`）は起動直後から daemon
  スレッドで読み取り端を block read する。本プロセスがどのような形で
  終了しても（正常終了・内部タイムアウトによる `killpg`・Rust 側からの
  `SIGKILL` を含む）、カーネルが書き込み端の最後の複製を自動的に閉じるため、
  read は必ず EOF（0 バイト）で返る。
- EOF を観測した `_worker` は、自分自身のプロセスグループ W
  （`start_new_session=True` で起動されているため pgid は `_worker` 自身の
  pid）へ `SIGKILL` を送り（`os.killpg(os.getpgrp(), signal.SIGKILL)`）、
  自分自身とその孫プロセスをまとめて終了させる。
- この設計により、**Rust 側は直接の子（本プロセス）だけを把握すればよい**
  （`crates/train/src/process.rs` モジュール doc 参照）。孫プロセス
  （`_worker` とその子）に至るまでの確実な掃除は、Rust 側の関与なしに
  worker 自身が担う。孫プロセスがさらに `start_new_session`／`setsid` で
  別セッションへ抜けた場合（lifeline の読み取り端を引き継がない独自の
  子プロセスを作った場合）は、この方式でも対象外である（限界として
  正直に記録する）。

**不変条件: `_worker` のプロセスグループ掃除（`killpg`）は回収（reap）
より必ず先に行う**（issue #178 PR #233 レビュー再指摘 P0「worker が先に
終了すると、子孫プロセスが残る」）: `_worker` は `start_new_session=True`
で起動されるため pgid は `_worker` 自身の pid と一致する。`_worker` が
まだ回収されていない（ゾンビとして存在する）間は、この pgid は OS に
返却されず、他のセッションへ再利用されることもない。回収（`Popen.wait()`
等）を先に行ってしまうと、この保証が失われ、`killpg` が理論上は
無関係な別プロセスグループを巻き込みうる。

`monitor_child` は `_worker` の終了を **回収せずに** 検知する
（`_current_child_status` の `is_zombie`）: 素直な実装は
`os.waitid(..., os.WNOWAIT)` だが、CPython は macOS でこの関数を提供
しない（`Modules/posixmodule.c` が `HAVE_WAITID && !defined(__APPLE__)`
でガードしている。issue #178 PR #233 レビュー再指摘の中で確認した）ため
使えない。代わりに、既存の RSS 監視が使う `ps -p <pid>` 呼び出しへ
`stat=`（プロセス状態）を相乗りさせ、`Z`（ゾンビ）かどうかで判定する
（ゾンビはプロセス表から消えないため `ps` に引き続き表示される。
新しい依存・子プロセス起動は増やさない）。`_terminate_worker`
（`killpg`）を呼んでから初めて `Popen.wait()` で回収する
（`_terminate_and_reap_after_self_exit`）。この順序は、正常終了・
`RLIMIT_CPU` 自己終了・タイムアウト・RSS 超過・監視失敗のいずれの経路でも
例外なく守る。`ps` 自体が使えない・対象を見つけられない場合は、安全に
検知する手段が無いため `monitor_failed` として fail-closed に扱う。
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from . import artifact as artifact_mod
from . import contract
from .errors import WorkerError
from .exitcode import ExitCode

#: 監視ループのポーリング間隔（秒）。
_POLL_INTERVAL_SECONDS = 0.1

#: 壁時計の上限（`time_limit_seconds`）に上乗せする猶予（秒）。ワーカー内の
#: 協調的なチェック（バッチの境目等）が自ら気づいて終了する猶予を与える。
#: それでも終わらない場合（単一の長い同期呼び出しの最中等）はここで強制終了する。
_TIME_LIMIT_GRACE_SECONDS = 5.0

#: 子プロセスの標準出力の上限（bytes）。
_MAX_WORKER_STDOUT_BYTES = 1 * 1024 * 1024

#: `ps` の絶対パス（`shell=True` を使わず、`PATH` に依存しない）。
_PS_BIN = "/bin/ps"

#: `trainer/launch.py`（学習ワーカーの唯一の起動口。Issue #12）の絶対パス。
#: 本ファイル（`src/fandhe_edge_trainer/supervisor.py`）から見て 2 階層上が
#: `trainer/` になる。
_LAUNCH_SCRIPT = Path(__file__).resolve().parent.parent.parent / "launch.py"

#: 7 種の終了コード（REQ-21）。子プロセスの終了コードがこの集合に無ければ
#: `runtime_error` として扱う。
_VALID_EXIT_CODES = {int(code) for code in ExitCode}


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _current_child_status(pid: int) -> tuple[int, bool] | None:
    """`ps -o rss=,stat= -p <pid>` で RSS（バイト単位に変換済み）とゾンビ
    状態かどうかを 1 回の呼び出しでまとめて取得する。取得できなければ
    `None`。

    `ps` の RSS 出力は KiB 単位（BSD/macOS・Linux とも `-o rss=` は KiB）。
    `stat` の先頭が `Z`（Linux・macOS/BSD 共通の意味）であれば、対象は
    終了しているがまだ回収（reap）されていない（ゾンビ）ことを示す。

    ゾンビはプロセス表から消えないため（回収されるまで pid・
    プロセスグループ ID が OS に返却されない）、`ps` に引き続き表示される。
    これを利用して「回収せずに終了を検知する」手段とする（issue #178
    PR #233 レビュー再指摘 P0「worker が先に終了すると、子孫プロセスが
    残る」。本来は `os.waitid(..., os.WNOWAIT)` が素直だが、CPython は
    macOS でこの関数を提供しない〔`Modules/posixmodule.c` が `HAVE_WAITID
    && !defined(__APPLE__)` でガードしている〕ため使えない。既存の RSS
    監視が使う `ps` 呼び出しへ相乗りすることで、新しい依存・子プロセス
    起動を増やさずに実現する）。
    """
    try:
        result = subprocess.run(  # noqa: S603 - 引数は固定リスト。shell 不使用。絶対パスの /bin/ps のみを呼ぶ
            [_PS_BIN, "-o", "rss=,stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    parts = text.split(None, 1)
    if len(parts) != 2:
        return None
    rss_text, stat_text = parts
    try:
        kib = int(rss_text)
    except ValueError:
        return None
    is_zombie = stat_text.startswith("Z")
    return kib * 1024, is_zombie


def _terminate_worker(proc: subprocess.Popen) -> None:
    """`_worker` のプロセスグループ W 全体（pgid = `_worker` 自身の pid）へ
    `SIGKILL` を送る。

    `_worker` は常に `start_new_session=True` で別プロセスグループ
    （リーダー = `_worker` 自身）として起動されるため（モジュール docstring
    「lifeline」節参照）、pgid は `_worker` の pid と一致する。本モジュール
    自身はこのグループに属さないため `os.killpg` に巻き込まれない。
    `_worker` がさらに起動した孫プロセスもこのグループ W に属する限り、
    まとめて終了する。`_worker` がゾンビであっても `os.getpgid(proc.pid)`
    自体は呼べるが、pgid = `_worker` の pid であることは呼び出し起動時から
    確定している設計上の不変条件のため、`getpgid` は呼ばず `proc.pid` を
    直接使う（`getpgid` の呼び出しと `killpg` の間に無用な窓を作らない）。

    **呼び出し元が守るべき不変条件**: 本関数は必ず `Popen.wait()`（回収）
    より先に呼ぶこと。`_worker` がまだ回収されていない（ゾンビとして存在
    する）間は pgid が OS に返却されず再利用もされないため安全に
    `killpg` できるが、先に回収してしまうと pgid が理論上は別のプロセス
    グループへ再利用されうる（issue #178 PR #233 レビュー再指摘 P0
    「回収より前に kill するのは、pgid が再利用されて別のプロセスを kill
    しないため」）。対象が既に存在しない（`ProcessLookupError`）場合は
    無視してよい。

    `proc.kill()`（保険）は内部で `self.poll()` を呼ぶため、まだ回収されて
    いなければこの時点で `_worker` を回収する副作用がある。これは
    `os.killpg` の**後**に呼んでいるため、上記の不変条件（kill してから
    回収する）は保たれる。
    """
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()  # 上の killpg が何らかの理由で効かなかった場合の保険


def _cpu_seconds_consumed_by_children(baseline: resource.struct_rusage) -> float:
    """`baseline`（監視開始時点の `RUSAGE_CHILDREN`）からの CPU 時間（user+sys 秒）の
    増分。`RUSAGE_CHILDREN` は「これまでに reap した子プロセス」の累積値のため、
    1 ジョブにつきワーカーを 1 つずつ順に起動する本モジュールの設計では、この
    増分は基本的に当該ワーカーに帰属する（`_current_child_status` が起動する
    `ps` の CPU 消費もわずかに混入しうるが、しきい値との比較粒度に対して無視できる
    ほど小さい）。
    """
    current = resource.getrusage(resource.RUSAGE_CHILDREN)
    before = baseline.ru_utime + baseline.ru_stime
    after = current.ru_utime + current.ru_stime
    return after - before


def _classify_self_exit(
    proc: subprocess.Popen, cpu_baseline: resource.struct_rusage, soft_cpu_limit_seconds: float
) -> str | None:
    """子プロセスが自分で終了した直後に、それが `RLIMIT_CPU`（ワーカー自身が
    `cli.py::_apply_rlimit_cpu_backstop` で設定するソフト上限）による自己終了か
    どうかを判定する（P1）。

    ソフト上限到達時の既定動作は `SIGXCPU` によるプロセスの終了なので、通常は
    `returncode == -signal.SIGXCPU` で検出できる。ワーカーが `SIGXCPU` を
    キャッチ・無視していた場合はハード上限で `SIGKILL` されるため、その場合は
    「消費した CPU 時間がソフト上限以上か」で判定する（`SIGKILL` は OOM killer・
    外部からの kill でも起こりうるため、CPU 消費量で区別する）。
    いずれにも該当しなければ `None`（呼び出し元が通常の終了処理・
    シグナル終了の判定を続ける）。
    """
    returncode = proc.returncode
    if returncode is None or returncode >= 0:
        return None
    sig = -returncode
    if sig == signal.SIGXCPU:
        return "cpu"
    if sig == signal.SIGKILL:
        consumed = _cpu_seconds_consumed_by_children(cpu_baseline)
        if consumed >= soft_cpu_limit_seconds:
            return "cpu"
    return None


def _terminate_and_reap_after_self_exit(
    proc: subprocess.Popen, cpu_baseline: resource.struct_rusage, soft_cpu_limit_seconds: float
) -> str | None:
    """`_worker` が自分で終了した（が、まだ回収されていない＝ゾンビとして
    存在する）ことを `_current_child_status` の `is_zombie` で確認した
    直後に呼ぶ。

    **回収（`Popen.wait()`）より必ず先に** `_terminate_worker`（`killpg`）
    を呼ぶ（issue #178 PR #233 レビュー再指摘 P0「worker が先に終了すると、
    子孫プロセスが残る」。`_terminate_worker` のドキュメント参照）。
    `_worker` が孫プロセスを残さずに終了していた場合、`killpg` は
    ESRCH 相当（`ProcessLookupError`）になるだけで無害である。
    """
    _terminate_worker(proc)
    proc.wait()  # 既に終了を確認済みのため即座に返る（回収を完了させる）。
    return _classify_self_exit(proc, cpu_baseline, soft_cpu_limit_seconds)


def monitor_child(
    proc: subprocess.Popen,
    *,
    time_limit_seconds: float,
    rss_limit_bytes: int,
    poll_interval: float = _POLL_INTERVAL_SECONDS,
    grace_seconds: float = _TIME_LIMIT_GRACE_SECONDS,
) -> str | None:
    """`proc` を監視する。正常終了したら `None` を返す。

    壁時計・RSS のいずれかが上限を超えた場合、または RSS の監視自体が失敗した
    場合（`ps` の失敗）はプロセスグループを強制終了し、理由
    （`"time"`/`"rss"`/`"monitor_failed"`）を返す（呼び出し元が `proc.wait()`
    済みであることを前提にせず、本関数が確実に終了させてから返る）。

    **`RLIMIT_CPU`（`cli.py::_apply_rlimit_cpu_backstop`）による自己終了は
    `"cpu"` として返す**（P1）: ワーカー内の kernel レベルの CPU 時間上限は
    `time_limit_seconds` と同じ値をソフト上限に使っているため、これに達して
    ワーカーが自ら終了した場合も「資源上限超過」（`limit_exceeded`）として扱う
    べきで、外部からの予期しない終了（`runtime_error`）と区別する
    （`_classify_self_exit` 参照）。

    **本関数は `proc.stdout` を一切読まない**（P1-1: 標準出力の読み出しは
    呼び出し元が別スレッドで並行して行う。本関数が読み出しを兼ねると、
    `stdout=PIPE` のバッファが埋まった子プロセスがブロックし、実際には
    正常に進んでいるだけなのに監視が「反応しない」ように見えてしまう）。

    テスト（`tests/test_supervisor.py`）は本関数を直接、ダミーの子プロセス
    （`sys.executable -c "..."`）に対して呼ぶことで、実際の学習ワーカーを
    起動せずに監視ロジックを検証する。

    **`_worker` の終了検知は回収せずに行う**（`_current_child_status` の
    `is_zombie`。`ps` が使えない場合と、対象が既に回収済み・存在しない
    場合は `monitor_failed` として fail-closed に扱う。モジュール
    docstring「不変条件」節参照。issue #178 PR #233 レビュー再指摘 P0）。
    """
    cpu_baseline = resource.getrusage(resource.RUSAGE_CHILDREN)
    deadline = time.monotonic() + time_limit_seconds + grace_seconds
    while True:
        status = _current_child_status(proc.pid)
        if status is None:
            # `ps` がこの pid を見つけられなかった。本モジュールが `proc`
            # を唯一の回収者であり続ける限り、通常は `ps` 自体の失敗を
            # 意味する（`_worker` が既に回収済みなら、それは本関数か
            # テストのモックが直接 `wait()` した場合に限られる）。
            # `proc.poll()` で確認する: 既に終了していれば（`poll()` 自体が
            # 回収を伴うが、この時点で pid は既に本関数の管理下から外れて
            # いる可能性が高く、これ以上安全に `killpg` する手段が無いため）
            # 通常の終了として分類する。そうでなければ監視できないことを
            # fail-closed に扱う。
            if proc.poll() is not None:
                return _classify_self_exit(proc, cpu_baseline, time_limit_seconds)
            _terminate_worker(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
            return "monitor_failed"
        rss, is_zombie = status
        if is_zombie:
            return _terminate_and_reap_after_self_exit(proc, cpu_baseline, time_limit_seconds)
        if time.monotonic() > deadline:
            _terminate_worker(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
            return "time"
        if rss > rss_limit_bytes:
            _terminate_worker(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
            return "rss"
        time.sleep(poll_interval)


def _drain_stdout(pipe: Any, result: dict[str, Any], cap: int = _MAX_WORKER_STDOUT_BYTES) -> None:
    """子プロセスの標準出力を、上限 `cap` バイトまで保持しつつ最後まで読み進める
    （P1-1）。

    `subprocess.Popen(stdout=PIPE)` はパイプに OS のバッファ容量（環境依存だが
    数十 KiB 程度）分しか溜め込めない。監視ループ（`monitor_child`）が読み出しを
    行わないまま長時間かかると、子プロセスが `stdout` への書き込みで
    ブロックし、実際には壁時計・RSS の上限に達していないのに、監視から見ると
    「反応が無い」状態になりうる。これを防ぐため、監視と並行する別スレッドで
    バッファを溜めずに読み続ける（上限を超えた分は保持せず破棄するが、
    読み出し自体は続けることで子プロセス側のブロックを防ぐ）。

    `result` へ `"data"`（保持したバイト列。最大 `cap` バイト）・
    `"oversized"`（上限を超えたか）を書き込む（スレッドの戻り値の代わり）。
    """
    chunks: list[bytes] = []
    kept = 0
    oversized = False
    try:
        while True:
            chunk = pipe.read(65536)
            if not chunk:
                break
            if not oversized:
                remaining = cap - kept
                if len(chunk) <= remaining:
                    chunks.append(chunk)
                    kept += len(chunk)
                else:
                    if remaining > 0:
                        chunks.append(chunk[:remaining])
                        kept += remaining
                    oversized = True
    except (OSError, ValueError):
        # 親側でパイプを閉じた等。読めた分だけを使う。
        pass
    result["data"] = b"".join(chunks)
    result["oversized"] = oversized


def _parse_worker_stdout(raw: bytes) -> dict[str, Any] | None:
    """子プロセスの標準出力が「ちょうど 1 つの妥当な JSON オブジェクト」で
    あることを確認する。複数行・JSON でない・オブジェクトでない場合は `None` を
    返す（サイズ上限超過は `_drain_stdout` の `oversized` で別途判定する）。
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def run_supervised_train(request_path: Path) -> ExitCode:
    """`train` サブコマンドの本体。`out_dir` を予約したうえで `_worker` を
    子プロセスとして起動・監視し、結果に応じて確定または解放する。
    """
    try:
        # リクエストファイルはここで 1 回だけ読む。パース前の生バイト列
        # （raw_request）を手元に残し、検証後もそのまま子プロセスへ渡す
        # （P1: ファイルパスを子へ渡して再読込させない。モジュール docstring 参照）。
        raw_request = contract.read_request_bytes(request_path)
        parsed = contract.parse_request_bytes(raw_request)
        request = contract.validate_request(parsed)
    except WorkerError as e:
        _emit({"status": "error", "code": e.code, "message": e.message})
        return e.exit_code

    time_limit_seconds = request.time_limit_seconds
    rss_limit_bytes = request.rss_limit_bytes
    # train_path・root の fd はスーパーバイザーには不要（_worker が独立に
    # 検証・open し直す）。out_dir の fd だけは、直後の予約のために保持する。
    request.train_path.close()
    request.root.close()

    try:
        reservation = contract.prepare_out_dir(request.out_dir)
    except WorkerError as e:
        request.out_dir.close()
        _emit({"status": "error", "code": e.code, "message": e.message})
        return e.exit_code

    try:
        return _spawn_worker_and_finalize(
            raw_request,
            reservation,
            time_limit_seconds=float(time_limit_seconds),
            rss_limit_bytes=rss_limit_bytes,
        )
    finally:
        reservation.entry.close()  # request.out_dir と同一オブジェクト


def worker_argv(out_fd: int, lifeline_fd: int) -> list[str]:
    """`_worker` を起動する argv を組み立てる（Issue #12）。

    `[sys.executable, "-I", <trainer/launch.py の絶対パス>, "_worker",
    "--out-fd", str(out_fd), "--lifeline-fd", str(lifeline_fd)]` を返す。
    `-m fandhe_edge_trainer` ではなく `trainer/launch.py`（`-I` 付き）を
    経由することで、呼び出し元の `PYTHONPATH` の設定漏れ・汚染に左右されず
    `trainer/src` を解決できる（`launch.py` のモジュール docstring 参照）。
    `lifeline_fd` は本モジュールが `pass_fds` で引き継いだ lifeline パイプの
    読み取り端の fd 番号（`_spawn_worker_and_finalize` 参照。issue #178
    PR #233 レビュー）。テスト（`tests/test_cli.py` の
    `test_worker_rejects_non_regular_stdin_without_blocking`）も本関数を
    再利用し、実際の起動経路と同じ argv で検証する。
    """
    return [
        sys.executable,
        "-I",
        str(_LAUNCH_SCRIPT),
        "_worker",
        "--out-fd",
        str(out_fd),
        "--lifeline-fd",
        str(lifeline_fd),
    ]


def _spawn_worker_and_finalize(
    raw_request: bytes,
    reservation: contract.OutDirReservation,
    *,
    time_limit_seconds: float,
    rss_limit_bytes: int,
) -> ExitCode:
    # lifeline（issue #178 PR #233 レビュー: Rust 側でのプロセスグループ管理
    # 〔`process_group(0)`・`/bin/kill` 呼び出し・`kill -0` 確認〕は PID
    # 再利用・ESRCH 誤判定・reap 済み pgid への誤送出等の構造的な欠陥が
    # 収束しなかったため全面撤去し、代わりに worker 自身が「親（本プロセス）
    # の死」を検知して自己終了する lifeline 方式へ移行した）。
    #
    # `os.pipe()` の読み取り端だけを `pass_fds` で worker へ渡す。書き込み端
    # （`lifeline_write_fd`）は本プロセスが本関数の最後まで（＝worker の
    # 監視・後始末が終わるまで）保持し続け、worker には一切渡さない。
    # worker 側（`cli.py::_start_lifeline_thread`）は起動直後から daemon
    # スレッドで読み取り端を block read し、本プロセスがどのような形で
    # 終了しても（Rust 側からの `SIGKILL` を含む）カーネルが書き込み端を
    # 自動的に閉じるため、read は必ず EOF で返る。EOF を観測した worker は
    # 自分自身のプロセスグループ（`start_new_session=True` で起動している
    # ため pgid は worker 自身の pid）へ `SIGKILL` を送り、worker 自身と
    # その孫プロセスをまとめて終了させる。
    lifeline_read_fd, lifeline_write_fd = os.pipe()
    try:
        argv = worker_argv(reservation.tmp_fd, lifeline_read_fd)
        # P1: 検証済みのリクエスト（raw_request）を、作成直後に unlink 済みの
        # 無名一時ファイル（stdlib のみ。mlx・onnx・numpy を import しない設計を
        # 崩さない）へ書いて子プロセスの標準入力として渡す。`--request <path>` は
        # 使わない（検証後のファイル書き換えによる TOCTOU を防ぐ。モジュール
        # docstring 参照）。
        #
        # セキュリティ監査指摘: `tempfile.TemporaryFile()`・`write`・`seek` は
        # （ENOSPC 等で）例外を送出しうるが、この時点で `out_dir` の予約
        # （空の予約済みディレクトリ・一時ディレクトリ）は既に確保済みである。
        # ここで送出されたあらゆる例外（`BaseException`）を外側の `try` で捕捉し、
        # `cleanup_reservation` で解放してから再送出することで、予約だけが
        # 残置される事態を防ぐ（REQ-39）。内側の `except OSError`（`Popen` 失敗）は
        # 既に cleanup 済みで `return` するため、外側には伝播せず二重 cleanup には
        # ならない。
        try:
            with tempfile.TemporaryFile() as req_file:
                req_file.write(raw_request)
                req_file.seek(0)
                try:
                    # `_worker` は常に新しいセッション（`start_new_session=True`）
                    # で起動する。worker はこのグループ W のリーダーになり、
                    # lifeline の EOF 検知時に自分自身のプロセスグループ
                    # （= 自分の pid）を `os.killpg` で終了させられる
                    # （モジュール docstring・本関数冒頭のコメント参照）。
                    proc = subprocess.Popen(  # noqa: S603 - 引数は固定リスト。shell 不使用。sys.executable は絶対パス
                        argv,
                        stdin=req_file,
                        stdout=subprocess.PIPE,
                        stderr=None,  # 継承（親の stderr へ直接流す。パイプを溜めて詰まらせない）
                        pass_fds=(reservation.tmp_fd, lifeline_read_fd),
                        start_new_session=True,
                    )
                except OSError as e:
                    contract.cleanup_reservation(reservation)
                    _emit(
                        {
                            "status": "error",
                            "code": "runtime_error",
                            "message": f"failed to start worker: {type(e).__name__}",
                        }
                    )
                    return ExitCode.RUNTIME_ERROR
        except BaseException:
            contract.cleanup_reservation(reservation)
            raise
        finally:
            # 子プロセスは起動時に読み取り端の複製を保持しているため、
            # 親側の複製はここで（起動の成否によらず）閉じてよい。
            # 書き込み端（`lifeline_write_fd`）は関数末尾まで開いたまま
            # にする（下記 `finally` 参照）。
            with contextlib.suppress(OSError):
                os.close(lifeline_read_fd)
        # `with` を抜けると req_file（親側の fd）は閉じるが、子プロセスは
        # 起動時に複製した自分の fd を保持しているため読み取りに支障はない。

        return _monitor_worker_and_finalize(
            proc,
            reservation,
            time_limit_seconds=time_limit_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )
    finally:
        with contextlib.suppress(OSError):
            os.close(lifeline_write_fd)


def _monitor_worker_and_finalize(
    proc: subprocess.Popen,
    reservation: contract.OutDirReservation,
    *,
    time_limit_seconds: float,
    rss_limit_bytes: int,
) -> ExitCode:
    """`_spawn_worker_and_finalize` が起動した `proc`（`_worker`）を監視し、
    結果に応じて `out_dir` の予約を確定または解放する（分離した理由:
    呼び出し元が lifeline パイプの書き込み端を `finally` で確実に閉じられる
    よう、本体を別関数へ切り出した）。
    """
    # P1-1: 監視（proc.wait を繰り返す）と並行して、別スレッドで標準出力を
    # 溜めずに読み進める。子プロセスがパイプを埋めてブロックするのを防ぐ。
    stdout_result: dict[str, Any] = {}
    reader_thread = threading.Thread(
        target=_drain_stdout, args=(proc.stdout, stdout_result), daemon=True
    )
    reader_thread.start()

    killed_reason = monitor_child(
        proc, time_limit_seconds=time_limit_seconds, rss_limit_bytes=rss_limit_bytes
    )

    reader_thread.join(timeout=10)
    if reader_thread.is_alive() and proc.stdout is not None:
        # 通常は proc の終了（パイプの書き手が閉じる）で reader は自然に
        # 終わるはずだが、万一残っていたら pipe を閉じて読み出しを解除する。
        with contextlib.suppress(OSError):
            proc.stdout.close()
        reader_thread.join(timeout=5)
    if proc.stdout is not None:
        with contextlib.suppress(OSError):
            proc.stdout.close()

    if killed_reason is not None:
        contract.cleanup_reservation(reservation)
        if killed_reason == "monitor_failed":
            _emit(
                {
                    "status": "error",
                    "code": "runtime_error",
                    "message": "resource monitoring failed (ps unavailable); worker terminated",
                }
            )
            return ExitCode.RUNTIME_ERROR
        _emit(
            {
                "status": "error",
                "code": "limit_exceeded",
                "message": f"worker exceeded the {killed_reason} limit and was terminated",
            }
        )
        return ExitCode.LIMIT_EXCEEDED

    returncode = proc.returncode
    if returncode is not None and returncode < 0:
        # シグナルによる終了（例: OOM killer・外部からの kill）。
        contract.cleanup_reservation(reservation)
        _emit(
            {
                "status": "error",
                "code": "runtime_error",
                "message": f"worker terminated by signal {-returncode}",
            }
        )
        return ExitCode.RUNTIME_ERROR

    if stdout_result.get("oversized"):
        contract.cleanup_reservation(reservation)
        _emit(
            {
                "status": "error",
                "code": "runtime_error",
                "message": "worker stdout exceeded size limit",
            }
        )
        return ExitCode.RUNTIME_ERROR

    payload = _parse_worker_stdout(stdout_result.get("data", b""))
    if payload is None:
        contract.cleanup_reservation(reservation)
        _emit(
            {
                "status": "error",
                "code": "runtime_error",
                "message": "worker stdout was not exactly one valid JSON object",
            }
        )
        return ExitCode.RUNTIME_ERROR

    if returncode not in _VALID_EXIT_CODES:
        contract.cleanup_reservation(reservation)
        _emit(
            {
                "status": "error",
                "code": "runtime_error",
                "message": f"worker exited with an unexpected code {returncode}",
            }
        )
        return ExitCode.RUNTIME_ERROR

    if returncode != int(ExitCode.OK):
        # ワーカー自身が 7 種のいずれかのエラーで終了した（例: invalid_config）。
        # 出力は確定させず、予約を解放してからそのままエラーを伝える。
        contract.cleanup_reservation(reservation)
        _emit(payload)
        return ExitCode(returncode)

    # P0: 確定（rename）の直前に、保持し続けている tmp_fd（名前を再解決しない）
    # に対して model.onnx の SHA-256 が artifact.json の記録と一致するかを
    # 確認する（AGENTS.md ガード層「完全性と版」・REQ-39。artifact.py の
    # モジュール docstring 参照）。不一致・欠落は出力を確定させない。
    try:
        artifact_mod.verify_output(reservation.tmp_fd)
    except WorkerError as e:
        contract.cleanup_reservation(reservation)
        _emit({"status": "error", "code": e.code, "message": e.message})
        return e.exit_code

    try:
        contract.finalize_out_dir(reservation)
    except WorkerError as e:
        _emit({"status": "error", "code": e.code, "message": e.message})
        return e.exit_code

    _emit(payload)
    return ExitCode.OK
