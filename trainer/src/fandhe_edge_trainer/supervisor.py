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
   `_terminate_and_reap` 参照。issue #178 PR #233 レビュー
   再指摘 P0「worker が先に終了すると子孫プロセスが残る」。`_worker` が
   何かの理由で孫プロセスを残したまま正常終了したケースを取りこぼさない
   ため）。**`_worker` の状態自体を確認できない場合（`ps` の失敗等）も
   正常終了として扱わず、同じく `killpg` してから `monitor_failed` とする**
   （issue #178 PR #233 レビュー再々指摘 P0。回収を伴う `poll()`／`wait()`
   を代替の確認手段として使わない）。回収後、標準出力が「ちょうど 1 つの
   妥当な JSON オブジェクトである」ことを確認し、終了コードが 7 種の
   いずれかであることも確認する。
   いずれかを満たさない、またはシグナルによる終了なら `runtime_error`
   （exit 70）とし、予約を解放する。終了コードが 0 以外（7 種のいずれかの
   エラー）なら、そのコード・JSON をそのまま使い、予約は解放する（出力を
   確定させない）。終了コードが 0（成功）なら、確定の前に
   `artifact.verify_output` で `model.onnx` の SHA-256 が `artifact.json` の
   記録と一致するかを確認し（P0: AGENTS.md ガード層「完全性と版」・REQ-39。
   不一致・欠落は予約を解放して確定させない）、一致すれば
   `contract.finalize_out_dir` で確定させたうえで、そのまま出力する。

**協調キャンセルと責務分担（REQ-34・TASK-34.1-2・issue #145）**: Rust 側
ジョブ管理（`crates/train/src/process.rs`）は `train --cancel-on-stdin-eof` で
本プロセスを `stdin` パイプ付きで起動し、キャンセル時にその書き込み端を閉じる
（EOF）。本プロセスは EOF を `cli.py::_start_cancel_watch` の daemon スレッドで
検知して `cancel_event` を立て、`monitor_child` が `_terminate_and_reap`
（`killpg` → 回収）で worker を止めたうえで、保持中の fd だけで予約を解放する
（`cleanup_reservation`。名前は再解決しない）。

- 予約・確定・解放（`out_dir` の所有権）: 本モジュールのみ
- キャンセル要求の伝達（stdin を閉じる）・猶予の管理・猶予超過時の `SIGKILL`:
  Rust 側 `process.rs`
- `Completed`／`Cancelled` の最終判定: Rust 側 `process.rs`（本プロセスの
  終了状態と結果 JSON に従う）
- **公開されている ⇔ 本プロセスが exit 0 と ok JSON を返した**: 確定
  （`finalize_out_dir`）の直前までキャンセルを確認し、確定した後に届いた
  キャンセルは無視して成功を報告する。これで「公開済みなのに `Cancelled`」
  「未公開なのに成功」が起きない
- Rust 側の猶予内に本プロセスが終了しなければ Rust が `SIGKILL` する
  （フォールバック）。この場合だけ、後始末が走らず空の予約済み `out_dir`・
  tmp が残りうる（安全側の残置。Rust は削除せず読み取り専用で検査して報告
  するだけで、やり直し時の案内・掃除は TASK-34.3）
- 協調キャンセル時の終了コード（exit 70・`runtime_error`）は暫定で、Rust 側は
  依存しない。キャンセルの写像は TASK-33.x で決める

`--cancel-on-stdin-eof` が無い単独起動では標準入力を監視しない
（`cancel_event` は `None`）。

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
（`_terminate_and_reap`。**`monitor_child` はこの関数だけを通じて `proc`
を回収し、他の箇所で `poll()`／`wait()`／`communicate()` を直接呼ばない**。
issue #178 PR #233 レビュー再々指摘 P0「`_current_child_status()` が
`None` を返す場合に `proc.poll()` が worker を回収してしまい `killpg` を
経由しない」への対応）。この順序は、正常終了・`RLIMIT_CPU` 自己終了・
タイムアウト・RSS 超過・監視失敗のいずれの経路でも例外なく守る。`ps`
自体が使えない・対象を見つけられない場合は、安全に検知する手段が無いため
リトライせず 1 回の失敗で `monitor_failed` として fail-closed に扱う。
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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import artifact as artifact_mod
from . import contract
from .errors import WorkerError
from .exitcode import ExitCode
from .limits import MAX_RESULT_BYTES, MAX_RESULT_BYTES_WITH_VALIDATION

#: 監視ループのポーリング間隔（秒）。
_POLL_INTERVAL_SECONDS = 0.1

#: 壁時計の上限（`time_limit_seconds`）に上乗せする猶予（秒）。ワーカー内の
#: 協調的なチェック（バッチの境目等）が自ら気づいて終了する猶予を与える。
#: それでも終わらない場合（単一の長い同期呼び出しの最中等）はここで強制終了する。
_TIME_LIMIT_GRACE_SECONDS = 5.0

#: `ps` 1 回の呼び出しの上限（秒）。実際の待ちは締め切りの残り時間でさらに
#: 切り詰める（`_bounded_timeout`）。
_PS_TIMEOUT_SECONDS = 5.0

#: `killpg` 後の回収（`Popen.wait`）の上限（秒）。`crates/train` の
#: `COOPERATIVE_CANCEL_GRACE_SECONDS`（15 秒）は本値に予約解放と JSON 出力の
#: 時間を加えた値を上回る前提で決めている。
_REAP_WAIT_SECONDS = 10.0

#: 標準出力 reader の合流待ちの上限（秒）。キャンセル要求が来たら打ち切る。
_READER_JOIN_SECONDS = 10.0

#: 子プロセスの標準出力の上限（bytes）。
_MAX_WORKER_STDOUT_BYTES = MAX_RESULT_BYTES

#: `validation_inputs` を持つリクエストの子プロセス標準出力の上限（bytes）。
#: 値は `limits.py::MAX_RESULT_BYTES_WITH_VALIDATION`（Rust 側の
#: `MAX_RESULT_BYTES_WITH_VALIDATION` と同じ。共有 fixture で照合）。
_MAX_WORKER_STDOUT_BYTES_WITH_VALIDATION = MAX_RESULT_BYTES_WITH_VALIDATION

#: `ps` の絶対パス（`shell=True` を使わず、`PATH` に依存しない）。
_PS_BIN = "/bin/ps"

#: `trainer/launch.py`（学習ワーカーの唯一の起動口。Issue #12）の絶対パス。
#: 本ファイル（`src/fandhe_edge_trainer/supervisor.py`）から見て 2 階層上が
#: `trainer/` になる。
_LAUNCH_SCRIPT = Path(__file__).resolve().parent.parent.parent / "launch.py"

#: 7 種の終了コード（REQ-21）。子プロセスの終了コードがこの集合に無ければ
#: `runtime_error` として扱う。
_VALID_EXIT_CODES = {int(code) for code in ExitCode}


#: 協調キャンセルの応答文（`crates/train/src/process.rs` の `CANCEL_ACK_MESSAGE`・
#: `CANCEL_CLEANUP_INCOMPLETE_MESSAGE` と両側で一致させる。REQ-34・#145）。
_CANCEL_ACK_MESSAGE = "training cancelled by caller"
_CANCEL_CLEANUP_INCOMPLETE_MESSAGE = "training cancelled but cleanup incomplete"
_RESERVED_MESSAGES = frozenset({_CANCEL_ACK_MESSAGE, _CANCEL_CLEANUP_INCOMPLETE_MESSAGE})


def _forward_worker_error(payload: dict[str, Any]) -> dict[str, Any]:
    """worker が返したエラー JSON を転送用に整える。

    キャンセル応答の予約文言と一致する `message` は、supervisor 自身の応答と
    取り違えられないよう固定の別文言へ置き換える（フィールドの追加・意味の変更は
    しない。JSON 契約は不変）。REQ-34・#145。
    """
    if payload.get("message") in _RESERVED_MESSAGES:
        return {**payload, "message": "worker reported an error"}
    return payload


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _remaining(deadline: float) -> float:
    """締め切り（`time.monotonic` 基準）までの残り秒数（負にならない）。

    本モジュールの待機はすべて、1 つの締め切りから残り時間を求めてここを
    通す（個々の固定タイムアウトが残り時間を超えないようにする規則を 1 か所に
    集約する。`crates/train/src/process.rs` の `bounded_deadline`・
    `classify_cancel_outcome` と同じ方針。REQ-34・REQ-39）。
    """
    return max(0.0, deadline - time.monotonic())


def _bounded_timeout(cap: float, deadline: float) -> float:
    """固定の上限 `cap` と締め切りまでの残り時間の小さい方（秒）。"""
    return min(cap, _remaining(deadline))


def _wait_cancel(cancel_event: threading.Event | None, timeout: float) -> bool:
    """最大 `timeout` 秒待ち、その間にキャンセルが要求されたかを返す。

    キャンセルが来たら即座に戻る（`time.sleep` のように猶予を食わない）。
    `cancel_event` が `None` なら単に `timeout` 秒眠り `False` を返す。
    """
    if cancel_event is None:
        time.sleep(timeout)
        return False
    return cancel_event.wait(timeout)


def _run_ps(pid: int, timeout: float, cancel_event: threading.Event | None) -> str | None:
    """`ps -o rss=,stat= -p <pid>` を実行して標準出力を返す。**例外を外へ出さない**。

    起動・待機・kill・回収・パイプの後始末をこの関数に閉じ込める。返り値は
    標準出力（成功）か `None`（不明）のどちらかで、起動失敗・タイムアウト・
    キャンセルによる中断・kill と終了の競合（`ProcessLookupError`）・
    `communicate` の `OSError`／`ValueError`（閉じたパイプ）・非ゼロ終了は
    すべて `None` に写す。終了時は必ず `ps` を回収しパイプを閉じる。呼び出し側
    （`monitor_child`）は `None` を既存の fail-closed 方針で扱い、予約の解放
    経路へ進む（REQ-34・REQ-39）。
    """
    try:
        ps = subprocess.Popen(  # noqa: S603 - 引数は固定リスト。shell 不使用。絶対パスの /bin/ps のみを呼ぶ
            [_PS_BIN, "-o", "rss=,stat=", "-p", str(pid)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, ValueError):
        return None
    result: str | None = None
    try:
        # `timeout`（締め切りの残りで切り詰め済み）まで、キャンセルを短い刻みで
        # 確認しながら待つ。キャンセルが来たら即座に戻り、協調キャンセルの猶予を
        # 状態確認で使い切らない（REQ-34）。
        stop_at = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                out, _ = ps.communicate(timeout=min(0.05, _remaining(stop_at)))
            except subprocess.TimeoutExpired:
                if _is_cancelled(cancel_event) or _remaining(stop_at) <= 0.0:
                    return None
                continue
            if ps.returncode == 0:
                result = out
            return result
    except (OSError, ValueError):
        return None
    finally:
        # 成功・失敗・例外のいずれでも、`ps` を止めて回収し、パイプを閉じる。
        with contextlib.suppress(OSError):
            if ps.poll() is None:
                ps.kill()
        with contextlib.suppress(OSError, ValueError, subprocess.TimeoutExpired):
            ps.communicate(timeout=1)
        with contextlib.suppress(OSError, ValueError, subprocess.TimeoutExpired):
            ps.wait(timeout=1)
        for pipe in (ps.stdout, ps.stdin, ps.stderr):
            if pipe is not None:
                with contextlib.suppress(OSError, ValueError):
                    pipe.close()


def _stop_reason(deadline: float, cancel_event: threading.Event | None) -> str | None:
    """監視を打ち切るべき理由（`"time"`／`"cancelled"`／`None`）を 1 つの規則で決める。

    壁時計の締め切りをキャンセルより優先する（`crates/train` の
    `classify_cancel_outcome` と同じ優先関係。REQ-34・REQ-39）。締め切りの判定は
    `_remaining` と同じ時計を使う。
    """
    if _remaining(deadline) <= 0.0:
        return "time"
    if _is_cancelled(cancel_event):
        return "cancelled"
    return None


def _current_child_status(
    pid: int,
    *,
    timeout: float = _PS_TIMEOUT_SECONDS,
    cancel_event: threading.Event | None = None,
) -> tuple[int, bool] | None:
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
    stdout_text = _run_ps(pid, timeout, cancel_event)
    if stdout_text is None:
        return None
    text = stdout_text.strip()
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


def _terminate_and_reap(proc: subprocess.Popen) -> None:
    """`_terminate_worker`（`killpg`）を呼んでから `Popen.wait()` で回収
    する、**`proc` を回収する唯一の経路**（issue #178 PR #233 レビュー
    再々指摘 P0「`_current_child_status()` が `None` を返す場合に
    `proc.poll()` が worker を回収してしまい、`killpg` を通らず子孫が
    残る」）。

    `monitor_child` は `proc` を `poll()`／`wait()`／`communicate()` 等で
    直接回収してはならず、**すべての回収呼び出しを本関数に一元化する**。
    本モジュールが `proc` の唯一の回収者であるため、本関数が呼ばれる
    時点で `proc` はまだ回収されていない（実行中、またはゾンビとして
    存在する）ことが保証されており、`killpg`（回収より必ず先に行う）は
    常に安全である。`_worker` が孫プロセスを残さずに終了していた場合、
    `killpg` は ESRCH 相当（`ProcessLookupError`）になるだけで無害である。
    """
    _terminate_worker(proc)
    # 回収待ちは kill 時点から数える専用の締め切り（`_REAP_WAIT_SECONDS`）で
    # 上限を持つ。キャンセル経路ではこの前に状態確認等のブロックが入らない
    # （`monitor_child` はキャンセルを先に確認する）ため、猶予の総量は
    # 「キャンセル検知の刻み + 本待ち」に収まる。
    reap_deadline = time.monotonic() + _REAP_WAIT_SECONDS
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_remaining(reap_deadline))


def monitor_child(
    proc: subprocess.Popen,
    *,
    time_limit_seconds: float,
    rss_limit_bytes: int,
    poll_interval: float = _POLL_INTERVAL_SECONDS,
    grace_seconds: float = _TIME_LIMIT_GRACE_SECONDS,
    cancel_event: threading.Event | None = None,
) -> str | None:
    """`proc` を監視する。正常終了したら `None` を返す。

    壁時計・RSS のいずれかが上限を超えた場合、または `_worker` の状態
    自体を確認できない場合（`ps` の失敗・タイムアウト・出力を解釈できない）
    はプロセスグループを強制終了し、理由（`"time"`/`"rss"`/
    `"monitor_failed"`）を返す（呼び出し元が `proc.wait()` 済みであることを
    前提にせず、本関数が確実に終了させてから返る）。

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
    `is_zombie`。モジュール docstring「不変条件」節参照。issue #178
    PR #233 レビュー再指摘 P0）。**状態を確認できない（`None`）場合は
    正常終了として扱わず、`_terminate_and_reap` → `monitor_failed` に
    fail-closed する**（issue #178 PR #233 レビュー再々指摘 P0。`ps` の
    一時的な失敗であっても、区別する安全な手段が無いため 1 回の失敗で
    打ち切る〔既存の「監視できないこと自体を fail-closed に扱う」方針を
    踏襲し、リトライで猶予を与えない〕）。

    **ゾンビ検知は「締め切り内に終了した」ことを意味しない**（issue #178
    PR #233 レビュー再々々指摘 P1）: `ps` のポーリング間隔・
    `_current_child_status` 自体の所要時間により、実際には壁時計の
    締め切りを過ぎてから初めてゾンビ状態に気づく場合がある。この場合、
    終了理由（自発的な正常終了・`RLIMIT_CPU` 自己終了のいずれであっても）
    を受理して成果物を確定させてはならないため、`_classify_self_exit` を
    呼ぶ前に必ず `time.monotonic()` と `deadline` を比較し、締め切りを
    過ぎていれば通常終了として扱わず `"time"` を返す（呼び出し元
    `_monitor_worker_and_finalize` はこれを他の強制終了理由と同様に
    予約解放のみ〔確定しない〕の経路へ流す。REQ-39）。`killpg` → 回収
    （`_terminate_and_reap`）の順序はこの分岐でも変わらず守る。

    **協調キャンセル**（REQ-34・TASK-34.1-2・#145）: `cancel_event` が立って
    いたら `_terminate_and_reap` で worker を止めて `"cancelled"` を返す
    （回収経路を増やさない）。ゾンビを検知した時点でキャンセル済みなら
    `"cancelled"` を優先し、既に終了した worker の成果物を確定しに行かない。
    """
    cpu_baseline = resource.getrusage(resource.RUSAGE_CHILDREN)
    deadline = time.monotonic() + time_limit_seconds + grace_seconds
    while True:
        # ブロックする呼び出し（`ps`・待機）の前に、非ブロッキングの確認を
        # 先に済ませる（規則は `_stop_reason` の 1 か所）。
        reason = _stop_reason(deadline, cancel_event)
        if reason is not None:
            _terminate_and_reap(proc)
            return reason
        # `ps` の待ちは締め切りの残りとキャンセル要求で打ち切る。
        status = _current_child_status(
            proc.pid,
            timeout=_bounded_timeout(_PS_TIMEOUT_SECONDS, deadline),
            cancel_event=cancel_event,
        )
        if status is None:
            _terminate_and_reap(proc)
            # 状態が不明になった理由が締め切り切れ・キャンセルなら、監視失敗
            # ではなくそちらで報告する。`monitor_failed` は本当に `ps` が使えない
            # 場合だけ（stdout の drain 待ちを含み Rust 側の猶予を超えうるため、
            # キャンセルは高速な経路で予約を解放させる。REQ-34・#145）。
            return _stop_reason(deadline, cancel_event) or "monitor_failed"
        rss, is_zombie = status
        if is_zombie:
            # issue #178 PR #233 レビュー再々々指摘 P1: ゾンビ（終了済み）を
            # 検知しても、それだけで「締め切り内に正常終了した」とは限らない。
            # `ps` の呼び出し自体の所要時間により、締め切りを過ぎてから初めて
            # ゾンビだと気づく場合がある。この場合は成果物を確定させず壁時計
            # 超過（`"time"`）として fail-closed に扱う（REQ-39）。締め切り内なら
            # `_classify_self_exit` で通常終了か `RLIMIT_CPU` 自己終了かを判定する。
            _terminate_and_reap(proc)
            reason = _stop_reason(deadline, cancel_event)
            if reason is not None:
                return reason
            return _classify_self_exit(proc, cpu_baseline, time_limit_seconds)
        reason = _stop_reason(deadline, cancel_event)
        if reason is not None:
            _terminate_and_reap(proc)
            return reason
        if rss > rss_limit_bytes:
            _terminate_and_reap(proc)
            return "rss"
        # 次の確認まで眠る。キャンセルが来たら即座に起きる。
        _wait_cancel(cancel_event, _bounded_timeout(poll_interval, deadline))


class CancelSignal(threading.Event):
    """確定処理と排他できる協調キャンセルのイベント（REQ-34・#145）。

    `set()` は `commit_lock` を取る。確定側は `_commit_section` で同じロックを
    保持したままキャンセル確認と `finalize_out_dir` を行うため、EOF の到着が
    確認後〜rename 前に挟まっても、どちらが先に成立したかで結果が決まる。
    `cli.py::_start_cancel_watch` が生成する。
    """

    def __init__(self) -> None:
        super().__init__()
        self.commit_lock = threading.Lock()

    def set(self) -> None:
        with self.commit_lock:
            super().set()


@contextlib.contextmanager
def _commit_section(cancel_event: threading.Event | None) -> Iterator[bool]:
    """キャンセル確認と確定を同期区間にまとめ、区間開始時点のキャンセル有無を渡す。

    `CancelSignal` 以外（`None`・素の `Event`）ではロックを取らず単に確認する。
    """
    lock = getattr(cancel_event, "commit_lock", None)
    if lock is None:
        yield _is_cancelled(cancel_event)
        return
    with lock:
        yield _is_cancelled(cancel_event)


def _is_cancelled(cancel_event: threading.Event | None) -> bool:
    """協調キャンセルが要求されたか（`None` は常に `False`）。"""
    return cancel_event is not None and cancel_event.is_set()


def _join_unless_cancelled(
    thread: threading.Thread, timeout: float, cancel_event: threading.Event | None
) -> bool:
    """`thread` を最大 `timeout` 秒待つ。待つ間にキャンセルが来たら `False`。

    合流待ち（子孫が書き込み端を保持すると最大 `timeout` 秒かかる）が協調
    キャンセルの猶予を使い切らないよう、短い刻みでキャンセルを確認する
    （REQ-34）。`True` はキャンセルされずに待ちを終えたこと（タイムアウトを含む）。
    """
    deadline = time.monotonic() + timeout
    while thread.is_alive() and _remaining(deadline) > 0.0:
        if _is_cancelled(cancel_event):
            return False
        thread.join(timeout=_bounded_timeout(0.05, deadline))
    return not _is_cancelled(cancel_event)


def _report_cancelled(reservation: contract.OutDirReservation) -> ExitCode:
    """キャンセルされたジョブの予約を解放し、エラー JSON を出力する。

    メッセージは固定の英語文字列（データ本文・パスを含めない）。`code` は既存の
    語彙（`runtime_error`）だけを使い、終了コードの 70 は暫定値で Rust 側は
    これに依存しない（キャンセルの写像は TASK-33.x。REQ-21・REQ-34）。
    """
    released = contract.cleanup_reservation(reservation)
    # 解放を確認できない場合は所定の協調キャンセル応答（Rust 側 `is_cancel_ack`）を
    # 出さない。別メッセージにより呼び出し側は Unconfirmed（残置あり）として扱う
    # （REQ-34「協調キャンセルでは公開場所に何も残らない」・#145）。
    message = _CANCEL_ACK_MESSAGE if released else _CANCEL_CLEANUP_INCOMPLETE_MESSAGE
    _emit({"status": "error", "code": "runtime_error", "message": message})
    return ExitCode.RUNTIME_ERROR


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
        # 読み出しエラー。読めた分だけを使う。
        pass
    finally:
        # パイプは reader 自身が閉じる。別スレッドから `close()` を呼ぶと、
        # バッファ付きストリームの読み取りロック待ちで停止しうる（子孫が書き込み端を
        # 保持している場合。#145 PR #286 レビュー）ため、他スレッドは閉じない。
        with contextlib.suppress(OSError, ValueError):
            pipe.close()
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


def run_supervised_train(
    request_path: Path, *, cancel_event: threading.Event | None = None
) -> ExitCode:
    """`train` サブコマンドの本体。`out_dir` を予約したうえで `_worker` を
    子プロセスとして起動・監視し、結果に応じて確定または解放する。

    `cancel_event` は協調キャンセル（モジュール docstring 参照）で、
    `cli.py::_start_cancel_watch` が標準入力の EOF で立てる。`None` なら
    キャンセルを受け付けない。
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
    # 標準出力の保持上限は、リクエストごとに計算した値（Rust 側
    # `TrainRequest::max_result_bytes` と同じ式。`validation_inputs` が無ければ 1 MiB）。
    stdout_cap = request.max_result_bytes
    # train_path・root の fd はスーパーバイザーには不要（_worker が独立に
    # 検証・open し直す）。out_dir の fd だけは、直後の予約のために保持する。
    request.train_path.close()
    request.root.close()

    # 予約前のキャンセル確認: 予約も worker 起動もせず、所定のキャンセル応答を返す
    # （起動処理が Rust 側の猶予を超えて SIGKILL され予約が残るのを防ぐ。REQ-34・#145）。
    if _is_cancelled(cancel_event):
        request.out_dir.close()
        _emit({"status": "error", "code": "runtime_error", "message": _CANCEL_ACK_MESSAGE})
        return ExitCode.RUNTIME_ERROR

    try:
        reservation = contract.prepare_out_dir(request.out_dir)
    except WorkerError as e:
        request.out_dir.close()
        _emit({"status": "error", "code": e.code, "message": e.message})
        return e.exit_code

    try:
        # 予約後・worker 起動前のキャンセル確認（保持中の fd で予約を解放して応答する）。
        if _is_cancelled(cancel_event):
            return _report_cancelled(reservation)
        return _spawn_worker_and_finalize(
            raw_request,
            reservation,
            time_limit_seconds=float(time_limit_seconds),
            rss_limit_bytes=rss_limit_bytes,
            stdout_cap=stdout_cap,
            cancel_event=cancel_event,
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
    stdout_cap: int = _MAX_WORKER_STDOUT_BYTES,
    cancel_event: threading.Event | None = None,
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
    # `os.pipe()` はファイルディスクリプタ数の上限超過等で `OSError` を
    # 送出しうる（issue #178 PR #233 レビュー再々指摘 P1）。この時点で
    # 既に `out_dir` の予約（`reservation`）は確保済みのため、ここで
    # 送出された場合も他の起動失敗（`Popen` 失敗）と同様に
    # `cleanup_reservation` で解放してから `runtime_error` を返す
    # （fail-closed。予約だけが残置される事態を防ぐ。REQ-39）。
    try:
        lifeline_read_fd, lifeline_write_fd = os.pipe()
    except OSError as e:
        contract.cleanup_reservation(reservation)
        _emit(
            {
                "status": "error",
                "code": "runtime_error",
                "message": f"failed to create lifeline pipe: {type(e).__name__}",
            }
        )
        return ExitCode.RUNTIME_ERROR

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
            stdout_cap=stdout_cap,
            cancel_event=cancel_event,
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
    stdout_cap: int = _MAX_WORKER_STDOUT_BYTES,
    cancel_event: threading.Event | None = None,
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
        target=_drain_stdout, args=(proc.stdout, stdout_result, stdout_cap), daemon=True
    )
    reader_thread.start()

    killed_reason = monitor_child(
        proc,
        time_limit_seconds=time_limit_seconds,
        rss_limit_bytes=rss_limit_bytes,
        cancel_event=cancel_event,
    )

    if killed_reason == "cancelled":
        # 協調キャンセル: worker は `monitor_child` が `killpg` → 回収済み。
        # 破棄される標準出力の drain は待たず、先に保持中の fd で予約を解放する
        # （drain の最大 15 秒待ちが Rust 側の猶予予算 15 秒を超え SIGKILL され、
        # 空の予約が残るのを防ぐ。REQ-34）。確定へは進まない。
        try:
            return _report_cancelled(reservation)
        finally:
            # 解放後は他スレッドから `proc.stdout` を閉じない（reader が自身で
            # 閉じる。読み取り中の close は停止しうる）。reader は daemon で、
            # 待ちは短く抑える。
            reader_thread.join(timeout=1)

    # 通常は proc の終了（パイプの書き手が閉じる）で reader は自然に終わる。
    # 子孫が書き込み端を保持して残った場合も、他スレッドから閉じずに待ちを
    # 打ち切る。その際 `stdout_result` は未設定のまま JSON 検証で fail-closed になる。
    # 資源上限で止めた場合（`killed_reason` あり）は壁時計・資源超過の結果を
    # キャンセルより優先する（`classify_cancel_outcome` と同じ優先関係）。
    join_cancel = cancel_event if killed_reason is None else None
    if not _join_unless_cancelled(reader_thread, _READER_JOIN_SECONDS, join_cancel):
        # 待っている間にキャンセルが届いた。確定前なので公開せず解放する。
        return _report_cancelled(reservation)

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
        _emit(_forward_worker_error(payload))
        return ExitCode(returncode)

    # P0: 確定（rename）の直前に、保持し続けている tmp_fd（名前を再解決しない）
    # に対して model.onnx の SHA-256 が artifact.json の記録と一致するかを
    # 確認する（AGENTS.md ガード層「完全性と版」・REQ-39。artifact.py の
    # モジュール docstring 参照）。不一致・欠落は出力を確定させない。
    if _is_cancelled(cancel_event):
        return _report_cancelled(reservation)

    try:
        artifact_mod.verify_output(reservation.tmp_fd)
    except WorkerError as e:
        contract.cleanup_reservation(reservation)
        _emit({"status": "error", "code": e.code, "message": e.message})
        return e.exit_code

    # キャンセル確認と確定（rename）を同じ同期区間で行う。`CancelSignal.set()` も
    # 同じロックを取るため、確認後〜rename 前に届いた要求は rename の完了まで
    # 待たされ、「キャンセルが先か確定が先か」で結果が一意に決まる。確定後に
    # 届いたキャンセルは無視して成功を報告する（モジュール docstring）。
    with _commit_section(cancel_event) as cancelled:
        if cancelled:
            return _report_cancelled(reservation)
        try:
            contract.finalize_out_dir(reservation)
        except WorkerError as e:
            _emit({"status": "error", "code": e.code, "message": e.message})
            return e.exit_code

    _emit(payload)
    return ExitCode.OK
