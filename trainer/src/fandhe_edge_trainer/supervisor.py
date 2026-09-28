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
3. `<sys.executable> -I <trainer/launch.py の絶対パス> _worker --out-fd <tmp_fd>`
   （`worker_argv` が組み立てる argv。Issue #12: `-I`〔隔離モード〕で呼び出し元の
   `PYTHONPATH` 等に依存せず `trainer/src` を解決する）を子プロセスとして
   起動する。**`start_new_session` の要否は `SUPERVISOR_GROUP_MANAGED_ENV`
   環境変数で切り替える**（`_is_group_managed_by_rust` 参照。issue #178
   PR #233 レビュー再々々指摘 P0）。`pass_fds=(tmp_fd,)` で一時ディレクトリの
   fd だけを引き継がせる。リクエストの内容は `--request <path>` では渡さない。
   代わりに `raw` を `tempfile.TemporaryFile()`（作成直後に unlink 済みの無名
   一時ファイル。stdlib のみで完結し、`supervisor.py` が mlx・onnx・numpy を
   import しない設計を崩さない）へ書き込み、`seek(0)` してから子プロセスの
   標準入力（`stdin=`）として渡す（P1）。
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
6. 超過を検出したら `_terminate_worker` を呼ぶ（モードによって挙動が
   異なる。後述の「単独モード／管理モード」節参照）うえで、予約
   （一時ディレクトリとその中身・空の予約済みディレクトリ）を
   `contract.cleanup_reservation` で解放する（本モジュールが保持し続けている
   fd だけを使う。名前を再解決しない）。
7. 子プロセスが自分で終了した場合: 標準出力が「ちょうど 1 つの妥当な JSON
   オブジェクトである」ことを確認し、終了コードが 7 種のいずれかであることも
   確認する。いずれかを満たさない、またはシグナルによる終了なら
   `runtime_error`（exit 70）とし、予約を解放する。終了コードが 0 以外（7 種の
   いずれかのエラー）なら、そのコード・JSON をそのまま使い、予約は解放する
   （出力を確定させない）。終了コードが 0（成功）なら、確定の前に
   `artifact.verify_output` で `model.onnx` の SHA-256 が `artifact.json` の
   記録と一致するかを確認し（P0: AGENTS.md ガード層「完全性と版」・REQ-39。
   不一致・欠落は予約を解放して確定させない）、一致すれば
   `contract.finalize_out_dir` で確定させたうえで、そのまま出力する。

Rust 側ジョブ管理（REQ-34）が最終的にはこの「外側のスーパーバイザー」の役割を
担う計画であり、本モジュールは Rust 側が無い・本ワーカーが単独プロセスとして
起動される場合の防御として存在する。本モジュール自身が SIGKILL 等で道連れに
終了した場合の後始末は、本モジュールの責務ではなく Rust 側ジョブ管理
（TASK-34.x）に委ねる。

**単独モード／管理モード（`SUPERVISOR_GROUP_MANAGED_ENV`。issue #178 PR
#233 レビュー再々々指摘 P0「単独起動時の防御を弱めている」）**:
`_worker` をどのプロセスグループへ属させるか、`monitor_child` の内部
タイムアウトで何を kill するかは、環境変数
`SUPERVISOR_GROUP_MANAGED_ENV`（値が厳密に `"1"` の場合だけ「管理モード」。
未設定・その他の値は安全側の「単独モード」）で切り替える。

- **単独モード（既定・安全側）**: 本モジュールを単独で起動する運用
  （Rust 側ジョブ管理を経由しないテスト・手動実行を含む）を想定し、
  以前の挙動（`_worker` を `start_new_session=True` で別セッション・
  プロセスグループとして起動し、内部タイムアウト時に本モジュール自身が
  `os.killpg` でそのグループごと終了させる）を維持する。`_worker` の
  孫プロセスまで本モジュール単体で確実に掃除できる。
- **管理モード（Rust 側 `run_train` が起動した場合）**: `_worker` を
  `start_new_session` なしで起動し、本モジュールと同じプロセスグループ
  （Rust 側が `process_group(0)` で確立したもの）に留める。内部タイムアウト
  時は `_worker`（本モジュールの未回収の直接の子。`pid` の再利用は起こら
  ない）だけへ `SIGKILL` を送り、`_worker` がさらに起動した孫プロセスの
  確実な掃除は行わない。これは Rust 側 `run_train` が `_worker` を含む
  プロセスグループ全体へ `SIGKILL` を送ることで担う責務移動であり
  （`crates/train/src/process.rs` モジュール doc「プロセスグループによる
  一括終了」参照）、管理モードで単独起動された場合（通常あり得ないが）は
  孫プロセスが残りうる。
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

#: Rust 側 `run_train` がプロセスグループを管理していることを伝える環境
#: 変数名（issue #178 PR #233 レビュー再々々指摘 P0「単独起動時の防御を
#: 弱めている」）。Rust 側の同名の定数
#: （`crates/train/src/process.rs::SUPERVISOR_GROUP_MANAGED_ENV`）と
#: 一致することを、共有 fixture
#: `fixtures/train_contract/supervisor_group_managed_env.json` 経由で
#: `trainer/tests/test_train_contract_fixture.py`・
#: `crates/train/tests/train_contract_fixture.rs` の双方から照合する。
SUPERVISOR_GROUP_MANAGED_ENV = "FANDHE_EDGE_SUPERVISOR_GROUP_MANAGED"

#: [`SUPERVISOR_GROUP_MANAGED_ENV`] が「管理モード」を示す値。この文字列と
#: 完全一致する場合だけ管理モードとみなす（未設定・その他の値は安全側の
#: 単独モード）。
_SUPERVISOR_GROUP_MANAGED_VALUE = "1"


def _is_group_managed_by_rust() -> bool:
    """Rust 側 `run_train` がプロセスグループを管理しているか。

    厳密に `SUPERVISOR_GROUP_MANAGED_ENV` が `"1"` の場合だけ `True`
    （未設定・その他の値は `False`＝安全側の単独モード。issue #178 PR #233
    レビュー再々々指摘 P0）。
    """
    return os.environ.get(SUPERVISOR_GROUP_MANAGED_ENV) == _SUPERVISOR_GROUP_MANAGED_VALUE


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


def _current_child_rss_bytes(pid: int) -> int | None:
    """`ps -o rss= -p <pid>`（bytes 単位に変換済み）。取得できなければ None。

    `ps` の RSS 出力は KiB 単位（BSD/macOS・Linux とも `-o rss=` は KiB）。
    """
    try:
        result = subprocess.run(  # noqa: S603 - 引数は固定リスト。shell 不使用。絶対パスの /bin/ps のみを呼ぶ
            [_PS_BIN, "-o", "rss=", "-p", str(pid)],
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
    try:
        kib = int(text)
    except ValueError:
        return None
    return kib * 1024


def _terminate_worker(proc: subprocess.Popen) -> None:
    """モード（[`_is_group_managed_by_rust`]）に応じて `_worker` を終了させる
    （issue #178 PR #233 レビュー再々々指摘 P0「単独起動時の防御を弱めて
    いる」）。

    - **単独モード**（既定・安全側）: `_worker` は `start_new_session=True`
      で別プロセスグループとして起動されているため、本モジュール自身は
      巻き込まれない。`os.killpg` でそのグループごと終了させ、`_worker` が
      さらに起動した孫プロセスまで本モジュール単体で掃除する（以前の挙動
      を維持）。プロセスグループの取得自体に失敗した場合の保険として
      `proc.kill()` も呼ぶ。
    - **管理モード**: `_worker` は本モジュールと同じプロセスグループに
      留まる設計のため、`os.killpg` を使うと監視ループを実行している
      本プロセス自身も巻き込んで終了してしまう。`proc.kill()`
      （`os.kill(proc.pid, SIGKILL)` と同等）で直接の子だけを対象にする。
      `proc` は本プロセスの未回収の直接の子であり、`wait()` するまで
      `pid` が OS に返却されない（＝再利用されない）ため、`pid` ベースの
      kill でも無関係なプロセスを誤って終了させる心配はない。孫プロセスは
      ここでは掃除しない（Rust 側 `run_train` のプロセスグループ一括
      `SIGKILL` の責務）。
    """
    if _is_group_managed_by_rust():
        with contextlib.suppress(OSError):
            proc.kill()
        return
    with contextlib.suppress(OSError):
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()  # プロセスグループの取得自体に失敗した場合の保険


def _cpu_seconds_consumed_by_children(baseline: resource.struct_rusage) -> float:
    """`baseline`（監視開始時点の `RUSAGE_CHILDREN`）からの CPU 時間（user+sys 秒）の
    増分。`RUSAGE_CHILDREN` は「これまでに reap した子プロセス」の累積値のため、
    1 ジョブにつきワーカーを 1 つずつ順に起動する本モジュールの設計では、この
    増分は基本的に当該ワーカーに帰属する（`_current_child_rss_bytes` が起動する
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
    """
    cpu_baseline = resource.getrusage(resource.RUSAGE_CHILDREN)
    deadline = time.monotonic() + time_limit_seconds + grace_seconds
    while True:
        try:
            proc.wait(timeout=poll_interval)
            return _classify_self_exit(proc, cpu_baseline, time_limit_seconds)
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() > deadline:
            _terminate_worker(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
            return "time"
        rss = _current_child_rss_bytes(proc.pid)
        if rss is None and proc.poll() is not None:
            # `wait` のタイムアウト直後に子が終了すると `ps` は PID を見つけられない。
            # 終了済みなら監視失敗ではなく通常の終了として扱う（成功した学習の
            # 成果物を monitor_failed で捨てない）。
            return _classify_self_exit(proc, cpu_baseline, time_limit_seconds)
        if rss is None:
            # 監視できないこと自体を fail-closed に扱う（上限を検査できない
            # まま子プロセスを走らせ続けない）。
            _terminate_worker(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
            return "monitor_failed"
        if rss > rss_limit_bytes:
            _terminate_worker(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
            return "rss"


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


def worker_argv(out_fd: int) -> list[str]:
    """`_worker` を起動する argv を組み立てる（Issue #12）。

    `[sys.executable, "-I", <trainer/launch.py の絶対パス>, "_worker",
    "--out-fd", str(out_fd)]` を返す。`-m fandhe_edge_trainer` ではなく
    `trainer/launch.py`（`-I` 付き）を経由することで、呼び出し元の
    `PYTHONPATH` の設定漏れ・汚染に左右されず `trainer/src` を解決できる
    （`launch.py` のモジュール docstring 参照）。テスト（`tests/test_cli.py`
    の `test_worker_rejects_non_regular_stdin_without_blocking`）も本関数を
    再利用し、実際の起動経路と同じ argv で検証する。
    """
    return [
        sys.executable,
        "-I",
        str(_LAUNCH_SCRIPT),
        "_worker",
        "--out-fd",
        str(out_fd),
    ]


def _spawn_worker_and_finalize(
    raw_request: bytes,
    reservation: contract.OutDirReservation,
    *,
    time_limit_seconds: float,
    rss_limit_bytes: int,
) -> ExitCode:
    argv = worker_argv(reservation.tmp_fd)
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
                # 管理モード（Rust 側 `run_train` が起動した場合）だけ
                # `_worker` を本モジュールと同じプロセスグループに留める
                # （`start_new_session=False`）。単独モード（既定・安全側）
                # では従来どおり別セッションへ切り離し、本モジュール単体で
                # 孫プロセスまで掃除できるようにする（モジュール docstring
                # 「単独モード／管理モード」参照。issue #178 PR #233
                # レビュー再々々指摘 P0「単独起動時の防御を弱めている」）。
                group_managed = _is_group_managed_by_rust()
                proc = subprocess.Popen(  # noqa: S603 - 引数は固定リスト。shell 不使用。sys.executable は絶対パス
                    argv,
                    stdin=req_file,
                    stdout=subprocess.PIPE,
                    stderr=None,  # 継承（親の stderr へ直接流す。パイプを溜めて詰まらせない）
                    pass_fds=(reservation.tmp_fd,),
                    start_new_session=not group_managed,
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
    # `with` を抜けると req_file（親側の fd）は閉じるが、子プロセスは
    # 起動時に複製した自分の fd を保持しているため読み取りに支障はない。

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
