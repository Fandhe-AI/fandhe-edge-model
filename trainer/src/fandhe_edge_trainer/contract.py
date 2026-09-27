"""学習リクエスト（Rust 側 CLI が子プロセスへ渡す JSON）の解析・検証。

schema_version 1 の形（`__main__.py` の `train` サブコマンドが `--request <path>` で
受け取る JSON ファイルの中身。スーパーバイザー〔`supervisor.py`〕はこのファイルを
1 回だけ読み、検証済みの同じバイト列を子プロセス〔`_worker`〕の標準入力へ渡す。
`_worker` はファイルパスからの再読込を行わない。P1）:

```json
{
  "schema_version": 1,
  "kind": "c3",
  "kind_version": 1,
  "config": {},
  "label_order": ["a", "b"],
  "max_bytes": 512,
  "seed": 0,
  "device": "cpu",
  "root": "/abs/path/to/project",
  "train_path": "train.jsonl",
  "out_dir": "out",
  "time_limit_seconds": 3600,
  "rss_limit_bytes": 8589934592
}
```

`root` は呼び出し元が許可する作業ルート（絶対パス）。`train_path`・`out_dir` は
`root` からの**相対パス**でなければならない（絶対パス・`..` 構成要素・空文字列は
拒否する）。`TrainRequest.train_path`・`TrainRequest.out_dir` は
`guard.ConfinedEntry`（root 配下へ dir_fd で閉じ込め済みの「親ディレクトリ fd +
最終コンポーネント名」）を保持する（`Path` 文字列ではない。TOCTOU 対策。
`guard.py` のモジュール docstring 参照）。`TrainRequest.root` は開いたままの
`guard.RootHandle`。**いずれも使い終わったら `TrainRequest.close_resources()` で
明示的に fd を閉じること**（`_worker`〔`cli.py::run_worker_train`〕・
スーパーバイザー〔`supervisor.py`〕がそれぞれ `finally` で行う）。

**`validate_request` は `train`（スーパーバイザー）・`_worker`（実際の学習・
書き出し）の両方から呼ばれる単一の検証経路**（`out_dir` の確認は本関数では
行わない。スーパーバイザーが `request.out_dir` を使って `prepare_out_dir` を
呼ぶ側であり、`_worker` は検証だけ行って `request.out_dir` を使わずに閉じる）。
スーパーバイザーは `read_request_bytes`（ファイル読み取り）→ `parse_request_bytes`
（JSON パース）→ `validate_request`（フィールド検証）の 3 段を経てから、
パース前の生バイト列を子プロセスの標準入力へ渡す。`_worker` は標準入力から
読んだバイト列に対して `parse_request_bytes` → `validate_request` を呼ぶ
（ファイルパスは一切受け取らない）。

`time_limit_seconds`・`rss_limit_bytes` は任意項目（省略時は `limits.py` の
`MAX_TRAIN_WALL_SECONDS`・`MAX_TRAIN_RSS_BYTES` を既定値として使う）。
指定する場合は、その上限を**下げる**ことしかできない（上限より大きい値は
`invalid_request` で拒否する）。学習ループ中の実際の検査は `budget.py`
（`ResourceBudget`）が担う。

学習データ（`train_path` が指す JSONL）は 1 行 1 オブジェクト `{"input": str,
"label": str}`。label は label_order に含まれる必要がある。

本モジュールは validation・test データを一切読まない（学習は train データのみで行う。
評価・選定・作り直し判定は評価器（TASK-24.1）・学習ワーカーの選定処理（別 TASK）の
責務であり、ここでは扱わない）。

**経路の閉じ込め（REQ-39 のガード層・PoC-20）は多層防御とする**: 一次防御は
Rust 側 CLI（呼び出し元）が担う設計だが、本ワーカーは単独プロセスとしても
起動されうるため、本ワーカー自身も `guard.py::confine` で `root` 配下への
閉じ込めを検証する（Rust 側の検証済みという前提だけに頼らない）。検証（fd を開く
時点）と使用（実際の読み書き・`mkdir`・`rename`）を同じ fd に束縛することで、
検証後に経路の途中がシンボリックリンクへ差し替えられる TOCTOU を防ぐ。

**`out_dir` の所有権はスーパーバイザーに一元化する**（P0-1・P0-2 の見直し）:
`prepare_out_dir`・`finalize_out_dir`・`cleanup_reservation` は
**スーパーバイザー（`supervisor.py`）だけが呼ぶ**。`_worker` は `out_dir` の
予約・確定・後始末のいずれにも関与しない。理由: `_worker` は壁時計・RSS 超過で
いつ強制終了されてもおかしくない別プロセスであり、強制終了された時点で
`_worker` が持っていた fd はすべて失われる。強制終了後に「経路の名前を頼りに
予約の後始末を再構築する」設計は、名前が一致するというだけの根拠で別物を
消してしまう TOCTOU を生む（実際に P0-1 として指摘された）。スーパーバイザーは
`_worker` を監視するだけで自身は強制終了されない前提のプロセスなので、
予約に使った fd（`parent_fd`・`tmp_fd`）をジョブの最初から最後まで手放さずに
持ち続けられる。強制終了は `_worker` だけに afflict し、スーパーバイザー自身が
（SIGKILL 等で）道連れに終了した場合の後始末は、本モジュールの責務ではなく
Rust 側ジョブ管理（TASK-34.x REQ-34）に委ねる。予約済み `out_dir` が空のまま
残ることは「成果物は公開されていない」ことの証拠になる。

`_worker` は、スーパーバイザーが `pass_fds` で渡した一時ディレクトリの fd
（`--out-fd <n>`）へ `artifact.json`・`model.onnx` を書き込むだけで、`out_dir` の
名前を一切扱わない（`cli.py::run_worker_train` 参照）。

**`out_dir` の確定は「空ディレクトリの予約 → 別ディレクトリで作業 → アトミックな
置き換え」の 3 段で行う**（TOCTOU 対策）。`prepare_out_dir` が `os.mkdir`
（`dir_fd` 相対）で `out_dir` そのものを排他的に作成する（既存なら
`FileExistsError` を検出できる。「存在確認 → 後で作成」という 2 手順では、
その間に別プロセスが先に作成できてしまう）。学習・書き出しは兄弟の一時
ディレクトリ（同じ `parent_fd` 配下）で行い、`finalize_out_dir` が予約時に
記録した `(st_dev, st_ino)` と現在の `out_dir`・一時ディレクトリのそれを
突き合わせてから（別物にすり替わっていないか。`OutDirReservation` のクラス
docstring 参照）`os.rename`（`dir_fd` 相対）で確定する。置き換え先が空
ディレクトリでなくなっていた場合（何者かが書き込んだ）は `os.rename` 自体が
`ENOTEMPTY` で失敗するため、その内容を消さずに残す。

**`os.replace` ではなく `os.rename` を使う**: `os.replace` は `dir_fd` 引数を
サポートしない（`os.replace not in os.supports_dir_fd`。本開発機の macOS で
実測確認済み）。POSIX の `rename(2)` は元々、置き換え先が空ディレクトリであれば
上書きする仕様のため（`os.replace` が Windows 向けに追加している「常に上書きする」
という意味論は POSIX の `rename` に元から備わっている）、`dir_fd` に対応した
`os.rename` で同じ効果が得られる。
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from . import budget, guard
from .encoding import normalize_input
from .errors import WorkerError, truncate_list_for_message
from .exitcode import ExitCode
from .limits import (
    MAX_LABEL_BYTES,
    MAX_LABELS,
    MAX_MAX_BYTES,
    MAX_REQUEST_BYTES,
    MAX_SEED,
    MAX_TRAIN_DATA_BYTES,
    MAX_TRAIN_EXAMPLES,
    MAX_TRAIN_LINE_BYTES,
    MAX_TRAIN_RSS_BYTES,
    MAX_TRAIN_WALL_SECONDS,
    MIN_LABELS,
    MIN_MAX_BYTES,
    MIN_SEED,
)

SCHEMA_VERSION = 1
_ALLOWED_DEVICES = ("cpu", "gpu")
_REQUEST_FIELDS = {
    "schema_version",
    "kind",
    "kind_version",
    "config",
    "label_order",
    "max_bytes",
    "seed",
    "device",
    "root",
    "train_path",
    "out_dir",
    "time_limit_seconds",
    "rss_limit_bytes",
}


@dataclass(frozen=True)
class TrainExample:
    """学習データ 1 件（正解ラベル付き）。"""

    input: str
    label: str


@dataclass(frozen=True)
class TrainRequest:
    """検証済みの学習リクエスト。

    `root`・`train_path`・`out_dir` は開いたままの fd を保持する
    （`guard.RootHandle`・`guard.ConfinedEntry`）。使い終わったら必ず
    `close_resources()` を呼ぶこと（`cli.py::run_worker_train` が `finally` で行う）。
    """

    kind: str
    kind_version: int
    config: dict[str, Any]
    label_order: list[str]
    max_bytes: int
    seed: int
    device: str
    root: guard.RootHandle
    train_path: guard.ConfinedEntry
    out_dir: guard.ConfinedEntry
    time_limit_seconds: int
    rss_limit_bytes: int

    def close_resources(self) -> None:
        """保持している fd（`train_path`・`out_dir`・`root`）をすべて閉じる。

        各 `close()` は冪等なので、`load_train_examples`/`prepare_out_dir` 等が
        個別に（用が済み次第）先に閉じていても問題ない。
        """
        self.train_path.close()
        self.out_dir.close()
        self.root.close()


def _invalid(message: str) -> WorkerError:
    return WorkerError("invalid_request", message, ExitCode.INVALID_INPUT)


def _limit(message: str) -> WorkerError:
    return WorkerError("limit_exceeded", message, ExitCode.LIMIT_EXCEEDED)


def _invalid_data(message: str) -> WorkerError:
    return WorkerError("invalid_data", message, ExitCode.INVALID_INPUT)


def _reject_special_floats(token: str) -> None:
    """`json.loads` の `parse_constant`: `NaN`/`Infinity`/`-Infinity`（JSON 標準外の
    数値トークン）を拒否する。既定の `json` モジュールはこれらを黙って受け付けるため、
    明示的に拒否しないと後続の検証（範囲比較）をすり抜けうる（例: `nan <= 0` は
    常に False なので、下限チェックだけでは弾けない）。
    """
    raise ValueError(f"disallowed numeric constant in JSON: {token}")


def _json_loads_strict(text: str) -> Any:
    """JSON を厳密に解析する（NaN 等の拒否・過度なネストの検出）。

    呼び出し側は `json.JSONDecodeError`（`ValueError` のサブクラス）を先に、
    次に `ValueError` を捕捉すること（本関数はどちらも送出しうる）。
    """
    try:
        return json.loads(text, parse_constant=_reject_special_floats)
    except RecursionError as e:
        # 深すぎるネストによる再帰上限到達。ValueError へ正規化して呼び出し側の
        # 例外処理を単純にする（RecursionError は json.JSONDecodeError の
        # サブクラスではないため、変換しないと別経路の捕捉が要る）。
        raise ValueError(f"JSON nesting too deep: {type(e).__name__}") from e


def _open_regular_file(path: Path, error_maker: Any) -> tuple[IO[bytes], os.stat_result]:
    """（root 非配下の）通常のパスからファイルを開く。`read_request_bytes` 専用。

    `--request <path>` は Rust 側 CLI が直接渡す絶対パスであり、`root` 配下への
    閉じ込め対象ではない（root 配下の `train_path`/`out_dir` は
    `_open_confined_regular_file`/`prepare_out_dir` が dir_fd ベースで扱う）。

    open → fstat の順にすることで、path に対する stat → open の間に別ファイルへ
    差し替えられる TOCTOU を避ける。`O_NONBLOCK` を付けて開く: 読み取り用に FIFO を
    通常どおり（ブロッキングで）開くと、書き込み側が現れるまで `open()` 自体が
    無期限にブロックする（`無制限待ちを作らない`。REQ-39）。`O_NONBLOCK` を付けると
    FIFO の open は書き込み側の有無によらず即座に返るため、その直後の
    `fstat`/`S_ISREG` 判定で確実に拒否できる。通常ファイルの読み取りには
    `O_NONBLOCK` は影響しない（意味を持つのは FIFO・一部のデバイス・ソケットのみ）。
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        raise error_maker(f"file not readable: {type(e).__name__}") from e
    try:
        st = os.fstat(fd)
    except OSError as e:
        os.close(fd)
        raise error_maker(f"file not stat-able: {type(e).__name__}") from e
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise error_maker("file is not a regular file")
    f = os.fdopen(fd, "rb")
    return f, st


def _open_confined_regular_file(
    entry: guard.ConfinedEntry, error_maker: Any
) -> tuple[IO[bytes], os.stat_result]:
    """`guard.confine` で得た `ConfinedEntry`（親ディレクトリ fd + 最終コンポーネント名）
    から、シンボリックリンクを追跡せずに通常ファイルとして開く（`train_path` 用）。

    `parent_fd` はこの `open` 呼び出しのためだけに必要なので、成功・失敗いずれの
    場合も呼び出し直後に閉じる（`entry.close()`。以後の読み取りは返されたファイル
    オブジェクト自身の fd で完結する）。
    """
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    try:
        fd = os.open(entry.name, flags, dir_fd=entry.parent_fd)
    except OSError as e:
        raise error_maker(f"file not readable: {type(e).__name__}") from e
    finally:
        entry.close()
    try:
        st = os.fstat(fd)
    except OSError as e:
        os.close(fd)
        raise error_maker(f"file not stat-able: {type(e).__name__}") from e
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise error_maker("file is not a regular file")
    f = os.fdopen(fd, "rb")
    return f, st


def read_request_bytes(path: Path) -> bytes:
    """学習リクエスト JSON ファイルをサイズ上限付きで読み、生バイト列を返す（パースしない）。

    **`supervisor.py::run_supervised_train` はファイルをここで 1 回だけ読み、
    その同じバイト列を `parse_request_bytes` でパース・検証したうえで、子プロセス
    （`_worker`）の標準入力へそのまま渡す**（P1: リクエストファイルへの経路を
    スーパーバイザーと子プロセスがそれぞれ再読込すると、検証後にファイルが
    書き換えられた場合、両者が異なる内容を見てしまう TOCTOU になる。検証済みの
    バイト列を固定して子へ渡すことでこれを防ぐ）。
    """
    f, st = _open_regular_file(path, _invalid)
    try:
        if st.st_size > MAX_REQUEST_BYTES:
            raise _limit(f"request file exceeds {MAX_REQUEST_BYTES} bytes limit")
        # stat 後にファイルが伸長される可能性（TOCTOU の残り）に備え、実際の読み取りも
        # limit+1 バイトで打ち切って超過を検出する（読み取り量そのものを上限で縛る）。
        raw = f.read(MAX_REQUEST_BYTES + 1)
    finally:
        f.close()
    if len(raw) > MAX_REQUEST_BYTES:
        raise _limit(f"request file exceeds {MAX_REQUEST_BYTES} bytes limit")
    return raw


def parse_request_bytes(raw: bytes) -> Any:
    """リクエストの生バイト列（`read_request_bytes` が返したもの、または子プロセスが
    標準入力から読んだもの）をパースする（フィールド検証はしない）。

    `_worker`（`cli.py::run_worker_train`）は標準入力から読んだバイト列を直接
    本関数へ渡す（サイズ上限は呼び出し側〔標準入力からの読み取り量〕で既に
    掛かっている前提。`limits.MAX_REQUEST_BYTES` は共通）。
    """
    if len(raw) > MAX_REQUEST_BYTES:
        raise _limit(f"request file exceeds {MAX_REQUEST_BYTES} bytes limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _invalid(f"request file not readable as utf-8: {type(e).__name__}") from e
    try:
        return _json_loads_strict(text)
    except json.JSONDecodeError as e:
        raise _invalid(f"request is not valid JSON: {e.msg} at line {e.lineno}") from e
    except ValueError as e:
        raise _invalid(f"request JSON rejected: {type(e).__name__}") from e


def load_request(path: Path) -> TrainRequest:
    """学習リクエスト JSON ファイルを読み、検証したうえで `TrainRequest` を返す。

    `_worker` は本関数を使わず、標準入力から読んだバイト列を
    `parse_request_bytes` → `validate_request` へ直接渡す（`cli.py::run_worker_train`
    参照。ファイルパスから読み直さない。P1）。
    """
    raw = read_request_bytes(path)
    parsed = parse_request_bytes(raw)
    return validate_request(parsed)


def validate_request(raw: Any) -> TrainRequest:
    if not isinstance(raw, dict):
        raise _invalid("request must be a JSON object")
    unknown = set(raw) - _REQUEST_FIELDS
    if unknown:
        # フィールド名はリクエスト JSON の全体サイズ上限（MAX_REQUEST_BYTES）まで
        # 利用者が自由に長くできるため、切り詰めてから埋め込む（P1-1）。
        raise _invalid(f"request has unknown fields: {truncate_list_for_message(sorted(unknown))}")

    schema_version = raw.get("schema_version")
    # `bool` は `int` のサブクラスであり、`True == 1` が成り立つ（`True !=
    # SCHEMA_VERSION` は False になり、`bool` を素通りさせてしまう）。
    # `type(v) is int` で明示的に除外してから比較する（P1）。文字列等の場合は
    # 値そのものを埋め込まない（無制限の長さになりうるため。型名だけ示す）。
    if type(schema_version) is not int:
        raise _invalid(
            f"schema_version must be an integer (expected {SCHEMA_VERSION}), "
            f"got {type(schema_version).__name__}"
        )
    if schema_version != SCHEMA_VERSION:
        raise _invalid(f"unsupported schema_version: {schema_version} (expected {SCHEMA_VERSION})")

    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        raise _invalid("kind must be a non-empty string")

    kind_version = raw.get("kind_version")
    if not isinstance(kind_version, int) or isinstance(kind_version, bool):
        raise _invalid("kind_version must be an integer")

    config = raw.get("config", {})
    if not isinstance(config, dict):
        raise _invalid("config must be an object")

    label_order = raw.get("label_order")
    _validate_label_order(label_order)

    max_bytes = raw.get("max_bytes")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
        raise _invalid("max_bytes must be an integer")
    if not (MIN_MAX_BYTES <= max_bytes <= MAX_MAX_BYTES):
        raise _invalid(f"max_bytes out of range [{MIN_MAX_BYTES}, {MAX_MAX_BYTES}]")

    seed = raw.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise _invalid("seed must be an integer")
    if not (MIN_SEED <= seed <= MAX_SEED):
        raise _invalid(f"seed out of range [{MIN_SEED}, {MAX_SEED}]")

    device = raw.get("device")
    if device not in _ALLOWED_DEVICES:
        raise _invalid(f"device must be one of {_ALLOWED_DEVICES}")

    # ジョブ全体の資源上限（REQ-39・budget.py）。任意項目で「既定の上限より
    # 下げる」ことだけを許す（上限そのものを緩める経路は無い）。
    time_limit_seconds = raw.get("time_limit_seconds", MAX_TRAIN_WALL_SECONDS)
    if (
        not isinstance(time_limit_seconds, int)
        or isinstance(time_limit_seconds, bool)
        or not (1 <= time_limit_seconds <= MAX_TRAIN_WALL_SECONDS)
    ):
        raise _invalid(f"time_limit_seconds must be an integer in [1, {MAX_TRAIN_WALL_SECONDS}]")

    rss_limit_bytes = raw.get("rss_limit_bytes", MAX_TRAIN_RSS_BYTES)
    if (
        not isinstance(rss_limit_bytes, int)
        or isinstance(rss_limit_bytes, bool)
        or not (1 <= rss_limit_bytes <= MAX_TRAIN_RSS_BYTES)
    ):
        raise _invalid(f"rss_limit_bytes must be an integer in [1, {MAX_TRAIN_RSS_BYTES}]")

    # 経路の閉じ込め（REQ-39 ガード層・PoC-20。多層防御。guard.py 参照）は最後に
    # 行う: ここより前の検証で弾かれるリクエストのために fd を開いて後始末する
    # 手間を避ける。ここから先で失敗したら、それまでに開いた fd をすべて
    # 閉じてから再送出する。
    root_handle = guard.resolve_root(raw.get("root"))
    try:
        train_path_entry = guard.confine(root_handle, raw.get("train_path"), "train_path")
        try:
            out_dir_entry = guard.confine(root_handle, raw.get("out_dir"), "out_dir")
        except BaseException:
            train_path_entry.close()
            raise
    except BaseException:
        root_handle.close()
        raise

    return TrainRequest(
        kind=kind,
        kind_version=kind_version,
        config=config,
        label_order=list(label_order),
        max_bytes=max_bytes,
        seed=seed,
        device=device,
        root=root_handle,
        train_path=train_path_entry,
        out_dir=out_dir_entry,
        time_limit_seconds=time_limit_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )


def _validate_label_order(label_order: Any) -> None:
    if not isinstance(label_order, list):
        raise _invalid("label_order must be a list")
    if not (MIN_LABELS <= len(label_order) <= MAX_LABELS):
        raise _invalid(f"label_order length out of range [{MIN_LABELS}, {MAX_LABELS}]")
    seen: set[str] = set()
    for i, label in enumerate(label_order):
        if not isinstance(label, str) or not label:
            raise _invalid(f"label_order[{i}] must be a non-empty string")
        if len(label.encode("utf-8")) > MAX_LABEL_BYTES:
            raise _invalid(f"label_order[{i}] exceeds {MAX_LABEL_BYTES} utf-8 bytes")
        if label in seen:
            raise _invalid(f"label_order[{i}] is a duplicate")
        seen.add(label)


def load_train_examples(
    entry: guard.ConfinedEntry,
    label_order: list[str],
    *,
    resource_budget: budget.ResourceBudget | None = None,
    check_every: int = 1024,
) -> list[TrainExample]:
    """学習データ（JSONL）を読み、検証したうえで `TrainExample` の一覧を返す。

    `entry` は `guard.confine` で得た `train_path` の `ConfinedEntry`（本関数が
    ファイルを開いた直後に `parent_fd` を閉じる。`_open_confined_regular_file` 参照）。

    データ本文（`input`・`label` の値）はエラーメッセージへ一切含めない
    （個人情報・機密情報を含みうる。security.md）。行番号・件数だけを報告する。

    資源上限（REQ-39・limits.py）を 4 段で適用する: (0) `fstat` の `st_size` に
    よる早期判定（1 行も読まずに拒否できる。読み取り前チェック）、(1) 1 行あたりの
    バイト数（JSON パース・NFKC 正規化の前に検査。巨大な 1 行によるメモリ確保を
    防ぐ）、(2) 累積読み取りバイト数（`st_size` の後にファイルが伸長される場合に
    備え、実際に読んだ量でも判定する）、(3) 検証を通過した examples の件数。

    `resource_budget` を渡すと、`check_every` 行ごと（既定 1024）に
    `resource_budget.check()`（壁時計・RSS。`budget.py` 参照）を呼ぶ（P0-1:
    最大 64 MiB・20 万件の学習データを読み込む処理自体も、学習ジョブ全体の
    資源上限の対象にする。呼び出し元〔`cli.py`〕が学習ループ・ONNX 書き出しと
    **同じ** `ResourceBudget` インスタンスを渡すことで、リクエスト全体を通じた
    単一の予算として扱う）。
    """
    label_set = set(label_order)
    examples: list[TrainExample] = []
    total_bytes = 0
    lineno = 0

    f, st = _open_confined_regular_file(entry, _invalid)
    if st.st_size > MAX_TRAIN_DATA_BYTES:
        f.close()
        raise _limit(f"train data exceeds {MAX_TRAIN_DATA_BYTES} bytes limit")
    try:
        while True:
            raw_line = f.readline(MAX_TRAIN_LINE_BYTES + 1)
            if raw_line == b"":
                break
            lineno += 1
            total_bytes += len(raw_line)
            if total_bytes > MAX_TRAIN_DATA_BYTES:
                raise _limit(f"train data exceeds {MAX_TRAIN_DATA_BYTES} bytes limit")
            if len(raw_line) > MAX_TRAIN_LINE_BYTES:
                raise _limit(f"train data line {lineno} exceeds {MAX_TRAIN_LINE_BYTES} bytes limit")

            line_bytes = raw_line.rstrip(b"\n").rstrip(b"\r")
            try:
                decoded = line_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                raise _invalid_data(
                    f"train data line {lineno} not readable as utf-8: {type(e).__name__}"
                ) from e
            if decoded.strip() == "":
                raise _invalid_data(f"train data has a blank line at line {lineno}")

            try:
                row = _json_loads_strict(decoded)
            except json.JSONDecodeError as e:
                raise _invalid_data(f"train data line {lineno} is not valid JSON: {e.msg}") from e
            except ValueError as e:
                raise _invalid_data(f"train data line {lineno} rejected: {type(e).__name__}") from e

            if not isinstance(row, dict) or set(row) != {"input", "label"}:
                raise _invalid_data(
                    f"train data line {lineno} must have exactly 'input' and 'label' fields"
                )
            text = row["input"]
            label = row["label"]
            if not isinstance(text, str):
                raise _invalid_data(f"train data line {lineno}: 'input' must be a string")
            if not isinstance(label, str):
                raise _invalid_data(f"train data line {lineno}: 'label' must be a string")
            if label not in label_set:
                raise _invalid_data(f"train data line {lineno}: label not in label_order")
            if normalize_input(text) == "":
                # 空白のみ等、正規化後に空となる入力はエンコード後 [0]（詰め物だけ）に
                # なり学習上意味を持たない（kinds/c3.py の詰め物マスク処理参照）。
                raise _invalid_data(f"train data line {lineno}: input is empty after normalization")
            if len(examples) >= MAX_TRAIN_EXAMPLES:
                raise _limit(f"train data exceeds {MAX_TRAIN_EXAMPLES} examples limit")
            examples.append(TrainExample(input=text, label=label))
            if resource_budget is not None and lineno % check_every == 0:
                resource_budget.check()
    finally:
        f.close()

    if resource_budget is not None:
        resource_budget.check()  # 端数分（check_every の倍数に満たない残り）の確認

    if not examples:
        raise WorkerError(
            "invalid_data", "train data must contain at least 1 example", ExitCode.INVALID_INPUT
        )
    distinct_labels = {ex.label for ex in examples}
    if len(distinct_labels) < 2:
        raise WorkerError(
            "invalid_data",
            "train data must contain at least 2 distinct labels",
            ExitCode.INVALID_INPUT,
        )
    return examples


#: `out_dir` の実体識別子（`st_dev`・`st_ino`）。予約時点のものを覚えておき、
#: 確定（`finalize_out_dir`）の直前に「まだ自分が予約した実体と同じか」を
#: 確認するために使う（TOCTOU 対策）。
ReservedId = tuple[int, int]


@dataclass
class OutDirReservation:
    """予約済みの `out_dir` と、その作業用一時ディレクトリの一式。

    `entry.parent_fd` を経由してのみ `out_dir`・一時ディレクトリを操作する
    （`entry` が指すディレクトリの外へ書き出す経路は無い）。

    **本予約の全ライフサイクル（作成・子プロセスへの fd 引き渡し・確定・
    後始末）はスーパーバイザー（`supervisor.py`）が一貫して保持する**
    （P0-1・P0-2 の見直し。以前はワーカー〔`_worker`〕がこれを保持していたが、
    ワーカーは強制終了されうる別プロセスであり、その場合ワーカー内の
    `tmp_fd`・`OutDirReservation` は失われる。そのため、強制終了後の後始末を
    「経路の名前を頼りに再構築する」処理が必要になり、名前が一致するというだけで
    他人のディレクトリを消してしまう TOCTOU が残っていた。fd を最初から
    最後まで手放さないスーパーバイザーだけがこの予約を扱うことで、
    「経路を再構築して後始末する」処理自体が丸ごと不要になる。ワーカーは
    `--out-fd <n>` で渡された一時ディレクトリの fd へ書き込むだけで、
    `out_dir` の名前・予約・確定・後始末のいずれにも一切関与しない）。
    """

    entry: guard.ConfinedEntry
    tmp_name: str
    tmp_fd: int
    tmp_id: ReservedId
    reserved_id: ReservedId
    _tmp_closed: bool = field(default=False, init=False, repr=False)

    def close_tmp_fd(self) -> None:
        if not self._tmp_closed:
            self._tmp_closed = True
            with contextlib.suppress(OSError):
                os.close(self.tmp_fd)


def _tmp_name_prefix(name: str) -> str:
    return f".{name}.tmp-"


def prepare_out_dir(entry: guard.ConfinedEntry) -> OutDirReservation:
    """`out_dir` を空ディレクトリとして排他的に予約し、書き込み用の一時
    ディレクトリ（兄弟。同じ `entry.parent_fd` 配下）を作って返す。

    呼び出し元は**スーパーバイザーに限る**（`OutDirReservation` のクラス
    docstring 参照）。返り値の `tmp_fd`・`parent_fd` はジョブが終わる
    （`finalize_out_dir` 成功 or `cleanup_reservation`）まで呼び出し元が
    保持し続けること。

    「存在確認 → 後で作成」という 2 手順では、その間に別プロセスが先に
    `out_dir` を作れてしまう（TOCTOU）。`os.mkdir`（`dir_fd` 相対）は対象が
    既に存在すれば `FileExistsError` を返すことがカーネルにより保証された
    アトミックな操作のため、予約自体を 1 手順で行う。学習・書き出しは兄弟の
    一時ディレクトリで行い、成功時に `finalize_out_dir` がこの予約済み空
    ディレクトリを `os.rename` で置き換える（本モジュールの docstring も参照）。

    **予約前に親ディレクトリの所有者・書き込み権限を確認する**（P0。
    `os.mkdir`・`os.rename` はいずれも `dir_fd` の指す実体そのものへは
    書き込むが、その親ディレクトリに他ユーザーが書き込める場合、`finalize_out_dir`
    の `stat` から `os.rename` までの間に、同じ親ディレクトリ配下で
    `tmp_name`（予約した一時ディレクトリと同名）を差し替えられる余地が残る
    〔`finalize_out_dir` のドキュメントコメント参照〕。親ディレクトリの実効的な
    所有者が自分自身（`os.geteuid()`）であり、かつ group/others に書き込み
    権限が無い（sticky な `/tmp` 型の共有ディレクトリを含め弾く）ことを
    `os.fstat(parent_fd)` で確認してから初めて `os.mkdir` する。確認に失敗
    した場合・条件を満たさない場合は、何も作成せず `output_conflict`
    （fail-closed。REQ-39 ガード層）。

    **既知の限界（セキュリティ監査指摘）**: 本チェックは POSIX のパーミッション
    ビット（`st_mode`）のみを見る。macOS の ACL（`chmod +a` 等で付与される
    追加の書き込み許可）はパーミッションビットに現れないため、ACL 経由で
    group/others に書き込みを許可された親ディレクトリはここでは検出できない
    （本チェックをすり抜けうる）。その場合でも、`finalize_out_dir` が
    `os.rename` 成功後に実体の `(st_dev, st_ino)` を照合するため（本モジュールの
    `finalize_out_dir` docstring 参照）、ACL によって差し替えられた実体が
    そのまま公開されることはなく、不一致を検出して `output_conflict`
    （fail-closed）になる。多層防御の 1 段目（本チェック）を回避されても、
    2 段目（rename 後照合）が最終的な公開の正しさを担保する。
    また、umask（例: `002`）の設定によっては `mkdir(parents=True)` 等で
    作られた親ディレクトリが `0o775`（group 書き込み可能）になりうる。
    その場合、たとえ自分が作った・自分が所有するディレクトリであっても
    本チェックは `output_conflict` として拒否する（緩和しない。呼び出し元が
    `root` に渡すディレクトリは `0o700` 等、group/others に書き込み権限の
    無い状態で用意すること）。

    `os.mkdir(name, dir_fd=...)` の成功後に発生したあらゆる例外（`stat` の失敗・
    一時ディレクトリの作成・open の失敗）は、予約（`out_dir`・一時ディレクトリ）を
    解放してから再送出する（残置しない）。
    """
    parent_fd = entry.parent_fd
    name = entry.name

    try:
        parent_st = os.fstat(parent_fd)
    except OSError as e:
        raise WorkerError(
            "output_conflict",
            f"out_dir parent not stat-able: {type(e).__name__}",
            ExitCode.INVALID_INPUT,
        ) from e
    if parent_st.st_uid != os.geteuid() or (parent_st.st_mode & 0o022) != 0:
        # 他ユーザーが所有する、または group/others に書き込み可能な親
        # ディレクトリでは、確定直前の名前差し替え（TOCTOU）を防ぎきれない
        # ため、予約自体を拒否する（何も作成しない。REQ-39 ガード層）。
        raise WorkerError(
            "output_conflict",
            "out_dir parent must be owned by the current user and not writable by group or others",
            ExitCode.INVALID_INPUT,
        )

    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError as e:
        raise WorkerError(
            "output_conflict", "out_dir already exists", ExitCode.INVALID_INPUT
        ) from e
    except OSError as e:
        raise _invalid(f"failed to create out_dir: {type(e).__name__}") from e

    tmp_name: str | None = None
    tmp_fd: int | None = None
    try:
        st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        reserved_id: ReservedId = (st.st_dev, st.st_ino)
        tmp_name = f"{_tmp_name_prefix(name)}{secrets.token_hex(8)}"
        os.mkdir(tmp_name, 0o700, dir_fd=parent_fd)
        tmp_fd = os.open(tmp_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        tmp_st = os.fstat(tmp_fd)
        tmp_id: ReservedId = (tmp_st.st_dev, tmp_st.st_ino)
    except BaseException as e:
        if tmp_fd is not None:
            with contextlib.suppress(OSError):
                os.close(tmp_fd)
        if tmp_name is not None:
            with contextlib.suppress(OSError):
                os.rmdir(tmp_name, dir_fd=parent_fd)
        # 予約直後で他プロセスの介入が無い限り必ず空のはずだが、万一に備えて
        # 失敗は無視する（cleanup_reserved_out_dir と同じく安全側の残置）。
        with contextlib.suppress(OSError):
            os.rmdir(name, dir_fd=parent_fd)
        if isinstance(e, WorkerError):
            raise
        raise _invalid(f"failed to prepare out_dir: {type(e).__name__}") from e

    return OutDirReservation(
        entry=entry, tmp_name=tmp_name, tmp_fd=tmp_fd, tmp_id=tmp_id, reserved_id=reserved_id
    )


def finalize_out_dir(reservation: OutDirReservation) -> None:
    """予約済みの `out_dir`（空ディレクトリ）を一時ディレクトリの内容で確定させる。

    確定前に 2 つの実体を確認する（P0-2: `rename` は名前しか受け取らないため、
    fd を握っているだけでは名前の差し替えを防げない。直前に必ず名前と fd の
    実体を突き合わせる）:
    1. `tmp_name` が今も自分の `tmp_fd` と同じ実体を指しているか
       （`fstat(tmp_fd)` と `stat(tmp_name, follow_symlinks=False)` の
       `(st_dev, st_ino)` が一致するか）。
    2. `out_dir`（`name`）が予約時点と同じ実体（`reserved_id`）を指しているか
       （シンボリックリンクや別物にすり替わっていないか）。

    いずれか一致しなければ、実体が何であれ触れずに `output_conflict`
    （exit 64）とする。両方一致すれば `os.rename`（`dir_fd` 相対）で置き換える:
    予約後に何者かが `out_dir` の中へファイルを書き込んでいた場合、`rename` は
    ディレクトリを空でない置き換え先へは適用できない（`ENOTEMPTY`）ため失敗する。
    この場合もその内容を削除せず残したまま `output_conflict` とする
    （利用者側の入力・実行環境に起因しうる競合であり、本ワーカーの内部バグでは
    ないため `runtime_error`〔exit 70〕ではなく `output_conflict`〔exit 64〕として
    扱う）。

    **`os.rename` の成功後にも、公開した実体が検証した実体と同一かを再確認する**
    （P0）: `stat` による事前確認から `os.rename` の呼び出しまでの間にも、同じ
    親ディレクトリへ書き込める別プロセスが `tmp_name` を（別ディレクトリへ）
    差し替える余地が理論上残る（`prepare_out_dir` の親ディレクトリ権限検査は
    この余地を大きく減らすが、実行時点の TOCTOU そのものを消しはしない）。
    `rename` は inode を保つため、成功直後に `os.stat(name, ...)` した実体の
    `(st_dev, st_ino)` が `reservation.tmp_id`（`fstat(tmp_fd)` で得た、保持中の
    fd が指す実体の識別子）と一致することを確認する。不一致・stat 失敗の場合は
    `out_dir`（`name`）に公開された実体が何であれ一切削除・rename し戻さず
    （検証していない実体を消す・移動すること自体が別の TOCTOU になるため）、
    自分の一時ディレクトリの後始末（`_release_tmp`。fd 経由で本物の中身だけを
    消す）だけを行ってから `output_conflict` とし、成功を報告しない
    （同一ユーザー内の残余競合であっても、検証していない実体の公開を成功
    扱いにしない。fail-closed）。
    """
    entry = reservation.entry
    parent_fd = entry.parent_fd
    name = entry.name

    try:
        tmp_fd_st = os.fstat(reservation.tmp_fd)
        tmp_name_st = os.stat(reservation.tmp_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as e:
        _release_tmp(reservation)
        raise WorkerError(
            "output_conflict",
            f"tmp entry vanished before finalize: {type(e).__name__}",
            ExitCode.INVALID_INPUT,
        ) from e
    if (tmp_fd_st.st_dev, tmp_fd_st.st_ino) != (tmp_name_st.st_dev, tmp_name_st.st_ino):
        _release_tmp(reservation)
        raise WorkerError(
            "output_conflict", "tmp entry was replaced before finalize", ExitCode.INVALID_INPUT
        )

    try:
        st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as e:
        _release_tmp(reservation)
        raise WorkerError(
            "output_conflict",
            f"reserved out_dir vanished before finalize: {type(e).__name__}",
            ExitCode.INVALID_INPUT,
        ) from e
    current_id = (st.st_dev, st.st_ino)
    if not stat.S_ISDIR(st.st_mode) or current_id != reservation.reserved_id:
        _release_tmp(reservation)
        raise WorkerError(
            "output_conflict",
            "out_dir was replaced by a different entry before finalize",
            ExitCode.INVALID_INPUT,
        )

    try:
        os.rename(reservation.tmp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    except OSError as e:
        # ENOTEMPTY（予約後に何者かが書き込んだ）等。out_dir の中身は触らない。
        _release_tmp(reservation)
        raise WorkerError(
            "output_conflict",
            f"failed to finalize out_dir: {type(e).__name__}",
            ExitCode.INVALID_INPUT,
        ) from e

    # P0: rename 成功後、公開された実体が検証済みの tmp_fd と同一かを再確認する
    # （本関数のドキュメントコメント参照。検証していない実体を成功扱いにしない）。
    try:
        published_st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as e:
        _release_tmp(reservation)
        raise WorkerError(
            "output_conflict",
            f"out_dir not stat-able after finalize: {type(e).__name__}",
            ExitCode.INVALID_INPUT,
        ) from e
    if (published_st.st_dev, published_st.st_ino) != reservation.tmp_id:
        # 公開された実体（name）には触れない（検証していないものを削除・
        # rename し戻すこと自体が別の TOCTOU になる）。自分の一時ディレクトリの
        # 後始末だけ行う（rename 後なので通常は ENOENT で何もしないはず）。
        _release_tmp(reservation)
        raise WorkerError(
            "output_conflict",
            "published out_dir does not match the verified entry after finalize",
            ExitCode.INVALID_INPUT,
        )

    reservation.close_tmp_fd()  # 成功: rename 後はもう不要


def cleanup_reservation(reservation: OutDirReservation) -> None:
    """学習・書き出しの失敗時（ワーカーの異常終了・強制終了を含む）に、確保済みの
    一時ディレクトリと予約済み `out_dir` の両方を解放する
    （`supervisor.py::run_supervised_train` の失敗時クリーンアップから呼ぶ）。
    """
    _release_tmp(reservation)
    cleanup_reserved_out_dir(reservation.entry, reservation.reserved_id)


def cleanup_reserved_out_dir(entry: guard.ConfinedEntry, reserved_id: ReservedId) -> None:
    """予約済みの `out_dir` を解放する。

    まだ自分が予約した実体（`st_dev`・`st_ino` が一致）であり、かつ空である
    場合にのみ `os.rmdir` する。他プロセスが何か書き込んでいる・別物へ
    すり替わっている場合は何もしない（残置は Rust 側ジョブ管理〔REQ-34〕が
    次回実行前に判断する。予約が空のまま残ることは「成果物は公開されていない」
    ことの証拠であり、安全側の残置である）。

    stat と rmdir の間には小さな間隙が残るが、`rmdir` は空ディレクトリしか
    削除できないため、この間隙で起こりうる最悪の事態は「空ディレクトリの
    取り違え」であり、データ（ファイル）を失うことはない。
    """
    parent_fd = entry.parent_fd
    name = entry.name
    try:
        st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return
    if not stat.S_ISDIR(st.st_mode) or (st.st_dev, st.st_ino) != reserved_id:
        return
    with contextlib.suppress(OSError):
        os.rmdir(name, dir_fd=parent_fd)  # 空でなければ ENOTEMPTY → 残置する


def _release_tmp(reservation: OutDirReservation) -> None:
    """作業用一時ディレクトリ（中身を含む）を解放する。`out_dir` 側には触れない。

    P0-2: 中身の削除は保持している `tmp_fd`（名前ではなく実体に束縛された fd）
    経由でのみ行い、名前を再解決しない（`_cleanup_tmp_contents_via_fd`）。
    `rmdir` の直前にだけ `tmp_name` が今も同じ実体（`tmp_id`）を指しているかを
    確認する。この stat→rmdir の間隙は残るが、`cleanup_reserved_out_dir` と
    同様に `rmdir` は空ディレクトリしか削除できないため、データを失うことはない。
    """
    _cleanup_tmp_contents_via_fd(reservation.tmp_fd)
    reservation.close_tmp_fd()
    parent_fd = reservation.entry.parent_fd
    try:
        st = os.stat(reservation.tmp_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return
    if (st.st_dev, st.st_ino) != reservation.tmp_id:
        return
    with contextlib.suppress(OSError):
        os.rmdir(reservation.tmp_name, dir_fd=parent_fd)


def _cleanup_tmp_contents_via_fd(tmp_fd: int) -> None:
    """既に開いている一時ディレクトリの fd から中身を列挙して削除する。

    P0-2: 名前で再オープンしない（`tmp_fd` を閉じた後に同名で開き直すと、
    その間に別プロセスがその名前を別のディレクトリへ差し替えていた場合、
    無関係なディレクトリの中身を削除してしまう TOCTOU になる）。
    渡された `tmp_fd` はここでは閉じない（呼び出し元が管理する）。
    """
    try:
        with os.scandir(tmp_fd) as it:
            names = [e.name for e in it]
    except OSError:
        return
    for name in names:
        with contextlib.suppress(OSError):
            os.unlink(name, dir_fd=tmp_fd)
