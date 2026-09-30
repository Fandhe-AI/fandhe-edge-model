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

`root` は呼び出し元が許可する作業ルート（絶対パス。`..` 構成要素は拒否、
`.`・末尾 `/`・連続 `/` は受理。REQ-39・#256）。`train_path`・`out_dir` は
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

`validation_inputs` は任意項目（学習ジョブ内での採点用 validation 入力。
`[{"id": str, "input": str}, ...]`。**正解ラベルは受け取らない**〔REQ-27。要素に
他のキーがあれば拒否する〕。検証は `_validate_validation_inputs`。あれば学習直後に
同じプロセス内で予測し、結果の `validation_predictions` として返す。issue #84
PR #238）。リクエスト全体が `MAX_REQUEST_BYTES` で縛られるため、運べる量の実効上限は
そちらで決まる。

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
持ち続けられる。強制終了は `_worker` だけに afflict する。キャンセル
（REQ-34・TASK-34.1-2）は Rust 側が stdin を閉じて伝え、スーパーバイザーが
保持中の fd で予約を解放する（協調キャンセル。`supervisor.py` モジュール
docstring 参照）。スーパーバイザー自身が SIGKILL された場合（協調の猶予を
超えたときの Rust 側のフォールバック等）は後始末が走らず、空の予約が残り
うる。これは安全側の残置で、Rust 側は削除せず読み取り専用で検査して報告
するだけである（やり直しの案内は Rust 側 `restart` が行い、自動では
掃除しない。TASK-34.3）。予約済み `out_dir` が
空のまま残ることは「成果物は公開されていない」ことの証拠になる。

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
import ctypes
import errno
import json
import os
import secrets
import stat
import struct
import sys
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
    MAX_RESULT_BYTES,
    MAX_RESULT_BYTES_WITH_VALIDATION,
    MAX_SEED,
    MAX_TRAIN_DATA_BYTES,
    MAX_TRAIN_EXAMPLES,
    MAX_TRAIN_LINE_BYTES,
    MAX_TRAIN_RSS_BYTES,
    MAX_TRAIN_WALL_SECONDS,
    MAX_VALIDATION_ID_BYTES,
    MAX_VALIDATION_INPUT_BYTES,
    MAX_VALIDATION_INPUT_TOTAL_BYTES,
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
    "validation_inputs",
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
    #: 学習ジョブ内での採点用 validation 入力 `(id, input)` の列（任意。無ければ
    #: `None`）。**正解ラベルは持たない**（REQ-27）。`validate_request` が
    #: `_validate_validation_inputs` で検証済み。
    validation_inputs: tuple[tuple[str, str], ...] | None = None
    #: このリクエストの結果 JSON の標準出力上限（bytes。`validation_inputs` が無ければ
    #: `MAX_RESULT_BYTES`。あれば `validation_result_bytes_bound`）。
    max_result_bytes: int = MAX_RESULT_BYTES

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

    validation_inputs = _validate_validation_inputs(
        raw.get("validation_inputs"), "validation_inputs" in raw
    )
    max_result_bytes = MAX_RESULT_BYTES
    if validation_inputs is not None:
        # 結果の上限をリクエストごとに正確に計算し、天井を超えるなら学習を始める前に
        # 拒否する（正常な結果が上限で弾かれる状況を作らない。P1 指摘対応。
        # `crates/train/src/request.rs::with_validation_inputs` と同じ）。
        max_result_bytes = validation_result_bytes_bound(
            label_order, [rid for rid, _text in validation_inputs]
        )
        if max_result_bytes > MAX_RESULT_BYTES_WITH_VALIDATION:
            raise _limit(
                "result for validation_inputs could exceed "
                f"{MAX_RESULT_BYTES_WITH_VALIDATION} bytes"
            )

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
        validation_inputs=validation_inputs,
        max_result_bytes=max_result_bytes,
    )


#: 予測 1 件の JSON の固定部分の最大バイト数（Rust 側 `VALIDATION_PREDICTION_FIXED_BYTES`
#: と同じ。`{"id":` 6・引用符 2・`,"status":` 10・`"abstain"` 9・`,"predicted_label":` 19・
#: 引用符 2・`}` 1・区切り `,` 1）。
_VALIDATION_PREDICTION_FIXED_BYTES = 50

#: 予測列のキー・括弧の余裕（Rust 側 `VALIDATION_PREDICTIONS_ARRAY_SLACK_BYTES`）。
_VALIDATION_PREDICTIONS_ARRAY_SLACK_BYTES = 64


def _json_escaped_len_bound(text: str) -> int:
    """JSON 文字列（引用符を除く）のエスケープ後の最大バイト長。制御文字は
    `\\uXXXX` の 6 バイトとする最悪値、`"`・`\\` は 2 バイト、それ以外は UTF-8 の
    バイト長（`json.dumps(ensure_ascii=False)` は非 ASCII をエスケープしない）。
    Rust 側 `json_escaped_len_bound` と同じ規則。
    """
    total = 0
    for ch in text:
        code = ord(ch)
        if ch in '"\\':
            total += 2
        elif code < 0x20:
            total += 6
        else:
            total += len(ch.encode("utf-8"))
    return total


def validation_result_bytes_bound(label_order: list[str], ids: list[str]) -> int:
    """`validation_inputs` 付きリクエストの結果 JSON の最大バイト数（標準出力の上限）。

    `MAX_RESULT_BYTES` ＋ 配列の余裕 ＋ Σ（予測 1 件の固定部分 ＋ その `id` の
    エスケープ後の最大長 ＋ ラベル集合のうちエスケープ後に最長のラベルの長さ）。
    Rust 側 `validation_result_bytes_bound` と同じ式（共有 fixture
    `result_cap_cases.json` で一致を確認する。issue #84 PR #238 レビュー）。
    """
    longest_label = max((_json_escaped_len_bound(label) for label in label_order), default=0)
    total = MAX_RESULT_BYTES + _VALIDATION_PREDICTIONS_ARRAY_SLACK_BYTES
    for rid in ids:
        total += _VALIDATION_PREDICTION_FIXED_BYTES + _json_escaped_len_bound(rid) + longest_label
    return total


def _validate_validation_inputs(value: Any, present: bool) -> tuple[tuple[str, str], ...] | None:
    """`validation_inputs`（任意。REQ-18・REQ-27・REQ-39）を検証する。

    キーが無ければ `None`。あれば 1 件以上の `{"id": str, "input": str}` の
    リストで、要素は他のキーを持てない（正解ラベルを紛れ込ませない。REQ-27）。
    `id` は非空・UTF-8 で `MAX_VALIDATION_ID_BYTES` 以下・重複なし、`input` は
    UTF-8 で `MAX_VALIDATION_INPUT_BYTES` 以下、`id`＋`input` の合計は
    `MAX_VALIDATION_INPUT_TOTAL_BYTES` 以下（`crates/train/src/request.rs` の
    `validate_validation_inputs` と同じ規則。リクエスト全体は既に
    `MAX_REQUEST_BYTES` で縛られているため実効上限はそちら）。エラーメッセージへ
    `id`・`input` の中身を含めない（security.md）。
    """
    if not present:
        return None
    if not isinstance(value, list) or not value:
        raise _invalid("validation_inputs must be a non-empty list")
    seen: set[str] = set()
    total = 0
    out: list[tuple[str, str]] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"id", "input"}:
            raise _invalid(f"validation_inputs[{i}] must be an object with exactly id and input")
        rid = item["id"]
        text = item["input"]
        if not isinstance(rid, str) or not isinstance(text, str):
            raise _invalid(f"validation_inputs[{i}] id and input must be strings")
        # UTF-8 バイト数は文字数以上のため、`encode` の前に文字数で足切りして
        # 巨大な値の再確保を避ける。孤立サロゲートは `encode` が失敗する。
        if not rid or len(rid) > MAX_VALIDATION_ID_BYTES or len(text) > MAX_VALIDATION_INPUT_BYTES:
            raise _invalid(f"validation_inputs[{i}] id is empty or a size limit is exceeded")
        try:
            id_len = len(rid.encode("utf-8"))
            input_len = len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise _invalid(f"validation_inputs[{i}] is not valid utf-8") from exc
        if id_len > MAX_VALIDATION_ID_BYTES or input_len > MAX_VALIDATION_INPUT_BYTES:
            raise _invalid(f"validation_inputs[{i}] exceeds a size limit")
        total += id_len + input_len
        if total > MAX_VALIDATION_INPUT_TOTAL_BYTES:
            raise _invalid("validation_inputs total bytes exceed the limit")
        if rid in seen:
            raise _invalid(f"validation_inputs[{i}] id is a duplicate")
        seen.add(rid)
        out.append((rid, text))
    return tuple(out)


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


# --------------------------------------------------------------------------
# macOS 拡張 ACL 検査（Codex レビュー再指摘・オーナー承認 2026-09-27）。
# `out_dir` の親ディレクトリへの書き込みが「所有者のみ」に限られていることを、
# POSIX パーミッションビットに加えて macOS の拡張 ACL（`chmod +a` 等で
# `group:everyone allow ...` のように付与される追加の許可）についても検査する。
# ACL はパーミッションビットに現れないため、uid/mode 検査だけでは検出できない。
#
# FFI（ctypes）についての注記（Rust の `// SAFETY:` に相当。オーナー承認
# 2026-09-27）:
# - `acl_get_fd_np`・`acl_get_entry`・`acl_get_tag_type`・`acl_free` は
#   macOS SDK の `<sys/acl.h>` で宣言されたシステム API。以下の
#   `restype`/`argtypes` は `sys/acl.h` の宣言に一致させてある（`acl_t`・
#   `acl_entry_t` は不透明なポインタ型のため `c_void_p` を割り当てる。
#   `acl_tag_t`・`acl_type_t` は C の enum＝`int` サイズ）。
# - 維持すべき不変条件: `acl_get_fd_np` が返すポインタ（非 NULL の場合）は
#   `acl_free` を呼ぶまで有効な参照として扱い、成功・失敗いずれの経路でも
#   `finally` で必ず 1 回だけ解放する（リーク・二重解放のいずれもしない）。
# - `ctypes` は stdlib のため、本検査を追加しても学習ワーカー側の依存方針
#   （dependency-policy.md）・`supervisor.py` の mlx・onnx・numpy 非依存
#   （P0-2）を崩さない。
# --------------------------------------------------------------------------

_ACL_TYPE_EXTENDED = 0x00000100
_ACL_FIRST_ENTRY = 0
_ACL_NEXT_ENTRY = -1
_ACL_EXTENDED_ALLOW = 1
_ACL_EXTENDED_DENY = 2


def _load_acl_functions() -> ctypes.CDLL | None:
    """ACL API（libSystem 経由。現在のプロセスに既にリンクされている）の
    関数ポインタを解決し、`restype`/`argtypes` を確定させる。失敗すれば
    `None`（呼び出し側が fail-closed に扱う）。
    """
    try:
        lib = ctypes.CDLL(None, use_errno=True)
        lib.acl_get_fd_np.restype = ctypes.c_void_p
        lib.acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.acl_get_entry.restype = ctypes.c_int
        lib.acl_get_entry.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.acl_get_tag_type.restype = ctypes.c_int
        lib.acl_get_tag_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.acl_free.restype = ctypes.c_int
        lib.acl_free.argtypes = [ctypes.c_void_p]
    except (OSError, AttributeError):
        return None
    return lib


# --------------------------------------------------------------------------
# Linux POSIX ACL 検査（build/trainer-linux-uv。main 承認済み設計）。
# macOS の拡張 ACL 検査と同じ検査意図を、Linux カーネルの POSIX ACL
# （`setfacl` 等で付与する named user/group エントリ）に対して行う。
# POSIX ACL は 2 種類の拡張属性で表現される:
#   - `system.posix_acl_access`: そのディレクトリ自体への実効 ACL
#   - `system.posix_acl_default`: そのディレクトリ配下に新規作成される
#     子（ファイル・ディレクトリ）へ継承される既定 ACL
# `prepare_out_dir`/`finalize_out_dir` は親ディレクトリ配下に子
# （予約した一時ディレクトリ・最終的な `out_dir`）を作成するため、
# access だけでなく default ACL も検査しないと、default ACL 経由で
# 子が他ユーザーに書き込み可能になる余地を見逃す。
#
# xattr の値は Linux カーネルの `posix_acl_xattr` 形式（標準ライブラリの
# `os.getxattr`/`os.setxattr` で読み書きする。追加依存なし）:
#   - 先頭 4 バイト: little-endian u32 の version（2 のみ受け付ける）
#   - 続く 8 バイト単位のエントリの並び: le16 e_tag・le16 e_perm・le32 e_id
# タグ: ACL_USER_OBJ=0x01・ACL_USER=0x02・ACL_GROUP_OBJ=0x04・
# ACL_GROUP=0x08・ACL_MASK=0x10・ACL_OTHER=0x20。perm の WRITE=0x02。
# USER_OBJ・GROUP_OBJ・OTHER・MASK は既存の uid/mode 検査
# （`_assert_parent_dir_exclusive`）の範囲に相当するため、本検査では
# 名前付きエントリ（ACL_USER・ACL_GROUP）の書き込み許可のみを見る。
# --------------------------------------------------------------------------

_POSIX_ACL_XATTR_NAMES = ("system.posix_acl_access", "system.posix_acl_default")
# NFSv4 ACL の xattr 名。`_reject_posix_acl_write_grant` は内容を評価できないため、
# 存在すれば拒否する。
_NFS4_ACL_XATTR_NAME = "system.nfs4_acl"
_POSIX_ACL_XATTR_VERSION = 2
_POSIX_ACL_XATTR_HEADER = struct.Struct("<I")
_POSIX_ACL_XATTR_ENTRY = struct.Struct("<HHI")

_ACL_USER_OBJ = 0x01
_ACL_USER = 0x02
_ACL_GROUP_OBJ = 0x04
_ACL_GROUP = 0x08
_ACL_MASK = 0x10
_ACL_OTHER = 0x20
_ACL_NAMED_TAGS = frozenset({_ACL_USER, _ACL_GROUP})
_ACL_KNOWN_TAGS = frozenset(
    {_ACL_USER_OBJ, _ACL_USER, _ACL_GROUP_OBJ, _ACL_GROUP, _ACL_MASK, _ACL_OTHER}
)
_ACL_PERM_WRITE = 0x02


def _posix_acl_has_named_write_entry(data: bytes) -> bool:
    """`posix_acl_xattr` 形式のバイト列を解析し、名前付きエントリ
    （ACL_USER・ACL_GROUP）に書き込み許可（WRITE ビット）が 1 つでも
    あれば `True` を返す（純粋関数。fd・OS に依存しない）。

    長さ不足（ヘッダ未満・エントリの端数）・未対応 version・未知のタグは
    いずれも `WorkerError`（`output_conflict`）で fail-closed とする
    （REQ-39 ガード層。壊れた／想定外の ACL 表現を安全側と誤読しない）。
    添字アクセスの範囲外を避けるため、`struct.unpack_from` の前に長さを
    検証する。
    """
    if len(data) < _POSIX_ACL_XATTR_HEADER.size:
        raise WorkerError(
            "output_conflict", "malformed POSIX ACL xattr: too short", ExitCode.INVALID_INPUT
        )
    (version,) = _POSIX_ACL_XATTR_HEADER.unpack_from(data, 0)
    if version != _POSIX_ACL_XATTR_VERSION:
        raise WorkerError(
            "output_conflict",
            f"unsupported POSIX ACL xattr version: {version}",
            ExitCode.INVALID_INPUT,
        )
    body = data[_POSIX_ACL_XATTR_HEADER.size :]
    if len(body) % _POSIX_ACL_XATTR_ENTRY.size != 0:
        raise WorkerError(
            "output_conflict",
            "malformed POSIX ACL xattr: truncated entry",
            ExitCode.INVALID_INPUT,
        )
    has_write = False
    for offset in range(0, len(body), _POSIX_ACL_XATTR_ENTRY.size):
        tag, perm, _entry_id = _POSIX_ACL_XATTR_ENTRY.unpack_from(body, offset)
        if tag not in _ACL_KNOWN_TAGS:
            raise WorkerError(
                "output_conflict",
                f"unexpected POSIX ACL tag type: {tag}",
                ExitCode.INVALID_INPUT,
            )
        if tag in _ACL_NAMED_TAGS and (perm & _ACL_PERM_WRITE) != 0:
            has_write = True
    return has_write


def _reject_posix_acl_write_grant(fd: int) -> None:
    """`fd` が指すディレクトリの POSIX ACL（access・default の両方）に
    名前付きエントリの書き込み許可があれば `output_conflict` で拒否する
    （Linux 版。`_reject_extended_acl_allow` から呼ばれる）。

    属性が存在しない（`ENODATA`）・ファイルシステムが ACL 非対応
    （`ENOTSUP`/`EOPNOTSUPP`。ACL が存在し得ないため合格扱いにできる。
    ACL を無効にマウントした tmpfs・xattr 非対応の overlayfs upper 層など）は
    合格として扱う。それ以外の `OSError` は fail-closed とする
    （REQ-39 ガード層）。

    NFSv4 ACL（NFS マウント等で `system.nfs4_acl` に置かれる）は POSIX ACL と
    別体系で、`system.posix_acl_*` には現れない。本関数はその内容を評価
    できないため、属性が存在するだけで fail-closed（拒否）とする。

    判定は名前付きエントリ自身の perm ビットで行い、ACL_MASK による実効権限の
    縮小（`perm & mask`）は計算しない。MASK は許可を縮める方向にしか働かない
    ため、この省略は過剰拒否の方向にだけ倒れる（迂回にはならない）。

    本検査の範囲外: root と同様に、`CAP_DAC_OVERRIDE`・`CAP_FOWNER` を持つ
    プロセス（fakeroot・一部のコンテナ環境）はパーミッション・ACL に
    関係なく書き込めるため、本検査では防げない。
    """
    try:
        os.getxattr(fd, _NFS4_ACL_XATTR_NAME)
    except OSError as e:
        if e.errno not in (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise WorkerError(
                "output_conflict",
                f"failed to read {_NFS4_ACL_XATTR_NAME} (errno={e.errno})",
                ExitCode.INVALID_INPUT,
            ) from e
    else:
        raise WorkerError(
            "output_conflict",
            "out_dir parent has an NFSv4 ACL, which this check cannot evaluate",
            ExitCode.INVALID_INPUT,
        )
    for name in _POSIX_ACL_XATTR_NAMES:
        try:
            data = os.getxattr(fd, name)
        except OSError as e:
            if e.errno in (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP):
                continue
            raise WorkerError(
                "output_conflict",
                f"failed to read {name} (errno={e.errno})",
                ExitCode.INVALID_INPUT,
            ) from e
        if _posix_acl_has_named_write_entry(data):
            raise WorkerError(
                "output_conflict",
                "out_dir parent has an ACL entry that allows additional write access",
                ExitCode.INVALID_INPUT,
            )


def _reject_extended_acl_allow(fd: int) -> None:
    """`fd` が指すディレクトリの拡張 ACL に、追加の書き込み許可を与える
    エントリが 1 つでもあれば `output_conflict` で拒否する。

    macOS では `ACL_TYPE_EXTENDED` の `ACL_EXTENDED_ALLOW` エントリ
    （`chmod +a` で付与する `group:everyone allow ...` 等）を、Linux では
    POSIX ACL の名前付きエントリ（`_reject_posix_acl_write_grant` 参照）を
    検査する。macOS の拡張 ACL・Linux の POSIX ACL は、いずれも POSIX の
    パーミッションビットに現れないため、uid/mode 検査だけでは検出
    できない（Codex レビュー再指摘。オーナー承認 2026-09-27）。macOS の
    deny エントリのみ（例: Finder が既定で `~/Documents` 等に付与する
    `group:everyone deny delete` 等）は追加の書き込み許可を与えないため
    合格とする。

    macOS・Linux 以外の OS・ACL API のシンボル解決の失敗・想定外の
    戻り値やタグは、いずれも fail-closed（`output_conflict`）とする
    （REQ-39 ガード層。「動くこと」より「検査の正しさ」を優先する）。
    """
    if sys.platform == "linux":
        _reject_posix_acl_write_grant(fd)
        return
    if sys.platform != "darwin":
        raise WorkerError(
            "output_conflict", "ACL check requires macOS or Linux", ExitCode.INVALID_INPUT
        )
    lib = _load_acl_functions()
    if lib is None:
        raise WorkerError(
            "output_conflict", "failed to resolve the ACL API", ExitCode.INVALID_INPUT
        )

    ctypes.set_errno(0)
    acl = lib.acl_get_fd_np(fd, _ACL_TYPE_EXTENDED)
    if not acl:
        err = ctypes.get_errno()
        if err == errno.ENOENT:
            return  # 拡張 ACL が無い（合格）。
        raise WorkerError(
            "output_conflict",
            f"failed to read the extended ACL (errno={err})",
            ExitCode.INVALID_INPUT,
        )

    try:
        entry = ctypes.c_void_p()
        entry_id = _ACL_FIRST_ENTRY
        while True:
            ctypes.set_errno(0)
            if lib.acl_get_entry(acl, entry_id, ctypes.byref(entry)) != 0:
                # 走査の終端は -1・errno=EINVAL（macOS の acl_get_entry の仕様。
                # 実機で確認済み）。それ以外の失敗は途中のエントリを読み飛ばした
                # 可能性があるため、終端と区別して fail-closed とする。
                err = ctypes.get_errno()
                if err == errno.EINVAL:
                    break
                raise WorkerError(
                    "output_conflict",
                    f"failed to enumerate ACL entries (errno={err})",
                    ExitCode.INVALID_INPUT,
                )
            tag_type = ctypes.c_int()
            if lib.acl_get_tag_type(entry, ctypes.byref(tag_type)) != 0:
                raise WorkerError(
                    "output_conflict", "failed to read an ACL entry tag", ExitCode.INVALID_INPUT
                )
            if tag_type.value == _ACL_EXTENDED_ALLOW:
                raise WorkerError(
                    "output_conflict",
                    "out_dir parent has an ACL entry that allows additional write access",
                    ExitCode.INVALID_INPUT,
                )
            if tag_type.value != _ACL_EXTENDED_DENY:
                raise WorkerError(
                    "output_conflict",
                    f"unexpected ACL entry tag type: {tag_type.value}",
                    ExitCode.INVALID_INPUT,
                )
            entry_id = _ACL_NEXT_ENTRY
    finally:
        lib.acl_free(acl)


def _assert_parent_dir_exclusive(parent_fd: int) -> None:
    """`parent_fd` が指す親ディレクトリへ、実行ユーザー以外が書き込めないことを
    確認する（POSIX の uid/mode 検査 + 拡張 ACL 検査。macOS は拡張 ACL、
    Linux は POSIX ACL を検査する）。`prepare_out_dir`
    （予約前）・`finalize_out_dir`（`os.rename` 直前）の 2 箇所から同じ検査を
    呼ぶ（P0。Codex レビュー再指摘。オーナー承認 2026-09-27）。

    親ディレクトリに書き込めるのは所有者（実行ユーザー）と root のみである
    ことを、呼び出し側が保持し続けている fd を使って予約時と rename 直前の
    両方で確認する。同一 uid の別プロセスはスーパーバイザー自身を ptrace 等で
    直接操作できるため、これはプロセスの信頼境界そのものと同じであり、
    本検査の対象外とする（検出しようとしても意味を持たない）。root は
    POSIX パーミッション・ACL のいずれも無視できるため、本検査で防げるのは
    「実行ユーザーと異なる非 root ユーザーによる書き込み」に限られる。
    `finalize_out_dir` 側の `os.rename` 後の inode 照合は、本検査をすり抜けた
    場合に備えた多層防御として残す（それ自体が公開の正しさを保証する根拠
    ではない）。
    """
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
        # ため拒否する（何も作成しない・確定させない。REQ-39 ガード層）。
        raise WorkerError(
            "output_conflict",
            "out_dir parent must be owned by the current user and not writable by group or others",
            ExitCode.INVALID_INPUT,
        )
    _reject_extended_acl_allow(parent_fd)


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

    **予約前に親ディレクトリが排他的（実行ユーザーのみ書き込み可能）であることを
    確認する**（P0。`os.mkdir`・`os.rename` はいずれも `dir_fd` の指す実体
    そのものへは書き込むが、その親ディレクトリに他ユーザーが書き込める場合、
    `finalize_out_dir` の `stat` から `os.rename` までの間に、同じ親
    ディレクトリ配下で `tmp_name`（予約した一時ディレクトリと同名）を
    差し替えられる余地が残る〔`finalize_out_dir` のドキュメントコメント
    参照〕。`_assert_parent_dir_exclusive`（POSIX の uid/mode 検査 + 拡張 ACL
    検査。関数 docstring 参照）を `os.mkdir` の前に呼ぶ。確認に失敗した
    場合・条件を満たさない場合は、何も作成せず `output_conflict`
    （fail-closed。REQ-39 ガード層）。同じ検査を `finalize_out_dir` の
    `os.rename` 直前でも再度行う（保持中の fd に対して再検査することで、
    予約からその時点までの間に親ディレクトリの状態が変わっていないかを
    確かめる）。
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

    # 予約前の排他性検査（uid/mode + 拡張 ACL）。何も作成する前に行う
    # （`_assert_parent_dir_exclusive` docstring・本関数 docstring 参照）。
    _assert_parent_dir_exclusive(parent_fd)

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

    **`os.rename` の直前に、親ディレクトリの排他性（uid/mode + 拡張 ACL）を
    保持中の `parent_fd` で再検査する**（P0。Codex レビュー再指摘。オーナー
    承認 2026-09-27）: `prepare_out_dir` での予約時点から本関数が呼ばれる
    までの間に、親ディレクトリの状態（所有者・パーミッション・ACL）が
    変わっている可能性がある。`_assert_parent_dir_exclusive` を rename 直前に
    もう一度呼び、失敗すれば作業用一時ディレクトリを解放
    （`_release_tmp`）したうえで、この時点まで空ディレクトリであることを
    直前の 2. で確認済みの予約済み `out_dir` 自体も
    `cleanup_reserved_out_dir` で解放してから `output_conflict` とする
    （まだ何も rename していないので、`out_dir` を安全に片付けられる）。

    **`os.rename` の成功後にも、公開した実体が検証した実体と同一かを再確認する**
    （P0）: 親ディレクトリの排他性を直前に再検査した時点で、`rename` の瞬間に
    `tmp_name` を差し替えられるのは実行ユーザー自身（プロセスの信頼境界と
    同じであり、本モジュールの検査対象外。`_assert_parent_dir_exclusive`
    docstring 参照）と root だけである。post-rename の inode 照合は、この
    前提が何らかの理由（本関数の検査の不備・環境固有の想定外挙動等）で
    破れた場合に備えた多層防御として残す（それ自体が公開の正しさを保証する
    根拠ではない）。`rename` は inode を保つ性質を使って、成功直後に
    `os.stat(name, ...)` した実体の
    `(st_dev, st_ino)` が `reservation.tmp_id`（`fstat(tmp_fd)` で得た、保持中の
    fd が指す実体の識別子）と一致することを確認する。不一致・stat 失敗の場合は
    `out_dir`（`name`）に公開された実体が何であれ一切削除・rename し戻さず
    （検証していない実体を消す・移動すること自体が別の TOCTOU になるため）、
    自分の一時ディレクトリの後始末（`_release_tmp`。fd 経由で本物の中身だけを
    消す）だけを行ってから `output_conflict` とし、成功を報告しない
    （検証していない実体の公開を成功扱いにしない。fail-closed）。
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

    # P0: rename 直前に親ディレクトリの排他性を再検査する（本関数の
    # ドキュメントコメント参照）。この時点までに out_dir が自分の予約と
    # 一致することは確認済み（直前のブロック）なので、失敗時は out_dir 自体を
    # 安全に片付けてよい（`cleanup_reserved_out_dir` は reserved_id 一致・
    # 空の場合にのみ rmdir するため、二重に安全）。`WorkerError` に限らず
    # あらゆる例外（`BaseException`。ACL API 呼び出し・ctypes 境界で
    # 想定外の例外が起きた場合を含む）で予約を残置しない（監査指摘 P2）。
    try:
        _assert_parent_dir_exclusive(parent_fd)
    except BaseException:
        _release_tmp(reservation)
        cleanup_reserved_out_dir(entry, reservation.reserved_id)
        raise

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


def cleanup_reservation(reservation: OutDirReservation) -> bool:
    """学習・書き出しの失敗時（ワーカーの異常終了・強制終了を含む）に、確保済みの
    一時ディレクトリと予約済み `out_dir` の両方を解放する
    （`supervisor.py::run_supervised_train` の失敗時クリーンアップから呼ぶ）。

    両方を解放できたことを確認できた場合のみ `True` を返す。stat・rmdir の失敗や
    非空・すり替わりで残置した場合は `False`（協調キャンセル完了を報告して
    よいかの判断に使う。REQ-34・#145）。
    """
    tmp_released = _release_tmp(reservation)
    out_released = cleanup_reserved_out_dir(reservation.entry, reservation.reserved_id)
    return tmp_released and out_released


def cleanup_reserved_out_dir(entry: guard.ConfinedEntry, reserved_id: ReservedId) -> bool:
    """予約済みの `out_dir` を解放する。

    まだ自分が予約した実体（`st_dev`・`st_ino` が一致）であり、かつ空である
    場合にのみ `os.rmdir` する。他プロセスが何か書き込んでいる・別物へ
    すり替わっている場合は何もしない（残置は次回実行前に判断する〔TASK-34.3〕。
    予約が空のまま残ることは「成果物は公開されていない」ことの証拠であり、
    安全側の残置である）。

    stat と rmdir の間には小さな間隙が残るが、`rmdir` は空ディレクトリしか
    削除できないため、この間隙で起こりうる最悪の事態は「空ディレクトリの
    取り違え」であり、データ（ファイル）を失うことはない。

    解放できた（名前が存在しない、または自分の予約を rmdir できた）場合のみ
    `True`。stat 失敗・すり替わり・非空・rmdir 失敗は `False`（残置）。
    """
    parent_fd = entry.parent_fd
    name = entry.name
    try:
        st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if not stat.S_ISDIR(st.st_mode) or (st.st_dev, st.st_ino) != reserved_id:
        return False
    try:
        os.rmdir(name, dir_fd=parent_fd)  # 空でなければ ENOTEMPTY → 残置する
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _release_tmp(reservation: OutDirReservation) -> bool:
    """作業用一時ディレクトリ（中身を含む）を解放する。`out_dir` 側には触れない。

    P0-2: 中身の削除は保持している `tmp_fd`（名前ではなく実体に束縛された fd）
    経由でのみ行い、名前を再解決しない（`_cleanup_tmp_contents_via_fd`）。
    `rmdir` の直前にだけ `tmp_name` が今も同じ実体（`tmp_id`）を指しているかを
    確認する。この stat→rmdir の間隙は残るが、`cleanup_reserved_out_dir` と
    同様に `rmdir` は空ディレクトリしか削除できないため、データを失うことはない。
    名前が既に無い、または自分の一時ディレクトリを rmdir できた場合のみ `True`。
    """
    _cleanup_tmp_contents_via_fd(reservation.tmp_fd)
    reservation.close_tmp_fd()
    parent_fd = reservation.entry.parent_fd
    try:
        st = os.stat(reservation.tmp_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if (st.st_dev, st.st_ino) != reservation.tmp_id:
        return False
    try:
        os.rmdir(reservation.tmp_name, dir_fd=parent_fd)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


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
