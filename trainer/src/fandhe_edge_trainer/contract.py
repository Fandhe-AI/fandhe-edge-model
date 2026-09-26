"""学習リクエスト（Rust 側 CLI が子プロセスへ渡す JSON）の解析・検証。

schema_version 1 の形（`__main__.py` の `train` サブコマンドが受け取る
`--request <path>` の中身）:

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
  "train_path": "train.jsonl",
  "out_dir": "out"
}
```

学習データ（`train_path` が指す JSONL）は 1 行 1 オブジェクト `{"input": str,
"label": str}`。label は label_order に含まれる必要がある。

本モジュールは validation・test データを一切読まない（学習は train データのみで行う。
評価・選定・作り直し判定は評価器（TASK-24.1）・学習ワーカーの選定処理（別 TASK）の
責務であり、ここでは扱わない）。

経路の閉じ込め（REQ-39 のガード層）は Rust 側 CLI が担う設計とし、本ワーカーは
Rust から渡された `train_path`・`out_dir` をそのまま信頼する（子プロセスの
呼び出し元が経路を検証済みという前提。本ワーカー単体で `../` 等を拒否する
経路検証は行わない。将来この前提が変わる場合は呼び出し元契約の見直しとして
報告する）。

**ジョブの再開・異常終了時の後片付けは本モジュールの責務ではない**（REQ-34）。
`prepare_out_dir`/`finalize_out_dir` は「同一プロセス内で正常終了 or 例外終了する」
場合の半端な書き込み防止だけを担う。本ワーカーが SIGKILL 等で強制終了した場合、
`<out_dir>.tmp-*` が残置されうるが、その回収（次回実行前の掃除・再試行判断）は
ジョブ管理を担う Rust 側（TASK-34.x）の責務とする。
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from .encoding import normalize_input
from .errors import WorkerError
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
    "train_path",
    "out_dir",
}


@dataclass(frozen=True)
class TrainExample:
    """学習データ 1 件（正解ラベル付き）。"""

    input: str
    label: str


@dataclass(frozen=True)
class TrainRequest:
    """検証済みの学習リクエスト。"""

    kind: str
    kind_version: int
    config: dict[str, Any]
    label_order: list[str]
    max_bytes: int
    seed: int
    device: str
    train_path: Path
    out_dir: Path


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
    """ファイルを開いたうえで、開いた fd に対して stat を取り「通常ファイルであること」
    を確認する（open → fstat の順にすることで、path に対する stat → open の間に
    別ファイルへ差し替えられる TOCTOU を避ける。REQ-39 ガード層の「形式の許可制」を
    学習ワーカー側でも最小限適用する）。

    `O_NONBLOCK` を付けて開く: 読み取り用に FIFO を通常どおり（ブロッキングで）開くと、
    書き込み側が現れるまで `open()` 自体が無期限にブロックする（`無制限待ちを作らない`。
    REQ-39）。`O_NONBLOCK` を付けると FIFO の open は書き込み側の有無によらず即座に
    返るため、その直後の `fstat`/`S_ISREG` 判定で確実に拒否できる。通常ファイルの
    読み取りには `O_NONBLOCK` は影響しない（意味を持つのは FIFO・一部のデバイス・
    ソケットのみ）。
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


def load_request(path: Path) -> TrainRequest:
    """学習リクエスト JSON ファイルを読み、検証したうえで `TrainRequest` を返す。"""
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
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _invalid(f"request file not readable as utf-8: {type(e).__name__}") from e
    try:
        parsed = _json_loads_strict(text)
    except json.JSONDecodeError as e:
        raise _invalid(f"request is not valid JSON: {e.msg} at line {e.lineno}") from e
    except ValueError as e:
        raise _invalid(f"request JSON rejected: {type(e).__name__}") from e
    return _validate_request(parsed)


def _validate_request(raw: Any) -> TrainRequest:
    if not isinstance(raw, dict):
        raise _invalid("request must be a JSON object")
    unknown = set(raw) - _REQUEST_FIELDS
    if unknown:
        raise _invalid(f"request has unknown fields: {sorted(unknown)}")

    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise _invalid(
            f"unsupported schema_version: {schema_version!r} (expected {SCHEMA_VERSION})"
        )

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

    train_path_raw = raw.get("train_path")
    if not isinstance(train_path_raw, str) or not train_path_raw:
        raise _invalid("train_path must be a non-empty string")

    out_dir_raw = raw.get("out_dir")
    if not isinstance(out_dir_raw, str) or not out_dir_raw:
        raise _invalid("out_dir must be a non-empty string")

    return TrainRequest(
        kind=kind,
        kind_version=kind_version,
        config=config,
        label_order=list(label_order),
        max_bytes=max_bytes,
        seed=seed,
        device=device,
        train_path=Path(train_path_raw),
        out_dir=Path(out_dir_raw),
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


def load_train_examples(train_path: Path, label_order: list[str]) -> list[TrainExample]:
    """学習データ（JSONL）を読み、検証したうえで `TrainExample` の一覧を返す。

    データ本文（`input`・`label` の値）はエラーメッセージへ一切含めない
    （個人情報・機密情報を含みうる。security.md）。行番号・件数だけを報告する。

    資源上限（REQ-39・limits.py）を 3 段で適用する: (1) 1 行あたりのバイト数
    （JSON パース・NFKC 正規化の前に検査。巨大な 1 行によるメモリ確保を防ぐ）、
    (2) 累積読み取りバイト数（ファイル全体のサイズを stat だけに頼らず、実際に
    読んだ量で判定する）、(3) 検証を通過した examples の件数。
    """
    label_set = set(label_order)
    examples: list[TrainExample] = []
    total_bytes = 0
    lineno = 0

    f, _st = _open_regular_file(train_path, _invalid)
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
    finally:
        f.close()

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


def prepare_out_dir(out_dir: Path) -> Path:
    """`out_dir` が未作成であることを確認し、書き込み用の一時ディレクトリを作って返す。

    半端な書き込みを避けるため、実際の出力は一時ディレクトリへ行い、成功時に
    `os.replace` で `out_dir` へ確定させる（ジョブの再開・チェックポイント機構は
    Rust 側 REQ-34 の責務であり、本関数はその前提を壊さないための最小限の対処。
    本関数のモジュール docstring も参照）。
    """
    if out_dir.exists():
        raise _invalid("out_dir already exists")
    parent = out_dir.parent
    if not parent.is_dir():
        raise _invalid("out_dir's parent directory does not exist")
    tmp = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}.tmp-", dir=parent))
    return tmp


def finalize_out_dir(tmp_dir: Path, out_dir: Path) -> None:
    """一時ディレクトリを `out_dir` へ確定させる。失敗時は一時ディレクトリを削除する。

    `os.replace` の失敗（例: `out_dir` が学習中に別プロセスから作られた）は
    利用者側の入力・実行環境に起因しうる競合であり、本ワーカーの内部バグではない
    ため `runtime_error`（exit 70）ではなく `invalid_request`（exit 64）として
    扱う（`prepare_out_dir` が「未作成であること」を確認した後の TOCTOU なので、
    完全には防げないレースだが、検出はできる）。
    """
    try:
        os.replace(tmp_dir, out_dir)
    except OSError as e:
        _cleanup_tmp_dir(tmp_dir)
        raise WorkerError(
            "output_conflict",
            f"failed to finalize out_dir (possibly created concurrently): {type(e).__name__}",
            ExitCode.INVALID_INPUT,
        ) from e


def _cleanup_tmp_dir(tmp_dir: Path) -> None:
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)
