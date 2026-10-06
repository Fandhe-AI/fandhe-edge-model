"""PoC-26 学習・採点 CLI の入出力（REQ-41・TASK-41.1-5・#390）。

役割: 定義ファイル（`options[].id` の宣言順）・学習 / 検証 / 入力の JSONL・adapter・出力ファイル
（`pred.jsonl`・`raw_scores.jsonl`・`adapters.safetensors`・`adapter_config.json`・`run.json`）の
読み書きを担う。入力は上限つき（`limits.py` の定数）で読み、出力は一時ディレクトリへ排他作成
（0600）して全部揃ってから rename で確定する（`OutputDir`。既存の出力は上書きせず、失敗時は
何も残さない）。`pred.jsonl` は `{id,status,predicted_label,scores}` のみ（余分なキーなし）。
`train.py`・`predict.py` から呼ばれる。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from tools.poc26.common import (
    SHA_RE,
    invalid,
    loads,
    read_input,
    sha256,
    too_large,
)

from fandhe_edge_trainer import limits
from fandhe_edge_trainer.kinds.autoregressive import (
    MAX_PREDICTION_ID_BYTES,
)

# 各ファイルの読み込み上限（バイト）。
MAX_DEFINITION_BYTES = limits.MAX_REQUEST_BYTES


MAX_ADAPTER_CONFIG_BYTES = 64 * 1024


MAX_ADAPTER_BYTES = 64 << 20


LORA_KEYS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class Record:
    """1 件の入力（`label` は学習データのみ）。本文・id はログに出さない。"""

    id: str
    input: str
    label: str | None


def check_id(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_PREDICTION_ID_BYTES:
        raise invalid("record id must be a non-empty string within the length limit")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise invalid("record id is not valid unicode") from None
    return value


def load_records(path: Path, *, labels: set[str] | None, what: str) -> tuple[list[Record], str]:
    """`{id,input[,output.intent]}` の JSONL を行サイズ・件数の上限つきで読む。

    `labels` を渡すと `output.intent` を必須とし、集合外のラベルを拒否する（学習データ用）。
    id の重複は拒否する。sha256 は `run.json` の記録用。
    """
    raw = read_input(path, limit=limits.MAX_TRAIN_DATA_BYTES, what=what)
    records: list[Record] = []
    seen: set[str] = set()
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        if len(line) > limits.MAX_TRAIN_LINE_BYTES:
            raise too_large(f"{what} line too large")
        if len(records) >= limits.MAX_TRAIN_EXAMPLES:
            raise too_large(f"{what} has too many records")
        row = loads(line, what)
        if not isinstance(row, dict) or not isinstance(row.get("input"), str):
            raise invalid(f"invalid {what} record")
        rid = check_id(row.get("id"))
        if rid in seen:
            raise invalid(f"duplicate id in {what}")
        seen.add(rid)
        label = None
        if labels is not None:
            out = row.get("output")
            label = out.get("intent") if isinstance(out, dict) else None
            if not isinstance(label, str) or label not in labels:
                raise invalid(f"{what} record has a label outside the definition options")
        records.append(Record(rid, row["input"], label))
    if not records:
        raise invalid(f"{what} has no records")
    return records, sha256(raw)


def load_label_order(path: Path) -> tuple[list[str], str]:
    """定義ファイルの `options[].id` を宣言順で読む（宣言順がタイブレークと scores の順）。"""
    raw = read_input(path, MAX_DEFINITION_BYTES, "definition")
    doc = loads(raw, "definition")
    options = doc.get("options") if isinstance(doc, dict) else None
    if not isinstance(options, list) or not (
        limits.MIN_LABELS <= len(options) <= limits.MAX_LABELS
    ):
        raise invalid("definition options must be a list within the label count limits")
    order: list[str] = []
    for opt in options:
        oid = opt.get("id") if isinstance(opt, dict) else None
        if (
            not isinstance(oid, str)
            or not oid
            or len(oid.encode("utf-8", "replace")) > (limits.MAX_LABEL_BYTES)
        ):
            raise invalid("definition option id must be a non-empty string within the limit")
        order.append(oid)
    if len(set(order)) != len(order):
        raise invalid("definition option ids must be unique")
    for oid in order:
        try:
            oid.encode("utf-8")
        except UnicodeEncodeError:
            raise invalid("definition option id is not valid unicode") from None
    return order, sha256(raw)


def load_weights(data: bytes, what: str) -> dict[str, mx.array]:
    """読み済みのバイト列から safetensors を読み、全配列を実体化する（REQ-39）。

    sha256 を取ったのと同じバイト列から読むので、ハッシュと読み込みの中身は常に一致する。
    ファイルを開き直さない（検査後の差し替えの影響を受けない）。
    """
    try:
        weights = mx.load(io.BytesIO(data), format="safetensors")
        mx.eval(weights)
    except (RuntimeError, ValueError, OSError):
        raise invalid(f"invalid {what}") from None
    if not isinstance(weights, dict):
        raise invalid(f"invalid {what}")
    return weights


def check_out_dir(out_dir: Path) -> None:
    """出力先は「存在しないこと」と親ディレクトリの存在を要求する（既存の出力を上書きしない）。

    親ディレクトリの途中の symlink は辿る（P3: PoC・ローカルの信頼できるパス前提）。
    """
    if out_dir.name in ("", ".", "..") or os.path.lexists(out_dir):
        raise invalid("out-dir must not exist yet (and must name a new directory)")
    if not out_dir.parent.is_dir():
        raise invalid("the parent directory of out-dir must exist")


def _write_new(dir_fd: int, name: str, data: bytes) -> None:
    """ディレクトリ fd を起点に排他作成（O_EXCL|O_NOFOLLOW・0600）で書く。"""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except OSError:
        raise invalid(f"cannot create {name} (exists or not writable)") from None


class OutputDir:
    """出力一式を一時ディレクトリへ書き、全部揃ってから `os.rename` で確定する（REQ-39）。

    - 出力先 `out_dir` は存在しないこと。同じ親に `.<name>.tmp-XXXX`（0700）を作り、各ファイルは
      そのディレクトリ fd 起点で排他作成（0600）する。
    - `commit()` で `out_dir` へ rename する。`with` を例外で抜けたとき（採点中の失敗など）は、
      自分が
      作った一時ディレクトリを `shutil.rmtree`（symlink を辿らない）で消し、半端な出力を残さない
      （同じ `out_dir` でそのまま再実行できる）。プロセスの強制終了（SIGKILL）では一時ディレクトリが
      残りうる（隠し名。手で消す）。
    """

    def __init__(self, out_dir: Path) -> None:
        check_out_dir(out_dir)
        self.out_dir = out_dir
        self.tmp: Path | None = None
        self._fd: int | None = None
        self._committed = False

    def __enter__(self) -> OutputDir:
        try:
            tmp = tempfile.mkdtemp(prefix=f".{self.out_dir.name}.tmp-", dir=self.out_dir.parent)
        except OSError:
            raise invalid("cannot create a temporary output directory") from None
        self.tmp = Path(tmp)
        self._fd = os.open(tmp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        return self

    def write(self, name: str, data: str | bytes) -> None:
        """一時ディレクトリへ 1 ファイルを排他作成する（`str` は UTF-8・LF）。"""
        if self._fd is None:
            raise RuntimeError("OutputDir is not open")
        raw = data.encode("utf-8") if isinstance(data, str) else data
        _write_new(self._fd, name, raw)

    def commit(self) -> None:
        """一時ディレクトリを `out_dir` へ確定する（`out_dir` が既にあれば 64）。"""
        if self.tmp is None:
            raise RuntimeError("OutputDir is not open")
        check_out_dir(self.out_dir)
        try:
            os.rename(self.tmp, self.out_dir)
        except OSError:
            raise invalid("cannot move the output directory into place") from None
        self._committed = True

    def __exit__(self, *_exc: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
        if self.tmp is not None and not self._committed:
            shutil.rmtree(self.tmp, ignore_errors=True)


def write_new_file(path: Path, data: str) -> None:
    """単一ファイルを親ディレクトリ fd 起点で排他作成する（probe 用。親は存在必須）。"""
    if os.path.lexists(path) or not path.parent.is_dir():
        raise invalid("output file must not exist and its parent directory must exist")
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        raise invalid("cannot open the parent directory of the output file") from None
    try:
        _write_new(dir_fd, path.name, data.encode("utf-8"))
    finally:
        os.close(dir_fd)


def adapters_to_bytes(adapters: dict[str, mx.array]) -> bytes:
    """adapter を safetensors のバイト列にする（書く内容と sha256 を同じバイト列にそろえる）。"""
    buf = io.BytesIO()
    mx.save_safetensors(buf, adapters)
    return buf.getvalue()


def json_text(doc: dict[str, Any]) -> str:
    return json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def write_scores(out: OutputDir, pred: list[str], raw: list[str]) -> None:
    """`pred.jsonl`・`raw_scores.jsonl` を出力ディレクトリへ書く。"""
    out.write("pred.jsonl", "".join(x + "\n" for x in pred))
    out.write("raw_scores.jsonl", "".join(x + "\n" for x in raw))


def common_run(
    a: argparse.Namespace, hashes: dict[str, str], sha: str, size: int
) -> dict[str, Any]:
    """`run.json` の共通部分（`hashes` は `Assets.hashes`: 照合済みの各ファイルの sha256）。"""
    return {
        "evidence": a.evidence,
        "command": a.command,
        "args": {  # パスはファイル名だけ（ホームのユーザー名などを残さない）
            k: Path(v).name if isinstance(v, Path) else v for k, v in vars(a).items() if k != "func"
        },
        "mlx_version": mx.__version__,
        "python_version": platform.python_version(),
        "device": a.device,
        "dtype": a.dtype,
        "model_safetensors_bytes": size,
        "model_safetensors_sha256": sha,  # 期待値と照合済み（--model-sha256）
        **hashes,
        "sha256_pinned": True,
        "truncated_count": 0,
    }


ADAPTER_CONFIG_KEYS = (
    "rank",
    "scale",
    "dropout",
    "num_layers",
    "lora_init_seed",
    "keys",
    "base_model_sha256",
    "config_sha256",
    "tokenizer_sha256",
    "tokenizer_config_sha256",
    "adapters_sha256",
    "dtype",
    "system_prompt_sha256",
    "label_order",
    "max_seq_length",
)


def read_adapter_config(path: Path) -> dict[str, Any]:
    """`adapter_config.json` を読み、既知のキーだけを検証して返す（余分なキーは捨てる）。"""
    raw = loads(
        read_input(path, MAX_ADAPTER_CONFIG_BYTES, "adapter_config.json"), "adapter_config.json"
    )
    if not isinstance(raw, dict) or not all(k in raw for k in ADAPTER_CONFIG_KEYS):
        raise invalid("invalid adapter_config.json")
    cfg = {k: raw[k] for k in ADAPTER_CONFIG_KEYS}

    def is_int(v: object) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def is_num(v: object) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    ok = (
        is_int(cfg["rank"])
        and is_int(cfg["num_layers"])
        and is_int(cfg["max_seq_length"])
        and is_int(cfg["lora_init_seed"])
        and is_num(cfg["scale"])
        and is_num(cfg["dropout"])
        and cfg["keys"] == LORA_KEYS
        and cfg["dtype"] in ("bf16", "float32")
        and isinstance(cfg["label_order"], list)
        and all(isinstance(x, str) for x in cfg["label_order"])
        and all(
            isinstance(cfg[k], str) and SHA_RE.fullmatch(cfg[k])
            for k in (
                "base_model_sha256",
                "config_sha256",
                "tokenizer_sha256",
                "tokenizer_config_sha256",
                "adapters_sha256",
                "system_prompt_sha256",
            )
        )
    )
    if not ok:
        raise invalid("invalid adapter_config.json")
    return cfg
