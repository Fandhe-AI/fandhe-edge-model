"""成果物（モデルパッケージ）の書き出し・完全性検証。

Rust 側 `Artifact` 構造体（PoC-16。`docs/spec/03-poc/core-cli-vertical-slice/core/src/
artifact.rs`）とフィールド名・型を揃える。本ワーカーは C3 の学習・ONNX 書き出しまでを
担い、複数候補（C1・C3 等）からの選定（TASK-18.x）は行わないため、`candidate_label`
には学習した種類の名前（例: "c3"）を入れる（将来 TASK-18.x 側の選定処理が複数候補の
成果物を比較する際、この値を選定結果で上書きする）。

**`onnx_sha256`（PoC-16 の `Artifact` 構造体からの拡張フィールド）**: `model.onnx`
に書き込んだ厳密なバイト列の SHA-256（小文字 16 進数）を記録する（AGENTS.md
ガード層「完全性と版」・REQ-39）。`verify_output` はワーカーが書いた直後に
自己整合性（「書いたと主張するバイト列」と「実際に読めるバイト列」が一致するか）
だけを確認する。これは暗号学的な改ざん検知（署名等）ではなく、ワーカー・
スーパーバイザーという同じ信頼境界内での自己矛盾の検出に留まる。配布パッケージを
実際に読み込む側（推論ランタイム・パッケージ層。REQ-39 の完全性・TASK-28・
TASK-30.x）が、読み込み時にこの `onnx_sha256` を検証する契約は別途そちら側の
責務とする（本モジュールはハッシュを記録するところまでを担う）。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
from datetime import UTC, datetime
from typing import IO, Any

from .errors import WorkerError
from .exitcode import ExitCode
from .limits import MAX_MODEL_BYTES

#: 選択口（本ワーカー）のバージョン。artifact.json の再現性の記録に使う。
SELECTOR_VERSION = "0.1"

ONNX_FILE_NAME = "model.onnx"
ARTIFACT_FILE_NAME = "artifact.json"

#: `verify_output` が `model.onnx` を読み込む際の上限（bytes）。学習側で検証済みの
#: `MAX_MODEL_BYTES`（パラメータ数からの見積もり）に、protobuf のオーバーヘッド分の
#: 余裕を持たせる（「小さな余裕」。厳密な上限は学習側の `budget.check_model_bytes`
#: が既に検査済みなので、ここでは異常な巨大ファイルを弾ければ十分）。
_ONNX_READ_CAP_BYTES = MAX_MODEL_BYTES + 4 * 1024 * 1024

#: `artifact.json` 自体の読み込み上限（bytes）。メタデータのみで巨大にはならない
#: ため、`limits.py::MAX_REQUEST_BYTES` と同じ桁数の暫定値を使う。
_ARTIFACT_JSON_READ_CAP_BYTES = 1 * 1024 * 1024


def now_utc() -> str:
    """`%Y-%m-%dT%H:%M:%SZ` 形式の現在時刻（UTC）。"""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class HashingWriter:
    """書き込みバイト列の SHA-256 を計算しながら、内側のファイルへそのまま
    転送する薄いラッパー（`io.RawIOBase`/`io.BufferedIOBase` 相当の `write` だけ
    実装する。`Kind.export_onnx` は「バイト列を書き込むだけ」の契約
    〔`kinds/__init__.py`〕なので、`export_onnx` 側を変更せずにハッシュを
    横取りできる）。
    """

    def __init__(self, inner: IO[bytes]) -> None:
        self._inner = inner
        self._hasher = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self._hasher.update(data)
        return self._inner.write(data)

    def hexdigest(self) -> str:
        """これまでに書き込んだバイト列の SHA-256（小文字 16 進数）。"""
        return self._hasher.hexdigest()


def build_artifact(
    *,
    kind: str,
    kind_version: int,
    config: dict[str, Any],
    label_order: list[str],
    output_type: str,
    max_bytes: int,
    candidate_label: str,
    onnx_sha256: str,
) -> dict[str, Any]:
    """Rust 側 `Artifact` 構造体と同じフィールドに `onnx_sha256`（拡張。本モジュール
    のドキュメント参照）を加えた辞書を作る。
    """
    return {
        "kind": kind,
        "kind_version": kind_version,
        "selector_version": SELECTOR_VERSION,
        "config": config,
        "label_order": label_order,
        "output_type": output_type,
        "max_bytes": max_bytes,
        "onnx_file": ONNX_FILE_NAME,
        "onnx_sha256": onnx_sha256,
        "created_utc": now_utc(),
        "candidate_label": candidate_label,
    }


def write_artifact(tmp_fd: int, artifact: dict[str, Any]) -> None:
    """`artifact.json` を書き出す（改行は LF 固定。`json.dumps` の既定どおり）。

    `tmp_fd` は呼び出し元（`cli.py::run_worker_train`）が `guard`/`contract` 経由で
    root 配下に閉じ込め済みの作業用一時ディレクトリの fd。経路文字列ではなく
    `dir_fd` 相対でファイルを作成することで、TOCTOU 無しに閉じ込めを保つ
    （`contract.py`・`guard.py` のモジュール docstring 参照）。新規作成のみを
    許す（`O_EXCL`。既に同名のファイルがあれば失敗させ、上書きしない）。
    """
    text = json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
    fd = os.open(
        ARTIFACT_FILE_NAME,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=tmp_fd,
    )
    with os.fdopen(fd, "wb") as f:
        f.write(text.encode("utf-8"))


def _integrity_error(message: str) -> WorkerError:
    return WorkerError("integrity_check_failed", message, ExitCode.RUNTIME_ERROR)


def _read_artifact_via_fd(dir_fd: int) -> Any:
    """`dir_fd` 配下の `artifact.json` を読み、パースして返す（`O_NOFOLLOW`・
    サイズ上限付き）。
    """
    try:
        fd = os.open(ARTIFACT_FILE_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    except OSError as e:
        raise _integrity_error(f"artifact.json not readable: {type(e).__name__}") from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise _integrity_error("artifact.json is not a regular file")
        if st.st_size > _ARTIFACT_JSON_READ_CAP_BYTES:
            raise _integrity_error("artifact.json exceeds size limit")
        with os.fdopen(fd, "rb") as f:
            fd = -1  # fdopen が所有権を持つ（with 終了時に閉じる）。二重クローズを避ける。
            raw = f.read(_ARTIFACT_JSON_READ_CAP_BYTES + 1)
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)
    if len(raw) > _ARTIFACT_JSON_READ_CAP_BYTES:
        raise _integrity_error("artifact.json exceeds size limit")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise _integrity_error(f"artifact.json is not valid JSON: {type(e).__name__}") from e


def _sha256_of_onnx_via_fd(dir_fd: int) -> str:
    """`dir_fd` 配下の `model.onnx` を `O_NOFOLLOW`・サイズ上限付きで読み、
    SHA-256（小文字 16 進数）を計算して返す。
    """
    try:
        fd = os.open(ONNX_FILE_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    except OSError as e:
        raise _integrity_error(f"model.onnx not readable: {type(e).__name__}") from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise _integrity_error("model.onnx is not a regular file")
        if st.st_size > _ONNX_READ_CAP_BYTES:
            raise _integrity_error(f"model.onnx exceeds {_ONNX_READ_CAP_BYTES} bytes")
        hasher = hashlib.sha256()
        total = 0
        with os.fdopen(fd, "rb") as f:
            fd = -1  # fdopen が所有権を持つ（with 終了時に閉じる）。二重クローズを避ける。
            while True:
                # fstat 後にファイルが伸長されうる（TOCTOU）ため、実際の読み取り量も
                # 上限で縛る。上限 +1 バイトまで読めた時点で超過として拒否する
                # （REQ-39 ガード層: 資源の上限）。
                chunk = f.read(min(1024 * 1024, _ONNX_READ_CAP_BYTES + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > _ONNX_READ_CAP_BYTES:
                    raise _integrity_error(f"model.onnx exceeds {_ONNX_READ_CAP_BYTES} bytes")
                hasher.update(chunk)
        return hasher.hexdigest()
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)


def verify_output(dir_fd: int) -> None:
    """`dir_fd` 配下の `model.onnx`・`artifact.json` を読み、`artifact.json` に
    記録された `onnx_sha256` と実際の `model.onnx` のバイト列から計算した
    SHA-256 が一致することを確認する（AGENTS.md ガード層「完全性と版」・
    REQ-39）。

    確定（`contract.finalize_out_dir`）の**直前**にスーパーバイザーが、保持し
    続けている `tmp_fd`（名前を再解決しない）に対して呼ぶ。不一致・欠落・
    読み込み失敗はすべて `WorkerError("integrity_check_failed",
    ExitCode.RUNTIME_ERROR)` とし、呼び出し元は `contract.cleanup_reservation`
    で予約を解放して `out_dir` を確定させない（本モジュールのドキュメント冒頭も
    参照。本関数はワーカー自身の自己矛盾の検出に留まり、配布パッケージの読み込み
    時の検証は推論ランタイム・パッケージ層の責務）。
    """
    artifact = _read_artifact_via_fd(dir_fd)
    if not isinstance(artifact, dict):
        raise _integrity_error("artifact.json is not a JSON object")
    expected = artifact.get("onnx_sha256")
    if not isinstance(expected, str) or not expected:
        raise _integrity_error("artifact.json missing onnx_sha256")
    actual = _sha256_of_onnx_via_fd(dir_fd)
    if actual != expected:
        raise _integrity_error("model.onnx sha256 mismatch")
