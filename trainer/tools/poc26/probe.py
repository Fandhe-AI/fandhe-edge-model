"""PoC-26 `probe`・`compare-probe` サブコマンド（REQ-41・TASK-41.1-5・#390）。

役割: 実重みでの人間の確認用。`probe` は固定の中立文の token id・最終位置 logits（float32）・
貪欲生成
（LoRA なし）を JSON に書く。`compare-probe` は自作側と参照側（HF tokenizers・mlx-lm）の 2 つの
probe JSON を token id 完全一致・logits の許容差・argmax で照合する（一致 0・不一致 10）。
手順と合格条件は追補 1 の 6 節。`cli.py` から配線される。
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import unicodedata
from typing import Any

import mlx.core as mx
import numpy as np
from tools.poc26.assets import (
    load_assets,
    load_base_model,
    pins_from_args,
)
from tools.poc26.common import (
    MAX_SEQ_LENGTH,
    Budget,
    invalid,
    loads,
    log,
    read_input,
    to_dtype,
)
from tools.poc26.io_records import (
    write_new_file,
)
from tools.poc26.qwen2_model import (
    Qwen2Model,
)
from tools.poc26.qwen2_tokenizer import Qwen2Tokenizer
from tools.poc26.safe_io import LimitExceededError
from tools.poc26.score import Prompting

from fandhe_edge_trainer.exitcode import ExitCode

MAX_PROBE_BYTES = 64 << 20


MAX_PROBE_LOGITS_BYTES = 4 << 20


GREEDY_TOKENS = 32


# probe の固定文。中立な文だけにし、added_tokens の文字列・学習データ本文は入れない。
PROBE_TEXTS = [
    "今日はいい天気ですね。",
    "ひらがなとカタカナと漢字、そして全角記号（テスト）！",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure they'll say we don't know, but DON'T worry.",
    "1 2 3 4 5 6 7 8 9 10 2024年10月6日",
    "全角数字１２３４５６",
    "line one\nline two\r\nline three\n\n\nend",
    "tab\tseparated\tvalues  and  double  spaces  ",
    "　全角空白　を含む　文　",
    "emoji \U0001f600\U0001f389 are fun",
    "café and résumé",
    "!! --> ... ?! ###",
    "def f(x):\n    return x * 2\n",
    "https://example.com/path?query=1&b=2",
    "Hello",
    "こんにちは世界",
    "a",
    "   leading spaces",
    "mixed 日本語 and English words 123",
    "end with newline\n",
]


def last_logits(model: Qwen2Model, ids: list[int]) -> np.ndarray:
    h = model.hidden_states(mx.array(np.array([ids], dtype=np.int32)))
    out = model.project(h[:, -1]).astype(mx.float32)
    mx.eval(out)
    return np.array(out, dtype="<f4")[0]


def safe_decode(tok: Qwen2Tokenizer, ids: list[int]) -> str:
    """モデル語彙には在るが tokenizer に無い id（実物は 151665 以降）を含んでも落とさない。"""
    try:
        return tok.decode(ids)
    except LimitExceededError:
        raise  # 上限超過は握りつぶさず 20 へ
    except ValueError:
        return "<undecodable>"


def cmd_probe(a: argparse.Namespace) -> int:
    """`probe`: 固定の中立文の token id・最終位置 logits・貪欲生成を書く（LoRA なし）。

    出力は親ディレクトリが存在し、ファイル自体は存在しないこと（排他作成・0600）。forward には
    壁時計上限（`--max-score-seconds`。超過は 20。モデル読み込みは対象外）が掛かる。
    """
    out_path = a.out
    if os.path.lexists(out_path) or not out_path.parent.is_dir():
        raise invalid("output file must not exist and its parent directory must exist")
    budget = Budget()
    assets = load_assets(a.model_dir, pins_from_args(a))
    model, sha, size = load_base_model(a.model_dir, to_dtype(a.dtype), pins_from_args(a))
    model.eval()
    budget = Budget(wall_limit=a.max_score_seconds)  # 読み込み後の forward だけを対象にする
    tok = assets.tok
    ctx = Prompting(tok, "", [], [], 0, model.config.vocab_size, MAX_SEQ_LENGTH)
    cases = []
    for text in PROBE_TEXTS:
        budget.check()
        ids = ctx.encode(text)
        logits = last_logits(model, ids)
        budget.check()  # forward の後にも確認する
        cases.append(
            {
                "text": text,
                "ids": ids,
                "roundtrip_ok": tok.decode(ids) == unicodedata.normalize("NFC", text),
                "argmax": int(np.argmax(logits)),
                "last_logits_f32_b64": base64.b64encode(logits.tobytes()).decode("ascii"),
            }
        )
    greedy = []
    end = tok.special_ids["im_end"]
    for text in PROBE_TEXTS[:3]:
        ids = tok.build_chat_ids("You are a helpful assistant.", text, add_generation_prompt=True)
        out: list[int] = []
        for _ in range(GREEDY_TOKENS):
            budget.check()
            nxt = int(np.argmax(last_logits(model, [*ids, *out])))
            budget.check()
            out.append(nxt)
            if nxt == end:
                break
        greedy.append({"prompt_text": text, "ids": out, "text": safe_decode(tok, out)})
    doc = {
        "evidence": a.evidence,
        "mlx_version": mx.__version__,
        "device": a.device,
        "dtype": a.dtype,
        "model_safetensors_sha256": sha,
        "model_safetensors_bytes": size,
        **assets.hashes,
        "vocab_size": model.config.vocab_size,
        "cases": cases,
        "greedy": greedy,
    }
    write_new_file(out_path, json.dumps(doc, ensure_ascii=False, allow_nan=False) + "\n")
    log(f"probe: cases={len(cases)} roundtrip_ok={sum(c['roundtrip_ok'] for c in cases)}")
    return 0


def decode_logits(b64: object, vocab: int) -> np.ndarray:
    if not isinstance(b64, str) or len(b64) > MAX_PROBE_LOGITS_BYTES * 2:
        raise invalid("invalid probe logits")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (ValueError, binascii.Error):  # binascii.Error は ValueError の部分型だが明示する
        raise invalid("invalid probe logits") from None
    if len(raw) != 4 * vocab or len(raw) > MAX_PROBE_LOGITS_BYTES:
        raise invalid("invalid probe logits length")
    arr = np.frombuffer(raw, dtype="<f4")
    if not np.all(np.isfinite(arr)):
        raise invalid("probe logits are not finite")
    return arr


def compare_probes(ours: Any, ref: Any, atol: float) -> dict[str, Any]:
    """2 つの probe JSON を照合する（id 完全一致・logits 差 `atol` 以下・argmax 一致）。"""
    ca = ours.get("cases") if isinstance(ours, dict) else None
    cb = ref.get("cases") if isinstance(ref, dict) else None
    if not (isinstance(ca, list) and isinstance(cb, list) and ca and len(ca) == len(cb)):
        raise invalid("probe files must have the same non-empty cases")
    # 両側の vocab_size が存在し一致することを要求する（logits の長さの解釈を固定する）。不一致は
    # モデル出力の差ではなく入力ファイルの不整合なので judged_fail ではなく 64 にする。
    sizes = [d.get("vocab_size") if isinstance(d, dict) else None for d in (ours, ref)]
    for v in sizes:
        if not isinstance(v, int) or isinstance(v, bool) or not 0 < v <= (1 << 20):
            raise invalid("invalid probe vocab_size")
    if sizes[0] != sizes[1]:
        raise invalid("probe vocab_size differs between the two files")
    vocab = sizes[0]
    id_mismatch = argmax_mismatch = logits_mismatch = 0
    max_diff = 0.0
    for x, y in zip(ca, cb, strict=True):
        if not (isinstance(x, dict) and isinstance(y, dict)) or x.get("text") != y.get("text"):
            raise invalid("probe cases do not correspond")
        if x.get("ids") != y.get("ids"):
            id_mismatch += 1
        la = decode_logits(x.get("last_logits_f32_b64"), vocab)
        lb = decode_logits(y.get("last_logits_f32_b64"), vocab)
        diff = float(np.max(np.abs(la.astype(np.float64) - lb.astype(np.float64))))
        max_diff = max(max_diff, diff)
        logits_mismatch += diff > atol
        argmax_mismatch += int(np.argmax(la)) != int(np.argmax(lb))
    ok = not (id_mismatch or argmax_mismatch or logits_mismatch)
    return {
        "status": "match" if ok else "mismatch",
        "cases": len(ca),
        "id_mismatches": id_mismatch,
        "argmax_mismatches": argmax_mismatch,
        "logits_mismatches": int(logits_mismatch),
        "max_abs_diff": max_diff,
        "atol": atol,
    }


def cmd_compare_probe(a: argparse.Namespace) -> int:
    """`compare-probe`: 一致なら 0、不一致なら 10（judged_fail）。結果は stdout に JSON 1 つ。"""
    docs = [
        loads(read_input(p, MAX_PROBE_BYTES, "probe file"), "probe file") for p in (a.ours, a.ref)
    ]
    result = compare_probes(docs[0], docs[1], a.atol)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "match" else int(ExitCode.JUDGED_FAIL)
