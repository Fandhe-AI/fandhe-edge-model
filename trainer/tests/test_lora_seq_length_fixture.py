"""REQ-41・REQ-39: qwen2_lora の max_seq_length 範囲が共有 fixture と一致することを確認する。

Rust 側は crates/core/tests/lora_seq_length_fixture.rs が同じ fixture を
fandhe_edge_core::limits と照合する。
"""

import json
from pathlib import Path

from fandhe_edge_trainer import limits

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "lora_limits" / "lora_seq_length.json"


def test_req41_lora_seq_length_matches_shared_fixture() -> None:
    fx = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert limits.MIN_LORA_SEQ_LENGTH == fx["min_lora_seq_length"] == 8
    assert limits.MAX_LORA_SEQ_LENGTH == fx["max_lora_seq_length"] == 4096
