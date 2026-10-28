"""REQ-41・REQ-39: ベースモデルの重み 1 ファイルの上限が共有 fixture と一致する（#665）。

Rust 側は crates/cli/tests/base_model_file_limit_fixture.rs が同じ fixture を
fandhe_edge_guard::base_model::MAX_BASE_MODEL_FILE_BYTES と照合する。
"""

import json
from pathlib import Path

from fandhe_edge_trainer import limits

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "lora_limits" / "base_model_file.json"


def test_req41_base_model_file_limit_matches_shared_fixture() -> None:
    fx = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert limits.MAX_BASE_MODEL_FILE_BYTES == fx["max_base_model_file_bytes"] == 5 << 30
