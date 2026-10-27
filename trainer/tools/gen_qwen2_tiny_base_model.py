"""合成の極小ベースモデル（Qwen2 系）を指定ディレクトリへ書き、sha256 の写像を出す（#628）。

CLI の `train --kind qwen2_lora --base-model-dir` を実 trainer で通す結合テスト
（`crates/cli/tests/pipeline_real_trainer_qwen2.rs`。`#[ignore]`）が、定義ファイルの
`base_model.sha256` に書く値を得るために使う。中身は `gen_qwen2_parity_fixture.write_base_model`
（`poc26/synthetic` の極小 config・seed 固定の乱数重み）と同じで、外部データ・事前学習済み
重みを含まない。通信しない（REQ-38）。

使い方: `python tools/gen_qwen2_tiny_base_model.py --out <新規ディレクトリ>`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_qwen2_parity_fixture import write_base_model


def main() -> None:
    """`--out` を作ってベースモデルを書き、`{ファイル名: sha256}` を 1 行の JSON で出す。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    d = write_base_model(Path(ap.parse_args().out))
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}
    print(json.dumps(sha, sort_keys=True))


if __name__ == "__main__":
    main()
