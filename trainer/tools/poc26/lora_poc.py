"""PoC-26 追加学習候補 P の学習・採点 CLI の入口（REQ-41・TASK-41.1-5・#390）。

役割: `python -m tools.poc26.lora_poc <train|predict|probe|compare-probe> ...` と
`python trainer/tools/poc26/lora_poc.py ...` の入口として薄く残す（実装は `cli.py` と各モジュール。
入口の名前を変えないので、追補・AGENTS.md・手順書の実行例がそのまま使える）。スクリプト実行でも
`tools`・`fandhe_edge_trainer` を import できるよう、trainer 配下の 2 つのパスを先頭に足す。

使い方・契約・終了コードは `cli.py` の docstring と
追補 `docs/design/poc26-preregistration-addendum-1.md`。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tools.poc26.cli import main

if __name__ == "__main__":
    sys.exit(main())
