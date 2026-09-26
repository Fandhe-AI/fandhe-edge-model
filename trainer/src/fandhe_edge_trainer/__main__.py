"""`python -m fandhe_edge_trainer` のエントリポイント。実体は `cli.main`。"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
