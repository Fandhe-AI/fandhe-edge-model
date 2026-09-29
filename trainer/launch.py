"""学習ワーカーの唯一の起動口（Issue #12）。

`trainer/pyproject.toml` は `package = false`（配布パッケージを持たないスタブ）
のため、`fandhe_edge_trainer` パッケージは素の `python -m fandhe_edge_trainer ...`
では解決できず、従来は呼び出し元が `PYTHONPATH=trainer/src` を自前で設定する
必要があった（`supervisor.py::_spawn_worker_and_finalize` が子プロセス
`_worker` を起動する箇所も、この設定漏れの影響をそのまま受けていた）。
本ファイルは `__file__` から `trainer/src` の絶対パスを解決して `sys.path` へ
挿入することで、呼び出し元の環境変数に依存せず起動できる契約に一本化する。

呼び出し契約（Rust 側ジョブ管理〔TASK-34.x〕・CLI・テストが従う唯一の起動方法）:

    <venv の python> -I trainer/launch.py <サブコマンド> [引数...]

`train` は `--request <path>` に加え、opt-in の `--cancel-on-stdin-eof` を取る
（REQ-34・TASK-34.1-2・#145）。付けると標準入力をキャンセル用パイプとして
監視し、EOF で worker を止めて予約を解放する（`supervisor.py` の協調キャンセル）。
標準入力がパイプでなければ起動前に `invalid_request` で拒否する。

`-I`（隔離モード）は必須である。`-I` は `PYTHONPATH`・`PYTHONHOME` 等の
`PYTHON*` 環境変数とユーザーサイトパッケージ（`site-packages` の外側の
`~/.local` 等）を無視させ、呼び出し元の環境変数に無関係なモジュール探索パスが
紛れ込むのを防ぐ（ガード層の観点。REQ-39 に準ずる防御）。venv 自体は
`pyvenv.cfg` を起点に解決されるため、`-I` を付けても `trainer/.venv` に
インストール済みの依存（mlx・numpy・onnx）は通常どおり解決できる（実機確認済み:
`PYTHONPATH=/bogus trainer/.venv/bin/python3 -I -c "import sys; print(sys.path)"`
の出力に venv の site-packages が含まれることを確認した。証拠種別: 実機）。

`supervisor.py::worker_argv` は本ファイルを `_worker` の起動にも使う
（`-I <本ファイルの絶対パス> _worker --out-fd <n>` という argv を組み立てる）。
"""

from __future__ import annotations

import sys
from pathlib import Path

#: `trainer/src`（`fandhe_edge_trainer` パッケージの親ディレクトリ）の絶対パス。
_SRC_DIR = str(Path(__file__).resolve().parent / "src")


def _ensure_src_on_path() -> None:
    """`trainer/src` を `sys.path` の先頭に挿入する（未挿入時のみ）。"""
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)


def main() -> int:
    """`fandhe_edge_trainer.cli.main` へ委譲する。

    `sys.path` へ `trainer/src` を挿入した後でなければ `fandhe_edge_trainer` を
    import できないため、意図的に関数内 import にする（モジュール冒頭の
    import にすると `_ensure_src_on_path` より先に解決が試みられて失敗する。
    ruff の E402〔モジュール先頭以外の import〕もこの関数内 import で
    自然に回避する）。
    """
    _ensure_src_on_path()
    from fandhe_edge_trainer.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    _ensure_src_on_path()
    sys.exit(main())
