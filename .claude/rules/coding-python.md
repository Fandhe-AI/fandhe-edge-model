# Python コーディング規約（学習ワーカー）

学習ワーカーは当面 Python（MLX）で実装する（オーナー判断 2026-09-27。Rust の環境調整中のため。spec は学習側の Rust 化範囲を「本フェーズでは確定しない」としている〔`04-requirements.md` L876 付近〕）。配置は `trainer/`（uv プロジェクト）。本規約は `trainer/` 配下に適用する。

## 位置づけと境界

- Python は学習ワーカー（候補学習・選定・書き出し）にのみ使う。**推論ランタイム・配布パッケージ・CLI コアに Python を持ち込まない**（REQ-32。Python・MLX が PATH に無い `env -i` 環境でも推論が成功すること）
- Rust 側（CLI・ジョブ管理）とは子プロセス＋JSON で通信する。入出力の JSON スキーマは Rust 側の定義を正とし、Python 側で独自のフィールドを増やさない
- ロードマップ（`06-roadmap.md` L322〜326 付近）は PoC の Python スクリプト（`calibrate.py`・`stats_mcnemar.py`・`split_train.py`・`train_mlx.py`・`selector.py` 等）の Rust への書き直しを想定している。当面 Python とする判断（2026-09-27）で Python 側に残すのは学習処理に限り、分割（データ契約）・統計・評価（評価器）は Rust 側の層に置く。Python 側に評価ロジックを再実装しない（評価器は TASK-24.1 の 1 つだけ）

## ツール

- 整形・lint は ruff（`ruff format`・`ruff check`。bandit 系の `S` ルールで `pickle`・`eval`・`shell=True` 等を機械検出する）、テストは pytest、環境・lock は uv（いずれも 2026-09-27 ユーザー承認済み。pytest の推移的依存は `iniconfig`・`packaging`・`pluggy`・`pygments`〔`trainer/uv.lock` で確認〕で、親の承認に包含。版は `trainer/pyproject.toml` と Makefile の `UV_VERSION` が正）
- `trainer/pyproject.toml` で依存を `==x.y.z` 固定し、`trainer/uv.lock` をコミットする。Python の版は `trainer/.python-version` で固定する。依存の追加・更新はユーザー承認を経る（[dependency-policy](./dependency-policy.md)）
- 検証は `make py-ci`（`py-fmt-check`・`py-lint`・`py-test`。`uv run --locked` で lock を暗黙更新しない）。CI は `.github/workflows/python-ci.yml`（[ci](./ci.md)）
- MLX を使うテストは CPU で実行し、Metal に依存するテストは実機前提テストとして既定の集合から分離する。CI（python-ci）は検証環境の macOS（arm64）で実行する。開発環境の Linux x86_64 でも `mlx[cpu]`（mlx-cpu）で `make py-ci` を実行できる（`trainer/pyproject.toml` の `[tool.uv] environments`）。macOS 固有の検査（拡張 ACL）には Linux 用の対（POSIX ACL）を用意し、片方の OS で検査を無効化しない

## コーディング

- 型ヒントを付ける（公開関数・Rust との境界の入出力は必須）
- 外部入力（データファイル・Rust から渡される JSON・設定）は読み込み時に検証し、サイズ上限を確認してから読み込む（REQ-39）
- `pickle`・`torch.load`（`weights_only=False`）・`eval` / `exec`・`yaml.load`（非 safe）など、任意コード実行につながる読み込みを使わない
- `subprocess` は引数リストで呼び、`shell=True` を使わない
- 実行時にネットワークへ接続しない（モデル重み・データセットの自動ダウンロードを含む。REQ-38）
- 乱数は seed を明示して設定し（Python・NumPy・MLX それぞれ）、決定性の要件は [evaluation-contract](./evaluation-contract.md) に従う。MLX の GPU 学習の非決定性を前提にし、決定的な再現が要るテストは CPU で行う
- エラーは Rust 側が 7 種の終了コードへ写せる形（非ゼロ終了＋機械可読な JSON の `code` / `message`）で返す（REQ-21）

## テスト・コメント

- pytest のテスト名または docstring に REQ-n を記す。期待値は具体値で書く
- モジュール・公開関数 / クラスに日本語の docstring を付ける（[code-comment-style](./code-comment-style.md)）
