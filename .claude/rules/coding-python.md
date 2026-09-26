# Python コーディング規約（学習ワーカー）

学習ワーカーの実装言語は未確定（Python〔MLX〕か Rust〔candle 等〕か。`04-requirements.md` L873 付近で「本フェーズでは確定しない」）。本規約は Python で実装する部分に適用し、配置・ツールの確定時に更新する。

## 位置づけと境界

- Python は学習ワーカー（候補学習・選定・書き出し）にのみ使う。**推論ランタイム・配布パッケージ・CLI コアに Python を持ち込まない**（REQ-32。Python・MLX が PATH に無い `env -i` 環境でも推論が成功すること）
- Rust 側（CLI・ジョブ管理）とは子プロセス＋JSON で通信する。入出力の JSON スキーマは Rust 側の定義を正とし、Python 側で独自のフィールドを増やさない
- PoC の Python スクリプト（`calibrate.py`・`stats_mcnemar.py`・`split_train.py`・`train_mlx.py`・`selector.py` 等）のうち、評価器・分割・選定・統計は Rust へ書き直す計画（`06-roadmap.md` L324 付近）。Python 側に評価ロジックを再実装しない（評価器は TASK-24.1 の 1 つだけ）

## ツール

- 整形・lint は ruff（`ruff format`・`ruff check`）、テストは pytest、環境・lock は uv を想定する（導入・バージョン固定は依存追加としてユーザー承認を経る。[dependency-policy](./dependency-policy.md)）
- `pyproject.toml` で依存を `==x.y.z` 固定し、lock ファイルをコミットする。Python の版も固定する
- ruff / pytest の Makefile ターゲット・CI 組み込みは Python コードの追加時に infra-builder が行う（[ci](./ci.md)）

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
