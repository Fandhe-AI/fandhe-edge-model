# fandhe-edge-model

用途専用の小型ローカル判定モデルを作成・評価するツールの実装リポジトリです。利用者が用意した選択肢（ラベル）・入出力構造・学習データ・独立した評価データから、数十 MB 以下を志向する判定モデルを作り、Claude Code・Codex などから呼び出せるようにすることを目指します。

## 位置づけ

- **本リポジトリは public** です（OSS として公開する方針）
- **仕様・要件定義**: [fandhe-edge-model-spec](https://github.com/Fandhe-AI/fandhe-edge-model-spec)（`docs/spec` に submodule 参照。**private リポジトリとして意図的に非公開を維持**する方針であり、アクセス権のない環境からは submodule を解決できません）

## ステータス

実装は未着手です（ロードマップの着手判定は「着手」済み。実装開始は別途の指示を経て行います）。要件は spec リポの [`04-requirements.md`](https://github.com/Fandhe-AI/fandhe-edge-model-spec/blob/main/04-requirements.md)（REQ-15〜40）、タスク定義は [`05-tasks.md`](https://github.com/Fandhe-AI/fandhe-edge-model-spec/blob/main/05-tasks.md)（87 件・35.0 人日）、マイルストーンは [`06-roadmap.md`](https://github.com/Fandhe-AI/fandhe-edge-model-spec/blob/main/06-roadmap.md)（M6〜M11。初期スコープは M6〜M10）を参照してください。

## 実装方針（要点）

- **薄い統合ツール**: 定義・データ検査・評価契約・成果物管理・CLI・推論ランタイムを自作し、学習は差し替え可能なワーカー（MLX など、または Rust の candle／burn）として呼び出します（暫定の推奨構成）
- **推論は学習に依存しない**: 推論ランタイムと配布パッケージに学習側の依存を持ち込みません
- **モデルの種類の選択口**: 判別型・生成型などの種類を同じ選択口から選べるようにし、種類は PoC の結果で増やします
- **評価契約**: 評価データの凍結・ハッシュの不変・下限基準に対する有意性の判定を、評価器の正しさとあわせて保証します
- **入力表現**: byte のみで確定しています
- **呼び出し口**: CLI＋JSON を基本とします。ローカル MCP・Codex からの呼び出しも要件に含みますが、実クライアントとの接続は初期スコープ外です
- **ローカル完結**: 推論・学習・評価はローカルで完結させます
- **実装言語の境界**: 共通コアを Rust 中心とする案を暫定で推奨しています。学習側をどこまで Rust に寄せるかは未決です

詳細な要件は spec リポの [`04-requirements.md`](https://github.com/Fandhe-AI/fandhe-edge-model-spec/blob/main/04-requirements.md) を唯一の正（SSOT）とします。

## 開発環境構築

```bash
git clone git@github.com:Fandhe-AI/fandhe-edge-model.git
cd fandhe-edge-model
git submodule update --init   # docs/spec（private・要アクセス権）
```

`docs/spec`（`fandhe-edge-model-spec`）は private リポジトリのため、アクセス権のない環境では submodule 取得が失敗します。実装コードのビルド・テストは `docs/spec` 抜きでも成立するよう維持します。

## ライセンス

MIT OR Apache-2.0 のデュアルライセンスです（[LICENSE-MIT](./LICENSE-MIT) / [LICENSE-APACHE](./LICENSE-APACHE)）。
