# ライセンス規約（リポ固有）

## 本体ライセンス

- 本リポは `MIT OR Apache-2.0` のデュアルライセンス（`LICENSE-MIT`・`LICENSE-APACHE`。README「ライセンス」）
- 各 crate の `Cargo.toml` に `license = "MIT OR Apache-2.0"` を記載する（`[workspace.package]` で共通化）。Python パッケージの `pyproject.toml` も同じ表記に揃える

## 依存ライセンス

- 依存は MIT / Apache-2.0 / BSD / ISC / Unlicense / Zlib 等の permissive ライセンスに限る（`deny.toml` の許可リスト）
- GPL / LGPL / AGPL 系・MPL-2.0 等のコピーレフト系の依存は導入しない
- `cargo deny check licenses`（`make deny`・`make ci`・CI の rust-ci に含まれる）で許可外ライセンスを検出したら fail させる。Python 依存は `cargo deny` の対象外のため、導入時に手動確認する

## モデル重み・データ

- 事前学習済み重み・語彙・データセットはソフトウェアと別のライセンス（利用規約・再配布制限・出力の利用制限を含む）を持つことが多い。取得・同梱・配布パッケージへの埋め込みの前にライセンスと再配布可否を確認し、ユーザーの判断を仰ぐ
- サンプル・テスト用データの生成元（LLM 生成等）と利用条件を記録する

## subagent への適用

- ライセンス判断が必要な事項（新規ライセンスの許可・重み / データの再配布可否・帰属表示の要否）は Agent が決めず、ユーザーへ報告する
