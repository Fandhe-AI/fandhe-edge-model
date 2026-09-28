# mcnemar フィクスチャの出典

## 出典

`generate_known_values.py`（本ディレクトリ）は、`crates/eval/src/mcnemar.rs`・
`crates/eval/tests/mcnemar_known_answer.rs` の既知値テストが使う期待値を、
`fractions.Fraction`・`math.comb` による厳密な有理数計算（Rust 実装の
`ln`/`exp` 近似とは独立した参照実装）で再生成するスクリプト。review 指摘
（issue #64・TASK-25.1-1）を受けて追加した。「計画時に Python で算出した」
参照値の生成手段が本リポにコミットされていなかったため、19 個の期待値の
再現性を確保する目的で置く。

## 生成元

合成された（手計算・厳密計算で導出した）数値ケースであり、実データではない
（個人情報・機密情報を含まない）。`(130, 29)`・`(123, 26)`・`(161, 52)`・
`(108, 36)` の 4 件のみ PoC-10 の実行結果（`logs/eval/summary.json`
seed0・n=650）に由来する `b`・`c`（不一致ペア数）を使うが、`p` 値自体は
本スクリプトの厳密計算で独立に求め直した値であり、`summary.json` の
`lgamma` 由来の値をそのまま転記したものではない（
`crates/eval/tests/mcnemar_known_answer.rs` の
`poc10_seed0_majority_comparisons` のコメント参照）。

## 生成方法

```bash
python3 fixtures/mcnemar/generate_known_values.py > fixtures/mcnemar/known_values.json
```

標準ライブラリのみを使用し、依存の追加は無い（`.claude/rules/
dependency-policy.md`）。本スクリプトは Rust のビルド・テストからは呼び出さ
れない参考データの生成専用ツールであり、`crates/eval` の `Cargo.toml` の
`[dependencies]` は空のまま（`.claude/rules/spec-reference.md`「ビルド・
テストは `docs/spec` 抜きで成立させる」と同じ趣旨で、テストの期待値算出は
テストコードへの直書きを正としている）。`trainer/`（uv プロジェクト）の
規約（`.claude/rules/coding-python.md`）は `trainer/` 配下限定のため、本
スクリプトはその対象外。

## known_values.json との照合

`known_values.json` は上記コマンドの実行結果をコミットしたもの。
`crates/eval/src/mcnemar.rs`（ユニットテスト）・
`crates/eval/tests/mcnemar_known_answer.rs`（結合テスト）にハードコードした
期待値は、本ファイルの `p_two_sided_f64_repr`（17 桁精度で丸めた十進表現）
と全件一致することを確認済み（証拠の種別: テストハーネス。手動照合。
2026-09-28）。値を変更する場合は `generate_known_values.py` の `CASES` と
テストコード側の期待値を両方更新すること。
