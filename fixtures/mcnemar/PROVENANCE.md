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
`(108, 36)` の 4 件は PoC-10 の実行結果（`logs/eval/summary.json`
seed0・n=650）に由来する `b`・`c`（不一致ペア数）を使うが、`p` 値自体は
本スクリプトの厳密計算で独立に求め直した値であり、`summary.json` の
`lgamma` 由来の値をそのまま転記したものではない（
`crates/eval/tests/mcnemar_known_answer.rs` の
`poc10_seed0_majority_comparisons` のコメント参照）。

同様に `(143, 43)`・`(125, 58)`・`(144, 40)`・`(134, 58)` の 4 件
（issue #65・TASK-25.1-2 で追加）は PoC-10 `eval_incat` の seed1・seed2
実行結果（`summary.json` の C3・C4 vs majority、n=650）に由来する `b`・`c`
で、`p` 値は同じく本スクリプトの厳密計算で独立に求め直した値である
（`crates/eval/tests/baseline_significance.rs` 参照）。

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

`known_values.json` は上記コマンドの実行結果をコミットしたもの。各ケースは
次の 2 つの数値フィールドを持つ。

- `p_two_sided_f64_repr`: 厳密な有理数値（`p_two_sided_exact_fraction`）を
  17 桁精度で丸めた十進表現の**文字列**。真値をそのまま記録する目的のため、
  `f64` の表現範囲（約 4.9e-324 未満）を下回るケースでもアンダーフローさせず
  非ゼロの文字列を保持する
- `p_two_sided_f64_actual`: その文字列を実際に `float()`（IEEE754 binary64。
  Rust の `f64` と同じ丸め・アンダーフロー規則）へ変換した後の値。テストの
  `f64` リテラルに実際に格納されるのはこちらの値

`crates/eval/src/mcnemar.rs`（ユニットテスト）・
`crates/eval/tests/mcnemar_known_answer.rs`（結合テスト）にハードコードした
期待値は、`p_two_sided_f64_actual` と全件一致することを確認済み（証拠の
種別: テストハーネス。手動照合。2026-09-28）。

**例外（アンダーフロー境界のケース）**: `(b=1100, c=0)` は真値が `2^-1099`
（`p_two_sided_f64_repr` は `"1.4724303658045725e-331"`）だが、`f64` の
最小の正の値（約 4.9e-324）を大きく下回るため `p_two_sided_f64_actual` は
`0.0` になる。このケースに限り `p_two_sided_f64_repr` と
`p_two_sided_f64_actual` は一致しない（前者は真値、後者は `f64` 変換後の
値であり、両者が異なること自体が意図した仕様）。テスト側
（`mcnemar_known_answer.rs::underflow_to_zero_is_not_an_error`）は
`p_two_sided_f64_actual` の `0.0` と一致させており、「期待値は全件一致」と
は `p_two_sided_f64_actual` を基準にした記述である。

値を変更する場合は `generate_known_values.py` の `CASES` とテストコード側の
期待値を両方更新すること。
