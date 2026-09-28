# holm フィクスチャの出典

## 出典

`generate_known_values.py`（本ディレクトリ）は、`crates/eval/src/holm.rs`・
`crates/eval/tests/holm_known_answer.rs` の既知値テストが使う期待値を、
`fractions.Fraction`・`math.comb` による厳密な有理数計算（Rust 実装の
`ln`/`exp` 近似とは独立した参照実装）で McNemar 両側正確 p 値を求めたうえで、
Holm 補正（昇順ソート → `(m - k) * p` → 累積最大による単調化 → 1 で打ち切り。
PoC-10 `stats_mcnemar.py` の `holm()` と同じ手順）を有理数のまま適用して
再生成するスクリプト（issue #67・TASK-25.3。`fixtures/mcnemar/
generate_known_values.py` と同じ方針）。

## 生成元

合成された（厳密計算で導出した）数値ケースであり、実データではない（個人
情報・機密情報を含まない）。各族の `b`・`c` の由来は次のとおり。

- `poc10_eval_incat_seed0`・`poc10_eval_incat_seed1`・`poc10_eval_incat_seed2`・
  `dropout_3_of_4`: PoC-10 `eval_incat`（`logs/eval/summary.json`。n=650）の
  各 seed・候補（C1〜C4）vs majority の分割表。`b`・`c` は
  `fixtures/mcnemar/PROVENANCE.md` と同一の転記元。`p` 値自体は本スクリプトの
  厳密計算で独立に求め直した値であり、`summary.json` の `lgamma` 由来の値を
  そのまま転記したものではない
- `poc10_ref_test_seed0`: PoC-10 `ref_test` seed0 の分割表（`summary.json`）
- `poc25_primary_seed0`: PoC-25（`fresh-eval-recheck`）primary seed0 の分割表
  （`score_result.json`）
- `alpha_boundary_solo`・`alpha_boundary_pair_flips`・`alpha_boundary_pair_holds`:
  α 境界の確認用に組み立てた合成値（`(13,4)` は
  `crates/eval/tests/baseline_significance.rs` 等で使われている既知の
  境界値 `p=0.049041748046875` と同じ組）

## 生成方法

```bash
python3 fixtures/holm/generate_known_values.py > fixtures/holm/known_values.json
```

標準ライブラリのみを使用し、依存の追加は無い（`.claude/rules/
dependency-policy.md`）。本スクリプトは Rust のビルド・テストからは呼び出さ
れない参考データの生成専用ツールであり、`crates/eval` の `Cargo.toml` の
`[dependencies]` は変更していない。テストの期待値算出はテストコードへの
直書きを正としている（`.claude/rules/spec-reference.md`「ビルド・テストは
`docs/spec` 抜きで成立させる」と同じ趣旨）。`trainer/`（uv プロジェクト）の
規約（`.claude/rules/coding-python.md`）は `trainer/` 配下限定のため、本
スクリプトはその対象外。

## known_values.json との照合

`known_values.json` は上記コマンドの実行結果をコミットしたもの。各候補は
`raw_p_f64_actual`（補正前の p 値を `f64` へ変換した実値）と
`adjusted_p_f64_actual`（Holm 補正後の同様の実値）を持つ。
`crates/eval/tests/holm_known_answer.rs` にハードコードした期待値は、
`adjusted_p_f64_actual`（該当する場合は `raw_p_f64_actual`）と全件一致する
ことを確認済み（証拠の種別: テストハーネス。手動照合。2026-09-28）。

## `poc25_primary_seed0` の累積最大に関する余裕の確認

`poc25_primary_seed0` の C1（`b=156, c=36`）は補正前 p に family_size(4) を
掛けた値が C4（`b=79, c=4`）の補正後値（`4 * raw_p(C4)`）を上回るため、
Holm 補正の累積最大により C1 の補正後値は C4 の補正後値に巻き取られて一致
する。この巻き取りが Rust 実装の `ln`/`exp` 近似誤差（相対 1e-14 程度。
`crates/eval/src/mcnemar.rs` の「誤差の見積もり」節）で覆らないことを
確認するため、`known_values.json` の `poc25_primary_seed0_margin_check` に
`relative_gap_c4x4_minus_c1x3_over_c4x4`（`(C4×4 − C1×3) / C4×4`）を記録した。

実測値は `0.0120884`（約 1.2%）であり、libm 近似誤差（相対 1e-14 程度）を
11 桁以上上回る。したがって `poc25_primary_seed0` の C1・C4 が補正後に
一致するという期待値は、Rust 実装の近似計算を使っても安定して成り立つ
（証拠の種別: 推定。厳密な有理数計算と近似計算の相対差の実測値〔`crates/eval/
src/mcnemar.rs` の「誤差の見積もり」節、最大 `n=39,000` で約 2e-11〕からの
外挿ではなく、本ケースの `n`〔C1: 192・C4: 83〕は実測範囲内に収まる）。

値を変更する場合は `generate_known_values.py` の `FAMILIES` とテストコード側
の期待値を両方更新すること。
