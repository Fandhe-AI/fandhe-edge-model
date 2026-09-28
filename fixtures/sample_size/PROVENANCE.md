# sample_size フィクスチャの出典

## 出典

`generate_known_values.py`（本ディレクトリ）は、`crates/eval/src/sample_size.rs`・
`crates/eval/tests/required_sample_size.rs` の既知値テストが使う期待値を
再生成するスクリプト（issue #66・TASK-25.2）。McNemar 検定で下限基準との差を
検出するための必要評価件数（Connor 1987 の公式。PoC-10
`scripts/required_n_mcnemar.py` の移植）を Rust（std のみ）で実装するにあたり、
逆正規分布関数（分位点）の既知値と、4 通りの仮定から算出した必要件数の
参照値を用意する。

本リポの開発・CI 環境には scipy が無く、取得は通信を伴うため
（`.claude/rules/dependency-policy.md`）行わない。代わりに Python 標準
ライブラリのみで 2 系統の独立実装により分位点を求め、相互の一致を
確認してから出力する。

## 参照の独立性（開示）

- **(i) `statistics.NormalDist().inv_cdf`**: CPython の実装自体が
  Wichura (1988) AS241 を移植したものであり、Rust 実装（同じく AS241）と
  ほぼ同一アルゴリズム・同一係数系列である。したがって (i) との一致は
  「係数の転記ミスが無いか」の照合であり、アルゴリズム自体の独立検証には
  ならない
- **(ii) `math.erfc` による `Φ(x) = 0.5 * erfc(-x / sqrt(2))` の二分法**:
  累積分布関数の計算に `erfc`（AS241 とは異なる近似）を使い、逆関数は
  単純な二分法で求める。分位点探索のアルゴリズムが (i)・Rust 実装のいずれ
  とも独立している
- **PoC-10・PoC-24 の事前登録値**（155・188・221・272）: 本スクリプトの
  実行より前に別途記録された整数値であり、(i)(ii) のどちらとも独立な
  第三の錨。本スクリプトは算出結果がこれらと一致しない場合に異常終了する
  （`_POC_REGISTERED_CEIL` の照合）ため、`known_values.json` に矛盾した
  値が記録される余地がない

生成データは合成値（仮定のパラメータ `p_b=0.15`・`p_c=0.05`・`power=0.8`
から計算した数値）であり、個人情報・機密情報を含まない。

## 生成方法

```bash
python3 fixtures/sample_size/generate_known_values.py > fixtures/sample_size/known_values.json
```

標準ライブラリのみを使用し、依存の追加は無い（`.claude/rules/
dependency-policy.md`）。本スクリプトは Rust のビルド・テストからは呼び出さ
れない参考データの生成専用ツールであり、`crates/eval` の `Cargo.toml` の
`[dependencies]` は変更していない。`trainer/`（uv プロジェクト）の規約
（`.claude/rules/coding-python.md`）は `trainer/` 配下限定のため、本
スクリプトはその対象外。

## known_values.json との照合

- `quantiles`: 各 `p` に対する分位点 `quantile`（(i)(ii) 一致確認済みの値。
  `statistics.NormalDist().inv_cdf` の戻り値）
- `sample_sizes`: `p_b`・`p_c`・`alpha`・`power` の 4 パラメータに対する
  丸め前の必要件数 `n`（f64）と `math.ceil(n)` の `ceil_n`

`crates/eval/src/sample_size.rs`（ユニットテスト）・
`crates/eval/tests/required_sample_size.rs`（結合テスト）にハードコードした
期待値は、本ファイルの値と絶対誤差・相対誤差ともに 1e-9 以内で一致すること
を確認済み（証拠の種別: テストハーネス。手動照合。2026-09-28）。`ceil_n` は
4 ケースとも境界（次の整数との距離）が最小でも約 0.18 あり、OS ごとの
libm 実装差による丸めの入れ替わりは想定していない。

値を変更する場合は `generate_known_values.py` の `QUANTILE_CASES`・
`SAMPLE_SIZE_CASES` とテストコード側の期待値を両方更新すること。
