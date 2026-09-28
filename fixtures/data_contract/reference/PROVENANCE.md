# 出典（`fixtures/data_contract/reference/`）

## 本ディレクトリは空である

`train.jsonl`・`validation.jsonl`・`test.jsonl`（PoC-9 の参照データ、合計 10,359 行）は
このリポジトリに同梱しない。以前のコミット（c1b1380）でバイト単位のコピーを
同梱していたが、security.md「学習・評価データ本文をログ・エラーメッセージ・
Issue・PR へ転記しない」・spec-reference.md「spec ファイルの丸ごとコピーは
しない」に反するとのレビュー指摘（PR #188・reviewThread
PRRT_kwDOUq-SxM6mgtLm 系統。P0）を受けて削除した。

その後、`FANDHE_EDGE_POC9_DATA_DIR` 環境変数で `docs/spec` 配下を指すことを
opt-in（`#[ignore]`）の結合テストで許す設計に差し替えていたが、これも
「テストから docs/spec 配下を参照しない」（spec-reference.md「運用」）に
反し、かつ既定の検証集合から回帰検出が外れるとの指摘（PR #188 レビュー
指摘 P0・reviewThread PRRT_kwDOUq-SxM6mg8Cy・PRRT_kwDOUq-SxM6mg8Ct）を受けて
撤回した。**本リポジトリのコード・テスト・`build.rs` のいずれも `docs/spec`
配下を参照しない**（`crates/data/tests/contradiction_reference.rs` 参照）。

矛盾レコード検出（[`find_contradictions`]）の既定テストは、本ディレクトリの
ファイルではなく `crates/data/tests/contradiction_reference.rs` にインライン
で書いた手書きの合成データ（13 件。distinct_gold_count が 2 と 3 の両方の
境界値・group をまたぐエントリを含む）だけで検証する。

## 出典情報（履歴的な記録。コードからは参照されない）

以下は本ディレクトリが実データを同梱していた時点の来歴の記録であり、
現在のテスト・コードはこの情報を入力として使わない（値の再照合が必要に
なった場合は、本リポ内の許可された合成 fixture を新設し、docs/spec を
入力経路にしない）。

- 出典 PoC: `docs/spec/03-poc/evaluation-contract`（PoC-9）
- 生成元の連鎖: `e2e-minimal/data/train.jsonl`（360 行）と
  `from-scratch-feasibility/data/train_10k.jsonl`（9,999 行。LLM 生成）を
  `prepare_reference.py` で group 化・分割（seed 20260923）した結果
- 行数: `train.jsonl` 7,593 行・`validation.jsonl` 1,258 行・`test.jsonl` 1,508 行
  （合計 10,359 行）
- sha256（`docs/spec/03-poc/evaluation-contract/data/SHA256SUMS` と一致確認済み。
  当時の削除前コミットで照合したもの。現在のテストでは照合していない）:
  - `train.jsonl`: `6bb72bf73fd9621c90bc78320a3b5945d5dbdb2b2a8f9644136f875da7c7f6b7`
  - `validation.jsonl`: `e208ed1db07ffc23d1caedc13337e3b26d4ca18e6ca04c9360542132603f8a70`
  - `test.jsonl`: `2aeee9358fa02d3c0439bf24a6f1da0bcc86509013a890866dda4fb42bddf95b`
- 当時の期待値: `docs/spec/03-poc/evaluation-contract/data/contradictions.json`
  （`reference_pool.input_level`）が示す矛盾 6 個（distinct 入力）・57 行・6 group
- 証拠の種別: テストハーネス（PoC-9 のデータを移植したもの。実機ではない。
  現在は撤回済み）
