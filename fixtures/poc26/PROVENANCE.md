# poc26 fixture の来歴

- `tokenizer_golden.json`: PoC-26 のトークナイザー golden（REQ-41・TASK-41.1-5・#390）。中立な probe 文 23 件と、その id 列を収める
- 生成元: Qwen/Qwen2.5-0.5B-Instruct（revision `7ae557604adf67be50417f59c2c2f167def9a775`）の `tokenizer.json` を、HF `tokenizers` 0.23.2 で encode した結果。生成手順は `docs/design/poc26-tokenizer-golden-procedure.md`。証拠種別: 実機（2026-10-07）
- 帰属: 元の `tokenizer.json` は Copyright 2024 Alibaba Cloud、Apache License 2.0。本 fixture は id 列のみを収めた派生物で、`tokenizer.json` 本体は同梱しない。Apache License 2.0 の全文はリポジトリ直下の `LICENSE-APACHE` にある
- 変更: 元ファイルは改変していない。probe 文は本リポで作成した中立な文で、学習・評価データの本文や個人情報・機密を含まない
- 同梱の判断: オーナーが 2026-10-07 に「問題ないものなら可」と判断（#387）。上の帰属表示を付けて同梱する

- `pretokenize_golden.json`: 事前分割（NFC・Split）の golden（REQ-41・TASK-41.4-3・#411）。生成元は Python 参照実装 `trainer/tools/poc26/qwen2_tokenizer.py` の `pre_tokenize` で、生成器は `trainer/tools/gen_pretokenize_golden.py`。証拠種別: テストハーネス。本文は本リポで作成した中立な文と seed 固定の乱択で、実データ・個人情報は含まない
- `crates/runtime/src/tokenizer/unicode_tables.rs`（生成物）: `\p{L}`・`\p{N}` の範囲表。Python 3.12 の `unicodedata`（Unicode Character Database 15.0.0）から導いた範囲データで、出典は Unicode License v3。著作権表示と許諾文は同ディレクトリの `UNICODE-LICENSE.txt` に同梱する（文面と配置はオーナー確認待ち。licensing.md）
