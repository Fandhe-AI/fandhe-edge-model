# preprocess fixture の来歴

入力表現ごとに 1 ファイルを置く。両側（Rust・Python）が同じ fixture を読み、不一致時はコードを直す（fixture・許容差を緩めない。fail-closed）。

## byte_encoding_vectors.json

- byte 入力表現（`normalize_input` + `encode_bytes`）の契約。Chore #10
- 読み手: `crates/runtime/tests/preprocess_golden.rs`、`trainer/tests/test_encoding.py`

## tokenizer_vectors.json

- tokenizer 入力表現の契約（REQ-41・TASK-41.3-3・#404）
- `_meta`: `input_representation`（固定値 `tokenizer`）・`tokenizer_json_sha256`（対象 `tokenizer.json` 本体の sha256。小文字 hex 64 桁）・`reference`（期待 ids を出した参照実装と版）・`generated_on`（`YYYY-MM-DD`）・`description`
- `vectors[]`: `name`（一意）・`input`（tokenizer の encode にそのまま渡す文字列。chat template なし・special token 追加なし）・`ids`（期待 token id 列。u32 範囲の非負整数）
- ASCII のみ（非 ASCII は `\uXXXX`）。2 スペースインデント・末尾改行
- 読み手（形式検査のみ）: `crates/runtime/tests/tokenizer_vectors_format.rs`、`trainer/tests/test_tokenizer_vectors.py`
- 中身は暫定で、`fixtures/poc26/tokenizer_golden.json` の 23 件を機械変換したもの（`name` は `poc26_01`〜`poc26_23`）。製品トークナイザーでの再生成と実トークナイザー・Rust 推論ランタイムとの全件一致は TASK-41.4-5（#413）で行う
- 未決: tokenizer 経路で `normalize_input`（NFKC・空白圧縮）を前段に挟むかは TASK-41.5・41.4 で決める。挟む場合は `normalized` 欄を追加する
- 帰属: id 列の元は Qwen/Qwen2.5-0.5B-Instruct の `tokenizer.json`（Copyright 2024 Alibaba Cloud、Apache License 2.0）。id 列のみを収めた派生物で、本体は同梱しない。全文は `LICENSE-APACHE`。詳細は `fixtures/poc26/PROVENANCE.md`
