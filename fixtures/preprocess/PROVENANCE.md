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
- 中身は `fixtures/poc26/tokenizer_golden.json` の 23 件を機械変換したもの（`name` は `poc26_01`〜`poc26_23`）。id 列は独立した参照実装（HF `tokenizers` 0.23.2）の出力を正とし、自前実装で出し直さない（参照の独立性を保つため。TASK-41.4-5・#413 の判断）
- Rust 推論ランタイムとの実物での全件一致: `crates/runtime/tests/tokenizer_golden.rs::req41_real_tokenizer_vectors_all_match`（実機前提・`#[ignore]`。手順は `AGENTS.md`）
- 未決: tokenizer 経路で `normalize_input`（NFKC・空白圧縮）を前段に挟むかは TASK-41.5・41.4 で決める。挟む場合は `normalized` 欄を追加する
- 帰属: id 列の元は Qwen/Qwen2.5-0.5B-Instruct の `tokenizer.json`（Copyright 2024 Alibaba Cloud、Apache License 2.0）。id 列のみを収めた派生物で、本体は同梱しない。全文は `LICENSE-APACHE`。詳細は `fixtures/poc26/PROVENANCE.md`

## tokenizer_synthetic.json・tokenizer_vectors_synthetic.json

- CI で回す tokenizer 全件一致用の合成 fixture（REQ-41・REQ-32・TASK-41.4-5・#413。証拠種別: テストハーネス）。本リポで作った合成データで、外部の派生物ではない
- 生成: `trainer/tools/gen_tokenizer_vectors.py`。`tokenizer_synthetic.json` は `tools/poc26/synthetic.py` の合成 tokenizer、`tokenizer_vectors_synthetic.json`（形式は `tokenizer_vectors.json` と同じ）の期待 id は学習側の製品実装 `Qwen2Tokenizer.encode` が出す。`_meta.tokenizer_json_sha256` は前者のバイト列の sha256
- 読み手: `crates/runtime/tests/tokenizer_golden.rs`・`env_isolation.rs`（`env -i`）・`tokenizer_vectors_format.rs`、`trainer/tests/test_tokenizer_vectors.py`（生成物が古くないことの検査を含む）
- ASCII のみ。sha256 照合のため `.gitattributes` で `tokenizer_synthetic.json` を LF 固定
