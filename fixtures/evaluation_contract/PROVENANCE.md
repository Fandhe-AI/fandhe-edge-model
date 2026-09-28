# evaluation_contract フィクスチャの出典

## 出典

`docs/spec/03-poc/evaluation-contract/fixtures/`（PoC-9。private submodule）から、
バイト単位でそのまま移植した（内容を加工していない）。ただし
`anomaly/01-empty-data/expected.json`・`anomaly/05-duplicate-id/expected.json`・
`anomaly/10-invalid-score/expected.json` の 3 ファイルは例外で、キー名・値を
改名・更新している（詳細は後述の「PoC-9 との既知の差分」）。

- `known/single-select/{gold.jsonl,pred.jsonl,labels.json,expected.json}`
- `anomaly/01-empty-data/{gold.jsonl,pred.jsonl,expected.json}`
- `anomaly/02-missing-gold/{gold.jsonl,pred.jsonl,expected.json}`
- `anomaly/03-unknown-label/{gold.jsonl,pred.jsonl,labels.json,expected.json}`
- `anomaly/04-type-invalid/{gold.jsonl,pred.jsonl,labels.json,expected.json}`
- `anomaly/05-duplicate-id/{gold.jsonl,pred.jsonl,expected.json}`
- `anomaly/06-duplicate-input/{gold.jsonl,pred.jsonl,expected.json}`
- `anomaly/07-contradiction/{gold.jsonl,pred.jsonl,expected.json}`
- `anomaly/08-label-order/{gold.jsonl,pred.jsonl,labels.json,labels_reordered.json,expected.json}`
- `anomaly/09-unseen-class/{gold.jsonl,pred.jsonl,labels.json,expected.json}`
- `anomaly/10-invalid-score/{gold.jsonl,pred.jsonl,labels.json,expected.json}`
- `anomaly/11-all-abstain/{gold.jsonl,pred.jsonl,labels.json,expected.json}`
- `anomaly/12-all-error/{gold.jsonl,pred.jsonl,labels.json,expected.json}`

`known/single-select/expected-derivation.md`（手計算の導出過程メモ）は移植していない。

## 移植範囲

issue #55（TASK-23.1-1）が `known/single-select` と `anomaly/01`〜`06` を、
issue #56（TASK-23.1-2）が `anomaly/07`〜`12`（矛盾・ラベル順序・未出現クラス・
不正なスコア・全件保留・全件失敗）を移植した。PoC-9 の `fixtures/anomaly/` は
これで全 12 ケースの移植を完了している。

## 生成元

PoC-9 のフィクスチャは、評価契約（REQ-21〜27・REQ-29）の異常系挙動を固定するために
手作りで作成された合成データである（実データではない。個人情報・機密情報を含まない）。

## labels.json が無いケースの扱い

`anomaly/01-empty-data`・`02-missing-gold`・`05-duplicate-id`・`06-duplicate-input`・
`07-contradiction` には `labels.json` が無い。PoC-9 のテストハーネス
（`evaluator/harness.py`）と同じく、これらのケースでは
`known/single-select/labels.json`（ラベル `A`・`B`・`C`・`D`）を共通のラベル定義として
使う（本リポの結合テスト `crates/data/tests/eval_input_anomaly.rs` も同じ規約に従う）。

## PoC-9 との既知の差分

- `01-empty-data` の `gold.jsonl`・`pred.jsonl` は PoC-9 と同じく 0 バイト
  （空行 1 つではなく完全な空ファイル）。`.editorconfig` の
  `insert_final_newline = true` は 0 バイトファイルには適用されない
  （editorconfig-checker は空ファイルを「改行なし」として指摘しないことを
  `make lint-docs` で確認済み。証拠種別: テストハーネス）。
- `10-invalid-score` の `expected.json` は当初 PoC-9 **v1.0** 時点の記述
  （`expected_action: "warn_exclude"`）のままバイト単位で移植していたが、
  実装（v1.1・addendum A-2「除外せず error として分母に含め、不正解として
  数える」＝`WarningAction::IncludeAsError`）と食い違い、期待値の新旧を
  区別できないとのレビュー指摘（PR #204）を受けて `expected_action`・
  `excluded_ids`・`included_ids` を v1.1 の挙動へ更新した（`01-empty-data`・
  `05-duplicate-id` と同様にキー名・値を改名した例外ファイルとして扱う）。
  `crates/data/src/eval_input.rs` の `prepare_evaluation_input` は v1.1 の
  挙動を固定する（結合テスト `req23_case10_invalid_score_include_as_error` が
  照合。ただし本ファイル自体は結合テストから参照されない参考データであり、
  期待値の照合はテストコード側でハードコードしている）。addendum A-4 が
  スコア合計の許容差を `1e-6` と定めている（評価契約の指標一致判定の許容差
  `1e-9` とは別物。`.claude/rules/evaluation-contract.md`「決定性」参照）。
- `07-contradiction` の PoC-9 ログは v1.0 で実行したもの。v1.1 ではラベル
  定義ファイルが無いと停止する（addendum A-6）ため、上記の
  「labels.json が無いケースの扱い」の規約（`known/single-select` を使う）を
  適用する。矛盾（`ContradictoryInput`）の除外方針自体は v1.0・v1.1 で
  変わらない。
- pred 側の `NaN`・`Infinity`・`-Infinity`（JSON 標準外リテラル。10-invalid-score
  の `pred.jsonl` が使用）は、PoC-9 の Python `json`（`allow_nan=True`）は
  受理するが `serde_json` は受け付けない。本リポはトップレベル `scores`
  フィールドの値限定で該当トークンを `null` へ置換して再パースし、該当行を
  無条件に `invalid_score` として扱う緩和パース
  （`eval_input::substitute_non_finite_literals`。pred 側限定・gold 側は
  緩和しない）で吸収する。`scores` 以外に出現した該当トークンは置換されず
  `malformed_json` として停止する（レビュー指摘。PR #204）。
- ケース 02・03・04・06 は、TASK-23.1-2 で `WarningCode::UnseenClass`
  （ケース 9 の未出現クラス検出）を追加した副作用として、既存のラベル定義に
  対して active な gold 行がすべてのラベルをカバーしていないため、
  `unseen_class` 警告が追加で出るようになった（PoC-9 v1.1 の stderr ログと
  一致。結合テストの `assert_last_warning_is_unseen_class` 参照）。
- `01-empty-data`・`05-duplicate-id` の `expected.json` は、PoC-9 が使う
  キー名 `expected_exit_code`（値 `2`）を `poc9_expected_exit_code` へ改名した
  （PR #203 レビュー指摘）。PoC-9 自身の終了コード契約は `2` を「停止」の
  意味で使うが、本リポの REQ-21 は 7 種の終了コード
  （`ok`=0・`judged_fail`=10・`out_of_scope`=11・`pending`=12・
  `limit_exceeded`=20・`invalid_input`=64・`runtime_error`=70）に固定して
  おり `2` は含まれない。元のキー名のままだと、本リポの評価器・CLI の
  期待値として `2` を誤って踏襲しかねないため、PoC-9 由来の参考値である
  ことが分かる名前に変更し、値そのもの（`2`）はバイト単位の移植として保持した
  （ExitCode への対応付けは CLI 側（TASK-21.2・TASK-33.x）の責務で、
  `crates/data/src/eval_input.rs` は `EvalInputStop::code()` の文字列
  （`empty_data`・`duplicate_id`）だけを返す）。
