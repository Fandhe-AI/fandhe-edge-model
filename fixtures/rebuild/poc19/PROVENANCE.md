# rebuild/poc19 フィクスチャの出典

## 出典・移植方針

`docs/spec/03-poc/model-lifecycle/`（PoC-19）の事前固定 5 パターン
（`definitions/catalog_v1.json`〜`catalog_v5.json` 相当）を、本番の定義ファイル
スキーマ（`fandhe-edge-model-definition/v1`。`crates/core/src/definition.rs`）へ
手作業で変換して移植した（`docs/spec` はビルド・テストから参照しない。
spec-reference.md「運用」）。TASK-20.1-2（issue #91・REQ-20）が
`crates/core/src/rebuild.rs`・`crates/core/tests/rebuild_decision.rs` の
具体値テストとして利用する。

## 変換規則

- `schema`: `fandhe-edge-model-catalog/v1` → `fandhe-edge-model-definition/v1`
- `catalog_id` → `name`（値 `topic_a` はそのまま）
- `version`: PoC の値（1〜5）を保持
- `merges`: 本番スキーマに存在しない（`RawDefinition` は `deny_unknown_fields`）
  ため削除した。v3（統合パターン）の対応関係は下表に文章で残す
- `options`（`id`・`display_name`・`description`）・`judgment_type`: そのまま
- `io`: `{ "input": "bytes" }` を追加（本番スキーマの必須フィールド。PoC には
  対応する概念がないため新設）

## パターン対応表

PoC-19 の事前登録パターンと同じ組（`rebuild.rs` モジュール doc 参照）。

| # | 旧ファイル | 新ファイル | 変更内容 | 期待される判定 |
| - | ---------- | ---------- | -------- | --------------- |
| P1 追加 | `v2_8rm.json` | `v1_9.json` | `tier-xl__high` を追加（8→9 選択肢） | `Required(OptionIdsChanged)` |
| P2 削除 | `v1_9.json` | `v2_8rm.json` | `tier-xl__high` を削除（9→8 選択肢） | `Required(OptionIdsChanged)` |
| P3 統合 | `v1_9.json` | `v3_8merge.json` | `tier-l__medium` + `tier-l__high` → `tier-l__midhigh` に統合（9→8 選択肢） | `Required(OptionIdsChanged)`（統合を独立理由にしない判断は `rebuild.rs` モジュール doc を参照） |
| P4 表示名・説明のみ | `v1_9.json` | `v4_rename.json` | `tier-xs__low` の `display_name`・`tier-xl__high` の `description` のみ変更（ID・件数は同一） | `NotRequired` |
| P5 判定型変更 | `v1_9.json` | `v5_multi.json` | `judgment_type` を `single_select` → `multi_select` に変更（ID・件数は同一） | 本番の `Definition::load`/`parse` は `multi_select` を `DefinitionError::UnsupportedValue { field: FieldPath::JudgmentType }` で拒否する（`definition.rs` の既存テストで固定済み）。`Required(JudgmentTypeChanged)` の具体値は本番バリアントではなく `crates/core/src/rebuild.rs` の `#[cfg(test)]` 限定の `JudgmentType::TestOnlyAlternate` で確認する（結合テストからは見えない。理由は `rebuild.rs` モジュール doc 2.2 節） |

## データの性質

ラベルは合成（`tier-{xs,s,m,l,xl}__{low,medium,high,midhigh}` の組み合わせ）で、
実データ・個人情報・秘密情報を含まない（証拠種別: テストハーネス）。

## 生成元

本 issue（#91）の実装時に、PoC-19 のパターン説明（追加・削除・統合・表示名変更・
判定型変更）から手作業で作成した合成データ。PoC-19 自体の実データを転記したもの
ではない。
