//! 新旧の定義ファイル（[`crate::definition::Definition`]）を比較し、
//! モデルの作り直しが「必要／不要／変更なし」のどれかを判定する（REQ-20）。
//!
//! # 呼び出し文脈
//!
//! 学習ワーカー層（`crates/train`・`trainer/`。REQ-18〜20）が、定義ファイル
//! の更新を検知した際にモデルの作り直し要否を判断するために使う想定。
//! 本モジュールは共通コア（`fandhe-edge-core`）に置くため、`Definition`・
//! [`crate::canonical::DefinitionIdentity`]・[`crate::canonical::DefinitionHash`]
//! の内部表現に直接アクセスできる（学習ワーカー層は本 crate を経由してのみ
//! これらの型を扱う。層の境界は破らない）。
//!
//! spec の 6 層表（`05-tasks.md`）では TASK-20.x は学習ワーカーの行にあるが、
//! 本モジュールは定義ファイルと正準化ハッシュ（REQ-15・TASK-15.5）を直接
//! 扱うため、issue #89 の判断に従い共通コア（`crates/core/`）に置く。
//!
//! # 出典・移植方針
//!
//! PoC-19（`03-poc/model-lifecycle/scripts/catalog.py` の `need_rebuild`）の
//! 判定規則を移植する（`docs/spec` は参照せず、規則のみを本ファイルへ書き出す。
//! spec-reference の「ビルド独立方針」）。規則は次の 3 つ:
//!
//! 1. 選択肢 ID の集合が変わったら「必要」（追加・削除・統合）
//! 2. `judgment_type` が変わったら「必要」
//! 3. それ以外（表示名・説明だけの差、または差なし）は「不要／変更なし」
//!
//! # 本 issue（#90・TASK-20.1-1）の範囲
//!
//! 判定結果の型と比較関数の骨格のみを実装する。次は範囲外とし、実装済みを
//! 装わない（各担当 TASK で追加する）:
//!
//! - PoC-19 の 5 パターン（追加・削除・統合・判定型変更・rename）の網羅的な
//!   具体値テスト、統合を独立の理由として区別するかの判断（TASK-20.1-2・#91）
//! - `NotRequired` への差分詳細（表示名・説明が変わった選択肢 ID）の追加
//!   （TASK-20.2・#92）
//! - ハッシュ完全一致時に以降の比較処理そのものを実行しないことの保証と
//!   専用テスト（TASK-20.3・#93。本実装は早期 return する骨格のみ）
//! - JSON 入出力契約（`serde::Serialize` の配線・CLI 出力）への接続
//!   （CLI 側 TASK-33.x の対象。本 issue では Rust 型の追加に留める）
//!
//! [`crate::definition::JudgmentType`] は現状 `SingleSelect` の 1 バリアント
//! しか持たないため、「判定型が変わった」経路は分岐として実装するが、現状の
//! 型ではテストで到達させられない（`canonical.rs` 末尾のコメントと同じ制約）。

use crate::canonical::{CanonicalError, DefinitionHash, DefinitionIdentity};
use crate::definition::{Definition, JudgmentType};
use std::collections::BTreeSet;

/// 作り直し判定の結果（REQ-20）。取りうる区分を 3 つの enum バリアントで
/// 固定し、壊れた値（理由が空の「必要」等）を表現できないようにする
/// （coding-rust.md「公開 API・型設計」）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RebuildDecision {
    /// 新旧の定義全体の正準化ハッシュが完全一致した（REQ-20 境界値）。
    /// この場合は同一性の比較を行わずに早期 return する
    /// （比較を実行しないことの機械照合テストは TASK-20.3・#93 が扱う）。
    Unchanged { hash: DefinitionHash },
    /// 同一性（選択肢 ID 集合＋判定型）は変わらず、ハッシュだけが違う
    /// （表示名・説明・`name`・`version` の差）。作り直しは不要。
    /// 差分の詳細（どの選択肢の表示名・説明が変わったか）は
    /// TASK-20.2（#92）で追加する。
    NotRequired,
    /// 作り直しが必要（理由は 1 件以上。空の理由では組み立てられない）。
    Required(RequiredRebuild),
}

/// 作り直しが「必要」と判定された理由の集合。空では組み立てられないことを
/// 型で保証する（[`RequiredRebuild::from_reasons`] のみが構築経路）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequiredRebuild {
    reasons: Vec<RebuildReason>,
}

impl RequiredRebuild {
    /// 理由の一覧を返す。空にはならない（構築経路が空集合を拒否するため）。
    #[must_use]
    pub fn reasons(&self) -> &[RebuildReason] {
        &self.reasons
    }

    /// 理由の `Vec` から組み立てる。空なら `None` を返し、呼び出し側
    /// （[`decide_rebuild`]）はその場合 `NotRequired` へ分岐する。
    /// 外部 crate から空の理由で `RequiredRebuild` を作れないよう、
    /// このコンストラクタは crate 内限定にする。
    pub(crate) fn from_reasons(reasons: Vec<RebuildReason>) -> Option<Self> {
        if reasons.is_empty() {
            None
        } else {
            Some(RequiredRebuild { reasons })
        }
    }
}

/// 作り直しが必要になった個別の理由（PoC-19 の規則 1・2 に対応）。
/// `#[non_exhaustive]` にして、将来 TASK-20.1-2（#91）以降で理由を追加しても
/// 既存の `match` を破壊的変更にしない。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RebuildReason {
    /// 選択肢 ID 集合の変化（追加・削除。統合は「削除＋（場合により）追加」
    /// として現れる。統合を独立の理由として区別するかは TASK-20.1-2（#91）
    /// の対象）。
    OptionIdsChanged {
        added: BTreeSet<String>,
        removed: BTreeSet<String>,
    },
    /// 判定型の変化。
    JudgmentTypeChanged {
        old: JudgmentType,
        new: JudgmentType,
    },
}

/// 新旧定義を比較し、作り直しの要否を判定する（REQ-20・TASK-20.1-1）。
///
/// 処理順:
/// 1. 新旧の定義全体の正準化ハッシュ（[`Definition::canonical_hash`]）が
///    一致すれば、以降の比較をせずに [`RebuildDecision::Unchanged`] を返す
/// 2. 一致しなければ、同一性（[`DefinitionIdentity`]）を比較し、選択肢 ID
///    集合・判定型の差から理由を集める（[`classify_change`]）
/// 3. 理由が 1 件以上なら [`RebuildDecision::Required`]、0 件なら
///    [`RebuildDecision::NotRequired`] を返す
///
/// # エラー
///
/// 正準化ハッシュの計算に失敗した場合（[`crate::canonical::CanonicalError`]）
/// はそのまま伝播する。`Definition` は検証済みの型付きフィールドのみを
/// 持つため実質的には起こらないが、ライブラリコードとして panic させず
/// `Result` を返す（coding-rust.md）。
pub fn decide_rebuild(
    old: &Definition,
    new: &Definition,
) -> Result<RebuildDecision, CanonicalError> {
    let old_hash = old.canonical_hash()?;
    let new_hash = new.canonical_hash()?;

    if old_hash == new_hash {
        return Ok(RebuildDecision::Unchanged { hash: new_hash });
    }

    let reasons = classify_change(&old.identity(), &new.identity());

    match RequiredRebuild::from_reasons(reasons) {
        Some(required) => Ok(RebuildDecision::Required(required)),
        None => Ok(RebuildDecision::NotRequired),
    }
}

/// 新旧の同一性を比較し、作り直しが必要になる理由を集める（内部実装）。
/// 理由の並びは OptionIds → JudgmentType の固定順にし、結果を決定的にする。
fn classify_change(old: &DefinitionIdentity, new: &DefinitionIdentity) -> Vec<RebuildReason> {
    let mut reasons = Vec::new();

    let old_ids = old.option_ids();
    let new_ids = new.option_ids();
    if old_ids != new_ids {
        let added: BTreeSet<String> = new_ids.difference(old_ids).cloned().collect();
        let removed: BTreeSet<String> = old_ids.difference(new_ids).cloned().collect();
        reasons.push(RebuildReason::OptionIdsChanged { added, removed });
    }

    if old.judgment_type() != new.judgment_type() {
        reasons.push(RebuildReason::JudgmentTypeChanged {
            old: old.judgment_type(),
            new: new.judgment_type(),
        });
    }

    reasons
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFINITION_A_JSON: &str = r#"{
        "schema": "fandhe-edge-model-definition/v1",
        "name": "sample_topic",
        "version": 1,
        "judgment_type": "single_select",
        "options": [
            { "id": "yes", "display_name": "Yes", "description": "肯定" },
            { "id": "no", "display_name": "No", "description": "否定" }
        ],
        "io": { "input": "bytes" }
    }"#;

    const DEFINITION_A_HASH_HEX: &str =
        "db372ae2b27530fafc7adbc02daa542a8bc1901d7ce916c2a05552d052aef7ac";

    fn definition_a() -> Definition {
        Definition::parse(DEFINITION_A_JSON).expect("固定 fixture は valid なはず")
    }

    /// AC1: 同一の定義ファイルを 2 つ渡すと `Unchanged` が返り、ハッシュは
    /// `canonical.rs` で独立に確認済みのゴールデン値と一致する
    /// （証拠の種別: テストハーネス。REQ-20・TASK-20.1-1）。
    #[test]
    fn req20_task20_1_1_identical_definitions_are_unchanged() {
        let def_a = definition_a();
        let def_b = definition_a();

        let decision = decide_rebuild(&def_a, &def_b).expect("失敗しないはず");

        match decision {
            RebuildDecision::Unchanged { hash } => {
                assert_eq!(hash.to_hex(), DEFINITION_A_HASH_HEX);
            }
            other => panic!("Unchanged を期待したが {other:?} だった"),
        }
    }

    /// キー順・空白だけが違う JSON でも `Unchanged` になり、ハッシュは同じ値
    /// になる（正準化ハッシュがキー順・空白の揺れを吸収することの確認。
    /// REQ-20・TASK-20.1-1）。
    #[test]
    fn req20_task20_1_1_key_order_whitespace_only_difference_is_unchanged() {
        let reordered = "{\n  \"version\" : 1,\n  \"io\": { \"input\" : \"bytes\" },\n  \"name\": \"sample_topic\",\n  \"schema\": \"fandhe-edge-model-definition/v1\",\n  \"judgment_type\": \"single_select\",\n  \"options\": [\n    { \"description\": \"肯定\", \"id\": \"yes\", \"display_name\": \"Yes\" },\n    { \"display_name\": \"No\", \"description\": \"否定\", \"id\": \"no\" }\n  ]\n}\n";
        let def_a = definition_a();
        let def_reordered = Definition::parse(reordered).expect("valid なはず");

        let decision = decide_rebuild(&def_a, &def_reordered).expect("失敗しないはず");

        match decision {
            RebuildDecision::Unchanged { hash } => {
                assert_eq!(hash.to_hex(), DEFINITION_A_HASH_HEX);
            }
            other => panic!("Unchanged を期待したが {other:?} だった"),
        }
    }

    /// 選択肢を 1 件足すと `Required` が返り、理由は `OptionIdsChanged` の
    /// 1 件のみで `added` に新設した ID が入る（REQ-20・TASK-20.1-1）。
    #[test]
    fn req20_task20_1_1_added_option_id_requires_rebuild() {
        let added_json = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "Yes", "description": "肯定" },
                { "id": "no", "display_name": "No", "description": "否定" },
                { "id": "maybe", "display_name": "Maybe", "description": "保留" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def_a = definition_a();
        let def_added = Definition::parse(added_json).expect("valid なはず");

        let decision = decide_rebuild(&def_a, &def_added).expect("失敗しないはず");

        let expected_added: BTreeSet<String> = ["maybe".to_string()].into_iter().collect();
        let expected_removed: BTreeSet<String> = BTreeSet::new();
        match decision {
            RebuildDecision::Required(required) => {
                assert_eq!(
                    required.reasons(),
                    &[RebuildReason::OptionIdsChanged {
                        added: expected_added,
                        removed: expected_removed,
                    }]
                );
            }
            other => panic!("Required を期待したが {other:?} だった"),
        }
    }

    /// 表示名だけを変えると `NotRequired` が返る（分岐の骨格のみを確認する。
    /// 差分の詳細検証は TASK-20.2・#92 の対象。REQ-20・TASK-20.1-1）。
    #[test]
    fn req20_task20_1_1_display_name_only_change_is_not_required() {
        let display_changed = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "はい", "description": "肯定" },
                { "id": "no", "display_name": "No", "description": "否定" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def_a = definition_a();
        let def_changed = Definition::parse(display_changed).expect("valid なはず");

        let decision = decide_rebuild(&def_a, &def_changed).expect("失敗しないはず");

        assert_eq!(decision, RebuildDecision::NotRequired);
    }

    // `JudgmentType` は現状 `SingleSelect` の 1 バリアントしか持たないため、
    // `JudgmentTypeChanged` 経路（clippy に「常に偽」と判定されない代入元が
    // 存在しない状態）はテストで到達させられない
    // （`canonical.rs` 末尾のコメントと同じ制約。TASK-20.1 の範囲で
    // バリアントが増えた時点でテストを追加する）。
}
