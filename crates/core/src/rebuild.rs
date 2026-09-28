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
//! # TASK-20.1-2（issue #91）で追加した内容
//!
//! #90（TASK-20.1-1）が残した「PoC-19 の 5 パターンの網羅的な具体値テスト」
//! 「統合を独立の理由として区別するかの判断」の 2 点を本 issue で解決する。
//!
//! ## 2.1 統合は独立の理由にしない（`OptionIdsChanged` のまま）
//!
//! 本番の定義ファイルスキーマ（`fandhe-edge-model-definition/v1`。
//! [`crate::definition::RawDefinition`] は `deny_unknown_fields`）には
//! PoC-19 の `merges` 欄が無い。`merges` の追加は定義ファイルスキーマの変更
//! （ユーザー承認事項）にあたるため行わない。ID 集合（削除 2・追加 1 等）
//! から統合を推定するのも不確実（「削除＋無関係な追加」と区別できない）。
//! よって統合パターン（fixture: `v3_8merge.json`）は
//! `OptionIdsChanged { added, removed }` の `added`・`removed` が両方
//! 非空の形で現れることをテストで固定する。`RebuildReason` は
//! `#[non_exhaustive]` のため、将来スキーマに `merges` 相当の情報が入った
//! 時点で独立の理由を追加できる。
//!
//! ## 2.2 判定型変更（P5）の扱い
//!
//! 本番の [`crate::definition::JudgmentType`] は `SingleSelect` の 1
//! バリアントのみで、`Definition::parse`/`load` は `"multi_select"` を
//! `DefinitionError::UnsupportedValue { field: FieldPath::JudgmentType }`
//! で拒否する（`definition.rs` の既存テストで固定済み）。本番バリアントへ
//! `multi_select` 等を追加することは判定型スキーマの変更（NR-2 は REQ-15
//! 対象外）であり、ユーザー承認事項として本 issue では行わない。
//!
//! 代わりに `JudgmentType` の `#[cfg(test)]` 限定バリアント
//! `TestOnlyAlternate` を経路確認の seam として使う（`definition.rs` の
//! doc を参照）。`#[cfg(test)]` は本 crate のユニットテストビルドにしか
//! 効かず、依存 crate（`crates/train`・`crates/data`・`crates/eval`・
//! `crates/cli` 等）の非 test ビルドには現れないため、それらの網羅
//! `match`（例: `crates/train/src/request.rs` の判定型分岐）に影響しない。
//! したがって `Required(JudgmentTypeChanged)` の具体値テストは本ファイルの
//! `mod tests`（unit test）に置く（`crates/core/tests/` の結合テストからは
//! この seam は見えない）。結合テスト側では PoC-19 の P5 相当の入力
//! （`judgment_type: "multi_select"`）が `Definition::load` の時点で拒否
//! されることを確認し、本番の観測挙動（学習へ進まない＝PoC-19 の
//! 「必要（案内のみ）」と整合）を示す。
//!
//! ## 2.3 PoC との意図的な差
//!
//! PoC-19 の `need_rebuild` は選択肢 ID 集合の変化を見つけた時点で早期
//! return する（理由は常に 1 件）。本モジュールの `classify_change` は
//! ID 集合 → 判定型の固定順で両方の理由を集める（理由が複数件になりうる）。
//! PoC-19 の 5 パターンはいずれか一方しか変えないため、結果には影響しない。
//!
//! # TASK-20.2（issue #92）で追加した内容
//!
//! `RebuildDecision::NotRequired` に差分詳細（[`NotRequiredRebuild`]）を追加した。
//! PoC-19（`need_rebuild`）は表示名・説明の変化を見つけた時点で「変更あり」の
//! 選択肢 ID を返すが、変化が無ければ何も返さず「変更なし」を表す
//! （本モジュールの `NotRequired` 1 本と等価）。本実装は PoC と異なり、
//! 表示名・説明どちらの変化かを型で区別した 2 つの ID 集合として返す
//! （出典: `fixtures/rebuild/poc19/PROVENANCE.md`）。
//!
//! 保持するのは選択肢 ID のみで、旧・新の表示名・説明の文字列は持たない
//! （データ本文をログ等へ持ち出さない方針。security.md「秘密情報の混入防止」）。
//! `name`・`version`・選択肢の宣言順のみが原因で `NotRequired` になる場合は
//! 両集合が空になる（`classify_display_only_change` の doc を参照）。
//!
//! ## 破壊的変更（BREAKING CHANGE）
//!
//! `RebuildDecision::NotRequired` は TASK-20.1-1（#237）時点では単位
//! バリアント（`NotRequired,`）だったが、本 TASK-20.2 で
//! `NotRequired(NotRequiredRebuild)` へ変更した。`RebuildDecision` に
//! `#[non_exhaustive]` は付けていない（3 バリアントで固定する設計意図。
//! 本ファイル冒頭のドキュメンテーションコメントを参照）ため、この型を
//! 網羅的にパターンマッチしていた既存呼び出し元は本変更でコンパイル不能に
//! なる。移行は `RebuildDecision::NotRequired(_)`（詳細が不要な場合）また
//! は `RebuildDecision::NotRequired(detail)`（[`NotRequiredRebuild`] の
//! アクセサ `display_name_changed`/`description_changed` を使う場合）へ
//! パターンを書き換える。
//!
//! # 引き続き範囲外（各担当 TASK で追加する）
//!
//! - ハッシュ完全一致時に以降の比較処理そのものを実行しないことの保証と
//!   専用テスト（TASK-20.3・#93。本実装は早期 return する骨格のみ）
//! - JSON 入出力契約（`serde::Serialize` の配線・CLI 出力）への接続
//!   （CLI 側 TASK-33.x の対象。本 issue では Rust 型の追加に留める）
//!
//! 判定型変更（`JudgmentTypeChanged`）経路のテスト方針は上記 2.2 節を参照。

use crate::canonical::{CanonicalError, DefinitionHash, DefinitionIdentity};
use crate::definition::{Choice, Definition, JudgmentType};
use std::collections::{BTreeMap, BTreeSet};

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
    /// （表示名・説明・`name`・`version`・選択肢の宣言順の差）。作り直しは
    /// 不要。差分の詳細は [`NotRequiredRebuild`] を参照（TASK-20.2・#92）。
    NotRequired(NotRequiredRebuild),
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

/// 作り直し「不要」の詳細（REQ-20 異常系・TASK-20.2・issue #92）。
///
/// 表示名が変わった選択肢 ID・説明が変わった選択肢 ID をそれぞれ集合で持つ
/// （同一 ID が両方に入ることもある）。保持するのは ID のみで、旧・新の
/// 表示名・説明の文字列は持たない（データ本文を持ち出さない方針。
/// security.md「秘密情報の混入防止」）。
///
/// 両集合が空になることもある。ハッシュ不一致（[`RebuildDecision::NotRequired`]
/// に分岐する時点でハッシュは必ず不一致）かつ同一性一致は、`name`・
/// `version`・選択肢の宣言順のみの差でも起こる（`canonical.rs` の
/// `req15_task15_5_name_or_version_change_keeps_identity`・
/// `req15_task15_5_option_order_change_same_identity_different_hash` で固定済み。
/// PoC-19 も同ケースを「変更なし」として扱う）。その場合は両集合を空にする
/// （fail-closed の `Required` 理由やエラー経路を新設しない）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotRequiredRebuild {
    display_name_changed: BTreeSet<String>,
    description_changed: BTreeSet<String>,
}

impl NotRequiredRebuild {
    /// crate 内限定の構築経路（[`classify_display_only_change`] のみが呼ぶ）。
    /// 外部 crate が任意内容の「不要」判定を偽造できないようにする
    /// （`RequiredRebuild::from_reasons` と同じ方針）。
    pub(crate) fn new(
        display_name_changed: BTreeSet<String>,
        description_changed: BTreeSet<String>,
    ) -> Self {
        Self {
            display_name_changed,
            description_changed,
        }
    }

    /// 表示名が変わった選択肢 ID の集合（辞書順で決定的に走査できる）。
    #[must_use]
    pub fn display_name_changed(&self) -> &BTreeSet<String> {
        &self.display_name_changed
    }

    /// 説明が変わった選択肢 ID の集合（辞書順で決定的に走査できる）。
    #[must_use]
    pub fn description_changed(&self) -> &BTreeSet<String> {
        &self.description_changed
    }
}

/// 作り直しが必要になった個別の理由（PoC-19 の規則 1・2 に対応）。
/// `#[non_exhaustive]` にして、将来 TASK-20.1-2（#91）以降で理由を追加しても
/// 既存の `match` を破壊的変更にしない。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RebuildReason {
    /// 選択肢 ID 集合の変化（追加・削除）。統合は `added`・`removed` の
    /// 双方が非空の値として現れる（独立の理由にしない判断はモジュール doc
    /// 2.1 節・TASK-20.1-2・#91 を参照）。
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
///
/// # 呼び出し例
///
/// 学習ワーカー層（`trainer/`・`crates/train/`）が定義ファイルの更新を検知した際、
/// 新旧の [`Definition`] を読み込んで `decide_rebuild` に渡し、3 通りの判定結果を
/// 次のように扱う想定（TASK-20.1-1 の範囲は型と比較骨格のみで、学習ワーカー側の
/// 実際の呼び出し配線は REQ-18〜20 の後続 TASK で行う）。
///
/// ```
/// use fandhe_edge_core::definition::Definition;
/// use fandhe_edge_core::rebuild::{decide_rebuild, RebuildDecision};
///
/// let old = Definition::parse(
///     r#"{
///         "schema": "fandhe-edge-model-definition/v1",
///         "name": "sample_topic",
///         "version": 1,
///         "judgment_type": "single_select",
///         "options": [
///             { "id": "yes", "display_name": "Yes", "description": "肯定" },
///             { "id": "no", "display_name": "No", "description": "否定" }
///         ],
///         "io": { "input": "bytes" }
///     }"#,
/// ).expect("valid な定義ファイルのはず");
/// let new = Definition::parse(
///     r#"{
///         "schema": "fandhe-edge-model-definition/v1",
///         "name": "sample_topic",
///         "version": 1,
///         "judgment_type": "single_select",
///         "options": [
///             { "id": "yes", "display_name": "Yes", "description": "肯定" },
///             { "id": "no", "display_name": "No", "description": "否定" },
///             { "id": "maybe", "display_name": "Maybe", "description": "保留" }
///         ],
///         "io": { "input": "bytes" }
///     }"#,
/// ).expect("valid な定義ファイルのはず");
///
/// match decide_rebuild(&old, &new).expect("失敗しないはず") {
///     // ハッシュ完全一致。定義ファイルに実質差分がなく、作り直し不要。
///     RebuildDecision::Unchanged { hash } => {
///         println!("unchanged: {}", hash.to_hex());
///     }
///     // 選択肢 ID 集合・判定型は変わらず、表示名等の差分のみ。作り直し不要。
///     RebuildDecision::NotRequired(not_required) => {
///         println!("display name changed: {:?}", not_required.display_name_changed());
///     }
///     // 選択肢 ID 集合または判定型が変化。モデルの作り直しが必要。
///     RebuildDecision::Required(required) => {
///         for reason in required.reasons() {
///             println!("rebuild reason: {reason:?}");
///         }
///     }
/// }
/// ```
///
/// 上記の例では選択肢が 1 件追加されているため `Required` に分岐し、
/// `reasons()` から `RebuildReason::OptionIdsChanged { added: {"maybe"}, .. }` が
/// 得られる。
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
        None => Ok(RebuildDecision::NotRequired(classify_display_only_change(
            old, new,
        ))),
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

/// 同一性一致（`classify_change` の理由が空）と分かった後にのみ呼ばれ、
/// 表示名・説明のみの差分を選択肢 ID 単位で集める（TASK-20.2・issue #92）。
///
/// `old`・`new` は選択肢 ID 集合が同一であることが呼び出し前提
/// （[`decide_rebuild`] の分岐順）。`&Definition` を受けるのは
/// `DefinitionIdentity` が表示名・説明を保持しないため。添字アクセス
/// （`[]`）・`unwrap`/`expect` は使わず `get()` で処理し、万一 ID が
/// 見つからない場合も panic せず「差分なし」として扱う
/// （coding-rust.md「外部入力」。同一性一致後のため実際には起こらない）。
///
/// `name`・`version`・選択肢の宣言順のみの差では両集合が空になる
/// （[`NotRequiredRebuild`] の doc を参照）。
fn classify_display_only_change(old: &Definition, new: &Definition) -> NotRequiredRebuild {
    let old_by_id: BTreeMap<&str, &Choice> =
        old.options().iter().map(|c| (c.id.as_str(), c)).collect();
    let new_by_id: BTreeMap<&str, &Choice> =
        new.options().iter().map(|c| (c.id.as_str(), c)).collect();

    let mut display_name_changed = BTreeSet::new();
    let mut description_changed = BTreeSet::new();

    for (id, old_choice) in &old_by_id {
        let Some(new_choice) = new_by_id.get(id) else {
            continue;
        };
        if old_choice.display_name != new_choice.display_name {
            display_name_changed.insert((*id).to_string());
        }
        if old_choice.description != new_choice.description {
            description_changed.insert((*id).to_string());
        }
    }

    NotRequiredRebuild::new(display_name_changed, description_changed)
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

    /// 表示名だけを変えると `NotRequired` が返り、`display_name_changed` に
    /// 変更した選択肢 ID が、`description_changed` は空集合になる
    /// （REQ-20・TASK-20.1-1・差分詳細は TASK-20.2・#92）。
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

        let expected_display_name_changed: BTreeSet<String> =
            ["yes".to_string()].into_iter().collect();
        match decision {
            RebuildDecision::NotRequired(not_required) => {
                assert_eq!(
                    not_required.display_name_changed(),
                    &expected_display_name_changed
                );
                assert_eq!(not_required.description_changed(), &BTreeSet::new());
            }
            other => panic!("NotRequired を期待したが {other:?} だった"),
        }
    }

    // PoC-19（`03-poc/model-lifecycle/`）の事前固定 5 パターンを本番の定義
    // ファイルスキーマへ変換した fixture（出典・変換規則は
    // `fixtures/rebuild/poc19/PROVENANCE.md` を参照。REQ-20・TASK-20.1-2・
    // issue #91）。ビルド時にコンパイル済みバイナリへ埋め込む
    // （`include_str!` はコンパイル時にのみファイルシステムへアクセスし、
    // 実行時の I/O ではないため資源上限〔REQ-39〕の対象外）。

    const POC19_V1_9: &str = include_str!("../../../fixtures/rebuild/poc19/v1_9.json");
    const POC19_V2_8RM: &str = include_str!("../../../fixtures/rebuild/poc19/v2_8rm.json");
    const POC19_V3_8MERGE: &str = include_str!("../../../fixtures/rebuild/poc19/v3_8merge.json");
    const POC19_V4_RENAME: &str = include_str!("../../../fixtures/rebuild/poc19/v4_rename.json");

    /// P1（追加）: `v2_8rm.json` → `v1_9.json` で `tier-xl__high` が追加され、
    /// `Required` の理由が `OptionIdsChanged` の 1 件のみになる
    /// （証拠種別: テストハーネス。出典: `fixtures/rebuild/poc19/PROVENANCE.md`。
    /// REQ-20・TASK-20.1-2・issue #91）。
    #[test]
    fn req20_task20_1_2_poc19_p1_added_option_requires_rebuild() {
        let old = Definition::parse(POC19_V2_8RM).expect("固定 fixture は valid なはず");
        let new = Definition::parse(POC19_V1_9).expect("固定 fixture は valid なはず");

        let decision = decide_rebuild(&old, &new).expect("失敗しないはず");

        let expected_added: BTreeSet<String> = ["tier-xl__high".to_string()].into_iter().collect();
        match decision {
            RebuildDecision::Required(required) => {
                assert_eq!(
                    required.reasons(),
                    &[RebuildReason::OptionIdsChanged {
                        added: expected_added,
                        removed: BTreeSet::new(),
                    }]
                );
            }
            other => panic!("Required を期待したが {other:?} だった"),
        }
    }

    /// P2（削除）: `v1_9.json` → `v2_8rm.json` で `tier-xl__high` が削除され、
    /// `Required` の理由が `OptionIdsChanged` の 1 件のみになる
    /// （証拠種別: テストハーネス。REQ-20・TASK-20.1-2・issue #91）。
    #[test]
    fn req20_task20_1_2_poc19_p2_removed_option_requires_rebuild() {
        let old = Definition::parse(POC19_V1_9).expect("固定 fixture は valid なはず");
        let new = Definition::parse(POC19_V2_8RM).expect("固定 fixture は valid なはず");

        let decision = decide_rebuild(&old, &new).expect("失敗しないはず");

        let expected_removed: BTreeSet<String> =
            ["tier-xl__high".to_string()].into_iter().collect();
        match decision {
            RebuildDecision::Required(required) => {
                assert_eq!(
                    required.reasons(),
                    &[RebuildReason::OptionIdsChanged {
                        added: BTreeSet::new(),
                        removed: expected_removed,
                    }]
                );
            }
            other => panic!("Required を期待したが {other:?} だった"),
        }
    }

    /// P3（統合）: `v1_9.json` → `v3_8merge.json` で `tier-l__medium` +
    /// `tier-l__high` が `tier-l__midhigh` へ統合され、`added`・`removed` の
    /// 双方が非空の `OptionIdsChanged` 1 件になる（統合を独立の理由に
    /// しない判断はモジュール doc 2.1 節を参照。REQ-20・TASK-20.1-2・
    /// issue #91）。
    #[test]
    fn req20_task20_1_2_poc19_p3_merged_options_require_rebuild() {
        let old = Definition::parse(POC19_V1_9).expect("固定 fixture は valid なはず");
        let new = Definition::parse(POC19_V3_8MERGE).expect("固定 fixture は valid なはず");

        let decision = decide_rebuild(&old, &new).expect("失敗しないはず");

        let expected_added: BTreeSet<String> =
            ["tier-l__midhigh".to_string()].into_iter().collect();
        let expected_removed: BTreeSet<String> =
            ["tier-l__medium".to_string(), "tier-l__high".to_string()]
                .into_iter()
                .collect();
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

    /// P4（表示名・説明のみ）: `v1_9.json` → `v4_rename.json` は選択肢 ID
    /// 集合・判定型が同一で、`tier-xs__low` の表示名・`tier-xl__high` の
    /// 説明のみが変わる（出典: `fixtures/rebuild/poc19/PROVENANCE.md`）。
    /// `NotRequired` になり、差分の選択肢 ID が具体値で一致する
    /// （REQ-20 異常系・TASK-20.1-2・issue #91・TASK-20.2・issue #92）。
    #[test]
    fn req20_task20_1_2_poc19_p4_display_only_change_is_not_required() {
        let old = Definition::parse(POC19_V1_9).expect("固定 fixture は valid なはず");
        let new = Definition::parse(POC19_V4_RENAME).expect("固定 fixture は valid なはず");

        let decision = decide_rebuild(&old, &new).expect("失敗しないはず");

        let expected_display_name_changed: BTreeSet<String> =
            ["tier-xs__low".to_string()].into_iter().collect();
        let expected_description_changed: BTreeSet<String> =
            ["tier-xl__high".to_string()].into_iter().collect();
        match decision {
            RebuildDecision::NotRequired(not_required) => {
                assert_eq!(
                    not_required.display_name_changed(),
                    &expected_display_name_changed
                );
                assert_eq!(
                    not_required.description_changed(),
                    &expected_description_changed
                );
            }
            other => panic!("NotRequired を期待したが {other:?} だった"),
        }
    }

    /// P5（判定型変更）: 選択肢 ID 集合を変えず `judgment_type` だけを
    /// 変更すると、`Required` の理由が `JudgmentTypeChanged` の 1 件のみに
    /// なる。本番の `JudgmentType` は `SingleSelect` の 1 バリアントしか
    /// 持たないため、`#[cfg(test)]` 限定の `TestOnlyAlternate`（モジュール
    /// doc 2.2 節）を経路確認の seam として使う。この seam は本 crate の
    /// ユニットテストビルドにしか存在せず、`crates/core/tests/` の結合
    /// テストからは見えない（PoC-19 の P5 相当・本番の観測挙動〔`multi_select`
    /// の拒否〕は `crates/core/tests/rebuild_decision.rs` で確認する。
    /// REQ-20・TASK-20.1-2・issue #91）。
    #[test]
    fn req20_task20_1_2_poc19_p5_judgment_type_change_requires_rebuild() {
        let mut value: serde_json::Value =
            serde_json::from_str(POC19_V1_9).expect("固定 fixture は valid JSON のはず");
        value["judgment_type"] = serde_json::json!("test_only_alternate");
        let new_text = serde_json::to_string(&value).expect("Value の再直列化は失敗しないはず");

        let old = Definition::parse(POC19_V1_9).expect("固定 fixture は valid なはず");
        let new = Definition::parse(&new_text)
            .expect("test_only_alternate は test cfg で有効な判定型のはず");

        let decision = decide_rebuild(&old, &new).expect("失敗しないはず");

        match decision {
            RebuildDecision::Required(required) => {
                assert_eq!(
                    required.reasons(),
                    &[RebuildReason::JudgmentTypeChanged {
                        old: JudgmentType::SingleSelect,
                        new: JudgmentType::TestOnlyAlternate,
                    }]
                );
            }
            other => panic!("Required を期待したが {other:?} だった"),
        }
    }

    /// PoC-19 の 5 パターンを 1 本で集計し、P1・P2・P3・P5 の 4 件が
    /// `Required`、P4 の 1 件が `NotRequired` になることを具体値で確認する
    /// （受入基準: PoC-19 の 5 パターンのうち 4 パターンで「作り直しが必要」
    /// と判定されること。REQ-20・TASK-20.1-2・issue #91）。
    #[test]
    fn req20_task20_1_2_poc19_five_patterns_four_required_one_not_required() {
        let v1 = Definition::parse(POC19_V1_9).expect("固定 fixture は valid なはず");
        let v2 = Definition::parse(POC19_V2_8RM).expect("固定 fixture は valid なはず");
        let v3 = Definition::parse(POC19_V3_8MERGE).expect("固定 fixture は valid なはず");
        let v4 = Definition::parse(POC19_V4_RENAME).expect("固定 fixture は valid なはず");
        let mut v5_value: serde_json::Value =
            serde_json::from_str(POC19_V1_9).expect("固定 fixture は valid JSON のはず");
        v5_value["judgment_type"] = serde_json::json!("test_only_alternate");
        let v5_text = serde_json::to_string(&v5_value).expect("Value の再直列化は失敗しないはず");
        let v5 = Definition::parse(&v5_text).expect("test cfg で有効な判定型のはず");

        let decisions = [
            decide_rebuild(&v2, &v1).expect("P1 は失敗しないはず"),
            decide_rebuild(&v1, &v2).expect("P2 は失敗しないはず"),
            decide_rebuild(&v1, &v3).expect("P3 は失敗しないはず"),
            decide_rebuild(&v1, &v4).expect("P4 は失敗しないはず"),
            decide_rebuild(&v1, &v5).expect("P5 は失敗しないはず"),
        ];

        let required_count = decisions
            .iter()
            .filter(|d| matches!(d, RebuildDecision::Required(_)))
            .count();
        let not_required_count = decisions
            .iter()
            .filter(|d| matches!(d, RebuildDecision::NotRequired(_)))
            .count();

        assert_eq!(required_count, 4);
        assert_eq!(not_required_count, 1);
    }

    /// 説明のみを変えると `NotRequired` が返り、`description_changed` に
    /// 変更した選択肢 ID が、`display_name_changed` は空集合になる
    /// （REQ-20 異常系・TASK-20.2・issue #92）。
    #[test]
    fn req20_task20_2_description_only_change_is_not_required() {
        let description_changed = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "Yes", "description": "肯定" },
                { "id": "no", "display_name": "No", "description": "いいえ" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def_a = definition_a();
        let def_changed = Definition::parse(description_changed).expect("valid なはず");

        let decision = decide_rebuild(&def_a, &def_changed).expect("失敗しないはず");

        let expected_description_changed: BTreeSet<String> =
            ["no".to_string()].into_iter().collect();
        match decision {
            RebuildDecision::NotRequired(not_required) => {
                assert_eq!(not_required.display_name_changed(), &BTreeSet::new());
                assert_eq!(
                    not_required.description_changed(),
                    &expected_description_changed
                );
            }
            other => panic!("NotRequired を期待したが {other:?} だった"),
        }
    }

    /// 同一 ID の表示名・説明を両方変更すると、両集合に同じ ID が入る
    /// （REQ-20 異常系・TASK-20.2・issue #92）。
    #[test]
    fn req20_task20_2_both_display_name_and_description_changed_for_same_id() {
        let both_changed = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "はい", "description": "肯定的" },
                { "id": "no", "display_name": "No", "description": "否定" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def_a = definition_a();
        let def_changed = Definition::parse(both_changed).expect("valid なはず");

        let decision = decide_rebuild(&def_a, &def_changed).expect("失敗しないはず");

        let expected: BTreeSet<String> = ["yes".to_string()].into_iter().collect();
        match decision {
            RebuildDecision::NotRequired(not_required) => {
                assert_eq!(not_required.display_name_changed(), &expected);
                assert_eq!(not_required.description_changed(), &expected);
            }
            other => panic!("NotRequired を期待したが {other:?} だった"),
        }
    }

    /// 複数 ID の表示名変更が辞書順（`BTreeSet`）で決定的に得られる
    /// （REQ-20 異常系・TASK-20.2・issue #92。coding-rust.md「数値・決定性」）。
    #[test]
    fn req20_task20_2_multiple_display_name_changes_are_sorted_deterministically() {
        let multi_changed = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "はい", "description": "肯定" },
                { "id": "no", "display_name": "いいえ", "description": "否定" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def_a = definition_a();
        let def_changed = Definition::parse(multi_changed).expect("valid なはず");

        let decision = decide_rebuild(&def_a, &def_changed).expect("失敗しないはず");

        let expected: BTreeSet<String> =
            ["no".to_string(), "yes".to_string()].into_iter().collect();
        match decision {
            RebuildDecision::NotRequired(not_required) => {
                assert_eq!(not_required.display_name_changed(), &expected);
                assert_eq!(
                    not_required
                        .display_name_changed()
                        .iter()
                        .collect::<Vec<_>>(),
                    vec!["no", "yes"]
                );
            }
            other => panic!("NotRequired を期待したが {other:?} だった"),
        }
    }

    /// `version` のみの変更（選択肢は不変）はハッシュ不一致・同一性一致の
    /// 境界にあたり、`NotRequired` の両集合が空になる（`canonical.rs` の
    /// `req15_task15_5_name_or_version_change_keeps_identity` と同じ境界。
    /// REQ-20 異常系・TASK-20.2・issue #92）。
    #[test]
    fn req20_task20_2_version_only_change_has_empty_diff_sets() {
        let version_changed = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 2,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "Yes", "description": "肯定" },
                { "id": "no", "display_name": "No", "description": "否定" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def_a = definition_a();
        let def_changed = Definition::parse(version_changed).expect("valid なはず");

        let decision = decide_rebuild(&def_a, &def_changed).expect("失敗しないはず");

        match decision {
            RebuildDecision::NotRequired(not_required) => {
                assert_eq!(not_required.display_name_changed(), &BTreeSet::new());
                assert_eq!(not_required.description_changed(), &BTreeSet::new());
            }
            other => panic!("NotRequired を期待したが {other:?} だった"),
        }
    }

    /// 選択肢の宣言順のみの変更（ID・表示名・説明は不変）は `NotRequired` の
    /// 両集合が空になる（現行挙動の固定。宣言順が評価契約の majority
    /// タイブレークに使われる点との整合は spec レベルの論点であり、本
    /// テストでは挙動を変えず固定するに留める。REQ-20 異常系・TASK-20.2・
    /// issue #92）。
    #[test]
    fn req20_task20_2_option_order_only_change_has_empty_diff_sets() {
        let reordered = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "no", "display_name": "No", "description": "否定" },
                { "id": "yes", "display_name": "Yes", "description": "肯定" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def_a = definition_a();
        let def_reordered = Definition::parse(reordered).expect("valid なはず");

        let decision = decide_rebuild(&def_a, &def_reordered).expect("失敗しないはず");

        match decision {
            RebuildDecision::NotRequired(not_required) => {
                assert_eq!(not_required.display_name_changed(), &BTreeSet::new());
                assert_eq!(not_required.description_changed(), &BTreeSet::new());
            }
            other => panic!("NotRequired を期待したが {other:?} だった"),
        }
    }

    /// 表示名変更と ID 追加が同時に起きた場合は `Required(OptionIdsChanged)`
    /// のまま（不要判定は同一性一致時のみ走ることの確認。表示名の差分は
    /// 集計されない。REQ-20 異常系・TASK-20.2・issue #92）。
    #[test]
    fn req20_task20_2_display_name_change_with_added_option_stays_required() {
        let added_and_renamed = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "はい", "description": "肯定" },
                { "id": "no", "display_name": "No", "description": "否定" },
                { "id": "maybe", "display_name": "Maybe", "description": "保留" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def_a = definition_a();
        let def_changed = Definition::parse(added_and_renamed).expect("valid なはず");

        let decision = decide_rebuild(&def_a, &def_changed).expect("失敗しないはず");

        let expected_added: BTreeSet<String> = ["maybe".to_string()].into_iter().collect();
        match decision {
            RebuildDecision::Required(required) => {
                assert_eq!(
                    required.reasons(),
                    &[RebuildReason::OptionIdsChanged {
                        added: expected_added,
                        removed: BTreeSet::new(),
                    }]
                );
            }
            other => panic!("Required を期待したが {other:?} だった"),
        }
    }
}
