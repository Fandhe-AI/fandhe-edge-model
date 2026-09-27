//! 定義ファイル（[`crate::definition::Definition`]）の正準化と、それに基づく
//! 2 種類の比較用の値を提供する（REQ-15・TASK-15.5）。
//!
//! # 2 つの型の違い（棚卸し追記 2026-09-27。issue #36）
//!
//! - [`DefinitionIdentity`]: 選択肢 ID の集合と `judgment_type` だけから作る
//!   **同一性**。表示名・説明・`name`・`version`・選択肢の宣言順は含めない。
//!   TASK-20.2（#92）の「表示名・説明だけの変更は作り直し不要」判定はこちらを使う
//! - [`DefinitionHash`]: 表示名・説明を含む定義全体を正準化した sha256。
//!   記録・改ざんの検知、TASK-20.3（#93）の「定義全体のハッシュが完全一致したら
//!   変更なし」判定に使う
//!
//! 2 つを別の型にしているのは、「同一性が同じ」と「ハッシュが完全一致」を
//! 取り違えて比べる誤用をコンパイルエラーで防ぐため（`PartialEq` は同じ型
//! 同士でしか比較できない）。真偽値だけを返す `is_same(...)` は作らない
//! （`.claude/rules/coding-rust.md`「公開 API・型設計」）。
//!
//! [`crate::hash`]（PR #190。評価データの凍結〔REQ-17〕向けの生バイト列の
//! sha256）とは用途が異なる。あちらは既に読み込んだバイト列をそのまま
//! ハッシュするが、こちらは `Definition` を正準化 JSON へ変換してからハッシュ
//! する。両者は独立したユーティリティで、片方をもう片方に依存させない。
//!
//! # 正準化の規則（1 箇所に集約。REQ-15）
//!
//! PoC-19（`03-poc/model-lifecycle/scripts/catalog.py`）の `canon_hash` と
//! 同じバイト列になるようにする。あちらは Python の
//! `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
//! を使うため、ここでも次を満たす:
//!
//! - オブジェクト: キーを **明示的に** バイト順（＝ UTF-8 のバイト順で、
//!   Unicode 符号位置順・Python の `sort_keys` と一致する）でソートしてから
//!   `{"k":v,...}` の形で書く。区切りは `,` と `:` のみで空白を入れない。
//!   `serde_json::Map` は既定で `BTreeMap`（挿入順ではなくキー順）だが、
//!   workspace 内のどこかの crate が `preserve_order` feature を有効にすると
//!   （Cargo の feature unification で workspace 全体に効く）挿入順の
//!   `IndexMap` に変わる。この落とし穴に暗黙に頼らず、ここで明示的にソートする
//! - 配列: 要素の順序を保つ。`options` の宣言順は PoC-9 追補 A-10
//!   （`docs/spec/03-poc/evaluation-contract/README.md`）の majority
//!   タイブレークに効くため意味を持つ設計判断であり、選択肢を並べ替えると
//!   `DefinitionHash` は変わる（`DefinitionIdentity` は変わらない）
//! - 文字列: `serde_json::to_string(&str)` でエスケープする（`"`・`\`・制御
//!   文字のみをエスケープし、非 ASCII は UTF-8 のまま。Python の
//!   `ensure_ascii=False` と同じ）
//! - 数値: 整数はそのまま 10 進で書く。浮動小数は現状の `Definition` に存在
//!   しないため `CanonicalError::UnsupportedNumber` で拒否する（fail-closed。
//!   数値表現の揺れを持ち込まない）
//! - 真偽値・null: `true`・`false`・`null`
//!
//! `Definition` の型付きフィールドへ一度パースしてから `serde_json::to_value`
//! で再シリアライズするため、キー順・空白・数値表現（`1` と `1.0` と `"1"` の
//! 混在等）の揺れは `Definition::parse` の時点で吸収済みになる。
//!
//! # 将来の拡張
//!
//! `Definition::io`（[`crate::definition::IoSchema`]）は現状 `Bytes` の
//! 1 バリアントしかないため同一性に含めていない。将来 `io` にバリアントが
//! 増えたら、同一性に含めるかどうかを改めて検討する（本 TASK では範囲を
//! 広げない）。

use crate::definition::{Definition, JudgmentType};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::fmt;

/// 定義の**同一性**（選択肢 ID の集合＋判定型）。表示名・説明・`name`・
/// `version`・選択肢の宣言順は含めない（境界値。REQ-15）。
///
/// 検証済みの [`Definition`] からしか作れない（[`Definition::identity`]
/// 経由に限定し、フィールドは非公開にする）。フィールドを公開すると
/// 別 crate が任意の ID 集合から `DefinitionIdentity` を組み立てられて
/// しまい、「検証済みの定義から作った同一性」という保証が崩れる。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefinitionIdentity {
    judgment_type: JudgmentType,
    option_ids: BTreeSet<String>,
}

impl DefinitionIdentity {
    /// 定義の判定型。
    #[must_use]
    pub fn judgment_type(&self) -> JudgmentType {
        self.judgment_type
    }

    /// 選択肢 ID の集合（辞書順で決定的に走査できる `BTreeSet`。`HashSet` は
    /// 走査順がプロセスごとに変わりうるため使わない）。
    #[must_use]
    pub fn option_ids(&self) -> &BTreeSet<String> {
        &self.option_ids
    }

    /// [`Definition`] から同一性を組み立てる（`parse` 済みであることが前提。
    /// 検証を経ていない値からは作れない）。
    pub(crate) fn from_definition(definition: &Definition) -> Self {
        DefinitionIdentity {
            judgment_type: definition.judgment_type(),
            option_ids: definition
                .options()
                .iter()
                .map(|choice| choice.id.clone())
                .collect(),
        }
    }
}

/// 定義全体（表示名・説明を含む）を正準化した JSON の sha256（32 バイト）。
///
/// `Display`・[`DefinitionHash::to_hex`] は小文字 16 進 64 桁を返す。
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefinitionHash([u8; 32]);

impl DefinitionHash {
    /// 正準化 JSON の UTF-8 バイト列から sha256 を計算する
    /// （[`Definition::canonical_hash`](crate::definition::Definition::canonical_hash)
    /// の内部実装）。
    pub(crate) fn from_json_bytes(bytes: &[u8]) -> Self {
        DefinitionHash(sha256_hex_bytes(bytes))
    }

    /// 生の 32 バイトを返す。
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// 小文字 16 進 64 桁の文字列表現。
    #[must_use]
    pub fn to_hex(&self) -> String {
        use fmt::Write as _;
        let mut out = String::with_capacity(self.0.len() * 2);
        for byte in self.0 {
            // `write!` は `String` への書き込みで失敗しないため `unwrap`/`expect`
            // を使わず戻り値を無視できる（`Result` を捨てるのではなく、この
            // 書き込み先に限り失敗しえないことが保証されている）。
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

impl fmt::Display for DefinitionHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for DefinitionHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DefinitionHash({})", self.to_hex())
    }
}

/// 正準化・ハッシュ計算時のエラー。
///
/// `Display` は固定の英語文言のみを返し、定義本文（表示名・説明・`name`）を
/// 含めない（security.md「秘密情報の混入防止」。`DefinitionError` の
/// `UnknownField`/`TypeMismatch` と同じ方針）。
#[derive(Debug)]
#[non_exhaustive]
pub enum CanonicalError {
    /// `serde_json::to_value`（`Definition` の `Serialize` 経由）が失敗した。
    /// `Definition` は検証済みの型付きフィールドしか持たないため実質的には
    /// 起こらないが、`Result` を返すライブラリコードとして `?` で伝播できる
    /// ようにしておく（coding-rust.md「ライブラリコードでは `Result` を返し、
    /// panic させない」）。
    Serialize { source: serde_json::Error },
    /// 正準化の対象に浮動小数（JSON の non-integer number）が含まれていた。
    /// 現状の `Definition` のスキーマには浮動小数のフィールドが存在しないが、
    /// `Definition` の将来のフィールド追加で紛れ込んだ場合に、数値表現の
    /// 揺れ（`1.0` と `1.00` 等が異なるハッシュになる）を正準化ハッシュへ
    /// 持ち込まないよう fail-closed で拒否する。
    UnsupportedNumber,
}

impl fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CanonicalError::Serialize { source } => {
                write!(f, "failed to canonicalize definition: {source}")
            }
            CanonicalError::UnsupportedNumber => {
                write!(
                    f,
                    "canonicalization does not support floating-point numbers"
                )
            }
        }
    }
}

impl std::error::Error for CanonicalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CanonicalError::Serialize { source } => Some(source),
            CanonicalError::UnsupportedNumber => None,
        }
    }
}

/// 任意の `Serialize` 値を正準化 JSON 文字列にする（PoC-19 の `canon_hash` と
/// 同じ規則。本モジュール冒頭の doc を参照）。
///
/// `Definition` に限定していないのは、[`Definition::canonical_json`] から
/// `serde_json::Value` へ変換した後の値を渡すためで、将来 `Definition` 以外の
/// 正準化ハッシュ（例: モデルの版管理台帳）が必要になった場合にも同じ規則を
/// 再利用できるようにするため（正準化の規則を 1 箇所に集約する。REQ-15）。
pub fn canonical_json<T: serde::Serialize>(value: &T) -> Result<String, CanonicalError> {
    let json_value =
        serde_json::to_value(value).map_err(|source| CanonicalError::Serialize { source })?;
    let mut out = String::new();
    write_canonical(&json_value, &mut out)?;
    Ok(out)
}

/// [`canonical_json`] の内部実装。`serde_json::Value` を再帰的に走査し、
/// オブジェクトのキーを明示的にソートして書き出す。
fn write_canonical(value: &serde_json::Value, out: &mut String) -> Result<(), CanonicalError> {
    match value {
        serde_json::Value::Null => {
            out.push_str("null");
            Ok(())
        }
        serde_json::Value::Bool(b) => {
            out.push_str(if *b { "true" } else { "false" });
            Ok(())
        }
        serde_json::Value::Number(number) => {
            // `u64`/`i64` として表現できる整数のみを許可する（fail-closed。
            // 本モジュール冒頭の doc の「数値」を参照）。
            if let Some(unsigned) = number.as_u64() {
                out.push_str(&unsigned.to_string());
                Ok(())
            } else if let Some(signed) = number.as_i64() {
                out.push_str(&signed.to_string());
                Ok(())
            } else {
                Err(CanonicalError::UnsupportedNumber)
            }
        }
        serde_json::Value::String(s) => {
            // `serde_json::to_string` は文字列値を JSON 文字列リテラルとして
            // エスケープする（`"`・`\`・制御文字のみ。非 ASCII は UTF-8 のまま。
            // Python の `ensure_ascii=False` と同じ挙動）。`Definition` の
            // 検証時にサイズ上限を通過済みの文字列のみが対象のため、ここでの
            // 追加のサイズ検査は行わない。
            let escaped =
                serde_json::to_string(s).map_err(|source| CanonicalError::Serialize { source })?;
            out.push_str(&escaped);
            Ok(())
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical(item, out)?;
            }
            out.push(']');
            Ok(())
        }
        serde_json::Value::Object(map) => {
            out.push('{');
            // `serde_json::Map` は既定で `BTreeMap`（`preserve_order` feature
            // 無効時）のためこの `iter()` は既にキー順だが、workspace のどこかで
            // `preserve_order` が有効化されると（feature unification）挿入順に
            // 変わる落とし穴があるため、ここで明示的に再ソートして依存しない
            // ようにする（本モジュール冒頭の doc を参照）。
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by_key(|(key, _)| *key);
            for (index, (key, val)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                let escaped_key = serde_json::to_string(key)
                    .map_err(|source| CanonicalError::Serialize { source })?;
                out.push_str(&escaped_key);
                out.push(':');
                write_canonical(val, out)?;
            }
            out.push('}');
            Ok(())
        }
    }
}

/// 正準化 JSON（UTF-8 バイト列）の sha256 を計算する。
pub(crate) fn sha256_hex_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::Definition;

    /// PoC-19 の `canon_hash` と同じ規則で Python 標準ライブラリ
    /// （`json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
    /// → `hashlib.sha256`）を使って独立に計算したゴールデン値（証拠の種別:
    /// テストハーネス）。実装担当は同じ手順（または
    /// `printf '%s' '<canonical>' | sha256sum`）で再確認する
    /// （create-plan の検証方法 §7）。
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

    const DEFINITION_A_CANONICAL: &str = "{\"io\":{\"input\":\"bytes\"},\"judgment_type\":\"single_select\",\"name\":\"sample_topic\",\"options\":[{\"description\":\"肯定\",\"display_name\":\"Yes\",\"id\":\"yes\"},{\"description\":\"否定\",\"display_name\":\"No\",\"id\":\"no\"}],\"schema\":\"fandhe-edge-model-definition/v1\",\"version\":1}";

    const DEFINITION_A_HASH_HEX: &str =
        "db372ae2b27530fafc7adbc02daa542a8bc1901d7ce916c2a05552d052aef7ac";

    fn definition_a() -> Definition {
        Definition::parse(DEFINITION_A_JSON).expect("固定 fixture は valid なはず")
    }

    #[test]
    fn req15_task15_5_canonical_json_matches_golden_string() {
        let def = definition_a();
        let json = def.canonical_json().expect("正準化に失敗しないはず");
        assert_eq!(json, DEFINITION_A_CANONICAL);
    }

    #[test]
    fn req15_task15_5_canonical_hash_matches_golden_sha256() {
        let def = definition_a();
        let hash = def.canonical_hash().expect("ハッシュ計算に失敗しないはず");
        assert_eq!(hash.to_hex(), DEFINITION_A_HASH_HEX);
    }

    #[test]
    fn req15_task15_5_sha256_known_vector() {
        assert_eq!(
            DefinitionHash(sha256_hex_bytes(b"abc")).to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            DefinitionHash(sha256_hex_bytes(b"")).to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn req15_task15_5_display_only_change_same_identity_different_hash() {
        let json_b = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "はい", "description": "肯定" },
                { "id": "no", "display_name": "No", "description": "否定する" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def_a = definition_a();
        let def_b = Definition::parse(json_b).expect("valid なはず");

        assert_eq!(def_a.identity(), def_b.identity());
        let expected_ids: BTreeSet<String> =
            ["no".to_string(), "yes".to_string()].into_iter().collect();
        assert_eq!(def_a.identity().option_ids(), &expected_ids);
        assert_eq!(def_a.identity().judgment_type(), JudgmentType::SingleSelect);

        let hash_a = def_a.canonical_hash().expect("失敗しないはず");
        let hash_b = def_b.canonical_hash().expect("失敗しないはず");
        assert_ne!(hash_a, hash_b);
        assert_eq!(
            hash_b.to_hex(),
            "7a8a8d3cf75dc165d28f350739e50c619410cc83d779159d905e4eb8fd6b0e6f"
        );
    }

    #[test]
    fn req15_task15_5_key_order_and_whitespace_do_not_change_hash() {
        // トップレベル・選択肢内のキー順と空白・改行を変えた同一内容の JSON。
        let reordered = "{\n  \"version\" : 1,\n  \"io\": { \"input\" : \"bytes\" },\n  \"name\": \"sample_topic\",\n  \"schema\": \"fandhe-edge-model-definition/v1\",\n  \"judgment_type\": \"single_select\",\n  \"options\": [\n    { \"description\": \"肯定\", \"id\": \"yes\", \"display_name\": \"Yes\" },\n    { \"display_name\": \"No\", \"description\": \"否定\", \"id\": \"no\" }\n  ]\n}\n";
        let def_a = definition_a();
        let def_reordered = Definition::parse(reordered).expect("valid なはず");

        assert_eq!(
            def_reordered
                .canonical_hash()
                .expect("失敗しないはず")
                .to_hex(),
            DEFINITION_A_HASH_HEX
        );
        assert_eq!(def_a.identity(), def_reordered.identity());
    }

    #[test]
    fn req15_task15_5_option_order_change_same_identity_different_hash() {
        let reversed_options = r#"{
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
        let def_reversed = Definition::parse(reversed_options).expect("valid なはず");

        assert_eq!(def_a.identity(), def_reversed.identity());
        assert_eq!(
            def_reversed
                .canonical_hash()
                .expect("失敗しないはず")
                .to_hex(),
            "f3779c6653a582b1f608bfb348be0dc0f9e1c5b8dc12937e9f10a1b338b20585"
        );
        assert_ne!(
            def_a.canonical_hash().expect("失敗しないはず"),
            def_reversed.canonical_hash().expect("失敗しないはず")
        );
    }

    #[test]
    fn req15_task15_5_option_id_change_changes_identity() {
        let renamed_id = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "Yes", "description": "肯定" },
                { "id": "maybe", "display_name": "No", "description": "否定" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let def_a = definition_a();
        let def_renamed = Definition::parse(renamed_id).expect("valid なはず");

        assert_ne!(def_a.identity(), def_renamed.identity());
        let expected_ids: BTreeSet<String> = ["maybe".to_string(), "yes".to_string()]
            .into_iter()
            .collect();
        assert_eq!(def_renamed.identity().option_ids(), &expected_ids);
        assert_ne!(
            def_a.canonical_hash().expect("失敗しないはず"),
            def_renamed.canonical_hash().expect("失敗しないはず")
        );
    }

    #[test]
    fn req15_task15_5_name_or_version_change_keeps_identity() {
        let renamed = r#"{
            "schema": "fandhe-edge-model-definition/v1",
            "name": "sample_topic_renamed",
            "version": 1,
            "judgment_type": "single_select",
            "options": [
                { "id": "yes", "display_name": "Yes", "description": "肯定" },
                { "id": "no", "display_name": "No", "description": "否定" }
            ],
            "io": { "input": "bytes" }
        }"#;
        let rebumped = r#"{
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
        let def_renamed = Definition::parse(renamed).expect("valid なはず");
        let def_rebumped = Definition::parse(rebumped).expect("valid なはず");

        assert_eq!(def_a.identity(), def_renamed.identity());
        assert_eq!(def_a.identity(), def_rebumped.identity());

        let hash_a = def_a.canonical_hash().expect("失敗しないはず");
        let hash_renamed = def_renamed.canonical_hash().expect("失敗しないはず");
        let hash_rebumped = def_rebumped.canonical_hash().expect("失敗しないはず");
        assert_ne!(hash_a, hash_renamed);
        assert_ne!(hash_a, hash_rebumped);
        assert_eq!(
            hash_renamed.to_hex(),
            "90021dc7734f3be879f6b59966fff3d7606e16fd89092b6bdacdc932645270e3"
        );
        assert_eq!(
            hash_rebumped.to_hex(),
            "7dc1f8125d42e308f540cd1ec420cd19012f9c5fa00ffd84b8db81bdd7540b3e"
        );
    }

    #[test]
    fn req15_task15_5_canonical_json_escapes_control_and_keeps_non_ascii() {
        let json = "{\n            \"schema\": \"fandhe-edge-model-definition/v1\",\n            \"name\": \"sample_topic\",\n            \"version\": 1,\n            \"judgment_type\": \"single_select\",\n            \"options\": [\n                { \"id\": \"yes\", \"display_name\": \"Yes\", \"description\": \"quote\\\"back\\\\slash\\nnewline\\u65e5\\u672c\\u8a9e\" },\n                { \"id\": \"no\", \"display_name\": \"No\", \"description\": \"\\u5426\\u5b9a\" }\n            ],\n            \"io\": { \"input\": \"bytes\" }\n        }";
        let def = Definition::parse(json).expect("valid なはず（JSON エスケープされた入力）");
        let canonical = def.canonical_json().expect("失敗しないはず");
        assert!(canonical.contains(r#"quote\"back\\slash\nnewline日本語"#));
        assert!(canonical.contains("否定"));
    }

    #[test]
    fn canonical_json_rejects_floating_point_numbers() {
        let value = serde_json::json!({ "x": 1.5 });
        let err = canonical_json(&value).unwrap_err();
        assert!(matches!(err, CanonicalError::UnsupportedNumber));
    }

    #[test]
    fn definition_hash_display_is_lowercase_hex_64_chars() {
        let hash = DefinitionHash([0u8; 32]);
        let s = hash.to_string();
        assert_eq!(s.len(), 64);
        assert!(
            s.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase())
        );
        assert_eq!(s, "0".repeat(64));
    }

    // `JudgmentType` は現状 `SingleSelect` の 1 バリアントしか持たないため、
    // 「判定型が変わると同一性も変わる」ケースはテストで作れない
    // （バリアントを増やせないと `match` を通さずには別の値を作れない）。
    // バリアントが増えた時点（TASK-20.1 の範囲）でテストを追加する。
}
