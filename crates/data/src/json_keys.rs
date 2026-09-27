//! JSON オブジェクトの重複キー検出プリミティブ（REQ-16・REQ-23）。
//!
//! `serde_json::Value` のパースはオブジェクトを `serde_json::Map` へ順に
//! `insert` するため、同一キーが複数回出現する行（トップレベルの `id` の
//! 重複・ネスト先（`output.intent` 等）の重複、Unicode エスケープによる
//! キー重複 smuggling を含む）は後勝ちの値のみが残り、パース結果だけでは
//! 重複の事実が分からない。本モジュールはパース成功後にもう一度、文字列
//! リテラル外に現れる `:` の個数（生テキスト上の key/value ペア数。
//! [`count_raw_key_value_separators`]）とパース後の木に残ったエントリ総数
//! （[`count_tree_entries`]）を突き合わせ、両者が一致しない場合
//! （＝重複キーで木のエントリが後勝ちに潰れている場合）に重複を検出する。
//! ネスト先を含め任意の深さの重複を検出できる（再帰は成功済みパースの木を
//! たどるだけのため、`serde_json` の再帰上限に既に収まっている）。
//!
//! [`inspect`][crate::inspect]（学習データ。REQ-16）と [`eval_input`][crate::eval_input]
//! （評価入力の gold・pred。REQ-23）の双方が検出プリミティブとしてのみ本モジュールを
//! 共有する。検出後にとる方針（レコードを除外して続行するか／処理全体を停止するか）は
//! 呼び出し元ごとに異なり、本モジュールは判定しない（`inspect` はレコード単位で
//! 除外して継続、`eval_input` は `id` による突き合わせが成立しなくなるため停止する。
//! 各モジュールの doc コメント参照）。

use serde_json::Value;

/// 生テキスト（1 行）中の、文字列リテラル外に現れる `:` の個数を数える。
///
/// JSON の key/value 区切りは文字列リテラルの外側にのみ出現するため、
/// ダブルクォートで囲まれた文字列内部（エスケープされた `"` を含む）の
/// `:` は数えない。返り値はパース前の生テキスト上に存在した key/value
/// ペアの総数（＝重複を含めたキーの総出現数）に等しい。
pub(crate) fn count_raw_key_value_separators(raw_line: &str) -> usize {
    let mut in_string = false;
    let mut escaped = false;
    let mut count = 0usize;
    for byte in raw_line.bytes() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
        } else if byte == b'"' {
            in_string = true;
        } else if byte == b':' {
            count += 1;
        }
    }
    count
}

/// パース済みの JSON 木に残ったオブジェクトのエントリ総数を数える。
///
/// 同一キーが複数回出現していた場合、`serde_json::Value` のパース時点で
/// 後勝ちの 1 エントリへ潰れているため、この値は生テキスト上のキー数より
/// 小さくなる（[`count_raw_key_value_separators`] との差分が重複検出の根拠）。
pub(crate) fn count_tree_entries(value: &Value) -> usize {
    match value {
        Value::Object(map) => map.len() + map.values().map(count_tree_entries).sum::<usize>(),
        Value::Array(items) => items.iter().map(count_tree_entries).sum(),
        _ => 0,
    }
}

/// 生テキストとパース済みの値を突き合わせ、重複 JSON キーが含まれるかを判定する。
///
/// `true` の場合、トップレベルまたはネスト先のいずれかで同一キーが複数回
/// 出現しており、`value` はパース時点で後勝ちの値へ潰れている。
pub(crate) fn has_duplicate_key(raw_line: &str, value: &Value) -> bool {
    count_raw_key_value_separators(raw_line) != count_tree_entries(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 重複の無いオブジェクトでは生テキストの `:` 数と木のエントリ数が一致し、
    /// 重複と誤検出しないこと。
    #[test]
    fn no_duplicate_key_matches_counts() {
        let raw = r#"{"id":"a","output":{"intent":"x"}}"#;
        let value: Value = serde_json::from_str(raw).unwrap();
        assert!(!has_duplicate_key(raw, &value));
    }

    /// トップレベルの重複キーを検出できること。
    #[test]
    fn duplicate_top_level_key_is_detected() {
        let raw = r#"{"id":"a","id":"b"}"#;
        let value: Value = serde_json::from_str(raw).unwrap();
        assert!(has_duplicate_key(raw, &value));
    }

    /// ネスト先（`output.intent`）の重複キーを検出できること。
    #[test]
    fn duplicate_nested_key_is_detected() {
        let raw = r#"{"id":"a","output":{"intent":"x","intent":"y"}}"#;
        let value: Value = serde_json::from_str(raw).unwrap();
        assert!(has_duplicate_key(raw, &value));
    }

    /// Unicode エスケープによるキー重複 smuggling（`"id"` と `"id"` は
    /// いずれも `id` を指す）を検出できること。
    #[test]
    fn duplicate_key_via_unicode_escape_is_detected() {
        let raw = r#"{"id":"a","id":"b"}"#;
        let value: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(value, json!({"id": "b"}));
        assert!(has_duplicate_key(raw, &value));
    }

    /// 文字列リテラル内部の `:`（重複ではない）を誤検出しないこと。
    #[test]
    fn colon_inside_string_value_is_not_miscounted() {
        let raw = r#"{"id":"a:b","label":"x"}"#;
        let value: Value = serde_json::from_str(raw).unwrap();
        assert!(!has_duplicate_key(raw, &value));
    }
}
