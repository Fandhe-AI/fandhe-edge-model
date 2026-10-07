//! 1 件ごとの予測の JSONL 行の組み立て（`evaluate` の保存と PoC-26 の採点入口の共有。REQ-27・REQ-41・#445）。
//!
//! # 役割
//!
//! 行形式は `{"id","status","predicted_label","scores"?}`。この形式は、データ契約層の
//! [`fandhe_edge_data::eval_input::prepare_evaluation_input`]（pred 側の読み手）が
//! [`Outcome`] と同じ分類へ戻せるものに限る（`ok`＝ラベル、`abstain`、`error`、`ok` かつ
//! `predicted_label:null`＝型不正）。cli は `serde_json` に依存しない（依存最小）ため、行は手組みし、
//! 任意の文字列（id・ラベル）は [`json_string`] でエスケープする。
//!
//! 評価データ本文は扱わない（id・ラベル・スコアのみ）。

use fandhe_edge_eval::metrics::Outcome;

/// JSON の文字列リテラル（引用符つき）にする。制御文字・引用符・バックスラッシュをエスケープする。
#[must_use]
pub(crate) fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// 有限な `f64` を JSON の数値にする（`NaN`・無限大は `null`）。
#[must_use]
pub(crate) fn json_f64(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".to_string()
    }
}

/// 予測 1 件の JSONL 行（改行なし）を作る。
///
/// `scores` は（選択肢 ID 列, スコア列）。長さが違う・非有限の値を含むときは `scores` を出さない
/// （読み手が不正なスコアを不正解に数えるため、壊れた値を書かない）。
#[must_use]
pub(crate) fn prediction_line(
    id: &str,
    outcome: &Outcome,
    scores: Option<(&[&str], &[f64])>,
) -> String {
    let (status, label) = match outcome {
        Outcome::Label(l) => ("ok", json_string(l)),
        Outcome::Invalid => ("ok", "null".to_string()),
        Outcome::Abstain => ("abstain", "null".to_string()),
        Outcome::Error => ("error", "null".to_string()),
    };
    let mut line = format!(
        "{{\"id\":{},\"status\":\"{status}\",\"predicted_label\":{label}",
        json_string(id)
    );
    if let Some((ids, values)) = scores
        && ids.len() == values.len()
        && values.iter().all(|v| v.is_finite())
    {
        let body: Vec<String> = ids
            .iter()
            .zip(values)
            .map(|(k, v)| format!("{}:{}", json_string(k), json_f64(*v)))
            .collect();
        line.push_str(&format!(",\"scores\":{{{}}}", body.join(",")));
    }
    line.push('}');
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-27: 行形式の具体値（ラベル・保留・エラー・型不正）。
    #[test]
    fn req27_prediction_line_shapes() {
        assert_eq!(
            prediction_line(
                "a\"1",
                &Outcome::Label("x".into()),
                Some((&["x", "y"], &[0.25, 0.75]))
            ),
            r#"{"id":"a\"1","status":"ok","predicted_label":"x","scores":{"x":0.25,"y":0.75}}"#
        );
        assert_eq!(
            prediction_line("b", &Outcome::Abstain, None),
            r#"{"id":"b","status":"abstain","predicted_label":null}"#
        );
        assert_eq!(
            prediction_line("c", &Outcome::Error, None),
            r#"{"id":"c","status":"error","predicted_label":null}"#
        );
        assert_eq!(
            prediction_line("d", &Outcome::Invalid, Some((&["x"], &[f64::NAN]))),
            r#"{"id":"d","status":"ok","predicted_label":null}"#
        );
    }

    /// REQ-27: 制御文字は `\u00XX` にエスケープする。
    #[test]
    fn req27_json_string_escapes_control_characters() {
        assert_eq!(json_string("a\u{1}\n\\"), "\"a\\u0001\\n\\\\\"");
    }
}
