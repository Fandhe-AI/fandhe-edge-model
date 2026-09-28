//! `kind` ごとの config 既定値（REQ-18・REQ-19・REQ-19b・REQ-21・REQ-39）。
//!
//! 正本は学習ワーカー側（`trainer/src/fandhe_edge_trainer/kinds/c1.py`・
//! `kinds/c3.py` の `DEFAULT_CONFIG`。選択口 `kinds/__init__.py::_registry`
//! に登録済みの種類のみ）。本モジュールは、その値をそのまま書き出した共有
//! fixture `fixtures/train_contract/kind_defaults.json`（`trainer/tests/
//! test_kind_defaults_fixture.py` が Python 側の `DEFAULT_CONFIG` との完全
//! 一致を機械照合する）をビルド時に [`include_str!`] で埋め込み、
//! [`crate::result::TrainOutcome::from_worker_stdout`] が成果物の
//! `artifact.config` を「defaults(kind) に `request.config` を上書きした
//! 実効 config」と完全一致で検査するために使う（codex review PR #220 P1
//! 「成果物の追加 config 値を検証せず成功扱いにしている」への対応。
//! 旧 `config_matches_explicit_keys` の部分一致検査を置き換える）。
//!
//! `kind` ごとの既定値の集合そのものは本 crate では検証しない（層の境界。
//! `.claude/rules/dependency-policy.md`「学習側の依存を推論ランタイムへ
//! 漏らさない」と同じ理由で、学習ワーカーの実装詳細〔具体的なハイパー
//! パラメータの意味〕をここへ複製しない。値の出典のみを共有する）。

use serde_json::{Map, Value};
use std::sync::OnceLock;

/// `fixtures/train_contract/kind_defaults.json`（`crates/train/` からの相対
/// パス。`docs/spec` 配下ではない）。ビルド時に埋め込む（実行時のファイル
/// 読み込みに失敗する経路を作らない）。
const KIND_DEFAULTS_JSON: &str =
    include_str!("../../../fixtures/train_contract/kind_defaults.json");

/// fixture 内でメタ情報に使う予約キー（`kind` としては解決しない）。
const RESERVED_META_KEY: &str = "_meta";

/// 解析結果を 1 度だけ計算してキャッシュする。解析失敗（fixture が壊れて
/// いる場合）は `Err` として保持し、以後の呼び出しでも panic させない
/// （`.claude/rules/coding-rust.md`「外部入力の経路では unwrap/expect を
/// 使わない」と同じ考え方をビルド時埋め込みにも適用する。fixture は本 crate
/// が管理する内部データだが、解析失敗時に `Result` で返す経路を用意する
/// ことで、将来 fixture が壊れて配布された場合でも学習リクエストの処理が
/// `panic` で落ちるのではなく `runtime_error` として扱えるようにする）。
fn parsed_kind_defaults() -> &'static Result<Map<String, Value>, String> {
    static PARSED: OnceLock<Result<Map<String, Value>, String>> = OnceLock::new();
    PARSED.get_or_init(|| {
        serde_json::from_str::<Value>(KIND_DEFAULTS_JSON)
            .map_err(|e| format!("kind_defaults.json is not valid JSON: {e}"))
            .and_then(|value| {
                value
                    .as_object()
                    .cloned()
                    .ok_or_else(|| "kind_defaults.json must be a JSON object".to_string())
            })
    })
}

/// `kind` に対応する既定 config（`fixtures/train_contract/kind_defaults.json`
/// に登録が無い・fixture の解析に失敗した場合は `None`。呼び出し元は
/// fail-closed でこれを「未知の種類」と同様に拒否する）。
pub(crate) fn defaults_for_kind(kind: &str) -> Option<&'static Map<String, Value>> {
    if kind == RESERVED_META_KEY {
        return None;
    }
    let parsed = parsed_kind_defaults().as_ref().ok()?;
    parsed.get(kind)?.as_object()
}

/// `defaults(kind)` に `request_config` を上書きした実効 config を返す
/// （`kinds/c1.py`・`kinds/c3.py::train` の `{**DEFAULT_CONFIG,
/// **request.config}` と同じ上書き規則。値は [`serde_json::Value`] の
/// 同一性で扱い、型変換・数値の丸めは行わない）。`kind` が
/// [`defaults_for_kind`] で解決できない場合は `None`。
pub(crate) fn effective_config(
    kind: &str,
    request_config: &Map<String, Value>,
) -> Option<Map<String, Value>> {
    let defaults = defaults_for_kind(kind)?;
    let mut effective = defaults.clone();
    for (key, value) in request_config {
        effective.insert(key.clone(), value.clone());
    }
    Some(effective)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-19・REQ-21: fixture に登録済みの `c1`・`c3` は解決できる。
    #[test]
    fn req19_defaults_for_known_kinds_are_present() {
        assert!(defaults_for_kind("c1").is_some());
        assert!(defaults_for_kind("c3").is_some());
        let c3_defaults = defaults_for_kind("c3").expect("c3 defaults");
        assert_eq!(c3_defaults.get("epochs"), Some(&serde_json::json!(40)));
    }

    /// REQ-19・REQ-39・P1（codex 指摘 PR #220）: fixture に無い `kind`・
    /// 予約キーは `None`（拒否対象）。
    #[test]
    fn req39_defaults_for_unknown_kind_is_none() {
        assert!(defaults_for_kind("unknown-kind").is_none());
        assert!(defaults_for_kind("_meta").is_none());
        assert!(defaults_for_kind("").is_none());
    }

    /// REQ-19・REQ-21: 上書き規則（明示キーは上書きし、それ以外は既定値の
    /// まま）。
    #[test]
    fn req19_effective_config_overrides_only_explicit_keys() {
        let mut requested = Map::new();
        requested.insert("epochs".to_string(), serde_json::json!(2));
        let effective = effective_config("c3", &requested).expect("c3 known kind");
        assert_eq!(effective.get("epochs"), Some(&serde_json::json!(2)));
        assert_eq!(effective.get("lr"), Some(&serde_json::json!(0.001)));
        assert_eq!(effective.len(), defaults_for_kind("c3").unwrap().len());
    }

    /// REQ-39・P1: 未知の `kind` は `None`（呼び出し元は拒否する）。
    #[test]
    fn req39_effective_config_for_unknown_kind_is_none() {
        assert!(effective_config("unknown-kind", &Map::new()).is_none());
    }
}
