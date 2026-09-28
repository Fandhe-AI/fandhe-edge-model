//! テストハーネス専用の JSONL ローダ・Python 互換直列化ユーティリティ
//! （TASK-16.2-2）。
//!
//! `fandhe-edge-data` の src 側はファイル I/O・JSON パーサを持たない方針
//! （`.claude/rules/coding-rust.md`「層の境界」）なので、フィクスチャの読み込み・
//! PoC-9 追補 A-10 の直列化照合（挿入順を保った `json.dumps` 相当）は
//! テストコードだけに閉じ込める。ここでの厳格な検証（欠落・型違いで panic）は
//! 「参照データの矛盾件数 57 が変わる余地を作らない」ための意図的な設計で、
//! 本番のデータ検査ローダ（TASK-16.1）の緩やかなエラー処理とは別物である。
//!
//! `mod support;` は結合テストのバイナリ（`contradiction_reference.rs`・
//! `metadata_injection.rs`）ごとに個別コンパイルされるため、一方のテストしか
//! 使わない関数・型は他方のバイナリでは未使用になる。両バイナリで共有する
//! 単一のモジュールとして保守性を優先し、`dead_code` は許可する。
#![allow(dead_code)]

use std::fs;
use std::path::Path;

use fandhe_edge_data::consistency::{ContradictionRecord, MetadataRecord};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

/// 挿入順を保持する JSON 値。
///
/// `serde_json::Value` のオブジェクトは既定でキーが整列されるため、
/// PoC-9 A-10 が要求する「Python の dict 挿入順」を再現できない。
/// 本型は `Visitor::visit_map` を独自実装し、オブジェクトを
/// `Vec<(String, OrderedValue)>` として保持する。
#[derive(Debug, Clone, PartialEq)]
pub enum OrderedValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<OrderedValue>),
    Object(Vec<(String, OrderedValue)>),
}

impl<'de> Deserialize<'de> for OrderedValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct OrderedValueVisitor;

        impl<'de> Visitor<'de> for OrderedValueVisitor {
            type Value = OrderedValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("any JSON value")
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(OrderedValue::Null)
            }

            fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E> {
                Ok(OrderedValue::Bool(v))
            }

            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
                Ok(OrderedValue::Number(v.into()))
            }

            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
                Ok(OrderedValue::Number(v.into()))
            }

            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                serde_json::Number::from_f64(v)
                    .map(OrderedValue::Number)
                    .ok_or_else(|| E::custom("invalid float value"))
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
                Ok(OrderedValue::String(v.to_string()))
            }

            fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
                Ok(OrderedValue::String(v))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element::<OrderedValue>()? {
                    items.push(item);
                }
                Ok(OrderedValue::Array(items))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut pairs = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, OrderedValue>()? {
                    pairs.push((k, v));
                }
                Ok(OrderedValue::Object(pairs))
            }
        }

        deserializer.deserialize_any(OrderedValueVisitor)
    }
}

impl OrderedValue {
    /// オブジェクトであれば `key` の値を取得する（無ければ `None`）。
    pub fn get(&self, key: &str) -> Option<&OrderedValue> {
        match self {
            OrderedValue::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            OrderedValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// オブジェクトのキーを昇順に並べ替えた新しい値を返す（配列・ネストした
    /// オブジェクトも再帰的に並べ替える）。Python の
    /// `json.dumps(v, sort_keys=True)` 相当の入力を作るために使う。
    pub fn sorted(&self) -> OrderedValue {
        match self {
            OrderedValue::Object(pairs) => {
                let mut sorted_pairs: Vec<(String, OrderedValue)> =
                    pairs.iter().map(|(k, v)| (k.clone(), v.sorted())).collect();
                sorted_pairs.sort_by(|a, b| a.0.cmp(&b.0));
                OrderedValue::Object(sorted_pairs)
            }
            OrderedValue::Array(items) => {
                OrderedValue::Array(items.iter().map(OrderedValue::sorted).collect())
            }
            other => other.clone(),
        }
    }
}

/// Python の `json.dumps(v, ensure_ascii=False)`（`separators` 既定値
/// `(', ', ': ')`）と同じ文字列を返す（`OrderedValue::sorted()` と組み合わせれば
/// `sort_keys=True` 相当になる）。
///
/// フィクスチャの注入文字列（`", "` / `": "` 区切り）と一致させるための
/// テスト専用実装で、数値・真偽値・文字列・配列・オブジェクトのみを
/// 対象とする（本テストで使う範囲を超える型は扱わない）。
pub fn py_dumps(v: &OrderedValue) -> String {
    match v {
        OrderedValue::Null => "null".to_string(),
        OrderedValue::Bool(b) => b.to_string(),
        OrderedValue::Number(n) => n.to_string(),
        OrderedValue::String(s) => serde_json::to_string(s).unwrap_or_default(),
        OrderedValue::Array(items) => {
            let parts: Vec<String> = items.iter().map(py_dumps).collect();
            format!("[{}]", parts.join(", "))
        }
        OrderedValue::Object(pairs) => {
            let parts: Vec<String> = pairs
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}: {}",
                        serde_json::to_string(k).unwrap_or_default(),
                        py_dumps(v)
                    )
                })
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// フィクスチャ 1 行分の生レコード（厳格ローダ）。
///
/// 欠落・型違いはテストの前提が崩れていることを意味するため、ここでは
/// `panic` で即座に落とす（本番のデータ検査ローダ TASK-16.1 は緩やかな
/// エラー処理を行う想定で、本構造体はそれを代替しない）。
#[derive(Debug, Clone)]
pub struct RawRecord {
    pub id: String,
    pub input: String,
    pub output: OrderedValue,
    pub group_id: Option<String>,
}

/// JSONL ファイルを 1 行ずつ厳格に読み込む（テスト専用。REQ-39 のガード層は
/// 本番ローダの責務であり、ここではサイズ上限を検査しない）。
///
/// 欠落フィールド・型違いの行があれば、そのファイルパスと行番号を含めて
/// `panic` する。
pub fn load_jsonl_strict(path: &Path) -> Vec<RawRecord> {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    let mut records = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: OrderedValue = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("{}:{}: invalid JSON: {e}", path.display(), line_no + 1));
        let id = value
            .get("id")
            .and_then(OrderedValue::as_str)
            .unwrap_or_else(|| {
                panic!(
                    "{}:{}: missing or non-string `id`",
                    path.display(),
                    line_no + 1
                )
            })
            .to_string();
        let input = value
            .get("input")
            .and_then(OrderedValue::as_str)
            .unwrap_or_else(|| {
                panic!(
                    "{}:{}: missing or non-string `input`",
                    path.display(),
                    line_no + 1
                )
            })
            .to_string();
        let output = value
            .get("output")
            .cloned()
            .unwrap_or_else(|| panic!("{}:{}: missing `output`", path.display(), line_no + 1));
        if !matches!(output, OrderedValue::Object(_)) {
            panic!(
                "{}:{}: `output` is not an object",
                path.display(),
                line_no + 1
            );
        }
        let group_id = value
            .get("group_id")
            .and_then(OrderedValue::as_str)
            .map(str::to_string);

        records.push(RawRecord {
            id,
            input,
            output,
            group_id,
        });
    }
    records
}

/// [`load_jsonl_strict`] を複数ファイル分読み込んで連結する（参照データの
/// train + validation + test プールを組み立てるのに使う。legacy_holdout は
/// 対象外。PoC-9 の `contradictions_pool` と同じ範囲）。
pub fn load_jsonl_pool(paths: &[&Path]) -> Vec<RawRecord> {
    let mut all = Vec::new();
    for path in paths {
        all.extend(load_jsonl_strict(path));
    }
    all
}

/// [`ContradictionRecord`] の実装（参照データ用）。
///
/// `gold_key` は `output`（`OrderedValue`）をキー整列してから
/// `serde_json::to_string` した文字列。挿入順ではなく整列済みキーを使うため、
/// 同じ意味の出力（キー順だけが違う場合を含む）は同じ `gold_key` になる。
pub struct ContradictionTestRecord {
    pub raw: RawRecord,
    pub gold_key: String,
}

impl ContradictionTestRecord {
    pub fn from_raw(raw: RawRecord) -> Self {
        let sorted_json = ordered_to_json(&raw.output.sorted());
        let gold_key = sorted_json.to_string();
        Self { raw, gold_key }
    }
}

impl ContradictionRecord for ContradictionTestRecord {
    fn id(&self) -> &str {
        &self.raw.id
    }
    fn input(&self) -> &str {
        &self.raw.input
    }
    fn gold_key(&self) -> &str {
        &self.gold_key
    }
    fn group_id(&self) -> Option<&str> {
        self.raw.group_id.as_deref()
    }
}

/// `OrderedValue` を `serde_json::Value` へ変換する（`gold_key` の
/// 決定的な文字列化に使う。`Value::Object` は既定で `BTreeMap` ベースのため、
/// 変換後は常にキー整列済みになる）。
fn ordered_to_json(v: &OrderedValue) -> Value {
    match v {
        OrderedValue::Null => Value::Null,
        OrderedValue::Bool(b) => Value::Bool(*b),
        OrderedValue::Number(n) => Value::Number(n.clone()),
        OrderedValue::String(s) => Value::String(s.clone()),
        OrderedValue::Array(items) => Value::Array(items.iter().map(ordered_to_json).collect()),
        OrderedValue::Object(pairs) => {
            let map = pairs
                .iter()
                .map(|(k, v)| (k.clone(), ordered_to_json(v)))
                .collect();
            Value::Object(map)
        }
    }
}

/// [`MetadataRecord`] の実装（メタデータ混入検出テスト用）。
///
/// `gold_serializations` は PoC-9 追補 A-10 のとおり、`output` 全体・
/// `arguments` の 2 種 × キー整列の有無 2 種の計 4 通りを候補として渡す。
pub struct MetadataTestRecord {
    pub raw: RawRecord,
    pub gold_label: Option<String>,
    pub gold_serializations: Vec<String>,
}

impl MetadataTestRecord {
    pub fn from_raw(raw: RawRecord) -> Self {
        let gold_label = raw
            .output
            .get("intent")
            .and_then(OrderedValue::as_str)
            .map(str::to_string);

        let mut gold_serializations = vec![py_dumps(&raw.output), py_dumps(&raw.output.sorted())];
        if let Some(arguments) = raw.output.get("arguments") {
            gold_serializations.push(py_dumps(arguments));
            gold_serializations.push(py_dumps(&arguments.sorted()));
        }

        Self {
            raw,
            gold_label,
            gold_serializations,
        }
    }
}

impl MetadataRecord for MetadataTestRecord {
    fn id(&self) -> &str {
        &self.raw.id
    }
    fn input(&self) -> &str {
        &self.raw.input
    }
    fn gold_label(&self) -> Option<&str> {
        self.gold_label.as_deref()
    }
    fn gold_serializations(&self) -> &[String] {
        &self.gold_serializations
    }
}
