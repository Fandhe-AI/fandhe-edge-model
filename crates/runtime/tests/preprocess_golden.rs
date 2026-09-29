//! 共有ゴールデンベクタ `fixtures/preprocess/byte_encoding_vectors.json` と Rust 前処理の
//! 機械照合（REQ-32・TASK-32.1-1・#112。証拠種別: テストハーネス）。
//!
//! fixture は学習ワーカー（Python）の `normalize_input` + `encode_bytes` を SSOT とする手動導出値。
//! 不一致時はコードを直し、fixture・許容差・assert を緩めない（fail-closed）。
//! fixture は外部入力として扱い、読み込み前にサイズを確認し、`[]` 添字を使わない。

use fandhe_edge_runtime::pipeline::Preprocessor;
use fandhe_edge_runtime::preprocess::{ByteEncodingPreprocessor, encode_bytes, normalize_input};
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

/// fixture のサイズ上限（読み込み前に確認。REQ-39 と同じ作法）。
const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

fn load_fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("preprocess")
        .join("byte_encoding_vectors.json");
    let len = fs::metadata(&path).expect("fixture metadata").len();
    assert!(len <= MAX_FIXTURE_BYTES, "fixture too large: {len}");
    let text = fs::read_to_string(&path).expect("fixture read");
    serde_json::from_str(&text).expect("fixture json")
}

fn vectors(root: &Value) -> &Vec<Value> {
    root.get("vectors")
        .and_then(Value::as_array)
        .expect("vectors array")
}

fn ids_of(v: &Value) -> Result<Vec<i64>, String> {
    v.get("ids")
        .and_then(Value::as_array)
        .ok_or("ids missing")?
        .iter()
        .map(|x| x.as_i64().ok_or_else(|| "ids element".to_string()))
        .collect()
}

/// 1 ベクタを (a) 正規化 (b) エンコード合成 (c) `Preprocessor` 経由の 3 点で照合する。
fn check_vector(v: &Value) -> Result<(), String> {
    let name = v
        .get("name")
        .and_then(Value::as_str)
        .ok_or("name missing")?;
    let input = v
        .get("input")
        .and_then(Value::as_str)
        .ok_or("input missing")?;
    let max_bytes = v
        .get("max_bytes")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("max_bytes missing")?;
    let normalized = v
        .get("normalized")
        .and_then(Value::as_str)
        .ok_or("normalized missing")?;
    let ids = ids_of(v)?;

    let got_norm = normalize_input(input);
    if got_norm != normalized {
        return Err(format!(
            "{name}: normalized mismatch (got len {}, want len {})",
            got_norm.len(),
            normalized.len()
        ));
    }
    let got_ids = encode_bytes(&got_norm, max_bytes);
    if got_ids != ids {
        return Err(format!(
            "{name}: ids mismatch (got len {}, want len {})",
            got_ids.len(),
            ids.len()
        ));
    }
    let via = ByteEncodingPreprocessor::new(max_bytes)
        .preprocess(input)
        .map_err(|e| format!("{name}: {}", e.code()))?;
    if via.as_slice() != ids.as_slice() {
        return Err(format!("{name}: preprocessor ids mismatch"));
    }
    Ok(())
}

fn find<'a>(root: &'a Value, name: &str) -> &'a Value {
    vectors(root)
        .iter()
        .find(|v| v.get("name").and_then(Value::as_str) == Some(name))
        .unwrap_or_else(|| panic!("vector {name} missing"))
}

/// REQ-32: 全ベクタで `normalized`・`ids` が Rust 出力と一致する。
#[test]
fn req32_all_shared_vectors_match() {
    let root = load_fixture();
    let vs = vectors(&root);
    assert!(vs.len() >= 28, "vector count {}", vs.len());
    let mut names = HashSet::new();
    let mut checked = 0usize;
    for v in vs {
        check_vector(v).unwrap_or_else(|e| panic!("{e}"));
        let n = v.get("name").and_then(Value::as_str).expect("name");
        assert!(names.insert(n.to_string()), "duplicate name {n}");
        checked += 1;
    }
    assert_eq!(checked, vs.len());
}

/// REQ-32: 名指しのベクタが存在し、具体値でも一致する（fixture からの欠落を検出）。
#[test]
fn req32_named_vectors_present_and_match() {
    let root = load_fixture();
    for name in [
        "basic_trim_and_compress",
        "empty_input",
        "truncate_mid_multibyte",
        "max_bytes_zero",
        "fullwidth_alnum_nfkc",
        "ligature_fi",
        "compat_hangul_parenthesized",
    ] {
        check_vector(find(&root, name)).unwrap_or_else(|e| panic!("{e}"));
    }
    assert_eq!(ids_of(find(&root, "empty_input")), Ok(vec![0]));
    assert_eq!(ids_of(find(&root, "max_bytes_zero")), Ok(vec![0]));
    assert_eq!(
        ids_of(find(&root, "truncate_mid_multibyte")),
        Ok(vec![228, 130, 131, 228])
    );
    assert_eq!(normalize_input("\u{ff21}\u{ff11}"), "A1");
    assert_eq!(encode_bytes("A1", 512), vec![66, 50]);
}

/// REQ-32: fixture と `unicode-normalization` の Unicode 版が 15.0.0 で一致する。
#[test]
fn req32_unicode_version_matches_fixture() {
    let root = load_fixture();
    assert_eq!(
        root.get("_meta")
            .and_then(|m| m.get("unicode_version"))
            .and_then(Value::as_str),
        Some("15.0.0")
    );
    assert_eq!(unicode_normalization::UNICODE_VERSION, (15, 0, 0));
}

/// REQ-32: 期待値を 1 つ改変すると照合が `Err` になる（丸めず fail-closed で拒否する）。
#[test]
fn req32_mismatch_is_rejected_fail_closed() {
    let root = load_fixture();
    let base = find(&root, "basic_trim_and_compress").clone();
    assert_eq!(check_vector(&base), Ok(()));

    let mut bad_ids = base.clone();
    if let Some(first) = bad_ids
        .get_mut("ids")
        .and_then(Value::as_array_mut)
        .and_then(|a| a.first_mut())
    {
        *first = Value::from(0);
    }
    assert!(check_vector(&bad_ids).is_err());

    let mut bad_norm = base;
    if let Some(slot) = bad_norm.get_mut("normalized") {
        *slot = Value::from("hello  world");
    }
    assert!(check_vector(&bad_norm).is_err());
}
