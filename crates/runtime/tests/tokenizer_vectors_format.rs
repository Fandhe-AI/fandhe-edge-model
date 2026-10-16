//! `fixtures/preprocess/tokenizer_vectors.json`（tokenizer 入力表現のゴールデンベクタ）の
//! 形式検査（REQ-41・TASK-41.3-3・#404。証拠種別: テストハーネス）。
//!
//! 実トークナイザーとの全件一致は TASK-41.4-5（#413）の範囲で、ここでは形式のみ検査する。
//! Python 側 `trainer/tests/test_tokenizer_vectors.py` と同じ項目・同じ順で検査する。
//! fixture は外部入力として扱い、読み込み前にサイズを確認し、`[]` 添字を使わない。

use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;
const MAX_VECTORS: usize = 1000;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("preprocess")
        .join("tokenizer_vectors.json")
}

fn is_lower_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn is_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10
        && b.iter().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                *c == b'-'
            } else {
                c.is_ascii_digit()
            }
        })
}

fn non_empty_str<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    match v.get(key).and_then(Value::as_str) {
        Some(s) if !s.is_empty() => Ok(s),
        _ => Err(format!("{key} missing or empty")),
    }
}

/// 形式を検査して件数を返す。違反は理由つき `Err`。
fn check_format(text: &str) -> Result<usize, String> {
    if !text.is_ascii() {
        return Err("non-ascii bytes".into());
    }
    let root: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let meta = root.get("_meta").ok_or("_meta missing")?;
    if non_empty_str(meta, "input_representation")? != "tokenizer" {
        return Err("input_representation must be tokenizer".into());
    }
    if !is_lower_hex64(non_empty_str(meta, "tokenizer_json_sha256")?) {
        return Err("tokenizer_json_sha256 must be lowercase hex64".into());
    }
    non_empty_str(meta, "reference")?;
    if !is_date(non_empty_str(meta, "generated_on")?) {
        return Err("generated_on must be YYYY-MM-DD".into());
    }
    let vectors = root
        .get("vectors")
        .and_then(Value::as_array)
        .ok_or("vectors missing")?;
    if vectors.is_empty() || vectors.len() > MAX_VECTORS {
        return Err(format!("vector count out of range: {}", vectors.len()));
    }
    let mut names = HashSet::new();
    for v in vectors {
        let name = non_empty_str(v, "name")?;
        if !names.insert(name.to_string()) {
            return Err(format!("duplicate name {name}"));
        }
        if v.get("input").and_then(Value::as_str).is_none() {
            return Err(format!("{name}: input must be string"));
        }
        let ids = v
            .get("ids")
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty())
            .ok_or_else(|| format!("{name}: ids missing or empty"))?;
        for id in ids {
            match id.as_u64() {
                Some(n) if n <= u64::from(u32::MAX) => {}
                _ => return Err(format!("{name}: id out of u32 range")),
            }
        }
    }
    Ok(vectors.len())
}

fn load() -> String {
    let path = fixture_path();
    let len = fs::metadata(&path).expect("fixture metadata").len();
    assert!(len <= MAX_FIXTURE_BYTES, "fixture too large: {len}");
    fs::read_to_string(&path).expect("fixture read")
}

/// REQ-41: 共有 fixture が形式を満たし、具体値でも空回りしていない。
#[test]
fn req41_tokenizer_vectors_format_ok() {
    let text = load();
    assert_eq!(check_format(&text), Ok(23));
    let root: Value = serde_json::from_str(&text).expect("json");
    let first = root
        .get("vectors")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .expect("first vector");
    assert_eq!(first.get("name").and_then(Value::as_str), Some("poc26_01"));
    let ids = first.get("ids").and_then(Value::as_array).expect("ids");
    assert_eq!(ids.len(), 6);
    assert_eq!(ids.first().and_then(Value::as_u64), Some(133165));
}

fn mutated(from: &str, to: &str) -> String {
    let text = load();
    assert!(text.contains(from), "mutation target missing: {from}");
    text.replacen(from, to, 1)
}

/// REQ-41: 違反（不正 sha・重複 name・負の id・入力表現違い）は fail-closed で拒否する。
#[test]
fn req41_tokenizer_vectors_format_rejects_violations() {
    let sha = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";
    let bad_sha = mutated(sha, &sha.to_uppercase());
    assert_eq!(
        check_format(&bad_sha),
        Err("tokenizer_json_sha256 must be lowercase hex64".into())
    );
    let dup = mutated("poc26_02", "poc26_01");
    assert_eq!(check_format(&dup), Err("duplicate name poc26_01".into()));
    let neg = mutated("133165", "-1");
    assert_eq!(
        check_format(&neg),
        Err("poc26_01: id out of u32 range".into())
    );
    let rep = mutated("\"tokenizer\"", "\"byte\"");
    assert_eq!(
        check_format(&rep),
        Err("input_representation must be tokenizer".into())
    );
}
