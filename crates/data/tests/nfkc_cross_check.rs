//! 既定正規化規則（NFKC）の学習ワーカーとの交差照合（REQ-16・TASK-16.2-2・
//! issue #42・PR #188 レビュー指摘 P1）。
//!
//! `fandhe_edge_data::normalize::NfkcWhitespaceNormalizer`（矛盾検出の既定
//! 正規化規則）が、学習ワーカー（Python）`trainer/src/fandhe_edge_trainer/
//! encoding.py::normalize_input` と同じ結果になることを、共有ゴールデン
//! ベクタ `fixtures/preprocess/byte_encoding_vectors.json`（本リポ内 SSOT。
//! `trainer/tests/test_encoding.py` が Python 側から同じベクタを照合して
//! いる）で機械照合する。特に NFKC を要するベクタ（`fullwidth_alnum_nfkc`・
//! `ligature_fi`・`compat_hangul_parenthesized`）を含む全件を照合し、
//! 「既定の矛盾検出が共通の NFKC 正規化規則と一致しない」状態
//! （crates/data/src/normalize.rs:66・71 に対する P1 指摘）を再発させない。
//!
//! # Unicode 版を学習ワーカーと揃える
//!
//! fixture の `_meta.unicode_version` は Python 側実行環境（UCD 15.0.0）を
//! 記録した値であり、本 crate が使う `unicode-normalization` 0.1.22 も
//! 同じ Unicode 15.0.0 のデータテーブルを使う（`crates/data/src/normalize.rs`
//! モジュール doc「Unicode 版を学習ワーカーと揃える」参照。当初導入した
//! 0.1.25 は Unicode 17.0.0 準拠で版がずれていたため、PR #188 Codex レビュー
//! P1 指摘を受けて 0.1.22 へ固定し直した）。本テストは fixture の
//! `_meta.unicode_version` と一致した版で動いていることを前提に既定の
//! 正規化規則の出力を照合し、`crates/data/src/normalize.rs` の
//! `req16_unicode_normalization_version_matches_python_ucd_15_0_0` が
//! `unicode_normalization::UNICODE_VERSION == (15, 0, 0)` を固定する
//! （Python 側は
//! `trainer/tests/test_encoding.py::test_fixture_unicode_version_matches_runtime`・
//! `test_unicode_version_matches_rust_contract` が固定する）。
//!
//! 証拠の種別: テストハーネス（実機測定なし）。

use std::fs;
use std::path::PathBuf;

use fandhe_edge_data::normalize::{InputNormalizer, NfkcWhitespaceNormalizer};
use serde_json::Value;

/// 外部入力と同じ作法で扱うための読み込み前サイズ上限（REQ-39）。
/// リポ内固定 fixture だが、想定外の巨大化で無制限アロケーションに繋げない。
const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

/// `CARGO_MANIFEST_DIR`（`crates/data`）から `Path::join` でリポ直下の
/// fixture パスを組み立てる（文字列連結・区切り文字のハードコードをしない。
/// `.claude/rules/coding-rust.md`）。
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("preprocess")
        .join("byte_encoding_vectors.json")
}

/// fixture のルート JSON を読み込む（`vectors` 配列・`_meta` の両方を
/// 参照する呼び出し元のための共通ローダ）。
fn load_root() -> Value {
    let path = fixture_path();
    let metadata = fs::metadata(&path)
        .unwrap_or_else(|e| panic!("failed to stat preprocess vectors fixture {path:?}: {e}"));
    assert!(
        metadata.len() <= MAX_FIXTURE_BYTES,
        "preprocess vectors fixture {path:?} exceeds size limit ({} bytes > {MAX_FIXTURE_BYTES})",
        metadata.len()
    );
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read preprocess vectors fixture {path:?}: {e}"));
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("failed to parse preprocess vectors fixture {path:?}: {e}"))
}

/// fixture を読み込み `vectors` 配列を返す。
fn load_vectors() -> Vec<Value> {
    let path = fixture_path();
    let root = load_root();
    root.get("vectors")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("preprocess vectors fixture {path:?} has no \"vectors\" array"))
        .clone()
}

/// REQ-16: 共有ゴールデンベクタ全件について、既定の正規化規則
/// （[`NfkcWhitespaceNormalizer`]）が学習ワーカーの `normalize_input` と
/// 同じ `normalized` を返すことを確認する。
#[test]
fn req16_default_normalizer_matches_all_shared_vectors() {
    let vectors = load_vectors();
    assert!(!vectors.is_empty(), "fixture must contain vectors");

    let normalizer = NfkcWhitespaceNormalizer;
    let mut checked = 0usize;
    for vector in &vectors {
        let name = vector
            .get("name")
            .and_then(Value::as_str)
            .expect("vector must have a string \"name\"");
        let input = vector
            .get("input")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("vector {name:?} must have a string \"input\""));
        let expected = vector
            .get("normalized")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("vector {name:?} must have a string \"normalized\""));

        let actual = normalizer.normalize(input);
        assert_eq!(
            actual.as_ref(),
            expected,
            "vector {name:?}: normalizer output disagrees with shared golden vector"
        );
        checked += 1;
    }
    assert!(checked >= 1, "expected at least one vector to be checked");
}

/// REQ-16: NFKC を要するベクタ（`fullwidth_alnum_nfkc`・`ligature_fi`・
/// `compat_hangul_parenthesized`）を名指しで具体値照合する（上の全件照合が
/// 通っても、この 3 件が fixture から欠落したら検出できないため、
/// 存在確認を兼ねて個別に確認する）。
#[test]
fn req16_nfkc_requiring_vectors_present_and_match() {
    let vectors = load_vectors();
    let normalizer = NfkcWhitespaceNormalizer;

    for (name, expected) in [("fullwidth_alnum_nfkc", "A1"), ("ligature_fi", "file")] {
        let vector = vectors
            .iter()
            .find(|v| v.get("name").and_then(Value::as_str) == Some(name))
            .unwrap_or_else(|| panic!("vector {name:?} not found in shared golden vectors"));
        let input = vector
            .get("input")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("vector {name:?} must have a string \"input\""));
        let fixture_expected = vector
            .get("normalized")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("vector {name:?} must have a string \"normalized\""));
        assert_eq!(
            fixture_expected, expected,
            "vector {name:?}: fixture's own \"normalized\" changed unexpectedly"
        );
        assert_eq!(normalizer.normalize(input).as_ref(), expected);
    }

    // `compat_hangul_parenthesized` は NFKC 互換分解を要する境界値ベクタ。
    // 内容（正解値）はテスト名・アサーションに埋め込まず fixture を正とし、
    // Rust 側の出力が fixture の記録値と一致することのみを確認する。
    let hangul_name = "compat_hangul_parenthesized";
    let vector = vectors
        .iter()
        .find(|v| v.get("name").and_then(Value::as_str) == Some(hangul_name))
        .unwrap_or_else(|| panic!("vector {hangul_name:?} not found in shared golden vectors"));
    let input = vector
        .get("input")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("vector {hangul_name:?} must have a string \"input\""));
    let expected = vector
        .get("normalized")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("vector {hangul_name:?} must have a string \"normalized\""));
    assert_eq!(normalizer.normalize(input).as_ref(), expected);
}

/// REQ-16: 本 crate が使う `unicode-normalization` の Unicode 版
/// （`UNICODE_VERSION`）が、共有フィクスチャの `_meta.unicode_version`
/// （Python 側実行環境の UCD 版）と一致することを確認する（学習ワーカーと
/// 版を揃える契約。`crates/data/src/normalize.rs` モジュール doc
/// 「Unicode 版を学習ワーカーと揃える」参照。Python 側の一致は
/// `trainer/tests/test_encoding.py::test_fixture_unicode_version_matches_runtime`
/// が別途確認する）。
#[test]
fn req16_unicode_normalization_version_matches_fixture_meta() {
    let root = load_root();
    let recorded = root
        .get("_meta")
        .and_then(|m| m.get("unicode_version"))
        .and_then(Value::as_str)
        .expect("fixture must have _meta.unicode_version");
    assert_eq!(recorded, "15.0.0");

    let (major, minor, patch) = unicode_normalization::UNICODE_VERSION;
    let actual = format!("{major}.{minor}.{patch}");
    assert_eq!(
        actual, recorded,
        "unicode-normalization's UNICODE_VERSION must match the fixture's recorded Python UCD version"
    );
}
