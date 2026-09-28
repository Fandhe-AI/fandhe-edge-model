//! REQ-23 境界値・TASK-23.2（issue #57）: 空入力の前処理食い違いの検知・報告の
//! 結合テスト。
//!
//! 学習ワーカー（`trainer/src/fandhe_edge_trainer/encoding.py`）の
//! `normalize_input`・`encode_bytes` に対する SSOT ゴールデンベクタ
//! （`fixtures/preprocess/byte_encoding_vectors.json`。Chore #10）を
//! `crates/data/src/preprocess_boundary.rs` から読み、学習ワーカー経路の
//! 「正規化後に空となる入力は詰め物トークン `[0]` を返す」契約を Rust 側でも
//! 固定する。**これは Rust 推論ランタイムの実装が無い本 TASK 時点では
//! 「Rust 実装の検証」ではなく「SSOT 契約の Rust 側での固定」である**
//! （`trainer/tests/test_encoding.py` が Python 側から同じベクタを照合して
//! いる。Rust 推論ランタイムを作成したうえでの実地の一致検証は Chore #10
//! Deliverable B・REQ-28 の担当）。
//!
//! 証拠の種別: テストハーネス（実機測定なし）。

use std::fs;
use std::path::PathBuf;

use fandhe_edge_data::preprocess_boundary::{
    EmptyInputConsistency, EmptyInputEncoding, classify_empty_input_encoding,
    compare_empty_input_encodings, is_empty_after_normalization,
};
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

/// fixture を読み込み `serde_json::Value` へパースする。data crate は
/// `serde` derive を持たない（`Cargo.toml` 参照）ため、依存追加を避けて
/// `Value` で扱う。
fn load_vectors() -> Vec<Value> {
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
    let root: Value = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("failed to parse preprocess vectors fixture {path:?}: {e}"));
    root.get("vectors")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("preprocess vectors fixture {path:?} has no \"vectors\" array"))
        .clone()
}

fn find_vector<'a>(vectors: &'a [Value], name: &str) -> &'a Value {
    vectors
        .iter()
        .find(|v| v.get("name").and_then(Value::as_str) == Some(name))
        .unwrap_or_else(|| panic!("vector {name:?} not found in preprocess vectors fixture"))
}

fn ids_as_i64(vector: &Value) -> Vec<i64> {
    vector
        .get("ids")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("vector has no \"ids\" array: {vector:?}"))
        .iter()
        .map(|v| {
            v.as_i64()
                .unwrap_or_else(|| panic!("id is not an i64: {v:?}"))
        })
        .collect()
}

/// REQ-23 境界値・TASK-23.2: SSOT ベクタ `empty_input` について、
/// - `normalized == ""`
/// - `is_empty_after_normalization(input)` が真
/// - `ids`（学習ワーカー経路。本リポでは既に `[0]`）を
///   [`classify_empty_input_encoding`] で分類すると `PaddingOnly { len: 1 }`
/// - PoC-16 時点の Rust 側既知値 `[0]` と比較すると `Consistent` になる
///   （両方とも `[0]` のため。「学習ワーカー経路の契約を Rust 側で固定する」
///   ことの確認であり、Rust 推論ランタイムの実装検証ではない）
///
/// であることを固定する。
#[test]
fn req23_empty_input_vector_matches_padding_only_contract() {
    let vectors = load_vectors();
    let vector = find_vector(&vectors, "empty_input");

    let input = vector
        .get("input")
        .and_then(Value::as_str)
        .expect("empty_input vector must have a string \"input\"");
    let normalized = vector
        .get("normalized")
        .and_then(Value::as_str)
        .expect("empty_input vector must have a string \"normalized\"");
    assert_eq!(normalized, "");
    assert!(is_empty_after_normalization(input));

    let ids = ids_as_i64(vector);
    assert_eq!(
        classify_empty_input_encoding(&ids),
        Ok(EmptyInputEncoding::PaddingOnly { len: 1 })
    );

    // PoC-16 時点の既知値（Rust 側 `[0]`）との比較。本リポに Rust 推論
    // ランタイムは無いため、ここでの `inference` 引数は「PoC-16 で実測された
    // 値」を表す固定値であり、本リポの実装から得た値ではない
    // （モジュール doc・`crates/data/src/preprocess_boundary.rs` モジュール
    // doc「本リポの現状」参照）。
    let poc16_rust_side_ids: [i64; 1] = [0];
    assert_eq!(
        compare_empty_input_encodings(&poc16_rust_side_ids, &ids),
        Ok(EmptyInputConsistency::Consistent(
            EmptyInputEncoding::PaddingOnly { len: 1 }
        ))
    );
}

/// PoC-16 で実測された食い違い（Rust 側 `[0]`・Python 側 `train_mlx.encode`
/// の `[]`）を、[`compare_empty_input_encodings`] が `Diverged` として検知
/// できることを固定する（検知の結合テスト。両方とも本リポの実装値ではなく
/// PoC-16 時点の既知値）。
#[test]
fn req23_detects_poc16_known_divergence_between_paths() {
    let poc16_rust_side_ids: [i64; 1] = [0];
    let poc16_python_side_ids: [i64; 0] = [];
    let result =
        compare_empty_input_encodings(&poc16_rust_side_ids, &poc16_python_side_ids).unwrap();
    assert_eq!(
        result,
        EmptyInputConsistency::Diverged {
            inference: EmptyInputEncoding::PaddingOnly { len: 1 },
            evaluation: EmptyInputEncoding::EmptySequence,
        }
    );
    assert_eq!(result.code(), "empty_input_preprocess_divergence");
}

/// SSOT ベクタ全件について「`normalized` が空文字列 ⇔
/// `is_empty_after_normalization(input)` が真」が一致することを照合する
/// （Python 実装〔`trainer/src/fandhe_edge_trainer/encoding.py`〕の空判定と
/// Rust 側の空判定がずれていないことの確認）。
#[test]
fn req23_is_empty_after_normalization_matches_all_vectors() {
    let vectors = load_vectors();
    assert!(!vectors.is_empty(), "fixture must contain vectors");
    for vector in &vectors {
        let name = vector
            .get("name")
            .and_then(Value::as_str)
            .expect("vector must have a string \"name\"");
        let input = vector
            .get("input")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("vector {name:?} must have a string \"input\""));
        let normalized = vector
            .get("normalized")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("vector {name:?} must have a string \"normalized\""));

        assert_eq!(
            is_empty_after_normalization(input),
            normalized.is_empty(),
            "vector {name:?}: is_empty_after_normalization disagrees with normalized emptiness"
        );
    }
}

/// SSOT ベクタのうち `normalized` が空文字列であるもの（`empty_input` に
/// 加え、情報分離文字のみ・空白文字のみの境界値ベクタ）は、いずれも
/// 学習ワーカー経路の契約どおり `PaddingOnly { len: 1 }`（`[0]`）に分類される
/// ことを確認する。
#[test]
fn req23_all_empty_normalized_vectors_classify_as_padding_only_len_one() {
    let vectors = load_vectors();
    let mut checked = 0usize;
    for vector in &vectors {
        let normalized = vector
            .get("normalized")
            .and_then(Value::as_str)
            .expect("vector must have a string \"normalized\"");
        if !normalized.is_empty() {
            continue;
        }
        let name = vector
            .get("name")
            .and_then(Value::as_str)
            .expect("vector must have a string \"name\"");
        let ids = ids_as_i64(vector);
        assert_eq!(
            classify_empty_input_encoding(&ids),
            Ok(EmptyInputEncoding::PaddingOnly { len: 1 }),
            "vector {name:?}: expected PaddingOnly {{ len: 1 }}"
        );
        checked += 1;
    }
    // fixture が空白のみベクタを含まなくなる将来の変更を検出するため、
    // 実際に 1 件以上照合したことを確認する（0 件通過を「検証済み」と
    // 報告しない。`.claude/rules/ci.md`）。
    assert!(
        checked >= 1,
        "expected at least one vector with empty \"normalized\""
    );
}
