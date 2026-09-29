//! C1・C3 の ONNX を学習依存の無い Rust 推論ランタイムで読み、予測ラベルが学習フレームワーク
//! （MLX）内の推論と全件一致することを確認する（REQ-32 正常系・REQ-28・TASK-32.1-2・#113）。
//!
//! 証拠種別: テストハーネス（PoC-14 の実測〔ラベル一致〕を踏襲した結合テスト。実機計測ではない）。
//! fixture `fixtures/onnx_parity/` は `trainer/tools/gen_onnx_parity_fixture.py` が CPU・固定 seed の
//! 極小設定で生成した合成データ由来で、`cases.json` が MLX の予測ラベル・確率を持つ
//! （出典・除外した僅差ケースは `PROVENANCE.md`）。不一致時はコードを直し、fixture・許容差を
//! 緩めない。失敗メッセージはケース名と件数のみで入力本文を出さない（security.md）。

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_runtime::onnx::{ModelKind, load_pipeline};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

/// MLX と推論ランタイムという異なる実装間の確率差の上限（学習ワーカーの `ATOL_MLX_ONNX` と同値。
/// 許容差は広げない）。
const PROB_TOLERANCE: f64 = 1e-5;
/// `cases.json` の読み込み上限（読み込み前に確認。REQ-39 と同じ作法）。
const MAX_CASES_BYTES: u64 = 4 * 1024 * 1024;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("onnx_parity")
}

fn load_cases() -> Value {
    let path = fixture_dir().join("cases.json");
    let len = fs::metadata(&path).expect("cases metadata").len();
    assert!(len <= MAX_CASES_BYTES, "cases.json too large: {len}");
    serde_json::from_str(&fs::read_to_string(&path).expect("cases read")).expect("cases json")
}

struct Case {
    name: String,
    input: String,
    label: usize,
    probs: Vec<f64>,
}

fn cases_of(entry: &Value) -> Vec<Case> {
    entry
        .get("cases")
        .and_then(Value::as_array)
        .expect("cases array")
        .iter()
        .map(|c| Case {
            name: c["name"].as_str().expect("name").to_string(),
            input: c["input"].as_str().expect("input").to_string(),
            label: usize::try_from(c["mlx_label_index"].as_u64().expect("label")).expect("usize"),
            probs: c["mlx_probs"]
                .as_array()
                .expect("probs")
                .iter()
                .map(|p| p.as_f64().expect("prob"))
                .collect(),
        })
        .collect()
}

fn check_kind(kind: ModelKind) {
    let doc = load_cases();
    let entry = doc["kinds"][kind.as_str()].clone();
    let max_bytes =
        usize::try_from(entry["max_bytes"].as_u64().expect("max_bytes")).expect("usize");
    let sha: Sha256Digest = entry["onnx_sha256"]
        .as_str()
        .expect("sha")
        .parse()
        .expect("sha parse");
    let path = fixture_dir().join(entry["onnx"].as_str().expect("onnx"));
    let pipeline = load_pipeline(kind, max_bytes, &path, &sha).expect("load pipeline");
    let cases = cases_of(&entry);
    assert!(
        cases.len() >= 100,
        "{}: too few cases: {}",
        kind.as_str(),
        cases.len()
    );
    let distinct: std::collections::BTreeSet<usize> = cases.iter().map(|c| c.label).collect();
    assert!(distinct.len() >= 2, "{}: labels not diverse", kind.as_str());

    // 単体推論: MLX ラベルと全件一致
    // 以降の 4 種の比較（MLX ラベル・確率差・バッチ・評価器用関数）はすべて集計してから
    // 最後に 1 回だけ失敗させ、1 回の実行で全種の不一致件数を得られるようにする
    // （docs/design/runtime-batch-mismatch-procedure.md）。記録するのはケース名と件数のみ。
    let mut label_mismatched: Vec<String> = Vec::new();
    let mut score_count_mismatched: Vec<String> = Vec::new();
    let mut max_diff = 0.0f64;
    let mut singles = Vec::new();
    for c in &cases {
        let p = pipeline.infer_one(&c.input).expect("infer_one");
        if p.label_index() != c.label {
            label_mismatched.push(c.name.clone());
        }
        if p.scores().len() != c.probs.len() {
            score_count_mismatched.push(c.name.clone());
        }
        for (a, b) in p.scores().iter().zip(&c.probs) {
            max_diff = max_diff.max((a - b).abs());
        }
        singles.push(p);
    }

    // REQ-28: バッチ推論・評価器用関数が単体推論と全件一致（スコアも同一）
    let inputs: Vec<&str> = cases.iter().map(|c| c.input.as_str()).collect();
    let batch = pipeline.infer_batch(&inputs).expect("infer_batch");
    let predict = pipeline.as_predict_fn();
    let mut batch_mismatched: Vec<String> = Vec::new();
    let mut fn_mismatched: Vec<String> = Vec::new();
    for ((c, single), b) in cases.iter().zip(&singles).zip(&batch) {
        let b = b.as_ref().expect("batch item");
        if b != single {
            batch_mismatched.push(c.name.clone());
        }
        let via_fn = predict(&c.input).expect("predict fn");
        if &via_fn != single {
            fn_mismatched.push(c.name.clone());
        }
    }

    let mut failures: Vec<String> = Vec::new();
    let total = cases.len();
    for (what, list) in [
        ("labels differ from MLX", &label_mismatched),
        ("score count differs from MLX", &score_count_mismatched),
        ("batch differs from single", &batch_mismatched),
        ("predict fn differs from single", &fn_mismatched),
    ] {
        if !list.is_empty() {
            failures.push(format!("{what}: {} of {total} cases: {list:?}", list.len()));
        }
    }
    if max_diff > PROB_TOLERANCE {
        failures.push(format!("max prob diff {max_diff} exceeds {PROB_TOLERANCE}"));
    }
    assert!(
        failures.is_empty(),
        "{}: {}",
        kind.as_str(),
        failures.join("; ")
    );
}

/// REQ-32 正常系: C1 の書き出し ONNX を推論ランタイムで読み、MLX 内推論と予測ラベルが全件一致する。
#[test]
fn req32_c1_labels_match_mlx() {
    check_kind(ModelKind::C1);
}

/// REQ-32 正常系: C3 の書き出し ONNX を推論ランタイムで読み、MLX 内推論と予測ラベルが全件一致する。
#[test]
fn req32_c3_labels_match_mlx() {
    check_kind(ModelKind::C3);
}
