//! `qwen2_lora` の自己完結パッケージを学習依存の無い Rust 推論ランタイムで読み、予測ラベルが学習側
//! （MLX 内推論）と全件一致することを確認する（REQ-41・REQ-32 正常系・REQ-28・TASK-41.6-5・#434）。
//!
//! 証拠種別: テストハーネス（合成の極小ベースモデル・CPU。実重みでの照合ではない。実重みは
//! TASK-41.11〔#442〕の実機作業）。fixture `fixtures/qwen2_parity/` は
//! `trainer/tools/gen_qwen2_parity_fixture.py` が学習ワーカーの CLI で生成したパッケージそのもので、
//! `cases.json` が MLX の予測ラベルと ReferenceEvaluator の確率を持つ（出典・除外した僅差ケースは
//! `PROVENANCE.md`）。不一致時はコードを直し、fixture・許容差を緩めない。失敗メッセージは
//! ケース名と件数のみで入力本文を出さない（security.md）。

#[path = "common/qwen2_package.rs"]
#[allow(dead_code)] // 共有の読み込み関数のうち本テストで使わないものがある
mod qwen2_package;

#[path = "common/proto_builder.rs"]
#[allow(dead_code)] // 共有の編集器のうち本テストで使わない関数がある
mod proto_builder;

use fandhe_edge_runtime::onnx::{ModelKind, OnnxBackend, OnnxLoadError};
use fandhe_edge_runtime::pipeline::{ScoringBackend, TokenIds};
use qwen2_package::{load_cases, load_pipeline, read_package};

/// 異なる実装間（ReferenceEvaluator と自作ランタイム）の確率差の上限（学習ワーカーの
/// `ATOL_MLX_ONNX` と同値。許容差は広げない）。
const PROB_TOLERANCE: f64 = 1e-5;
/// 評価器用関数の照合件数（上の理由で先頭のみ）。
const PREDICT_FN_CASES: usize = 20;

/// REQ-41・REQ-32・REQ-28: 単体・バッチ・評価器用関数の予測ラベルが MLX と全件一致し、確率が
/// ReferenceEvaluator と許容差内で一致する。
#[test]
fn req41_qwen2_lora_runtime_matches_training_side_for_all_cases() {
    let (pipeline, k) = load_pipeline();
    let cases = load_cases();
    assert!(cases.len() >= 100, "too few cases: {}", cases.len());
    let distinct: std::collections::BTreeSet<usize> = cases.iter().map(|c| c.label).collect();
    assert!(distinct.len() >= 2, "labels not diverse");

    let mut label_mismatched: Vec<String> = Vec::new();
    let mut score_count_mismatched: Vec<String> = Vec::new();
    let mut max_diff = 0.0f64;
    let mut singles = Vec::new();
    for c in &cases {
        let p = pipeline.infer_one(&c.input).expect("infer_one");
        if p.label_index() != c.label {
            label_mismatched.push(c.name.clone());
        }
        if p.scores().len() != k || c.probs.len() != k {
            score_count_mismatched.push(c.name.clone());
        }
        for (a, b) in p.scores().iter().zip(&c.probs) {
            max_diff = max_diff.max((a - b).abs());
        }
        singles.push(p);
    }

    let inputs: Vec<&str> = cases.iter().map(|c| c.input.as_str()).collect();
    let batch = pipeline.infer_batch(&inputs).expect("infer_batch");
    let predict = pipeline.as_predict_fn();
    let mut batch_mismatched: Vec<String> = Vec::new();
    let mut fn_mismatched: Vec<String> = Vec::new();
    for (i, ((c, single), b)) in cases.iter().zip(&singles).zip(&batch).enumerate() {
        if b.as_ref().expect("batch item") != single {
            batch_mismatched.push(c.name.clone());
        }
        // 評価器用関数は単体推論と同じ経路の薄い包み。debug ビルドでの実行時間のため先頭
        // `PREDICT_FN_CASES` 件だけ照合する（単体・バッチは全件）
        if i < PREDICT_FN_CASES && &predict(&c.input).expect("predict fn") != single {
            fn_mismatched.push(c.name.clone());
        }
    }

    let total = cases.len();
    let mut failures: Vec<String> = Vec::new();
    for (what, list) in [
        ("labels differ from MLX", &label_mismatched),
        ("score count differs", &score_count_mismatched),
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
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// REQ-39: 外部データを持つ種類は shard 無しの経路では読めず、持たない種類は shard 経路で読めない
/// （C1・C3 の経路から誤って実行させない。fail-closed）。
#[test]
fn req39_external_data_kind_is_loadable_only_with_shards() {
    let (bytes, shards) = read_package();
    assert!(matches!(
        OnnxBackend::from_bytes(&bytes, ModelKind::Qwen2Lora),
        Err(OnnxLoadError::UnsupportedKind)
    ));
    assert!(matches!(
        OnnxBackend::from_bytes_with_shards(&bytes, ModelKind::C1, shards),
        Err(OnnxLoadError::UnsupportedKind)
    ));
}

/// REQ-39: shard の sha256 が期待値と一致しなければ読み込みを拒否する。
#[test]
fn req39_shard_hash_mismatch_is_rejected() {
    let (bytes, mut shards) = read_package();
    if let Some(s) = shards.first_mut() {
        s.sha256 = fandhe_edge_core::hash::Sha256Digest::of_bytes(b"tampered");
    }
    assert!(matches!(
        OnnxBackend::from_bytes_with_shards(&bytes, ModelKind::Qwen2Lora, shards),
        Err(OnnxLoadError::IntegrityMismatch)
    ));
}

/// REQ-39: 許可リスト外の演算を含むグラフは読み込み時に拒否する。
#[test]
fn req39_unsupported_op_is_rejected_at_load() {
    let (bytes, shards) = read_package();
    let edited = proto_builder::set_op_type(&bytes, 0, "Erf");
    assert!(matches!(
        OnnxBackend::from_bytes_with_shards(&edited, ModelKind::Qwen2Lora, shards),
        Err(OnnxLoadError::UnsupportedGraph)
    ));
}

/// REQ-39: 語彙外の token id（負値・語彙数以上）と空列は推論時に拒否する。
#[test]
fn req39_out_of_vocabulary_ids_are_rejected() {
    let (bytes, shards) = read_package();
    let backend = OnnxBackend::from_bytes_with_shards(&bytes, ModelKind::Qwen2Lora, shards)
        .expect("load qwen2_lora");
    for ids in [vec![-1], vec![i64::MAX], vec![1, 100_000]] {
        assert!(backend.scores(&TokenIds::new(ids)).is_err());
    }
    assert!(backend.scores(&TokenIds::new(Vec::new())).is_err());
}
