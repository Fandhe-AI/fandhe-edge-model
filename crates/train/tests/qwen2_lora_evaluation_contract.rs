//! REQ-25・REQ-26・REQ-28・REQ-41（TASK-41.10-2・issue #439）: 追加学習の種類 `qwen2_lora` を
//! 選定・評価に載せても評価契約が既定候補（c1・c3）と同じであることを確かめる結合テスト
//! （証拠種別: テストハーネス。合成の極小モデル・CPU の予測ラベルで、実重みではない。
//! 実重みでの測定は TASK-41.11・#442）。
//!
//! - 入力: `fixtures/qwen2_eval_contract/predictions.json`（生成器
//!   `trainer/tools/gen_qwen2_eval_contract_fixture.py` が学習ワーカーの実出力から記録した
//!   予測ラベル。c1・c3 は seed 0、qwen2_lora は seed 0・1・2）
//! - 統計は評価器（`fandhe-edge-eval`）と select の実経路
//!   （`assess_selection_significance`）の関数だけを使い、再実装しない
//! - 全件一致の分担: 単体＝バッチ＝学習側 MLX の一致は `crates/runtime/tests/qwen2_parity.rs`
//!   （#617）が確かめ、本ファイルは評価に使う seed 0 の予測がその fixture の MLX ラベルと
//!   一致することで、評価に使う予測へ連鎖をつなぐ
//! - CLI の `select`・`evaluate`・`infer` は qwen2_lora の成果物に未対応のため（#620 の対象外節）、
//!   本テストはライブラリ層で確かめる。CLI 配線後は `crates/cli/tests/pipeline_e2e.rs` へ
//!   移す将来仕様（REQ-41・TASK-41.10-2）

use std::collections::BTreeMap;
use std::path::PathBuf;

use fandhe_edge_core::fs::read_bounded;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_eval::baseline::fit_majority;
use fandhe_edge_eval::holm::{FamilySize, holm_adjust};
use fandhe_edge_eval::mcnemar::{mcnemar_exact_two_sided, paired_counts};
use fandhe_edge_eval::metrics::Outcome;
use fandhe_edge_eval::reproducibility::{OverlapVerdict, SeedRun, judge_reproducibility};
use fandhe_edge_eval::significance::{BaselineVerdict, RequiredSampleSize};
use fandhe_edge_train::kind_resolution::{CommonTrainParams, resolve_with_explicit_kinds};
use fandhe_edge_train::request::Device;
use fandhe_edge_train::selection_significance::{
    CandidateValidation, SelectionSignificanceInput, assess_selection_significance,
};

const MAX_FIXTURE_BYTES: u64 = 8 * 1024 * 1024;
/// 下限基準比較の必要件数。`pipeline_e2e` の `BASELINE_ASSUMPTION_REQUIRED_7`
/// （`9500/0/8000`）と同じ事前登録の仮定から求めた値（評価 120 件で足りる）。
const REQUIRED: u64 = 7;
const N: usize = 120;

struct Fixture {
    label_order: Vec<String>,
    train_labels: Vec<String>,
    gold: Vec<String>,
    case_names: Vec<String>,
    predictions: BTreeMap<String, Vec<Option<String>>>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_json(rel: &str) -> serde_json::Value {
    let bytes = read_bounded(&root().join(rel), MAX_FIXTURE_BYTES).expect("read fixture");
    serde_json::from_slice(&bytes).expect("parse fixture")
}

fn strings(v: &serde_json::Value) -> Vec<String> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_str().expect("string").to_string())
        .collect()
}

fn load() -> Fixture {
    let doc = read_json("fixtures/qwen2_eval_contract/predictions.json");
    let cases = doc["cases"].as_array().expect("cases");
    let predictions = doc["predictions"]
        .as_object()
        .expect("predictions")
        .iter()
        .map(|(k, v)| {
            let labels = v
                .as_array()
                .expect("array")
                .iter()
                .map(|x| x.as_str().map(str::to_string))
                .collect();
            (k.clone(), labels)
        })
        .collect();
    Fixture {
        label_order: strings(&doc["_meta"]["label_order"]),
        train_labels: strings(&doc["train_labels"]),
        gold: cases
            .iter()
            .map(|c| c["gold"].as_str().expect("gold").to_string())
            .collect(),
        case_names: cases
            .iter()
            .map(|c| c["name"].as_str().expect("name").to_string())
            .collect(),
        predictions,
    }
}

impl Fixture {
    fn preds(&self, key: &str) -> &[Option<String>] {
        self.predictions.get(key).expect("run key").as_slice()
    }

    /// 判定不能（`null`）は不正解として扱う（select の `validation_outcomes` と同じ規則）。
    fn outcomes(&self, key: &str) -> Vec<Outcome> {
        self.preds(key)
            .iter()
            .map(|p| p.clone().map_or(Outcome::Abstain, Outcome::Label))
            .collect()
    }

    fn correct(&self, key: &str) -> Vec<bool> {
        self.preds(key)
            .iter()
            .zip(&self.gold)
            .map(|(p, g)| p.as_deref() == Some(g.as_str()))
            .collect()
    }

    fn correct_count(&self, key: &str) -> u64 {
        self.correct(key).iter().filter(|c| **c).count() as u64
    }
}

/// 極小の p 値は絶対許容差 1e-9 では比較にならないため、相対 1e-9 で比べる。
fn assert_rel(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= expected.abs() * 1e-9,
        "actual={actual:e} expected={expected:e}"
    );
}

fn common() -> CommonTrainParams {
    CommonTrainParams {
        label_order: load().label_order,
        max_bytes: 64,
        seed: 0,
        device: Device::Cpu,
        root: "/nonexistent-root".to_string(),
        train_path: "train.jsonl".to_string(),
        out_dir: "out".to_string(),
        time_limit_seconds: None,
        rss_limit_bytes: None,
    }
}

#[test]
fn req28_req41_eval_predictions_match_parity_fixture() {
    let fx = load();
    assert_eq!(fx.case_names.len(), N);
    let parity = read_json("fixtures/qwen2_parity/cases.json");
    let cases = parity["cases"].as_array().expect("parity cases");
    assert_eq!(cases.len(), N);
    let got = fx.preds("qwen2_lora:seed0");
    let mismatched: Vec<&str> = cases
        .iter()
        .filter(|c| {
            let name = c["name"].as_str().expect("name");
            let idx = fx
                .case_names
                .iter()
                .position(|n| n == name)
                .expect("case name");
            let mlx = c["mlx_label_index"].as_u64().expect("index") as usize;
            got.get(idx).and_then(|p| p.as_deref()) != fx.label_order.get(mlx).map(String::as_str)
        })
        .map(|c| c["name"].as_str().expect("name"))
        .collect();
    assert_eq!(
        mismatched,
        Vec::<&str>::new(),
        "eval predictions differ from MLX labels"
    );
}

#[test]
fn req25_req41_qwen2_lora_mcnemar_against_majority() {
    let fx = load();
    let order: Vec<&str> = fx.label_order.iter().map(String::as_str).collect();
    let train: Vec<&str> = fx.train_labels.iter().map(String::as_str).collect();
    assert_eq!(fit_majority(&order, &train).expect("majority"), "red");
    let gold: Vec<&str> = fx.gold.iter().map(String::as_str).collect();
    let o = fx.outcomes("qwen2_lora:seed0");
    let candidates = [CandidateValidation {
        candidate_id: "qwen2_lora",
        outcomes: &o,
    }];
    let sig = assess_selection_significance(&SelectionSignificanceInput {
        label_order: &order,
        train_labels: &train,
        validation_gold: &gold,
        candidates: &candidates,
        selected_candidate_id: "qwen2_lora",
        required: RequiredSampleSize::new(REQUIRED).expect("required"),
        family_size: FamilySize::new(1).expect("family"),
    })
    .expect("assess");
    // 合成の極小モデルは majority（red）をほぼ上回らない。有意性を装わないことを具体値で固定する
    let c = sig.counts();
    assert_eq!((c.n, c.b_candidate_only, c.c_baseline_only), (120, 17, 15));
    assert_rel(sig.raw_p().value(), 0.860_050_065_908_581);
    assert_eq!(sig.verdict(), BaselineVerdict::NotSignificantlyBetter);
}

#[test]
fn req25_req41_holm_family_includes_c1_c3_and_qwen2_lora() {
    let fx = load();
    let resolution =
        resolve_with_explicit_kinds(&["qwen2_lora".to_string()], common()).expect("resolve");
    let ids: Vec<&str> = resolution
        .candidates()
        .iter()
        .map(|c| c.candidate_id.as_str())
        .collect();
    assert_eq!(ids, ["c1", "c3", "qwen2_lora"]);

    let order: Vec<&str> = fx.label_order.iter().map(String::as_str).collect();
    let train: Vec<&str> = fx.train_labels.iter().map(String::as_str).collect();
    let gold: Vec<&str> = fx.gold.iter().map(String::as_str).collect();
    let outs: Vec<Vec<Outcome>> = ["c1:seed0", "c3:seed0", "qwen2_lora:seed0"]
        .iter()
        .map(|k| fx.outcomes(k))
        .collect();
    let candidates: Vec<CandidateValidation<'_>> = ids
        .iter()
        .zip(&outs)
        .map(|(id, o)| CandidateValidation {
            candidate_id: id,
            outcomes: o,
        })
        .collect();
    let assess = |selected: &str, m: usize| {
        assess_selection_significance(&SelectionSignificanceInput {
            label_order: &order,
            train_labels: &train,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: selected,
            required: RequiredSampleSize::new(REQUIRED).expect("required"),
            family_size: FamilySize::new(m).expect("family"),
        })
    };

    // 族は 3 候補（qwen2_lora を含む）。c1 の補正後 p = 3 × 生 p、qwen2_lora は 1 × 0.86
    let c1 = assess("c1", ids.len()).expect("assess c1");
    assert_eq!(
        (c1.counts().b_candidate_only, c1.counts().c_baseline_only),
        (77, 1)
    );
    assert_rel(c1.raw_p().value(), 5.227_781_471_335_135e-22);
    assert_rel(c1.adjusted_p().value(), 3.0 * 5.227_781_471_335_135e-22);
    assert_eq!(c1.verdict(), BaselineVerdict::SignificantlyBetter);
    let q = assess("qwen2_lora", ids.len()).expect("assess qwen2_lora");
    assert_rel(q.adjusted_p().value(), 0.860_050_065_908_581);
    assert_eq!(q.verdict(), BaselineVerdict::NotSignificantlyBetter);
    // 事前登録の族サイズを 1 つ増やす（脱落候補）と補正後 p が保守側に動く。候補数未満は拒否
    let wider = assess("c1", ids.len() + 1).expect("assess c1 wider");
    assert_rel(wider.adjusted_p().value(), 4.0 * 5.227_781_471_335_135e-22);
    assert!(assess("c1", ids.len() - 1).is_err());

    // 種類どうし: qwen2_lora は c1・c3 を有意に上回らない（c > b）。族サイズ 2 で Holm 補正する
    let pair = |other: &str| {
        let counts = paired_counts(&fx.correct("qwen2_lora:seed0"), &fx.correct(other))
            .expect("paired counts");
        (
            counts.b_candidate_only,
            counts.c_baseline_only,
            mcnemar_exact_two_sided(counts.b_candidate_only, counts.c_baseline_only)
                .expect("mcnemar"),
        )
    };
    let (b1, c1n, t1) = pair("c1:seed0");
    let (b3, c3n, t3) = pair("c3:seed0");
    assert_eq!((b1, c1n, b3, c3n), (6, 80, 4, 70));
    let adj = holm_adjust(
        &[t1.p_two_sided(), t3.p_two_sided()],
        FamilySize::new(2).expect("m"),
    )
    .expect("holm");
    assert_rel(adj[0].value(), 2.0 * 1.311_109_746_589_136e-17);
    assert_rel(adj[1].value(), 1.289_846_949_002_271_6e-16);
    assert!(adj.iter().all(|p| p.value() < 0.05) && b1 < c1n && b3 < c3n);
}

#[test]
fn req26_req41_qwen2_lora_three_seed_wilson_overlap() {
    let fx = load();
    // 全 run が同じ凍結評価データに適用された共通ハッシュ（ケース名と gold から求める）
    let data = fx
        .case_names
        .iter()
        .zip(&fx.gold)
        .map(|(n, g)| format!("{n}:{g}\n"))
        .collect::<String>();
    let hash = Sha256Digest::of_bytes(data.as_bytes());
    let run = |seed: u64| SeedRun {
        seed,
        correct: fx.correct_count(&format!("qwen2_lora:seed{seed}")),
        total: N as u64,
        eval_data_hash: hash,
    };
    let runs = [run(0), run(1), run(2)];
    assert_eq!(runs.map(|r| r.correct), [17, 12, 15]);
    let report = judge_reproducibility(&runs).expect("reproducibility");
    assert_eq!(report.run_count(), 3);
    assert_eq!(report.verdict(), OverlapVerdict::AllPairsOverlap);

    // 空振りでないことの確認: 1 run を全問正解にすると区間が離れる
    let mut altered = runs;
    altered[1].correct = N as u64;
    let report = judge_reproducibility(&altered).expect("reproducibility");
    assert_eq!(report.verdict(), OverlapVerdict::SomePairsDisjoint);
    let pairs: Vec<(usize, usize)> = report
        .disjoint_pairs()
        .iter()
        .map(|p| (p.first(), p.second()))
        .collect();
    assert_eq!(pairs, [(0, 1), (1, 2)]);
}
