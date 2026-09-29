//! 再現性判定パイプラインの CPU 決定性確認（REQ-26 境界値・TASK-26.3-2・issue #105。親 #103）。
//!
//! 「seed ごとの合成予測 → `metrics::evaluate_single_select` → `SeedRun` →
//! `reproducibility::judge_reproducibility`」という評価器側の連鎖を、同一 seed で
//! 繰り返し実行しても結果が一致することを確認する。CLI の `evaluate` 工程から
//! 呼ばれる評価器の後半にあたり、乱数は seed を引数で受け取る std のみの
//! SplitMix64 で、グローバル状態・時刻・`HashMap` の反復順に依存しない。
//!
//! 証拠の種別: テストハーネス（CPU・合成データ）。実際の再学習ではなく、
//! 再学習結果の代わりに seed から決定的に生成した予測を使う。学習ワーカー側の
//! CPU 決定性は `trainer/tests/test_c1_train.py`・`test_c3_train.py` が担う。
//!
//! 限界: 事前登録条件である Mac の GPU（Metal / MLX）での 3 seed 以上の実再学習は
//! 未確認で、人間担当の作業（#103「人の対応予定」）。MLX の GPU 学習は同一 seed でも
//! 完全再現しないため、その非決定性を許容差の拡大で吸収しない（許容差は 1e-9 のまま）。
//!
//! 期待値（`correct` 件数・区間・重なりの判定）は本生成器・本 seed から導出した
//! ゴールデン値で、生成器を変更した場合は更新が必要。判定の向き（重なり / 非重なり）は
//! 生成パラメータ（誤り率の差）から期待どおりであることを各シナリオの doc に併記する。

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_eval::metrics::{EvalRecord, Outcome, evaluate_single_select};
use fandhe_edge_eval::reproducibility::{
    OVERLAP_TOLERANCE, OverlapVerdict, SeedRun, judge_reproducibility,
};
use fandhe_edge_eval::wilson::wilson_ci95;

/// 浮動小数の許容差（1e-9。coding-rust「数値・決定性」）。
const TOLERANCE: f64 = OVERLAP_TOLERANCE;
const LABELS: [&str; 3] = ["a", "b", "c"];
const EVAL_COUNT: usize = 300;

/// seed を引数で受け取る純粋な乱数（SplitMix64）。外部 crate を追加しないための実装。
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// 全 seed 共通の「同一凍結評価データ」の正解ラベル（添字の剰余）。
fn golds() -> Vec<&'static str> {
    (0..EVAL_COUNT).map(|i| LABELS[i % LABELS.len()]).collect()
}

/// 全 seed 共通の固定ハッシュ（REQ-17。値自体に意味はない）。
fn eval_data_hash() -> Sha256Digest {
    Sha256Digest::of_bytes(b"reproducibility_determinism test fixture eval data")
}

/// 再学習結果の代替となる合成予測。誤り率（千分率）の割合で誤りとし、誤りの内訳は
/// 別ラベル・`Invalid`・`Abstain`・`Error` へ決定的に割り当てる。
fn synthetic_retrain_outcomes(seed: u64, golds: &[&str], error_permille: u64) -> Vec<Outcome> {
    let mut rng = SplitMix64::new(seed);
    golds
        .iter()
        .map(|gold| {
            if rng.next_u64() % 1000 >= error_permille {
                return Outcome::Label((*gold).to_string());
            }
            match rng.next_u64() % 4 {
                0 => {
                    let other = LABELS.iter().find(|l| **l != *gold).unwrap_or(&"a");
                    Outcome::Label((*other).to_string())
                }
                1 => Outcome::Invalid,
                2 => Outcome::Abstain,
                _ => Outcome::Error,
            }
        })
        .collect()
}

/// パイプライン全体の結果（比較対象）。
#[derive(Debug, Clone, PartialEq)]
struct PipelineResult {
    corrects: Vec<u64>,
    totals: Vec<u64>,
    intervals: Vec<(f64, f64)>,
    verdict: OverlapVerdict,
    disjoint: Vec<(usize, usize)>,
    run_count: usize,
}

fn run_pipeline(scenario: &[(u64, u64)]) -> PipelineResult {
    let golds = golds();
    let mut runs = Vec::new();
    let mut corrects = Vec::new();
    let mut totals = Vec::new();
    let mut intervals = Vec::new();
    for (seed, error_permille) in scenario {
        let outcomes = synthetic_retrain_outcomes(*seed, &golds, *error_permille);
        let records: Vec<EvalRecord> = golds
            .iter()
            .zip(outcomes.iter())
            .map(|(gold, outcome)| EvalRecord { gold, outcome })
            .collect();
        let metrics = evaluate_single_select(&LABELS, &records).expect("valid synthetic records");
        let overall = &metrics.accuracy.overall;
        let (correct, total) = (overall.numerator(), overall.denominator());
        let ci = wilson_ci95(correct, total).expect("total > 0");
        corrects.push(correct);
        totals.push(total);
        intervals.push((ci.lo(), ci.hi()));
        runs.push(SeedRun {
            seed: *seed,
            correct,
            total,
            eval_data_hash: eval_data_hash(),
        });
    }
    let report = judge_reproducibility(&runs).expect("valid runs");
    PipelineResult {
        corrects,
        totals,
        intervals,
        verdict: report.verdict(),
        disjoint: report
            .disjoint_pairs()
            .iter()
            .map(|p| (p.first(), p.second()))
            .collect(),
        run_count: report.run_count(),
    }
}

/// 整数・列挙は完全一致、浮動小数は許容差 1e-9 で照合する。
fn assert_identical(a: &PipelineResult, b: &PipelineResult) {
    assert_eq!(a.corrects, b.corrects);
    assert_eq!(a.totals, b.totals);
    assert_eq!(a.verdict, b.verdict);
    assert_eq!(a.disjoint, b.disjoint);
    assert_eq!(a.run_count, b.run_count);
    assert_eq!(a.intervals.len(), b.intervals.len());
    for (x, y) in a.intervals.iter().zip(b.intervals.iter()) {
        assert!((x.0 - y.0).abs() <= TOLERANCE, "lo differs");
        assert!((x.1 - y.1).abs() <= TOLERANCE, "hi differs");
    }
}

/// (A) 誤り率がほぼ同じ 5 seed（3 seed 以上の境界より上）。全区間が重なる想定。
const SCENARIO_OVERLAP: [(u64, u64); 5] = [(0, 300), (1, 300), (2, 300), (3, 300), (4, 300)];
/// (B) 誤り率を 10% / 50% / 90% と大きく変えた 3 seed。区間が離れる想定。
const SCENARIO_DISJOINT: [(u64, u64); 3] = [(0, 100), (1, 500), (2, 900)];

/// (A) のゴールデン値（正解率は誤り率 30% から約 70%。全 seed で近く、区間が重なる）。
fn golden_overlap() -> PipelineResult {
    PipelineResult {
        corrects: vec![223, 207, 216, 206, 211],
        totals: vec![300; 5],
        intervals: vec![
            (0.6910460832802706, 0.7894674474541646),
            (0.6355381623460284, 0.7396573342548319),
            (0.6666550937163631, 0.7677817970846332),
            (0.6320966534895827, 0.7365164659779293),
            (0.649335553589633, 0.7521894515446212),
        ],
        verdict: OverlapVerdict::AllPairsOverlap,
        disjoint: vec![],
        run_count: 5,
    }
}

/// (B) のゴールデン値（正解率は約 90% / 50% / 10%。どの 2 seed も区間が離れる）。
fn golden_disjoint() -> PipelineResult {
    PipelineResult {
        corrects: vec![273, 149, 19],
        totals: vec![300; 3],
        intervals: vec![
            (0.8722221947968226, 0.9374101926050341),
            (0.440488541129282, 0.5529290817373694),
            (0.04091655604878176, 0.09679203948257407),
        ],
        verdict: OverlapVerdict::SomePairsDisjoint,
        disjoint: vec![(0, 1), (0, 2), (1, 2)],
        run_count: 3,
    }
}

/// REQ-26・TASK-26.3-2。証拠の種別: テストハーネス（CPU・合成データ）。
/// 同一 seed の 5 回の逐次実行がすべて一致し、ゴールデン値とも一致する（全区間が重なる場合）。
#[test]
fn same_seeds_repeated_runs_are_identical_on_cpu_all_overlap() {
    let golden = golden_overlap();
    for _ in 0..5 {
        assert_identical(&run_pipeline(&SCENARIO_OVERLAP), &golden);
    }
}

/// REQ-26・TASK-26.3-2。証拠の種別: テストハーネス（CPU・合成データ）。
/// 区間が離れる場合も 5 回の逐次実行が一致し、`disjoint_pairs` の添字も一致する。
#[test]
fn same_seeds_repeated_runs_are_identical_on_cpu_some_disjoint() {
    let golden = golden_disjoint();
    for _ in 0..5 {
        assert_identical(&run_pipeline(&SCENARIO_DISJOINT), &golden);
    }
}

/// REQ-26・TASK-26.3-2。証拠の種別: テストハーネス（CPU・合成データ）。
/// 4 スレッド並列でも全結果が逐次実行と一致し、グローバル状態に依存しない。
#[test]
fn same_seeds_parallel_threads_are_identical_on_cpu() {
    let baseline_overlap = run_pipeline(&SCENARIO_OVERLAP);
    let baseline_disjoint = run_pipeline(&SCENARIO_DISJOINT);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                (
                    run_pipeline(&SCENARIO_OVERLAP),
                    run_pipeline(&SCENARIO_DISJOINT),
                )
            })
        })
        .collect();
    for handle in handles {
        let (a, b) = handle.join().expect("worker thread must not panic");
        assert_identical(&a, &baseline_overlap);
        assert_identical(&b, &baseline_disjoint);
    }
}

/// REQ-26・TASK-26.3-2。seed が実際に予測へ効いていること（別 seed で `correct` が異なる）。
#[test]
fn synthetic_generator_is_seed_dependent() {
    let golden = golden_overlap();
    assert_ne!(golden.corrects[0], golden.corrects[1]);
    let g = golds();
    assert_ne!(
        synthetic_retrain_outcomes(0, &g, 300),
        synthetic_retrain_outcomes(1, &g, 300)
    );
}
