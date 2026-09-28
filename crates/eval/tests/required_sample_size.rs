//! McNemar 検定の必要件数の事前計算（Connor 式）の結合テスト
//! （REQ-25 異常系・TASK-25.2・issue #66）。
//!
//! [`fandhe_edge_eval::sample_size::required_sample_size_mcnemar`] の算出値
//! と、[`fandhe_edge_eval::significance::compare_with_baseline`] を通した
//! 端から端までの判定不能分岐を確認する。前提（判定不能分岐そのものの実装）
//! は TASK-25.1-2（issue #65・PR #219）で完了済みで、本テストは「事前計算
//! した必要件数を使っても、件数不足なら合格扱いにならない」ことを固定する。
//!
//! `required_sample_size_mcnemar` は評価データ総件数を仮定した正確検定の
//! 実際の検出力を探索して返すため、正規近似（Connor 式）の `ceil(n)` とは
//! 一致しない（正確検定は正規近似より保守的なため、真の必要件数は正規
//! 近似を上回る。PR #230 レビュー指摘・P0 の修正）。
//!
//! 証拠の種別: テストハーネス（PoC 実測なし。判定不能分岐が実際に発火した
//! 実例は PoC に存在しない。`p_b=0.15`・`p_c=0.05`・`power=0.8` は PoC-10・
//! PoC-24 の事前登録における仮定であり、実測値ではない）。
//!
//! 参照値の独立性: `fixtures/sample_size/known_values.json`・
//! `fixtures/sample_size/PROVENANCE.md` を参照。

use fandhe_edge_eval::metrics::{EvalRecord, Outcome};
use fandhe_edge_eval::sample_size::{McNemarSampleSizeAssumption, required_sample_size_mcnemar};
use fandhe_edge_eval::significance::{
    BaselineVerdict, InsufficientSamples, PairedRecord, compare_with_baseline, correctness,
};

const ABS_EPSILON: f64 = 1e-9;
const REL_EPSILON: f64 = 1e-9;

/// 絶対誤差・相対誤差の両方を満たす場合にのみ一致とみなす
/// （`crates/eval/tests/baseline_significance.rs` と同じ方針。
/// `.claude/rules/coding-rust.md`「浮動小数の比較は許容差を明示する」）。
fn approx_eq(actual: f64, expected: f64) -> bool {
    let abs_diff = (actual - expected).abs();
    abs_diff <= ABS_EPSILON || abs_diff <= REL_EPSILON * expected.abs()
}

/// PoC-10・PoC-24 の事前登録の仮定（`p_b=0.15`・`p_c=0.05`・`power=0.8`。
/// [仮定]）から、4 通りの α で正規近似（Connor 式）の丸め前の値が
/// `fixtures/sample_size/known_values.json` の参照値と一致することを
/// 確認する。`required_sample_size_mcnemar`（正確検定の検出力探索）は
/// 正規近似の `ceil(n)`（同フィクスチャの `ceil_n`）とは一致しない
/// （正確検定は正規近似より保守的なため、真の必要件数は正規近似を上回る。
/// PR #230 レビュー指摘・P0 の修正）ため、フィクスチャとは別に期待値を
/// 固定する。
#[test]
fn poc10_poc24_required_sample_sizes_match_known_values() {
    // (alpha, 丸め前の n の参照値〔fixtures/sample_size/known_values.json
    // の `sample_sizes` と同じ値〕, 正確検定の検出力探索による必要件数
    // 〔known_values.json の `ceil_n` より常に大きい〕, ラベル)。
    let cases = [
        (0.05, 154.598_569_560_211_02, 168u64, "単純比較"),
        (
            0.025,
            187.481_807_727_266_04,
            196u64,
            "Holm m=2 最厳段（PoC-24）",
        ),
        (
            0.0125,
            220.184_654_277_274_8,
            229u64,
            "Holm m=4 最厳段（PoC-10 事前登録の下限）",
        ),
        (
            0.05 / 12.0,
            271.668_693_520_934_2,
            278u64,
            "4 候補 x 3 seed を 1 族",
        ),
    ];

    for (alpha, expected_n, expected_required, label) in cases {
        let assumption = McNemarSampleSizeAssumption::new(0.15, 0.05, alpha, 0.8)
            .unwrap_or_else(|e| panic!("case {label}: 仮定の構築に失敗: {e}"));

        let n = fandhe_edge_eval::sample_size::mcnemar_sample_size_estimate(&assumption)
            .unwrap_or_else(|e| panic!("case {label}: 推定値の算出に失敗: {e}"));
        assert!(
            approx_eq(n, expected_n),
            "case {label}: n={n} が参照値 {expected_n} と一致しない"
        );

        let required = required_sample_size_mcnemar(&assumption)
            .unwrap_or_else(|e| panic!("case {label}: 必要件数の算出に失敗: {e}"));
        assert_eq!(
            required.get(),
            expected_required,
            "case {label}: 必要件数が期待値と一致しない"
        );
    }
}

/// `gold`・`candidate`・`baseline` の 3 本を、指定の内訳
/// （`b_candidate_only`・`c_baseline_only`・`both_correct`）から機械的に
/// 組み立てる（`baseline_significance.rs` の `build_rows` と同じ方針。
/// 乱数は使わず決定的に構築する）。
///
/// - b 行（候補のみ正解）: gold="B"・candidate="B"・baseline="A"
/// - c 行（下限基準のみ正解）: gold="A"・candidate="B"・baseline="A"
/// - both_correct 行: gold="A"・candidate="A"・baseline="A"
fn build_rows(
    b_candidate_only: u64,
    c_baseline_only: u64,
    both_correct: u64,
) -> (Vec<String>, Vec<Outcome>, Vec<Outcome>) {
    let mut gold = Vec::new();
    let mut candidate = Vec::new();
    let mut baseline = Vec::new();

    for _ in 0..b_candidate_only {
        gold.push("B".to_string());
        candidate.push(Outcome::Label("B".to_string()));
        baseline.push(Outcome::Label("A".to_string()));
    }
    for _ in 0..c_baseline_only {
        gold.push("A".to_string());
        candidate.push(Outcome::Label("B".to_string()));
        baseline.push(Outcome::Label("A".to_string()));
    }
    for _ in 0..both_correct {
        gold.push("A".to_string());
        candidate.push(Outcome::Label("A".to_string()));
        baseline.push(Outcome::Label("A".to_string()));
    }

    (gold, candidate, baseline)
}

fn build_paired_records<'a>(
    gold: &'a [String],
    candidate: &'a [Outcome],
    baseline: &'a [Outcome],
) -> Vec<PairedRecord<'a>> {
    gold.iter()
        .zip(candidate.iter())
        .zip(baseline.iter())
        .map(|((g, c), b)| PairedRecord {
            gold: g.as_str(),
            candidate: c,
            baseline: b,
        })
        .collect()
}

/// REQ-25 異常系・TASK-25.2 の端から端までの確認: 仮定
/// （`p_b=0.15`・`p_c=0.05`・`power=0.8`・`alpha=0.05`）から算出した必要件数
/// R=168 に対し、評価件数が `R - 1` 件だと `compare_with_baseline` は
/// `Undeterminable` を返し、合格扱い（`SignificantlyBetter`）にしない。
///
/// `b=30`・`c=5` は単独では `p ≈ 2.24e-5 < 0.05` かつ `b > c` で、件数さえ
/// 足りていれば `SignificantlyBetter` になる値（次の境界値テストで確認）。
#[test]
fn undeterminable_when_evaluated_count_is_one_below_required() {
    let assumption = McNemarSampleSizeAssumption::new(0.15, 0.05, 0.05, 0.8).unwrap();
    let required = required_sample_size_mcnemar(&assumption).unwrap();
    assert_eq!(required.get(), 168);

    let b = 30u64;
    let c = 5u64;
    let discordant = b + c;
    let total = required.get() - 1; // R - 1 = 167
    let both_correct = total - discordant;

    let labels = ["A", "B"];
    let (gold, candidate, baseline) = build_rows(b, c, both_correct);
    let records = build_paired_records(&gold, &candidate, &baseline);

    let comparison = compare_with_baseline(&labels, &records, required).unwrap();
    assert_eq!(comparison.counts().n, total);
    assert_eq!(comparison.counts().b_candidate_only, b);
    assert_eq!(comparison.counts().c_baseline_only, c);
    assert!(
        comparison.test().p_two_sided().value() < 0.05,
        "この p 値は件数さえ足りていれば SignificantlyBetter になる値であることの前提確認"
    );
    assert_eq!(
        comparison.verdict(),
        BaselineVerdict::Undeterminable(InsufficientSamples {
            required,
            actual: total,
        }),
        "評価件数 {total} が必要件数 {required:?} 未満のため、\
         p 値が小さくても合格扱いにならないこと（REQ-25 異常系）"
    );
}

/// 同じ `(b, c)` の内訳で、評価件数がちょうど必要件数 R=168 なら判定不能に
/// ならず、`SignificantlyBetter` になる（境界値）。
#[test]
fn significantly_better_when_evaluated_count_equals_required() {
    let assumption = McNemarSampleSizeAssumption::new(0.15, 0.05, 0.05, 0.8).unwrap();
    let required = required_sample_size_mcnemar(&assumption).unwrap();
    assert_eq!(required.get(), 168);

    let b = 30u64;
    let c = 5u64;
    let discordant = b + c;
    let total = required.get(); // R = 168
    let both_correct = total - discordant;

    let labels = ["A", "B"];
    let (gold, candidate, baseline) = build_rows(b, c, both_correct);
    let records = build_paired_records(&gold, &candidate, &baseline);

    let comparison = compare_with_baseline(&labels, &records, required).unwrap();
    assert_eq!(comparison.counts().n, total);
    assert_eq!(comparison.verdict(), BaselineVerdict::SignificantlyBetter);
}

/// `correctness` を通しても、同じ内訳の正誤ベクトルが得られることを
/// 確認する（`compare_with_baseline` 用の候補・下限基準スライスの
/// 正しさを、独立した `correctness` 呼び出しでも裏付ける）。
#[test]
fn correctness_matches_paired_breakdown() {
    let labels = ["A", "B"];
    let (gold, candidate, _baseline) = build_rows(3, 2, 5);
    let records: Vec<EvalRecord<'_>> = gold
        .iter()
        .zip(candidate.iter())
        .map(|(g, c)| EvalRecord {
            gold: g.as_str(),
            outcome: c,
        })
        .collect();

    let correct = correctness(&labels, &records).unwrap();
    // build_rows: 先頭 3 件は b 行（candidate="B"==gold="B" で正解）、
    // 次の 2 件は c 行（candidate="B" != gold="A" で不正解）、
    // 最後の 5 件は both_correct 行（candidate="A"==gold="A" で正解）。
    let expected = [true, true, true, false, false, true, true, true, true, true];
    assert_eq!(correct, expected);
}
