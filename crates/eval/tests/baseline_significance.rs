//! 下限基準（majority）比較への接続の結合テスト（REQ-25 正常系・
//! TASK-25.1-2・issue #65）。
//!
//! [`fandhe_edge_eval::baseline::fit_majority`]（学習データからの多数決
//! ラベル生成）と [`fandhe_edge_eval::significance::compare_with_baseline`]
//! （McNemar 検定＋α=0.05 での有意性判定への接続）を、PoC-10 `eval_incat`
//! （n=650・majority の正解数=119）の分割表（`b`・`c`）を再現した合成データで
//! 端から端まで確認する。
//!
//! `b`・`c` は PoC-10 実測の分割表を転記し、期待 p 値は
//! `fixtures/mcnemar/generate_known_values.py`（`fractions.Fraction` による
//! 独立な厳密計算）で求め直した値を使う（`Case(143, 43, ...)` 等。
//! `fixtures/mcnemar/PROVENANCE.md` 参照）。証拠の種別: テストハーネス
//! （b・c は PoC-10 実測値の転記、p は厳密計算による独立照合）。
//!
//! データの生成は決定的（乱数を使わない）。行の並びは件数（both_correct・
//! b・c・both_wrong）だけから機械的に組み立てる。

use fandhe_edge_eval::baseline::fit_majority;
use fandhe_edge_eval::metrics::{EvalRecord, Outcome};
use fandhe_edge_eval::sample_size::{McNemarSampleSizeAssumption, required_sample_size_mcnemar};
use fandhe_edge_eval::significance::{
    BaselineVerdict, PairedRecord, RequiredSampleSize, compare_with_baseline,
};

/// PoC-10 の事前登録手続きで算出された必要件数を、算出関数
/// （[`required_sample_size_mcnemar`]。TASK-25.2・issue #66）で求め直した値。
/// 仮定（`p_b=0.15`・`p_c=0.05`・`power=0.8`・`alpha=0.0125`。Holm m=4
/// 最厳段。`fixtures/sample_size/known_values.json` の
/// `holm_m4`〔PoC-10 事前登録の下限〕相当）は PoC-10 の事前登録値であり、
/// 算出結果が PoC-10 の記録どおり 221 になることを
/// `poc10_required_sample_size_is_221` で確認する（転記ではなく算出値を
/// 使うことを端から端まで示す。証拠の種別: テストハーネス）。
fn poc10_required_sample_size() -> u64 {
    let assumption = McNemarSampleSizeAssumption::new(0.15, 0.05, 0.0125, 0.8)
        .expect("PoC-10 の仮定は McNemarSampleSizeAssumption の検証を満たす");
    required_sample_size_mcnemar(&assumption)
        .expect("PoC-10 の仮定から算出した必要件数は MAX_EVAL_RECORDS 以内")
        .get()
}

/// [`poc10_required_sample_size`] が PoC-10 の事前登録値（221）と一致する
/// ことを固定する（REQ-25・TASK-25.2・issue #66。
/// `fixtures/sample_size/known_values.json` の `ceil_n=221` と同じ値）。
#[test]
fn poc10_required_sample_size_is_221() {
    assert_eq!(poc10_required_sample_size(), 221);
}

const ABS_EPSILON: f64 = 1e-9;
const REL_EPSILON: f64 = 1e-9;

/// 絶対誤差・相対誤差の両方を満たす場合にのみ一致とみなす
/// （`crates/eval/tests/mcnemar_known_answer.rs` と同じ方針）。
///
/// 期待値が 0.0 の既知値（f64 のアンダーフロー境界など）は許容差を設けず、
/// 実値も厳密に 0.0 であることを要求する。絶対誤差だけで判定すると
/// 1e-9 未満の任意の非ゼロ値を通してしまい、既知値を検証できないため。
fn approx_eq(actual: f64, expected: f64) -> bool {
    let abs_diff = (actual - expected).abs();
    if expected == 0.0 {
        return actual == 0.0;
    }
    let rel_diff = abs_diff / expected.abs();
    abs_diff < ABS_EPSILON && rel_diff < REL_EPSILON
}

/// gold が `"A"`（下限基準＝majority が正解する行）で候補が不正解になる
/// 予測を、件数の位置から決定的に生成する。`Label` 以外（`Invalid`・
/// `Abstain`・`Error`）も混ぜ、それらがすべて「不正解」扱いになることを
/// 同時に確認する。
fn wrong_outcome_for_gold_a(i: u64) -> Outcome {
    match i % 4 {
        0 => Outcome::Label("B".to_string()),
        1 => Outcome::Invalid,
        2 => Outcome::Abstain,
        _ => Outcome::Error,
    }
}

/// gold が `"B"`（下限基準が不正解の行）で候補も不正解になる予測を、
/// 件数の位置から決定的に生成する（`wrong_outcome_for_gold_a` と同様）。
fn wrong_outcome_for_gold_b(i: u64) -> Outcome {
    match i % 4 {
        0 => Outcome::Label("C".to_string()),
        1 => Outcome::Invalid,
        2 => Outcome::Abstain,
        _ => Outcome::Error,
    }
}

/// `PairedCounts`（both_correct・b・c・both_wrong）の内訳から、対応する
/// gold・候補・下限基準の予測列を決定的に組み立てる。
///
/// 下限基準は常に `majority`（[`fit_majority`] が学習データから求めた
/// 多数決ラベルの戻り値。呼び出し側から渡す）を予測する多数決分類器を模す
/// （gold が `majority` の行でのみ正解する。Codex 指摘。PR #219。以前は
/// `"A"` を関数内でハードコードしており、[`fit_majority`] の戻り値とは
/// 独立に決め打ちしていた）。
///
/// - both_correct 行: gold=`majority`、候補・下限基準ともに `Label(majority)`
///   （正解）
/// - c 行（下限基準のみ正解）: gold=`majority`、下限基準は `Label(majority)`
///   （正解）、候補は不正解（[`wrong_outcome_for_gold_a`]）
/// - b 行（候補のみ正解）: gold=`"B"`、候補は `Label("B")`（正解）、
///   下限基準は `Label(majority)`（不正解。`majority != "B"` が前提）
/// - both_wrong 行: gold=`"B"`、候補は不正解（[`wrong_outcome_for_gold_b`]）、
///   下限基準は `Label(majority)`（不正解）
fn build_rows(
    majority: &'static str,
    both_correct: u64,
    b: u64,
    c: u64,
    both_wrong: u64,
) -> (Vec<&'static str>, Vec<Outcome>, Vec<Outcome>) {
    assert_ne!(
        majority, "B",
        "b・both_wrong 行の gold=\"B\" は majority と異なる前提"
    );

    let mut gold = Vec::new();
    let mut candidate = Vec::new();
    let mut baseline = Vec::new();

    for _ in 0..both_correct {
        gold.push(majority);
        candidate.push(Outcome::Label(majority.to_string()));
        baseline.push(Outcome::Label(majority.to_string()));
    }
    for i in 0..c {
        gold.push(majority);
        candidate.push(wrong_outcome_for_gold_a(i));
        baseline.push(Outcome::Label(majority.to_string()));
    }
    for _ in 0..b {
        gold.push("B");
        candidate.push(Outcome::Label("B".to_string()));
        baseline.push(Outcome::Label(majority.to_string()));
    }
    for i in 0..both_wrong {
        gold.push("B");
        candidate.push(wrong_outcome_for_gold_b(i));
        baseline.push(Outcome::Label(majority.to_string()));
    }

    (gold, candidate, baseline)
}

fn build_paired_records<'a>(
    gold: &'a [&'a str],
    candidate: &'a [Outcome],
    baseline: &'a [Outcome],
) -> Vec<PairedRecord<'a>> {
    gold.iter()
        .zip(candidate.iter())
        .zip(baseline.iter())
        .map(|((&g, c), b)| PairedRecord {
            gold: g,
            candidate: c,
            baseline: b,
        })
        .collect()
}

/// PoC-10 `eval_incat`（n=650・majority 正解数=119）の 8 通りの分割表
/// （C1・C2 は 3 seed とも同一〔決定的な学習パイプラインのため〕、C3・C4 は
/// seed ごとに異なる。4 方式 × 3 seed = 12 組のうち、C1・C2 は値が重複する
/// ため実データは 8 通り）。期待 p は
/// `fixtures/mcnemar/generate_known_values.py` の厳密計算で求めた値。
struct Case {
    label: &'static str,
    both_correct: u64,
    b: u64,
    c: u64,
    both_wrong: u64,
    expected_p: f64,
}

const CASES: &[Case] = &[
    Case {
        label: "C1 (seed 0/1/2)",
        both_correct: 90,
        b: 130,
        c: 29,
        both_wrong: 401,
        expected_p: 1.8055891875528008e-16,
    },
    Case {
        label: "C2 (seed 0/1/2)",
        both_correct: 93,
        b: 123,
        c: 26,
        both_wrong: 408,
        expected_p: 2.748735875495548e-16,
    },
    Case {
        label: "C3 seed0",
        both_correct: 67,
        b: 161,
        c: 52,
        both_wrong: 370,
        expected_p: 3.664901027505431e-14,
    },
    Case {
        label: "C3 seed1",
        both_correct: 76,
        b: 143,
        c: 43,
        both_wrong: 388,
        expected_p: 9.52366198189899e-14,
    },
    Case {
        label: "C3 seed2",
        both_correct: 79,
        b: 144,
        c: 40,
        both_wrong: 387,
        expected_p: 5.518207697056318e-15,
    },
    Case {
        label: "C4 seed0",
        both_correct: 83,
        b: 108,
        c: 36,
        both_wrong: 423,
        expected_p: 1.4957587495764933e-9,
    },
    Case {
        label: "C4 seed1",
        both_correct: 61,
        b: 125,
        c: 58,
        both_wrong: 406,
        expected_p: 8.13607591316581e-7,
    },
    Case {
        label: "C4 seed2",
        both_correct: 61,
        b: 134,
        c: 58,
        both_wrong: 397,
        expected_p: 4.179048819596723e-8,
    },
];

/// 受入基準: PoC-10 相当のデータで、4 方式・3 seed（C1・C2 は値が重複する
/// ため 8 通り）のすべてが「有意」（`SignificantlyBetter`）と判定される
/// （REQ-25・TASK-25.1-2・issue #65）。
///
/// Codex 指摘（PR #219）: 下限基準の予測列は [`fit_majority`] の戻り値
/// （学習ラベルから求めた多数決ラベル）で生成し、その列を
/// [`compare_with_baseline`] へ渡す。以前は `build_rows` 内部で `"A"` を
/// 直接埋め込んでおり、`fit_majority` の結果とは独立に決め打ちしていた。
#[test]
fn all_poc10_cases_are_significantly_better_than_majority() {
    let labels = ["A", "B", "C"];
    // 学習データでは "A" が最頻になるようにする（[`fit_majority`] が
    // 実際に "A" を選ぶことをテストで確認したうえで `build_rows` へ渡す。
    // 学習データの割合はここでの分割表〔both_correct・b・c・both_wrong〕とは
    // 無関係。REQ-17・REQ-27: 下限基準は学習データのみから決め、評価データの
    // gold・分割表からは求めない）。
    let train_labels = ["A", "A", "A", "B", "C"];
    let majority = fit_majority(&labels, &train_labels).unwrap();
    assert_eq!(
        majority, "A",
        "build_rows は majority に対応する gold 行を組み立てるため、\
         想定外のラベルに変わっていないことを確認する"
    );

    for case in CASES {
        let (gold, candidate, baseline) =
            build_rows(majority, case.both_correct, case.b, case.c, case.both_wrong);
        let records = build_paired_records(&gold, &candidate, &baseline);

        let n = case.both_correct + case.b + case.c + case.both_wrong;
        assert_eq!(
            n, 650,
            "case {}: n は PoC-10 eval_incat と同じ 650 件",
            case.label
        );

        let required = RequiredSampleSize::new(poc10_required_sample_size()).unwrap();
        let comparison = compare_with_baseline(&labels, &records, required)
            .unwrap_or_else(|e| panic!("case {}: 比較に失敗: {e}", case.label));

        let counts = comparison.counts();
        assert_eq!(counts.n, n, "case {}: n が一致しない", case.label);
        assert_eq!(
            counts.both_correct, case.both_correct,
            "case {}: both_correct が一致しない",
            case.label
        );
        assert_eq!(
            counts.b_candidate_only, case.b,
            "case {}: b が一致しない",
            case.label
        );
        assert_eq!(
            counts.c_baseline_only, case.c,
            "case {}: c が一致しない",
            case.label
        );
        assert_eq!(
            counts.both_wrong, case.both_wrong,
            "case {}: both_wrong が一致しない",
            case.label
        );

        assert_eq!(
            comparison.candidate_correct(),
            case.both_correct + case.b,
            "case {}: candidate_correct が一致しない",
            case.label
        );
        assert_eq!(
            comparison.baseline_correct(),
            119,
            "case {}: baseline_correct は PoC-10 の majority 正解数 119 と一致するはず",
            case.label
        );

        let p = comparison.test().p_two_sided().value();
        assert!(
            approx_eq(p, case.expected_p),
            "case {}: p 値が期待値と一致しない（actual={p}, expected={}）",
            case.label,
            case.expected_p
        );

        assert_eq!(
            comparison.verdict(),
            BaselineVerdict::SignificantlyBetter,
            "case {}: 有意に上回ると判定されるはず",
            case.label
        );
    }
}

/// [`fit_majority`] は学習データのラベルだけから `"A"` を選ぶ
/// （評価データの gold からは求めない。REQ-17・REQ-27）。
#[test]
fn majority_is_fit_from_train_labels_only() {
    let labels = ["A", "B", "C"];
    // 評価データ側は gold の大半が "B"（both_wrong・b 行）だが、
    // 学習データでは "A" が最頻になるようにする。
    let train_labels = ["A", "A", "A", "B", "C"];
    let majority = fit_majority(&labels, &train_labels).unwrap();
    assert_eq!(majority, "A");
}

/// 候補が下限基準より有意に劣る合成ケースは `NotSignificantlyBetter` になる
/// （方向の確認。b < c なので `judge` の規則上 Better にはならない）。
#[test]
fn candidate_worse_than_majority_is_not_significantly_better() {
    let labels = ["A", "B", "C"];
    // b=3, c=12（下限基準のほうが強い）。
    let (gold, candidate, baseline) = build_rows("A", 50, 3, 12, 50);
    let records = build_paired_records(&gold, &candidate, &baseline);

    // 必要件数は本テストの主題（方向の確認）とは無関係のため最小値にする。
    let required = RequiredSampleSize::new(1).unwrap();
    let comparison = compare_with_baseline(&labels, &records, required).unwrap();
    assert_eq!(comparison.counts().b_candidate_only, 3);
    assert_eq!(comparison.counts().c_baseline_only, 12);
    assert_eq!(
        comparison.verdict(),
        BaselineVerdict::NotSignificantlyBetter
    );
}

/// 評価契約（REQ-27）: `compare_with_baseline` の呼び出し前後で入力スライス
/// （gold・候補・下限基準の予測）が変わらないこと。
#[test]
fn compare_with_baseline_preserves_input_records() {
    let labels = ["A", "B", "C"];
    let (gold, candidate, baseline) = build_rows("A", 10, 5, 3, 12);
    let records = build_paired_records(&gold, &candidate, &baseline);

    let before: Vec<(&str, Outcome, Outcome)> = records
        .iter()
        .map(|r| (r.gold, r.candidate.clone(), r.baseline.clone()))
        .collect();

    let required = RequiredSampleSize::new(1).unwrap();
    let _ = compare_with_baseline(&labels, &records, required).unwrap();

    let after: Vec<(&str, Outcome, Outcome)> = records
        .iter()
        .map(|r| (r.gold, r.candidate.clone(), r.baseline.clone()))
        .collect();

    assert_eq!(
        before, after,
        "評価の前後で入力レコードが変わってはならない（REQ-27）"
    );
}

/// [`fandhe_edge_eval::significance::correctness`] は
/// [`EvalRecord`] 列に対しても同じ正誤規則で判定する（候補単体の正誤集計に
/// も使える汎用性の確認）。
#[test]
fn correctness_matches_compare_with_baseline_candidate_side() {
    use fandhe_edge_eval::significance::correctness;

    let labels = ["A", "B", "C"];
    let (gold, candidate, _baseline) = build_rows("A", 10, 5, 3, 12);
    let eval_records: Vec<EvalRecord<'_>> = gold
        .iter()
        .zip(candidate.iter())
        .map(|(&g, o)| EvalRecord {
            gold: g,
            outcome: o,
        })
        .collect();

    let flags = correctness(&labels, &eval_records).unwrap();
    let correct_count = flags.iter().filter(|&&v| v).count() as u64;
    // both_correct(10) + b(5) = 候補の正解件数。
    assert_eq!(correct_count, 15);
}
