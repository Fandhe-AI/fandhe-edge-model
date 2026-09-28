//! REQ-18・REQ-25（TASK-18.3-2・issue #88）: 評価件数が McNemar 比較の
//! 必要件数（[`RequiredSampleSize`]）に満たない場合、選定結果の有意性判定が
//! 判定不能（[`BaselineVerdict::Undeterminable`]）になり、合格扱い
//! （[`BaselineVerdict::SignificantlyBetter`]）にならないことを、件数不足の
//! 合成データで確認する結合テスト（証拠種別: テストハーネス）。
//!
//! 親 issue #86（TASK-18.3）の 1 つ目の子 #87（TASK-18.3-1・PR #236）で
//! `fandhe_edge_train::selection_significance::assess_selection_significance`
//! （評価器 `fandhe-edge-eval` の `baseline::fit_majority`・
//! `significance::compare_with_baseline`・`holm::compare_candidates_with_holm`
//! を接続するだけの本リポの層）が実装済み。本ファイルは**テストの追加のみ**で、
//! 判定規則（[`fandhe_edge_eval::significance::judge`]）の再実装・変更は
//! 行わない（`.claude/rules/coding-rust.md`「評価器は 1 つに集約」・
//! `.claude/rules/evaluation-contract.md`「件数不足は判定不能として返し、
//! 合格扱いにしない」）。
//!
//! 選定記録型（TASK-18.1-2・issue #84）は未実装のため、本ファイルでは
//! `SelectionSignificance::verdict()`（Holm 補正後の最終判定）と
//! `raw_verdict()`（補正前）がともに `Undeterminable(InsufficientSamples)`
//! であり `SignificantlyBetter` でないことをもって「合格扱いにならない」と
//! 確認する。`crates/eval`・`Cargo.toml`・`docs/spec` は参照・変更しない。

use fandhe_edge_eval::holm::FamilySize;
use fandhe_edge_eval::metrics::Outcome;
use fandhe_edge_eval::sample_size::{McNemarSampleSizeAssumption, required_sample_size_mcnemar};
use fandhe_edge_eval::significance::{BaselineVerdict, InsufficientSamples, RequiredSampleSize};
use fandhe_edge_train::selection_significance::{
    CandidateValidation, SelectionSignificanceInput, assess_selection_significance,
};

/// テストヘルパー。外部入力経路ではないため `expect` を許容する
/// （`.claude/rules/coding-rust.md`「外部入力の経路では unwrap/expect を
/// 使わない」の対象外）。
fn req(n: u64) -> RequiredSampleSize {
    RequiredSampleSize::new(n).expect("test helper requires a non-zero value")
}

fn family(n: usize) -> FamilySize {
    FamilySize::new(n).expect("test helper requires a non-zero family size")
}

/// 学習ラベル（多数決 = "A"。A×5・B×3・C×2）。
fn train_labels_majority_a() -> Vec<&'static str> {
    vec!["A", "A", "A", "A", "A", "B", "B", "B", "C", "C"]
}

/// 決定的に (both_correct, b, c, both_wrong) から validation 行を組み立てる
/// （`crates/train/src/selection_significance.rs` の既存 unit test
/// `build_outcomes` と同じ作り方をテストファイル内に再定義したもの。
/// gold=A・候補正解: both_correct 行 / gold=B・候補正解: b 行 /
/// gold=A・候補不正解: c 行 / gold=B・候補不正解: both_wrong 行）。
fn build_outcomes(
    both_correct: usize,
    b: usize,
    c: usize,
    both_wrong: usize,
) -> (Vec<&'static str>, Vec<Outcome>) {
    let mut gold = Vec::new();
    let mut outcomes = Vec::new();
    for _ in 0..both_correct {
        gold.push("A");
        outcomes.push(Outcome::Label("A".to_string()));
    }
    for _ in 0..b {
        gold.push("B");
        outcomes.push(Outcome::Label("B".to_string()));
    }
    for _ in 0..c {
        gold.push("A");
        outcomes.push(Outcome::Label("B".to_string()));
    }
    for _ in 0..both_wrong {
        gold.push("B");
        outcomes.push(Outcome::Label("C".to_string()));
    }
    (gold, outcomes)
}

/// 件数不足（n=6 < required=7）なら、p 値だけを見れば有意に見える
/// （raw_p ≈ 0.03125 < 0.05）場合でも判定不能になり、合格扱い
/// （`SignificantlyBetter`）にならないこと（REQ-18・REQ-25・
/// TASK-18.3-2・issue #88。証拠種別: テストハーネス）。
#[test]
fn undeterminable_when_below_required_even_if_p_is_significant() {
    // gold = B×6、候補予測 = B×6（下限基準 "A" は全不正解）→ b=6・c=0。
    let (gold, outcomes) = build_outcomes(0, 6, 0, 0);
    let train_labels = train_labels_majority_a();
    let labels = ["A", "B", "C"];
    let candidates = [CandidateValidation {
        candidate_id: "c1-a-i8",
        outcomes: &outcomes,
    }];
    let input = SelectionSignificanceInput {
        label_order: &labels,
        train_labels: &train_labels,
        validation_gold: &gold,
        candidates: &candidates,
        selected_candidate_id: "c1-a-i8",
        required: req(7),
        family_size: family(1),
    };

    let result = assess_selection_significance(&input).unwrap();
    assert_eq!(result.counts().n, 6);
    assert_eq!(result.counts().b_candidate_only, 6);
    assert_eq!(result.counts().c_baseline_only, 0);
    assert!((result.raw_p().value() - 0.031_25).abs() < 1e-9);
    assert!(result.raw_p().value() < 0.05);

    let expected = BaselineVerdict::Undeterminable(InsufficientSamples {
        required: req(7),
        actual: 6,
    });
    assert_eq!(result.verdict(), expected);
    assert_eq!(result.raw_verdict(), expected);
    assert!(!matches!(
        result.verdict(),
        BaselineVerdict::SignificantlyBetter
    ));
}

/// #1 と同一データで `required=6`（N == required）にすると判定可能になり
/// `SignificantlyBetter` を返す対照テスト。件数だけが判定不能の原因で
/// あることを示す（REQ-18・REQ-25・TASK-18.3-2・issue #88。
/// 証拠種別: テストハーネス）。
#[test]
fn significantly_better_at_required_boundary_contrast() {
    let (gold, outcomes) = build_outcomes(0, 6, 0, 0);
    let train_labels = train_labels_majority_a();
    let labels = ["A", "B", "C"];
    let candidates = [CandidateValidation {
        candidate_id: "c1-a-i8",
        outcomes: &outcomes,
    }];
    let input = SelectionSignificanceInput {
        label_order: &labels,
        train_labels: &train_labels,
        validation_gold: &gold,
        candidates: &candidates,
        selected_candidate_id: "c1-a-i8",
        required: req(6),
        family_size: family(1),
    };

    let result = assess_selection_significance(&input).unwrap();
    assert_eq!(result.counts().n, 6);
    assert_eq!(result.verdict(), BaselineVerdict::SignificantlyBetter);
    assert_eq!(result.raw_verdict(), BaselineVerdict::SignificantlyBetter);
}

/// 必要件数算出（TASK-25.2 の [`required_sample_size_mcnemar`]）の実経路と
/// 接続した確認: PoC-10 事前登録の Holm m=4 最厳段の仮定
/// （p_b=0.15・p_c=0.05・alpha=0.0125・power=0.8）から算出した必要件数
/// （229）に対し、評価件数 221 件では判定不能になること
/// （REQ-18・REQ-25・TASK-18.3-2・issue #88。証拠種別: テストハーネス）。
#[test]
fn undeterminable_with_required_from_sample_size_estimate() {
    let (gold, outcomes) = build_outcomes(100, 13, 4, 104);
    assert_eq!(gold.len(), 221);
    let train_labels = train_labels_majority_a();
    let labels = ["A", "B", "C"];
    let candidates = [CandidateValidation {
        candidate_id: "c1-a-i8",
        outcomes: &outcomes,
    }];

    let assumption = McNemarSampleSizeAssumption::new(0.15, 0.05, 0.0125, 0.8)
        .expect("test helper requires a valid assumption");
    let required = required_sample_size_mcnemar(&assumption)
        .expect("test helper requires a computable sample size");
    assert_eq!(required.get(), 229);

    let input = SelectionSignificanceInput {
        label_order: &labels,
        train_labels: &train_labels,
        validation_gold: &gold,
        candidates: &candidates,
        selected_candidate_id: "c1-a-i8",
        required,
        family_size: family(1),
    };

    let result = assess_selection_significance(&input).unwrap();
    assert_eq!(result.counts().n, 221);
    assert!((result.raw_p().value() - 0.049_041_748_046_875).abs() < 1e-9);
    assert!(result.raw_p().value() < 0.05);

    let expected = BaselineVerdict::Undeterminable(InsufficientSamples {
        required,
        actual: 221,
    });
    assert_eq!(result.verdict(), expected);
    assert_eq!(result.raw_verdict(), expected);
}

/// 全候補で共有する validation_gold（gold1）の上で、`b`・`c` 件数だけを
/// 指定して別候補の予測列を組み立てる（`selection_significance.rs` の
/// 既存 unit test `build_second_candidate_outcomes` と同じ作り方）。
fn build_second_candidate_outcomes(gold: &[&str], b: usize, c: usize) -> Vec<Outcome> {
    let mut b_left = b;
    let mut c_left = c;
    gold.iter()
        .map(|&g| {
            if g == "B" && b_left > 0 {
                b_left -= 1;
                Outcome::Label("B".to_string())
            } else if g == "A" && c_left > 0 {
                c_left -= 1;
                Outcome::Label("B".to_string())
            } else if g == "B" {
                Outcome::Label("C".to_string())
            } else {
                Outcome::Label("A".to_string())
            }
        })
        .collect()
}

/// Holm 補正（族サイズ m=2・m=3）を通しても、件数不足による判定不能が
/// 維持されること（Holm 補正は判定不能を有意側へ動かさない）
/// （REQ-18・REQ-25・TASK-18.3-2・issue #88。証拠種別: テストハーネス）。
#[test]
fn holm_correction_keeps_undeterminable() {
    // gold = B×6、選定候補 = B×6（p=0.03125）。
    let (gold, outcomes1) = build_outcomes(0, 6, 0, 0);
    // 他候補 = 下限基準と一致（b=c=0 → p=1.0。選定候補と p 値を分ける）。
    let outcomes2 = build_second_candidate_outcomes(&gold, 0, 0);
    let train_labels = train_labels_majority_a();
    let labels = ["A", "B", "C"];

    for family_size_n in [2usize, 3usize] {
        let candidates = [
            CandidateValidation {
                candidate_id: "c1-a-i8",
                outcomes: &outcomes1,
            },
            CandidateValidation {
                candidate_id: "c3-b64",
                outcomes: &outcomes2,
            },
        ];
        let input = SelectionSignificanceInput {
            label_order: &labels,
            train_labels: &train_labels,
            validation_gold: &gold,
            candidates: &candidates,
            selected_candidate_id: "c1-a-i8",
            required: req(7),
            family_size: family(family_size_n),
        };

        let result = assess_selection_significance(&input).unwrap();
        assert_eq!(result.family_size().get(), family_size_n);

        let expected_adjusted = 0.031_25 * family_size_n as f64;
        assert!((result.adjusted_p().value() - expected_adjusted).abs() < 1e-9);

        let expected = BaselineVerdict::Undeterminable(InsufficientSamples {
            required: req(7),
            actual: 6,
        });
        assert_eq!(result.verdict(), expected);
        assert_eq!(result.raw_verdict(), expected);
        assert!(!matches!(
            result.verdict(),
            BaselineVerdict::SignificantlyBetter
        ));
    }
}

/// 下限基準が有利な向き（`c > b`）でも、件数不足なら
/// `NotSignificantlyBetter` ではなく判定不能が優先されること
/// （[`fandhe_edge_eval::significance::judge`] の規則 1 が規則 2 より
/// 先に評価されることの確認。REQ-18・REQ-25・TASK-18.3-2・issue #88。
/// 証拠種別: テストハーネス）。
#[test]
fn undeterminable_regardless_of_direction() {
    // gold = A×6、候補予測 = B×6（下限基準 "A" のみ正解）→ b=0・c=6。
    let (gold, outcomes) = build_outcomes(0, 0, 6, 0);
    let train_labels = train_labels_majority_a();
    let labels = ["A", "B", "C"];
    let candidates = [CandidateValidation {
        candidate_id: "c1-a-i8",
        outcomes: &outcomes,
    }];
    let input = SelectionSignificanceInput {
        label_order: &labels,
        train_labels: &train_labels,
        validation_gold: &gold,
        candidates: &candidates,
        selected_candidate_id: "c1-a-i8",
        required: req(7),
        family_size: family(1),
    };

    let result = assess_selection_significance(&input).unwrap();
    assert_eq!(result.counts().n, 6);
    assert_eq!(result.counts().b_candidate_only, 0);
    assert_eq!(result.counts().c_baseline_only, 6);
    assert!((result.raw_p().value() - 0.031_25).abs() < 1e-9);

    let expected = BaselineVerdict::Undeterminable(InsufficientSamples {
        required: req(7),
        actual: 6,
    });
    assert_eq!(result.verdict(), expected);
    assert_eq!(result.raw_verdict(), expected);
}
