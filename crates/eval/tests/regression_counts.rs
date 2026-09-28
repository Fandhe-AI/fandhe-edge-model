//! 旧モデルとの回帰件数算出の結合テスト（REQ-26 正常系・TASK-26.1-1・
//! issue #100）。
//!
//! [`fandhe_edge_eval::regression::compare_with_previous`]・
//! [`fandhe_edge_eval::regression::regression_counts`] を、受入基準の合成
//! データと PoC-19（`03-poc/model-lifecycle/scripts/exp_rebuild.py` の
//! `paired_bootstrap_ci` 内の `b`/`c` 集計）実測値の転記で確認する。
//!
//! 証拠の種別: テストハーネス（受入基準データは決定的に組み立て、PoC-19 の
//! 件数は実測値の転記。`jobs/rebuild/result.json` から整数のみを転記し、
//! `docs/spec` は読まない）。
//!
//! Wilson 95% 信頼区間（TASK-26.1-2・issue #101）のテストについては、
//! 区間の期待値は独立計算（テストハーネス）である。PoC-19 から転記したのは
//! 整数の件数（58/22/531・44/45/650 等）だけで、PoC-19 自体は遷移率の
//! Wilson 区間を出していない。区間の期待値は center／margin 形式と、
//! 代数的に別の形（`(2k + z² ± z·sqrt(z² + 4k(n−k)/n)) / (2(n + z²))`）の
//! 2 通りで独立に計算し（Python・z=1.96）、両者の差が最大 2.1e-17 であることを
//! 本 issue の実装時に確認済み（`.claude/rules/evaluation-contract.md`
//! 「決定性と証拠の種別」）。

use fandhe_edge_eval::metrics::Outcome;
use fandhe_edge_eval::regression::{RegressionRecord, compare_with_previous, regression_counts};

const EPSILON: f64 = 1e-9;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

/// 受入基準の合成データ: 「両方正解 2・正解→不正解 2・不正解→正解 1・
/// 両方不正解 2」（n=7）。非対称な 2 対 1 で方向の取り違えを検出する。
#[test]
fn compare_with_previous_matches_acceptance_criteria_counts() {
    let labels = ["A", "B", "C"];
    let correct = Outcome::Label("A".to_string());
    let wrong = Outcome::Label("B".to_string());

    let records = [
        // 両方正解 x2
        RegressionRecord {
            gold: "A",
            previous: &correct,
            current: &correct,
        },
        RegressionRecord {
            gold: "A",
            previous: &correct,
            current: &correct,
        },
        // 正解→不正解（回帰）x2: 旧は正解・新は不正解。
        RegressionRecord {
            gold: "A",
            previous: &correct,
            current: &wrong,
        },
        RegressionRecord {
            gold: "A",
            previous: &correct,
            current: &wrong,
        },
        // 不正解→正解（改善）x1: 旧は不正解・新は正解。
        RegressionRecord {
            gold: "A",
            previous: &wrong,
            current: &correct,
        },
        // 両方不正解 x2
        RegressionRecord {
            gold: "A",
            previous: &wrong,
            current: &wrong,
        },
        RegressionRecord {
            gold: "A",
            previous: &wrong,
            current: &wrong,
        },
    ];

    let counts = compare_with_previous(&labels, &records).unwrap();
    assert_eq!(counts.n(), 7);
    assert_eq!(counts.both_correct(), 2);
    assert_eq!(counts.correct_to_incorrect(), 2);
    assert_eq!(counts.incorrect_to_correct(), 1);
    assert_eq!(counts.both_wrong(), 2);
    assert_eq!(counts.previous_correct(), 4);
    assert_eq!(counts.current_correct(), 3);
}

/// REQ-26・TASK-26.1-2: 受入基準の合成データ（n=7・正解→不正解 2・
/// 不正解→正解 1）の Wilson 95% 信頼区間が、独立計算の具体値と
/// 許容差 1e-9 で一致する（証拠の種別: 独立計算によるテストハーネス）。
#[test]
fn compare_with_previous_attaches_wilson_ci95_to_acceptance_counts() {
    let labels = ["A", "B", "C"];
    let correct = Outcome::Label("A".to_string());
    let wrong = Outcome::Label("B".to_string());

    let records = [
        RegressionRecord {
            gold: "A",
            previous: &correct,
            current: &correct,
        },
        RegressionRecord {
            gold: "A",
            previous: &correct,
            current: &correct,
        },
        RegressionRecord {
            gold: "A",
            previous: &correct,
            current: &wrong,
        },
        RegressionRecord {
            gold: "A",
            previous: &correct,
            current: &wrong,
        },
        RegressionRecord {
            gold: "A",
            previous: &wrong,
            current: &correct,
        },
        RegressionRecord {
            gold: "A",
            previous: &wrong,
            current: &wrong,
        },
        RegressionRecord {
            gold: "A",
            previous: &wrong,
            current: &wrong,
        },
    ];

    let counts = compare_with_previous(&labels, &records).unwrap();

    // 正解→不正解（回帰）2/7。
    let regression_ci = counts.correct_to_incorrect_ci95().unwrap();
    assert!(approx_eq(regression_ci.lo(), 0.0822171657090155));
    assert!(approx_eq(regression_ci.hi(), 0.6410709098517873));

    // 不正解→正解（改善）1/7。
    let improvement_ci = counts.incorrect_to_correct_ci95().unwrap();
    assert!(approx_eq(improvement_ci.lo(), 0.02567895594897482));
    assert!(approx_eq(improvement_ci.hi(), 0.5131345033190299));
}

/// 方向の取り違え検出: 上記と同じデータで `previous`/`current` を入れ替えると
/// `correct_to_incorrect`/`incorrect_to_correct` が入れ替わる。
#[test]
fn swapping_previous_and_current_swaps_the_transition_counts() {
    let labels = ["A", "B", "C"];
    let correct = Outcome::Label("A".to_string());
    let wrong = Outcome::Label("B".to_string());

    let records = [
        RegressionRecord {
            gold: "A",
            previous: &correct,
            current: &correct,
        },
        RegressionRecord {
            gold: "A",
            previous: &correct,
            current: &correct,
        },
        // 入れ替え: 旧は不正解・新は正解 x2（元は正解→不正解 x2 だった箇所）。
        RegressionRecord {
            gold: "A",
            previous: &wrong,
            current: &correct,
        },
        RegressionRecord {
            gold: "A",
            previous: &wrong,
            current: &correct,
        },
        // 入れ替え: 旧は正解・新は不正解 x1（元は不正解→正解 x1 だった箇所）。
        RegressionRecord {
            gold: "A",
            previous: &correct,
            current: &wrong,
        },
        RegressionRecord {
            gold: "A",
            previous: &wrong,
            current: &wrong,
        },
        RegressionRecord {
            gold: "A",
            previous: &wrong,
            current: &wrong,
        },
    ];

    let counts = compare_with_previous(&labels, &records).unwrap();
    assert_eq!(counts.correct_to_incorrect(), 1);
    assert_eq!(counts.incorrect_to_correct(), 2);
}

/// PoC-19 P1（M8rm→M9）実測値の転記: n=531・両方正解 100・正解→不正解 58・
/// 不正解→正解 22・両方不正解 351（旧正解 158・新正解 122）。
#[test]
fn poc19_p1_addition_counts() {
    let previous: Vec<bool> = build_correct_vec(100, 58, 22, 351, TransitionOrder::PoC19);
    let current: Vec<bool> = build_current_vec(100, 58, 22, 351);

    let counts = regression_counts(&previous, &current).unwrap();
    assert_eq!(counts.n(), 531);
    assert_eq!(counts.both_correct(), 100);
    assert_eq!(counts.correct_to_incorrect(), 58);
    assert_eq!(counts.incorrect_to_correct(), 22);
    assert_eq!(counts.both_wrong(), 351);
    assert_eq!(counts.previous_correct(), 158);
    assert_eq!(counts.current_correct(), 122);
}

/// PoC-19 P2（M9→M8rm。P1 の逆向き）実測値の転記: n=531・両方正解 100・
/// 正解→不正解 22・不正解→正解 58・両方不正解 351。P1 と 58/22 が
/// 入れ替わることを確認する。
#[test]
fn poc19_p2_removal_counts_are_reversed_from_p1() {
    let previous: Vec<bool> = build_correct_vec(100, 22, 58, 351, TransitionOrder::PoC19);
    let current: Vec<bool> = build_current_vec(100, 22, 58, 351);

    let counts = regression_counts(&previous, &current).unwrap();
    assert_eq!(counts.n(), 531);
    assert_eq!(counts.both_correct(), 100);
    assert_eq!(counts.correct_to_incorrect(), 22);
    assert_eq!(counts.incorrect_to_correct(), 58);
    assert_eq!(counts.both_wrong(), 351);
}

/// PoC-19 P3（M9→M8merge。統合）実測値の転記: n=650・両方正解 170・
/// 正解→不正解 44・不正解→正解 45・両方不正解 391（旧正解 214・新正解 215）。
/// bool 単位 API（`regression_counts`）で確認する。各モデル自身のラベル
/// 空間での正誤ビットを比較したケースのため（ラベル集合相違の前提明記は
/// TASK-26.2 の対象）。
#[test]
fn poc19_p3_merge_counts() {
    let previous: Vec<bool> = build_correct_vec(170, 44, 45, 391, TransitionOrder::PoC19);
    let current: Vec<bool> = build_current_vec(170, 44, 45, 391);

    let counts = regression_counts(&previous, &current).unwrap();
    assert_eq!(counts.n(), 650);
    assert_eq!(counts.both_correct(), 170);
    assert_eq!(counts.correct_to_incorrect(), 44);
    assert_eq!(counts.incorrect_to_correct(), 45);
    assert_eq!(counts.both_wrong(), 391);
    assert_eq!(counts.previous_correct(), 214);
    assert_eq!(counts.current_correct(), 215);
}

/// REQ-26・TASK-26.1-2: PoC-19 P1（58/531・22/531）の Wilson 95% 信頼区間。
/// 件数は PoC-19 の実測値の転記だが、区間自体は PoC-19 の出力ではなく
/// 独立計算（テストハーネス。ファイル冒頭の doc コメント参照）。
#[test]
fn poc19_p1_wilson_ci95() {
    let previous: Vec<bool> = build_correct_vec(100, 58, 22, 351, TransitionOrder::PoC19);
    let current: Vec<bool> = build_current_vec(100, 58, 22, 351);
    let counts = regression_counts(&previous, &current).unwrap();

    let regression_ci = counts.correct_to_incorrect_ci95().unwrap();
    assert!(approx_eq(regression_ci.lo(), 0.08545021452040413));
    assert!(approx_eq(regression_ci.hi(), 0.1386191174088998));

    let improvement_ci = counts.incorrect_to_correct_ci95().unwrap();
    assert!(approx_eq(improvement_ci.lo(), 0.027517252252540213));
    assert!(approx_eq(improvement_ci.hi(), 0.06193278304763089));
}

/// REQ-26・TASK-26.1-2: PoC-19 P2（P1 の逆向き）では、正解→不正解・
/// 不正解→正解の Wilson 区間が P1 と入れ替わることを確認する
/// （方向の取り違え検出）。
#[test]
fn poc19_p2_wilson_ci95_is_swapped_from_p1() {
    let previous: Vec<bool> = build_correct_vec(100, 22, 58, 351, TransitionOrder::PoC19);
    let current: Vec<bool> = build_current_vec(100, 22, 58, 351);
    let counts = regression_counts(&previous, &current).unwrap();

    // P1 の不正解→正解（22/531）の区間と一致する。
    let regression_ci = counts.correct_to_incorrect_ci95().unwrap();
    assert!(approx_eq(regression_ci.lo(), 0.027517252252540213));
    assert!(approx_eq(regression_ci.hi(), 0.06193278304763089));

    // P1 の正解→不正解（58/531）の区間と一致する。
    let improvement_ci = counts.incorrect_to_correct_ci95().unwrap();
    assert!(approx_eq(improvement_ci.lo(), 0.08545021452040413));
    assert!(approx_eq(improvement_ci.hi(), 0.1386191174088998));
}

/// REQ-26・TASK-26.1-2: PoC-19 P3（44/650・45/650）の Wilson 95% 信頼区間。
#[test]
fn poc19_p3_wilson_ci95() {
    let previous: Vec<bool> = build_correct_vec(170, 44, 45, 391, TransitionOrder::PoC19);
    let current: Vec<bool> = build_current_vec(170, 44, 45, 391);
    let counts = regression_counts(&previous, &current).unwrap();

    let regression_ci = counts.correct_to_incorrect_ci95().unwrap();
    assert!(approx_eq(regression_ci.lo(), 0.05080936994768587));
    assert!(approx_eq(regression_ci.hi(), 0.08965523187636448));

    let improvement_ci = counts.incorrect_to_correct_ci95().unwrap();
    assert!(approx_eq(improvement_ci.lo(), 0.052140156934892226));
    assert!(approx_eq(improvement_ci.hi(), 0.09138328972252448));
}

/// テストデータ組み立ての意図を示すためだけのマーカー型（現状は単一の並び
/// 方針のみ）。将来 P1/P2/P3 で異なる並び規則が要る場合に備え、呼び出し側の
/// 意図を型で示す。
enum TransitionOrder {
    PoC19,
}

/// 件数（両方正解・正解→不正解・不正解→正解・両方不正解）から `previous`
/// 側の正誤 bool 列を機械的に組み立てる（乱数不使用。順序は
/// both_correct → correct_to_incorrect → incorrect_to_correct → both_wrong）。
fn build_correct_vec(
    both_correct: usize,
    correct_to_incorrect: usize,
    incorrect_to_correct: usize,
    both_wrong: usize,
    _order: TransitionOrder,
) -> Vec<bool> {
    let mut v =
        Vec::with_capacity(both_correct + correct_to_incorrect + incorrect_to_correct + both_wrong);
    v.extend(std::iter::repeat_n(true, both_correct));
    v.extend(std::iter::repeat_n(true, correct_to_incorrect));
    v.extend(std::iter::repeat_n(false, incorrect_to_correct));
    v.extend(std::iter::repeat_n(false, both_wrong));
    v
}

/// `build_correct_vec` と対になる `current` 側の正誤 bool 列。
fn build_current_vec(
    both_correct: usize,
    correct_to_incorrect: usize,
    incorrect_to_correct: usize,
    both_wrong: usize,
) -> Vec<bool> {
    let mut v =
        Vec::with_capacity(both_correct + correct_to_incorrect + incorrect_to_correct + both_wrong);
    v.extend(std::iter::repeat_n(true, both_correct));
    v.extend(std::iter::repeat_n(false, correct_to_incorrect));
    v.extend(std::iter::repeat_n(true, incorrect_to_correct));
    v.extend(std::iter::repeat_n(false, both_wrong));
    v
}
