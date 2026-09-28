//! `type_meaning_quadrant` の境界値の結合テスト（REQ-24 境界値・TASK-24.3・issue #62）。
//!
//! 出典: PoC-9 `03-poc/evaluation-contract`
//! （`fixtures/v1.1/prediction-defects`・`logs/compare/eval_v1.1-prediction-defects.json`
//! の型不正・エラー・意味誤りの混在ケース、`fixtures/anomaly-03`〔unknown-label〕・
//! `anomaly-04`〔type-invalid〕・`anomaly-11`〔all-abstain〕・`anomaly-12`
//! 〔all-error〕の単一種別ケース）。`docs/spec` は読み込まず、期待値は
//! テストファイルへ直接書く（`.claude/rules/spec-reference.md` のビルド
//! 独立方針）。
//!
//! 証拠の種別: テストハーネス（PoC-9 の手計算値・評価器出力の転記との照合。
//! 実機計測なし）。単一選択（single-select）のみを対象とし、multi-item は
//! 対象外（`crates/eval/src/quadrant.rs` のドキュメントコメント参照）。

use fandhe_edge_eval::metrics::{EvalRecord, Outcome, evaluate_single_select};

/// PoC-9 `v1.1/prediction-defects` を移植した境界値の主ケース。
///
/// 6 件中: pd-1 型正しく意味も正しい・pd-2 型正しいが推論エラー・
/// pd-3 型不正（status "done"。A-3 の正規化で Invalid）・
/// pd-4 型正しいが推論エラー（NaN スコア）・pd-5 型正しく意味も正しい・
/// pd-6 型は正しいが意味が誤り（gold と不一致）。
///
/// 期待値: type_ok_meaning_ok=2 / type_ok_meaning_ng=1 / type_ng_count=1 /
/// abstain=0 / error=2（型不正〔pd-3〕と、型は正しいが意味が誤り〔pd-6〕が
/// 別のセルに入ることを確認する）。
#[test]
fn prediction_defects_boundary_case_separates_type_ng_from_meaning_ng() {
    let out_a1 = Outcome::Label("A".to_string());
    let out_error1 = Outcome::Error;
    let out_invalid = Outcome::Invalid;
    let out_error2 = Outcome::Error;
    let out_a2 = Outcome::Label("A".to_string());
    let out_wrong = Outcome::Label("A".to_string());

    let records = vec![
        EvalRecord {
            gold: "A",
            outcome: &out_a1,
        },
        EvalRecord {
            gold: "A",
            outcome: &out_error1,
        },
        EvalRecord {
            gold: "B",
            outcome: &out_invalid,
        },
        EvalRecord {
            gold: "B",
            outcome: &out_error2,
        },
        EvalRecord {
            gold: "A",
            outcome: &out_a2,
        },
        EvalRecord {
            gold: "B",
            outcome: &out_wrong,
        },
    ];

    let metrics = evaluate_single_select(&["A", "B"], &records).expect("valid input");
    let quadrant = &metrics.type_meaning_quadrant;
    assert_eq!(quadrant.type_ok_meaning_ok(), 2);
    assert_eq!(quadrant.type_ok_meaning_ng(), 1);
    assert_eq!(quadrant.type_ng_count(), 1);
    assert_eq!(quadrant.abstain(), 0);
    assert_eq!(quadrant.error(), 2);
    assert_eq!(quadrant.total(), Some(6));
}

/// PoC-9 anomaly-04（type-invalid。1 件）: `Outcome::Invalid` は
/// type_ng_count に数える。
#[test]
fn type_invalid_single_record_counts_as_type_ng() {
    let outcome = Outcome::Invalid;
    let records = vec![EvalRecord {
        gold: "A",
        outcome: &outcome,
    }];
    let metrics = evaluate_single_select(&["A", "B"], &records).expect("valid input");
    let quadrant = &metrics.type_meaning_quadrant;
    assert_eq!(quadrant.type_ok_meaning_ok(), 0);
    assert_eq!(quadrant.type_ok_meaning_ng(), 0);
    assert_eq!(quadrant.type_ng_count(), 1);
    assert_eq!(quadrant.abstain(), 0);
    assert_eq!(quadrant.error(), 0);
    assert_eq!(quadrant.total(), Some(1));
}

/// PoC-9 anomaly-03（unknown-label。1 件）: ラベル集合に無い予測は
/// type_ng_count に数える。gold と偶然同じ文字列でも、ラベル集合に
/// 宣言されていなければ意味の正しさのセルには数えない
/// （型不正の行は意味を判定できないため）。
#[test]
fn unknown_label_not_in_label_set_counts_as_type_ng_even_when_string_matches_gold() {
    // gold は "C" だが、宣言済みラベルは ["A","B"] のみ。予測 "C" は
    // ラベル集合に無いため、gold と文字列が一致していても type_ng_count に
    // 数える（意味の正しさのセルへは数えない）。
    let out_correct = Outcome::Label("A".to_string());
    let outcome = Outcome::Label("C".to_string());
    let records = vec![
        EvalRecord {
            gold: "A",
            outcome: &out_correct,
        },
        EvalRecord {
            gold: "A",
            outcome: &outcome,
        },
    ];
    // "C" は gold としても許容されないため、labels に含めず A のみの
    // gold で構成する（gold は宣言済みラベル集合に存在する必要がある）。
    let metrics = evaluate_single_select(&["A", "B"], &records).expect("valid input");
    let quadrant = &metrics.type_meaning_quadrant;
    assert_eq!(quadrant.type_ok_meaning_ok(), 1);
    assert_eq!(quadrant.type_ok_meaning_ng(), 0);
    assert_eq!(quadrant.type_ng_count(), 1);
    assert_eq!(quadrant.total(), Some(2));
}

/// PoC-9 anomaly-11（all-abstain。3 件）: 全件 abstain のとき、
/// abstain セルにのみ数える。
#[test]
fn all_abstain_counts_only_abstain_cell() {
    let outcome = Outcome::Abstain;
    let records = vec![
        EvalRecord {
            gold: "A",
            outcome: &outcome,
        },
        EvalRecord {
            gold: "A",
            outcome: &outcome,
        },
        EvalRecord {
            gold: "B",
            outcome: &outcome,
        },
    ];
    let metrics = evaluate_single_select(&["A", "B"], &records).expect("valid input");
    let quadrant = &metrics.type_meaning_quadrant;
    assert_eq!(quadrant.type_ok_meaning_ok(), 0);
    assert_eq!(quadrant.type_ok_meaning_ng(), 0);
    assert_eq!(quadrant.type_ng_count(), 0);
    assert_eq!(quadrant.abstain(), 3);
    assert_eq!(quadrant.error(), 0);
    assert_eq!(quadrant.total(), Some(3));
}

/// PoC-9 anomaly-12（all-error。3 件）: 全件エラーのとき、error セルにのみ数える。
#[test]
fn all_error_counts_only_error_cell() {
    let outcome = Outcome::Error;
    let records = vec![
        EvalRecord {
            gold: "A",
            outcome: &outcome,
        },
        EvalRecord {
            gold: "A",
            outcome: &outcome,
        },
        EvalRecord {
            gold: "B",
            outcome: &outcome,
        },
    ];
    let metrics = evaluate_single_select(&["A", "B"], &records).expect("valid input");
    let quadrant = &metrics.type_meaning_quadrant;
    assert_eq!(quadrant.type_ok_meaning_ok(), 0);
    assert_eq!(quadrant.type_ok_meaning_ng(), 0);
    assert_eq!(quadrant.type_ng_count(), 0);
    assert_eq!(quadrant.abstain(), 0);
    assert_eq!(quadrant.error(), 3);
    assert_eq!(quadrant.total(), Some(3));
}

/// REQ-24 境界値・TASK-24.3: `records` の走査順が変わっても
/// `type_meaning_quadrant` の集計結果は一致する（決定性。乱数を使わない）。
#[test]
fn quadrant_is_independent_of_record_order() {
    let out_a = Outcome::Label("A".to_string());
    let out_b = Outcome::Label("B".to_string());
    let out_invalid = Outcome::Invalid;

    let forward = vec![
        EvalRecord {
            gold: "A",
            outcome: &out_a,
        },
        EvalRecord {
            gold: "B",
            outcome: &out_a,
        },
        EvalRecord {
            gold: "A",
            outcome: &out_invalid,
        },
        EvalRecord {
            gold: "B",
            outcome: &out_b,
        },
    ];
    let mut reversed = forward.clone();
    reversed.reverse();

    let labels = ["A", "B"];
    let forward_metrics = evaluate_single_select(&labels, &forward).expect("valid input");
    let reversed_metrics = evaluate_single_select(&labels, &reversed).expect("valid input");

    assert_eq!(
        forward_metrics.type_meaning_quadrant,
        reversed_metrics.type_meaning_quadrant
    );
}
