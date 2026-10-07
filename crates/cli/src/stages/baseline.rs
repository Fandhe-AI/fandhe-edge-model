//! 下限基準（majority）との McNemar 比較の準備・実行・照合（REQ-25・REQ-27・#339）。
//!
//! # 呼び出し文脈
//!
//! - `evaluate` 工程が、最終 test の適用権を取る**前**に [`prepare_baseline`] で majority と必要件数を
//!   確定し、適用後の `finish` 内で [`compare`] を呼んで評価記録へ残す
//! - `package` 工程が、同じ [`prepare_baseline`] で計算し直した値で、評価記録の比較欄を
//!   [`record_matches`] で照合する（記録の改変の検出）
//!
//! 比較の部品（majority の作成・必要件数・McNemar・判定）は評価器 `fandhe-edge-eval` にあり、
//! 本ファイルは定義・分割・記録との接続だけを持つ（評価ロジックを再実装しない。REQ-24）。
//!
//! # 評価の独立性（REQ-27）
//!
//! [`prepare_baseline`] が受け取るのは定義・全レコード・分割だけで、評価データ（凍結した最終 test の
//! バイト列・凍結記録）は引数に無い。majority は分割の `train` に割り当てた行のラベルだけから作る
//! （validation・test のラベルは含めない）。この「評価データを渡せない」シグネチャが独立性の型上の保証。
//! 必要件数も定義（正準化ハッシュに含まれる事前登録）の仮定だけから求める。
//!
//! # エラー
//!
//! 準備の失敗はすべて `invalid_input`（64）で、message は固定語彙（データの本文・ラベルを含めない）。

use fandhe_edge_core::definition::{BaselineComparisonAssumption, Definition};
use fandhe_edge_core::evaluation_record::{BaselineComparisonRecord, BaselineComparisonVerdict};
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_data::inspect::ValidRecord;
use fandhe_edge_data::split::SplitResult;
use fandhe_edge_eval::baseline::fit_majority;
use fandhe_edge_eval::mcnemar::mcnemar_exact_two_sided;
use fandhe_edge_eval::metrics::Outcome;
use fandhe_edge_eval::sample_size::{McNemarSampleSizeAssumption, required_sample_size_mcnemar};
use fandhe_edge_eval::significance::{
    BaselineVerdict, PairedRecord, RequiredSampleSize, SIGNIFICANCE_ALPHA, compare_with_baseline,
    judge,
};

use crate::project::{invalid, runtime};

use super::train::train_rows;

/// 1 万分率の分母。
const BP_DENOMINATOR: f64 = 10_000.0;

/// 適用前に確定した比較の準備（majority と事前計算した必要件数）。
#[derive(Debug, Clone)]
pub(crate) struct PreparedBaseline {
    majority_label: String,
    required: RequiredSampleSize,
}

impl PreparedBaseline {
    /// 凍結した評価データの正解ラベル列に対し、majority を答えたときの正解数を数える（`package` 用）。
    ///
    /// 記録の `baseline_correct` を、記録に頼らず凍結データから計算し直した値と照合するための部品。
    pub(crate) fn baseline_correct_on<'a>(&self, golds: impl Iterator<Item = &'a str>) -> u64 {
        golds
            .filter(|g| *g == self.majority_label)
            .fold(0_u64, |n, _| n.saturating_add(1))
    }
}

/// 定義の仮定から必要件数を求める。`evaluate` と `package` が同じ値を得るよう、計算はこの 1 関数に限る。
///
/// # Errors
/// 仮定を評価器が受理しない・必要件数が探索上限を超える場合は `invalid_input`。
fn required_from_assumption(
    assumption: &BaselineComparisonAssumption,
) -> Result<RequiredSampleSize, ErrorReport> {
    let bp = |v: u32| f64::from(v) / BP_DENOMINATOR;
    let model = McNemarSampleSizeAssumption::new(
        bp(assumption.assumed_p_b_bp()),
        bp(assumption.assumed_p_c_bp()),
        SIGNIFICANCE_ALPHA,
        bp(assumption.power_bp()),
    )
    .map_err(|_| invalid("baseline comparison sample size cannot be computed"))?;
    required_sample_size_mcnemar(&model)
        .map_err(|_| invalid("baseline comparison sample size cannot be computed"))
}

/// 定義に `baseline_comparison` があれば、train 分割だけから majority を作り、必要件数を求める。
///
/// 欄が無ければ `Ok(None)`（比較しない）。引数に評価データを含めない（モジュール doc。REQ-27）。
/// 同数の多数派は選択肢の宣言順で先のラベルになる（評価器 `fit_majority` の規則）。
///
/// # Errors
/// train の行が無い・ラベルが選択肢に無い・必要件数を算出できない場合は `invalid_input`。
pub(crate) fn prepare_baseline(
    definition: &Definition,
    records: &[ValidRecord],
    split: &SplitResult,
) -> Result<Option<PreparedBaseline>, ErrorReport> {
    let Some(assumption) = definition.baseline_comparison() else {
        return Ok(None);
    };
    let majority_label = majority_from_train(definition, records, split)?;
    let required = required_from_assumption(assumption)?;
    Ok(Some(PreparedBaseline {
        majority_label,
        required,
    }))
}

/// train 分割のラベルだけから majority のラベルを求める（`prepare_baseline` と PoC-26 の採点入口
/// `score_predictions` が共有する。評価データを引数に取らない。REQ-27・REQ-41・#445）。
///
/// # Errors
/// train の行が無い・ラベルが選択肢に無い場合は `invalid_input`。
pub(crate) fn majority_from_train(
    definition: &Definition,
    records: &[ValidRecord],
    split: &SplitResult,
) -> Result<String, ErrorReport> {
    let labels: Vec<&str> = definition.options().iter().map(|c| c.id.as_str()).collect();
    let train_labels: Vec<&str> = train_rows(records, split)
        .map(|r| r.label_id.as_str())
        .collect();
    fit_majority(&labels, &train_labels)
        .map(str::to_string)
        .map_err(|_| invalid("baseline comparison cannot be prepared"))
}

/// 評価器の判定を記録の語彙へ写す。将来 variant が増えたら黙って通さず失敗させる（fail-closed）。
fn map_verdict(verdict: BaselineVerdict) -> Result<BaselineComparisonVerdict, ErrorReport> {
    match verdict {
        BaselineVerdict::SignificantlyBetter => Ok(BaselineComparisonVerdict::SignificantlyBetter),
        BaselineVerdict::NotSignificantlyBetter => {
            Ok(BaselineComparisonVerdict::NotSignificantlyBetter)
        }
        BaselineVerdict::Undeterminable(_) => Ok(BaselineComparisonVerdict::Undeterminable),
        _ => Err(runtime("unsupported baseline verdict")),
    }
}

/// 候補の予測と majority を対応づけて McNemar 比較を行い、評価記録の比較欄を作る。
///
/// `golds` と `outcomes` は同じ長さ・同じ順序。戻り値の 2 つ目は評価器が数えた候補の正解数で、
/// 呼び出し側が指標の正解数と照合する（正誤の規則を 2 重化した結果のずれを記録前に止める）。
///
/// # Errors
/// 評価器が比較できない場合は `runtime_error`。
pub(crate) fn compare(
    prepared: &PreparedBaseline,
    labels: &[&str],
    golds: &[String],
    outcomes: &[Outcome],
) -> Result<(BaselineComparisonRecord, u64), ErrorReport> {
    let baseline = Outcome::Label(prepared.majority_label.clone());
    let paired: Vec<PairedRecord<'_>> = golds
        .iter()
        .zip(outcomes)
        .map(|(gold, candidate)| PairedRecord {
            gold,
            candidate,
            baseline: &baseline,
        })
        .collect();
    let comparison = compare_with_baseline(labels, &paired, prepared.required)
        .map_err(|_| runtime("cannot compute baseline comparison"))?;
    let counts = comparison.counts();
    let record = BaselineComparisonRecord {
        majority_label: prepared.majority_label.clone(),
        baseline_correct: comparison.baseline_correct(),
        b: counts.b_candidate_only,
        c: counts.c_baseline_only,
        required_n: comparison.required().get(),
        verdict: map_verdict(comparison.verdict())?,
    };
    Ok((record, comparison.candidate_correct()))
}

/// 評価記録の比較欄が、計算し直した majority・必要件数・判定と矛盾しないかを確かめる（`package` 用）。
///
/// 件数どうしの整合（`both_correct` が候補側と下限基準側で一致する等）も checked 演算で確認し、
/// `baseline_correct` は凍結した評価データから数え直した `expected_baseline_correct` と一致を求める。
///
/// 検出できる範囲: majority・必要件数・`baseline_correct`・件数の算術的整合・`verdict`（再計算）。
/// 検出できない範囲: `b` と `c` を同じ量だけずらし、`both_correct` を同量だけ減らす改変
/// （例 total=100・correct=80・baseline_correct=60 で b=30,c=10 → b=31,c=11）。`both_correct` は
/// 候補の予測ごとの正誤でしか確かめられず、`package` は評価データへ推論を再適用しない（REQ-27）ため。
/// 完全な検証には候補の予測の封印記録が要る（外部台帳 #168 の範囲）。記録ファイルを丸ごと作り直せる
/// 主体も同様に防げない。
pub(crate) fn record_matches(
    prepared: &PreparedBaseline,
    rec: &BaselineComparisonRecord,
    (correct, total): (u64, u64),
    expected_baseline_correct: u64,
) -> bool {
    let Some(both_by_candidate) = correct.checked_sub(rec.b) else {
        return false;
    };
    let Some(both_by_baseline) = rec.baseline_correct.checked_sub(rec.c) else {
        return false;
    };
    let counts_ok = both_by_candidate == both_by_baseline
        && rec.baseline_correct <= total
        && both_by_candidate
            .checked_add(rec.b)
            .and_then(|v| v.checked_add(rec.c))
            .is_some_and(|v| v <= total);
    if !counts_ok
        || rec.required_n != prepared.required.get()
        || rec.majority_label != prepared.majority_label
        || rec.baseline_correct != expected_baseline_correct
    {
        return false;
    }
    let Ok(test) = mcnemar_exact_two_sided(rec.b, rec.c) else {
        return false;
    };
    let verdict = judge(rec.b, rec.c, test.p_two_sided(), total, prepared.required);
    map_verdict(verdict).is_ok_and(|v| v == rec.verdict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::train::test_support::records_and_split_labeled;
    use fandhe_edge_core::exitcode::ExitCode;
    use fandhe_edge_data::split::Split;

    fn definition(baseline: &str) -> Definition {
        Definition::parse(&format!(
            r#"{{
                "schema": "fandhe-edge-model-definition/v1",
                "name": "t", "version": 1, "judgment_type": "single_select",
                "options": [
                    {{ "id": "a", "display_name": "A", "description": "a" }},
                    {{ "id": "b", "display_name": "B", "description": "b" }}
                ],
                "io": {{ "input": "bytes" }}{baseline}
            }}"#
        ))
        .expect("definition")
    }

    const BASE_9500: &str = r#", "baseline_comparison": { "assumed_p_b_bp": 9500, "assumed_p_c_bp": 0, "power_bp": 8000 }"#;

    fn prepared(label: &str, required: u64) -> PreparedBaseline {
        PreparedBaseline {
            majority_label: label.to_string(),
            required: RequiredSampleSize::new(required).expect("required"),
        }
    }

    /// REQ-27・#339: majority は train 分割のラベルだけから作る。train の多数派が `b`、
    /// validation・test の多数派が `a` の組で `b` になり、validation・test のラベルを入れ替えても変わらない。
    #[test]
    fn req27_issue339_majority_uses_train_split_only() {
        let def = definition(BASE_9500);
        for eval_label in ["a", "b"] {
            let (records, split) = records_and_split_labeled(&[
                ("t1", Split::Train, "b"),
                ("t2", Split::Train, "b"),
                ("t3", Split::Train, "a"),
                ("v1", Split::Validation, eval_label),
                ("v2", Split::Validation, eval_label),
                ("v3", Split::Validation, eval_label),
                ("e1", Split::Test, eval_label),
                ("e2", Split::Test, eval_label),
                ("e3", Split::Test, eval_label),
            ]);
            let p = prepare_baseline(&def, &records, &split)
                .expect("prepare")
                .expect("some");
            assert_eq!(p.majority_label, "b", "eval_label={eval_label}");
        }
    }

    /// REQ-25・#339: 同数は選択肢の宣言順で先のラベルになる。
    #[test]
    fn req25_issue339_majority_tie_uses_declaration_order() {
        let def = definition(BASE_9500);
        let (records, split) =
            records_and_split_labeled(&[("t1", Split::Train, "b"), ("t2", Split::Train, "a")]);
        let p = prepare_baseline(&def, &records, &split)
            .expect("prepare")
            .expect("some");
        assert_eq!(p.majority_label, "a");
    }

    /// REQ-25・#339: 欄が無い定義では準備しない。
    #[test]
    fn req25_issue339_no_baseline_comparison_means_none() {
        let def = definition("");
        let (records, split) = records_and_split_labeled(&[("t1", Split::Train, "a")]);
        assert!(
            prepare_baseline(&def, &records, &split)
                .expect("ok")
                .is_none()
        );
    }

    /// REQ-25・#339: 事前登録値の必要件数（P0 の `9500/0/8000` は 7、`1500/500/8000` は α=0.05 で 168。PoC-10 の 221 は α=0.0125 の値）。
    #[test]
    fn req25_issue339_required_from_assumption_known_values() {
        for (json, expected) in [
            (BASE_9500, 7),
            (
                r#", "baseline_comparison": { "assumed_p_b_bp": 1500, "assumed_p_c_bp": 500, "power_bp": 8000 }"#,
                168,
            ),
        ] {
            let def = definition(json);
            let req = required_from_assumption(def.baseline_comparison().expect("some"))
                .expect("required");
            assert_eq!(req.get(), expected, "{json}");
        }
    }

    /// REQ-25・#339: 和が 10000 の境界の組はすべて評価器の仮定として構築できる。
    #[test]
    fn req25_issue339_bp_sum_10000_boundary_is_accepted() {
        for k in 5001_u32..=10_000 {
            let p_b = f64::from(k) / BP_DENOMINATOR;
            let p_c = f64::from(10_000 - k) / BP_DENOMINATOR;
            assert!(
                McNemarSampleSizeAssumption::new(p_b, p_c, SIGNIFICANCE_ALPHA, 0.8).is_ok(),
                "k={k}"
            );
        }
    }

    /// REQ-25・#339: train の行が無い・必要件数が算出できない場合は固定 message の `invalid_input`。
    #[test]
    fn req25_issue339_prepare_fails_on_empty_train_and_unattainable_sample_size() {
        let (records, split) = records_and_split_labeled(&[("v1", Split::Validation, "a")]);
        let err = prepare_baseline(&definition(BASE_9500), &records, &split).unwrap_err();
        assert_eq!(err.code, ExitCode::InvalidInput);
        assert_eq!(err.message, "baseline comparison cannot be prepared");

        let (records, split) = records_and_split_labeled(&[("t1", Split::Train, "a")]);
        let def = definition(
            r#", "baseline_comparison": { "assumed_p_b_bp": 2, "assumed_p_c_bp": 1, "power_bp": 9999 }"#,
        );
        let err = prepare_baseline(&def, &records, &split).unwrap_err();
        assert_eq!(err.code, ExitCode::InvalidInput);
        assert_eq!(
            err.message,
            "baseline comparison sample size cannot be computed"
        );
    }

    fn many(label: &str, n: usize) -> Vec<String> {
        vec![label.to_string(); n]
    }

    fn outcomes(labels: &[String]) -> Vec<Outcome> {
        labels.iter().map(|l| Outcome::Label(l.clone())).collect()
    }

    /// 先頭 `bs` 件が `b`・残りが `a` の gold 列（12 件）。
    fn golds(bs: usize) -> Vec<String> {
        [many("b", bs), many("a", 12 - bs)].concat()
    }

    /// REQ-25・#339: 3 つの判定を具体値で記録に写す（n=12・majority=`a`）。
    #[test]
    fn req25_issue339_compare_records_three_verdicts() {
        let labels = ["a", "b"];
        // 候補が全件 gold どおり: 6 件の `b` で候補だけ正解 → b=6・c=0・p=0.03125 → 有意に上回る。
        let g = golds(6);
        let (rec, cand) = compare(&prepared("a", 7), &labels, &g, &outcomes(&g)).expect("compare");
        assert_eq!((rec.b, rec.c, rec.baseline_correct, cand), (6, 0, 6, 12));
        assert_eq!(rec.verdict, BaselineComparisonVerdict::SignificantlyBetter);
        assert_eq!(rec.majority_label, "a");
        assert_eq!(rec.required_n, 7);
        // 5 件では p=0.0625 で有意差なし。
        let g = golds(5);
        let (rec, _) = compare(&prepared("a", 7), &labels, &g, &outcomes(&g)).expect("compare");
        assert_eq!((rec.b, rec.c), (5, 0));
        assert_eq!(
            rec.verdict,
            BaselineComparisonVerdict::NotSignificantlyBetter
        );
        // 候補が全件 `b` と答える: gold の `a` で下限基準だけ正解（c=6）→ 有意差なし。
        let g = golds(6);
        let all_b = outcomes(&many("b", 12));
        let (rec, cand) = compare(&prepared("a", 7), &labels, &g, &all_b).expect("compare");
        assert_eq!((rec.b, rec.c, cand), (6, 6, 6));
        assert_eq!(
            rec.verdict,
            BaselineComparisonVerdict::NotSignificantlyBetter
        );
        // 必要件数 168 > 12 件は、(6, 0) でも判定不能（合格扱いにしない）。
        let g = golds(6);
        let (rec, _) = compare(&prepared("a", 168), &labels, &g, &outcomes(&g)).expect("compare");
        assert_eq!((rec.b, rec.c, rec.required_n), (6, 0, 168));
        assert_eq!(rec.verdict, BaselineComparisonVerdict::Undeterminable);
    }

    fn valid_record() -> BaselineComparisonRecord {
        // total=12・correct=10: both_correct=4・b=6・c=0・baseline_correct=4 → (6,0) は有意（required 7）。
        BaselineComparisonRecord {
            majority_label: "a".to_string(),
            baseline_correct: 4,
            b: 6,
            c: 0,
            required_n: 7,
            verdict: BaselineComparisonVerdict::SignificantlyBetter,
        }
    }

    /// REQ-27・#339: majority の正解数は凍結データの正解ラベルから数え直す。
    #[test]
    fn req27_issue339_baseline_correct_on_counts_majority_hits() {
        let p = prepared("a", 7);
        assert_eq!(p.baseline_correct_on(["a", "b", "a", "a"].into_iter()), 3);
        assert_eq!(p.baseline_correct_on(std::iter::empty()), 0);
    }

    /// REQ-27・#339: `baseline_correct` を凍結データの数え直しと食い違わせる改変は検出する。
    /// b・c の同量ずらし（both_correct を減らす改変）は検出範囲外（関数の doc に明記）。
    #[test]
    fn req27_issue339_record_matches_documents_undetectable_shift() {
        let p = prepared("a", 7);
        assert!(!record_matches(&p, &valid_record(), (10, 12), 5));
        // 限界の固定: total=100・correct=80・baseline_correct=60 の記録で、b=30,c=10 も
        // b=31,c=11 も算術的整合と判定の再計算を通る（検出できない。#168 の範囲）。
        let mk = |b: u64, c: u64| BaselineComparisonRecord {
            majority_label: "a".to_string(),
            baseline_correct: 60,
            b,
            c,
            required_n: 7,
            verdict: BaselineComparisonVerdict::SignificantlyBetter,
        };
        assert!(record_matches(&p, &mk(30, 10), (80, 100), 60));
        assert!(record_matches(&p, &mk(31, 11), (80, 100), 60));
    }

    /// REQ-27・#339: 正しい記録は通り、各欄の改変・範囲外の値は false になる。
    #[test]
    fn req27_issue339_record_matches_detects_tampering() {
        let p = prepared("a", 7);
        assert!(record_matches(&p, &valid_record(), (10, 12), 4));
        type Mutation = fn(&mut BaselineComparisonRecord);
        let mutations: [Mutation; 9] = [
            |r| r.majority_label = "b".to_string(),
            |r| r.b = 5,
            |r| r.c = 1,
            // b・c を同じ量だけずらすと件数は整合するが、判定の再計算で食い違う。
            |r| {
                r.b = 7;
                r.c = 1;
            },
            |r| r.verdict = BaselineComparisonVerdict::NotSignificantlyBetter,
            |r| r.required_n = 6,
            |r| r.baseline_correct = 5,
            |r| r.b = u64::MAX,
            |r| r.baseline_correct = 13,
        ];
        for (i, mutate) in mutations.iter().enumerate() {
            let mut r = valid_record();
            mutate(&mut r);
            assert!(!record_matches(&p, &r, (10, 12), 4), "mutation {i}");
        }
    }
}
