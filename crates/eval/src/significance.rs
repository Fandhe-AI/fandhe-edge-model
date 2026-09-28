//! 行ごとの正誤 → McNemar 検定 → 下限基準（majority）に対する有意性判定。
//!
//! [`crate::mcnemar`]（統計計算コア。TASK-25.1-1・issue #64）と
//! [`crate::baseline`]（下限基準の予測生成）を接続し、CLI の `evaluate`
//! 工程（REQ-33）が「モデルは下限基準（majority）を有意に上回るか」を
//! 判定できるようにする（REQ-25 正常系・TASK-25.1-2・issue #65）。
//!
//! # 対象外（本 issue の範囲外）
//!
//! - Holm 補正（複数候補比較。REQ-25）は [`judge`] に `p` を引数で渡す形に
//!   なっているため、[`crate::holm`]（TASK-25.3・issue #67）が補正後の
//!   p 値をそのまま渡して呼び直せる。[`BaselineComparison::required`] は
//!   [`crate::holm::compare_candidates_with_holm`] が判定を出し直す際に
//!   必要件数を取り違えないよう保持している
//! - 件数不足による「判定不能」（[`BaselineVerdict::Undeterminable`]）は
//!   実装済み（TASK-25.1-2・issue #65・PR #219）。必要件数
//!   （[`RequiredSampleSize`]）を事前登録の手続きから算出する関数
//!   （Connor 式の正規近似を起点に、実際に使う両側正確検定の検出力で
//!   引き上げる）は [`crate::sample_size::required_sample_size_mcnemar`]
//!   として実装済み（REQ-25・TASK-25.2・issue #66・PR #230 レビュー
//!   指摘・P0）。呼び出し側はその戻り値を [`judge`]・
//!   [`compare_with_baseline`] の引数として渡す（spec に固定の必要件数は
//!   無く、事前登録時に算出する手続きだけが定められている。
//!   `04-requirements.md` L503-517。PoC-10 の事前登録値は正規近似の
//!   `ceil(n)=221` 件だが、正確検定の検出力探索で求め直すと 229 件になる。
//!   `crates/eval/tests/baseline_significance.rs`
//!   `poc10_required_sample_size_is_229` 参照）
//! - CLI の JSON 出力・終了コードへの写像は行わない（TASK-33.x / TASK-18.3）

use crate::baseline::{self, BaselineError};
use crate::mcnemar::{self, McNemarExact, PValue, PairedCounts};
use crate::metrics::{EvalRecord, Outcome};

/// [`correctness`]・[`compare_with_baseline`] が受け付ける評価レコード数の上限
/// （REQ-39 ガード層「資源の上限」）。
///
/// レコード件数の主たる上限検証はデータ契約層（呼び出し側）の責務だが
/// （crate ドキュメント「層の境界・不変条件」参照）、本モジュールの 2 関数は
/// 外部入力由来の `records.len()` をそのまま候補用・下限基準用の正誤 `Vec`
/// の容量へ使うため、確保前に拒否する防御層をここにも置く（件数を上限検証
/// してからアロケーションに使う。`.claude/rules/security.md`「資源の上限」。
/// Review 指摘。TASK-25.1-2・issue #65）。値は [`metrics::MAX_LABELS`]・
/// [`mcnemar::MAX_DISCORDANT_PAIRS`] と同様、データ契約層の確定値が無い
/// 段階の暫定値（実測に基づく調整は後続 TASK）。
///
/// [`metrics::MAX_LABELS`]: crate::metrics::MAX_LABELS
pub const MAX_EVAL_RECORDS: usize = 1_000_000;

/// 下限基準（majority）に対する有意性判定の α（評価契約で固定。REQ-25）。
///
/// `.claude/rules/evaluation-contract.md`「有意性・指標」で定めた値であり、
/// 呼び出し側が変更できるよう引数にはしない（変更は評価契約の変更にあたり、
/// main の設計判断とユーザー承認を要する）。
pub const SIGNIFICANCE_ALPHA: f64 = 0.05;

/// 下限基準比較に必要な最小評価件数（N）。0 を拒否し、壊れた値
/// （未設定・0 件）を表現できないようにする（`.claude/rules/coding-rust.md`
/// 「公開 API・型設計」）。
///
/// 値は事前登録の手続き（PoC-10 相当）で算出したものを呼び出し側が用意する。
/// spec（`04-requirements.md` L503-517。REQ-25）は固定の必要件数を定めておらず、
/// 事前登録時に算出する手続きだけを定めている。算出関数（Connor 式）は
/// [`crate::sample_size::required_sample_size_mcnemar`] として実装済み
/// （TASK-25.2・issue #66。PoC-10 の事前登録値は 221 件）。本モジュール自体は
/// 算出済みの値を受け取るだけで、算出ロジックには依存しない（循環依存を
/// 避けるため `sample_size` → `significance` の一方向のみ）。
///
/// 既定値は持たない。呼び出し側が必ず明示的に値を渡す
/// （2026-09-28 オーナー承認の設計）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RequiredSampleSize(u64);

impl RequiredSampleSize {
    /// `required` が 1 以上なら `Some` を返す。0 は「必要件数が未設定」と
    /// 区別できず壊れた値になるため拒否する。
    pub fn new(required: u64) -> Option<Self> {
        if required == 0 {
            None
        } else {
            Some(Self(required))
        }
    }

    /// 中身の `u64` 値を取り出す。
    pub fn get(&self) -> u64 {
        self.0
    }
}

/// 件数不足で判定不能になった理由。
///
/// [`BaselineVerdict::Undeterminable`] の内側に持たせる。テストが期待値を
/// `assert_eq!` で構築できるよう `#[non_exhaustive]` は付けない
/// （`required`・`actual` の両方が固定フィールドで、将来の拡張は本 struct
/// 自体を非網羅にするより variant 追加で扱う想定）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsufficientSamples {
    /// 事前登録で算出した必要件数。
    pub required: RequiredSampleSize,
    /// 実際の評価件数（N。不一致ペア数 `b + c` ではなく比較対象レコードの総数）。
    pub actual: u64,
}

/// 下限基準（majority）に対する判定。
///
/// `#[non_exhaustive]` にしてあるのは、将来 variant が増える余地を残す
/// ためだが、Holm 補正（TASK-25.3・issue #67）は本 enum に variant を
/// 追加していない。補正後の判定も同じ 3 状態（`SignificantlyBetter` /
/// `NotSignificantlyBetter` / `Undeterminable`）で表せるため、
/// [`crate::holm::HolmComparison::verdict`] は本 enum をそのまま返す。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BaselineVerdict {
    /// 候補が下限基準を有意に上回る（`b > c` かつ `p < `[`SIGNIFICANCE_ALPHA`]）。
    SignificantlyBetter,
    /// 有意に上回るとは言えない（`b <= c`、または `p >= `[`SIGNIFICANCE_ALPHA`]）。
    NotSignificantlyBetter,
    /// 評価件数が必要件数（[`RequiredSampleSize`]）未満で判定できない
    /// （REQ-25・TASK-25.2）。p 値・`b`・`c` の値によらず、合格扱い
    /// （[`BaselineVerdict::SignificantlyBetter`]）にはならない。
    Undeterminable(InsufficientSamples),
}

/// `α = `[`SIGNIFICANCE_ALPHA`]`` での有意性判定（純粋関数）。
///
/// 規則（優先順）:
///
/// 1. `n_evaluated < required.get()`（評価件数が必要件数未満）なら
///    [`BaselineVerdict::Undeterminable`]（`b`・`c`・`p` の値、`b > c` の
///    向きによらず判定不能。REQ-25・TASK-25.2。件数不足の下限基準比較は、
///    `(b, c) = (6, 0)`（p = 0.03125 < 0.05）のように p 値だけを見れば
///    有意に見える場合でも合格扱い〔`SignificantlyBetter`〕にしない）
/// 2. それ以外で `b > c` かつ `p < α` なら
///    [`BaselineVerdict::SignificantlyBetter`]
/// 3. それ以外は [`BaselineVerdict::NotSignificantlyBetter`]
///
/// （PoC-10 `stats_mcnemar.py` の「有意に上回る」の定義に、件数不足の
/// 判定不能〔TASK-25.2〕を加えたもの。複数候補の Holm 補正は
/// [`crate::holm::compare_candidates_with_holm`] が本関数を補正後の p 値で
/// 呼び直す形で行う〔TASK-25.3・issue #67〕）
///
/// `n_evaluated` は比較に使う評価レコードの総件数 N（[`PairedCounts::n`]）
/// であり、不一致ペア数 `b + c` ではない（PoC-10 事前登録の基準。
/// `required` は [`RequiredSampleSize`] のドキュメント参照）。
///
/// 比較は厳密な `<` とし、許容差は付けない。評価契約（REQ-24）が
/// 「p < 0.05」を境界として定めており、判定を緩めない設計判断として
/// 許容差を持ち込まないのが正しい（`.claude/rules/evaluation-contract.md`
/// 「有意性・指標」）。
///
/// 数学的には正確検定の p 値は `k / 2^n` の形の二進有理数で、`0.05`
/// （= 1/20。分母に素因数 5 を含む）とちょうど一致することは原理上ない。
/// ただし [`mcnemar::two_sided_p_value`] の実装は `ln`/`exp`（f64 の
/// libm）による近似計算であり、返る値はその厳密な有理数そのものではなく
/// 浮動小数近似値である（Review 指摘。TASK-25.1-2・issue #65。以前の
/// 記述は実装と不整合だった）。したがって上記の数学的事実だけを根拠に
/// 「境界値問題が実質的に生じない」とは言えない。実際に `0.05` へ
/// 極めて近づく `(b, c)` では近似誤差により判定が理論値と入れ替わる
/// 余地があるが、`<` を許容差なしで使うという設計判断自体は変えない
/// （評価契約の変更にあたり、緩和には main の設計判断とユーザー承認を要する）。
pub fn judge(
    b: u64,
    c: u64,
    p: PValue,
    n_evaluated: u64,
    required: RequiredSampleSize,
) -> BaselineVerdict {
    if n_evaluated < required.get() {
        BaselineVerdict::Undeterminable(InsufficientSamples {
            required,
            actual: n_evaluated,
        })
    } else if b > c && p.value() < SIGNIFICANCE_ALPHA {
        BaselineVerdict::SignificantlyBetter
    } else {
        BaselineVerdict::NotSignificantlyBetter
    }
}

/// 1 件の [`Outcome`] が `gold` と一致するかを判定する。
///
/// `Outcome::Label(l)` で `l == gold` の場合のみ正解。未知ラベル・
/// `Invalid`・`Abstain`・`Error` はすべて不正解として扱う（PoC-10:
/// 予測の欠落・`status` が ok 以外・ラベル違いはすべて不正解）。
fn is_correct(gold: &str, outcome: &Outcome) -> bool {
    matches!(outcome, Outcome::Label(l) if l == gold)
}

/// 評価レコード列から、行ごとの正誤（`Outcome` が `gold` と一致するか）を求める。
///
/// - `labels` 自体の検証（空・空 ID・重複・`MAX_LABELS` 超過）は
///   [`baseline::validate_label_order`] に委ね、[`fit_majority`]（下限基準の
///   予測生成）と同じ規則に揃える（Review 指摘。TASK-25.1-2・issue #65。
///   以前は `labels.contains(&record.gold)` の素通し判定のみで、`labels`
///   が空の場合に本来の [`crate::metrics::EvalError::EmptyLabels`] ではなく
///   `UnknownGoldLabel { index: 0 }` を返し、重複ラベルも検出できなかった）
/// - gold がラベル集合に存在しない場合は [`BaselineError::UnknownGoldLabel`]
///   を返す（`evaluate_single_select` と同じ fail-closed。PoC-10 の
///   `stats_mcnemar.py` のように黙って除外しない。正解側の欠陥はデータ契約層
///   の検査で扱う）
/// - `records` が空の場合は [`BaselineError::EmptyRecords`]（評価済みを
///   装わない。0 件は必要件数の多寡によらず判定できないため、[`judge`]の
///   `Undeterminable` ではなくエラーとして扱う）
/// - `records.len()` が [`MAX_EVAL_RECORDS`] を超える場合は確保前に
///   [`BaselineError::TooManyRecords`] を返す（REQ-39。Review 指摘。
///   TASK-25.1-2・issue #65）
///
/// 入力は `&` 参照でのみ受け取り、書き換えない（REQ-27）。
///
/// [`fit_majority`]: crate::baseline::fit_majority
pub fn correctness(
    labels: &[&str],
    records: &[EvalRecord<'_>],
) -> Result<Vec<bool>, BaselineError> {
    let index = baseline::validate_label_order(labels)?;

    if records.is_empty() {
        return Err(BaselineError::EmptyRecords);
    }
    if records.len() > MAX_EVAL_RECORDS {
        return Err(BaselineError::TooManyRecords {
            n_records: records.len(),
            limit: MAX_EVAL_RECORDS,
        });
    }

    let mut result = Vec::with_capacity(records.len());
    for (i, record) in records.iter().enumerate() {
        if !index.contains_key(record.gold) {
            return Err(BaselineError::UnknownGoldLabel { index: i });
        }
        result.push(is_correct(record.gold, record.outcome));
    }
    Ok(result)
}

/// 候補・下限基準を対応させた 1 行分の入力。
///
/// index のずれをスライスの並行走査ではなく型で防ぐため、候補・下限基準の
/// 予測を 1 つの構造体にまとめて渡す。
#[derive(Debug, Clone, Copy)]
pub struct PairedRecord<'a> {
    /// 正解ラベル ID。
    pub gold: &'a str,
    /// 候補モデルの推論結果。
    pub candidate: &'a Outcome,
    /// 下限基準（majority 等）の推論結果。
    pub baseline: &'a Outcome,
}

/// 候補・下限基準の比較結果（McNemar 検定＋有意性判定）。
///
/// フィールドは非公開にし、アクセサ経由で公開する（[`mcnemar::PValue`] を
/// 含む [`McNemarExact`] を保持するため `PartialEq` は derive しない）。
/// PoC `compare()` のキー（n・cand_correct・base_correct・b_cand_only・
/// c_base_only・p_exact_two_sided）と 1 対 1 に対応させ、CLI（TASK-33）が
/// JSON へそのまま写せるようにしている。
#[derive(Debug, Clone, Copy)]
pub struct BaselineComparison {
    counts: PairedCounts,
    test: McNemarExact,
    verdict: BaselineVerdict,
    required: RequiredSampleSize,
}

impl BaselineComparison {
    /// 対応のある正誤の集計（n・both_correct・b・c・both_wrong）。
    pub fn counts(&self) -> PairedCounts {
        self.counts
    }

    /// 候補モデルの正解件数（`both_correct + b_candidate_only`）。
    pub fn candidate_correct(&self) -> u64 {
        // `paired_counts` が checked 演算で導出した非負値どうしの和で、
        // 生成元の `PairedCounts::n` がオーバーフローしないことを検証済み
        // （`mcnemar::paired_counts`）ため、ここでの再計算も溢れない。
        self.counts.both_correct + self.counts.b_candidate_only
    }

    /// 下限基準の正解件数（`both_correct + c_baseline_only`）。
    pub fn baseline_correct(&self) -> u64 {
        self.counts.both_correct + self.counts.c_baseline_only
    }

    /// McNemar の正確検定（両側）の結果。`p_two_sided()` で p 値を取り出す。
    pub fn test(&self) -> McNemarExact {
        self.test
    }

    /// α = 0.05 での有意性判定。
    pub fn verdict(&self) -> BaselineVerdict {
        self.verdict
    }

    /// この比較で使った必要件数（事前登録の値）。
    ///
    /// [`crate::holm::compare_candidates_with_holm`] が Holm 補正後の p 値で
    /// [`judge`] を呼び直す際、比較時と同じ必要件数を使うために保持する
    /// （TASK-25.3・issue #67。引数で再度受け取る設計にすると、比較時と
    /// 補正時で必要件数が食い違う経路が生まれるため、型で防ぐ）。
    pub fn required(&self) -> RequiredSampleSize {
        self.required
    }
}

/// ラベル集合・候補・下限基準の対応データから [`BaselineComparison`] を求める。
///
/// 手順: ラベル検証 → 行ごとの正誤（候補・下限基準それぞれ [`is_correct`]
/// と同じ規則）→ [`mcnemar::paired_counts`] → [`mcnemar::mcnemar_exact_two_sided`]
/// → [`judge`]（評価件数 N として `counts.n` を渡す）。
///
/// `required` は事前登録で算出した必要件数（[`RequiredSampleSize`] の
/// ドキュメント参照。REQ-25・TASK-25.2）。既定値は無く、呼び出し側が必ず
/// 明示的に渡す。McNemar の統計量（`b`・`c`・p 値）は件数不足の場合も
/// 計算して返す（CLI の JSON 出力が判定不能の場合でも統計量を提示できる
/// ようにするため。判定〔[`BaselineComparison::verdict`]〕だけが
/// `Undeterminable` になる）。
///
/// `labels` 自体の検証（空・空 ID・重複・`MAX_LABELS` 超過）は
/// [`baseline::validate_label_order`] に委ね、[`correctness`]・
/// [`fit_majority`] と同じ規則に揃える（Review 指摘。TASK-25.1-2・issue #65）。
///
/// gold がラベル集合に無い場合は [`BaselineError::UnknownGoldLabel`]、
/// `records` が空の場合は [`BaselineError::EmptyRecords`]、
/// `records.len()` が [`MAX_EVAL_RECORDS`] を超える場合は確保前に
/// [`BaselineError::TooManyRecords`] を返す（[`correctness`] と同じ規則。
/// REQ-39。Review 指摘。TASK-25.1-2・issue #65）。
///
/// [`fit_majority`]: crate::baseline::fit_majority
pub fn compare_with_baseline(
    labels: &[&str],
    records: &[PairedRecord<'_>],
    required: RequiredSampleSize,
) -> Result<BaselineComparison, BaselineError> {
    let index = baseline::validate_label_order(labels)?;

    if records.is_empty() {
        return Err(BaselineError::EmptyRecords);
    }
    if records.len() > MAX_EVAL_RECORDS {
        return Err(BaselineError::TooManyRecords {
            n_records: records.len(),
            limit: MAX_EVAL_RECORDS,
        });
    }

    let mut candidate_correct = Vec::with_capacity(records.len());
    let mut baseline_correct = Vec::with_capacity(records.len());
    for (i, record) in records.iter().enumerate() {
        if !index.contains_key(record.gold) {
            return Err(BaselineError::UnknownGoldLabel { index: i });
        }
        candidate_correct.push(is_correct(record.gold, record.candidate));
        baseline_correct.push(is_correct(record.gold, record.baseline));
    }

    let counts = mcnemar::paired_counts(&candidate_correct, &baseline_correct)?;
    let test = mcnemar::mcnemar_exact_two_sided(counts.b_candidate_only, counts.c_baseline_only)?;
    let verdict = judge(
        counts.b_candidate_only,
        counts.c_baseline_only,
        test.p_two_sided(),
        counts.n,
        required,
    );

    Ok(BaselineComparison {
        counts,
        test,
        verdict,
        required,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcnemar::mcnemar_exact_two_sided;

    /// テスト用の [`RequiredSampleSize`] 生成ヘルパー（不正値は使わない前提）。
    fn req(n: u64) -> RequiredSampleSize {
        RequiredSampleSize::new(n).expect("test helper requires a non-zero value")
    }

    /// (13,4) は α のすぐ下（p = 0.049041748046875 < 0.05）→ Better。
    /// 評価件数 N=221 は必要件数（PoC-10 の事前登録値）ちょうどで、
    /// 件数不足による判定不能は起きない。
    #[test]
    fn judge_significantly_better_just_under_alpha() {
        let result = mcnemar_exact_two_sided(13, 4).unwrap();
        assert!(result.p_two_sided().value() < SIGNIFICANCE_ALPHA);
        assert_eq!(
            judge(13, 4, result.p_two_sided(), 221, req(221)),
            BaselineVerdict::SignificantlyBetter
        );
    }

    /// (22,10) は α のすぐ上（p ≈ 0.0501 >= 0.05）→ Not。
    #[test]
    fn judge_not_significantly_better_just_over_alpha() {
        let result = mcnemar_exact_two_sided(22, 10).unwrap();
        assert!(result.p_two_sided().value() >= SIGNIFICANCE_ALPHA);
        assert_eq!(
            judge(22, 10, result.p_two_sided(), 221, req(221)),
            BaselineVerdict::NotSignificantlyBetter
        );
    }

    /// (6,0) は p = 0.03125 → Better（評価件数が必要件数以上の場合）。
    #[test]
    fn judge_significantly_better_small_case() {
        let result = mcnemar_exact_two_sided(6, 0).unwrap();
        assert_eq!(
            judge(6, 0, result.p_two_sided(), 221, req(221)),
            BaselineVerdict::SignificantlyBetter
        );
    }

    /// (5,0) は p = 0.0625 → Not。
    #[test]
    fn judge_not_significantly_better_small_case() {
        let result = mcnemar_exact_two_sided(5, 0).unwrap();
        assert_eq!(
            judge(5, 0, result.p_two_sided(), 221, req(221)),
            BaselineVerdict::NotSignificantlyBetter
        );
    }

    /// (9,3) は p = 0.14599609375 → Not。
    #[test]
    fn judge_not_significantly_better_moderate_p() {
        let result = mcnemar_exact_two_sided(9, 3).unwrap();
        assert_eq!(
            judge(9, 3, result.p_two_sided(), 221, req(221)),
            BaselineVerdict::NotSignificantlyBetter
        );
    }

    /// (3,12) は b < c（下限基準のほうが強い）→ p が小さくても Not（方向の確認）。
    #[test]
    fn judge_not_significantly_better_when_baseline_wins() {
        let result = mcnemar_exact_two_sided(3, 12).unwrap();
        // p = 0.03515625 < 0.05 だが b < c なので Not。
        assert!(result.p_two_sided().value() < SIGNIFICANCE_ALPHA);
        assert_eq!(
            judge(3, 12, result.p_two_sided(), 221, req(221)),
            BaselineVerdict::NotSignificantlyBetter
        );
    }

    /// (0,0) は p = 1.0 → Not。
    #[test]
    fn judge_not_significantly_better_zero_discordant() {
        let result = mcnemar_exact_two_sided(0, 0).unwrap();
        assert_eq!(
            judge(0, 0, result.p_two_sided(), 221, req(221)),
            BaselineVerdict::NotSignificantlyBetter
        );
    }

    /// REQ-25・TASK-25.2: 評価件数 N が必要件数未満なら、`(b,c)=(6,0)`
    /// （p = 0.03125 < 0.05。単独では `SignificantlyBetter` になる値）でも
    /// 判定不能（`Undeterminable`）になり、合格扱いにならないことを確認する。
    #[test]
    fn judge_undeterminable_when_below_required_sample_size() {
        let result = mcnemar_exact_two_sided(6, 0).unwrap();
        assert!(result.p_two_sided().value() < SIGNIFICANCE_ALPHA);
        assert_eq!(
            judge(6, 0, result.p_two_sided(), 220, req(221)),
            BaselineVerdict::Undeterminable(InsufficientSamples {
                required: req(221),
                actual: 220,
            })
        );
    }

    /// REQ-25・TASK-25.2: 評価件数 N がちょうど必要件数なら判定不能にならない
    /// （境界値）。
    #[test]
    fn judge_significantly_better_at_required_sample_size_boundary() {
        let result = mcnemar_exact_two_sided(6, 0).unwrap();
        assert_eq!(
            judge(6, 0, result.p_two_sided(), 221, req(221)),
            BaselineVerdict::SignificantlyBetter
        );
    }

    /// REQ-25・TASK-25.2: 評価件数 N が必要件数を上回っても判定不能にならない。
    #[test]
    fn judge_significantly_better_above_required_sample_size() {
        let result = mcnemar_exact_two_sided(6, 0).unwrap();
        assert_eq!(
            judge(6, 0, result.p_two_sided(), 222, req(221)),
            BaselineVerdict::SignificantlyBetter
        );
    }

    /// REQ-25・TASK-25.2: 件数不足は `b > c` の向きに関係なく判定不能になる
    /// （下限基準が優勢な `(b,c)=(3,12)` でも `NotSignificantlyBetter` では
    /// なく `Undeterminable` になることの確認）。
    #[test]
    fn judge_undeterminable_regardless_of_direction() {
        let result = mcnemar_exact_two_sided(3, 12).unwrap();
        assert_eq!(
            judge(3, 12, result.p_two_sided(), 220, req(221)),
            BaselineVerdict::Undeterminable(InsufficientSamples {
                required: req(221),
                actual: 220,
            })
        );
    }

    /// `RequiredSampleSize::new(0)` は不正値として拒否される。
    #[test]
    fn required_sample_size_rejects_zero() {
        assert_eq!(RequiredSampleSize::new(0), None);
    }

    /// `RequiredSampleSize::new(1)` は最小の正当値として受け付けられる。
    #[test]
    fn required_sample_size_accepts_one() {
        let size = RequiredSampleSize::new(1).unwrap();
        assert_eq!(size.get(), 1);
    }

    /// `correctness`: `Label` 一致は正解。
    #[test]
    fn correctness_label_match_is_correct() {
        let labels = ["A", "B"];
        let outcome = Outcome::Label("A".to_string());
        let records = [EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        assert_eq!(correctness(&labels, &records).unwrap(), vec![true]);
    }

    /// `correctness`: `Label` 不一致は不正解。
    #[test]
    fn correctness_label_mismatch_is_incorrect() {
        let labels = ["A", "B"];
        let outcome = Outcome::Label("B".to_string());
        let records = [EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        assert_eq!(correctness(&labels, &records).unwrap(), vec![false]);
    }

    /// `correctness`: ラベル集合外の予測は不正解。
    #[test]
    fn correctness_out_of_set_label_is_incorrect() {
        let labels = ["A", "B"];
        let outcome = Outcome::Label("Z".to_string());
        let records = [EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        assert_eq!(correctness(&labels, &records).unwrap(), vec![false]);
    }

    /// `correctness`: `Invalid`・`Abstain`・`Error` はすべて不正解。
    #[test]
    fn correctness_non_label_outcomes_are_incorrect() {
        let labels = ["A"];
        let invalid = Outcome::Invalid;
        let abstain = Outcome::Abstain;
        let error = Outcome::Error;
        let records = [
            EvalRecord {
                gold: "A",
                outcome: &invalid,
            },
            EvalRecord {
                gold: "A",
                outcome: &abstain,
            },
            EvalRecord {
                gold: "A",
                outcome: &error,
            },
        ];
        assert_eq!(
            correctness(&labels, &records).unwrap(),
            vec![false, false, false]
        );
    }

    /// `correctness`: 未知の gold はエラー。
    #[test]
    fn correctness_unknown_gold_is_error() {
        let labels = ["A", "B"];
        let outcome = Outcome::Label("A".to_string());
        let records = [EvalRecord {
            gold: "Z",
            outcome: &outcome,
        }];
        let err = correctness(&labels, &records).unwrap_err();
        assert_eq!(err, BaselineError::UnknownGoldLabel { index: 0 });
    }

    /// `correctness`: 空の records はエラー。
    #[test]
    fn correctness_empty_records_is_error() {
        let labels = ["A"];
        let err = correctness(&labels, &[]).unwrap_err();
        assert_eq!(err, BaselineError::EmptyRecords);
    }

    /// `compare_with_baseline`: 空の records はエラー。
    #[test]
    fn compare_with_baseline_empty_records_is_error() {
        let labels = ["A", "B"];
        let err = compare_with_baseline(&labels, &[], req(1)).unwrap_err();
        assert_eq!(err, BaselineError::EmptyRecords);
    }

    /// `compare_with_baseline`: 未知の gold はエラー。
    #[test]
    fn compare_with_baseline_unknown_gold_is_error() {
        let labels = ["A", "B"];
        let cand = Outcome::Label("A".to_string());
        let base = Outcome::Label("A".to_string());
        let records = [PairedRecord {
            gold: "Z",
            candidate: &cand,
            baseline: &base,
        }];
        let err = compare_with_baseline(&labels, &records, req(1)).unwrap_err();
        assert_eq!(err, BaselineError::UnknownGoldLabel { index: 0 });
    }

    /// 評価契約（REQ-27）: 入力スライスが評価の前後で変わらないこと
    /// （clone との比較で、書き換えられていないことを確認する）。
    #[test]
    fn compare_with_baseline_does_not_mutate_input() {
        let labels = ["A", "B"];
        let cand_correct = Outcome::Label("A".to_string());
        let cand_wrong = Outcome::Label("B".to_string());
        let base_correct = Outcome::Label("A".to_string());
        let records = [
            PairedRecord {
                gold: "A",
                candidate: &cand_correct,
                baseline: &base_correct,
            },
            PairedRecord {
                gold: "A",
                candidate: &cand_wrong,
                baseline: &base_correct,
            },
        ];
        let before: Vec<(&str, Outcome, Outcome)> = records
            .iter()
            .map(|r| (r.gold, r.candidate.clone(), r.baseline.clone()))
            .collect();
        let _ = compare_with_baseline(&labels, &records, req(1)).unwrap();
        let after: Vec<(&str, Outcome, Outcome)> = records
            .iter()
            .map(|r| (r.gold, r.candidate.clone(), r.baseline.clone()))
            .collect();
        assert_eq!(before, after);
    }

    /// Review 指摘（issue #65）: `correctness` は `labels` が空の場合
    /// [`fit_majority`] と同じ [`crate::metrics::EvalError::EmptyLabels`]
    /// を返す（以前は `UnknownGoldLabel { index: 0 }` になっていた）。
    ///
    /// [`fit_majority`]: crate::baseline::fit_majority
    #[test]
    fn correctness_empty_labels_is_eval_error() {
        let labels: [&str; 0] = [];
        let outcome = Outcome::Label("A".to_string());
        let records = [EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        let err = correctness(&labels, &records).unwrap_err();
        assert_eq!(
            err,
            BaselineError::Eval(crate::metrics::EvalError::EmptyLabels)
        );
    }

    /// Review 指摘（issue #65）: `correctness` は重複ラベルを検出する
    /// （以前は `labels.contains` の素通し判定のみで重複を検出できなかった）。
    #[test]
    fn correctness_duplicate_labels_is_eval_error() {
        let labels = ["A", "A"];
        let outcome = Outcome::Label("A".to_string());
        let records = [EvalRecord {
            gold: "A",
            outcome: &outcome,
        }];
        let err = correctness(&labels, &records).unwrap_err();
        assert_eq!(
            err,
            BaselineError::Eval(crate::metrics::EvalError::DuplicateLabel {
                label: "A".to_string()
            })
        );
    }

    /// Review 指摘（issue #65）: `compare_with_baseline` も `labels` が
    /// 空の場合 `EmptyLabels` を返す（`correctness` と同じ規則）。
    #[test]
    fn compare_with_baseline_empty_labels_is_eval_error() {
        let labels: [&str; 0] = [];
        let cand = Outcome::Label("A".to_string());
        let base = Outcome::Label("A".to_string());
        let records = [PairedRecord {
            gold: "A",
            candidate: &cand,
            baseline: &base,
        }];
        let err = compare_with_baseline(&labels, &records, req(1)).unwrap_err();
        assert_eq!(
            err,
            BaselineError::Eval(crate::metrics::EvalError::EmptyLabels)
        );
    }

    /// codex/review 指摘（TASK-25.1-2・issue #65。PR #219）: `correctness` は
    /// `records.len()` が [`MAX_EVAL_RECORDS`] を超える場合、正誤 `Vec` を
    /// 確保する前に `TooManyRecords` で拒否する（REQ-39 資源の上限）。
    #[test]
    fn correctness_too_many_records_is_error() {
        let labels = ["A"];
        let outcome = Outcome::Label("A".to_string());
        let records: Vec<EvalRecord<'_>> = (0..=MAX_EVAL_RECORDS)
            .map(|_| EvalRecord {
                gold: "A",
                outcome: &outcome,
            })
            .collect();
        let err = correctness(&labels, &records).unwrap_err();
        assert_eq!(
            err,
            BaselineError::TooManyRecords {
                n_records: MAX_EVAL_RECORDS + 1,
                limit: MAX_EVAL_RECORDS,
            }
        );
    }

    /// codex/review 指摘（TASK-25.1-2・issue #65。PR #219）: `compare_with_baseline`
    /// も同じ規則で `records.len()` を確保前に検証する（候補用・下限基準用の
    /// 2 本の `Vec` の容量として使われるため）。
    #[test]
    fn compare_with_baseline_too_many_records_is_error() {
        let labels = ["A", "B"];
        let cand = Outcome::Label("A".to_string());
        let base = Outcome::Label("A".to_string());
        let records: Vec<PairedRecord<'_>> = (0..=MAX_EVAL_RECORDS)
            .map(|_| PairedRecord {
                gold: "A",
                candidate: &cand,
                baseline: &base,
            })
            .collect();
        let err = compare_with_baseline(&labels, &records, req(1)).unwrap_err();
        assert_eq!(
            err,
            BaselineError::TooManyRecords {
                n_records: MAX_EVAL_RECORDS + 1,
                limit: MAX_EVAL_RECORDS,
            }
        );
    }

    /// REQ-25・TASK-25.2: `compare_with_baseline` を通しても、評価件数 N が
    /// 必要件数未満なら判定不能になる（`(b,c)=(6,0)`。単体では p=0.03125 で
    /// `SignificantlyBetter` になる値だが、N=6 < required=7 で判定できない）。
    #[test]
    fn compare_with_baseline_undeterminable_when_below_required_sample_size() {
        let labels = ["A", "B"];
        let cand = Outcome::Label("B".to_string());
        let base = Outcome::Label("A".to_string());
        // 6 件とも候補のみ正解（gold="B"）、下限基準は "A" を予測し不正解。
        let records: Vec<PairedRecord<'_>> = (0..6)
            .map(|_| PairedRecord {
                gold: "B",
                candidate: &cand,
                baseline: &base,
            })
            .collect();

        let comparison = compare_with_baseline(&labels, &records, req(7)).unwrap();
        assert_eq!(comparison.counts().n, 6);
        assert_eq!(comparison.counts().b_candidate_only, 6);
        assert_eq!(comparison.counts().c_baseline_only, 0);
        assert_eq!(
            comparison.verdict(),
            BaselineVerdict::Undeterminable(InsufficientSamples {
                required: req(7),
                actual: 6,
            })
        );
    }

    /// 同じ 6 件データで必要件数がちょうど 6 なら判定不能にならず、
    /// 従来どおり `SignificantlyBetter` になる（境界値）。
    #[test]
    fn compare_with_baseline_significantly_better_at_required_sample_size_boundary() {
        let labels = ["A", "B"];
        let cand = Outcome::Label("B".to_string());
        let base = Outcome::Label("A".to_string());
        let records: Vec<PairedRecord<'_>> = (0..6)
            .map(|_| PairedRecord {
                gold: "B",
                candidate: &cand,
                baseline: &base,
            })
            .collect();

        let comparison = compare_with_baseline(&labels, &records, req(6)).unwrap();
        assert_eq!(comparison.verdict(), BaselineVerdict::SignificantlyBetter);
    }
}
